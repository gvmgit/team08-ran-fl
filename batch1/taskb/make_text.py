"""Task B text corpus generator (SageMaker Processing, CPU). Deterministic and seeded; no language model is involved,
so the generator code is the data lineage and the Task B grader stays independent of any model family.

For each site it turns REAL rows of that site's telemetry CSV into alarm / counter / ticket style prompts and a
structured answer:   FAMILY: <family> | CAUSE: <cause> | ACTION: <action>

  * Each example is anchored on a real event (event_congestion / event_mobility == 1) at cell c, minute T, or on a
    normal minute. The prompt shows the counters at T-100, T-60 and T-20 minutes taken from the CSV.
  * Alarm codes are raised by numeric thresholds on those same rows (see ALARM_RULES), so the text stays grounded.
  * The cause is a deterministic rule on the rows and the cell tilt; the action follows from the cause.
  * One shared format for both sites (no site tag, cell id or time of day): a site-specific token would let a model
    shortcut the family and turn the transfer test into a test of the shortcut.
  * The failure family a site "never logged" (config "withhold") gets zero training and validation examples at that
    site. Test sets always contain both families, rendered the same way.

Output (per site): train.jsonl, val.jsonl, test.jsonl and manifest.json with counts, seeds and file hashes.
"""
import hashlib
import json
import os
import time

import boto3
import numpy as np
import pandas as pd

CODE = "/opt/ml/processing/input/code"
OUT = "/opt/ml/processing/output"
WORK = "/tmp/work"
os.makedirs(OUT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)

N_CELLS, T, HORIZON = 30, 172800, 120
OFFSETS = (100, 60, 20)  # minutes before the event at which the counters are shown
DAY_RANGES = {"train": (0, 83), "val": (85, 94), "test": (96, 119)}
FAMILIES = ("congestion", "mobility")

ACTIONS = {
    "UE_SURGE": "ENABLE_LOAD_BALANCING",
    "COVERAGE_OVERLAP": "INCREASE_ELECTRICAL_TILT",
    "PRB_EXHAUSTION": "ADD_CARRIER_OR_OFFLOAD",
    "HO_PARAM_MISTUNED": "TUNE_A3_OFFSET_AND_TTT",
    "COVERAGE_HOLE": "REVIEW_COVERAGE_AND_NEIGHBOR_PLAN",
    "PINGPONG_NEIGHBOR": "FIX_NEIGHBOR_RELATIONS",
}
TICKETS = [
    "ticket auto-opened from alarm correlation",
    "ticket opened by the shift engineer",
    "ticket created after a customer complaint",
    "ticket raised by the optimisation team",
]


def alarms_at(ue, sinr, rsrp, ho, weak):
    out = []
    if ue >= 95: out.append("PRB_UTIL_HIGH")
    if ue >= 125: out.append("UE_CONTEXT_LIMIT_NEAR")
    if sinr <= 5.0: out.append("DL_THROUGHPUT_DEGRADED")
    if ho >= 11: out.append("HO_FAILURE_RATE_HIGH")
    if ho >= 9 and rsrp <= -103: out.append("NEIGHBOR_RELATION_UNSTABLE")
    if rsrp <= -105: out.append("RLF_RATE_RISING")
    if weak >= 0.18: out.append("WEAK_COVERAGE_UE_SHARE_HIGH")
    return out


def cause_of(family, tilt, last):
    ue, sinr, rsrp, ho, weak = last
    if family == "congestion":
        if ue >= 130: return "UE_SURGE"
        if tilt <= 4.5: return "COVERAGE_OVERLAP"
        return "PRB_EXHAUSTION"
    if ho >= 14: return "HO_PARAM_MISTUNED"
    if rsrp <= -108.5: return "COVERAGE_HOLE"
    return "PINGPONG_NEIGHBOR"


def render(a, c, t_end, family, rng):
    """Prompt + answer for cell c, anchored at minute t_end (the event minute, or a normal minute)."""
    tilt = float(a["cell_el_deg"][c, 0])
    lines, seen, last = [f"cell tilt {tilt:.0f} deg"], {}, None
    for off in OFFSETS:
        t = t_end - off
        ue, sinr, rsrp, ho, weak = (float(a[k][c, t]) for k in ("ue_count", "sinr_p10_db", "rsrp_p10_dbm", "ho_count", "weak_ue_frac"))
        lines.append(f"t-{off}m: UE={int(ue)} SINRp10={sinr:.1f}dB RSRPp10={rsrp:.0f}dBm HO/min={int(ho)} weakUE={weak:.2f}")
        for code in alarms_at(ue, sinr, rsrp, ho, weak):
            seen.setdefault(code, off)
        last = (ue, sinr, rsrp, ho, weak)
    alarm_txt = ", ".join(f"{code}@t-{off}m" for code, off in sorted(seen.items(), key=lambda kv: -kv[1])) or "none"
    lines.append(f"alarms: {alarm_txt}")
    lines.append(f"ticket: {TICKETS[int(rng.integers(len(TICKETS)))]}")
    lines.append("Question: which failure family is developing, what is the likely cause, and what is the first action?")
    lines.append("Answer:")
    if family == "none":
        cause, action = "NONE", "NO_ACTION"
    else:
        cause = cause_of(family, tilt, last)
        action = ACTIONS[cause]
    return "\n".join(lines), f" FAMILY: {family.upper() if family != 'none' else 'NONE'} | CAUSE: {cause} | ACTION: {action}", cause, action


def load_site(path):
    cols = ["day", "minute", "cell_id", "ue_count", "rsrp_p10_dbm", "sinr_p10_db", "ho_count", "weak_ue_frac",
            "cell_el_deg", "y_congestion", "y_mobility", "event_congestion", "event_mobility"]
    df = pd.read_csv(path, usecols=cols)
    return {c: df[c].to_numpy().reshape(N_CELLS, T) for c in cols[3:]}


def pick_events(a, family, split, n, rng):
    lo, hi = DAY_RANGES[split][0] * 1440, (DAY_RANGES[split][1] + 1) * 1440
    cells, ts = np.nonzero(a[f"event_{family}"][:, :])
    keep = (ts >= max(lo, max(OFFSETS))) & (ts < hi)
    cells, ts = cells[keep], ts[keep]
    if len(ts) < n:
        raise RuntimeError(f"only {len(ts)} {family} events in {split}, need {n}")
    idx = rng.choice(len(ts), size=n, replace=False)
    return [(int(cells[i]), int(ts[i])) for i in idx]


def pick_normal(a, split, n, rng):
    lo, hi = DAY_RANGES[split][0] * 1440, (DAY_RANGES[split][1] + 1) * 1440
    out = []
    while len(out) < n:
        c, t = int(rng.integers(N_CELLS)), int(rng.integers(max(lo, max(OFFSETS)), hi))
        if a["y_congestion"][c, t] == 0 and a["y_mobility"][c, t] == 0 and np.isfinite(a["y_congestion"][c, t]):
            out.append((c, t))
    return out


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


cfg = json.load(open(f"{CODE}/text_config.json"))
manifest = {"generator": "ran-fl/taskb/make_text.py", "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "config": cfg, "sites": {}}
for site, loc in cfg["sites"].items():
    path = os.path.join(WORK, f"site_{site}.csv")
    boto3.client("s3").download_file(loc["bucket"], loc["key"], path)
    a = load_site(path)
    os.remove(path)
    rng = np.random.default_rng(int(loc["seed"]))
    withheld = cfg["withhold"].get(site, "")
    majority = [f for f in FAMILIES if f != withheld][0] if withheld else loc["majority"]
    site_dir = os.path.join(OUT, site)
    os.makedirs(site_dir, exist_ok=True)
    manifest["sites"][site] = {"withheld_family_at_home": withheld or None, "splits": {}}
    for split in ("train", "val", "test"):
        counts = cfg["counts"][split]
        plan = []  # (family, n)
        if split == "test":
            plan = [("congestion", counts["congestion"]), ("mobility", counts["mobility"]), ("none", counts["none"])]
        else:
            for fam in FAMILIES:
                n = counts["majority"] if fam == majority else int(round(counts["majority"] * cfg.get("rare_fraction", 0.0)))
                if fam == withheld:
                    n = 0
                if n > 0:
                    plan.append((fam, n))
            plan.append(("none", counts["none"]))
        rows = []
        for fam, n in plan:
            picks = pick_normal(a, split, n, rng) if fam == "none" else pick_events(a, fam, split, n, rng)
            for c, t in picks:
                prompt, completion, cause, action = render(a, c, t, fam, rng)
                rows.append({"site": site, "split": split, "family": fam.upper() if fam != "none" else "NONE", "cause": cause, "action": action,
                             "cell": c, "minute_index": t, "prompt": prompt, "completion": completion})
        order = rng.permutation(len(rows))
        rows = [rows[i] for i in order]
        fp = os.path.join(site_dir, f"{split}.jsonl")
        with open(fp, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        by_family = {}
        for r in rows:
            by_family[r["family"]] = by_family.get(r["family"], 0) + 1
        manifest["sites"][site]["splits"][split] = {"examples": len(rows), "by_family": by_family, "sha256": sha(fp)}
        print(site, split, len(rows), by_family, flush=True)
manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=2)
print(json.dumps(manifest["sites"], indent=2))
