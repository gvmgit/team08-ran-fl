"""Baselines and grader for Task A (SageMaker Processing, CPU deep-learning image, Python 3.12).

Reuses model.py (windows, scaling, split rule, network) so every comparison runs the same code. The grader is a
deterministic script, not a model, so it is independent of whatever produced the answers.

job_config.json (next to this file):
{
  "sites":   {"a": {"bucket": "...", "key": "..."}, "b": {...}},
  "train":   [{"name": "local_a", "sites": ["a"], "epochs": 6}, {"name": "central", "sites": ["a", "b"], "epochs": 6}],
  "score":   [{"name": "local_a", "weights": "trained:local_a"}, {"name": "federated", "weights": "s3://bucket/key"}],
  "dp":      {"n": ..., "batch_size": 512, "noise_multiplier": ..., "local_epochs": 2, "global_iterations": 3},   (optional)
  "expected_test_slice_sha256": {"a": "...", "b": "..."}                                                     (optional)
}

Scoring rules (declared before any run):
  * AUROC and F1 for y_congestion and y_mobility on the frozen test days (96-119), every minute (stride 1).
  * F1 uses the threshold that maximises F1 on that site's validation days (85-94) for that model and label.
  * Lead-time recall: an event (event_* == 1 at minute T) counts as warned >= 60 minutes ahead when the model's
    probability reaches the threshold at any minute in [T-120, T-60]. Events with T < 96*1440 + 120 are excluded so
    every lead window lies inside the test days.
"""
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

CODE = "/opt/ml/processing/input/code"
OUT = "/opt/ml/processing/output"
WORK = "/tmp/work"
os.makedirs(OUT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)
report: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def save():
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd):
    log("+ " + " ".join(cmd))
    return subprocess.run(cmd, check=False)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_slice_sha256(path, lo_day, hi_day):
    """SHA-256 over the raw CSV lines of the frozen test days, joined with newlines."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        next(f)  # header
        for line in f:
            day = int(line.split(b",", 1)[0])
            if lo_day <= day <= hi_day:
                h.update(line.rstrip(b"\r\n") + b"\n")
    return h.hexdigest()


try:
    cfg = json.load(open(f"{CODE}/job_config.json"))
    wheel = glob.glob(f"{CODE}/crosstrainer_training_client_bundled-*.whl")[0]
    r = run([sys.executable, "-m", "pip", "install", "-q", wheel, "pandas"])
    report["pip_exit"] = r.returncode
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from safetensors.torch import load_file, save_file
    from torch.utils.data import ConcatDataset, DataLoader

    sys.path.insert(0, f"{CODE}/model-code")
    import model as M

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    report["torch"] = torch.__version__
    report["split"] = {"train_days": M.TRAIN_DAYS, "val_days": M.VAL_DAYS, "test_days": M.TEST_DAYS, "train_stride": M.TRAIN_STRIDE}

    # ---- data -------------------------------------------------------------------------------------------------
    import boto3

    tensors, events, hashes = {}, {}, {}
    for site, loc in cfg["sites"].items():
        path = os.path.join(WORK, f"site_{site}.csv")
        t0 = time.time()
        boto3.client("s3").download_file(loc["bucket"], loc["key"], path)
        hashes[site] = {"file_sha256": sha256_file(path), "test_slice_sha256": test_slice_sha256(path, *M.TEST_DAYS)}
        tensors[site] = M.load_site_csv(path)
        ev = pd.read_csv(path, usecols=["event_congestion", "event_mobility"], dtype=np.int8)
        events[site] = ev.to_numpy().reshape(M.N_CELLS, M.T, 2)
        os.remove(path)
        log(f"site {site} ready in {time.time() - t0:.0f}s")
    report["data_hashes"] = hashes
    exp = cfg.get("expected_test_slice_sha256")
    if exp:
        report["test_set_hash_matches_declaration"] = {s: hashes[s]["test_slice_sha256"] == exp[s] for s in exp}
    save()

    init = load_file(f"{CODE}/initial.safetensors")

    # Stress experiment: {"a": "mobility", "b": "congestion"} means site a never logged mobility and site b never
    # logged congestion. Those labels are withheld (all zeros) for TRAINING only; scoring always uses the true labels.
    withhold = cfg.get("withhold", {})
    _train_label_cache = {}

    def train_labels(site):
        if site not in _train_label_cache:
            y = tensors[site][1]
            lab = withhold.get(site)
            if lab:
                k = M.LABELS.index(f"y_{lab}")
                y = y.clone()
                col = y[:, :, k]
                y[:, :, k] = torch.where(torch.isnan(col), col, torch.zeros_like(col))
            _train_label_cache[site] = y
        return _train_label_cache[site]

    report["withheld_labels_at_home"] = withhold

    # ---- baselines --------------------------------------------------------------------------------------------
    trained = {}
    report["training"] = {}
    for spec in cfg.get("train", []):
        name, sites, epochs = spec["name"], spec["sites"], spec["epochs"]
        torch.manual_seed(1234)
        np.random.seed(1234)
        parts = [M.WindowDataset(tensors[s][0], train_labels(s), M.make_anchors(train_labels(s), M.TRAIN_DAYS, M.TRAIN_STRIDE)) for s in sites]
        loader = DataLoader(ConcatDataset(parts), batch_size=M.RanEarlyWarning.batch_size, shuffle=True)
        net = M.build_net()
        net.load_state_dict(init)
        opt = torch.optim.Adam(net.parameters(), lr=M.RanEarlyWarning.learning_rate)
        crit = nn.BCEWithLogitsLoss()
        curve = []
        for epoch in range(epochs):
            net.train()
            t0, run_loss, nb = time.time(), 0.0, 0
            for x, y in loader:
                opt.zero_grad(set_to_none=True)
                loss = crit(net(x), y)
                loss.backward()
                opt.step()
                run_loss += float(loss.item())
                nb += 1
            curve.append({"epoch": epoch + 1, "train_loss": run_loss / max(1, nb), "seconds": round(time.time() - t0, 1)})
            log(f"train {name} epoch {epoch + 1}/{epochs} loss {curve[-1]['train_loss']:.4f}")
        trained[name] = {k: v.detach().clone() for k, v in net.state_dict().items()}
        save_file(trained[name], os.path.join(OUT, f"{name}.safetensors"))
        report["training"][name] = {"sites": sites, "epochs": epochs, "windows": int(sum(len(p) for p in parts)), "curve": curve}
        save()

    # ---- scoring ----------------------------------------------------------------------------------------------
    def predict(net, x, lo, hi):
        """Probabilities for every minute t in [lo, hi) (all cells), shape (cells, hi - lo, 2)."""
        net.eval()
        out = torch.empty(M.N_CELLS, hi - lo, 2)
        with torch.inference_mode():
            for c in range(M.N_CELLS):
                seg = x[c, lo - (M.WINDOW - 1): hi]  # (hi - lo + 59, F)
                win = seg.unfold(0, M.WINDOW, 1)  # (hi - lo, F, 60)
                for i in range(0, win.shape[0], 16384):
                    chunk = win[i: i + 16384].permute(0, 2, 1).reshape(-1, M.WINDOW * M.N_FEAT)
                    out[c, i: i + chunk.shape[0]] = torch.sigmoid(net(chunk))
        return out.numpy()

    def best_f1_threshold(y, p):
        order = np.argsort(-p, kind="mergesort")
        ys = y[order]
        tp = np.cumsum(ys)
        fp = np.cumsum(1 - ys)
        fn = ys.sum() - tp
        f1 = 2 * tp / np.maximum(1, 2 * tp + fp + fn)
        i = int(np.argmax(f1))
        return float(p[order][i])

    def prf(y, pred):
        tp = float(((pred == 1) & (y == 1)).sum())
        fp = float(((pred == 1) & (y == 0)).sum())
        fn = float(((pred == 0) & (y == 1)).sum())
        precision = tp / max(1.0, tp + fp)
        recall = tp / max(1.0, tp + fn)
        f1 = 2 * tp / max(1.0, 2 * tp + fp + fn)
        return precision, recall, f1

    val_lo, val_hi = M.VAL_DAYS[0] * M.MIN_PER_DAY, (M.VAL_DAYS[1] + 1) * M.MIN_PER_DAY
    te_lo, te_hi = M.TEST_DAYS[0] * M.MIN_PER_DAY, (M.TEST_DAYS[1] + 1) * M.MIN_PER_DAY
    report["scores"] = {}
    for spec in cfg.get("score", []):
        name, wsrc = spec["name"], spec["weights"]
        if wsrc.startswith("trained:"):
            sd = trained[wsrc.split(":", 1)[1]]
        else:
            bucket, key = wsrc[5:].split("/", 1)
            local = os.path.join(WORK, "w.safetensors")
            boto3.client("s3").download_file(bucket, key, local)
            raw = load_file(local)
            sd = {k.removeprefix("_module."): v for k, v in raw.items()}
            report.setdefault("weight_files", {})[name] = {"source": wsrc, "sha256": sha256_file(local)}
        net = M.build_net()
        net.load_state_dict(sd)
        report["scores"][name] = {}
        preds = {s: (predict(net, tensors[s][0], val_lo, val_hi), predict(net, tensors[s][0], te_lo, te_hi)) for s in cfg["sites"]}
        for site in cfg["sites"]:
            x, y = tensors[site]
            pv, pt = preds[site]
            site_res = {}
            for k, label in enumerate(M.LABELS):
                short = label.replace("y_", "")
                yv = y[:, val_lo:val_hi, k].numpy()
                yt = y[:, te_lo:te_hi, k].numpy()
                mv, mt = ~np.isnan(yv), ~np.isnan(yt)
                withheld_here = withhold.get(site) == short
                if withheld_here:
                    # This site never logged the family, so it has no labels to tune a threshold on: calibrate it
                    # on the validation days of a site that does log it (same model, true labels there).
                    ref = [s for s in cfg["sites"] if s != site and withhold.get(s) != short][0]
                    ref_y = tensors[ref][1][:, val_lo:val_hi, k].numpy()
                    ref_m = ~np.isnan(ref_y)
                    thr = best_f1_threshold(ref_y[ref_m].astype(np.int8), preds[ref][0][:, :, k][ref_m])
                    thr_source = ref
                else:
                    thr = best_f1_threshold(yv[mv].astype(np.int8), pv[:, :, k][mv])
                    thr_source = site
                yy, pp = yt[mt].astype(np.int8), pt[:, :, k][mt]
                precision, recall, f1 = prf(yy, (pp >= thr).astype(np.int8))
                ev = events[site][:, :, k]
                warned60 = warned_any = n_events = 0
                records = []
                for c, t_evt in zip(*np.nonzero(ev[:, te_lo + M.HORIZON: te_hi])):
                    t_evt = int(t_evt) + te_lo + M.HORIZON
                    n_events += 1
                    win = pt[c, t_evt - M.HORIZON - te_lo: t_evt - te_lo, k]  # minutes T-120 .. T-1
                    warned_any += int(win.max() >= thr)
                    warned60 += int(win[: M.HORIZON - 60 + 1].max() >= thr)  # minutes T-120 .. T-60
                    hit = np.nonzero(win >= thr)[0]
                    records.append({
                        "run_id": os.environ.get("RUN_ID", ""), "model": name, "site": site, "label": label, "cell": int(c),
                        "event_minute": t_evt, "day": t_evt // M.MIN_PER_DAY, "minute_of_day": t_evt % M.MIN_PER_DAY,
                        "threshold": float(thr),
                        "max_prob_lead_120_to_60": float(win[: M.HORIZON - 60 + 1].max()), "max_prob_lead_120_to_1": float(win.max()),
                        "first_alarm_lead_minutes": int(M.HORIZON - hit[0]) if len(hit) else None,
                        "warned_60min_ahead": bool(win[: M.HORIZON - 60 + 1].max() >= thr), "warned_any_lead": bool(win.max() >= thr),
                    })
                os.makedirs(os.path.join(OUT, "records"), exist_ok=True)
                rec_path = os.path.join(OUT, "records", f"{name}__{site}__{label}.jsonl")
                with open(rec_path, "w") as rf:
                    for rec in records:
                        rf.write(json.dumps(rec) + "\n")
                report.setdefault("records_sha256", {})[f"{name}__{site}__{label}"] = sha256_file(rec_path)
                site_res[label] = {
                    "auroc": M.auroc(yy, pp), "f1": f1, "precision": precision, "recall": recall,
                    "threshold_from_validation": thr, "threshold_source_site": thr_source, "withheld_at_home": withheld_here,
                    "test_windows": int(mt.sum()), "test_positives": int(yy.sum()),
                    "events_scored": n_events,
                    "fraction_warned_60min_ahead": warned60 / max(1, n_events),
                    "fraction_warned_any_lead": warned_any / max(1, n_events),
                }
            report["scores"][name][site] = site_res
            log(f"scored {name} on site {site}")
        save()

    # ---- epsilon ----------------------------------------------------------------------------------------------
    if cfg.get("dp"):
        from opacus.accountants import rdp as rdp_accountant
        from opacus.accountants.analysis import rdp as rdp_analysis

        from crosstrainer_training_client.configuration.hfl_pytorch_configuration import HFLPytorchConfiguration
        from crosstrainer_training_client.typedefs import NNHFLWithDPProtocolName

        d = cfg["dp"]
        iters, b = int(d["global_iterations"]), M.RanEarlyWarning.batch_size
        orders = rdp_accountant.RDPAccountant.DEFAULT_ALPHAS

        class _BaseAccounting(M.RanEarlyWarning):
            """Same code, but with the framework's original accountant (each local epoch counted as one step)."""
            _compute_privacy_budget = HFLPytorchConfiguration._compute_privacy_budget

        report["epsilon"] = {}
        for site in cfg["sites"]:
            n = int(len(M.make_anchors(tensors[site][1], M.TRAIN_DAYS, M.TRAIN_STRIDE)))
            spe = -(-n // b)
            delta = float(n) ** -2

            def eps(sigma, steps):
                rdp = rdp_analysis.compute_rdp(q=1.0 / spe, noise_multiplier=sigma, steps=steps, orders=orders)
                return float(rdp_analysis.get_privacy_spent(orders=orders, rdp=rdp, delta=delta)[0])

            honest = M.RanEarlyWarning(iteration=1, context={}, n_global_fl_iterations=iters, protocol_name=NNHFLWithDPProtocolName)
            base = _BaseAccounting(iteration=1, context={}, n_global_fl_iterations=iters, protocol_name=NNHFLWithDPProtocolName)
            std, clip = honest._compute_dp_sgd_parameters(n, b)
            std_b, clip_b = base._compute_dp_sgd_parameters(n, b)
            real_steps = honest.n_local_epochs() * iters * spe
            report["epsilon"][site] = {
                "target_epsilon": honest.privacy_budget(), "delta": delta, "n_training_windows": n, "batch_size": b,
                "local_epochs": honest.n_local_epochs(), "global_iterations": iters,
                "steps_per_epoch": spe, "real_optimizer_steps": real_steps, "sampling_rate": 1.0 / spe,
                "noise_multiplier_used": std / clip, "max_grad_norm_used": clip,
                "epsilon_spent_real_steps": eps(std / clip, real_steps),
                "if_framework_accounting_were_left_unchanged": {
                    "noise_multiplier_it_would_have_chosen": std_b / clip_b, "max_grad_norm": clip_b,
                    "real_epsilon_that_would_have_resulted": eps(std_b / clip_b, real_steps),
                },
            }
    report["status"] = "ok"
except Exception as exc:  # noqa: BLE001
    report["status"] = "error"
    report["error"] = f"{type(exc).__name__}: {exc}"
    report["traceback"] = traceback.format_exc()
    print(report["traceback"], flush=True)
finally:
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save()
    print(json.dumps({k: v for k, v in report.items() if k not in ("scores", "training")}, indent=2, default=str), flush=True)
