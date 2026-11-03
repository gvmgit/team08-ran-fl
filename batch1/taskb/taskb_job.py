"""Task B baselines and grader (SageMaker TRAINING job, script mode, GPU).

GPU Processing is not available in this account (quota 0), so this runs as a script-mode training job. It reuses the
federated model code (taskb/model-code/model.py): the same base model, LoRA setup, text datasets, learning rate and
batch size, so every comparison runs the same code.

job_config.json:
  sites      {"a": {"bucket": ..., "prefix": "text/"}, "b": {...}}
  base_s3    "bucket/prefix" of the frozen base model (env NLM_BASE_S3 is set from it)
  train      [{"name": "local_a", "sites": ["a"], "epochs": 6}, {"name": "central", "sites": ["a", "b"], "epochs": 6}]
  score      [{"name": "untuned_base", "adapter": "initial"}, {"name": "local_a", "adapter": "trained:local_a"},
              {"name": "federated", "adapter": "s3://bucket/key.safetensors"}]

Scoring (deterministic, no generation needed): for every test example the model's log-likelihood ranks the candidate
FAMILY (3), then the CAUSE (3 within the true family), then the ACTION (3 within the true family and cause). The
headline is FAMILY accuracy on the family each site never logged at home. Three greedy generations per site and
family are saved as qualitative samples.

Evidence written to /opt/ml/model: results.json (counts, adapter file hashes, test-set hashes) and records/<model>__<site>.jsonl,
one line per test example with the true labels, the candidate log-likelihoods at each stage and the choice made. The run id
(env RUN_ID, the training-job name) is in every record.
"""
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

SRC = os.path.dirname(os.path.abspath(__file__))
OUT = "/opt/ml/model"
os.makedirs(OUT, exist_ok=True)
report = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def save():
    json.dump(report, open(os.path.join(OUT, "results.json"), "w"), indent=2, default=str)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


EVIDENCE = {}  # filled from job_config.json "evidence": {"bucket": ..., "region": ...}


def put_evidence(local_path, rel):
    """Copy a result file to s3://<evidence bucket>/ran-fl/jobs/<RUN_ID>/output/<rel>, so the read-only live test can fetch a
    single record and its S3 write time can be compared with this job's own log times. A failed upload is recorded, not fatal."""
    if not EVIDENCE.get("bucket"):
        return
    key = f"ran-fl/jobs/{os.environ.get('RUN_ID', 'unknown')}/output/{rel}"
    try:
        import boto3

        boto3.client("s3", region_name=EVIDENCE.get("region", "us-east-2")).upload_file(local_path, EVIDENCE["bucket"], key)
        report.setdefault("evidence_uploaded", []).append(key)
    except Exception as exc:  # noqa: BLE001
        report.setdefault("evidence_upload_errors", []).append(f"{key}: {type(exc).__name__}: {exc}")


try:
    cfg = json.load(open(os.path.join(SRC, "job_config.json")))
    os.environ["NLM_BASE_S3"] = cfg["base_s3"]
    EVIDENCE.update(cfg.get("evidence", {}))
    wheel = glob.glob(os.path.join(SRC, "crosstrainer_training_client_bundled-*.whl"))[0]
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", wheel], check=False)
    report["wheel_install_exit"] = r.returncode
    import boto3
    import numpy as np
    import torch
    from safetensors.torch import load_file, save_file
    from torch.utils.data import ConcatDataset, DataLoader

    sys.path.insert(0, os.path.join(SRC, "model-code"))
    import model as M

    report["torch"], report["gpu"] = torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s3 = boto3.client("s3")

    t0 = time.time()
    base_dir = M.fetch_base_model("/tmp/nlm")
    tok = M.load_tokenizer(base_dir)
    report["base_download_seconds"] = round(time.time() - t0, 1)

    rows = {}
    for site, loc in cfg["sites"].items():
        rows[site] = {}
        for split in ("train", "val", "test"):
            p = f"/tmp/{site}_{split}.jsonl"
            s3.download_file(loc["bucket"], f"{loc['prefix']}{split}.jsonl", p)
            rows[site][split] = M.read_jsonl(p)
    report["corpus"] = {s: {sp: len(v) for sp, v in d.items()} for s, d in rows.items()}

    init_path = os.path.join(SRC, "initial_adapter.safetensors")
    init = {k: v.numpy() for k, v in load_file(init_path).items()}
    torch.manual_seed(1234)
    net = M.build_peft_model(base_dir, device)
    report["load_seconds"] = round(time.time() - t0, 1)
    save()

    # ---- baselines ---------------------------------------------------------------------------------------------
    trained = {}
    report["training"] = {}
    for spec in cfg.get("train", []):
        name, sites, epochs = spec["name"], spec["sites"], spec["epochs"]
        M.load_adapter_state(net, init)
        torch.manual_seed(1234)
        train_ds = ConcatDataset([M.TextDataset(rows[s]["train"], tok) for s in sites])
        val_ds = ConcatDataset([M.TextDataset(rows[s]["val"], tok) for s in sites])
        collate = M.make_collate(tok.pad_token_id)
        loader = DataLoader(train_ds, batch_size=M.BATCH_SIZE, shuffle=True, collate_fn=collate)
        val_loader = DataLoader(val_ds, batch_size=M.BATCH_SIZE, collate_fn=collate)
        opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=M.LEARNING_RATE, weight_decay=0.0)
        curve = []
        for epoch in range(epochs):
            net.train()
            t1, run_loss, nb = time.time(), 0.0, 0
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    loss = net(**batch).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 1.0)
                opt.step()
                run_loss += float(loss.item())
                nb += 1
            net.eval()
            vt, vn = 0.0, 0
            with torch.inference_mode():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        l = net(**batch).loss
                    n = int((batch["labels"][:, 1:] != -100).sum())
                    vt += float(l.item()) * n
                    vn += n
            curve.append({"epoch": epoch + 1, "train_loss": run_loss / max(1, nb), "val_loss": vt / max(1, vn), "seconds": round(time.time() - t1, 1)})
            log(f"train {name} epoch {epoch + 1}/{epochs} train {curve[-1]['train_loss']:.4f} val {curve[-1]['val_loss']:.4f}")
        trained[name] = M.adapter_state(net)
        save_file({k: torch.from_numpy(v) for k, v in trained[name].items()}, os.path.join(OUT, f"{name}.safetensors"))
        report["training"][name] = {"sites": sites, "epochs": epochs, "examples": len(train_ds), "curve": curve}
        save()

    # ---- scoring -----------------------------------------------------------------------------------------------
    def sha256_file(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def adapter_for(spec):
        src = spec["adapter"]
        if src == "initial":
            report.setdefault("adapter_files", {})[spec["name"]] = {"source": "initial_adapter.safetensors", "sha256": sha256_file(init_path), "tensors": len(init)}
            return init
        if src.startswith("trained:"):
            return trained[src.split(":", 1)[1]]
        bucket, key = src[5:].split("/", 1)
        local = "/tmp/adapter_dl.safetensors"
        s3.download_file(bucket, key, local)
        raw = load_file(local)
        report.setdefault("adapter_files", {})[spec["name"]] = {"source": src, "sha256": sha256_file(local), "tensors": len(raw)}
        return {k: v.numpy() for k, v in raw.items()}

    # One record per test example (the evidence a judge can open): the true labels, the log-likelihood the model gave to every
    # candidate at each of the three stages, and the choice made. The run id is the training-job name (env RUN_ID).
    run_id = os.environ.get("RUN_ID", "")
    report["run_id"] = run_id
    report["test_set_sha256"] = {s: sha256_file(f"/tmp/{s}_test.jsonl") for s in cfg["sites"]}
    rec_dir = os.path.join(OUT, "records")
    os.makedirs(rec_dir, exist_ok=True)
    report["records_sha256"] = {}

    def lp(scores, options):
        return {o: round(float(s), 4) for o, s in zip(options, scores)}

    report["scores"], report["samples"] = {}, {}
    for spec in cfg.get("score", []):
        name = spec["name"]
        M.load_adapter_state(net, adapter_for(spec))
        net.eval()
        report["scores"][name], report["samples"][name] = {}, {}
        t1 = time.time()
        for site in cfg["sites"]:
            per = {}
            records = []
            for idx, ex in enumerate(rows[site]["test"]):
                fam = ex["family"]
                d = per.setdefault(fam, {"n": 0, "family_correct": 0, "cause_correct": 0, "action_correct": 0, "confusion": {}})
                fam_scores = M.candidate_logprobs(net, tok, ex["prompt"], [f" FAMILY: {f}" for f in M.FAMILIES])
                pred = M.FAMILIES[int(np.argmax(fam_scores))]
                rec = {"run_id": run_id, "model": name, "site": site, "test_index": idx, "cell": ex.get("cell"), "minute_index": ex.get("minute_index"),
                       "true_family": fam, "true_cause": ex.get("cause"), "true_action": ex.get("action"),
                       "family_logprobs": lp(fam_scores, M.FAMILIES), "pred_family": pred}
                d["n"] += 1
                d["family_correct"] += int(pred == fam)
                d["confusion"][pred] = d["confusion"].get(pred, 0) + 1
                if fam != "NONE":
                    cause_opts = M.CAUSES[fam]
                    cause_scores = M.candidate_logprobs(net, tok, ex["prompt"], [f" FAMILY: {fam} | CAUSE: {c}" for c in cause_opts])
                    pc = cause_opts[int(np.argmax(cause_scores))]
                    action_opts = M.ACTIONS[fam]
                    action_scores = M.candidate_logprobs(net, tok, ex["prompt"], [f" FAMILY: {fam} | CAUSE: {ex['cause']} | ACTION: {a}" for a in action_opts])
                    pa = action_opts[int(np.argmax(action_scores))]
                    d["cause_correct"] += int(pc == ex["cause"])
                    d["action_correct"] += int(pa == ex["action"])
                    rec.update({"cause_logprobs": lp(cause_scores, cause_opts), "pred_cause": pc, "action_logprobs": lp(action_scores, action_opts), "pred_action": pa})
                records.append(rec)
            rec_path = os.path.join(rec_dir, f"{name}__{site}.jsonl")
            with open(rec_path, "w") as f:
                f.write("".join(json.dumps(r) + "\n" for r in records))
            report["records_sha256"][f"{name}__{site}"] = sha256_file(rec_path)
            put_evidence(rec_path, f"records/{name}__{site}.jsonl")
            for fam, d in per.items():
                d["family_acc"] = d["family_correct"] / d["n"]
                if fam != "NONE":
                    d["cause_acc_given_family"] = d["cause_correct"] / d["n"]
                    d["action_acc_given_cause"] = d["action_correct"] / d["n"]
            total = sum(d["n"] for d in per.values())
            per["ALL"] = {"n": total, "family_acc": sum(d["family_correct"] for k, d in per.items() if k != "ALL") / max(1, total)}
            report["scores"][name][site] = per
            # three greedy generations per family, for display
            samples = []
            for fam in ("CONGESTION", "MOBILITY"):
                for ex in [e for e in rows[site]["test"] if e["family"] == fam][:3]:
                    ids = tok(ex["prompt"], add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
                    with torch.inference_mode():
                        out = net.generate(input_ids=ids, max_new_tokens=40, do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
                    samples.append({"true_family": fam, "expected": ex["completion"].strip(), "generated": tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()})
            report["samples"][name][site] = samples
        log(f"scored {name} in {time.time() - t1:.0f}s")
        save()
    report["status"] = "ok"
except Exception as exc:  # noqa: BLE001
    report["status"] = "error"
    report["error"] = f"{type(exc).__name__}: {exc}"
    report["traceback"] = traceback.format_exc()
    print(report["traceback"], flush=True)
finally:
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save()
    put_evidence(os.path.join(OUT, "results.json"), "results.json")
    print(json.dumps({k: v for k, v in report.items() if k not in ("scores", "training", "samples")}, indent=2, default=str), flush=True)
