"""Runs inside a SageMaker Processing job (CPU deep-learning image, Python 3.12).

1. Installs the Crosstrainer training-client wheel (its pinned dependencies include opacus and safetensors).
2. Runs the utility's own model-code validator with DP enabled (the same loader as the runtime).
3. Dry-runs one federated iteration locally on the real silo CSV with differential privacy on: fetch data, load the
   initial weights, let the framework wrap the optimizer/loader with Opacus, train, evaluate, read the state dict back.
4. Records how the base-class epsilon accounting compares with the honest one for the same noise settings.

Writes report.json to /opt/ml/processing/output.
"""
import glob
import importlib
import json
import os
import subprocess
import sys
import time
import traceback

CODE = "/opt/ml/processing/input/code"
OUT = "/opt/ml/processing/output"
os.makedirs(OUT, exist_ok=True)
report: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=False)


def save():
    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)


try:
    wheel = glob.glob(f"{CODE}/crosstrainer_training_client_bundled-*.whl")[0]
    r = run([sys.executable, "-m", "pip", "install", "-q", wheel, "pandas"])
    report["pip_install_exit"] = r.returncode
    report["python"] = sys.version
    import torch
    report["torch"] = torch.__version__

    # 1. The validator, exactly as `validate-model` / `create-session` run it.
    vjson = os.path.join(OUT, "validate_model.json")
    r = run([sys.executable, f"{CODE}/validate_model.py", "--entry-file", "model.py",
             "--model-code-dir", f"{CODE}/model-code", "--dp-enabled", "--output-json", vjson])
    report["validate_model_exit"] = r.returncode
    if os.path.exists(vjson):
        report["validate_model"] = json.load(open(vjson))
    save()

    # 2. Dry run of one federated iteration with DP on.
    sys.path.insert(0, f"{CODE}/model-code")
    model = importlib.import_module("model")
    from crosstrainer_training_client.configuration.hfl_pytorch_configuration import HFLPytorchConfiguration
    from crosstrainer_training_client.typedefs import NNHFLWithDPProtocolName
    from safetensors.numpy import load_file

    os.environ.setdefault("CLIENT_INDEX", "1")
    cfg = model.RanEarlyWarning(iteration=1, context={}, n_global_fl_iterations=3, protocol_name=NNHFLWithDPProtocolName)

    t0 = time.time()
    train, val, test = cfg.fetch_and_prepare_data()
    report["data_seconds"] = round(time.time() - t0, 1)
    report["train_windows"] = len(train.dataset)
    report["val_windows"] = len(val.dataset)

    m, opt = cfg.create_model()
    sd = load_file(f"{CODE}/initial.safetensors")
    report["initial_keys"] = sorted(sd.keys())
    report["model_keys"] = sorted(m.state_dict().keys())
    m = cfg.set_weights_from_state_dict(sd, m)
    report["before_training"] = cfg.evaluate(m, train, val, test)
    ds = train.dataset
    report["withhold_env"] = os.environ.get("RAN_WITHHOLD_LABEL", "")
    report["train_label_means_congestion_mobility"] = [float(torch.nanmean(ds.y[ds.anchors, 0])), float(torch.nanmean(ds.y[ds.anchors, 1]))]
    if os.environ.get("RUN_TRAIN", "1") == "0":
        report["status"] = "ok (data, validator and evaluate only; DP training skipped)"
        save()
        sys.exit(0)

    tr, va, te, m, opt = cfg.prepare_data_and_model(train, val, test, m, opt)
    n, b = len(train.dataset), train.batch_size
    std, clip = cfg._compute_dp_sgd_parameters(n, b)
    honest = cfg._compute_privacy_budget(n, std, clip, b)
    base = HFLPytorchConfiguration._compute_privacy_budget(cfg, n, std, clip, b)
    report["dp"] = {
        "target_epsilon": cfg.privacy_budget(),
        "noise_multiplier": float(opt.noise_multiplier),
        "max_grad_norm": float(opt.max_grad_norm),
        "gaussian_std_dev": std,
        "clipping_bound": clip,
        "n_training_samples": n,
        "batch_size": b,
        "delta": float(n) ** -2,
        "epsilon_honest_accounting": honest,
        "epsilon_base_class_accounting_same_noise": base,
        "steps_per_epoch": -(-n // b),
        "total_steps": cfg.n_local_epochs() * 3 * (-(-n // b)),
    }
    save()

    t0 = time.time()
    out = cfg.train(m, tr, va, te, opt)
    report["train_seconds"] = round(time.time() - t0, 1)
    report["num_samples_returned"] = out.num_samples
    report["after_one_iteration"] = cfg.evaluate(out.model, tr, va, te)
    state = cfg.get_state_dict(out.model)
    report["state_dict_keys_after_dp"] = sorted(state.keys())
    report["state_dict_shapes"] = {k: list(v.shape) for k, v in state.items()}
    report["status"] = "ok"
except Exception as exc:  # noqa: BLE001
    report["status"] = "error"
    report["error"] = f"{type(exc).__name__}: {exc}"
    report["traceback"] = traceback.format_exc()
    print(report["traceback"], flush=True)
finally:
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save()
    print(json.dumps(report, indent=2, default=str), flush=True)
