"""Task B model-code check (SageMaker Processing, CPU). Validates the code the way the utility does, loads the silo
text and the base model, and verifies that the INITIAL ADAPTER's tensor names and shapes are exactly what a client
exchanges (adapter_state) - a mismatch would fail the very first round. No training and no scoring here (CPU)."""
import glob
import json
import os
import subprocess
import sys
import time
import traceback

CODE = "/opt/ml/processing/input/code"
OUT = "/opt/ml/processing/output"
os.makedirs(OUT, exist_ok=True)
report = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def save():
    json.dump(report, open(os.path.join(OUT, "report.json"), "w"), indent=2, default=str)


try:
    wheel = glob.glob(f"{CODE}/crosstrainer_training_client_bundled-*.whl")[0]
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", wheel, "transformers==4.52.4", "peft==0.15.2", "accelerate"], check=False)
    report["pip_exit"] = r.returncode
    import numpy as np
    import torch
    from safetensors.torch import load_file

    # 1. the utility's own validator (no DP for Task B)
    vjson = os.path.join(OUT, "validate_model.json")
    r = subprocess.run([sys.executable, f"{CODE}/validate_model.py", "--entry-file", "model.py", "--model-code-dir", f"{CODE}/model-code", "--output-json", vjson], check=False)
    report["validate_model_exit"] = r.returncode
    if os.path.exists(vjson):
        report["validate_model"] = json.load(open(vjson))
    save()

    # 2. the client's own methods
    sys.path.insert(0, f"{CODE}/model-code")
    import model as M

    cfg = M.RanNlmLora(iteration=1, context={}, n_global_fl_iterations=3, protocol_name=None)
    t0 = time.time()
    train, val, test = cfg.fetch_and_prepare_data()
    report["data_and_base_seconds"] = round(time.time() - t0, 1)
    report["train_examples"], report["val_examples"] = len(train.dataset), len(val.dataset)
    batch = next(iter(train))
    report["batch_shapes"] = {k: list(v.shape) for k, v in batch.items()}
    report["longest_example_tokens"] = int(max(len(ids) for ids, _ in train.dataset.items))
    t0 = time.time()
    model, optimizer = cfg.create_model()
    report["create_model_seconds"] = round(time.time() - t0, 1)

    init = load_file(f"{CODE}/initial_adapter.safetensors")
    state = cfg.get_state_dict(model)
    report["initial_vs_client_state"] = {
        "same_names": sorted(init) == sorted(state),
        "same_shapes": all(tuple(init[k].shape) == tuple(state[k].shape) for k in state if k in init),
        "tensors": len(state), "values": int(sum(v.size for v in state.values())),
        "megabytes_exchanged_per_round": round(sum(v.nbytes for v in state.values()) / 1e6, 1),
    }
    model = cfg.set_weights_from_state_dict({k: v.numpy() for k, v in init.items()}, model)
    after = cfg.get_state_dict(model)
    report["round_trip_exact"] = all(np.array_equal(after[k], init[k].numpy()) for k in init)
    report["only_adapter_in_state"] = all("lora_" in k for k in state)
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
