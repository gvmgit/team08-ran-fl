"""Builds the initial LoRA adapter (SageMaker Processing, CPU) with the very same code the clients use, so the tensor
names and shapes match exactly. PEFT initialises lora_A randomly and lora_B to zero, so the initial adapter leaves the
base model's behaviour unchanged. Also runs a first end-to-end sanity check on a real generated example."""
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
    json.dump(report, open(os.path.join(OUT, "initial_adapter_report.json"), "w"), indent=2, default=str)


try:
    wheel = glob.glob(f"{CODE}/crosstrainer_training_client_bundled-*.whl")[0]
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", wheel, "transformers==4.52.4", "peft==0.15.2", "accelerate"], check=False)
    report["pip_exit"] = r.returncode
    import numpy as np
    import torch
    from safetensors.torch import save_file

    sys.path.insert(0, f"{CODE}/model-code")
    import model as M

    report["torch"], report["python"] = torch.__version__, sys.version.split()[0]
    import peft
    import transformers
    report["transformers"], report["peft"] = transformers.__version__, peft.__version__

    t0 = time.time()
    base_dir = M.fetch_base_model("/tmp/nlm")
    report["base_download_seconds"] = round(time.time() - t0, 1)
    report["base_files"] = sorted(os.listdir(base_dir))
    torch.manual_seed(20261102)
    t0 = time.time()
    net = M.build_peft_model(base_dir, torch.device("cpu"))
    report["model_load_seconds"] = round(time.time() - t0, 1)

    state = M.adapter_state(net)
    names = sorted(state)
    report["adapter"] = {
        "tensors": len(state), "values": int(sum(v.size for v in state.values())),
        "megabytes_float32": round(sum(v.nbytes for v in state.values()) / 1e6, 1),
        "lora_B_all_zero": all(float(np.abs(v).max()) == 0.0 for k, v in state.items() if "lora_B" in k),
        "lora_A_nonzero": all(float(np.abs(v).max()) > 0.0 for k, v in state.items() if "lora_A" in k),
        "first_names": names[:3], "last_name": names[-1],
    }
    save_file({k: torch.from_numpy(v) for k, v in state.items()}, os.path.join(OUT, "initial_adapter.safetensors"))
    trainable_total = sum(p.numel() for p in net.parameters() if p.requires_grad)
    all_total = sum(p.numel() for p in net.parameters())
    report["parameters"] = {"trainable": int(trainable_total), "all": int(all_total), "fraction_trainable": trainable_total / all_total}

    # Sanity: tokenizer, a real example, a forward pass, and a round trip of the adapter state.
    tok = M.load_tokenizer(base_dir)
    rows = M.read_jsonl(f"{CODE}/sample_test.jsonl")
    report["sample_prompt_tokens"] = len(tok(rows[0]["prompt"], add_special_tokens=False)["input_ids"])
    t0 = time.time()
    scores = M.candidate_logprobs(net, tok, rows[0]["prompt"], [f" FAMILY: {f}" for f in M.FAMILIES])
    report["forward_seconds_cpu"] = round(time.time() - t0, 1)
    report["untuned_family_logprobs"] = dict(zip(M.FAMILIES, scores))
    report["untuned_prediction"], report["true_family"] = M.FAMILIES[int(np.argmax(scores))], rows[0]["family"]
    M.load_adapter_state(net, state)  # round trip must not raise
    ds = M.TextDataset(rows[:4], tok)
    batch = M.make_collate(tok.pad_token_id)([ds[i] for i in range(4)])
    out = net(**batch)
    report["loss_untuned_on_4_examples"] = float(out.loss)
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
