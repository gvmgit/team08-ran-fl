"""Task B: federated LoRA adapter on network alarm text.

The base model (OTel-LLM-1.7B-IT, a Qwen3 model) is frozen and identical in every silo. Only LoRA adapter tensors are
trained and exchanged: get_state_dict / set_weights_from_state_dict are overridden so the base weights never leave the
silo. One HFLPytorchConfiguration subclass, identical in every silo, like Task A.

Environment (set by the utility or with --env):
  DATA_BUCKET_NAME   the silo's own bucket (from --dataset-bucket); text/train.jsonl, text/val.jsonl live there
  NLM_BASE_S3        "bucket/prefix" of the frozen base model, downloaded once per container
  CLIENT_INDEX, RUN_ID   required / logged

Shared helpers (windows of text, model construction, candidate scoring) live here so the baselines and the grader
import the very same code.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from crosstrainer_training_client.configuration.hfl_configuration import Data, Model, TrainingOutput
from crosstrainer_training_client.configuration.hfl_pytorch_configuration import HFLPytorchConfiguration

# The framework's log handler only passes INFO from loggers inside its own namespace.
LOG = logging.getLogger("crosstrainer_training_client.user_model.ran_nlm_lora")

FAMILIES = ("CONGESTION", "MOBILITY", "NONE")
CAUSES = {
    "CONGESTION": ("PRB_EXHAUSTION", "UE_SURGE", "COVERAGE_OVERLAP"),
    "MOBILITY": ("HO_PARAM_MISTUNED", "COVERAGE_HOLE", "PINGPONG_NEIGHBOR"),
    "NONE": ("NONE",),
}
ACTIONS = {
    "CONGESTION": ("ADD_CARRIER_OR_OFFLOAD", "ENABLE_LOAD_BALANCING", "INCREASE_ELECTRICAL_TILT"),
    "MOBILITY": ("TUNE_A3_OFFSET_AND_TTT", "REVIEW_COVERAGE_AND_NEIGHBOR_PLAN", "FIX_NEIGHBOR_RELATIONS"),
    "NONE": ("NO_ACTION",),
}
LORA_R, LORA_ALPHA = 8, 16
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MAX_LEN = 360
BATCH_SIZE = 8
LEARNING_RATE = 2e-4
LOCAL_EPOCHS = 2

_BASE_DIR: str | None = None  # process-level cache: the base model is downloaded once per container


def _require_env(name: str) -> str:
    try:
        return os.environ[name]
    except KeyError:
        raise RuntimeError(f"{name} is not set (pass it with 'start-training --env {name}=...').") from None


def fetch_base_model(root: str | None = None) -> str:
    """Download the frozen base model from NLM_BASE_S3 ('bucket/prefix') into a local folder, once per process."""
    global _BASE_DIR
    if _BASE_DIR and os.path.isdir(_BASE_DIR):
        return _BASE_DIR
    import boto3

    bucket, prefix = _require_env("NLM_BASE_S3").split("/", 1)
    prefix = prefix.rstrip("/")
    dest = os.path.join(root or tempfile.mkdtemp(prefix="nlm-base-"), "base")
    os.makedirs(dest, exist_ok=True)
    s3 = boto3.client("s3")
    n = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix + "/"):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix) + 1:]
            if not rel or rel.endswith("/"):
                continue
            target = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            s3.download_file(bucket, obj["Key"], target)
            n += 1
    LOG.info("base model downloaded", extra={"bucket": bucket, "prefix": prefix, "files": n})
    _BASE_DIR = dest
    return dest


def load_tokenizer(base_dir: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_dir)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def build_peft_model(base_dir: str, device: torch.device):
    """Frozen base (bf16) + LoRA adapters (fp32). PEFT initialises lora_A randomly and lora_B to zero."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype=torch.bfloat16)
    cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0, target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    return get_peft_model(base, cfg).to(device)


def adapter_state(model) -> dict[str, np.ndarray]:
    """Only the trainable LoRA tensors, by name, as float32 arrays. This is all that is ever exchanged."""
    return {n: p.detach().float().cpu().numpy() for n, p in model.named_parameters() if p.requires_grad}


def load_adapter_state(model, state: dict[str, np.ndarray]):
    params = dict(model.named_parameters())
    unknown = [k for k in state if k not in params]
    if unknown:
        raise KeyError(f"{len(unknown)} adapter tensors do not match the model, e.g. {unknown[:2]}")
    expected = {n for n, p in params.items() if p.requires_grad}
    if expected - set(state):
        raise KeyError(f"{len(expected - set(state))} trainable tensors missing from the state, e.g. {sorted(expected - set(state))[:2]}")
    with torch.no_grad():
        for k, v in state.items():
            params[k].copy_(torch.from_numpy(np.ascontiguousarray(v)).to(device=params[k].device, dtype=params[k].dtype))
    return model


class TextDataset(Dataset):
    """Prompt + completion pairs. Loss is on the completion tokens only (prompt labels are -100)."""

    def __init__(self, rows: list[dict], tok) -> None:
        self.items = []
        for r in rows:
            p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
            c = tok(r["completion"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            ids = (p + c)[-MAX_LEN:]
            labels = ([-100] * len(p) + c)[-MAX_LEN:]
            self.items.append((ids, labels))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        return self.items[i]


def make_collate(pad_id: int):
    def collate(batch):
        longest = max(len(ids) for ids, _ in batch)
        ids = torch.full((len(batch), longest), pad_id, dtype=torch.long)
        lab = torch.full((len(batch), longest), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), longest), dtype=torch.long)
        for i, (a, b) in enumerate(batch):
            ids[i, : len(a)] = torch.tensor(a)
            lab[i, : len(b)] = torch.tensor(b)
            mask[i, : len(a)] = 1
        return {"input_ids": ids, "attention_mask": mask, "labels": lab}

    return collate


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@torch.no_grad()
def candidate_logprobs(model, tok, prompt: str, candidates: list[str]) -> list[float]:
    """Sum of token log-probabilities of each candidate continuation given the prompt (one forward pass)."""
    device = next(model.parameters()).device
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    seqs, spans = [], []
    for c in candidates:
        c_ids = tok(c, add_special_tokens=False)["input_ids"]
        seqs.append(p_ids + c_ids)
        spans.append((len(p_ids), len(p_ids) + len(c_ids)))
    longest = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), longest), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), longest), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits.float()
    logp = torch.log_softmax(logits[:, :-1], dim=-1)
    out = []
    for i, (a, b) in enumerate(spans):
        tgt = ids[i, a:b].to(device)
        out.append(float(logp[i, a - 1: b - 1].gather(-1, tgt[:, None]).sum()))
    return out


def predict_family(model, tok, prompt: str) -> str:
    scores = candidate_logprobs(model, tok, prompt, [f" FAMILY: {f}" for f in FAMILIES])
    return FAMILIES[int(np.argmax(scores))]


def predict_cause(model, tok, prompt: str, family: str) -> str:
    options = CAUSES[family]
    scores = candidate_logprobs(model, tok, prompt, [f" FAMILY: {family} | CAUSE: {c}" for c in options])
    return options[int(np.argmax(scores))]


def predict_action(model, tok, prompt: str, family: str, cause: str) -> str:
    options = ACTIONS[family]
    scores = candidate_logprobs(model, tok, prompt, [f" FAMILY: {family} | CAUSE: {cause} | ACTION: {a}" for a in options])
    return options[int(np.argmax(scores))]


# --------------------------------------------------------------------------------------------------------------
# Federated configuration
# --------------------------------------------------------------------------------------------------------------
class RanNlmLora(HFLPytorchConfiguration):
    def n_local_epochs(self) -> int:
        # default 2 (the first two Task B sessions); the fix runs pass --env NLM_LOCAL_EPOCHS=1 and --iterations 6
        return int(os.environ.get("NLM_LOCAL_EPOCHS", LOCAL_EPOCHS))

    def prepare_client(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fetch_and_prepare_data(self):
        bucket = _require_env("DATA_BUCKET_NAME")
        client_index = _require_env("CLIENT_INDEX")
        prefix = os.environ.get("RAN_TEXT_PREFIX", "text/")
        import boto3

        s3 = boto3.client("s3")
        rows = {}
        for split in ("train", "val"):
            local = os.path.join(self._tempdir.name, f"{split}.jsonl")
            s3.download_file(bucket, f"{prefix}{split}.jsonl", local)
            rows[split] = read_jsonl(local)
        self._tok = load_tokenizer(fetch_base_model())
        train = TextDataset(rows["train"], self._tok)
        val = TextDataset(rows["val"], self._tok)
        self._val_rows = rows["val"]
        fam = {}
        for r in rows["train"]:
            fam[r["family"]] = fam.get(r["family"], 0) + 1
        LOG.info("silo text ready", extra={"run_id": os.environ.get("RUN_ID", ""), "client_index": client_index,
                                           "train_examples": len(train), "val_examples": len(val), "train_by_family": fam})
        collate = make_collate(self._tok.pad_token_id)
        return (
            DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate),
            DataLoader(val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate),
            None,
        )

    def create_model(self):
        model = build_peft_model(fetch_base_model(), self._device)
        trainable = [p for p in model.parameters() if p.requires_grad]
        LOG.info("adapter model ready", extra={"trainable_tensors": len(trainable), "trainable_values": int(sum(p.numel() for p in trainable))})
        return model, torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=0.0)

    def get_state_dict(self, model):
        return adapter_state(model)

    def set_weights_from_state_dict(self, state_dict, model):
        return load_adapter_state(model, state_dict)

    def train(self, model: Model, training_data: Data, validation_data: Data | None, test_data: Data | None, optimizer: Any | None = None) -> TrainingOutput:
        if optimizer is None:
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE, weight_decay=0.0)
        device = next(model.parameters()).device
        model.train()
        for epoch in range(self.n_local_epochs()):
            running, batches = 0.0, 0
            for batch in training_data:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    loss = model(**batch).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                running += float(loss.item())
                batches += 1
            LOG.info("local epoch finished", extra={"epoch": epoch + 1, "train_loss": running / max(1, batches), "batches": batches})
        return TrainingOutput(model=model, num_samples=len(training_data.dataset))

    def evaluate(self, model: Model, training_data: Data, validation_data: Data | None, test_data: Data | None):
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()
        total, tokens = 0.0, 0
        try:
            with torch.inference_mode():
                for batch in validation_data:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        loss = model(**batch).loss
                    n = int((batch["labels"][:, 1:] != -100).sum())
                    total += float(loss.item()) * n
                    tokens += n
                correct = sum(predict_family(model, self._tok, r["prompt"]) == r["family"] for r in self._val_rows)
        finally:
            model.train(was_training)
        return {"eval_loss": total / max(1, tokens), "eval_family_acc": correct / max(1, len(self._val_rows)), "eval_examples": float(len(self._val_rows))}

    def teardown(self) -> None:
        self._tempdir.cleanup()
