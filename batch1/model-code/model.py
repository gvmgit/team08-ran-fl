"""Federated early-warning model for two synthetic RAN sites (Task A).

One HFLPytorchConfiguration subclass, identical in every silo. Each client reads its own CSV from its own
silo bucket (bucket from DATA_BUCKET_NAME, which the runner sets from --dataset-bucket; key from RAN_DATA_KEY,
passed with --env). The network reads the last 60 minutes of one cell and predicts two early-warning labels:
y_congestion and y_mobility (an event within the next 120 minutes).

Everything shared with the grader and the baselines (layout, split rule, scaling, windows, network, metrics) lives
in this file, so every comparison runs the same code.

Frozen split rule (by day, the same in every run):
    train 0-83   validation 85-94   test 96-119   (days 84 and 95 are dropped as gaps)
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from crosstrainer_training_client.configuration.hfl_configuration import Data, Model, TrainingOutput
from crosstrainer_training_client.configuration.hfl_pytorch_configuration import HFLPytorchConfiguration

# The framework's log handler only passes INFO from loggers inside its own namespace; anything else needs WARNING.
LOG = logging.getLogger("crosstrainer_training_client.user_model.ran_early_warning")

# --------------------------------------------------------------------------------------------------------------
# Frozen data layout and split rule
# --------------------------------------------------------------------------------------------------------------
N_CELLS = 30
DAYS = 120
MIN_PER_DAY = 1440
T = DAYS * MIN_PER_DAY  # minutes per cell
HORIZON = 120
WINDOW = 60

FEATURES = [
    "ue_count", "ue_share", "rsrp_p10_dbm", "sinr_p10_db", "ho_count",
    "pingpong_count", "rlf_count", "weak_ue_frac", "cell_el_deg",
]
LABELS = ["y_congestion", "y_mobility"]
N_FEAT = len(FEATURES)

TRAIN_DAYS = (0, 83)
VAL_DAYS = (85, 94)
TEST_DAYS = (96, 119)
TRAIN_STRIDE = 5
VAL_STRIDE = 5

# Fixed scaling constants (no per-silo statistics are computed or shared): x_scaled = (x - offset) / scale.
_OFFSET = np.array([50.0, 1.0 / 30.0, -95.0, 8.0, 3.0, 0.0, 0.0, 0.07, 6.0], dtype=np.float32)
_SCALE = np.array([30.0, 0.017, 8.0, 4.0, 4.0, 3.0, 1.5, 0.05, 3.0], dtype=np.float32)

# Process-level cache so later FL iterations do not download and parse the CSV again.
_CACHE: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}


# --------------------------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------------------------
def load_site_csv(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Parse one site CSV into scaled features (cells, T, F) and labels (cells, T, 2; NaN = unlabeled tail)."""
    import pandas as pd

    cols = ["day", "minute", "cell_id"] + FEATURES + LABELS
    dtypes: dict[str, Any] = {c: np.float32 for c in FEATURES + LABELS}
    dtypes.update({"day": np.int32, "minute": np.int32, "cell_id": np.int32})
    df = pd.read_csv(path, usecols=cols, dtype=dtypes)
    if len(df) != N_CELLS * T:
        raise ValueError(f"expected {N_CELLS * T} rows, found {len(df)}")
    absolute = df["day"].to_numpy(np.int64) * MIN_PER_DAY + df["minute"].to_numpy(np.int64)
    if not np.array_equal(absolute, np.tile(np.arange(T, dtype=np.int64), N_CELLS)):
        raise ValueError("rows are not ordered cell-major and time-ascending as the layout requires")
    if not np.array_equal(df["cell_id"].to_numpy(np.int64), np.repeat(np.arange(N_CELLS, dtype=np.int64), T)):
        raise ValueError("cell_id column does not match the expected layout")
    x = (df[FEATURES].to_numpy(np.float32) - _OFFSET) / _SCALE
    y = df[LABELS].to_numpy(np.float32)
    return torch.from_numpy(x.reshape(N_CELLS, T, N_FEAT)), torch.from_numpy(y.reshape(N_CELLS, T, 2))


def make_anchors(y: torch.Tensor, days: tuple[int, int], stride: int) -> torch.Tensor:
    """Flat indices (cell * T + t) of every labeled anchor minute inside the day range, at the given stride."""
    lo, hi = days[0] * MIN_PER_DAY, (days[1] + 1) * MIN_PER_DAY
    t = np.arange(lo, hi, stride, dtype=np.int64)
    t = t[t >= WINDOW - 1]
    t_t = torch.as_tensor(t)
    labeled = ~torch.isnan(y[:, t_t, 0])
    cells, idx = torch.nonzero(labeled, as_tuple=True)
    return cells.to(torch.int64) * T + t_t[idx]


class WindowDataset(Dataset):
    """Map-style dataset: anchor index -> (last 60 minutes of that cell flattened to 540 values, 2 labels)."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, anchors: torch.Tensor) -> None:
        self.x = x.reshape(-1, N_FEAT)
        self.y = y.reshape(-1, 2)
        self.anchors = anchors

    def __len__(self) -> int:
        return int(self.anchors.numel())

    def __getitem__(self, i: int):
        a = int(self.anchors[i])
        return self.x[a - (WINDOW - 1): a + 1].reshape(-1), self.y[a]


def build_net() -> nn.Sequential:
    """Small MLP. No BatchNorm, so it stays compatible with Opacus. state_dict keys: 0/2/4 .weight/.bias."""
    return nn.Sequential(
        nn.Linear(WINDOW * N_FEAT, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 2),
    )


def auroc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUROC with average ranks for ties. Returns -1.0 when a class is missing."""
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(score, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return -1.0
    order = np.argsort(s, kind="mergesort")
    uniq, inv, counts = np.unique(s[order], return_inverse=True, return_counts=True)
    avg_rank = np.cumsum(counts) - (counts - 1) / 2.0
    ranks = np.empty(s.size, dtype=np.float64)
    ranks[order] = avg_rank[inv]
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _require_env(name: str) -> str:
    try:
        return os.environ[name]
    except KeyError:
        raise RuntimeError(
            f"{name} is not set. Pass it with 'start-training --env {name}=<value>' "
            "(DATA_BUCKET_NAME comes from --dataset-bucket). Defaulting could make two clients read the same silo."
        ) from None


# --------------------------------------------------------------------------------------------------------------
# Federated configuration
# --------------------------------------------------------------------------------------------------------------
class RanEarlyWarning(HFLPytorchConfiguration):
    batch_size = 512
    learning_rate = 1e-3

    def n_local_epochs(self) -> int:
        return 2

    def privacy_budget(self) -> float:
        # Target epsilon for NN_HFL_WITH_DP. Ignored by NN_HFL.
        return 1.0

    def prepare_client(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fetch_and_prepare_data(self):
        bucket = _require_env("DATA_BUCKET_NAME")
        key = _require_env("RAN_DATA_KEY")
        client_index = _require_env("CLIENT_INDEX")
        cache_key = (bucket, key)
        if cache_key not in _CACHE:
            import boto3

            local = os.path.join(self._tempdir.name, os.path.basename(key))
            LOG.info("downloading silo data", extra={"bucket": bucket, "key": key, "client_index": client_index})
            boto3.client("s3").download_file(bucket, key, local)
            _CACHE[cache_key] = load_site_csv(local)
            os.remove(local)
        x, y = _CACHE[cache_key]
        # Stress experiment only (unset in the main run, where this does nothing): the site "never logged" one
        # failure family, so that family's labels are withheld as all zeros. Telemetry is untouched.
        withheld = os.environ.get("RAN_WITHHOLD_LABEL", "").strip().lower()
        if withheld:
            if withheld not in ("congestion", "mobility"):
                raise RuntimeError(f"RAN_WITHHOLD_LABEL must be 'congestion' or 'mobility', got {withheld!r}")
            k = LABELS.index(f"y_{withheld}")
            y = y.clone()
            col = y[:, :, k]
            y[:, :, k] = torch.where(torch.isnan(col), col, torch.zeros_like(col))
        train = WindowDataset(x, y, make_anchors(y, TRAIN_DAYS, TRAIN_STRIDE))
        val = WindowDataset(x, y, make_anchors(y, VAL_DAYS, VAL_STRIDE))
        LOG.info(
            "silo data ready",
            extra={
                "run_id": os.environ.get("RUN_ID", ""),
                "withheld_label_at_home": withheld or "none",
                "client_index": client_index, "train_windows": len(train), "val_windows": len(val),
                "train_rate_congestion": float(np.nanmean(y[:, : (TRAIN_DAYS[1] + 1) * MIN_PER_DAY, 0].numpy())),
                "train_rate_mobility": float(np.nanmean(y[:, : (TRAIN_DAYS[1] + 1) * MIN_PER_DAY, 1].numpy())),
            },
        )
        # The test days are never touched inside the client; the grader scores them once, offline.
        return (
            DataLoader(train, batch_size=self.batch_size, shuffle=True),
            DataLoader(val, batch_size=4096, shuffle=False),
            None,
        )

    def create_model(self):
        model = build_net().to(self._device)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        return model, optimizer

    def train(
        self,
        model: Model,
        training_data: Data,
        validation_data: Data | None,
        test_data: Data | None,
        optimizer: Any | None = None,
    ) -> TrainingOutput:
        if optimizer is None:
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        device = next(model.parameters()).device
        criterion = nn.BCEWithLogitsLoss()
        model.train()
        for epoch in range(self.n_local_epochs()):
            running, batches = 0.0, 0
            for x, y in training_data:
                if x.shape[0] == 0:  # Poisson sampling under DP can yield an empty batch
                    continue
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                running += float(loss.item())
                batches += 1
            LOG.info("local epoch finished", extra={"epoch": epoch + 1, "train_loss": running / max(1, batches), "batches": batches})
        return TrainingOutput(model=model, num_samples=len(training_data.dataset))

    def evaluate(
        self,
        model: Model,
        training_data: Data,
        validation_data: Data | None,
        test_data: Data | None,
    ):
        data = validation_data
        device = next(model.parameters()).device
        # Opacus refuses a model that is not in training mode, and the framework evaluates before it wraps the
        # model, so evaluation must leave the mode exactly as it found it.
        was_training = model.training
        model.eval()
        total, count = 0.0, 0
        probs, labels = [], []
        criterion = nn.BCEWithLogitsLoss(reduction="sum")
        try:
            with torch.inference_mode():
                for x, y in data:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    total += float(criterion(logits, y).item())
                    count += int(y.numel())
                    probs.append(torch.sigmoid(logits).cpu())
                    labels.append(y.cpu())
        finally:
            model.train(was_training)
        p = torch.cat(probs).numpy()
        yy = torch.cat(labels).numpy()
        return {
            "eval_loss": total / max(1, count),
            "eval_auroc_congestion": auroc(yy[:, 0], p[:, 0]),
            "eval_auroc_mobility": auroc(yy[:, 1], p[:, 1]),
            "eval_windows": float(len(yy)),
        }

    # ----------------------------------------------------------------------------------------------------------
    # Differential privacy: honest accounting
    # ----------------------------------------------------------------------------------------------------------
    def _compute_privacy_budget(
        self,
        n_training_samples: int,
        gaussian_std_dev: float,
        clipping_bound: float,
        batch_size: int,
    ) -> float:
        """Epsilon of the DP-SGD run that will actually execute.

        The base class counts each local EPOCH as one optimizer step, while Opacus takes ceil(n / batch_size) steps per
        epoch. This override counts the real steps (sampling rate q = 1 / steps_per_epoch, steps = local_epochs *
        global_iterations * steps_per_epoch), with delta = n^-2 as in the base class, so the figure compared with
        privacy_budget() is the epsilon of the run that actually executes.

        Measured on this task (eval_job.py, "epsilon" section): both accountants end up choosing the same noise
        multiplier (2.0, clip 1.0) because the search is coarse and the RDP orders floor epsilon near 0.4, so the
        override changes the reported figure (0.43 vs 0.38) but not the noise that was applied.
        """
        from opacus.accountants import rdp as rdp_accountant
        from opacus.accountants.analysis import rdp as rdp_analysis

        noise_multiplier = gaussian_std_dev / clipping_bound
        steps_per_epoch = math.ceil(n_training_samples / batch_size)
        steps = self.n_local_epochs() * int(self.n_global_fl_iterations) * steps_per_epoch
        delta = float(n_training_samples) ** -2
        orders = rdp_accountant.RDPAccountant.DEFAULT_ALPHAS
        rdp = rdp_analysis.compute_rdp(q=1.0 / steps_per_epoch, noise_multiplier=noise_multiplier, steps=steps, orders=orders)
        epsilon, _ = rdp_analysis.get_privacy_spent(orders=orders, rdp=rdp, delta=delta)
        return float(epsilon)

    def _configure_differential_privacy(self, training_data: Any, model: Model, optimizer: Any):
        new_training_data, new_model, new_optimizer = super()._configure_differential_privacy(training_data, model, optimizer)
        n = len(training_data.dataset)
        LOG.info(
            "differential privacy configured",
            extra={
                "target_epsilon": self.privacy_budget(),
                "delta": float(n) ** -2,
                "noise_multiplier": float(new_optimizer.noise_multiplier),
                "max_grad_norm": float(new_optimizer.max_grad_norm),
                "n_training_samples": n,
                "batch_size": training_data.batch_size,
                "local_epochs": self.n_local_epochs(),
                "global_iterations": self.n_global_fl_iterations,
                "steps_per_epoch": math.ceil(n / training_data.batch_size),
            },
        )
        return new_training_data, new_model, new_optimizer

    def teardown(self) -> None:
        self._tempdir.cleanup()
