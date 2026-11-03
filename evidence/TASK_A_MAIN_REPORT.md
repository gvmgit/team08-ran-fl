# Task A - main federated run (NN_HFL_WITH_DP)

| Item | Value |
|---|---|
| Run id | `ran-fed-dp-20261102T0811` (declared in `evidence/declared/` before the session was created) |
| Crosstrainer session | `0c11aa8c44f41a55a53a8`, status COMPLETED (stop condition: 3 rounds) |
| Protocol | `NN_HFL_WITH_DP`, 3 iterations, `--required-clients 2`, certificate route (P-521 client certificates, own CA as trust anchor) |
| Clients | `ran-client-a` (Site A, CLIENT_INDEX=1, ml.g6.2xlarge), `ran-client-b` (Site B, CLIENT_INDEX=2, ml.g5.2xlarge), GPU image `pytorch-training:2.8.0-gpu-py312-cu129`, volume 100 GB |
| Data | each client reads only its own silo bucket (`--dataset-bucket`); the transfer bucket holds weights, source archives and certificate bundles only (no CSV) |
| Model | MLP 540-128-64-2 on the last 60 minutes of one cell, two outputs (`y_congestion`, `y_mobility`) |
| Split (frozen) | train days 0-83, validation 85-94, test 96-119, gap days 84 and 95 |
| Test-set hash | site A `4e96a5cb...6642420e`, site B `c74c888b...34de93ede` (match the declaration) |
| Scored model | highest iteration (4): `...-4-global.safetensor`; both clients' copies hold identical tensors (77,634 values, 0 differences) |

## Per-round evaluation inside each client (validation days 85-94, 86,400 windows)

| Silo | Model | Round 1 | Round 2 | Round 3 |
|---|---|---|---|---|
| A | global (incoming) eval_loss | 0.69627 | 0.03606 | 0.02738 |
| A | local (after local DP training) eval_loss | 0.02763 | 0.02409 | 0.02084 |
| B | global (incoming) eval_loss | 0.68680 | 0.02944 | 0.02322 |
| B | local (after local DP training) eval_loss | 0.02964 | 0.02494 | 0.02253 |

Global evaluation loss fell every round on both silos. Round 1's global model is the initial weights.

## Final scoring on the frozen test days (every minute, 1,036,800 windows per site)

| Model | Site | Label | AUROC | F1 | warned >= 60 min ahead |
|---|---|---|---|---|---|
| federated (DP) | A | congestion | 0.99803 | 0.9881 | 1.000 (1668 events) |
| federated (DP) | A | mobility | 0.99973 | 0.9903 | 1.000 (297) |
| federated (DP) | B | congestion | 0.99960 | 0.9865 | 1.000 (315) |
| federated (DP) | B | mobility | 0.99829 | 0.9903 | 1.000 (1662) |
| local-only (non-DP) | both | both | 1.0000 | 0.9988-1.0000 | 1.000 |
| central (non-DP, baseline only) | both | both | 1.0000 | 0.9999-1.0000 | 1.000 |

**Finding.** On the supplied data the task is saturated: local-only models already reach AUROC 1.0000 on both sites,
including the family each site rarely sees and including the other site's data, so federation has no headroom to show a
gain. The cost of differential privacy is about 0.002 AUROC. See `TASK_A_STRESS_REPORT.md` for the experiment that does
show a federation effect (rare family withheld at home).

## Privacy budget (from `aws/eval_job.py`, code-derived, one value per silo)

target epsilon 1.0 | delta 1.9004e-12 (n^-2, n = 725,400 windows) | batch 512 | 1,417 steps per epoch | 8,502 real optimizer steps
| noise multiplier 2.0, clip 1.0 | **epsilon spent 0.4268** (RDP accountant over the real steps). The framework's own accountant
(each local epoch counted as one step) picks the same noise multiplier, so the override did not change the noise applied.

## Checks

| Check | Result |
|---|---|
| Session completed, both clients COMPLETED (exit code 0) | pass |
| Both silos logged a drop in global eval loss each round | pass (A 0.696 -> 0.036 -> 0.027, B 0.687 -> 0.029 -> 0.023) |
| `_global` objects present; highest iteration is the scored model | pass (iteration 4 = N+1 for 3 rounds) |
| Per-silo AUROC and F1 for both labels and the 60-minute lead recall | pass (table above) |
| Site A federated mobility and Site B federated congestion next to local-only and central | pass (0.99973 and 0.99960 vs 1.0000) |
| Transfer bucket lists weights, source archives and certificate bundles only, no CSV | pass |
| Epsilon recorded with the run id | pass (target 1.0, spent 0.4268) |

## Gaps in this run's evidence (also in GAPS.md)

- The clients' own log lines (DP settings, run id) were dropped by the framework's log filter in this run, because my
  logger sat outside its namespace; fixed afterwards. The DP parameters above are recomputed deterministically from the
  same code and data by the grader, not read from the clients' logs.
- Model code at run time differed from the repository copy only by the later logger name and the no-op stress switch.
