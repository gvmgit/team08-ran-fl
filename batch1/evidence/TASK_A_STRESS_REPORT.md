# Task A - stress experiment: a family the site never logged at home

**Why.** On the supplied CSVs the early-warning task is saturated (see `TASK_A_MAIN_REPORT.md`): a local-only model already
scores AUROC 1.0000, even on the family each site rarely sees and even on the other site's data, so federation has no
headroom. The proposal's headline claim is different: *a site that never saw a fault family detects it after federation*.
This experiment tests exactly that claim, on the same CSVs and the same model code.

**How.** For training only, each site's rare family is withheld as all-zero labels (Site A: `y_mobility`; Site B:
`y_congestion`), as if never logged. The telemetry is untouched. Scoring always uses the true labels on the frozen test days
(96-119). A withheld family has no labels at home to tune a threshold on, so its F1 threshold is calibrated on the other
site's validation days (same model, true labels there). AUROC needs no threshold. Switch: `RAN_WITHHOLD_LABEL` in
`model-code/model.py` (does nothing when unset); `withhold` in `aws/eval_job.py`.

| Run | Run id | What |
|---|---|---|
| local-only (non-DP) | `ran-stress-local-20261102T082611` | each site alone, rare family withheld |
| central (non-DP, baseline only) | `ran-stress-central-20261102T082613` | both sites pooled, same withholding |
| federated (DP) | `ran-stress-fed-dp-20261102T0835` (session `d5e0f36e5c6ec12db57a1`) | `NN_HFL_WITH_DP`, 3 iterations, 2 GPU clients, certificate route |

Each was declared (`evidence/declared/`) before it started. Test-set hashes matched the declaration in every job.

## Result: AUROC on the test days for the family withheld at home

| Site / family | Local-only | Central | **Federated (DP)** | Declared target |
|---|---|---|---|---|
| Site A, mobility | 0.2312 | 0.9967 | **0.9987** | local <= 0.60, central >= 0.95, federated >= 0.90 |
| Site B, congestion | 0.1990 | 0.9875 | **0.9960** | same |

A model that never saw the family at home is not merely blind to it: it scores **worse than chance**, because it learned that
this drift pattern means "no failure". Federation recovers the family completely while differential privacy is on.

| Withheld family at home | model | F1 | precision | recall | events warned >= 60 min ahead |
|---|---|---|---|---|---|
| Site A, mobility | local-only | 0.067 | | | 1.000* |
| Site A, mobility | central | 0.600 | | | 0.758 |
| Site A, mobility | **federated (DP)** | **0.879** | 0.795 | 0.984 | **1.000** |
| Site B, congestion | local-only | 0.071 | | | 1.000* |
| Site B, congestion | central | 0.626 | | | 0.724 |
| Site B, congestion | **federated (DP)** | **0.739** | 0.598 | 0.968 | **1.000** |

\* local-only "warned" values are meaningless: its threshold comes from the other site and it fires almost constantly, so
precision is near zero. The federated model's lead-time recall is also 1.000, but precision on the withheld family is only
0.79 (A) and 0.60 (B): the threshold calibrated on the other site over-alarms. Better cross-site calibration is the open
weakness, not detection.

The families each site does log stay at AUROC 0.9980 (A congestion) and 0.9985 (B mobility) under DP.

## Privacy budget

target epsilon 1.0 | noise multiplier 2.0, clip 1.0 | 8,502 real optimizer steps per client | delta 1.9004e-12 |
**epsilon spent 0.4268** per client.

## Declared targets, all met

federated AUROC >= 0.90 on both withheld pairs (0.9987, 0.9960) | lead recall >= 0.5 (1.000, 1.000) | epsilon <= 1.0 (0.4268)
| local-only <= 0.60 (0.231, 0.199) | central >= 0.95 (0.997, 0.988).

## Caveats (also in GAPS.md)

- Federated beating central here is a statement about this setup, not a general result: the central baseline is plain
  training (6 epochs, no DP) and was not tuned. They agree within 0.01 AUROC, which is the honest takeaway.
- In-client per-round loss is not a quality measure in this run: the global model correctly flags the other family, which
  the silo's withheld labels call negative, so the loss stays higher. The in-client AUROC of a withheld family reads -1.0
  (no positives). Only the offline grader, which uses the true labels, measures it.
- One seed per configuration; no confidence intervals.
