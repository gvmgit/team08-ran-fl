track-2

# team08: federated learning on RAN telemetry and text (FYUZ x AWS "Breaking Barriers", Track 2)

## The problem in three lines

Two radio-network sites each hold telemetry (and the incident text built from it) that they cannot pool: Site A mostly sees congestion, Site B mostly mobility failures, and neither gets enough examples of the other's failure mode.
We train, with the Crosstrainer private-federated-learning service, (A) an early-warning model with differential privacy (epsilon target 1.0) and (B) a LoRA adapter on a telecom language model, so each site gains the other's failure mode without sharing a row or a line of text.
Every number below comes from a run declared before it started, scored on a frozen test set by a deterministic grader, with one record per scored item under one run id.

## Results at a glance

All figures are rebuilt from `evidence/` by `node tools/rebuild.js` (no AWS access). Details and caveats: `evidence/TASK_A_MAIN_REPORT.md`, `evidence/TASK_A_STRESS_REPORT.md`, `evidence/TASK_B_REPORT.md`, `GAPS.md`.

**Task A, early-warning model (two sites, DP epsilon 1.0 target, 3 rounds, secure aggregation).** Epsilon spent 0.4268 (delta 1.9e-12).
On the supplied data every model, including local-only, scores AUROC about 1.0 (the data are saturated, see GAPS.md), so there is nothing for federation to add; federated with DP is 0.998 to 0.9997.
When each site's rare failure family is withheld from its own labels (declared stress experiment), local-only collapses and federated does not:

| Test pair (rare family withheld at home) | Local-only AUROC | Central AUROC (upper bound) | Federated DP AUROC |
|---|---|---|---|
| Site A mobility | 0.231 | 0.997 | **0.999** |
| Site B congestion | 0.199 | 0.988 | **0.996** |

**Task B, federated LoRA adapter (family accuracy on 520 held-out examples, both sites pooled; no differential privacy).** Each site never logged one family at home. With the first round structure (3 rounds x 2 local epochs) the adapter was **not stable**: 0.948 in one session, 0.610 in an identical replicate. A fix that keeps the same total local training but averages twice as often (6 rounds x 1 epoch, declared before running, chosen after seeing the first test results) met the declared target in both of its sessions. Caveat: even then the global adapter flips between families in rounds 3 to 5 and settles only at round 6 (GAPS.md B1), and only two sessions were run.

| Model | Overall | Site A mobility (never logged at A) | Site B congestion (never logged at B) |
|---|---|---|---|
| Untuned base | 0.646 | 0.94 | 0.23 |
| Local-only adapters | 0.60 | 0.00 | 0.00 |
| Central, pooled text (upper bound, not allowed in the real setting) | 0.988 | 1.00 | 1.00 |
| Federated, first recipe, session 1 | 0.948 | 1.00 | 0.89 |
| Federated, first recipe, replicate | 0.610 | 0.00 | 1.00 |
| **Federated, fix recipe (6 x 1), session 1** | **0.975** | **0.97** | **1.00** |
| **Federated, fix recipe (6 x 1), session 2** | **0.979** | **0.98** | **1.00** |
## Architecture

```
 Site A silo bucket (own IAM role)           Site B silo bucket (own IAM role)
   data/*.csv, text/*.jsonl                     data/*.csv, text/*.jsonl
          |  read only by its own client               |
   SageMaker training job (GPU)                SageMaker training job (GPU)
   Crosstrainer client A                       Crosstrainer client B
          \  weights only, signed with the client's certificate  /
           v                                                    v
                Crosstrainer service (secure aggregation, DP accounting)
                         |  aggregated global model, weights only
                         v
            transfer bucket (weights, source archives, certificate bundles: no CSV, no text)
                         |
   grader (SageMaker job, deterministic, no model writes labels)
                         |
   evidence bucket: declared/<run id>.json, records/<model>__<site>...jsonl   ->  evidence/ in this repo
```

- One AWS account (the team account), us-west-2 for training, us-east-2 for CPU jobs and the evidence bucket. Silo isolation is by IAM role and bucket, not by account (see GAPS.md).
- Certificate route: P-521 client certificates issued by our own CA; keys live in Secrets Manager, never in the repo.
- Task A model: small MLP (540-128-64-2) on the last 60 minutes of one cell, two outputs (congestion, mobility). Task B model: OTel-LLM-1.7B-IT frozen, LoRA r=8 adapter (392 tensors, about 35 MB) is the only thing exchanged.

## One command per stage

PowerShell, from the repository root, signed in with `aws sso login --profile fyuz --use-device-code`. Settings and optional overrides: `aws/env.ps1` (behind a TLS-inspecting proxy also set `RAN_CA_BUNDLE` for the AWS CLI and `RAN_CORP_ROOT` for the Node utility; the utility needs a fresh `aws sso login --profile fyuz --use-device-code` when its token expires). Job folders are assembled by `aws/make_job_folder.ps1`
(it fetches the Crosstrainer wheel and verifies its published SHA-512); every reported job's config is in `aws/configs/`. Every runner takes `-Declaration` and uploads the declaration BEFORE the run starts.

| Stage | Command |
|---|---|
| 0. Certificate authority (once) | `powershell -File tools\make_certs.ps1 ca` (client certificates are issued per session by the federated runner) |
| 1. Task A model-code validation | `aws\make_job_folder.ps1 -Kind taska-validate -Dest <dir>` then `aws\run_processing.ps1 -JobName ran-validate -CodeDir <dir> -Entry validate_job.py -Env @{DATA_BUCKET_NAME='<site A silo bucket>'; RAN_DATA_KEY='data/site_a_cell_minute.csv'}` |
| 2. Task A local-only baseline + grader | `aws\make_job_folder.ps1 -Kind taska-eval -Config aws\configs\taskA_local_only.json -Dest <dir>` then `aws\run_processing.ps1 -JobName ran-local-only -CodeDir <dir> -Entry eval_job.py -InstanceType ml.m5.2xlarge -Env @{AWS_DEFAULT_REGION='us-west-2'} -Declaration evidence\templates\declare_local_only.json` |
| 3. Task A central baseline | same with `taskA_central.json`, `-JobName ran-central`, `-Declaration evidence\templates\declare_central.json` |
| 4. Task A federated, DP epsilon 1.0 | `aws\run_federated.ps1 -RunPrefix ran-fed-dp -Declaration evidence\templates\declare_federated.json` |
| 5. Task A scoring of all models (one record per event) | `taska-eval` with `taskA_rescore_main.json`, then `run_processing.ps1 -JobName ran-rescore-main ...` |
| 6. Task A stress experiment | `aws\run_federated.ps1 -RunPrefix ran-stress-fed-dp -Declaration evidence\templates\declare_stress_federated.json -WithholdA mobility -WithholdB congestion`, baselines with `taskA_stress_*.json`, scoring with `taskA_rescore_stress.json` |
| 7. Task B text from the RAN rows | `aws\make_job_folder.ps1 -Kind taskb-make-text -Config aws\configs\taskB_make_text.json -Dest <dir>` then `aws\run_processing.ps1 -JobName ran-maketext -CodeDir <dir> -Entry make_text.py` |
| 8. Task B initial adapter and validation | `-Kind taskb-initial-adapter` (entry `make_initial_adapter.py`, env `NLM_BASE_S3`), then `-Kind taskb-validate -InitialAdapter <file>` (entry `taskb_validate_job.py`, env `NLM_BASE_S3`, `DATA_BUCKET_NAME`, `CLIENT_INDEX=1`), each run with `run_processing.ps1` |
| 9. Task B baselines (untuned, local-only, central) | `aws\make_job_folder.ps1 -Kind taskb-grader -Config aws\configs\taskB_baselines.json -InitialAdapter <file> -Dest <dir>` then `aws\run_training_job.ps1 -JobName ran-nlm-baselines -CodeDir <dir> -Entry taskb_job.py -Declaration evidence\templates\declare_nlm_baselines.json` |
| 10. Task B federated session (adapter-only, no DP) | `aws\run_federated.ps1 -RunPrefix ran-nlm-fed -Declaration evidence\templates\declare_nlm_federated.json -ModelCodeDir taskb\model-code -InitialWeights <initial_adapter.safetensors> -Protocol NN_HFL -DataKeyTemplate "" -ExtraEnv @("NLM_BASE_S3=<bucket>/base-model/otel-llm-1.7b-it")` (replicate: `declare_nlm_federated_replicate.json`, prefix `ran-nlm-fed-rep2`; fix recipe, 6 rounds x 1 epoch: add `-Iterations 6` and `NLM_LOCAL_EPOCHS=1` to `-ExtraEnv`, declarations `declare_nlm_federated_r6e1.json` and `_r6e1_replicate.json`) |
| 11. Task B scoring of all adapters (one record per example) | `-Kind taskb-grader -Config aws\configs\taskB_regrade.json`, then `run_training_job.ps1 -JobName ran-nlm-regrade -Declaration evidence\templates\declare_nlm_regrade.json` |
| 12. Rebuild every reported number, no AWS | `node tools\rebuild.js` (exit 0 = evidence unchanged and every declared target met or disclosed as missed; exit 1 otherwise) |
| 13. Live test of one record, read-only | `node tools\show_record.js --run-id <run id> --item "<item>"` (examples in `judging.json` and `tools\show_record.js`) |

The Crosstrainer utility is a Linux tool; we ran it from Windows with a few local patches, listed in `tools\UTILITY_WINDOWS_PATCHES.md`.

## Evidence chain

- **Declared before the run:** `evidence/declared/<run id>.json` (target, frozen test-set hash, model id, grader), uploaded to the evidence bucket before the job starts.
- **Frozen test sets:** Task A test days 96 to 119 (hash in `test_slice_sha256.json`), Task B test text (`evidence/taskb_test/`, hash checked by the rebuild).
- **Independent grader:** `aws/eval_job.py` and `taskb/taskb_job.py` are deterministic scripts; labels come from the generator's rules, never from a model.
- **Same conditions:** every model in a comparison runs through the same code, prompt format and parser.
- **Records:** one line per scored item (Task A event, Task B test example) carrying the run id; `records_sha256` in each `results.json`; the rebuild recomputes lead-time recall (Task A) and every count (Task B) from them.
- **Live test:** `tools/show_record.js` reads one record from S3 and prints where it lives, and compares its S3 write time with the job's CloudWatch log times.

## Before Nov 1

All code in this repository was written on 1 and 2 November 2026 during the event; no earlier project code was reused.
Inputs that existed before the event and are not ours: the organisers' synthetic RAN CSVs and the proposal PDF, the OTel-LLM-1.7B-IT base model, and the Crosstrainer service, utility and training-client wheel.

## Honest limits (full list with numbers and run ids in GAPS.md)

The supplied Task A data are saturated, so the federated gain is shown only in the declared stress experiment. The Task B federated adapter was unstable with the first round structure and worked in two sessions of the fix recipe, but only at the last round (the global adapter still flips between families in the rounds before), tested on the same frozen test text whose first results we had already seen when we chose the recipe; it has no differential privacy and is trained on templated synthetic text.
Federated trails the central (pooled) baseline, which is not allowed in the real setting and is shown only as an upper bound.
