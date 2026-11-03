# DATA.md - datasets, models, lineage

Every dataset in this repository is synthetic or comes from the organisers. No real subscriber, customer or network data
is used anywhere. Account ids and bucket names appear in `judging.json` and the run records because the judges need them.

## 1. Datasets

| Dataset | What it is | Source and licence | Hash | Used for |
|---|---|---|---|---|
| `site_a_cell_minute.csv` | Synthetic RAN telemetry, Site A (30 cells x 172,800 minutes, congestion-heavy) | Provided by the hackathon organisers as the synthetic RAN data set. A public source URL and licence were not stated in the files we received; asked in `FACILITATOR_MESSAGE.md`. Treated as synthetic data for the hackathon only. | SHA-256 `F173CF19C87B971F063BFE1A8C6E90086B7094138901D36EE01EDEC978E129BA` (322,262,445 bytes) | Task A training and scoring; Task B text source |
| `site_b_cell_minute.csv` | Synthetic RAN telemetry, Site B (mobility-heavy) | Same as above | SHA-256 `3A4ECB38009F9E46871692B5DDFCAFEC770CCB37E4C5F333492DEDEBB8E19C4B` (324,141,418 bytes) | Same |
| Task A frozen test slice | Test days 96-119 of each CSV, every minute, 1,036,800 windows per site | Derived from the two CSVs by `model-code/model.py` (`TEST_DAYS`) | `test_slice_sha256.json`: A `4e96a5cb...6642420e`, B `c74c888b...34de93ede` | Task A grader |
| Task B text corpus | Alarm / counter / ticket style prompts with a structured answer, 700 train + 90 validation + 260 test examples per site | **Generated** from the real rows of the two CSVs by `taskb/make_text.py`. No language model is involved: seeded rules only (see 2). | See `evidence/taskb_test/corpus_manifest.json` and the table in 2 | Task B training and scoring |
| Base language model | OTel-LLM-1.7B-IT, a telecom model fine-tuned from Qwen/Qwen3-1.7B (Qwen3 architecture, 28 layers, hidden size 2048) | Supplied by the organisers in their bucket (copied unchanged into the team bucket). Model card: Apache 2.0, based on `Qwen/Qwen3-1.7B`, fine-tuned on the OTel-LLM data set. | Files in the team bucket under `base-model/otel-llm-1.7b-it/` (not re-hashed; see GAPS.md) | Task B (frozen; only a LoRA adapter is trained) |

The CSVs stay in their silo buckets (`data/`). The repository does not contain them. Rebuilding the Task B text needs the CSVs
and one command (see README).

## 2. Task B text: how it is generated (the generator code is the lineage)

`taskb/make_text.py`, run as SageMaker Processing job `ran-maketext-20261102T084413` (config in the job; manifest in
`evidence/taskb_test/corpus_manifest.json`).

- Each example is anchored on a real event row (`event_congestion` or `event_mobility` = 1) at cell c, minute T, or on a normal minute.
  The prompt shows the counters (UE count, SINR p10, RSRP p10, handovers per minute, weak-UE fraction) at T-100, T-60 and T-20 minutes,
  the cell tilt, the alarms those counters trigger (numeric thresholds in `ALARM_RULES`), and a ticket line.
- The answer is `FAMILY: <family> | CAUSE: <cause> | ACTION: <action>`. The cause is a deterministic rule on the counters and the tilt;
  the action follows from the cause. Labels are therefore written by rules, not by a model, so the grader is independent of any model family.
- One shared format for both sites (no site tag, no cell id in the prompt, no time of day), so a model cannot shortcut the family.
- Split by day as in Task A: train days 0-83, validation 85-94, test 96-119. Seeds: Site A 4201, Site B 4302.
- **By construction** Site A's training and validation text contains no mobility example and Site B's no congestion example.
  The test sets contain both families at both sites (100 congestion, 100 mobility, 60 normal each).

| Site | Split | Examples | By family | SHA-256 |
|---|---|---|---|---|
| A | train | 700 | 500 CONGESTION, 200 NONE | `dd73e7762f9e468bc645ebd3e3013d02c6dadbb7204c1beef011f04867de089b` |
| A | validation | 90 | 60 CONGESTION, 30 NONE | `2f913112f6ad5032bbd19abe4022dcfbf57f180d2b2b37998aa1bf3c2ee98739` |
| A | test | 260 | 100 CONGESTION, 100 MOBILITY, 60 NONE | `07ade1409500c09cad174cb0b876ce59fe10afc1b34d68d3a1e6ab99d34f5324` |
| B | train | 700 | 500 MOBILITY, 200 NONE | `d5cb540b5bcce261f6a15206323023728c9d332b23deb5fb720782acf0b4192f` |
| B | validation | 90 | 60 MOBILITY, 30 NONE | `6cb60137709e9eabd561f6da052f9ddad8e95726cde23b8bd2879f09c32c35a8` |
| B | test | 260 | 100 MOBILITY, 100 CONGESTION, 60 NONE | `4338e882075b9e174cffb152a150ab96245358eaedf092bf33afbccd2522ab6c` |

The two test files are kept in the repository (`evidence/taskb_test/`) so `node tools/rebuild.js` can check their hashes against the values
declared before any run, with no AWS access.

## 3. Model and output lineage

| Item | Hash or id | Where |
|---|---|---|
| Task A initial weights (MLP 540-128-64-2, 77,634 values) | SHA-256 `c70d744c309b6c6c6d1a4bfc5b4d5bc51f88c71c5e1921cfe0a99facc53694da` | `initial.safetensors`, made by `tools/make_initial_weights.js` |
| Task B initial adapter (LoRA r=8, 392 tensors, 8.7M values) | SHA-256 `b76ee5341b92509929d87f8b75ae0623581538bccabdbdc27ad80da385b9c45f` | made by `taskb/make_initial_adapter.py` (Processing job `ran-initial-adapter-20261102T085643`) |
| Task A federated final model | iteration-4 global object of session `0c11aa8c44f41a55a53a8`, run id `ran-fed-dp-20261102T0811` | `evidence/results/ran-rescore-main-20261102T090733/results.json` |
| Task B adapters scored (untuned, local A, local B, central, federated, federated replicate) | SHA-256 of each adapter file: section 5 | `evidence/results/ran-nlm-regrade-20261102T121931/results.json` (`adapter_files`) |
| Per-example outputs (Task A events, Task B test examples) | SHA-256 of every record file is written by the grader into `records_sha256` | `evidence/results/<run id>/records/` and `s3://<evidence bucket>/ran-fl/jobs/<run id>/output/records/` |

### Prompts

Task A has no prompt (a small neural network on counters). Task B scores a fixed prompt format; one real example, as stored in the
test text (`evidence/taskb_test/site_a_test.jsonl`, line 1), ends with:

```
Question: which failure family is developing, what is the likely cause, and what is the first action?
Answer:
```

and the model's candidate continuations are ` FAMILY: <f>`, then ` FAMILY: <f> | CAUSE: <c>`, then ` FAMILY: <f> | CAUSE: <c> | ACTION: <a>`.
The same prompt and parser are used for every model compared (untuned, local-only, central, federated). The grader ranks candidates by
log-likelihood, so no free-text generation has to be parsed. Three greedy generations per site and family are kept as qualitative samples only.

## 4. What is not covered

- The base model's own training data (OTel-LLM) is described in its model card; we did not audit it.
- We did not re-hash the base model files. They were copied byte for byte by `aws s3 cp` from the organisers' bucket.
- The public source and licence of the two CSVs are unknown to us (see above).

## 5. Task B adapters and per-example outputs (written by the grader of `ran-nlm-regrade-20261102T121931`)

Every adapter has 392 `lora_*` tensors. Prompt, parser and test text are the same for all six; the test text hashes are in section 2.

| Adapter | Origin | SHA-256 of the file scored |
|---|---|---|
| untuned_base | initial adapter (`make_initial_adapter.py`) | `b76ee5341b92509929d87f8b75ae0623581538bccabdbdc27ad80da385b9c45f` |
| local_a | baselines job `ran-nlm-baselines-20261102T090507` | `6911b92fb8a287400fedd10f38f40c6eadad7bac203c75306c2160802ab0590f` |
| local_b | baselines job `ran-nlm-baselines-20261102T090507` | `7283189ed36ef86a28fc75f7da294d56b4a1a091bd26820fc128847e440d3d0b` |
| central | baselines job `ran-nlm-baselines-20261102T090507` | `6480ab773830d244407a138c6016e0c836f0d0c5f2253f478ea6286755363b07` |
| federated | session `ran-nlm-fed-20261102T0911`, iteration 4 global adapter | `855a2a8bbf385fa859658c7d319357ecbb66b8c2bb20fc2f06989beba65d3365` |
| federated_rep2 | session `ran-nlm-fed-rep2-20261102T1149`, iteration 4 global adapter | `b4bc315affd7cda436594537f27585425e782cb354a6332909f2d965cd5274ba` |

| Record file (one JSON line per test example) | SHA-256 |
|---|---|
| `records/untuned_base__a.jsonl` | `afd6bf4dcaf24f07dc6598da7da6bee41d9e4376a5d72c561c0db79a7d2ec4cb` |
| `records/untuned_base__b.jsonl` | `5708c17b8ae9239c8567e6e6ce380c6dd9b8d8f77ae6d947fc1459ebcb7ff336` |
| `records/local_a__a.jsonl` | `58995092032a9e96be1c8fb1d86d74232d4a584da978bca1f8ada52bb9ee9e6d` |
| `records/local_a__b.jsonl` | `729eba19897c19855929ca3f01d707471589a56fa012fa61c2ceb30de62c1cea` |
| `records/local_b__a.jsonl` | `d56fcdcfe05ba412af33fc6a93f99a18b44f8086c95d12d8ddb4113b7e4d2c9e` |
| `records/local_b__b.jsonl` | `20430ab16b6c401650b527295f7e4b3265e5d1b4933af5e6f42308a8ae411aa5` |
| `records/central__a.jsonl` | `b8a177ee225a052a70b14ce55bcf3c4658891284d3f7007c0f4f2b87710774bd` |
| `records/central__b.jsonl` | `babcda436d6a647388d244d4a9cf25d88ea8bb9c0baa0e1595b2a374daed6101` |
| `records/federated__a.jsonl` | `933b803f146a925e5a5a5a802095d5982d63858b6f8b9dd18d88595b9a453078` |
| `records/federated__b.jsonl` | `9c60ad44a8c323c82337c934b2f0246afa9b302b1e1773e8187e5268ad691cab` |
| `records/federated_rep2__a.jsonl` | `f3a53fa78bd27c81421eb8359e02372fdbdb0ffd77c2072c2c6136c77999b632` |
| `records/federated_rep2__b.jsonl` | `fdd6dbe6d60a3b72bff66dde7f430ea2bbfb8e96b7fb94b8b9ae153ccdbaf5bf` |

The record files are in `evidence/results/ran-nlm-regrade-20261102T121931/records/` and in the evidence bucket under `ran-fl/jobs/ran-nlm-regrade-20261102T121931/output/records/`. `node tools/rebuild.js` checks every hash and recomputes every count from them.
