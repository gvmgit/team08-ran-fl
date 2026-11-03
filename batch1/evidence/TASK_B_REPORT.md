# Task B - federated LoRA adapter on RAN incident text (NN_HFL, adapter-only exchange)

## What was run

| Item | Value |
|---|---|
| Text corpus | Generated from the real rows of each site's CSV by `taskb/make_text.py` (seeded rules, no language model), job `ran-maketext-20261102T084413`. 700 train + 90 validation + 260 test examples per site. Site A's train and validation text has no mobility example; Site B's has no congestion example. Hashes: `DATA.md`. |
| Base model | OTel-LLM-1.7B-IT (Qwen3 architecture), frozen. Trainable: LoRA r=8, alpha 16 on q,k,v,o,gate,up,down; 392 tensors, 8,716,288 values (about 35 MB). AdamW, learning rate 2e-4, batch 8, bf16 autocast. |
| Federated session 1 | run id `ran-nlm-fed-20261102T0911`, session `6a8e8b392c740247af8d8`, `NN_HFL`, 3 iterations x 2 local epochs, `--required-clients 2`, certificate route. Clients `ran-client-a` (ml.g6.2xlarge) and `ran-client-b` (ml.g5.2xlarge), GPU image `pytorch-training:2.8.0-gpu-py312-cu129`, volume 100 GB. Both COMPLETED. |
| Federated session 2 (replicate) | run id `ran-nlm-fed-rep2-20261102T1149`, session `a52098ddf059850d1efca`. Same code, data, initial adapter, instance types and settings, declared as a replicate before it started (`evidence/declared/ran-nlm-fed-rep2-20261102T1149.json`). Both clients COMPLETED. |
| Fix recipe sessions (6 rounds x 1 local epoch) | Added after the replicate showed the instability; the ONLY change is the round structure (`NLM_LOCAL_EPOCHS=1`, `--iterations 6`: the same 6 local epochs in total, averaged twice as often). Declared before running, with the statement that it was chosen after seeing the first recipe's test results (`evidence/declared/ran-nlm-fed-r6e1a-20261102T1350.json`, `...r6e1b-20261102T1430.json`). Session 1: `ran-nlm-fed-r6e1a-20261102T1350` (session `ec75776baea8e4ad51021`); session 2, the declared replicate: `ran-nlm-fed-r6e1b-20261102T1430` (session `6e0290860decda618358a`; its client A waited 40 minutes for ml.g6.2xlarge and was restarted on ml.g5.4xlarge, GAPS C16). Both sessions COMPLETED; the two clients' final tensors are identical in each (8,716,288 values, 0 differences). Scored in `ran-nlm-regrade-r6e1a-20261102T144016` and `ran-nlm-regrade-r6e1b-20261102T170637` (all six global adapters of each session, one record per test example). |
| Exchanged state | 392 `lora_*` tensors only (adapter-only `get_state_dict` / `set_weights_from_state_dict`). No base weights, no text. Differential privacy is **not** applied to Task B. |
| Scored model | The highest-iteration global adapter (iteration 4 for 3 rounds). Both clients' copies hold identical tensors: 8,716,288 values, 0 differences (replicate checked; the files differ only in a header field naming the client). |
| Grader | `taskb/taskb_job.py`: for every test example, log-likelihood ranks the 3 candidate FAMILIES, then the 3 CAUSES within the true family, then the 3 ACTIONS within the true family and cause. Deterministic; no LLM judge; labels come from the generator's rules. |
| Baselines | Job `ran-nlm-baselines-20261102T090507` (g5.4xlarge): untuned base (initial adapter), local-only adapters (6 epochs each, same total local epochs as 3 x 2), central adapter (both sites' text pooled, baseline only). Same code, prompt, parser and test text for every model. |
| Run of record for per-example evidence | Job `ran-nlm-regrade-20261102T121931` (declared before it ran: `evidence/declared/ran-nlm-regrade-20261102T121931.json`). It scored all six adapters and wrote one record per test example and site; the job itself copied the records to `s3://<evidence bucket>/ran-fl/jobs/ran-nlm-regrade-20261102T121931/output/records/` (live test: `node tools/show_record.js --run-id ran-nlm-regrade-20261102T121931 --item "model=federated,site=a,family=MOBILITY"`). Its record file write times lie between the job's first and last CloudWatch events. The numbers below are from these records. |
| Other scoring jobs | `ran-nlm-trajectory1-20261102T133915` scored the global adapters after aggregation 1 and 2 of the first recipe's two sessions (a diagnostic with no declared pass/fail target). Scoring attempts that waited more than 30 minutes for a GPU were stopped and are listed in GAPS C17. |

## In-client evaluation per round (CloudWatch, each client's own validation text)

Site A validation text is 60 congestion + 30 normal; Site B's is 60 mobility + 30 normal. "Global" is the incoming aggregated adapter before local training, "local" the same client after its local epochs. Iteration 1's global adapter is the initial (zero-effect) adapter, so it is the base model. The final aggregation (iteration 4) is not evaluated in-client; it is graded offline below.

| Silo | Metric | Iteration 1 | Iteration 2 | Iteration 3 |
|---|---|---|---|---|
| A, session 1 | global eval loss / family acc | 3.709 / 0.489 | 0.192 / 0.333 | 0.233 / 0.533 |
| A, session 1 | local eval loss / family acc | 0.0184 / 0.978 | 0.0052 / 0.978 | 0.0064 / 0.978 |
| B, session 1 | global eval loss / family acc | 3.896 / 0.878 | 0.371 / 0.333 | 0.058 / 0.978 |
| B, session 1 | local eval loss / family acc | 0.0281 / 0.978 | 0.0296 / 0.978 | 0.0176 / 0.978 |
| A, replicate | global eval loss / family acc | 3.709 / 0.489 | 0.182 / 0.333 | 0.052 / 0.422 |
| A, replicate | local eval loss / family acc | 0.0215 / 0.978 | 0.0047 / 0.978 | 0.0059 / 0.978 |
| B, replicate | global eval loss / family acc | 3.896 / 0.878 | 0.283 / 0.333 | 0.042 / 0.978 |
| B, replicate | local eval loss / family acc | 0.0322 / 0.978 | 0.0270 / 0.978 | 0.0283 / 0.978 |

How to read it: a client's local adapter fits its own two classes at once (0.978). The first aggregation of two adapters trained in opposite directions lands on a model that answers "none" for everything (0.333 on both sites' validation text, in both sessions); by the third iteration the global adapter is good on mobility text (0.978 at Site B) but still weak on congestion text (0.53 and 0.42 at Site A) before its last round. Global eval loss falls from 3.709 (Site A) and 3.896 (Site B) at iteration 1 to 0.233 and 0.058 at iteration 3 in session 1, and to 0.052 and 0.042 in the replicate. Family accuracy of the global adapter is not monotone (0.333 at iteration 2), so the in-client numbers are a progress signal, not the result; the result is the offline grading below. Congestion is the slower family to recover, in both sessions, which matches the offline result.

## Results (frozen test text, 260 examples per site; family accuracy = the model's most likely FAMILY equals the true one)

Pooled over both sites (520 examples), with 95% Wilson intervals:

| Model | Overall family accuracy | 95% interval |
|---|---|---|
| Untuned base | 0.646 (336/520) | 0.604-0.686 |
| Local-only, trained at Site A | 0.606 (315/520) | 0.563-0.647 |
| Local-only, trained at Site B | 0.602 (313/520) | 0.559-0.643 |
| Central (pooled text; baseline only, not allowed in the real setting) | 0.988 (514/520) | 0.975-0.995 |
| **Federated, first recipe, session 1** (`ran-nlm-fed-20261102T0911`) | 0.948 (493/520) | 0.926-0.964 |
| **Federated, first recipe, replicate** (`ran-nlm-fed-rep2-20261102T1149`) | 0.610 (317/520) | 0.567-0.651 |
| **Federated, fix recipe, session 1** (`ran-nlm-fed-r6e1a-20261102T1350`) | 0.975 (507/520) | 0.958-0.985 |
| **Federated, fix recipe, session 2** (`ran-nlm-fed-r6e1b-20261102T1430`) | 0.979 (509/520) | 0.963-0.988 |

Site A test (Site A never logged mobility; its text is congestion + normal):

| Model | Overall | Congestion (100) | Mobility (100) | Normal (60) | Cause acc, cong. / mob. | Action acc, cong. / mob. |
|---|---|---|---|---|---|---|
| Untuned base | 0.665 | 0.220 (22) | 0.940 (94) | 0.950 (57) | 0.17 / 0.14 | 0.34 / 0.52 |
| Local-only, trained at Site A | 0.604 | 1.000 (100) | 0.000 (0) | 0.950 (57) | 1.00 / 0.14 | 1.00 / 0.52 |
| Local-only, trained at Site B | 0.596 | 0.000 (0) | 0.970 (97) | 0.967 (58) | 0.62 / 0.93 | 0.34 / 1.00 |
| Central (pooled text; baseline only, not allowed in the real setting) | 0.985 | 1.000 (100) | 1.000 (100) | 0.933 (56) | 1.00 / 1.00 | 1.00 / 1.00 |
| **Federated, first recipe, session 1** (`ran-nlm-fed-20261102T0911`) | 0.950 | 0.900 (90) | 1.000 (100) | 0.950 (57) | 0.95 / 0.65 | 1.00 / 1.00 |
| **Federated, first recipe, replicate** (`ran-nlm-fed-rep2-20261102T1149`) | 0.604 | 1.000 (100) | 0.000 (0) | 0.950 (57) | 0.51 / 0.68 | 1.00 / 1.00 |
| **Federated, fix recipe, session 1** (`ran-nlm-fed-r6e1a-20261102T1350`) | 0.973 | 1.000 (100) | 0.970 (97) | 0.933 (56) | 0.51 / 0.87 | 1.00 / 1.00 |
| **Federated, fix recipe, session 2** (`ran-nlm-fed-r6e1b-20261102T1430`) | 0.977 | 1.000 (100) | 0.980 (98) | 0.933 (56) | 1.00 / 0.91 | 1.00 / 1.00 |

Site B test (Site B never logged congestion; its text is mobility + normal):

| Model | Overall | Congestion (100) | Mobility (100) | Normal (60) | Cause acc, cong. / mob. | Action acc, cong. / mob. |
|---|---|---|---|---|---|---|
| Untuned base | 0.627 | 0.230 (23) | 0.920 (92) | 0.800 (48) | 0.59 / 0.23 | 0.28 / 0.46 |
| Local-only, trained at Site A | 0.608 | 1.000 (100) | 0.000 (0) | 0.967 (58) | 0.97 / 0.23 | 1.00 / 0.46 |
| Local-only, trained at Site B | 0.608 | 0.000 (0) | 0.990 (99) | 0.983 (59) | 0.24 / 0.88 | 0.28 / 1.00 |
| Central (pooled text; baseline only, not allowed in the real setting) | 0.992 | 1.000 (100) | 1.000 (100) | 0.967 (58) | 1.00 / 1.00 | 1.00 / 1.00 |
| **Federated, first recipe, session 1** (`ran-nlm-fed-20261102T0911`) | 0.946 | 0.890 (89) | 1.000 (100) | 0.950 (57) | 0.81 / 0.58 | 1.00 / 1.00 |
| **Federated, first recipe, replicate** (`ran-nlm-fed-rep2-20261102T1149`) | 0.615 | 1.000 (100) | 0.010 (1) | 0.983 (59) | 0.87 / 0.66 | 1.00 / 1.00 |
| **Federated, fix recipe, session 1** (`ran-nlm-fed-r6e1a-20261102T1350`) | 0.977 | 1.000 (100) | 0.950 (95) | 0.983 (59) | 0.87 / 0.76 | 1.00 / 1.00 |
| **Federated, fix recipe, session 2** (`ran-nlm-fed-r6e1b-20261102T1430`) | 0.981 | 1.000 (100) | 0.980 (98) | 0.950 (57) | 0.91 / 0.82 | 1.00 / 1.00 |

The family each site never logged at home (the declared headline) is mobility at Site A and congestion at Site B. Cause accuracy is measured given the true family, action accuracy given the true cause; the action follows deterministically from the cause in the generator, so action accuracy is easy and carries little information.

### How the global adapter evolves, round by round (family accuracy by true family, Site A test / Site B test)

First recipe, session 1 (3 rounds x 2 epochs):

| Global adapter after | Congestion (A / B) | Mobility (A / B) | Normal (A / B) |
|---|---|---|---|
| aggregation 1 | 0.00 / 0.00 | 0.00 / 0.00 | 1.00 / 1.00 |
| aggregation 2 | 0.25 / 0.34 | 1.00 / 1.00 | 0.97 / 0.95 |
| aggregation 3 (final) | 0.90 / 0.89 | 1.00 / 1.00 | 0.95 / 0.95 |

First recipe, replicate:

| Global adapter after | Congestion (A / B) | Mobility (A / B) | Normal (A / B) |
|---|---|---|---|
| aggregation 1 | 0.00 / 0.00 | 0.00 / 0.00 | 1.00 / 1.00 |
| aggregation 2 | 0.10 / 0.07 | 1.00 / 1.00 | 0.95 / 0.95 |
| aggregation 3 (final) | 1.00 / 1.00 | 0.00 / 0.01 | 0.95 / 0.98 |

Fix recipe, session 1 (6 rounds x 1 epoch):

| Global adapter after | Congestion (A / B) | Mobility (A / B) | Normal (A / B) |
|---|---|---|---|
| aggregation 1 | 0.00 / 0.00 | 0.00 / 0.00 | 1.00 / 1.00 |
| aggregation 2 | 0.06 / 0.26 | 0.12 / 0.08 | 0.98 / 1.00 |
| aggregation 3 | 0.00 / 0.00 | 1.00 / 1.00 | 0.93 / 0.97 |
| aggregation 4 | 1.00 / 1.00 | 0.00 / 0.00 | 0.93 / 0.95 |
| aggregation 5 | 0.38 / 0.36 | 1.00 / 1.00 | 0.93 / 0.95 |
| aggregation 6 (final) | 1.00 / 1.00 | 0.97 / 0.95 | 0.93 / 0.98 |

Fix recipe, session 2 (replicate):

| Global adapter after | Congestion (A / B) | Mobility (A / B) | Normal (A / B) |
|---|---|---|---|
| aggregation 1 | 0.00 / 0.00 | 0.00 / 0.00 | 1.00 / 1.00 |
| aggregation 2 | 0.06 / 0.02 | 0.00 / 0.01 | 1.00 / 1.00 |
| aggregation 3 | 0.00 / 0.00 | 1.00 / 1.00 | 0.95 / 0.95 |
| aggregation 4 | 0.96 / 0.95 | 0.00 / 0.02 | 0.95 / 1.00 |
| aggregation 5 | 0.00 / 0.00 | 1.00 / 1.00 | 0.95 / 0.95 |
| aggregation 6 (final) | 1.00 / 1.00 | 0.98 / 0.98 | 0.93 / 0.95 |

## Declared targets

| Declared before the run | Result |
|---|---|
| Federated run 1: family accuracy at least 0.85 on the unseen family at both sites (`ran-nlm-fed-20261102T0911.json`) | **Met**: Site A mobility 1.00, Site B congestion 0.89 (from the first scoring job; the re-grading gives 1.00 and 0.89) |
| Replicate: same target, reported next to run 1 and not replacing it (`ran-nlm-fed-rep2-20261102T1149.json`) | **Missed** at Site A mobility (0.00; Wilson 0.000-0.037). Site B congestion 1.00 met |
| Fix recipe, session 1 (an-nlm-fed-r6e1a-20261102T1350.json): at least 0.85 on the unseen family at both sites | **Met**: Site A mobility 0.97 (Wilson 0.915-0.990), Site B congestion 1.00 (0.963-1.000) |
| Fix recipe, session 2, the declared replicate (an-nlm-fed-r6e1b-20261102T1430.json): same target; the recipe counts as stable only if both sessions meet it | **Met**: Site A mobility 0.98 (0.930-0.994), Site B congestion 1.00 (0.963-1.000). Both fix sessions met it |
| Local-only adapters at or below 0.35 on the unseen family (`ran-nlm-baselines-20261102T090507.json`) | Met: 0.00 and 0.00 |
| Central at or above 0.90 on the unseen family | Met: 1.00 and 1.00 |
| Untuned base at or below 0.45 on the unseen family | **Missed** at Site A mobility (0.94): the base model already reads mobility text. Site B congestion 0.23 met |
| Only LoRA tensors are exchanged: 392 `lora_*` tensors | Met in both sessions (392 tensors; the transfer bucket holds 68 objects: 56 weight files, 8 certificate bundles, 4 source archives; no CSV or text files) |

## What the results say

1. **The fix worked in both sessions we ran.** With 6 rounds of 1 local epoch (the same total local training as the first recipe and as the local-only baselines) the federated adapter reached **0.975 and 0.979** pooled family accuracy, against 0.988 for central (not allowed in the real setting), about 0.60 for local-only and 0.646 for the untuned base. Both sessions met the declared target: mobility at Site A 0.97 and 0.98, congestion at Site B 1.00 and 1.00. Across all four federated sessions of both recipes, three met the target; the one that did not used the first recipe (0.948 and 0.610 pooled for its two sessions).
2. **How strong that evidence is.** Two sessions of one recipe; the recipe was chosen after we saw the first recipe's test results on the same frozen test text (stated in the declaration), so this is not an untouched confirmation. The intervals in the table do not cover run-to-run variation. Treat it as a positive result for this recipe in this setup, not as a general statement.
3. **Local-only training forgets the unseen family completely** (0.00 at both sites, in both directions) and ends below the untuned base overall (0.60 vs 0.65). The federated fix sessions repair this without sharing any text.
4. **The untuned base is not a blank slate.** It already identifies mobility text 92% to 94% of the time but calls congestion text "normal" 77% to 78% of the time. Federation's gain over the base is therefore mainly on congestion and on the cause labels, not on mobility.
5. **Cause accuracy is still uneven.** In the fix sessions it ranges from 0.51 (congestion cause at Site A, session 1) to 1.00 (congestion cause at Site A, session 2); mobility cause is 0.87 and 0.76 in session 1 and 0.91 and 0.82 in session 2, against 1.00 for central. Family is easier than cause.
6. **The global adapter still swings from round to round, and the last round is where it settles (measured, tables above).** In both fix sessions the pattern is the same: after aggregation 1 the adapter answers "normal" for everything; after 2 it is still close to that; after 3 it answers "mobility" for nearly everything; after 4 it answers "congestion" (0.95 to 1.00) and has lost mobility (0.00 to 0.02); after 5 it is back to mobility; after 6 it reads both families well. The first recipe showed the same swings in its first rounds, and its two sessions stopped in different places (one balanced, one congestion-only). We do not know whether the sixth aggregation is the start of a damped equilibrium or one phase of a continuing oscillation, because no seventh or eighth round was run. The good result is therefore tied to stopping at six rounds with this recipe; do not read it as "any round count is fine".
7. **Why the fix helps is not established.** Halving the local steps per round and doubling the number of averaging rounds is consistent with less client drift per round, but this was one single change, and we did not separate "fewer steps per round" from "more rounds". The persistent oscillation in point 6 shows the drift has not gone away.
8. **Comparability.** Client A trained on ml.g6.2xlarge in fix session 1 and on ml.g5.4xlarge in session 2 (capacity wait, GAPS C16); client B used ml.g5.2xlarge in both. All scoring ran on A10G GPUs. The first recipe's sessions used g6.2xlarge and g5.2xlarge.
9. **Re-grading reproduced four adapters exactly and one slightly differently.** Untuned, both local-only and the central adapter give identical counts to the first scoring job. The first recipe's session 1 adapter differs in a few examples: Site A congestion 90/100 instead of 89, cause correct 95 instead of 94 (congestion) and 65 instead of 69 (mobility); Site B congestion cause 81 instead of 82. We read this as GPU numerical differences on near-tied candidates (different instance, same code and file) and did not investigate further. The per-example records are the reported numbers.
10. **Final adapters of the first recipe.** Both have similar size (change from the shared initial adapter: Euclidean norm 6.1 vs 5.9 for lora_A, 6.4 vs 6.4 for lora_B) and lie at the same distance from both local-only adapters (cosine 0.52 to 0.54 in lora_B), but point in different directions (cosine between the two runs 0.67 for lora_A, 0.69 for lora_B).

## What would make it better (not done)

- Run 7 or 8 rounds and score the last three global adapters, to see whether the good state at round 6 persists or oscillates away.
- Several more replicates of the fix recipe, to report a spread; other round counts; and a fresh test text written before scoring, since this recipe was chosen after seeing the first test results.
- A proximal term (FedProx) in the clients' local training, to damp the swings directly (not built).
- Differential privacy for the adapter (not attempted: Opacus on a 1.7B-parameter model).