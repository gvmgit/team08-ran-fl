// Rebuild command: reprints the reported Task A results from evidence/ with NO AWS access, and exits 1 if any evidence
// changed or any declared target is not met.
//
//   node tools/rebuild.js               verify + print + check
//   node tools/rebuild.js --write-manifest   (re)write evidence/MANIFEST.json after copying fresh results into evidence/results/
//
// What it can and cannot recompute: AUROC and F1 come from the grader's results.json (the per-window predictions are not
// kept, they are 1M rows per model and site). The per-EVENT records are kept, so the 60-minute lead-time recall is
// recomputed from them and compared with the reported value. Every file is verified against SHA-256 hashes.
const crypto = require("crypto");
const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const evid = path.join(root, "evidence");
const sha = (p) => crypto.createHash("sha256").update(fs.readFileSync(p)).digest("hex");
const checks = JSON.parse(fs.readFileSync(path.join(evid, "checks.json"), "utf8"));
const manifestPath = path.join(evid, "MANIFEST.json");

function listFiles(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) => (e.isDirectory() ? listFiles(path.join(dir, e.name)) : [path.join(dir, e.name)]));
}

if (process.argv.includes("--write-manifest")) {
  const files = [...listFiles(path.join(evid, "results")), ...listFiles(path.join(evid, "declared")), ...listFiles(path.join(evid, "taskb_test"))];
  const m = Object.fromEntries(files.sort().map((f) => [path.relative(root, f).split(path.sep).join("/"), sha(f)]));
  fs.writeFileSync(manifestPath, JSON.stringify(m, null, 2));
  console.log(`wrote ${manifestPath} (${files.length} files)`);
  process.exit(0);
}

let failures = 0;
let disclosedMisses = 0;   // declared targets that were not met and are explained in GAPS.md; any other miss is a failure
let disclosedDifferences = 0;   // re-grading results that differ from the earlier scoring of the same adapter and are explained in GAPS.md
const fail = (msg) => { failures++; console.log(`  FAIL  ${msg}`); };
const ok = (msg) => console.log(`  pass  ${msg}`);

console.log("== 1. evidence integrity (SHA-256 against evidence/MANIFEST.json) ==");
const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
for (const [rel, expected] of Object.entries(manifest)) {
  const p = path.join(root, rel);
  if (!fs.existsSync(p)) { fail(`${rel} is missing`); continue; }
  if (sha(p) !== expected) fail(`${rel} changed`);
}
if (!failures) ok(`${Object.keys(manifest).length} files match the manifest`);

const results = {};
console.log("\n== 2. per-event records agree with the reported results ==");
for (const run of checks.runs) {
  const dir = path.join(evid, "results", run.run_id);
  const r = JSON.parse(fs.readFileSync(path.join(dir, "results.json"), "utf8"));
  results[run.key] = r;
  if (r.status !== "ok") fail(`${run.run_id}: grader status ${r.status}`);
  if (r.test_set_hash_matches_declaration && Object.values(r.test_set_hash_matches_declaration).some((v) => v !== true)) fail(`${run.run_id}: test-set hash does not match the declaration`);
  for (const [model, sites] of Object.entries(r.scores)) {
    for (const [site, labels] of Object.entries(sites)) {
      for (const [label, v] of Object.entries(labels)) {
        const rec = path.join(dir, "records", `${model}__${site}__${label}.jsonl`);
        const rows = fs.readFileSync(rec, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l));
        if (r.records_sha256[`${model}__${site}__${label}`] !== sha(rec)) fail(`${run.key}/${model}/${site}/${label}: records file hash differs from the grader's`);
        const w60 = rows.filter((x) => x.warned_60min_ahead).length / Math.max(1, rows.length);
        if (rows.length !== v.events_scored || Math.abs(w60 - v.fraction_warned_60min_ahead) > 1e-9) fail(`${run.key}/${model}/${site}/${label}: records give ${rows.length} events, lead recall ${w60.toFixed(4)}; reported ${v.events_scored}, ${v.fraction_warned_60min_ahead.toFixed(4)}`);
        if (rows.some((x) => x.run_id !== run.run_id)) fail(`${run.key}/${model}/${site}/${label}: a record carries a different run id`);
      }
    }
  }
}
if (!failures) ok("every record file hashes to the grader's value, carries its run id, and reproduces the reported lead-time recall");

console.log("\n== 3. reported results (from results.json; lead recall recomputed from records) ==");
for (const run of checks.runs) {
  const r = results[run.key];
  console.log(`\n${run.title}   [run id ${run.run_id}]`);
  console.log("model".padEnd(20) + "site  " + "label".padEnd(14) + "withheld  " + "AUROC".padStart(8) + "F1".padStart(8) + "warn60".padStart(8));
  for (const [model, sites] of Object.entries(r.scores)) for (const [site, labels] of Object.entries(sites)) for (const [label, v] of Object.entries(labels))
    console.log(model.padEnd(20) + site.padEnd(6) + label.padEnd(14) + String(!!v.withheld_at_home).padEnd(10) + v.auroc.toFixed(4).padStart(8) + v.f1.toFixed(4).padStart(8) + v.fraction_warned_60min_ahead.toFixed(3).padStart(8));
  if (r.epsilon) for (const [site, e] of Object.entries(r.epsilon)) console.log(`epsilon, silo ${site}: target ${e.target_epsilon}, spent ${e.epsilon_spent_real_steps.toFixed(4)} (delta ${e.delta.toExponential(3)}, noise multiplier ${e.noise_multiplier_used}, ${e.real_optimizer_steps} real steps)`);
}

console.log("\n== 4. declared targets (pass/fail) ==");
const cmp = { ">=": (a, b) => a >= b, "<=": (a, b) => a <= b, ">": (a, b) => a > b, "<": (a, b) => a < b };
const verdict = (good, disclosed, text) => {
  if (good) ok(text);
  else if (disclosed) { disclosedMisses++; console.log(`  MISS  ${text}   <- declared before the run, not met, explained in GAPS.md`); }
  else fail(text);
};
for (const c of checks.targets) {
  const r = results[c.run];
  const values = [];
  if (c.metric === "epsilon_spent") for (const e of Object.values(r.epsilon)) values.push(e.epsilon_spent_real_steps);
  else for (const [site, label] of c.pairs === "all" ? Object.keys(r.scores[c.model]).flatMap((s) => Object.keys(r.scores[c.model][s]).map((l) => [s, l])) : c.pairs) values.push(r.scores[c.model][site][label][c.metric]);
  const good = values.every((v) => cmp[c.op](v, c.value));
  verdict(good, c.disclosed_miss, `${c.name}: values ${values.map((v) => v.toFixed(4)).join(", ")} ${c.op} ${c.value}   (declared in ${c.declaration})`);
}
console.log("\n== 4b. declared relative targets (the federated model must beat the local-only control) ==");
for (const t of checks.relative_targets || []) {
  const get = (x) => results[t.run].scores[x.model][x.pair[0]][x.pair[1]][t.metric];
  const a = get(t.better), b = get(t.than);
  verdict(cmp[t.op](a, b), t.disclosed_miss, `${t.name}: ${t.better.model} ${a.toFixed(4)} ${t.op} ${t.than.model} ${b.toFixed(4)}   (declared in ${t.declaration})`);
}

console.log("\n== 5. Task B: family accuracy recomputed from the grader's confusion matrices ==");
const tb = checks.taskb;
const tbRes = {};
for (const [site, file] of Object.entries(tb.test_corpus)) {   // the frozen test text, kept in the repo so this can be checked offline
  const declared = tb.runs.map((r) => JSON.parse(fs.readFileSync(path.join(evid, "declared", tb.declarations[r.key]), "utf8")).test_corpus_sha256[`site_${site}_test`]);
  const actual = sha(path.join(evid, file));
  if (declared.some((h) => h !== actual)) fail(`Task B site ${site} test text (${file}) does not hash to the value declared before the runs`);
}
if (!failures) ok("Task B test text for both sites hashes to the value declared before the baselines and the federated run");
for (const run of tb.runs) {
  const r = JSON.parse(fs.readFileSync(path.join(evid, "results", run.results_dir, "results.json"), "utf8"));
  tbRes[run.key] = r;
  if (r.status !== "ok") fail(`${run.run_id}: grader status ${r.status}`);
  console.log(`\n${run.title}   [run id ${run.run_id}]`);
  console.log("model".padEnd(14) + "site  " + "true family".padEnd(13) + "n".padStart(5) + "correct".padStart(9) + "accuracy".padStart(10) + "   called instead");
  for (const [model, sites] of Object.entries(r.scores)) for (const [site, fams] of Object.entries(sites)) for (const [fam, v] of Object.entries(fams)) {
    if (fam === "ALL") { console.log(model.padEnd(14) + site.padEnd(6) + "ALL".padEnd(13) + String(v.n).padStart(5) + "".padStart(9) + v.family_acc.toFixed(4).padStart(10)); continue; }
    const counted = Object.values(v.confusion).reduce((a, b) => a + b, 0);
    const right = v.confusion[fam] || 0;
    if (counted !== v.n || right !== v.family_correct || Math.abs(right / v.n - v.family_acc) > 1e-9) fail(`${run.key}/${model}/${site}/${fam}: confusion counts (${right}/${counted}) disagree with the reported ${v.family_correct}/${v.n}, ${v.family_acc}`);
    const other = Object.entries(v.confusion).filter(([k]) => k !== fam).map(([k, c]) => `${k} ${c}`).join(", ");
    console.log(model.padEnd(14) + site.padEnd(6) + fam.padEnd(13) + String(v.n).padStart(5) + String(right).padStart(9) + (right / v.n).toFixed(4).padStart(10) + "   " + other);
  }
}
const wilson = (k, n) => { const z = 1.96, p = k / n, d = 1 + z * z / n, c = (p + z * z / (2 * n)) / d, h = (z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d; return [Math.max(0, c - h), Math.min(1, c + h)]; };
console.log("\n== 6. Task B declared targets (pass / fail / disclosed miss; 95% Wilson interval in brackets) ==");
const tbTarget = (t) => {
  const v = tbRes[t.run].scores[t.model][t.site][t.family];
  const right = v.confusion[t.family] || 0;
  const acc = right / v.n;
  const good = cmp[t.op](acc, t.value);
  const ci = wilson(right, v.n);
  const text = `${t.name}: ${acc.toFixed(4)} [${ci[0].toFixed(3)}, ${ci[1].toFixed(3)}] ${t.op} ${t.value}   (declared in ${t.declaration})`;
  verdict(good, t.disclosed_miss, text);
};
const tbAdapterOnly = (ao) => {
  const f = tbRes[ao.run].adapter_files[ao.model];
  const good = f && f.tensors === ao.tensors && /(federated_final_)?iteration\d+\.safetensors$/.test(f.source);
  (good ? ok : fail)(`only the LoRA adapter is exchanged and scored (${ao.model}): ${f ? f.tensors : "?"} tensors (declared ${ao.tensors}), file ${f ? f.source.split("/").pop() : "?"}   (declared in ${ao.declaration})`);
};
for (const t of tb.targets) tbTarget(t);
const aoList = Array.isArray(tb.adapter_only) ? tb.adapter_only : [tb.adapter_only];
const aoDone = new Set();
for (const ao of aoList.filter((a) => tbRes[a.run])) { tbAdapterOnly(ao); aoDone.add(ao); }

if ((tb.rescore_runs || []).length) console.log("\n== 7. Task B per-example records (re-grading jobs: one record per test example) ==");
for (const run of tb.rescore_runs || []) {
  const dir = path.join(evid, "results", run.results_dir);
  const r = JSON.parse(fs.readFileSync(path.join(dir, "results.json"), "utf8"));
  tbRes[run.key] = r;
  console.log(`\n${run.title}   [run id ${run.run_id}]`);
  if (r.status !== "ok") fail(`${run.run_id}: grader status ${r.status}`);
  if (r.run_id !== run.run_id) fail(`${run.run_id}: results carry run id ${r.run_id}`);
  for (const [site, file] of Object.entries(tb.test_corpus)) if (r.test_set_sha256[site] !== sha(path.join(evid, file))) fail(`${run.run_id}: test text of site ${site} differs from the frozen test text`);
  for (const [model, a] of Object.entries(r.adapter_files || {})) console.log(`  adapter ${model.padEnd(13)} ${a.tensors} tensors  sha256 ${a.sha256.slice(0, 16)}...  ${a.source.split("/").pop()}`);
  console.log("model".padEnd(14) + "site  " + "records".padStart(8) + "  family acc by true family (CONGESTION / MOBILITY / NONE) and overall" + "      same counts as the earlier scoring");
  for (const [model, sites] of Object.entries(r.scores)) for (const [site, fams] of Object.entries(sites)) {
    const rec = path.join(dir, "records", `${model}__${site}.jsonl`);
    const rows = fs.readFileSync(rec, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l));
    const tag = `${run.key}/${model}/${site}`;
    if (r.records_sha256[`${model}__${site}`] !== sha(rec)) fail(`${tag}: records file hash differs from the grader's`);
    if (rows.some((x) => x.run_id !== run.run_id || x.model !== model || x.site !== site)) fail(`${tag}: a record carries a different run id, model or site`);
    const isBest = (obj, pick) => obj[pick] !== undefined && obj[pick] === Math.max(...Object.values(obj));
    const got = {};
    for (const x of rows) {
      if (!isBest(x.family_logprobs, x.pred_family)) fail(`${tag} #${x.test_index}: recorded family choice is not the most likely candidate`);
      if (x.pred_cause !== undefined && (!isBest(x.cause_logprobs, x.pred_cause) || !isBest(x.action_logprobs, x.pred_action))) fail(`${tag} #${x.test_index}: recorded cause or action choice is not the most likely candidate`);
      const g = (got[x.true_family] = got[x.true_family] || { n: 0, family_correct: 0, cause_correct: 0, action_correct: 0, confusion: {} });
      g.n++; g.family_correct += x.pred_family === x.true_family ? 1 : 0; g.confusion[x.pred_family] = (g.confusion[x.pred_family] || 0) + 1;
      if (x.pred_cause !== undefined) { g.cause_correct += x.pred_cause === x.true_cause ? 1 : 0; g.action_correct += x.pred_action === x.true_action ? 1 : 0; }
    }
    for (const [fam, v] of Object.entries(fams)) {
      if (fam === "ALL") continue;
      const g = got[fam] || { n: 0, family_correct: 0, cause_correct: 0, action_correct: 0, confusion: {} };
      const same = g.n === v.n && g.family_correct === v.family_correct && g.cause_correct === v.cause_correct && g.action_correct === v.action_correct && JSON.stringify(Object.entries(g.confusion).sort()) === JSON.stringify(Object.entries(v.confusion).sort());
      if (!same) fail(`${tag}/${fam}: the records do not reproduce the summary counts in results.json`);
    }
    if (rows.length !== fams.ALL.n) fail(`${tag}: ${rows.length} records but ${fams.ALL.n} test examples`);
    // same counts as the original (confusion-matrix only) scoring job of this model?
    const orig = (run.compare_to || {})[model];
    let cmpText = "n/a";
    if (orig) {
      const o = tbRes[orig].scores[model][site];
      const equal = Object.keys(o).every((fam) => fam === "ALL" || (o[fam].n === got[fam].n && o[fam].family_correct === got[fam].family_correct && o[fam].cause_correct === got[fam].cause_correct && o[fam].action_correct === got[fam].action_correct && JSON.stringify(Object.entries(o[fam].confusion).sort()) === JSON.stringify(Object.entries(got[fam].confusion).sort())));
      if (equal) cmpText = `identical to ${orig}`;
      else if ((run.allowed_differences || []).includes(model)) { disclosedDifferences++; cmpText = `DIFFERS from ${orig} (disclosed in GAPS.md; the records are the reported numbers)`; }
      else { cmpText = `DIFFERS from ${orig}`; fail(`${tag}: family/cause/action counts differ from the earlier scoring (${orig})`); }
    }
    const f = (fam) => (got[fam] ? (got[fam].family_correct / got[fam].n).toFixed(3) : "  -  ");
    const overall = Object.values(got).reduce((a, g) => a + g.family_correct, 0) / rows.length;
    console.log(model.padEnd(14) + site.padEnd(6) + String(rows.length).padStart(8) + `  ${f("CONGESTION")} / ${f("MOBILITY")} / ${f("NONE")}   overall ${overall.toFixed(4)}      ${cmpText}`);
  }
}
if ((tb.rescore_runs || []).length && !failures) ok("every per-example record file hashes to the grader's value, carries its run id, picks the most likely candidate, and reproduces the summary counts");
if ((tb.rescore_targets || []).length) console.log("\n== 7b. Task B declared targets for adapters scored only in the re-grading run (replicate) ==");
for (const t of tb.rescore_targets || []) tbTarget(t);
for (const ao of aoList.filter((a) => !aoDone.has(a))) tbAdapterOnly(ao);   // runs that were not loaded yet when section 6 ran

console.log("\n== 8. judging.json agrees with the evidence ==");
{
  const judgingPath = path.join(root, "judging.json");
  if (!fs.existsSync(judgingPath)) fail("judging.json is missing");
  else {
    const have = JSON.parse(fs.readFileSync(judgingPath, "utf8"));
    const want = require("./make_judging.js").build(root);
    if (JSON.stringify(have) === JSON.stringify(want)) ok("judging.json equals what tools/make_judging.js builds from the stored results");
    else fail("judging.json differs from what tools/make_judging.js builds from the stored results (run: node tools/make_judging.js)");
    if (have.run_ids.length > 20) fail(`judging.json lists ${have.run_ids.length} run ids; the limit is 20`);
    const regradeIds = new Set(checks.runs.map((r) => r.run_id));   // Task A re-grading jobs: their targets were declared by the runs they grade
    for (const id of have.run_ids) {
      const f = path.join(evid, "declared", `${id}.json`);
      if (fs.existsSync(f)) {
        if (JSON.parse(fs.readFileSync(f, "utf8")).run_id !== id) fail(`${id}: the declaration file carries a different run id`);
      } else if (regradeIds.has(id)) console.log(`  note  ${id}: re-grading job with no declaration of its own; its targets are declared by the runs listed in checks.json`);
      else fail(`${id}: no declaration in evidence/declared/`);
    }
    if (!failures) ok(`${have.run_ids.length} run ids, each with its declaration (or named as a re-grading job above)`);
  }
}

if (failures) console.log(`\nREBUILD FAILED: ${failures} problem(s)`);
else console.log(`\nREBUILD OK: evidence unchanged, every declared target met${disclosedMisses ? ` except ${disclosedMisses} disclosed miss(es) listed above` : ""}${disclosedDifferences ? `; ${disclosedDifferences} re-graded result(s) differ from the earlier scoring (disclosed)` : ""}`);
process.exit(failures ? 1 : 0);
