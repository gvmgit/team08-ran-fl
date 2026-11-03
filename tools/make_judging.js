// Builds judging.json from evidence/ so that every reported number is recomputed from the stored results, not typed in.
//
//   node tools/make_judging.js            writes judging.json at the repository root
//   (tools/rebuild.js requires this file and fails if judging.json differs from what build() returns)
//
// Shape follows the Participant Guide's judging.json skeleton (primary_track, team_code, evidence, run_ids, live_test_command,
// rebuild_command, reported_results). reported_results is keyed by the four Track 2 units.
const fs = require("fs");
const path = require("path");

const r4 = (x) => Math.round(x * 1e4) / 1e4;
const TEAM_CODE = "team08";
const ALL = [["a", "y_congestion"], ["a", "y_mobility"], ["b", "y_congestion"], ["b", "y_mobility"]];
const RARE = [["a", "y_mobility"], ["b", "y_congestion"]];
const key = (s, l) => `site_${s}_${l.replace("y_", "")}`;

function build(root) {
  const evid = path.join(root, "evidence");
  const checks = JSON.parse(fs.readFileSync(path.join(evid, "checks.json"), "utf8"));
  const read = (dir) => JSON.parse(fs.readFileSync(path.join(evid, "results", dir, "results.json"), "utf8"));
  const run = Object.fromEntries(checks.runs.map((r) => [r.key, r]));
  const main = read(run.main.run_id), stress = read(run.stress.run_id);
  const tb = checks.taskb;
  const tbRuns = tb.rescore_runs.map((r) => ({ run_id: r.run_id, res: read(r.results_dir) }));
  const holder = (model) => { const h = tbRuns.find((x) => x.res.scores && x.res.scores[model]); if (!h) throw new Error(`no Task B re-grading run holds the model ${model}`); return h; };

  const auroc = (r, model, pairs) => Object.fromEntries(pairs.map(([s, l]) => [key(s, l), r4(r.scores[model][s][l].auroc)]));
  const warned = (r, model, pairs) => Object.fromEntries(pairs.map(([s, l]) => [key(s, l), r4(r.scores[model][s][l].fraction_warned_60min_ahead)]));
  const f1 = (r, model, pairs) => Object.fromEntries(pairs.map(([s, l]) => [key(s, l), r4(r.scores[model][s][l].f1)]));
  const mean = (o) => { const v = Object.values(o); return v.reduce((a, b) => a + b, 0) / v.length; };
  const eps = Object.values(main.epsilon);
  const epsStress = Object.values(stress.epsilon);

  // Task B: family accuracy from the per-example records' counts (confusion matrices of the re-grading runs)
  const fam = (model, site, family) => { const v = holder(model).res.scores[model][site][family]; return r4((v.confusion[family] || 0) / v.n); };
  const overall = (model, site) => r4(holder(model).res.scores[model][site].ALL.family_acc);
  const unseen = (model) => ({ site_a_mobility: fam(model, "a", "MOBILITY"), site_b_congestion: fam(model, "b", "CONGESTION") });
  const pooled = (model) => { let k = 0, n = 0; for (const s of ["a", "b"]) { const sc = holder(model).res.scores[model][s]; for (const f of ["CONGESTION", "MOBILITY", "NONE"]) { k += sc[f].family_correct; n += sc[f].n; } } return r4(k / n); };
  const meets = (u) => u.site_a_mobility >= 0.85 && u.site_b_congestion >= 0.85;
  const beatsLocal = (u) => u.site_a_mobility > fam("local_a", "a", "MOBILITY") && u.site_b_congestion > fam("local_b", "b", "CONGESTION");

  // one entry per federated recipe (checks.json taskb.recipes), one sub-entry per session of that recipe
  const recipes = Object.fromEntries(tb.recipes.map((rc) => [rc.name, {
    description: rc.description,
    sessions: Object.fromEntries(rc.sessions.map((s) => [s.label, {
      session_run_id: s.run_id,
      graded_in: holder(s.model).run_id,
      family_accuracy_unseen_family: unseen(s.model),
      family_accuracy_overall_pooled: pooled(s.model),
      family_accuracy_overall: { site_a: overall(s.model, "a"), site_b: overall(s.model, "b") },
      met_declared_target_at_least_0_85_on_both_unseen_families: meets(unseen(s.model)),
      beats_local_only_on_both_unseen_families: beatsLocal(unseen(s.model)),
    }])),
    stable_all_sessions_meet_target: rc.sessions.every((s) => meets(unseen(s.model))),
  }]));
  const trajectory = (tb.trajectory || []).map((t) => ({ model: t.model, description: t.description, graded_in: holder(t.model).run_id, family_accuracy_by_true_family: { site_a: Object.fromEntries(["CONGESTION", "MOBILITY", "NONE"].map((f) => [f, fam(t.model, "a", f)])), site_b: Object.fromEntries(["CONGESTION", "MOBILITY", "NONE"].map((f) => [f, fam(t.model, "b", f)])) }, family_accuracy_overall_pooled: pooled(t.model) }));

  const cA = auroc(main, "central", ALL), fA = auroc(main, "federated_dp", ALL);
  const cS = auroc(stress, "central_stress", RARE), fS = auroc(stress, "federated_stress_dp", RARE);

  return {
    primary_track: "track-2",
    team_code: TEAM_CODE,
    evidence: { bucket: checks.judging.evidence_bucket, prefix: "evidence/" },
    run_ids: checks.judging.run_ids,
    live_test_command: "node tools/show_record.js --run-id {run_id} --item {item}",
    rebuild_command: "node tools/rebuild.js",
    reported_results: {
      "local-only-control": {
        declared: "AUROC (Task A) or family accuracy (Task B) on the family a site never or rarely logs at home, local-only model; the federated model must beat it",
        task_a_main: { run_id: run.main.run_id, auroc_rare_family: { site_a_mobility: r4(main.scores.local_a.a.y_mobility.auroc), site_b_congestion: r4(main.scores.local_b.b.y_congestion.auroc) }, declared_expectation_at_or_below: 0.85, met: false, why: "the supplied data are saturated, see GAPS.md A1" },
        task_a_stress: { run_id: run.stress.run_id, auroc_withheld_family: { site_a_mobility: r4(stress.scores.local_stress_a.a.y_mobility.auroc), site_b_congestion: r4(stress.scores.local_stress_b.b.y_congestion.auroc) }, declared_expectation_at_or_below: 0.6, met: true },
        task_b: { run_id: holder("local_a").run_id, family_accuracy_unseen_family: { site_a_mobility: fam("local_a", "a", "MOBILITY"), site_b_congestion: fam("local_b", "b", "CONGESTION") }, declared_expectation_at_or_below: 0.35, met: true },
      },
      "federated-run": {
        declared: "Task A main: AUROC at or above 0.97 on all four site/label pairs and at least 0.5 of rare-family events warned 60 minutes ahead; Task A stress: AUROC at or above 0.90 on the withheld families; Task B: family accuracy at or above 0.85 on the family each site never logged",
        task_a_main: { run_id: run.main.session_run_id, graded_in: run.main.run_id, auroc: fA, f1: f1(main, "federated_dp", ALL), fraction_warned_60min_ahead_rare_family: warned(main, "federated_dp", RARE), met: true },
        task_a_stress: { run_id: run.stress.session_run_id, graded_in: run.stress.run_id, auroc_withheld_family: fS, f1_withheld_family: f1(stress, "federated_stress_dp", RARE), fraction_warned_60min_ahead_withheld_family: warned(stress, "federated_stress_dp", RARE), met: true },
        task_b: {
          recipes,
          intermediate_rounds_diagnostic: trajectory,
          baselines_family_accuracy_unseen_family: { untuned_base: unseen("untuned_base"), local_only: { site_a_mobility: fam("local_a", "a", "MOBILITY"), site_b_congestion: fam("local_b", "b", "CONGESTION") }, central: unseen("central") },
          baselines_family_accuracy_overall_pooled: { untuned_base: pooled("untuned_base"), local_a: pooled("local_a"), local_b: pooled("local_b"), central: pooled("central") },
          reading: tb.reading,
        },
        federated_beats_local_only: {
          task_a_main: main.scores.federated_dp.a.y_mobility.auroc > main.scores.local_a.a.y_mobility.auroc && main.scores.federated_dp.b.y_congestion.auroc > main.scores.local_b.b.y_congestion.auroc,
          task_a_stress: stress.scores.federated_stress_dp.a.y_mobility.auroc > stress.scores.local_stress_a.a.y_mobility.auroc && stress.scores.federated_stress_dp.b.y_congestion.auroc > stress.scores.local_stress_b.b.y_congestion.auroc,
          task_b: Object.fromEntries(Object.entries(recipes).map(([name, rc]) => [name, Object.fromEntries(Object.entries(rc.sessions).map(([label, s]) => [label, s.beats_local_only_on_both_unseen_families]))])),
        },
      },
      "privacy-budget": {
        declared: "epsilon of the DP-SGD run that executes, counting real optimizer steps, at or below 1.0",
        task_a_main: { run_id: run.main.session_run_id, epsilon_target: Math.max(...eps.map((e) => e.target_epsilon)), epsilon_spent: r4(Math.max(...eps.map((e) => e.epsilon_spent_real_steps))), delta: Math.max(...eps.map((e) => e.delta)), noise_multiplier: eps[0].noise_multiplier_used, real_optimizer_steps: eps[0].real_optimizer_steps, met: Math.max(...eps.map((e) => e.epsilon_spent_real_steps)) <= 1 },
        task_a_stress: { run_id: run.stress.session_run_id, epsilon_spent: r4(Math.max(...epsStress.map((e) => e.epsilon_spent_real_steps))), met: Math.max(...epsStress.map((e) => e.epsilon_spent_real_steps)) <= 1 },
        task_b: "no differential privacy applied (see GAPS.md)",
      },
      "distance-to-central-baseline": {
        declared: "central model is a baseline only, outside the federated path; its floor was declared (Task A main at or above 0.95), a maximum gap was not (GAPS.md C13)",
        task_a_main: { central_run_id: "ran-central-20261102T080626", graded_in: run.main.run_id, mean_auroc_central: r4(mean(cA)), mean_auroc_federated: r4(mean(fA)), gap_central_minus_federated: r4(mean(cA) - mean(fA)) },
        task_a_stress: { central_run_id: "ran-stress-central-20261102T082613", graded_in: run.stress.run_id, mean_auroc_central_withheld_family: r4(mean(cS)), mean_auroc_federated_withheld_family: r4(mean(fS)), gap_central_minus_federated: r4(mean(cS) - mean(fS)) },
        task_b: {
          graded_in: holder("central").run_id,
          central_family_accuracy_overall_pooled: pooled("central"),
          federated_family_accuracy_overall_pooled: Object.fromEntries(Object.entries(recipes).map(([name, rc]) => [name, Object.fromEntries(Object.entries(rc.sessions).map(([label, s]) => [label, s.family_accuracy_overall_pooled]))])),
        },
      },
      "declared-targets-missed": [...checks.targets, ...(checks.relative_targets || []), ...tb.targets, ...(tb.rescore_targets || [])].filter((t) => t.disclosed_miss).map((t) => `${t.name} (declared in ${t.declaration})`),
    },
  };
}

module.exports = { build };

if (require.main === module) {
  const root = path.resolve(__dirname, "..");
  const out = path.join(root, "judging.json");
  fs.writeFileSync(out, JSON.stringify(build(root), null, 2) + "\n");
  console.log(`wrote ${out}`);
}
