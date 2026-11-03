// Read-only live-test tool: shows ONE per-item record of a scored run and where it lives.
// Uses only read calls: S3 GetObject/HeadObject, CloudWatch Logs DescribeLogStreams. Nothing is invoked or written.
//
// Task A (early-warning model; the run id is the SageMaker Processing job name of the grader):
//   node tools/show_record.js --run-id ran-rescore-stress-20261102T090736 --item "model=federated_stress_dp,site=a,label=y_mobility,day=100,cell=12"
//   item keys: model, site, label (y_congestion|y_mobility), and any of cell, day, minute (event minute of the day),
//   window (minutes either side of `minute`, default 30).
//
// Task B (federated LoRA text adapter; the run id is the SageMaker Training job name of the grader):
//   node tools/show_record.js --run-id <task-b-grader-run-id> --item "model=federated,site=a,family=MOBILITY,test_index=7"
//   item keys: model (untuned_base|local_a|local_b|central|federated...), site, and any of family (the TRUE family),
//   test_index, cell, minute (minute_index of the example), window (default 30).
//
// The mode is chosen by the item: `label=` means Task A, anything else means Task B. In both cases the run id is in the
// S3 key, in every record and in the CloudWatch log stream, and the tool compares the S3 write time with the job's log times.
const { spawnSync } = require("child_process");
const fs = require("fs");

const args = Object.fromEntries(process.argv.slice(2).reduce((a, v, i, all) => (v.startsWith("--") ? [...a, [v.slice(2), all[i + 1]]] : a), []));
if (!args["run-id"] || !args.item) { console.error('usage: node show_record.js --run-id <id> --item "model=...,site=a,label=y_mobility,day=100,cell=12"   (Task A)\n   or: node show_record.js --run-id <id> --item "model=federated,site=a,family=MOBILITY,test_index=7"   (Task B)'); process.exit(2); }
const item = Object.fromEntries(args.item.split(",").map((kv) => kv.split("=")));
for (const k of ["model", "site"]) if (!item[k]) { console.error(`item needs ${k}=`); process.exit(2); }
const taskA = item.label !== undefined;

const awsExe = process.env.AWS_EXE || (fs.existsSync("C:\\Program Files\\Amazon\\AWSCLIV2\\aws.exe") ? "C:\\Program Files\\Amazon\\AWSCLIV2\\aws.exe" : "aws");
const profile = process.env.AWS_PROFILE || "fyuz";
function aws(a, opts = {}) {
  const r = spawnSync(awsExe, [...a, "--profile", profile], { encoding: "utf8", maxBuffer: 256 * 1024 * 1024, ...opts });
  if (r.status !== 0) { console.error(`aws ${a.slice(0, 3).join(" ")} failed: ${(r.stderr || "").trim().slice(0, 300)}`); process.exit(3); }
  return r.stdout;
}
const account = JSON.parse(aws(["sts", "get-caller-identity", "--output", "json"])).Account;
const bucket = args.bucket || process.env.EVIDENCE_BUCKET || `bb26-p-team08-data-${account}-use2`;
const runId = args["run-id"];
const file = taskA ? `${item.model}__${item.site}__${item.label}.jsonl` : `${item.model}__${item.site}.jsonl`;
const key = `ran-fl/jobs/${runId}/output/records/${file}`;
// S3 holds the evidence in us-east-2. The Task A grader is a Processing job in us-east-2, the Task B grader a Training job in us-west-2.
const logGroup = taskA ? "/aws/sagemaker/ProcessingJobs" : "/aws/sagemaker/TrainingJobs";
const logRegion = taskA ? "us-east-2" : "us-west-2";

const body = aws(["s3", "cp", `s3://${bucket}/${key}`, "-", "--region", "us-east-2"]);
const records = body.split("\n").filter(Boolean).map((l) => JSON.parse(l));
const win = Number(item.window || 30);
const hits = taskA
  ? records.filter((r) =>
      (item.cell === undefined || r.cell === Number(item.cell)) &&
      (item.day === undefined || r.day === Number(item.day)) &&
      (item.minute === undefined || Math.abs(r.minute_of_day - Number(item.minute)) <= win))
  : records.filter((r) =>
      (item.family === undefined || r.true_family === item.family) &&
      (item.test_index === undefined || r.test_index === Number(item.test_index)) &&
      (item.cell === undefined || r.cell === Number(item.cell)) &&
      (item.minute === undefined || Math.abs(r.minute_index - Number(item.minute)) <= win));

console.log(`record file : s3://${bucket}/${key}  (${records.length} ${taskA ? "events" : "test examples"} in the file, ${hits.length} match)`);
const head = JSON.parse(aws(["s3api", "head-object", "--bucket", bucket, "--key", key, "--region", "us-east-2", "--output", "json"]));
console.log(`S3 recorded : last modified ${head.LastModified}`);
const streams = JSON.parse(aws(["logs", "describe-log-streams", "--log-group-name", logGroup, "--log-stream-name-prefix", `${runId}/`, "--region", logRegion, "--output", "json"])).logStreams || [];
if (streams.length) {
  // describe-log-streams reports first/lastEventTimestamp lazily (they can lag for a long job), so read the real first and last events.
  const eventTs = (name, fromHead) => Number(aws(["logs", "get-log-events", "--log-group-name", logGroup, "--log-stream-name", name, "--limit", "1", fromHead ? "--start-from-head" : "--no-start-from-head", "--region", logRegion, "--query", "events[0].timestamp", "--output", "text"]).trim());
  const firsts = streams.map((s) => eventTs(s.logStreamName, true)).filter(Number.isFinite), lasts = streams.map((s) => eventTs(s.logStreamName, false)).filter(Number.isFinite);
  const first = new Date(Math.min(...firsts)), last = new Date(Math.max(...lasts));
  const modified = new Date(head.LastModified);
  console.log(`CloudWatch  : ${logGroup} (${logRegion}) log stream ${streams[0].logStreamName}  first event ${first.toISOString()}  last event ${last.toISOString()}`);
  console.log(`times agree : ${modified >= first && modified <= new Date(last.getTime() + 10 * 60 * 1000) ? "yes (record file written between the job's first log event and 10 minutes after its last)" : "NO"}`);
} else console.log("CloudWatch  : no log stream found for this run id");
for (const h of hits.slice(0, 5)) console.log("\n" + JSON.stringify(h, null, 2));
process.exit(hits.length ? 0 : 1);
