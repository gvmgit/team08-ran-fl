# Uploads a job folder to the evidence bucket, runs it as a SageMaker Processing job in us-east-2 (CPU deep-learning
# image), waits, and downloads the output folder. No credentials are handled here; it uses the SSO profile.
#   powershell -File run_processing.ps1 -JobName ran-validate -CodeDir <folder> -Entry validate_job.py `
#       -Env @{DATA_BUCKET_NAME='...'; RAN_DATA_KEY='...'} [-InstanceType ml.m5.xlarge] [-TimeoutMinutes 40]
param(
  [Parameter(Mandatory)] [string] $JobName,
  [Parameter(Mandatory)] [string] $CodeDir,
  [Parameter(Mandatory)] [string] $Entry,
  [hashtable] $Env = @{},
  [string] $InstanceType = "ml.m5.xlarge",
  [int] $TimeoutMinutes = 40,
  [string] $OutDir = "",
  [string] $Declaration = "",  # optional declaration template; "__RUN_ID__" is replaced and it is uploaded BEFORE the run starts
  [string] $Image = ""         # default: CPU deep-learning image; pass the GPU image for GPU instance types
)
$ErrorActionPreference = "Stop"
# Profile, account and the optional CA bundle for a TLS-inspecting proxy come from env.ps1.
. "$PSScriptRoot\env.ps1"
$aws = $RanAws
$region = "us-east-2"
$acct = $RanAccount
$bucket = "bb26-p-team08-data-$acct-use2"
$role = "arn:aws:iam::${acct}:role/bb26-SageMakerExecution"
$image = if ($Image) { $Image } else { "763104351884.dkr.ecr.$region.amazonaws.com/pytorch-training:2.9.0-cpu-py312-ubuntu22.04-sagemaker" }
$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmss")
$name = "$JobName-$stamp"
$codeUri = "s3://$bucket/ran-fl/jobs/$name/code"
$outUri = "s3://$bucket/ran-fl/jobs/$name/output"
if (-not $OutDir) { $OutDir = Join-Path (Split-Path $CodeDir -Parent) "out-$name" }

# Declare before you run: the declaration (target, frozen test-set hash, model id, grader) is written to the evidence
# prefix first. If that upload fails, the run does not start.
if ($Declaration) {
  $text = (Get-Content $Declaration -Raw).Replace("__RUN_ID__", $name)
  $tmp = Join-Path $env:TEMP "$name.declared.json"
  [System.IO.File]::WriteAllText($tmp, $text, (New-Object System.Text.ASCIIEncoding))
  & $aws s3 cp $tmp "s3://$bucket/evidence/declared/$name.json" --region $region --profile $RanProfile --only-show-errors
  if ($LASTEXITCODE -ne 0) { throw "declaration upload failed; not starting the run" }
  $repoDecl = Join-Path (Split-Path $PSScriptRoot -Parent) "evidence\declared"
  New-Item -ItemType Directory -Force $repoDecl | Out-Null
  Copy-Item $tmp (Join-Path $repoDecl "$name.json") -Force
  "declared: s3://$bucket/evidence/declared/$name.json"
}

& $aws s3 cp $CodeDir $codeUri --recursive --region $region --profile $RanProfile --only-show-errors
if ($LASTEXITCODE -ne 0) { throw "code upload failed" }

$envJson = @{}
foreach ($k in $Env.Keys) { $envJson[$k] = [string]$Env[$k] }
if (-not $envJson.ContainsKey("RUN_ID")) { $envJson["RUN_ID"] = $name }   # one run id end to end: it reaches the job's records
$spec = [ordered]@{
  ProcessingJobName = $name
  RoleArn = $role
  AppSpecification = [ordered]@{ ImageUri = $image; ContainerEntrypoint = @("python3", "/opt/ml/processing/input/code/$Entry") }
  ProcessingResources = @{ ClusterConfig = [ordered]@{ InstanceCount = 1; InstanceType = $InstanceType; VolumeSizeInGB = 50 } }
  ProcessingInputs = @(@{ InputName = "code"; S3Input = [ordered]@{ S3Uri = $codeUri; LocalPath = "/opt/ml/processing/input/code"; S3DataType = "S3Prefix"; S3InputMode = "File" } })
  ProcessingOutputConfig = @{ Outputs = @(@{ OutputName = "out"; S3Output = [ordered]@{ S3Uri = $outUri; LocalPath = "/opt/ml/processing/output"; S3UploadMode = "EndOfJob" } }) }
  StoppingCondition = @{ MaxRuntimeInSeconds = $TimeoutMinutes * 60 }
  Environment = $envJson
}
$specPath = Join-Path $env:TEMP "$name.json"
[System.IO.File]::WriteAllText($specPath, ($spec | ConvertTo-Json -Depth 8), (New-Object System.Text.ASCIIEncoding))
& $aws sagemaker create-processing-job --cli-input-json "file://$specPath" --region $region --profile $RanProfile --query ProcessingJobArn --output text
if ($LASTEXITCODE -ne 0) { throw "create-processing-job failed" }
Remove-Item $specPath -ErrorAction SilentlyContinue
"job: $name"

$deadline = (Get-Date).AddMinutes($TimeoutMinutes + 10)
do {
  Start-Sleep -Seconds 20
  $st = & $aws sagemaker describe-processing-job --processing-job-name $name --region $region --profile $RanProfile --query "[ProcessingJobStatus,FailureReason]" --output text
  "{0}  {1}" -f (Get-Date -Format "HH:mm:ss"), $st
  # An empty answer means the call itself failed (for example a proxy hiccup), not that the job ended: keep waiting.
} while ((-not $st -or ($st -split "`t")[0] -in @("InProgress", "Stopping")) -and (Get-Date) -lt $deadline)

New-Item -ItemType Directory -Force $OutDir | Out-Null
& $aws s3 cp $outUri $OutDir --recursive --region $region --profile $RanProfile --only-show-errors
"output folder: $OutDir"
"== log tail =="
$streams = & $aws logs describe-log-streams --log-group-name /aws/sagemaker/ProcessingJobs --log-stream-name-prefix "$name/" --region $region --profile $RanProfile --query "logStreams[].logStreamName" --output text
foreach ($s in ($streams -split "\s+" | Where-Object { $_ })) {
  & $aws logs get-log-events --log-group-name /aws/sagemaker/ProcessingJobs --log-stream-name $s --region $region --profile $RanProfile --start-from-head --query "events[].message" --output text | Select-Object -Last 60
}
