# Runs a folder as a SageMaker script-mode TRAINING job on a GPU instance in us-west-2 (GPU Processing is unavailable in
# this account). The folder must contain the entry script and a requirements.txt. Results are whatever the script writes
# to /opt/ml/model; they are downloaded and unpacked here.
#   run_training_job.ps1 -JobName ran-nlm-baselines -CodeDir <folder> -Entry taskb_job.py -InstanceType ml.g5.4xlarge
param(
  [Parameter(Mandatory)] [string] $JobName,
  [Parameter(Mandatory)] [string] $CodeDir,
  [Parameter(Mandatory)] [string] $Entry,
  [string] $InstanceType = "ml.g5.4xlarge",
  [hashtable] $Env = @{},
  [int] $TimeoutMinutes = 90,
  [int] $VolumeGb = 100,
  [string] $OutDir = "",
  [string] $Declaration = ""
)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\env.ps1"   # profile, account and the optional CA bundle for a TLS-inspecting proxy
$aws = $RanAws
$region = "us-west-2"
$acct = $RanAccount
$bucket = "bb26-p-team08-nlm-$acct-usw2"
$evidenceBucket = "bb26-p-team08-data-$acct-use2"
$role = "arn:aws:iam::${acct}:role/bb26-SageMakerExecution"
$image = "763104351884.dkr.ecr.$region.amazonaws.com/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-sagemaker"
$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmss")
$name = "$JobName-$stamp"
if (-not $OutDir) { $OutDir = Join-Path (Split-Path $CodeDir -Parent) "out-$name" }

if ($Declaration) {   # declare before you run
  $text = (Get-Content $Declaration -Raw).Replace("__RUN_ID__", $name)
  $tmp = Join-Path $env:TEMP "$name.declared.json"
  [System.IO.File]::WriteAllText($tmp, $text, (New-Object System.Text.ASCIIEncoding))
  & $aws s3 cp $tmp "s3://$evidenceBucket/evidence/declared/$name.json" --region us-east-2 --profile $RanProfile --only-show-errors
  if ($LASTEXITCODE -ne 0) { throw "declaration upload failed; not starting the run" }
  $repoDecl = Join-Path (Split-Path $PSScriptRoot -Parent) "evidence\declared"
  New-Item -ItemType Directory -Force $repoDecl | Out-Null
  Copy-Item $tmp (Join-Path $repoDecl "$name.json") -Force
  "declared: s3://$evidenceBucket/evidence/declared/$name.json"
}

$tgz = Join-Path $env:TEMP "$name.source.tar.gz"
& tar.exe -czf $tgz -C $CodeDir .
if ($LASTEXITCODE -ne 0) { throw "tar failed" }
$srcUri = "s3://$bucket/code/$name/source.tar.gz"
& $aws s3 cp $tgz $srcUri --region $region --profile $RanProfile --only-show-errors
if ($LASTEXITCODE -ne 0) { throw "code upload failed" }

$hp = [ordered]@{
  sagemaker_program = "`"$Entry`""
  sagemaker_submit_directory = "`"$srcUri`""
  sagemaker_region = "`"$region`""
  sagemaker_job_name = "`"$name`""
  sagemaker_container_log_level = "20"
}
$envJson = @{}
foreach ($k in $Env.Keys) { $envJson[$k] = [string]$Env[$k] }
if (-not $envJson.ContainsKey("RUN_ID")) { $envJson["RUN_ID"] = $name }   # one run id end to end: it reaches the job's records
$spec = [ordered]@{
  TrainingJobName = $name
  RoleArn = $role
  AlgorithmSpecification = [ordered]@{ TrainingImage = $image; TrainingInputMode = "File" }
  ResourceConfig = [ordered]@{ InstanceType = $InstanceType; InstanceCount = 1; VolumeSizeInGB = $VolumeGb }
  OutputDataConfig = @{ S3OutputPath = "s3://$bucket/outputs/$name" }
  StoppingCondition = @{ MaxRuntimeInSeconds = $TimeoutMinutes * 60 }
  HyperParameters = $hp
  Environment = $envJson
}
$specPath = Join-Path $env:TEMP "$name.json"
[System.IO.File]::WriteAllText($specPath, ($spec | ConvertTo-Json -Depth 8), (New-Object System.Text.ASCIIEncoding))
& $aws sagemaker create-training-job --cli-input-json "file://$specPath" --region $region --profile $RanProfile --query TrainingJobArn --output text
if ($LASTEXITCODE -ne 0) { throw "create-training-job failed" }
Remove-Item $specPath -ErrorAction SilentlyContinue
"job: $name  ($InstanceType)"

$deadline = (Get-Date).AddMinutes($TimeoutMinutes + 20)
do {
  Start-Sleep -Seconds 30
  $st = & $aws sagemaker describe-training-job --training-job-name $name --region $region --profile $RanProfile --query "[TrainingJobStatus,SecondaryStatus,FailureReason]" --output text
  "{0}  {1}" -f (Get-Date -Format "HH:mm:ss"), $st
} while ((-not $st -or ($st -split "`t")[0] -in @("InProgress", "Stopping")) -and (Get-Date) -lt $deadline)

New-Item -ItemType Directory -Force $OutDir | Out-Null
$mt = Join-Path $OutDir "model.tar.gz"
& $aws s3 cp "s3://$bucket/outputs/$name/$name/output/model.tar.gz" $mt --region $region --profile $RanProfile --only-show-errors
if (Test-Path $mt) { & tar.exe -xzf $mt -C $OutDir; Remove-Item $mt }
"output folder: $OutDir  ->  " + ((Get-ChildItem $OutDir -Name) -join ", ")
