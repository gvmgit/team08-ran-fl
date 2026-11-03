# Assembles the folder that run_processing.ps1 / run_training_job.ps1 upload and run on AWS, from files in this repo plus the
# two things that are not in it: the Crosstrainer training-client wheel (organisers' distribution bucket, checked against the
# published SHA-512) and the utility's own validate_model.py.
#
#   make_job_folder.ps1 -Kind taska-eval   -Config aws\configs\taskA_central.json -Dest <dir>
#   make_job_folder.ps1 -Kind taska-validate -Dest <dir>
#   make_job_folder.ps1 -Kind taskb-grader -Config aws\configs\taskB_baselines.json -InitialAdapter <initial_adapter.safetensors> -Dest <dir>
#   make_job_folder.ps1 -Kind taskb-validate -InitialAdapter <initial_adapter.safetensors> -Dest <dir>
#   make_job_folder.ps1 -Kind taskb-initial-adapter -Dest <dir>
#   make_job_folder.ps1 -Kind taskb-make-text -Config aws\configs\taskB_make_text.json -Dest <dir>
#
# The configs in aws\configs are the ones the reported runs used (bucket names carry the team account id).
param(
  [Parameter(Mandatory)] [ValidateSet("taska-validate", "taska-eval", "taskb-validate", "taskb-grader", "taskb-initial-adapter", "taskb-make-text")] [string] $Kind,
  [Parameter(Mandatory)] [string] $Dest,
  [string] $Config = "",
  [string] $InitialAdapter = "",
  [string] $Utility = $(Join-Path $(if ($env:RAN_TOOLS_DIR) { $env:RAN_TOOLS_DIR } else { Join-Path $env:USERPROFILE "tools" }) "crosstrainer-utility")
)
$ErrorActionPreference = "Stop"
if ($env:RAN_CA_BUNDLE -and (Test-Path $env:RAN_CA_BUNDLE)) { $env:AWS_CA_BUNDLE = $env:RAN_CA_BUNDLE }   # only with a TLS-inspecting proxy
$repo = Split-Path $PSScriptRoot -Parent
$aws = if (Test-Path "C:\Program Files\Amazon\AWSCLIV2\aws.exe") { "C:\Program Files\Amazon\AWSCLIV2\aws.exe" } else { "aws" }
$profileName = if ($env:AWS_PROFILE) { $env:AWS_PROFILE } else { "fyuz" }
$wheelName = "crosstrainer_training_client_bundled-0.1.0-py3-none-any.whl"
$wheelCache = Join-Path $env:TEMP "crosstrainer-wheel"

function Get-Wheel {
  New-Item -ItemType Directory -Force $wheelCache | Out-Null
  $whl = Join-Path $wheelCache $wheelName
  if (-not (Test-Path $whl)) {
    foreach ($f in $wheelName, "sha512sums.txt") {
      & $aws s3 cp "s3://crosstrainer-training-client-605383994221-prod-pdx/wheel/$f" (Join-Path $wheelCache $f) --region us-west-2 --profile $profileName --only-show-errors
      if ($LASTEXITCODE -ne 0) { throw "could not download $f" }
    }
  }
  $line = Get-Content (Join-Path $wheelCache "sha512sums.txt") | Where-Object { $_ -match [regex]::Escape($wheelName) } | Select-Object -First 1
  if (-not $line) { throw "sha512sums.txt has no entry for $wheelName" }
  $want = ($line -split "\s+")[0].ToLower()
  $got = (Get-FileHash $whl -Algorithm SHA512).Hash.ToLower()
  if ($got -ne $want) { throw "wheel SHA-512 does not match the published value" }
  return $whl
}
function Need($path, $what) { if (-not $path -or -not (Test-Path $path)) { throw "$what is required for -Kind $Kind" } }

if (Test-Path $Dest) { Remove-Item $Dest -Recurse -Force }
New-Item -ItemType Directory -Force $Dest | Out-Null
Copy-Item (Get-Wheel) $Dest
$validator = Join-Path $Utility "node_modules\@amzn\crosstrainer-user-utilities\assets\validate_model.py"

function Copy-ModelCode($from) {
  New-Item -ItemType Directory -Force "$Dest\model-code" | Out-Null
  Copy-Item "$from\model.py", "$from\requirements.txt" "$Dest\model-code"
}

switch ($Kind) {
  "taska-validate" {
    Need $validator "the utility's validate_model.py (-Utility)"
    Copy-Item "$repo\aws\validate_job.py", $validator, "$repo\initial.safetensors" $Dest
    Copy-ModelCode "$repo\model-code"
  }
  "taska-eval" {
    Need $Config "-Config"
    Copy-Item "$repo\aws\eval_job.py", "$repo\initial.safetensors" $Dest
    Copy-Item $Config "$Dest\job_config.json"
    Copy-ModelCode "$repo\model-code"
  }
  "taskb-validate" {
    Need $validator "the utility's validate_model.py (-Utility)"
    Need $InitialAdapter "-InitialAdapter"
    Copy-Item "$repo\taskb\taskb_validate_job.py", $validator $Dest
    Copy-Item $InitialAdapter "$Dest\initial_adapter.safetensors"
    Copy-ModelCode "$repo\taskb\model-code"
  }
  "taskb-grader" {
    Need $Config "-Config"
    Need $InitialAdapter "-InitialAdapter"
    Copy-Item "$repo\taskb\taskb_job.py" $Dest
    Copy-Item $Config "$Dest\job_config.json"
    Copy-Item $InitialAdapter "$Dest\initial_adapter.safetensors"
    Copy-Item "$repo\taskb\model-code\requirements.txt" "$Dest\requirements.txt"
    Copy-ModelCode "$repo\taskb\model-code"
  }
  "taskb-initial-adapter" {
    Copy-Item "$repo\taskb\make_initial_adapter.py" $Dest
    Get-Content "$repo\evidence\taskb_test\site_a_test.jsonl" -TotalCount 12 | Set-Content "$Dest\sample_test.jsonl" -Encoding ascii
    Copy-ModelCode "$repo\taskb\model-code"
  }
  "taskb-make-text" {
    Need $Config "-Config"
    Remove-Item "$Dest\$wheelName"
    Copy-Item "$repo\taskb\make_text.py" $Dest
    Copy-Item $Config "$Dest\text_config.json"
  }
}
"job folder ready: $Dest"
Get-ChildItem $Dest -Recurse -File | ForEach-Object { "  " + $_.FullName.Substring($Dest.Length + 1) + "  (" + $_.Length + " bytes)" }
