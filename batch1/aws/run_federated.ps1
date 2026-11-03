# One command for a federated Task A run: declare, create the session, join, issue both client certificates, launch both
# clients in parallel on GPU, wait. Same code, same stack, same two secrets as the main run.
#   run_federated.ps1 -RunPrefix ran-stress-fed-dp -Declaration <template.json> -WithholdA mobility -WithholdB congestion
param(
  [Parameter(Mandatory)] [string] $RunPrefix,
  [Parameter(Mandatory)] [string] $Declaration,
  [string] $WithholdA = "",
  [string] $WithholdB = "",
  [string] $InstanceA = "ml.g6.2xlarge",
  [string] $InstanceB = "ml.g5.2xlarge",
  [int] $Iterations = 3,
  [string] $ModelCodeDir = "",                                  # default: <repo>\model-code (Task A)
  [string] $InitialWeights = "",                                # default: <repo>\initial.safetensors (Task A)
  [string] $Protocol = "NN_HFL_WITH_DP",
  [string] $DataKeyTemplate = "data/site_{s}_cell_minute.csv",  # empty = do not pass RAN_DATA_KEY (Task B reads text/ itself)
  [string[]] $ExtraEnv = @()                                    # KEY=VALUE passed to both clients
)
$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
. "$PSScriptRoot\env.ps1"   # profile, account, tools and private folders, optional CA bundle for a TLS-inspecting proxy
$ct = Join-Path $repo "tools\ct.ps1"   # launcher for the Crosstrainer utility (install folder: $RanTools\crosstrainer-utility)
$priv = $RanPrivate
$tools = $RanTools
$aws = $RanAws
$acct = $RanAccount
$evidenceBucket = "bb26-p-team08-data-$acct-use2"
$gpu = "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-sagemaker"
$arns = Get-Content "$priv\secret-arns.txt"
$runId = "$RunPrefix-" + (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmm")
"RUN_ID = $runId"

# 1. Declare first. If this fails nothing starts.
$text = (Get-Content $Declaration -Raw).Replace("__RUN_ID__", $runId)
$tmp = Join-Path $env:TEMP "$runId.declared.json"
[System.IO.File]::WriteAllText($tmp, $text, (New-Object System.Text.ASCIIEncoding))
& $aws s3 cp $tmp "s3://$evidenceBucket/evidence/declared/$runId.json" --region us-east-2 --profile $RanProfile --only-show-errors
if ($LASTEXITCODE -ne 0) { throw "declaration upload failed; not starting the run" }
New-Item -ItemType Directory -Force "$repo\evidence\declared" | Out-Null
Copy-Item $tmp "$repo\evidence\declared\$runId.json" -Force
"declared: s3://$evidenceBucket/evidence/declared/$runId.json"

# 2. Session
if (-not $ModelCodeDir) { $ModelCodeDir = "$repo\model-code" }
if (-not $InitialWeights) { $InitialWeights = "$repo\initial.safetensors" }
$out = & $ct create-session --collaboration-name ran-ews --session-name $runId --model-code $ModelCodeDir --initial-weights $InitialWeights --hflconfiguration model.py --protocol $Protocol --iterations $Iterations --required-clients 2 --auto-confirm --skip-validation --region us-west-2
$m = [regex]::Match(($out -join "`n"), 'Session ID:\s+([0-9a-f]+)')
if (-not $m.Success) { $out | Select-Object -Last 40; throw "create-session failed" }
$sid = $m.Groups[1].Value
"SESSION ID = $sid"
Set-Content "$tools\${RunPrefix}_session_id.txt" $sid -Encoding ASCII
Set-Content "$tools\${RunPrefix}_run_id.txt" $runId -Encoding ASCII

# 3. Join, then 4. one certificate per client (the certificates name this session)
$out = & $ct create-membership --session-id $sid --silo-name ran --mode sagemaker --auto-confirm
if (($out -join "`n") -notmatch 'Status:\s+ACTIVE') { $out | Select-Object -Last 30; throw "create-membership failed" }
"membership ACTIVE"
for ($i = 1; $i -le 2; $i++) {
  & powershell -NoProfile -ExecutionPolicy Bypass -File "$repo\tools\make_certs.ps1" issue -Client $i -SessionId $sid -SecretArn $arns[$i - 1] 2>&1 | Where-Object { $_ -match 'FAIL|stored|failed|issued' }
  if ($LASTEXITCODE -ne 0) { throw "certificate issue failed for client $i" }
}

# 5. Both clients in parallel, each in its own process
function Start-Client($label, $siloLetter, $index, $instance, $secretArn, $withhold) {
  $args_ = "-NoProfile -ExecutionPolicy Bypass -File `"$ct`" start-training --mode sagemaker --session-id $sid --silo-name ran --client-name ran-client-$siloLetter " +
    "--dataset-bucket bb26-p-team08-silo-$siloLetter-$acct-usw2 --env CLIENT_INDEX=$index --env RUN_ID=$runId " +
    $(if ($DataKeyTemplate) { "--env RAN_DATA_KEY=" + $DataKeyTemplate.Replace("{s}", $siloLetter) + " " } else { "" }) +
    $(if ($withhold) { "--env RAN_WITHHOLD_LABEL=$withhold " } else { "" }) +
    (($ExtraEnv | ForEach-Object { "--env $_ " }) -join "") +
    "--instance-type $instance --training-image $gpu --volume-size 100 --key-material-secret-arn $secretArn --trust-anchors `"$priv\collaboration-cas.pem`" --auto-confirm"
  return Start-Process -FilePath powershell.exe -ArgumentList $args_ -RedirectStandardOutput "$tools\${RunPrefix}_client_$siloLetter.log" -RedirectStandardError "$tools\${RunPrefix}_client_$siloLetter.err" -PassThru -WindowStyle Hidden
}
$pa = Start-Client "A" "a" 1 $InstanceA $arns[0] $WithholdA
$pb = Start-Client "B" "b" 2 $InstanceB $arns[1] $WithholdB
"clients launched (pids $($pa.Id), $($pb.Id)); logs: $tools\${RunPrefix}_client_a.log / _b.log"
Wait-Process -Id $pa.Id, $pb.Id
"clients exited: A=$($pa.ExitCode) B=$($pb.ExitCode)"
(& $ct status --session-id $sid) -split "`n" | Select-Object -Skip 3 -First 16
