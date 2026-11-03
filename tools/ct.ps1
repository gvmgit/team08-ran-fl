# Launcher for the Crosstrainer utility (a Node program): sets PATH, the AWS profile and Region, optionally trusts a corporate
# proxy's root certificate for Node only, then runs the utility. See UTILITY_WINDOWS_PATCHES.md for the Windows changes to the utility.
#   powershell -File tools\ct.ps1 <crosstrainer-utility arguments>
#
#   RAN_TOOLS_DIR   folder that holds the crosstrainer-utility install        (default: %USERPROFILE%\tools)
#   RAN_AWS_PROFILE AWS CLI profile                                           (default: fyuz)
#   RAN_CORP_ROOT   optional PEM file with your web proxy's PUBLIC root certificate; passed to Node as NODE_EXTRA_CA_CERTS
#                   (certificate verification stays on)
$ErrorActionPreference = "Continue"
$tools = if ($env:RAN_TOOLS_DIR) { $env:RAN_TOOLS_DIR } else { Join-Path $env:USERPROFILE "tools" }
$env:Path = "C:\Program Files\nodejs;C:\Program Files\Amazon\AWSCLIV2;" + $env:Path
$env:AWS_PROFILE = if ($env:RAN_AWS_PROFILE) { $env:RAN_AWS_PROFILE } else { "fyuz" }
$env:AWS_REGION = "us-west-2"
$env:AWS_DEFAULT_REGION = "us-west-2"
if ($env:RAN_CORP_ROOT -and (Test-Path $env:RAN_CORP_ROOT)) { $env:NODE_EXTRA_CA_CERTS = $env:RAN_CORP_ROOT }
$env:NODE_NO_WARNINGS = "1"
Push-Location (Join-Path $tools "crosstrainer-utility")
& npx.cmd crosstrainer-utility @args 2>&1 | Out-String -Width 240
"exit code: $LASTEXITCODE"
Pop-Location
