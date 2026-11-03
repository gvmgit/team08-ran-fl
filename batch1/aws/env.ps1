# Shared settings for the scripts in this folder. Dot-source it:   . "$PSScriptRoot\env.ps1"
# Everything can be overridden with environment variables. Nothing in here is a secret.
#
#   RAN_AWS_PROFILE   AWS CLI profile that holds the team's SSO login             (default: fyuz)
#   RAN_TOOLS_DIR     folder with ct.ps1 and the crosstrainer-utility install     (default: %USERPROFILE%\tools)
#   RAN_PRIVATE_DIR   folder for the collaboration CA and client keys; keep it OUTSIDE the repo
#                                                                                 (default: %USERPROFILE%\ran-fl-private)
#   RAN_CA_BUNDLE     optional PEM file (Amazon roots + your web proxy's root). Only needed when a corporate proxy
#                     re-signs TLS to AWS; it is applied to this process only, certificate checks stay on.
$RanProfile = if ($env:RAN_AWS_PROFILE) { $env:RAN_AWS_PROFILE } else { "fyuz" }
$RanTools = if ($env:RAN_TOOLS_DIR) { $env:RAN_TOOLS_DIR } else { Join-Path $env:USERPROFILE "tools" }
$RanPrivate = if ($env:RAN_PRIVATE_DIR) { $env:RAN_PRIVATE_DIR } else { Join-Path $env:USERPROFILE "ran-fl-private" }
$RanAws = if (Test-Path "C:\Program Files\Amazon\AWSCLIV2\aws.exe") { "C:\Program Files\Amazon\AWSCLIV2\aws.exe" } else { "aws" }
if ($env:RAN_CA_BUNDLE -and (Test-Path $env:RAN_CA_BUNDLE)) { $env:AWS_CA_BUNDLE = $env:RAN_CA_BUNDLE }
# the account the profile is logged in to (the team account); fails early if the SSO session has expired
$RanAccount = (& $RanAws sts get-caller-identity --profile $RanProfile --query Account --output text)
if ($LASTEXITCODE -ne 0 -or -not $RanAccount) { throw "could not read the AWS account; run 'aws sso login --profile $RanProfile --use-device-code' first" }
