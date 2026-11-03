# Certificate route for the Crosstrainer clients, on Windows (no openssl needed). Uses .NET for the certificates and
# Node for an independent check. Private material stays OUTSIDE the repo, in $Priv.
#
#   make_certs.ps1 ca                                   one CA (P-384) for the collaboration; writes collaboration-cas.pem
#   make_certs.ps1 issue -Client 1 -SessionId <sid> [-SecretArn <arn>] [-LocalOnly]
#                                                       P-521 key + certificate for one client; checks it; stores key
#                                                       and chain in that client's Secrets Manager secret
#   make_certs.ps1 inspect -Client 1                    re-run the checks on an issued certificate
#
# Requirements come from the Crosstrainer USER_GUIDE.md ("Creating a key-material secret"): the CA is a critical CA:TRUE
# with keyCertSign and a subject key identifier; the client certificate is not a CA, has a P-521 key, an authority key
# identifier, the clientAuth usage, exactly one account SAN URI and exactly one session SAN URI; the chain is PEM,
# end-entity first, no root, under 32 KiB; the secret is {"privateKey": "...", "certificateChain": "..."}.
param(
  [Parameter(Mandatory, Position = 0)] [ValidateSet("ca", "issue", "inspect")] [string] $Command,
  [int] $Client = 0,
  [string] $SessionId = "",
  [string] $Account = "",   # default: the account the AWS profile is logged in to
  [string] $SecretArn = "",
  [string] $Region = "us-west-2",
  [int] $CaDays = 10,
  [int] $LeafDays = 7,
  [switch] $LocalOnly
)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Security
$Priv = if ($env:RAN_PRIVATE_DIR) { $env:RAN_PRIVATE_DIR } else { Join-Path $env:USERPROFILE "ran-fl-private" }
New-Item -ItemType Directory -Force $Priv | Out-Null
$aws = if (Test-Path "C:\Program Files\Amazon\AWSCLIV2\aws.exe") { "C:\Program Files\Amazon\AWSCLIV2\aws.exe" } else { "aws" }
$node = if (Test-Path "C:\Program Files\nodejs\node.exe") { "C:\Program Files\nodejs\node.exe" } else { "node" }
$prof = if ($env:RAN_AWS_PROFILE) { $env:RAN_AWS_PROFILE } else { "fyuz" }
if ($env:RAN_CA_BUNDLE -and (Test-Path $env:RAN_CA_BUNDLE)) { $env:AWS_CA_BUNDLE = $env:RAN_CA_BUNDLE }

# ---- tiny DER writer (for the SEC1 EC private key) ---------------------------------------------------------------
function Join-Bytes([object[]] $parts) {
  $ms = New-Object System.IO.MemoryStream
  foreach ($p in $parts) { $b = [byte[]]$p; $ms.Write($b, 0, $b.Length) }
  return , $ms.ToArray()
}
function Der-Len([int] $n) {
  if ($n -lt 128) { return , ([byte[]]@($n)) }
  if ($n -lt 256) { return , ([byte[]]@(0x81, $n)) }
  return , ([byte[]]@(0x82, [byte]($n -shr 8), [byte]($n -band 255)))
}
function Der([int] $tag, [byte[]] $content) { return , (Join-Bytes @(([byte[]]@($tag)), (Der-Len $content.Length), $content)) }
function Pad([byte[]] $b, [int] $len) { if ($b.Length -ge $len) { return , $b }; return , (Join-Bytes @((New-Object byte[] ($len - $b.Length)), $b)) }
function To-Pem([byte[]] $der, [string] $label) {
  $b64 = [Convert]::ToBase64String($der)
  $lines = for ($i = 0; $i -lt $b64.Length; $i += 64) { $b64.Substring($i, [Math]::Min(64, $b64.Length - $i)) }
  return "-----BEGIN $label-----`n" + ($lines -join "`n") + "`n-----END $label-----`n"
}
function EcKey-ToSec1Pem($ecdsa, [byte[]] $curveOidDer, [int] $coordLen) {
  $p = $ecdsa.ExportParameters($true)
  $d = Pad $p.D $coordLen
  $pub = Join-Bytes @(([byte[]]@(0x04)), (Pad $p.Q.X $coordLen), (Pad $p.Q.Y $coordLen))
  $inner = Join-Bytes @(
    (Der 0x02 ([byte[]]@(1))),
    (Der 0x04 $d),
    (Der 0xA0 $curveOidDer),
    (Der 0xA1 (Der 0x03 (Join-Bytes @(([byte[]]@(0x00)), $pub))))
  )
  return To-Pem (Der 0x30 $inner) "EC PRIVATE KEY"
}
$OidP384 = [byte[]]@(0x06, 0x05, 0x2B, 0x81, 0x04, 0x00, 0x22)
$OidP521 = [byte[]]@(0x06, 0x05, 0x2B, 0x81, 0x04, 0x00, 0x23)

$X = "System.Security.Cryptography.X509Certificates"
function New-Req([string] $subject, $key) {
  return New-Object "$X.CertificateRequest" -ArgumentList $subject, $key, ([System.Security.Cryptography.HashAlgorithmName]::SHA384)
}
function Load-Ca {
  $pfx = Join-Path $Priv "ca.pfx"; $pwd = (Get-Content (Join-Path $Priv "ca.pwd") -Raw).Trim()
  if (-not (Test-Path $pfx)) { throw "no CA yet; run: make_certs.ps1 ca" }
  return New-Object "$X.X509Certificate2" -ArgumentList $pfx, $pwd, ([System.Security.Cryptography.X509Certificates.X509KeyStorageFlags]::Exportable)
}
function Pem-Of([System.Security.Cryptography.X509Certificates.X509Certificate2] $c) { return To-Pem $c.RawData "CERTIFICATE" }

# ---- node-based independent checks ---------------------------------------------------------------------------------
function Node-Check([string] $caPem, [string] $leafPem) {
  $js = @'
const { X509Certificate } = require("node:crypto");
const fs = require("fs");
const ca = new X509Certificate(fs.readFileSync(process.argv[2]));
const leaf = new X509Certificate(fs.readFileSync(process.argv[3]));
const checks = [];
const add = (name, ok, detail) => checks.push([ok ? "PASS" : "FAIL", name, detail || ""]);
add("CA is a CA", ca.ca === true);
add("leaf is not a CA", leaf.ca === false);
add("leaf signed by CA", leaf.verify(ca.publicKey) === true);
add("leaf issuer matches CA subject", leaf.checkIssued(ca) === true);
add("leaf key is P-521", leaf.publicKey.asymmetricKeyDetails.namedCurve === "secp521r1", leaf.publicKey.asymmetricKeyDetails.namedCurve);
add("CA key is P-384", ca.publicKey.asymmetricKeyDetails.namedCurve === "secp384r1", ca.publicKey.asymmetricKeyDetails.namedCurve);
const san = (leaf.subjectAltName || "").split(",").map(s => s.trim()).filter(Boolean);
const acct = san.filter(s => s.startsWith("URI:urn:crosstrainer:account:"));
const sess = san.filter(s => s.startsWith("URI:urn:crosstrainer:session:"));
add("exactly one account SAN URI", acct.length === 1, acct.join(" "));
add("exactly one session SAN URI", sess.length === 1, sess.join(" "));
add("no other SAN entries", san.length === acct.length + sess.length, String(san.length));
add("clientAuth extended key usage", (leaf.keyUsage || []).includes("1.3.6.1.5.5.7.3.2"), (leaf.keyUsage || []).join(","));
const now = Date.now();
add("leaf in date", Date.parse(leaf.validFrom) <= now && now <= Date.parse(leaf.validTo), leaf.validFrom + " -> " + leaf.validTo);
add("CA in date", Date.parse(ca.validFrom) <= now && now <= Date.parse(ca.validTo), ca.validFrom + " -> " + ca.validTo);
for (const [s, n, d] of checks) console.log(`${s}  ${n}${d ? "  [" + d + "]" : ""}`);
process.exit(checks.some(c => c[0] === "FAIL") ? 1 : 0);
'@
  $f = Join-Path $env:TEMP "ct_check_$([guid]::NewGuid().ToString('N')).js"
  [System.IO.File]::WriteAllText($f, $js, (New-Object System.Text.ASCIIEncoding))
  & $node $f $caPem $leafPem | Out-Host
  $code = $LASTEXITCODE
  Remove-Item $f -ErrorAction SilentlyContinue
  return $code
}
function Dotnet-ExtCheck([string] $leafPemPath) {
  $c = New-Object "$X.X509Certificate2" -ArgumentList $leafPemPath
  $oids = $c.Extensions | ForEach-Object { $_.Oid.Value }
  foreach ($pair in @(@("2.5.29.35", "authority key identifier"), @("2.5.29.37", "extended key usage"), @("2.5.29.17", "subject alt name"), @("2.5.29.19", "basic constraints"), @("2.5.29.15", "key usage"))) {
    $has = $oids -contains $pair[0]
    "{0}  .NET sees {1} ({2})" -f $(if ($has) { "PASS" } else { "FAIL" }), $pair[1], $pair[0]
    if (-not $has) { $script:extFail = $true }
  }
}

# ---- commands -------------------------------------------------------------------------------------------------------
switch ($Command) {
  "ca" {
    if (Test-Path (Join-Path $Priv "ca.pfx")) { "CA already exists; keeping it." }
    else {
      # Curves are built from explicit OIDs: the NamedCurves shortcuts come back empty in Windows PowerShell 5.1.
      $key = New-Object System.Security.Cryptography.ECDsaCng -ArgumentList ([System.Security.Cryptography.ECCurve]::CreateFromValue("1.3.132.0.34"))
      $req = New-Req "CN=team08-fl-ca" $key
      $req.CertificateExtensions.Add((New-Object "$X.X509BasicConstraintsExtension" -ArgumentList $true, $false, 0, $true))
      $req.CertificateExtensions.Add((New-Object "$X.X509KeyUsageExtension" -ArgumentList ([System.Security.Cryptography.X509Certificates.X509KeyUsageFlags]::KeyCertSign -bor [System.Security.Cryptography.X509Certificates.X509KeyUsageFlags]::CrlSign), $true))
      $req.CertificateExtensions.Add((New-Object "$X.X509SubjectKeyIdentifierExtension" -ArgumentList $req.PublicKey, $false))
      $cert = $req.CreateSelfSigned([DateTimeOffset]::UtcNow.AddMinutes(-5), [DateTimeOffset]::UtcNow.AddDays($CaDays))
      $pwd = [Convert]::ToBase64String((New-Object byte[] 24 | ForEach-Object { Get-Random -Maximum 256 }))
      [System.IO.File]::WriteAllBytes((Join-Path $Priv "ca.pfx"), $cert.Export([System.Security.Cryptography.X509Certificates.X509ContentType]::Pfx, $pwd))
      [System.IO.File]::WriteAllText((Join-Path $Priv "ca.pwd"), $pwd, (New-Object System.Text.ASCIIEncoding))
    }
    $ca = Load-Ca
    $pem = Pem-Of $ca
    [System.IO.File]::WriteAllText((Join-Path $Priv "collaboration-cas.pem"), $pem, (New-Object System.Text.ASCIIEncoding))
    "CA subject:     " + $ca.Subject
    "CA thumbprint:  " + $ca.Thumbprint
    "CA valid until: " + $ca.NotAfter.ToUniversalTime().ToString("u")
    "trust anchors:  " + (Join-Path $Priv "collaboration-cas.pem") + "  (one certificate)"
  }
  { $_ -in "issue", "inspect" } {
    if ($Client -lt 1) { throw "-Client is required" }
    $dir = Join-Path $Priv "client$Client"
    New-Item -ItemType Directory -Force $dir | Out-Null
    $ca = Load-Ca
    $caPemPath = Join-Path $Priv "collaboration-cas.pem"
    $chainPath = Join-Path $dir "client.chain.pem"
    if ($Command -eq "issue") {
      if (-not $SessionId) { throw "-SessionId is required (the certificate names the session)" }
      if (-not $Account) { $Account = (& $aws sts get-caller-identity --profile $prof --query Account --output text) }
      if ($Account -notmatch '^\d{12}$') { throw "account must be 12 digits" }
      $key = New-Object System.Security.Cryptography.ECDsaCng -ArgumentList ([System.Security.Cryptography.ECCurve]::CreateFromValue("1.3.132.0.35"))
      $req = New-Req "CN=crosstrainer-client" $key
      $req.CertificateExtensions.Add((New-Object "$X.X509BasicConstraintsExtension" -ArgumentList $false, $false, 0, $true))
      $req.CertificateExtensions.Add((New-Object "$X.X509KeyUsageExtension" -ArgumentList ([System.Security.Cryptography.X509Certificates.X509KeyUsageFlags]::DigitalSignature -bor [System.Security.Cryptography.X509Certificates.X509KeyUsageFlags]::KeyAgreement), $true))
      $eku = New-Object System.Security.Cryptography.OidCollection
      [void]$eku.Add((New-Object System.Security.Cryptography.Oid "1.3.6.1.5.5.7.3.2"))
      $req.CertificateExtensions.Add((New-Object "$X.X509EnhancedKeyUsageExtension" -ArgumentList $eku, $false))
      $san = New-Object "$X.SubjectAlternativeNameBuilder"
      $san.AddUri([Uri]"urn:crosstrainer:account:$Account")
      $san.AddUri([Uri]"urn:crosstrainer:session:$SessionId")
      $req.CertificateExtensions.Add($san.Build($false))
      $ski = ($ca.Extensions | Where-Object { $_.Oid.Value -eq "2.5.29.14" } | Select-Object -First 1).SubjectKeyIdentifier
      $skiBytes = [byte[]]@(for ($i = 0; $i -lt $ski.Length; $i += 2) { [Convert]::ToByte($ski.Substring($i, 2), 16) })
      $akiRaw = Der 0x30 (Der 0x80 $skiBytes)
      $req.CertificateExtensions.Add((New-Object "$X.X509Extension" -ArgumentList "2.5.29.35", $akiRaw, $false))
      $serial = New-Object byte[] 16
      [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($serial)
      $serial[0] = $serial[0] -band 0x7F
      $leaf = $req.Create($ca, [DateTimeOffset]::UtcNow.AddMinutes(-5), [DateTimeOffset]::UtcNow.AddDays($LeafDays), $serial)
      $chainPem = Pem-Of $leaf
      [System.IO.File]::WriteAllText($chainPath, $chainPem, (New-Object System.Text.ASCIIEncoding))
      $keyPem = EcKey-ToSec1Pem $key $OidP521 66
      $secretPath = Join-Path $dir "client.secret.json"
      $json = [ordered]@{ privateKey = $keyPem; certificateChain = $chainPem } | ConvertTo-Json -Compress
      [System.IO.File]::WriteAllText($secretPath, $json, (New-Object System.Text.ASCIIEncoding))
      "issued: " + $leaf.Subject + "  thumbprint " + $leaf.Thumbprint
      "chain size (bytes): " + $chainPem.Length + " (limit 32768)"
    }
    "== checks (Node) =="
    $code = Node-Check $caPemPath $chainPath
    "== checks (.NET) =="
    $script:extFail = $false
    Dotnet-ExtCheck $chainPath
    if ($code -ne 0 -or $script:extFail) { if (Test-Path (Join-Path $dir "client.secret.json")) { Remove-Item (Join-Path $dir "client.secret.json") }; throw "certificate checks failed; nothing was stored" }
    if ($Command -eq "issue") {
      $secretPath = Join-Path $dir "client.secret.json"
      if ($LocalOnly -or -not $SecretArn) { "local-only: not uploaded. Secret JSON kept at $secretPath (delete it when done)." }
      else {
        & $aws secretsmanager put-secret-value --secret-id $SecretArn --secret-string "file://$secretPath" --region $Region --profile $prof --query VersionId --output text
        if ($LASTEXITCODE -ne 0) { throw "put-secret-value failed" }
        Remove-Item $secretPath -Force
        "stored key + chain in the secret; local secret file removed."
      }
    }
  }
}
