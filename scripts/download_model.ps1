<#
.SYNOPSIS
    Download a complete Hugging Face model repository.
.DESCRIPTION
    Downloads every file at one resolved commit, preserving subdirectories.
    Uses hf (or huggingface-cli), with a resumable curl.exe fallback. Existing
    files are checked against Hub sizes and hashes, not just their presence.
.PARAMETER TokenizerRepo
    Fallback for tokenizer.json only when the model repo lacks it. Set to an
    empty string to disable. Never replaces the model's own tokenizer.
.PARAMETER Token
    Access token. Defaults to HF_TOKEN, then the Hugging Face login token file.
.PARAMETER Force
    Download fresh copies, ignoring existing files and partial downloads.
.EXAMPLE
    .\scripts\download_model.ps1 -Repo Ar4ikov/VibeVoice-ASR-AWQ-W4A16-ASYM
.EXAMPLE
    .\scripts\download_model.ps1 -Repo owner/model -Revision main -OutputDir D:\models\vv
#>
[CmdletBinding()]
param(
    [string]$OutputDir = "model_hf",
    [string]$Repo = "scerz/VibeVoice-ASR-4bit",
    [string]$TokenizerRepo = "Qwen/Qwen2.5-7B",
    [string]$Token = $env:HF_TOKEN,
    [string]$Revision = "main",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$PreviousToken = $env:HF_TOKEN
$Endpoint = "https://huggingface.co"
if ($env:HF_ENDPOINT) { $Endpoint = $env:HF_ENDPOINT.TrimEnd('/') }
$Headers = @{}

function Encode-Path([string]$Value) {
    return (($Value.Split('/') | ForEach-Object { [Uri]::EscapeDataString($_) }) -join '/')
}

function Get-Snapshot([string]$RepoId, [string]$Ref) {
    $url = "$Endpoint/api/models/$(Encode-Path $RepoId)/revision/$([Uri]::EscapeDataString($Ref))?blobs=true"
    $info = Invoke-RestMethod -Uri $url -Headers $Headers -TimeoutSec 60
    if ($info.sha -notmatch '^[a-fA-F0-9]{40}$' -or $null -eq $info.siblings) {
        throw "Hub returned an invalid file listing for $RepoId."
    }
    return $info
}

function Get-TargetPath([string]$Name) {
    $path = $OutputDir
    foreach ($part in $Name.Split('/')) {
        if (-not $part -or $part -in @('.', '..') -or
            $part.IndexOfAny([IO.Path]::GetInvalidFileNameChars()) -ge 0) {
            throw "Unsupported repository path: $Name"
        }
        $path = Join-Path $path $part
        if ((Test-Path -LiteralPath $path) -and
            ((Get-Item -LiteralPath $path -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw "Refusing to write through a symbolic link: $path"
        }
    }
    return $path
}

function Get-FreeSpace([string]$Path) {
    try { return (New-Object IO.DriveInfo ([IO.Path]::GetPathRoot([IO.Path]::GetFullPath($Path)))).AvailableFreeSpace }
    catch { return $null }
}

function Assert-FreeSpace([string]$Path, [long]$Needed) {
    # curl reports a full disk as a plain write failure, so refuse before asking.
    $free = Get-FreeSpace $Path
    if ($null -ne $free -and $free -lt $Needed) {
        throw ("Not enough free space on {0}: {1:N1} GB needed, {2:N1} GB available. " -f
               [IO.Path]::GetPathRoot([IO.Path]::GetFullPath($Path)), ($Needed / 1GB), ($free / 1GB)) +
              "Free up space or pass -OutputDir on another drive."
    }
}

function Test-Complete([string]$Path, $File) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    $length = (Get-Item -LiteralPath $Path -Force).Length
    if ($null -eq $File.size -or $length -ne [long]$File.size) { return $false }
    if ($File.lfs.sha256) {
        $sha = [Security.Cryptography.SHA256]::Create()
        $stream = [IO.File]::OpenRead($Path)
        try {
            return ([BitConverter]::ToString($sha.ComputeHash($stream)).Replace('-', '')) -eq $File.lfs.sha256
        } finally { $stream.Dispose(); $sha.Dispose() }
    }
    if (-not $File.blobId) { throw "Missing checksum for $($File.rfilename)." }
    # Git blob IDs hash the header as well as the file contents.
    $sha = [Security.Cryptography.SHA1]::Create()
    $stream = [IO.File]::OpenRead($Path)
    try {
        $header = [Text.Encoding]::UTF8.GetBytes("blob $length" + [char]0)
        [void]$sha.TransformBlock($header, 0, $header.Length, $header, 0)
        $buffer = New-Object byte[] (1MB)
        while (($count = $stream.Read($buffer, 0, $buffer.Length)) -gt 0) {
            [void]$sha.TransformBlock($buffer, 0, $count, $buffer, 0)
        }
        [void]$sha.TransformFinalBlock($buffer, 0, 0)
        return ([BitConverter]::ToString($sha.Hash).Replace('-', '')) -eq $File.blobId
    } finally { $stream.Dispose(); $sha.Dispose() }
}

function Download-File([string]$RepoId, [string]$Commit, $File, [bool]$Fresh) {
    if (-not $CurlPath) { throw "curl.exe is required to download/repair $($File.rfilename)." }
    $target = Get-TargetPath $File.rfilename
    # A commit-specific cache prevents resuming bytes from another revision.
    $key = [Security.Cryptography.SHA256]::Create()
    try {
        $id = [BitConverter]::ToString($key.ComputeHash([Text.Encoding]::UTF8.GetBytes("$RepoId/$($File.rfilename)"))).Replace('-', '')
    } finally { $key.Dispose() }
    $cache = Get-TargetPath ".cache/vibevoice-download/$Commit"
    [IO.Directory]::CreateDirectory($cache) | Out-Null
    $partial = Get-TargetPath ".cache/vibevoice-download/$Commit/$id.partial"
    if (Test-Path -LiteralPath $partial) {
        if ($Fresh -or (Get-Item -LiteralPath $partial).Length -gt [long]$File.size) {
            Remove-Item -LiteralPath $partial
        } elseif ((Get-Item -LiteralPath $partial).Length -eq [long]$File.size -and
                  -not (Test-Complete $partial $File)) {
            Remove-Item -LiteralPath $partial
        }
    }
    $have = if (Test-Path -LiteralPath $partial) { (Get-Item -LiteralPath $partial).Length } else { 0 }
    # The partial is moved onto the target, so the file is only paid for once.
    Assert-FreeSpace $cache ([long]$File.size - $have)
    $url = "$Endpoint/$(Encode-Path $RepoId)/resolve/$Commit/$(Encode-Path $File.rfilename)"
    Write-Host "[vibevoice] GET $($File.rfilename) ($($File.size) bytes)"
    for ($attempt = 0; $attempt -lt 2; $attempt++) {
        if (Test-Complete $partial $File) { break }
        $curlArgs = @('--fail', '--location', '--retry', '3', '--connect-timeout', '30',
                      '--progress-bar', '--continue-at', '-', '--output', $partial, $url)
        if ($Token) {
            # Keep the credential out of curl's command line. The config goes
            # through a file because PowerShell would give stdin a BOM, which
            # curl reads as part of the first option name.
            $config = 'header = "Authorization: Bearer ' + $Token.Replace('\', '\\').Replace('"', '\"') + '"'
            $configFile = [IO.Path]::Combine([IO.Path]::GetTempPath(), [IO.Path]::GetRandomFileName())
            try {
                [IO.File]::WriteAllText($configFile, $config, (New-Object Text.UTF8Encoding $false))
                & $CurlPath --config $configFile @curlArgs
            } finally { Remove-Item -LiteralPath $configFile -Force -ErrorAction SilentlyContinue }
        } else { & $CurlPath @curlArgs }
        $code = $LASTEXITCODE
        if ($code -eq 0 -and (Test-Complete $partial $File)) { break }
        # A server without Range support or a corrupt partial needs a fresh retry.
        if ($code -notin @(0, 33, 36) -or $attempt -eq 1) {
            if ($code -eq 23) {
                Assert-FreeSpace $cache ([long]$File.size)
                throw "Could not write $($File.rfilename): the download destination rejected the write."
            }
            throw "Download failed for $($File.rfilename) (curl $code). Re-run to resume."
        }
        if (Test-Path -LiteralPath $partial) { Remove-Item -LiteralPath $partial }
    }
    if (-not (Test-Complete $partial $File)) { throw "Checksum mismatch: $($File.rfilename)" }
    [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($target)) | Out-Null
    Move-Item -LiteralPath $partial -Destination $target -Force
}

try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    if (-not $Token) {
        $tokenPath = $env:HF_TOKEN_PATH
        if (-not $tokenPath) {
            $hfDir = $env:HF_HOME
            if (-not $hfDir) { $hfDir = Join-Path $HOME '.cache/huggingface' }
            $tokenPath = Join-Path $hfDir 'token'
        }
        if (Test-Path -LiteralPath $tokenPath -PathType Leaf) { $Token = [IO.File]::ReadAllText($tokenPath).Trim() }
    }
    if ($Token) {
        if ($Token -match '[\r\n]') { throw 'Invalid token: contains a newline.' }
        $Headers.Authorization = "Bearer $Token"
        $env:HF_TOKEN = $Token
    }
    $OutputDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDir)
    [IO.Directory]::CreateDirectory($OutputDir) | Out-Null
    $CurlPath = (Get-Command curl.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1).Source
    $cli = Get-Command hf, huggingface-cli -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    $snapshot = Get-Snapshot $Repo $Revision
    $files = @($snapshot.siblings)
    foreach ($file in $files) { $null = Get-TargetPath $file.rfilename }
    Write-Host "[vibevoice] Entire repository: $Repo @ $($snapshot.sha) ($($files.Count) files)"
    Write-Host "[vibevoice] Output: $OutputDir"
    $missing = [long](($files | Where-Object { -not (Test-Path -LiteralPath (Get-TargetPath $_.rfilename)) } |
                       Measure-Object -Property size -Sum).Sum)
    Assert-FreeSpace $OutputDir $missing
    $cliSucceeded = $false
    if ($cli) {
        $cliArgs = @('download', $Repo, '--revision', $snapshot.sha, '--local-dir', $OutputDir)
        if ($Force) { $cliArgs += '--force-download' }
        & $cli.Source @cliArgs
        $cliSucceeded = $LASTEXITCODE -eq 0
        if (-not $cliSucceeded) { Write-Warning 'Hugging Face CLI failed; continuing with curl.' }
    }
    foreach ($file in $files) {
        $target = Get-TargetPath $file.rfilename
        if (($Force -and -not $cliSucceeded) -or -not (Test-Complete $target $file)) {
            Download-File $Repo $snapshot.sha $file ([bool]$Force)
        } else { Write-Host "[vibevoice] OK $($file.rfilename)" }
    }
    if ($TokenizerRepo -and 'tokenizer.json' -notin @($files.rfilename)) {
        Write-Host "[vibevoice] No tokenizer.json in $Repo; fetching fallback from $TokenizerRepo."
        $fallback = Get-Snapshot $TokenizerRepo 'main'
        $tokenizer = $fallback.siblings | Where-Object { $_.rfilename -eq 'tokenizer.json' } | Select-Object -First 1
        if (-not $tokenizer) { throw "No tokenizer.json in fallback repository $TokenizerRepo." }
        if ($Force -or -not (Test-Complete (Get-TargetPath 'tokenizer.json') $tokenizer)) {
            Download-File $TokenizerRepo $fallback.sha $tokenizer ([bool]$Force)
        }
    }
    Write-Host "[vibevoice] Complete repository downloaded and verified."
} catch {
    Write-Host "[vibevoice] ERROR: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
} finally { $env:HF_TOKEN = $PreviousToken }
