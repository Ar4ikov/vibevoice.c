<#
.SYNOPSIS
    Download VibeVoice-ASR model from HuggingFace.

.DESCRIPTION
    Downloads the 4-bit NF4 quantized model weights and tokenizer files
    needed to run vv_cli.exe. Uses huggingface-cli if available, otherwise
    falls back to curl.exe (ships with Windows 10+).

.PARAMETER OutputDir
    Target directory for model files. Default: .\model_hf

.PARAMETER Repo
    HuggingFace repo for model weights. Default: scerz/VibeVoice-ASR-4bit

.PARAMETER TokenizerRepo
    HuggingFace repo for tokenizer.json (not included in the 4-bit repo).
    VibeVoice uses Qwen2.5 tokenizer as base, patched with audio tokens.
    Default: Qwen/Qwen2.5-7B

.PARAMETER Token
    HuggingFace access token (for gated/private repos). Optional.

.PARAMETER Force
    Re-download files even if they already exist.

.EXAMPLE
    .\scripts\download_model.ps1
    .\scripts\download_model.ps1 -OutputDir D:\models\vibevoice
    .\scripts\download_model.ps1 -Token hf_xxxxxxxxxxxx
#>
[CmdletBinding()]
param(
    [string]$OutputDir    = "model_hf",
    [string]$Repo         = "scerz/VibeVoice-ASR-4bit",
    [string]$TokenizerRepo = "Qwen/Qwen2.5-7B",
    [string]$Token        = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

# ─── File manifest ────────────────────────────────────────────────────────────
# Files from the 4-bit repo
$ModelFiles = @(
    @{ Name = "config.json";                       Size = "4 KB"   }
    @{ Name = "generation_config.json";            Size = "73 B"   }
    @{ Name = "preprocessor_config.json";          Size = "189 B"  }
    @{ Name = "model.safetensors.index.json";      Size = "232 KB" }
    @{ Name = "model-00001-of-00002.safetensors";  Size = "4.97 GB"}
    @{ Name = "model-00002-of-00002.safetensors";  Size = "2.69 GB"}
)

# Tokenizer file from the base model repo
$TokenizerFiles = @(
    @{ Name = "tokenizer.json"; Size = "~7 MB" }
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

function Write-Header {
    param([string]$Text)
    Write-Host ""
    Write-Host "========================================================================" -ForegroundColor Cyan
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host "========================================================================" -ForegroundColor Cyan
}

function Write-Step {
    param([string]$Text)
    Write-Host "[vibevoice] $Text" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Text)
    Write-Host "[vibevoice] WARNING: $Text" -ForegroundColor Yellow
}

function Write-Err {
    param([string]$Text)
    Write-Host "[vibevoice] ERROR: $Text" -ForegroundColor Red
}

function Test-FileComplete {
    param([string]$Path)
    # Consider file present if it exists and is non-empty
    return (Test-Path $Path) -and ((Get-Item $Path).Length -gt 0)
}

# ─── Detect download method ──────────────────────────────────────────────────

$UseHfCli = $false
$HfCliPath = $null
$CurlPath = $null

# Check for huggingface-cli
$HfCliPath = Get-Command "huggingface-cli" -ErrorAction SilentlyContinue |
             Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue
if ($HfCliPath) {
    $UseHfCli = $true
    Write-Step "Using huggingface-cli: $HfCliPath"
} else {
    # Fall back to curl.exe (ships with Windows 10 1803+)
    $CurlPath = Get-Command "curl.exe" -ErrorAction SilentlyContinue |
                Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue
    if (-not $CurlPath) {
        Write-Err "Neither huggingface-cli nor curl.exe found."
        Write-Err "Install huggingface-hub:  pip install huggingface-hub"
        Write-Err "Or ensure curl.exe is in PATH (ships with Windows 10+)."
        exit 1
    }
    Write-Step "Using curl.exe: $CurlPath"
    if (-not $Token) {
        Write-Warn "No --Token provided. If the repo is gated, set -Token hf_xxx"
    }
}

# ─── Prepare output directory ─────────────────────────────────────────────────

$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    Write-Step "Created output directory: $OutputDir"
} else {
    Write-Step "Output directory: $OutputDir"
}

# ─── Download functions ───────────────────────────────────────────────────────

function Download-WithHfCli {
    param(
        [string]$RepoId,
        [string[]]$Files,
        [string]$TargetDir
    )
    $args_ = @("download", $RepoId)
    $args_ += $Files
    $args_ += @("--local-dir", $TargetDir)

    if ($Token) {
        $args_ += @("--token", $Token)
    }

    Write-Step "huggingface-cli download $RepoId -> $TargetDir"
    Write-Step "  Files: $($Files -join ', ')"

    & huggingface-cli @args_
    if ($LASTEXITCODE -ne 0) {
        Write-Err "huggingface-cli failed (exit code $LASTEXITCODE)"
        return $false
    }
    return $true
}

function Download-WithCurl {
    param(
        [string]$RepoId,
        [string]$FileName,
        [string]$TargetDir,
        [string]$DisplaySize
    )
    $url = "https://huggingface.co/$RepoId/resolve/main/$FileName"
    $outPath = Join-Path $TargetDir $FileName

    if (-not $Force -and (Test-FileComplete $outPath)) {
        Write-Step "  SKIP $FileName (already exists, use -Force to re-download)"
        return $true
    }

    Write-Step "  GET  $FileName ($DisplaySize)"

    $curlArgs = @("-L", "--progress-bar", "-o", $outPath, $url)
    if ($Token) {
        $curlArgs += @("-H", "Authorization: Bearer $Token")
    }
    # Resume partial downloads for large files
    $curlArgs += @("-C", "-")

    & curl.exe @curlArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Failed to download $FileName"
        return $false
    }
    return $true
}

# ─── Main download ────────────────────────────────────────────────────────────

$TotalSize = "~7.66 GB"
Write-Header "Downloading VibeVoice-ASR model ($TotalSize)"
Write-Step "Model weights:  $Repo"
Write-Step "Tokenizer:      $TokenizerRepo"
Write-Step "Output:         $OutputDir"

$Failed = $false

if ($UseHfCli) {
    # ── huggingface-cli: batch download ──
    $modelFileNames = $ModelFiles | ForEach-Object { $_.Name }

    # Skip already-downloaded files unless -Force
    if (-not $Force) {
        $toDownload = @()
        foreach ($f in $modelFileNames) {
            $p = Join-Path $OutputDir $f
            if (Test-FileComplete $p) {
                Write-Step "  SKIP $f (already exists)"
            } else {
                $toDownload += $f
            }
        }
        $modelFileNames = $toDownload
    }

    if ($modelFileNames.Count -gt 0) {
        Write-Header "Downloading model weights from $Repo"
        $ok = Download-WithHfCli -RepoId $Repo -Files $modelFileNames -TargetDir $OutputDir
        if (-not $ok) { $Failed = $true }
    } else {
        Write-Step "All model weight files already present."
    }

    # Tokenizer
    $tokPath = Join-Path $OutputDir "tokenizer.json"
    if ($Force -or -not (Test-FileComplete $tokPath)) {
        Write-Header "Downloading tokenizer from $TokenizerRepo"
        $ok = Download-WithHfCli -RepoId $TokenizerRepo -Files @("tokenizer.json") -TargetDir $OutputDir
        if (-not $ok) { $Failed = $true }
    } else {
        Write-Step "tokenizer.json already present."
    }

} else {
    # ── curl.exe: file-by-file ──
    Write-Header "Downloading model weights from $Repo"
    foreach ($file in $ModelFiles) {
        $ok = Download-WithCurl -RepoId $Repo -FileName $file.Name `
                                -TargetDir $OutputDir -DisplaySize $file.Size
        if (-not $ok) { $Failed = $true; break }
    }

    if (-not $Failed) {
        Write-Header "Downloading tokenizer from $TokenizerRepo"
        foreach ($file in $TokenizerFiles) {
            $ok = Download-WithCurl -RepoId $TokenizerRepo -FileName $file.Name `
                                    -TargetDir $OutputDir -DisplaySize $file.Size
            if (-not $ok) { $Failed = $true }
        }
    }
}

# ─── Verify ───────────────────────────────────────────────────────────────────

Write-Header "Verifying downloaded files"

$AllFiles = $ModelFiles + $TokenizerFiles
$Missing = @()
foreach ($file in $AllFiles) {
    $p = Join-Path $OutputDir $file.Name
    if (Test-FileComplete $p) {
        $sz = (Get-Item $p).Length
        $szMB = [math]::Round($sz / 1MB, 1)
        Write-Step "  OK   $($file.Name) ($szMB MB)"
    } else {
        Write-Err "  MISS $($file.Name)"
        $Missing += $file.Name
    }
}

if ($Missing.Count -gt 0) {
    Write-Host ""
    Write-Err "$($Missing.Count) file(s) missing. Re-run the script or download manually."
    exit 1
}

if ($Failed) {
    Write-Err "Some downloads may have failed. Check the files above."
    exit 1
}

Write-Host ""
Write-Step "All files downloaded successfully!"
Write-Host ""
Write-Host "  Run inference:" -ForegroundColor White
Write-Host "    build\vv_cli.exe --model $OutputDir --audio recording.wav" -ForegroundColor Gray
Write-Host ""
