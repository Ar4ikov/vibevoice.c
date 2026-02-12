<#
.SYNOPSIS
    vibevoice.c -- CUDA Toolkit Installer and Environment Configurator

.DESCRIPTION
    Downloads and installs the NVIDIA CUDA Toolkit with a selected version,
    configures PATH, CUDA_PATH, and verifies the installation.
    Designed for the vibevoice.c project (minimum: CUDA 12.2, Ampere GPUs).

    Installs only the components needed for vibevoice.c development:
    nvcc, cudart, cublas, cufft, cusparse, cusolver, nvtx, cupti,
    nsight_systems, nsight_compute, visual_studio_integration, Display.Driver

.PARAMETER ConfigOnly
    Skip download/install entirely. Scan existing CUDA installations,
    let user pick one, and configure PATH + environment variables.
    Use this if CUDA is already installed but paths are not set up.

.PARAMETER Version
    CUDA version to install. Supported: 12.2, 12.4, 12.6, 12.8
    If not specified, an interactive menu is shown.

.PARAMETER InstallPath
    Custom install path (default: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X)

.PARAMETER Full
    Install ALL components (not just the dev subset)

.PARAMETER SkipDriverUpdate
    Do not include Display.Driver in the silent install

.PARAMETER DownloadOnly
    Download the installer but do not run it

.PARAMETER InstallerPath
    Path to an already-downloaded CUDA installer exe (skip download)

.PARAMETER Force
    Skip confirmation prompts

.EXAMPLE
    .\install_cuda.ps1 -ConfigOnly              # Scan + configure PATH only
    .\install_cuda.ps1                           # Interactive version menu
    .\install_cuda.ps1 -Version 12.6             # Install CUDA 12.6
    .\install_cuda.ps1 -Version 12.8 -Full       # Install 12.8 with all components
    .\install_cuda.ps1 -Version 12.6 -SkipDriverUpdate
    .\install_cuda.ps1 -InstallerPath .\cuda_12.6.3_windows.exe
#>

#Requires -RunAsAdministrator

[CmdletBinding()]
param(
    [switch]$ConfigOnly,

    [ValidateSet("12.2", "12.4", "12.6", "12.8")]
    [string]$Version = "",

    [string]$InstallPath = "",
    [string]$InstallerPath = "",

    [switch]$Full,
    [switch]$SkipDriverUpdate,
    [switch]$DownloadOnly,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$skipInstall = $false

# =============================================================================
#  Configuration: Known CUDA versions, URLs, and subpackage mappings
# =============================================================================

$CudaVersions = [ordered]@{
    "12.8" = @{
        DisplayName  = "CUDA 12.8.1  -- latest, Feb 2025"
        PatchVersion = "12.8.1"
        SubVer       = "12.8"
        NetworkUrl   = "https://developer.download.nvidia.com/compute/cuda/12.8.1/network_installers/cuda_12.8.1_windows_network.exe"
        LocalUrl     = "https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_553.62_windows.exe"
        ArchivePage  = "https://developer.nvidia.com/cuda-12-8-1-download-archive"
        MinDriver    = "553.62"
    }
    "12.6" = @{
        DisplayName  = "CUDA 12.6.3  -- stable, recommended"
        PatchVersion = "12.6.3"
        SubVer       = "12.6"
        NetworkUrl   = "https://developer.download.nvidia.com/compute/cuda/12.6.3/network_installers/cuda_12.6.3_windows_network.exe"
        LocalUrl     = "https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_546.80_windows.exe"
        ArchivePage  = "https://developer.nvidia.com/cuda-12-6-0-download-archive"
        MinDriver    = "546.80"
    }
    "12.4" = @{
        DisplayName  = "CUDA 12.4.1  -- widely tested"
        PatchVersion = "12.4.1"
        SubVer       = "12.4"
        NetworkUrl   = "https://developer.download.nvidia.com/compute/cuda/12.4.1/network_installers/cuda_12.4.1_windows_network.exe"
        LocalUrl     = "https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_551.78_windows.exe"
        ArchivePage  = "https://developer.nvidia.com/cuda-12-4-0-download-archive"
        MinDriver    = "551.78"
    }
    "12.2" = @{
        DisplayName  = "CUDA 12.2.2  -- minimum for vibevoice.c"
        PatchVersion = "12.2.2"
        SubVer       = "12.2"
        NetworkUrl   = "https://developer.download.nvidia.com/compute/cuda/12.2.2/network_installers/cuda_12.2.2_windows_network.exe"
        LocalUrl     = "https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_537.13_windows.exe"
        ArchivePage  = "https://developer.nvidia.com/cuda-12-2-0-download-archive"
        MinDriver    = "537.13"
    }
}

# Components needed for vibevoice.c development
function Get-DevComponents([string]$sv) {
    return @(
        "nvcc_$sv"
        "cudart_$sv"
        "cublas_$sv"
        "cublas_dev_$sv"
        "cufft_$sv"
        "cufft_dev_$sv"
        "cusparse_$sv"
        "cusparse_dev_$sv"
        "cusolver_$sv"
        "cusolver_dev_$sv"
        "curand_$sv"
        "curand_dev_$sv"
        "nvtx_$sv"
        "cupti_$sv"
        "cuda_profiler_api_$sv"
        "nvrtc_$sv"
        "nvrtc_dev_$sv"
        "nvjitlink_$sv"
        "nvml_dev_$sv"
        "thrust_$sv"
        "nsight_systems_$sv"
        "nsight_compute_$sv"
        "nsight_vse_$sv"
        "visual_studio_integration_$sv"
        "occupancy_calculator_$sv"
        "compute_sanitizer_$sv"
    )
}

# =============================================================================
#  Helper functions
# =============================================================================

function Write-Banner {
    Write-Host ""
    Write-Host "  +==================================================+" -ForegroundColor Cyan
    Write-Host "  |    NVIDIA CUDA Toolkit Installer                  |" -ForegroundColor Cyan
    Write-Host "  |    for vibevoice.c                                |" -ForegroundColor Cyan
    Write-Host "  +==================================================+" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Step   ([string]$msg) { Write-Host "  [*] $msg" -ForegroundColor Cyan    }
function Write-Ok     ([string]$msg) { Write-Host "  [+] $msg" -ForegroundColor Green   }
function Write-Warn   ([string]$msg) { Write-Host "  [!] $msg" -ForegroundColor Yellow  }
function Write-Err    ([string]$msg) { Write-Host "  [-] $msg" -ForegroundColor Red     }
function Write-Info   ([string]$msg) { Write-Host "      $msg" -ForegroundColor Gray    }

function Test-IsAdmin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Test-UrlReachable([string]$url) {
    try {
        $request = [System.Net.HttpWebRequest]::Create($url)
        $request.Method = "HEAD"
        $request.Timeout = 10000
        $response = $request.GetResponse()
        $status = $response.StatusCode
        $response.Close()
        return ($status -eq "OK")
    } catch {
        return $false
    }
}

function Get-ExistingCudaInstalls {
    $installs = @()
    $cudaBase = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path $cudaBase) {
        Get-ChildItem $cudaBase -Directory | ForEach-Object {
            $nvcc = Join-Path $_.FullName "bin\nvcc.exe"
            if (Test-Path $nvcc) {
                $ver = & $nvcc --version 2>&1 | Select-String "release" |
                    ForEach-Object { $_ -replace '.*release\s+([\d.]+).*','$1' }
                $installs += [PSCustomObject]@{
                    Path    = $_.FullName
                    Version = $ver
                    Folder  = $_.Name
                }
            }
        }
    }
    return $installs
}

# =============================================================================
#  Pre-flight checks
# =============================================================================

Write-Banner

# Check admin
if (-not (Test-IsAdmin)) {
    Write-Err "This script must be run as Administrator!"
    Write-Err "Right-click PowerShell -> 'Run as administrator', then re-run."
    exit 1
}
Write-Ok "Running as Administrator"

# Check GPU
Write-Step "Checking GPU..."
$hasSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($hasSmi) {
    $gpuInfo = & nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader 2>$null
    if ($gpuInfo) {
        $parts = $gpuInfo.Split(",") | ForEach-Object { $_.Trim() }
        $gpuName = $parts[0]
        $gpuCC   = $parts[1]
        $gpuDrv  = $parts[2]
        $gpuVram = $parts[3]
        Write-Ok "GPU: $gpuName"
        Write-Info "Compute Capability: $gpuCC, Driver: $gpuDrv, VRAM: $gpuVram"

        $ccMajor = [int]($gpuCC.Split(".")[0])
        if ($ccMajor -lt 8) {
            Write-Warn "GPU compute capability $gpuCC is below SM 8.0 -- Ampere minimum."
            Write-Warn "vibevoice.c requires Ampere RTX 30XX or newer."
            if (-not $Force) {
                $confirm = Read-Host "      Continue anyway? [y/N]"
                if ($confirm -ne "y") { exit 0 }
            }
        }
    }
} else {
    Write-Warn "nvidia-smi not found. Cannot verify GPU. Proceeding..."
}

# Show existing installs
Write-Step "Checking for existing CUDA installations..."
$existing = Get-ExistingCudaInstalls
if ($existing.Count -gt 0) {
    Write-Info "Found existing CUDA installations:"
    $existing | ForEach-Object {
        $f = $_.Folder; $v = $_.Version; $p = $_.Path
        Write-Info "  $f  CUDA $v  ->  $p"
    }
    Write-Host ""
} else {
    Write-Info "No existing CUDA installations found."
}

# Current CUDA_PATH
if ($env:CUDA_PATH) {
    Write-Info "Current CUDA_PATH: $env:CUDA_PATH"
}

# =============================================================================
#  ConfigOnly mode -- scan, pick, configure, verify, exit
# =============================================================================

if ($ConfigOnly) {
    Write-Host ""
    Write-Host "  ====================================================" -ForegroundColor Cyan
    Write-Step "ConfigOnly mode -- scanning installed CUDA toolkits..."
    Write-Host "  ====================================================" -ForegroundColor Cyan
    Write-Host ""

    # Deep scan: check standard location + CUDA_PATH + custom InstallPath
    $cudaBase = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    [System.Collections.ArrayList]$installList = @()

    # Scan standard directory
    if (Test-Path $cudaBase) {
        Get-ChildItem $cudaBase -Directory | ForEach-Object {
            $nvccPath = Join-Path $_.FullName "bin\nvcc.exe"
            if (Test-Path $nvccPath) {
                $ver = & $nvccPath --version 2>&1 | Select-String "release" |
                    ForEach-Object { $_ -replace '.*release\s+([\d.]+).*','$1' }
                $null = $installList.Add([PSCustomObject]@{
                    Path    = $_.FullName
                    Version = [string]$ver
                    Folder  = $_.Name
                    Source  = "standard"
                })
            }
        }
    }

    # Check current CUDA_PATH if set
    if ($env:CUDA_PATH -and (Test-Path $env:CUDA_PATH)) {
        $nvccCur = Join-Path $env:CUDA_PATH "bin\nvcc.exe"
        if (Test-Path $nvccCur) {
            $already = $installList | Where-Object { $_.Path -eq $env:CUDA_PATH }
            if (-not $already) {
                $ver = & $nvccCur --version 2>&1 | Select-String "release" |
                    ForEach-Object { $_ -replace '.*release\s+([\d.]+).*','$1' }
                $null = $installList.Add([PSCustomObject]@{
                    Path    = $env:CUDA_PATH
                    Version = [string]$ver
                    Folder  = Split-Path $env:CUDA_PATH -Leaf
                    Source  = "CUDA_PATH"
                })
            }
        }
    }

    # Check custom path
    if ($InstallPath -and (Test-Path $InstallPath)) {
        $nvccCustom = Join-Path $InstallPath "bin\nvcc.exe"
        if (Test-Path $nvccCustom) {
            $already = $installList | Where-Object { $_.Path -eq $InstallPath }
            if (-not $already) {
                $ver = & $nvccCustom --version 2>&1 | Select-String "release" |
                    ForEach-Object { $_ -replace '.*release\s+([\d.]+).*','$1' }
                $null = $installList.Add([PSCustomObject]@{
                    Path    = $InstallPath
                    Version = [string]$ver
                    Folder  = Split-Path $InstallPath -Leaf
                    Source  = "custom"
                })
            }
        }
    }

    $totalFound = $installList.Count
    if ($totalFound -eq 0) {
        Write-Err "No CUDA installations found!"
        Write-Info "Searched:"
        Write-Info "  $cudaBase"
        if ($env:CUDA_PATH) { Write-Info "  $env:CUDA_PATH" }
        if ($InstallPath)   { Write-Info "  $InstallPath" }
        Write-Host ""
        Write-Info "Install CUDA first:  .\install_cuda.ps1 -Version 12.6"
        exit 1
    }

    # Show what we found
    Write-Ok "Found $totalFound CUDA installation(s):"
    Write-Host ""

    $idx = 1
    foreach ($inst in $installList) {
        $isCurrent = ""
        if ($env:CUDA_PATH -and $inst.Path -eq $env:CUDA_PATH) {
            $isCurrent = " [current CUDA_PATH]"
        }
        $instVer  = $inst.Version
        $instPath = $inst.Path
        Write-Host "    [$idx] CUDA $instVer  --  $instPath$isCurrent" -ForegroundColor White

        # Show what's inside
        $binPath = Join-Path $inst.Path "bin"
        $incPath = Join-Path $inst.Path "include"
        $libPath = Join-Path $inst.Path "lib\x64"
        $statusParts = @()
        if (Test-Path $binPath) { $statusParts += "bin" }
        if (Test-Path $incPath) { $statusParts += "include" }
        if (Test-Path $libPath) { $statusParts += "lib" }
        $statusStr = $statusParts -join ", "
        Write-Host "        dirs: $statusStr" -ForegroundColor DarkGray

        $idx++
    }
    Write-Host ""

    # Pick one
    $selectedInstall = $null
    if ($totalFound -eq 1) {
        $selectedInstall = $installList[0]
        Write-Step "Only one installation found, selecting it automatically."
    } else {
        do {
            $pick = Read-Host "  Select installation to configure [1-$totalFound]"
            $pickNum = 0
            [int]::TryParse($pick, [ref]$pickNum) | Out-Null
        } while ($pickNum -lt 1 -or $pickNum -gt $totalFound)
        $selectedInstall = $installList[$pickNum - 1]
    }

    $foundPath = $selectedInstall.Path
    $selVersion = [string]$selectedInstall.Version
    Write-Host ""
    Write-Ok "Configuring: CUDA $selVersion at $foundPath"

    # Extract major.minor version for CUDA_PATH_vX_Y
    $verParts = $selVersion.Split(".")
    if ($verParts.Count -ge 2) {
        $Version = "$($verParts[0]).$($verParts[1])"
    } else {
        $Version = $selVersion
    }

    # Fall through to the shared "Configure environment" + "Verify" sections below
    # We need $foundPath and $Version set, which we just did.
    # Skip everything else by jumping past install sections.
    $skipInstall = $true

    # Also set selPatch for the summary banner
    $selPatch = $selVersion
}

# =============================================================================
#  Version selection  (skipped in ConfigOnly mode)
# =============================================================================

if (-not $Version) {
    Write-Host ""
    Write-Host "  Select CUDA Toolkit version to install:" -ForegroundColor White
    Write-Host ""

    $idx = 1
    $versionKeys = @($CudaVersions.Keys)
    foreach ($key in $versionKeys) {
        $info = $CudaVersions[$key]
        $marker = ""
        $existingMatch = $existing | Where-Object {
            $_.Folder -eq "v$($info.PatchVersion)" -or $_.Folder -eq "v$key"
        }
        if ($existingMatch) { $marker = " [INSTALLED]" }
        if ($key -eq "12.6") { $marker += " [recommended]" }

        $color = if ($existingMatch) { "DarkGray" } else { "White" }
        $dispName = $info.DisplayName
        Write-Host "    [$idx] $dispName$marker" -ForegroundColor $color
        $idx++
    }

    Write-Host "    [0] Cancel" -ForegroundColor DarkGray
    Write-Host ""

    $maxChoice = $versionKeys.Count
    do {
        $choice = Read-Host "  Enter choice [1-$maxChoice]"
        if ($choice -eq "0") { Write-Host "  Cancelled."; exit 0 }
        $choiceNum = 0
        [int]::TryParse($choice, [ref]$choiceNum) | Out-Null
    } while ($choiceNum -lt 1 -or $choiceNum -gt $maxChoice)

    $Version = $versionKeys[$choiceNum - 1]
}

if (-not $ConfigOnly) {
    $cfg = $CudaVersions[$Version]
    Write-Host ""
    $selName = $cfg.DisplayName
    $selPatch = $cfg.PatchVersion
    $selMinDrv = $cfg.MinDriver
    Write-Ok "Selected: $selName"
    Write-Info "Patch version: $selPatch"
    Write-Info "Minimum driver: $selMinDrv"

    # =========================================================================
    #  Check if already installed
    # =========================================================================

    $targetPath = if ($InstallPath) {
        $InstallPath
    } else {
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$selPatch"
    }

    # Also check the major.minor path
    $altPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$Version"
    $nvccTarget = Join-Path $targetPath "bin\nvcc.exe"
    $nvccAlt    = Join-Path $altPath    "bin\nvcc.exe"

    if ((Test-Path $nvccTarget) -or (Test-Path $nvccAlt)) {
        $foundAt = if (Test-Path $nvccTarget) { $targetPath } else { $altPath }
        Write-Warn "CUDA $Version appears to already be installed at:"
        Write-Info $foundAt

        if (-not $Force) {
            $confirm = Read-Host "      Reinstall / repair? [y/N]"
            if ($confirm -ne "y") {
                Write-Step "Skipping install. Will configure environment for existing install..."
                $foundPath = $foundAt
                $skipInstall = $true
            }
        }
    }
} # end if (-not $ConfigOnly)

# =============================================================================
#  Download + Install  (skipped if $skipInstall)
# =============================================================================

if (-not $skipInstall) {

    # --- Download ---

    $downloadDir = Join-Path $env:TEMP "cuda_installers"
    if (-not (Test-Path $downloadDir)) {
        New-Item -ItemType Directory -Path $downloadDir -Force | Out-Null
    }

    if ($InstallerPath -and (Test-Path $InstallerPath)) {
        Write-Ok "Using provided installer: $InstallerPath"
        $installerExe = $InstallerPath
    } else {
        # Always prefer LOCAL installer -- it is self-contained, reliable,
        # and supports selective component install.
        # Network installer often fails with cryptic exit codes.
        $localFileName = Split-Path $cfg.LocalUrl -Leaf
        $installerExe = Join-Path $downloadDir $localFileName

        if (Test-Path $installerExe) {
            $fileSize = [math]::Round((Get-Item $installerExe).Length / 1MB, 1)
            Write-Ok "Installer already downloaded: $localFileName -- $fileSize MB"

            if (-not $Force) {
                $reuse = Read-Host "      Reuse existing download? [Y/n]"
                if ($reuse -eq "n") { Remove-Item $installerExe -Force }
            }
        }

        if (-not (Test-Path $installerExe)) {
            $downloadUrl = $cfg.LocalUrl
            $dlSizeNote = "~3 GB"
            Write-Step "Downloading CUDA $selPatch local installer -- $dlSizeNote ..."
            Write-Info "URL: $downloadUrl"
            Write-Info "This is the full offline installer -- most reliable for silent install."
            Write-Host ""

            Write-Step "Checking download URL..."
            $urlOk = Test-UrlReachable $downloadUrl
            if (-not $urlOk) {
                $archPage = $cfg.ArchivePage
                Write-Err "Download URL not reachable."
                Write-Err "Please download the LOCAL installer manually from:"
                Write-Info $archPage
                Write-Host ""
                Write-Info "Select: Windows -> x86_64 -> 11 -> exe [local]"
                Write-Info "Then re-run:  .\install_cuda.ps1 -Version $Version -InstallerPath 'path_to_exe'"
                Start-Process $archPage
                exit 1
            }
            Write-Ok "URL reachable, starting download..."
            Write-Info "This will take a few minutes depending on your connection."
            Write-Host ""

            try {
                Start-BitsTransfer -Source $downloadUrl -Destination $installerExe `
                    -DisplayName "CUDA $selPatch local installer" `
                    -Description "Downloading CUDA Toolkit -- $dlSizeNote ..." `
                    -ErrorAction Stop
                Write-Ok "Download complete!"
            } catch {
                Write-Warn "BITS transfer failed, falling back to Invoke-WebRequest..."
                try {
                    # Invoke-WebRequest with progress
                    $ProgressPreference = 'Continue'
                    Invoke-WebRequest -Uri $downloadUrl -OutFile $installerExe -UseBasicParsing
                    Write-Ok "Download complete!"
                } catch {
                    Write-Err "Download failed: $_"
                    $archPage = $cfg.ArchivePage
                    Write-Err "Please download the LOCAL installer manually from:"
                    Write-Info $archPage
                    Write-Info "Select: Windows -> x86_64 -> 11 -> exe [local]"
                    Start-Process $archPage
                    exit 1
                }
            }

            $fileSize = [math]::Round((Get-Item $installerExe).Length / 1MB, 1)
            Write-Info "Installer size: $fileSize MB"
        }
    }

    if ($DownloadOnly) {
        Write-Ok "Download-only mode. Installer saved to:"
        Write-Info $installerExe
        exit 0
    }

    # --- Install ---

    Write-Host ""
    Write-Host "  ====================================================" -ForegroundColor White
    Write-Step "Installing CUDA $selPatch..."
    Write-Host "  ====================================================" -ForegroundColor White
    Write-Host ""

    $sv = $cfg.SubVer

    if ($Full) {
        Write-Step "Mode: FULL silent install -- all components"
        $installArgs = @("-s")
    } else {
        # Local installer supports selective component install
        $components = Get-DevComponents $sv

        if (-not $SkipDriverUpdate) {
            $components += "Display.Driver"
            Write-Info "Including Display.Driver update"
        } else {
            Write-Info "Skipping Display.Driver -- using existing driver"
        }

        $compCount = $components.Count
        Write-Step "Mode: Development install -- $compCount components"
        Write-Info "Components: nvcc, cudart, cublas, cufft, cusparse, cusolver,"
        Write-Info "            curand, nvtx, cupti, nvrtc, thrust, nsight, VS integration"

        $installArgs = @("-s") + $components
    }

    if (-not $Force) {
        Write-Host ""
        $confirm = Read-Host "  Proceed with installation? [Y/n]"
        if ($confirm -eq "n") { Write-Host "  Cancelled."; exit 0 }
    }

    Write-Host ""
    Write-Step "Running installer -- this may take 5-15 minutes..."
    Write-Info "The screen may flicker if Display.Driver is being updated."
    Write-Info "Do NOT close this window."
    Write-Host ""

    # Let the installer run in its own window -- -NoNewWindow causes issues
    $argsString = $installArgs -join " "
    Write-Info "Command: $installerExe $argsString"
    Write-Host ""
    $installProcess = Start-Process -FilePath $installerExe `
        -ArgumentList $argsString `
        -Wait -PassThru

    $exitCode = $installProcess.ExitCode

    if ($exitCode -eq 0) {
        Write-Ok "CUDA installer completed successfully -- exit code 0."
    } elseif ($exitCode -eq 1) {
        Write-Warn "Installer returned exit code 1 -- partial success or reboot needed."
    } else {
        Write-Err "Installer returned exit code: $exitCode"
        Write-Host ""

        # Search for log files in known NVIDIA locations
        $logSearchPaths = @(
            "$env:TEMP\cuda_install.log",
            "$env:TEMP\CUDA Setup.log",
            "$env:TEMP\CUDA",
            "$env:PROGRAMDATA\NVIDIA Corporation",
            "$env:TEMP"
        )
        $foundLogs = @()
        foreach ($lp in $logSearchPaths) {
            if (Test-Path $lp) {
                if ((Get-Item $lp).PSIsContainer) {
                    $recent = Get-ChildItem $lp -Filter "*.log" -Recurse -ErrorAction SilentlyContinue |
                        Where-Object { $_.LastWriteTime -gt (Get-Date).AddHours(-1) } |
                        Sort-Object LastWriteTime -Descending |
                        Select-Object -First 3
                    if ($recent) { $foundLogs += $recent }
                } elseif (Test-Path $lp) {
                    $foundLogs += Get-Item $lp
                }
            }
        }

        if ($foundLogs.Count -gt 0) {
            Write-Info "Recent install logs found:"
            foreach ($fl in $foundLogs) {
                Write-Info "  $($fl.FullName)"
            }
            # Show tail of most recent log
            $newest = $foundLogs | Sort-Object LastWriteTime -Descending | Select-Object -First 1
            Write-Host ""
            Write-Info "Last 15 lines of $($newest.Name):"
            Get-Content $newest.FullName -Tail 15 -ErrorAction SilentlyContinue | ForEach-Object {
                Write-Host "      $_" -ForegroundColor DarkGray
            }
        } else {
            Write-Info "No install logs found."
        }

        Write-Host ""
        Write-Info "Troubleshooting:"
        Write-Info "  1. Close ALL NVIDIA apps (GeForce Experience, etc.)"
        Write-Info "  2. Temporarily disable antivirus"
        Write-Info "  3. Reboot and try again"
        Write-Info "  4. Or install manually from: $($cfg.ArchivePage)"
        exit 1
    }

} # end if (-not $skipInstall)

# =============================================================================
#  Find the actual install path  (in ConfigOnly mode, $foundPath is already set)
# =============================================================================

if (-not $foundPath) {
    Write-Host ""
    Write-Step "Locating CUDA installation..."

    $cudaBase = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    $foundPath = ""

    # Build search paths from $cfg if available, otherwise just version string
    $searchPaths = @()
    if ($cfg) {
        $searchPaths += (Join-Path $cudaBase "v$($cfg.PatchVersion)")
        $searchPaths += (Join-Path $cudaBase "v$($cfg.SubVer)")
    }
    $searchPaths += (Join-Path $cudaBase "v$Version")
    if ($InstallPath) { $searchPaths = @($InstallPath) + $searchPaths }

    foreach ($sp in $searchPaths) {
        $testNvcc = Join-Path $sp "bin\nvcc.exe"
        if (Test-Path $testNvcc) {
            $foundPath = $sp
            break
        }
    }

    # Fallback: find newest CUDA folder
    if (-not $foundPath) {
        if (Test-Path $cudaBase) {
            $newest = Get-ChildItem $cudaBase -Directory |
                Where-Object { Test-Path (Join-Path $_.FullName "bin\nvcc.exe") } |
                Sort-Object LastWriteTime -Descending |
                Select-Object -First 1

            if ($newest) {
                $foundPath = $newest.FullName
                Write-Warn "Expected path not found, using newest: $foundPath"
            }
        }
    }

    if (-not $foundPath) {
        Write-Err "Could not locate CUDA installation!"
        Write-Err "You may need to reboot and check manually."
        exit 1
    }
}

Write-Ok "CUDA installed at: $foundPath"

# =============================================================================
#  Configure environment variables -- system-wide, persistent
# =============================================================================

Write-Host ""
Write-Host "  ====================================================" -ForegroundColor White
Write-Step "Configuring environment variables..."
Write-Host "  ====================================================" -ForegroundColor White
Write-Host ""

$cudaBin       = Join-Path $foundPath "bin"
$cudaLibnvvp   = Join-Path $foundPath "libnvvp"
$cudaExtras    = Join-Path $foundPath "extras\CUPTI\lib64"
$cudaInclude   = Join-Path $foundPath "include"
$cudaLib       = Join-Path $foundPath "lib\x64"

# --- CUDA_PATH ---
Write-Step "Setting CUDA_PATH = $foundPath"
[System.Environment]::SetEnvironmentVariable("CUDA_PATH", $foundPath, "Machine")
$env:CUDA_PATH = $foundPath

# --- CUDA_PATH_vX_Y (versioned) ---
$verUnderscore = $Version.Replace('.','_')
$versionedVar = "CUDA_PATH_v$verUnderscore"
Write-Step "Setting $versionedVar = $foundPath"
[System.Environment]::SetEnvironmentVariable($versionedVar, $foundPath, "Machine")
[System.Environment]::SetEnvironmentVariable($versionedVar, $foundPath, "Process")

# --- CUDA_HOME (compatibility) ---
Write-Step "Setting CUDA_HOME = $foundPath"
[System.Environment]::SetEnvironmentVariable("CUDA_HOME", $foundPath, "Machine")
$env:CUDA_HOME = $foundPath

# --- PATH ---
Write-Step "Updating system PATH..."

$systemPath = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
$pathEntries = $systemPath -split ";" | Where-Object { $_ -ne "" }

# Remove any old CUDA paths to avoid duplicates/conflicts
$escapedFound = [regex]::Escape($foundPath)
$cleanedPaths = $pathEntries | Where-Object {
    $_ -notmatch "NVIDIA GPU Computing Toolkit\\CUDA\\v[\d.]+" -or
    $_ -match $escapedFound
}

# Paths to add
$newPaths = @($cudaBin, $cudaLibnvvp)
if (Test-Path $cudaExtras) { $newPaths += $cudaExtras }

$pathsToAdd = @()
foreach ($np in $newPaths) {
    if ($cleanedPaths -notcontains $np) {
        $pathsToAdd += $np
        Write-Info "  Adding: $np"
    } else {
        Write-Info "  Already in PATH: $np"
    }
}

if ($pathsToAdd.Count -gt 0) {
    $newSystemPath = ($pathsToAdd + $cleanedPaths) -join ";"
    [System.Environment]::SetEnvironmentVariable("Path", $newSystemPath, "Machine")
    $addedCount = $pathsToAdd.Count
    Write-Ok "System PATH updated -- $addedCount entries added"
} else {
    Write-Ok "System PATH already up to date"
}

# Update current session PATH
$sessionPath = "$cudaBin;$cudaLibnvvp"
$env:Path = "$sessionPath;$env:Path"

# --- INCLUDE / LIB for MSVC discovery ---
[System.Environment]::SetEnvironmentVariable("CUDA_INC_PATH", $cudaInclude, "Machine")
[System.Environment]::SetEnvironmentVariable("CUDA_LIB_PATH", $cudaLib, "Machine")
$env:CUDA_INC_PATH = $cudaInclude
$env:CUDA_LIB_PATH = $cudaLib
Write-Info "Set CUDA_INC_PATH = $cudaInclude"
Write-Info "Set CUDA_LIB_PATH = $cudaLib"

# =============================================================================
#  Verify installation
# =============================================================================

Write-Host ""
Write-Host "  ====================================================" -ForegroundColor White
Write-Step "Verifying installation..."
Write-Host "  ====================================================" -ForegroundColor White
Write-Host ""

$allOk = $true

# nvcc
$nvccExe = Join-Path $cudaBin "nvcc.exe"
if (Test-Path $nvccExe) {
    $nvccVer = & $nvccExe --version 2>&1 | Select-String "release" |
        ForEach-Object { $_ -replace '.*release\s+([\d.]+).*','$1' }
    Write-Ok "nvcc $nvccVer"
} else {
    Write-Err "nvcc.exe not found at: $nvccExe"
    $allOk = $false
}

# Key libraries
$libs = @(
    @{ Name = "cudart64";   Pattern = "cudart64_*.dll" }
    @{ Name = "cublas64";   Pattern = "cublas64_*.dll" }
    @{ Name = "cufft64";    Pattern = "cufft64_*.dll" }
    @{ Name = "cusparse64"; Pattern = "cusparse64_*.dll" }
    @{ Name = "nvrtc64";    Pattern = "nvrtc64_*.dll" }
)

foreach ($lib in $libs) {
    $dllFound = Get-ChildItem (Join-Path $cudaBin $lib.Pattern) -ErrorAction SilentlyContinue
    if ($dllFound) {
        $dllName = $dllFound[0].Name
        Write-Ok "$($lib.Name): $dllName"
    } else {
        $libName = $lib.Name
        Write-Warn "$libName DLL not found in bin/"
    }
}

# Headers
$headers = @("cuda.h", "cuda_runtime.h", "cublas_v2.h", "cuda_fp16.h")
foreach ($h in $headers) {
    $hpath = Join-Path $cudaInclude $h
    if (Test-Path $hpath) {
        Write-Ok "Header: $h"
    } else {
        Write-Warn "Header not found: $h"
    }
}

# Import libraries
$importLibs = @("cudart.lib", "cublas.lib", "cuda.lib")
foreach ($il in $importLibs) {
    $ilpath = Join-Path $cudaLib $il
    if (Test-Path $ilpath) {
        Write-Ok "Import lib: $il"
    } else {
        Write-Warn "Import lib not found: $il"
    }
}

# nsight-sys
$nsys = Join-Path $cudaBin "nsys.exe"
if (-not (Test-Path $nsys)) {
    $nsysAlt = Get-ChildItem "C:\Program Files\NVIDIA Corporation\Nsight Systems*" `
        -Directory -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($nsysAlt) {
        $nsys = Join-Path $nsysAlt.FullName "target-windows-x64\nsys.exe"
    }
}
if (Test-Path $nsys) {
    Write-Ok "Nsight Systems: found"
} else {
    Write-Warn "Nsight Systems nsys not found -- optional for profiling"
}

# =============================================================================
#  Summary
# =============================================================================

Write-Host ""

$modeLabel = if ($ConfigOnly) { "configured" } else { "installed" }

if ($allOk) {
    Write-Host "  +==================================================+" -ForegroundColor Green
    Write-Host "  |  CUDA $selPatch $modeLabel successfully!               |" -ForegroundColor Green
    Write-Host "  +==================================================+" -ForegroundColor Green
} else {
    Write-Host "  +==================================================+" -ForegroundColor Yellow
    Write-Host "  |  CUDA $modeLabel with warnings -- see above       |" -ForegroundColor Yellow
    Write-Host "  +==================================================+" -ForegroundColor Yellow
}

Write-Host ""
Write-Info "Installation path:   $foundPath"
Write-Info "CUDA_PATH:           $env:CUDA_PATH"

$nvccCheck = Join-Path $cudaBin "nvcc.exe"
if (Test-Path $nvccCheck) {
    $nvccLine = & $nvccCheck --version 2>&1 | Select-String "release"
    Write-Info "nvcc:                $nvccLine"
}

Write-Host ""
Write-Info "Environment variables have been set SYSTEM-WIDE -- persistent."
Write-Info "New terminal windows will pick them up automatically."
Write-Info "This terminal session is already configured."
Write-Host ""
Write-Step "Next steps:"
Write-Info "  1. Open a NEW terminal or Developer Command Prompt"
Write-Info "  2. Verify:  nvcc --version"
Write-Info "  3. Build:   .\build.ps1 -BuildType Release"
Write-Host ""

# Check if reboot might be needed (driver update) -- not in ConfigOnly
if (-not $ConfigOnly -and -not $SkipDriverUpdate -and -not $Full) {
    Write-Warn "If you updated Display.Driver, a REBOOT may be needed."
    if (-not $Force) {
        $reboot = Read-Host "      Reboot now? [y/N]"
        if ($reboot -eq "y") {
            Write-Step "Rebooting in 10 seconds... Ctrl+C to cancel"
            Start-Sleep -Seconds 10
            Restart-Computer -Force
        }
    }
}

Write-Host ""
