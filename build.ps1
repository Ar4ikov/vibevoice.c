<#
.SYNOPSIS
    vibevoice.c — Windows Build Script (PowerShell)

.DESCRIPTION
    Configures and builds the vibevoice.c project using CMake + MSVC + CUDA.

.PARAMETER BuildType
    Build configuration: Debug, Release (default), RelWithDebInfo, MinSizeRel

.PARAMETER Clean
    Remove the build directory before building

.PARAMETER Test
    Run CTest after successful build

.PARAMETER Bench
    Run benchmark suite after successful build

.PARAMETER NoBuild
    Only configure, skip build

.PARAMETER CudaArch
    CUDA architectures (default: "80;86;89")

.PARAMETER BuildDir
    Build output directory (default: "build")

.PARAMETER Generator
    CMake generator (auto-detected: Ninja > VS 2022)

.PARAMETER Jobs
    Parallel build jobs (default: CPU count)

.PARAMETER Verbose
    Verbose build output

.PARAMETER NVTX
    Enable NVTX profiling markers

.EXAMPLE
    .\build.ps1
    .\build.ps1 -BuildType Debug
    .\build.ps1 -Clean -Test
    .\build.ps1 -BuildType Release -Bench -NVTX
    .\build.ps1 -CudaArch "86"
#>

[CmdletBinding()]
param(
    [ValidateSet("Debug", "Release", "RelWithDebInfo", "MinSizeRel")]
    [string]$BuildType = "Release",

    [switch]$Clean,
    [switch]$Test,
    [switch]$Bench,
    [switch]$NoBuild,
    [switch]$Verbose,
    [switch]$NVTX,

    [string]$CudaArch = "80;86;89",
    [string]$BuildDir = "build",
    [string]$Generator = "",
    [int]$Jobs = 0
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot

# ─── Colours ──────────────────────────────────────────────────────────────────
function Write-Step   { param([string]$msg) Write-Host "[vibevoice] $msg" -ForegroundColor Cyan }
function Write-Ok     { param([string]$msg) Write-Host "[vibevoice] $msg" -ForegroundColor Green }
function Write-Warn   { param([string]$msg) Write-Host "[vibevoice] $msg" -ForegroundColor Yellow }
function Write-Err    { param([string]$msg) Write-Host "[ERROR]     $msg" -ForegroundColor Red }

# ─── Banner ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  ╔══════════════════════════════════════════════╗" -ForegroundColor Magenta
Write-Host "  ║       vibevoice.c  Build System              ║" -ForegroundColor Magenta
Write-Host "  ║       Pure C  |  CUDA                        ║" -ForegroundColor Magenta
Write-Host "  ╚══════════════════════════════════════════════╝" -ForegroundColor Magenta
Write-Host ""

# ─── Clean ────────────────────────────────────────────────────────────────────
$BuildPath = Join-Path $ProjectRoot $BuildDir

if ($Clean) {
    if (Test-Path $BuildPath) {
        Write-Step "Cleaning $BuildDir..."
        Remove-Item -Recurse -Force $BuildPath
        Write-Ok "Clean complete."
    } else {
        Write-Step "Build directory doesn't exist, nothing to clean."
    }
}

# ─── Detect MSVC ──────────────────────────────────────────────────────────────
$hasCompiler = Get-Command cl -ErrorAction SilentlyContinue
if (-not $hasCompiler) {
    Write-Step "MSVC not in PATH, searching for Visual Studio..."
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"

    if (-not (Test-Path $vswhere)) {
        Write-Err "vswhere not found. Install Visual Studio 2022 with C++ workload."
        exit 1
    }

    $vsPath = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath 2>$null | Select-Object -First 1

    if (-not $vsPath) {
        Write-Err "No Visual Studio with C++ found."
        exit 1
    }

    Write-Step "Found Visual Studio at: $vsPath"
    $vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"

    if (-not (Test-Path $vcvars)) {
        Write-Err "vcvars64.bat not found."
        exit 1
    }

    # Import MSVC environment into PowerShell
    Write-Step "Loading MSVC environment..."
    $envBefore = @{}
    Get-ChildItem env: | ForEach-Object { $envBefore[$_.Name] = $_.Value }

    cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
        if ($_ -match "^([^=]+)=(.*)$") {
            [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
        }
    }
    Write-Ok "MSVC environment loaded"
}

# Verify cl is now available
$clPath = (Get-Command cl -ErrorAction SilentlyContinue).Source
if ($clPath) {
    $clVersion = (& cl 2>&1 | Select-Object -First 1) -replace '.*Version\s+([\d.]+).*','$1'
    Write-Ok "MSVC cl.exe: $clVersion"
} else {
    Write-Err "cl.exe still not found after environment setup!"
    exit 1
}

# ─── Detect CUDA ──────────────────────────────────────────────────────────────
$cudaPath = $env:CUDA_PATH
if (-not $cudaPath) {
    $cudaBase = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path $cudaBase) {
        $cudaPath = Get-ChildItem $cudaBase -Directory | Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty FullName
    }
}

if (-not $cudaPath -or -not (Test-Path $cudaPath)) {
    Write-Err "CUDA Toolkit not found. Set CUDA_PATH environment variable."
    exit 1
}

$nvccPath = Join-Path $cudaPath "bin\nvcc.exe"
if (Test-Path $nvccPath) {
    $nvccVer = & $nvccPath --version 2>&1 | Select-String "release" | ForEach-Object { $_ -replace '.*release\s+([\d.]+).*','$1' }
    Write-Ok "CUDA $nvccVer at: $cudaPath"
} else {
    Write-Err "nvcc not found in CUDA_PATH."
    exit 1
}

# Make sure nvcc is on PATH
$env:PATH = "$cudaPath\bin;$env:PATH"

# ─── Detect generator ────────────────────────────────────────────────────────
if (-not $Generator) {
    if (Get-Command ninja -ErrorAction SilentlyContinue) {
        $Generator = "Ninja"
    } else {
        $Generator = "Visual Studio 17 2022"
    }
}
Write-Step "Generator: $Generator"
Write-Step "CUDA archs: $CudaArch"
Write-Step "Build type: $BuildType"

# ─── Parallel jobs ────────────────────────────────────────────────────────────
if ($Jobs -le 0) {
    $Jobs = [Environment]::ProcessorCount
}
Write-Step "Parallel jobs: $Jobs"

# ─── Configure ────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=" * 72 -ForegroundColor White
Write-Step "Configuring..."
Write-Host "=" * 72 -ForegroundColor White
Write-Host ""

if (-not (Test-Path $BuildPath)) {
    New-Item -ItemType Directory -Path $BuildPath -Force | Out-Null
}

$nvtxFlag = if ($NVTX) { "-DVV_ENABLE_NVTX=ON" } else { "-DVV_ENABLE_NVTX=OFF" }

$cmakeArgs = @(
    "-S", $ProjectRoot,
    "-B", $BuildPath,
    "-G", $Generator,
    "-DCMAKE_BUILD_TYPE=$BuildType",
    "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch",
    "-DVV_BUILD_TESTS=ON",
    "-DVV_BUILD_BENCH=ON",
    "-DVV_BUILD_CLI=ON",
    $nvtxFlag
)

& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) {
    Write-Err "CMake configure failed!"
    exit 1
}
Write-Ok "Configure successful."

# ─── Build ────────────────────────────────────────────────────────────────────
if (-not $NoBuild) {
    Write-Host ""
    Write-Host "=" * 72 -ForegroundColor White
    Write-Step "Building ($Jobs jobs)..."
    Write-Host "=" * 72 -ForegroundColor White
    Write-Host ""

    $buildArgs = @("--build", $BuildPath, "--config", $BuildType, "--parallel", $Jobs)
    if ($Verbose) {
        $buildArgs += "--verbose"
    }

    & cmake @buildArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Build failed!"
        exit 1
    }

    Write-Host ""
    Write-Ok "Build successful!"

    # List built binaries
    Write-Host ""
    Write-Step "Built artifacts:"
    $exeGlob = Join-Path $BuildPath "*.exe"
    $exeGlobSub = Join-Path $BuildPath "$BuildType\*.exe"
    $exes = @()
    if (Test-Path $exeGlob) { $exes += Get-ChildItem $exeGlob }
    if (Test-Path $exeGlobSub) { $exes += Get-ChildItem $exeGlobSub }
    $exes | ForEach-Object {
        $size = [math]::Round($_.Length / 1MB, 2)
        Write-Host "   $($_.Name)  ($size MB)" -ForegroundColor White
    }
}

# ─── Tests ────────────────────────────────────────────────────────────────────
if ($Test) {
    Write-Host ""
    Write-Host "=" * 72 -ForegroundColor White
    Write-Step "Running tests..."
    Write-Host "=" * 72 -ForegroundColor White
    Write-Host ""

    Push-Location $BuildPath
    & ctest --build-config $BuildType --output-on-failure --verbose
    $testResult = $LASTEXITCODE
    Pop-Location

    if ($testResult -ne 0) {
        Write-Warn "Some tests failed! (exit code: $testResult)"
    } else {
        Write-Ok "All tests passed!"
    }
}

# ─── Benchmark ────────────────────────────────────────────────────────────────
if ($Bench) {
    Write-Host ""
    Write-Host "=" * 72 -ForegroundColor White
    Write-Step "Running benchmarks..."
    Write-Host "=" * 72 -ForegroundColor White
    Write-Host ""

    # Ninja puts exe in build root, VS puts it in build/Release/
    $benchExe = Join-Path $BuildPath "vv_bench.exe"
    if (-not (Test-Path $benchExe)) {
        $benchExe = Join-Path $BuildPath "$BuildType\vv_bench.exe"
    }

    if (Test-Path $benchExe) {
        & $benchExe
    } else {
        Write-Warn "vv_bench.exe not found."
    }
}

# ─── Summary ──────────────────────────────────────────────────────────────────
Write-Host ""
Write-Ok "Done! Build directory: $BuildDir"
Write-Host ""
