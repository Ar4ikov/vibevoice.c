@echo off
REM ============================================================================
REM  TurboQwen — GPU Info & Compatibility Check
REM
REM  Checks if your GPU meets the Ampere (SM 8.0+) minimum requirement.
REM ============================================================================

echo.
echo  TurboQwen GPU Compatibility Check
echo  ════════════════════════════════════
echo.

REM ─── nvidia-smi ─────────────────────────────────────────────────────────────
where nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] nvidia-smi not found. NVIDIA drivers not installed?
    exit /b 1
)

echo [GPU Info]
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,memory.free --format=csv,noheader
echo.

REM ─── Check compute capability ──────────────────────────────────────────────
for /f "tokens=2 delims=," %%C in ('nvidia-smi --query-gpu=compute_cap --format=csv,noheader') do (
    set "CC=%%C"
)

REM Trim whitespace
set "CC=%CC: =%"

echo [Compute Capability] %CC%

REM Parse major version (before the dot)
for /f "tokens=1 delims=." %%M in ("%CC%") do set "CC_MAJOR=%%M"

if %CC_MAJOR% GEQ 8 (
    echo [PASS] GPU is Ampere ^(SM 8.0^) or newer — compatible!
) else (
    echo [FAIL] GPU compute capability %CC% is below SM 8.0 ^(Ampere^).
    echo        Minimum supported: RTX 3060 / RTX 3070 / RTX 3080 / RTX 3090
    echo        or newer ^(RTX 40XX, RTX 50XX^).
    exit /b 1
)

REM ─── Check free memory ─────────────────────────────────────────────────────
echo.
for /f "tokens=1,2 delims=," %%A in ('nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits') do (
    set "TOTAL_MB=%%A"
    set "FREE_MB=%%B"
)
set "TOTAL_MB=%TOTAL_MB: =%"
set "FREE_MB=%FREE_MB: =%"

echo [VRAM] Total: %TOTAL_MB% MiB, Free: %FREE_MB% MiB

REM VibeVoice-ASR 4-bit needs roughly 5-6 GB
set /a REQUIRED_MB=6000
if %FREE_MB% GEQ %REQUIRED_MB% (
    echo [PASS] Enough free VRAM for VibeVoice-ASR 4-bit
) else (
    echo [WARN] Free VRAM might be tight. VibeVoice-ASR 4-bit needs ~5-6 GB.
    echo        Close other GPU applications to free memory.
)

REM ─── CUDA version ──────────────────────────────────────────────────────────
echo.
if defined CUDA_PATH (
    echo [CUDA Toolkit] %CUDA_PATH%
    if exist "%CUDA_PATH%\bin\nvcc.exe" (
        "%CUDA_PATH%\bin\nvcc.exe" --version 2>&1 | findstr "release"
    )
) else (
    echo [CUDA Toolkit] CUDA_PATH not set
)

echo.
echo  Check complete.
echo.
