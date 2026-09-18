@echo off
setlocal enabledelayedexpansion
REM ============================================================================
REM  vibevoice.c — Environment Setup Script
REM
REM  Run this once per terminal session to configure MSVC + CUDA.
REM  After running, you can use cmake/build.bat/ninja directly.
REM
REM  Usage:
REM    scripts\setup_env.bat
REM ============================================================================

echo.
echo  vibevoice.c Environment Setup
echo  ══════════════════════════════
echo.

REM ─── Step 1: MSVC ──────────────────────────────────────────────────────────
echo [1/3] Looking for MSVC...

where cl >nul 2>&1
if %errorlevel% equ 0 (
    echo       MSVC already in PATH — OK
    goto :check_cuda
)

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo [ERROR] vswhere.exe not found. Install Visual Studio 2022 Build Tools.
    echo         https://visualstudio.microsoft.com/downloads/
    goto :error
)

for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    set "VS_PATH=%%i"
)

if not defined VS_PATH (
    echo [ERROR] Visual Studio with C++ not found.
    goto :error
)

set "VCVARS=!VS_PATH!\VC\Auxiliary\Build\vcvars64.bat"
if not exist "!VCVARS!" (
    echo [ERROR] vcvars64.bat not found at: !VCVARS!
    goto :error
)

echo       Loading MSVC from: !VS_PATH!
REM We need to use endlocal trick to persist environment changes
endlocal
call "%VCVARS%" >nul 2>&1
setlocal enabledelayedexpansion
echo       MSVC: OK

:check_cuda
REM ─── Step 2: CUDA ──────────────────────────────────────────────────────────
echo [2/3] Looking for CUDA Toolkit...

if defined CUDA_PATH (
    if exist "%CUDA_PATH%\bin\nvcc.exe" (
        echo       CUDA_PATH: %CUDA_PATH%
        goto :check_cuda_ver
    )
)

REM Auto-detect
set "CUDA_BASE=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
if exist "%CUDA_BASE%" (
    for /d %%D in ("%CUDA_BASE%\v*") do (
        set "CUDA_PATH=%%D"
    )
)

if not defined CUDA_PATH (
    echo [ERROR] CUDA Toolkit not found!
    echo         Download from: https://developer.nvidia.com/cuda-toolkit
    goto :error
)

echo       Auto-detected: %CUDA_PATH%

:check_cuda_ver
set "PATH=%CUDA_PATH%\bin;%PATH%"
for /f "tokens=*" %%V in ('nvcc --version 2^>^&1 ^| findstr "release"') do (
    echo       nvcc: %%V
)

:check_cmake
REM ─── Step 3: CMake + Ninja ─────────────────────────────────────────────────
echo [3/3] Checking build tools...

where cmake >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] cmake not found! Add it to PATH.
    echo         Download from: https://cmake.org/download/
    goto :error
)
for /f "tokens=3" %%V in ('cmake --version 2^>^&1 ^| findstr "version"') do (
    echo       cmake: %%V
)

where ninja >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%V in ('ninja --version') do echo       ninja: %%V (recommended!)
) else (
    echo       ninja: not found (will use Visual Studio generator)
    echo       Install: pip install ninja  OR  choco install ninja
)

where git >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=3" %%V in ('git --version') do echo       git: %%V
)

REM ─── Summary ────────────────────────────────────────────────────────────────
echo.
echo  ══════════════════════════════════════════════════
echo  Environment ready! You can now run:
echo.
echo    build.bat                  — full Release build
echo    build.bat debug            — Debug build
echo    build.bat test             — build + run tests
echo    build.bat clean            — clean build dir
echo.
echo    .\build.ps1                — PowerShell alternative
echo    .\build.ps1 -BuildType Debug -Test -Verbose
echo  ══════════════════════════════════════════════════
echo.
exit /b 0

:error
echo.
echo  Setup failed. Fix the errors above and try again.
exit /b 1
