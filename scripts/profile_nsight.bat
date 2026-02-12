@echo off
setlocal enabledelayedexpansion
REM ============================================================================
REM  vibevoice.c — Profile with NVIDIA Nsight Systems
REM
REM  Usage:
REM    scripts\profile_nsight.bat [path_to_audio.wav] [extra nsys args...]
REM
REM  Prerequisites:
REM    - Build with NVTX: build.ps1 -NVTX   OR set VV_ENABLE_NVTX=ON in cmake
REM    - Nsight Systems installed (comes with CUDA Toolkit or standalone)
REM
REM  Output:
REM    - Creates .nsys-rep file in profiles/ directory
REM    - Open with Nsight Systems GUI for analysis
REM ============================================================================

set "PROJECT_ROOT=%~dp0.."
set "PROFILE_DIR=%PROJECT_ROOT%\profiles"
set "BUILD_DIR=%PROJECT_ROOT%\build"

REM Find vv_cli executable
set "CLI_EXE="
if exist "%BUILD_DIR%\vv_cli.exe" set "CLI_EXE=%BUILD_DIR%\vv_cli.exe"
if exist "%BUILD_DIR%\Release\vv_cli.exe" set "CLI_EXE=%BUILD_DIR%\Release\vv_cli.exe"
if exist "%BUILD_DIR%\RelWithDebInfo\vv_cli.exe" set "CLI_EXE=%BUILD_DIR%\RelWithDebInfo\vv_cli.exe"

if not defined CLI_EXE (
    echo [ERROR] vv_cli.exe not found. Build the project first:
    echo         build.bat release
    exit /b 1
)

REM Check nsys
where nsys >nul 2>&1
if %errorlevel% neq 0 (
    REM Try CUDA path
    if exist "%CUDA_PATH%\bin\nsys.exe" (
        set "PATH=%CUDA_PATH%\bin;%PATH%"
    ) else (
        echo [ERROR] nsys (Nsight Systems) not found in PATH.
        echo         Install from: https://developer.nvidia.com/nsight-systems
        exit /b 1
    )
)

REM Create profiles directory
if not exist "%PROFILE_DIR%" mkdir "%PROFILE_DIR%"

REM Generate timestamp for output
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set "DT=%%I"
set "TIMESTAMP=%DT:~0,8%_%DT:~8,6%"

set "AUDIO_FILE=%~1"
if "%AUDIO_FILE%"=="" set "AUDIO_FILE=test.wav"

echo [vibevoice] Profiling with Nsight Systems...
echo [vibevoice] CLI: %CLI_EXE%
echo [vibevoice] Audio: %AUDIO_FILE%
echo [vibevoice] Output: %PROFILE_DIR%\vv_profile_%TIMESTAMP%.nsys-rep
echo.

nsys profile ^
    --output="%PROFILE_DIR%\vv_profile_%TIMESTAMP%" ^
    --trace=cuda,nvtx,osrt ^
    --cuda-memory-usage=true ^
    --stats=true ^
    --force-overwrite=true ^
    "%CLI_EXE%" --input "%AUDIO_FILE%" %2 %3 %4 %5

if %errorlevel% neq 0 (
    echo [ERROR] Profiling failed!
    exit /b 1
)

echo.
echo [vibevoice] Profile saved to: %PROFILE_DIR%\vv_profile_%TIMESTAMP%.nsys-rep
echo [vibevoice] Open with: nsys-ui "%PROFILE_DIR%\vv_profile_%TIMESTAMP%.nsys-rep"
