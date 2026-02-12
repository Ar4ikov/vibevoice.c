@echo off
setlocal enabledelayedexpansion
REM ─────────────────────────────────────────────────────────────────────────────
REM  download_model.bat — Download VibeVoice-ASR model from HuggingFace
REM
REM  Usage:
REM    download_model.bat                             -- default: 4-bit model -> .\model_hf
REM    download_model.bat --output D:\models\vv       -- custom output dir
REM    download_model.bat --token hf_xxxxxxxxx        -- with HF token
REM    download_model.bat --force                     -- re-download all files
REM    download_model.bat --repo user/OtherModel      -- different HF repo
REM ─────────────────────────────────────────────────────────────────────────────

set "SCRIPT_DIR=%~dp0"
set "OUTPUT_DIR=model_hf"
set "REPO=scerz/VibeVoice-ASR-4bit"
set "TOKENIZER_REPO=Qwen/Qwen2.5-7B"
set "HF_TOKEN="
set "FORCE="

REM --- Parse arguments ---------------------------------------------------------
:parse_args
if "%~1"=="" goto :done_args
if /i "%~1"=="--output"  (set "OUTPUT_DIR=%~2" & shift & shift & goto :parse_args)
if /i "%~1"=="-o"        (set "OUTPUT_DIR=%~2" & shift & shift & goto :parse_args)
if /i "%~1"=="--repo"    (set "REPO=%~2"       & shift & shift & goto :parse_args)
if /i "%~1"=="--token"   (set "HF_TOKEN=%~2"   & shift & shift & goto :parse_args)
if /i "%~1"=="--force"   (set "FORCE=-Force"   & shift & goto :parse_args)
if /i "%~1"=="--help" goto :usage
if /i "%~1"=="-h"     goto :usage
echo [vibevoice] Unknown argument: %~1
goto :usage
:done_args

REM --- Run the PowerShell script ----------------------------------------------
echo [vibevoice] Downloading model...
echo [vibevoice] Output:  %OUTPUT_DIR%
echo [vibevoice] Repo:    %REPO%
echo.

set "PS_CMD=& '%SCRIPT_DIR%download_model.ps1' -OutputDir '%OUTPUT_DIR%' -Repo '%REPO%' -TokenizerRepo '%TOKENIZER_REPO%'"
if defined HF_TOKEN set "PS_CMD=!PS_CMD! -Token '%HF_TOKEN%'"
if defined FORCE    set "PS_CMD=!PS_CMD! -Force"

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "!PS_CMD!"
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Download failed. See output above for details.
    exit /b 1
)

exit /b 0

REM --- Usage ------------------------------------------------------------------
:usage
echo.
echo  Usage: download_model.bat [OPTIONS]
echo.
echo  Options:
echo    --output DIR     Output directory  (default: model_hf)
echo    --repo   REPO    HuggingFace model repo (default: scerz/VibeVoice-ASR-4bit)
echo    --token  TOKEN   HuggingFace access token (for gated repos)
echo    --force          Re-download files even if they exist
echo    -h, --help       Show this help
echo.
echo  Examples:
echo    download_model.bat
echo    download_model.bat --output D:\models\vibevoice
echo    download_model.bat --token hf_xxxxxxxxxxxx --force
echo.
exit /b 0
