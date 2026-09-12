@echo off
setlocal DisableDelayedExpansion
set "SCRIPT_DIR=%~dp0"
REM Download the entire model repository. HF_TOKEN is inherited unless overridden.
set "OUTPUT_DIR=model_hf"
set "REPO=scerz/VibeVoice-ASR-4bit"
set "TOKENIZER_REPO=Qwen/Qwen2.5-7B"
set "REVISION=main"
set "FORCE="

:parse_args
if "%~1"=="" goto :download
if /i "%~1"=="--help" goto :usage
if /i "%~1"=="-h" goto :usage
if /i "%~1"=="--force" (
    set "FORCE=-Force"
    shift
    goto :parse_args
)
if /i "%~1"=="--output" goto :value
if /i "%~1"=="-o" goto :value
if /i "%~1"=="--repo" goto :value
if /i "%~1"=="--tokenizer-repo" goto :value
if /i "%~1"=="--revision" goto :value
if /i "%~1"=="--token" goto :value
echo [vibevoice] Unknown argument: %~1
exit /b 1

:value
if "%~2"=="" goto :missing_value
set "VALUE=%~2"
if "%VALUE:~0,2%"=="--" goto :missing_value
if /i "%~1"=="--output" set "OUTPUT_DIR=%~2"
if /i "%~1"=="-o" set "OUTPUT_DIR=%~2"
if /i "%~1"=="--repo" set "REPO=%~2"
if /i "%~1"=="--tokenizer-repo" set "TOKENIZER_REPO=%~2"
if /i "%~1"=="--revision" set "REVISION=%~2"
if /i "%~1"=="--token" set "HF_TOKEN=%~2"
shift
shift
goto :parse_args

:download
REM -File passes arguments as data, including paths containing apostrophes or !.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%download_model.ps1" -OutputDir "%OUTPUT_DIR%" -Repo "%REPO%" -TokenizerRepo "%TOKENIZER_REPO%" -Revision "%REVISION%" %FORCE%
exit /b %errorlevel%

:missing_value
echo [vibevoice] Missing value for %~1
exit /b 1

:usage
echo Usage: download_model.bat [OPTIONS]
echo.
echo   --output DIR          Output directory (default: model_hf)
echo   --repo REPO           Download this entire Hugging Face model repository
echo                         (default: scerz/VibeVoice-ASR-4bit)
echo   --revision REF        Branch, tag, or commit (default: main)
echo   --tokenizer-repo REPO  Fallback only when the model lacks tokenizer.json
echo   --token TOKEN         Access token (or set HF_TOKEN)
echo   --force               Re-download all repository files
echo   -h, --help            Show help
exit /b 0
