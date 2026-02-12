@echo off
REM ============================================================================
REM  Install Ninja build system (recommended for faster builds)
REM
REM  Ninja is ~2-5x faster than MSBuild for incremental builds.
REM  This script downloads the latest Ninja release to the project's tools dir.
REM ============================================================================

set "INSTALL_DIR=%~dp0..\tools\bin"
set "NINJA_URL=https://github.com/nicknisi/ninja-builds/releases/latest"

echo [vibevoice] Checking for Ninja...

where ninja >nul 2>&1
if %errorlevel% equ 0 (
    echo [vibevoice] Ninja already installed:
    ninja --version
    echo [vibevoice] Nothing to do.
    exit /b 0
)

echo [vibevoice] Ninja not found. Attempting install...
echo.

REM Try pip first (most reliable on Windows with Python)
where pip >nul 2>&1
if %errorlevel% equ 0 (
    echo [vibevoice] Installing via pip...
    pip install ninja
    if %errorlevel% equ 0 (
        echo [vibevoice] Ninja installed successfully via pip!
        ninja --version
        exit /b 0
    )
)

REM Try winget
where winget >nul 2>&1
if %errorlevel% equ 0 (
    echo [vibevoice] Installing via winget...
    winget install Ninja-build.Ninja
    if %errorlevel% equ 0 (
        echo [vibevoice] Ninja installed successfully via winget!
        echo [vibevoice] You may need to restart your terminal.
        exit /b 0
    )
)

REM Try choco
where choco >nul 2>&1
if %errorlevel% equ 0 (
    echo [vibevoice] Installing via chocolatey...
    choco install ninja -y
    if %errorlevel% equ 0 (
        echo [vibevoice] Ninja installed successfully via chocolatey!
        exit /b 0
    )
)

echo.
echo [vibevoice] Automatic install failed. Please install manually:
echo   pip install ninja
echo   OR: winget install Ninja-build.Ninja
echo   OR: choco install ninja
echo   OR: Download from https://github.com/ninja-build/ninja/releases
echo.
exit /b 1
