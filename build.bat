@echo off
setlocal enabledelayedexpansion
REM ============================================================================
REM  vibevoice.c -- Windows Build Script (CMD)
REM
REM  Usage:
REM    build.bat                     -- Release build, auto-detect everything
REM    build.bat debug               -- Debug build
REM    build.bat release             -- Release build (default)
REM    build.bat relwithdebinfo      -- Release + debug info
REM    build.bat clean               -- Remove build directory
REM    build.bat test                -- Build + run tests
REM    build.bat bench               -- Build + run benchmarks
REM    build.bat rebuild             -- Clean + full rebuild
REM
REM  Environment variables (optional overrides):
REM    CUDA_PATH           -- CUDA Toolkit root (auto-detected if not set)
REM    VV_CUDA_ARCH        -- CUDA architectures, e.g. "80;86;89" (default)
REM    VV_BUILD_DIR        -- Build directory (default: build)
REM    VV_GENERATOR        -- CMake generator (default: "Ninja" if found, else VS)
REM    VV_JOBS             -- Parallel jobs for build (default: all cores)
REM ============================================================================

set "SCRIPT_DIR=%~dp0"
REM Remove trailing backslash -- cmake -S "path\" breaks (backslash escapes the quote)
set "PROJECT_ROOT=%SCRIPT_DIR:~0,-1%"

REM --- Parse arguments --------------------------------------------------------
set "BUILD_TYPE=Release"
set "ACTION=build"
set "RUN_TESTS=0"
set "RUN_BENCH=0"

if /i "%~1"=="debug"           (set "BUILD_TYPE=Debug"          & shift)
if /i "%~1"=="release"         (set "BUILD_TYPE=Release"        & shift)
if /i "%~1"=="relwithdebinfo"  (set "BUILD_TYPE=RelWithDebInfo" & shift)
if /i "%~1"=="clean"           (set "ACTION=clean"              & shift)
if /i "%~1"=="rebuild"         (set "ACTION=rebuild"            & shift)
if /i "%~1"=="test"            (set "RUN_TESTS=1"               & shift)
if /i "%~1"=="bench"           (set "RUN_BENCH=1"               & shift)

REM --- Build directory --------------------------------------------------------
if not defined VV_BUILD_DIR set "VV_BUILD_DIR=build"

REM --- Clean action -----------------------------------------------------------
if "%ACTION%"=="clean" (
    echo [vibevoice] Cleaning build directory: %VV_BUILD_DIR%
    if exist "%VV_BUILD_DIR%" rmdir /s /q "%VV_BUILD_DIR%"
    echo [vibevoice] Clean complete.
    exit /b 0
)
if "%ACTION%"=="rebuild" (
    echo [vibevoice] Rebuild: cleaning first...
    if exist "%VV_BUILD_DIR%" rmdir /s /q "%VV_BUILD_DIR%"
)

REM --- Detect MSVC (Visual Studio) -- must be x64 ----------------------------
set "NEED_MSVC=0"
where cl >nul 2>&1
if %errorlevel% neq 0 (
    set "NEED_MSVC=1"
) else (
    REM Check if current cl.exe is x64 -- CUDA requires 64-bit host compiler
    set "CL_ARCH="
    for /f "tokens=*" %%L in ('cl 2^>^&1 ^| findstr /i "x64 amd64"') do set "CL_ARCH=x64"
    if not defined CL_ARCH (
        echo [vibevoice] WARNING: 32-bit MSVC compiler detected -- CUDA requires x64
        echo [vibevoice] Loading x64 toolchain...
        set "NEED_MSVC=1"
        REM Stale CMake cache from x86 run will break linking -- force clean
        if exist "!VV_BUILD_DIR!\CMakeCache.txt" (
            echo [vibevoice] Removing stale x86 CMake cache...
            rmdir /s /q "!VV_BUILD_DIR!"
        )
    )
)
if "!NEED_MSVC!"=="1" (
    echo [vibevoice] Setting up x64 MSVC environment...
    call :find_msvc
    if !errorlevel! neq 0 (
        echo [ERROR] Could not find MSVC x64. Please run from "x64 Native Tools Command Prompt"
        echo         or install Visual Studio 2022 with C++ workload.
        exit /b 1
    )
)
echo [vibevoice] MSVC: OK (x64)

REM --- Detect CUDA ------------------------------------------------------------
if not defined CUDA_PATH (
    if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" (
        for /d %%D in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*") do (
            set "CUDA_PATH=%%D"
        )
    )
)
if not defined CUDA_PATH (
    echo [ERROR] CUDA Toolkit not found. Set CUDA_PATH environment variable.
    exit /b 1
)
echo [vibevoice] CUDA: %CUDA_PATH%

REM Ensure nvcc is on PATH
set "PATH=%CUDA_PATH%\bin;%PATH%"

REM --- CUDA architectures -----------------------------------------------------
if not defined VV_CUDA_ARCH set "VV_CUDA_ARCH=80;86;89"
echo [vibevoice] CUDA architectures: %VV_CUDA_ARCH%

REM --- Detect generator -------------------------------------------------------
if not defined VV_GENERATOR (
    where ninja >nul 2>&1
    if !errorlevel! equ 0 (
        set "VV_GENERATOR=Ninja"
    ) else (
        set "VV_GENERATOR=Visual Studio 17 2022"
    )
)
echo [vibevoice] Generator: %VV_GENERATOR%

REM --- Parallel jobs ----------------------------------------------------------
if not defined VV_JOBS set "VV_JOBS=%NUMBER_OF_PROCESSORS%"

REM --- Configure --------------------------------------------------------------
echo.
echo ========================================================================
echo  Configuring vibevoice.c  [%BUILD_TYPE%]
echo ========================================================================
echo.

if not exist "%VV_BUILD_DIR%" mkdir "%VV_BUILD_DIR%"

echo [vibevoice] cmake -S "%PROJECT_ROOT%" -B "%VV_BUILD_DIR%" -G "%VV_GENERATOR%" ...
cmake -S "%PROJECT_ROOT%" -B "%VV_BUILD_DIR%" -G "%VV_GENERATOR%" -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DCMAKE_CUDA_ARCHITECTURES="%VV_CUDA_ARCH%" -DVV_BUILD_TESTS=ON -DVV_BUILD_BENCH=ON -DVV_BUILD_CLI=ON

if %errorlevel% neq 0 (
    echo [ERROR] CMake configure failed!
    exit /b 1
)

REM --- Build ------------------------------------------------------------------
echo.
echo ========================================================================
echo  Building vibevoice.c  [%BUILD_TYPE%]  (%VV_JOBS% jobs)
echo ========================================================================
echo.

cmake --build "%VV_BUILD_DIR%" --config %BUILD_TYPE% --parallel %VV_JOBS%
if %errorlevel% neq 0 (
    echo [ERROR] Build failed!
    exit /b 1
)

echo.
echo [vibevoice] Build successful!
echo [vibevoice] Binaries in: %VV_BUILD_DIR%\

REM --- Run tests --------------------------------------------------------------
if "%RUN_TESTS%"=="1" (
    echo.
    echo ========================================================================
    echo  Running tests
    echo ========================================================================
    echo.
    cd "%VV_BUILD_DIR%"
    ctest --build-config %BUILD_TYPE% --output-on-failure --verbose
    set "TEST_RESULT=!errorlevel!"
    cd "%PROJECT_ROOT%"
    if !TEST_RESULT! neq 0 (
        echo [WARNING] Some tests failed!
    ) else (
        echo [vibevoice] All tests passed!
    )
)

REM --- Run benchmarks ---------------------------------------------------------
if "%RUN_BENCH%"=="1" (
    echo.
    echo ========================================================================
    echo  Running benchmarks
    echo ========================================================================
    echo.
    if "%VV_GENERATOR%"=="Ninja" (
        set "BENCH_EXE=%VV_BUILD_DIR%\vv_bench.exe"
    ) else (
        set "BENCH_EXE=%VV_BUILD_DIR%\%BUILD_TYPE%\vv_bench.exe"
    )
    if exist "!BENCH_EXE!" (
        "!BENCH_EXE!"
    ) else (
        echo [WARNING] vv_bench.exe not found at: !BENCH_EXE!
    )
)

echo.
echo [vibevoice] Done.
exit /b 0

REM --- Helper: Find MSVC via vswhere ------------------------------------------
:find_msvc
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo [vibevoice] vswhere not found
    exit /b 1
)

for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    set "VS_PATH=%%i"
)

if not defined VS_PATH (
    echo [vibevoice] No Visual Studio with C++ found
    exit /b 1
)

echo [vibevoice] Found Visual Studio at: %VS_PATH%

set "VCVARS=%VS_PATH%\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
    echo [vibevoice] vcvars64.bat not found
    exit /b 1
)

call "%VCVARS%" >nul 2>&1
echo [vibevoice] MSVC environment loaded
exit /b 0
