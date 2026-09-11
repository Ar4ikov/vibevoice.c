<#
.SYNOPSIS
    Build the shipping Windows binary: one .exe, no CUDA toolkit on the
    target machine, every GPU from Turing to Blackwell.

.DESCRIPTION
    What makes it self-contained:
      - the CUDA runtime is linked statically, so there is no cudart64_*.dll
        to ship or find;
      - cuBLAS is not used at all (src/cuda/gemm.cu is hand-written WMMA),
        which keeps the binary at a few MB instead of shipping hundreds;
      - nvcuda.dll comes with the display driver and is loaded lazily by the
        static runtime, so the .exe also starts on a machine with no NVIDIA
        driver and falls back to the CPU kernels.

    Cubins are emitted for every architecture the installed toolkit knows,
    plus PTX for the newest, which the driver JITs for anything newer still.

.PARAMETER BuildDir
    Build output directory. Default: build-release
#>
param(
    [string]$BuildDir = "build-release"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not $env:CUDA_PATH) {
    $found = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" `
        -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending |
        Select-Object -First 1
    if (-not $found) { throw "CUDA toolkit not found; set CUDA_PATH" }
    $env:CUDA_PATH = $found.FullName
}
Write-Host "CUDA: $env:CUDA_PATH"

$nvccOut = & "$env:CUDA_PATH\bin\nvcc.exe" --version
$ver = [int](($nvccOut -join "`n" | Select-String 'release (\d+)\.(\d+)').Matches[0].Groups[1].Value) * 10 +
       [int](($nvccOut -join "`n" | Select-String 'release (\d+)\.(\d+)').Matches[0].Groups[2].Value)

# sm_100 and sm_120 only exist from CUDA 12.8 on; asking an earlier nvcc for
# them is a hard error, so the list is trimmed to what the toolkit knows.
$archs = if ($ver -ge 128) {
    "75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual"
} else {
    "75-real;80-real;86-real;89-real;90-real;90-virtual"
}
Write-Host "architectures: $archs"

cmake -B $BuildDir -G "Visual Studio 17 2022" -A x64 `
    -DCMAKE_BUILD_TYPE=Release `
    -DVV_ENABLE_TRT=OFF `
    -DVV_BUILD_TESTS=OFF `
    -DVV_BUILD_BENCH=OFF `
    -DCMAKE_CUDA_ARCHITECTURES="$archs"
if ($LASTEXITCODE -ne 0) { throw "configure failed" }

cmake --build $BuildDir --config Release --parallel
if ($LASTEXITCODE -ne 0) { throw "build failed" }

$exe = Join-Path $BuildDir "Release\vv_cli.exe"
if (-not (Test-Path $exe)) { $exe = Join-Path $BuildDir "vv_cli.exe" }
Write-Host ""
Write-Host "binary: $exe ($([math]::Round((Get-Item $exe).Length / 1MB, 1)) MB)"
