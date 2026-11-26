param([switch]$Test, [switch]$Bench, [string]$Filter = "", [switch]$OpticStudio)
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$ofx = Join-Path $root "third_party\openfx"
if (-not (Test-Path (Join-Path $ofx "Support\Library"))) {
    git clone https://github.com/AcademySoftwareFoundation/openfx.git $ofx
    git -C $ofx checkout 4249fe1b590a7e65caf4b1e4693b6b53220703f5
}
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw "MSVC build tools not found" }
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
$build = Join-Path $root "build"
$cmds = @("`"$vcvars`" >nul", "cmake -S `"$root`" -B `"$build`" -G Ninja -DCMAKE_BUILD_TYPE=Release", "cmake --build `"$build`"")
if ($OpticStudio) { $cmds += "cmake --build `"$build`" --target TEMPORAL_Optic" }
if ($Test) { $cmds += "`"$build\tp_tests.exe`" $Filter" }
if ($Bench) { $cmds += "`"$build\tp_bench.exe`" $Filter" }
cmd /c ($cmds -join " && ")
if ($LASTEXITCODE -ne 0) { throw "build/test failed with exit code $LASTEXITCODE" }
