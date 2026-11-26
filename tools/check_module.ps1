param([Parameter(Mandatory)][string]$Id, [Parameter(Mandatory)][string]$Maker, [switch]$Bench)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
$out = Join-Path $root "build-mod\$Id"
New-Item -ItemType Directory -Force $out | Out-Null
$reg = Join-Path $out "reg.cpp"
@"
#include "modules/modules.h"
namespace tp {
const std::vector<const ModuleDef*>& allModules() {
    static const std::vector<const ModuleDef*> l = [] { auto* m = new ModuleDef($Maker()); addFrameworkParams(*m); return std::vector<const ModuleDef*>{m}; }();
    return l;
}
}
"@ | Set-Content -Encoding UTF8 $reg
$common = "/nologo /std:c++20 /O2 /EHsc /utf-8 /permissive- /arch:AVX2 /fp:precise /W3 /wd4244 /wd4267 /wd4305 /DNOMINMAX /D_USE_MATH_DEFINES /I`"$root\src`" /I`"$root\tests`""
$srcs = "`"$root\src\gpu\gpu_stub.cpp`" `"$root\src\core\framework.cpp`" `"$root\src\modules\$Id.cpp`" `"$reg`""
$tests = "`"$root\tests\tests.cpp`""
$modTest = Join-Path $root "tests\module_$Id.cpp"
if (Test-Path $modTest) { $tests += " `"$modTest`"" }
$cmd = "`"$vcvars`" >nul && cl $common $srcs $tests /Fo`"$out\\`" /Fe`"$out\tp_tests.exe`" && `"$out\tp_tests.exe`""
if ($Bench) { $cmd += " && cl $common $srcs `"$root\tests\bench.cpp`" /Fo`"$out\\`" /Fe`"$out\tp_bench.exe`" && `"$out\tp_bench.exe`"" }
cmd /c $cmd
exit $LASTEXITCODE
