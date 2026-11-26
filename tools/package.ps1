param([string]$Publisher = "Alastor")
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot
$build = Join-Path $root "build"
$version = (Select-String -Path (Join-Path $root "CMakeLists.txt") -Pattern "project\(TEMPORAL VERSION ([0-9.]+)").Matches[0].Groups[1].Value
$bundles = "TEMPORAL.ofx.bundle", "TEMPORAL-Optic.ofx.bundle" | ForEach-Object { Join-Path $build $_ }
foreach ($b in $bundles) {
    $name = (Split-Path $b -Leaf) -replace "\.ofx\.bundle$", ""
    if (-not (Test-Path (Join-Path $b "Contents\Win64\$name.ofx"))) { throw "Missing $b. Run ./build.ps1 -OpticStudio first." }
}

$stage = Join-Path ([IO.Path]::GetTempPath()) ("temporal_pkg_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory $stage | Out-Null
Add-Type -AssemblyName System.IO.Compression, System.IO.Compression.FileSystem
$archive = [IO.Compression.ZipFile]::Open((Join-Path $stage "payload.zip"), "Create")
foreach ($b in $bundles) {
    $parent = Split-Path $b
    foreach ($f in Get-ChildItem $b -Recurse -File) {
        $name = $f.FullName.Substring($parent.Length + 1).Replace("\", "/")
        [void][IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive, $f.FullName, $name, "Optimal")
    }
}
$archive.Dispose()
Copy-Item (Join-Path $PSScriptRoot "installer\uninstall.ps1") $stage
Set-Content (Join-Path $stage "version.txt") "$version|$Publisher" -Encoding ascii -NoNewline

$net = Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319"
$dist = Join-Path $root "dist"
New-Item -ItemType Directory -Force $dist | Out-Null
$exe = Join-Path $dist "TEMPORAL-$version-win64.exe"
& (Join-Path $net "csc.exe") /nologo /target:winexe /optimize+ "/out:$exe" "/win32icon:$(Join-Path $root 'assets\brand\temporal.ico')" `
    "/resource:$(Join-Path $stage 'payload.zip'),payload.zip" "/resource:$(Join-Path $stage 'uninstall.ps1'),uninstall.ps1" `
    "/resource:$(Join-Path $stage 'version.txt'),version.txt" /reference:System.IO.Compression.dll `
    /reference:System.IO.Compression.FileSystem.dll /reference:System.Windows.Forms.dll (Join-Path $PSScriptRoot "installer\Installer.cs")
if ($LASTEXITCODE -ne 0) { throw "csc failed with exit code $LASTEXITCODE" }
Get-Item $exe | Select-Object FullName, Length
