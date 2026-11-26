[CmdletBinding(SupportsShouldProcess)]
param([string]$Bundle = (Join-Path $PSScriptRoot "build\TEMPORAL.ofx.bundle"))

$ErrorActionPreference = "Stop"
$target = "C:\Program Files\Common Files\OFX\Plugins"
$folder = Get-Item -LiteralPath $Bundle -ErrorAction SilentlyContinue
if (!$folder -or !$folder.PSIsContainer -or $folder.Name -notin "TEMPORAL.ofx.bundle", "TEMPORAL-Optic.ofx.bundle") {
    throw "Expected a TEMPORAL or TEMPORAL-Optic OFX bundle: $Bundle"
}
$plugin = [IO.Path]::GetFileNameWithoutExtension($folder.Name)
if (!(Test-Path -LiteralPath (Join-Path $folder.FullName "Contents\Win64\$plugin") -PathType Leaf)) {
    throw "Missing Contents\Win64\$plugin in $($folder.FullName)"
}
$destination = Join-Path $target $folder.Name
if (!$PSCmdlet.ShouldProcess($destination, "Install OFX bundle")) { return }

$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (!$principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    $child = Start-Process powershell -Verb RunAs -Wait -PassThru -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Bundle `"$($folder.FullName)`""
    if ($child.ExitCode -ne 0) { throw "Elevated installation failed with exit code $($child.ExitCode)" }
    return
}
if (Get-Process Resolve -ErrorAction SilentlyContinue) { throw "Quit DaVinci Resolve before installing (the .ofx is locked while it runs)." }
New-Item -ItemType Directory -Force $target | Out-Null
Copy-Item -Recurse -Force -LiteralPath $folder.FullName -Destination $target
Write-Host "Installed to $destination"
