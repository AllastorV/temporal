param([switch]$Quiet)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Windows.Forms
function Show-Result([string]$text, [string]$icon) {
    if (-not $Quiet) { [System.Windows.Forms.MessageBox]::Show($text, "TEMPORAL", "OK", $icon) | Out-Null }
}
$root = Join-Path $env:LOCALAPPDATA "Temporal"
$plugins = Join-Path $root "OpenFX"
try {
    if (Get-Process Resolve -ErrorAction SilentlyContinue) { throw "Quit DaVinci Resolve first." }
    foreach ($name in "TEMPORAL.ofx.bundle", "TEMPORAL-Optic.ofx.bundle") {
        $target = Join-Path $plugins $name
        if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Recurse -Force }
    }
    $parts = @([Environment]::GetEnvironmentVariable("OFX_PLUGIN_PATH", "User") -split ";" | Where-Object { $_ -and $_ -ne $plugins })
    [Environment]::SetEnvironmentVariable("OFX_PLUGIN_PATH", $(if ($parts) { $parts -join ";" } else { $null }), "User")
    Remove-Item -Force "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\TEMPORAL" -ErrorAction SilentlyContinue
    Remove-Item -Force (Join-Path $root "uninstall.ps1") -ErrorAction SilentlyContinue
    Show-Result "TEMPORAL was removed." "Information"
} catch {
    Show-Result "Uninstall failed: $($_.Exception.Message)" "Error"
    exit 1
}
