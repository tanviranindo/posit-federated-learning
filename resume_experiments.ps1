# ==============================================================================
# Posit Federated Learning - Resume Parallel Experiments (Windows)
# ==============================================================================

$ErrorActionPreference = "Stop"
$PROJECT_ROOT = $PSScriptRoot

if (!(Test-Path -Path "$PROJECT_ROOT\.experiment_pids")) {
    Write-Host "❌ No .experiment_pids file found." -ForegroundColor Red
    exit 1
}

$pids = Get-Content -Path "$PROJECT_ROOT\.experiment_pids"
Write-Host "▶️ To resume the suspended processes on Windows:" -ForegroundColor Cyan
Write-Host "1. Open Resource Monitor (resmon.exe)"
Write-Host "2. Go to CPU tab -> Right click the python.exe processes with PIDs $pids -> Resume Process"
Write-Host "`n(A fully automated suspend/resume script requires downloading PsSuspend.exe from Microsoft Sysinternals)."
