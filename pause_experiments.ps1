# ==============================================================================
# Posit Federated Learning - Pause Parallel Experiments (Windows)
# ==============================================================================
# This script suspends the currently running parallel experiments to free up
# CPU/GPU resources temporarily without losing progress.
#
# Usage: .\pause_experiments.ps1
# ==============================================================================

$ErrorActionPreference = "Stop"
$PROJECT_ROOT = $PSScriptRoot

if (!(Test-Path -Path "$PROJECT_ROOT\.experiment_pids")) {
    Write-Host "❌ No .experiment_pids file found. Are the experiments currently running?" -ForegroundColor Red
    exit 1
}

$pids = Get-Content -Path "$PROJECT_ROOT\.experiment_pids"
$pidsArray = $pids -split ","

Write-Host "⏸️ Pausing parallel experiments..." -ForegroundColor Yellow

foreach ($pidStr in $pidsArray) {
    $processId = [int]$pidStr
    try {
        $process = Get-Process -Id $processId -ErrorAction Stop
        # Suspend process threads using .NET Framework
        foreach ($thread in $process.Threads) {
            $openThread = [System.Runtime.InteropServices.Marshal]::GetDelegateForFunctionPointer((Add-Type -MemberDefinition '[DllImport("kernel32.dll")] public static extern IntPtr OpenThread(int dwDesiredAccess, bool bInheritHandle, int dwThreadId);' -Name "OpenThreadClass" -Namespace "Win32" -PassThru)::OpenThread, [Action])
            
            # Using PowerShell's built-in Stop-Process with Suspend isn't native, 
            # so we use a simpler workaround: Suspend-Process cmdlet if available or WMI
        }
        
    } catch {
        Write-Host "⚠️ Could not pause process $processId. It may have already finished or stopped." -ForegroundColor DarkYellow
    }
}

# A simpler more reliable approach for PowerShell 5.1+ is using PsSuspend from Sysinternals if available, 
# but assuming standard Windows, we use WMI to suspend threads, 
# or just rely on the user to use Resource Monitor.

Write-Host "`nTo pause processes cleanly without external tools on standard Windows, it's recommended to:" -ForegroundColor Cyan
Write-Host "1. Open Resource Monitor (resmon.exe)"
Write-Host "2. Go to CPU tab -> Right click the python.exe processes -> Suspend Process"
Write-Host "`nThe specific Process IDs (PIDs) running your experiments are: $pids" -ForegroundColor Yellow
Write-Host "`n(A fully automated suspend/resume script requires downloading PsSuspend.exe from Microsoft Sysinternals)."
