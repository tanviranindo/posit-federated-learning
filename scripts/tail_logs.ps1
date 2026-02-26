# ==============================================================================
# Posit Federated Learning - Tail Logs Dashboard (Windows)
# ==============================================================================
# Usage: .\tail_logs.ps1 -Timestamp "20240223_120000"
# ==============================================================================

param (
    [string]$Timestamp
)

if (-not $Timestamp) {
    Write-Host "Please provide the timestamp suffix for the log files. Example: .\tail_logs.ps1 -Timestamp 20240223_120000" -ForegroundColor Red
    
    # Try to find the latest log file automatically
    $latestLog = Get-ChildItem -Path "logs" -Filter "scenario_1_*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latestLog) {
        $Timestamp = $latestLog.Name.Replace("scenario_1_", "").Replace(".log", "")
        Write-Host "Automatically detected latest run: $Timestamp" -ForegroundColor Green
    } else {
        exit
    }
}

Write-Host "[*] Monitoring All 4 Scenarios..." -ForegroundColor Cyan

# Use Get-Content with -Tail and -Wait
Get-Content -Path "logs\scenario_*_$Timestamp.log" -Tail 5 -Wait
