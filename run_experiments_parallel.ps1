# ==============================================================================
# Posit Federated Learning - Parallel Experiment Runner (Windows)
# ==============================================================================
# This script runs the 4 Major Scenarios in parallel to greatly speed up 
# execution time on multi-core systems.
#
# Usage: .\run_experiments_parallel.ps1
# ==============================================================================

$ErrorActionPreference = "Stop"

$PROJECT_ROOT = $PSScriptRoot
Set-Location -Path $PROJECT_ROOT

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "⚡ Starting Full Phase 2 Federated Learning Experiments (PARALLEL MODE)" -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Cyan

# Stop the sequential run if it is still running
Write-Host "Stopping any existing sequential experiment runs..."
Get-Process | Where-Object { $_.CommandLine -like "*main_experiment.py --mode full*" } | Stop-Process -Force -ErrorAction SilentlyContinue

# Create logs directory
if (!(Test-Path -Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}
$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"

# Activate Python Virtual Environment
$env:PYTHONPATH = $PROJECT_ROOT
$VENV_ACTIVATE = ".\venv\Scripts\activate.ps1"

Write-Host "`n⏳ Launching Scenario 1 (Precision Validation) in background..." -ForegroundColor Yellow
$scenario1 = Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile -ExecutionPolicy Bypass -Command `". $VENV_ACTIVATE; python main_experiment.py --mode full --scenario 1 > logs\scenario_1_$TIMESTAMP.log 2>&1`"" -PassThru -WindowStyle Hidden

Write-Host "⏳ Launching Scenario 2 (Performance Analysis) in background..." -ForegroundColor Yellow
$scenario2 = Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile -ExecutionPolicy Bypass -Command `". $VENV_ACTIVATE; python main_experiment.py --mode full --scenario 2 > logs\scenario_2_$TIMESTAMP.log 2>&1`"" -PassThru -WindowStyle Hidden

Write-Host "⏳ Launching Scenario 3 (Scalability Analysis) in background..." -ForegroundColor Yellow
$scenario3 = Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile -ExecutionPolicy Bypass -Command `". $VENV_ACTIVATE; python main_experiment.py --mode full --scenario 3 > logs\scenario_3_$TIMESTAMP.log 2>&1`"" -PassThru -WindowStyle Hidden

Write-Host "⏳ Launching Scenario 4 (Comprehensive Comparison) in background..." -ForegroundColor Yellow
$scenario4 = Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile -ExecutionPolicy Bypass -Command `". $VENV_ACTIVATE; python main_experiment.py --mode full --scenario 4 > logs\scenario_4_$TIMESTAMP.log 2>&1`"" -PassThru -WindowStyle Hidden

# Save PIDs for pause/resume
$pids = @($scenario1.Id, $scenario2.Id, $scenario3.Id, $scenario4.Id)
$pids -join "," | Out-File -FilePath "$PROJECT_ROOT\.experiment_pids"

Write-Host "`n✅ All 4 Scenarios launched in parallel!" -ForegroundColor Green
Write-Host "PID 1 (Precision): $($scenario1.Id)"
Write-Host "PID 2 (Performance): $($scenario2.Id)"
Write-Host "PID 3 (Scalability): $($scenario3.Id)"
Write-Host "PID 4 (Comprehensive): $($scenario4.Id)"
Write-Host "`n📁 Output is being written to the 'logs' directory." -ForegroundColor Cyan
Write-Host "`n📊 To monitor progress in real-time, run:" -ForegroundColor Yellow
Write-Host "   .\tail_logs.ps1 -Timestamp $TIMESTAMP"
Write-Host "`n🛑 To stop all parallel experiments early, run:" -ForegroundColor Red
Write-Host "   Stop-Process -Id $($scenario1.Id), $($scenario2.Id), $($scenario3.Id), $($scenario4.Id) -Force"
Write-Host "======================================================================" -ForegroundColor Cyan
