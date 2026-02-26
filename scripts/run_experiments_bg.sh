#!/bin/bash

# ==============================================================================
# Posit Federated Learning - Automated Experiment Runner
# ==============================================================================
# This script runs the full, scaled-up Phase 2 experiments in the background.
# It ensures the process continues even if the terminal is closed.
#
# Usage: ./run_experiments_bg.sh
# ==============================================================================

# Set up paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${PROJECT_ROOT}/experiment_full_run_$(date +%Y%m%d_%H%M%S).log"

echo "======================================================================"
echo "🚀 Starting Full Phase 2 Federated Learning Experiments"
echo "======================================================================"
echo "Dataset: CIFAR-10 (50,000 train / 10,000 test)"
echo "Clients: 10 (Mixed x86_64 and arm64)"
echo "Rounds: 50"
echo "Baselines: FedProx, Kahan Summation, Standard IEEE754"
echo "======================================================================"

# Ensure we are in the correct directory
cd "${PROJECT_ROOT}"

# Make sure psutil is installed (soft-fail if it complains about managed environments, 
# assuming we installed it earlier with --break-system-packages)
python3 -c "import psutil" 2>/dev/null || {
    echo "⚠️ Python psutil package might be missing. Attempting to install..."
    pip3 install psutil --break-system-packages || echo "⚠️ Proceeding anyway, but resource monitoring might fail."
}

# Run the experiment in the background using nohup
# This allows the user to close the terminal while the job finishes
echo ""
echo "⏳ Launching experiment in the background..."
nohup python3 main_experiment.py --mode full > "${LOG_FILE}" 2>&1 &

# Capture the Process ID
PID=$!

echo "✅ Experiment successfully launched!"
echo "PID: ${PID}"
echo ""
echo "📁 Output is being written to:"
echo "   ${LOG_FILE}"
echo ""
echo "📊 To monitor progress in real-time, run:"
echo "   tail -f ${LOG_FILE}"
echo ""
echo "🛑 To stop the experiment early, run:"
echo "   kill ${PID}"
echo "======================================================================"
