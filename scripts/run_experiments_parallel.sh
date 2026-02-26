#!/bin/bash

# ==============================================================================
# Posit Federated Learning - Parallel Experiment Runner
# ==============================================================================
# This script runs the 4 Major Scenarios in parallel to greatly speed up 
# execution time on multi-core systems.
#
# Usage: ./run_experiments_parallel.sh
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

echo "======================================================================"
echo "⚡ Starting Full Phase 2 Federated Learning Experiments (PARALLEL MODE)"
echo "======================================================================"

# Stop the sequential run if it is still running
echo "Stopping any existing sequential experiment runs..."
pkill -f "main_experiment.py --mode full" || echo "No sequential runs found."

# Create logs directory
mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo ""
echo "⏳ Launching Scenario 1 (Precision Validation) in background..."
nohup python3 main_experiment.py --mode full --scenario 1 > logs/scenario_1_${TIMESTAMP}.log 2>&1 &
PID1=$!

echo "⏳ Launching Scenario 2 (Performance Analysis) in background..."
nohup python3 main_experiment.py --mode full --scenario 2 > logs/scenario_2_${TIMESTAMP}.log 2>&1 &
PID2=$!

echo "⏳ Launching Scenario 3 (Scalability Analysis) in background..."
nohup python3 main_experiment.py --mode full --scenario 3 > logs/scenario_3_${TIMESTAMP}.log 2>&1 &
PID3=$!

echo "⏳ Launching Scenario 4 (Comprehensive Comparison) in background..."
nohup python3 main_experiment.py --mode full --scenario 4 > logs/scenario_4_${TIMESTAMP}.log 2>&1 &
PID4=$!

echo ""
echo "✅ All 4 Scenarios launched in parallel!"
echo "PID 1 (Precision): ${PID1}"
echo "PID 2 (Performance): ${PID2}"
echo "PID 3 (Scalability): ${PID3}"
echo "PID 4 (Comprehensive): ${PID4}"
echo ""
echo "📁 Output is being written to the 'logs' directory."
echo ""
echo "📊 To monitor progress in real-time, run:"
echo "   tail -f logs/scenario_*_${TIMESTAMP}.log"
echo ""
echo "🛑 To stop all parallel experiments early, run:"
echo "   kill ${PID1} ${PID2} ${PID3} ${PID4}"
echo "======================================================================"
