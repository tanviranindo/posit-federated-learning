#!/bin/bash
# Pauses the running experiments by sending a SIGSTOP signal
echo "Suspending active experiment processes..."
pkill -STOP -f "main_experiment.py --mode full"
echo "⏸️  Experiments paused. The processes are safely sleeping in memory and using 0% CPU."
echo "You can resume them later by running: ./resume_experiments.sh"
