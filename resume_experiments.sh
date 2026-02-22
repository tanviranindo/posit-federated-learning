#!/bin/bash
# Resumes the paused experiments by sending a SIGCONT signal
echo "Waking up suspended experiment processes..."
pkill -CONT -f "main_experiment.py --mode full"
echo "▶️  Experiments resumed. They will pick up exactly where they left off!"
