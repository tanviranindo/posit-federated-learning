#!/usr/bin/env bash
# Run the full cross-ISA pilot on the current platform.
#
#   PY=python3 ROUNDS=20 bash experiments/xisa/run_pilot.sh
#
# Writes to experiments/results/xisa/<platform-tag>/ (override with TAG=...).
# Outputs: op probe at 1 thread and all threads, the FL trajectory twice at
# 1 thread (self-check), once at all threads, and once with a different
# minibatch order (seed-to-seed baseline).
set -euo pipefail

cd "$(dirname "$0")/../.."
PY="${PY:-python3}"
ROUNDS="${ROUNDS:-20}"
NPROC="$("$PY" -c 'import os; print(os.cpu_count())')"
TAG_ARGS=()
[[ -n "${TAG:-}" ]] && TAG_ARGS=(--tag "$TAG")

"$PY" experiments/xisa/prepare_data.py
cd experiments/xisa
"$PY" op_probe.py --threads 1 "${TAG_ARGS[@]}"
"$PY" op_probe.py --threads "$NPROC" "${TAG_ARGS[@]}"
"$PY" fl_trajectory.py --threads 1 --rounds "$ROUNDS" "${TAG_ARGS[@]}"
"$PY" fl_trajectory.py --threads 1 --rounds "$ROUNDS" --label _rep "${TAG_ARGS[@]}"
"$PY" fl_trajectory.py --threads "$NPROC" --rounds "$ROUNDS" "${TAG_ARGS[@]}"
"$PY" fl_trajectory.py --threads 1 --rounds "$ROUNDS" --order-seed 8 "${TAG_ARGS[@]}"
