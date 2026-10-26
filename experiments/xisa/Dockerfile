# Linux runner for the cross-ISA pilot. Build natively on each host (no QEMU):
#   docker build -f experiments/xisa/Dockerfile -t xisa:torch2.9.1 .
#   docker run --rm -v "$PWD":/work -w /work xisa:torch2.9.1 bash experiments/xisa/run_pilot.sh
FROM python:3.12-slim

RUN pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir numpy==2.3.5

ENV PY=python3
