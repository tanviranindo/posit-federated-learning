# Posit-Enhanced Federated Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A federated learning framework that combines Docker-based cross-architecture deployment with Posit arithmetic and quire-based exact accumulation. Tested on heterogeneous x86_64 + ARM64 deployments with CIFAR-10.

## Measured Results

Full-scale experiments: 50 rounds, 10 clients (5 x86_64 + 5 ARM64), CIFAR-10, Docker-deployed.

| Metric | Standard FedAvg | Our Method | Notes |
|--------|----------------|------------|-------|
| Accuracy | 84.70% | 83.92% | Comparable (within 1%) |
| Aggregation Variance | 0.0668 | 0.0583 | 12.8% reduction (heterogeneous) |
| Deployment Consistency | 0.52 | 0.98 | Highest across all baselines |
| ARM64 Energy | 0.195 Wh | 0.179 Wh | 8.3% savings |

**Important finding:** Posit variance benefits only appear at 5+ clients. At 2-3 clients, parameter mapping overhead dominates.

### Bitwise Reproducibility (B4 Experiment)

We tested whether IEEE 754 float32 aggregation produces different results across ISAs by running identical sequential weighted sums in Docker containers on ARM64 and x86_64 (same PyTorch version, same seeds, single-threaded).

**Result: aggregation is bitwise identical across ISAs.** The 12.8% variance reduction in heterogeneous deployments comes from differences in local training (BLAS libraries, BatchNorm running statistics, gradient computation), not from aggregation rounding. See `experiments/b4_compare.py` and `experiments/results/b4/b4_report.json`.

## Getting Started

```bash
git clone https://github.com/tanviranindo/posit-federated-learning.git
cd posit-federated-learning
pip install -r requirements.txt

# Quick demo (2-3 minutes, 2 clients, 3 rounds)
python main_experiment.py --mode demo

# Moderate test (10-15 minutes, 3 clients, 5 rounds)
python main_experiment.py --mode quick

# Full-scale (18+ hours, 10 clients, 50 rounds)
python main_experiment.py --mode full
```

CIFAR-10 is downloaded automatically on first run.

## Repository Structure

```
posit-federated-learning/
├── src/                          Source code
│   ├── core/posit_engine.py        Posit config, QuireAccumulator, PositTensor
│   ├── federated/
│   │   ├── cross_arch_trainer.py   Per-client trainer with architecture detection
│   │   └── posit_fedavg.py         Posit-enhanced FedAvg aggregation
│   ├── models/adaptive_cnn.py      Architecture-adaptive CNN (320K ARM64, 1.28M x86_64)
│   ├── data/cifar_federated.py     Federated CIFAR-10 splits (IID and Non-IID)
│   ├── docker/multi_arch_manager.py Multi-arch Docker image build and deployment
│   └── utils/metrics.py            Statistical analysis and precision metrics
├── experiments/                  Experiment runners and results
│   ├── comprehensive_research_experiment.py  Full 4-scenario runner
│   ├── b4_bitwise_repro.py       Bitwise reproducibility test (runs in Docker)
│   ├── b4_compare.py             Cross-ISA comparison script
│   ├── Dockerfile.b4             Minimal Docker image for B4 experiment
│   └── results/                  Raw experiment outputs
│       ├── experiment_results_full_scenario_{1..4}.json
│       └── b4/                   B4 cross-ISA comparison results
├── scripts/                      Automation (bash + PowerShell)
│   ├── run_experiments_bg.sh       Background experiment runner
│   ├── run_experiments_parallel.*  Parallel execution (sh + ps1)
│   ├── pause_experiments.*         Pause running experiments
│   ├── resume_experiments.*        Resume paused experiments
│   └── tail_logs.ps1               Live log viewer (Windows)
├── docker/                       Production Dockerfiles
│   ├── Dockerfile.x86_64
│   └── Dockerfile.arm64
├── main_experiment.py            Entry point (demo/quick/full modes)
├── requirements.txt              Python dependencies
├── EXPERIMENT_RESULTS_AND_OBSERVATIONS.md  Detailed analysis
├── HOW_IT_WORKS.md               Architecture overview
├── LICENSE                       MIT
└── README.md
```

## B4 Bitwise Reproducibility Experiment

To reproduce the cross-ISA aggregation test:

```bash
# Build identical images for both platforms
docker build --platform linux/arm64 -f experiments/Dockerfile.b4 -t b4:arm64 .
docker build --platform linux/amd64 -f experiments/Dockerfile.b4 -t b4:x86 .

# Run on both platforms
docker run --rm --platform linux/arm64 b4:arm64 > experiments/results/b4/b4_arm64.json
docker run --rm --platform linux/amd64 b4:x86 > experiments/results/b4/b4_x86.json

# Compare
python experiments/b4_compare.py \
    experiments/results/b4/b4_arm64.json \
    experiments/results/b4/b4_x86.json \
    --output experiments/results/b4/b4_report.json
```

Tests 8 scenarios (5/10 clients, float32/float64, model parameter shapes) across 10 random seeds. Float64 serves as a control.

## Docker Deployment

```bash
docker build -f docker/Dockerfile.x86_64 -t posit-fl:x86 .
docker build -f docker/Dockerfile.arm64 -t posit-fl:arm .
```

## Experiment Results

Raw results: `experiments/results/experiment_results_full_scenario_{1,2,3,4}.json`

B4 bitwise comparison: `experiments/results/b4/`

Detailed analysis: [`EXPERIMENT_RESULTS_AND_OBSERVATIONS.md`](EXPERIMENT_RESULTS_AND_OBSERVATIONS.md)

## License

MIT
