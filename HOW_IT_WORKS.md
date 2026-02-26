# How Everything Connects

## The Problem

Federated learning across heterogeneous hardware (x86_64 servers + ARM64 edge devices) introduces two practical issues:

1. **Deployment consistency**: Different architectures need different BLAS libraries, build configurations, and runtime environments. Minor dependency mismatches cause non-reproducible results.
2. **Training-time numerical divergence**: Local training on different ISAs produces slightly different gradients and running statistics (due to BLAS implementation differences, BatchNorm accumulation, and compiler optimizations), which compound over federation rounds.

**Note:** Our [B4 bitwise reproducibility experiment](experiments/b4_compare.py) showed that the *aggregation step itself* (sequential weighted sum) is bitwise identical across ARM64 and x86_64 in PyTorch. The cross-architecture variance comes from local training differences, not from aggregation rounding.

## Our Approach

### 1. Docker Containerization (`src/docker/`)

Multi-stage Dockerfiles detect the target ISA at build time and produce self-contained environments with locked dependencies. This eliminates deployment inconsistency.

### 2. Posit Arithmetic with Quire Accumulation (`src/core/posit_engine.py`)

The quire data structure provides exact intermediate accumulation during the aggregation step:

```python
# Standard IEEE 754 aggregation
# Each addition introduces a rounding error (K-1 total for K clients)
result = weight1 * model1 + weight2 * model2 + weight3 * model3

# Quire-based aggregation
# All additions are exact; only one rounding at extraction
quire = QuireAccumulator()
quire.add_weighted_tensor(model1, weight1)  # Exact
quire.add_weighted_tensor(model2, weight2)  # Exact
quire.add_weighted_tensor(model3, weight3)  # Exact
result = quire.extract_result()             # Single rounding
```

While B4 showed that IEEE 754 aggregation is already bitwise identical across ISAs for *sequential* weighted sums, the quire approach provides guarantees that hold regardless of summation order, library implementation, or future changes to PyTorch internals. It also reduces the K rounding errors to 1, which matters when accumulating many clients.

### 3. Architecture-Adaptive Models (`src/models/adaptive_cnn.py`)

The CNN automatically adapts to the target hardware:
- **x86_64**: 32-64-128 channels, 1.28M parameters (full capacity)
- **ARM64**: 16-32-64 channels, 320K parameters (energy-efficient)

Parameter mapping handles the size mismatch during aggregation (global model parameters are sliced/padded per architecture).

## What Happens in a Federation Round

1. **Architecture Detection**: Each client identifies its ISA via `platform.machine()`
2. **Model Adaptation**: Network size adjusted for the hardware
3. **Local Training**: Standard SGD on local CIFAR-10 partition
4. **Parameter Upload**: Client sends model updates to server
5. **Quire Aggregation**: Server combines updates with exact arithmetic
6. **Global Distribution**: Updated model sent back to all clients

## Key Files

| File | Purpose |
|------|---------|
| `main_experiment.py` | Entry point (demo/quick/full modes) |
| `src/core/posit_engine.py` | Posit config, quire accumulator |
| `src/federated/cross_arch_trainer.py` | Per-client trainer with arch detection |
| `src/federated/posit_fedavg.py` | Posit-enhanced FedAvg |
| `src/models/adaptive_cnn.py` | Architecture-adaptive CNN |
| `experiments/b4_bitwise_repro.py` | Bitwise reproducibility test |
| `experiments/b4_compare.py` | Cross-ISA comparison |

## What We Measured

- **12.8% variance reduction** in heterogeneous (mixed x86_64 + ARM64) deployments
- **8.3% energy savings** on ARM64
- **0.98 deployment consistency** (vs 0.52 for standard FedAvg)
- **Comparable accuracy**: 83.92% vs 84.70% for best baseline (within 1%)
- **Client threshold**: Posit benefits appear at 5+ clients; at 2-3 clients, parameter mapping overhead dominates
