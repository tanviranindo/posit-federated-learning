# Cross-platform matrix

Threads 1, order seed 7; all platforms saw identical inputs and initial weights.

## Platforms

| Platform | CPU | OS | BLAS | SIMD |
|---|---|---|---|---|
| darwin-arm64-accelerate | Apple M2 Pro | Darwin arm64 | accelerate | DEFAULT |
| gha-macos-latest | Apple M1 (Virtual) | Darwin arm64 | accelerate | DEFAULT |
| gha-ubuntu-24.04 | AMD EPYC 7763 64-Core Processor | Linux x86_64 | mkl | AVX2 |
| gha-ubuntu-24.04-arm | Neoverse-N2 | Linux aarch64 | open | DEFAULT |
| gha-windows-latest | AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD | Windows AMD64 | mkl | AVX2 |
| linux-arm64-open | Arm part 0x000 | Linux aarch64 | open | DEFAULT |

## Same-platform checks

| Platform | Rerun identical | 1 vs N threads: disagreement at last round | Seed baseline |
|---|---|---|---|
| darwin-arm64-accelerate | True | traj_t10_o7: 0.00% | 20.60% |
| gha-macos-latest | True | traj_t3_o7: 0.00% | 20.60% |
| gha-ubuntu-24.04 | True | traj_t4_o7: 0.00% | 20.15% |
| gha-ubuntu-24.04-arm | True | traj_t4_o7: 5.20% | 19.95% |
| gha-windows-latest | True | traj_t4_o7: 0.00% | 21.00% |
| linux-arm64-open | True | traj_t10_o7: 3.65% | 20.90% |

## Pairs

| Pair | Ops differing | Categories | Pred. disagreement r1 / r10 / last | Param rel. L2 |
|---|---|---|---|---|
| darwin-arm64-accelerate | gha-macos-latest | 0/42 | none | 0.00% / 0.00% / 0.00% | 0.0e+00 |
| darwin-arm64-accelerate | gha-ubuntu-24.04 | 34/42 | BatchNorm, GEMM, conv, element-wise, model fwd/bwd, reductions, transcendental | 0.35% / 1.60% / 3.20% | 8.4e-03 |
| darwin-arm64-accelerate | gha-ubuntu-24.04-arm | 24/42 | BatchNorm, GEMM, conv, model fwd/bwd | 0.45% / 1.30% / 4.90% | 8.8e-03 |
| darwin-arm64-accelerate | gha-windows-latest | 33/42 | GEMM, conv, element-wise, model fwd/bwd, reductions, transcendental | 0.40% / 1.10% / 3.90% | 8.5e-03 |
| darwin-arm64-accelerate | linux-arm64-open | 23/42 | BatchNorm, GEMM, conv, model fwd/bwd | 0.45% / 1.45% / 3.80% | 7.7e-03 |
| gha-macos-latest | gha-ubuntu-24.04 | 34/42 | BatchNorm, GEMM, conv, element-wise, model fwd/bwd, reductions, transcendental | 0.35% / 1.60% / 3.20% | 8.4e-03 |
| gha-macos-latest | gha-ubuntu-24.04-arm | 24/42 | BatchNorm, GEMM, conv, model fwd/bwd | 0.45% / 1.30% / 4.90% | 8.8e-03 |
| gha-macos-latest | gha-windows-latest | 33/42 | GEMM, conv, element-wise, model fwd/bwd, reductions, transcendental | 0.40% / 1.10% / 3.90% | 8.5e-03 |
| gha-macos-latest | linux-arm64-open | 23/42 | BatchNorm, GEMM, conv, model fwd/bwd | 0.45% / 1.45% / 3.80% | 7.7e-03 |
| gha-ubuntu-24.04 | gha-ubuntu-24.04-arm | 32/42 | GEMM, conv, element-wise, model fwd/bwd, reductions, transcendental | 0.10% / 0.95% / 3.45% | 8.5e-03 |
| gha-ubuntu-24.04 | gha-windows-latest | 22/42 | BatchNorm, model fwd/bwd, transcendental | 0.15% / 1.10% / 4.25% | 9.9e-03 |
| gha-ubuntu-24.04 | linux-arm64-open | 32/42 | GEMM, conv, element-wise, model fwd/bwd, reductions, transcendental | 0.10% / 1.00% / 3.05% | 9.1e-03 |
| gha-ubuntu-24.04-arm | gha-windows-latest | 33/42 | BatchNorm, GEMM, conv, element-wise, model fwd/bwd, reductions, transcendental | 0.15% / 1.10% / 4.90% | 1.0e-02 |
| gha-ubuntu-24.04-arm | linux-arm64-open | 14/42 | GEMM, model fwd/bwd | 0.00% / 0.95% / 2.50% | 7.7e-03 |
| gha-windows-latest | linux-arm64-open | 33/42 | BatchNorm, GEMM, conv, element-wise, model fwd/bwd, reductions, transcendental | 0.15% / 0.85% / 4.35% | 9.0e-03 |
