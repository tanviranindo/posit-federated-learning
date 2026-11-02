# Mitigation: exact-accumulation stack

## M1 reproducibility and M2 overhead (5 local epochs, 3 rounds, seed 7)

| Platform | CPU | BLAS | Same hashes as first | std s/epoch | det s/epoch | ratio |
|---|---|---|---|---|---|---|
| darwin-arm64-accelerate | Apple | accelerate | yes | 0.45 | 2.47 | 5.4x |
| gha-macos-latest-a | Apple | accelerate | yes | 0.66 | 9.19 | 13.9x |
| gha-ubuntu-24.04-a | AMD EPYC AVX2 | mkl | yes | 0.82 | 12.91 | 15.7x |
| gha-ubuntu-24.04-arm-a | Neoverse-N2 | open | yes | 1.31 | 11.58 | 8.9x |
| gha-ubuntu-24.04-b | AMD EPYC AVX512 | mkl | yes | 0.59 | 9.68 | 16.4x |
| gha-windows-latest-a | AMD Zen5 AVX512 | mkl | yes | 0.51 | 6.28 | 12.3x |
| gha-windows-latest-b | AMD Zen5 AVX512 | mkl | yes | 0.44 | 5.23 | 11.9x |
| linux-arm64-open | Apple (Linux VM) | open | yes | 0.98 | 5.13 | 5.2x |

M1 (every client, global and logits hash identical on 8 platforms, all rounds): **pass**. Replay tolerance is therefore 0.
Local ratios use an idle-machine re-run (`det_t1_o7_e5_bench`) where present.

## M3 accuracy parity (1 local epoch, 20 rounds, M2 Pro)

| Seed | standard | deterministic | difference (pp) |
|---|---|---|---|
| 7 | 55.75 | 55.90 | +0.15 |
| 8 | 57.25 | 57.60 | +0.35 |

M3 (|difference| <= 2.00 pp): **pass**
