# Replay tolerance (traj_t1_o7_e5)

Honest mismatch ||replayed - reported|| / ||update|| per round, min-max over replayed clients (%).

| Verifier | Source | CPU (verifier <- source) | Identical | r1 | r2 | r3 | r6 | r11 |
|---|---|---|---|---|---|---|---|---|
| darwin-arm64-accelerate | gha-ubuntu-24.04 | Apple <- Intel Xeon AVX512 | no | 2.3-3.5 | 13.1-17.1 | 17.9-21.4 | 12.4-27.5 | 13.3-16.5 |
| darwin-arm64-accelerate | gha-windows-latest | Apple <- Intel Xeon AVX512 | no | 2.3-3.5 | 13.0-14.9 | 15.2-24.6 | 15.1-18.4 | 11.8-16.8 |
| gha-macos-latest | gha-ubuntu-24.04 | Apple <- Intel Xeon AVX512 | no | 2.3-3.5 | 13.1-17.1 | 17.9-21.4 | 12.4-27.5 | 13.3-16.5 |
| gha-macos-latest | gha-windows-latest | Apple <- Intel Xeon AVX512 | no | 2.3-3.5 | 13.0-14.9 | 15.2-24.6 | 15.1-18.4 | 11.8-16.8 |
| gha-ubuntu-24.04 | gha-macos-latest | AMD EPYC AVX2 <- Apple | no | 2.2-3.6 | 13.2-16.9 | 17.5-23.7 | 14.4-20.6 | 12.9-19.5 |
| gha-ubuntu-24.04 | gha-ubuntu-24.04-arm | AMD EPYC AVX2 <- Neoverse-N2 | no | 2.1-3.4 | 8.8-15.5 | 8.6-19.8 | 7.8-12.3 | 12.0-13.1 |
| gha-ubuntu-24.04-arm | gha-ubuntu-24.04 | Neoverse-N2 <- Intel Xeon AVX512 | no | 2.5-3.4 | 9.5-17.3 | 12.5-21.2 | 11.1-20.8 | 5.9-25.7 |
| gha-ubuntu-24.04-arm | gha-windows-latest | Neoverse-N2 <- Intel Xeon AVX512 | no | 2.0-3.3 | 9.9-13.7 | 14.5-21.8 | 10.0-18.8 | 8.9-14.7 |
| gha-windows-latest | gha-macos-latest | AMD Zen3 AVX2 <- Apple | no | 2.2-3.5 | 13.8-19.7 | 17.6-30.5 | 13.5-19.3 | 10.0-20.0 |
| gha-windows-latest | gha-ubuntu-24.04-arm | AMD Zen3 AVX2 <- Neoverse-N2 | no | 1.2-3.5 | 11.1-16.8 | 10.7-19.7 | 7.8-13.8 | 11.6-13.5 |
| linux-arm64-open | gha-ubuntu-24.04 | Apple (Linux VM) <- Intel Xeon AVX512 | no | 1.8-3.0 | 8.0-13.8 | 12.3-22.3 | 8.9-20.2 | 6.8-25.9 |
| linux-arm64-open | gha-windows-latest | Apple (Linux VM) <- Intel Xeon AVX512 | no | 1.5-3.0 | 10.3-13.7 | 12.2-21.0 | 11.0-19.2 | 11.1-14.1 |
| darwin-arm64-accelerate | gha-macos-latest | Apple <- Apple | yes | 0.0-0.0 | 0.0-0.0 | 0.0-0.0 | 0.0-0.0 | 0.0-0.0 |
| darwin-arm64-accelerate | gha-ubuntu-24.04-arm | Apple <- Neoverse-N2 | no | 2.5-3.8 | 12.7-18.0 | 12.6-22.5 | 12.4-21.3 | 14.0-19.7 |
| darwin-arm64-accelerate | linux-arm64-open | Apple <- Apple (Linux VM) | no | 2.5-3.6 | 11.8-15.0 | 14.5-23.6 | 13.6-26.5 |  |
| gha-macos-latest | gha-macos-latest | Apple <- Apple | yes | 0.0-0.0 | 0.0-0.0 | 0.0-0.0 | 0.0-0.0 | 0.0-0.0 |
| gha-macos-latest | gha-ubuntu-24.04-arm | Apple <- Neoverse-N2 | no | 2.5-3.8 | 12.7-18.0 | 12.6-22.5 | 12.4-21.3 | 14.0-19.7 |
| gha-ubuntu-24.04 | gha-ubuntu-24.04 | AMD EPYC AVX2 <- Intel Xeon AVX512 | no | 1.8-2.8 | 5.2-18.6 | 10.0-21.3 | 11.1-19.8 | 5.9-14.4 |
| gha-ubuntu-24.04 | gha-windows-latest | AMD EPYC AVX2 <- Intel Xeon AVX512 | no | 1.5-2.5 | 11.9-15.6 | 9.6-20.6 | 10.4-17.6 | 8.5-16.8 |
| gha-ubuntu-24.04-arm | gha-macos-latest | Neoverse-N2 <- Apple | no | 2.5-3.8 | 13.5-22.4 | 15.4-33.7 | 14.0-20.8 | 13.0-17.2 |
| gha-ubuntu-24.04-arm | gha-ubuntu-24.04-arm | Neoverse-N2 <- Neoverse-N2 | yes | 0.0-0.0 | 0.0-0.0 | 0.0-0.0 | 0.0-0.0 | 0.0-0.0 |
| gha-windows-latest | gha-ubuntu-24.04 | AMD Zen3 AVX2 <- Intel Xeon AVX512 | no | 2.3-3.4 | 10.1-16.6 | 12.9-22.4 | 8.1-14.5 | 6.9-26.1 |
| gha-windows-latest | gha-windows-latest | AMD Zen3 AVX2 <- Intel Xeon AVX512 | no | 2.0-3.3 | 11.9-15.8 | 16.0-18.7 | 14.3-16.7 | 8.0-16.1 |
| linux-arm64-open | darwin-arm64-accelerate | Apple (Linux VM) <- Apple | no | 2.5-3.6 | 13.5-14.6 | 15.9-30.1 | 13.7-19.1 |  |
| linux-arm64-open | gha-macos-latest | Apple (Linux VM) <- Apple | no | 2.5-3.7 | 13.5-21.3 | 15.9-30.1 | 13.7-19.1 | 13.4-17.5 |
| linux-arm64-open | gha-ubuntu-24.04-arm | Apple (Linux VM) <- Neoverse-N2 | no | 0.6-2.6 | 9.4-15.7 | 9.4-10.2 | 6.3-11.1 | 2.2-12.9 |

Replays on the same CPU and OS bitwise identical: True (2 pairs).
C1 (every cross-ISA mismatch >= 10% at rounds >= 3, 12 pairs): **fail** (10 values below)

| Budget | r1 | r2 | r3 | r6 | r11 | --tau-schedule |
|---|---|---|---|---|---|---|
| B_min | 1.2% | 8.0% | 8.6% | 7.7% | 5.9% | `1:0.012,2:0.08,3:0.086,6:0.077,11:0.059` |
| B_half | 0.6% | 4.0% | 4.3% | 3.8% | 2.9% | `1:0.006,2:0.04,3:0.043,6:0.038,11:0.029` |
