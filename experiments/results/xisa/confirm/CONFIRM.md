# Confirmation C2: hidden attack on the full test set

Round-20 values in pp. flip gap = targeted - random cat->dog rate; acc gap = random - targeted accuracy; drop = honest - targeted accuracy.

| Budget | Seeds | flip gap per seed | flip gap mean [95% CI] | acc gap mean [95% CI] | drop mean | cat->dog honest / random / targeted | Pass | Stealth |
|---|---|---|---|---|---|---|---|---|
| b_half (`1:0.006,2:0.04,3:0.043,6:0.038,11:0.029`) | 5 | +4.1, +5.2, +0.2, -2.5, +3.5 | +2.1 [-1.8, +6.0] | -0.24 [-0.67, +0.19] | -0.20 | 18.6 / 18.2 / 20.3 | no | yes |
| b_min (`1:0.012,2:0.08,3:0.086,6:0.077,11:0.059`) | 5 | +5.5, +5.5, +4.3, -0.8, +7.7 | +4.4 [+0.5, +8.4] | +0.13 [-0.32, +0.58] | +0.04 | 18.6 / 18.8 / 23.2 | no | yes |
