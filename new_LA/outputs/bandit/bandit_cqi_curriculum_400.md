# One-step rate table, variant cqi, curriculum

curriculum: off mean 20 for 100 eps, ramp over 150 eps, count decay 0.99 until the ramp ends. Train seeds 1000-1399, eps 0.5->0.05, greedy probe every 10 eps. Final rho 1.166. Eval seeds [205, 206, 207, 208, 209], paired with OLLA.

| policy | return | diff vs OLLA | 95% CI | BLER | MCS | MCS off/on burst | BLER off/on burst |
| --- | ---: | ---: | --- | ---: | ---: | --- | --- |
| OLLA | 1044.4 | +0.0 | [+0.0, +0.0] | 0.110 | 15.1 | 15.2 / 14.1 | 0.079 / 0.426 |
| ILLA | 856.3 | -188.1 | [-232.2, -144.0] | 0.382 | 23.2 | 23.4 / 20.5 | 0.336 / 0.838 |
| cqi | 1119.0 | +74.6 | [+41.6, +107.5] | 0.061 | 10.1 | 9.9 / 11.4 | 0.034 / 0.320 |

Per-seed return: OLLA 1066.5, 942.0, 1044.5, 1153.7, 1015.5; ILLA 862.6, 754.8, 870.5, 918.0, 875.7; cqi 1144.4, 1023.0, 1079.4, 1223.9, 1124.3
