# Supervised check: can the DDQN net learn the one-step table?

Data: 100 eps-greedy (eps 0.3) episodes of `bandit_cqi_nack_fixed_400_table.npz`, 50093 decisions, last 10% held out for argmax agreement. rho 1.199. QNet hidden 256, 40 epochs, Adam 1e-3, batch 256. Eval seeds [205, 206, 207, 208, 209].

| policy | return | diff vs OLLA | 95% CI | argmax = table (held out) |
| --- | ---: | ---: | --- | ---: |
| OLLA | 1044.4 | +0.0 | [+0.0, +0.0] | - |
| table | 1120.4 | +76.0 | [+49.7, +102.2] | - |
| net dense (full state) | 1112.0 | +67.5 | [+32.8, +102.2] | 0.576 |
| net sample (full state) | 1095.1 | +50.7 | [+17.1, +84.2] | 0.107 |
| net sample (CQI, NACK, queue) | 1100.8 | +56.4 | [+32.0, +80.7] | 0.111 |

Per-seed return: OLLA 1066.5, 942.0, 1044.5, 1153.7, 1015.5; table 1152.1, 1000.9, 1097.3, 1230.6, 1121.0; net dense (full state) 1147.0, 983.0, 1078.4, 1242.6, 1108.7; net sample (full state) 1080.8, 1010.4, 1128.5, 1203.2, 1052.7; net sample (CQI, NACK, queue) 1119.9, 1020.3, 1073.8, 1202.3, 1087.9
