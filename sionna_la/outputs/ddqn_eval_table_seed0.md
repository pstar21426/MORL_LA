# DDQN eval vs ILLA / OLLA (train seed=0, 200 ep)


| seed | DDQN ret | ILLA ret | OLLA ret | DDQN BLER | ILLA BLER | OLLA BLER | DDQN MCS | ILLA MCS | OLLA MCS | DDQN drops | ILLA drops | OLLA drops |
| ---- | -------- | -------- | -------- | --------- | --------- | --------- | -------- | -------- | -------- | ---------- | ---------- | ---------- |
| 1    | 448.0    | 966.4    | 1102.9   | 0.503     | 0.279     | 0.106     | 18.8     | 14.0     | 9.8      | 117        | 66         | 29         |
| 2    | 688.2    | 1074.8   | 1287.6   | 0.540     | 0.291     | 0.104     | 19.8     | 15.5     | 11.3     | 87         | 59         | 23         |
| 3    | 1521.4   | 1956.6   | 2066.9   | 0.426     | 0.248     | 0.102     | 22.0     | 18.6     | 15.0     | 64         | 35         | 13         |


**mean±std return** DDQN 885.9±460.0 · ILLA 1332.6±443.4 · OLLA 1485.8±417.8

**mean 1st-tx BLER** DDQN 0.490 · ILLA 0.273 · OLLA 0.104