# DDQN eval vs ILLA / OLLA (train seed=0, 200 ep)

| seed | DDQN ret | ILLA ret | OLLA ret | DDQN BLER | ILLA BLER | OLLA BLER | DDQN MCS | ILLA MCS | OLLA MCS | DDQN drops | ILLA drops | OLLA drops |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 201 | 936.3 | 766.2 | 978.6 | 0.152 | 0.363 | 0.113 | 13.0 | 21.8 | 13.3 | 352.1 | 518.9 | 304.7 |
| 202 | 1057.4 | 874.2 | 1055.1 | 0.099 | 0.401 | 0.109 | 11.7 | 21.6 | 14.0 | 259.6 | 415.4 | 260.7 |
| 203 | 948.4 | 816.1 | 967.6 | 0.122 | 0.468 | 0.113 | 13.0 | 23.0 | 15.0 | 352.1 | 492.6 | 336.3 |

**mean±std return** DDQN 980.7±54.5 · ILLA 818.8±44.1 · OLLA 1000.4±38.9

**mean 1st-tx BLER** DDQN 0.124 · ILLA 0.411 · OLLA 0.112
