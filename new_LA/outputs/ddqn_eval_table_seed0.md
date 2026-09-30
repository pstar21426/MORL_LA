# DDQN eval vs ILLA / OLLA (train seed=0, 200 ep)

| seed | DDQN ret | ILLA ret | OLLA ret | DDQN BLER | ILLA BLER | OLLA BLER | DDQN MCS | ILLA MCS | OLLA MCS | DDQN drops | ILLA drops | OLLA drops |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 201 | 961.9 | 937.6 | 1171.0 | 0.384 | 0.361 | 0.111 | 20.7 | 21.8 | 15.1 | 322.7 | 347.4 | 117.3 |
| 202 | 872.3 | 918.3 | 1091.8 | 0.329 | 0.377 | 0.109 | 20.4 | 21.6 | 14.6 | 420.4 | 382.5 | 224.0 |
| 203 | 910.8 | 974.1 | 1137.8 | 0.348 | 0.397 | 0.111 | 19.7 | 21.9 | 16.1 | 378.3 | 334.6 | 170.3 |

**mean±std return** DDQN 915.0±36.7 · ILLA 943.3±23.1 · OLLA 1133.5±32.5

**mean 1st-tx BLER** DDQN 0.354 · ILLA 0.379 · OLLA 0.110
