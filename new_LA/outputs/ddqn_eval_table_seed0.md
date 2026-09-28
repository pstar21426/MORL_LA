# DDQN eval vs ILLA / OLLA (train seed=0, 200 ep)

| seed | DDQN ret | ILLA ret | OLLA ret | DDQN BLER | ILLA BLER | OLLA BLER | DDQN MCS | ILLA MCS | OLLA MCS | DDQN drops | ILLA drops | OLLA drops |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 201 | 580.5 | 1040.2 | 1154.2 | 0.345 | 0.244 | 0.105 | 20.1 | 17.6 | 14.2 | 537.7 | 406.1 | 367.0 |
| 202 | 283.0 | 702.8 | 768.8 | 0.343 | 0.244 | 0.103 | 19.9 | 17.0 | 12.7 | 668.4 | 558.1 | 574.0 |
| 203 | 616.7 | 1005.6 | 1097.5 | 0.327 | 0.226 | 0.105 | 20.2 | 17.7 | 14.0 | 528.7 | 418.8 | 412.9 |

**mean±std return** DDQN 493.4±149.5 · ILLA 916.2±151.6 · OLLA 1006.8±169.9

**mean 1st-tx BLER** DDQN 0.338 · ILLA 0.238 · OLLA 0.104
