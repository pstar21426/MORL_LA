# DDQN eval vs ILLA / OLLA (train seed=0, 200 ep)

| seed | DDQN ret | ILLA ret | OLLA ret | DDQN BLER | ILLA BLER | OLLA BLER | DDQN MCS | ILLA MCS | OLLA MCS | DDQN drops | ILLA drops | OLLA drops |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 201 | 1384.4 | 1675.6 | 1903.0 | 0.300 | 0.281 | 0.106 | 18.0 | 17.6 | 14.6 | 60 | 39 | 22 |
| 202 | 1182.9 | 1348.5 | 1248.9 | 0.298 | 0.278 | 0.103 | 17.2 | 15.9 | 10.8 | 75 | 55 | 35 |
| 203 | 1493.3 | 1528.9 | 1587.4 | 0.256 | 0.267 | 0.105 | 18.1 | 16.9 | 12.5 | 63 | 47 | 20 |

**mean±std return** DDQN 1353.5±128.6 · ILLA 1517.6±133.8 · OLLA 1579.7±267.1

**mean 1st-tx BLER** DDQN 0.285 · ILLA 0.275 · OLLA 0.104
