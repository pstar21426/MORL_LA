# DDQN γ=0.7 vs ILLA / OLLA, queue_feat=tbs, seeds 500-529 (n=30)

| | ILLA | OLLA | DDQN | DDQN-OLLA | SE |
| --- | ---: | ---: | ---: | ---: | ---: |
| episode return | 833.0 | 1056.8 | 1058.2 | +1.4 | 6.5 |
| 1st-tx BLER | 0.455 | 0.112 | 0.117 | +0.005 | 0.005 |
| MCS | 22.03 | 14.00 | 12.00 | -2.00 | 0.32 |

## Burst split, pooled over seeds

| | OLLA off | DDQN off | OLLA on | DDQN on |
| --- | ---: | ---: | ---: | ---: |
| return / slot | 1.189 | 1.196 | 0.502 | 0.511 |
| 1st-tx BLER | 0.087 | 0.092 | 0.409 | 0.360 |
| MCS | 13.93 | 11.97 | 14.42 | 12.16 |
| slots / TB | 1.53 | 1.56 | 4.07 | 3.64 |

DDQN - OLLA return +1.4 (SE 6.5), positive seeds 16/30
