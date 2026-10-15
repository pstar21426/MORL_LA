# Desideratum 3.2. gamma=0.9, 120 episodes.
Train seeds [0, 1000, 2000]. Each seed uses env seeds [seed, seed+120).
alea: one N(mean mu, (mean sigma)^2) sample per action, then argmax.
epist: one ensemble member, then argmax of its mu.
Greedy eval uses the mean mu on seeds [205, 206, 207, 208, 209].

| train seed | alea last10 | epist last10 | epist-alea | alea greedy | epist greedy | greedy epist-alea |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 928.9 | 1149.6 | +220.6 | 1089.5 | 1133.1 | +43.6 |
| 1000 | 985.9 | 1158.6 | +172.7 | 1119.5 | 1083.1 | -36.5 |
| 2000 | 917.6 | 1078.9 | +161.4 | 1130.6 | 1094.3 | -36.3 |

**epist − alea, last 10 training episodes:** +184.9 ± 18.2 (SE over 3 train seeds)

**epist − alea, greedy mean on eval seeds:** -9.7 ± 26.7 (SE over 3 train seeds)

