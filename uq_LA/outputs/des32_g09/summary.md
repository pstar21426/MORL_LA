# Desideratum 3.2. Same ensemble, 80 episodes, seeds 0-79, gamma=0.9.
alea: one N(mean mu, (mean sigma)^2) sample per action, then argmax.
epist: one ensemble member, then argmax of its mu.
Greedy eval uses the mean mu. Seeds 205-207.

| | train return last 10 | greedy mean | OLLA | greedy-OLLA | SE |
| --- | ---: | ---: | ---: | ---: | ---: |
| alea | 924.0 | 1065.4 | 1017.7 | +47.8 | 17.4 |
| epist | 1122.3 | 1121.6 | 1017.7 | +103.9 | 10.2 |
