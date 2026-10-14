# Desideratum 3.2. Same ensemble, 80 episodes, seeds 0-79.
alea: one N(mean mu, (mean sigma)^2) sample per action, then argmax.
epist: one ensemble member, then argmax of its mu.
Greedy eval uses the mean mu. Seeds 205-207.

| | train return last 10 | greedy mean | OLLA | greedy-OLLA | SE |
| --- | ---: | ---: | ---: | ---: | ---: |
| alea | 862.4 | 1081.0 | 1017.7 | +63.3 | 10.7 |
| epist | 1129.9 | 1098.7 | 1017.7 | +81.1 | 9.2 |
