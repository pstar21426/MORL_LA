# epsilon-greedy ensemble. gamma=0.95, 120 episodes.
Train seeds [0, 1000, 2000]. Eval seeds [205, 206, 207, 208, 209].
Action rule: epsilon-greedy on the mean mu. Same ensemble and NLL as alea/epist.

| train seed | mode | last 10 train return | greedy mean |
| --- | --- | ---: | ---: |
| 0 | alea | 932.7 | 730.6 |
| 0 | epist | 1098.7 | 1114.9 |
| 0 | epsilon | 1115.0 | 1099.5 |
| 1000 | alea | 1030.7 | 1006.3 |
| 1000 | epist | 1168.7 | 1118.0 |
| 1000 | epsilon | 1168.0 | 1108.5 |
| 2000 | alea | 937.1 | 1123.8 |
| 2000 | epist | 1075.3 | 1099.2 |
| 2000 | epsilon | 1100.3 | 1114.0 |
