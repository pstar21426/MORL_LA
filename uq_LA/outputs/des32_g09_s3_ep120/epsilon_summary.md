# epsilon-greedy ensemble. gamma=0.9, 120 episodes.
Train seeds [0, 1000, 2000]. Eval seeds [205, 206, 207, 208, 209].
Action rule: epsilon-greedy on the mean mu. Same ensemble and NLL as alea/epist.

| train seed | mode | last 10 train return | greedy mean |
| --- | --- | ---: | ---: |
| 0 | epsilon | 1129.5 | 1101.1 |
| 1000 | epsilon | 1180.2 | 1112.0 |
| 2000 | epsilon | 1117.7 | 1111.5 |
