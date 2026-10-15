# gamma=0.9, 120 episodes

Training seeds are 0, 1000, and 2000 (n=3). Each training seed is averaged
over eval seeds 205-209 before the mean and standard deviation across training
seeds are taken. ± values are the sample standard deviation (ddof=1), not the
standard error. n=3 is too small for a significance test.
This file, Fig. 1, and Fig. 2 are one run of report_des32.py.

Paired return is the policy minus OLLA on the same eval seed,
averaged over seeds 205-209 within a training seed, then mean ± standard
deviation across the 3 training seeds. OLLA's own ± is across eval seeds
and is not the same kind of number.

alea: train-seed paired gaps -123.0, -129.2, -140.5; mean -130.9 ± 8.9. HARQ discard by train seed 22633.6, 21370.2, 23026.2.
epist: train-seed paired gaps +84.8, +32.3, +38.4; mean +51.8 ± 28.7. HARQ discard by train seed 4419.0, 8254.0, 6741.4.
epsilon: train-seed paired gaps +62.6, +56.4, +60.4; mean +59.8 ± 3.1. HARQ discard by train seed 6631.6, 7325.8, 7054.4.

Spearman correlation of alea per-MCS selection rate and alea sigma: 0.956.
Bin mean sigma from the alea model: low 1.23, mid 3.04, high 4.64.

Fig. 1 caption: training episode return, 5-episode trailing mean.
Band is ±1 standard deviation across 3 training seeds.
The vertical line is where epsilon reaches 0.01 (30000 decisions,
about 516 decisions per episode, episode 58).
The horizontal line is the OLLA evaluation mean 1044.4
on seeds 205-209, not a training curve.

Fig. 2 caption: selection mass on MCS 3-11, 12-20, and 21-28.
Tick labels give the unweighted mean of per-MCS sigma from the alea
ensemble, averaged on states the alea behavior visited.
Epsilon bars use epsilon=0.01.

Episode return is the sum of delivered bits/RE over the 1000-slot episode.
Fig. 1 plots that quantity during training. The table column is the same
quantity on eval seeds 205-209, then summarized over the 3 training seeds.
Overflow and HARQ discard penalties are zero, so return equals delivered
spectral efficiency; the loss columns are the bits that were not delivered.

Behavior rows use the training action rule. OLLA has no training seed;
its ± value is the standard deviation across the 5 eval seeds.

Epsilon-greedy uses the same ensemble, gamma, episodes, and training seeds.
During training, epsilon decays from 1 to 0.01 over 30000 decisions.
One episode is about 516 decisions, so this ends near episode 58,
not at episode 30. The table row evaluates the saved agent at
epsilon=0.01. Alea and epist rows keep their own stationary rules
(Gaussian sample, or one member's mean). They do not use epsilon.

Epistemic sampling argmaxes one member's mean mu and ignores sigma.
Its MCS mass on 3-11 / 12-20 / 21-28 is
0.56 / 0.33 / 0.11.
It does select MCS above 12; the mode is simply lower because high MCS
has a lower mean return under r - rho * tau, not because index 13 is masked.

Fig. 2 shows only the aleatoric behavior. Its selection mass is
0.14 / 0.29 / 0.56
on MCS 3-11 / 12-20 / 21-28, and sigma rises toward the high bin.
Epistemic behavior, not drawn, puts
0.56 / 0.33 / 0.11
on those same bins, so most of its mass is in the low bin.

Greedy return is the mean mu policy, already averaged over eval seeds 205-209.
Across 3 training seeds it is alea 1113.2 ± 21.3,
epist 1103.5 ± 26.2.
Seed 0 favors epist; seeds 1000 and 2000 favor alea.

Mean retransmissions are
alea 0.430 ± 0.010,
epist 0.143 ± 0.006,
OLLA 0.181 ± 0.025.
Final queue bits are
alea 9482 ± 405,
epist 6951 ± 544,
OLLA 7255 ± 4478.

| policy | n | episode return | first-tx BLER | HARQ discard [bits] | overflow [bits] |
| --- | --- | --- | --- | --- | --- |
| alea behavior | 3 train seeds | 913.5 ± 8.9 | 0.265 ± 0.005 | 22343.3 ± 865.3 | 95482.1 ± 2068.8 |
| epist behavior | 3 train seeds | 1096.3 ± 28.7 | 0.091 ± 0.011 | 6471.5 ± 1931.7 | 59059.1 ± 6534.8 |
| epsilon behavior | 3 train seeds | 1104.3 ± 3.1 | 0.103 ± 0.013 | 7003.9 ± 349.8 | 56803.7 ± 586.0 |
| OLLA | 5 eval seeds | 1044.4 ± 77.0 | 0.110 ± 0.005 | 9239.4 ± 3104.3 | 71536.8 ± 23772.8 |

## Per training seed, after averaging eval seeds 205-209

| train seed | policy | episode return | first-tx BLER | HARQ discard [bits] | overflow [bits] |
| --- | --- | --- | --- | --- | --- |
| 0 | alea | 921.5 | 0.262 | 22633.6 | 93259.8 |
| 0 | epist | 1129.2 | 0.103 | 4419.0 | 51548.6 |
| 0 | epsilon | 1107.0 | 0.117 | 6631.6 | 56664.4 |
| 1000 | alea | 915.2 | 0.263 | 21370.2 | 95834.2 |
| 1000 | epist | 1076.7 | 0.089 | 8254.0 | 63445.2 |
| 1000 | epsilon | 1100.9 | 0.093 | 7325.8 | 57446.8 |
| 2000 | alea | 903.9 | 0.271 | 23026.2 | 97352.2 |
| 2000 | epist | 1082.9 | 0.080 | 6741.4 | 62183.6 |
| 2000 | epsilon | 1104.8 | 0.098 | 7054.4 | 56300.0 |

## Every eval episode

| train seed | policy | eval seed | episode return | first-tx BLER | HARQ discard [bits] | overflow [bits] |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | alea | 205 | 888.7 | 0.255 | 30016.0 | 99539.0 |
| 0 | alea | 206 | 850.0 | 0.298 | 22107.0 | 113260.0 |
| 0 | alea | 207 | 908.7 | 0.279 | 20590.0 | 95452.0 |
| 0 | alea | 208 | 1044.9 | 0.268 | 16471.0 | 65842.0 |
| 0 | alea | 209 | 915.0 | 0.210 | 23984.0 | 92206.0 |
| 0 | epist | 205 | 1162.9 | 0.091 | 4920.0 | 43346.0 |
| 0 | epist | 206 | 1011.1 | 0.129 | 5606.0 | 82153.0 |
| 0 | epist | 207 | 1121.2 | 0.126 | 3010.0 | 53608.0 |
| 0 | epist | 208 | 1260.7 | 0.080 | 4694.0 | 18074.0 |
| 0 | epist | 209 | 1090.2 | 0.088 | 3865.0 | 60562.0 |
| 0 | epsilon | 205 | 1129.8 | 0.117 | 6888.0 | 50504.0 |
| 0 | epsilon | 206 | 1001.1 | 0.147 | 5678.0 | 89372.0 |
| 0 | epsilon | 207 | 1099.7 | 0.114 | 7736.0 | 54747.0 |
| 0 | epsilon | 208 | 1237.8 | 0.098 | 6696.0 | 23044.0 |
| 0 | epsilon | 209 | 1066.8 | 0.112 | 6160.0 | 65655.0 |
| 1000 | alea | 205 | 917.9 | 0.258 | 23136.0 | 98713.0 |
| 1000 | alea | 206 | 833.3 | 0.293 | 27880.0 | 113625.0 |
| 1000 | alea | 207 | 925.2 | 0.267 | 14224.0 | 95074.0 |
| 1000 | alea | 208 | 983.8 | 0.273 | 18523.0 | 78880.0 |
| 1000 | alea | 209 | 915.8 | 0.226 | 23088.0 | 92879.0 |
| 1000 | epist | 205 | 1091.4 | 0.071 | 10080.0 | 59553.0 |
| 1000 | epist | 206 | 998.1 | 0.101 | 8629.0 | 86738.0 |
| 1000 | epist | 207 | 1067.8 | 0.116 | 6650.0 | 60486.0 |
| 1000 | epist | 208 | 1125.9 | 0.084 | 9855.0 | 49509.0 |
| 1000 | epist | 209 | 1100.4 | 0.075 | 6056.0 | 60940.0 |
| 1000 | epsilon | 205 | 1108.8 | 0.088 | 6680.0 | 56515.0 |
| 1000 | epsilon | 206 | 989.2 | 0.104 | 10357.0 | 86700.0 |
| 1000 | epsilon | 207 | 1103.6 | 0.108 | 7688.0 | 52653.0 |
| 1000 | epsilon | 208 | 1229.1 | 0.076 | 6688.0 | 24048.0 |
| 1000 | epsilon | 209 | 1073.8 | 0.089 | 5216.0 | 67318.0 |
| 2000 | alea | 205 | 895.1 | 0.261 | 30576.0 | 97236.0 |
| 2000 | alea | 206 | 809.8 | 0.308 | 26088.0 | 123004.0 |
| 2000 | alea | 207 | 901.3 | 0.297 | 16932.0 | 98846.0 |
| 2000 | alea | 208 | 979.4 | 0.276 | 20159.0 | 78813.0 |
| 2000 | alea | 209 | 933.9 | 0.214 | 21376.0 | 88862.0 |
| 2000 | epist | 205 | 1120.6 | 0.074 | 6728.0 | 54317.0 |
| 2000 | epist | 206 | 967.7 | 0.093 | 9941.0 | 90766.0 |
| 2000 | epist | 207 | 1096.8 | 0.087 | 4744.0 | 58077.0 |
| 2000 | epist | 208 | 1165.1 | 0.075 | 7894.0 | 40986.0 |
| 2000 | epist | 209 | 1064.2 | 0.072 | 4400.0 | 66772.0 |
| 2000 | epsilon | 205 | 1098.6 | 0.095 | 9172.0 | 58454.0 |
| 2000 | epsilon | 206 | 1000.1 | 0.115 | 7789.0 | 83836.0 |
| 2000 | epsilon | 207 | 1094.2 | 0.109 | 8843.0 | 56200.0 |
| 2000 | epsilon | 208 | 1273.5 | 0.080 | 3127.0 | 16262.0 |
| 2000 | epsilon | 209 | 1057.8 | 0.093 | 6341.0 | 66748.0 |
| none | OLLA | 205 | 1066.5 | 0.104 | 8217.0 | 69043.0 |
| none | OLLA | 206 | 942.0 | 0.110 | 10884.0 | 101931.0 |
| none | OLLA | 207 | 1044.5 | 0.114 | 4256.0 | 72001.0 |
| none | OLLA | 208 | 1153.7 | 0.107 | 11895.0 | 35813.0 |
| none | OLLA | 209 | 1015.5 | 0.116 | 10945.0 | 78896.0 |
