# gamma=0.95, 120 episodes

Training seeds are 0, 1000, and 2000 (n=3). Each training seed is averaged
over eval seeds 205-209 before the mean and standard deviation across training
seeds are taken. ± values are the sample standard deviation (ddof=1), not the
standard error. n=3 is too small for a significance test.
This file, Fig. 1, and Fig. 2 are one run of report_des32.py.

Paired return is the policy minus OLLA on the same eval seed,
averaged over seeds 205-209 within a training seed, then mean ± standard
deviation across the 3 training seeds. OLLA's own ± is across eval seeds
and is not the same kind of number.

alea: train-seed paired gaps -85.1, -87.8, -102.8; mean -91.9 ± 9.5. HARQ discard by train seed 15798.6, 17905.2, 21766.6.
epist: train-seed paired gaps +57.4, +69.1, +41.6; mean +56.0 ± 13.8. HARQ discard by train seed 5745.2, 5403.4, 5744.6.
epsilon: train-seed paired gaps +46.1, +52.9, +78.9; mean +59.3 ± 17.3. HARQ discard by train seed 8032.4, 6442.2, 5548.8.

Spearman correlation of alea per-MCS selection rate and alea sigma: 0.265.
Bin mean sigma from the alea model: low 3.87, mid 5.01, high 6.15.

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
0.39 / 0.54 / 0.07.
It does select MCS above 12; the mode is simply lower because high MCS
has a lower mean return under r - rho * tau, not because index 13 is masked.

Fig. 2 shows only the aleatoric behavior. Its selection mass is
0.35 / 0.34 / 0.31
on MCS 3-11 / 12-20 / 21-28, and sigma rises toward the high bin.
Epistemic behavior, not drawn, puts
0.39 / 0.54 / 0.07
on those same bins, so most of its mass is in the low bin.

Greedy return is the mean mu policy, already averaged over eval seeds 205-209.
Across 3 training seeds it is alea 953.6 ± 201.8,
epist 1110.7 ± 10.1.
Seed 0 favors epist; seeds 1000 and 2000 favor alea.

Mean retransmissions are
alea 0.302 ± 0.067,
epist 0.132 ± 0.009,
OLLA 0.181 ± 0.025.
Final queue bits are
alea 9169 ± 999,
epist 6480 ± 819,
OLLA 7255 ± 4478.

| policy | n | episode return | first-tx BLER | HARQ discard [bits] | overflow [bits] |
| --- | --- | --- | --- | --- | --- |
| alea behavior | 3 train seeds | 952.6 ± 9.5 | 0.183 ± 0.038 | 18490.1 ± 3026.7 | 87933.7 ± 1141.0 |
| epist behavior | 3 train seeds | 1100.5 ± 13.8 | 0.083 ± 0.009 | 5631.1 ± 197.2 | 59111.3 ± 3859.3 |
| epsilon behavior | 3 train seeds | 1103.8 ± 17.3 | 0.099 ± 0.013 | 6674.5 ± 1258.0 | 56785.1 ± 3115.7 |
| OLLA | 5 eval seeds | 1044.4 ± 77.0 | 0.110 ± 0.005 | 9239.4 ± 3104.3 | 71536.8 ± 23772.8 |

## Per training seed, after averaging eval seeds 205-209

| train seed | policy | episode return | first-tx BLER | HARQ discard [bits] | overflow [bits] |
| --- | --- | --- | --- | --- | --- |
| 0 | alea | 959.3 | 0.149 | 15798.6 | 87932.4 |
| 0 | epist | 1101.8 | 0.084 | 5745.2 | 59530.8 |
| 0 | epsilon | 1090.6 | 0.114 | 8032.4 | 59130.4 |
| 1000 | alea | 956.7 | 0.175 | 17905.2 | 86793.4 |
| 1000 | epist | 1113.5 | 0.092 | 5403.4 | 55059.4 |
| 1000 | epsilon | 1097.4 | 0.091 | 6442.2 | 57975.2 |
| 2000 | alea | 941.7 | 0.224 | 21766.6 | 89075.4 |
| 2000 | epist | 1086.1 | 0.074 | 5744.6 | 62743.8 |
| 2000 | epsilon | 1123.4 | 0.092 | 5548.8 | 53249.8 |

## Every eval episode

| train seed | policy | eval seed | episode return | first-tx BLER | HARQ discard [bits] | overflow [bits] |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | alea | 205 | 1010.2 | 0.142 | 13854.0 | 80303.0 |
| 0 | alea | 206 | 841.7 | 0.167 | 20208.0 | 114671.0 |
| 0 | alea | 207 | 983.1 | 0.164 | 10792.0 | 79834.0 |
| 0 | alea | 208 | 999.0 | 0.145 | 17094.0 | 78924.0 |
| 0 | alea | 209 | 962.8 | 0.129 | 17045.0 | 85930.0 |
| 0 | epist | 205 | 1109.8 | 0.090 | 6141.0 | 58122.0 |
| 0 | epist | 206 | 992.6 | 0.099 | 7450.0 | 88747.0 |
| 0 | epist | 207 | 1122.2 | 0.093 | 3513.0 | 52144.0 |
| 0 | epist | 208 | 1199.7 | 0.065 | 7191.0 | 33692.0 |
| 0 | epist | 209 | 1084.8 | 0.074 | 4431.0 | 64949.0 |
| 0 | epsilon | 205 | 1082.6 | 0.109 | 9608.0 | 62199.0 |
| 0 | epsilon | 206 | 1023.3 | 0.126 | 7296.0 | 78519.0 |
| 0 | epsilon | 207 | 1093.2 | 0.123 | 7856.0 | 58703.0 |
| 0 | epsilon | 208 | 1212.2 | 0.107 | 9352.0 | 24349.0 |
| 0 | epsilon | 209 | 1041.6 | 0.105 | 6050.0 | 71882.0 |
| 1000 | alea | 205 | 951.7 | 0.173 | 23585.0 | 88116.0 |
| 1000 | alea | 206 | 865.4 | 0.209 | 14920.0 | 116999.0 |
| 1000 | alea | 207 | 979.3 | 0.188 | 15458.0 | 77139.0 |
| 1000 | alea | 208 | 1010.2 | 0.159 | 18382.0 | 69707.0 |
| 1000 | alea | 209 | 976.8 | 0.145 | 17181.0 | 82006.0 |
| 1000 | epist | 205 | 1113.8 | 0.089 | 5809.0 | 57178.0 |
| 1000 | epist | 206 | 977.3 | 0.110 | 7260.0 | 88241.0 |
| 1000 | epist | 207 | 1128.2 | 0.104 | 4376.0 | 50432.0 |
| 1000 | epist | 208 | 1254.5 | 0.079 | 4290.0 | 19570.0 |
| 1000 | epist | 209 | 1093.7 | 0.078 | 5282.0 | 59876.0 |
| 1000 | epsilon | 205 | 1118.3 | 0.084 | 7659.0 | 52879.0 |
| 1000 | epsilon | 206 | 1037.7 | 0.104 | 3448.0 | 79621.0 |
| 1000 | epsilon | 207 | 1077.4 | 0.108 | 5784.0 | 61279.0 |
| 1000 | epsilon | 208 | 1172.2 | 0.082 | 9152.0 | 35660.0 |
| 1000 | epsilon | 209 | 1081.2 | 0.080 | 6168.0 | 60437.0 |
| 2000 | alea | 205 | 930.8 | 0.224 | 28270.0 | 89698.0 |
| 2000 | alea | 206 | 851.0 | 0.239 | 23040.0 | 116675.0 |
| 2000 | alea | 207 | 917.9 | 0.225 | 19969.0 | 91994.0 |
| 2000 | alea | 208 | 1049.0 | 0.228 | 19982.0 | 59473.0 |
| 2000 | alea | 209 | 959.6 | 0.204 | 17572.0 | 87537.0 |
| 2000 | epist | 205 | 1055.4 | 0.072 | 7016.0 | 73500.0 |
| 2000 | epist | 206 | 986.7 | 0.096 | 7921.0 | 88479.0 |
| 2000 | epist | 207 | 1075.8 | 0.077 | 5512.0 | 58544.0 |
| 2000 | epist | 208 | 1248.3 | 0.055 | 3473.0 | 23457.0 |
| 2000 | epist | 209 | 1064.2 | 0.069 | 4801.0 | 69739.0 |
| 2000 | epsilon | 205 | 1095.3 | 0.097 | 6456.0 | 60769.0 |
| 2000 | epsilon | 206 | 1001.1 | 0.107 | 8072.0 | 80088.0 |
| 2000 | epsilon | 207 | 1161.3 | 0.089 | 3648.0 | 44268.0 |
| 2000 | epsilon | 208 | 1264.1 | 0.069 | 5808.0 | 16203.0 |
| 2000 | epsilon | 209 | 1095.0 | 0.097 | 3760.0 | 64921.0 |
| none | OLLA | 205 | 1066.5 | 0.104 | 8217.0 | 69043.0 |
| none | OLLA | 206 | 942.0 | 0.110 | 10884.0 | 101931.0 |
| none | OLLA | 207 | 1044.5 | 0.114 | 4256.0 | 72001.0 |
| none | OLLA | 208 | 1153.7 | 0.107 | 11895.0 | 35813.0 |
| none | OLLA | 209 | 1015.5 | 0.116 | 10945.0 | 78896.0 |
