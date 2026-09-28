# sim diagnostic

Seeds 201-203, greedy DDQN from outputs/ddqn_seed0.pt. Replay buffer was not saved.

## ledger
episodes checked: 11, failures: 0
- ILLA seed 201: balance 0, offered 599659, delivered 463305, discard 14712, overflow 107114, final_q 14528
- ILLA seed 202: balance 0, offered 601306, delivered 419000, discard 20368, overflow 147063, final_q 14875
- ILLA seed 203: balance 0, offered 599329, delivered 464816, discard 18740, overflow 106904, final_q 8869
- OLLA seed 201: balance 0, offered 599659, delivered 474796, discard 9210, overflow 100893, final_q 14760
- OLLA seed 202: balance 0, offered 601306, delivered 418708, discard 7940, overflow 164259, final_q 10399
- OLLA seed 203: balance 0, offered 599329, delivered 463469, discard 5186, overflow 118669, final_q 12005
- DDQN seed 201: balance 0, offered 599659, delivered 423355, discard 43948, overflow 117356, final_q 15000
- DDQN seed 202: balance 0, offered 601306, delivered 387405, discard 50986, overflow 149539, final_q 13376
- DDQN seed 203: balance 0, offered 599329, delivered 431754, discard 44073, overflow 114532, final_q 8970
- uniform seed 0: balance 0, offered 606416, delivered 288885, discard 72080, overflow 230451, final_q 15000
- uniform seed 1: balance 0, offered 600881, delivered 288306, discard 69264, overflow 228585, final_q 14726

## decision stats (201-203 pooled)
ILLA     n=1780  q=9098 (p10/50/90 654/10954/15000)  full=0.20  q<TBS=0.17  MCS=17.4 (4/18/28)  BLER=0.238  slots/TB=1.69  corr(MCS,q)=-0.65  corr(MCS,CQI)=+0.99
    state mismatches=0  MCS range 3-28  mean TBS 949  delivered bits/RE 4490.4
OLLA     n=2317  q=9619 (p10/50/90 709/11652/15000)  full=0.28  q<TBS=0.14  MCS=13.6 (3/13/26)  BLER=0.104  slots/TB=1.29  corr(MCS,q)=-0.66  corr(MCS,CQI)=+0.93
    state mismatches=0  MCS range 3-28  mean TBS 717  delivered bits/RE 4523.2
DDQN     n=1447  q=8045 (p10/50/90 574/9062/14965)  full=0.10  q<TBS=0.25  MCS=20.0 (10/22/27)  BLER=0.338  slots/TB=2.07  corr(MCS,q)=-0.44  corr(MCS,CQI)=+0.58
    state mismatches=0  MCS range 3-28  mean TBS 1109  delivered bits/RE 4141.7
uniform  n= 853  q=11821 (p10/50/90 5585/13825/14879)  full=0.08  q<TBS=0.00  MCS=16.0 (5/16/26)  BLER=0.346  slots/TB=2.34  corr(MCS,q)=+0.00  corr(MCS,CQI)=+0.02
    state mismatches=0  MCS range 3-28  mean TBS 845  delivered bits/RE 1924.0

## MCS by queue bin (mean MCS, 1st-tx BLER, count)
- ILLA: 0-3000: MCS 25.3 BLER 0.22 n=448 | 3000-6000: MCS 19.0 BLER 0.25 n=177 | 6000-9000: MCS 18.5 BLER 0.28 n=157 | 9000-12000: MCS 17.9 BLER 0.26 n=177 | 12000-15000: MCS 12.6 BLER 0.23 n=821
- OLLA: 0-3000: MCS 22.2 BLER 0.13 n=540 | 3000-6000: MCS 15.6 BLER 0.14 n=154 | 6000-9000: MCS 14.3 BLER 0.13 n=213 | 9000-12000: MCS 14.2 BLER 0.11 n=289 | 12000-15000: MCS 8.9 BLER 0.08 n=1121
- DDQN: 0-3000: MCS 22.6 BLER 0.16 n=463 | 3000-6000: MCS 22.8 BLER 0.46 n=140 | 6000-9000: MCS 22.3 BLER 0.37 n=118 | 9000-12000: MCS 21.1 BLER 0.30 n=171 | 12000-15000: MCS 16.4 BLER 0.45 n=555

## training log
episodes 200, return mean 471.8 (first 20 41.4, last 40 550.8), BLER last 40 0.299, epsilon end 0.010, NaN 0

Figures: queue_seed201.png, queue_hist.png, mcs_choice.png, state_dist.png, train_log.png
