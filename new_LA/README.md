# new_LA

`sionna_la` 코어 복사본입니다. 원본 시뮬은 수정하지 않습니다.

- `la_env.py`, `channel.py`, `harq.py`, `cqi.py`, `policies.py`
- `run_la_sim.py` — ILLA / OLLA Rollout
- `ddqn.py`, `train_ddqn.py` — 큐 길이가 포함된 상태로 DDQN
- `plot_compare.py` — 슬롯 축 ILLA / OLLA / DDQN (return, BLER, drops)
- `configs/downlink_la.yaml`

```bash
conda activate sionna_la
cd new_LA
python train_ddqn.py
python plot_compare.py
```

`plot_compare.py`는 `outputs/ddqn_seed0.pt`를 읽고, 학습에 안 쓴 시드에서 세 정책을 다시 굴려 `outputs/compare_seed*.png`를 만듭니다. drops는 HARQ 포기와 버퍼 오버플로를 합친 값입니다.
