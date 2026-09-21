# MORL_LA

세 작업 트리를 나눠 두었습니다. **시뮬을 바꿀 때는 `sionna_la`를 건드리지 말고 `bandit_la`에서만 수정합니다.**

| 폴더 | 내용 |
|------|------|
| `toy_mdp/` | Appendix toy LA + MOPO (기존 루트 코드) |
| `sionna_la/` | Sionna 5G DL LA 시뮬. 그대로 보존 |
| `bandit_la/` | `sionna_la` 코어 복사본. OLLA vs 밴딧(LTS/UCB) 실험용 |

논문 PDF는 루트에 그대로 있습니다.
