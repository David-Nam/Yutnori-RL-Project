# Yutnori RL Project

전통 게임 **윷놀이**를 강화학습 환경으로 구현하고, 고정된 **Rule-based Agent** 상대
1:1 대전에서 승률 **60% 이상**을 목표로 강화학습 agent를 학습·비교한 프로젝트입니다.

이 저장소는 팀원별로 진행하던 **두 개의 프로젝트를 하나로 합친 것**입니다.
각 폴더는 독립적으로 실행 가능하며, 자세한 설계·실험·실행 방법은 폴더별 README와
보고서를 참고하세요.

| 폴더 | 담당 | 대표 모델 | 접근 방식 |
| --- | --- | --- | --- |
| [project-RF-/](project-RF-/README.md) | 이기쁨 | Hybrid PPO | reward·imitation·tactical **prior**에 전략 지식 주입 |
| [RL-project/](RL-project/README.md) | 남준우 | MaskablePPO | **observation/state**에 전략 지식 주입 |

> **핵심 비교 질문**: 전략 지식을 reward/policy/prior에 넣는 방식과,
> observation/state에 넣는 방식은 성능에 어떤 차이를 만드는가?

---

## 1. 프로젝트 주제 및 목표

- **문제**: 2인 윷놀이(4말, 잡기/업기/지름길/윷·모 추가 턴)를 MDP로 정의하고
  강화학습 agent가 전략을 학습할 수 있는지 분석.
- **목표**: 고정 Rule-based Agent(`common_rule_based`) 상대 1:1 대전에서 승률 60% 이상.
- 초기에는 팀원이 서로 다른 알고리즘(value-based DQN vs policy-gradient PPO)을 학습·비교하는 것에서 출발했습니다.

## 2. State · Action · Reward 설계

| 요소 | 공통 설계 | 비고 |
| --- | --- | --- |
| **State** | 내/상대 말 위치, 사용 가능한 윷 결과, 게임 진행 상태 + 프로젝트별 tactical feature | RL-project는 `base` vs `tactical` observation 비교 |
| **Action** | 이동할 말 × 사용할 윷 결과 (`4 pieces × 5 results = 20 actions`) + **action masking** | legal action만 선택 |
| **Reward** | 승리 `+1` / 패배 `-1` (terminal) + 프로젝트별 reward shaping | project-RF는 capture-aware dense reward 실험 |

자세한 규칙·평가 프로토콜:
[RL-project/docs/RULES.md](RL-project/docs/RULES.md),
[project-RF-/docs/evaluation_protocol.md](project-RF-/docs/evaluation_protocol.md)

## 3. 알고리즘

- **DQN / Dueling DQN** — project-RF 초기 pure RL baseline (sparse reward·긴 episode에서 학습 불안정)
- **PPO / MaskablePPO** — 두 프로젝트의 주력 policy-gradient 모델
- **Hybrid PPO** — project-RF 최종 모델 (PPO + Imitation Learning + Tactical Prior)
- 보조 기법: Reward Shaping, State Engineering, Tactical Prior, Imitation Learning

## 4. 실험 결과 (요약)

> 모든 평가는 공통 **paired evaluation**(2,500 base seed × 선/후공 교환 = 5,000판)으로 수행했으며,
> 두 프로젝트 모두 illegal action / evaluation error는 **0건**입니다.

**공통 Rule-based Agent 상대 승률**

| Agent | Win Rate | 비고 |
| --- | ---: | --- |
| RL-project (MaskablePPO) | **59.76%** | 40M 재학습, 3-seed 평균 |
| project-RF (Hybrid PPO) | 59.46% | `ppo_capture_imitation` |

**RL-project 최종 모델 (`full_backdo_v1`, 50M 학습, seed 2)**

```text
공식 paired 5,000판 평가:   61.08% (3,054승 1,946패)
독립 재평가 5,000판:        60.62% (3,031승 1,969패)
3-seed 공식 합산:           60.72% → 세 seed 모두 60% threshold 통과
```

**Head-to-Head (project-RF vs RL-project 직접 대전)**

| Matchup | Win Rate |
| --- | ---: |
| project-RF Hybrid | **53.98%** |
| RL-project | 46.02% |

**Ablation** — Tactical Prior 제거 시 project-RF 승률: `17.87%`

### Key Findings

- 두 접근 모두 Rule-based baseline 대비 약 **60% 승률**을 달성.
- PPO 계열이 DQN 계열보다 안정적으로 학습됨 (DQN 최고 eval 승률 26.4%).
- **State Engineering**(tactical observation)과 **Reward Design**이 성능에 큰 영향.
  RL-project에서는 reward shaping보다 tactical observation의 기여가 더 컸음.
- **Tactical Prior**는 project-RF의 직접 대전 성능 향상에 결정적으로 기여.
- 평가 기준(공통 평가 vs head-to-head)에 따라 우수 agent가 달라지므로,
  단일 승률만으로 일반적 우위를 판단하기 어려움.

## 5. 빠른 실행

각 프로젝트는 폴더 안에서 독립 실행합니다.

```bash
# RL-project (MaskablePPO)
cd RL-project
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
.venv/bin/python -m pytest -q          # 회귀 테스트

# project-RF- (Hybrid PPO)
cd project-RF-
pip install -r requirements.txt
python experiments/validate_common_env.py
```

학습/평가/재현 명령과 모델 다운로드 링크는 각 폴더 README를 참고하세요.

## 6. 학습된 모델 (다운로드)

| 모델 | 위치 |
| --- | --- |
| RL-project 최종 MaskablePPO (50M, seed 2) | [model.zip](RL-project/bests/ppo_common_rule_50m_backdo_subproc/common_rule_based_seed_2_50m_nenv32_tactical/model.zip?raw=1) |
| project-RF PPO checkpoints (zip) | [local_artifacts.zip](project-RF-/local_artifacts.zip) |

## 7. 문서 및 보고서

- **RL-project**: [통합 최종 보고서](RL-project/docs/PROJECT_REPORT.md) · [게임·평가 규칙](RL-project/docs/RULES.md)
- **project-RF-**: [최종 보고서 요약](project-RF-/docs/final_report_summary.md) · [설계 비교](project-RF-/docs/design_comparison.md) · [평가 프로토콜](project-RF-/docs/evaluation_protocol.md) · [DQN baseline 결과](project-RF-/docs/pure_dueling_dqn_baseline_result.md) · [프로젝트 결과 보고서](project-RF-/result.md)

---

## Authors

- 남준우 — RL-project (MaskablePPO + Tactical Observation)
- 이기쁨 — project-RF (Hybrid PPO + Imitation Learning + Tactical Prior)
