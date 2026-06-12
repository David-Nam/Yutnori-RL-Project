# 윷놀이 강화학습 프로젝트 최종 보고서

## 1. 프로젝트 개요

이 프로젝트는 윷놀이 규칙을 강화학습 환경으로 구현하고, 학습된 agent가
고정된 Rule-based Agent를 상대로 안정적으로 높은 승률을 달성하는지
검증한다.

최종 시스템은 다음 요소로 구성된다.

- 전체 윷놀이 규칙을 처리하는 게임 엔진
- Gymnasium 호환 단일 학습자 환경
- legal action만 선택하게 하는 action masking
- 기본 및 전술적 observation
- baseline 및 공통 Rule-based Agent
- `sb3-contrib`의 `MaskablePPO` 학습 파이프라인
- seed와 선공·후공을 통제한 공통 평가 도구
- 서로 다른 프로젝트의 checkpoint를 비교하는 adapter

현재 규칙 버전은 `full_backdo_v1`이다. 뒷도를 포함한 6개 윷 결과와
24개 action을 사용한다.

## 2. 목표와 평가 기준

주요 목표는 학습된 PPO agent가 공통 Rule-based Agent를 상대로 5,000판
평가에서 60% 이상의 관측 승률을 기록하는 것이다.

최종 평가 기준은 다음과 같다.

| 항목 | 기준 |
| --- | --- |
| 상대 | `common_rule_based` |
| 게임 수 | 5,000판 |
| base seed | 2,500개 |
| 선공·후공 | 각 seed에서 한 번씩 교환 |
| 정책 | deterministic |
| 합법 행동 | action mask 적용 |
| 오류 보고 | illegal action과 evaluation error 기록 |
| 통계 | 전체·선공·후공 승률, Wilson 95% 신뢰구간 |

60%는 관측된 point estimate의 통과 기준이다. 신뢰구간이 60%를 포함하는
경우에는 모집단 승률이 60%보다 확실히 높다고 단정하지 않는다.

## 3. 프로젝트 구조

```text
yutnori/core
  board.py                 보드 좌표, route, 전진·후진 이동
  game.py                  턴, pool, 잡기, 업기, 승패
  yut.py                   윷 결과, 이동량, 확률

yutnori/env
  yutnori_env.py           Gymnasium wrapper, observation, reward, mask

yutnori/agents
  baseline.py              baseline 및 공통 Rule-based Agent
  tactical_features.py     action별 전술 feature
  project_rf_checkpoint.py project-RF checkpoint adapter

yutnori/training
  env_factory.py           단일·벡터 환경 생성
  ppo_evaluation.py        PPO 평가
  common_evaluation.py     paired-seed 공통 평가
  reward_shaping.py        선택적 shaped reward

yutnori/eval
  tournament.py            baseline tournament
  legacy_head_to_head.py   legacy PPO 직접 대전

scripts
  train_ppo.py             MaskablePPO 학습
  evaluate_common_rule.py  공통 Rule-based 평가
  evaluate_project_rf_common.py
  evaluate_legacy_head_to_head.py

tests                       규칙·환경·학습·평가 회귀 테스트
```

세부 게임 규칙과 좌표계는 [RULES.md](RULES.md)에 정리되어 있다.

## 4. 게임 엔진

### 4.1 상태 모델

게임 상태는 `GameState`가 관리한다.

- 플레이어: 2명
- 플레이어당 말: 4개
- 말 상태: `WAITING`, `ON_BOARD`, `FINISHED`
- 현재 플레이어와 윷 결과 pool
- 각 말의 logical route와 physical cell
- 잡기, 업기, 완주, 보너스 던지기
- 승자, turn 수, decision 수
- 뒷도 관련 통계

같은 physical cell이라도 다음 이동 경로가 다를 수 있으므로, 말의 위치에는
route와 index가 함께 저장된다. `CENTER`와 `HOME`에서는 진입 경로도 보존해
뒷도의 목적지를 결정한다.

### 4.2 윷 결과

| 결과 | 이동량 | 확률 |
| --- | ---: | ---: |
| 도 | +1 | 11.52% |
| 개 | +2 | 34.56% |
| 걸 | +3 | 34.56% |
| 윷 | +4 | 12.96% |
| 모 | +5 | 2.56% |
| 뒷도 | -1 | 3.84% |

윷과 모가 나오면 보너스 던지기를 수행한다. 실제로 상대 말을 잡은 경우에도
추가 던지기를 얻는다.

### 4.3 행동 공간

action은 말과 윷 결과의 조합이다.

```python
action = piece_id * 6 + yut_result_id
```

- `piece_id`: 0~3
- `yut_result_id`: 도, 개, 걸, 윷, 모, 뒷도
- 전체 action 수: 24

동일한 칸에 업힌 말은 가장 작은 `piece_id`를 대표로 사용한다. pool에 없는
윷 결과, 완주한 말, stack의 비대표 말, 대기 말의 뒷도는 action mask에서
제외된다.

## 5. 강화학습 환경

### 5.1 Gymnasium 인터페이스

`YutnoriEnv`는 한 명의 learner 관점에서 게임을 제공한다.

- `reset()`은 learner가 행동할 수 있는 첫 상태까지 진행한다.
- 상대 턴은 환경 내부에서 자동 실행한다.
- `step()`은 learner의 행동과 이어지는 상대 턴을 처리한다.
- `action_masks()`는 현재 합법 action을 반환한다.
- seed를 지정하면 윷 결과와 선공 선택을 재현할 수 있다.

### 5.2 Observation

두 observation mode를 구현했다.

#### Base observation

`base` observation은 62차원이다.

- 양측 말 4개의 physical position
- 양측 말 4개의 status
- 양측 말 4개의 logical track
- 양측 4x4 stack matrix
- pool에 남은 6개 윷 결과 count

#### Tactical observation

`tactical` observation은 base 62차원에 24개 action 각각의 10개 feature를
추가한 302차원 벡터다.

| Feature | 의미 |
| --- | --- |
| legal | 합법 action 여부 |
| capture | 잡기 여부 |
| captured count | 잡는 상대 말 수 |
| finish | 완주 여부 |
| finished count | 완주하는 내 말 수 |
| moved count | 함께 이동하는 내 말 수 |
| waiting move | 대기 말 출발 여부 |
| stack size | 이동 stack 크기 |
| distance after | 이동 후 완주까지 거리 |
| RF score | 공통 Rule-based 점수 |

이 feature는 행동 결과를 직접 선택하지 않으며, policy가 현재 legal action의
전술적 의미를 학습할 수 있도록 observation에 제공된다.

### 5.3 Reward

두 reward mode를 비교했다.

- `terminal`: 승리 `+1`, 패배 `-1`, 중간 step `0`
- `rf_shaped`: terminal reward에 잡기·완주·지름길 보상과 상대 이벤트
  penalty를 추가

후보 실험에서는 shaped reward보다 `tactical + terminal` 구성이 최종
승률에 더 효과적이었다. 최종 모델도 terminal reward를 사용한다.

## 6. 비교 Agent

### 6.1 Baseline

- `RandomAgent`: 합법 action 중 균등 선택
- `CaptureFirstAgent`: 잡기 가능한 action 우선
- `GreedyFinishAgent`: 완주와 전진을 우선
- `ProjectRFRuleBasedAgent`: project-RF의 휴리스틱을 로컬 엔진에 이식
- `CommonRuleBasedAgent`: 고정된 점수식과 작은 action ID 동점 처리 사용

공통 Rule-based Agent의 점수 요소는 완주, 잡기, 새 말 출발, stack 이동,
완주까지 남은 거리다.

### 6.2 모델 유형 구분

비교 결과를 해석할 때 다음 유형을 구분한다.

- `Pure PPO`: 신경망 출력과 action mask만으로 행동
- `RL + Rule Hybrid`: 신경망 출력에 inference-time tactical prior 추가
- `Rule-based`: 고정 점수식으로 행동

project-RF의 `ppo_capture_imitation`과 `ppo_tactical`은 tactical prior를
사용하므로 Pure PPO가 아니라 RL + Rule Hybrid로 분류한다.

## 7. 학습 방법

최종 학습에는 `sb3-contrib`의 `MaskablePPO`를 사용했다.

| 항목 | 값 |
| --- | --- |
| Policy | `MlpPolicy` |
| observation | `tactical`, 302차원 |
| action | 24개 |
| reward | `terminal` |
| learning rate | `3e-4` |
| gamma | `0.99` |
| GAE lambda | `0.95` |
| entropy coefficient | `0` |
| vector env | `SubprocVecEnv` |
| env 수 | 32 |
| rollout step | 2,048 |
| batch size | 2,048 |
| 학습 seed | 0, 1, 2 |
| timestep | seed당 50M |

학습 상대와 최종 평가 상대를 모두 `common_rule_based`로 맞췄다. 선공은
환경 reset마다 결정되며, learner가 후공이면 상대의 첫 턴을 환경 내부에서
처리한 뒤 learner decision state를 반환한다.

## 8. 실험 과정

실험은 다음 순서로 진행했다.

1. 규칙 엔진과 baseline으로 게임 종료 및 합법 행동 검증
2. base·tactical observation과 terminal·shaped reward 비교
3. `project_rf_rule` 상대 3M 후보 실험
4. 선택한 PPO 구성을 10M과 30M으로 확장
5. 선공·후공을 통제한 공통 평가 프로토콜 도입
6. 공통 Rule-based Agent를 상대로 40M 재학습
7. 전체 뒷도 규칙을 적용해 50M fresh training
8. checkpoint 비교와 독립 holdout으로 대표 모델 선택

### 8.1 3M 후보 비교

| Observation | Reward | 평균 승률 |
| --- | --- | ---: |
| base | terminal | 37.4% |
| base | rf_shaped | 32.4% |
| tactical | terminal | **53.3%** |
| tactical | rf_shaped | 52.3% |

전술 정보를 observation으로 제공한 효과가 가장 컸고, reward shaping은
전체 승률을 추가로 높이지 못했다.

### 8.2 Legacy 환경 장기 학습

뒷도가 없던 20-action 환경의 결과는 다음과 같다.

| 학습 설정 | 공통 평가 평균 |
| --- | ---: |
| 30M, 기존 상대 학습 | 56.49% |
| 40M, 공통 상대 재학습 | 59.76% |

40M 공통 상대 재학습에서는 seed 1이 60.46%, seed 2가 60.40%를 기록했다.
다만 이 결과는 현재 `full_backdo_v1`과 action·observation·규칙이 다르므로
최종 50M 결과와 직접적인 단일 변수 비교로 사용하지 않는다.

## 9. Full-Backdo 50M 최종 결과

### 9.1 학습 결과

세 seed는 각각 약 50M timestep을 완료했다. 전체 완료 episode는
5,801,253판이다.

| Seed | 완료 게임 | 학습 중 승률 | 평균 decision | 평균 turn |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1,933,301 | 56.52% | 50.54 | 33.23 |
| 1 | 1,925,730 | 56.55% | 50.67 | 33.34 |
| 2 | 1,942,222 | 57.00% | 50.14 | 33.24 |
| 합산 | 5,801,253 | 56.69% | - | - |

학습 중 승률은 stochastic policy가 변화하는 전 과정을 누적한 값이다.
최종 deterministic checkpoint의 평가 승률과 직접 비교하지 않는다.

### 9.2 공식 paired 평가

각 모델을 2,500개 base seed에서 선공·후공으로 한 번씩 실행했다.

| Seed | 전체 | 선공 | 후공 | Wilson 95% CI |
| ---: | ---: | ---: | ---: | --- |
| 0 | 60.76% | 61.28% | 60.24% | 59.40%~62.10% |
| 1 | 60.32% | 60.36% | 60.28% | 58.96%~61.67% |
| 2 | **61.08%** | 62.56% | 59.60% | 59.72%~62.42% |
| 합산 | **60.72%** | 61.40% | 60.04% | 59.94%~61.50% |

- 공식 평가: 9,108승 / 15,000판
- 세 seed 모두 60% point threshold 통과
- seed 간 표준편차: 약 0.31%p
- illegal action: 0
- evaluation error: 0

합산 신뢰구간 하한은 60%보다 0.06%p 낮다. 따라서 전체 평균은 60% 부근의
경계선 결과로 해석한다.

### 9.3 Checkpoint 비교

checkpoint 선별에 사용하지 않은 별도 holdout에서 40M과 최종 50M을
비교했다.

| Seed | 40M | 최종 50M | 변화 |
| ---: | ---: | ---: | ---: |
| 0 | 60.32% | 60.98% | +0.66%p |
| 1 | 60.36% | 59.34% | -1.02%p |
| 2 | 59.94% | 62.04% | +2.10%p |
| 합산 | 60.21% | **60.79%** | +0.58%p |

신뢰구간이 겹치므로 50M의 우위를 강하게 주장할 수는 없지만, 최종 모델이
전체적으로 열화됐다는 증거도 없다.

### 9.4 대표 모델

공식 평가와 첫 holdout을 기준으로 seed 2를 대표 모델로 선정했다.
선택에 사용하지 않은 별도 5,000판에서도 60.62%를 기록했다.

seed 2의 서로 겹치지 않는 세 평가 세트를 합치면 다음과 같다.

```text
games: 15,000
wins: 9,187
win rate: 61.25%
Wilson 95% CI: 60.46%~62.02%
illegal actions: 0
evaluation errors: 0
```

대표 모델:

```text
bests/ppo_common_rule_50m_backdo_subproc/
  common_rule_based_seed_2_50m_nenv32_tactical/model.zip
```

해당 디렉터리에는 config, 평가 JSON, SHA-256 checksum과 release 설명도
함께 보관한다.

## 10. 뒷도 규칙 검증

전체 학습 게임에서 양쪽 플레이어를 합친 뒷도 관련 이벤트는 다음과 같다.

| 지표 | 횟수 |
| --- | ---: |
| 뒷도 등장 | 11,369,312 |
| 뒷도 action | 9,733,580 |
| 뒷도 잡기 | 1,783,405 |
| 뒷도로 잡힌 말 | 2,141,191 |
| `O1 -> HOME` | 1,300,626 |
| `HOME -> 이전 칸` | 309,092 |
| 합법 action 없음 auto-pass | 1,499,029 |

첫 칸에서 HOME으로 후진, HOME에서 후진, 역방향 잡기, 뒷도만 있고 움직일
말이 없을 때의 자동 패스가 실제 rollout에서 충분히 발생했다.

이 카운터는 PPO와 상대 agent를 합친 값이므로 PPO 단독 행동 비율로
해석하지 않는다.

## 11. project-RF 비교

### 11.1 공통 환경 교차 평가

project-RF checkpoint를 로컬 공통 환경에서 실행하도록 adapter를 구현했다.
adapter는 252차원 입력과 20-action 출력을 변환하며, 평가 환경의 미래 RNG
상태를 모델에 전달하지 않는다.

| Model | 유형 | 전체 | 선공 | 후공 |
| --- | --- | ---: | ---: | ---: |
| `ppo_capture_imitation` | RL + Rule Hybrid | 59.46% | 60.20% | 58.72% |
| `ppo_tactical` | RL + Rule Hybrid | 55.40% | 57.40% | 53.40% |

두 모델 모두 illegal action과 evaluation error는 0이었다.
`ppo_capture_imitation`은 60% 기준에 27승 부족했다. network-only 100판
smoke 결과는 13%였으므로 높은 성능의 상당 부분이 inference-time
tactical prior에 의존한다.

### 11.2 Legacy 직접 대전

뒷도가 없는 공통 legacy 환경에서 RL 40M PPO와 project-RF
`ppo_capture_imitation`을 직접 대전시켰다.

| Model | 승 | 승률 | Wilson 95% CI |
| --- | ---: | ---: | --- |
| RL 40M seed 1 | 2,434 | 48.68% | 47.30%~50.07% |
| project-RF hybrid | 2,566 | 51.32% | 49.93%~52.70% |

project-RF hybrid가 2.64%p 높았지만 신뢰구간이 50%를 포함하므로 명확한
실력 차이로 단정하지 않는다.

## 12. 검증

현재 전체 테스트 결과:

```text
176 passed
```

주요 검증 범위:

- 윷 확률과 seed 재현성
- 보드 전진·후진 및 route 문맥
- 잡기, 업기, 완주, bonus roll
- 뒷도 legal action과 auto-pass
- observation shape와 값
- action mask
- baseline agent
- Gymnasium 및 vector environment
- reward mode
- PPO model config
- paired-seed 공통 평가
- project-RF checkpoint adapter
- legacy 직접 대전

실행 명령:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q yutnori scripts tests
```

## 13. 실행 방법

### 13.1 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 13.2 짧은 학습

```bash
.venv/bin/python scripts/train_ppo.py \
  --total-timesteps 100000 \
  --seed 0 \
  --opponent common_rule_based \
  --observation-mode tactical \
  --reward-mode terminal \
  --n-envs 2 \
  --vec-env dummy \
  --device cpu \
  --run-dir runs/ppo_quickstart
```

### 13.3 Full-Backdo 50M 학습

```bash
scripts/run_common_rule_50m_backdo_training.sh --dry-run
scripts/run_common_rule_50m_backdo_training.sh
```

기본 설정은 seed 0, 1, 2에 대해 각각 50M timestep, 32개 subprocess
environment, tactical observation과 terminal reward를 사용한다.

### 13.4 공통 평가

```bash
.venv/bin/python scripts/evaluate_common_rule.py \
  --model-path bests/ppo_common_rule_50m_backdo_subproc/common_rule_based_seed_2_50m_nenv32_tactical/model.zip \
  --training-seed 2 \
  --device cpu \
  --output runs/representative_eval.json
```

## 14. 재현성과 산출물

학습 run은 다음 파일을 저장한다.

- `config.json`: 실행 인자, 버전, 환경 설정
- `summary.json`: 학습량과 평가 요약
- `model.zip`: PPO checkpoint
- `episodes.jsonl`: episode 통계
- paired evaluation JSON
- 일정 간격 checkpoint

대표 모델은 `bests/`에 config, 평가 결과, checksum과 함께 별도 보관한다.
실험의 원시 결과는 `runs/`에 남아 있어 보고서 수치를 다시 확인할 수 있다.

## 15. 한계

1. 개별 5,000판 평가의 신뢰구간은 60%를 포함한다.
2. seed 1은 한 holdout에서 59.34%로 내려가 seed 간 차이가 완전히
   사라지지 않았다.
3. tactical observation은 Rule-based score를 입력으로 사용하므로,
   완전히 특징 없는 end-to-end 학습과는 다르다.
4. project-RF adapter는 서로 다른 보드 표현을 변환하므로 원래 환경과
   완전히 동일하지 않다.
5. legacy 모델과 full-backdo 모델은 규칙과 observation shape가 달라
   직접적인 성능 향상 비교가 어렵다.
6. 뒷도 통계는 PPO와 상대의 이벤트를 합친 값이다.

## 16. 결론

- 전체 뒷도 규칙과 action masking이 포함된 학습 환경을 구현했다.
- tactical observation이 base observation보다 높은 성능을 보였다.
- final 50M 세 seed의 공식 합산 승률은 60.72%였다.
- 세 seed 모두 5,000판 point estimate 기준 60%를 통과했다.
- 대표 seed 2는 독립된 15,000판 합산에서 61.25%를 기록했다.
- 모든 주요 평가에서 illegal action과 evaluation error는 0이었다.

따라서 프로젝트의 60% 관측 승률 목표는 달성했다. 다만 전체 평균의
통계적 margin은 크지 않으므로 결과를 60% 부근의 경계선 성능으로
해석하는 것이 적절하다.

## 17. 사용 라이브러리와 참고 자료

이 프로젝트는 다음 공개 라이브러리의 API와 알고리즘 구현을 사용한다.

- Gymnasium: 강화학습 환경 인터페이스
- Stable-Baselines3: 학습 기반 구성 요소
- SB3-Contrib MaskablePPO: action masking을 지원하는 PPO
- PyTorch: 신경망 실행
- NumPy, pandas, matplotlib, tqdm: 수치 처리와 분석

프로젝트의 윷놀이 게임 엔진, observation, tactical feature, baseline,
평가 protocol과 adapter는 이 저장소에서 구현했다.

참고 링크:

- <https://gymnasium.farama.org/>
- <https://stable-baselines3.readthedocs.io/>
- <https://sb3-contrib.readthedocs.io/>
- <https://github.com/DLR-RM/stable-baselines3>
- <https://github.com/Stable-Baselines-Team/stable-baselines3-contrib>
