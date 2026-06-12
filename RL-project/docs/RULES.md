# 윷놀이 게임 및 평가 규칙

이 문서는 현재 코드가 구현하는 `full_backdo_v1` 규칙과 공통 평가 조건을
정리한다.

## 1. 기본 구성

- 플레이어: 2명
- 플레이어당 말: 4개
- 선공: 게임 seed에 따라 결정
- 승리 조건: 자기 말 4개가 모두 `FINISHED`
- 지름길: 분기점에 정확히 도착하면 자동 진입
- 특수 규칙: 뒷도 사용, 낙과 날윷은 제외

말은 다음 상태 중 하나를 가진다.

| 상태 | 의미 |
| --- | --- |
| `WAITING` | 아직 보드에 진입하지 않은 말 |
| `ON_BOARD` | 보드 위에서 이동 중인 말 |
| `FINISHED` | HOME을 통과해 완주한 말 |

## 2. 윷 결과

각 윷가락의 배 확률을 0.6, 등 확률을 0.4로 둔다. 표시된 윷가락의
단일 배 조합을 뒷도로 분리한다.

| 결과 | 이동량 | 확률 | 보너스 던지기 |
| --- | ---: | ---: | :---: |
| 도 | +1 | 11.52% | 아니오 |
| 개 | +2 | 34.56% | 아니오 |
| 걸 | +3 | 34.56% | 아니오 |
| 윷 | +4 | 12.96% | 예 |
| 모 | +5 | 2.56% | 예 |
| 뒷도 | -1 | 3.84% | 아니오 |

확률의 합은 100%다.

## 3. 턴과 Pool

한 턴에 던진 결과는 pool에 누적된다.

1. 턴 시작 시 윷을 던진다.
2. 윷 또는 모가 나오면 보너스 던지기를 이어서 수행한다.
3. 나온 결과를 pool에 저장한다.
4. 플레이어는 pool의 결과 중 하나와 이동할 말을 선택한다.
5. 사용한 결과 하나를 pool에서 제거한다.
6. pool에 결과가 남고 합법 action이 있으면 같은 플레이어가 계속 행동한다.
7. pool이 비었거나 합법 action이 없으면 상대 턴으로 넘어간다.

실제로 상대 말을 잡으면 추가 던지기를 얻는다. 사용한 결과가 윷 또는
모인 경우에는 이미 결과 자체의 보너스가 반영되므로 잡기 보너스를
중복하지 않는다.

모든 말이 `WAITING`인 상태에서 pool에 뒷도만 있으면 합법 action이 없다.
이 경우 pool을 비우고 상대 턴으로 자동 전환한다.

## 4. 보드 좌표

보드는 29개의 physical cell을 사용한다.

### 4.1 외곽

```text
HOME -> O1 -> O2 -> O3 -> O4 -> C1
     -> O6 -> O7 -> O8 -> O9 -> C2
     -> O11 -> O12 -> O13 -> O14 -> C3
     -> O16 -> O17 -> O18 -> O19 -> HOME
```

| ID | Cell | ID | Cell |
| ---: | --- | ---: | --- |
| 0 | `HOME` | 10 | `C2` |
| 1 | `O1` | 11 | `O11` |
| 2 | `O2` | 12 | `O12` |
| 3 | `O3` | 13 | `O13` |
| 4 | `O4` | 14 | `O14` |
| 5 | `C1` | 15 | `C3` |
| 6 | `O6` | 16 | `O16` |
| 7 | `O7` | 17 | `O17` |
| 8 | `O8` | 18 | `O18` |
| 9 | `O9` | 19 | `O19` |

### 4.2 내부

```text
C1 -> A1 -> A2 -> CENTER -> A3 -> A4 -> C3
C2 -> B1 -> B2 -> CENTER -> B3 -> B4 -> HOME
```

| ID | Cell | 설명 |
| ---: | --- | --- |
| 20 | `A1` | C1 지름길 |
| 21 | `A2` | C1 지름길 |
| 22 | `CENTER` | 중앙 공유 칸 |
| 23 | `A3` | CENTER에서 C3 방향 |
| 24 | `A4` | C3 직전 |
| 25 | `B1` | C2 지름길 |
| 26 | `B2` | C2 지름길 |
| 27 | `B3` | CENTER에서 HOME 방향 |
| 28 | `B4` | HOME 직전 |

## 5. Logical Route

잡기와 업기는 physical cell로 판정하지만, 다음 이동 방향과 뒷도 목적지는
logical route로 결정한다.

```text
OUTER:
HOME, O1, O2, O3, O4, C1, O6, O7, O8, O9, C2,
O11, O12, O13, O14, C3, O16, O17, O18, O19, HOME

C1_DIAGONAL:
C1, A1, A2, CENTER, A3, A4, C3, O16, O17, O18, O19, HOME

C2_DIAGONAL:
C2, B1, B2, CENTER, B3, B4, HOME

CENTER_TO_HOME:
CENTER, B3, B4, HOME
```

`CENTER_TO_HOME`의 position은 `entry_route`를 보존한다. 같은 CENTER나
HOME에 있어도 진입 경로에 따라 뒷도 목적지가 달라질 수 있기 때문이다.

## 6. 지름길

- OUTER에서 C1에 정확히 도착하면 `C1_DIAGONAL`로 진입한다.
- OUTER에서 C2에 정확히 도착하면 `C2_DIAGONAL`로 진입한다.
- C3에는 별도의 새 지름길이 없다.
- CENTER에 정확히 도착하면 `CENTER_TO_HOME`으로 진입한다.
- 분기점을 지나치기만 하면 현재 route를 유지한다.
- 지름길 진입은 플레이어가 선택하지 않는다.

## 7. 출발과 완주

- `WAITING` 말은 사용한 양수 이동량만큼 HOME에서 출발한다.
- HOME에 정확히 도착하면 `ON_BOARD` 상태로 HOME에 머문다.
- HOME을 초과하면 즉시 `FINISHED`가 된다.
- HOME에 머문 말은 다음 양수 이동으로 `FINISHED`가 된다.
- HOME에 머문 상대 말은 잡을 수 있다.
- stack이 HOME을 통과하면 stack 전체가 완주한다.

예:

```text
O18 + 개 -> HOME
O18 + 걸 -> FINISHED
O19 + 도 -> HOME
O19 + 개 -> FINISHED
HOME + 양수 결과 -> FINISHED
```

## 8. 업기와 잡기

- 자기 말이 있는 physical cell에 도착하면 자동으로 업는다.
- 업힌 말은 분리하지 않고 stack 전체가 함께 이동한다.
- 상대 말이 있는 physical cell에 도착하면 해당 칸의 상대 말 전체를 잡는다.
- 잡힌 말은 `WAITING`으로 돌아간다.
- 지나가는 칸에서는 잡기나 업기가 발생하지 않는다.
- 전진과 후진 모두 도착 physical cell에서 같은 규칙을 적용한다.

같은 stack을 중복 action으로 표현하지 않기 위해 가장 작은 `piece_id`만
대표 action으로 허용한다.

## 9. 뒷도

- `ON_BOARD` 말만 뒷도를 사용할 수 있다.
- `WAITING`과 `FINISHED` 말은 뒷도를 사용할 수 없다.
- 현재 logical route를 기준으로 한 칸 뒤로 이동한다.
- 뒷도로 분기점에 도착해도 새 route로 갈아타지 않는다.
- 목적지에서 일반 이동과 동일하게 잡기와 업기를 처리한다.

주요 역방향 이동:

```text
O1 -> HOME
OUTER HOME -> O19
CENTER_TO_HOME HOME -> B4
C1 경유 CENTER -> A2
C2 경유 CENTER -> B2
C1_DIAGONAL C3 -> A4
OUTER C3 -> O14
C1 -> O4
C2 -> O9
```

## 10. Action Encoding

```python
action = piece_id * 6 + yut_result_id
piece_id = action // 6
yut_result_id = action % 6
```

`yut_result_id`의 순서:

```text
0 DO
1 GAE
2 GEOL
3 YUT
4 MO
5 BACK_DO
```

합법 action 조건:

- 현재 플레이어의 차례다.
- pool에 해당 윷 결과가 남아 있다.
- 말이 `FINISHED`가 아니다.
- 뒷도라면 말이 `ON_BOARD` 상태다.
- stack의 대표 말이다.

## 11. Observation

### Base

62차원:

- 양측 말 position 8개
- 양측 말 status 8개
- 양측 말 logical track 8개
- 양측 stack matrix 32개
- pool count 6개

### Tactical

302차원:

```text
base 62 + 24 actions x 10 tactical features
```

action feature는 legal, 잡기, 완주, 이동 말 수, 대기 말 출발, stack,
이동 후 거리와 Rule-based score를 포함한다.

## 12. Reward

최종 학습의 기본 reward는 terminal reward다.

```text
learner win  +1
learner loss -1
non-terminal  0
```

선택적 `rf_shaped` mode도 구현되어 있지만 최종 대표 모델에는 사용하지
않는다.

## 13. 공통 Rule-based Agent

공통 agent는 각 합법 action에 다음 점수를 적용한다.

- 완주: +100
- 잡기: +50
- 대기 말 출발: +5
- stack 추가 말: 말당 +4
- 완주까지 거리: 칸당 -0.5

동점이면 가장 작은 action ID를 선택한다. 이 tie-break를 포함한 agent
동작은 평가 중 변경하지 않는다.

## 14. 공통 평가

- base seed 2,500개 사용
- 각 seed에서 모델 선공과 후공을 한 번씩 실행
- 총 5,000판
- deterministic inference
- 평가 RNG 상태나 미래 윷 결과를 agent에 제공하지 않음
- illegal action은 모델 패배로 처리
- 전체·선공·후공 승률과 Wilson 95% 신뢰구간 보고
- seed 목록 또는 범위와 SHA-256 기록

평가 결과에는 다음 값이 포함된다.

- 완료 게임 수
- 승·패와 전체 승률
- 선공·후공 승률
- 평균 turn과 decision
- illegal action 수
- evaluation error 수
- ruleset, action size, observation mode

## 15. 제외 규칙

- 낙
- 날윷
- 함정 또는 퐁당
- 대기 말의 뒷도
- 플레이어가 직접 선택하는 지름길
- 평가 중 미래 RNG 참조

## 16. 구현 위치

| 규칙 | 파일 |
| --- | --- |
| 윷 결과와 확률 | `yutnori/core/yut.py` |
| 보드와 route | `yutnori/core/board.py` |
| 턴, pool, 잡기, 승패 | `yutnori/core/game.py` |
| observation과 mask | `yutnori/env/yutnori_env.py` |
| 공통 Rule-based Agent | `yutnori/agents/baseline.py` |
| tactical feature | `yutnori/agents/tactical_features.py` |
| paired 평가 | `yutnori/training/common_evaluation.py` |
