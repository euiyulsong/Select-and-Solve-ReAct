# ReAct vs Plan-ReAct vs Router Retrieval Experiment

## Experiment

Single-turn retrieval QA에서 다음 세 가지 execution strategy를 비교했다.

- **Always ReAct**: original query로 검색 후, evidence가 부족하면 follow-up query를 생성하며 최대 3회 검색
- **Always Plan-ReAct**: 검색 전에 retrieval plan을 생성한 뒤 ReAct 수행
- **Router**: LLM이 질문만 보고 `0=ReAct`, `1=Plan-ReAct`를 선택

Dataset:

- **SQuAD**: 100 queries
- **HotpotQA**: 100 queries
- Total: **200 queries**

모든 평가 query의 gold/supporting document는 검색 corpus에 포함되도록 구성했다.

---

## Overall Results

| Method | Support Recall | Full Support | Hit Any | MRR | Search Calls | Early Stop | Latency (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Always Plan-ReAct | 0.8675 | 0.790 | **0.945** | 0.8361 | 1.630 | 0.800 | 4.111 |
| **Always ReAct** | **0.8725** | **0.820** | 0.925 | **0.8555** | **1.495** | **0.850** | 2.041 |
| Router | 0.8650 | 0.805 | 0.925 | 0.8417 | 1.525 | 0.845 | 1.275* |

Overall에서는 **Always ReAct가 가장 좋은 retrieval-quality / efficiency trade-off**를 보였다.

Always ReAct는 Plan-ReAct 대비:

- Support Recall: **+0.5%p**
- Full Support: **+3.0%p**
- MRR: **+1.93%p**
- Search calls: **1.63 → 1.50**
- Latency: **4.11s → 2.04s**

Plan-ReAct는 `hit_any`가 2%p 높았지만, 필요한 evidence를 **전부 찾는 Full Support와 ranking quality에서는 ReAct가 더 좋았다.**

---

## HotpotQA

| Method | Support Recall | Full Support | Hit Any | MRR | Search Calls | Latency (s) |
|---|---:|---:|---:|---:|---:|---:|
| Always Plan-ReAct | 0.825 | 0.67 | **0.98** | 0.8567 | 1.77 | 4.388 |
| **Always ReAct** | **0.865** | **0.76** | 0.97 | **0.8823** | **1.59** | 2.288 |
| Router | 0.850 | 0.73 | 0.97 | 0.8548 | 1.64 | 1.325* |

가장 흥미로운 결과는 **multi-hop dataset인 HotpotQA에서도 Plan-ReAct보다 ReAct가 더 좋았다는 점**이다.

ReAct는 Plan-ReAct 대비:

- Support Recall: **+4%p**
- Full Support: **+9%p**
- MRR: **+2.56%p**
- Search calls: **1.77 → 1.59**

즉,

> **Multi-hop이라고 해서 upfront planning이 항상 필요한 것은 아니다.**

HotpotQA의 많은 질문은 첫 검색으로 intermediate entity를 얻은 뒤, 해당 observation을 이용해 다음 query를 생성하는 **sequential ReAct 방식**으로 충분히 해결할 수 있었다.

예를 들어 다음과 같은 dependency chain은 upfront decomposition보다 observation 기반 검색이 자연스럽다.

```text
Question
  ↓
Search #1
  ↓
Intermediate entity 발견
  ↓
Follow-up query 생성
  ↓
Search #2
```

이는 질문 전체를 처음부터 완전히 decomposition하는 것보다 실제 retrieval 결과를 이용해 다음 hop을 결정하는 것이 더 효과적일 수 있음을 보여준다.

---

## SQuAD

| Method | Support Recall | Full Support | MRR | Search Calls | Latency (s) |
|---|---:|---:|---:|---:|---:|
| **Always Plan-ReAct** | **0.910** | **0.91** | 0.8156 | 1.49 | 3.834 |
| Always ReAct | 0.880 | 0.88 | **0.8287** | **1.40** | 1.793 |
| Router | 0.880 | 0.88 | **0.8287** | 1.41 | 1.225* |

SQuAD에서는 예상과 다르게 Plan-ReAct가 Support Recall을 **3%p 개선**했다.

하지만 이를 곧바로 **“single-hop에서도 planning이 좋다”**고 해석하면 안 된다.

현재 Plan-ReAct의 첫 단계는 original query를 그대로 검색하지 않고 planner가 새로운 `first_query`를 생성한다.

따라서 실제 비교는 부분적으로:

```text
ReAct
Original Query → Retrieval
```

vs.

```text
Plan-ReAct
LLM-generated Search Query → Retrieval
```

이 된다.

즉 SQuAD에서의 +3%p는 **planning 효과뿐 아니라 query rewriting / reformulation 효과가 포함된 결과**일 가능성이 높다.

또한 Plan-ReAct는 recall은 높지만 MRR은 낮았다.

```text
Recall
Plan-ReAct  0.910
ReAct       0.880

MRR
Plan-ReAct  0.8156
ReAct       0.8287
```

이는 planner-generated query가 더 많은 gold document를 찾는 데는 도움이 되었지만, gold document를 항상 더 높은 순위에 배치하지는 못했음을 의미한다.

---

## Router Behavior

Router의 실제 선택 비율은 다음과 같았다.

| Dataset | ReAct | Plan-ReAct |
|---|---:|---:|
| HotpotQA | **81%** | 19% |
| SQuAD | **98%** | 2% |

Router는 전체적으로 **ReAct를 매우 강하게 선호**했다.

특히 HotpotQA조차 81%를 ReAct로 routing했다.

이는 처음 설정했던 단순 heuristic:

```text
SQuAD   → ReAct
Hotpot  → Plan-ReAct
```

과 상당히 다르다.

실제 agreement는:

```text
Router agreement with dataset-static rule = 58.5%
```

였다.

그러나 이 **58.5%를 router accuracy로 해석하면 안 된다.**

실험 결과 자체가 HotpotQA에서:

```text
ReAct > Plan-ReAct
```

였기 때문이다.

따라서 `HotpotQA = Plan-ReAct`라는 static label 자체가 좋은 ground truth가 아니다.

---

## Did the Router Help?

현재 결과에서는 **아직 아니다.**

Overall retrieval quality:

```text
Support Recall
Always ReAct    0.8725
Router          0.8650

Full Support
Always ReAct    0.820
Router          0.805

MRR
Always ReAct    0.8555
Router          0.8417
```

Router가 대부분 ReAct를 선택했음에도 불구하고, 가끔 선택한 Plan-ReAct가 Always ReAct를 넘어설 만큼의 이득을 만들지 못했다.

즉 현재 zero-shot router는:

```text
Always ReAct
   >
Router
   >
Always Plan-ReAct
```

에 가까운 결과를 보였다.

현재 데이터에서는 **ReAct가 강한 fixed baseline**이다.

---

## Important Latency Caveat

Router latency는 직접 비교하면 안 된다.

```text
Always ReAct latency = 2.041 s
Plan-ReAct latency   = 4.111 s
Router latency       = 1.275 s
```

Router가 실제로 ReAct보다 빠른 것이 아니다.

이번 구현에서는 동일한 LLM prompt 결과를 strategy 사이에서 공유하는 **LLM cache**를 사용했다.

Router의 평균:

```text
llm_calls      = 1.000
llm_cache_hits = 1.475
```

즉 Router를 실행하기 전에 Always ReAct / Plan-ReAct에서 계산된 follow-up 결과 상당수를 재사용했다.

따라서 현재 latency는:

> **실험 전체 실행 시간을 줄이기 위한 cached latency이며 standalone production latency가 아니다.**

공정한 end-to-end latency 비교를 위해서는 별도로:

```bash
--no-cache
```

실험을 수행해야 한다.

Retrieval quality 비교에는 cache가 영향을 주지 않지만 latency/cost 비교에는 영향을 준다.

---

## Main Findings

### 1. Multi-hop ≠ Plan-ReAct

가장 중요한 결과다.

HotpotQA에서도 ReAct가:

```text
Support Recall +4%p
Full Support  +9%p
```

높았다.

Multi-hop 여부만으로 planning을 결정하는 것은 지나치게 단순하다.

---

### 2. Sequential dependency는 ReAct에 잘 맞는다

다음 hop이 이전 retrieval 결과에 의존하는 경우:

```text
A 검색
 ↓
B 발견
 ↓
B를 이용해 C 검색
```

처음부터 계획하는 것보다 **Search → Observe → Follow-up Query**가 자연스럽다.

---

### 3. Planning은 branching/composition에서 선택적으로 필요할 가능성이 높다

Plan-ReAct가 유리할 가능성이 높은 문제는 단순한 multi-hop보다는:

```text
A 조사 ─┐
        ├─ Compare / Compose
B 조사 ─┘
```

처럼 여러 evidence need가 병렬적으로 존재하는 경우다.

따라서 router는:

```text
single-hop vs multi-hop
```

를 분류하기보다,

```text
Will upfront decomposition outperform sequential retrieval?
```

을 예측하는 것이 더 적절하다.

---

### 4. 현재 Zero-shot Router는 Best Fixed Strategy를 넘지 못했다

Select-then-Solve 관점에서도 중요한 결과다.

현재 router:

```text
Question
 ↓
GPT Router
 ↓
ReAct / Plan-ReAct
```

는 best fixed strategy인 Always ReAct를 이기지 못했다.

따라서 다음 단계는 heuristic routing을 더 복잡하게 만드는 것보다 **실제 strategy outcome을 label로 사용하는 것**이 더 자연스럽다.

---

## Recommended Next Experiment

각 question마다 ReAct와 Plan-ReAct를 모두 실행하고 실제 winner를 계산한다.

예:

```text
utility =
    support_recall
  + λ * full_support
  + α * MRR
  - β * search_calls
  - γ * latency
```

그리고:

```text
ReAct utility > Plan-ReAct utility
→ label = 0

Plan-ReAct utility > ReAct utility
→ label = 1
```

로 만든다.

이후:

```text
Question
   ↓
Learned Router
   ↓
0 / 1
```

을 학습하면 **Select-then-Solve의 paradigm routing 방식에 더 가까운 실험**이 된다.

특히 분석해야 할 것은 다음 네 그룹이다.

```text
SQuAD + ReAct wins
SQuAD + Plan-ReAct wins

Hotpot + ReAct wins
Hotpot + Plan-ReAct wins
```

이 네 그룹의 question pattern을 비교하면 실제로 **어떤 semantic/structural feature가 planning의 marginal utility를 결정하는지** 확인할 수 있다.

---

## Conclusion

이번 실험에서는 **Always ReAct가 가장 강한 baseline**이었다.

```text
Overall Support Recall

ReAct       0.8725  ← Best
Plan-ReAct  0.8675
Router      0.8650
```

특히 HotpotQA에서 ReAct가 Plan-ReAct보다 높은 Full Support를 기록했다.

```text
HotpotQA Full Support

ReAct       0.76
Router      0.73
Plan-ReAct  0.67
```

따라서 현재 결과는:

> **Multi-hop이라는 이유만으로 upfront planning을 추가할 필요는 없으며, retrieval observation을 이용해 다음 query를 생성하는 ReAct가 상당히 강하다.**

또한 현재 zero-shot Router는 best fixed strategy를 넘지 못했다.

다음 단계에서는 dataset type이나 heuristic complexity를 ground truth로 사용하기보다, **각 query에서 실제 ReAct / Plan-ReAct execution 결과를 비교해 winner label을 만들고 이를 학습하는 outcome-based routing**이 더 적절하다.
