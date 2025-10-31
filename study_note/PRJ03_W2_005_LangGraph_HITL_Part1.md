# LangGraph HITL (Human-in-the-Loop) - Part 1: 기본 개념 및 Breakpoint

## 📚 학습 목표

이 가이드를 완료하면 다음을 할 수 있습니다:

1. **HITL 개념 이해**: Human-in-the-Loop 패턴의 필요성과 활용 시나리오를 이해한다
2. **Breakpoint 설정**: 정적 및 동적 브레이크포인트를 설정하고 차이점을 파악한다
3. **interrupt 함수 활용**: 노드 실행 중 동적으로 사용자 개입을 요청하는 방법을 구현한다
4. **상태 관리**: `get_state()`로 실행 상태를 확인하고 이해한다
5. **실행 제어**: `Command` 객체를 사용하여 워크플로우 실행을 제어한다
6. **resume 패턴**: 중단된 워크플로우를 다양한 방식으로 재개하는 방법을 습득한다
7. **실무 적용**: HITL 패턴을 실제 프로젝트에 적용할 수 있는 능력을 배양한다

## 🔑 핵심 개념

### Human-in-the-Loop (HITL)란?

**HITL (Human-in-the-Loop)**은 AI 워크플로우에서 사용자의 개입이 필요한 시점에 실행을 중단하고, 사용자의 확인, 수정, 승인을 받은 후 다시 실행을 재개하는 패턴입니다.

**왜 HITL이 필요한가?**
- **품질 보장**: LLM의 출력을 사용자가 검증하여 오류를 방지
- **정확성 향상**: 중요한 의사결정 단계에서 인간의 판단력 활용
- **위험 관리**: 비용이 많이 드는 작업(API 호출, 외부 시스템 연동) 전 사용자 승인
- **맞춤화**: 사용자 피드백을 즉시 반영하여 결과 개선
- **신뢰성 확보**: 자동화된 워크플로우에 대한 사용자 신뢰 구축

### LangGraph의 Breakpoint 메커니즘

LangGraph는 두 가지 방식으로 Breakpoint를 설정할 수 있습니다:

#### 1. 정적 브레이크포인트 (Static Breakpoint)
- **설정 위치**: `compile()` 함수의 파라미터로 설정
- **설정 시점**: 그래프 컴파일 시 고정
- **파라미터**:
  - `interrupt_before=["node_name"]`: 특정 노드 실행 **전**에 중단
  - `interrupt_after=["node_name"]`: 특정 노드 실행 **후**에 중단
- **특징**: 디버깅, 고정된 검토 지점에 유용

#### 2. 동적 브레이크포인트 (Dynamic Breakpoint)
- **설정 위치**: 노드 함수 내부에서 `interrupt()` 함수 호출
- **설정 시점**: 실행 시점에 조건에 따라 동적으로 결정
- **파라미터**: `interrupt(value)` - 사용자에게 전달할 데이터
- **특징**: 조건부 개입, 사용자 입력이 필요한 경우에 유용

### 주요 구성 요소

```python
# 1. Checkpointer (필수)
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()

# 2. 정적 브레이크포인트
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["step_3"]  # 노드 이름 지정
)

# 3. 동적 브레이크포인트
from langgraph.types import interrupt

def my_node(state):
    user_input = interrupt({
        "message": "계속하시겠습니까?",
        "options": ["yes", "no"]
    })
    # user_input 사용
    return state

# 4. 상태 확인
state = graph.get_state(config)

# 5. 실행 재개
from langgraph.types import Command
result = graph.invoke(Command(resume="user_value"), config)
```

### Checkpointer의 역할

HITL을 사용하려면 **반드시 Checkpointer가 필요**합니다:
- **상태 저장**: 중단 시점의 그래프 상태를 저장
- **재개 지원**: 저장된 상태에서 실행을 재개
- **옵션**:
  - `InMemorySaver()`: 메모리 내 저장 (개발/테스트용)
  - `SqliteSaver()`: SQLite DB에 저장 (프로덕션용)
  - `PostgresSaver()`: PostgreSQL DB에 저장 (프로덕션용)

### Command 객체

실행을 제어하는 다양한 방법:
- `Command(resume=value)`: 중단된 지점에서 값을 전달하며 재개
- `Command(goto="node_name")`: 특정 노드로 이동하여 실행
- `graph.invoke(None, config)`: 아무 값도 전달하지 않고 재개

## 🛠 환경 설정

### 필수 라이브러리 설치

```bash
pip install -qU \
    langgraph \
    langchain-openai \
    python-dotenv
```

### API 키 설정

`.env` 파일에 OpenAI API 키를 설정합니다:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 기본 설정 코드

```python
from dotenv import load_dotenv
import os

# 환경 변수 로드
load_dotenv()

# API 키 확인
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

print("환경 설정 완료!")
```

## 💻 단계별 구현

### Step 1: 정적 브레이크포인트 구현

정적 브레이크포인트는 컴파일 시점에 고정된 위치에 중단점을 설정합니다.

```python
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display

# 1. 상태 정의
class SimpleState(TypedDict):
    input: str

# 2. 노드 함수 정의
def step_1(state):
    print("---Step 1 실행---")
    return {"input": state.get("input", "") + " (1)"}

def step_2(state):
    print("---Step 2 실행---")
    return {"input": state.get("input", "") + " (2)"}

def step_3(state):
    print("---Step 3 실행---")
    return {"input": state.get("input", "") + " (3)"}

# 3. 그래프 구성
builder = StateGraph(SimpleState)
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)

builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)

# 4. Checkpointer 설정
checkpointer = InMemorySaver()

# 5. 컴파일 시 정적 브레이크포인트 설정
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["step_3"]  # step_3 실행 전에 중단
)

# 그래프 시각화
display(Image(graph.get_graph().draw_mermaid_png()))
```

#### 실행 및 중단 확인

```python
import uuid

# 스레드 설정
thread = {"configurable": {"thread_id": str(uuid.uuid4())}}

# 초기 실행 - interrupt_before로 인해 step_3 전에 중단됨
initial_input = {"input": "hello world"}
result = graph.invoke(initial_input, thread)

print(f"실행 결과: {result}")
# 출력:
# ---Step 1 실행---
# ---Step 2 실행---
# 실행 결과: {'input': 'hello world (1) (2)'}
```

#### 상태 확인

```python
# 현재 상태 조회
state = graph.get_state(thread)

print(f"다음 실행될 노드: {state.next}")  # ('step_3',)
print(f"현재 값: {state.values}")          # {'input': 'hello world (1) (2)'}
print(f"메타데이터: {state.metadata}")
```

**상태 객체의 주요 속성:**
- `state.values`: 현재 그래프의 상태 값
- `state.next`: 다음에 실행될 노드 튜플
- `state.config`: 현재 스레드 설정
- `state.metadata`: 실행 메타데이터 (step, source 등)
- `state.tasks`: 현재 실행 대기 중인 작업

#### 실행 재개

```python
# None을 입력으로 전달하여 재개
print("\n=== 실행 재개 ===")
final_result = graph.invoke(None, thread)

print(f"최종 결과: {final_result}")
# 출력:
# ---Step 3 실행---
# 최종 결과: {'input': 'hello world (1) (2) (3)'}
```

**재개 방법:**
- `graph.invoke(None, thread)`: 저장된 상태에서 그대로 재개
- `graph.invoke({"input": "new value"}, thread)`: 새로운 값으로 상태 업데이트 후 재개

### Step 2: 동적 브레이크포인트 구현 (interrupt 함수)

동적 브레이크포인트는 노드 실행 중 조건에 따라 중단하고 사용자 입력을 받습니다.

```python
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from IPython.display import Image, display
import uuid

# 1. 상태 정의 - user_decision 필드 추가
class SimpleState(TypedDict):
    input: str
    user_decision: str

# 2. 노드 함수 정의
def step_1(state):
    print("---Step 1 실행---")
    return {"input": state.get("input", "") + " (1)"}

def step_2(state):
    print("---Step 2 실행---")
    return {"input": state.get("input", "") + " (2)"}

def step_3_with_interrupt(state):
    print("---Step 3 시작 (동적 브레이크포인트)---")

    # 동적 브레이크포인트: interrupt() 함수 호출
    user_decision = interrupt({
        "message": "Step 3를 실행하시겠습니까?",
        "current_state": state.get("input", ""),
        "options": ["proceed", "skip", "modify"]
    })

    # user_decision 값에 따라 분기 처리
    if user_decision == "proceed":
        print("---Step 3 실행 (사용자 승인됨)---")
        return {
            "input": state.get("input", "") + " (3)",
            "user_decision": user_decision
        }
    elif user_decision == "skip":
        print("---Step 3 건너뜀 (사용자 요청)---")
        return {
            "input": state.get("input", "") + " (skipped)",
            "user_decision": user_decision
        }
    else:  # modify
        print("---Step 3 수정됨 (사용자 요청)---")
        return {
            "input": state.get("input", "") + " (modified)",
            "user_decision": user_decision
        }

# 3. 그래프 구성
builder = StateGraph(SimpleState)
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3_with_interrupt)

builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)

# 4. Checkpointer 설정 (interrupt 사용을 위해 필수)
checkpointer = InMemorySaver()

# 5. 컴파일 (동적 브레이크포인트는 코드 내에서 interrupt()로 설정)
graph = builder.compile(checkpointer=checkpointer)

# 그래프 시각화
display(Image(graph.get_graph().draw_mermaid_png()))
```

#### 실행 및 interrupt 확인

```python
# 스레드 설정
thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}

# 초기 실행 - interrupt()가 호출되면 중단됨
initial_input = {"input": "hello world"}
result = graph.invoke(initial_input, thread_config)

print(f"실행 결과: {result}")
# 출력:
# ---Step 1 실행---
# ---Step 2 실행---
# ---Step 3 시작 (동적 브레이크포인트)---
# 실행 결과: {'input': 'hello world (1) (2)'}
```

#### interrupt 정보 확인

```python
# interrupt에 전달된 데이터 확인
print(f"Interrupt 내용: {result.get('__interrupt__')}")
# 출력: [Interrupt(value={'message': 'Step 3를 실행하시겠습니까?', ...})]
```

#### 상태 확인

```python
# 현재 상태 조회
state = graph.get_state(thread_config)

print(f"다음 실행될 노드: {state.next}")  # ('step_3',)
print(f"현재 값: {state.values}")
print(f"Interrupt 정보: {state.tasks}")
# tasks에서 interrupt의 상세 정보를 확인할 수 있음
```

### Step 3: Command를 사용한 실행 제어

`Command` 객체를 사용하면 재개 시 값을 전달할 수 있습니다.

#### 패턴 1: resume으로 값 전달

```python
# "proceed" 선택으로 재개
final_result = graph.invoke(
    Command(resume="proceed"),  # interrupt()의 반환값이 "proceed"가 됨
    thread_config
)

print(f"최종 결과: {final_result}")
# 출력:
# ---Step 3 실행 (사용자 승인됨)---
# 최종 결과: {'input': 'hello world (1) (2) (3)', 'user_decision': 'proceed'}
```

#### 패턴 2: 다른 옵션으로 재개

```python
# 새로운 스레드로 다른 옵션 테스트
thread_config_2 = {"configurable": {"thread_id": str(uuid.uuid4())}}

# 초기 실행
result = graph.invoke(initial_input, thread_config_2)

# "skip" 선택으로 재개
final_result_2 = graph.invoke(
    Command(resume="skip"),  # interrupt()의 반환값이 "skip"이 됨
    thread_config_2
)

print(f"Skip 결과: {final_result_2}")
# 출력:
# ---Step 3 건너뜀 (사용자 요청)---
# Skip 결과: {'input': 'hello world (1) (2) (skipped)', 'user_decision': 'skip'}
```

#### 패턴 3: 상태 확인 후 조건부 재개

```python
# 상태 확인
state = graph.get_state(thread_config)

if state.next:
    # 사용자 입력 받기 (실제로는 UI에서 받음)
    user_choice = "modify"  # 예시

    # 사용자 선택에 따라 재개
    result = graph.invoke(Command(resume=user_choice), thread_config)
    print(f"결과: {result}")
```

### Step 4: 핵심 패턴 정리

#### 정적 vs 동적 브레이크포인트 비교

| 특성 | 정적 브레이크포인트 | 동적 브레이크포인트 |
|------|-------------------|-------------------|
| 설정 위치 | `compile()` 파라미터 | 노드 내부 `interrupt()` |
| 설정 시점 | 컴파일 시 고정 | 실행 시 동적 결정 |
| 조건부 설정 | 불가능 | 가능 (조건문 사용) |
| 데이터 전달 | 불가능 | 가능 (interrupt에 데이터 전달) |
| 재개 방법 | `invoke(None, config)` | `invoke(Command(resume=value), config)` |
| 사용 사례 | 디버깅, 고정 검토 지점 | 조건부 개입, 사용자 입력 필요 |

#### Checkpointer 필수 사항

```python
# ❌ 잘못된 예시 - Checkpointer 없음
graph = builder.compile()
# HITL 사용 불가능!

# ✅ 올바른 예시 - Checkpointer 설정
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
# HITL 사용 가능!
```

#### 상태 관리 패턴

```python
# 1. 현재 상태 확인
state = graph.get_state(config)

# 2. 다음 실행될 노드 확인
if state.next:
    print(f"대기 중인 노드: {state.next}")
else:
    print("워크플로우 완료")

# 3. 실행 이력 확인
for checkpoint in graph.get_state_history(config):
    print(f"Step: {checkpoint.metadata.get('step')}")
    print(f"Values: {checkpoint.values}")
```

#### Command 활용 패턴

```python
# 패턴 1: 값 전달하며 재개
graph.invoke(Command(resume="user_input"), config)

# 패턴 2: 특정 노드로 이동
graph.invoke(Command(goto="node_name"), config)

# 패턴 3: 값 전달 없이 재개
graph.invoke(None, config)

# 패턴 4: 새로운 입력으로 상태 업데이트 후 재개
graph.invoke({"new_field": "value"}, config)
```

## 🎯 실습 문제

### 문제 1: 기본 승인 워크플로우 (난이도: ⭐⭐⭐)

문서 생성 워크플로우를 구현하세요. 다음 단계를 포함해야 합니다:
1. 초안 생성 (draft)
2. **사용자 승인 요청 (정적 브레이크포인트)**
3. 최종 문서 생성 (finalize)

**요구사항:**
- `interrupt_before`를 사용하여 finalize 전에 중단
- 상태에 `document`, `approved` 필드 포함
- 사용자가 승인하면 finalize 실행, 거부하면 draft로 돌아가기

**힌트:**
```python
class DocumentState(TypedDict):
    document: str
    approved: bool

# interrupt_before=["finalize"] 사용
```

### 문제 2: 조건부 검토 시스템 (난이도: ⭐⭐⭐⭐)

코드 리뷰 시스템을 구현하세요. 다음 조건을 만족해야 합니다:
1. 코드 분석 노드 실행
2. **위험도가 "high"인 경우에만 interrupt()로 사용자 검토 요청**
3. 사용자 선택에 따라 분기:
   - "approve": 배포 진행
   - "reject": 수정 요청
   - "review_later": 대기 상태로 전환

**요구사항:**
- 동적 브레이크포인트 사용 (interrupt 함수)
- 위험도 계산 로직 포함
- Command를 사용한 재개 구현

**힌트:**
```python
def review_code(state):
    risk_level = calculate_risk(state["code"])

    if risk_level == "high":
        decision = interrupt({
            "message": "고위험 코드 발견",
            "risk_level": risk_level,
            "options": ["approve", "reject", "review_later"]
        })
        # decision 값에 따라 분기

    return state
```

### 문제 3: 다단계 승인 워크플로우 (난이도: ⭐⭐⭐⭐⭐)

예산 승인 시스템을 구현하세요. 다음 단계를 포함해야 합니다:
1. 예산 요청서 작성
2. **팀 리더 승인** (interrupt)
3. **금액이 $10,000 이상이면 임원 승인** (조건부 interrupt)
4. 최종 승인 또는 거부

**요구사항:**
- 두 개의 동적 브레이크포인트 사용
- 금액에 따른 조건부 워크플로우
- 각 승인 단계에서 피드백 입력 가능
- 상태 이력 관리 (누가, 언제, 어떤 결정을 했는지)

**힌트:**
```python
class BudgetState(TypedDict):
    request: dict
    amount: float
    team_leader_approved: bool
    executive_approved: bool
    feedback: list

def team_leader_review(state):
    decision = interrupt({
        "message": "팀 리더 승인 필요",
        "request": state["request"],
        "options": ["approve", "reject", "request_changes"]
    })
    # ...

def executive_review(state):
    if state["amount"] >= 10000:
        decision = interrupt({
            "message": "임원 승인 필요",
            "amount": state["amount"]
        })
        # ...
    return state
```

## ✅ 솔루션 예시

### 문제 1 솔루션: 기본 승인 워크플로우

```python
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
import uuid

# 1. 상태 정의
class DocumentState(TypedDict):
    document: str
    approved: bool

# 2. 노드 함수 정의
def draft(state):
    """초안 생성"""
    print("--- 초안 생성 중 ---")
    return {
        "document": "문서 초안: LangGraph HITL 가이드",
        "approved": False
    }

def finalize(state):
    """최종 문서 생성"""
    print("--- 최종 문서 생성 중 ---")
    return {
        "document": state["document"] + " [최종본]",
        "approved": True
    }

# 3. 그래프 구성
builder = StateGraph(DocumentState)
builder.add_node("draft", draft)
builder.add_node("finalize", finalize)

builder.add_edge(START, "draft")
builder.add_edge("draft", "finalize")
builder.add_edge("finalize", END)

# 4. Checkpointer 및 정적 브레이크포인트 설정
checkpointer = InMemorySaver()
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["finalize"]  # finalize 전에 중단
)

# 5. 실행 테스트
thread = {"configurable": {"thread_id": str(uuid.uuid4())}}

# 초기 실행 - finalize 전에 중단됨
result = graph.invoke({"document": "", "approved": False}, thread)
print(f"\n중단 시점 결과: {result}")

# 상태 확인
state = graph.get_state(thread)
print(f"다음 실행될 노드: {state.next}")  # ('finalize',)
print(f"현재 문서: {state.values['document']}")

# 사용자 승인 시나리오 1: 승인
print("\n=== 사용자 승인 ===")
final_result = graph.invoke(None, thread)  # 재개
print(f"최종 결과: {final_result}")

# 사용자 승인 시나리오 2: 거부 (새로운 스레드)
thread2 = {"configurable": {"thread_id": str(uuid.uuid4())}}
result = graph.invoke({"document": "", "approved": False}, thread2)

print("\n=== 사용자 거부 - 처음부터 다시 시작 ===")
# 거부 시 새로운 스레드로 처음부터 다시 실행
thread3 = {"configurable": {"thread_id": str(uuid.uuid4())}}
result = graph.invoke({"document": "", "approved": False}, thread3)
print(f"재시작 결과: {result}")
```

**실행 결과:**
```
--- 초안 생성 중 ---

중단 시점 결과: {'document': '문서 초안: LangGraph HITL 가이드', 'approved': False}
다음 실행될 노드: ('finalize',)
현재 문서: 문서 초안: LangGraph HITL 가이드

=== 사용자 승인 ===
--- 최종 문서 생성 중 ---
최종 결과: {'document': '문서 초안: LangGraph HITL 가이드 [최종본]', 'approved': True}
```

### 문제 2 솔루션: 조건부 검토 시스템

```python
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
import uuid

# 1. 상태 정의
class CodeReviewState(TypedDict):
    code: str
    risk_level: str
    review_decision: str
    status: str

# 2. 위험도 계산 함수
def calculate_risk(code: str) -> str:
    """코드 위험도 계산 (단순 예시)"""
    dangerous_patterns = ["eval", "exec", "os.system", "subprocess"]

    for pattern in dangerous_patterns:
        if pattern in code.lower():
            return "high"

    if len(code) > 1000:
        return "medium"

    return "low"

# 3. 노드 함수 정의
def analyze_code(state):
    """코드 분석"""
    print("--- 코드 분석 중 ---")
    code = state.get("code", "")
    risk_level = calculate_risk(code)

    print(f"위험도: {risk_level}")
    return {
        "code": code,
        "risk_level": risk_level,
        "status": "analyzed"
    }

def review_code(state):
    """코드 검토 (고위험 시 사용자 개입)"""
    print("--- 코드 검토 중 ---")

    risk_level = state.get("risk_level", "low")

    # 고위험 코드인 경우에만 사용자 검토 요청
    if risk_level == "high":
        print("⚠️ 고위험 코드 발견 - 사용자 검토 필요")

        decision = interrupt({
            "message": "고위험 코드가 발견되었습니다. 검토해주세요.",
            "code": state.get("code", ""),
            "risk_level": risk_level,
            "options": ["approve", "reject", "review_later"]
        })

        print(f"사용자 결정: {decision}")
        return {
            "review_decision": decision,
            "status": f"reviewed_{decision}"
        }
    else:
        # 저/중 위험도는 자동 승인
        print("✅ 자동 승인 (저위험)")
        return {
            "review_decision": "auto_approved",
            "status": "reviewed_auto_approved"
        }

def deploy_code(state):
    """코드 배포"""
    decision = state.get("review_decision", "")

    if decision in ["approve", "auto_approved"]:
        print("--- 배포 진행 중 ---")
        return {"status": "deployed"}
    elif decision == "reject":
        print("--- 배포 거부됨 ---")
        return {"status": "rejected"}
    else:  # review_later
        print("--- 나중에 검토 예정 ---")
        return {"status": "pending"}

# 4. 그래프 구성
builder = StateGraph(CodeReviewState)
builder.add_node("analyze", analyze_code)
builder.add_node("review", review_code)
builder.add_node("deploy", deploy_code)

builder.add_edge(START, "analyze")
builder.add_edge("analyze", "review")
builder.add_edge("review", "deploy")
builder.add_edge("deploy", END)

# 5. Checkpointer 설정
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# 6. 테스트 시나리오

# 시나리오 1: 고위험 코드 - 승인
print("=== 시나리오 1: 고위험 코드 (eval 포함) ===")
thread1 = {"configurable": {"thread_id": str(uuid.uuid4())}}

high_risk_code = """
def process_data(user_input):
    result = eval(user_input)  # 위험!
    return result
"""

result = graph.invoke({"code": high_risk_code}, thread1)
print(f"중단 시점: {result}")

# 사용자 승인
final_result = graph.invoke(Command(resume="approve"), thread1)
print(f"최종 결과: {final_result}\n")

# 시나리오 2: 고위험 코드 - 거부
print("=== 시나리오 2: 고위험 코드 - 거부 ===")
thread2 = {"configurable": {"thread_id": str(uuid.uuid4())}}

result = graph.invoke({"code": high_risk_code}, thread2)

# 사용자 거부
final_result = graph.invoke(Command(resume="reject"), thread2)
print(f"최종 결과: {final_result}\n")

# 시나리오 3: 저위험 코드 - 자동 승인
print("=== 시나리오 3: 저위험 코드 (자동 승인) ===")
thread3 = {"configurable": {"thread_id": str(uuid.uuid4())}}

low_risk_code = """
def add_numbers(a, b):
    return a + b
"""

final_result = graph.invoke({"code": low_risk_code}, thread3)
print(f"최종 결과 (자동 승인): {final_result}")
```

**실행 결과:**
```
=== 시나리오 1: 고위험 코드 (eval 포함) ===
--- 코드 분석 중 ---
위험도: high
--- 코드 검토 중 ---
⚠️ 고위험 코드 발견 - 사용자 검토 필요
중단 시점: {...}
사용자 결정: approve
--- 배포 진행 중 ---
최종 결과: {'code': '...', 'risk_level': 'high', 'review_decision': 'approve', 'status': 'deployed'}

=== 시나리오 2: 고위험 코드 - 거부 ===
사용자 결정: reject
--- 배포 거부됨 ---
최종 결과: {..., 'status': 'rejected'}

=== 시나리오 3: 저위험 코드 (자동 승인) ===
--- 코드 분석 중 ---
위험도: low
--- 코드 검토 중 ---
✅ 자동 승인 (저위험)
--- 배포 진행 중 ---
최종 결과: {..., 'review_decision': 'auto_approved', 'status': 'deployed'}
```

### 문제 3 솔루션: 다단계 승인 워크플로우

```python
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from datetime import datetime
import uuid

# 1. 상태 정의
class BudgetState(TypedDict):
    request: dict
    amount: float
    team_leader_approved: bool
    executive_approved: bool
    feedback: list
    status: str

# 2. 노드 함수 정의
def submit_request(state):
    """예산 요청서 제출"""
    print("--- 예산 요청서 제출 ---")
    request = state.get("request", {})
    amount = state.get("amount", 0)

    print(f"요청 금액: ${amount:,.2f}")
    print(f"요청 내용: {request.get('description', '')}")

    return {
        "request": request,
        "amount": amount,
        "team_leader_approved": False,
        "executive_approved": False,
        "feedback": [],
        "status": "submitted"
    }

def team_leader_review(state):
    """팀 리더 승인"""
    print("\n--- 팀 리더 검토 단계 ---")

    decision = interrupt({
        "message": "팀 리더 승인이 필요합니다",
        "request": state.get("request", {}),
        "amount": state.get("amount", 0),
        "options": ["approve", "reject", "request_changes"],
        "reviewer": "team_leader"
    })

    # 피드백 추가
    feedback_entry = {
        "reviewer": "team_leader",
        "decision": decision,
        "timestamp": datetime.now().isoformat()
    }

    current_feedback = state.get("feedback", [])
    current_feedback.append(feedback_entry)

    if decision == "approve":
        print("✅ 팀 리더 승인")
        return {
            "team_leader_approved": True,
            "feedback": current_feedback,
            "status": "team_leader_approved"
        }
    elif decision == "reject":
        print("❌ 팀 리더 거부")
        return {
            "team_leader_approved": False,
            "feedback": current_feedback,
            "status": "rejected_by_team_leader"
        }
    else:  # request_changes
        print("🔄 팀 리더 수정 요청")
        return {
            "team_leader_approved": False,
            "feedback": current_feedback,
            "status": "changes_requested_by_team_leader"
        }

def executive_review(state):
    """임원 승인 (고액인 경우)"""
    amount = state.get("amount", 0)

    # $10,000 이상인 경우에만 임원 승인 필요
    if amount >= 10000:
        print(f"\n--- 임원 검토 단계 (${amount:,.2f}) ---")

        decision = interrupt({
            "message": f"고액 예산(${amount:,.2f}) - 임원 승인이 필요합니다",
            "request": state.get("request", {}),
            "amount": amount,
            "team_leader_feedback": state.get("feedback", []),
            "options": ["approve", "reject", "request_changes"],
            "reviewer": "executive"
        })

        # 피드백 추가
        feedback_entry = {
            "reviewer": "executive",
            "decision": decision,
            "timestamp": datetime.now().isoformat()
        }

        current_feedback = state.get("feedback", [])
        current_feedback.append(feedback_entry)

        if decision == "approve":
            print("✅ 임원 승인")
            return {
                "executive_approved": True,
                "feedback": current_feedback,
                "status": "executive_approved"
            }
        elif decision == "reject":
            print("❌ 임원 거부")
            return {
                "executive_approved": False,
                "feedback": current_feedback,
                "status": "rejected_by_executive"
            }
        else:  # request_changes
            print("🔄 임원 수정 요청")
            return {
                "executive_approved": False,
                "feedback": current_feedback,
                "status": "changes_requested_by_executive"
            }
    else:
        # $10,000 미만은 임원 승인 불필요
        print(f"\n--- 임원 승인 불필요 (${amount:,.2f}) ---")
        return {
            "executive_approved": True,  # 자동 승인
            "status": "auto_approved_low_amount"
        }

def finalize_decision(state):
    """최종 결정"""
    print("\n--- 최종 결정 ---")

    team_leader_approved = state.get("team_leader_approved", False)
    executive_approved = state.get("executive_approved", False)
    status = state.get("status", "")

    if team_leader_approved and executive_approved:
        print("✅ 예산 승인 완료")
        return {"status": "approved"}
    else:
        print(f"❌ 예산 승인 거부 (상태: {status})")
        return {"status": f"final_{status}"}

# 3. 그래프 구성
builder = StateGraph(BudgetState)
builder.add_node("submit", submit_request)
builder.add_node("team_leader", team_leader_review)
builder.add_node("executive", executive_review)
builder.add_node("finalize", finalize_decision)

builder.add_edge(START, "submit")
builder.add_edge("submit", "team_leader")
builder.add_edge("team_leader", "executive")
builder.add_edge("executive", "finalize")
builder.add_edge("finalize", END)

# 4. Checkpointer 설정
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# 5. 테스트 시나리오

# 시나리오 1: 소액 예산 - 팀 리더만 승인
print("=" * 80)
print("=== 시나리오 1: 소액 예산 ($5,000) - 팀 리더 승인만 필요 ===")
print("=" * 80)

thread1 = {"configurable": {"thread_id": str(uuid.uuid4())}}

budget_request_1 = {
    "request": {
        "description": "개발 서버 업그레이드",
        "department": "Engineering"
    },
    "amount": 5000.0
}

# 초기 실행 - 팀 리더 승인에서 중단
result = graph.invoke(budget_request_1, thread1)

# 팀 리더 승인
print("\n>>> 팀 리더가 승인을 선택합니다")
result = graph.invoke(Command(resume="approve"), thread1)

# executive는 자동 승인되고 finalize까지 실행됨
print(f"\n최종 결과: {result['status']}")
print(f"피드백 이력: {result['feedback']}")

# 시나리오 2: 고액 예산 - 팀 리더 및 임원 승인 필요
print("\n" + "=" * 80)
print("=== 시나리오 2: 고액 예산 ($15,000) - 두 단계 승인 필요 ===")
print("=" * 80)

thread2 = {"configurable": {"thread_id": str(uuid.uuid4())}}

budget_request_2 = {
    "request": {
        "description": "새로운 AI 인프라 구축",
        "department": "AI Research"
    },
    "amount": 15000.0
}

# 초기 실행 - 팀 리더 승인에서 중단
result = graph.invoke(budget_request_2, thread2)

# 팀 리더 승인
print("\n>>> 팀 리더가 승인을 선택합니다")
result = graph.invoke(Command(resume="approve"), thread2)

# 임원 승인에서 중단됨
state = graph.get_state(thread2)
print(f"\n현재 상태: {state.values['status']}")
print(f"다음 노드: {state.next}")

# 임원 승인
print("\n>>> 임원이 승인을 선택합니다")
final_result = graph.invoke(Command(resume="approve"), thread2)

print(f"\n최종 결과: {final_result['status']}")
print(f"피드백 이력:")
for feedback in final_result['feedback']:
    print(f"  - {feedback['reviewer']}: {feedback['decision']} at {feedback['timestamp']}")

# 시나리오 3: 고액 예산 - 임원 거부
print("\n" + "=" * 80)
print("=== 시나리오 3: 고액 예산 - 임원 거부 ===")
print("=" * 80)

thread3 = {"configurable": {"thread_id": str(uuid.uuid4())}}

# 초기 실행 및 팀 리더 승인
result = graph.invoke(budget_request_2, thread3)
result = graph.invoke(Command(resume="approve"), thread3)

# 임원 거부
print("\n>>> 임원이 거부를 선택합니다")
final_result = graph.invoke(Command(resume="reject"), thread3)

print(f"\n최종 결과: {final_result['status']}")
print(f"팀 리더 승인: {final_result['team_leader_approved']}")
print(f"임원 승인: {final_result['executive_approved']}")
```

**실행 결과:**
```
================================================================================
=== 시나리오 1: 소액 예산 ($5,000) - 팀 리더 승인만 필요 ===
================================================================================
--- 예산 요청서 제출 ---
요청 금액: $5,000.00
요청 내용: 개발 서버 업그레이드

--- 팀 리더 검토 단계 ---

>>> 팀 리더가 승인을 선택합니다
✅ 팀 리더 승인

--- 임원 승인 불필요 ($5,000.00) ---

--- 최종 결정 ---
✅ 예산 승인 완료

최종 결과: approved
피드백 이력: [{'reviewer': 'team_leader', 'decision': 'approve', 'timestamp': '2025-01-15T...'}]

================================================================================
=== 시나리오 2: 고액 예산 ($15,000) - 두 단계 승인 필요 ===
================================================================================
--- 예산 요청서 제출 ---
요청 금액: $15,000.00
요청 내용: 새로운 AI 인프라 구축

--- 팀 리더 검토 단계 ---

>>> 팀 리더가 승인을 선택합니다
✅ 팀 리더 승인

--- 임원 검토 단계 ($15,000.00) ---

현재 상태: team_leader_approved
다음 노드: ('executive',)

>>> 임원이 승인을 선택합니다
✅ 임원 승인

--- 최종 결정 ---
✅ 예산 승인 완료

최종 결과: approved
피드백 이력:
  - team_leader: approve at 2025-01-15T...
  - executive: approve at 2025-01-15T...
```

## 🚀 실무 활용 예시

### 예시 1: 콘텐츠 생성 및 검토 시스템

소셜 미디어 콘텐츠를 자동 생성하고 게시 전 사용자 검토를 받는 시스템입니다.

```python
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langchain_openai import ChatOpenAI
import uuid

# 1. 상태 정의
class ContentState(TypedDict):
    topic: str
    content: str
    image_prompt: str
    review_feedback: str
    status: str

# 2. LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 3. 노드 함수 정의
def generate_content(state):
    """콘텐츠 생성"""
    print("--- 콘텐츠 생성 중 ---")
    topic = state.get("topic", "")

    # LLM으로 콘텐츠 생성
    prompt = f"다음 주제로 소셜 미디어 게시글을 작성해주세요 (200자 이내): {topic}"
    response = llm.invoke(prompt)
    content = response.content

    # 이미지 프롬프트도 생성
    image_prompt_text = f"이 게시글에 어울리는 이미지 생성 프롬프트를 작성해주세요: {content}"
    image_response = llm.invoke(image_prompt_text)
    image_prompt = image_response.content

    print(f"생성된 콘텐츠: {content[:100]}...")

    return {
        "topic": topic,
        "content": content,
        "image_prompt": image_prompt,
        "status": "generated"
    }

def review_content(state):
    """사용자 검토"""
    print("\n--- 사용자 검토 단계 ---")

    # 사용자에게 검토 요청
    review_result = interrupt({
        "message": "생성된 콘텐츠를 검토해주세요",
        "content": state.get("content", ""),
        "image_prompt": state.get("image_prompt", ""),
        "options": {
            "approve": "승인하고 게시",
            "edit": "수정 요청",
            "regenerate": "다시 생성",
            "cancel": "취소"
        }
    })

    return {
        "review_feedback": review_result,
        "status": f"reviewed_{review_result}"
    }

def handle_feedback(state):
    """피드백 처리"""
    feedback = state.get("review_feedback", "")

    if feedback == "approve":
        print("✅ 콘텐츠 승인 - 게시 진행")
        return {"status": "approved"}

    elif feedback == "edit":
        print("✏️ 수정 요청 - 피드백 입력 받기")

        # 수정 사항 입력 요청
        edit_feedback = interrupt({
            "message": "어떻게 수정할까요?",
            "current_content": state.get("content", ""),
            "placeholder": "수정할 내용을 입력해주세요"
        })

        # 피드백 반영하여 재생성
        print(f"피드백 반영 중: {edit_feedback}")

        prompt = f"""다음 콘텐츠를 사용자 피드백에 따라 수정해주세요:

원본 콘텐츠: {state.get('content', '')}
수정 요청: {edit_feedback}
"""
        response = llm.invoke(prompt)
        revised_content = response.content

        return {
            "content": revised_content,
            "status": "revised"
        }

    elif feedback == "regenerate":
        print("🔄 다시 생성 요청")
        return {"status": "regenerate"}

    else:  # cancel
        print("❌ 취소")
        return {"status": "cancelled"}

def publish_content(state):
    """콘텐츠 게시"""
    print("\n--- 콘텐츠 게시 중 ---")

    # 실제로는 API 호출 등으로 게시
    print(f"게시 내용: {state.get('content', '')}")
    print(f"이미지 프롬프트: {state.get('image_prompt', '')}")

    return {"status": "published"}

# 4. 그래프 구성
builder = StateGraph(ContentState)
builder.add_node("generate", generate_content)
builder.add_node("review", review_content)
builder.add_node("handle_feedback", handle_feedback)
builder.add_node("publish", publish_content)

builder.add_edge(START, "generate")
builder.add_edge("generate", "review")
builder.add_edge("review", "handle_feedback")

# 조건부 엣지
def should_continue(state):
    status = state.get("status", "")
    if status == "approved":
        return "publish"
    elif status == "regenerate":
        return "generate"
    elif status == "revised":
        return "review"
    else:
        return END

builder.add_conditional_edges(
    "handle_feedback",
    should_continue,
    {
        "publish": "publish",
        "generate": "generate",
        "review": "review",
        END: END
    }
)

builder.add_edge("publish", END)

# 5. 컴파일
checkpointer = InMemorySaver()
content_graph = builder.compile(checkpointer=checkpointer)

# 6. 실행 예시
thread = {"configurable": {"thread_id": str(uuid.uuid4())}}

# 초기 실행
result = content_graph.invoke({
    "topic": "LangGraph를 활용한 AI 에이전트 개발",
    "content": "",
    "image_prompt": "",
    "review_feedback": "",
    "status": ""
}, thread)

# 사용자 검토 단계에서 중단됨
print(f"\n중단 시점 상태: {result['status']}")

# 사용자가 승인
final_result = content_graph.invoke(Command(resume="approve"), thread)
print(f"\n최종 상태: {final_result['status']}")
```

**활용 효과:**
- 자동 콘텐츠 생성으로 시간 절약
- 사용자 검토로 품질 보장
- 수정 요청으로 정확성 향상
- 반복 가능한 워크플로우

### 예시 2: 고객 지원 티켓 처리 시스템

고객 문의를 자동 분류하고 중요도에 따라 사람의 검토를 받는 시스템입니다.

```python
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langchain_openai import ChatOpenAI
import uuid

# 1. 상태 정의
class TicketState(TypedDict):
    ticket_id: str
    customer_message: str
    category: str
    priority: str
    auto_response: str
    agent_decision: str
    final_response: str
    status: str

# 2. LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# 3. 노드 함수
def classify_ticket(state):
    """티켓 분류 및 우선순위 결정"""
    print("--- 티켓 분류 중 ---")

    message = state.get("customer_message", "")

    # LLM으로 분류
    classification_prompt = f"""다음 고객 메시지를 분석하여 JSON 형식으로 분류해주세요:

고객 메시지: {message}

다음 형식으로 답변:
{{
    "category": "기술지원|결제문의|계정문제|기타",
    "priority": "high|medium|low",
    "requires_human": true|false,
    "reason": "이유 설명"
}}
"""

    response = llm.invoke(classification_prompt)
    # 실제로는 JSON 파싱 필요

    # 단순 예시
    category = "기술지원"
    priority = "high" if "긴급" in message or "오류" in message else "medium"

    print(f"분류: {category}, 우선순위: {priority}")

    return {
        "category": category,
        "priority": priority,
        "status": "classified"
    }

def generate_response(state):
    """자동 응답 생성"""
    print("--- 자동 응답 생성 중 ---")

    message = state.get("customer_message", "")
    category = state.get("category", "")

    response_prompt = f"""다음 고객 문의에 대한 응답을 작성해주세요:

카테고리: {category}
고객 메시지: {message}

친절하고 전문적인 톤으로 작성해주세요.
"""

    response = llm.invoke(response_prompt)
    auto_response = response.content

    print(f"자동 응답: {auto_response[:100]}...")

    return {
        "auto_response": auto_response,
        "status": "response_generated"
    }

def human_review(state):
    """사람 검토 (고우선순위만)"""
    priority = state.get("priority", "")

    # 고우선순위인 경우에만 검토 요청
    if priority == "high":
        print("\n⚠️ 고우선순위 티켓 - 상담원 검토 필요")

        agent_decision = interrupt({
            "message": "고우선순위 티켓입니다. 자동 응답을 검토해주세요.",
            "customer_message": state.get("customer_message", ""),
            "auto_response": state.get("auto_response", ""),
            "category": state.get("category", ""),
            "options": {
                "approve": "자동 응답 승인",
                "modify": "응답 수정",
                "escalate": "상급자 에스컬레이션"
            }
        })

        return {
            "agent_decision": agent_decision,
            "status": f"reviewed_{agent_decision}"
        }
    else:
        # 중/저우선순위는 자동 승인
        print("✅ 자동 승인 (중/저우선순위)")
        return {
            "agent_decision": "auto_approved",
            "final_response": state.get("auto_response", ""),
            "status": "auto_approved"
        }

def finalize_response(state):
    """최종 응답 처리"""
    decision = state.get("agent_decision", "")

    if decision in ["approve", "auto_approved"]:
        print("--- 응답 전송 ---")
        return {
            "final_response": state.get("auto_response", ""),
            "status": "sent"
        }

    elif decision == "modify":
        # 수정된 응답 입력 받기
        modified_response = interrupt({
            "message": "수정된 응답을 입력해주세요",
            "original_response": state.get("auto_response", ""),
            "placeholder": "수정된 응답"
        })

        print("--- 수정된 응답 전송 ---")
        return {
            "final_response": modified_response,
            "status": "sent_modified"
        }

    else:  # escalate
        print("--- 상급자 에스컬레이션 ---")
        return {
            "status": "escalated"
        }

# 4. 그래프 구성
builder = StateGraph(TicketState)
builder.add_node("classify", classify_ticket)
builder.add_node("generate", generate_response)
builder.add_node("review", human_review)
builder.add_node("finalize", finalize_response)

builder.add_edge(START, "classify")
builder.add_edge("classify", "generate")
builder.add_edge("generate", "review")
builder.add_edge("review", "finalize")
builder.add_edge("finalize", END)

# 5. 컴파일
checkpointer = InMemorySaver()
ticket_graph = builder.compile(checkpointer=checkpointer)

# 6. 테스트 시나리오
print("=" * 80)
print("=== 시나리오 1: 고우선순위 티켓 (긴급 오류) ===")
print("=" * 80)

thread1 = {"configurable": {"thread_id": str(uuid.uuid4())}}

ticket1 = {
    "ticket_id": "T001",
    "customer_message": "긴급! 로그인이 안 됩니다. 오류 메시지가 계속 뜹니다.",
    "category": "",
    "priority": "",
    "auto_response": "",
    "agent_decision": "",
    "final_response": "",
    "status": ""
}

# 실행 - 상담원 검토에서 중단
result = ticket_graph.invoke(ticket1, thread1)

# 상담원이 자동 응답 승인
print("\n>>> 상담원이 자동 응답을 승인합니다")
final_result = ticket_graph.invoke(Command(resume="approve"), thread1)

print(f"\n최종 상태: {final_result['status']}")
print(f"전송된 응답: {final_result.get('final_response', '')[:100]}...")

# 시나리오 2: 저우선순위 티켓 (자동 처리)
print("\n" + "=" * 80)
print("=== 시나리오 2: 저우선순위 티켓 (자동 처리) ===")
print("=" * 80)

thread2 = {"configurable": {"thread_id": str(uuid.uuid4())}}

ticket2 = {
    "ticket_id": "T002",
    "customer_message": "비밀번호 재설정 방법을 알려주세요.",
    "category": "",
    "priority": "",
    "auto_response": "",
    "agent_decision": "",
    "final_response": "",
    "status": ""
}

# 실행 - 자동으로 끝까지 처리됨
final_result = ticket_graph.invoke(ticket2, thread2)

print(f"\n최종 상태: {final_result['status']}")
print(f"자동 전송된 응답: {final_result.get('final_response', '')[:100]}...")
```

**활용 효과:**
- 저우선순위 티켓 자동 처리로 효율성 향상
- 고우선순위 티켓은 사람 검토로 품질 보장
- 상담원 업무 부담 감소
- 고객 대응 시간 단축

### 예시 3: 데이터 분석 파이프라인 검증

데이터 분석 결과를 자동 생성하고 배포 전 데이터 사이언티스트가 검증하는 시스템입니다.

```python
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
import uuid

# 1. 상태 정의
class AnalysisPipelineState(TypedDict):
    dataset_name: str
    analysis_results: dict
    anomalies_detected: list
    validation_status: str
    scientist_feedback: str
    status: str

# 2. 노드 함수
def run_analysis(state):
    """데이터 분석 실행"""
    print("--- 데이터 분석 실행 중 ---")

    # 실제로는 복잡한 분석 로직
    results = {
        "total_records": 10000,
        "mean_value": 45.7,
        "std_dev": 12.3,
        "outliers_count": 50,
        "confidence_interval": (43.2, 48.2)
    }

    print(f"분석 완료: {results}")

    return {
        "analysis_results": results,
        "status": "analyzed"
    }

def detect_anomalies(state):
    """이상치 탐지"""
    print("--- 이상치 탐지 중 ---")

    # 이상치 탐지 로직
    anomalies = [
        {"record_id": 1234, "value": 150, "reason": "값이 3 표준편차 초과"},
        {"record_id": 5678, "value": -50, "reason": "음수 값 (불가능한 값)"}
    ]

    print(f"이상치 {len(anomalies)}개 발견")

    return {
        "anomalies_detected": anomalies,
        "status": "anomalies_detected"
    }

def validate_results(state):
    """결과 검증 (이상치가 있으면 사람 검토)"""
    anomalies = state.get("anomalies_detected", [])

    if len(anomalies) > 0:
        print(f"\n⚠️ 이상치 {len(anomalies)}개 발견 - 데이터 사이언티스트 검증 필요")

        validation_decision = interrupt({
            "message": "이상치가 발견되었습니다. 검증해주세요.",
            "analysis_results": state.get("analysis_results", {}),
            "anomalies": anomalies,
            "options": {
                "approve": "이상치 정상 (분석 진행)",
                "clean": "이상치 제거 후 재분석",
                "investigate": "추가 조사 필요"
            }
        })

        return {
            "validation_status": validation_decision,
            "status": f"validated_{validation_decision}"
        }
    else:
        print("✅ 이상치 없음 - 자동 승인")
        return {
            "validation_status": "auto_approved",
            "status": "validated_auto"
        }

def handle_validation(state):
    """검증 결과 처리"""
    validation = state.get("validation_status", "")

    if validation in ["approve", "auto_approved"]:
        print("--- 분석 결과 배포 ---")
        return {"status": "deployed"}

    elif validation == "clean":
        print("--- 이상치 제거 후 재분석 ---")
        # 실제로는 데이터 클리닝 수행
        return {"status": "reanalyze"}

    else:  # investigate
        print("--- 추가 조사 필요 ---")

        investigation_notes = interrupt({
            "message": "조사 내용을 입력해주세요",
            "anomalies": state.get("anomalies_detected", []),
            "placeholder": "조사 내용 및 발견사항"
        })

        return {
            "scientist_feedback": investigation_notes,
            "status": "investigated"
        }

# 3. 그래프 구성
builder = StateGraph(AnalysisPipelineState)
builder.add_node("analyze", run_analysis)
builder.add_node("detect", detect_anomalies)
builder.add_node("validate", validate_results)
builder.add_node("handle", handle_validation)

builder.add_edge(START, "analyze")
builder.add_edge("analyze", "detect")
builder.add_edge("detect", "validate")
builder.add_edge("validate", "handle")

def should_continue(state):
    status = state.get("status", "")
    if status == "reanalyze":
        return "analyze"
    else:
        return END

builder.add_conditional_edges(
    "handle",
    should_continue,
    {
        "analyze": "analyze",
        END: END
    }
)

# 4. 컴파일
checkpointer = InMemorySaver()
pipeline_graph = builder.compile(checkpointer=checkpointer)

# 5. 실행
thread = {"configurable": {"thread_id": str(uuid.uuid4())}}

result = pipeline_graph.invoke({
    "dataset_name": "customer_data_2024",
    "analysis_results": {},
    "anomalies_detected": [],
    "validation_status": "",
    "scientist_feedback": "",
    "status": ""
}, thread)

# 데이터 사이언티스트 검증 단계
print("\n>>> 데이터 사이언티스트가 이상치를 검토하고 승인합니다")
final_result = pipeline_graph.invoke(Command(resume="approve"), thread)

print(f"\n최종 상태: {final_result['status']}")
```

**활용 효과:**
- 자동화된 분석 파이프라인으로 속도 향상
- 이상치 발견 시 전문가 검증으로 정확성 보장
- 잘못된 분석 결과 배포 방지
- 데이터 품질 관리 강화

## 📖 참고 자료

### 공식 문서
- [LangGraph HITL 공식 가이드](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)
- [Checkpointer 개념](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [interrupt 함수 API](https://langchain-ai.github.io/langgraph/reference/types/#langgraph.types.interrupt)
- [Command 객체 레퍼런스](https://langchain-ai.github.io/langgraph/reference/types/#langgraph.types.Command)

### 관련 개념
- **Checkpointer 종류**: InMemorySaver, SqliteSaver, PostgresSaver 비교
- **State Management**: LangGraph 상태 관리 패턴
- **Conditional Edges**: 조건부 분기와 워크플로우 제어
- **Error Handling**: HITL에서의 오류 처리 전략

### 추가 학습 자료
- LangGraph 튜토리얼: Multi-Agent Systems
- Human-in-the-Loop Design Patterns
- Production Deployment with Checkpointers
- Advanced State Management Techniques

---

**다음 단계**: Part 2에서는 웹 검색 기반 리서치 시스템을 구현하며 HITL을 실제 복잡한 워크플로우에 적용하는 방법을 학습합니다.
