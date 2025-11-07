# LangGraph 서브그래프 (Sub-graph)

## 📚 학습 목표

이 가이드를 완료하면 다음을 할 수 있습니다:

1. **서브그래프 개념 이해**: 서브그래프의 정의와 모듈화를 통한 장점을 파악한다
2. **서브그래프 구현**: 독립적으로 동작하는 서브그래프를 설계하고 구현한다
3. **상태 공유 패턴**: 부모 그래프와 서브그래프 간의 상태 공유 방법을 익힌다
4. **상태 변환 패턴**: 서로 다른 상태 키를 사용하는 그래프 간 상태 변환 방법을 구현한다
5. **재사용 가능한 컴포넌트**: 다양한 시스템에서 재사용할 수 있는 서브그래프를 설계한다
6. **복잡한 시스템 구축**: 여러 서브그래프를 조합하여 실무 시스템을 구현한다
7. **실무 적용**: FAQ 챗봇과 같은 실제 서비스에 서브그래프 패턴을 적용한다

## 🔑 핵심 개념

### 서브그래프 (Sub-graph)란?

**서브그래프**는 독립적으로 작동하는 하위 그래프 구조로, 더 큰 **부모 그래프** 내에서 하나의 노드로 동작하여 모듈화를 실현합니다.

```
┌────────────────────────────────────────┐
│        부모 그래프 (Parent Graph)        │
│                                        │
│  ┌─────┐    ┌──────────────┐   ┌────┐ │
│  │START│ → │  서브그래프    │ → │END │ │
│  └─────┘    │  (Subgraph)   │   └────┘ │
│             │                │         │
│             │  ┌───┐  ┌───┐ │         │
│             │  │ A │→│ B │ │         │
│             │  └───┘  └───┘ │         │
│             └──────────────┘         │
└────────────────────────────────────────┘
```

### 서브그래프의 장점

**1. 모듈화 (Modularity)**
- 코드 구조가 명확해지고 유지보수가 간편함
- 각 서브그래프가 특정 기능을 담당하여 책임 분리

**2. 재사용성 (Reusability)**
- 독립적인 구조로 다양한 시스템에서 재사용 가능
- 한 번 구현한 서브그래프를 여러 프로젝트에 활용

**3. 확장성 (Scalability)**
- 새로운 기능을 서브그래프 형태로 쉽게 추가
- 기존 시스템을 수정하지 않고 기능 확장

**4. 독립적 테스트 (Independent Testing)**
- 각 서브그래프를 개별적으로 테스트 가능
- 문제 발생 시 격리된 범위에서 디버깅

**5. 협업 효율성 (Collaboration)**
- 팀원들이 서로 다른 서브그래프를 독립적으로 개발
- 인터페이스만 맞추면 통합 가능

### 서브그래프 구현 패턴

#### 패턴 1: 상태 공유 (State Sharing)

부모 그래프와 서브그래프가 **동일한 상태 키**를 사용하는 방식입니다.

```python
# 공통 상태 정의
class SharedState(TypedDict):
    num1: int
    num2: int
    result: int

# 서브그래프와 부모 그래프가 같은 상태 사용
subgraph = StateGraph(SharedState)
parent_graph = StateGraph(SharedState)

# 컴파일된 서브그래프를 부모 그래프 노드로 추가
parent_graph.add_node("calculate", subgraph.compile())
```

**장점:**
- 구현이 간단하고 직관적
- 상태 변환 로직 불필요
- 성능상 오버헤드 없음

**단점:**
- 서브그래프와 부모 그래프가 강하게 결합됨
- 재사용성이 제한될 수 있음

#### 패턴 2: 상태 변환 (State Transformation)

부모 그래프와 서브그래프가 **서로 다른 상태 키**를 사용하는 방식입니다.

```python
# 서브그래프 전용 상태
class SubgraphState(TypedDict):
    a: int
    b: int

# 부모 그래프 상태
class ParentState(TypedDict):
    num1: int
    num2: int
    result: int

# 상태 변환 함수
def call_subgraph(state: ParentState):
    # 부모 상태 → 서브그래프 상태 변환
    subgraph_input = {"a": state["num1"], "b": state["num2"]}

    # 서브그래프 실행
    subgraph_output = subgraph.invoke(subgraph_input)

    # 서브그래프 상태 → 부모 상태 변환
    return {"result": subgraph_output["sum"]}
```

**장점:**
- 서브그래프의 독립성 보장
- 높은 재사용성
- 명확한 인터페이스 정의

**단점:**
- 변환 로직 추가 필요
- 약간의 성능 오버헤드

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
from pprint import pprint

# 환경 변수 로드
load_dotenv()

# API 키 확인
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

print("환경 설정 완료!")
```

## 💻 단계별 구현

### Step 1: 간단한 서브그래프 - 덧셈과 곱셈 연산

가장 기본적인 서브그래프를 구현하여 개념을 이해합니다.

#### 1-1. 서브그래프 정의 (상태 공유 방식)

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from IPython.display import Image, display

# 1. 서브그래프 상태 정의
class SubgraphState(TypedDict):
    num1: int           # 첫 번째 숫자
    num2: int           # 두 번째 숫자
    sum_val: int        # 더하기 결과
    product_val: int    # 곱하기 결과

# 2. add 노드: 두 숫자를 더하는 함수
def add_node(state: SubgraphState):
    """두 숫자를 더한 결과를 반환"""
    result = state["num1"] + state["num2"]
    print(f"Add: {state['num1']} + {state['num2']} = {result}")
    return {"sum_val": result}

# 3. multiply 노드: 두 숫자를 곱하는 함수
def multiply_node(state: SubgraphState):
    """두 숫자를 곱한 결과를 반환"""
    result = state["num1"] * state["num2"]
    print(f"Multiply: {state['num1']} × {state['num2']} = {result}")
    return {"product_val": result}

# 4. 서브그래프 빌더 생성
subgraph_builder = StateGraph(SubgraphState)

# 5. 노드 추가
subgraph_builder.add_node("add", add_node)
subgraph_builder.add_node("multiply", multiply_node)

# 6. 엣지 연결
subgraph_builder.add_edge(START, "add")
subgraph_builder.add_edge("add", "multiply")
subgraph_builder.add_edge("multiply", END)  # 서브그래프의 끝

# 7. 서브그래프 컴파일
subgraph = subgraph_builder.compile()

# 8. 서브그래프 시각화
display(Image(subgraph.get_graph().draw_mermaid_png()))
```

**실행 결과:**
```
서브그래프 구조:
START → add → multiply → END
```

#### 1-2. 부모 그래프 정의

```python
# 1. 부모 그래프 상태 정의 (서브그래프와 동일한 키 사용)
class ParentState(TypedDict):
    num1: int
    num2: int
    sum_val: int
    product_val: int

# 2. 부모 그래프 빌더 생성
parent_builder = StateGraph(ParentState)

# 3. 컴파일된 서브그래프를 'calculate' 노드로 추가
parent_builder.add_node("calculate", subgraph)

# 4. 엣지 연결
parent_builder.add_edge(START, "calculate")
parent_builder.add_edge("calculate", END)

# 5. 부모 그래프 컴파일
parent_graph = parent_builder.compile()

# 6. 부모 그래프 시각화 (xray=True: 서브그래프 내부도 표시)
display(Image(parent_graph.get_graph(xray=True).draw_mermaid_png()))
```

**시각화 결과:**
```
부모 그래프:
START → calculate (서브그래프) → END

서브그래프 내부 (xray=True):
START → add → multiply → END
```

#### 1-3. 그래프 실행

```python
# 부모 그래프 실행
print("=== 그래프 실행 ===")
for result in parent_graph.stream({"num1": 5, "num2": 3}):
    print(result)
```

**실행 결과:**
```
=== 그래프 실행 ===
Add: 5 + 3 = 8
Multiply: 5 × 3 = 15
{'calculate': {'num1': 5, 'num2': 3, 'sum_val': 8, 'product_val': 15}}
```

**결과 분석:**
- `num1=5`, `num2=3` 입력
- `add` 노드에서 `sum_val=8` 계산
- `multiply` 노드에서 `product_val=15` 계산
- 모든 결과가 부모 그래프로 반환됨

### Step 2: 상태 변환을 사용하는 서브그래프

부모 그래프와 서브그래프가 서로 다른 상태 키를 사용하는 경우입니다.

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# 1. 서브그래프 상태 정의 (부모와 다른 키 사용)
class SubgraphState(TypedDict):
    a: int  # num1 대신 a 사용
    b: int  # num2 대신 b 사용
    sum_val: int
    product_val: int

# 2. add, multiply 노드 (상태 키만 변경)
def add_node(state: SubgraphState):
    result = state["a"] + state["b"]
    print(f"Add: {state['a']} + {state['b']} = {result}")
    return {"sum_val": result}

def multiply_node(state: SubgraphState):
    result = state["a"] * state["b"]
    print(f"Multiply: {state['a']} × {state['b']} = {result}")
    return {"product_val": result}

# 3. 서브그래프 구성 및 컴파일
subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node("add", add_node)
subgraph_builder.add_node("multiply", multiply_node)
subgraph_builder.add_edge(START, "add")
subgraph_builder.add_edge("add", "multiply")
subgraph_builder.add_edge("multiply", END)
subgraph = subgraph_builder.compile()

# 4. 부모 그래프 상태 정의 (원래 키 사용)
class ParentState(TypedDict):
    num1: int
    num2: int
    sum_val: int
    product_val: int

# 5. 서브그래프 호출 및 상태 변환 함수
def call_subgraph(state: ParentState):
    """
    부모 그래프와 서브그래프 간 상태 변환
    """
    print("\n--- 상태 변환 및 서브그래프 호출 ---")

    # 부모 그래프 상태 → 서브그래프 상태 변환
    subgraph_input = {
        "a": state["num1"],  # num1 → a
        "b": state["num2"]   # num2 → b
    }

    print(f"입력 변환: num1={state['num1']}, num2={state['num2']} → a={subgraph_input['a']}, b={subgraph_input['b']}")

    # 서브그래프 실행
    subgraph_output = subgraph.invoke(subgraph_input)

    # 서브그래프 상태 → 부모 그래프 상태 변환
    result = {
        "sum_val": subgraph_output["sum_val"],
        "product_val": subgraph_output["product_val"]
    }

    print(f"출력 변환: sum_val={result['sum_val']}, product_val={result['product_val']}")

    return result

# 6. 부모 그래프 빌더 생성
parent_builder = StateGraph(ParentState)

# 7. 서브그래프 호출 함수를 노드로 추가
parent_builder.add_node("calculate", call_subgraph)

# 8. 엣지 연결
parent_builder.add_edge(START, "calculate")
parent_builder.add_edge("calculate", END)

# 9. 부모 그래프 컴파일
parent_graph = parent_builder.compile()

# 10. 실행
print("=== 상태 변환 방식 그래프 실행 ===")
for result in parent_graph.stream({"num1": 5, "num2": 3}):
    print(result)
```

**실행 결과:**
```
=== 상태 변환 방식 그래프 실행 ===

--- 상태 변환 및 서브그래프 호출 ---
입력 변환: num1=5, num2=3 → a=5, b=3
Add: 5 + 3 = 8
Multiply: 5 × 3 = 15
출력 변환: sum_val=8, product_val=15
{'calculate': {'sum_val': 8, 'product_val': 15}}
```

**상태 변환 흐름:**
```
부모 그래프 상태: {num1: 5, num2: 3}
        ↓
   상태 변환 (입력)
        ↓
서브그래프 상태: {a: 5, b: 3}
        ↓
   서브그래프 실행
        ↓
서브그래프 출력: {sum_val: 8, product_val: 15}
        ↓
   상태 변환 (출력)
        ↓
부모 그래프 상태: {sum_val: 8, product_val: 15}
```

### Step 3: 실무 예제 - FAQ 챗봇 시스템

고객 서비스 시스템에서 FAQ 처리를 서브그래프로 구현합니다.

#### 3-1. 기본 구조 설정

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import HumanMessage, AIMessage
from IPython.display import Image, display

# 1. 공통 상태 정의 - 상속을 위한 베이스 클래스
class CustomerServiceState(MessagesState):
    """고객 서비스 시스템의 기본 상태"""
    customer_id: str              # 고객 ID
    issue_category: str           # 문의 카테고리
    satisfaction_score: float     # 만족도 점수

# 2. FAQ 상태는 CustomerServiceState를 상속
class FAQState(CustomerServiceState):
    """FAQ 서브그래프 전용 상태"""
    faq_matched: bool            # FAQ 매칭 여부
    faq_answer: str              # FAQ 답변 내용
```

**상태 상속 구조:**
```
MessagesState (LangGraph 기본)
    ↓
CustomerServiceState (고객 서비스 공통)
    ├─ messages: list
    ├─ customer_id: str
    ├─ issue_category: str
    └─ satisfaction_score: float
        ↓
FAQState (FAQ 전용)
    ├─ (CustomerServiceState 모든 필드 상속)
    ├─ faq_matched: bool
    └─ faq_answer: str
```

#### 3-2. FAQ 처리 서브그래프 구현

```python
def check_faq(state: FAQState):
    """
    FAQ 매칭 확인
    - 키워드 기반 매칭 (실무에서는 벡터 DB 사용)
    - 고객별 맞춤 FAQ 검색 가능
    """
    last_message = state["messages"][-1]
    customer_id = state["customer_id"]

    print(f"Checking FAQ for customer: {customer_id}")

    # 간단한 키워드 매칭 (실무에서는 벡터 DB 사용)
    if "운영시간" in last_message.content:
        return {
            "faq_matched": True,
            "faq_answer": f"안녕하세요 {customer_id}님, 저희 매장은 평일 9시부터 18시까지 운영합니다."
        }
    elif "환불" in last_message.content:
        return {
            "faq_matched": True,
            "faq_answer": f"{customer_id}님, 구매일로부터 7일 이내 환불 가능합니다."
        }
    elif "배송" in last_message.content:
        return {
            "faq_matched": True,
            "faq_answer": f"{customer_id}님, 평균 2-3일 소요되며 주문 후 배송 조회가 가능합니다."
        }

    return {"faq_matched": False}

def respond_faq(state: FAQState):
    """FAQ 응답 생성"""
    print(f"Responding with FAQ: {state['faq_answer'][:50]}...")
    return {
        "messages": [AIMessage(content=state["faq_answer"])]
    }

# FAQ 서브그래프 구성
faq_graph = StateGraph(FAQState)
faq_graph.add_node("check_faq", check_faq)
faq_graph.add_node("respond_faq", respond_faq)

# FAQ 그래프 로직 설정
def should_respond_faq(state: FAQState):
    """FAQ 매칭 여부에 따른 분기"""
    if state["faq_matched"]:
        return "respond_faq"
    return "end"

faq_graph.add_edge(START, "check_faq")
faq_graph.add_conditional_edges(
    "check_faq",
    should_respond_faq,
    {
        "respond_faq": "respond_faq",
        "end": END
    }
)
faq_graph.add_edge("respond_faq", END)

# FAQ 서브그래프 컴파일
faq_subgraph = faq_graph.compile()

# FAQ 그래프 시각화
display(Image(faq_subgraph.get_graph().draw_mermaid_png()))
```

**FAQ 서브그래프 흐름:**
```
START
  ↓
check_faq (FAQ 매칭 확인)
  ↓
[조건부 분기]
  ├─ faq_matched = True  → respond_faq → END
  └─ faq_matched = False → END
```

#### 3-3. 메인 고객 서비스 그래프 구현

```python
from langchain_openai import ChatOpenAI

# 1. 문의 유형 분류 노드
def route_inquiry(state: CustomerServiceState):
    """OpenAI를 활용한 문의 유형 자동 분류"""
    model = ChatOpenAI(model="gpt-4o-mini")
    last_message = state["messages"][-1]

    response = model.invoke([
        HumanMessage(content=f"""
        다음 고객 문의의 카테고리를 분류해주세요:
        {last_message.content}

        [카테고리 선택]:
        - technical: 기술적 문제
        - billing: 결제 문제
        - general: 일반 문의
        """)
    ])

    print(f"문의 분류 결과: {response.content}")
    return {"issue_category": response.content}

# 2. 기술 문제 처리 노드
def handle_technical(state: CustomerServiceState):
    """기술 지원팀으로 에스컬레이션"""
    print("---Handling technical issue---")
    return {
        "messages": [AIMessage(
            content="기술 지원팀에 문의가 전달되었습니다. 곧 전문가가 연락드릴 예정입니다."
        )],
        "satisfaction_score": 0.8
    }

# 3. 결제 문제 처리 노드
def handle_billing(state: CustomerServiceState):
    """결제 팀으로 에스컬레이션"""
    print("---Handling billing issue---")
    return {
        "messages": [AIMessage(
            content="결제 팀에서 확인 후 24시간 이내에 연락드리겠습니다."
        )],
        "satisfaction_score": 0.7
    }

# 4. 메인 그래프 구성
main_graph = StateGraph(CustomerServiceState)

# 5. FAQ 서브그래프를 함수로 래핑
def process_faq(state: CustomerServiceState):
    """
    메인 그래프에서 FAQ 서브그래프 호출
    - CustomerServiceState → FAQState 변환
    - FAQ 서브그래프 실행
    - 결과를 CustomerServiceState로 반환
    """
    # FAQ 서브그래프 입력 준비
    faq_state = {
        "messages": state["messages"],
        "customer_id": state["customer_id"],
        "faq_matched": False,
        "faq_answer": ""
    }

    # FAQ 서브그래프 실행
    result = faq_subgraph.invoke(faq_state)

    # FAQ 답변이 있으면 메시지 반환
    if len(result.get("messages", [])) > 0:
        return {"messages": result["messages"]}

    return {}

# 6. 노드 추가
main_graph.add_node("faq", process_faq)
main_graph.add_node("route", route_inquiry)
main_graph.add_node("technical", handle_technical)
main_graph.add_node("billing", handle_billing)

# 7. 라우팅 로직
def route_to_handler(state: CustomerServiceState):
    """분류 결과에 따른 핸들러 선택"""
    if "technical" in state["issue_category"]:
        print("---Routing to technical handler---")
        return "technical"
    elif "billing" in state["issue_category"]:
        print("---Routing to billing handler---")
        return "billing"
    return "end"

# 8. 엣지 설정
main_graph.add_edge(START, "faq")
main_graph.add_edge("faq", "route")
main_graph.add_conditional_edges(
    "route",
    route_to_handler,
    {
        "technical": "technical",
        "billing": "billing",
        "end": END
    }
)
main_graph.add_edge("technical", END)
main_graph.add_edge("billing", END)

# 9. 컴파일
customer_service = main_graph.compile()

# 10. 메인 그래프 시각화 (xray=True: 서브그래프 내부도 표시)
display(Image(customer_service.get_graph(xray=True).draw_mermaid_png()))
```

**메인 그래프 흐름:**
```
START
  ↓
faq (FAQ 서브그래프)
  ├─ check_faq
  └─ respond_faq (매칭 시)
  ↓
route (문의 분류)
  ↓
[조건부 분기]
  ├─ technical → handle_technical → END
  ├─ billing   → handle_billing → END
  └─ general   → END
```

#### 3-4. 시스템 테스트

```python
from pprint import pprint

# 테스트 1: FAQ 질문
print("=" * 80)
print("=== 테스트 1: FAQ 질문 (운영시간) ===")
print("=" * 80)

faq_input = {
    "messages": [HumanMessage(content="매장 운영시간이 어떻게 되나요?")],
    "customer_id": "user123",
    "issue_category": "",
    "satisfaction_score": 0.0
}

for event in customer_service.stream(faq_input, stream_mode="values"):
    pprint(event)
    print("-" * 80)
```

**실행 결과:**
```
================================================================================
=== 테스트 1: FAQ 질문 (운영시간) ===
================================================================================
Checking FAQ for customer: user123
Responding with FAQ: 안녕하세요 user123님, 저희 매장은 평일 9시부터 18시까지...
문의 분류 결과: - general: 일반 문의

{'customer_id': 'user123',
 'issue_category': '- general: 일반 문의',
 'messages': [HumanMessage(content='매장 운영시간이 어떻게 되나요?'),
              AIMessage(content='안녕하세요 user123님, 저희 매장은 평일 9시부터 18시까지 운영합니다.')],
 'satisfaction_score': 0.0}
```

```python
# 테스트 2: 기술 문의
print("\n" + "=" * 80)
print("=== 테스트 2: 기술 문의 ===")
print("=" * 80)

tech_input = {
    "messages": [HumanMessage(content="로그인이 안 되는데 어떻게 해결하나요?")],
    "customer_id": "user124",
    "issue_category": "",
    "satisfaction_score": 0.0
}

for event in customer_service.stream(tech_input, stream_mode="values"):
    pprint(event)
    print("-" * 80)
```

**실행 결과:**
```
================================================================================
=== 테스트 2: 기술 문의 ===
================================================================================
Checking FAQ for customer: user124
문의 분류 결과: - technical: 기술적 문제
---Routing to technical handler---
---Handling technical issue---

{'customer_id': 'user124',
 'issue_category': '- technical: 기술적 문제',
 'messages': [HumanMessage(content='로그인이 안 되는데 어떻게 해결하나요?'),
              AIMessage(content='기술 지원팀에 문의가 전달되었습니다. 곧 전문가가 연락드릴 예정입니다.')],
 'satisfaction_score': 0.8}
```

## 🎯 실습 문제

### 문제 1: 계산기 서브그래프 확장 (난이도: ⭐⭐⭐)

기존 덧셈/곱셈 서브그래프에 **뺄셈과 나눗셈** 기능을 추가하세요.

**요구사항:**
- `subtract` 노드 추가 (num1 - num2)
- `divide` 노드 추가 (num1 / num2)
- 나눗셈은 0으로 나누는 경우 에러 처리
- 모든 연산 결과를 상태에 저장

**힌트:**
```python
class CalculatorState(TypedDict):
    num1: int
    num2: int
    sum_val: int
    product_val: int
    diff_val: int      # 뺄셈 결과
    quotient_val: float  # 나눗셈 결과

def subtract_node(state: CalculatorState):
    # 구현
    pass

def divide_node(state: CalculatorState):
    # 0으로 나누기 에러 처리
    pass
```

### 문제 2: 다국어 FAQ 서브그래프 (난이도: ⭐⭐⭐⭐)

사용자 언어를 감지하고 해당 언어로 FAQ를 제공하는 서브그래프를 구현하세요.

**요구사항:**
- 언어 감지 노드 추가 (한국어, 영어, 일본어)
- 각 언어별 FAQ 데이터베이스
- 감지된 언어에 맞는 답변 제공
- 지원하지 않는 언어는 기본 언어(한국어)로 응답

**힌트:**
```python
class MultilingualFAQState(FAQState):
    detected_language: str
    supported_languages: list

def detect_language(state: MultilingualFAQState):
    # LLM을 사용한 언어 감지
    pass

def get_faq_by_language(state: MultilingualFAQState):
    # 언어별 FAQ 조회
    pass
```

### 문제 3: 승인 워크플로우 서브그래프 (난이도: ⭐⭐⭐⭐⭐)

여러 단계의 승인이 필요한 워크플로우를 서브그래프로 구현하세요.

**요구사항:**
- 1차 승인자 검토 서브그래프
- 2차 승인자 검토 서브그래프
- 각 단계별 승인/거부/수정 요청 처리
- 거부 시 이전 단계로 되돌아가기
- 모든 승인 완료 시 최종 승인 처리

**힌트:**
```python
class ApprovalState(TypedDict):
    document: str
    stage: str  # 'first_review', 'second_review', 'approved', 'rejected'
    first_approver_decision: str
    second_approver_decision: str
    feedback: list

def first_review_subgraph():
    # 1차 승인자 검토 서브그래프
    pass

def second_review_subgraph():
    # 2차 승인자 검토 서브그래프
    pass
```

## ✅ 솔루션 예시

### 문제 1 솔루션: 계산기 서브그래프 확장

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# 1. 확장된 계산기 상태
class CalculatorState(TypedDict):
    num1: int
    num2: int
    sum_val: int
    product_val: int
    diff_val: int
    quotient_val: float
    error: str  # 에러 메시지

# 2. 기존 노드
def add_node(state: CalculatorState):
    return {"sum_val": state["num1"] + state["num2"]}

def multiply_node(state: CalculatorState):
    return {"product_val": state["num1"] * state["num2"]}

# 3. 새로운 노드 - 뺄셈
def subtract_node(state: CalculatorState):
    result = state["num1"] - state["num2"]
    print(f"Subtract: {state['num1']} - {state['num2']} = {result}")
    return {"diff_val": result}

# 4. 새로운 노드 - 나눗셈 (에러 처리 포함)
def divide_node(state: CalculatorState):
    if state["num2"] == 0:
        print("Error: Division by zero!")
        return {
            "quotient_val": 0.0,
            "error": "0으로 나눌 수 없습니다"
        }

    result = state["num1"] / state["num2"]
    print(f"Divide: {state['num1']} ÷ {state['num2']} = {result:.2f}")
    return {"quotient_val": result, "error": ""}

# 5. 서브그래프 구성
calc_builder = StateGraph(CalculatorState)
calc_builder.add_node("add", add_node)
calc_builder.add_node("subtract", subtract_node)
calc_builder.add_node("multiply", multiply_node)
calc_builder.add_node("divide", divide_node)

# 6. 엣지 연결 (순차 실행)
calc_builder.add_edge(START, "add")
calc_builder.add_edge("add", "subtract")
calc_builder.add_edge("subtract", "multiply")
calc_builder.add_edge("multiply", "divide")
calc_builder.add_edge("divide", END)

# 7. 컴파일
calculator_subgraph = calc_builder.compile()

# 8. 테스트
print("=== 계산기 테스트 ===")
result = calculator_subgraph.invoke({
    "num1": 10,
    "num2": 3,
    "sum_val": 0,
    "product_val": 0,
    "diff_val": 0,
    "quotient_val": 0.0,
    "error": ""
})

print(f"\n결과:")
print(f"덧셈: {result['sum_val']}")
print(f"뺄셈: {result['diff_val']}")
print(f"곱셈: {result['product_val']}")
print(f"나눗셈: {result['quotient_val']:.2f}")

# 9. 0으로 나누기 테스트
print("\n=== 0으로 나누기 테스트 ===")
result_error = calculator_subgraph.invoke({
    "num1": 10,
    "num2": 0,
    "sum_val": 0,
    "product_val": 0,
    "diff_val": 0,
    "quotient_val": 0.0,
    "error": ""
})

print(f"에러 메시지: {result_error['error']}")
```

### 문제 2 솔루션: 다국어 FAQ 서브그래프

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# 1. 다국어 FAQ 상태
class MultilingualFAQState(FAQState):
    detected_language: str
    supported_languages: list

# 2. 언어별 FAQ 데이터
FAQ_DATABASE = {
    "ko": {
        "운영시간": "저희 매장은 평일 9시부터 18시까지 운영합니다.",
        "환불": "구매일로부터 7일 이내 환불 가능합니다.",
        "배송": "평균 2-3일 소요되며 주문 후 배송 조회가 가능합니다."
    },
    "en": {
        "operating hours": "We are open from 9 AM to 6 PM on weekdays.",
        "refund": "Refunds are available within 7 days of purchase.",
        "shipping": "Average delivery time is 2-3 days with tracking available."
    },
    "ja": {
        "営業時間": "平日9時から18時まで営業しております。",
        "返金": "購入日から7日以内に返金可能です。",
        "配送": "平均2-3日かかり、注文後に配送追跡が可能です。"
    }
}

# 3. 언어 감지 노드
def detect_language(state: MultilingualFAQState):
    """LLM을 사용한 언어 감지"""
    model = ChatOpenAI(model="gpt-4o-mini")
    last_message = state["messages"][-1]

    response = model.invoke([
        HumanMessage(content=f"""
        다음 텍스트의 언어를 감지하세요:
        {last_message.content}

        다음 중 하나로만 답변하세요: ko, en, ja
        """)
    ])

    detected = response.content.strip().lower()
    if detected not in ["ko", "en", "ja"]:
        detected = "ko"  # 기본 언어

    print(f"감지된 언어: {detected}")
    return {"detected_language": detected}

# 4. FAQ 매칭 (다국어)
def check_multilingual_faq(state: MultilingualFAQState):
    """언어별 FAQ 매칭"""
    last_message = state["messages"][-1]
    lang = state["detected_language"]
    customer_id = state["customer_id"]

    faq_db = FAQ_DATABASE.get(lang, FAQ_DATABASE["ko"])

    # 키워드 매칭
    for keyword, answer in faq_db.items():
        if keyword in last_message.content or keyword.lower() in last_message.content.lower():
            return {
                "faq_matched": True,
                "faq_answer": f"[{lang.upper()}] {answer}"
            }

    return {"faq_matched": False}

# 5. FAQ 응답
def respond_multilingual_faq(state: MultilingualFAQState):
    """다국어 FAQ 응답"""
    return {
        "messages": [AIMessage(content=state["faq_answer"])]
    }

# 6. 서브그래프 구성
multilingual_faq_graph = StateGraph(MultilingualFAQState)
multilingual_faq_graph.add_node("detect_language", detect_language)
multilingual_faq_graph.add_node("check_faq", check_multilingual_faq)
multilingual_faq_graph.add_node("respond_faq", respond_multilingual_faq)

# 7. 엣지 설정
multilingual_faq_graph.add_edge(START, "detect_language")
multilingual_faq_graph.add_edge("detect_language", "check_faq")

def should_respond(state: MultilingualFAQState):
    return "respond_faq" if state["faq_matched"] else "end"

multilingual_faq_graph.add_conditional_edges(
    "check_faq",
    should_respond,
    {"respond_faq": "respond_faq", "end": END}
)
multilingual_faq_graph.add_edge("respond_faq", END)

# 8. 컴파일
multilingual_faq_subgraph = multilingual_faq_graph.compile()

# 9. 테스트
print("=== 다국어 FAQ 테스트 ===\n")

# 한국어 테스트
ko_test = multilingual_faq_subgraph.invoke({
    "messages": [HumanMessage(content="운영시간이 어떻게 되나요?")],
    "customer_id": "user_ko",
    "detected_language": "",
    "faq_matched": False,
    "faq_answer": ""
})
print(f"한국어: {ko_test['messages'][-1].content}\n")

# 영어 테스트
en_test = multilingual_faq_subgraph.invoke({
    "messages": [HumanMessage(content="What are your operating hours?")],
    "customer_id": "user_en",
    "detected_language": "",
    "faq_matched": False,
    "faq_answer": ""
})
print(f"영어: {en_test['messages'][-1].content}\n")

# 일본어 테스트
ja_test = multilingual_faq_subgraph.invoke({
    "messages": [HumanMessage(content="営業時間を教えてください")],
    "customer_id": "user_ja",
    "detected_language": "",
    "faq_matched": False,
    "faq_answer": ""
})
print(f"일본어: {ja_test['messages'][-1].content}")
```

## 🚀 실무 활용 예시

### 예시 1: 모듈형 데이터 처리 파이프라인

데이터 전처리, 검증, 변환을 각각 서브그래프로 구현합니다.

```python
# 1. 전처리 서브그래프
def build_preprocessing_subgraph():
    """데이터 전처리 서브그래프"""
    class PreprocessState(TypedDict):
        raw_data: list
        cleaned_data: list

    def remove_nulls(state):
        cleaned = [x for x in state["raw_data"] if x is not None]
        return {"cleaned_data": cleaned}

    def remove_duplicates(state):
        cleaned = list(set(state["cleaned_data"]))
        return {"cleaned_data": cleaned}

    graph = StateGraph(PreprocessState)
    graph.add_node("remove_nulls", remove_nulls)
    graph.add_node("remove_duplicates", remove_duplicates)
    graph.add_edge(START, "remove_nulls")
    graph.add_edge("remove_nulls", "remove_duplicates")
    graph.add_edge("remove_duplicates", END)

    return graph.compile()

# 2. 검증 서브그래프
def build_validation_subgraph():
    """데이터 검증 서브그래프"""
    # 검증 로직 구현
    pass

# 3. 메인 파이프라인에서 서브그래프 조합
pipeline = StateGraph(PipelineState)
pipeline.add_node("preprocess", build_preprocessing_subgraph())
pipeline.add_node("validate", build_validation_subgraph())
```

### 예시 2: 멀티 에이전트 시스템

각 에이전트를 서브그래프로 구현하여 협업 시스템을 만듭니다.

```python
# 1. 연구 에이전트 서브그래프
def build_research_agent():
    """웹 검색 및 정보 수집"""
    # 구현
    pass

# 2. 분석 에이전트 서브그래프
def build_analysis_agent():
    """수집된 정보 분석"""
    # 구현
    pass

# 3. 보고서 작성 에이전트 서브그래프
def build_writer_agent():
    """분석 결과로 보고서 작성"""
    # 구현
    pass

# 4. 멀티 에이전트 시스템
multi_agent_system = StateGraph(SystemState)
multi_agent_system.add_node("research", build_research_agent())
multi_agent_system.add_node("analysis", build_analysis_agent())
multi_agent_system.add_node("writer", build_writer_agent())
```

### 예시 3: 마이크로서비스 오케스트레이션

각 마이크로서비스를 서브그래프로 추상화합니다.

```python
# 1. 인증 서비스 서브그래프
def auth_service_subgraph():
    """사용자 인증 처리"""
    # JWT 토큰 발급, 검증
    pass

# 2. 결제 서비스 서브그래프
def payment_service_subgraph():
    """결제 처리"""
    # 결제 검증, 트랜잭션 처리
    pass

# 3. 알림 서비스 서브그래프
def notification_service_subgraph():
    """알림 발송"""
    # 이메일, SMS 발송
    pass

# 4. 주문 처리 오케스트레이터
order_orchestrator = StateGraph(OrderState)
order_orchestrator.add_node("auth", auth_service_subgraph())
order_orchestrator.add_node("payment", payment_service_subgraph())
order_orchestrator.add_node("notification", notification_service_subgraph())
```

## 📖 참고 자료

### 공식 문서
- [LangGraph 서브그래프 가이드](https://langchain-ai.github.io/langgraph/concepts/low_level/#subgraphs)
- [StateGraph API 레퍼런스](https://langchain-ai.github.io/langgraph/reference/graphs/)
- [그래프 시각화](https://langchain-ai.github.io/langgraph/how-tos/visualization/)

### 관련 개념
- **모듈형 프로그래밍**: 재사용 가능한 컴포넌트 설계
- **마이크로서비스 아키텍처**: 독립적인 서비스 조합
- **워크플로우 오케스트레이션**: 복잡한 프로세스 관리
- **상태 관리 패턴**: 그래프 간 데이터 전달

### 추가 학습 자료
- LangGraph 고급 패턴
- 서브그래프 성능 최적화
- 대규모 시스템 설계
- 에러 처리 및 복구 전략

---

**다음 단계**: 실습 문제에서 서브그래프 활용 RAG 시스템을 구현하여 실무 역량을 강화하세요.
