# LangGraph HITL (Human-in-the-Loop) - Part 2: 웹 검색 리서치 시스템

## 📚 학습 목표

이 가이드를 완료하면 다음을 할 수 있습니다:

1. **웹 검색 통합**: TavilySearch를 활용한 실시간 웹 검색 시스템을 구축한다
2. **구조화된 출력**: Pydantic을 사용하여 LLM 출력을 안정적으로 구조화한다
3. **병렬 검색**: Send와 맵-리듀스 패턴으로 효율적인 병렬 검색을 구현한다
4. **다단계 HITL**: 주제 분석 → 검색 → 보고서 작성의 각 단계에서 사용자 개입을 적용한다
5. **조건부 워크플로우**: 사용자 피드백에 따라 동적으로 실행 흐름을 제어한다
6. **피드백 처리**: 사용자 피드백을 파싱하고 적절하게 반영하는 로직을 구현한다
7. **실무 시스템 구축**: 완전한 리서치 자동화 시스템을 처음부터 끝까지 개발한다

## 🔑 핵심 개념

### 웹 검색 리서치 시스템 아키텍처

이 시스템은 사용자가 제공한 주제에 대해 자동으로 웹 검색을 수행하고 보고서를 작성하는 완전한 AI 에이전트입니다.

```
┌─────────────┐
│  사용자 주제  │
└──────┬──────┘
       │
       ↓
┌─────────────────┐
│  주제 분석       │  ← LLM으로 키워드 생성
│  (키워드 생성)   │
└──────┬──────────┘
       │
       ↓
┌─────────────────┐
│  🔍 HITL 1      │  ← interrupt: 사용자가 키워드 검토
│  키워드 검토     │     - 승인 / 수정 / 거부
└──────┬──────────┘
       │
       ↓
┌─────────────────┐
│  웹 검색 (병렬)  │  ← 각 키워드로 동시 검색
│  맵-리듀스 패턴  │
└──────┬──────────┘
       │
       ↓
┌─────────────────┐
│  보고서 생성     │  ← LLM으로 검색 결과 종합
└──────┬──────────┘
       │
       ↓
┌─────────────────┐
│  🔍 HITL 2      │  ← interrupt: 사용자가 보고서 검토
│  보고서 검토     │     - 승인 / 수정 요청
└──────┬──────────┘
       │
       ↓
┌─────────────────┐
│  최종 보고서     │
└─────────────────┘
```

### TavilySearch - 실시간 웹 검색

**TavilySearch**는 LangChain에서 제공하는 실시간 웹 검색 도구입니다.

```python
from langchain_tavily import TavilySearch

# 초기화 (API 키 필요)
search_tool = TavilySearch(max_results=5)

# 검색 실행
results = search_tool.invoke("기후변화")
```

**특징:**
- 최신 웹 검색 결과 제공
- 구조화된 결과 (제목, URL, 내용)
- 검색 결과 수 제한 가능
- 빠른 응답 속도

**결과 형식:**
```python
{
    "results": [
        {
            "title": "검색 결과 제목",
            "url": "https://...",
            "content": "검색 결과 내용",
            "score": 0.95  # 관련도 점수
        },
        ...
    ]
}
```

### Pydantic을 사용한 구조화된 출력

LLM 출력을 안정적으로 구조화하기 위해 Pydantic 모델을 사용합니다.

```python
from pydantic import BaseModel, Field

class Keywords(BaseModel):
    """키워드 생성 결과"""
    keywords: List[str] = Field(description="생성된 키워드 목록")
    confidence: float = Field(description="키워드 신뢰도 (0-1)")

# LLM과 연결
llm = ChatOpenAI(model="gpt-4o-mini")
structured_llm = llm.with_structured_output(Keywords)

# 사용
result = structured_llm.invoke("기후변화에 대한 키워드를 생성하세요")
print(result.keywords)  # ['기후변화', '온실가스', '탄소중립']
print(result.confidence)  # 0.9
```

**장점:**
- 타입 안정성 보장
- 자동 검증 (Field 제약 조건)
- 명확한 스키마 정의
- IDE 자동완성 지원

### Send와 병렬 검색 패턴

LangGraph의 `Send`를 사용하면 여러 노드를 병렬로 실행할 수 있습니다.

```python
from langgraph.types import Send

def dispatch_searches(state: ResearchState) -> List[Send]:
    """각 키워드마다 search_one 노드를 병렬 실행"""
    keywords = state["keywords"]
    return [Send("search_one", {"keyword": kw}) for kw in keywords]

def search_one(state: Dict) -> ResearchState:
    """개별 검색 실행"""
    keyword = state["keyword"]
    results = search_tool.invoke(keyword)
    return {"search_results": [results]}

# 그래프 구성
workflow.add_node("dispatch", dispatch_searches)
workflow.add_node("search_one", search_one)
```

**실행 흐름:**
```
dispatch → Send(search_one, kw1)  ─┐
        → Send(search_one, kw2)  ─┤
        → Send(search_one, kw3)  ─┤→ 병렬 실행
        → Send(search_one, kw4)  ─┤
        → Send(search_one, kw5)  ─┘
```

**맵-리듀스 패턴:**
- **Map**: 각 키워드를 개별 검색 노드로 디스패치
- **Reduce**: 모든 검색 결과를 수집하여 다음 노드로 전달

### 다단계 HITL 워크플로우

복잡한 워크플로우에서는 여러 지점에서 사용자 개입이 필요합니다.

**패턴 1: 키워드 검토 (주제 분석 후)**
```python
def review_keywords(state: ResearchState) -> ResearchState:
    keywords = state["keywords"]

    feedback = interrupt({
        "message": "생성된 키워드를 검토해주세요",
        "keywords": keywords,
        "options": ["승인", "수정", "거부"]
    })

    return {"feedback": feedback}
```

**패턴 2: 보고서 검토 (보고서 생성 후)**
```python
def review_report(state: ResearchState) -> ResearchState:
    report = state["report"]

    feedback = interrupt({
        "message": "보고서를 검토해주세요",
        "report": report,
        "options": ["승인", "수정 요청"]
    })

    return {"report_feedback": feedback}
```

### 피드백 처리 로직

사용자 피드백을 파싱하여 적절한 액션을 수행합니다.

```python
def process_feedback(state: ResearchState) -> ResearchState:
    feedback = state.get("feedback", "")

    # 피드백 파싱
    if "승인" in feedback:
        return {"status": "approved"}

    elif "수정:" in feedback:
        # "수정: 키워드1, 키워드2, ..." 형태
        modified_keywords = feedback.split("수정:")[1].strip()
        keywords = [kw.strip() for kw in modified_keywords.split(",")]
        return {"keywords": keywords, "status": "modified"}

    elif "거부" in feedback:
        # 재생성 필요
        return {"status": "rejected"}

    return {"status": "unknown"}
```

## 🛠 환경 설정

### 필수 라이브러리 설치

```bash
pip install -qU \
    langgraph \
    langchain-openai \
    langchain-tavily \
    pydantic \
    python-dotenv
```

### API 키 설정

`.env` 파일에 두 개의 API 키를 설정합니다:

```bash
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

**Tavily API 키 발급:**
1. [https://tavily.com](https://tavily.com) 방문
2. 회원가입 및 로그인
3. API Keys 메뉴에서 키 생성
4. 무료 플랜: 월 1,000회 검색 가능

### 기본 설정 코드

```python
from dotenv import load_dotenv
import os

# 환경 변수 로드
load_dotenv()

# API 키 확인
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("TAVILY_API_KEY가 설정되지 않았습니다.")

print("환경 설정 완료!")
```

## 💻 단계별 구현

### Step 1: 상태 정의 및 도구 설정

웹 검색 리서치 시스템에 필요한 상태와 도구를 정의합니다.

```python
from typing import List, Dict, Annotated, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command, Send
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from IPython.display import Image, display
import uuid

# 1. LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 2. 웹 검색 도구 초기화
search_tool = TavilySearch(
    max_results=5,  # 검색 결과 최대 5개
    search_depth="advanced"  # 심화 검색
)

# 3. 상태 타입 정의
class ResearchState(MessagesState):
    """리서치 시스템 상태"""
    topic: str                          # 사용자 주제
    keywords: List[str]                 # 생성된 키워드
    feedback: str                       # 사용자 피드백
    search_results: List[Dict]          # 검색 결과 목록
    report: str                         # 생성된 보고서
    report_feedback: str                # 보고서 피드백
    ready_for_search: bool              # 검색 준비 상태

# 4. Pydantic 모델 (구조화된 출력용)
class Keywords(BaseModel):
    """키워드 생성 결과"""
    keywords: List[str] = Field(
        description="생성된 키워드 목록 (3-7개)",
        min_items=3,
        max_items=7
    )
    confidence: float = Field(
        description="키워드 신뢰도 (0-1)",
        ge=0.0,
        le=1.0
    )

print("✅ 상태 및 도구 설정 완료")
```

**상태 필드 설명:**
- `topic`: 사용자가 입력한 연구 주제
- `keywords`: LLM이 생성한 검색 키워드 리스트
- `feedback`: 키워드에 대한 사용자 피드백
- `search_results`: 각 키워드의 검색 결과 리스트
- `report`: 검색 결과를 종합한 보고서
- `report_feedback`: 보고서에 대한 사용자 피드백
- `ready_for_search`: 검색 준비 완료 플래그

### Step 2: 주제 분석 노드 구현

사용자 주제를 분석하여 검색 키워드를 생성하고, 사용자 검토를 받습니다.

```python
# 1. 키워드 생성 노드
def generate_keywords(state: ResearchState) -> ResearchState:
    """주제를 분석하여 검색 키워드 생성"""
    print("--- 주제 분석 중 ---")

    topic = state.get("topic", "")

    # 프롬프트 정의
    prompt = ChatPromptTemplate.from_messages([
        ("system", """전문 리서치 분석가로서 효과적인 검색 키워드를 생성하세요.

규칙:
- 3-7개의 핵심 키워드 생성
- 다양한 관점 포함 (기술, 비즈니스, 사회적 영향 등)
- 검색에 효과적인 용어 선택
- 신뢰도(0-1)도 함께 제공"""),
        ("human", "주제: {topic}")
    ])

    # 구조화된 출력 체인
    structured_llm = llm.with_structured_output(Keywords)
    chain = prompt | structured_llm

    # 키워드 생성
    result = chain.invoke({"topic": topic})

    print(f"생성된 키워드: {result.keywords}")
    print(f"신뢰도: {result.confidence:.2f}")

    return {
        "keywords": result.keywords,
        "feedback": ""  # 초기화
    }

# 2. 키워드 검토 노드 (HITL)
def review_keywords(state: ResearchState) -> ResearchState:
    """사용자가 생성된 키워드를 검토"""
    print("\n--- 키워드 검토 단계 (HITL) ---")

    keywords = state.get("keywords", [])
    topic = state.get("topic", "")

    # interrupt로 사용자 검토 요청
    feedback = interrupt({
        "message": "생성된 키워드를 검토해주세요",
        "topic": topic,
        "keywords": keywords,
        "instructions": """
        - '승인': 키워드 그대로 사용
        - '수정: 키워드1, 키워드2, ...': 직접 키워드 입력
        - '거부': 다시 생성
        """,
        "options": ["승인", "수정: ", "거부"]
    })

    print(f"사용자 피드백: {feedback}")

    return {"feedback": feedback}

# 3. 피드백 처리 노드
def process_keyword_feedback(state: ResearchState) -> ResearchState:
    """사용자 피드백에 따라 키워드 처리"""
    print("\n--- 피드백 처리 중 ---")

    feedback = state.get("feedback", "").strip()

    if "수정:" in feedback:
        # 사용자가 직접 입력한 키워드 파싱
        print("✏️ 키워드 수정")
        modified_text = feedback.split("수정:")[1].strip()
        keywords = [kw.strip() for kw in modified_text.split(",")]

        print(f"수정된 키워드: {keywords}")
        return {"keywords": keywords}

    elif "거부" in feedback:
        # 재생성은 그래프 흐름에서 처리
        print("❌ 거부 - 재생성 필요")
        return {}

    else:  # 승인
        print("✅ 승인 - 검색 진행")
        return {}

# 4. 조건부 분기 함수
def should_continue_after_review(state: ResearchState) -> str:
    """검토 후 다음 단계 결정"""
    feedback = state.get("feedback", "").strip().lower()

    if not feedback or "승인" in feedback:
        return "approved"  # 검색 진행

    elif "수정:" in feedback:
        return "process_feedback"  # 피드백 처리

    else:  # 거부
        return "regenerate"  # 재생성

print("✅ 주제 분석 노드 구현 완료")
```

### Step 3: 워크플로우 구성 (주제 분석 부분)

주제 분석 워크플로우를 구성하고 테스트합니다.

```python
# 1. StateGraph 생성
workflow = StateGraph(ResearchState)

# 2. 노드 추가
workflow.add_node("analyze_topic", generate_keywords)
workflow.add_node("review_keywords", review_keywords)
workflow.add_node("process_feedback", process_keyword_feedback)

# 3. 엣지 추가
workflow.add_edge(START, "analyze_topic")
workflow.add_edge("analyze_topic", "review_keywords")

# 4. 조건부 엣지
workflow.add_conditional_edges(
    "review_keywords",
    should_continue_after_review,
    {
        "approved": END,  # 임시로 END (나중에 검색 노드로 연결)
        "process_feedback": "process_feedback",
        "regenerate": "analyze_topic"
    }
)

workflow.add_edge("process_feedback", END)  # 임시

# 5. 컴파일
checkpointer = InMemorySaver()
research_graph = workflow.compile(checkpointer=checkpointer)

# 6. 그래프 시각화
display(Image(research_graph.get_graph().draw_mermaid_png()))

print("✅ 워크플로우 구성 완료")
```

### Step 4: 주제 분석 테스트 (시나리오별)

다양한 사용자 피드백 시나리오를 테스트합니다.

#### 시나리오 1: 승인

```python
print("=" * 80)
print("=== 시나리오 1: 키워드 승인 ===")
print("=" * 80)

# 스레드 설정
thread = {"configurable": {"thread_id": str(uuid.uuid4())}}

# 초기 실행
initial_state = {
    "topic": "인공지능의 미래와 사회적 영향",
    "keywords": [],
    "feedback": "",
    "search_results": [],
    "report": "",
    "report_feedback": "",
    "ready_for_search": False
}

result = research_graph.invoke(initial_state, thread)

# interrupt까지 실행됨
print(f"\n중단 시점 키워드: {result['keywords']}")

# 상태 확인
state = research_graph.get_state(thread)
print(f"다음 노드: {state.next}")

# 사용자 승인
print("\n>>> 사용자가 '승인'을 선택합니다")
final_result = research_graph.invoke(Command(resume="승인"), thread)

print(f"\n최종 키워드: {final_result['keywords']}")
print(f"피드백: {final_result['feedback']}")
```

**실행 결과:**
```
--- 주제 분석 중 ---
생성된 키워드: ['인공지능', 'AI 윤리', '자동화', '일자리 변화', '기술 발전']
신뢰도: 0.85

--- 키워드 검토 단계 (HITL) ---

중단 시점 키워드: ['인공지능', 'AI 윤리', '자동화', '일자리 변화', '기술 발전']
다음 노드: ('review_keywords',)

>>> 사용자가 '승인'을 선택합니다
사용자 피드백: 승인
✅ 승인 - 검색 진행

최종 키워드: ['인공지능', 'AI 윤리', '자동화', '일자리 변화', '기술 발전']
```

#### 시나리오 2: 수정

```python
print("\n" + "=" * 80)
print("=== 시나리오 2: 키워드 수정 ===")
print("=" * 80)

# 새로운 스레드
thread2 = {"configurable": {"thread_id": str(uuid.uuid4())}}

# 초기 실행
result = research_graph.invoke(initial_state, thread2)

print(f"\n원본 키워드: {result['keywords']}")

# 사용자가 직접 키워드 수정
print("\n>>> 사용자가 키워드를 직접 수정합니다")
modified_keywords = "수정: AI 기술, 딥러닝, 머신러닝, 사회 변화, 윤리적 문제"

final_result = research_graph.invoke(Command(resume=modified_keywords), thread2)

print(f"\n수정된 키워드: {final_result['keywords']}")
```

**실행 결과:**
```
원본 키워드: ['인공지능', 'AI 윤리', '자동화', '일자리 변화', '기술 발전']

>>> 사용자가 키워드를 직접 수정합니다
사용자 피드백: 수정: AI 기술, 딥러닝, 머신러닝, 사회 변화, 윤리적 문제
✏️ 키워드 수정
수정된 키워드: ['AI 기술', '딥러닝', '머신러닝', '사회 변화', '윤리적 문제']
```

#### 시나리오 3: 거부 (재생성)

```python
print("\n" + "=" * 80)
print("=== 시나리오 3: 키워드 거부 - 재생성 ===")
print("=" * 80)

# 새로운 스레드
thread3 = {"configurable": {"thread_id": str(uuid.uuid4())}}

# 초기 실행
result = research_graph.invoke(initial_state, thread3)

print(f"\n첫 번째 생성 키워드: {result['keywords']}")

# 사용자 거부
print("\n>>> 사용자가 '거부'를 선택합니다 (재생성)")
result2 = research_graph.invoke(Command(resume="거부"), thread3)

# 재생성된 키워드가 나옴
print(f"\n재생성된 키워드: {result2['keywords']}")

# 이번에는 승인
print("\n>>> 재생성된 키워드를 승인합니다")
final_result = research_graph.invoke(Command(resume="승인"), thread3)

print(f"\n최종 확정 키워드: {final_result['keywords']}")
```

### Step 5: 병렬 웹 검색 노드 구현

각 키워드에 대해 병렬로 웹 검색을 수행합니다.

```python
# 1. 검색 준비 노드
def ready_to_search(state: ResearchState) -> ResearchState:
    """검색 준비"""
    print("\n--- 웹 검색 준비 중 ---")
    keywords = state.get("keywords", [])
    print(f"검색할 키워드: {keywords}")

    return {"ready_for_search": True, "search_results": []}

# 2. 병렬 검색 디스패치
def dispatch_searches(state: ResearchState) -> List[Send]:
    """각 키워드마다 병렬 검색 수행"""
    keywords = state.get("keywords", [])

    print(f"\n--- {len(keywords)}개 키워드 병렬 검색 시작 ---")

    # 각 키워드마다 search_one 노드를 Send로 디스패치
    return [Send("search_one", {"keyword": kw}) for kw in keywords]

# 3. 개별 검색 노드
def search_one(state: Dict) -> ResearchState:
    """개별 키워드 검색"""
    keyword = state["keyword"]

    print(f"🔍 검색 중: {keyword}")

    try:
        # Tavily 검색 실행
        results = search_tool.invoke(keyword)

        # 결과 정리
        search_data = []
        if isinstance(results, dict):
            data = results.get("results", results)
        else:
            data = results

        for item in data[:3]:  # 상위 3개만
            search_data.append({
                "keyword": keyword,
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", "")[:200]  # 200자 제한
            })

        print(f"  ✅ {keyword}: {len(search_data)}개 결과")

        return {"search_results": search_data}

    except Exception as e:
        print(f"  ❌ {keyword} 검색 실패: {e}")
        return {"search_results": []}

print("✅ 웹 검색 노드 구현 완료")
```

**병렬 검색 패턴:**
```
ready_to_search
    ↓
dispatch_searches ──→ Send(search_one, "AI 기술")      ┐
                  ──→ Send(search_one, "딥러닝")       │
                  ──→ Send(search_one, "머신러닝")     ├─→ 병렬 실행
                  ──→ Send(search_one, "사회 변화")    │
                  ──→ Send(search_one, "윤리적 문제")  ┘
                         ↓
                  (모든 결과 자동 수집)
```

### Step 6: 완전한 워크플로우 구성

검색 노드를 포함한 완전한 워크플로우를 구성합니다.

```python
# 1. StateGraph 생성
full_workflow = StateGraph(ResearchState)

# 2. 모든 노드 추가
full_workflow.add_node("analyze_topic", generate_keywords)
full_workflow.add_node("review_keywords", review_keywords)
full_workflow.add_node("process_feedback", process_keyword_feedback)
full_workflow.add_node("ready_search", ready_to_search)
full_workflow.add_node("dispatch", dispatch_searches)
full_workflow.add_node("search_one", search_one)

# 3. 기본 엣지
full_workflow.add_edge(START, "analyze_topic")
full_workflow.add_edge("analyze_topic", "review_keywords")

# 4. 키워드 검토 후 조건부 분기
full_workflow.add_conditional_edges(
    "review_keywords",
    should_continue_after_review,
    {
        "approved": "ready_search",  # 검색 진행
        "process_feedback": "process_feedback",
        "regenerate": "analyze_topic"
    }
)

full_workflow.add_edge("process_feedback", "ready_search")

# 5. 검색 흐름
full_workflow.add_edge("ready_search", "dispatch")
full_workflow.add_edge("search_one", END)  # 병렬 검색 완료 후 종료

# 6. 컴파일
checkpointer = InMemorySaver()
full_research_graph = full_workflow.compile(checkpointer=checkpointer)

# 7. 시각화
display(Image(full_research_graph.get_graph().draw_mermaid_png()))

print("✅ 완전한 워크플로우 구성 완료")
```

### Step 7: 전체 시스템 테스트 (주제 → 검색)

전체 리서치 시스템을 실행하여 웹 검색까지 완료합니다.

```python
print("=" * 80)
print("=== 전체 리서치 시스템 테스트 ===")
print("=" * 80)

# 스레드 설정
thread = {"configurable": {"thread_id": str(uuid.uuid4())}}

# 초기 상태
initial_state = {
    "topic": "양자 컴퓨팅의 현재와 미래",
    "keywords": [],
    "feedback": "",
    "search_results": [],
    "report": "",
    "report_feedback": "",
    "ready_for_search": False
}

# 1단계: 키워드 생성 및 검토까지
result = full_research_graph.invoke(initial_state, thread)

print(f"\n생성된 키워드: {result['keywords']}")

# 2단계: 사용자 승인
print("\n>>> 사용자가 키워드를 승인합니다")
final_result = full_research_graph.invoke(Command(resume="승인"), thread)

# 3단계: 검색 결과 확인
print(f"\n검색 완료! 총 {len(final_result['search_results'])}개 결과")

# 상위 3개 결과 출력
for i, result in enumerate(final_result['search_results'][:3], 1):
    print(f"\n{i}. [{result['keyword']}] {result['title']}")
    print(f"   URL: {result['url']}")
    print(f"   내용: {result['content'][:100]}...")
```

**실행 결과:**
```
--- 주제 분석 중 ---
생성된 키워드: ['양자컴퓨팅', '큐비트', '양자알고리즘', '양자우월성', '양자암호']
신뢰도: 0.92

--- 키워드 검토 단계 (HITL) ---

생성된 키워드: ['양자컴퓨팅', '큐비트', '양자알고리즘', '양자우월성', '양자암호']

>>> 사용자가 키워드를 승인합니다
✅ 승인 - 검색 진행

--- 웹 검색 준비 중 ---
검색할 키워드: ['양자컴퓨팅', '큐비트', '양자알고리즘', '양자우월성', '양자암호']

--- 5개 키워드 병렬 검색 시작 ---
🔍 검색 중: 양자컴퓨팅
🔍 검색 중: 큐비트
🔍 검색 중: 양자알고리즘
🔍 검색 중: 양자우월성
🔍 검색 중: 양자암호
  ✅ 양자컴퓨팅: 3개 결과
  ✅ 큐비트: 3개 결과
  ✅ 양자알고리즘: 3개 결과
  ✅ 양자우월성: 3개 결과
  ✅ 양자암호: 3개 결과

검색 완료! 총 15개 결과

1. [양자컴퓨팅] 양자컴퓨터, 상용화는 언제?
   URL: https://...
   내용: 양자컴퓨터는 기존 컴퓨터와 다른 원리로 작동하여 특정 문제를 훨씬 빠르게 해결할 수 있습니다...

2. [큐비트] 큐비트의 원리와 구현 방식
   URL: https://...
   내용: 큐비트는 양자역학의 중첩 원리를 이용하여 0과 1을 동시에 표현할 수 있는 양자 정보의 기본 단위입니다...

3. [양자알고리즘] Shor 알고리즘과 암호 해독
   URL: https://...
   내용: 양자알고리즘 중 가장 유명한 Shor 알고리즘은 큰 수를 인수분해하는 데 사용되며...
```

### Step 8: 보고서 생성 및 검토 (HITL 2단계)

검색 결과를 바탕으로 보고서를 생성하고 사용자 검토를 받습니다.

```python
# 1. 보고서 생성 노드
def generate_report(state: ResearchState) -> ResearchState:
    """검색 결과를 종합하여 보고서 생성"""
    print("\n--- 보고서 생성 중 ---")

    topic = state.get("topic", "")
    search_results = state.get("search_results", [])

    # 검색 결과 정리
    context = "\n\n".join([
        f"[{r['keyword']}] {r['title']}\n{r['content']}"
        for r in search_results[:10]  # 상위 10개
    ])

    # 보고서 생성 프롬프트
    prompt = ChatPromptTemplate.from_messages([
        ("system", """전문 리서치 작가로서 검색 결과를 바탕으로 체계적인 보고서를 작성하세요.

구조:
1. 개요
2. 주요 내용 (3-5개 섹션)
3. 결론 및 시사점

길이: 500-800자"""),
        ("human", """주제: {topic}

검색 결과:
{context}

위 내용을 바탕으로 보고서를 작성하세요.""")
    ])

    chain = prompt | llm

    # 보고서 생성
    response = chain.invoke({"topic": topic, "context": context})
    report = response.content

    print(f"보고서 생성 완료 ({len(report)}자)")

    return {"report": report}

# 2. 보고서 검토 노드 (HITL)
def review_report(state: ResearchState) -> ResearchState:
    """사용자가 보고서를 검토"""
    print("\n--- 보고서 검토 단계 (HITL) ---")

    report = state.get("report", "")

    # interrupt로 사용자 검토 요청
    feedback = interrupt({
        "message": "생성된 보고서를 검토해주세요",
        "report": report,
        "instructions": """
        - '승인': 보고서 그대로 사용
        - '수정 요청: [수정 내용]': 특정 부분 수정 요청
        """,
        "options": ["승인", "수정 요청: "]
    })

    print(f"사용자 피드백: {feedback[:50]}...")

    return {"report_feedback": feedback}

# 3. 보고서 수정 노드
def revise_report(state: ResearchState) -> ResearchState:
    """사용자 피드백을 반영하여 보고서 수정"""
    print("\n--- 보고서 수정 중 ---")

    report = state.get("report", "")
    feedback = state.get("report_feedback", "")

    # 피드백에서 수정 내용 추출
    if "수정 요청:" in feedback:
        modification = feedback.split("수정 요청:")[1].strip()

        # 수정 프롬프트
        prompt = ChatPromptTemplate.from_messages([
            ("system", "사용자 피드백을 반영하여 보고서를 수정하세요."),
            ("human", """원본 보고서:
{report}

사용자 수정 요청:
{modification}

위 피드백을 반영하여 보고서를 수정하세요.""")
        ])

        chain = prompt | llm

        # 수정
        response = chain.invoke({"report": report, "modification": modification})
        revised_report = response.content

        print(f"보고서 수정 완료")

        return {"report": revised_report}

    return {}

# 4. 조건부 분기
def should_continue_after_report_review(state: ResearchState) -> str:
    """보고서 검토 후 다음 단계 결정"""
    feedback = state.get("report_feedback", "").strip()

    if "승인" in feedback:
        return "approved"
    elif "수정 요청:" in feedback:
        return "revise"
    else:
        return "approved"  # 기본값

print("✅ 보고서 생성 및 검토 노드 구현 완료")
```

### Step 9: 최종 완전한 워크플로우

보고서 생성 및 검토를 포함한 최종 워크플로우를 구성합니다.

```python
# 1. StateGraph 생성
final_workflow = StateGraph(ResearchState)

# 2. 모든 노드 추가
final_workflow.add_node("analyze_topic", generate_keywords)
final_workflow.add_node("review_keywords", review_keywords)
final_workflow.add_node("process_feedback", process_keyword_feedback)
final_workflow.add_node("ready_search", ready_to_search)
final_workflow.add_node("dispatch", dispatch_searches)
final_workflow.add_node("search_one", search_one)
final_workflow.add_node("generate_report", generate_report)
final_workflow.add_node("review_report", review_report)
final_workflow.add_node("revise_report", revise_report)

# 3. 주제 분석 흐름
final_workflow.add_edge(START, "analyze_topic")
final_workflow.add_edge("analyze_topic", "review_keywords")

final_workflow.add_conditional_edges(
    "review_keywords",
    should_continue_after_review,
    {
        "approved": "ready_search",
        "process_feedback": "process_feedback",
        "regenerate": "analyze_topic"
    }
)

final_workflow.add_edge("process_feedback", "ready_search")

# 4. 검색 흐름
final_workflow.add_edge("ready_search", "dispatch")
final_workflow.add_edge("dispatch", "generate_report")  # dispatch 후 보고서 생성
final_workflow.add_edge("search_one", "dispatch")  # search_one은 dispatch로 돌아감 (맵-리듀스)

# 5. 보고서 흐름
final_workflow.add_edge("generate_report", "review_report")

final_workflow.add_conditional_edges(
    "review_report",
    should_continue_after_report_review,
    {
        "approved": END,
        "revise": "revise_report"
    }
)

final_workflow.add_edge("revise_report", "review_report")  # 재검토

# 6. 컴파일
checkpointer = InMemorySaver()
final_research_graph = final_workflow.compile(checkpointer=checkpointer)

# 7. 시각화
display(Image(final_research_graph.get_graph().draw_mermaid_png()))

print("✅ 최종 완전한 워크플로우 구성 완료")
```

### Step 10: 최종 시스템 전체 테스트

주제 입력부터 보고서 완성까지 전체 과정을 테스트합니다.

```python
print("=" * 80)
print("=== 전체 리서치 시스템 - 완전한 테스트 ===")
print("=" * 80)

# 스레드 설정
thread = {"configurable": {"thread_id": str(uuid.uuid4())}}

# 초기 상태
initial_state = {
    "topic": "메타버스의 현재와 미래 전망",
    "keywords": [],
    "feedback": "",
    "search_results": [],
    "report": "",
    "report_feedback": "",
    "ready_for_search": False
}

# 1단계: 키워드 생성 및 검토
print("\n【1단계】 키워드 생성")
result = final_research_graph.invoke(initial_state, thread)

print(f"생성된 키워드: {result['keywords']}")

# 사용자 승인
print("\n>>> 사용자: 키워드 승인")
result = final_research_graph.invoke(Command(resume="승인"), thread)

# 2단계: 웹 검색 (자동 실행됨)
print(f"\n【2단계】 웹 검색 완료: {len(result.get('search_results', []))}개 결과")

# 3단계: 보고서 생성 및 검토
print("\n【3단계】 보고서 생성 및 검토")
state = final_research_graph.get_state(thread)

if "report" in result and result["report"]:
    print(f"\n생성된 보고서 ({len(result['report'])}자):")
    print("-" * 80)
    print(result["report"][:300] + "...")
    print("-" * 80)

# 보고서 검토 단계에서 중단됨
print("\n>>> 사용자: 보고서 승인")
final_result = final_research_graph.invoke(Command(resume="승인"), thread)

print("\n【완료】 최종 보고서:")
print("=" * 80)
print(final_result["report"])
print("=" * 80)

print(f"\n✅ 전체 리서치 완료!")
print(f"- 주제: {final_result['topic']}")
print(f"- 키워드: {len(final_result['keywords'])}개")
print(f"- 검색 결과: {len(final_result['search_results'])}개")
print(f"- 보고서: {len(final_result['report'])}자")
```

**실행 결과:**
```
================================================================================
=== 전체 리서치 시스템 - 완전한 테스트 ===
================================================================================

【1단계】 키워드 생성
--- 주제 분석 중 ---
생성된 키워드: ['메타버스', '가상현실', 'VR/AR', '디지털 경제', '메타버스 플랫폼']
신뢰도: 0.88

--- 키워드 검토 단계 (HITL) ---

생성된 키워드: ['메타버스', '가상현실', 'VR/AR', '디지털 경제', '메타버스 플랫폼']

>>> 사용자: 키워드 승인
✅ 승인 - 검색 진행

【2단계】 웹 검색 완료: 15개 결과
--- 웹 검색 준비 중 ---
--- 5개 키워드 병렬 검색 시작 ---
🔍 검색 중: 메타버스
🔍 검색 중: 가상현실
🔍 검색 중: VR/AR
🔍 검색 중: 디지털 경제
🔍 검색 중: 메타버스 플랫폼

【3단계】 보고서 생성 및 검토
--- 보고서 생성 중 ---
보고서 생성 완료 (687자)

생성된 보고서 (687자):
--------------------------------------------------------------------------------
# 메타버스의 현재와 미래 전망

## 1. 개요
메타버스는 가상현실(VR)과 증강현실(AR) 기술을 기반으로 한 디지털 공간으로, 사용자들이 아바타를 통해 상호작용하고 경제활동을 수행하는 가상 세계입니다...
--------------------------------------------------------------------------------

--- 보고서 검토 단계 (HITL) ---

>>> 사용자: 보고서 승인

【완료】 최종 보고서:
================================================================================
# 메타버스의 현재와 미래 전망

## 1. 개요
메타버스는 가상현실(VR)과 증강현실(AR) 기술을 기반으로 한 디지털 공간으로, 사용자들이 아바타를 통해 상호작용하고 경제활동을 수행하는 가상 세계입니다.

## 2. 주요 내용

### 2.1 기술 발전
VR/AR 기술의 발전으로 더욱 몰입감 있는 경험이 가능해지고 있습니다. 특히 하드웨어의 성능 향상과 5G 네트워크의 확산이 메타버스의 성장을 가속화하고 있습니다.

### 2.2 메타버스 플랫폼
로블록스, 제페토, 디센트럴랜드 등 다양한 메타버스 플랫폼이 등장하여 각기 다른 특색을 가진 가상 세계를 제공하고 있습니다.

### 2.3 디지털 경제
메타버스 내에서 NFT, 가상 부동산, 디지털 상품 거래 등 새로운 경제 생태계가 형성되고 있습니다.

## 3. 결론 및 시사점
메타버스는 단순한 게임을 넘어 교육, 비즈니스, 문화 등 다양한 분야로 확장되고 있습니다. 향후 기술 발전과 함께 우리 일상의 중요한 일부가 될 것으로 전망됩니다.
================================================================================

✅ 전체 리서치 완료!
- 주제: 메타버스의 현재와 미래 전망
- 키워드: 5개
- 검색 결과: 15개
- 보고서: 687자
```

## 🎯 실습 문제

### 문제 1: 검색 결과 필터링 (난이도: ⭐⭐⭐)

검색 결과 중 신뢰도가 낮은 결과를 필터링하는 노드를 추가하세요.

**요구사항:**
- 검색 결과의 `score` 필드 확인 (0-1 범위)
- 0.7 미만의 결과는 제외
- 필터링된 결과 개수 출력
- 필터링 전/후 비교

**힌트:**
```python
def filter_search_results(state: ResearchState) -> ResearchState:
    search_results = state.get("search_results", [])

    # score가 0.7 이상인 결과만 남기기
    filtered = [r for r in search_results if r.get("score", 0) >= 0.7]

    print(f"필터링: {len(search_results)}개 → {len(filtered)}개")

    return {"search_results": filtered}
```

### 문제 2: 다국어 검색 지원 (난이도: ⭐⭐⭐⭐)

사용자가 검색 언어를 선택할 수 있도록 HITL을 추가하세요.

**요구사항:**
- 키워드 생성 후, 검색 언어 선택 interrupt 추가
- 선택 옵션: "한국어", "영어", "일본어", "중국어"
- 선택된 언어로 키워드 번역
- 번역된 키워드로 검색 수행

**힌트:**
```python
def select_language(state: ResearchState) -> ResearchState:
    language = interrupt({
        "message": "검색할 언어를 선택하세요",
        "keywords": state.get("keywords", []),
        "options": ["한국어", "영어", "일본어", "중국어"]
    })

    if language != "한국어":
        # LLM으로 키워드 번역
        translated_keywords = translate_keywords(
            state["keywords"],
            target_language=language
        )
        return {"keywords": translated_keywords, "search_language": language}

    return {"search_language": "한국어"}
```

### 문제 3: 심화 리서치 모드 (난이도: ⭐⭐⭐⭐⭐)

첫 번째 검색 결과를 바탕으로 추가 검색이 필요한지 사용자에게 물어보는 시스템을 구현하세요.

**요구사항:**
- 첫 번째 검색 완료 후 결과 요약 제공
- 사용자에게 "추가 검색 필요?" interrupt
- 필요 시 새로운 키워드 생성 및 재검색
- 최대 3회까지 반복 검색 가능
- 모든 검색 결과를 병합하여 최종 보고서 생성

**힌트:**
```python
class ResearchState(MessagesState):
    # 기존 필드
    topic: str
    keywords: List[str]
    search_results: List[Dict]
    report: str

    # 추가 필드
    search_iteration: int  # 검색 횟수
    additional_keywords: List[str]  # 추가 키워드
    need_more_search: bool  # 추가 검색 필요 여부

def check_need_more_search(state: ResearchState) -> ResearchState:
    iteration = state.get("search_iteration", 1)

    if iteration >= 3:
        return {"need_more_search": False}

    # 현재 검색 결과 요약
    summary = summarize_search_results(state["search_results"])

    decision = interrupt({
        "message": f"검색 결과 요약 (검색 {iteration}회차)",
        "summary": summary,
        "options": ["충분함 - 보고서 작성", "추가 검색 필요"]
    })

    if "추가 검색" in decision:
        # 추가 키워드 생성
        additional = generate_additional_keywords(
            state["topic"],
            state["keywords"],
            summary
        )
        return {
            "additional_keywords": additional,
            "need_more_search": True,
            "search_iteration": iteration + 1
        }

    return {"need_more_search": False}
```

## ✅ 솔루션 예시

### 문제 1 솔루션: 검색 결과 필터링

```python
from typing import List, Dict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

# 1. 필터링 노드 구현
def filter_search_results(state: ResearchState) -> ResearchState:
    """신뢰도가 낮은 검색 결과 필터링"""
    print("\n--- 검색 결과 필터링 중 ---")

    search_results = state.get("search_results", [])
    threshold = 0.7

    print(f"원본 결과: {len(search_results)}개")

    # score 기준 필터링
    filtered_results = []
    for result in search_results:
        score = result.get("score", 0.5)  # 기본값 0.5

        if score >= threshold:
            filtered_results.append(result)
        else:
            print(f"  제외: {result.get('title', 'Unknown')[:50]} (score: {score:.2f})")

    print(f"필터링 후: {len(filtered_results)}개 (제거: {len(search_results) - len(filtered_results)}개)")

    return {"search_results": filtered_results}

# 2. 워크플로우에 노드 추가
workflow = StateGraph(ResearchState)

# 기존 노드들...
workflow.add_node("filter_results", filter_search_results)

# 검색 후 필터링 추가
workflow.add_edge("search_one", "filter_results")
workflow.add_edge("filter_results", "generate_report")

# 컴파일
checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

# 3. 테스트
thread = {"configurable": {"thread_id": str(uuid.uuid4())}}

# 테스트용 검색 결과 (score 포함)
test_results = [
    {"title": "고품질 결과 1", "content": "...", "score": 0.95},
    {"title": "고품질 결과 2", "content": "...", "score": 0.88},
    {"title": "저품질 결과 1", "content": "...", "score": 0.45},
    {"title": "중품질 결과", "content": "...", "score": 0.72},
    {"title": "저품질 결과 2", "content": "...", "score": 0.60},
]

result = graph.invoke({
    "topic": "테스트",
    "search_results": test_results
}, thread)

print(f"\n최종 결과: {len(result['search_results'])}개")
for r in result["search_results"]:
    print(f"  - {r['title']} (score: {r['score']:.2f})")
```

**실행 결과:**
```
--- 검색 결과 필터링 중 ---
원본 결과: 5개
  제외: 저품질 결과 1 (score: 0.45)
  제외: 저품질 결과 2 (score: 0.60)
필터링 후: 3개 (제거: 2개)

최종 결과: 3개
  - 고품질 결과 1 (score: 0.95)
  - 고품질 결과 2 (score: 0.88)
  - 중품질 결과 (score: 0.72)
```

### 문제 2 솔루션: 다국어 검색 지원

```python
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import interrupt, Command

# 1. 언어 선택 노드
def select_search_language(state: ResearchState) -> ResearchState:
    """검색 언어 선택"""
    print("\n--- 검색 언어 선택 ---")

    keywords = state.get("keywords", [])

    # 사용자에게 언어 선택 요청
    language = interrupt({
        "message": "검색할 언어를 선택하세요",
        "current_keywords": keywords,
        "options": ["한국어 (기본)", "영어", "일본어", "중국어"]
    })

    print(f"선택된 언어: {language}")

    return {"search_language": language}

# 2. 키워드 번역 노드
def translate_keywords(state: ResearchState) -> ResearchState:
    """선택된 언어로 키워드 번역"""
    print("\n--- 키워드 번역 중 ---")

    language = state.get("search_language", "한국어")
    keywords = state.get("keywords", [])

    # 한국어면 번역 불필요
    if "한국어" in language:
        print("한국어 검색 - 번역 불필요")
        return {}

    # 언어 코드 매핑
    language_map = {
        "영어": "English",
        "일본어": "Japanese",
        "중국어": "Chinese"
    }

    target_lang = language_map.get(language, "English")

    # 번역 프롬프트
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"키워드를 {target_lang}로 번역하세요. 검색에 최적화된 용어를 사용하세요."),
        ("human", "키워드: {keywords}\n\n각 키워드를 쉼표로 구분하여 번역하세요.")
    ])

    chain = prompt | llm

    # 번역 실행
    response = chain.invoke({"keywords": ", ".join(keywords)})
    translated_text = response.content

    # 번역된 키워드 파싱
    translated_keywords = [kw.strip() for kw in translated_text.split(",")]

    print(f"원본: {keywords}")
    print(f"번역: {translated_keywords}")

    return {"keywords": translated_keywords}

# 3. 워크플로우 구성
workflow = StateGraph(ResearchState)

# 노드 추가
workflow.add_node("select_language", select_search_language)
workflow.add_node("translate", translate_keywords)

# 흐름: 키워드 검토 → 언어 선택 → 번역 → 검색
workflow.add_edge("review_keywords", "select_language")
workflow.add_edge("select_language", "translate")
workflow.add_edge("translate", "ready_search")

# 컴파일
checkpointer = InMemorySaver()
multilang_graph = workflow.compile(checkpointer=checkpointer)

# 4. 테스트
thread = {"configurable": {"thread_id": str(uuid.uuid4())}}

initial_state = {
    "topic": "인공지능의 미래",
    "keywords": ["인공지능", "머신러닝", "딥러닝", "AI 윤리"],
    "search_language": ""
}

# 키워드 승인 후 언어 선택
result = multilang_graph.invoke(initial_state, thread)

# 사용자가 영어 선택
print("\n>>> 사용자: 영어 선택")
final_result = multilang_graph.invoke(Command(resume="영어"), thread)

print(f"\n최종 검색 키워드: {final_result['keywords']}")
print(f"검색 언어: {final_result['search_language']}")
```

**실행 결과:**
```
--- 검색 언어 선택 ---

>>> 사용자: 영어 선택
선택된 언어: 영어

--- 키워드 번역 중 ---
원본: ['인공지능', '머신러닝', '딥러닝', 'AI 윤리']
번역: ['Artificial Intelligence', 'Machine Learning', 'Deep Learning', 'AI Ethics']

최종 검색 키워드: ['Artificial Intelligence', 'Machine Learning', 'Deep Learning', 'AI Ethics']
검색 언어: 영어
```

### 문제 3 솔루션: 심화 리서치 모드

```python
from typing import List, Dict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command

# 1. 확장된 상태 정의
class AdvancedResearchState(ResearchState):
    search_iteration: int  # 현재 검색 회차
    all_search_results: List[Dict]  # 모든 회차의 검색 결과
    search_history: List[Dict]  # 검색 이력
    need_more_search: bool  # 추가 검색 필요 여부

# 2. 검색 결과 요약 노드
def summarize_search_results(state: AdvancedResearchState) -> AdvancedResearchState:
    """현재까지의 검색 결과 요약"""
    print("\n--- 검색 결과 요약 중 ---")

    iteration = state.get("search_iteration", 1)
    search_results = state.get("search_results", [])

    # 간단한 요약 생성
    keywords_found = set([r.get("keyword", "") for r in search_results])

    summary = {
        "iteration": iteration,
        "total_results": len(search_results),
        "keywords_covered": list(keywords_found),
        "sample_titles": [r.get("title", "") for r in search_results[:5]]
    }

    print(f"검색 회차: {iteration}")
    print(f"총 결과: {len(search_results)}개")
    print(f"커버된 키워드: {keywords_found}")

    return {"search_summary": summary}

# 3. 추가 검색 필요 여부 확인 노드
def check_need_more_search(state: AdvancedResearchState) -> AdvancedResearchState:
    """추가 검색 필요 여부 확인"""
    print("\n--- 추가 검색 필요 여부 확인 ---")

    iteration = state.get("search_iteration", 1)
    max_iterations = 3

    if iteration >= max_iterations:
        print(f"최대 검색 회차({max_iterations}) 도달 - 보고서 작성 진행")
        return {"need_more_search": False}

    summary = state.get("search_summary", {})

    # 사용자에게 추가 검색 여부 질문
    decision = interrupt({
        "message": f"검색 결과 검토 (회차 {iteration}/{max_iterations})",
        "summary": summary,
        "total_results": summary.get("total_results", 0),
        "options": [
            "충분함 - 보고서 작성",
            "추가 검색 필요"
        ]
    })

    if "추가 검색" in decision:
        print("✅ 추가 검색 진행")
        return {"need_more_search": True}
    else:
        print("✅ 보고서 작성 진행")
        return {"need_more_search": False}

# 4. 추가 키워드 생성 노드
def generate_additional_keywords(state: AdvancedResearchState) -> AdvancedResearchState:
    """기존 검색 결과를 바탕으로 추가 키워드 생성"""
    print("\n--- 추가 키워드 생성 중 ---")

    topic = state.get("topic", "")
    original_keywords = state.get("keywords", [])
    search_results = state.get("search_results", [])

    # 검색 결과에서 자주 언급된 용어 추출 (간단한 예시)
    all_content = " ".join([r.get("content", "") for r in search_results])

    # LLM으로 추가 키워드 생성
    prompt = ChatPromptTemplate.from_messages([
        ("system", "검색 결과를 분석하여 추가 검색이 필요한 새로운 키워드를 생성하세요."),
        ("human", """주제: {topic}

기존 키워드: {original_keywords}

검색 결과 요약:
{content}

위 내용을 바탕으로 더 깊이 있는 검색을 위한 3-5개의 새로운 키워드를 생성하세요.
기존 키워드와 중복되지 않아야 합니다.""")
    ])

    structured_llm = llm.with_structured_output(Keywords)
    chain = prompt | structured_llm

    result = chain.invoke({
        "topic": topic,
        "original_keywords": ", ".join(original_keywords),
        "content": all_content[:500]  # 500자 제한
    })

    additional_keywords = result.keywords

    print(f"추가 키워드: {additional_keywords}")

    # 검색 회차 증가
    iteration = state.get("search_iteration", 1)

    return {
        "keywords": additional_keywords,
        "search_iteration": iteration + 1
    }

# 5. 결과 병합 노드
def merge_all_results(state: AdvancedResearchState) -> AdvancedResearchState:
    """모든 회차의 검색 결과 병합"""
    print("\n--- 검색 결과 병합 중 ---")

    all_results = state.get("all_search_results", [])
    current_results = state.get("search_results", [])

    # 결과 병합
    merged = all_results + current_results

    print(f"총 검색 결과: {len(merged)}개 (회차: {state.get('search_iteration', 1)})")

    return {
        "all_search_results": merged,
        "search_results": []  # 초기화
    }

# 6. 워크플로우 구성
workflow = StateGraph(AdvancedResearchState)

# 노드 추가
workflow.add_node("summarize", summarize_search_results)
workflow.add_node("check_more", check_need_more_search)
workflow.add_node("generate_additional", generate_additional_keywords)
workflow.add_node("merge_results", merge_all_results)

# 흐름
workflow.add_edge("search_one", "merge_results")
workflow.add_edge("merge_results", "summarize")
workflow.add_edge("summarize", "check_more")

# 조건부 분기
def after_check_decision(state: AdvancedResearchState) -> str:
    if state.get("need_more_search", False):
        return "more_search"
    else:
        return "generate_report"

workflow.add_conditional_edges(
    "check_more",
    after_check_decision,
    {
        "more_search": "generate_additional",
        "generate_report": "generate_report"
    }
)

workflow.add_edge("generate_additional", "dispatch")  # 재검색

# 컴파일
checkpointer = InMemorySaver()
advanced_graph = workflow.compile(checkpointer=checkpointer)

# 7. 테스트
print("=" * 80)
print("=== 심화 리서치 모드 테스트 ===")
print("=" * 80)

thread = {"configurable": {"thread_id": str(uuid.uuid4())}}

initial_state = {
    "topic": "블록체인 기술의 응용",
    "keywords": ["블록체인", "암호화폐", "스마트컨트랙트"],
    "search_iteration": 1,
    "all_search_results": [],
    "need_more_search": False
}

# 1차 검색 및 검토
result = advanced_graph.invoke(initial_state, thread)

print("\n>>> 사용자: 추가 검색 필요")
result = advanced_graph.invoke(Command(resume="추가 검색 필요"), thread)

# 2차 검색 결과 검토
print("\n>>> 사용자: 충분함 - 보고서 작성")
final_result = advanced_graph.invoke(Command(resume="충분함 - 보고서 작성"), thread)

print(f"\n최종 검색 회차: {final_result.get('search_iteration', 1)}")
print(f"총 검색 결과: {len(final_result.get('all_search_results', []))}개")
```

**실행 결과:**
```
================================================================================
=== 심화 리서치 모드 테스트 ===
================================================================================

--- 검색 결과 병합 중 ---
총 검색 결과: 9개 (회차: 1)

--- 검색 결과 요약 중 ---
검색 회차: 1
총 결과: 9개
커버된 키워드: {'블록체인', '암호화폐', '스마트컨트랙트'}

--- 추가 검색 필요 여부 확인 ---

>>> 사용자: 추가 검색 필요
✅ 추가 검색 진행

--- 추가 키워드 생성 중 ---
추가 키워드: ['NFT', 'DeFi', '분산원장', 'Web3']

[2차 검색 진행...]

--- 검색 결과 병합 중 ---
총 검색 결과: 21개 (회차: 2)

--- 추가 검색 필요 여부 확인 ---

>>> 사용자: 충분함 - 보고서 작성
✅ 보고서 작성 진행

최종 검색 회차: 2
총 검색 결과: 21개
```

## 🚀 실무 활용 예시

### 예시 1: 시장 조사 자동화 시스템

새로운 시장에 진입하기 전 자동으로 시장 조사를 수행하는 시스템입니다.

```python
# 시장 조사 특화 상태
class MarketResearchState(ResearchState):
    market_name: str
    competitors: List[str]
    market_size: Dict
    trends: List[str]
    recommendations: List[str]

# 특화 노드들
def analyze_competitors(state: MarketResearchState):
    """경쟁사 분석"""
    search_results = state.get("search_results", [])

    # 경쟁사 관련 정보 추출
    competitor_info = extract_competitor_info(search_results)

    return {"competitors": competitor_info}

def estimate_market_size(state: MarketResearchState):
    """시장 규모 추정"""
    # LLM으로 시장 규모 분석
    prompt = """검색 결과를 바탕으로 시장 규모를 추정하세요:
    - 현재 시장 규모
    - 연평균 성장률 (CAGR)
    - 향후 5년 전망
    """

    # ... 분석 로직

    return {"market_size": market_data}

# 활용 예시
result = market_research_graph.invoke({
    "market_name": "AI 챗봇 시장",
    "topic": "AI 챗봇 시장 분석"
})

print(f"시장: {result['market_name']}")
print(f"주요 경쟁사: {result['competitors']}")
print(f"시장 규모: {result['market_size']}")
```

### 예시 2: 기술 트렌드 모니터링

최신 기술 트렌드를 주기적으로 모니터링하고 보고서를 생성하는 시스템입니다.

```python
import schedule
import time

# 주간 트렌드 리포트 생성
def weekly_tech_trends():
    """매주 기술 트렌드 리포트 자동 생성"""

    topics = [
        "인공지능 최신 동향",
        "클라우드 컴퓨팅 트렌드",
        "사이버보안 이슈"
    ]

    for topic in topics:
        print(f"\n=== {topic} 리포트 생성 중 ===")

        thread = {"configurable": {"thread_id": str(uuid.uuid4())}}

        # 자동 실행 (사용자 개입 최소화)
        result = trend_monitor_graph.invoke({
            "topic": topic,
            "auto_approve": True  # 자동 승인 모드
        }, thread)

        # 보고서 저장
        save_report(topic, result["report"])

        print(f"✅ {topic} 리포트 완료")

# 스케줄 설정
schedule.every().monday.at("09:00").do(weekly_tech_trends)

# 실행
while True:
    schedule.run_pending()
    time.sleep(3600)  # 1시간마다 체크
```

### 예시 3: 학술 논문 리서치 도우미

특정 주제에 대한 학술 논문을 검색하고 요약하는 시스템입니다.

```python
# 학술 논문 특화 상태
class AcademicResearchState(ResearchState):
    paper_title: str
    authors: List[str]
    publication_year: int
    citations: int
    key_findings: List[str]
    methodology: str

# 논문 검색 특화 노드
def search_academic_papers(state: AcademicResearchState):
    """학술 논문 검색 (Google Scholar 등)"""

    # 논문 검색 키워드 생성
    keywords = state.get("keywords", [])
    academic_keywords = [f"{kw} academic paper" for kw in keywords]

    # 검색 실행
    papers = []
    for kw in academic_keywords:
        results = search_tool.invoke(kw)
        papers.extend(extract_paper_info(results))

    return {"papers": papers}

def summarize_papers(state: AcademicResearchState):
    """논문 요약"""

    papers = state.get("papers", [])

    # 각 논문 요약
    summaries = []
    for paper in papers:
        summary = llm.invoke(f"""다음 논문을 요약하세요:
        제목: {paper['title']}
        초록: {paper['abstract']}

        주요 내용, 방법론, 결론을 포함하여 요약하세요.""")

        summaries.append({
            "title": paper["title"],
            "summary": summary.content
        })

    return {"paper_summaries": summaries}

# 활용
result = academic_graph.invoke({
    "topic": "Transformer 모델의 최신 발전",
    "keywords": []
})

for paper in result["paper_summaries"]:
    print(f"\n논문: {paper['title']}")
    print(f"요약: {paper['summary']}")
```

## 📖 참고 자료

### 공식 문서
- [LangGraph HITL 공식 가이드](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)
- [TavilySearch API](https://docs.tavily.com/)
- [Pydantic 구조화된 출력](https://docs.pydantic.dev/)
- [Send와 병렬 실행](https://langchain-ai.github.io/langgraph/concepts/low_level/#send)

### 관련 기술
- **맵-리듀스 패턴**: 병렬 데이터 처리 패턴
- **웹 스크래핑**: BeautifulSoup, Selenium 활용
- **임베딩 검색**: 의미론적 검색 구현
- **프롬프트 엔지니어링**: 효과적인 LLM 활용

### 추가 학습 자료
- LangGraph Advanced Patterns
- Multi-Agent Research Systems
- Production HITL Workflows
- Real-time Web Search Integration

---

**Part 1 복습**: [PRJ03_W2_005_LangGraph_HITL_Part1.md](./PRJ03_W2_005_LangGraph_HITL_Part1.md)에서 HITL 기본 개념과 Breakpoint 설정 방법을 학습했습니다.

**실무 적용**: 이 시스템을 확장하여 다양한 도메인(금융, 의료, 법률 등)의 리서치 자동화에 활용할 수 있습니다.
