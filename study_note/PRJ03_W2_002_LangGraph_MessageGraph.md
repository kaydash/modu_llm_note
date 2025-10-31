# LangGraph MessageGraph - Reducer와 병렬 처리 패턴

## 📚 학습 목표

- **Reducer의 개념과 종류**를 이해하고 상태 관리 전략을 선택할 수 있다
- **operator.add, add_messages, Custom Reducer**를 상황에 맞게 활용할 수 있다
- **MessagesState**를 사용하여 대화 기반 시스템을 구현할 수 있다
- **병렬 처리 패턴** (Fan-out/Fan-in, 조건부 분기, 다단계 분기)을 설계할 수 있다
- **Send API**를 활용한 동적 Map-Reduce 패턴을 구현할 수 있다

## 🔑 핵심 개념

### Reducer란?

**Reducer(리듀서)**는 LangGraph의 상태 관리 핵심 메커니즘으로, 각 노드의 출력을 전체 그래프 상태에 어떻게 통합할지 정의하는 함수입니다.

#### Reducer가 필요한 이유

```python
# ❌ 문제 상황: Reducer 없이 리스트 업데이트
class State(TypedDict):
    documents: List[str]  # Reducer 지정 안 함

# 초기 상태
state = {"documents": ["doc1.pdf"]}

# node_2가 실행
def node_2(state):
    return {"documents": ["doc2.pdf", "doc3.pdf"]}

# 결과: 이전 값이 사라짐!
# state = {"documents": ["doc2.pdf", "doc3.pdf"]}  # doc1.pdf 없어짐!
```

```python
# ✅ 해결: operator.add Reducer 사용
from typing import Annotated
from operator import add

class State(TypedDict):
    documents: Annotated[List[str], add]  # Reducer 지정!

# 같은 상황에서
# 결과: 누적됨!
# state = {"documents": ["doc1.pdf", "doc2.pdf", "doc3.pdf"]}  # 모두 유지!
```

### Reducer의 세 가지 유형

#### 1. 기본 Reducer (덮어쓰기)

Reducer를 지정하지 않으면 **완전히 덮어쓰기**가 발생합니다.

**사용 시기:**
- 단순 값 (문자열, 숫자, 불린)
- 최신 값만 필요한 경우
- 예: 현재 쿼리, 최종 결과, 신뢰도 점수

**동작 방식:**
```python
새로운_상태[키] = 노드_반환값[키]  # 이전 값 무시
```

#### 2. operator.add (리스트 누적)

Python의 `+` 연산자를 사용하여 리스트를 연결합니다.

**사용 시기:**
- 리스트 형태의 데이터 누적
- 순서가 중요한 경우
- 예: 검색 결과, 메시지 히스토리

**동작 방식:**
```python
[1, 2, 3] + [4, 5] = [1, 2, 3, 4, 5]
```

#### 3. Custom Reducer (사용자 정의)

복잡한 병합 로직이 필요할 때 직접 구현합니다.

**사용 시기:**
- 중복 제거
- 정렬, 필터링
- 조건부 병합
- 최대/최소값 유지

**구현 예시:**
```python
def reduce_unique(left: list | None, right: list | None) -> list:
    """중복 제거 Reducer"""
    if not left:
        left = []
    if not right:
        right = []

    seen = set()
    result = []

    for item in left + right:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result
```

### MessagesState

**MessagesState**는 대화 기반 시스템을 위한 미리 정의된 상태 타입으로, `add_messages` Reducer가 자동으로 적용됩니다.

#### operator.add vs add_messages

| 특성 | operator.add | add_messages |
|------|-------------|--------------|
| **기본 동작** | 리스트 연결 (`+`) | 메시지 ID 기반 관리 |
| **중복 처리** | 중복 허용 | ID로 중복 감지 |
| **메시지 수정** | 불가능 | 같은 ID 메시지 업데이트 가능 |
| **메시지 포맷** | 명시적 객체만 | 다양한 포맷 자동 변환 |
| **사용 사례** | 단순 리스트 누적 | 채팅 대화 관리 |

#### MessagesState 사용법

```python
from langgraph.graph import MessagesState

# 방법 1: 기본 사용
class State(MessagesState):
    pass  # messages 필드 자동 포함

# 방법 2: 커스텀 필드 추가
class CustomState(MessagesState):
    user_id: str
    emotion: Optional[str]
    session_info: dict
```

### 병렬 처리 패턴

LangGraph는 다양한 병렬 처리 패턴을 지원하여 독립적인 작업들을 동시에 실행할 수 있습니다.

#### 1. Fan-out/Fan-in (기본 병렬)

하나의 노드에서 여러 병렬 노드로 분산 후 다시 하나로 수렴합니다.

```
START → node_a → [node_b1, node_b2, node_b3] → node_c → END
               ↘    (동시 실행)         ↙
```

#### 2. 조건부 분기

조건에 따라 선택적으로 병렬 실행합니다.

```python
def router(state) -> list[str]:
    if "weather" in state["intent"]:
        return ["weather_service", "news_service"]
    else:
        return ["news_service"]
```

#### 3. 다단계 분기

각 병렬 경로가 여러 단계를 가집니다.

```
       ┌─ fetch_a → process_a ─┐
START ─┤                        ├─ combine → END
       └─ fetch_b → process_b ─┘
```

#### 4. Map-Reduce 패턴 (Send API)

**Send API**를 사용하여 동적으로 병렬 작업을 생성합니다.

```python
from langgraph.types import Send

def mapper(state):
    # 각 아이템에 대해 Send 객체 생성
    return [
        Send("process_item", {"item": item})
        for item in state["items"]
    ]
```

**핵심 특징:**
- **동적 엣지**: 실행 시점에 병렬 작업 수 결정
- **개별 상태**: 각 병렬 작업이 독립적인 상태 사용
- **자동 수집**: 모든 병렬 작업 완료 후 자동 통합

## 🛠 환경 설정

### 1. 필수 라이브러리 설치

```bash
# LangGraph 및 관련 라이브러리
pip install langgraph langchain-openai langchain-community langchain-core

# 검색 도구
pip install tavily-python

# 환경 변수 관리
pip install python-dotenv

# 데이터 검증
pip install pydantic
```

### 2. 환경 변수 설정

`.env` 파일에 API 키를 설정합니다:

```bash
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 3. 기본 설정 코드

```python
# 환경 변수 로드
from dotenv import load_dotenv
load_dotenv()

# 기본 라이브러리
import os
import operator
from glob import glob
from pprint import pprint
from typing import TypedDict, Annotated, List, Optional, Literal

# LangChain 및 LangGraph
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.types import Send

# 데이터 검증
from pydantic import BaseModel, Field

# 시각화
from IPython.display import Image, display
```

## 💻 단계별 구현

### 1단계: Reducer 기본 사용

#### 기본 Reducer (덮어쓰기)

```python
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

# Reducer 없는 상태 정의
class DocumentState(TypedDict):
    query: str
    documents: List[str]  # Reducer 지정 안 함 = 덮어쓰기

# 노드 정의
def node_1(state: DocumentState):
    print("---Node 1---")
    return {"query": state["query"]}

def node_2(state: DocumentState):
    print("---Node 2: 새 문서 추가---")
    return {"documents": ["doc1.pdf", "doc2.pdf"]}

def node_3(state: DocumentState):
    print("---Node 3: 또 다른 문서 추가---")
    return {"documents": ["doc3.pdf"]}  # 이전 문서 사라짐!

# 그래프 구성
workflow = StateGraph(DocumentState)
workflow.add_node("node_1", node_1)
workflow.add_node("node_2", node_2)
workflow.add_node("node_3", node_3)

workflow.add_edge(START, "node_1")
workflow.add_edge("node_1", "node_2")
workflow.add_edge("node_2", "node_3")
workflow.add_edge("node_3", END)

graph = workflow.compile()

# 실행
result = graph.invoke({
    "query": "채식주의자를 위한 비건 음식을 추천해주세요."
})

print("최종 문서:", result['documents'])
# 출력: ['doc3.pdf']  # node_2의 문서들이 사라짐!
```

#### operator.add Reducer (누적)

```python
from operator import add
from typing import Annotated

# operator.add Reducer 사용
class ReducerState(TypedDict):
    query: str
    documents: Annotated[List[str], add]  # 리스트 누적!

# 노드는 동일
def node_2(state: ReducerState):
    return {"documents": ["doc1.pdf", "doc2.pdf"]}

def node_3(state: ReducerState):
    return {"documents": ["doc3.pdf"]}

# 그래프 구성 (동일)
workflow = StateGraph(ReducerState)
# ... (노드 추가 및 엣지 설정)

graph = workflow.compile()

# 실행
result = graph.invoke({
    "query": "채식주의자를 위한 비건 음식을 추천해주세요."
})

print("최종 문서:", result['documents'])
# 출력: ['doc1.pdf', 'doc2.pdf', 'doc3.pdf']  # 모두 유지!
```

### 2단계: Custom Reducer 구현

중복 제거 및 정렬 기능이 포함된 Custom Reducer를 구현합니다.

```python
from typing import Annotated, List

def reduce_unique_sorted(left: list | None, right: list | None) -> list:
    """
    중복 제거 및 내림차순 정렬 Reducer

    Args:
        left: 기존 문서 리스트
        right: 새로 추가할 문서 리스트

    Returns:
        중복이 제거되고 정렬된 문서 리스트
    """
    if not left:
        left = []
    if not right:
        right = []

    # 중복 제거
    unique_docs = list(set(left + right))

    # 내림차순 정렬
    return sorted(unique_docs, reverse=True)

# 상태 정의
class CustomState(TypedDict):
    query: str
    documents: Annotated[List[str], reduce_unique_sorted]

# 노드 정의
def node_2(state: CustomState):
    return {"documents": ["doc1.pdf", "doc2.pdf", "doc3.pdf"]}

def node_3(state: CustomState):
    return {"documents": ["doc2.pdf", "doc4.pdf"]}  # doc2.pdf 중복

# 그래프 구성 및 실행
workflow = StateGraph(CustomState)
workflow.add_node("node_1", lambda s: {"query": s["query"]})
workflow.add_node("node_2", node_2)
workflow.add_node("node_3", node_3)

workflow.add_edge(START, "node_1")
workflow.add_edge("node_1", "node_2")
workflow.add_edge("node_2", "node_3")
workflow.add_edge("node_3", END)

graph = workflow.compile()

result = graph.invoke({
    "query": "채식주의자를 위한 비건 음식을 추천해주세요."
})

print("최종 문서:", result['documents'])
# 출력: ['doc4.pdf', 'doc3.pdf', 'doc2.pdf', 'doc1.pdf']
# 중복 제거 + 정렬됨!
```

### 3단계: MessagesState 활용

대화 기반 챗봇을 MessagesState로 구현합니다.

#### 기본 사용

```python
from langgraph.graph import MessagesState
from langchain_openai import ChatOpenAI

# MessagesState 상속
class ChatState(MessagesState):
    pass  # messages 필드 자동 포함

# LLM 인스턴스
llm = ChatOpenAI(model="gpt-4.1-mini")

# 챗봇 노드
def chatbot(state: ChatState):
    # LLM에 messages 전달하여 응답 생성
    return {"messages": [llm.invoke(state["messages"])]}

# 그래프 구성
builder = StateGraph(ChatState)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()

# 실행
for event in graph.stream(
    {"messages": [("user", "안녕하세요!")]},
    stream_mode="values"
):
    pprint(event['messages'])
```

#### 커스텀 필드 추가

```python
from typing import Optional

# 감정 분석 기능 추가
class EmotionChatState(MessagesState):
    emotion: Optional[str]  # 감정 상태 추적

llm = ChatOpenAI(model="gpt-4.1-mini")

# 감정 분석 노드
def analyze_emotion(state: EmotionChatState):
    """사용자 메시지의 감정 분석"""
    user_message = state["messages"][-1].content

    prompt = f"""
    사용자 메시지의 감정 상태를 파악하세요.
    가능한 감정: 행복, 슬픔, 화남, 중립

    메시지: {user_message}

    감정만 한 단어로:
    """

    emotion = llm.invoke(prompt).content.strip()
    return {"emotion": emotion}

# 감정 기반 응답 노드
def respond_with_emotion(state: EmotionChatState):
    """감정에 맞춰 응답 생성"""
    emotion = state.get("emotion", "중립")

    prompt = f"""
    사용자의 감정({emotion})을 고려하여 공감하며 응답하세요.

    대화 히스토리:
    {state["messages"]}

    응답:
    """

    response = llm.invoke(prompt)
    return {"messages": [response]}

# 그래프 구성
builder = StateGraph(EmotionChatState)
builder.add_node("analyze_emotion", analyze_emotion)
builder.add_node("respond", respond_with_emotion)

builder.add_edge(START, "analyze_emotion")
builder.add_edge("analyze_emotion", "respond")
builder.add_edge("respond", END)

graph = builder.compile()

# 실행
result = graph.invoke({
    "messages": [("user", "오늘 정말 힘든 하루였어요...")]
})

print(f"감정: {result['emotion']}")
print(f"응답: {result['messages'][-1].content}")
```

### 4단계: 병렬 처리 - Fan-out/Fan-in

여러 데이터 소스에서 동시에 검색하는 시스템을 구현합니다.

```python
import operator
from typing import Annotated

class SearchState(TypedDict):
    query: str
    results: Annotated[list[str], operator.add]

# 병렬 검색 노드들
def search_db(state: SearchState):
    """데이터베이스 검색"""
    print(f"📊 DB 검색: {state['query']}")
    return {"results": ["DB: 데이터 1", "DB: 데이터 2"]}

def search_web(state: SearchState):
    """웹 검색"""
    print(f"🌐 웹 검색: {state['query']}")
    return {"results": ["웹: 정보 1", "웹: 정보 2"]}

def search_api(state: SearchState):
    """API 검색"""
    print(f"🔌 API 검색: {state['query']}")
    return {"results": ["API: 결과 1", "API: 결과 2"]}

def aggregate(state: SearchState):
    """결과 통합"""
    print(f"📋 총 {len(state['results'])}개 결과 수집 완료")
    return {}

# 그래프 구성
workflow = StateGraph(SearchState)

# 노드 추가
workflow.add_node("search_db", search_db)
workflow.add_node("search_web", search_web)
workflow.add_node("search_api", search_api)
workflow.add_node("aggregate", aggregate)

# Fan-out: START → 3개 병렬 노드
workflow.add_edge(START, "search_db")
workflow.add_edge(START, "search_web")
workflow.add_edge(START, "search_api")

# Fan-in: 3개 노드 → aggregate
workflow.add_edge("search_db", "aggregate")
workflow.add_edge("search_web", "aggregate")
workflow.add_edge("search_api", "aggregate")

workflow.add_edge("aggregate", END)

graph = workflow.compile()

# 실행
result = graph.invoke({"query": "LangGraph 튜토리얼"})
for r in result["results"]:
    print(f"  - {r}")
```

### 5단계: 조건부 병렬 분기

사용자 의도에 따라 선택적으로 서비스를 병렬 실행합니다.

```python
from typing import Literal

class IntentState(TypedDict):
    messages: Annotated[list, operator.add]
    user_intent: str

# 서비스 노드들
def greet_service(state: IntentState):
    return {"messages": ["안녕하세요!"]}

def weather_service(state: IntentState):
    return {"messages": ["오늘 날씨는 맑습니다."]}

def news_service(state: IntentState):
    return {"messages": ["최신 뉴스를 전달합니다."]}

def end_service(state: IntentState):
    return {"messages": ["서비스를 종료합니다."]}

# 라우팅 함수
def route_services(state: IntentState) -> list[Literal["weather_service", "news_service"]]:
    """의도에 따라 실행할 서비스 선택"""
    intent = state["user_intent"]

    if "weather" in intent and "news" in intent:
        return ["weather_service", "news_service"]
    elif "weather" in intent:
        return ["weather_service"]
    elif "news" in intent:
        return ["news_service"]
    else:
        return []

# 그래프 구성
builder = StateGraph(IntentState)

builder.add_node("greet_service", greet_service)
builder.add_node("weather_service", weather_service)
builder.add_node("news_service", news_service)
builder.add_node("end_service", end_service)

builder.add_edge(START, "greet_service")

# 조건부 엣지: greet → 선택적 서비스들
builder.add_conditional_edges(
    "greet_service",
    route_services,
    ["weather_service", "news_service"]
)

builder.add_edge("weather_service", "end_service")
builder.add_edge("news_service", "end_service")
builder.add_edge("end_service", END)

graph = builder.compile()

# 테스트 1: 날씨 + 뉴스
print("=== weather_news ===")
result = graph.invoke({"user_intent": "weather_news"})
print(result["messages"])

# 테스트 2: 뉴스만
print("\n=== news ===")
result = graph.invoke({"user_intent": "news"})
print(result["messages"])
```

### 6단계: Send API로 동적 Map-Reduce 구현

URL 개수에 관계없이 동적으로 병렬 스크래핑을 수행합니다.

```python
from langgraph.types import Send
import time

# 전체 상태
class ScrapingState(TypedDict):
    urls: List[str]
    scraped_data: Annotated[List[dict], operator.add]

# 개별 URL 상태
class URLState(TypedDict):
    url: str

# 1. URL 목록 준비 (Map 시작점)
def prepare_urls(state: ScrapingState):
    """스크래핑할 URL 목록 확인"""
    print(f"📋 총 {len(state['urls'])}개 URL 준비")
    return {}

# 2. Send를 사용한 동적 분배
def distribute_urls(state: ScrapingState):
    """각 URL을 별도 노드로 분배"""
    print(f"🔀 {len(state['urls'])}개 URL을 병렬로 분배...")

    # 각 URL에 대해 Send 객체 생성
    return [
        Send("scrape_url", {"url": url})
        for url in state["urls"]
    ]

# 3. 개별 스크래핑 노드 (병렬 실행)
def scrape_url(state: URLState) -> ScrapingState:
    """단일 URL 스크래핑"""
    url = state["url"]
    print(f"🌐 스크래핑: {url}")

    # 실제로는 웹 스크래핑 수행
    time.sleep(0.5)  # 시뮬레이션

    # 결과 반환 (전체 상태에 추가됨)
    return {
        "scraped_data": [{
            "url": url,
            "title": f"Title from {url}",
            "content": f"Content from {url}"
        }]
    }

# 4. 결과 통합
def aggregate_results(state: ScrapingState):
    """스크래핑 결과 통합"""
    print(f"✅ {len(state['scraped_data'])}개 결과 수집 완료")
    return {}

# 그래프 구성
builder = StateGraph(ScrapingState)

builder.add_node("prepare_urls", prepare_urls)
builder.add_node("scrape_url", scrape_url)
builder.add_node("aggregate_results", aggregate_results)

builder.add_edge(START, "prepare_urls")

# 조건부 엣지에서 Send 사용
builder.add_conditional_edges(
    "prepare_urls",
    distribute_urls,  # Send 객체 리스트 반환
    ["scrape_url"]
)

builder.add_edge("scrape_url", "aggregate_results")
builder.add_edge("aggregate_results", END)

graph = builder.compile()

# 실행 - URL 개수는 동적!
result = graph.invoke({
    "urls": [
        "https://example.com",
        "https://example.org",
        "https://example.net",
        "https://example.io"
    ]
})

print("\n=== 스크래핑 결과 ===")
for data in result["scraped_data"]:
    print(f"✅ {data['url']}: {data['title']}")
```

## 🎯 실습 문제

### 문제 1: Custom Reducer - 우선순위 기반 문서 관리 (난이도: ⭐⭐)

**요구사항:**
문서에 우선순위를 부여하고, 높은 우선순위 문서를 앞쪽에 유지하는 Custom Reducer를 구현하세요.

1. 문서 구조: `{"name": str, "priority": int}`
2. Reducer 동작:
   - 기존 문서와 새 문서를 병합
   - 우선순위 기준 내림차순 정렬
   - 같은 이름의 문서는 더 높은 우선순위로 업데이트

**힌트:**
```python
def reduce_priority_docs(left: list | None, right: list | None) -> list:
    # 문서 병합 로직
    # 중복 제거 (같은 이름)
    # 우선순위 정렬
    pass
```

---

### 문제 2: MessagesState - 다국어 채팅봇 (난이도: ⭐⭐⭐)

**요구사항:**
사용자의 언어를 자동 감지하고 해당 언어로 응답하는 채팅봇을 구현하세요.

1. State 구성:
   - `MessagesState` 상속
   - `detected_language` 필드 추가
   - `translation_enabled` 필드 추가

2. 노드 구성:
   - `detect_language`: 언어 감지
   - `translate_if_needed`: 필요시 번역
   - `generate_response`: 응답 생성

3. 지원 언어: 한국어, 영어, 일본어

---

### 문제 3: 병렬 처리 - 멀티 소스 RAG 시스템 (난이도: ⭐⭐⭐)

**요구사항:**
여러 데이터 소스를 병렬로 검색하고 결과를 통합하는 RAG 시스템을 구현하세요.

1. 데이터 소스:
   - 로컬 벡터 DB (Chroma)
   - 웹 검색 (Tavily)
   - 내부 API (Mock)

2. 병렬 처리:
   - 3개 소스 동시 검색
   - 각 소스별 신뢰도 점수 계산
   - 신뢰도 기반 결과 가중 통합

3. 최종 응답:
   - 통합된 컨텍스트로 답변 생성
   - 출처 정보 포함

---

### 문제 4: Send API - 동적 데이터 파이프라인 (난이도: ⭐⭐⭐⭐)

**요구사항:**
다양한 형식의 파일들을 병렬로 처리하는 데이터 파이프라인을 Send API로 구현하세요.

1. 지원 파일 형식: PDF, DOCX, TXT, CSV

2. 처리 흐름:
   - 파일 목록 수집
   - 파일 형식별로 다른 처리 노드로 라우팅
   - 각 파일 병렬 처리 (텍스트 추출)
   - 추출된 텍스트 통합 및 요약

3. Send API 활용:
   - 파일 형식에 따라 동적으로 노드 선택
   - 각 파일에 개별 상태 전달
   - 병렬 처리 후 자동 통합

**힌트:**
```python
def route_by_file_type(state: PipelineState):
    return [
        Send(
            f"process_{get_file_type(file)}",
            {"file_path": file}
        )
        for file in state["files"]
    ]
```

## ✅ 솔루션 예시

### 문제 1 솔루션: 우선순위 기반 문서 관리

```python
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END

def reduce_priority_docs(left: list | None, right: list | None) -> list:
    """
    우선순위 기반 문서 병합 Reducer

    - 중복 문서는 더 높은 우선순위로 업데이트
    - 우선순위 내림차순 정렬
    """
    if not left:
        left = []
    if not right:
        right = []

    # 문서 병합 (이름 기준 최고 우선순위 유지)
    doc_dict = {}

    for doc in left + right:
        name = doc["name"]
        if name not in doc_dict or doc["priority"] > doc_dict[name]["priority"]:
            doc_dict[name] = doc

    # 우선순위 내림차순 정렬
    result = sorted(doc_dict.values(), key=lambda x: x["priority"], reverse=True)

    return result

# 상태 정의
class DocumentState(TypedDict):
    query: str
    documents: Annotated[List[dict], reduce_priority_docs]

# 노드 정의
def search_primary(state: DocumentState):
    """주요 문서 검색"""
    return {
        "documents": [
            {"name": "doc1.pdf", "priority": 5},
            {"name": "doc2.pdf", "priority": 3},
            {"name": "doc3.pdf", "priority": 4}
        ]
    }

def search_secondary(state: DocumentState):
    """추가 문서 검색"""
    return {
        "documents": [
            {"name": "doc2.pdf", "priority": 8},  # 우선순위 업데이트!
            {"name": "doc4.pdf", "priority": 7},
            {"name": "doc5.pdf", "priority": 2}
        ]
    }

def display_results(state: DocumentState):
    """결과 출력"""
    print("\n=== 최종 문서 목록 (우선순위 순) ===")
    for i, doc in enumerate(state["documents"], 1):
        print(f"{i}. {doc['name']} (우선순위: {doc['priority']})")
    return {}

# 그래프 구성
workflow = StateGraph(DocumentState)

workflow.add_node("search_primary", search_primary)
workflow.add_node("search_secondary", search_secondary)
workflow.add_node("display_results", display_results)

workflow.add_edge(START, "search_primary")
workflow.add_edge("search_primary", "search_secondary")
workflow.add_edge("search_secondary", "display_results")
workflow.add_edge("display_results", END)

graph = workflow.compile()

# 실행
result = graph.invoke({"query": "test"})

# 예상 출력:
# 1. doc2.pdf (우선순위: 8)  # 업데이트됨!
# 2. doc4.pdf (우선순위: 7)
# 3. doc1.pdf (우선순위: 5)
# 4. doc3.pdf (우선순위: 4)
# 5. doc5.pdf (우선순위: 2)
```

### 문제 2 솔루션: 다국어 채팅봇

```python
from typing import Optional
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_openai import ChatOpenAI

# 상태 정의
class MultilingualChatState(MessagesState):
    detected_language: Optional[str]
    translation_enabled: bool = False

llm = ChatOpenAI(model="gpt-4.1-mini")

# 언어 감지 노드
def detect_language(state: MultilingualChatState):
    """사용자 메시지의 언어 감지"""
    user_message = state["messages"][-1].content

    prompt = f"""
    다음 텍스트의 언어를 감지하세요.
    가능한 언어: korean, english, japanese

    텍스트: {user_message}

    언어만 소문자로 응답:
    """

    language = llm.invoke(prompt).content.strip().lower()
    print(f"🌍 감지된 언어: {language}")

    return {"detected_language": language}

# 번역 노드 (필요시)
def translate_if_needed(state: MultilingualChatState):
    """영어가 아니면 영어로 번역"""
    language = state.get("detected_language", "english")

    if language == "english":
        print("✅ 번역 불필요")
        return {}

    user_message = state["messages"][-1].content

    prompt = f"""
    다음 {language} 텍스트를 영어로 번역하세요:

    {user_message}

    번역만 출력:
    """

    translated = llm.invoke(prompt).content
    print(f"🔄 번역 완료: {translated[:50]}...")

    # 번역된 메시지를 추가
    return {
        "messages": [("system", f"[Translated from {language}] {translated}")],
        "translation_enabled": True
    }

# 응답 생성 노드
def generate_response(state: MultilingualChatState):
    """응답 생성 (원래 언어로)"""
    language = state.get("detected_language", "english")
    translation_enabled = state.get("translation_enabled", False)

    # LLM에 메시지 전달
    ai_response = llm.invoke(state["messages"])

    # 원래 언어로 번역 (필요시)
    if translation_enabled and language != "english":
        prompt = f"""
        다음 영어 텍스트를 {language}로 번역하세요:

        {ai_response.content}

        번역만 출력:
        """

        final_response = llm.invoke(prompt).content
        print(f"🔄 응답을 {language}로 번역")
    else:
        final_response = ai_response.content

    return {"messages": [("assistant", final_response)]}

# 그래프 구성
builder = StateGraph(MultilingualChatState)

builder.add_node("detect_language", detect_language)
builder.add_node("translate_if_needed", translate_if_needed)
builder.add_node("generate_response", generate_response)

builder.add_edge(START, "detect_language")
builder.add_edge("detect_language", "translate_if_needed")
builder.add_edge("translate_if_needed", "generate_response")
builder.add_edge("generate_response", END)

graph = builder.compile()

# 테스트
test_messages = [
    "안녕하세요! LangGraph에 대해 알려주세요.",
    "Hello! Tell me about LangGraph.",
    "こんにちは！LangGraphについて教えてください。"
]

for msg in test_messages:
    print(f"\n{'='*60}")
    print(f"입력: {msg}")
    result = graph.invoke({"messages": [("user", msg)]})
    print(f"응답: {result['messages'][-1].content}")
```

### 문제 3 솔루션: 멀티 소스 RAG 시스템

```python
import operator
from typing import Annotated, List, Optional
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults

class MultiSourceRAGState(TypedDict):
    query: str
    vector_results: Annotated[List[dict], operator.add]
    web_results: Annotated[List[dict], operator.add]
    api_results: Annotated[List[dict], operator.add]
    final_context: Optional[str]
    answer: Optional[str]

llm = ChatOpenAI(model="gpt-4.1-mini")

# 병렬 검색 노드들
def search_vector_db(state: MultiSourceRAGState):
    """벡터 DB 검색"""
    print("📊 벡터 DB 검색 중...")

    # 실제로는 Chroma DB 사용
    # db = Chroma(embedding_function=OpenAIEmbeddings())
    # results = db.similarity_search(state["query"], k=3)

    # Mock 결과
    results = [
        {"content": "벡터 DB 결과 1", "confidence": 0.9},
        {"content": "벡터 DB 결과 2", "confidence": 0.85}
    ]

    return {"vector_results": results}

def search_web(state: MultiSourceRAGState):
    """웹 검색"""
    print("🌐 웹 검색 중...")

    search_tool = TavilySearchResults(max_results=2)
    results = search_tool.invoke(state["query"])

    # 신뢰도 추가
    web_results = [
        {"content": r["content"], "confidence": 0.75}
        for r in results
    ]

    return {"web_results": web_results}

def search_api(state: MultiSourceRAGState):
    """내부 API 검색"""
    print("🔌 API 검색 중...")

    # Mock API 결과
    results = [
        {"content": "API 결과 1", "confidence": 0.8}
    ]

    return {"api_results": results}

# 결과 통합 노드
def integrate_results(state: MultiSourceRAGState):
    """검색 결과 통합 및 가중 평균"""
    all_results = (
        state.get("vector_results", []) +
        state.get("web_results", []) +
        state.get("api_results", [])
    )

    # 신뢰도 기준 정렬
    sorted_results = sorted(all_results, key=lambda x: x["confidence"], reverse=True)

    # 상위 5개 선택
    top_results = sorted_results[:5]

    # 컨텍스트 구성
    context_parts = []
    for i, r in enumerate(top_results, 1):
        context_parts.append(
            f"{i}. (신뢰도: {r['confidence']:.2f}) {r['content']}"
        )

    final_context = "\n".join(context_parts)

    print(f"✅ {len(top_results)}개 결과 통합 완료")

    return {"final_context": final_context}

# 답변 생성 노드
def generate_answer(state: MultiSourceRAGState):
    """통합된 컨텍스트로 답변 생성"""
    prompt = f"""
    다음 정보들을 바탕으로 질문에 답변하세요.

    질문: {state['query']}

    참고 자료:
    {state['final_context']}

    답변 (출처 포함):
    """

    answer = llm.invoke(prompt).content
    return {"answer": answer}

# 그래프 구성
builder = StateGraph(MultiSourceRAGState)

builder.add_node("search_vector_db", search_vector_db)
builder.add_node("search_web", search_web)
builder.add_node("search_api", search_api)
builder.add_node("integrate_results", integrate_results)
builder.add_node("generate_answer", generate_answer)

# Fan-out: 병렬 검색
builder.add_edge(START, "search_vector_db")
builder.add_edge(START, "search_web")
builder.add_edge(START, "search_api")

# Fan-in: 통합
builder.add_edge("search_vector_db", "integrate_results")
builder.add_edge("search_web", "integrate_results")
builder.add_edge("search_api", "integrate_results")

builder.add_edge("integrate_results", "generate_answer")
builder.add_edge("generate_answer", END)

graph = builder.compile()

# 실행
result = graph.invoke({"query": "LangGraph의 주요 기능은?"})

print("\n=== 최종 답변 ===")
print(result["answer"])
```

### 문제 4 솔루션: 동적 데이터 파이프라인

```python
from langgraph.types import Send
from pathlib import Path
from typing import Dict

# 전체 상태
class PipelineState(TypedDict):
    files: List[str]
    extracted_texts: Annotated[List[Dict[str, str]], operator.add]
    final_summary: Optional[str]

# 개별 파일 상태
class FileState(TypedDict):
    file_path: str

def get_file_type(file_path: str) -> str:
    """파일 확장자 추출"""
    return Path(file_path).suffix[1:].lower()

# 1. 파일 목록 수집
def collect_files(state: PipelineState):
    """처리할 파일 목록 확인"""
    print(f"📁 총 {len(state['files'])}개 파일 발견")
    return {}

# 2. 동적 라우팅 (Send 사용)
def route_by_file_type(state: PipelineState):
    """파일 형식별로 적절한 노드로 라우팅"""
    print("🔀 파일 형식별 라우팅...")

    sends = []
    for file_path in state["files"]:
        file_type = get_file_type(file_path)
        node_name = f"process_{file_type}"

        sends.append(
            Send(node_name, {"file_path": file_path})
        )
        print(f"  → {file_path} → {node_name}")

    return sends

# 3. 파일 형식별 처리 노드들
def process_pdf(state: FileState) -> PipelineState:
    """PDF 파일 처리"""
    file_path = state["file_path"]
    print(f"📄 PDF 처리: {file_path}")

    # 실제로는 PyPDF2, pdfplumber 등 사용
    text = f"[PDF 텍스트 from {file_path}]"

    return {
        "extracted_texts": [{
            "file": file_path,
            "type": "pdf",
            "text": text
        }]
    }

def process_docx(state: FileState) -> PipelineState:
    """DOCX 파일 처리"""
    file_path = state["file_path"]
    print(f"📝 DOCX 처리: {file_path}")

    # 실제로는 python-docx 사용
    text = f"[DOCX 텍스트 from {file_path}]"

    return {
        "extracted_texts": [{
            "file": file_path,
            "type": "docx",
            "text": text
        }]
    }

def process_txt(state: FileState) -> PipelineState:
    """TXT 파일 처리"""
    file_path = state["file_path"]
    print(f"📃 TXT 처리: {file_path}")

    # 실제로는 파일 읽기
    text = f"[TXT 텍스트 from {file_path}]"

    return {
        "extracted_texts": [{
            "file": file_path,
            "type": "txt",
            "text": text
        }]
    }

def process_csv(state: FileState) -> PipelineState:
    """CSV 파일 처리"""
    file_path = state["file_path"]
    print(f"📊 CSV 처리: {file_path}")

    # 실제로는 pandas 사용
    text = f"[CSV 데이터 from {file_path}]"

    return {
        "extracted_texts": [{
            "file": file_path,
            "type": "csv",
            "text": text
        }]
    }

# 4. 통합 및 요약
def summarize_all(state: PipelineState):
    """추출된 텍스트 통합 및 요약"""
    print(f"\n✅ {len(state['extracted_texts'])}개 파일 처리 완료")

    # 모든 텍스트 통합
    all_texts = "\n\n".join([
        f"[{item['type'].upper()}] {item['file']}\n{item['text']}"
        for item in state["extracted_texts"]
    ])

    # LLM으로 요약
    prompt = f"""
    다음 문서들의 내용을 종합하여 요약하세요:

    {all_texts}

    종합 요약:
    """

    summary = llm.invoke(prompt).content

    return {"final_summary": summary}

# 그래프 구성
builder = StateGraph(PipelineState)

builder.add_node("collect_files", collect_files)
builder.add_node("process_pdf", process_pdf)
builder.add_node("process_docx", process_docx)
builder.add_node("process_txt", process_txt)
builder.add_node("process_csv", process_csv)
builder.add_node("summarize_all", summarize_all)

builder.add_edge(START, "collect_files")

# 조건부 엣지에서 Send 사용
builder.add_conditional_edges(
    "collect_files",
    route_by_file_type,
    ["process_pdf", "process_docx", "process_txt", "process_csv"]
)

# 모든 처리 노드 → 요약
builder.add_edge("process_pdf", "summarize_all")
builder.add_edge("process_docx", "summarize_all")
builder.add_edge("process_txt", "summarize_all")
builder.add_edge("process_csv", "summarize_all")

builder.add_edge("summarize_all", END)

graph = builder.compile()

# 실행
result = graph.invoke({
    "files": [
        "report.pdf",
        "notes.docx",
        "readme.txt",
        "data.csv",
        "summary.pdf",
        "analysis.txt"
    ]
})

print("\n=== 최종 요약 ===")
print(result["final_summary"])
```

## 🚀 실무 활용 예시

### 예시 1: 고급 문서 팩트체크 시스템

실무에서 사용할 수 있는 종합적인 팩트체크 시스템입니다.

```python
from typing import Annotated, List, Optional
from pydantic import BaseModel, Field
from langgraph.types import Send
import operator

# 팩트체크 결과 모델
class FactCheckResult(BaseModel):
    sentence: str
    score: float
    reasoning: str
    sources: List[str]

# 전체 상태
class FactCheckState(TypedDict):
    query: str
    search_results: Optional[str]
    summary: Optional[str]
    fact_checks: Annotated[List[FactCheckResult], operator.add]
    overall_reliability: Optional[float]

# 개별 문장 상태
class SentenceState(TypedDict):
    sentence: str
    search_results: str  # 참고 자료

llm = ChatOpenAI(model="gpt-4.1-mini")

# 1. 검색 노드
def search_information(state: FactCheckState):
    """주제에 대한 정보 검색"""
    search_tool = TavilySearchResults(max_results=5)
    query = state["query"]

    print(f"🔍 검색: {query}")

    results = search_tool.invoke(query)

    # 검색 결과 텍스트 통합
    search_text = "\n\n".join([
        f"[출처 {i+1}] {r['content']}"
        for i, r in enumerate(results)
    ])

    return {"search_results": search_text}

# 2. 요약 노드
def generate_summary(state: FactCheckState):
    """검색 결과 요약"""
    prompt = f"""
    다음 정보를 3-4개의 핵심 문장으로 요약하세요.
    각 문장은 한 줄씩 분리하세요.

    정보:
    {state['search_results']}

    요약:
    """

    summary = llm.invoke(prompt).content
    print("📝 요약 완료")

    return {"summary": summary}

# 3. 문장 분배 노드 (Send 사용)
def distribute_sentences(state: FactCheckState):
    """각 문장을 병렬 팩트체크"""
    if not state["summary"]:
        return []

    sentences = [s.strip() for s in state["summary"].split("\n") if s.strip()]

    print(f"🔀 {len(sentences)}개 문장 병렬 팩트체크")

    return [
        Send(
            "fact_check_sentence",
            {
                "sentence": sentence,
                "search_results": state["search_results"]
            }
        )
        for sentence in sentences
    ]

# 4. 개별 팩트체크 노드
def fact_check_sentence(state: SentenceState) -> FactCheckState:
    """단일 문장 팩트체크"""
    sentence = state["sentence"]
    search_results = state["search_results"]

    print(f"✅ 팩트체크: {sentence[:50]}...")

    prompt = f"""
    다음 문장의 사실 여부를 검증하세요.

    문장: {sentence}

    참고 자료:
    {search_results}

    다음 정보를 JSON 형식으로 제공하세요:
    {{
        "score": 0.0-1.0 사이의 신뢰도 점수,
        "reasoning": "평가 근거 (2-3문장)",
        "sources": ["관련 출처 번호들"]
    }}
    """

    fact_check_llm = llm.with_structured_output(FactCheckResult)

    try:
        result = fact_check_llm.invoke(prompt)
        result.sentence = sentence
    except:
        # 파싱 실패 시 기본값
        result = FactCheckResult(
            sentence=sentence,
            score=0.5,
            reasoning="평가 실패",
            sources=[]
        )

    return {"fact_checks": [result]}

# 5. 종합 평가 노드
def calculate_overall_reliability(state: FactCheckState):
    """전체 신뢰도 계산"""
    fact_checks = state.get("fact_checks", [])

    if not fact_checks:
        return {"overall_reliability": 0.0}

    # 평균 신뢰도
    avg_score = sum(fc.score for fc in fact_checks) / len(fact_checks)

    print(f"\n📊 전체 신뢰도: {avg_score:.2f}")

    # 상세 결과 출력
    print("\n=== 문장별 팩트체크 결과 ===")
    for i, fc in enumerate(fact_checks, 1):
        print(f"\n{i}. {fc.sentence}")
        print(f"   신뢰도: {fc.score:.2f}")
        print(f"   근거: {fc.reasoning}")
        print(f"   출처: {', '.join(fc.sources)}")

    return {"overall_reliability": avg_score}

# 그래프 구성
builder = StateGraph(FactCheckState)

builder.add_node("search_information", search_information)
builder.add_node("generate_summary", generate_summary)
builder.add_node("fact_check_sentence", fact_check_sentence)
builder.add_node("calculate_overall_reliability", calculate_overall_reliability)

builder.add_edge(START, "search_information")
builder.add_edge("search_information", "generate_summary")

# 조건부 엣지에서 Send 사용
builder.add_conditional_edges(
    "generate_summary",
    distribute_sentences,
    ["fact_check_sentence"]
)

builder.add_edge("fact_check_sentence", "calculate_overall_reliability")
builder.add_edge("calculate_overall_reliability", END)

graph = builder.compile()

# 실행
result = graph.invoke({
    "query": "인공지능의 환경 영향에 대해 알려주세요"
})

print(f"\n{'='*60}")
print(f"요약:\n{result['summary']}")
print(f"\n전체 신뢰도: {result['overall_reliability']:.2%}")
```

### 예시 2: 실시간 멀티 에이전트 뉴스 분석 시스템

여러 에이전트가 병렬로 뉴스를 분석하는 시스템입니다.

```python
from typing import Annotated, List
from enum import Enum

class NewsTopic(str, Enum):
    TECHNOLOGY = "technology"
    BUSINESS = "business"
    POLITICS = "politics"
    SCIENCE = "science"

# 전체 상태
class NewsAnalysisState(TypedDict):
    topics: List[NewsTopic]
    topic_analyses: Annotated[List[dict], operator.add]
    summary_report: Optional[str]
    trending_keywords: Optional[List[str]]

# 개별 토픽 상태
class TopicState(TypedDict):
    topic: NewsTopic

llm = ChatOpenAI(model="gpt-4.1-mini")

# 1. 토픽 선택 노드
def select_topics(state: NewsAnalysisState):
    """분석할 토픽 확인"""
    topics = state.get("topics", [NewsTopic.TECHNOLOGY])
    print(f"📰 {len(topics)}개 토픽 분석 시작")
    return {}

# 2. 토픽별 분배 (Send 사용)
def distribute_topics(state: NewsAnalysisState):
    """각 토픽을 병렬 분석"""
    return [
        Send("analyze_topic", {"topic": topic})
        for topic in state["topics"]
    ]

# 3. 개별 토픽 분석 노드
def analyze_topic(state: TopicState) -> NewsAnalysisState:
    """단일 토픽 뉴스 분석"""
    topic = state["topic"]
    print(f"🔍 {topic.value} 뉴스 분석 중...")

    # 실제로는 뉴스 API 호출
    search_tool = TavilySearchResults(max_results=5)
    query = f"latest {topic.value} news"

    results = search_tool.invoke(query)

    # LLM으로 요약 및 분석
    prompt = f"""
    다음 {topic.value} 뉴스들을 분석하세요:

    {chr(10).join([r['content'] for r in results[:3]])}

    다음 항목을 JSON으로 제공:
    {{
        "summary": "핵심 요약 (3문장)",
        "sentiment": "positive/neutral/negative",
        "key_events": ["주요 이벤트 3개"],
        "keywords": ["키워드 5개"]
    }}
    """

    analysis = llm.invoke(prompt).content

    # 간단한 파싱 (실제로는 structured output 사용)
    return {
        "topic_analyses": [{
            "topic": topic.value,
            "analysis": analysis,
            "timestamp": "2025-10-31"
        }]
    }

# 4. 트렌드 키워드 추출 노드
def extract_trending_keywords(state: NewsAnalysisState):
    """모든 토픽에서 공통 키워드 추출"""
    analyses = state.get("topic_analyses", [])

    # 모든 분석 통합
    all_analyses = "\n\n".join([
        f"[{a['topic']}]\n{a['analysis']}"
        for a in analyses
    ])

    prompt = f"""
    다음 뉴스 분석들에서 가장 많이 언급된 트렌드 키워드 10개를 추출하세요:

    {all_analyses}

    키워드만 쉼표로 구분:
    """

    keywords_text = llm.invoke(prompt).content
    keywords = [k.strip() for k in keywords_text.split(",")][:10]

    print(f"🔥 트렌딩 키워드: {', '.join(keywords[:5])}...")

    return {"trending_keywords": keywords}

# 5. 종합 리포트 생성 노드
def generate_summary_report(state: NewsAnalysisState):
    """최종 종합 리포트"""
    analyses = state.get("topic_analyses", [])
    keywords = state.get("trending_keywords", [])

    prompt = f"""
    다음 토픽별 분석을 바탕으로 종합 뉴스 리포트를 작성하세요:

    토픽별 분석:
    {chr(10).join([f"- {a['topic']}: {a['analysis'][:200]}..." for a in analyses])}

    트렌딩 키워드: {', '.join(keywords)}

    종합 리포트 (5문장):
    """

    report = llm.invoke(prompt).content

    return {"summary_report": report}

# 그래프 구성
builder = StateGraph(NewsAnalysisState)

builder.add_node("select_topics", select_topics)
builder.add_node("analyze_topic", analyze_topic)
builder.add_node("extract_trending_keywords", extract_trending_keywords)
builder.add_node("generate_summary_report", generate_summary_report)

builder.add_edge(START, "select_topics")

builder.add_conditional_edges(
    "select_topics",
    distribute_topics,
    ["analyze_topic"]
)

builder.add_edge("analyze_topic", "extract_trending_keywords")
builder.add_edge("extract_trending_keywords", "generate_summary_report")
builder.add_edge("generate_summary_report", END)

graph = builder.compile()

# 실행
result = graph.invoke({
    "topics": [
        NewsTopic.TECHNOLOGY,
        NewsTopic.BUSINESS,
        NewsTopic.SCIENCE
    ]
})

print("\n" + "="*60)
print("📊 종합 뉴스 리포트")
print("="*60)
print(f"\n{result['summary_report']}")
print(f"\n🔥 트렌딩 키워드:")
print(", ".join(result['trending_keywords']))
```

## 📖 참고 자료

### 공식 문서
- [LangGraph Reducer 가이드](https://langchain-ai.github.io/langgraph/how-tos/state-reducers/)
- [MessagesState API 레퍼런스](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.MessagesState)
- [Send API 문서](https://langchain-ai.github.io/langgraph/how-tos/map-reduce/)
- [병렬 처리 패턴](https://langchain-ai.github.io/langgraph/how-tos/branching/)

### 추가 학습 자료
- [Map-Reduce 패턴 실전 예제](https://github.com/langchain-ai/langgraph/tree/main/examples/map-reduce)
- [Custom Reducer 고급 활용](https://langchain-ai.github.io/langgraph/tutorials/custom-reducers/)
- [병렬 처리 최적화 가이드](https://python.langchain.com/docs/langgraph/performance)

### 관련 기술 스택
- **Pydantic**: 데이터 검증 - [Pydantic 문서](https://docs.pydantic.dev/)
- **Tavily Search**: 웹 검색 API - [Tavily 문서](https://tavily.com/docs)
- **Operator 모듈**: Python 내장 연산자 - [Python 공식 문서](https://docs.python.org/3/library/operator.html)

### 병렬 처리 패턴 참고
- **Fan-out/Fan-in 패턴**: [Martin Fowler's Enterprise Patterns](https://www.enterpriseintegrationpatterns.com/patterns/messaging/BroadcastAggregate.html)
- **Map-Reduce**: [Google의 MapReduce 논문](https://research.google/pubs/pub62/)

---

**다음 단계:**
- 복잡한 워크플로우에서 Reducer와 Send API 조합하기
- 에러 처리 및 재시도 로직 구현
- 성능 모니터링 및 병렬 처리 최적화
- Human-in-the-Loop 패턴과 통합하기
