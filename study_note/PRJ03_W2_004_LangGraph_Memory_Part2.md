# LangGraph 메모리 관리 - Part 2: 장기 메모리

## 📚 학습 목표

이 가이드를 완료하면 다음을 할 수 있습니다:

- **장기 메모리 (Long-term Memory)**의 개념과 필요성을 이해할 수 있다
- **InMemoryStore**를 사용하여 스레드 간 정보를 공유할 수 있다
- **네임스페이스 (Namespace)**를 활용하여 메모리를 체계적으로 구조화할 수 있다
- **시맨틱 검색**을 구현하여 의미 기반으로 메모리를 조회할 수 있다
- **체크포인터와 스토어를 연동**하여 단기/장기 메모리를 통합할 수 있다
- **크로스 스레드 메모리 공유**를 통해 다른 대화 세션 간 정보를 활용할 수 있다

## 🔑 핵심 개념

### 장기 메모리 (Long-term Memory)란?

**장기 메모리**는 다중 세션과 스레드에 걸쳐 정보를 유지하는 메모리 시스템입니다.

**단기 메모리 vs 장기 메모리**:

| 특성 | 단기 메모리 (Checkpoints) | 장기 메모리 (Store) |
|------|---------------------------|----------------------|
| **범위** | 단일 스레드 (대화 세션) | 다중 스레드 (크로스 세션) |
| **목적** | 대화 컨텍스트 유지 | 지식 축적 및 공유 |
| **저장 내용** | 메시지 히스토리, 상태 | 사용자 정보, 선호도, 지식 |
| **접근 방법** | thread_id 기반 | namespace + key 기반 |
| **검색 방법** | 시간 순서 | 시맨틱 검색 가능 |
| **사용 예시** | "이전에 뭐라고 했지?" | "이 사용자는 한식을 좋아함" |

**장기 메모리의 활용**:
```
스레드 1 (2024-10-01): "김치찌개를 좋아합니다" → 장기 메모리 저장
스레드 2 (2024-10-15): "추천 메뉴 알려주세요"
  → 장기 메모리 검색: "김치찌개 선호"
  → "한식 메뉴를 추천드립니다"
```

### InMemoryStore: 스레드 간 메모리 공유

**InMemoryStore**는 LangGraph의 장기 메모리 인터페이스로, 스레드 간 정보를 공유하고 저장합니다.

**주요 특징**:
- **Namespace 기반**: 튜플 형태로 메모리 구분 (예: `("user_123", "preferences")`)
- **Key-Value 저장**: 고유 key와 dictionary value
- **시맨틱 검색**: 임베딩 기반 의미 검색
- **체크포인터 연동**: 단기/장기 메모리 통합

**기본 구조**:
```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

# 네임스페이스: (사용자_ID, 메모리_타입)
namespace = ("user_123", "preferences")

# 메모리 저장
memory_id = "pref_001"
memory = {"food": "김치찌개", "hobby": "등산"}
store.put(namespace, memory_id, memory)

# 메모리 검색
results = store.search(namespace)
```

### Namespace: 메모리 조직화

**Namespace**는 메모리를 논리적으로 구분하는 튜플입니다.

**네임스페이스 설계 패턴**:

```python
# 패턴 1: 사용자 + 카테고리
namespace = ("user_123", "preferences")     # 사용자 선호도
namespace = ("user_123", "history")         # 사용자 대화 이력
namespace = ("user_123", "profile")         # 사용자 프로필

# 패턴 2: 조직 계층
namespace = ("company", "team_a", "policies")   # 팀 정책
namespace = ("company", "team_b", "policies")   # 다른 팀 정책

# 패턴 3: 시간 기반
namespace = ("user_123", "2024-10")         # 월별 메모리
namespace = ("user_123", "2024-11")

# 패턴 4: 컨텍스트 기반
namespace = ("session_abc", "context")      # 세션별 컨텍스트
namespace = ("project_xyz", "knowledge")    # 프로젝트별 지식
```

**네임스페이스 활용**:
- 독립적인 메모리 공간 생성
- 검색 범위 제한
- 권한 및 접근 제어
- 메모리 구조화 및 관리

### 시맨틱 검색 (Semantic Search)

키워드가 정확히 일치하지 않아도 **의미적으로 유사한** 메모리를 찾습니다.

**키워드 검색 vs 시맨틱 검색**:

```python
# 저장된 메모리:
# 1. "매운 음식을 좋아합니다"
# 2. "한식을 선호합니다"
# 3. "양식을 좋아합니다"

# 키워드 검색:
query = "한식"  # → "한식을 선호합니다"만 찾음

# 시맨틱 검색:
query = "김치찌개"  # → "한식을 선호합니다", "매운 음식을 좋아합니다" 모두 찾음
query = "스테이크"  # → "양식을 좋아합니다" 찾음
```

**시맨틱 검색 구현**:
```python
# 임베딩 함수 정의
def embed(texts: list[str]) -> list[list[float]]:
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    return embeddings_model.embed_documents(texts)

# 시맨틱 검색 지원 스토어
semantic_store = InMemoryStore(
    index={
        "embed": embed,          # 임베딩 함수
        "dims": 1536,            # OpenAI 임베딩 차원
        "fields": ["content"]    # 임베딩할 필드
    }
)
```

### 체크포인터와 스토어 연동

단기 메모리(Checkpoints)와 장기 메모리(Store)를 함께 사용합니다.

**통합 아키텍처**:
```
         단기 메모리 (Checkpointer)
              ↓
         [대화 히스토리]
         thread_1, thread_2, ...
              ↓
        각 스레드의 메시지 저장
              ↓
         장기 메모리 (Store)
              ↓
    [크로스 스레드 지식 공유]
    사용자 선호도, 프로필, 학습 내용
```

**연동 방법**:
```python
graph = builder.compile(
    checkpointer=MemorySaver(),    # 단기 메모리
    store=InMemoryStore()          # 장기 메모리
)
```

**노드에서 스토어 접근**:
```python
def my_node(state: State, config: RunnableConfig, *, store: BaseStore):
    # 장기 메모리에서 검색
    memories = store.search(namespace, query="preference")

    # 장기 메모리에 저장
    store.put(namespace, key, value)

    return state
```

## 🛠 환경 설정

### 필요한 라이브러리

```bash
pip install langchain langchain-openai langchain-chroma
pip install langgraph
pip install python-dotenv
```

### 기본 설정

```python
from dotenv import load_dotenv
load_dotenv()

# 기본 라이브러리
from pprint import pprint
import json
import uuid
from datetime import datetime
from dataclasses import dataclass

# LangChain 및 LangGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langgraph.prebuilt import ToolNode, tools_condition

from typing import List

print("환경 설정 완료!")
```

## 💻 단계별 구현

### 1단계: InMemoryStore 기본 사용법

#### 스토어 생성 및 메모리 저장

```python
from langgraph.store.memory import InMemoryStore
import uuid

# InMemoryStore 생성
store = InMemoryStore()

# 네임스페이스 정의
user_id = "user_001"
namespace = (user_id, "preferences")  # 튜플 형태

# 메모리 저장
memory_id = str(uuid.uuid4())  # 고유 ID 생성
memory = {
    "food_preference": "김치찌개를 좋아합니다",
    "hobby": "등산"
}

store.put(namespace, memory_id, memory)
print(f"메모리 저장 완료: {memory_id}")
```

**put() 메서드**:
- `namespace`: 메모리 카테고리 (튜플)
- `key`: 고유 식별자 (문자열)
- `value`: 저장할 데이터 (딕셔너리)

#### 메모리 검색

```python
# 네임스페이스 내 모든 메모리 검색
memories = store.search(namespace)

for memory in memories:
    print("검색 결과:")
    pprint(memory.dict())
    print("-" * 80)
```

**출력 예시**:
```python
검색 결과:
{'created_at': '2025-10-30T11:57:11+00:00',
 'key': 'a36410f8-b2b3-42a9-bb90-3bbab40829f0',
 'namespace': ['user_001', 'preferences'],
 'score': None,
 'updated_at': '2025-10-30T11:57:11+00:00',
 'value': {'food_preference': '김치찌개를 좋아합니다', 'hobby': '등산'}}
```

**Item 속성**:
- `key`: 메모리 고유 ID
- `namespace`: 메모리 카테고리
- `value`: 저장된 데이터
- `created_at`, `updated_at`: 타임스탬프
- `score`: 시맨틱 검색 시 유사도 점수

#### 여러 메모리 저장 및 조회

```python
# 여러 사용자의 메모리 저장
users_data = [
    ("user_001", {"food": "김치찌개", "hobby": "등산"}),
    ("user_002", {"food": "파스타", "hobby": "영화"}),
    ("user_003", {"food": "스테이크", "hobby": "독서"})
]

for user_id, data in users_data:
    namespace = (user_id, "preferences")
    memory_id = str(uuid.uuid4())
    store.put(namespace, memory_id, data)

# 특정 사용자 메모리 조회
user_namespace = ("user_002", "preferences")
user_memories = store.search(user_namespace)

print(f"User 002 preferences:")
for mem in user_memories:
    print(mem.value)
```

### 2단계: 시맨틱 검색 구현

#### 임베딩 함수 정의

```python
from langchain_openai import OpenAIEmbeddings

def embed(texts: list[str]) -> list[list[float]]:
    """텍스트를 임베딩 벡터로 변환"""
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    return embeddings_model.embed_documents(texts)

# 시맨틱 검색 지원 스토어 생성
semantic_store = InMemoryStore(
    index={
        "embed": embed,                           # 임베딩 함수
        "dims": 1536,                            # text-embedding-3-small 차원
        "fields": ["food_preference", "hobby"]   # 임베딩할 필드
    }
)
```

**index 파라미터**:
- `embed`: 임베딩 함수 (텍스트 → 벡터)
- `dims`: 임베딩 벡터 차원 (OpenAI small: 1536)
- `fields`: 임베딩할 필드 리스트

#### 다양한 메모리 저장

```python
# 네임스페이스
namespace = ("user_005", "preferences")

# 여러 메모리 저장
memories_to_store = [
    {
        "food_preference": "매운 음식을 좋아합니다",
        "hobby": "영화 감상"
    },
    {
        "food_preference": "한식을 선호합니다",
        "hobby": "등산과 캠핑"
    },
    {
        "food_preference": "양식을 좋아합니다",
        "hobby": "요리"
    }
]

for memory in memories_to_store:
    memory_id = str(uuid.uuid4())
    semantic_store.put(namespace, memory_id, memory)

print(f"{len(memories_to_store)}개 메모리 저장 완료")
```

#### 시맨틱 검색 실행

```python
# 의미 기반 검색
search_results = semantic_store.search(
    namespace,
    query="캠핑에 어울리는 영화",  # 키워드가 정확히 일치하지 않아도 됨
    limit=2
)

print("검색 쿼리: '캠핑에 어울리는 영화'")
print("\n검색 결과:")
for i, result in enumerate(search_results, 1):
    print(f"\n{i}. Score: {result.score:.4f}")
    print(f"   Food: {result.value['food_preference']}")
    print(f"   Hobby: {result.value['hobby']}")
```

**출력 예시**:
```
검색 쿼리: '캠핑에 어울리는 영화'

검색 결과:

1. Score: 0.8532
   Food: 매운 음식을 좋아합니다
   Hobby: 영화 감상

2. Score: 0.8124
   Food: 한식을 선호합니다
   Hobby: 등산과 캠핑
```

**시맨틱 검색의 강력함**:
- "캠핑"과 "등산과 캠핑"의 의미적 유사성 인식
- "영화"와 "영화 감상"의 관련성 인식
- 정확한 키워드 일치 불필요

#### 다양한 검색 쿼리 테스트

```python
# 테스트 쿼리 목록
test_queries = [
    "한국 음식",
    "야외 활동",
    "이탈리아 요리",
    "집에서 보내는 시간"
]

for query in test_queries:
    print(f"\n검색 쿼리: '{query}'")
    results = semantic_store.search(namespace, query=query, limit=1)

    if results:
        top_result = results[0]
        print(f"  가장 관련성 높은 메모리:")
        print(f"  - Food: {top_result.value['food_preference']}")
        print(f"  - Hobby: {top_result.value['hobby']}")
        print(f"  - Score: {top_result.score:.4f}")
```

**출력 예시**:
```
검색 쿼리: '한국 음식'
  가장 관련성 높은 메모리:
  - Food: 한식을 선호합니다
  - Hobby: 등산과 캠핑
  - Score: 0.8921

검색 쿼리: '야외 활동'
  가장 관련성 높은 메모리:
  - Food: 한식을 선호합니다
  - Hobby: 등산과 캠핑
  - Score: 0.8756

검색 쿼리: '이탈리아 요리'
  가장 관련성 높은 메모리:
  - Food: 양식을 좋아합니다
  - Hobby: 요리
  - Score: 0.8634
```

### 3단계: 체크포인터와 스토어 연동

단기 메모리와 장기 메모리를 통합하여 사용합니다.

#### 네임스페이스 관리 클래스

```python
from dataclasses import dataclass

@dataclass
class Namespace:
    """네임스페이스 관리 클래스"""
    user_id: str
    memory_type: str

    def to_tuple(self) -> tuple:
        """튜플로 변환"""
        return (self.user_id, self.memory_type)
```

**활용 예시**:
```python
ns = Namespace(user_id="user_123", memory_type="conversation")
namespace_tuple = ns.to_tuple()  # ("user_123", "conversation")
```

#### 상태 정의

```python
class GraphState(MessagesState):
    """메시지 상태 + 추가 필드"""
    summary: str  # 대화 요약 (선택 사항)
```

#### 메모리 업데이트 노드

```python
def update_memory(state: GraphState, config: RunnableConfig, *, store: BaseStore):
    """대화 내용을 장기 메모리에 저장"""

    # config에서 네임스페이스 정보 추출
    namespace = Namespace(
        user_id=config.get("configurable", {}).get("user_id", "default"),
        memory_type=config.get("configurable", {}).get("memory_type", "conversation")
    )

    # 마지막 메시지 추출
    last_message = state["messages"][-1]

    # 메모리 구성
    memory = {
        "conversation": last_message.content,
        "timestamp": str(datetime.now()),
        "type": last_message.type  # "human" or "ai"
    }

    # 장기 메모리에 저장
    store.put(namespace.to_tuple(), str(uuid.uuid4()), memory)

    return state
```

**update_memory 노드의 역할**:
- 각 대화 턴이 끝날 때마다 호출
- 메시지를 장기 메모리에 저장
- 다른 스레드에서도 접근 가능

#### LLM 호출 노드 (장기 메모리 활용)

```python
def call_model(state: GraphState, config: RunnableConfig, *, store: BaseStore):
    """LLM 호출 + 장기 메모리 검색"""

    system_prompt = SystemMessage("""You are a helpful AI assistant.
답변 시 이전 대화 내용을 참고하세요.""")

    # 네임스페이스 생성
    namespace = Namespace(
        user_id=config.get("configurable", {}).get("user_id", "default"),
        memory_type=config.get("configurable", {}).get("memory_type", "conversation")
    )

    # 장기 메모리에서 관련 대화 검색
    memories = store.search(
        namespace.to_tuple(),
        query=state["messages"][-1].content,  # 현재 질문으로 검색
        limit=3  # 최대 3개
    )

    # 검색된 메모리를 컨텍스트에 추가
    if memories:
        memory_context = "\n이전 관련 대화:\n" + "\n".join(
            f"- {m.value['conversation']}" for m in memories
        )
        context_message = SystemMessage(content=memory_context)
        messages = [system_prompt, context_message] + state["messages"]
    else:
        messages = [system_prompt] + state["messages"]

    # LLM 호출
    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}
```

**call_model 노드의 역할**:
- 현재 질문과 관련된 이전 대화를 장기 메모리에서 검색
- 검색된 메모리를 컨텍스트로 추가
- LLM이 이전 대화를 참고하여 답변

#### 그래프 구성

```python
# LLM 및 도구 설정
llm = ChatOpenAI(model="gpt-4o-mini")
tools = [search_menu, search_wine]  # 이전에 정의한 도구
llm_with_tools = llm.bind_tools(tools)

# 그래프 구성
builder = StateGraph(GraphState)

# 노드 추가
builder.add_node("agent", call_model)
builder.add_node("memory", update_memory)  # 메모리 업데이트 노드
builder.add_node("tools", ToolNode(tools))

# 엣지 연결
builder.add_edge(START, "agent")
builder.add_edge("agent", "memory")      # LLM → 메모리 저장
builder.add_conditional_edges("memory", tools_condition)
builder.add_edge("tools", "agent")
```

**그래프 흐름**:
```
START → agent (LLM 호출)
         ↓
       memory (장기 메모리 저장)
         ↓
   tools_condition (도구 필요?)
     ↓         ↓
  tools      END
     ↓
   agent
```

#### 스토어 생성 및 컴파일

```python
# 임베딩 함수 정의
def embed(texts: list[str]) -> list[list[float]]:
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    return embeddings_model.embed_documents(texts)

# 시맨틱 검색 지원 스토어 생성
conversation_store = InMemoryStore(
    index={
        "embed": embed,
        "dims": 1536,
        "fields": ["conversation"]  # conversation 필드를 임베딩
    }
)

# 단기 + 장기 메모리 통합 컴파일
graph_with_store = builder.compile(
    checkpointer=MemorySaver(),      # 단기 메모리
    store=conversation_store         # 장기 메모리
)
```

### 4단계: 크로스 스레드 메모리 공유

#### 첫 번째 스레드에서 대화

```python
# 스레드 1 설정
config_thread1 = {
    "configurable": {
        "thread_id": "thread_1",
        "user_id": "user_123",
        "memory_type": "conversation"
    }
}

# 첫 번째 질문
print("=" * 80)
print("스레드 1: 첫 번째 질문")
print("=" * 80)

result = graph_with_store.invoke({
    "messages": [HumanMessage(content="스테이크 메뉴의 가격은 얼마인가요?")]
}, config_thread1)

for msg in result['messages']:
    msg.pretty_print()
```

**실행 결과**:
```
================================================================================
스레드 1: 첫 번째 질문
================================================================================
================================ Human Message =================================
스테이크 메뉴의 가격은 얼마인가요?

================================== Ai Message ==================================
Tool Calls:
  search_menu (call_...)
  Args: query: 스테이크

================================= Tool Message =================================
[Document(...샤토브리앙 스테이크...₩42,000...)]

================================== Ai Message ==================================
스테이크 메뉴의 가격은 다음과 같습니다:

1. 샤토브리앙 스테이크: ₩42,000
2. 안심 스테이크 샐러드: ₩26,000
```

#### 장기 메모리 확인

```python
# 저장된 메모리 확인
namespace = ("user_123", "conversation")
memories = conversation_store.search(namespace, limit=5)

print("\n저장된 메모리:")
for i, mem in enumerate(memories, 1):
    print(f"\n{i}. [{mem.value['type']}] {mem.value['conversation'][:60]}...")
    print(f"   Timestamp: {mem.value['timestamp']}")
```

**출력 예시**:
```
저장된 메모리:

1. [human] 스테이크 메뉴의 가격은 얼마인가요?...
   Timestamp: 2025-10-30 20:57:19

2. [ai] 스테이크 메뉴의 가격은 다음과 같습니다: 1. 샤토브리앙 스테이크...
   Timestamp: 2025-10-30 20:57:20
```

#### 두 번째 스레드에서 이전 정보 활용

```python
# 스레드 2 설정 (다른 thread_id, 같은 user_id)
config_thread2 = {
    "configurable": {
        "thread_id": "thread_2",    # 다른 스레드
        "user_id": "user_123",      # 같은 사용자
        "memory_type": "conversation"
    }
}

# 이전 대화 참조하는 질문
print("\n" + "=" * 80)
print("스레드 2: 이전 대화 참조")
print("=" * 80)

result = graph_with_store.invoke({
    "messages": [HumanMessage(content="스테이크 메뉴 가격이 얼마라고 했나요? 더 저렴한 메뉴는 무엇인가요?")]
}, config_thread2)

result['messages'][-1].pretty_print()
```

**실행 결과**:
```
================================================================================
스레드 2: 이전 대화 참조
================================================================================
================================== Ai Message ==================================

이전 대화에서 스테이크 메뉴 가격을 다음과 같이 안내드렸습니다:

1. 샤토브리앙 스테이크 - ₩42,000
2. 안심 스테이크 샐러드 - ₩26,000

더 저렴한 메뉴는 **안심 스테이크 샐러드**로, 가격은 ₩26,000입니다.
```

**크로스 스레드 메모리의 핵심**:
- 스레드 1에서 저장한 정보를 스레드 2에서 활용
- `user_id`가 같으므로 같은 네임스페이스 공유
- 시맨틱 검색으로 관련 대화를 자동으로 찾음

#### 시맨틱 검색 테스트

```python
# 다양한 검색 쿼리로 장기 메모리 조회
test_queries = [
    "가격",
    "샐러드",
    "저렴한 메뉴",
    "42000원"
]

print("\n" + "=" * 80)
print("장기 메모리 시맨틱 검색 테스트")
print("=" * 80)

for query in test_queries:
    print(f"\n검색 쿼리: '{query}'")

    results = conversation_store.search(
        namespace,
        query=query,
        limit=2
    )

    for i, result in enumerate(results, 1):
        print(f"  {i}. Score: {result.score:.4f}")
        print(f"     Content: {result.value['conversation'][:50]}...")
```

**출력 예시**:
```
================================================================================
장기 메모리 시맨틱 검색 테스트
================================================================================

검색 쿼리: '가격'
  1. Score: 0.8921
     Content: 스테이크 메뉴의 가격은 얼마인가요?...
  2. Score: 0.8534
     Content: 스테이크 메뉴의 가격은 다음과 같습니다...

검색 쿼리: '샐러드'
  1. Score: 0.8756
     Content: 안심 스테이크 샐러드 - ₩26,000...
  2. Score: 0.8123
     Content: 샤토브리앙 스테이크...
```

## 🎯 실습 문제

### 실습 1: 사용자 프로필 관리 시스템 (난이도: ⭐⭐⭐)

**문제**: 사용자별로 프로필 정보를 장기 메모리에 저장하고, 다른 스레드에서 활용하는 시스템을 구현하세요.

**요구사항**:
- 사용자 프로필 저장: 이름, 선호 음식, 알레르기, 선호 좌석
- 프로필 기반 맞춤 추천
- 네임스페이스: `(user_id, "profile")`
- 시맨틱 검색으로 관련 프로필 조회

**힌트**:
```python
# 프로필 저장 함수
def save_user_profile(store, user_id, profile_data):
    namespace = (user_id, "profile")
    profile_id = str(uuid.uuid4())
    store.put(namespace, profile_id, profile_data)

# 프로필 조회 함수
def get_user_profile(store, user_id, query=""):
    namespace = (user_id, "profile")
    if query:
        results = store.search(namespace, query=query, limit=1)
    else:
        results = store.search(namespace, limit=10)
    return results
```

**테스트 시나리오**:
1. 사용자 프로필 저장: "이름: 홍길동, 선호 음식: 한식, 알레르기: 갑각류, 선호 좌석: 창가"
2. 다른 스레드에서 추천: "메뉴 추천해주세요" → 한식 위주, 갑각류 제외
3. 프로필 업데이트: "이제 양식도 좋아합니다"
4. 프로필 기반 검색: "알레르기 정보"

### 실습 2: 팀별 지식 베이스 (난이도: ⭐⭐⭐⭐)

**문제**: 여러 팀의 지식을 독립적으로 관리하고, 팀원들이 공유하는 지식 베이스를 구현하세요.

**요구사항**:
- 팀별 네임스페이스: `(team_id, "knowledge")`
- 팀원별 개인 네임스페이스: `(team_id, user_id, "notes")`
- 지식 저장 시 카테고리 태깅
- 시맨틱 검색으로 관련 지식 조회
- 팀 vs 개인 지식 구분

**힌트**:
```python
# 팀 지식 저장
def save_team_knowledge(store, team_id, knowledge):
    namespace = (team_id, "knowledge")
    knowledge_id = str(uuid.uuid4())
    store.put(namespace, knowledge_id, knowledge)

# 개인 노트 저장
def save_personal_note(store, team_id, user_id, note):
    namespace = (team_id, user_id, "notes")
    note_id = str(uuid.uuid4())
    store.put(namespace, note_id, note)

# 통합 검색 (팀 + 개인)
def search_knowledge(store, team_id, user_id, query):
    # 팀 지식 검색
    team_ns = (team_id, "knowledge")
    team_results = store.search(team_ns, query=query, limit=3)

    # 개인 노트 검색
    personal_ns = (team_id, user_id, "notes")
    personal_results = store.search(personal_ns, query=query, limit=2)

    return team_results + personal_results
```

**테스트 시나리오**:
- Team A: "FastAPI 배포 방법", "Docker 설정 가이드"
- Team B: "React 컴포넌트 패턴", "상태 관리 전략"
- User 1 (Team A): "내가 자주 쓰는 Docker 명령어"
- 검색: "배포" → Team A 지식 + User 1 노트 반환

### 실습 3: 장기 메모리 기반 RAG 시스템 (난이도: ⭐⭐⭐⭐)

**문제**: 실습 1에서 구현한 다국어 DB 도구와 장기 메모리를 결합하여, 사용자의 과거 질문과 답변을 학습하는 시스템을 만드세요.

**요구사항**:
- 한국어/영어 DB 검색 도구 사용
- 모든 질문-답변 쌍을 장기 메모리에 저장
- 유사한 이전 질문이 있으면 그 답변도 참고
- 네임스페이스: `(user_id, "qa_history")`
- 시맨틱 검색으로 유사 질문 찾기

**힌트**:
```python
# QA 저장 노드
def save_qa(state: GraphState, config: RunnableConfig, *, store: BaseStore):
    user_id = config.get("configurable", {}).get("user_id", "default")
    namespace = (user_id, "qa_history")

    messages = state["messages"]
    if len(messages) >= 2:
        question = messages[-2].content if len(messages) >= 2 else ""
        answer = messages[-1].content

        qa_record = {
            "question": question,
            "answer": answer,
            "timestamp": str(datetime.now())
        }

        store.put(namespace, str(uuid.uuid4()), qa_record)

    return state

# 유사 질문 검색 노드
def search_similar_qa(state: GraphState, config: RunnableConfig, *, store: BaseStore):
    user_id = config.get("configurable", {}).get("user_id", "default")
    namespace = (user_id, "qa_history")

    current_question = state["messages"][-1].content

    # 유사한 이전 질문 검색
    similar_qas = store.search(namespace, query=current_question, limit=2)

    if similar_qas:
        context = "\n이전 유사 질문:\n" + "\n".join(
            f"Q: {qa.value['question']}\nA: {qa.value['answer']}"
            for qa in similar_qas
        )
        # 컨텍스트 추가
        # ...

    return state
```

**테스트 시나리오**:
1. "테슬라의 창업자는 누구인가요?" → 답변 저장
2. "테슬라는 언제 설립되었나요?" → 답변 저장
3. "리비안의 창업자는?" → 답변 저장
4. "Tesla의 founder는?" → 이전 유사 질문 활용 (한국어 질문 참고)

## ✅ 솔루션 예시

### 실습 1 솔루션

```python
# 시맨틱 검색 지원 스토어
profile_store = InMemoryStore(
    index={
        "embed": embed,
        "dims": 1536,
        "fields": ["name", "food_preference", "allergies", "seat_preference"]
    }
)

# 프로필 관리 함수
def save_user_profile(store, user_id, profile_data):
    """사용자 프로필 저장"""
    namespace = (user_id, "profile")
    profile_id = str(uuid.uuid4())
    store.put(namespace, profile_id, profile_data)
    print(f"프로필 저장 완료: {user_id}")

def get_user_profile(store, user_id, query=""):
    """사용자 프로필 조회"""
    namespace = (user_id, "profile")

    if query:
        results = store.search(namespace, query=query, limit=5)
    else:
        results = store.search(namespace, limit=10)

    return results

def update_user_profile(store, user_id, updates):
    """사용자 프로필 업데이트"""
    namespace = (user_id, "profile")

    # 기존 프로필 조회
    existing = store.search(namespace, limit=1)

    if existing:
        # 기존 프로필 업데이트
        old_profile = existing[0].value
        old_profile.update(updates)
        store.put(namespace, str(uuid.uuid4()), old_profile)
    else:
        # 새 프로필 생성
        save_user_profile(store, user_id, updates)

# 테스트: 프로필 저장
user_id = "user_hong"
profile = {
    "name": "홍길동",
    "food_preference": "한식을 좋아합니다",
    "allergies": "갑각류 알레르기가 있습니다",
    "seat_preference": "창가 자리를 선호합니다"
}

save_user_profile(profile_store, user_id, profile)

# 프로필 조회
print("\n전체 프로필:")
profiles = get_user_profile(profile_store, user_id)
for p in profiles:
    pprint(p.value)

# 특정 정보 검색
print("\n알레르기 정보 검색:")
allergy_info = get_user_profile(profile_store, user_id, query="알레르기")
for info in allergy_info:
    print(f"  {info.value}")

# 프로필 업데이트
print("\n프로필 업데이트:")
update_user_profile(profile_store, user_id, {
    "food_preference": "한식과 양식을 좋아합니다"
})

profiles = get_user_profile(profile_store, user_id)
print(f"업데이트된 선호 음식: {profiles[0].value['food_preference']}")
```

**실행 결과**:
```
프로필 저장 완료: user_hong

전체 프로필:
{'allergies': '갑각류 알레르기가 있습니다',
 'food_preference': '한식을 좋아합니다',
 'name': '홍길동',
 'seat_preference': '창가 자리를 선호합니다'}

알레르기 정보 검색:
  {'allergies': '갑각류 알레르기가 있습니다', ...}

프로필 업데이트:
업데이트된 선호 음식: 한식과 양식을 좋아합니다
```

**프로필 기반 추천 시스템**:

```python
def recommend_with_profile(store, user_id, tools):
    """프로필 기반 맞춤 추천"""

    # 프로필 조회
    profiles = get_user_profile(store, user_id)

    if not profiles:
        return "프로필 정보가 없습니다. 먼저 프로필을 등록해주세요."

    profile = profiles[0].value

    # 추천 로직
    food_pref = profile.get("food_preference", "")
    allergies = profile.get("allergies", "")

    recommendation = f"""
    [{profile['name']}님을 위한 맞춤 추천]

    선호하시는 {food_pref}를 고려하여 추천드립니다.
    {allergies}이므로 해당 식재료는 제외했습니다.

    추천 메뉴:
    """

    # 실제로는 도구를 사용하여 메뉴 검색
    # ...

    return recommendation

# 사용 예시
print(recommend_with_profile(profile_store, "user_hong", tools))
```

### 실습 2 솔루션

```python
# 팀 지식 베이스 스토어
knowledge_store = InMemoryStore(
    index={
        "embed": embed,
        "dims": 1536,
        "fields": ["title", "content", "category"]
    }
)

# 팀 지식 관리
class TeamKnowledgeBase:
    def __init__(self, store):
        self.store = store

    def save_team_knowledge(self, team_id, knowledge):
        """팀 공유 지식 저장"""
        namespace = (team_id, "knowledge")
        knowledge_id = str(uuid.uuid4())
        self.store.put(namespace, knowledge_id, knowledge)
        print(f"[{team_id}] 팀 지식 저장: {knowledge['title']}")

    def save_personal_note(self, team_id, user_id, note):
        """개인 노트 저장"""
        namespace = (team_id, user_id, "notes")
        note_id = str(uuid.uuid4())
        self.store.put(namespace, note_id, note)
        print(f"[{team_id}/{user_id}] 개인 노트 저장: {note['title']}")

    def search_knowledge(self, team_id, query, limit=5):
        """팀 지식 검색"""
        namespace = (team_id, "knowledge")
        results = self.store.search(namespace, query=query, limit=limit)
        return results

    def search_personal(self, team_id, user_id, query, limit=3):
        """개인 노트 검색"""
        namespace = (team_id, user_id, "notes")
        results = self.store.search(namespace, query=query, limit=limit)
        return results

    def search_all(self, team_id, user_id, query):
        """통합 검색 (팀 + 개인)"""
        team_results = self.search_knowledge(team_id, query, limit=3)
        personal_results = self.search_personal(team_id, user_id, query, limit=2)

        return {
            "team_knowledge": team_results,
            "personal_notes": personal_results
        }

# 사용 예시
kb = TeamKnowledgeBase(knowledge_store)

# Team A 지식
kb.save_team_knowledge("team_a", {
    "title": "FastAPI 배포 방법",
    "content": "Docker를 사용한 FastAPI 애플리케이션 배포 가이드...",
    "category": "deployment"
})

kb.save_team_knowledge("team_a", {
    "title": "Docker 설정 가이드",
    "content": "Docker Compose를 활용한 개발 환경 설정...",
    "category": "devops"
})

# Team B 지식
kb.save_team_knowledge("team_b", {
    "title": "React 컴포넌트 패턴",
    "content": "재사용 가능한 React 컴포넌트 설계 패턴...",
    "category": "frontend"
})

# User 1 (Team A) 개인 노트
kb.save_personal_note("team_a", "user_1", {
    "title": "자주 쓰는 Docker 명령어",
    "content": "docker-compose up -d, docker ps, docker logs...",
    "category": "personal"
})

# 검색 테스트
print("\n" + "="*80)
print("Team A - User 1: '배포' 검색")
print("="*80)

results = kb.search_all("team_a", "user_1", "배포")

print("\n팀 지식:")
for i, r in enumerate(results["team_knowledge"], 1):
    print(f"  {i}. {r.value['title']}")
    print(f"     Category: {r.value['category']}")
    print(f"     Score: {r.score:.4f}")

print("\n개인 노트:")
for i, r in enumerate(results["personal_notes"], 1):
    print(f"  {i}. {r.value['title']}")
    print(f"     Category: {r.value['category']}")
    print(f"     Score: {r.score:.4f}")
```

**실행 결과**:
```
[team_a] 팀 지식 저장: FastAPI 배포 방법
[team_a] 팀 지식 저장: Docker 설정 가이드
[team_b] 팀 지식 저장: React 컴포넌트 패턴
[team_a/user_1] 개인 노트 저장: 자주 쓰는 Docker 명령어

================================================================================
Team A - User 1: '배포' 검색
================================================================================

팀 지식:
  1. FastAPI 배포 방법
     Category: deployment
     Score: 0.8921
  2. Docker 설정 가이드
     Category: devops
     Score: 0.8356

개인 노트:
  1. 자주 쓰는 Docker 명령어
     Category: personal
     Score: 0.8123
```

### 실습 3 솔루션

```python
# QA 히스토리 스토어
qa_store = InMemoryStore(
    index={
        "embed": embed,
        "dims": 1536,
        "fields": ["question", "answer"]
    }
)

# 상태 정의
class QAGraphState(MessagesState):
    similar_questions: list = []  # 유사 질문 저장

# QA 저장 노드
def save_qa(state: QAGraphState, config: RunnableConfig, *, store: BaseStore):
    """질문-답변 쌍을 장기 메모리에 저장"""
    user_id = config.get("configurable", {}).get("user_id", "default")
    namespace = (user_id, "qa_history")

    messages = state["messages"]
    if len(messages) >= 2:
        # 마지막 Human-AI 메시지 쌍 추출
        question = None
        answer = None

        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], AIMessage) and not answer:
                answer = messages[i].content
            elif isinstance(messages[i], HumanMessage) and answer and not question:
                question = messages[i].content
                break

        if question and answer:
            qa_record = {
                "question": question,
                "answer": answer,
                "timestamp": str(datetime.now())
            }

            store.put(namespace, str(uuid.uuid4()), qa_record)
            print(f"[QA 저장] Q: {question[:50]}...")

    return state

# 유사 질문 검색 및 LLM 호출 노드
def call_model_with_qa_history(state: QAGraphState, config: RunnableConfig, *, store: BaseStore):
    """장기 메모리에서 유사 질문을 검색하고 LLM 호출"""
    user_id = config.get("configurable", {}).get("user_id", "default")
    namespace = (user_id, "qa_history")

    current_question = state["messages"][-1].content

    # 유사한 이전 QA 검색
    similar_qas = store.search(namespace, query=current_question, limit=2)

    system_prompt = SystemMessage("""You are a helpful AI assistant.
답변 시 이전 유사 질문의 답변도 참고하세요.""")

    # 유사 질문 컨텍스트 추가
    if similar_qas:
        qa_context = "\n\n이전 유사 질문과 답변:\n" + "\n".join(
            f"Q: {qa.value['question']}\nA: {qa.value['answer'][:100]}..."
            for qa in similar_qas
        )
        context_message = SystemMessage(content=qa_context)
        messages = [system_prompt, context_message] + state["messages"]

        print(f"\n[유사 질문 발견] {len(similar_qas)}개")
    else:
        messages = [system_prompt] + state["messages"]
        print("\n[유사 질문 없음]")

    # LLM 호출
    response = llm_with_db_tools.invoke(messages)

    return {"messages": [response]}

# 그래프 구성
builder = StateGraph(QAGraphState)

builder.add_node("agent", call_model_with_qa_history)
builder.add_node("save_qa", save_qa)
builder.add_node("tools", ToolNode([search_kor, search_eng]))

# 엣지
builder.add_edge(START, "agent")
builder.add_edge("agent", "save_qa")
builder.add_conditional_edges("save_qa", tools_condition)
builder.add_edge("tools", "agent")

# 컴파일
qa_graph = builder.compile(
    checkpointer=MemorySaver(),
    store=qa_store
)

# 테스트 시나리오
config = {
    "configurable": {
        "thread_id": "thread_1",
        "user_id": "user_qa_test",
    }
}

print("="*80)
print("질문 1: 테슬라의 창업자는 누구인가요?")
print("="*80)
result = qa_graph.invoke({
    "messages": [HumanMessage(content="테슬라의 창업자는 누구인가요?")]
}, config)
result['messages'][-1].pretty_print()

config["configurable"]["thread_id"] = "thread_2"
print("\n" + "="*80)
print("질문 2: 테슬라는 언제 설립되었나요?")
print("="*80)
result = qa_graph.invoke({
    "messages": [HumanMessage(content="테슬라는 언제 설립되었나요?")]
}, config)
result['messages'][-1].pretty_print()

config["configurable"]["thread_id"] = "thread_3"
print("\n" + "="*80)
print("질문 3: Tesla의 founder는? (유사 질문 활용)")
print("="*80)
result = qa_graph.invoke({
    "messages": [HumanMessage(content="Who is the founder of Tesla?")]
}, config)
result['messages'][-1].pretty_print()

# QA 히스토리 확인
print("\n" + "="*80)
print("저장된 QA 히스토리")
print("="*80)
namespace = ("user_qa_test", "qa_history")
all_qas = qa_store.search(namespace, limit=10)

for i, qa in enumerate(all_qas, 1):
    print(f"\n{i}. Q: {qa.value['question']}")
    print(f"   A: {qa.value['answer'][:80]}...")
```

**실행 결과**:
```
================================================================================
질문 1: 테슬라의 창업자는 누구인가요?
================================================================================
[유사 질문 없음]
[QA 저장] Q: 테슬라의 창업자는 누구인가요?...

================================== Ai Message ==================================
테슬라(Tesla)는 2003년에 Martin Eberhard와 Marc Tarpenning에 의해 설립되었습니다.

================================================================================
질문 2: 테슬라는 언제 설립되었나요?
================================================================================
[유사 질문 발견] 1개
[QA 저장] Q: 테슬라는 언제 설립되었나요?...

================================== Ai Message ==================================
이전 답변에서 말씀드린 것처럼, 테슬라는 2003년에 설립되었습니다.

================================================================================
질문 3: Tesla의 founder는? (유사 질문 활용)
================================================================================
[유사 질문 발견] 2개
[QA 저장] Q: Who is the founder of Tesla?...

================================== Ai Message ==================================
As mentioned in the previous similar questions in Korean, Tesla was founded
by Martin Eberhard and Marc Tarpenning in 2003. [도구: search_eng]

================================================================================
저장된 QA 히스토리
================================================================================

1. Q: Who is the founder of Tesla?
   A: As mentioned in the previous similar questions, Tesla was founded by Martin ...

2. Q: 테슬라는 언제 설립되었나요?
   A: 이전 답변에서 말씀드린 것처럼, 테슬라는 2003년에 설립되었습니다...

3. Q: 테슬라의 창업자는 누구인가요?
   A: 테슬라(Tesla)는 2003년에 Martin Eberhard와 Marc Tarpenning에 의해 설립...
```

**솔루션 포인트**:
- 모든 QA를 장기 메모리에 자동 저장
- 새 질문에 유사한 이전 질문이 있으면 참고
- 영어 질문에도 한국어 QA 활용 (시맨틱 검색)
- 크로스 스레드로 지식 축적

## 🚀 실무 활용 예시

### 예시 1: 개인화된 학습 어시스턴트

사용자의 학습 진도와 이해도를 장기 메모리에 저장하여 맞춤형 학습을 제공합니다.

```python
# 학습 진도 스토어
learning_store = InMemoryStore(
    index={
        "embed": embed,
        "dims": 1536,
        "fields": ["topic", "concept", "difficulty", "mastery_level"]
    }
)

class LearningAssistant:
    """개인화 학습 어시스턴트"""

    def __init__(self, store, graph):
        self.store = store
        self.graph = graph

    def record_progress(self, user_id, learning_record):
        """학습 진도 기록"""
        namespace = (user_id, "learning_progress")
        record_id = str(uuid.uuid4())

        record = {
            "topic": learning_record["topic"],
            "concept": learning_record["concept"],
            "difficulty": learning_record["difficulty"],
            "mastery_level": learning_record["mastery_level"],
            "timestamp": str(datetime.now()),
            "notes": learning_record.get("notes", "")
        }

        self.store.put(namespace, record_id, record)
        print(f"학습 진도 기록: {record['topic']} - {record['concept']}")

    def get_weak_areas(self, user_id):
        """취약 영역 분석"""
        namespace = (user_id, "learning_progress")
        all_records = self.store.search(namespace, limit=50)

        # 낮은 mastery_level 찾기
        weak_areas = [
            r for r in all_records
            if r.value.get("mastery_level", 0) < 70
        ]

        return weak_areas

    def recommend_next_topic(self, user_id):
        """다음 학습 주제 추천"""
        weak_areas = self.get_weak_areas(user_id)

        if weak_areas:
            # 가장 낮은 mastery_level 주제
            weakest = min(weak_areas, key=lambda x: x.value["mastery_level"])
            return weakest.value["topic"]

        # 취약 영역이 없으면 새 주제
        return "새로운 고급 주제"

    def get_personalized_explanation(self, user_id, concept):
        """개인화된 설명"""
        namespace = (user_id, "learning_progress")

        # 관련 학습 기록 검색
        related_records = self.store.search(
            namespace,
            query=concept,
            limit=3
        )

        # 사용자의 이해도 수준 파악
        if related_records:
            avg_mastery = sum(r.value["mastery_level"] for r in related_records) / len(related_records)
            difficulty = "beginner" if avg_mastery < 50 else "intermediate" if avg_mastery < 80 else "advanced"
        else:
            difficulty = "beginner"

        return f"[{difficulty} 수준] {concept}에 대한 설명을 제공합니다..."

# 사용 예시
assistant = LearningAssistant(learning_store, None)

# 학습 기록
assistant.record_progress("student_001", {
    "topic": "Python",
    "concept": "List Comprehension",
    "difficulty": "intermediate",
    "mastery_level": 85,
    "notes": "잘 이해함"
})

assistant.record_progress("student_001", {
    "topic": "Python",
    "concept": "Decorators",
    "difficulty": "advanced",
    "mastery_level": 45,
    "notes": "더 연습 필요"
})

assistant.record_progress("student_001", {
    "topic": "Data Structures",
    "concept": "Binary Trees",
    "difficulty": "advanced",
    "mastery_level": 60,
    "notes": "기본은 이해"
})

# 취약 영역 분석
print("\n취약 영역:")
weak = assistant.get_weak_areas("student_001")
for w in weak:
    print(f"  - {w.value['topic']}: {w.value['concept']} (숙달도: {w.value['mastery_level']}%)")

# 다음 추천 주제
next_topic = assistant.recommend_next_topic("student_001")
print(f"\n다음 추천 학습 주제: {next_topic}")
```

### 예시 2: 고객 선호도 기반 추천 시스템

여러 세션에 걸친 고객의 선호도를 학습하여 맞춤 추천을 제공합니다.

```python
class PreferenceBasedRecommender:
    """선호도 기반 추천 시스템"""

    def __init__(self, store):
        self.store = store

    def record_interaction(self, user_id, interaction):
        """사용자 상호작용 기록"""
        namespace = (user_id, "interactions")
        interaction_id = str(uuid.uuid4())

        record = {
            "item": interaction["item"],
            "action": interaction["action"],  # viewed, liked, purchased, skipped
            "category": interaction.get("category", ""),
            "price_range": interaction.get("price_range", ""),
            "timestamp": str(datetime.now())
        }

        self.store.put(namespace, interaction_id, record)

    def get_preferences(self, user_id):
        """사용자 선호도 분석"""
        namespace = (user_id, "interactions")
        all_interactions = self.store.search(namespace, limit=100)

        # 긍정적 상호작용만 추출
        positive = [
            i for i in all_interactions
            if i.value["action"] in ["liked", "purchased"]
        ]

        # 카테고리별 집계
        category_counts = {}
        for interaction in positive:
            cat = interaction.value.get("category", "unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1

        # 선호 카테고리
        if category_counts:
            preferred_category = max(category_counts, key=category_counts.get)
            return {
                "preferred_category": preferred_category,
                "category_distribution": category_counts
            }

        return None

    def recommend_items(self, user_id, available_items):
        """아이템 추천"""
        prefs = self.get_preferences(user_id)

        if not prefs:
            # 신규 사용자: 인기 아이템 추천
            return available_items[:5]

        # 선호 카테고리 우선 추천
        preferred_cat = prefs["preferred_category"]
        recommendations = [
            item for item in available_items
            if item.get("category") == preferred_cat
        ]

        # 다양성을 위해 다른 카테고리도 일부 포함
        other_items = [
            item for item in available_items
            if item.get("category") != preferred_cat
        ]

        return recommendations[:3] + other_items[:2]

# 사용 예시
recommender = PreferenceBasedRecommender(InMemoryStore(
    index={"embed": embed, "dims": 1536, "fields": ["item", "category"]}
))

# 상호작용 기록
user_id = "customer_001"

recommender.record_interaction(user_id, {
    "item": "스테이크",
    "action": "purchased",
    "category": "양식",
    "price_range": "high"
})

recommender.record_interaction(user_id, {
    "item": "파스타",
    "action": "liked",
    "category": "양식",
    "price_range": "medium"
})

recommender.record_interaction(user_id, {
    "item": "김치찌개",
    "action": "viewed",
    "category": "한식",
    "price_range": "medium"
})

recommender.record_interaction(user_id, {
    "item": "피자",
    "action": "skipped",
    "category": "양식",
    "price_range": "medium"
})

# 선호도 분석
prefs = recommender.get_preferences(user_id)
print("사용자 선호도:")
print(f"  선호 카테고리: {prefs['preferred_category']}")
print(f"  카테고리 분포: {prefs['category_distribution']}")

# 추천
available = [
    {"name": "리조또", "category": "양식", "price": "high"},
    {"name": "비빔밥", "category": "한식", "price": "medium"},
    {"name": "스파게티", "category": "양식", "price": "medium"},
    {"name": "된장찌개", "category": "한식", "price": "low"},
    {"name": "샐러드", "category": "양식", "price": "low"},
]

recommendations = recommender.recommend_items(user_id, available)
print("\n추천 메뉴:")
for i, item in enumerate(recommendations, 1):
    print(f"  {i}. {item['name']} ({item['category']})")
```

### 예시 3: 프로젝트 지식 관리 시스템

프로젝트 전반에 걸친 결정사항, 이슈, 해결책을 장기 메모리에 저장하고 활용합니다.

```python
class ProjectKnowledgeManager:
    """프로젝트 지식 관리 시스템"""

    def __init__(self, store):
        self.store = store

    def record_decision(self, project_id, decision):
        """설계 결정 기록"""
        namespace = (project_id, "decisions")
        decision_id = str(uuid.uuid4())

        record = {
            "title": decision["title"],
            "decision": decision["decision"],
            "rationale": decision["rationale"],
            "alternatives": decision.get("alternatives", []),
            "date": str(datetime.now()),
            "tags": decision.get("tags", [])
        }

        self.store.put(namespace, decision_id, record)
        print(f"설계 결정 기록: {record['title']}")

    def record_issue_resolution(self, project_id, issue):
        """이슈 해결 기록"""
        namespace = (project_id, "issues")
        issue_id = str(uuid.uuid4())

        record = {
            "title": issue["title"],
            "problem": issue["problem"],
            "solution": issue["solution"],
            "root_cause": issue.get("root_cause", ""),
            "date": str(datetime.now()),
            "tags": issue.get("tags", [])
        }

        self.store.put(namespace, issue_id, record)
        print(f"이슈 해결 기록: {record['title']}")

    def search_similar_decisions(self, project_id, query):
        """유사한 설계 결정 검색"""
        namespace = (project_id, "decisions")
        results = self.store.search(namespace, query=query, limit=3)
        return results

    def search_similar_issues(self, project_id, query):
        """유사한 이슈 검색"""
        namespace = (project_id, "issues")
        results = self.store.search(namespace, query=query, limit=3)
        return results

    def get_project_summary(self, project_id):
        """프로젝트 요약"""
        decisions_ns = (project_id, "decisions")
        issues_ns = (project_id, "issues")

        decisions = self.store.search(decisions_ns, limit=50)
        issues = self.store.search(issues_ns, limit=50)

        return {
            "total_decisions": len(decisions),
            "total_issues": len(issues),
            "recent_decisions": decisions[:5],
            "recent_issues": issues[:5]
        }

# 사용 예시
pkm = ProjectKnowledgeManager(InMemoryStore(
    index={
        "embed": embed,
        "dims": 1536,
        "fields": ["title", "decision", "rationale", "problem", "solution"]
    }
))

project_id = "project_alpha"

# 설계 결정 기록
pkm.record_decision(project_id, {
    "title": "데이터베이스 선택",
    "decision": "PostgreSQL 사용",
    "rationale": "복잡한 쿼리와 트랜잭션 지원 필요",
    "alternatives": ["MongoDB", "MySQL"],
    "tags": ["database", "architecture"]
})

pkm.record_decision(project_id, {
    "title": "캐싱 전략",
    "decision": "Redis를 사용한 2단계 캐싱",
    "rationale": "응답 속도 개선 및 DB 부하 감소",
    "alternatives": ["Memcached", "In-memory caching"],
    "tags": ["performance", "caching"]
})

# 이슈 해결 기록
pkm.record_issue_resolution(project_id, {
    "title": "API 응답 속도 저하",
    "problem": "특정 엔드포인트의 응답 시간이 3초 이상",
    "solution": "N+1 쿼리 문제 해결, eager loading 적용",
    "root_cause": "ORM 관계 설정에서 lazy loading 사용",
    "tags": ["performance", "database"]
})

# 유사한 결정 검색
print("\n'데이터베이스 성능' 관련 결정:")
similar_decisions = pkm.search_similar_decisions(project_id, "데이터베이스 성능")
for d in similar_decisions:
    print(f"  - {d.value['title']}")
    print(f"    결정: {d.value['decision']}")
    print(f"    이유: {d.value['rationale']}")

# 유사한 이슈 검색
print("\n'성능' 관련 이슈:")
similar_issues = pkm.search_similar_issues(project_id, "성능 문제")
for i in similar_issues:
    print(f"  - {i.value['title']}")
    print(f"    해결책: {i.value['solution']}")

# 프로젝트 요약
summary = pkm.get_project_summary(project_id)
print(f"\n프로젝트 요약:")
print(f"  총 설계 결정: {summary['total_decisions']}개")
print(f"  총 해결된 이슈: {summary['total_issues']}개")
```

## 📖 참고 자료

### 공식 문서
- [LangGraph Store 개념](https://langchain-ai.github.io/langgraph/concepts/persistence/#memory-store)
- [InMemoryStore API](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.memory.InMemoryStore)
- [Cross-Thread Persistence](https://langchain-ai.github.io/langgraph/how-tos/cross-thread-persistence/)
- [Semantic Search in Store](https://langchain-ai.github.io/langgraph/how-tos/memory/semantic-search/)

### 스토어 구현
- [Custom Store Implementation](https://langchain-ai.github.io/langgraph/how-tos/persistence_postgres/#long-term-memory)
- [PostgreSQL Store](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.postgres.PostgresStore)
- [Store with Embeddings](https://langchain-ai.github.io/langgraph/how-tos/memory/semantic-search/)

### 실무 패턴
- [User Preferences and Profiles](https://langchain-ai.github.io/langgraph/how-tos/memory/semantic-search/)
- [Multi-User Memory Management](https://langchain-ai.github.io/langgraph/concepts/multi_tenancy/)
- [Memory Namespace Patterns](https://langchain-ai.github.io/langgraph/how-tos/cross-thread-persistence/)

### 추가 학습 자료
- [Semantic Search Patterns](https://python.langchain.com/docs/modules/data_connection/vectorstores/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [Memory Management Best Practices](https://langchain-ai.github.io/langgraph/concepts/memory/)

---

**완료!** Part 1과 Part 2를 통해 LangGraph의 단기 메모리와 장기 메모리를 모두 학습했습니다.

**다음 학습**: [LangGraph 고급 패턴 - Human-in-the-Loop, 스트리밍, 서브그래프]
