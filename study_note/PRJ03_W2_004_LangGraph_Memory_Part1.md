# LangGraph 메모리 관리 - Part 1: 단기 메모리

## 📚 학습 목표

이 가이드를 완료하면 다음을 할 수 있습니다:

- **체크포인트 (Checkpoints)**의 개념과 역할을 이해하고 구현할 수 있다
- **MemorySaver**를 사용하여 대화 세션 기반의 단기 메모리를 구현할 수 있다
- **thread_id**를 통해 독립적인 대화 세션을 관리할 수 있다
- **상태 히스토리**를 조회하고 특정 체크포인트로 재생(Replay)할 수 있다
- **상태 업데이트**를 통해 그래프 실행 중 상태를 직접 수정할 수 있다
- **긴 대화 관리**를 위한 메시지 삭제 전략을 구현할 수 있다
- **RemoveMessage**를 사용하여 선택적으로 메시지를 제거할 수 있다

## 🔑 핵심 개념

### 체크포인트 (Checkpoints)란?

**체크포인트**는 그래프 처리 과정의 상태를 저장하고 관리하는 시스템으로, LangGraph의 **단기 메모리**를 제공합니다.

체크포인트는 각 실행 단계에서 생성되는 그래프 상태의 **스냅샷(Snapshot)**으로 구성되며, 다음 정보를 포함합니다:

```python
class StateSnapshot:
    config: dict          # 체크포인트 관련 설정 (thread_id, checkpoint_id)
    metadata: dict        # 메타데이터 (source, step, parents)
    values: dict          # 해당 시점의 상태 채널 값
    next: tuple          # 다음에 실행할 노드 이름
    tasks: tuple         # 다음에 실행할 작업 정보 (PregelTask 객체)
```

**체크포인트의 활용**:
- 대화 컨텍스트 유지
- 특정 시점으로 되돌리기 (Time Travel)
- 상태 히스토리 추적
- 오류 발생 시 복구

### MemorySaver: 스레드 기반 단기 메모리

**MemorySaver**는 LangGraph에서 제공하는 인메모리 체크포인터로, 디버깅과 테스트 용도로 적합합니다.

**특징**:
- **스레드 기반**: `thread_id`로 독립적인 대화 세션 관리
- **단기 메모리**: 하나의 대화 세션 동안만 정보 유지
- **자동 저장**: 그래프의 각 단계마다 상태를 자동으로 기록
- **메모리 저장**: 프로세스 메모리에 저장 (재시작 시 소실)

**프로덕션 환경에서는**:
- `SqliteSaver`: 로컬 파일 기반 영구 저장
- `PostgresSaver`: 데이터베이스 기반 확장 가능한 저장
- 커스텀 체크포인터: 특정 요구사항에 맞게 구현

### thread_id: 대화 세션 식별자

`thread_id`는 독립적인 대화 세션을 구분하는 식별자입니다.

```python
# 사용자 A의 대화
config_a = {"configurable": {"thread_id": "user_a"}}
graph.invoke({"messages": [HumanMessage("안녕하세요")]}, config_a)

# 사용자 B의 대화 (완전히 독립적)
config_b = {"configurable": {"thread_id": "user_b"}}
graph.invoke({"messages": [HumanMessage("Hello")]}, config_b)
```

**thread_id 활용 시나리오**:
- 사용자별 대화 관리: `thread_id = f"user_{user_id}"`
- 세션별 대화 관리: `thread_id = f"session_{session_id}"`
- 주제별 대화 관리: `thread_id = f"topic_{topic_name}"`

### 상태 재생 (Replay)

특정 체크포인트부터 그래프를 다시 실행할 수 있습니다.

**재생의 특징**:
- 이전 단계는 **실제로 재실행하지 않고** 결과만 가져옴
- 불필요한 재실행 방지 (효율적, 비용 절감)
- 체크포인트 이후 단계만 실행

```python
# 특정 체크포인트 이후부터 재생
config_with_checkpoint = {
    "configurable": {
        "thread_id": "1",
        "checkpoint_id": "abc-123-def"
    }
}
graph.invoke(None, config_with_checkpoint)  # 이전 입력 재사용
```

### 메시지 관리 전략

긴 대화에서 LLM의 컨텍스트 제한을 초과하지 않기 위한 전략:

**1. 직접 삭제 방식**:
- 커스텀 리듀서로 메시지 개수 제한
- 최근 N개 메시지만 유지
- Tool Call과 Response 쌍 보존

**2. RemoveMessage 방식**:
- LangGraph 내장 메커니즘
- 메시지 ID 기반으로 정확하게 삭제
- 더 안전하고 권장되는 방법

**3. 요약 방식** (Part 2에서 다룸):
- 오래된 메시지를 요약으로 압축
- 컨텍스트는 유지하면서 토큰 절약

## 🛠 환경 설정

### 필요한 라이브러리 설치

```bash
pip install langchain langchain-openai langchain-chroma
pip install langgraph
pip install python-dotenv
```

### API 키 설정

`.env` 파일:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 기본 설정 코드

```python
from dotenv import load_dotenv
load_dotenv()

# 기본 라이브러리
import os
from glob import glob
from pprint import pprint
import json

# LangChain 및 LangGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

from typing import List, Annotated, Optional, Union
from typing_extensions import TypedDict
from operator import add

print("환경 설정 완료!")
```

### 레스토랑 메뉴 도구 설정

이 가이드에서는 레스토랑 메뉴/와인 검색 도구를 사용합니다.

```python
# 임베딩 모델
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# 벡터스토어 로드
menu_db = Chroma(
    embedding_function=embeddings_model,
    collection_name="restaurant_menu",
    persist_directory="./chroma_db"
)

wine_db = Chroma(
    embedding_function=embeddings_model,
    collection_name="restaurant_wine",
    persist_directory="./chroma_db"
)

# 도구 정의
@tool
def search_menu(query: str, k: int = 2) -> List[Document]:
    """레스토랑 메뉴 정보를 검색합니다."""
    docs = menu_db.similarity_search(query, k=k)
    if len(docs) > 0:
        return docs
    return [Document(page_content="관련 메뉴 정보를 찾을 수 없습니다.")]

@tool
def search_wine(query: str, k: int = 2) -> List[Document]:
    """레스토랑 와인 정보를 검색합니다."""
    docs = wine_db.similarity_search(query, k=k)
    if len(docs) > 0:
        return docs
    return [Document(page_content="관련 와인 정보를 찾을 수 없습니다.")]
```

## 💻 단계별 구현

### 1단계: MemorySaver로 단기 메모리 구현

#### 상태 정의

```python
from typing import Annotated, Optional
from typing_extensions import TypedDict
from operator import add

class State(TypedDict):
    query: str                                # 사용자 질문
    search_results: Annotated[list[str], add] # 검색 결과 (누적)
    summary: Optional[str]                    # 요약
```

**포인트**:
- `search_results`는 `add` 리듀서로 누적
- `summary`는 `Optional`로 초기값 없을 수 있음

#### 노드 정의

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import ToolNode

# LLM에 도구 바인딩
llm = ChatOpenAI(model="gpt-4o-mini")
tools = [search_menu, search_wine]
llm_with_tools = llm.bind_tools(tools)

# 도구 노드
tool_node = ToolNode(tools=tools)

# 요약 체인
system_prompt = """
You are an AI assistant helping a user find information about a restaurant menu and wine list.
Answer in the same language as the user's query.
"""

user_prompt = """
Summarize the following search results.

[GUIDELINES]
- Provide a brief summary of the search results.
- Include the key information from the search results.
- Use 1-2 sentences to summarize the information.

[Search Results]
{search_results}

[Summary]
"""

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", user_prompt)
])

summary_chain = summary_prompt | llm

# 검색 노드
def search_node(state: State):
    """데이터베이스 검색 수행"""
    query = state['query']

    # LLM이 도구 선택
    tool_call = llm_with_tools.invoke(query)
    # 도구 실행
    tool_results = tool_node.invoke({"messages": [tool_call]})

    if tool_results['messages']:
        print(f"검색 문서 개수: {len(tool_results['messages'])}")
        return {"search_results": tool_results['messages']}

    return {"query": query}

# 요약 노드
def summarize_node(state: State):
    """검색 결과를 요약"""
    search_results = state['search_results']

    if search_results:
        summary_text = summary_chain.invoke({"search_results": search_results})
        summary = f"Summary of results for '{state['query']}': {summary_text.content.strip()}"
    else:
        summary = "No results found."

    return {"summary": summary}
```

#### 그래프 구성 및 컴파일

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# StateGraph 생성
workflow = StateGraph(State)

# 노드 추가
workflow.add_node("search", search_node)
workflow.add_node("summarize", summarize_node)

# 엣지 연결
workflow.add_edge(START, "search")
workflow.add_edge("search", "summarize")
workflow.add_edge("summarize", END)

# 메모리 저장소 생성
checkpointer = MemorySaver()

# 체크포인터를 지정하여 컴파일
graph_memory = workflow.compile(checkpointer=checkpointer)
```

**중요 포인트**:
- `checkpointer=checkpointer` 파라미터로 메모리 저장소 지정
- 이제 `graph_memory`는 모든 실행 단계에서 상태를 자동 저장

#### 그래프 시각화

```python
from IPython.display import Image, display
display(Image(graph_memory.get_graph().draw_mermaid_png()))
```

### 2단계: thread_id로 대화 세션 관리

```python
# thread_id 설정
config = {"configurable": {"thread_id": "1"}}

# 초기 쿼리
initial_input = {
    "query": "스테이크 메뉴가 있나요? 어울리는 와인도 추천해주세요."
}

# 그래프 실행
output = graph_memory.invoke(initial_input, config)

# 결과 출력
pprint(output)
```

**실행 결과**:
```
검색 문서 개수: 2
{'query': '스테이크 메뉴가 있나요? 어울리는 와인도 추천해주세요.',
 'search_results': [ToolMessage(...샤토브리앙 스테이크...),
                    ToolMessage(...와인 추천...)],
 'summary': "Summary of results for '...': 샤토브리앙 스테이크..."}
```

**여러 스레드 관리**:

```python
# 스레드 1: 첫 번째 사용자
config_1 = {"configurable": {"thread_id": "user_1"}}
graph_memory.invoke({"query": "파스타 메뉴 알려주세요"}, config_1)

# 스레드 2: 두 번째 사용자 (독립적)
config_2 = {"configurable": {"thread_id": "user_2"}}
graph_memory.invoke({"query": "디저트 메뉴 추천해주세요"}, config_2)

# 스레드 1로 돌아가기 (이전 컨텍스트 유지됨)
graph_memory.invoke({"query": "그 파스타 가격은요?"}, config_1)
```

### 3단계: 상태 조회 및 히스토리

#### 현재 상태 조회

```python
# 최신 상태 가져오기
current_state = graph_memory.get_state(config)

print(f"Config: {current_state.config}")
print(f"Metadata: {current_state.metadata}")
print(f"Next: {current_state.next}")
print(f"Tasks: {current_state.tasks}")
print(f"Values: {current_state.values}")
```

**출력 예시**:
```
Config: {'configurable': {'thread_id': '1', 'checkpoint_id': '1f0b5877-...'}}
Metadata: {'source': 'loop', 'step': 2, 'parents': {}}
Next: ()
Tasks: ()
Values: {'query': '...', 'search_results': [...], 'summary': '...'}
```

**StateSnapshot 속성**:
- `config`: thread_id와 checkpoint_id 포함
- `metadata`: 실행 정보 (source, step, parents)
- `next`: 다음 실행할 노드 (빈 튜플이면 완료)
- `tasks`: 다음 작업 정보
- `values`: 현재 상태 값

#### 상태 히스토리 조회

```python
# 전체 실행 히스토리 가져오기
state_history = list(graph_memory.get_state_history(config))

for i, snapshot in enumerate(state_history):
    print(f"Checkpoint {i}:")
    print(f"  Next: {snapshot.next}")
    print(f"  Metadata: {snapshot.metadata}")
    print(f"  Values (query): {snapshot.values.get('query', 'N/A')}")
    print("-" * 80)
```

**출력 예시**:
```
Checkpoint 0:
  Next: ()
  Metadata: {'source': 'loop', 'step': 2, 'parents': {}}
  Values (query): 스테이크 메뉴가 있나요?
--------------------------------------------------------------------------------
Checkpoint 1:
  Next: ('summarize',)
  Metadata: {'source': 'loop', 'step': 1, 'parents': {}}
  Values (query): 스테이크 메뉴가 있나요?
--------------------------------------------------------------------------------
Checkpoint 2:
  Next: ('search',)
  Metadata: {'source': 'loop', 'step': 0, 'parents': {}}
  Values (query): 스테이크 메뉴가 있나요?
```

**히스토리 순서**:
- 인덱스 0: 가장 최근 체크포인트
- 인덱스 N: 가장 오래된 체크포인트
- 역순으로 저장됨

### 4단계: 상태 재생 (Replay)

특정 체크포인트부터 그래프를 다시 실행합니다.

```python
# 'summarize' 노드가 실행되기 직전 체크포인트 찾기
snapshot_before_summarize = None
for snapshot in state_history:
    if snapshot.next == ('summarize',):
        snapshot_before_summarize = snapshot
        break

print(f"Found snapshot: {snapshot_before_summarize.config}")

# 해당 체크포인트부터 재생
output = graph_memory.invoke(None, snapshot_before_summarize.config)

# None을 전달하면 이전 입력을 재사용
# 'search' 노드는 재실행하지 않고, 'summarize' 노드만 실행
```

**재생 흐름**:
```
[Checkpoint: search 완료]
  ↓ (재생 시작점)
[summarize 노드 실행]
  ↓
[END]
```

**재생 후 히스토리 확인**:

```python
# 재생 후 히스토리 확인
new_history = list(graph_memory.get_state_history(snapshot_before_summarize.config))

print(f"재생 후 체크포인트 수: {len(new_history)}")
for i, snapshot in enumerate(new_history[:3]):
    print(f"Checkpoint {i}: next={snapshot.next}, step={snapshot.metadata['step']}")
```

**재생의 장점**:
- 불필요한 재실행 방지 (시간 절약)
- API 호출 비용 절감 (LLM, 도구 호출 재사용)
- 디버깅 및 테스트에 유용

### 5단계: 상태 업데이트

실행 중인 그래프의 상태를 직접 수정할 수 있습니다.

```python
# 특정 체크포인트 선택
checkpoint_config = snapshot_before_summarize.config

# 쿼리 수정
update_input = {
    "query": "메뉴 이름과 가격 정보만 간단하게 출력하세요."
}

# 상태 업데이트
graph_memory.update_state(checkpoint_config, update_input)

# 업데이트된 상태 확인
updated_state = graph_memory.get_state(config)
print(f"업데이트된 쿼리: {updated_state.values['query']}")
```

**상태 업데이트 파라미터**:
- `config`: 업데이트할 체크포인트 설정
- `values`: 업데이트할 값 (딕셔너리)
- `as_node`: 업데이트를 수행할 노드 지정 (선택 사항)

**업데이트 후 재생**:

```python
# 업데이트된 상태로 이어서 실행
output = graph_memory.invoke(None, config)

# 최종 상태 확인
final_state = graph_memory.get_state(config)
pprint(final_state.values)
```

**활용 시나리오**:
- 사용자 피드백 반영: "더 자세히", "더 간단하게"
- 오류 수정: 잘못된 입력 교정
- 디버깅: 특정 상태 값으로 테스트

### 6단계: 긴 대화 관리 - 직접 메시지 삭제

#### 커스텀 메시지 관리자 구현

```python
from typing import Union, Annotated

def manage_list(existing: list, updates: Union[list, dict]):
    """커스텀 메시지 관리 리듀서"""

    # 업데이트가 리스트인 경우: 기존 메시지에 추가
    if isinstance(updates, list):
        return existing + updates

    # 업데이트가 딕셔너리인 경우: 메시지 관리 작업 수행
    elif isinstance(updates, dict) and updates["type"] == "keep":
        # 지정된 범위의 메시지만 선택
        recent_messages = existing[updates["from"]:updates["to"]]

        # Tool Call과 Response 쌍 + 일반 메시지 보존
        kept_indices = set()

        for i, msg in enumerate(recent_messages):
            # Tool Call이 있는 메시지
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                # 다음 메시지가 ToolMessage인지 확인
                if i+1 < len(recent_messages) and isinstance(recent_messages[i+1], ToolMessage):
                    kept_indices.add(i)     # Tool Call 메시지
                    kept_indices.add(i+1)   # Tool Response 메시지
            # 일반 메시지 (Tool Call 아님)
            elif not isinstance(msg, ToolMessage):
                kept_indices.add(i)

        # 원본 순서 유지하면서 선택된 메시지만 반환
        return [msg for i, msg in enumerate(recent_messages) if i in kept_indices]

    return existing

# 상태 정의
class GraphState(MessagesState):
    messages: Annotated[list, manage_list]  # 커스텀 리듀서 적용
```

#### 메시지 관리 노드 구현

```python
def message_manager(state: GraphState):
    """최근 5개 메시지만 유지"""
    return {
        "messages": {"type": "keep", "from": -5, "to": None}
    }

# 에이전트 노드
def call_model(state: GraphState):
    system_prompt = SystemMessage("""You are a helpful AI assistant...""")

    messages = [system_prompt] + state['messages']
    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}
```

#### 그래프 구성

```python
# LLM에 도구 바인딩
llm_with_tools = llm.bind_tools(tools=[search_menu, search_wine])

# 그래프 구성
builder = StateGraph(GraphState)

builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(tools))
builder.add_node("message_manager", message_manager)

# 엣지 연결
builder.add_edge(START, "message_manager")  # 먼저 메시지 관리
builder.add_edge("message_manager", "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "message_manager")  # 도구 실행 후 다시 메시지 관리

# 메모리와 함께 컴파일
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
```

**그래프 흐름**:
```
START → message_manager → agent → tools → message_manager → agent → END
         (최근 5개 유지)            (도구 실행)   (최근 5개 유지)
```

#### 테스트

```python
config = {"configurable": {"thread_id": "test_1"}}

# 여러 질문 연속 실행
questions = [
    "스테이크 메뉴가 있나요?",
    "가격은 얼마인가요?",
    "어울리는 와인 추천해주세요",
    "다른 메뉴도 있나요?",
    "디저트 메뉴는요?",
    "가장 인기 있는 메뉴는?",
]

for q in questions:
    print(f"\n질문: {q}")
    result = graph.invoke({"messages": [HumanMessage(content=q)]}, config)

    # 현재 메시지 수 확인
    current_state = graph.get_state(config)
    print(f"메시지 수: {len(current_state.values['messages'])}")

    # 마지막 응답 출력
    result['messages'][-1].pretty_print()
```

**결과**:
- 메시지 수가 5개로 제한됨
- Tool Call과 Response 쌍은 보존됨
- 오래된 메시지는 자동으로 삭제됨

### 7단계: RemoveMessage를 사용한 메시지 삭제

LangGraph 내장 방식으로 더 안전하게 메시지를 삭제합니다.

#### RemoveMessage 개념

```python
from langgraph.graph import MessagesState
from langchain_core.messages import RemoveMessage

# RemoveMessage는 메시지 ID를 기반으로 삭제
remove_msg = RemoveMessage(id="message_id_to_remove")
```

**특징**:
- 메시지 ID 기반으로 정확하게 삭제
- Tool Call과 Response 관계 유지
- add_messages 리듀서와 자동 통합

#### 메시지 필터 노드 구현

```python
def filter_messages(state: GraphState):
    """오래된 메시지 삭제"""
    messages = state['messages']

    # 최근 6개 메시지만 유지 (시스템 프롬프트 제외)
    if len(messages) > 6:
        # 삭제할 메시지 ID 수집
        messages_to_remove = []

        for msg in messages[:-6]:  # 마지막 6개 제외한 나머지
            # 시스템 메시지는 유지
            if not isinstance(msg, SystemMessage):
                messages_to_remove.append(RemoveMessage(id=msg.id))

        return {"messages": messages_to_remove}

    return {}  # 삭제할 메시지 없음
```

#### 그래프 구성

```python
# 상태 정의 (일반 MessagesState 사용)
class GraphState(MessagesState):
    pass

# 그래프 구성
builder = StateGraph(GraphState)

builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(tools))
builder.add_node("filter", filter_messages)

# 엣지 연결
builder.add_edge(START, "filter")  # 먼저 필터링
builder.add_edge("filter", "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "filter")  # 도구 실행 후 다시 필터링

# 컴파일
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
```

#### 테스트 및 검증

```python
config = {"configurable": {"thread_id": "remove_test"}}

# 10개의 질문 연속 실행
for i in range(10):
    result = graph.invoke({
        "messages": [HumanMessage(content=f"질문 {i+1}: 메뉴 추천해주세요")]
    }, config)

    # 메시지 수 확인
    state = graph.get_state(config)
    msg_count = len(state.values['messages'])
    print(f"질문 {i+1} 후 메시지 수: {msg_count}")

# 최종 메시지 확인
final_state = graph.get_state(config)
for msg in final_state.values['messages']:
    print(f"{type(msg).__name__}: {msg.content[:50]}...")
```

**출력 예시**:
```
질문 1 후 메시지 수: 2
질문 2 후 메시지 수: 4
질문 3 후 메시지 수: 6
질문 4 후 메시지 수: 6  # 최대 6개 유지
질문 5 후 메시지 수: 6
...
```

## 🎯 실습 문제

### 실습 1: 체크포인터를 사용한 다국어 RAG (난이도: ⭐⭐⭐)

**문제**: 한국어/영어 DB를 사용하는 ReAct 에이전트에 체크포인터를 추가하여 대화 기록을 유지하세요.

**요구사항**:
- 한국어 DB 도구 (`search_kor`)와 영어 DB 도구 (`search_eng`) 정의
- ReAct 에이전트 그래프 구성 (`tools_condition` 사용)
- `MemorySaver`를 사용한 체크포인터 추가
- 연속된 대화에서 이전 컨텍스트 유지 확인
- 상태 히스토리 조회 및 출력

**힌트**:
```python
# 벡터스토어 로드
db_korean = Chroma(
    embedding_function=embeddings_openai,
    collection_name="db_korean_cosine_metadata",
    persist_directory="./chroma_db"
)

# 도구 정의
@tool
def search_kor(query: str, k: int = 2) -> List[Document]:
    """한국어 질문이 주어지면, 한국어 문서에서 정보를 검색합니다."""
    # 구현하세요
    pass
```

**테스트 시나리오**:
1. "테슬라의 창업자는 누구인가요?" (한국어)
2. "설립 년도는 언제인가요?" (이전 대화 참조)
3. "Who is the CEO of Tesla?" (영어, 새로운 컨텍스트)
4. 상태 히스토리 출력

### 실습 2: 선택적 메시지 제거 (난이도: ⭐⭐⭐)

**문제**: Tool Call이 있는 메시지만 선택적으로 제거하는 필터를 구현하세요.

**요구사항**:
- Tool Call 메시지와 ToolMessage 쌍을 선택적으로 삭제
- 일반 대화 메시지는 모두 보존
- 최근 2개의 Tool Call 쌍만 유지
- RemoveMessage 사용

**힌트**:
```python
def filter_tool_messages(state: GraphState):
    """Tool Call 메시지만 선택적으로 제거"""
    messages = state['messages']

    # Tool Call이 있는 메시지 쌍 찾기
    tool_pairs = []
    for i, msg in enumerate(messages):
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            if i+1 < len(messages) and isinstance(messages[i+1], ToolMessage):
                tool_pairs.append((i, i+1))

    # 최근 2개 쌍만 유지, 나머지 삭제
    if len(tool_pairs) > 2:
        # 구현하세요
        pass
```

**테스트 쿼리**:
- "메뉴 추천해주세요" (Tool Call 발생)
- "와인도 추천해주세요" (Tool Call 발생)
- "감사합니다" (일반 메시지)
- "가격은 얼마인가요?" (Tool Call 발생)
- "좋아요" (일반 메시지)

**기대 결과**:
- 3개의 Tool Call 쌍 중 2개만 유지
- 모든 일반 메시지는 보존

### 실습 3: 사용자별 대화 컨텍스트 관리 (난이도: ⭐⭐⭐⭐)

**문제**: 여러 사용자의 독립적인 대화를 관리하는 시스템을 구현하세요.

**요구사항**:
- 사용자별 thread_id 자동 생성
- 사용자별 메시지 히스토리 조회 함수
- 사용자별 컨텍스트 초기화 기능
- 최소 3명의 사용자 동시 대화 시뮬레이션

**힌트**:
```python
class ConversationManager:
    def __init__(self, graph):
        self.graph = graph

    def get_config(self, user_id: str):
        """사용자별 config 생성"""
        return {"configurable": {"thread_id": f"user_{user_id}"}}

    def chat(self, user_id: str, message: str):
        """사용자 메시지 처리"""
        config = self.get_config(user_id)
        # 구현하세요

    def get_history(self, user_id: str):
        """사용자의 대화 히스토리 조회"""
        # 구현하세요

    def clear_history(self, user_id: str):
        """사용자의 대화 기록 초기화"""
        # 구현하세요
```

**테스트 시나리오**:
```python
manager = ConversationManager(graph)

# 사용자 A
manager.chat("alice", "스테이크 메뉴 알려주세요")
manager.chat("alice", "가격은요?")

# 사용자 B (독립적)
manager.chat("bob", "파스타 메뉴 추천해주세요")
manager.chat("bob", "와인도 추천해주세요")

# 사용자 C (독립적)
manager.chat("charlie", "디저트 메뉴는요?")

# 각 사용자의 히스토리 조회
print(manager.get_history("alice"))
print(manager.get_history("bob"))
print(manager.get_history("charlie"))
```

## ✅ 솔루션 예시

### 실습 1 솔루션

```python
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

# 임베딩 모델
embeddings_openai = OpenAIEmbeddings(model="text-embedding-3-small")

# 벡터스토어 로드
db_korean = Chroma(
    embedding_function=embeddings_openai,
    collection_name="db_korean_cosine_metadata",
    persist_directory="./chroma_db"
)

db_english = Chroma(
    embedding_function=embeddings_openai,
    collection_name="eng_db_openai",
    persist_directory="./chroma_db"
)

print(f"한국어 문서 수: {db_korean._collection.count()}")
print(f"영어 문서 수: {db_english._collection.count()}")

# 도구 정의
@tool
def search_kor(query: str, k: int = 2) -> List[Document]:
    """한국어 질문이 주어지면, 한국어 문서에서 정보를 검색합니다."""
    docs = db_korean.similarity_search(query, k=k)
    if len(docs) > 0:
        return docs
    return [Document(page_content="관련 정보를 찾을 수 없습니다.")]

@tool
def search_eng(query: str, k: int = 2) -> List[Document]:
    """영어 질문이 주어지면, 영어 문서에서 정보를 검색합니다."""
    docs = db_english.similarity_search(query, k=k)
    if len(docs) > 0:
        return docs
    return [Document(page_content="No relevant information found.")]

# 상태 정의
class GraphState(MessagesState):
    pass

# LLM 및 도구
llm = ChatOpenAI(model="gpt-4o-mini")
tools = [search_kor, search_eng]
llm_with_tools = llm.bind_tools(tools=tools)

# 에이전트 노드
def call_model(state: GraphState):
    system_prompt = SystemMessage("""You are a helpful AI assistant.
Please respond to the user's query to the best of your ability!

중요: 사용자의 질문 언어와 동일한 언어로 답변해야 합니다.
답변 시 반드시 정보의 출처를 명시하세요: [도구: 도구이름]""")

    messages = [system_prompt] + state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# 그래프 구성
builder = StateGraph(GraphState)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

# 메모리 추가
memory = MemorySaver()
graph_memory = builder.compile(checkpointer=memory)

# 테스트 시나리오
config = {"configurable": {"thread_id": "multilang_test"}}

print("\n" + "="*80)
print("질문 1: 테슬라의 창업자는 누구인가요?")
print("="*80)
result = graph_memory.invoke({
    "messages": [HumanMessage(content="테슬라의 창업자는 누구인가요?")]
}, config)
result['messages'][-1].pretty_print()

print("\n" + "="*80)
print("질문 2: 설립 년도는 언제인가요?")
print("="*80)
result = graph_memory.invoke({
    "messages": [HumanMessage(content="설립 년도는 언제인가요?")]
}, config)
result['messages'][-1].pretty_print()

print("\n" + "="*80)
print("질문 3: Who is the CEO of Tesla?")
print("="*80)
result = graph_memory.invoke({
    "messages": [HumanMessage(content="Who is the CEO of Tesla?")]
}, config)
result['messages'][-1].pretty_print()

# 상태 히스토리 출력
print("\n" + "="*80)
print("상태 히스토리")
print("="*80)
history = list(graph_memory.get_state_history(config))
print(f"총 체크포인트 수: {len(history)}")
for i, snapshot in enumerate(history[:5]):  # 최근 5개만
    print(f"\nCheckpoint {i}:")
    print(f"  Next: {snapshot.next}")
    print(f"  Step: {snapshot.metadata.get('step', 'N/A')}")
    print(f"  Messages: {len(snapshot.values['messages'])}")
```

**실행 결과**:
```
================================================================================
질문 1: 테슬라의 창업자는 누구인가요?
================================================================================
================================== Ai Message ==================================
테슬라(Tesla)는 2003년 7월 1일에 Martin Eberhard와 Marc Tarpenning에 의해 설립되었습니다.
[도구: search_kor]

================================================================================
질문 2: 설립 년도는 언제인가요?
================================================================================
================================== Ai Message ==================================
테슬라는 2003년에 설립되었습니다. [도구: search_kor]

================================================================================
질문 3: Who is the CEO of Tesla?
================================================================================
================================== Ai Message ==================================
Elon Musk is the CEO of Tesla. He became CEO in 2008. [도구: search_eng]

================================================================================
상태 히스토리
================================================================================
총 체크포인트 수: 15

Checkpoint 0:
  Next: ()
  Step: 8
  Messages: 6

Checkpoint 1:
  Next: ('agent',)
  Step: 7
  Messages: 6
...
```

**솔루션 포인트**:
- 체크포인터로 대화 기록 유지
- 질문 2에서 "테슬라"를 명시하지 않아도 이전 컨텍스트 활용
- 한국어/영어 도구가 자동으로 선택됨
- 상태 히스토리에서 전체 실행 흐름 확인 가능

### 실습 2 솔루션

```python
from langchain_core.messages import RemoveMessage

def filter_tool_messages(state: GraphState):
    """Tool Call 메시지만 선택적으로 제거 (최근 2개 쌍 유지)"""
    messages = state['messages']

    # Tool Call 쌍 찾기 (AIMessage with tool_calls + ToolMessage)
    tool_pairs = []
    for i, msg in enumerate(messages):
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            # 다음 메시지가 ToolMessage인지 확인
            if i+1 < len(messages) and isinstance(messages[i+1], ToolMessage):
                tool_pairs.append((i, i+1, msg.id, messages[i+1].id))

    # 최근 2개 쌍만 유지하고 나머지 삭제
    if len(tool_pairs) > 2:
        # 삭제할 메시지 ID 수집 (오래된 쌍들)
        messages_to_remove = []
        for pair in tool_pairs[:-2]:  # 마지막 2개 제외
            messages_to_remove.append(RemoveMessage(id=pair[2]))  # AIMessage
            messages_to_remove.append(RemoveMessage(id=pair[3]))  # ToolMessage

        return {"messages": messages_to_remove}

    return {}

# 그래프 구성
builder = StateGraph(GraphState)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode([search_menu, search_wine]))
builder.add_node("filter", filter_tool_messages)

builder.add_edge(START, "filter")
builder.add_edge("filter", "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "filter")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# 테스트
config = {"configurable": {"thread_id": "filter_test"}}

queries = [
    "메뉴 추천해주세요",          # Tool Call 1
    "와인도 추천해주세요",         # Tool Call 2
    "감사합니다",                 # 일반 메시지
    "가격은 얼마인가요?",          # Tool Call 3
    "좋아요",                     # 일반 메시지
]

for q in queries:
    print(f"\n질문: {q}")
    result = graph.invoke({"messages": [HumanMessage(content=q)]}, config)

    # 현재 상태 확인
    state = graph.get_state(config)
    messages = state.values['messages']

    # Tool Call 쌍 수 세기
    tool_count = sum(1 for msg in messages if hasattr(msg, 'tool_calls') and msg.tool_calls)
    print(f"현재 메시지 수: {len(messages)}, Tool Call 쌍: {tool_count}")

# 최종 메시지 확인
print("\n" + "="*80)
print("최종 메시지 목록")
print("="*80)
final_state = graph.get_state(config)
for i, msg in enumerate(final_state.values['messages']):
    msg_type = type(msg).__name__
    content = msg.content[:50] if hasattr(msg, 'content') else "N/A"
    has_tools = "Yes" if hasattr(msg, 'tool_calls') and msg.tool_calls else "No"
    print(f"{i+1}. {msg_type} (Tool Call: {has_tools}): {content}...")
```

**실행 결과**:
```
질문: 메뉴 추천해주세요
현재 메시지 수: 2, Tool Call 쌍: 1

질문: 와인도 추천해주세요
현재 메시지 수: 4, Tool Call 쌍: 2

질문: 감사합니다
현재 메시지 수: 6, Tool Call 쌍: 2

질문: 가격은 얼마인가요?
현재 메시지 수: 6, Tool Call 쌍: 2  # Tool Call 1이 삭제됨

질문: 좋아요
현재 메시지 수: 8, Tool Call 쌍: 2

================================================================================
최종 메시지 목록
================================================================================
1. HumanMessage (Tool Call: No): 와인도 추천해주세요...
2. AIMessage (Tool Call: Yes): ...
3. ToolMessage (Tool Call: No): [Document(...)]
4. HumanMessage (Tool Call: No): 감사합니다...
5. AIMessage (Tool Call: No): 천만에요...
6. HumanMessage (Tool Call: No): 가격은 얼마인가요?...
7. AIMessage (Tool Call: Yes): ...
8. ToolMessage (Tool Call: No): [Document(...)]
9. HumanMessage (Tool Call: No): 좋아요...
10. AIMessage (Tool Call: No): 감사합니다...
```

**솔루션 포인트**:
- Tool Call 쌍만 선택적으로 삭제
- 일반 대화 메시지는 모두 보존
- 최근 2개 Tool Call 쌍 유지
- RemoveMessage로 안전하게 삭제

### 실습 3 솔루션

```python
class ConversationManager:
    """여러 사용자의 대화를 관리하는 클래스"""

    def __init__(self, graph):
        self.graph = graph

    def get_config(self, user_id: str):
        """사용자별 config 생성"""
        return {"configurable": {"thread_id": f"user_{user_id}"}}

    def chat(self, user_id: str, message: str):
        """사용자 메시지 처리"""
        config = self.get_config(user_id)

        print(f"\n[{user_id}] {message}")
        result = self.graph.invoke({
            "messages": [HumanMessage(content=message)]
        }, config)

        # 마지막 응답 출력
        response = result['messages'][-1]
        print(f"[Assistant] {response.content[:100]}...")

        return result

    def get_history(self, user_id: str, limit: int = 10):
        """사용자의 대화 히스토리 조회"""
        config = self.get_config(user_id)
        state = self.graph.get_state(config)

        messages = state.values.get('messages', [])

        print(f"\n{'='*80}")
        print(f"대화 히스토리: {user_id}")
        print('='*80)
        print(f"총 메시지 수: {len(messages)}")

        for i, msg in enumerate(messages[-limit:], 1):
            msg_type = type(msg).__name__
            content = msg.content[:60] if hasattr(msg, 'content') else "N/A"
            print(f"{i}. {msg_type}: {content}...")

        return messages

    def clear_history(self, user_id: str):
        """사용자의 대화 기록 초기화"""
        config = self.get_config(user_id)

        # 새로운 초기 상태로 업데이트
        self.graph.update_state(config, {"messages": []})

        print(f"\n[{user_id}] 대화 기록이 초기화되었습니다.")

    def get_all_users(self):
        """활성 사용자 목록 조회"""
        # MemorySaver는 실제로 모든 thread_id를 조회하는 기능이 없으므로
        # 실제 구현에서는 별도 추적 필요
        print("활성 사용자 추적은 별도 구현 필요")

# 테스트
manager = ConversationManager(graph_memory)

print("="*80)
print("다중 사용자 대화 시뮬레이션")
print("="*80)

# 사용자 A (Alice)
manager.chat("alice", "스테이크 메뉴 알려주세요")
manager.chat("alice", "가격은요?")

# 사용자 B (Bob) - 독립적
manager.chat("bob", "파스타 메뉴 추천해주세요")
manager.chat("bob", "와인도 추천해주세요")

# 사용자 C (Charlie) - 독립적
manager.chat("charlie", "디저트 메뉴는요?")

# 각 사용자로 다시 대화 (컨텍스트 유지 확인)
manager.chat("alice", "그 스테이크 주문할게요")  # 이전 대화 참조
manager.chat("bob", "그 와인 가격은요?")         # 이전 대화 참조

# 히스토리 조회
manager.get_history("alice")
manager.get_history("bob")
manager.get_history("charlie")

# Alice의 히스토리 초기화
manager.clear_history("alice")

# 초기화 후 대화
manager.chat("alice", "안녕하세요")  # 이전 컨텍스트 없음
manager.get_history("alice")
```

**실행 결과**:
```
================================================================================
다중 사용자 대화 시뮬레이션
================================================================================

[alice] 스테이크 메뉴 알려주세요
[Assistant] 저희 레스토랑의 스테이크 메뉴를 소개해드리겠습니다: 1. 샤토브리앙 스테이크 (₩42,000) ...

[alice] 가격은요?
[Assistant] 샤토브리앙 스테이크는 ₩42,000입니다. [도구: search_menu]

[bob] 파스타 메뉴 추천해주세요
[Assistant] 저희 레스토랑의 파스타 메뉴를 추천해드리겠습니다...

[bob] 와인도 추천해주세요
[Assistant] 파스타와 잘 어울리는 와인을 추천해드리겠습니다...

[charlie] 디저트 메뉴는요?
[Assistant] 디저트 메뉴를 안내해드리겠습니다...

[alice] 그 스테이크 주문할게요
[Assistant] 샤토브리앙 스테이크 주문 도와드리겠습니다...

[bob] 그 와인 가격은요?
[Assistant] 추천해드린 와인의 가격은 ...

================================================================================
대화 히스토리: alice
================================================================================
총 메시지 수: 6
1. HumanMessage: 스테이크 메뉴 알려주세요...
2. AIMessage: 저희 레스토랑의 스테이크 메뉴를...
3. HumanMessage: 가격은요?...
4. AIMessage: 샤토브리앙 스테이크는 ₩42,000입니다...
5. HumanMessage: 그 스테이크 주문할게요...
6. AIMessage: 샤토브리앙 스테이크 주문 도와드리겠습니다...

[alice] 대화 기록이 초기화되었습니다.

[alice] 안녕하세요
[Assistant] 안녕하세요! 무엇을 도와드릴까요?...
```

**솔루션 포인트**:
- 사용자별 독립적인 thread_id 관리
- 각 사용자의 컨텍스트 완전 분리
- 이전 대화 참조 가능 ("그 스테이크", "그 와인")
- 대화 초기화 기능으로 컨텍스트 리셋

## 🚀 실무 활용 예시

### 예시 1: 챗봇 대화 기록 관리

웹 애플리케이션에서 사용자별 챗봇 대화를 관리합니다.

```python
from datetime import datetime
import uuid

class ChatbotSession:
    """챗봇 세션 관리자"""

    def __init__(self, graph):
        self.graph = graph
        self.sessions = {}  # session_id -> user_info

    def create_session(self, user_id: str, metadata: dict = None):
        """새 세션 생성"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "user_id": user_id,
            "created_at": datetime.now(),
            "metadata": metadata or {}
        }

        config = {"configurable": {"thread_id": session_id}}
        return session_id, config

    def send_message(self, session_id: str, message: str):
        """메시지 전송"""
        if session_id not in self.sessions:
            raise ValueError("Invalid session ID")

        config = {"configurable": {"thread_id": session_id}}
        result = self.graph.invoke({
            "messages": [HumanMessage(content=message)]
        }, config)

        # 마지막 응답 반환
        return result['messages'][-1].content

    def get_conversation(self, session_id: str):
        """대화 내역 조회"""
        if session_id not in self.sessions:
            raise ValueError("Invalid session ID")

        config = {"configurable": {"thread_id": session_id}}
        state = self.graph.get_state(config)

        conversation = []
        for msg in state.values['messages']:
            if isinstance(msg, HumanMessage):
                conversation.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                conversation.append({"role": "assistant", "content": msg.content})

        return conversation

    def end_session(self, session_id: str):
        """세션 종료"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            print(f"Session {session_id} ended")

# 사용 예시
chatbot = ChatbotSession(graph_memory)

# 사용자 1 세션
session1, config1 = chatbot.create_session(
    "user_001",
    metadata={"name": "홍길동", "language": "ko"}
)

print(chatbot.send_message(session1, "안녕하세요"))
print(chatbot.send_message(session1, "메뉴 추천해주세요"))

# 사용자 2 세션 (독립적)
session2, config2 = chatbot.create_session(
    "user_002",
    metadata={"name": "John", "language": "en"}
)

print(chatbot.send_message(session2, "Hello"))
print(chatbot.send_message(session2, "Recommend a menu"))

# 대화 내역 조회
conv1 = chatbot.get_conversation(session1)
print(f"\nUser 1 Conversation: {len(conv1)} messages")
for msg in conv1:
    print(f"  {msg['role']}: {msg['content'][:50]}...")

# 세션 종료
chatbot.end_session(session1)
```

### 예시 2: 고객 지원 티켓 시스템

고객 지원 티켓에 대화 기록을 연결합니다.

```python
class SupportTicket:
    """고객 지원 티켓 관리"""

    def __init__(self, graph):
        self.graph = graph
        self.tickets = {}

    def create_ticket(self, customer_id: str, issue: str):
        """티켓 생성"""
        ticket_id = f"TICKET-{len(self.tickets) + 1:04d}"

        self.tickets[ticket_id] = {
            "customer_id": customer_id,
            "issue": issue,
            "status": "open",
            "created_at": datetime.now(),
            "thread_id": f"ticket_{ticket_id}"
        }

        # 초기 메시지 전송
        config = {"configurable": {"thread_id": f"ticket_{ticket_id}"}}
        self.graph.invoke({
            "messages": [HumanMessage(content=f"Issue: {issue}")]
        }, config)

        return ticket_id

    def add_message(self, ticket_id: str, message: str, role: str = "customer"):
        """티켓에 메시지 추가"""
        if ticket_id not in self.tickets:
            raise ValueError("Invalid ticket ID")

        config = {"configurable": {"thread_id": f"ticket_{ticket_id}"}}

        if role == "customer":
            result = self.graph.invoke({
                "messages": [HumanMessage(content=message)]
            }, config)
            return result['messages'][-1].content
        else:
            # 지원 담당자 메시지
            self.graph.update_state(config, {
                "messages": [AIMessage(content=message)]
            })
            return message

    def get_ticket_history(self, ticket_id: str):
        """티켓 대화 이력"""
        if ticket_id not in self.tickets:
            raise ValueError("Invalid ticket ID")

        config = {"configurable": {"thread_id": f"ticket_{ticket_id}"}}
        state = self.graph.get_state(config)

        ticket_info = self.tickets[ticket_id]
        messages = state.values['messages']

        return {
            "ticket_id": ticket_id,
            "customer_id": ticket_info["customer_id"],
            "status": ticket_info["status"],
            "created_at": ticket_info["created_at"],
            "message_count": len(messages),
            "messages": messages
        }

    def close_ticket(self, ticket_id: str):
        """티켓 종료"""
        if ticket_id in self.tickets:
            self.tickets[ticket_id]["status"] = "closed"
            print(f"Ticket {ticket_id} closed")

# 사용 예시
support = SupportTicket(graph_memory)

# 티켓 1: 메뉴 문의
ticket1 = support.create_ticket("CUST-001", "스테이크 메뉴 정보 필요")
print(support.add_message(ticket1, "가격과 재료 알려주세요"))
print(support.add_message(ticket1, "감사합니다"))

# 티켓 2: 예약 문의
ticket2 = support.create_ticket("CUST-002", "4명 예약 가능 시간")
print(support.add_message(ticket2, "이번 주 금요일 저녁 가능한가요?"))

# 지원 담당자 메시지 추가
support.add_message(ticket1, "도움이 되어 기쁩니다. 추가 문의사항 있으시면 말씀해주세요.", role="agent")

# 티켓 이력 조회
history = support.get_ticket_history(ticket1)
print(f"\nTicket {history['ticket_id']}")
print(f"Customer: {history['customer_id']}")
print(f"Status: {history['status']}")
print(f"Messages: {history['message_count']}")

# 티켓 종료
support.close_ticket(ticket1)
```

### 예시 3: A/B 테스트 및 실험 추적

다양한 프롬프트나 설정을 테스트하고 추적합니다.

```python
class ExperimentTracker:
    """A/B 테스트 및 실험 추적"""

    def __init__(self, graph_a, graph_b):
        self.graph_a = graph_a  # 변형 A
        self.graph_b = graph_b  # 변형 B
        self.experiments = {}

    def run_experiment(self, experiment_id: str, test_queries: List[str]):
        """실험 실행"""
        results = {
            "experiment_id": experiment_id,
            "variant_a": [],
            "variant_b": [],
            "timestamp": datetime.now()
        }

        for i, query in enumerate(test_queries):
            # 변형 A 테스트
            config_a = {"configurable": {"thread_id": f"{experiment_id}_a_{i}"}}
            result_a = self.graph_a.invoke({
                "messages": [HumanMessage(content=query)]
            }, config_a)

            # 변형 B 테스트
            config_b = {"configurable": {"thread_id": f"{experiment_id}_b_{i}"}}
            result_b = self.graph_b.invoke({
                "messages": [HumanMessage(content=query)]
            }, config_b)

            results["variant_a"].append({
                "query": query,
                "response": result_a['messages'][-1].content,
                "message_count": len(result_a['messages'])
            })

            results["variant_b"].append({
                "query": query,
                "response": result_b['messages'][-1].content,
                "message_count": len(result_b['messages'])
            })

        self.experiments[experiment_id] = results
        return results

    def compare_results(self, experiment_id: str):
        """결과 비교"""
        if experiment_id not in self.experiments:
            raise ValueError("Experiment not found")

        exp = self.experiments[experiment_id]

        print(f"\n{'='*80}")
        print(f"Experiment: {experiment_id}")
        print('='*80)

        for i in range(len(exp["variant_a"])):
            print(f"\nQuery {i+1}: {exp['variant_a'][i]['query']}")
            print(f"\nVariant A Response:")
            print(f"  {exp['variant_a'][i]['response'][:100]}...")
            print(f"  Messages: {exp['variant_a'][i]['message_count']}")

            print(f"\nVariant B Response:")
            print(f"  {exp['variant_b'][i]['response'][:100]}...")
            print(f"  Messages: {exp['variant_b'][i]['message_count']}")
            print("-" * 80)

# 사용 예시
# 두 가지 프롬프트 변형으로 그래프 생성
# (실제로는 시스템 프롬프트나 설정을 다르게)
tracker = ExperimentTracker(graph_memory, graph_memory)

# 실험 실행
test_queries = [
    "스테이크 메뉴 추천해주세요",
    "와인 페어링 알려주세요",
    "가격대는 어떻게 되나요?"
]

results = tracker.run_experiment("exp_001", test_queries)
tracker.compare_results("exp_001")
```

## 📖 참고 자료

### 공식 문서
- [LangGraph Persistence 가이드](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [Memory and Checkpointing](https://langchain-ai.github.io/langgraph/how-tos/memory/)
- [MemorySaver API](https://langchain-ai.github.io/langgraph/reference/checkpoints/#langgraph.checkpoint.memory.MemorySaver)
- [Managing Message History](https://langchain-ai.github.io/langgraph/how-tos/manage-conversation-history/)

### 체크포인터 구현
- [SqliteSaver](https://langchain-ai.github.io/langgraph/reference/checkpoints/#langgraph.checkpoint.sqlite.SqliteSaver) - 로컬 파일 기반
- [PostgresSaver](https://langchain-ai.github.io/langgraph/reference/checkpoints/#langgraph.checkpoint.postgres.PostgresSaver) - 데이터베이스 기반
- [커스텀 체크포인터 구현](https://langchain-ai.github.io/langgraph/how-tos/persistence_postgres/)

### 메시지 관리 전략
- [Trimming and Filtering Messages](https://langchain-ai.github.io/langgraph/how-tos/memory/manage-conversation-history/)
- [RemoveMessage 사용법](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.modifier.RemoveMessage.html)
- [메시지 요약 전략](https://langchain-ai.github.io/langgraph/how-tos/memory/summary/)

### 추가 학습 자료
- [Time Travel and Replay](https://langchain-ai.github.io/langgraph/concepts/low_level/#time-travel)
- [Human-in-the-Loop with Checkpoints](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/review_tool_calls/)
- [Multi-User Applications](https://langchain-ai.github.io/langgraph/concepts/multi_tenancy/)

---

**다음 학습**: [LangGraph 메모리 관리 - Part 2: 장기 메모리 (InMemoryStore, 크로스 스레드 메모리)]
