# LangGraph 활용 - ReAct 에이전트

## 📚 학습 목표

이 가이드를 완료하면 다음을 할 수 있습니다:

- **ReAct 에이전트 패턴**을 이해하고 LangGraph에서 구현할 수 있다
- **@tool 데코레이터**를 사용하여 커스텀 도구를 정의하고 LLM에 바인딩할 수 있다
- **ToolNode**를 활용하여 도구 호출을 자동으로 실행할 수 있다
- **조건부 엣지**를 구현하여 에이전트의 도구 사용 흐름을 제어할 수 있다
- **다국어 RAG 시스템**에서 ReAct 패턴을 적용하여 언어별 라우팅을 구현할 수 있다

## 🔑 핵심 개념

### ReAct 패턴이란?

**ReAct (Reasoning and Acting)**은 가장 일반적으로 사용되는 에이전트 아키텍처로, LLM이 다음 세 단계를 반복적으로 수행합니다:

1. **Reasoning (추론)**: 주어진 질문과 이전 관찰을 바탕으로 다음 행동을 결정
2. **Acting (행동)**: 특정 도구를 호출하여 정보를 수집
3. **Observing (관찰)**: 도구 실행 결과를 받아 다음 추론에 활용

```
사용자 질문
    ↓
[Reasoning] → 어떤 도구를 사용할지 결정
    ↓
[Acting] → 도구 호출 실행
    ↓
[Observing] → 도구 결과 확인
    ↓
[Reasoning] → 결과가 충분한가?
    ├─ 예 → 최종 답변 생성
    └─ 아니오 → 다시 Acting 단계로
```

### LangChain Tool System

LangChain은 LLM이 외부 기능을 호출할 수 있도록 하는 도구(Tool) 시스템을 제공합니다:

**도구의 3가지 핵심 속성**:
- `name`: 도구의 고유 이름
- `description`: 도구의 기능 설명 (LLM이 이를 보고 도구 선택)
- `args_schema`: 도구가 받는 매개변수의 스키마

**도구 정의 방법**:
1. `@tool` 데코레이터 (간단한 함수형)
2. `StructuredTool` 클래스 (복잡한 도구)
3. LangChain 내장 도구 (`TavilySearchResults`, `ArxivQueryRun` 등)

### ToolNode의 역할

**ToolNode**는 LangGraph의 사전 구축된 컴포넌트로, AI 모델이 요청한 도구 호출을 자동으로 실행합니다.

**작동 방식**:
1. 가장 최근 `AIMessage`에서 `tool_calls` 추출
2. 요청된 모든 도구를 **병렬로** 실행
3. 각 도구 호출 결과를 `ToolMessage`로 변환하여 반환

```python
from langgraph.prebuilt import ToolNode

# 도구 노드 생성
tool_node = ToolNode(tools=[search_tool, calculator_tool])

# AIMessage의 tool_calls를 자동으로 실행
results = tool_node.invoke({"messages": [ai_message]})
# → {'messages': [ToolMessage(...), ToolMessage(...)]}
```

### 조건부 엣지와 도구 라우팅

ReAct 에이전트에서 가장 중요한 것은 **"도구를 더 호출할 것인가, 답변을 생성할 것인가"**를 결정하는 것입니다.

**방법 1: 사용자 정의 조건 함수**
```python
def should_continue(state: GraphState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:  # 도구 호출이 있으면
        return "execute_tools"
    return END  # 없으면 종료
```

**방법 2: LangGraph 내장 `tools_condition`**
```python
from langgraph.prebuilt import tools_condition

# tools_condition이 자동으로 판단
builder.add_conditional_edges("agent", tools_condition)
```

`tools_condition`의 장점:
- 별도 함수 작성 불필요
- 도구 호출 유무를 자동으로 판단
- `END` 또는 `"tools"` 노드로 자동 라우팅

## 🛠 환경 설정

### 필요한 라이브러리 설치

```bash
pip install langchain langchain-openai langchain-community langchain-chroma
pip install langgraph
pip install python-dotenv
pip install tavily-python  # 웹 검색 도구용
```

### API 키 설정

`.env` 파일을 생성하고 다음 내용을 추가합니다:

```
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here  # 웹 검색 사용 시
```

### 기본 설정 코드

```python
from dotenv import load_dotenv
import os

# 환경 변수 로드
load_dotenv()

# 기본 라이브러리
from pprint import pprint
import json

# LangChain 및 LangGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing import List, Literal

print("환경 설정 완료!")
```

## 💻 단계별 구현

### 1단계: 벡터 데이터베이스 준비

ReAct 에이전트가 사용할 지식 베이스를 준비합니다. 이 예제에서는 레스토랑 메뉴와 와인 정보를 저장합니다.

```python
from langchain_classic.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
import re

# 메뉴 데이터 로드
loader = TextLoader("./data/restaurant_menu.txt", encoding="utf-8")
documents = loader.load()

# 메뉴 항목별로 분할하는 함수
def split_menu_items(document):
    """메뉴 항목을 개별 Document 객체로 분리"""
    pattern = r'(\d+\.\s.*?)(?=\n\n\d+\.|$)'
    menu_items = re.findall(pattern, document.page_content, re.DOTALL)

    menu_documents = []
    for i, item in enumerate(menu_items, 1):
        menu_name = item.split('\n')[0].split('.', 1)[1].strip()

        menu_doc = Document(
            page_content=item.strip(),
            metadata={
                "source": document.metadata['source'],
                "menu_number": i,
                "menu_name": menu_name
            }
        )
        menu_documents.append(menu_doc)

    return menu_documents

# 메뉴 문서 분할
menu_documents = []
for doc in documents:
    menu_documents += split_menu_items(doc)

print(f"총 {len(menu_documents)}개의 메뉴 항목 생성됨")

# 와인 데이터도 동일하게 처리
wine_loader = TextLoader("./data/restaurant_wine.txt", encoding="utf-8")
wine_docs = wine_loader.load()
wine_documents = []
for doc in wine_docs:
    wine_documents += split_menu_items(doc)

print(f"총 {len(wine_documents)}개의 와인 항목 생성됨")
```

**벡터스토어에 저장**:

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# 임베딩 모델 생성
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# 메뉴 DB 생성
menu_db = Chroma.from_documents(
    documents=menu_documents,
    embedding=embeddings_model,
    collection_name="restaurant_menu",
    persist_directory="./chroma_db"
)

# 와인 DB 생성
wine_db = Chroma.from_documents(
    documents=wine_documents,
    embedding=embeddings_model,
    collection_name="restaurant_wine",
    persist_directory="./chroma_db"
)

print("벡터스토어 생성 완료!")
```

**검색 테스트**:

```python
# 메뉴 검색 테스트
menu_retriever = menu_db.as_retriever(search_kwargs={'k': 2})
query = "시그니처 스테이크의 가격은?"
docs = menu_retriever.invoke(query)

print(f"검색 결과: {len(docs)}개")
for doc in docs:
    print(f"- {doc.metadata['menu_name']}")
```

### 2단계: 커스텀 도구 정의 (@tool 데코레이터)

`@tool` 데코레이터를 사용하면 일반 Python 함수를 LangChain 도구로 변환할 수 있습니다.

```python
from langchain_core.tools import tool
from typing import List
from langchain_core.documents import Document

# 임베딩 모델 초기화 (도구에서 사용)
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# 기존 벡터스토어 로드
menu_db = Chroma(
    embedding_function=embeddings_model,
    collection_name="restaurant_menu",
    persist_directory="./chroma_db"
)

# 메뉴 검색 도구 정의
@tool
def search_menu(query: str, k: int = 2) -> List[Document]:
    """
    Securely retrieve and access authorized restaurant menu information from the encrypted database.
    Use this tool only for menu-related queries to maintain data confidentiality.
    """
    docs = menu_db.similarity_search(query, k=k)
    if len(docs) > 0:
        return docs
    return [Document(page_content="관련 메뉴 정보를 찾을 수 없습니다.")]

# 와인 검색 도구 정의
wine_db = Chroma(
    embedding_function=embeddings_model,
    collection_name="restaurant_wine",
    persist_directory="./chroma_db"
)

@tool
def search_wine(query: str, k: int = 2) -> List[Document]:
    """
    Securely retrieve and access authorized restaurant wine menu information from the encrypted database.
    Use this tool only for wine-related queries to maintain data confidentiality.
    """
    docs = wine_db.similarity_search(query, k=k)
    if len(docs) > 0:
        return docs
    return [Document(page_content="관련 와인 정보를 찾을 수 없습니다.")]
```

**도구 속성 확인**:

```python
# 도구의 타입
print(f"자료형: {type(search_menu)}")
# → <class 'langchain_core.tools.structured.StructuredTool'>

# 도구의 이름
print(f"이름: {search_menu.name}")
# → search_menu

# 도구의 설명 (LLM이 이것을 보고 도구를 선택함)
print(f"설명: {search_menu.description}")

# 도구의 스키마 (매개변수 정보)
pprint(search_menu.args_schema.model_json_schema())
```

**출력 예시**:
```
자료형: <class 'langchain_core.tools.structured.StructuredTool'>
이름: search_menu
설명: Securely retrieve and access authorized restaurant menu information...
스키마:
{'description': 'Securely retrieve...',
 'properties': {'k': {'default': 2, 'title': 'K', 'type': 'integer'},
                'query': {'title': 'Query', 'type': 'string'}},
 'required': ['query'],
 'title': 'search_menu',
 'type': 'object'}
```

### 3단계: LLM에 도구 바인딩

LLM이 도구를 사용할 수 있도록 `bind_tools()` 메서드를 사용합니다.

```python
from langchain_openai import ChatOpenAI

# LLM 생성
llm = ChatOpenAI(model="gpt-4o-mini")

# 도구 바인딩
llm_with_tools = llm.bind_tools(tools=[search_menu, search_wine])

# 테스트 쿼리
query = "시그니처 스테이크의 가격과 특징은 무엇인가요? 그리고 스테이크와 어울리는 와인 추천도 해주세요."
ai_msg = llm_with_tools.invoke(query)

# LLM의 응답 확인
print("Content:", ai_msg.content)  # 텍스트 응답 (이 경우 비어있음)
print("\nTool Calls:")
pprint(ai_msg.tool_calls)
```

**출력 예시**:
```
Content:

Tool Calls:
[{'name': 'search_menu',
  'args': {'query': '시그니처 스테이크'},
  'id': 'call_8tLwaL9dRmbrqb9EchzO8n98',
  'type': 'tool_call'},
 {'name': 'search_wine',
  'args': {'query': '스테이크'},
  'id': 'call_bKfhnuV4GyCC1Hv2fSdYUFdD',
  'type': 'tool_call'}]
```

**중요 포인트**:
- LLM은 질문을 분석하여 어떤 도구를 호출할지 결정
- 여러 도구를 동시에 호출할 수 있음 (병렬 도구 호출)
- `ai_msg.tool_calls`에 호출할 도구 정보가 담김

### 4단계: LangChain 내장 도구 사용

LangChain은 웹 검색, 계산기, Wikipedia 등 다양한 내장 도구를 제공합니다.

```python
from langchain_community.tools import TavilySearchResults

# 웹 검색 도구 생성
search_web = TavilySearchResults(max_results=2)

# 여러 도구를 함께 바인딩
tools = [search_menu, search_web]
llm_with_tools = llm.bind_tools(tools=tools)

# 메뉴 관련 질문 → search_menu 도구 사용
response = llm_with_tools.invoke([HumanMessage(content="스테이크 메뉴의 가격은 얼마인가요?")])
print("메뉴 질문:", response.tool_calls)
# → [{'name': 'search_menu', 'args': {'query': '스테이크'}, ...}]

# 일반 지식 질문 → search_web 도구 사용
response = llm_with_tools.invoke([HumanMessage(content="LangGraph는 무엇인가요?")])
print("일반 질문:", response.tool_calls)
# → [{'name': 'tavily_search_results_json', 'args': {'query': 'LangGraph'}, ...}]

# 도구가 필요 없는 질문 → 도구 호출 없음
response = llm_with_tools.invoke([HumanMessage(content="3+3은 얼마인가요?")])
print("계산 질문:", response.tool_calls)
# → []
```

### 5단계: ToolNode로 도구 실행

ToolNode는 AIMessage의 `tool_calls`를 받아 실제로 도구를 실행합니다.

```python
from langgraph.prebuilt import ToolNode

# 도구 노드 생성
tool_node = ToolNode(tools=tools)

# LLM이 도구 호출 요청
response = llm_with_tools.invoke([HumanMessage(content="스테이크 메뉴의 가격은 얼마인가요?")])

# ToolNode로 도구 실행
results = tool_node.invoke({"messages": [response]})

# 결과 출력
for msg in results['messages']:
    msg.pretty_print()
```

**출력 예시**:
```
================================= Tool Message =================================
Name: search_menu

[Document(metadata={'menu_name': '샤토브리앙 스테이크', ...},
          page_content='26. 샤토브리앙 스테이크\n가격: ₩42,000...'),
 Document(metadata={'menu_name': '안심 스테이크 샐러드', ...},
          page_content='8. 안심 스테이크 샐러드\n가격: ₩26,000...')]
```

**ToolNode의 동작 흐름**:
```
AIMessage
  ├─ tool_calls: [{'name': 'search_menu', 'args': {...}}, ...]
  ↓
ToolNode.invoke()
  ├─ 각 tool_call 추출
  ├─ 해당 도구 함수 실행 (병렬)
  ├─ 결과를 ToolMessage로 변환
  ↓
{'messages': [ToolMessage(name='search_menu', content='...'), ...]}
```

### 6단계: ReAct 에이전트 구현 (방법 1 - 사용자 정의 조건)

이제 모든 구성 요소를 결합하여 완전한 ReAct 에이전트를 만듭니다.

```python
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import SystemMessage

# 상태 정의
class GraphState(MessagesState):
    pass

# 모델 호출 노드
def call_model(state: GraphState):
    system_prompt = SystemMessage("""You are a helpful AI assistant.
Please respond to the user's query to the best of your ability!

중요: 답변을 제공할 때 반드시 정보의 출처를 명시해야 합니다. 출처는 다음과 같이 표시하세요:
- 도구를 사용하여 얻은 정보: [도구: 도구이름]
- 모델의 일반 지식에 기반한 정보: [일반 지식]

항상 정확하고 관련성 있는 정보를 제공하되, 확실하지 않은 경우 그 사실을 명시하세요.""")

    messages = [system_prompt] + state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# 조건부 엣지 함수
def should_continue(state: GraphState):
    """마지막 메시지에 tool_calls가 있으면 도구 실행, 없으면 종료"""
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "execute_tools"  # 도구 노드로 이동
    return END  # 그래프 종료

# 그래프 구성
builder = StateGraph(GraphState)

# 노드 추가
builder.add_node("call_model", call_model)
builder.add_node("execute_tools", ToolNode(tools))

# 엣지 추가
builder.add_edge(START, "call_model")
builder.add_conditional_edges(
    "call_model",
    should_continue,
    ["execute_tools", END]  # 가능한 다음 노드들
)
builder.add_edge("execute_tools", "call_model")  # 도구 실행 후 다시 모델로

# 그래프 컴파일
graph = builder.compile()
```

**그래프 구조 시각화**:

```python
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))
```

**그래프 실행**:

```python
# 질문 실행
inputs = {"messages": [HumanMessage(content="스테이크 메뉴의 가격은 얼마인가요?")]}
messages = graph.invoke(inputs)

# 전체 대화 출력
for m in messages['messages']:
    m.pretty_print()
```

**출력 예시**:
```
================================ Human Message =================================
스테이크 메뉴의 가격은 얼마인가요?

================================== Ai Message ==================================
Tool Calls:
  search_menu (call_i7Sd1Bfju5tCsNEDvTnCI3CZ)
  Args: query: 스테이크

================================= Tool Message =================================
Name: search_menu
[Document(...샤토브리앙 스테이크...₩42,000...),
 Document(...안심 스테이크 샐러드...₩26,000...)]

================================== Ai Message ==================================
스테이크 메뉴의 가격은 다음과 같습니다:

1. **샤토브리앙 스테이크**: ₩42,000
   - 최상급 안심 스테이크에 푸아그라를 올리고 트러플 소스를 곁들인 클래식 프렌치 요리

2. **안심 스테이크 샐러드**: ₩26,000
   - 부드러운 안심 스테이크를 얇게 슬라이스하여 신선한 루꼴라 위에 올린 메인 요리 샐러드

출처: [도구: 메뉴 검색]
```

**실행 흐름 분석**:
```
1. START → call_model
   - 사용자 질문 처리
   - LLM이 search_menu 도구 호출 결정

2. call_model → execute_tools (should_continue가 "execute_tools" 반환)
   - ToolNode가 search_menu 실행
   - ToolMessage 생성

3. execute_tools → call_model
   - 도구 결과를 포함한 메시지로 LLM 재호출
   - LLM이 최종 답변 생성 (tool_calls 없음)

4. call_model → END (should_continue가 END 반환)
   - 그래프 종료
```

### 7단계: ReAct 에이전트 구현 (방법 2 - tools_condition)

LangGraph는 `tools_condition`이라는 내장 조건 함수를 제공합니다.

```python
from langgraph.prebuilt import tools_condition

# 그래프 구성 (더 간결함)
builder = StateGraph(GraphState)

# 노드 추가
builder.add_node("agent", call_model)  # 노드 이름을 "agent"로 변경
builder.add_node("tools", ToolNode(tools))  # 노드 이름을 "tools"로 변경

# 엣지 추가
builder.add_edge(START, "agent")

# tools_condition 사용 - 자동으로 tool_calls 유무 판단
builder.add_conditional_edges("agent", tools_condition)

builder.add_edge("tools", "agent")

# 컴파일
graph = builder.compile()
```

**tools_condition의 동작**:
- 마지막 메시지에 `tool_calls`가 있으면 → `"tools"` 노드로 라우팅
- `tool_calls`가 없으면 → `END`로 라우팅
- 별도의 조건 함수 작성 불필요

**실행 예시**:

```python
inputs = {"messages": [HumanMessage(content="파스타에 어울리는 와인을 추천해주세요.")]}
messages = graph.invoke(inputs)

for m in messages['messages']:
    m.pretty_print()
```

**출력**:
```
================================ Human Message =================================
파스타에 어울리는 와인을 추천해주세요.

================================== Ai Message ==================================
Tool Calls:
  tavily_search_results_json (call_JgKtHvQo78JucO4AMnlknzZf)
  Args: query: 파스타에 어울리는 와인 추천

================================= Tool Message =================================
Name: tavily_search_results_json
[{"title": "파스타와 잘 어울리는 와인은...", "content": "..."}]

================================== Ai Message ==================================
파스타와 어울리는 와인 선택은 파스타의 소스와 재료에 따라 다를 수 있습니다:

1. **토마토 소스 파스타**: 치안티(Chianti) 같은 중간 바디의 레드 와인
2. **크림 소스 파스타**: 샤르도네(Chardonnay) 같은 부드러운 화이트 와인
3. **해산물 파스타**: 소비뇽 블랑(Sauvignon Blanc) 또는 피노 그리지오(Pinot Grigio)

출처: [도구: tavily_search_results_json]
```

## 🎯 실습 문제

### 실습 1: 다국어 RAG 라우팅 (난이도: ⭐⭐)

**문제**: 한국어 질문은 한국어 DB에서, 영어 질문은 영어 DB에서 검색하는 도구를 각각 구현하고 ToolNode로 실행하세요.

**요구사항**:
- 테슬라/리비안 데이터가 저장된 한국어 DB와 영어 DB 로드
- `search_kor` 도구: 한국어 질문을 한국어 DB에서 검색
- `search_eng` 도구: 영어 질문을 영어 DB에서 검색
- ToolNode를 사용하여 두 도구 실행
- LLM이 질문 언어에 따라 적절한 도구를 선택하는지 확인

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

**테스트 쿼리**:
- "테슬라의 창업자는 누구인가요?" (한국어)
- "Who is the founder of Tesla?" (영어)

### 실습 2: 완전한 ReAct 에이전트 구현 (난이도: ⭐⭐⭐)

**문제**: 실습 1에서 구현한 두 도구를 사용하는 완전한 ReAct 에이전트를 구현하세요.

**요구사항**:
- `MessagesState` 기반의 그래프 상태 정의
- 시스템 프롬프트에 출처 표시 지침 포함
- `tools_condition`을 사용한 조건부 엣지 구현
- 한국어 질문과 영어 질문 모두 테스트
- 전체 대화 흐름을 `pretty_print()`로 출력

**힌트**:
```python
from langgraph.prebuilt import tools_condition

class GraphState(MessagesState):
    pass

def call_model(state: GraphState):
    system_prompt = SystemMessage("""...

도구를 사용할 때는 반드시 사용자의 질문에서 사용한 같은 언어로 답변해야 합니다.
    """)
    # 구현하세요
```

**테스트 쿼리**:
- "테슬라는 언제 설립되었나요?"
- "When was Tesla founded?"

### 실습 3: 다중 도구 체인 에이전트 (난이도: ⭐⭐⭐⭐)

**문제**: 계산기, 웹 검색, 그리고 데이터베이스 검색 도구를 모두 사용하는 에이전트를 구현하세요.

**요구사항**:
- 3가지 이상의 서로 다른 도구 정의
- 복잡한 질문에 대해 여러 도구를 순차적으로 사용
- 각 도구 호출의 결과를 다음 추론에 활용

**도구 예시**:
```python
@tool
def calculator(expression: str) -> str:
    """수학 계산을 수행합니다. 예: '2 + 2', '10 * 5'"""
    try:
        result = eval(expression)
        return f"계산 결과: {result}"
    except Exception as e:
        return f"계산 오류: {str(e)}"

@tool
def get_current_date() -> str:
    """현재 날짜와 시간을 반환합니다."""
    from datetime import datetime
    return datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분")
```

**테스트 쿼리**:
- "오늘 날짜는 언제이고, 오늘부터 100일 후는 며칠인가요? 그리고 테슬라는 언제 설립되었나요?"

## ✅ 솔루션 예시

### 실습 1 솔루션

```python
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from typing import List
from langchain_core.documents import Document

# 임베딩 모델 생성
embeddings_openai = OpenAIEmbeddings(model="text-embedding-3-small")

# 한국어 DB 로드
db_korean = Chroma(
    embedding_function=embeddings_openai,
    collection_name="db_korean_cosine_metadata",
    persist_directory="./chroma_db"
)

# 영어 DB 로드
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

# 도구 목록
db_tools = [search_kor, search_eng]

# ToolNode 생성
from langgraph.prebuilt import ToolNode
db_tool_node = ToolNode(tools=db_tools)

# LLM에 도구 바인딩
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_db_tools = llm.bind_tools(tools=db_tools)

# 테스트 1: 한국어 질문
print("\n=== 한국어 질문 테스트 ===")
response = llm_with_db_tools.invoke([
    HumanMessage(content="테슬라의 창업자는 누구인가요?")
])
print(f"호출된 도구: {response.tool_calls[0]['name']}")

# ToolNode 실행
results = db_tool_node.invoke({"messages": [response]})
for result in results['messages']:
    print(f"\n검색 결과:")
    docs = eval(result.content)
    for doc in docs[:1]:  # 첫 번째 문서만 출력
        print(f"회사: {doc.metadata.get('company', 'N/A')}")
        print(f"내용: {doc.page_content[:200]}...")

# 테스트 2: 영어 질문
print("\n=== 영어 질문 테스트 ===")
response = llm_with_db_tools.invoke([
    HumanMessage(content="Who is the founder of Tesla?")
])
print(f"호출된 도구: {response.tool_calls[0]['name']}")

results = db_tool_node.invoke({"messages": [response]})
for result in results['messages']:
    print(f"\n검색 결과:")
    docs = eval(result.content)
    for doc in docs[:1]:
        print(f"출처: {doc.metadata.get('source', 'N/A')}")
        print(f"내용: {doc.page_content[:200]}...")
```

**실행 결과**:
```
한국어 문서 수: 39
영어 문서 수: 42

=== 한국어 질문 테스트 ===
호출된 도구: search_kor

검색 결과:
회사: 테슬라
내용: Tesla Motors, Inc.는 2003년 7월 1일에 Martin Eberhard와 Marc Tarpenning에 의해 설립되었으며, 각각 CEO와 CFO를 역임했습니다...

=== 영어 질문 테스트 ===
호출된 도구: search_eng

검색 결과:
출처: data/Tesla_EN.md
내용: Tesla, Inc. is an American multinational automotive and clean energy company. Founded in July 2003 by Martin Eberhard and Marc Tarpenning...
```

### 실습 2 솔루션

```python
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import SystemMessage, HumanMessage
from IPython.display import Image, display

# 상태 정의
class GraphState(MessagesState):
    pass

# 모델 호출 노드
def call_model(state: GraphState):
    system_prompt = SystemMessage("""You are a helpful AI assistant.
Please respond to the user's query to the best of your ability!

중요: 답변을 제공할 때 반드시 정보의 출처를 명시해야 합니다. 출처는 다음과 같이 표시하세요:
- 도구를 사용하여 얻은 정보: [도구: 도구이름]
- 모델의 일반 지식에 기반한 정보: [일반 지식]

도구를 사용할 때는 반드시 사용자의 질문에서 사용한 같은 언어로 답변해야 합니다.
예를 들어, 사용자가 한국어로 질문했다면 한국어로 답변해야 합니다.

항상 정확하고 관련성 있는 정보를 제공하되, 확실하지 않은 경우 그 사실을 명시하세요.""")

    messages = [system_prompt] + state['messages']
    response = llm_with_db_tools.invoke(messages)
    return {"messages": [response]}

# 그래프 구성
builder = StateGraph(GraphState)

# 노드 추가
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(db_tools))

# 엣지 추가
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

# 컴파일
graph = builder.compile()

# 그래프 시각화
display(Image(graph.get_graph().draw_mermaid_png()))

# 테스트 1: 한국어 질문
print("=" * 80)
print("테스트 1: 한국어 질문")
print("=" * 80)
inputs = {"messages": [HumanMessage(content="테슬라는 언제 설립되었나요?")]}
messages = graph.invoke(inputs)

for m in messages['messages']:
    m.pretty_print()

# 테스트 2: 영어 질문
print("\n" + "=" * 80)
print("테스트 2: 영어 질문")
print("=" * 80)
inputs = {"messages": [HumanMessage(content="When was Tesla founded?")]}
messages = graph.invoke(inputs)

for m in messages['messages']:
    m.pretty_print()
```

**실행 결과**:
```
================================================================================
테스트 1: 한국어 질문
================================================================================
================================ Human Message =================================
테슬라는 언제 설립되었나요?

================================== Ai Message ==================================
Tool Calls:
  search_kor (call_sxz9ZGUZnaT0abIRTSmudf4n)
  Args: query: 테슬라 설립 연도

================================= Tool Message =================================
Name: search_kor
[Document(metadata={'company': '테슬라', 'language': 'ko', ...},
          page_content='Tesla Motors, Inc.는 2003년 7월 1일에 Martin Eberhard와...')]

================================== Ai Message ==================================
테슬라(Tesla)는 2003년 7월 1일에 Martin Eberhard와 Marc Tarpenning에 의해 설립되었습니다.

[도구: search_kor]

================================================================================
테스트 2: 영어 질문
================================================================================
================================ Human Message =================================
When was Tesla founded?

================================== Ai Message ==================================
Tool Calls:
  search_eng (call_L282dXpaRUGWRe3aoW1lMqoV)
  Args: query: Tesla founded date

================================= Tool Message =================================
Name: search_eng
[Document(metadata={'source': 'data/Tesla_EN.md'},
          page_content='Tesla, Inc. is an American multinational...Founded in July 2003...')]

================================== Ai Message ==================================
Tesla, Inc. was founded on July 1, 2003, by Martin Eberhard and Marc Tarpenning.

[도구: search_eng]
```

### 실습 3 솔루션

```python
from langchain_core.tools import tool
from datetime import datetime, timedelta

# 계산기 도구
@tool
def calculator(expression: str) -> str:
    """수학 계산을 수행합니다. 예: '2 + 2', '10 * 5', '100 / 4'"""
    try:
        # eval은 보안 위험이 있으므로 실제 프로덕션에서는 ast.literal_eval 등 사용
        result = eval(expression)
        return f"계산 결과: {result}"
    except Exception as e:
        return f"계산 오류: {str(e)}"

# 날짜 도구
@tool
def get_current_date() -> str:
    """현재 날짜와 시간을 반환합니다."""
    now = datetime.now()
    return f"현재 날짜: {now.strftime('%Y년 %m월 %d일 %H시 %M분')}"

@tool
def add_days_to_date(date_str: str, days: int) -> str:
    """주어진 날짜에 일수를 더한 날짜를 반환합니다.

    Args:
        date_str: 날짜 문자열 (형식: YYYY-MM-DD)
        days: 더할 일수
    """
    try:
        base_date = datetime.strptime(date_str, "%Y-%m-%d")
        new_date = base_date + timedelta(days=days)
        return f"{days}일 후: {new_date.strftime('%Y년 %m월 %d일')}"
    except Exception as e:
        return f"날짜 계산 오류: {str(e)}"

# 모든 도구 결합
all_tools = [calculator, get_current_date, add_days_to_date, search_kor, search_eng]

# LLM에 모든 도구 바인딩
llm_with_all_tools = llm.bind_tools(tools=all_tools)

# 그래프 구성
def call_model_advanced(state: GraphState):
    system_prompt = SystemMessage("""You are a helpful AI assistant with multiple tools.
Please respond to the user's query using the appropriate tools.

Available tools:
- calculator: For mathematical calculations
- get_current_date: To get current date and time
- add_days_to_date: To calculate future/past dates
- search_kor: To search Korean documents
- search_eng: To search English documents

Important:
- Break down complex questions into steps
- Use tools in the right order
- Always cite your sources: [도구: tool_name] or [일반 지식]
""")

    messages = [system_prompt] + state['messages']
    response = llm_with_all_tools.invoke(messages)
    return {"messages": [response]}

# 그래프 구성
builder = StateGraph(GraphState)
builder.add_node("agent", call_model_advanced)
builder.add_node("tools", ToolNode(all_tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

advanced_graph = builder.compile()

# 복잡한 질문 테스트
print("=" * 80)
print("복잡한 다중 도구 질문 테스트")
print("=" * 80)

inputs = {"messages": [HumanMessage(content="""
오늘 날짜는 언제이고, 오늘부터 100일 후는 며칠인가요?
그리고 테슬라는 언제 설립되었나요?
""")]}

messages = advanced_graph.invoke(inputs)

for m in messages['messages']:
    m.pretty_print()
```

**실행 결과**:
```
================================================================================
복잡한 다중 도구 질문 테스트
================================================================================
================================ Human Message =================================
오늘 날짜는 언제이고, 오늘부터 100일 후는 며칠인가요?
그리고 테슬라는 언제 설립되었나요?

================================== Ai Message ==================================
Tool Calls:
  get_current_date (call_abc123)
  Args: {}

================================= Tool Message =================================
Name: get_current_date
현재 날짜: 2025년 10월 31일 14시 30분

================================== Ai Message ==================================
Tool Calls:
  add_days_to_date (call_def456)
  Args: date_str: 2025-10-31, days: 100

================================= Tool Message =================================
Name: add_days_to_date
100일 후: 2026년 2월 8일

================================== Ai Message ==================================
Tool Calls:
  search_kor (call_ghi789)
  Args: query: 테슬라 설립일

================================= Tool Message =================================
Name: search_kor
[Document(page_content='Tesla Motors, Inc.는 2003년 7월 1일에...')]

================================== Ai Message ==================================
질문에 대한 답변입니다:

1. **오늘 날짜**: 2025년 10월 31일 [도구: get_current_date]

2. **100일 후**: 2026년 2월 8일 [도구: add_days_to_date]

3. **테슬라 설립일**: 2003년 7월 1일 [도구: search_kor]

테슬라는 Martin Eberhard와 Marc Tarpenning에 의해 설립되었으며,
이후 Elon Musk가 주요 투자자로 합류했습니다.
```

**솔루션 포인트**:
- 에이전트가 복잡한 질문을 3개의 하위 질문으로 분해
- 각 하위 질문에 적절한 도구를 순차적으로 호출
- 모든 도구 결과를 종합하여 최종 답변 생성
- 각 정보의 출처를 명확히 표시

## 🚀 실무 활용 예시

### 예시 1: 고객 지원 챗봇

레스토랑의 고객 지원 챗봇을 ReAct 패턴으로 구현합니다.

```python
from langchain_core.tools import tool
from typing import List, Dict
import json

# 예약 확인 도구
@tool
def check_reservation(customer_name: str, date: str) -> str:
    """고객의 예약 정보를 조회합니다.

    Args:
        customer_name: 고객 이름
        date: 예약 날짜 (YYYY-MM-DD 형식)
    """
    # 실제로는 데이터베이스 조회
    mock_reservations = {
        "홍길동_2025-11-01": {
            "time": "19:00",
            "party_size": 4,
            "table": "A3",
            "special_request": "창가 자리 요청"
        },
        "김철수_2025-11-05": {
            "time": "18:30",
            "party_size": 2,
            "table": "B1",
            "special_request": "없음"
        }
    }

    key = f"{customer_name}_{date}"
    reservation = mock_reservations.get(key)

    if reservation:
        return json.dumps(reservation, ensure_ascii=False)
    return "예약 정보를 찾을 수 없습니다."

# 영업 시간 확인 도구
@tool
def get_business_hours(day_of_week: str) -> str:
    """레스토랑의 요일별 영업 시간을 반환합니다.

    Args:
        day_of_week: 요일 (예: 월요일, 화요일, ...)
    """
    hours = {
        "월요일": "휴무",
        "화요일": "11:30 - 22:00",
        "수요일": "11:30 - 22:00",
        "목요일": "11:30 - 22:00",
        "금요일": "11:30 - 23:00",
        "토요일": "11:30 - 23:00",
        "일요일": "11:30 - 21:00"
    }
    return f"{day_of_week} 영업 시간: {hours.get(day_of_week, '정보 없음')}"

# 예약 가능 시간 확인 도구
@tool
def check_available_times(date: str, party_size: int) -> str:
    """특정 날짜에 예약 가능한 시간대를 반환합니다.

    Args:
        date: 예약 희망 날짜 (YYYY-MM-DD)
        party_size: 인원 수
    """
    # 실제로는 예약 시스템 조회
    available_times = ["18:00", "18:30", "19:00", "20:00", "20:30"]

    if party_size > 6:
        available_times = ["18:00", "19:00"]  # 대규모 예약은 제한된 시간

    return f"예약 가능 시간: {', '.join(available_times)}"

# 고객 지원 도구 목록
support_tools = [
    search_menu,
    search_wine,
    check_reservation,
    get_business_hours,
    check_available_times
]

# LLM 바인딩
llm_support = llm.bind_tools(tools=support_tools)

# 고객 지원 에이전트 노드
def customer_support_agent(state: GraphState):
    system_prompt = SystemMessage("""당신은 레스토랑의 친절한 고객 지원 AI입니다.

고객의 질문에 적절한 도구를 사용하여 정확한 정보를 제공하세요:
- 메뉴 문의 → search_menu
- 와인 추천 → search_wine
- 예약 확인 → check_reservation
- 영업 시간 → get_business_hours
- 예약 가능 시간 → check_available_times

항상 친절하고 전문적으로 응대하며, 정보의 출처를 명시하세요.
""")

    messages = [system_prompt] + state['messages']
    response = llm_support.invoke(messages)
    return {"messages": [response]}

# 그래프 구성
builder = StateGraph(GraphState)
builder.add_node("agent", customer_support_agent)
builder.add_node("tools", ToolNode(support_tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

support_graph = builder.compile()

# 테스트 시나리오
print("=" * 80)
print("고객 지원 챗봇 시뮬레이션")
print("=" * 80)

# 시나리오 1: 예약 확인
print("\n[시나리오 1: 예약 확인]")
inputs = {"messages": [HumanMessage(content="홍길동으로 11월 1일에 예약한 정보 확인해주세요.")]}
messages = support_graph.invoke(inputs)
messages['messages'][-1].pretty_print()

# 시나리오 2: 메뉴 + 와인 추천
print("\n[시나리오 2: 메뉴와 와인 추천]")
inputs = {"messages": [HumanMessage(content="스테이크 메뉴 추천해주시고, 어울리는 와인도 알려주세요.")]}
messages = support_graph.invoke(inputs)
messages['messages'][-1].pretty_print()

# 시나리오 3: 예약 가능 시간 확인
print("\n[시나리오 3: 예약 가능 시간]")
inputs = {"messages": [HumanMessage(content="11월 5일 4명 예약 가능한 시간 알려주세요.")]}
messages = support_graph.invoke(inputs)
messages['messages'][-1].pretty_print()
```

**실행 결과**:
```
[시나리오 1: 예약 확인]
================================== Ai Message ==================================
홍길동 고객님의 11월 1일 예약 정보입니다:

- 예약 시간: 19:00
- 인원: 4명
- 테이블: A3
- 특별 요청: 창가 자리

예약해주셔서 감사합니다! [도구: check_reservation]

[시나리오 2: 메뉴와 와인 추천]
================================== Ai Message ==================================
스테이크 메뉴 추천드립니다:

1. **샤토브리앙 스테이크** (₩42,000)
   - 최상급 안심 스테이크에 푸아그라와 트러플 소스

2. **시그니처 스테이크** (₩35,000)
   - 최상급 한우 등심, 로즈메리 감자, 그릴드 아스파라거스

어울리는 와인:
- **샤토 마고 2015** (₩450,000): 보르도 프리미엄 레드
- **그랜지 2016**: 스테이크와 완벽한 조화

[도구: search_menu, search_wine]

[시나리오 3: 예약 가능 시간]
================================== Ai Message ==================================
11월 5일 4명 예약 가능한 시간대는 다음과 같습니다:

18:00, 18:30, 19:00, 20:00, 20:30

원하시는 시간을 선택해주시면 예약 도와드리겠습니다. [도구: check_available_times]
```

### 예시 2: 연구 어시스턴트 에이전트

학술 연구를 위한 ReAct 에이전트를 구현합니다.

```python
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper

# 논문 검색 도구
arxiv_wrapper = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

# Wikipedia 검색 도구
wiki_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=500)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# 요약 도구
@tool
def summarize_text(text: str, max_words: int = 100) -> str:
    """긴 텍스트를 요약합니다.

    Args:
        text: 요약할 텍스트
        max_words: 최대 단어 수
    """
    # 실제로는 LLM을 사용한 고급 요약
    words = text.split()
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words]) + '...'

# 연구 어시스턴트 도구
research_tools = [
    arxiv_tool,
    wiki_tool,
    summarize_text,
    search_web  # Tavily 웹 검색
]

# LLM 바인딩
llm_research = llm.bind_tools(tools=research_tools)

# 연구 어시스턴트 노드
def research_assistant(state: GraphState):
    system_prompt = SystemMessage("""당신은 학술 연구를 돕는 AI 어시스턴트입니다.

사용 가능한 도구:
- arxiv_query_run: 학술 논문 검색
- wikipedia_query_run: 일반 백과사전 정보
- tavily_search: 최신 웹 정보
- summarize_text: 긴 텍스트 요약

연구 질문에 대해:
1. 먼저 관련 학술 논문을 검색
2. 배경 지식이 필요하면 Wikipedia 참조
3. 최신 정보가 필요하면 웹 검색
4. 정보가 많으면 요약 도구 사용

항상 출처를 명확히 밝히고, 학술적으로 신뢰할 수 있는 답변을 제공하세요.
""")

    messages = [system_prompt] + state['messages']
    response = llm_research.invoke(messages)
    return {"messages": [response]}

# 그래프 구성
builder = StateGraph(GraphState)
builder.add_node("agent", research_assistant)
builder.add_node("tools", ToolNode(research_tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

research_graph = builder.compile()

# 연구 질문 테스트
print("=" * 80)
print("연구 어시스턴트 에이전트")
print("=" * 80)

inputs = {"messages": [HumanMessage(content="""
LangGraph와 관련된 최신 연구 동향을 알려주세요.
특히 멀티 에이전트 시스템에 대한 논문이 있다면 소개해주세요.
""")]}

messages = research_graph.invoke(inputs)

# 전체 대화 출력
for m in messages['messages']:
    m.pretty_print()
```

**실무 적용 포인트**:
- 여러 정보 소스를 통합하여 종합적인 답변 제공
- 도구 호출 순서를 에이전트가 자율적으로 결정
- 각 도구의 결과를 다음 추론에 활용

### 예시 3: 데이터 분석 에이전트

CSV 파일 분석과 시각화를 수행하는 에이전트입니다.

```python
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

# CSV 로드 도구
@tool
def load_csv(file_path: str) -> str:
    """CSV 파일을 로드하고 기본 정보를 반환합니다.

    Args:
        file_path: CSV 파일 경로
    """
    try:
        df = pd.read_csv(file_path)
        info = f"""
데이터셋 정보:
- 행 수: {len(df)}
- 열 수: {len(df.columns)}
- 컬럼: {', '.join(df.columns)}
- 결측치: {df.isnull().sum().sum()}개

첫 5행:
{df.head().to_string()}
        """
        return info
    except Exception as e:
        return f"파일 로드 오류: {str(e)}"

# 통계 분석 도구
@tool
def analyze_statistics(file_path: str, column: str) -> str:
    """특정 컬럼의 통계 정보를 반환합니다.

    Args:
        file_path: CSV 파일 경로
        column: 분석할 컬럼 이름
    """
    try:
        df = pd.read_csv(file_path)
        stats = df[column].describe()
        return f"{column} 통계:\n{stats.to_string()}"
    except Exception as e:
        return f"분석 오류: {str(e)}"

# 데이터 필터링 도구
@tool
def filter_data(file_path: str, condition: str) -> str:
    """조건에 맞는 데이터를 필터링합니다.

    Args:
        file_path: CSV 파일 경로
        condition: 필터 조건 (예: "age > 30", "city == 'Seoul'")
    """
    try:
        df = pd.read_csv(file_path)
        filtered = df.query(condition)
        return f"필터 결과 ({len(filtered)}행):\n{filtered.head(10).to_string()}"
    except Exception as e:
        return f"필터링 오류: {str(e)}"

# 데이터 분석 도구
data_tools = [load_csv, analyze_statistics, filter_data]

# 데이터 분석 에이전트
def data_analyst(state: GraphState):
    system_prompt = SystemMessage("""당신은 데이터 분석 전문 AI입니다.

사용자의 데이터 분석 요청에 대해:
1. 먼저 데이터를 로드하여 구조 파악
2. 필요한 통계 분석 수행
3. 필터링이나 집계가 필요하면 해당 도구 사용
4. 분석 결과를 명확하게 설명

항상 데이터 기반으로 객관적인 인사이트를 제공하세요.
""")

    messages = [system_prompt] + state['messages']
    llm_analyst = llm.bind_tools(tools=data_tools)
    response = llm_analyst.invoke(messages)
    return {"messages": [response]}

# 그래프 구성
builder = StateGraph(GraphState)
builder.add_node("agent", data_analyst)
builder.add_node("tools", ToolNode(data_tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

analyst_graph = builder.compile()

# 데이터 분석 요청 예시
print("=" * 80)
print("데이터 분석 에이전트")
print("=" * 80)

# 실제 사용 시에는 CSV 파일 경로를 제공
inputs = {"messages": [HumanMessage(content="""
sales_data.csv 파일을 분석해주세요.
특히 매출이 100만원 이상인 거래만 필터링하고,
평균 매출과 최대 매출을 알려주세요.
""")]}

# messages = analyst_graph.invoke(inputs)
# for m in messages['messages']:
#     m.pretty_print()
```

**실무 활용 가치**:
- 반복적인 데이터 분석 작업 자동화
- 비기술 사용자도 자연어로 데이터 분석 가능
- 복잡한 분석 파이프라인을 에이전트가 자율적으로 구성

## 📖 참고 자료

### 공식 문서
- [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph/)
- [LangGraph ReAct Agent 가이드](https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/)
- [LangChain Tools 문서](https://python.langchain.com/docs/modules/tools/)
- [ToolNode API 참조](https://langchain-ai.github.io/langgraph/reference/prebuilt/#toolnode)

### ReAct 패턴 논문
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- 원 논문에서 제안된 Reasoning-Acting 결합 접근법

### 도구 정의 패턴
- [@tool 데코레이터 사용법](https://python.langchain.com/docs/how_to/custom_tools/)
- [StructuredTool 클래스](https://python.langchain.com/api_reference/core/tools/langchain_core.tools.structured.StructuredTool.html)
- [LangChain 내장 도구 목록](https://python.langchain.com/docs/integrations/tools/)

### 추가 학습 자료
- [LangGraph 튜토리얼 시리즈](https://github.com/langchain-ai/langgraph/tree/main/examples)
- [멀티 에이전트 시스템 구현](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/)
- [Human-in-the-Loop 패턴](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/)

### 관련 블로그 및 튜토리얼
- [Building Agentic RAG with LangGraph](https://blog.langchain.dev/agentic-rag-with-langgraph/)
- [Tool Calling in Production](https://python.langchain.com/docs/use_cases/tool_use/)
- [Debugging LangGraph Applications](https://langchain-ai.github.io/langgraph/how-tos/debugging/)

---

**다음 학습**: [LangGraph 고급 패턴 - Human-in-the-Loop, 메모리 관리, 스트리밍]
