# LangGraph 활용 - Adaptive RAG (Part2-1) - StateGraph 기본 구조

## 📚 학습 목표

이 학습 가이드를 통해 다음을 달성할 수 있습니다:

1. **LangGraph StateGraph의 기본 구조**를 이해하고 상태 기반 워크플로우를 설계할 수 있다
2. **서브그래프(Subgraph) 패턴**을 활용하여 모듈화된 그래프 구조를 구축할 수 있다
3. **Command 객체**를 사용하여 동적 라우팅 및 상태 업데이트를 수행할 수 있다
4. **조건부 엣지(Conditional Edges)**를 통해 런타임에 다음 노드를 동적으로 결정할 수 있다
5. **쿼리 분석 서브그래프**를 구현하여 Adaptive RAG의 라우팅 전략을 자동화할 수 있다
6. **NoRetrieval 노드**를 LangGraph 워크플로우에 통합할 수 있다
7. **SingleShotRAG 노드**의 기본 구조(검색, 포맷팅, 생성)를 StateGraph에서 구현할 수 있다

## 🔑 핵심 개념

### LangGraph StateGraph란?

**LangGraph**는 LangChain 생태계의 상태 기반 워크플로우 오케스트레이션 라이브러리입니다. 복잡한 AI 애플리케이션을 **그래프 구조**로 표현하고 실행할 수 있습니다.

**주요 특징**:
- **상태 관리**: TypedDict 기반의 타입 안전한 상태
- **노드**: 상태를 변환하는 함수
- **엣지**: 노드 간 연결 (조건부 가능)
- **서브그래프**: 재사용 가능한 모듈화된 그래프
- **동적 라우팅**: Command 객체로 런타임 흐름 제어

### Adaptive RAG with LangGraph

Part1에서는 체인 기반으로 Adaptive RAG를 구현했다면, Part2에서는 **LangGraph StateGraph**를 사용하여:

1. **명확한 워크플로우 시각화**
2. **상태 기반 추적 및 디버깅**
3. **복잡한 조건부 로직 처리**
4. **모듈화 및 재사용성 향상**

### 핵심 구성 요소

#### 1. **StateGraph 구조**

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# 상태 정의
class MyState(TypedDict):
    field1: str
    field2: list

# 그래프 생성
workflow = StateGraph(MyState)

# 노드 추가
workflow.add_node("node_name", node_function)

# 엣지 추가
workflow.add_edge(START, "node_name")
workflow.add_edge("node_name", END)

# 컴파일
graph = workflow.compile()
```

#### 2. **Command 객체**

Command 객체는 노드에서 반환되어 다음 동작을 제어합니다:

```python
from langgraph.types import Command
from typing import Literal

def my_node(state: MyState) -> Command[Literal["next_node"]]:
    return Command(
        goto="next_node",  # 다음 노드 지정
        update={            # 상태 업데이트
            "field1": "new_value"
        }
    )
```

#### 3. **조건부 엣지 (Conditional Edges)**

상태를 기반으로 동적으로 다음 노드를 결정합니다:

```python
def route_function(state: MyState) -> str:
    if state["field1"] == "option_a":
        return "node_a"
    else:
        return "node_b"

workflow.add_conditional_edges(
    "source_node",
    route_function,
    {
        "node_a": "target_a",
        "node_b": "target_b"
    }
)
```

#### 4. **서브그래프 (Subgraph)**

독립적인 그래프를 노드로 사용:

```python
# 서브그래프 생성
subgraph_workflow = StateGraph(SubState)
# ... 노드 및 엣지 추가
subgraph = subgraph_workflow.compile()

# 메인 그래프에 서브그래프 추가
main_workflow.add_node("analysis", subgraph)
```

### Adaptive RAG 워크플로우

```
START
  ↓
AnalyzeQuery (서브그래프)
  ↓
[조건부 라우팅]
  ├→ NoRetrieval → Final → END
  ├→ SingleShotRAG → Final → END
  └→ IterativeRAG → Final → END
```

### 배경 지식

이 가이드를 학습하기 전에 다음 내용을 이해하고 있어야 합니다:

- **Adaptive RAG 기본 개념** (Part 2-0 참조)
- **LangChain 기초**: 체인, 프롬프트, 파서
- **Python TypedDict**: 타입 힌팅 기반 딕셔너리
- **그래프 이론 기초**: 노드, 엣지, DAG(Directed Acyclic Graph)

## 🛠 환경 설정

### 필요한 라이브러리 설치

```bash
# LangGraph (핵심)
pip install langgraph

# LangChain 생태계
pip install langchain langchain-openai langchain-chroma langchain-core

# Pydantic
pip install pydantic

# 벡터 데이터베이스
pip install chromadb

# 시각화
pip install pygraphviz  # 선택사항

# 모니터링
pip install langfuse

# 기타
pip install python-dotenv
```

### API 키 설정

`.env` 파일:

```bash
# OpenAI API 키 (필수)
OPENAI_API_KEY=your_openai_api_key_here

# Langfuse (선택사항)
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 기본 설정 코드

```python
# 환경 변수 로드
from dotenv import load_dotenv
load_dotenv()

# 기본 라이브러리
import os
from glob import glob
from pprint import pprint
import json
import warnings
warnings.filterwarnings("ignore")

# Langfuse 콜백 핸들러 (선택사항)
from langfuse.langchain import CallbackHandler
langfuse_handler = CallbackHandler()
```

## 💻 단계별 구현

### 단계 1: 상태 정의

StateGraph의 모든 노드가 공유하는 상태 구조를 정의합니다.

```python
from typing import TypedDict

class AdaptiveRagState(TypedDict):
    """Adaptive RAG 시스템의 전체 상태"""
    query: str                        # 입력 쿼리
    retrieval_strategy: str           # 선택된 검색 전략 (no_retrieval, single_lookup, iterative)
    retrieved_documents: list         # 검색된 문서
    intermediate_responses: list      # 중간 응답들 (IterativeRAG에서 사용)
    final_response: str               # 최종 응답
```

**핵심 포인트**:
- `TypedDict`: 타입 안전성 제공
- 모든 노드는 이 상태를 읽고 수정
- 상태는 그래프 실행 중 유지됨

### 단계 2: AnalyzeQuery 서브그래프 구현

쿼리를 분석하여 최적의 RAG 전략을 선택하는 독립적인 서브그래프입니다.

#### 2.1 서브그래프 상태 정의

```python
from typing import Literal
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START
from langgraph.types import Command

# 서브그래프 전용 상태
class QueryAnalysisState(TypedDict):
    """쿼리 분석 서브그래프의 상태"""
    query: str                    # 입력 쿼리
    retrieval_strategy: str       # 선택된 전략
    query_analysis: str           # 분석 결과 설명

# Pydantic 출력 스키마
class QueryRoute(BaseModel):
    """라우팅 결정 구조화"""
    strategy: Literal["no_retrieval", "single_lookup", "iterative"]
    reason: str
```

**핵심 포인트**:
- 서브그래프는 자체 상태를 가짐
- 부모 그래프 상태와 호환되는 필드 필요 (`query`, `retrieval_strategy`)

#### 2.2 서브그래프 노드 함수

```python
def create_analysis_graph():
    """쿼리 분석 서브그래프 생성"""

    def route_query(state: QueryAnalysisState):
        """쿼리를 분석하여 전략 선택"""

        router_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            당신은 레스토랑 서비스 담당자입니다.
            레스토랑에서 제공하는 메뉴와 와인에 대한 DB를 각각 보유하고 있습니다.

            주어진 질문을 분석하여 다음 전략 중 하나를 선택하세요:
            1. no_retrieval: 일반 상식, 간단한 계산 등 검색 불필요
            2. single_lookup: 단순 사실 확인, 정의 등 1회 검색
            3. iterative: 페어링, 비교, 복잡한 분석, 다단계 추론 필요한 경우
            """),
            ("user", "{query}")
        ])

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        chain = router_prompt | llm.with_structured_output(QueryRoute)
        routing = chain.invoke({"query": state["query"]})

        # Command로 상태 업데이트 및 종료
        return Command(
            goto=END,
            update={
                "retrieval_strategy": routing.strategy,
                "query_analysis": routing.reason
            }
        )

    # 서브그래프 구성
    workflow = StateGraph(QueryAnalysisState)
    workflow.add_node("route", route_query)
    workflow.add_edge(START, "route")

    return workflow.compile()

# 서브그래프 생성
analysis_graph = create_analysis_graph()
```

**핵심 포인트**:
- `Command(goto=END)`: 서브그래프 종료 및 부모로 복귀
- `update`: 부모 그래프 상태에 병합될 필드
- `with_structured_output()`: Pydantic 모델로 자동 파싱

#### 2.3 서브그래프 시각화 및 테스트

```python
from IPython.display import Image, display

# 시각화
display(Image(analysis_graph.get_graph().draw_mermaid_png()))

# 테스트
test_queries = [
    "안녕하세요!",
    "시그니처 스테이크의 가격은 얼마인가요?",
    "와인과 스테이크 페어링 추천해주세요",
]

for query in test_queries:
    initial_state = {"query": query}
    result = analysis_graph.invoke(initial_state)

    print(f"\n쿼리: {query}")
    print(f"전략: {result['retrieval_strategy']}")
    print(f"분석: {result['query_analysis']}")
    print("-" * 80)
```

**예상 결과**:
```
쿼리: 안녕하세요!
전략: no_retrieval
분석: 일반적인 인사말로 검색이 필요 없습니다.

쿼리: 시그니처 스테이크의 가격은 얼마인가요?
전략: single_lookup
분석: 메뉴 DB에서 단순 가격 조회로 충분합니다.

쿼리: 와인과 스테이크 페어링 추천해주세요
전략: iterative
분석: 메뉴와 와인 DB를 함께 검색하여 최적의 조합을 찾아야 합니다.
```

### 단계 3: 메인 그래프 기본 구조

서브그래프와 연결되는 메인 그래프를 구성합니다.

#### 3.1 노드 함수 정의 (시뮬레이션)

```python
from typing import Literal

def no_retrieval(state: AdaptiveRagState) -> Command[Literal["final"]]:
    """검색 없이 직접 응답 생성"""
    response = "검색 없이 직접 답변을 생성합니다."

    return Command(
        goto="final",
        update={
            "final_response": response,
            "intermediate_responses": state.get("intermediate_responses", []) + ["직접 응답 생성 완료"]
        }
    )

def single_lookup(state: AdaptiveRagState) -> Command[Literal["final"]]:
    """단일 검색 후 응답 생성"""
    # 실제 구현에서는 여기서 문서 검색
    documents = ["검색된 문서 1", "검색된 문서 2"]
    response = "단일 검색 결과를 바탕으로 답변을 생성합니다."

    return Command(
        goto="final",
        update={
            "retrieved_documents": documents,
            "final_response": response,
            "intermediate_responses": state.get("intermediate_responses", []) + ["단일 검색 및 응답 생성 완료"]
        }
    )

def iterative(state: AdaptiveRagState) -> Command[Literal["final"]]:
    """반복 검색 및 분석 후 응답 생성"""
    # 실제 구현에서는 여기서 다단계 검색 및 분석
    documents = ["상세 문서 1", "상세 문서 2", "상세 문서 3"]
    response = "다단계 검색 및 분석을 통해 답변을 생성합니다."

    return Command(
        goto="final",
        update={
            "retrieved_documents": documents,
            "final_response": response,
            "intermediate_responses": state.get("intermediate_responses", []) + ["다단계 검색 및 분석 완료"]
        }
    )

def final_node(state: AdaptiveRagState):
    """최종 응답 정리"""
    return {
        "final_response": state.get("final_response", "최종 응답이 없습니다."),
        "intermediate_responses": state.get("intermediate_responses", []) + ["최종 응답 생성 완료"]
    }
```

**핵심 포인트**:
- `Command[Literal["final"]]`: 타입 힌팅으로 다음 노드 명시
- `state.get()`: 안전한 상태 접근
- `intermediate_responses`: 실행 과정 추적

#### 3.2 라우팅 함수

```python
def route_to_strategy(state: AdaptiveRagState) -> str:
    """선택된 전략에 따라 다음 노드 결정"""
    return state["retrieval_strategy"]
```

**핵심 포인트**:
- 조건부 엣지에서 사용
- 상태 기반으로 문자열 반환 (노드 이름)

#### 3.3 메인 그래프 조립

```python
def create_main_graph():
    """메인 그래프 생성 및 조립"""

    # 메인 그래프 초기화
    workflow = StateGraph(AdaptiveRagState)

    # 서브그래프 생성
    analysis_graph = create_analysis_graph()

    # 노드 추가
    workflow.add_node("analysis", analysis_graph)     # 서브그래프
    workflow.add_node("no_retrieval", no_retrieval)
    workflow.add_node("single_lookup", single_lookup)
    workflow.add_node("iterative", iterative)
    workflow.add_node("final", final_node)

    # 엣지 추가
    workflow.add_edge(START, "analysis")  # 시작 → 분석

    # 조건부 엣지: 분석 결과에 따라 라우팅
    workflow.add_conditional_edges(
        "analysis",
        route_to_strategy,
        {
            "no_retrieval": "no_retrieval",
            "single_lookup": "single_lookup",
            "iterative": "iterative"
        }
    )

    workflow.add_edge("final", END)  # 최종 → 종료

    return workflow.compile()

# 메인 그래프 생성
main_graph = create_main_graph()
```

**핵심 포인트**:
- 서브그래프는 일반 노드처럼 추가
- `add_conditional_edges`: 동적 라우팅
- `START`와 `END`: 특수 노드

#### 3.4 메인 그래프 시각화 및 테스트

```python
# X-Ray 모드로 서브그래프 내부까지 시각화
display(Image(main_graph.get_graph(xray=True).draw_mermaid_png()))

# 테스트 실행
test_queries = [
    "안녕하세요!",
    "시그니처 스테이크의 가격은 얼마인가요?",
    "와인과 스테이크 페어링 추천해주세요",
]

for query in test_queries:
    print(f"\n{'='*100}")
    print(f"쿼리: {query}")
    print(f"{'='*100}")

    initial_state = {"query": query}
    result = main_graph.invoke(initial_state)

    print(f"\n[결과]")
    print(f"  전략: {result['retrieval_strategy']}")
    print(f"  최종 응답: {result['final_response']}")
    print(f"  실행 과정: {result['intermediate_responses']}")
```

**예상 결과**:
```
쿼리: 안녕하세요!

[결과]
  전략: no_retrieval
  최종 응답: 검색 없이 직접 답변을 생성합니다.
  실행 과정: ['직접 응답 생성 완료', '최종 응답 생성 완료']
```

### 단계 4: NoRetrieval 노드 실제 구현

시뮬레이션 함수를 실제 LLM 호출로 교체합니다.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def no_retrieval_response(query: str) -> str:
    """외부 지식 없이 LLM 내장 지식으로 직접 답변"""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 도움이 되는 인공지능 어시스턴트입니다. 외부 지식을 사용하지 않고 직접 질문에 답변하세요."),
        ("user", "{query}"),
    ])

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"query": query})

    return response

# 노드 함수 업데이트
def no_retrieval(state: AdaptiveRagState) -> Command[Literal["final"]]:
    """NoRetrieval 전략 실행"""
    response = no_retrieval_response(state["query"])

    return Command(
        goto="final",
        update={
            "final_response": response,
            "intermediate_responses": state.get("intermediate_responses", []) + [response]
        }
    )
```

**핵심 포인트**:
- Part1의 `no_retrieval_response` 함수 재사용
- StateGraph 노드에서 체인 호출
- 상태에 응답 저장

#### 업데이트된 메인 그래프 테스트

```python
# 그래프 재생성 (업데이트된 노드 함수 반영)
main_graph = create_main_graph()

# 시각화
display(Image(main_graph.get_graph().draw_mermaid_png()))

# NoRetrieval 전략 테스트
test_query = "안녕하세요!"

initial_state = {"query": test_query}
result = main_graph.invoke(initial_state)

print(f"쿼리: {test_query}")
print(f"전략: {result['retrieval_strategy']}")
print(f"최종 응답: {result['final_response']}")
```

**예상 결과**:
```
쿼리: 안녕하세요!
전략: no_retrieval
최종 응답: 안녕하세요! 무엇을 도와드릴까요? 레스토랑에 관한 질문이나 메뉴, 와인에 대해 궁금하신 점이 있으시면 언제든 말씀해 주세요!
```

### 단계 5: SingleShotRAG 노드 기본 구현

벡터 데이터베이스 검색 및 RAG 체인을 StateGraph에 통합합니다.

#### 5.1 벡터 데이터베이스 로드

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# 임베딩 모델
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Chroma 벡터 DB 로드
menu_db = Chroma(
    embedding_function=embeddings_model,
    collection_name="restaurant_menu",
    persist_directory="./chroma_db",
)

wine_db = Chroma(
    embedding_function=embeddings_model,
    collection_name="restaurant_wine",
    persist_directory="./chroma_db",
)

# Retriever 생성
menu_retriever = menu_db.as_retriever(search_kwargs={"k": 3})
wine_retriever = wine_db.as_retriever(search_kwargs={"k": 3})

# 검색 테스트
query = "스테이크와 어울리는 와인을 추천해주세요."
menu_results = menu_retriever.invoke(query)

print(f"검색된 메뉴 문서 수: {len(menu_results)}")
for i, doc in enumerate(menu_results, 1):
    print(f"\n[문서 {i}]")
    print(doc.page_content[:100] + "...")
```

#### 5.2 검색 도구 정의

```python
from langchain_core.tools import tool

@tool
def search_menu(query: str) -> list:
    """레스토랑에서 제공하는 메뉴 검색"""
    return menu_retriever.invoke(query)

@tool
def search_wine(query: str) -> list:
    """레스토랑에서 제공하는 와인 검색"""
    return wine_retriever.invoke(query)

# 도구 테스트
menu_docs = search_menu.invoke({"query": "스테이크"})
print(f"메뉴 검색 결과: {len(menu_docs)}개")
```

**핵심 포인트**:
- `@tool` 데코레이터: LangChain 도구로 등록
- Retriever를 도구로 래핑하여 재사용성 향상

#### 5.3 문서 포맷팅 함수

```python
from langchain_core.documents import Document

def format_docs(docs: list[Document]) -> str:
    """검색 문서를 RAG 프롬프트용 텍스트로 변환"""

    formatted_result = ""
    for doc in docs:
        formatted_result += f"{doc.page_content}\n"
        formatted_result += f"(출처: {doc.metadata['source']} - "
        formatted_result += f"[{doc.metadata['menu_number']}]{doc.metadata['menu_name']})\n"
        formatted_result += "-" * 80 + "\n"

    return formatted_result

# 테스트
if menu_results:
    formatted = format_docs(menu_results)
    print(formatted[:300] + "...")
```

**핵심 포인트**:
- 메타데이터 포함: 출처, 메뉴 번호, 이름
- RAG 프롬프트에 컨텍스트로 전달

#### 5.4 RAG 체인 생성

```python
from langchain_core.vectorstores import VectorStore
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from textwrap import dedent
from operator import itemgetter

def create_rag_chain(vectorstore: VectorStore, k: int = 3) -> RunnableParallel:
    """RAG 체인 생성 함수"""

    # Retriever 설정
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    # RAG 프롬프트 템플릿
    prompt = ChatPromptTemplate.from_messages([
        ("system", dedent("""
            주어진 컨텍스트를 사용하여 질문에 답변하세요.

            [가이드라인]
            1. 컨텍스트에 정보가 없거나 부족하면 '근거가 없습니다'라고 답변하세요.
            2. 답변을 할 때는 참조한 출처 또는 근거를 표시합니다.
               (출처: 파일경로 또는 URL - [메뉴번호]메뉴이름)
        """)),
        ("user", "[컨텍스트]\n{context}\n\n[질문]\n{question}\n\n[답변]\n"),
    ])

    # RAG 체인 정의
    chain = (
        RunnableParallel({
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "docs": retriever
        })
        | RunnableParallel({
            "answer": prompt | llm | StrOutputParser(),
            "docs": itemgetter("docs")
        })
    )

    return chain

def single_rag_response(query: str, vectorstore: VectorStore) -> dict:
    """단일 RAG 실행 함수"""

    # RAG 체인 생성 및 실행
    chain = create_rag_chain(vectorstore)
    response = chain.invoke(query)

    return response

# 테스트
query = "스테이크와 어울리는 와인을 추천해주세요."

print("Menu DB 검색:")
response = single_rag_response(query, menu_db)
print(f"답변: {response['answer']}")
print(f"검색된 문서 수: {len(response['docs'])}")
print("-" * 100)

print("\nWine DB 검색:")
response = single_rag_response(query, wine_db)
print(f"답변: {response['answer']}")
print(f"검색된 문서 수: {len(response['docs'])}")
```

**핵심 포인트**:
- `RunnableParallel`: 병렬 실행으로 검색과 질문 동시 처리
- `format_docs`: Retriever 결과를 문자열로 변환
- `itemgetter("docs")`: 원본 문서 리스트도 반환

## 🎯 실습 문제

### 문제 1: 다중 DB 검색 서브그래프 ⭐⭐⭐

**목표**: Menu DB와 Wine DB를 모두 검색하는 서브그래프를 구현하세요.

**요구사항**:
1. 두 DB를 병렬로 검색
2. 결과를 병합하여 반환
3. 각 문서에 출처 DB 정보 추가

**힌트**:
```python
class MultiDBSearchState(TypedDict):
    query: str
    menu_docs: list
    wine_docs: list
    all_docs: list

def search_menu_node(state: MultiDBSearchState):
    # menu_db 검색
    pass

def search_wine_node(state: MultiDBSearchState):
    # wine_db 검색
    pass

def merge_results(state: MultiDBSearchState):
    # 결과 병합
    pass
```

### 문제 2: 동적 k값 조정 라우팅 ⭐⭐⭐⭐

**목표**: 쿼리 복잡도에 따라 검색 문서 수(k)를 동적으로 조정하는 시스템을 구현하세요.

**요구사항**:
1. 쿼리 복잡도 분석 (simple, moderate, complex)
2. 복잡도에 따라 k값 결정 (2, 3, 5)
3. 상태에 k값 저장
4. SingleShotRAG 노드에서 k값 사용

**힌트**:
```python
class ComplexityRoute(BaseModel):
    complexity: Literal["simple", "moderate", "complex"]
    suggested_k: int
    reason: str

# 상태에 k 추가
class AdaptiveRagState(TypedDict):
    # ... 기존 필드
    search_k: int  # 추가
```

### 문제 3: 에러 핸들링 및 Fallback ⭐⭐⭐⭐⭐

**목표**: RAG 실행 중 에러 발생 시 대체 전략으로 폴백하는 시스템을 구현하세요.

**요구사항**:
1. try-except로 에러 캐치
2. 에러 발생 시 NoRetrieval로 폴백
3. 에러 정보를 상태에 기록
4. 에러 복구 과정을 시각화

**힌트**:
```python
class AdaptiveRagState(TypedDict):
    # ... 기존 필드
    errors: list           # 에러 로그
    fallback_used: bool    # 폴백 사용 여부

def single_lookup_with_fallback(state: AdaptiveRagState):
    try:
        # RAG 실행
        pass
    except Exception as e:
        # 에러 로깅 및 폴백
        return Command(
            goto="no_retrieval",
            update={"errors": state.get("errors", []) + [str(e)], "fallback_used": True}
        )
```

## ✅ 솔루션 예시

### 솔루션 1: 다중 DB 검색 서브그래프

```python
from typing import List
from langchain_core.documents import Document

class MultiDBSearchState(TypedDict):
    """다중 DB 검색 서브그래프 상태"""
    query: str
    menu_docs: list
    wine_docs: list
    all_docs: list

def create_multidb_search_graph():
    """다중 DB 검색 서브그래프 생성"""

    def search_menu_node(state: MultiDBSearchState):
        """Menu DB 검색 노드"""
        docs = menu_retriever.invoke(state["query"])

        # 출처 DB 정보 추가
        for doc in docs:
            doc.metadata['source_db'] = 'menu_db'

        return {"menu_docs": docs}

    def search_wine_node(state: MultiDBSearchState):
        """Wine DB 검색 노드"""
        docs = wine_retriever.invoke(state["query"])

        # 출처 DB 정보 추가
        for doc in docs:
            doc.metadata['source_db'] = 'wine_db'

        return {"wine_docs": docs}

    def merge_results(state: MultiDBSearchState):
        """검색 결과 병합"""
        all_docs = state.get("menu_docs", []) + state.get("wine_docs", [])

        # 관련성 점수로 정렬 (가정: 상위 5개만)
        all_docs = all_docs[:5]

        return Command(
            goto=END,
            update={"all_docs": all_docs}
        )

    # 서브그래프 구성
    workflow = StateGraph(MultiDBSearchState)

    workflow.add_node("search_menu", search_menu_node)
    workflow.add_node("search_wine", search_wine_node)
    workflow.add_node("merge", merge_results)

    # 병렬 검색을 위한 엣지
    workflow.add_edge(START, "search_menu")
    workflow.add_edge(START, "search_wine")
    workflow.add_edge("search_menu", "merge")
    workflow.add_edge("search_wine", "merge")

    return workflow.compile()

# 서브그래프 생성 및 테스트
multidb_search_graph = create_multidb_search_graph()

# 시각화
display(Image(multidb_search_graph.get_graph().draw_mermaid_png()))

# 테스트
test_query = "스테이크와 어울리는 와인"
result = multidb_search_graph.invoke({"query": test_query})

print(f"쿼리: {test_query}")
print(f"Menu DB 문서: {len(result.get('menu_docs', []))}개")
print(f"Wine DB 문서: {len(result.get('wine_docs', []))}개")
print(f"총 병합 문서: {len(result.get('all_docs', []))}개")

# 병합된 문서 출처 확인
for i, doc in enumerate(result['all_docs'], 1):
    source_db = doc.metadata.get('source_db', 'unknown')
    menu_name = doc.metadata.get('menu_name', 'N/A')
    print(f"{i}. [{source_db}] {menu_name}")
```

### 솔루션 2: 동적 k값 조정 라우팅

```python
from pydantic import Field

class ComplexityRoute(BaseModel):
    """쿼리 복잡도 분석 결과"""
    complexity: Literal["simple", "moderate", "complex"]
    suggested_k: int = Field(..., description="권장 검색 문서 수")
    reason: str

class AdaptiveRagStateV2(TypedDict):
    """동적 k값을 포함한 상태"""
    query: str
    retrieval_strategy: str
    search_k: int                    # 추가: 검색 문서 수
    retrieved_documents: list
    intermediate_responses: list
    final_response: str

def create_analysis_graph_with_k():
    """복잡도 분석 및 k값 결정 서브그래프"""

    class QueryAnalysisStateV2(TypedDict):
        query: str
        retrieval_strategy: str
        query_analysis: str
        search_k: int                # 추가

    def route_query_with_complexity(state: QueryAnalysisStateV2):
        """쿼리 분석 + 복잡도 평가"""

        router_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            당신은 레스토랑 서비스 담당자입니다.

            주어진 질문을 분석하여:
            1. 전략 선택: no_retrieval, single_lookup, iterative
            2. 복잡도 평가:
               - simple: 단일 개념, 명확한 답변 (k=2)
               - moderate: 2-3개 개념, 비교 필요 (k=3)
               - complex: 다중 개념, 추론 필요 (k=5)
            """),
            ("user", "{query}")
        ])

        # 결합 스키마
        class CombinedRoute(BaseModel):
            strategy: Literal["no_retrieval", "single_lookup", "iterative"]
            complexity: Literal["simple", "moderate", "complex"]
            suggested_k: int
            reason: str

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        chain = router_prompt | llm.with_structured_output(CombinedRoute)
        routing = chain.invoke({"query": state["query"]})

        return Command(
            goto=END,
            update={
                "retrieval_strategy": routing.strategy,
                "search_k": routing.suggested_k,
                "query_analysis": routing.reason
            }
        )

    # 서브그래프 구성
    workflow = StateGraph(QueryAnalysisStateV2)
    workflow.add_node("route", route_query_with_complexity)
    workflow.add_edge(START, "route")

    return workflow.compile()

# SingleShotRAG 노드에서 k값 사용
def single_lookup_with_dynamic_k(state: AdaptiveRagStateV2) -> Command[Literal["final"]]:
    """동적 k값을 사용하는 SingleShotRAG"""

    k = state.get("search_k", 3)  # 기본값 3

    # 동적 k로 RAG 체인 생성
    chain = create_rag_chain(menu_db, k=k)
    response = chain.invoke(state["query"])

    return Command(
        goto="final",
        update={
            "retrieved_documents": response['docs'],
            "final_response": response['answer'],
            "intermediate_responses": state.get("intermediate_responses", []) + [f"검색 문서 수: {k}개"]
        }
    )

# 메인 그래프 재구성
def create_main_graph_v2():
    """동적 k값 지원 메인 그래프"""

    workflow = StateGraph(AdaptiveRagStateV2)

    # 서브그래프 (k값 포함)
    analysis_graph = create_analysis_graph_with_k()

    # 노드 추가
    workflow.add_node("analysis", analysis_graph)
    workflow.add_node("no_retrieval", no_retrieval)
    workflow.add_node("single_lookup", single_lookup_with_dynamic_k)
    workflow.add_node("iterative", iterative)
    workflow.add_node("final", final_node)

    # 엣지
    workflow.add_edge(START, "analysis")
    workflow.add_conditional_edges(
        "analysis",
        lambda state: state["retrieval_strategy"],
        {
            "no_retrieval": "no_retrieval",
            "single_lookup": "single_lookup",
            "iterative": "iterative"
        }
    )
    workflow.add_edge("final", END)

    return workflow.compile()

# 테스트
main_graph_v2 = create_main_graph_v2()

test_queries = [
    "파스타 있나요?",                         # simple (k=2)
    "스테이크 메뉴 추천",                      # moderate (k=3)
    "채식 메뉴와 와인 페어링 추천",             # complex (k=5)
]

for query in test_queries:
    print(f"\n{'='*80}")
    print(f"쿼리: {query}")

    result = main_graph_v2.invoke({"query": query})

    print(f"전략: {result['retrieval_strategy']}")
    print(f"검색 문서 수 (k): {result.get('search_k', 'N/A')}")
    print(f"실제 검색된 문서: {len(result.get('retrieved_documents', []))}개")
```

### 솔루션 3: 에러 핸들링 및 Fallback

```python
import logging

# 로거 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AdaptiveRAG")

class AdaptiveRagStateV3(TypedDict):
    """에러 핸들링을 포함한 상태"""
    query: str
    retrieval_strategy: str
    search_k: int
    retrieved_documents: list
    intermediate_responses: list
    final_response: str
    errors: list               # 추가: 에러 로그
    fallback_used: bool        # 추가: 폴백 사용 여부
    original_strategy: str     # 추가: 원래 전략

def single_lookup_with_fallback(state: AdaptiveRagStateV3) -> Command:
    """에러 핸들링 및 폴백이 있는 SingleShotRAG"""

    try:
        logger.info(f"SingleShotRAG 실행 중: {state['query'][:50]}...")

        k = state.get("search_k", 3)

        # RAG 체인 생성 및 실행
        chain = create_rag_chain(menu_db, k=k)
        response = chain.invoke(state["query"])

        logger.info(f"SingleShotRAG 성공: {len(response['docs'])}개 문서 검색")

        return Command(
            goto="final",
            update={
                "retrieved_documents": response['docs'],
                "final_response": response['answer'],
                "intermediate_responses": state.get("intermediate_responses", []) + [f"SingleShotRAG 성공 (k={k})"]
            }
        )

    except Exception as e:
        # 에러 로깅
        error_msg = f"SingleShotRAG 실패: {str(e)}"
        logger.error(error_msg)

        # 폴백: NoRetrieval로 전환
        return Command(
            goto="no_retrieval",
            update={
                "errors": state.get("errors", []) + [error_msg],
                "fallback_used": True,
                "original_strategy": "single_lookup",
                "intermediate_responses": state.get("intermediate_responses", []) + ["SingleShotRAG 실패 → NoRetrieval 폴백"]
            }
        )

def no_retrieval_with_fallback_info(state: AdaptiveRagStateV3) -> Command[Literal["final"]]:
    """폴백 정보를 포함한 NoRetrieval"""

    response = no_retrieval_response(state["query"])

    # 폴백으로 실행된 경우 안내 메시지 추가
    if state.get("fallback_used", False):
        fallback_notice = f"\n\n[시스템 안내] 원래 전략({state.get('original_strategy')})에서 문제가 발생하여 직접 답변을 제공합니다."
        response = response + fallback_notice

        logger.warning(f"폴백 실행: {state.get('original_strategy')} → no_retrieval")

    return Command(
        goto="final",
        update={
            "final_response": response,
            "intermediate_responses": state.get("intermediate_responses", []) + [response]
        }
    )

# 메인 그래프 V3
def create_main_graph_v3():
    """에러 핸들링 포함 메인 그래프"""

    workflow = StateGraph(AdaptiveRagStateV3)

    # 서브그래프
    analysis_graph = create_analysis_graph_with_k()

    # 노드 추가
    workflow.add_node("analysis", analysis_graph)
    workflow.add_node("no_retrieval", no_retrieval_with_fallback_info)
    workflow.add_node("single_lookup", single_lookup_with_fallback)
    workflow.add_node("iterative", iterative)
    workflow.add_node("final", final_node)

    # 엣지
    workflow.add_edge(START, "analysis")
    workflow.add_conditional_edges(
        "analysis",
        lambda state: state["retrieval_strategy"],
        {
            "no_retrieval": "no_retrieval",
            "single_lookup": "single_lookup",
            "iterative": "iterative"
        }
    )
    workflow.add_edge("final", END)

    return workflow.compile()

# 에러 시뮬레이션 테스트
def test_error_handling():
    """에러 핸들링 테스트"""

    # 의도적으로 에러를 발생시키는 체인
    def create_failing_rag_chain(vectorstore, k=3):
        raise ValueError("시뮬레이션 에러: 벡터 DB 연결 실패")

    # 원래 함수 백업 및 교체
    original_create_rag_chain = create_rag_chain
    globals()['create_rag_chain'] = create_failing_rag_chain

    try:
        main_graph_v3 = create_main_graph_v3()

        test_query = "파스타 메뉴 추천해주세요"

        print(f"\n{'='*100}")
        print(f"에러 핸들링 테스트")
        print(f"{'='*100}")
        print(f"쿼리: {test_query}")

        result = main_graph_v3.invoke({"query": test_query})

        print(f"\n[결과]")
        print(f"  전략: {result['retrieval_strategy']}")
        print(f"  폴백 사용: {result.get('fallback_used', False)}")
        print(f"  원래 전략: {result.get('original_strategy', 'N/A')}")
        print(f"  에러 로그: {result.get('errors', [])}")
        print(f"  최종 응답: {result['final_response'][:150]}...")
        print(f"  실행 과정: {result['intermediate_responses']}")

    finally:
        # 원래 함수 복원
        globals()['create_rag_chain'] = original_create_rag_chain

# 테스트 실행
test_error_handling()
```

**실행 결과**:
```
에러 핸들링 테스트
쿼리: 파스타 메뉴 추천해주세요

[결과]
  전략: single_lookup
  폴백 사용: True
  원래 전략: single_lookup
  에러 로그: ['SingleShotRAG 실패: 시뮬레이션 에러: 벡터 DB 연결 실패']
  최종 응답: 파스타 메뉴를 추천드리겠습니다! 레스토랑에서는 다양한 종류의 파스타를 제공하고 있을 것입니다...

[시스템 안내] 원래 전략(single_lookup)에서 문제가 발생하여 직접 답변을 제공합니다.
  실행 과정: ['SingleShotRAG 실패 → NoRetrieval 폴백', '답변 내용...']
```

## 🚀 실무 활용 예시

### 활용 예시 1: 상태 추적 및 디버깅

LangGraph의 상태 추적 기능을 활용한 디버깅 시스템입니다.

```python
from langgraph.checkpoint.memory import MemorySaver

# 체크포인터 생성
checkpointer = MemorySaver()

# 체크포인터를 사용하는 그래프 컴파일
main_graph_with_checkpoint = create_main_graph_v3().compile(
    checkpointer=checkpointer
)

# 실행
config = {"configurable": {"thread_id": "test-123"}}
result = main_graph_with_checkpoint.invoke(
    {"query": "스테이크 추천"},
    config
)

# 상태 히스토리 조회
from langgraph.graph import get_state_history

for state_snapshot in main_graph_with_checkpoint.get_state_history(config):
    print(f"\n[Step {state_snapshot.step}]")
    print(f"  노드: {state_snapshot.next}")
    print(f"  쿼리: {state_snapshot.values.get('query', 'N/A')}")
    print(f"  전략: {state_snapshot.values.get('retrieval_strategy', 'N/A')}")
    print(f"  응답: {state_snapshot.values.get('final_response', 'N/A')[:50]}...")
```

### 활용 예시 2: 스트리밍 실행

실시간으로 그래프 실행 상태를 스트리밍합니다.

```python
# 스트리밍 실행
for event in main_graph.stream({"query": "와인 추천"}):
    for node_name, node_output in event.items():
        print(f"\n[노드: {node_name}]")
        if "final_response" in node_output:
            print(f"  최종 응답: {node_output['final_response'][:100]}...")
        if "retrieval_strategy" in node_output:
            print(f"  선택된 전략: {node_output['retrieval_strategy']}")
```

### 활용 예시 3: 병렬 실행 최적화

여러 쿼리를 병렬로 처리합니다.

```python
from concurrent.futures import ThreadPoolExecutor

queries = [
    "파스타 메뉴",
    "와인 추천",
    "스테이크 가격",
    "채식 메뉴"
]

def process_query(query):
    """단일 쿼리 처리"""
    result = main_graph.invoke({"query": query})
    return {
        "query": query,
        "strategy": result["retrieval_strategy"],
        "response": result["final_response"][:100]
    }

# 병렬 실행
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_query, queries))

# 결과 출력
for result in results:
    print(f"\n쿼리: {result['query']}")
    print(f"전략: {result['strategy']}")
    print(f"응답: {result['response']}...")
```

## 📖 참고 자료

### 공식 문서

- **LangGraph 문서**: [https://langchain-ai.github.io/langgraph/](https://langchain-ai.github.io/langgraph/)
- **LangGraph Tutorials**: [https://langchain-ai.github.io/langgraph/tutorials/](https://langchain-ai.github.io/langgraph/tutorials/)
- **StateGraph API**: [https://langchain-ai.github.io/langgraph/reference/graphs/](https://langchain-ai.github.io/langgraph/reference/graphs/)

### 추가 학습 자료

- **Adaptive RAG Part 1**: PRJ03_W3_001_LangGraph_AdaptiveRAG_Part1.md
- **LangGraph Subgraph Pattern**: Week 2 - LangGraph Subgraph 가이드
- **State Management**: LangGraph State 관리 패턴

### 관련 블로그 및 튜토리얼

- [LangGraph Blog - State Management](https://blog.langchain.dev/)
- [Building Complex Workflows with LangGraph](https://python.langchain.com/docs/langgraph)

### 추천 다음 단계

1. **Part 2-2 학습**: IterativeRAG 및 통합 실행
2. **체크포인팅**: 상태 저장 및 복구
3. **Human-in-the-Loop**: 사용자 개입 패턴
4. **프로덕션 배포**: FastAPI + LangGraph

---

**학습 완료 후 다음을 확인하세요**:
- [ ] StateGraph의 기본 구조를 이해했다
- [ ] 서브그래프 패턴을 활용할 수 있다
- [ ] Command 객체로 동적 라우팅을 수행할 수 있다
- [ ] 조건부 엣지를 구현할 수 있다
- [ ] 쿼리 분석 서브그래프를 만들 수 있다
- [ ] NoRetrieval 노드를 StateGraph에 통합했다
- [ ] SingleShotRAG 기본 구조를 이해했다

**다음 학습**: PRJ03_W3_002_LangGraph_AdaptiveRAG_Part2_Part2.md - IterativeRAG 및 전체 통합
