# LangGraph 활용 - Adaptive RAG (Part2-2) - 실습 및 통합 실행

## 📚 학습 목표

이 학습 가이드를 통해 다음을 달성할 수 있습니다:

1. **SingleShotRAG 노드**를 다중 DB 검색으로 확장하여 구현할 수 있다
2. **IterativeRAG 노드**를 LangGraph에 통합하여 반복 검색 및 쿼리 개선을 수행할 수 있다
3. **품질 평가 노드**를 구현하여 응답의 품질을 자동으로 측정할 수 있다
4. **품질 기반 재시도 로직**을 조건부 엣지로 구현할 수 있다
5. **전체 Adaptive RAG 시스템**을 하나의 그래프로 통합할 수 있다
6. **실전 프로젝트**로 법률 문서 기반 Adaptive RAG 시스템을 구축할 수 있다
7. **LangGraph 디버깅 및 시각화** 기법을 활용할 수 있다

## 🔑 핵심 개념

### 실습 중심 학습

Part 2-1에서는 기본 구조를 학습했다면, Part 2-2에서는:

1. **실제 RAG 구현**: 시뮬레이션 → 실제 벡터 DB 검색
2. **노드 통합**: 개별 노드 → 전체 그래프
3. **품질 관리**: 응답 품질 평가 및 자동 재시도
4. **실전 프로젝트**: 법률 문서 RAG 시스템 구축

### 고급 패턴

#### 1. **다중 DB 통합 검색**

여러 벡터 DB를 동시에 검색하여 결과를 통합:

```python
def multi_db_search(query: str, databases: dict) -> list:
    all_docs = []
    for db_name, db in databases.items():
        docs = db.as_retriever(search_kwargs={"k": 3}).invoke(query)
        for doc in docs:
            doc.metadata['source_db'] = db_name
        all_docs.extend(docs)
    return all_docs
```

#### 2. **품질 기반 조건부 라우팅**

응답 품질이 낮으면 자동으로 재시도:

```python
def route_after_final(state) -> Literal["analysis", END]:
    quality_score = state.get("quality_score", 1.0)
    retry_count = state.get("retry_count", 0)

    if quality_score < 0.7 and retry_count < max_retries:
        return "analysis"  # 재시도
    else:
        return END  # 종료
```

#### 3. **재시도 횟수 관리**

상태에서 재시도 횟수를 추적:

```python
class ExtendedState(TypedDict):
    # ... 기존 필드
    retry_count: int       # 재시도 횟수
    quality_score: float   # 품질 점수
```

## 💻 단계별 구현

### 단계 1: SingleShotRAG 노드 실제 구현

시뮬레이션 함수를 실제 RAG 기능으로 교체합니다.

#### 1.1 다중 DB 검색 및 통합

```python
from typing import Literal
from langgraph.types import Command
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def single_lookup(state: AdaptiveRagState) -> Command[Literal["final"]]:
    """SingleShotRAG 전략을 실행하는 노드 함수"""

    query = state["query"]

    # 두 데이터베이스 모두에서 검색
    menu_response = single_rag_response(query, menu_db)
    wine_response = single_rag_response(query, wine_db)

    # 검색된 문서 수집
    all_docs = []
    all_docs.extend(menu_response.get("docs", []))
    all_docs.extend(wine_response.get("docs", []))

    # 응답 수집
    responses = []
    if menu_response["answer"] and "근거가 없습니다" not in menu_response["answer"]:
        responses.append(f"[메뉴 DB] {menu_response['answer']}")
    if wine_response["answer"] and "근거가 없습니다" not in wine_response["answer"]:
        responses.append(f"[와인 DB] {wine_response['answer']}")

    # 최종 응답 생성
    if responses:
        # 두 DB의 결과를 통합
        combined_response = "\n\n".join(responses)

        # 통합 답변 생성
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            당신은 레스토랑 서비스 전문가입니다.
            메뉴 DB와 와인 DB의 검색 결과를 통합하여 고객에게 최적의 답변을 제공하세요.

            [가이드라인]
            1. 두 DB의 정보를 자연스럽게 통합
            2. 중복 정보는 제거하고 핵심만 전달
            3. 구체적이고 실용적인 조언 제공
            4. 친절하고 전문적인 톤 유지
            """),
            ("user", "[검색 결과]\n{results}\n\n[질문]\n{query}\n\n[통합 답변]")
        ])

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        chain = final_prompt | llm | StrOutputParser()

        final_response = chain.invoke({
            "results": combined_response,
            "query": query
        })
    else:
        final_response = "죄송하지만 관련 정보를 찾을 수 없습니다. 다른 질문을 해주시거나 직원에게 문의해주세요."

    return Command(
        goto="final",
        update={
            "retrieved_documents": all_docs,
            "final_response": final_response,
            "intermediate_responses": state.get("intermediate_responses", []) + [final_response]
        }
    )

print("SingleShotRAG 노드 함수 구현 완료!")
```

**핵심 포인트**:
- Part 1의 `single_rag_response` 함수 재사용
- 메뉴 DB와 와인 DB 병렬 검색
- 두 DB 결과를 하나의 통합 답변으로 생성
- "근거가 없습니다" 필터링으로 무의미한 응답 제거

#### 1.2 업데이트된 그래프 테스트

```python
# 메인 그래프 재생성 (업데이트된 single_lookup 반영)
main_graph = create_main_graph()

# 테스트
test_query = "시그니처 스테이크의 가격은 얼마인가요?"

initial_state = {"query": test_query}
result = main_graph.invoke(initial_state, debug=True)

print(f"\n쿼리: {test_query}")
print(f"전략: {result['retrieval_strategy']}")
print(f"최종 응답: {result['final_response']}")
print(f"검색된 문서 수: {len(result['retrieved_documents'])}")
```

**디버그 모드**:
- `debug=True`: 각 노드 실행 과정 출력
- 상태 변화 추적 가능

### 단계 2: IterativeRAG 노드 구현

반복적인 검색과 쿼리 개선을 StateGraph에 통합합니다.

#### 2.1 쿼리 개선 함수

```python
def refine_query(query: str) -> str:
    """쿼리 개선을 위한 체인"""

    query_improvement_prompt = ChatPromptTemplate.from_messages([
        ("system", "원래 쿼리를 분석하고 검색 쿼리를 개선하세요"),
        ("user", "[쿼리]{query}\n\n[개선된 쿼리]\n")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    query_improvement_chain = query_improvement_prompt | llm | StrOutputParser()

    improved_query = query_improvement_chain.invoke({"query": query})

    return improved_query

# 테스트
query = "스테이크와 어울리는 와인을 추천해주세요."
improved_query = refine_query(query)
print(f"원래 쿼리: {query}")
print(f"개선된 쿼리: {improved_query}")
```

**예상 결과**:
```
원래 쿼리: 스테이크와 어울리는 와인을 추천해주세요.
개선된 쿼리: 스테이크와 잘 어울리는 레드 와인 종류와 추천 브랜드, 페어링 팁을 알려주세요.
```

#### 2.2 반복적 RAG 함수

```python
from langchain_core.vectorstores import VectorStore

def iterative_rag_response(
    query: str,
    vectorstore: VectorStore,
    k: int = 3,
    max_iterations: int = 3
) -> tuple:
    """반복적인 RAG 기반 답변 생성"""

    # RAG 체인 생성
    rag_chain = create_rag_chain(vectorstore, k)

    intermediate_responses = []
    retrieved_docs = []

    for i in range(max_iterations):
        # RAG 실행
        response = rag_chain.invoke(query)
        intermediate_response = response["answer"]
        intermediate_docs = response["docs"]

        # 쿼리 개선
        query = refine_query(query)
        print(f"[반복 {i+1}] 쿼리 개선: {query}")

        # 근거가 없는 경우 다음 반복
        if len(intermediate_docs) == 0 or "근거가 없습니다" in intermediate_response:
            continue
        else:
            # 중간 결과 저장
            intermediate_responses.append(intermediate_response)
            for doc in intermediate_docs:
                if doc not in retrieved_docs:
                    retrieved_docs.append(doc)

    return intermediate_responses, retrieved_docs

# 테스트
query = "스테이크와 어울리는 와인을 추천해주세요."
responses, docs = iterative_rag_response(query, wine_db)

print("\n생성된 답변:")
for i, response in enumerate(responses, 1):
    print(f"\n[반복 {i}]")
    print(response[:200] + "...")

print(f"\n검색된 문서: {len(docs)}개")
```

#### 2.3 IterativeRAG 노드 구현

```python
def iterative(state: AdaptiveRagState) -> Command[Literal["final"]]:
    """IterativeRAG 전략을 실행하는 노드 함수"""

    query = state["query"]
    max_iterations = 3

    # 두 데이터베이스에서 반복 검색
    menu_responses, menu_docs = iterative_rag_response(query, menu_db, k=3, max_iterations=max_iterations)
    wine_responses, wine_docs = iterative_rag_response(query, wine_db, k=3, max_iterations=max_iterations)

    # 모든 문서 수집
    all_docs = []
    all_docs.extend(menu_docs)
    all_docs.extend(wine_docs)

    # 중간 응답 수집
    all_intermediate_responses = []

    if menu_responses:
        all_intermediate_responses.append("[메뉴 DB 검색 결과]")
        all_intermediate_responses.extend(menu_responses)

    if wine_responses:
        all_intermediate_responses.append("[와인 DB 검색 결과]")
        all_intermediate_responses.extend(wine_responses)

    # 최종 통합 답변 생성
    if all_intermediate_responses:
        combined_context = "\n\n".join(all_intermediate_responses)

        final_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            당신은 레스토랑 서비스 전문가입니다.
            여러 검색 결과를 종합하여 고객에게 최고의 답변을 제공하세요.

            [답변 작성 가이드라인]
            1. **통합성**: 메뉴와 와인 정보를 자연스럽게 연결
            2. **구체성**: 구체적인 추천과 설명 제공
            3. **실용성**: 고객이 바로 선택할 수 있는 정보 제공
            4. **완결성**: 추가 질문 없이도 이해 가능한 답변

            [답변 구조]
            - 핵심 추천을 먼저 제시
            - 옵션이 여러 개라면 나열
            - 페어링 제안이나 추가 팁 포함
            """),
            ("user", """
            [고객 질문]
            {query}

            [수집된 정보]
            {context}

            위 정보를 바탕으로 최적의 답변을 작성하세요.
            """)
        ])

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        chain = final_prompt | llm | StrOutputParser()

        final_response = chain.invoke({
            "query": query,
            "context": combined_context
        })
    else:
        final_response = "죄송하지만 관련 정보를 찾을 수 없습니다. 직원에게 문의해주시면 더 자세한 안내를 드리겠습니다."

    return Command(
        goto="final",
        update={
            "retrieved_documents": all_docs,
            "final_response": final_response,
            "intermediate_responses": state.get("intermediate_responses", []) + all_intermediate_responses + [final_response]
        }
    )

print("IterativeRAG 노드 함수 구현 완료!")
```

**핵심 포인트**:
- 각 반복마다 쿼리 개선
- 중간 응답을 모두 수집하여 최종 통합
- 메뉴 DB와 와인 DB 독립적으로 반복 검색

#### 2.4 그래프 업데이트 및 테스트

```python
# 메인 그래프 재생성
main_graph = create_main_graph()

# IterativeRAG 전략 테스트
test_query = "스테이크와 잘 어울리는 와인을 추천해주세요."

initial_state = {"query": test_query}
result = main_graph.invoke(initial_state, debug=True)

print(f"\n쿼리: {test_query}")
print(f"전략: {result['retrieval_strategy']}")
print(f"최종 응답:\n{result['final_response']}")
print(f"\n중간 응답 수: {len(result['intermediate_responses'])}")
print(f"검색된 문서 수: {len(result['retrieved_documents'])}")
```

### 단계 3: GenerateResponse 및 품질 평가

최종 응답 생성과 품질 평가를 통합합니다.

#### 3.1 최종 응답 생성 함수

```python
from typing import List

def generate_response(contexts: List[str], query: str) -> str:
    """수집된 컨텍스트를 통합하여 최종 답변 생성"""

    if contexts:
        context = "\n\n".join(contexts)
    else:
        context = ""

    prompt = ChatPromptTemplate.from_messages([
        ("system", "주어진 맥락을 바탕으로 명확하고 간단한 답변을 생성하세요:\n\n[맥락]\n{context}"),
        ("user", "{question}")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "context": context,
        "question": query
    })

# 테스트 (이전 iterative RAG 결과 사용)
if 'responses' in locals() and responses:
    response = generate_response(responses, query)
    print(f"맥락: {len(responses)}개 중간 응답")
    print(f"질문: {query}")
    print(f"답변: {response[:200]}...")
```

#### 3.2 응답 품질 평가

```python
from pydantic import BaseModel, Field

class ResponseQuality(BaseModel):
    """응답 품질 평가 메트릭"""
    relevance: float = Field(..., description="검색된 문서와 질문의 관련성 점수 (0-1)")
    completeness: float = Field(..., description="답변의 완성도 점수 (0-1)")
    consistency: float = Field(..., description="답변의 일관성 점수 (0-1)")
    overall_score: float = Field(..., description="종합 점수 (0-1)")
    explanation: str = Field(..., description="평가 결과에 대한 설명과 의견")

def evaluate_response(
    query: str,
    response: str,
    retrieved_docs: List[str],
) -> ResponseQuality:
    """응답 품질을 평가하는 함수"""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", "응답 품질을 평가하라:\n\n[질문]\n{query}\n\n[응답]\n{response}\n\n[검색 문서]\n{docs}"),
        ("user", "관련성, 완성도, 일관성을 0-1 사이 점수로 평가하시오.")
    ])

    chain = eval_prompt | llm.with_structured_output(ResponseQuality)

    scores = chain.invoke({
        "query": query,
        "response": response,
        "docs": "\n\n".join(retrieved_docs)
    })

    return scores

# 테스트
if 'response' in locals() and 'docs' in locals():
    quality = evaluate_response(
        query=query,
        response=response,
        retrieved_docs=[doc.page_content for doc in docs]
    )

    print(f"\n[품질 평가]")
    print(f"관련성: {quality.relevance:.2f}")
    print(f"완성도: {quality.completeness:.2f}")
    print(f"일관성: {quality.consistency:.2f}")
    print(f"종합점수: {quality.overall_score:.2f}")
    print(f"설명: {quality.explanation}")
```

#### 3.3 품질 기반 재시도 로직

```python
# 확장된 상태 정의
class ExtendedAdaptiveRagState(AdaptiveRagState):
    retry_count: int         # 재시도 횟수
    quality_score: float     # 품질 점수
    quality_feedback: str    # 품질 피드백

def final_node_with_evaluation(state: AdaptiveRagState) -> dict:
    """최종 응답을 생성하고 품질을 평가하는 노드"""

    query = state["query"]
    final_response = state.get("final_response", "")
    retrieved_docs = state.get("retrieved_documents", [])
    retry_count = state.get("retry_count", 0)

    # 최종 응답이 있으면 품질 평가 수행
    if final_response and final_response != "최종 응답이 없습니다.":
        # 문서가 Document 객체인 경우 문자열로 변환
        doc_contents = []
        for doc in retrieved_docs:
            if hasattr(doc, 'page_content'):
                doc_contents.append(doc.page_content)
            else:
                doc_contents.append(str(doc))

        # 품질 평가 수행
        if doc_contents:
            quality = evaluate_response(
                query=query,
                response=final_response,
                retrieved_docs=doc_contents
            )

            quality_score = quality.overall_score
            quality_feedback = quality.explanation

            print(f"\n[품질 평가]")
            print(f"종합 점수: {quality_score:.2f}")
            print(f"관련성: {quality.relevance:.2f} | 완성도: {quality.completeness:.2f} | 일관성: {quality.consistency:.2f}")
            print(f"피드백: {quality_feedback}")
        else:
            # 문서가 없는 경우 (no_retrieval 전략)
            quality_score = 0.8  # 기본 점수
            quality_feedback = "검색 없이 생성된 답변"
    else:
        quality_score = 0.0
        quality_feedback = "답변이 생성되지 않음"

    return {
        "final_response": final_response,
        "intermediate_responses": state.get("intermediate_responses", []) + ["최종 응답 생성 및 평가 완료"],
        "retry_count": retry_count,
        "quality_score": quality_score,
        "quality_feedback": quality_feedback
    }

def route_after_final(state: AdaptiveRagState) -> Literal["analysis", END]:
    """최종 노드 이후 라우팅: 품질 점수에 따라 재시도 또는 종료"""

    quality_score = state.get("quality_score", 1.0)
    retry_count = state.get("retry_count", 0)
    max_retries = 2
    quality_threshold = 0.7

    print(f"\n[라우팅 판단]")
    print(f"현재 품질 점수: {quality_score:.2f}")
    print(f"재시도 횟수: {retry_count}/{max_retries}")

    # 품질이 낮고 재시도 가능한 경우
    if quality_score < quality_threshold and retry_count < max_retries:
        print(f"→ 품질이 {quality_threshold} 미만이므로 재시도합니다.")
        return "analysis"
    else:
        if quality_score >= quality_threshold:
            print(f"→ 품질이 충분하므로 종료합니다.")
        else:
            print(f"→ 최대 재시도 횟수 도달로 종료합니다.")
        return END

print("GenerateResponse와 평가 체인 통합 완료!")
print("품질 점수 0.7 미만 시 자동 재시도 로직 구현 완료!")
```

**핵심 포인트**:
- `final_node_with_evaluation`: 응답 생성 + 품질 평가
- `route_after_final`: 품질 기반 조건부 라우팅
- 재시도 횟수 제한 (무한 루프 방지)

### 단계 4: 전체 그래프 통합

모든 노드를 하나의 완전한 Adaptive RAG 그래프로 통합합니다.

```python
from langgraph.graph import StateGraph, START, END

def create_main_graph_final():
    """최종 완성된 Adaptive RAG 메인 그래프"""

    # 메인 그래프 생성
    workflow = StateGraph(ExtendedAdaptiveRagState)

    # 서브그래프 (쿼리 분석)
    analysis_graph = create_analysis_graph()

    # 노드 추가
    workflow.add_node("analysis", analysis_graph)
    workflow.add_node("no_retrieval", no_retrieval)
    workflow.add_node("single_lookup", single_lookup)
    workflow.add_node("iterative", iterative)
    workflow.add_node("final", final_node_with_evaluation)

    # 엣지 추가
    workflow.add_edge(START, "analysis")

    # 조건부 엣지 1: 전략별 라우팅
    workflow.add_conditional_edges(
        "analysis",
        lambda state: state["retrieval_strategy"],
        {
            "no_retrieval": "no_retrieval",
            "single_lookup": "single_lookup",
            "iterative": "iterative"
        }
    )

    # 조건부 엣지 2: 품질 기반 재시도
    workflow.add_conditional_edges(
        "final",
        route_after_final,
        {
            "analysis": "analysis",
            END: END
        }
    )

    return workflow.compile()

# 최종 그래프 생성
final_graph = create_main_graph_final()

# 시각화
from IPython.display import Image, display
display(Image(final_graph.get_graph().draw_mermaid_png()))
```

#### 통합 테스트

```python
# 종합 테스트
test_queries = [
    "안녕하세요!",
    "시그니처 스테이크의 가격은 얼마인가요?",
    "와인과 스테이크 페어링 추천해주세요",
]

for query in test_queries:
    print(f"\n{'='*100}")
    print(f"쿼리: {query}")
    print(f"{'='*100}")

    initial_state = {
        "query": query,
        "retrieval_strategy": "",
        "retrieved_documents": [],
        "intermediate_responses": [],
        "final_response": "",
        "retry_count": 0,
        "quality_score": 0.0,
        "quality_feedback": ""
    }

    result = final_graph.invoke(initial_state)

    print(f"\n[결과]")
    print(f"  전략: {result['retrieval_strategy']}")
    print(f"  품질 점수: {result.get('quality_score', 0):.2f}")
    print(f"  재시도 횟수: {result.get('retry_count', 0)}")
    print(f"\n[최종 응답]")
    print(f"  {result['final_response']}")
```

## 🎯 실습 문제

### 문제 1: 스트리밍 응답 구현 ⭐⭐⭐

**목표**: 그래프 실행 과정을 실시간으로 스트리밍하여 사용자에게 진행 상황을 보여주세요.

**요구사항**:
1. `graph.stream()` 사용
2. 각 노드 실행 시 상태 출력
3. 최종 응답을 단계별로 표시

**힌트**:
```python
for event in final_graph.stream(initial_state):
    for node_name, node_output in event.items():
        print(f"\n[노드: {node_name}]")
        # 상태 출력
```

### 문제 2: 체크포인트 기반 상태 복구 ⭐⭐⭐⭐

**목표**: 그래프 실행 중 상태를 저장하고, 중단된 시점부터 재개할 수 있도록 구현하세요.

**요구사항**:
1. MemorySaver 또는 SqliteSaver 사용
2. 각 노드 실행 후 체크포인트 저장
3. thread_id로 세션 관리
4. 중단된 실행 재개 기능

**힌트**:
```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = create_main_graph_final().compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "session-123"}}
# 실행, 중단, 재개...
```

### 문제 3: 최종 실전 프로젝트 - 법률 문서 RAG ⭐⭐⭐⭐⭐

**목표**: 법률 문서(주택임대차법, 근로기준법, 개인정보보호법)를 기반으로 Adaptive RAG 시스템을 구축하세요.

**요구사항**:
1. PDF 문서 로드 및 청크 분할
2. 법률별 벡터 DB 구축
3. 법률 도메인 특화 프롬프트
4. 다중 법률 교차 검색
5. 법률 용어 기반 품질 평가

**데이터**:
- `data/housing_leasing_law.pdf`
- `data/labor_law.pdf`
- `data/personal_info_law.pdf`

## ✅ 솔루션 예시

### 솔루션 1: 스트리밍 응답 구현

```python
def streaming_rag_execution(query: str):
    """스트리밍 방식으로 RAG 실행"""

    initial_state = {
        "query": query,
        "retrieval_strategy": "",
        "retrieved_documents": [],
        "intermediate_responses": [],
        "final_response": "",
        "retry_count": 0,
        "quality_score": 0.0,
        "quality_feedback": ""
    }

    print(f"\n{'='*80}")
    print(f"쿼리: {query}")
    print(f"{'='*80}")
    print("\n실시간 실행 과정:")

    for event in final_graph.stream(initial_state):
        for node_name, node_output in event.items():
            print(f"\n[노드 실행: {node_name}]")

            # 전략 선택
            if "retrieval_strategy" in node_output and node_output["retrieval_strategy"]:
                print(f"  → 선택된 전략: {node_output['retrieval_strategy']}")

            # 중간 응답
            if "intermediate_responses" in node_output:
                responses = node_output["intermediate_responses"]
                if responses:
                    print(f"  → 중간 응답: {len(responses)}개")

            # 품질 점수
            if "quality_score" in node_output:
                score = node_output.get("quality_score", 0)
                if score > 0:
                    print(f"  → 품질 점수: {score:.2f}")

            # 최종 응답
            if "final_response" in node_output and node_output["final_response"]:
                final = node_output["final_response"]
                if final and final != "최종 응답이 없습니다.":
                    print(f"  → 최종 응답 생성 완료 ({len(final)}자)")

    print(f"\n{'='*80}")
    print("실행 완료!")
    print(f"{'='*80}")

# 테스트
streaming_rag_execution("스테이크와 어울리는 와인 추천해주세요")
```

### 솔루션 2: 체크포인트 기반 상태 복구

```python
from langgraph.checkpoint.memory import MemorySaver
import uuid

# 체크포인터 생성
checkpointer = MemorySaver()

# 체크포인터를 사용하는 그래프 컴파일
graph_with_checkpoint = create_main_graph_final().compile(
    checkpointer=checkpointer
)

# 세션 ID 생성
session_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": session_id}}

print(f"세션 ID: {session_id}")

# 초기 실행
initial_state = {
    "query": "와인 추천",
    "retrieval_strategy": "",
    "retrieved_documents": [],
    "intermediate_responses": [],
    "final_response": "",
    "retry_count": 0,
    "quality_score": 0.0,
    "quality_feedback": ""
}

print("\n[초기 실행]")
result = graph_with_checkpoint.invoke(initial_state, config)
print(f"전략: {result['retrieval_strategy']}")
print(f"최종 응답: {result['final_response'][:100]}...")

# 상태 히스토리 조회
print(f"\n[상태 히스토리]")
for i, state_snapshot in enumerate(graph_with_checkpoint.get_state_history(config)):
    print(f"\nStep {i}:")
    print(f"  다음 노드: {state_snapshot.next}")
    print(f"  전략: {state_snapshot.values.get('retrieval_strategy', 'N/A')}")

# 같은 세션으로 새로운 쿼리
print(f"\n[같은 세션에서 새 쿼리]")
new_state = {
    "query": "파스타 메뉴 추천",
    "retrieval_strategy": "",
    "retrieved_documents": [],
    "intermediate_responses": [],
    "final_response": "",
    "retry_count": 0,
    "quality_score": 0.0,
    "quality_feedback": ""
}

result2 = graph_with_checkpoint.invoke(new_state, config)
print(f"전략: {result2['retrieval_strategy']}")
print(f"최종 응답: {result2['final_response'][:100]}...")
```

### 솔루션 3: 법률 문서 RAG 시스템 (전체 구현)

```python
# ============================================================================
# [최종 실습] 법률 문서 기반 Adaptive RAG 시스템 구현
# ============================================================================

print("="*100)
print("법률 문서 기반 Adaptive RAG 시스템 구축 시작")
print("="*100)

# ----------------------------------------------------------------------------
# 1. 문서 로드 및 전처리
# ----------------------------------------------------------------------------

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

# 법률 문서 경로
law_documents = {
    "housing": "data/housing_leasing_law.pdf",
    "labor": "data/labor_law.pdf",
    "personal_info": "data/personal_info_law.pdf"
}

def load_law_documents(file_paths: dict) -> dict:
    """법률 문서를 로드하고 분할"""

    all_documents = {}
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    for law_type, file_path in file_paths.items():
        if Path(file_path).exists():
            print(f"\n[{law_type}] 문서 로드 중: {file_path}")
            loader = PyPDFLoader(file_path)
            documents = loader.load()

            # 문서 분할
            splits = text_splitter.split_documents(documents)

            # 메타데이터 추가
            for doc in splits:
                doc.metadata['law_type'] = law_type
                doc.metadata['source'] = file_path

            all_documents[law_type] = splits
            print(f"  → {len(documents)}개 페이지, {len(splits)}개 청크로 분할")
        else:
            print(f"\n[{law_type}] 경고: 파일이 존재하지 않음 - {file_path}")
            all_documents[law_type] = []

    return all_documents

# 문서 로드
law_docs = load_law_documents(law_documents)

# ----------------------------------------------------------------------------
# 2. 벡터 데이터베이스 구축
# ----------------------------------------------------------------------------

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

print(f"\n{'='*100}")
print("벡터 데이터베이스 구축")
print(f"{'='*100}")

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# 각 법률별로 별도의 컬렉션 생성
law_databases = {}

for law_type, documents in law_docs.items():
    if documents:
        print(f"\n[{law_type}] 벡터 DB 생성 중...")

        law_databases[law_type] = Chroma.from_documents(
            documents=documents,
            embedding=embeddings_model,
            collection_name=f"law_{law_type}",
            persist_directory=f"./chroma_db_law"
        )
        print(f"  → {len(documents)}개 문서 임베딩 완료")

# ----------------------------------------------------------------------------
# 3. 상태 정의
# ----------------------------------------------------------------------------

from typing import TypedDict, List

class LawRagState(TypedDict):
    query: str
    retrieval_strategy: str
    query_analysis: str
    retrieved_documents: List
    intermediate_responses: List
    final_response: str
    retry_count: int
    quality_score: float
    quality_feedback: str

# ----------------------------------------------------------------------------
# 4. 법률 특화 노드 정의
# ----------------------------------------------------------------------------

from pydantic import BaseModel
from typing import Literal

class LawQueryRoute(BaseModel):
    strategy: Literal["no_retrieval", "single_lookup", "iterative"]
    reason: str

def create_law_analysis_graph():
    """법률 쿼리 분석 서브그래프"""

    def route_law_query(state: LawRagState):
        router_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            당신은 법률 상담 전문가입니다.
            주택임대차법, 근로기준법, 개인정보보호법에 대한 법률 데이터베이스를 보유하고 있습니다.

            질문을 분석하여 다음 전략 중 하나를 선택하세요:

            1. no_retrieval:
               - 일반적인 인사, 간단한 상식 질문
               - 법률과 무관한 질문

            2. single_lookup:
               - 특정 법령 조항 확인
               - 단순 정의나 용어 설명
               - 명확한 단일 법률 질문

            3. iterative:
               - 복잡한 법률 해석이 필요한 경우
               - 여러 법률이 연관된 질문
               - 사례 분석이나 비교가 필요한 경우
            """),
            ("user", "{query}")
        ])

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        chain = router_prompt | llm.with_structured_output(LawQueryRoute)
        routing = chain.invoke({"query": state["query"]})

        return Command(
            goto=END,
            update={
                "retrieval_strategy": routing.strategy,
                "query_analysis": routing.reason
            }
        )

    workflow = StateGraph(LawRagState)
    workflow.add_node("route", route_law_query)
    workflow.add_edge(START, "route")

    return workflow.compile()

def law_single_lookup(state: LawRagState) -> Command[Literal["final"]]:
    """단일 검색 기반 법률 응답"""

    query = state["query"]

    # 모든 법률 DB에서 검색
    all_docs = []
    all_results = []

    for law_type, db in law_databases.items():
        retriever = db.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)

        if docs:
            all_docs.extend(docs)

            # 각 법률별 응답 생성
            context = "\n\n".join([doc.page_content for doc in docs])

            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""
                당신은 {law_type} 전문 법률가입니다.
                주어진 법률 조항을 바탕으로 정확하고 명확하게 답변하세요.

                [답변 가이드라인]
                1. 관련 법조항을 명시하세요
                2. 법률 용어를 쉽게 설명하세요
                3. 실무적 조언을 포함하세요
                4. 근거가 불충분하면 명시하세요
                """),
                ("user", "[법률 조항]\n{context}\n\n[질문]\n{question}\n\n[답변]")
            ])

            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
            chain = prompt | llm | StrOutputParser()

            answer = chain.invoke({"context": context, "question": query})

            if answer and "근거가" not in answer.lower():
                all_results.append(f"[{law_type.upper()}]\n{answer}")

    # 통합 답변 생성
    if all_results:
        combined = "\n\n".join(all_results)

        final_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            여러 법률 전문가의 의견을 종합하여 명확한 답변을 제공하세요.
            중복을 제거하고 핵심만 전달하세요.
            """),
            ("user", "[전문가 의견]\n{results}\n\n[질문]\n{query}\n\n[통합 답변]")
        ])

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        chain = final_prompt | llm | StrOutputParser()

        final_response = chain.invoke({"results": combined, "query": query})
    else:
        final_response = "죄송합니다. 관련 법률 조항을 찾을 수 없습니다. 전문 법률 상담을 받으시기 바랍니다."

    return Command(
        goto="final",
        update={
            "retrieved_documents": all_docs,
            "final_response": final_response,
            "intermediate_responses": state.get("intermediate_responses", []) + [final_response]
        }
    )

# (더 많은 노드 함수들... 간결성을 위해 생략)

# ----------------------------------------------------------------------------
# 5. 메인 그래프 구축 및 테스트
# ----------------------------------------------------------------------------

print(f"\n{'='*100}")
print("법률 상담 Adaptive RAG 시스템 구축 완료!")
print(f"{'='*100}")

# 테스트 질문
test_queries = [
    "전세 계약 시 주의사항은 무엇인가요?",
    "근로자의 연차휴가는 며칠인가요?",
    "임대인이 임대차 계약을 갱신 거절할 수 있는 경우는?",
]

# (테스트 실행 코드...)
```

## 📖 참고 자료

### 공식 문서

- **LangGraph Checkpointing**: [https://langchain-ai.github.io/langgraph/how-tos/persistence/](https://langchain-ai.github.io/langgraph/how-tos/persistence/)
- **Streaming**: [https://langchain-ai.github.io/langgraph/how-tos/streaming-tokens/](https://langchain-ai.github.io/langgraph/how-tos/streaming-tokens/)

### 추가 학습 자료

- **Adaptive RAG Part 1**: PRJ03_W3_001_LangGraph_AdaptiveRAG_Part1.md
- **Adaptive RAG Part 2-1**: PRJ03_W3_002_LangGraph_AdaptiveRAG_Part2_Part1.md

---

**학습 완료 후 다음을 확인하세요**:
- [ ] SingleShotRAG를 다중 DB로 확장했다
- [ ] IterativeRAG를 StateGraph에 통합했다
- [ ] 품질 평가 및 재시도를 구현했다
- [ ] 전체 Adaptive RAG 시스템을 완성했다
- [ ] 실전 법률 문서 RAG를 구축했다
- [ ] 스트리밍 및 체크포인트를 활용할 수 있다

**다음 학습**: 프로덕션 배포 및 성능 최적화
