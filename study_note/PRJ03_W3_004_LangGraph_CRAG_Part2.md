# LangGraph 활용 - Corrective RAG (CRAG) Part2 - LangGraph 워크플로우 구현

## 📚 학습 목표

이 학습 가이드를 완료하면 다음을 할 수 있습니다:

1. **LangGraph StateGraph**를 사용하여 CRAG 워크플로우를 설계하고 구현할 수 있습니다
2. **Map-Reduce 패턴**을 활용하여 병렬 문서 처리를 구현할 수 있습니다
3. **Send 객체**를 사용하여 동적으로 여러 노드를 실행할 수 있습니다
4. **조건부 엣지**를 활용하여 복잡한 분기 로직을 구현할 수 있습니다
5. **상태 관리**를 통해 워크플로우 전체에서 데이터를 추적하고 업데이트할 수 있습니다
6. **재시도 로직**을 구현하여 검색 실패 시 질문을 개선하고 웹 검색을 시도할 수 있습니다
7. 실제 프로젝트에 적용 가능한 **완전한 CRAG 시스템**을 구축할 수 있습니다

## 🔑 핵심 개념

### LangGraph StateGraph 개요

**LangGraph**는 LLM 애플리케이션을 위한 상태 기반 워크플로우 관리 프레임워크입니다. 복잡한 다단계 프로세스를 명확하게 표현하고 관리할 수 있습니다.

#### StateGraph의 핵심 요소

```
StateGraph 구조:
┌─────────────┐
│   START     │
└──────┬──────┘
       ↓
┌──────────────┐    [State]
│   Node 1     │ ←  • 데이터 저장
└──────┬───────┘    • 타입 안전
       ↓            • 업데이트 추적
┌──────────────┐
│ Conditional  │
│    Edge      │
└──┬────────┬──┘
   ↓        ↓
┌─────┐  ┌─────┐
│Node2│  │Node3│
└──┬──┘  └──┬──┘
   ↓        ↓
┌──────────────┐
│     END      │
└──────────────┘
```

### CRAG를 위한 LangGraph 워크플로우

```
[START]
   ↓
[retrieve] ─────────────────→ 문서 검색 (다단계 전략)
   ↓
[Send: distribute_grading]
   ├→ [grade_doc_1] ─┐
   ├→ [grade_doc_2] ─┼→ [collect_graded] ← Map-Reduce 패턴
   └→ [grade_doc_3] ─┘
           ↓
[Send: distribute_refining]
   ├→ [refine_doc_1] ─┐
   ├→ [refine_doc_2] ─┼→ [collect_refined]
   └→ [refine_doc_3] ─┘
           ↓
   [decide_to_generate] ← 조건부 분기
           ↓
    ┌──────┴──────┐
    ↓             ↓
[generate]  [transform_query]
    ↓             ↓
  [END]     [web_search] → (다시 평가로)
```

### Map-Reduce 패턴

**Map-Reduce**는 대량의 데이터를 병렬 처리하기 위한 프로그래밍 모델입니다.

#### Map 단계
- **목적**: 각 아이템을 독립적으로 처리
- **LangGraph**: `Send` 객체로 여러 노드 병렬 실행
- **CRAG 예시**: 각 문서를 개별적으로 평가

```python
def distribute_documents_for_grading(state):
    """Map: 각 문서를 개별 평가 노드로 분산"""
    return [
        Send("grade_single_document", {"document": doc, "question": q})
        for doc in state["retrieved_documents"]
    ]
```

#### Reduce 단계
- **목적**: 처리된 결과를 수집하고 통합
- **LangGraph**: `Annotated[list, operator.add]`로 자동 수집
- **CRAG 예시**: 평가된 문서들을 하나의 리스트로 통합

```python
class GraphState(TypedDict):
    graded_documents: Annotated[list, operator.add]  # Reducer
```

### Send 객체의 역할

**Send 객체**는 LangGraph에서 동적으로 여러 노드를 생성하고 실행하는 메커니즘입니다.

```python
# Send 사용 예시
Send(
    "target_node_name",  # 실행할 노드 이름
    {"key": "value"}     # 노드에 전달할 상태
)
```

**특징**:
- 런타임에 실행할 노드 수를 결정
- 각 Send는 독립적으로 실행 (병렬 처리)
- 모든 Send가 완료되면 다음 노드로 진행

## 💻 단계별 구현

### 단계 1: GraphState 정의

CRAG 워크플로우에 필요한 모든 상태를 TypedDict로 정의합니다.

```python
from typing import TypedDict, List, Tuple, Annotated
from langchain_core.documents import Document
import operator

class GraphState(TypedDict):
    """
    Corrective-RAG 그래프 상태
    """
    # 기본 정보
    question: str                                       # 사용자 질문
    generation: str                                     # 최종 생성된 답변
    num_generations: int                                # 답변 생성 횟수 (재시도 추적)

    # 문서 관련
    retrieved_documents: List[Tuple[Document, str]]     # 검색된 문서 + 평가 상태
    knowledge_strips: List[Tuple[Document, str]]        # 정제된 지식 조각

    # Map-Reduce를 위한 리듀서 필드
    graded_documents: Annotated[list, operator.add]     # 평가 결과 수집 (자동 병합)
    refined_knowledge: Annotated[list, operator.add]    # 정제 결과 수집 (자동 병합)
```

**주요 필드 설명**:
- `retrieved_documents`: (Document, grade) 튜플 리스트. grade는 "correct", "incorrect", "ambiguous"
- `knowledge_strips`: 최종적으로 답변 생성에 사용되는 정제된 지식
- `graded_documents`, `refined_knowledge`: Map-Reduce의 Reduce 단계에서 자동으로 수집
  - `Annotated[list, operator.add]`: 각 노드의 반환값이 자동으로 리스트에 추가됨

### 단계 2: Node 함수 구현

각 노드는 상태를 입력받아 처리하고 업데이트된 상태를 반환합니다.

#### 2.1 retrieve 노드 (문서 검색)

```python
def retrieve(state: GraphState) -> GraphState:
    """문서를 검색하는 함수"""
    logging.info("--- 문서 검색 ---")
    question = state["question"]

    # 다단계 검색 (Part1에서 구현한 AdaptiveVectorStore 사용)
    retrieved_documents = vector_db.multi_stage_search(question)

    # 초기 상태는 "ambiguous"로 설정 (아직 평가하지 않음)
    retrieved_documents = [(doc, "ambiguous") for doc in retrieved_documents]

    logging.info(f"검색 완료: {len(retrieved_documents)}개 문서")

    return {"retrieved_documents": retrieved_documents}
```

#### 2.2 web_search 노드 (웹 검색)

```python
def web_search(state: GraphState) -> GraphState:
    """웹 검색을 수행하는 함수"""
    logging.info("--- 웹 검색 ---")
    question = state["question"]

    # Tavily Search 사용
    search_results = search_tool.invoke(question)['results']

    # 검색 결과를 Document로 변환
    retrieved_documents = [
        (Document(page_content=str(result)), "ambiguous")
        for result in search_results
    ]

    logging.info(f"웹 검색 완료: {len(retrieved_documents)}개 결과")

    return {"retrieved_documents": retrieved_documents}
```

#### 2.3 Map 단계: distribute_documents_for_grading

```python
def distribute_documents_for_grading(state: GraphState):
    """문서 평가를 위해 각 문서를 개별 노드로 보내는 함수 (Map)"""
    retrieved_documents = state.get("retrieved_documents", [])
    question = state["question"]

    logging.info(f"--- 문서 평가 분배: {len(retrieved_documents)}개 ---")

    # Send 객체들의 리스트를 반환
    return [
        Send("grade_single_document", {
            "question": question,
            "document": doc,
            "grade": grade
        })
        for doc, grade in retrieved_documents
    ]
```

#### 2.4 개별 문서 평가: grade_single_document

```python
from typing import Dict

def grade_single_document(state: Dict) -> Dict:
    """개별 문서의 관련성을 평가하는 함수 (Send용)"""
    logging.info("--- 개별 문서 관련성 평가 ---")
    question = state["question"]
    document = state["document"]

    # Retrieval Grader 사용 (Part1에서 구현)
    score = retrieval_grader.invoke({
        "question": question,
        "document": document.page_content
    })
    grade = score.relevance_score.lower()

    if grade == "correct":
        logging.info("✅ 문서 관련성: 높음")
        return {"graded_documents": [(document, "correct")]}
    elif grade == "incorrect":
        logging.info("❌ 문서 관련성: 낮음")
        return {"graded_documents": [(document, "incorrect")]}
    else:
        logging.info("⚠️ 문서 관련성: 모호함")
        return {"graded_documents": [(document, "ambiguous")]}
```

#### 2.5 Reduce 단계: collect_graded_documents

```python
def collect_graded_documents(state: GraphState) -> GraphState:
    """평가된 문서들을 수집하는 함수 (Reduce 단계)"""
    logging.info("--- 문서 평가 결과 수집 ---")
    graded_documents = state.get("graded_documents", [])

    logging.info(f"수집된 graded_documents: {len(graded_documents)}개")

    # 중첩된 리스트를 평면화
    flattened_docs = []
    for item in graded_documents:
        if isinstance(item, list):
            # 리스트인 경우 각 요소 추가
            for sub_item in item:
                if sub_item:  # 빈 요소가 아닌 경우만
                    flattened_docs.append(sub_item)
        elif item:  # 단일 요소이고 비어있지 않은 경우
            flattened_docs.append(item)

    logging.info(f"총 {len(flattened_docs)}개 문서 평가 완료")

    # retrieved_documents 업데이트, graded_documents 초기화
    return {
        "retrieved_documents": flattened_docs,
        "graded_documents": []  # 리셋
    }
```

#### 2.6 Map 단계: distribute_documents_for_refining

```python
def distribute_documents_for_refining(state: GraphState):
    """지식 정제를 위해 각 문서를 개별 노드로 보내는 함수 (Map)"""
    graded_documents = state.get("graded_documents", [])
    question = state["question"]

    # graded_documents를 평면화
    flattened_docs = []
    for item in graded_documents:
        if isinstance(item, list):
            flattened_docs.extend(item)
        else:
            flattened_docs.append(item)

    logging.info(f"--- 지식 정제 분배: {len(flattened_docs)}개 ---")

    # Send 객체들의 리스트를 반환
    return [
        Send("refine_single_knowledge", {
            "question": question,
            "document": doc,
            "grade": grade
        })
        for doc, grade in flattened_docs
    ]
```

#### 2.7 개별 지식 정제: refine_single_knowledge

```python
def refine_single_knowledge(state: Dict) -> Dict:
    """개별 문서의 지식을 정제하는 함수 (Send용)"""
    logging.info("--- 개별 지식 정제 ---")
    question = state["question"]
    document = state["document"]
    grade = state.get("grade", "")

    # 관련성이 없는 문서는 제외
    if grade == "incorrect":
        logging.info("❌ 관련성 없음: 제외")
        return {"refined_knowledge": []}

    # Knowledge Refiner 사용 (Part1에서 구현)
    refined_knowledge = knowledge_refiner.invoke({
        "question": question,
        "document": document.page_content
    })

    knowledge = refined_knowledge.knowledge_strip
    binary_score = refined_knowledge.binary_score

    if binary_score == "yes":
        logging.info("✅ 정제된 지식: 추가")
        return {"refined_knowledge": [(Document(page_content=knowledge), "correct")]}
    else:
        logging.info("❌ 정제된 지식: 제외")
        return {"refined_knowledge": []}
```

#### 2.8 Reduce 단계: collect_refined_knowledge

```python
def collect_refined_knowledge(state: GraphState) -> GraphState:
    """정제된 지식들을 수집하는 함수 (Reduce 단계)"""
    logging.info("--- 정제된 지식 수집 ---")
    refined_knowledge = state.get("refined_knowledge", [])

    logging.info(f"수집된 refined_knowledge: {len(refined_knowledge)}개")

    # 중첩된 리스트를 평면화하고 빈 리스트 제거
    knowledge_strips = []
    for item in refined_knowledge:
        if isinstance(item, list):
            # 리스트인 경우 각 요소 확인
            for sub_item in item:
                if sub_item:  # 빈 요소가 아닌 경우만
                    knowledge_strips.append(sub_item)
        elif item:  # 단일 요소이고 비어있지 않은 경우
            knowledge_strips.append(item)

    logging.info(f"총 {len(knowledge_strips)}개 지식 정제 완료")

    return {
        "knowledge_strips": knowledge_strips,
        "refined_knowledge": []  # 리셋
    }
```

#### 2.9 generate 노드 (답변 생성)

```python
def generate(state: GraphState) -> GraphState:
    """답변을 생성하는 함수"""
    logging.info("--- 답변 생성 ---")
    num_generations = state.get("num_generations", 0)
    question = state["question"]
    knowledge_strips = state.get("knowledge_strips", [])

    # 지식이 없는 경우 처리
    if not knowledge_strips:
        generation = "죄송합니다. 질문에 대한 충분한 정보를 찾을 수 없습니다."
        logging.warning("⚠️ 지식 부족: 기본 답변 반환")
    else:
        # RAG를 이용한 답변 생성
        doc_texts = [doc for doc, _ in knowledge_strips]
        generation = generator_answer(question, docs=doc_texts)
        logging.info("✅ 답변 생성 완료")

    # 생성 횟수 업데이트
    num_generations += 1

    return {
        "generation": generation,
        "num_generations": num_generations
    }
```

#### 2.10 transform_query 노드 (질문 개선)

```python
def transform_query(state: GraphState) -> GraphState:
    """질문을 개선하는 함수"""
    logging.info("--- 질문 개선 ---")
    question = state["question"]

    # 질문 재작성 (Part1에서 구현)
    rewritten_question = rewrite_question(question)

    logging.info(f"원본 질문: {question}")
    logging.info(f"개선된 질문: {rewritten_question}")

    return {"question": rewritten_question}
```

### 단계 3: Edge 조건 함수 구현

조건부 엣지는 상태를 기반으로 다음에 실행할 노드를 결정합니다.

```python
def decide_to_generate(state: GraphState) -> str:
    """답변 생성 여부를 결정하는 함수"""
    logging.info("--- 평가된 문서 분석 ---")
    knowledge_strips = state.get("knowledge_strips", [])
    num_generations = state.get("num_generations", 0)

    # 생성 횟수가 3회 이상이면 종료
    if num_generations >= 3:
        logging.info("⚠️ 생성 횟수 초과: 강제 종료")
        return "generate"

    if not knowledge_strips:
        logging.info("🔄 지식 부족 → 질문 개선 및 웹 검색")
        return "transform_query"
    else:
        logging.info("✅ 충분한 지식 → 답변 생성")
        return "generate"
```

### 단계 4: 그래프 구축 및 연결

모든 노드와 엣지를 연결하여 완전한 그래프를 구성합니다.

```python
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display

# 워크플로우 그래프 초기화
builder = StateGraph(GraphState)

# ═══════════════════════════════════════════════════════════
# 노드 정의
# ═══════════════════════════════════════════════════════════
builder.add_node("retrieve", retrieve)                        # 문서 검색
builder.add_node("web_search", web_search)                    # 웹 검색

# Map-Reduce를 위한 노드들
builder.add_node("grade_single_document", grade_single_document)          # 개별 문서 평가 (Map)
builder.add_node("collect_graded_documents", collect_graded_documents)    # 문서 평가 결과 수집 (Reduce)
builder.add_node("refine_single_knowledge", refine_single_knowledge)      # 개별 지식 정제 (Map)
builder.add_node("collect_refined_knowledge", collect_refined_knowledge)  # 지식 정제 결과 수집 (Reduce)

# 기타 노드들
builder.add_node("generate", generate)                        # 답변 생성
builder.add_node("transform_query", transform_query)          # 질문 개선

# ═══════════════════════════════════════════════════════════
# 경로 정의
# ═══════════════════════════════════════════════════════════

# 시작점
builder.add_edge(START, "retrieve")

# ───────────────────────────────────────────────────────────
# 문서 평가를 위한 Map-Reduce 패턴
# ───────────────────────────────────────────────────────────
# retrieve → distribute (Map) → grade_single (개별 처리) → collect (Reduce)
builder.add_conditional_edges(
    "retrieve",
    distribute_documents_for_grading,  # Map: 각 문서를 개별 평가로 분산
    ["grade_single_document"]          # Send가 보낼 수 있는 노드 목록
)
builder.add_edge("grade_single_document", "collect_graded_documents")  # Reduce: 평가 결과 수집

# ───────────────────────────────────────────────────────────
# 지식 정제를 위한 Map-Reduce 패턴
# ───────────────────────────────────────────────────────────
# collect_graded → distribute (Map) → refine_single (개별 처리) → collect (Reduce)
builder.add_conditional_edges(
    "collect_graded_documents",
    distribute_documents_for_refining,  # Map: 각 문서를 개별 정제로 분산
    ["refine_single_knowledge"]         # Send가 보낼 수 있는 노드 목록
)
builder.add_edge("refine_single_knowledge", "collect_refined_knowledge")  # Reduce: 정제 결과 수집

# ───────────────────────────────────────────────────────────
# 조건부 엣지: 문서 평가 후 결정
# ───────────────────────────────────────────────────────────
# collect_refined → decide → generate OR transform_query
builder.add_conditional_edges(
    "collect_refined_knowledge",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)

# ───────────────────────────────────────────────────────────
# 웹 검색 후 다시 문서 평가로 진입
# ───────────────────────────────────────────────────────────
# web_search → distribute (Map) → grade_single → collect
builder.add_conditional_edges(
    "web_search",
    distribute_documents_for_grading,  # 웹 검색 후에도 Map-Reduce 패턴 적용
    ["grade_single_document"]
)

# ───────────────────────────────────────────────────────────
# 추가 경로
# ───────────────────────────────────────────────────────────
builder.add_edge("transform_query", "web_search")  # 질문 개선 → 웹 검색
builder.add_edge("generate", END)                   # 답변 생성 → 종료

# ═══════════════════════════════════════════════════════════
# 그래프 컴파일
# ═══════════════════════════════════════════════════════════
graph = builder.compile()

print("✅ CRAG 그래프 구축 완료")
```

### 단계 5: 그래프 시각화

```python
# 그래프 시각화
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
    print("✅ 그래프 시각화 완료")
except Exception as e:
    print(f"⚠️ 그래프 시각화 실패: {e}")
```

### 단계 6: 그래프 실행

```python
# 예시 1: 내부 DB에 정보가 있는 경우
print("\n" + "="*70)
print("테스트 1: 내부 DB 검색")
print("="*70)

inputs = {"question": "스테이크 메뉴의 가격은 얼마인가요?"}
final_state = graph.invoke(inputs)

print(f"\n질문: {inputs['question']}")
print(f"최종 답변: {final_state.get('generation', '답변 생성 실패')}")
print(f"생성 횟수: {final_state.get('num_generations', 0)}")

# 예시 2: 내부 DB에 정보가 부족하여 웹 검색이 필요한 경우
print("\n" + "="*70)
print("테스트 2: 웹 검색 필요")
print("="*70)

inputs = {"question": "스테이크에 어울리는 와인을 추천해주세요."}
final_state = graph.invoke(inputs)

print(f"\n질문: {inputs['question']}")
print(f"최종 답변: {final_state.get('generation', '답변 생성 실패')}")
print(f"생성 횟수: {final_state.get('num_generations', 0)}")
```

**실행 흐름 예시** (웹 검색이 필요한 경우):

```
--- 문서 검색 ---
🔍 1차 검색 모드: 0개의 문서 검색
🔍 2차 검색 모드: 0개의 문서 검색
🔍 3차 검색 모드: 0개의 문서 검색
--- 문서 평가 분배: 0개 ---
--- 문서 평가 결과 수집 ---
--- 지식 정제 분배: 0개 ---
--- 정제된 지식 수집 ---
--- 평가된 문서 분석 ---
🔄 지식 부족 → 질문 개선 및 웹 검색
--- 질문 개선 ---
원본 질문: 스테이크에 어울리는 와인을 추천해주세요.
개선된 질문: 스테이크 와인 페어링 추천 레드 화이트
--- 웹 검색 ---
웹 검색 완료: 5개 결과
--- 문서 평가 분배: 5개 ---
--- 개별 문서 관련성 평가 ---
✅ 문서 관련성: 높음
--- 개별 문서 관련성 평가 ---
✅ 문서 관련성: 높음
...
--- 문서 평가 결과 수집 ---
총 5개 문서 평가 완료
--- 지식 정제 분배: 5개 ---
--- 개별 지식 정제 ---
✅ 정제된 지식: 추가
...
--- 정제된 지식 수집 ---
총 3개 지식 정제 완료
--- 평가된 문서 분석 ---
✅ 충분한 지식 → 답변 생성
--- 답변 생성 ---
✅ 답변 생성 완료

질문: 스테이크에 어울리는 와인을 추천해주세요.
최종 답변: 스테이크와 잘 어울리는 와인으로는 카베르네 소비뇽, 시라/쉬라즈, 말벡 등의
          풀바디 레드 와인을 추천드립니다. 특히 카베르네 소비뇽은 마블링이 많은
          리브아이나 티본 스테이크와 환상적인 조화를 이룹니다...
생성 횟수: 1
```

## 🎯 실습 문제

### 문제: 법률 문서 기반 Corrective RAG 시스템 구현 (⭐⭐⭐)

**과제**: Part 1에서 배운 모든 개념을 활용하여 법률 문서를 기반으로 한 완전한 CRAG 시스템을 구현하세요.

**요구사항**:

1. **PDF 문서 로딩 및 벡터 저장소 생성**
   - 3개의 법률 PDF 파일 로딩: `housing_leasing_law.pdf`, `labor_law.pdf`, `personal_info_law.pdf`
   - RecursiveCharacterTextSplitter로 청크 분할 (chunk_size=800, overlap=100)
   - ChromaDB 벡터 저장소 생성

2. **법률 문서용 적응형 벡터 저장소**
   - AdaptiveVectorStore를 상속받아 LawVectorStore 클래스 구현
   - 법률 도메인에 맞게 검색 임계값 조정

3. **LangGraph StateGraph 구현**
   - GraphState 정의 (Part1의 GraphState 기반)
   - 모든 노드 함수 구현 (retrieve, web_search, grade, refine, generate, transform_query)
   - Map-Reduce 패턴을 활용한 병렬 처리
   - 조건부 엣지로 복잡한 분기 로직 구현

4. **테스트 및 검증**
   - 3개 이상의 법률 질문으로 시스템 테스트
   - 각 질문에 대한 답변, 사용된 지식 개수, 생성 횟수 출력

**데이터**:
- `data/housing_leasing_law.pdf` (주택임대차보호법)
- `data/labor_law.pdf` (근로기준법)
- `data/personal_info_law.pdf` (개인정보보호법)

**힌트**: 노트북의 마지막 셀에 전체 구현 코드가 제공되어 있습니다.

## ✅ 솔루션 예시

```python
# ============================================================
# [실습] 법률 문서 기반 Corrective RAG 시스템 구현
# ============================================================

# 1. PDF 문서 로딩 및 벡터 저장소 생성
# ------------------------------------------------------------
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

print("📚 법률 문서 로딩 중...")

# PDF 파일 경로
pdf_files = [
    "data/housing_leasing_law.pdf",
    "data/labor_law.pdf",
    "data/personal_info_law.pdf"
]

# 문서 로딩
all_documents = []
for pdf_path in pdf_files:
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    all_documents.extend(documents)
    print(f"✅ {pdf_path} 로딩 완료: {len(documents)}개 페이지")

# 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\\n\\n", "\\n", ".", " ", ""]
)
split_docs = text_splitter.split_documents(all_documents)
print(f"📄 총 {len(split_docs)}개의 청크로 분할 완료")

# 벡터 저장소 생성
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
law_vector_db = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    collection_name="law_documents",
    persist_directory="./chroma_law_db"
)
print("✅ 벡터 저장소 생성 완료\\n")


# 2. 법률 문서용 적응형 벡터 저장소
# ------------------------------------------------------------
class LawVectorStore:
    """법률 문서 검색을 위한 적응형 벡터 저장소"""

    def __init__(self, vector_db):
        self.vector_db = vector_db
        # 법률 문서는 더 엄격한 임계값 사용
        self.search_configs = {
            "initial": {"k": 3, "score_threshold": 0.4},
            "expanded": {"k": 5, "score_threshold": 0.2},
            "exhaustive": {"k": 8, "score_threshold": 0.0}
        }

    def multi_stage_search(self, query: str):
        """다단계 검색 전략"""
        # 1차: 정밀 검색
        initial = self.vector_db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs=self.search_configs["initial"]
        ).invoke(query)

        logger.info(f"🔍 1차 검색: {len(initial)}개 문서")

        if len(initial) < 1:
            # 2차: 확장 검색
            expanded = self.vector_db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs=self.search_configs["expanded"]
            ).invoke(query)

            logger.info(f"🔍 2차 검색: {len(expanded)}개 문서")

            if len(expanded) < 1:
                # 3차: 포괄 검색
                exhaustive = self.vector_db.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs=self.search_configs["exhaustive"]
                ).invoke(query)

                logger.info(f"🔍 3차 검색: {len(exhaustive)}개 문서")
                return exhaustive

            return expanded

        return initial

# 법률 문서용 벡터 저장소 초기화
law_retriever = LawVectorStore(law_vector_db)


# 3. State 정의
# ------------------------------------------------------------
class LawGraphState(TypedDict):
    """법률 문서 CRAG 그래프 상태"""
    question: str
    generation: str
    retrieved_documents: List[Tuple[Document, str]]
    knowledge_strips: List[Tuple[Document, str]]
    num_generations: int
    graded_documents: Annotated[list, operator.add]
    refined_knowledge: Annotated[list, operator.add]


# 4. Node 구성
# ------------------------------------------------------------

# (1) 문서 검색
def law_retrieve(state: LawGraphState) -> LawGraphState:
    """법률 문서 검색"""
    logging.info("--- 법률 문서 검색 ---")
    question = state["question"]

    retrieved_documents = law_retriever.multi_stage_search(question)
    retrieved_documents = [(doc, "ambiguous") for doc in retrieved_documents]

    logging.info(f"검색된 문서: {len(retrieved_documents)}개")
    return {"retrieved_documents": retrieved_documents}


# (2) 웹 검색
def law_web_search(state: LawGraphState) -> LawGraphState:
    """법률 관련 웹 검색"""
    logging.info("--- 법률 관련 웹 검색 ---")
    question = state["question"]

    # 법률 관련 검색어 강화
    enhanced_query = f"대한민국 법률 {question}"
    search_results = search_tool.invoke(enhanced_query)['results']

    retrieved_documents = [(Document(page_content=str(result)), "ambiguous")
                          for result in search_results[:3]]

    logging.info(f"웹 검색 결과: {len(retrieved_documents)}개")
    return {"retrieved_documents": retrieved_documents}


# (3) 문서 평가 분배
def law_distribute_grading(state: LawGraphState):
    """문서 평가를 위한 분배"""
    retrieved_documents = state.get("retrieved_documents", [])
    question = state["question"]

    return [
        Send("law_grade_single", {
            "question": question,
            "document": doc,
            "grade": grade
        })
        for doc, grade in retrieved_documents
    ]


# (4) 개별 문서 평가
def law_grade_single(state: Dict) -> Dict:
    """개별 법률 문서 평가"""
    logging.info("--- 개별 문서 평가 ---")
    question = state["question"]
    document = state["document"]

    score = retrieval_grader.invoke({
        "question": question,
        "document": document.page_content
    })
    grade = score.relevance_score.lower()

    if grade == "correct":
        logging.info("✅ 문서 관련성: 높음")
        return {"graded_documents": [(document, "correct")]}
    elif grade == "incorrect":
        logging.info("❌ 문서 관련성: 낮음")
        return {"graded_documents": [(document, "incorrect")]}
    else:
        logging.info("⚠️ 문서 관련성: 모호함")
        return {"graded_documents": [(document, "ambiguous")]}


# (5) 평가된 문서 수집
def law_collect_graded(state: LawGraphState) -> LawGraphState:
    """평가된 문서 수집"""
    logging.info("--- 평가 결과 수집 ---")
    graded_documents = state.get("graded_documents", [])

    # 리스트 평면화
    flattened_docs = []
    for item in graded_documents:
        if isinstance(item, list):
            flattened_docs.extend([i for i in item if i])
        elif item:
            flattened_docs.append(item)

    logging.info(f"수집 완료: {len(flattened_docs)}개 문서")
    return {"retrieved_documents": flattened_docs, "graded_documents": []}


# (6) 지식 정제 분배
def law_distribute_refining(state: LawGraphState):
    """지식 정제를 위한 분배"""
    graded_documents = state.get("graded_documents", [])
    question = state["question"]

    # 평면화
    flattened_docs = []
    for item in graded_documents:
        if isinstance(item, list):
            flattened_docs.extend(item)
        else:
            flattened_docs.append(item)

    return [
        Send("law_refine_single", {
            "question": question,
            "document": doc,
            "grade": grade
        })
        for doc, grade in flattened_docs
    ]


# (7) 개별 지식 정제
def law_refine_single(state: Dict) -> Dict:
    """개별 법률 지식 정제"""
    logging.info("--- 개별 지식 정제 ---")
    question = state["question"]
    document = state["document"]
    grade = state.get("grade", "")

    # 관련성 없는 문서 제외
    if grade == "incorrect":
        return {"refined_knowledge": []}

    refined = knowledge_refiner.invoke({
        "question": question,
        "document": document.page_content
    })

    knowledge = refined.knowledge_strip
    binary_score = refined.binary_score

    if binary_score == "yes":
        logging.info("✅ 지식 추가")
        return {"refined_knowledge": [(Document(page_content=knowledge), "correct")]}
    else:
        logging.info("❌ 지식 제외")
        return {"refined_knowledge": []}


# (8) 정제된 지식 수집
def law_collect_refined(state: LawGraphState) -> LawGraphState:
    """정제된 지식 수집"""
    logging.info("--- 정제된 지식 수집 ---")
    refined_knowledge = state.get("refined_knowledge", [])

    # 평면화 및 빈 항목 제거
    knowledge_strips = []
    for item in refined_knowledge:
        if isinstance(item, list):
            knowledge_strips.extend([i for i in item if i])
        elif item:
            knowledge_strips.append(item)

    logging.info(f"정제 완료: {len(knowledge_strips)}개 지식")
    return {"knowledge_strips": knowledge_strips, "refined_knowledge": []}


# (9) 답변 생성
def law_generate(state: LawGraphState) -> LawGraphState:
    """법률 답변 생성"""
    logging.info("--- 답변 생성 ---")
    num_generations = state.get("num_generations", 0)
    question = state["question"]
    knowledge_strips = state.get("knowledge_strips", [])

    if not knowledge_strips:
        generation = "죄송합니다. 해당 질문에 대한 법률 정보를 찾을 수 없습니다."
    else:
        doc_texts = [doc for doc, _ in knowledge_strips]
        generation = generator_answer(question, docs=doc_texts)

    num_generations += 1
    return {"generation": generation, "num_generations": num_generations}


# (10) 질문 변환
def law_transform_query(state: LawGraphState) -> LawGraphState:
    """법률 질문 재작성"""
    logging.info("--- 질문 재작성 ---")
    question = state["question"]

    # 법률 도메인에 최적화된 질문 재작성
    rewritten = rewrite_question(question)
    logging.info(f"원본: {question}")
    logging.info(f"재작성: {rewritten}")

    return {"question": rewritten}


# 5. Edge 정의
# ------------------------------------------------------------
def law_decide_to_generate(state: LawGraphState) -> str:
    """답변 생성 여부 결정"""
    logging.info("--- 생성 여부 결정 ---")
    knowledge_strips = state.get("knowledge_strips", [])
    num_generations = state.get("num_generations", 0)

    # 최대 생성 횟수 제한
    if num_generations >= 3:
        logging.info("⚠️ 최대 생성 횟수 도달")
        return "generate"

    if not knowledge_strips:
        logging.info("🔄 지식 부족 -> 질문 재작성")
        return "transform_query"
    else:
        logging.info("✅ 답변 생성 가능")
        return "generate"


# 6. 그래프 구축
# ------------------------------------------------------------
from langgraph.graph import StateGraph, START, END

print("\\n🔧 법률 CRAG 그래프 구축 중...")

# 그래프 초기화
law_builder = StateGraph(LawGraphState)

# 노드 추가
law_builder.add_node("law_retrieve", law_retrieve)
law_builder.add_node("law_web_search", law_web_search)
law_builder.add_node("law_grade_single", law_grade_single)
law_builder.add_node("law_collect_graded", law_collect_graded)
law_builder.add_node("law_refine_single", law_refine_single)
law_builder.add_node("law_collect_refined", law_collect_refined)
law_builder.add_node("law_generate", law_generate)
law_builder.add_node("law_transform_query", law_transform_query)

# 경로 정의
law_builder.add_edge(START, "law_retrieve")

# Map-Reduce: 문서 평가
law_builder.add_conditional_edges(
    "law_retrieve",
    law_distribute_grading,
    ["law_grade_single"]
)
law_builder.add_edge("law_grade_single", "law_collect_graded")

# Map-Reduce: 지식 정제
law_builder.add_conditional_edges(
    "law_collect_graded",
    law_distribute_refining,
    ["law_refine_single"]
)
law_builder.add_edge("law_refine_single", "law_collect_refined")

# 조건부 분기: 답변 생성 또는 질문 재작성
law_builder.add_conditional_edges(
    "law_collect_refined",
    law_decide_to_generate,
    {
        "transform_query": "law_transform_query",
        "generate": "law_generate"
    }
)

# 웹 검색 후 평가
law_builder.add_conditional_edges(
    "law_web_search",
    law_distribute_grading,
    ["law_grade_single"]
)

# 질문 재작성 후 웹 검색
law_builder.add_edge("law_transform_query", "law_web_search")

# 답변 생성 후 종료
law_builder.add_edge("law_generate", END)

# 그래프 컴파일
law_graph = law_builder.compile()

print("✅ 그래프 구축 완료\\n")

# 그래프 시각화
try:
    from IPython.display import Image, display
    display(Image(law_graph.get_graph().draw_mermaid_png()))
except Exception as e:
    print(f"⚠️ 그래프 시각화 실패: {e}")


# 7. 테스트 실행
# ------------------------------------------------------------
print("\\n" + "="*60)
print("🧪 법률 CRAG 시스템 테스트")
print("="*60 + "\\n")

# 테스트 질문
test_questions = [
    "전세 계약 시 임차인이 보호받을 수 있는 권리는 무엇인가요?",
    "근로자의 연차 휴가 사용 권리에 대해 설명해주세요.",
    "개인정보 수집 시 동의를 받아야 하는 경우는 언제인가요?"
]

for i, question in enumerate(test_questions, 1):
    print(f"\\n{'='*60}")
    print(f"📋 테스트 {i}: {question}")
    print('='*60)

    # 그래프 실행
    result = law_graph.invoke({"question": question})

    # 결과 출력
    print(f"\\n💡 최종 답변:\\n{result.get('generation', '답변 없음')}")
    print(f"\\n📊 생성 횟수: {result.get('num_generations', 0)}")
    print(f"📚 사용된 지식: {len(result.get('knowledge_strips', []))}개")
    print('-'*60)

print("\\n✅ 모든 테스트 완료!")
```

**실행 결과 예시**:

```
📚 법률 문서 로딩 중...
✅ data/housing_leasing_law.pdf 로딩 완료: 15개 페이지
✅ data/labor_law.pdf 로딩 완료: 23개 페이지
✅ data/personal_info_law.pdf 로딩 완료: 18개 페이지
📄 총 247개의 청크로 분할 완료
✅ 벡터 저장소 생성 완료

🔧 법률 CRAG 그래프 구축 중...
✅ 그래프 구축 완료

============================================================
🧪 법률 CRAG 시스템 테스트
============================================================

============================================================
📋 테스트 1: 전세 계약 시 임차인이 보호받을 수 있는 권리는 무엇인가요?
============================================================
--- 법률 문서 검색 ---
🔍 1차 검색: 3개 문서
--- 문서 평가 분배: 3개 ---
--- 개별 문서 평가 ---
✅ 문서 관련성: 높음
...
--- 평가 결과 수집 ---
수집 완료: 3개 문서
--- 지식 정제 분배: 3개 ---
--- 개별 지식 정제 ---
✅ 지식 추가
...
--- 정제된 지식 수집 ---
정제 완료: 2개 지식
--- 생성 여부 결정 ---
✅ 답변 생성 가능
--- 답변 생성 ---
✅ 답변 생성 완료

💡 최종 답변:
전세 계약 시 임차인은 주택임대차보호법에 따라 여러 권리를 보호받을 수 있습니다.
주요 권리로는:
1. 대항력: 주택 인도와 전입신고를 마치면 제3자에 대해 대항력을 갖습니다.
2. 우선변제권: 확정일자를 받으면 경매 시 다른 채권자보다 우선하여 보증금을 변제받을 수 있습니다.
3. 계약갱신요구권: 일정 조건 하에 계약 갱신을 요구할 수 있는 권리가 있습니다.

📊 생성 횟수: 1
📚 사용된 지식: 2개
------------------------------------------------------------

✅ 모든 테스트 완료!
```

## 📖 참고 자료

### 공식 문서
- [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph/)
- [LangGraph StateGraph API](https://langchain-ai.github.io/langgraph/reference/graphs/)
- [LangGraph Send 문서](https://langchain-ai.github.io/langgraph/how-tos/map-reduce/)
- [LangChain LCEL 문서](https://python.langchain.com/docs/expression_language/)

### 논문 및 기술 자료
- [CRAG 논문](https://arxiv.org/pdf/2401.15884) - Corrective Retrieval Augmented Generation
- [Map-Reduce Programming Model](https://research.google/pubs/pub62/) - Google Research

### 관련 튜토리얼
- [LangGraph Quick Start](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
- [Building Agentic RAG with LangGraph](https://blog.langchain.dev/agentic-rag-with-langgraph/)
- [Map-Reduce Pattern in LangGraph](https://langchain-ai.github.io/langgraph/how-tos/map-reduce/)

### 추가 학습 주제
- **고급 LangGraph 패턴**: 서브그래프, 체크포인트, 스트리밍
- **프로덕션 최적화**: 캐싱 전략, 병렬 처리, 에러 핸들링
- **멀티 에이전트 시스템**: 여러 에이전트가 협력하는 복잡한 워크플로우
- **RAG 성능 개선**: 하이브리드 검색, 재순위 지정, 응답 품질 평가

---

**이전 단계**: Part 1에서 CRAG의 기본 개념과 구성 요소를 학습했습니다.

**완료**: Part 2에서 LangGraph를 활용하여 완전한 CRAG 워크플로우를 구현했습니다. 이제 실제 프로젝트에 적용 가능한 고급 RAG 시스템을 구축할 수 있습니다.
