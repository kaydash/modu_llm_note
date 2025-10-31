# LangGraph Send 객체 - 병렬 문서 처리 시스템

## 📚 학습 목표

- **Send 객체**의 개념과 동작 원리를 이해한다
- **병렬 처리**를 통한 대용량 문서 처리 방법을 익힌다
- **문서 요약 시스템**을 StateGraph로 구현할 수 있다
- **청크 분할 → 병렬 요약 → 통합** 워크플로우를 설계할 수 있다
- **순서 유지**를 보장하며 병렬 처리 결과를 통합할 수 있다

## 🔑 핵심 개념

### Send 객체란?

**Send**는 LangGraph에서 하나의 노드가 여러 개의 병렬 작업을 생성할 수 있도록 하는 특별한 객체입니다.

**주요 특징:**
- **동적 병렬 실행**: 런타임에 병렬 작업 개수 결정
- **팬아웃-팬인 패턴**: 하나의 작업을 여러 개로 분산 후 다시 통합
- **효율성**: 독립적인 작업들을 동시에 처리하여 시간 단축
- **유연성**: 조건부 분기로 필요한 만큼만 병렬 작업 생성

### Send vs 일반 엣지

| 비교 항목 | 일반 엣지 | Send 객체 |
|----------|---------|----------|
| 실행 방식 | 순차적 | 병렬 |
| 작업 수 | 고정 (1개) | 동적 (N개) |
| 사용 사례 | 선형 워크플로우 | 대용량 데이터 처리 |
| 상태 전달 | 전체 상태 | 부분 상태 |

### 문서 요약 워크플로우

```
                    ┌─────────────────┐
                    │ 문서 로드        │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ 청크 분할        │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
    ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
    │ 청크1 요약   │   │ 청크2 요약   │   │ 청크3 요약   │
    └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
           │                 │                 │
           └─────────────────┼─────────────────┘
                             │
                    ┌────────▼────────┐
                    │ 최종 요약 통합   │
                    └─────────────────┘
```

### 관련 기술 스택

```python
# LangGraph 핵심
langgraph              # Graph 구성 및 Send 객체
langgraph.types        # Send 타입
langgraph.graph        # StateGraph, START, END

# LangChain
langchain-openai       # OpenAI LLM
langchain-core         # Document 객체

# Python
typing                 # 타입 힌팅
operator               # reduce 연산자
```

## 🛠 환경 설정

### 필요한 라이브러리 설치

```bash
pip install langgraph langchain-openai langchain-core
pip install python-dotenv
```

### API 키 설정

```.env
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

print("✅ 환경 설정 완료!")
```

## 💻 단계별 구현

### 1단계: 상태 정의

#### 1.1 타입 정의

```python
import operator
from typing import Annotated, List, Dict, Tuple, Any
from typing_extensions import TypedDict
from langchain_core.documents import Document

# 전체 워크플로우 상태
class SummarizationState(TypedDict):
    contents: List[Document]           # 초기 Document 객체 리스트
    chunks: List[Dict[str, Any]]       # 청크 리스트 (인덱스, 내용, 메타데이터)
    summaries: Annotated[List[Tuple[int, str]], operator.add]  # (인덱스, 요약) 튜플
    final_summary: str                 # 최종 통합 요약

# 개별 청크 처리 상태
class DocumentState(TypedDict):
    content: str      # 청크 내용
    index: int        # 청크 순서 (순서 유지용)
```

**주요 포인트:**
- `Annotated[List[Tuple[int, str]], operator.add]`: 병렬 작업 결과를 자동으로 리스트에 추가
- `operator.add`: reduce 연산자로 여러 노드의 결과를 하나의 리스트로 통합
- `index`: 병렬 처리 후에도 원래 순서를 유지하기 위한 인덱스

### 2단계: 노드 함수 구현

#### 2.1 문서 청크 분할 노드

```python
def split_documents(state: SummarizationState):
    """각 Document를 순서를 유지하며 청크로 분할"""
    chunks = []
    chunk_size = 1000  # 청크 크기 (문자 단위)
    global_chunk_index = 0

    # 각 Document를 순차적으로 처리
    for doc_index, document in enumerate(state["contents"]):
        content = document.page_content

        # 해당 문서를 청크로 분할
        for i in range(0, len(content), chunk_size):
            chunk_content = content[i:i + chunk_size]

            # 빈 청크는 스킵
            if chunk_content.strip():
                chunks.append({
                    "index": global_chunk_index,
                    "content": chunk_content,
                    "source_document": doc_index,
                    "source_metadata": document.metadata
                })
                global_chunk_index += 1

    return {"chunks": chunks}
```

**주요 파라미터:**
- `chunk_size`: 각 청크의 최대 크기 (문자 수)
- `global_chunk_index`: 전체 청크의 순서를 추적
- `source_document`: 원본 문서 인덱스 (추적용)

#### 2.2 개별 청크 요약 노드

```python
from langchain_openai import ChatOpenAI

# LLM 모델 초기화
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def summarize_document(state: DocumentState):
    """개별 문서 청크를 요약"""
    prompt = f"""다음 텍스트를 2-3문장으로 간결하게 요약해주세요:

    {state['content']}
    """

    try:
        response = model.invoke(prompt)
        summary = response.content
    except Exception as e:
        summary = f"요약 생성 중 오류 발생: {str(e)}"

    # 순서 정보와 함께 요약 반환
    return {"summaries": [(state["index"], summary)]}
```

**요약 전략:**
- **간결성**: 2-3문장으로 제한하여 핵심만 추출
- **에러 처리**: 예외 발생 시에도 정상적으로 처리
- **순서 유지**: (index, summary) 튜플로 반환

#### 2.3 조건부 엣지 함수 (Send 생성)

```python
from langgraph.types import Send

def continue_to_summarization(state: SummarizationState):
    """각 청크를 병렬로 요약하도록 Send 작업 생성"""
    return [
        Send("summarize_document", {
            "content": chunk["content"],
            "index": chunk["index"]
        })
        for chunk in state["chunks"]
    ]
```

**Send 객체 작동 원리:**
1. 청크 개수만큼 `Send` 객체 생성
2. 각 `Send`는 `summarize_document` 노드를 호출
3. 모든 `Send` 작업이 병렬로 실행됨
4. 결과는 `summaries` 리스트에 자동으로 추가됨 (operator.add 덕분)

#### 2.4 최종 요약 통합 노드

```python
def create_final_summary(state: SummarizationState):
    """순서를 유지하며 최종 요약 생성"""
    # 인덱스별로 요약을 정렬
    sorted_summaries = sorted(state["summaries"], key=lambda x: x[0])

    # 순서대로 요약들을 결합
    ordered_summaries = [summary for _, summary in sorted_summaries]
    combined_summaries = "\n\n".join(ordered_summaries)

    prompt = f"""다음은 문서를 청크별로 요약한 내용들입니다.
    이들을 종합하여 하나의 포괄적이고 일관성 있는 최종 요약을 작성해주세요.
    원본 문서의 순서와 흐름을 유지하면서 핵심 내용을 간결하게 정리해주세요:

    {combined_summaries}

    최종 요약:
    """

    try:
        response = model.invoke(prompt)
        final_summary = response.content
    except Exception as e:
        final_summary = f"최종 요약 생성 중 오류 발생: {str(e)}"

    return {"final_summary": final_summary}
```

**통합 전략:**
1. **정렬**: 인덱스 순서대로 요약 정렬
2. **결합**: 순서대로 요약을 하나의 텍스트로 결합
3. **재요약**: LLM을 사용하여 일관성 있는 최종 요약 생성

### 3단계: 그래프 구성

#### 3.1 StateGraph 생성

```python
from langgraph.graph import END, START, StateGraph

# 그래프 구성
builder = StateGraph(SummarizationState)

# 노드 추가
builder.add_node("split_documents", split_documents)
builder.add_node("summarize_document", summarize_document)
builder.add_node("create_final_summary", create_final_summary)
```

#### 3.2 엣지 연결

```python
# 엣지 연결
builder.add_edge(START, "split_documents")

# 조건부 엣지 (Send 객체 생성)
builder.add_conditional_edges(
    "split_documents",                    # 출발 노드
    continue_to_summarization,            # Send 생성 함수
    ["summarize_document"]                # 목적지 노드 (병렬 실행됨)
)

builder.add_edge("summarize_document", "create_final_summary")
builder.add_edge("create_final_summary", END)

# 그래프 컴파일
graph = builder.compile()
```

**엣지 유형:**
- `add_edge`: 일반 엣지 (순차 실행)
- `add_conditional_edges`: 조건부 엣지 (Send 객체로 병렬 실행)

#### 3.3 그래프 시각화

```python
from IPython.display import Image, display

# 그래프 시각화
display(Image(graph.get_graph().draw_mermaid_png()))
```

**그래프 구조:**
```
START → split_documents → [summarize_document (병렬)] → create_final_summary → END
```

### 4단계: 실행 및 테스트

#### 4.1 테스트 데이터 준비

```python
from langchain_core.documents import Document

# 긴 텍스트 생성 (여러 청크로 분할될 수 있도록)
test_text = """
LangGraph는 LangChain의 고급 워크플로우 관리 도구입니다.
이 도구를 사용하면 복잡한 다단계 작업을 효율적으로 구성하고 실행할 수 있습니다.
특히 Send 객체를 활용하면 병렬 처리가 가능하여 대용량 문서 처리에 유용합니다.

LangGraph의 주요 기능 중 하나는 StateGraph입니다.
StateGraph를 통해 상태 기반의 워크플로우를 정의할 수 있으며,
각 노드는 특정 작업을 수행하고 상태를 업데이트합니다.
노드 간의 연결은 엣지로 표현되며, 조건부 엣지를 통해 동적인 워크플로우 구성이 가능합니다.

Send 객체는 특히 강력한 기능입니다.
이를 통해 하나의 노드에서 여러 개의 병렬 작업을 생성할 수 있습니다.
예를 들어, 긴 문서를 여러 청크로 나눈 후 각 청크를 병렬로 처리할 수 있습니다.
이는 처리 시간을 크게 단축시키고 효율성을 높입니다.

문서 요약 작업의 경우, Send를 활용한 병렬 처리가 매우 효과적입니다.
먼저 문서를 적절한 크기의 청크로 분할한 후,
각 청크를 독립적으로 요약하고,
마지막으로 모든 요약을 통합하여 최종 요약을 생성합니다.

이러한 접근 방식은 대용량 문서를 처리할 때 특히 유용합니다.
각 청크가 독립적으로 처리되므로 병렬성이 높고,
결과적으로 전체 처리 시간이 크게 감소합니다.
또한 메모리 사용도 효율적으로 관리할 수 있습니다.

LangGraph는 다양한 체크포인트 기능도 제공합니다.
MemorySaver를 통해 세션 간 상태를 유지할 수 있으며,
InMemoryStore를 활용하면 장기 메모리 구현도 가능합니다.
이러한 기능들을 조합하면 복잡한 대화형 AI 시스템을 구축할 수 있습니다.

실제 프로덕션 환경에서는 다양한 고려사항이 있습니다.
에러 핸들링, 재시도 로직, 타임아웃 관리 등이 중요합니다.
LangGraph는 이러한 요구사항을 충족할 수 있는 유연성을 제공합니다.
개발자는 자신의 요구에 맞게 워크플로우를 커스터마이징할 수 있습니다.

결론적으로 LangGraph는 현대적인 LLM 애플리케이션 개발에 필수적인 도구입니다.
복잡한 워크플로우를 간단하게 관리하고,
병렬 처리를 통해 성능을 최적화하며,
상태 관리를 통해 안정적인 시스템을 구축할 수 있습니다.
"""

# 여러 페이지를 시뮬레이션하기 위해 3개의 Document 생성
documents = [
    Document(
        page_content=test_text,
        metadata={"page": 1, "source": "test_document_1"}
    ),
    Document(
        page_content=test_text + "\n\n추가 내용: 이 문서는 테스트를 위한 두 번째 페이지입니다.",
        metadata={"page": 2, "source": "test_document_2"}
    ),
    Document(
        page_content="마지막 페이지의 내용입니다. LangGraph를 활용하면 효율적인 문서 처리 시스템을 구축할 수 있습니다.",
        metadata={"page": 3, "source": "test_document_3"}
    )
]

print(f"로드된 페이지 수: {len(documents)}")
print(f"첫 번째 문서 길이: {len(documents[0].page_content)} 문자")
```

**출력:**
```
로드된 페이지 수: 3
첫 번째 문서 길이: 1096 문자
```

#### 4.2 그래프 실행

```python
# 초기 상태 설정
initial_state = {
    "contents": documents,
}

# 그래프 스트리밍 실행
for step in graph.stream(initial_state, stream_mode="values"):
    if "chunks" in step:
        print(f"처리 중인 청크 수: {len(step['chunks'])}")
    if "summaries" in step:
        print(f"현재까지 생성된 요약 수: {len(step['summaries'])}")
    if "final_summary" in step:
        print("\n" + "="*80)
        print("최종 요약:")
        print("="*80)
        print(step["final_summary"])
    print("-"*80)
```

**실행 흐름:**
```
처리 중인 청크 수: 5
--------------------------------------------------------------------------------
현재까지 생성된 요약 수: 5
--------------------------------------------------------------------------------
================================================================================
최종 요약:
================================================================================
LangGraph는 LangChain의 고급 워크플로우 관리 도구로, 복잡한 다단계 작업을 효율적으로
구성하고 실행할 수 있도록 지원합니다. 이 도구는 Send 객체를 활용한 병렬 처리 기능을
통해 대용량 문서 처리를 용이하게 하며, StateGraph를 통해 상태 기반의 동적인 워크플로우를
정의할 수 있습니다. 또한, 다양한 체크포인트 기능과 유연한 커스터마이징 옵션을 제공하여
복잡한 대화형 AI 시스템 구축에 적합합니다. LangGraph는 복잡한 워크플로우를 간단하게
관리하고 성능을 최적화하며 안정적인 시스템을 구축하는 데 필수적인 도구입니다.
--------------------------------------------------------------------------------
```

#### 4.3 최종 상태 확인

```python
# 최종 상태 출력
final_state = step  # 마지막 step이 최종 상태

print("\n최종 상태 요약:")
print(f"- 전체 문서 수: {len(final_state.get('contents', []))}")
print(f"- 전체 청크 수: {len(final_state.get('chunks', []))}")
print(f"- 전체 요약 수: {len(final_state.get('summaries', []))}")
print(f"- 최종 요약 길이: {len(final_state.get('final_summary', ''))} 문자")
```

**출력:**
```
최종 상태 요약:
- 전체 문서 수: 3
- 전체 청크 수: 5
- 전체 요약 수: 5
- 최종 요약 길이: 458 문자
```

## 🎯 실습 문제

### 실습 1: 청크 크기 최적화 (⭐⭐)

**문제:**
청크 크기를 조정하여 요약 품질을 개선하세요.

**요구사항:**
1. 500자, 1000자, 2000자 세 가지 청크 크기로 테스트
2. 각 크기별로 생성되는 청크 수와 요약 시간 비교
3. 최적의 청크 크기 결정

**힌트:**
- `chunk_size` 파라미터 조정
- `time` 모듈로 실행 시간 측정
- 요약 품질은 최종 요약의 일관성으로 평가

### 실습 2: PDF 파일 처리 (⭐⭐⭐)

**문제:**
실제 PDF 파일을 로드하여 요약 시스템으로 처리하세요.

**요구사항:**
1. PyPDFLoader를 사용하여 PDF 파일 로드
2. 페이지별로 메타데이터 유지
3. 최종 요약에 페이지 출처 포함

**힌트:**
```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("document.pdf")
documents = loader.load()
```

### 실습 3: 다단계 요약 시스템 (⭐⭐⭐⭐)

**문제:**
2단계 요약 시스템을 구현하세요.

**요구사항:**
1. 1단계: 청크별 상세 요약 (3-5문장)
2. 2단계: 상세 요약들을 다시 청크로 나눠 중간 요약 생성
3. 3단계: 중간 요약들을 통합하여 최종 요약 생성

**힌트:**
- 새로운 노드 `create_intermediate_summary` 추가
- Send 객체를 두 번 사용 (1단계와 2단계)
- 순서 유지를 위한 인덱스 관리

### 실습 4: 에러 복구 메커니즘 (⭐⭐⭐⭐⭐)

**문제:**
일부 청크 요약이 실패해도 전체 프로세스가 계속되도록 개선하세요.

**요구사항:**
1. 청크 요약 실패 시 재시도 로직 (최대 3회)
2. 재시도 후에도 실패하면 해당 청크 스킵
3. 최종 요약에 실패한 청크 정보 포함
4. 실패 로그 기록

**힌트:**
- `try-except`로 에러 캡처
- 재시도 카운터 추가
- 실패 정보를 상태에 저장

## ✅ 솔루션 예시

### 실습 1 솔루션: 청크 크기 최적화

```python
import time

def test_chunk_sizes(documents, chunk_sizes):
    """다양한 청크 크기로 테스트"""
    results = {}

    for size in chunk_sizes:
        print(f"\n{'='*80}")
        print(f"청크 크기: {size}자 테스트 중...")
        print('='*80)

        # 청크 크기를 파라미터로 받는 수정된 split_documents 함수
        def split_documents_with_size(state: SummarizationState):
            chunks = []
            global_chunk_index = 0

            for doc_index, document in enumerate(state["contents"]):
                content = document.page_content

                for i in range(0, len(content), size):  # 청크 크기 동적 설정
                    chunk_content = content[i:i + size]

                    if chunk_content.strip():
                        chunks.append({
                            "index": global_chunk_index,
                            "content": chunk_content,
                            "source_document": doc_index,
                            "source_metadata": document.metadata
                        })
                        global_chunk_index += 1

            return {"chunks": chunks}

        # 그래프 재구성
        builder = StateGraph(SummarizationState)
        builder.add_node("split_documents", split_documents_with_size)
        builder.add_node("summarize_document", summarize_document)
        builder.add_node("create_final_summary", create_final_summary)

        builder.add_edge(START, "split_documents")
        builder.add_conditional_edges(
            "split_documents",
            continue_to_summarization,
            ["summarize_document"]
        )
        builder.add_edge("summarize_document", "create_final_summary")
        builder.add_edge("create_final_summary", END)

        graph = builder.compile()

        # 실행 시간 측정
        start_time = time.time()

        final_state = None
        for step in graph.stream({"contents": documents}, stream_mode="values"):
            final_state = step

        elapsed_time = time.time() - start_time

        # 결과 저장
        results[size] = {
            "chunk_count": len(final_state.get("chunks", [])),
            "summary_count": len(final_state.get("summaries", [])),
            "final_summary": final_state.get("final_summary", ""),
            "elapsed_time": elapsed_time
        }

        print(f"청크 수: {results[size]['chunk_count']}")
        print(f"실행 시간: {elapsed_time:.2f}초")
        print(f"최종 요약 길이: {len(results[size]['final_summary'])}자")

    return results

# 테스트 실행
chunk_sizes = [500, 1000, 2000]
results = test_chunk_sizes(documents, chunk_sizes)

# 결과 비교
print("\n" + "="*80)
print("청크 크기별 결과 비교")
print("="*80)
for size, result in results.items():
    print(f"\n청크 크기: {size}자")
    print(f"  - 청크 수: {result['chunk_count']}")
    print(f"  - 실행 시간: {result['elapsed_time']:.2f}초")
    print(f"  - 최종 요약 길이: {len(result['final_summary'])}자")
```

**예상 출력:**
```
================================================================================
청크 크기별 결과 비교
================================================================================

청크 크기: 500자
  - 청크 수: 10
  - 실행 시간: 8.5초
  - 최종 요약 길이: 520자

청크 크기: 1000자
  - 청크 수: 5
  - 실행 시간: 5.2초
  - 최종 요약 길이: 458자

청크 크기: 2000자
  - 청크 수: 3
  - 실행 시간: 3.8초
  - 최종 요약 길이: 385자
```

**분석:**
- 청크가 작을수록: 더 많은 병렬 작업, 더 긴 실행 시간, 더 상세한 요약
- 청크가 클수록: 적은 병렬 작업, 빠른 실행 시간, 간결한 요약
- **최적 청크 크기**: 1000자 (품질과 성능의 균형)

### 실습 2 솔루션: PDF 파일 처리

```python
from langchain_community.document_loaders import PyPDFLoader

def summarize_pdf(pdf_path: str):
    """PDF 파일을 로드하여 요약"""
    print(f"PDF 로드 중: {pdf_path}")

    # PDF 로더로 문서 로드
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"로드된 페이지 수: {len(documents)}")

    # 페이지 정보 출력
    for i, doc in enumerate(documents[:3]):  # 처음 3페이지만
        print(f"\n페이지 {i+1} 메타데이터:")
        print(f"  - source: {doc.metadata.get('source', 'N/A')}")
        print(f"  - page: {doc.metadata.get('page', 'N/A')}")
        print(f"  - 길이: {len(doc.page_content)}자")

    # 그래프 실행
    initial_state = {"contents": documents}

    final_state = None
    for step in graph.stream(initial_state, stream_mode="values"):
        if "final_summary" in step:
            final_state = step

    # 최종 요약에 페이지 출처 추가
    print("\n" + "="*80)
    print("PDF 문서 요약")
    print("="*80)
    print(f"파일: {pdf_path}")
    print(f"총 페이지: {len(documents)}")
    print(f"청크 수: {len(final_state.get('chunks', []))}")
    print("\n최종 요약:")
    print(final_state.get("final_summary", "요약 생성 실패"))

    return final_state

# 실행
pdf_result = summarize_pdf("data/sample_document.pdf")
```

**페이지 출처를 포함한 개선된 요약:**

```python
def create_final_summary_with_sources(state: SummarizationState):
    """페이지 출처를 포함한 최종 요약 생성"""
    # 인덱스별로 요약을 정렬
    sorted_summaries = sorted(state["summaries"], key=lambda x: x[0])

    # 청크별 출처 정보 수집
    chunk_sources = {}
    for chunk in state["chunks"]:
        chunk_sources[chunk["index"]] = chunk["source_metadata"]

    # 요약과 출처를 함께 결합
    summaries_with_sources = []
    for idx, summary in sorted_summaries:
        source_info = chunk_sources.get(idx, {})
        page = source_info.get("page", "Unknown")
        summaries_with_sources.append(f"[페이지 {page}] {summary}")

    combined_summaries = "\n\n".join(summaries_with_sources)

    prompt = f"""다음은 PDF 문서를 페이지별로 요약한 내용들입니다.
    이들을 종합하여 하나의 포괄적인 최종 요약을 작성해주세요.
    중요한 내용에는 페이지 번호를 참조해주세요:

    {combined_summaries}

    최종 요약 (페이지 참조 포함):
    """

    response = model.invoke(prompt)
    return {"final_summary": response.content}
```

### 실습 3 솔루션: 다단계 요약 시스템

```python
# 다단계 요약을 위한 확장된 상태
class MultiStageSummarizationState(TypedDict):
    contents: List[Document]
    chunks: List[Dict[str, Any]]
    detailed_summaries: Annotated[List[Tuple[int, str]], operator.add]  # 1단계 상세 요약
    intermediate_chunks: List[Dict[str, Any]]                            # 중간 청크
    intermediate_summaries: Annotated[List[Tuple[int, str]], operator.add]  # 2단계 중간 요약
    final_summary: str

def create_detailed_summary(state: DocumentState):
    """1단계: 청크별 상세 요약 (3-5문장)"""
    prompt = f"""다음 텍스트를 3-5문장으로 상세하게 요약해주세요.
    중요한 세부사항과 핵심 개념을 모두 포함하세요:

    {state['content']}
    """

    response = model.invoke(prompt)
    return {"detailed_summaries": [(state["index"], response.content)]}

def create_intermediate_chunks(state: MultiStageSummarizationState):
    """상세 요약들을 다시 청크로 분할"""
    # 상세 요약들을 정렬
    sorted_summaries = sorted(state["detailed_summaries"], key=lambda x: x[0])

    # 상세 요약들을 결합
    all_summaries = "\n\n".join([s for _, s in sorted_summaries])

    # 중간 청크로 분할 (청크 크기: 1500자)
    chunk_size = 1500
    intermediate_chunks = []

    for i in range(0, len(all_summaries), chunk_size):
        chunk_content = all_summaries[i:i + chunk_size]
        if chunk_content.strip():
            intermediate_chunks.append({
                "index": len(intermediate_chunks),
                "content": chunk_content
            })

    return {"intermediate_chunks": intermediate_chunks}

def create_intermediate_summary(state: DocumentState):
    """2단계: 중간 요약 생성"""
    prompt = f"""다음은 문서의 상세 요약들입니다.
    이를 2-3문장으로 압축하여 핵심만 추출해주세요:

    {state['content']}
    """

    response = model.invoke(prompt)
    return {"intermediate_summaries": [(state["index"], response.content)]}

def continue_to_intermediate_summarization(state: MultiStageSummarizationState):
    """중간 요약을 위한 Send 생성"""
    return [
        Send("create_intermediate_summary", {
            "content": chunk["content"],
            "index": chunk["index"]
        })
        for chunk in state["intermediate_chunks"]
    ]

def create_final_summary_multistage(state: MultiStageSummarizationState):
    """3단계: 최종 요약 생성"""
    # 중간 요약들을 정렬하고 결합
    sorted_summaries = sorted(state["intermediate_summaries"], key=lambda x: x[0])
    combined_summaries = "\n\n".join([s for _, s in sorted_summaries])

    prompt = f"""다음은 문서의 중간 요약들입니다.
    이들을 종합하여 포괄적이고 일관성 있는 최종 요약을 작성해주세요:

    {combined_summaries}

    최종 요약:
    """

    response = model.invoke(prompt)
    return {"final_summary": response.content}

# 다단계 그래프 구성
builder = StateGraph(MultiStageSummarizationState)

# 노드 추가
builder.add_node("split_documents", split_documents)
builder.add_node("create_detailed_summary", create_detailed_summary)
builder.add_node("create_intermediate_chunks", create_intermediate_chunks)
builder.add_node("create_intermediate_summary", create_intermediate_summary)
builder.add_node("create_final_summary", create_final_summary_multistage)

# 엣지 연결
builder.add_edge(START, "split_documents")
builder.add_conditional_edges(
    "split_documents",
    continue_to_summarization,
    ["create_detailed_summary"]
)
builder.add_edge("create_detailed_summary", "create_intermediate_chunks")
builder.add_conditional_edges(
    "create_intermediate_chunks",
    continue_to_intermediate_summarization,
    ["create_intermediate_summary"]
)
builder.add_edge("create_intermediate_summary", "create_final_summary")
builder.add_edge("create_final_summary", END)

# 컴파일 및 실행
multistage_graph = builder.compile()

# 실행
result = multistage_graph.invoke({"contents": documents})

print("="*80)
print("다단계 요약 결과")
print("="*80)
print(f"1단계 상세 요약 수: {len(result['detailed_summaries'])}")
print(f"2단계 중간 청크 수: {len(result['intermediate_chunks'])}")
print(f"2단계 중간 요약 수: {len(result['intermediate_summaries'])}")
print(f"\n최종 요약:\n{result['final_summary']}")
```

**다단계 요약의 장점:**
- ✅ 더 정확한 요약 (단계별 정제)
- ✅ 긴 문서에 효과적 (계층적 처리)
- ✅ 세부사항 보존 (1단계 상세 요약)
- ✅ 일관성 향상 (2단계 통합)

### 실습 4 솔루션: 에러 복구 메커니즘

```python
# 재시도 로직이 포함된 상태
class ResilientSummarizationState(TypedDict):
    contents: List[Document]
    chunks: List[Dict[str, Any]]
    summaries: Annotated[List[Tuple[int, str]], operator.add]
    failed_chunks: List[Dict[str, Any]]  # 실패한 청크 정보
    final_summary: str

class DocumentStateWithRetry(TypedDict):
    content: str
    index: int
    retry_count: int  # 재시도 횟수

def summarize_document_with_retry(state: DocumentStateWithRetry):
    """재시도 로직이 포함된 요약 함수"""
    max_retries = 3
    retry_count = state.get("retry_count", 0)

    prompt = f"""다음 텍스트를 2-3문장으로 간결하게 요약해주세요:

    {state['content']}
    """

    try:
        # LLM 호출 시도
        response = model.invoke(prompt)
        summary = response.content

        # 성공 시 요약 반환
        return {"summaries": [(state["index"], summary)]}

    except Exception as e:
        print(f"⚠️ 청크 {state['index']} 요약 실패 (시도 {retry_count + 1}/{max_retries}): {str(e)}")

        # 재시도 가능하면 재시도
        if retry_count < max_retries - 1:
            import time
            time.sleep(1)  # 1초 대기 후 재시도

            # 재시도 카운터 증가하여 재귀 호출
            return summarize_document_with_retry({
                "content": state["content"],
                "index": state["index"],
                "retry_count": retry_count + 1
            })
        else:
            # 최대 재시도 초과 시 실패 처리
            print(f"❌ 청크 {state['index']} 요약 최종 실패")

            # 실패 정보를 별도로 기록
            failed_info = {
                "index": state["index"],
                "error": str(e),
                "retry_count": retry_count + 1
            }

            # 실패한 청크는 기본 메시지로 대체
            return {
                "summaries": [(state["index"], f"[요약 실패: 청크 {state['index']}]")],
                "failed_chunks": [failed_info]
            }

def create_final_summary_resilient(state: ResilientSummarizationState):
    """실패 정보를 포함한 최종 요약 생성"""
    # 요약 정렬
    sorted_summaries = sorted(state["summaries"], key=lambda x: x[0])
    ordered_summaries = [summary for _, summary in sorted_summaries]
    combined_summaries = "\n\n".join(ordered_summaries)

    # 실패한 청크 정보
    failed_chunks = state.get("failed_chunks", [])
    failed_info = ""
    if failed_chunks:
        failed_indices = [chunk["index"] for chunk in failed_chunks]
        failed_info = f"\n\n⚠️ 주의: 청크 {failed_indices}의 요약이 실패했습니다."

    prompt = f"""다음은 문서를 청크별로 요약한 내용들입니다.
    이들을 종합하여 하나의 포괄적인 최종 요약을 작성해주세요:

    {combined_summaries}
    {failed_info}

    최종 요약:
    """

    try:
        response = model.invoke(prompt)
        final_summary = response.content

        # 실패 정보 추가
        if failed_chunks:
            final_summary += f"\n\n⚠️ 일부 청크({len(failed_chunks)}개)의 요약이 실패했습니다."

        return {"final_summary": final_summary}

    except Exception as e:
        return {"final_summary": f"최종 요약 생성 실패: {str(e)}"}

# 복원력 있는 그래프 구성
builder = StateGraph(ResilientSummarizationState)
builder.add_node("split_documents", split_documents)
builder.add_node("summarize_document", summarize_document_with_retry)
builder.add_node("create_final_summary", create_final_summary_resilient)

builder.add_edge(START, "split_documents")
builder.add_conditional_edges(
    "split_documents",
    continue_to_summarization,
    ["summarize_document"]
)
builder.add_edge("summarize_document", "create_final_summary")
builder.add_edge("create_final_summary", END)

resilient_graph = builder.compile()

# 실행
result = resilient_graph.invoke({"contents": documents})

print("="*80)
print("복원력 있는 요약 시스템 결과")
print("="*80)
print(f"성공한 요약 수: {len(result['summaries']) - len(result.get('failed_chunks', []))}")
print(f"실패한 청크 수: {len(result.get('failed_chunks', []))}")
print(f"\n최종 요약:\n{result['final_summary']}")
```

**에러 복구 메커니즘의 장점:**
- ✅ 일시적 오류 복구 (네트워크 이슈 등)
- ✅ 부분 실패 허용 (전체 프로세스 중단 방지)
- ✅ 실패 추적 (문제 디버깅 용이)
- ✅ 사용자 피드백 (실패 정보 명시)

## 🚀 실무 활용 예시

### 예시 1: 대용량 법률 문서 요약 시스템

```python
from langchain_community.document_loaders import PyPDFLoader
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LegalDocumentSummarizer:
    """법률 문서 전용 요약 시스템"""

    def __init__(self, model_name="gpt-4o-mini"):
        self.model = ChatOpenAI(model=model_name, temperature=0)
        self.graph = self._build_graph()

    def _build_graph(self):
        """법률 문서용 커스터마이징된 그래프 구성"""
        builder = StateGraph(SummarizationState)

        # 법률 문서 전용 요약 함수
        def summarize_legal_chunk(state: DocumentState):
            prompt = f"""다음 법률 문서의 내용을 전문적으로 요약해주세요.

            요약 시 포함해야 할 사항:
            - 주요 법적 조항
            - 권리와 의무
            - 중요한 날짜와 기한
            - 법적 용어 설명

            텍스트:
            {state['content']}

            전문적인 법률 요약:
            """

            try:
                response = self.model.invoke(prompt)
                return {"summaries": [(state["index"], response.content)]}
            except Exception as e:
                logger.error(f"청크 {state['index']} 요약 실패: {e}")
                return {"summaries": [(state["index"], f"[요약 실패: {str(e)}]")]}

        # 노드 추가
        builder.add_node("split_documents", split_documents)
        builder.add_node("summarize_document", summarize_legal_chunk)
        builder.add_node("create_final_summary", self._create_legal_final_summary)

        # 엣지 연결
        builder.add_edge(START, "split_documents")
        builder.add_conditional_edges(
            "split_documents",
            continue_to_summarization,
            ["summarize_document"]
        )
        builder.add_edge("summarize_document", "create_final_summary")
        builder.add_edge("create_final_summary", END)

        return builder.compile()

    def _create_legal_final_summary(self, state: SummarizationState):
        """법률 문서 전용 최종 요약"""
        sorted_summaries = sorted(state["summaries"], key=lambda x: x[0])
        ordered_summaries = [summary for _, summary in sorted_summaries]
        combined_summaries = "\n\n".join(ordered_summaries)

        prompt = f"""다음은 법률 문서를 부분별로 요약한 내용입니다.
        이를 종합하여 법률 전문가를 위한 포괄적인 요약을 작성해주세요:

        포함 사항:
        1. 문서의 주요 목적과 범위
        2. 핵심 법적 조항 및 의무사항
        3. 중요 날짜 및 기한
        4. 주의사항 및 법적 위험 요소

        부분별 요약:
        {combined_summaries}

        ## 법률 문서 종합 요약
        """

        try:
            response = self.model.invoke(prompt)
            return {"final_summary": response.content}
        except Exception as e:
            logger.error(f"최종 요약 생성 실패: {e}")
            return {"final_summary": f"최종 요약 생성 실패: {str(e)}"}

    def summarize_pdf(self, pdf_path: str):
        """PDF 법률 문서 요약"""
        logger.info(f"법률 문서 처리 시작: {pdf_path}")

        # PDF 로드
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        logger.info(f"로드된 페이지 수: {len(documents)}")

        # 그래프 실행
        result = self.graph.invoke({"contents": documents})

        logger.info("요약 완료")

        return {
            "file": pdf_path,
            "pages": len(documents),
            "chunks": len(result.get("chunks", [])),
            "summary": result.get("final_summary", "")
        }

# 사용 예시
summarizer = LegalDocumentSummarizer()

# 여러 법률 문서 처리
legal_docs = [
    "contracts/service_agreement.pdf",
    "contracts/nda.pdf",
    "contracts/employment_contract.pdf"
]

for doc_path in legal_docs:
    result = summarizer.summarize_pdf(doc_path)

    print(f"\n{'='*80}")
    print(f"파일: {result['file']}")
    print(f"페이지: {result['pages']}, 청크: {result['chunks']}")
    print('='*80)
    print(result['summary'])
    print('='*80)
```

### 예시 2: 다국어 문서 요약 시스템

```python
from langchain_community.document_loaders import UnstructuredFileLoader

class MultilingualDocumentSummarizer:
    """다국어 문서 요약 시스템"""

    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.supported_languages = {
            "ko": "한국어",
            "en": "영어",
            "ja": "일본어",
            "zh": "중국어"
        }

    def detect_language(self, text: str) -> str:
        """텍스트 언어 감지"""
        prompt = f"""다음 텍스트의 언어를 감지하고 언어 코드만 반환해주세요.
        (ko, en, ja, zh 중 하나)

        텍스트: {text[:200]}

        언어 코드:
        """

        response = self.model.invoke(prompt)
        lang_code = response.content.strip().lower()
        return lang_code if lang_code in self.supported_languages else "en"

    def summarize_multilingual_chunk(self, content: str, index: int, target_lang: str = "ko"):
        """다국어 청크 요약 (목표 언어로 변환)"""
        # 원본 언어 감지
        source_lang = self.detect_language(content)

        prompt = f"""다음 텍스트를 요약하고 {self.supported_languages[target_lang]}로 번역해주세요.

        원본 언어: {self.supported_languages.get(source_lang, '알 수 없음')}
        목표 언어: {self.supported_languages[target_lang]}

        텍스트:
        {content}

        {self.supported_languages[target_lang]} 요약:
        """

        try:
            response = self.model.invoke(prompt)
            return {
                "summaries": [(index, response.content)],
                "language": source_lang
            }
        except Exception as e:
            return {
                "summaries": [(index, f"[요약 실패: {str(e)}]")],
                "language": "unknown"
            }

    def process_document(self, file_path: str, target_lang: str = "ko"):
        """다국어 문서 처리"""
        # 파일 로드
        loader = UnstructuredFileLoader(file_path)
        documents = loader.load()

        print(f"처리 중: {file_path}")
        print(f"목표 언어: {self.supported_languages[target_lang]}")

        # 문서별 언어 감지
        detected_languages = []
        for doc in documents:
            lang = self.detect_language(doc.page_content)
            detected_languages.append(lang)

        print(f"감지된 언어: {set(detected_languages)}")

        # 청크 분할 및 요약
        all_summaries = []
        for i, doc in enumerate(documents):
            result = self.summarize_multilingual_chunk(
                doc.page_content,
                i,
                target_lang
            )
            all_summaries.extend(result["summaries"])

        # 최종 요약 통합
        sorted_summaries = sorted(all_summaries, key=lambda x: x[0])
        combined = "\n\n".join([s for _, s in sorted_summaries])

        final_prompt = f"""다음은 다국어 문서를 {self.supported_languages[target_lang]}로 요약한 내용들입니다.
        이를 하나의 일관성 있는 {self.supported_languages[target_lang]} 요약으로 통합해주세요:

        {combined}

        최종 {self.supported_languages[target_lang]} 요약:
        """

        final_response = self.model.invoke(final_prompt)

        return {
            "file": file_path,
            "source_languages": list(set(detected_languages)),
            "target_language": target_lang,
            "final_summary": final_response.content
        }

# 사용 예시
multilingual_summarizer = MultilingualDocumentSummarizer()

# 다양한 언어의 문서 처리
documents = [
    {"file": "docs/english_report.pdf", "target": "ko"},
    {"file": "docs/japanese_manual.pdf", "target": "ko"},
    {"file": "docs/korean_contract.pdf", "target": "en"}
]

for doc_info in documents:
    result = multilingual_summarizer.process_document(
        doc_info["file"],
        doc_info["target"]
    )

    print(f"\n{'='*80}")
    print(f"파일: {result['file']}")
    print(f"원본 언어: {result['source_languages']}")
    print(f"목표 언어: {result['target_language']}")
    print('='*80)
    print(result['final_summary'])
```

## 📖 참고 자료

### 공식 문서
- [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph/)
- [LangGraph Send 객체](https://langchain-ai.github.io/langgraph/concepts/low_level/#send)
- [StateGraph API](https://langchain-ai.github.io/langgraph/reference/graphs/)
- [LangChain Document Loaders](https://python.langchain.com/docs/modules/data_connection/document_loaders/)

### Send 패턴 관련
- [Map-Reduce 패턴](https://langchain-ai.github.io/langgraph/how-tos/map-reduce/)
- [병렬 처리 가이드](https://langchain-ai.github.io/langgraph/how-tos/branching/)
- [동적 그래프 구성](https://langchain-ai.github.io/langgraph/how-tos/dynamic-breakpoints/)

### 문서 처리
- [PDF 처리 가이드](https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf)
- [텍스트 분할 전략](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [문서 요약 베스트 프랙티스](https://python.langchain.com/docs/use_cases/summarization)

### 추가 학습 자료
- [LangGraph 튜토리얼](https://github.com/langchain-ai/langgraph/tree/main/examples)
- [병렬 처리 최적화](https://docs.python.org/3/library/concurrent.futures.html)
- [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/)
