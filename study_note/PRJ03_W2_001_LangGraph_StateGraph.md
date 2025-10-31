# LangGraph StateGraph 활용 - 상태 기반 워크플로우 구현

## 📚 학습 목표

- **StateGraph의 기본 구조**를 이해하고 State, Node, Edge를 활용한 워크플로우를 구현할 수 있다
- **조건부 엣지(Conditional Edge)**를 사용하여 동적 라우팅 시스템을 설계할 수 있다
- **Command 객체**를 활용하여 상태 업데이트와 흐름 제어를 동시에 수행할 수 있다
- **invoke vs stream** 실행 방식의 차이를 이해하고 상황에 맞게 선택할 수 있다
- **다국어 RAG 시스템**에서 StateGraph를 활용한 라우팅 로직을 구현할 수 있다

## 🔑 핵심 개념

### StateGraph란?

**StateGraph**는 LangGraph의 핵심 구성 요소로, **상태 기반의 그래프 구조**를 통해 복잡한 대화 흐름과 데이터 처리 과정을 체계적으로 관리할 수 있는 프레임워크입니다.

#### 주요 구성 요소

1. **State (상태)**
   - 그래프에서 처리하는 데이터의 기본 구조
   - TypedDict를 사용하여 명확한 타입 정의
   - 각 노드가 상태를 읽고 업데이트

2. **Node (노드)**
   - 독립적인 작업 단위를 수행하는 함수
   - 상태를 입력으로 받아 처리하고 업데이트된 상태 반환
   - 각 노드는 특정 비즈니스 로직을 캡슐화

3. **Edge (엣지)**
   - 노드 간의 연결 경로를 정의
   - 단순 엣지: 무조건적인 흐름 전환
   - 조건부 엣지: 상태에 따라 동적으로 경로 결정

4. **Command 객체**
   - 상태 업데이트와 흐름 제어를 동시에 수행
   - `goto`로 다음 노드 지정, `update`로 상태 변경
   - 조건부 엣지의 대안으로 더 간결한 코드 작성 가능

### StateGraph vs 기존 체인 방식

| 구분 | 기존 체인 | StateGraph |
|------|----------|------------|
| 흐름 제어 | 선형적, 고정적 | 동적, 조건부 분기 가능 |
| 상태 관리 | 암묵적 | 명시적, TypedDict로 정의 |
| 복잡도 | 단순한 파이프라인 | 복잡한 워크플로우 |
| 디버깅 | 어려움 | 시각화 및 스트리밍으로 용이 |
| 재사용성 | 낮음 | 높음 (노드 단위 재사용) |

### 언제 StateGraph를 사용하나?

- **조건부 분기가 필요한 경우**: 사용자 입력이나 상태에 따라 다른 처리 경로 필요
- **복잡한 워크플로우**: 여러 단계의 처리와 검증이 필요한 시스템
- **상태 추적이 중요한 경우**: 각 단계의 결과를 명시적으로 관리해야 하는 상황
- **다중 에이전트 시스템**: 여러 에이전트가 협력하여 작업을 수행하는 시나리오

## 🛠 환경 설정

### 1. 필수 라이브러리 설치

```bash
# LangGraph 및 관련 라이브러리 설치
pip install langgraph langchain-openai langchain-chroma langchain-core

# 환경 변수 관리
pip install python-dotenv

# 언어 감지 라이브러리 (실습용)
pip install langdetect
```

### 2. 환경 변수 설정

`.env` 파일을 생성하고 OpenAI API 키를 설정합니다:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. 기본 설정 코드

```python
# 환경 변수 로드
from dotenv import load_dotenv
load_dotenv()

# 기본 라이브러리
import os
from pprint import pprint
from typing import TypedDict, Literal, List

# LangChain 및 LangGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

# 시각화
from IPython.display import Image, display
```

## 💻 단계별 구현

### 1단계: 기본 StateGraph 구성

#### State 정의

State는 그래프에서 처리하는 데이터의 스키마를 정의합니다. TypedDict를 사용하여 명확한 타입을 지정합니다.

```python
from typing import TypedDict

# 상태 정의 - 문서 요약 시스템 예제
class State(TypedDict):
    original_text: str   # 원본 텍스트
    summary: str         # 요약본
    final_summary: str   # 최종 요약본
```

**핵심 포인트:**
- TypedDict로 타입 안정성 확보
- 각 필드는 그래프 실행 중 업데이트 가능
- 다른 노드 간 데이터 공유의 중심

#### Node 함수 작성

노드는 상태를 입력받아 처리하고 업데이트된 상태를 반환하는 함수입니다.

```python
from langchain_openai import ChatOpenAI

# LLM 인스턴스 생성
llm = ChatOpenAI(model="gpt-4.1-mini")

def generate_summary(state: State):
    """원본 텍스트를 요약하는 노드"""
    prompt = f"""다음 텍스트를 핵심 내용 중심으로 간단히 요약해주세요:

    [텍스트]
    {state['original_text']}

    [요약]
    """
    response = llm.invoke(prompt)

    # 상태 업데이트 딕셔너리 반환
    return {"summary": response.content}
```

**핵심 포인트:**
- 노드 함수는 `state` 파라미터를 받음
- 반환값은 업데이트할 상태 필드의 딕셔너리
- 기존 상태는 자동으로 병합됨

#### Graph 구성 및 컴파일

```python
from langgraph.graph import StateGraph, START, END

# StateGraph 객체 생성
workflow = StateGraph(State)

# 노드 추가
workflow.add_node("generate_summary", generate_summary)

# 엣지 추가: START -> generate_summary -> END
workflow.add_edge(START, "generate_summary")
workflow.add_edge("generate_summary", END)

# 그래프 컴파일
graph = workflow.compile()

# 그래프 시각화
display(Image(graph.get_graph().draw_mermaid_png()))
```

**그래프 구조:**
```
START → generate_summary → END
```

#### 실행: invoke 방식

`invoke`는 가장 기본적인 실행 방식으로, 전체 처리가 완료된 후 최종 결과만 반환합니다.

```python
# 초기 상태 설정
text = """
인공지능(AI)은 컴퓨터 과학의 한 분야로, 인간의 학습능력과 추론능력, 지각능력,
자연언어의 이해능력 등을 컴퓨터 프로그램으로 실현한 기술이다.
최근에는 기계학습과 딥러닝의 발전으로 다양한 분야에서 활용되고 있다.
"""

initial_state = {
    "original_text": text,
}

# 그래프 실행
final_state = graph.invoke(initial_state)

# 결과 출력
for key, value in final_state.items():
    print(f"{key}: {value}")
```

**예상 출력:**
```
original_text: 인공지능(AI)은 컴퓨터 과학의 한 분야로...
summary: 인공지능(AI)은 인간의 학습, 추론, 지각, 자연언어 이해 능력을 컴퓨터로 구현한 기술로, 최근 기계학습과 딥러닝의 발전으로 다양한 분야에서 활용되고 있다.
```

### 2단계: 조건부 엣지 활용

조건부 엣지를 사용하면 상태에 따라 동적으로 다음 노드를 선택할 수 있습니다.

#### 품질 검사 함수 작성

```python
from typing import Literal

def check_summary_quality(state: State) -> Literal["needs_improvement", "good"]:
    """요약의 품질을 체크하고 개선이 필요한지 판단하는 함수"""
    prompt = f"""다음 요약의 품질을 평가해주세요.
    요약이 명확하고 핵심을 잘 전달하면 'good'을,
    개선이 필요하면 'needs_improvement'를 응답해주세요.

    요약본: {state['summary']}
    """
    response = llm.invoke(prompt).content.lower().strip()

    if "good" in response:
        print("✅ 품질 검사 통과")
        return "good"
    else:
        print("⚠️ 개선 필요")
        return "needs_improvement"
```

**핵심 포인트:**
- 반환 타입은 `Literal`로 가능한 경로를 명시
- 반환값은 조건부 엣지의 매핑 키와 일치해야 함

#### 추가 노드 작성

```python
def improve_summary(state: State):
    """요약을 개선하고 다듬는 노드"""
    prompt = f"""다음 요약을 더 명확하고 간결하게 개선해주세요:

    요약본: {state['summary']}
    """
    response = llm.invoke(prompt)

    return {"final_summary": response.content}

def finalize_summary(state: State):
    """현재 요약을 최종 요약으로 설정하는 노드"""
    return {"final_summary": state["summary"]}
```

#### 조건부 엣지가 포함된 Graph 구성

```python
# 워크플로우 구성
workflow = StateGraph(State)

# 노드 추가
workflow.add_node("generate_summary", generate_summary)
workflow.add_node("improve_summary", improve_summary)
workflow.add_node("finalize_summary", finalize_summary)

# 조건부 엣지 추가
workflow.add_conditional_edges(
    "generate_summary",          # 시작 노드
    check_summary_quality,       # 조건 판단 함수
    {
        "needs_improvement": "improve_summary",  # 개선 필요 시
        "good": "finalize_summary"               # 품질 통과 시
    }
)

# 기본 엣지 추가
workflow.add_edge(START, "generate_summary")
workflow.add_edge("improve_summary", END)
workflow.add_edge("finalize_summary", END)

# 그래프 컴파일
graph = workflow.compile()

# 그래프 시각화
display(Image(graph.get_graph().draw_mermaid_png()))
```

**그래프 구조:**
```
                    ┌─ check_summary_quality ─┐
START → generate_summary                       ↓
                    ├─→ improve_summary → END
                    └─→ finalize_summary → END
```

### 3단계: Stream 실행 방식

Stream 방식은 그래프 실행의 중간 과정을 실시간으로 확인할 수 있어 디버깅과 모니터링에 유용합니다.

#### stream_mode="values"

각 단계에서의 전체 상태 값을 스트리밍합니다.

```python
# values 모드: 전체 상태 값 확인
for chunk in graph.stream(initial_state, stream_mode="values"):
    print("=== 현재 상태 ===")
    pprint(chunk)
    print()
```

**예상 출력:**
```
=== 현재 상태 ===
{'original_text': '인공지능(AI)은 컴퓨터 과학의...'}

=== 현재 상태 ===
{'original_text': '인공지능(AI)은...', 'summary': '인공지능(AI)은 인간의 학습...'}

=== 현재 상태 ===
{'original_text': '...', 'summary': '...', 'final_summary': '인공지능(AI)은...'}
```

#### stream_mode="updates"

어떤 노드가 상태를 업데이트했는지 확인할 수 있습니다 (디버깅용).

```python
# updates 모드: 노드별 업데이트 내역 확인
for chunk in graph.stream(initial_state, stream_mode="updates"):
    print("=== 노드 업데이트 ===")
    pprint(chunk)
    print()
```

**예상 출력:**
```
=== 노드 업데이트 ===
{'generate_summary': {'summary': '인공지능(AI)은 인간의 학습...'}}

=== 노드 업데이트 ===
{'finalize_summary': {'final_summary': '인공지능(AI)은...'}}
```

**invoke vs stream 선택 기준:**

| 상황 | 권장 방식 | 이유 |
|------|----------|------|
| 단순 결과만 필요 | invoke | 간결하고 빠름 |
| 진행 상황 표시 | stream (values) | 사용자 경험 향상 |
| 디버깅 | stream (updates) | 노드별 추적 가능 |
| 실시간 응답 | stream | 점진적 피드백 |

### 4단계: Command 객체 활용

Command 객체는 상태 업데이트와 흐름 제어를 동시에 수행할 수 있어 조건부 엣지보다 간결한 코드 작성이 가능합니다.

#### Command 기반 노드 작성

```python
from langgraph.types import Command

def generate_summary_with_command(state: State) -> Command[Literal["improve_summary", "finalize_summary"]]:
    """요약 생성 및 품질 평가를 한 번에 수행하는 노드"""

    # 1. 요약 생성
    summary_prompt = f"""다음 텍스트를 핵심 내용 중심으로 간단히 요약해주세요:
    [텍스트]
    {state['original_text']}
    [요약]
    """
    summary = llm.invoke(summary_prompt).content

    # 2. 품질 평가
    eval_prompt = f"""다음 요약의 품질을 평가해주세요.
    요약이 명확하고 핵심을 잘 전달하면 'good'을,
    개선이 필요하면 'needs_improvement'를 응답해주세요.

    요약본: {summary}
    """
    quality = llm.invoke(eval_prompt).content.lower().strip()

    # 3. Command로 상태 업데이트와 라우팅을 동시에 수행
    return Command(
        goto="finalize_summary" if "good" in quality else "improve_summary",
        update={"summary": summary}
    )

def improve_summary_with_command(state: State) -> Command[Literal[END]]:
    """요약을 개선하는 노드 (Command 사용)"""
    prompt = f"""다음 요약을 더 명확하고 간결하게 개선해주세요:
    [기존 요약]
    {state['summary']}
    [개선 요약]
    """
    improved_summary = llm.invoke(prompt).content

    return Command(
        goto=END,
        update={"final_summary": improved_summary}
    )

def finalize_summary_with_command(state: State) -> Command[Literal[END]]:
    """현재 요약을 최종 요약으로 설정하는 노드 (Command 사용)"""
    return Command(
        goto=END,
        update={"final_summary": state["summary"]}
    )
```

#### Command 기반 Graph 구성

```python
# 워크플로우 구성
workflow = StateGraph(State)

# 노드 추가
workflow.add_node("generate_summary", generate_summary_with_command)
workflow.add_node("improve_summary", improve_summary_with_command)
workflow.add_node("finalize_summary", finalize_summary_with_command)

# 기본 엣지만 추가 (Command가 라우팅 담당)
workflow.add_edge(START, "generate_summary")

# 그래프 컴파일
graph = workflow.compile()
```

**Command vs 조건부 엣지 비교:**

| 구분 | 조건부 엣지 | Command 객체 |
|------|-----------|-------------|
| 로직 위치 | 노드 외부 (별도 함수) | 노드 내부 |
| 코드 간결성 | 낮음 (분리된 함수) | 높음 (통합) |
| 상태 업데이트 | 노드 반환 + 엣지 함수 | Command 한 번에 |
| 정보 전달 | 제한적 | 유연함 |
| 사용 시기 | 단순 분기 | 복잡한 로직 + 상태 업데이트 |

## 🎯 실습 문제

### 문제 1: 기본 StateGraph - 문서 번역 시스템 (난이도: ⭐⭐)

**요구사항:**
다음 조건을 만족하는 문서 번역 시스템을 StateGraph로 구현하세요.

1. State 정의:
   - `original_text`: 원본 텍스트
   - `detected_language`: 감지된 언어
   - `translated_text`: 번역된 텍스트

2. 노드 구성:
   - `detect_language`: 원본 텍스트의 언어 감지
   - `translate_to_english`: 영어로 번역
   - `translate_to_korean`: 한국어로 번역

3. 흐름:
   - 언어 감지 → 영어면 한국어로 번역, 한국어면 영어로 번역

**힌트:**
```python
# State 정의 예시
class TranslationState(TypedDict):
    original_text: str
    detected_language: str
    translated_text: str
```

---

### 문제 2: 조건부 엣지 - 다단계 품질 검증 시스템 (난이도: ⭐⭐⭐)

**요구사항:**
다음 조건을 만족하는 문서 품질 검증 시스템을 구현하세요.

1. State 정의:
   - `document`: 문서 내용
   - `grammar_score`: 문법 점수 (0-100)
   - `clarity_score`: 명확성 점수 (0-100)
   - `final_document`: 최종 문서
   - `revision_count`: 수정 횟수

2. 노드 구성:
   - `check_grammar`: 문법 점수 계산
   - `check_clarity`: 명확성 점수 계산
   - `improve_grammar`: 문법 개선
   - `improve_clarity`: 명확성 개선
   - `finalize`: 최종 승인

3. 흐름:
   - 문법 검사 → 70점 미만이면 문법 개선, 70점 이상이면 명확성 검사
   - 명확성 검사 → 70점 미만이면 명확성 개선, 70점 이상이면 최종 승인
   - 개선 후 다시 해당 검사로 돌아가기 (최대 3회)

**힌트:**
```python
def check_grammar(state: TranslationState) -> Literal["improve_grammar", "check_clarity"]:
    # 문법 점수 계산 로직
    if state['grammar_score'] < 70:
        return "improve_grammar"
    return "check_clarity"
```

---

### 문제 3: Command 활용 - 스마트 고객 지원 라우터 (난이도: ⭐⭐⭐)

**요구사항:**
Command 객체를 사용하여 고객 문의를 적절한 부서로 라우팅하는 시스템을 구현하세요.

1. State 정의:
   - `customer_query`: 고객 문의
   - `category`: 문의 카테고리 (기술지원/결제/일반)
   - `priority`: 우선순위 (긴급/보통/낮음)
   - `assigned_department`: 배정된 부서
   - `response`: 응답

2. 노드 구성:
   - `categorize_query`: 문의 분류 및 우선순위 판단
   - `technical_support`: 기술 지원 응답
   - `billing_support`: 결제 지원 응답
   - `general_support`: 일반 지원 응답

3. 흐름:
   - categorize_query에서 Command로 카테고리와 우선순위를 동시에 업데이트하고 적절한 부서로 라우팅
   - 각 부서 노드는 우선순위에 따라 다른 응답 생성

**힌트:**
```python
def categorize_query(state: SupportState) -> Command[Literal["technical_support", "billing_support", "general_support"]]:
    # LLM으로 카테고리와 우선순위 판단
    analysis = llm.invoke(f"분석: {state['customer_query']}")

    return Command(
        goto=determine_department(analysis),
        update={
            "category": extract_category(analysis),
            "priority": extract_priority(analysis)
        }
    )
```

---

### 문제 4: 실전 프로젝트 - 다국어 RAG 시스템 개선 (난이도: ⭐⭐⭐⭐)

**요구사항:**
노트북의 실습 문제를 확장하여 다음 기능을 추가하세요.

1. 지원 언어 확장:
   - 한국어, 영어, 일본어 3개 언어 지원
   - 각 언어별 벡터 DB 구성

2. 혼합 언어 처리:
   - 한 문장에 여러 언어가 섞여 있는 경우 처리
   - 주 언어를 감지하여 해당 DB 사용

3. 폴백(Fallback) 메커니즘:
   - 해당 언어 DB에서 결과가 없으면 다른 언어 DB도 검색
   - 번역을 통한 크로스 언어 검색

4. 응답 품질 향상:
   - 검색 결과가 부족하면 추가 검색 수행
   - 여러 DB의 결과를 종합하여 최종 답변 생성

**힌트:**
```python
class MultilingualRAGState(TypedDict):
    user_query: str
    detected_languages: List[str]  # 감지된 언어 목록
    primary_language: str
    search_results_ko: List[str]
    search_results_en: List[str]
    search_results_ja: List[str]
    confidence_score: float
    final_answer: str
    fallback_used: bool
```

## ✅ 솔루션 예시

### 문제 1 솔루션: 문서 번역 시스템

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

# State 정의
class TranslationState(TypedDict):
    original_text: str
    detected_language: str
    translated_text: str

# LLM 인스턴스
llm = ChatOpenAI(model="gpt-4.1-mini")

# 노드 함수들
def detect_language(state: TranslationState):
    """텍스트의 언어를 감지하는 노드"""
    prompt = f"""다음 텍스트의 언어를 감지하세요. 'korean' 또는 'english'로만 답변하세요.

    텍스트: {state['original_text']}

    언어:"""

    language = llm.invoke(prompt).content.strip().lower()
    return {"detected_language": language}

def translate_to_english(state: TranslationState):
    """한국어를 영어로 번역하는 노드"""
    prompt = f"""다음 한국어 텍스트를 자연스러운 영어로 번역하세요.

    한국어: {state['original_text']}

    영어:"""

    translation = llm.invoke(prompt).content
    return {"translated_text": translation}

def translate_to_korean(state: TranslationState):
    """영어를 한국어로 번역하는 노드"""
    prompt = f"""다음 영어 텍스트를 자연스러운 한국어로 번역하세요.

    영어: {state['original_text']}

    한국어:"""

    translation = llm.invoke(prompt).content
    return {"translated_text": translation}

# 조건부 함수
def route_translation(state: TranslationState) -> Literal["translate_to_english", "translate_to_korean"]:
    """언어에 따라 번역 방향을 결정"""
    if "korean" in state['detected_language']:
        return "translate_to_english"
    else:
        return "translate_to_korean"

# 그래프 구성
workflow = StateGraph(TranslationState)

# 노드 추가
workflow.add_node("detect_language", detect_language)
workflow.add_node("translate_to_english", translate_to_english)
workflow.add_node("translate_to_korean", translate_to_korean)

# 엣지 추가
workflow.add_edge(START, "detect_language")
workflow.add_conditional_edges(
    "detect_language",
    route_translation,
    {
        "translate_to_english": "translate_to_english",
        "translate_to_korean": "translate_to_korean"
    }
)
workflow.add_edge("translate_to_english", END)
workflow.add_edge("translate_to_korean", END)

# 컴파일
translation_graph = workflow.compile()

# 테스트
test_cases = [
    "인공지능 기술은 빠르게 발전하고 있습니다.",
    "Artificial intelligence is rapidly advancing."
]

for text in test_cases:
    print(f"\n원본: {text}")
    result = translation_graph.invoke({"original_text": text})
    print(f"감지된 언어: {result['detected_language']}")
    print(f"번역: {result['translated_text']}")
```

### 문제 2 솔루션: 다단계 품질 검증 시스템

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

# State 정의
class QualityCheckState(TypedDict):
    document: str
    grammar_score: int
    clarity_score: int
    final_document: str
    revision_count: int

llm = ChatOpenAI(model="gpt-4.1-mini")

# 노드 함수들
def check_grammar(state: QualityCheckState) -> Literal["improve_grammar", "check_clarity", "finalize"]:
    """문법을 검사하는 노드"""
    # 최대 수정 횟수 체크
    if state.get('revision_count', 0) >= 3:
        return "finalize"

    prompt = f"""다음 문서의 문법을 0-100점으로 평가하세요. 숫자만 응답하세요.

    문서: {state['document']}

    점수:"""

    score = int(llm.invoke(prompt).content.strip())

    if score < 70:
        return "improve_grammar"
    return "check_clarity"

def check_clarity(state: QualityCheckState) -> Literal["improve_clarity", "finalize"]:
    """명확성을 검사하는 노드"""
    prompt = f"""다음 문서의 명확성을 0-100점으로 평가하세요. 숫자만 응답하세요.

    문서: {state['document']}

    점수:"""

    score = int(llm.invoke(prompt).content.strip())

    if score < 70 and state.get('revision_count', 0) < 3:
        return "improve_clarity"
    return "finalize"

def improve_grammar(state: QualityCheckState):
    """문법을 개선하는 노드"""
    prompt = f"""다음 문서의 문법을 개선하세요.

    원본 문서: {state['document']}

    개선된 문서:"""

    improved = llm.invoke(prompt).content
    revision_count = state.get('revision_count', 0) + 1

    return {
        "document": improved,
        "revision_count": revision_count
    }

def improve_clarity(state: QualityCheckState):
    """명확성을 개선하는 노드"""
    prompt = f"""다음 문서를 더 명확하고 이해하기 쉽게 개선하세요.

    원본 문서: {state['document']}

    개선된 문서:"""

    improved = llm.invoke(prompt).content
    revision_count = state.get('revision_count', 0) + 1

    return {
        "document": improved,
        "revision_count": revision_count
    }

def finalize(state: QualityCheckState):
    """최종 문서를 확정하는 노드"""
    return {"final_document": state['document']}

# 그래프 구성
workflow = StateGraph(QualityCheckState)

# 노드 추가
workflow.add_node("check_grammar", check_grammar)
workflow.add_node("check_clarity", check_clarity)
workflow.add_node("improve_grammar", improve_grammar)
workflow.add_node("improve_clarity", improve_clarity)
workflow.add_node("finalize", finalize)

# 엣지 추가
workflow.add_edge(START, "check_grammar")

# 조건부 엣지
workflow.add_conditional_edges(
    "check_grammar",
    check_grammar,
    {
        "improve_grammar": "improve_grammar",
        "check_clarity": "check_clarity",
        "finalize": "finalize"
    }
)

workflow.add_conditional_edges(
    "check_clarity",
    check_clarity,
    {
        "improve_clarity": "improve_clarity",
        "finalize": "finalize"
    }
)

# 개선 후 다시 검사로
workflow.add_edge("improve_grammar", "check_grammar")
workflow.add_edge("improve_clarity", "check_clarity")
workflow.add_edge("finalize", END)

# 컴파일
quality_graph = workflow.compile()

# 테스트
test_document = """
AI 기술이 발전하고있음. 이것은 많은 분야에 영향끼침.
"""

result = quality_graph.invoke({
    "document": test_document,
    "revision_count": 0
})

print(f"최종 문서: {result['final_document']}")
print(f"수정 횟수: {result['revision_count']}")
```

### 문제 3 솔루션: 스마트 고객 지원 라우터 (Command 사용)

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

# State 정의
class SupportState(TypedDict):
    customer_query: str
    category: str
    priority: str
    assigned_department: str
    response: str

llm = ChatOpenAI(model="gpt-4.1-mini")

def categorize_query(state: SupportState) -> Command[Literal["technical_support", "billing_support", "general_support"]]:
    """문의를 분류하고 우선순위를 판단하는 노드"""
    prompt = f"""다음 고객 문의를 분석하여 JSON 형식으로 응답하세요.

    고객 문의: {state['customer_query']}

    다음 형식으로 응답하세요:
    {{
        "category": "기술지원|결제|일반",
        "priority": "긴급|보통|낮음"
    }}
    """

    analysis = llm.invoke(prompt).content

    # 간단한 파싱 (실제로는 json.loads 사용)
    if "기술지원" in analysis:
        category = "기술지원"
        next_node = "technical_support"
    elif "결제" in analysis:
        category = "결제"
        next_node = "billing_support"
    else:
        category = "일반"
        next_node = "general_support"

    if "긴급" in analysis:
        priority = "긴급"
    elif "낮음" in analysis:
        priority = "낮음"
    else:
        priority = "보통"

    return Command(
        goto=next_node,
        update={
            "category": category,
            "priority": priority,
            "assigned_department": next_node
        }
    )

def technical_support(state: SupportState) -> Command[Literal[END]]:
    """기술 지원 응답 생성"""
    priority_context = {
        "긴급": "즉시 해결하겠습니다.",
        "보통": "빠른 시일 내에 해결하겠습니다.",
        "낮음": "순차적으로 처리하겠습니다."
    }

    prompt = f"""기술 지원 담당자로서 다음 문의에 응답하세요.
    우선순위: {state['priority']}

    고객 문의: {state['customer_query']}

    응답 (친절하고 전문적으로):"""

    response = llm.invoke(prompt).content
    response = f"[기술지원팀 - {state['priority']}]\n{response}\n{priority_context.get(state['priority'], '')}"

    return Command(
        goto=END,
        update={"response": response}
    )

def billing_support(state: SupportState) -> Command[Literal[END]]:
    """결제 지원 응답 생성"""
    prompt = f"""결제 지원 담당자로서 다음 문의에 응답하세요.
    우선순위: {state['priority']}

    고객 문의: {state['customer_query']}

    응답 (친절하고 정확하게):"""

    response = llm.invoke(prompt).content
    response = f"[결제지원팀 - {state['priority']}]\n{response}"

    return Command(
        goto=END,
        update={"response": response}
    )

def general_support(state: SupportState) -> Command[Literal[END]]:
    """일반 지원 응답 생성"""
    prompt = f"""고객 지원 담당자로서 다음 문의에 응답하세요.

    고객 문의: {state['customer_query']}

    응답 (친절하고 도움이 되게):"""

    response = llm.invoke(prompt).content
    response = f"[고객지원팀 - {state['priority']}]\n{response}"

    return Command(
        goto=END,
        update={"response": response}
    )

# 그래프 구성
workflow = StateGraph(SupportState)

workflow.add_node("categorize_query", categorize_query)
workflow.add_node("technical_support", technical_support)
workflow.add_node("billing_support", billing_support)
workflow.add_node("general_support", general_support)

workflow.add_edge(START, "categorize_query")

support_graph = workflow.compile()

# 테스트
test_queries = [
    "로그인이 안 됩니다. 지금 당장 해결해주세요!",
    "이번 달 결제 내역을 확인하고 싶습니다.",
    "서비스 사용 방법을 알려주세요."
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"문의: {query}")
    result = support_graph.invoke({"customer_query": query})
    print(f"\n카테고리: {result['category']}")
    print(f"우선순위: {result['priority']}")
    print(f"배정 부서: {result['assigned_department']}")
    print(f"\n응답:\n{result['response']}")
```

### 문제 4 솔루션: 다국어 RAG 시스템 개선

```python
from typing import TypedDict, List, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# State 정의
class MultilingualRAGState(TypedDict):
    user_query: str
    detected_languages: List[str]
    primary_language: str
    search_results_ko: List[str]
    search_results_en: List[str]
    search_results_ja: List[str]
    confidence_score: float
    final_answer: str
    fallback_used: bool

# 벡터 DB 설정 (실제 환경에서는 각 언어별 DB를 미리 구축)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 실제로는 이렇게 각 언어별 DB를 로드
# db_korean = Chroma(embedding_function=embeddings, persist_directory="./chroma_ko")
# db_english = Chroma(embedding_function=embeddings, persist_directory="./chroma_en")
# db_japanese = Chroma(embedding_function=embeddings, persist_directory="./chroma_ja")

llm = ChatOpenAI(model="gpt-4.1-mini")

def analyze_query_languages(state: MultilingualRAGState) -> Command[Literal["search_primary", "search_all"]]:
    """쿼리의 언어를 분석하고 주 언어를 결정"""
    prompt = f"""다음 텍스트에 포함된 언어들을 분석하세요.

    텍스트: {state['user_query']}

    다음 형식으로 응답하세요:
    주 언어: korean|english|japanese
    포함된 언어: korean, english, japanese (해당하는 것만)
    """

    analysis = llm.invoke(prompt).content

    # 간단한 파싱
    if "korean" in analysis.lower():
        primary = "korean"
        detected = ["korean"]
    elif "japanese" in analysis.lower():
        primary = "japanese"
        detected = ["japanese"]
    else:
        primary = "english"
        detected = ["english"]

    # 혼합 언어 감지
    if "," in analysis:
        detected = [lang.strip() for lang in analysis.split("포함된 언어:")[-1].split("\n")[0].split(",")]

    # 단일 언어면 primary 검색, 혼합이면 all 검색
    next_node = "search_primary" if len(detected) == 1 else "search_all"

    return Command(
        goto=next_node,
        update={
            "detected_languages": detected,
            "primary_language": primary
        }
    )

def search_primary(state: MultilingualRAGState) -> Command[Literal["evaluate_results", "search_fallback"]]:
    """주 언어 DB에서 검색"""
    primary_lang = state['primary_language']
    query = state['user_query']

    # 실제 검색 (여기서는 mock)
    # if primary_lang == "korean":
    #     results = db_korean.similarity_search(query, k=3)
    # elif primary_lang == "japanese":
    #     results = db_japanese.similarity_search(query, k=3)
    # else:
    #     results = db_english.similarity_search(query, k=3)

    # Mock 결과
    results = [f"[{primary_lang.upper()}] Mock result {i+1} for query: {query}" for i in range(3)]

    # 결과 품질 평가
    confidence = 0.8 if len(results) >= 2 else 0.3

    # 업데이트할 필드 결정
    update_dict = {"confidence_score": confidence}
    if primary_lang == "korean":
        update_dict["search_results_ko"] = results
    elif primary_lang == "japanese":
        update_dict["search_results_ja"] = results
    else:
        update_dict["search_results_en"] = results

    next_node = "evaluate_results" if confidence > 0.5 else "search_fallback"

    return Command(
        goto=next_node,
        update=update_dict
    )

def search_fallback(state: MultilingualRAGState) -> Command[Literal["evaluate_results"]]:
    """폴백: 다른 언어 DB에서도 검색"""
    query = state['user_query']
    primary = state['primary_language']

    # 번역 및 검색
    other_langs = ["korean", "english", "japanese"]
    other_langs.remove(primary)

    update_dict = {"fallback_used": True}

    for lang in other_langs:
        # 실제로는 번역 후 검색
        translated_query = f"[TRANSLATED to {lang}] {query}"
        results = [f"[{lang.upper()}] Fallback result {i+1}" for i in range(2)]

        if lang == "korean":
            update_dict["search_results_ko"] = results
        elif lang == "japanese":
            update_dict["search_results_ja"] = results
        else:
            update_dict["search_results_en"] = results

    update_dict["confidence_score"] = 0.6

    return Command(
        goto="evaluate_results",
        update=update_dict
    )

def search_all(state: MultilingualRAGState) -> Command[Literal["evaluate_results"]]:
    """모든 언어 DB에서 검색 (혼합 언어 쿼리)"""
    query = state['user_query']

    # 각 언어 DB에서 검색
    results_ko = [f"[KO] Mixed lang result {i+1}" for i in range(2)]
    results_en = [f"[EN] Mixed lang result {i+1}" for i in range(2)]
    results_ja = [f"[JA] Mixed lang result {i+1}" for i in range(2)]

    return Command(
        goto="evaluate_results",
        update={
            "search_results_ko": results_ko,
            "search_results_en": results_en,
            "search_results_ja": results_ja,
            "confidence_score": 0.7
        }
    )

def evaluate_results(state: MultilingualRAGState) -> Command[Literal["generate_answer"]]:
    """검색 결과 평가 및 종합"""
    # 모든 검색 결과 수집
    all_results = []
    all_results.extend(state.get('search_results_ko', []))
    all_results.extend(state.get('search_results_en', []))
    all_results.extend(state.get('search_results_ja', []))

    # 신뢰도 재평가
    total_results = len([r for r in all_results if r])
    confidence = min(0.9, state['confidence_score'] + (total_results * 0.05))

    return Command(
        goto="generate_answer",
        update={"confidence_score": confidence}
    )

def generate_answer(state: MultilingualRAGState) -> Command[Literal[END]]:
    """최종 답변 생성"""
    all_results = []
    all_results.extend(state.get('search_results_ko', []))
    all_results.extend(state.get('search_results_en', []))
    all_results.extend(state.get('search_results_ja', []))

    prompt = f"""다음 검색 결과들을 바탕으로 사용자 질문에 답변하세요.

    사용자 질문: {state['user_query']}
    주 언어: {state['primary_language']}

    검색 결과:
    {chr(10).join(all_results)}

    답변 ({state['primary_language']}로 작성):"""

    answer = llm.invoke(prompt).content

    if state.get('fallback_used', False):
        answer += "\n\n(참고: 여러 언어 데이터베이스를 종합하여 답변했습니다)"

    return Command(
        goto=END,
        update={"final_answer": answer}
    )

# 그래프 구성
workflow = StateGraph(MultilingualRAGState)

workflow.add_node("analyze_query_languages", analyze_query_languages)
workflow.add_node("search_primary", search_primary)
workflow.add_node("search_fallback", search_fallback)
workflow.add_node("search_all", search_all)
workflow.add_node("evaluate_results", evaluate_results)
workflow.add_node("generate_answer", generate_answer)

workflow.add_edge(START, "analyze_query_languages")

multilingual_rag_graph = workflow.compile()

# 테스트
test_queries = [
    "테슬라의 창업자는 누구인가요?",
    "Who founded Tesla?",
    "Tesla와 Rivian을 비교해주세요",  # 혼합 언어 (영어 단어 + 한글)
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"질문: {query}")
    result = multilingual_rag_graph.invoke({"user_query": query})
    print(f"감지된 언어: {result['detected_languages']}")
    print(f"주 언어: {result['primary_language']}")
    print(f"신뢰도: {result['confidence_score']:.2f}")
    print(f"폴백 사용: {result.get('fallback_used', False)}")
    print(f"\n답변:\n{result['final_answer']}")
```

## 🚀 실무 활용 예시

### 예시 1: 고급 문서 처리 파이프라인

실무에서 자주 사용되는 문서 처리 워크플로우입니다. PDF/Word 문서를 받아 요약, 분류, 키워드 추출을 수행합니다.

```python
from typing import TypedDict, List, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

class DocumentProcessingState(TypedDict):
    document_path: str
    document_text: str
    document_type: str  # 계약서, 보고서, 제안서 등
    summary: str
    key_points: List[str]
    entities: List[dict]  # 인물, 조직, 날짜 등
    action_items: List[str]
    metadata: dict

llm = ChatOpenAI(model="gpt-4.1-mini")

def extract_text(state: DocumentProcessingState) -> Command[Literal["classify_document"]]:
    """문서에서 텍스트 추출"""
    # 실제로는 PyPDF2, python-docx 등 사용
    with open(state['document_path'], 'r', encoding='utf-8') as f:
        text = f.read()

    return Command(
        goto="classify_document",
        update={"document_text": text}
    )

def classify_document(state: DocumentProcessingState) -> Command[Literal["process_contract", "process_report", "process_proposal"]]:
    """문서 유형 분류"""
    prompt = f"""다음 문서의 유형을 분류하세요: 계약서, 보고서, 제안서, 기타

    문서 내용 (앞 500자):
    {state['document_text'][:500]}

    유형:"""

    doc_type = llm.invoke(prompt).content.strip()

    # 문서 유형별 처리 경로 결정
    if "계약서" in doc_type:
        next_node = "process_contract"
    elif "보고서" in doc_type:
        next_node = "process_report"
    else:
        next_node = "process_proposal"

    return Command(
        goto=next_node,
        update={"document_type": doc_type}
    )

def process_contract(state: DocumentProcessingState) -> Command[Literal["finalize_processing"]]:
    """계약서 특화 처리"""
    text = state['document_text']

    # 계약 조건 추출
    prompt = f"""다음 계약서에서 중요 정보를 추출하세요:
    1. 계약 당사자
    2. 계약 기간
    3. 주요 조건
    4. 금액/수량

    계약서:
    {text}

    JSON 형식으로 응답:"""

    extraction = llm.invoke(prompt).content

    # 요약 생성
    summary_prompt = f"다음 계약서를 3문장으로 요약하세요:\n{text}"
    summary = llm.invoke(summary_prompt).content

    return Command(
        goto="finalize_processing",
        update={
            "summary": summary,
            "key_points": [extraction],
            "entities": [],  # 실제로는 NER 수행
            "metadata": {"contract_terms": extraction}
        }
    )

def process_report(state: DocumentProcessingState) -> Command[Literal["finalize_processing"]]:
    """보고서 특화 처리"""
    text = state['document_text']

    # 핵심 내용 추출
    prompt = f"""다음 보고서에서:
    1. 주요 발견사항
    2. 데이터/통계
    3. 결론 및 제언

    보고서:
    {text}

    구조화된 형식으로 정리:"""

    analysis = llm.invoke(prompt).content

    # 요약
    summary = llm.invoke(f"다음 보고서를 5문장으로 요약:\n{text}").content

    return Command(
        goto="finalize_processing",
        update={
            "summary": summary,
            "key_points": [analysis],
            "action_items": []  # 실제로는 액션 아이템 추출
        }
    )

def process_proposal(state: DocumentProcessingState) -> Command[Literal["finalize_processing"]]:
    """제안서 특화 처리"""
    text = state['document_text']

    # 제안 내용 분석
    prompt = f"""다음 제안서를 분석하세요:
    1. 제안 배경
    2. 제안 내용
    3. 기대 효과
    4. 소요 예산/일정

    제안서:
    {text}

    구조화:"""

    analysis = llm.invoke(prompt).content
    summary = llm.invoke(f"제안서 요약:\n{text}").content

    return Command(
        goto="finalize_processing",
        update={
            "summary": summary,
            "key_points": [analysis]
        }
    )

def finalize_processing(state: DocumentProcessingState):
    """처리 완료 및 메타데이터 저장"""
    metadata = {
        "processed_at": "2025-10-31",
        "document_type": state['document_type'],
        "summary_length": len(state['summary']),
        "key_points_count": len(state.get('key_points', []))
    }

    return {"metadata": metadata}

# 그래프 구성
workflow = StateGraph(DocumentProcessingState)

workflow.add_node("extract_text", extract_text)
workflow.add_node("classify_document", classify_document)
workflow.add_node("process_contract", process_contract)
workflow.add_node("process_report", process_report)
workflow.add_node("process_proposal", process_proposal)
workflow.add_node("finalize_processing", finalize_processing)

workflow.add_edge(START, "extract_text")
workflow.add_edge("finalize_processing", END)

document_pipeline = workflow.compile()

# 사용 예시
result = document_pipeline.invoke({
    "document_path": "/path/to/contract.txt"
})

print(f"문서 유형: {result['document_type']}")
print(f"요약: {result['summary']}")
print(f"핵심 사항: {result['key_points']}")
```

### 예시 2: 지능형 고객 지원 시스템

실제 고객 지원 센터에서 사용할 수 있는 티켓 처리 시스템입니다.

```python
from typing import TypedDict, List, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from datetime import datetime

class SupportTicketState(TypedDict):
    ticket_id: str
    customer_id: str
    issue_description: str
    category: str
    priority: str
    sentiment: str
    escalation_required: bool
    similar_tickets: List[str]
    suggested_solutions: List[str]
    agent_notes: str
    resolution: str
    satisfaction_score: float

llm = ChatOpenAI(model="gpt-4.1-mini")

def analyze_ticket(state: SupportTicketState) -> Command[Literal["search_knowledge_base", "escalate_immediately"]]:
    """티켓 분석: 카테고리, 우선순위, 감정 분석"""
    issue = state['issue_description']

    analysis_prompt = f"""고객 문의를 분석하세요:

    문의: {issue}

    다음 항목을 분석:
    1. 카테고리: 기술/결제/계정/기타
    2. 우선순위: 긴급/높음/보통/낮음
    3. 감정: 매우불만/불만/중립/만족
    4. 즉시 에스컬레이션 필요 여부: yes/no

    JSON 형식:"""

    analysis = llm.invoke(analysis_prompt).content

    # 간단한 파싱
    category = "기술" if "기술" in analysis else "일반"
    priority = "긴급" if "긴급" in analysis else "보통"
    sentiment = "불만" if "불만" in analysis else "중립"
    escalation = "yes" in analysis.lower()

    next_node = "escalate_immediately" if escalation else "search_knowledge_base"

    return Command(
        goto=next_node,
        update={
            "category": category,
            "priority": priority,
            "sentiment": sentiment,
            "escalation_required": escalation
        }
    )

def search_knowledge_base(state: SupportTicketState) -> Command[Literal["generate_solution", "escalate_to_specialist"]]:
    """유사 사례 및 해결책 검색"""
    issue = state['issue_description']

    # 실제로는 벡터 DB 검색
    similar_tickets = [
        "티켓 #1234: 유사한 로그인 문제 - 캐시 삭제로 해결",
        "티켓 #5678: 동일 증상 - 비밀번호 재설정 필요"
    ]

    # 해결책 생성
    solution_prompt = f"""다음 유사 사례를 참고하여 해결책을 제안하세요:

    현재 문의: {issue}

    유사 사례:
    {chr(10).join(similar_tickets)}

    제안 해결책 (3개):"""

    solutions = llm.invoke(solution_prompt).content
    suggested_solutions = solutions.split("\n")[:3]

    # 신뢰도 평가
    confidence = 0.8 if len(similar_tickets) >= 2 else 0.4

    next_node = "generate_solution" if confidence > 0.6 else "escalate_to_specialist"

    return Command(
        goto=next_node,
        update={
            "similar_tickets": similar_tickets,
            "suggested_solutions": suggested_solutions
        }
    )

def generate_solution(state: SupportTicketState) -> Command[Literal["finalize_ticket"]]:
    """고객 응답 생성"""
    issue = state['issue_description']
    solutions = state['suggested_solutions']

    response_prompt = f"""친절한 고객 지원 담당자로서 응답을 작성하세요:

    고객 문의: {issue}

    제안 해결책:
    {chr(10).join(solutions)}

    고객 응답 (단계별 안내 포함):"""

    resolution = llm.invoke(response_prompt).content

    # 에이전트 노트 작성
    notes = f"[{datetime.now()}] 자동 솔루션 제공. 유사 사례 {len(state['similar_tickets'])}건 참조."

    return Command(
        goto="finalize_ticket",
        update={
            "resolution": resolution,
            "agent_notes": notes
        }
    )

def escalate_immediately(state: SupportTicketState) -> Command[Literal["finalize_ticket"]]:
    """즉시 에스컬레이션"""
    resolution = f"""[긴급 에스컬레이션]

    이 문의는 즉시 매니저에게 에스컬레이션되었습니다.
    고객: {state['customer_id']}
    이슈: {state['issue_description']}
    감정: {state['sentiment']}

    담당 매니저가 곧 연락드릴 예정입니다."""

    notes = f"[{datetime.now()}] 긴급 에스컬레이션 - 매니저 알림 발송"

    return Command(
        goto="finalize_ticket",
        update={
            "resolution": resolution,
            "agent_notes": notes
        }
    )

def escalate_to_specialist(state: SupportTicketState) -> Command[Literal["finalize_ticket"]]:
    """전문가에게 에스컬레이션"""
    resolution = f"""[전문가 에스컬레이션]

    카테고리: {state['category']}
    우선순위: {state['priority']}

    {state['category']} 전문가가 배정되었습니다.
    참고 자료: {state['similar_tickets']}

    1-2 영업일 내 상세 답변 제공 예정입니다."""

    notes = f"[{datetime.now()}] 전문가 에스컬레이션 - {state['category']} 팀에 배정"

    return Command(
        goto="finalize_ticket",
        update={
            "resolution": resolution,
            "agent_notes": notes
        }
    )

def finalize_ticket(state: SupportTicketState):
    """티켓 마무리"""
    # 만족도 예측 (실제로는 ML 모델 사용)
    satisfaction = 0.9 if not state['escalation_required'] else 0.7

    return {"satisfaction_score": satisfaction}

# 그래프 구성
workflow = StateGraph(SupportTicketState)

workflow.add_node("analyze_ticket", analyze_ticket)
workflow.add_node("search_knowledge_base", search_knowledge_base)
workflow.add_node("generate_solution", generate_solution)
workflow.add_node("escalate_immediately", escalate_immediately)
workflow.add_node("escalate_to_specialist", escalate_to_specialist)
workflow.add_node("finalize_ticket", finalize_ticket)

workflow.add_edge(START, "analyze_ticket")
workflow.add_edge("finalize_ticket", END)

support_system = workflow.compile()

# 테스트
test_tickets = [
    {
        "ticket_id": "T001",
        "customer_id": "C12345",
        "issue_description": "로그인이 계속 실패합니다. 비밀번호를 여러 번 재설정했는데도 안 됩니다."
    },
    {
        "ticket_id": "T002",
        "customer_id": "C67890",
        "issue_description": "결제가 두 번 청구되었습니다! 즉시 환불해주세요!"
    }
]

for ticket in test_tickets:
    print(f"\n{'='*60}")
    print(f"티켓 ID: {ticket['ticket_id']}")
    result = support_system.invoke(ticket)
    print(f"카테고리: {result['category']}")
    print(f"우선순위: {result['priority']}")
    print(f"감정: {result['sentiment']}")
    print(f"에스컬레이션 필요: {result['escalation_required']}")
    print(f"\n해결책:\n{result['resolution']}")
    print(f"\n에이전트 노트: {result['agent_notes']}")
    print(f"예상 만족도: {result['satisfaction_score']}")
```

## 📖 참고 자료

### 공식 문서
- [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph/)
- [LangGraph StateGraph API 레퍼런스](https://langchain-ai.github.io/langgraph/reference/graphs/)
- [LangGraph Command 객체 가이드](https://langchain-ai.github.io/langgraph/how-tos/command/)
- [LangGraph Studio 설정 가이드](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/)

### 추가 학습 자료
- [LangGraph 실전 예제 모음](https://github.com/langchain-ai/langgraph/tree/main/examples)
- [Multi-Agent 시스템 구축 가이드](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/)
- [LangGraph vs LangChain 비교](https://python.langchain.com/docs/langgraph)

### 관련 기술 스택
- **TypedDict**: Python 타입 힌팅 - [Python 공식 문서](https://docs.python.org/3/library/typing.html#typing.TypedDict)
- **Literal Types**: 타입 제약 - [PEP 586](https://peps.python.org/pep-0586/)
- **Async/Await 패턴**: 비동기 처리 - [Python Async 가이드](https://docs.python.org/3/library/asyncio.html)

### 디버깅 도구
- **LangSmith**: LangGraph 실행 추적 및 디버깅 - [LangSmith 문서](https://docs.smith.langchain.com/)
- **Mermaid 다이어그램**: 그래프 시각화 - [Mermaid 문법](https://mermaid.js.org/)

---

**다음 단계:**
- LangGraph Send 객체를 활용한 병렬 처리 학습
- Checkpointer를 사용한 상태 영속화
- 복잡한 Multi-Agent 시스템 구축
- Human-in-the-Loop 패턴 구현
