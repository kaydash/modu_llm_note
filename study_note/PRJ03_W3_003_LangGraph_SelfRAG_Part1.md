# LangGraph 활용 - Self-RAG (Part 1) - 핵심 컴포넌트

## 📚 학습 목표

이 학습 가이드를 완료하면 다음을 수행할 수 있습니다:

1. **Self-RAG 개념 이해**: 자기 성찰(Self-Reflection)을 통한 검색 증강 생성의 핵심 원리를 이해합니다
2. **검색 문서 평가**: Retrieval Grader를 구현하여 검색된 문서의 관련성을 자동으로 평가할 수 있습니다
3. **환각 감지 시스템**: Hallucination Grader를 통해 LLM 답변의 사실 근거를 검증할 수 있습니다
4. **답변 품질 평가**: Answer Grader로 생성된 답변이 질문에 적절히 대응하는지 판단할 수 있습니다
5. **질문 최적화**: Question Re-writer를 사용하여 검색에 최적화된 질문으로 변환할 수 있습니다
6. **Pydantic 구조화 출력**: LangChain의 구조화된 출력을 활용하여 평가 결과를 체계적으로 관리할 수 있습니다
7. **Self-RAG 파이프라인**: 6개 핵심 컴포넌트를 통합하여 자율적으로 품질을 개선하는 RAG 시스템을 설계할 수 있습니다

## 🔑 핵심 개념

### Self-RAG란?

**Self-RAG (Retrieval-Augmented Generation with Self-Reflection)**는 검색 증강 생성(RAG)에 자기 성찰(Self-Reflection) 메커니즘을 추가한 고급 RAG 패턴입니다.

기존 RAG의 문제점:
- 검색된 모든 문서를 무조건 사용 → 관련 없는 정보로 인한 품질 저하
- 생성된 답변의 정확성 검증 없음 → 환각(hallucination) 발생 가능
- 질문이 부적절해도 그대로 사용 → 검색 실패 시 대응 불가

Self-RAG의 해결책:
- **자동 품질 검증**: 각 단계에서 결과를 평가하고 재시도
- **적응적 처리**: 상황에 따라 검색, 재작성, 재생성 등 동적 결정
- **환각 방지**: 사실 근거가 없는 답변을 자동으로 필터링

### Self-RAG의 주요 단계

```
1. 검색 결정 (Retrieval Decision)
   ↓
2. 문서 관련성 평가 (Retrieval Grader) ← 관련 없는 문서 필터링
   ↓
3. 답변 생성 (Answer Generator)
   ↓
4. 환각 평가 (Hallucination Grader) ← 사실 근거 검증
   ↓
5. 답변 품질 평가 (Answer Grader) ← 질문 대응도 확인
   ↓
6. 질문 재작성 (Question Re-writer) ← 검색 실패 시 질문 개선
   ↓
   (다시 1번으로)
```

### 핵심 컴포넌트 (Part 1에서 다루는 내용)

#### 1. **Vector Store (벡터 저장소)**
- 문서를 임베딩 벡터로 변환하여 저장
- 의미적 유사도 기반 검색 지원
- ChromaDB를 활용한 영구 저장

#### 2. **Retrieval Grader (검색 문서 평가자)**
- 검색된 각 문서가 질문과 관련 있는지 평가
- 키워드 관련성 + 의미적 관련성 종합 판단
- 'yes'/'no' 이진 평가로 필터링

#### 3. **Answer Generator (답변 생성기)**
- 관련 문서만 사용하여 답변 생성
- 문맥 외 정보 사용 금지 (추측 배제)
- 정보 부족 시 명시적 표현

#### 4. **Hallucination Grader (환각 평가자)**
- 생성된 답변이 문서 내용에 근거하는지 확인
- 사실 기반: 'yes', 추측/환각: 'no'
- 환각 감지 시 답변 재생성 또는 거부

#### 5. **Answer Grader (답변 품질 평가자)**
- 답변이 질문에 실제로 대응하는지 평가
- 관련 정보 포함 여부 확인
- 부적절한 답변 감지 시 재시도

#### 6. **Question Re-writer (질문 재작성기)**
- 검색에 실패한 경우 질문을 개선
- 불필요한 정보 제거, 핵심 의도 명확화
- 벡터 검색에 최적화된 형태로 변환

### Pydantic 구조화 출력

Self-RAG의 모든 평가 컴포넌트는 Pydantic 모델을 사용하여 구조화된 출력을 생성합니다:

```python
from pydantic import BaseModel, Field
from typing import Literal

class GradeDocuments(BaseModel):
    """검색 문서 관련성 평가 결과"""
    binary_score: Literal['yes', 'no'] = Field(
        description="문서가 질문과 관련 있으면 'yes', 아니면 'no'"
    )

class GradeHallucinations(BaseModel):
    """환각 평가 결과"""
    binary_score: str = Field(
        description="답변이 사실에 근거하면 'yes', 아니면 'no'"
    )

class GradeAnswer(BaseModel):
    """답변 품질 평가 결과"""
    binary_score: str = Field(
        description="답변이 질문에 적절하면 'yes', 아니면 'no'"
    )
```

장점:
- **타입 안전성**: 평가 결과가 항상 예측 가능한 형식
- **검증 자동화**: Pydantic이 자동으로 데이터 유효성 검증
- **통합 용이**: LangGraph 상태 관리와 완벽 호환

## 🛠 환경 설정

### 필수 라이브러리 설치

```bash
pip install langchain langchain-openai langchain-chroma langchain-core
pip install pydantic python-dotenv openai chromadb
```

### 환경 변수 설정

`.env` 파일을 생성하고 OpenAI API 키를 설정합니다:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 기본 설정 코드

```python
from dotenv import load_dotenv
import os

# 환경 변수 로드
load_dotenv()

# 필수 라이브러리 임포트
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import Literal, List
import warnings

warnings.filterwarnings('ignore')

# 임베딩 모델 초기화
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# LLM 모델 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

## 💻 단계별 구현

### 단계 1: 벡터 저장소 준비

Self-RAG의 첫 단계는 검색할 문서를 벡터 저장소에 준비하는 것입니다.

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# 임베딩 모델 초기화
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Chroma 벡터 저장소 로드 (이미 생성된 경우)
vector_db = Chroma(
    embedding_function=embeddings_model,
    collection_name="restaurant_menu",  # 컬렉션 이름
    persist_directory="./chroma_db",    # 저장 경로
)

# 검색기 생성 (상위 3개 문서 반환)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# 검색 테스트
query = "스테이크와 어울리는 와인을 추천해주세요."
results = retriever.invoke(query)

print(f"검색된 문서 수: {len(results)}")
for i, doc in enumerate(results, 1):
    print(f"\n문서 {i}:")
    print(doc.page_content[:200])  # 첫 200자만 출력
```

**실행 결과 예시**:
```
검색된 문서 수: 3

문서 1:
와인 페어링 가이드:
- 스테이크: 풀바디 레드 와인 (카베르네 소비뇽, 말벡)
- 해산물: 화이트 와인 (샤도네이, 소비뇽 블랑)
...
```

### 단계 2: Retrieval Grader (검색 문서 관련성 평가)

검색된 문서가 실제로 질문과 관련이 있는지 평가하여 관련 없는 문서를 필터링합니다.

#### 2.1 Pydantic 모델 정의

```python
from pydantic import BaseModel, Field
from typing import Literal

class GradeDocuments(BaseModel):
    """검색된 문서의 관련성 평가 결과"""

    binary_score: Literal['yes', 'no'] = Field(
        description="문서가 질문과 관련 있으면 'yes', 아니면 'no'"
    )
```

#### 2.2 평가 시스템 프롬프트

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# LLM 모델 초기화 및 구조화된 출력 설정
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# 문서 관련성 평가를 위한 시스템 프롬프트
system_prompt = """당신은 사용자 질문에 대한 검색 결과의 관련성을 평가하는 전문가입니다.

평가 기준:
1. 키워드 관련성: 문서가 질문의 주요 단어나 유사어를 포함하는지 확인
2. 의미적 관련성: 문서의 전반적인 주제가 질문의 의도와 일치하는지 평가
3. 부분 관련성: 질문의 일부를 다루거나 맥락 정보를 제공하는 문서도 고려
4. 답변 가능성: 직접적인 답이 아니더라도 답변 형성에 도움될 정보 포함 여부 평가

점수 체계:
- 관련 있으면 'yes', 없으면 'no'로 평가
- 확실하지 않은 경우 'no'로 평가하여 불포함 쪽으로 결정

주의사항:
- 단순 단어 매칭이 아닌 질문의 전체 맥락을 고려하세요
- 완벽한 답변이 아니어도 유용한 정보가 있다면 관련 있다고 판단하세요

당신의 평가는 정보 검색 시스템 개선에 중요합니다. 균형 잡힌 평가를 해주세요."""

human_prompt = """
다음 문서가 사용자 질문과 관련이 있는지 평가하세요:

[질문]
{question}

[검색된 문서]
{document}

이 문서가 질문에 답하는 데 유용한 정보를 제공합니까?"""

# 프롬프트 템플릿 생성
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_prompt)
])

# Retrieval Grader 체인 생성
retrieval_grader = grade_prompt | structured_llm_grader
```

#### 2.3 관련성 평가 실행

```python
# 질문에 대한 문서 검색
question = "이 식당을 대표하는 메뉴는 무엇인가요?"
retrieved_docs = vector_db.similarity_search(question, k=2)

print(f"검색된 문서 수: {len(retrieved_docs)}")
print("=" * 80)
print()

relevant_docs = []

for i, doc in enumerate(retrieved_docs, 1):
    print(f"문서 {i}:", doc.page_content)
    print("-" * 80)

    # 관련성 평가
    relevance = retrieval_grader.invoke({
        "question": question,
        "document": doc.page_content
    })

    print(f"평가 결과: {relevance.binary_score}")

    if relevance.binary_score == 'yes':
        relevant_docs.append(doc)
        print("→ 이 문서는 관련 있음 (답변 생성에 사용)")
    else:
        print("→ 이 문서는 관련 없음 (제외)")

    print()

print(f"최종 선택된 관련 문서 수: {len(relevant_docs)}/{len(retrieved_docs)}")
```

**실행 결과 예시**:
```
검색된 문서 수: 2
================================================================================

문서 1: 대표 메뉴: 트러플 크림 파스타
특징: 이탈리아산 트러플과 신선한 크림 소스의 조화
가격: 28,000원
--------------------------------------------------------------------------------
평가 결과: yes
→ 이 문서는 관련 있음 (답변 생성에 사용)

문서 2: 영업 시간: 평일 11:00-22:00, 주말 10:00-23:00
주차: 발레파킹 서비스 제공
--------------------------------------------------------------------------------
평가 결과: no
→ 이 문서는 관련 없음 (제외)

최종 선택된 관련 문서 수: 1/2
```

### 단계 3: Answer Generator (답변 생성기)

관련 있는 문서만 사용하여 질문에 대한 답변을 생성합니다.

#### 3.1 답변 생성 함수 구현

```python
from langchain_core.output_parsers import StrOutputParser

def generate_answer(question: str, docs: List) -> str:
    """
    주어진 문서를 기반으로 질문에 대한 답변을 생성합니다.

    Args:
        question: 사용자 질문
        docs: 관련 문서 리스트

    Returns:
        생성된 답변 문자열
    """

    # 답변 생성 프롬프트 템플릿
    template = """당신은 주어진 문맥(context)만을 사용하여 질문에 답변하는 AI 어시스턴트입니다.

[지침]
1. 질문과 관련된 정보를 문맥에서 신중하게 확인합니다.
2. 답변에 질문과 직접 관련된 정보만 사용합니다.
3. 문맥에 명시되지 않은 내용에 대해 추측하지 않습니다.
4. 불필요한 정보를 피하고, 답변을 간결하고 명확하게 작성합니다.
5. 문맥에서 답을 찾을 수 없으면 "주어진 정보만으로는 답할 수 없습니다."라고 답변합니다.
6. 적절한 경우 문맥의 구체적인 내용을 인용합니다.

[문맥]
{context}

[질문]
{question}

[답변]"""

    # 문서 내용 결합
    context = "\n\n".join([doc.page_content for doc in docs])

    # 프롬프트 생성
    prompt = ChatPromptTemplate.from_template(template)

    # LLM 체인 생성
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm | StrOutputParser()

    # 답변 생성
    answer = chain.invoke({
        "context": context,
        "question": question
    })

    return answer
```

#### 3.2 답변 생성 실행

```python
# 질문과 관련 문서로 답변 생성
question = "이 식당을 대표하는 메뉴는 무엇인가요?"

# 앞서 필터링된 관련 문서 사용
answer = generate_answer(question, relevant_docs)

print("질문:", question)
print("\n답변:")
print(answer)
print("\n사용된 문서 수:", len(relevant_docs))
```

**실행 결과 예시**:
```
질문: 이 식당을 대표하는 메뉴는 무엇인가요?

답변:
이 식당의 대표 메뉴는 트러플 크림 파스타입니다.
이탈리아산 트러플과 신선한 크림 소스의 조화가 특징이며,
가격은 28,000원입니다.

사용된 문서 수: 1
```

### 단계 4: Hallucination Grader (환각 평가자)

생성된 답변이 실제로 문서 내용에 근거하는지 확인하여 환각(hallucination)을 감지합니다.

#### 4.1 환각 평가 모델 정의

```python
from pydantic import BaseModel, Field

class GradeHallucinations(BaseModel):
    """환각 평가 결과"""

    binary_score: str = Field(
        description="답변이 사실에 근거하면 'yes', 환각이면 'no'"
    )
```

#### 4.2 환각 평가 시스템 구현

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# LLM 모델 초기화 및 구조화된 출력 설정
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# 환각 평가를 위한 시스템 프롬프트
system_prompt = """당신은 AI가 생성한 답변이 주어진 사실에 근거하는지 평가하는 전문가입니다.

평가 기준:
1. 사실 기반: 답변의 모든 주장이 주어진 문서에서 직접 확인 가능한가?
2. 추측 여부: 문서에 없는 내용을 추론하거나 추측했는가?
3. 과장 여부: 문서의 내용을 과장하거나 왜곡했는가?
4. 일관성: 답변이 문서의 맥락과 일치하는가?

평가 규칙:
- 답변이 문서 내용만으로 구성: 'yes' (근거 있음)
- 답변에 문서에 없는 정보 포함: 'no' (환각 있음)
- 의심스러운 경우: 'no' (보수적 평가)

환각(Hallucination) 예시:
- 문서: "가격은 28,000원" → 답변: "가격은 저렴한 편" (환각)
- 문서: "트러플 크림 파스타" → 답변: "트러플 크림 파스타가 인기" (환각)
- 문서: "평일 11:00-22:00" → 답변: "평일 11:00-22:00" (사실 기반)

답변이 문서에 명확히 근거하는지 엄격하게 평가하세요."""

human_prompt = """다음 답변이 주어진 문서에 근거하는지 평가하세요:

[문서]
{documents}

[생성된 답변]
{generation}

이 답변의 모든 내용이 문서에서 직접 확인 가능합니까?"""

# 프롬프트 템플릿 생성
hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_prompt)
])

# Hallucination Grader 체인 생성
hallucination_grader = hallucination_prompt | structured_llm_grader
```

#### 4.3 환각 평가 실행

```python
def check_hallucination(generation: str, documents: List) -> dict:
    """
    생성된 답변의 환각 여부를 평가합니다.

    Args:
        generation: 생성된 답변
        documents: 참조 문서 리스트

    Returns:
        평가 결과 딕셔너리 {'score': 'yes'/'no', 'is_grounded': bool}
    """

    # 문서 내용 결합
    docs_content = "\n\n".join([doc.page_content for doc in documents])

    # 환각 평가
    result = hallucination_grader.invoke({
        "documents": docs_content,
        "generation": generation
    })

    return {
        'score': result.binary_score,
        'is_grounded': result.binary_score == 'yes'
    }

# 환각 평가 실행
hallucination_result = check_hallucination(answer, relevant_docs)

print("생성된 답변:")
print(answer)
print("\n환각 평가 결과:")
print(f"평가 점수: {hallucination_result['score']}")
print(f"사실 근거 여부: {hallucination_result['is_grounded']}")

if hallucination_result['is_grounded']:
    print("✅ 답변이 문서에 근거합니다 (환각 없음)")
else:
    print("❌ 답변에 환각이 있습니다 (문서 외 정보 포함)")
```

**실행 결과 예시**:
```
생성된 답변:
이 식당의 대표 메뉴는 트러플 크림 파스타입니다.
이탈리아산 트러플과 신선한 크림 소스의 조화가 특징이며,
가격은 28,000원입니다.

환각 평가 결과:
평가 점수: yes
사실 근거 여부: True
✅ 답변이 문서에 근거합니다 (환각 없음)
```

### 단계 5: Answer Grader (답변 품질 평가자)

생성된 답변이 실제로 질문에 적절히 대응하는지 평가합니다.

#### 5.1 답변 품질 평가 모델 정의

```python
from pydantic import BaseModel, Field

class GradeAnswer(BaseModel):
    """답변 품질 평가 결과"""

    binary_score: str = Field(
        description="답변이 질문에 적절하면 'yes', 아니면 'no'"
    )
```

#### 5.2 답변 품질 평가 시스템 구현

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# LLM 모델 초기화 및 구조화된 출력 설정
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# 답변 평가를 위한 시스템 프롬프트
system_prompt = """당신은 AI가 생성한 답변이 사용자 질문에 효과적으로 대응하는지 평가하는 전문가입니다.

평가 기준:
1. 직접적 대응: 답변이 질문에서 요구한 정보를 포함하는가?
2. 완전성: 질문의 모든 요소에 답변했는가?
3. 명확성: 답변이 질문에 대한 해답으로 명확한가?
4. 관련성: 답변 내용이 질문과 관련 있는가?

평가 규칙:
- 답변이 질문에 실질적으로 대응: 'yes'
- 답변이 질문을 회피하거나 무관: 'no'
- 부분적으로만 답변: 'no' (완전한 답변 요구)

예시:
질문: "대표 메뉴는 무엇인가요?"
답변: "트러플 크림 파스타입니다" → 'yes'
답변: "다양한 메뉴가 있습니다" → 'no' (구체적 답변 없음)
답변: "영업시간은 11시부터입니다" → 'no' (질문과 무관)

질문의 의도를 파악하고 답변의 적절성을 평가하세요."""

human_prompt = """다음 답변이 질문에 적절히 대응하는지 평가하세요:

[질문]
{question}

[생성된 답변]
{generation}

이 답변이 질문에서 요구한 정보를 효과적으로 제공합니까?"""

# 프롬프트 템플릿 생성
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_prompt)
])

# Answer Grader 체인 생성
answer_grader = answer_prompt | structured_llm_grader
```

#### 5.3 답변 품질 평가 실행

```python
def check_answer_quality(question: str, generation: str) -> dict:
    """
    생성된 답변의 품질을 평가합니다.

    Args:
        question: 원본 질문
        generation: 생성된 답변

    Returns:
        평가 결과 딕셔너리 {'score': 'yes'/'no', 'is_useful': bool}
    """

    # 답변 품질 평가
    result = answer_grader.invoke({
        "question": question,
        "generation": generation
    })

    return {
        'score': result.binary_score,
        'is_useful': result.binary_score == 'yes'
    }

# 답변 품질 평가 실행
quality_result = check_answer_quality(question, answer)

print("질문:", question)
print("\n생성된 답변:")
print(answer)
print("\n답변 품질 평가 결과:")
print(f"평가 점수: {quality_result['score']}")
print(f"유용성 여부: {quality_result['is_useful']}")

if quality_result['is_useful']:
    print("✅ 답변이 질문에 적절히 대응합니다")
else:
    print("❌ 답변이 질문에 부적절합니다 (재생성 필요)")
```

**실행 결과 예시**:
```
질문: 이 식당을 대표하는 메뉴는 무엇인가요?

생성된 답변:
이 식당의 대표 메뉴는 트러플 크림 파스타입니다.
이탈리아산 트러플과 신선한 크림 소스의 조화가 특징이며,
가격은 28,000원입니다.

답변 품질 평가 결과:
평가 점수: yes
유용성 여부: True
✅ 답변이 질문에 적절히 대응합니다
```

### 단계 6: Question Re-writer (질문 재작성기)

검색에 실패했거나 답변 품질이 낮을 때 질문을 개선하여 재검색합니다.

#### 6.1 질문 재작성 함수 구현

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def rewrite_question(question: str) -> str:
    """
    주어진 질문을 벡터 저장소 검색에 최적화된 형태로 다시 작성합니다.

    Args:
        question: 원본 질문 문자열

    Returns:
        다시 작성된 질문 문자열
    """

    # LLM 모델 초기화
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 시스템 프롬프트 정의
    system_prompt = """당신은 질문을 벡터 저장소 검색에 최적화된 형태로 개선하는 전문가입니다.

개선 원칙:
1. 명확성: 질문의 핵심 의도를 명확히 표현
2. 간결성: 불필요한 정보를 제거하고 핵심만 남김
3. 검색 친화성: 검색 키워드를 명확히 포함
4. 의미 보존: 원래 질문의 의도를 유지

개선 방법:
- 구어체 → 문어체 변환
- 모호한 표현 → 구체적 표현
- 복합 질문 → 단일 초점 질문
- 맥락 정보 추가 (필요 시)

예시:
원본: "거기 뭐 괜찮은 거 있어?"
개선: "추천 메뉴는 무엇인가요?"

원본: "와인이랑 잘 어울리는 음식 있나요? 가격도 알려주세요."
개선: "와인 페어링에 적합한 메뉴와 가격을 알려주세요."

원본 질문의 의도를 정확히 파악하고 검색에 최적화하세요."""

    human_prompt = """다음 질문을 벡터 저장소 검색에 최적화된 형태로 개선하세요:

[원본 질문]
{question}

[개선된 질문]"""

    # 프롬프트 템플릿 생성
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])

    # 체인 생성
    chain = rewrite_prompt | llm | StrOutputParser()

    # 질문 재작성 실행
    rewritten = chain.invoke({"question": question})

    return rewritten
```

#### 6.2 질문 재작성 실행 및 재검색

```python
# 원본 질문
original_question = "거기 대표적인 거 뭐야?"

print("원본 질문:", original_question)

# 질문 재작성
improved_question = rewrite_question(original_question)
print("개선된 질문:", improved_question)
print()

# 개선된 질문으로 재검색
print("=== 재검색 실행 ===")
new_docs = vector_db.similarity_search(improved_question, k=3)

print(f"검색된 문서 수: {len(new_docs)}")
for i, doc in enumerate(new_docs, 1):
    print(f"\n문서 {i}:")
    print(doc.page_content[:150])
```

**실행 결과 예시**:
```
원본 질문: 거기 대표적인 거 뭐야?
개선된 질문: 이 식당의 대표 메뉴는 무엇인가요?

=== 재검색 실행 ===
검색된 문서 수: 3

문서 1:
대표 메뉴: 트러플 크림 파스타
특징: 이탈리아산 트러플과 신선한 크림 소스의 조화
가격: 28,000원

문서 2:
시그니처 메뉴:
1. 트러플 크림 파스타
2. 블랙 앵거스 스테이크
3. 랍스터 리조또

문서 3:
메뉴 설명:
트러플 크림 파스타는 이 식당의 대표 메뉴로...
```

### 단계 7: 통합 Self-RAG 파이프라인

6개 컴포넌트를 통합하여 완전한 Self-RAG 파이프라인을 구성합니다.

```python
def self_rag_pipeline(question: str, max_retries: int = 2) -> dict:
    """
    Self-RAG 전체 파이프라인을 실행합니다.

    Args:
        question: 사용자 질문
        max_retries: 최대 재시도 횟수

    Returns:
        결과 딕셔너리 {
            'answer': 최종 답변,
            'is_grounded': 사실 근거 여부,
            'is_useful': 유용성 여부,
            'retries': 재시도 횟수,
            'documents_used': 사용된 문서 수
        }
    """

    current_question = question
    retry_count = 0

    while retry_count <= max_retries:
        print(f"\n{'='*80}")
        print(f"시도 #{retry_count + 1}: {current_question}")
        print('='*80)

        # 1. 문서 검색
        print("\n[1단계] 문서 검색")
        retrieved_docs = vector_db.similarity_search(current_question, k=3)
        print(f"검색된 문서: {len(retrieved_docs)}개")

        # 2. 관련성 평가 (Retrieval Grader)
        print("\n[2단계] 문서 관련성 평가")
        relevant_docs = []
        for doc in retrieved_docs:
            relevance = retrieval_grader.invoke({
                "question": current_question,
                "document": doc.page_content
            })
            if relevance.binary_score == 'yes':
                relevant_docs.append(doc)

        print(f"관련 문서: {len(relevant_docs)}/{len(retrieved_docs)}개")

        # 관련 문서가 없으면 질문 재작성
        if not relevant_docs:
            print("→ 관련 문서 없음, 질문 재작성")
            current_question = rewrite_question(current_question)
            retry_count += 1
            continue

        # 3. 답변 생성 (Answer Generator)
        print("\n[3단계] 답변 생성")
        answer = generate_answer(current_question, relevant_docs)
        print(f"답변: {answer[:100]}...")

        # 4. 환각 평가 (Hallucination Grader)
        print("\n[4단계] 환각 평가")
        hallucination_result = check_hallucination(answer, relevant_docs)
        print(f"사실 근거: {hallucination_result['is_grounded']}")

        if not hallucination_result['is_grounded']:
            print("→ 환각 감지, 재시도")
            retry_count += 1
            continue

        # 5. 답변 품질 평가 (Answer Grader)
        print("\n[5단계] 답변 품질 평가")
        quality_result = check_answer_quality(question, answer)
        print(f"답변 유용성: {quality_result['is_useful']}")

        if not quality_result['is_useful']:
            print("→ 답변 부적절, 질문 재작성")
            current_question = rewrite_question(current_question)
            retry_count += 1
            continue

        # 모든 평가 통과
        print("\n✅ Self-RAG 파이프라인 성공")
        return {
            'answer': answer,
            'is_grounded': True,
            'is_useful': True,
            'retries': retry_count,
            'documents_used': len(relevant_docs)
        }

    # 최대 재시도 초과
    print("\n❌ 최대 재시도 횟수 초과")
    return {
        'answer': "죄송합니다. 적절한 답변을 생성할 수 없습니다.",
        'is_grounded': False,
        'is_useful': False,
        'retries': retry_count,
        'documents_used': 0
    }

# Self-RAG 파이프라인 실행
result = self_rag_pipeline("이 식당의 대표 메뉴는 무엇인가요?")

print("\n" + "="*80)
print("최종 결과")
print("="*80)
print(f"답변: {result['answer']}")
print(f"사실 근거: {result['is_grounded']}")
print(f"유용성: {result['is_useful']}")
print(f"재시도 횟수: {result['retries']}")
print(f"사용된 문서 수: {result['documents_used']}")
```

**실행 결과 예시**:
```
================================================================================
시도 #1: 이 식당의 대표 메뉴는 무엇인가요?
================================================================================

[1단계] 문서 검색
검색된 문서: 3개

[2단계] 문서 관련성 평가
관련 문서: 2/3개

[3단계] 답변 생성
답변: 이 식당의 대표 메뉴는 트러플 크림 파스타입니다...

[4단계] 환각 평가
사실 근거: True

[5단계] 답변 품질 평가
답변 유용성: True

✅ Self-RAG 파이프라인 성공

================================================================================
최종 결과
================================================================================
답변: 이 식당의 대표 메뉴는 트러플 크림 파스타입니다. 이탈리아산 트러플과 신선한 크림 소스의 조화가 특징이며, 가격은 28,000원입니다.
사실 근거: True
유용성: True
재시도 횟수: 0
사용된 문서 수: 2
```

## 🎯 실습 문제

### 실습 1: Retrieval Grader 성능 개선

**문제**: 현재 Retrieval Grader가 너무 엄격하여 유용한 문서도 'no'로 평가하는 경우가 있습니다. 평가 기준을 개선하세요.

**요구사항**:
1. 시스템 프롬프트에 "부분 관련성"에 대한 명확한 기준 추가
2. 평가 예시를 3개 이상 포함
3. 개선된 grader로 같은 질문에 대해 평가 비교

**힌트**:
```python
improved_system_prompt = """
당신은 검색 결과의 관련성을 평가하는 전문가입니다.

평가 기준:
1. 직접 관련성 (High): 질문에 직접 답변 가능 → 'yes'
2. 부분 관련성 (Medium): 맥락 정보 제공 → ???
3. 무관 (Low): 질문과 전혀 무관 → 'no'

... (계속 작성) ...
"""
```

### 실습 2: 답변 품질 다차원 평가

**문제**: 현재 Answer Grader는 'yes'/'no' 이진 평가만 합니다. 답변을 여러 차원에서 평가하는 시스템을 구현하세요.

**요구사항**:
1. Pydantic 모델로 다음 평가 항목 포함:
   - `relevance`: 관련성 (0-10)
   - `completeness`: 완전성 (0-10)
   - `clarity`: 명확성 (0-10)
   - `overall_score`: 종합 점수
   - `feedback`: 개선 제안 (문자열)

2. 종합 점수가 7점 미만이면 재생성

**힌트**:
```python
from pydantic import BaseModel, Field
from typing import Literal

class DetailedAnswerGrade(BaseModel):
    relevance: int = Field(ge=0, le=10, description="질문 관련성")
    completeness: int = Field(ge=0, le=10, description="답변 완전성")
    clarity: int = Field(ge=0, le=10, description="표현 명확성")
    overall_score: float = Field(description="종합 점수")
    feedback: str = Field(description="개선 제안")
```

### 실습 3: 질문 재작성 전략 개선

**문제**: 검색 실패 원인에 따라 다른 재작성 전략을 적용하는 시스템을 구현하세요.

**요구사항**:
1. 실패 원인 분류:
   - `too_vague`: 질문이 너무 모호함
   - `too_specific`: 질문이 너무 구체적임
   - `wrong_context`: 잘못된 맥락 정보

2. 각 원인에 맞는 재작성 전략 구현

3. 재작성 전후 비교 출력

**힌트**:
```python
def analyze_failure_reason(question: str, retrieved_docs: List) -> str:
    """검색 실패 원인 분석"""
    # 여기에 구현
    pass

def rewrite_by_strategy(question: str, reason: str) -> str:
    """원인별 재작성 전략 적용"""
    strategies = {
        'too_vague': "질문을 더 구체적으로 작성...",
        'too_specific': "질문을 더 일반적으로 작성...",
        'wrong_context': "맥락 정보를 수정..."
    }
    # 여기에 구현
    pass
```

### 실습 4: Self-RAG 성능 모니터링 대시보드

**문제**: Self-RAG 파이프라인의 성능을 추적하는 모니터링 시스템을 구현하세요.

**요구사항**:
1. 다음 메트릭 수집:
   - 평균 재시도 횟수
   - 각 단계별 성공률
   - 평균 응답 시간
   - 문서 관련성 비율

2. 결과를 시각화 (텍스트 기반)

**힌트**:
```python
class SelfRAGMonitor:
    def __init__(self):
        self.metrics = {
            'total_queries': 0,
            'successful_queries': 0,
            'avg_retries': 0,
            'stage_success_rates': {},
            'avg_response_time': 0
        }

    def track_query(self, result: dict, response_time: float):
        # 여기에 구현
        pass

    def print_dashboard(self):
        # 여기에 구현
        pass
```

## ✅ 솔루션 예시

### 솔루션 1: Retrieval Grader 성능 개선

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal

class ImprovedGradeDocuments(BaseModel):
    """개선된 문서 관련성 평가 결과"""
    binary_score: Literal['yes', 'no'] = Field(
        description="문서 관련성: 'yes' (관련 있음) 또는 'no' (관련 없음)"
    )
    relevance_level: Literal['high', 'medium', 'low'] = Field(
        description="관련성 수준: high (직접 관련), medium (부분 관련), low (무관)"
    )
    reason: str = Field(
        description="평가 이유 설명"
    )

# 개선된 시스템 프롬프트
improved_system_prompt = """당신은 검색 결과의 관련성을 평가하는 전문가입니다.

평가 기준 (3단계):

1. 직접 관련성 (High):
   - 문서가 질문에 직접 답변할 수 있는 정보를 포함
   - 핵심 키워드가 명확히 일치
   - 평가: 'yes', 관련성: 'high'

2. 부분 관련성 (Medium):
   - 질문의 일부 요소만 다루거나 맥락 정보 제공
   - 간접적으로 답변 형성에 도움
   - 유사 키워드 또는 관련 주제 포함
   - 평가: 'yes' (유용한 정보이므로 포함), 관련성: 'medium'

3. 무관 (Low):
   - 질문과 전혀 관련 없는 내용
   - 답변 형성에 도움 안 됨
   - 평가: 'no', 관련성: 'low'

평가 예시:

예시 1:
질문: "대표 메뉴는 무엇인가요?"
문서: "대표 메뉴: 트러플 크림 파스타"
→ binary_score: 'yes', relevance_level: 'high', reason: "질문에 직접 답변"

예시 2:
질문: "대표 메뉴는 무엇인가요?"
문서: "트러플 크림 파스타는 이탈리아산 트러플을 사용합니다"
→ binary_score: 'yes', relevance_level: 'medium', reason: "대표 메뉴의 상세 정보 제공"

예시 3:
질문: "대표 메뉴는 무엇인가요?"
문서: "영업 시간은 평일 11시부터입니다"
→ binary_score: 'no', relevance_level: 'low', reason: "질문과 무관한 정보"

예시 4:
질문: "스테이크와 어울리는 와인은?"
문서: "레드 와인은 육류와 잘 어울립니다"
→ binary_score: 'yes', relevance_level: 'medium', reason: "일반적인 페어링 원칙 제공"

원칙:
- 의심스러우면 'medium'으로 평가하여 포함 (나중에 Answer Generator가 선별)
- 완전히 무관한 경우만 'no'
- 부분 정보도 유용하다면 'yes'"""

human_prompt = """다음 문서가 질문과 관련 있는지 평가하세요:

[질문]
{question}

[검색된 문서]
{document}

관련성을 평가하고 이유를 설명하세요."""

# 개선된 Retrieval Grader 생성
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
improved_retrieval_grader = (
    ChatPromptTemplate.from_messages([
        ("system", improved_system_prompt),
        ("human", human_prompt)
    ])
    | llm.with_structured_output(ImprovedGradeDocuments)
)

# 비교 테스트
test_question = "이 식당의 분위기는 어떤가요?"
test_docs = [
    "트러플 크림 파스타는 대표 메뉴입니다.",  # 부분 관련
    "조용하고 아늑한 분위기로 데이트에 적합합니다.",  # 직접 관련
    "영업 시간은 평일 11시부터 22시까지입니다."  # 무관
]

print("=== 기존 vs 개선된 Retrieval Grader 비교 ===\n")
for i, doc_content in enumerate(test_docs, 1):
    print(f"문서 {i}: {doc_content}\n")

    # 기존 grader (Part 1에서 구현한 것)
    original_result = retrieval_grader.invoke({
        "question": test_question,
        "document": doc_content
    })

    # 개선된 grader
    improved_result = improved_retrieval_grader.invoke({
        "question": test_question,
        "document": doc_content
    })

    print(f"기존 평가: {original_result.binary_score}")
    print(f"개선 평가: {improved_result.binary_score} (수준: {improved_result.relevance_level})")
    print(f"평가 이유: {improved_result.reason}")
    print("-" * 80)
    print()
```

### 솔루션 2: 답변 품질 다차원 평가

```python
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

class DetailedAnswerGrade(BaseModel):
    """다차원 답변 품질 평가 결과"""

    relevance: int = Field(
        ge=0, le=10,
        description="질문 관련성 (0: 무관, 10: 완벽히 관련)"
    )

    completeness: int = Field(
        ge=0, le=10,
        description="답변 완전성 (0: 불완전, 10: 모든 요소 답변)"
    )

    clarity: int = Field(
        ge=0, le=10,
        description="표현 명확성 (0: 모호함, 10: 매우 명확)"
    )

    overall_score: float = Field(
        description="종합 점수 (3개 항목 평균)"
    )

    feedback: str = Field(
        description="개선 제안 사항"
    )

# 다차원 평가 시스템 프롬프트
detailed_grading_prompt = """당신은 AI 답변의 품질을 다차원으로 평가하는 전문가입니다.

평가 항목 (각 0-10점):

1. 관련성 (Relevance):
   - 10점: 질문의 모든 요소에 직접 대응
   - 7-9점: 질문의 핵심에 대응하나 일부 누락
   - 4-6점: 부분적으로만 관련
   - 0-3점: 질문과 거의 무관

2. 완전성 (Completeness):
   - 10점: 질문의 모든 측면을 포괄적으로 답변
   - 7-9점: 주요 내용은 포함하나 세부 사항 일부 누락
   - 4-6점: 기본 정보만 제공
   - 0-3점: 대부분의 정보 누락

3. 명확성 (Clarity):
   - 10점: 매우 명확하고 이해하기 쉬움
   - 7-9점: 대체로 명확하나 일부 모호한 표현
   - 4-6점: 이해 가능하나 불명확한 부분 존재
   - 0-3점: 모호하고 이해하기 어려움

종합 점수:
- (관련성 + 완전성 + 명확성) / 3
- 7점 이상: 우수
- 5-6점: 보통 (개선 필요)
- 5점 미만: 부족 (재생성 필요)

Feedback:
- 구체적인 개선 사항 제시
- 부족한 항목 명시
- 개선 방향 제안"""

human_detailed_prompt = """다음 답변을 평가하세요:

[질문]
{question}

[생성된 답변]
{generation}

각 항목을 0-10점으로 평가하고 개선 제안을 제공하세요."""

# 다차원 평가 grader 생성
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
detailed_answer_grader = (
    ChatPromptTemplate.from_messages([
        ("system", detailed_grading_prompt),
        ("human", human_detailed_prompt)
    ])
    | llm.with_structured_output(DetailedAnswerGrade)
)

def evaluate_answer_detailed(question: str, answer: str) -> dict:
    """답변을 다차원으로 평가하고 재생성 필요 여부 판단"""

    # 평가 실행
    evaluation = detailed_answer_grader.invoke({
        "question": question,
        "generation": answer
    })

    # 재생성 필요 여부 (종합 점수 7점 미만)
    needs_regeneration = evaluation.overall_score < 7.0

    return {
        'evaluation': evaluation,
        'needs_regeneration': needs_regeneration,
        'quality_level': 'excellent' if evaluation.overall_score >= 8
                        else 'good' if evaluation.overall_score >= 7
                        else 'acceptable' if evaluation.overall_score >= 5
                        else 'poor'
    }

# 테스트: 다양한 품질의 답변 평가
test_cases = [
    {
        'question': "이 식당의 대표 메뉴와 가격을 알려주세요.",
        'answer': "대표 메뉴는 트러플 크림 파스타이며 가격은 28,000원입니다."
    },
    {
        'question': "이 식당의 대표 메뉴와 가격을 알려주세요.",
        'answer': "트러플 크림 파스타가 있습니다."  # 불완전
    },
    {
        'question': "이 식당의 대표 메뉴와 가격을 알려주세요.",
        'answer': "영업 시간은 11시부터입니다."  # 무관
    }
]

print("=== 다차원 답변 품질 평가 ===\n")
for i, case in enumerate(test_cases, 1):
    print(f"테스트 케이스 {i}")
    print(f"질문: {case['question']}")
    print(f"답변: {case['answer']}\n")

    result = evaluate_answer_detailed(case['question'], case['answer'])
    eval_data = result['evaluation']

    print(f"관련성: {eval_data.relevance}/10")
    print(f"완전성: {eval_data.completeness}/10")
    print(f"명확성: {eval_data.clarity}/10")
    print(f"종합 점수: {eval_data.overall_score:.1f}/10")
    print(f"품질 수준: {result['quality_level']}")
    print(f"재생성 필요: {'예' if result['needs_regeneration'] else '아니오'}")
    print(f"\n개선 제안:\n{eval_data.feedback}")
    print("=" * 80)
    print()
```

### 솔루션 3: 질문 재작성 전략 개선

```python
from typing import List, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

class FailureAnalysis(BaseModel):
    """검색 실패 원인 분석 결과"""

    reason: Literal['too_vague', 'too_specific', 'wrong_context', 'no_issue'] = Field(
        description="실패 원인 분류"
    )

    explanation: str = Field(
        description="원인 설명"
    )

    confidence: float = Field(
        ge=0.0, le=1.0,
        description="분석 신뢰도"
    )

def analyze_failure_reason(
    question: str,
    retrieved_docs: List,
    relevant_count: int
) -> FailureAnalysis:
    """검색 실패 원인을 분석합니다"""

    # 원인 분석 프롬프트
    analysis_prompt = """당신은 검색 실패 원인을 분석하는 전문가입니다.

원인 분류:

1. too_vague (너무 모호함):
   - 질문이 추상적이거나 맥락 없음
   - 예: "그거 뭐야?", "추천해줘"
   - 해결: 구체적인 대상과 의도 명시

2. too_specific (너무 구체적):
   - 질문이 지나치게 세부적이어서 매칭 실패
   - 예: "2023년 3월 15일 저녁 7시 메뉴"
   - 해결: 일반화하여 범위 확대

3. wrong_context (잘못된 맥락):
   - 질문의 맥락이나 용어가 문서와 불일치
   - 예: 문서에 "파스타"인데 "면 요리" 검색
   - 해결: 용어와 맥락 정렬

4. no_issue (문제 없음):
   - 질문은 적절하나 문서에 정보가 없음
   - 해결: 답변 불가 명시

검색 결과:
- 검색된 문서: {total_docs}개
- 관련 문서: {relevant_docs}개

질문: {question}

문서 샘플:
{doc_samples}

원인을 분석하고 신뢰도를 평가하세요."""

    # 문서 샘플 추출
    doc_samples = "\n".join([
        f"- {doc.page_content[:100]}"
        for doc in retrieved_docs[:3]
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    analyzer = (
        ChatPromptTemplate.from_template(analysis_prompt)
        | llm.with_structured_output(FailureAnalysis)
    )

    result = analyzer.invoke({
        "question": question,
        "total_docs": len(retrieved_docs),
        "relevant_docs": relevant_count,
        "doc_samples": doc_samples
    })

    return result

def rewrite_by_strategy(question: str, reason: str) -> str:
    """원인별 맞춤 재작성 전략 적용"""

    strategies = {
        'too_vague': """질문이 너무 모호합니다. 다음을 명확히 하세요:
- 구체적인 대상 명시
- 의도와 목적 추가
- 맥락 정보 포함

예시:
모호: "추천해줘"
명확: "처음 방문하는데 이 식당의 대표 메뉴를 추천해주세요"

모호: "가격이 어때?"
명확: "대표 메뉴의 가격대는 어느 정도인가요?"
""",

        'too_specific': """질문이 너무 구체적입니다. 다음과 같이 일반화하세요:
- 날짜/시간 등 과도한 제약 제거
- 핵심 정보 위주로 간소화
- 검색 범위 확대

예시:
구체적: "2024년 3월 15일 저녁 7시 예약 가능한 창가 자리 메뉴"
일반화: "저녁 메뉴와 예약 가능 여부"

구체적: "이탈리아산 트러플을 사용한 크림 파스타의 정확한 레시피"
일반화: "트러플 크림 파스타의 주요 특징"
""",

        'wrong_context': """질문의 맥락이 문서와 맞지 않습니다. 다음을 조정하세요:
- 용어를 문서 표현에 맞게 변경
- 맥락 정보 재정렬
- 동의어 활용

예시:
불일치: "면 요리 있나요?" (문서에는 "파스타")
정렬: "파스타 메뉴가 있나요?"

불일치: "저렴한 음식" (문서에는 구체적 가격)
정렬: "가격대가 합리적인 메뉴"
"""
    }

    # 전략 선택
    strategy_prompt = strategies.get(reason, "질문을 검색에 최적화하세요")

    # 재작성 실행
    rewrite_prompt = f"""{strategy_prompt}

원본 질문: {{question}}

위 전략에 따라 질문을 개선하세요. 개선된 질문만 출력하세요."""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = ChatPromptTemplate.from_template(rewrite_prompt) | llm | StrOutputParser()

    improved = chain.invoke({"question": question})

    return improved

# 통합 테스트
def test_adaptive_rewriting():
    """원인별 재작성 전략 테스트"""

    test_cases = [
        {
            'question': "추천해줘",
            'expected_reason': 'too_vague'
        },
        {
            'question': "2024년 3월 15일 저녁 7시에 예약 가능한 창가 자리의 특별 메뉴",
            'expected_reason': 'too_specific'
        },
        {
            'question': "면 요리 중에 저렴한 것",
            'expected_reason': 'wrong_context'
        }
    ]

    print("=== 적응적 질문 재작성 전략 테스트 ===\n")

    for i, case in enumerate(test_cases, 1):
        print(f"테스트 케이스 {i}")
        print(f"원본 질문: {case['question']}")
        print(f"예상 원인: {case['expected_reason']}\n")

        # 문서 검색
        retrieved = vector_db.similarity_search(case['question'], k=3)

        # 관련성 평가
        relevant_count = sum(
            1 for doc in retrieved
            if retrieval_grader.invoke({
                "question": case['question'],
                "document": doc.page_content
            }).binary_score == 'yes'
        )

        # 원인 분석
        analysis = analyze_failure_reason(case['question'], retrieved, relevant_count)

        print(f"분석된 원인: {analysis.reason}")
        print(f"설명: {analysis.explanation}")
        print(f"신뢰도: {analysis.confidence:.2f}\n")

        # 원인별 재작성
        improved = rewrite_by_strategy(case['question'], analysis.reason)

        print(f"개선된 질문: {improved}")
        print("=" * 80)
        print()

# 실행
test_adaptive_rewriting()
```

### 솔루션 4: Self-RAG 성능 모니터링

```python
import time
from typing import Dict, List
from datetime import datetime

class SelfRAGMonitor:
    """Self-RAG 파이프라인 성능 모니터링 시스템"""

    def __init__(self):
        self.metrics = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'total_retries': 0,
            'total_response_time': 0.0,

            # 단계별 성공률
            'retrieval_success': 0,
            'hallucination_pass': 0,
            'answer_quality_pass': 0,

            # 문서 관련성
            'total_retrieved_docs': 0,
            'total_relevant_docs': 0,

            # 재작성 통계
            'questions_rewritten': 0,

            # 쿼리 히스토리
            'query_history': []
        }

    def track_query(
        self,
        question: str,
        result: dict,
        response_time: float,
        stage_results: dict
    ):
        """쿼리 실행 결과를 추적"""

        self.metrics['total_queries'] += 1
        self.metrics['total_response_time'] += response_time

        # 성공/실패 분류
        if result['is_useful'] and result['is_grounded']:
            self.metrics['successful_queries'] += 1
        else:
            self.metrics['failed_queries'] += 1

        # 재시도 횟수
        self.metrics['total_retries'] += result.get('retries', 0)

        # 단계별 통계
        if stage_results.get('retrieval_found_relevant'):
            self.metrics['retrieval_success'] += 1

        if stage_results.get('no_hallucination'):
            self.metrics['hallucination_pass'] += 1

        if stage_results.get('answer_quality_ok'):
            self.metrics['answer_quality_pass'] += 1

        # 문서 통계
        self.metrics['total_retrieved_docs'] += stage_results.get('docs_retrieved', 0)
        self.metrics['total_relevant_docs'] += stage_results.get('docs_relevant', 0)

        # 재작성 통계
        if stage_results.get('was_rewritten'):
            self.metrics['questions_rewritten'] += 1

        # 히스토리 저장
        self.metrics['query_history'].append({
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'success': result['is_useful'] and result['is_grounded'],
            'retries': result.get('retries', 0),
            'response_time': response_time
        })

    def get_statistics(self) -> dict:
        """통계 계산"""
        total = self.metrics['total_queries']

        if total == 0:
            return {
                'success_rate': 0.0,
                'avg_retries': 0.0,
                'avg_response_time': 0.0,
                'doc_relevance_rate': 0.0
            }

        return {
            'success_rate': self.metrics['successful_queries'] / total,
            'avg_retries': self.metrics['total_retries'] / total,
            'avg_response_time': self.metrics['total_response_time'] / total,
            'doc_relevance_rate': (
                self.metrics['total_relevant_docs'] / self.metrics['total_retrieved_docs']
                if self.metrics['total_retrieved_docs'] > 0 else 0.0
            ),
            'retrieval_success_rate': self.metrics['retrieval_success'] / total,
            'hallucination_pass_rate': self.metrics['hallucination_pass'] / total,
            'answer_quality_pass_rate': self.metrics['answer_quality_pass'] / total,
            'rewrite_rate': self.metrics['questions_rewritten'] / total
        }

    def print_dashboard(self):
        """모니터링 대시보드 출력"""
        stats = self.get_statistics()
        total = self.metrics['total_queries']

        print("\n" + "=" * 80)
        print("📊 Self-RAG 성능 모니터링 대시보드")
        print("=" * 80)

        # 전체 통계
        print("\n🔢 전체 통계:")
        print(f"  총 쿼리 수: {total}")
        print(f"  성공: {self.metrics['successful_queries']} ({stats['success_rate']*100:.1f}%)")
        print(f"  실패: {self.metrics['failed_queries']} ({(1-stats['success_rate'])*100:.1f}%)")

        # 성능 지표
        print("\n⚡ 성능 지표:")
        print(f"  평균 응답 시간: {stats['avg_response_time']:.2f}초")
        print(f"  평균 재시도 횟수: {stats['avg_retries']:.2f}회")
        print(f"  질문 재작성 비율: {stats['rewrite_rate']*100:.1f}%")

        # 단계별 성공률
        print("\n✅ 단계별 성공률:")
        print(f"  검색 성공: {stats['retrieval_success_rate']*100:.1f}%")
        print(f"  환각 없음: {stats['hallucination_pass_rate']*100:.1f}%")
        print(f"  답변 품질 양호: {stats['answer_quality_pass_rate']*100:.1f}%")

        # 문서 관련성
        print("\n📄 문서 관련성:")
        print(f"  총 검색 문서: {self.metrics['total_retrieved_docs']}개")
        print(f"  관련 문서: {self.metrics['total_relevant_docs']}개")
        print(f"  관련성 비율: {stats['doc_relevance_rate']*100:.1f}%")

        # 최근 쿼리 히스토리 (최근 5개)
        print("\n📝 최근 쿼리 히스토리 (최근 5개):")
        for query in self.metrics['query_history'][-5:]:
            status = "✅" if query['success'] else "❌"
            print(f"  {status} {query['question'][:50]}...")
            print(f"     재시도: {query['retries']}회, 응답시간: {query['response_time']:.2f}초")

        print("\n" + "=" * 80)

# 모니터링이 통합된 Self-RAG 파이프라인
def monitored_self_rag_pipeline(
    question: str,
    monitor: SelfRAGMonitor,
    max_retries: int = 2
) -> dict:
    """모니터링이 통합된 Self-RAG 파이프라인"""

    start_time = time.time()

    # 단계별 결과 추적
    stage_results = {
        'retrieval_found_relevant': False,
        'no_hallucination': False,
        'answer_quality_ok': False,
        'docs_retrieved': 0,
        'docs_relevant': 0,
        'was_rewritten': False
    }

    current_question = question
    retry_count = 0

    while retry_count <= max_retries:
        # 1. 문서 검색
        retrieved_docs = vector_db.similarity_search(current_question, k=3)
        stage_results['docs_retrieved'] += len(retrieved_docs)

        # 2. 관련성 평가
        relevant_docs = []
        for doc in retrieved_docs:
            relevance = retrieval_grader.invoke({
                "question": current_question,
                "document": doc.page_content
            })
            if relevance.binary_score == 'yes':
                relevant_docs.append(doc)

        stage_results['docs_relevant'] += len(relevant_docs)

        if not relevant_docs:
            current_question = rewrite_question(current_question)
            stage_results['was_rewritten'] = True
            retry_count += 1
            continue

        stage_results['retrieval_found_relevant'] = True

        # 3. 답변 생성
        answer = generate_answer(current_question, relevant_docs)

        # 4. 환각 평가
        hallucination_result = check_hallucination(answer, relevant_docs)
        if not hallucination_result['is_grounded']:
            retry_count += 1
            continue

        stage_results['no_hallucination'] = True

        # 5. 답변 품질 평가
        quality_result = check_answer_quality(question, answer)
        if not quality_result['is_useful']:
            current_question = rewrite_question(current_question)
            stage_results['was_rewritten'] = True
            retry_count += 1
            continue

        stage_results['answer_quality_ok'] = True

        # 성공
        result = {
            'answer': answer,
            'is_grounded': True,
            'is_useful': True,
            'retries': retry_count,
            'documents_used': len(relevant_docs)
        }

        response_time = time.time() - start_time
        monitor.track_query(question, result, response_time, stage_results)

        return result

    # 실패
    result = {
        'answer': "적절한 답변을 생성할 수 없습니다.",
        'is_grounded': False,
        'is_useful': False,
        'retries': retry_count,
        'documents_used': 0
    }

    response_time = time.time() - start_time
    monitor.track_query(question, result, response_time, stage_results)

    return result

# 테스트: 여러 쿼리로 모니터링
monitor = SelfRAGMonitor()

test_questions = [
    "이 식당의 대표 메뉴는 무엇인가요?",
    "스테이크와 어울리는 와인을 추천해주세요.",
    "가격대가 합리적인 메뉴가 있나요?",
    "영업 시간은 언제인가요?",
    "뭐 맛있는 거 있어?"  # 모호한 질문
]

print("=== Self-RAG 파이프라인 실행 ===\n")
for question in test_questions:
    print(f"질문: {question}")
    result = monitored_self_rag_pipeline(question, monitor)
    print(f"답변: {result['answer'][:100]}...")
    print()

# 대시보드 출력
monitor.print_dashboard()
```

## 🚀 실무 활용 예시

### 예시 1: 고객 지원 챗봇 (환각 방지)

고객 문의에 정확한 정보만 제공하는 챗봇 시스템:

```python
class CustomerSupportBot:
    """환각 방지가 적용된 고객 지원 챗봇"""

    def __init__(self, knowledge_base_path: str):
        # 지식 베이스 로드
        self.vector_db = Chroma(
            embedding_function=OpenAIEmbeddings(),
            collection_name="customer_support_kb",
            persist_directory=knowledge_base_path
        )

        # Self-RAG 컴포넌트 초기화
        self.setup_components()

    def setup_components(self):
        """Self-RAG 컴포넌트 초기화"""
        # Retrieval Grader, Hallucination Grader 등 설정
        # (Part 1에서 구현한 컴포넌트 사용)
        pass

    def answer_customer_query(self, query: str) -> dict:
        """
        고객 문의에 답변 (환각 방지 보장)

        Returns:
            {
                'answer': 답변 텍스트,
                'confidence': 신뢰도 (0-1),
                'sources': 참조 문서 리스트,
                'has_hallucination': 환각 감지 여부
            }
        """

        # Self-RAG 파이프라인 실행
        result = self_rag_pipeline(query, max_retries=2)

        # 환각이 감지되면 명시적으로 표시
        if not result['is_grounded']:
            return {
                'answer': "죄송합니다. 확실한 정보를 찾을 수 없습니다. 담당자에게 연결하시겠습니까?",
                'confidence': 0.0,
                'sources': [],
                'has_hallucination': True
            }

        return {
            'answer': result['answer'],
            'confidence': 0.9 if result['is_useful'] else 0.5,
            'sources': result.get('documents_used', 0),
            'has_hallucination': False
        }

# 사용 예시
bot = CustomerSupportBot("./customer_kb")
response = bot.answer_customer_query("환불 정책이 어떻게 되나요?")

print(f"답변: {response['answer']}")
print(f"신뢰도: {response['confidence']}")
print(f"참조 문서: {response['sources']}개")
```

### 예시 2: 법률 문서 QA 시스템 (정확성 중시)

법률 문서에서 정확한 정보만 추출하는 시스템:

```python
class LegalDocumentQA:
    """법률 문서 질의응답 시스템 (Self-RAG 기반)"""

    def __init__(self):
        self.vector_db = self.load_legal_documents()
        self.strict_mode = True  # 엄격 모드: 환각 0% 허용

    def load_legal_documents(self):
        """법률 문서 로드"""
        # 법률 문서를 벡터 DB에 저장
        pass

    def query_legal_info(self, question: str) -> dict:
        """
        법률 정보 조회 (정확성 최우선)

        - 환각이 조금이라도 있으면 답변 거부
        - 인용문 필수 포함
        - 출처 명시
        """

        # Self-RAG 실행
        result = self_rag_pipeline(question, max_retries=3)

        # 엄격 모드: 환각 있으면 답변 거부
        if self.strict_mode and not result['is_grounded']:
            return {
                'answer': None,
                'error': "확실한 법률 근거를 찾을 수 없습니다.",
                'recommendation': "법률 전문가와 상담하시기 바랍니다."
            }

        # 출처와 인용문 추가
        return {
            'answer': result['answer'],
            'sources': self.extract_sources(result),
            'citations': self.extract_citations(result),
            'confidence': 'high' if result['is_grounded'] and result['is_useful'] else 'low'
        }

    def extract_sources(self, result: dict) -> List[str]:
        """답변의 출처 추출"""
        # 문서 메타데이터에서 출처 정보 추출
        pass

    def extract_citations(self, result: dict) -> List[str]:
        """답변의 인용문 추출"""
        # 답변에 사용된 원문 발췌
        pass

# 사용 예시
legal_qa = LegalDocumentQA()
response = legal_qa.query_legal_info("근로기준법상 최저임금 산정 기준은?")

if response['answer']:
    print(f"답변: {response['answer']}")
    print(f"\n출처: {', '.join(response['sources'])}")
    print(f"\n인용:")
    for citation in response['citations']:
        print(f"  - {citation}")
else:
    print(f"오류: {response['error']}")
    print(f"권장사항: {response['recommendation']}")
```

### 예시 3: 의료 정보 검색 시스템

의료 정보의 정확성과 안전성을 보장하는 시스템:

```python
class MedicalInfoSearch:
    """의료 정보 검색 시스템 (Self-RAG + 추가 안전장치)"""

    def __init__(self):
        self.vector_db = self.load_medical_database()
        self.safety_checker = self.init_safety_checker()

    def search_medical_info(self, query: str) -> dict:
        """
        의료 정보 검색 (안전성 최우선)

        추가 안전장치:
        - 환각 감지
        - 위험 키워드 필터링
        - 전문가 확인 권장
        """

        # 위험 키워드 체크
        risk_keywords = ['암', '수술', '투약', '진단']
        is_high_risk = any(keyword in query for keyword in risk_keywords)

        # Self-RAG 실행
        result = self_rag_pipeline(query, max_retries=3)

        # 응답 구성
        response = {
            'answer': result['answer'] if result['is_grounded'] else None,
            'is_reliable': result['is_grounded'] and result['is_useful'],
            'risk_level': 'high' if is_high_risk else 'low',
            'disclaimer': "이 정보는 참고용이며 전문의 상담을 대체할 수 없습니다."
        }

        # 고위험 질문은 전문가 상담 권장
        if is_high_risk:
            response['warning'] = "⚠️ 의료 전문가와 상담하시기 바랍니다."

        return response

    def init_safety_checker(self):
        """안전성 검사기 초기화"""
        # 의료 정보 안전성 검증 로직
        pass

    def load_medical_database(self):
        """의료 정보 데이터베이스 로드"""
        # 검증된 의료 정보만 로드
        pass

# 사용 예시
medical_search = MedicalInfoSearch()

# 일반 질문
response1 = medical_search.search_medical_info("두통에 좋은 일반적인 방법은?")
print(f"답변: {response1['answer']}")
print(f"신뢰도: {response1['is_reliable']}")

# 고위험 질문
response2 = medical_search.search_medical_info("암 진단 후 치료 방법은?")
if response2.get('warning'):
    print(f"⚠️ {response2['warning']}")
print(f"면책조항: {response2['disclaimer']}")
```

## 📖 참고 자료

### 공식 문서

1. **LangChain 공식 문서**
   - [Structured Output](https://python.langchain.com/docs/modules/model_io/output_parsers/structured)
   - [RAG Patterns](https://python.langchain.com/docs/use_cases/question_answering/)

2. **Pydantic 공식 문서**
   - [Pydantic v2](https://docs.pydantic.dev/latest/)
   - [Field Validators](https://docs.pydantic.dev/latest/concepts/validators/)

3. **OpenAI API 문서**
   - [Function Calling](https://platform.openai.com/docs/guides/function-calling)
   - [Embeddings](https://platform.openai.com/docs/guides/embeddings)

### 학술 논문

1. **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection**
   - Akari Asai et al., 2023
   - [arXiv:2310.11511](https://arxiv.org/abs/2310.11511)

2. **RAFT: Adapting Language Model to Domain Specific RAG**
   - Tianjun Zhang et al., 2024
   - [arXiv:2403.10131](https://arxiv.org/abs/2403.10131)

### 추가 학습 자료

1. **LangChain RAG Tutorials**
   - [Advanced RAG Patterns](https://python.langchain.com/docs/use_cases/question_answering/quickstart)

2. **Self-RAG 구현 예제**
   - [LangGraph Self-RAG Example](https://github.com/langchain-ai/langgraph/tree/main/examples)

3. **환각 감지 기법**
   - [Hallucination Detection in LLMs](https://arxiv.org/abs/2305.14251)

### 다음 학습

- **Part 2**: LangGraph로 Self-RAG 그래프 구현 및 통합 실행
- Self-RAG 워크플로우 시각화
- 상태 기반 조건부 라우팅
- 실전 프로젝트: 법률 문서 QA 시스템

---

**Part 1 완료** ✅

Part 2에서는 LangGraph를 사용하여 6개 컴포넌트를 통합한 완전한 Self-RAG 그래프를 구현하고 실전 프로젝트를 진행합니다.
