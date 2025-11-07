# LangGraph 활용 - Corrective RAG (CRAG) Part1 - 기본 개념 및 구성 요소

## 📚 학습 목표

이 학습 가이드를 완료하면 다음을 할 수 있습니다:

1. **Corrective RAG (CRAG)의 개념과 동작 원리를 이해**하고 기존 RAG와의 차이점을 설명할 수 있습니다
2. **다단계 검색 전략**을 활용한 적응형 벡터 저장소를 구현할 수 있습니다
3. **Retrieval Grader**를 이용하여 검색된 문서의 관련성을 3단계로 평가할 수 있습니다
4. **Knowledge Refiner**를 통해 문서에서 핵심 지식을 추출하고 정제할 수 있습니다
5. **Question Re-writer**를 활용하여 검색에 최적화된 질문으로 재작성할 수 있습니다
6. Pydantic을 사용한 **구조화된 LLM 출력**을 설계하고 활용할 수 있습니다
7. CRAG 시스템의 각 구성 요소를 독립적으로 테스트하고 검증할 수 있습니다

## 🔑 핵심 개념

### Corrective RAG (CRAG)란?

**Corrective RAG**는 검색 증강 생성(RAG) 시스템의 품질을 개선하기 위한 고급 기법입니다. 기존 RAG가 검색된 문서를 그대로 사용하는 것과 달리, CRAG는 검색 결과를 **평가하고 정제**하는 추가 단계를 거칩니다.

#### 기존 RAG vs CRAG

| 특징 | 기존 RAG | Corrective RAG (CRAG) |
|------|----------|----------------------|
| **검색 전략** | 단일 검색 | 다단계 적응형 검색 |
| **문서 평가** | 없음 | 3단계 관련성 평가 (correct/incorrect/ambiguous) |
| **지식 정제** | 전체 문서 사용 | 관련 지식 조각만 추출 |
| **질문 개선** | 없음 | 검색 실패 시 질문 재작성 |
| **외부 지식** | 내부 DB만 | 필요 시 웹 검색 추가 |

### CRAG의 핵심 프로세스

```
질문 입력
   ↓
[1단계] 문서 검색 (다단계 검색 전략)
   ↓
[2단계] 문서 관련성 평가 (Retrieval Grader)
   ↓ ← correct/incorrect/ambiguous
[3단계] 지식 정제 (Knowledge Refiner)
   ↓ ← 관련 지식 조각만 추출
[4단계] 충분한 정보?
   ├─ Yes → 답변 생성
   └─ No → 질문 재작성 + 웹 검색 → [2단계]로
```

### 주요 구성 요소

#### 1. **Adaptive Vector Store** (적응형 벡터 저장소)
- **다단계 검색 전략**: 정밀 검색 → 확장 검색 → 포괄 검색
- **점수 임계값 조정**: 0.3 → 0.1 → 0.0으로 점진적 완화
- **검색 개수 증가**: 3개 → 5개 → 10개로 점진적 확대

#### 2. **Retrieval Grader** (문서 관련성 평가자)
- **3단계 평가**: correct (명확히 관련), incorrect (명확히 무관), ambiguous (모호함)
- **평가 기준**: 키워드 관련성, 의미적 관련성, 부분 관련성, 답변 가능성
- **Pydantic 모델**: 구조화된 평가 결과 반환

#### 3. **Knowledge Refiner** (지식 정제자)
- **지식 조각 추출**: 문서를 핵심 정보로 분할
- **Binary 평가**: yes (관련 있음) / no (관련 없음)
- **정제된 지식**: 질문과 직접 관련된 정보만 유지

#### 4. **Question Re-writer** (질문 재작성자)
- **검색 최적화**: 모호한 질문을 명확하고 검색 친화적으로 변환
- **개선 전략**: 핵심 키워드 추출, 구체적 용어 대체, 동의어 추가
- **예시**: "그거 얼마야?" → "스테이크 메뉴 가격 정보"

#### 5. **Web Search** (웹 검색)
- **외부 지식 보강**: 내부 DB에 정보가 부족할 때 활용
- **Tavily Search API**: 고품질 웹 검색 결과 제공

### 기술 스택

- **LangChain**: RAG 파이프라인 구축
- **LangGraph**: 복잡한 워크플로우 관리 (Part 2에서 다룸)
- **ChromaDB**: 벡터 저장소
- **OpenAI**: 임베딩 및 LLM
- **Pydantic**: 구조화된 출력 정의
- **Tavily**: 웹 검색 API

## 🛠 환경 설정

### 필수 라이브러리 설치

```bash
# 핵심 라이브러리
pip install langchain langchain-community langchain-openai langchain-chroma

# LangGraph (Part 2에서 사용)
pip install langgraph

# 웹 검색
pip install tavily-python langchain-tavily

# 유틸리티
pip install python-dotenv pydantic

# PDF 로딩 (실습용)
pip install pypdf
```

### 환경 변수 설정

`.env` 파일을 프로젝트 루트에 생성:

```bash
# OpenAI API 키
OPENAI_API_KEY=your_openai_api_key_here

# Tavily Search API 키
TAVILY_API_KEY=your_tavily_api_key_here
```

### 기본 설정 코드

```python
# 환경 변수 로딩
from dotenv import load_dotenv
load_dotenv()

# 기본 라이브러리
import os
import warnings
import logging
from datetime import datetime
import operator
from typing import TypedDict, Union, List, Dict, Tuple, Any, Annotated
from langchain_core.documents import Document

# 경고 무시
warnings.filterwarnings("ignore")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
```

## 💻 단계별 구현

### 단계 1: 적응형 벡터 저장소 구현

적응형 벡터 저장소는 검색 결과가 부족할 때 자동으로 검색 전략을 확장하는 지능형 검색 시스템입니다.

#### 1.1 AdaptiveVectorStore 클래스 정의

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

class AdaptiveVectorStore:
    """적응형 검색을 위한 벡터 저장소"""

    def __init__(self, collection_name: str, persist_dir: str = "./chroma_db"):
        # 임베딩 모델 초기화
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
        )

        # Chroma DB 초기화
        self.vector_db = Chroma(
            embedding_function=self.embeddings,
            collection_name=collection_name,
            persist_directory=persist_dir,
        )

        # 다단계 검색 전략 정의
        self.search_configs = {
            "initial": {"k": 3, "score_threshold": 0.3},      # 1차: 정밀 검색
            "expanded": {"k": 5, "score_threshold": 0.1},     # 2차: 확장 검색
            "exhaustive": {"k": 10, "score_threshold": 0.0}   # 3차: 포괄 검색
        }

        logger.info(f"✅ Vector store '{collection_name}' initialized")

    def multi_stage_search(self, query: str):
        """단계별 검색 전략"""
        # 1차: 정밀 검색 (높은 임계값)
        initial = self.vector_db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs=self.search_configs["initial"]
        ).invoke(query)

        logger.info(f"🔍 1차 검색 모드: {len(initial)}개의 문서 검색")

        if len(initial) < 1:
            # 2차: 확장 검색 (중간 임계값)
            expanded = self.vector_db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs=self.search_configs["expanded"]
            ).invoke(query)

            logger.info(f"🔍 2차 검색 모드: {len(expanded)}개의 문서 검색")

            if len(expanded) < 1:
                # 3차: 포괄 검색 (낮은 임계값)
                exhaustive = self.vector_db.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs=self.search_configs["exhaustive"]
                ).invoke(query)

                logger.info(f"🔍 3차 검색 모드: {len(exhaustive)}개의 문서 검색")
                return exhaustive

            return expanded
        else:
            return initial
```

#### 1.2 벡터 저장소 사용 예시

```python
# 벡터 저장소 초기화
vector_db = AdaptiveVectorStore(collection_name="restaurant_menu")

# 검색 테스트
results = vector_db.multi_stage_search("채식주의자를 위한 메뉴가 있나요?")

print(f"검색 결과: {len(results)}개 문서")
for i, doc in enumerate(results, 1):
    print(f"\n[문서 {i}]")
    print(doc.page_content[:200])  # 처음 200자만 출력
```

**실행 결과 예시**:
```
✅ Vector store 'restaurant_menu' initialized
🔍 1차 검색 모드: 0개의 문서 검색
🔍 2차 검색 모드: 0개의 문서 검색
🔍 3차 검색 모드: 0개의 문서 검색
검색 결과: 0개 문서
```

### 단계 2: 웹 검색 도구 설정

내부 DB에 정보가 부족할 때 외부 지식을 활용하기 위한 웹 검색 기능을 구현합니다.

#### 2.1 Tavily Search 설정

```python
from langchain_tavily import TavilySearch

# 웹 검색 도구 초기화 (최대 5개 결과 반환)
search_tool = TavilySearch(max_results=5)

# 웹 검색 테스트
search_results = search_tool.invoke("스테이크와 어울리는 와인")

print(f"검색된 결과: {len(search_results['results'])}개")
for i, result in enumerate(search_results['results'], 1):
    print(f"\n[결과 {i}]")
    print(f"제목: {result['title']}")
    print(f"URL: {result['url']}")
    print(f"내용: {result['content'][:150]}...")
```

**실행 결과 예시**:
```
검색된 결과: 5개

[결과 1]
제목: 스테이크와 찰떡궁합! 스테이크에 어울리는 와인 5가지
URL: https://alcohol.hobby-tech.com/entry/...
내용: 카베르네 소비뇽 (Cabernet Sauvignon) – 스테이크 와인의 대표주자
      어울리는 스테이크: 리브아이 스테이크, 티본 스테이크...
```

### 단계 3: Retrieval Grader (문서 관련성 평가자) 구현

검색된 문서가 질문과 관련이 있는지 3단계로 평가하는 시스템을 구현합니다.

#### 3.1 Pydantic 모델 정의

```python
from pydantic import BaseModel, Field
from typing import Literal

class GradeDocuments(BaseModel):
    """문서 관련성 평가 결과를 위한 데이터 모델"""
    relevance_score: Literal["correct", "incorrect", "ambiguous"] = Field(
        description="문서 관련성: 'correct' (명확히 관련), 'incorrect' (명확히 무관), 'ambiguous' (모호함)"
    )
```

#### 3.2 Retrieval Grader 파이프라인 구성

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# LLM 모델 초기화 및 구조화된 출력 설정
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# 문서 관련성 평가를 위한 시스템 프롬프트
system_prompt = """
You are an expert evaluator tasked with assessing the relevance of retrieved documents to a user's question.
Your role is crucial in enhancing the quality of information retrieval systems.

[평가 기준]
1. 키워드 관련성: 문서가 질문의 주요 단어나 유사어를 포함하는지 확인
2. 의미적 관련성: 문서의 전반적인 주제가 질문의 의도와 일치하는지 평가
3. 부분 관련성: 질문의 일부를 다루거나 맥락 정보를 제공하는 문서도 고려
4. 답변 가능성: 직접적인 답이 아니더라도 답변 형성에 도움될 정보 포함 여부 평가

[점수 체계]
- 'correct': 문서가 명확히 관련 있고, 질문에 답하는 데 필요한 정보를 포함함.
- 'incorrect': 문서가 명확히 무관하거나, 질문에 도움이 되지 않는 정보를 포함함.
- 'ambiguous': 문서의 관련성이 불분명하거나, 일부 관련 정보는 있지만 유용성이 확실하지 않음,
               혹은 질문과 약간만 관련 있음.

[주의사항]
- 단순 단어 매칭이 아닌 질문의 전체 맥락을 고려하세요
- 완벽한 답변이 아니어도 유용한 정보가 있다면 관련 있다고 판단하세요

Your evaluation plays a critical role in improving the overall performance of the information retrieval system.
Strive for balanced and thoughtful assessments.
"""

# 채점 프롬프트 템플릿 생성
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Document: \n\n {document} \n\n Question: {question}"),
])

# Retrieval Grader 파이프라인
retrieval_grader = grade_prompt | structured_llm_grader
```

#### 3.3 Retrieval Grader 사용 예시

```python
# 테스트 질문 및 문서
question = "해산물 요리를 추천해주세요."
retrieved_docs = vector_db.multi_stage_search(question)

print(f"검색된 문서 수: {len(retrieved_docs)}\n")

for i, doc in enumerate(retrieved_docs, 1):
    print(f"[문서 {i}]")
    print(f"내용: {doc.page_content[:100]}...")

    # 관련성 평가
    relevance = retrieval_grader.invoke({
        "question": question,
        "document": doc.page_content
    })

    print(f"관련성 평가: {relevance.relevance_score}")
    print("="*60)
```

**실행 결과 예시**:
```
검색된 문서 수: 3

[문서 1]
내용: 메뉴: 갈릭 버터 새우 파스타
설명: 신선한 새우와 마늘, 버터의 조화...
관련성 평가: correct
============================================================

[문서 2]
내용: 메뉴: 스테이크 세트
설명: 프리미엄 소고기 스테이크...
관련성 평가: incorrect
============================================================
```

### 단계 4: Answer Generator (답변 생성자) 구현

검색된 문서를 기반으로 질문에 대한 답변을 생성하는 시스템을 구현합니다.

#### 4.1 Answer Generator 함수 정의

```python
from langchain_core.output_parsers import StrOutputParser

def generator_answer(question, docs):
    """문서를 기반으로 질문에 대한 답변을 생성"""

    template = """당신은 정확하고 도움되는 답변을 제공하는 AI 어시스턴트입니다.

    [지침]
    1. 제공된 문맥만을 사용하여 답변
    2. 불확실한 경우 명확히 표시
    3. 간결하되 완전한 답변 제공

    [문맥]
    {context}

    [질문]
    {question}

    [답변]"""

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model='gpt-4.1-mini', temperature=0)

    def format_docs(docs):
        """문서 리스트를 하나의 문자열로 포맷팅"""
        return "\n\n".join([d.page_content for d in docs])

    # RAG 체인 구성
    rag_chain = prompt | llm | StrOutputParser()

    # 답변 생성
    generation = rag_chain.invoke({
        "context": format_docs(docs),
        "question": question
    })

    return generation
```

#### 4.2 Answer Generator 사용 예시

```python
question = "해산물 요리를 추천해주세요."
retrieved_docs = vector_db.multi_stage_search(question)

# 답변 생성
generation = generator_answer(question, docs=retrieved_docs)

print(f"질문: {question}")
print(f"\n답변:\n{generation}")
```

**실행 결과 예시**:
```
질문: 해산물 요리를 추천해주세요.

답변:
제공된 문맥에 해산물 요리에 대한 정보가 없어 추천이 어렵습니다.
```

### 단계 5: Question Re-writer (질문 재작성자) 구현

검색 결과가 부족할 때 질문을 개선하여 더 나은 검색 결과를 얻을 수 있도록 질문을 재작성합니다.

#### 5.1 Question Re-writer 함수 정의

```python
def rewrite_question(question: str) -> str:
    """
    주어진 질문을 벡터 저장소 검색에 최적화된 형태로 다시 작성합니다.
    """
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    system_prompt = """당신은 검색 최적화 전문가입니다.

    질문 개선 전략:
    1. 핵심 키워드 추출 및 강조
    2. 모호한 대명사를 구체적 용어로 대체
    3. 동의어 및 관련 용어 추가
    4. 시간적/공간적 맥락 명확화
    5. 복합 질문을 단순 질문으로 분해

    예시:
    - 원본: "그거 얼마야?"
    - 개선: "스테이크 메뉴 가격 정보"

    - 원본: "여기서 뭐가 제일 맛있어?"
    - 개선: "레스토랑 인기 메뉴 추천 시그니처 요리"

    주의사항:
    - 원래 의도 유지
    - 과도한 확장 지양
    - 검색 친화적 표현 사용"""

    re_write_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """[원본 질문]
{question}

검색에 최적화된 질문으로 재작성하세요.
간결하고 명확하게, 핵심 키워드를 포함하여."""),
    ])

    question_rewriter = re_write_prompt | llm | StrOutputParser()
    rewritten_question = question_rewriter.invoke({"question": question})

    return rewritten_question
```

#### 5.2 Question Re-writer 사용 예시

```python
# 다양한 질문 테스트
test_questions = [
    "해산물 요리를 추천해주세요.",
    "그거 얼마야?",
    "여기서 뭐가 제일 맛있어?",
    "채식주의자 메뉴 있나요?"
]

for original_q in test_questions:
    rewritten_q = rewrite_question(original_q)
    print(f"원본: {original_q}")
    print(f"재작성: {rewritten_q}")
    print("-" * 60)
```

**실행 결과 예시**:
```
원본: 해산물 요리를 추천해주세요.
재작성: 해산물 요리 추천 인기 메뉴 베스트
------------------------------------------------------------
원본: 그거 얼마야?
재작성: 메뉴 가격 정보 비용
------------------------------------------------------------
원본: 여기서 뭐가 제일 맛있어?
재작성: 레스토랑 인기 메뉴 추천 시그니처 베스트
------------------------------------------------------------
원본: 채식주의자 메뉴 있나요?
재작성: 채식주의자 메뉴 비건 베지테리안 옵션
------------------------------------------------------------
```

### 단계 6: Knowledge Refiner (지식 정제자) 구현

검색된 문서에서 질문과 관련된 핵심 지식만 추출하고 정제하는 시스템을 구현합니다.

#### 6.1 Pydantic 모델 정의

```python
class RefinedKnowledge(BaseModel):
    """
    문서에서 추출된 정제된 지식 조각을 나타냅니다.
    """
    knowledge_strip: str = Field(description="문서에서 추출된 정제된 지식 조각")
    binary_score: str = Field(
        description="문서가 질문과 관련이 있는지 여부, 'yes' 또는 'no'"
    )
```

#### 6.2 Knowledge Refiner 파이프라인 구성

```python
# 구조화된 출력을 위한 LLM 설정
structured_llm_refiner = llm.with_structured_output(RefinedKnowledge)

# 지식 정제를 위한 프롬프트
refine_system_prompt = """
당신은 지식 정제 전문가입니다. 주어진 질문과 관련하여 문서에서 핵심 정보를 추출하고
관련성을 평가하는 것이 당신의 임무입니다.

[지시사항]
1. 질문과 문서를 주의 깊게 읽으세요.
2. 질문에 답하는 데 관련이 있는 문서의 핵심 정보들을 식별하세요.
3. 각 핵심 정보에 대해:
   a. 간결하게 추출하고 요약하세요 (정보당 1-2문장을 목표로 함).
   b. 질문과의 관련성을 'yes' (관련 있음) 또는 'no' (관련 없음)로 평가하세요.
4. 각 정보를 다음 형식으로 새 줄에 제시하세요:
   [추출된 정보] (yes/no)

[예시 출력]
AI 시스템은 훈련 데이터에 존재하는 편향을 나타낼 수 있습니다. (yes)
의사결정에서 AI 사용은 개인정보 보호 우려를 제기합니다. (yes)
기계학습 모델은 상당한 컴퓨팅 자원을 필요로 합니다. (no)

[참고사항]
사실적이고 객관적인 정보 추출에 집중하세요. 개인적인 의견이나 추측은 피하세요.
3-5개의 핵심 정보 제공을 목표로 하되, 문서에 관련 내용이 특히 풍부한 경우 더 많이 포함해도 됩니다.
"""

refine_prompt = ChatPromptTemplate.from_messages([
    ("system", refine_system_prompt),
    ("human", "[문서]\n{document}\n\n[사용자 질문]\n{question}"),
])

# Knowledge Refiner 파이프라인
knowledge_refiner = refine_prompt | structured_llm_refiner
```

#### 6.3 Knowledge Refiner 사용 예시

```python
question = "해산물 요리를 추천해주세요."
retrieved_docs = vector_db.multi_stage_search(question)

print(f"검색된 문서 수: {len(retrieved_docs)}\n")

refined_knowledge_list = []

for i, doc in enumerate(retrieved_docs, 1):
    print(f"[문서 {i}]")
    print(f"원본 내용: {doc.page_content[:150]}...")

    # 지식 정제
    refined = knowledge_refiner.invoke({
        "question": question,
        "document": doc.page_content
    })

    print(f"\n정제된 지식: {refined.knowledge_strip}")
    print(f"관련성 평가: {refined.binary_score}")

    # yes인 경우만 수집
    if refined.binary_score == "yes":
        refined_knowledge_list.append(refined.knowledge_strip)

    print("="*60)

print(f"\n✅ 최종 수집된 지식: {len(refined_knowledge_list)}개")
for i, knowledge in enumerate(refined_knowledge_list, 1):
    print(f"{i}. {knowledge}")
```

**실행 결과 예시**:
```
검색된 문서 수: 3

[문서 1]
원본 내용: 메뉴: 갈릭 버터 새우 파스타
설명: 신선한 새우와 마늘, 버터가 어우러진 인기 파스타 메뉴입니다.
      해산물을 좋아하시는 분들께 추천드립니다...

정제된 지식: 갈릭 버터 새우 파스타는 신선한 새우와 마늘, 버터의 조화로 만든 인기 메뉴입니다.
관련성 평가: yes
============================================================

[문서 2]
원본 내용: 메뉴: 스테이크 세트
설명: 프리미엄 소고기 스테이크를 최상의 상태로 제공합니다...

정제된 지식: 스테이크 세트는 프리미엄 소고기를 사용한 메뉴입니다.
관련성 평가: no
============================================================

✅ 최종 수집된 지식: 1개
1. 갈릭 버터 새우 파스타는 신선한 새우와 마늘, 버터의 조화로 만든 인기 메뉴입니다.
```

## 🎯 실습 문제

### 문제 1: Retrieval Grader 커스터마이징 (⭐)

**과제**: Retrieval Grader를 재구성하여 평가 기준을 더 세밀하게 조정하세요.

**요구사항**:
1. 4단계 평가 시스템으로 확장: `excellent`, `good`, `poor`, `irrelevant`
2. 각 등급에 대한 명확한 기준 정의
3. 평가 결과에 신뢰도 점수(0-1) 추가

**힌트**:
```python
class DetailedGradeDocuments(BaseModel):
    relevance_score: Literal["excellent", "good", "poor", "irrelevant"]
    confidence: float = Field(ge=0, le=1, description="평가 신뢰도")
    reasoning: str = Field(description="평가 근거")
```

### 문제 2: Multi-Domain Vector Store (⭐⭐)

**과제**: 여러 도메인의 문서를 관리할 수 있는 다중 벡터 저장소 시스템을 구현하세요.

**요구사항**:
1. 3개 이상의 도메인별 컬렉션 생성 (예: 메뉴, 와인, 리뷰)
2. 질문 분석을 통한 자동 도메인 라우팅
3. 여러 도메인에서 검색 후 결과 통합

**힌트**:
```python
class MultiDomainVectorStore:
    def __init__(self, domains: Dict[str, str]):
        self.stores = {}
        for domain, collection_name in domains.items():
            self.stores[domain] = AdaptiveVectorStore(collection_name)

    def route_and_search(self, query: str):
        # 질문 분석 → 도메인 선택 → 검색 → 통합
        pass
```

### 문제 3: Cascading Search Strategy (⭐⭐⭐)

**과제**: 검색 실패 시 자동으로 다음 전략을 시도하는 캐스케이딩 검색 시스템을 구현하세요.

**요구사항**:
1. 5단계 검색 전략 정의 (매우 정밀 → 매우 포괄)
2. 각 단계에서 최소 문서 개수 설정
3. 검색 이력 추적 및 통계 제공

**시나리오**:
```
Level 1: threshold=0.5, k=2  → 실패 (0개 검색)
Level 2: threshold=0.4, k=3  → 실패 (0개 검색)
Level 3: threshold=0.3, k=5  → 성공 (3개 검색)
```

## ✅ 솔루션 예시

### 솔루션 1: Retrieval Grader 커스터마이징

```python
from pydantic import BaseModel, Field, field_validator
from typing import Literal

class DetailedGradeDocuments(BaseModel):
    """세밀한 4단계 문서 관련성 평가"""
    relevance_score: Literal["excellent", "good", "poor", "irrelevant"] = Field(
        description="""
        - excellent: 질문에 직접적이고 완전한 답변 포함
        - good: 질문과 명확히 관련되어 있으나 부분적 정보
        - poor: 질문과 약간 관련되어 있으나 유용성 낮음
        - irrelevant: 질문과 전혀 무관
        """
    )
    confidence: float = Field(
        ge=0, le=1,
        description="평가 신뢰도 (0: 매우 불확실, 1: 매우 확실)"
    )
    reasoning: str = Field(
        description="평가 근거 (왜 이 등급을 부여했는지 설명)"
    )

    @field_validator('confidence')
    def validate_confidence(cls, v):
        """신뢰도 값 검증"""
        if not 0 <= v <= 1:
            raise ValueError("신뢰도는 0과 1 사이여야 합니다")
        return v

# LLM 설정
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
detailed_grader = llm.with_structured_output(DetailedGradeDocuments)

# 프롬프트
detailed_system_prompt = """
당신은 문서 관련성을 4단계로 세밀하게 평가하는 전문가입니다.

[평가 등급]
1. excellent (탁월):
   - 질문에 직접적이고 완전한 답변 포함
   - 추가 정보가 거의 불필요
   - 예: "스테이크 가격은?" → "스테이크 가격: 35,000원"

2. good (좋음):
   - 질문과 명확히 관련
   - 유용한 정보 포함하나 부분적
   - 예: "스테이크 가격은?" → "스테이크 메뉴는 다양하며..."

3. poor (부족):
   - 질문과 약간 관련
   - 간접적 정보만 제공
   - 예: "스테이크 가격은?" → "레스토랑 영업시간은..."

4. irrelevant (무관):
   - 질문과 전혀 무관
   - 도움이 되지 않음
   - 예: "스테이크 가격은?" → "날씨 정보..."

[평가 지침]
- 질문의 의도를 정확히 파악
- 문서의 정보 완전성 평가
- 평가 근거를 명확히 제시
- 신뢰도는 평가의 확신 정도를 반영
"""

detailed_grade_prompt = ChatPromptTemplate.from_messages([
    ("system", detailed_system_prompt),
    ("human", "Document: \n\n {document} \n\n Question: {question}"),
])

# 파이프라인
detailed_retrieval_grader = detailed_grade_prompt | detailed_grader

# 테스트
test_doc = """
메뉴: 프리미엄 립아이 스테이크
가격: 45,000원
설명: 최상급 한우 립아이를 사용한 시그니처 메뉴입니다.
     부드러운 육질과 풍부한 마블링이 특징입니다.
추천: 미디엄 레어로 즐기시는 것을 추천드립니다.
"""

question = "스테이크 메뉴의 가격은 얼마인가요?"

result = detailed_retrieval_grader.invoke({
    "question": question,
    "document": test_doc
})

print(f"질문: {question}\n")
print(f"관련성 등급: {result.relevance_score}")
print(f"신뢰도: {result.confidence:.2f}")
print(f"평가 근거: {result.reasoning}")
```

**실행 결과**:
```
질문: 스테이크 메뉴의 가격은 얼마인가요?

관련성 등급: excellent
신뢰도: 0.95
평가 근거: 문서에 "프리미엄 립아이 스테이크" 메뉴의 가격이 명확히 "45,000원"으로
          명시되어 있어 질문에 직접적이고 완전한 답변을 제공합니다.
```

### 솔루션 2: Multi-Domain Vector Store

```python
class MultiDomainVectorStore:
    """다중 도메인 벡터 저장소 관리 시스템"""

    def __init__(self, domains: Dict[str, str]):
        """
        domains: {"domain_name": "collection_name"} 형태
        예: {"menu": "restaurant_menu", "wine": "wine_list", "review": "customer_reviews"}
        """
        self.domains = domains
        self.stores = {}

        # 각 도메인별 벡터 저장소 초기화
        for domain, collection_name in domains.items():
            self.stores[domain] = AdaptiveVectorStore(collection_name)
            logger.info(f"✅ Domain '{domain}' initialized with collection '{collection_name}'")

    def analyze_query_domain(self, query: str) -> List[str]:
        """질문을 분석하여 관련 도메인 식별"""

        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

        domain_prompt = f"""
        다음 질문을 분석하여 관련된 도메인을 선택하세요.

        사용 가능한 도메인:
        {', '.join(self.domains.keys())}

        질문: {query}

        관련 도메인을 쉼표로 구분하여 나열하세요 (예: menu, wine).
        여러 도메인이 관련될 수 있습니다.
        """

        response = llm.invoke(domain_prompt).content.strip()

        # 응답에서 도메인 추출
        selected_domains = [
            d.strip() for d in response.split(',')
            if d.strip() in self.domains
        ]

        if not selected_domains:
            # 모든 도메인 검색
            selected_domains = list(self.domains.keys())

        logger.info(f"선택된 도메인: {selected_domains}")
        return selected_domains

    def route_and_search(self, query: str) -> List[Document]:
        """질문 분석 → 도메인 라우팅 → 검색 → 결과 통합"""

        # 1. 도메인 분석
        target_domains = self.analyze_query_domain(query)

        # 2. 각 도메인에서 검색
        all_results = []
        for domain in target_domains:
            logger.info(f"🔍 Searching in domain: {domain}")
            results = self.stores[domain].multi_stage_search(query)

            # 도메인 메타데이터 추가
            for doc in results:
                doc.metadata['source_domain'] = domain

            all_results.extend(results)

        # 3. 결과 통합 (중복 제거 및 정렬)
        logger.info(f"✅ Total results: {len(all_results)} from {len(target_domains)} domains")
        return all_results

    def search_specific_domains(self, query: str, domains: List[str]) -> List[Document]:
        """특정 도메인에서만 검색"""
        all_results = []

        for domain in domains:
            if domain in self.stores:
                results = self.stores[domain].multi_stage_search(query)
                for doc in results:
                    doc.metadata['source_domain'] = domain
                all_results.extend(results)

        return all_results

# 사용 예시
domains = {
    "menu": "restaurant_menu",
    "wine": "wine_list",
    "review": "customer_reviews"
}

multi_store = MultiDomainVectorStore(domains)

# 자동 도메인 라우팅 검색
query = "스테이크와 어울리는 와인을 추천해주세요"
results = multi_store.route_and_search(query)

print(f"\n검색 결과: {len(results)}개")
for doc in results:
    print(f"- 도메인: {doc.metadata.get('source_domain', 'unknown')}")
    print(f"  내용: {doc.page_content[:100]}...")
    print()
```

### 솔루션 3: Cascading Search Strategy

```python
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class SearchLevel:
    """검색 레벨 정의"""
    level: int
    threshold: float
    k: int
    min_docs: int
    description: str

@dataclass
class SearchResult:
    """검색 결과 및 통계"""
    documents: List[Document]
    level_used: int
    levels_attempted: List[int]
    total_attempts: int
    search_history: List[Dict]

class CascadingSearchStrategy:
    """캐스케이딩 검색 전략"""

    def __init__(self, vector_db: Chroma):
        self.vector_db = vector_db

        # 5단계 검색 전략 정의
        self.search_levels = [
            SearchLevel(1, 0.5, 2, 2, "매우 정밀 검색 (높은 품질)"),
            SearchLevel(2, 0.4, 3, 2, "정밀 검색"),
            SearchLevel(3, 0.3, 5, 1, "표준 검색"),
            SearchLevel(4, 0.2, 7, 1, "확장 검색"),
            SearchLevel(5, 0.0, 10, 0, "포괄 검색 (모든 결과)"),
        ]

        self.search_history = []

    def search(self, query: str) -> SearchResult:
        """캐스케이딩 검색 실행"""

        levels_attempted = []
        history = []

        for level in self.search_levels:
            levels_attempted.append(level.level)

            logger.info(f"🔍 Level {level.level}: {level.description}")
            logger.info(f"   threshold={level.threshold}, k={level.k}, min_docs={level.min_docs}")

            # 검색 실행
            start_time = datetime.now()

            results = self.vector_db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": level.k, "score_threshold": level.threshold}
            ).invoke(query)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # 검색 이력 기록
            history_entry = {
                "level": level.level,
                "threshold": level.threshold,
                "k": level.k,
                "results_count": len(results),
                "duration_seconds": duration,
                "timestamp": start_time.isoformat()
            }
            history.append(history_entry)

            logger.info(f"   ✓ {len(results)}개 문서 검색 ({duration:.3f}초)")

            # 최소 문서 개수 충족 시 종료
            if len(results) >= level.min_docs:
                logger.info(f"✅ Level {level.level}에서 충분한 문서 발견")

                search_result = SearchResult(
                    documents=results,
                    level_used=level.level,
                    levels_attempted=levels_attempted,
                    total_attempts=len(levels_attempted),
                    search_history=history
                )

                self.search_history.append(search_result)
                return search_result

        # 모든 레벨 시도 후에도 결과 없음
        logger.warning("⚠️ 모든 검색 레벨에서 문서를 찾지 못했습니다")

        search_result = SearchResult(
            documents=[],
            level_used=-1,
            levels_attempted=levels_attempted,
            total_attempts=len(levels_attempted),
            search_history=history
        )

        self.search_history.append(search_result)
        return search_result

    def get_statistics(self) -> Dict:
        """검색 통계 제공"""
        if not self.search_history:
            return {"total_searches": 0}

        stats = {
            "total_searches": len(self.search_history),
            "level_usage": {},
            "average_attempts": sum(r.total_attempts for r in self.search_history) / len(self.search_history),
            "success_rate": sum(1 for r in self.search_history if r.documents) / len(self.search_history)
        }

        # 레벨별 사용 통계
        for result in self.search_history:
            level = result.level_used
            stats["level_usage"][level] = stats["level_usage"].get(level, 0) + 1

        return stats

# 사용 예시
vector_db = Chroma(
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_name="test_collection",
    persist_directory="./chroma_db"
)

cascading_search = CascadingSearchStrategy(vector_db)

# 검색 실행
query = "채식주의자를 위한 메뉴가 있나요?"
result = cascading_search.search(query)

# 결과 출력
print(f"\n{'='*60}")
print(f"검색 결과 요약")
print(f"{'='*60}")
print(f"사용된 레벨: Level {result.level_used}")
print(f"시도한 레벨: {result.levels_attempted}")
print(f"총 시도 횟수: {result.total_attempts}")
print(f"검색된 문서: {len(result.documents)}개\n")

print(f"{'='*60}")
print(f"검색 이력")
print(f"{'='*60}")
for entry in result.search_history:
    print(f"Level {entry['level']}: {entry['results_count']}개 ({entry['duration_seconds']:.3f}초)")

# 통계 조회
stats = cascading_search.get_statistics()
print(f"\n{'='*60}")
print(f"전체 통계")
print(f"{'='*60}")
print(f"총 검색 횟수: {stats['total_searches']}")
print(f"평균 시도 횟수: {stats['average_attempts']:.2f}")
print(f"성공률: {stats['success_rate']*100:.1f}%")
print(f"레벨별 사용: {stats['level_usage']}")
```

## 🚀 실무 활용 예시

### 활용 1: 고객 지원 챗봇

```python
class CustomerSupportCRAG:
    """고객 지원을 위한 CRAG 시스템"""

    def __init__(self):
        # FAQ 벡터 저장소
        self.faq_store = AdaptiveVectorStore("customer_faq")

        # 제품 매뉴얼 벡터 저장소
        self.manual_store = AdaptiveVectorStore("product_manual")

        # 웹 검색 (최신 정보)
        self.web_search = TavilySearch(max_results=3)

    def answer_customer_query(self, query: str) -> Dict:
        """고객 질문에 답변"""

        # 1. FAQ 검색
        faq_docs = self.faq_store.multi_stage_search(query)

        # 2. 문서 평가
        relevant_docs = []
        for doc in faq_docs:
            grade = retrieval_grader.invoke({
                "question": query,
                "document": doc.page_content
            })
            if grade.relevance_score == "correct":
                relevant_docs.append(doc)

        # 3. 충분한 정보가 있으면 답변 생성
        if relevant_docs:
            answer = generator_answer(query, relevant_docs)
            return {
                "answer": answer,
                "source": "FAQ",
                "confidence": "high"
            }

        # 4. FAQ에 없으면 매뉴얼 검색
        manual_docs = self.manual_store.multi_stage_search(query)

        if manual_docs:
            # 지식 정제
            refined_knowledge = []
            for doc in manual_docs:
                refined = knowledge_refiner.invoke({
                    "question": query,
                    "document": doc.page_content
                })
                if refined.binary_score == "yes":
                    refined_knowledge.append(Document(page_content=refined.knowledge_strip))

            if refined_knowledge:
                answer = generator_answer(query, refined_knowledge)
                return {
                    "answer": answer,
                    "source": "Product Manual",
                    "confidence": "medium"
                }

        # 5. 내부 문서에도 없으면 질문 재작성 + 웹 검색
        rewritten = rewrite_question(query)
        web_results = self.web_search.invoke(rewritten)

        web_docs = [
            Document(page_content=result['content'])
            for result in web_results['results']
        ]

        if web_docs:
            answer = generator_answer(query, web_docs)
            return {
                "answer": answer,
                "source": "Web Search",
                "confidence": "low",
                "suggestion": "답변을 검증하시기 바랍니다."
            }

        return {
            "answer": "죄송합니다. 해당 질문에 대한 정보를 찾을 수 없습니다. 고객센터로 문의해 주세요.",
            "source": None,
            "confidence": "none"
        }

# 사용 예시
support_bot = CustomerSupportCRAG()

customer_queries = [
    "제품 보증 기간은 얼마나 되나요?",
    "배송 지연 시 어떻게 하나요?",
    "최신 소프트웨어 업데이트는 언제인가요?"
]

for query in customer_queries:
    print(f"\n{'='*60}")
    print(f"고객 질문: {query}")
    print(f"{'='*60}")

    response = support_bot.answer_customer_query(query)

    print(f"답변: {response['answer']}")
    print(f"출처: {response['source']}")
    print(f"신뢰도: {response['confidence']}")
    if 'suggestion' in response:
        print(f"참고: {response['suggestion']}")
```

### 활용 2: 법률 문서 분석 시스템

```python
class LegalDocumentAnalyzer:
    """법률 문서 분석을 위한 CRAG 시스템"""

    def __init__(self, law_domains: Dict[str, str]):
        """
        law_domains: {"domain": "collection_name"}
        예: {"housing": "housing_law", "labor": "labor_law"}
        """
        self.law_stores = {}
        for domain, collection in law_domains.items():
            self.law_stores[domain] = AdaptiveVectorStore(collection)

    def analyze_legal_question(self, question: str) -> Dict:
        """법률 질문 분석 및 답변"""

        results = {
            "question": question,
            "relevant_laws": [],
            "answer": "",
            "references": []
        }

        # 모든 법률 도메인에서 검색
        all_docs = []
        for domain, store in self.law_stores.items():
            docs = store.multi_stage_search(question)

            # 도메인 정보 추가
            for doc in docs:
                doc.metadata['law_domain'] = domain

            all_docs.extend(docs)

        if not all_docs:
            # 질문 재작성 후 재검색
            rewritten = rewrite_question(question)
            for domain, store in self.law_stores.items():
                docs = store.multi_stage_search(rewritten)
                for doc in docs:
                    doc.metadata['law_domain'] = domain
                all_docs.extend(docs)

        # 문서 평가 및 정제
        refined_knowledge = []
        for doc in all_docs:
            # 관련성 평가
            grade = retrieval_grader.invoke({
                "question": question,
                "document": doc.page_content
            })

            if grade.relevance_score in ["correct", "ambiguous"]:
                # 지식 정제
                refined = knowledge_refiner.invoke({
                    "question": question,
                    "document": doc.page_content
                })

                if refined.binary_score == "yes":
                    refined_doc = Document(
                        page_content=refined.knowledge_strip,
                        metadata=doc.metadata
                    )
                    refined_knowledge.append(refined_doc)

                    # 관련 법률 기록
                    if doc.metadata['law_domain'] not in results['relevant_laws']:
                        results['relevant_laws'].append(doc.metadata['law_domain'])

        # 답변 생성
        if refined_knowledge:
            results['answer'] = generator_answer(question, refined_knowledge)
            results['references'] = [
                f"{doc.metadata.get('law_domain', 'unknown')}: {doc.page_content[:100]}..."
                for doc in refined_knowledge
            ]
        else:
            results['answer'] = "죄송합니다. 해당 질문에 대한 법률 정보를 찾을 수 없습니다. 전문 법률 상담을 권장드립니다."

        return results

# 사용 예시
law_domains = {
    "housing": "housing_leasing_law",
    "labor": "labor_standards_law",
    "privacy": "personal_info_law"
}

legal_analyzer = LegalDocumentAnalyzer(law_domains)

legal_questions = [
    "전세 계약 시 임차인 보호 권리는 무엇인가요?",
    "근로자의 연차 휴가 사용 권리에 대해 설명해주세요.",
    "개인정보 수집 시 동의가 필요한 경우는 언제인가요?"
]

for question in legal_questions:
    print(f"\n{'='*70}")
    print(f"법률 질문: {question}")
    print(f"{'='*70}")

    result = legal_analyzer.analyze_legal_question(question)

    print(f"\n관련 법률: {', '.join(result['relevant_laws'])}")
    print(f"\n답변:\n{result['answer']}")
    print(f"\n참고 조항:")
    for i, ref in enumerate(result['references'], 1):
        print(f"{i}. {ref}")
```

## 📖 참고 자료

### 공식 문서
- [LangChain 공식 문서](https://python.langchain.com/docs/get_started/introduction)
- [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph/)
- [ChromaDB 문서](https://docs.trychroma.com/)
- [OpenAI API 문서](https://platform.openai.com/docs/introduction)
- [Pydantic 문서](https://docs.pydantic.dev/)
- [Tavily Search API](https://tavily.com/)

### 논문
- [CRAG 논문 (Corrective Retrieval Augmented Generation)](https://arxiv.org/pdf/2401.15884) - Corrective RAG의 원본 논문

### 관련 블로그 및 튜토리얼
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [Building Production-Ready RAG Applications](https://www.langchain.com/blog)
- [Advanced RAG Techniques](https://blog.langchain.dev/)

### 추가 학습 자료
- **LangGraph 활용**: Part 2에서 다룰 예정 - StateGraph를 이용한 복잡한 워크플로우 구현
- **RAG 최적화**: 임베딩 모델 선택, 청크 크기 최적화, 하이브리드 검색
- **프로덕션 배포**: API 엔드포인트 구성, 캐싱 전략, 모니터링

---

**다음 단계**: Part 2에서는 LangGraph의 StateGraph를 활용하여 CRAG 시스템을 완전한 워크플로우로 구현합니다. Map-Reduce 패턴, 조건부 라우팅, 상태 관리 등 고급 기법을 다룰 예정입니다.
