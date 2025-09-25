# PRJ02_W1_004 검색 성능 향상 기법 매뉴얼 - 쿼리 확장 (Query Expansion)

## 📋 개요

이 노트북은 RAG 시스템의 검색 성능을 향상시키는 고급 쿼리 확장 기법들을 종합적으로 다룹니다. Query Reformulation, Multi Query, Decomposition, Step-Back Prompting, HyDE 등 5가지 핵심 기법을 통해 원본 쿼리를 개선하고 확장하여 더 정확하고 포괄적인 검색 결과를 얻는 방법을 학습합니다.

### 📊 실험 환경 및 결과 요약
- **데이터셋**: 한국어 전기차 관련 문서 (Tesla, Rivian 등)
- **벡터 저장소**: ChromaDB (text-embedding-3-small)
- **테스트 쿼리**: "미국 전기차 시장에서 Tesla와 다른 회사들의 차이점은?"
- **평가 지표**: Hit Rate, MRR, MAP, NDCG (k=2,3,4)
- **주요 결과**: 각 기법별 실제 검색 성능 및 문서 수 측정

### 🎯 학습 목표
- 쿼리 확장(Query Expansion) 기법을 구현하고 성능 개선을 측정
- 5가지 쿼리 확장 전략의 특징과 적용 상황 이해
- LLM을 활용한 지능형 쿼리 변환 시스템 구축
- 실습을 통한 각 기법의 개선 방법론 학습

## 🛠️ 환경 설정

### 1. 필수 패키지
```python
# 기본 라이브러리
import os
from glob import glob
from pprint import pprint
import json

# LangChain 관련
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List

# Langfuse 트레이싱
from langfuse.langchain import CallbackHandler
```

### 2. 벡터 저장소 초기화
```python
# 실제 구현된 벡터 저장소 로드
chroma_db = Chroma(
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_name="db_korean_cosine",
    persist_directory="./chroma_db"
)

# 기본 검색기 설정 (k=4)
chroma_k_retriever = chroma_db.as_retriever(search_kwargs={"k": 4})

# 실험용 테스트 쿼리 정의
test_queries = [
    "미국 전기차 시장에서 Tesla와 다른 회사들의 차이점은?",
    "리비안의 사업 경쟁력은 어디서 나오나요?",
    "테슬라의 기술력은 어떤가요?"
]

# K-RAG 평가자 설정
from krag.evaluators import OfflineRetrievalEvaluators

def setup_evaluator(actual_docs, predicted_docs):
    """평가자 초기화"""
    return OfflineRetrievalEvaluators(
        actual_docs, predicted_docs,
        match_method="text"
    )
```

## 🔍 쿼리 확장 기법 분류

### 1. Query Reformulation (쿼리 재구성)

**개념**: LLM을 활용해 원본 질문을 검색에 최적화된 형태로 재구성

**특징**:
- 🎯 **명확성 향상**: 모호한 질문을 구체적으로 변환
- 🔄 **검색 최적화**: 검색 엔진에 친화적인 표현으로 변경
- 📝 **단일 쿼리**: 하나의 개선된 쿼리 생성

**기본 구현**:
```python
# 실제 구현된 쿼리 리포뮬레이션 템플릿
reformulation_template = """다음 질문을 검색 성능을 향상시키기 위해 다시 작성해주세요:

[질문]
{question}

[개선된 질문]
"""

# 체인 구성
prompt = ChatPromptTemplate.from_template(reformulation_template)
llm = ChatOpenAI(model='gpt-4.1-mini', temperature=0)
reformulation_chain = prompt | llm | StrOutputParser()

# LCEL 패턴으로 검색기와 결합
reformulation_retriever = reformulation_chain | chroma_k_retriever

# 실제 테스트 결과 예시
query = "미국 전기차 시장에서 Tesla와 다른 회사들의 차이점은?"
reformulated = reformulation_chain.invoke({"question": query})
print(f"🔸 기존 리포뮬레이션 쿼리: {reformulated}")

# 검색 성능 비교
original_docs = chroma_k_retriever.invoke(query)
reformulated_docs = chroma_k_retriever.invoke(reformulated)
print(f"원본 검색 문서 수: {len(original_docs)}")
print(f"리포뮬레이션 검색 문서 수: {len(reformulated_docs)}")
```

**개선된 구현**:
```python
# 개선된 쿼리 리포뮬레이션 템플릿
improved_reformulation_template = """당신은 정보 검색 전문가입니다. 다음 질문을 검색 성능을 최대화하기 위해 개선해주세요.

개선 지침:
1. 구체적이고 명확한 키워드 사용
2. 검색 의도를 명확히 표현
3. 관련 동의어나 유사 표현 포함
4. 한국어 자연어 유지

[원본 질문]
{question}

[검색 최적화된 질문]
"""

# Langfuse 트레이싱이 포함된 체인
improved_reformulation_chain = ChatPromptTemplate.from_template(improved_reformulation_template) | \
                              ChatOpenAI(model='gpt-4.1-mini', temperature=0.3, callbacks=[langfuse_handler]) | \
                              StrOutputParser()
```

### 2. Multi Query (다중 쿼리)

**개념**: 단일 질문을 다양한 관점의 여러 쿼리로 확장하여 포괄적 검색 수행

**특징**:
- 🔀 **다각도 접근**: 다양한 관점에서 질문 생성
- 📈 **재현율 향상**: 더 많은 관련 문서 검색
- 🎯 **포괄성**: 놓칠 수 있는 정보까지 포착

**출력 파서 구현**:
```python
class LineListOutputParser(BaseOutputParser[List[str]]):
    """LLM 출력을 질문 리스트로 변환하는 파서"""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split('\n')
        return [line.strip() for line in lines if line.strip()]

    @property
    def _type(self) -> str:
        return "line_list"
```

**기본 구현**:
```python
# 쿼리 생성 프롬프트
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""주어진 질문과 관련된 서로 다른 3개의 질문을 생성해주세요.
각 질문은 원본 질문의 다른 측면을 다뤄야 합니다.
한 줄에 하나의 질문만 작성하세요.

원본 질문: {question}

생성된 질문들:
"""
)

# 멀티쿼리 체인 구성
multiquery_chain = QUERY_PROMPT | llm | LineListOutputParser()

# MultiQueryRetriever와 결합
multi_query_retriever = MultiQueryRetriever(
    retriever=chroma_k_retriever,
    llm_chain=multiquery_chain,
    parser_key="lines"
)
```

**개선된 구현**:
```python
# 개선된 멀티쿼리 프롬프트
IMPROVED_MULTIQUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""당신은 전문 연구원입니다. 주어진 질문에 대해 포괄적인 정보 수집을 위한 다양한 관점의 질문을 생성해주세요.

생성 지침:
1. 서로 다른 접근 방식의 질문 (기술적, 비즈니스적, 역사적 관점 등)
2. 구체적이고 검색 가능한 표현 사용
3. 각 질문은 독립적이면서 원본 질문과 관련성 유지
4. 정확히 4개의 질문 생성

원본 질문: {question}

다각도 분석 질문들:
"""
)

# 개선된 체인 (다양성을 위해 temperature 상승)
improved_multiquery_chain = IMPROVED_MULTIQUERY_PROMPT | \
                           ChatOpenAI(model="gpt-4.1-mini", temperature=0.7, callbacks=[langfuse_handler]) | \
                           LineListOutputParser()
```

### 3. Decomposition (쿼리 분해)

**개념**: 복잡한 질문을 작은 단위의 하위 질문들로 분해하여 단계별로 처리

**특징**:
- 🔧 **복잡성 단순화**: 어려운 문제를 작은 단위로 분할
- 📋 **단계별 해결**: 각 하위 문제를 순차적으로 해결
- 🎯 **정확성 향상**: 세분화된 검색으로 정밀도 증가

**기본 구현**:
```python
# 분해 프롬프트 (MultiQuery와 동일한 기본 구조 사용)
decomposition_chain = QUERY_PROMPT | llm | LineListOutputParser()

multi_query_decomposition_retriever = MultiQueryRetriever(
    retriever=chroma_k_retriever,
    llm_chain=decomposition_chain,
    parser_key="lines"
)
```

**개선된 구현**:
```python
# 개선된 분해 프롬프트
IMPROVED_DECOMPOSITION_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""당신은 체계적 분석 전문가입니다. 복잡한 질문을 논리적 순서의 하위 질문들로 분해해주세요.

분해 원칙:
1. 논리적 순서: 기본 개념 → 구체적 내용 → 응용/비교
2. 독립성: 각 하위 질문은 독립적으로 답변 가능
3. 완전성: 모든 하위 답변을 통합하면 원본 질문의 완전한 답변 구성
4. 실행 가능성: 실제 검색으로 답변을 찾을 수 있는 질문

복합 질문: {question}

논리적 하위 질문들:
"""
)

# 구조화된 분해를 위해 중간 온도 설정
improved_decomposition_chain = IMPROVED_DECOMPOSITION_PROMPT | \
                              ChatOpenAI(model="gpt-4.1-mini", temperature=0.5, callbacks=[langfuse_handler]) | \
                              LineListOutputParser()
```

### 4. Step-Back Prompting (단계적 후퇴)

**개념**: 구체적 질문을 일반적이고 포괄적인 맥락에서 접근하는 방식

**특징**:
- 🔙 **추상화**: 구체적 → 일반적 관점으로 후퇴
- 🌐 **맥락 확장**: 더 넓은 배경 지식 활용
- 🎯 **원리 중심**: 근본적 원리에서 출발

**Few-Shot 예제 설정**:
```python
# Few-Shot 예제 정의
examples = [
    {
        "input": "테슬라 Model 3의 2023년 판매량은 얼마인가요?",
        "output": "전기차 시장에서 테슬라의 전체적인 판매 성과와 시장 점유율은 어떻게 변화하고 있나요?"
    },
    {
        "input": "리비안 R1T의 배터리 용량은?",
        "output": "전기 픽업트럭 시장에서 배터리 기술과 성능 경쟁은 어떻게 이루어지고 있나요?"
    },
    {
        "input": "포드 F-150 Lightning의 충전 시간은?",
        "output": "전기차 충전 인프라와 충전 기술의 발전 현황은 어떠한가요?"
    }
]

# Few-Shot 프롬프트 구성
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# Step-Back 프롬프트
step_back_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 분석적 사고의 전문가입니다. 구체적인 질문을 더 일반적이고 포괄적인 관점의 질문으로 변환해주세요."),
    few_shot_prompt,
    ("human", "{question}"),
])
```

**체인 구성 및 답변 생성**:
```python
# Step-Back 체인
step_back_chain = step_back_prompt | llm | StrOutputParser()

# 답변 생성을 위한 이중 검색 체인
response_prompt = ChatPromptTemplate.from_template(
    """당신은 전문가입니다. 다음 컨텍스트와 질문을 바탕으로 포괄적인 답변을 제공해주세요.

컨텍스트:
{context}

질문: {question}

답변:
"""
)

def format_docs(docs):
    return "\n".join([doc.page_content for doc in docs])

# 이중 검색 및 답변 생성 체인
answer_chain = (
    {
        "step_back_docs": step_back_chain | chroma_k_retriever,
        "normal_docs": chroma_k_retriever,
        "question": lambda x: x
    }
    | (lambda x: {
        "context": format_docs(x["step_back_docs"] + x["normal_docs"]),
        "question": x["question"]
    })
    | response_prompt
    | llm
    | StrOutputParser()
)
```

### 5. HyDE (Hypothetical Document Embedding)

**개념**: 질문에 대한 가상의 이상적인 답변 문서를 생성하고, 이를 기반으로 유사 문서를 검색

**특징**:
- 🎭 **가상 문서**: 이상적인 답변 문서 생성
- 🔍 **의미적 매칭**: 가상 문서와 유사한 실제 문서 검색
- 📈 **검색 품질**: 문서-문서 유사성을 통한 높은 정확도

**기본 구현**:
```python
# HyDE 문서 생성 프롬프트
hyde_template = """주어진 질문에 대한 이상적인 문서 내용을 생성해주세요.
문서는 학술적이고 전문적인 톤으로 작성되어야 합니다.

질문: {question}

이상적인 문서:
"""

# HyDE 체인
hyde_prompt = ChatPromptTemplate.from_template(hyde_template)
hyde_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
hyde_chain = hyde_prompt | hyde_llm | StrOutputParser()

# RAG 체인 (가상 문서 기반 검색 → 실제 답변)
rag_template = """다음 컨텍스트를 바탕으로 질문에 답변해주세요:

컨텍스트:
{context}

질문: {question}

답변:
"""

rag_prompt = ChatPromptTemplate.from_template(rag_template)
rag_chain = rag_prompt | hyde_llm | StrOutputParser()
```

**다관점 HyDE 구현**:
```python
# 기술적 관점 HyDE
technical_hyde_template = """기술 전문가 관점에서 다음 질문에 대한 상세한 기술 분석 문서를 작성해주세요.

포함 요소:
- 기술적 사양과 성능 지표
- 기술적 혁신점과 차별화 요소
- 기술적 한계와 개선 방향
- 경쟁 기술과의 비교

질문: {question}

기술 분석 문서:
"""

# 비즈니스 관점 HyDE
business_hyde_template = """비즈니스 전략 전문가 관점에서 다음 질문에 대한 종합적인 시장 분석 문서를 작성해주세요.

포함 요소:
- 시장 동향과 경쟁 환경
- 비즈니스 모델과 수익 구조
- 투자와 성장 전략
- 위험 요소와 기회 분석

질문: {question}

시장 분석 문서:
"""

# 다중 관점 HyDE 체인들
technical_hyde_chain = ChatPromptTemplate.from_template(technical_hyde_template) | \
                      ChatOpenAI(model="gpt-4.1-mini", temperature=0.7, callbacks=[langfuse_handler]) | \
                      StrOutputParser()

business_hyde_chain = ChatPromptTemplate.from_template(business_hyde_template) | \
                     ChatOpenAI(model="gpt-4.1-mini", temperature=0.7, callbacks=[langfuse_handler]) | \
                     StrOutputParser()
```

## 🧪 실습 과제 및 개선 방법론

### 실습 1: Query Reformulation 개선

**개선 요소**:
1. **구체성 강화**: 모호한 표현을 구체적 키워드로 변환
2. **검색 친화적**: 검색 엔진 최적화 고려
3. **맥락 보존**: 원본 의도 유지

```python
# 개선 전후 비교 테스트
test_queries = [
    "리비안의 사업 경쟁력은 어디서 나오나요?",
    "테슬라의 기술력은 어떤가요?",
    "전기차 시장 전망은?"
]

for query in test_queries:
    original = reformulation_chain.invoke({"question": query})
    improved = improved_reformulation_chain.invoke({"question": query})

    print(f"원본: {query}")
    print(f"기본 개선: {original}")
    print(f"고급 개선: {improved}")
    print("-" * 80)
```

### 실습 2: MultiQuery 구조 분석 및 개선

**분석 요소**:
1. **질문 다양성**: 생성된 질문들의 관점 차이
2. **검색 커버리지**: 다양한 문서 검색 범위
3. **중복 제거**: 유사한 질문 필터링

```python
def analyze_query_diversity(queries, title):
    """생성된 질문의 다양성 분석"""
    print(f"\n{title}")

    # 질문별 키워드 추출
    all_words = []
    for q in queries:
        words = set(q.lower().replace('?', '').split())
        all_words.extend(words)

    unique_words = len(set(all_words))
    total_words = len(all_words)
    diversity_ratio = unique_words / total_words if total_words > 0 else 0

    print(f"총 단어: {total_words}, 고유 단어: {unique_words}")
    print(f"다양성 비율: {diversity_ratio:.3f}")

    return diversity_ratio, unique_words
```

### 실습 3: Query Decomposition 최적화

**최적화 전략**:
1. **논리적 순서**: 기본 개념 → 세부 사항 → 응용
2. **독립성**: 각 하위 질문의 독립적 실행 가능성
3. **완전성**: 전체 답변 구성을 위한 충분성

```python
def analyze_subquery_search(subqueries, retriever, title):
    """서브 질문별 검색 성능 분석"""
    print(f"\n{title}")

    all_docs = []
    source_counts = {}

    for i, subq in enumerate(subqueries):
        docs = retriever.invoke(subq)
        all_docs.extend(docs)

        print(f"서브질문 {i+1}: {subq}")
        print(f"검색 문서 수: {len(docs)}")

        # 출처 분석
        for doc in docs:
            source = doc.metadata.get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1

    return len(all_docs), source_counts
```

### 실습 4: Step-Back Prompting 고도화

**고도화 방향**:
1. **도메인 특화**: 특정 분야에 맞는 추상화 패턴
2. **계층적 접근**: 다단계 추상화 레벨
3. **맥락 통합**: 일반론과 구체론의 조화

```python
# 개선된 Few-Shot 예제 (도메인 특화)
enhanced_examples = [
    {
        "input": "리비안 R1T의 최대 견인력은 얼마인가요?",
        "output": "전기 픽업트럭 시장에서 견인 성능과 실용성 경쟁은 어떻게 이루어지고 있으며, 이것이 소비자 선택에 미치는 영향은 무엇인가요?"
    },
    {
        "input": "테슬라 슈퍼차저 V4의 충전 속도는?",
        "output": "전기차 충전 인프라의 기술 발전이 전기차 보급과 사용자 경험에 미치는 영향은 어떠하며, 향후 발전 방향은 무엇인가요?"
    }
]

# 개선된 Step-Back 시스템
enhanced_step_back_prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 전략적 분석 전문가입니다. 구체적인 질문을 더 포괄적이고 통찰력 있는 관점으로 변환해주세요.

변환 지침:
1. 단순 사실 → 트렌드와 영향 분석
2. 개별 제품 → 시장 생태계 관점
3. 기술 스펙 → 기술 발전의 의미와 방향
4. 즉석 답변보다는 깊이 있는 이해 추구"""),
    FewShotChatMessagePromptTemplate(
        example_prompt=ChatPromptTemplate.from_messages([
            ("human", "{input}"),
            ("ai", "{output}"),
        ]),
        examples=enhanced_examples,
    ),
    ("human", "{question}"),
])
```

### 실습 5: HyDE 시스템 정교화

**정교화 요소**:
1. **다중 관점**: 기술적, 비즈니스적, 사용자 관점
2. **품질 제어**: 생성된 문서의 품질 검증
3. **적응적 선택**: 질문 유형에 따른 최적 관점 선택

```python
class AdaptiveHyDESystem:
    """적응적 HyDE 시스템"""

    def __init__(self, retriever):
        self.retriever = retriever
        self.technical_chain = technical_hyde_chain
        self.business_chain = business_hyde_chain
        self.general_chain = hyde_chain

    def _classify_query_domain(self, query):
        """쿼리 도메인 분류"""
        technical_keywords = ['기술', '성능', '사양', '엔진', '배터리', '충전']
        business_keywords = ['시장', '경쟁', '전략', '투자', '수익', '점유율']

        if any(keyword in query for keyword in technical_keywords):
            return 'technical'
        elif any(keyword in query for keyword in business_keywords):
            return 'business'
        else:
            return 'general'

    def invoke(self, query):
        """도메인별 적응적 HyDE 실행"""
        domain = self._classify_query_domain(query)

        if domain == 'technical':
            hypothetical_doc = self.technical_chain.invoke({"question": query})
        elif domain == 'business':
            hypothetical_doc = self.business_chain.invoke({"question": query})
        else:
            hypothetical_doc = self.general_chain.invoke({"question": query})

        # 가상 문서로 실제 문서 검색
        retrieved_docs = self.retriever.invoke(hypothetical_doc)

        return {
            'domain': domain,
            'hypothetical_doc': hypothetical_doc,
            'retrieved_docs': retrieved_docs
        }
```

## 🎯 성능 최적화 전략

### 1. 기법별 특성 비교

| 기법 | 장점 | 단점 | 최적 상황 |
|------|------|------|-----------|
| **Query Reformulation** | 간단, 빠름 | 제한적 확장 | 명확한 질문 개선 |
| **Multi Query** | 포괄적 검색 | 중복 위험 | 탐색적 검색 |
| **Decomposition** | 논리적 구조 | 복잡성 증가 | 복합 질문 |
| **Step-Back** | 맥락 이해 | 추상화 과다 | 배경 지식 필요 |
| **HyDE** | 높은 정확도 | 계산 비용 | 정밀 검색 |

### 2. 하이브리드 전략

```python
class QueryExpansionOrchestrator:
    """다중 쿼리 확장 기법 통합 관리자"""

    def __init__(self, retriever):
        self.retriever = retriever
        self.strategies = {
            'reformulation': improved_reformulation_chain,
            'multi_query': improved_multiquery_chain,
            'decomposition': improved_decomposition_chain,
            'step_back': enhanced_step_back_chain,
            'hyde': AdaptiveHyDESystem(retriever)
        }

    def _select_strategy(self, query, context=None):
        """질문 특성에 따른 최적 전략 선택"""
        query_lower = query.lower()

        # 복합 질문 감지
        if '그리고' in query or '또한' in query or len(query.split('?')) > 2:
            return 'decomposition'

        # 구체적 사실 질문
        elif any(word in query_lower for word in ['언제', '얼마', '몇', '어디']):
            return 'hyde'

        # 비교/분석 질문
        elif any(word in query_lower for word in ['비교', '차이', '장단점', '어떤']):
            return 'step_back'

        # 탐색적 질문
        elif any(word in query_lower for word in ['어떻게', '왜', '방법']):
            return 'multi_query'

        # 기본: reformulation
        else:
            return 'reformulation'

    def expand_query(self, query, strategy=None):
        """쿼리 확장 실행"""
        if strategy is None:
            strategy = self._select_strategy(query)

        selected_strategy = self.strategies[strategy]

        if strategy == 'hyde':
            return selected_strategy.invoke(query)
        else:
            expanded_query = selected_strategy.invoke({"question": query})
            retrieved_docs = self.retriever.invoke(expanded_query)

            return {
                'strategy': strategy,
                'expanded_query': expanded_query,
                'retrieved_docs': retrieved_docs
            }
```

### 3. 성능 모니터링

```python
class QueryExpansionMonitor:
    """쿼리 확장 성능 모니터링"""

    def __init__(self):
        self.metrics_history = []
        self.strategy_performance = {}

    def evaluate_expansion(self, original_query, expanded_result, ground_truth):
        """확장 결과 평가"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score

        # 검색 품질 평가 (간단한 예시)
        retrieved_docs = expanded_result.get('retrieved_docs', [])
        relevance_scores = [
            self._calculate_relevance(doc, ground_truth)
            for doc in retrieved_docs
        ]

        metrics = {
            'avg_relevance': np.mean(relevance_scores) if relevance_scores else 0,
            'doc_count': len(retrieved_docs),
            'strategy': expanded_result.get('strategy', 'unknown'),
            'timestamp': datetime.now()
        }

        self.metrics_history.append(metrics)
        return metrics

    def _calculate_relevance(self, doc, ground_truth):
        """문서-정답 관련성 계산 (ROUGE-like)"""
        # 간단한 단어 겹침 기반 유사도
        doc_words = set(doc.page_content.lower().split())
        truth_words = set(ground_truth.lower().split())

        intersection = doc_words.intersection(truth_words)
        union = doc_words.union(truth_words)

        return len(intersection) / len(union) if union else 0

    def get_strategy_rankings(self):
        """전략별 성능 순위"""
        strategy_scores = {}

        for metric in self.metrics_history:
            strategy = metric['strategy']
            score = metric['avg_relevance']

            if strategy not in strategy_scores:
                strategy_scores[strategy] = []
            strategy_scores[strategy].append(score)

        # 평균 성능 계산
        avg_scores = {
            strategy: np.mean(scores)
            for strategy, scores in strategy_scores.items()
        }

        # 순위 반환
        return sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
```

## 💡 실무 활용 가이드

### 1. 상황별 기법 선택

```python
def recommend_expansion_technique(query_type, complexity, domain):
    """상황별 최적 기법 추천"""

    recommendations = []

    # 복잡도 기반
    if complexity == 'high':
        recommendations.append('decomposition')
    elif complexity == 'medium':
        recommendations.append('multi_query')
    else:
        recommendations.append('reformulation')

    # 도메인 기반
    if domain == 'technical':
        recommendations.append('hyde')
    elif domain == 'exploratory':
        recommendations.extend(['step_back', 'multi_query'])

    # 질문 유형 기반
    if query_type == 'factual':
        recommendations.append('hyde')
    elif query_type == 'comparative':
        recommendations.append('step_back')
    elif query_type == 'procedural':
        recommendations.append('decomposition')

    # 가장 많이 추천된 기법 선택
    from collections import Counter
    return Counter(recommendations).most_common(1)[0][0]
```

### 2. 성능 벤치마킹

```python
class ExpansionBenchmark:
    """쿼리 확장 기법 벤치마킹"""

    def __init__(self, test_queries, ground_truths):
        self.test_queries = test_queries
        self.ground_truths = ground_truths
        self.results = {}

    def run_benchmark(self, techniques):
        """벤치마크 실행"""
        for technique_name, technique_func in techniques.items():
            print(f"벤치마킹: {technique_name}")

            technique_results = []

            for query, truth in zip(self.test_queries, self.ground_truths):
                try:
                    start_time = time.time()
                    result = technique_func(query)
                    end_time = time.time()

                    # 성능 메트릭 계산
                    metrics = self._calculate_metrics(result, truth)
                    metrics['latency'] = end_time - start_time

                    technique_results.append(metrics)

                except Exception as e:
                    print(f"오류 발생 - {technique_name}: {e}")
                    technique_results.append(None)

            self.results[technique_name] = technique_results

    def generate_report(self):
        """벤치마크 결과 리포트 생성"""
        report = "## 쿼리 확장 기법 벤치마크 결과\n\n"

        for technique, results in self.results.items():
            valid_results = [r for r in results if r is not None]

            if valid_results:
                avg_relevance = np.mean([r['relevance'] for r in valid_results])
                avg_latency = np.mean([r['latency'] for r in valid_results])
                success_rate = len(valid_results) / len(results)

                report += f"### {technique}\n"
                report += f"- 평균 관련성: {avg_relevance:.3f}\n"
                report += f"- 평균 지연시간: {avg_latency:.3f}초\n"
                report += f"- 성공률: {success_rate:.1%}\n\n"

        return report
```

## 🔧 문제 해결 및 최적화

### 1. 일반적인 문제들

```python
# 1. LLM 응답 불안정성 해결
def robust_llm_invoke(chain, input_data, max_retries=3):
    """견고한 LLM 호출"""
    for attempt in range(max_retries):
        try:
            return chain.invoke(input_data)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2 ** attempt)  # 지수 백오프

# 2. 토큰 제한 관리
def manage_token_limits(text, max_tokens=4000):
    """토큰 제한 관리"""
    # 간단한 토큰 추정 (실제로는 tiktoken 사용 권장)
    estimated_tokens = len(text.split()) * 1.3

    if estimated_tokens > max_tokens:
        # 텍스트 축약
        words = text.split()
        target_words = int(max_tokens / 1.3)
        return ' '.join(words[:target_words])

    return text

# 3. 결과 품질 검증
def validate_expansion_quality(original_query, expanded_query):
    """확장 결과 품질 검증"""
    checks = {
        'length_reasonable': 10 <= len(expanded_query) <= 1000,
        'contains_keywords': any(word in expanded_query.lower()
                               for word in original_query.lower().split()),
        'is_question': '?' in expanded_query or any(word in expanded_query
                                                 for word in ['무엇', '어떻게', '왜', '언제']),
        'no_repetition': expanded_query != original_query
    }

    return all(checks.values()), checks
```

### 2. 성능 최적화

```python
# 캐싱 시스템
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_query_expansion(query, strategy):
    """쿼리 확장 결과 캐싱"""
    # 실제 구현에서는 strategy에 따른 적절한 체인 호출
    pass

# 배치 처리
def batch_query_expansion(queries, strategy, batch_size=5):
    """배치 단위 쿼리 확장"""
    results = []

    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]

        # 배치별 병렬 처리 (실제로는 asyncio 사용 권장)
        batch_results = [
            expand_single_query(query, strategy)
            for query in batch
        ]

        results.extend(batch_results)

    return results
```

## 📚 고급 활용 패턴

### 1. 적응형 쿼리 확장

```python
class AdaptiveQueryExpander:
    """사용자 피드백을 통한 적응형 확장"""

    def __init__(self):
        self.user_preferences = {}
        self.success_history = {}

    def expand_with_learning(self, query, user_id=None):
        """학습 기반 쿼리 확장"""
        # 사용자별 선호도 고려
        if user_id and user_id in self.user_preferences:
            preferred_strategy = self.user_preferences[user_id]
        else:
            preferred_strategy = self._get_best_strategy_for_query(query)

        result = self._apply_strategy(query, preferred_strategy)

        return {
            'result': result,
            'strategy_used': preferred_strategy,
            'feedback_id': self._generate_feedback_id()
        }

    def update_from_feedback(self, feedback_id, satisfaction_score, user_id=None):
        """사용자 피드백으로 모델 업데이트"""
        # 피드백 기반 전략 조정 로직
        pass
```

### 2. 다중 언어 지원

```python
class MultilingualQueryExpander:
    """다국어 쿼리 확장 시스템"""

    def __init__(self):
        self.language_chains = {
            'ko': korean_expansion_chain,
            'en': english_expansion_chain,
            'ja': japanese_expansion_chain
        }

    def detect_language(self, query):
        """언어 감지"""
        # 간단한 언어 감지 로직
        if any(ord(char) >= 0xAC00 and ord(char) <= 0xD7A3 for char in query):
            return 'ko'
        elif any(ord(char) >= 0x3040 and ord(char) <= 0x309F for char in query):
            return 'ja'
        else:
            return 'en'

    def expand_multilingual(self, query):
        """다국어 대응 확장"""
        detected_lang = self.detect_language(query)
        appropriate_chain = self.language_chains.get(detected_lang, self.language_chains['en'])

        return appropriate_chain.invoke({"question": query})
```

## 🚀 다음 단계 및 발전 방향

### 1. 고급 기법 탐구
- **Neural Query Expansion**: 딥러닝 기반 확장
- **Reinforcement Learning**: 강화학습을 통한 최적화
- **Federated Learning**: 분산 학습을 통한 성능 향상

### 2. 실무 통합
- **A/B Testing Framework**: 실제 서비스에서의 성능 비교
- **Real-time Optimization**: 실시간 성능 최적화
- **User Experience Integration**: 사용자 경험과 통합된 평가

### 3. 도메인 특화
- **의료 정보 검색**: 의료 도메인 특화 확장
- **법률 문서 검색**: 법률 전문 용어 처리
- **기술 문서 검색**: 기술 사양 중심 확장

---

## 📊 실습 완료 요약

### 🎯 실습 성과
1. **5가지 쿼리 확장 기법 구현 완료**
   - Query Reformulation, MultiQuery, Decomposition, Step-Back, HyDE
   - 각 기법별 기본 및 개선된 버전 구현

2. **실제 성능 측정 완료**
   - K-RAG 패키지를 활용한 체계적 평가
   - Hit Rate, MRR, MAP, NDCG 지표 측정
   - k=2,3,4 설정에서 성능 비교

3. **검색 성능 비교 분석**
   - 각 기법별 검색된 문서 수 측정
   - 다관점 검색 결과 분석
   - 성능 개선 방향 도출

### 📝 주요 실험 결과
```python
# 최종 평가 결과 저장
print("💾 결과 저장:")
try:
    results_df.to_csv('multiquery_evaluation_results.csv')
    print("   ✅ 결과가 'multiquery_evaluation_results.csv'에 저장되었습니다.")
except Exception as e:
    print(f"   ❌ 저장 실패: {e}")

print("\n🎉 실습 6 완료!")
```

### 🔍 핵심 학습 내용
1. **쿼리 확장의 중요성**: 원본 질문의 한계를 극복하고 더 포괄적인 검색 수행
2. **기법별 특성 이해**: 각 기법의 장단점과 적용 상황 파악
3. **실무 적용 방법론**: 상황에 맞는 기법 선택과 하이브리드 접근법

### 💡 실무 적용 가이드
- **단순 질문**: Query Reformulation
- **탐색적 검색**: MultiQuery
- **복합 질문**: Decomposition
- **배경 지식 필요**: Step-Back
- **정밀 검색**: HyDE

---

**💡 핵심 인사이트**: 쿼리 확장은 단순한 기법의 조합이 아니라, 사용자의 정보 요구를 정확히 파악하고 이를 검색 시스템이 이해할 수 있는 형태로 변환하는 지능적 중개 과정입니다. 각 기법의 특성을 이해하고 상황에 맞게 적절히 조합하여 사용하는 것이 핵심입니다. 본 실습에서는 실제 한국어 전기차 데이터를 대상으로 각 기법의 성능을 체계적으로 측정하고 분석했습니다.