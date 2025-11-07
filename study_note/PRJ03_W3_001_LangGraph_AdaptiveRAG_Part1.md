# LangGraph 활용 - Adaptive RAG (Part1)

## 📚 학습 목표

이 학습 가이드를 통해 다음을 달성할 수 있습니다:

1. **Adaptive RAG의 개념과 동작 원리**를 이해하고 설명할 수 있다
2. **쿼리 분석 및 라우팅 전략**을 구현하여 최적의 응답 방식을 선택할 수 있다
3. **3가지 RAG 전략**(No Retrieval, Single-shot RAG, Iterative RAG)을 구현하고 적절히 활용할 수 있다
4. **Pydantic 구조화 출력**을 사용하여 LLM 응답의 타입 안정성을 확보할 수 있다
5. **반복적 검색 및 쿼리 개선** 기법을 활용하여 복잡한 질문에 대한 정확도를 높일 수 있다
6. **응답 품질 평가 시스템**을 구축하여 RAG 시스템의 성능을 측정하고 개선할 수 있다
7. **다중 벡터 데이터베이스 통합**을 통해 여러 지식 소스를 활용하는 시스템을 구축할 수 있다

## 🔑 핵심 개념

### Adaptive RAG란?

**Adaptive RAG (적응형 검색 증강 생성)**는 쿼리의 특성을 분석하여 가장 적합한 검색 및 응답 전략을 자동으로 선택하는 지능형 RAG 시스템입니다.

![Adaptive RAG 개념도](https://arxiv.org/abs/2403.14403)

**논문 참조**: [Adaptive RAG Paper](https://arxiv.org/abs/2403.14403)

### 핵심 구성 요소

#### 1. **쿼리 분석 (Query Analysis)**
- 입력된 질문의 복잡성, 도메인, 필요 지식 수준을 평가
- Pydantic 모델을 사용한 구조화된 라우팅 결정
- 신뢰도 점수와 데이터베이스 선택 정보 포함

#### 2. **3가지 RAG 전략**

| 전략 | 설명 | 적합한 질문 유형 | 장점 | 단점 |
|------|------|-----------------|------|------|
| **No Retrieval** | 검색 없이 LLM 내장 지식으로 응답 | 일반 상식, 간단한 인사, 레스토랑 무관 질문 | 빠른 응답, 리소스 절약 | 최신 정보 부족, 도메인 특화 지식 제한 |
| **Single-shot RAG** | 1회 검색 후 응답 생성 | 단순 사실 확인, 정의, 메뉴 조회 | 효율적, 빠른 속도 | 복잡한 질문에 부족 |
| **Iterative RAG** | 반복 검색 및 쿼리 개선 | 복합 분석, 페어링 추천, 비교 질문 | 높은 정확도, 다단계 추론 | 느린 속도, 높은 비용 |

#### 3. **주요 기술 패턴**

**Pydantic 구조화 출력**
```python
class QueryRoute(BaseModel):
    strategy: Literal["no_retrieval", "single_lookup", "iterative"]
    reason: str
    confidence: float  # 0-1 사이 신뢰도
    suggested_databases: list[str]  # 검색할 DB 목록
```

**다중 벡터 데이터베이스**
- Menu DB: 레스토랑 메뉴 정보
- Wine DB: 와인 정보
- 동시 검색 및 통합 처리

**반복적 쿼리 개선**
- 초기 검색 결과 분석
- 쿼리 재작성 및 재검색
- 최대 반복 횟수 제어

### 관련 기술 스택

- **LangChain**: RAG 체인 구성 및 프롬프트 관리
- **LangGraph**: 상태 기반 워크플로우 (Part2에서 다룸)
- **Pydantic**: 타입 안전한 데이터 검증
- **ChromaDB**: 벡터 저장소
- **OpenAI**: 임베딩 및 LLM
- **Langfuse**: 모니터링 및 추적

### 배경 지식

이 가이드를 학습하기 전에 다음 내용을 이해하고 있어야 합니다:

- **RAG 기본 개념**: 검색 증강 생성의 기본 원리
- **벡터 데이터베이스**: 임베딩 및 유사도 검색
- **LangChain 기초**: 체인 구성 및 프롬프트 템플릿
- **Python 타입 힌팅**: Pydantic 모델 활용을 위한 기초

## 🛠 환경 설정

### 필요한 라이브러리 설치

```bash
# LangChain 생태계
pip install langchain langchain-openai langchain-chroma langchain-core

# Pydantic (타입 검증)
pip install pydantic

# 벡터 데이터베이스
pip install chromadb

# 모니터링 (선택사항)
pip install langfuse

# 기타 유틸리티
pip install python-dotenv
```

### API 키 설정

`.env` 파일을 생성하고 다음 API 키를 설정합니다:

```bash
# OpenAI API 키 (필수)
OPENAI_API_KEY=your_openai_api_key_here

# Langfuse 추적 (선택사항)
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

# Langfuse 콜백 핸들러 설정 (선택사항)
from langfuse.langchain import CallbackHandler
langfuse_handler = CallbackHandler()
```

### 벡터 데이터베이스 준비

이 가이드에서는 사전에 준비된 벡터 데이터베이스를 사용합니다:

- **Menu DB**: `./chroma_db/restaurant_menu` - 레스토랑 메뉴 정보
- **Wine DB**: `./chroma_db/restaurant_wine` - 와인 정보

> **참고**: 벡터 DB 생성 방법은 Week 2의 RAG 구현 가이드를 참조하세요.

## 💻 단계별 구현

### 단계 1: 쿼리 분석 및 라우팅 (AnalyzeQuery)

쿼리 분석은 입력된 질문을 분석하여 최적의 RAG 전략을 선택하는 핵심 단계입니다.

#### 1.1 기본 라우팅 구현

```python
from typing import Literal
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class QueryRoute(BaseModel):
    """쿼리 라우팅 결정 구조"""
    strategy: Literal["no_retrieval", "single_lookup", "iterative"]
    reason: str

def route_query(query: str) -> QueryRoute:
    """쿼리를 분석하여 적절한 RAG 전략을 선택하는 함수"""

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

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    chain = router_prompt | llm.with_structured_output(QueryRoute)
    routing = chain.invoke({"query": query})

    return routing

# 테스트
test_queries = [
    "파스타 메뉴가 있나요?",
    "스테이크와 어울리는 와인을 추천해주세요.",
    "프랑스의 수도는 어디인가요?",
]

for query in test_queries:
    routing = route_query(query)
    print(f"쿼리: {query}")
    print(f"라우팅: {routing}")
    print("-" * 100)
```

**실행 결과**:
```
쿼리: 파스타 메뉴가 있나요?
라우팅: strategy='single_lookup' reason='파스타 메뉴의 유무를 확인하는 단순 사실 확인 질문이므로, 메뉴 DB에서 한번 검색하면 충분하다.'
----------------------------------------------------------------------------------------------------
쿼리: 스테이크와 어울리는 와인을 추천해주세요.
라우팅: strategy='single_lookup' reason='스테이크와 잘 어울리는 와인 종류를 데이터베이스에서 한 번 조회하여 추천할 수 있기 때문입니다.'
----------------------------------------------------------------------------------------------------
쿼리: 프랑스의 수도는 어디인가요?
라우팅: strategy='no_retrieval' reason='프랑스의 수도는 일반 상식에 해당하는 정보로, 별도의 데이터 검색 없이 답변할 수 있습니다.'
```

#### 1.2 개선된 라우팅 (신뢰도 및 DB 선택 포함)

```python
from pydantic import BaseModel, Field

class QueryRoute(BaseModel):
    """개선된 쿼리 라우팅 구조"""
    strategy: Literal["no_retrieval", "single_lookup", "iterative"]
    reason: str
    confidence: float = Field(..., description="라우팅 결정의 신뢰도 (0-1)")
    suggested_databases: list[str] = Field(
        default_factory=list,
        description="검색할 데이터베이스 목록"
    )

def improved_route_query(query: str) -> QueryRoute:
    """개선된 쿼리 분석 및 라우팅 함수"""

    router_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        당신은 레스토랑 서비스 전문가입니다.
        두 개의 데이터베이스를 보유하고 있습니다:
        1. menu_db: 레스토랑 메뉴 정보 (음식, 가격, 재료, 조리법 등)
        2. wine_db: 와인 정보 (종류, 특성, 페어링 추천 등)

        질문을 분석하여 최적의 전략과 데이터베이스를 선택하세요:

        [전략 선택 기준]
        1. no_retrieval:
           - 일반 상식, 간단한 인사, 레스토랑과 무관한 질문
           - 예: "안녕하세요", "프랑스의 수도는?", "1+1은?"

        2. single_lookup:
           - 단일 정보 검색으로 답변 가능
           - 단순 메뉴 확인, 가격 문의, 특정 와인 정보
           - 예: "파스타 있나요?", "샤르도네 와인 있나요?"

        3. iterative:
           - 복잡한 추론과 다중 검색 필요
           - 음식-와인 페어링, 비교 분석, 추천
           - 예: "스테이크와 어울리는 와인 추천", "채식 메뉴와 와인 조합"

        [데이터베이스 선택]
        - menu_db: 메뉴, 음식, 요리 관련 질문
        - wine_db: 와인, 음료 관련 질문
        - 둘 다: 페어링, 조합, 추천 질문
        """),
        ("user", "{query}")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    chain = router_prompt | llm.with_structured_output(QueryRoute)
    routing = chain.invoke({"query": query})

    return routing

# 개선된 라우팅 테스트
test_queries = [
    "파스타 메뉴가 있나요?",
    "스테이크와 어울리는 와인을 추천해주세요.",
    "프랑스의 수도는 어디인가요?",
    "채식주의자를 위한 메뉴와 와인 페어링을 추천해주세요.",
    "가장 비싼 메뉴는 무엇인가요?",
]

print("=" * 100)
print("개선된 쿼리 라우팅 테스트")
print("=" * 100)

for query in test_queries:
    routing = improved_route_query(query)
    print(f"\n쿼리: {query}")
    print(f"전략: {routing.strategy}")
    print(f"신뢰도: {routing.confidence:.2f}")
    print(f"데이터베이스: {routing.suggested_databases}")
    print(f"이유: {routing.reason}")
    print("-" * 100)
```

**핵심 포인트**:
- `with_structured_output()`: Pydantic 모델로 자동 파싱
- `confidence`: 라우팅 결정의 신뢰도 (0-1)
- `suggested_databases`: 어떤 DB를 검색할지 명시

### 단계 2: NoRetrieval 전략 구현

검색 없이 LLM의 내장 지식만으로 응답하는 전략입니다.

#### 2.1 기본 NoRetrieval 구현

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def no_retrieval_response(query: str) -> str:
    """외부 지식 없이 질문에 직접 답변하는 함수"""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 도움이 되는 인공지능 어시스턴트입니다. 외부 지식을 사용하지 않고 직접 질문에 답변하세요."),
        ("user", "{query}"),
    ])

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"query": query})

    return response

# 테스트 실행
response = no_retrieval_response("스테이크와 어울리는 와인을 추천해주세요.")
print(response)
```

**실행 결과**:
```
스테이크와 어울리는 와인은 일반적으로 풍부한 바디와 탄닌이 있는 레드 와인이 잘 어울립니다. 대표적으로는 다음과 같은 와인들이 있습니다:

1. 카베르네 소비뇽 (Cabernet Sauvignon) – 강한 탄닌과 진한 과일 맛이 스테이크의 풍미를 잘 살려줍니다.
2. 멜롯 (Merlot) – 부드럽고 과일향이 풍부해 부담 없이 즐길 수 있습니다.
3. 쉬라즈/시라 (Shiraz/Syrah) – 스파이시하고 진한 맛이 스테이크와 잘 어울립니다.
4. 말벡 (Malbec) – 진한 다크 베리 향과 부드러운 탄닌으로 고기 요리와 좋은 조합입니다.
```

#### 2.2 개선된 NoRetrieval (레스토랑 컨텍스트 포함)

```python
from textwrap import dedent

def improved_no_retrieval_response(query: str) -> dict:
    """개선된 외부 지식 없이 질문에 직접 답변하는 함수"""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    prompt = ChatPromptTemplate.from_messages([
        ("system", dedent("""
            당신은 친절한 레스토랑 서비스 전문가입니다.

            [역할]
            - 고객의 일반적인 질문에 친절하고 전문적으로 답변합니다
            - 레스토랑 맥락에서 벗어난 질문도 정중하게 응대합니다
            - 필요시 레스토랑 정보로 자연스럽게 유도합니다

            [가이드라인]
            1. 간결하고 명확하게 답변하세요
            2. 전문적이면서도 친근한 톤을 유지하세요
            3. 레스토랑과 무관한 질문에도 정중하게 응답하세요
            4. 필요시 "메뉴나 와인에 대해 더 궁금하신 점이 있으시면 말씀해주세요"와 같은 안내를 추가하세요
        """)),
        ("user", "{query}"),
    ])

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"query": query})

    return {
        "answer": response,
        "strategy": "no_retrieval",
        "source": "LLM 내장 지식"
    }

# 테스트 실행
test_queries = [
    "안녕하세요!",
    "프랑스의 수도는 어디인가요?",
    "와인과 맥주의 차이는 무엇인가요?",
    "스테이크는 어떻게 먹는 게 좋나요?",
]

print("=" * 100)
print("개선된 NoRetrieval 테스트")
print("=" * 100)

for query in test_queries:
    print(f"\n질문: {query}")
    result = improved_no_retrieval_response(query)
    print(f"답변: {result['answer']}")
    print(f"전략: {result['strategy']}")
    print("-" * 100)
```

**핵심 포인트**:
- 레스토랑 컨텍스트를 유지하면서 일반 질문에도 응답
- 구조화된 반환값 (답변, 전략, 출처)
- 친절하고 전문적인 톤 유지

### 단계 3: SingleShotRAG 전략 구현

1회 검색으로 답변을 생성하는 효율적인 RAG 전략입니다.

#### 3.1 벡터 데이터베이스 로드

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# 임베딩 모델 설정
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

# 검색 도구 생성
menu_retriever = menu_db.as_retriever(search_kwargs={"k": 3})
wine_retriever = wine_db.as_retriever(search_kwargs={"k": 3})

# 검색 테스트
query = "스테이크와 어울리는 와인을 추천해주세요."
menu_results = menu_retriever.invoke(query)
print(f"검색된 메뉴 문서 수: {len(menu_results)}")
```

#### 3.2 검색 문서 포맷팅

```python
from langchain_core.documents import Document

def format_docs(docs: list[Document]) -> str:
    """문서 출처를 함께 표시하는 포맷 변환 함수"""
    formatted_result = ""
    for doc in docs:
        formatted_result += f"{doc.page_content}\n"
        formatted_result += f"(출처: {doc.metadata['source']} - "
        formatted_result += f"[{doc.metadata['menu_number']}]{doc.metadata['menu_name']})\n"
        formatted_result += "-" * 80 + "\n"

    return formatted_result

# 테스트
if menu_results:
    print(format_docs(menu_results))
```

#### 3.3 RAG 체인 구성

```python
from langchain_core.vectorstores import VectorStore
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from operator import itemgetter

def create_rag_chain(vectorstore: VectorStore, k: int = 3) -> RunnableParallel:
    """RAG 체인을 생성하는 함수"""

    # Retriever 설정
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    # RAG 프롬프트 템플릿
    prompt = ChatPromptTemplate.from_messages([
        ("system", dedent("""
            주어진 컨텍스트를 사용하여 질문에 답변하세요.

            [가이드라인]
            1. 컨텍스트에 정보가 없거나 부족하면 '근거가 없습니다'라고 답변하세요.
            2. 답변을 할 때는 참조한 출처 또는 근거를 표시합니다. (출처: 파일경로 또는 URL - [메뉴번호]메뉴이름)
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
    """단일 질문에 대한 RAG 기반 답변을 생성하는 함수"""

    # RAG 체인 생성 및 실행
    chain = create_rag_chain(vectorstore)
    response = chain.invoke(query)

    return response

# 테스트 실행
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

#### 3.4 개선된 SingleShotRAG (다중 DB 지원)

```python
from typing import List, Dict

def improved_format_docs(docs: List[Document]) -> str:
    """개선된 문서 포맷팅 함수"""
    if not docs:
        return "검색된 정보가 없습니다."

    formatted_result = ""
    for i, doc in enumerate(docs, 1):
        formatted_result += f"[문서 {i}]\n"
        formatted_result += f"{doc.page_content}\n"

        # 메타데이터가 있는 경우에만 출처 표시
        if doc.metadata:
            source = doc.metadata.get('source', '알 수 없음')
            menu_name = doc.metadata.get('menu_name', '')
            menu_number = doc.metadata.get('menu_number', '')

            if menu_name and menu_number:
                formatted_result += f"(출처: {source} - [{menu_number}] {menu_name})\n"
            elif source != '알 수 없음':
                formatted_result += f"(출처: {source})\n"

        formatted_result += "-" * 80 + "\n"

    return formatted_result

def improved_single_rag_response(
    query: str,
    databases: Dict[str, VectorStore],
    k: int = 5
) -> dict:
    """개선된 단일 RAG 응답 생성 함수 - 다중 데이터베이스 지원"""

    # 다중 데이터베이스에서 검색
    all_docs = []
    for db_name, vectorstore in databases.items():
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query)
        # 데이터베이스 출처 정보 추가
        for doc in docs:
            doc.metadata['database'] = db_name
        all_docs.extend(docs)

    # 상위 k개만 선택 (관련성 높은 문서)
    all_docs = all_docs[:k]

    # RAG 프롬프트 구성
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    prompt = ChatPromptTemplate.from_messages([
        ("system", dedent("""
            당신은 레스토랑 서비스 전문가입니다.
            주어진 컨텍스트를 활용하여 고객의 질문에 답변하세요.

            [가이드라인]
            1. 컨텍스트에 정보가 있으면 이를 기반으로 구체적으로 답변하세요
            2. 컨텍스트에 정보가 부족하거나 없으면 "죄송하지만 해당 정보를 찾을 수 없습니다"라고 답변하세요
            3. 답변할 때는 참조한 출처를 명시하세요
            4. 여러 옵션이 있다면 고객이 선택할 수 있도록 나열하세요
            5. 친절하고 전문적인 톤을 유지하세요
        """)),
        ("user", "[컨텍스트]\n{context}\n\n[질문]\n{question}\n\n[답변]\n"),
    ])

    # 컨텍스트 생성
    context = improved_format_docs(all_docs)

    # 체인 실행
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": query})

    return {
        "answer": answer,
        "docs": all_docs,
        "strategy": "single_shot_rag",
        "num_docs": len(all_docs)
    }

# 테스트 실행
databases = {
    "menu_db": menu_db,
    "wine_db": wine_db
}

test_queries = [
    "파스타 메뉴가 있나요?",
    "레드 와인 추천해주세요",
    "스테이크와 어울리는 와인은 무엇인가요?",
]

print("=" * 100)
print("개선된 SingleShotRAG 테스트")
print("=" * 100)

for query in test_queries:
    print(f"\n질문: {query}")
    result = improved_single_rag_response(query, databases, k=3)
    print(f"답변: {result['answer']}")
    print(f"검색된 문서 수: {result['num_docs']}")
    print(f"전략: {result['strategy']}")
    print("-" * 100)
```

**핵심 포인트**:
- **다중 DB 통합**: Menu DB와 Wine DB를 동시에 검색
- **메타데이터 추가**: 각 문서에 DB 출처 정보 포함
- **상위 k개 선택**: 관련성 높은 문서만 사용
- **컨텍스트 기반 응답**: 정보가 없으면 명확히 안내

### 단계 4: IterativeRAG 전략 구현

반복적인 검색과 쿼리 개선을 통해 복잡한 질문에 정확하게 답변하는 전략입니다.

#### 4.1 쿼리 개선 체인

```python
def refine_query(query: str) -> str:
    """쿼리 개선을 위한 체인을 생성하는 함수"""

    # 쿼리 개선 프롬프트
    query_improvement_prompt = ChatPromptTemplate.from_messages([
        ("system", "원래 쿼리를 분석하고 검색 쿼리를 개선하세요"),
        ("user", "[쿼리]{query}\n\n[개선된 쿼리]\n")
    ])

    # 쿼리 개선 체인 생성
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    query_improvement_chain = query_improvement_prompt | llm | StrOutputParser()

    # 쿼리 개선
    improved_query = query_improvement_chain.invoke({"query": query})

    return improved_query

# 쿼리 개선 테스트
query = "스테이크와 어울리는 와인을 추천해주세요."
improved_query = refine_query(query)
print(f"원래 쿼리: {query}")
print(f"개선된 쿼리: {improved_query}")
```

#### 4.2 반복적 RAG 구현

```python
def iterative_rag_response(
    query: str,
    vectorstore: VectorStore,
    k: int = 3,
    max_iterations: int = 3
) -> tuple:
    """반복적인 RAG 기반 답변 생성을 위한 함수"""

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

        # 근거가 없는 경우 다음 반복으로
        if len(intermediate_docs) == 0 or "근거가 없습니다" in intermediate_response:
            continue
        else:
            # 중간 결과 저장
            intermediate_responses.append(intermediate_response)
            for doc in intermediate_docs:
                if doc not in retrieved_docs:
                    retrieved_docs.append(doc)

    return intermediate_responses, retrieved_docs

# 반복적인 RAG 기반 답변 생성 테스트
query = "스테이크와 어울리는 와인을 추천해주세요."
responses, docs = iterative_rag_response(query, wine_db)

print("\n생성된 답변:")
for i, response in enumerate(responses, 1):
    print(f"\n[반복 {i}]")
    print(response)

print(f"\n검색된 문서 수: {len(docs)}")
```

#### 4.3 개선된 IterativeRAG (전략적 쿼리 개선)

```python
from pydantic import BaseModel

class QueryRefinement(BaseModel):
    """쿼리 개선 결과 구조"""
    refined_query: str
    search_focus: str
    reasoning: str

def improved_refine_query(
    original_query: str,
    iteration: int,
    previous_results: List[str] = None
) -> QueryRefinement:
    """개선된 쿼리 정제 함수 - 이전 결과를 고려한 전략적 쿼리 생성"""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

    # 이전 결과 컨텍스트 생성
    context = ""
    if previous_results and iteration > 1:
        context = f"\n[이전 검색 결과]\n" + "\n".join(previous_results[:2])  # 최근 2개만

    prompt = ChatPromptTemplate.from_messages([
        ("system", dedent("""
            당신은 검색 쿼리 최적화 전문가입니다.

            [역할]
            원래 질문의 의도를 유지하면서 더 나은 검색 결과를 얻을 수 있도록 쿼리를 개선합니다.

            [개선 전략]
            1. 첫 번째 반복: 핵심 키워드 추출 및 구체화
            2. 두 번째 반복: 관련 개념 확장 (예: 스테이크 → 고기 요리, 레드와인)
            3. 세 번째 반복: 대안 표현 및 유사 개념 (예: 추천 → 페어링, 어울리는)

            [가이드라인]
            - 간결하고 검색에 적합한 형태로 변환
            - 불필요한 조사나 문장 구조 제거
            - 핵심 검색어 중심으로 재구성
        """)),
        ("user", f"[원래 질문]\n{original_query}\n{context}\n\n[반복 횟수]\n{iteration}\n\n개선된 쿼리를 생성하세요.")
    ])

    chain = prompt | llm.with_structured_output(QueryRefinement)
    result = chain.invoke({"query": original_query, "iteration": iteration})

    return result

def improved_iterative_rag_response(
    query: str,
    databases: Dict[str, VectorStore],
    k: int = 3,
    max_iterations: int = 3,
    min_docs_threshold: int = 1
) -> Dict:
    """개선된 반복적 RAG 응답 생성 함수"""

    # RAG 체인 준비
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    prompt = ChatPromptTemplate.from_messages([
        ("system", dedent("""
            당신은 레스토랑 서비스 전문가입니다.
            주어진 컨텍스트를 활용하여 질문에 답변하세요.

            [가이드라인]
            1. 컨텍스트 정보를 최대한 활용하여 구체적으로 답변
            2. 정보가 충분하지 않으면 그 사실을 명시
            3. 출처를 명확히 표시
        """)),
        ("user", "[컨텍스트]\n{context}\n\n[질문]\n{question}\n\n[답변]")
    ])

    # 반복 검색 수행
    all_responses = []
    all_docs = []
    search_queries = [query]
    current_query = query

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'='*80}")
        print(f"[반복 {iteration}/{max_iterations}]")
        print(f"검색 쿼리: {current_query}")
        print(f"{'='*80}")

        # 다중 데이터베이스 검색
        iteration_docs = []
        for db_name, vectorstore in databases.items():
            retriever = vectorstore.as_retriever(search_kwargs={"k": k})
            docs = retriever.invoke(current_query)
            for doc in docs:
                doc.metadata['database'] = db_name
                doc.metadata['iteration'] = iteration
            iteration_docs.extend(docs)

        # 중복 제거
        unique_docs = []
        seen_contents = set()
        for doc in iteration_docs:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_docs.append(doc)
                if doc not in all_docs:
                    all_docs.append(doc)

        print(f"검색된 고유 문서 수: {len(unique_docs)}")

        # 문서가 있으면 답변 생성
        if unique_docs:
            context = improved_format_docs(unique_docs)
            chain = prompt | llm | StrOutputParser()
            answer = chain.invoke({"context": context, "question": query})
            all_responses.append({
                "iteration": iteration,
                "query": current_query,
                "answer": answer,
                "num_docs": len(unique_docs)
            })
            print(f"답변 생성 완료 (문서 {len(unique_docs)}개 활용)")
        else:
            print(f"검색 결과 없음")

        # 충분한 문서를 찾았으면 조기 종료
        if len(all_docs) >= min_docs_threshold * max_iterations:
            print(f"\n충분한 정보 수집 완료 ({len(all_docs)}개 문서)")
            break

        # 다음 반복을 위한 쿼리 개선
        if iteration < max_iterations:
            previous_answers = [r["answer"] for r in all_responses]
            refinement = improved_refine_query(query, iteration + 1, previous_answers)
            current_query = refinement.refined_query
            search_queries.append(current_query)
            print(f"다음 검색 전략: {refinement.search_focus}")
            print(f"개선 근거: {refinement.reasoning}")

    # 최종 통합 답변 생성
    if all_responses:
        combined_context = "\n\n".join([
            f"[검색 {r['iteration']}] {r['answer']}"
            for r in all_responses
        ])

        final_prompt = ChatPromptTemplate.from_messages([
            ("system", dedent("""
                당신은 레스토랑 서비스 전문가입니다.
                여러 검색 결과를 통합하여 최종 답변을 생성하세요.

                [가이드라인]
                1. 중복 정보는 제거하고 핵심만 정리
                2. 가장 관련성 높은 정보 우선 제시
                3. 구체적이고 실용적인 답변 제공
                4. 자연스럽고 일관된 문장으로 통합
            """)),
            ("user", f"[질문]\n{query}\n\n[수집된 정보]\n{combined_context}\n\n[최종 통합 답변]")
        ])

        final_chain = final_prompt | llm | StrOutputParser()
        final_answer = final_chain.invoke({})
    else:
        final_answer = "죄송하지만 관련 정보를 찾을 수 없습니다."

    return {
        "final_answer": final_answer,
        "intermediate_responses": all_responses,
        "all_docs": all_docs,
        "search_queries": search_queries,
        "total_iterations": len(all_responses),
        "total_docs": len(all_docs)
    }

# 테스트 실행
print("=" * 100)
print("개선된 IterativeRAG 테스트")
print("=" * 100)

query = "스테이크와 어울리는 와인을 추천해주세요"
result = improved_iterative_rag_response(
    query,
    databases={"menu_db": menu_db, "wine_db": wine_db},
    k=2,
    max_iterations=3
)

print(f"\n\n{'='*100}")
print("최종 결과")
print(f"{'='*100}")
print(f"\n질문: {query}")
print(f"\n최종 답변:\n{result['final_answer']}")
print(f"\n총 반복 횟수: {result['total_iterations']}")
print(f"수집된 총 문서 수: {result['total_docs']}")
print(f"\n검색 쿼리 변화:")
for i, q in enumerate(result['search_queries'], 1):
    print(f"  {i}. {q}")
```

**핵심 포인트**:
- **전략적 쿼리 개선**: 반복마다 다른 전략 적용
- **중복 제거**: 같은 내용의 문서는 1번만 포함
- **조기 종료**: 충분한 정보 수집 시 중단
- **최종 통합**: 모든 중간 답변을 하나로 통합

### 단계 5: GenerateResponse - 최종 응답 생성 및 품질 평가

수집된 정보를 바탕으로 최종 답변을 생성하고 품질을 평가합니다.

#### 5.1 최종 답변 생성

```python
def generate_response(contexts: List[str], query: str) -> str:
    """수집된 컨텍스트를 통합하여 최종 답변 생성"""

    context = "\n\n".join(contexts)

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
if 'result' in locals() and result.get('intermediate_responses'):
    contexts = [r['answer'] for r in result['intermediate_responses']]
    query = "스테이크와 어울리는 와인을 추천해주세요."

    response = generate_response(contexts, query)
    print(f"질문: {query}")
    print(f"\n최종 답변:\n{response}")
```

#### 5.2 답변 품질 평가

```python
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

# 응답 품질 평가 테스트
if 'result' in locals() and 'response' in locals():
    quality = evaluate_response(
        query=query,
        response=response,
        retrieved_docs=[doc.page_content for doc in result['all_docs']]
    )

    print(f"\n{'='*80}")
    print("답변 품질 평가")
    print(f"{'='*80}")
    print(f"관련성: {quality.relevance}")
    print(f"완성도: {quality.completeness}")
    print(f"일관성: {quality.consistency}")
    print(f"종합점수: {quality.overall_score}")
    print(f"설명: {quality.explanation}")
```

#### 5.3 개선된 GenerateResponse 및 평가 시스템

```python
class ImprovedResponseQuality(BaseModel):
    """개선된 응답 품질 평가 메트릭"""
    relevance: float = Field(..., description="검색된 문서와 질문의 관련성 점수 (0-1)")
    completeness: float = Field(..., description="답변의 완성도 점수 (0-1)")
    consistency: float = Field(..., description="답변의 일관성 점수 (0-1)")
    accuracy: float = Field(..., description="정보의 정확성 점수 (0-1)")
    helpfulness: float = Field(..., description="고객에게 유용한 정도 (0-1)")
    overall_score: float = Field(..., description="종합 점수 (0-1)")
    strengths: List[str] = Field(default_factory=list, description="답변의 강점")
    weaknesses: List[str] = Field(default_factory=list, description="답변의 약점")
    suggestions: List[str] = Field(default_factory=list, description="개선 제안")
    explanation: str = Field(..., description="전체 평가에 대한 상세 설명")

def improved_generate_response(
    contexts: List[str],
    query: str,
    strategy: str = "unknown"
) -> Dict:
    """개선된 최종 응답 생성 함수"""

    if not contexts or all(not c.strip() for c in contexts):
        return {
            "answer": "죄송하지만 해당 질문에 대한 충분한 정보를 찾을 수 없습니다. 다른 방식으로 질문해주시거나, 담당 직원에게 문의해주세요.",
            "strategy": strategy,
            "context_available": False
        }

    # 컨텍스트 결합 및 정리
    combined_context = "\n\n".join([
        f"[정보 {i+1}]\n{ctx}"
        for i, ctx in enumerate(contexts) if ctx.strip()
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", dedent("""
            당신은 레스토랑 서비스 전문가입니다.
            수집된 정보를 바탕으로 고객에게 최고의 답변을 제공하세요.

            [답변 작성 가이드라인]
            1. **구조화**: 정보를 논리적으로 구조화하여 제시
            2. **구체성**: 구체적인 예시와 세부 정보 포함
            3. **실용성**: 고객이 실제로 활용할 수 있는 조언 제공
            4. **친절함**: 따뜻하고 전문적인 톤 유지
            5. **완결성**: 추가 질문 없이도 이해할 수 있는 완전한 답변

            [답변 형식]
            - 핵심 답변을 먼저 제시
            - 필요시 옵션을 나열하여 선택 가능하게 제공
            - 관련된 추가 정보나 팁 포함
            - 출처가 명확한 경우 간략히 언급
        """)),
        ("user", dedent("""
            [고객 질문]
            {question}

            [수집된 정보]
            {context}

            [사용된 전략]
            {strategy}

            위 정보를 바탕으로 최적의 답변을 작성하세요.
        """))
    ])

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": combined_context,
        "question": query,
        "strategy": strategy
    })

    return {
        "answer": answer,
        "strategy": strategy,
        "context_available": True,
        "num_contexts": len([c for c in contexts if c.strip()])
    }

def improved_evaluate_response(
    query: str,
    response: str,
    retrieved_docs: List[Document],
    strategy: str = "unknown"
) -> ImprovedResponseQuality:
    """개선된 응답 품질 평가 함수"""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 문서 내용 요약
    docs_summary = "\n".join([
        f"- {doc.page_content[:200]}..." if len(doc.page_content) > 200
        else f"- {doc.page_content}"
        for doc in retrieved_docs[:5]  # 최대 5개만
    ]) if retrieved_docs else "문서 없음"

    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", dedent("""
            당신은 RAG 시스템 품질 평가 전문가입니다.
            레스토랑 고객 서비스 관점에서 답변의 품질을 평가하세요.

            [평가 기준]

            1. **Relevance (관련성)**: 0.0-1.0
               - 답변이 질문과 얼마나 관련되어 있는가?
               - 검색된 문서가 질문과 관련성이 있는가?

            2. **Completeness (완성도)**: 0.0-1.0
               - 답변이 질문에 충분히 답하고 있는가?
               - 추가 질문 없이 이해 가능한가?

            3. **Consistency (일관성)**: 0.0-1.0
               - 답변 내용이 서로 모순되지 않는가?
               - 검색된 문서와 답변이 일치하는가?

            4. **Accuracy (정확성)**: 0.0-1.0
               - 제공된 정보가 정확한가?
               - 출처가 명확한가?

            5. **Helpfulness (유용성)**: 0.0-1.0
               - 고객에게 실제로 도움이 되는가?
               - 실행 가능한 정보를 제공하는가?

            [종합 점수 계산]
            overall_score = (relevance * 0.25 + completeness * 0.25 +
                           consistency * 0.2 + accuracy * 0.15 + helpfulness * 0.15)

            [평가 결과]
            - 각 항목별 점수와 근거 제시
            - 강점 2-3개 나열
            - 약점 1-2개 나열 (있는 경우)
            - 구체적인 개선 제안 1-2개
        """)),
        ("user", dedent("""
            [질문]
            {query}

            [생성된 답변]
            {response}

            [검색된 문서 요약]
            {docs}

            [사용된 전략]
            {strategy}

            위 정보를 바탕으로 답변의 품질을 평가하세요.
        """))
    ])

    chain = eval_prompt | llm.with_structured_output(ImprovedResponseQuality)

    scores = chain.invoke({
        "query": query,
        "response": response,
        "docs": docs_summary,
        "strategy": strategy
    })

    return scores

# 통합 테스트
print("=" * 100)
print("개선된 GenerateResponse 및 평가 체인 통합 테스트")
print("=" * 100)

# IterativeRAG 결과를 활용한 테스트
if 'result' in locals() and result.get('intermediate_responses'):
    test_query = "스테이크와 어울리는 와인을 추천해주세요"

    # 중간 답변들을 컨텍스트로 사용
    contexts = [r['answer'] for r in result['intermediate_responses']]

    # 최종 답변 생성
    print(f"\n질문: {test_query}")
    print(f"수집된 컨텍스트 수: {len(contexts)}")

    final_response = improved_generate_response(
        contexts=contexts,
        query=test_query,
        strategy="iterative_rag"
    )

    print(f"\n{'='*80}")
    print("최종 생성 답변")
    print(f"{'='*80}")
    print(final_response['answer'])

    # 품질 평가
    if result.get('all_docs'):
        print(f"\n\n{'='*80}")
        print("답변 품질 평가")
        print(f"{'='*80}")

        quality = improved_evaluate_response(
            query=test_query,
            response=final_response['answer'],
            retrieved_docs=result['all_docs'],
            strategy="iterative_rag"
        )

        print(f"\n[평가 점수]")
        print(f"  관련성 (Relevance): {quality.relevance:.2f}")
        print(f"  완성도 (Completeness): {quality.completeness:.2f}")
        print(f"  일관성 (Consistency): {quality.consistency:.2f}")
        print(f"  정확성 (Accuracy): {quality.accuracy:.2f}")
        print(f"  유용성 (Helpfulness): {quality.helpfulness:.2f}")
        print(f"  종합 점수: {quality.overall_score:.2f}")

        print(f"\n[강점]")
        for i, strength in enumerate(quality.strengths, 1):
            print(f"  {i}. {strength}")

        if quality.weaknesses:
            print(f"\n[약점]")
            for i, weakness in enumerate(quality.weaknesses, 1):
                print(f"  {i}. {weakness}")

        if quality.suggestions:
            print(f"\n[개선 제안]")
            for i, suggestion in enumerate(quality.suggestions, 1):
                print(f"  {i}. {suggestion}")

        print(f"\n[상세 설명]")
        print(f"  {quality.explanation}")
```

**핵심 포인트**:
- **다차원 품질 평가**: 관련성, 완성도, 일관성, 정확성, 유용성
- **가중치 적용**: 항목별 중요도에 따른 가중 평균
- **정성적 피드백**: 강점, 약점, 개선 제안
- **전략별 평가**: 사용된 RAG 전략에 따른 맞춤 평가

## 🎯 실습 문제

### 문제 1: 하이브리드 라우팅 시스템 구현 ⭐⭐⭐

**목표**: 쿼리 특성에 따라 동적으로 k값(검색 문서 수)을 조정하는 라우팅 시스템을 구현하세요.

**요구사항**:
1. 쿼리 복잡도에 따라 k값을 자동 조정 (simple: k=2, moderate: k=3, complex: k=5)
2. 신뢰도가 0.7 이하인 경우 사용자에게 확인 요청
3. 라우팅 결정 로깅 기능 포함

**힌트**:
```python
class DynamicQueryRoute(BaseModel):
    strategy: Literal["no_retrieval", "single_lookup", "iterative"]
    complexity: Literal["simple", "moderate", "complex"]
    suggested_k: int
    confidence: float
    # ... 추가 필드
```

### 문제 2: 다단계 품질 검증 시스템 ⭐⭐⭐⭐

**목표**: 답변 생성 후 품질이 기준에 미달하면 자동으로 재검색하는 시스템을 구현하세요.

**요구사항**:
1. 응답 품질이 0.7 미만이면 쿼리를 개선하여 재검색
2. 최대 3회까지 재시도
3. 각 시도마다 품질 점수 기록
4. 최종적으로 가장 높은 품질의 답변 선택

**힌트**:
```python
def auto_quality_improvement(
    query: str,
    databases: Dict[str, VectorStore],
    quality_threshold: float = 0.7,
    max_retries: int = 3
) -> Dict:
    # 품질 검증 및 재시도 로직 구현
    pass
```

### 문제 3: 멀티모달 RAG 시스템 확장 ⭐⭐⭐⭐⭐

**목표**: 텍스트뿐만 아니라 이미지 메타데이터를 함께 활용하는 RAG 시스템을 구현하세요.

**요구사항**:
1. 문서에 이미지 URL 메타데이터 포함
2. 이미지 관련 질문 감지 (예: "사진 보여줘", "어떻게 생겼나요")
3. 이미지 메타데이터를 포함한 응답 생성
4. 이미지가 있는 경우 마크다운 형식으로 삽입

**힌트**:
```python
class MultimodalDocument(Document):
    image_url: Optional[str] = None
    image_caption: Optional[str] = None

def detect_image_query(query: str) -> bool:
    # 이미지 관련 질문 감지
    pass
```

## ✅ 솔루션 예시

### 솔루션 1: 하이브리드 라우팅 시스템

```python
from typing import Literal, Optional
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DynamicQueryRoute(BaseModel):
    """동적 k값 조정을 포함한 라우팅 구조"""
    strategy: Literal["no_retrieval", "single_lookup", "iterative"]
    complexity: Literal["simple", "moderate", "complex"]
    suggested_k: int = Field(..., description="권장 검색 문서 수")
    confidence: float = Field(..., description="라우팅 신뢰도 (0-1)")
    reason: str
    suggested_databases: list[str] = Field(default_factory=list)
    user_confirmation_needed: bool = Field(
        default=False,
        description="사용자 확인 필요 여부"
    )

def dynamic_route_query(query: str) -> DynamicQueryRoute:
    """동적 k값 조정을 포함한 쿼리 라우팅"""

    router_prompt = ChatPromptTemplate.from_messages([
        ("system", dedent("""
            당신은 레스토랑 서비스 전문가입니다.

            질문을 분석하여 다음을 결정하세요:

            [전략 선택]
            - no_retrieval: 일반 상식 질문
            - single_lookup: 단순 정보 조회
            - iterative: 복합 분석 및 추천

            [복잡도 평가]
            - simple: 단일 개념, 명확한 답변 (k=2)
              예: "파스타 있나요?", "영업시간은?"
            - moderate: 2-3개 개념, 비교 필요 (k=3)
              예: "파스타 종류", "레드 와인 추천"
            - complex: 다중 개념, 추론 필요 (k=5)
              예: "채식 메뉴와 와인 페어링", "가격대별 추천"

            [신뢰도]
            - 0.9-1.0: 확신
            - 0.7-0.9: 높은 신뢰
            - 0.5-0.7: 중간 (사용자 확인 권장)
            - 0.0-0.5: 낮음 (사용자 확인 필수)
        """)),
        ("user", "{query}")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    chain = router_prompt | llm.with_structured_output(DynamicQueryRoute)

    routing = chain.invoke({"query": query})

    # 신뢰도가 0.7 이하면 확인 필요 플래그 설정
    if routing.confidence <= 0.7:
        routing.user_confirmation_needed = True

    # 로깅
    logger.info(f"쿼리: {query}")
    logger.info(f"라우팅: strategy={routing.strategy}, complexity={routing.complexity}, k={routing.suggested_k}")
    logger.info(f"신뢰도: {routing.confidence:.2f}, 확인필요: {routing.user_confirmation_needed}")

    return routing

def adaptive_rag_with_dynamic_k(
    query: str,
    databases: Dict[str, VectorStore]
) -> Dict:
    """동적 k값을 사용하는 적응형 RAG"""

    # 1. 라우팅 결정
    routing = dynamic_route_query(query)

    # 2. 사용자 확인이 필요한 경우
    if routing.user_confirmation_needed:
        print(f"\n⚠️ 라우팅 신뢰도가 낮습니다 (신뢰도: {routing.confidence:.2f})")
        print(f"추천 전략: {routing.strategy}")
        print(f"추천 복잡도: {routing.complexity} (k={routing.suggested_k})")

        # 실제 환경에서는 사용자 입력을 받을 수 있습니다
        # confirmation = input("이 설정으로 진행하시겠습니까? (y/n): ")
        # if confirmation.lower() != 'y':
        #     return {"error": "사용자가 취소했습니다"}

    # 3. 전략에 따른 실행
    if routing.strategy == "no_retrieval":
        return improved_no_retrieval_response(query)

    elif routing.strategy == "single_lookup":
        return improved_single_rag_response(
            query,
            databases,
            k=routing.suggested_k
        )

    elif routing.strategy == "iterative":
        return improved_iterative_rag_response(
            query,
            databases,
            k=routing.suggested_k,
            max_iterations=3
        )

# 테스트
test_queries = [
    "안녕하세요!",  # simple, no_retrieval
    "파스타 메뉴가 있나요?",  # simple, single_lookup, k=2
    "레드 와인 추천해주세요",  # moderate, single_lookup, k=3
    "채식주의자를 위한 메뉴와 와인 페어링 추천",  # complex, iterative, k=5
]

print("=" * 100)
print("동적 k값 조정 라우팅 테스트")
print("=" * 100)

for query in test_queries:
    print(f"\n{'='*80}")
    routing = dynamic_route_query(query)
    print(f"쿼리: {query}")
    print(f"전략: {routing.strategy}")
    print(f"복잡도: {routing.complexity}")
    print(f"권장 k: {routing.suggested_k}")
    print(f"신뢰도: {routing.confidence:.2f}")
    print(f"확인 필요: {routing.user_confirmation_needed}")
    print(f"이유: {routing.reason}")
```

### 솔루션 2: 다단계 품질 검증 시스템

```python
def auto_quality_improvement(
    query: str,
    databases: Dict[str, VectorStore],
    quality_threshold: float = 0.7,
    max_retries: int = 3
) -> Dict:
    """품질 기준 미달 시 자동 재시도하는 RAG 시스템"""

    attempts = []
    current_query = query

    for attempt in range(1, max_retries + 1):
        print(f"\n{'='*80}")
        print(f"시도 {attempt}/{max_retries}")
        print(f"쿼리: {current_query}")
        print(f"{'='*80}")

        # 1. RAG 실행
        response = improved_single_rag_response(
            current_query,
            databases,
            k=3
        )

        # 2. 품질 평가
        quality = improved_evaluate_response(
            query=query,  # 원래 질문으로 평가
            response=response['answer'],
            retrieved_docs=response['docs'],
            strategy="single_shot_rag_with_retry"
        )

        # 3. 시도 기록
        attempts.append({
            "attempt": attempt,
            "query": current_query,
            "answer": response['answer'],
            "quality_score": quality.overall_score,
            "quality_details": {
                "relevance": quality.relevance,
                "completeness": quality.completeness,
                "consistency": quality.consistency,
                "accuracy": quality.accuracy,
                "helpfulness": quality.helpfulness
            },
            "docs": response['docs']
        })

        print(f"\n품질 점수: {quality.overall_score:.2f}")

        # 4. 품질 기준 충족 시 조기 종료
        if quality.overall_score >= quality_threshold:
            print(f"✅ 품질 기준 충족! (목표: {quality_threshold:.2f})")
            break

        # 5. 재시도가 남았다면 쿼리 개선
        if attempt < max_retries:
            print(f"⚠️ 품질 기준 미달 (목표: {quality_threshold:.2f})")
            print(f"쿼리를 개선하여 재시도합니다...")

            # 품질 피드백을 반영한 쿼리 개선
            refinement_prompt = ChatPromptTemplate.from_messages([
                ("system", dedent("""
                    당신은 검색 쿼리 최적화 전문가입니다.

                    이전 시도에서 품질이 낮았던 이유를 분석하고,
                    더 나은 검색 결과를 얻을 수 있도록 쿼리를 개선하세요.

                    [개선 전략]
                    - 낮은 관련성: 핵심 키워드 추가, 동의어 활용
                    - 낮은 완성도: 구체적인 세부 정보 요청
                    - 낮은 정확성: 명확한 제약 조건 추가
                """)),
                ("user", dedent(f"""
                    [원래 질문]
                    {query}

                    [현재 쿼리]
                    {current_query}

                    [이전 답변]
                    {response['answer']}

                    [품질 문제]
                    - 관련성: {quality.relevance:.2f}
                    - 완성도: {quality.completeness:.2f}
                    - 일관성: {quality.consistency:.2f}
                    - 정확성: {quality.accuracy:.2f}
                    - 유용성: {quality.helpfulness:.2f}

                    [약점]
                    {chr(10).join(quality.weaknesses) if quality.weaknesses else '없음'}

                    쿼리를 개선하세요.
                """))
            ])

            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
            chain = refinement_prompt | llm | StrOutputParser()
            current_query = chain.invoke({})

            print(f"개선된 쿼리: {current_query}")

    # 6. 최고 품질의 답변 선택
    best_attempt = max(attempts, key=lambda x: x['quality_score'])

    print(f"\n\n{'='*100}")
    print("최종 결과 - 최고 품질 답변 선택")
    print(f"{'='*100}")
    print(f"총 시도 횟수: {len(attempts)}")
    print(f"최고 품질 점수: {best_attempt['quality_score']:.2f} (시도 {best_attempt['attempt']})")
    print(f"\n최종 답변:\n{best_attempt['answer']}")

    return {
        "final_answer": best_attempt['answer'],
        "best_quality_score": best_attempt['quality_score'],
        "total_attempts": len(attempts),
        "all_attempts": attempts,
        "quality_threshold_met": best_attempt['quality_score'] >= quality_threshold
    }

# 테스트
query = "스테이크와 어울리는 와인을 추천해주세요"

result = auto_quality_improvement(
    query=query,
    databases={"menu_db": menu_db, "wine_db": wine_db},
    quality_threshold=0.75,
    max_retries=3
)

print(f"\n\n품질 기준 충족: {result['quality_threshold_met']}")
```

### 솔루션 3: 멀티모달 RAG 시스템 확장

```python
from typing import Optional

class MultimodalDocument(BaseModel):
    """이미지 메타데이터를 포함한 문서 구조"""
    content: str
    image_url: Optional[str] = None
    image_caption: Optional[str] = None
    source: str
    menu_number: Optional[str] = None
    menu_name: Optional[str] = None

def detect_image_query(query: str) -> bool:
    """이미지 관련 질문 감지"""

    image_keywords = [
        "사진", "이미지", "그림", "보여줘", "보여주세요",
        "어떻게 생겼", "모습", "외관", "비주얼"
    ]

    return any(keyword in query.lower() for keyword in image_keywords)

def format_multimodal_docs(docs: List[Document], include_images: bool = False) -> str:
    """멀티모달 문서 포맷팅"""

    formatted_result = ""
    for i, doc in enumerate(docs, 1):
        formatted_result += f"[문서 {i}]\n"
        formatted_result += f"{doc.page_content}\n"

        # 메타데이터 처리
        if doc.metadata:
            source = doc.metadata.get('source', '알 수 없음')
            menu_name = doc.metadata.get('menu_name', '')
            menu_number = doc.metadata.get('menu_number', '')

            if menu_name and menu_number:
                formatted_result += f"(출처: {source} - [{menu_number}] {menu_name})\n"

            # 이미지 메타데이터가 있고, 이미지 포함이 요청된 경우
            if include_images:
                image_url = doc.metadata.get('image_url')
                image_caption = doc.metadata.get('image_caption')

                if image_url:
                    formatted_result += f"\n📷 이미지: ![{image_caption or menu_name}]({image_url})\n"
                    if image_caption:
                        formatted_result += f"   설명: {image_caption}\n"

        formatted_result += "-" * 80 + "\n"

    return formatted_result

def multimodal_rag_response(
    query: str,
    databases: Dict[str, VectorStore],
    k: int = 3
) -> Dict:
    """멀티모달 RAG 응답 생성"""

    # 1. 이미지 관련 질문인지 감지
    is_image_query = detect_image_query(query)

    print(f"이미지 관련 질문: {is_image_query}")

    # 2. 검색 수행
    all_docs = []
    for db_name, vectorstore in databases.items():
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query)
        for doc in docs:
            doc.metadata['database'] = db_name
        all_docs.extend(docs)

    all_docs = all_docs[:k]

    # 3. 프롬프트 구성 (이미지 요청 시 특별 안내 추가)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    system_message = dedent("""
        당신은 레스토랑 서비스 전문가입니다.
        주어진 컨텍스트를 활용하여 고객의 질문에 답변하세요.

        [가이드라인]
        1. 컨텍스트에 정보가 있으면 이를 기반으로 구체적으로 답변하세요
        2. 이미지 정보가 있으면 마크다운 형식으로 포함하세요
        3. 답변할 때는 참조한 출처를 명시하세요
        4. 친절하고 전문적인 톤을 유지하세요
    """)

    if is_image_query:
        system_message += dedent("""

            [이미지 관련 질문 처리]
            - 컨텍스트에 이미지가 있으면 반드시 포함하세요
            - 이미지가 없으면 "죄송하지만 해당 메뉴의 이미지를 찾을 수 없습니다"라고 안내하세요
            - 이미지 설명(caption)이 있으면 함께 제공하세요
        """)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "[컨텍스트]\n{context}\n\n[질문]\n{question}\n\n[답변]\n"),
    ])

    # 4. 컨텍스트 생성 (이미지 포함 여부 결정)
    context = format_multimodal_docs(all_docs, include_images=is_image_query)

    # 5. 응답 생성
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": query})

    return {
        "answer": answer,
        "docs": all_docs,
        "is_image_query": is_image_query,
        "num_docs": len(all_docs),
        "has_images": any(doc.metadata.get('image_url') for doc in all_docs)
    }

# 테스트 (예시 - 실제 DB에 이미지 메타데이터가 있다고 가정)
test_queries = [
    "스테이크 메뉴를 추천해주세요",  # 일반 질문
    "스테이크 사진을 보여주세요",  # 이미지 요청
    "파스타는 어떻게 생겼나요?",  # 이미지 요청
]

print("=" * 100)
print("멀티모달 RAG 테스트")
print("=" * 100)

for query in test_queries:
    print(f"\n질문: {query}")
    result = multimodal_rag_response(query, databases, k=2)
    print(f"답변:\n{result['answer']}")
    print(f"이미지 질문: {result['is_image_query']}")
    print(f"이미지 포함: {result['has_images']}")
    print("-" * 100)
```

## 🚀 실무 활용 예시

### 활용 예시 1: 고객 서비스 챗봇

레스토랑 웹사이트의 고객 서비스 챗봇에 Adaptive RAG를 적용합니다.

```python
class RestaurantChatbot:
    """레스토랑 고객 서비스 챗봇"""

    def __init__(self, databases: Dict[str, VectorStore]):
        self.databases = databases
        self.conversation_history = []
        self.logger = logging.getLogger("RestaurantChatbot")

    def chat(self, user_message: str) -> str:
        """사용자 메시지 처리 및 응답"""

        # 1. 대화 기록에 추가
        self.conversation_history.append({
            "role": "user",
            "message": user_message,
            "timestamp": datetime.now()
        })

        # 2. 라우팅 결정
        routing = dynamic_route_query(user_message)

        # 3. 전략별 응답 생성
        if routing.strategy == "no_retrieval":
            response = improved_no_retrieval_response(user_message)
            answer = response['answer']

        elif routing.strategy == "single_lookup":
            response = improved_single_rag_response(
                user_message,
                self.databases,
                k=routing.suggested_k
            )
            answer = response['answer']

        elif routing.strategy == "iterative":
            response = improved_iterative_rag_response(
                user_message,
                self.databases,
                k=routing.suggested_k,
                max_iterations=2  # 실시간 응답을 위해 2회로 제한
            )
            answer = response['final_answer']

        # 4. 대화 기록에 추가
        self.conversation_history.append({
            "role": "assistant",
            "message": answer,
            "strategy": routing.strategy,
            "timestamp": datetime.now()
        })

        # 5. 로깅
        self.logger.info(f"User: {user_message}")
        self.logger.info(f"Strategy: {routing.strategy} (k={routing.suggested_k})")
        self.logger.info(f"Response: {answer[:100]}...")

        return answer

    def get_conversation_history(self) -> List[Dict]:
        """대화 기록 조회"""
        return self.conversation_history

    def clear_history(self):
        """대화 기록 초기화"""
        self.conversation_history = []

# 챗봇 사용 예시
chatbot = RestaurantChatbot(databases={"menu_db": menu_db, "wine_db": wine_db})

print("=" * 100)
print("레스토랑 챗봇 대화 시뮬레이션")
print("=" * 100)

conversation = [
    "안녕하세요!",
    "오늘의 추천 메뉴가 있나요?",
    "스테이크와 어울리는 와인을 추천해주세요",
    "감사합니다!"
]

for message in conversation:
    print(f"\n👤 고객: {message}")
    response = chatbot.chat(message)
    print(f"🤖 챗봇: {response}")
    print("-" * 80)

# 대화 기록 조회
print("\n\n대화 기록 요약:")
for i, entry in enumerate(chatbot.get_conversation_history(), 1):
    role = "고객" if entry['role'] == 'user' else "챗봇"
    print(f"{i}. [{role}] {entry['message'][:50]}...")
```

### 활용 예시 2: A/B 테스트 시스템

서로 다른 RAG 전략의 효과를 비교하는 A/B 테스트 시스템입니다.

```python
import random
from typing import Literal

class RAGABTestSystem:
    """RAG 전략 A/B 테스트 시스템"""

    def __init__(self, databases: Dict[str, VectorStore]):
        self.databases = databases
        self.test_results = []

    def run_test(
        self,
        query: str,
        variant: Literal["A", "B"] = None
    ) -> Dict:
        """A/B 테스트 실행"""

        # 변형 선택 (지정되지 않았으면 무작위)
        if variant is None:
            variant = random.choice(["A", "B"])

        if variant == "A":
            # A: 기본 SingleShotRAG
            response = improved_single_rag_response(
                query,
                self.databases,
                k=3
            )
            strategy = "single_shot_rag"
        else:
            # B: 개선된 IterativeRAG
            response = improved_iterative_rag_response(
                query,
                self.databases,
                k=2,
                max_iterations=2
            )
            strategy = "iterative_rag"
            response = {
                "answer": response['final_answer'],
                "docs": response['all_docs']
            }

        # 품질 평가
        quality = improved_evaluate_response(
            query=query,
            response=response['answer'],
            retrieved_docs=response['docs'],
            strategy=strategy
        )

        # 결과 기록
        test_result = {
            "query": query,
            "variant": variant,
            "strategy": strategy,
            "answer": response['answer'],
            "quality_score": quality.overall_score,
            "quality_details": {
                "relevance": quality.relevance,
                "completeness": quality.completeness,
                "consistency": quality.consistency,
                "accuracy": quality.accuracy,
                "helpfulness": quality.helpfulness
            },
            "timestamp": datetime.now()
        }

        self.test_results.append(test_result)

        return test_result

    def get_statistics(self) -> Dict:
        """A/B 테스트 통계 분석"""

        if not self.test_results:
            return {"error": "테스트 결과가 없습니다"}

        # 변형별 그룹화
        variant_a = [r for r in self.test_results if r['variant'] == 'A']
        variant_b = [r for r in self.test_results if r['variant'] == 'B']

        def calculate_stats(results):
            if not results:
                return None
            scores = [r['quality_score'] for r in results]
            return {
                "count": len(results),
                "mean_quality": sum(scores) / len(scores),
                "min_quality": min(scores),
                "max_quality": max(scores),
                "mean_relevance": sum(r['quality_details']['relevance'] for r in results) / len(results),
                "mean_completeness": sum(r['quality_details']['completeness'] for r in results) / len(results),
            }

        stats_a = calculate_stats(variant_a)
        stats_b = calculate_stats(variant_b)

        # 승자 결정
        if stats_a and stats_b:
            winner = "A" if stats_a['mean_quality'] > stats_b['mean_quality'] else "B"
            improvement = abs(stats_a['mean_quality'] - stats_b['mean_quality'])
        else:
            winner = None
            improvement = 0

        return {
            "total_tests": len(self.test_results),
            "variant_a": stats_a,
            "variant_b": stats_b,
            "winner": winner,
            "improvement": improvement
        }

# A/B 테스트 실행
ab_test = RAGABTestSystem(databases={"menu_db": menu_db, "wine_db": wine_db})

test_queries = [
    "파스타 메뉴 추천해주세요",
    "스테이크와 어울리는 와인은?",
    "채식 메뉴가 있나요?",
    "가장 인기 있는 메뉴는?",
    "레드 와인과 화이트 와인의 차이는?",
]

print("=" * 100)
print("A/B 테스트 실행")
print("=" * 100)

# 각 쿼리에 대해 A, B 변형 모두 테스트
for query in test_queries:
    print(f"\n쿼리: {query}")

    # A 변형
    result_a = ab_test.run_test(query, variant="A")
    print(f"  [A] {result_a['strategy']}: 품질 {result_a['quality_score']:.2f}")

    # B 변형
    result_b = ab_test.run_test(query, variant="B")
    print(f"  [B] {result_b['strategy']}: 품질 {result_b['quality_score']:.2f}")

# 통계 분석
print(f"\n\n{'='*100}")
print("A/B 테스트 결과 분석")
print(f"{'='*100}")

stats = ab_test.get_statistics()

print(f"\n총 테스트 수: {stats['total_tests']}")

print(f"\n[변형 A - SingleShotRAG]")
print(f"  테스트 수: {stats['variant_a']['count']}")
print(f"  평균 품질: {stats['variant_a']['mean_quality']:.3f}")
print(f"  평균 관련성: {stats['variant_a']['mean_relevance']:.3f}")
print(f"  평균 완성도: {stats['variant_a']['mean_completeness']:.3f}")

print(f"\n[변형 B - IterativeRAG]")
print(f"  테스트 수: {stats['variant_b']['count']}")
print(f"  평균 품질: {stats['variant_b']['mean_quality']:.3f}")
print(f"  평균 관련성: {stats['variant_b']['mean_relevance']:.3f}")
print(f"  평균 완성도: {stats['variant_b']['mean_completeness']:.3f}")

print(f"\n🏆 승자: 변형 {stats['winner']}")
print(f"📈 개선율: {stats['improvement']:.1%}")
```

### 활용 예시 3: 실시간 모니터링 대시보드

RAG 시스템의 성능을 실시간으로 모니터링하는 대시보드입니다.

```python
from datetime import datetime, timedelta
from collections import defaultdict

class RAGMonitoringDashboard:
    """RAG 시스템 모니터링 대시보드"""

    def __init__(self):
        self.metrics = []
        self.start_time = datetime.now()

    def log_request(
        self,
        query: str,
        strategy: str,
        response_time: float,
        quality_score: float,
        success: bool = True,
        error: str = None
    ):
        """요청 로깅"""

        self.metrics.append({
            "timestamp": datetime.now(),
            "query": query,
            "strategy": strategy,
            "response_time": response_time,
            "quality_score": quality_score if success else 0,
            "success": success,
            "error": error
        })

    def get_summary(self, time_window: int = 60) -> Dict:
        """시간 윈도우 내 요약 통계"""

        # 최근 N분 데이터만 필터링
        cutoff_time = datetime.now() - timedelta(minutes=time_window)
        recent_metrics = [
            m for m in self.metrics
            if m['timestamp'] >= cutoff_time
        ]

        if not recent_metrics:
            return {"message": "데이터 없음"}

        # 전략별 그룹화
        by_strategy = defaultdict(list)
        for m in recent_metrics:
            by_strategy[m['strategy']].append(m)

        # 통계 계산
        summary = {
            "time_window": f"최근 {time_window}분",
            "total_requests": len(recent_metrics),
            "success_rate": sum(1 for m in recent_metrics if m['success']) / len(recent_metrics),
            "avg_response_time": sum(m['response_time'] for m in recent_metrics) / len(recent_metrics),
            "avg_quality_score": sum(m['quality_score'] for m in recent_metrics if m['success']) / sum(1 for m in recent_metrics if m['success']) if any(m['success'] for m in recent_metrics) else 0,
            "by_strategy": {}
        }

        # 전략별 통계
        for strategy, metrics in by_strategy.items():
            successful = [m for m in metrics if m['success']]
            summary['by_strategy'][strategy] = {
                "count": len(metrics),
                "success_rate": len(successful) / len(metrics),
                "avg_response_time": sum(m['response_time'] for m in metrics) / len(metrics),
                "avg_quality_score": sum(m['quality_score'] for m in successful) / len(successful) if successful else 0
            }

        return summary

    def print_dashboard(self):
        """대시보드 출력"""

        summary = self.get_summary(time_window=60)

        print("\n" + "=" * 100)
        print("📊 RAG 시스템 모니터링 대시보드")
        print("=" * 100)

        print(f"\n⏱️ 기간: {summary['time_window']}")
        print(f"📈 총 요청: {summary['total_requests']}")
        print(f"✅ 성공률: {summary['success_rate']:.1%}")
        print(f"⚡ 평균 응답시간: {summary['avg_response_time']:.2f}초")
        print(f"⭐ 평균 품질점수: {summary['avg_quality_score']:.2f}")

        print(f"\n\n전략별 성능:")
        print("-" * 100)
        print(f"{'전략':<20} {'요청수':>10} {'성공률':>10} {'응답시간':>12} {'품질점수':>12}")
        print("-" * 100)

        for strategy, stats in summary['by_strategy'].items():
            print(f"{strategy:<20} {stats['count']:>10} {stats['success_rate']:>9.1%} {stats['avg_response_time']:>11.2f}s {stats['avg_quality_score']:>11.2f}")

# 모니터링 시스템 사용 예시
import time

dashboard = RAGMonitoringDashboard()

# 시뮬레이션: 여러 요청 처리
test_scenarios = [
    ("안녕하세요", "no_retrieval", 0.1, 0.95),
    ("파스타 메뉴", "single_shot_rag", 0.8, 0.85),
    ("스테이크 와인 페어링", "iterative_rag", 2.5, 0.90),
    ("채식 메뉴 추천", "single_shot_rag", 0.7, 0.88),
    ("영업시간", "no_retrieval", 0.1, 0.92),
]

print("=" * 100)
print("RAG 시스템 요청 처리 시뮬레이션")
print("=" * 100)

for query, strategy, response_time, quality in test_scenarios:
    print(f"\n처리 중: {query} ({strategy})")

    # 실제로는 여기서 RAG 실행
    time.sleep(0.1)  # 시뮬레이션

    # 로깅
    dashboard.log_request(
        query=query,
        strategy=strategy,
        response_time=response_time,
        quality_score=quality,
        success=True
    )

    print(f"✅ 완료 ({response_time:.1f}s, 품질: {quality:.2f})")

# 대시보드 출력
dashboard.print_dashboard()
```

## 📖 참고 자료

### 공식 문서

- **Adaptive RAG 논문**: [https://arxiv.org/abs/2403.14403](https://arxiv.org/abs/2403.14403)
- **LangChain 문서**: [https://python.langchain.com/docs/](https://python.langchain.com/docs/)
- **LangGraph 문서**: [https://langchain-ai.github.io/langgraph/](https://langchain-ai.github.io/langgraph/)
- **Pydantic 문서**: [https://docs.pydantic.dev/](https://docs.pydantic.dev/)
- **ChromaDB 문서**: [https://docs.trychroma.com/](https://docs.trychroma.com/)

### 추가 학습 자료

- **RAG 기초**: Week 2 - RAG 구현 가이드
- **프롬프트 엔지니어링**: Week 3 - 고급 프롬프트 엔지니어링
- **LangGraph 심화**: PRJ03_W3_002_LangGraph_AdaptiveRAG_Part2 (다음 가이드)

### 관련 블로그 및 튜토리얼

- [LangChain Blog - Advanced RAG Techniques](https://blog.langchain.dev/)
- [OpenAI Cookbook - RAG Best Practices](https://cookbook.openai.com/)
- [Pinecone Learning Center - RAG 101](https://www.pinecone.io/learn/retrieval-augmented-generation/)

### 추천 다음 단계

1. **Part 2 학습**: LangGraph를 활용한 상태 기반 Adaptive RAG 구현
2. **프로덕션 배포**: FastAPI를 사용한 REST API 구축
3. **성능 최적화**: 캐싱, 배치 처리, 비동기 실행
4. **고급 평가**: RAGAS, LangSmith를 활용한 체계적 평가

---

**학습 완료 후 다음을 확인하세요**:
- [ ] 3가지 RAG 전략의 차이를 설명할 수 있다
- [ ] Pydantic으로 구조화된 출력을 만들 수 있다
- [ ] 쿼리 라우팅 시스템을 구현할 수 있다
- [ ] 반복적 검색 및 쿼리 개선을 구현할 수 있다
- [ ] 응답 품질 평가 시스템을 만들 수 있다
- [ ] 실무 챗봇 시스템에 적용할 수 있다

**다음 학습**: PRJ03_W3_002_LangGraph_AdaptiveRAG_Part2.md - LangGraph 상태 기반 구현
