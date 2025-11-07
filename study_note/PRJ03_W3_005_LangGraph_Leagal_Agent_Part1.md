# 법률 문서 기반 검색 에이전트 Part1 - 환경 구축 및 도구 정의

## 📚 학습 목표

이 학습 가이드를 완료하면 다음을 할 수 있습니다:

1. **법률 문서 처리 파이프라인**을 구축하여 PDF 문서를 벡터 저장소에 저장할 수 있습니다
2. **도메인별 벡터 저장소**를 생성하여 법률 분야별로 독립적인 검색 시스템을 구축할 수 있습니다
3. **LangChain Tool**을 활용하여 검색 도구와 웹 검색 도구를 정의할 수 있습니다
4. **함수 호출(Function Calling)**을 이해하고 LLM이 적절한 도구를 선택하도록 구현할 수 있습니다
5. **RecursiveCharacterTextSplitter**를 사용하여 법률 문서를 적절한 크기로 분할할 수 있습니다
6. **ChromaDB**를 활용하여 임베딩 기반 검색 시스템을 구축할 수 있습니다
7. 법률 도메인에 특화된 **검색 도구**를 설계하고 구현할 수 있습니다

## 🔑 핵심 개념

### 법률 문서 기반 검색 시스템

**법률 문서 검색 시스템**은 방대한 법률 정보를 효율적으로 검색하고 활용하기 위한 시스템입니다. 일반 문서 검색과 달리 법률 도메인의 특수성을 고려해야 합니다.

#### 법률 도메인의 특징

| 특징 | 설명 | 구현 방법 |
|------|------|----------|
| **전문 용어** | 법률 특화 용어 및 개념 | 도메인별 벡터 저장소 분리 |
| **정확성 요구** | 잘못된 정보는 법적 문제 발생 | 문서 청크 크기 최적화 |
| **맥락 중요성** | 조항 간 관계 및 상호 참조 | Overlap을 활용한 문맥 유지 |
| **최신성** | 법령 개정 및 판례 변화 | 웹 검색 도구 통합 |

### 시스템 아키텍처

```
[법률 PDF 문서]
   ↓
[PyPDFLoader]
   ↓
[RecursiveCharacterTextSplitter]
   ├─ chunk_size: 1000
   └─ chunk_overlap: 200
   ↓
[OpenAI Embeddings]
   ↓
[ChromaDB 벡터 저장소]
   ├─ personal_law_db (개인정보보호법)
   ├─ labor_law_db (근로기준법)
   └─ housing_law_db (주택임대차보호법)
   ↓
[검색 도구]
   ├─ personal_law_search
   ├─ labor_law_search
   └─ housing_law_search
   ↓
[LLM + Function Calling]
   ↓
[답변 생성]
```

### 도메인별 벡터 저장소 전략

**단일 벡터 저장소 vs 도메인별 벡터 저장소**:

| 방식 | 장점 | 단점 |
|------|------|------|
| **단일 저장소** | 관리 간편, 통합 검색 가능 | 도메인 혼재, 정확도 저하 |
| **도메인별 저장소** | 높은 정확도, 명확한 분리 | 관리 복잡, 라우팅 필요 |

본 프로젝트는 **도메인별 벡터 저장소** 방식을 채택하여 각 법률 분야에 특화된 검색을 제공합니다.

### Function Calling (함수 호출)

**Function Calling**은 LLM이 외부 도구를 호출할 수 있도록 하는 기능입니다.

```python
# LLM이 질문을 분석하여 적절한 도구를 선택
User: "연차휴가 부여 기준에 대해서 설명해주세요."

LLM 분석:
- 질문 키워드: 연차휴가, 부여 기준
- 관련 법률: 근로기준법
- 선택 도구: labor_law_search

도구 호출:
labor_law_search.invoke({"query": "연차휴가 부여 기준"})
→ 근로기준법 벡터 저장소에서 검색
→ 관련 조항 반환
→ LLM이 답변 생성
```

### 기술 스택

- **LangChain**: RAG 파이프라인 구축
- **ChromaDB**: 벡터 저장소
- **OpenAI**: 임베딩 (text-embedding-3-small) 및 LLM (gpt-4.1-mini)
- **PyPDFLoader**: PDF 문서 로딩
- **RecursiveCharacterTextSplitter**: 텍스트 청크 분할
- **Tavily Search**: 웹 검색 (최신 정보)
- **LangGraph**: 에이전트 워크플로우 (Part 2)

## 🛠 환경 설정

### 필수 라이브러리 설치

```bash
# 핵심 라이브러리
pip install langchain langchain-community langchain-openai langchain-chroma

# PDF 처리
pip install pypdf

# 웹 검색
pip install tavily-python langchain-tavily

# 유틸리티
pip install python-dotenv pydantic
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
import re
import os
import json
from glob import glob
from textwrap import dedent
from pprint import pprint
import uuid
import warnings

warnings.filterwarnings("ignore")

print("✅ 환경 설정 완료")
```

## 💻 단계별 구현

### 단계 1: 법률 문서 로드 및 벡터 저장소 구축

법률 PDF 문서를 로드하고 도메인별 벡터 저장소를 생성합니다.

#### 1.1 PDF 파일 확인

```python
# PDF 파일 목록 확인
pdf_files = glob(os.path.join('data', '*_law.pdf'))

print("📚 법률 문서 목록:")
for file in pdf_files:
    print(f"  - {os.path.basename(file)}")

# 출력 예시:
# 📚 법률 문서 목록:
#   - personal_info_law.pdf (개인정보보호법)
#   - labor_law.pdf (근로기준법)
#   - housing_leasing_law.pdf (주택임대차보호법)
```

#### 1.2 개인정보보호법 벡터 저장소 구축

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# PDF 로더
personal_loader = PyPDFLoader("data/personal_info_law.pdf")
personal_docs = personal_loader.load()

print(f"✅ 개인정보보호법 로딩 완료: {len(personal_docs)}개 페이지")

# 텍스트 분할
personal_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # 청크 크기: 1000자
    chunk_overlap=200,      # 중복: 200자 (맥락 유지)
    separators=["\n\n", "\n", " ", ""]  # 분할 우선순위
)

personal_splits = personal_splitter.split_documents(personal_docs)
print(f"📄 청크 분할 완료: {len(personal_splits)}개 청크")

# 벡터 저장소 생성
personal_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
personal_vectorstore = Chroma.from_documents(
    documents=personal_splits,
    embedding=personal_embeddings,
    collection_name="personal_info_law",
    persist_directory="./chroma_personal_db"
)

print("✅ 개인정보보호법 벡터 저장소 생성 완료")

# 검색 테스트
test_query = "개인정보 수집 시 동의를 받아야 하는 경우는?"
test_results = personal_vectorstore.similarity_search(test_query, k=2)

print(f"\n🔍 검색 테스트: '{test_query}'")
print(f"검색 결과: {len(test_results)}개")
for i, doc in enumerate(test_results, 1):
    print(f"\n[결과 {i}]")
    print(doc.page_content[:200] + "...")
```

**실행 결과 예시**:
```
✅ 개인정보보호법 로딩 완료: 18개 페이지
📄 청크 분할 완료: 247개 청크
✅ 개인정보보호법 벡터 저장소 생성 완료

🔍 검색 테스트: '개인정보 수집 시 동의를 받아야 하는 경우는?'
검색 결과: 2개

[결과 1]
제15조(개인정보의 수집·이용) ① 개인정보처리자는 다음 각 호의 어느 하나에 해당하는 경우에는
개인정보를 수집할 수 있으며 그 수집 목적의 범위에서 이용할 수 있다.
1. 정보주체의 동의를 받은 경우...
```

#### 1.3 근로기준법 벡터 저장소 구축

```python
# 근로기준법 로드 및 벡터 저장소 생성
labor_loader = PyPDFLoader("data/labor_law.pdf")
labor_docs = labor_loader.load()

print(f"✅ 근로기준법 로딩 완료: {len(labor_docs)}개 페이지")

# 텍스트 분할
labor_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

labor_splits = labor_splitter.split_documents(labor_docs)
print(f"📄 청크 분할 완료: {len(labor_splits)}개 청크")

# 벡터 저장소 생성
labor_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
labor_vectorstore = Chroma.from_documents(
    documents=labor_splits,
    embedding=labor_embeddings,
    collection_name="labor_law",
    persist_directory="./chroma_labor_db"
)

print("✅ 근로기준법 벡터 저장소 생성 완료")

# 검색 테스트
test_query = "연차휴가 부여 기준"
test_results = labor_vectorstore.similarity_search(test_query, k=2)

print(f"\n🔍 검색 테스트: '{test_query}'")
for i, doc in enumerate(test_results, 1):
    print(f"\n[결과 {i}]")
    print(doc.page_content[:200] + "...")
```

#### 1.4 주택임대차보호법 벡터 저장소 구축

```python
# 주택임대차보호법 로드 및 벡터 저장소 생성
housing_loader = PyPDFLoader("data/housing_leasing_law.pdf")
housing_docs = housing_loader.load()

print(f"✅ 주택임대차보호법 로딩 완료: {len(housing_docs)}개 페이지")

# 텍스트 분할
housing_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

housing_splits = housing_splitter.split_documents(housing_docs)
print(f"📄 청크 분할 완료: {len(housing_splits)}개 청크")

# 벡터 저장소 생성
housing_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
housing_vectorstore = Chroma.from_documents(
    documents=housing_splits,
    embedding=housing_embeddings,
    collection_name="housing_law",
    persist_directory="./chroma_housing_db"
)

print("✅ 주택임대차보호법 벡터 저장소 생성 완료")

# 검색 테스트
test_query = "전세 계약 시 임차인 권리"
test_results = housing_vectorstore.similarity_search(test_query, k=2)

print(f"\n🔍 검색 테스트: '{test_query}'")
for i, doc in enumerate(test_results, 1):
    print(f"\n[결과 {i}]")
    print(doc.page_content[:200] + "...")
```

### 단계 2: 검색 도구 정의

LangChain Tool을 사용하여 각 법률 도메인에 특화된 검색 도구를 정의합니다.

#### 2.1 검색 도구 생성

```python
from langchain.tools.retriever import create_retriever_tool

# 개인정보보호법 검색 도구
personal_law_search = create_retriever_tool(
    retriever=personal_vectorstore.as_retriever(search_kwargs={"k": 3}),
    name="personal_law_search",
    description="""
    개인정보보호법과 관련된 질문에 답변할 때 사용합니다.
    개인정보의 수집, 이용, 제공, 파기, 정보주체의 권리 등에 대한 질문에 유용합니다.

    예시 질문:
    - 개인정보 수집 시 동의를 받아야 하는 경우는?
    - 개인정보 처리 방침에 포함되어야 할 내용은?
    - 정보주체가 가지는 권리는 무엇인가요?
    """
)

# 근로기준법 검색 도구
labor_law_search = create_retriever_tool(
    retriever=labor_vectorstore.as_retriever(search_kwargs={"k": 3}),
    name="labor_law_search",
    description="""
    근로기준법과 관련된 질문에 답변할 때 사용합니다.
    근로 시간, 휴게, 휴일, 연차휴가, 임금, 해고, 퇴직금 등에 대한 질문에 유용합니다.

    예시 질문:
    - 연차휴가 부여 기준에 대해 설명해주세요.
    - 법정 근로시간은 어떻게 되나요?
    - 퇴직금 계산 방법은?
    """
)

# 주택임대차보호법 검색 도구
housing_law_search = create_retriever_tool(
    retriever=housing_vectorstore.as_retriever(search_kwargs={"k": 3}),
    name="housing_law_search",
    description="""
    주택임대차보호법과 관련된 질문에 답변할 때 사용합니다.
    전월세 계약, 임차인 권리, 보증금 반환, 계약 갱신 등에 대한 질문에 유용합니다.

    예시 질문:
    - 전세 계약 시 임차인이 보호받을 수 있는 권리는?
    - 보증금 반환을 보장받으려면 어떻게 해야 하나요?
    - 계약 갱신 요구권이란 무엇인가요?
    """
)

print("✅ 검색 도구 생성 완료:")
print(f"  - {personal_law_search.name}")
print(f"  - {labor_law_search.name}")
print(f"  - {housing_law_search.name}")
```

#### 2.2 웹 검색 도구 추가

```python
from langchain_community.tools.tavily_search import TavilySearchResults

# 웹 검색 도구
web_search = TavilySearchResults(
    name="web_search",
    description="""
    최신 정보나 법률 문서에 없는 정보를 검색할 때 사용합니다.
    법령 개정 사항, 최신 판례, 통계 자료 등을 찾을 때 유용합니다.

    예시 질문:
    - 2024년 최저임금은 얼마인가요?
    - 최근 개인정보보호법 개정 내용은?
    - 2023년 연차휴가 사용 비율은?
    """,
    max_results=3
)

print(f"✅ 웹 검색 도구 추가: {web_search.name}")
```

#### 2.3 도구 목록 정의

```python
# 모든 도구를 하나의 리스트로 관리
tools = [
    personal_law_search,
    labor_law_search,
    housing_law_search,
    web_search
]

print(f"\n📦 총 {len(tools)}개의 도구 준비 완료:")
for tool in tools:
    print(f"  - {tool.name}: {tool.description.split('.')[0]}")
```

### 단계 3: LLM 모델 및 Function Calling 설정

LLM이 적절한 도구를 선택하여 호출할 수 있도록 설정합니다.

#### 3.1 LLM 모델 초기화

```python
from langchain_openai import ChatOpenAI

# 기본 LLM 모델
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0  # 일관된 답변을 위해 0으로 설정
)

# 도구가 바인딩된 LLM
llm_with_tools = llm.bind_tools(tools)

print("✅ LLM 모델 초기화 완료")
print(f"  - 모델: gpt-4.1-mini")
print(f"  - 바인딩된 도구: {len(tools)}개")
```

#### 3.2 Function Calling 테스트 1: 근로기준법 질문

```python
# 근로기준법 관련 질문
query = "연차휴가 부여 기준에 대해서 설명해주세요."

print(f"\n🧪 테스트 1: {query}")

# LLM 호출
ai_msg = llm_with_tools.invoke(query)

# 도구 호출 확인
if ai_msg.tool_calls:
    print(f"\n✅ LLM이 도구 호출을 선택했습니다:")
    for tool_call in ai_msg.tool_calls:
        print(f"  - 도구: {tool_call['name']}")
        print(f"  - 인자: {tool_call['args']}")
else:
    print("\n⚠️ 도구 호출 없이 직접 답변:")
    print(ai_msg.content)
```

**실행 결과 예시**:
```
🧪 테스트 1: 연차휴가 부여 기준에 대해서 설명해주세요.

✅ LLM이 도구 호출을 선택했습니다:
  - 도구: labor_law_search
  - 인자: {'query': '연차휴가 부여 기준'}
```

#### 3.3 Function Calling 테스트 2: 도구 불필요한 질문

```python
# 도구가 필요 없는 일반 질문
query = "안녕하세요?"

print(f"\n🧪 테스트 2: {query}")

ai_msg = llm_with_tools.invoke(query)

if ai_msg.tool_calls:
    print(f"\n✅ LLM이 도구 호출을 선택했습니다:")
    for tool_call in ai_msg.tool_calls:
        print(f"  - 도구: {tool_call['name']}")
else:
    print("\n✅ 도구 호출 없이 직접 답변:")
    print(ai_msg.content)
```

**실행 결과 예시**:
```
🧪 테스트 2: 안녕하세요?

✅ 도구 호출 없이 직접 답변:
안녕하세요! 무엇을 도와드릴까요?
```

#### 3.4 Function Calling 테스트 3: 복합 질문 (벡터 검색 + 웹 검색)

```python
# 벡터 검색과 웹 검색이 모두 필요한 질문
query = "연차휴가 부여 기준에 대해서 설명해주세요. 2023년 연차휴가 사용 비율은 어느 정도인가요?"

print(f"\n🧪 테스트 3: {query}")

ai_msg = llm_with_tools.invoke(query)

if ai_msg.tool_calls:
    print(f"\n✅ LLM이 {len(ai_msg.tool_calls)}개의 도구 호출을 선택했습니다:")
    for i, tool_call in enumerate(ai_msg.tool_calls, 1):
        print(f"\n  [{i}] 도구: {tool_call['name']}")
        print(f"      인자: {tool_call['args']}")
else:
    print("\n⚠️ 도구 호출 없이 직접 답변:")
    print(ai_msg.content)
```

**실행 결과 예시**:
```
🧪 테스트 3: 연차휴가 부여 기준에 대해서 설명해주세요. 2023년 연차휴가 사용 비율은 어느 정도인가요?

✅ LLM이 2개의 도구 호출을 선택했습니다:

  [1] 도구: labor_law_search
      인자: {'query': '연차휴가 부여 기준'}

  [2] 도구: web_search
      인자: {'query': '2023년 연차휴가 사용 비율'}
```

## 🎯 실습 문제

### 문제 1: 커스텀 청크 전략 구현 (⭐)

**과제**: 법률 문서의 특성에 맞는 커스텀 청크 분할 전략을 구현하세요.

**요구사항**:
1. 조항 번호(제1조, 제2조 등)를 기준으로 분할
2. 각 조항이 하나의 청크가 되도록 구현
3. 조항이 너무 길면 적절히 분할하되 조항 번호 정보 유지

**힌트**:
```python
def law_article_splitter(text: str, max_chunk_size: int = 1000):
    """조항 기반 청크 분할"""
    # 조항 패턴: 제n조, 제n조의n
    article_pattern = r'제\d+조(?:의\d+)?'
    # 구현...
    pass
```

### 문제 2: 하이브리드 검색 구현 (⭐⭐)

**과제**: 키워드 검색과 벡터 검색을 결합한 하이브리드 검색 도구를 구현하세요.

**요구사항**:
1. BM25 알고리즘을 사용한 키워드 검색
2. 벡터 유사도 검색
3. 두 결과를 가중치로 결합 (예: 키워드 0.3 + 벡터 0.7)

**힌트**:
```python
from rank_bm25 import BM25Okapi

def hybrid_search(query: str, vectorstore, documents, alpha=0.7):
    """하이브리드 검색: 키워드 + 벡터"""
    # BM25 검색
    bm25_results = ...
    # 벡터 검색
    vector_results = ...
    # 결합
    combined_results = ...
    return combined_results
```

### 문제 3: 도구 사용 통계 추적 (⭐⭐⭐)

**과제**: 각 도구의 사용 빈도와 성능을 추적하는 시스템을 구현하세요.

**요구사항**:
1. 도구 호출 횟수 추적
2. 평균 응답 시간 측정
3. 성공/실패 비율 계산
4. 시각화 (matplotlib 또는 plotly)

## ✅ 솔루션 예시

### 솔루션 1: 커스텀 청크 전략 구현

```python
import re
from typing import List
from langchain_core.documents import Document

def law_article_splitter(text: str, max_chunk_size: int = 1000) -> List[str]:
    """
    법률 문서를 조항 기준으로 분할하는 함수
    """
    # 조항 패턴: 제n조, 제n조의n
    article_pattern = r'(제\d+조(?:의\d+)?)'

    # 조항으로 분할
    parts = re.split(article_pattern, text)

    chunks = []
    current_article = ""
    current_content = ""

    for i, part in enumerate(parts):
        if re.match(article_pattern, part):
            # 이전 조항 저장
            if current_article and current_content:
                chunk_text = f"{current_article} {current_content}".strip()

                # 조항이 너무 길면 분할
                if len(chunk_text) > max_chunk_size:
                    sub_chunks = split_long_article(chunk_text, current_article, max_chunk_size)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(chunk_text)

            # 새 조항 시작
            current_article = part
            current_content = ""
        else:
            current_content += part

    # 마지막 조항 저장
    if current_article and current_content:
        chunk_text = f"{current_article} {current_content}".strip()
        if len(chunk_text) > max_chunk_size:
            sub_chunks = split_long_article(chunk_text, current_article, max_chunk_size)
            chunks.extend(sub_chunks)
        else:
            chunks.append(chunk_text)

    return chunks

def split_long_article(text: str, article_number: str, max_size: int) -> List[str]:
    """긴 조항을 여러 청크로 분할 (조항 번호 유지)"""
    # 문장 단위로 분할
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = f"{article_number} "

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_size:
            current_chunk += sentence + " "
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = f"{article_number} (계속) {sentence} "

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

# 사용 예시
personal_docs = personal_loader.load()
full_text = "\n".join([doc.page_content for doc in personal_docs])

# 커스텀 분할 적용
custom_chunks = law_article_splitter(full_text, max_chunk_size=1000)

print(f"✅ 커스텀 분할 완료: {len(custom_chunks)}개 청크")
print(f"\n첫 번째 청크 예시:")
print(custom_chunks[0][:300] + "...")

# Document 객체로 변환
custom_documents = [
    Document(page_content=chunk, metadata={"source": "personal_info_law"})
    for chunk in custom_chunks
]

# 벡터 저장소 생성
custom_vectorstore = Chroma.from_documents(
    documents=custom_documents,
    embedding=personal_embeddings,
    collection_name="personal_law_custom",
    persist_directory="./chroma_custom_db"
)

print("✅ 커스텀 벡터 저장소 생성 완료")
```

### 솔루션 2: 하이브리드 검색 구현

```python
from rank_bm25 import BM25Okapi
from typing import List, Tuple
import numpy as np

class HybridRetriever:
    """하이브리드 검색: BM25 + 벡터 유사도"""

    def __init__(self, vectorstore, documents: List[Document], alpha: float = 0.7):
        """
        alpha: 벡터 검색 가중치 (0~1)
               1에 가까울수록 벡터 검색 중시
               0에 가까울수록 키워드 검색 중시
        """
        self.vectorstore = vectorstore
        self.documents = documents
        self.alpha = alpha

        # BM25 인덱스 생성
        tokenized_docs = [doc.page_content.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

        print(f"✅ 하이브리드 검색기 초기화 (alpha={alpha})")

    def search(self, query: str, k: int = 5) -> List[Document]:
        """하이브리드 검색 실행"""

        # 1. BM25 키워드 검색
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # 점수 정규화 (0~1)
        bm25_scores_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-10)

        # 2. 벡터 유사도 검색
        vector_results = self.vectorstore.similarity_search_with_score(query, k=len(self.documents))

        # 벡터 점수를 딕셔너리로 변환 (거리 → 유사도)
        vector_scores = {}
        for doc, score in vector_results:
            # 거리를 유사도로 변환 (거리가 작을수록 유사도 높음)
            similarity = 1 / (1 + score)
            vector_scores[doc.page_content] = similarity

        # 벡터 점수 정규화
        vector_scores_array = np.array(list(vector_scores.values()))
        vector_scores_norm = (vector_scores_array - vector_scores_array.min()) / \
                            (vector_scores_array.max() - vector_scores_array.min() + 1e-10)

        # 3. 하이브리드 점수 계산
        hybrid_scores = []
        for i, doc in enumerate(self.documents):
            bm25_score = bm25_scores_norm[i]
            vector_score = vector_scores_norm[i]

            # 가중 평균
            hybrid_score = (1 - self.alpha) * bm25_score + self.alpha * vector_score
            hybrid_scores.append((doc, hybrid_score))

        # 점수로 정렬
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)

        # 상위 k개 반환
        return [doc for doc, score in hybrid_scores[:k]]

# 사용 예시
hybrid_retriever = HybridRetriever(
    vectorstore=personal_vectorstore,
    documents=personal_splits,
    alpha=0.7  # 벡터 검색 70%, 키워드 검색 30%
)

# 검색 테스트
query = "개인정보 수집 동의"
results = hybrid_retriever.search(query, k=3)

print(f"\n🔍 하이브리드 검색: '{query}'")
for i, doc in enumerate(results, 1):
    print(f"\n[결과 {i}]")
    print(doc.page_content[:200] + "...")

# 하이브리드 검색 도구로 변환
from langchain.tools import Tool

hybrid_search_tool = Tool(
    name="hybrid_personal_law_search",
    description="개인정보보호법을 하이브리드 검색 (키워드 + 벡터)으로 검색합니다.",
    func=lambda q: hybrid_retriever.search(q, k=3)
)

print("\n✅ 하이브리드 검색 도구 생성 완료")
```

### 솔루션 3: 도구 사용 통계 추적

```python
import time
from datetime import datetime
from typing import Dict, List
import matplotlib.pyplot as plt
from collections import defaultdict

class ToolUsageTracker:
    """도구 사용 통계 추적"""

    def __init__(self, tools: List):
        self.tools = {tool.name: tool for tool in tools}

        # 통계 저장
        self.stats = {
            tool.name: {
                'call_count': 0,
                'success_count': 0,
                'failure_count': 0,
                'total_time': 0.0,
                'call_times': [],
                'queries': []
            }
            for tool in tools
        }

        print(f"✅ 도구 사용 추적 시작: {len(tools)}개 도구")

    def track_call(self, tool_name: str, query: str, success: bool, duration: float):
        """도구 호출 추적"""
        if tool_name not in self.stats:
            return

        stats = self.stats[tool_name]
        stats['call_count'] += 1
        stats['total_time'] += duration
        stats['call_times'].append(duration)
        stats['queries'].append(query)

        if success:
            stats['success_count'] += 1
        else:
            stats['failure_count'] += 1

    def get_summary(self) -> Dict:
        """통계 요약"""
        summary = {}

        for tool_name, stats in self.stats.items():
            call_count = stats['call_count']

            if call_count > 0:
                avg_time = stats['total_time'] / call_count
                success_rate = stats['success_count'] / call_count
            else:
                avg_time = 0.0
                success_rate = 0.0

            summary[tool_name] = {
                '호출 횟수': call_count,
                '성공 횟수': stats['success_count'],
                '실패 횟수': stats['failure_count'],
                '성공률': f"{success_rate * 100:.1f}%",
                '평균 응답 시간': f"{avg_time:.3f}초",
                '총 응답 시간': f"{stats['total_time']:.3f}초"
            }

        return summary

    def visualize(self):
        """통계 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 도구별 호출 횟수
        tool_names = list(self.stats.keys())
        call_counts = [self.stats[name]['call_count'] for name in tool_names]

        axes[0, 0].bar(tool_names, call_counts, color='skyblue')
        axes[0, 0].set_title('도구별 호출 횟수', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('도구')
        axes[0, 0].set_ylabel('호출 횟수')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # 2. 도구별 평균 응답 시간
        avg_times = [
            (self.stats[name]['total_time'] / self.stats[name]['call_count'])
            if self.stats[name]['call_count'] > 0 else 0
            for name in tool_names
        ]

        axes[0, 1].bar(tool_names, avg_times, color='lightcoral')
        axes[0, 1].set_title('도구별 평균 응답 시간', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('도구')
        axes[0, 1].set_ylabel('시간 (초)')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # 3. 도구별 성공률
        success_rates = [
            (self.stats[name]['success_count'] / self.stats[name]['call_count'] * 100)
            if self.stats[name]['call_count'] > 0 else 0
            for name in tool_names
        ]

        axes[1, 0].bar(tool_names, success_rates, color='lightgreen')
        axes[1, 0].set_title('도구별 성공률', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('도구')
        axes[1, 0].set_ylabel('성공률 (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].set_ylim([0, 100])

        # 4. 시간에 따른 호출 패턴
        total_calls = [self.stats[name]['call_count'] for name in tool_names]
        axes[1, 1].pie(total_calls, labels=tool_names, autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('도구별 사용 비율', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig('tool_usage_stats.png', dpi=300, bbox_inches='tight')
        print("✅ 통계 그래프 저장: tool_usage_stats.png")
        plt.show()

# 사용 예시
tracker = ToolUsageTracker(tools)

# 도구 호출 시뮬레이션
test_queries = [
    ("personal_law_search", "개인정보 수집 동의"),
    ("labor_law_search", "연차휴가 부여 기준"),
    ("housing_law_search", "전세 계약 권리"),
    ("web_search", "2024년 최저임금"),
    ("labor_law_search", "근로시간"),
    ("personal_law_search", "정보주체 권리"),
]

for tool_name, query in test_queries:
    start_time = time.time()

    try:
        # 도구 실행
        tool = tracker.tools[tool_name]
        result = tool.invoke(query)
        success = True
    except Exception as e:
        success = False

    duration = time.time() - start_time
    tracker.track_call(tool_name, query, success, duration)

    time.sleep(0.1)  # 잠시 대기

# 통계 확인
print("\n📊 도구 사용 통계:")
summary = tracker.get_summary()
for tool_name, stats in summary.items():
    print(f"\n{tool_name}:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

# 시각화
tracker.visualize()
```

## 🚀 실무 활용 예시

### 활용: 법률 상담 챗봇 기초

```python
from langchain.schema import HumanMessage, AIMessage, SystemMessage

class LegalConsultationBot:
    """법률 상담 챗봇"""

    def __init__(self, tools, llm):
        self.tools = {tool.name: tool for tool in tools}
        self.llm = llm.bind_tools(tools)
        self.conversation_history = []

    def consult(self, user_query: str) -> str:
        """법률 상담 처리"""

        # 시스템 메시지
        system_msg = SystemMessage(content="""
        당신은 법률 상담 전문 AI입니다.
        사용자의 질문에 정확하고 친절하게 답변하세요.
        필요한 경우 적절한 검색 도구를 사용하세요.
        """)

        # 대화 이력 추가
        self.conversation_history.append(HumanMessage(content=user_query))

        # LLM 호출
        messages = [system_msg] + self.conversation_history
        response = self.llm.invoke(messages)

        # 도구 호출이 있는 경우
        if response.tool_calls:
            tool_results = []

            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']

                # 도구 실행
                tool = self.tools[tool_name]
                result = tool.invoke(tool_args)
                tool_results.append(f"[{tool_name}] {result}")

            # 검색 결과를 포함하여 최종 답변 생성
            final_prompt = f"""
            사용자 질문: {user_query}

            검색 결과:
            {chr(10).join(tool_results)}

            위 정보를 바탕으로 사용자에게 친절하고 정확한 답변을 제공하세요.
            """

            final_response = self.llm.invoke(final_prompt)
            answer = final_response.content
        else:
            answer = response.content

        # 대화 이력에 추가
        self.conversation_history.append(AIMessage(content=answer))

        return answer

# 사용 예시
bot = LegalConsultationBot(tools, llm)

# 상담 시뮬레이션
queries = [
    "연차휴가를 사용하지 못한 경우 어떻게 되나요?",
    "개인정보 유출 사고가 발생하면 어떤 조치를 취해야 하나요?",
    "전세 계약 만기 시 보증금을 제때 돌려받으려면 어떻게 해야 하나요?"
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"👤 사용자: {query}")
    print('='*60)

    answer = bot.consult(query)
    print(f"🤖 법률 상담 봇:\n{answer}")
```

## 📖 참고 자료

### 공식 문서
- [LangChain Tools 문서](https://python.langchain.com/docs/modules/agents/tools/)
- [ChromaDB 문서](https://docs.trychroma.com/)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Tavily Search API](https://tavily.com/)

### 관련 논문
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)

### 추가 학습 자료
- **LangGraph 활용**: Part 2에서 다룰 예정 - 에이전트 워크플로우 구현
- **RAG 최적화**: 청크 크기, 임베딩 모델 선택, 하이브리드 검색
- **프로덕션 배포**: API 서버, 캐싱, 모니터링

---

**다음 단계**: Part 2에서는 각 법률에 특화된 RAG 에이전트를 구현하고, 질문 라우팅, 답변 평가, HITL(Human-in-the-Loop), Gradio 챗봇을 다룰 예정입니다.
