# PRJ02_W1_001 RAG 성능평가 개요 매뉴얼

## 📋 개요

이 노트북은 RAGAS(Retrieval Augmented Generation Assessment)를 사용한 RAG 시스템 성능 평가의 전체적인 프로세스를 다루는 입문 과정입니다. RAG 시스템의 검색과 생성 단계를 체계적으로 평가하는 방법론을 학습할 수 있습니다.

### 🎯 학습 목표
- RAGAS를 사용한 RAG 성능 평가 프로세스 이해
- RAG 시스템의 검색 및 생성 단계별 평가 방법 학습
- 평가 데이터셋 구축 및 활용 방법 습득

## 🛠️ 환경 설정

### 1. 필수 패키지
```python
# 환경변수 로드
from dotenv import load_dotenv

# 기본 라이브러리
import os, glob, json
import pandas as pd
import numpy as np
from pprint import pprint

# LangChain 관련
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# RAGAS 관련
from ragas.testset.persona import Persona
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate, EvaluationDataset
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

# Langfuse 트레이싱
from langfuse.langchain import CallbackHandler
```

### 2. API 키 설정
```bash
# .env 파일에 다음 키들을 설정해야 합니다:
OPENAI_API_KEY=your_openai_api_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_HOST=your_langfuse_host
```

## 🏗️ RAG 시스템 성능 평가 개념

### 평가 체계
RAG 시스템 평가는 크게 두 단계로 구분됩니다:

#### 1. 검색(Retrieval) 단계 평가
- **관련성(Relevance)**: 검색된 문서와 쿼리 간의 연관성
- **정확성(Accuracy)**: 적절한 문서를 식별하는 능력

#### 2. 생성(Generation) 단계 평가
- **연관성(Relevance)**: 응답과 쿼리의 관련성
- **충실도(Faithfulness)**: 응답과 관련 문서 간의 일치도
- **정확성(Correctness)**: 응답과 정답 간의 정확도

### 핵심 성능 지표
- **Latency**: 응답 속도
- **Diversity**: 검색 다양성
- **Noise Robustness**: 잡음 내구성
- **Safety**: 안전성 평가 (오정보 식별, 유해성 등)

## 📊 데이터 준비 및 처리

### 1. 문서 로드 및 전처리

```python
# 텍스트 파일 로드 함수
def load_text_files(txt_files):
    data = []
    for text_file in txt_files:
        loader = TextLoader(text_file, encoding='utf-8')
        data += loader.load()
    return data

# 한국어 문서 로드
korean_txt_files = glob(os.path.join('data', '*_KR.md'))
korean_data = load_text_files(korean_txt_files)
```

### 2. 문서 분할 (Text Splitting)

```python
# 토큰 기반 문서 분할
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    separators=['\n\n', '\n', r'(?<=[.!?])\s+'],
    chunk_size=300,  # 300 토큰 단위로 분할
    chunk_overlap=0,
    is_separator_regex=True,
    keep_separator=True,
)

korean_docs = text_splitter.split_documents(korean_data)
```

### 3. 벡터 저장소 구축

```python
# OpenAI 임베딩 모델 설정
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Chroma 벡터 저장소 생성
vector_store = Chroma.from_documents(
    documents=korean_docs,
    embedding=embedding_model,
    collection_name="db_korean_cosine",
    persist_directory="./chroma_db",
    collection_metadata={'hnsw:space': 'cosine'}
)
```

## 🧪 평가 데이터셋 구축

### 1. Persona 정의
다양한 관점에서 질문을 생성하기 위해 페르소나를 정의합니다:

```python
personas = [
    Persona(
        name="graduate_researcher",
        role_description="미국 전기차 시장을 연구하는 한국인 박사과정 연구원으로, 전기차 정책과 시장 동향에 대해 깊이 있는 분석을 하고 있습니다. 한국어만을 사용합니다."
    ),
    Persona(
        name="masters_student",
        role_description="전기차 산업을 공부하는 한국인 석사과정 학생으로, 미국 전기차 시장의 기초적인 개념과 트렌드를 이해하려 노력하고 있습니다. 한국어만을 사용합니다."
    ),
    # ... 추가 페르소나들
]
```

### 2. 합성 데이터 생성

```python
# LLM 및 임베딩 래퍼 설정
generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini"))
generator_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(model="text-embedding-3-small")
)

# 테스트셋 생성기
generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=generator_embeddings,
    persona_list=personas
)

# 합성 데이터 생성
dataset = generator.generate_with_langchain_docs(
    korean_docs,
    testset_size=50
)
```

## 🔍 RAG 체인 구성

### 1. RAG 체인 구성

```python
# 검색기 생성
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# LLM 및 프롬프트 설정
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

template = """Answer the question based only on the following context:

[Context]
{context}

[Question]
{query}

[Answer]
"""

prompt = ChatPromptTemplate.from_template(template)
qa_chain = prompt | llm | StrOutputParser()

def format_docs(relevant_docs):
    return "\n".join(doc.page_content for doc in relevant_docs)
```

### 2. RAG 체인 실행 예시

```python
query = "Tesla는 언제 누가 만들었나?"
relevant_docs = retriever.invoke(query)
response = qa_chain.invoke({
    "context": format_docs(relevant_docs),
    "query": query
})
```

## 📈 평가 수행

### 1. 평가 데이터셋 준비

```python
# 평가용 데이터셋 구성
dataset = []
for row in testset.itertuples():
    query = row.user_input
    reference = row.reference
    relevant_docs = retriever.invoke(query)
    response = qa_chain.invoke({
        "context": format_docs(relevant_docs),
        "query": query,
    }, config={"callbacks": [langfuse_handler]})

    dataset.append({
        "user_input": query,
        "retrieved_contexts": [rdoc.page_content for rdoc in relevant_docs],
        "response": response,
        "reference": reference,
    })

evaluation_dataset = EvaluationDataset.from_list(dataset)
```

### 2. RAGAS 평가 실행

```python
# 평가자 LLM 설정
evaluator_llm = LangchainLLMWrapper(llm)

# 평가 수행
result = evaluate(
    dataset=evaluation_dataset,
    metrics=[
        LLMContextRecall(),      # 컨텍스트 검색 성능
        Faithfulness(),          # 응답 충실도
        FactualCorrectness()     # 사실 정확성
    ],
    llm=evaluator_llm,
    callbacks=[langfuse_handler]
)
```

### 3. 평가 결과 해석

평가 결과는 0~1 사이의 값으로 제공됩니다:

- **Context Recall (0.8633)**: 검색 성능이 우수함
- **Faithfulness (0.8941)**: 생성된 답변이 검색된 컨텍스트에 충실함
- **Factual Correctness (0.6329)**: 사실 정확성은 개선 여지 있음

## 💡 주요 기능 및 활용법

### 1. 다국어 지원
- 한국어와 영어 문서 모두 처리 가능
- 언어별 적절한 전처리 적용

### 2. 트레이싱 및 모니터링
- Langfuse를 통한 실시간 추적
- 각 단계별 성능 모니터링

### 3. 모듈화된 구조
- 각 구성요소를 독립적으로 교체 가능
- 다양한 실험 설정 지원

## 🔧 문제 해결

### 자주 발생하는 오류들

1. **API 키 오류**
   ```python
   # .env 파일 확인 및 재로드
   load_dotenv()
   ```

2. **메모리 부족**
   ```python
   # 청크 크기 축소
   chunk_size=200  # 300에서 200으로 감소
   ```

3. **검색 성능 저하**
   ```python
   # 임베딩 모델 변경 고려
   embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
   ```

## 📚 참고 자료

- [RAGAS 공식 문서](https://docs.ragas.io/)
- [LangChain 문서](https://python.langchain.com/)
- [OpenAI API 문서](https://platform.openai.com/docs)

## 🚀 다음 단계

1. **PRJ02_W1_002**: 검색 평가 지표 (Hit Rate, MRR, NDCG) 학습
2. **PRJ02_W1_003**: 키워드 검색 및 하이브리드 검색 실습
3. **PRJ02_W1_004**: 쿼리 확장 기법 학습

---

**💡 팁**: 이 노트북을 실행하기 전에 모든 환경변수가 올바르게 설정되었는지 확인하고, 데이터 폴더에 필요한 문서 파일들이 존재하는지 확인하세요.