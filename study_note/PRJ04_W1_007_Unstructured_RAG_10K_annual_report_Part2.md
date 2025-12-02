# PRJ04_W1_007: Unstructured & LangChain을 활용한 10-K 보고서 RAG 시스템 구축 - Part 2

## 📚 학습 목표

Part 2에서는 다음을 학습합니다:

1. **Parent Document Retriever**: 작은 청크로 검색하고 큰 문맥을 반환하는 고급 검색 기법
2. **Multi-Vector Retrieval**: 요약 기반 검색으로 의미론적 매칭 정확도 향상
3. **실전 최적화 기법**:
   - Element 타입별 가중치 부여
   - 청킹 전략 A/B 테스트
   - 하이브리드 검색 (BM25 + Vector)
   - MMR을 활용한 다양성 확보
   - 성능 비교 대시보드

## 🎯 핵심 개념

### Parent Document Retriever

- **문제 상황**: 청킹의 딜레마
  - 작은 청크: 임베딩 정확도 ↑, 문맥 손실 ↑
  - 큰 청크: 문맥 보존 ↑, 검색 정확도 ↓

- **해결 방법**:
  1. **자식 문서 (Child Documents)**: 작은 청크로 벡터 DB에 저장 → 검색에 사용
  2. **부모 문서 (Parent Documents)**: 큰 청크로 문서 저장소에 저장 → LLM에 전달
  3. **ID 매칭**: 자식 문서의 `parent_id`로 부모 문서 조회

- **장점**:
  - 검색 정확도 유지 (작은 청크 임베딩)
  - 문맥 보존 (큰 청크 반환)
  - 메모리 효율성 (중복 저장 최소화)

### Multi-Vector Retrieval

- **핵심 아이디어**: 원본 문서 대신 **요약문**으로 검색
  - 요약문은 핵심 개념만 포함 → 의미론적 매칭 정확도 ↑
  - 검색 후 원본 문서 반환 → 완전한 문맥 제공

- **구성 요소**:
  1. **요약 생성**: LLM으로 각 섹션의 2~3줄 요약 생성
  2. **벡터 저장소**: 요약문 임베딩 저장
  3. **문서 저장소**: 원본 문서 저장
  4. **ID 매칭**: `doc_id`로 연결

## 📖 단계별 구현

### 1단계: Parent Document Retriever 구현

#### 섹션별 문서 분할

Part 1에서 생성한 섹션별 결합 문서(`section_docs_combined`)를 청크로 분할합니다.

```python
from langchain_core.documents import Document
import tiktoken
import pickle

# pickle 파일 로드 (Part 1에서 생성)
with open("data/tesla_10k_sections.pkl", "rb") as f:
    section_docs_combined = pickle.load(f)

# tiktoken 인코더
tokenizer = tiktoken.get_encoding("cl100k_base")

# 토큰 수 기준으로 3000개 이상이면 분할
section_docs_split = {}

for section, doc in section_docs_combined.items():

    # 메타데이터 필터링 (pickle 직렬화 가능한 타입만)
    filtered_metadata = {
        'element_id': doc.metadata['element_id'],
        'parent_id': doc.metadata['parent_id'] if 'parent_id' in doc.metadata else None,
        'source': doc.metadata['source'],
        'page_number': doc.metadata['page_number'],
        'section': section
    }

    # 토큰 수 계산
    tokens = len(tokenizer.encode(doc.page_content))

    if tokens > 3000:
        # 문서 분할 (3000자 단위)
        split_docs = []
        for i in range(0, len(doc.page_content), 3000):
            split_doc = Document(
                page_content=doc.page_content[i:i+3000],
                metadata={
                    **filtered_metadata,
                    "order": i // 3000 + 1  # 순서 정보
                }
            )
            split_docs.append(split_doc)
        section_docs_split[section] = split_docs
    else:
        # 문서 그대로 유지
        doc.metadata = filtered_metadata
        doc.metadata["order"] = 1
        section_docs_split[section] = [doc]

# 총 문서 수 확인
total_docs = sum(len(docs) for docs in section_docs_split.values())
print(f"✅ 분할 완료: {total_docs}개 문서")
```

#### PickleFileStore 정의

부모 문서를 로컬 파일 시스템에 저장하기 위한 커스텀 저장소입니다.

```python
from langchain.storage import LocalFileStore
import pickle

class PickleFileStore(LocalFileStore):
    """Pickle 직렬화를 지원하는 로컬 파일 저장소"""

    def mget(self, keys):
        """Get the values for the given keys."""
        return [
            pickle.loads(v) if v is not None else None
            for v in super().mget(keys)
        ]

    def mset(self, key_value_pairs):
        """Set the values for the given key-value pairs."""
        serialized_pairs = [
            (k, pickle.dumps(v)) for k, v in key_value_pairs
        ]
        super().mset(serialized_pairs)
```

**💡 핵심 포인트:**
- `LangChain의 `LocalFileStore`는 기본적으로 문자열만 저장
- Document 객체를 저장하려면 pickle 직렬화 필요
- `mget`/`mset` 메서드 오버라이드

#### ParentDocumentRetriever 설정

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os

# 저장소 경로 설정
storage_path = "./document_store"
os.makedirs(storage_path, exist_ok=True)

# 부모 문서 저장소 초기화
store = PickleFileStore(storage_path)

# 부모 문서용 텍스트 스플리터 (큰 청크)
parent_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=2000,      # 부모 문서 크기
    chunk_overlap=400,
    separators=["\n\n", "\n", " ", ""]
)

# 자식 문서용 텍스트 스플리터 (작은 청크, 검색용)
child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500,       # 자식 문서 크기
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)

# 벡터 스토어 초기화
vectorstore = Chroma(
    collection_name="tesla_10k_sections",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="./chroma_db"
)

# ParentDocumentRetriever 설정
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# 문서 추가
retriever.add_documents(
    [
        doc
        for docs in section_docs_split.values()
        for doc in docs
    ]
)

print(f"✅ ParentDocumentRetriever 설정 완료")
```

**동작 원리:**
1. 입력 문서를 `parent_splitter`로 분할 → 부모 문서 생성
2. 각 부모 문서를 `child_splitter`로 재분할 → 자식 문서 생성
3. 자식 문서를 벡터 DB에 저장 (임베딩)
4. 부모 문서를 `docstore`에 저장 (ID 매핑)

#### 문서 수 확인

```python
# 벡터 스토어 문서 수 (자식 문서)
print(f"Vectorstore: {vectorstore._collection.count()}개 문서")

# 로컬 저장소 문서 수 (부모 문서)
all_keys = list(store.yield_keys())
print(f"Docstore: {len(all_keys)}개 문서")
```

**예상 출력:**
```
Vectorstore: 1,234개 문서  (작은 자식 청크)
Docstore: 287개 문서       (큰 부모 청크)
```

#### 검색 테스트

```python
# 테스트 쿼리
query = "What are the main risk factors?"

# 검색 실행
retrieved_docs = retriever.invoke(query)

print(f"✅ 검색 결과: {len(retrieved_docs)}개 문서")

# 결과 미리보기
for i, doc in enumerate(retrieved_docs, 1):
    print(f"\n[{i}] Section: {doc.metadata.get('section', 'Unknown')}")
    print(f"{doc.page_content[:500]}...")
    print("-" * 100)
```

#### 저장된 Retriever 재사용

```python
# 벡터 스토어 로드
vectorstore = Chroma(
    collection_name="tesla_10k_sections",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="./chroma_db"
)

# 로컬 파일 저장소 로드
store = PickleFileStore("./document_store")

# ParentDocumentRetriever 재생성
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# 검색 테스트
query = "Where is Tesla's headquarters located?"
retrieved_docs = retriever.invoke(query)

print(f"✅ 재로드 후 검색: {len(retrieved_docs)}개 문서")
```

### 2단계: Multi-Vector Retrieval 구현

#### 요약 생성 함수

```python
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def generate_summary(text):
    """섹션 내용을 2~3줄로 요약"""
    summary_prompt = ChatPromptTemplate.from_template(
        "Summarize the following 10-K report section in 2-3 sentences:\n\n{text}"
    )
    model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    summary_chain = summary_prompt | model | StrOutputParser()
    return summary_chain.invoke({"text": text})

# 테스트
test_text = section_docs_split['Business'][0].page_content[:1000]
summary = generate_summary(test_text)
print(f"📄 원본 (1000자):\n{test_text[:200]}...\n")
print(f"📝 요약:\n{summary}")
```

#### MultiVectorRetriever 설정

```python
import os
import pickle

# 저장소 디렉토리 생성
os.makedirs("./summary_store", exist_ok=True)

# 로컬 파일 저장소 초기화
doc_store = PickleFileStore("./summary_store")

# 벡터 스토어 초기화 (새 컬렉션)
vectorstore = Chroma(
    collection_name="tesla_10k_summaries",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="./chroma_db"
)

# MultiVectorRetriever 설정
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=doc_store,
    id_key="doc_id"   # 문서 ID 키 설정
)

print("✅ MultiVectorRetriever 초기화 완료")
```

#### 요약 생성 및 문서 추가

```python
from langchain_core.documents import Document

summary_docs = []
original_docs = []
doc_ids = []

# 섹션별 문서 리스트
to_index_docs = [
    doc
    for docs in section_docs_split.values()
    for doc in docs
]

# 각 문서에 대한 요약 생성
print(f"🔄 {len(to_index_docs)}개 문서 요약 생성 중...")

for i, doc in enumerate(to_index_docs):
    doc_id = f"doc_{i}"

    # 요약 생성
    summary = generate_summary(doc.page_content)

    # 요약 문서 생성
    summary_doc = Document(
        page_content=summary,
        metadata={"doc_id": doc_id, "section": doc.metadata["section"]}
    )
    summary_docs.append(summary_doc)

    # 원본 문서와 ID 저장
    original_docs.append(doc)
    doc_ids.append(doc_id)

    # 진행 상황 출력
    if (i + 1) % 10 == 0:
        print(f"   {i + 1}/{len(to_index_docs)} 완료...")

# 벡터 스토어에 요약 문서 추가
retriever.vectorstore.add_documents(summary_docs)

# 원본 문서를 docstore에 추가
retriever.docstore.mset(list(zip(doc_ids, original_docs)))

print(f"✅ 요약 기반 인덱싱 완료")
```

**처리 시간:** 약 2~3분 (문서 수에 따라 달라짐)

#### 저장된 문서 수 확인

```python
print(f"Vectorstore (요약): {vectorstore._collection.count()}개")
all_keys = list(doc_store.yield_keys())
print(f"Docstore (원본): {len(all_keys)}개")
```

#### 검색 테스트

```python
# 저장된 Retriever 재로드
doc_store = PickleFileStore("./summary_store")
vectorstore = Chroma(
    collection_name="tesla_10k_summaries",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="./chroma_db"
)

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=doc_store,
    id_key="doc_id"
)

# 검색 실행
query = "What are the main risk factors?"
retrieved_docs = retriever.invoke(query)

print(f"✅ 검색 결과: {len(retrieved_docs)}개 문서")

for i, doc in enumerate(retrieved_docs, 1):
    print(f"\n[{i}] Section: {doc.metadata.get('section')}")
    print(f"{doc.page_content[:500]}...")
    print("-" * 100)
```

#### RAG 체인 구성

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# RAG 프롬프트
rag_prompt = ChatPromptTemplate.from_template("""
Answer the following question based on the provided context from a 10-K report.
If you cannot find the answer in the context, say "제공된 문서에서 답을 찾을 수 없습니다."

<Context>
{context}
</Context>

<Question>
{question}
</Question>

<Answer>""")

# LLM
model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# RAG 체인
chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | model
    | StrOutputParser()
)

# 질문 실행
question = "What are the main risk factors?"
answer = chain.invoke(question)

print(f"🤔 Question: {question}")
print(f"💡 Answer:\n{answer}")
```

## 🛠 실습 문제

### 실습 1: Element 타입별 분리 처리

**목표**: Title 요소에 가중치를 부여하여 섹션 제목 기반 검색 정확도 향상

#### 요소 타입 통계 분석

```python
from collections import Counter
import pandas as pd

# pickle 파일 로드 (Part 1에서 생성한 toc_based_docs_sorted)
with open("data/tesla_10k_toc.pkl", "rb") as f:
    toc_based_docs_sorted = pickle.load(f)

# 요소 타입별 개수
element_counts = Counter([doc.metadata['category'] for doc in toc_based_docs_sorted])

# 요소 타입별 평균 길이
element_stats = {}
for category in element_counts.keys():
    docs = [doc for doc in toc_based_docs_sorted if doc.metadata['category'] == category]
    avg_length = sum(len(doc.page_content) for doc in docs) / len(docs)
    element_stats[category] = {
        'count': element_counts[category],
        'avg_length': int(avg_length)
    }

# 데이터프레임으로 시각화
df_stats = pd.DataFrame(element_stats).T
print("📊 Element 타입별 통계:")
print(df_stats)
```

#### Title 가중치 부여

```python
title_boosted_docs = []

for doc in toc_based_docs_sorted:
    # Title 요소에 is_title 플래그 추가
    if doc.metadata['category'] == 'Title':
        new_doc = Document(
            page_content=doc.page_content,
            metadata={
                **doc.metadata,
                'is_title': True,
                'boost_score': 1.5  # 검색 시 가중치 1.5배
            }
        )
    else:
        new_doc = Document(
            page_content=doc.page_content,
            metadata={
                **doc.metadata,
                'is_title': False,
                'boost_score': 1.0
            }
        )

    title_boosted_docs.append(new_doc)

print(f"✅ Title 가중치 적용 완료: {len(title_boosted_docs)}개 문서")
print(f"   Title 요소: {sum(1 for d in title_boosted_docs if d.metadata['is_title'])}개")
```

### 실습 2: 청킹 전략 A/B 테스트

**목표**: 3가지 청크 크기를 비교하여 최적 파라미터 발견

#### 청킹 함수 정의

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_chunked_docs(docs, chunk_size, overlap, strategy_name):
    """
    문서를 청크로 분할하고 메타데이터에 전략명 추가
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunked_docs = splitter.split_documents(docs)

    # 메타데이터에 청킹 전략 추가
    for doc in chunked_docs:
        doc.metadata['chunking_strategy'] = strategy_name
        doc.metadata['chunk_size'] = chunk_size

    return chunked_docs

print("✅ 청킹 함수 정의 완료")
```

#### 3가지 전략 적용

```python
# 섹션별 문서를 하나의 리스트로 통합
all_section_docs = [
    doc
    for docs in section_docs_split.values()
    for doc in docs
]

# 청킹 전략 정의
strategies = {
    'small': {'chunk_size': 500, 'overlap': 100},
    'medium': {'chunk_size': 1000, 'overlap': 200},
    'large': {'chunk_size': 2000, 'overlap': 400}
}

# 각 전략으로 청킹
chunking_results = {}

for name, params in strategies.items():
    print(f"\n🔄 {name.upper()} 전략 청킹 중...")
    chunked = create_chunked_docs(
        all_section_docs,
        params['chunk_size'],
        params['overlap'],
        name
    )
    chunking_results[name] = chunked
    print(f"   ✅ 완료: {len(chunked)}개 청크 생성")

# 결과 요약
print("\n📊 청킹 전략 비교:")
for name, chunks in chunking_results.items():
    avg_length = sum(len(c.page_content) for c in chunks) / len(chunks)
    print(f"  {name.upper()}: {len(chunks)}개 청크, 평균 {int(avg_length)}자")
```

#### 시각화

```python
import matplotlib.pyplot as plt
import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (name, chunks) in enumerate(chunking_results.items()):
    # 각 청크의 토큰 수 계산
    token_counts = [len(tokenizer.encode(chunk.page_content)) for chunk in chunks]

    # 히스토그램 그리기
    axes[idx].hist(token_counts, bins=30, edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'{name.upper()} Strategy\n({len(chunks)} chunks)')
    axes[idx].set_xlabel('Tokens')
    axes[idx].set_ylabel('Frequency')
    axes[idx].axvline(strategies[name]['chunk_size'], color='red',
                      linestyle='--', label=f"Target: {strategies[name]['chunk_size']}")
    axes[idx].legend()

plt.tight_layout()
plt.savefig('output_images/chunking_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ 시각화 완료: output_images/chunking_comparison.png 저장됨")
```

### 실습 3: 정제 파이프라인 강화

**목표**: Unstructured의 정제 함수를 체계적으로 적용하여 텍스트 품질 향상

#### 정제 함수 적용

```python
from unstructured.cleaners.core import (
    clean_extra_whitespace,
    replace_unicode_quotes,
    clean_non_ascii_chars,
    group_broken_paragraphs
)

def apply_cleaning_pipeline(text):
    """4단계 정제 파이프라인 적용"""
    # 1단계: 공백 정리
    text = clean_extra_whitespace(text)

    # 2단계: 유니코드 따옴표 변환
    text = replace_unicode_quotes(text)

    # 3단계: 비 ASCII 문자 제거
    text = clean_non_ascii_chars(text)

    # 4단계: 문단 재구성
    text = group_broken_paragraphs(text)

    return text

# Medium 전략 문서에 정제 적용
medium_docs = chunking_results['medium']
cleaned_docs = []

for doc in medium_docs:
    cleaned_content = apply_cleaning_pipeline(doc.page_content)
    cleaned_doc = Document(
        page_content=cleaned_content,
        metadata={
            **doc.metadata,
            'cleaned': True
        }
    )
    cleaned_docs.append(cleaned_doc)

# 전/후 비교 샘플
sample_idx = 10
print("🔍 정제 전/후 비교 (샘플):")
print("\n[정제 전]")
print(medium_docs[sample_idx].page_content[:300])
print("\n[정제 후]")
print(cleaned_docs[sample_idx].page_content[:300])

print(f"\n✅ 정제 완료: {len(cleaned_docs)}개 문서")
```

### 실습 4: 하이브리드 검색 구현

**목표**: BM25(키워드) + Vector(의미) 검색을 결합하여 검색 성능 극대화

#### BM25 Retriever 생성

```python
from langchain_community.retrievers import BM25Retriever

# BM25 검색기 생성
bm25_retriever = BM25Retriever.from_documents(cleaned_docs)
bm25_retriever.k = 4  # 상위 4개 반환

# 테스트
test_query = "What are the main risk factors?"
bm25_results = bm25_retriever.invoke(test_query)

print(f"✅ BM25 Retriever 생성 완료")
print(f"   검색 결과: {len(bm25_results)}개 문서")
print(f"\n[상위 1개 결과 미리보기]")
print(bm25_results[0].page_content[:200])
```

#### Vector Retriever 생성

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Chroma vectorstore 생성
vectorstore_hybrid = Chroma(
    collection_name="tesla_10k_hybrid",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="./chroma_db"
)

# 문서 추가
vectorstore_hybrid.add_documents(cleaned_docs)

# Vector 검색기 생성
vector_retriever = vectorstore_hybrid.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# 테스트
vector_results = vector_retriever.invoke(test_query)

print(f"✅ Vector Retriever 생성 완료")
print(f"   벡터 DB: {vectorstore_hybrid._collection.count()}개 문서")
print(f"   검색 결과: {len(vector_results)}개 문서")
```

#### Ensemble Retriever 생성

```python
from langchain.retrievers import EnsembleRetriever

# 하이브리드 검색기 생성
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7]  # BM25: 30%, Vector: 70%
)

# 테스트
ensemble_results = ensemble_retriever.invoke(test_query)

print(f"✅ Ensemble Retriever 생성 완료")
print(f"   가중치: BM25(0.3) + Vector(0.7)")
print(f"   검색 결과: {len(ensemble_results)}개 문서")
```

#### 3가지 검색 방법 비교

```python
test_queries = [
    "What are the main risk factors?",
    "Tesla's revenue sources",
    "Manufacturing facilities location"
]

results_comparison = []

for query in test_queries:
    bm25_docs = bm25_retriever.invoke(query)
    vector_docs = vector_retriever.invoke(query)
    ensemble_docs = ensemble_retriever.invoke(query)

    results_comparison.append({
        'Query': query[:30] + "...",
        'BM25': len(bm25_docs),
        'Vector': len(vector_docs),
        'Ensemble': len(ensemble_docs)
    })

# 데이터프레임 출력
df_comparison = pd.DataFrame(results_comparison)
print("📊 검색 방법 비교:")
print(df_comparison.to_string(index=False))
```

### 실습 5: MMR 및 메타데이터 필터링

**목표**: 검색 다양성 향상 및 특정 섹션 타겟팅

#### MMR 검색 구현

```python
# MMR 검색기 생성
mmr_retriever = vectorstore_hybrid.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,              # 최종 반환 개수
        "fetch_k": 20,       # 초기 후보 개수
        "lambda_mult": 0.7   # 관련성(0.7) vs 다양성(0.3)
    }
)

# 테스트
mmr_results = mmr_retriever.invoke("Tesla's business strategy")

print(f"✅ MMR Retriever 생성 완료")
print(f"   설정: k=4, fetch_k=20, lambda=0.7")
print(f"\n[검색 결과 섹션 다양성 확인]")
sections = [doc.metadata.get('section', 'Unknown') for doc in mmr_results]
print(f"   검색된 섹션: {set(sections)}")
```

**MMR 파라미터 설명:**
- `k`: 최종 반환할 문서 개수
- `fetch_k`: 초기 유사도 검색으로 가져올 후보 개수
- `lambda_mult`: 0에 가까울수록 다양성 우선, 1에 가까울수록 관련성 우선

#### 메타데이터 필터링 검색

```python
def search_with_section_filter(query, section_name, top_k=4):
    """특정 섹션만 검색"""
    # 전체 검색 후 필터링
    all_results = vectorstore_hybrid.similarity_search(query, k=20)

    # 섹션 필터링
    filtered = [
        doc for doc in all_results
        if doc.metadata.get('section') == section_name
    ]

    return filtered[:top_k]

# 테스트: Risk Factors 섹션만 검색
risk_results = search_with_section_filter(
    "What could impact Tesla's business?",
    "Risk Factors"
)

print(f"✅ 섹션 필터링 검색 완료")
print(f"   대상 섹션: Risk Factors")
print(f"   결과: {len(risk_results)}개 문서")

for i, doc in enumerate(risk_results, 1):
    print(f"\n   [{i}] Section: {doc.metadata.get('section')}")
    print(f"       {doc.page_content[:150]}...")
```

### 실습 6: 성능 비교 대시보드

**목표**: 모든 검색 전략의 성능을 정량적으로 비교

#### 성능 측정 함수

```python
import time

def evaluate_retriever(retriever, query, expected_section=None):
    """검색기 성능 평가"""
    # 검색 시간 측정
    start_time = time.time()
    results = retriever.invoke(query)
    latency = (time.time() - start_time) * 1000  # ms

    # 고유 섹션 개수
    sections = [doc.metadata.get('section', 'Unknown') for doc in results]
    unique_sections = len(set(sections))

    # 섹션 일치 확인
    section_match = None
    if expected_section:
        section_match = any(s == expected_section for s in sections)

    return {
        'latency_ms': round(latency, 2),
        'num_results': len(results),
        'unique_sections': unique_sections,
        'section_match': section_match,
        'sections': sections
    }

print("✅ 평가 함수 정의 완료")
```

#### 전체 전략 성능 비교

```python
# 테스트 케이스 정의
test_cases = [
    {"query": "What are Tesla's main risk factors?", "expected": "Risk Factors"},
    {"query": "Tesla's business model", "expected": "Business"},
    {"query": "Financial performance", "expected": "Management's Discussion and Analysis of Financial Condition and Results of Operations"}
]

# 검색기 딕셔너리 (Part 1에서 생성한 retriever 사용)
retrievers_dict = {
    'ParentDoc': retriever,
    'Hybrid': ensemble_retriever,
    'MMR': mmr_retriever
}

# 성능 측정
performance_results = []

for test_case in test_cases:
    query = test_case['query']
    expected = test_case['expected']

    for name, ret in retrievers_dict.items():
        metrics = evaluate_retriever(ret, query, expected)
        performance_results.append({
            'Query': query[:25] + "...",
            'Strategy': name,
            **metrics
        })

# 데이터프레임 생성
df_performance = pd.DataFrame(performance_results)

print("📊 검색 전략 성능 비교:")
print(df_performance[['Query', 'Strategy', 'latency_ms', 'unique_sections', 'section_match']])
```

#### 시각화 대시보드

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 검색 시간 비교 (박스플롯)
strategies_list = ['ParentDoc', 'Hybrid', 'MMR']
latency_data = [
    df_performance[df_performance['Strategy'] == strategy]['latency_ms'].values
    for strategy in strategies_list
]
axes[0, 0].boxplot(latency_data, labels=strategies_list)
axes[0, 0].set_title('검색 시간 비교 (ms)')
axes[0, 0].set_ylabel('Latency (ms)')
axes[0, 0].grid(axis='y', alpha=0.3)

# 2. 고유 섹션 수 비교 (바 차트)
diversity_data = df_performance.groupby('Strategy')['unique_sections'].mean()
axes[0, 1].bar(diversity_data.index, diversity_data.values, color=['#1f77b4', '#2ca02c', '#d62728'])
axes[0, 1].set_title('검색 다양성 (평균 고유 섹션 수)')
axes[0, 1].set_ylabel('Unique Sections')
axes[0, 1].set_ylim(0, 5)
axes[0, 1].grid(axis='y', alpha=0.3)

# 3. 섹션 일치율 (정확도)
accuracy_data = df_performance.groupby('Strategy')['section_match'].mean() * 100
axes[1, 0].bar(accuracy_data.index, accuracy_data.values, color=['#1f77b4', '#2ca02c', '#d62728'])
axes[1, 0].set_title('섹션 일치율 (%)')
axes[1, 0].set_ylabel('Accuracy (%)')
axes[1, 0].set_ylim(0, 100)
axes[1, 0].grid(axis='y', alpha=0.3)

# 4. 종합 스코어 (속도 + 정확도 + 다양성)
normalized_speed = 1 - (df_performance.groupby('Strategy')['latency_ms'].mean() /
                        df_performance['latency_ms'].max())
normalized_accuracy = accuracy_data / 100
normalized_diversity = df_performance.groupby('Strategy')['unique_sections'].mean() / 5

overall_score = (normalized_speed + normalized_accuracy + normalized_diversity) / 3

axes[1, 1].bar(overall_score.index, overall_score.values, color=['#1f77b4', '#2ca02c', '#d62728'])
axes[1, 1].set_title('종합 점수 (속도 + 정확도 + 다양성)')
axes[1, 1].set_ylabel('Score (0-1)')
axes[1, 1].set_ylim(0, 1)
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('output_images/performance_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ 성능 대시보드 생성 완료")
print("   저장: output_images/performance_dashboard.png")

# 최고 성능 전략 추천
best_overall = overall_score.idxmax()
print(f"\n🏆 종합 추천 전략: {best_overall}")
print(f"   종합 점수: {overall_score[best_overall]:.3f}")
```

## 📊 실습 결과 요약

### ✅ 구현 완료 항목

1. **Parent Document Retriever**
   - 작은 청크로 검색, 큰 청크 반환
   - 벡터 DB + 로컬 파일 저장소 활용

2. **Multi-Vector Retrieval**
   - 요약 기반 검색 시스템
   - LLM 요약 생성 → 임베딩 → 원본 반환

3. **Element 타입별 분리 처리**
   - Title 요소에 가중치 부여 (boost_score: 1.5)

4. **청킹 전략 A/B 테스트**
   - Small (500), Medium (1000), Large (2000) 비교
   - 토큰 분포 히스토그램 시각화

5. **정제 파이프라인 강화**
   - 4단계 정제 함수 적용

6. **하이브리드 검색 구현**
   - BM25 + Vector Ensemble
   - 가중치 조합 (0.3 + 0.7)

7. **MMR 및 메타데이터 필터링**
   - 다양성 검색 (lambda=0.7)
   - 섹션별 필터링

8. **성능 비교 대시보드**
   - 속도, 정확도, 다양성 종합 분석

### 📈 주요 개선 사항

| 항목 | 기존 | 개선 | 효과 |
|------|------|------|------|
| 검색 방식 | Vector만 | Hybrid | 정확도 향상 |
| 청킹 전략 | 단일 크기 | 3가지 비교 | 최적화 |
| 문맥 보존 | 작은 청크 | Parent Doc | 문맥 유지 |
| 의미 매칭 | 원본 검색 | 요약 검색 | 정확도 ↑ |
| 다양성 | 없음 | MMR | 중복 제거 |

### 🎯 최종 권장 사항

**추천 아키텍처:**
```
쿼리
  ↓
Ensemble Retriever (BM25 0.3 + Vector 0.7)
  ↓
MMR로 다양성 확보 (lambda=0.7)
  ↓
Parent Document Retriever로 문맥 확장
  ↓
LLM 답변 생성
```

**권장 파라미터:**
- 청킹: Medium (1000 토큰, overlap 200)
- 검색: Hybrid Ensemble
- 다양성: MMR (lambda=0.7)
- 반환: Parent Document (2000 토큰)

## 🚀 다음 단계

이 가이드를 완료하셨다면 다음을 시도해보세요:

1. **Gradio UI 구축**: 대화형 질의응답 인터페이스
2. **배치 평가 시스템**: 자동화된 성능 테스트
3. **다른 10-K 보고서 적용**: Apple, Google, Microsoft 등
4. **시계열 분석**: 여러 연도 보고서 비교
5. **멀티모달 확장**: 차트/그래프 이미지 분석

## 📚 추가 학습 자료

- [Unstructured 공식 문서](https://unstructured-io.github.io/unstructured/)
- [LangChain Retrieval 가이드](https://python.langchain.com/docs/modules/data_connection/retrievers/)
- [Chroma 벡터 DB 문서](https://docs.trychroma.com/)
- [SEC EDGAR 데이터베이스](https://www.sec.gov/edgar)

완료되었습니다! 🎉
