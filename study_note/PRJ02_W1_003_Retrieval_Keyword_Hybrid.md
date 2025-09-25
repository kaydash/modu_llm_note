# PRJ02_W1_003 검색 성능 향상 기법 매뉴얼 - 키워드 검색 / 하이브리드 검색

## 📋 개요

이 노트북은 RAG 시스템의 검색 성능을 향상시키기 위한 핵심 기법들을 다룹니다. 의미론적 검색(Semantic Search), 키워드 검색(Keyword Search), 그리고 두 방식을 결합한 하이브리드 검색(Hybrid Search)을 비교 분석하여 최적의 검색 전략을 학습할 수 있습니다.

### 🎯 학습 목표
- 키워드 검색과 하이브리드 검색 방식 실습 및 비교 분석
- BM25 알고리즘과 의미론적 검색의 차이점 이해
- EnsembleRetriever를 활용한 하이브리드 검색 구현
- 한국어 텍스트 처리를 위한 Kiwi 형태소 분석기 활용
- K-RAG 패키지를 통한 정량적 성능 평가

## 🛠️ 환경 설정

### 1. 필수 패키지
```python
# 환경변수 및 기본 라이브러리
from dotenv import load_dotenv
import os
from glob import glob
from pprint import pprint
import json

# LangChain 관련
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

# 한국어 처리
from kiwipiepy import Kiwi

# K-RAG 평가 패키지
from krag.tokenizers import KiwiTokenizer
from krag.retrievers import KiWiBM25RetrieverWithScore
from krag.evaluators import RougeOfflineRetrievalEvaluators

# 데이터 분석 및 시각화
import pandas as pd
import matplotlib.pyplot as plt

# Langfuse 트레이싱
from langfuse.callback import CallbackHandler  # 실제 import 경로

# 환경변수 로드
load_dotenv()
```

### 2. 데이터 준비 및 전처리
```python
# 한국어 전기차 데이터 로드 함수
def load_text_files(txt_files):
    data = []
    for text_file in txt_files:
        loader = TextLoader(text_file, encoding='utf-8')
        data += loader.load()
    return data

# 데이터 로드
korean_txt_files = glob(os.path.join('data', '*_KR.md'))
korean_data = load_text_files(korean_txt_files)

# 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    separators=['\n\n', '\n', r'(?<=[.!?])\s+'],
    chunk_size=300,
    chunk_overlap=50,
    is_separator_regex=True,
    keep_separator=True,
)

korean_chunks = text_splitter.split_documents(korean_data)

# Document 객체에 메타데이터 추가
korean_docs = []
for chunk in korean_chunks:
    doc = Document(page_content=chunk.page_content, metadata=chunk.metadata)
    # 회사명 식별을 위한 메타데이터 추가
    doc.metadata['company'] = '테슬라' if '테슬라' in doc.metadata['source'] else '리비안'
    doc.metadata['language'] = 'ko'
    # 출처 정보를 문서 내용에 포함
    doc.page_content = f"[출처] 이 문서는 {doc.metadata['company']}에 대한 문서입니다.\n----------------------------------\n{doc.page_content}"
    korean_docs.append(doc)

print(f"총 {len(korean_docs)}개의 문서 청크 생성됨")
```

## 🔍 RAG 검색기 유형

### 1. Semantic Search (의미론적 검색)

**개념**: 텍스트의 벡터 표현을 활용한 의미적 유사성 기반 검색

**특징**:
- 🎯 **의미적 유사성**: 단어가 달라도 비슷한 의미면 검색 가능
- 🌐 **다국어 지원**: 임베딩 모델의 다국어 능력 활용
- 🔄 **컨텍스트 이해**: 문맥을 고려한 검색

**구현**:
```python
# OpenAI 임베딩 모델 설정
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Chroma 벡터 저장소 생성
chroma_db = Chroma.from_documents(
    documents=korean_docs,
    embedding=embeddings,
    collection_name="db_korean_cosine",
    persist_directory="./chroma_db",
    collection_metadata={'hnsw:space': 'cosine'}
)

# 의미론적 검색기 초기화
chroma_k_retriever = chroma_db.as_retriever(search_kwargs={"k": 2})

# 테스트 검색
query = "리비안은 언제 사업을 시작했나요?"
retrieved_docs = chroma_k_retriever.invoke(query)

print(f"검색된 문서 수: {len(retrieved_docs)}")
for i, doc in enumerate(retrieved_docs):
    print(f"\n[문서 {i+1}] {doc.page_content[:100]}...")
```

### 2. Keyword Search (키워드 검색)

**개념**: BM25 등 전통적 알고리즘 기반의 단어 매칭 방식

**특징**:
- 🎯 **정확한 매칭**: 정확한 키워드 일치에 강점
- ⚡ **빠른 속도**: 계산 효율성이 높음
- 📊 **통계 기반**: TF-IDF, BM25 등 검증된 알고리즘

**BM25 알고리즘 특징**:
- Term Frequency (TF): 문서 내 단어 빈도
- Inverse Document Frequency (IDF): 전체 문서집합에서의 희귀성
- Document Length Normalization: 문서 길이 정규화

**기본 구현**:
```python
# Chroma DB에서 문서 추출
documents = chroma_db.get()["documents"]
metadatas = chroma_db.get()["metadatas"]

# Document 객체로 변환
docs = [Document(page_content=content, metadata=meta)
        for content, meta in zip(documents, metadatas)]

# 기본 BM25 검색기 생성
bm25_retriever = BM25Retriever.from_documents(docs)

# 기본 검색 테스트
query = "리비안은 언제 사업을 시작했나요?"
retrieved_docs = bm25_retriever.invoke(query)

print(f"기본 BM25 검색 결과: {len(retrieved_docs)}개 문서")
for i, doc in enumerate(retrieved_docs):
    print(f"[{i}] BM25 Score: 0.0")  # 키워드 매칭 실패로 인한 낮은 점수
```

**한국어 특화 문제점**:
- "리비안은" → 조사 "은"으로 인해 "리비안"과 매칭 실패
- 한국어 어미 변화와 조사로 인한 검색 성능 저하

### 3. 한국어 특화 BM25 구현

**문제점**: 기본 BM25는 한국어 특성 반영 부족
- 조사, 어미 변화
- 복합어 구조
- 띄어쓰기 불규칙성

**해결책**: Kiwi 한국어 형태소 분석기 활용

```python
from kiwipiepy import Kiwi

# Kiwi 형태소 분석기 초기화
kiwi = Kiwi()

# 전기차 관련 전문 용어 등록
kiwi.add_user_word('리비안', 'NNP')  # 고유명사
kiwi.add_user_word('테슬라', 'NNP')

# 한국어 특화 토크나이징 함수
def bm25_process_func(text, kiwi_model=Kiwi()):
    """
    BM25Retriever에서 사용할 전처리 함수
    한국어 토크나이저를 사용하여 문장을 토큰화
    :param text: 토큰화할 문장
    :param kiwi_model: Kiwi 객체
    """
    return [t.form for t in kiwi_model.tokenize(text)]

# 한국어 특화 BM25 검색기 생성
bm25_retriever = BM25Retriever.from_documents(
    documents=docs,
    preprocess_func=lambda x: bm25_process_func(x, kiwi_model=kiwi),
)

# 개선된 검색 테스트
query = "리비안이 설립된 연도는?"
retrieved_docs = bm25_retriever.invoke(query)

print("=== 한국어 특화 BM25 검색 결과 ===")
for i, doc in enumerate(retrieved_docs[:3]):
    print(f"[{i}] BM25 점수 개선됨")
    print(f"내용: {doc.page_content[:100]}...")
    print("-" * 50)
```

**실제 성능 향상**:
- "리비안은" (실패) → "리비안이" (성공): BM25 점수 4.61 → 7.79로 향상
- 형태소 분석을 통한 어근 추출로 정확한 키워드 매칭

### 4. Hybrid Search (하이브리드 검색)

**개념**: 키워드 검색과 의미론적 검색을 결합한 통합 접근법

**장점**:
- 🎯 **정확성 + 유연성**: 정확한 매칭과 의미적 유사성 모두 활용
- 🔄 **상호 보완**: 각 방식의 약점을 보완
- ⚖️ **가중치 조절**: 상황에 맞는 비중 조정 가능

**K-RAG 패키지를 활용한 고급 구현**:
```python
from krag.tokenizers import KiwiTokenizer
from krag.retrievers import KiWiBM25RetrieverWithScore

# 전문 K-RAG BM25 검색기 (점수 포함)
retriever_bm25_kiwi = KiWiBM25RetrieverWithScore(
    documents=korean_docs,
    kiwi_tokenizer=KiwiTokenizer(model_type='knlm', typos='basic'),
    k=3,
)

# Chroma 검색기
retriever_chroma_db = chroma_db.as_retriever(search_kwargs={"k": 5})

# EnsembleRetriever로 하이브리드 검색
from langchain.retrievers import EnsembleRetriever

ensemble_retriever = EnsembleRetriever(
    retrievers=[retriever_bm25_kiwi, retriever_chroma_db],
    weights=[0.5, 0.5]  # BM25와 Chroma 동일 가중치
)

# 테스트 검색
test_query = "리비안이 설립된 연도는?"
test_results = ensemble_retriever.invoke(test_query)

print(f"하이브리드 검색 결과: {len(test_results)}개 문서 검색됨")
print("구성: BM25 검색기(가중치: 0.5) + Chroma 검색기(가중치: 0.5)")
```

## 📊 성능 평가 시스템

### 1. 테스트 데이터셋 준비

```python
# 테스트 데이터셋 로드 (Excel 파일)
import pandas as pd
df_qa_test = pd.read_excel("data/testset.xlsx")

print("테스트 데이터셋 정보:")
print(f"- 총 질문 수: {len(df_qa_test)}")
print(f"- 첫 번째 질문: {df_qa_test['user_input'].iloc[0]}")
```

### 2. K-RAG 기반 평가 함수

```python
from krag.evaluators import RougeOfflineRetrievalEvaluators

def evaluate_qa_test(df_qa_test, retriever, k=3):
    """QA 테스트 데이터셋에 대한 검색 성능 평가"""

    # 실제 관련 문서와 검색 결과 수집
    actual_docs = []
    predicted_docs = []

    for _, row in df_qa_test.iterrows():
        question = row['user_input']
        reference_contexts = eval(row['reference_contexts'])

        # 참조 컨텍스트를 Document 객체로 변환
        context_docs = [Document(page_content=ctx) for ctx in reference_contexts]
        actual_docs.append(context_docs)

        # 검색 수행
        retrieved_docs = retriever.invoke(question)
        predicted_docs.append(retrieved_docs)

    # ROUGE 기반 평가 수행
    evaluator = RougeOfflineRetrievalEvaluators(
        actual_docs=actual_docs,
        predicted_docs=predicted_docs,
        match_method="rouge2",
        threshold=0.3
    )

    # 평가지표 계산
    hit_rate = evaluator.calculate_hit_rate(k=k)['hit_rate']
    mrr = evaluator.calculate_mrr(k=k)['mrr']
    recall = evaluator.calculate_recall(k=k)['macro_recall']
    precision = evaluator.calculate_precision(k=k)['macro_precision']
    f1_score = evaluator.calculate_f1_score(k=k)['macro_f1']
    map_score = evaluator.calculate_map(k=k)['map']
    ndcg = evaluator.calculate_ndcg(k=k)['ndcg']

    return {
        'hit_rate': hit_rate,
        'mrr': mrr,
        'recall': recall,
        'precision': precision,
        'f1_score': f1_score,
        'map': map_score,
        'ndcg': ndcg
    }
```

### 2. 평가 함수 구현

```python
def evaluate_qa_test(df_qa_test, retriever, k=3):
    """QA 테스트 데이터셋에 대한 검색 성능 평가"""

    # 실제 관련 문서와 검색 결과 수집
    actual_docs = []
    predicted_docs = []

    for _, row in df_qa_test.iterrows():
        question = row['user_input']
        reference_contexts = eval(row['reference_contexts'])  # 문자열을 리스트로 변환

        # 참조 컨텍스트를 Document 객체로 변환
        context_docs = [Document(page_content=ctx) for ctx in reference_contexts]
        actual_docs.append(context_docs)

        # 검색 수행
        retrieved_docs = retriever.invoke(question)
        predicted_docs.append(retrieved_docs)

    # ROUGE 기반 평가 수행
    evaluator = RougeOfflineRetrievalEvaluators(
        actual_docs=actual_docs,
        predicted_docs=predicted_docs,
        match_method="rouge2",
        threshold=0.3
    )

    # 평가지표 계산
    hit_rate = evaluator.calculate_hit_rate(k=k)['hit_rate']
    mrr = evaluator.calculate_mrr(k=k)['mrr']
    recall = evaluator.calculate_recall(k=k)['macro_recall']
    precision = evaluator.calculate_precision(k=k)['macro_precision']
    f1_score = evaluator.calculate_f1_score(k=k)['macro_f1']
    map_score = evaluator.calculate_map(k=k)['map']
    ndcg = evaluator.calculate_ndcg(k=k)['ndcg']

    return {
        'hit_rate': hit_rate,
        'mrr': mrr,
        'recall': recall,
        'precision': precision,
        'f1_score': f1_score,
        'map': map_score,
        'ndcg': ndcg
    }
```

## 🧪 실험 및 비교 분석

### 1. k값에 따른 성능 비교 실험

```python
# k=1~5에 대한 성능 평가 실행
k_values = [1, 2, 3, 4, 5]

# BM25 검색기 성능 (실제 측정 결과)
for k in k_values:
    retriever_bm25_kiwi.k = k
    result = evaluate_qa_test(df_qa_test, retriever_bm25_kiwi, k=k)
    print(f"BM25 k={k}: Hit_Rate={result['hit_rate']:.3f}, MRR={result['mrr']:.3f}")

# 실제 출력 결과:
# BM25 k=1: Hit_Rate=0.286, MRR=0.735
# BM25 k=2: Hit_Rate=0.449, MRR=0.755
# BM25 k=3: Hit_Rate=0.653, MRR=0.769
# BM25 k=4: Hit_Rate=0.735, MRR=0.784
# BM25 k=5: Hit_Rate=0.776, MRR=0.788

# Chroma 검색기 성능 (실제 측정 결과)
for k in k_values:
    retriever_chroma_db.search_kwargs = {"k": k}
    result = evaluate_qa_test(df_qa_test, retriever_chroma_db, k=k)
    print(f"Chroma k={k}: Hit_Rate={result['hit_rate']:.3f}, MRR={result['mrr']:.3f}")

# 실제 출력 결과:
# Chroma k=1: Hit_Rate=0.286, MRR=0.673
# Chroma k=2: Hit_Rate=0.408, MRR=0.714
# Chroma k=3: Hit_Rate=0.551, MRR=0.741
# Chroma k=4: Hit_Rate=0.592, MRR=0.752
# Chroma k=5: Hit_Rate=0.633, MRR=0.756

# Ensemble (하이브리드) 검색기 성능 (실제 측정 결과)
for k in k_values:
    ensemble_result = evaluate_qa_test(df_qa_test, ensemble_retriever, k=k)
    print(f"Ensemble k={k}: Hit_Rate={ensemble_result['hit_rate']:.3f}, MRR={ensemble_result['mrr']:.3f}")

# 실제 출력 결과:
# Ensemble k=1: Hit_Rate=0.286, MRR=0.735
# Ensemble k=2: Hit_Rate=0.469, MRR=0.776
# Ensemble k=3: Hit_Rate=0.633, MRR=0.782
# Ensemble k=4: Hit_Rate=0.714, MRR=0.793
# Ensemble k=5: Hit_Rate=0.776, MRR=0.810
```

### 2. 성능 분석 결과표

```python
# 실제 실험 결과 종합표
results_comparison = pd.DataFrame({
    'k': [1, 2, 3, 4, 5],
    'BM25_Hit_Rate': [0.286, 0.449, 0.653, 0.735, 0.776],
    'BM25_MRR': [0.735, 0.755, 0.769, 0.784, 0.788],
    'Chroma_Hit_Rate': [0.286, 0.408, 0.551, 0.592, 0.633],
    'Chroma_MRR': [0.673, 0.714, 0.741, 0.752, 0.756],
    'Ensemble_Hit_Rate': [0.286, 0.469, 0.633, 0.714, 0.776],
    'Ensemble_MRR': [0.735, 0.776, 0.782, 0.793, 0.810]
})

print("=== 검색 성능 비교 결과 ===")
print(results_comparison)

# 주요 성능 지표 요약
print("\n=== k=3에서의 성능 비교 ===")
print(f"BM25 (키워드):     Hit Rate = 0.653, MRR = 0.769")
print(f"Chroma (의미론적): Hit Rate = 0.551, MRR = 0.741")
print(f"Ensemble (하이브리드): Hit Rate = 0.633, MRR = 0.782")
```

### 3. 성능 시각화

```python
import matplotlib.pyplot as plt

# 한글 폰트 설정 (matplotlib 한글 깨짐 방지)
plt.rcParams['font.family'] = 'DejaVu Sans'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Hit Rate 비교
ax1.plot(results_comparison['k'], results_comparison['BM25_Hit_Rate'],
         'o-', label='BM25 (Keyword)', linewidth=2, markersize=8)
ax1.plot(results_comparison['k'], results_comparison['Chroma_Hit_Rate'],
         's-', label='Chroma (Semantic)', linewidth=2, markersize=8)
ax1.plot(results_comparison['k'], results_comparison['Ensemble_Hit_Rate'],
         '^-', label='Ensemble (Hybrid)', linewidth=2, markersize=8)
ax1.set_xlabel('k (검색 문서 수)')
ax1.set_ylabel('Hit Rate')
ax1.set_title('Hit Rate 비교')
ax1.legend()
ax1.grid(True, alpha=0.3)

# MRR 비교
ax2.plot(results_comparison['k'], results_comparison['BM25_MRR'],
         'o-', label='BM25 (Keyword)', linewidth=2, markersize=8)
ax2.plot(results_comparison['k'], results_comparison['Chroma_MRR'],
         's-', label='Chroma (Semantic)', linewidth=2, markersize=8)
ax2.plot(results_comparison['k'], results_comparison['Ensemble_MRR'],
         '^-', label='Ensemble (Hybrid)', linewidth=2, markersize=8)
ax2.set_xlabel('k (검색 문서 수)')
ax2.set_ylabel('MRR')
ax2.set_title('MRR 비교')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## 🎛️ 하이브리드 검색 최적화

### 1. 가중치 튜닝

```python
def optimize_ensemble_weights(df_qa_test, bm25_retriever, semantic_retriever, k=3):
    """하이브리드 검색의 최적 가중치 탐색"""

    weight_combinations = [
        (0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6), (0.5, 0.5),
        (0.6, 0.4), (0.7, 0.3), (0.8, 0.2), (0.9, 0.1)
    ]

    best_score = 0
    best_weights = (0.5, 0.5)
    results = {}

    for semantic_weight, keyword_weight in weight_combinations:
        ensemble_retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, bm25_retriever],
            weights=[semantic_weight, keyword_weight]
        )

        result = evaluate_qa_test(df_qa_test, ensemble_retriever, k=k)
        f1_score = result['f1_score']

        results[(semantic_weight, keyword_weight)] = result

        if f1_score > best_score:
            best_score = f1_score
            best_weights = (semantic_weight, keyword_weight)

    return best_weights, best_score, results
```

### 2. 동적 가중치 조정

```python
class AdaptiveEnsembleRetriever:
    """쿼리 유형에 따라 동적으로 가중치를 조정하는 검색기"""

    def __init__(self, semantic_retriever, keyword_retriever):
        self.semantic_retriever = semantic_retriever
        self.keyword_retriever = keyword_retriever

    def _analyze_query_type(self, query):
        """쿼리 유형 분석"""
        # 고유명사나 정확한 용어가 많으면 키워드 검색 비중 증가
        proper_nouns = ['테슬라', '리비안', '포드', 'Tesla', 'Rivian']
        exact_terms = ['가격', '수량', '날짜', '수치']

        if any(noun in query for noun in proper_nouns):
            return 'factual'  # 사실적 질문
        elif any(term in query for term in exact_terms):
            return 'precise'  # 정확한 정보 요구
        else:
            return 'conceptual'  # 개념적 질문

    def invoke(self, query):
        """쿼리 유형에 따른 적응적 검색"""
        query_type = self._analyze_query_type(query)

        if query_type == 'factual':
            weights = [0.3, 0.7]  # 키워드 검색 비중 높임
        elif query_type == 'precise':
            weights = [0.2, 0.8]  # 키워드 검색 더욱 강화
        else:
            weights = [0.7, 0.3]  # 의미론적 검색 비중 높임

        ensemble = EnsembleRetriever(
            retrievers=[self.semantic_retriever, self.keyword_retriever],
            weights=weights
        )

        return ensemble.invoke(query)
```

## 💡 검색 방식별 특성 분석

### 1. 실험 결과 분석

**주요 발견사항**:

1. **BM25 (키워드 검색)**:
   - Hit Rate: 0.286 → 0.776 (k=1→5)
   - MRR: 0.735 → 0.788 (안정적 성능)
   - **장점**: 정확한 용어 매칭, 빠른 응답 속도
   - **단점**: 의미적 유사성 파악 한계

2. **Chroma (의미론적 검색)**:
   - Hit Rate: 0.286 → 0.633 (BM25보다 낮음)
   - MRR: 0.673 → 0.756 (BM25보다 낮음)
   - **장점**: 의미적 유사성 파악, 컨텍스트 이해
   - **단점**: 정확한 키워드 매칭에서 상대적 약세

3. **Ensemble (하이브리드 검색)**:
   - Hit Rate: 0.286 → 0.776 (BM25와 동등)
   - MRR: 0.735 → 0.810 (모든 방식 중 최고)
   - **장점**: 두 방식의 장점 결합, 최고 성능
   - **특히 MRR에서 우수한 성능** (첫 번째 정답 위치 최적화)

### 2. 성능 비교표

| 특성 | BM25 (키워드) | Chroma (의미론적) | Ensemble (하이브리드) |
|------|-------------|-----------------|-------------------|
| **Hit Rate @3** | 0.653 🟢 | 0.551 🔶 | 0.633 🟡 |
| **MRR @3** | 0.769 🟡 | 0.741 🔶 | 0.782 🟢 |
| **Hit Rate @5** | 0.776 🟢 | 0.633 🔶 | 0.776 🟢 |
| **MRR @5** | 0.788 🟡 | 0.756 🔶 | 0.810 🟢 |
| **처리 속도** | 🟢 빠름 | 🔶 보통 | 🔶 보통 |
| **메모리 사용량** | 🟢 낮음 | 🔶 높음 | 🔶 높음 |

### 3. 실무 권장사항

```python
# 상황별 최적 검색 전략
def recommend_search_strategy(query_type, performance_priority):
    """
    상황별 검색 전략 추천
    """
    if performance_priority == 'speed':
        return 'BM25'  # 빠른 응답이 중요한 경우
    elif performance_priority == 'accuracy':
        return 'Ensemble'  # 최고 정확도가 중요한 경우
    elif query_type == 'exact_match':
        return 'BM25'  # 정확한 키워드 매칭이 중요한 경우
    elif query_type == 'semantic':
        return 'Chroma'  # 의미적 유사성이 중요한 경우
    else:
        return 'Ensemble'  # 일반적인 경우
```

## 🛠️ 고급 기능 구현

### 1. 다단계 검색 시스템

```python
class MultiStageRetriever:
    """다단계 검색을 통한 성능 향상"""

    def __init__(self, retrievers, thresholds):
        self.retrievers = retrievers
        self.thresholds = thresholds

    def invoke(self, query, target_count=5):
        """단계별 검색 수행"""
        all_results = []

        for i, (retriever, threshold) in enumerate(zip(self.retrievers, self.thresholds)):
            results = retriever.invoke(query)

            # 점수 기반 필터링
            filtered_results = [
                doc for doc in results
                if doc.metadata.get('score', 0) > threshold
            ]

            all_results.extend(filtered_results)

            if len(all_results) >= target_count:
                break

        # 중복 제거 및 상위 결과 반환
        unique_results = self._remove_duplicates(all_results)
        return unique_results[:target_count]

    def _remove_duplicates(self, docs):
        """문서 중복 제거"""
        seen = set()
        unique_docs = []

        for doc in docs:
            content_hash = hash(doc.page_content)
            if content_hash not in seen:
                seen.add(content_hash)
                unique_docs.append(doc)

        return unique_docs
```

### 2. 성능 모니터링 시스템

```python
class RetrievalMonitor:
    """검색 성능 모니터링 및 로깅"""

    def __init__(self):
        self.query_logs = []
        self.performance_history = []

    def log_query(self, query, retriever_type, results, metrics):
        """쿼리 및 결과 로깅"""
        log_entry = {
            'timestamp': datetime.now(),
            'query': query,
            'retriever_type': retriever_type,
            'result_count': len(results),
            'metrics': metrics
        }
        self.query_logs.append(log_entry)

    def analyze_performance_trends(self):
        """성능 트렌드 분석"""
        if not self.query_logs:
            return None

        df = pd.DataFrame(self.query_logs)

        # 검색기별 평균 성능
        performance_by_type = df.groupby('retriever_type').agg({
            'metrics': lambda x: np.mean([m['f1_score'] for m in x])
        })

        return performance_by_type
```

## 🔧 문제 해결 가이드

### 1. 한국어 처리 문제

```python
# 인코딩 오류 해결
def safe_load_korean_text(file_path):
    """안전한 한국어 텍스트 로드"""
    encodings = ['utf-8', 'cp949', 'euc-kr']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Could not decode {file_path}")

# 형태소 분석 오류 처리
def robust_tokenize(text, kiwi_model):
    """견고한 토크나이징"""
    try:
        return bm25_process_func(text, kiwi_model)
    except Exception as e:
        # 기본 공백 분리로 fallback
        return text.split()
```

### 2. 성능 최적화

```python
# 캐싱을 통한 성능 향상
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_embedding(text):
    """임베딩 결과 캐싱"""
    return embedding_model.embed_query(text)

# 배치 처리를 통한 효율성 향상
def batch_search(queries, retriever, batch_size=10):
    """배치 단위 검색 처리"""
    results = []

    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]
        batch_results = [retriever.invoke(q) for q in batch]
        results.extend(batch_results)

    return results
```

## 📚 실무 활용 팁

### 1. 검색 방식 선택 가이드

```python
def recommend_search_strategy(query_characteristics):
    """쿼리 특성에 따른 검색 전략 추천"""

    if query_characteristics['has_exact_terms']:
        return 'keyword_heavy'  # 키워드 검색 비중 높임
    elif query_characteristics['is_conceptual']:
        return 'semantic_heavy'  # 의미론적 검색 비중 높임
    elif query_characteristics['is_multilingual']:
        return 'semantic_only'  # 의미론적 검색만 사용
    else:
        return 'balanced_hybrid'  # 균형 잡힌 하이브리드
```

### 2. A/B 테스트 프레임워크

```python
class RetrievalABTest:
    """검색 시스템 A/B 테스트"""

    def __init__(self, strategy_a, strategy_b):
        self.strategy_a = strategy_a
        self.strategy_b = strategy_b
        self.results = {'a': [], 'b': []}

    def run_test(self, test_queries, metrics=['f1_score', 'mrr']):
        """A/B 테스트 실행"""
        for query in test_queries:
            # 전략 A 테스트
            result_a = self.strategy_a.invoke(query)
            self.results['a'].append(self._evaluate(query, result_a))

            # 전략 B 테스트
            result_b = self.strategy_b.invoke(query)
            self.results['b'].append(self._evaluate(query, result_b))

    def analyze_results(self):
        """결과 분석 및 통계적 유의성 검정"""
        from scipy.stats import ttest_rel

        scores_a = [r['f1_score'] for r in self.results['a']]
        scores_b = [r['f1_score'] for r in self.results['b']]

        t_stat, p_value = ttest_rel(scores_a, scores_b)

        return {
            'mean_a': np.mean(scores_a),
            'mean_b': np.mean(scores_b),
            'p_value': p_value,
            'significant': p_value < 0.05
        }
```

## 🚀 다음 단계

이 노트북을 완료한 후 다음 과정들을 진행하세요:

1. **PRJ02_W1_004**: 쿼리 확장(Query Expansion) 기법 학습
2. **고급 하이브리드 전략**: 더 복잡한 앙상블 방법론
3. **실시간 성능 모니터링**: 프로덕션 환경에서의 성능 추적

---

**💡 핵심 인사이트**: 단일 검색 방식의 한계를 극복하기 위해 하이브리드 접근법이 필수적이며, 특히 한국어와 같은 교착어는 형태소 분석이 검색 성능에 큰 영향을 미칩니다.