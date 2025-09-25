# PRJ02_W1_005 검색 성능 향상 기법 매뉴얼 - 재순위화 & 맥락적 압축

## 📋 개요

이 노트북은 RAG 시스템의 검색 성능을 극대화하기 위한 고급 기법들을 다룹니다. Re-ranking(재순위화)과 Contextual Compression(맥락적 압축) 기법을 통해 검색 결과의 품질과 관련성을 획기적으로 향상시키는 방법을 실습합니다.

### 📊 실험 환경 및 성과 요약
- **시스템 환경**: 16코어 128GB 최적화 시스템
- **실행 시간**: 총 5.475초 (문서 로딩 0.120초, 임베딩 설정 0.611초, 검색 테스트 3.230초)
- **처리 성능**: 89개 청크, 733.1청크/초 분할 성능
- **주요 기술**: CrossEncoderReranker, LLMListwiseRerank, LLMChainFilter, LLMChainExtractor, EmbeddingsFilter
- **테스트 쿼리**: "테슬라 트럭 모델이 있나요?"

### 🎯 학습 목표
- Re-ranking 기법으로 검색 결과의 순위 최적화
- Contextual Compression으로 문서의 관련성 및 압축 수행
- DocumentCompressorPipeline을 통한 다단계 문서 처리
- RAG 체인과 통합된 고성능 검색 시스템 구축

## 🛠️ 환경 설정

### 1. 필수 패키지
```python
# 환경변수 및 기본 라이브러리
from dotenv import load_dotenv
import os
from glob import glob
from pprint import pprint
import json

# LangChain 핵심 모듈
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# 재순위화 및 압축 모듈
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    CrossEncoderReranker,
    LLMListwiseRerank,
    LLMChainFilter,
    LLMChainExtractor,
    EmbeddingsFilter,
    DocumentCompressorPipeline
)
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_transformers import EmbeddingsRedundantFilter

# 성능 모니터링
import asyncio
import time
import psutil
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path

# Langfuse 트레이싱
from langfuse.langchain import CallbackHandler

# 환경변수 로드
load_dotenv()
```

### 2. 벡터 저장소 및 기본 검색기 설정
```python
# 벡터 저장소 로드
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
chroma_db = Chroma(
    embedding_function=embeddings,
    collection_name="db_korean_cosine",
    persist_directory="./chroma_db"
)

# 기본 retriever 초기화
chroma_k_retriever = chroma_db.as_retriever(search_kwargs={"k": 4})

# LangChain 콜백 핸들러 생성
langfuse_handler = CallbackHandler()
```

## 🎯 Re-rank (재순위화) 기법

### 개념과 원리
재순위화는 검색 결과를 재분석하여 최적의 순서로 정렬하는 고도화된 기술입니다:

- **이중 단계 프로세스**: 기본 검색 후 정교한 기준으로 재평가 진행
- **검색 품질 향상**: 체계적인 최적화 방법론으로 관련성 극대화
- **맥락적 이해**: 단순 키워드 매칭을 넘어선 의미적 관련성 분석

### 1. CrossEncoderReranker

**특징**:
- Cross-Encoder 모델을 활용한 정밀한 재정렬
- 쌍(pair) 단위 데이터 처리로 문서-쿼리 관계 분석
- 통합 인코딩 방식으로 유사도 정확도 향상

**구현**:
```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# CrossEncoderReranker 모델 초기화
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")

# CrossEncoderReranker 모델을 사용한 re-ranker 초기화 (top_n: 3)
re_ranker = CrossEncoderReranker(model=model, top_n=3)

# CrossEncoderReranker를 사용한 retriever 초기화
cross_encoder_reranker_retriever = ContextualCompressionRetriever(
    base_compressor=re_ranker,
    base_retriever=chroma_k_retriever
)

# 검색 수행
query = "테슬라 트럭 모델이 있나요?"
retrieved_docs = cross_encoder_reranker_retriever.invoke(query, config={"callbacks": [langfuse_handler]})

for doc in retrieved_docs:
    print(f"{doc.page_content} [출처: {doc.metadata['source']}]")
    print("="*200)
```

### 2. LLMListwiseRerank

**특징**:
- 대규모 언어 모델을 활용한 재순위화
- 쿼리와 문서 간 관련성 분석으로 최적 순서 도출
- 전문화된 재순위화 모델 적용

**구현**:
```python
from langchain.retrievers.document_compressors import LLMListwiseRerank
from langchain_openai import ChatOpenAI

# ChatOpenAI 모델 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# LLMListwiseRerank 모델 초기화 (top_n: 3)
re_ranker = LLMListwiseRerank.from_llm(llm, top_n=3)

# LLMListwiseRerank 모델을 사용한 re-ranker 초기화
llm_reranker_retriever = ContextualCompressionRetriever(
    base_compressor=re_ranker,
    base_retriever=chroma_k_retriever
)

# 검색 수행 및 결과 확인
query = "테슬라 트럭 모델이 있나요?"
retrieved_docs = llm_reranker_retriever.invoke(query, config={"callbacks": [langfuse_handler]})

for doc in retrieved_docs:
    print(f"{doc.page_content} [출처: {doc.metadata['source']}]")
    print("="*200)
```

## 🗜️ Contextual Compression (맥락적 압축) 기법

### 개념과 원리
맥락적 압축은 검색된 문서에서 쿼리와 관련된 핵심 내용만을 선별하고 압축하는 기법입니다:

- **관련성 중심**: 쿼리와 직접적으로 관련된 내용만 추출
- **효율성 향상**: 불필요한 정보 제거로 처리 속도 개선
- **품질 보장**: 핵심 정보 보존으로 답변 품질 유지

### 1. LLMChainFilter

**특징**:
- LLM 기반 필터링으로 문서 포함 여부 결정
- 관련성 기준 문서 선별
- 이진 결정 (포함/제외) 방식

**구현**:
```python
from langchain.retrievers.document_compressors import LLMChainFilter

# ChatOpenAI 모델 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# LLMChainFilter 모델 초기화
context_filter = LLMChainFilter.from_llm(llm)

# LLMChainFilter 모델을 사용한 retriever 초기화
llm_filter_compression_retriever = ContextualCompressionRetriever(
    base_compressor=context_filter,     # LLM 기반 압축기
    base_retriever=chroma_k_retriever   # 기본 검색기
)

# 검색 수행
query = "테슬라 트럭 모델이 있나요?"
retrieved_docs = llm_filter_compression_retriever.invoke(query, config={"callbacks": [langfuse_handler]})
```

### 2. LLMChainExtractor

**특징**:
- LLM 기반 추출로 쿼리 관련 핵심 내용 선별
- 맞춤형 요약을 통한 쿼리 최적화 압축 결과 생성
- 문서 내용의 정교한 편집과 추출

**구현**:
```python
from langchain.retrievers.document_compressors import LLMChainExtractor

# ChatOpenAI 모델 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# LLMChainExtractor 모델 초기화
compressor = LLMChainExtractor.from_llm(llm)

# LLMChainExtractor 모델을 사용한 retriever 초기화
llm_extractor_compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,                        # LLM 기반 압축기
    base_retriever=chroma_k_retriever                  # 기본 검색기
)

# 검색 수행
query = "테슬라 트럭 모델이 있나요?"
retrieved_docs = llm_extractor_compression_retriever.invoke(query, config={"callbacks": [langfuse_handler]})
```

### 3. EmbeddingsFilter

**특징**:
- 임베딩 기반 필터링으로 문서-쿼리 유사도 계산
- 유사도 임계값 기반 문서 선별
- 빠른 처리 속도와 효율적 필터링

**구현**:
```python
from langchain.retrievers.document_compressors import EmbeddingsFilter

# 임베딩 기반 압축기 초기화
embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.4)

# 임베딩 기반 압축기를 사용한 retriever 초기화
embeddings_filter_compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter,    # 임베딩 기반 압축기
    base_retriever=chroma_k_retriever     # 기본 검색기
)

# 검색 수행
query = "테슬라 트럭 모델이 있나요?"
retrieved_docs = embeddings_filter_compression_retriever.invoke(query, config={"callbacks": [langfuse_handler]})
```

### 4. DocumentCompressorPipeline

**특징**:
- 파이프라인 구조로 여러 압축기 순차 연결
- 복합 변환 기능으로 다양한 처리 가능
- 최적화된 다단계 문서 처리

**구현**:
```python
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter

# 임베딩 기반 필터 초기화 - 중복 제거
redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)

# 임베딩 기반 필터 초기화 - 유사도 기반 필터 (임베딩 유사도 0.4 이상)
relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.4)

# Re-ranking 모델 초기화
re_ranker = LLMListwiseRerank.from_llm(llm, top_n=2)

# DocumentCompressorPipeline 초기화 (순차적으로 적용)
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[redundant_filter, relevant_filter, re_ranker]
)

# DocumentCompressorPipeline을 사용한 retriever 초기화
pipeline_compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor,    # DocumentCompressorPipeline 기반 압축기
    base_retriever=chroma_k_retriever,      # 기본 검색기
)
```

## 🚀 고성능 시스템 최적화 실습

### 실습 1: 16코어 128GB 최적화 검색기 설정

**성능 모니터링 클래스**:
```python
class DetailedPerformanceMonitor:
    """상세 시간 측정 및 성능 모니터링 클래스"""
    def __init__(self):
        self.start_time = time.time()
        self.stage_times = {}
        self.stage_starts = {}

    def log_stage_start(self, stage_name):
        """단계 시작 시간 기록"""
        start_time = time.time()
        start_datetime = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        self.stage_starts[stage_name] = start_time

        print(f"⏰ [{start_datetime}] 🚀 {stage_name} 시작")
        print(f"   💻 시작 시 시스템 상태: {self.get_detailed_system_stats()}")

    def log_stage_end(self, stage_name):
        """단계 종료 시간 기록"""
        end_time = time.time()
        end_datetime = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        if stage_name in self.stage_starts:
            start_time = self.stage_starts[stage_name]
            duration = end_time - start_time

            self.stage_times[stage_name] = {
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'start_datetime': start_datetime,
                'end_datetime': end_datetime
            }

            print(f"⏰ [{end_datetime}] ✅ {stage_name} 완료")
            print(f"   ⏱️  소요시간: {duration:.3f}초")
            print(f"   💻 종료 시 시스템 상태: {self.get_detailed_system_stats()}")
            print(f"   📊 성능 요약: {self.get_performance_summary(stage_name)}")

            return duration
        return 0

    def get_detailed_system_stats(self):
        """상세 시스템 통계"""
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
        memory = psutil.virtual_memory()

        return {
            'cpu_avg': f'{cpu_percent:.1f}%',
            'cpu_cores': [f'C{i}:{core:.1f}%' for i, core in enumerate(cpu_per_core)],
            'memory_used': f'{memory.used // (1024**3)}GB/{memory.total // (1024**3)}GB ({memory.percent:.1f}%)',
            'memory_available': f'{memory.available // (1024**3)}GB',
            'active_threads': threading.active_count()
        }
```

**실제 실행 결과**:
```python
# 실제 성능 측정 결과
print("🚀 16코어 128GB 시스템 최적화 초고성능 검색기 실행 (Jupyter 호환)")
print("="*100)

# 총 실행 시간: 5.475초
# 단계별 상세 시간 분석:
# 전체_검색기_초고속_설정: 2.027초 (37.0%)
# 고성능_문서_로딩: 0.120초 (⚡ 초고속)
# 고성능_텍스트_분할: 0.121초 (⚡ 초고속, 89개 청크, 733.1청크/초)
# 초고성능_임베딩_설정: 0.611초 (⚡ 초고속, 64개 동시 요청)
# 초고속_BM25_인덱싱: 0.120초 (⚡ 초고속, 8898.3문서/초)
# 초고속_병렬_검색_테스트: 3.230초 (🚀 빠름, 1.2검색기/초)
```

### 실습 2: 검색기법 고도화

**4단계 파이프라인 구성**:
```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    LLMListwiseRerank,
    LLMChainExtractor,
    EmbeddingsFilter,
    DocumentCompressorPipeline
)
from langchain_community.document_transformers import EmbeddingsRedundantFilter

print("🔧 실습 2: 검색기법 고도화 시작")

# 1) LLM Reranker 설정
print("1️⃣ LLM Reranker 설정...")
llm_reranker = LLMListwiseRerank.from_llm(llm, top_n=3)

# 2) LLM Chain Extractor 설정 (맥락 압축)
print("2️⃣ LLM Chain Extractor 설정...")
llm_extractor = LLMChainExtractor.from_llm(llm)

# 3) Embeddings Filter 설정 (유사도 기반 필터링)
print("3️⃣ Embeddings Filter 설정...")
embeddings_filter = EmbeddingsFilter(
    embeddings=embeddings,
    similarity_threshold=0.5
)

# 4) Redundant Filter 설정 (중복 제거)
print("4️⃣ Redundant Filter 설정...")
redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)

# 5) Pipeline Compressor 구성
print("5️⃣ Pipeline Compressor 구성...")
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[
        redundant_filter,      # 1단계: 중복 문서 제거
        embeddings_filter,     # 2단계: 유사도 기반 필터링
        llm_extractor,         # 3단계: 맥락 압축
        llm_reranker           # 4단계: 재순위화
    ]
)

# 고도화된 검색기 구성
advanced_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor,
    base_retriever=hybrid_retriever
)

print(f"\n🎯 Pipeline 구성:")
print("1. 중복 제거 (EmbeddingsRedundantFilter)")
print("2. 유사도 필터링 (EmbeddingsFilter, threshold=0.5)")
print("3. 맥락 압축 (LLMChainExtractor)")
print("4. 재순위화 (LLMListwiseRerank, top_n=3)")

print(f"\n✅ 실습 2 완료!")
```

**실제 검색 결과 비교**:
```python
# 기본 검색 결과
print("--- 기본 검색 결과 ---")
basic_docs = hybrid_retriever.invoke(query)
print(f"검색된 문서 수: {len(basic_docs)}")  # 결과: 10개

# 고도화된 검색 결과
print("--- 고도화된 검색 결과 (Pipeline) ---")
advanced_docs = advanced_retriever.invoke(query, config={"callbacks": [langfuse_handler]})
print(f"최종 문서 수: {len(advanced_docs)}")  # 결과: 2개 (압축됨)

# 실제 Cybertruck 관련 정보가 정확히 추출됨
print("1. - **Cybertruck:** 2019년 11월에 처음 발표된 풀사이즈 픽업 트럭...")
```

### 실습 3: RAG 체인 연결

**LCEL을 활용한 RAG 체인 구성**:
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# 1) 프롬프트 템플릿 정의
template = """다음 문서들을 참고하여 질문에 답변해주세요:

{context}

질문: {question}
답변:"""

prompt = ChatPromptTemplate.from_template(template)

# 2) LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 3) 문서 포맷팅 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 4) RAG 체인 구성 (LCEL)
rag_chain = (
    RunnableParallel({
        "context": advanced_retriever | format_docs,  # 실습2의 고도화된 검색기 사용
        "question": RunnablePassthrough()
    })
    | prompt
    | llm
    | StrOutputParser()
)

# 5) 테스트 질문들
test_questions = [
    "테슬라 Cybertruck의 특징과 출시년도를 알려주세요",
    "리비안 회사의 주요 전기차 모델은 무엇인가요?",
    "전기차 충전 인프라의 현재 상황을 설명해주세요"
]

print("🚗 RAG 체인 성능 테스트")
print("="*80)

for i, question in enumerate(test_questions, 1):
    print(f"\n질문 {i}: {question}")
    print("-"*80)

    try:
        # RAG 체인 실행
        answer = rag_chain.invoke(question, config={"callbacks": [langfuse_handler]})
        print(f"답변: {answer}")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
```

## 📊 성능 분석 및 최적화

### 실행 시간 분석
```python
# 전체 성능 분석 보고서
print("📊 전체 성능 분석 보고서")
print("="*100)
print("🕐 총 실행 시간: 5.475초")

print("\n📋 단계별 상세 시간 분석:")
print("단계명                       시작시간         종료시간         소요시간       성능평가")
print("-"*90)
print("전체_검색기_초고속_설정         12:51:37.591 12:51:39.618 2.027    s 🚀 빠름 (2.03s)")
print("고성능_문서_로딩               12:51:37.701 12:51:37.821 0.120    s ⚡ 초고속 (120ms)")
print("고성능_텍스트_분할             12:51:37.931 12:51:38.052 0.121    s ⚡ 초고속 (121ms)")
print("초고성능_임베딩_설정           12:51:38.162 12:51:38.772 0.611    s ⚡ 초고속 (611ms)")
print("초고속_BM25_인덱싱            12:51:38.883 12:51:39.003 0.120    s ⚡ 초고속 (120ms)")
print("초고속_병렬_검색_테스트         12:51:39.725 12:51:42.955 3.230    s 🚀 빠름 (3.23s)")

print("\n🏆 최대 시간 소요 단계:")
print("   1. 초고속_병렬_검색_테스트: 3.23초 (59.0%)")
print("   2. 전체_검색기_초고속_설정: 2.027초 (37.0%)")
print("   3. 초고성능_임베딩_설정: 0.611초 (11.2%)")
```

### 검색 성능 비교
```python
# 초고성능 병렬 검색 결과
print("🎯 초고성능 병렬 검색 결과")
print("="*100)

# 키워드 검색 결과
print("🔍 === 키워드 검색 결과 ===")
print("1. Rivian은 \"스케이트보드\" 플랫폼(R1T 및 R1S 모델)을 기반으로 한 전기 스포츠 유틸리티 차량...")

# 하이브리드 검색 결과
print("🔄 === 하이브리드 검색 결과 ===")
print("1. - **Cybertruck:** 2019년 11월에 처음 발표된 풀사이즈 픽업 트럭...")

# 의미 검색 결과
print("🧠 === 의미 검색 결과 ===")
print("1. - **Cybertruck:** Tesla의 전기 픽업 트럭 모델...")

# 멀티쿼리 검색 결과
print("🎯 === 멀티쿼리 검색 결과 ===")
print("1. Tesla Cybertruck 관련 종합 정보...")
```

## 💡 기법별 특성 비교

### Re-ranking 기법 비교

| 기법 | 처리 방식 | 장점 | 단점 | 최적 사용 상황 |
|------|-----------|------|------|----------------|
| **CrossEncoderReranker** | Cross-Encoder 모델 | 높은 정확도, 쌍 단위 분석 | 계산 비용 높음 | 정밀한 재순위화 필요 |
| **LLMListwiseRerank** | LLM 기반 리스트 분석 | 맥락적 이해, 유연성 | 응답 시간 상대적 느림 | 복잡한 관련성 판단 |

### Compression 기법 비교

| 기법 | 압축 방식 | 장점 | 단점 | 최적 사용 상황 |
|------|-----------|------|------|----------------|
| **LLMChainFilter** | 이진 필터링 | 명확한 결정, 빠른 처리 | 정보 손실 위험 | 명확한 관련성 기준 |
| **LLMChainExtractor** | 내용 추출 | 핵심 정보 보존, 맞춤형 요약 | 처리 시간 소요 | 정밀한 내용 추출 |
| **EmbeddingsFilter** | 유사도 기반 | 빠른 처리, 효율적 | 의미적 한계 | 대용량 데이터 처리 |
| **DocumentCompressorPipeline** | 다단계 처리 | 종합적 최적화 | 복잡성 증가 | 최고 품질 추구 |

## 🚀 실무 활용 가이드

### 1. 기법 선택 기준

```python
def select_optimization_strategy(data_size, accuracy_requirement, latency_requirement):
    """상황별 최적화 전략 선택"""

    if latency_requirement == "low" and data_size == "large":
        return {
            "reranking": "EmbeddingsFilter",
            "compression": "EmbeddingsFilter",
            "pipeline": False
        }

    elif accuracy_requirement == "high":
        return {
            "reranking": "CrossEncoderReranker",
            "compression": "LLMChainExtractor",
            "pipeline": True
        }

    else:  # 균형잡힌 성능
        return {
            "reranking": "LLMListwiseRerank",
            "compression": "DocumentCompressorPipeline",
            "pipeline": True
        }
```

### 2. 성능 최적화 팁

```python
# 캐싱을 통한 성능 향상
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_rerank(query, doc_hash):
    """재순위화 결과 캐싱"""
    # 실제 재순위화 로직
    pass

# 배치 처리로 효율성 증대
def batch_compression(docs, batch_size=10):
    """배치 단위 압축 처리"""
    results = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        batch_results = process_batch(batch)
        results.extend(batch_results)
    return results

# 임계값 동적 조정
def adaptive_threshold(query_complexity):
    """쿼리 복잡도에 따른 임계값 조정"""
    if query_complexity == "high":
        return 0.7  # 높은 임계값으로 엄격한 필터링
    elif query_complexity == "medium":
        return 0.5  # 중간 임계값
    else:
        return 0.3  # 낮은 임계값으로 포용적 필터링
```

### 3. 모니터링 및 디버깅

```python
class RetrievalPerformanceMonitor:
    """검색 성능 모니터링"""

    def __init__(self):
        self.metrics = []

    def log_retrieval(self, query, original_docs, final_docs, processing_time):
        """검색 결과 로깅"""
        metric = {
            'timestamp': datetime.now(),
            'query': query,
            'original_count': len(original_docs),
            'final_count': len(final_docs),
            'compression_ratio': len(final_docs) / len(original_docs),
            'processing_time': processing_time
        }
        self.metrics.append(metric)

    def get_performance_summary(self):
        """성능 요약 통계"""
        if not self.metrics:
            return "데이터 없음"

        avg_compression = np.mean([m['compression_ratio'] for m in self.metrics])
        avg_time = np.mean([m['processing_time'] for m in self.metrics])

        return {
            'average_compression_ratio': avg_compression,
            'average_processing_time': avg_time,
            'total_queries': len(self.metrics)
        }
```

## 🔧 문제 해결 및 최적화

### 일반적인 문제들

1. **메모리 부족**
```python
# 배치 크기 조정
batch_size = min(50, available_memory // estimated_doc_size)

# 캐시 크기 제한
@lru_cache(maxsize=100)  # 기본 1000에서 100으로 감소
def cached_function(params):
    pass
```

2. **처리 시간 과다**
```python
# 병렬 처리 활용
from concurrent.futures import ThreadPoolExecutor

def parallel_compression(docs, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(compress_doc, doc) for doc in docs]
        results = [future.result() for future in futures]
    return results
```

3. **품질 저하**
```python
# 다단계 검증
def validate_compression_quality(original, compressed):
    # 핵심 정보 보존 확인
    key_terms = extract_key_terms(original)
    preserved_ratio = count_preserved_terms(compressed, key_terms) / len(key_terms)

    return preserved_ratio > 0.8  # 80% 이상 보존 요구
```

## 📚 고급 활용 패턴

### 1. 적응형 압축 시스템

```python
class AdaptiveCompressionSystem:
    """쿼리 특성에 따른 적응형 압축"""

    def __init__(self):
        self.compression_strategies = {
            'factual': EmbeddingsFilter,
            'analytical': LLMChainExtractor,
            'comparative': DocumentCompressorPipeline
        }

    def classify_query_type(self, query):
        """쿼리 유형 분류"""
        if any(word in query for word in ['무엇', '언제', '어디']):
            return 'factual'
        elif any(word in query for word in ['왜', '어떻게', '분석']):
            return 'analytical'
        elif any(word in query for word in ['비교', '차이', '대비']):
            return 'comparative'
        else:
            return 'factual'

    def compress(self, query, docs):
        """적응형 압축 수행"""
        query_type = self.classify_query_type(query)
        strategy = self.compression_strategies[query_type]
        return strategy.compress(docs)
```

### 2. 다층 품질 보장 시스템

```python
class MultiLayerQualityAssurance:
    """다층 품질 보장 시스템"""

    def __init__(self):
        self.quality_layers = [
            self.relevance_check,
            self.completeness_check,
            self.coherence_check
        ]

    def relevance_check(self, query, docs):
        """관련성 검사"""
        # 임베딩 유사도 기반 관련성 점수
        pass

    def completeness_check(self, original_docs, compressed_docs):
        """완전성 검사"""
        # 핵심 정보 보존 여부 확인
        pass

    def coherence_check(self, docs):
        """일관성 검사"""
        # 문서 간 일관성 및 논리적 흐름 확인
        pass

    def validate(self, query, original_docs, compressed_docs):
        """다층 검증 수행"""
        for layer in self.quality_layers:
            if not layer(query, original_docs, compressed_docs):
                return False
        return True
```

## 📊 실습 완료 요약

### 🎯 실습 성과
1. **Re-ranking 기법 마스터**
   - CrossEncoderReranker와 LLMListwiseRerank 구현
   - 실제 검색 품질 향상 확인

2. **Contextual Compression 구현**
   - 4가지 압축 기법 (Filter, Extractor, EmbeddingsFilter, Pipeline)
   - 문서 수 10개 → 2개로 압축하면서 품질 유지

3. **고성능 시스템 최적화**
   - 16코어 128GB 시스템 최적화로 5.475초 내 전체 처리
   - 89개 청크 생성, 733.1청크/초 성능 달성

4. **RAG 체인 통합**
   - LCEL을 활용한 end-to-end 시스템 구축
   - 실제 Tesla Cybertruck 질문에 대한 정확한 답변 생성

### 📝 핵심 성능 지표
- **문서 로딩**: 0.120초 (⚡ 초고속)
- **텍스트 분할**: 0.121초 (733.1청크/초)
- **임베딩 설정**: 0.611초 (64개 동시 요청)
- **BM25 인덱싱**: 0.120초 (8898.3문서/초)
- **검색 테스트**: 3.230초 (1.2검색기/초)

### 🔍 핵심 학습 내용
1. **재순위화의 중요성**: 기본 검색 결과의 품질을 획기적으로 개선
2. **맥락적 압축의 효과**: 관련성 높은 핵심 정보만 선별하여 효율성 극대화
3. **파이프라인 최적화**: 다단계 처리로 최고 품질의 검색 결과 달성
4. **시스템 최적화**: 하드웨어 자원을 최대 활용한 고성능 구현

---

**💡 핵심 인사이트**: Re-ranking과 Contextual Compression은 RAG 시스템의 성능을 질적으로 향상시키는 핵심 기법입니다. 단순히 더 많은 문서를 검색하는 것이 아니라, 가장 관련성 높고 유용한 정보만을 정확히 선별하여 제공함으로써 사용자 경험을 극대화할 수 있습니다. 본 실습에서는 실제 Tesla Cybertruck 관련 질문에 대해 정확한 정보(2019년 11월 발표, 풀사이즈 픽업 트럭)를 성공적으로 추출하는 것을 확인했습니다.