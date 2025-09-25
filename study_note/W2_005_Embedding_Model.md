# W2_005 임베딩 모델 활용

## 🎯 학습 목표
- 텍스트 임베딩의 개념과 중요성 이해하기
- OpenAI, Hugging Face, Ollama 임베딩 모델 비교 활용하기
- 임베딩 기반 유사도 검색 시스템 구현하기

## 📚 핵심 개념

### 문서 임베딩(Document Embedding)이란?
문서 임베딩은 텍스트를 고정 길이의 벡터(숫자 배열)로 변환하는 과정입니다.

**주요 목적:**
- **의미 표현**: 텍스트의 의미적 특성을 수치화
- **유사도 계산**: 텍스트 간 의미적 유사성 측정
- **벡터 검색**: 고차원 벡터 공간에서의 효율적 검색
- **RAG 구현**: 의미 기반 문서 검색 및 생성

**임베딩 벡터의 특성:**
- **고정 차원**: 모든 텍스트가 같은 길이의 벡터로 변환
- **의미 보존**: 유사한 의미의 텍스트는 가까운 벡터값
- **연산 가능**: 벡터 연산을 통한 의미적 관계 분석

### LangChain 임베딩 모델 종류

#### 1. OpenAI Embeddings
- **특징**: 높은 품질, 다국어 지원, 일관된 성능
- **모델**: text-embedding-3-small/large, text-embedding-ada-002
- **장점**: 우수한 성능, 간편한 사용
- **단점**: API 비용, 인터넷 연결 필요

#### 2. Hugging Face Embeddings
- **특징**: 로컬 실행 가능, 다양한 모델 선택
- **모델**: BAAI/bge-m3, sentence-transformers 계열
- **장점**: 무료 사용, 커스터마이징 가능
- **단점**: 로컬 자원 필요, 초기 다운로드

#### 3. Ollama Embeddings
- **특징**: 경량화, 로컬 서버 기반
- **모델**: nomic-embed-text, bge-m3 등
- **장점**: 빠른 추론, 프라이버시 보호
- **단점**: 별도 서버 설정 필요

## 🔧 환경 설정

### 필수 라이브러리 설치
```bash
# OpenAI 임베딩
pip install langchain-openai

# Hugging Face 임베딩
pip install langchain-huggingface transformers sentence-transformers

# Ollama 임베딩
pip install langchain-ollama

# 유사도 계산
pip install numpy scipy scikit-learn
```

### 환경 변수 설정
```python
from dotenv import load_dotenv
import os
import numpy as np
from typing import List, Tuple, Dict, Any
from pprint import pprint

load_dotenv()

# OpenAI API 키 설정 확인
if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY not set")

# 기본 라이브러리
from langchain_core.documents import Document
from langchain_community.utils.math import cosine_similarity
```

## 💻 코드 예제

### 1. OpenAI 임베딩 모델

#### 기본 사용법
```python
from langchain_openai import OpenAIEmbeddings

def create_openai_embeddings(
    model: str = "text-embedding-3-small",
    dimensions: Optional[int] = None
) -> OpenAIEmbeddings:
    """OpenAI 임베딩 모델 생성"""
    return OpenAIEmbeddings(
        model=model,
        dimensions=dimensions,  # 차원 축소 가능 (None=기본값)
        chunk_size=1000,        # 배치 크기
        max_retries=2,          # 재시도 횟수
        show_progress_bar=False # 진행상태 표시
    )

# 모델 생성
embeddings_openai = create_openai_embeddings()

print(f"모델: {embeddings_openai.model}")
print(f"컨텍스트 길이: {embeddings_openai.embedding_ctx_length}")
print(f"차원: {embeddings_openai.dimensions}")
```

#### 문서 임베딩 생성
```python
def embed_documents_openai(
    documents: List[str],
    model: OpenAIEmbeddings
) -> List[List[float]]:
    """문서 컬렉션 임베딩 생성"""
    document_embeddings = model.embed_documents(documents)

    print(f"임베딩된 문서 수: {len(document_embeddings)}")
    print(f"임베딩 차원: {len(document_embeddings[0])}")

    return document_embeddings

# 예제 문서들
documents = [
    "인공지능은 컴퓨터 과학의 한 분야입니다.",
    "머신러닝은 인공지능의 하위 분야입니다.",
    "딥러닝은 머신러닝의 한 종류입니다.",
    "자연어 처리는 컴퓨터가 인간의 언어를 이해하고 생성하는 기술입니다.",
    "컴퓨터 비전은 컴퓨터가 디지털 이미지나 비디오를 이해하는 방법을 연구합니다."
]

# 문서 임베딩
document_embeddings = embed_documents_openai(documents, embeddings_openai)

# 첫 번째 문서의 임베딩 일부 확인
print(f"첫 번째 문서 임베딩 (처음 10개 값): {document_embeddings[0][:10]}")
```

#### 쿼리 임베딩 및 검색
```python
def find_most_similar_document(
    query: str,
    documents: List[str],
    document_embeddings: List[List[float]],
    embeddings_model: OpenAIEmbeddings
) -> Tuple[str, float, int]:
    """쿼리와 가장 유사한 문서 찾기"""
    # 쿼리 임베딩 생성
    query_embedding = embeddings_model.embed_query(query)

    # 코사인 유사도 계산
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]

    # 가장 유사한 문서 인덱스
    most_similar_idx = np.argmax(similarities)

    return (
        documents[most_similar_idx],
        float(similarities[most_similar_idx]),
        most_similar_idx
    )

# 검색 테스트
queries = [
    "인공지능이란 무엇인가요?",
    "딥러닝과 머신러닝의 관계는 어떻게 되나요?",
    "컴퓨터가 이미지를 이해하는 방법은?"
]

print("=== OpenAI 임베딩 기반 검색 결과 ===")
for query in queries:
    doc, similarity, idx = find_most_similar_document(
        query, documents, document_embeddings, embeddings_openai
    )
    print(f"쿼리: {query}")
    print(f"가장 유사한 문서: {doc}")
    print(f"유사도: {similarity:.4f}")
    print(f"문서 인덱스: {idx}")
    print("-" * 50)
```

#### 차원 비교 분석
```python
def compare_embedding_dimensions(
    texts: List[str],
    dimensions_list: List[int] = [512, 1024, 1536]
) -> Dict[int, Dict[str, Any]]:
    """다양한 차원으로 임베딩 성능 비교"""
    results = {}

    for dim in dimensions_list:
        print(f"\n=== {dim}차원 임베딩 테스트 ===")

        # 모델 생성
        model = create_openai_embeddings(dimensions=dim)

        # 임베딩 생성
        embeddings = model.embed_documents(texts)

        # 통계 계산
        embedding_matrix = np.array(embeddings)

        results[dim] = {
            "model": model,
            "embeddings": embeddings,
            "shape": embedding_matrix.shape,
            "mean": np.mean(embedding_matrix),
            "std": np.std(embedding_matrix),
            "norm_avg": np.mean([np.linalg.norm(emb) for emb in embeddings])
        }

        print(f"임베딩 형태: {embedding_matrix.shape}")
        print(f"평균값: {results[dim]['mean']:.6f}")
        print(f"표준편차: {results[dim]['std']:.6f}")
        print(f"평균 노름: {results[dim]['norm_avg']:.6f}")

    return results

# 차원 비교 실행
sample_texts = [
    "인공지능은 현대 사회를 변화시키고 있다",
    "AI 기술이 우리의 미래를 바꾸고 있다"
]

dimension_results = compare_embedding_dimensions(sample_texts)

# 차원별 유사도 비교
print("\n=== 차원별 코사인 유사도 비교 ===")
for dim, result in dimension_results.items():
    emb1, emb2 = result["embeddings"]
    similarity = cosine_similarity([emb1], [emb2])[0][0]
    print(f"{dim}차원: 유사도 = {similarity:.6f}")
```

### 2. Hugging Face 임베딩 모델

#### 기본 설정 및 사용
```python
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

def create_huggingface_embeddings(
    model_name: str = "BAAI/bge-m3",
    device: str = "cpu"
) -> HuggingFaceEmbeddings:
    """Hugging Face 임베딩 모델 생성"""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},  # "cpu", "cuda", "mps"
        encode_kwargs={"normalize_embeddings": True},  # 정규화
        show_progress=False,
        multi_process=False
    )

# 다양한 모델 비교
hf_models = {
    "bge-m3": "BAAI/bge-m3",                           # 1024차원, 다국어
    "multilingual-e5": "intfloat/multilingual-e5-large", # 1024차원, 다국어
    "bge-small": "BAAI/bge-small-en-v1.5",            # 384차원, 영어
    "all-MiniLM": "sentence-transformers/all-MiniLM-L6-v2"  # 384차원, 다국어
}

# 모델별 성능 비교
def compare_hf_models(documents: List[str], query: str) -> Dict[str, Any]:
    """Hugging Face 모델별 성능 비교"""
    results = {}

    for model_alias, model_name in hf_models.items():
        print(f"\n=== {model_alias} 모델 테스트 ===")

        try:
            # 모델 로드
            embeddings_model = create_huggingface_embeddings(model_name)

            # 문서 임베딩
            doc_embeddings = embeddings_model.embed_documents(documents)

            # 쿼리 검색
            doc, similarity, idx = find_most_similar_document(
                query, documents, doc_embeddings, embeddings_model
            )

            results[model_alias] = {
                "model_name": model_name,
                "embedding_dim": len(doc_embeddings[0]),
                "best_match": doc,
                "similarity": similarity,
                "doc_index": idx
            }

            print(f"모델명: {model_name}")
            print(f"임베딩 차원: {len(doc_embeddings[0])}")
            print(f"최고 유사도: {similarity:.4f}")
            print(f"매칭 문서: {doc[:50]}...")

        except Exception as e:
            print(f"모델 {model_alias} 로드 실패: {e}")
            results[model_alias] = {"error": str(e)}

    return results

# Hugging Face 모델 비교 실행
hf_results = compare_hf_models(documents, "인공지능과 머신러닝의 차이점은?")

# 결과 요약
print("\n=== Hugging Face 모델 성능 요약 ===")
for model_alias, result in hf_results.items():
    if "error" not in result:
        print(f"{model_alias}: {result['embedding_dim']}차원, 유사도 {result['similarity']:.4f}")
```

#### 한국어 특화 모델 사용
```python
def create_korean_embeddings() -> HuggingFaceEmbeddings:
    """한국어에 특화된 임베딩 모델"""
    return HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",  # 한국어 특화
        # model_name="BM-K/KoSimCSE-roberta-multitask",  # 대안 한국어 모델
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

# 한국어 문서 테스트
korean_documents = [
    "서울은 대한민국의 수도입니다.",
    "부산은 대한민국의 제2의 도시입니다.",
    "김치는 한국의 전통 발효 음식입니다.",
    "한글은 세종대왕이 만든 문자입니다.",
    "태극기는 대한민국의 국기입니다."
]

korean_queries = [
    "한국의 수도는 어디인가요?",
    "한국의 전통 음식은 무엇인가요?",
    "한국의 문자는 누가 만들었나요?"
]

def test_korean_performance():
    """한국어 성능 테스트"""
    print("=== 한국어 특화 모델 성능 테스트 ===")

    try:
        korean_embeddings = create_korean_embeddings()
        korean_doc_embeddings = korean_embeddings.embed_documents(korean_documents)

        print(f"한국어 임베딩 차원: {len(korean_doc_embeddings[0])}")

        for query in korean_queries:
            doc, similarity, idx = find_most_similar_document(
                query, korean_documents, korean_doc_embeddings, korean_embeddings
            )
            print(f"\n쿼리: {query}")
            print(f"답변: {doc}")
            print(f"유사도: {similarity:.4f}")

    except Exception as e:
        print(f"한국어 모델 테스트 실패: {e}")

test_korean_performance()
```

### 3. Ollama 임베딩 모델

#### Ollama 설정 및 사용
```python
from langchain_ollama import OllamaEmbeddings

def create_ollama_embeddings(
    model: str = "nomic-embed-text",
    base_url: str = "http://localhost:11434"
) -> OllamaEmbeddings:
    """Ollama 임베딩 모델 생성"""
    return OllamaEmbeddings(
        model=model,
        base_url=base_url,
        # 추가 파라미터
        num_ctx=2048,      # 컨텍스트 길이
        temperature=0.0,   # 결정적 출력
    )

def test_ollama_connection(base_url: str = "http://localhost:11434") -> bool:
    """Ollama 서버 연결 테스트"""
    try:
        import requests
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            available_models = response.json().get("models", [])
            print(f"사용 가능한 Ollama 모델: {[m['name'] for m in available_models]}")
            return True
    except Exception as e:
        print(f"Ollama 서버 연결 실패: {e}")
        return False

    return False

# Ollama 서버 확인 및 모델 테스트
if test_ollama_connection():
    print("\n=== Ollama 임베딩 테스트 ===")

    ollama_models = ["nomic-embed-text", "bge-m3", "all-minilm"]

    for model_name in ollama_models:
        try:
            print(f"\n--- {model_name} 모델 테스트 ---")

            ollama_embeddings = create_ollama_embeddings(model_name)

            # 샘플 임베딩 생성
            sample_embeddings = ollama_embeddings.embed_documents(documents[:2])
            print(f"임베딩 차원: {len(sample_embeddings[0])}")

            # 검색 테스트
            doc, similarity, idx = find_most_similar_document(
                "AI 기술에 대해 알려주세요",
                documents,
                ollama_embeddings.embed_documents(documents),
                ollama_embeddings
            )

            print(f"검색 결과: {doc[:50]}...")
            print(f"유사도: {similarity:.4f}")

        except Exception as e:
            print(f"모델 {model_name} 테스트 실패: {e}")
else:
    print("Ollama 서버가 실행되지 않아 테스트를 건너뜁니다.")
```

### 4. 모델 간 성능 비교

#### 벤치마크 시스템
```python
import time
from typing import Optional

def benchmark_embedding_models(
    documents: List[str],
    queries: List[str],
    models_config: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """임베딩 모델 벤치마크"""

    results = {}

    for model_name, config in models_config.items():
        print(f"\n=== {model_name} 벤치마크 ===")

        try:
            # 모델 생성
            start_time = time.time()

            if config["type"] == "openai":
                model = OpenAIEmbeddings(
                    model=config["model_name"],
                    dimensions=config.get("dimensions")
                )
            elif config["type"] == "huggingface":
                model = HuggingFaceEmbeddings(
                    model_name=config["model_name"],
                    model_kwargs={"device": config.get("device", "cpu")}
                )
            elif config["type"] == "ollama":
                model = OllamaEmbeddings(
                    model=config["model_name"],
                    base_url=config.get("base_url", "http://localhost:11434")
                )

            model_load_time = time.time() - start_time

            # 문서 임베딩 생성
            start_time = time.time()
            doc_embeddings = model.embed_documents(documents)
            embed_time = time.time() - start_time

            # 쿼리별 검색 성능
            query_results = []
            total_search_time = 0

            for query in queries:
                start_time = time.time()
                doc, similarity, idx = find_most_similar_document(
                    query, documents, doc_embeddings, model
                )
                search_time = time.time() - start_time

                query_results.append({
                    "query": query,
                    "best_match": doc,
                    "similarity": similarity,
                    "search_time": search_time
                })

                total_search_time += search_time

            # 결과 저장
            results[model_name] = {
                "config": config,
                "model_load_time": model_load_time,
                "embedding_time": embed_time,
                "embedding_dimension": len(doc_embeddings[0]),
                "total_search_time": total_search_time,
                "avg_search_time": total_search_time / len(queries),
                "query_results": query_results,
                "success": True
            }

            print(f"모델 로드: {model_load_time:.3f}초")
            print(f"임베딩 생성: {embed_time:.3f}초")
            print(f"평균 검색 시간: {results[model_name]['avg_search_time']:.3f}초")
            print(f"임베딩 차원: {len(doc_embeddings[0])}")

        except Exception as e:
            print(f"벤치마크 실패: {e}")
            results[model_name] = {
                "config": config,
                "error": str(e),
                "success": False
            }

    return results

# 벤치마크 실행
benchmark_config = {
    "OpenAI-Small": {
        "type": "openai",
        "model_name": "text-embedding-3-small"
    },
    "OpenAI-Small-512": {
        "type": "openai",
        "model_name": "text-embedding-3-small",
        "dimensions": 512
    },
    "BGE-M3": {
        "type": "huggingface",
        "model_name": "BAAI/bge-m3"
    },
    "MiniLM": {
        "type": "huggingface",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2"
    }
}

# 벤치마크 데이터
test_documents = documents + [
    "자연어 처리 기술은 텍스트 분석에 활용됩니다.",
    "컴퓨터 비전은 이미지 인식 기술입니다.",
    "로봇공학은 인공지능과 기계공학을 결합합니다."
]

test_queries = [
    "AI 기술은 무엇인가요?",
    "머신러닝 알고리즘에 대해 설명해주세요.",
    "딥러닝과 신경망의 관계는?",
    "자연어 처리의 응용 분야는?"
]

benchmark_results = benchmark_embedding_models(
    test_documents,
    test_queries,
    benchmark_config
)

# 벤치마크 결과 요약
print("\n" + "="*60)
print("벤치마크 결과 요약")
print("="*60)

for model_name, result in benchmark_results.items():
    if result["success"]:
        print(f"\n{model_name}:")
        print(f"  차원: {result['embedding_dimension']}")
        print(f"  임베딩 시간: {result['embedding_time']:.3f}초")
        print(f"  평균 검색 시간: {result['avg_search_time']:.3f}초")

        # 평균 유사도
        similarities = [qr["similarity"] for qr in result["query_results"]]
        avg_similarity = sum(similarities) / len(similarities)
        print(f"  평균 유사도: {avg_similarity:.4f}")
    else:
        print(f"\n{model_name}: 실패 - {result['error']}")
```

## 🚀 실습해보기

### 실습 1: 다국어 임베딩 비교
다양한 언어로 된 텍스트의 임베딩 품질을 비교해보세요.

```python
def multilingual_embedding_test():
    """다국어 임베딩 성능 테스트"""
    multilingual_texts = {
        "korean": "인공지능은 미래를 바꿀 기술입니다.",
        "english": "Artificial intelligence is technology that will change the future.",
        "japanese": "人工知能は未来を変える技術です。",
        "chinese": "人工智能是将改变未来的技术。"
    }

    # TODO: OpenAI와 다국어 HuggingFace 모델로 각각 임베딩
    # TODO: 언어별 임베딩 벡터 분석
    # TODO: 언어 간 의미적 유사도 계산
    # TODO: 결과 비교 및 분석
    pass
```

### 실습 2: 도메인 특화 임베딩 최적화
특정 도메인(의료, 법률, 기술 등)에 최적화된 임베딩을 구현해보세요.

```python
def domain_specific_embeddings():
    """도메인 특화 임베딩 최적화"""
    domains = {
        "medical": [
            "당뇨병은 혈당 조절 장애로 인한 질병입니다.",
            "고혈압은 심혈관 질환의 주요 위험 요인입니다.",
            "MRI는 자기공명영상을 이용한 진단 방법입니다."
        ],
        "legal": [
            "계약서는 당사자 간의 약속을 명문화한 문서입니다.",
            "저작권은 창작물에 대한 독점적 권리입니다.",
            "민법은 개인 간의 법률 관계를 규정합니다."
        ],
        "technical": [
            "API는 응용 프로그램 간 상호작용을 위한 인터페이스입니다.",
            "클라우드 컴퓨팅은 인터넷을 통한 컴퓨팅 서비스입니다.",
            "블록체인은 분산 원장 기술입니다."
        ]
    }

    # TODO: 도메인별로 최적화된 임베딩 모델 선택
    # TODO: 도메인 내 문서 간 유사도 vs 도메인 간 유사도 비교
    # TODO: 도메인 특화 검색 성능 평가
    pass
```

### 실습 3: 임베딩 기반 클러스터링
임베딩 벡터를 활용한 문서 클러스터링 시스템을 구현해보세요.

```python
def embedding_based_clustering():
    """임베딩 기반 문서 클러스터링"""
    # TODO: 다양한 주제의 문서들 수집
    # TODO: 문서들을 임베딩으로 변환
    # TODO: K-means 또는 DBSCAN 클러스터링 적용
    # TODO: 클러스터별 대표 문서 선정
    # TODO: 클러스터링 결과 시각화 (t-SNE, UMAP)
    pass
```

## 📋 해답

### 실습 1 해답: 다국어 임베딩 비교
```python
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

def multilingual_embedding_test():
    """다국어 임베딩 성능 테스트"""
    multilingual_texts = {
        "korean": "인공지능은 미래를 바꿀 기술입니다.",
        "english": "Artificial intelligence is technology that will change the future.",
        "japanese": "人工知能は未来を変える技術です。",
        "chinese": "人工智能是将改变未来的技术。"
    }

    # 모델들 설정
    models = {
        "OpenAI": OpenAIEmbeddings(model="text-embedding-3-small"),
        "BGE-M3": HuggingFaceEmbeddings(model_name="BAAI/bge-m3"),
        "Multilingual-E5": HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    }

    results = {}

    for model_name, model in models.items():
        print(f"\n=== {model_name} 다국어 테스트 ===")

        try:
            # 각 언어별 임베딩 생성
            language_embeddings = {}
            for lang, text in multilingual_texts.items():
                embedding = model.embed_query(text)
                language_embeddings[lang] = embedding
                print(f"{lang}: 임베딩 차원 {len(embedding)}")

            # 언어 간 유사도 매트릭스 계산
            languages = list(multilingual_texts.keys())
            embeddings_matrix = [language_embeddings[lang] for lang in languages]

            similarity_matrix = cosine_similarity(embeddings_matrix)

            # 결과 저장
            results[model_name] = {
                "embeddings": language_embeddings,
                "similarity_matrix": similarity_matrix,
                "languages": languages
            }

            # 유사도 매트릭스 출력
            print("언어 간 코사인 유사도:")
            print("        ", "  ".join(f"{lang:8}" for lang in languages))
            for i, lang1 in enumerate(languages):
                row = f"{lang1:8}"
                for j, lang2 in enumerate(languages):
                    row += f"  {similarity_matrix[i][j]:6.4f}"
                print(row)

            # 평균 유사도 계산 (대각선 제외)
            non_diagonal_similarities = []
            for i in range(len(languages)):
                for j in range(len(languages)):
                    if i != j:
                        non_diagonal_similarities.append(similarity_matrix[i][j])

            avg_similarity = np.mean(non_diagonal_similarities)
            print(f"평균 언어 간 유사도: {avg_similarity:.4f}")

        except Exception as e:
            print(f"모델 {model_name} 테스트 실패: {e}")
            results[model_name] = {"error": str(e)}

    # 모델별 성능 비교
    print("\n" + "="*50)
    print("모델별 다국어 성능 비교")
    print("="*50)

    for model_name, result in results.items():
        if "error" not in result:
            # 평균 유사도 계산
            sim_matrix = result["similarity_matrix"]
            non_diagonal = []
            for i in range(len(sim_matrix)):
                for j in range(len(sim_matrix)):
                    if i != j:
                        non_diagonal.append(sim_matrix[i][j])

            avg_sim = np.mean(non_diagonal)
            std_sim = np.std(non_diagonal)

            print(f"{model_name}:")
            print(f"  평균 언어간 유사도: {avg_sim:.4f} (±{std_sim:.4f})")

            # 가장 유사한 언어 쌍
            max_sim = 0
            max_pair = None
            for i, lang1 in enumerate(result["languages"]):
                for j, lang2 in enumerate(result["languages"]):
                    if i != j and sim_matrix[i][j] > max_sim:
                        max_sim = sim_matrix[i][j]
                        max_pair = (lang1, lang2)

            if max_pair:
                print(f"  가장 유사한 언어 쌍: {max_pair[0]}-{max_pair[1]} ({max_sim:.4f})")

    return results

# 다국어 테스트 실행
multilingual_results = multilingual_embedding_test()
```

### 실습 2 해답: 도메인 특화 임베딩 최적화
```python
from sklearn.metrics import silhouette_score
from collections import defaultdict

def domain_specific_embeddings():
    """도메인 특화 임베딩 최적화"""
    domains = {
        "medical": [
            "당뇨병은 혈당 조절 장애로 인한 질병입니다.",
            "고혈압은 심혈관 질환의 주요 위험 요인입니다.",
            "MRI는 자기공명영상을 이용한 진단 방법입니다.",
            "항생제는 세균 감염을 치료하는 약물입니다.",
            "백신은 면역 체계를 강화하여 질병을 예방합니다."
        ],
        "legal": [
            "계약서는 당사자 간의 약속을 명문화한 문서입니다.",
            "저작권은 창작물에 대한 독점적 권리입니다.",
            "민법은 개인 간의 법률 관계를 규정합니다.",
            "형법은 범죄와 형벌에 관한 법률입니다.",
            "헌법은 국가의 기본법으로 최고 규범입니다."
        ],
        "technical": [
            "API는 응용 프로그램 간 상호작용을 위한 인터페이스입니다.",
            "클라우드 컴퓨팅은 인터넷을 통한 컴퓨팅 서비스입니다.",
            "블록체인은 분산 원장 기술입니다.",
            "머신러닝은 데이터로부터 패턴을 학습하는 기술입니다.",
            "데이터베이스는 구조화된 데이터를 저장하고 관리하는 시스템입니다."
        ]
    }

    # 모든 문서와 도메인 라벨 준비
    all_documents = []
    domain_labels = []
    document_to_domain = {}

    for domain, docs in domains.items():
        for doc in docs:
            all_documents.append(doc)
            domain_labels.append(domain)
            document_to_domain[doc] = domain

    # 다양한 임베딩 모델 테스트
    models_to_test = {
        "OpenAI-General": OpenAIEmbeddings(model="text-embedding-3-small"),
        "BGE-M3": HuggingFaceEmbeddings(model_name="BAAI/bge-m3"),
        # "BioBERT": HuggingFaceEmbeddings(model_name="dmis-lab/biobert-base-cased-v1.1"),  # 의료 특화
        # "LegalBERT": HuggingFaceEmbeddings(model_name="nlpaueb/legal-bert-base-uncased"),  # 법률 특화
    }

    domain_analysis_results = {}

    for model_name, model in models_to_test.items():
        print(f"\n=== {model_name} 도메인 분석 ===")

        try:
            # 모든 문서 임베딩
            all_embeddings = model.embed_documents(all_documents)

            # 도메인 내 유사도 vs 도메인 간 유사도 분석
            within_domain_similarities = []
            between_domain_similarities = []

            for i, doc1 in enumerate(all_documents):
                for j, doc2 in enumerate(all_documents):
                    if i != j:
                        similarity = cosine_similarity(
                            [all_embeddings[i]],
                            [all_embeddings[j]]
                        )[0][0]

                        if document_to_domain[doc1] == document_to_domain[doc2]:
                            within_domain_similarities.append(similarity)
                        else:
                            between_domain_similarities.append(similarity)

            # 통계 계산
            within_mean = np.mean(within_domain_similarities)
            within_std = np.std(within_domain_similarities)
            between_mean = np.mean(between_domain_similarities)
            between_std = np.std(between_domain_similarities)

            # 도메인 구분 능력 측정 (클러스터링 품질)
            embeddings_array = np.array(all_embeddings)

            # 실루엣 스코어 계산
            try:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                numeric_labels = le.fit_transform(domain_labels)
                silhouette = silhouette_score(embeddings_array, numeric_labels)
            except:
                silhouette = 0.0

            # 도메인별 중심점 계산
            domain_centers = {}
            for domain in domains.keys():
                domain_embeddings = [
                    all_embeddings[i] for i, label in enumerate(domain_labels)
                    if label == domain
                ]
                domain_centers[domain] = np.mean(domain_embeddings, axis=0)

            # 결과 저장
            domain_analysis_results[model_name] = {
                "within_domain_mean": within_mean,
                "within_domain_std": within_std,
                "between_domain_mean": between_mean,
                "between_domain_std": between_std,
                "separation_ratio": within_mean / between_mean if between_mean > 0 else 0,
                "silhouette_score": silhouette,
                "domain_centers": domain_centers,
                "all_embeddings": all_embeddings
            }

            print(f"도메인 내 평균 유사도: {within_mean:.4f} (±{within_std:.4f})")
            print(f"도메인 간 평균 유사도: {between_mean:.4f} (±{between_std:.4f})")
            print(f"분리 비율: {within_mean/between_mean:.4f}" if between_mean > 0 else "분리 비율: N/A")
            print(f"실루엣 스코어: {silhouette:.4f}")

            # 도메인별 검색 성능 테스트
            domain_queries = {
                "medical": "혈압과 심장병의 관계는?",
                "legal": "계약 위반 시 법적 책임은?",
                "technical": "클라우드와 데이터베이스 연동 방법은?"
            }

            print("\n도메인 특화 검색 테스트:")
            for query_domain, query in domain_queries.items():
                doc, similarity, idx = find_most_similar_document(
                    query, all_documents, all_embeddings, model
                )
                predicted_domain = domain_labels[idx]

                correct = predicted_domain == query_domain
                print(f"  {query_domain} 쿼리: {'✓' if correct else '✗'} "
                      f"(예측: {predicted_domain}, 유사도: {similarity:.4f})")

        except Exception as e:
            print(f"모델 {model_name} 분석 실패: {e}")
            domain_analysis_results[model_name] = {"error": str(e)}

    # 최종 결과 비교
    print("\n" + "="*60)
    print("도메인 특화 성능 비교 요약")
    print("="*60)

    for model_name, result in domain_analysis_results.items():
        if "error" not in result:
            print(f"\n{model_name}:")
            print(f"  도메인 분리 능력: {result['separation_ratio']:.4f}")
            print(f"  클러스터링 품질: {result['silhouette_score']:.4f}")
            print(f"  도메인 내 일관성: {result['within_domain_mean']:.4f}")

    return domain_analysis_results

# 도메인 특화 분석 실행
domain_results = domain_specific_embeddings()
```

### 실습 3 해답: 임베딩 기반 클러스터링
```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def embedding_based_clustering():
    """임베딩 기반 문서 클러스터링"""

    # 다양한 주제의 문서들
    diverse_documents = [
        # AI/Tech 클러스터
        "인공지능은 컴퓨터가 인간의 지능을 모방하는 기술입니다.",
        "머신러닝은 데이터로부터 패턴을 학습하는 알고리즘입니다.",
        "딥러닝은 신경망을 이용한 기계학습 방법입니다.",
        "자연어 처리는 컴퓨터가 인간의 언어를 이해하는 기술입니다.",

        # 의료/건강 클러스터
        "당뇨병은 혈당 조절에 문제가 있는 질병입니다.",
        "고혈압은 심혈관 질환의 주요 위험 요인입니다.",
        "운동은 건강 유지에 필수적인 활동입니다.",
        "균형 잡힌 식단은 건강한 삶의 기초입니다.",

        # 교육/학습 클러스터
        "교육은 개인의 성장과 발전을 위한 과정입니다.",
        "온라인 학습은 디지털 시대의 새로운 교육 방법입니다.",
        "독서는 지식 습득과 사고력 향상에 도움이 됩니다.",
        "창의성 교육은 미래 인재 양성의 핵심입니다.",

        # 환경/지속가능성 클러스터
        "기후 변화는 전 지구적 환경 문제입니다.",
        "재생 에너지는 지속 가능한 발전의 열쇠입니다.",
        "플라스틱 오염은 해양 생태계를 위협합니다.",
        "친환경 기술은 환경 보호의 중요한 수단입니다."
    ]

    # 실제 주제 라벨 (검증용)
    true_labels = [
        "AI/Tech", "AI/Tech", "AI/Tech", "AI/Tech",
        "Healthcare", "Healthcare", "Healthcare", "Healthcare",
        "Education", "Education", "Education", "Education",
        "Environment", "Environment", "Environment", "Environment"
    ]

    # 임베딩 모델 선택
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    print("=== 문서 클러스터링 시스템 ===")

    # 1. 문서 임베딩 생성
    print("1. 문서 임베딩 생성 중...")
    document_embeddings = embedding_model.embed_documents(diverse_documents)
    embeddings_array = np.array(document_embeddings)

    print(f"   - 문서 수: {len(diverse_documents)}")
    print(f"   - 임베딩 차원: {len(document_embeddings[0])}")

    # 2. K-means 클러스터링
    print("\n2. K-means 클러스터링...")
    n_clusters = 4  # 실제로는 4개 주제

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(embeddings_array)

    # K-means 결과 분석
    kmeans_silhouette = silhouette_score(embeddings_array, kmeans_labels)

    print(f"   - 클러스터 수: {n_clusters}")
    print(f"   - 실루엣 스코어: {kmeans_silhouette:.4f}")

    # 3. DBSCAN 클러스터링
    print("\n3. DBSCAN 클러스터링...")

    # 적절한 eps 값 찾기 (간단한 휴리스틱)
    from sklearn.neighbors import NearestNeighbors

    neighbors = NearestNeighbors(n_neighbors=4)
    neighbors_fit = neighbors.fit(embeddings_array)
    distances, indices = neighbors_fit.kneighbors(embeddings_array)
    distances = np.sort(distances[:, 3])  # 4번째 최근접 이웃 거리

    # 거리의 변화율이 큰 지점을 eps로 사용
    eps = np.percentile(distances, 75)

    dbscan = DBSCAN(eps=eps, min_samples=2)
    dbscan_labels = dbscan.fit_predict(embeddings_array)

    n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)

    print(f"   - 자동 감지된 클러스터 수: {n_clusters_dbscan}")
    print(f"   - 노이즈 포인트 수: {n_noise}")

    if n_clusters_dbscan > 0:
        # 노이즈 포인트 제외하고 실루엣 스코어 계산
        non_noise_mask = dbscan_labels != -1
        if np.sum(non_noise_mask) > 1 and len(set(dbscan_labels[non_noise_mask])) > 1:
            dbscan_silhouette = silhouette_score(
                embeddings_array[non_noise_mask],
                dbscan_labels[non_noise_mask]
            )
            print(f"   - 실루엣 스코어: {dbscan_silhouette:.4f}")
        else:
            dbscan_silhouette = 0
            print(f"   - 실루엣 스코어: N/A (클러스터 부족)")
    else:
        dbscan_silhouette = 0
        print(f"   - 실루엣 스코어: N/A (클러스터 없음)")

    # 4. 클러스터별 대표 문서 선정
    print("\n4. 클러스터별 대표 문서 선정...")

    def find_cluster_representative(cluster_embeddings, cluster_docs, cluster_indices):
        """클러스터 중심에서 가장 가까운 문서 찾기"""
        if len(cluster_embeddings) == 0:
            return None, None, -1

        cluster_center = np.mean(cluster_embeddings, axis=0)

        # 중심점과의 거리 계산
        distances = [
            np.linalg.norm(embedding - cluster_center)
            for embedding in cluster_embeddings
        ]

        closest_idx = np.argmin(distances)
        return cluster_docs[closest_idx], distances[closest_idx], cluster_indices[closest_idx]

    # K-means 클러스터별 대표 문서
    print("\n   K-means 클러스터별 대표 문서:")
    for cluster_id in range(n_clusters):
        cluster_mask = kmeans_labels == cluster_id
        cluster_docs = [diverse_documents[i] for i, mask in enumerate(cluster_mask) if mask]
        cluster_embeddings = embeddings_array[cluster_mask]
        cluster_indices = [i for i, mask in enumerate(cluster_mask) if mask]

        rep_doc, distance, doc_idx = find_cluster_representative(
            cluster_embeddings, cluster_docs, cluster_indices
        )

        print(f"   클러스터 {cluster_id} (크기: {len(cluster_docs)}):")
        print(f"     대표 문서: {rep_doc}")
        print(f"     실제 주제: {true_labels[doc_idx]}")
        print()

    # 5. 클러스터링 결과 시각화
    print("5. 클러스터링 결과 시각화...")

    # 차원 축소 (t-SNE)
    print("   t-SNE 차원 축소 수행 중...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(diverse_documents)-1))
    embeddings_2d = tsne.fit_transform(embeddings_array)

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 실제 주제별 분포
    true_label_colors = {'AI/Tech': 'red', 'Healthcare': 'blue', 'Education': 'green', 'Environment': 'orange'}
    for i, label in enumerate(true_labels):
        axes[0].scatter(embeddings_2d[i, 0], embeddings_2d[i, 1],
                       c=true_label_colors[label], label=label if label not in [true_labels[j] for j in range(i)] else "",
                       alpha=0.7, s=100)

    axes[0].set_title('실제 주제 분포')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # K-means 결과
    scatter = axes[1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                             c=kmeans_labels, cmap='viridis', alpha=0.7, s=100)
    axes[1].set_title(f'K-means 클러스터링 (실루엣: {kmeans_silhouette:.3f})')
    plt.colorbar(scatter, ax=axes[1])
    axes[1].grid(True, alpha=0.3)

    # DBSCAN 결과
    scatter = axes[2].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                             c=dbscan_labels, cmap='viridis', alpha=0.7, s=100)
    axes[2].set_title(f'DBSCAN 클러스터링 (클러스터: {n_clusters_dbscan})')
    plt.colorbar(scatter, ax=axes[2])
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('clustering_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 6. 클러스터링 품질 평가
    print("\n6. 클러스터링 품질 평가...")

    # 실제 라벨과의 일치도 계산 (Adjusted Rand Index)
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    # 실제 라벨을 숫자로 변환
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    true_numeric_labels = le.fit_transform(true_labels)

    # K-means 평가
    kmeans_ari = adjusted_rand_score(true_numeric_labels, kmeans_labels)
    kmeans_nmi = normalized_mutual_info_score(true_numeric_labels, kmeans_labels)

    print(f"K-means 성능:")
    print(f"   - Adjusted Rand Index: {kmeans_ari:.4f}")
    print(f"   - Normalized Mutual Information: {kmeans_nmi:.4f}")
    print(f"   - 실루엣 스코어: {kmeans_silhouette:.4f}")

    # DBSCAN 평가 (노이즈 포인트 고려)
    if n_clusters_dbscan > 0:
        dbscan_ari = adjusted_rand_score(true_numeric_labels, dbscan_labels)
        dbscan_nmi = normalized_mutual_info_score(true_numeric_labels, dbscan_labels)

        print(f"\nDBSCAN 성능:")
        print(f"   - Adjusted Rand Index: {dbscan_ari:.4f}")
        print(f"   - Normalized Mutual Information: {dbscan_nmi:.4f}")
        print(f"   - 실루엣 스코어: {dbscan_silhouette:.4f}")

    # 결과 요약 반환
    return {
        "documents": diverse_documents,
        "true_labels": true_labels,
        "embeddings": embeddings_array,
        "embeddings_2d": embeddings_2d,
        "kmeans_labels": kmeans_labels,
        "dbscan_labels": dbscan_labels,
        "kmeans_metrics": {
            "silhouette": kmeans_silhouette,
            "ari": kmeans_ari,
            "nmi": kmeans_nmi
        },
        "dbscan_metrics": {
            "silhouette": dbscan_silhouette if n_clusters_dbscan > 0 else 0,
            "ari": dbscan_ari if n_clusters_dbscan > 0 else 0,
            "nmi": dbscan_nmi if n_clusters_dbscan > 0 else 0,
            "n_clusters": n_clusters_dbscan,
            "n_noise": n_noise
        }
    }

# 클러스터링 시스템 실행
clustering_results = embedding_based_clustering()

print("\n" + "="*50)
print("클러스터링 시스템 최종 요약")
print("="*50)

kmeans_metrics = clustering_results["kmeans_metrics"]
dbscan_metrics = clustering_results["dbscan_metrics"]

print(f"최적 클러스터링 알고리즘:")
if kmeans_metrics["ari"] > dbscan_metrics["ari"]:
    print(f"  → K-means (ARI: {kmeans_metrics['ari']:.4f})")
else:
    print(f"  → DBSCAN (ARI: {dbscan_metrics['ari']:.4f})")

print(f"\n주요 성능 지표:")
print(f"  K-means ARI: {kmeans_metrics['ari']:.4f}")
print(f"  DBSCAN ARI: {dbscan_metrics['ari']:.4f}")
print(f"  DBSCAN 자동 감지 클러스터 수: {dbscan_metrics['n_clusters']}")
```

## 🔍 참고 자료

### 공식 문서
- [LangChain Embeddings](https://python.langchain.com/docs/modules/data_connection/text_embedding/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Hugging Face sentence-transformers](https://www.sbert.net/)

### 임베딩 모델 비교
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) - 임베딩 모델 성능 순위
- [BGE Models](https://huggingface.co/BAAI) - 고성능 다국어 임베딩
- [E5 Models](https://huggingface.co/intfloat) - Microsoft E5 임베딩 시리즈

### 성능 최적화
```python
# 임베딩 캐싱 예제
from functools import lru_cache

class CachedEmbeddings:
    def __init__(self, base_embeddings):
        self.base_embeddings = base_embeddings

    @lru_cache(maxsize=1000)
    def embed_query(self, text: str) -> List[float]:
        return self.base_embeddings.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

# 배치 처리 최적화
def batch_embed_documents(
    texts: List[str],
    model,
    batch_size: int = 100
) -> List[List[float]]:
    """대용량 문서 배치 처리"""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)

    return all_embeddings
```

### 모델 선택 가이드
| 모델 | 차원 | 언어 | 용도 | 비용 |
|------|------|------|------|------|
| OpenAI text-embedding-3-small | 1536 | 다국어 | 범용, 고품질 | 유료 |
| OpenAI text-embedding-3-large | 3072 | 다국어 | 최고 성능 | 고비용 |
| BAAI/bge-m3 | 1024 | 다국어 | 균형, 무료 | 무료 |
| sentence-transformers/all-MiniLM-L6-v2 | 384 | 다국어 | 경량, 빠름 | 무료 |
| intfloat/multilingual-e5-large | 1024 | 다국어 | 고성능 | 무료 |