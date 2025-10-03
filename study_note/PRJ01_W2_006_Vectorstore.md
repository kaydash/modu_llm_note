# W2_006 벡터 데이터베이스

## 🎯 학습 목표
- 벡터 저장소의 개념과 필요성 이해하기
- Chroma, FAISS, Pinecone 등 다양한 벡터 DB 활용하기
- 효율적인 벡터 검색 시스템 구축하기

## 📚 핵심 개념

### 벡터 저장소(Vector Store)란?
벡터 저장소는 벡터화된 데이터를 효율적으로 저장하고 검색하기 위한 특수 데이터베이스 시스템입니다.

**주요 특징:**
- **고차원 벡터 저장**: 임베딩 벡터를 최적화된 방식으로 저장
- **유사도 검색**: 의미적으로 가까운 데이터를 빠르게 검색
- **메타데이터 관리**: 벡터와 관련된 부가 정보를 함께 저장
- **확장성**: 대용량 데이터 처리 및 실시간 검색 지원

### 벡터 저장소의 필요성

#### 전통적 DB vs 벡터 DB
- **전통적 DB**: 정확한 키워드 매칭, 구조화된 데이터
- **벡터 DB**: 의미적 유사도 기반 검색, 비정형 데이터

#### 주요 활용 사례
- **시맨틱 검색**: 의미 기반 문서 검색
- **추천 시스템**: 유사한 아이템 추천
- **중복 감지**: 유사한 콘텐츠 식별
- **RAG 시스템**: 질의응답을 위한 관련 문서 검색

### LangChain 지원 벡터 저장소

#### 1. Chroma
- **특징**: 경량화, 로컬 개발 친화적
- **장점**: 간편한 설치, 빠른 시작
- **용도**: 프로토타이핑, 소규모 프로젝트

#### 2. FAISS
- **특징**: Facebook AI의 고성능 검색 라이브러리
- **장점**: 빠른 검색 속도, 다양한 인덱스 알고리즘
- **용도**: 중대규모 데이터, 성능 중시

#### 3. Pinecone
- **특징**: 완전 관리형 클라우드 서비스
- **장점**: 확장성, 고가용성, 관리 부담 없음
- **용도**: 프로덕션 환경, 대규모 서비스

## 🔧 환경 설정

### 필수 라이브러리 설치
```bash
# Chroma 벡터 저장소
pip install langchain-chroma

# FAISS 벡터 저장소
pip install faiss-cpu  # CPU 버전
# pip install faiss-gpu  # GPU 버전

# Pinecone 벡터 저장소
pip install langchain-pinecone pinecone-client

# 임베딩 모델
pip install langchain-huggingface
```

### 환경 변수 설정
```python
from dotenv import load_dotenv
import os
import uuid
from typing import List, Dict, Any, Optional, Tuple
from pprint import pprint

load_dotenv()

# Pinecone API 키 (필요한 경우)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# 기본 라이브러리
from langchain_core.documents import Document
```

## 💻 코드 예제

### 1. Chroma 벡터 저장소

#### 기본 설정 및 초기화
```python
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

def create_chroma_vectorstore(
    collection_name: str = "default_collection",
    persist_directory: str = "./chroma_db"
) -> Chroma:
    """Chroma 벡터 저장소 생성"""
    # 임베딩 모델 설정
    embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    # Chroma 벡터 저장소 생성
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings_model,
        persist_directory=persist_directory,
    )

    return vectorstore

# 벡터 저장소 생성
chroma_db = create_chroma_vectorstore("ai_sample_collection")

# 현재 저장된 데이터 확인
stored_data = chroma_db.get()
print(f"저장된 문서 수: {len(stored_data['documents'])}")
```

#### 문서 관리 (CRUD Operations)
```python
def add_documents_to_chroma(
    vectorstore: Chroma,
    contents: List[str],
    metadatas: List[Dict[str, Any]],
    ids: Optional[List[str]] = None
) -> List[str]:
    """문서를 Chroma에 추가"""
    # Document 객체 생성
    documents = [
        Document(page_content=content, metadata=metadata)
        for content, metadata in zip(contents, metadatas)
    ]

    # ID 자동 생성 (제공되지 않은 경우)
    if ids is None:
        ids = [f"DOC_{i+1}" for i in range(len(documents))]

    # 문서 추가
    added_ids = vectorstore.add_documents(documents=documents, ids=ids)
    print(f"{len(added_ids)}개 문서 추가 완료")

    return added_ids

# 샘플 문서 데이터
sample_contents = [
    "인공지능은 컴퓨터 과학의 한 분야입니다.",
    "머신러닝은 인공지능의 하위 분야입니다.",
    "딥러닝은 머신러닝의 한 종류입니다.",
    "자연어 처리는 컴퓨터가 인간의 언어를 이해하고 생성하는 기술입니다.",
    "컴퓨터 비전은 컴퓨터가 디지털 이미지나 비디오를 이해하는 방법을 연구합니다."
]

sample_metadatas = [
    {"source": "AI 개론", "topic": "정의"},
    {"source": "AI 개론", "topic": "분야"},
    {"source": "딥러닝 입문", "topic": "분야"},
    {"source": "AI 개론", "topic": "기술"},
    {"source": "딥러닝 입문", "topic": "기술"}
]

# 문서 추가
doc_ids = add_documents_to_chroma(chroma_db, sample_contents, sample_metadatas)
```

#### 문서 검색 기능
```python
def search_chroma_documents(
    vectorstore: Chroma,
    query: str,
    k: int = 3,
    filter_metadata: Optional[Dict[str, Any]] = None,
    search_type: str = "similarity"
) -> List[Any]:
    """Chroma에서 문서 검색"""

    if search_type == "similarity":
        # 기본 유사도 검색
        results = vectorstore.similarity_search(
            query=query,
            k=k,
            filter=filter_metadata
        )

    elif search_type == "similarity_with_score":
        # 유사도 점수 포함 검색
        results = vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter_metadata
        )

    elif search_type == "relevance_score":
        # 관련성 점수 포함 검색
        results = vectorstore.similarity_search_with_relevance_scores(
            query=query,
            k=k,
            filter=filter_metadata
        )

    return results

# 다양한 검색 방법 테스트
query = "인공지능과 머신러닝의 관계는?"

# 1. 기본 유사도 검색
print("=== 기본 유사도 검색 ===")
basic_results = search_chroma_documents(chroma_db, query, k=2)
for doc in basic_results:
    print(f"- {doc.page_content} [출처: {doc.metadata['source']}]")

# 2. 점수 포함 검색
print("\n=== 점수 포함 검색 ===")
score_results = search_chroma_documents(chroma_db, query, k=2, search_type="similarity_with_score")
for doc, score in score_results:
    print(f"- 점수: {score:.4f}")
    print(f"  내용: {doc.page_content}")
    print(f"  메타데이터: {doc.metadata}")

# 3. 필터링 검색
print("\n=== 필터링 검색 (AI 개론 출처만) ===")
filtered_results = search_chroma_documents(
    chroma_db, query, k=3,
    filter_metadata={"source": "AI 개론"}
)
for doc in filtered_results:
    print(f"- {doc.page_content}")
```

#### 문서 수정 및 삭제
```python
def update_chroma_document(
    vectorstore: Chroma,
    document_id: str,
    new_content: str,
    new_metadata: Dict[str, Any]
) -> None:
    """Chroma 문서 업데이트"""
    updated_document = Document(
        page_content=new_content,
        metadata=new_metadata
    )

    vectorstore.update_document(document_id=document_id, document=updated_document)
    print(f"문서 {document_id} 업데이트 완료")

def delete_chroma_documents(vectorstore: Chroma, ids: List[str]) -> None:
    """Chroma 문서 삭제"""
    vectorstore.delete(ids=ids)
    print(f"문서 삭제 완료: {ids}")

# 문서 업데이트 예시
update_chroma_document(
    chroma_db,
    "DOC_1",
    "인공지능은 컴퓨터 과학의 핵심 분야 중 하나로, 기계학습과 딥러닝을 포함합니다.",
    {"source": "AI 개론", "topic": "정의", "updated": True}
)

# 문서 삭제 예시
delete_chroma_documents(chroma_db, ["DOC_5"])

# 변경 사항 확인
updated_results = search_chroma_documents(chroma_db, "인공지능 정의", k=1)
print(f"업데이트된 문서: {updated_results[0].page_content}")
```

### 2. FAISS 벡터 저장소

#### 기본 설정 및 초기화
```python
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

def create_faiss_vectorstore(
    embedding_model,
    dimension: Optional[int] = None
) -> FAISS:
    """FAISS 벡터 저장소 생성"""

    # 임베딩 차원 계산 (제공되지 않은 경우)
    if dimension is None:
        sample_embedding = embedding_model.embed_query("test")
        dimension = len(sample_embedding)

    # FAISS 인덱스 초기화 (L2 거리 사용)
    faiss_index = faiss.IndexFlatL2(dimension)

    # FAISS 벡터 저장소 생성
    vectorstore = FAISS(
        embedding_function=embedding_model,
        index=faiss_index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    print(f"FAISS 인덱스 생성 완료 (차원: {dimension})")
    return vectorstore

# FAISS 벡터 저장소 생성
embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
faiss_db = create_faiss_vectorstore(embeddings_model)

print(f"저장된 벡터 수: {faiss_db.index.ntotal}")
```

#### FAISS 문서 관리
```python
def add_documents_to_faiss(
    vectorstore: FAISS,
    documents: List[Document],
    ids: Optional[List[str]] = None
) -> List[str]:
    """FAISS에 문서 추가"""
    if ids is None:
        ids = [str(uuid.uuid4()) for _ in range(len(documents))]

    added_ids = vectorstore.add_documents(documents=documents, ids=ids)
    print(f"FAISS에 {len(added_ids)}개 문서 추가")

    return added_ids

def search_faiss_documents(
    vectorstore: FAISS,
    query: str,
    k: int = 3,
    filter_metadata: Optional[Dict[str, Any]] = None
) -> List[Document]:
    """FAISS에서 문서 검색"""
    results = vectorstore.similarity_search(
        query=query,
        k=k,
        filter=filter_metadata
    )

    return results

# Document 객체 생성
documents = [
    Document(page_content=content, metadata=metadata)
    for content, metadata in zip(sample_contents, sample_metadatas)
]

# FAISS에 문서 추가
faiss_ids = add_documents_to_faiss(faiss_db, documents)

# 검색 테스트
faiss_results = search_faiss_documents(faiss_db, "머신러닝과 딥러닝", k=2)
print("\n=== FAISS 검색 결과 ===")
for doc in faiss_results:
    print(f"- {doc.page_content}")
    print(f"  메타데이터: {doc.metadata}")

# FAISS 저장 및 로드
faiss_db.save_local("./faiss_index")
print("FAISS 인덱스 로컬 저장 완료")

# 저장된 FAISS 로드
loaded_faiss_db = FAISS.load_local(
    "./faiss_index",
    embeddings_model,
    allow_dangerous_deserialization=True
)
print(f"로드된 FAISS 벡터 수: {loaded_faiss_db.index.ntotal}")
```

### 3. Pinecone 벡터 저장소

#### Pinecone 설정 및 초기화
```python
def setup_pinecone_vectorstore(
    api_key: str,
    index_name: str,
    dimension: int = 1024,
    metric: str = "euclidean"
) -> Any:
    """Pinecone 벡터 저장소 설정"""

    try:
        from pinecone import Pinecone, ServerlessSpec
        from langchain_pinecone import PineconeVectorStore

        # Pinecone 클라이언트 초기화
        pc = Pinecone(api_key=api_key)

        # 기존 인덱스 확인
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

        # 인덱스 생성 (없는 경우)
        if index_name not in existing_indexes:
            print(f"새 Pinecone 인덱스 생성: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

            # 인덱스 준비 대기
            import time
            while not pc.describe_index(index_name).status["ready"]:
                time.sleep(1)

        # 인덱스 연결
        index = pc.Index(index_name)

        # PineconeVectorStore 생성
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embeddings_model
        )

        print(f"Pinecone 벡터 저장소 준비 완료")
        return vectorstore, pc

    except ImportError:
        print("Pinecone 라이브러리가 설치되지 않았습니다.")
        return None, None
    except Exception as e:
        print(f"Pinecone 설정 실패: {e}")
        return None, None

# Pinecone 벡터 저장소 설정 (API 키가 있는 경우)
if PINECONE_API_KEY:
    pinecone_db, pinecone_client = setup_pinecone_vectorstore(
        api_key=PINECONE_API_KEY,
        index_name="ai-sample-index",
        dimension=1024
    )

    if pinecone_db:
        # 문서 추가
        pinecone_ids = pinecone_db.add_documents(documents=documents)
        print(f"Pinecone에 {len(pinecone_ids)}개 문서 추가")

        # 검색 테스트
        pinecone_results = pinecone_db.similarity_search("자연어 처리", k=2)
        print("\n=== Pinecone 검색 결과 ===")
        for doc in pinecone_results:
            print(f"- {doc.page_content}")
else:
    print("PINECONE_API_KEY가 설정되지 않아 Pinecone 예제를 건너뜁니다.")
```

### 4. 벡터 저장소 성능 비교

#### 벤치마크 시스템
```python
import time
from typing import Callable

def benchmark_vectorstore_performance(
    vectorstores: Dict[str, Any],
    test_documents: List[Document],
    test_queries: List[str],
    k: int = 5
) -> Dict[str, Dict[str, float]]:
    """벡터 저장소 성능 벤치마크"""

    results = {}

    for name, vectorstore in vectorstores.items():
        print(f"\n=== {name} 벤치마크 ===")

        try:
            # 1. 문서 추가 성능
            start_time = time.time()
            test_ids = [f"TEST_{i}" for i in range(len(test_documents))]
            vectorstore.add_documents(documents=test_documents, ids=test_ids)
            add_time = time.time() - start_time

            # 2. 검색 성능
            search_times = []
            for query in test_queries:
                start_time = time.time()
                _ = vectorstore.similarity_search(query, k=k)
                search_time = time.time() - start_time
                search_times.append(search_time)

            avg_search_time = sum(search_times) / len(search_times)

            results[name] = {
                "add_time": add_time,
                "avg_search_time": avg_search_time,
                "total_search_time": sum(search_times),
                "docs_per_second": len(test_documents) / add_time,
                "searches_per_second": 1 / avg_search_time
            }

            print(f"문서 추가 시간: {add_time:.3f}초")
            print(f"평균 검색 시간: {avg_search_time:.3f}초")
            print(f"처리량: {results[name]['docs_per_second']:.1f} docs/sec")

            # 테스트 문서 정리
            vectorstore.delete(ids=test_ids)

        except Exception as e:
            print(f"벤치마크 실패: {e}")
            results[name] = {"error": str(e)}

    return results

# 테스트 데이터 준비
test_docs = [
    Document(page_content=f"테스트 문서 {i}: AI 관련 내용", metadata={"test_id": i})
    for i in range(50)  # 50개 테스트 문서
]

test_queries = [
    "인공지능 기술",
    "머신러닝 알고리즘",
    "딥러닝 모델",
    "자연어 처리"
]

# 벡터 저장소별 성능 비교
vectorstores_to_test = {
    "Chroma": chroma_db,
    "FAISS": faiss_db,
}

if 'pinecone_db' in locals() and pinecone_db:
    vectorstores_to_test["Pinecone"] = pinecone_db

benchmark_results = benchmark_vectorstore_performance(
    vectorstores_to_test,
    test_docs,
    test_queries
)

# 결과 요약
print("\n" + "="*60)
print("벡터 저장소 성능 비교 요약")
print("="*60)

for name, metrics in benchmark_results.items():
    if "error" not in metrics:
        print(f"\n{name}:")
        print(f"  문서 추가 속도: {metrics['docs_per_second']:.1f} docs/sec")
        print(f"  검색 속도: {metrics['searches_per_second']:.1f} searches/sec")
        print(f"  평균 검색 시간: {metrics['avg_search_time']:.3f}초")
```

### 5. 고급 벡터 검색 기법

#### 하이브리드 검색 (키워드 + 벡터)
```python
def hybrid_search(
    vectorstore: Any,
    query: str,
    k_vector: int = 10,
    k_final: int = 5,
    alpha: float = 0.7
) -> List[Document]:
    """하이브리드 검색: 벡터 검색 + 키워드 검색"""

    # 1. 벡터 유사도 검색
    vector_results = vectorstore.similarity_search_with_score(query, k=k_vector)

    # 2. 간단한 키워드 기반 점수 계산
    query_words = set(query.lower().split())

    hybrid_scores = []
    for doc, vector_score in vector_results:
        # 키워드 매칭 점수
        doc_words = set(doc.page_content.lower().split())
        keyword_score = len(query_words & doc_words) / len(query_words)

        # 하이브리드 점수 계산 (벡터 점수는 거리이므로 역수 사용)
        vector_similarity = 1 / (1 + vector_score)  # 거리 → 유사도 변환
        hybrid_score = alpha * vector_similarity + (1 - alpha) * keyword_score

        hybrid_scores.append((doc, hybrid_score))

    # 점수순 정렬 및 상위 k_final개 반환
    hybrid_scores.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in hybrid_scores[:k_final]]

# 하이브리드 검색 테스트
hybrid_results = hybrid_search(chroma_db, "머신러닝 알고리즘", k_final=3)
print("=== 하이브리드 검색 결과 ===")
for doc in hybrid_results:
    print(f"- {doc.page_content}")
```

#### 메타데이터 기반 필터링
```python
def advanced_metadata_filtering(
    vectorstore: Any,
    query: str,
    filters: Dict[str, Any],
    k: int = 5
) -> List[Document]:
    """고급 메타데이터 필터링"""

    # 복합 필터 조건 구성
    filter_conditions = {}

    for key, value in filters.items():
        if isinstance(value, list):
            # 리스트인 경우 OR 조건으로 처리 (Chroma의 $in 연산자)
            filter_conditions[key] = {"$in": value}
        elif isinstance(value, dict):
            # 범위 조건 등 복잡한 필터
            filter_conditions[key] = value
        else:
            # 단순 일치
            filter_conditions[key] = value

    # 필터링된 검색 수행
    results = vectorstore.similarity_search(
        query=query,
        k=k,
        filter=filter_conditions
    )

    return results

# 고급 필터링 예시
# 먼저 다양한 메타데이터를 가진 문서들을 추가
advanced_documents = [
    Document(
        page_content="Python 기반 웹 개발 가이드",
        metadata={"category": "programming", "language": "python", "difficulty": "beginner"}
    ),
    Document(
        page_content="JavaScript 고급 기법들",
        metadata={"category": "programming", "language": "javascript", "difficulty": "advanced"}
    ),
    Document(
        page_content="데이터베이스 설계 패턴",
        metadata={"category": "database", "language": "sql", "difficulty": "intermediate"}
    ),
    Document(
        page_content="기계학습 알고리즘 개념",
        metadata={"category": "ai", "language": "python", "difficulty": "advanced"}
    )
]

# 문서 추가
adv_ids = chroma_db.add_documents(advanced_documents)

# 복합 필터링 검색
filtered_results = advanced_metadata_filtering(
    chroma_db,
    "프로그래밍 개발",
    {
        "category": ["programming", "ai"],  # programming 또는 ai
        "language": "python"  # python 언어
    },
    k=3
)

print("=== 고급 필터링 검색 결과 ===")
for doc in filtered_results:
    print(f"- {doc.page_content}")
    print(f"  메타데이터: {doc.metadata}")
```

## 🚀 실습해보기

### 실습 1: 벡터 저장소 관리 시스템
완전한 CRUD 기능을 가진 벡터 저장소 관리 시스템을 구현해보세요.

```python
class VectorStoreManager:
    """벡터 저장소 통합 관리 클래스"""

    def __init__(self, vectorstore_type: str = "chroma"):
        # TODO: 벡터 저장소 타입에 따른 초기화
        # TODO: 임베딩 모델 설정
        # TODO: 기본 컬렉션 생성
        pass

    def add_document(self, content: str, metadata: Dict[str, Any]) -> str:
        # TODO: 단일 문서 추가
        # TODO: 자동 ID 생성
        # TODO: 중복 체크
        pass

    def batch_add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        # TODO: 대량 문서 일괄 추가
        # TODO: 트랜잭션 처리
        pass

    def search_documents(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        # TODO: 통합 검색 인터페이스
        # TODO: 다양한 검색 옵션 지원
        pass

    def update_document(self, doc_id: str, content: str, metadata: Dict) -> bool:
        # TODO: 문서 업데이트
        pass

    def delete_documents(self, doc_ids: List[str]) -> int:
        # TODO: 문서 삭제
        # TODO: 삭제된 문서 수 반환
        pass

# 테스트
manager = VectorStoreManager("chroma")
```

### 실습 2: 성능 최적화된 검색 시스템
다양한 검색 전략을 조합한 고성능 검색 시스템을 구현해보세요.

```python
class OptimizedSearchSystem:
    """성능 최적화된 벡터 검색 시스템"""

    def __init__(self):
        # TODO: 여러 벡터 저장소 초기화
        # TODO: 캐싱 시스템 구현
        # TODO: 검색 전략 설정
        pass

    def semantic_search(self, query: str, k: int = 5) -> List[Document]:
        # TODO: 순수 의미 기반 검색
        pass

    def keyword_search(self, query: str, k: int = 5) -> List[Document]:
        # TODO: 키워드 기반 검색
        pass

    def hybrid_search(
        self,
        query: str,
        semantic_weight: float = 0.7,
        k: int = 5
    ) -> List[Document]:
        # TODO: 하이브리드 검색 구현
        # TODO: 점수 정규화 및 결합
        pass

    def cached_search(self, query: str) -> List[Document]:
        # TODO: 캐시된 결과 활용
        pass

# 테스트
search_system = OptimizedSearchSystem()
```

### 실습 3: 벡터 저장소 마이그레이션 도구
서로 다른 벡터 저장소 간 데이터 마이그레이션 도구를 구현해보세요.

```python
class VectorStoreMigrator:
    """벡터 저장소 간 데이터 마이그레이션"""

    def migrate_data(
        self,
        source_store: Any,
        target_store: Any,
        batch_size: int = 100
    ) -> Dict[str, int]:
        # TODO: 소스에서 데이터 추출
        # TODO: 배치 단위로 타겟에 삽입
        # TODO: 진행상황 표시
        # TODO: 마이그레이션 통계 반환
        pass

    def verify_migration(
        self,
        source_store: Any,
        target_store: Any
    ) -> Dict[str, Any]:
        # TODO: 데이터 무결성 검증
        # TODO: 벡터 유사도 검증
        # TODO: 메타데이터 일관성 확인
        pass

# 테스트
migrator = VectorStoreMigrator()
```

## 📋 해답

### 실습 1 해답: 벡터 저장소 관리 시스템
```python
import uuid
from typing import Dict, List, Any, Optional
import time
from dataclasses import dataclass

@dataclass
class SearchResult:
    """검색 결과 데이터 클래스"""
    content: str
    metadata: Dict[str, Any]
    score: float
    doc_id: str

class VectorStoreManager:
    """벡터 저장소 통합 관리 클래스"""

    def __init__(self, vectorstore_type: str = "chroma", **kwargs):
        self.vectorstore_type = vectorstore_type
        self.embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

        # 벡터 저장소 초기화
        if vectorstore_type.lower() == "chroma":
            self._init_chroma(**kwargs)
        elif vectorstore_type.lower() == "faiss":
            self._init_faiss(**kwargs)
        else:
            raise ValueError(f"지원하지 않는 벡터 저장소: {vectorstore_type}")

        # 통계 정보
        self.stats = {
            "total_documents": 0,
            "total_searches": 0,
            "last_updated": None
        }

    def _init_chroma(self, **kwargs):
        """Chroma 벡터 저장소 초기화"""
        from langchain_chroma import Chroma

        collection_name = kwargs.get("collection_name", "managed_collection")
        persist_directory = kwargs.get("persist_directory", "./managed_chroma")

        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings_model,
            persist_directory=persist_directory
        )

        # 기존 문서 수 확인
        existing_data = self.vectorstore.get()
        self.stats["total_documents"] = len(existing_data["documents"])

    def _init_faiss(self, **kwargs):
        """FAISS 벡터 저장소 초기화"""
        import faiss
        from langchain_community.docstore.in_memory import InMemoryDocstore
        from langchain_community.vectorstores import FAISS

        dimension = kwargs.get("dimension", 1024)
        faiss_index = faiss.IndexFlatL2(dimension)

        self.vectorstore = FAISS(
            embedding_function=self.embeddings_model,
            index=faiss_index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )

    def add_document(
        self,
        content: str,
        metadata: Dict[str, Any],
        doc_id: Optional[str] = None
    ) -> str:
        """단일 문서 추가"""

        if doc_id is None:
            doc_id = str(uuid.uuid4())

        # 중복 체크 (기존 ID가 있는지 확인)
        if self._document_exists(doc_id):
            raise ValueError(f"문서 ID {doc_id}가 이미 존재합니다.")

        # 메타데이터에 추가 정보 포함
        enhanced_metadata = {
            **metadata,
            "created_at": time.time(),
            "doc_id": doc_id
        }

        document = Document(page_content=content, metadata=enhanced_metadata)

        try:
            self.vectorstore.add_documents(documents=[document], ids=[doc_id])
            self.stats["total_documents"] += 1
            self.stats["last_updated"] = time.time()

            print(f"문서 추가 완료: {doc_id}")
            return doc_id

        except Exception as e:
            print(f"문서 추가 실패: {e}")
            raise

    def batch_add_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 50
    ) -> List[str]:
        """대량 문서 일괄 추가"""

        added_ids = []
        total_batches = (len(documents) + batch_size - 1) // batch_size

        for batch_idx in range(0, len(documents), batch_size):
            batch = documents[batch_idx:batch_idx + batch_size]
            batch_docs = []
            batch_ids = []

            for doc_data in batch:
                doc_id = doc_data.get("id", str(uuid.uuid4()))
                content = doc_data["content"]
                metadata = doc_data.get("metadata", {})

                # 중복 체크 건너뛰기 (성능상 이유)
                enhanced_metadata = {
                    **metadata,
                    "created_at": time.time(),
                    "doc_id": doc_id,
                    "batch_id": batch_idx // batch_size
                }

                document = Document(page_content=content, metadata=enhanced_metadata)
                batch_docs.append(document)
                batch_ids.append(doc_id)

            try:
                # 배치 단위로 추가
                self.vectorstore.add_documents(documents=batch_docs, ids=batch_ids)
                added_ids.extend(batch_ids)

                print(f"배치 {(batch_idx//batch_size)+1}/{total_batches} 완료 ({len(batch_ids)}개 문서)")

            except Exception as e:
                print(f"배치 {(batch_idx//batch_size)+1} 처리 실패: {e}")
                continue

        self.stats["total_documents"] += len(added_ids)
        self.stats["last_updated"] = time.time()

        print(f"총 {len(added_ids)}개 문서 일괄 추가 완료")
        return added_ids

    def search_documents(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict] = None,
        search_type: str = "similarity",
        min_score: Optional[float] = None
    ) -> List[SearchResult]:
        """통합 검색 인터페이스"""

        self.stats["total_searches"] += 1

        try:
            if search_type == "similarity":
                raw_results = self.vectorstore.similarity_search(
                    query=query, k=k, filter=filters
                )
                results = [
                    SearchResult(
                        content=doc.page_content,
                        metadata=doc.metadata,
                        score=1.0,  # 기본 점수
                        doc_id=doc.metadata.get("doc_id", "unknown")
                    )
                    for doc in raw_results
                ]

            elif search_type == "similarity_with_score":
                raw_results = self.vectorstore.similarity_search_with_score(
                    query=query, k=k, filter=filters
                )
                results = [
                    SearchResult(
                        content=doc.page_content,
                        metadata=doc.metadata,
                        score=float(score),
                        doc_id=doc.metadata.get("doc_id", "unknown")
                    )
                    for doc, score in raw_results
                ]

            elif search_type == "relevance_score":
                raw_results = self.vectorstore.similarity_search_with_relevance_scores(
                    query=query, k=k, filter=filters
                )
                results = [
                    SearchResult(
                        content=doc.page_content,
                        metadata=doc.metadata,
                        score=float(score),
                        doc_id=doc.metadata.get("doc_id", "unknown")
                    )
                    for doc, score in raw_results
                ]

            # 최소 점수 필터링
            if min_score is not None:
                if search_type == "similarity_with_score":
                    # 거리 점수인 경우 (낮을수록 좋음)
                    results = [r for r in results if r.score <= min_score]
                else:
                    # 유사도 점수인 경우 (높을수록 좋음)
                    results = [r for r in results if r.score >= min_score]

            print(f"검색 완료: {len(results)}개 결과 (쿼리: '{query}')")
            return results

        except Exception as e:
            print(f"검색 실패: {e}")
            return []

    def update_document(
        self,
        doc_id: str,
        content: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """문서 업데이트"""

        try:
            # 메타데이터에 업데이트 정보 추가
            enhanced_metadata = {
                **metadata,
                "updated_at": time.time(),
                "doc_id": doc_id
            }

            updated_document = Document(
                page_content=content,
                metadata=enhanced_metadata
            )

            if self.vectorstore_type.lower() == "chroma":
                self.vectorstore.update_document(
                    document_id=doc_id,
                    document=updated_document
                )
            else:
                # FAISS는 직접적인 업데이트 지원하지 않으므로 삭제 후 추가
                self.vectorstore.delete([doc_id])
                self.vectorstore.add_documents(documents=[updated_document], ids=[doc_id])

            self.stats["last_updated"] = time.time()
            print(f"문서 업데이트 완료: {doc_id}")
            return True

        except Exception as e:
            print(f"문서 업데이트 실패: {e}")
            return False

    def delete_documents(self, doc_ids: List[str]) -> int:
        """문서 삭제"""

        try:
            # 존재하는 문서만 삭제 시도
            existing_ids = [doc_id for doc_id in doc_ids if self._document_exists(doc_id)]

            if not existing_ids:
                print("삭제할 문서가 없습니다.")
                return 0

            self.vectorstore.delete(ids=existing_ids)

            deleted_count = len(existing_ids)
            self.stats["total_documents"] -= deleted_count
            self.stats["last_updated"] = time.time()

            print(f"{deleted_count}개 문서 삭제 완료")
            return deleted_count

        except Exception as e:
            print(f"문서 삭제 실패: {e}")
            return 0

    def get_statistics(self) -> Dict[str, Any]:
        """관리 통계 정보 반환"""
        stats = self.stats.copy()

        if stats["last_updated"]:
            stats["last_updated_human"] = time.ctime(stats["last_updated"])

        return stats

    def _document_exists(self, doc_id: str) -> bool:
        """문서 존재 여부 확인"""
        try:
            if self.vectorstore_type.lower() == "chroma":
                result = self.vectorstore.get(ids=[doc_id])
                return len(result["ids"]) > 0
            elif self.vectorstore_type.lower() == "faiss":
                return doc_id in self.vectorstore.index_to_docstore_id.values()
            return False
        except:
            return False

# 벡터 저장소 관리 시스템 테스트
print("=== 벡터 저장소 관리 시스템 테스트 ===")

# 관리자 초기화
manager = VectorStoreManager("chroma", collection_name="test_managed_collection")

# 단일 문서 추가
doc_id = manager.add_document(
    content="벡터 데이터베이스는 AI 애플리케이션의 핵심 구성 요소입니다.",
    metadata={"category": "database", "importance": "high"}
)

# 대량 문서 추가
batch_documents = [
    {
        "content": f"테스트 문서 {i}: AI와 머신러닝 관련 내용",
        "metadata": {"category": "ai", "test_batch": True, "number": i}
    }
    for i in range(10)
]

batch_ids = manager.batch_add_documents(batch_documents)

# 검색 테스트
search_results = manager.search_documents(
    "AI 머신러닝",
    k=5,
    search_type="similarity_with_score"
)

print("\n검색 결과:")
for result in search_results:
    print(f"- 점수: {result.score:.4f}")
    print(f"  내용: {result.content}")
    print(f"  ID: {result.doc_id}")

# 통계 정보 확인
stats = manager.get_statistics()
print(f"\n통계 정보:")
for key, value in stats.items():
    print(f"  {key}: {value}")

# 문서 업데이트
manager.update_document(
    doc_id,
    "벡터 데이터베이스는 현대 AI 애플리케이션에서 필수적인 구성 요소입니다.",
    {"category": "database", "importance": "critical", "version": 2}
)

# 문서 삭제
deleted_count = manager.delete_documents(batch_ids[:5])  # 처음 5개 삭제
```

### 실습 2 해답: 성능 최적화된 검색 시스템
```python
from functools import lru_cache
import re
from collections import Counter
import hashlib

class OptimizedSearchSystem:
    """성능 최적화된 벡터 검색 시스템"""

    def __init__(self, cache_size: int = 1000):
        # 여러 벡터 저장소 초기화
        self.embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
        self.chroma_db = self._create_chroma_store()
        self.cache_size = cache_size

        # 검색 통계
        self.search_stats = {
            "semantic_searches": 0,
            "keyword_searches": 0,
            "hybrid_searches": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }

        # 간단한 메모리 캐시 (실제 환경에서는 Redis 등 사용)
        self._search_cache = {}

    def _create_chroma_store(self):
        """Chroma 벡터 저장소 생성"""
        from langchain_chroma import Chroma
        return Chroma(
            collection_name="optimized_search",
            embedding_function=self.embeddings_model,
            persist_directory="./optimized_chroma"
        )

    def _get_cache_key(self, query: str, search_type: str, k: int) -> str:
        """캐시 키 생성"""
        cache_string = f"{query}:{search_type}:{k}"
        return hashlib.md5(cache_string.encode()).hexdigest()

    def _preprocess_query(self, query: str) -> str:
        """쿼리 전처리"""
        # 소문자 변환, 특수문자 제거
        processed = re.sub(r'[^\w\s]', ' ', query.lower())
        # 중복 공백 제거
        processed = re.sub(r'\s+', ' ', processed).strip()
        return processed

    def semantic_search(self, query: str, k: int = 5) -> List[Document]:
        """순수 의미 기반 검색"""
        self.search_stats["semantic_searches"] += 1

        # 캐시 확인
        cache_key = self._get_cache_key(query, "semantic", k)
        if cache_key in self._search_cache:
            self.search_stats["cache_hits"] += 1
            return self._search_cache[cache_key]

        self.search_stats["cache_misses"] += 1

        # 의미 기반 검색 수행
        results = self.chroma_db.similarity_search(query, k=k)

        # 캐시에 저장 (크기 제한)
        if len(self._search_cache) < self.cache_size:
            self._search_cache[cache_key] = results

        return results

    def keyword_search(self, query: str, k: int = 5) -> List[Document]:
        """키워드 기반 검색"""
        self.search_stats["keyword_searches"] += 1

        # 전체 문서 가져오기 (실제로는 역색인 사용 권장)
        all_docs_data = self.chroma_db.get()
        all_documents = [
            Document(page_content=content, metadata=metadata)
            for content, metadata in zip(all_docs_data["documents"], all_docs_data["metadatas"])
        ]

        # 쿼리 키워드 추출
        query_words = set(self._preprocess_query(query).split())

        # 각 문서의 키워드 매칭 점수 계산
        scored_docs = []
        for doc in all_documents:
            doc_words = set(self._preprocess_query(doc.page_content).split())

            # 교집합 기반 점수 (Jaccard 유사도)
            intersection = len(query_words & doc_words)
            union = len(query_words | doc_words)

            if union > 0:
                score = intersection / union

                # TF 점수 추가 고려
                word_counts = Counter(self._preprocess_query(doc.page_content).split())
                tf_score = sum(word_counts[word] for word in query_words if word in word_counts)

                combined_score = score + (tf_score * 0.1)  # TF 가중치
                scored_docs.append((doc, combined_score))

        # 점수순 정렬 후 상위 k개 반환
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:k]]

    def hybrid_search(
        self,
        query: str,
        semantic_weight: float = 0.7,
        k: int = 5,
        intermediate_k: int = 20
    ) -> List[Document]:
        """하이브리드 검색 구현"""
        self.search_stats["hybrid_searches"] += 1

        # 캐시 확인
        cache_key = self._get_cache_key(f"{query}:hybrid:{semantic_weight}", "hybrid", k)
        if cache_key in self._search_cache:
            self.search_stats["cache_hits"] += 1
            return self._search_cache[cache_key]

        self.search_stats["cache_misses"] += 1

        # 1. 의미적 검색 (점수 포함)
        semantic_results = self.chroma_db.similarity_search_with_score(
            query, k=intermediate_k
        )

        # 2. 키워드 검색 결과
        keyword_results = self.keyword_search(query, k=intermediate_k)

        # 3. 문서별 점수 통합
        doc_scores = {}
        query_words = set(self._preprocess_query(query).split())

        # 의미적 점수 정규화 및 저장
        for doc, distance_score in semantic_results:
            doc_content = doc.page_content
            # 거리를 유사도로 변환 (낮은 거리 = 높은 유사도)
            semantic_sim = 1 / (1 + distance_score)
            doc_scores[doc_content] = {
                "doc": doc,
                "semantic_score": semantic_sim,
                "keyword_score": 0.0
            }

        # 키워드 점수 계산 및 추가
        for doc in keyword_results:
            doc_content = doc.page_content
            doc_words = set(self._preprocess_query(doc_content).split())

            # 키워드 매칭 점수
            if len(query_words) > 0:
                keyword_score = len(query_words & doc_words) / len(query_words)
            else:
                keyword_score = 0.0

            if doc_content in doc_scores:
                doc_scores[doc_content]["keyword_score"] = keyword_score
            else:
                doc_scores[doc_content] = {
                    "doc": doc,
                    "semantic_score": 0.0,
                    "keyword_score": keyword_score
                }

        # 4. 하이브리드 점수 계산 및 정렬
        hybrid_results = []
        for doc_content, scores in doc_scores.items():
            hybrid_score = (
                semantic_weight * scores["semantic_score"] +
                (1 - semantic_weight) * scores["keyword_score"]
            )
            hybrid_results.append((scores["doc"], hybrid_score))

        # 점수순 정렬 후 상위 k개 반환
        hybrid_results.sort(key=lambda x: x[1], reverse=True)
        final_results = [doc for doc, _ in hybrid_results[:k]]

        # 캐시에 저장
        if len(self._search_cache) < self.cache_size:
            self._search_cache[cache_key] = final_results

        return final_results

    def cached_search(self, query: str, search_type: str = "semantic", k: int = 5) -> List[Document]:
        """캐시 우선 검색"""
        cache_key = self._get_cache_key(query, search_type, k)

        if cache_key in self._search_cache:
            self.search_stats["cache_hits"] += 1
            return self._search_cache[cache_key]

        # 캐시 미스인 경우 해당 검색 방법 실행
        if search_type == "semantic":
            return self.semantic_search(query, k)
        elif search_type == "keyword":
            return self.keyword_search(query, k)
        elif search_type == "hybrid":
            return self.hybrid_search(query, k=k)
        else:
            raise ValueError(f"지원하지 않는 검색 타입: {search_type}")

    def clear_cache(self) -> None:
        """캐시 비우기"""
        self._search_cache.clear()
        print("검색 캐시를 비웠습니다.")

    def get_search_statistics(self) -> Dict[str, Any]:
        """검색 통계 반환"""
        stats = self.search_stats.copy()

        # 캐시 적중률 계산
        total_cache_requests = stats["cache_hits"] + stats["cache_misses"]
        if total_cache_requests > 0:
            stats["cache_hit_rate"] = stats["cache_hits"] / total_cache_requests
        else:
            stats["cache_hit_rate"] = 0.0

        stats["cache_size"] = len(self._search_cache)
        return stats

# 성능 최적화 검색 시스템 테스트
print("=== 성능 최적화된 검색 시스템 테스트 ===")

# 검색 시스템 초기화
search_system = OptimizedSearchSystem()

# 테스트용 문서 추가
test_documents = [
    Document(
        page_content="인공지능과 기계학습 기술의 발전",
        metadata={"category": "ai", "tags": ["ai", "ml", "technology"]}
    ),
    Document(
        page_content="데이터베이스 시스템 설계 및 최적화",
        metadata={"category": "database", "tags": ["database", "optimization"]}
    ),
    Document(
        page_content="웹 개발을 위한 JavaScript 프로그래밍",
        metadata={"category": "programming", "tags": ["web", "javascript"]}
    ),
    Document(
        page_content="머신러닝 알고리즘과 딥러닝 네트워크",
        metadata={"category": "ai", "tags": ["ml", "deep learning", "algorithms"]}
    ),
    Document(
        page_content="클라우드 컴퓨팅 아키텍처와 마이크로서비스",
        metadata={"category": "cloud", "tags": ["cloud", "microservices", "architecture"]}
    )
]

search_system.chroma_db.add_documents(test_documents)

# 다양한 검색 방법 테스트
query = "인공지능 머신러닝"

print(f"\n쿼리: '{query}'\n")

# 1. 의미 기반 검색
print("1. 의미 기반 검색:")
semantic_results = search_system.semantic_search(query, k=3)
for i, doc in enumerate(semantic_results, 1):
    print(f"   {i}. {doc.page_content}")

# 2. 키워드 검색
print("\n2. 키워드 검색:")
keyword_results = search_system.keyword_search(query, k=3)
for i, doc in enumerate(keyword_results, 1):
    print(f"   {i}. {doc.page_content}")

# 3. 하이브리드 검색
print("\n3. 하이브리드 검색:")
hybrid_results = search_system.hybrid_search(query, semantic_weight=0.6, k=3)
for i, doc in enumerate(hybrid_results, 1):
    print(f"   {i}. {doc.page_content}")

# 4. 캐시된 검색 (두 번째 실행)
print("\n4. 캐시된 검색 (반복 쿼리):")
cached_results = search_system.cached_search(query, "hybrid", k=3)
for i, doc in enumerate(cached_results, 1):
    print(f"   {i}. {doc.page_content}")

# 검색 통계 확인
stats = search_system.get_search_statistics()
print(f"\n검색 통계:")
for key, value in stats.items():
    print(f"  {key}: {value}")
```

### 실습 3 해답: 벡터 저장소 마이그레이션 도구
```python
import time
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class VectorStoreMigrator:
    """벡터 저장소 간 데이터 마이그레이션"""

    def __init__(self):
        self.migration_log = []

    def migrate_data(
        self,
        source_store: Any,
        target_store: Any,
        batch_size: int = 100,
        verify_embeddings: bool = False
    ) -> Dict[str, int]:
        """소스에서 타겟으로 데이터 마이그레이션"""

        print("=== 벡터 저장소 마이그레이션 시작 ===")
        start_time = time.time()

        # 1. 소스 데이터 추출
        print("1. 소스 데이터 추출 중...")
        source_data = self._extract_all_data(source_store)

        total_docs = len(source_data["documents"])
        print(f"   추출된 문서 수: {total_docs}")

        if total_docs == 0:
            return {"migrated": 0, "failed": 0, "skipped": 0}

        # 2. 배치별 마이그레이션
        print("2. 데이터 마이그레이션 중...")

        migrated_count = 0
        failed_count = 0
        skipped_count = 0

        # 진행상황 표시를 위한 tqdm 사용
        with tqdm(total=total_docs, desc="마이그레이션") as pbar:
            for i in range(0, total_docs, batch_size):
                batch_end = min(i + batch_size, total_docs)
                batch_docs = source_data["documents"][i:batch_end]
                batch_metadatas = source_data["metadatas"][i:batch_end]
                batch_ids = source_data["ids"][i:batch_end]

                # Document 객체 생성
                documents = [
                    Document(page_content=content, metadata=metadata)
                    for content, metadata in zip(batch_docs, batch_metadatas)
                ]

                try:
                    # 타겟에 배치 삽입
                    target_store.add_documents(documents=documents, ids=batch_ids)
                    migrated_count += len(documents)

                    # 임베딩 검증 (옵션)
                    if verify_embeddings:
                        verification_result = self._verify_batch_embeddings(
                            source_store, target_store, batch_ids[0:1]  # 첫 번째만 검증
                        )
                        if not verification_result["success"]:
                            print(f"   경고: 배치 {i//batch_size + 1} 임베딩 검증 실패")

                except Exception as e:
                    failed_count += len(documents)
                    self.migration_log.append({
                        "batch": i // batch_size + 1,
                        "error": str(e),
                        "failed_ids": batch_ids
                    })
                    print(f"   배치 {i//batch_size + 1} 마이그레이션 실패: {e}")

                pbar.update(batch_end - i)

                # 잠깐 대기 (시스템 부하 방지)
                time.sleep(0.1)

        end_time = time.time()
        migration_time = end_time - start_time

        # 3. 결과 요약
        result = {
            "migrated": migrated_count,
            "failed": failed_count,
            "skipped": skipped_count,
            "total_time": migration_time,
            "docs_per_second": migrated_count / migration_time if migration_time > 0 else 0
        }

        print(f"\n=== 마이그레이션 완료 ===")
        print(f"성공: {migrated_count}개")
        print(f"실패: {failed_count}개")
        print(f"총 소요 시간: {migration_time:.2f}초")
        print(f"처리 속도: {result['docs_per_second']:.1f} docs/sec")

        return result

    def verify_migration(
        self,
        source_store: Any,
        target_store: Any,
        sample_size: int = 10,
        embedding_threshold: float = 0.95
    ) -> Dict[str, Any]:
        """마이그레이션 검증"""

        print("=== 마이그레이션 검증 시작 ===")

        # 1. 데이터 무결성 검증
        print("1. 데이터 무결성 검증...")
        integrity_result = self._verify_data_integrity(source_store, target_store)

        # 2. 임베딩 유사도 검증
        print("2. 임베딩 유사도 검증...")
        embedding_result = self._verify_embedding_similarity(
            source_store, target_store, sample_size, embedding_threshold
        )

        # 3. 검색 기능 검증
        print("3. 검색 기능 검증...")
        search_result = self._verify_search_functionality(source_store, target_store)

        # 전체 검증 결과
        overall_success = (
            integrity_result["success"] and
            embedding_result["success"] and
            search_result["success"]
        )

        verification_result = {
            "overall_success": overall_success,
            "data_integrity": integrity_result,
            "embedding_similarity": embedding_result,
            "search_functionality": search_result
        }

        print(f"\n=== 검증 완료 ===")
        print(f"전체 검증 결과: {'성공' if overall_success else '실패'}")

        return verification_result

    def _extract_all_data(self, vectorstore: Any) -> Dict[str, List]:
        """벡터 저장소에서 모든 데이터 추출"""
        try:
            if hasattr(vectorstore, 'get'):
                # Chroma 스타일
                data = vectorstore.get()
                return {
                    "documents": data["documents"],
                    "metadatas": data["metadatas"],
                    "ids": data["ids"]
                }
            elif hasattr(vectorstore, 'docstore'):
                # FAISS 스타일
                documents = []
                metadatas = []
                ids = []

                for idx, doc_id in vectorstore.index_to_docstore_id.items():
                    doc = vectorstore.docstore.search(doc_id)
                    documents.append(doc.page_content)
                    metadatas.append(doc.metadata)
                    ids.append(doc_id)

                return {
                    "documents": documents,
                    "metadatas": metadatas,
                    "ids": ids
                }
            else:
                raise NotImplementedError(f"지원하지 않는 벡터 저장소 타입")

        except Exception as e:
            print(f"데이터 추출 실패: {e}")
            return {"documents": [], "metadatas": [], "ids": []}

    def _verify_data_integrity(
        self,
        source_store: Any,
        target_store: Any
    ) -> Dict[str, Any]:
        """데이터 무결성 검증"""

        try:
            source_data = self._extract_all_data(source_store)
            target_data = self._extract_all_data(target_store)

            source_count = len(source_data["documents"])
            target_count = len(target_data["documents"])

            # 문서 수 비교
            count_match = source_count == target_count

            # ID 일치 확인
            source_ids = set(source_data["ids"])
            target_ids = set(target_data["ids"])
            ids_match = source_ids == target_ids

            # 내용 일치 확인 (샘플링)
            content_matches = 0
            total_checked = 0

            for i, doc_id in enumerate(source_data["ids"][:10]):  # 처음 10개만 확인
                if doc_id in target_data["ids"]:
                    target_idx = target_data["ids"].index(doc_id)
                    if source_data["documents"][i] == target_data["documents"][target_idx]:
                        content_matches += 1
                    total_checked += 1

            content_match_rate = content_matches / total_checked if total_checked > 0 else 0

            return {
                "success": count_match and ids_match and content_match_rate > 0.9,
                "source_count": source_count,
                "target_count": target_count,
                "count_match": count_match,
                "ids_match": ids_match,
                "content_match_rate": content_match_rate
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _verify_embedding_similarity(
        self,
        source_store: Any,
        target_store: Any,
        sample_size: int,
        threshold: float
    ) -> Dict[str, Any]:
        """임베딩 유사도 검증"""

        try:
            # 샘플 문서 선택
            source_data = self._extract_all_data(source_store)

            if len(source_data["documents"]) == 0:
                return {"success": False, "error": "검증할 문서가 없습니다"}

            sample_indices = np.random.choice(
                len(source_data["documents"]),
                size=min(sample_size, len(source_data["documents"])),
                replace=False
            )

            similarities = []

            for idx in sample_indices:
                doc_content = source_data["documents"][idx]

                # 각 벡터 저장소에서 임베딩 생성
                if hasattr(source_store, 'embedding_function'):
                    source_embedding = source_store.embedding_function.embed_query(doc_content)
                else:
                    continue

                if hasattr(target_store, 'embedding_function'):
                    target_embedding = target_store.embedding_function.embed_query(doc_content)
                else:
                    continue

                # 코사인 유사도 계산
                similarity = cosine_similarity(
                    [source_embedding], [target_embedding]
                )[0][0]
                similarities.append(similarity)

            if similarities:
                avg_similarity = np.mean(similarities)
                min_similarity = np.min(similarities)
                success = avg_similarity >= threshold
            else:
                avg_similarity = 0
                min_similarity = 0
                success = False

            return {
                "success": success,
                "avg_similarity": float(avg_similarity),
                "min_similarity": float(min_similarity),
                "threshold": threshold,
                "samples_checked": len(similarities)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _verify_search_functionality(
        self,
        source_store: Any,
        target_store: Any
    ) -> Dict[str, Any]:
        """검색 기능 검증"""

        try:
            test_queries = ["테스트", "인공지능", "데이터베이스"]

            search_results = []

            for query in test_queries:
                try:
                    source_results = source_store.similarity_search(query, k=3)
                    target_results = target_store.similarity_search(query, k=3)

                    # 결과 수 비교
                    count_match = len(source_results) == len(target_results)

                    # 상위 결과 내용 비교
                    content_match = False
                    if source_results and target_results:
                        content_match = (
                            source_results[0].page_content == target_results[0].page_content
                        )

                    search_results.append({
                        "query": query,
                        "count_match": count_match,
                        "content_match": content_match
                    })

                except Exception as e:
                    search_results.append({
                        "query": query,
                        "error": str(e)
                    })

            # 전체 성공률 계산
            successful_searches = sum(
                1 for result in search_results
                if result.get("count_match", False) and result.get("content_match", False)
            )

            success_rate = successful_searches / len(test_queries)

            return {
                "success": success_rate >= 0.8,  # 80% 이상 성공
                "success_rate": success_rate,
                "detailed_results": search_results
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _verify_batch_embeddings(
        self,
        source_store: Any,
        target_store: Any,
        sample_ids: List[str]
    ) -> Dict[str, bool]:
        """배치 임베딩 검증"""
        try:
            # 간단한 검색 테스트로 대체
            for doc_id in sample_ids:
                source_search = source_store.similarity_search("test", k=1)
                target_search = target_store.similarity_search("test", k=1)

                if not source_search or not target_search:
                    return {"success": False}

            return {"success": True}

        except Exception:
            return {"success": False}

    def get_migration_log(self) -> List[Dict[str, Any]]:
        """마이그레이션 로그 반환"""
        return self.migration_log.copy()

# 벡터 저장소 마이그레이션 도구 테스트
print("=== 벡터 저장소 마이그레이션 도구 테스트 ===")

# 마이그레이터 초기화
migrator = VectorStoreMigrator()

# 소스 및 타겟 벡터 저장소 생성
embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# 소스: Chroma
source_chroma = Chroma(
    collection_name="migration_source",
    embedding_function=embeddings_model,
    persist_directory="./migration_source"
)

# 타겟: 새로운 Chroma (또는 FAISS)
target_chroma = Chroma(
    collection_name="migration_target",
    embedding_function=embeddings_model,
    persist_directory="./migration_target"
)

# 소스에 테스트 데이터 추가
test_migration_docs = [
    Document(
        page_content=f"마이그레이션 테스트 문서 {i}: AI 기술과 데이터 과학",
        metadata={"doc_type": "test", "number": i, "category": "migration"}
    )
    for i in range(20)
]

source_chroma.add_documents(test_migration_docs)
print(f"소스에 {len(test_migration_docs)}개 문서 추가 완료")

# 마이그레이션 실행
migration_result = migrator.migrate_data(
    source_store=source_chroma,
    target_store=target_chroma,
    batch_size=5,
    verify_embeddings=True
)

print(f"\n마이그레이션 결과:")
for key, value in migration_result.items():
    print(f"  {key}: {value}")

# 마이그레이션 검증
verification_result = migrator.verify_migration(
    source_store=source_chroma,
    target_store=target_chroma,
    sample_size=5
)

print(f"\n검증 결과 요약:")
print(f"  전체 성공: {verification_result['overall_success']}")
print(f"  데이터 무결성: {verification_result['data_integrity']['success']}")
print(f"  임베딩 유사도: {verification_result['embedding_similarity']['success']}")
print(f"  검색 기능: {verification_result['search_functionality']['success']}")

# 마이그레이션 로그 확인
migration_log = migrator.get_migration_log()
if migration_log:
    print(f"\n마이그레이션 오류 로그: {len(migration_log)}개")
    for log_entry in migration_log:
        print(f"  배치 {log_entry['batch']}: {log_entry['error']}")
else:
    print("\n마이그레이션 오류 없음")
```

## 🔍 참고 자료

### 공식 문서
- [LangChain Vector Stores](https://python.langchain.com/docs/modules/data_connection/vectorstores/)
- [Chroma Documentation](https://docs.trychroma.com/)
- [FAISS Documentation](https://faiss.ai/)
- [Pinecone Documentation](https://docs.pinecone.io/)

### 벡터 DB 비교
| 특성 | Chroma | FAISS | Pinecone |
|------|--------|-------|----------|
| **타입** | 오픈소스 | 오픈소스 | 상용 SaaS |
| **배포** | 로컬/클라우드 | 로컬 | 클라우드만 |
| **확장성** | 중간 | 높음 | 매우 높음 |
| **비용** | 무료 | 무료 | 종량제 |
| **설정 난이도** | 쉬움 | 중간 | 쉬움 |
| **성능** | 좋음 | 매우 좋음 | 좋음 |

### 최적화 팁

#### 인덱싱 최적화
```python
# FAISS 인덱스 타입별 선택
index_types = {
    "small": faiss.IndexFlatL2,      # < 1M 벡터
    "medium": faiss.IndexIVFFlat,    # 1M-10M 벡터
    "large": faiss.IndexIVFPQ        # > 10M 벡터
}

# Chroma 배치 크기 최적화
OPTIMAL_BATCH_SIZES = {
    "small_docs": 100,    # < 1KB 문서
    "medium_docs": 50,    # 1-10KB 문서
    "large_docs": 20      # > 10KB 문서
}
```

#### 메모리 관리
```python
# 대용량 데이터 처리 예시
def process_large_dataset(documents: List[Document], batch_size: int = 1000):
    """메모리 효율적인 대용량 데이터 처리"""
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]

        # 배치 처리
        vectorstore.add_documents(batch)

        # 메모리 정리
        del batch
        import gc
        gc.collect()
```

#### 검색 성능 향상
```python
# 검색 성능 최적화 설정
search_params = {
    "chroma": {
        "include": ["metadatas", "documents", "distances"],
        "n_results": 10
    },
    "faiss": {
        "nprobe": 10,  # 검색할 클러스터 수
        "k": 10
    }
}
```