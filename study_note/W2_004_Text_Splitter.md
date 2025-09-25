# W2_004 텍스트 분할 전략

## 🎯 학습 목표
- 다양한 텍스트 분할기의 특징과 사용 사례 이해하기
- 문서의 특성에 따른 최적의 분할 전략 선택하기
- 토큰 기반 분할과 의미 기반 분할의 차이점 학습하기

## 📚 핵심 개념

### 텍스트 분할이 중요한 이유
대규모 텍스트 문서를 처리할 때 매우 중요한 전처리 단계입니다.

**주요 고려사항:**
1. **문서의 구조와 형식** - PDF, 웹페이지, 책 등
2. **원하는 청크 크기** - 토큰 수, 문자 수 제한
3. **문맥 보존의 중요도** - 의미 단위 vs 길이 단위
4. **처리 속도** - 실시간 vs 배치 처리

### 텍스트 분할기 종류

#### 1. CharacterTextSplitter
- **특징**: 문자 수 기준으로 분할하는 가장 기본적인 방식
- **장점**: 단순하고 빠름
- **단점**: 문맥을 고려하지 않음

#### 2. RecursiveCharacterTextSplitter
- **특징**: 여러 구분자를 순차적으로 적용하는 재귀적 분할
- **장점**: 문맥을 더 잘 보존
- **단점**: 완벽한 의미 보존은 어려움

#### 3. SemanticChunker
- **특징**: 임베딩을 활용한 의미 기반 분할
- **장점**: 의미적으로 일관된 청크 생성
- **단점**: 계산 비용이 높음

## 🔧 환경 설정

### 필수 라이브러리 설치
```bash
# 기본 텍스트 분할기
pip install langchain-text-splitters

# 의미 기반 분할기 (실험 기능)
pip install langchain-experimental

# 토큰화 라이브러리
pip install tiktoken transformers

# 임베딩 모델
pip install sentence-transformers
```

### 환경 변수 설정
```python
from dotenv import load_dotenv
import os
from typing import List, Dict, Any, Optional
from pprint import pprint
import json
import tiktoken
import statistics

load_dotenv()

# 기본 라이브러리
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
```

## 💻 코드 예제

### 1. CharacterTextSplitter 사용법

#### 기본 사용법
```python
from langchain_text_splitters import CharacterTextSplitter

def create_character_splitter(
    separator: str = "\n\n",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    is_separator_regex: bool = False
) -> CharacterTextSplitter:
    """CharacterTextSplitter 생성"""
    return CharacterTextSplitter(
        separator=separator,              # 청크 구분자
        chunk_size=chunk_size,            # 청크 길이
        chunk_overlap=chunk_overlap,      # 청크 중첩
        length_function=len,              # 길이 함수
        is_separator_regex=is_separator_regex,  # 정규식 사용 여부
        keep_separator=False,             # 구분자 유지 여부
        add_start_index=False,            # 시작 인덱스 추가 여부
        strip_whitespace=True,            # 공백 제거 여부
    )

# 기본 사용 예시
text_splitter = create_character_splitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200
)

# 텍스트 분할
long_text = "여기에 긴 텍스트 입력..."
chunks = text_splitter.split_text(long_text)

print(f"분할된 텍스트 개수: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"청크 {i+1} 길이: {len(chunk)}")
```

#### Document 객체 분할
```python
def split_pdf_documents(pdf_path: str) -> List[Document]:
    """PDF 문서를 로드하고 텍스트 분할"""
    # PDF 로더로 문서 로드
    pdf_loader = PyPDFLoader(pdf_path)
    pdf_docs = pdf_loader.load()

    # 텍스트 분할기 생성
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200
    )

    # Document 객체들을 분할
    chunks = text_splitter.split_documents(pdf_docs)

    print(f"원본 문서 수: {len(pdf_docs)}")
    print(f"분할된 청크 수: {len(chunks)}")

    # 각 청크의 길이 출력
    for i, chunk in enumerate(chunks):
        print(f"청크 {i+1} 길이: {len(chunk.page_content)}")

    return chunks

# 사용 예시
chunks = split_pdf_documents('./data/transformer.pdf')
```

#### 정규표현식을 활용한 문장 단위 분할
```python
def create_sentence_splitter() -> CharacterTextSplitter:
    """문장 단위로 분할하는 분할기"""
    return CharacterTextSplitter(
        separator=r'(?<=[.!?])\s+',  # 마침표, 느낌표, 물음표 뒤 공백
        chunk_size=1000,
        chunk_overlap=200,
        is_separator_regex=True,      # 정규식 사용
        keep_separator=True,          # 구분자 유지
    )

# 사용 예시
sentence_splitter = create_sentence_splitter()
sentence_chunks = sentence_splitter.split_documents(pdf_docs)

print(f"문장 단위 분할 결과: {len(sentence_chunks)}개 청크")
```

### 2. RecursiveCharacterTextSplitter 사용법

#### 기본 재귀 분할
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_recursive_splitter(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: Optional[List[str]] = None
) -> RecursiveCharacterTextSplitter:
    """재귀적 텍스트 분할기 생성"""
    if separators is None:
        # 기본 구분자 순서: 문단 → 줄 → 공백 → 문자
        separators = ["\n\n", "\n", " ", ""]

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=separators,  # 재귀적으로 순차 적용
    )

# 사용 예시
recursive_splitter = create_recursive_splitter()
recursive_chunks = recursive_splitter.split_documents(pdf_docs)

print(f"재귀 분할 결과: {len(recursive_chunks)}개 청크")

# 청크 길이 분포 확인
chunk_lengths = [len(chunk.page_content) for chunk in recursive_chunks]
print(f"청크 길이 분포: {chunk_lengths}")
```

#### 청크 겹침 분석
```python
def analyze_chunk_overlap(chunks: List[Document]) -> Dict[str, Any]:
    """청크 간 겹침 분석"""
    overlap_analysis = {
        "total_chunks": len(chunks),
        "overlaps": []
    }

    for i in range(len(chunks) - 1):
        current_chunk = chunks[i].page_content
        next_chunk = chunks[i + 1].page_content

        # 겹치는 부분 찾기 (간단한 접근법)
        overlap_length = 0
        min_length = min(len(current_chunk), len(next_chunk))

        for j in range(1, min_length + 1):
            if current_chunk[-j:] == next_chunk[:j]:
                overlap_length = j

        overlap_analysis["overlaps"].append({
            "chunk_pair": f"{i+1}-{i+2}",
            "overlap_length": overlap_length,
            "overlap_percentage": round(overlap_length / len(current_chunk) * 100, 2)
        })

    return overlap_analysis

# 겹침 분석 실행
overlap_result = analyze_chunk_overlap(recursive_chunks)
pprint(overlap_result)
```

### 3. 토큰 기반 분할

#### TikToken 활용
```python
def create_tiktoken_splitter(
    model_name: str = "gpt-4-mini",
    chunk_size: int = 300,
    chunk_overlap: int = 50
) -> RecursiveCharacterTextSplitter:
    """TikToken 기반 텍스트 분할기"""
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",  # OpenAI 모델용 인코딩
        # model_name=model_name,      # 또는 모델명 직접 지정
        chunk_size=chunk_size,        # 토큰 수 기준
        chunk_overlap=chunk_overlap,
    )

def analyze_token_usage(chunks: List[Document]) -> Dict[str, Any]:
    """토큰 사용량 분석"""
    tokenizer = tiktoken.get_encoding("cl100k_base")

    token_analysis = {
        "total_chunks": len(chunks),
        "token_counts": [],
        "total_tokens": 0
    }

    for i, chunk in enumerate(chunks):
        tokens = tokenizer.encode(chunk.page_content)
        token_count = len(tokens)

        token_analysis["token_counts"].append(token_count)
        token_analysis["total_tokens"] += token_count

        print(f"청크 {i+1}: {token_count} 토큰")
        print(f"샘플 토큰: {tokens[:5]}")  # 첫 5개 토큰
        print(f"샘플 텍스트: {tokenizer.decode(tokens[:5])}")
        print("-" * 50)

    # 통계 계산
    token_counts = token_analysis["token_counts"]
    if token_counts:
        token_analysis.update({
            "avg_tokens": round(statistics.mean(token_counts), 2),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "median_tokens": statistics.median(token_counts)
        })

    return token_analysis

# 토큰 기반 분할 실행
tiktoken_splitter = create_tiktoken_splitter(chunk_size=300)
tiktoken_chunks = tiktoken_splitter.split_documents([pdf_docs[0]])

# 토큰 분석
token_stats = analyze_token_usage(tiktoken_chunks)
pprint(token_stats)
```

#### Hugging Face 토크나이저 활용
```python
from transformers import AutoTokenizer

def create_hf_tokenizer_splitter(
    model_name: str = "BAAI/bge-m3",
    chunk_size: int = 300,
    chunk_overlap: int = 50
) -> RecursiveCharacterTextSplitter:
    """Hugging Face 토크나이저 기반 분할기"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

def compare_tokenizers(text: str) -> Dict[str, Any]:
    """서로 다른 토크나이저 비교"""
    # TikToken
    tiktoken_tokenizer = tiktoken.get_encoding("cl100k_base")
    tiktoken_tokens = tiktoken_tokenizer.encode(text)

    # Hugging Face
    hf_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    hf_tokens = hf_tokenizer.encode(text)

    return {
        "text_length": len(text),
        "tiktoken_tokens": len(tiktoken_tokens),
        "hf_tokens": len(hf_tokens),
        "tiktoken_sample": tiktoken_tokens[:10],
        "hf_sample": hf_tokens[:10],
        "tiktoken_decoded": tiktoken_tokenizer.decode(tiktoken_tokens[:10]),
        "hf_decoded": hf_tokenizer.decode(hf_tokens[:10], skip_special_tokens=True)
    }

# 토크나이저 비교
sample_text = "안녕하세요. 반갑습니다. Hello world!"
comparison = compare_tokenizers(sample_text)
pprint(comparison)
```

### 4. 의미 기반 분할 (SemanticChunker)

#### 기본 의미 분할
```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

def create_semantic_chunker(
    threshold_type: str = "percentile",
    embedding_model: str = "text-embedding-3-small"
) -> SemanticChunker:
    """의미 기반 텍스트 분할기 생성"""
    embeddings = OpenAIEmbeddings(model=embedding_model)

    return SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=threshold_type,  # gradient, percentile, standard_deviation, interquartile
    )

# 사용 예시
semantic_splitter = create_semantic_chunker(threshold_type="gradient")
semantic_chunks = semantic_splitter.split_documents([pdf_docs[0]])

print(f"의미 기반 분할 결과: {len(semantic_chunks)}개 청크")

# 각 청크의 의미적 일관성 확인
for i, chunk in enumerate(semantic_chunks):
    print(f"\n=== 의미 청크 {i+1} ===")
    print(f"길이: {len(chunk.page_content)} characters")
    print(f"내용 미리보기: {chunk.page_content[:200]}...")
```

#### 임계값 타입별 비교
```python
def compare_semantic_thresholds(document: Document) -> Dict[str, List[Document]]:
    """다양한 임계값 타입으로 의미 분할 비교"""
    threshold_types = ["gradient", "percentile", "standard_deviation", "interquartile"]
    results = {}

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    for threshold_type in threshold_types:
        try:
            splitter = SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type=threshold_type,
            )

            chunks = splitter.split_documents([document])
            results[threshold_type] = chunks

            print(f"{threshold_type}: {len(chunks)}개 청크 생성")

        except Exception as e:
            print(f"{threshold_type} 처리 중 오류: {e}")
            results[threshold_type] = []

    return results

# 임계값 비교 실행
semantic_results = compare_semantic_thresholds(pdf_docs[0])

# 결과 요약
for threshold_type, chunks in semantic_results.items():
    if chunks:
        chunk_lengths = [len(chunk.page_content) for chunk in chunks]
        print(f"\n{threshold_type}:")
        print(f"  청크 수: {len(chunks)}")
        print(f"  평균 길이: {round(statistics.mean(chunk_lengths), 2)}")
        print(f"  길이 범위: {min(chunk_lengths)} - {max(chunk_lengths)}")
```

### 5. 통합 분할 전략 비교

#### 다중 분할기 비교 함수
```python
def compare_all_splitters(document: Document) -> Dict[str, Any]:
    """모든 분할기 성능 비교"""
    results = {}

    # 1. CharacterTextSplitter
    char_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=50
    )
    char_chunks = char_splitter.split_documents([document])

    # 2. RecursiveCharacterTextSplitter
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    recursive_chunks = recursive_splitter.split_documents([document])

    # 3. TikToken 기반
    tiktoken_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=150,  # 토큰 수
        chunk_overlap=20
    )
    tiktoken_chunks = tiktoken_splitter.split_documents([document])

    # 4. SemanticChunker (가능한 경우)
    semantic_chunks = []
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        semantic_splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile"
        )
        semantic_chunks = semantic_splitter.split_documents([document])
    except:
        pass

    # 결과 정리
    splitter_results = {
        "CharacterTextSplitter": char_chunks,
        "RecursiveCharacterTextSplitter": recursive_chunks,
        "TikToken": tiktoken_chunks,
        "SemanticChunker": semantic_chunks
    }

    # 통계 계산
    comparison_stats = {}
    for name, chunks in splitter_results.items():
        if chunks:
            lengths = [len(chunk.page_content) for chunk in chunks]
            comparison_stats[name] = {
                "chunk_count": len(chunks),
                "avg_length": round(statistics.mean(lengths), 2),
                "min_length": min(lengths),
                "max_length": max(lengths),
                "total_length": sum(lengths)
            }
        else:
            comparison_stats[name] = {"chunk_count": 0}

    return {
        "chunks": splitter_results,
        "stats": comparison_stats
    }

# 전체 비교 실행
all_results = compare_all_splitters(pdf_docs[0])

# 결과 출력
print("=== 텍스트 분할기 성능 비교 ===")
for name, stats in all_results["stats"].items():
    print(f"\n{name}:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
```

#### 분할 품질 평가 함수
```python
def evaluate_chunk_quality(chunks: List[Document]) -> Dict[str, Any]:
    """청크 품질 평가"""
    if not chunks:
        return {"error": "청크가 없습니다"}

    # 길이 일관성 평가
    lengths = [len(chunk.page_content) for chunk in chunks]
    length_variance = statistics.variance(lengths) if len(lengths) > 1 else 0
    length_consistency = 1 / (1 + length_variance / 1000)  # 정규화된 일관성 점수

    # 내용 중복 평가 (간단한 접근법)
    overlap_scores = []
    for i in range(len(chunks) - 1):
        current = chunks[i].page_content.lower().split()
        next_chunk = chunks[i + 1].page_content.lower().split()

        # 공통 단어 비율 계산
        common_words = set(current) & set(next_chunk)
        overlap_ratio = len(common_words) / max(len(set(current)), len(set(next_chunk)), 1)
        overlap_scores.append(overlap_ratio)

    avg_overlap = statistics.mean(overlap_scores) if overlap_scores else 0

    return {
        "chunk_count": len(chunks),
        "length_consistency": round(length_consistency, 3),
        "avg_overlap_ratio": round(avg_overlap, 3),
        "length_stats": {
            "mean": round(statistics.mean(lengths), 2),
            "variance": round(length_variance, 2),
            "min": min(lengths),
            "max": max(lengths)
        }
    }

# 품질 평가 실행
for name, chunks in all_results["chunks"].items():
    if chunks:
        quality = evaluate_chunk_quality(chunks)
        print(f"\n=== {name} 품질 평가 ===")
        pprint(quality)
```

## 🚀 실습해보기

### 실습 1: 맞춤형 분할기 구현
다양한 문서 타입에 최적화된 분할 전략을 구현해보세요.

```python
def create_adaptive_splitter(document_type: str) -> Any:
    """문서 타입별 최적화된 분할기"""
    if document_type == "academic_paper":
        # TODO: 학술 논문용 분할기 구현
        # 섹션 헤더, 참고문헌 고려
        pass
    elif document_type == "news_article":
        # TODO: 뉴스 기사용 분할기 구현
        # 문단 단위, 짧은 청크
        pass
    elif document_type == "legal_document":
        # TODO: 법률 문서용 분할기 구현
        # 조항 단위, 정확한 경계
        pass
    else:
        # TODO: 일반 문서용 기본 분할기
        pass

# 테스트
adaptive_splitter = create_adaptive_splitter("academic_paper")
```

### 실습 2: 성능 최적화된 분할 파이프라인
대용량 문서 처리를 위한 효율적인 파이프라인을 구현해보세요.

```python
def create_efficient_pipeline(
    documents: List[Document],
    target_chunk_size: int = 1000
) -> List[Document]:
    """효율적인 텍스트 분할 파이프라인"""
    # TODO: 문서 크기별 다른 전략 적용
    # TODO: 병렬 처리 구현
    # TODO: 메모리 최적화
    # TODO: 진행상황 표시
    pass

# 테스트 데이터
large_documents = pdf_docs * 5  # 가상의 대용량 데이터
efficient_chunks = create_efficient_pipeline(large_documents)
```

### 실습 3: 동적 청크 크기 조정
문서 내용에 따라 동적으로 청크 크기를 조정하는 시스템을 구현해보세요.

```python
def dynamic_chunk_resizer(
    document: Document,
    complexity_threshold: float = 0.5
) -> List[Document]:
    """문서 복잡도에 따른 동적 청크 크기 조정"""
    # TODO: 텍스트 복잡도 분석
    # TODO: 복잡도에 따른 청크 크기 결정
    # TODO: 적응적 분할 수행
    pass

# 테스트
adaptive_chunks = dynamic_chunk_resizer(pdf_docs[0])
```

## 📋 해답

### 실습 1 해답: 맞춤형 분할기 구현
```python
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
import re

def create_adaptive_splitter(document_type: str) -> Any:
    """문서 타입별 최적화된 분할기"""
    if document_type == "academic_paper":
        # 학술 논문: 섹션 헤더와 문단을 고려한 분할
        return RecursiveCharacterTextSplitter(
            separators=[
                "\n# ",      # 주요 섹션
                "\n## ",     # 서브섹션
                "\n### ",    # 세부섹션
                "\n\n",      # 문단
                "\n",        # 줄바꿈
                ". ",        # 문장
                " ",         # 단어
                ""           # 문자
            ],
            chunk_size=1500,  # 학술 논문은 긴 청크가 유리
            chunk_overlap=200,
        )

    elif document_type == "news_article":
        # 뉴스 기사: 문단 단위의 짧은 청크
        return CharacterTextSplitter(
            separator="\n\n",    # 문단 구분
            chunk_size=800,      # 짧은 청크
            chunk_overlap=100,
            keep_separator=True,
        )

    elif document_type == "legal_document":
        # 법률 문서: 조항과 항목 단위 정확한 분할
        return CharacterTextSplitter(
            separator=r'\n(?=\d+\.|\([a-z]\)|\([0-9]+\))',  # 조항 번호 패턴
            chunk_size=1200,
            chunk_overlap=50,    # 최소 중복 (정확성 우선)
            is_separator_regex=True,
            keep_separator=True,
        )

    else:
        # 일반 문서: 균형 잡힌 기본 설정
        return RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

# 테스트용 샘플 문서 생성
def create_sample_documents():
    academic_text = """
# Introduction
This paper presents a comprehensive study on artificial intelligence.

## Methodology
We employed machine learning algorithms for data analysis.

### Data Collection
Data was collected from multiple sources over a period of six months.

## Results
Our findings indicate significant improvements in accuracy.
"""

    news_text = """
Breaking News: Major Breakthrough in AI Technology

Scientists at a leading university have announced a revolutionary development in artificial intelligence.

The new system demonstrates unprecedented capabilities in natural language processing.

Industry experts predict this will transform multiple sectors including healthcare and education.
"""

    legal_text = """
ARTICLE I - DEFINITIONS

1. For purposes of this agreement:
   (a) "Company" means the entity entering into this contract
   (b) "Service" means the work performed under this agreement

2. Terms and Conditions:
   (1) All work must be completed within specified timeframes
   (2) Quality standards must be maintained throughout
"""

    return {
        "academic_paper": Document(page_content=academic_text, metadata={"type": "academic"}),
        "news_article": Document(page_content=news_text, metadata={"type": "news"}),
        "legal_document": Document(page_content=legal_text, metadata={"type": "legal"})
    }

# 각 문서 타입별 테스트
sample_docs = create_sample_documents()

for doc_type, document in sample_docs.items():
    print(f"\n=== {doc_type.upper()} 분할 테스트 ===")

    splitter = create_adaptive_splitter(doc_type)
    chunks = splitter.split_documents([document])

    print(f"청크 수: {len(chunks)}")
    for i, chunk in enumerate(chunks, 1):
        print(f"청크 {i} (길이: {len(chunk.page_content)}):")
        print(f"  {chunk.page_content[:100]}...")
        print()
```

### 실습 2 해답: 성능 최적화된 분할 파이프라인
```python
import concurrent.futures
import time
from typing import Generator
from tqdm import tqdm

def create_efficient_pipeline(
    documents: List[Document],
    target_chunk_size: int = 1000,
    max_workers: int = 4
) -> List[Document]:
    """효율적인 텍스트 분할 파이프라인"""

    def classify_document_size(doc: Document) -> str:
        """문서 크기별 분류"""
        length = len(doc.page_content)
        if length < 2000:
            return "small"
        elif length < 10000:
            return "medium"
        else:
            return "large"

    def get_optimized_splitter(doc_size: str) -> Any:
        """문서 크기별 최적화된 분할기"""
        if doc_size == "small":
            # 작은 문서: 간단한 분할기
            return CharacterTextSplitter(
                separator="\n\n",
                chunk_size=target_chunk_size // 2,
                chunk_overlap=50
            )
        elif doc_size == "medium":
            # 중간 문서: 균형 잡힌 분할기
            return RecursiveCharacterTextSplitter(
                chunk_size=target_chunk_size,
                chunk_overlap=100
            )
        else:
            # 큰 문서: 효율적인 분할기
            return RecursiveCharacterTextSplitter(
                chunk_size=target_chunk_size * 2,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""]
            )

    def process_document(doc: Document) -> List[Document]:
        """단일 문서 처리"""
        doc_size = classify_document_size(doc)
        splitter = get_optimized_splitter(doc_size)
        return splitter.split_documents([doc])

    def process_documents_parallel(docs: List[Document]) -> List[Document]:
        """문서 병렬 처리"""
        all_chunks = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 진행상황 표시를 위한 tqdm 사용
            futures = {executor.submit(process_document, doc): doc for doc in docs}

            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(docs),
                desc="문서 처리 중"
            ):
                try:
                    chunks = future.result()
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"문서 처리 중 오류: {e}")

        return all_chunks

    def process_documents_batch(docs: List[Document], batch_size: int = 10) -> List[Document]:
        """배치 처리로 메모리 최적화"""
        all_chunks = []

        for i in tqdm(range(0, len(docs), batch_size), desc="배치 처리 중"):
            batch = docs[i:i + batch_size]
            batch_chunks = process_documents_parallel(batch)
            all_chunks.extend(batch_chunks)

            # 메모리 사용량 모니터링 (간단한 구현)
            if len(all_chunks) > 1000:  # 임계점
                print(f"현재 청크 수: {len(all_chunks)}")

        return all_chunks

    # 문서 수에 따른 처리 전략 선택
    start_time = time.time()

    if len(documents) < 50:
        # 소규모: 직접 처리
        result_chunks = process_documents_parallel(documents)
    else:
        # 대규모: 배치 처리
        result_chunks = process_documents_batch(documents)

    end_time = time.time()

    # 성능 통계 출력
    print(f"\n=== 처리 완료 ===")
    print(f"처리 시간: {end_time - start_time:.2f}초")
    print(f"원본 문서 수: {len(documents)}")
    print(f"생성된 청크 수: {len(result_chunks)}")
    print(f"처리 속도: {len(documents) / (end_time - start_time):.2f} 문서/초")

    return result_chunks

# 테스트 데이터 생성
def create_test_documents(count: int = 20) -> List[Document]:
    """테스트용 문서 생성"""
    test_docs = []

    base_texts = [
        "짧은 문서 내용입니다. " * 50,
        "중간 길이의 문서 내용입니다. " * 200,
        "긴 문서 내용입니다. " * 500,
    ]

    for i in range(count):
        text_type = i % len(base_texts)
        content = base_texts[text_type] + f" 문서 번호: {i+1}"

        doc = Document(
            page_content=content,
            metadata={"doc_id": i+1, "type": f"test_doc_{text_type}"}
        )
        test_docs.append(doc)

    return test_docs

# 효율적인 파이프라인 테스트
test_documents = create_test_documents(count=25)
efficient_chunks = create_efficient_pipeline(test_documents, target_chunk_size=800)

# 결과 분석
chunk_lengths = [len(chunk.page_content) for chunk in efficient_chunks]
print(f"\n청크 길이 통계:")
print(f"평균: {statistics.mean(chunk_lengths):.2f}")
print(f"최소: {min(chunk_lengths)}")
print(f"최대: {max(chunk_lengths)}")
```

### 실습 3 해답: 동적 청크 크기 조정
```python
import re
from collections import Counter

def dynamic_chunk_resizer(
    document: Document,
    complexity_threshold: float = 0.5
) -> List[Document]:
    """문서 복잡도에 따른 동적 청크 크기 조정"""

    def calculate_text_complexity(text: str) -> float:
        """텍스트 복잡도 계산"""
        # 다양한 복잡도 지표 계산
        words = text.split()
        sentences = re.split(r'[.!?]+', text)

        # 1. 평균 문장 길이
        avg_sentence_length = len(words) / max(len(sentences), 1)

        # 2. 어휘 다양성 (unique words / total words)
        unique_words = len(set(word.lower() for word in words))
        lexical_diversity = unique_words / max(len(words), 1)

        # 3. 구두점 밀도
        punctuation_count = len(re.findall(r'[^\w\s]', text))
        punctuation_density = punctuation_count / max(len(text), 1)

        # 4. 숫자와 특수문자 비율
        special_chars = len(re.findall(r'[\d\(\)\[\]{}]', text))
        special_char_ratio = special_chars / max(len(text), 1)

        # 복잡도 점수 계산 (0-1 범위로 정규화)
        complexity = (
            min(avg_sentence_length / 20, 1) * 0.3 +  # 문장 길이 가중치
            lexical_diversity * 0.3 +                  # 어휘 다양성 가중치
            min(punctuation_density * 20, 1) * 0.2 +   # 구두점 가중치
            min(special_char_ratio * 10, 1) * 0.2      # 특수문자 가중치
        )

        return min(complexity, 1.0)

    def determine_chunk_size(complexity: float, base_size: int = 1000) -> int:
        """복잡도에 따른 청크 크기 결정"""
        if complexity < 0.3:
            # 낮은 복잡도: 큰 청크
            return int(base_size * 1.5)
        elif complexity > 0.7:
            # 높은 복잡도: 작은 청크
            return int(base_size * 0.7)
        else:
            # 중간 복잡도: 기본 청크
            return base_size

    def analyze_document_sections(document: Document) -> List[Dict[str, Any]]:
        """문서를 섹션별로 분석"""
        content = document.page_content

        # 문단으로 분할
        paragraphs = content.split('\n\n')
        sections = []

        for i, paragraph in enumerate(paragraphs):
            if paragraph.strip():
                complexity = calculate_text_complexity(paragraph)
                sections.append({
                    'index': i,
                    'content': paragraph,
                    'complexity': complexity,
                    'length': len(paragraph)
                })

        return sections

    # 문서 섹션별 분석
    sections = analyze_document_sections(document)

    if not sections:
        # 빈 문서의 경우 기본 분할기 사용
        basic_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return basic_splitter.split_documents([document])

    # 전체 문서 복잡도 계산
    overall_complexity = statistics.mean([section['complexity'] for section in sections])
    print(f"전체 문서 복잡도: {overall_complexity:.3f}")

    # 동적 청크 생성
    adaptive_chunks = []
    current_chunk = ""
    current_metadata = document.metadata.copy()

    for section in sections:
        section_complexity = section['complexity']
        target_chunk_size = determine_chunk_size(section_complexity)

        print(f"섹션 {section['index']}: 복잡도={section_complexity:.3f}, 목표크기={target_chunk_size}")

        # 현재 청크에 섹션 추가
        if current_chunk:
            test_chunk = current_chunk + "\n\n" + section['content']
        else:
            test_chunk = section['content']

        # 청크 크기 확인
        if len(test_chunk) <= target_chunk_size or not current_chunk:
            current_chunk = test_chunk
        else:
            # 현재 청크 완료, 새로운 청크 시작
            if current_chunk:
                chunk_doc = Document(
                    page_content=current_chunk,
                    metadata={**current_metadata, "chunk_type": "adaptive"}
                )
                adaptive_chunks.append(chunk_doc)

            current_chunk = section['content']

    # 마지막 청크 처리
    if current_chunk:
        chunk_doc = Document(
            page_content=current_chunk,
            metadata={**current_metadata, "chunk_type": "adaptive"}
        )
        adaptive_chunks.append(chunk_doc)

    # 결과가 없는 경우 기본 분할 수행
    if not adaptive_chunks:
        basic_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        adaptive_chunks = basic_splitter.split_documents([document])

    return adaptive_chunks

# 복잡한 테스트 문서 생성
def create_complex_document() -> Document:
    """다양한 복잡도를 가진 테스트 문서 생성"""
    content = """
인공지능 기술 개요

인공지능은 현대 기술의 핵심 분야입니다. 머신러닝과 딥러닝을 통해 다양한 문제를 해결합니다.

수학적 모델링과 알고리즘 설계

복잡한 수학적 개념을 포함합니다: f(x) = ∑(i=1 to n) wi * xi + b
신경망의 활성화 함수로는 ReLU(x) = max(0, x), Sigmoid(x) = 1/(1+e^(-x)) 등이 있습니다.
역전파 알고리즘은 ∂L/∂w = (∂L/∂y) * (∂y/∂w)로 표현됩니다.

간단한 실생활 응용

음성인식은 일상생활에서 매우 유용합니다. 스마트폰의 음성비서가 대표적인 예입니다.
이미지 인식 기술도 사진 앱에서 얼굴을 자동으로 찾아주는 기능으로 사용됩니다.

고급 연구 동향과 미래 전망

최신 연구에서는 Transformer 아키텍처의 attention mechanism이 핵심적 역할을 합니다.
Multi-head self-attention의 수식은 Attention(Q,K,V) = softmax(QK^T/√d_k)V입니다.
GPT, BERT 등의 대규모 언어모델은 수십억 개의 파라미터를 가지며,
training loss는 cross-entropy: L = -∑(i=1 to n) yi * log(pi)로 계산됩니다.

결론

AI 기술은 계속 발전하고 있습니다. 앞으로 더 많은 혁신이 기대됩니다.
"""

    return Document(
        page_content=content,
        metadata={"source": "ai_overview.txt", "type": "educational"}
    )

# 동적 청크 크기 조정 테스트
complex_doc = create_complex_document()
adaptive_chunks = dynamic_chunk_resizer(complex_doc, complexity_threshold=0.5)

print(f"\n=== 동적 분할 결과 ===")
print(f"생성된 청크 수: {len(adaptive_chunks)}")

for i, chunk in enumerate(adaptive_chunks, 1):
    complexity = calculate_text_complexity(chunk.page_content) if 'calculate_text_complexity' in locals() else 0
    print(f"\n청크 {i}:")
    print(f"  길이: {len(chunk.page_content)} characters")
    print(f"  복잡도: {complexity:.3f}")
    print(f"  내용 미리보기: {chunk.page_content[:100]}...")

# 기본 분할기와 비교
basic_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
basic_chunks = basic_splitter.split_documents([complex_doc])

print(f"\n=== 기본 분할과 비교 ===")
print(f"동적 분할: {len(adaptive_chunks)}개 청크")
print(f"기본 분할: {len(basic_chunks)}개 청크")

# 길이 분포 비교
adaptive_lengths = [len(chunk.page_content) for chunk in adaptive_chunks]
basic_lengths = [len(chunk.page_content) for chunk in basic_chunks]

print(f"\n길이 통계 비교:")
print(f"동적 분할 - 평균: {statistics.mean(adaptive_lengths):.1f}, 범위: {min(adaptive_lengths)}-{max(adaptive_lengths)}")
print(f"기본 분할 - 평균: {statistics.mean(basic_lengths):.1f}, 범위: {min(basic_lengths)}-{max(basic_lengths)}")
```

## 🔍 참고 자료

### 공식 문서
- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [RecursiveCharacterTextSplitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter)
- [SemanticChunker](https://python.langchain.com/docs/modules/data_connection/document_transformers/semantic-chunker)

### 토큰화 관련
- [TikToken Documentation](https://github.com/openai/tiktoken)
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/)
- [OpenAI Token Limits](https://platform.openai.com/docs/models/gpt-4)

### 관련 연구
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 아키텍처
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Text Segmentation](https://en.wikipedia.org/wiki/Text_segmentation) - 텍스트 분할 이론

### 최적화 팁

#### 성능 최적화
- **대용량 문서**: 병렬 처리와 배치 처리 활용
- **메모리 관리**: 청크 단위로 처리하여 메모리 사용량 제어
- **토큰 효율성**: 모델별 토큰 한계 고려한 청크 크기 설정

#### 품질 향상
- **문맥 보존**: 적절한 중복(overlap) 설정으로 연속성 유지
- **의미 단위**: SemanticChunker로 의미적 일관성 확보
- **도메인 특화**: 문서 타입별 맞춤형 분할 전략 적용

#### 실전 가이드
```python
# 권장 설정값
CHUNK_SIZES = {
    "short_form": 500,    # 뉴스, 블로그
    "medium_form": 1000,  # 일반 문서
    "long_form": 1500,    # 학술 논문, 책
    "technical": 800,     # 기술 문서, API 문서
}

OVERLAPS = {
    "conservative": 50,   # 최소 중복
    "balanced": 200,      # 균형잡힌 중복
    "aggressive": 300,    # 최대 문맥 보존
}
```