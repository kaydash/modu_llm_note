# W2_003 Document Loader 사용법

## 🎯 학습 목표
- LangChain의 다양한 Document Loader 이해하고 활용하기
- PDF, 웹, JSON, CSV, 디렉토리 등 다양한 데이터 소스 처리 방법 학습하기
- 실제 프로젝트에서 효율적인 문서 로딩 전략 수립하기

## 📚 핵심 개념

### Document Loader란?
Document Loader는 다양한 소스에서 문서를 로드하여 LangChain의 표준 Document 형식으로 변환하는 컴포넌트입니다.

**주요 특징:**
- BaseLoader 인터페이스를 통한 일관된 구현
- `.load()` (동기) 또는 `.lazy_load()` (비동기) 메서드 제공
- 대용량 데이터셋의 경우 메모리 효율을 위해 `.lazy_load()` 권장
- 다양한 문서 형식 지원 (PDF, 웹, JSON, CSV, 텍스트, 디렉토리 등)

### Document 객체 구조
```python
Document(
    page_content="문서의 실제 텍스트 내용",
    metadata={
        "source": "파일 경로 또는 URL",
        "page": "페이지 번호",
        # 기타 메타데이터
    }
)
```

## 🔧 환경 설정

### 필수 라이브러리 설치
```bash
# 기본 설치
pip install langchain-community

# PDF 처리용
pip install pypdf

# JSON 처리용
pip install jq

# 웹 페이지 처리용
pip install beautifulsoup4 requests
```

### 환경 변수 설정
```python
from dotenv import load_dotenv
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

load_dotenv()

# 기본 라이브러리
from glob import glob
from pprint import pprint
import json
```

## 💻 코드 예제

### 1. PDF 파일 로더

#### 기본 사용법
```python
from langchain_community.document_loaders import PyPDFLoader
from typing import List
from langchain_core.documents import Document

def load_pdf_documents(file_path: str) -> List[Document]:
    """PDF 파일을 로드하여 Document 객체 리스트로 반환"""
    pdf_loader = PyPDFLoader(file_path)
    documents = pdf_loader.load()

    print(f'PDF 문서 개수: {len(documents)}')
    return documents

# 사용 예시
pdf_docs = load_pdf_documents('./data/transformer.pdf')

# 첫 번째 문서 확인
print("첫 번째 페이지 내용 미리보기:")
print(pdf_docs[0].page_content[:200])
print("\n메타데이터:")
print(pdf_docs[0].metadata)
```

#### 비동기 로딩 (대용량 파일 권장)
```python
async def load_pdf_async(file_path: str) -> None:
    """PDF를 비동기로 페이지별 처리"""
    pdf_loader = PyPDFLoader(file_path)

    async for page in pdf_loader.alazy_load():
        print(f"페이지 {page.metadata.get('page', 0) + 1}:")
        print(f"텍스트 길이: {len(page.page_content)}")
        print("-" * 80)
```

### 2. 웹 문서 로더

#### 기본 웹 로딩
```python
from langchain_community.document_loaders import WebBaseLoader
from typing import List, Optional

def load_web_documents(urls: List[str]) -> List[Document]:
    """웹 페이지를 로드하여 Document 객체로 변환"""
    web_loader = WebBaseLoader(web_paths=urls)
    documents = web_loader.load()

    print(f"로드된 웹 문서 개수: {len(documents)}")
    return documents

# 사용 예시
web_docs = load_web_documents([
    "https://python.langchain.com/",
    "https://js.langchain.com/"
])

# 메타데이터 확인
for doc in web_docs:
    print(f"제목: {doc.metadata.get('title', 'N/A')}")
    print(f"언어: {doc.metadata.get('language', 'N/A')}")
    print(f"설명: {doc.metadata.get('description', 'N/A')}")
    print("-" * 50)
```

#### BeautifulSoup을 활용한 선택적 파싱
```python
import bs4

def load_web_selective(url: str, target_class: str) -> List[Document]:
    """특정 HTML 요소만 선택적으로 파싱"""
    web_loader = WebBaseLoader(
        web_paths=[url],
        bs_kwargs={
            "parse_only": bs4.SoupStrainer(class_=target_class),
        },
        bs_get_text_kwargs={
            "separator": " | ",    # 구분자
            "strip": True          # 공백 제거
        }
    )

    return web_loader.load()

# 사용 예시
selective_docs = load_web_selective(
    "https://python.langchain.com/",
    "theme-doc-markdown markdown"
)
```

### 3. JSON 파일 로더

#### JSON 구조화된 데이터 처리
```python
from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document

def load_json_content(file_path: str, jq_schema: str, text_content: bool = True) -> List[Document]:
    """JSON 파일에서 특정 필드 추출"""
    json_loader = JSONLoader(
        file_path=file_path,
        jq_schema=jq_schema,      # JQ 스키마로 필드 선택
        text_content=text_content, # 텍스트 내용 여부
    )

    return json_loader.load()

# 메시지 내용만 추출
message_docs = load_json_content(
    "./data/kakao_chat.json",
    ".messages[].content"
)

# 전체 메시지 객체 추출
full_message_docs = load_json_content(
    "./data/kakao_chat.json",
    ".messages[]",
    text_content=False
)
```

#### JSONL 파일 처리
```python
def load_jsonl_file(file_path: str, content_key: str) -> List[Document]:
    """JSONL(JSON Lines) 파일 로드"""
    json_loader = JSONLoader(
        file_path=file_path,
        jq_schema=".",
        content_key=content_key,
        json_lines=True,  # JSONL 형식 지정
    )

    return json_loader.load()

# JSONL 파일 로드
jsonl_docs = load_jsonl_file("./data/kakao_chat.jsonl", "content")
```

#### 한글 유니코드 처리
```python
def decode_unicode_json(documents: List[Document]) -> List[Document]:
    """유니코드 이스케이프 시퀀스 디코딩"""
    decoded_docs = []

    for doc in documents:
        try:
            decoded_data = json.loads(doc.page_content)
            decoded_content = json.dumps(decoded_data, ensure_ascii=False)

            decoded_doc = Document(
                page_content=decoded_content,
                metadata=doc.metadata
            )
            decoded_docs.append(decoded_doc)

        except json.JSONDecodeError as e:
            print(f"JSON 디코딩 오류: {e}")
            decoded_docs.append(doc)  # 원본 유지

    return decoded_docs
```

### 4. CSV 파일 로더

#### 기본 CSV 로딩
```python
from langchain_community.document_loaders.csv_loader import CSVLoader

def load_csv_documents(file_path: str, source_column: Optional[str] = None) -> List[Document]:
    """CSV 파일을 Document 객체로 변환"""
    csv_loader = CSVLoader(
        file_path=file_path,
        source_column=source_column,  # 메타데이터 source 필드로 사용할 컬럼
    )

    documents = csv_loader.load()
    print(f"로드된 CSV 행 수: {len(documents)}")

    return documents

# 기본 사용법
csv_docs = load_csv_documents("./data/kbo_teams_2023.csv")

# 특정 컬럼을 소스로 사용
csv_docs_with_source = load_csv_documents("./data/kbo_teams_2023.csv", "Team")
```

#### CSV 파싱 옵션 커스터마이징
```python
def load_csv_custom(file_path: str, delimiter: str = ",", quotechar: str = '"') -> List[Document]:
    """CSV 파싱 옵션 커스터마이징"""
    csv_loader = CSVLoader(
        file_path=file_path,
        csv_args={
            "delimiter": delimiter,     # 구분자 지정
            "quotechar": quotechar,     # 따옴표 문자 지정
        }
    )

    return csv_loader.load()
```

### 5. 디렉토리 로더

#### 디렉토리 일괄 로딩
```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

def load_directory_documents(
    path: str,
    glob_pattern: str = "**/*.txt",
    loader_class=TextLoader,
    show_progress: bool = True
) -> List[Document]:
    """디렉토리에서 파일들을 일괄 로딩"""
    dir_loader = DirectoryLoader(
        path=path,
        glob=glob_pattern,           # 와일드카드 패턴
        loader_cls=loader_class,     # 사용할 로더 클래스
        show_progress=show_progress, # 진행상태 표시
    )

    documents = dir_loader.load()
    print(f"로드된 파일 수: {len(documents)}")

    return documents

# 특정 패턴의 텍스트 파일들 로드
dir_docs = load_directory_documents(
    path="./",
    glob_pattern="**/*_KR.txt",
    loader_class=TextLoader
)

# 로드된 파일 정보 출력
for doc in dir_docs:
    print(f"파일: {doc.metadata['source']}")
    print(f"텍스트 길이: {len(doc.page_content)}")
    print("-" * 50)
```

### 6. 문서 로딩 유틸리티 함수

#### 문서 정보 분석 함수
```python
def analyze_documents(documents: List[Document]) -> Dict[str, Any]:
    """로드된 문서들의 통계 정보 분석"""
    if not documents:
        return {"count": 0, "total_length": 0, "avg_length": 0}

    total_length = sum(len(doc.page_content) for doc in documents)
    avg_length = total_length / len(documents)

    # 메타데이터 키 분석
    metadata_keys = set()
    for doc in documents:
        metadata_keys.update(doc.metadata.keys())

    return {
        "count": len(documents),
        "total_length": total_length,
        "avg_length": avg_length,
        "metadata_keys": list(metadata_keys),
        "sources": [doc.metadata.get("source", "Unknown") for doc in documents]
    }

# 사용 예시
analysis = analyze_documents(pdf_docs)
pprint(analysis)
```

#### 문서 내용 미리보기 함수
```python
def preview_documents(documents: List[Document], max_preview: int = 200) -> None:
    """문서 내용 미리보기"""
    for i, doc in enumerate(documents):
        print(f"=== 문서 {i+1} ===")
        print(f"소스: {doc.metadata.get('source', 'Unknown')}")
        print(f"길이: {len(doc.page_content)} characters")

        # 내용 미리보기
        content_preview = doc.page_content[:max_preview]
        if len(doc.page_content) > max_preview:
            content_preview += "..."

        print(f"내용 미리보기:\n{content_preview}")
        print("-" * 80)

# 사용 예시
preview_documents(csv_docs, max_preview=150)
```

## 🚀 실습해보기

### 실습 1: PDF 문서 분석 시스템
PDF 파일을 로드하고 각 페이지의 내용을 분석하는 시스템을 구현해보세요.

```python
def pdf_analysis_system(pdf_path: str) -> Dict[str, Any]:
    """PDF 문서 분석 시스템"""
    # TODO: PDF 로더로 문서 로드
    # TODO: 각 페이지의 텍스트 길이 계산
    # TODO: 전체 통계 정보 반환
    pass

# 테스트 실행
result = pdf_analysis_system("./articles/notionai.pdf")
print(f"분석 결과: {result}")
```

### 실습 2: 멀티 포맷 문서 통합 로더
여러 형식의 문서를 한 번에 로드하는 통합 로더를 구현해보세요.

```python
def multi_format_loader(file_configs: List[Dict[str, Any]]) -> List[Document]:
    """다양한 형식의 파일들을 통합 로딩

    Args:
        file_configs: 파일 설정 리스트
            [
                {"type": "pdf", "path": "doc.pdf"},
                {"type": "csv", "path": "data.csv", "source_column": "id"},
                {"type": "json", "path": "data.json", "jq_schema": ".content"}
            ]
    """
    # TODO: 파일 타입별 적절한 로더 선택
    # TODO: 각 파일 로드 후 결과 병합
    # TODO: 통합된 Document 리스트 반환
    pass

# 테스트 설정
configs = [
    {"type": "pdf", "path": "./data/transformer.pdf"},
    {"type": "csv", "path": "./data/kbo_teams_2023.csv"},
    {"type": "json", "path": "./data/kakao_chat.json", "jq_schema": ".messages[].content"}
]

documents = multi_format_loader(configs)
print(f"총 {len(documents)}개 문서 로드됨")
```

### 실습 3: 웹 컨텐츠 수집기
여러 웹사이트에서 특정 내용만 선택적으로 수집하는 시스템을 구현해보세요.

```python
def web_content_collector(
    url_configs: List[Dict[str, Any]]
) -> List[Document]:
    """웹 컨텐츠 선택적 수집기

    Args:
        url_configs: URL 설정 리스트
            [
                {
                    "url": "https://example.com",
                    "target_class": "content",
                    "separator": " | "
                }
            ]
    """
    # TODO: URL별 BeautifulSoup 설정 적용
    # TODO: 선택적 컨텐츠 추출
    # TODO: Document 객체로 변환하여 반환
    pass
```

## 📋 해답

### 실습 1 해답: PDF 문서 분석 시스템
```python
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from typing import Dict, Any, List

def pdf_analysis_system(pdf_path: str) -> Dict[str, Any]:
    """PDF 문서 분석 시스템"""
    # PDF 파일 존재 확인
    if not Path(pdf_path).exists():
        # 여러 위치에서 파일 검색
        cwd = Path.cwd()
        candidates = [
            cwd / "articles" / Path(pdf_path).name,
            cwd / "data" / Path(pdf_path).name
        ]
        candidates += list(cwd.rglob(Path(pdf_path).name))

        found_path = None
        for candidate in candidates:
            if candidate.exists():
                found_path = str(candidate)
                break

        if not found_path:
            return {"error": f"PDF 파일을 찾을 수 없습니다: {pdf_path}"}

        pdf_path = found_path

    # PDF 로더로 문서 로드
    pdf_loader = PyPDFLoader(pdf_path)
    documents = pdf_loader.load()

    # 각 페이지 분석
    page_lengths = []
    for i, doc in enumerate(documents):
        page_length = len(doc.page_content)
        page_lengths.append(page_length)
        print(f"페이지 {i+1}: {page_length} characters")

    # 전체 통계 계산
    total_pages = len(documents)
    total_characters = sum(page_lengths)
    avg_length = total_characters / total_pages if total_pages > 0 else 0

    return {
        "file_path": pdf_path,
        "total_pages": total_pages,
        "total_characters": total_characters,
        "average_length_per_page": round(avg_length, 2),
        "page_lengths": page_lengths,
        "longest_page": max(page_lengths) if page_lengths else 0,
        "shortest_page": min(page_lengths) if page_lengths else 0
    }

# 테스트 실행
result = pdf_analysis_system("notionai.pdf")
pprint(result)
```

### 실습 2 해답: 멀티 포맷 문서 통합 로더
```python
from langchain_community.document_loaders import (
    PyPDFLoader, CSVLoader, JSONLoader, TextLoader
)

def multi_format_loader(file_configs: List[Dict[str, Any]]) -> List[Document]:
    """다양한 형식의 파일들을 통합 로딩"""
    all_documents = []

    for config in file_configs:
        file_type = config.get("type", "").lower()
        file_path = config.get("path", "")

        if not Path(file_path).exists():
            print(f"파일을 찾을 수 없습니다: {file_path}")
            continue

        try:
            if file_type == "pdf":
                loader = PyPDFLoader(file_path)

            elif file_type == "csv":
                source_column = config.get("source_column")
                loader = CSVLoader(
                    file_path=file_path,
                    source_column=source_column
                )

            elif file_type == "json":
                jq_schema = config.get("jq_schema", ".")
                text_content = config.get("text_content", True)
                json_lines = config.get("json_lines", False)

                loader = JSONLoader(
                    file_path=file_path,
                    jq_schema=jq_schema,
                    text_content=text_content,
                    json_lines=json_lines
                )

            elif file_type == "txt" or file_type == "text":
                loader = TextLoader(file_path)

            else:
                print(f"지원하지 않는 파일 형식: {file_type}")
                continue

            # 문서 로드
            documents = loader.load()
            all_documents.extend(documents)
            print(f"{file_type.upper()} 로드 완료: {len(documents)}개 문서")

        except Exception as e:
            print(f"{file_path} 로드 중 오류: {e}")

    return all_documents

# 테스트 실행
configs = [
    {"type": "pdf", "path": "./data/transformer.pdf"},
    {"type": "csv", "path": "./data/kbo_teams_2023.csv", "source_column": "Team"},
    {
        "type": "json",
        "path": "./data/kakao_chat.json",
        "jq_schema": ".messages[].content",
        "text_content": True
    }
]

documents = multi_format_loader(configs)
print(f"\n총 {len(documents)}개 문서가 로드되었습니다.")

# 로드된 문서들의 소스 확인
sources = {}
for doc in documents:
    source = doc.metadata.get("source", "Unknown")
    source_ext = Path(source).suffix if source != "Unknown" else "Unknown"
    sources[source_ext] = sources.get(source_ext, 0) + 1

print("\n파일 형식별 문서 수:")
for ext, count in sources.items():
    print(f"{ext}: {count}개")
```

### 실습 3 해답: 웹 컨텐츠 수집기
```python
import bs4
from langchain_community.document_loaders import WebBaseLoader

def web_content_collector(url_configs: List[Dict[str, Any]]) -> List[Document]:
    """웹 컨텐츠 선택적 수집기"""
    all_documents = []

    for config in url_configs:
        url = config.get("url", "")
        target_class = config.get("target_class")
        target_id = config.get("target_id")
        target_tag = config.get("target_tag")
        separator = config.get("separator", " ")

        if not url:
            print("URL이 제공되지 않았습니다.")
            continue

        try:
            # BeautifulSoup 설정 구성
            bs_kwargs = {}
            bs_get_text_kwargs = {
                "separator": separator,
                "strip": True
            }

            # 선택자 설정
            if target_class:
                bs_kwargs["parse_only"] = bs4.SoupStrainer(class_=target_class)
            elif target_id:
                bs_kwargs["parse_only"] = bs4.SoupStrainer(id=target_id)
            elif target_tag:
                bs_kwargs["parse_only"] = bs4.SoupStrainer(target_tag)

            # 웹 로더 생성
            web_loader = WebBaseLoader(
                web_paths=[url],
                bs_kwargs=bs_kwargs,
                bs_get_text_kwargs=bs_get_text_kwargs
            )

            # 문서 로드
            documents = web_loader.load()
            all_documents.extend(documents)

            print(f"웹 페이지 로드 완료: {url}")
            print(f"추출된 문서 수: {len(documents)}")

        except Exception as e:
            print(f"{url} 로드 중 오류: {e}")

    return all_documents

# 테스트 실행
url_configs = [
    {
        "url": "https://python.langchain.com/",
        "target_class": "theme-doc-markdown markdown",
        "separator": " | "
    }
]

web_documents = web_content_collector(url_configs)
print(f"\n총 {len(web_documents)}개의 웹 문서가 수집되었습니다.")

# 수집된 내용 미리보기
for i, doc in enumerate(web_documents):
    print(f"\n=== 문서 {i+1} ===")
    print(f"제목: {doc.metadata.get('title', 'N/A')}")
    print(f"URL: {doc.metadata.get('source', 'N/A')}")
    print(f"내용 길이: {len(doc.page_content)}")
    print(f"내용 미리보기: {doc.page_content[:200]}...")
```

## 🔍 참고 자료

### 공식 문서
- [LangChain Document Loaders](https://python.langchain.com/docs/modules/data_connection/document_loaders/)
- [PyPDFLoader Documentation](https://python.langchain.com/docs/integrations/document_loaders/pypdf/)
- [WebBaseLoader Documentation](https://python.langchain.com/docs/integrations/document_loaders/web_base/)

### 추가 학습 자료
- [BeautifulSoup Documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [jq Manual](https://stedolan.github.io/jq/manual/) - JSON 쿼리 언어
- [Python CSV Module](https://docs.python.org/3/library/csv.html)

### 관련 패키지
```python
# 핵심 패키지
langchain-community     # 커뮤니티 로더들
langchain-core         # 기본 Document 클래스

# 특화 패키지
pypdf                  # PDF 처리
beautifulsoup4         # HTML/XML 파싱
jq                     # JSON 쿼리
python-dotenv          # 환경변수 관리
```

### 성능 최적화 팁
- 대용량 파일: `.lazy_load()` 사용하여 메모리 절약
- 웹 페이지: BeautifulSoup 파싱 옵션으로 필요한 요소만 추출
- JSON 파일: jq 스키마로 필요한 필드만 선택적 추출
- 디렉토리 로딩: glob 패턴을 구체적으로 지정하여 불필요한 파일 제외