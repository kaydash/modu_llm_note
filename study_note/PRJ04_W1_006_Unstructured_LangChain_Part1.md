# PRJ04_W1_006 - Unstructured & LangChain 통합 Part1: 기본 개념 및 로더

## 📚 학습 목표

이 가이드를 완료하면 다음을 할 수 있습니다:

1. **Unstructured & LangChain 통합 이해**: 두 라이브러리의 결합으로 얻을 수 있는 이점 파악
2. **UnstructuredLoader 활용**: 다양한 파일 형식을 하나의 로더로 처리
3. **파일 형식별 전용 로더 사용**: PDF, Word, Excel, PowerPoint 등 각 형식에 최적화된 로더 활용
4. **문서 후처리 적용**: 추출된 텍스트의 품질 향상을 위한 후처리 기법
5. **이미지/테이블 추출**: 문서 내 이미지와 표를 정확하게 추출하는 방법
6. **청킹(Chunking) 전략**: 문서를 의미 단위로 분할하는 기법

---

## 🎯 핵심 개념

### 1. Unstructured와 LangChain의 통합

**Unstructured**는 다양한 비정형 데이터를 구조화된 형태로 변환하는 라이브러리입니다.

**LangChain**은 LLM 애플리케이션 개발을 위한 프레임워크입니다.

**통합의 이점:**
```
Unstructured (문서 파싱) + LangChain (LLM 통합) = 강력한 문서 처리 시스템
```

**지원하는 파일 형식:**
- 텍스트: TXT, MD
- 문서: PDF, DOCX, PPTX, XLSX
- 웹: HTML, Email
- 미디어: 이미지 (OCR)

### 2. Document Loader 개념

**Document Loader**는 외부 소스에서 데이터를 Document 객체로 변환하는 역할을 합니다.

**Document 객체 구조:**
```python
{
    "page_content": "문서의 텍스트 내용",
    "metadata": {
        "source": "파일 경로",
        "page": 1,
        ...
    }
}
```

**주요 메서드:**
- `load()`: 전체 문서를 한 번에 로드
- `lazy_load()`: 페이지 단위로 순차적으로 로드 (메모리 효율적)

### 3. Partitioning (파티셔닝)

문서를 **의미 있는 단위(요소)**로 분할하는 과정입니다.

**주요 요소 타입:**
- **Title**: 제목
- **NarrativeText**: 본문 텍스트
- **ListItem**: 목록 항목
- **Table**: 표
- **Image**: 이미지
- **Header/Footer**: 머리말/꼬리말

**전략:**
- `auto`: 자동 선택 (기본값)
- `fast`: 빠른 처리 (텍스트 기반)
- `hi_res`: 고해상도 (레이아웃 분석)
- `ocr_only`: OCR 전용

### 4. Post-Processing (후처리)

추출된 텍스트를 정제하는 과정입니다.

**주요 후처리 함수:**
- `clean_extra_whitespace`: 불필요한 공백 제거
- `replace_unicode_quotes`: 유니코드 따옴표를 ASCII로 변환
- `clean_non_ascii_chars`: 비ASCII 문자 제거
- `clean_bullets`: 불릿 문자 정리

**사용 예시:**
```python
from unstructured.cleaners.core import clean_extra_whitespace

loader = UnstructuredLoader(
    "file.pdf",
    post_processors=[clean_extra_whitespace]
)
```

---

## 🔧 환경 설정

### 1. 필수 라이브러리 설치

```bash
# LangChain Unstructured 통합
pip install langchain-unstructured

# 또는 langchain-community 사용
pip install langchain-community

# 기본 라이브러리
pip install unstructured pandas python-dotenv

# PDF 처리용 (선택)
pip install "unstructured[pdf]"

# 이미지 처리용 (선택)
pip install "unstructured[all-docs]"
```

### 2. 시스템 의존성 설치

**PDF 처리 (Poppler):**

Windows:
```powershell
# Chocolatey 사용
choco install poppler

# 또는 수동 설치
# https://github.com/oschwartz10612/poppler-windows/releases
```

Mac:
```bash
brew install poppler
```

Linux:
```bash
sudo apt-get install poppler-utils
```

**OCR (Tesseract):**

Windows:
```powershell
# Chocolatey 사용
choco install tesseract

# 또는 수동 설치
# https://github.com/UB-Mannheim/tesseract/wiki
```

Mac:
```bash
brew install tesseract tesseract-lang
```

Linux:
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-kor
```

### 3. 환경 변수 설정

```python
from dotenv import load_dotenv
import os

# 환경 변수 로드
load_dotenv()

# OpenAI API 키 (RAG 사용 시 필요)
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "your-api-key-here"

print("✅ 환경 설정 완료!")
```

### 4. 로깅 설정 (경고 메시지 숨기기)

```python
import logging
import warnings

# 경고 메시지 필터링
warnings.filterwarnings('ignore')

# Unstructured 로깅 레벨 설정
logging.getLogger('pdfminer').setLevel(logging.ERROR)
logging.getLogger('unstructured').setLevel(logging.ERROR)

print("✅ 로깅 설정 완료!")
```

---

## 💻 단계별 구현 가이드

### 단계 1: UnstructuredLoader 기본 사용법

#### 1.1 단일 파일 로드

**UnstructuredLoader**는 여러 파일 형식을 자동으로 인식하고 처리합니다.

```python
# ⚠️ 최신 버전 (권장)
from langchain_unstructured import UnstructuredLoader

# ⚠️ 또는 Community 버전
# from langchain_community.document_loaders import UnstructuredFileLoader

# 단일 파일 로더 생성
loader = UnstructuredLoader("data/테슬라_KR.txt")

# 문서 로드
docs = loader.load()

print(f"✅ 로드된 문서 수: {len(docs)}")
print(f"\n📄 문서 내용 (일부):\n{docs[0].page_content[:200]}...")
print(f"\n📊 메타데이터:\n{docs[0].metadata}")
```

**출력 예시:**
```
✅ 로드된 문서 수: 1

📄 문서 내용 (일부):
테슬라(Tesla, Inc.)는 미국의 전기자동차 및 청정 에너지 회사입니다.
2003년 마틴 에버하드와 마크 타페닝이 설립했으며...

📊 메타데이터:
{'source': 'data/테슬라_KR.txt', 'category': 'NarrativeText'}
```

#### 1.2 여러 파일 로드

**방법 1: DirectoryLoader 사용 (같은 디렉토리)**

```python
from langchain_community.document_loaders import DirectoryLoader
from langchain_unstructured import UnstructuredLoader

# 디렉토리 내 모든 PDF와 TXT 파일 로드
loader = DirectoryLoader(
    "data/",
    glob="**/*.{pdf,txt}",  # PDF와 TXT 파일
    loader_cls=UnstructuredLoader
)

docs = loader.load()
print(f"✅ 로드된 문서 수: {len(docs)}")
```

**방법 2: 개별 로더 순회 (다른 위치)**

```python
# 여러 파일 경로 정의
file_paths = [
    "data/리비안_KR_with_table.pdf",
    "data/테슬라_KR.txt",
    "data/개인정보보호법.docx"
]

# 각 파일을 개별적으로 로드
all_docs = []
for file_path in file_paths:
    loader = UnstructuredLoader(file_path)
    docs = loader.load()
    all_docs.extend(docs)
    print(f"✅ {file_path}: {len(docs)}개 문서")

print(f"\n총 {len(all_docs)}개 문서 로드 완료!")
```

#### 1.3 일반 로딩 vs 지연 로딩

**일반 로딩 (`load()`):**
- 전체 문서를 한 번에 메모리에 로드
- 빠르지만 메모리 사용량 큼
- 작은 파일에 적합

```python
# 일반 로딩
docs = loader.load()
print(f"로드된 문서: {len(docs)}개")
```

**지연 로딩 (`lazy_load()`):**
- 페이지/요소 단위로 순차 처리
- 메모리 효율적
- 대용량 파일에 적합

```python
# 지연 로딩 (메모리 효율적)
for doc in loader.lazy_load():
    print(f"📄 {doc.metadata.get('category', 'Unknown')}: {doc.page_content[:100]}...")
    # 각 문서를 순차적으로 처리
```

---

### 단계 2: 후처리(Post Processing)

#### 2.1 불필요한 공백 제거

```python
from unstructured.cleaners.core import clean_extra_whitespace

# 후처리 적용
loader = UnstructuredLoader(
    "data/리비안_KR_with_table.pdf",
    post_processors=[clean_extra_whitespace]  # 공백 정리
)

# 문서 로드
docs = loader.load()

# 결과 확인
for doc in docs[:3]:  # 처음 3개만
    print(f"\n{doc.metadata.get('category', 'Unknown')}:")
    print(doc.page_content[:200])
    print("-" * 60)
```

#### 2.2 여러 후처리 함수 조합

```python
from unstructured.cleaners.core import (
    clean_extra_whitespace,
    replace_unicode_quotes,
    clean_non_ascii_chars
)

# 여러 후처리 함수 적용
loader = UnstructuredLoader(
    "data/document.pdf",
    post_processors=[
        clean_extra_whitespace,      # 1. 공백 정리
        replace_unicode_quotes,       # 2. 따옴표 정규화
        clean_non_ascii_chars         # 3. 비ASCII 문자 제거
    ]
)

docs = loader.load()
```

**후처리 전/후 비교:**
```python
# 후처리 전
"테슬라는    전기차를     만듭니다."

# 후처리 후 (clean_extra_whitespace 적용)
"테슬라는 전기차를 만듭니다."
```

---

### 단계 3: 이미지/테이블 처리

#### 3.1 고해상도 파티셔닝

```python
# 고해상도 전략으로 이미지와 표 추출
loader = UnstructuredLoader(
    "data/리비안_KR_with_table.pdf",

    # 파티셔닝 전략
    strategy="hi_res",              # 고해상도 레이아웃 분석
    infer_table_structure=True,     # 표 구조 자동 추론
    languages=["eng", "kor"],       # 영어와 한국어 지원

    # 이미지 추출 설정
    extract_images_in_pdf=True,     # PDF 내 이미지 추출
    extract_image_block_types=[
        "Image",                    # 일반 이미지
        "Table"                     # 표 이미지
    ],
    extract_image_block_output_dir="data/images/extracted"  # 저장 경로
)

# 문서 로드
docs = loader.load()

print(f"✅ 총 {len(docs)}개 요소 추출")
```

#### 3.2 요소 타입별 분류

```python
# 요소 타입별 개수 확인
from collections import Counter

categories = [doc.metadata.get('category', 'Unknown') for doc in docs]
category_counts = Counter(categories)

print("\n📊 요소 타입별 개수:")
for category, count in category_counts.items():
    print(f"  - {category}: {count}개")
```

**출력 예시:**
```
📊 요소 타입별 개수:
  - Title: 5개
  - NarrativeText: 23개
  - ListItem: 12개
  - Table: 3개
  - Image: 2개
```

#### 3.3 표(Table) 추출 및 HTML 변환

```python
# 표만 필터링
tables = [doc for doc in docs if doc.metadata.get('category') == 'Table']

print(f"\n📋 추출된 표: {len(tables)}개\n")

# 표 내용 확인
for i, table in enumerate(tables, 1):
    print(f"표 {i}:")
    print(table.page_content[:300])  # 텍스트 형태

    # HTML 형식으로 변환된 표 구조
    if 'text_as_html' in table.metadata:
        print("\nHTML 형식:")
        print(table.metadata['text_as_html'][:200])

    print("-" * 80)
```

**표 데이터 예시:**
```html
<table>
<tr>
  <th>모델</th>
  <th>가격</th>
  <th>주행거리</th>
</tr>
<tr>
  <td>Model 3</td>
  <td>$40,000</td>
  <td>358 miles</td>
</tr>
</table>
```

---

### 단계 4: 청킹(Chunking) 전략

#### 4.1 기본 청킹

Unstructured의 청킹은 **문서 구조를 고려**하여 의미 단위로 분할합니다.

```python
# 기본 청킹 전략
loader = UnstructuredLoader(
    "data/리비안_KR_with_table.pdf",
    strategy="fast",                # 빠른 파티셔닝
    chunking_strategy="basic",      # 기본 청킹 전략
    max_characters=1000,            # 최대 1000자
    new_after_n_chars=800,          # 800자 이후 새 청크 시작 검토
    overlap=200                     # 청크 간 200자 중복
)

docs = loader.load()

print(f"✅ 청킹 결과: {len(docs)}개 청크")

# 청크 확인
for i, doc in enumerate(docs[:3], 1):
    print(f"\n청크 {i} ({len(doc.page_content)}자):")
    print(doc.page_content[:200])
    print(f"카테고리: {doc.metadata.get('category')}")
    print("-" * 60)
```

#### 4.2 청킹 결과 분석

```python
# 청크 길이 분석
chunk_lengths = [len(doc.page_content) for doc in docs]

print("\n📊 청크 길이 통계:")
print(f"  - 최소: {min(chunk_lengths)}자")
print(f"  - 최대: {max(chunk_lengths)}자")
print(f"  - 평균: {sum(chunk_lengths)/len(chunk_lengths):.0f}자")
print(f"  - 총 청크 수: {len(docs)}개")
```

#### 4.3 Table과 TableChunk 확인

```python
# Table 요소 확인
for doc in docs:
    if "Table" in doc.metadata.get('category', ''):
        print(f"\n📋 {doc.metadata['category']}:")
        print(doc.page_content[:300])

        # 테이블 메타데이터 확인
        if hasattr(doc, 'table'):
            print("테이블 구조 메타데이터 존재")

        print("-" * 60)
```

---

### 단계 5: 파일 형식별 전용 로더

각 파일 형식에 최적화된 전용 로더를 사용할 수 있습니다.

#### 5.1 PDF 로더

```python
from langchain_community.document_loaders import UnstructuredPDFLoader

# PDF 전용 로더
loader = UnstructuredPDFLoader(
    "data/리비안_KR_with_table.pdf",
    strategy="hi_res",              # 고해상도 전략
    infer_table_structure=True      # 표 구조 추론
)

documents = loader.load()

print(f"✅ PDF 문서: {len(documents)}개 요소")
print(f"\n첫 번째 요소:")
print(documents[0].page_content[:300])
```

#### 5.2 Word 문서 로더

```python
from langchain_community.document_loaders import UnstructuredWordDocumentLoader

# Word 파일 로드
loader = UnstructuredWordDocumentLoader("data/개인정보보호법.docx")
documents = loader.load()

print(f"✅ Word 문서: {len(documents)}개 요소")
print(f"\n문서 내용 (일부):")
print(documents[0].page_content[:300])
```

#### 5.3 Excel 파일 로더

```python
from langchain_community.document_loaders import UnstructuredExcelLoader

# Excel 파일 로드
loader = UnstructuredExcelLoader(
    "data/경기도교육청_행정구역별_학제별_학급_학생_교원_20240401.xlsx"
)
documents = loader.load()

print(f"✅ Excel 문서: {len(documents)}개 요소")
print(f"\n시트 데이터 (일부):")
print(documents[0].page_content[:500])
```

**Excel 데이터 예시:**
```
행정구역	학제	학급수	학생수	교원수
수원시	초등학교	1250	35000	2100
성남시	초등학교	980	27000	1850
...
```

#### 5.4 PowerPoint 로더

```python
from langchain_community.document_loaders import UnstructuredPowerPointLoader

# PowerPoint 파일 로드
loader = UnstructuredPowerPointLoader(
    "data/한국철도공사_8대도시(목포)_관광 형태 분석 보고서_20220101.pptx"
)
documents = loader.load()

print(f"✅ PowerPoint: {len(documents)}개 슬라이드/요소")
print(f"\n첫 슬라이드 내용:")
print(documents[0].page_content[:300])
```

#### 5.5 HTML 로더

```python
from langchain_community.document_loaders import UnstructuredHTMLLoader

# HTML 파일 로드
loader = UnstructuredHTMLLoader("data/example.html")
documents = loader.load()

print(f"✅ HTML 문서: {len(documents)}개 요소")
print(f"\n추출된 텍스트:")
print(documents[0].page_content[:300])
```

**HTML 파싱 특징:**
- HTML 태그 제거
- 텍스트만 추출
- 구조 정보는 메타데이터에 저장

#### 5.6 이메일 로더

```python
from langchain_community.document_loaders import UnstructuredEmailLoader

# 이메일 파일 로드
loader = UnstructuredEmailLoader("data/email.eml")
documents = loader.load()

print(f"✅ 이메일: {len(documents)}개 요소")
print(f"\n이메일 내용:")
print(documents[0].page_content[:300])

# 메타데이터 확인
print(f"\n이메일 메타데이터:")
print(documents[0].metadata)
```

**이메일 메타데이터 예시:**
```python
{
    'source': 'data/email.eml',
    'subject': '회의 안내',
    'sent_from': ['sender@example.com'],
    'sent_to': ['receiver@example.com']
}
```

#### 5.7 이미지 로더 (OCR)

```python
from langchain_community.document_loaders import UnstructuredImageLoader

# 이미지 파일 로드 (OCR 사용)
loader = UnstructuredImageLoader(
    "data/transformer_table.jpg",
    mode="elements"  # 요소별로 분리
)
documents = loader.load()

print(f"✅ 이미지 OCR: {len(documents)}개 요소")
print(f"\n추출된 텍스트:")
print(documents[0].page_content[:300])
```

**OCR 주의사항:**
- Tesseract 설치 필요
- 이미지 품질에 따라 정확도 차이
- 한글 인식 시 `tesseract-ocr-kor` 필요

#### 5.8 URL 로더

```python
from langchain_community.document_loaders import UnstructuredURLLoader

# URL에서 콘텐츠 로드
urls = ["https://example.com"]
loader = UnstructuredURLLoader(urls=urls)
documents = loader.load()

print(f"✅ 웹 페이지: {len(documents)}개 요소")
print(f"\n페이지 내용:")
print(documents[0].page_content[:300])
```

**웹 크롤링 특징:**
- HTML 파싱 및 텍스트 추출
- JavaScript 렌더링 미지원
- 정적 콘텐츠만 추출

---

## 🎯 핵심 요약

### 1. 로더 선택 가이드

| 상황 | 추천 로더 | 이유 |
|------|---------|-----|
| 다양한 형식 혼합 | UnstructuredLoader | 자동 형식 인식 |
| PDF 전용 | UnstructuredPDFLoader | PDF 최적화 |
| 대용량 파일 | lazy_load() | 메모리 효율 |
| 표 추출 중요 | hi_res 전략 | 레이아웃 분석 |
| OCR 필요 | 이미지 로더 + Tesseract | 텍스트 인식 |

### 2. 파티셔닝 전략 비교

| 전략 | 속도 | 정확도 | 사용 사례 |
|------|------|--------|----------|
| `fast` | 빠름 | 중간 | 텍스트 중심 문서 |
| `hi_res` | 느림 | 높음 | 복잡한 레이아웃, 표 |
| `ocr_only` | 중간 | 중간 | 스캔 이미지 |
| `auto` | 자동 | 자동 | 기본 선택 |

### 3. 주요 파라미터

```python
UnstructuredLoader(
    file_path,
    strategy="hi_res",              # 파티셔닝 전략
    chunking_strategy="basic",      # 청킹 전략
    max_characters=1000,            # 최대 청크 크기
    overlap=200,                    # 청크 간 중복
    infer_table_structure=True,     # 표 구조 추론
    extract_images_in_pdf=True,     # 이미지 추출
    post_processors=[...],          # 후처리 함수
    languages=["eng", "kor"]        # 지원 언어
)
```

### 4. 체크리스트

**환경 설정:**
- [ ] langchain-unstructured 설치
- [ ] Poppler 설치 (PDF 처리)
- [ ] Tesseract 설치 (OCR 사용 시)
- [ ] 환경 변수 설정

**문서 로딩:**
- [ ] 적절한 로더 선택
- [ ] 파티셔닝 전략 설정
- [ ] 후처리 함수 적용 (필요 시)
- [ ] 메타데이터 확인

**성능 최적화:**
- [ ] lazy_load() 사용 (대용량)
- [ ] 청킹 전략 조정
- [ ] 불필요한 요소 필터링

---

## 📌 다음 단계

Part2에서는 **실습: 법률 문서 RAG 구현**을 다룹니다:
- 다중 파일 형식 통합 RAG 시스템
- Chroma DB를 활용한 벡터 검색
- 법률 전문가 컨텍스트 프롬프트
- 법적 근거 하이라이팅
- 법률 용어 자동 설명

---

이 가이드를 통해 **Unstructured와 LangChain을 통합하여 다양한 문서 형식을 효율적으로 처리**할 수 있습니다!
