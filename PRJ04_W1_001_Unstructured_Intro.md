# 비정형 문서 처리의 이해 (Unstructured)

## 📚 학습 목표

이 학습 가이드를 완료하면 다음을 수행할 수 있습니다:

1. **Unstructured 라이브러리 이해**: 다양한 형식의 비정형 데이터를 구조화된 형태로 변환할 수 있습니다
2. **파티셔닝 (Partitioning)**: PDF, HTML 등 문서를 의미 있는 요소(제목, 본문, 표)로 자동 분할할 수 있습니다
3. **청킹 (Chunking)**: 문서를 RAG 시스템에 최적화된 크기의 청크로 나눌 수 있습니다
4. **정제 (Cleaning)**: 텍스트 데이터의 품질을 향상시키는 전처리를 수행할 수 있습니다
5. **RAG 시스템 구축**: Unstructured를 활용한 완전한 RAG 파이프라인을 구현할 수 있습니다
6. **전략 선택**: 문서 특성에 맞는 파티셔닝 전략(fast, hi_res, ocr_only)을 적용할 수 있습니다
7. **실무 활용**: 실제 프로젝트에서 비정형 문서 처리 파이프라인을 구축할 수 있습니다

## 🔑 핵심 개념

### Unstructured란?

**Unstructured**는 PDF, 이미지, HTML 등 다양한 형식의 비정형 데이터를 구조화된 형태로 변환하는 오픈소스 도구입니다.

#### 주요 특징

```
┌─────────────────────────────────────────────────────────────┐
│                  Unstructured 핵심 기능                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   │
│  │ Partitioning │──▶│   Chunking   │──▶│   Cleaning   │   │
│  │  (파티셔닝)   │   │   (청킹)      │   │   (정제)      │   │
│  └──────────────┘   └──────────────┘   └──────────────┘   │
│        │                   │                   │            │
│        │                   │                   │            │
│  문서 요소 분할        의미 단위 청크      텍스트 품질 향상   │
│  (제목, 본문, 표)     (RAG 최적화)        (NLP 전처리)      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### 지원 문서 형식

- **텍스트**: PDF, TXT, RTF, MD
- **오피스**: DOCX, XLSX, PPTX
- **이메일**: EML, MSG
- **이미지**: PNG, JPG, TIFF
- **기타**: HTML, XML, EPUB

#### 오픈소스 vs API 서비스

**오픈소스 버전** (무료):
- ✅ 기본 파티셔닝, 청킹, 정제 기능
- ✅ 로컬 환경에서 완전한 제어
- ❌ 고급 OCR 모델 미포함
- ❌ by-page, by-similarity 청킹 전략 미지원
- ❌ 프로덕션 최적화 제한

**API 서비스** (유료):
- ✅ 고급 OCR 및 테이블 감지
- ✅ 모든 청킹 전략 지원
- ✅ 확장성 및 성능 최적화
- ✅ 보안 및 규정 준수 기능

### 핵심 개념: Element (요소)

파티셔닝 후 얻게 되는 기본 구성 단위입니다.

#### Element 타입

**1. 텍스트 관련 Element**
- `NarrativeText`: 본문 텍스트
- `Title`: 제목
- `ListItem`: 목록 항목
- `UncategorizedText`: 분류되지 않은 텍스트

**2. 구조적 Element**
- `Table`: 표
- `Header`: 헤더
- `Footer`: 푸터
- `PageBreak`: 페이지 구분

**3. 특수 Element**
- `Address`: 주소
- `Formula`: 수식
- `FigureCaption`: 그림 설명
- `Image`: 이미지 메타데이터
- `EmailAddress`: 이메일 주소
- `CodeSnippet`: 코드 스니펫

## 🛠 환경 설정

### 방법 1: pip 설치 (간단)

#### 전체 설치

```bash
# 모든 문서 타입 지원
pip install "unstructured[all-docs]"

# 또는 Poetry 사용
poetry add "unstructured[all-docs]"
```

#### 시스템 의존성 (필수)

Unstructured는 다음 시스템 라이브러리가 필요합니다:

- **libmagic**: 파일 형식 감지
- **poppler**: PDF 처리
- **libreoffice**: 오피스 문서 처리
- **pandoc**: 문서 변환
- **tesseract**: OCR (이미지 텍스트 인식)

**주의**: 시스템 의존성 설치가 복잡할 수 있으므로 Docker 사용을 권장합니다.

### 방법 2: Docker 설치 (권장)

공식 Docker 이미지를 사용하면 모든 의존성이 사전 설치되어 있어 즉시 사용 가능합니다.

#### 기본 Docker 사용

```bash
# Docker 이미지 다운로드
docker pull downloads.unstructured.io/unstructured-io/unstructured:latest

# 컨테이너 생성
docker run -dt --name unstructured downloads.unstructured.io/unstructured-io/unstructured:latest

# 컨테이너 내부 접속
docker exec -it unstructured bash
```

#### 개발 환경 Docker Compose

```bash
# 저장소 클론
git clone https://github.com/tsdata/unstructured-dev-vscode.git

# 디렉토리 이동
cd unstructured-dev-vscode

# Docker Compose로 빌드 및 실행
docker-compose up -d --build
```

**출처**: https://github.com/tsdata/unstructured-dev-vscode

### 방법 3: VS Code Remote Development (개발 환경)

#### 단계 1: 확장 프로그램 설치

VS Code에서 `Remote Development` 확장 프로그램을 설치합니다 (Dev Containers 포함).

#### 단계 2: 컨테이너 연결

1. VS Code 명령 팔레트 열기 (Ctrl+Shift+P / Cmd+Shift+P)
2. `Attach to Running Container...` 선택
3. 실행 중인 `unstructured` 컨테이너 선택

#### 단계 3: 프로젝트 폴더 열기

1. `Open Folder` 선택
2. `/workspace` 폴더 선택

#### 단계 4: Python 환경 설정

1. Python/Jupyter 확장 프로그램 설치
2. 커널 선택: `/home/notebook-user/venv/bin/python`

#### 단계 5: 추가 라이브러리 설치

```bash
pip install "langchain<1.0" \
            "langchain-community<1.0" \
            "langchain-huggingface<1.0" \
            "langchain-openai<1.0" \
            sentence_transformers \
            faiss-cpu
```

### 기본 설정 코드

```python
from dotenv import load_dotenv
import os
import warnings

# 환경 변수 로드
load_dotenv()

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')

# 로깅 레벨 설정 (불필요한 로그 제거)
import logging
logging.getLogger('pdfminer').setLevel(logging.ERROR)
logging.getLogger('unstructured').setLevel(logging.ERROR)

# 기본 라이브러리
from glob import glob
from pprint import pprint
import json
import pandas as pd
import numpy as np

print("✅ 환경 설정 완료!")
```

### 트러블슈팅

**onnxruntime 오류 발생 시**:

```bash
# onnxruntime 제거
pip uninstall onnxruntime

# 특정 버전 재설치
pip install onnxruntime==1.19.2
```

## 💻 단계별 구현

### 1. 파티셔닝 (Partitioning)

파티셔닝은 비정형 문서를 구조화된 요소로 나누는 Unstructured의 핵심 기능입니다.

#### 1.1 기본 파티셔너 (Auto)

가장 간단한 방법은 `partition` 함수를 사용하는 것입니다. 파일 형식을 자동으로 감지합니다.

```python
from unstructured.partition.auto import partition

# 방법 1: 파일명으로 파티셔닝
elements = partition(filename="data/transformer.pdf")

print(f"파티션된 요소 수: {len(elements)}")
print("\n처음 3개 요소:")
for i, elem in enumerate(elements[:3], 1):
    print(f"\n[{i}] Type: {elem.category}")
    print(f"    Text: {elem.text[:100]}...")
```

**실행 결과 예시**:
```
파티션된 요소 수: 156

처음 3개 요소:

[1] Type: Title
    Text: Attention Is All You Need...

[2] Type: NarrativeText
    Text: The dominant sequence transduction models are based on complex recurrent or convolutional...

[3] Type: Title
    Text: 1 Introduction...
```

**방법 2: 파일 객체로 파티셔닝**

```python
# 파일 객체 사용
with open("data/transformer.pdf", "rb") as f:
    elements = partition(file=f)

print(f"파티션된 요소 수: {len(elements)}")
```

#### Element 속성 접근

```python
# 첫 번째 요소 상세 분석
first_elem = elements[0]

print("Element 정보:")
print(f"  - Type: {first_elem.category}")
print(f"  - Text: {first_elem.text}")
print(f"  - Page: {first_elem.metadata.page_number}")
print(f"  - Language: {first_elem.metadata.languages}")

# Dictionary로 변환
print("\nElement as Dict:")
pprint(first_elem.to_dict())
```

**실행 결과 예시**:
```
Element 정보:
  - Type: Title
  - Text: Attention Is All You Need
  - Page: 1
  - Language: ['eng']

Element as Dict:
{'element_id': 'abc123...',
 'metadata': {'filename': 'transformer.pdf',
              'languages': ['eng'],
              'page_number': 1},
 'text': 'Attention Is All You Need',
 'type': 'Title'}
```

#### 1.2 문서 유형별 전용 파티셔너

각 문서 형식에 최적화된 전용 파티셔너를 사용하면 더 나은 성능을 얻을 수 있습니다.

**PDF 문서 전용**:

```python
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf(filename="data/transformer.pdf")
print(f"PDF 파티션 요소 수: {len(elements)}")
```

**HTML 문서 전용**:

```python
from unstructured.partition.html import partition_html

elements = partition_html(filename="data/example.html")
print(f"HTML 파티션 요소 수: {len(elements)}")

# HTML 요소 확인
for elem in elements[:3]:
    print(f"Type: {elem.category}, Text: {elem.text[:80]}...")
```

#### 1.3 파티셔닝 전략 선택

`strategy` 매개변수로 문서 특성에 맞는 처리 방법을 선택할 수 있습니다.

**전략 1: fast (빠른 처리)**

단순 텍스트 추출이 가능한 문서에 적합합니다.

```python
from unstructured.partition.pdf import partition_pdf

# fast 전략: 빠른 텍스트 추출
elements_fast = partition_pdf(
    filename="data/transformer.pdf",
    strategy="fast"
)

print(f"Fast 전략 - 요소 수: {len(elements_fast)}")
print(f"첫 번째 요소: {elements_fast[0].category}")
print(f"내용: {elements_fast[0].text[:150]}...")
```

**장점**:
- ✅ 빠른 처리 속도
- ✅ 단순한 텍스트 문서에 효과적

**단점**:
- ❌ 복잡한 레이아웃 처리 제한
- ❌ 표나 이미지 감지 불완전

**전략 2: hi_res (고해상도 처리)**

복잡한 레이아웃, 표, 이미지가 있는 문서에 적합합니다.

```python
# hi_res 전략: 고품질 문서 분석
elements_hires = partition_pdf(
    filename="data/transformer.pdf",
    strategy="hi_res",
    include_page_breaks=True  # 페이지 구분 포함
)

print(f"Hi-res 전략 - 요소 수: {len(elements_hires)}")

# 요소 타입 분포 확인
from collections import Counter
types_count = Counter([elem.category for elem in elements_hires])
print("\n요소 타입 분포:")
for elem_type, count in types_count.most_common():
    print(f"  - {elem_type}: {count}개")
```

**실행 결과 예시**:
```
Hi-res 전략 - 요소 수: 178

요소 타입 분포:
  - NarrativeText: 89개
  - Title: 45개
  - Table: 12개
  - ListItem: 18개
  - PageBreak: 14개
```

**장점**:
- ✅ 표 감지 및 구조화
- ✅ 복잡한 레이아웃 처리
- ✅ 이미지 메타데이터 추출

**단점**:
- ❌ 처리 시간 증가

**전략 3: ocr_only (OCR 전용)**

스캔된 문서나 이미지 기반 PDF에 적합합니다.

```python
# ocr_only 전략: 이미지 텍스트 인식
elements_ocr = partition_pdf(
    filename="data/transformer.pdf",
    strategy="ocr_only",
    include_page_breaks=True
)

print(f"OCR 전략 - 요소 수: {len(elements_ocr)}")
print(f"첫 번째 요소: {elements_ocr[0].text[:100]}...")
```

**적용 사례**:
- ✅ 스캔된 문서
- ✅ 이미지로 저장된 PDF
- ✅ 텍스트 레이어가 없는 문서

**단점**:
- ❌ OCR 정확도에 의존
- ❌ 처리 시간이 가장 김

#### 전략 비교 예시

```python
import time

strategies = ["fast", "hi_res", "ocr_only"]
results = []

for strategy in strategies:
    start_time = time.time()

    elements = partition_pdf(
        filename="data/transformer.pdf",
        strategy=strategy
    )

    elapsed_time = time.time() - start_time

    results.append({
        "전략": strategy,
        "요소 수": len(elements),
        "처리 시간": f"{elapsed_time:.2f}초"
    })

# 결과 비교
import pandas as pd
df = pd.DataFrame(results)
print(df.to_string(index=False))
```

**실행 결과 예시**:
```
    전략  요소 수  처리 시간
    fast     156     2.34초
  hi_res     178    15.67초
ocr_only     145    45.23초
```

### 2. 청킹 (Chunking)

청킹은 문서를 RAG 시스템에 최적화된 크기의 청크로 나누는 과정입니다.

#### 2.1 기본 청킹 전략

```python
from unstructured.partition.auto import partition
from unstructured.chunking.basic import chunk_elements

# 1. 문서 파티셔닝
elements = partition(filename="data/transformer.pdf")
print(f"파티션 요소 수: {len(elements)}")

# 2. 청킹 수행
chunks = chunk_elements(
    elements,
    max_characters=500,        # 하드 최대 크기
    new_after_n_chars=400,     # 소프트 최대 크기
    overlap=100,               # 청크 간 중복 문자 수
    overlap_all=True           # 모든 청크 간 중복 적용
)

print(f"청크 수: {len(chunks)}")

# 청크 길이 분포 확인
chunk_lengths = [len(chunk.text) for chunk in chunks]
import pandas as pd
print("\n청크 길이 통계:")
print(pd.Series(chunk_lengths).describe())
```

**실행 결과 예시**:
```
파티션 요소 수: 156
청크 수: 87

청크 길이 통계:
count     87.000000
mean     412.563218
std       78.234567
min      234.000000
25%      375.000000
50%      405.000000
75%      485.000000
max      500.000000
```

#### 청킹 매개변수 설명

**`max_characters`** (하드 최대값):
- 청크의 절대 최대 크기
- 이 값을 초과하면 강제로 분할
- 단일 요소가 이 값을 초과하면 텍스트 분할

**`new_after_n_chars`** (소프트 최대값):
- 새 청크 생성을 시작하는 기준
- 이 값 이후에 요소 경계에서 청크 분할
- `max_characters`보다 작아야 함

**`overlap`**:
- 연속된 청크 간 중복 문자 수
- 문맥 연속성 유지에 도움
- 일반적으로 10-20% 중복 권장

**`overlap_all`**:
- `True`: 모든 청크 간 중복 적용
- `False`: 특정 조건에서만 중복

#### 청크 내용 확인

```python
# 첫 번째 청크 분석
first_chunk = chunks[0]

print("첫 번째 청크 정보:")
print(f"  - 길이: {len(first_chunk.text)}자")
print(f"  - 타입: {first_chunk.category}")
print(f"  - 내용:\n{first_chunk.text}\n")

# 두 번째 청크와 중복 확인
second_chunk = chunks[1]
print("두 번째 청크 정보:")
print(f"  - 길이: {len(second_chunk.text)}자")
print(f"  - 내용:\n{second_chunk.text}\n")

# 중복 부분 확인
overlap_text = first_chunk.text[-100:]
if overlap_text in second_chunk.text:
    print("✅ 청크 간 중복 확인됨 (문맥 연속성 유지)")
```

#### 2.2 청킹 전략 비교

다양한 청킹 전략을 비교하여 최적의 설정을 찾습니다.

```python
from unstructured.chunking.basic import chunk_elements
import pandas as pd

# 파티셔닝 (한 번만 수행)
elements = partition(filename="data/transformer.pdf")

# 청킹 전략 정의
strategies = [
    {
        "name": "소형 청크 (빠른 검색)",
        "params": {
            "max_characters": 300,
            "new_after_n_chars": 250,
            "overlap": 50,
            "overlap_all": True
        }
    },
    {
        "name": "중형 청크 (균형형)",
        "params": {
            "max_characters": 800,
            "new_after_n_chars": 600,
            "overlap": 150,
            "overlap_all": True
        }
    },
    {
        "name": "대형 청크 (문맥 보존)",
        "params": {
            "max_characters": 1500,
            "new_after_n_chars": 1200,
            "overlap": 300,
            "overlap_all": False
        }
    }
]

# 각 전략 실행 및 분석
results = []

for strategy in strategies:
    chunks = chunk_elements(elements, **strategy['params'])
    chunk_lengths = [len(chunk.text) for chunk in chunks]

    results.append({
        "전략": strategy['name'],
        "청크 수": len(chunks),
        "평균 길이": f"{pd.Series(chunk_lengths).mean():.0f}",
        "표준편차": f"{pd.Series(chunk_lengths).std():.0f}",
        "최소": min(chunk_lengths),
        "최대": max(chunk_lengths)
    })

# 결과 비교
df = pd.DataFrame(results)
print("="*80)
print("청킹 전략 비교")
print("="*80)
print(df.to_string(index=False))

print("\n💡 전략별 권장 사용 사례:")
print("""
소형 청크:
  • 빠른 검색과 정확한 매칭
  • 짧은 질의응답(QA) 시스템
  • 키워드 기반 검색

중형 청크:
  • 균형 잡힌 검색과 문맥 이해
  • 일반적인 RAG 시스템 (권장)
  • 대부분의 사용 사례

대형 청크:
  • 문맥이 중요한 복잡한 질문
  • 긴 설명이 필요한 경우
  • 문서 요약 작업
""")
```

### 3. 정제 (Cleaning)

정제는 NLP 모델 사용 전에 텍스트 품질을 향상시키는 전처리 과정입니다.

#### 3.1 기본 정제 함수

`clean` 함수로 여러 정제 옵션을 한 번에 적용할 수 있습니다.

```python
from unstructured.cleaners.core import clean

# 여러 정제 옵션 동시 적용
original_text = "● An excellent ---   \n\n point."

cleaned_text = clean(
    original_text,
    bullets=True,              # 글머리 기호 제거
    extra_whitespace=True,     # 추가 공백 제거
    dashes=True,               # 대시 제거
    trailing_punctuation=True, # 후행 문장부호 제거
    lowercase=True             # 소문자 변환
)

print(f"원본: '{original_text}'")
print(f"정제: '{cleaned_text}'")
```

**실행 결과**:
```
원본: '● An excellent ---

 point.'
정제: 'an excellent point'
```

#### 3.2 특정 요소 정제

개별 정제 함수를 사용하여 특정 요소만 처리할 수 있습니다.

**글머리 기호 제거**:

```python
from unstructured.cleaners.core import clean_bullets

text = clean_bullets("● An excellent point!")
print(text)  # "An excellent point!"
```

**추가 공백 제거**:

```python
from unstructured.cleaners.core import clean_extra_whitespace

text = clean_extra_whitespace("ITEM 1A:     RISK-FACTORS")
print(text)  # "ITEM 1A: RISK-FACTORS"
```

**대시 제거**:

```python
from unstructured.cleaners.core import clean_dashes

text = clean_dashes("ITEM 1A: RISK-FACTORS–")
print(text)  # "ITEM 1A: RISK FACTORS"
```

**문장부호 제거**:

```python
from unstructured.cleaners.core import remove_punctuation

text = remove_punctuation('"A lovely quote!"')
print(text)  # "A lovely quote"
```

#### 3.3 문단 재구성

줄바꿈으로 분리된 문단을 하나로 결합합니다.

```python
from unstructured.cleaners.core import group_broken_paragraphs

text = """The big brown fox
was walking down the lane.

At the end of the lane, the
fox met a bear."""

grouped_text = group_broken_paragraphs(text)
print(grouped_text)
```

**실행 결과**:
```
The big brown fox was walking down the lane.

At the end of the lane, the fox met a bear.
```

#### 3.4 유니코드 문자 처리

**유니코드 따옴표 변환**:

```python
from unstructured.cleaners.core import replace_unicode_quotes

text = replace_unicode_quotes("\x93A lovely quote!\x94")
print(text)  # '"A lovely quote!"'
```

**비 ASCII 문자 제거**:

```python
from unstructured.cleaners.core import clean_non_ascii_chars

text = clean_non_ascii_chars("\x88This text contains ®non-ascii characters!●")
print(text)  # "This text contains non-ascii characters!"
```

#### 3.5 번역 기능

```python
from unstructured.cleaners.translate import translate_text

# 자동 언어 감지 및 영어로 번역
korean_text = "저는 대한민국 서울에 살고 있어요!"
english_text = translate_text(korean_text)
print(f"한글: {korean_text}")
print(f"영어: {english_text}")

# 특정 언어 지정
english_text = translate_text(
    korean_text,
    source_lang="ko",
    target_lang="en"
)
print(f"번역: {english_text}")
```

#### 3.6 Element 객체에 정제 적용

```python
from unstructured.documents.elements import Text
from unstructured.cleaners.core import replace_unicode_quotes

# Element 생성
text_element = Text(text="Philadelphia Eagles' victory")

# 정제 적용
text_element.apply(replace_unicode_quotes)

print(f"정제된 텍스트: {text_element.text}")
```

#### 3.7 통합 정제 예시

```python
from unstructured.cleaners.core import (
    clean, clean_bullets, clean_extra_whitespace,
    clean_dashes, group_broken_paragraphs
)

# 복잡한 텍스트 예시
text = """● 첫 번째 항목
    ■ 중첩된 항목

텍스트는    여러 개의     공백을 포함합니다.
또한 — 여러 종류의 ― 대시도 포함됩니다."""

print("원본 텍스트:")
print(text)
print("\n" + "="*80 + "\n")

# 통합 정제
cleaned = clean(
    text,
    bullets=True,
    extra_whitespace=True,
    dashes=True,
    trailing_punctuation=False,
    lowercase=False
)

print("정제된 텍스트:")
print(cleaned)
```

### 4. RAG 구현

Unstructured의 세 가지 기능(파티셔닝, 청킹, 정제)을 통합하여 완전한 RAG 시스템을 구축합니다.

#### 4.1 문서 전처리

```python
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from langchain_core.documents import Document

# 1. 문서 파티셔닝
print("📄 1단계: 문서 파티셔닝")
elements = partition("data/transformer.pdf")
print(f"   파티션 요소: {len(elements)}개")

# 2. 청킹 (제목 기반)
print("\n✂️ 2단계: 청킹")
chunks = chunk_by_title(elements)
print(f"   청크: {len(chunks)}개")

# 3. LangChain Document 변환
print("\n🔄 3단계: LangChain Document 변환")
documents = []
for chunk in chunks:
    # 필요한 메타데이터만 추출
    metadata = {}
    for key, value in chunk.metadata.to_dict().items():
        if key in ["filename", "page_number"]:
            metadata[key] = value

    documents.append(
        Document(page_content=chunk.text, metadata=metadata)
    )

print(f"   문서: {len(documents)}개")

# 처리 결과 확인
print("\n📊 처리 결과:")
print(f"   파티션 → 청킹 → 문서: {len(elements)} → {len(chunks)} → {len(documents)}")
```

**실행 결과 예시**:
```
📄 1단계: 문서 파티셔닝
   파티션 요소: 156개

✂️ 2단계: 청킹
   청크: 45개

🔄 3단계: LangChain Document 변환
   문서: 45개

📊 처리 결과:
   파티션 → 청킹 → 문서: 156 → 45 → 45
```

#### 4.2 문서 인덱싱

```python
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 임베딩 모델 설정
print("🤖 임베딩 모델 로드 중...")
embeddings = HuggingFaceEmbeddings(
    model_name="Alibaba-NLP/gte-multilingual-base",
    model_kwargs={"trust_remote_code": True}
)
print("   ✅ 로드 완료")

# 벡터 스토어 생성
print("\n💾 벡터 스토어 생성 중...")
db = FAISS.from_documents(documents, embeddings)
print("   ✅ 생성 완료")

# Retriever 설정
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}  # 상위 4개 문서 검색
)

# 검색 테스트
print("\n🔍 검색 테스트:")
query = "Explain the working principle of transformer"
print(f"   Query: {query}")

results = retriever.invoke(query)
print(f"   검색 결과: {len(results)}개 문서\n")

# 결과 출력
for i, doc in enumerate(results, 1):
    print(f"[{i}] {doc.metadata}")
    print(f"    {doc.page_content[:150]}...\n")
```

**실행 결과 예시**:
```
🤖 임베딩 모델 로드 중...
   ✅ 로드 완료

💾 벡터 스토어 생성 중...
   ✅ 생성 완료

🔍 검색 테스트:
   Query: Explain the working principle of transformer
   검색 결과: 4개 문서

[1] {'filename': 'transformer.pdf', 'page_number': 1}
    The Transformer is a model architecture eschewing recurrence and instead relying entirely on an attention mechanism...

[2] {'filename': 'transformer.pdf', 'page_number': 3}
    The attention mechanism allows the model to focus on different parts of the input sequence...
```

#### 4.3 RAG 체인 구성

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# LLM 설정
print("🤖 LLM 설정 중...")
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    top_p=0.7,
)
print("   ✅ 설정 완료")

# 프롬프트 템플릿
system_prompt = """당신은 인공지능 전문가입니다. 사용자의 질문에 대한 답변을 작성해야 합니다.

[작성 요령]
- 질문에 대한 답변을 간결하게 작성하세요.
- 가능한 한 사용자가 이해할 수 있도록 쉽게 설명하세요.
- 전문용어는 원어를 사용하고, 한국어를 함께 표기하세요. (예: Transformer(트랜스포머))
"""

human_prompt = """반드시 주어진 문맥에 근거하여 답변해주세요.
근거가 부족하면 답변을 거부합니다.
출처를 반드시 표시합니다. (출처: [문서 제목], [페이지 번호])

[문맥]
{context}

[질문]
{question}

이 질문에 대한 답변을 작성해주세요."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_prompt)
])

# 문서 포맷 함수
def format_document(doc):
    """단일 문서를 포맷팅"""
    return f"**{doc.metadata['filename']} /page {doc.metadata['page_number']}**\n\n{doc.page_content}"

def format_documents(docs):
    """여러 문서를 포맷팅"""
    return "\n\n".join([format_document(doc) for doc in docs])

# LCEL 체인 생성
chain = (
    {
        "context": retriever | format_documents,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("\n✅ RAG 체인 구성 완료")
```

#### 4.4 RAG 시스템 테스트

```python
# 테스트 질문
test_queries = [
    "트랜스포머 모델의 작동 원리를 설명해주세요.",
    "What is the attention mechanism?",
    "트랜스포머의 주요 장점은 무엇인가요?"
]

print("="*80)
print("RAG 시스템 테스트")
print("="*80)

for i, query in enumerate(test_queries, 1):
    print(f"\n[질문 {i}] {query}")
    print("-"*80)

    # 답변 생성
    response = chain.invoke(query)
    print(f"\n{response}\n")
```

**실행 결과 예시**:
```
================================================================================
RAG 시스템 테스트
================================================================================

[질문 1] 트랜스포머 모델의 작동 원리를 설명해주세요.
--------------------------------------------------------------------------------

트랜스포머(Transformer) 모델은 순환 신경망(RNN)이나 합성곱 신경망(CNN)을
사용하지 않고, 전적으로 어텐션 메커니즘(Attention Mechanism)에 의존하는
모델 아키텍처입니다.

주요 작동 원리:

1. **Self-Attention**: 입력 시퀀스의 각 위치가 다른 모든 위치와의 관계를
   계산하여 문맥을 파악합니다.

2. **Multi-Head Attention**: 여러 개의 어텐션 헤드를 병렬로 사용하여
   다양한 표현 정보를 학습합니다.

3. **Position Encoding**: 순환 구조가 없기 때문에 위치 정보를
   명시적으로 추가합니다.

출처: transformer.pdf, page 1-3
```

## 🎯 실습 문제

### 실습 1: PDF 문서 파티셔닝 전략 비교

**문제**: 두 개의 PDF 문서에 대해 서로 다른 파티셔닝 전략을 적용하고 결과를 비교 분석하세요.

**데이터**:
- `data/리비안_KR_without_table.pdf` (표 없는 문서)
- `data/리비안_KR_with_table.pdf` (표 포함 문서)

**요구사항**:
1. 각 문서에 `fast`와 `hi_res` 전략을 모두 적용
2. 요소 타입별 개수 비교
3. 처리 시간 측정
4. 표 포함 문서에서 Table 요소 감지 여부 확인

**힌트**:
```python
from unstructured.partition.pdf import partition_pdf
from collections import Counter
import time

strategies = ["fast", "hi_res"]
pdf_files = [
    "data/리비안_KR_without_table.pdf",
    "data/리비안_KR_with_table.pdf"
]

# 각 조합 테스트
for pdf_file in pdf_files:
    for strategy in strategies:
        # 시간 측정 및 파티셔닝
        # 요소 타입 분석
        # 결과 출력
```

### 실습 2: 청킹 전략 최적화

**문제**: 다양한 청킹 전략을 실험하여 RAG 시스템에 최적인 설정을 찾으세요.

**데이터**: `data/리비안_KR_without_table.pdf`

**요구사항**:
1. 최소 3가지 다른 청킹 전략 정의
2. 각 전략의 청크 수, 평균 길이, 표준편차 계산
3. 첫 번째 청크 샘플 출력
4. 권장 사용 사례 제시

**평가 기준**:
- 소형 청크 (< 400자)
- 중형 청크 (400-1000자)
- 대형 청크 (> 1000자)

### 실습 3: 텍스트 정제 파이프라인

**문제**: 복잡한 텍스트를 여러 정제 방법으로 처리하고 결과를 비교하세요.

**샘플 텍스트**:
```python
text = """● 첫 번째 항목
    ■ 중첩된 항목

텍스트는    여러 개의     공백을 포함합니다.
또한 — 여러 종류의 ― 대시도 포함됩니다."""
```

**요구사항**:
1. 개별 정제 함수 적용 (bullets, whitespace, dashes)
2. 조합 적용 (bullets + whitespace, 모두 적용)
3. 통합 `clean` 함수 사용
4. 각 방법의 결과 길이 비교
5. 권장 방법 제시

### 실습 4: 완전한 RAG 시스템 구축

**문제**: Transformer 논문 PDF로 완전한 RAG 시스템을 구축하고 테스트하세요.

**데이터**: `data/transformer.pdf`

**요구사항**:
1. 파티셔닝 → 청킹 → 문서 변환 파이프라인 구현
2. 벡터 스토어 생성 및 인덱싱
3. RAG 체인 구성
4. 검색 단계와 생성 단계 분리 테스트
5. 3개 이상의 질문으로 시스템 평가

**테스트 질문 예시**:
- "Transformer의 작동 원리는?"
- "Attention mechanism이란?"
- "Transformer의 장점은?"

## ✅ 솔루션 예시

### 솔루션 1: PDF 문서 파티셔닝 전략 비교

```python
from unstructured.partition.pdf import partition_pdf
from collections import Counter
import time
import pandas as pd

print("="*100)
print("PDF 파티셔닝 전략 비교")
print("="*100)

pdf_files = [
    "data/리비안_KR_without_table.pdf",
    "data/리비안_KR_with_table.pdf"
]

strategies = ["fast", "hi_res"]
all_results = []

for pdf_file in pdf_files:
    print(f"\n{'='*100}")
    print(f"📄 파일: {pdf_file}")
    print(f"{'='*100}")

    for strategy in strategies:
        print(f"\n  🔧 전략: {strategy}")
        print(f"  {'-'*96}")

        # 시간 측정 시작
        start_time = time.time()

        # 파티셔닝 수행
        elements = partition_pdf(
            filename=pdf_file,
            strategy=strategy,
            include_page_breaks=True
        )

        # 처리 시간 계산
        elapsed_time = time.time() - start_time

        # 요소 타입 분석
        element_types = Counter([elem.category for elem in elements])

        # 결과 출력
        print(f"  - 총 요소: {len(elements)}개")
        print(f"  - 처리 시간: {elapsed_time:.2f}초")
        print(f"  - 요소 타입 분포:")
        for elem_type, count in element_types.most_common():
            print(f"    • {elem_type}: {count}개")

        # Table 요소 확인
        table_count = element_types.get('Table', 0)
        if 'table' in pdf_file.lower():
            if table_count > 0:
                print(f"  ✅ 표 감지: {table_count}개")
            else:
                print(f"  ❌ 표 감지 실패")

        # 결과 저장
        all_results.append({
            "파일": pdf_file.split('/')[-1],
            "전략": strategy,
            "요소 수": len(elements),
            "처리 시간": f"{elapsed_time:.2f}s",
            "Table 개수": table_count,
            "주요 타입": element_types.most_common(1)[0][0] if element_types else "N/A"
        })

# 전체 비교 표
print(f"\n{'='*100}")
print("📊 전체 비교")
print(f"{'='*100}")
df = pd.DataFrame(all_results)
print(df.to_string(index=False))

print(f"\n{'='*100}")
print("💡 분석 결과")
print(f"{'='*100}")
print("""
✅ fast 전략:
  • 빠른 처리 속도
  • 단순 텍스트 문서에 적합
  • 표 감지 제한적

✅ hi_res 전략:
  • 고품질 문서 분석
  • 표와 복잡한 레이아웃 처리
  • 처리 시간이 더 소요

📌 권장사항:
  • 표 없는 문서 → fast 전략
  • 표 포함 문서 → hi_res 전략
""")
```

### 솔루션 2: 청킹 전략 최적화

```python
from unstructured.partition.auto import partition
from unstructured.chunking.basic import chunk_elements
import pandas as pd

print("="*100)
print("청킹 전략 최적화")
print("="*100)

# 파티셔닝
elements = partition(filename="data/리비안_KR_without_table.pdf")
print(f"\n✅ 파티셔닝 완료: {len(elements)}개 요소\n")

# 청킹 전략 정의
strategies = [
    {
        "name": "소형 청크 (빠른 검색)",
        "params": {
            "max_characters": 300,
            "new_after_n_chars": 250,
            "overlap": 50,
            "overlap_all": True
        },
        "use_case": "키워드 기반 검색, 짧은 QA"
    },
    {
        "name": "중형 청크 (균형형)",
        "params": {
            "max_characters": 800,
            "new_after_n_chars": 600,
            "overlap": 150,
            "overlap_all": True
        },
        "use_case": "일반 RAG 시스템, 대부분의 경우"
    },
    {
        "name": "대형 청크 (문맥 보존)",
        "params": {
            "max_characters": 1500,
            "new_after_n_chars": 1200,
            "overlap": 300,
            "overlap_all": False
        },
        "use_case": "복잡한 질문, 문서 요약"
    }
]

results = []

for strategy in strategies:
    print(f"{'='*100}")
    print(f"🔧 {strategy['name']}")
    print(f"{'='*100}")
    print(f"설정:")
    for key, value in strategy['params'].items():
        print(f"  - {key}: {value}")

    # 청킹 수행
    chunks = chunk_elements(elements, **strategy['params'])

    # 통계 계산
    chunk_lengths = [len(chunk.text) for chunk in chunks]
    stats = pd.Series(chunk_lengths)

    print(f"\n📊 결과:")
    print(f"  - 청크 수: {len(chunks)}개")
    print(f"  - 평균 길이: {stats.mean():.1f}자")
    print(f"  - 표준편차: {stats.std():.1f}자")
    print(f"  - 최소 길이: {stats.min()}자")
    print(f"  - 최대 길이: {stats.max()}자")
    print(f"  - 중간값: {stats.median():.1f}자")

    # 첫 번째 청크 샘플
    if chunks:
        print(f"\n📝 첫 번째 청크 (200자):")
        print(f"  {chunks[0].text[:200]}...")

    print(f"\n💡 권장 사용 사례:")
    print(f"  {strategy['use_case']}")

    # 결과 저장
    results.append({
        "전략": strategy['name'],
        "청크 수": len(chunks),
        "평균 길이": f"{stats.mean():.0f}자",
        "표준편차": f"{stats.std():.0f}자",
        "최소/최대": f"{stats.min()}/{stats.max()}자"
    })

# 비교 표
print(f"\n{'='*100}")
print("📊 전략 비교 요약")
print(f"{'='*100}")
df = pd.DataFrame(results)
print(df.to_string(index=False))
```

### 솔루션 3: 텍스트 정제 파이프라인

```python
from unstructured.cleaners.core import (
    clean, clean_bullets, clean_extra_whitespace,
    clean_dashes, group_broken_paragraphs
)

text = """● 첫 번째 항목
    ■ 중첩된 항목

텍스트는    여러 개의     공백을 포함합니다.
또한 — 여러 종류의 ― 대시도 포함됩니다."""

print("="*100)
print("텍스트 정제 파이프라인")
print("="*100)

print("\n📝 원본 텍스트:")
print("-"*100)
print(text)
print(f"길이: {len(text)}자")

# 정제 방법 정의
methods = [
    {
        "name": "1. 글머리 기호 제거",
        "function": lambda t: clean_bullets(t),
    },
    {
        "name": "2. 추가 공백 제거",
        "function": lambda t: clean_extra_whitespace(t),
    },
    {
        "name": "3. 대시 제거",
        "function": lambda t: clean_dashes(t),
    },
    {
        "name": "4. 글머리 + 공백",
        "function": lambda t: clean_extra_whitespace(clean_bullets(t)),
    },
    {
        "name": "5. 글머리 + 공백 + 대시",
        "function": lambda t: clean_dashes(clean_extra_whitespace(clean_bullets(t))),
    },
    {
        "name": "6. 통합 clean 함수",
        "function": lambda t: clean(t, bullets=True, extra_whitespace=True, dashes=True),
    },
    {
        "name": "7. 통합 + 문단 재구성",
        "function": lambda t: group_broken_paragraphs(
            clean(t, bullets=True, extra_whitespace=True, dashes=True)
        ),
    }
]

results = []

for method in methods:
    print(f"\n{'='*100}")
    print(f"🔧 {method['name']}")
    print(f"{'='*100}")

    result_text = method['function'](text)

    print(f"결과:")
    print(f"{result_text}")
    print(f"\n길이: {len(result_text)}자 (원본 대비 {len(result_text) - len(text):+d}자)")

    results.append({
        "방법": method['name'],
        "결과 길이": len(result_text),
        "길이 변화": f"{len(result_text) - len(text):+d}자"
    })

# 비교 표
print(f"\n{'='*100}")
print("📊 정제 방법 비교")
print(f"{'='*100}")
import pandas as pd
df = pd.DataFrame(results)
print(df.to_string(index=False))

print(f"\n{'='*100}")
print("💡 권장사항")
print(f"{'='*100}")
print("""
✅ 일반적인 경우:
  • 방법 6: 통합 clean 함수
  • 한 번에 여러 정제 옵션 적용

✅ 문단 구조 중요한 경우:
  • 방법 7: 통합 + 문단 재구성
  • 줄바꿈 정리 포함

✅ 특정 요소만 제거:
  • 방법 1-3: 개별 함수 사용
  • 정밀한 제어 가능
""")
```

### 솔루션 4: 완전한 RAG 시스템 구축

```python
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from langchain_core.documents import Document
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

print("="*100)
print("완전한 RAG 시스템 구축")
print("="*100)

# ====================
# 1단계: 문서 전처리
# ====================
print("\n📚 1단계: 문서 전처리")
print("-"*100)

# 파티셔닝
elements = partition("data/transformer.pdf")
print(f"✅ 파티셔닝: {len(elements)}개 요소")

# 청킹
chunks = chunk_by_title(elements)
print(f"✅ 청킹: {len(chunks)}개 청크")

# LangChain Document 변환
documents = []
for chunk in chunks:
    metadata = {}
    for key, value in chunk.metadata.to_dict().items():
        if key in ["filename", "page_number"]:
            metadata[key] = value

    documents.append(
        Document(page_content=chunk.text, metadata=metadata)
    )

print(f"✅ 문서 변환: {len(documents)}개 문서")

# ====================
# 2단계: 벡터 스토어 생성
# ====================
print("\n💾 2단계: 벡터 스토어 생성")
print("-"*100)

# 임베딩 모델
embeddings = HuggingFaceEmbeddings(
    model_name="Alibaba-NLP/gte-multilingual-base",
    model_kwargs={"trust_remote_code": True}
)
print("✅ 임베딩 모델 로드 완료")

# 벡터 스토어
db = FAISS.from_documents(documents, embeddings)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
print("✅ 벡터 스토어 생성 완료")

# ====================
# 3단계: 검색 테스트
# ====================
print("\n🔍 3단계: 검색 단계 테스트")
print("-"*100)

test_query = "Explain the working principle of transformer"
print(f"Query: {test_query}\n")

retrieved_docs = retriever.invoke(test_query)
print(f"✅ 검색 결과: {len(retrieved_docs)}개 문서")

for i, doc in enumerate(retrieved_docs, 1):
    print(f"\n[문서 {i}]")
    print(f"  파일: {doc.metadata.get('filename', 'N/A')}")
    print(f"  페이지: {doc.metadata.get('page_number', 'N/A')}")
    print(f"  내용: {doc.page_content[:120]}...")

# ====================
# 4단계: RAG 체인 구성
# ====================
print("\n\n🤖 4단계: RAG 체인 구성")
print("-"*100)

# LLM 설정
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, top_p=0.7)
print("✅ LLM 설정 완료")

# 프롬프트 템플릿
system_prompt = """당신은 인공지능 전문가입니다. 사용자의 질문에 대한 답변을 작성해야 합니다.

[작성 요령]
- 질문에 대한 답변을 간결하게 작성하세요.
- 가능한 한 사용자가 이해할 수 있도록 쉽게 설명하세요.
- 전문용어는 원어를 사용하고, 한국어를 함께 표기하세요. (예: Transformer(트랜스포머))
"""

human_prompt = """반드시 주어진 문맥에 근거하여 답변해주세요.
근거가 부족하면 답변을 거부합니다.
출처를 반드시 표시합니다. (출처: [문서 제목], [페이지 번호])

[문맥]
{context}

[질문]
{question}

이 질문에 대한 답변을 작성해주세요."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_prompt)
])

# 문서 포맷 함수
def format_document(doc):
    return f"**{doc.metadata['filename']} /page {doc.metadata['page_number']}**\n\n{doc.page_content}"

def format_documents(docs):
    return "\n\n".join([format_document(doc) for doc in docs])

# LCEL 체인
chain = (
    {
        "context": retriever | format_documents,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("✅ RAG 체인 구성 완료")

# ====================
# 5단계: 생성 테스트
# ====================
print("\n\n💬 5단계: 생성 단계 테스트")
print("-"*100)

test_queries = [
    "트랜스포머 모델의 작동 원리를 설명해주세요.",
    "What is the attention mechanism?",
    "트랜스포머의 주요 장점은 무엇인가요?"
]

for i, query in enumerate(test_queries, 1):
    print(f"\n[질문 {i}] {query}")
    print("="*100)

    # 검색 단계
    retrieved = retriever.invoke(query)
    print(f"검색된 문서: {len(retrieved)}개")
    print(f"관련 페이지: {', '.join([str(d.metadata.get('page_number', 'N/A')) for d in retrieved])}")

    # 생성 단계
    answer = chain.invoke(query)
    print(f"\n답변:\n{answer}\n")

print("\n" + "="*100)
print("✅ RAG 시스템 구축 및 테스트 완료!")
print("="*100)
```

## 🚀 실무 활용 예시

### 예시 1: 다중 문서 처리 시스템

```python
from glob import glob
from unstructured.partition.auto import partition
from unstructured.chunking.basic import chunk_elements
from langchain_core.documents import Document

class MultiDocumentProcessor:
    """여러 문서를 처리하는 시스템"""

    def __init__(self, directory: str, file_pattern: str = "*.pdf"):
        self.directory = directory
        self.file_pattern = file_pattern
        self.all_documents = []

    def process_all_documents(self):
        """디렉토리 내 모든 문서 처리"""

        # 파일 목록 가져오기
        file_paths = glob(f"{self.directory}/{self.file_pattern}")
        print(f"📁 발견된 파일: {len(file_paths)}개\n")

        for file_path in file_paths:
            print(f"처리 중: {file_path}")

            # 파티셔닝
            elements = partition(filename=file_path)

            # 청킹
            chunks = chunk_elements(
                elements,
                max_characters=800,
                new_after_n_chars=600,
                overlap=150
            )

            # Document 변환
            for chunk in chunks:
                metadata = {
                    "source": file_path,
                    "filename": file_path.split('/')[-1]
                }

                # 추가 메타데이터
                if hasattr(chunk.metadata, 'page_number'):
                    metadata['page_number'] = chunk.metadata.page_number

                self.all_documents.append(
                    Document(page_content=chunk.text, metadata=metadata)
                )

            print(f"  ✅ 청크: {len(chunks)}개\n")

        print(f"총 {len(self.all_documents)}개 문서 생성 완료")
        return self.all_documents

# 사용 예시
processor = MultiDocumentProcessor("data/", "*.pdf")
documents = processor.process_all_documents()

# 벡터 스토어에 저장
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="Alibaba-NLP/gte-multilingual-base",
    model_kwargs={"trust_remote_code": True}
)

db = FAISS.from_documents(documents, embeddings)
print(f"\n✅ {len(documents)}개 문서를 벡터 스토어에 저장 완료")
```

### 예시 2: 문서 타입별 처리 전략

```python
class SmartDocumentProcessor:
    """문서 타입에 따라 최적 전략을 자동 선택"""

    def __init__(self):
        self.strategy_rules = {
            'simple_text': {
                'strategy': 'fast',
                'max_chars': 500,
                'overlap': 50
            },
            'with_tables': {
                'strategy': 'hi_res',
                'max_chars': 1000,
                'overlap': 150
            },
            'scanned': {
                'strategy': 'ocr_only',
                'max_chars': 800,
                'overlap': 100
            }
        }

    def detect_document_type(self, filename: str) -> str:
        """문서 타입 감지"""

        # 간단한 테스트 파티셔닝
        elements = partition(filename=filename, strategy="fast")

        # 표 포함 여부 확인
        has_tables = any(elem.category == 'Table' for elem in elements)

        # 이미지 요소 비율 확인
        image_ratio = sum(1 for elem in elements if elem.category == 'Image') / len(elements)

        if image_ratio > 0.3:
            return 'scanned'
        elif has_tables:
            return 'with_tables'
        else:
            return 'simple_text'

    def process_document(self, filename: str):
        """문서 타입에 맞게 처리"""

        # 타입 감지
        doc_type = self.detect_document_type(filename)
        print(f"문서 타입: {doc_type}")

        # 전략 선택
        config = self.strategy_rules[doc_type]

        # 파티셔닝
        from unstructured.partition.pdf import partition_pdf
        elements = partition_pdf(
            filename=filename,
            strategy=config['strategy']
        )

        # 청킹
        from unstructured.chunking.basic import chunk_elements
        chunks = chunk_elements(
            elements,
            max_characters=config['max_chars'],
            overlap=config['overlap']
        )

        return chunks

# 사용 예시
processor = SmartDocumentProcessor()
chunks = processor.process_document("data/transformer.pdf")
print(f"처리 결과: {len(chunks)}개 청크")
```

### 예시 3: 실시간 문서 파이프라인

```python
import os
from pathlib import Path

class DocumentPipeline:
    """문서 처리 파이프라인"""

    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.processed_files = set()

    def watch_directory(self, directory: str):
        """디렉토리 감시 및 새 문서 자동 처리"""

        while True:
            # 디렉토리의 모든 PDF 파일
            for file_path in Path(directory).glob("*.pdf"):
                file_path_str = str(file_path)

                # 이미 처리된 파일은 건너뛰기
                if file_path_str in self.processed_files:
                    continue

                print(f"\n새 문서 발견: {file_path_str}")

                try:
                    # 문서 처리
                    documents = self.process_single_file(file_path_str)

                    # 벡터 스토어에 추가
                    self.vector_store.add_documents(documents)

                    # 처리 완료 표시
                    self.processed_files.add(file_path_str)
                    print(f"✅ 처리 완료: {len(documents)}개 문서 추가")

                except Exception as e:
                    print(f"❌ 처리 실패: {e}")

            # 10초 대기
            import time
            time.sleep(10)

    def process_single_file(self, file_path: str):
        """단일 파일 처리"""

        # 파티셔닝
        elements = partition(filename=file_path)

        # 청킹
        from unstructured.chunking.basic import chunk_elements
        chunks = chunk_elements(
            elements,
            max_characters=800,
            new_after_n_chars=600,
            overlap=150
        )

        # Document 변환
        from langchain_core.documents import Document
        documents = []
        for chunk in chunks:
            documents.append(
                Document(
                    page_content=chunk.text,
                    metadata={"source": file_path}
                )
            )

        return documents

# 사용 예시 (백그라운드 실행)
# pipeline = DocumentPipeline(db)
# pipeline.watch_directory("data/incoming/")
```

## 📖 참고 자료

### 공식 문서

1. **Unstructured 공식 문서**
   - [Installation Guide](https://docs.unstructured.io/open-source/installation)
   - [Partitioning](https://docs.unstructured.io/open-source/core-functionality/partitioning)
   - [Chunking](https://docs.unstructured.io/open-source/core-functionality/chunking)
   - [Cleaning](https://docs.unstructured.io/open-source/core-functionality/cleaning)

2. **LangChain 공식 문서**
   - [Document Loaders](https://python.langchain.com/docs/modules/data_connection/document_loaders/)
   - [Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)

### GitHub 저장소

1. **Unstructured**
   - [unstructured-io/unstructured](https://github.com/Unstructured-IO/unstructured)

2. **개발 환경**
   - [unstructured-dev-vscode](https://github.com/tsdata/unstructured-dev-vscode)

### 추가 학습 자료

1. **RAG 패턴**
   - [Advanced RAG Techniques](https://python.langchain.com/docs/use_cases/question_answering/)
   - [Vector Store Best Practices](https://python.langchain.com/docs/modules/data_connection/vectorstores/)

2. **문서 처리 최적화**
   - PDF Layout Analysis
   - Table Detection and Extraction
   - OCR Best Practices

---

**학습 완료** ✅

Unstructured 라이브러리를 활용한 비정형 문서 처리의 전체 파이프라인을 학습했습니다. 이제 실제 프로젝트에서 다양한 형식의 문서를 처리하고 고품질 RAG 시스템을 구축할 수 있습니다!
