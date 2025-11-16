# Unstructured Partitioning Part1 - PDF/이미지 문서 파티셔닝

## 📚 학습 목표
- 파티셔닝(Partitioning)의 개념과 문서 처리에서의 역할을 이해한다
- PDF와 이미지 문서에서 구조화된 요소(Title, Text, Table 등)를 추출할 수 있다
- 4가지 파티셔닝 전략(auto, fast, hi_res, ocr_only)의 특징과 차이점을 이해한다
- 문서 특성에 따라 적절한 파티셔닝 전략을 선택할 수 있다
- 추출한 요소의 메타데이터를 분석하고 활용할 수 있다

## 🔑 핵심 개념

### 파티셔닝(Partitioning)이란?
**파티셔닝**은 비정형 문서를 **Title**, **NarrativeText**, **ListItem**, **Table** 등 **구조화된 요소**로 분할하는 과정입니다. 문서의 각 부분을 의미 있는 단위로 구분하여 데이터 활용도를 향상시킵니다.

### 파티셔닝의 장점
- **구조화**: 비정형 문서를 체계적인 요소로 분류
- **선택적 추출**: 필요한 콘텐츠만 선택적으로 추출 가능
- **메타데이터**: 각 요소의 위치, 페이지, 카테고리 정보 제공
- **RAG 최적화**: 검색 증강 생성 시스템에 적합한 청크 생성

### PDF/이미지 문서 파티셔닝 전략 비교

| 전략 | 주요 기술 | 속도 | 정확도 | 적합한 문서 | 주요 특징 |
|------|----------|------|--------|------------|----------|
| **auto** | 자동 선택 | 중간 | 중간 | 모든 문서 | 문서 특성에 따라 fast/ocr_only 자동 선택 |
| **fast** | pdfminer | 매우 빠름 | 중간 | 텍스트 PDF | PDF에서 직접 텍스트 추출, 최고 속도 |
| **hi_res** | 객체 탐지 모델 | 느림 | 매우 높음 | 복잡한 레이아웃 | 레이아웃 분석, 테이블 구조 인식 |
| **ocr_only** | Tesseract OCR | 느림 | 높음 | 스캔 이미지, 다중 열 | 이미지 기반 텍스트 인식 |

### 문서 요소 타입
- **Title**: 제목, 헤더
- **NarrativeText**: 본문 텍스트
- **ListItem**: 목록 항목
- **Table**: 테이블
- **Image**: 이미지
- **Footer**: 푸터
- **Header**: 헤더
- **PageBreak**: 페이지 나누기

### 관련 기술 스택
- **Unstructured**: 문서 파티셔닝 라이브러리
- **pdfminer**: PDF 텍스트 추출
- **Tesseract OCR**: 광학 문자 인식
- **Poppler**: PDF 렌더링 및 이미지 변환
- **객체 탐지 모델**: 레이아웃 분석 (YOLOX, Detectron2 등)

## 🛠 환경 설정

### 1. 라이브러리 설치

```bash
# 기본 라이브러리 설치
pip install unstructured

# PDF 처리 라이브러리
pip install "unstructured[pdf]"

# 전체 기능 설치 (권장)
pip install "unstructured[all-docs]"
```

### 2. Poppler와 Tesseract 설치

**Poppler** (hi_res, ocr_only 전략 사용 시 필요):
- **Windows**: https://github.com/oschwartz10612/poppler-windows/releases/ 에서 다운로드 후 PATH 설정
- **macOS**: `brew install poppler`
- **Linux**: `sudo apt-get install poppler-utils`

**Tesseract OCR** (ocr_only 전략 사용 시 필요):
- **Windows**: https://github.com/UB-Mannheim/tesseract/wiki 에서 다운로드 후 PATH 설정
- **macOS**: `brew install tesseract`
- **Linux**: `sudo apt-get install tesseract-ocr`

### 3. 환경 변수 설정

`.env` 파일에 Poppler와 Tesseract 경로 설정:

```bash
# Poppler 경로 (Windows 예시)
POPPLER_PATH=C:\Program Files\poppler-24.02.0\Library\bin

# Tesseract 데이터 경로 (Windows 예시)
TESSDATA_PREFIX=C:\Program Files\Tesseract-OCR\tessdata
```

### 4. 기본 설정 코드

```python
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# Poppler와 Tesseract PATH 설정
poppler_path = os.getenv('POPPLER_PATH')
if poppler_path:
    os.environ['PATH'] = poppler_path + os.pathsep + os.environ['PATH']

tessdata_prefix = os.getenv('TESSDATA_PREFIX')
if tessdata_prefix:
    os.environ['TESSDATA_PREFIX'] = tessdata_prefix
```

### 5. 기본 라이브러리 임포트

```python
import os
from glob import glob
from pprint import pprint
import json
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')
```

## 💻 단계별 구현

### 단계 1: 기본 파티셔닝 (auto 전략)

**auto 전략**은 문서 특성에 따라 최적의 파티셔닝 방법을 자동으로 선택합니다. 텍스트 추출 가능성에 따라 **fast** 또는 **ocr_only** 전략으로 자동 전환됩니다.

```python
from unstructured.partition.pdf import partition_pdf

# auto 전략으로 PDF 파티셔닝
elements = partition_pdf(
    filename="data/transformer.pdf",
    strategy="auto",  # 기본 전략 (자동 선택)
)

print(f"총 요소 개수: {len(elements)}개")
```

**출력 예시**:
```
총 요소 개수: 78개
```

### 단계 2: 요소별 개수 확인

추출한 요소를 타입별로 분류하여 문서 구조를 파악합니다.

```python
def count_elements(elements):
    """문서 구성 요소별 개수 확인"""

    elements_by_type = {}

    for element in elements:
        if "Title" in str(type(element)):
            if "Title" not in elements_by_type:
                elements_by_type["Title"] = []
            elements_by_type["Title"].append(element)

        elif "Table" in str(type(element)):
            if "Table" not in elements_by_type:
                elements_by_type["Table"] = []
            elements_by_type["Table"].append(element)

        elif "Text" in str(type(element)):
            if "Text" not in elements_by_type:
                elements_by_type["Text"] = []
            elements_by_type["Text"].append(element)
        else:
            if "Other" not in elements_by_type:
                elements_by_type["Other"] = []
            elements_by_type["Other"].append(element)

    return elements_by_type

# 문서 구성 요소별 개수 확인
elements_by_type = count_elements(elements)

for key, value in elements_by_type.items():
    print(f"{key}: {len(value)}개")
```

**출력 예시**:
```
Title: 25개
Text: 50개
Table: 2개
Other: 1개
```

### 단계 3: 개별 요소 분석

각 요소의 타입, 텍스트 내용, 메타데이터를 확인합니다.

#### (1) Text 요소 분석

```python
# Text 타입의 요소 선택
text_el = elements_by_type["Text"][10]

# 요소의 타입 확인
print(f"타입: {type(text_el)}")

# 요소의 텍스트 내용 확인
print(f"\n텍스트 내용:\n{text_el.text}")

# 요소의 메타데이터 확인
print(f"\n메타데이터:")
print(f"  - category: {text_el.category}")
print(f"  - page_number: {text_el.metadata.page_number if hasattr(text_el.metadata, 'page_number') else 'N/A'}")

# 요소를 딕셔너리로 변환하여 전체 구조 확인
print(f"\n전체 구조:")
pprint(text_el.to_dict())
```

**출력 예시**:
```
타입: <class 'unstructured.documents.elements.NarrativeText'>

텍스트 내용:
The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks that include an encoder and a decoder.

메타데이터:
  - category: NarrativeText
  - page_number: 1

전체 구조:
{'element_id': '...',
 'metadata': {'coordinates': {...},
              'filename': 'transformer.pdf',
              'filetype': 'application/pdf',
              'page_number': 1},
 'text': 'The dominant sequence transduction models...',
 'type': 'NarrativeText'}
```

#### (2) Title 요소 분석

```python
# Title 타입의 요소 선택
title_el = elements_by_type["Title"][10]

# 요소의 타입 확인
print(f"타입: {type(title_el)}")

# 요소의 텍스트 내용 확인
print(f"\n텍스트 내용:\n{title_el.text}")

# 요소의 메타데이터 확인
print(f"\n메타데이터:")
print(f"  - category: {title_el.category}")
print(f"  - page_number: {title_el.metadata.page_number if hasattr(title_el.metadata, 'page_number') else 'N/A'}")

# 요소를 딕셔너리로 변환하여 전체 구조 확인
print(f"\n전체 구조:")
pprint(title_el.to_dict())
```

**출력 예시**:
```
타입: <class 'unstructured.documents.elements.Title'>

텍스트 내용:
3.2 Attention

메타데이터:
  - category: Title
  - page_number: 3

전체 구조:
{'element_id': '...',
 'metadata': {'coordinates': {...},
              'filename': 'transformer.pdf',
              'filetype': 'application/pdf',
              'page_number': 3},
 'text': '3.2 Attention',
 'type': 'Title'}
```

#### (3) Other 요소 분석

```python
# Other 타입의 요소 확인
other_el = elements_by_type["Other"][0]

# 요소의 타입 확인
print(f"타입: {type(other_el)}")

# 요소의 텍스트 내용 확인
print(f"\n텍스트 내용:\n{other_el.text}")

# 요소의 메타데이터 확인
print(f"\n메타데이터:")
print(f"  - category: {other_el.category}")
print(f"  - page_number: {other_el.metadata.page_number if hasattr(other_el.metadata, 'page_number') else 'N/A'}")
```

**출력 예시**:
```
타입: <class 'unstructured.documents.elements.Footer'>

텍스트 내용:
Page 1 of 15

메타데이터:
  - category: Footer
  - page_number: 1
```

### 단계 4: fast 전략 사용

**fast 전략**은 pdfminer 라이브러리를 사용하여 PDF에서 직접 텍스트를 추출합니다. 텍스트 추출이 가능한 문서에서 최고 속도를 제공합니다.

```python
from unstructured.partition.pdf import partition_pdf

# fast 전략으로 PDF 파티셔닝
fast_elements = partition_pdf(
    filename="data/transformer.pdf",
    strategy="fast",  # PDF에서 직접 텍스트 추출
)

print(f"총 요소 개수: {len(fast_elements)}개")

# 요소별 개수 확인
fast_elements_by_type = count_elements(fast_elements)

for key, value in fast_elements_by_type.items():
    print(f"{key}: {len(value)}개")
```

**출력 예시**:
```
총 요소 개수: 76개
Title: 24개
Text: 48개
Table: 0개
Other: 4개
```

**특징**:
- **빠른 속도**: pdfminer를 사용하여 PDF에서 직접 텍스트 추출
- **제한된 구조 분석**: 테이블 구조는 인식하지 못함
- **텍스트 기반 PDF에 최적**: 스캔 이미지는 처리 불가

#### Footer 제거 예제

Footer 요소를 제외하고 나머지 요소만 필터링합니다.

```python
# Footer 제외 필터링
filtered_elements = []

for element in fast_elements:
    # Footer 타입이 아닌 요소만 추가
    if "Footer" not in str(type(element)):
        filtered_elements.append(element)

print(f"원본 요소 개수: {len(fast_elements)}개")
print(f"Footer 제거 후: {len(filtered_elements)}개")

# 제거된 Footer 개수
removed_count = len(fast_elements) - len(filtered_elements)
print(f"제거된 Footer: {removed_count}개")
```

**출력 예시**:
```
원본 요소 개수: 76개
Footer 제거 후: 72개
제거된 Footer: 4개
```

### 단계 5: hi_res 전략 사용

**hi_res 전략**은 객체 탐지 모델을 활용하여 문서 레이아웃을 분석합니다. 높은 정밀도가 필요한 문서 요소 분류 작업에 적합하며, 테이블 구조를 정확하게 추출할 수 있습니다.

```python
from unstructured.partition.pdf import partition_pdf

# hi_res 전략으로 PDF 파티셔닝
hi_res_elements = partition_pdf(
    filename="data/transformer.pdf",
    strategy="hi_res",  # 레이아웃 분석 모델 사용
    infer_table_structure=True,  # 테이블 구조 인식 활성화
)

print(f"총 요소 개수: {len(hi_res_elements)}개")

# 요소별 개수 확인
hi_res_elements_by_type = count_elements(hi_res_elements)

for key, value in hi_res_elements_by_type.items():
    print(f"{key}: {len(value)}개")
```

**출력 예시**:
```
총 요소 개수: 82개
Title: 26개
Text: 52개
Table: 4개
Other: 0개
```

**특징**:
- **높은 정확도**: 객체 탐지 모델로 레이아웃 분석
- **테이블 구조 인식**: 복잡한 테이블도 정확하게 추출
- **느린 속도**: 딥러닝 모델 사용으로 처리 시간 증가
- **복잡한 레이아웃에 최적**: 학술 논문, 재무 보고서 등

#### 테이블 요소 분석

```python
# 테이블 요소가 있는지 확인
if "Table" in hi_res_elements_by_type:
    tables = hi_res_elements_by_type["Table"]
    print(f"\n테이블 개수: {len(tables)}개")

    # 첫 번째 테이블 확인
    if len(tables) > 0:
        table_el = tables[0]

        print(f"\n[첫 번째 테이블]")
        print(f"타입: {type(table_el)}")
        print(f"페이지: {table_el.metadata.page_number}")

        # 테이블 텍스트 출력
        print(f"\n테이블 텍스트:")
        print(table_el.text[:200] + "...")

        # 테이블 메타데이터에 HTML이 있는지 확인
        if hasattr(table_el.metadata, 'text_as_html'):
            print(f"\nHTML 형식:")
            print(table_el.metadata.text_as_html[:200] + "...")
```

**출력 예시**:
```
테이블 개수: 4개

[첫 번째 테이블]
타입: <class 'unstructured.documents.elements.Table'>
페이지: 5

테이블 텍스트:
Model  Encoder  Decoder  Attention  FFN  Parameters  BLEU
Transformer (base)  6  6  8-head  2048  65M  27.3
Transformer (big)  6  6  16-head  4096  213M  28.4...

HTML 형식:
<table>
<tr><th>Model</th><th>Encoder</th><th>Decoder</th><th>Attention</th><th>FFN</th><th>Parameters</th><th>BLEU</th></tr>
<tr><td>Transformer (base)</td><td>6</td><td>6</td><td>8-head</td><td>2048</td><td>65M</td><td>27.3</td></tr>...
```

#### 테이블을 마크다운으로 변환

```python
from io import StringIO
import pandas as pd

def table_to_markdown(table_element):
    """테이블 요소를 마크다운으로 변환"""

    # HTML이 있는 경우 pandas로 변환
    if hasattr(table_element.metadata, 'text_as_html'):
        try:
            # HTML을 DataFrame으로 읽기
            df = pd.read_html(StringIO(table_element.metadata.text_as_html))[0]

            # 마크다운으로 변환
            markdown = df.to_markdown(index=False)
            return markdown
        except Exception as e:
            print(f"변환 실패: {e}")
            return table_element.text
    else:
        return table_element.text

# 두 번째 테이블 변환
if len(tables) > 1:
    table_el = tables[1]

    print(f"[두 번째 테이블 - 마크다운 형식]")
    markdown = table_to_markdown(table_el)
    print(markdown)

    print(f"\n[두 번째 테이블 - DataFrame]")
    if hasattr(table_el.metadata, 'text_as_html'):
        df = pd.read_html(StringIO(table_el.metadata.text_as_html))[0]
        print(df)
```

**출력 예시**:
```
[두 번째 테이블 - 마크다운 형식]
| Model              | Encoder | Decoder | Attention | FFN  | Parameters | BLEU |
|:-------------------|:--------|:--------|:----------|:-----|:-----------|:-----|
| Transformer (base) | 6       | 6       | 8-head    | 2048 | 65M        | 27.3 |
| Transformer (big)  | 6       | 6       | 16-head   | 4096 | 213M       | 28.4 |

[두 번째 테이블 - DataFrame]
              Model  Encoder  Decoder  Attention   FFN Parameters  BLEU
0  Transformer (base)        6        6     8-head  2048        65M  27.3
1   Transformer (big)        6        6    16-head  4096       213M  28.4
```

#### LLM으로 테이블 요약

```python
from langchain_openai import ChatOpenAI

# LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 테이블 요약 프롬프트
def summarize_table_with_llm(table_element):
    """LLM을 사용하여 테이블 내용 요약"""

    # 마크다운 형식으로 변환
    markdown = table_to_markdown(table_element)

    prompt = f"""다음 테이블의 내용을 한글로 요약해주세요:

{markdown}

요약 시 다음을 포함하세요:
1. 테이블의 주요 내용
2. 중요한 수치나 비교
3. 핵심 인사이트
"""

    response = llm.invoke(prompt)
    return response.content

# 두 번째 테이블 요약
if len(tables) > 1:
    table_el = tables[1]

    print(f"[테이블 요약 - LLM 분석]")
    summary = summarize_table_with_llm(table_el)
    print(summary)
```

**출력 예시**:
```
[테이블 요약 - LLM 분석]
이 테이블은 Transformer 모델의 두 가지 변형을 비교하고 있습니다:

1. **주요 내용**:
   - Transformer (base)와 Transformer (big) 두 모델의 구조적 차이와 성능을 비교

2. **중요한 수치**:
   - Base 모델: 8-head attention, 2048 FFN, 65M 파라미터, BLEU 27.3
   - Big 모델: 16-head attention, 4096 FFN, 213M 파라미터, BLEU 28.4
   - Big 모델이 파라미터 수가 약 3.3배 증가하면서 성능이 1.1점 향상

3. **핵심 인사이트**:
   - 모델 크기를 확장하면 번역 성능이 향상되지만, 파라미터 증가 대비 성능 향상은 선형적이지 않음
```

### 단계 6: ocr_only 전략 사용

**ocr_only 전략**은 Tesseract OCR을 활용하여 이미지 기반 문서에서 텍스트를 추출합니다. 스캔된 PDF나 다중 열 구조의 복잡한 문서에 적합합니다.

```python
from unstructured.partition.pdf import partition_pdf

# ocr_only 전략으로 PDF 파티셔닝
ocr_elements = partition_pdf(
    filename="data/scanned_document.pdf",
    strategy="ocr_only",  # Tesseract OCR 사용
    languages=["kor", "eng"],  # 한국어 + 영어 인식
)

print(f"총 요소 개수: {len(ocr_elements)}개")

# 요소별 개수 확인
ocr_elements_by_type = count_elements(ocr_elements)

for key, value in ocr_elements_by_type.items():
    print(f"{key}: {len(value)}개")
```

**특징**:
- **이미지 텍스트 인식**: 스캔된 문서에서 텍스트 추출
- **다국어 지원**: languages 매개변수로 언어 지정 가능
- **다중 열 처리**: 복잡한 레이아웃의 대안
- **느린 속도**: OCR 처리로 시간 소요

#### hi_res vs ocr_only 비교

한국어가 포함된 복잡한 PDF 문서를 두 전략으로 처리하여 비교합니다.

```python
# hi_res 전략
hi_res_elements = partition_pdf(
    filename="data/리비안_KR_with_table.pdf",
    strategy="hi_res",
    infer_table_structure=True,
    languages=["kor", "eng"],  # 한국어 + 영어
)

print(f"[hi_res 전략]")
print(f"총 요소 개수: {len(hi_res_elements)}개")

hi_res_by_type = count_elements(hi_res_elements)
for key, value in hi_res_by_type.items():
    print(f"  {key}: {len(value)}개")

# ocr_only 전략
ocr_elements = partition_pdf(
    filename="data/리비안_KR_with_table.pdf",
    strategy="ocr_only",
    languages=["kor", "eng"],  # 한국어 + 영어
)

print(f"\n[ocr_only 전략]")
print(f"총 요소 개수: {len(ocr_elements)}개")

ocr_by_type = count_elements(ocr_elements)
for key, value in ocr_by_type.items():
    print(f"  {key}: {len(value)}개")
```

**출력 예시**:
```
[hi_res 전략]
총 요소 개수: 45개
  Title: 12개
  Text: 28개
  Table: 5개

[ocr_only 전략]
총 요소 개수: 42개
  Title: 10개
  Text: 30개
  Table: 2개
```

#### 테이블 추출 비교

```python
# hi_res 테이블
if "Table" in hi_res_by_type and len(hi_res_by_type["Table"]) > 0:
    hi_res_table = hi_res_by_type["Table"][0]

    print(f"[hi_res 테이블]")
    print(f"페이지: {hi_res_table.metadata.page_number}")
    print(f"텍스트 길이: {len(hi_res_table.text)}자")
    print(f"텍스트 미리보기:")
    print(hi_res_table.text[:200] + "...")

# ocr_only 테이블
if "Table" in ocr_by_type and len(ocr_by_type["Table"]) > 0:
    ocr_table = ocr_by_type["Table"][0]

    print(f"\n[ocr_only 테이블]")
    print(f"페이지: {ocr_table.metadata.page_number}")
    print(f"텍스트 길이: {len(ocr_table.text)}자")
    print(f"텍스트 미리보기:")
    print(ocr_table.text[:200] + "...")
```

**비교 분석**:
- **hi_res**: 테이블 구조를 더 정확하게 인식, HTML 형식 제공
- **ocr_only**: 텍스트 인식은 정확하지만 구조 정보는 제한적
- **선택 기준**: 테이블 구조가 중요하면 hi_res, 텍스트만 필요하면 ocr_only

### 단계 7: 전략별 성능 비교

```python
import time

def benchmark_strategies(pdf_path):
    """파티셔닝 전략별 성능 비교"""

    strategies = ["fast", "hi_res", "ocr_only"]
    results = {}

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"[{strategy} 전략 테스트]")
        print(f"{'='*60}")

        start_time = time.time()

        try:
            # 전략별 설정
            kwargs = {
                "filename": pdf_path,
                "strategy": strategy,
            }

            # hi_res는 테이블 구조 인식 추가
            if strategy == "hi_res":
                kwargs["infer_table_structure"] = True

            # ocr_only는 언어 설정 추가
            if strategy == "ocr_only":
                kwargs["languages"] = ["kor", "eng"]

            # 파티셔닝 실행
            elements = partition_pdf(**kwargs)

            # 실행 시간 측정
            elapsed_time = time.time() - start_time

            # 요소별 개수 확인
            elements_by_type = count_elements(elements)

            # 결과 저장
            results[strategy] = {
                "총_요소": len(elements),
                "실행_시간": f"{elapsed_time:.2f}초",
                "요소별_개수": {k: len(v) for k, v in elements_by_type.items()}
            }

            # 결과 출력
            print(f"✅ 완료")
            print(f"   총 요소: {len(elements)}개")
            print(f"   실행 시간: {elapsed_time:.2f}초")

            for key, value in elements_by_type.items():
                print(f"   {key}: {len(value)}개")

        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            results[strategy] = {"오류": str(e)}

    return results

# 성능 비교 실행
results = benchmark_strategies("data/transformer.pdf")

# 결과 요약
print(f"\n{'='*60}")
print(f"[성능 비교 요약]")
print(f"{'='*60}")

import pandas as pd
summary_data = []

for strategy, data in results.items():
    if "오류" not in data:
        summary_data.append({
            "전략": strategy,
            "총_요소": data["총_요소"],
            "실행_시간": data["실행_시간"],
            "Title": data["요소별_개수"].get("Title", 0),
            "Text": data["요소별_개수"].get("Text", 0),
            "Table": data["요소별_개수"].get("Table", 0),
        })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))
```

**출력 예시**:
```
============================================================
[성능 비교 요약]
============================================================
   전략  총_요소  실행_시간  Title  Text  Table
   fast      76    2.35초     24    48      0
 hi_res      82   45.67초     26    52      4
ocr_only      78   67.23초     25    50      3
```

## 📖 참고 자료

### 공식 문서
- **Unstructured 공식 문서**: https://unstructured-io.github.io/unstructured/
- **Unstructured GitHub**: https://github.com/Unstructured-IO/unstructured
- **파티셔닝 전략 가이드**: https://unstructured-io.github.io/unstructured/core/partition.html

### 관련 라이브러리
- **pdfminer**: https://github.com/pdfminer/pdfminer.six
- **Tesseract OCR**: https://github.com/tesseract-ocr/tesseract
- **Poppler**: https://poppler.freedesktop.org/

### 추가 학습 자료
- **Unstructured 블로그**: https://unstructured.io/blog
- **PDF 처리 모범 사례**: https://unstructured-io.github.io/unstructured/best_practices.html

---

**다음 Part2에서 계속됩니다**: HTML, MS Office 문서, 텍스트/CSV 파일 파티셔닝 및 실습 문제
