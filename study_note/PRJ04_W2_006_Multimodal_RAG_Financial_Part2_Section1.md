# PRJ04_W2_006: 멀티모달 RAG 구현 (금융 분석 보고서) - Part 1

## 🎯 학습 목표

이 실습을 완료하면 다음을 수행할 수 있습니다:

1. **옵션 2 방식 이해**: 이미지 요약 기반 텍스트 RAG 방식의 원리와 장단점을 설명할 수 있습니다
2. **PDF 멀티모달 파싱**: unstructured 라이브러리를 사용해 PDF에서 텍스트, 이미지, 테이블을 분리 추출할 수 있습니다
3. **청크 타입 분석**: CompositeElement, TableChunk 등 다양한 청크 타입의 구조와 메타데이터를 이해합니다
4. **테이블 구조 처리**: HTML 형식 테이블을 DataFrame으로 변환하고 마크다운으로 포맷팅할 수 있습니다
5. **파싱 최적화**: 환경 변수를 통한 테이블 이미지 패딩 설정으로 파싱 품질을 개선할 수 있습니다

---

## 📚 핵심 개념

### 1. 멀티모달 RAG 3가지 옵션 비교

Part 1에서 구현할 **옵션 2**는 이미지 요약 기반 텍스트 RAG 방식입니다:

| 옵션 | 임베딩 방식 | 벡터 DB 저장 | 답변 생성 LLM | 이미지 활용 | 장점 | 단점 |
|---|---|---|---|---|---|---|
| **옵션 1** | 멀티모달 (CLIP) | 이미지 임베딩, 텍스트 임베딩 | Multimodal | 직접 활용 (base64) | 최고 이미지 활용도, 높은 답변 품질 | 높은 비용, base64 오버헤드 |
| **옵션 2** ⭐ | 텍스트 (OpenAI) | 텍스트 임베딩 (이미지 요약) | Text-only | 텍스트 요약 간접 활용 | 비용 효율성, 텍스트 RAG 인프라 활용 | 이미지 정보 손실, 답변 품질 제한적 |
| **옵션 3** | 텍스트 (OpenAI) | 텍스트 임베딩 (이미지 요약 + 참조) | Multimodal | 원본 이미지 참조 활용 | 옵션 1과 2의 절충, 이미지 정보 손실 감소 | 옵션 1보다 이미지 활용도 낮음, 이미지 참조 관리 필요 |

**선택 가이드:**
- **최고 품질 답변:** 옵션 1 (비용 고려)
- **비용 효율성 우선:** 옵션 2 ⭐ (답변 품질 제한 감수)
- **품질과 효율성 균형:** 옵션 3

### 2. 옵션 2 파이프라인 구조

```
PDF 문서
    ↓
[unstructured 파싱]
    ↓
텍스트, 이미지, 테이블 분리
    ↓
[멀티모달 LLM으로 이미지 → 텍스트 요약]
    ↓
모든 콘텐츠 → 텍스트 형식
    ↓
[텍스트 임베딩]
    ↓
벡터 DB 저장
    ↓
[검색 및 답변 생성]
```

**핵심 특징:**
- 이미지를 직접 벡터화하지 않고 텍스트 요약으로 변환
- 기존 텍스트 RAG 인프라 재사용 가능
- 이미지 정보는 LLM 요약을 통해 간접 활용

### 3. unstructured 청크 타입

| 타입 | 설명 | 주요 속성 |
|------|------|-----------|
| **CompositeElement** | 여러 요소를 포함하는 복합 청크 | `.metadata.orig_elements` 로 하위 요소 접근 |
| **TableChunk** | 테이블 데이터 청크 | `.metadata.text_as_html` 로 HTML 테이블 접근 |
| **Table** | 개별 테이블 요소 | `.metadata.text_as_html`, `.metadata.image_base64` |
| **Image** | 이미지 요소 | `.metadata.image_base64` |
| **Text** | 순수 텍스트 요소 | `.text` 속성으로 내용 접근 |

---

## 🛠️ 환경 설정

### 1. 필수 라이브러리 설치

```bash
# 코어 라이브러리
pip install langchain langchain-openai langchain-chroma langchain-core

# PDF 파싱
pip install unstructured[all-docs]
pip install pdf2image pdfplumber pillow

# 유틸리티
pip install python-dotenv pandas numpy

# 추적 (선택)
pip install langfuse
```

### 2. 환경 변수 설정

`.env` 파일:
```env
OPENAI_API_KEY=your_openai_key_here

# 테이블 이미지 패딩 설정 (선택)
EXTRACT_IMAGE_BLOCK_CROP_HORIZONTAL_PAD=20
EXTRACT_IMAGE_BLOCK_CROP_VERTICAL_PAD=10
```

### 3. 기본 임포트

```python
from dotenv import load_dotenv
load_dotenv()

import os
from glob import glob
from pprint import pprint
import json

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')
```

---

## 📝 단계별 구현

### 1단계: 유틸리티 함수 정의

```python
import base64
import io
from io import BytesIO

import numpy as np
from PIL import Image
from IPython.display import HTML, display
from langchain_core.documents import Document

import pickle
from langchain_core.stores import InMemoryStore


def is_base64(s):
    """문자열이 Base64로 인코딩되었는지 확인합니다"""
    try:
        return base64.b64encode(base64.b64decode(s)) == s.encode()
    except Exception:
        return False


def split_image_text_types(docs):
    """numpy 배열 이미지와 텍스트를 분리합니다"""
    images = []
    text = []
    for doc in docs:
        if isinstance(doc, str):
            pass
        elif isinstance(doc, Document):
            doc = doc.page_content  # 문서 내용 추출
        else:
            doc = doc.text # 문서 내용 추출

        if is_base64(doc):
            images.append(doc)  # base64로 인코딩된 문자열
        else:
            text.append(doc)
    return {"images": images, "texts": text}


def plt_img_base64(img_base64):
    """
    Base64로 인코딩된 이미지를 주피터 노트북에 표시

    매개변수:
    img_base64 (str): Base64로 인코딩된 이미지 문자열
    """
    # Base64 문자열을 소스로 하는 HTML img 태그 생성
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'

    # HTML을 렌더링하여 이미지 표시
    display(HTML(image_html))
```

**주요 함수:**
- `is_base64()`: Base64 인코딩 여부 확인 (이미지 vs 텍스트 구분)
- `split_image_text_types()`: 문서에서 이미지와 텍스트 분리
- `plt_img_base64()`: Base64 이미지를 주피터 노트북에 표시

### 2단계: PDF 문서 로딩 및 파싱

```python
from unstructured.partition.pdf import partition_pdf
from unstructured.cleaners.core import (
    clean_extra_whitespace,
    group_broken_paragraphs
)

# 데이터 저장 폴더 경로
data_path = "data/analyst_reports/"

# PDF 파일이 있는  경로
pdf_files = glob(os.path.join(data_path, "*.pdf"))

# 이미지를 저장할 경로
image_output_dir = os.path.join(data_path, 'images')

os.makedirs(image_output_dir, exist_ok=True)

print(f"PDF files:")
pprint(pdf_files)

# PDF 파일을 로드하고 파티셔닝
pdf_chunks = []
for pdf_file in pdf_files:
    chunks = partition_pdf(
        filename=pdf_file,
        strategy="hi_res",              # 고해상도 파싱
        infer_table_structure=True,     # 테이블 구조 추론
        languages=["eng", "kor"],       # 언어 설정

        # 이미지 추출 설정
        extract_image_block_types=["Image", "Table"],  # 이미지와 테이블 추출
        extract_image_block_to_payload=True,           # base64로 저장

        # 후처리 설정
        post_processors=[
            group_broken_paragraphs,  # 끊긴 단락 병합
            clean_extra_whitespace,   # 불필요한 공백 제거
        ],

        # 문서 청킹 설정
        chunking_strategy="by_title",     # 제목 기준 청킹
        max_characters=1200,              # 최대 문자 수
        new_after_n_chars=800,            # 새 청크 시작 기준
        combine_text_under_n_chars=600,   # 병합 기준

    )
    pdf_chunks.extend(chunks)


# 청크 개수 확인
print(f"총 청크 개수: {len(pdf_chunks)}")
```

**핵심 파라미터:**
- `strategy="hi_res"`: 고해상도 파싱 (테이블/이미지 정확도 향상)
- `infer_table_structure=True`: HTML 테이블 구조 추론
- `extract_image_block_to_payload=True`: 이미지를 base64로 메타데이터에 저장
- `chunking_strategy="by_title"`: 문서 제목 기준으로 의미 단위 청킹

### 3단계: 타입별 분석 및 저장

```python
# 타입별 분석
chunk_types = {}
for chunk in pdf_chunks:
    chunk_type = str(type(chunk))
    if chunk_type in chunk_types:
        chunk_types[chunk_type] += 1
    else:
        chunk_types[chunk_type] = 1

print("📋 청크 타입 분포:")
for chunk_type, count in chunk_types.items():
    print(f"  • {chunk_type.split('.')[-1].replace('>', '')}: {count}개")

# 청크 내용 확인
for chunk in pdf_chunks[1:3]:
    print(chunk)
    print("=" * 100)

# 청크 저장 (재사용을 위해)
import pickle

with open(os.path.join(data_path, "pdf_base64_chunks.pkl"), "wb") as f:
    pickle.dump(pdf_chunks, f)

print(f"💾 청크 저장 완료: {os.path.join(data_path, 'pdf_base64_chunks.pkl')}")
```

**출력 예시:**
```
📋 청크 타입 분포:
  • CompositeElement: 45개
  • TableChunk: 12개
  • Title: 8개
```

### 4단계: 타입별 구조 확인

#### 4-1. CompositeElement 구조

```python
# 각 청크 문서의 타입 확인
set([str(type(el)) for el in pdf_chunks])

# CompositeElement 확인
pdf_chunks[1]

# CompositeElement는 하위 구성 요소를 포함하고 있음
pdf_chunks[1].metadata.orig_elements
```

**CompositeElement 특징:**
- 여러 요소(텍스트, 이미지, 테이블)를 포함하는 복합 청크
- `.metadata.orig_elements`로 하위 요소 리스트 접근
- 각 하위 요소는 독립적인 타입과 메타데이터 보유

#### 4-2. Table 객체 처리

```python
# Table 객체 확인
tab = pdf_chunks[1].metadata.orig_elements[2]
tab.to_dict()

# Table 객체의 데이터프레임 변환
tab_df = pd.read_html(tab.metadata.text_as_html)[0]

# 데이터프레임 확인
display(tab_df)

# 데이터프레임을 마크다운 변환
tab_md = tab_df.to_markdown(index=False)
print(tab_md)

# image_base64 이미지 확인
plt_img_base64(tab.metadata.image_base64)
```

**Table 객체 주요 속성:**
- `.metadata.text_as_html`: HTML 형식 테이블 (DataFrame 변환 가능)
- `.metadata.image_base64`: 테이블 이미지 (OCR 결과)
- `.to_dict()`: 전체 메타데이터 딕셔너리

#### 4-3. TableChunk 확인 및 디버깅

```python
# TableChunk 객체 확인
for i, c in enumerate(pdf_chunks):
    if "TableChunk" in str(type(c)):
        print(f"TableChunk {i}:")
        print(c)
        print("=" * 100)

# TableChunk 객체의 orig_elements 확인
pdf_chunks[4].metadata.orig_elements

tab = pdf_chunks[4].metadata.orig_elements[0]

# Table 객체 확인
tab.to_dict()
```

**text_as_html이 None인 경우 디버깅:**

```python
# Table 객체의 데이터프레임 변환
# text_as_html이 None인 경우 디버깅 정보 출력

chunk_idx = 4
element_idx = 0

# 디버깅: 청크 정보 확인
print(f"📌 대상 청크: pdf_chunks[{chunk_idx}]")
print(f"   청크 타입: {type(pdf_chunks[chunk_idx])}")

if hasattr(pdf_chunks[chunk_idx], 'metadata') and hasattr(pdf_chunks[chunk_idx].metadata, 'orig_elements'):
    orig_elements = pdf_chunks[chunk_idx].metadata.orig_elements
    print(f"   orig_elements 개수: {len(orig_elements)}")

    if len(orig_elements) > element_idx:
        target_element = orig_elements[element_idx]
        print(f"\n📌 대상 요소: orig_elements[{element_idx}]")
        print(f"   요소 타입: {type(target_element)}")

        if hasattr(target_element, 'metadata'):
            print(f"   metadata 속성들: {[attr for attr in dir(target_element.metadata) if not attr.startswith('_')]}")

            # text_as_html 확인
            if hasattr(target_element.metadata, 'text_as_html'):
                table_html = target_element.metadata.text_as_html
                print(f"   text_as_html: {'있음' if table_html else 'None'}")

                if table_html is not None:
                    table_df = pd.read_html(table_html)[0]
                    print(f"\n✅ 테이블 변환 성공!")
                    display(table_df)
                else:
                    print(f"\n⚠️ text_as_html이 None입니다.")
                    print(f"   → PDF 파싱 시 테이블 구조 추출 실패")
                    print(f"   → infer_table_structure=True 옵션이 적용되었는지 확인")

                    # 대체 테이블 찾기
                    print(f"\n🔍 text_as_html이 있는 테이블 요소 검색 중...")
                    found_tables = []
                    for i, chunk in enumerate(pdf_chunks):
                        if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
                            for j, el in enumerate(chunk.metadata.orig_elements):
                                if hasattr(el, 'metadata') and hasattr(el.metadata, 'text_as_html'):
                                    if el.metadata.text_as_html is not None:
                                        found_tables.append((i, j, type(el).__name__))

                    if found_tables:
                        print(f"   발견된 테이블 {len(found_tables)}개:")
                        for i, j, el_type in found_tables[:5]:
                            print(f"   - pdf_chunks[{i}].metadata.orig_elements[{j}] ({el_type})")

                        # 첫 번째 유효한 테이블 표시
                        first_i, first_j, _ = found_tables[0]
                        table_df = pd.read_html(pdf_chunks[first_i].metadata.orig_elements[first_j].metadata.text_as_html)[0]
                        print(f"\n📊 대체 테이블 (pdf_chunks[{first_i}].orig_elements[{first_j}]):")
                        display(table_df)
                    else:
                        print(f"   ❌ text_as_html이 있는 테이블을 찾을 수 없습니다.")
            else:
                print(f"   ⚠️ text_as_html 속성이 없습니다.")
        else:
            print(f"   ⚠️ metadata 속성이 없습니다.")
    else:
        print(f"   ⚠️ orig_elements[{element_idx}]가 존재하지 않습니다.")
else:
    print(f"   ⚠️ orig_elements 속성이 없습니다.")
```

### 5단계: 표 가장자리 인식 개선

테이블 이미지의 가장자리가 잘리는 문제를 환경 변수로 해결:

**`.env` 파일에 추가:**
```env
# 테이블 이미지의 가장자리 인식 개선을 위한 패딩 설정
EXTRACT_IMAGE_BLOCK_CROP_HORIZONTAL_PAD=20
EXTRACT_IMAGE_BLOCK_CROP_VERTICAL_PAD=10
```

**⚠️ 커널 재시작 후 적용됨**

```python
# 패딩 설정 후 재파싱
data_path = "data/analyst_reports/"
pdf_files = glob(os.path.join(data_path, "*.pdf"))

# 새 이미지 저장 경로
image_output_dir = os.path.join(data_path, 'images_pad_with_w20_h10')
os.makedirs(image_output_dir, exist_ok=True)

# PDF 재파싱 (동일한 partition_pdf 설정)
pdf_chunks = []
for pdf_file in pdf_files:
    chunks = partition_pdf(
        filename=pdf_file,
        strategy="hi_res",
        infer_table_structure=True,
        languages=["eng", "kor"],

        # 이미지 추출 설정
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=True,

        # 후처리 설정
        post_processors=[
            group_broken_paragraphs,
            clean_extra_whitespace,
        ],

        # 문서 청킹 설정
        chunking_strategy="by_title",
        max_characters=1200,
        new_after_n_chars=800,
        combine_text_under_n_chars=600,
    )
    pdf_chunks.extend(chunks)

print(f"✅ 패딩 적용 후 총 청크 개수: {len(pdf_chunks)}")

# 타입별 분석
chunk_types = {}
for chunk in pdf_chunks:
    chunk_type = str(type(chunk))
    if chunk_type in chunk_types:
        chunk_types[chunk_type] += 1
    else:
        chunk_types[chunk_type] = 1

print("\n📋 청크 타입 분포:")
for chunk_type, count in chunk_types.items():
    print(f"  • {chunk_type.split('.')[-1].replace('>', '')}: {count}개")

# TableChunk 객체 확인
for i, c in enumerate(pdf_chunks):
    if "TableChunk" in str(type(c)):
        print(f"\nTableChunk {i}:")
        print(c)
        print("=" * 100)
```

**패딩 값 조정 가이드:**
- 테이블 가장자리가 잘리는 경우: 패딩 값 증가 (30, 15 등)
- 불필요한 배경이 포함되는 경우: 패딩 값 감소 (10, 5 등)
- 문서 복잡도와 테이블 크기에 따라 실험 필요

---

## 🎯 실습 문제

### 실습 1 (기본): PDF 파싱 및 타입 분석

**문제:**
`data/analyst_reports/` 폴더의 모든 PDF 파일을 파싱하고, 다음 정보를 출력하세요:
1. 총 청크 개수
2. 청크 타입별 개수
3. TableChunk만 필터링하여 개수와 첫 3개의 내용 출력

**힌트:**
```python
# TableChunk 필터링
table_chunks = [c for c in pdf_chunks if "TableChunk" in str(type(c))]
```

### 실습 2 (중급): 테이블 추출 및 변환 시스템

**문제:**
모든 청크에서 테이블을 추출하고 다음을 수행하는 클래스를 작성하세요:
1. `text_as_html`이 있는 모든 Table 객체 찾기
2. DataFrame으로 변환하여 리스트로 저장
3. 각 테이블을 마크다운 형식으로 저장

**요구사항:**
```python
class TableExtractor:
    def __init__(self, chunks):
        self.chunks = chunks
        self.tables = []

    def extract_tables(self) -> List[pd.DataFrame]:
        """모든 테이블을 DataFrame으로 추출"""
        pass

    def save_as_markdown(self, output_dir: str):
        """테이블을 마크다운 파일로 저장"""
        pass

    def get_table_summary(self) -> dict:
        """테이블 요약 정보 반환 (개수, 평균 행/열 수 등)"""
        pass
```

### 실습 3 (고급): 적응형 패딩 최적화 시스템

**문제:**
다양한 패딩 값을 테스트하여 최적의 패딩 설정을 자동으로 찾는 시스템을 구현하세요:

**요구사항:**
1. 패딩 조합 (horizontal: 0~30, vertical: 0~20) 테스트
2. 각 조합에 대해 파싱 후 `text_as_html`이 있는 테이블 개수 측정
3. 가장 많은 테이블을 추출한 조합 찾기
4. 결과를 JSON으로 저장

```python
class PaddingOptimizer:
    def __init__(self, pdf_file, test_ranges):
        self.pdf_file = pdf_file
        self.test_ranges = test_ranges
        self.results = []

    def test_padding(self, h_pad: int, v_pad: int) -> dict:
        """특정 패딩 값으로 파싱하고 테이블 개수 반환"""
        pass

    def run_optimization(self) -> dict:
        """모든 패딩 조합 테스트"""
        pass

    def get_best_padding(self) -> tuple:
        """최적의 패딩 값 반환"""
        pass

    def save_results(self, output_path: str):
        """결과를 JSON으로 저장"""
        pass
```

---

## 💡 솔루션 예시

### 솔루션 1: PDF 파싱 및 타입 분석

```python
from unstructured.partition.pdf import partition_pdf
from unstructured.cleaners.core import clean_extra_whitespace, group_broken_paragraphs
from glob import glob
import os

# 데이터 경로 설정
data_path = "data/analyst_reports/"
pdf_files = glob(os.path.join(data_path, "*.pdf"))

# PDF 파싱
pdf_chunks = []
for pdf_file in pdf_files:
    chunks = partition_pdf(
        filename=pdf_file,
        strategy="hi_res",
        infer_table_structure=True,
        languages=["eng", "kor"],
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=True,
        post_processors=[group_broken_paragraphs, clean_extra_whitespace],
        chunking_strategy="by_title",
        max_characters=1200,
        new_after_n_chars=800,
        combine_text_under_n_chars=600,
    )
    pdf_chunks.extend(chunks)

# 1. 총 청크 개수
print(f"📄 총 청크 개수: {len(pdf_chunks)}")

# 2. 청크 타입별 개수
chunk_types = {}
for chunk in pdf_chunks:
    chunk_type = str(type(chunk)).split("'")[1].split(".")[-1]
    chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1

print("\n📋 청크 타입별 개수:")
for chunk_type, count in sorted(chunk_types.items(), key=lambda x: -x[1]):
    print(f"  • {chunk_type}: {count}개")

# 3. TableChunk 필터링
table_chunks = [c for c in pdf_chunks if "TableChunk" in str(type(c))]
print(f"\n📊 TableChunk 개수: {len(table_chunks)}개")

print("\n📝 첫 3개의 TableChunk 내용:")
for i, tc in enumerate(table_chunks[:3]):
    print(f"\n--- TableChunk {i+1} ---")
    print(f"텍스트: {tc.text[:200]}..." if len(tc.text) > 200 else f"텍스트: {tc.text}")
    if hasattr(tc.metadata, 'orig_elements') and tc.metadata.orig_elements:
        print(f"하위 요소 개수: {len(tc.metadata.orig_elements)}")
```

**출력 예시:**
```
📄 총 청크 개수: 127

📋 청크 타입별 개수:
  • CompositeElement: 78개
  • TableChunk: 23개
  • Title: 15개
  • Header: 11개

📊 TableChunk 개수: 23개

📝 첫 3개의 TableChunk 내용:

--- TableChunk 1 ---
텍스트: 2023년 2분기 매출은 12.5조원으로 전년 동기 대비 15% 증가했습니다...
하위 요소 개수: 1

--- TableChunk 2 ---
텍스트: 영업이익률은 8.3%로 전 분기 대비 1.2%p 개선되었습니다...
하위 요소 개수: 1
```

### 솔루션 2: 테이블 추출 및 변환 시스템

```python
import pandas as pd
import os
from typing import List
from pathlib import Path


class TableExtractor:
    def __init__(self, chunks):
        self.chunks = chunks
        self.tables = []
        self.table_metadata = []

    def extract_tables(self) -> List[pd.DataFrame]:
        """모든 테이블을 DataFrame으로 추출"""
        self.tables = []
        self.table_metadata = []

        for chunk_idx, chunk in enumerate(self.chunks):
            # orig_elements가 있는지 확인
            if not (hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements')):
                continue

            # orig_elements에서 테이블 찾기
            for el_idx, el in enumerate(chunk.metadata.orig_elements):
                if not (hasattr(el, 'metadata') and hasattr(el.metadata, 'text_as_html')):
                    continue

                table_html = el.metadata.text_as_html
                if table_html is None:
                    continue

                try:
                    # HTML을 DataFrame으로 변환
                    table_df = pd.read_html(table_html)[0]
                    self.tables.append(table_df)

                    # 메타데이터 저장
                    self.table_metadata.append({
                        'chunk_idx': chunk_idx,
                        'element_idx': el_idx,
                        'rows': len(table_df),
                        'cols': len(table_df.columns)
                    })

                except Exception as e:
                    print(f"⚠️ 테이블 변환 실패 (chunk {chunk_idx}, element {el_idx}): {e}")

        print(f"✅ 총 {len(self.tables)}개의 테이블 추출 완료")
        return self.tables

    def save_as_markdown(self, output_dir: str):
        """테이블을 마크다운 파일로 저장"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for idx, (table, meta) in enumerate(zip(self.tables, self.table_metadata)):
            # 마크다운 변환
            table_md = table.to_markdown(index=False)

            # 파일 저장
            file_path = output_path / f"table_{idx+1}.md"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# 테이블 {idx+1}\n\n")
                f.write(f"**출처:** Chunk {meta['chunk_idx']}, Element {meta['element_idx']}\n")
                f.write(f"**크기:** {meta['rows']}행 × {meta['cols']}열\n\n")
                f.write(table_md)

        print(f"✅ {len(self.tables)}개의 테이블을 {output_dir}에 저장 완료")

    def get_table_summary(self) -> dict:
        """테이블 요약 정보 반환"""
        if not self.tables:
            return {"error": "테이블이 추출되지 않았습니다"}

        total_rows = sum(meta['rows'] for meta in self.table_metadata)
        total_cols = sum(meta['cols'] for meta in self.table_metadata)

        return {
            'total_tables': len(self.tables),
            'avg_rows': total_rows / len(self.tables),
            'avg_cols': total_cols / len(self.tables),
            'max_rows': max(meta['rows'] for meta in self.table_metadata),
            'max_cols': max(meta['cols'] for meta in self.table_metadata),
            'min_rows': min(meta['rows'] for meta in self.table_metadata),
            'min_cols': min(meta['cols'] for meta in self.table_metadata),
        }


# 사용 예시
extractor = TableExtractor(pdf_chunks)
tables = extractor.extract_tables()

# 요약 정보 출력
summary = extractor.get_table_summary()
print("\n📊 테이블 요약:")
print(f"  총 테이블 수: {summary['total_tables']}개")
print(f"  평균 행 수: {summary['avg_rows']:.1f}행")
print(f"  평균 열 수: {summary['avg_cols']:.1f}열")
print(f"  최대 크기: {summary['max_rows']}행 × {summary['max_cols']}열")

# 마크다운으로 저장
extractor.save_as_markdown("data/analyst_reports/tables_md")
```

**출력 예시:**
```
✅ 총 23개의 테이블 추출 완료

📊 테이블 요약:
  총 테이블 수: 23개
  평균 행 수: 8.3행
  평균 열 수: 4.6열
  최대 크기: 25행 × 8열

✅ 23개의 테이블을 data/analyst_reports/tables_md에 저장 완료
```

### 솔루션 3: 적응형 패딩 최적화 시스템

```python
import os
import json
from typing import List, Tuple
from unstructured.partition.pdf import partition_pdf
from unstructured.cleaners.core import clean_extra_whitespace, group_broken_paragraphs


class PaddingOptimizer:
    def __init__(self, pdf_file, test_ranges):
        self.pdf_file = pdf_file
        self.test_ranges = test_ranges  # {'h_pad': [0, 10, 20], 'v_pad': [0, 5, 10]}
        self.results = []

    def test_padding(self, h_pad: int, v_pad: int) -> dict:
        """특정 패딩 값으로 파싱하고 테이블 개수 반환"""
        # 환경 변수 설정
        os.environ['EXTRACT_IMAGE_BLOCK_CROP_HORIZONTAL_PAD'] = str(h_pad)
        os.environ['EXTRACT_IMAGE_BLOCK_CROP_VERTICAL_PAD'] = str(v_pad)

        try:
            # PDF 파싱
            chunks = partition_pdf(
                filename=self.pdf_file,
                strategy="hi_res",
                infer_table_structure=True,
                languages=["eng", "kor"],
                extract_image_block_types=["Image", "Table"],
                extract_image_block_to_payload=True,
                post_processors=[group_broken_paragraphs, clean_extra_whitespace],
                chunking_strategy="by_title",
                max_characters=1200,
                new_after_n_chars=800,
                combine_text_under_n_chars=600,
            )

            # text_as_html이 있는 테이블 개수 세기
            table_count = 0
            for chunk in chunks:
                if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
                    for el in chunk.metadata.orig_elements:
                        if (hasattr(el, 'metadata') and
                            hasattr(el.metadata, 'text_as_html') and
                            el.metadata.text_as_html is not None):
                            table_count += 1

            return {
                'h_pad': h_pad,
                'v_pad': v_pad,
                'table_count': table_count,
                'total_chunks': len(chunks),
                'success': True
            }

        except Exception as e:
            return {
                'h_pad': h_pad,
                'v_pad': v_pad,
                'table_count': 0,
                'total_chunks': 0,
                'success': False,
                'error': str(e)
            }

    def run_optimization(self) -> dict:
        """모든 패딩 조합 테스트"""
        print(f"🔍 패딩 최적화 시작: {self.pdf_file}")
        print(f"   테스트 범위: H={self.test_ranges['h_pad']}, V={self.test_ranges['v_pad']}")

        total_tests = len(self.test_ranges['h_pad']) * len(self.test_ranges['v_pad'])
        current_test = 0

        for h_pad in self.test_ranges['h_pad']:
            for v_pad in self.test_ranges['v_pad']:
                current_test += 1
                print(f"\n[{current_test}/{total_tests}] 테스트 중: H={h_pad}, V={v_pad}")

                result = self.test_padding(h_pad, v_pad)
                self.results.append(result)

                if result['success']:
                    print(f"   ✅ 테이블 {result['table_count']}개 추출")
                else:
                    print(f"   ❌ 실패: {result.get('error', 'Unknown')}")

        # 최적 결과 찾기
        best = self.get_best_padding()
        print(f"\n🏆 최적 패딩: H={best['h_pad']}, V={best['v_pad']} (테이블 {best['table_count']}개)")

        return {
            'best_padding': best,
            'all_results': self.results
        }

    def get_best_padding(self) -> dict:
        """최적의 패딩 값 반환"""
        if not self.results:
            return {}

        # 테이블 개수가 가장 많은 결과 찾기
        successful_results = [r for r in self.results if r['success']]
        if not successful_results:
            return {}

        best_result = max(successful_results, key=lambda x: x['table_count'])
        return best_result

    def save_results(self, output_path: str):
        """결과를 JSON으로 저장"""
        results_data = {
            'pdf_file': self.pdf_file,
            'test_ranges': self.test_ranges,
            'best_padding': self.get_best_padding(),
            'all_results': self.results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)

        print(f"\n💾 결과 저장: {output_path}")


# 사용 예시
optimizer = PaddingOptimizer(
    pdf_file="data/analyst_reports/sample_report.pdf",
    test_ranges={
        'h_pad': [0, 10, 20, 30],
        'v_pad': [0, 5, 10, 15, 20]
    }
)

# 최적화 실행
results = optimizer.run_optimization()

# 결과 저장
optimizer.save_results("data/analyst_reports/padding_optimization_results.json")

# 최적 패딩 적용
best = optimizer.get_best_padding()
print(f"\n✅ 최적 설정을 .env에 적용:")
print(f"EXTRACT_IMAGE_BLOCK_CROP_HORIZONTAL_PAD={best['h_pad']}")
print(f"EXTRACT_IMAGE_BLOCK_CROP_VERTICAL_PAD={best['v_pad']}")
```

**출력 예시:**
```
🔍 패딩 최적화 시작: data/analyst_reports/sample_report.pdf
   테스트 범위: H=[0, 10, 20, 30], V=[0, 5, 10, 15, 20]

[1/20] 테스트 중: H=0, V=0
   ✅ 테이블 18개 추출

[2/20] 테스트 중: H=0, V=5
   ✅ 테이블 19개 추출

...

[12/20] 테스트 중: H=20, V=10
   ✅ 테이블 23개 추출

...

[20/20] 테스트 중: H=30, V=20
   ✅ 테이블 22개 추출

🏆 최적 패딩: H=20, V=10 (테이블 23개)

💾 결과 저장: data/analyst_reports/padding_optimization_results.json

✅ 최적 설정을 .env에 적용:
EXTRACT_IMAGE_BLOCK_CROP_HORIZONTAL_PAD=20
EXTRACT_IMAGE_BLOCK_CROP_VERTICAL_PAD=10
```

---

## 🌟 실무 활용 예시

### 예시 1: 금융 보고서 자동 분석 시스템

```python
import os
from glob import glob
from typing import Dict, List
import pandas as pd
from unstructured.partition.pdf import partition_pdf
from unstructured.cleaners.core import clean_extra_whitespace, group_broken_paragraphs


class FinancialReportParser:
    """금융 분석 보고서 자동 파싱 및 구조화 시스템"""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.reports = []

    def parse_reports(self, pdf_pattern: str = "*.pdf") -> List[Dict]:
        """모든 보고서 파싱"""
        pdf_files = glob(os.path.join(self.data_dir, pdf_pattern))
        print(f"📄 {len(pdf_files)}개의 PDF 파일 발견")

        for pdf_file in pdf_files:
            report_name = os.path.basename(pdf_file).replace('.pdf', '')
            print(f"\n처리 중: {report_name}")

            # PDF 파싱
            chunks = partition_pdf(
                filename=pdf_file,
                strategy="hi_res",
                infer_table_structure=True,
                languages=["eng", "kor"],
                extract_image_block_types=["Image", "Table"],
                extract_image_block_to_payload=True,
                post_processors=[group_broken_paragraphs, clean_extra_whitespace],
                chunking_strategy="by_title",
                max_characters=1200,
                new_after_n_chars=800,
                combine_text_under_n_chars=600,
            )

            # 구조화
            structured_data = self._structure_chunks(chunks, report_name)
            self.reports.append(structured_data)

            print(f"   ✅ 텍스트: {len(structured_data['texts'])}개")
            print(f"   ✅ 테이블: {len(structured_data['tables'])}개")
            print(f"   ✅ 이미지: {len(structured_data['images'])}개")

        return self.reports

    def _structure_chunks(self, chunks, report_name: str) -> Dict:
        """청크를 타입별로 구조화"""
        texts = []
        tables = []
        images = []

        for chunk_idx, chunk in enumerate(chunks):
            # 텍스트 추출
            if hasattr(chunk, 'text') and chunk.text:
                texts.append({
                    'content': chunk.text,
                    'chunk_idx': chunk_idx,
                    'type': str(type(chunk)).split("'")[1].split(".")[-1]
                })

            # orig_elements에서 테이블과 이미지 추출
            if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
                for el_idx, el in enumerate(chunk.metadata.orig_elements):
                    # 테이블
                    if (hasattr(el, 'metadata') and
                        hasattr(el.metadata, 'text_as_html') and
                        el.metadata.text_as_html):
                        try:
                            table_df = pd.read_html(el.metadata.text_as_html)[0]
                            tables.append({
                                'dataframe': table_df,
                                'html': el.metadata.text_as_html,
                                'markdown': table_df.to_markdown(index=False),
                                'chunk_idx': chunk_idx,
                                'element_idx': el_idx
                            })
                        except:
                            pass

                    # 이미지
                    if (hasattr(el, 'metadata') and
                        hasattr(el.metadata, 'image_base64') and
                        el.metadata.image_base64):
                        images.append({
                            'base64': el.metadata.image_base64,
                            'chunk_idx': chunk_idx,
                            'element_idx': el_idx
                        })

        return {
            'report_name': report_name,
            'texts': texts,
            'tables': tables,
            'images': images
        }

    def export_tables_to_excel(self, output_dir: str):
        """모든 테이블을 Excel 파일로 내보내기"""
        os.makedirs(output_dir, exist_ok=True)

        for report in self.reports:
            report_name = report['report_name']
            tables = report['tables']

            if not tables:
                continue

            # Excel writer 생성
            excel_path = os.path.join(output_dir, f"{report_name}_tables.xlsx")
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                for idx, table in enumerate(tables):
                    sheet_name = f"Table_{idx+1}"
                    table['dataframe'].to_excel(writer, sheet_name=sheet_name, index=False)

            print(f"📊 {report_name}: {len(tables)}개 테이블 → {excel_path}")

    def export_summary_json(self, output_path: str):
        """전체 파싱 결과를 JSON으로 내보내기"""
        summary = []
        for report in self.reports:
            summary.append({
                'report_name': report['report_name'],
                'statistics': {
                    'total_texts': len(report['texts']),
                    'total_tables': len(report['tables']),
                    'total_images': len(report['images'])
                },
                'texts': [{'content': t['content'][:200], 'type': t['type']}
                         for t in report['texts'][:5]],  # 처음 5개만
                'tables': [{'markdown': t['markdown'][:500], 'shape': t['dataframe'].shape}
                          for t in report['tables'][:5]]  # 처음 5개만
            })

        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"💾 요약 저장: {output_path}")


# 사용 예시
parser = FinancialReportParser(data_dir="data/analyst_reports")

# 모든 보고서 파싱
reports = parser.parse_reports()

# 테이블을 Excel로 내보내기
parser.export_tables_to_excel(output_dir="data/analyst_reports/tables_excel")

# 요약을 JSON으로 내보내기
parser.export_summary_json(output_path="data/analyst_reports/parsing_summary.json")
```

**활용 시나리오:**
- 증권사 애널리스트 보고서 자동 수집 및 구조화
- 재무제표 테이블 자동 추출 및 데이터베이스 저장
- 차트/그래프 이미지 수집 및 메타데이터 관리

### 예시 2: PDF 품질 검증 시스템

```python
from typing import Dict, List
import pandas as pd
from collections import defaultdict


class PDFQualityValidator:
    """PDF 파싱 품질 검증 및 리포트 생성 시스템"""

    def __init__(self, chunks):
        self.chunks = chunks
        self.validation_results = {}

    def validate_all(self) -> Dict:
        """전체 검증 실행"""
        self.validation_results = {
            'chunk_distribution': self._validate_chunk_distribution(),
            'table_quality': self._validate_table_quality(),
            'image_quality': self._validate_image_quality(),
            'text_quality': self._validate_text_quality(),
            'metadata_completeness': self._validate_metadata()
        }
        return self.validation_results

    def _validate_chunk_distribution(self) -> Dict:
        """청크 타입 분포 검증"""
        distribution = defaultdict(int)
        for chunk in self.chunks:
            chunk_type = str(type(chunk)).split("'")[1].split(".")[-1]
            distribution[chunk_type] += 1

        # 이상 패턴 감지
        warnings = []
        if distribution.get('TableChunk', 0) == 0:
            warnings.append("⚠️ 테이블이 전혀 감지되지 않았습니다")
        if distribution.get('CompositeElement', 0) == 0:
            warnings.append("⚠️ CompositeElement가 감지되지 않았습니다")

        return {
            'distribution': dict(distribution),
            'total_chunks': len(self.chunks),
            'warnings': warnings
        }

    def _validate_table_quality(self) -> Dict:
        """테이블 품질 검증"""
        total_tables = 0
        valid_html = 0
        invalid_html = 0
        empty_tables = 0

        for chunk in self.chunks:
            if not (hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements')):
                continue

            for el in chunk.metadata.orig_elements:
                if not (hasattr(el, 'metadata') and hasattr(el.metadata, 'text_as_html')):
                    continue

                total_tables += 1
                table_html = el.metadata.text_as_html

                if table_html is None:
                    invalid_html += 1
                else:
                    try:
                        df = pd.read_html(table_html)[0]
                        if df.empty or len(df) == 0:
                            empty_tables += 1
                        else:
                            valid_html += 1
                    except:
                        invalid_html += 1

        quality_score = (valid_html / total_tables * 100) if total_tables > 0 else 0

        return {
            'total_tables': total_tables,
            'valid_html': valid_html,
            'invalid_html': invalid_html,
            'empty_tables': empty_tables,
            'quality_score': quality_score,
            'status': '✅ 양호' if quality_score >= 80 else '⚠️ 개선 필요'
        }

    def _validate_image_quality(self) -> Dict:
        """이미지 품질 검증"""
        total_images = 0
        valid_base64 = 0
        invalid_base64 = 0

        for chunk in self.chunks:
            if not (hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements')):
                continue

            for el in chunk.metadata.orig_elements:
                if not (hasattr(el, 'metadata') and hasattr(el.metadata, 'image_base64')):
                    continue

                total_images += 1
                image_data = el.metadata.image_base64

                if image_data and len(image_data) > 100:  # 최소 크기 체크
                    valid_base64 += 1
                else:
                    invalid_base64 += 1

        quality_score = (valid_base64 / total_images * 100) if total_images > 0 else 0

        return {
            'total_images': total_images,
            'valid_base64': valid_base64,
            'invalid_base64': invalid_base64,
            'quality_score': quality_score,
            'status': '✅ 양호' if quality_score >= 90 else '⚠️ 개선 필요'
        }

    def _validate_text_quality(self) -> Dict:
        """텍스트 품질 검증"""
        total_texts = 0
        empty_texts = 0
        short_texts = 0
        normal_texts = 0

        for chunk in self.chunks:
            if hasattr(chunk, 'text'):
                total_texts += 1
                text_len = len(chunk.text.strip())

                if text_len == 0:
                    empty_texts += 1
                elif text_len < 50:
                    short_texts += 1
                else:
                    normal_texts += 1

        quality_score = (normal_texts / total_texts * 100) if total_texts > 0 else 0

        return {
            'total_texts': total_texts,
            'empty_texts': empty_texts,
            'short_texts': short_texts,
            'normal_texts': normal_texts,
            'quality_score': quality_score,
            'status': '✅ 양호' if quality_score >= 70 else '⚠️ 개선 필요'
        }

    def _validate_metadata(self) -> Dict:
        """메타데이터 완성도 검증"""
        chunks_with_metadata = 0
        chunks_with_orig_elements = 0

        for chunk in self.chunks:
            if hasattr(chunk, 'metadata'):
                chunks_with_metadata += 1
                if hasattr(chunk.metadata, 'orig_elements'):
                    chunks_with_orig_elements += 1

        metadata_score = (chunks_with_metadata / len(self.chunks) * 100) if self.chunks else 0

        return {
            'total_chunks': len(self.chunks),
            'chunks_with_metadata': chunks_with_metadata,
            'chunks_with_orig_elements': chunks_with_orig_elements,
            'completeness_score': metadata_score,
            'status': '✅ 양호' if metadata_score >= 95 else '⚠️ 개선 필요'
        }

    def generate_report(self) -> str:
        """검증 리포트 생성"""
        if not self.validation_results:
            self.validate_all()

        report = []
        report.append("=" * 60)
        report.append("📋 PDF 파싱 품질 검증 리포트")
        report.append("=" * 60)

        # 청크 분포
        dist = self.validation_results['chunk_distribution']
        report.append(f"\n📦 청크 분포 (총 {dist['total_chunks']}개)")
        for chunk_type, count in dist['distribution'].items():
            report.append(f"   • {chunk_type}: {count}개")
        for warning in dist['warnings']:
            report.append(f"   {warning}")

        # 테이블 품질
        table = self.validation_results['table_quality']
        report.append(f"\n📊 테이블 품질 {table['status']}")
        report.append(f"   • 총 테이블: {table['total_tables']}개")
        report.append(f"   • 유효한 HTML: {table['valid_html']}개")
        report.append(f"   • 무효한 HTML: {table['invalid_html']}개")
        report.append(f"   • 품질 점수: {table['quality_score']:.1f}%")

        # 이미지 품질
        image = self.validation_results['image_quality']
        report.append(f"\n🖼️ 이미지 품질 {image['status']}")
        report.append(f"   • 총 이미지: {image['total_images']}개")
        report.append(f"   • 유효한 Base64: {image['valid_base64']}개")
        report.append(f"   • 품질 점수: {image['quality_score']:.1f}%")

        # 텍스트 품질
        text = self.validation_results['text_quality']
        report.append(f"\n📝 텍스트 품질 {text['status']}")
        report.append(f"   • 총 텍스트: {text['total_texts']}개")
        report.append(f"   • 정상 길이: {text['normal_texts']}개")
        report.append(f"   • 짧은 텍스트: {text['short_texts']}개")
        report.append(f"   • 품질 점수: {text['quality_score']:.1f}%")

        # 메타데이터 완성도
        meta = self.validation_results['metadata_completeness']
        report.append(f"\n🔍 메타데이터 완성도 {meta['status']}")
        report.append(f"   • 메타데이터 있음: {meta['chunks_with_metadata']}개")
        report.append(f"   • orig_elements 있음: {meta['chunks_with_orig_elements']}개")
        report.append(f"   • 완성도 점수: {meta['completeness_score']:.1f}%")

        report.append("\n" + "=" * 60)

        return "\n".join(report)


# 사용 예시
validator = PDFQualityValidator(pdf_chunks)

# 검증 실행
validation_results = validator.validate_all()

# 리포트 생성 및 출력
report = validator.generate_report()
print(report)

# 리포트 저장
with open("data/analyst_reports/quality_validation_report.txt", 'w', encoding='utf-8') as f:
    f.write(report)
```

**출력 예시:**
```
============================================================
📋 PDF 파싱 품질 검증 리포트
============================================================

📦 청크 분포 (총 127개)
   • CompositeElement: 78개
   • TableChunk: 23개
   • Title: 15개
   • Header: 11개

📊 테이블 품질 ✅ 양호
   • 총 테이블: 23개
   • 유효한 HTML: 22개
   • 무효한 HTML: 1개
   • 품질 점수: 95.7%

🖼️ 이미지 품질 ✅ 양호
   • 총 이미지: 45개
   • 유효한 Base64: 44개
   • 품질 점수: 97.8%

📝 텍스트 품질 ✅ 양호
   • 총 텍스트: 127개
   • 정상 길이: 98개
   • 짧은 텍스트: 25개
   • 품질 점수: 77.2%

🔍 메타데이터 완성도 ✅ 양호
   • 메타데이터 있음: 127개
   • orig_elements 있음: 98개
   • 완성도 점수: 100.0%

============================================================
```

**활용 시나리오:**
- PDF 파싱 후 품질 자동 검증
- 파싱 설정 최적화를 위한 데이터 수집
- 문제가 있는 PDF 파일 자동 필터링
- 파싱 파이프라인 모니터링

---

## 🎓 Part 1 요약

### 핵심 내용
1. **옵션 2 방식**: 이미지 요약 기반 텍스트 RAG (비용 효율적, 기존 인프라 활용)
2. **PDF 멀티모달 파싱**: unstructured 라이브러리로 텍스트, 이미지, 테이블 분리
3. **청크 타입 이해**: CompositeElement, TableChunk, Table 등의 구조와 메타데이터
4. **테이블 처리**: HTML → DataFrame → 마크다운 변환 파이프라인
5. **파싱 최적화**: 환경 변수 패딩 설정으로 테이블 이미지 품질 개선

### Part 2 예고
Part 2에서는 다음을 다룹니다:
1. **타입별 요약 생성**: Table/Text/Image 각각에 대한 LLM 요약 체인
2. **벡터 스토어 구축**: MultiVectorRetriever로 요약과 원본 분리 저장
3. **RAG 파이프라인**: 검색 및 답변 생성 전체 흐름
4. **실습 문제**: 페이지별 요약, Transformer 논문 RAG 구현

---

**다음 파일: PRJ04_W2_006_Multimodal_RAG_Financial_Part2_Part2.md**
