# PRJ04_W2_007: 멀티모달 RAG 구현 (금융 분석 보고서) - Part 3-1 (옵션 3)

## 🎯 학습 목표

이 실습을 완료하면 다음을 수행할 수 있습니다:

1. **옵션 3 방식 이해**: 이미지 요약 + 원본 참조 하이브리드 RAG 방식의 원리와 장단점을 설명할 수 있습니다
2. **PDF 멀티모달 파싱**: unstructured 라이브러리로 금융 보고서에서 텍스트, 이미지, 테이블을 분리할 수 있습니다
3. **청크 타입 분석**: CompositeElement, TableChunk 등 다양한 청크 타입의 구조와 메타데이터를 이해합니다
4. **하이브리드 전략 설계**: 텍스트 임베딩과 이미지 참조를 결합하는 전략을 설계할 수 있습니다
5. **타입별 데이터 처리**: Text/Table/Image 각 타입의 특성에 맞는 처리 방법을 구현할 수 있습니다

---

## 📚 핵심 개념

### 1. 멀티모달 RAG 3가지 옵션 비교

Part 3-1에서 구현할 **옵션 3**은 이미지 요약 + 원본 참조 하이브리드 방식입니다:

| 옵션 | 임베딩 방식 | 벡터 DB 저장 | 답변 생성 LLM | 이미지 활용 | 장점 | 단점 |
|---|---|---|---|---|---|---|
| **옵션 1** | 멀티모달 (CLIP) | 이미지 임베딩, 텍스트 임베딩 | Multimodal | 직접 활용 (base64) | 최고 이미지 활용도, 높은 답변 품질 | 높은 비용, base64 오버헤드 |
| **옵션 2** | 텍스트 (OpenAI) | 텍스트 임베딩 (이미지 요약) | Text-only | 텍스트 요약 간접 활용 | 비용 효율성, 텍스트 RAG 인프라 활용 | 이미지 정보 손실, 답변 품질 제한적 |
| **옵션 3** ⭐ | 텍스트 (OpenAI) | 텍스트 임베딩 (이미지 요약 + 참조) | Multimodal | 원본 이미지 참조 활용 | 옵션 1과 2의 절충, 이미지 정보 손실 감소 | 옵션 1보다 이미지 활용도 낮음, 이미지 참조 관리 필요 |

**선택 가이드:**
- **최고 품질 답변:** 옵션 1 (비용 고려)
- **비용 효율성 우선:** 옵션 2 (답변 품질 제한 감수)
- **품질과 효율성 균형:** 옵션 3 ⭐ (권장)

### 2. 옵션 3 파이프라인 구조

```
PDF 문서
    ↓
[unstructured 파싱]
    ↓
텍스트, 이미지, 테이블 분리
    ↓
[멀티모달 LLM으로 이미지 → 텍스트 요약]
    ↓
[텍스트 임베딩]
    ├─ 요약 → 벡터 DB 저장 (검색용)
    └─ 원본 이미지 참조 저장 (답변 생성용)
    ↓
[검색]
    ├─ 질의 → 요약 검색 (텍스트 임베딩)
    └─ 관련 원본 이미지 ID 추출
    ↓
[답변 생성]
    └─ 멀티모달 LLM (텍스트 컨텍스트 + 원본 이미지)
```

**핵심 특징:**
- 검색: 텍스트 요약으로 빠른 의미 검색
- 답변 생성: 원본 이미지로 고품질 답변
- 이미지 관리: 참조 ID로 원본 이미지 매칭

### 3. 옵션 3 vs 옵션 2 차이점

| 구분 | 옵션 2 | 옵션 3 |
|------|--------|--------|
| **벡터 DB 저장** | 이미지 요약 텍스트만 | 이미지 요약 + 원본 참조 ID |
| **검색 결과** | 텍스트 요약만 반환 | 텍스트 요약 + 원본 이미지 ID |
| **답변 생성 LLM** | Text-only | Multimodal (이미지 포함) |
| **답변 품질** | 요약에 의존 (정보 손실) | 원본 이미지 활용 (고품질) |
| **비용** | 낮음 | 중간 (검색은 저렴, 생성은 고비용) |

### 4. unstructured 청크 타입 복습

| 타입 | 설명 | 주요 속성 |
|------|------|-----------|
| **CompositeElement** | 여러 요소를 포함하는 복합 청크 | `.metadata.orig_elements` |
| **TableChunk** | 테이블 데이터 청크 | `.metadata.text_as_html` |
| **Table** | 개별 테이블 요소 | `.metadata.text_as_html`, `.metadata.image_base64` |
| **Image** | 이미지 요소 | `.metadata.image_base64` |
| **Text** | 순수 텍스트 요소 | `.text` 속성 |

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

# Langfuse (선택)
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
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

### 4. Langfuse 핸들러 설정 (선택)

```python
from langfuse.langchain import CallbackHandler

# 콜백 핸들러 생성
langfuse_handler = CallbackHandler()
```

---

## 📝 단계별 구현

### 1단계: 유틸리티 함수 정의

옵션 3에서 사용할 핵심 유틸리티 함수들:

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
- `is_base64()`: Base64 인코딩 여부 확인
- `split_image_text_types()`: 문서에서 이미지와 텍스트 분리
- `plt_img_base64()`: Base64 이미지 주피터에 표시

---

## 💻 PDF 문서 로딩 및 파싱

### 2단계: 데이터 로딩 및 파싱

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
```

**출력 예시:**
```
PDF files:
['data/analyst_reports/삼성전기_2024_1Q.pdf',
 'data/analyst_reports/SK하이닉스_2024_전망.pdf']
```

### 3단계: PDF 파티셔닝

```python
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
- `extract_image_block_to_payload=True`: 이미지를 base64로 메타데이터에 저장 (옵션 3의 핵심!)
- `strategy="hi_res"`: 고해상도 파싱으로 테이블과 이미지 정확도 향상
- `chunking_strategy="by_title"`: 의미 단위로 청킹

### 4단계: 타입별 분석

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
```

**출력 예시:**
```
📋 청크 타입 분포:
  • CompositeElement: 78개
  • TableChunk: 23개
  • Title: 15개
  • Header: 11개
```

### 5단계: 청크 내용 확인

```python
# 청크 내용 확인
for chunk in pdf_chunks[1:3]:
    print(chunk)
    print("=" * 100)
```

### 6단계: 청크 저장 및 로드

```python
# 청크 저장
import pickle

with open(os.path.join(data_path, "pdf_base64_chunks.pkl"), "wb") as f:
    pickle.dump(pdf_chunks, f)

print(f"💾 청크 저장 완료: {os.path.join(data_path, 'pdf_base64_chunks.pkl')}")
```

```python
# 청크 로드
import pickle

with open(os.path.join(data_path, "pdf_base64_chunks.pkl"), "rb") as f:
    pdf_chunks = pickle.load(f)

# 청크 개수 확인
print(f"📂 청크 로드 완료: {len(pdf_chunks)}개")
```

---

## 🔍 타입별 구조 확인

### 7단계: 청크 타입 확인

```python
# 각 청크 문서의 타입 확인
unique_types = set([str(type(el)) for el in pdf_chunks])
print("발견된 청크 타입:")
for t in unique_types:
    print(f"  • {t}")
```

### 8단계: CompositeElement 구조 분석

```python
# CompositeElement 확인
print("CompositeElement 예시:")
print(pdf_chunks[1])
print("\n" + "=" * 60 + "\n")

# CompositeElement는 하위 구성 요소를 포함하고 있음
print("하위 orig_elements:")
print(pdf_chunks[1].metadata.orig_elements)
```

**CompositeElement 특징:**
- 여러 요소(텍스트, 이미지, 테이블)를 포함하는 복합 청크
- `.metadata.orig_elements`로 하위 요소 리스트 접근
- 각 하위 요소는 독립적인 타입과 메타데이터 보유

### 9단계: Table 객체 처리

```python
# Table 객체 확인
tab = pdf_chunks[1].metadata.orig_elements[2]
print("Table 객체 전체 구조:")
print(tab.to_dict())
```

**Table 객체의 DataFrame 변환:**

```python
# Table 객체의 데이터프레임 변환
tab_df = pd.read_html(tab.metadata.text_as_html)[0]

# 데이터프레임 확인
display(tab_df)

# 데이터프레임을 마크다운 변환
tab_md = tab_df.to_markdown(index=False)
print("\n마크다운 형식:")
print(tab_md)
```

**이미지 확인:**

```python
# image_base64 이미지 확인
print("테이블 이미지:")
plt_img_base64(tab.metadata.image_base64)
```

### 10단계: TableChunk 분석

```python
# TableChunk 객체 확인
print("TableChunk 객체 탐색:")
for i, c in enumerate(pdf_chunks):
    if "TableChunk" in str(type(c)):
        print(f"\nTableChunk {i}:")
        print(c)
        print("=" * 100)
```

**TableChunk의 orig_elements 확인:**

```python
# TableChunk 객체 확인
print("TableChunk의 orig_elements:")
print(pdf_chunks[4].metadata.orig_elements)
```

**TableChunk에서 Table 추출:**

```python
# TableChunk에서 Table 객체 확인
# pdf_chunks[4]는 CompositeElement이므로 TableChunk인 pdf_chunks[8] 사용
tab = pdf_chunks[8].metadata.orig_elements[0]

# Table 객체 확인
print("TableChunk 내 Table 객체:")
print(tab.to_dict())
```

**DataFrame 변환:**

```python
# Table 객체의 데이터프레임 변환
# TableChunk의 첫 번째 Table 요소 사용
table_html = pdf_chunks[8].metadata.orig_elements[0].metadata.text_as_html
table_df = pd.read_html(table_html)[0]  # 첫 번째 테이블

# 데이터프레임 확인
display(table_df)
```

---

## 🎨 옵션 3 소개 및 타입별 구분

### 11단계: 옵션 3 개념 이해

**옵션 3의 핵심 전략:**

1. **멀티모달 LLM으로 이미지 → 텍스트 요약 생성**
2. **생성된 텍스트를 임베딩하고 필요할 때 검색**
3. **검색 시 텍스트 청크와 함께 원본 이미지를 가져옴**
4. **최종 답변 생성 단계에서 멀티모달 LLM에 텍스트와 원본 이미지를 함께 전달**

```
[검색 단계]
질의 → 텍스트 요약 검색 (빠름, 저비용)
    ↓
관련 요약 발견 → 원본 이미지 ID 추출

[답변 생성 단계]
텍스트 컨텍스트 + 원본 이미지 → 멀티모달 LLM
    ↓
고품질 답변 생성
```

### 12단계: 타입별 구분

옵션 3의 첫 단계는 청크를 타입별로 분류하는 것입니다:

```python
# 타입별 분류
tables = []
texts = []
images = []

for chunk in pdf_chunks:
    # 테이블 청크
    if "Table" in str(type(chunk)):
        tables.append(chunk)

    # 텍스트 청크 (CompositeElement)
    elif "CompositeElement" in str(type(chunk)):
        texts.append(chunk)

# 이미지 추출 (metadata에서)
for chunk in pdf_chunks:
    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
        for el in chunk.metadata.orig_elements:
            # 이미지 요소가 있는 경우만 추출
            if hasattr(el.metadata, 'image_base64') and el.metadata.image_base64:
                images.append(el.metadata.image_base64)

print(f"📊 테이블 청크: {len(tables)}개")
print(f"📝 텍스트 청크: {len(texts)}개")
print(f"🖼️ 이미지: {len(images)}개")
```

**출력 예시:**
```
📊 테이블 청크: 23개
📝 텍스트 청크: 78개
🖼️ 이미지: 45개
```

### 13단계: 분류 결과 확인

```python
# 테이블 청크 확인
print("테이블 청크 예시:")
print(tables[0])
print("\n" + "=" * 60 + "\n")

# 텍스트 청크 확인
print("텍스트 청크 예시:")
print(texts[0])
print("\n" + "=" * 60 + "\n")

# 이미지 확인
print("이미지 예시:")
plt_img_base64(images[0])
```

---

## 🎯 실습 문제

### 실습 1 (기본): 타입별 통계 분석

**문제:**
분류된 tables, texts, images에 대해 다음 통계를 계산하세요:

1. 각 타입의 평균 콘텐츠 길이
2. 가장 긴 콘텐츠와 가장 짧은 콘텐츠
3. 이미지의 평균 Base64 크기

**힌트:**
```python
# 텍스트 길이 계산
text_lengths = [len(t.text) for t in texts if hasattr(t, 'text')]

# Base64 크기 계산
image_sizes = [len(img) for img in images]
```

### 실습 2 (중급): 이미지-텍스트 매칭 시스템

**문제:**
각 이미지가 어떤 청크에서 왔는지 추적하는 시스템을 구현하세요:

**요구사항:**
1. 이미지와 원본 청크를 매핑하는 딕셔너리 생성
2. 이미지의 메타데이터 (페이지 번호, 청크 인덱스 등) 추출
3. 특정 이미지로부터 원본 청크를 찾는 함수 구현

```python
class ImageChunkMapper:
    def __init__(self, chunks):
        self.chunks = chunks
        self.image_map = {}

    def build_map(self):
        """이미지와 청크를 매핑"""
        pass

    def get_chunk_for_image(self, image_base64: str) -> dict:
        """이미지의 원본 청크 정보 반환"""
        pass

    def get_images_by_page(self, page_number: int) -> list:
        """특정 페이지의 모든 이미지 반환"""
        pass
```

### 실습 3 (고급): 스마트 청크 필터링 시스템

**문제:**
품질이 낮은 청크를 자동으로 필터링하는 시스템을 구현하세요:

**필터링 기준:**
1. 텍스트가 너무 짧은 청크 (< 50자)
2. 테이블이 비어있거나 형식이 잘못된 청크
3. 이미지 크기가 너무 작은 청크 (< 10KB)
4. 중복 콘텐츠 제거

```python
class ChunkQualityFilter:
    def __init__(self, texts, tables, images, min_text_length=50, min_image_size=10*1024):
        self.texts = texts
        self.tables = tables
        self.images = images
        self.min_text_length = min_text_length
        self.min_image_size = min_image_size

    def filter_texts(self) -> list:
        """품질 기준을 통과한 텍스트만 반환"""
        pass

    def filter_tables(self) -> list:
        """유효한 테이블만 반환"""
        pass

    def filter_images(self) -> list:
        """품질 기준을 통과한 이미지만 반환"""
        pass

    def get_quality_report(self) -> dict:
        """필터링 통계 반환"""
        pass
```

---

## 💡 솔루션 예시

### 솔루션 1: 타입별 통계 분석

```python
import numpy as np
from typing import Dict

def analyze_chunk_statistics(texts, tables, images) -> Dict:
    """타입별 청크 통계 분석"""

    # 1. 텍스트 통계
    text_lengths = [len(t.text) for t in texts if hasattr(t, 'text') and t.text]

    text_stats = {
        'count': len(texts),
        'avg_length': np.mean(text_lengths) if text_lengths else 0,
        'max_length': max(text_lengths) if text_lengths else 0,
        'min_length': min(text_lengths) if text_lengths else 0,
        'std_length': np.std(text_lengths) if text_lengths else 0
    }

    # 2. 테이블 통계
    table_lengths = []
    for t in tables:
        if hasattr(t, 'text') and t.text:
            table_lengths.append(len(t.text))
        elif hasattr(t, 'metadata') and hasattr(t.metadata, 'text_as_html'):
            if t.metadata.text_as_html:
                table_lengths.append(len(t.metadata.text_as_html))

    table_stats = {
        'count': len(tables),
        'avg_length': np.mean(table_lengths) if table_lengths else 0,
        'max_length': max(table_lengths) if table_lengths else 0,
        'min_length': min(table_lengths) if table_lengths else 0,
        'std_length': np.std(table_lengths) if table_lengths else 0
    }

    # 3. 이미지 통계
    image_sizes = [len(img) for img in images]

    image_stats = {
        'count': len(images),
        'avg_size_bytes': np.mean(image_sizes) if image_sizes else 0,
        'avg_size_kb': np.mean(image_sizes) / 1024 if image_sizes else 0,
        'max_size_bytes': max(image_sizes) if image_sizes else 0,
        'min_size_bytes': min(image_sizes) if image_sizes else 0,
        'total_size_mb': sum(image_sizes) / (1024 * 1024) if image_sizes else 0
    }

    return {
        'texts': text_stats,
        'tables': table_stats,
        'images': image_stats
    }


# 사용 예시
stats = analyze_chunk_statistics(texts, tables, images)

print("=" * 60)
print("📊 청크 통계 분석 결과")
print("=" * 60)

print(f"\n📝 텍스트 청크:")
print(f"  총 개수: {stats['texts']['count']}개")
print(f"  평균 길이: {stats['texts']['avg_length']:.0f}자")
print(f"  최대 길이: {stats['texts']['max_length']}자")
print(f"  최소 길이: {stats['texts']['min_length']}자")
print(f"  표준편차: {stats['texts']['std_length']:.0f}자")

print(f"\n📊 테이블 청크:")
print(f"  총 개수: {stats['tables']['count']}개")
print(f"  평균 길이: {stats['tables']['avg_length']:.0f}자")
print(f"  최대 길이: {stats['tables']['max_length']}자")
print(f"  최소 길이: {stats['tables']['min_length']}자")

print(f"\n🖼️ 이미지:")
print(f"  총 개수: {stats['images']['count']}개")
print(f"  평균 크기: {stats['images']['avg_size_kb']:.1f} KB")
print(f"  최대 크기: {stats['images']['max_size_bytes'] / 1024:.1f} KB")
print(f"  최소 크기: {stats['images']['min_size_bytes'] / 1024:.1f} KB")
print(f"  전체 크기: {stats['images']['total_size_mb']:.2f} MB")
```

**출력 예시:**
```
============================================================
📊 청크 통계 분석 결과
============================================================

📝 텍스트 청크:
  총 개수: 78개
  평균 길이: 542자
  최대 길이: 1198자
  최소 길이: 87자
  표준편차: 298자

📊 테이블 청크:
  총 개수: 23개
  평균 길이: 856자
  최대 길이: 2145자
  최소 길이: 234자

🖼️ 이미지:
  총 개수: 45개
  평균 크기: 127.3 KB
  최대 크기: 456.2 KB
  최소 크기: 12.8 KB
  전체 크기: 5.73 MB
```

### 솔루션 2: 이미지-텍스트 매칭 시스템

```python
from typing import Dict, List, Optional


class ImageChunkMapper:
    """이미지와 원본 청크를 매핑하는 시스템"""

    def __init__(self, chunks):
        self.chunks = chunks
        self.image_map = {}  # {image_hash: {chunk_info}}
        self.page_images = {}  # {page_number: [image_hashes]}

    def _hash_image(self, image_base64: str) -> str:
        """이미지 해시 생성 (처음 100자 + 마지막 100자)"""
        if len(image_base64) > 200:
            return image_base64[:100] + image_base64[-100:]
        return image_base64

    def build_map(self):
        """이미지와 청크를 매핑"""
        for chunk_idx, chunk in enumerate(self.chunks):
            if not (hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements')):
                continue

            # 청크 메타데이터 추출
            chunk_metadata = {
                'chunk_idx': chunk_idx,
                'chunk_type': str(type(chunk)).split("'")[1],
                'page_number': getattr(chunk.metadata, 'page_number', None),
                'filename': getattr(chunk.metadata, 'filename', None),
            }

            # orig_elements에서 이미지 추출
            for el_idx, el in enumerate(chunk.metadata.orig_elements):
                if not (hasattr(el, 'metadata') and hasattr(el.metadata, 'image_base64')):
                    continue

                image_b64 = el.metadata.image_base64
                if not image_b64:
                    continue

                # 이미지 해시 생성
                img_hash = self._hash_image(image_b64)

                # 이미지 정보 저장
                self.image_map[img_hash] = {
                    **chunk_metadata,
                    'element_idx': el_idx,
                    'image_base64': image_b64,
                    'image_size': len(image_b64),
                    'element_type': str(type(el)).split("'")[1]
                }

                # 페이지별 이미지 인덱스
                page_num = chunk_metadata['page_number']
                if page_num not in self.page_images:
                    self.page_images[page_num] = []
                self.page_images[page_num].append(img_hash)

        print(f"✅ 매핑 완료: {len(self.image_map)}개 이미지")
        print(f"   페이지 수: {len(self.page_images)}개")

    def get_chunk_for_image(self, image_base64: str) -> Optional[Dict]:
        """이미지의 원본 청크 정보 반환"""
        img_hash = self._hash_image(image_base64)
        return self.image_map.get(img_hash)

    def get_images_by_page(self, page_number: int) -> List[Dict]:
        """특정 페이지의 모든 이미지 반환"""
        if page_number not in self.page_images:
            return []

        return [
            self.image_map[img_hash]
            for img_hash in self.page_images[page_number]
        ]

    def get_summary(self) -> Dict:
        """매핑 요약 정보 반환"""
        return {
            'total_images': len(self.image_map),
            'total_pages': len(self.page_images),
            'images_per_page': {
                page: len(imgs) for page, imgs in self.page_images.items()
            }
        }


# 사용 예시
mapper = ImageChunkMapper(pdf_chunks)
mapper.build_map()

# 요약 정보
summary = mapper.get_summary()
print(f"\n📊 매핑 요약:")
print(f"  총 이미지: {summary['total_images']}개")
print(f"  총 페이지: {summary['total_pages']}개")

# 특정 이미지의 원본 찾기
if images:
    sample_image = images[0]
    chunk_info = mapper.get_chunk_for_image(sample_image)
    if chunk_info:
        print(f"\n🔍 샘플 이미지 정보:")
        print(f"  청크 인덱스: {chunk_info['chunk_idx']}")
        print(f"  청크 타입: {chunk_info['chunk_type']}")
        print(f"  페이지 번호: {chunk_info['page_number']}")
        print(f"  파일명: {chunk_info['filename']}")
        print(f"  이미지 크기: {chunk_info['image_size'] / 1024:.1f} KB")

# 특정 페이지의 이미지들
page_5_images = mapper.get_images_by_page(5)
print(f"\n📄 페이지 5의 이미지: {len(page_5_images)}개")
for idx, img_info in enumerate(page_5_images):
    print(f"  이미지 {idx+1}: 크기 {img_info['image_size'] / 1024:.1f} KB")
```

**출력 예시:**
```
✅ 매핑 완료: 45개 이미지
   페이지 수: 18개

📊 매핑 요약:
  총 이미지: 45개
  총 페이지: 18개

🔍 샘플 이미지 정보:
  청크 인덱스: 3
  청크 타입: CompositeElement
  페이지 번호: 2
  파일명: 삼성전기_2024_1Q.pdf
  이미지 크기: 145.3 KB

📄 페이지 5의 이미지: 3개
  이미지 1: 크기 98.7 KB
  이미지 2: 크기 156.2 KB
  이미지 3: 크기 72.4 KB
```

### 솔루션 3: 스마트 청크 필터링 시스템

```python
import hashlib
from typing import List, Dict


class ChunkQualityFilter:
    """청크 품질 필터링 시스템"""

    def __init__(self, texts, tables, images, min_text_length=50, min_image_size=10*1024):
        self.texts = texts
        self.tables = tables
        self.images = images
        self.min_text_length = min_text_length
        self.min_image_size = min_image_size

        self.filtered_texts = []
        self.filtered_tables = []
        self.filtered_images = []

        self.stats = {
            'texts': {'total': len(texts), 'filtered': 0, 'reasons': {}},
            'tables': {'total': len(tables), 'filtered': 0, 'reasons': {}},
            'images': {'total': len(images), 'filtered': 0, 'reasons': {}}
        }

    def _record_filter(self, category: str, reason: str):
        """필터링 이유 기록"""
        if reason not in self.stats[category]['reasons']:
            self.stats[category]['reasons'][reason] = 0
        self.stats[category]['reasons'][reason] += 1
        self.stats[category]['filtered'] += 1

    def filter_texts(self) -> List:
        """품질 기준을 통과한 텍스트만 반환"""
        seen_hashes = set()

        for text in self.texts:
            # 텍스트 추출
            if hasattr(text, 'text'):
                content = text.text.strip()
            else:
                self._record_filter('texts', 'no_text_attribute')
                continue

            # 1. 길이 체크
            if len(content) < self.min_text_length:
                self._record_filter('texts', 'too_short')
                continue

            # 2. 중복 체크
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash in seen_hashes:
                self._record_filter('texts', 'duplicate')
                continue

            seen_hashes.add(content_hash)
            self.filtered_texts.append(text)

        return self.filtered_texts

    def filter_tables(self) -> List:
        """유효한 테이블만 반환"""
        seen_hashes = set()

        for table in self.tables:
            # 테이블 HTML 추출
            try:
                if hasattr(table, 'metadata') and hasattr(table.metadata, 'orig_elements'):
                    if table.metadata.orig_elements:
                        table_html = table.metadata.orig_elements[0].metadata.text_as_html
                    else:
                        self._record_filter('tables', 'no_orig_elements')
                        continue
                else:
                    self._record_filter('tables', 'no_metadata')
                    continue

                # 1. HTML 존재 체크
                if not table_html or table_html.strip() == '':
                    self._record_filter('tables', 'empty_html')
                    continue

                # 2. DataFrame 변환 가능 체크
                try:
                    df = pd.read_html(table_html)[0]

                    # 3. 테이블 크기 체크 (최소 2행 2열)
                    if df.shape[0] < 2 or df.shape[1] < 2:
                        self._record_filter('tables', 'too_small')
                        continue

                    # 4. 중복 체크
                    table_hash = hashlib.md5(table_html.encode()).hexdigest()
                    if table_hash in seen_hashes:
                        self._record_filter('tables', 'duplicate')
                        continue

                    seen_hashes.add(table_hash)
                    self.filtered_tables.append(table)

                except Exception:
                    self._record_filter('tables', 'invalid_html')
                    continue

            except Exception as e:
                self._record_filter('tables', f'error: {str(e)[:20]}')
                continue

        return self.filtered_tables

    def filter_images(self) -> List:
        """품질 기준을 통과한 이미지만 반환"""
        seen_hashes = set()

        for image in self.images:
            # 1. 크기 체크
            image_size = len(image)
            if image_size < self.min_image_size:
                self._record_filter('images', 'too_small')
                continue

            # 2. Base64 유효성 체크
            try:
                import base64
                base64.b64decode(image)
            except Exception:
                self._record_filter('images', 'invalid_base64')
                continue

            # 3. 중복 체크
            image_hash = hashlib.md5(image.encode()).hexdigest()
            if image_hash in seen_hashes:
                self._record_filter('images', 'duplicate')
                continue

            seen_hashes.add(image_hash)
            self.filtered_images.append(image)

        return self.filtered_images

    def filter_all(self):
        """모든 타입 필터링 실행"""
        self.filter_texts()
        self.filter_tables()
        self.filter_images()

    def get_quality_report(self) -> Dict:
        """필터링 통계 반환"""
        report = {
            'texts': {
                'original': self.stats['texts']['total'],
                'filtered_out': self.stats['texts']['filtered'],
                'passed': len(self.filtered_texts),
                'pass_rate': len(self.filtered_texts) / self.stats['texts']['total'] * 100 if self.stats['texts']['total'] > 0 else 0,
                'reasons': self.stats['texts']['reasons']
            },
            'tables': {
                'original': self.stats['tables']['total'],
                'filtered_out': self.stats['tables']['filtered'],
                'passed': len(self.filtered_tables),
                'pass_rate': len(self.filtered_tables) / self.stats['tables']['total'] * 100 if self.stats['tables']['total'] > 0 else 0,
                'reasons': self.stats['tables']['reasons']
            },
            'images': {
                'original': self.stats['images']['total'],
                'filtered_out': self.stats['images']['filtered'],
                'passed': len(self.filtered_images),
                'pass_rate': len(self.filtered_images) / self.stats['images']['total'] * 100 if self.stats['images']['total'] > 0 else 0,
                'reasons': self.stats['images']['reasons']
            }
        }
        return report


# 사용 예시
filter_system = ChunkQualityFilter(
    texts=texts,
    tables=tables,
    images=images,
    min_text_length=50,
    min_image_size=10*1024  # 10KB
)

# 모든 타입 필터링
filter_system.filter_all()

# 품질 리포트
report = filter_system.get_quality_report()

print("=" * 60)
print("🔍 청크 품질 필터링 결과")
print("=" * 60)

print(f"\n📝 텍스트:")
print(f"  원본: {report['texts']['original']}개")
print(f"  통과: {report['texts']['passed']}개")
print(f"  제외: {report['texts']['filtered_out']}개")
print(f"  통과율: {report['texts']['pass_rate']:.1f}%")
if report['texts']['reasons']:
    print(f"  제외 이유:")
    for reason, count in report['texts']['reasons'].items():
        print(f"    - {reason}: {count}개")

print(f"\n📊 테이블:")
print(f"  원본: {report['tables']['original']}개")
print(f"  통과: {report['tables']['passed']}개")
print(f"  제외: {report['tables']['filtered_out']}개")
print(f"  통과율: {report['tables']['pass_rate']:.1f}%")
if report['tables']['reasons']:
    print(f"  제외 이유:")
    for reason, count in report['tables']['reasons'].items():
        print(f"    - {reason}: {count}개")

print(f"\n🖼️ 이미지:")
print(f"  원본: {report['images']['original']}개")
print(f"  통과: {report['images']['passed']}개")
print(f"  제외: {report['images']['filtered_out']}개")
print(f"  통과율: {report['images']['pass_rate']:.1f}%")
if report['images']['reasons']:
    print(f"  제외 이유:")
    for reason, count in report['images']['reasons'].items():
        print(f"    - {reason}: {count}개")

# 필터링된 데이터 사용
print(f"\n✅ 최종 데이터:")
print(f"  텍스트: {len(filter_system.filtered_texts)}개")
print(f"  테이블: {len(filter_system.filtered_tables)}개")
print(f"  이미지: {len(filter_system.filtered_images)}개")
```

**출력 예시:**
```
============================================================
🔍 청크 품질 필터링 결과
============================================================

📝 텍스트:
  원본: 78개
  통과: 72개
  제외: 6개
  통과율: 92.3%
  제외 이유:
    - too_short: 5개
    - duplicate: 1개

📊 테이블:
  원본: 23개
  통과: 21개
  제외: 2개
  통과율: 91.3%
  제외 이유:
    - empty_html: 1개
    - too_small: 1개

🖼️ 이미지:
  원본: 45개
  통과: 42개
  제외: 3개
  통과율: 93.3%
  제외 이유:
    - too_small: 2개
    - duplicate: 1개

✅ 최종 데이터:
  텍스트: 72개
  테이블: 21개
  이미지: 42개
```

---

## 🎓 Part 3-1 요약

### 핵심 내용
1. **옵션 3 개념**: 이미지 요약 + 원본 참조 하이브리드 방식
2. **PDF 파싱**: unstructured로 금융 보고서 멀티모달 추출
3. **청크 타입 이해**: CompositeElement, TableChunk, Table 구조
4. **타입별 분류**: Text/Table/Image 각각 분리 및 검증
5. **품질 관리**: 통계 분석, 매핑, 필터링 시스템

### 옵션 3의 특징
- ✅ **균형잡힌 접근**: 검색 효율성(텍스트) + 답변 품질(이미지)
- ✅ **비용 최적화**: 검색 단계는 저비용, 생성 단계만 고비용
- ✅ **정보 보존**: 원본 이미지로 시각적 정보 완전 활용
- ⚠️ **복잡성**: 이미지 참조 관리 시스템 필요
- ⚠️ **저장 공간**: 요약 + 원본 이미지 모두 저장

### Part 3-2 예고
Part 3-2에서는 다음을 다룹니다:
1. **이미지/표 요약 생성**: 멀티모달 LLM으로 텍스트 요약
2. **벡터스토어 구축**: 요약 + 원본 참조 ID 저장
3. **RAG 파이프라인**: 검색 + 멀티모달 답변 생성
4. **실습 문제**: DoclingLoader, 실전 프로젝트

---

**다음 파일: PRJ04_W2_007_Multimodal_RAG_Financial_Part3_Part2.md**
