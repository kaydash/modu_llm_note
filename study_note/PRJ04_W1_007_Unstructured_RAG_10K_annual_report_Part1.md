# PRJ04_W1_007: Unstructured & LangChain을 활용한 10-K 보고서 RAG 시스템 구축 - Part 1

## 📚 학습 목표

이 가이드를 완료하면 다음을 수행할 수 있습니다:

1. **10-K 연례 보고서 구조 이해**: 미국 상장기업의 공식 재무 보고서 형식 파악
2. **Unstructured 고급 파싱 기법**: hi_res 전략을 활용한 정밀 문서 분석
3. **섹션 기반 문서 재구조화**: 목차(TOC)를 활용한 의미론적 문서 그룹화
4. **청킹 전략 최적화**: 문서 유형과 길이에 따른 적정 청크 크기 결정
5. **표 데이터 변환**: HTML 형식의 표를 마크다운으로 변환하여 LLM 친화적 형태로 처리

## 🎯 핵심 개념

### 10-K 보고서 (10-K Annual Report)
- **정의**: 미국 증권거래위원회(SEC)에 제출하는 연례 재무 보고서
- **특징**:
  - 표준화된 섹션 구조 (Business, Risk Factors, Financial Data 등)
  - 100~200 페이지 분량의 대규모 문서
  - 텍스트, 표, 그래프가 혼합된 비정형 데이터
- **RAG 활용 가치**:
  - 투자자 질의응답
  - 재무 분석 자동화
  - 리스크 평가 시스템

### Unstructured Hi-Res 전략
- **hi_res 모드**: YOLO 기반 객체 탐지를 활용한 고정밀 파싱
- **장점**:
  - 표 구조 정확하게 인식 (`infer_table_structure=True`)
  - 이미지 블록 추출 (`extract_images_in_pdf=True`)
  - 요소 타입 자동 분류 (Title, Table, NarrativeText 등)
- **Trade-off**: 처리 시간 증가 (fast 대비 3~5배)

### 섹션 기반 재구조화
- **목적**: 문서의 의미론적 단위로 청킹하여 검색 정확도 향상
- **방법**:
  1. 목차 페이지에서 섹션 항목 추출 (LLM 활용)
  2. 문서 전체를 순회하며 섹션 헤더 매칭
  3. 각 청크에 `section` 메타데이터 추가
- **효과**: 특정 섹션 필터링 검색 가능 (예: "Risk Factors만 검색")

### 청킹 전략
- **도전 과제**: 10-K 보고서는 섹션별로 길이가 매우 불균형
  - Business 섹션: ~3,000 토큰
  - Financial Statements: ~15,000 토큰
- **해결책**:
  - 섹션별 토큰 수 분석
  - 3,000 토큰 이상 섹션만 분할
  - `order` 메타데이터로 순서 유지

## 🛠 환경 설정

### 1. 필수 라이브러리 설치

```bash
# Unstructured 및 PDF 처리
pip install unstructured unstructured[pdf]
pip install "unstructured-inference"
pip install pdfminer.six pdf2image pytesseract pillow

# LangChain 생태계
pip install langchain langchain-community langchain-openai
pip install langchain-chroma langchain-text-splitters

# 데이터 처리 및 시각화
pip install pandas matplotlib tiktoken

# 환경 변수 관리
pip install python-dotenv
```

### 2. 환경 변수 설정

`.env` 파일을 생성하고 OpenAI API 키를 설정합니다:

```bash
OPENAI_API_KEY=your_api_key_here
```

### 3. 프로젝트 구조

```
project/
├── data/
│   ├── tsla-20241231-gen.pdf          # 테슬라 10-K 보고서
│   ├── tesla_10k.pkl                  # 파싱된 문서 객체
│   ├── tesla_10k_toc.pkl             # 목차 기반 재구조화 문서
│   ├── tesla_10k_sections.pkl        # 섹션별 결합 문서
│   └── images/
│       └── tesla_10k/                 # 추출된 이미지
├── chroma_db/                         # Chroma 벡터 DB
└── document_store/                    # 부모 문서 저장소
```

## 📖 단계별 구현

### 1단계: 문서 로드 및 기본 분석

#### PyPDFLoader를 사용한 페이지 단위 로드

```python
from dotenv import load_dotenv
load_dotenv()

import os
from langchain_community.document_loaders import PyPDFLoader
import logging

# 로깅 설정 (불필요한 경고 제거)
logging.getLogger('pdfminer').setLevel(logging.ERROR)
logging.getLogger('unstructured').setLevel(logging.ERROR)

# PDF 파일 경로
file_path = "data/tsla-20241231-gen.pdf"

# PyPDFLoader 초기화
loader = PyPDFLoader(file_path)

# 문서 로드
pypdf_docs = loader.load()

print(f"✅ 로드된 페이지 수: {len(pypdf_docs)}")
```

**출력 예시:**
```
✅ 로드된 페이지 수: 144
```

#### 첫 페이지 내용 확인

```python
from pprint import pprint

# 첫 페이지 내용 출력
print(f"{pypdf_docs[0].page_content[:500]}\n")
print("-" * 100)

# 메타데이터 확인
pprint(pypdf_docs[0].metadata)
```

**메타데이터 구조:**
```python
{
 'creationdate': '2025-01-30T11:11:07+00:00',
 'creator': 'wkhtmltopdf 0.12.6',
 'page': 0,
 'page_label': '1',
 'producer': 'Qt 5.15.2',
 'source': 'data/tsla-20241231-gen.pdf',
 'title': '',
 'total_pages': 144
}
```

**💡 핵심 포인트:**
- `page`: 0부터 시작하는 페이지 번호
- `page_label`: 문서 내 실제 페이지 표시 (목차, 로마숫자 등)
- `total_pages`: 전체 페이지 수

### 2단계: Unstructured를 활용한 고급 파싱

#### Hi-Res 전략으로 문서 파싱

```python
from langchain_community.document_loaders import UnstructuredFileLoader
from unstructured.cleaners.core import (
    clean_extra_whitespace,
    replace_unicode_quotes,
    clean_non_ascii_chars,
    group_broken_paragraphs
)

# 이미지 저장 폴더 생성
image_folder = "data/images/tesla_10k"
os.makedirs(image_folder, exist_ok=True)

# UnstructuredFileLoader 설정
loader = UnstructuredFileLoader(
    file_path,

    # 파티셔닝 전략 설정
    strategy="hi_res",                  # 고정밀 모드
    hi_res_model_name="yolox",          # YOLO 객체 탐지 모델
    infer_table_structure=True,         # 표 구조 인식
    languages=["eng"],                  # 언어 설정

    # 이미지 추출 설정
    extract_images_in_pdf=True,
    extract_image_block_types=["Image", "Table"],
    extract_image_block_output_dir=image_folder,

    # 후처리 설정
    post_processors=[
        clean_extra_whitespace,  # 불필요한 공백 제거
        replace_unicode_quotes,  # 유니코드 따옴표 정규화
        clean_non_ascii_chars,   # 비 ASCII 문자 제거
        group_broken_paragraphs, # 줄바꿈 문단 결합
    ],
)

# Lazy loading으로 메모리 효율 향상
docs = []
for doc in loader.lazy_load():
    docs.append(doc)

print(f"✅ 파싱된 요소 개수: {len(docs)}")
```

**⚠️ 주의사항:**
- `UnstructuredFileLoader`는 LangChain 0.2.8 이후 deprecated
- 새 프로젝트에서는 `langchain-unstructured` 패키지의 `UnstructuredLoader` 사용 권장
- 이 가이드에서는 기존 코드와의 호환성을 위해 `UnstructuredFileLoader` 사용

#### 문서 요소 확인

```python
# 처음 5개 요소 출력
for doc in docs[:5]:
    pprint(doc.page_content)
    print("-" * 100)
    pprint(doc.metadata)
    print("=" * 100)
    print()
```

**메타데이터 예시:**
```python
{
 'category': 'Title',
 'element_id': '5c8e9f2a1b3d4e6f7a8b9c0d1e2f3a4b',
 'languages': ['eng'],
 'page_number': 1,
 'parent_id': 'a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6',
 'source': 'data/tsla-20241231-gen.pdf',
 'text_as_html': None
}
```

**category 타입:**
- `Title`: 제목 (섹션 헤더)
- `NarrativeText`: 일반 텍스트
- `Table`: 표 데이터
- `Header`: 페이지 머리글
- `Footer`: 페이지 바닥글
- `Image`: 이미지

#### 문서 객체 저장 (재사용을 위해)

```python
import pickle

# pickle 형식으로 저장
with open("data/tesla_10k.pkl", "wb") as f:
    pickle.dump(docs, f)

print("✅ 문서 객체 저장 완료")
```

#### 저장된 문서 불러오기

```python
# pickle 파일 로드
with open("data/tesla_10k.pkl", "rb") as f:
    pickled_docs = pickle.load(f)

print(f"✅ 로드된 문서 수: {len(pickled_docs)}")
print(f"\n첫 번째 요소 내용:\n{pickled_docs[0].page_content}")
```

### 3단계: 표 데이터 변환

#### 요소 타입별 분포 확인

```python
# 카테고리별 문서 개수 집계
category_counts = {}

for doc in docs:
    category = doc.metadata["category"]
    if category in category_counts:
        category_counts[category] += 1
    else:
        category_counts[category] = 1

# 결과 출력
pprint(category_counts)
```

**출력 예시:**
```python
{
 'Title': 45,
 'NarrativeText': 312,
 'Table': 67,
 'Header': 144,
 'Footer': 144,
 'Image': 8
}
```

#### 표를 마크다운으로 변환

10-K 보고서의 재무 데이터는 대부분 표 형식입니다. LLM이 표를 효과적으로 이해하려면 마크다운 형식으로 변환해야 합니다.

```python
import pandas as pd
from langchain_core.documents import Document

new_docs = []

for doc in docs:

    # Table 카테고리 처리
    if doc.metadata["category"] == "Table":
        # HTML 형식의 표를 판다스로 읽기
        _df = pd.read_html(doc.metadata['text_as_html'])[0]

        # 마크다운 형식으로 변환
        _md = _df.to_markdown(index=False)

        # 새로운 문서 객체 생성
        new_docs.append(Document(page_content=_md, metadata=doc.metadata))

    # Header, Footer, Image는 제외 (노이즈 제거)
    elif doc.metadata["category"] in ["Header", "Footer", "Image"]:
        continue

    # 나머지는 그대로 유지
    else:
        new_docs.append(doc)

print(f"✅ 변환 완료: {len(new_docs)}개 문서")
```

**변환 전 (HTML):**
```html
<table>
  <tr><th>Year</th><th>Revenue</th></tr>
  <tr><td>2024</td><td>$97.7B</td></tr>
</table>
```

**변환 후 (Markdown):**
```markdown
| Year | Revenue |
|------|---------|
| 2024 | $97.7B  |
```

**💡 핵심 포인트:**
- LLM은 마크다운 표를 훨씬 잘 이해합니다
- Header/Footer는 검색 노이즈를 발생시키므로 제거
- Image는 별도 처리가 필요하므로 일단 제외

### 4단계: 텍스트 청킹 전략 수립

#### 문서 길이 분포 분석

청킹 전에 문서의 길이 분포를 파악하여 적정 청크 크기를 결정합니다.

```python
import matplotlib.pyplot as plt

# 각 문서의 길이 계산
doc_lengths = [len(doc.page_content) for doc in new_docs]

# 히스토그램 시각화
plt.figure(figsize=(10, 6))
plt.hist(doc_lengths, bins=50, edgecolor='black')
plt.xlabel('Document Length (characters)')
plt.ylabel('Frequency')
plt.title('Distribution of Document Lengths')
plt.show()
```

#### 카테고리별 길이 분석

```python
# 카테고리별로 길이 분석
category_lengths = {}
for doc in new_docs:
    category = doc.metadata["category"]
    if category not in category_lengths:
        category_lengths[category] = []
    category_lengths[category].append(len(doc.page_content))

# 카테고리별 시각화
for category, lengths in category_lengths.items():
    plt.figure(figsize=(10, 4))
    plt.hist(lengths, bins=50, edgecolor='black')
    plt.title(f"Document Lengths for {category}")
    plt.xlabel("Length (characters)")
    plt.ylabel("Frequency")
    plt.show()
```

#### 토큰 수 기준 분석

실제 LLM에서는 문자 수가 아닌 토큰 수가 중요합니다.

```python
import tiktoken

# tiktoken 인코더 초기화 (GPT-4 기준)
tokenizer = tiktoken.get_encoding("cl100k_base")

# 카테고리별 토큰 수 분석
category_tokens = {}
for doc in new_docs:
    category = doc.metadata["category"]
    if category not in category_tokens:
        category_tokens[category] = []

    # 토큰 수 계산
    tokens = len(tokenizer.encode(doc.page_content))
    category_tokens[category].append(tokens)

# 시각화
for category, tokens in category_tokens.items():
    plt.figure(figsize=(10, 4))
    plt.hist(tokens, bins=50, edgecolor='black')
    plt.title(f"Token Distribution for {category}")
    plt.xlabel("Tokens")
    plt.ylabel("Frequency")
    plt.axvline(x=1000, color='red', linestyle='--', label='Max chunk size')
    plt.legend()
    plt.show()
```

**💡 청킹 전략 결정:**
- **NarrativeText**: 평균 100~300 토큰 → 그대로 유지
- **Table**: 평균 200~500 토큰 → 그대로 유지
- **Title**: 평균 10~50 토큰 → 다음 문단과 결합 고려

### 5단계: 목차 기반 섹션 추출

10-K 보고서는 표준화된 섹션 구조를 가지고 있습니다. 목차를 추출하여 문서를 의미론적으로 재구조화합니다.

#### 목차 페이지에서 항목 추출

```python
# 3페이지가 목차인 경우 (문서마다 다를 수 있음)
toc_items = [
    doc.page_content
    for doc in new_docs
    if doc.metadata["page_number"] == 3
]

print(f"✅ 목차 항목 수: {len(toc_items)}")
for item in toc_items[:5]:
    print(f"- {item}")
```

#### LLM을 활용한 목차 구조화

목차 항목을 LLM으로 파싱하여 구조화합니다.

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# 목차 항목 모델 정의
class Item(BaseModel):
    number: str = Field(description="목차 항목 번호 (예: 1, 1A, 1B, 2, 3 등)")
    title: str = Field(description="목차 항목 제목")

class Section(BaseModel):
    section: str = Field(description="목차 항목 그룹")
    items: list[Item] = Field(description="목차 항목 리스트")

# 프롬프트 템플릿
TOC_PROMPT = """
You are a helpful assistant.
The user will provide you with a list of items.

Your task is to group these items into sections based on their content.

Please provide the output in JSON format.
The JSON should contain the following fields:
- section: The name of the section
- items: A list of items that belong to this section

The items are as follows:
{items}
"""

toc_prompt_template = PromptTemplate(
    input_variables=["items"],
    template=TOC_PROMPT
)

# LLM 체인 구성
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
llm_structured = llm.with_structured_output(Section)

toc_chain = toc_prompt_template | llm_structured

# 목차 항목 파싱
toc_section = toc_chain.invoke({"items": "\n\n".join(toc_items)})

# 결과 출력
print("📋 추출된 섹션:")
for item in toc_section.items:
    print(f"{item.number}: {item.title}")
```

**출력 예시:**
```
📋 추출된 섹션:
1: Business
1A: Risk Factors
1B: Unresolved Staff Comments
2: Properties
3: Legal Proceedings
4: Mine Safety Disclosures
7: Management's Discussion and Analysis of Financial Condition and Results of Operations
7A: Quantitative and Qualitative Disclosures About Market Risk
8: Financial Statements and Supplementary Data
9: Changes in and Disagreements With Accountants on Accounting and Financial Disclosure
9A: Controls and Procedures
9B: Other Information
```

### 6단계: 문서를 섹션별로 재구조화

추출한 목차 정보를 바탕으로 전체 문서를 섹션별로 그룹화합니다.

#### 섹션 헤더 매칭 함수

```python
def get_item_header(item_number, item_title):
    """가능한 목차 항목 제목 변형을 반환하는 함수"""
    variations = [
        f"ITEM {item_number.upper()}. {item_title.upper()}",
    ]
    return variations
```

#### 문서에 섹션 메타데이터 추가

```python
from langchain_core.documents import Document

toc_based_docs = []
current_section = None

for i, doc in enumerate(new_docs):
    # 이 문서가 새로운 섹션의 시작인지 확인
    new_section_found = False

    for item in toc_section.items:
        item_headers = get_item_header(item.number, item.title)

        # 목차 항목 헤더가 문서에 포함되어 있는지 확인
        if any(header in doc.page_content for header in item_headers):
            current_section = item.title
            new_section_found = True
            break

    # 새 섹션이 시작되면 헤더 부분부터 추출
    if new_section_found:
        start_index = min([
            doc.page_content.index(header)
            for header in item_headers
            if header in doc.page_content
        ])
        doc.page_content = doc.page_content[start_index:]

    # 섹션 메타데이터 추가
    doc_copy = Document(
        page_content=doc.page_content,
        metadata={
            **doc.metadata,
            "section": current_section if current_section else "Unknown"
        }
    )
    toc_based_docs.append(doc_copy)

# 섹션별 문서 개수 확인
section_counts = {}
for doc in toc_based_docs:
    section = doc.metadata.get("section", "Unknown")
    section_counts[section] = section_counts.get(section, 0) + 1

print("📊 섹션별 문서 분포:")
for section, count in section_counts.items():
    print(f"{section}: {count}개 문서")
```

**출력 예시:**
```
📊 섹션별 문서 분포:
Business: 45개 문서
Risk Factors: 78개 문서
Properties: 12개 문서
Legal Proceedings: 5개 문서
Management's Discussion and Analysis...: 123개 문서
Financial Statements...: 89개 문서
Controls and Procedures: 23개 문서
Unknown: 15개 문서
```

#### 섹션 순서대로 정렬

```python
# 목차 순서에 따라 섹션 정렬
section_order = {item.title: idx for idx, item in enumerate(toc_section.items)}
section_order["Unknown"] = len(section_order)  # Unknown은 마지막

# 문서 정렬
toc_based_docs_sorted = sorted(
    toc_based_docs,
    key=lambda x: section_order.get(x.metadata.get("section", "Unknown"), float('inf'))
)

print(f"✅ 섹션 순서 정렬 완료: {len(toc_based_docs_sorted)}개 문서")
```

#### 정렬된 문서 저장

```python
# pickle로 저장
with open("data/tesla_10k_toc.pkl", "wb") as f:
    pickle.dump(toc_based_docs_sorted, f)

print("✅ 목차 기반 문서 저장 완료")
```

### 7단계: 섹션별 문서 결합

같은 섹션에 속한 문서들을 하나로 결합하여 더 큰 의미 단위를 생성합니다.

```python
# Unknown 섹션을 제외하고 섹션별로 결합
section_docs = {}

for doc in toc_based_docs_sorted:
    section = doc.metadata.get("section", "Unknown")
    if section == "Unknown":
        continue
    if section not in section_docs:
        section_docs[section] = []
    section_docs[section].append(doc)

# 섹션별로 문서 결합
section_docs_combined = {}
for section, docs in section_docs.items():
    # 모든 문서의 page_content를 결합
    combined_content = "\n\n".join([doc.page_content for doc in docs])

    # 첫 번째 문서의 메타데이터 사용
    combined_metadata = docs[0].metadata

    # 새 Document 객체 생성
    section_docs_combined[section] = Document(
        page_content=combined_content,
        metadata=combined_metadata
    )

print(f"✅ 섹션별 결합 완료: {len(section_docs_combined)}개 섹션")
```

#### 결합된 문서 저장

```python
# pickle로 저장
with open("data/tesla_10k_sections.pkl", "wb") as f:
    pickle.dump(section_docs_combined, f)

print("✅ 섹션별 결합 문서 저장 완료")
```

#### 섹션 내용 확인

```python
# Business 섹션 내용 확인
print(f"📄 Business 섹션 내용:\n{section_docs_combined['Business'].page_content[:500]}...")
print(f"\n메타데이터:")
pprint(section_docs_combined['Business'].metadata)
```

## 📊 Part 1 완료 체크리스트

- [x] 10-K 보고서 로드 및 기본 분석
- [x] Unstructured hi_res 전략으로 고급 파싱
- [x] 표 데이터 마크다운 변환
- [x] 문서 길이 및 토큰 분포 분석
- [x] 목차 추출 및 구조화
- [x] 섹션별 문서 재구조화
- [x] 섹션별 문서 결합

## 🎯 다음 단계 (Part 2 예고)

Part 2에서는 다음 내용을 다룹니다:

1. **Parent Document Retriever**: 작은 청크로 검색하고 큰 문맥 반환
2. **Multi-Vector Retrieval**: 요약 기반 검색 시스템
3. **실전 실습**:
   - Element 타입별 분리 처리
   - 청킹 전략 A/B 테스트
   - 하이브리드 검색 (BM25 + Vector)
   - MMR 및 메타데이터 필터링
   - 성능 비교 대시보드

Part 2로 계속됩니다! 🚀
