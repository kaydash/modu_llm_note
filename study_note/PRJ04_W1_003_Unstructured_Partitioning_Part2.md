# Unstructured Partitioning Part2 - HTML, Office, 텍스트 파일 파티셔닝

> **이전 Part**: [Part1 - PDF/이미지 문서 파티셔닝](PRJ04_W1_003_Unstructured_Partitioning_Part1.md)에서 PDF와 이미지 문서의 4가지 파티셔닝 전략을 학습했습니다.

## 📚 학습 목표
- HTML 문서에서 구조화된 요소를 추출하고 계층적 관계를 처리할 수 있다
- MS Office 문서(DOCX, PPTX, XLSX)의 구조적 요소를 파티셔닝할 수 있다
- 일반 텍스트 파일의 단락을 그룹핑하고 구조화할 수 있다
- CSV/TSV 파일을 테이블 구조로 파싱할 수 있다
- 실무에서 문서 타입을 자동 감지하고 적절한 전처리 전략을 적용할 수 있다

## 💻 단계별 구현

### 단계 1: HTML 파티셔닝

HTML 문서는 **웹 문서 구조**를 분석하여 요소를 체계적으로 추출합니다. 로컬 파일과 URL 기반 처리 모두 지원합니다.

#### (1) 로컬 HTML 파일 파티셔닝

```python
from unstructured.partition.html import partition_html

# 로컬 HTML 파일 파티셔닝
elements = partition_html(
    filename="data/sample.html"  # 로컬 HTML 파일 경로
)

print(f"총 요소 개수: {len(elements)}개")

# 요소별 개수 확인
from collections import Counter
element_types = [type(el).__name__ for el in elements]
type_counts = Counter(element_types)

for element_type, count in type_counts.items():
    print(f"{element_type}: {count}개")
```

**출력 예시**:
```
총 요소 개수: 42개
Title: 5개
NarrativeText: 32개
ListItem: 3개
Table: 2개
```

**주요 요소 타입**:
- `Title`: HTML 헤더 태그 (`<h1>`, `<h2>`, ...)
- `NarrativeText`: 단락 (`<p>`)
- `ListItem`: 목록 항목 (`<li>`)
- `Table`: 테이블 (`<table>`)

#### (2) URL에서 직접 파티셔닝

```python
from unstructured.partition.html import partition_html

# URL에서 직접 HTML 파티셔닝
url = "https://example.com/article"

elements = partition_html(
    url=url,  # URL 지정
    headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }  # 사용자 정의 헤더
)

print(f"총 요소 개수: {len(elements)}개")

# 처음 5개 요소 확인
for i, element in enumerate(elements[:5], start=1):
    print(f"\n[요소 {i}]")
    print(f"타입: {type(element).__name__}")
    print(f"텍스트: {element.text[:100]}...")
```

**주요 매개변수**:
- `url`: 웹페이지 URL
- `headers`: HTTP 요청 헤더 (User-Agent 등)
- `ssl_verify`: SSL 인증서 검증 여부 (기본값: True)

#### (3) 계층적 관계 처리

HTML 문서의 요소들은 **계층적 구조**를 가집니다. 각 요소는 `element_id`를 가지며, `parent_id`를 통해 상위 요소와의 관계가 정의됩니다.

```python
def build_hierarchy_tree(elements):
    """요소의 계층적 구조를 트리로 재구성"""

    # element_id를 키로 하는 딕셔너리 생성
    element_dict = {}
    for element in elements:
        element_id = element.metadata.element_id
        element_dict[element_id] = {
            "element": element,
            "children": []
        }

    # 부모-자식 관계 구성
    root_elements = []

    for element in elements:
        element_id = element.metadata.element_id
        parent_id = element.metadata.parent_id

        if parent_id and parent_id in element_dict:
            # 부모 요소의 children 리스트에 추가
            element_dict[parent_id]["children"].append(element_dict[element_id])
        else:
            # 부모가 없으면 루트 요소
            root_elements.append(element_dict[element_id])

    return root_elements

def print_tree(node, level=0):
    """트리 구조를 재귀적으로 출력"""
    indent = "  " * level
    element = node["element"]

    print(f"{indent}[{type(element).__name__}] {element.text[:50]}...")

    # 자식 요소 출력
    for child in node["children"]:
        print_tree(child, level + 1)

# HTML 파티셔닝
elements = partition_html(filename="data/sample.html")

# 계층적 구조 구성
hierarchy = build_hierarchy_tree(elements)

# 트리 출력
print("=== 문서 계층 구조 ===")
for root in hierarchy:
    print_tree(root)
```

**출력 예시**:
```
=== 문서 계층 구조 ===
[Title] Introduction to Machine Learning...
  [NarrativeText] Machine learning is a subset of artificial intel...
  [NarrativeText] There are three main types of machine learning:...
    [ListItem] Supervised Learning...
    [ListItem] Unsupervised Learning...
    [ListItem] Reinforcement Learning...
[Title] Applications...
  [NarrativeText] Machine learning has numerous applications...
  [Table] Model | Accuracy | F1-Score...
```

#### (4) LangChain Document 변환 및 계층적 검색

계층적 구조를 활용하여 **상위 요소와 하위 요소를 함께 검색**하는 RAG 시스템을 구축합니다.

```python
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# HTML 파티셔닝
elements = partition_html(filename="data/sample.html")

# LangChain Document로 변환
documents = []

for element in elements:
    # 메타데이터 추출
    metadata = {
        "element_id": element.metadata.element_id,
        "parent_id": element.metadata.parent_id or "",
        "category": element.category,
        "filename": element.metadata.filename,
    }

    # Document 객체 생성
    doc = Document(
        page_content=element.text,
        metadata=metadata
    )

    documents.append(doc)

print(f"총 Document 개수: {len(documents)}개")

# 벡터 스토어 생성
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(documents, embeddings)

print(f"✅ 벡터 스토어 생성 완료")
```

**계층적 검색 Retriever 구현**:

```python
from typing import List, Any
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

class HierarchicalRetriever(VectorStoreRetriever):
    """부모-자식 관계를 고려한 계층적 검색 Retriever"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_kwargs = kwargs.get("search_kwargs", {})
        self.search_type = kwargs.get("search_type", "similarity")

        # 메타데이터와 document ID 캐싱
        self._metadatas = self.vectorstore.get()['metadatas']
        self._doc_ids = self.vectorstore.get()['ids']

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        """관련 문서 검색"""
        _kwargs = self.search_kwargs | kwargs

        # 유사도 검색
        if self.search_type == "similarity":
            docs = self.vectorstore.similarity_search(query, **_kwargs)
            docs = self._get_additional_docs(docs)
        else:
            raise ValueError(f"search_type {self.search_type} not supported")

        return docs

    def _get_additional_docs(self, docs: List[Document]) -> List[Document]:
        """parent_id가 같은 문서들을 추가로 검색"""
        additional_docs = []

        for doc in docs:
            parent_id = doc.metadata.get('parent_id', '')

            if parent_id:
                # parent_id와 일치하는 문서들 추가
                same_parent_ids = [
                    doc_id for doc_id, meta in zip(self._doc_ids, self._metadatas)
                    if meta.get('parent_id') == parent_id
                ]

                # 벡터 스토어에서 문서 가져오기
                added_docs = self.vectorstore.get_by_ids(same_parent_ids)
                additional_docs.extend(added_docs)

        # 중복 제거
        unique_docs = []
        seen_ids = set()

        for doc in docs + additional_docs:
            doc_id = doc.metadata.get('element_id')
            if doc_id not in seen_ids:
                unique_docs.append(doc)
                seen_ids.add(doc_id)

        print(f"📊 검색 결과: 초기 {len(docs)}개 → 최종 {len(unique_docs)}개")
        return unique_docs

# 계층적 Retriever 생성
retriever = HierarchicalRetriever(
    vectorstore=vectorstore,
    search_kwargs={"k": 3}
)

# 검색 실행
query = "What are the types of machine learning?"
results = retriever.invoke(query)

print(f"\n❓ 질문: {query}\n")

for i, doc in enumerate(results, start=1):
    print(f"[문서 {i}]")
    print(f"카테고리: {doc.metadata['category']}")
    print(f"내용: {doc.page_content[:150]}...\n")
```

**출력 예시**:
```
📊 검색 결과: 초기 3개 → 최종 5개

❓ 질문: What are the types of machine learning?

[문서 1]
카테고리: NarrativeText
내용: There are three main types of machine learning: supervised, unsupervised, and reinforcement learning...

[문서 2]
카테고리: ListItem
내용: Supervised Learning: The algorithm learns from labeled data...

[문서 3]
카테고리: ListItem
내용: Unsupervised Learning: The algorithm finds patterns in unlabeled data...
```

### 단계 2: MS Office 문서 파티셔닝

#### (1) DOCX 파일 파티셔닝

**partition_docx**는 Word 문서의 구조적 요소를 추출합니다. 스타일 메타데이터를 활용하여 요소를 분류합니다.

```python
from unstructured.partition.docx import partition_docx

# DOCX 파일 파티셔닝
elements = partition_docx(
    filename="data/개인정보보호법.docx"
)

print(f"총 요소 개수: {len(elements)}개")

# 요소별 개수 확인
from collections import Counter
element_types = [type(el).__name__ for el in elements]
type_counts = Counter(element_types)

for element_type, count in type_counts.items():
    print(f"{element_type}: {count}개")
```

**출력 예시**:
```
총 요소 개수: 152개
Title: 28개
NarrativeText: 98개
ListItem: 20개
Header: 3개
Footer: 3개
```

**스타일 기반 요소 분류**:
- `"Heading 1"` → Title 요소
- `"Body Text"` → NarrativeText 요소
- `"Normal"` → NarrativeText 요소

**처음 5개 요소 확인**:

```python
# 문서 구성 요소 확인
for element in elements[:5]:
    print(f"\n{'='*60}")
    print(f"타입: {type(element).__name__}")
    print(f"카테고리: {element.category}")
    print(f"텍스트: {element.text[:100]}...")

    # 메타데이터 확인
    if hasattr(element.metadata, 'page_number'):
        print(f"페이지: {element.metadata.page_number}")

    if hasattr(element.metadata, 'category_depth'):
        print(f"계층 깊이: {element.metadata.category_depth}")
```

**출력 예시**:
```
============================================================
타입: Title
카테고리: Title
텍스트: 개인정보 보호법...
페이지: 1
계층 깊이: 0

============================================================
타입: NarrativeText
카테고리: NarrativeText
텍스트: 제1조(목적) 이 법은 개인정보의 처리 및 보호에 관한 사항을 정함으로써...
페이지: 1
계층 깊이: 1
```

**헤더/푸터 추출**:

```python
# 헤더만 추출
headers = [el for el in elements if el.category == "Header"]

print(f"📑 헤더 개수: {len(headers)}개\n")

# 헤더 유형별 분류
for header in headers:
    header_type = header.metadata.header_footer_type
    print(f"유형: {header_type}")
    print(f"내용: {header.text}")
    print()
```

**출력 예시**:
```
📑 헤더 개수: 3개

유형: primary
내용: 개인정보 보호법

유형: first_page
내용: 법률 제18583호

유형: even_page
내용: 개인정보 보호법 (짝수 페이지)
```

**헤더/푸터 유형**:
- `"primary"`: 기본 헤더/푸터
- `"first_page"`: 첫 페이지 전용 헤더/푸터
- `"even_page"`: 짝수 페이지 전용 헤더/푸터

**특정 페이지 요소만 추출**:

```python
# 2페이지의 요소만 추출
page_2_elements = [
    el for el in elements
    if hasattr(el.metadata, 'page_number') and el.metadata.page_number == 2
]

print(f"📄 2페이지 요소 개수: {len(page_2_elements)}개\n")

for el in page_2_elements[:3]:
    print(f"타입: {type(el).__name__}")
    print(f"텍스트: {el.text[:80]}...")
    print()
```

#### (2) PPTX 파일 파티셔닝

**partition_pptx**는 PowerPoint 프레젠테이션의 슬라이드를 구조적으로 분석합니다.

```python
from unstructured.partition.pptx import partition_pptx

# PPTX 파일 파티셔닝
elements = partition_pptx(
    filename="data/한국철도공사_8대도시(목포)_관광 형태 분석 보고서_20220101.pptx",
    include_page_breaks=True  # 슬라이드 간 구분자 추가
)

print(f"총 요소 개수: {len(elements)}개")

# 요소별 개수 확인
from collections import Counter
element_types = [type(el).__name__ for el in elements]
type_counts = Counter(element_types)

for element_type, count in type_counts.items():
    print(f"{element_type}: {count}개")
```

**출력 예시**:
```
총 요소 개수: 85개
Title: 15개
NarrativeText: 42개
ListItem: 18개
Table: 5개
FigureCaption: 3개
PageBreak: 14개
```

**슬라이드별 요소 분리**:

```python
# PageBreak를 기준으로 슬라이드 분리
slides = []
current_slide = []

for element in elements:
    if type(element).__name__ == "PageBreak":
        if current_slide:
            slides.append(current_slide)
            current_slide = []
    else:
        current_slide.append(element)

# 마지막 슬라이드 추가
if current_slide:
    slides.append(current_slide)

print(f"총 슬라이드 수: {len(slides)}개\n")

# 각 슬라이드 요약
for i, slide in enumerate(slides, start=1):
    print(f"[슬라이드 {i}]")
    print(f"요소 개수: {len(slide)}개")

    # 제목 추출
    titles = [el.text for el in slide if type(el).__name__ == "Title"]
    if titles:
        print(f"제목: {titles[0]}")
    print()
```

**출력 예시**:
```
총 슬라이드 수: 15개

[슬라이드 1]
요소 개수: 2개
제목: 목포 관광 형태 분석 보고서

[슬라이드 2]
요소 개수: 6개
제목: 목차

[슬라이드 3]
요소 개수: 8개
제목: 목포 관광객 현황
```

#### (3) XLSX 파일 파티셔닝

**partition_xlsx**는 Excel 문서를 테이블 구조로 변환합니다.

```python
from unstructured.partition.xlsx import partition_xlsx

# XLSX 파일 파티셔닝
elements = partition_xlsx(
    filename="data/sales_data.xlsx"
)

print(f"총 요소 개수: {len(elements)}개")

# 모든 요소는 Table 타입
for i, element in enumerate(elements, start=1):
    print(f"\n[워크시트 {i}]")
    print(f"타입: {type(element).__name__}")
    print(f"텍스트 길이: {len(element.text)}자")

    # HTML 테이블로 변환
    if hasattr(element.metadata, 'text_as_html'):
        print(f"HTML 형식 사용 가능")

        # pandas로 변환
        import pandas as pd
        from io import StringIO

        df = pd.read_html(StringIO(element.metadata.text_as_html))[0]
        print(f"DataFrame 크기: {df.shape[0]} rows × {df.shape[1]} cols")
        print(f"\n미리보기:")
        print(df.head())
```

**출력 예시**:
```
총 요소 개수: 3개

[워크시트 1]
타입: Table
텍스트 길이: 1523자
HTML 형식 사용 가능
DataFrame 크기: 50 rows × 5 cols

미리보기:
   Date       Product  Quantity  Price  Total
0  2024-01-01  Product A      10   1000  10000
1  2024-01-02  Product B      15   1500  22500
2  2024-01-03  Product C       8    800   6400
```

### 단계 3: 일반 텍스트 파일 파티셔닝

**partition_text**는 일반 텍스트 파일을 구조화된 요소로 분할합니다.

#### (1) 기본 텍스트 파티셔닝

```python
from unstructured.partition.text import partition_text

# 텍스트 파일 파티셔닝
elements = partition_text(
    filename="data/article.txt"
)

print(f"총 요소 개수: {len(elements)}개")

# 처음 3개 요소 확인
for i, element in enumerate(elements[:3], start=1):
    print(f"\n[요소 {i}]")
    print(f"타입: {type(element).__name__}")
    print(f"텍스트: {element.text[:100]}...")
```

**출력 예시**:
```
총 요소 개수: 12개

[요소 1]
타입: Title
텍스트: Introduction to Natural Language Processing...

[요소 2]
타입: NarrativeText
텍스트: Natural Language Processing (NLP) is a subfield of artificial intelligence that focuses on the...

[요소 3]
타입: NarrativeText
텍스트: NLP techniques are used in various applications such as machine translation, sentiment analysis...
```

#### (2) 단락 그룹핑

줄바꿈으로 분리된 텍스트를 하나의 단락으로 병합합니다.

```python
from unstructured.partition.text import partition_text

# 단락 그룹핑 활성화
elements = partition_text(
    filename="data/article.txt",
    paragraph_grouper=True  # 단락 그룹핑 활성화
)

print(f"그룹핑 후 요소 개수: {len(elements)}개")

# 그룹핑 없이 처리
elements_no_group = partition_text(
    filename="data/article.txt",
    paragraph_grouper=False
)

print(f"그룹핑 없이: {len(elements_no_group)}개")

# 차이 확인
print(f"\n그룹핑으로 {len(elements_no_group) - len(elements)}개 요소 병합됨")
```

**기본 동작**:
- `\n`으로 분리된 줄 → 하나의 단락으로 병합
- `\n\n`은 단락 구분자로 유지

#### (3) 커스텀 단락 구분

```python
from unstructured.partition.text import partition_text

# 커스텀 구분자 설정
elements = partition_text(
    filename="data/article.txt",
    paragraph_grouper=True,
    # 커스텀 구분자 설정 (선택사항)
    # line_split="\n",      # 줄 구분자 (기본값)
    # paragraph_split="\n\n"  # 단락 구분자 (기본값)
)

print(f"총 요소 개수: {len(elements)}개")

# 각 요소의 길이 확인
for i, element in enumerate(elements, start=1):
    print(f"요소 {i}: {len(element.text)}자")
```

### 단계 4: CSV/TSV 파일 파티셔닝

**partition_csv**와 **partition_tsv**는 CSV/TSV 파일을 단일 테이블 구조로 파싱합니다.

#### (1) CSV 파일 파티셔닝

```python
from unstructured.partition.csv import partition_csv
import pandas as pd
from io import StringIO

# CSV 파일 파티셔닝
elements = partition_csv(
    filename="data/sales_data.csv"
)

print(f"총 요소 개수: {len(elements)}개")

# CSV는 단일 테이블로 추출됨
table_element = elements[0]

print(f"\n타입: {type(table_element).__name__}")
print(f"텍스트 길이: {len(table_element.text)}자")

# HTML 테이블로 변환
if hasattr(table_element.metadata, 'text_as_html'):
    df = pd.read_html(StringIO(table_element.metadata.text_as_html))[0]

    print(f"\nDataFrame 크기: {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"\n미리보기:")
    print(df.head())
```

**출력 예시**:
```
총 요소 개수: 1개

타입: Table
텍스트 길이: 2345자

DataFrame 크기: 100 rows × 6 cols

미리보기:
   Date       Product  Category  Quantity  Price  Total
0  2024-01-01  Product A  Electronics    10   1000  10000
1  2024-01-02  Product B  Clothing       15   1500  22500
2  2024-01-03  Product C  Food            8    800   6400
```

#### (2) TSV 파일 파티셔닝

```python
from unstructured.partition.tsv import partition_tsv
import pandas as pd
from io import StringIO

# TSV 파일 파티셔닝
elements = partition_tsv(
    filename="data/data.tsv"
)

print(f"총 요소 개수: {len(elements)}개")

# TSV도 단일 테이블로 추출됨
table_element = elements[0]

# pandas DataFrame으로 변환
if hasattr(table_element.metadata, 'text_as_html'):
    df = pd.read_html(StringIO(table_element.metadata.text_as_html))[0]

    print(f"\nDataFrame 크기: {df.shape[0]} rows × {df.shape[1]} cols")
    print(df.head())
```

## 🎯 실습 문제

### 실습 1: HTML 계층적 검색 시스템 구축

**문제**: HTML 문서를 파티셔닝하고 계층적 관계를 활용한 검색 시스템을 구축하세요.

**요구사항**:
1. HTML 파일 파티셔닝
2. LangChain Document로 변환
3. FAISS 벡터 스토어 생성
4. HierarchicalRetriever 구현
5. 검색 결과에서 부모-자식 관계 확인

**힌트**:
- `partition_html()` 사용
- `parent_id` 메타데이터 활용
- `get_by_ids()` 메서드로 관련 문서 가져오기

### 실습 2: DOCX 문서 구조 분석 및 시각화

**문제**: Word 문서를 파티셔닝하고 문서 구조를 분석하여 시각화하세요.

**요구사항**:
1. DOCX 파일 파티셔닝
2. 페이지별 요소 개수 집계
3. 헤더/푸터 추출 및 유형 분류
4. Title 요소의 계층 구조 확인 (category_depth 활용)
5. 결과를 DataFrame으로 정리

**힌트**:
- `partition_docx()` 사용
- `page_number` 메타데이터로 페이지별 그룹핑
- `header_footer_type` 메타데이터 확인

### 실습 3: PowerPoint 슬라이드 요약 생성

**문제**: PPTX 파일을 파티셔닝하고 각 슬라이드의 내용을 LLM으로 요약하세요.

**요구사항**:
1. PPTX 파일 파티셔닝 (include_page_breaks=True)
2. PageBreak를 기준으로 슬라이드 분리
3. 각 슬라이드의 제목과 본문 추출
4. LLM을 사용하여 슬라이드별 요약 생성
5. 전체 슬라이드 요약을 마크다운 형식으로 저장

**힌트**:
- `partition_pptx()` 사용
- PageBreak 타입으로 슬라이드 구분
- ChatOpenAI로 요약 생성

### 실습 4: 다중 문서 타입 자동 처리 파이프라인

**문제**: 여러 타입의 문서를 자동으로 감지하고 적절한 파티셔닝 전략을 적용하세요.

**요구사항**:
1. 폴더 내 모든 파일 검색 (PDF, DOCX, PPTX, HTML, TXT, CSV)
2. 파일 확장자에 따라 적절한 partition 함수 자동 선택
3. 각 파일의 요소를 LangChain Document로 변환
4. 통합 벡터 스토어 생성
5. 처리 결과 통계를 CSV 파일로 저장

**힌트**:
- `glob.glob()` 으로 파일 검색
- 확장자별 함수 매핑 딕셔너리 사용
- 예외 처리 포함

## ✅ 솔루션 예시

### 솔루션 1: HTML 계층적 검색 시스템 구축

```python
from unstructured.partition.html import partition_html
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Any
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.vectorstores import VectorStoreRetriever

# 1단계: HTML 파티셔닝
elements = partition_html(filename="data/sample.html")
print(f"✅ 총 {len(elements)}개 요소 추출")

# 2단계: LangChain Document로 변환
documents = []

for element in elements:
    metadata = {
        "element_id": element.metadata.element_id,
        "parent_id": element.metadata.parent_id or "",
        "category": element.category,
        "filename": element.metadata.filename,
    }

    doc = Document(
        page_content=element.text,
        metadata=metadata
    )

    documents.append(doc)

print(f"✅ {len(documents)}개 Document 생성")

# 3단계: 벡터 스토어 생성
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(documents, embeddings)
print(f"✅ 벡터 스토어 생성 완료")

# 4단계: HierarchicalRetriever 구현
class HierarchicalRetriever(VectorStoreRetriever):
    """부모-자식 관계를 고려한 계층적 검색 Retriever"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_kwargs = kwargs.get("search_kwargs", {})
        self.search_type = kwargs.get("search_type", "similarity")
        self._metadatas = self.vectorstore.get()['metadatas']
        self._doc_ids = self.vectorstore.get()['ids']

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        _kwargs = self.search_kwargs | kwargs

        if self.search_type == "similarity":
            docs = self.vectorstore.similarity_search(query, **_kwargs)
            docs = self._get_additional_docs(docs)
        else:
            raise ValueError(f"search_type {self.search_type} not supported")

        return docs

    def _get_additional_docs(self, docs: List[Document]) -> List[Document]:
        """parent_id가 같은 문서들을 추가로 검색"""
        additional_docs = []

        for doc in docs:
            parent_id = doc.metadata.get('parent_id', '')

            if parent_id:
                same_parent_ids = [
                    doc_id for doc_id, meta in zip(self._doc_ids, self._metadatas)
                    if meta.get('parent_id') == parent_id
                ]

                added_docs = self.vectorstore.get_by_ids(same_parent_ids)
                additional_docs.extend(added_docs)

        unique_docs = []
        seen_ids = set()

        for doc in docs + additional_docs:
            doc_id = doc.metadata.get('element_id')
            if doc_id not in seen_ids:
                unique_docs.append(doc)
                seen_ids.add(doc_id)

        print(f"📊 검색 결과: 초기 {len(docs)}개 → 최종 {len(unique_docs)}개")
        return unique_docs

# 5단계: Retriever 생성 및 검색
retriever = HierarchicalRetriever(
    vectorstore=vectorstore,
    search_kwargs={"k": 3}
)

query = "What is the main topic?"
results = retriever.invoke(query)

print(f"\n❓ 질문: {query}\n")

for i, doc in enumerate(results, start=1):
    print(f"[문서 {i}]")
    print(f"카테고리: {doc.metadata['category']}")
    print(f"Parent ID: {doc.metadata['parent_id']}")
    print(f"내용: {doc.page_content[:100]}...\n")
```

### 솔루션 2: DOCX 문서 구조 분석 및 시각화

```python
from unstructured.partition.docx import partition_docx
import pandas as pd
from collections import defaultdict

# 1단계: DOCX 파티셔닝
elements = partition_docx(filename="data/개인정보보호법.docx")
print(f"✅ 총 {len(elements)}개 요소 추출")

# 2단계: 페이지별 요소 개수 집계
page_stats = defaultdict(lambda: defaultdict(int))

for element in elements:
    if hasattr(element.metadata, 'page_number'):
        page_num = element.metadata.page_number
        element_type = type(element).__name__

        page_stats[page_num][element_type] += 1
        page_stats[page_num]["total"] += 1

# DataFrame 생성
page_data = []
for page_num in sorted(page_stats.keys()):
    stats = page_stats[page_num]
    page_data.append({
        "페이지": page_num,
        "총_요소": stats["total"],
        "Title": stats.get("Title", 0),
        "NarrativeText": stats.get("NarrativeText", 0),
        "ListItem": stats.get("ListItem", 0),
        "Header": stats.get("Header", 0),
        "Footer": stats.get("Footer", 0),
    })

page_df = pd.DataFrame(page_data)

print("\n📊 페이지별 요소 통계:")
print(page_df.to_string(index=False))

# 3단계: 헤더/푸터 추출 및 분류
headers = [el for el in elements if el.category == "Header"]
footers = [el for el in elements if el.category == "Footer"]

print(f"\n📑 헤더 개수: {len(headers)}개")
print(f"📑 푸터 개수: {len(footers)}개")

# 헤더 유형별 분류
header_types = defaultdict(list)
for header in headers:
    header_type = header.metadata.header_footer_type
    header_types[header_type].append(header.text)

print("\n[헤더 유형별 분류]")
for header_type, texts in header_types.items():
    print(f"{header_type}: {len(texts)}개")
    for text in texts[:2]:  # 처음 2개만 출력
        print(f"  - {text[:50]}...")

# 4단계: Title 계층 구조 확인
titles = [el for el in elements if el.category == "Title"]

print(f"\n📚 Title 계층 구조:")
for title in titles[:10]:  # 처음 10개만
    depth = title.metadata.category_depth if hasattr(title.metadata, 'category_depth') else 0
    indent = "  " * depth
    print(f"{indent}[Depth {depth}] {title.text}")

# 5단계: 결과 DataFrame 저장
page_df.to_csv("data/docx_structure_analysis.csv", index=False, encoding='utf-8-sig')
print(f"\n💾 결과 저장: data/docx_structure_analysis.csv")
```

### 솔루션 3: PowerPoint 슬라이드 요약 생성

```python
from unstructured.partition.pptx import partition_pptx
from langchain_openai import ChatOpenAI

# 1단계: PPTX 파티셔닝
elements = partition_pptx(
    filename="data/한국철도공사_8대도시(목포)_관광 형태 분석 보고서_20220101.pptx",
    include_page_breaks=True
)
print(f"✅ 총 {len(elements)}개 요소 추출")

# 2단계: PageBreak를 기준으로 슬라이드 분리
slides = []
current_slide = []

for element in elements:
    if type(element).__name__ == "PageBreak":
        if current_slide:
            slides.append(current_slide)
            current_slide = []
    else:
        current_slide.append(element)

if current_slide:
    slides.append(current_slide)

print(f"✅ 총 {len(slides)}개 슬라이드 분리")

# 3단계: 각 슬라이드의 제목과 본문 추출
slide_contents = []

for i, slide in enumerate(slides, start=1):
    # 제목 추출
    titles = [el.text for el in slide if type(el).__name__ == "Title"]
    title = titles[0] if titles else f"슬라이드 {i}"

    # 본문 추출
    body = []
    for el in slide:
        if type(el).__name__ in ["NarrativeText", "ListItem"]:
            body.append(el.text)

    slide_contents.append({
        "slide_num": i,
        "title": title,
        "body": "\n".join(body)
    })

# 4단계: LLM으로 슬라이드별 요약 생성
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
summaries = []

for slide_info in slide_contents:
    print(f"📊 슬라이드 {slide_info['slide_num']} 요약 중...")

    prompt = f"""다음 슬라이드의 내용을 2-3문장으로 요약해주세요:

**제목**: {slide_info['title']}

**내용**:
{slide_info['body']}

요약:"""

    response = llm.invoke(prompt)
    summary = response.content

    summaries.append({
        "slide_num": slide_info['slide_num'],
        "title": slide_info['title'],
        "summary": summary
    })

# 5단계: 마크다운 형식으로 저장
markdown_lines = ["# 프레젠테이션 요약\n"]

for summary_info in summaries:
    markdown_lines.append(f"## 슬라이드 {summary_info['slide_num']}: {summary_info['title']}\n")
    markdown_lines.append(f"{summary_info['summary']}\n")

markdown_content = "\n".join(markdown_lines)

# 파일 저장
with open("data/pptx_summary.md", "w", encoding="utf-8") as f:
    f.write(markdown_content)

print(f"\n✅ 요약 완료")
print(f"💾 저장: data/pptx_summary.md")
```

### 솔루션 4: 다중 문서 타입 자동 처리 파이프라인

```python
import os
from glob import glob
from pathlib import Path
import pandas as pd
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.html import partition_html
from unstructured.partition.text import partition_text
from unstructured.partition.csv import partition_csv

# 1단계: 파일 확장자별 partition 함수 매핑
PARTITION_FUNCTIONS = {
    ".pdf": lambda f: partition_pdf(filename=f, strategy="fast"),
    ".docx": lambda f: partition_docx(filename=f),
    ".pptx": lambda f: partition_pptx(filename=f),
    ".html": lambda f: partition_html(filename=f),
    ".txt": lambda f: partition_text(filename=f),
    ".csv": lambda f: partition_csv(filename=f),
}

# 2단계: 폴더 내 모든 파일 검색
data_folder = "data"
all_files = []

for ext in PARTITION_FUNCTIONS.keys():
    pattern = f"{data_folder}/*{ext}"
    files = glob(pattern)
    all_files.extend(files)

print(f"📂 총 {len(all_files)}개 파일 발견")

# 3단계: 각 파일 처리
all_documents = []
processing_stats = []

for file_path in all_files:
    file_name = Path(file_path).name
    file_ext = Path(file_path).suffix

    print(f"\n⏳ 처리 중: {file_name}")

    try:
        # 적절한 partition 함수 선택
        partition_func = PARTITION_FUNCTIONS.get(file_ext)

        if partition_func is None:
            print(f"⚠️ 지원하지 않는 파일 형식: {file_ext}")
            continue

        # 파티셔닝 실행
        elements = partition_func(file_path)

        # LangChain Document로 변환
        for element in elements:
            metadata = {
                "filename": file_name,
                "file_type": file_ext,
                "category": element.category,
                "element_id": element.metadata.element_id if hasattr(element.metadata, 'element_id') else None,
            }

            doc = Document(
                page_content=element.text,
                metadata=metadata
            )

            all_documents.append(doc)

        # 통계 저장
        processing_stats.append({
            "파일명": file_name,
            "파일_타입": file_ext,
            "요소_개수": len(elements),
            "상태": "SUCCESS"
        })

        print(f"✅ 완료: {len(elements)}개 요소 추출")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        processing_stats.append({
            "파일명": file_name,
            "파일_타입": file_ext,
            "요소_개수": 0,
            "상태": f"ERROR: {str(e)}"
        })

# 4단계: 통합 벡터 스토어 생성
print(f"\n🔍 벡터 스토어 생성 중...")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(all_documents, embeddings)

print(f"✅ 벡터 스토어 생성 완료")
print(f"   - 총 Document 수: {len(all_documents)}개")

# 5단계: 처리 결과 통계 저장
stats_df = pd.DataFrame(processing_stats)

# 요약 통계 추가
summary = {
    "파일명": "===== 요약 =====",
    "파일_타입": "",
    "요소_개수": stats_df[stats_df["상태"] == "SUCCESS"]["요소_개수"].sum(),
    "상태": f"성공: {len(stats_df[stats_df['상태'] == 'SUCCESS'])}개"
}

stats_df = pd.concat([stats_df, pd.DataFrame([summary])], ignore_index=True)

# CSV 저장
stats_df.to_csv("data/processing_stats.csv", index=False, encoding='utf-8-sig')

print(f"\n📊 처리 통계:")
print(stats_df.to_string(index=False))
print(f"\n💾 통계 저장: data/processing_stats.csv")
```

## 🚀 실무 활용 예시

### 예시 1: 법률 문서 검색 시스템

Word와 PDF 법률 문서를 통합하여 조항별 검색이 가능한 시스템을 구축합니다.

```python
from unstructured.partition.docx import partition_docx
from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import re

class LegalDocumentProcessor:
    """법률 문서 처리 및 검색 시스템"""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.documents = []
        self.vectorstore = None

    def extract_article_number(self, text):
        """조항 번호 추출 (예: 제1조, 제2조)"""
        pattern = r'제(\d+)조'
        match = re.search(pattern, text)
        return int(match.group(1)) if match else None

    def process_docx(self, filename):
        """DOCX 파일 처리"""
        print(f"📄 처리 중: {filename}")

        elements = partition_docx(filename=filename)

        for element in elements:
            # 조항 번호 추출
            article_num = self.extract_article_number(element.text)

            metadata = {
                "filename": filename,
                "category": element.category,
                "article_number": article_num,
                "page_number": element.metadata.page_number if hasattr(element.metadata, 'page_number') else None,
            }

            doc = Document(
                page_content=element.text,
                metadata=metadata
            )

            self.documents.append(doc)

        print(f"✅ {len(elements)}개 요소 추출")

    def process_pdf(self, filename):
        """PDF 파일 처리"""
        print(f"📄 처리 중: {filename}")

        elements = partition_pdf(
            filename=filename,
            strategy="fast"
        )

        for element in elements:
            article_num = self.extract_article_number(element.text)

            metadata = {
                "filename": filename,
                "category": element.category,
                "article_number": article_num,
                "page_number": element.metadata.page_number if hasattr(element.metadata, 'page_number') else None,
            }

            doc = Document(
                page_content=element.text,
                metadata=metadata
            )

            self.documents.append(doc)

        print(f"✅ {len(elements)}개 요소 추출")

    def build_vectorstore(self):
        """벡터 스토어 생성"""
        print(f"\n🔍 벡터 스토어 생성 중...")

        self.vectorstore = FAISS.from_documents(
            self.documents,
            self.embeddings
        )

        print(f"✅ 벡터 스토어 생성 완료 ({len(self.documents)}개 문서)")

    def search_by_article(self, article_number):
        """조항 번호로 검색"""
        results = [
            doc for doc in self.documents
            if doc.metadata.get('article_number') == article_number
        ]

        return results

    def query(self, question):
        """질문 응답"""
        if self.vectorstore is None:
            raise ValueError("벡터 스토어가 생성되지 않았습니다.")

        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        result = qa_chain.invoke({"query": question})

        return result

# 사용 예시
processor = LegalDocumentProcessor()

# 문서 처리
processor.process_docx("data/개인정보보호법.docx")
processor.process_pdf("data/근로기준법.pdf")

# 벡터 스토어 생성
processor.build_vectorstore()

# 조항 번호로 검색
article_5_docs = processor.search_by_article(5)

print(f"\n제5조 관련 문서: {len(article_5_docs)}개")
for doc in article_5_docs:
    print(f"- {doc.page_content[:100]}...")

# 질문 응답
question = "개인정보 수집 시 동의를 받아야 하나요?"
result = processor.query(question)

print(f"\n❓ 질문: {question}")
print(f"\n💡 답변:")
print(result['result'])

print(f"\n📚 참고 문서:")
for doc in result['source_documents'][:3]:
    print(f"- 파일: {doc.metadata['filename']}")
    print(f"  조항: 제{doc.metadata['article_number']}조" if doc.metadata['article_number'] else "  조항: N/A")
    print(f"  내용: {doc.page_content[:100]}...")
```

### 예시 2: 프레젠테이션 콘텐츠 재활용 시스템

여러 PPTX 파일에서 특정 주제의 슬라이드를 검색하고 재활용합니다.

```python
from unstructured.partition.pptx import partition_pptx
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from glob import glob
from pathlib import Path

class PresentationContentManager:
    """프레젠테이션 콘텐츠 관리 시스템"""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.slides = []  # 슬라이드별 정보
        self.vectorstore = None

    def process_pptx_folder(self, folder_path):
        """폴더 내 모든 PPTX 파일 처리"""
        pptx_files = glob(f"{folder_path}/*.pptx")

        print(f"📂 총 {len(pptx_files)}개 PPTX 파일 발견")

        for pptx_file in pptx_files:
            self.process_pptx(pptx_file)

    def process_pptx(self, filename):
        """PPTX 파일 처리"""
        print(f"\n📊 처리 중: {Path(filename).name}")

        elements = partition_pptx(
            filename=filename,
            include_page_breaks=True
        )

        # PageBreak를 기준으로 슬라이드 분리
        current_slide = []
        slide_num = 1

        for element in elements:
            if type(element).__name__ == "PageBreak":
                if current_slide:
                    self._process_slide(filename, slide_num, current_slide)
                    current_slide = []
                    slide_num += 1
            else:
                current_slide.append(element)

        # 마지막 슬라이드
        if current_slide:
            self._process_slide(filename, slide_num, current_slide)

        print(f"✅ {slide_num}개 슬라이드 처리 완료")

    def _process_slide(self, filename, slide_num, elements):
        """개별 슬라이드 처리"""
        # 제목 추출
        titles = [el.text for el in elements if type(el).__name__ == "Title"]
        title = titles[0] if titles else f"슬라이드 {slide_num}"

        # 본문 추출
        body_parts = []
        for el in elements:
            if type(el).__name__ in ["NarrativeText", "ListItem"]:
                body_parts.append(el.text)

        body = "\n".join(body_parts)

        # 슬라이드 정보 저장
        slide_info = {
            "filename": Path(filename).name,
            "slide_number": slide_num,
            "title": title,
            "body": body,
            "full_content": f"{title}\n\n{body}"
        }

        self.slides.append(slide_info)

    def build_vectorstore(self):
        """벡터 스토어 생성"""
        print(f"\n🔍 벡터 스토어 생성 중...")

        documents = []

        for slide_info in self.slides:
            metadata = {
                "filename": slide_info["filename"],
                "slide_number": slide_info["slide_number"],
                "title": slide_info["title"],
            }

            doc = Document(
                page_content=slide_info["full_content"],
                metadata=metadata
            )

            documents.append(doc)

        self.vectorstore = FAISS.from_documents(documents, self.embeddings)

        print(f"✅ 벡터 스토어 생성 완료 ({len(documents)}개 슬라이드)")

    def search_slides(self, query, k=5):
        """주제로 슬라이드 검색"""
        if self.vectorstore is None:
            raise ValueError("벡터 스토어가 생성되지 않았습니다.")

        results = self.vectorstore.similarity_search(query, k=k)

        return results

# 사용 예시
manager = PresentationContentManager()

# 모든 PPTX 파일 처리
manager.process_pptx_folder("data")

# 벡터 스토어 생성
manager.build_vectorstore()

# 주제로 슬라이드 검색
query = "관광객 통계"
results = manager.search_slides(query, k=5)

print(f"\n🔍 검색 쿼리: {query}")
print(f"📊 검색 결과: {len(results)}개 슬라이드\n")

for i, doc in enumerate(results, start=1):
    print(f"[슬라이드 {i}]")
    print(f"파일: {doc.metadata['filename']}")
    print(f"슬라이드 번호: {doc.metadata['slide_number']}")
    print(f"제목: {doc.metadata['title']}")
    print(f"내용: {doc.page_content[:100]}...")
    print()
```

### 예시 3: PDF 자동 문서 감지 및 전처리

PDF 문서에 테이블이 포함되어 있는지 자동 감지하고 적절한 전처리를 수행합니다.

```python
from unstructured.partition.pdf import partition_pdf
from collections import Counter

class PDFAutoProcessor:
    """PDF 자동 처리 시스템"""

    def __init__(self):
        self.has_table = False
        self.elements = []

    def detect_and_process(self, filename):
        """테이블 감지 및 자동 처리"""
        print(f"📄 처리 중: {filename}")

        # 1단계: hi_res 전략으로 테이블 감지
        print("🔍 테이블 감지 중...")

        elements = partition_pdf(
            filename=filename,
            strategy="hi_res",
            infer_table_structure=True
        )

        # 2단계: 테이블 존재 여부 확인
        element_types = [type(el).__name__ for el in elements]
        type_counts = Counter(element_types)

        self.has_table = "Table" in type_counts and type_counts["Table"] > 0

        if self.has_table:
            print(f"✅ 테이블 감지됨: {type_counts['Table']}개")
            self.elements = self._process_with_tables(elements)
        else:
            print(f"ℹ️  테이블 없음: 텍스트만 추출")
            self.elements = self._process_text_only(elements)

        return self.elements

    def _process_with_tables(self, elements):
        """테이블이 있는 경우: HTML 요소를 텍스트 요소와 함께 추출"""
        print("📊 테이블 포함 처리 모드")

        processed_elements = []

        for element in elements:
            # 테이블 요소는 HTML 형식으로 변환
            if type(element).__name__ == "Table":
                if hasattr(element.metadata, 'text_as_html'):
                    processed_elements.append({
                        "type": "table",
                        "content": element.metadata.text_as_html,
                        "text": element.text,
                        "page": element.metadata.page_number if hasattr(element.metadata, 'page_number') else None
                    })
            else:
                # 일반 텍스트 요소
                processed_elements.append({
                    "type": "text",
                    "category": element.category,
                    "content": element.text,
                    "page": element.metadata.page_number if hasattr(element.metadata, 'page_number') else None
                })

        print(f"✅ 처리 완료: {len(processed_elements)}개 요소")
        return processed_elements

    def _process_text_only(self, elements):
        """테이블이 없는 경우: 텍스트 요소만 추출"""
        print("📝 텍스트 전용 처리 모드")

        processed_elements = []

        for element in elements:
            if type(element).__name__ != "Table":
                processed_elements.append({
                    "type": "text",
                    "category": element.category,
                    "content": element.text,
                    "page": element.metadata.page_number if hasattr(element.metadata, 'page_number') else None
                })

        print(f"✅ 처리 완료: {len(processed_elements)}개 요소")
        return processed_elements

    def export_to_markdown(self):
        """마크다운 형식으로 내보내기"""
        markdown_lines = []

        for element in self.elements:
            if element["type"] == "table":
                markdown_lines.append(f"\n## 테이블 (페이지 {element['page']})\n")
                markdown_lines.append(element["text"])
                markdown_lines.append("\n")
            elif element["type"] == "text":
                if element["category"] == "Title":
                    markdown_lines.append(f"\n## {element['content']}\n")
                else:
                    markdown_lines.append(element["content"])
                    markdown_lines.append("\n")

        return "\n".join(markdown_lines)

# 사용 예시
processor = PDFAutoProcessor()

# 테이블이 있는 문서 처리
elements = processor.detect_and_process("data/transformer.pdf")

# 마크다운 내보내기
markdown = processor.export_to_markdown()

# 파일 저장
with open("data/auto_processed.md", "w", encoding="utf-8") as f:
    f.write(markdown)

print(f"\n💾 저장: data/auto_processed.md")
print(f"   - 테이블 포함 여부: {'예' if processor.has_table else '아니오'}")
print(f"   - 총 요소 수: {len(processor.elements)}개")
```

## 📖 참고 자료

### 공식 문서
- **Unstructured 공식 문서**: https://unstructured-io.github.io/unstructured/
- **HTML 파티셔닝**: https://unstructured-io.github.io/unstructured/core/partition.html#partition-html
- **Office 문서 파티셔닝**: https://unstructured-io.github.io/unstructured/core/partition.html#partition-docx

### 추가 학습 자료
- **LangChain 통합**: https://python.langchain.com/docs/integrations/document_loaders/unstructured_file
- **계층적 검색**: https://python.langchain.com/docs/how_to/parent_document_retriever/

---

**이전 Part**: [Part1 - PDF/이미지 문서 파티셔닝](PRJ04_W1_003_Unstructured_Partitioning_Part1.md)에서 PDF 파티셔닝의 4가지 전략을 학습했습니다.

**완료**: Unstructured Partitioning 전체 학습 가이드가 완료되었습니다!
