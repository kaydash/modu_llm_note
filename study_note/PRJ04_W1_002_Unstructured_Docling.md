# Docling을 활용한 PDF 문서 처리 - 고급 문서 파싱 가이드

## 📚 학습 목표
- Docling과 다른 문서 처리 라이브러리(Unstructured, LlamaParse)의 특징과 차이점을 이해한다
- Docling을 사용하여 PDF 문서를 구조화된 데이터로 변환할 수 있다
- OCR과 테이블 구조 분석 등 고급 문서 처리 기능을 활용할 수 있다
- 문서의 텍스트와 테이블 요소를 구분하여 추출하고 저장할 수 있다
- 실무에서 대량의 문서를 체계적으로 처리하는 파이프라인을 구축할 수 있다

## 🔑 핵심 개념

### Docling이란?
Docling은 **PDF, DOCX, HTML 등 다양한 문서 형식을 구조화된 데이터로 변환**하는 오픈소스 라이브러리입니다. IBM Research에서 개발했으며, 특히 **복잡한 테이블 구조**와 **문서 레이아웃**을 정확하게 파싱하는 데 강점이 있습니다.

### 주요 특징
- **MIT 라이선스**: 완전 무료 오픈소스
- **높은 정확도**: 테이블 추출 정확도 97.9%
- **로컬 실행**: 외부 API 없이 로컬에서 실행 가능
- **다양한 출력 형식**: 마크다운, 텍스트, JSON, HTML 등
- **LangChain 통합**: RAG 시스템과 손쉬운 연동

### 문서 처리 라이브러리 비교

| 항목 | Unstructured | Docling | LlamaParse |
|------|-------------|---------|------------|
| **라이선스** | 오픈소스 + 상용 | MIT (완전 무료) | 상용 (무료 플랜) |
| **파일 형식** | 64+ 형식 | 주요 형식 지원 | 10+ 형식 |
| **처리 속도** | 느림 (50p: 141초) | 중간 (선형 확장) | 매우 빠름 (6초) |
| **정확도** | 75-100% | 97.9% | 중간 |
| **테이블 추출** | 단순: 100%, 복잡: 75% | 97.9% | 개선 필요 |
| **OCR 지원** | ✅ | ✅ | ✅ |
| **로컬 실행** | ✅ | ✅ | ❌ (API만) |
| **LangChain 통합** | ✅ | ✅ | ✅ |

### 라이브러리 선택 가이드

**🏆 최고 정확도가 필요한 경우 → Docling**
- 금융 보고서, 법률 문서 등 정확도가 중요한 비즈니스 문서
- 복잡한 테이블이 포함된 학술 논문
- 문서 구조 분석이 중요한 경우

**⚡ 최고 속도가 필요한 경우 → LlamaParse**
- 실시간 문서 처리 시스템
- 대량의 문서를 빠르게 처리하는 배치 작업
- 프로토타이핑 및 초기 개발 단계

**🔧 최고 유연성이 필요한 경우 → Unstructured**
- 다양한 문서 형식을 처리하는 통합 시스템
- 복잡한 ETL 파이프라인 구축
- 엔터프라이즈급 데이터 처리 플랫폼

### 관련 기술 스택
- **Python 3.8+**: 기본 실행 환경
- **RapidOCR**: 광학 문자 인식 (OCR)
- **ONNX Runtime**: 딥러닝 모델 실행
- **Pandas**: 테이블 데이터 처리
- **LangChain**: RAG 시스템 통합

## 🛠 환경 설정

### 1. 라이브러리 설치

#### pip 사용
```bash
pip install docling
```

#### uv 사용 (권장)
```bash
uv add docling
```

### 2. 환경 변수 설정

```python
from dotenv import load_dotenv
load_dotenv()
```

### 3. 기본 라이브러리 임포트

```python
import os
from glob import glob
from pathlib import Path
from pprint import pprint
import json
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')
```

### 4. LangSmith 추적 설정 (선택사항)

```python
# LangSmith 추적 활성화 여부 확인
import os
print(os.getenv('LANGSMITH_TRACING'))  # 'true' 또는 'false'
```

## 💻 단계별 구현

### 단계 1: 기본 PDF 변환

Docling의 가장 기본적인 사용법은 **DocumentConverter**를 초기화하고 PDF 파일을 변환하는 것입니다.

```python
from docling.document_converter import DocumentConverter

# DocumentConverter 초기화 (기본 설정)
converter = DocumentConverter()

# PDF 파일 경로
pdf_path = "data/labor_law.pdf"  # 근로기준법 문서 경로

try:
    # 변환 실행
    result = converter.convert(pdf_path)

    # 변환 결과 확인
    if result.status.name in ['SUCCESS', 'PARTIAL_SUCCESS']:
        print(f"✅ 변환 성공: {result.status.name}")
        print(f"📄 문서 제목: {result.document.name}")
    else:
        print(f"❌ 변환 실패: {result.status}")

except Exception as e:
    print(f"⚠️ 오류 발생: {e}")
```

**주요 포인트**:
- `DocumentConverter()`: 기본 설정으로 초기화
- `convert()`: PDF 파일을 변환
- `result.status.name`: 변환 상태 확인 (SUCCESS, PARTIAL_SUCCESS, FAILURE)
- `result.document`: 변환된 문서 객체 (DoclingDocument)

**변환 결과 타입**:
```python
type(result)  # <class 'docling.datamodel.document.ConversionResult'>
```

**ConversionResult 속성**:
```python
result.model_dump().keys()
# dict_keys(['input', 'status', 'errors', 'pages', 'assembled',
#            'timings', 'confidence', 'document'])
```

### 단계 2: 텍스트 추출 및 마크다운 변환

변환된 문서에서 텍스트를 추출하고 다양한 형식으로 내보낼 수 있습니다.

```python
# DoclingDocument 객체로 변환
document = result.document

# 마크다운으로 내보내기
markdown_text = document.export_to_markdown()
print(markdown_text[:500] + "...")  # 처음 500자만 출력

# 순수 텍스트로 내보내기
plain_text = document.export_to_text()
print(f"📏 텍스트 길이: {len(plain_text)}자")
print(plain_text[:300] + "...")  # 처음 300자만 출력

# 파일로 저장
save_folder = Path("data/docling_output")
save_folder.mkdir(parents=True, exist_ok=True)

with open(save_folder / "labor_law.txt", "w", encoding="utf-8") as f:
    f.write(plain_text)

print(f"💾 저장 완료: {save_folder / 'labor_law.txt'}")
```

**주요 메서드**:
- `export_to_markdown()`: 마크다운 형식으로 내보내기 (테이블 구조 유지)
- `export_to_text()`: 순수 텍스트로 내보내기
- `export_to_html()`: HTML 형식으로 내보내기

**출력 예시**:
```markdown
## 제1장 총칙

- 제1조(목적) 이 법은 헌법에 따라 근로조건의 기준을 정함으로써
  근로자의 기본적 생활을 보장, 향상시키며 균형 있는 국민경제의
  발전을 꾀하는 것을 목적으로 한다.

제2조(정의) ① 이 법에서 사용하는 용어의 뜻은 다음과 같다.
1. '근로자'란 직업의 종류와 관계없이 임금을 목적으로 사업이나
   사업장에 근로를 제공하는 사람을 말한다.
```

### 단계 3: 고급 설정 (OCR, 테이블 구조)

복잡한 문서를 처리하려면 **OCR**과 **테이블 구조 분석**을 활성화해야 합니다.

```python
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorDevice

class AdvancedDocProcessor:
    """고급 문서 처리기"""

    def __init__(self, enable_ocr=False, enable_table_structure=True):
        """
        Args:
            enable_ocr: OCR 기능 활성화 (스캔된 문서용)
            enable_table_structure: 테이블 구조 분석 활성화
        """

        # 파이프라인 옵션 설정
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = enable_ocr  # OCR 활성화 여부
        pipeline_options.do_table_structure = enable_table_structure  # 테이블 구조 분석

        # 테이블 구조 분석 세부 설정
        if enable_table_structure:
            pipeline_options.table_structure_options.do_cell_matching = True  # 셀 매칭

        # CPU 사용 설정 (GPU가 없는 환경)
        pipeline_options.accelerator_options.device = AcceleratorDevice.CPU

        # DocumentConverter 초기화
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

        print(f"🔧 처리기 초기화 완료:")
        print(f"   - OCR: {'활성화' if enable_ocr else '비활성화'}")
        print(f"   - 테이블 구조 분석: {'활성화' if enable_table_structure else '비활성화'}")

    def process_pdf(self, pdf_path):
        """PDF 처리"""
        try:
            print(f"⏳ 처리 시작: {pdf_path}")
            result = self.converter.convert(str(pdf_path))

            if result.status.name in ['SUCCESS', 'PARTIAL_SUCCESS']:
                print(f"✅ 처리 완료: {result.status.name}")
                return result.document
            else:
                print(f"❌ 처리 실패: {result.status}")
                return None

        except Exception as e:
            print(f"⚠️ 오류 발생: {e}")
            return None
```

#### (1) 텍스트만 추출하기

```python
# 기본 설정으로 처리 (OCR 비활성화, 테이블 구조 분석 비활성화)
basic_processor = AdvancedDocProcessor(
    enable_ocr=False,
    enable_table_structure=False
)

# PDF 처리 실행
pdf_path = "data/transformer.pdf"
document = basic_processor.process_pdf(pdf_path)

# 결과 분석
markdown = document.export_to_markdown()

print(f"📄 문서명: {document.name}")
print(f"📏 텍스트 길이: {len(markdown)}자")

# 테이블이 있는지 확인
if "| " in markdown or "|--" in markdown:
    print("   🔍 테이블 구조 감지됨")
else:
    print("   📝 일반 텍스트 문서")

# 마크다운 파일로 저장
save_folder = Path("data/docling_output")
save_folder.mkdir(parents=True, exist_ok=True)

with open(save_folder / "transformer_analysis.md", "w", encoding="utf-8") as f:
    f.write(markdown)
```

#### (2) 테이블 구조 추출하기

복잡한 테이블이 포함된 문서(예: 재무 보고서)를 처리할 때는 **OCR**과 **테이블 구조 분석**을 모두 활성화합니다.

```python
# 고급 설정으로 처리 (OCR 활성화, 테이블 구조 분석 활성화)
ocr_processor = AdvancedDocProcessor(
    enable_ocr=True,  # OCR 활성화
    enable_table_structure=True  # 테이블 구조 분석 활성화
)

# 테슬라 10-K 보고서 처리 (복잡한 테이블 포함)
pdf_path = "data/tsla-20241231-gen.pdf"
document = ocr_processor.process_pdf(pdf_path)

# 결과 분석
markdown = document.export_to_markdown()

print(f"📄 문서명: {document.name}")
print(f"📏 텍스트 길이: {len(markdown)}자")

# 테이블이 있는지 확인
if "| " in markdown or "|--" in markdown:
    print("   🔍 테이블 구조 감지됨")
else:
    print("   📝 일반 텍스트 문서")

# 마크다운 파일로 저장
with open(save_folder / "tsla_analysis_ocr.md", "w", encoding="utf-8") as f:
    f.write(markdown)
```

**주요 설정 옵션**:
- `do_ocr`: OCR (광학 문자 인식) 활성화
- `do_table_structure`: 테이블 구조 분석 활성화
- `do_cell_matching`: 테이블 셀 매칭 (테이블 구조 분석 시 필수)
- `AcceleratorDevice.CPU`: CPU 사용 (GPU 없는 환경)

**성능 차이**:
- 기본 처리: transformer.pdf (15페이지) → 약 17초
- OCR + 테이블 처리: tsla-20241231-gen.pdf (100+ 페이지) → 약 290초

### 단계 4: 문서 구조 분석

Docling은 문서를 **구조화된 요소**로 분해합니다. 이를 통해 텍스트와 테이블을 구분하고, 각 요소의 메타데이터를 추출할 수 있습니다.

#### DoclingDocument의 주요 속성

```python
# DoclingDocument의 속성 확인
document.model_dump().keys()
# dict_keys(['schema_name', 'version', 'name', 'origin', 'furniture',
#            'body', 'groups', 'texts', 'pictures', 'tables',
#            'key_value_items', 'form_items', 'pages'])
```

**주요 속성 설명**:
- `name`: 문서명
- `texts`: 모든 텍스트 요소 리스트
- `tables`: 모든 테이블 요소 리스트
- `pictures`: 모든 이미지 요소 리스트
- `pages`: 페이지 정보

#### 텍스트 요소 분석

```python
# 첫 번째 텍스트 아이템의 속성 확인
first_text = document.texts[0].model_dump()
pprint(first_text)

"""
{
    'self_ref': '#/texts/0',
    'parent': {'cref': '#/body'},
    'children': [],
    'content_layer': 'body',
    'label': 'text',  # 요소 타입 (text, list_item, section_header 등)
    'prov': [  # 위치 정보
        {
            'page_no': 1,  # 페이지 번호
            'bbox': {  # Bounding box (위치 좌표)
                'l': 14.36,
                't': 668.761,
                'r': 61.752,
                'b': 660.041,
                'coord_origin': 'BOTTOMLEFT'
            },
            'charspan': (0, 10)  # 문자 범위
        }
    ],
    'orig': '(Mark\tOne)',  # 원본 텍스트
    'text': '(Mark\tOne)',  # 정제된 텍스트
    'formatting': None,
    'hyperlink': None
}
"""
```

**TextItem 주요 필드**:
- `label`: 요소 타입 (text, list_item, section_header 등)
- `prov`: 위치 정보 (페이지 번호, bounding box)
- `text`: 텍스트 내용
- `page_no`: 페이지 번호

#### 테이블 요소 분석

```python
# 첫 번째 테이블 아이템의 속성 확인
first_table = document.tables[0].model_dump()

"""
{
    'self_ref': '#/tables/0',
    'label': 'table',
    'prov': [{'page_no': 1, 'bbox': {...}}],
    'data': {
        'table_cells': [  # 셀 정보 리스트
            {
                'bbox': {...},
                'row_span': 1,
                'col_span': 1,
                'start_row_offset_idx': 0,
                'end_row_offset_idx': 1,
                'start_col_offset_idx': 0,
                'end_col_offset_idx': 1,
                'text': 'Title of each class',
                'column_header': True,  # 열 헤더 여부
                'row_header': False,
                'row_section': False,
                'fillable': False
            },
            ...
        ],
        'num_rows': 2,
        'num_cols': 3,
        'grid': [...]  # 테이블 그리드 구조
    }
}
"""
```

**TableItem 주요 필드**:
- `data.table_cells`: 모든 셀 정보
- `data.num_rows`: 행 개수
- `data.num_cols`: 열 개수
- `data.grid`: 테이블 그리드 구조

#### 문서 요소를 순서대로 JSON으로 변환

문서의 모든 요소(텍스트, 테이블)를 **원본 순서대로** 추출하는 함수를 작성합니다.

```python
from docling_core.types.doc import TextItem, TableItem
import pandas as pd

def convert_document_to_ordered_json(document):
    """문서 요소를 원본 순서대로 JSON 배열로 변환 (테이블은 마크다운+딕셔너리 형식)"""
    elements = []

    for item, level in document.iterate_items():
        if isinstance(item, TextItem):
            elements.append({
                "type": "text",
                "content": item.text.replace("\t", " ") or "",
                "page": item.prov[0].page_no if item.prov else None,
                "label": item.label.value if item.label else None,
                "level": level,
                "element": item  # 원본 요소 추가
            })
        elif isinstance(item, TableItem):
            table_content = {}

            try:
                # DataFrame으로 변환
                df = item.export_to_dataframe()

                # 마크다운 형식으로 변환
                markdown_content = df.to_markdown(index=False)  # 인덱스 제외

                # 딕셔너리 형식으로 변환 (각 행을 딕셔너리로)
                dict_content = df.to_dict('records')

                table_content = {
                    "markdown": markdown_content,
                    "data": dict_content,
                    "status": "success"
                }

            except Exception as e:
                # DataFrame 변환이 실패한 경우 HTML 대안 시도
                try:
                    html_content = item.export_to_html()
                    table_content = {
                        "html": html_content,
                        "status": "html_fallback",
                        "error": str(e)
                    }
                except Exception as e2:
                    table_content = {
                        "status": "failed",
                        "error": f"DataFrame 변환 실패: {str(e)}, HTML 변환 실패: {str(e2)}"
                    }

            elements.append({
                "type": "table",
                "content": table_content,
                "page": item.prov[0].page_no if item.prov else None,
                "level": level,
                "element": item  # 원본 요소 추가
            })

    return elements

# 문서 요소를 원본 순서대로 JSON 배열로 변환
ordered_json = convert_document_to_ordered_json(document)
print(f"📊 총 {len(ordered_json)}개 요소 추출 완료")
```

#### 테이블과 텍스트 요소 분리

```python
# 테이블 인덱스 찾기
table_indices = []
for i, item in enumerate(ordered_json):
    if item['type'] == 'table':
        table_indices.append(i)

print(f"📋 테이블 개수: {len(table_indices)}개")
print(f"📍 테이블 인덱스: {table_indices[:10]}...")  # 처음 10개만 출력

# 첫 번째 테이블 요소 확인
if len(table_indices) > 0:
    first_table_idx = table_indices[0]
    item = ordered_json[first_table_idx]

    print(f"\n📊 첫 번째 테이블:")
    print(f"   - 인덱스: {first_table_idx}")
    print(f"   - 페이지: {item.get('page', 'N/A')}")
    print(f"   - 상태: {item['content']['status']}")

    if item['content']['status'] == 'success':
        print(f"\n[마크다운 형식 미리보기]")
        markdown = item['content']['markdown']
        print(markdown)
```

**출력 예시**:
```markdown
| Title of each class | Trading Symbol(s) | Name of each exchange on which registered |
|:--------------------|:------------------|:-------------------------------------------|
| Common stock        | TSLA              | The Nasdaq Global Select Market            |
```

#### 테이블을 DataFrame으로 변환

```python
# 특정 테이블을 DataFrame으로 변환
table_idx = table_indices[0]  # 첫 번째 테이블
item = ordered_json[table_idx]

if item['content']['status'] == 'success':
    # DataFrame으로 변환하여 표시
    df = pd.DataFrame(item['content']['data'])
    print(f"📊 테이블 크기: {df.shape[0]} rows × {df.shape[1]} columns")
    display(df)
else:
    print(f"⚠️ 테이블 변환 실패: {item['content'].get('error')}")
```

#### Pickle로 저장

추출한 JSON 데이터를 **pickle** 형식으로 저장하면 나중에 빠르게 로드할 수 있습니다.

```python
import pickle

save_folder = Path("data/docling_output")
pickle_path = save_folder / "tsla_analysis_ocr.pkl"

with open(pickle_path, "wb") as f:
    pickle.dump(ordered_json, f)

print(f"💾 Pickle 파일로 저장됨: {pickle_path}")
```

**Pickle 파일 로드**:
```python
with open(pickle_path, "rb") as f:
    loaded_json = pickle.load(f)

print(f"📂 총 {len(loaded_json)}개 요소 로드됨")
```

## 🎯 실습 문제

### 실습 1: 기본 PDF 변환 및 텍스트 추출

**문제**: `data/transformer.pdf` 파일을 처리하여 순수 텍스트와 마크다운 형식으로 추출하고 저장하세요.

**요구사항**:
1. DocumentConverter를 사용하여 PDF 변환
2. 순수 텍스트와 마크다운 형식으로 내보내기
3. 각각 `transformer_text.txt`, `transformer_markdown.md` 파일로 저장
4. 텍스트 길이와 저장 경로를 출력

**힌트**:
- `export_to_text()` 메서드 사용
- `export_to_markdown()` 메서드 사용
- `Path` 객체로 디렉토리 생성

### 실습 2: 테이블 구조 추출 및 분석

**문제**: `data/transformer.pdf` 파일에서 모든 테이블을 추출하여 개별 마크다운 파일로 저장하세요.

**요구사항**:
1. `AdvancedDocProcessor` 클래스를 사용하여 테이블 구조 분석 활성화
2. `convert_document_to_ordered_json` 함수로 문서 요소 추출
3. 테이블만 필터링하여 개수 출력
4. 각 테이블을 `table_1.md`, `table_2.md` 형식으로 저장
5. 각 테이블의 페이지 번호와 크기 정보 출력

**힌트**:
- `enable_table_structure=True` 설정
- `item['type'] == 'table'` 조건으로 필터링
- 테이블 인덱스를 사용하여 파일명 생성

### 실습 3: 텍스트와 테이블 요소 분리 및 저장

**문제**: `data/transformer.pdf` 파일을 처리하여 텍스트와 테이블 요소를 구분하여 저장하세요.

**요구사항**:
1. OCR 비활성화, 테이블 구조 분석 활성화로 처리
2. 텍스트 요소는 하나의 텍스트 파일로 저장 (`transformer_text.txt`)
3. 테이블 요소는 하나의 마크다운 파일로 저장 (`transformer_tables.md`)
4. 각 섹션별로 페이지 번호와 요소 개수 출력

**힌트**:
- 텍스트 요소: `item['type'] == 'text'`
- 테이블 요소: `item['type'] == 'table'`
- 각 섹션별로 헤더 추가 (`## 테이블 1`, `## 텍스트 섹션 1`)

### 실습 4: 대용량 문서 배치 처리

**문제**: `data/` 폴더의 모든 PDF 파일을 자동으로 처리하여 텍스트를 추출하고 요약 통계를 출력하세요.

**요구사항**:
1. `glob` 모듈로 모든 PDF 파일 검색
2. 각 파일을 `AdvancedDocProcessor`로 처리
3. 각 파일의 텍스트 길이, 테이블 개수, 페이지 수 출력
4. 전체 통계를 DataFrame으로 정리하여 CSV 파일로 저장 (`pdf_processing_stats.csv`)

**힌트**:
- `glob.glob("data/*.pdf")` 사용
- 각 파일별로 통계 딕셔너리 생성
- `pd.DataFrame(stats_list)` 로 DataFrame 생성

## ✅ 솔루션 예시

### 솔루션 1: 기본 PDF 변환 및 텍스트 추출

```python
from docling.document_converter import DocumentConverter
from pathlib import Path

# 1단계: DocumentConverter 초기화
converter = DocumentConverter()

# 2단계: PDF 변환
pdf_path = "data/transformer.pdf"
result = converter.convert(pdf_path)

if result.status.name in ['SUCCESS', 'PARTIAL_SUCCESS']:
    document = result.document

    # 3단계: 순수 텍스트로 내보내기
    plain_text = document.export_to_text()

    # 4단계: 마크다운으로 내보내기
    markdown_text = document.export_to_markdown()

    # 5단계: 저장 폴더 생성
    save_folder = Path("data/docling_output")
    save_folder.mkdir(parents=True, exist_ok=True)

    # 6단계: 텍스트 파일로 저장
    text_path = save_folder / "transformer_text.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(plain_text)

    # 7단계: 마크다운 파일로 저장
    markdown_path = save_folder / "transformer_markdown.md"
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)

    # 8단계: 결과 출력
    print(f"✅ 변환 완료")
    print(f"📏 텍스트 길이: {len(plain_text)}자")
    print(f"💾 저장 경로:")
    print(f"   - 텍스트: {text_path}")
    print(f"   - 마크다운: {markdown_path}")
else:
    print(f"❌ 변환 실패: {result.status}")
```

### 솔루션 2: 테이블 구조 추출 및 분석

```python
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorDevice
from docling_core.types.doc import TextItem, TableItem
import pandas as pd
from pathlib import Path

# AdvancedDocProcessor 클래스 (이전 단계에서 정의한 클래스 사용)
class AdvancedDocProcessor:
    """고급 문서 처리기"""

    def __init__(self, enable_ocr=False, enable_table_structure=True):
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = enable_ocr
        pipeline_options.do_table_structure = enable_table_structure

        if enable_table_structure:
            pipeline_options.table_structure_options.do_cell_matching = True

        pipeline_options.accelerator_options.device = AcceleratorDevice.CPU

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

        print(f"🔧 처리기 초기화 완료:")
        print(f"   - OCR: {'활성화' if enable_ocr else '비활성화'}")
        print(f"   - 테이블 구조 분석: {'활성화' if enable_table_structure else '비활성화'}")

    def process_pdf(self, pdf_path):
        try:
            print(f"⏳ 처리 시작: {pdf_path}")
            result = self.converter.convert(str(pdf_path))

            if result.status.name in ['SUCCESS', 'PARTIAL_SUCCESS']:
                print(f"✅ 처리 완료: {result.status.name}")
                return result.document
            else:
                print(f"❌ 처리 실패: {result.status}")
                return None

        except Exception as e:
            print(f"⚠️ 오류 발생: {e}")
            return None

def convert_document_to_ordered_json(document):
    """문서 요소를 원본 순서대로 JSON 배열로 변환"""
    elements = []

    for item, level in document.iterate_items():
        if isinstance(item, TextItem):
            elements.append({
                "type": "text",
                "content": item.text.replace("\t", " ") or "",
                "page": item.prov[0].page_no if item.prov else None,
                "label": item.label.value if item.label else None,
                "level": level,
                "element": item
            })
        elif isinstance(item, TableItem):
            table_content = {}

            try:
                df = item.export_to_dataframe()
                markdown_content = df.to_markdown(index=False)
                dict_content = df.to_dict('records')

                table_content = {
                    "markdown": markdown_content,
                    "data": dict_content,
                    "status": "success"
                }
            except Exception as e:
                try:
                    html_content = item.export_to_html()
                    table_content = {
                        "html": html_content,
                        "status": "html_fallback",
                        "error": str(e)
                    }
                except Exception as e2:
                    table_content = {
                        "status": "failed",
                        "error": f"DataFrame 변환 실패: {str(e)}, HTML 변환 실패: {str(e2)}"
                    }

            elements.append({
                "type": "table",
                "content": table_content,
                "page": item.prov[0].page_no if item.prov else None,
                "level": level,
                "element": item
            })

    return elements

# 1단계: PDF 파일 처리
processor = AdvancedDocProcessor(
    enable_ocr=False,
    enable_table_structure=True
)

pdf_path = "data/transformer.pdf"
document = processor.process_pdf(pdf_path)

# 2단계: 문서 요소 추출
ordered_json = convert_document_to_ordered_json(document)

# 3단계: 테이블 필터링
table_elements = [item for item in ordered_json if item['type'] == 'table']

print(f"\n📊 테이블 추출 완료")
print(f"   - 총 테이블 개수: {len(table_elements)}개")

# 4단계: 테이블 저장
save_folder = Path("data/docling_output")
save_folder.mkdir(parents=True, exist_ok=True)

for idx, table in enumerate(table_elements, start=1):
    # 테이블 정보 출력
    print(f"\n테이블 {idx}:")
    print(f"   - 페이지: {table['page']}")

    if table['content']['status'] == 'success':
        # DataFrame 크기 확인
        df = pd.DataFrame(table['content']['data'])
        print(f"   - 크기: {df.shape[0]} rows × {df.shape[1]} columns")

        # 마크다운 파일로 저장
        table_path = save_folder / f"table_{idx}.md"
        with open(table_path, "w", encoding="utf-8") as f:
            f.write(f"# 테이블 {idx}\n\n")
            f.write(f"**페이지**: {table['page']}\n\n")
            f.write(table['content']['markdown'])

        print(f"   - 저장: {table_path}")
    else:
        print(f"   ⚠️ 오류: {table['content'].get('error')}")
```

### 솔루션 3: 텍스트와 테이블 요소 분리 및 저장

```python
# (이전 솔루션의 클래스와 함수 사용)

# 1단계: PDF 파일 처리
processor = AdvancedDocProcessor(
    enable_ocr=False,
    enable_table_structure=True
)

pdf_path = "data/transformer.pdf"
document = processor.process_pdf(pdf_path)

# 2단계: 문서 요소 추출
ordered_json = convert_document_to_ordered_json(document)

# 3단계: 텍스트와 테이블 요소 분리
text_elements = []
table_elements = []

for item in ordered_json:
    if item['type'] == 'text':
        text_elements.append(item)
    elif item['type'] == 'table':
        table_elements.append(item)

print(f"📊 요소 분리 완료")
print(f"   - 텍스트 요소: {len(text_elements)}개")
print(f"   - 테이블 요소: {len(table_elements)}개")

# 4단계: 텍스트 요소 저장
save_folder = Path("data/docling_output")
save_folder.mkdir(parents=True, exist_ok=True)

text_content = []
for item in text_elements:
    if item['content']:  # 내용이 있는 경우만
        text_content.append(item['content'])

text_path = save_folder / "transformer_text.txt"
with open(text_path, "w", encoding="utf-8") as f:
    f.write("\n".join(text_content))

print(f"\n✅ 텍스트 파일 저장: {text_path}")
print(f"   - 총 {len(text_elements)}개 요소 중 {len(text_content)}개 저장")
print(f"   - 전체 텍스트 길이: {len(''.join(text_content))}자")

# 5단계: 테이블 요소 저장
tables_md_content = []
for idx, table in enumerate(table_elements, start=1):
    tables_md_content.append(f"## 테이블 {idx}\n")
    tables_md_content.append(f"**페이지**: {table['page']}\n\n")

    if table['content']['status'] == 'success':
        tables_md_content.append(table['content']['markdown'])
    elif table['content']['status'] == 'html_fallback':
        tables_md_content.append(f"```html\n{table['content']['html']}\n```")
    else:
        tables_md_content.append(f"*오류: {table['content']['error']}*")

    tables_md_content.append("\n\n---\n\n")

tables_md_path = save_folder / "transformer_tables.md"
with open(tables_md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(tables_md_content))

print(f"\n✅ 테이블 마크다운 저장: {tables_md_path}")
print(f"   - 총 {len(table_elements)}개 테이블 저장")

# 6단계: 결과 요약
print(f"\n" + "="*60)
print(f"📊 처리 결과 요약")
print(f"="*60)
print(f"문서명: {document.name}")
print(f"총 요소 수: {len(ordered_json)}개")
print(f"  - 텍스트: {len(text_elements)}개")
print(f"  - 테이블: {len(table_elements)}개")
print(f"\n저장된 파일:")
print(f"  - {text_path}")
print(f"  - {tables_md_path}")
print(f"="*60)
```

### 솔루션 4: 대용량 문서 배치 처리

```python
from glob import glob
import pandas as pd
from pathlib import Path

# (이전 솔루션의 클래스와 함수 사용)

# 1단계: 모든 PDF 파일 검색
pdf_files = glob("data/*.pdf")
print(f"📂 총 {len(pdf_files)}개 PDF 파일 발견")

# 2단계: 통계 저장 리스트
stats_list = []

# 3단계: 처리기 초기화
processor = AdvancedDocProcessor(
    enable_ocr=False,
    enable_table_structure=True
)

# 4단계: 각 PDF 파일 처리
for idx, pdf_path in enumerate(pdf_files, start=1):
    print(f"\n{'='*60}")
    print(f"[{idx}/{len(pdf_files)}] 처리 중: {pdf_path}")
    print(f"{'='*60}")

    try:
        # PDF 처리
        document = processor.process_pdf(pdf_path)

        if document is None:
            print(f"⚠️ 처리 실패: {pdf_path}")
            continue

        # 문서 요소 추출
        ordered_json = convert_document_to_ordered_json(document)

        # 텍스트와 테이블 분리
        text_elements = [item for item in ordered_json if item['type'] == 'text']
        table_elements = [item for item in ordered_json if item['type'] == 'table']

        # 텍스트 길이 계산
        text_content = [item['content'] for item in text_elements if item['content']]
        total_text_length = len(''.join(text_content))

        # 페이지 수 계산
        page_numbers = set()
        for item in ordered_json:
            if item.get('page'):
                page_numbers.add(item['page'])
        num_pages = len(page_numbers)

        # 통계 저장
        stats = {
            '파일명': Path(pdf_path).name,
            '텍스트_길이': total_text_length,
            '테이블_개수': len(table_elements),
            '페이지_수': num_pages,
            '총_요소_수': len(ordered_json),
            '처리_상태': 'SUCCESS'
        }
        stats_list.append(stats)

        # 결과 출력
        print(f"✅ 처리 완료")
        print(f"   - 텍스트 길이: {total_text_length:,}자")
        print(f"   - 테이블 개수: {len(table_elements)}개")
        print(f"   - 페이지 수: {num_pages}페이지")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        stats = {
            '파일명': Path(pdf_path).name,
            '텍스트_길이': 0,
            '테이블_개수': 0,
            '페이지_수': 0,
            '총_요소_수': 0,
            '처리_상태': f'ERROR: {str(e)}'
        }
        stats_list.append(stats)

# 5단계: 통계 DataFrame 생성
stats_df = pd.DataFrame(stats_list)

# 6단계: CSV 파일로 저장
save_folder = Path("data/docling_output")
save_folder.mkdir(parents=True, exist_ok=True)

csv_path = save_folder / "pdf_processing_stats.csv"
stats_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

# 7단계: 전체 통계 출력
print(f"\n{'='*60}")
print(f"📊 전체 통계")
print(f"{'='*60}")
print(stats_df.to_string(index=False))
print(f"\n💾 CSV 파일 저장: {csv_path}")
```

## 🚀 실무 활용 예시

### 예시 1: 법률 문서 자동 분석 시스템

법률 문서에서 조항과 테이블을 자동으로 추출하여 검색 가능한 데이터베이스를 구축합니다.

```python
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorDevice
from docling_core.types.doc import TextItem, TableItem
import json
from pathlib import Path

class LegalDocumentProcessor:
    """법률 문서 처리 전문 클래스"""

    def __init__(self):
        # 테이블 구조 분석 활성화
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.accelerator_options.device = AcceleratorDevice.CPU

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

    def extract_articles(self, document):
        """조항 추출 (정규표현식 사용)"""
        import re

        articles = []
        for item, level in document.iterate_items():
            if isinstance(item, TextItem):
                # 조항 패턴: 제N조(제목) 형식
                pattern = r'제(\d+)조\(([^)]+)\)'
                matches = re.finditer(pattern, item.text)

                for match in matches:
                    article_num = match.group(1)
                    article_title = match.group(2)

                    articles.append({
                        "article_num": int(article_num),
                        "title": article_title,
                        "text": item.text,
                        "page": item.prov[0].page_no if item.prov else None
                    })

        return articles

    def extract_tables(self, document):
        """테이블 추출"""
        tables = []
        for item, level in document.iterate_items():
            if isinstance(item, TableItem):
                try:
                    df = item.export_to_dataframe()
                    tables.append({
                        "data": df.to_dict('records'),
                        "page": item.prov[0].page_no if item.prov else None,
                        "rows": df.shape[0],
                        "cols": df.shape[1]
                    })
                except:
                    pass

        return tables

    def process_legal_document(self, pdf_path):
        """법률 문서 처리"""
        print(f"⚖️ 법률 문서 처리 시작: {pdf_path}")

        # PDF 변환
        result = self.converter.convert(str(pdf_path))

        if result.status.name not in ['SUCCESS', 'PARTIAL_SUCCESS']:
            print(f"❌ 변환 실패")
            return None

        document = result.document

        # 조항 추출
        articles = self.extract_articles(document)

        # 테이블 추출
        tables = self.extract_tables(document)

        # 결과 반환
        result = {
            "document_name": document.name,
            "articles": articles,
            "tables": tables,
            "total_articles": len(articles),
            "total_tables": len(tables)
        }

        print(f"✅ 처리 완료")
        print(f"   - 조항 개수: {len(articles)}개")
        print(f"   - 테이블 개수: {len(tables)}개")

        return result

# 사용 예시
processor = LegalDocumentProcessor()
result = processor.process_legal_document("data/labor_law.pdf")

# JSON으로 저장
if result:
    save_folder = Path("data/docling_output")
    save_folder.mkdir(parents=True, exist_ok=True)

    json_path = save_folder / "labor_law_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"💾 JSON 파일 저장: {json_path}")
```

### 예시 2: 재무 보고서 자동 분석 시스템

재무 보고서에서 재무 테이블을 자동으로 추출하여 시계열 분석을 수행합니다.

```python
import pandas as pd
from pathlib import Path

class FinancialReportProcessor:
    """재무 보고서 처리 전문 클래스"""

    def __init__(self):
        # OCR + 테이블 구조 분석 활성화
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.accelerator_options.device = AcceleratorDevice.CPU

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

    def extract_financial_tables(self, document):
        """재무 테이블 추출"""
        financial_tables = []

        for item, level in document.iterate_items():
            if isinstance(item, TableItem):
                try:
                    df = item.export_to_dataframe()

                    # 재무 테이블 필터링 (숫자 열이 많은 테이블)
                    numeric_cols = df.select_dtypes(include=['number']).columns

                    if len(numeric_cols) >= 2:  # 최소 2개 이상의 숫자 열
                        financial_tables.append({
                            "dataframe": df,
                            "page": item.prov[0].page_no if item.prov else None,
                            "numeric_columns": len(numeric_cols),
                            "total_columns": len(df.columns),
                            "rows": df.shape[0]
                        })
                except:
                    pass

        return financial_tables

    def analyze_revenue_growth(self, tables):
        """매출 성장률 분석"""
        revenue_data = []

        for table in tables:
            df = table['dataframe']

            # 'Revenue' 또는 '매출' 키워드 검색
            for col in df.columns:
                if 'revenue' in str(col).lower() or '매출' in str(col):
                    for idx, row in df.iterrows():
                        try:
                            value = pd.to_numeric(row[col], errors='coerce')
                            if pd.notna(value):
                                revenue_data.append({
                                    'page': table['page'],
                                    'column': col,
                                    'value': value
                                })
                        except:
                            pass

        return revenue_data

    def process_financial_report(self, pdf_path):
        """재무 보고서 처리"""
        print(f"💰 재무 보고서 처리 시작: {pdf_path}")

        # PDF 변환
        result = self.converter.convert(str(pdf_path))

        if result.status.name not in ['SUCCESS', 'PARTIAL_SUCCESS']:
            print(f"❌ 변환 실패")
            return None

        document = result.document

        # 재무 테이블 추출
        financial_tables = self.extract_financial_tables(document)

        # 매출 성장률 분석
        revenue_data = self.analyze_revenue_growth(financial_tables)

        print(f"✅ 처리 완료")
        print(f"   - 재무 테이블 개수: {len(financial_tables)}개")
        print(f"   - 매출 데이터 개수: {len(revenue_data)}개")

        return {
            "financial_tables": financial_tables,
            "revenue_data": revenue_data
        }

# 사용 예시
processor = FinancialReportProcessor()
result = processor.process_financial_report("data/tsla-20241231-gen.pdf")

# 재무 테이블 저장
if result and result['financial_tables']:
    save_folder = Path("data/docling_output/financial_tables")
    save_folder.mkdir(parents=True, exist_ok=True)

    for idx, table in enumerate(result['financial_tables'], start=1):
        df = table['dataframe']

        # CSV로 저장
        csv_path = save_folder / f"financial_table_{idx}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        print(f"💾 테이블 {idx} 저장: {csv_path}")
        print(f"   - 페이지: {table['page']}")
        print(f"   - 크기: {table['rows']} rows × {table['total_columns']} cols")
```

### 예시 3: RAG 시스템용 문서 전처리 파이프라인

Docling으로 처리한 문서를 LangChain과 통합하여 RAG 시스템을 구축합니다.

```python
from langchain_community.document_loaders import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from pathlib import Path
import os

class DoclingRAGPipeline:
    """Docling 기반 RAG 파이프라인"""

    def __init__(self, openai_api_key=None):
        """
        Args:
            openai_api_key: OpenAI API 키 (없으면 환경변수에서 자동 로드)
        """
        if openai_api_key:
            os.environ['OPENAI_API_KEY'] = openai_api_key

        # Embedding 모델 초기화
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

        # LLM 초기화
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )

        self.vectorstore = None
        self.qa_chain = None

    def load_documents(self, pdf_path):
        """Docling으로 문서 로드"""
        print(f"📂 문서 로드 중: {pdf_path}")

        # DoclingLoader 사용
        loader = DoclingLoader(
            file_path=str(pdf_path),
            export_type="markdown"  # 마크다운 형식으로 내보내기
        )

        documents = loader.load()

        print(f"✅ {len(documents)}개 문서 로드 완료")
        return documents

    def split_documents(self, documents):
        """문서 청킹"""
        print(f"✂️ 문서 청킹 중...")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = text_splitter.split_documents(documents)

        print(f"✅ {len(chunks)}개 청크 생성 완료")
        return chunks

    def create_vectorstore(self, chunks):
        """벡터 스토어 생성"""
        print(f"🔍 벡터 스토어 생성 중...")

        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )

        print(f"✅ 벡터 스토어 생성 완료")

    def create_qa_chain(self):
        """QA 체인 생성"""
        if self.vectorstore is None:
            raise ValueError("벡터 스토어가 생성되지 않았습니다.")

        print(f"🔗 QA 체인 생성 중...")

        # Retriever 설정
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        # QA 체인 생성
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        print(f"✅ QA 체인 생성 완료")

    def query(self, question):
        """질문 응답"""
        if self.qa_chain is None:
            raise ValueError("QA 체인이 생성되지 않았습니다.")

        print(f"\n❓ 질문: {question}")

        result = self.qa_chain.invoke({"query": question})

        print(f"\n💡 답변:")
        print(result['result'])

        print(f"\n📚 참고 문서:")
        for idx, doc in enumerate(result['source_documents'], start=1):
            print(f"\n[문서 {idx}]")
            print(doc.page_content[:200] + "...")

        return result

    def build_pipeline(self, pdf_path):
        """전체 파이프라인 실행"""
        # 1. 문서 로드
        documents = self.load_documents(pdf_path)

        # 2. 문서 청킹
        chunks = self.split_documents(documents)

        # 3. 벡터 스토어 생성
        self.create_vectorstore(chunks)

        # 4. QA 체인 생성
        self.create_qa_chain()

        print(f"\n🎉 RAG 파이프라인 구축 완료!")

# 사용 예시
pipeline = DoclingRAGPipeline()
pipeline.build_pipeline("data/labor_law.pdf")

# 질문 응답
pipeline.query("근로시간의 기준은 무엇인가요?")
pipeline.query("연장근로에 대한 규정을 설명해주세요.")
```

**출력 예시**:
```
📂 문서 로드 중: data/labor_law.pdf
✅ 1개 문서 로드 완료
✂️ 문서 청킹 중...
✅ 45개 청크 생성 완료
🔍 벡터 스토어 생성 중...
✅ 벡터 스토어 생성 완료
🔗 QA 체인 생성 중...
✅ QA 체인 생성 완료

🎉 RAG 파이프라인 구축 완료!

❓ 질문: 근로시간의 기준은 무엇인가요?

💡 답변:
근로시간의 기준은 1주 40시간, 1일 8시간을 초과할 수 없습니다.
다만, 당사자 간 합의에 따라 1주 12시간을 한도로 연장근로가 가능합니다.

📚 참고 문서:

[문서 1]
제50조(근로시간) ① 1주 간의 근로시간은 휴게시간을 제외하고 40시간을
초과할 수 없다. ② 1일의 근로시간은 휴게시간을 제외하고 8시간을
초과할 수 없다...
```

## 📖 참고 자료

### 공식 문서
- **Docling GitHub**: https://github.com/docling-project/docling
- **Docling 문서**: https://docling-project.github.io/docling/
- **LangChain Docling 통합**: https://python.langchain.com/docs/integrations/document_loaders/docling

### 비교 자료
- **PDF Data Extraction Benchmark 2025**: https://procycons.com/en/blogs/pdf-data-extraction-benchmark/
- **Unstructured GitHub**: https://github.com/Unstructured-IO/unstructured
- **LlamaParse 공식 사이트**: https://www.llamaindex.ai/llamaparse

### 추가 학습 자료
- **RapidOCR GitHub**: https://github.com/RapidAI/RapidOCR
- **ONNX Runtime**: https://onnxruntime.ai/
- **Pandas 공식 문서**: https://pandas.pydata.org/docs/

### 관련 블로그 및 튜토리얼
- **Docling: The Power of Open Source AI for Document Processing**: https://medium.com/@ibm-research
- **Building RAG Systems with Docling**: https://python.langchain.com/docs/tutorials/rag/
