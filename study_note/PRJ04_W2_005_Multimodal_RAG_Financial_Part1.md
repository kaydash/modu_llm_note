# 멀티모달 RAG 구현 - 금융 문서 분석 (Part 1)

## 학습 목표

이 실습을 완료하면 다음을 할 수 있습니다:

1. **멀티모달 RAG 이해**: 텍스트, 이미지, 테이블을 통합 처리하는 RAG 시스템의 3가지 옵션을 이해합니다
2. **PDF 파싱 기술**: PDF 문서에서 이미지, 텍스트, 테이블을 자동으로 분리하고 구조화합니다
3. **CLIP 임베딩 활용**: 다국어 CLIP 모델로 이미지와 텍스트를 동일한 벡터 공간에 임베딩합니다
4. **벡터 DB 설계**: 멀티모달 데이터를 효율적으로 저장하고 검색하는 벡터 데이터베이스를 구축합니다
5. **RAG 파이프라인 구현**: 멀티모달 검색 결과를 통합하여 고품질 답변을 생성하는 시스템을 완성합니다

---

## 핵심 개념

### 1. 멀티모달 RAG란?

**멀티모달 RAG (Retrieval-Augmented Generation)**는 텍스트뿐만 아니라 이미지, 테이블, 차트 등 다양한 형태의 데이터를 검색하고 활용하여 답변을 생성하는 시스템입니다.

**전통적 RAG vs 멀티모달 RAG:**

| 구분 | 전통적 RAG | 멀티모달 RAG |
|------|-----------|-------------|
| **입력 데이터** | 텍스트만 | 텍스트 + 이미지 + 테이블 + 차트 |
| **임베딩** | 텍스트 임베딩 (OpenAI, BERT 등) | 멀티모달 임베딩 (CLIP, BLIP 등) |
| **검색 대상** | 텍스트 청크 | 텍스트 + 이미지 + 테이블 |
| **LLM 입력** | 텍스트만 | 텍스트 + 이미지 (멀티모달 LLM) |
| **활용 문서** | 텍스트 중심 문서 | 금융 보고서, 의료 자료, 기술 문서 등 |

### 2. 멀티모달 RAG 3가지 구현 옵션

| 옵션 | 임베딩 방식 | 벡터 DB 저장 | 답변 생성 LLM | 이미지 활용 | 장점 | 단점 |
|------|-----------|------------|--------------|----------|------|------|
| **옵션 1<br>멀티모달 임베딩** | CLIP (이미지+텍스트) | 이미지 임베딩<br>텍스트 임베딩 | GPT-4o<br>(Multimodal) | 원본 이미지<br>직접 활용<br>(base64) | • 최고 이미지 활용도<br>• 높은 답변 품질<br>• 시각 정보 완전 활용 | • 높은 비용<br>• base64 오버헤드<br>• 처리 시간 증가 |
| **옵션 2<br>텍스트 요약** | OpenAI<br>(텍스트만) | 텍스트 임베딩<br>(이미지 요약) | GPT-4o-mini<br>(Text-only) | 이미지 요약<br>텍스트로 변환 | • 비용 효율성<br>• 기존 RAG 인프라 활용<br>• 빠른 처리 속도 | • 이미지 정보 손실<br>• 답변 품질 제한적<br>• 시각 디테일 부족 |
| **옵션 3<br>하이브리드** | OpenAI<br>(텍스트만) | 텍스트 임베딩<br>(이미지 요약 + 참조) | GPT-4o<br>(Multimodal) | 요약 검색 후<br>원본 이미지 참조 | • 품질과 효율성 균형<br>• 이미지 손실 감소<br>• 유연한 구조 | • 옵션 1보다 품질 낮음<br>• 이미지 참조 관리 필요<br>• 복잡한 파이프라인 |

**선택 가이드:**
- 🏆 **최고 품질 답변이 필요한 경우**: 옵션 1 (비용 감수)
- 💰 **비용 효율성이 최우선인 경우**: 옵션 2 (품질 제한 감수)
- ⚖️ **품질과 효율성의 균형**: 옵션 3 (권장)

### 3. PDF 문서 파싱 전략

**금융 문서의 특징:**
- 복잡한 레이아웃 (다단 구성, 헤더/푸터)
- 다양한 요소 혼재 (텍스트, 이미지, 차트, 표)
- 구조화된 정보 (재무제표, 통계 데이터)

**파싱 프로세스:**

```
PDF 문서
    ↓
[파싱 엔진]
    ↓
┌─────────┬──────────┬─────────┐
│  텍스트   │  이미지   │  테이블  │
└─────────┴──────────┴─────────┘
    ↓          ↓          ↓
[청크 분할] [이미지 추출] [표 구조화]
    ↓          ↓          ↓
[텍스트    [이미지     [테이블
 임베딩]    임베딩]     임베딩]
    ↓          ↓          ↓
    Vector Database
```

**사용 라이브러리:**
- **unstructured**: 고급 PDF 파싱, 요소 타입 자동 분류
- **pdf2image**: PDF를 이미지로 변환
- **PIL (Pillow)**: 이미지 처리 및 저장
- **pdfplumber**: 테이블 추출

### 4. CLIP (Contrastive Language-Image Pre-training)

**CLIP의 작동 원리:**

```
[이미지] ──────► Image Encoder ──────┐
                                      ├──► Similarity Score
[텍스트] ──────► Text Encoder ───────┘
```

**핵심 특징:**
- 이미지와 텍스트를 동일한 벡터 공간에 임베딩
- 코사인 유사도로 이미지-텍스트 매칭
- 다국어 모델 지원 (한국어, 영어 등)
- Zero-shot 학습 가능

**멀티모달 RAG에서의 역할:**
- 텍스트 쿼리로 관련 이미지 검색
- 이미지 쿼리로 유사 이미지 검색
- 이미지-텍스트 통합 검색

### 5. 벡터 데이터베이스 설계

**Chroma DB 구조:**

```python
Collection: "multimodal_financial_docs"
├─ Document 1:
│  ├─ id: "text_001"
│  ├─ embedding: [0.123, -0.456, ...]  # 512-dim
│  ├─ metadata:
│  │  ├─ type: "text"
│  │  ├─ page: 1
│  │  └─ source: "annual_report.pdf"
│  └─ content: "2024년 매출액은..."
│
├─ Document 2:
│  ├─ id: "image_001"
│  ├─ embedding: [0.789, -0.234, ...]  # 512-dim
│  ├─ metadata:
│  │  ├─ type: "image"
│  │  ├─ page: 3
│  │  ├─ image_path: "images/chart_001.png"
│  │  └─ summary: "2024년 분기별 매출 추이 차트"
│  └─ content: "차트 요약 텍스트"
```

**메타데이터 활용:**
- `type`: 텍스트/이미지/테이블 구분
- `page`: 원본 페이지 번호
- `image_path`: 원본 이미지 파일 경로 (옵션 3)
- `summary`: 이미지/테이블 요약 (옵션 2, 3)

---

## 환경 설정

### 필요한 라이브러리 설치

```bash
# PDF 파싱
pip install unstructured pdf2image pdfplumber
pip install "unstructured[pdf]"

# 이미지 처리
pip install pillow opencv-python

# 멀티모달 임베딩
pip install open-clip-torch
pip install langchain-experimental

# 벡터 DB
pip install chromadb

# LangChain 및 LLM
pip install langchain langchain-openai langchain-chroma
pip install python-dotenv

# 유틸리티
pip install tqdm numpy
```

### 환경 변수 설정

`.env` 파일을 생성하고 API 키를 설정합니다:

```env
OPENAI_API_KEY=your_openai_key
```

### 기본 import

```python
from dotenv import load_dotenv
load_dotenv()

import os
from pathlib import Path
from typing import List, Dict, Any

# PDF 파싱
from unstructured.partition.pdf import partition_pdf
from pdf2image import convert_from_path
import pdfplumber

# 이미지 처리
from PIL import Image
import base64
from io import BytesIO

# 멀티모달 임베딩
from langchain_experimental.open_clip import OpenCLIPEmbeddings
import open_clip
import torch

# 벡터 DB
from langchain_chroma import Chroma
from langchain.schema import Document

# LLM
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

# 유틸리티
from tqdm import tqdm
import numpy as np

import warnings
warnings.filterwarnings('ignore')
```

---

## 단계별 구현

### 1단계: PDF 문서 파싱 - 이미지, 텍스트, 테이블 분리

PDF 문서를 구성 요소별로 분리하고 추출합니다.

```python
from unstructured.partition.pdf import partition_pdf
from PIL import Image
import os

class PDFParser:
    """PDF 문서를 이미지, 텍스트, 테이블로 분리하는 파서"""

    def __init__(self, output_dir: str = "./parsed_documents"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/tables", exist_ok=True)

    def parse_pdf(
        self,
        pdf_path: str,
        extract_images: bool = True,
        extract_tables: bool = True
    ) -> Dict[str, List]:
        """
        PDF를 파싱하여 요소별로 분리

        Args:
            pdf_path: PDF 파일 경로
            extract_images: 이미지 추출 여부
            extract_tables: 테이블 추출 여부

        Returns:
            {
                'texts': [텍스트 요소 리스트],
                'images': [이미지 정보 리스트],
                'tables': [테이블 정보 리스트]
            }
        """

        print(f"📄 PDF 파싱 시작: {pdf_path}")

        # unstructured 라이브러리로 PDF 파싱
        raw_elements = partition_pdf(
            filename=pdf_path,
            extract_images_in_pdf=extract_images,
            infer_table_structure=extract_tables,
            chunking_strategy="by_title",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
            image_output_dir_path=f"{self.output_dir}/images"
        )

        # 요소 타입별로 분류
        texts = []
        images = []
        tables = []

        for idx, element in enumerate(raw_elements):
            element_type = type(element).__name__

            # 텍스트 요소
            if element_type in ["Title", "NarrativeText", "Text", "ListItem"]:
                texts.append({
                    "type": "text",
                    "content": str(element),
                    "metadata": element.metadata.to_dict() if hasattr(element, 'metadata') else {},
                    "index": idx
                })

            # 이미지 요소
            elif element_type == "Image" and extract_images:
                # 이미지 파일 경로 추출
                image_path = None
                if hasattr(element.metadata, 'image_path'):
                    image_path = element.metadata.image_path

                images.append({
                    "type": "image",
                    "image_path": image_path,
                    "metadata": element.metadata.to_dict() if hasattr(element, 'metadata') else {},
                    "index": idx
                })

            # 테이블 요소
            elif element_type == "Table" and extract_tables:
                tables.append({
                    "type": "table",
                    "content": str(element),
                    "html": element.metadata.text_as_html if hasattr(element.metadata, 'text_as_html') else None,
                    "metadata": element.metadata.to_dict() if hasattr(element, 'metadata') else {},
                    "index": idx
                })

        print(f"✅ 파싱 완료:")
        print(f"  - 텍스트: {len(texts)}개")
        print(f"  - 이미지: {len(images)}개")
        print(f"  - 테이블: {len(tables)}개")

        return {
            "texts": texts,
            "images": images,
            "tables": tables
        }

# 사용 예시
parser = PDFParser(output_dir="./financial_documents")

# PDF 파싱
parsed_data = parser.parse_pdf(
    pdf_path="data/company_report.pdf",
    extract_images=True,
    extract_tables=True
)

# 결과 확인
print("\n=== 텍스트 샘플 ===")
for text in parsed_data['texts'][:3]:
    print(f"- {text['content'][:100]}...")

print("\n=== 이미지 샘플 ===")
for img in parsed_data['images'][:3]:
    print(f"- 경로: {img['image_path']}")

print("\n=== 테이블 샘플 ===")
for table in parsed_data['tables'][:3]:
    print(f"- {table['content'][:100]}...")
```

**출력 예시:**
```
📄 PDF 파싱 시작: data/company_report.pdf
✅ 파싱 완료:
  - 텍스트: 45개
  - 이미지: 12개
  - 테이블: 8개

=== 텍스트 샘플 ===
- 2024년 연간 실적 보고서...
- 당사는 전년 대비 15% 성장을 달성했습니다...
- 주요 제품 라인업은 다음과 같습니다...

=== 이미지 샘플 ===
- 경로: ./financial_documents/images/figure_001.png
- 경로: ./financial_documents/images/figure_002.png
- 경로: ./financial_documents/images/figure_003.png

=== 테이블 샘플 ===
- 분기 | 매출 | 영업이익 | 순이익 Q1 | 100억 | 20억 | 15억...
```

### 2단계: 이미지 요약 생성 (옵션 2, 3용)

이미지를 텍스트로 요약하여 텍스트 임베딩에 활용합니다.

```python
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import base64

class ImageSummarizer:
    """이미지를 텍스트로 요약하는 클래스"""

    def __init__(self, model_name="gpt-4o-mini"):
        self.model = ChatOpenAI(model=model_name, temperature=0)

    def encode_image_to_base64(self, image_path: str) -> str:
        """이미지를 base64로 인코딩"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def summarize_image(self, image_path: str) -> str:
        """
        이미지를 분석하여 텍스트 요약 생성

        Args:
            image_path: 이미지 파일 경로

        Returns:
            이미지 요약 텍스트
        """

        # 이미지 인코딩
        image_data = self.encode_image_to_base64(image_path)

        # 프롬프트
        prompt = """이 이미지를 분석하여 다음 정보를 포함하는 상세한 요약을 작성하세요:

1. 이미지 타입: 차트, 그래프, 사진, 다이어그램 등
2. 주요 내용: 핵심 정보와 데이터
3. 시각적 요소: 색상, 레이아웃, 구성
4. 주요 인사이트: 이미지에서 얻을 수 있는 핵심 통찰

요약은 검색과 정보 검색에 최적화된 형태로 작성하세요."""

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                }
            ]
        )

        response = self.model.invoke([message])
        return response.content

    def batch_summarize(self, image_paths: List[str]) -> List[Dict[str, str]]:
        """여러 이미지 일괄 요약"""
        results = []

        print(f"🖼️ 총 {len(image_paths)}개 이미지 요약 시작...")

        for idx, image_path in enumerate(image_paths, 1):
            print(f"[{idx}/{len(image_paths)}] {image_path} 처리 중...")

            try:
                summary = self.summarize_image(image_path)
                results.append({
                    "image_path": image_path,
                    "summary": summary,
                    "success": True
                })
                print(f"  ✅ 완료: {summary[:100]}...")

            except Exception as e:
                print(f"  ❌ 오류: {str(e)}")
                results.append({
                    "image_path": image_path,
                    "summary": "",
                    "success": False,
                    "error": str(e)
                })

        return results

# 사용 예시
summarizer = ImageSummarizer()

# 단일 이미지 요약
summary = summarizer.summarize_image("./financial_documents/images/figure_001.png")
print("=== 이미지 요약 ===")
print(summary)

# 여러 이미지 일괄 요약
image_list = [img['image_path'] for img in parsed_data['images'] if img['image_path']]
summaries = summarizer.batch_summarize(image_list)
```

**출력 예시:**
```
🖼️ 총 12개 이미지 요약 시작...
[1/12] ./financial_documents/images/figure_001.png 처리 중...
  ✅ 완료: 이미지 타입: 막대 그래프
주요 내용: 2024년 분기별 매출액 추이를 나타낸 그래프입니다...

[2/12] ./financial_documents/images/figure_002.png 처리 중...
  ✅ 완료: 이미지 타입: 원형 차트 (파이 차트)
주요 내용: 제품 카테고리별 매출 비중을 보여줍니다...
```

### 3단계: CLIP 임베딩 모델 준비

다국어 CLIP 모델로 이미지와 텍스트를 임베딩합니다.

```python
from langchain_experimental.open_clip import OpenCLIPEmbeddings
import open_clip
import torch
from PIL import Image

class MultimodalEmbedder:
    """CLIP 기반 멀티모달 임베딩 생성기"""

    def __init__(
        self,
        model_name: str = "ViT-B-16-quickgelu",
        checkpoint: str = "./models/metaclip_400m/open_clip_pytorch_model.bin",
        device: str = None
    ):
        """
        Args:
            model_name: CLIP 모델 이름
            checkpoint: 모델 체크포인트 경로
            device: 실행 디바이스 (cuda/cpu)
        """

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 디바이스: {self.device}")

        # LangChain OpenCLIP 임베딩 (텍스트용)
        self.text_embedder = OpenCLIPEmbeddings(
            model_name=model_name,
            checkpoint=checkpoint
        )

        # Open CLIP 모델 직접 로드 (이미지용)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=checkpoint,
            device=self.device
        )
        self.model.eval()

        print(f"✅ CLIP 모델 로드 완료")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """텍스트를 임베딩 벡터로 변환"""
        embeddings = self.text_embedder.embed_documents(texts)
        return np.array(embeddings)

    def embed_images(self, image_paths: List[str]) -> np.ndarray:
        """이미지를 임베딩 벡터로 변환"""
        embeddings = []

        for image_path in image_paths:
            # 이미지 로드 및 전처리
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            # 임베딩 생성
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = self.model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            embeddings.append(image_features.cpu().numpy())

        return np.vstack(embeddings)

    def calculate_similarity(self, text_embedding: np.ndarray, image_embeddings: np.ndarray) -> np.ndarray:
        """텍스트-이미지 유사도 계산"""
        # 코사인 유사도
        similarity = np.matmul(text_embedding, image_embeddings.T)
        return similarity

# 사용 예시
embedder = MultimodalEmbedder(
    model_name="ViT-B-16-quickgelu",
    checkpoint="./models/metaclip_400m/open_clip_pytorch_model.bin"
)

# 텍스트 임베딩
texts = ["2024년 매출액 증가", "제품 라인업 다양화", "글로벌 시장 확장"]
text_embeddings = embedder.embed_texts(texts)
print(f"텍스트 임베딩 shape: {text_embeddings.shape}")  # (3, 512)

# 이미지 임베딩
image_paths = [img['image_path'] for img in parsed_data['images'][:5] if img['image_path']]
image_embeddings = embedder.embed_images(image_paths)
print(f"이미지 임베딩 shape: {image_embeddings.shape}")  # (5, 512)

# 유사도 계산
similarity = embedder.calculate_similarity(text_embeddings[0:1], image_embeddings)
print(f"유사도 행렬 shape: {similarity.shape}")  # (1, 5)
print(f"가장 유사한 이미지 인덱스: {np.argmax(similarity)}")
```

**출력 예시:**
```
🔧 디바이스: cuda
✅ CLIP 모델 로드 완료
텍스트 임베딩 shape: (3, 512)
이미지 임베딩 shape: (5, 512)
유사도 행렬 shape: (1, 5)
가장 유사한 이미지 인덱스: 2
```

### 4단계: 벡터 데이터베이스 인덱싱 (옵션 1: 멀티모달 임베딩)

CLIP 임베딩을 Chroma DB에 저장합니다.

```python
from langchain_chroma import Chroma
from langchain.schema import Document
from typing import List
import uuid

class MultimodalVectorDB:
    """멀티모달 벡터 데이터베이스 (옵션 1)"""

    def __init__(
        self,
        collection_name: str = "multimodal_financial_docs",
        persist_directory: str = "./chroma_db"
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # CLIP 임베딩 함수
        self.embedder = MultimodalEmbedder()

        # Chroma DB 초기화
        self.vectordb = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedder.text_embedder,
            persist_directory=persist_directory
        )

        print(f"✅ 벡터 DB 초기화 완료: {collection_name}")

    def index_texts(self, texts: List[Dict]) -> List[str]:
        """텍스트를 벡터 DB에 인덱싱"""

        documents = []
        for text_data in texts:
            doc = Document(
                page_content=text_data['content'],
                metadata={
                    "type": "text",
                    "index": text_data['index'],
                    **text_data.get('metadata', {})
                }
            )
            documents.append(doc)

        # 벡터 DB에 추가
        ids = self.vectordb.add_documents(documents)
        print(f"✅ {len(documents)}개 텍스트 인덱싱 완료")

        return ids

    def index_images(self, images: List[Dict], summaries: List[Dict]) -> List[str]:
        """이미지를 벡터 DB에 인덱싱 (요약 텍스트 사용)"""

        documents = []
        for img_data, summary_data in zip(images, summaries):
            if not summary_data['success']:
                continue

            # 이미지 요약을 document content로 사용
            doc = Document(
                page_content=summary_data['summary'],
                metadata={
                    "type": "image",
                    "image_path": img_data['image_path'],
                    "index": img_data['index'],
                    **img_data.get('metadata', {})
                }
            )
            documents.append(doc)

        # 벡터 DB에 추가
        ids = self.vectordb.add_documents(documents)
        print(f"✅ {len(documents)}개 이미지 인덱싱 완료")

        return ids

    def index_tables(self, tables: List[Dict]) -> List[str]:
        """테이블을 벡터 DB에 인덱싱"""

        documents = []
        for table_data in tables:
            doc = Document(
                page_content=table_data['content'],
                metadata={
                    "type": "table",
                    "index": table_data['index'],
                    "html": table_data.get('html'),
                    **table_data.get('metadata', {})
                }
            )
            documents.append(doc)

        # 벡터 DB에 추가
        ids = self.vectordb.add_documents(documents)
        print(f"✅ {len(documents)}개 테이블 인덱싱 완료")

        return ids

    def search(self, query: str, k: int = 5, filter_type: str = None) -> List[Document]:
        """
        유사도 검색

        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            filter_type: 필터링할 타입 (text, image, table)

        Returns:
            검색 결과 문서 리스트
        """

        if filter_type:
            filter_dict = {"type": filter_type}
            results = self.vectordb.similarity_search(query, k=k, filter=filter_dict)
        else:
            results = self.vectordb.similarity_search(query, k=k)

        return results

# 사용 예시
vectordb = MultimodalVectorDB(
    collection_name="financial_report_2024",
    persist_directory="./chroma_db"
)

# 텍스트 인덱싱
text_ids = vectordb.index_texts(parsed_data['texts'])

# 이미지 인덱싱 (요약과 함께)
image_ids = vectordb.index_images(parsed_data['images'], summaries)

# 테이블 인덱싱
table_ids = vectordb.index_tables(parsed_data['tables'])

# 검색 테스트
query = "2024년 매출액 증가 추이는?"
results = vectordb.search(query, k=5)

print(f"\n=== 검색 결과 (Top 5) ===")
for idx, doc in enumerate(results, 1):
    print(f"\n[{idx}] 타입: {doc.metadata['type']}")
    print(f"내용: {doc.page_content[:200]}...")
```

**출력 예시:**
```
✅ 벡터 DB 초기화 완료: financial_report_2024
✅ 45개 텍스트 인덱싱 완료
✅ 12개 이미지 인덱싱 완료
✅ 8개 테이블 인덱싱 완료

=== 검색 결과 (Top 5) ===

[1] 타입: table
내용: 분기 | 매출액 | 성장률
Q1 2024 | 250억 | 12%
Q2 2024 | 280억 | 15%
Q3 2024 | 310억 | 18%
Q4 2024 | 340억 | 20%...

[2] 타입: text
내용: 2024년 연간 매출액은 전년 대비 15% 증가한 1,180억원을 기록했습니다. 이는 주요 제품 라인의 경쟁력 강화와 글로벌 시장 확대에 기인합니다...

[3] 타입: image
내용: 이미지 타입: 막대 그래프
주요 내용: 2024년 분기별 매출액 추이를 나타낸 그래프입니다. Q1부터 Q4까지 지속적인 성장세를 보여주며...
```

### 5단계: RAG 파이프라인 구현

검색된 멀티모달 컨텍스트를 활용하여 답변을 생성합니다.

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
import base64

class MultimodalRAG:
    """멀티모달 RAG 시스템"""

    def __init__(
        self,
        vectordb: MultimodalVectorDB,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0
    ):
        self.vectordb = vectordb
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)

    def encode_image_to_base64(self, image_path: str) -> str:
        """이미지를 base64로 인코딩"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def retrieve(self, query: str, k: int = 5) -> Dict[str, List]:
        """멀티모달 검색"""

        # 벡터 DB에서 검색
        results = self.vectordb.search(query, k=k)

        # 타입별로 분류
        texts = []
        images = []
        tables = []

        for doc in results:
            if doc.metadata['type'] == 'text':
                texts.append(doc)
            elif doc.metadata['type'] == 'image':
                images.append(doc)
            elif doc.metadata['type'] == 'table':
                tables.append(doc)

        return {
            "texts": texts,
            "images": images,
            "tables": tables
        }

    def generate_answer(self, query: str, retrieved_context: Dict) -> str:
        """
        검색된 컨텍스트를 바탕으로 답변 생성

        Args:
            query: 사용자 질문
            retrieved_context: retrieve() 결과

        Returns:
            생성된 답변
        """

        # 텍스트 컨텍스트 구성
        text_context = "\n\n".join([
            f"[{doc.metadata['type'].upper()}]\n{doc.page_content}"
            for doc in retrieved_context['texts'] + retrieved_context['tables']
        ])

        # 이미지가 있는 경우 멀티모달 프롬프트
        if retrieved_context['images']:
            # 첫 번째 이미지만 사용 (토큰 절약)
            image_doc = retrieved_context['images'][0]
            image_path = image_doc.metadata.get('image_path')

            if image_path and os.path.exists(image_path):
                image_data = self.encode_image_to_base64(image_path)

                prompt = f"""다음 정보를 바탕으로 질문에 답변하세요:

질문: {query}

텍스트 컨텍스트:
{text_context}

이미지 설명:
{image_doc.page_content}

제공된 이미지와 텍스트 정보를 종합하여 정확하고 자세한 답변을 제공하세요."""

                message = HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                        }
                    ]
                )

                response = self.llm.invoke([message])
                return response.content

        # 이미지가 없는 경우 텍스트만
        prompt = f"""다음 정보를 바탕으로 질문에 답변하세요:

질문: {query}

컨텍스트:
{text_context}

제공된 정보를 바탕으로 정확하고 자세한 답변을 제공하세요."""

        response = self.llm.invoke(prompt)
        return response.content

    def query(self, question: str, k: int = 5) -> Dict:
        """
        질문에 대한 답변 생성 (검색 + 생성)

        Args:
            question: 사용자 질문
            k: 검색할 문서 수

        Returns:
            {
                'answer': 답변,
                'context': 검색된 컨텍스트,
                'sources': 출처 정보
            }
        """

        print(f"🔍 질문: {question}")

        # 1. 검색
        context = self.retrieve(question, k=k)
        print(f"📚 검색 결과: 텍스트 {len(context['texts'])}개, "
              f"이미지 {len(context['images'])}개, 테이블 {len(context['tables'])}개")

        # 2. 답변 생성
        answer = self.generate_answer(question, context)

        # 3. 출처 정보 정리
        sources = []
        for doc_list in [context['texts'], context['images'], context['tables']]:
            for doc in doc_list:
                sources.append({
                    "type": doc.metadata['type'],
                    "content_preview": doc.page_content[:100],
                    "metadata": doc.metadata
                })

        return {
            "answer": answer,
            "context": context,
            "sources": sources
        }

# 사용 예시
rag = MultimodalRAG(vectordb=vectordb, model_name="gpt-4o-mini")

# 질문 1: 텍스트 + 테이블
question1 = "2024년 매출액 증가 추이를 설명해주세요."
result1 = rag.query(question1, k=5)

print(f"\n=== 질문 1 ===")
print(f"Q: {question1}")
print(f"\nA: {result1['answer']}")
print(f"\n출처: {len(result1['sources'])}개 문서")

# 질문 2: 이미지 포함
question2 = "제품 카테고리별 매출 비중을 분석해주세요."
result2 = rag.query(question2, k=5)

print(f"\n=== 질문 2 ===")
print(f"Q: {question2}")
print(f"\nA: {result2['answer']}")
print(f"\n출처: {len(result2['sources'])}개 문서")
```

**출력 예시:**
```
🔍 질문: 2024년 매출액 증가 추이를 설명해주세요.
📚 검색 결과: 텍스트 3개, 이미지 1개, 테이블 1개

=== 질문 1 ===
Q: 2024년 매출액 증가 추이를 설명해주세요.

A: 2024년 매출액은 분기별로 지속적인 성장세를 보였습니다.

분기별 매출액:
- Q1: 250억원 (12% 성장)
- Q2: 280억원 (15% 성장)
- Q3: 310억원 (18% 성장)
- Q4: 340억원 (20% 성장)

연간 매출액은 1,180억원으로 전년 대비 15% 증가했습니다.
주요 성장 요인은 주력 제품의 경쟁력 강화와 글로벌 시장 확대입니다.

출처: 5개 문서
```

---

## 정리

이번 Part 1에서 학습한 내용:

### 핵심 개념
1. **멀티모달 RAG 3가지 옵션**: 멀티모달 임베딩, 텍스트 요약, 하이브리드
2. **PDF 파싱**: unstructured로 이미지/텍스트/테이블 자동 분리
3. **CLIP 임베딩**: 이미지와 텍스트를 동일한 벡터 공간에 임베딩
4. **벡터 DB 설계**: 메타데이터 활용한 효율적인 멀티모달 저장
5. **RAG 파이프라인**: 검색 → 컨텍스트 통합 → 답변 생성

### 구현 완료
- ✅ PDF 파서 (PDFParser 클래스)
- ✅ 이미지 요약기 (ImageSummarizer 클래스)
- ✅ CLIP 임베더 (MultimodalEmbedder 클래스)
- ✅ 벡터 DB (MultimodalVectorDB 클래스)
- ✅ RAG 시스템 (MultimodalRAG 클래스)

### Part 2에서 계속...

Part 2에서는 다음 내용을 다룹니다:
- 옵션 2, 3 구현 (텍스트 임베딩 기반)
- 성능 비교 및 최적화
- 실습 문제 및 솔루션
- 프로덕션 환경 배포 전략

멀티모달 RAG는 금융, 의료, 기술 문서 등 복잡한 정보를 담은 문서를 효과적으로 분석하는 강력한 도구입니다!
