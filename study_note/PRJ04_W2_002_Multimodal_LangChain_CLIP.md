# PRJ04_W2_002: 멀티모달 학습과 CLIP을 활용한 이미지-텍스트 검색

## 📚 학습 목표

이 가이드를 완료하면 다음을 수행할 수 있습니다:

1. **멀티모달 학습의 개념 이해**: 텍스트, 이미지 등 다양한 데이터 형식을 동시에 처리하는 AI 접근 방식 파악
2. **CLIP 모델 활용**: OpenAI의 CLIP을 사용하여 이미지와 텍스트 간 의미적 유사성 계산
3. **임베딩 기반 검색 구현**: FAISS 벡터 데이터베이스를 활용한 대규모 이미지 검색 시스템 구축
4. **크로스모달 검색 수행**: 텍스트 쿼리로 이미지 검색, 이미지 쿼리로 유사 이미지 검색
5. **다국어 멀티모달 시스템 개발**: XLM-RoBERTa 기반 모델로 다국어 이미지 검색 구현

## 🔑 핵심 개념

### 멀티모달 학습 (Multi-Modal Learning)

**정의**
- 텍스트, 이미지, 오디오, 비디오 등 **서로 다른 형식의 데이터**를 동시에 처리하는 AI 접근 방식
- 인간의 감각 통합 방식과 유사하게 여러 정보 채널을 **동시에 학습**하여 더 깊은 이해 제공

**주요 모달리티 (Modality)**

| 모달리티 | 데이터 유형 | 특화 모델 | 활용 예시 |
|----------|-------------|-----------|-----------|
| 텍스트 | 자연어 문장, 문서, 메타데이터 | BERT, GPT, T5 | 문서 분류, 감정 분석 |
| 이미지 | 사진, 그래프, 차트, 다이어그램 | CNN, Vision Transformer | 객체 인식, 이미지 분류 |
| 오디오 | 음성, 음악, 환경음 | WaveNet, Mel-Spectrogram | 음성 인식, 음악 생성 |
| 비디오 | 동영상 시퀀스 | 3D CNN, TimeSformer | 행동 인식, 비디오 요약 |
| 센서 | IoT 센서, 생체신호 | RNN, LSTM | 이상 탐지, 예측 분석 |

**모달리티 특징 융합 (Feature Fusion) 방법**

1. **Early Fusion (조기 융합)**
   - 입력 단계에서 모달리티 특징을 **직접 결합**
   - 장점: 모달리티 간 저수준 상호작용 학습
   - 단점: 모달리티 불균형 시 성능 저하

2. **Late Fusion (후기 융합)**
   - 각 모달리티를 **개별적으로 처리** 후 최종 결과 통합
   - 장점: 각 모달리티의 독립적 학습 가능
   - 단점: 모달리티 간 상호작용 제한적

3. **Hybrid Fusion (하이브리드 융합)**
   - **다단계 특징 결합**을 통한 유연한 통합
   - 장점: Early와 Late의 장점 결합
   - 단점: 구조 복잡도 증가

### CLIP (Contrastive Language-Image Pre-training)

**핵심 개념**
- OpenAI가 개발한 **이미지-텍스트 멀티모달 모델**
- 4억 개의 이미지-텍스트 쌍으로 사전 학습
- **제로샷 학습 (Zero-Shot Learning)** 능력: 추가 학습 없이 새로운 작업 수행

**작동 원리**

```
[이미지] → Image Encoder (ViT/ResNet) → Image Embedding (512차원)
                                              ↓
                                          유사도 계산
                                              ↑
[텍스트] → Text Encoder (Transformer) → Text Embedding (512차원)
```

**주요 특징**

| 특징 | 설명 | 장점 |
|------|------|------|
| Contrastive Learning | 이미지-텍스트 쌍의 유사도를 최대화, 비관련 쌍은 최소화 | 의미적 연결 학습 |
| Zero-Shot Transfer | 특정 작업에 대한 추가 학습 없이 적용 가능 | 범용성 향상 |
| Natural Language Supervision | 자연어 설명으로 학습 | 사람의 의도 이해 |
| Scale | 대규모 데이터셋으로 학습 | 일반화 능력 우수 |

### OpenCLIP

- CLIP의 **오픈소스 구현**
- LangChain과 통합하여 사용 가능
- 다양한 크기와 성능의 모델 지원:
  - `ViT-B-32`: 작고 빠른 모델
  - `ViT-B-16`: 중간 크기, 균형잡힌 성능
  - `ViT-L-14`: 큰 모델, 높은 정확도
  - `xlm-roberta-base-ViT-B-32`: 다국어 지원

### FAISS (Facebook AI Similarity Search)

- Facebook AI Research에서 개발한 **고속 벡터 유사도 검색** 라이브러리
- 수백만~수억 개의 벡터에서 최근접 이웃 빠르게 검색
- GPU 가속 지원

**주요 인덱스 유형**

| 인덱스 유형 | 특징 | 사용 사례 |
|------------|------|-----------|
| IndexFlatL2 | 정확한 L2 거리 계산 | 소규모 데이터 (< 100만 개) |
| IndexIVFFlat | 클러스터 기반 근사 검색 | 중규모 데이터 (100만~1000만 개) |
| IndexHNSW | 그래프 기반 빠른 검색 | 대규모 데이터 (> 1000만 개) |

## 🛠 환경 설정

### 1. 필수 라이브러리 설치

```bash
# CLIP 및 LangChain 관련
pip install --upgrade --quiet langchain-experimental
pip install --upgrade --quiet open_clip_torch torch

# 이미지 처리
pip install pillow matplotlib

# 벡터 검색
pip install faiss-cpu  # CPU 버전
# pip install faiss-gpu  # GPU 버전 (CUDA 필요)

# 유틸리티
pip install sentence-transformers tqdm
```

### 2. 환경 변수 설정

`.env` 파일을 생성하고 HuggingFace 토큰을 설정합니다:

```bash
HF_TOKEN=your_huggingface_token_here
OPENAI_API_KEY=your_openai_api_key_here  # (선택사항)
```

HuggingFace 토큰은 [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)에서 생성할 수 있습니다.

### 3. 기본 Import

```python
from dotenv import load_dotenv
load_dotenv()

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# HuggingFace 인증
from huggingface_hub import login

hf_token = os.getenv('HF_TOKEN')
if hf_token:
    try:
        login(token=hf_token)
        print("✅ HuggingFace 인증 성공!")
    except Exception as e:
        print(f"⚠️ HuggingFace 인증 실패: {e}")
else:
    print("⚠️ HF_TOKEN 환경 변수가 설정되지 않았습니다.")

# PyTorch 버전 확인
print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
```

## 💻 단계별 구현

### 1단계: CLIP 모델 로드 및 기본 사용

#### 사용 가능한 모델 확인

```python
import open_clip

def check_available_models():
    """사용 가능한 OpenCLIP 모델 목록 확인"""
    models = open_clip.list_pretrained()
    print(f"총 {len(models)}개의 사전 학습 모델 사용 가능")
    print("\n주요 모델:")

    # 주요 모델만 출력
    main_models = [
        ('ViT-B-32', 'openai'),
        ('ViT-B-16', 'openai'),
        ('ViT-L-14', 'openai'),
        ('ViT-B-32', 'metaclip_400m'),
        ('xlm-roberta-base-ViT-B-32', 'laion5b_s13b_b90k'),
    ]

    for model_name, checkpoint in main_models:
        print(f"  - {model_name} ({checkpoint})")

check_available_models()
```

#### CLIP 모델 다운로드 및 로드

```python
from huggingface_hub import snapshot_download
from langchain_experimental.open_clip import OpenCLIPEmbeddings

# 1. 모델 다운로드 (처음 한 번만 실행)
model_dir = "./models/metaclip_400m"

snapshot_download(
    repo_id="timm/vit_base_patch16_clip_224.metaclip_400m",
    local_dir=model_dir,
    local_dir_use_symlinks=False
)

print(f"✅ 모델 다운로드 완료: {model_dir}")

# 2. LangChain OpenCLIPEmbeddings로 로드
clip_embd_model = OpenCLIPEmbeddings(
    model_name="ViT-B-16-quickgelu",
    checkpoint=f"{model_dir}/open_clip_pytorch_model.bin"
)

print("✅ CLIP 임베딩 모델 로드 완료")
```

**💡 모델 선택 가이드:**
- `ViT-B-32`: 빠른 속도, 작은 메모리 (약 300MB)
- `ViT-B-16`: 균형잡힌 성능 (약 350MB)
- `ViT-L-14`: 높은 정확도, 큰 메모리 (약 900MB)

### 2단계: 이미지-텍스트 임베딩 생성

#### 샘플 이미지 다운로드

```python
import urllib.request

# COCO 데이터셋에서 샘플 이미지 다운로드
urls = [
    "http://images.cocodataset.org/val2017/000000000285.jpg",  # 곰
    "http://images.cocodataset.org/val2017/000000039769.jpg",  # 고양이들
    "http://images.cocodataset.org/val2017/000000000776.jpg",  # 테디베어
]

# 이미지 설명
texts = ["bear", "cats", "Teddy bears"]

# 이미지 다운로드
for url, text in zip(urls, texts):
    fname = text.replace(" ", "_").lower() + ".jpg"
    urllib.request.urlretrieve(url, fname)
    print(f"✅ 다운로드 완료: {fname}")

print(f"\n총 {len(urls)}개의 샘플 이미지 준비 완료")
```

#### 이미지 시각화 함수

```python
def show_image(image_path, title=None):
    """이미지를 화면에 표시"""
    image = Image.open(image_path)
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.axis('off')
    if title:
        plt.title(title, fontsize=14)
    plt.show()

# 샘플 이미지 출력
image_paths = [f"{text.replace(' ', '_').lower()}.jpg" for text in texts]

for image_path, text in zip(image_paths, texts):
    show_image(image_path, title=text)
```

#### 임베딩 생성

```python
# 이미지와 텍스트 임베딩 생성
img_features = clip_embd_model.embed_image(image_paths)
text_features = clip_embd_model.embed_documents(texts)

# NumPy 배열로 변환
img_features_np = np.array(img_features)
text_features_np = np.array(text_features)

# 임베딩 크기 확인
print(f"이미지 임베딩 shape: {img_features_np.shape}")
print(f"텍스트 임베딩 shape: {text_features_np.shape}")
print(f"\n임베딩 차원: {img_features_np.shape[1]}")
```

**출력 예시:**
```
이미지 임베딩 shape: (3, 512)
텍스트 임베딩 shape: (3, 512)

임베딩 차원: 512
```

### 3단계: 이미지-텍스트 유사도 계산 및 시각화

#### 코사인 유사도 계산

```python
# 이미지-텍스트 유사도 계산 (코사인 유사도)
similarity = np.matmul(text_features_np, img_features_np.T)

print("유사도 행렬:")
print(similarity)
print(f"\nShape: {similarity.shape}")
```

**유사도 행렬 해석:**
- 각 행: 텍스트 쿼리
- 각 열: 이미지
- 값: 유사도 (0~1, 높을수록 유사)

#### 유사도 시각화

```python
# 결과 시각화
count = len(texts)
plt.figure(figsize=(20, 14))
plt.imshow(similarity, vmin=0.1, vmax=0.3, cmap='coolwarm')

# Y축: 텍스트 레이블
plt.yticks(range(count), texts, fontsize=18)
plt.xticks([])

# X축: 이미지 표시
for i, image_path in enumerate(image_paths):
    image = Image.open(image_path)
    plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")

# 유사도 점수 표시
for x in range(similarity.shape[1]):
    for y in range(similarity.shape[0]):
        score = similarity[y, x]
        color = 'white' if score > 0.2 else 'black'
        plt.text(x, y, f"{score:.2f}", ha="center", va="center",
                size=12, color=color, weight='bold')

# 그래프 정리
for side in ["left", "top", "right", "bottom"]:
    plt.gca().spines[side].set_visible(False)

plt.xlim([-0.5, count - 0.5])
plt.ylim([count + 0.5, -2])
plt.title("이미지-텍스트 코사인 유사도", size=20, pad=20)
plt.tight_layout()
plt.savefig('similarity_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ 유사도 시각화 완료: similarity_matrix.png")
```

**💡 해석 가이드:**
- 대각선 값이 높으면 올바른 매칭
- "bear" 텍스트는 곰 이미지와 높은 유사도
- 오분류 케이스 분석 가능

### 4단계: FAISS를 활용한 대규모 이미지 검색

#### UNSPLASH 데이터셋 준비

```python
import zipfile
from sentence_transformers import util

# 이미지 다운로드 (약 2GB)
img_folder = 'unsplash/'
photo_filename = 'unsplash-25k-photos.zip'

if not os.path.exists(img_folder) or len(os.listdir(img_folder)) == 0:
    os.makedirs(img_folder, exist_ok=True)

    if not os.path.exists(photo_filename):
        print("📥 UNSPLASH 데이터셋 다운로드 중... (약 2GB, 시간이 걸릴 수 있습니다)")
        util.http_get('http://sbert.net/datasets/'+photo_filename, photo_filename)

    print("📦 압축 해제 중...")
    with zipfile.ZipFile(photo_filename, 'r') as zf:
        for member in tqdm(zf.infolist(), desc='압축 해제'):
            zf.extract(member, img_folder)

# 이미지 파일 경로 리스트 생성
from glob import glob
img_files = glob(os.path.join(img_folder, '*.jpg'))

print(f"✅ 이미지 파일 개수: {len(img_files)}")
print(f"샘플 경로: {img_files[:3]}")
```

#### CLIP 모델로 이미지 임베딩 계산

```python
import open_clip

# 로컬 체크포인트에서 모델 로드
checkpoint_path = "./models/metaclip_400m/open_clip_pytorch_model.bin"
device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-16-quickgelu',
    pretrained=checkpoint_path,
    device=device
)
model.eval()

print(f"✅ 모델 로드 완료: {device}")

# 테스트를 위해 1000개만 사용 (전체 사용 시 시간이 오래 걸림)
img_files = img_files[:1000]

# 배치 처리로 임베딩 계산
img_embeddings = []
batch_size = 64

for i in tqdm(range(0, len(img_files), batch_size), desc='이미지 임베딩 계산'):
    batch_files = img_files[i:i+batch_size]

    # 배치 이미지 로드 및 전처리
    batch_images = []
    for img_path in batch_files:
        try:
            img = Image.open(img_path).convert('RGB')
            batch_images.append(preprocess(img))
        except Exception as e:
            print(f"이미지 로드 실패 {img_path}: {e}")
            continue

    if len(batch_images) == 0:
        continue

    # 배치 텐서로 변환
    batch_tensor = torch.stack(batch_images).to(device)

    # 임베딩 계산
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(batch_tensor)
        # 정규화 (코사인 유사도 계산을 위해)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # NumPy로 변환하여 저장
    batch_embeddings = image_features.cpu().numpy()
    img_embeddings.append(batch_embeddings)

# 모든 임베딩을 하나의 배열로 결합
img_embeddings = np.vstack(img_embeddings)

print(f"✅ 이미지 임베딩 Shape: {img_embeddings.shape}")
```

#### FAISS 인덱스 생성 및 저장

```python
import faiss

# FAISS 인덱스 생성 (L2 거리 기반)
embedding_dim = img_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)

# 임베딩 추가
index.add(img_embeddings)

# 인덱스 파일로 저장
faiss.write_index(index, 'img_embeddings.index')

print(f"✅ FAISS 인덱스 생성 완료")
print(f"   임베딩 차원: {embedding_dim}")
print(f"   인덱싱된 이미지 수: {index.ntotal}")
```

#### FAISS 인덱스 로드 및 검색

```python
# 저장된 인덱스 로드
index = faiss.read_index('img_embeddings.index')

print(f"✅ FAISS 인덱스 로드 완료")
print(f"   인덱싱된 이미지 수: {index.ntotal}")

# 이미지 기반 검색 함수
def search_similar_images(query_image_path, k=5):
    """
    쿼리 이미지와 유사한 이미지를 검색

    Args:
        query_image_path: 쿼리 이미지 경로
        k: 반환할 유사 이미지 개수

    Returns:
        distances: 거리 값 (낮을수록 유사)
        indices: 유사 이미지의 인덱스
    """
    # 이미지 로드 및 전처리
    query_img = Image.open(query_image_path).convert('RGB')
    query_tensor = preprocess(query_img).unsqueeze(0).to(device)

    # 이미지 임베딩 생성
    with torch.no_grad(), torch.cuda.amp.autocast():
        query_embedding = model.encode_image(query_tensor)
        query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)

    # NumPy 배열로 변환
    query_embedding = query_embedding.cpu().numpy().astype(np.float32)

    # FAISS 검색
    distances, indices = index.search(query_embedding, k)

    return distances[0], indices[0]

# 쿼리 이미지 선택
query_image_path = img_files[0]

print(f"📌 쿼리 이미지: {os.path.basename(query_image_path)}")
show_image(query_image_path, title="쿼리 이미지")

# 유사 이미지 검색
distances, indices = search_similar_images(query_image_path, k=5)

# 결과 출력
print("\n🔍 유사한 이미지 검색 결과:")
for i, (idx, dist) in enumerate(zip(indices, distances), 1):
    print(f"{i}. {os.path.basename(img_files[idx])} (거리: {dist:.4f})")
    show_image(img_files[idx], title=f"유사 이미지 {i} (거리: {dist:.4f})")
```

### 5단계: 텍스트 쿼리로 이미지 검색

#### 텍스트 기반 검색 함수

```python
def search_images_by_text(text_query, k=5):
    """
    텍스트 쿼리로 유사한 이미지 검색

    Args:
        text_query: 텍스트 쿼리 (영어 또는 한국어)
        k: 반환할 이미지 개수

    Returns:
        distances: 거리 값
        indices: 이미지 인덱스
    """
    # 텍스트 토큰화
    text_tokens = open_clip.tokenize([text_query]).to(device)

    # 텍스트 임베딩 생성
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_embedding = model.encode_text(text_tokens)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

    # NumPy 배열로 변환
    text_embedding = text_embedding.cpu().numpy().astype(np.float32)

    # FAISS 검색
    distances, indices = index.search(text_embedding, k)

    return distances[0], indices[0]

# 텍스트 쿼리 예시
text_queries = [
    "a black dog running in the park",
    "공원에서 달리는 검은 개",
    "a beautiful sunset over the ocean",
    "바다 위의 아름다운 일몰",
]

# 각 쿼리에 대해 이미지 검색
for text_query in text_queries:
    print("=" * 80)
    print(f"📝 텍스트 쿼리: {text_query}")
    print("=" * 80)

    # 유사 이미지 검색
    distances, indices = search_images_by_text(text_query, k=3)

    # 결과 출력
    for i, (idx, dist) in enumerate(zip(indices, distances), 1):
        print(f"\n{i}. {os.path.basename(img_files[idx])} (거리: {dist:.4f})")
        show_image(img_files[idx], title=f"{text_query[:50]}... (거리: {dist:.4f})")

    print()
```

**💡 검색 팁:**
- 구체적인 설명일수록 정확한 결과
- 영어와 한국어 모두 지원 (모델에 따라 차이)
- 거리 값이 낮을수록 더 유사

## 🎯 실습 문제

### 실습 1: 커스텀 텍스트로 이미지 검색 (기초)

**문제**: 다양한 텍스트 쿼리로 이미지를 검색하고 결과를 분석하세요.

**요구사항**:
- 최소 5개의 서로 다른 텍스트 쿼리 작성
- 각 쿼리당 상위 3개 이미지 검색
- 영어와 한국어 쿼리 모두 포함
- 검색 결과의 적합성 평가

**힌트**:
```python
custom_queries = [
    "자연 풍경 관련",
    "동물 관련",
    "도시 스카이라인",
    "음식 사진",
    "사람 초상화"
]
```

### 실습 2: 이미지-이미지 유사도 매트릭스 (응용)

**문제**: 여러 이미지 간의 유사도 매트릭스를 계산하고 시각화하세요.

**요구사항**:
- 5~10개의 이미지 선택
- 모든 이미지 쌍 간의 유사도 계산
- 히트맵으로 시각화
- 가장 유사한 이미지 쌍 찾기

**힌트**:
```python
def compute_similarity_matrix(image_paths):
    # 모든 이미지의 임베딩 계산
    # 유사도 행렬 생성
    # 히트맵 시각화
    pass
```

### 실습 3: 다국어 CLIP 모델 활용 (심화)

**문제**: XLM-RoBERTa 기반 다국어 CLIP 모델로 다국어 이미지 검색 시스템을 구축하세요.

**요구사항**:
- `xlm-roberta-base-ViT-B-32` 모델 다운로드 및 로드
- 새로운 FAISS 인덱스 생성
- 영어, 한국어, 일본어, 스페인어 쿼리로 검색
- 언어별 검색 결과 비교 분석

**힌트**:
```python
# 다국어 모델 로드
multilingual_model, _, multilingual_preprocess = open_clip.create_model_and_transforms(
    'xlm-roberta-base-ViT-B-32',
    pretrained='laion5b_s13b_b90k',
    device=device
)

multilingual_queries = {
    'en': "a beautiful sunset",
    'ko': "아름다운 일몰",
    'ja': "美しい夕日",
    'es': "una hermosa puesta de sol"
}
```

## ✅ 솔루션 예시

### 실습 1 솔루션

```python
# 커스텀 텍스트 쿼리 정의
custom_text_queries = [
    ("자연 풍경", "a beautiful landscape with mountains and forests"),
    ("자연 풍경 (한국어)", "산과 숲이 있는 아름다운 풍경"),
    ("동물", "wild animals in their natural habitat"),
    ("동물 (한국어)", "자연 서식지에 있는 야생 동물"),
    ("도시", "modern city skyline at sunset"),
    ("도시 (한국어)", "일몰 시의 현대적인 도시 스카이라인"),
    ("음식", "delicious food on a plate"),
    ("음식 (한국어)", "접시에 담긴 맛있는 음식"),
]

print("=" * 80)
print("실습 1: 다양한 텍스트 쿼리를 사용한 이미지 검색")
print("=" * 80)

# 각 쿼리에 대해 검색 수행
for category, text_query in custom_text_queries:
    print(f"\n{'=' * 80}")
    print(f"📂 카테고리: {category}")
    print(f"📝 쿼리: {text_query}")
    print(f"{'=' * 80}")

    # 유사 이미지 검색
    distances, indices = search_images_by_text(text_query, k=3)

    # 결과 출력
    for i, (idx, dist) in enumerate(zip(indices, distances), 1):
        print(f"\n유사 이미지 {i}: {os.path.basename(img_files[idx])}")
        print(f"거리(낮을수록 유사): {dist:.4f}")
        show_image(img_files[idx], title=f"{category} - {i} (거리: {dist:.4f})")

print(f"\n{'=' * 80}")
print("✅ 실습 1 완료")
print("=" * 80)

# 결과 분석
print("\n📊 분석:")
print("1. 영어와 한국어 쿼리 모두 이미지 검색이 가능합니다.")
print("2. 같은 의미의 영어/한국어 쿼리가 유사한 결과를 반환하는지 확인하세요.")
print("3. 구체적인 설명일수록 더 정확한 검색 결과를 얻을 수 있습니다.")
print("4. 거리 값이 낮을수록 텍스트와 이미지가 더 유사합니다.")
```

### 실습 2 솔루션

```python
import seaborn as sns

def compute_similarity_matrix(image_paths):
    """
    여러 이미지 간의 유사도 매트릭스 계산

    Args:
        image_paths: 이미지 경로 리스트

    Returns:
        similarity_matrix: 유사도 매트릭스 (NxN)
    """
    # 모든 이미지의 임베딩 계산
    embeddings = []

    print("🔄 이미지 임베딩 계산 중...")
    for img_path in tqdm(image_paths):
        img = Image.open(img_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            embedding = model.encode_image(img_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        embeddings.append(embedding.cpu().numpy())

    embeddings = np.vstack(embeddings)

    # 유사도 행렬 계산 (코사인 유사도)
    similarity_matrix = np.matmul(embeddings, embeddings.T)

    return similarity_matrix

# 샘플 이미지 선택 (다양한 주제)
sample_indices = [0, 50, 100, 200, 300, 400, 500, 600, 700, 800]
sample_paths = [img_files[i] for i in sample_indices]

# 유사도 매트릭스 계산
similarity_matrix = compute_similarity_matrix(sample_paths)

# 시각화
plt.figure(figsize=(12, 10))

# 히트맵
sns.heatmap(
    similarity_matrix,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    vmin=0,
    vmax=1,
    xticklabels=[os.path.basename(p)[:10] for p in sample_paths],
    yticklabels=[os.path.basename(p)[:10] for p in sample_paths],
    cbar_kws={'label': '코사인 유사도'}
)

plt.title('이미지 간 유사도 매트릭스', fontsize=16, pad=20)
plt.xlabel('이미지', fontsize=12)
plt.ylabel('이미지', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('similarity_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# 가장 유사한 쌍 찾기 (대각선 제외)
similarity_copy = similarity_matrix.copy()
np.fill_diagonal(similarity_copy, -1)  # 자기 자신 제외

max_sim_idx = np.unravel_index(similarity_copy.argmax(), similarity_copy.shape)
max_similarity = similarity_copy[max_sim_idx]

print(f"\n🏆 가장 유사한 이미지 쌍:")
print(f"이미지 1: {os.path.basename(sample_paths[max_sim_idx[0]])}")
print(f"이미지 2: {os.path.basename(sample_paths[max_sim_idx[1]])}")
print(f"유사도: {max_similarity:.4f}")

# 두 이미지 나란히 표시
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, ax in zip(max_sim_idx, axes):
    img = Image.open(sample_paths[idx])
    ax.imshow(img)
    ax.set_title(f"{os.path.basename(sample_paths[idx])}", fontsize=10)
    ax.axis('off')

plt.suptitle(f'가장 유사한 이미지 쌍 (유사도: {max_similarity:.4f})', fontsize=14)
plt.tight_layout()
plt.show()

print("✅ 실습 2 완료: similarity_heatmap.png 저장됨")
```

### 실습 3 솔루션

```python
from huggingface_hub import snapshot_download

print("=" * 80)
print("실습 3: 다국어 CLIP 모델 활용")
print("=" * 80)

# 1. 다국어 모델 다운로드
multilingual_model_dir = "./models/xlm_roberta_vit_b32"

if not os.path.exists(multilingual_model_dir):
    print("\n📥 다국어 CLIP 모델 다운로드 중... (약 10-20분 소요)")
    snapshot_download(
        repo_id="laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k",
        local_dir=multilingual_model_dir,
        local_dir_use_symlinks=False
    )
    print("✅ 다운로드 완료!")
else:
    print(f"✅ 이미 다운로드됨: {multilingual_model_dir}")

# 2. 체크포인트 파일 찾기
checkpoint_files = [
    f"{multilingual_model_dir}/open_clip_pytorch_model.bin",
    f"{multilingual_model_dir}/pytorch_model.bin",
]

multilingual_checkpoint = None
for ckpt in checkpoint_files:
    if os.path.exists(ckpt):
        multilingual_checkpoint = ckpt
        break

if multilingual_checkpoint is None:
    # 디렉토리 내 .bin 파일 검색
    for f in os.listdir(multilingual_model_dir):
        if f.endswith('.bin'):
            multilingual_checkpoint = os.path.join(multilingual_model_dir, f)
            break

print(f"✅ 체크포인트: {multilingual_checkpoint}")

# 3. 다국어 모델 로드
print("\n🔄 다국어 CLIP 모델 로드 중...")
multilingual_model, _, multilingual_preprocess = open_clip.create_model_and_transforms(
    'xlm-roberta-base-ViT-B-32',
    pretrained=multilingual_checkpoint,
    device=device
)
multilingual_model.eval()
print("✅ 다국어 모델 로드 완료!")

# 4. 다국어 이미지 임베딩 재계산
print("\n🔄 다국어 모델로 이미지 임베딩 계산 중...")
multilingual_img_embeddings = []

for i in tqdm(range(0, len(img_files), batch_size), desc='다국어 임베딩 계산'):
    batch_files = img_files[i:i+batch_size]

    batch_images = []
    for img_path in batch_files:
        try:
            img = Image.open(img_path).convert('RGB')
            batch_images.append(multilingual_preprocess(img))
        except:
            continue

    if len(batch_images) == 0:
        continue

    batch_tensor = torch.stack(batch_images).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = multilingual_model.encode_image(batch_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    batch_embeddings = image_features.cpu().numpy()
    multilingual_img_embeddings.append(batch_embeddings)

multilingual_img_embeddings = np.vstack(multilingual_img_embeddings)
print(f"✅ 다국어 이미지 임베딩 Shape: {multilingual_img_embeddings.shape}")

# 5. 다국어 FAISS 인덱스 생성
print("\n🔄 다국어 FAISS 인덱스 생성 중...")
multilingual_index = faiss.IndexFlatL2(multilingual_img_embeddings.shape[1])
multilingual_index.add(multilingual_img_embeddings)
faiss.write_index(multilingual_index, 'multilingual_img_embeddings.index')
print("✅ 다국어 FAISS 인덱스 생성 완료!")

# 6. 다국어 검색 함수
def search_multilingual_images(text_query, k=3):
    """다국어 모델을 사용한 이미지 검색"""
    text_tokens = open_clip.tokenize([text_query]).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        text_embedding = multilingual_model.encode_text(text_tokens)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

    text_embedding = text_embedding.cpu().numpy().astype(np.float32)
    distances, indices = multilingual_index.search(text_embedding, k)

    return distances[0], indices[0]

# 7. 다양한 언어로 검색
multilingual_queries = [
    ("영어", "a beautiful sunset over the ocean"),
    ("한국어", "바다 위의 아름다운 일몰"),
    ("일본어", "海に沈む美しい夕日"),
    ("스페인어", "una hermosa puesta de sol sobre el océano"),
    ("프랑스어", "un beau coucher de soleil sur l'océan"),
]

print("\n🌍 다국어 이미지 검색 수행")
print("=" * 80)

for language, text_query in multilingual_queries:
    print(f"\n{'=' * 80}")
    print(f"🌍 언어: {language}")
    print(f"📝 쿼리: {text_query}")
    print(f"{'=' * 80}")

    distances, indices = search_multilingual_images(text_query, k=3)

    for i, (idx, dist) in enumerate(zip(indices, distances), 1):
        print(f"\n유사 이미지 {i}: {os.path.basename(img_files[idx])}")
        print(f"거리: {dist:.4f}")
        show_image(img_files[idx], title=f"{language} - {i} (거리: {dist:.4f})")

print(f"\n{'=' * 80}")
print("✅ 실습 3 완료!")
print("=" * 80)

print("\n📊 다국어 모델 분석:")
print("1. XLM-RoBERTa 기반 모델은 100개 이상의 언어를 지원합니다.")
print("2. 같은 의미의 다국어 쿼리가 유사한 이미지를 검색하는지 확인하세요.")
print("3. 다국어 모델은 언어 간 의미 공간을 공유하여 cross-lingual 검색이 가능합니다.")
```

## 🚀 실무 활용 예시

### 예시 1: 전자상거래 이미지 검색

온라인 쇼핑몰에서 텍스트 또는 이미지로 상품을 검색하는 시스템:

```python
class ProductImageSearch:
    """전자상거래 이미지 검색 시스템"""

    def __init__(self, model_name, checkpoint, device='cpu'):
        self.device = device

        # CLIP 모델 로드
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=checkpoint,
            device=device
        )
        self.model.eval()

        # FAISS 인덱스
        self.index = None
        self.product_ids = []
        self.product_metadata = {}

    def index_products(self, product_data):
        """
        상품 이미지를 인덱싱

        Args:
            product_data: [{'id': '001', 'image_path': '...', 'name': '...', 'price': ...}, ...]
        """
        print(f"🔄 {len(product_data)}개 상품 인덱싱 중...")

        embeddings = []

        for product in tqdm(product_data):
            img = Image.open(product['image_path']).convert('RGB')
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)

            with torch.no_grad(), torch.cuda.amp.autocast():
                embedding = self.model.encode_image(img_tensor)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)

            embeddings.append(embedding.cpu().numpy())
            self.product_ids.append(product['id'])
            self.product_metadata[product['id']] = {
                'name': product['name'],
                'price': product['price'],
                'image_path': product['image_path']
            }

        embeddings = np.vstack(embeddings)

        # FAISS 인덱스 생성
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

        print(f"✅ 인덱싱 완료: {len(self.product_ids)}개 상품")

    def search_by_text(self, query, k=10):
        """텍스트로 상품 검색"""
        text_tokens = open_clip.tokenize([query]).to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            text_embedding = self.model.encode_text(text_tokens)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

        text_embedding = text_embedding.cpu().numpy().astype(np.float32)
        distances, indices = self.index.search(text_embedding, k)

        # 결과 반환
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            product_id = self.product_ids[idx]
            metadata = self.product_metadata[product_id]
            results.append({
                'product_id': product_id,
                'distance': float(dist),
                'similarity': 1 / (1 + dist),  # 유사도로 변환
                **metadata
            })

        return results

    def search_by_image(self, image_path, k=10):
        """이미지로 유사 상품 검색"""
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            img_embedding = self.model.encode_image(img_tensor)
            img_embedding = img_embedding / img_embedding.norm(dim=-1, keepdim=True)

        img_embedding = img_embedding.cpu().numpy().astype(np.float32)
        distances, indices = self.index.search(img_embedding, k)

        # 결과 반환
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            product_id = self.product_ids[idx]
            metadata = self.product_metadata[product_id]
            results.append({
                'product_id': product_id,
                'distance': float(dist),
                'similarity': 1 / (1 + dist),
                **metadata
            })

        return results

# 사용 예시
search_system = ProductImageSearch(
    model_name='ViT-B-16-quickgelu',
    checkpoint='./models/metaclip_400m/open_clip_pytorch_model.bin',
    device=device
)

# 상품 데이터 (예시)
product_data = [
    {'id': '001', 'image_path': img_files[0], 'name': '상품 A', 'price': 10000},
    {'id': '002', 'image_path': img_files[1], 'name': '상품 B', 'price': 20000},
    # ... 더 많은 상품
]

# 인덱싱
search_system.index_products(product_data)

# 텍스트 검색
print("\n📝 텍스트 검색: '아름다운 풍경'")
results = search_system.search_by_text("beautiful landscape", k=5)
for i, result in enumerate(results, 1):
    print(f"{i}. {result['name']} (유사도: {result['similarity']:.3f}, 가격: {result['price']}원)")

# 이미지 검색
print("\n🖼️ 이미지 검색: 유사한 상품 찾기")
results = search_system.search_by_image(img_files[0], k=5)
for i, result in enumerate(results, 1):
    print(f"{i}. {result['name']} (유사도: {result['similarity']:.3f}, 가격: {result['price']}원)")
```

### 예시 2: 콘텐츠 자동 태깅 시스템

이미지에 자동으로 태그를 붙이는 시스템:

```python
class AutoTagging:
    """이미지 자동 태깅 시스템"""

    def __init__(self, model, preprocess, device):
        self.model = model
        self.preprocess = preprocess
        self.device = device

        # 사전 정의된 태그 목록
        self.tag_categories = {
            '장소': ['beach', 'mountain', 'city', 'park', 'forest', 'desert'],
            '날씨': ['sunny', 'cloudy', 'rainy', 'snowy', 'foggy'],
            '시간': ['morning', 'afternoon', 'sunset', 'night'],
            '대상': ['person', 'animal', 'building', 'vehicle', 'food'],
            '분위기': ['peaceful', 'energetic', 'romantic', 'mysterious'],
        }

    def tag_image(self, image_path, threshold=0.2):
        """
        이미지에 자동으로 태그 부여

        Args:
            image_path: 이미지 경로
            threshold: 태그 선택 임계값

        Returns:
            tags: 카테고리별 태그 딕셔너리
        """
        # 이미지 임베딩
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            img_embedding = self.model.encode_image(img_tensor)
            img_embedding = img_embedding / img_embedding.norm(dim=-1, keepdim=True)

        # 카테고리별 태그 점수 계산
        tags = {}

        for category, tag_list in self.tag_categories.items():
            # 태그 임베딩
            tag_tokens = open_clip.tokenize(tag_list).to(self.device)

            with torch.no_grad(), torch.cuda.amp.autocast():
                tag_embeddings = self.model.encode_text(tag_tokens)
                tag_embeddings = tag_embeddings / tag_embeddings.norm(dim=-1, keepdim=True)

            # 유사도 계산
            similarities = (img_embedding @ tag_embeddings.T).squeeze(0).cpu().numpy()

            # 임계값 이상인 태그만 선택
            selected_tags = [
                (tag, float(sim))
                for tag, sim in zip(tag_list, similarities)
                if sim >= threshold
            ]

            # 점수 순으로 정렬
            selected_tags.sort(key=lambda x: x[1], reverse=True)

            if selected_tags:
                tags[category] = selected_tags

        return tags

    def print_tags(self, tags):
        """태그 출력"""
        print("\n🏷️ 자동 생성 태그:")
        for category, tag_list in tags.items():
            print(f"\n{category}:")
            for tag, score in tag_list:
                print(f"  - {tag}: {score:.3f}")

# 사용 예시
auto_tagger = AutoTagging(model, preprocess, device)

# 샘플 이미지 태깅
sample_image = img_files[0]
print(f"📸 이미지: {os.path.basename(sample_image)}")
show_image(sample_image)

tags = auto_tagger.tag_image(sample_image, threshold=0.15)
auto_tagger.print_tags(tags)
```

## 📖 참고 자료

### 공식 문서
- [OpenAI CLIP 논문](https://arxiv.org/abs/2103.00020)
- [OpenCLIP GitHub](https://github.com/mlfoundations/open_clip)
- [FAISS 공식 문서](https://faiss.ai/)
- [LangChain Experimental](https://python.langchain.com/docs/integrations/text_embedding/open_clip/)

### 추가 학습 자료
- [CLIP: Connecting Text and Images](https://openai.com/research/clip)
- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [Contrastive Learning 이해하기](https://lilianweng.github.io/posts/2021-05-31-contrastive/)

### 관련 기술
- **BLIP**: 부트스트래핑 기반 이미지-텍스트 모델
- **DALL-E**: 텍스트에서 이미지 생성
- **Stable Diffusion**: 오픈소스 이미지 생성 모델
- **LLaVA**: 대규모 언어-비전 모델

---

이 가이드를 완료하셨다면 멀티모달 AI와 CLIP 모델을 활용한 이미지-텍스트 검색 시스템을 구축할 수 있습니다. 다음 단계로 멀티모달 LLM(GPT-4V, LLaVA 등)을 활용한 고급 비전-언어 작업을 학습하시길 권장합니다! 🎉
