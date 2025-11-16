# PRJ04_W1_005 - LLM 파인튜닝 실습 Part2: 임베딩 모델 파인튜닝 및 실전 활용

## 📚 학습 목표

이 가이드를 완료하면 다음을 할 수 있습니다:

1. **임베딩 개념 이해**: 텍스트를 벡터로 변환하는 원리와 활용 방법 파악
2. **Sentence Transformers 활용**: 임베딩 모델 파인튜닝 라이브러리 사용
3. **Matryoshka Loss 적용**: 다양한 차원의 임베딩을 하나의 모델로 생성
4. **대조 학습 이해**: Positive/Negative pairs를 활용한 학습 방법
5. **임베딩 평가**: 코사인 유사도를 통한 성능 측정
6. **실전 적용**: RAG 시스템, 검색 엔진 등에 임베딩 모델 활용

---

## 🎯 핵심 개념

### 1. 임베딩(Embedding)이란?

**임베딩**은 텍스트를 고정 길이의 숫자 벡터로 변환하는 기술입니다.

**원리:**
```
텍스트: "강아지는 귀엽다"
↓
임베딩 모델
↓
벡터: [0.23, -0.45, 0.67, ..., 0.12]  # 예: 1024차원
```

**특징:**
- 의미적으로 유사한 텍스트 → 가까운 벡터
- 예: "강아지"와 "개" → 유사한 벡터
- 고정 길이: 입력 길이와 무관하게 항상 같은 차원

### 2. 임베딩의 활용

**1. 유사도 검색 (Semantic Search):**
```
쿼리: "저렴한 주식 ETF"
→ 임베딩 생성
→ 데이터베이스의 모든 ETF 임베딩과 비교
→ 가장 유사한 ETF 반환
```

**2. 클러스터링:**
- 비슷한 ETF를 자동으로 그룹화
- 예: 국내주식 ETF, 해외주식 ETF, 채권 ETF

**3. 분류:**
- 텍스트를 카테고리로 분류
- 예: 고위험/중위험/저위험 ETF 분류

**4. RAG (Retrieval-Augmented Generation):**
```
사용자 질문 → 관련 문서 검색 (임베딩 사용) → LLM에 제공 → 답변 생성
```

### 3. Sentence-BERT (SBERT)

**BERT의 문제:**
- 두 문장 유사도 계산 시 매번 함께 입력 필요
- 느린 속도 (n개 문장 비교 → n² 연산)

**SBERT의 해결:**
- 각 문장을 독립적으로 임베딩
- 사전 계산 가능
- 빠른 속도 (BERT 대비 100배)

**Siamese 네트워크:**
```
문장1 → BERT → Pooling → 임베딩1 ─┐
                                  ├─→ 유사도 계산
문장2 → BERT → Pooling → 임베딩2 ─┘
```

### 4. 대조 학습 (Contrastive Learning)

**Multiple Negatives Ranking Loss:**

**원리:**
- Positive Pair: 의미적으로 유사한 문장 쌍
- Negative Samples: 배치 내 다른 문장들
- 목표: Positive는 가깝게, Negative는 멀게

**예시:**
```
Anchor: "TIGER 200은 KOSPI 200을 추종합니다"
Positive: "이 ETF는 한국 주요 200개 기업에 투자합니다"
Negative1: "레버리지 ETF는 고위험 상품입니다"
Negative2: "오늘 날씨가 좋네요"
```

**배치 크기의 중요성:**
- 배치 크기가 클수록 더 많은 negative samples
- 더 효과적인 학습
- 권장: 32-128

### 5. Matryoshka Embeddings

**마트료시카 인형 원리:**

러시아 전통 인형처럼 하나 안에 여러 개가 들어있는 구조입니다.

**개념:**
- 하나의 모델이 여러 차원의 임베딩 생성
- 예: 1024차원, 768차원, 512차원, 256차원...
- 사용 시 필요한 차원만 선택

**장점:**
```
전체 1024차원: 높은 정확도, 느린 속도, 큰 메모리
↓ (앞 512차원만 사용)
512차원: 중간 정확도, 중간 속도, 중간 메모리
↓ (앞 256차원만 사용)
256차원: 낮은 정확도, 빠른 속도, 작은 메모리
```

**사용 예시:**
- 빠른 필터링: 256차원으로 후보군 추출
- 정밀 검색: 1024차원으로 최종 순위 결정

### 6. 평가 지표

**코사인 유사도 (Cosine Similarity):**
```
cos_sim(A, B) = (A · B) / (||A|| × ||B||)
```
- 범위: -1 ~ 1
- 1: 완전히 동일한 방향 (매우 유사)
- 0: 직교 (관련 없음)
- -1: 완전히 반대 방향

**Spearman 상관계수:**
- 순위 기반 상관관계 측정
- 인간의 유사도 판단과 모델 예측 비교
- 범위: -1 ~ 1 (1에 가까울수록 좋음)

---

## 💻 단계별 구현 가이드

### 단계 1: 임베딩 모델 로드

#### 1.1 Sentence Transformers 설치

```bash
# Sentence Transformers 설치
pip install -U "sentence-transformers>=3.0"
```

#### 1.2 BGE-M3 모델 로드

**BGE-M3**는 다국어 지원 고성능 임베딩 모델입니다.

**특징:**
- **B**AIRU **G**eneral **E**mbedding
- **M**ulti-lingual: 다국어 지원 (한국어 포함)
- **M**ulti-granularity: 다양한 길이 (단어~문서)
- **M**ulti-functionality: 다양한 태스크

```python
from sentence_transformers import SentenceTransformer

# 모델 로드
emb_model = SentenceTransformer("BAAI/bge-m3")

print(f"임베딩 차원: {emb_model.get_sentence_embedding_dimension()}")
print(f"최대 시퀀스 길이: {emb_model.max_seq_length}")
```

**출력:**
```
임베딩 차원: 1024
최대 시퀀스 길이: 8192
```

---

### 단계 2: 데이터셋 로드

#### 2.1 임베딩 데이터셋 형식

```python
{
    "sentence1": "TIGER 200은 KOSPI 200을 추종합니다",
    "sentence2": "이 ETF는 한국 주요 200개 기업에 투자합니다",
    "label": 1  # 1=유사, 0=유사하지 않음
}
```

#### 2.2 Hugging Face에서 로드

```python
from datasets import load_dataset

# 임베딩 데이터셋 로드
username = 'your_username'  # 실제 사용자명으로 변경
embedding_dataset = load_dataset(f"{username}/etf-embedding-v1")

print(f"✅ 임베딩 데이터셋 로드 완료!")
print(f"   Train: {len(embedding_dataset['train'])}개")
print(f"   Test: {len(embedding_dataset['test'])}개")

# 샘플 확인
print("\n임베딩 샘플:")
print(embedding_dataset['train'][0])
```

**출력 예시:**
```
✅ 임베딩 데이터셋 로드 완료!
   Train: 39442개
   Test: 3035개

임베딩 샘플:
{
    'sentence1': 'KB RISE 국채선물10년인버스증권상장지수투자신탁(채권-파생형)',
    'sentence2': '10년국채선물지수를 추종하는 케이비자산운용 운용 ETF',
    'label': 1
}
```

---

### 단계 3: 손실 함수 및 평가자 설정

#### 3.1 Multiple Negatives Ranking Loss

```python
from sentence_transformers.losses import MultipleNegativesRankingLoss, MatryoshkaLoss
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

# 1. 기본 손실: Multiple Negatives Ranking Loss
base_loss = MultipleNegativesRankingLoss(emb_model)

# 2. Matryoshka Loss로 래핑
train_loss = MatryoshkaLoss(
    emb_model,
    base_loss,
    matryoshka_dims=[1024, 768, 512, 256, 128, 64]  # 지원할 차원들
)

print("✅ 손실 함수 설정 완료")
print(f"   지원 차원: {[1024, 768, 512, 256, 128, 64]}")
```

**Matryoshka Loss의 작동:**
```
전체 1024차원 임베딩 생성
↓
여러 차원에 대해 손실 계산:
- 1024차원으로 손실 계산
- 앞 768차원으로 손실 계산
- 앞 512차원으로 손실 계산
...
↓
모든 손실의 평균으로 학습
```

#### 3.2 평가자 설정

```python
# 평가자 설정
evaluator = EmbeddingSimilarityEvaluator(
    sentences1=embedding_dataset["test"]["sentence1"],
    sentences2=embedding_dataset["test"]["sentence2"],
    scores=embedding_dataset["test"]["label"],
    name="etf-eval"
)

# 학습 전 성능 평가
print("학습 전 모델 평가:")
score_before = evaluator(emb_model)
print(score_before)
```

**평가 지표 해석:**
```python
{
    'etf-eval_pearson_cosine': -0.65,      # 피어슨 상관계수
    'etf-eval_spearman_cosine': -0.52     # 스피어만 상관계수
}
```

음수 값은 학습 전 모델이 ETF 도메인에 최적화되지 않았음을 의미합니다.

---

### 단계 4: 임베딩 모델 학습

#### 4.1 Trainer 설정

```python
from sentence_transformers import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
import time

# 임베딩 학습
emb_trainer = SentenceTransformerTrainer(
    model=emb_model,
    train_dataset=embedding_dataset['train'],
    eval_dataset=embedding_dataset['test'],
    loss=train_loss,
    evaluator=evaluator,
    args=SentenceTransformerTrainingArguments(
        output_dir=f"{MODEL_DIR}/etf_embedding",

        # 학습 설정
        num_train_epochs=5,  # 총 5 에포크
        per_device_train_batch_size=32,  # 큰 배치 (대조 학습에 유리)
        per_device_eval_batch_size=32,

        # 최적화
        warmup_ratio=0.1,        # 전체의 10%를 워밍업
        learning_rate=2e-5,      # 임베딩 모델은 낮은 학습률
        fp16=True,               # Mixed Precision

        # 평가 및 저장
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,            # 최근 3개 체크포인트만 유지
        load_best_model_at_end=True,   # 최고 성능 모델 로드

        # 로깅
        logging_steps=10,
        report_to="none",
    )
)

print("🚀 임베딩 학습 시작...")
start_time = time.time()

emb_trainer.train()

elapsed = time.time() - start_time
print(f"✅ 임베딩 학습 완료! (소요 시간: {elapsed/60:.2f}분)")
```

**학습 시간 예상:**
- RTX 4090: 약 15-20분
- A6000: 약 20-25분
- 배치 크기 32로 빠른 학습

#### 4.2 학습 후 평가

```python
# 학습 후 성능 평가
print("\n학습 후 모델 평가:")
score_after = evaluator(emb_model)
print(score_after)

# 개선도 계산
print("\n성능 개선:")
print(f"  Spearman 개선: {score_before['etf-eval_spearman_cosine']:.3f} → {score_after['etf-eval_spearman_cosine']:.3f}")
improvement = score_after['etf-eval_spearman_cosine'] - score_before['etf-eval_spearman_cosine']
print(f"  개선도: {improvement:+.3f} ({improvement/abs(score_before['etf-eval_spearman_cosine'])*100:+.1f}%)")
```

---

### 단계 5: 모델 저장 및 배포

#### 5.1 로컬 저장

```python
# 임베딩 모델 저장
emb_save_dir = f"{MODEL_DIR}/etf_embedding_final"
emb_model.save(emb_save_dir)
print(f"✅ 로컬 저장: {emb_save_dir}")
```

#### 5.2 Hugging Face Hub 업로드

```python
from huggingface_hub import HfApi

# 업로드 리포지토리
emb_repo = f"{username}/etf-embedding-bge-m3"

try:
    # 방법 1: 간단한 업로드
    emb_model.push_to_hub(
        emb_repo,
        private=True
    )
    print(f"✅ HF Hub 업로드: https://huggingface.co/{emb_repo}")

except Exception as e:
    print(f"⚠️ 오류: {e}")
    print("대안 방법 사용 중...")

    # 방법 2: HfApi 사용
    api = HfApi()

    # 레포지토리 생성
    api.create_repo(
        repo_id=emb_repo,
        token=os.environ["HUGGINGFACE_TOKEN"],
        private=True,
        exist_ok=True,
        repo_type="model"
    )

    # 파일 업로드
    api.upload_folder(
        folder_path=emb_save_dir,
        repo_id=emb_repo,
        repo_type="model",
        token=os.environ["HUGGINGFACE_TOKEN"]
    )
    print(f"✅ HF Hub 업로드 (HfApi): https://huggingface.co/{emb_repo}")
```

---

### 단계 6: 임베딩 모델 테스트

#### 6.1 코사인 유사도 계산

```python
from sentence_transformers.util import cos_sim

# 테스트 문장들
test_sentences = [
    "TIGER 200은 KOSPI 200을 추종하는 ETF입니다.",
    "이 ETF는 한국 주요 200개 기업에 투자합니다.",
    "레버리지 ETF는 고위험 고수익 상품입니다.",
    "오늘 날씨가 참 좋네요."
]

# 임베딩 생성
embeddings = emb_model.encode(test_sentences, convert_to_tensor=True)

print(f"임베딩 shape: {embeddings.shape}")  # (4, 1024)

# 유사도 행렬 계산
similarity_matrix = cos_sim(embeddings, embeddings)

print("\n📊 문장 간 유사도 행렬:\n")
print("\t" + "\t".join([f"문장{i+1}" for i in range(len(test_sentences))]))
for i, row in enumerate(similarity_matrix):
    print(f"문장{i+1}\t" + "\t".join([f"{val:.3f}" for val in row]))
```

**출력 예시 (학습 후):**
```
📊 문장 간 유사도 행렬:

        문장1   문장2   문장3   문장4
문장1   1.000   0.856   0.412   0.123
문장2   0.856   1.000   0.398   0.145
문장3   0.412   0.398   1.000   0.089
문장4   0.123   0.145   0.089   1.000
```

**해석:**
- 문장1 ↔ 문장2: 0.856 (매우 유사, 둘 다 KOSPI 200 관련)
- 문장1 ↔ 문장3: 0.412 (중간, 둘 다 ETF 관련)
- 문장1 ↔ 문장4: 0.123 (낮음, 관련 없음)

#### 6.2 다양한 차원 테스트

**Matryoshka 임베딩 활용:**

```python
def test_different_dimensions(sentences, dims=[1024, 512, 256, 128, 64]):
    """다양한 차원의 임베딩 성능 비교"""

    print("\n📐 차원별 유사도 비교:\n")

    # 전체 임베딩 생성
    full_embeddings = emb_model.encode(sentences, convert_to_tensor=True)

    for dim in dims:
        # 앞 dim개만 사용
        truncated_embeddings = full_embeddings[:, :dim]

        # 유사도 계산
        sim = cos_sim(truncated_embeddings[0:1], truncated_embeddings[1:2])[0][0]

        print(f"{dim:4d}차원: 문장1 ↔ 문장2 유사도 = {sim:.4f}")

# 테스트 실행
test_different_dimensions(test_sentences[:2])
```

**출력 예시:**
```
📐 차원별 유사도 비교:

1024차원: 문장1 ↔ 문장2 유사도 = 0.8562
 512차원: 문장1 ↔ 문장2 유사도 = 0.8471
 256차원: 문장1 ↔ 문장2 유사도 = 0.8298
 128차원: 문장1 ↔ 문장2 유사도 = 0.8012
  64차원: 문장1 ↔ 문장2 유사도 = 0.7645
```

**관찰:**
- 차원이 높을수록 더 정확
- 128차원도 합리적인 성능
- 속도와 정확도의 트레이드오프

---

## 🎯 실습 문제

### 문제 1: 학습 전후 성능 비교 시각화

학습 전과 후의 Spearman 상관계수를 막대 그래프로 시각화하세요.

**요구사항:**
- matplotlib 사용
- 학습 전/후 비교
- 개선도 표시

<details>
<summary>솔루션 보기</summary>

```python
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
# plt.rcParams['font.family'] = 'AppleGothic'  # macOS
plt.rcParams['axes.unicode_minus'] = False

def visualize_training_improvement(score_before, score_after):
    """학습 전후 성능 비교 시각화"""

    # 데이터 추출
    before = abs(score_before['etf-eval_spearman_cosine'])
    after = abs(score_after['etf-eval_spearman_cosine'])
    improvement = after - before

    # 그래프 생성
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 1. 학습 전후 비교
    categories = ['학습 전', '학습 후']
    values = [before, after]
    colors = ['#FF6B6B', '#4ECDC4']

    bars = ax1.bar(categories, values, color=colors, alpha=0.8)
    ax1.set_ylabel('Spearman 상관계수 (절댓값)', fontsize=12)
    ax1.set_title('학습 전후 성능 비교', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3)

    # 값 표시
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height,
                f'{value:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 2. 개선도
    ax2.bar(['개선도'], [improvement], color='#95E1D3', alpha=0.8)
    ax2.set_ylabel('개선 정도', fontsize=12)
    ax2.set_title('성능 개선도', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # 값 표시
    ax2.text(0, improvement,
            f'+{improvement:.3f}\n({improvement/before*100:+.1f}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.show()

    print("\n📊 성능 요약:")
    print(f"  학습 전: {before:.3f}")
    print(f"  학습 후: {after:.3f}")
    print(f"  개선도: +{improvement:.3f} ({improvement/before*100:+.1f}%)")

# 실행
visualize_training_improvement(score_before, score_after)
```

</details>

---

### 문제 2: ETF 유사도 검색 시스템

사용자가 쿼리를 입력하면 가장 유사한 ETF 3개를 반환하는 함수를 작성하세요.

**요구사항:**
- 임베딩 모델 사용
- 코사인 유사도 기반 순위
- Top-3 결과 반환

<details>
<summary>솔루션 보기</summary>

```python
import numpy as np
from sentence_transformers.util import cos_sim

def create_etf_search_system(etf_descriptions):
    """
    ETF 검색 시스템 생성

    Args:
        etf_descriptions: List[str], ETF 설명 리스트

    Returns:
        search_func: 검색 함수
    """
    # ETF 설명 임베딩 (사전 계산)
    etf_embeddings = emb_model.encode(etf_descriptions, convert_to_tensor=True)

    def search(query, top_k=3):
        """
        쿼리와 유사한 ETF 검색

        Args:
            query: str, 검색 쿼리
            top_k: int, 반환할 결과 수

        Returns:
            List[Tuple[int, float, str]]: (인덱스, 유사도, 설명)
        """
        # 쿼리 임베딩
        query_embedding = emb_model.encode(query, convert_to_tensor=True)

        # 유사도 계산
        similarities = cos_sim(query_embedding, etf_embeddings)[0]

        # Top-k 추출
        top_indices = similarities.argsort(descending=True)[:top_k]

        results = []
        for idx in top_indices:
            results.append((
                idx.item(),
                similarities[idx].item(),
                etf_descriptions[idx.item()]
            ))

        return results

    return search

# ETF 데이터 준비
etf_descriptions = [
    "TIGER 200은 KOSPI 200 지수를 추종하는 국내 대표 ETF입니다.",
    "KODEX 레버리지는 KOSPI 200의 2배 수익률을 추구하는 고위험 상품입니다.",
    "ACE 미국S&P500은 미국 대형주 500개 기업에 투자하는 해외 ETF입니다.",
    "TIGER 미국나스닥100은 미국 기술주에 집중 투자하는 ETF입니다.",
    "KODEX 200선물인버스2X는 KOSPI 200 하락 시 수익을 내는 인버스 상품입니다.",
    "TIGER 국채3년은 안정적인 국채에 투자하는 채권 ETF입니다.",
]

# 검색 시스템 생성
search_etf = create_etf_search_system(etf_descriptions)

# 테스트
test_queries = [
    "안정적인 투자 상품을 찾습니다",
    "미국 기술주에 투자하고 싶어요",
    "국내 주식 시장 전체에 투자하려면?"
]

print("🔍 ETF 검색 시스템 테스트\n" + "="*60)
for query in test_queries:
    print(f"\n📝 쿼리: {query}")
    results = search_etf(query, top_k=3)

    for rank, (idx, score, desc) in enumerate(results, 1):
        print(f"{rank}. [{score:.3f}] {desc}")
    print("-"*60)
```

**출력 예시:**
```
🔍 ETF 검색 시스템 테스트
============================================================

📝 쿼리: 안정적인 투자 상품을 찾습니다
1. [0.782] TIGER 국채3년은 안정적인 국채에 투자하는 채권 ETF입니다.
2. [0.543] TIGER 200은 KOSPI 200 지수를 추종하는 국내 대표 ETF입니다.
3. [0.412] ACE 미국S&P500은 미국 대형주 500개 기업에 투자하는 해외 ETF입니다.
------------------------------------------------------------
```

</details>

---

### 문제 3: Matryoshka 차원 최적화 분석

여러 차원에서의 검색 정확도와 속도를 비교 분석하세요.

**요구사항:**
- 차원: 1024, 512, 256, 128, 64
- 검색 시간 측정
- 정확도(Top-1 일치율) 계산

<details>
<summary>솔루션 보기</summary>

```python
import time
import numpy as np

def benchmark_dimensions(queries, etf_descriptions, dims=[1024, 512, 256, 128, 64]):
    """
    다양한 차원에서 검색 성능 벤치마크

    Args:
        queries: List[str], 검색 쿼리 리스트
        etf_descriptions: List[str], ETF 설명 리스트
        dims: List[int], 테스트할 차원 리스트
    """
    # 전체 임베딩 생성
    query_embs_full = emb_model.encode(queries, convert_to_tensor=True)
    etf_embs_full = emb_model.encode(etf_descriptions, convert_to_tensor=True)

    # 1024차원 기준 결과 (Ground Truth)
    gt_results = []
    for query_emb in query_embs_full:
        sims = cos_sim(query_emb, etf_embs_full)[0]
        top_idx = sims.argmax().item()
        gt_results.append(top_idx)

    print("🔬 Matryoshka 차원별 성능 분석\n" + "="*70)
    print(f"{'차원':>6} | {'검색시간(ms)':>12} | {'Top-1 일치율':>12} | {'속도 향상':>10}")
    print("-"*70)

    base_time = None
    for dim in dims:
        # 차원 축소
        query_embs = query_embs_full[:, :dim]
        etf_embs = etf_embs_full[:, :dim]

        # 검색 시간 측정
        start = time.time()
        results = []
        for query_emb in query_embs:
            sims = cos_sim(query_emb, etf_embs)[0]
            top_idx = sims.argmax().item()
            results.append(top_idx)
        elapsed = (time.time() - start) * 1000  # ms

        if base_time is None:
            base_time = elapsed

        # 정확도 계산
        accuracy = sum(1 for r, gt in zip(results, gt_results) if r == gt) / len(results)

        # 속도 향상
        speedup = base_time / elapsed

        print(f"{dim:>6} | {elapsed:>11.2f}ms | {accuracy*100:>11.1f}% | {speedup:>9.2f}x")

    print("="*70)
    print("\n💡 관찰:")
    print("  - 차원이 낮을수록 빠르지만 정확도 감소")
    print("  - 256-512차원이 속도와 정확도의 균형점")
    print("  - 실시간 검색: 128-256차원")
    print("  - 고정밀 검색: 512-1024차원")

# 테스트 쿼리
test_queries = [
    "안정적인 채권 투자",
    "미국 기술주",
    "국내 대형주",
    "고위험 레버리지",
    "해외 분산 투자"
]

# 벤치마크 실행
benchmark_dimensions(test_queries, etf_descriptions)
```

**출력 예시:**
```
🔬 Matryoshka 차원별 성능 분석
======================================================================
   차원 | 검색시간(ms) | Top-1 일치율 | 속도 향상
----------------------------------------------------------------------
  1024 |       12.34ms |       100.0% |      1.00x
   512 |        8.56ms |       100.0% |      1.44x
   256 |        5.23ms |        80.0% |      2.36x
   128 |        3.45ms |        60.0% |      3.58x
    64 |        2.12ms |        40.0% |      5.82x
======================================================================
```

</details>

---

### 문제 4: RAG 시스템 구축

임베딩 모델을 활용한 간단한 RAG (Retrieval-Augmented Generation) 시스템을 구축하세요.

**요구사항:**
- 질문에 대해 관련 문서 검색
- 검색된 문서를 컨텍스트로 LLM에 제공
- 최종 답변 생성

<details>
<summary>솔루션 보기</summary>

```python
def create_rag_system(documents, llm_model, llm_tokenizer):
    """
    RAG 시스템 생성

    Args:
        documents: List[str], 문서 컬렉션
        llm_model: LLM 모델
        llm_tokenizer: LLM 토크나이저

    Returns:
        rag_query: RAG 쿼리 함수
    """
    # 문서 임베딩 (사전 계산)
    doc_embeddings = emb_model.encode(documents, convert_to_tensor=True)

    def retrieve(query, top_k=3):
        """관련 문서 검색"""
        query_emb = emb_model.encode(query, convert_to_tensor=True)
        sims = cos_sim(query_emb, doc_embeddings)[0]
        top_indices = sims.argsort(descending=True)[:top_k]
        return [documents[idx] for idx in top_indices]

    def rag_query(question):
        """
        RAG 쿼리 실행

        Args:
            question: str, 사용자 질문

        Returns:
            str, 답변
        """
        # 1. 관련 문서 검색
        relevant_docs = retrieve(question, top_k=3)
        context = "\n\n".join([f"문서 {i+1}: {doc}" for i, doc in enumerate(relevant_docs)])

        # 2. 프롬프트 구성
        rag_prompt = f"""아래 문서들을 참고하여 질문에 답변하세요.

### 참고 문서:
{context}

### 질문:
{question}

### 답변:
"""

        # 3. LLM 생성
        inputs = llm_tokenizer(rag_prompt, return_tensors="pt").to("cuda")
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9
        )
        answer = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = answer.split("### 답변:")[-1].strip()

        return {
            "question": question,
            "retrieved_docs": relevant_docs,
            "answer": answer
        }

    return rag_query

# ETF 문서 컬렉션
etf_documents = [
    "TIGER 200은 삼성자산운용에서 운용하는 ETF로, KOSPI 200 지수를 추종합니다. 총보수는 연 0.15%이며, 국내 대표 200개 기업에 분산 투자합니다.",
    "KODEX 레버리지는 한국투자신탁운용의 ETF로, KOSPI 200의 일일 수익률 2배를 추구합니다. 고위험 고수익 상품으로 단기 투자에 적합합니다.",
    "ACE 미국S&P500은 한국투자신탁운용의 해외 ETF로, 미국 대형주 500개 기업에 투자합니다. 환헤지 미적용으로 환율 변동 영향을 받습니다.",
    "TIGER 국채3년은 미래에셋자산운용의 채권 ETF로, 한국 국채에 투자합니다. 안정적인 수익을 추구하며 금리 변동에 민감합니다.",
]

# RAG 시스템 생성 (LLM 모델 필요)
# rag_query = create_rag_system(etf_documents, model, tokenizer)

# 테스트 (예시)
print("🤖 RAG 시스템 구조:")
print("  1. 질문 입력")
print("  2. 임베딩 모델로 관련 문서 검색")
print("  3. 검색된 문서를 컨텍스트로 LLM에 제공")
print("  4. LLM이 답변 생성")
print("\n💡 실제 사용:")
print("  result = rag_query('안정적인 ETF 추천해주세요')")
print("  print(result['answer'])")
```

</details>

---

## 🚀 실무 활용 예시

### 예시 1: ETF 추천 시스템

**시나리오**: 사용자의 투자 성향에 맞는 ETF 추천

```python
def create_etf_recommender(etf_data):
    """
    ETF 추천 시스템

    Args:
        etf_data: DataFrame, ETF 정보

    Returns:
        recommender: 추천 함수
    """
    # ETF 설명 생성
    etf_descriptions = []
    for _, row in etf_data.iterrows():
        desc = f"{row['name_kr']}은 {row['base_asset_category']} 자산에 투자하는 ETF로, " \
               f"{row['manager']}에서 운용하며 총보수는 연 {row['total_expense_ratio']:.2f}%입니다. " \
               f"기초지수는 {row['base_index']}이며, {row['replication_method']} 방식으로 운용됩니다."
        etf_descriptions.append(desc)

    # 임베딩 생성
    etf_embeddings = emb_model.encode(etf_descriptions, convert_to_tensor=True)

    def recommend(user_profile, top_k=5):
        """
        사용자 프로필 기반 ETF 추천

        Args:
            user_profile: str, 사용자 투자 성향
            top_k: int, 추천 개수

        Returns:
            List[Dict], 추천 ETF 리스트
        """
        # 사용자 프로필 임베딩
        profile_emb = emb_model.encode(user_profile, convert_to_tensor=True)

        # 유사도 계산
        similarities = cos_sim(profile_emb, etf_embeddings)[0]

        # Top-k 추출
        top_indices = similarities.argsort(descending=True)[:top_k]

        recommendations = []
        for idx in top_indices:
            recommendations.append({
                'rank': len(recommendations) + 1,
                'name': etf_data.iloc[idx]['name_kr'],
                'ticker': etf_data.iloc[idx]['ticker'],
                'similarity': similarities[idx].item(),
                'description': etf_descriptions[idx],
                'manager': etf_data.iloc[idx]['manager'],
                'expense_ratio': etf_data.iloc[idx]['total_expense_ratio']
            })

        return recommendations

    return recommend

# 사용 예시
# etf_recommender = create_etf_recommender(df)
#
# user_profiles = [
#     "20대 직장인으로 장기 투자를 원하며, 안정적이면서도 성장 가능성이 있는 상품을 찾습니다.",
#     "은퇴를 앞둔 50대로 원금 보존이 중요하며, 안정적인 배당 수익을 원합니다.",
#     "투자 경험이 풍부하며 높은 수익률을 위해 적극적인 위험을 감수할 수 있습니다."
# ]
#
# for profile in user_profiles:
#     print(f"\n👤 프로필: {profile}")
#     results = etf_recommender(profile, top_k=3)
#     for rec in results:
#         print(f"{rec['rank']}. {rec['name']} ({rec['ticker']})")
#         print(f"   유사도: {rec['similarity']:.3f} | 보수: {rec['expense_ratio']:.2f}%")
```

---

### 예시 2: 실시간 ETF 뉴스 분류 시스템

**시나리오**: ETF 관련 뉴스를 자동으로 분류하고 관련 ETF 매칭

```python
def create_news_classifier(etf_data):
    """
    ETF 뉴스 분류 및 매칭 시스템
    """
    # ETF별 키워드 임베딩
    etf_keywords = []
    for _, row in etf_data.iterrows():
        keywords = f"{row['name_kr']} {row['short_name_kr']} {row['base_index']} " \
                   f"{row['base_asset_category']} {row['base_market_category']}"
        etf_keywords.append(keywords)

    etf_embs = emb_model.encode(etf_keywords, convert_to_tensor=True)

    def classify_news(news_title, news_content, threshold=0.5):
        """
        뉴스를 분석하여 관련 ETF 찾기

        Args:
            news_title: str, 뉴스 제목
            news_content: str, 뉴스 본문
            threshold: float, 관련성 임계값

        Returns:
            List[Dict], 관련 ETF 리스트
        """
        # 뉴스 전체 텍스트 임베딩
        news_text = f"{news_title} {news_content}"
        news_emb = emb_model.encode(news_text, convert_to_tensor=True)

        # 유사도 계산
        similarities = cos_sim(news_emb, etf_embs)[0]

        # 임계값 이상인 ETF 추출
        relevant_indices = (similarities > threshold).nonzero(as_tuple=True)[0]

        results = []
        for idx in relevant_indices:
            results.append({
                'etf_name': etf_data.iloc[idx]['name_kr'],
                'ticker': etf_data.iloc[idx]['ticker'],
                'relevance': similarities[idx].item(),
                'category': etf_data.iloc[idx]['base_asset_category']
            })

        # 관련성 순으로 정렬
        results.sort(key=lambda x: x['relevance'], reverse=True)

        return results

    return classify_news

# 사용 예시
# news_classifier = create_news_classifier(df)
#
# sample_news = [
#     {
#         "title": "코스피 200 지수 사상 최고치 경신",
#         "content": "국내 주요 200개 기업으로 구성된 코스피 200 지수가 사상 최고치를 경신했습니다..."
#     },
#     {
#         "title": "미국 기술주 급등, 나스닥 5% 상승",
#         "content": "애플, 마이크로소프트 등 미국 기술주가 강세를 보이며 나스닥 지수가 급등했습니다..."
#     }
# ]
#
# for news in sample_news:
#     print(f"\n📰 뉴스: {news['title']}")
#     related = news_classifier(news['title'], news['content'], threshold=0.6)
#     print(f"   관련 ETF {len(related)}개:")
#     for etf in related[:5]:
#         print(f"   - {etf['etf_name']} ({etf['ticker']}): {etf['relevance']:.3f}")
```

---

### 예시 3: ETF 포트폴리오 다양화 분석

**시나리오**: 포트폴리오 내 ETF들의 유사도를 분석하여 중복 투자 경고

```python
def analyze_portfolio_diversity(portfolio_tickers, etf_data):
    """
    포트폴리오 다양화 분석

    Args:
        portfolio_tickers: List[str], 보유 ETF 티커 리스트
        etf_data: DataFrame, ETF 정보

    Returns:
        Dict, 분석 결과
    """
    # 포트폴리오 ETF 필터링
    portfolio = etf_data[etf_data['ticker'].isin(portfolio_tickers)]

    # ETF 설명 생성
    descriptions = []
    for _, row in portfolio.iterrows():
        desc = f"{row['name_kr']}은 {row['base_index']}를 추종하며, " \
               f"{row['base_asset_category']} 자산에 {row['base_market_category']} 시장에서 투자합니다."
        descriptions.append(desc)

    # 임베딩 및 유사도 행렬 계산
    embeddings = emb_model.encode(descriptions, convert_to_tensor=True)
    similarity_matrix = cos_sim(embeddings, embeddings).cpu().numpy()

    # 대각선 제외 (자기 자신과의 유사도)
    np.fill_diagonal(similarity_matrix, 0)

    # 분석
    n = len(portfolio)
    avg_similarity = similarity_matrix.sum() / (n * (n - 1))  # 평균 유사도

    # 높은 유사도 쌍 찾기
    high_sim_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            if similarity_matrix[i, j] > 0.7:  # 유사도 70% 이상
                high_sim_pairs.append({
                    'etf1': portfolio.iloc[i]['name_kr'],
                    'etf2': portfolio.iloc[j]['name_kr'],
                    'similarity': similarity_matrix[i, j]
                })

    # 다양성 점수 (0~100)
    diversity_score = (1 - avg_similarity) * 100

    return {
        'portfolio_size': n,
        'average_similarity': avg_similarity,
        'diversity_score': diversity_score,
        'high_similarity_pairs': high_sim_pairs,
        'recommendation': '다양화 우수' if diversity_score > 60 else '중복 투자 주의'
    }

# 사용 예시
# sample_portfolio = ['069500', '233740', '114800']  # KODEX 200, KODEX 은행, TIGER 200
#
# analysis = analyze_portfolio_diversity(sample_portfolio, df)
#
# print("📊 포트폴리오 다양화 분석")
# print(f"  포트폴리오 크기: {analysis['portfolio_size']}개")
# print(f"  평균 유사도: {analysis['average_similarity']:.3f}")
# print(f"  다양성 점수: {analysis['diversity_score']:.1f}/100")
# print(f"  평가: {analysis['recommendation']}")
#
# if analysis['high_similarity_pairs']:
#     print("\n⚠️ 높은 유사도 ETF 쌍:")
#     for pair in analysis['high_similarity_pairs']:
#         print(f"  - {pair['etf1']} ↔ {pair['etf2']}: {pair['similarity']:.3f}")
#     print("\n💡 조언: 포트폴리오를 더 다양화하는 것을 고려하세요.")
```

---

## 📌 핵심 요약

### 1. 임베딩 모델 학습 플로우

```
모델 로드 → 데이터 준비 → 손실 함수 설정 → 학습 → 평가 → 배포
```

### 2. 주요 개념 정리

| 개념 | 설명 | 핵심 값 |
|------|------|--------|
| **임베딩 차원** | 벡터 크기 | 256-1024 |
| **배치 크기** | 대조 학습에 중요 | 32-128 |
| **학습률** | 임베딩 모델 학습률 | 2e-5 |
| **Matryoshka** | 다차원 지원 | [1024, 512, 256...] |

### 3. 성능 최적화

- ✅ 큰 배치: 더 많은 negative samples
- ✅ Matryoshka: 유연한 차원 선택
- ✅ Mixed Precision: 학습 속도 2배
- ✅ 사전 계산: 문서 임베딩 캐싱

### 4. 실전 팁

**속도 vs 정확도:**
- 빠른 필터링: 128-256차원
- 정밀 검색: 512-1024차원
- 균형: 256-512차원

**RAG 시스템:**
- Top-3~5 문서 검색
- 유사도 임계값: 0.5-0.7
- 컨텍스트 길이 제한

---

## 🔗 참고 자료

- **Sentence Transformers**: https://www.sbert.net/
- **BGE-M3 모델**: https://huggingface.co/BAAI/bge-m3
- **Matryoshka Embeddings 논문**: https://arxiv.org/abs/2205.13147
- **RAG 가이드**: https://python.langchain.com/docs/use_cases/question_answering/

---

이 가이드를 통해 **임베딩 모델 파인튜닝과 실전 활용**의 전체 과정을 익힐 수 있습니다!
