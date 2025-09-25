# PRJ02_W1_002 정보 검색 평가지표 매뉴얼

## 📋 개요

이 노트북은 정보 검색 시스템의 성능을 평가하는 핵심 지표들(Hit Rate, MRR, NDCG, mAP)을 심도 있게 다룹니다. K-RAG 패키지를 활용하여 실제 검색 성능을 체계적으로 측정하고 분석하는 방법을 학습할 수 있습니다.

### 🎯 학습 목표
- Hit Rate, MRR, NDCG, mAP 등 핵심 검색 평가 지표의 이해
- K-RAG 패키지를 활용한 실제 평가 수행
- Precision, Recall, F1 Score의 Micro/Macro Average 계산
- 검색 성능 평가의 실무적 적용

## 🛠️ 환경 설정

### 1. 필수 패키지 설치
```bash
# K-RAG 패키지 설치
uv pip install krag
```

### 2. 필수 라이브러리
```python
# 기본 라이브러리
import os
from glob import glob
from pprint import pprint
import json
import numpy as np
from typing import List, Tuple

# LangChain 관련
from langchain_core.documents import Document
from textwrap import dedent

# K-RAG 평가 라이브러리
from krag.evaluators import (
    OfflineRetrievalEvaluators,
    RougeOfflineRetrievalEvaluators,
)

# Langfuse 트레이싱
from langfuse.langchain import CallbackHandler
```

## 📊 평가 지표 개념

### 1. 검색 평가의 두 가지 접근법

#### Non-Rank Based Metrics
- **Accuracy, Precision, Recall@k**: 관련성의 이진적 평가
- 순서를 고려하지 않는 기본적인 평가 방식

#### Rank-Based Metrics
- **MRR, MAP, NDCG**: 검색 결과의 순위를 고려한 평가
- 실제 사용자 경험을 더 잘 반영

### 2. 생성 평가 방식

#### 전통적 평가
- **ROUGE**: 요약 품질 측정
- **BLEU**: 번역 품질 측정
- **BertScore**: 의미 유사도 측정

#### LLM 기반 평가
- 응집성, 관련성, 유창성을 종합적으로 판단
- 참조 답변이 없는 경우에도 평가 가능

## 🎯 평가 데이터 준비

### 1. 샘플 문서 데이터 구성

```python
# 실제 문서 (정답)
actual_docs = [
    [  # 첫 번째 쿼리의 정답 문서들
        Document(
            page_content="고객 문의: 제품 배송 지연\n...",
            metadata={"id": "doc1", "category": "배송", "priority": "높음"}
        ),
        Document(
            page_content="고객 문의: 결제 오류\n...",
            metadata={"id": "doc2", "category": "결제", "priority": "높음"}
        ),
        # ... 추가 문서들
    ],
    [  # 두 번째 쿼리의 정답 문서들
        Document(
            page_content="고객 문의: 제품 교환 요청\n...",
            metadata={"id": "doc3", "category": "교환/반품", "priority": "중간"}
        ),
        # ... 추가 문서들
    ]
]

# 예측 문서 (검색 결과)
predicted_docs = [
    [  # 첫 번째 쿼리의 검색 결과
        # 정확한 검색 결과들
    ],
    [  # 두 번째 쿼리의 검색 결과
        # 일부 오답 포함된 결과들
    ]
]
```

## 📈 성능 지표 계산

### 1. TP, FP, FN 개념

```python
# True Positive: 정확하게 검색된 관련 문서
true_positives = [
    [doc.metadata["id"] for doc in actual if doc in predicted]
    for actual, predicted in zip(actual_docs, predicted_docs)
]

# False Positive: 잘못 검색된 무관한 문서
false_positives = [
    [doc.metadata["id"] for doc in predicted if doc not in actual]
    for actual, predicted in zip(actual_docs, predicted_docs)
]

# False Negative: 놓친 관련 문서
false_negatives = [
    [doc.metadata["id"] for doc in actual if doc not in predicted]
    for actual, predicted in zip(actual_docs, predicted_docs)
]
```

### 2. Precision, Recall, F1 계산

```python
# 각 쿼리별 성능 계산
for i, (tp, fp, fn) in enumerate(zip(true_positives, false_positives, false_negatives)):
    precision = len(tp) / (len(tp) + len(fp)) if len(tp) + len(fp) > 0 else 0
    recall = len(tp) / (len(tp) + len(fn)) if len(tp) + len(fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    print(f"쿼리 {i+1}: Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}")
```

### 3. Macro vs Micro Average

#### Macro Average (각 클래스 동등 가중)
```python
def calculate_macro_metrics(true_positives, false_positives, false_negatives):
    n_classes = len(true_positives)
    precisions, recalls, f1_scores = [], [], []

    for i in range(n_classes):
        tp, fp, fn = len(true_positives[i]), len(false_positives[i]), len(false_negatives[i])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return np.mean(precisions), np.mean(recalls), np.mean(f1_scores)
```

#### Micro Average (전체 데이터 기준)
```python
def calculate_micro_metrics(true_positives, false_positives, false_negatives):
    # 전체 TP, FP, FN 합계
    total_tp = sum(len(tp) for tp in true_positives)
    total_fp = sum(len(fp) for fp in false_positives)
    total_fn = sum(len(fn) for fn in false_negatives)

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) \
        if (micro_precision + micro_recall) > 0 else 0

    return micro_precision, micro_recall, micro_f1
```

## 🔍 K-RAG 패키지 활용

### 1. RAG 평가자 클래스 구성

```python
class RAGEvaluator:
    def __init__(self, match_method="text", rouge_threshold=0.5):
        self.match_method = match_method  # "text", "rouge1", "rouge2", "rougeL"
        self.rouge_threshold = rouge_threshold
        self.evaluator = None

    def _initialize_evaluator(self, actual_docs, predicted_docs):
        if self.match_method in ["rouge1", "rouge2", "rougeL"]:
            self.evaluator = RougeOfflineRetrievalEvaluators(
                actual_docs, predicted_docs,
                match_method=self.match_method,
                threshold=self.rouge_threshold
            )
        else:
            self.evaluator = OfflineRetrievalEvaluators(
                actual_docs, predicted_docs,
                match_method=self.match_method,
            )
        return self.evaluator

    def evaluate_all(self, actual_docs, predicted_docs, k=10, visualize=False):
        if self.evaluator is None:
            self._initialize_evaluator(actual_docs, predicted_docs)

        results = {
            'hit_rate': self.evaluator.calculate_hit_rate(k=k),
            'mrr': self.evaluator.calculate_mrr(k=k),
            'recall': self.evaluator.calculate_recall(k=k),
            'precision': self.evaluator.calculate_precision(k=k),
            'f1_score': self.evaluator.calculate_f1_score(k=k),
            'map': self.evaluator.calculate_map(k=k),
            'ndcg': self.evaluator.calculate_ndcg(k=k)
        }

        return results
```

### 2. 평가 실행

```python
# 텍스트 일치 기반 평가
evaluator = RAGEvaluator(match_method="text")
results = evaluator.evaluate_all(actual_docs, predicted_docs, k=10)

# ROUGE 기반 평가 (의미적 유사성)
rouge_evaluator = RAGEvaluator(match_method="rouge2", rouge_threshold=0.8)
rouge_results = rouge_evaluator.evaluate_all(actual_docs, predicted_docs, k=10)
```

## 📊 핵심 평가 지표 상세

### 1. Hit Rate (적중률)

**개념**: 검색 결과에 정답 문서가 모두 포함되어 있는지를 이진법으로 평가

**계산 방식**:
```python
# k=3일 때 예시
# 쿼리 1: [doc1, doc2, doc5] 모두 정답 → 1
# 쿼리 2: doc3 누락, doc4만 찾음 → 0
hit_rate = (1 + 0) / 2 = 0.5
```

**특징**:
- 0~1 사이 값, 1에 가까울수록 우수
- 순서 고려하지 않음
- 기본적인 검색 성능 지표

### 2. MRR (Mean Reciprocal Rank)

**개념**: 첫 번째 관련 문서의 등장 순위 역수의 평균

**계산 방식**:
```python
# 쿼리 1: doc1이 1위 → 1/1 = 1.0
# 쿼리 2: doc4가 2위 → 1/2 = 0.5
MRR = (1.0 + 0.5) / 2 = 0.75
```

**특징**:
- 사용자 경험 관점에서 중요
- 첫 번째 정답의 위치만 고려
- 빠른 정보 접근성 측정

### 3. mAP@k (Mean Average Precision)

**개념**: 상위 k개 결과 내에서 관련 문서들의 정확도를 종합적으로 평가

**계산 방식**:
```mathematica
AP = (각 정답 문서 위치에서의 Precision의 합) / (전체 정답 문서 수)
mAP = 모든 쿼리의 AP 평균
```

**예시**:
```python
# 쿼리 1: [doc1, doc2, doc5] 모두 정답
# P(1)=1/1, P(2)=2/2, P(3)=3/3
# AP = (1 + 1 + 1) / 3 = 1.0

# 쿼리 2: [doc6, doc4, doc5], 정답은 [doc3, doc4]
# P(2)=1/2 (doc4만 정답)
# AP = (0.5) / 2 = 0.25

# mAP = (1.0 + 0.25) / 2 = 0.625
```

### 4. NDCG (Normalized Discounted Cumulative Gain)

**개념**: 문서의 관련성과 검색 순위를 동시에 고려한 정규화 점수

**계산 방식**:
```mathematica
DCG@k = Σ(i=1 to k) (2^rel_i - 1) / log₂(i+1)
NDCG@k = DCG@k / IDCG@k
```

**특징**:
- 상위 결과에 더 높은 가중치
- 이상적 순위와 비교하여 정규화
- 0~1 사이 값, 1이 완벽한 순위

## 💡 평가 지표 비교

| 지표 | 목적 | 순위 고려 | 계산 복잡도 | 활용 상황 |
|------|------|-----------|-------------|-----------|
| **Hit Rate** | 기본 포함 여부 | ❌ | 낮음 | 빠른 성능 체크 |
| **MRR** | 첫 정답 위치 | ✅ | 낮음 | QA 시스템 |
| **mAP** | 전체 정확도 | ✅ | 중간 | 일반 검색 |
| **NDCG** | 순위 품질 | ✅ | 높음 | 추천 시스템 |

## 🛠️ 실무 활용 팁

### 1. 지표 선택 가이드

```python
# 기본 성능 확인
if quick_check:
    use_metrics = ['hit_rate']

# QA 시스템
elif system_type == 'qa':
    use_metrics = ['mrr', 'hit_rate']

# 일반 검색 엔진
elif system_type == 'search':
    use_metrics = ['map', 'ndcg', 'mrr']

# 추천 시스템
elif system_type == 'recommendation':
    use_metrics = ['ndcg', 'map']
```

### 2. k 값 설정

```python
# 사용자가 일반적으로 확인하는 상위 결과 수에 맞춰 설정
k_values = [1, 3, 5, 10]

for k in k_values:
    results = evaluator.evaluate_all(actual_docs, predicted_docs, k=k)
    print(f"@{k}: {results}")
```

### 3. 임계값 최적화

```python
# ROUGE 임계값 최적화
thresholds = [0.3, 0.5, 0.7, 0.9]
best_threshold = 0.5
best_score = 0

for threshold in thresholds:
    evaluator = RAGEvaluator(match_method="rouge2", rouge_threshold=threshold)
    results = evaluator.evaluate_all(actual_docs, predicted_docs)

    if results['f1_score']['macro_f1'] > best_score:
        best_score = results['f1_score']['macro_f1']
        best_threshold = threshold
```

## 🔧 문제 해결

### 자주 발생하는 문제들

1. **빈 검색 결과**
   ```python
   # 검색 결과가 비어있을 때 처리
   if not retrieved_docs:
       retrieved_docs = [Document(page_content="No results found")]
   ```

2. **메타데이터 불일치**
   ```python
   # ID 기반 매칭 시 메타데이터 형식 통일
   for doc in docs:
       doc.metadata["id"] = str(doc.metadata["id"])
   ```

3. **성능 최적화**
   ```python
   # 대용량 데이터 처리 시 배치 처리
   batch_size = 100
   for i in range(0, len(queries), batch_size):
       batch = queries[i:i+batch_size]
       # 배치 단위로 평가 수행
   ```

## 📚 참고 자료

- [K-RAG 패키지 문서](https://github.com/your-repo/krag)
- [Information Retrieval 평가 지표](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval))
- [ROUGE 메트릭 설명](https://en.wikipedia.org/wiki/ROUGE_(metric))

## 🚀 다음 단계

1. **PRJ02_W1_003**: 키워드 검색과 하이브리드 검색 비교
2. **PRJ02_W1_004**: 쿼리 확장 기법으로 검색 성능 향상
3. **고급 평가**: 사용자 만족도, 다양성 지표 추가 학습

---

**💡 실무 팁**: 단일 지표에만 의존하지 말고, 시스템의 목적에 맞는 복수의 지표를 조합하여 종합적으로 평가하는 것이 중요합니다.