# RAG 답변 평가 - 정량적 지표 활용 정리노트

## 📖 이 수업에서 배운 것

### 🎯 핵심 개념
- **RAG 평가**: AI가 생성한 답변이 얼마나 좋은지 숫자로 측정하는 방법
- **정량적 지표**: 주관적인 판단이 아닌 객관적인 숫자로 평가
- **검색 평가**: 관련 문서를 잘 찾았는지 평가
- **생성 평가**: 답변이 얼마나 좋은지 평가

---

## 🔍 왜 RAG 시스템을 평가해야 할까?

### 실제 사용하기 전에 확인해야 할 것들
1. **정확성**: 답변이 맞는가?
2. **관련성**: 질문과 관련된 답변인가?
3. **일관성**: 같은 질문에 비슷한 답변을 하는가?
4. **완성도**: 답변이 충분히 자세한가?

---

## 🛠️ 실습에서 사용한 검색 도구들

### 1. ChromaDB (벡터 검색)
```python
# 의미가 비슷한 문서를 찾는 검색
chroma_k = chroma_db.as_retriever(search_kwargs={'k': 4})
```

### 2. BM25 (키워드 검색)
```python
# 단어가 일치하는 문서를 찾는 검색
from krag.tokenizers import KiwiTokenizer
from krag.retrievers import KiWiBM25RetrieverWithScore

kiwi_tokenizer = KiwiTokenizer(model_type='knlm', typos='basic')
bm25_db = KiWiBM25RetrieverWithScore(documents=documents, kiwi_tokenizer=kiwi_tokenizer, k=4)
```

### 3. 하이브리드 검색 (둘 다 사용)
```python
# 벡터 검색 + 키워드 검색을 합친 검색
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_db, chroma_k],
    weights=[0.5, 0.5]  # 50:50 비율로 합치기
)
```

---

## 📏 3가지 평가 방법

### 1. 휴리스틱 평가 (간단한 규칙으로 평가)

**길이 평가** - 답변이 너무 짧거나 길지 않은지 확인
```python
def evaluate_string_length(text, min_length=50, max_length=200):
    length = len(text)
    return {
        "score": min_length <= length <= max_length,
        "length": length
    }
```

**토큰 수 평가** - 단어 개수가 적당한지 확인
```python
def evaluate_token_length(text, tokenizer, min_tokens=10, max_tokens=100):
    tokens = tokenizer.tokenize(text)
    return {
        "score": min_tokens <= len(tokens) <= max_tokens,
        "num_tokens": len(tokens)
    }
```

### 2. ROUGE 점수 (단어 겹치는 정도 측정)

**개념**: 정답과 AI 답변에서 같은 단어가 얼마나 겹치는지 측정

```python
from korouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    tokenizer=kiwi_tokenizer
)

scores = scorer.score(정답_텍스트, AI_답변_텍스트)
```

**종류**:
- **ROUGE-1**: 단어 하나씩 비교 (예: "테슬라", "회장")
- **ROUGE-2**: 단어 두 개씩 비교 (예: "테슬라 회장", "일론 머스크")
- **ROUGE-L**: 가장 긴 공통 문장 찾기

### 3. BLEU 점수 (번역 품질 측정 방식)

**개념**: 원래 번역 평가용이지만 텍스트 생성 품질 측정에도 사용

```python
from nltk.translate.bleu_score import sentence_bleu

def calculate_bleu_score(reference, hypothesis, tokenizer):
    references = [tokenizer.tokenize(reference)]
    hypothesis_tokens = tokenizer.tokenize(hypothesis)

    score = sentence_bleu(references, hypothesis_tokens)
    return score
```

---

## 🧠 임베딩 기반 평가

### 문자열 거리 평가
```python
# 글자 단위로 얼마나 다른지 측정
string_evaluator = load_evaluator(
    evaluator="string_distance",
    distance="levenshtein"  # 편집 거리
)
```

### 의미 유사도 평가
```python
# AI가 이해하는 의미가 얼마나 비슷한지 측정
embedding_evaluator = load_evaluator(
    evaluator='embedding_distance',
    distance_metric='cosine',
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
)
```

---

## 💡 실습에서 겪은 문제와 해결

### 문제 1: `answer` 변수가 정의되지 않음
**원인**: RAG 체인을 먼저 실행하지 않음
**해결**:
```python
# 먼저 답변 생성
answer = openai_rag_chain.invoke("질문")
# 그 다음 평가
result = evaluate_string_length(answer)
```

### 문제 2: `ragas_evaluation` 데이터가 없음
**원인**: 평가용 CSV 파일이 없음
**해결**: `data/evaluation_result.csv` 파일 생성

### 문제 3: f-string에서 문법 오류
**원인**: f-string 안에서 같은 종류의 따옴표 중복 사용
**해결**:
```python
# 잘못된 코드 - f-string과 딕셔너리 키에서 모두 " 사용
print(f"BM25 점수: {doc.metadata["bm25_score"]:.2f}")

# 올바른 코드 - 딕셔너리 키에는 ' 사용
print(f"BM25 점수: {doc.metadata['bm25_score']:.2f}")

# 또는 변수로 미리 빼내기
score = doc.metadata["bm25_score"]
print(f"BM25 점수: {score:.2f}")
```

---

## 📊 평가 점수 해석하는 법

### ROUGE 점수 (0~1, 높을수록 좋음)
- **0.6 이상**: 아주 좋음 😊 (상용 서비스 수준)
- **0.4~0.6**: 괜찮음 😐 (실험/개발 단계 목표)
- **0.2~0.4**: 개선 필요 😞 (기본 기능은 작동)
- **0.2 미만**: 심각한 문제 😱 (시스템 점검 필요)

### BLEU 점수 (0~1, 높을수록 좋음)
- **0.4 이상**: 아주 좋음 😊 (전문가 수준)
- **0.3~0.4**: 좋음 😐 (실용 가능한 수준)
- **0.2~0.3**: 보통 😐 (개선 여지 있음)
- **0.1~0.2**: 개선 필요 😞 (기본 동작)
- **0.1 미만**: 심각한 문제 😱 (다시 설계 필요)

### 코사인 유사도 (0~1, 높을수록 좋음)
- **0.8 이상**: 매우 유사 😊 (의미적으로 거의 동일)
- **0.6~0.8**: 유사 😐 (같은 맥락, 표현만 다름)
- **0.4~0.6**: 보통 😐 (관련은 있으나 차이 존재)
- **0.2~0.4**: 다름 😞 (약간의 관련성만 있음)
- **0.2 미만**: 전혀 다름 😱 (관련성 거의 없음)

### 🎯 실제 프로젝트 목표 점수
- **프로토타입 단계**: ROUGE > 0.3, BLEU > 0.2, 코사인 > 0.5
- **베타 테스트**: ROUGE > 0.5, BLEU > 0.3, 코사인 > 0.7
- **상용 서비스**: ROUGE > 0.6, BLEU > 0.4, 코사인 > 0.8

---

## 🔧 실제 평가해본 과정

### 1단계: 데이터 준비
```python
# 테스트 데이터 로드
df_qa_test = pd.read_excel("data/testset.xlsx")
ragas_evaluation = pd.read_csv('data/evaluation_result.csv')
```

### 2단계: RAG 시스템으로 답변 생성
```python
question = df_qa_test.iloc[0]['user_input']
answer = openai_rag_chain.invoke(question)
ground_truth = df_qa_test.iloc[0]['reference']
```

### 3단계: 여러 지표로 평가
```python
# ROUGE 점수
rouge_scores = scorer.score(ground_truth, answer)

# BLEU 점수
bleu_score = calculate_bleu_score(ground_truth, answer, tokenizer)

# 임베딩 유사도
embedding_score = embedding_evaluator.evaluate_strings(
    prediction=answer,
    reference=ground_truth
)
```

---

## 🎯 평가의 한계점과 주의사항

### 한계점
1. **단어 매칭만 확인**: "좋다"와 "훌륭하다"를 다르게 봄
2. **문맥 이해 부족**: "강아지가 공을 물었다"와 "공을 강아지가 물었다"를 다르게 평가
3. **언어별 특성 무시**: 한국어 특성을 완전히 반영하지 못함

### 해결 방법
1. **여러 지표 함께 사용**: ROUGE + BLEU + 임베딩 유사도
2. **사람이 직접 평가**: 자동 평가와 함께 전문가 검토
3. **도메인별 맞춤 평가**: 분야에 맞는 평가 기준 개발

---

## 🤔 복습 질문

1. **ROUGE와 BLEU의 차이는?**
   - ROUGE: 재현율(Recall) 중심 - 정답에 있는 단어를 얼마나 찾았나
   - BLEU: 정밀도(Precision) 중심 - 생성한 단어가 얼마나 정확한가

2. **하이브리드 검색이란?**
   - 벡터 검색(의미 기반) + BM25 검색(키워드 기반)을 합친 것

3. **임베딩 유사도 평가의 장점은?**
   - 단어가 달라도 의미가 비슷하면 높은 점수를 줌

4. **언제 어떤 평가 지표를 사용해야 할까?**
   - **요약 작업**: ROUGE (핵심 정보 포함 여부 중요)
   - **번역 작업**: BLEU (정확한 단어 선택 중요)
   - **질문-답변**: 임베딩 유사도 (의미 전달이 중요)
   - **창작 작업**: 휴리스틱 + 사람 평가 (창의성은 수치로 측정 어려움)

5. **평가 점수가 낮을 때 개선 방법은?**
   - **ROUGE 낮음**: 더 관련성 높은 문서 검색, 프롬프트 개선
   - **BLEU 낮음**: 더 정확한 단어 사용하도록 모델 튜닝
   - **임베딩 유사도 낮음**: 전체적인 답변 맥락과 구조 개선

---

## 💻 내가 구현해본 평가 시스템

### 기본 패턴
1. 테스트 데이터 준비
2. RAG 시스템으로 답변 생성
3. 여러 지표로 평가 (ROUGE, BLEU, 임베딩)
4. 결과를 표로 정리
5. 평균 점수 계산

### 핵심 코드 패턴
```python
# 1. 평가 도구 준비
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], tokenizer=tokenizer)

# 2. 각 질문에 대해 평가
for idx, row in test_data.iterrows():
    reference = row['정답']
    prediction = rag_chain.invoke(row['질문'])

    # 3. 점수 계산
    rouge_scores = scorer.score(reference, prediction)
    bleu_score = calculate_bleu_score(reference, prediction, tokenizer)

    # 4. 결과 저장
    results.append({
        'rouge1': rouge_scores['rouge1'].fmeasure,
        'bleu': bleu_score
    })

# 5. 평균 계산
print(f"평균 ROUGE-1: {np.mean([r['rouge1'] for r in results]):.3f}")
```

---

## 🎯 다음에 공부할 것
- [ ] 더 고급 평가 지표 (BertScore, METEOR 등)
- [ ] 사람이 직접 평가하는 방법
- [ ] LLM을 평가자로 사용하는 방법 (LLM-as-judge)
- [ ] 실시간 평가 시스템 구축
- [ ] A/B 테스트로 서로 다른 RAG 시스템 비교
- [ ] 도메인별 평가 기준 설정 (의료, 법률, 기술 등)

## 💡 실전 꿀팁

### 평가할 때 꼭 기억할 것
1. **한 가지 지표만 믿지 말기**: 3개 이상의 다른 지표로 종합 판단
2. **사람 눈으로 확인하기**: 점수가 높아도 실제 답변이 이상할 수 있음
3. **도메인별 기준 다르게**: 창작은 ROUGE 낮아도 OK, 사실 확인은 높아야 함
4. **지속적 모니터링**: 처음에 좋았던 시스템도 시간 지나면 성능 떨어질 수 있음

### 효율적인 평가 워크플로우
1. **개발 단계**: 휴리스틱 평가로 빠른 확인
2. **테스트 단계**: ROUGE, BLEU, 임베딩 점수로 정량 평가
3. **배포 전**: 실제 사용자가 직접 평가
4. **운영 중**: 정기적인 성능 모니터링

---

*이 노트는 'PRJ02_W2_002_Generation_Metrics.ipynb' 수업 내용을 정리한 것입니다.*