# RAG 시스템 성능 평가 - 생성 메트릭과 정량적 평가 가이드

## 📚 학습 목표
- RAG 시스템의 검색과 생성 단계별 평가 방법론을 이해한다
- ROUGE, BLEU 등 정량적 평가 지표의 원리와 활용법을 습득한다
- 휴리스틱 평가, 문자열 거리, 임베딩 기반 평가 기법을 실습한다
- LangSmith를 활용한 체계적인 평가 시스템을 구축할 수 있다
- 실무에서 RAG 시스템 품질을 객관적으로 측정하고 개선할 수 있다

## 🔑 핵심 개념

### RAG 평가의 두 축
- **검색(Retrieval) 평가**: 관련 문서를 얼마나 잘 찾아오는가?
- **생성(Generation) 평가**: 찾아온 문서를 바탕으로 얼마나 좋은 답변을 생성하는가?

### 평가 방법론 분류
1. **휴리스틱 평가**: 명확한 규칙 기반 (길이, JSON 유효성 등)
2. **정량적 메트릭**: ROUGE, BLEU 등 수치 기반 비교
3. **의미적 평가**: 임베딩 거리, 코사인 유사도 등
4. **LLM-as-Judge**: 대형 언어모델을 활용한 품질 평가

### 평가 지표의 한계와 보완
- **단어 중첩 기반 지표의 한계**: 의미적 유사성 포착 어려움
- **참조 답안의 품질 의존성**: 평가 기준의 객관성 확보 필요
- **문맥 이해 부족**: 단순 비교로는 실제 품질 측정 한계
- **다차원적 접근**: 여러 지표를 종합하여 균형있는 평가 필요

## 🛠 환경 설정

### 필수 라이브러리 설치
```bash
# 기본 평가 라이브러리
pip install langchain langchain-openai langchain-chroma
pip install korouge-score nltk
pip install rapidfuzz  # 문자열 거리 계산용
pip install kiwisolver kiwipiepy  # 한국어 토크나이저

# LangSmith 평가 도구
pip install langsmith

# 추가 분석 도구
pip install pandas numpy matplotlib seaborn
```

### 환경 변수 설정
```python
# .env 파일
OPENAI_API_KEY=your_openai_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=RAG_Evaluation
```

### 기본 설정
```python
import os
from dotenv import load_dotenv
import warnings
import pandas as pd
import numpy as np

# 환경 변수 로드
load_dotenv()
warnings.filterwarnings("ignore")

# 벡터 저장소 설정
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
chroma_db = Chroma(
    collection_name="db_korean_cosine_metadata",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)
```

## 💻 단계별 구현

### 1단계: 테스트 데이터셋 준비

```python
# 테스트 데이터셋 구조 예시
test_data = {
    'questions': [
        "테슬라의 CEO는 누구인가요?",
        "전기차의 주요 장점은 무엇인가요?",
        "자율주행 기술의 현재 상황은 어떤가요?"
    ],
    'reference_answers': [
        "테슬라의 CEO는 일론 머스크입니다.",
        "전기차의 주요 장점으로는 환경 친화성, 연료비 절약, 조용한 운행 등이 있습니다.",
        "자율주행 기술은 현재 레벨 2-3 단계에 있으며, 완전 자율주행을 위해 지속적으로 발전하고 있습니다."
    ],
    'contexts': [
        ["테슬라는 일론 머스크가 CEO로 있는 전기차 회사입니다.", "..."],
        ["전기차는 배터리로 구동되는 친환경 자동차입니다.", "..."],
        ["자율주행은 인공지능 기술을 활용한 무인 운전 시스템입니다.", "..."]
    ]
}

df_test = pd.DataFrame(test_data)
```

### 2단계: 다양한 검색기 설정

```python
from krag.tokenizers import KiwiTokenizer
from krag.retrievers import KiWiBM25RetrieverWithScore
from langchain.retrievers import EnsembleRetriever

# BM25 검색기 설정 (키워드 기반)
kiwi_tokenizer = KiwiTokenizer(model_type='knlm', typos='basic')
bm25_retriever = KiWiBM25RetrieverWithScore(
    documents=documents,
    kiwi_tokenizer=kiwi_tokenizer,
    k=4
)

# 벡터 검색기 설정 (의미 기반)
vector_retriever = chroma_db.as_retriever(search_kwargs={'k': 4})

# 하이브리드 검색기 설정 (BM25 + Vector)
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]  # 동등한 가중치
)
```

### 3단계: RAG 체인 구성

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

def create_rag_chain(retriever, llm):
    """
    RAG 체인을 생성하는 함수
    """
    template = """다음 맥락을 바탕으로 질문에 답하세요.
    맥락이 질문과 관련이 없다면 '답변에 필요한 근거를 찾지 못했습니다.'라고 답하세요.

    [맥락]
    {context}

    [질문]
    {question}

    [답변]
    """

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

# RAG 체인 생성
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.1)
rag_chain = create_rag_chain(hybrid_retriever, llm)
```

## 🔍 평가 방법론 상세 구현

### 1. 휴리스틱 평가

```python
def evaluate_response_quality(response: str) -> dict:
    """
    응답의 기본적인 품질을 휴리스틱하게 평가
    """
    results = {}

    # 길이 평가
    char_length = len(response)
    results['length_score'] = 1.0 if 50 <= char_length <= 500 else 0.5
    results['char_length'] = char_length

    # 토큰 길이 평가
    kiwi_tokenizer = KiwiTokenizer()
    tokens = kiwi_tokenizer.tokenize(response)
    token_count = len(tokens)
    results['token_score'] = 1.0 if 10 <= token_count <= 150 else 0.5
    results['token_count'] = token_count

    # 완성도 평가 (마침표 확인)
    results['completeness_score'] = 1.0 if response.endswith(('.', '!', '?')) else 0.5

    # 부정적 응답 확인
    negative_phrases = ['모르겠습니다', '답변할 수 없습니다', '근거를 찾지 못했습니다']
    results['positive_response'] = not any(phrase in response for phrase in negative_phrases)

    # 종합 점수 계산
    scores = [results['length_score'], results['token_score'],
              results['completeness_score']]
    results['overall_score'] = sum(scores) / len(scores)

    return results

# 사용 예시
test_response = "테슬라의 CEO는 일론 머스크입니다. 그는 2008년부터 테슬라를 이끌고 있습니다."
quality_results = evaluate_response_quality(test_response)
print("휴리스틱 평가 결과:", quality_results)
```

### 2. ROUGE 점수 계산

```python
from korouge_score import rouge_scorer
from krag.tokenizers import KiwiTokenizer

class CustomKiwiTokenizer(KiwiTokenizer):
    def tokenize(self, text):
        return [t.form for t in super().tokenize(text)]

def calculate_rouge_scores(reference: str, generated: str) -> dict:
    """
    ROUGE 점수를 계산하는 함수
    """
    kiwi_tokenizer = CustomKiwiTokenizer(model_type='knlm', typos='basic')
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        tokenizer=kiwi_tokenizer
    )

    scores = scorer.score(reference, generated)

    return {
        'rouge1_f1': scores['rouge1'].fmeasure,
        'rouge1_precision': scores['rouge1'].precision,
        'rouge1_recall': scores['rouge1'].recall,
        'rouge2_f1': scores['rouge2'].fmeasure,
        'rouge2_precision': scores['rouge2'].precision,
        'rouge2_recall': scores['rouge2'].recall,
        'rougeL_f1': scores['rougeL'].fmeasure,
        'rougeL_precision': scores['rougeL'].precision,
        'rougeL_recall': scores['rougeL'].recall
    }

# 사용 예시
reference = "테슬라의 CEO는 일론 머스크입니다."
generated = "일론 머스크가 테슬라의 최고경영자입니다."
rouge_results = calculate_rouge_scores(reference, generated)
print("ROUGE 점수:", rouge_results)
```

### 3. BLEU 점수 계산

```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import List, Union

def calculate_bleu_score(
    reference: Union[str, List[str]],
    hypothesis: str,
    weights: tuple = (0.25, 0.25, 0.25, 0.25),
    tokenizer=None
) -> dict:
    """
    BLEU 점수 계산 함수
    """
    if tokenizer is None:
        tokenizer = CustomKiwiTokenizer(model_type='knlm', typos='basic')

    try:
        # 참조 텍스트 처리
        if isinstance(reference, str):
            references = [tokenizer.tokenize(reference)]
        else:
            references = [tokenizer.tokenize(ref) for ref in reference]

        # 생성 텍스트 토크나이징
        hypothesis_tokens = tokenizer.tokenize(hypothesis)

        # 개별 n-gram 점수 계산
        bleu_scores = {}
        for i in range(1, 5):  # BLEU-1부터 BLEU-4까지
            weight = [0] * 4
            weight[i-1] = 1.0
            score = sentence_bleu(
                references,
                hypothesis_tokens,
                weights=weight,
                smoothing_function=SmoothingFunction().method1
            )
            bleu_scores[f'bleu_{i}'] = score

        # 전체 BLEU 점수 (균등 가중치)
        overall_bleu = sentence_bleu(
            references,
            hypothesis_tokens,
            weights=weights,
            smoothing_function=SmoothingFunction().method1
        )
        bleu_scores['bleu_overall'] = overall_bleu

        return bleu_scores

    except Exception as e:
        print(f"BLEU 점수 계산 오류: {str(e)}")
        return {'bleu_overall': 0.0, 'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0}

# 사용 예시
bleu_results = calculate_bleu_score(reference, generated)
print("BLEU 점수:", bleu_results)
```

### 4. 임베딩 기반 유사도 평가

```python
from langchain.evaluation import load_evaluator
from langchain_openai import OpenAIEmbeddings

def calculate_semantic_similarity(reference: str, generated: str) -> dict:
    """
    임베딩 기반 의미적 유사도 계산
    """
    # 임베딩 기반 평가기 생성
    embedding_evaluator = load_evaluator(
        evaluator='embedding_distance',
        distance_metric='cosine',
        embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
    )

    # 코사인 유사도 계산
    result = embedding_evaluator.evaluate_strings(
        prediction=generated,
        reference=reference
    )

    # 문자열 거리도 함께 계산
    string_evaluator = load_evaluator(
        evaluator="string_distance",
        distance="levenshtein"
    )

    string_result = string_evaluator.evaluate_strings(
        prediction=generated,
        reference=reference
    )

    return {
        'cosine_similarity': 1 - result['score'],  # 거리를 유사도로 변환
        'cosine_distance': result['score'],
        'levenshtein_distance': string_result['score'],
        'levenshtein_similarity': 1 - string_result['score']
    }

# 사용 예시
similarity_results = calculate_semantic_similarity(reference, generated)
print("유사도 평가 결과:", similarity_results)
```

## 🎯 실습 문제

### 기초 실습
1. **기본 평가 시스템 구축**
   - 5개의 질문-답변 쌍으로 테스트 데이터셋을 만드세요
   - RAG 체인으로 답변을 생성하고 기본 품질을 평가하세요

2. **메트릭 비교 실습**
   - 동일한 질문에 대해 서로 다른 3개의 답변을 만드세요
   - ROUGE, BLEU, 임베딩 유사도로 각각 평가하고 결과를 비교하세요

### 응용 실습
3. **다중 모델 비교**
   - OpenAI, Google Gemini, Ollama 모델로 동일한 질문에 답변 생성
   - 각 모델의 답변을 다양한 지표로 평가하고 성능표 작성

4. **검색기 성능 비교**
   - BM25, Vector, Hybrid 검색기를 각각 사용한 RAG 시스템 구축
   - 검색 품질이 최종 답변 품질에 미치는 영향 분석

### 심화 실습
5. **종합 평가 시스템**
   - 휴리스틱, 정량적, 의미적 평가를 통합한 종합 점수 시스템 개발
   - 가중치를 조정하며 최적의 평가 모델 찾기

## ✅ 솔루션 예시

### 실습 1: 종합 평가 함수
```python
def comprehensive_evaluation(reference: str, generated: str) -> dict:
    """
    모든 평가 지표를 종합한 평가 함수
    """
    results = {}

    # 휴리스틱 평가
    heuristic_results = evaluate_response_quality(generated)
    results.update({f"heuristic_{k}": v for k, v in heuristic_results.items()})

    # ROUGE 점수
    rouge_results = calculate_rouge_scores(reference, generated)
    results.update(rouge_results)

    # BLEU 점수
    bleu_results = calculate_bleu_score(reference, generated)
    results.update(bleu_results)

    # 의미적 유사도
    similarity_results = calculate_semantic_similarity(reference, generated)
    results.update(similarity_results)

    # 종합 점수 계산 (가중 평균)
    weights = {
        'quality': 0.2,
        'rouge': 0.3,
        'bleu': 0.2,
        'semantic': 0.3
    }

    quality_score = heuristic_results['overall_score']
    rouge_score = rouge_results['rouge1_f1']
    bleu_score = bleu_results['bleu_overall']
    semantic_score = similarity_results['cosine_similarity']

    overall_score = (
        weights['quality'] * quality_score +
        weights['rouge'] * rouge_score +
        weights['bleu'] * bleu_score +
        weights['semantic'] * semantic_score
    )

    results['comprehensive_score'] = overall_score

    return results

# 사용 예시
evaluation_results = comprehensive_evaluation(reference, generated)
print("종합 평가 결과:", evaluation_results)
```

### 실습 2: 배치 평가 시스템
```python
def batch_evaluate_rag_system(test_questions: List[str],
                             reference_answers: List[str],
                             rag_chain) -> pd.DataFrame:
    """
    배치로 RAG 시스템 평가
    """
    results = []

    for i, (question, reference) in enumerate(zip(test_questions, reference_answers)):
        print(f"평가 중... {i+1}/{len(test_questions)}")

        # RAG로 답변 생성
        try:
            generated = rag_chain.invoke(question)

            # 종합 평가 실행
            eval_result = comprehensive_evaluation(reference, generated)
            eval_result['question'] = question
            eval_result['reference'] = reference
            eval_result['generated'] = generated
            eval_result['success'] = True

        except Exception as e:
            eval_result = {
                'question': question,
                'reference': reference,
                'generated': f"오류: {str(e)}",
                'success': False,
                'comprehensive_score': 0.0
            }

        results.append(eval_result)

    df_results = pd.DataFrame(results)
    return df_results

# 사용 예시
test_questions = [
    "테슬라의 CEO는 누구인가요?",
    "전기차의 장점은 무엇인가요?",
    "자율주행 기술의 현 상황은?"
]

reference_answers = [
    "테슬라의 CEO는 일론 머스크입니다.",
    "전기차는 환경 친화적이고 연료비가 절약됩니다.",
    "자율주행 기술은 현재 레벨 2-3 단계입니다."
]

evaluation_df = batch_evaluate_rag_system(test_questions, reference_answers, rag_chain)
print("배치 평가 완료!")
print(evaluation_df[['question', 'comprehensive_score', 'rouge1_f1', 'cosine_similarity']].head())
```

### 실습 3: 성능 시각화 대시보드
```python
import matplotlib.pyplot as plt
import seaborn as sns

def create_evaluation_dashboard(df_results: pd.DataFrame):
    """
    평가 결과를 시각화하는 대시보드 생성
    """
    plt.rcParams['font.family'] = 'DejaVu Sans'  # 한글 폰트 설정

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 종합 점수 분포
    axes[0,0].hist(df_results['comprehensive_score'], bins=10, alpha=0.7)
    axes[0,0].set_title('종합 점수 분포')
    axes[0,0].set_xlabel('점수')
    axes[0,0].set_ylabel('빈도')

    # 2. 메트릭별 상관관계
    metrics = ['rouge1_f1', 'bleu_overall', 'cosine_similarity', 'comprehensive_score']
    correlation_matrix = df_results[metrics].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=axes[0,1])
    axes[0,1].set_title('메트릭 간 상관관계')

    # 3. 질문별 성능
    axes[1,0].bar(range(len(df_results)), df_results['comprehensive_score'])
    axes[1,0].set_title('질문별 성능')
    axes[1,0].set_xlabel('질문 번호')
    axes[1,0].set_ylabel('종합 점수')

    # 4. 메트릭 비교
    metrics_to_plot = ['rouge1_f1', 'bleu_overall', 'cosine_similarity']
    x = range(len(df_results))
    width = 0.25

    for i, metric in enumerate(metrics_to_plot):
        axes[1,1].bar([xi + width*i for xi in x], df_results[metric],
                     width, label=metric, alpha=0.8)

    axes[1,1].set_title('메트릭별 성능 비교')
    axes[1,1].set_xlabel('질문 번호')
    axes[1,1].set_ylabel('점수')
    axes[1,1].legend()

    plt.tight_layout()
    plt.show()

    # 요약 통계
    print("\n=== 평가 결과 요약 ===")
    print(f"평균 종합 점수: {df_results['comprehensive_score'].mean():.3f}")
    print(f"평균 ROUGE-1 F1: {df_results['rouge1_f1'].mean():.3f}")
    print(f"평균 BLEU: {df_results['bleu_overall'].mean():.3f}")
    print(f"평균 코사인 유사도: {df_results['cosine_similarity'].mean():.3f}")
    print(f"성공률: {df_results['success'].mean():.1%}")

# 사용 예시
create_evaluation_dashboard(evaluation_df)
```

## 🚀 실무 활용 예시

### 1. A/B 테스트 시스템

```python
class RAGABTester:
    def __init__(self):
        self.models = {}
        self.test_results = []

    def register_model(self, name: str, rag_chain):
        """모델 등록"""
        self.models[name] = rag_chain

    def run_ab_test(self, test_questions: List[str],
                    reference_answers: List[str]) -> pd.DataFrame:
        """A/B 테스트 실행"""
        all_results = []

        for model_name, rag_chain in self.models.items():
            print(f"\n{model_name} 모델 테스트 중...")

            model_results = batch_evaluate_rag_system(
                test_questions, reference_answers, rag_chain
            )
            model_results['model'] = model_name
            all_results.append(model_results)

        combined_results = pd.concat(all_results, ignore_index=True)
        return combined_results

    def analyze_results(self, results_df: pd.DataFrame):
        """결과 분석 및 리포트 생성"""
        model_performance = results_df.groupby('model').agg({
            'comprehensive_score': ['mean', 'std'],
            'rouge1_f1': 'mean',
            'bleu_overall': 'mean',
            'cosine_similarity': 'mean',
            'success': 'mean'
        }).round(3)

        print("\n=== 모델별 성능 비교 ===")
        print(model_performance)

        # 통계적 유의성 검정
        from scipy import stats
        models = results_df['model'].unique()

        if len(models) == 2:
            model_a = results_df[results_df['model'] == models[0]]['comprehensive_score']
            model_b = results_df[results_df['model'] == models[1]]['comprehensive_score']

            t_stat, p_value = stats.ttest_ind(model_a, model_b)
            print(f"\n통계적 유의성 검정:")
            print(f"t-statistic: {t_stat:.4f}")
            print(f"p-value: {p_value:.4f}")
            print(f"유의미한 차이: {'예' if p_value < 0.05 else '아니오'}")

        return model_performance

# 사용 예시
ab_tester = RAGABTester()
ab_tester.register_model("OpenAI", openai_rag_chain)
ab_tester.register_model("Gemini", gemini_rag_chain)

ab_results = ab_tester.run_ab_test(test_questions, reference_answers)
performance_summary = ab_tester.analyze_results(ab_results)
```

### 2. 지속적 품질 모니터링 시스템

```python
class RAGQualityMonitor:
    def __init__(self, quality_threshold: float = 0.7):
        self.quality_threshold = quality_threshold
        self.performance_history = []
        self.alerts = []

    def evaluate_single_response(self, question: str, reference: str,
                                generated: str) -> dict:
        """단일 응답 평가"""
        try:
            evaluation = comprehensive_evaluation(reference, generated)
            evaluation['timestamp'] = pd.Timestamp.now()
            evaluation['question'] = question
            evaluation['reference'] = reference
            evaluation['generated'] = generated

            # 품질 임계값 확인
            if evaluation['comprehensive_score'] < self.quality_threshold:
                self.alerts.append({
                    'timestamp': evaluation['timestamp'],
                    'issue': 'Low Quality Response',
                    'score': evaluation['comprehensive_score'],
                    'question': question
                })

            self.performance_history.append(evaluation)
            return evaluation

        except Exception as e:
            error_eval = {
                'timestamp': pd.Timestamp.now(),
                'question': question,
                'reference': reference,
                'generated': generated,
                'error': str(e),
                'comprehensive_score': 0.0
            }
            self.alerts.append({
                'timestamp': error_eval['timestamp'],
                'issue': 'System Error',
                'error': str(e),
                'question': question
            })
            return error_eval

    def get_performance_report(self, days: int = 7) -> dict:
        """성능 리포트 생성"""
        if not self.performance_history:
            return {"message": "평가 데이터가 없습니다."}

        df_history = pd.DataFrame(self.performance_history)
        cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days)
        recent_data = df_history[df_history['timestamp'] > cutoff_date]

        if recent_data.empty:
            return {"message": f"최근 {days}일 데이터가 없습니다."}

        report = {
            'period': f'최근 {days}일',
            'total_evaluations': len(recent_data),
            'average_score': recent_data['comprehensive_score'].mean(),
            'score_trend': 'improving' if recent_data['comprehensive_score'].corr(
                range(len(recent_data))) > 0 else 'declining',
            'quality_rate': (recent_data['comprehensive_score'] >= self.quality_threshold).mean(),
            'recent_alerts': len([a for a in self.alerts
                                if a['timestamp'] > cutoff_date])
        }

        return report

    def plot_performance_trend(self, days: int = 30):
        """성능 트렌드 시각화"""
        if not self.performance_history:
            print("평가 데이터가 없습니다.")
            return

        df_history = pd.DataFrame(self.performance_history)
        cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days)
        recent_data = df_history[df_history['timestamp'] > cutoff_date]

        if recent_data.empty:
            print(f"최근 {days}일 데이터가 없습니다.")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(recent_data['timestamp'], recent_data['comprehensive_score'],
                marker='o', alpha=0.7)
        plt.axhline(y=self.quality_threshold, color='r', linestyle='--',
                   label=f'품질 임계값 ({self.quality_threshold})')
        plt.title(f'RAG 시스템 성능 트렌드 (최근 {days}일)')
        plt.xlabel('날짜')
        plt.ylabel('종합 점수')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# 사용 예시
quality_monitor = RAGQualityMonitor(quality_threshold=0.75)

# 실제 운영 중 평가
for question, reference in zip(test_questions, reference_answers):
    generated = rag_chain.invoke(question)
    evaluation = quality_monitor.evaluate_single_response(question, reference, generated)

# 성능 리포트 확인
performance_report = quality_monitor.get_performance_report()
print("성능 리포트:", performance_report)

# 트렌드 시각화
quality_monitor.plot_performance_trend(days=30)
```

### 3. 자동 평가 파이프라인

```python
from typing import Callable, Dict, Any
import json
from datetime import datetime

class AutoEvaluationPipeline:
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.evaluation_history = []
        self.benchmarks = {}

    def _load_config(self, config_path: str) -> dict:
        """설정 파일 로드"""
        default_config = {
            'evaluation_metrics': ['rouge', 'bleu', 'semantic', 'heuristic'],
            'quality_thresholds': {
                'rouge1_f1': 0.3,
                'bleu_overall': 0.2,
                'cosine_similarity': 0.7,
                'comprehensive_score': 0.6
            },
            'alert_settings': {
                'enable_alerts': True,
                'alert_threshold': 0.5,
                'consecutive_failures': 3
            }
        }

        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                custom_config = json.load(f)
            default_config.update(custom_config)

        return default_config

    def register_benchmark(self, name: str, test_data: Dict[str, Any]):
        """벤치마크 데이터 등록"""
        self.benchmarks[name] = {
            'questions': test_data['questions'],
            'references': test_data['references'],
            'created_at': datetime.now()
        }

    def run_evaluation(self, rag_chain, benchmark_name: str) -> Dict[str, Any]:
        """자동 평가 실행"""
        if benchmark_name not in self.benchmarks:
            raise ValueError(f"벤치마크 '{benchmark_name}'를 찾을 수 없습니다.")

        benchmark = self.benchmarks[benchmark_name]
        questions = benchmark['questions']
        references = benchmark['references']

        print(f"벤치마크 '{benchmark_name}' 평가 시작...")

        results = []
        failed_count = 0

        for i, (question, reference) in enumerate(zip(questions, references)):
            try:
                # RAG 체인으로 답변 생성
                generated = rag_chain.invoke(question)

                # 종합 평가
                evaluation = comprehensive_evaluation(reference, generated)
                evaluation.update({
                    'question_id': i,
                    'question': question,
                    'reference': reference,
                    'generated': generated,
                    'success': True
                })

                # 품질 임계값 확인
                quality_checks = {}
                for metric, threshold in self.config['quality_thresholds'].items():
                    if metric in evaluation:
                        quality_checks[f'{metric}_pass'] = evaluation[metric] >= threshold

                evaluation.update(quality_checks)
                results.append(evaluation)

                print(f"진행률: {i+1}/{len(questions)} ✓")

            except Exception as e:
                failed_count += 1
                error_result = {
                    'question_id': i,
                    'question': question,
                    'reference': reference,
                    'generated': f'ERROR: {str(e)}',
                    'success': False,
                    'comprehensive_score': 0.0,
                    'error': str(e)
                }
                results.append(error_result)
                print(f"진행률: {i+1}/{len(questions)} ✗ (오류: {str(e)[:50]}...)")

        # 평가 결과 요약
        evaluation_summary = self._summarize_results(results, benchmark_name)

        # 히스토리에 저장
        self.evaluation_history.append({
            'benchmark_name': benchmark_name,
            'timestamp': datetime.now(),
            'results': results,
            'summary': evaluation_summary
        })

        return evaluation_summary

    def _summarize_results(self, results: List[Dict], benchmark_name: str) -> Dict:
        """평가 결과 요약"""
        df_results = pd.DataFrame(results)

        summary = {
            'benchmark_name': benchmark_name,
            'timestamp': datetime.now().isoformat(),
            'total_questions': len(results),
            'success_rate': df_results['success'].mean(),
            'failed_count': (~df_results['success']).sum(),
            'metrics': {}
        }

        # 성공한 케이스에 대한 메트릭 계산
        successful_results = df_results[df_results['success'] == True]

        if len(successful_results) > 0:
            metric_columns = ['comprehensive_score', 'rouge1_f1', 'bleu_overall', 'cosine_similarity']

            for metric in metric_columns:
                if metric in successful_results.columns:
                    summary['metrics'][metric] = {
                        'mean': float(successful_results[metric].mean()),
                        'std': float(successful_results[metric].std()),
                        'min': float(successful_results[metric].min()),
                        'max': float(successful_results[metric].max())
                    }

        # 품질 임계값 통과율
        threshold_metrics = {}
        for metric, threshold in self.config['quality_thresholds'].items():
            pass_column = f'{metric}_pass'
            if pass_column in df_results.columns:
                threshold_metrics[metric] = {
                    'threshold': threshold,
                    'pass_rate': float(df_results[pass_column].mean())
                }

        summary['threshold_analysis'] = threshold_metrics

        # 알림 조건 확인
        if self.config['alert_settings']['enable_alerts']:
            overall_score = summary['metrics'].get('comprehensive_score', {}).get('mean', 0)
            if overall_score < self.config['alert_settings']['alert_threshold']:
                summary['alert'] = {
                    'type': 'LOW_PERFORMANCE',
                    'message': f"평균 점수({overall_score:.3f})가 임계값({self.config['alert_settings']['alert_threshold']})보다 낮습니다.",
                    'severity': 'HIGH' if overall_score < 0.3 else 'MEDIUM'
                }

        return summary

    def export_results(self, filepath: str, format: str = 'json'):
        """평가 결과 내보내기"""
        if not self.evaluation_history:
            print("내보낼 평가 결과가 없습니다.")
            return

        if format.lower() == 'json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.evaluation_history, f, ensure_ascii=False, indent=2, default=str)
        elif format.lower() == 'csv':
            # 모든 평가 결과를 하나의 DataFrame으로 합치기
            all_results = []
            for evaluation in self.evaluation_history:
                for result in evaluation['results']:
                    result['evaluation_timestamp'] = evaluation['timestamp']
                    result['benchmark_name'] = evaluation['benchmark_name']
                    all_results.append(result)

            df_all = pd.DataFrame(all_results)
            df_all.to_csv(filepath, index=False, encoding='utf-8-sig')

        print(f"평가 결과가 {filepath}에 저장되었습니다.")

# 사용 예시
pipeline = AutoEvaluationPipeline()

# 벤치마크 등록
benchmark_data = {
    'questions': test_questions,
    'references': reference_answers
}
pipeline.register_benchmark('tesla_qa_v1', benchmark_data)

# 자동 평가 실행
evaluation_summary = pipeline.run_evaluation(rag_chain, 'tesla_qa_v1')
print("자동 평가 완료!")
print(json.dumps(evaluation_summary, indent=2, ensure_ascii=False, default=str))

# 결과 내보내기
pipeline.export_results('evaluation_results.json')
pipeline.export_results('evaluation_results.csv', format='csv')
```

## 📖 참고 자료

### 평가 메트릭 관련
- [ROUGE 논문 원본](https://aclanthology.org/W04-1013/)
- [BLEU 논문 원본](https://www.aclweb.org/anthology/P02-1040/)
- [RAG 평가 방법론 최신 연구](https://arxiv.org/abs/2405.07437)

### LangChain 평가 도구
- [LangChain Evaluation Guide](https://python.langchain.com/docs/guides/evaluation/)
- [LangSmith Evaluation Docs](https://docs.smith.langchain.com/evaluation)

### 한국어 자연어 처리
- [KiWi 토크나이저](https://github.com/bab2min/kiwipiepy)
- [한국어 ROUGE 구현](https://github.com/gucci-j/korouge-score)

### 추가 학습 자료
- [RAG 시스템 평가 best practices](https://python.langchain.com/docs/use_cases/question_answering/evaluation/)
- [멀티모달 RAG 평가 방법](https://python.langchain.com/docs/integrations/retrievers/)
- [프로덕션 RAG 시스템 모니터링](https://docs.smith.langchain.com/tracing)

이 가이드를 통해 RAG 시스템의 품질을 객관적으로 측정하고, 지속적으로 개선할 수 있는 체계적인 평가 시스템을 구축할 수 있습니다. 실무에서는 여러 평가 지표를 조합하여 균형잡힌 품질 평가를 수행하는 것이 중요합니다.