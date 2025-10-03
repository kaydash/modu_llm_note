# LLM-as-Judge 평가 시스템 - 언어모델 기반 품질 평가 가이드

## 📚 학습 목표
- LLM-as-Judge의 개념과 동작 원리를 이해한다
- Reference-free와 Reference-based 평가 방식의 차이점을 파악한다
- LangChain을 활용한 체계적인 LLM 평가 시스템을 구축할 수 있다
- 다양한 평가 기준과 프롬프트 설계 기법을 습득한다
- 실무에서 자동화된 품질 평가 시스템을 구현할 수 있다

## 🔑 핵심 개념

### LLM-as-Judge란?
- **정의**: 대형 언어모델을 평가자(Judge)로 활용하여 텍스트 품질을 자동으로 평가하는 방법론
- **배경**: 전통적인 메트릭(ROUGE, BLEU)의 한계를 극복하고, 인간의 주관적 평가를 모방
- **장점**: 문맥 이해력, 다차원적 평가, 유연한 기준 적용 가능
- **활용 분야**: RAG 시스템, 챗봇, 요약, 번역, 창작 등 다양한 자연어 생성 작업

### 평가 방식 분류
1. **Reference-free 평가**: 참조 답안 없이 독립적 품질 기준으로 평가
2. **Reference-based 평가**: 참조 답안과 비교하여 상대적 품질 평가
3. **Pairwise 평가**: 두 답변을 직접 비교하여 우열 판단
4. **Scoring 평가**: 절대적 점수로 품질을 수치화

### 평가 기준 요소
- **정확성(Accuracy)**: 사실적 정보의 정확도
- **관련성(Relevance)**: 질문과의 연관성
- **완전성(Completeness)**: 답변의 충분성
- **명확성(Clarity)**: 이해하기 쉬운 표현
- **일관성(Consistency)**: 논리적 모순 없음
- **유용성(Helpfulness)**: 사용자에게 도움이 되는 정도

## 🛠 환경 설정

### 필수 라이브러리 설치
```bash
# 기본 LangChain 라이브러리
pip install langchain langchain-openai langchain-chroma
pip install langsmith  # 평가 추적 및 관리

# 추가 평가 도구
pip install pandas numpy matplotlib
pip install scikit-learn  # 메트릭 계산용
```

### 환경 변수 설정
```python
# .env 파일
OPENAI_API_KEY=your_openai_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=LLM_Judge_Evaluation
```

### 기본 설정
```python
import os
from dotenv import load_dotenv
import warnings
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

# 환경 변수 로드
load_dotenv()
warnings.filterwarnings("ignore")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.evaluation import load_evaluator

# 기본 LLM 설정 (평가용)
judge_llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.1,  # 일관된 평가를 위해 낮은 temperature
    max_tokens=1000
)
```

## 💻 단계별 구현

### 1단계: Reference-free 평가 시스템

```python
class ReferenceFreeJudge:
    def __init__(self, llm, evaluation_criteria: List[str] = None):
        self.llm = llm
        self.criteria = evaluation_criteria or [
            "정확성", "명확성", "완전성", "유용성"
        ]

    def create_evaluation_prompt(self) -> ChatPromptTemplate:
        """평가 프롬프트 생성"""
        criteria_text = "\n".join([f"- {criterion}" for criterion in self.criteria])

        template = """당신은 텍스트 품질을 평가하는 전문가입니다.
주어진 질문과 답변을 다음 기준으로 평가해주세요:

{criteria}

각 기준에 대해 1-5점으로 점수를 매기고, 간단한 설명을 제공해주세요.

[질문]
{question}

[답변]
{answer}

[평가 결과]
다음 형식으로 응답해주세요:

정확성: X/5 - 설명
명확성: X/5 - 설명
완전성: X/5 - 설명
유용성: X/5 - 설명

전체 점수: X/20
종합 의견: (간단한 평가 의견)
"""

        return ChatPromptTemplate.from_template(template)

    def evaluate(self, question: str, answer: str) -> Dict[str, Any]:
        """단일 답변 평가"""
        prompt = self.create_evaluation_prompt()

        criteria_text = "\n".join([f"- {criterion}" for criterion in self.criteria])

        chain = prompt | self.llm | StrOutputParser()

        try:
            evaluation_result = chain.invoke({
                "criteria": criteria_text,
                "question": question,
                "answer": answer
            })

            # 결과 파싱
            parsed_result = self._parse_evaluation(evaluation_result)
            parsed_result.update({
                "question": question,
                "answer": answer,
                "raw_evaluation": evaluation_result,
                "success": True
            })

            return parsed_result

        except Exception as e:
            return {
                "question": question,
                "answer": answer,
                "error": str(e),
                "success": False,
                "total_score": 0.0
            }

    def _parse_evaluation(self, evaluation_text: str) -> Dict[str, Any]:
        """평가 결과 파싱"""
        lines = evaluation_text.split('\n')
        scores = {}
        total_score = 0.0
        overall_comment = ""

        for line in lines:
            line = line.strip()
            if ':' in line:
                if '/' in line and any(criterion in line for criterion in self.criteria):
                    # 점수 라인 파싱
                    parts = line.split(':')
                    if len(parts) >= 2:
                        criterion = parts[0].strip()
                        score_part = parts[1].strip()
                        try:
                            if '/' in score_part:
                                score = float(score_part.split('/')[0])
                                scores[criterion] = score
                        except ValueError:
                            continue
                elif "전체 점수" in line or "총점" in line:
                    try:
                        score_part = line.split(':')[1].strip()
                        if '/' in score_part:
                            total_score = float(score_part.split('/')[0])
                    except (ValueError, IndexError):
                        continue
                elif "종합 의견" in line:
                    try:
                        overall_comment = line.split(':')[1].strip()
                    except IndexError:
                        continue

        return {
            "individual_scores": scores,
            "total_score": total_score,
            "max_score": len(self.criteria) * 5,
            "overall_comment": overall_comment
        }

# 사용 예시
judge = ReferenceFreeJudge(judge_llm)

question = "테슬라의 CEO는 누구인가요?"
answer = "테슬라의 CEO는 일론 머스크입니다. 그는 2008년부터 테슬라를 이끌고 있으며, 전기차 산업의 혁신을 주도하고 있습니다."

evaluation_result = judge.evaluate(question, answer)
print("Reference-free 평가 결과:")
print(f"총점: {evaluation_result['total_score']}/{evaluation_result['max_score']}")
print(f"개별 점수: {evaluation_result['individual_scores']}")
print(f"종합 의견: {evaluation_result['overall_comment']}")
```

### 2단계: Reference-based 평가 시스템

```python
class ReferenceBasedJudge:
    def __init__(self, llm):
        self.llm = llm

    def create_comparison_prompt(self) -> ChatPromptTemplate:
        """참조 답안 비교 프롬프트 생성"""
        template = """당신은 텍스트 품질을 평가하는 전문가입니다.
주어진 질문에 대한 참조 답안과 생성된 답변을 비교하여 평가해주세요.

[질문]
{question}

[참조 답안]
{reference_answer}

[생성된 답변]
{generated_answer}

다음 기준으로 생성된 답변을 평가해주세요:

1. 정확성: 참조 답안과 비교한 사실적 정확도 (1-5점)
2. 완전성: 참조 답안의 핵심 내용을 얼마나 포함하는가 (1-5점)
3. 유사성: 참조 답안과의 의미적 유사성 (1-5점)
4. 부가가치: 참조 답안에 없는 유용한 정보 제공 여부 (1-5점)

[평가 결과]
정확성: X/5 - 설명
완전성: X/5 - 설명
유사성: X/5 - 설명
부가가치: X/5 - 설명

전체 점수: X/20
비교 의견: (참조 답안 대비 생성 답변의 장단점)
"""

        return ChatPromptTemplate.from_template(template)

    def evaluate_with_reference(self, question: str, reference_answer: str,
                               generated_answer: str) -> Dict[str, Any]:
        """참조 답안과 비교 평가"""
        prompt = self.create_comparison_prompt()
        chain = prompt | self.llm | StrOutputParser()

        try:
            evaluation_result = chain.invoke({
                "question": question,
                "reference_answer": reference_answer,
                "generated_answer": generated_answer
            })

            # 결과 파싱 (Reference-free와 유사한 방식)
            parsed_result = self._parse_comparison_evaluation(evaluation_result)
            parsed_result.update({
                "question": question,
                "reference_answer": reference_answer,
                "generated_answer": generated_answer,
                "raw_evaluation": evaluation_result,
                "success": True
            })

            return parsed_result

        except Exception as e:
            return {
                "question": question,
                "reference_answer": reference_answer,
                "generated_answer": generated_answer,
                "error": str(e),
                "success": False,
                "total_score": 0.0
            }

    def _parse_comparison_evaluation(self, evaluation_text: str) -> Dict[str, Any]:
        """비교 평가 결과 파싱"""
        lines = evaluation_text.split('\n')
        scores = {}
        total_score = 0.0
        comparison_comment = ""

        criteria = ["정확성", "완전성", "유사성", "부가가치"]

        for line in lines:
            line = line.strip()
            if ':' in line:
                if '/' in line and any(criterion in line for criterion in criteria):
                    parts = line.split(':')
                    if len(parts) >= 2:
                        criterion = parts[0].strip()
                        score_part = parts[1].strip()
                        try:
                            if '/' in score_part:
                                score = float(score_part.split('/')[0])
                                scores[criterion] = score
                        except ValueError:
                            continue
                elif "전체 점수" in line or "총점" in line:
                    try:
                        score_part = line.split(':')[1].strip()
                        if '/' in score_part:
                            total_score = float(score_part.split('/')[0])
                    except (ValueError, IndexError):
                        continue
                elif "비교 의견" in line:
                    try:
                        comparison_comment = line.split(':')[1].strip()
                    except IndexError:
                        continue

        return {
            "individual_scores": scores,
            "total_score": total_score,
            "max_score": 20,
            "comparison_comment": comparison_comment
        }

# 사용 예시
ref_judge = ReferenceBasedJudge(judge_llm)

reference = "테슬라의 CEO는 일론 머스크입니다."
generated = "일론 머스크가 테슬라의 최고경영자로 활동하고 있습니다."

comparison_result = ref_judge.evaluate_with_reference(question, reference, generated)
print("\nReference-based 평가 결과:")
print(f"총점: {comparison_result['total_score']}/{comparison_result['max_score']}")
print(f"개별 점수: {comparison_result['individual_scores']}")
print(f"비교 의견: {comparison_result['comparison_comment']}")
```

### 3단계: Pairwise 비교 평가 시스템

```python
class PairwiseJudge:
    def __init__(self, llm):
        self.llm = llm

    def create_pairwise_prompt(self) -> ChatPromptTemplate:
        """쌍별 비교 프롬프트 생성"""
        template = """당신은 두 답변의 품질을 비교하는 전문가입니다.
주어진 질문에 대한 두 답변 중 어느 것이 더 나은지 평가해주세요.

[질문]
{question}

[답변 A]
{answer_a}

[답변 B]
{answer_b}

다음 기준으로 비교해주세요:
1. 정확성: 어느 답변이 더 정확한가?
2. 완전성: 어느 답변이 더 완전한가?
3. 명확성: 어느 답변이 더 명확한가?
4. 유용성: 어느 답변이 더 유용한가?

[비교 결과]
우수한 답변: A 또는 B 또는 동등함
이유: (구체적인 근거 제시)

점수 차이: X점 (1-5점 척도에서 차이)
각 기준별 분석:
- 정확성: A가 우수/B가 우수/동등함 - 이유
- 완전성: A가 우수/B가 우수/동등함 - 이유
- 명확성: A가 우수/B가 우수/동등함 - 이유
- 유용성: A가 우수/B가 우수/동등함 - 이유
"""

        return ChatPromptTemplate.from_template(template)

    def compare_answers(self, question: str, answer_a: str,
                       answer_b: str) -> Dict[str, Any]:
        """두 답변 비교 평가"""
        prompt = self.create_pairwise_prompt()
        chain = prompt | self.llm | StrOutputParser()

        try:
            comparison_result = chain.invoke({
                "question": question,
                "answer_a": answer_a,
                "answer_b": answer_b
            })

            parsed_result = self._parse_pairwise_evaluation(comparison_result)
            parsed_result.update({
                "question": question,
                "answer_a": answer_a,
                "answer_b": answer_b,
                "raw_comparison": comparison_result,
                "success": True
            })

            return parsed_result

        except Exception as e:
            return {
                "question": question,
                "answer_a": answer_a,
                "answer_b": answer_b,
                "error": str(e),
                "success": False,
                "winner": "error"
            }

    def _parse_pairwise_evaluation(self, evaluation_text: str) -> Dict[str, Any]:
        """쌍별 비교 결과 파싱"""
        lines = evaluation_text.split('\n')
        winner = "동등함"
        reasoning = ""
        score_difference = 0
        criteria_analysis = {}

        for line in lines:
            line = line.strip()
            if "우수한 답변:" in line:
                try:
                    winner = line.split(':')[1].strip()
                except IndexError:
                    continue
            elif "이유:" in line:
                try:
                    reasoning = line.split(':')[1].strip()
                except IndexError:
                    continue
            elif "점수 차이:" in line:
                try:
                    score_part = line.split(':')[1].strip()
                    score_difference = float(score_part.split('점')[0])
                except (ValueError, IndexError):
                    continue
            elif any(criterion in line for criterion in ["정확성", "완전성", "명확성", "유용성"]):
                if ':' in line:
                    try:
                        criterion = line.split(':')[0].strip()
                        analysis = line.split(':')[1].strip()
                        criteria_analysis[criterion] = analysis
                    except IndexError:
                        continue

        return {
            "winner": winner,
            "reasoning": reasoning,
            "score_difference": score_difference,
            "criteria_analysis": criteria_analysis
        }

# 사용 예시
pairwise_judge = PairwiseJudge(judge_llm)

answer_a = "테슬라의 CEO는 일론 머스크입니다."
answer_b = "일론 머스크가 테슬라의 최고경영자입니다. 그는 2008년부터 테슬라를 이끌고 있습니다."

pairwise_result = pairwise_judge.compare_answers(question, answer_a, answer_b)
print("\nPairwise 비교 결과:")
print(f"우승자: {pairwise_result['winner']}")
print(f"이유: {pairwise_result['reasoning']}")
print(f"점수 차이: {pairwise_result['score_difference']}")
```

### 4단계: 종합 평가 시스템

```python
class ComprehensiveJudge:
    def __init__(self, llm):
        self.llm = llm
        self.reference_free_judge = ReferenceFreeJudge(llm)
        self.reference_based_judge = ReferenceBasedJudge(llm)
        self.pairwise_judge = PairwiseJudge(llm)

    def comprehensive_evaluation(self, question: str, generated_answer: str,
                               reference_answer: str = None,
                               comparison_answer: str = None) -> Dict[str, Any]:
        """종합적인 평가 수행"""
        results = {
            "question": question,
            "generated_answer": generated_answer,
            "timestamp": pd.Timestamp.now()
        }

        # 1. Reference-free 평가
        ref_free_result = self.reference_free_judge.evaluate(question, generated_answer)
        results["reference_free"] = ref_free_result

        # 2. Reference-based 평가 (참조 답안이 있는 경우)
        if reference_answer:
            ref_based_result = self.reference_based_judge.evaluate_with_reference(
                question, reference_answer, generated_answer
            )
            results["reference_based"] = ref_based_result

        # 3. Pairwise 비교 (비교 답변이 있는 경우)
        if comparison_answer:
            pairwise_result = self.pairwise_judge.compare_answers(
                question, generated_answer, comparison_answer
            )
            results["pairwise"] = pairwise_result

        # 4. 종합 점수 계산
        overall_score = self._calculate_overall_score(results)
        results["overall_assessment"] = overall_score

        return results

    def _calculate_overall_score(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """종합 점수 계산"""
        scores = []
        weights = []

        # Reference-free 점수
        if "reference_free" in results and results["reference_free"]["success"]:
            ref_free_score = results["reference_free"]["total_score"] / results["reference_free"]["max_score"]
            scores.append(ref_free_score)
            weights.append(0.4)  # 40% 가중치

        # Reference-based 점수
        if "reference_based" in results and results["reference_based"]["success"]:
            ref_based_score = results["reference_based"]["total_score"] / results["reference_based"]["max_score"]
            scores.append(ref_based_score)
            weights.append(0.6)  # 60% 가중치 (참조 답안이 있을 때 더 신뢰)

        # 종합 점수 계산 (가중 평균)
        if scores:
            overall_score = sum(score * weight for score, weight in zip(scores, weights)) / sum(weights)
        else:
            overall_score = 0.0

        # 등급 부여
        if overall_score >= 0.9:
            grade = "A+"
        elif overall_score >= 0.8:
            grade = "A"
        elif overall_score >= 0.7:
            grade = "B"
        elif overall_score >= 0.6:
            grade = "C"
        else:
            grade = "D"

        return {
            "overall_score": overall_score,
            "grade": grade,
            "individual_scores": scores,
            "weights_used": weights
        }

# 사용 예시
comprehensive_judge = ComprehensiveJudge(judge_llm)

# 종합 평가 실행
comprehensive_result = comprehensive_judge.comprehensive_evaluation(
    question=question,
    generated_answer=answer,
    reference_answer=reference,
    comparison_answer="테슬라의 CEO는 일론 머스크입니다."
)

print("\n=== 종합 평가 결과 ===")
print(f"전체 점수: {comprehensive_result['overall_assessment']['overall_score']:.3f}")
print(f"등급: {comprehensive_result['overall_assessment']['grade']}")

if "reference_free" in comprehensive_result:
    print(f"Reference-free 점수: {comprehensive_result['reference_free']['total_score']}/20")

if "reference_based" in comprehensive_result:
    print(f"Reference-based 점수: {comprehensive_result['reference_based']['total_score']}/20")
```

## 🎯 실습 문제

### 기초 실습
1. **단순 평가기 구현**
   - 하나의 평가 기준(예: 정확성)으로 답변을 평가하는 간단한 시스템 구현
   - 5개의 질문-답변 쌍으로 테스트

2. **평가 기준 커스터마이징**
   - 본인만의 평가 기준을 3개 정의하고 평가 시스템에 적용
   - 각 기준의 가중치를 다르게 설정

### 응용 실습
3. **다중 모델 품질 비교**
   - OpenAI, Gemini, Ollama 모델의 답변을 LLM-as-Judge로 평가
   - 어떤 모델이 어떤 기준에서 우수한지 분석

4. **평가자 간 일치도 분석**
   - 같은 답변을 서로 다른 평가 프롬프트로 평가
   - 평가 결과의 일관성과 신뢰성 분석

### 심화 실습
5. **자동 평가 파이프라인**
   - 배치로 여러 답변을 평가하고 결과를 데이터베이스에 저장
   - 시간별/모델별 성능 트렌드 분석 대시보드 구현

## ✅ 솔루션 예시

### 실습 1: 배치 평가 시스템
```python
def batch_llm_evaluation(questions: List[str], answers: List[str],
                        references: List[str] = None) -> pd.DataFrame:
    """배치로 LLM 평가 수행"""
    judge = ComprehensiveJudge(judge_llm)
    results = []

    for i, (question, answer) in enumerate(zip(questions, answers)):
        print(f"평가 중... {i+1}/{len(questions)}")

        reference = references[i] if references else None

        evaluation = judge.comprehensive_evaluation(
            question=question,
            generated_answer=answer,
            reference_answer=reference
        )

        # 결과를 플랫한 구조로 변환
        flat_result = {
            'question_id': i,
            'question': question,
            'answer': answer,
            'reference': reference,
            'overall_score': evaluation['overall_assessment']['overall_score'],
            'grade': evaluation['overall_assessment']['grade'],
        }

        # Reference-free 결과 추가
        if 'reference_free' in evaluation and evaluation['reference_free']['success']:
            ref_free = evaluation['reference_free']
            flat_result.update({
                'ref_free_total': ref_free['total_score'],
                'ref_free_max': ref_free['max_score'],
                'ref_free_comment': ref_free['overall_comment']
            })

        # Reference-based 결과 추가
        if 'reference_based' in evaluation and evaluation['reference_based']['success']:
            ref_based = evaluation['reference_based']
            flat_result.update({
                'ref_based_total': ref_based['total_score'],
                'ref_based_max': ref_based['max_score'],
                'ref_based_comment': ref_based['comparison_comment']
            })

        results.append(flat_result)

    return pd.DataFrame(results)

# 테스트 데이터 준비
test_questions = [
    "테슬라의 CEO는 누구인가요?",
    "전기차의 주요 장점은 무엇인가요?",
    "자율주행 기술의 현재 수준은?"
]

test_answers = [
    "테슬라의 CEO는 일론 머스크입니다. 그는 2008년부터 테슬라를 이끌고 있습니다.",
    "전기차는 환경친화적이고 연료비가 절약되며 조용합니다.",
    "자율주행 기술은 현재 레벨 2-3 단계에 있으며 지속 발전하고 있습니다."
]

test_references = [
    "테슬라의 CEO는 일론 머스크입니다.",
    "전기차의 장점으로는 환경 보호, 경제성, 정숙성 등이 있습니다.",
    "자율주행 기술은 레벨 2-3 수준으로 부분 자동화가 가능합니다."
]

# 배치 평가 실행
batch_results = batch_llm_evaluation(test_questions, test_answers, test_references)
print("배치 평가 완료!")
print(batch_results[['question_id', 'overall_score', 'grade']].head())
```

### 실습 2: 평가자 신뢰성 분석
```python
class EvaluatorReliabilityAnalyzer:
    def __init__(self):
        self.evaluation_history = []

    def test_evaluator_consistency(self, question: str, answer: str,
                                 num_trials: int = 5) -> Dict[str, Any]:
        """평가자 일관성 테스트"""
        judge = ReferenceFreeJudge(judge_llm)
        trial_results = []

        for i in range(num_trials):
            print(f"평가 시행 {i+1}/{num_trials}")
            result = judge.evaluate(question, answer)
            if result['success']:
                trial_results.append(result)

        if not trial_results:
            return {"error": "모든 평가가 실패했습니다."}

        # 점수 분석
        scores = [r['total_score'] for r in trial_results]

        analysis = {
            "num_trials": len(trial_results),
            "scores": scores,
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "coefficient_of_variation": np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0,
            "consistency_rating": self._rate_consistency(np.std(scores))
        }

        return analysis

    def _rate_consistency(self, std_dev: float) -> str:
        """일관성 등급 부여"""
        if std_dev < 0.5:
            return "매우 일관됨"
        elif std_dev < 1.0:
            return "일관됨"
        elif std_dev < 2.0:
            return "보통"
        else:
            return "불일치함"

# 신뢰성 분석 실행
analyzer = EvaluatorReliabilityAnalyzer()
reliability_result = analyzer.test_evaluator_consistency(
    question="테슬라의 CEO는 누구인가요?",
    answer="테슬라의 CEO는 일론 머스크입니다.",
    num_trials=5
)

print("평가자 신뢰성 분석 결과:")
print(f"평균 점수: {reliability_result['mean_score']:.2f}")
print(f"표준편차: {reliability_result['std_score']:.2f}")
print(f"일관성 등급: {reliability_result['consistency_rating']}")
```

## 🚀 실무 활용 예시

### 1. 실시간 품질 모니터링 시스템

```python
import asyncio
from datetime import datetime, timedelta
import json

class RealTimeQualityMonitor:
    def __init__(self, judge_llm, quality_threshold: float = 0.7):
        self.judge = ComprehensiveJudge(judge_llm)
        self.quality_threshold = quality_threshold
        self.monitoring_data = []
        self.alerts = []

    async def monitor_response(self, question: str, answer: str,
                             reference: str = None) -> Dict[str, Any]:
        """실시간 응답 품질 모니터링"""
        start_time = datetime.now()

        # 평가 실행
        evaluation = self.judge.comprehensive_evaluation(
            question=question,
            generated_answer=answer,
            reference_answer=reference
        )

        end_time = datetime.now()
        evaluation_time = (end_time - start_time).total_seconds()

        # 모니터링 데이터 저장
        monitoring_record = {
            "timestamp": start_time,
            "question": question,
            "answer": answer,
            "reference": reference,
            "overall_score": evaluation["overall_assessment"]["overall_score"],
            "grade": evaluation["overall_assessment"]["grade"],
            "evaluation_time": evaluation_time
        }

        self.monitoring_data.append(monitoring_record)

        # 품질 임계값 체크
        if evaluation["overall_assessment"]["overall_score"] < self.quality_threshold:
            alert = {
                "timestamp": start_time,
                "alert_type": "LOW_QUALITY",
                "score": evaluation["overall_assessment"]["overall_score"],
                "threshold": self.quality_threshold,
                "question": question[:100] + "..." if len(question) > 100 else question
            }
            self.alerts.append(alert)

        return monitoring_record

    def get_quality_report(self, hours: int = 24) -> Dict[str, Any]:
        """품질 리포트 생성"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_data = [
            record for record in self.monitoring_data
            if record["timestamp"] > cutoff_time
        ]

        if not recent_data:
            return {"message": f"최근 {hours}시간 데이터가 없습니다."}

        scores = [record["overall_score"] for record in recent_data]

        report = {
            "period": f"최근 {hours}시간",
            "total_evaluations": len(recent_data),
            "average_score": np.mean(scores),
            "score_std": np.std(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "quality_pass_rate": sum(1 for score in scores if score >= self.quality_threshold) / len(scores),
            "alerts_count": len([
                alert for alert in self.alerts
                if alert["timestamp"] > cutoff_time
            ]),
            "grade_distribution": self._calculate_grade_distribution(recent_data)
        }

        return report

    def _calculate_grade_distribution(self, data: List[Dict]) -> Dict[str, int]:
        """등급별 분포 계산"""
        grade_counts = {}
        for record in data:
            grade = record["grade"]
            grade_counts[grade] = grade_counts.get(grade, 0) + 1
        return grade_counts

# 실시간 모니터링 시스템 사용 예시
monitor = RealTimeQualityMonitor(judge_llm, quality_threshold=0.75)

# 실시간 모니터링 (비동기 실행)
async def simulate_real_time_monitoring():
    test_cases = [
        ("테슬라의 CEO는?", "일론 머스크입니다.", "테슬라의 CEO는 일론 머스크입니다."),
        ("전기차 장점은?", "환경친화적입니다.", "전기차는 환경친화적이고 경제적입니다."),
        ("잘못된 질문", "모르겠습니다.", "정확한 정보를 제공해드리겠습니다.")
    ]

    for question, answer, reference in test_cases:
        result = await monitor.monitor_response(question, answer, reference)
        print(f"모니터링: {result['grade']} (점수: {result['overall_score']:.3f})")

        # 1초 대기 (실제로는 실시간 처리)
        await asyncio.sleep(1)

# 실행
# asyncio.run(simulate_real_time_monitoring())
```

### 2. 다중 평가자 투표 시스템

```python
class MultiJudgeVotingSystem:
    def __init__(self, judge_llms: List, voting_method: str = "majority"):
        self.judges = [ReferenceFreeJudge(llm) for llm in judge_llms]
        self.voting_method = voting_method

    def evaluate_with_voting(self, question: str, answer: str) -> Dict[str, Any]:
        """다중 평가자 투표 시스템"""
        all_evaluations = []

        for i, judge in enumerate(self.judges):
            print(f"평가자 {i+1} 평가 중...")
            evaluation = judge.evaluate(question, answer)

            if evaluation['success']:
                evaluation['judge_id'] = i
                all_evaluations.append(evaluation)

        if not all_evaluations:
            return {"error": "모든 평가자의 평가가 실패했습니다."}

        # 투표 결과 계산
        if self.voting_method == "majority":
            final_result = self._majority_voting(all_evaluations)
        elif self.voting_method == "average":
            final_result = self._average_voting(all_evaluations)
        elif self.voting_method == "weighted":
            final_result = self._weighted_voting(all_evaluations)
        else:
            final_result = self._average_voting(all_evaluations)  # 기본값

        final_result.update({
            "question": question,
            "answer": answer,
            "num_judges": len(all_evaluations),
            "voting_method": self.voting_method,
            "individual_evaluations": all_evaluations
        })

        return final_result

    def _majority_voting(self, evaluations: List[Dict]) -> Dict[str, Any]:
        """다수결 투표"""
        # 등급으로 변환하여 다수결
        grades = []
        for eval_result in evaluations:
            score = eval_result['total_score'] / eval_result['max_score']
            if score >= 0.8:
                grades.append("우수")
            elif score >= 0.6:
                grades.append("양호")
            else:
                grades.append("보완필요")

        from collections import Counter
        grade_counts = Counter(grades)
        majority_grade = grade_counts.most_common(1)[0][0]

        # 다수결 등급에 해당하는 평가들의 평균 계산
        selected_evaluations = [
            eval_result for eval_result, grade in zip(evaluations, grades)
            if grade == majority_grade
        ]

        avg_score = np.mean([e['total_score'] for e in selected_evaluations])

        return {
            "voting_result": "majority",
            "final_grade": majority_grade,
            "final_score": avg_score,
            "vote_distribution": dict(grade_counts),
            "confidence": grade_counts.most_common(1)[0][1] / len(grades)
        }

    def _average_voting(self, evaluations: List[Dict]) -> Dict[str, Any]:
        """평균 투표"""
        scores = [eval_result['total_score'] for eval_result in evaluations]
        max_scores = [eval_result['max_score'] for eval_result in evaluations]

        avg_score = np.mean(scores)
        avg_max_score = np.mean(max_scores)
        normalized_score = avg_score / avg_max_score

        # 등급 부여
        if normalized_score >= 0.8:
            final_grade = "우수"
        elif normalized_score >= 0.6:
            final_grade = "양호"
        else:
            final_grade = "보완필요"

        return {
            "voting_result": "average",
            "final_grade": final_grade,
            "final_score": avg_score,
            "normalized_score": normalized_score,
            "score_std": np.std(scores),
            "agreement_level": 1 - (np.std(scores) / np.mean(scores)) if np.mean(scores) > 0 else 0
        }

    def _weighted_voting(self, evaluations: List[Dict]) -> Dict[str, Any]:
        """가중 투표 (신뢰도 기반)"""
        # 각 평가자의 신뢰도 계산 (여기서는 간단히 일관성 기반)
        weights = []
        for evaluation in evaluations:
            # 개별 점수들의 분산이 낮을수록 높은 가중치
            individual_scores = list(evaluation.get('individual_scores', {}).values())
            if individual_scores:
                consistency = 1 / (1 + np.std(individual_scores))
            else:
                consistency = 1.0
            weights.append(consistency)

        # 가중치 정규화
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # 가중 평균 계산
        weighted_score = sum(
            eval_result['total_score'] * weight
            for eval_result, weight in zip(evaluations, weights)
        )

        weighted_max_score = sum(
            eval_result['max_score'] * weight
            for eval_result, weight in zip(evaluations, weights)
        )

        normalized_score = weighted_score / weighted_max_score

        if normalized_score >= 0.8:
            final_grade = "우수"
        elif normalized_score >= 0.6:
            final_grade = "양호"
        else:
            final_grade = "보완필요"

        return {
            "voting_result": "weighted",
            "final_grade": final_grade,
            "final_score": weighted_score,
            "normalized_score": normalized_score,
            "weights_used": weights,
            "confidence": max(weights)  # 가장 높은 가중치를 신뢰도로 사용
        }

# 다중 평가자 시스템 사용 예시
# 여러 LLM 모델을 평가자로 사용
judge_models = [
    ChatOpenAI(model="gpt-4.1-mini", temperature=0.1),
    ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1),
    ChatOpenAI(model="gpt-4.1-mini", temperature=0.2)  # 다른 설정
]

voting_system = MultiJudgeVotingSystem(judge_models, voting_method="average")

voting_result = voting_system.evaluate_with_voting(
    question="테슬라의 CEO는 누구인가요?",
    answer="테슬라의 CEO는 일론 머스크입니다. 그는 혁신적인 기업가로 평가받고 있습니다."
)

print("다중 평가자 투표 결과:")
print(f"최종 등급: {voting_result['final_grade']}")
print(f"최종 점수: {voting_result['final_score']:.2f}")
print(f"평가자 수: {voting_result['num_judges']}")
print(f"합의 수준: {voting_result.get('agreement_level', 'N/A'):.3f}")
```

## 📖 참고 자료

### LLM-as-Judge 관련 연구
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
- [LLM-as-a-Judge: A Comprehensive Study](https://arxiv.org/abs/2403.16950)
- [Can Large Language Models Be Good Judges?](https://arxiv.org/abs/2308.02312)

### LangChain 평가 도구
- [LangChain Evaluation Documentation](https://python.langchain.com/docs/guides/evaluation/)
- [LangSmith Evaluation Guide](https://docs.smith.langchain.com/evaluation)
- [Custom Evaluator 구현 가이드](https://python.langchain.com/docs/guides/evaluation/string/custom)

### 프롬프트 엔지니어링
- [Prompt Engineering for LLM-as-Judge](https://platform.openai.com/docs/guides/prompt-engineering)
- [Constitutional AI와 AI 피드백](https://www.anthropic.com/index/constitutional-ai-harmlessness-from-ai-feedback)

### 추가 학습 자료
- [자동 평가 시스템 설계 원칙](https://research.google/pubs/pub48671/)
- [RAG 시스템 품질 평가 방법론](https://python.langchain.com/docs/use_cases/question_answering/evaluation/)
- [다중 에이전트 평가 시스템](https://python.langchain.com/docs/use_cases/agent_simulations/)

이 가이드를 통해 LLM-as-Judge 시스템을 구축하여 텍스트 생성 품질을 자동으로 평가할 수 있는 능력을 기르시기 바랍니다. 실무에서는 여러 평가 방식을 조합하여 더욱 신뢰할 수 있는 품질 평가 시스템을 구축하는 것이 중요합니다.