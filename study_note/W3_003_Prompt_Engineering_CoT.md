# W3_003_Prompt_Engineering_CoT.md - Chain of Thought 프롬프트 엔지니어링

## 🎯 학습 목표

- Chain-of-Thought (CoT) 추론 기법을 활용한 복잡한 문제 해결 방법을 학습합니다
- Zero-shot, One-shot, CoT 프롬프팅 방식의 차이점과 적용 시나리오를 이해합니다
- Self-Consistency, Program-Aided Language (PAL), Reflexion 등 고급 추론 기법을 실습합니다
- 다양한 AI 모델에서의 추론 성능을 비교하고 최적화 전략을 개발합니다
- 복잡한 논리 문제와 수학 문제 해결에 CoT를 효과적으로 활용할 수 있는 능력을 개발합니다

## 📚 핵심 개념

### 1. Chain of Thought (CoT)란?

Chain of Thought는 AI 모델이 복잡한 문제를 해결할 때 각 단계별 사고 과정을 명시적으로 보여주도록 하는 프롬프팅 기법입니다. 이를 통해 모델의 추론 과정을 투명하게 확인할 수 있고 더 정확한 결과를 도출할 수 있습니다.

#### CoT의 핵심 특징
```python
cot_structure = {
    "transparency": "추론 과정의 명시적 표현",
    "accuracy": "단계별 검증을 통한 정확도 향상",
    "debuggability": "오류 발생 지점의 명확한 식별",
    "interpretability": "결과에 대한 이해도 증진"
}
```

#### CoT vs 기존 방식 비교
```python
comparison = {
    "direct_answer": {
        "input": "2 + 3 × 4 = ?",
        "output": "14",
        "pros": ["빠른 응답", "간결함"],
        "cons": ["과정 불투명", "오류 검증 어려움"]
    },
    "chain_of_thought": {
        "input": "2 + 3 × 4 = ?",
        "output": """
        1단계: 연산 순서 확인 (곱셈 우선)
        2단계: 3 × 4 = 12 계산
        3단계: 2 + 12 = 14 계산
        답: 14
        """,
        "pros": ["투명한 과정", "오류 검증 가능", "학습 효과"],
        "cons": ["긴 응답", "높은 토큰 사용량"]
    }
}
```

### 2. 프롬프팅 기법의 진화

#### 2.1 Zero-Shot 프롬프팅
예시 없이 직접적인 지시만으로 답을 구하는 방식입니다.

**특징:**
- 가장 단순한 형태
- 빠른 응답 속도
- 메모리 효율적
- 단순한 문제에 적합

**적용 사례:**
```python
zero_shot_examples = {
    "translation": "다음을 영어로 번역하세요: 안녕하세요",
    "classification": "다음 감정을 분류하세요: 오늘 정말 기분이 좋아요",
    "simple_math": "15 × 8은 얼마입니까?"
}
```

#### 2.2 One-Shot/Few-Shot 프롬프팅
하나 이상의 예시를 통해 문제 해결 방식을 학습시키는 방식입니다.

**특징:**
- 예시를 통한 패턴 학습
- Zero-shot보다 향상된 성능
- 중간 복잡도 문제에 효과적
- 형식화된 출력에 유리

**구조:**
```python
few_shot_structure = """
예시 1:
문제: [문제 설명]
풀이: [단계별 해결 과정]
답: [결과]

예시 2:
문제: [문제 설명]
풀이: [단계별 해결 과정]
답: [결과]

새로운 문제: {user_question}
풀이:
"""
```

#### 2.3 Chain of Thought (CoT) 프롬프팅
체계적이고 명시적인 단계별 추론을 통해 문제를 해결하는 방식입니다.

**특징:**
- 가장 체계적인 문제 해결
- 복잡한 추론에 최적화
- 높은 정확도
- 과정 검증 가능

**템플릿 구조:**
```python
cot_template = """
문제: {problem}

해결 과정:
1단계: 문제 이해
- 주어진 정보 파악
- 구해야 할 것 정리

2단계: 해결 전략 계획
- 사용할 방법 선택
- 단계별 계획 수립

3단계: 계획 실행
- 각 단계 순차 실행
- 중간 결과 확인

4단계: 결과 검증
- 답 확인
- 대안 방법 검토

답: [최종 결과]
"""
```

### 3. 고급 CoT 기법들

#### 3.1 Self-Consistency
하나의 문제를 여러 방식으로 해결하여 결과의 일관성을 확인하는 기법입니다.

**원리:**
- 다양한 접근법으로 동일한 문제 해결
- 결과 간 일관성 검증
- 최종 답안의 신뢰도 향상

**적용 방법:**
```python
self_consistency_approach = {
    "method_1": "직접 계산",
    "method_2": "비율 활용",
    "method_3": "단계별 분해",
    "verification": "세 방법의 결과 일치성 확인",
    "confidence": "일치하는 답안의 신뢰도 증가"
}
```

#### 3.2 Program-Aided Language (PAL)
자연어 문제를 프로그래밍적 사고로 접근하는 기법입니다.

**특징:**
- 코드 구조를 활용한 논리적 해결
- 실행 가능한 의사코드 생성
- 검증과 디버깅 용이
- 수학적 계산에 특히 효과적

**구조:**
```python
pal_structure = """
def solve_problem():
    # 1. 변수 정의
    [주어진 값들을 변수로 저장]

    # 2. 계산 과정
    [단계별 계산 수행]

    # 3. 결과 반환
    [최종 결과 계산 및 반환]

    return result
"""
```

#### 3.3 Reflexion
자기 평가와 개선을 통한 메타인지적 접근 기법입니다.

**과정:**
1. 초기 답안 생성
2. 자체 평가 수행
3. 개선점 식별
4. 수정된 답안 작성

**장점:**
- 지속적인 품질 개선
- 오류 자가 발견
- 메타인지적 사고 촉진

## 🔧 환경 설정

### 1. 필수 라이브러리 설치

```bash
# uv 사용 (권장)
uv add langchain langchain-openai langchain-ollama python-dotenv

# pip 사용
pip install langchain langchain-openai langchain-ollama python-dotenv
```

### 2. 환경 변수 설정

```bash
# .env 파일에 추가
OPENAI_API_KEY=your_openai_api_key_here
OLLAMA_HOST=http://localhost:11434  # Ollama 서버 주소 (선택사항)
```

### 3. 기본 임포트

```python
import os
from dotenv import load_dotenv
from pprint import pprint
from typing import Dict, List, Any, Optional, Tuple

# LangChain 관련
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

# 환경 변수 로드
load_dotenv()

# LLM 초기화
openai_llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.3,
    top_p=0.9
)

# Ollama LLMs (옵션)
phi3_llm = ChatOllama(
    model="phi3:mini",
    temperature=0.3,
    top_p=0.9
)

gemma_llm = ChatOllama(
    model="gemma2",
    temperature=0.3,
    top_p=0.9
)
```

## 💻 코드 예제

### 1. 프롬프팅 기법 비교 시스템

```python
class PromptingMethodComparator:
    def __init__(self, models: Dict[str, Any]):
        """프롬프팅 방법 비교 시스템"""
        self.models = models

    def create_zero_shot_prompt(self, question: str) -> PromptTemplate:
        """Zero-shot 프롬프트 생성"""
        template = """
다음 문제를 해결하시오:

문제: {question}

답안:
"""
        return PromptTemplate(
            input_variables=["question"],
            template=template
        )

    def create_one_shot_prompt(self, question: str) -> PromptTemplate:
        """One-shot 프롬프트 생성"""
        template = """
다음은 수학 문제를 해결하는 예시입니다:

예시 문제: 한 학급에 30명의 학생이 있습니다. 이 중 40%가 남학생이라면, 여학생은 몇 명인가요?

예시 풀이:
1) 먼저 남학생 수를 계산합니다:
   - 전체 학생의 40% = 30 × 0.4 = 12명이 남학생

2) 여학생 수를 계산합니다:
   - 전체 학생 수 - 남학생 수 = 30 - 12 = 18명이 여학생

따라서 여학생은 18명입니다.

이제 아래 문제를 같은 방식으로 해결하시오:

새로운 문제: {question}

답안:
"""
        return PromptTemplate(
            input_variables=["question"],
            template=template
        )

    def create_cot_prompt(self, question: str) -> PromptTemplate:
        """Chain of Thought 프롬프트 생성"""
        template = """
다음 문제를 논리적 단계에 따라 해결하시오:

문제: {question}

해결 과정:
1단계: 문제 이해하기
- 주어진 정보 파악
- 구해야 할 것 정리

2단계: 해결 방법 계획
- 사용할 수 있는 전략 검토
- 최적의 방법 선택

3단계: 계획 실행
- 선택한 방법 적용
- 중간 결과 확인

4단계: 검토
- 답안 확인
- 다른 방법 가능성 검토

답안:
"""
        return PromptTemplate(
            input_variables=["question"],
            template=template
        )

    def compare_methods(self, question: str, model_name: str = "openai") -> Dict[str, Any]:
        """세 가지 방법 성능 비교"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")

        model = self.models[model_name]
        results = {}

        # Zero-shot 테스트
        zero_shot_prompt = self.create_zero_shot_prompt(question)
        zero_shot_chain = zero_shot_prompt | model | StrOutputParser()
        results['zero_shot'] = zero_shot_chain.invoke({"question": question})

        # One-shot 테스트
        one_shot_prompt = self.create_one_shot_prompt(question)
        one_shot_chain = one_shot_prompt | model | StrOutputParser()
        results['one_shot'] = one_shot_chain.invoke({"question": question})

        # CoT 테스트
        cot_prompt = self.create_cot_prompt(question)
        cot_chain = cot_prompt | model | StrOutputParser()
        results['cot'] = cot_chain.invoke({"question": question})

        return results

    def evaluate_response_quality(self, results: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        """응답 품질 평가"""
        evaluation = {}

        for method, response in results.items():
            evaluation[method] = {
                "length": len(response),
                "has_calculation": any(char in response for char in "×÷+-="),
                "step_by_step": "단계" in response or "step" in response.lower(),
                "shows_reasoning": len(response.split('\n')) > 3,
                "confidence_score": self._calculate_confidence(response)
            }

        return evaluation

    def _calculate_confidence(self, response: str) -> float:
        """응답 신뢰도 계산 (간단한 휴리스틱)"""
        indicators = [
            "따라서" in response,
            "계산" in response,
            "=" in response,
            len(response) > 100,  # 충분한 설명
            "단계" in response or "step" in response.lower()
        ]
        return sum(indicators) / len(indicators)

# 사용 예시
models = {
    "openai": openai_llm,
    "phi3": phi3_llm,
    "gemma": gemma_llm
}

comparator = PromptingMethodComparator(models)

# 테스트 문제
test_question = """
학교에서 500명의 학생이 있습니다. 이 중 30%는 5학년이고, 20%는 6학년 학생입니다.
5학년 학생들 중 60%는 수학 동아리에 있고, 나머지는 과학 동아리에 있습니다.
6학년 학생들 중 70%는 수학 동아리에 있고, 나머지는 과학 동아리에 있습니다.
과학 동아리에는 몇 명의 학생이 있나요?
"""

# 비교 실행
comparison_results = comparator.compare_methods(test_question, "openai")
quality_evaluation = comparator.evaluate_response_quality(comparison_results)

print("=== 프롬프팅 방법 비교 결과 ===")
for method, result in comparison_results.items():
    print(f"\n【{method.upper()}】")
    print(result[:300] + "..." if len(result) > 300 else result)

    quality = quality_evaluation[method]
    print(f"평가: 길이={quality['length']}, 추론과정={quality['shows_reasoning']}, 신뢰도={quality['confidence_score']:.2f}")
```

### 2. Self-Consistency 시스템

```python
class SelfConsistencySystem:
    def __init__(self, model: Any, num_attempts: int = 3):
        """Self-Consistency 시스템"""
        self.model = model
        self.num_attempts = num_attempts

    def create_multi_method_prompt(self, question: str) -> PromptTemplate:
        """다중 방법 프롬프트 생성"""
        template = """
다음 문제를 세 가지 다른 방법으로 해결하시오:

문제: {question}

세 가지 풀이 방법:
1) 직접 계산 방법:
   - 주어진 숫자를 직접 계산

2) 비율 활용 방법:
   - 전체에 대한 비율로 계산

3) 단계별 분해 방법:
   - 문제를 작은 부분으로 나누어 계산

각 방법의 답안을 제시하고, 결과가 일치하는지 확인하시오.

답안:
"""
        return PromptTemplate(
            input_variables=["question"],
            template=template
        )

    def execute_self_consistency(self, question: str) -> Dict[str, Any]:
        """Self-Consistency 실행"""
        prompt = self.create_multi_method_prompt(question)
        chain = prompt | self.model | StrOutputParser()

        # 여러 번 실행하여 일관성 확인
        responses = []
        for i in range(self.num_attempts):
            response = chain.invoke({"question": question})
            responses.append(response)

        # 일관성 분석
        consistency_analysis = self._analyze_consistency(responses)

        return {
            "question": question,
            "responses": responses,
            "consistency_analysis": consistency_analysis,
            "final_answer": self._extract_final_answer(responses[0])  # 첫 번째 응답에서 답 추출
        }

    def _analyze_consistency(self, responses: List[str]) -> Dict[str, Any]:
        """일관성 분석"""
        # 간단한 키워드 기반 분석
        answers = []
        for response in responses:
            answer = self._extract_final_answer(response)
            answers.append(answer)

        # 가장 빈번한 답 찾기
        from collections import Counter
        answer_counts = Counter(answers)
        most_common = answer_counts.most_common(1)[0] if answer_counts else ("", 0)

        consistency_score = most_common[1] / len(answers) if answers else 0

        return {
            "all_answers": answers,
            "most_common_answer": most_common[0],
            "consistency_score": consistency_score,
            "agreement_level": "높음" if consistency_score >= 0.8 else "보통" if consistency_score >= 0.6 else "낮음"
        }

    def _extract_final_answer(self, response: str) -> str:
        """응답에서 최종 답 추출"""
        import re

        # 숫자 패턴 찾기
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response)

        # "답:", "따라서", "결론" 등의 키워드 근처에서 숫자 찾기
        answer_indicators = ["답:", "따라서", "결론", "최종", "총"]

        for indicator in answer_indicators:
            if indicator in response:
                # 해당 부분 이후의 첫 번째 숫자 찾기
                parts = response.split(indicator, 1)
                if len(parts) > 1:
                    after_indicator = parts[1]
                    found_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', after_indicator[:100])  # 처음 100자 내에서
                    if found_numbers:
                        return found_numbers[0]

        # 마지막 숫자 반환 (fallback)
        return numbers[-1] if numbers else "추출 실패"

# 사용 예시
self_consistency = SelfConsistencySystem(openai_llm, num_attempts=2)

consistency_result = self_consistency.execute_self_consistency(test_question)

print("=== Self-Consistency 결과 ===")
print(f"질문: {consistency_result['question']}")
print(f"최종 답: {consistency_result['final_answer']}")
print(f"일관성 점수: {consistency_result['consistency_analysis']['consistency_score']:.2f}")
print(f"일치도: {consistency_result['consistency_analysis']['agreement_level']}")

print("\n=== 세부 응답 ===")
for i, response in enumerate(consistency_result['responses'], 1):
    print(f"\n응답 {i}:")
    print(response[:400] + "..." if len(response) > 400 else response)
```

### 3. Program-Aided Language (PAL) 시스템

```python
class PALSystem:
    def __init__(self, model: Any):
        """Program-Aided Language 시스템"""
        self.model = model

    def create_pal_prompt(self, question: str) -> PromptTemplate:
        """PAL 프롬프트 생성"""
        template = """
다음 문제를 Python 프로그래밍 방식으로 해결하시오:

문제: {question}

# 문제 해결을 위한 Python 스타일 의사코드:
def solve_problem():
    # 1. 변수 정의
    # - 주어진 값들을 변수로 저장

    # 2. 계산 과정
    # - 필요한 계산을 단계별로 수행

    # 3. 결과 반환
    # - 최종 결과 계산 및 반환

    return result

# 실행 및 결과 출력
print(solve_problem())

답안:
"""
        return PromptTemplate(
            input_variables=["question"],
            template=template
        )

    def execute_pal(self, question: str) -> Dict[str, Any]:
        """PAL 실행"""
        prompt = self.create_pal_prompt(question)
        chain = prompt | self.model | StrOutputParser()

        response = chain.invoke({"question": question})

        # 코드 추출 및 분석
        code_analysis = self._analyze_code(response)

        return {
            "question": question,
            "response": response,
            "code_analysis": code_analysis
        }

    def _analyze_code(self, response: str) -> Dict[str, Any]:
        """생성된 코드 분석"""
        import re

        # Python 코드 블록 찾기
        code_blocks = re.findall(r'```python\n(.*?)\n```', response, re.DOTALL)
        if not code_blocks:
            # 코드 블록이 없으면 def로 시작하는 부분 찾기
            def_pattern = r'def\s+\w+.*?(?=\n[^\s]|$)'
            code_blocks = re.findall(def_pattern, response, re.DOTALL)

        analysis = {
            "has_function_definition": "def " in response,
            "has_variables": any(keyword in response for keyword in ["=", "total", "count", "num"]),
            "has_calculations": any(op in response for op in ["*", "+", "-", "/", "%"]),
            "has_return_statement": "return" in response,
            "code_blocks_found": len(code_blocks),
            "executable": self._check_executability(code_blocks[0] if code_blocks else "")
        }

        return analysis

    def _check_executability(self, code: str) -> bool:
        """코드 실행 가능성 검사 (간단한 구문 검사)"""
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False

    def create_logic_puzzle_pal_prompt(self, question: str) -> PromptTemplate:
        """논리 퍼즐용 PAL 프롬프트"""
        template = """
다음 논리 문제를 Python 프로그래밍 방식으로 해결하시오:

문제: {question}

# 해결 접근법:
def solve_puzzle():
    # 1. 상태 정의
    # - 현재 상태를 변수로 표현

    # 2. 제약조건 함수
    # - 안전성 검사 함수 정의

    # 3. 이동 시뮬레이션
    # - 가능한 이동을 순차적으로 시뮬레이션

    # 4. 최적 해 탐색
    # - 최소 이동 횟수 계산

    return moves_count

답안:
"""
        return PromptTemplate(
            input_variables=["question"],
            template=template
        )

# 사용 예시
pal_system = PALSystem(openai_llm)

# 수학 문제 테스트
math_result = pal_system.execute_pal(test_question)

print("=== PAL 결과 (수학 문제) ===")
print("생성된 코드:")
print(math_result['response'])
print("\n코드 분석:")
analysis = math_result['code_analysis']
for key, value in analysis.items():
    print(f"- {key}: {value}")

# 논리 퍼즐 테스트
puzzle_question = """
농부가 늑대, 양, 양배추를 데리고 강을 건너야 합니다.
제약조건:
1. 농부가 없을 때 늑대와 양이 같이 있으면 늑대가 양을 잡아먹습니다
2. 농부가 없을 때 양과 양배추가 같이 있으면 양이 양배추를 먹어버립니다
3. 보트에는 농부와 한 물건만 실을 수 있습니다
모두 안전하게 건너는데 몇 번 이동이 필요할까요?
"""

puzzle_prompt = pal_system.create_logic_puzzle_pal_prompt(puzzle_question)
puzzle_chain = puzzle_prompt | openai_llm | StrOutputParser()
puzzle_result = puzzle_chain.invoke({"question": puzzle_question})

print(f"\n=== PAL 결과 (논리 퍼즐) ===")
print(puzzle_result)
```

### 4. Reflexion 시스템

```python
class ReflexionSystem:
    def __init__(self, model: Any):
        """Reflexion 시스템"""
        self.model = model

    def create_reflexion_prompt(self, question: str) -> PromptTemplate:
        """Reflexion 프롬프트 생성"""
        template = """
다음 문제에 대해 단계적으로 해결하여 초기 답안을 작성하고, 자체 평가 후 개선하시오:

문제: {question}

1단계: 초기 답안
---
[여기에 첫 번째 답안 작성]

2단계: 자체 평가
---
- 정확성 검토: 계산과 논리가 올바른가?
- 논리적 오류 확인: 추론 과정에 빈틈은 없는가?
- 설명의 명확성 평가: 이해하기 쉽게 설명했는가?
- 개선이 필요한 부분 식별: 어떤 부분을 보완해야 하는가?

3단계: 개선된 답안
---
[평가를 바탕으로 개선된 답안 작성]

답안:
"""
        return PromptTemplate(
            input_variables=["question"],
            template=template
        )

    def execute_reflexion(self, question: str) -> Dict[str, Any]:
        """Reflexion 실행"""
        prompt = self.create_reflexion_prompt(question)
        chain = prompt | self.model | StrOutputParser()

        response = chain.invoke({"question": question})

        # 응답 분석
        reflection_analysis = self._analyze_reflection(response)

        return {
            "question": question,
            "full_response": response,
            "reflection_analysis": reflection_analysis
        }

    def _analyze_reflection(self, response: str) -> Dict[str, Any]:
        """Reflexion 응답 분석"""
        stages = ["1단계", "2단계", "3단계"]
        stage_content = {}

        for i, stage in enumerate(stages):
            if stage in response:
                # 현재 단계와 다음 단계 사이의 내용 추출
                start_idx = response.find(stage)
                if i < len(stages) - 1:
                    next_stage_idx = response.find(stages[i + 1])
                    if next_stage_idx != -1:
                        stage_content[stage] = response[start_idx:next_stage_idx].strip()
                    else:
                        stage_content[stage] = response[start_idx:].strip()
                else:
                    stage_content[stage] = response[start_idx:].strip()

        analysis = {
            "has_initial_answer": "1단계" in response,
            "has_self_evaluation": "2단계" in response,
            "has_improved_answer": "3단계" in response,
            "stage_content": stage_content,
            "improvement_indicators": self._find_improvement_indicators(response),
            "self_correction_score": self._calculate_self_correction_score(response)
        }

        return analysis

    def _find_improvement_indicators(self, response: str) -> List[str]:
        """개선 지표 찾기"""
        indicators = []
        improvement_keywords = [
            "수정", "보완", "개선", "정정", "추가", "명확히", "더 정확하게", "구체적으로"
        ]

        for keyword in improvement_keywords:
            if keyword in response:
                indicators.append(keyword)

        return indicators

    def _calculate_self_correction_score(self, response: str) -> float:
        """자기 교정 점수 계산"""
        score_factors = [
            "오류" in response or "틀린" in response,  # 오류 인식
            "수정" in response or "개선" in response,  # 수정 의도
            len(response.split("3단계")) > 1,  # 개선된 답안 존재
            "더 정확" in response or "더 명확" in response,  # 품질 향상 언급
            "검토" in response or "확인" in response  # 검증 과정
        ]

        return sum(score_factors) / len(score_factors)

    def create_iterative_reflexion_prompt(self, question: str, previous_response: str) -> PromptTemplate:
        """반복적 Reflexion 프롬프트"""
        template = """
이전 답안을 검토하고 추가로 개선해보세요:

원래 문제: {question}

이전 답안:
{previous_response}

추가 개선사항:
1. 이전 답안의 강점과 약점 분석
2. 놓친 부분이나 추가 고려사항 식별
3. 더 나은 설명 방식이나 접근법 제안
4. 최종 개선된 답안 제시

개선된 답안:
"""
        return PromptTemplate(
            input_variables=["question", "previous_response"],
            template=template
        )

# 사용 예시
reflexion_system = ReflexionSystem(openai_llm)

# 기본 Reflexion 실행
reflexion_result = reflexion_system.execute_reflexion(test_question)

print("=== Reflexion 결과 ===")
print("전체 응답:")
print(reflexion_result['full_response'])

print(f"\n=== 분석 결과 ===")
analysis = reflexion_result['reflection_analysis']
print(f"초기 답안 존재: {analysis['has_initial_answer']}")
print(f"자체 평가 존재: {analysis['has_self_evaluation']}")
print(f"개선된 답안 존재: {analysis['has_improved_answer']}")
print(f"자기 교정 점수: {analysis['self_correction_score']:.2f}")
print(f"개선 지표: {', '.join(analysis['improvement_indicators'])}")

# 반복적 개선 (옵션)
if reflexion_result['reflection_analysis']['self_correction_score'] < 0.8:
    print("\n=== 추가 개선 수행 ===")
    iterative_prompt = reflexion_system.create_iterative_reflexion_prompt(
        test_question,
        reflexion_result['full_response']
    )
    iterative_chain = iterative_prompt | openai_llm | StrOutputParser()
    improved_response = iterative_chain.invoke({
        "question": test_question,
        "previous_response": reflexion_result['full_response']
    })
    print("개선된 답안:")
    print(improved_response)
```

### 5. 통합 CoT 벤치마킹 시스템

```python
class CoTBenchmarkingSystem:
    def __init__(self, models: Dict[str, Any]):
        """CoT 벤치마킹 시스템"""
        self.models = models
        self.methods = {
            "zero_shot": self._create_zero_shot,
            "one_shot": self._create_one_shot,
            "cot": self._create_cot,
            "self_consistency": self._create_self_consistency,
            "pal": self._create_pal,
            "reflexion": self._create_reflexion
        }

    def _create_zero_shot(self, question: str) -> str:
        """Zero-shot 프롬프트"""
        return f"다음 문제를 해결하시오:\n\n문제: {question}\n\n답안:"

    def _create_one_shot(self, question: str) -> str:
        """One-shot 프롬프트"""
        return f"""
다음은 수학 문제 해결 예시입니다:

예시: 30명 중 40%가 남학생이면 여학생은?
풀이: 30 × 0.4 = 12명(남학생), 30 - 12 = 18명(여학생)

문제: {question}
답안:
"""

    def _create_cot(self, question: str) -> str:
        """CoT 프롬프트"""
        return f"""
다음 문제를 단계별로 해결하시오:

문제: {question}

1단계: 문제 이해 및 정보 정리
2단계: 해결 전략 계획
3단계: 단계별 계산 실행
4단계: 결과 검증

답안:
"""

    def _create_self_consistency(self, question: str) -> str:
        """Self-Consistency 프롬프트"""
        return f"""
다음 문제를 세 가지 방법으로 해결하고 결과를 비교하시오:

문제: {question}

방법1: 직접 계산
방법2: 비율 활용
방법3: 단계별 분해

각 방법의 결과가 일치하는지 확인하시오.

답안:
"""

    def _create_pal(self, question: str) -> str:
        """PAL 프롬프트"""
        return f"""
다음 문제를 Python 코드로 해결하시오:

문제: {question}

def solve():
    # 변수 정의 및 계산 과정
    return result

답안:
"""

    def _create_reflexion(self, question: str) -> str:
        """Reflexion 프롬프트"""
        return f"""
다음 문제를 해결하고 자체 평가 후 개선하시오:

문제: {question}

1단계: 초기 답안
2단계: 자체 평가 (정확성, 논리성, 명확성)
3단계: 개선된 답안

답안:
"""

    def benchmark_single_question(
        self,
        question: str,
        correct_answer: str = None,
        model_name: str = "openai"
    ) -> Dict[str, Any]:
        """단일 문제 벤치마킹"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")

        model = self.models[model_name]
        results = {}

        for method_name, method_func in self.methods.items():
            try:
                prompt_text = method_func(question)

                # 간단한 프롬프트 템플릿 사용
                from langchain_core.prompts import PromptTemplate
                prompt = PromptTemplate(
                    input_variables=[],
                    template=prompt_text
                )

                chain = prompt | model | StrOutputParser()
                response = chain.invoke({})

                # 응답 분석
                analysis = self._analyze_response(response, correct_answer, method_name)

                results[method_name] = {
                    "response": response,
                    "analysis": analysis
                }

            except Exception as e:
                results[method_name] = {
                    "response": f"Error: {str(e)}",
                    "analysis": {"error": True}
                }

        return results

    def _analyze_response(
        self,
        response: str,
        correct_answer: str = None,
        method: str = None
    ) -> Dict[str, Any]:
        """응답 분석"""
        analysis = {
            "length": len(response),
            "has_numbers": bool(re.findall(r'\d+', response)),
            "has_calculation": any(op in response for op in ["×", "÷", "+", "-", "=", "*", "/"]),
            "step_count": len([line for line in response.split('\n') if line.strip() and any(c.isdigit() for c in line)]),
            "shows_process": "단계" in response or "step" in response.lower(),
            "confidence_indicators": sum([
                "따라서" in response,
                "결론" in response,
                "답:" in response,
                "=" in response
            ])
        }

        # 정답과 비교 (제공된 경우)
        if correct_answer:
            extracted_answer = self._extract_answer(response)
            analysis["extracted_answer"] = extracted_answer
            analysis["is_correct"] = extracted_answer == correct_answer

        return analysis

    def _extract_answer(self, response: str) -> str:
        """응답에서 답 추출"""
        import re

        # 일반적인 답 패턴들
        patterns = [
            r'답[:：]\s*(\d+)',
            r'결론[:：]\s*(\d+)',
            r'따라서\s*(\d+)',
            r'총\s*(\d+)',
            r'=\s*(\d+)',
            r'(\d+)명',
            r'(\d+)개'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                return matches[-1]  # 마지막 매치 반환

        # 숫자만 추출 (fallback)
        numbers = re.findall(r'\d+', response)
        return numbers[-1] if numbers else "추출실패"

    def run_comprehensive_benchmark(
        self,
        test_cases: List[Dict[str, str]],
        model_names: List[str] = None
    ) -> Dict[str, Any]:
        """종합 벤치마킹"""
        if model_names is None:
            model_names = list(self.models.keys())

        all_results = {}

        for model_name in model_names:
            print(f"\n=== 모델: {model_name} 벤치마킹 시작 ===")
            model_results = {}

            for i, test_case in enumerate(test_cases):
                question = test_case["question"]
                correct_answer = test_case.get("correct_answer")

                print(f"문제 {i+1} 처리 중...")

                results = self.benchmark_single_question(
                    question,
                    correct_answer,
                    model_name
                )
                model_results[f"question_{i+1}"] = results

            all_results[model_name] = model_results

        # 통계 생성
        stats = self._generate_statistics(all_results)

        return {
            "detailed_results": all_results,
            "statistics": stats
        }

    def _generate_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """통계 생성"""
        stats = {}

        for model_name, model_results in results.items():
            model_stats = {
                "method_performance": {},
                "overall_metrics": {}
            }

            # 방법별 성능
            for method in self.methods.keys():
                method_scores = []
                for question_key, question_results in model_results.items():
                    if method in question_results and "analysis" in question_results[method]:
                        analysis = question_results[method]["analysis"]
                        if not analysis.get("error", False):
                            # 간단한 점수 계산
                            score = 0
                            if analysis.get("has_calculation", False):
                                score += 1
                            if analysis.get("shows_process", False):
                                score += 1
                            if analysis.get("confidence_indicators", 0) > 0:
                                score += 1
                            if analysis.get("is_correct", False):
                                score += 2  # 정답일 경우 가산점

                            method_scores.append(score / 5.0)  # 0-1 정규화

                if method_scores:
                    model_stats["method_performance"][method] = {
                        "average_score": sum(method_scores) / len(method_scores),
                        "question_count": len(method_scores)
                    }

            stats[model_name] = model_stats

        return stats

# 사용 예시
models = {
    "openai": openai_llm,
    "phi3": phi3_llm if 'phi3_llm' in globals() else None,
    "gemma": gemma_llm if 'gemma_llm' in globals() else None
}

# None 값 제거
models = {k: v for k, v in models.items() if v is not None}

benchmark_system = CoTBenchmarkingSystem(models)

# 테스트 케이스
test_cases = [
    {
        "question": "학교에서 500명의 학생이 있습니다. 이 중 30%는 5학년이고, 20%는 6학년 학생입니다. 5학년 학생들 중 60%는 수학 동아리에 있고, 나머지는 과학 동아리에 있습니다. 6학년 학생들 중 70%는 수학 동아리에 있고, 나머지는 과학 동아리에 있습니다. 과학 동아리에는 몇 명의 학생이 있나요?",
        "correct_answer": "90"
    },
    {
        "question": "한 상자에 빨간 공 15개와 파란 공 25개가 있습니다. 전체 공의 40%를 뽑았을 때, 뽑은 공 중 빨간 공이 6개였다면 파란 공은 몇 개를 뽑았나요?",
        "correct_answer": "10"
    }
]

# 벤치마킹 실행 (OpenAI 모델만 사용)
benchmark_results = benchmark_system.run_comprehensive_benchmark(
    test_cases,
    model_names=["openai"]
)

# 결과 출력
print("\n=== 벤치마킹 결과 요약 ===")
stats = benchmark_results["statistics"]

for model_name, model_stats in stats.items():
    print(f"\n【{model_name.upper()} 모델 성능】")
    method_perf = model_stats["method_performance"]

    # 방법별 점수 정렬
    sorted_methods = sorted(
        method_perf.items(),
        key=lambda x: x[1]["average_score"],
        reverse=True
    )

    for method, perf in sorted_methods:
        print(f"{method:15}: {perf['average_score']:.3f} (테스트 {perf['question_count']}개)")
```

## 🚀 실습해보기

### 실습 1: 논리 퍼즐 해결 시스템

농부와 늑대, 양, 양배추 문제를 다양한 CoT 기법으로 해결해보세요.

```python
# 논리 퍼즐 해결 시스템 구현
class LogicPuzzleSolver:
    def __init__(self):
        # 초기화 코드 작성
        pass

    def solve_with_cot(self, puzzle_description):
        # CoT 방식으로 퍼즐 해결
        pass

    def solve_with_pal(self, puzzle_description):
        # PAL 방식으로 퍼즐 해결
        pass

    def compare_approaches(self, puzzle_description):
        # 여러 접근 방식 비교
        pass
```

### 실습 2: 수학 문제 자동 해결 시스템

복잡한 수학 문제를 단계별로 해결하는 시스템을 구현해보세요.

```python
# 수학 문제 자동 해결 시스템 구현
class MathProblemSolver:
    def __init__(self):
        # 초기화 코드 작성
        pass

    def analyze_problem_type(self, problem):
        # 문제 유형 분석
        pass

    def generate_solution_strategy(self, problem, problem_type):
        # 해결 전략 생성
        pass

    def execute_with_self_consistency(self, problem):
        # Self-Consistency 적용
        pass
```

### 실습 3: 멀티모달 추론 시스템

텍스트와 수식을 함께 처리하는 추론 시스템을 구현해보세요.

```python
# 멀티모달 추론 시스템 구현
class MultimodalReasoningSystem:
    def __init__(self):
        # 초기화 코드 작성
        pass

    def parse_mathematical_expressions(self, text):
        # 수식 파싱
        pass

    def create_visual_explanation(self, solution_steps):
        # 시각적 설명 생성
        pass

    def integrate_reasoning_modes(self, problem):
        # 추론 모드 통합
        pass
```

## 📋 해답

### 실습 1: 논리 퍼즐 해결 시스템

```python
class LogicPuzzleSolver:
    def __init__(self, models: Dict[str, Any]):
        """논리 퍼즐 해결 시스템"""
        self.models = models

    def solve_with_cot(self, puzzle_description: str, model_name: str = "openai") -> Dict[str, Any]:
        """CoT 방식으로 퍼즐 해결"""
        cot_template = """
다음 논리 퍼즐을 단계별로 해결하시오:

문제: {puzzle}

해결 과정:
1단계: 문제 상황 이해
- 등장 인물/물체 파악
- 제약 조건 정리
- 목표 상태 설정

2단계: 해결 전략 계획
- 가능한 접근 방법 검토
- 최적 해결 순서 계획

3단계: 단계별 해결 실행
- 각 이동 단계 시뮬레이션
- 제약 조건 준수 확인
- 중간 상태 검증

4단계: 결과 검증 및 최적화
- 해결 완료 확인
- 최소 이동 횟수 검증
- 다른 해법 가능성 검토

답안:
"""

        prompt = PromptTemplate(
            input_variables=["puzzle"],
            template=cot_template
        )

        model = self.models[model_name]
        chain = prompt | model | StrOutputParser()

        response = chain.invoke({"puzzle": puzzle_description})

        # 해법 분석
        analysis = self._analyze_solution(response)

        return {
            "method": "Chain of Thought",
            "response": response,
            "analysis": analysis
        }

    def solve_with_pal(self, puzzle_description: str, model_name: str = "openai") -> Dict[str, Any]:
        """PAL 방식으로 퍼즐 해결"""
        pal_template = """
다음 논리 퍼즐을 Python 프로그래밍 방식으로 해결하시오:

문제: {puzzle}

# 프로그래밍적 해결 접근:
def solve_river_crossing():
    # 상태 표현: (farmer, wolf, sheep, cabbage) - 0: 왼쪽, 1: 오른쪽
    initial_state = (0, 0, 0, 0)  # 모두 왼쪽 시작
    target_state = (1, 1, 1, 1)   # 모두 오른쪽 목표

    def is_safe_state(state):
        # 안전성 검사 함수
        farmer, wolf, sheep, cabbage = state

        # 농부가 없는 쪽에서 위험한 조합 체크
        if farmer == 0:  # 농부가 왼쪽에 있을 때, 오른쪽 상태 체크
            if wolf == 1 and sheep == 1:  # 늑대와 양이 같이
                return False
            if sheep == 1 and cabbage == 1:  # 양과 양배추가 같이
                return False
        else:  # 농부가 오른쪽에 있을 때, 왼쪽 상태 체크
            if wolf == 0 and sheep == 0:
                return False
            if sheep == 0 and cabbage == 0:
                return False
        return True

    def get_possible_moves(state):
        # 가능한 이동 생성
        moves = []
        farmer, wolf, sheep, cabbage = state

        # 농부 혼자 이동
        new_farmer = 1 - farmer
        new_state = (new_farmer, wolf, sheep, cabbage)
        if is_safe_state(new_state):
            moves.append(("farmer", new_state))

        # 농부 + 늑대
        if wolf == farmer:
            new_state = (new_farmer, new_farmer, sheep, cabbage)
            if is_safe_state(new_state):
                moves.append(("farmer_wolf", new_state))

        # 농부 + 양
        if sheep == farmer:
            new_state = (new_farmer, wolf, new_farmer, cabbage)
            if is_safe_state(new_state):
                moves.append(("farmer_sheep", new_state))

        # 농부 + 양배추
        if cabbage == farmer:
            new_state = (new_farmer, wolf, sheep, new_farmer)
            if is_safe_state(new_state):
                moves.append(("farmer_cabbage", new_state))

        return moves

    def solve_bfs():
        # BFS로 최단 경로 탐색
        from collections import deque

        queue = deque([(initial_state, [])])
        visited = {initial_state}

        while queue:
            current_state, path = queue.popleft()

            if current_state == target_state:
                return path

            for move_name, next_state in get_possible_moves(current_state):
                if next_state not in visited:
                    visited.add(next_state)
                    new_path = path + [(move_name, next_state)]
                    queue.append((next_state, new_path))

        return None  # 해가 없는 경우

    # 해결 실행
    solution_path = solve_bfs()

    if solution_path:
        print(f"최소 이동 횟수: {len(solution_path)}")
        print("이동 순서:")
        for i, (move, state) in enumerate(solution_path, 1):
            print(f"{i}. {move}: {state}")
        return len(solution_path)
    else:
        return -1  # 해가 없음

# 실행
result = solve_river_crossing()
print(f"답: {result}번 이동")

답안:
"""

        prompt = PromptTemplate(
            input_variables=["puzzle"],
            template=pal_template
        )

        model = self.models[model_name]
        chain = prompt | model | StrOutputParser()

        response = chain.invoke({"puzzle": puzzle_description})

        # 코드 분석
        code_analysis = self._analyze_generated_code(response)

        return {
            "method": "Program-Aided Language",
            "response": response,
            "code_analysis": code_analysis
        }

    def solve_with_reflexion(self, puzzle_description: str, model_name: str = "openai") -> Dict[str, Any]:
        """Reflexion 방식으로 퍼즐 해결"""
        reflexion_template = """
다음 논리 퍼즐을 해결하고 자체 검토를 통해 개선하시오:

문제: {puzzle}

1단계: 초기 해결 시도
---
[첫 번째 해결 과정과 답안]

2단계: 자체 평가
---
- 논리적 정확성: 각 단계가 제약조건을 만족하는가?
- 완전성: 모든 요소가 최종 목표에 도달했는가?
- 최적성: 더 적은 이동으로 해결 가능한가?
- 검증: 각 단계에서 안전성이 보장되는가?

3단계: 개선된 해결책
---
[평가를 바탕으로 개선된 답안]

답안:
"""

        prompt = PromptTemplate(
            input_variables=["puzzle"],
            template=reflexion_template
        )

        model = self.models[model_name]
        chain = prompt | model | StrOutputParser()

        response = chain.invoke({"puzzle": puzzle_description})

        # Reflexion 분석
        reflexion_analysis = self._analyze_reflexion_quality(response)

        return {
            "method": "Reflexion",
            "response": response,
            "reflexion_analysis": reflexion_analysis
        }

    def compare_approaches(self, puzzle_description: str, model_name: str = "openai") -> Dict[str, Any]:
        """여러 접근 방식 비교"""
        results = {}

        # 각 방법으로 해결
        print("CoT 방식으로 해결 중...")
        results["cot"] = self.solve_with_cot(puzzle_description, model_name)

        print("PAL 방식으로 해결 중...")
        results["pal"] = self.solve_with_pal(puzzle_description, model_name)

        print("Reflexion 방식으로 해결 중...")
        results["reflexion"] = self.solve_with_reflexion(puzzle_description, model_name)

        # 비교 분석
        comparison = self._compare_solution_quality(results)

        return {
            "individual_results": results,
            "comparison_analysis": comparison
        }

    def _analyze_solution(self, response: str) -> Dict[str, Any]:
        """해법 분석"""
        import re

        # 이동 횟수 추출
        move_patterns = [
            r'(\d+)번 이동',
            r'(\d+)회 이동',
            r'총 (\d+)',
            r'최소 (\d+)'
        ]

        move_count = None
        for pattern in move_patterns:
            matches = re.findall(pattern, response)
            if matches:
                move_count = int(matches[0])
                break

        return {
            "extracted_move_count": move_count,
            "has_step_by_step": "단계" in response,
            "has_constraint_check": "제약" in response or "조건" in response,
            "has_verification": "검증" in response or "확인" in response,
            "response_length": len(response),
            "shows_reasoning": "따라서" in response or "그러므로" in response
        }

    def _analyze_generated_code(self, response: str) -> Dict[str, Any]:
        """생성된 코드 분석"""
        code_analysis = {
            "has_function_definition": "def " in response,
            "has_state_representation": "state" in response,
            "has_safety_check": "is_safe" in response or "safe" in response,
            "has_bfs_or_search": "bfs" in response.lower() or "search" in response,
            "has_queue_or_stack": "queue" in response or "stack" in response,
            "has_visited_tracking": "visited" in response,
            "code_complexity": response.count("def ") + response.count("if ") + response.count("for ")
        }

        return code_analysis

    def _analyze_reflexion_quality(self, response: str) -> Dict[str, Any]:
        """Reflexion 품질 분석"""
        stages = ["1단계", "2단계", "3단계"]
        stage_presence = {stage: stage in response for stage in stages}

        improvement_keywords = [
            "수정", "개선", "보완", "최적화", "더 나은", "효율적", "정확한"
        ]

        improvement_count = sum(1 for keyword in improvement_keywords if keyword in response)

        return {
            "all_stages_present": all(stage_presence.values()),
            "stage_presence": stage_presence,
            "improvement_mentions": improvement_count,
            "self_critique_quality": "평가" in response and "약점" in response,
            "iterative_improvement": improvement_count >= 2
        }

    def _compare_solution_quality(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """솔루션 품질 비교"""
        comparison = {
            "method_scores": {},
            "best_method": None,
            "consistency_check": {}
        }

        # 각 방법별 점수 계산
        for method, result in results.items():
            score = 0

            if method == "cot":
                analysis = result.get("analysis", {})
                score += 2 if analysis.get("has_step_by_step") else 0
                score += 1 if analysis.get("has_constraint_check") else 0
                score += 1 if analysis.get("shows_reasoning") else 0

            elif method == "pal":
                analysis = result.get("code_analysis", {})
                score += 3 if analysis.get("has_function_definition") else 0
                score += 2 if analysis.get("has_safety_check") else 0
                score += 2 if analysis.get("has_bfs_or_search") else 0

            elif method == "reflexion":
                analysis = result.get("reflexion_analysis", {})
                score += 3 if analysis.get("all_stages_present") else 0
                score += 2 if analysis.get("iterative_improvement") else 0
                score += 1 if analysis.get("self_critique_quality") else 0

            comparison["method_scores"][method] = score

        # 최고 방법 선정
        if comparison["method_scores"]:
            best_method = max(comparison["method_scores"], key=comparison["method_scores"].get)
            comparison["best_method"] = best_method

        # 답안 일관성 체크 (이동 횟수 기준)
        extracted_answers = {}
        for method, result in results.items():
            if method == "cot" and "analysis" in result:
                move_count = result["analysis"].get("extracted_move_count")
                if move_count:
                    extracted_answers[method] = move_count

        if len(set(extracted_answers.values())) <= 1:
            comparison["consistency_check"]["consistent"] = True
        else:
            comparison["consistency_check"]["consistent"] = False
            comparison["consistency_check"]["answers"] = extracted_answers

        return comparison

# 실습 1 테스트
models = {"openai": openai_llm}
puzzle_solver = LogicPuzzleSolver(models)

river_crossing_puzzle = """
농부가 늑대, 양, 양배추를 데리고 강을 건너야 합니다.

제약조건:
1. 농부가 없을 때 늑대와 양이 같이 있으면 늑대가 양을 잡아먹습니다
2. 농부가 없을 때 양과 양배추가 같이 있으면 양이 양배추를 먹어버립니다
3. 보트에는 농부와 한 물건만 실을 수 있습니다

모두 안전하게 건너는데 몇 번 이동이 필요할까요?
"""

print("=== 논리 퍼즐 해결 시스템 테스트 ===")
comparison_results = puzzle_solver.compare_approaches(river_crossing_puzzle)

# 결과 출력
for method, result in comparison_results["individual_results"].items():
    print(f"\n【{method.upper()} 방식 결과】")
    print(result["response"][:500] + "..." if len(result["response"]) > 500 else result["response"])

# 비교 분석 출력
print(f"\n=== 방법별 비교 분석 ===")
comparison_analysis = comparison_results["comparison_analysis"]
print("방법별 점수:")
for method, score in comparison_analysis["method_scores"].items():
    print(f"- {method}: {score}점")

if comparison_analysis["best_method"]:
    print(f"최고 성능 방법: {comparison_analysis['best_method']}")

consistency = comparison_analysis["consistency_check"]
if consistency.get("consistent", False):
    print("답안 일관성: 일치")
else:
    print(f"답안 일관성: 불일치 - {consistency.get('answers', {})}")
```

### 실습 2: 수학 문제 자동 해결 시스템

```python
class MathProblemSolver:
    def __init__(self, models: Dict[str, Any]):
        """수학 문제 자동 해결 시스템"""
        self.models = models
        self.problem_types = {
            "percentage": ["퍼센트", "%", "비율"],
            "arithmetic": ["더하기", "빼기", "곱하기", "나누기", "+", "-", "×", "÷"],
            "word_problem": ["학생", "명", "개", "마리", "그룹"],
            "geometry": ["넓이", "둘레", "부피", "각도", "삼각형", "사각형"],
            "algebra": ["방정식", "x", "y", "변수", "미지수"]
        }

    def analyze_problem_type(self, problem: str) -> Dict[str, Any]:
        """문제 유형 분석"""
        problem_lower = problem.lower()
        type_scores = {}

        for prob_type, keywords in self.problem_types.items():
            score = sum(1 for keyword in keywords if keyword in problem)
            if score > 0:
                type_scores[prob_type] = score

        # 숫자와 연산자 분석
        import re
        numbers = re.findall(r'\d+(?:\.\d+)?', problem)
        operators = re.findall(r'[+\-×÷%]', problem)

        analysis = {
            "detected_types": type_scores,
            "primary_type": max(type_scores, key=type_scores.get) if type_scores else "general",
            "numbers_found": len(numbers),
            "operators_found": len(operators),
            "complexity_score": len(numbers) + len(operators) + len(type_scores),
            "extracted_numbers": numbers
        }

        return analysis

    def generate_solution_strategy(self, problem: str, problem_type: str) -> str:
        """해결 전략 생성"""
        strategies = {
            "percentage": """
1단계: 전체 값과 퍼센트 값 식별
2단계: 퍼센트를 소수로 변환 (예: 30% → 0.3)
3단계: 전체 × 퍼센트 또는 부분 ÷ 퍼센트 계산
4단계: 결과 검증 및 단위 확인
""",
            "word_problem": """
1단계: 문제에서 주어진 정보 정리
2단계: 구해야 할 것 명확히 파악
3단계: 적절한 연산 방법 선택
4단계: 단계별 계산 수행
5단계: 답이 합리적인지 검토
""",
            "arithmetic": """
1단계: 연산 순서 확인 (괄호 → 곱셈/나눗셈 → 덧셈/뺄셈)
2단계: 각 연산을 순서대로 수행
3단계: 중간 결과 확인
4단계: 최종 답 검증
""",
            "geometry": """
1단계: 주어진 도형과 치수 파악
2단계: 적용할 공식 선택
3단계: 공식에 값 대입
4단계: 계산 수행 및 단위 확인
""",
            "general": """
1단계: 문제 상황 이해
2단계: 주어진 조건과 구할 것 파악
3단계: 해결 방법 계획
4단계: 단계별 실행
5단계: 결과 검증
"""
        }

        return strategies.get(problem_type, strategies["general"])

    def solve_with_strategy(self, problem: str, model_name: str = "openai") -> Dict[str, Any]:
        """전략 기반 문제 해결"""
        # 문제 유형 분석
        type_analysis = self.analyze_problem_type(problem)
        primary_type = type_analysis["primary_type"]

        # 해결 전략 생성
        strategy = self.generate_solution_strategy(problem, primary_type)

        # 전략적 프롬프트 템플릿
        strategic_template = """
다음 수학 문제를 분석된 유형과 전략에 따라 해결하시오:

문제: {problem}

문제 유형: {problem_type}
복잡도: {complexity}/10

해결 전략:
{strategy}

위 전략을 따라 단계별로 문제를 해결하시오:

답안:
"""

        prompt = PromptTemplate(
            input_variables=["problem", "problem_type", "complexity", "strategy"],
            template=strategic_template
        )

        model = self.models[model_name]
        chain = prompt | model | StrOutputParser()

        response = chain.invoke({
            "problem": problem,
            "problem_type": primary_type,
            "complexity": min(type_analysis["complexity_score"], 10),
            "strategy": strategy
        })

        return {
            "problem": problem,
            "type_analysis": type_analysis,
            "applied_strategy": strategy,
            "response": response
        }

    def execute_with_self_consistency(self, problem: str, model_name: str = "openai", num_attempts: int = 3) -> Dict[str, Any]:
        """Self-Consistency 적용"""
        consistency_template = """
다음 수학 문제를 세 가지 서로 다른 방법으로 해결하시오:

문제: {problem}

방법 1: 직접 계산 접근법
- 주어진 수치를 직접 사용하여 단계별로 계산

방법 2: 비례식 활용 접근법
- 비율과 비례 관계를 이용하여 해결

방법 3: 검산 중심 접근법
- 역산이나 다른 방법으로 검증하면서 해결

각 방법의 결과를 비교하고 일치하는지 확인하시오.

답안:
"""

        prompt = PromptTemplate(
            input_variables=["problem"],
            template=consistency_template
        )

        model = self.models[model_name]
        chain = prompt | model | StrOutputParser()

        responses = []
        for attempt in range(num_attempts):
            response = chain.invoke({"problem": problem})
            responses.append(response)

        # 일관성 분석
        consistency_analysis = self._analyze_math_consistency(responses)

        return {
            "problem": problem,
            "responses": responses,
            "consistency_analysis": consistency_analysis
        }

    def create_step_by_step_explanation(self, problem: str, solution: str, model_name: str = "openai") -> str:
        """단계별 설명 생성"""
        explanation_template = """
다음 수학 문제와 해답을 바탕으로 초보자도 이해할 수 있는 단계별 설명을 작성하시오:

문제: {problem}

해답: {solution}

요구사항:
1. 각 계산 단계를 명확히 구분
2. 왜 그 방법을 선택했는지 설명
3. 중간 계산 과정을 모두 표시
4. 최종 답이 올바른지 검증 과정 포함
5. 비슷한 문제를 풀 때 적용할 수 있는 일반적 원칙 제시

단계별 설명:
"""

        prompt = PromptTemplate(
            input_variables=["problem", "solution"],
            template=explanation_template
        )

        model = self.models[model_name]
        chain = prompt | model | StrOutputParser()

        return chain.invoke({
            "problem": problem,
            "solution": solution
        })

    def _analyze_math_consistency(self, responses: List[str]) -> Dict[str, Any]:
        """수학 문제 일관성 분석"""
        import re

        extracted_answers = []
        for response in responses:
            # 다양한 답 패턴으로 숫자 추출
            answer_patterns = [
                r'답[:：]\s*(\d+(?:\.\d+)?)',
                r'결과[:：]\s*(\d+(?:\.\d+)?)',
                r'최종.*?(\d+(?:\.\d+)?)',
                r'따라서.*?(\d+(?:\.\d+)?)',
                r'=\s*(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)명',
                r'(\d+(?:\.\d+)?)개'
            ]

            found_answer = None
            for pattern in answer_patterns:
                matches = re.findall(pattern, response)
                if matches:
                    found_answer = matches[-1]  # 마지막 매치 사용
                    break

            if found_answer:
                extracted_answers.append(float(found_answer))

        # 일관성 계산
        if extracted_answers:
            unique_answers = list(set(extracted_answers))
            consistency_score = extracted_answers.count(max(set(extracted_answers), key=extracted_answers.count)) / len(extracted_answers)

            return {
                "extracted_answers": extracted_answers,
                "unique_answers": unique_answers,
                "consistency_score": consistency_score,
                "most_common_answer": max(set(extracted_answers), key=extracted_answers.count),
                "is_consistent": len(unique_answers) == 1,
                "agreement_level": "높음" if consistency_score >= 0.8 else "보통" if consistency_score >= 0.6 else "낮음"
            }
        else:
            return {
                "extracted_answers": [],
                "consistency_score": 0,
                "is_consistent": False,
                "error": "답안 추출 실패"
            }

    def comprehensive_solve(self, problem: str, model_name: str = "openai") -> Dict[str, Any]:
        """종합적 문제 해결"""
        print(f"문제 분석 중: {problem[:50]}...")

        # 1. 전략 기반 해결
        strategic_result = self.solve_with_strategy(problem, model_name)
        print("전략 기반 해결 완료")

        # 2. Self-Consistency 해결
        consistency_result = self.execute_with_self_consistency(problem, model_name)
        print("Self-Consistency 해결 완료")

        # 3. 단계별 설명 생성
        explanation = self.create_step_by_step_explanation(
            problem,
            strategic_result["response"],
            model_name
        )
        print("단계별 설명 생성 완료")

        return {
            "problem": problem,
            "strategic_solution": strategic_result,
            "consistency_solution": consistency_result,
            "detailed_explanation": explanation,
            "overall_assessment": self._assess_solution_quality(strategic_result, consistency_result)
        }

    def _assess_solution_quality(self, strategic_result: Dict, consistency_result: Dict) -> Dict[str, Any]:
        """솔루션 품질 평가"""
        assessment = {
            "strategy_effectiveness": 0,
            "consistency_reliability": 0,
            "overall_confidence": 0
        }

        # 전략 효과성 평가
        type_analysis = strategic_result["type_analysis"]
        if type_analysis["primary_type"] != "general":
            assessment["strategy_effectiveness"] += 0.3
        if type_analysis["complexity_score"] > 0:
            assessment["strategy_effectiveness"] += 0.2

        response_quality_indicators = [
            "단계" in strategic_result["response"],
            "계산" in strategic_result["response"],
            "따라서" in strategic_result["response"],
            len(strategic_result["response"]) > 200
        ]
        assessment["strategy_effectiveness"] += sum(response_quality_indicators) * 0.125

        # 일관성 신뢰도 평가
        consistency_analysis = consistency_result["consistency_analysis"]
        if not consistency_analysis.get("error"):
            assessment["consistency_reliability"] = consistency_analysis.get("consistency_score", 0)

        # 전체 신뢰도
        assessment["overall_confidence"] = (assessment["strategy_effectiveness"] + assessment["consistency_reliability"]) / 2

        # 신뢰도 레벨
        confidence_level = assessment["overall_confidence"]
        if confidence_level >= 0.8:
            assessment["confidence_level"] = "매우 높음"
        elif confidence_level >= 0.6:
            assessment["confidence_level"] = "높음"
        elif confidence_level >= 0.4:
            assessment["confidence_level"] = "보통"
        else:
            assessment["confidence_level"] = "낮음"

        return assessment

# 실습 2 테스트
models = {"openai": openai_llm}
math_solver = MathProblemSolver(models)

test_problems = [
    "학교에서 500명의 학생이 있습니다. 이 중 30%는 5학년이고, 20%는 6학년 학생입니다. 5학년 학생들 중 60%는 수학 동아리에 있고, 나머지는 과학 동아리에 있습니다. 6학년 학생들 중 70%는 수학 동아리에 있고, 나머지는 과학 동아리에 있습니다. 과학 동아리에는 몇 명의 학생이 있나요?",
    "한 상자에 빨간 공 24개와 파란 공 36개가 있습니다. 전체 공의 25%를 무작위로 뽑았을 때, 뽑은 공 중에 빨간 공이 8개가 있었다면 파란 공은 몇 개를 뽑았나요?",
    "직사각형 모양의 화단이 있습니다. 가로가 12m, 세로가 8m입니다. 이 화단 둘레에 1m 간격으로 나무를 심으려고 합니다. 모서리에도 나무를 심는다면 총 몇 그루의 나무가 필요한가요?"
]

print("=== 수학 문제 자동 해결 시스템 테스트 ===")

for i, problem in enumerate(test_problems, 1):
    print(f"\n{'='*20} 문제 {i} {'='*20}")
    result = math_solver.comprehensive_solve(problem)

    print(f"문제: {problem}")
    print(f"\n【문제 분석】")
    type_analysis = result["strategic_solution"]["type_analysis"]
    print(f"주요 유형: {type_analysis['primary_type']}")
    print(f"복잡도: {type_analysis['complexity_score']}/10")
    print(f"추출된 숫자: {type_analysis['extracted_numbers']}")

    print(f"\n【전략 기반 해결】")
    print(result["strategic_solution"]["response"][:300] + "...")

    print(f"\n【일관성 분석】")
    consistency = result["consistency_solution"]["consistency_analysis"]
    if not consistency.get("error"):
        print(f"일관성 점수: {consistency['consistency_score']:.2f}")
        print(f"일치도: {consistency['agreement_level']}")
        print(f"추출된 답들: {consistency['extracted_answers']}")

    print(f"\n【품질 평가】")
    assessment = result["overall_assessment"]
    print(f"전략 효과성: {assessment['strategy_effectiveness']:.2f}")
    print(f"일관성 신뢰도: {assessment['consistency_reliability']:.2f}")
    print(f"전체 신뢰도: {assessment['confidence_level']}")
```

### 실습 3: 멀티모달 추론 시스템

```python
class MultimodalReasoningSystem:
    def __init__(self, models: Dict[str, Any]):
        """멀티모달 추론 시스템"""
        self.models = models
        self.math_symbols = {
            "×": "*", "÷": "/", "²": "**2", "³": "**3",
            "√": "sqrt", "∑": "sum", "∏": "product",
            "≤": "<=", "≥": ">=", "≠": "!=", "≈": "≈"
        }

    def parse_mathematical_expressions(self, text: str) -> Dict[str, Any]:
        """수식 파싱"""
        import re

        # 수식 패턴들
        patterns = {
            "fractions": r'\d+/\d+',
            "percentages": r'\d+(?:\.\d+)?%',
            "equations": r'[a-zA-Z]\s*=\s*[\d\w\+\-\*/\(\)]+',
            "formulas": r'[A-Za-z]+\s*=\s*[^,\n]+',
            "numbers": r'\d+(?:\.\d+)?',
            "operations": r'[+\-×÷*/=]',
            "variables": r'\b[a-zA-Z]\b',
            "parentheses": r'[\(\)]'
        }

        extracted = {}
        for pattern_name, pattern in patterns.items():
            matches = re.findall(pattern, text)
            extracted[pattern_name] = matches

        # 수식 복잡도 계산
        complexity_score = (
            len(extracted["numbers"]) * 1 +
            len(extracted["operations"]) * 2 +
            len(extracted["variables"]) * 3 +
            len(extracted["formulas"]) * 4 +
            len(extracted["equations"]) * 5
        )

        return {
            "extracted_patterns": extracted,
            "complexity_score": complexity_score,
            "has_algebra": len(extracted["variables"]) > 0 or len(extracted["equations"]) > 0,
            "has_geometry": any(keyword in text.lower() for keyword in ["넓이", "둘레", "부피", "각도", "반지름"]),
            "expression_count": sum(len(v) for v in extracted.values())
        }

    def create_visual_explanation(self, solution_steps: List[str], model_name: str = "openai") -> Dict[str, Any]:
        """시각적 설명 생성"""
        visual_template = """
다음 수학 문제 해결 단계들을 시각적으로 표현할 수 있는 설명을 작성하시오:

해결 단계들:
{steps}

다음 형식으로 시각적 설명을 제공하시오:

**1. 문제 상황 다이어그램**
- 주어진 정보를 도표나 그림으로 표현
- 관계를 화살표나 연결선으로 표시

**2. 계산 과정 시각화**
- 각 계산 단계를 박스나 플로우차트로 표현
- 중간 결과를 명확히 표시

**3. 검증 과정 표현**
- 답안 확인 과정을 역산이나 다른 방법으로 시각화
- 결과의 타당성을 그래프나 비교표로 표현

**4. 일반화 패턴**
- 이 문제 유형의 일반적 해결 패턴을 플로우차트로 표현
- 비슷한 문제에 적용할 수 있는 템플릿 제시

시각적 설명:
"""

        steps_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(solution_steps))

        prompt = PromptTemplate(
            input_variables=["steps"],
            template=visual_template
        )

        model = self.models[model_name]
        chain = prompt | model | StrOutputParser()

        visual_explanation = chain.invoke({"steps": steps_text})

        # 시각적 요소 분석
        visual_analysis = self._analyze_visual_content(visual_explanation)

        return {
            "visual_explanation": visual_explanation,
            "visual_analysis": visual_analysis
        }

    def integrate_reasoning_modes(self, problem: str, model_name: str = "openai") -> Dict[str, Any]:
        """추론 모드 통합"""

        # 1. 수식 파싱
        math_parsing = self.parse_mathematical_expressions(problem)

        # 2. 추론 모드 결정
        reasoning_modes = self._determine_reasoning_modes(problem, math_parsing)

        # 3. 통합 프롬프트 생성
        integrated_template = """
다음 수학 문제를 다중 추론 모드로 해결하시오:

문제: {problem}

수식 분석 결과:
- 복잡도 점수: {complexity_score}/20
- 대수적 요소: {has_algebra}
- 기하학적 요소: {has_geometry}
- 추출된 숫자: {numbers}

권장 추론 모드: {reasoning_modes}

**단계 1: 논리적 분해**
- 문제를 논리적 구성요소로 분해
- 각 요소 간의 관계 파악

**단계 2: 수학적 모델링**
- 문제를 수학적 모델로 변환
- 적절한 공식이나 방정식 설정

**단계 3: 단계별 계산**
- 체계적인 계산 수행
- 중간 결과 검증

**단계 4: 시각적 검토**
- 결과를 시각적으로 표현
- 직관적 타당성 확인

**단계 5: 다중 방법 검증**
- 다른 방법으로 재검증
- 일관성 확인

통합 해결:
"""

        prompt = PromptTemplate(
            input_variables=[
                "problem", "complexity_score", "has_algebra",
                "has_geometry", "numbers", "reasoning_modes"
            ],
            template=integrated_template
        )

        model = self.models[model_name]
        chain = prompt | model | StrOutputParser()

        response = chain.invoke({
            "problem": problem,
            "complexity_score": math_parsing["complexity_score"],
            "has_algebra": math_parsing["has_algebra"],
            "has_geometry": math_parsing["has_geometry"],
            "numbers": ", ".join(math_parsing["extracted_patterns"]["numbers"]),
            "reasoning_modes": ", ".join(reasoning_modes)
        })

        # 해결 단계 추출
        solution_steps = self._extract_solution_steps(response)

        # 시각적 설명 생성
        visual_result = self.create_visual_explanation(solution_steps, model_name)

        return {
            "problem": problem,
            "math_parsing": math_parsing,
            "reasoning_modes": reasoning_modes,
            "integrated_solution": response,
            "solution_steps": solution_steps,
            "visual_explanation": visual_result,
            "integration_quality": self._assess_integration_quality(response, visual_result)
        }

    def _determine_reasoning_modes(self, problem: str, math_parsing: Dict[str, Any]) -> List[str]:
        """추론 모드 결정"""
        modes = []

        # 복잡도 기반
        if math_parsing["complexity_score"] > 10:
            modes.append("단계적 분해")

        # 대수적 요소
        if math_parsing["has_algebra"]:
            modes.extend(["방정식 모델링", "변수 추론"])

        # 기하학적 요소
        if math_parsing["has_geometry"]:
            modes.extend(["공간적 시각화", "공식 적용"])

        # 수치 계산
        if len(math_parsing["extracted_patterns"]["numbers"]) > 3:
            modes.append("다단계 계산")

        # 퍼센트나 비율
        if math_parsing["extracted_patterns"]["percentages"]:
            modes.append("비례 추론")

        # 기본 모드
        if not modes:
            modes = ["논리적 추론", "산술 계산"]

        return modes

    def _extract_solution_steps(self, response: str) -> List[str]:
        """해결 단계 추출"""
        import re

        # 단계 패턴들
        step_patterns = [
            r'단계\s*\d+[:：]\s*([^\n]+)',
            r'\d+\.\s*([^\n]+)',
            r'Step\s*\d+[:：]\s*([^\n]+)',
            r'[①②③④⑤⑥⑦⑧⑨⑩]\s*([^\n]+)'
        ]

        steps = []
        for pattern in step_patterns:
            matches = re.findall(pattern, response)
            if matches:
                steps.extend(matches)
                break  # 첫 번째로 매치된 패턴 사용

        # 패턴이 없으면 문장 단위로 분할
        if not steps:
            sentences = [s.strip() for s in response.split('.') if len(s.strip()) > 10]
            steps = sentences[:8]  # 최대 8단계

        return steps

    def _analyze_visual_content(self, visual_explanation: str) -> Dict[str, Any]:
        """시각적 내용 분석"""
        visual_indicators = {
            "diagram_mentions": ["다이어그램", "도표", "그림", "chart", "diagram"],
            "flowchart_mentions": ["플로우차트", "흐름도", "flowchart", "flow"],
            "graph_mentions": ["그래프", "graph", "차트", "chart"],
            "table_mentions": ["표", "테이블", "table"],
            "visual_elements": ["화살표", "박스", "연결선", "색깔", "하이라이트"]
        }

        analysis = {}
        for category, indicators in visual_indicators.items():
            count = sum(1 for indicator in indicators if indicator in visual_explanation)
            analysis[category] = count

        # 시각화 품질 점수
        total_visual_elements = sum(analysis.values())
        analysis["visual_richness_score"] = min(total_visual_elements / 5.0, 1.0)  # 0-1 정규화

        # 구조화 수준
        structure_indicators = ["**", "##", "1.", "2.", "3.", "•", "-"]
        structure_count = sum(1 for indicator in structure_indicators if indicator in visual_explanation)
        analysis["structure_level"] = min(structure_count / 10.0, 1.0)

        return analysis

    def _assess_integration_quality(self, integrated_solution: str, visual_result: Dict[str, Any]) -> Dict[str, Any]:
        """통합 품질 평가"""

        # 통합 솔루션 품질
        integration_indicators = [
            "단계" in integrated_solution,
            "모델링" in integrated_solution,
            "검증" in integrated_solution,
            "시각적" in integrated_solution,
            len(integrated_solution) > 500,  # 충분한 상세도
            "따라서" in integrated_solution or "결론" in integrated_solution
        ]

        integration_score = sum(integration_indicators) / len(integration_indicators)

        # 시각적 설명 품질
        visual_analysis = visual_result["visual_analysis"]
        visual_score = (
            visual_analysis["visual_richness_score"] * 0.4 +
            visual_analysis["structure_level"] * 0.6
        )

        # 전체 품질 점수
        overall_quality = (integration_score * 0.7 + visual_score * 0.3)

        return {
            "integration_score": integration_score,
            "visual_score": visual_score,
            "overall_quality": overall_quality,
            "quality_level": (
                "우수" if overall_quality >= 0.8 else
                "양호" if overall_quality >= 0.6 else
                "보통" if overall_quality >= 0.4 else "개선필요"
            ),
            "strengths": self._identify_strengths(integration_indicators, visual_analysis),
            "improvement_areas": self._identify_improvements(integration_indicators, visual_analysis)
        }

    def _identify_strengths(self, integration_indicators: List[bool], visual_analysis: Dict[str, Any]) -> List[str]:
        """강점 식별"""
        strengths = []

        if integration_indicators[0]:  # 단계적 해결
            strengths.append("체계적인 단계별 접근")
        if integration_indicators[2]:  # 검증 포함
            strengths.append("검증 과정 포함")
        if visual_analysis["visual_richness_score"] > 0.6:
            strengths.append("풍부한 시각적 설명")
        if visual_analysis["structure_level"] > 0.7:
            strengths.append("명확한 구조화")

        return strengths

    def _identify_improvements(self, integration_indicators: List[bool], visual_analysis: Dict[str, Any]) -> List[str]:
        """개선점 식별"""
        improvements = []

        if not integration_indicators[1]:  # 모델링 부족
            improvements.append("수학적 모델링 강화 필요")
        if not integration_indicators[3]:  # 시각적 요소 부족
            improvements.append("시각적 표현 추가 필요")
        if visual_analysis["visual_richness_score"] < 0.4:
            improvements.append("더 다양한 시각적 요소 활용")
        if visual_analysis["structure_level"] < 0.5:
            improvements.append("구조화된 설명 방식 개선")

        return improvements

# 실습 3 테스트
models = {"openai": openai_llm}
multimodal_system = MultimodalReasoningSystem(models)

complex_problem = """
한 회사에서 직원들의 급여를 다음과 같이 정했습니다:
- 기본급: 월 300만원
- 성과급: 기본급의 a% (a는 개인 성과에 따라 10~50% 범위)
- 팀 보너스: 전체 급여의 15%
- 세금: 전체 급여의 22%

만약 한 직원이 성과급을 30% 받았다면, 실제 받는 급여(세후)는 얼마인가요?
또한 이 직원이 연간 받는 총 급여는 얼마인가요?
"""

print("=== 멀티모달 추론 시스템 테스트 ===")
multimodal_result = multimodal_system.integrate_reasoning_modes(complex_problem)

print(f"문제: {complex_problem}")

print(f"\n【수식 분석】")
math_parsing = multimodal_result["math_parsing"]
print(f"복잡도 점수: {math_parsing['complexity_score']}/20")
print(f"대수적 요소: {math_parsing['has_algebra']}")
print(f"기하학적 요소: {math_parsing['has_geometry']}")
print(f"추출된 숫자: {math_parsing['extracted_patterns']['numbers']}")
print(f"추출된 퍼센트: {math_parsing['extracted_patterns']['percentages']}")

print(f"\n【권장 추론 모드】")
print(f"선택된 모드: {', '.join(multimodal_result['reasoning_modes'])}")

print(f"\n【통합 해결 과정】")
print(multimodal_result["integrated_solution"][:800] + "...")

print(f"\n【추출된 해결 단계】")
for i, step in enumerate(multimodal_result["solution_steps"], 1):
    print(f"{i}. {step}")

print(f"\n【시각적 설명】")
visual_explanation = multimodal_result["visual_explanation"]["visual_explanation"]
print(visual_explanation[:600] + "...")

print(f"\n【품질 평가】")
quality = multimodal_result["integration_quality"]
print(f"통합 점수: {quality['integration_score']:.2f}")
print(f"시각적 점수: {quality['visual_score']:.2f}")
print(f"전체 품질: {quality['overall_quality']:.2f}")
print(f"품질 수준: {quality['quality_level']}")

if quality["strengths"]:
    print(f"강점: {', '.join(quality['strengths'])}")
if quality["improvement_areas"]:
    print(f"개선점: {', '.join(quality['improvement_areas'])}")
```

## 🔍 참고 자료

### 공식 문서
- [LangChain Prompt Templates](https://python.langchain.com/docs/modules/model_io/prompts/)
- [OpenAI GPT Best Practices](https://platform.openai.com/docs/guides/gpt-best-practices)
- [Chain of Thought Prompting Guide](https://www.promptingguide.ai/techniques/cot)

### 학술 자료
- Wei, J., et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
- Wang, X., et al. (2022). "Self-Consistency Improves Chain of Thought Reasoning in Language Models"
- Gao, L., et al. (2023). "PAL: Program-aided Language Models"
- Shinn, N., et al. (2023). "Reflexion: Language Agents with Verbal Reinforcement Learning"

### 실무 가이드
- [Advanced Prompting Techniques](https://example.com/advanced-prompting)
- [Chain of Thought Implementation Guide](https://example.com/cot-implementation)
- [Mathematical Reasoning with LLMs](https://example.com/math-reasoning)

### 도구 및 리소스
- [Prompt Engineering Toolkit](https://example.com/prompt-toolkit)
- [CoT Benchmarking Datasets](https://example.com/cot-benchmarks)
- [Mathematical Problem Collections](https://example.com/math-problems)

---

**다음 학습**: W3_004_Chat_History.md - 채팅 히스토리 관리와 대화 맥락 유지 기법을 학습합니다.