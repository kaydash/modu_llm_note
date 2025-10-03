# W3_001_Prompt_Engineering_Basic.md - 프롬프트 엔지니어링 기초

## 🎯 학습 목표

- 효과적인 프롬프트 템플릿의 기본 구조와 설계 원칙을 이해합니다
- 다양한 프롬프트 유형(질문형, 지시형, 대화형, 조건부, 예시 기반)을 실습합니다
- LangChain의 PromptTemplate과 ChatPromptTemplate을 활용합니다
- 명확성, 맥락성, 구조화의 프롬프트 엔지니어링 핵심 원칙을 적용합니다
- 맞춤형 학습 도우미 챗봇을 구현하여 실무 적용 능력을 개발합니다

## 📚 핵심 개념

### 1. 프롬프트 엔지니어링이란?

프롬프트 엔지니어링은 AI 모델에게 효과적인 지시를 제공하여 원하는 결과를 얻어내는 기술입니다. 입력(프롬프트)을 최적화하여 출력의 품질을 향상시키는 방법론으로, 현대 AI 개발에서 핵심적인 역할을 합니다.

#### 프롬프트 엔지니어링의 중요성
```python
# 일반적인 프롬프트 vs 최적화된 프롬프트

# 기본 프롬프트
basic_prompt = "AI에 대해 설명해줘"

# 최적화된 프롬프트
optimized_prompt = """
주제: 인공지능
대상: 고등학생
목적: 진로 선택을 위한 기초 이해

다음 형식으로 설명해주세요:
1. 정의 (2문장)
2. 주요 분야 3가지 (각각 1문장씩)
3. 미래 전망 (2문장)

총 500자 이내로 작성하고, 전문용어에는 간단한 설명을 괄호 안에 포함해주세요.
"""
```

### 2. 프롬프트 엔지니어링의 핵심 원칙

#### 2.1 명확성(Clarity)
모호하지 않은 명확한 지시사항을 제공하는 것이 중요합니다.

**나쁜 예시:**
```python
vague_prompt = "요약해주세요"
```

**좋은 예시:**
```python
clear_prompt = """
다음 텍스트를 정확히 3문장으로 요약하세요:
- 각 문장은 20단어 이내
- 핵심 내용만 포함
- 전문용어는 일반용어로 대체
"""
```

#### 2.2 맥락성(Context)
관련 배경 정보와 사용 목적을 명확히 제공해야 합니다.

```python
contextual_prompt = """
배경: 65세 이상 노인 대상 스마트폰 교육 프로그램
목적: 처음 스마트폰을 사용하는 노인들의 기본 기능 학습
대상: 디지털 기기 사용 경험이 거의 없는 노인

위 맥락을 고려하여 스마트폰 기본 기능을 설명해주세요:
- 쉬운 용어 사용
- 단계별 설명
- 실생활 예시 포함
"""
```

#### 2.3 구조화(Structure)
체계적인 형식과 단계별 지시사항을 제공해야 합니다.

```python
structured_prompt = """
[작업 단계]
1. 텍스트 분석
2. 핵심 키워드 추출
3. 카테고리 분류
4. 요약문 작성

[출력 형식]
- 키워드: [키워드1, 키워드2, 키워드3]
- 카테고리: [분류명]
- 요약: [2문장 요약]

[제약사항]
- 객관적 사실만 포함
- 200자 이내 작성
"""
```

### 3. 프롬프트 유형별 특성

#### 3.1 질문형 프롬프트 (Question Prompts)
정보 추출에 효과적이며 구체적인 답변을 유도할 수 있습니다.

**특징:**
- 명확한 정보 요청
- 구체적 답변 유도
- 분석적 사고 촉진

```python
question_types = {
    'factual': "양자 컴퓨팅의 정의는 무엇인가요?",
    'analytical': "다음 텍스트의 주요 논점은 무엇인가요?",
    'comparative': "A와 B의 차이점은 무엇인가요?",
    'explanatory': "이 현상이 발생하는 이유는 무엇인가요?"
}
```

#### 3.2 지시형 프롬프트 (Instruction Prompts)
명확한 작업 수행을 지시하며 단계별 처리가 가능합니다.

**특징:**
- 구체적 작업 지시
- 단계별 프로세스
- 명확한 출력 형식

```python
instruction_types = {
    'translation': "다음 텍스트를 한국어로 번역하세요",
    'summarization': "다음 텍스트를 3문장으로 요약하세요",
    'analysis': "다음 데이터를 분석하고 패턴을 찾으세요",
    'generation': "주어진 주제로 에세이를 작성하세요"
}
```

#### 3.3 대화형 프롬프트 (Conversational Prompts)
자연스러운 상호작용과 문맥 유지가 가능합니다.

**특징:**
- 역할 기반 대화
- 문맥 연속성
- 개인화된 응답

```python
conversation_roles = {
    'system': "당신은 친절한 고객 서비스 담당자입니다",
    'assistant': "전문적이면서도 이해하기 쉽게 설명하는 교사입니다",
    'user': "구체적인 도움이 필요한 학습자입니다"
}
```

#### 3.4 조건부 프롬프트 (Conditional Prompts)
입력에 따라 다른 처리 방식을 적용합니다.

```python
conditional_logic = """
입력 유형에 따른 처리:
- 질문인 경우: 명확한 답변 제공
- 진술문인 경우: 사실 여부 검증
- 요청사항인 경우: 단계별 방법 설명
"""
```

#### 3.5 예시 기반 프롬프트 (Few-Shot Prompts)
구체적인 예시를 통해 원하는 출력 형식을 학습시킵니다.

```python
few_shot_example = """
예시 1:
입력: "오늘 날씨가 정말 좋네요"
출력: 감정=긍정, 주제=날씨, 강도=높음

예시 2:
입력: "교통이 너무 막혀서 짜증나요"
출력: 감정=부정, 주제=교통, 강도=높음

이제 다음 텍스트를 같은 형식으로 분석하세요:
입력: {user_input}
출력:
"""
```

## 🔧 환경 설정

### 1. 필수 라이브러리 설치

```bash
# uv 사용 (권장)
uv add langchain langchain-openai python-dotenv

# pip 사용
pip install langchain langchain-openai python-dotenv
```

### 2. 환경 변수 설정

```bash
# .env 파일에 추가
OPENAI_API_KEY=your_openai_api_key_here

# Langfuse 추적 (선택사항)
LANGFUSE_ENABLED=false
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 3. 기본 임포트

```python
import os
from dotenv import load_dotenv
from pprint import pprint
from typing import Dict, List, Any, Optional

# LangChain 관련
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 환경 변수 로드
load_dotenv()

# LLM 초기화
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)
```

## 💻 코드 예제

### 1. 기본 프롬프트 템플릿 활용

```python
class BasicPromptManager:
    def __init__(self, model_name: str = "gpt-4.1-mini", temperature: float = 0):
        """기본 프롬프트 관리자 초기화"""
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)

    def create_question_prompt(self, template: str, variables: List[str]) -> PromptTemplate:
        """질문형 프롬프트 생성"""
        return PromptTemplate(
            template=template,
            input_variables=variables
        )

    def create_instruction_prompt(self, template: str, variables: List[str]) -> PromptTemplate:
        """지시형 프롬프트 생성"""
        return PromptTemplate(
            template=template,
            input_variables=variables
        )

    def execute_chain(self, prompt: PromptTemplate, inputs: Dict[str, Any]) -> str:
        """프롬프트 체인 실행"""
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(inputs)

# 사용 예시
prompt_manager = BasicPromptManager()

# 1. 질문형 프롬프트 예시
question_template = """
다음 주제에 대해 세 가지 관점에서 분석해주세요:

[주제]
{topic}

[분석 관점]
1. 기술적 측면
2. 경제적 측면
3. 사회적 측면

각 관점별로 2-3문장씩 설명해주세요.
"""

question_prompt = prompt_manager.create_question_prompt(
    template=question_template,
    variables=["topic"]
)

result = prompt_manager.execute_chain(
    prompt=question_prompt,
    inputs={"topic": "인공지능의 발전"}
)

print("=== 질문형 프롬프트 결과 ===")
print(result)

# 2. 지시형 프롬프트 예시
instruction_template = """
다음 텍스트에 대해 아래 작업을 순서대로 수행하세요:

[텍스트]
{text}

[작업 순서]
1. 텍스트를 1문장으로 요약
2. 핵심 키워드 3개 추출
3. 감정 분석 수행 (긍정/부정/중립)
4. 카테고리 분류 (기술/경제/사회/문화/기타)

[작업 결과]
요약:
키워드:
감정:
카테고리:
"""

instruction_prompt = prompt_manager.create_instruction_prompt(
    template=instruction_template,
    variables=["text"]
)

text_input = """
최근 AI 기술의 급속한 발전으로 많은 산업 분야에서 혁신이 일어나고 있다.
특히 자연어 처리, 컴퓨터 비전, 로봇 공학 등의 영역에서 놀라운 성과를 보이고 있으며,
이는 우리의 일상생활과 업무 환경을 크게 변화시키고 있다.
"""

result = prompt_manager.execute_chain(
    prompt=instruction_prompt,
    inputs={"text": text_input}
)

print("\n=== 지시형 프롬프트 결과 ===")
print(result)
```

### 2. 대화형 프롬프트 시스템

```python
class ConversationalPromptSystem:
    def __init__(self, model_name: str = "gpt-4.1-mini", temperature: float = 0.1):
        """대화형 프롬프트 시스템 초기화"""
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)

    def create_chat_prompt(
        self,
        system_message: str,
        human_template: str,
        variables: List[str]
    ) -> ChatPromptTemplate:
        """채팅 프롬프트 템플릿 생성"""
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_template)
        ])

    def create_role_based_prompt(
        self,
        system_template: str,
        human_template: str,
        system_variables: List[str] = None,
        human_variables: List[str] = None
    ) -> ChatPromptTemplate:
        """역할 기반 프롬프트 생성"""
        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ]
        return ChatPromptTemplate.from_messages(messages)

    def execute_conversation(self, prompt: ChatPromptTemplate, inputs: Dict[str, Any]) -> str:
        """대화 실행"""
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(inputs)

# 사용 예시
chat_system = ConversationalPromptSystem()

# 1. 고객 서비스 챗봇 예시
customer_service_prompt = chat_system.create_chat_prompt(
    system_message="당신은 친절하고 전문적인 고객 서비스 담당자입니다. 고객의 문제를 정확히 파악하고 해결책을 제시해주세요.",
    human_template="고객 문의: {customer_message}",
    variables=["customer_message"]
)

customer_response = chat_system.execute_conversation(
    prompt=customer_service_prompt,
    inputs={"customer_message": "제품 배송이 지연되고 있는데 언제쯤 받을 수 있을까요?"}
)

print("=== 고객 서비스 응답 ===")
print(customer_response)

# 2. 교육 튜터 시스템 예시
tutor_system_message = """
당신은 경험이 풍부한 {subject} 교사입니다.
학생의 수준: {level}
교육 목표: {goal}

다음 원칙에 따라 가르쳐주세요:
- 학생 수준에 맞는 용어와 예시 사용
- 단계별로 설명하여 이해를 돕기
- 궁금한 점이 있으면 언제든 질문하도록 격려
- 실생활 예시를 통한 이해 증진
"""

tutor_prompt = chat_system.create_role_based_prompt(
    system_template=tutor_system_message,
    human_template="학생 질문: {student_question}",
    system_variables=["subject", "level", "goal"],
    human_variables=["student_question"]
)

tutor_response = chat_system.execute_conversation(
    prompt=tutor_prompt,
    inputs={
        "subject": "수학",
        "level": "중학교 2학년",
        "goal": "이차함수의 기본 개념 이해",
        "student_question": "이차함수가 왜 포물선 모양이 되는지 이해가 안돼요"
    }
)

print("\n=== 교육 튜터 응답 ===")
print(tutor_response)
```

### 3. 조건부 및 Few-Shot 프롬프트

```python
class AdvancedPromptTechniques:
    def __init__(self, model_name: str = "gpt-4.1-mini", temperature: float = 0):
        """고급 프롬프트 기법 클래스"""
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)

    def create_conditional_prompt(self, template: str, variables: List[str]) -> PromptTemplate:
        """조건부 프롬프트 생성"""
        return PromptTemplate(template=template, input_variables=variables)

    def create_few_shot_prompt(
        self,
        examples: List[Dict[str, str]],
        input_template: str,
        variables: List[str]
    ) -> PromptTemplate:
        """Few-Shot 프롬프트 생성"""
        example_text = ""
        for i, example in enumerate(examples, 1):
            example_text += f"\n예시 {i}:\n"
            for key, value in example.items():
                example_text += f"{key}: {value}\n"

        full_template = f"{example_text}\n이제 다음 입력을 같은 방식으로 처리해주세요:\n{input_template}"

        return PromptTemplate(template=full_template, input_variables=variables)

    def execute_prompt(self, prompt: PromptTemplate, inputs: Dict[str, Any]) -> str:
        """프롬프트 실행"""
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(inputs)

# 사용 예시
advanced_prompts = AdvancedPromptTechniques()

# 1. 조건부 프롬프트 예시
conditional_template = """
입력 텍스트: {text}

다음 조건에 따라 처리해주세요:

조건 1: 입력이 질문인 경우
→ 명확하고 구체적인 답변 제공
→ 필요시 예시 포함

조건 2: 입력이 진술문인 경우
→ 진술문의 사실 여부 검증
→ 근거와 출처 제시

조건 3: 입력이 요청사항인 경우
→ 수행 방법을 단계별로 설명
→ 주의사항이나 팁 포함

응답 형식:
유형: [질문/진술문/요청사항]
내용: [상세 응답]
추가 정보: [필요한 경우만]
"""

conditional_prompt = advanced_prompts.create_conditional_prompt(
    template=conditional_template,
    variables=["text"]
)

# 테스트 입력들
test_inputs = [
    "머신러닝과 딥러닝의 차이점은 무엇인가요?",  # 질문
    "Python은 프로그래밍 언어 중에서 가장 쉬운 언어이다.",  # 진술문
    "파이썬으로 웹 크롤링을 시작하는 방법을 알려주세요."  # 요청사항
]

print("=== 조건부 프롬프트 테스트 ===")
for i, test_input in enumerate(test_inputs, 1):
    result = advanced_prompts.execute_prompt(
        prompt=conditional_prompt,
        inputs={"text": test_input}
    )
    print(f"\n--- 테스트 {i} ---")
    print(f"입력: {test_input}")
    print(f"결과: {result}")

# 2. Few-Shot 프롬프트 예시
sentiment_examples = [
    {
        "리뷰": "이 제품 정말 좋아요! 품질이 훌륭하고 배송도 빨랐습니다.",
        "분석": "감정=긍정, 만족도=높음, 주요 요소=[품질, 배송], 평점=5/5"
    },
    {
        "리뷰": "가격 대비 괜찮은 편이지만 디자인이 아쉬워요.",
        "분석": "감정=중립, 만족도=보통, 주요 요소=[가격, 디자인], 평점=3/5"
    },
    {
        "리뷰": "완전 실망했습니다. 품질도 안좋고 고객서비스도 불친절해요.",
        "분석": "감정=부정, 만족도=낮음, 주요 요소=[품질, 서비스], 평점=1/5"
    }
]

few_shot_prompt = advanced_prompts.create_few_shot_prompt(
    examples=sentiment_examples,
    input_template="리뷰: {review}\n분석:",
    variables=["review"]
)

test_review = "배송은 빨랐는데 포장이 너무 허술해서 제품이 약간 손상되었어요. 그래도 사용하는데 문제는 없네요."

few_shot_result = advanced_prompts.execute_prompt(
    prompt=few_shot_prompt,
    inputs={"review": test_review}
)

print("\n\n=== Few-Shot 프롬프트 테스트 ===")
print(f"테스트 리뷰: {test_review}")
print(f"분석 결과: {few_shot_result}")
```

### 4. 프롬프트 최적화 시스템

```python
class PromptOptimizer:
    def __init__(self, model_name: str = "gpt-4.1-mini"):
        """프롬프트 최적화 시스템"""
        self.llm = ChatOpenAI(model=model_name, temperature=0)

    def analyze_prompt_quality(self, prompt_text: str) -> Dict[str, Any]:
        """프롬프트 품질 분석"""
        analysis_prompt = PromptTemplate(
            template="""
다음 프롬프트의 품질을 분석하고 개선점을 제안해주세요:

[분석 대상 프롬프트]
{prompt_text}

[분석 기준]
1. 명확성: 지시사항이 명확한가?
2. 구체성: 구체적인 요구사항이 있는가?
3. 구조화: 체계적으로 구성되어 있는가?
4. 맥락성: 충분한 맥락 정보가 있는가?

[출력 형식]
점수 (1-5점):
- 명확성: X/5
- 구체성: X/5
- 구조화: X/5
- 맥락성: X/5

개선점:
- [구체적인 개선 제안]

개선된 프롬프트:
[최적화된 버전]
""",
            input_variables=["prompt_text"]
        )

        chain = analysis_prompt | self.llm | StrOutputParser()
        return chain.invoke({"prompt_text": prompt_text})

    def create_optimized_prompt(
        self,
        task_description: str,
        target_audience: str,
        desired_output: str,
        constraints: List[str] = None
    ) -> str:
        """최적화된 프롬프트 생성"""
        optimization_template = """
다음 요구사항을 바탕으로 최적화된 프롬프트를 작성해주세요:

작업 설명: {task_description}
대상 사용자: {target_audience}
원하는 출력: {desired_output}
제약사항: {constraints}

프롬프트 엔지니어링 원칙을 적용하여:
1. 명확하고 구체적인 지시사항
2. 적절한 맥락 정보 포함
3. 체계적인 구조화
4. 원하는 출력 형식 명시

최적화된 프롬프트:
"""

        prompt = PromptTemplate(
            template=optimization_template,
            input_variables=["task_description", "target_audience", "desired_output", "constraints"]
        )

        chain = prompt | self.llm | StrOutputParser()

        constraints_text = ", ".join(constraints) if constraints else "없음"

        return chain.invoke({
            "task_description": task_description,
            "target_audience": target_audience,
            "desired_output": desired_output,
            "constraints": constraints_text
        })

# 사용 예시
optimizer = PromptOptimizer()

# 1. 기존 프롬프트 품질 분석
existing_prompt = "AI에 대해 설명해주세요."

quality_analysis = optimizer.analyze_prompt_quality(existing_prompt)
print("=== 프롬프트 품질 분석 ===")
print(quality_analysis)

# 2. 최적화된 프롬프트 생성
optimized_prompt = optimizer.create_optimized_prompt(
    task_description="고등학생을 위한 인공지능 개념 설명",
    target_audience="인공지능에 대한 사전 지식이 없는 고등학생",
    desired_output="정의, 응용 분야, 미래 전망을 포함한 500자 이내 설명",
    constraints=["쉬운 용어 사용", "실생활 예시 포함", "진로 연결 정보 제공"]
)

print("\n=== 최적화된 프롬프트 ===")
print(optimized_prompt)
```

## 🚀 실습해보기

### 실습 1: 다단계 분석 시스템

두 개의 문장을 비교 분석하는 체인을 구성해보세요.

```python
# 문장 비교 분석 시스템 구현
class SentenceComparator:
    def __init__(self):
        # 초기화 코드 작성
        pass

    def create_comparison_prompt(self):
        # 비교 분석 프롬프트 작성
        pass

    def analyze_sentences(self, sentence1, sentence2):
        # 문장 비교 분석 실행
        pass
```

### 실습 2: 리뷰 분석 시스템

상품 리뷰를 분석하는 대화형 AI 시스템을 구현해보세요.

```python
# 리뷰 분석 전문가 시스템 구현
class ReviewAnalyzer:
    def __init__(self):
        # 초기화 코드 작성
        pass

    def create_review_analyzer(self):
        # 리뷰 분석 프롬프트 작성
        pass

    def analyze_multiple_reviews(self, reviews):
        # 다중 리뷰 분석
        pass
```

### 실습 3: 맞춤형 학습 도우미

특정 주제에 대한 학습을 돕는 챗봇을 만들어보세요.

```python
# 학습 도우미 챗봇 구현
class LearningAssistant:
    def __init__(self, subject):
        # 초기화 코드 작성
        pass

    def create_quiz_system(self):
        # 퀴즈 생성 시스템 구현
        pass

    def create_explanation_system(self):
        # 개념 설명 시스템 구현
        pass
```

## 📋 해답

### 실습 1: 다단계 분석 시스템

```python
class SentenceComparator:
    def __init__(self, model_name: str = "gpt-4.1-mini", temperature: float = 0):
        """문장 비교 분석 시스템 초기화"""
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)

    def create_comparison_prompt(self) -> PromptTemplate:
        """비교 분석 프롬프트 생성"""
        template = """
다음 두 문장의 맥락 일치도를 단계별로 분석하세요:

[A문장]
{sentence_a}

[B문장]
{sentence_b}

[분석 단계]
1. 각 문장을 1문장으로 핵심 요약
2. 각 문장에서 핵심 키워드 3개 추출
3. 키워드 기반 유사도 분석
4. 의미적 맥락 일치도 판단
5. 최종 결론 (일치/부분일치/불일치)

[출력 형식]
**1단계 - 요약**
A문장 요약:
B문장 요약:

**2단계 - 키워드 추출**
A문장 키워드: [키워드1, 키워드2, 키워드3]
B문장 키워드: [키워드1, 키워드2, 키워드3]

**3단계 - 유사도 분석**
공통 키워드:
관련 키워드:
차이점:

**4단계 - 맥락 분석**
주제 일치도:
논조 유사성:
의도 일치도:

**5단계 - 최종 결론**
맥락 일치도: [일치/부분일치/불일치]
일치도 점수: X/10점
근거:
"""
        return PromptTemplate(
            template=template,
            input_variables=["sentence_a", "sentence_b"]
        )

    def analyze_sentences(self, sentence1: str, sentence2: str) -> Dict[str, Any]:
        """문장 비교 분석 실행"""
        prompt = self.create_comparison_prompt()
        chain = prompt | self.llm | StrOutputParser()

        result = chain.invoke({
            "sentence_a": sentence1,
            "sentence_b": sentence2
        })

        return {
            "sentence_a": sentence1,
            "sentence_b": sentence2,
            "analysis": result
        }

    def batch_comparison(self, sentence_pairs: List[tuple]) -> List[Dict[str, Any]]:
        """다중 문장 쌍 비교"""
        results = []

        for i, (sent_a, sent_b) in enumerate(sentence_pairs, 1):
            print(f"\n=== 비교 분석 {i} ===")
            result = self.analyze_sentences(sent_a, sent_b)
            results.append(result)
            print(result['analysis'])

        return results

# 실습 1 테스트
comparator = SentenceComparator()

test_pairs = [
    (
        "사람은 언어를 사용하여 의사소통을 한다.",
        "인간은 언어를 통해 서로 소통한다."
    ),
    (
        "인공지능은 미래 사회를 변화시킬 것이다.",
        "고양이는 귀여운 동물이다."
    ),
    (
        "기술의 발전은 인류에게 도움이 된다.",
        "과학 기술의 진보는 사회 발전에 기여한다."
    )
]

comparison_results = comparator.batch_comparison(test_pairs)

# 결과 요약
print("\n=== 비교 분석 요약 ===")
for i, result in enumerate(comparison_results, 1):
    print(f"{i}. 문장 쌍의 맥락 일치 여부 분석 완료")
```

### 실습 2: 리뷰 분석 시스템

```python
class ReviewAnalyzer:
    def __init__(self, model_name: str = "gpt-4.1-mini", temperature: float = 0.1):
        """리뷰 분석 전문가 시스템 초기화"""
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)

    def create_review_analyzer(self) -> ChatPromptTemplate:
        """리뷰 분석 프롬프트 생성"""
        system_message = """
당신은 상품 리뷰 분석 전문가입니다. 다음과 같은 분석을 수행합니다:

분석 범위:
1. 감정 분석 (긍정/중립/부정, 강도 1-5)
2. 주요 언급 요소 추출 (품질, 가격, 배송, 서비스 등)
3. 장점과 단점 구분
4. 구매 결정 요인 분석
5. 개선점 제안

출력 형식:
**감정 분석**
- 전체 감정: [긍정/중립/부정]
- 감정 강도: X/5
- 감정 근거: [구체적 근거]

**주요 언급 요소**
- 품질: [언급내용/미언급]
- 가격: [언급내용/미언급]
- 배송: [언급내용/미언급]
- 서비스: [언급내용/미언급]
- 기타: [기타 요소들]

**장단점 분석**
- 장점: [구체적 장점들]
- 단점: [구체적 단점들]

**구매 결정 요인**
- 주요 결정 요인: [핵심 요인들]
- 망설임 요인: [우려사항들]

**종합 평가**
- 추천도: X/5
- 핵심 메시지: [한 문장 요약]
- 개선 제안: [판매자를 위한 제안]
"""

        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "다음 리뷰를 분석해주세요:\n\n{review}")
        ])

    def analyze_single_review(self, review: str) -> Dict[str, Any]:
        """단일 리뷰 분석"""
        prompt = self.create_review_analyzer()
        chain = prompt | self.llm | StrOutputParser()

        analysis = chain.invoke({"review": review})

        return {
            "original_review": review,
            "analysis": analysis,
            "review_length": len(review),
            "word_count": len(review.split())
        }

    def analyze_multiple_reviews(self, reviews: List[str]) -> Dict[str, Any]:
        """다중 리뷰 분석"""
        individual_analyses = []

        for i, review in enumerate(reviews, 1):
            print(f"\n=== 리뷰 {i} 분석 중 ===")
            analysis = self.analyze_single_review(review)
            individual_analyses.append(analysis)
            print(f"리뷰 길이: {analysis['word_count']}단어")

        # 종합 분석
        summary = self._generate_summary_analysis(individual_analyses)

        return {
            "individual_analyses": individual_analyses,
            "summary_analysis": summary,
            "total_reviews": len(reviews)
        }

    def _generate_summary_analysis(self, analyses: List[Dict[str, Any]]) -> str:
        """종합 분석 생성"""
        summary_prompt = PromptTemplate(
            template="""
다음은 {review_count}개 리뷰의 개별 분석 결과입니다:

{analyses_text}

이를 바탕으로 종합 분석을 수행해주세요:

**전체 감정 분포**
- 긍정적 리뷰: X개 (X%)
- 중립적 리뷰: X개 (X%)
- 부정적 리뷰: X개 (X%)

**공통 언급 요소**
- 가장 많이 언급된 장점:
- 가장 많이 언급된 단점:
- 주요 관심사:

**종합 평가**
- 전체 만족도: X/5
- 주요 강점:
- 개선 필요사항:
- 구매 추천도:

**비즈니스 인사이트**
- 마케팅 포인트:
- 제품 개선 방향:
- 고객 서비스 개선점:
""",
            input_variables=["review_count", "analyses_text"]
        )

        # 분석 결과들을 텍스트로 합치기
        analyses_text = ""
        for i, analysis in enumerate(analyses, 1):
            analyses_text += f"\n--- 리뷰 {i} 분석 ---\n"
            analyses_text += analysis["analysis"][:500] + "...\n"

        chain = summary_prompt | self.llm | StrOutputParser()

        return chain.invoke({
            "review_count": len(analyses),
            "analyses_text": analyses_text
        })

    def create_response_templates(self) -> Dict[str, ChatPromptTemplate]:
        """고객 응답 템플릿 생성"""
        templates = {}

        # 긍정적 리뷰 응답
        templates['positive'] = ChatPromptTemplate.from_messages([
            ("system", "당신은 친절한 고객 서비스 담당자입니다. 긍정적인 리뷰에 감사 인사를 표현하세요."),
            ("human", "다음 긍정적 리뷰에 대한 감사 응답을 작성해주세요:\n{review}")
        ])

        # 부정적 리뷰 응답
        templates['negative'] = ChatPromptTemplate.from_messages([
            ("system", "당신은 문제 해결 전문 고객 서비스 담당자입니다. 부정적 리뷰에 대해 사과하고 해결책을 제시하세요."),
            ("human", "다음 부정적 리뷰에 대한 사과 및 해결 방안을 제시해주세요:\n{review}")
        ])

        return templates

# 실습 2 테스트
analyzer = ReviewAnalyzer()

sample_reviews = [
    "이 노트북 정말 가볍고 좋아요! 배터리도 오래가고 화면도 선명해요. 다만 가격이 조금 비싸네요.",
    "배송은 빨랐는데 포장이 너무 허술해서 제품이 약간 손상되었어요. 그래도 사용하는데 문제는 없네요.",
    "완전 실망했습니다. 설명과 다르고 품질도 안좋아요. 환불 요청드립니다.",
    "가성비 좋은 제품이네요. 기대보다 훨씬 만족스럽고 추천하고 싶어요!",
    "보통 수준이에요. 특별히 좋지도 나쁘지도 않은 평범한 제품입니다."
]

# 다중 리뷰 분석 실행
analysis_results = analyzer.analyze_multiple_reviews(sample_reviews)

print("\n=== 종합 분석 결과 ===")
print(analysis_results['summary_analysis'])

# 응답 템플릿 테스트
templates = analyzer.create_response_templates()
positive_response_chain = templates['positive'] | analyzer.llm | StrOutputParser()

positive_response = positive_response_chain.invoke({
    "review": sample_reviews[0]  # 첫 번째 긍정적 리뷰
})

print("\n=== 고객 응답 예시 ===")
print(positive_response)
```

### 실습 3: 맞춤형 학습 도우미

```python
class LearningAssistant:
    def __init__(self, subject: str, model_name: str = "gpt-4.1-mini"):
        """학습 도우미 초기화"""
        self.subject = subject
        self.llm = ChatOpenAI(model=model_name, temperature=0.3)

    def create_quiz_system(self) -> Dict[str, ChatPromptTemplate]:
        """퀴즈 생성 시스템 구현"""
        quiz_templates = {}

        # 객관식 퀴즈 생성
        quiz_templates['multiple_choice'] = ChatPromptTemplate.from_messages([
            ("system", f"""
당신은 {self.subject} 과목의 전문 교육자입니다.
학습 효과를 높이는 객관식 퀴즈를 만들어주세요.

퀴즈 생성 규칙:
1. 문제는 명확하고 구체적으로 작성
2. 정답 1개, 오답 3개로 총 4개 선택지
3. 오답은 그럴듯하지만 틀린 내용으로 구성
4. 난이도 조절: 기초/중급/고급 중 선택
5. 해설 포함: 왜 정답인지, 오답은 왜 틀렸는지 설명

출력 형식:
**문제**
[문제 내용]

**선택지**
1) [선택지 1]
2) [선택지 2]
3) [선택지 3]
4) [선택지 4]

**정답: X번**

**해설**
[정답 해설 및 오답 분석]
"""),
            ("human", "주제: {topic}\n난이도: {difficulty}\n학습 목표: {objective}")
        ])

        # 주관식 퀴즈 생성
        quiz_templates['short_answer'] = ChatPromptTemplate.from_messages([
            ("system", f"""
당신은 {self.subject} 과목의 전문 교육자입니다.
사고력을 기르는 주관식 문제를 만들어주세요.

문제 생성 규칙:
1. 단순 암기가 아닌 이해와 응용을 요구하는 문제
2. 명확한 평가 기준 제시
3. 예시 답안과 채점 기준 포함
4. 학습자 수준에 맞는 적절한 난이도

출력 형식:
**문제**
[주관식 문제]

**예시 답안**
[모범 답안 예시]

**채점 기준**
- 만점 답안 요건:
- 부분점수 기준:
- 주요 체크포인트:

**학습 팁**
[문제 해결 접근법]
"""),
            ("human", "주제: {topic}\n학습 목표: {objective}\n제한 조건: {constraints}")
        ])

        return quiz_templates

    def create_explanation_system(self) -> Dict[str, ChatPromptTemplate]:
        """개념 설명 시스템 구현"""
        explanation_templates = {}

        # 기본 개념 설명
        explanation_templates['basic_concept'] = ChatPromptTemplate.from_messages([
            ("system", f"""
당신은 {self.subject} 분야의 뛰어난 교사입니다.
복잡한 개념을 이해하기 쉽게 설명하는 것이 전문 분야입니다.

설명 원칙:
1. 학습자 수준에 맞는 용어 사용
2. 구체적인 예시와 비유 활용
3. 단계별 설명으로 이해도 증진
4. 실생활 연결을 통한 의미 부여
5. 시각적 이해를 돕는 구조화

설명 구조:
**핵심 정의**
[간단명료한 정의]

**왜 중요한가요?**
[실생활 연결 및 중요성]

**단계별 이해**
[복잡한 개념을 단계별로 분해]

**구체적 예시**
[실제 사례나 일상 비유]

**연관 개념**
[관련된 다른 개념들과의 연결]

**기억 팁**
[암기나 이해를 돕는 방법]
"""),
            ("human", "설명할 개념: {concept}\n학습자 수준: {level}\n학습 목적: {purpose}")
        ])

        # 비교 설명
        explanation_templates['comparison'] = ChatPromptTemplate.from_messages([
            ("system", f"""
당신은 {self.subject} 교육 전문가입니다.
서로 다른 개념들을 비교하여 명확한 이해를 돕는 전문가입니다.

비교 설명 구조:
**개념 소개**
[각 개념의 기본 정의]

**공통점**
[두 개념의 유사한 특성]

**차이점 분석**
[핵심 차이점들을 표로 정리]

**언제 사용하나요?**
[각 개념의 적용 상황]

**실제 예시**
[구체적 사례 비교]

**혼동 주의사항**
[자주 헷갈리는 부분과 구분법]
"""),
            ("human", "비교할 개념들: {concepts}\n비교 목적: {purpose}\n중점 사항: {focus}")
        ])

        return explanation_templates

    def generate_quiz(self, topic: str, quiz_type: str = "multiple_choice", **kwargs) -> str:
        """퀴즈 생성"""
        quiz_templates = self.create_quiz_system()

        if quiz_type not in quiz_templates:
            raise ValueError(f"지원하지 않는 퀴즈 유형: {quiz_type}")

        chain = quiz_templates[quiz_type] | self.llm | StrOutputParser()

        # 기본 매개변수 설정
        params = {
            "topic": topic,
            "difficulty": kwargs.get("difficulty", "중급"),
            "objective": kwargs.get("objective", f"{topic}에 대한 이해도 평가"),
            "constraints": kwargs.get("constraints", "특별한 제한 없음")
        }

        return chain.invoke(params)

    def explain_concept(self, concept: str, explanation_type: str = "basic_concept", **kwargs) -> str:
        """개념 설명"""
        explanation_templates = self.create_explanation_system()

        if explanation_type not in explanation_templates:
            raise ValueError(f"지원하지 않는 설명 유형: {explanation_type}")

        chain = explanation_templates[explanation_type] | self.llm | StrOutputParser()

        # 기본 매개변수 설정
        params = {
            "concept": concept,
            "concepts": kwargs.get("concepts", concept),
            "level": kwargs.get("level", "고등학교"),
            "purpose": kwargs.get("purpose", "기본 개념 이해"),
            "focus": kwargs.get("focus", "핵심 차이점")
        }

        return chain.invoke(params)

    def create_study_plan(self, topics: List[str], study_period: str) -> str:
        """학습 계획 생성"""
        study_plan_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""
당신은 {self.subject} 학습 전문 컨설턴트입니다.
효과적인 개인 맞춤 학습 계획을 수립해주세요.

학습 계획 구성요소:
1. 전체 학습 로드맵
2. 주차별 세부 계획
3. 각 주제별 학습 방법
4. 중간 점검 및 평가 방법
5. 학습 리소스 추천

출력 형식:
**학습 목표**
[전체적인 학습 성과 목표]

**학습 로드맵**
[주차별 진행 계획]

**주제별 학습 전략**
[각 주제의 효과적 학습법]

**평가 및 점검**
[중간 평가 방법과 시기]

**추천 리소스**
[교재, 온라인 자료, 실습 도구]

**학습 팁**
[동기 유지와 효율성 향상 방법]
"""),
            ("human", "학습 주제들: {topics}\n학습 기간: {period}\n현재 수준: {current_level}\n목표 수준: {target_level}")
        ])

        chain = study_plan_prompt | self.llm | StrOutputParser()

        return chain.invoke({
            "topics": ", ".join(topics),
            "period": study_period,
            "current_level": "초급",
            "target_level": "중급"
        })

# 실습 3 테스트
# 수학 학습 도우미 생성
math_assistant = LearningAssistant("수학")

# 1. 퀴즈 생성 테스트
print("=== 객관식 퀴즈 생성 ===")
math_quiz = math_assistant.generate_quiz(
    topic="이차함수",
    quiz_type="multiple_choice",
    difficulty="중급",
    objective="이차함수의 기본 성질 이해"
)
print(math_quiz)

print("\n=== 주관식 퀴즈 생성 ===")
short_answer_quiz = math_assistant.generate_quiz(
    topic="미분의 의미",
    quiz_type="short_answer",
    objective="미분의 기하학적 의미 이해",
    constraints="그래프를 그려 설명하도록 요구"
)
print(short_answer_quiz)

# 2. 개념 설명 테스트
print("\n=== 기본 개념 설명 ===")
concept_explanation = math_assistant.explain_concept(
    concept="극한",
    explanation_type="basic_concept",
    level="고등학교 3학년",
    purpose="대학 수학 준비"
)
print(concept_explanation)

print("\n=== 개념 비교 설명 ===")
comparison_explanation = math_assistant.explain_concept(
    concepts="미분과 적분",
    explanation_type="comparison",
    purpose="미적분 통합 이해",
    focus="정의와 응용의 차이점"
)
print(comparison_explanation)

# 3. 학습 계획 생성 테스트
print("\n=== 학습 계획 생성 ===")
study_topics = ["함수", "미분", "적분", "확률과 통계"]
study_plan = math_assistant.create_study_plan(
    topics=study_topics,
    study_period="12주"
)
print(study_plan)

# 4. 다른 과목 테스트 - 물리 학습 도우미
print("\n" + "="*50)
print("=== 물리 학습 도우미 테스트 ===")

physics_assistant = LearningAssistant("물리")

physics_quiz = physics_assistant.generate_quiz(
    topic="뉴턴의 운동법칙",
    quiz_type="multiple_choice",
    difficulty="기초",
    objective="물리학 기본 원리 이해"
)

print(physics_quiz)
```

## 🔍 참고 자료

### 공식 문서
- [LangChain Prompt Templates](https://python.langchain.com/docs/modules/model_io/prompts/)
- [OpenAI GPT Best Practices](https://platform.openai.com/docs/guides/gpt-best-practices)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

### 학술 자료
- Liu, P., et al. (2023). "Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods"
- Reynolds, L., & McDonell, K. (2021). "Prompt Programming for Large Language Models"
- Wei, J., et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"

### 실무 가이드
- [Prompt Engineering Best Practices](https://example.com/prompt-engineering-guide)
- [LangChain Prompt Templates Guide](https://example.com/langchain-prompts)
- [Advanced Prompting Techniques](https://example.com/advanced-prompting)

### 도구 및 리소스
- [Prompt Testing Platforms](https://example.com/prompt-testing)
- [Template Libraries](https://example.com/template-libraries)
- [Community Prompt Repositories](https://example.com/community-prompts)

---

**다음 학습**: W3_002_Prompt_Engineering_Fewshot.md - Few-Shot 프롬프팅과 in-context learning 기법을 학습합니다.