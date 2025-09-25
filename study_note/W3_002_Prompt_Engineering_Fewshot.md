# W3_002_Prompt_Engineering_Fewshot.md - Few-Shot 프롬프트 엔지니어링

## 🎯 학습 목표

- Zero-shot과 Few-shot 프롬프팅 기법의 특징과 적용 방법을 이해합니다
- One-shot, Few-shot, Dynamic Few-Shot 프롬프팅을 실습합니다
- FewShotChatMessagePromptTemplate과 SemanticSimilarityExampleSelector를 활용합니다
- 상황에 맞는 최적의 프롬프팅 전략을 선택할 수 있는 능력을 개발합니다
- 실제 업무에 적용할 수 있는 고도화된 프롬프팅 시스템을 구현합니다

## 📚 핵심 개념

### 1. 프롬프팅 기법의 분류

#### 1.1 Zero-Shot 프롬프팅
예시 없이 명확한 지시사항만으로 AI가 작업을 수행하는 기법입니다.

**특징:**
- 사용이 간단하고 직관적
- 프롬프트 길이가 짧아 비용 효율적
- 단순하고 일반적인 작업에 적합
- 복잡한 작업에서는 성능 한계 존재

**적용 시나리오:**
```python
zero_shot_examples = {
    "번역": "다음 문장을 한국어로 번역하세요: {text}",
    "요약": "다음 텍스트를 3문장으로 요약하세요: {text}",
    "분류": "다음 텍스트의 감정을 긍정/부정으로 분류하세요: {text}",
    "질의응답": "다음 질문에 답변하세요: {question}"
}
```

#### 1.2 One-Shot 프롬프팅
단일 예시를 통해 패턴을 학습시키는 기법입니다.

**특징:**
- Zero-shot보다 향상된 성능
- 형식화된 작업에 특히 효과적
- 예시 선택이 결과에 큰 영향
- 과의존 위험 존재

**구조:**
```python
one_shot_structure = """
예시:
입력: {example_input}
출력: {example_output}

이제 다음 입력을 처리하세요:
입력: {user_input}
출력:
"""
```

#### 1.3 Few-Shot 프롬프팅
2-5개의 예시를 제공하여 패턴을 학습시키는 고급 기법입니다.

**특징:**
- 가장 높은 성능과 일관성 제공
- 복잡한 작업에 효과적
- 프롬프트 길이 증가로 인한 비용 상승
- 예시 품질이 성능에 결정적 영향

**최적 예시 수:**
```python
optimal_examples = {
    "simple_classification": 2-3,  # 감정 분석, 카테고리 분류
    "complex_analysis": 3-5,       # 문서 분석, 구조화된 추출
    "creative_generation": 2-4,    # 창작, 스타일 모방
    "technical_tasks": 4-6         # 코드 생성, 기술 문서 작성
}
```

### 2. Few-Shot 프롬프팅의 고급 기법

#### 2.1 Fixed Few-Shot
미리 정의된 고정 예시를 사용하는 방식입니다.

**장점:**
- 일관된 결과 보장
- 예측 가능한 성능
- 구현이 단순

**단점:**
- 상황 적응성 부족
- 모든 케이스 커버 어려움

#### 2.2 Dynamic Few-Shot
입력 상황에 따라 가장 적절한 예시를 동적으로 선택하는 방식입니다.

**장점:**
- 상황별 최적화
- 효율적인 프롬프트 길이 관리
- 높은 범용성

**단점:**
- 구현 복잡성 증가
- 예시 선택 오버헤드
- 벡터 저장소 필요

### 3. 예시 선택 전략

#### 3.1 의미적 유사도 기반 선택
```python
similarity_selection_process = """
1. 사용자 입력 벡터화
2. 예시 데이터베이스 검색
3. 코사인 유사도 계산
4. 상위 K개 예시 선택
5. 프롬프트에 포함
"""
```

#### 3.2 카테고리 기반 선택
```python
category_selection = {
    "input_analysis": "입력 텍스트 카테고리 분류",
    "example_mapping": "카테고리별 예시 매핑",
    "selection": "해당 카테고리 예시 선택"
}
```

#### 3.3 난이도 기반 선택
```python
difficulty_based = {
    "complexity_analysis": "입력 복잡도 평가",
    "example_matching": "복잡도에 맞는 예시 선택",
    "gradual_learning": "단계별 예시 제공"
}
```

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
from typing import Dict, List, Any, Optional
from textwrap import dedent

# LangChain 관련
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.prompts import FewShotChatMessagePromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser

# 환경 변수 로드
load_dotenv()

# LLM 초기화
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.3,
    top_p=0.9
)
```

## 💻 코드 예제

### 1. Zero-Shot vs One-Shot vs Few-Shot 비교

```python
class PromptingMethodComparator:
    def __init__(self, model_name: str = "gpt-4.1-mini"):
        """프롬프팅 기법 비교 클래스"""
        self.llm = ChatOpenAI(model=model_name, temperature=0.3)

    def create_zero_shot_prompt(self, task_description: str, input_variable: str = "input") -> PromptTemplate:
        """Zero-shot 프롬프트 생성"""
        return PromptTemplate(
            input_variables=[input_variable],
            template=f"{task_description}: {{{input_variable}}}"
        )

    def create_one_shot_prompt(
        self,
        task_description: str,
        example_input: str,
        example_output: str,
        input_variable: str = "input"
    ) -> PromptTemplate:
        """One-shot 프롬프트 생성"""
        template = f"""다음은 {task_description}의 예시입니다:

입력: {example_input}
출력: {example_output}

이제 다음을 처리하세요:
입력: {{{input_variable}}}
출력:"""

        return PromptTemplate(
            input_variables=[input_variable],
            template=template
        )

    def create_few_shot_prompt(
        self,
        task_description: str,
        examples: List[Dict[str, str]],
        input_variable: str = "input"
    ) -> PromptTemplate:
        """Few-shot 프롬프트 생성"""
        examples_text = ""
        for i, example in enumerate(examples, 1):
            examples_text += f"\n예시 {i}:\n입력: {example['input']}\n출력: {example['output']}\n"

        template = f"""다음은 {task_description}의 예시들입니다:
{examples_text}
이제 다음을 처리하세요:
입력: {{{input_variable}}}
출력:"""

        return PromptTemplate(
            input_variables=[input_variable],
            template=template
        )

    def compare_methods(
        self,
        task_description: str,
        test_input: str,
        example_input: str,
        example_output: str,
        few_shot_examples: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """세 가지 방법 비교 실행"""

        # Zero-shot
        zero_shot_prompt = self.create_zero_shot_prompt(task_description)
        zero_shot_chain = zero_shot_prompt | self.llm | StrOutputParser()
        zero_shot_result = zero_shot_chain.invoke({"input": test_input})

        # One-shot
        one_shot_prompt = self.create_one_shot_prompt(
            task_description, example_input, example_output
        )
        one_shot_chain = one_shot_prompt | self.llm | StrOutputParser()
        one_shot_result = one_shot_chain.invoke({"input": test_input})

        # Few-shot
        few_shot_prompt = self.create_few_shot_prompt(task_description, few_shot_examples)
        few_shot_chain = few_shot_prompt | self.llm | StrOutputParser()
        few_shot_result = few_shot_chain.invoke({"input": test_input})

        return {
            "test_input": test_input,
            "results": {
                "zero_shot": zero_shot_result,
                "one_shot": one_shot_result,
                "few_shot": few_shot_result
            },
            "prompt_lengths": {
                "zero_shot": len(zero_shot_prompt.template),
                "one_shot": len(one_shot_prompt.template),
                "few_shot": len(few_shot_prompt.template)
            }
        }

# 사용 예시
comparator = PromptingMethodComparator()

# 감정 분석 비교
sentiment_examples = [
    {"input": "이 제품 정말 만족스럽고 품질이 뛰어나요!", "output": "긍정"},
    {"input": "서비스가 너무 느리고 불친절합니다.", "output": "부정"},
    {"input": "이 음식은 맛있고 가격도 합리적입니다.", "output": "긍정"},
    {"input": "배송이 지연되어 매우 실망스럽습니다.", "output": "부정"}
]

comparison_result = comparator.compare_methods(
    task_description="텍스트의 감정을 긍정 또는 부정으로 분류",
    test_input="제품이 기대보다 훌륭하고 추천하고 싶어요!",
    example_input=sentiment_examples[0]["input"],
    example_output=sentiment_examples[0]["output"],
    few_shot_examples=sentiment_examples[:3]
)

print("=== 프롬프팅 방법 비교 결과 ===")
print(f"테스트 입력: {comparison_result['test_input']}")
print("\n결과:")
for method, result in comparison_result['results'].items():
    print(f"- {method}: {result}")

print("\n프롬프트 길이:")
for method, length in comparison_result['prompt_lengths'].items():
    print(f"- {method}: {length} characters")
```

### 2. 고정 Few-Shot 프롬프트 시스템

```python
class FixedFewShotSystem:
    def __init__(self, model_name: str = "gpt-4.1-mini"):
        """고정 Few-Shot 시스템"""
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)

    def create_news_keyword_extractor(self) -> ChatPromptTemplate:
        """뉴스 키워드 추출기 생성"""
        examples = [
            {
                "input": dedent("""
                    정부는 의과대학 입학 정원을 2000명 증가시킬 계획의 세부사항을 이달 20일에 공개할 예정이다.
                    지역별 의료 서비스 향상과 소규모 의과대학의 발전을 목표로, 지역 중심의 국립대학 및 소형 의과대학의
                    입학 정원이 최소한 두 배 가량 확대될 것으로 보인다.
                """),
                "output": "의대 | 정원 | 확대"
            },
            {
                "input": dedent("""
                    세계보건기구(WHO)는 최근 새로운 건강 위기에 대응하기 위해 국제 협력의 중요성을 강조했다.
                    전염병 대응 역량의 강화와 글로벌 보건 시스템의 개선이 필요하다고 발표했다.
                """),
                "output": "세계보건기구 | 건강위기 | 국제협력"
            },
            {
                "input": dedent("""
                    삼성전자가 내년 초에 자체적으로 개발한 인공지능(AI) 가속기를 처음으로 출시할 예정이다.
                    이는 AI 반도체 시장에서 지배적인 위치를 차지하고 있는 엔비디아의 독점을 도전하려는 시도이다.
                """),
                "output": "삼성전자 | AI가속기 | 반도체"
            }
        ]

        # 예시 프롬프트 템플릿
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{input}"),
            ("assistant", "{output}")
        ])

        # Few-shot 프롬프트 템플릿
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples
        )

        # 최종 프롬프트
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", """
당신은 뉴스 텍스트에서 핵심 키워드 3개를 추출하는 전문가입니다.
키워드는 '|'로 구분하여 제시하며, 각 키워드는 뉴스의 핵심 내용을 대표해야 합니다.
"""),
            few_shot_prompt,
            ("human", "{input}")
        ])

        return final_prompt

    def create_competitor_analyzer(self) -> PromptTemplate:
        """경쟁사 분석기 생성"""
        examples = """
시장: 스마트폰
경쟁업체:
- 애플(미국): 프리미엄 시장 주도, iPhone으로 경쟁
- 샤오미(중국): 중저가 시장 강세, 글로벌 확장 중
- 구글(미국): Pixel로 AI 기능 강조

시장: TV
경쟁업체:
- LG전자(한국): OLED 기술로 프리미엄 시장 경쟁
- Sony(일본): 고품질 디스플레이 기술 경쟁
- TCL(중국): 중저가 시장 공략

시장: 메모리 반도체
경쟁업체:
- SK하이닉스(한국): DRAM과 NAND 플래시 경쟁
- 마이크론(미국): 메모리 솔루션 전 분야 경쟁
- 키옥시아(일본): NAND 플래시 시장 경쟁
"""

        return PromptTemplate(
            input_variables=["market"],
            template=f"""다음은 여러 시장에서 삼성전자의 경쟁업체를 분석한 예시들입니다:

{examples}

이제 다음 시장에서 삼성전자의 경쟁업체를 분석해주세요:
시장: {{market}}
경쟁업체:"""
        )

    def execute_keyword_extraction(self, news_text: str) -> str:
        """키워드 추출 실행"""
        prompt = self.create_news_keyword_extractor()
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"input": news_text})

    def execute_competitor_analysis(self, market: str) -> str:
        """경쟁사 분석 실행"""
        prompt = self.create_competitor_analyzer()
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"market": market})

# 사용 예시
fixed_system = FixedFewShotSystem()

# 키워드 추출 테스트
news_text = dedent("""
    네이버가 새로운 AI 검색 서비스를 출시하며 구글과의 경쟁을 본격화한다고 발표했다.
    이 서비스는 자연어 처리 기술을 활용하여 사용자의 질문에 더 정확한 답변을 제공할 예정이다.
    네이버는 이를 통해 국내 검색 시장에서의 점유율을 더욱 확대할 계획이라고 밝혔다.
""")

keywords = fixed_system.execute_keyword_extraction(news_text)
print("=== 키워드 추출 결과 ===")
print(keywords)

# 경쟁사 분석 테스트
competitor_analysis = fixed_system.execute_competitor_analysis("인공지능 반도체")
print("\n=== 경쟁사 분석 결과 ===")
print(competitor_analysis)
```

### 3. 동적 Few-Shot 프롬프트 시스템

```python
class DynamicFewShotSystem:
    def __init__(self, model_name: str = "gpt-4.1-mini", embedding_model: str = "bge-m3"):
        """동적 Few-Shot 시스템"""
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)
        self.embeddings = OllamaEmbeddings(model=embedding_model)

    def create_customer_service_bot(self, examples: List[Dict[str, str]]) -> ChatPromptTemplate:
        """고객 서비스 봇 생성"""

        # 예시를 벡터화할 텍스트로 변환
        to_vectorize = [f"{example['input']} {example['output']}" for example in examples]

        # 벡터 스토어 생성
        vector_store = InMemoryVectorStore.from_texts(
            to_vectorize,
            self.embeddings,
            metadatas=examples
        )

        # 의미적 유사도 기반 예시 선택기
        example_selector = SemanticSimilarityExampleSelector(
            vectorstore=vector_store,
            k=2  # 상위 2개 예시 선택
        )

        # Few-shot 프롬프트 템플릿
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            input_variables=["input"],
            example_selector=example_selector,
            example_prompt=ChatPromptTemplate.from_messages([
                ("human", "{input}"),
                ("assistant", "{output}")
            ])
        )

        # 최종 프롬프트
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", """
당신은 친절하고 전문적인 고객 서비스 담당자입니다.
고객의 문의사항에 대해 정확하고 도움이 되는 응답을 제공해주세요.
"""),
            few_shot_prompt,
            ("human", "{input}")
        ])

        return final_prompt, example_selector

    def create_technical_support_system(self) -> tuple:
        """기술 지원 시스템 생성"""
        technical_examples = [
            {
                "input": "인터넷이 계속 끊어져요",
                "output": "네트워크 연결 문제 해결을 도와드리겠습니다.\n1. 공유기를 30초간 끄고 다시 켜주세요\n2. 네트워크 케이블 연결 상태를 확인해주세요\n3. 문제가 지속되면 ISP에 문의하시거나 기술지원팀으로 연락해주세요."
            },
            {
                "input": "프로그램이 실행되지 않아요",
                "output": "프로그램 실행 문제를 해결해보겠습니다.\n1. 프로그램을 완전히 종료 후 재시작해주세요\n2. 관리자 권한으로 실행해보세요\n3. 호환성 모드로 실행해보세요\n4. 문제가 계속되면 프로그램을 재설치해주세요."
            },
            {
                "input": "컴퓨터가 너무 느려요",
                "output": "컴퓨터 성능 향상을 위한 조치를 안내해드리겠습니다.\n1. 작업 관리자에서 CPU/메모리 사용률을 확인해주세요\n2. 불필요한 시작프로그램을 비활성화해주세요\n3. 디스크 정리를 수행해주세요\n4. 필요시 메모리 업그레이드를 고려해보세요."
            },
            {
                "input": "비밀번호를 잊어버렸어요",
                "output": "비밀번호 재설정을 도와드리겠습니다.\n1. 로그인 페이지에서 '비밀번호 찾기'를 클릭해주세요\n2. 등록된 이메일 주소를 입력해주세요\n3. 이메일로 발송된 재설정 링크를 클릭해주세요\n4. 새로운 비밀번호를 설정해주세요."
            },
            {
                "input": "파일이 삭제되었어요",
                "output": "삭제된 파일 복구를 시도해보겠습니다.\n1. 휴지통에서 파일을 확인해주세요\n2. 시스템 복원 기능을 이용해보세요\n3. 파일 히스토리나 백업에서 복원해보세요\n4. 전문 복구 소프트웨어 사용을 고려해보세요."
            },
            {
                "input": "화면이 깜빡거려요",
                "output": "화면 깜빡임 문제 해결을 도와드리겠습니다.\n1. 모니터 케이블 연결 상태를 확인해주세요\n2. 화면 해상도와 주사율을 조정해보세요\n3. 그래픽 드라이버를 업데이트해주세요\n4. 모니터 설정에서 자동 조정을 실행해보세요."
            }
        ]

        return self.create_customer_service_bot(technical_examples)

    def create_product_support_system(self) -> tuple:
        """제품 지원 시스템 생성"""
        product_examples = [
            {
                "input": "환불하고 싶어요",
                "output": "환불 절차를 안내해드리겠습니다.\n1. 마이페이지에서 주문 내역을 확인해주세요\n2. '환불 신청' 버튼을 클릭해주세요\n3. 환불 사유를 선택하고 상세 내용을 입력해주세요\n4. 상품을 원래 포장 상태로 반송해주세요\n5. 상품 확인 후 3-5일 내 환불 처리됩니다."
            },
            {
                "input": "교환하고 싶어요",
                "output": "교환 절차를 안내해드리겠습니다.\n1. 14일 이내 교환 신청이 가능합니다\n2. 상품이 미사용 상태여야 합니다\n3. 교환 신청서를 작성해주세요\n4. 교환할 상품의 재고를 확인해드리겠습니다\n5. 교환 상품 발송 후 기존 상품을 회수합니다."
            },
            {
                "input": "배송이 지연되고 있어요",
                "output": "배송 지연으로 불편을 드려 죄송합니다.\n1. 주문번호를 확인해주시면 배송 상태를 조회해드리겠습니다\n2. 택배사 사정으로 지연될 수 있습니다\n3. 예상 배송일을 재안내해드리겠습니다\n4. 추가 지연 시 즉시 연락드리겠습니다."
            },
            {
                "input": "사이즈가 안 맞아요",
                "output": "사이즈 불만족으로 불편을 드려 죄송합니다.\n1. 무료 사이즈 교환이 가능합니다\n2. 사이즈 가이드를 참고하여 올바른 사이즈를 선택해주세요\n3. 교환 신청 후 새 상품 먼저 발송해드릴 수 있습니다\n4. 기존 상품은 새 상품 수령 후 반송해주세요."
            },
            {
                "input": "제품에 하자가 있어요",
                "output": "제품 하자로 불편을 드려 대단히 죄송합니다.\n1. 하자 부분의 사진을 찍어 1:1 문의로 보내주세요\n2. 즉시 새 제품으로 교체해드리겠습니다\n3. 배송비는 저희가 부담하겠습니다\n4. 추가 피해나 불편 사항이 있으시면 말씀해주세요."
            }
        ]

        return self.create_customer_service_bot(product_examples)

    def test_dynamic_selection(
        self,
        prompt_template: ChatPromptTemplate,
        example_selector: SemanticSimilarityExampleSelector,
        test_inputs: List[str]
    ) -> Dict[str, Any]:
        """동적 선택 테스트"""
        results = {}

        for test_input in test_inputs:
            # 선택된 예시 확인
            selected_examples = example_selector.select_examples({"input": test_input})

            # 응답 생성
            chain = prompt_template | self.llm | StrOutputParser()
            response = chain.invoke({"input": test_input})

            results[test_input] = {
                "selected_examples": selected_examples,
                "response": response
            }

        return results

# 사용 예시
dynamic_system = DynamicFewShotSystem()

# 기술 지원 시스템 생성
tech_prompt, tech_selector = dynamic_system.create_technical_support_system()

# 제품 지원 시스템 생성
product_prompt, product_selector = dynamic_system.create_product_support_system()

# 기술 지원 테스트
tech_test_inputs = [
    "컴퓨터가 부팅이 안 돼요",
    "와이파이 연결이 안 돼요",
    "프로그램이 자꾸 멈춰요"
]

tech_results = dynamic_system.test_dynamic_selection(
    tech_prompt, tech_selector, tech_test_inputs
)

print("=== 기술 지원 시스템 테스트 ===")
for test_input, result in tech_results.items():
    print(f"\n입력: {test_input}")
    print("선택된 예시:")
    for example in result["selected_examples"]:
        print(f"  - {example['input']}")
    print(f"응답: {result['response']}")

# 제품 지원 테스트
product_test_inputs = [
    "배송 상품이 파손되었어요",
    "주문을 취소하고 싶어요",
    "다른 색상으로 바꾸고 싶어요"
]

product_results = dynamic_system.test_dynamic_selection(
    product_prompt, product_selector, product_test_inputs
)

print("\n\n=== 제품 지원 시스템 테스트 ===")
for test_input, result in product_results.items():
    print(f"\n입력: {test_input}")
    print("선택된 예시:")
    for example in result["selected_examples"]:
        print(f"  - {example['input']}")
    print(f"응답: {result['response']}")
```

### 4. 성능 최적화 Few-Shot 시스템

```python
class OptimizedFewShotSystem:
    def __init__(self, model_name: str = "gpt-4.1-mini"):
        """최적화된 Few-Shot 시스템"""
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)
        self.example_cache = {}
        self.performance_metrics = {}

    def create_adaptive_example_selector(
        self,
        examples: List[Dict[str, str]],
        max_examples: int = 3,
        min_similarity: float = 0.5
    ) -> SemanticSimilarityExampleSelector:
        """적응적 예시 선택기 생성"""

        # 캐시 키 생성
        cache_key = f"examples_{len(examples)}_{max_examples}"

        if cache_key in self.example_cache:
            return self.example_cache[cache_key]

        # 벡터화
        to_vectorize = [f"{ex['input']} {ex['output']}" for ex in examples]

        embeddings = OllamaEmbeddings(model="bge-m3")
        vector_store = InMemoryVectorStore.from_texts(
            to_vectorize, embeddings, metadatas=examples
        )

        # 선택기 생성
        selector = SemanticSimilarityExampleSelector(
            vectorstore=vector_store,
            k=max_examples
        )

        # 캐시 저장
        self.example_cache[cache_key] = selector

        return selector

    def create_context_aware_prompt(
        self,
        task_name: str,
        system_message: str,
        examples: List[Dict[str, str]],
        context_variables: List[str] = None
    ) -> tuple:
        """문맥 인식 프롬프트 생성"""

        # 예시 선택기 생성
        selector = self.create_adaptive_example_selector(examples)

        # Few-shot 프롬프트
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            input_variables=["input"],
            example_selector=selector,
            example_prompt=ChatPromptTemplate.from_messages([
                ("human", "{input}"),
                ("assistant", "{output}")
            ])
        )

        # 문맥 변수가 있는 경우
        if context_variables:
            messages = [("system", system_message)]

            # 문맥 정보 추가
            for var in context_variables:
                messages.append(("system", f"{var}: {{{var}}}"))

            messages.extend([few_shot_prompt, ("human", "{input}")])
        else:
            messages = [
                ("system", system_message),
                few_shot_prompt,
                ("human", "{input}")
            ]

        final_prompt = ChatPromptTemplate.from_messages(messages)

        return final_prompt, selector

    def create_multilingual_translator(self) -> tuple:
        """다국어 번역기 생성"""
        translation_examples = [
            {
                "input": "Hello, how are you? | Korean",
                "output": "안녕하세요, 어떻게 지내세요?"
            },
            {
                "input": "Thank you for your help | Korean",
                "output": "도움을 주셔서 감사합니다"
            },
            {
                "input": "안녕하세요, 만나서 반갑습니다 | English",
                "output": "Hello, nice to meet you"
            },
            {
                "input": "오늘 날씨가 정말 좋네요 | English",
                "output": "The weather is really nice today"
            },
            {
                "input": "¿Cómo está usted? | Korean",
                "output": "어떻게 지내세요?"
            },
            {
                "input": "Gracias por todo | Korean",
                "output": "모든 것에 감사합니다"
            },
            {
                "input": "Je suis très content | Korean",
                "output": "저는 매우 기쁩니다"
            },
            {
                "input": "안녕히 가세요 | Spanish",
                "output": "Adiós, que tenga un buen día"
            }
        ]

        system_message = """
당신은 전문 번역가입니다. 주어진 텍스트를 정확하고 자연스럽게 번역해주세요.
입력 형식: [번역할 텍스트] | [목표 언어]
번역 시 문화적 맥락과 뉘앙스를 고려하여 자연스러운 번역을 제공해주세요.
"""

        return self.create_context_aware_prompt(
            "multilingual_translation",
            system_message,
            translation_examples
        )

    def create_sentiment_analyzer_with_confidence(self) -> tuple:
        """신뢰도가 포함된 감정 분석기 생성"""
        sentiment_examples = [
            {
                "input": "이 제품 정말 훌륭하고 만족스럽습니다!",
                "output": "감정: 긍정 | 신뢰도: 95% | 근거: '훌륭하고', '만족스럽습니다' 등 강한 긍정 표현"
            },
            {
                "input": "서비스가 엉망이고 직원들도 불친절해요",
                "output": "감정: 부정 | 신뢰도: 90% | 근거: '엉망', '불친절' 등 명확한 부정 표현"
            },
            {
                "input": "그냥 그저 그런 것 같아요",
                "output": "감정: 중립 | 신뢰도: 85% | 근거: '그저 그런' 등 중립적 표현"
            },
            {
                "input": "가격은 비싸지만 품질은 좋네요",
                "output": "감정: 중립 | 신뢰도: 80% | 근거: 긍정('품질 좋음')과 부정('비쌈') 요소 혼재"
            },
            {
                "input": "최고예요! 다시 구매할 의향이 있습니다",
                "output": "감정: 긍정 | 신뢰도: 98% | 근거: '최고', '다시 구매' 등 매우 강한 긍정 의도"
            },
            {
                "input": "완전 실망했어요. 돈만 버렸네요",
                "output": "감정: 부정 | 신뢰도: 95% | 근거: '완전 실망', '돈만 버렸다' 등 강한 부정 감정"
            }
        ]

        system_message = """
당신은 텍스트 감정 분석 전문가입니다.
입력된 텍스트의 감정을 분석하고 다음 형식으로 응답해주세요:

감정: [긍정/부정/중립]
신뢰도: [0-100%]
근거: [판단 근거와 핵심 단어/구문]

감정 판단 기준:
- 긍정: 만족, 기쁨, 칭찬 등의 긍정적 표현
- 부정: 불만, 실망, 비판 등의 부정적 표현
- 중립: 객관적 서술이나 긍정/부정 요소가 혼재된 경우
"""

        return self.create_context_aware_prompt(
            "sentiment_analysis",
            system_message,
            sentiment_examples
        )

    def measure_performance(
        self,
        prompt_template: ChatPromptTemplate,
        test_cases: List[Dict[str, str]],
        task_name: str
    ) -> Dict[str, Any]:
        """성능 측정"""
        import time

        chain = prompt_template | self.llm | StrOutputParser()

        start_time = time.time()
        results = []

        for test_case in test_cases:
            case_start = time.time()
            try:
                result = chain.invoke({"input": test_case["input"]})
                case_time = time.time() - case_start

                results.append({
                    "input": test_case["input"],
                    "output": result,
                    "expected": test_case.get("expected", ""),
                    "response_time": case_time,
                    "success": True
                })
            except Exception as e:
                results.append({
                    "input": test_case["input"],
                    "error": str(e),
                    "response_time": time.time() - case_start,
                    "success": False
                })

        total_time = time.time() - start_time
        success_rate = sum(1 for r in results if r["success"]) / len(results)
        avg_response_time = sum(r["response_time"] for r in results) / len(results)

        performance_report = {
            "task_name": task_name,
            "total_test_cases": len(test_cases),
            "success_rate": success_rate,
            "total_time": total_time,
            "average_response_time": avg_response_time,
            "results": results
        }

        self.performance_metrics[task_name] = performance_report
        return performance_report

# 사용 예시
optimized_system = OptimizedFewShotSystem()

# 다국어 번역기 생성 및 테스트
translator_prompt, translator_selector = optimized_system.create_multilingual_translator()

translation_test_cases = [
    {"input": "Good morning, have a nice day! | Korean"},
    {"input": "감사합니다, 좋은 하루 되세요 | English"},
    {"input": "¿Dónde está el baño? | Korean"},
    {"input": "오늘 회의가 몇 시에 있나요? | Spanish"}
]

print("=== 다국어 번역 테스트 ===")
translation_chain = translator_prompt | optimized_system.llm | StrOutputParser()

for test_case in translation_test_cases:
    result = translation_chain.invoke({"input": test_case["input"]})
    print(f"입력: {test_case['input']}")
    print(f"번역: {result}\n")

# 감정 분석기 생성 및 성능 측정
sentiment_prompt, sentiment_selector = optimized_system.create_sentiment_analyzer_with_confidence()

sentiment_test_cases = [
    {
        "input": "정말 완벽한 서비스였어요! 강력 추천합니다",
        "expected": "긍정"
    },
    {
        "input": "최악의 경험이었습니다. 다신 이용 안 할게요",
        "expected": "부정"
    },
    {
        "input": "보통 수준이네요. 나쁘지도 좋지도 않고",
        "expected": "중립"
    },
    {
        "input": "배송은 빨랐는데 품질이 아쉬워요",
        "expected": "중립"
    },
    {
        "input": "와! 대박이에요! 정말 마음에 들어요",
        "expected": "긍정"
    }
]

# 성능 측정 실행
performance_report = optimized_system.measure_performance(
    sentiment_prompt,
    sentiment_test_cases,
    "sentiment_analysis_with_confidence"
)

print("=== 감정 분석 성능 보고서 ===")
print(f"작업명: {performance_report['task_name']}")
print(f"테스트 케이스 수: {performance_report['total_test_cases']}")
print(f"성공률: {performance_report['success_rate']:.2%}")
print(f"평균 응답 시간: {performance_report['average_response_time']:.2f}초")
print(f"전체 실행 시간: {performance_report['total_time']:.2f}초")

print("\n상세 결과:")
for i, result in enumerate(performance_report['results'], 1):
    if result['success']:
        print(f"{i}. {result['input']}")
        print(f"   결과: {result['output']}")
        print(f"   시간: {result['response_time']:.2f}초\n")
```

## 🚀 실습해보기

### 실습 1: 감정 분석기 구현

Zero-shot과 Few-shot 방식으로 감정 분석기를 구현하고 성능을 비교해보세요.

```python
# 감정 분석기 비교 시스템 구현
class SentimentAnalysisComparator:
    def __init__(self):
        # 초기화 코드 작성
        pass

    def create_zero_shot_analyzer(self):
        # Zero-shot 감정 분석기 구현
        pass

    def create_few_shot_analyzer(self):
        # Few-shot 감정 분석기 구현
        pass

    def compare_performance(self, test_texts):
        # 성능 비교 분석
        pass
```

### 실습 2: 동적 고객 서비스 봇

고객 문의 유형에 따라 적절한 예시를 선택하는 동적 시스템을 구현해보세요.

```python
# 동적 고객 서비스 봇 구현
class DynamicCustomerServiceBot:
    def __init__(self):
        # 초기화 코드 작성
        pass

    def setup_example_database(self):
        # 예시 데이터베이스 구축
        pass

    def create_dynamic_selector(self):
        # 동적 예시 선택기 구현
        pass

    def handle_customer_inquiry(self, inquiry):
        # 고객 문의 처리
        pass
```

### 실습 3: 멀티태스크 Few-Shot 시스템

하나의 시스템에서 여러 작업을 처리할 수 있는 Few-Shot 시스템을 구현해보세요.

```python
# 멀티태스크 Few-Shot 시스템 구현
class MultiTaskFewShotSystem:
    def __init__(self):
        # 초기화 코드 작성
        pass

    def setup_task_router(self):
        # 작업 유형 라우터 구현
        pass

    def create_task_specific_prompts(self):
        # 작업별 특화 프롬프트 생성
        pass

    def execute_task(self, input_text, task_type):
        # 작업 실행
        pass
```

## 📋 해답

### 실습 1: 감정 분석기 구현

```python
class SentimentAnalysisComparator:
    def __init__(self, model_name: str = "gpt-4.1-mini"):
        """감정 분석기 비교 시스템"""
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)

    def create_zero_shot_analyzer(self) -> PromptTemplate:
        """Zero-shot 감정 분석기 구현"""
        return PromptTemplate(
            input_variables=["text"],
            template="""
다음 텍스트의 감정을 분석하고 긍정, 부정, 중립 중 하나로 분류해주세요.
또한 판단 근거를 함께 제시해주세요.

텍스트: {text}

형식:
감정: [긍정/부정/중립]
근거: [판단 이유]
"""
        )

    def create_few_shot_analyzer(self) -> PromptTemplate:
        """Few-shot 감정 분석기 구현"""
        examples = """
예시 1:
텍스트: 이 제품 정말 마음에 들어요! 품질도 좋고 가격도 합리적입니다.
감정: 긍정
근거: '마음에 들어요', '품질도 좋고', '합리적' 등 만족감과 긍정적 평가 표현

예시 2:
텍스트: 서비스가 너무 느리고 직원들도 불친절해서 실망했습니다.
감정: 부정
근거: '너무 느리고', '불친절', '실망' 등 불만과 부정적 경험 표현

예시 3:
텍스트: 그냥 보통 수준이네요. 특별히 좋지도 나쁘지도 않아요.
감정: 중립
근거: '보통 수준', '좋지도 나쁘지도 않다' 등 중립적이고 객관적인 표현

예시 4:
텍스트: 가격은 비싸지만 그래도 품질은 괜찮은 편입니다.
감정: 중립
근거: 부정 요소('비싸다')와 긍정 요소('품질 괜찮다')가 균형을 이루는 표현
"""

        return PromptTemplate(
            input_variables=["text"],
            template=f"""
다음은 텍스트 감정 분석의 예시들입니다:

{examples}

이제 다음 텍스트의 감정을 분석해주세요:

텍스트: {{text}}

형식:
감정: [긍정/부정/중립]
근거: [판단 이유]
"""
        )

    def extract_sentiment(self, analysis_result: str) -> str:
        """분석 결과에서 감정만 추출"""
        lines = analysis_result.split('\n')
        for line in lines:
            if line.startswith('감정:'):
                return line.replace('감정:', '').strip()
        return "알 수 없음"

    def compare_performance(self, test_texts: List[Dict[str, str]]) -> Dict[str, Any]:
        """성능 비교 분석"""
        zero_shot_prompt = self.create_zero_shot_analyzer()
        few_shot_prompt = self.create_few_shot_analyzer()

        zero_shot_chain = zero_shot_prompt | self.llm | StrOutputParser()
        few_shot_chain = few_shot_prompt | self.llm | StrOutputParser()

        results = {
            "test_cases": len(test_texts),
            "zero_shot": {"correct": 0, "total": 0, "results": []},
            "few_shot": {"correct": 0, "total": 0, "results": []},
            "detailed_comparison": []
        }

        for test_case in test_texts:
            text = test_case["text"]
            expected = test_case["expected"]

            # Zero-shot 분석
            zero_shot_result = zero_shot_chain.invoke({"text": text})
            zero_shot_sentiment = self.extract_sentiment(zero_shot_result)
            zero_shot_correct = zero_shot_sentiment.lower() == expected.lower()

            # Few-shot 분석
            few_shot_result = few_shot_chain.invoke({"text": text})
            few_shot_sentiment = self.extract_sentiment(few_shot_result)
            few_shot_correct = few_shot_sentiment.lower() == expected.lower()

            # 결과 기록
            results["zero_shot"]["total"] += 1
            results["few_shot"]["total"] += 1

            if zero_shot_correct:
                results["zero_shot"]["correct"] += 1
            if few_shot_correct:
                results["few_shot"]["correct"] += 1

            results["zero_shot"]["results"].append({
                "text": text,
                "predicted": zero_shot_sentiment,
                "expected": expected,
                "correct": zero_shot_correct,
                "full_result": zero_shot_result
            })

            results["few_shot"]["results"].append({
                "text": text,
                "predicted": few_shot_sentiment,
                "expected": expected,
                "correct": few_shot_correct,
                "full_result": few_shot_result
            })

            results["detailed_comparison"].append({
                "text": text,
                "expected": expected,
                "zero_shot": zero_shot_sentiment,
                "few_shot": few_shot_sentiment,
                "zero_shot_correct": zero_shot_correct,
                "few_shot_correct": few_shot_correct
            })

        # 정확도 계산
        results["zero_shot"]["accuracy"] = results["zero_shot"]["correct"] / results["zero_shot"]["total"]
        results["few_shot"]["accuracy"] = results["few_shot"]["correct"] / results["few_shot"]["total"]

        return results

# 실습 1 테스트
sentiment_comparator = SentimentAnalysisComparator()

test_cases = [
    {"text": "이 상품 정말 완벽해요! 강력 추천드립니다!", "expected": "긍정"},
    {"text": "최악의 서비스였어요. 돈만 아깝습니다.", "expected": "부정"},
    {"text": "그냥 그저 그런 평범한 제품이네요.", "expected": "중립"},
    {"text": "가격은 비싸지만 품질만큼은 확실해요.", "expected": "중립"},
    {"text": "완전 대박! 진짜 마음에 들어요!", "expected": "긍정"},
    {"text": "실망스럽고 화가 나네요.", "expected": "부정"},
    {"text": "배송은 빨랐는데 포장이 조금 아쉬워요.", "expected": "중립"},
    {"text": "없어서는 안 될 필수템이에요!", "expected": "긍정"},
    {"text": "환불하고 싶을 정도로 후회됩니다.", "expected": "부정"},
    {"text": "보통 수준이라고 생각합니다.", "expected": "중립"}
]

comparison_results = sentiment_comparator.compare_performance(test_cases)

print("=== 감정 분석기 성능 비교 ===")
print(f"테스트 케이스 수: {comparison_results['test_cases']}")
print(f"Zero-shot 정확도: {comparison_results['zero_shot']['accuracy']:.2%}")
print(f"Few-shot 정확도: {comparison_results['few_shot']['accuracy']:.2%}")

print("\n=== 상세 비교 결과 ===")
for i, comparison in enumerate(comparison_results['detailed_comparison'], 1):
    print(f"{i}. {comparison['text']}")
    print(f"   예상: {comparison['expected']}")
    print(f"   Zero-shot: {comparison['zero_shot']} {'✓' if comparison['zero_shot_correct'] else '✗'}")
    print(f"   Few-shot: {comparison['few_shot']} {'✓' if comparison['few_shot_correct'] else '✗'}")
    print()
```

### 실습 2: 동적 고객 서비스 봇

```python
class DynamicCustomerServiceBot:
    def __init__(self, model_name: str = "gpt-4.1-mini"):
        """동적 고객 서비스 봇"""
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)
        self.embeddings = OllamaEmbeddings(model="bge-m3")
        self.example_selectors = {}

    def setup_example_database(self) -> Dict[str, List[Dict[str, str]]]:
        """예시 데이터베이스 구축"""
        return {
            "product_inquiry": [
                {
                    "input": "이 제품의 사양이 어떻게 되나요?",
                    "output": "제품 사양을 상세히 안내해드리겠습니다. 구체적으로 어떤 부분이 궁금하신지 말씀해주시면 더 정확한 정보를 제공할 수 있습니다."
                },
                {
                    "input": "재고가 있나요?",
                    "output": "재고 현황을 확인해드리겠습니다. 상품명이나 상품 코드를 알려주시면 즉시 확인 가능합니다."
                },
                {
                    "input": "색상 옵션이 있나요?",
                    "output": "해당 제품의 색상 옵션을 확인해드리겠습니다. 현재 이용 가능한 색상과 재고 상황을 함께 안내해드리겠습니다."
                }
            ],
            "shipping_inquiry": [
                {
                    "input": "언제 배송되나요?",
                    "output": "주문번호를 알려주시면 정확한 배송 예정일을 확인해드리겠습니다. 일반적으로 주문 후 1-2일 내 배송됩니다."
                },
                {
                    "input": "배송비는 얼마인가요?",
                    "output": "배송비는 지역과 주문 금액에 따라 달라집니다. 50,000원 이상 주문 시 무료배송이며, 그 이하는 2,500원입니다."
                },
                {
                    "input": "배송 추적을 하고 싶어요",
                    "output": "운송장 번호를 안내해드리겠습니다. 주문번호나 주문자명을 알려주시면 배송 상태를 확인할 수 있습니다."
                }
            ],
            "return_exchange": [
                {
                    "input": "환불하고 싶어요",
                    "output": "환불 절차를 안내해드리겠습니다. 구매일로부터 7일 이내 신청 가능하며, 상품 상태 확인 후 처리됩니다."
                },
                {
                    "input": "교환 가능한가요?",
                    "output": "교환은 구매일로부터 7일 이내, 미사용 상태에서 가능합니다. 교환 사유와 희망 상품을 알려주세요."
                },
                {
                    "input": "사이즈가 안 맞아요",
                    "output": "사이즈 불일치로 불편을 드려 죄송합니다. 무료 사이즈 교환이 가능하니 희망하는 사이즈를 알려주세요."
                }
            ],
            "technical_support": [
                {
                    "input": "설치 방법을 알려주세요",
                    "output": "제품 설치 방법을 단계별로 안내해드리겠습니다. 제품 매뉴얼과 함께 자세한 설명을 제공해드릴게요."
                },
                {
                    "input": "작동이 안 돼요",
                    "output": "문제 상황을 파악해보겠습니다. 어떤 상황에서 작동하지 않는지 구체적으로 설명해주시겠어요?"
                },
                {
                    "input": "오류가 발생해요",
                    "output": "오류 메시지나 상황을 자세히 알려주세요. 단계별 해결 방법을 안내해드리겠습니다."
                }
            ]
        }

    def classify_inquiry_type(self, inquiry: str) -> str:
        """문의 유형 분류"""

        # 키워드 기반 분류
        classification_keywords = {
            "product_inquiry": ["제품", "상품", "사양", "기능", "특징", "재고", "색상", "옵션"],
            "shipping_inquiry": ["배송", "택배", "언제", "빨리", "운송", "도착", "배송비"],
            "return_exchange": ["환불", "교환", "반품", "사이즈", "불량", "하자", "취소"],
            "technical_support": ["설치", "사용법", "오류", "고장", "작동", "문제", "설정"]
        }

        inquiry_lower = inquiry.lower()
        scores = {}

        for category, keywords in classification_keywords.items():
            score = sum(1 for keyword in keywords if keyword in inquiry_lower)
            scores[category] = score

        # 가장 높은 점수의 카테고리 반환
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        else:
            return "product_inquiry"  # 기본값

    def create_dynamic_selector(self, category: str, examples: List[Dict[str, str]]) -> SemanticSimilarityExampleSelector:
        """동적 예시 선택기 구현"""

        if category in self.example_selectors:
            return self.example_selectors[category]

        # 벡터화
        to_vectorize = [f"{ex['input']} {ex['output']}" for ex in examples]

        vector_store = InMemoryVectorStore.from_texts(
            to_vectorize, self.embeddings, metadatas=examples
        )

        # 선택기 생성
        selector = SemanticSimilarityExampleSelector(
            vectorstore=vector_store,
            k=2  # 상위 2개 예시 선택
        )

        self.example_selectors[category] = selector
        return selector

    def create_category_prompt(self, category: str, examples: List[Dict[str, str]]) -> ChatPromptTemplate:
        """카테고리별 프롬프트 생성"""

        selector = self.create_dynamic_selector(category, examples)

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            input_variables=["input"],
            example_selector=selector,
            example_prompt=ChatPromptTemplate.from_messages([
                ("human", "{input}"),
                ("assistant", "{output}")
            ])
        )

        category_descriptions = {
            "product_inquiry": "당신은 제품 정보 전문가입니다. 고객의 제품 관련 문의에 정확하고 상세한 정보를 제공해주세요.",
            "shipping_inquiry": "당신은 배송 전문가입니다. 고객의 배송 관련 문의에 명확한 정보와 해결책을 제공해주세요.",
            "return_exchange": "당신은 교환/환불 처리 전문가입니다. 고객의 불만사항을 이해하고 적절한 해결방안을 제시해주세요.",
            "technical_support": "당신은 기술 지원 전문가입니다. 고객의 기술적 문제를 단계별로 해결해주세요."
        }

        final_prompt = ChatPromptTemplate.from_messages([
            ("system", category_descriptions.get(category, "당신은 친절한 고객 서비스 담당자입니다.")),
            few_shot_prompt,
            ("human", "{input}")
        ])

        return final_prompt

    def handle_customer_inquiry(self, inquiry: str) -> Dict[str, Any]:
        """고객 문의 처리"""

        # 1. 문의 유형 분류
        category = self.classify_inquiry_type(inquiry)

        # 2. 예시 데이터베이스에서 해당 카테고리 예시 가져오기
        example_database = self.setup_example_database()
        examples = example_database.get(category, example_database["product_inquiry"])

        # 3. 선택된 예시 확인
        selector = self.create_dynamic_selector(category, examples)
        selected_examples = selector.select_examples({"input": inquiry})

        # 4. 프롬프트 생성 및 응답
        prompt = self.create_category_prompt(category, examples)
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"input": inquiry})

        return {
            "inquiry": inquiry,
            "classified_category": category,
            "selected_examples": selected_examples,
            "response": response
        }

    def batch_handle_inquiries(self, inquiries: List[str]) -> List[Dict[str, Any]]:
        """다중 문의 일괄 처리"""
        results = []

        for inquiry in inquiries:
            result = self.handle_customer_inquiry(inquiry)
            results.append(result)

        return results

# 실습 2 테스트
service_bot = DynamicCustomerServiceBot()

test_inquiries = [
    "이 스마트폰의 배터리 용량이 어떻게 되나요?",
    "주문한 상품이 언제 도착하나요?",
    "제품이 불량인 것 같아서 환불받고 싶어요",
    "앱이 계속 꺼져서 사용할 수 없어요",
    "다른 색상으로 교환 가능한가요?",
    "배송비는 얼마인가요?",
    "설치 매뉴얼을 받을 수 있나요?",
    "주문을 취소하고 싶습니다"
]

print("=== 동적 고객 서비스 봇 테스트 ===")
batch_results = service_bot.batch_handle_inquiries(test_inquiries)

for i, result in enumerate(batch_results, 1):
    print(f"\n{i}. 문의: {result['inquiry']}")
    print(f"   분류: {result['classified_category']}")
    print("   선택된 예시:")
    for example in result['selected_examples']:
        print(f"     - {example['input']}")
    print(f"   응답: {result['response']}")
    print("-" * 80)

# 카테고리별 분류 정확도 확인
category_counts = {}
for result in batch_results:
    category = result['classified_category']
    category_counts[category] = category_counts.get(category, 0) + 1

print(f"\n=== 카테고리별 분류 결과 ===")
for category, count in category_counts.items():
    print(f"{category}: {count}개")
```

### 실습 3: 멀티태스크 Few-Shot 시스템

```python
class MultiTaskFewShotSystem:
    def __init__(self, model_name: str = "gpt-4.1-mini"):
        """멀티태스크 Few-Shot 시스템"""
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)
        self.embeddings = OllamaEmbeddings(model="bge-m3")
        self.task_prompts = {}
        self.task_selectors = {}

    def setup_task_router(self) -> Dict[str, List[str]]:
        """작업 유형 라우터 구현"""
        return {
            "translation": ["번역", "translate", "영어로", "한국어로", "중국어로", "일본어로"],
            "summarization": ["요약", "summary", "정리", "간단히", "핵심만"],
            "sentiment": ["감정", "sentiment", "긍정", "부정", "기분", "느낌"],
            "classification": ["분류", "category", "카테고리", "유형", "종류"],
            "qa": ["질문", "답변", "what", "why", "how", "무엇", "왜", "어떻게"],
            "generation": ["작성", "생성", "만들어", "써줘", "create", "generate"]
        }

    def identify_task_type(self, input_text: str) -> str:
        """입력 텍스트로부터 작업 유형 식별"""
        task_keywords = self.setup_task_router()

        input_lower = input_text.lower()
        task_scores = {}

        for task_type, keywords in task_keywords.items():
            score = sum(1 for keyword in keywords if keyword in input_lower)
            task_scores[task_type] = score

        # 패턴 기반 추가 점수
        if any(lang in input_lower for lang in ["영어로", "english", "korean", "한국어로"]):
            task_scores["translation"] = task_scores.get("translation", 0) + 2

        if any(word in input_lower for word in ["요약", "간단히", "핵심"]):
            task_scores["summarization"] = task_scores.get("summarization", 0) + 2

        if any(word in input_lower for word in ["어떻게", "방법", "어떤"]):
            task_scores["qa"] = task_scores.get("qa", 0) + 1

        # 가장 높은 점수의 작업 유형 반환
        if max(task_scores.values()) > 0:
            return max(task_scores, key=task_scores.get)
        else:
            return "qa"  # 기본값

    def create_task_specific_prompts(self) -> Dict[str, Dict]:
        """작업별 특화 프롬프트 생성"""

        task_examples = {
            "translation": [
                {
                    "input": "Hello, how are you? -> 한국어로",
                    "output": "안녕하세요, 어떻게 지내세요?"
                },
                {
                    "input": "감사합니다 -> English",
                    "output": "Thank you"
                },
                {
                    "input": "今日は天気がいいですね -> 한국어로",
                    "output": "오늘은 날씨가 좋네요"
                }
            ],
            "summarization": [
                {
                    "input": "인공지능은 인간의 학습능력과 추론능력을 컴퓨터로 구현한 기술이다. 머신러닝과 딥러닝을 포함하며 자율주행, 음성인식, 이미지 인식 등 다양한 분야에 활용되고 있다. 최근에는 GPT와 같은 대화형 AI가 주목받고 있다.",
                    "output": "인공지능은 인간의 학습과 추론 능력을 컴퓨터로 구현한 기술로, 머신러닝과 딥러닝을 포함하여 자율주행, 음성인식 등에 활용되며, 최근 대화형 AI가 주목받고 있다."
                },
                {
                    "input": "전자상거래 시장이 급성장하면서 배송업계에도 큰 변화가 일어나고 있다. 당일배송과 새벽배송 서비스가 보편화되고, 드론과 로봇을 활용한 무인배송 기술도 도입되고 있다. 이러한 변화는 소비자의 편의성을 높이고 있지만, 배송비 상승과 환경 문제라는 새로운 과제도 만들고 있다.",
                    "output": "전자상거래 성장으로 당일배송, 새벽배송이 보편화되고 무인배송 기술이 도입되어 소비자 편의성은 높아졌으나, 배송비 상승과 환경 문제 등의 과제가 발생하고 있다."
                }
            ],
            "sentiment": [
                {
                    "input": "이 제품 정말 마음에 들어요! 품질도 좋고 디자인도 예뻐요.",
                    "output": "긍정 - '마음에 들어요', '품질도 좋고', '예뻐요' 등의 만족과 호감 표현"
                },
                {
                    "input": "서비스가 너무 느리고 직원 태도도 불친절해서 실망스러워요.",
                    "output": "부정 - '너무 느리고', '불친절', '실망스러워요' 등의 불만과 실망 표현"
                },
                {
                    "input": "그냥 보통 수준이네요. 특별할 건 없어요.",
                    "output": "중립 - '보통 수준', '특별할 건 없다' 등의 객관적이고 중립적인 평가"
                }
            ],
            "classification": [
                {
                    "input": "애플, 바나나, 오렌지를 분류해주세요",
                    "output": "과일류 - 애플(사과), 바나나, 오렌지 모두 과일에 해당"
                },
                {
                    "input": "축구, 농구, 수영을 카테고리별로 나눠주세요",
                    "output": "스포츠 - 축구(구기종목), 농구(구기종목), 수영(수상종목)"
                }
            ],
            "qa": [
                {
                    "input": "파이썬에서 리스트와 튜플의 차이점은 무엇인가요?",
                    "output": "리스트는 변경 가능한(mutable) 자료형으로 요소를 추가/삭제/수정할 수 있고, 튜플은 변경 불가능한(immutable) 자료형으로 한 번 생성되면 요소를 변경할 수 없습니다. 리스트는 []로, 튜플은 ()로 표현합니다."
                },
                {
                    "input": "인공지능이 왜 중요한가요?",
                    "output": "인공지능은 인간의 한계를 보완하고 복잡한 문제를 효율적으로 해결할 수 있어 생산성 향상, 새로운 서비스 창출, 사회 문제 해결 등에 기여할 수 있기 때문에 중요합니다."
                }
            ],
            "generation": [
                {
                    "input": "여행 계획에 대한 짧은 글을 써주세요",
                    "output": "이번 여름 제주도 여행을 계획하고 있다. 3박 4일 일정으로 한라산 등반과 해변 휴식, 그리고 맛있는 흑돼지와 해산물을 맛보는 것이 주요 목표다. 렌터카를 빌려 섬 곳곳을 자유롭게 돌아다니며 아름다운 풍경을 만끽할 예정이다."
                }
            ]
        }

        task_system_messages = {
            "translation": "당신은 전문 번역가입니다. 주어진 텍스트를 정확하고 자연스럽게 번역해주세요.",
            "summarization": "당신은 요약 전문가입니다. 핵심 내용을 간결하고 명확하게 요약해주세요.",
            "sentiment": "당신은 감정 분석 전문가입니다. 텍스트의 감정을 분석하고 근거를 제시해주세요.",
            "classification": "당신은 분류 전문가입니다. 주어진 항목들을 적절한 카테고리로 분류해주세요.",
            "qa": "당신은 질의응답 전문가입니다. 질문에 대해 정확하고 도움이 되는 답변을 제공해주세요.",
            "generation": "당신은 창작 전문가입니다. 주어진 주제에 대해 창의적이고 유용한 콘텐츠를 생성해주세요."
        }

        # 각 작업별 프롬프트와 선택기 생성
        for task_type, examples in task_examples.items():
            # 벡터 저장소 생성
            to_vectorize = [f"{ex['input']} {ex['output']}" for ex in examples]
            vector_store = InMemoryVectorStore.from_texts(
                to_vectorize, self.embeddings, metadatas=examples
            )

            # 예시 선택기 생성
            selector = SemanticSimilarityExampleSelector(
                vectorstore=vector_store,
                k=2
            )

            # Few-shot 프롬프트 템플릿
            few_shot_prompt = FewShotChatMessagePromptTemplate(
                input_variables=["input"],
                example_selector=selector,
                example_prompt=ChatPromptTemplate.from_messages([
                    ("human", "{input}"),
                    ("assistant", "{output}")
                ])
            )

            # 최종 프롬프트
            final_prompt = ChatPromptTemplate.from_messages([
                ("system", task_system_messages[task_type]),
                few_shot_prompt,
                ("human", "{input}")
            ])

            self.task_prompts[task_type] = final_prompt
            self.task_selectors[task_type] = selector

        return task_examples

    def execute_task(self, input_text: str, task_type: str = None) -> Dict[str, Any]:
        """작업 실행"""

        # 작업 유형이 지정되지 않은 경우 자동 식별
        if task_type is None:
            task_type = self.identify_task_type(input_text)

        # 작업별 프롬프트가 없는 경우 생성
        if task_type not in self.task_prompts:
            self.create_task_specific_prompts()

        # 선택된 예시 확인
        selector = self.task_selectors.get(task_type)
        selected_examples = selector.select_examples({"input": input_text}) if selector else []

        # 프롬프트 실행
        prompt = self.task_prompts[task_type]
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"input": input_text})

        return {
            "input": input_text,
            "identified_task": task_type,
            "selected_examples": selected_examples,
            "result": result
        }

    def batch_execute(self, inputs: List[str]) -> List[Dict[str, Any]]:
        """다중 작업 일괄 실행"""
        return [self.execute_task(input_text) for input_text in inputs]

    def get_task_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """작업 통계 생성"""
        task_counts = {}
        for result in results:
            task = result["identified_task"]
            task_counts[task] = task_counts.get(task, 0) + 1
        return task_counts

# 실습 3 테스트
multi_task_system = MultiTaskFewShotSystem()

# 예시 프롬프트 생성
multi_task_system.create_task_specific_prompts()

test_inputs = [
    "Hello, nice to meet you! -> 한국어로 번역해주세요",
    "이 긴 문장을 간단히 요약해주세요: 기후변화는 지구온난화로 인한 기온 상승, 해수면 상승, 극지방 빙하 감소 등 다양한 환경 변화를 의미한다. 이로 인해 생태계 파괴, 농업 생산성 감소, 자연재해 증가 등의 문제가 발생하고 있으며, 국제사회는 온실가스 감축을 위한 다양한 노력을 기울이고 있다.",
    "이 리뷰의 감정을 분석해주세요: 제품이 기대했던 것보다 훨씬 좋네요! 디자인도 예쁘고 기능도 만족스러워요",
    "사과, 당근, 브로콜리, 바나나를 채소와 과일로 분류해주세요",
    "머신러닝과 딥러닝의 차이점이 무엇인가요?",
    "봄에 대한 짧은 시를 써주세요",
    "감사합니다 -> English로 번역",
    "긍정적인 마음가짐이 중요한 이유를 설명해주세요",
    "다음 텍스트의 감정은? '서비스가 별로였어요. 다음엔 다른 곳을 이용할 것 같아요.'",
    "여행 블로그 포스팅 제목 5개를 만들어주세요"
]

print("=== 멀티태스크 Few-Shot 시스템 테스트 ===")
batch_results = multi_task_system.batch_execute(test_inputs)

for i, result in enumerate(batch_results, 1):
    print(f"\n{i}. 입력: {result['input']}")
    print(f"   식별된 작업: {result['identified_task']}")
    print("   선택된 예시:")
    for example in result['selected_examples']:
        print(f"     입력: {example['input'][:50]}...")
    print(f"   결과: {result['result']}")
    print("-" * 100)

# 작업 통계 출력
task_stats = multi_task_system.get_task_statistics(batch_results)
print(f"\n=== 작업 유형별 통계 ===")
for task_type, count in task_stats.items():
    print(f"{task_type}: {count}개")

# 작업 유형별 정확도 평가 (수동 검증 필요)
print(f"\n=== 작업 식별 정확도 검증 ===")
expected_tasks = [
    "translation", "summarization", "sentiment", "classification",
    "qa", "generation", "translation", "qa", "sentiment", "generation"
]

correct_identifications = 0
for i, (result, expected) in enumerate(zip(batch_results, expected_tasks)):
    identified = result['identified_task']
    is_correct = identified == expected
    correct_identifications += is_correct
    print(f"{i+1}. 예상: {expected}, 식별: {identified} {'✓' if is_correct else '✗'}")

accuracy = correct_identifications / len(expected_tasks)
print(f"\n작업 식별 정확도: {accuracy:.2%}")
```

## 🔍 참고 자료

### 공식 문서
- [LangChain Few-Shot Prompting](https://python.langchain.com/docs/modules/model_io/prompts/few_shot_examples/)
- [OpenAI Few-Shot Learning](https://platform.openai.com/docs/guides/gpt-best-practices/strategy-provide-examples)
- [LangChain Example Selectors](https://python.langchain.com/docs/modules/model_io/prompts/example_selectors/)

### 학술 자료
- Brown, T., et al. (2020). "Language Models are Few-Shot Learners"
- Min, S., et al. (2022). "Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?"
- Dong, Q., et al. (2023). "A Survey for In-context Learning"

### 실무 가이드
- [Few-Shot Prompting Best Practices](https://example.com/few-shot-guide)
- [Dynamic Example Selection Strategies](https://example.com/dynamic-examples)
- [Prompt Engineering for Production](https://example.com/production-prompting)

### 도구 및 리소스
- [LangChain Example Selector Documentation](https://example.com/example-selectors)
- [Vector Store Integration Guide](https://example.com/vector-stores)
- [Performance Optimization Techniques](https://example.com/optimization)

---

**다음 학습**: W3_003_Prompt_Engineering_CoT.md - Chain of Thought 프롬프팅과 추론 능력 향상 기법을 학습합니다.