# W3_006_LMStudio - LM Studio와 LangChain 연동

## 학습 목표

이 가이드에서는 LM Studio를 사용한 로컬 LLM 실행과 LangChain 연동을 학습합니다:

- **LM Studio 설치 및 설정**: 로컬 환경에서 LLM 실행
- **OpenAI 호환 API**: LM Studio의 OpenAI 호환 서버 활용
- **LangChain 연동**: ChatOpenAI 클래스로 로컬 모델 연결
- **스트리밍 응답**: 실시간 텍스트 생성
- **대화 기록 유지**: ConversationChain으로 컨텍스트 관리
- **프롬프트 템플릿**: 구조화된 프롬프트 활용

### 선수 지식
- LangChain 기본 개념
- Python 환경 설정
- API 호출 기본 개념

---

## 핵심 개념

### LM Studio란?

**LM Studio**는 로컬 환경에서 대규모 언어 모델(LLM)을 실행할 수 있는 데스크톱 애플리케이션입니다.

**주요 특징**:
- 🖥️ **로컬 실행**: 인터넷 없이 오프라인 사용 가능
- 🔒 **프라이버시**: 데이터가 외부로 전송되지 않음
- 💰 **비용 절감**: API 호출 비용 없음
- 🚀 **다양한 모델**: Llama, Mistral, Qwen 등 지원
- 🌐 **OpenAI 호환**: 기존 OpenAI 코드 재사용 가능

**지원 플랫폼**:
- Windows (10/11)
- macOS (Intel/Apple Silicon)
- Linux (Ubuntu, Debian 등)

### OpenAI 호환 API

LM Studio는 OpenAI API와 동일한 인터페이스를 제공합니다.

**장점**:
```python
# OpenAI API 사용
from langchain_openai import ChatOpenAI

llm_openai = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini"
)

# LM Studio 사용 (코드 거의 동일!)
llm_local = ChatOpenAI(
    base_url="http://localhost:14000/v1",
    api_key="lm-studio",  # 형식상 필요
    model="local-model"
)
```

**차이점**:
| 특징 | OpenAI API | LM Studio |
|------|-----------|-----------|
| 실행 위치 | 클라우드 | 로컬 |
| 인터넷 필요 | O | X |
| API 비용 | O | X |
| 모델 선택 | 제한적 | 자유로움 |
| 응답 속도 | 빠름 | GPU 성능 의존 |
| 프라이버시 | 낮음 | 높음 |

### 로컬 LLM의 장단점

#### 장점
✅ **완전한 프라이버시**: 민감한 데이터 처리 가능
✅ **비용 절감**: API 호출 비용 없음
✅ **커스터마이징**: 모델 파라미터 직접 조정
✅ **오프라인**: 인터넷 없이 사용 가능
✅ **개발/테스트**: 무제한 실험 가능

#### 단점
❌ **하드웨어 요구사항**: 고성능 GPU 필요
❌ **초기 설정**: 모델 다운로드 및 설정 시간
❌ **성능 제한**: 최신 GPT-4 대비 성능 낮음
❌ **유지보수**: 모델 업데이트 수동 관리

---

## 환경 설정

### 1단계: LM Studio 설치

#### 다운로드
1. [lmstudio.ai](https://lmstudio.ai) 접속
2. 운영체제에 맞는 버전 다운로드
3. 설치 프로그램 실행

#### 시스템 요구사항
**최소**:
- RAM: 8GB
- 저장공간: 10GB (모델 크기에 따라 증가)
- CPU: 64-bit 프로세서

**권장**:
- RAM: 16GB 이상
- GPU: NVIDIA (CUDA 지원) 또는 Apple Silicon
- 저장공간: 50GB 이상

### 2단계: 모델 다운로드

#### 모델 선택 가이드
| 모델 | 크기 | RAM 요구 | 특징 |
|------|------|----------|------|
| Qwen2.5 0.5B | ~500MB | 2GB | 매우 빠름, 간단한 작업 |
| Llama 3.2 3B | ~2GB | 4GB | 빠름, 일반 작업 |
| Qwen2.5 7B | ~4GB | 8GB | 균형잡힌 성능 |
| Llama 3.1 8B | ~5GB | 10GB | 높은 품질 |
| Mistral 7B | ~4GB | 8GB | 코딩 특화 |
| Qwen2.5 14B | ~8GB | 16GB | 고성능 |

#### 다운로드 방법
1. LM Studio 실행
2. 좌측 메뉴에서 **Search** 클릭
3. 모델 검색 (예: "qwen2.5")
4. 원하는 모델의 **Download** 버튼 클릭
5. 다운로드 완료 대기 (모델 크기에 따라 시간 소요)

**추천 모델**:
```
초보자: qwen2.5-3b-instruct (빠르고 가벼움)
중급자: llama-3.2-3b-instruct (균형잡힌 성능)
고급자: qwen2.5-14b-instruct (높은 품질)
```

### 3단계: 로컬 서버 실행

#### 서버 시작
1. LM Studio에서 **Local Server** 탭 클릭
2. 상단 드롭다운에서 다운로드한 모델 선택
3. **Start Server** 버튼 클릭
4. 서버 상태가 "Running" 확인

**기본 설정**:
- 주소: `http://localhost:14000`
- 프로토콜: OpenAI 호환 API
- 포트: 14000 (설정에서 변경 가능)

#### 서버 설정 옵션
```
Context Length: 컨텍스트 길이 (기본 2048)
Temperature: 창의성 조절 (0.0-1.0)
Max Tokens: 최대 생성 토큰 수
GPU Layers: GPU에 로드할 레이어 수 (성능 최적화)
```

### 4단계: Python 환경 설정

```bash
# 필수 라이브러리 설치
pip install langchain langchain-openai python-dotenv
```

```python
# .env 파일 (선택 사항)
# LM Studio는 API 키가 필요 없지만, 일관성을 위해 설정 가능
LM_STUDIO_BASE_URL=http://localhost:14000/v1
```

---

## 단계별 구현

### 1단계: 기본 연결 설정

#### LM Studio 연결
```python
from langchain_openai import ChatOpenAI

# LM Studio 로컬 서버 연결
llm = ChatOpenAI(
    base_url="http://localhost:14000/v1",
    api_key="lm-studio",  # 형식상 필요 (실제 검증 안 함)
    model="qwen2.5-3b-instruct",  # 실제 모델명은 무관
    temperature=0.7,
)

print("✓ LM Studio 연결 설정 완료")
print(f"  - Base URL: http://localhost:14000/v1")
print(f"  - Temperature: 0.7")
```

**파라미터 설명**:
- `base_url`: LM Studio 서버 주소
- `api_key`: 형식상 필요 (임의 값 가능)
- `model`: 모델 이름 (LM Studio에서 로드한 모델 자동 사용)
- `temperature`: 0.0 (결정적) ~ 1.0 (창의적)

#### 연결 테스트
```python
try:
    response = llm.invoke("안녕하세요! 간단한 인사를 해주세요.")
    print("✓ 연결 성공!")
    print(f"\n응답: {response.content}")
except Exception as e:
    print(f"❌ 연결 실패: {e}")
    print("\n해결 방법:")
    print("  1. LM Studio 실행 확인")
    print("  2. Local Server 탭에서 서버 시작 확인")
    print("  3. 주소가 http://localhost:14000 인지 확인")
```

**예상 출력**:
```
✓ 연결 성공!

응답: 안녕하세요! 저는 AI 어시스턴트입니다. 무엇을 도와드릴까요?
```

---

### 2단계: 기본 질의응답

#### 단일 질문 처리
```python
def ask_question(question: str) -> str:
    """
    단일 질문에 대한 응답 생성

    Parameters:
        question (str): 사용자 질문

    Returns:
        str: AI 응답
    """
    response = llm.invoke(question)
    return response.content

# 테스트
question = "LangChain은 무엇인가요?"
answer = ask_question(question)

print(f"질문: {question}")
print(f"답변: {answer}")
```

**출력 예시**:
```
질문: LangChain은 무엇인가요?
답변: LangChain은 대규모 언어 모델(LLM)을 기반으로 애플리케이션을
개발하는 데 사용되는 오픈소스 프레임워크입니다. OpenAI, Hugging Face
등 다양한 LLM과 연동하여 채팅봇, 문서 분석, 데이터 처리 등의 기능을
쉽게 구축할 수 있습니다.
```

#### 다중 질문 배치 처리
```python
def batch_questions(questions: list[str]):
    """
    여러 질문을 순차적으로 처리

    Parameters:
        questions (list[str]): 질문 리스트
    """
    print("=" * 80)
    print("다중 질문 처리")
    print("=" * 80)

    for i, question in enumerate(questions, 1):
        print(f"\n[질문 {i}] {question}")
        answer = ask_question(question)
        # 처음 200자만 출력
        print(f"[답변 {i}] {answer[:200]}...")
        print("-" * 80)

# 테스트
questions = [
    "파이썬에서 리스트와 튜플의 차이점은 무엇인가요?",
    "머신러닝과 딥러닝의 차이를 간단히 설명해주세요.",
    "ETF란 무엇인가요?"
]

batch_questions(questions)
```

---

### 3단계: 스트리밍 응답

실시간으로 텍스트를 생성하며 출력합니다.

#### 기본 스트리밍
```python
def stream_response(question: str):
    """
    질문에 대한 스트리밍 응답

    Parameters:
        question (str): 사용자 질문

    실시간으로 응답을 출력하며 생성
    """
    print(f"질문: {question}\n")
    print("답변: ", end="", flush=True)

    full_response = ""
    for chunk in llm.stream(question):
        # 각 청크 출력
        print(chunk.content, end="", flush=True)
        full_response += chunk.content

    print("\n")
    return full_response

# 테스트
question = "인공지능의 역사를 간단히 설명해주세요."
response = stream_response(question)
```

**출력 예시** (실시간으로 표시됨):
```
질문: 인공지능의 역사를 간단히 설명해주세요.

답변:
인공지능의 역사는 1950년대 앨런 튜링의 투링 테스트 제안으로 시작되었습니다.
1956년 다트머스 회의에서 "인공지능"이라는 용어가 창안되었고...
```

#### 진행률 표시 스트리밍
```python
import sys

def stream_with_progress(question: str):
    """진행률을 표시하며 스트리밍"""
    print(f"질문: {question}\n")
    print("생성 중: ", end="", flush=True)

    char_count = 0
    for chunk in llm.stream(question):
        content = chunk.content
        print(content, end="", flush=True)
        char_count += len(content)

        # 매 100자마다 진행 표시
        if char_count % 100 == 0:
            sys.stderr.write(".")
            sys.stderr.flush()

    print(f"\n\n✓ 총 {char_count}자 생성 완료")

# 테스트
stream_with_progress("머신러닝의 주요 알고리즘 5가지를 설명해주세요.")
```

---

### 4단계: 대화 기록 유지

이전 대화 내용을 기억하는 대화형 시스템을 구현합니다.

#### ConversationChain 사용
```python
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# 메모리 초기화
memory = ConversationBufferMemory()

# 대화 체인 생성
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False  # True로 설정하면 프롬프트 표시
)

print("✓ 대화 체인 생성 완료")
```

#### 연속 대화 테스트
```python
def conversation_test():
    """대화 기록 유지 테스트"""
    print("=" * 80)
    print("대화 기록 유지 테스트")
    print("=" * 80)

    # 연속된 대화
    conversations = [
        "제 이름은 김철수입니다.",
        "제가 방금 뭐라고 했죠?",
        "제 이름의 성은 무엇인가요?"
    ]

    for i, user_input in enumerate(conversations, 1):
        print(f"\n[대화 {i}]")
        print(f"사용자: {user_input}")

        # 응답 생성
        response = conversation.predict(input=user_input)
        print(f"AI: {response}")
        print("-" * 80)

    # 메모리 확인
    print("\n메모리에 저장된 대화:")
    print(memory.buffer)

# 실행
conversation_test()
```

**출력 예시**:
```
[대화 1]
사용자: 제 이름은 김철수입니다.
AI: 안녕하세요, 김철수님! 만나서 반갑습니다.

[대화 2]
사용자: 제가 방금 뭐라고 했죠?
AI: 방금 '제 이름은 김철수입니다.'라고 말씀하셨습니다.

[대화 3]
사용자: 제 이름의 성은 무엇인가요?
AI: 성은 '김'입니다.

메모리에 저장된 대화:
Human: 제 이름은 김철수입니다.
AI: 안녕하세요, 김철수님! 만나서 반갑습니다.
Human: 제가 방금 뭐라고 했죠?
AI: 방금 '제 이름은 김철수입니다.'라고 말씀하셨습니다.
...
```

#### 대화 기록 관리
```python
# 메모리 초기화
memory.clear()

# 특정 대화만 저장
memory.save_context(
    {"input": "안녕하세요"},
    {"output": "안녕하세요! 무엇을 도와드릴까요?"}
)

# 메모리 내용 확인
print(memory.load_memory_variables({}))
```

---

### 5단계: 프롬프트 템플릿 활용

구조화된 프롬프트로 다양한 역할과 스타일을 구현합니다.

#### 기본 템플릿
```python
from langchain_core.prompts import ChatPromptTemplate

# 시스템 메시지 + 사용자 메시지 템플릿
template = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절한 {role} 전문가입니다. 사용자의 질문에 {style}로 답변해주세요."),
    ("human", "{question}")
])

# 체인 생성
chain = template | llm

# 실행
response = chain.invoke({
    "role": "금융",
    "style": "쉽고 간단하게",
    "question": "ETF가 무엇인가요?"
})

print(response.content)
```

**출력**:
```
ETF는 거래소에서 실시간으로 거래할 수 있는 펀드로, 한 번에 여러
주식이나 채권을 투자해 특정 지수를 추적하는 상품입니다.
비용이 적고 편리해서 많은 투자자들이 활용하고 있습니다.
```

#### 다양한 역할 테스트
```python
def test_roles():
    """다양한 역할과 스타일 테스트"""
    test_cases = [
        {
            "role": "금융",
            "style": "쉽고 간단하게",
            "question": "ETF가 무엇인가요?"
        },
        {
            "role": "프로그래밍",
            "style": "기술적으로 상세하게",
            "question": "파이썬의 데코레이터는 무엇인가요?"
        },
        {
            "role": "교육",
            "style": "초등학생도 이해할 수 있게",
            "question": "인공지능이 무엇인가요?"
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[테스트 {i}]")
        print(f"역할: {test_case['role']}, 스타일: {test_case['style']}")
        print(f"질문: {test_case['question']}")

        response = chain.invoke(test_case)
        print(f"\n답변:\n{response.content[:300]}...")
        print("-" * 80)

# 실행
test_roles()
```

#### 복잡한 템플릿
```python
# 다단계 템플릿
complex_template = ChatPromptTemplate.from_messages([
    ("system", """당신은 {expertise} 분야의 전문가입니다.

    응답 지침:
    1. {tone} 어조로 답변
    2. {length} 길이로 설명
    3. {examples}개의 예시 포함

    대상 청중: {audience}
    """),
    ("human", "{question}"),
])

# 실행
response = (complex_template | llm).invoke({
    "expertise": "데이터 과학",
    "tone": "친절하고 전문적인",
    "length": "중간 정도",
    "examples": 2,
    "audience": "초급 개발자",
    "question": "랜덤 포레스트 알고리즘을 설명해주세요."
})

print(response.content)
```

---

## 실전 활용 예제

### 예제 1: 대화형 챗봇

```python
class LocalChatbot:
    """LM Studio 기반 대화형 챗봇"""

    def __init__(self, llm, system_prompt: str = None):
        self.llm = llm
        self.memory = ConversationBufferMemory()
        self.system_prompt = system_prompt or "당신은 친절한 AI 어시스턴트입니다."

        # 대화 체인 생성
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=False
        )

    def chat(self, user_input: str) -> str:
        """사용자 입력에 대한 응답 생성"""
        response = self.conversation.predict(input=user_input)
        return response

    def reset(self):
        """대화 기록 초기화"""
        self.memory.clear()

    def get_history(self) -> str:
        """대화 기록 반환"""
        return self.memory.buffer

# 사용 예시
chatbot = LocalChatbot(llm, system_prompt="당신은 Python 프로그래밍 튜터입니다.")

print("=" * 80)
print("대화형 챗봇 테스트")
print("=" * 80)

conversations = [
    "안녕하세요! 파이썬을 배우고 싶습니다.",
    "리스트를 어떻게 만드나요?",
    "방금 설명한 내용을 예시로 보여주세요.",
    "고마워요!"
]

for user_input in conversations:
    print(f"\n사용자: {user_input}")
    response = chatbot.chat(user_input)
    print(f"챗봇: {response}")
```

### 예제 2: 문서 요약 시스템

```python
def summarize_document(text: str, max_length: int = 200) -> str:
    """
    문서를 요약하는 함수

    Parameters:
        text (str): 원본 문서
        max_length (int): 요약 최대 길이

    Returns:
        str: 요약된 텍스트
    """
    template = ChatPromptTemplate.from_messages([
        ("system", f"다음 문서를 {max_length}자 이내로 요약해주세요. 핵심 내용만 간결하게 정리하세요."),
        ("human", "{text}")
    ])

    chain = template | llm
    response = chain.invoke({"text": text})

    return response.content

# 테스트
sample_text = """
인공지능(AI)은 인간의 학습능력, 추론능력, 지각능력을 인공적으로 구현한
컴퓨터 시스템입니다. 최근 딥러닝 기술의 발전으로 이미지 인식, 자연어 처리,
음성 인식 등 다양한 분야에서 괄목할 만한 성과를 거두고 있습니다.
특히 ChatGPT와 같은 대규모 언어 모델의 등장으로 AI는 우리 일상에
더욱 깊숙이 들어오고 있습니다.
"""

summary = summarize_document(sample_text, max_length=100)
print(f"원본 ({len(sample_text)}자):\n{sample_text}\n")
print(f"요약 ({len(summary)}자):\n{summary}")
```

### 예제 3: 번역 시스템

```python
def translate(text: str, target_language: str = "English") -> str:
    """
    텍스트를 다른 언어로 번역

    Parameters:
        text (str): 번역할 텍스트
        target_language (str): 목표 언어

    Returns:
        str: 번역된 텍스트
    """
    template = ChatPromptTemplate.from_messages([
        ("system", f"Translate the following text to {target_language}. Only provide the translation without any explanations."),
        ("human", "{text}")
    ])

    chain = template | llm
    response = chain.invoke({"text": text})

    return response.content

# 테스트
korean_text = "안녕하세요. LM Studio를 사용한 로컬 AI 개발에 오신 것을 환영합니다."

english = translate(korean_text, "English")
japanese = translate(korean_text, "Japanese")

print(f"한국어: {korean_text}")
print(f"영어: {english}")
print(f"일본어: {japanese}")
```

### 예제 4: 코드 생성 및 설명

```python
def generate_code(description: str, language: str = "Python") -> dict:
    """
    자연어 설명으로부터 코드 생성

    Parameters:
        description (str): 원하는 기능 설명
        language (str): 프로그래밍 언어

    Returns:
        dict: {'code': 생성된 코드, 'explanation': 설명}
    """
    template = ChatPromptTemplate.from_messages([
        ("system", f"""You are an expert {language} programmer.
        Generate clean, efficient code based on the description.

        Format your response as:
        CODE:
        ```{language.lower()}
        [your code here]
        ```

        EXPLANATION:
        [brief explanation in Korean]
        """),
        ("human", "{description}")
    ])

    chain = template | llm
    response = chain.invoke({"description": description})

    # 응답 파싱
    content = response.content
    code_start = content.find("```")
    code_end = content.rfind("```")

    if code_start != -1 and code_end != -1:
        code = content[code_start:code_end + 3]
        explanation = content[code_end + 3:].strip()
    else:
        code = content
        explanation = ""

    return {
        "code": code,
        "explanation": explanation
    }

# 테스트
result = generate_code(
    "리스트에서 짝수만 필터링하고 제곱하여 새 리스트를 만드는 함수",
    "Python"
)

print("생성된 코드:")
print(result["code"])
print("\n설명:")
print(result["explanation"])
```

---

## 연습 문제

### 기본 문제

**문제 1**: 연결 검증 함수
```python
# 과제: LM Studio 서버 연결 상태를 확인하는 함수 작성

def check_lm_studio_connection(base_url: str = "http://localhost:14000/v1") -> bool:
    """
    LM Studio 서버 연결 상태 확인

    TODO:
    1. ChatOpenAI 객체 생성
    2. 간단한 테스트 요청
    3. 성공 여부 반환

    Returns:
        bool: 연결 성공 시 True
    """
    pass
```

**문제 2**: 응답 시간 측정
```python
# 과제: 응답 생성 시간 측정

import time

def measure_response_time(question: str) -> dict:
    """
    질문에 대한 응답 시간 측정

    TODO:
    1. 시작 시간 기록
    2. LLM 호출
    3. 종료 시간 기록
    4. 응답과 소요 시간 반환

    Returns:
        dict: {'response': str, 'time': float}
    """
    pass
```

**문제 3**: 간단한 대화 로그
```python
# 과제: 대화 내역을 파일로 저장

def save_conversation_log(conversation: ConversationChain, filename: str):
    """
    대화 내역을 텍스트 파일로 저장

    TODO:
    1. 메모리에서 대화 내역 추출
    2. 타임스탬프 추가
    3. 파일로 저장
    """
    pass
```

### 중급 문제

**문제 4**: 컨텍스트 윈도우 관리
```python
# 과제: 대화 길이를 제한하여 메모리 관리

from langchain.memory import ConversationBufferWindowMemory

class ManagedConversation:
    def __init__(self, llm, max_turns: int = 5):
        """
        대화 턴 수를 제한하는 대화 관리자

        TODO:
        1. ConversationBufferWindowMemory 사용
        2. max_turns 설정
        3. 대화 체인 생성
        """
        pass

    def chat(self, user_input: str) -> str:
        """대화 처리"""
        pass
```

**문제 5**: 다중 모델 비교
```python
# 과제: 여러 LM Studio 모델 응답 비교

def compare_models(question: str, models: list[str]):
    """
    여러 모델의 응답 비교

    TODO:
    1. 각 모델로 ChatOpenAI 생성
    2. 동일한 질문으로 응답 생성
    3. 비교 결과 출력

    Parameters:
        question: 질문
        models: 모델 이름 리스트 (LM Studio에서 순차 로드 필요)
    """
    pass
```

**문제 6**: 스트리밍 속도 측정
```python
# 과제: 스트리밍 응답의 토큰/초 측정

def measure_streaming_speed(question: str):
    """
    스트리밍 속도 측정

    TODO:
    1. 시작 시간 기록
    2. 청크별로 토큰 수 계산
    3. 총 시간 및 속도 계산

    Returns:
        dict: {
            'total_tokens': int,
            'total_time': float,
            'tokens_per_second': float
        }
    """
    pass
```

### 고급 문제

**문제 7**: RAG 시스템 통합
```python
# 과제: LM Studio와 로컬 임베딩을 사용한 RAG

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class LocalRAG:
    def __init__(self, llm):
        """
        로컬 RAG 시스템

        TODO:
        1. 로컬 임베딩 모델 초기화 (HuggingFace)
        2. 벡터 저장소 생성
        3. 검색 체인 구성
        """
        self.llm = llm
        self.embeddings = None
        self.vector_store = None

    def add_documents(self, texts: list[str]):
        """문서 추가"""
        pass

    def query(self, question: str) -> str:
        """RAG 기반 질의응답"""
        pass
```

**문제 8**: 자동 재시도 메커니즘
```python
# 과제: 오류 발생 시 자동 재시도

from functools import wraps
import time

def retry_on_error(max_retries: int = 3, delay: float = 1.0):
    """
    LLM 호출 실패 시 자동 재시도 데코레이터

    TODO:
    1. 예외 처리
    2. 지수 백오프 (exponential backoff)
    3. 최대 재시도 횟수 제한
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 구현
            pass
        return wrapper
    return decorator

@retry_on_error(max_retries=3, delay=2.0)
def robust_llm_call(question: str):
    """안정적인 LLM 호출"""
    return llm.invoke(question)
```

**문제 9**: 비용 및 성능 모니터링
```python
# 과제: LLM 사용량 모니터링 시스템

class LLMMonitor:
    def __init__(self, llm):
        """
        LLM 사용량 모니터링

        TODO:
        1. 호출 횟수 추적
        2. 토큰 사용량 계산
        3. 평균 응답 시간 측정
        4. 통계 리포트 생성
        """
        self.llm = llm
        self.stats = {
            'total_calls': 0,
            'total_tokens': 0,
            'total_time': 0.0,
            'errors': 0
        }

    def invoke(self, question: str):
        """모니터링이 포함된 호출"""
        pass

    def get_report(self) -> dict:
        """통계 리포트 반환"""
        pass
```

---

## 문제 해결 가이드

### 일반적인 문제

#### 1. 서버 연결 실패
```python
# 문제: ConnectionError 발생

# 원인 1: LM Studio 서버 미실행
# 해결: LM Studio 실행 → Local Server → Start Server

# 원인 2: 포트 번호 불일치
# 해결: LM Studio 설정 확인
llm = ChatOpenAI(
    base_url="http://localhost:14000/v1",  # 포트 확인
    ...
)

# 원인 3: 방화벽 차단
# 해결: 방화벽 예외 추가 또는 비활성화
```

#### 2. 응답 없음
```python
# 문제: 무한 대기

# 원인: 모델 로딩 실패
# 해결: LM Studio에서 모델 상태 확인
# - Model 탭에서 모델 다운로드 확인
# - Local Server에서 모델 로드 확인

# 타임아웃 설정
llm = ChatOpenAI(
    base_url="http://localhost:14000/v1",
    timeout=60,  # 60초 타임아웃
)
```

#### 3. 한글 깨짐
```python
# 문제: 한글 응답이 깨져서 출력

# 해결 1: 인코딩 설정
import sys
sys.stdout.reconfigure(encoding='utf-8')

# 해결 2: 모델 선택
# 한글 지원 모델 사용 (Qwen, Llama 3.x 등)

# 해결 3: 프롬프트에 명시
template = ChatPromptTemplate.from_messages([
    ("system", "Please respond in Korean (한국어로 답변해주세요)."),
    ("human", "{question}")
])
```

#### 4. 메모리 부족
```python
# 문제: Out of Memory 오류

# 해결 1: 작은 모델 사용
# 7B → 3B 모델로 변경

# 해결 2: GPU 레이어 조정
# LM Studio 설정 → GPU Layers 줄이기

# 해결 3: 컨텍스트 길이 축소
# LM Studio 설정 → Context Length 2048로 설정

# 해결 4: 대화 기록 제한
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=5)  # 최근 5개만 유지
```

### 성능 최적화

#### GPU 활용
```python
# LM Studio 설정에서:
# - GPU Layers: 최대값으로 설정
# - GPU Offload: 활성화

# NVIDIA GPU 확인
import subprocess
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print(result.stdout)
```

#### 배치 처리
```python
def batch_invoke(questions: list[str]) -> list[str]:
    """
    여러 질문을 효율적으로 처리

    현재 LM Studio는 진정한 배치를 지원하지 않으므로
    순차 처리하되 세션 재사용
    """
    responses = []
    for question in questions:
        response = llm.invoke(question)
        responses.append(response.content)
    return responses
```

#### 캐싱
```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=100)
def cached_invoke(question_hash: str):
    """응답 캐싱"""
    # 실제로는 원본 질문 필요
    return llm.invoke(question)

def invoke_with_cache(question: str):
    """캐시를 활용한 호출"""
    question_hash = hashlib.md5(question.encode()).hexdigest()
    return cached_invoke(question_hash)
```

---

## 추가 학습 자료

### 공식 문서
- [LM Studio Documentation](https://lmstudio.ai/docs)
- [LangChain Local Models](https://python.langchain.com/docs/guides/local_llms)
- [Hugging Face Model Hub](https://huggingface.co/models)

### 추천 모델
- **Qwen2.5**: 다국어 지원 우수
- **Llama 3.2**: 균형잡힌 성능
- **Mistral**: 코딩 특화
- **Phi-3**: 소형 모델 중 최고 성능

### 다음 단계
1. **로컬 임베딩**: HuggingFace 임베딩 모델
2. **RAG 구축**: 로컬 벡터 데이터베이스
3. **파인튜닝**: 도메인 특화 모델 학습
4. **멀티모달**: 이미지 처리 모델 추가
5. **프로덕션 배포**: API 서버 구축

### 심화 주제
- **Quantization**: 모델 양자화로 메모리 절감
- **LoRA**: 효율적인 파인튜닝
- **GGUF Format**: 최적화된 모델 포맷
- **Ollama**: 대안 로컬 LLM 도구
- **vLLM**: 고성능 추론 엔진

---

## 요약

이 가이드에서 학습한 핵심 내용:

✅ **LM Studio 기본**
- 설치 및 모델 다운로드
- 로컬 서버 실행
- OpenAI 호환 API 활용

✅ **LangChain 연동**
- ChatOpenAI로 로컬 모델 연결
- 기본 질의응답
- 스트리밍 응답

✅ **고급 기능**
- ConversationChain으로 대화 기록 유지
- 프롬프트 템플릿 활용
- 다양한 역할 구현

✅ **실전 활용**
- 대화형 챗봇
- 문서 요약 시스템
- 번역 및 코드 생성

이제 완전히 오프라인에서 작동하는 프라이버시 보호 AI 시스템을 구축할 수 있습니다!
