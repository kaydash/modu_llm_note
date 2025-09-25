# W1_003_LangChain_Components.md - LangChain 기본 구조와 컴포넌트

## 🎯 학습 목표
- LangChain의 기본 개념과 아키텍처 이해
- LangChain 핵심 컴포넌트 활용법 학습
- 모델, 메시지, 프롬프트 템플릿, 출력 파서 실전 사용법 습득
- 체인 구성을 통한 파이프라인 구축 능력 개발

## 📚 핵심 개념

### LangChain 개념
- **LangChain**: LLM 기반 애플리케이션 개발을 위한 프레임워크
- **Chain**: 작업을 순차적으로 실행하는 파이프라인 구조
- **Agent**: 자율적 의사결정이 가능한 실행 단위
- **모듈성**: 독립적인 컴포넌트들을 조합해 복잡한 시스템 구현 가능

### 컴포넌트 구조
- **언어 처리 기능**: LLM/ChatModel이 중심, Prompt와 Memory로 대화 관리
- **문서 처리와 검색**: Document Loader, Text Splitter, Embedding, Vectorstore
- **통합 인터페이스**: 다양한 모델 제공자를 동일한 방식으로 사용 가능

## 🔧 환경 설정

### 필수 라이브러리 설치
```bash
# UV 패키지 매니저 사용 시
uv add ipykernel python-dotenv langchain langchain-openai langchain-google-genai

# pip 사용 시
pip install ipykernel python-dotenv langchain langchain-openai langchain-google-genai
```

### 환경 변수 설정
```bash
# .env 파일 생성
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

### 기본 환경 로드
```python
from dotenv import load_dotenv
load_dotenv()
```

## 💻 코드 예제

### 1. 모델 (Models) 활용

#### 다중 모델 비교
```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Gemini 모델 초기화
gemini_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# OpenAI 모델 초기화
openai_model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# 각 모델 응답 비교
gemini_response = gemini_model.invoke("안녕하세요!")
openai_response = openai_model.invoke("안녕하세요!")

print("Gemini 답변:", gemini_response.content)
print("OpenAI 답변:", openai_response.content)
```

#### 응답 객체 메타데이터 분석
```python
# 응답 객체 구조 확인
print("응답 타입:", type(gemini_response))
print("메시지 내용:", gemini_response.content)
print("메타데이터:", gemini_response.response_metadata)
print("사용량 정보:", gemini_response.usage_metadata)
```

### 2. 메시지 (Messages) 시스템

#### HumanMessage 사용
```python
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# 사용자 메시지 생성
human_message = HumanMessage(content="Glory를 한국어로 번역해주세요.")

# 번역 요청
response = model.invoke([human_message])
print("번역 결과:", response.content)
```

#### SystemMessage와 HumanMessage 조합
```python
from langchain_core.messages import SystemMessage, HumanMessage

# 시스템 메시지로 역할 설정
system_msg = SystemMessage(content="당신은 영어를 한국어로 번역하는 AI 어시스턴트입니다.")
human_msg = HumanMessage(content="Glory")

# 메시지 리스트로 전달
messages = [system_msg, human_msg]
response = model.invoke(messages)

print("답변:", response.content)  # 출력: "영광"
```

### 3. 프롬프트 템플릿 (Prompt Template)

#### 기본 문자열 템플릿
```python
from langchain_core.prompts import PromptTemplate

# 템플릿 생성
template = PromptTemplate.from_template("{topic}에 대한 이야기를 해줘")

# 템플릿 사용
prompt = template.invoke({"topic": "고양이"})
print(prompt)  # StringPromptValue(text='고양이에 대한 이야기를 해줘')
```

#### 채팅 프롬프트 템플릿
```python
from langchain_core.prompts import ChatPromptTemplate

# 채팅 템플릿 생성
template = ChatPromptTemplate.from_messages([
    ("system", "당신은 도움이 되는 비서입니다"),
    ("user", "{subject}에 대해 설명해주세요")
])

# 템플릿 사용
prompt = template.invoke({"subject": "인공지능"})
print(prompt.messages)
```

#### MessagesPlaceholder 활용
```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 메시지 플레이스홀더 포함 템플릿
template = ChatPromptTemplate.from_messages([
    ("system", "당신은 도움이 되는 비서입니다"),
    MessagesPlaceholder("chat_history")  # 채팅 기록을 위한 플레이스홀더
])

# 채팅 히스토리와 함께 사용
prompt = template.invoke({
    "chat_history": [
        HumanMessage(content="안녕하세요! 제 이름은 스티브입니다."),
        AIMessage(content="안녕하세요! 무엇을 도와드릴까요?"),
        HumanMessage(content="제 이름을 기억하나요?")
    ]
})

print(prompt.messages)
```

### 4. 출력 파서 (Output Parser)

#### 기본 문자열 파서
```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 파이프라인 구성
parser = StrOutputParser()
prompt = PromptTemplate.from_template("도시 {city}의 특징을 알려주세요")
model = ChatOpenAI(model='gpt-4.1-mini')

# 체인 생성 및 실행
chain = prompt | model | parser
result = chain.invoke({"city": "서울"})

print(result)  # 문자열 형태의 응답
```

#### 구조화된 출력 파서
```python
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Pydantic 모델로 출력 구조 정의
class CityInfo(BaseModel):
    name: str = Field(description="도시 이름")
    description: str = Field(description="도시의 특징")

# 구조화된 출력 모델 생성
prompt = PromptTemplate.from_template("도시 {city}의 특징을 알려주세요.")
model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
structured_model = model.with_structured_output(CityInfo)

# 체인 실행
chain = prompt | structured_model
result = chain.invoke({"city": "서울"})

print(f"도시 이름: {result.name}")
print(f"특징: {result.description}")
```

## 🚀 실습해보기

### 실습 1: 다중 모델 텍스트 생성
**목표**: Gemini 모델을 사용하여 텍스트 생성하고 응답 객체 확인
```python
# 코드를 작성해보세요
from langchain_google_genai import ChatGoogleGenerativeAI

# Gemini 모델 초기화
gemini = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# 메시지 전송 및 응답 확인
response = gemini.invoke("안녕하세요!")

# 결과 분석
print("답변:", response.content)
print("메타데이터:", response.response_metadata)
```

### 실습 2: 채팅 메시지 시스템 활용
**목표**: 시스템 메시지와 사용자 메시지를 조합하여 번역 시스템 구현
```python
# 코드를 작성해보세요
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# 모델 및 메시지 구성
gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
system_msg = SystemMessage(content="당신은 영어를 한국어로 번역하는 AI 어시스턴트입니다.")
human_message = HumanMessage(content="Glory")

# 메시지 리스트로 처리
messages = [system_msg, human_message]
response = gemini.invoke(messages)

print("답변:", response.content)
print("메타데이터:", response.response_metadata)
```

### 실습 3: 텍스트 요약 템플릿 구현
**목표**: 프롬프트 템플릿을 사용한 요약 시스템 구축
```python
# 코드를 작성해보세요
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# 모델 및 템플릿 구성
gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

template = ChatPromptTemplate.from_messages([
    ("system", "당신은 텍스트 요약을 잘하는 AI 어시스턴트입니다."),
    ("user", "다음 텍스트를 1~2 문장으로 핵심 내용을 위주로 간결하게 요약해주세요: {text}")
])

# 요약 체인 생성
summarization_chain = template | gemini

# 테스트 텍스트
text = """
인공지능은 컴퓨터 시스템이 인간의 지능을 모방하여 학습, 추론, 문제 해결 등을 수행하는 기술입니다.
인공지능은 머신러닝, 딥러닝, 자연어 처리 등 다양한 분야에서 활용되며, 자율주행차, 음성 인식, 이미지 분석 등 여러 응용 프로그램에 적용됩니다.
생성형 인공지능은 텍스트, 이미지, 음악 등 다양한 콘텐츠를 생성하는 데 사용되며, 창의적인 작업에서도 큰 역할을 하고 있습니다.
"""

response = summarization_chain.invoke({"text": text})
print("답변:", response.content)
```

### 실습 4: 구조화된 뉴스 분석 시스템
**목표**: Pydantic 모델을 활용한 구조화된 출력 시스템 구현
```python
# 코드를 작성해보세요
from typing import Optional
from pydantic import BaseModel, Field, confloat
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# 뉴스 분석 결과 구조 정의
class NewsAnalysis(BaseModel):
    언론사: str = Field(description="뉴스 기사의 언론사 이름")
    기사_제목: str = Field(description="뉴스 기사의 제목", alias="기사 제목")
    작성자: str = Field(description="뉴스 기사의 작성자")
    작성일: str = Field(description="뉴스 기사의 작성일")
    요약: str = Field(description="뉴스 기사 내용의 간결한 요약 (20-30자)")
    분야: str = Field(description="뉴스 기사의 분야 (경제/사회/정치/국제/문화/IT/과학 등)")
    중요도: confloat(ge=0.0, le=1.0) = Field(description="뉴스 기사의 중요도 (0.00-1.00)")
    confidence: confloat(ge=0.0, le=1.0) = Field(description="추출의 확신도 (0.00-1.00)")

# 모델 및 구조화된 출력 설정
gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
structured_gemini = gemini.with_structured_output(NewsAnalysis)

# 분석 템플릿
template = ChatPromptTemplate.from_messages([
    ("system", "당신은 텍스트 요약과 분석을 잘하는 AI 어시스턴트입니다."),
    ("user", """
     다음 뉴스 기사에서 언론사, 기사 제목, 작성자, 작성일, 기사 내용 요약(20-30자), 분야를 추출해주세요.
     추출한 정보의 중요도(0.00~1.00 사이)와 추출에 대한 확신도(0.00~1.00 사이)도 함께 평가해 주세요.

     ```기사 내용
     {news_article}
     ```
     """)
])

# 분석 체인 생성
analysis_chain = template | structured_gemini

# 테스트 (실제 뉴스 기사 텍스트를 news_article 변수에 할당 후 실행)
# result = analysis_chain.invoke({"news_article": news_article})
# print(f"언론사: {result.언론사}")
# print(f"기사 제목: {result.기사_제목}")
```

## 📋 해답

### 해답 1: 다중 모델 텍스트 생성
```python
from langchain_google_genai import ChatGoogleGenerativeAI

gemini = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

response = gemini.invoke("안녕하세요!")

print("답변:", response.content)
# 출력: 안녕하세요! 무엇을 도와드릴까요? 😊

print("메타데이터:", response.response_metadata)
# 출력: {'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}
```

### 해답 2: 채팅 메시지 시스템 활용
```python
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

system_msg = SystemMessage(content="당신은 영어를 한국어로 번역하는 AI 어시스턴트입니다.")
human_message = HumanMessage(content="Glory")

messages = [system_msg, human_message]
response = gemini.invoke(messages)

print("답변:", response.content)  # 출력: 영광
print("메타데이터:", response.response_metadata)
```

### 해답 3: 텍스트 요약 템플릿 구현
```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

template = ChatPromptTemplate.from_messages([
    ("system", "당신은 텍스트 요약을 잘하는 AI 어시스턴트입니다."),
    ("user", "다음 텍스트를 1~2 문장으로 핵심 내용을 위주로 간결하게 요약해주세요: {text}")
])

summarization_chain = template | gemini

text = """
인공지능은 컴퓨터 시스템이 인간의 지능을 모방하여 학습, 추론, 문제 해결 등을 수행하는 기술입니다.
인공지능은 머신러닝, 딥러닝, 자연어 처리 등 다양한 분야에서 활용되며, 자율주행차, 음성 인식, 이미지 분석 등 여러 응용 프로그램에 적용됩니다.
생성형 인공지능은 텍스트, 이미지, 음악 등 다양한 콘텐츠를 생성하는 데 사용되며, 창의적인 작업에서도 큰 역할을 하고 있습니다.
"""

response = summarization_chain.invoke({"text": text})
print("답변:", response.content)
# 출력: 인공지능은 인간의 지능을 모방하여 학습, 추론, 문제 해결 등을 수행하는 기술로, 머신러닝, 딥러닝, 자연어 처리 등 다양한 분야에서 활용됩니다. 특히 생성형 인공지능은 텍스트, 이미지, 음악 등 다양한 콘텐츠를 생성하며 창의적인 작업에 기여합니다.
```

### 해답 4: 구조화된 뉴스 분석 시스템
```python
from typing import Optional
from pydantic import BaseModel, Field, confloat
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

class NewsAnalysis(BaseModel):
    언론사: str = Field(description="뉴스 기사의 언론사 이름")
    기사_제목: str = Field(description="뉴스 기사의 제목", alias="기사 제목")
    작성자: str = Field(description="뉴스 기사의 작성자")
    작성일: str = Field(description="뉴스 기사의 작성일")
    요약: str = Field(description="뉴스 기사 내용의 간결한 요약 (20-30자)")
    분야: str = Field(description="뉴스 기사의 분야 (경제/사회/정치/국제/문화/IT/과학 등)")
    중요도: confloat(ge=0.0, le=1.0) = Field(description="뉴스 기사의 중요도 (0.00-1.00)")
    confidence: confloat(ge=0.0, le=1.0) = Field(description="추출의 확신도 (0.00-1.00)")

gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
structured_gemini = gemini.with_structured_output(NewsAnalysis)

template = ChatPromptTemplate.from_messages([
    ("system", "당신은 텍스트 요약과 분석을 잘하는 AI 어시스턴트입니다."),
    ("user", """
     다음 뉴스 기사에서 언론사, 기사 제목, 작성자, 작성일, 기사 내용 요약(20-30자), 분야를 추출해주세요.
     추출한 정보의 중요도(0.00~1.00 사이)와 추출에 대한 확신도(0.00~1.00 사이)도 함께 평가해 주세요.

     ```기사 내용
     {news_article}
     ```
     """)
])

analysis_chain = template | structured_gemini

# 실제 뉴스 기사로 테스트 시:
# result = analysis_chain.invoke({"news_article": news_article})
# print(f"언론사: {result.언론사}")
# print(f"기사 제목: {result.기사_제목}")
# print(f"중요도: {result.중요도:.2f}")
```

## 🔍 참고 자료

### 공식 문서
- [LangChain 공식 문서](https://python.langchain.com/docs/introduction/)
- [Google Generative AI 통합](https://python.langchain.com/docs/integrations/chat/google_generative_ai/)
- [OpenAI 통합](https://python.langchain.com/docs/integrations/chat/openai/)
- [프롬프트 템플릿 가이드](https://python.langchain.com/docs/concepts/prompt_templates/)

### 학습 자료
- [LangChain Cookbook](https://github.com/langchain-ai/langchain/tree/master/cookbook)
- [Pydantic 모델 문서](https://docs.pydantic.dev/latest/)
- [구조화된 출력 가이드](https://python.langchain.com/docs/how_to/structured_output/)

### 개발 도구
- [LangSmith](https://smith.langchain.com/) - 디버깅 및 모니터링
- [LangServe](https://github.com/langchain-ai/langserve) - API 배포
- [Google AI Studio](https://aistudio.google.com/) - Gemini API 키 발급
- [OpenAI Platform](https://platform.openai.com/) - OpenAI API 키 발급

### 추가 학습
- LangChain Expression Language (LCEL) 학습
- 벡터 데이터베이스와 임베딩 모델 연동
- RAG (Retrieval-Augmented Generation) 시스템 구축
- 에이전트 시스템과 도구 통합