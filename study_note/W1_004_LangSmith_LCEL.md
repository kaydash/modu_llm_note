# W1_004_LangSmith_LCEL.md - LangSmith와 LCEL 마스터하기

## 🎯 학습 목표
- Langfuse를 활용한 LLM Observability 구축 능력 습득
- LCEL(LangChain Expression Language) 파이프라인 구성법 학습
- Runnable 인터페이스의 다양한 실행 패턴 이해
- 복잡한 체인 구조 설계 및 성능 모니터링 기법 습득

## 📚 핵심 개념

### Langfuse (LLM Observability)
- **관찰성(Observability)**: LLM 애플리케이션의 실행 과정과 성능 추적
- **디버깅**: 체인과 에이전트의 효과적인 디버깅 지원
- **성능 측정**: 토큰 사용량, 실행 시간, 비용 분석
- **추적**: 전체 파이프라인의 데이터 흐름 시각화

### LCEL (LangChain Expression Language)
- **선언적 체이닝**: `|` 연산자를 사용한 컴포넌트 연결
- **재사용성**: 정의된 체인을 다른 체인의 컴포넌트로 활용 가능
- **다양한 실행 방식**: `.invoke()`, `.batch()`, `.stream()`, `.astream()` 지원
- **자동 최적화**: 배치 처리 시 효율적인 작업 수행

## 🔧 환경 설정

### Langfuse 설정

#### 1. 환경 변수 설정
```bash
# .env 파일
LANGFUSE_ENABLED=true
LANGFUSE_HOST=your_langfuse_host_url
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_PUBLIC_KEY=your_public_key
```

#### 2. Langfuse 초기화
```python
from dotenv import load_dotenv
import os
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

load_dotenv()

# Langfuse 클라이언트 초기화
langfuse_client = Langfuse(
    public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
    secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
    host=os.getenv('LANGFUSE_HOST')
)

# CallbackHandler 초기화
langfuse_handler = CallbackHandler()

print("✅ Langfuse 초기화 완료")
```

#### 3. 연결 상태 진단
```python
import requests

# 서버 연결 테스트
host = os.getenv('LANGFUSE_HOST')
response = requests.get(f"{host}/api/public/health", timeout=10)
print(f"서버 상태: {response.status_code}")

# 인증 정보 확인
public_key = os.getenv('LANGFUSE_PUBLIC_KEY')
secret_key = os.getenv('LANGFUSE_SECRET_KEY')
print(f"PUBLIC_KEY: {public_key[:10]}...")
print(f"SECRET_KEY: {secret_key[:10]}...")
```

## 💻 LCEL 기본 구조

### 1. Prompt + LLM 체인

#### 기본 구조
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Langfuse 추적이 활성화된 모델
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    callbacks=[langfuse_handler]
)

# 프롬프트 템플릿
prompt = PromptTemplate.from_template(
    "당신은 {topic} 분야의 전문가입니다. {topic}에 관한 다음 질문에 답변해주세요.\n질문: {question}\n답변:"
)

# 체인 구성
chain = prompt | llm

# 실행
response = chain.invoke({
    "topic": "화학(Chemistry)",
    "question": "탄소의 원자 번호는 무엇인가요?"
})

print(f"답변: {response.content}")
```

### 2. Prompt + LLM + Output Parser

#### 완전한 파이프라인
```python
# 출력 파서 추가
output_parser = StrOutputParser()

# 전체 파이프라인
chain = prompt | llm | output_parser

# 실행 결과는 문자열
result = chain.invoke({
    "topic": "화학(Chemistry)",
    "question": "탄소의 원자 번호는 무엇인가요?"
})

print(f"결과: {result}")  # 문자열 출력
```

#### 입력 스키마 확인
```python
# 체인의 입력 구조 확인
schema = chain.input_schema.model_json_schema()
print(f"필수 입력: {schema['required']}")
print(f"속성: {list(schema['properties'].keys())}")
```

## 🔄 Runnable 인터페이스

### 1. RunnableSequence (순차 실행)

```python
from langchain_core.runnables import RunnableSequence

# 번역 체인 구성
translation_prompt = PromptTemplate.from_template("'{text}'를 영어로 번역해주세요. 번역된 문장만을 출력해주세요.")

# RunnableSequence 생성
translation_chain = RunnableSequence(
    first=translation_prompt,
    middle=[llm],
    last=output_parser
)

# 실행
result = translation_chain.invoke({"text": "안녕하세요"})
print(f"번역 결과: {result}")  # "Hello"
```

### 2. RunnableParallel (병렬 실행)

#### 질문 분석 체인
```python
from langchain_core.runnables import RunnableParallel
from operator import itemgetter

# 질문 분류 체인
question_prompt = PromptTemplate.from_template("""
다음 카테고리 중 하나로 입력을 분류하세요:
- 화학(Chemistry)
- 물리(Physics)
- 생물(Biology)

질문: {question}
분류:""")

question_chain = question_prompt | llm | output_parser

# 언어 감지 체인
language_prompt = PromptTemplate.from_template("""
입력된 텍스트의 언어를 분류하세요:
- 영어(English)
- 한국어(Korean)
- 기타(Others)

입력: {question}
언어:""")

language_chain = language_prompt | llm | output_parser

# 병렬 실행 체인
parallel_chain = RunnableParallel({
    "topic": question_chain,
    "language": language_chain,
    "question": itemgetter("question")
})

# 실행
result = parallel_chain.invoke({
    "question": "탄소의 원자 번호는 무엇인가요?"
})

print(f"주제: {result['topic']}")      # 화학(Chemistry)
print(f"언어: {result['language']}")   # 한국어(Korean)
print(f"질문: {result['question']}")   # 원본 질문
```

### 3. RunnablePassthrough (데이터 투과)

```python
from langchain_core.runnables import RunnablePassthrough
import re

# 입력 데이터를 보존하며 변환
runnable = RunnableParallel({
    "passed": RunnablePassthrough(),  # 원본 보존
    "modified": lambda x: int(re.search(r'\d+', x).group())  # 숫자 추출
})

result = runnable.invoke('탄소의 원자 번호는 6입니다.')
print(f"원본: {result['passed']}")     # '탄소의 원자 번호는 6입니다.'
print(f"추출된 숫자: {result['modified']}")  # 6
```

### 4. RunnableLambda (커스텀 함수)

```python
from langchain_core.runnables import RunnableLambda

# 전처리 함수
def preprocess_text(text: str) -> str:
    return text.strip().lower()

# 후처리 함수
def postprocess_response(response) -> dict:
    response_text = response.content
    return {
        "processed_response": response_text.upper(),
        "length": len(response_text)
    }

# 처리 파이프라인 구성
processing_chain = (
    RunnableLambda(preprocess_text) |
    PromptTemplate.from_template("다음 주제에 대해 영어 한 문장으로 설명해주세요: {0}") |
    llm |
    RunnableLambda(postprocess_response)
)

# 실행
result = processing_chain.invoke("  Artificial Intelligence  ")
print(f"처리된 응답: {result['processed_response']}")
print(f"응답 길이: {result['length']}")
```

## 🚀 실습해보기

### 실습: 텍스트 요약 및 감정 분석 시스템

**목표**: 사용자 입력을 요약하고 감정을 분석하는 LCEL 파이프라인 구축

#### 단계별 구현
```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from operator import itemgetter

# 모델 설정 (Langfuse 추적 포함)
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    callbacks=[langfuse_handler]
)

# 요약 프롬프트
summarize_prompt = PromptTemplate.from_template(
    "다음 텍스트를 핵심 내용만 간단히 요약해주세요:\n\n{text}\n\n요약:"
)

# 감정 분석 프롬프트
sentiment_prompt = PromptTemplate.from_template(
    "다음 요약된 내용의 감정을 분석해주세요. '긍정', '부정', '중립' 중 하나로만 답변하세요:\n\n{summary}\n\n감정:"
)

# 출력 파서
output_parser = StrOutputParser()

# 체인 구성
summarize_chain = summarize_prompt | model | output_parser
sentiment_chain = sentiment_prompt | model | output_parser

# 전체 파이프라인
chain = (
    {"text": itemgetter("text")} |          # 입력 텍스트 전달
    {"summary": summarize_chain} |          # 요약 실행
    RunnableParallel({
        "summary": itemgetter("summary"),   # 요약 결과 전달
        "sentiment": sentiment_chain        # 감정 분석 실행
    })
)

# 테스트 텍스트
text = """오늘 시험을 봤습니다. 준비를 열심히 했기 때문에 긴장했지만 문제를 잘 풀 수 있었습니다.
결과적으로 만점을 받았고 매우 기뻤습니다.
선생님께서도 칭찬해 주셔서 보람을 느꼈습니다.
노력하면 좋은 결과가 따른다는 것을 다시 깨달았습니다."""

# 실행
result = chain.invoke({"text": text})

print(f"요약: {result['summary']}")
print(f"감정: {result['sentiment']}")
```

## 📋 해답

### 텍스트 요약 및 감정 분석 시스템 해답

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from operator import itemgetter

# Langfuse 추적이 활성화된 모델
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    callbacks=[langfuse_handler]
)

# 프롬프트 템플릿 정의
summarize_prompt = PromptTemplate.from_template(
    "다음 텍스트를 핵심 내용만 간단히 요약해주세요:\n\n{text}\n\n요약:"
)

sentiment_prompt = PromptTemplate.from_template(
    "다음 요약된 내용의 감정을 분석해주세요. '긍정', '부정', '중립' 중 하나로만 답변하세요:\n\n{summary}\n\n감정:"
)

# 출력 파서
output_parser = StrOutputParser()

# 개별 체인 구성
summarize_chain = summarize_prompt | model | output_parser
sentiment_chain = sentiment_prompt | model | output_parser

# 전체 파이프라인 구성
chain = (
    {"text": itemgetter("text")} |
    {"summary": summarize_chain} |
    RunnableParallel({
        "summary": itemgetter("summary"),
        "sentiment": sentiment_chain
    })
)

# 테스트 실행
test_text = """오늘 시험을 봤습니다. 준비를 열심히 했기 때문에 긴장했지만 문제를 잘 풀 수 있었습니다.
결과적으로 만점을 받았고 매우 기뻤습니다.
선생님께서도 칭찬해 주셔서 보람을 느꼈습니다.
노력하면 좋은 결과가 따른다는 것을 다시 깨달았습니다."""

result = chain.invoke({"text": test_text})

print("=== 처리 결과 ===")
print(f"요약: {result['summary']}")          # 시험에서 만점을 받아 기쁘고, 노력의 중요성을 다시 깨달았다.
print(f"감정: {result['sentiment']}")        # 긍정

# Langfuse 데이터 전송
langfuse_client.flush()
print("✅ Langfuse 추적 데이터 전송 완료")
```

## 🔍 Langfuse 모니터링

### 주요 추적 기능

#### 1. 체인 실행 추적
- 각 컴포넌트별 실행 시간 측정
- 입력/출력 데이터 기록
- 토큰 사용량 및 비용 분석

#### 2. 성능 분석
- 병목 지점 식별
- 최적화 포인트 발견
- 배치 처리 효율성 측정

#### 3. 디버깅 지원
- 전체 데이터 흐름 시각화
- 에러 발생 지점 추적
- 파라미터 변화에 따른 성능 비교

### 웹 인터페이스 활용
```python
# Langfuse 웹에서 확인 가능한 정보
print(f"🔗 Langfuse 웹 인터페이스: {os.getenv('LANGFUSE_HOST')}")
print("📊 확인 가능한 정보:")
print("- 트레이스별 실행 시간")
print("- 토큰 사용량 및 비용")
print("- 프롬프트와 응답 데이터")
print("- 성능 메트릭 및 분석")
```

## 🎯 고급 활용법

### 1. 배치 처리
```python
# 여러 입력을 배치로 처리
inputs = [
    {"text": "첫 번째 텍스트..."},
    {"text": "두 번째 텍스트..."},
    {"text": "세 번째 텍스트..."}
]

# 배치 실행
results = chain.batch(inputs)

for i, result in enumerate(results):
    print(f"결과 {i+1}: 요약={result['summary']}, 감정={result['sentiment']}")
```

### 2. 스트리밍 처리
```python
# 실시간 스트리밍
for chunk in chain.stream({"text": "분석할 텍스트..."}):
    print(f"처리 중: {chunk}")
```

### 3. 비동기 처리
```python
import asyncio

async def process_async():
    result = await chain.ainvoke({"text": "비동기 처리할 텍스트..."})
    return result

# 비동기 실행
result = asyncio.run(process_async())
```

## 📚 참고 자료

### 공식 문서
- [LangChain LCEL 가이드](https://python.langchain.com/docs/concepts/lcel/)
- [Langfuse 문서](https://langfuse.com/docs)
- [Runnable 인터페이스](https://python.langchain.com/docs/concepts/runnables/)
- [CallbackHandler 사용법](https://python.langchain.com/docs/integrations/providers/langfuse/)

### 학습 자료
- [LCEL 쿡북](https://github.com/langchain-ai/langchain/tree/master/cookbook)
- [Langfuse 예제 모음](https://github.com/langfuse/langfuse/tree/main/cookbook)
- [성능 최적화 가이드](https://python.langchain.com/docs/how_to/chain_performance/)

### 개발 도구
- [Langfuse Cloud](https://cloud.langfuse.com/) - 클라우드 버전
- [Langfuse Self-hosted](https://github.com/langfuse/langfuse) - 자체 호스팅
- [LangSmith](https://smith.langchain.com/) - LangChain 공식 모니터링

### 추가 학습
- 복잡한 체인 구조 설계 패턴
- 성능 최적화 및 비용 관리
- 프로덕션 환경에서의 모니터링 전략
- A/B 테스트를 위한 체인 비교 방법