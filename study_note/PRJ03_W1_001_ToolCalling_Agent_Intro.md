# Tool Calling & Agent - 도구 호출과 에이전트 개념

## 📚 학습 목표
- Tool Calling 메커니즘을 이해하고 LLM이 외부 시스템과 상호작용하는 방법을 학습한다
- `@tool` 데코레이터를 사용하여 커스텀 도구를 생성하고 모델에 바인딩하는 방법을 구현한다
- LangChain Agent의 자동 의사결정 과정을 이해하고 실무에 활용할 수 있다
- ReAct 패턴 기반의 Agent를 구현하여 복잡한 작업을 자동화한다

## 🔑 핵심 개념

### Tool Calling이란?
**Tool Calling**은 LLM이 외부 시스템과 상호작용하기 위한 함수 호출 메커니즘입니다. LLM은 정의된 도구나 함수를 통해 데이터베이스, API, 외부 서비스 등과 통신하고 작업을 수행할 수 있습니다.

**주요 특징:**
- **구조화된 출력**: JSON 스키마 기반으로 API나 데이터베이스 요구사항 충족
- **스키마 자동 인식**: 함수 시그니처로부터 자동으로 입력 스키마 생성
- **유효성 검증**: 타입 힌트를 활용한 자동 입력 검증
- **시스템 통합**: 외부 시스템과의 안전하고 효율적인 통신

![Tool Calling Concept](https://python.langchain.com/assets/images/tool_calling_concept-552a73031228ff9144c7d59f26dedbbf.png)

*참조: [LangChain Tool Calling Documentation](https://python.langchain.com/docs/concepts/tool_calling/)*

### Agent란?
**Agent**는 LLM을 의사결정 엔진으로 사용하여 작업을 자율적으로 수행하는 시스템입니다. 사용자의 요청을 분석하고, 필요한 도구를 선택하며, 결과를 해석하여 최종 답변을 제공합니다.

**Agent의 동작 원리:**
1. **사용자 요청 분석**: 질문의 의도와 필요한 정보 파악
2. **도구 선택**: 사용 가능한 도구 중 적절한 도구 결정
3. **도구 실행**: 선택한 도구를 필요한 파라미터로 실행
4. **결과 해석**: 도구 실행 결과를 분석하고 추가 작업 필요 여부 판단
5. **응답 생성**: 최종 답변을 사용자에게 제공

### ReAct 패턴
**ReAct**(Reasoning + Acting)는 Agent가 추론(Reason)과 행동(Act)을 반복하며 문제를 해결하는 패턴입니다.

```
사용자 질문 → [추론: 무엇을 해야 하나?] → [행동: 도구 실행] → [관찰: 결과 확인] → [추론: 추가 작업 필요?] → ... → 최종 답변
```

---

## 🛠 환경 설정

### 필요한 라이브러리
```bash
pip install langchain langchain-openai langchain-chroma langchain-core langgraph python-dotenv
```

### API 키 설정
`.env` 파일에 OpenAI API 키를 설정합니다:
```bash
OPENAI_API_KEY=your_api_key_here
```

### 기본 환경 구성
```python
# 환경 변수 로드
from dotenv import load_dotenv
load_dotenv()

# 기본 라이브러리
import os
from glob import glob
from pprint import pprint
import json
```

---

# 1단계: Tool Creation - 도구 생성

## 1.1 @tool 데코레이터 기본 사용법

### 핵심 개념
`@tool` 데코레이터는 일반 Python 함수를 LLM이 호출할 수 있는 도구로 변환합니다. 함수의 시그니처와 docstring을 분석하여 자동으로 스키마를 생성합니다.

### 날씨 조회 도구 예제
```python
from langchain_core.tools import tool
from typing import Literal

@tool
def get_weather(city: Literal["서울", "부산", "대구", "인천", "광주"]):
    """한국 주요 도시의 날씨 정보를 가져옵니다."""
    # 실제로는 API 호출을 하지만, 여기서는 예제 데이터 사용
    weather_data = {
        "서울": "맑음",
        "부산": "흐림",
        "대구": "맑음",
        "인천": "비",
        "광주": "구름많음"
    }

    if city in weather_data:
        return f"{city}은(는) {weather_data[city]}"
    else:
        raise AssertionError("지원하지 않는 도시입니다")
```

**코드 설명:**
- `@tool`: 함수를 LangChain 도구로 등록
- `Literal` 타입: 허용되는 값을 명시적으로 제한
- `docstring`: 도구의 용도를 LLM에게 설명 (매우 중요!)

### 도구 실행 테스트
```python
# 도구를 직접 실행
result = get_weather.invoke("서울")
print(result)  # 출력: 서울은(는) 맑음
```

## 1.2 ChromaDB 검색 도구 구현

### 벡터 저장소 설정
```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# 임베딩 모델 초기화
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ChromaDB 로드 (이전 프로젝트에서 생성한 벡터 저장소)
chroma_db = Chroma(
    collection_name="db_korean_cosine_metadata",
    embedding_function=embeddings,
    persist_directory="./chroma_db",  # 프로젝트 2에서 복사한 디렉토리
)

print(f"ChromaDB에 저장된 문서 개수: {chroma_db._collection.count()}")
```

### 검색기 구성 및 테스트
```python
# 검색기 생성 (상위 2개 문서 반환)
chroma_k_retriever = chroma_db.as_retriever(
    search_kwargs={"k": 2},
)

# 검색 테스트
query = "리비안은 언제 사업을 시작했나요?"
retrieved_docs = chroma_k_retriever.invoke(query)

print(f"쿼리: {query}")
print("검색 결과:")
for doc in retrieved_docs:
    # 문서 내용 일부와 출처 정보 출력
    print(f"- {doc.page_content[:100]}... [출처: {doc.metadata['source']}]")
```

### 문서 검색 도구 생성
```python
from langchain.tools import tool

@tool
def search_db(query: str):
    """리비안, 테슬라 회사에 대한 정보를 관련 데이터베이스에서 검색합니다."""
    return chroma_k_retriever.invoke(query)

# 도구 실행 테스트
result = search_db.invoke("리비안은 언제 사업을 시작했나요?")
print(f"검색된 문서 개수: {len(result)}")
print(f"첫 번째 문서 내용: {result[0].page_content[:200]}...")
```

**실무 팁:**
- docstring은 LLM이 도구를 선택할 때 참고하므로 명확하고 구체적으로 작성
- 타입 힌트를 정확히 지정하면 LLM이 올바른 인자를 생성
- 에러 처리를 포함하면 더 안정적인 도구 실행 가능

### 대안 구현 방법 (참고)

검색 결과를 문자열로 결합하는 방식도 가능합니다:

```python
# 대안: 검색 결과를 하나의 문자열로 결합하는 방식
@tool
def search_documents(query: str) -> str:
    '''ChromaDB 벡터 저장소에서 테슬라와 리비안 관련 문서를 검색합니다.

    Args:
        query: 검색할 질문이나 키워드

    Returns:
        검색된 문서 내용들을 결합한 문자열
    '''
    retriever = chroma_db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)

    # 검색된 문서들을 하나의 문자열로 결합
    result = "\n\n---\n\n".join([doc.page_content for doc in docs])
    return result

# 도구 테스트
print("\n=== 문서 검색 도구 테스트 ===")
result = search_documents.invoke("리비안의 주요 특징은?")
print(result[:300] + "...")
```

## 1.3 Tool Calling 사용 시 고려사항

Tool Calling을 효과적으로 활용하기 위해서는 다음 사항들을 고려해야 합니다:

### 모델 호환성
- **모델 호환성**이 Tool Calling 성능에 직접 영향을 미칩니다
- OpenAI의 GPT-4, GPT-3.5-turbo 등 최신 모델이 더 정확한 도구 선택을 수행합니다
- 모델마다 지원하는 도구 개수와 복잡도가 다를 수 있습니다

### 명확한 도구 정의
- **명확한 도구 정의**가 모델의 이해도와 활용도를 향상시킵니다
- docstring을 구체적으로 작성하여 도구의 용도와 사용법을 명시하세요
- 파라미터 이름과 타입을 명확히 정의하세요

### 단순한 기능
- **단순한 기능**의 도구가 더 효과적으로 작동합니다
- 하나의 도구는 하나의 명확한 기능만 수행하도록 설계하세요
- 복잡한 작업은 여러 단순한 도구로 분해하는 것이 좋습니다

### 과다한 도구
- **과다한 도구**는 모델 성능 저하를 유발할 수 있습니다
- 한 번에 너무 많은 도구를 제공하면 모델이 적절한 도구를 선택하기 어려워집니다
- 일반적으로 5-10개 이하의 도구를 권장합니다

---

# 2단계: Tool Binding - 모델에 도구 연결

## 2.1 모델 초기화 및 도구 바인딩

### 모델 초기화
```python
from langchain.chat_models import init_chat_model

# LLM 모델 초기화
model = init_chat_model(
    "openai:gpt-4.1-nano",
    temperature=0.7,      # 응답의 다양성 조절 (0: 결정적, 1: 창의적)
    timeout=30,           # API 호출 타임아웃 (초)
    max_tokens=1000,      # 최대 생성 토큰 수
)
```

**대안 방법 (ChatOpenAI 직접 사용):**
```python
from langchain_openai import ChatOpenAI

# 모델
model = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

# 도구 목록
tools = [get_weather]

# 도구를 모델에 바인딩 (bind_tools 메소드 사용)
model_with_tools = model.bind_tools([get_weather])
```

### 도구를 모델에 바인딩
```python
# 사용할 도구 목록
tools = [get_weather, search_db]

# 도구를 모델에 바인딩
model_with_tools = model.bind_tools(tools)
```

**bind_tools()의 역할:**
1. 도구의 스키마 정보를 모델에 전달
2. 모델이 도구를 인식하고 필요 시 호출할 수 있도록 설정
3. 자동으로 입력 유효성 검증 수행

## 2.2 모델 응답 구조 이해

### 기본 사용 예제
```python
# 사용자 쿼리를 모델에 전달
result = model_with_tools.invoke("서울 날씨 어때?")

# 결과 출력
print(result)
```

**출력 예시 (AIMessage 객체):**
```
content=''
tool_calls=[{
    'name': 'get_weather',
    'args': {'city': '서울'},
    'id': 'call_DvSK2oX3eat9PVOr9JSL8obk',
    'type': 'tool_call'
}]
usage_metadata={'input_tokens': 109, 'output_tokens': 14, 'total_tokens': 123}
```

### AIMessage 객체 구조 분석
```python
# AIMessage의 주요 속성 확인
for key in dict(result).keys():
    print(f"{key}: {dict(result)[key]}\n")
```

**주요 속성:**
- `content`: 텍스트 응답 (도구 호출 시에는 보통 비어있음)
- `tool_calls`: 호출할 도구 정보 리스트
- `response_metadata`: 토큰 사용량, 모델 정보 등
- `usage_metadata`: 입력/출력 토큰 통계

### tool_calls 상세 정보
```python
pprint(result.tool_calls)
```

**출력:**
```python
[{
    'name': 'get_weather',           # 호출할 도구 이름
    'args': {'city': '서울'},         # 도구에 전달할 인자
    'id': 'call_DvSK2oX3...',       # 고유 호출 ID
    'type': 'tool_call'              # 호출 타입
}]
```

## 2.3 실습: 검색 도구 바인딩

```python
# search_db 도구만 바인딩
model_with_search = model.bind_tools([search_db])

# 쿼리 실행
search_result = model_with_search.invoke("테슬라의 배터리 기술에 대해 알려주세요")

# 결과 확인
print("=== 모델 응답 결과 ===")
print(search_result)
print("\n=== Tool Calls 확인 ===")
pprint(search_result.tool_calls)
```

**실행 결과 분석:**
```python
[{
    'name': 'search_db',
    'args': {'query': '테슬라 배터리 기술'},  # 모델이 자동으로 검색어 최적화
    'id': 'call_WxsDdwoMup19o9i1H5pEoJSF',
    'type': 'tool_call'
}]
```

모델은 "테슬라의 배터리 기술에 대해 알려주세요"라는 자연어를 분석하여 `search_db` 도구를 호출하기로 결정하고, 검색에 적합한 쿼리 "테슬라 배터리 기술"을 자동으로 생성했습니다.

---

# 3단계: Tool Calling - 도구 호출 과정

## 3.1 모델의 도구 선택 과정

LLM은 다음과 같은 과정으로 도구를 선택합니다:

1. **사용자 요청 분석**: 질문의 의도 파악
2. **도구 목록 검토**: 사용 가능한 도구의 docstring 및 스키마 확인
3. **적합도 평가**: 각 도구가 요청을 처리할 수 있는지 판단
4. **도구 선택 및 파라미터 생성**: 가장 적합한 도구와 필요한 인자 결정

## 3.2 tool_calls 속성 분석

```python
# AIMessage 객체의 모든 속성 출력
print("=== search_result의 모든 속성 ===")
for k in dict(search_result).keys():
    print(f"\n{k}: ")
    print(dict(search_result)[k])
    print("-"*100)
```

### 주요 속성 설명

**1. content** - 텍스트 응답
- 도구 호출이 필요한 경우 보통 비어있음
- 도구 실행 후 최종 답변 생성 시 채워짐

**2. tool_calls** - 도구 호출 정보
```python
{
    'name': 'search_db',                    # 호출할 도구 이름
    'args': {'query': '테슬라 배터리 기술'},  # 도구에 전달할 인자 (JSON 형식)
    'id': 'call_WxsDd...',                  # 고유 호출 ID (추적용)
    'type': 'tool_call'                     # 호출 타입
}
```

**3. response_metadata** - 응답 메타데이터
- `token_usage`: 입력/출력 토큰 사용량
- `model_name`: 사용된 모델 이름
- `finish_reason`: 응답 종료 이유 ('tool_calls' 또는 'stop')

**4. usage_metadata** - 토큰 사용 통계
```python
{
    'input_tokens': 68,     # 입력 토큰 수
    'output_tokens': 20,    # 출력 토큰 수
    'total_tokens': 88      # 총 토큰 수
}
```

## 3.3 스키마 기반 응답 생성

LLM은 도구의 스키마를 참조하여 정확한 형식의 인자를 생성합니다:

```python
# 도구 정의 (타입 힌트 포함)
@tool
def search_db(query: str):  # str 타입 명시
    """리비안, 테슬라 회사에 대한 정보를 검색합니다."""
    return chroma_k_retriever.invoke(query)

# 모델은 자동으로 str 타입의 query 값을 생성
# 잘못된 타입(예: 숫자)을 전달하면 자동 검증 실패
```

---

# 4단계: Tool Execution - 도구 실행

## 4.1 두 가지 실행 방식

### 방식 1: 직접 인자 전달
```python
# 함수의 인자를 직접 전달하는 방식
result = get_weather.invoke("서울")
print(result)  # 출력: 서울은(는) 맑음
```

### 방식 2: ToolCall 객체 전달
```python
# ToolCall 객체를 전달하는 방식 (Agent에서 사용)
tool_message = get_weather.invoke(result.tool_calls[0])
print(tool_message)
```

**출력 (ToolMessage 객체):**
```
ToolMessage(
    content='서울은(는) 맑음',
    name='get_weather',
    tool_call_id='call_DvSK2oX3eat9PVOr9JSL8obk'
)
```

## 4.2 ToolMessage 객체 이해

ToolMessage는 도구 실행 결과를 대화 흐름에 통합하기 위한 메시지 객체입니다.

```python
# ToolCall 객체로 도구 실행
if search_result.tool_calls:
    # 첫 번째 tool call을 사용하여 도구 실행
    tool_message = search_db.invoke(search_result.tool_calls[0])

    print("=== ToolMessage 객체 ===")
    print(f"타입: {type(tool_message)}")
    print(f"도구 이름: {tool_message.name}")
    print(f"Tool Call ID: {tool_message.tool_call_id}")
    print(f"실행 결과: {tool_message.content}")
```

### ToolMessage의 주요 속성
- `content`: 도구 실행 결과 (문서 리스트, 텍스트 등)
- `name`: 실행된 도구 이름
- `tool_call_id`: 어떤 tool call에 대한 응답인지 추적

## 4.3 실행 결과 처리

```python
# 검색 결과 확인
if search_result.tool_calls:
    # 도구 실행
    tool_message = search_db.invoke(search_result.tool_calls[0])

    # 결과가 Document 리스트인 경우 처리
    documents = tool_message.content
    print(f"검색된 문서 개수: {len(documents)}")

    for idx, doc in enumerate(documents):
        print(f"\n문서 {idx+1}:")
        print(f"출처: {doc.metadata['source']}")
        print(f"내용: {doc.page_content[:200]}...")
```

---

# 5단계: Agent 구현

## 5.1 Agent란 무엇인가?

Agent는 LLM을 "두뇌"로 사용하여 복잡한 작업을 자율적으로 수행하는 시스템입니다. 도구를 선택하고, 실행하고, 결과를 해석하여 최종 답변을 생성하는 전체 과정을 자동화합니다.

### ReAct Agent의 동작 흐름
```
1. 사용자 질문 → "서울 날씨는?"
2. [Agent 추론] "날씨를 알려면 get_weather 도구 필요"
3. [Agent 행동] get_weather.invoke("서울")
4. [도구 응답] "서울은(는) 맑음"
5. [Agent 추론] "충분한 정보를 얻었으니 답변 생성"
6. 최종 답변 → "서울의 날씨는 맑습니다."
```

## 5.2 create_react_agent로 Agent 생성

### 필요한 라이브러리
```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
```

### 기본 Agent 구현
```python
# 모델 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 사용할 도구 목록
tools = [get_weather, calculate]

# 시스템 프롬프트 정의
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 사용자의 요청을 처리하는 AI Assistant입니다."),
    ("placeholder", "{messages}"),
])

# Agent 생성
agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=prompt
)

print("에이전트가 성공적으로 생성되었습니다.")
```

### 계산 도구 추가
```python
@tool
def calculate(expression: str) -> float:
    """수학 계산을 수행합니다. 사칙연산을 포함한 Python 표현식을 평가합니다."""
    return eval(expression)

# 도구 실행 테스트
result = calculate.invoke("3+2")
print(result)  # 출력: 5
```

## 5.3 Agent 실행 및 결과 분석

### 기본 실행
```python
# Agent 실행
response = agent.invoke(
    {"messages": [{"role": "user", "content": "서울의 날씨는 어떤가요?"}]},
)

# 결과 출력 (전체 메시지 히스토리 포함)
pprint(response)
```

**응답 구조:**
```python
{
    'messages': [
        HumanMessage(content='서울의 날씨는 어떤가요?'),          # 사용자 질문
        AIMessage(tool_calls=[...]),                           # Agent의 도구 호출 결정
        ToolMessage(content='서울은(는) 맑음'),                # 도구 실행 결과
        AIMessage(content='서울의 날씨는 맑습니다.')            # 최종 답변
    ]
}
```

### 메시지 히스토리 확인
```python
# 각 메시지를 보기 좋게 출력
for msg in response['messages']:
    msg.pretty_print()
```

**출력 예시:**
```
================================ Human Message =================================
서울의 날씨는 어떤가요?

================================== Ai Message ==================================
Tool Calls:
  get_weather (call_Kb0PQKayN2FCfdI9wpnrkY3a)
  Args:
    city: 서울

================================= Tool Message =================================
Name: get_weather
서울은(는) 맑음

================================== Ai Message ==================================
서울의 날씨는 맑습니다.
```

### 계산 도구 사용 예제
```python
# 수학 계산 요청
response = agent.invoke(
    {"messages": [{"role": "user", "content": "32 더하기 18은 얼마인가요?"}]},
)

# 결과 출력
for msg in response['messages']:
    msg.pretty_print()
```

**Agent의 동작:**
1. "계산이 필요하다" → calculate 도구 선택
2. 적절한 표현식 생성 → `"32 + 18"`
3. 도구 실행 → `50`
4. 결과를 자연어로 변환 → "32 더하기 18은 50입니다."

## 5.4 스트리밍 모드로 중간 과정 확인

```python
# 스트리밍 모드로 Agent 실행
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "32 곱하기 18은 얼마인가요?"}]},
    stream_mode="values"
):
    # 각 단계의 메시지 출력
    chunk["messages"][-1].pretty_print()
```

**스트리밍 출력:**
```
1단계: Human Message - "32 곱하기 18은 얼마인가요?"
2단계: Ai Message - Tool Calls: calculate(expression: 32 * 18)
3단계: Tool Message - 576
4단계: Ai Message - "32 곱하기 18은 576입니다."
```

스트리밍을 통해 Agent가 어떻게 사고하고 도구를 선택하는지 실시간으로 확인할 수 있습니다.

## 5.5 실전 예제: 문서 검색 Agent

```python
# 문서 검색 전문 Agent 시스템 프롬프트
system_message = """당신은 테슬라와 리비안 전기차에 대한 전문 상담 AI입니다.
사용자의 질문에 대해 ChromaDB에 저장된 문서를 검색하여 정확한 정보를 제공하세요.

검색된 문서 내용을 바탕으로:
1. 질문에 대한 직접적인 답변을 제공하세요
2. 관련된 세부 정보를 포함하세요
3. 검색 결과가 없으면 솔직하게 알려주세요
"""

# 프롬프트 템플릿 생성
doc_prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    ("placeholder", "{messages}"),
])

# 문서 검색 Agent 생성
doc_agent = create_react_agent(
    model=llm,
    tools=[search_db],  # 문서 검색 도구만 사용
    prompt=doc_prompt
)

print("문서 검색 에이전트가 생성되었습니다.")
```

### Agent 실행 예제
```python
# 테슬라 관련 질문
print("=== 테슬라 관련 질문 ===")
response1 = doc_agent.invoke(
    {"messages": [{"role": "user", "content": "테슬라의 주요 기술적 특징은 무엇인가요?"}]}
)

for msg in response1['messages']:
    msg.pretty_print()
```

**Agent의 동작 분석:**
1. 질문 분석: "테슬라 기술 정보가 필요함"
2. 도구 선택: `search_db` 호출 결정
3. 검색어 최적화: "테슬라 주요 기술적 특징"
4. 문서 검색 실행
5. 검색 결과를 바탕으로 구조화된 답변 생성

```python
# 리비안 관련 질문
print("\n=== 리비안 관련 질문 ===")
response2 = doc_agent.invoke(
    {"messages": [{"role": "user", "content": "리비안 전기차의 특징을 알려주세요"}]}
)

# 최종 답변만 출력
final_message = response2['messages'][-1]
print(f"\n최종 답변:\n{final_message.content}")
```

---

# 🎯 실습 문제

## 실습 1: 다중 도시 날씨 조회 도구
**난이도: ⭐⭐**

여러 도시의 날씨를 한 번에 조회할 수 있는 도구를 만들어보세요.

```python
# TODO: 다음 요구사항을 만족하는 도구를 작성하세요
# 1. 여러 도시 이름을 리스트로 받습니다
# 2. 각 도시의 날씨를 조회하여 딕셔너리로 반환합니다
# 3. 존재하지 않는 도시는 "정보 없음"으로 표시합니다

@tool
def get_multiple_weather(cities: list[str]):
    """여러 도시의 날씨 정보를 한 번에 조회합니다."""
    # 여기에 코드를 작성하세요
    pass
```

## 실습 2: 검색 결과 필터링 도구
**난이도: ⭐⭐⭐**

검색 결과를 특정 회사로 필터링하는 도구를 만들어보세요.

```python
# TODO: 다음 요구사항을 만족하는 도구를 작성하세요
# 1. 검색어와 회사 이름(테슬라/리비안)을 인자로 받습니다
# 2. ChromaDB에서 검색 후 해당 회사 문서만 필터링합니다
# 3. 필터링된 문서 리스트를 반환합니다

@tool
def search_by_company(query: str, company: Literal["테슬라", "리비안"]):
    """특정 회사의 정보만 검색합니다."""
    # 여기에 코드를 작성하세요
    pass
```

## 실습 3: 간단한 계산기 Agent
**난이도: ⭐⭐**

사칙연산을 수행하는 4개의 도구를 만들고, 이를 사용하는 Agent를 구현하세요.

```python
# TODO: 다음 도구들을 작성하세요
@tool
def add(a: float, b: float):
    """두 수를 더합니다."""
    pass

@tool
def subtract(a: float, b: float):
    """두 수를 뺍니다."""
    pass

@tool
def multiply(a: float, b: float):
    """두 수를 곱합니다."""
    pass

@tool
def divide(a: float, b: float):
    """두 수를 나눕니다. 0으로 나누기는 에러를 발생시킵니다."""
    pass

# TODO: 위 도구들을 사용하는 Agent를 생성하세요
# calculator_agent = create_react_agent(...)
```

## 실습 4: 멀티 도구 Agent
**난이도: ⭐⭐⭐⭐**

날씨 조회, 문서 검색, 계산을 모두 수행할 수 있는 범용 Agent를 만들어보세요.

```python
# TODO: 다음 요구사항을 만족하는 Agent를 작성하세요
# 1. get_weather, search_db, calculate 도구를 모두 사용
# 2. 적절한 시스템 프롬프트 작성
# 3. 복합 질문 처리 가능 (예: "서울 날씨를 알려주고, 테슬라와 리비안 중 어느 회사가 먼저 설립되었는지 알려줘")

# 힌트: Agent는 여러 도구를 순차적으로 호출할 수 있습니다
```

## 실습 5: 에러 처리가 있는 견고한 도구
**난이도: ⭐⭐⭐⭐⭐**

예외 상황을 처리하는 안전한 검색 도구를 만들어보세요.

```python
# TODO: 다음 요구사항을 만족하는 도구를 작성하세요
# 1. ChromaDB 연결 실패 시 적절한 에러 메시지 반환
# 2. 검색 결과가 없을 때 유용한 안내 메시지 제공
# 3. 잘못된 입력에 대한 검증 수행
# 4. 로깅을 통한 디버깅 지원

import logging

@tool
def safe_search_db(query: str):
    """안전한 데이터베이스 검색을 수행합니다. 에러 처리가 포함되어 있습니다."""
    # 여기에 코드를 작성하세요
    pass
```

---

# ✅ 솔루션 예시

## 솔루션 1: 다중 도시 날씨 조회

```python
@tool
def get_multiple_weather(cities: list[str]):
    """여러 도시의 날씨 정보를 한 번에 조회합니다."""
    weather_data = {
        "서울": "맑음",
        "부산": "흐림",
        "대구": "맑음",
        "인천": "비",
        "광주": "구름많음"
    }

    # 결과를 저장할 딕셔너리
    results = {}

    for city in cities:
        # 각 도시의 날씨 조회
        if city in weather_data:
            results[city] = weather_data[city]
        else:
            results[city] = "정보 없음"

    return results

# 테스트
result = get_multiple_weather.invoke(["서울", "부산", "제주"])
print(result)  # {'서울': '맑음', '부산': '흐림', '제주': '정보 없음'}
```

## 솔루션 2: 회사별 검색 필터링

```python
@tool
def search_by_company(query: str, company: Literal["테슬라", "리비안"]):
    """특정 회사의 정보만 검색합니다."""
    # 전체 검색 수행
    all_docs = chroma_k_retriever.invoke(query)

    # 회사 이름으로 필터링
    filtered_docs = [
        doc for doc in all_docs
        if doc.metadata.get('company') == company
    ]

    if not filtered_docs:
        return f"{company}에 대한 검색 결과가 없습니다."

    return filtered_docs

# 테스트
result = search_by_company.invoke("배터리 기술", "테슬라")
print(f"검색된 문서 수: {len(result)}")
for doc in result:
    print(f"- {doc.page_content[:100]}...")
```

## 솔루션 3: 계산기 Agent

```python
@tool
def add(a: float, b: float):
    """두 수를 더합니다."""
    return a + b

@tool
def subtract(a: float, b: float):
    """두 수를 뺍니다."""
    return a - b

@tool
def multiply(a: float, b: float):
    """두 수를 곱합니다."""
    return a * b

@tool
def divide(a: float, b: float):
    """두 수를 나눕니다. 0으로 나누기는 에러를 발생시킵니다."""
    if b == 0:
        raise ValueError("0으로 나눌 수 없습니다")
    return a / b

# Agent 생성
calc_tools = [add, subtract, multiply, divide]
calc_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 수학 계산을 도와주는 계산기 AI입니다. 사용자의 계산 요청을 정확히 수행하세요."),
    ("placeholder", "{messages}"),
])

calculator_agent = create_react_agent(
    model=llm,
    tools=calc_tools,
    prompt=calc_prompt
)

# 테스트
response = calculator_agent.invoke({
    "messages": [{"role": "user", "content": "15 곱하기 4를 한 다음, 거기서 23을 빼줘"}]
})

for msg in response['messages']:
    msg.pretty_print()
```

## 솔루션 4: 멀티 도구 Agent

```python
# 시스템 프롬프트 작성
multi_system_prompt = """당신은 다양한 기능을 제공하는 AI Assistant입니다.

사용 가능한 기능:
1. 날씨 조회: 한국 주요 도시의 현재 날씨 정보
2. 문서 검색: 테슬라와 리비안 전기차 회사에 대한 정보
3. 계산: 수학 계산 수행

사용자의 요청을 분석하여 적절한 도구를 선택하고, 필요 시 여러 도구를 조합하여 사용하세요.
"""

multi_prompt = ChatPromptTemplate.from_messages([
    ("system", multi_system_prompt),
    ("placeholder", "{messages}"),
])

# Agent 생성
multi_agent = create_react_agent(
    model=llm,
    tools=[get_weather, search_db, calculate],
    prompt=multi_prompt
)

# 복합 질문 테스트
response = multi_agent.invoke({
    "messages": [{
        "role": "user",
        "content": "서울 날씨를 알려주고, 테슬라와 리비안 중 어느 회사가 먼저 설립되었는지 알려줘"
    }]
})

# 최종 답변 출력
print(response['messages'][-1].content)
```

**Agent의 동작:**
1. 질문 분석: 두 가지 요청 인식 (날씨 + 회사 설립 정보)
2. 첫 번째 도구 호출: `get_weather("서울")`
3. 두 번째 도구 호출: `search_db("테슬라 리비안 설립")`
4. 검색 결과에서 설립 연도 비교
5. 통합된 답변 생성

## 솔루션 5: 견고한 검색 도구

```python
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@tool
def safe_search_db(query: str):
    """안전한 데이터베이스 검색을 수행합니다. 에러 처리가 포함되어 있습니다."""

    # 입력 검증
    if not query or not query.strip():
        logger.warning("빈 검색어가 입력되었습니다")
        return "검색어를 입력해주세요."

    # 검색어 길이 제한
    if len(query) > 200:
        logger.warning(f"검색어가 너무 깁니다: {len(query)} 글자")
        return "검색어는 200자 이하로 입력해주세요."

    try:
        # ChromaDB 연결 확인
        if not chroma_db._collection:
            logger.error("ChromaDB 컬렉션에 접근할 수 없습니다")
            return "데이터베이스 연결에 실패했습니다. 잠시 후 다시 시도해주세요."

        # 검색 수행
        logger.info(f"검색 실행: {query}")
        results = chroma_k_retriever.invoke(query)

        # 결과 확인
        if not results:
            logger.info("검색 결과가 없습니다")
            return "검색 결과를 찾을 수 없습니다. 다른 검색어를 시도해보세요."

        logger.info(f"검색 성공: {len(results)}개 문서 발견")
        return results

    except Exception as e:
        # 예상치 못한 에러 처리
        logger.error(f"검색 중 에러 발생: {str(e)}")
        return f"검색 중 오류가 발생했습니다: {type(e).__name__}"

# 테스트
print("테스트 1: 정상 검색")
result1 = safe_search_db.invoke("테슬라")
print(f"결과: {len(result1) if isinstance(result1, list) else result1}\n")

print("테스트 2: 빈 검색어")
result2 = safe_search_db.invoke("")
print(f"결과: {result2}\n")

print("테스트 3: 너무 긴 검색어")
long_query = "테슬라 " * 100
result3 = safe_search_db.invoke(long_query)
print(f"결과: {result3}")
```

---

# 🚀 실무 활용 예시

## 예시 1: RAG 기반 고객 지원 Agent

### 시나리오
전기차 판매 회사의 고객 지원 챗봇을 구현합니다. 제품 정보 데이터베이스를 검색하여 고객 질문에 정확하게 답변합니다.

```python
# 고객 지원 Agent 시스템 프롬프트
customer_support_prompt = """당신은 전기차 전문 고객 지원 AI입니다.

역할:
- 고객의 질문을 친절하고 정확하게 답변
- 제품 정보는 반드시 데이터베이스 검색 결과를 기반으로 제공
- 검색 결과가 없는 경우 솔직하게 안내하고 대체 방안 제시

응답 스타일:
- 친절하고 전문적인 톤
- 기술 용어는 쉽게 설명
- 구체적인 숫자와 사실 정보 포함
"""

# Agent 생성
support_agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o-mini", temperature=0.3),
    tools=[search_db],
    prompt=ChatPromptTemplate.from_messages([
        ("system", customer_support_prompt),
        ("placeholder", "{messages}"),
    ])
)

# 실제 고객 질문 예시
customer_questions = [
    "테슬라 모델 3의 주행거리는 얼마나 되나요?",
    "리비안 전기차는 충전 시간이 얼마나 걸리나요?",
    "두 회사의 배터리 기술 차이점을 알려주세요"
]

for question in customer_questions:
    print(f"\n고객 질문: {question}")
    response = support_agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })
    print(f"AI 답변: {response['messages'][-1].content}")
```

## 예시 2: 멀티 소스 정보 통합 Agent

### 시나리오
여러 데이터 소스(데이터베이스, 계산, 외부 API)를 조합하여 복잡한 질문에 답변합니다.

```python
# 추가 도구: 간단한 가격 계산
@tool
def calculate_total_cost(base_price: float, tax_rate: float = 0.1):
    """차량 가격에 세금을 포함한 총 비용을 계산합니다."""
    total = base_price * (1 + tax_rate)
    return f"기본 가격: {base_price:,.0f}원\n세금({tax_rate*100}%): {base_price*tax_rate:,.0f}원\n총 비용: {total:,.0f}원"

# 통합 Agent
integrated_agent = create_react_agent(
    model=llm,
    tools=[search_db, calculate_total_cost, calculate],
    prompt=ChatPromptTemplate.from_messages([
        ("system", "다양한 도구를 활용하여 사용자의 복잡한 질문에 답변하세요."),
        ("placeholder", "{messages}"),
    ])
)

# 복잡한 질문
complex_question = """
테슬라 Model Y의 가격 정보를 찾아서,
기본 가격이 5천만원이라면 세금 10%를 포함한 총 비용은 얼마인지 계산해줘.
그리고 월 100만원씩 납부한다면 몇 개월이 걸리는지도 알려줘.
"""

response = integrated_agent.invoke({
    "messages": [{"role": "user", "content": complex_question}]
})

print("최종 답변:")
print(response['messages'][-1].content)
```

## 예시 3: 대화 기록을 유지하는 상태 저장 Agent

### 시나리오
이전 대화 내용을 기억하고 문맥을 이해하는 Agent를 구현합니다.

```python
# 대화 히스토리 관리
conversation_history = []

def chat_with_agent(user_message):
    """대화 기록을 유지하며 Agent와 대화합니다."""

    # 사용자 메시지 추가
    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    # Agent 실행
    response = doc_agent.invoke({
        "messages": conversation_history
    })

    # Agent 응답을 히스토리에 추가
    assistant_message = response['messages'][-1].content
    conversation_history.append({
        "role": "assistant",
        "content": assistant_message
    })

    return assistant_message

# 연속된 대화 테스트
print("대화 1:")
answer1 = chat_with_agent("테슬라의 주요 제품은 뭐야?")
print(f"AI: {answer1}\n")

print("대화 2 (이전 문맥 참조):")
answer2 = chat_with_agent("그 회사는 언제 설립되었어?")  # "테슬라"를 명시하지 않아도 이해
print(f"AI: {answer2}\n")

print("대화 3 (비교 질문):")
answer3 = chat_with_agent("리비안과 비교하면 어때?")
print(f"AI: {answer3}")
```

## 예시 4: 에러 복구 및 Fallback 전략

### 시나리오
도구 실행 실패 시 대체 방안을 제시하는 견고한 Agent를 구현합니다.

```python
# 여러 검색 전략을 가진 도구
@tool
def advanced_search(query: str, strategy: Literal["exact", "fuzzy", "broad"] = "exact"):
    """다양한 검색 전략으로 문서를 검색합니다.

    Args:
        query: 검색어
        strategy:
            - exact: 정확한 매칭 (k=2)
            - fuzzy: 유사도 기반 (k=5)
            - broad: 광범위 검색 (k=10)
    """
    k_values = {"exact": 2, "fuzzy": 5, "broad": 10}

    retriever = chroma_db.as_retriever(
        search_kwargs={"k": k_values[strategy]}
    )

    results = retriever.invoke(query)

    if not results:
        return "검색 결과가 없습니다. 다른 검색어나 전략을 시도해보세요."

    return results

# Fallback이 있는 Agent
fallback_prompt = """당신은 지능적인 검색 AI입니다.

검색 전략:
1. 먼저 exact 전략으로 정확한 정보 검색
2. 결과가 없으면 fuzzy 전략으로 유사 문서 검색
3. 여전히 없으면 broad 전략으로 광범위 검색
4. 모든 전략이 실패하면 사용자에게 검색어 수정 제안

항상 검색 결과의 품질을 평가하고 최선의 답변을 제공하세요.
"""

fallback_agent = create_react_agent(
    model=llm,
    tools=[advanced_search],
    prompt=ChatPromptTemplate.from_messages([
        ("system", fallback_prompt),
        ("placeholder", "{messages}"),
    ])
)

# 테스트: 애매한 검색어
response = fallback_agent.invoke({
    "messages": [{"role": "user", "content": "전기 자동차의 미래"}]
})

print(response['messages'][-1].content)
```

---

# 📖 참고 자료

## 공식 문서
- [LangChain Tool Calling 개념](https://python.langchain.com/docs/concepts/tool_calling/)
- [LangChain Agents 가이드](https://python.langchain.com/docs/tutorials/agents/)
- [LangGraph ReAct Agent](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
- [OpenAI Function Calling API](https://platform.openai.com/docs/guides/function-calling)

## 추가 학습 자료
- [ReAct: Synergizing Reasoning and Acting in Language Models (논문)](https://arxiv.org/abs/2210.03629)
- [Tool Use and Agents - LangChain Blog](https://blog.langchain.dev/tag/agents/)
- [Building Production-Ready RAG Applications](https://www.pinecone.io/learn/retrieval-augmented-generation/)

## 관련 기술
- **Vector Databases**: Chroma, Pinecone, Weaviate, Milvus
- **LLM Frameworks**: LangChain, LlamaIndex, Haystack
- **Agent Frameworks**: AutoGPT, BabyAGI, LangGraph

## 실무 적용 사례
- 고객 지원 챗봇 (RAG + Agent)
- 문서 분석 자동화 (Multi-tool Agent)
- 데이터 분석 Assistant (계산 + 검색 + 시각화)
- 코드 생성 및 디버깅 Agent

---

**학습을 마치며**

Tool Calling과 Agent는 LLM의 능력을 실제 시스템과 연결하는 핵심 기술입니다. 이 가이드를 통해 기본 개념부터 실무 활용까지 체계적으로 학습했습니다.

**다음 단계 학습 권장사항:**
1. 더 복잡한 도구 체인 구현 (Multi-step reasoning)
2. 메모리 기능이 있는 상태 저장 Agent
3. 여러 Agent를 협업시키는 Multi-Agent 시스템
4. 프로덕션 환경을 위한 에러 처리 및 모니터링

계속해서 실습하고 자신만의 Agent를 만들어보세요! 🚀
