# PRJ03_W1_003_LangChain_Custom_Tool_Part1

## 📚 학습 목표

이 학습 가이드를 완료하면 다음을 수행할 수 있습니다:

1. **@tool 데코레이터 활용**: 함수를 LangChain 도구로 변환하는 기본 방법 이해
2. **StructuredTool 사용**: 기존 함수를 재활용하여 도구 생성
3. **Runnable to Tool**: 복잡한 체인을 도구로 변환하여 재사용
4. **비동기 도구**: 동기/비동기 도구 구현 및 성능 최적화
5. **도구 커스터마이징**: 입출력 스키마, 이름, 설명 등 세부 설정

## 🔑 핵심 개념

### 1. Custom Tool이란?

**사용자 정의 도구 (Custom Tool)**는 개발자가 직접 설계하고 구현하는 맞춤형 함수나 도구입니다:

- LLM이 호출할 수 있는 **고유한 기능** 정의
- 특정 작업에 최적화된 도구 생성 가능
- 입력값, 출력값, 기능을 자유롭게 정의
- Built-in Tool과 달리 프로젝트 요구사항에 맞춤 구현

### 2. Custom Tool 생성 방법

LangChain에서 제공하는 3가지 주요 방법:

```python
# 1. @tool 데코레이터 (가장 간단)
@tool
def my_tool(query: str) -> str:
    """도구 설명"""
    return process(query)

# 2. StructuredTool (기존 함수 재활용)
tool = StructuredTool.from_function(
    func=existing_function,
    name="tool_name",
    description="설명"
)

# 3. Runnable as Tool (체인을 도구로 변환)
chain_tool = my_chain.as_tool(
    name="chain_tool",
    description="설명"
)
```

### 3. 도구의 주요 속성

모든 LangChain 도구는 다음 속성을 가집니다:

- `name`: 도구의 고유 이름
- `description`: 도구의 기능 설명 (LLM이 선택 시 참조)
- `args`: 입력 파라미터 스키마
- `output_schema`: 출력 형식 스키마
- `return_direct`: 결과를 직접 반환할지 여부

## 🛠 환경 설정

### 필수 라이브러리 설치

```bash
pip install langchain langchain-openai langchain-community
pip install langchain-chroma chromadb
pip install sentence-transformers
pip install python-dotenv
```

### 환경 변수 설정

`.env` 파일 생성:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 기본 임포트

```python
import os
from dotenv import load_dotenv
from typing import Optional, Literal
from pprint import pprint

from langchain_core.tools import tool, StructuredTool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from pydantic import BaseModel, Field

# 환경 변수 로드
load_dotenv()
```

### ChromaDB 벡터 저장소 로드

```python
# 임베딩 모델 초기화
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ChromaDB 로드 (이전 실습에서 생성한 DB 사용)
chroma_db = Chroma(
    collection_name="db_korean_cosine_metadata",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)
```

## 💻 단계별 구현

### Step 1: @tool 데코레이터로 기본 도구 만들기

#### 1.1 간단한 검색 도구 생성

```python
from langchain_core.tools import tool
from typing import Optional

@tool
def search_database(query: str, k: Optional[int] = 4) -> str:
    """
    데이터베이스에서 주어진 쿼리로 검색을 수행합니다.

    Args:
        query: 검색할 텍스트 쿼리
        k: 반환할 결과의 개수 (기본값: 4)
    """
    retriever = chroma_db.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)

# 도구 속성 확인
print("도구 이름:", search_database.name)
print("도구 설명:", search_database.description)
print("도구 인자:", search_database.args)
print("출력 스키마:", search_database.output_schema.model_json_schema())
```

**출력 예시:**
```
도구 이름: search_database
도구 설명: 데이터베이스에서 주어진 쿼리로 검색을 수행합니다.

Args:
    query: 검색할 텍스트 쿼리
    k: 반환할 결과의 개수 (기본값: 4)
도구 인자: {'query': {'title': 'Query', 'type': 'string'}, 'k': {...}}
출력 스키마: {'title': 'search_database_output'}
```

#### 1.2 도구 실행

```python
# 직접 호출
docs = search_database.invoke("리비안은 언제 설립되었나요?")
pprint(docs)

# LLM과 바인딩
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)
llm_with_tools = llm.bind_tools([search_database])

# 도구 사용
result = llm_with_tools.invoke("리비안은 언제 설립되었나요?")
pprint(result.tool_calls)
```

**출력 예시:**
```python
[{'args': {'k': 1, 'query': '리비안 설립 연도'},
  'id': 'call_xxx',
  'name': 'search_database',
  'type': 'tool_call'}]
```

#### 1.3 도구 이름 및 스키마 커스터마이징

```python
from pydantic import BaseModel, Field

# 입력 스키마 정의
class ChromaDBInput(BaseModel):
    """ ChromaDB 검색 도구 입력 스키마 """
    query: str = Field(description="검색할 쿼리")
    k: int = Field(4, description="반환할 문서의 개수")

# 커스텀 도구 생성
@tool("ChromaDB-Search", args_schema=ChromaDBInput)
def search_database(query: str, k: int = 4) -> str:
    """
    데이터베이스에서 주어진 쿼리로 검색을 수행합니다.

    Args:
        query: 검색할 텍스트 쿼리
        k: 반환할 결과의 개수 (기본값: 4)
    """
    retriever = chroma_db.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)

# 속성 확인
print("커스텀 도구 이름:", search_database.name)  # ChromaDB-Search
print("커스텀 인자:", search_database.args)
```

#### 1.4 비동기 도구 만들기

```python
@tool
async def search_database(query: str, k: int = 4) -> str:
    """
    데이터베이스에서 주어진 쿼리로 검색을 수행합니다. (비동기 버전)

    Args:
        query: 검색할 텍스트 쿼리
        k: 반환할 결과의 개수 (기본값: 4)
    """
    retriever = chroma_db.as_retriever(search_kwargs={"k": k})
    return await retriever.ainvoke(query)

# 비동기 실행
docs = await search_database.ainvoke("리비안은 언제 설립되었나요?")
pprint(docs)
```

**핵심 포인트:**
- `async def`로 함수 정의
- `await retriever.ainvoke()` 사용
- `await tool.ainvoke()` 로 호출

### Step 2: StructuredTool로 도구 생성

#### 2.1 입출력 스키마 정의

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, Literal
from langchain_core.prompts import ChatPromptTemplate

# 입력 스키마
class TextAnalysisInput(BaseModel):
    text: str = Field(description="분석할 텍스트")
    include_sentiment: bool = Field(
        description="감성 분석 포함 여부",
        default=False
    )

# 출력 스키마
class SentimentOutput(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="감성 분석 결과")
```

#### 2.2 도구 작업을 함수로 정의

```python
# 동기 함수
def analyze_text(text: str, include_sentiment: bool = False) -> dict:
    """텍스트를 분석하여 단어 수, 문자 수 등의 정보를 반환합니다."""
    result = {
        "word_count": len(text.split()),
        "char_count": len(text),
        "sentence_count": len(text.split('.')),
    }

    if include_sentiment:
        # 감성 분석 수행
        prompt = ChatPromptTemplate.from_messages([
            ("system", "입력된 문장에 대해서 감성 분석을 수행합니다."),
            ("user", "{input}"),
        ])
        llm = ChatOpenAI(model="gpt-4.1-mini")
        llm_with_structure = llm.with_structured_output(SentimentOutput)
        sentiment_chain = prompt | llm_with_structure
        sentiment = sentiment_chain.invoke({"input": text})
        result["sentiment"] = sentiment.sentiment

    return result

# 비동기 함수
async def analyze_text_async(text: str, include_sentiment: bool = False) -> dict:
    """텍스트 분석의 비동기 버전입니다."""
    return analyze_text(text, include_sentiment)
```

#### 2.3 StructuredTool 생성

```python
# 도구 생성
text_analyzer = StructuredTool.from_function(
    func=analyze_text,               # 동기 함수
    name="TextAnalyzer",              # 도구 이름
    description="텍스트의 기본 통계와 선택적으로 감성 분석을 수행합니다.",
    args_schema=TextAnalysisInput,    # 입력 스키마
    coroutine=analyze_text_async,     # 비동기 함수
    return_direct=True                # 결과를 직접 반환
)

# 도구 속성 확인
print(text_analyzer.name)
print(text_analyzer.description)
print(text_analyzer.args)
```

#### 2.4 도구 실행

```python
text = "안녕하세요. 오늘은 날씨가 좋네요. 산책하기 좋은 날입니다."

# 동기 호출
result1 = text_analyzer.invoke({
    "text": text,
    "include_sentiment": True
})
print("텍스트 분석 결과:", result1)
# 출력: {'word_count': 7, 'char_count': 33, 'sentence_count': 4, 'sentiment': 'positive'}

# 비동기 호출
result2 = await text_analyzer.ainvoke({
    "text": text,
    "include_sentiment": False
})
print("비동기 텍스트 분석 결과:", result2)
# 출력: {'word_count': 7, 'char_count': 33, 'sentence_count': 4}
```

#### 2.5 StructuredTool vs @tool 데코레이터

**StructuredTool이 더 적합한 경우:**

**1. 기존 함수의 재사용**
```python
# 이미 존재하는 함수를 도구로 변환
def existing_function(x: int) -> str:
    return str(x)

# @tool은 함수 수정 필요
@tool
def modified_function(x: int) -> str:
    """ 숫자 변환 도구 """
    return str(x)

# StructuredTool은 기존 함수 그대로 사용
tool = StructuredTool.from_function(
    func=existing_function,
    name="convert_number",
    description="숫자를 문자열로 변환합니다.",
)
```

**2. 동일한 함수에 대해 다른 설정의 도구 생성**
```python
def multiply(a: int, b: int) -> int:
    return a * b

# 같은 함수로 다른 설정의 도구들 생성
basic_calculator = StructuredTool.from_function(
    func=multiply,
    name="basic_multiply",
    description="기본 곱셈 계산기",
)

advanced_calculator = StructuredTool.from_function(
    func=multiply,
    name="output_multiply",
    description="결과 출력용",
    return_direct=True   # 결과를 직접 반환
)
```

**3. 동기/비동기 함수 동시 지원**
```python
def sync_func(x: int) -> int:
    return x * 2

async def async_func(x: int) -> int:
    return sync_func(x)

# 동기/비동기 함수를 하나의 도구로 결합
tool = StructuredTool.from_function(
    func=sync_func,         # 동기 함수
    coroutine=async_func,   # 비동기 함수
    name="multiply_by_2",
    description="입력된 숫자에 2를 곱합니다."
)

# 사용
result1 = tool.invoke({"x": 5})        # 동기 호출
result2 = await tool.ainvoke({"x": 5}) # 비동기 호출
```

#### 2.6 return_direct 활용 예시

```python
from langchain.agents import create_agent

# return_direct=False: Agent가 결과를 받아 추가 처리
basic_agent = create_agent(
    model=llm,
    tools=[basic_calculator],
    system_prompt="당신은 수학 계산을 도와주는 AI 어시스턴트입니다."
)

result = basic_agent.invoke({
    "messages": [{"role": "user", "content": "2와 3을 곱해줘"}]
})

# Agent가 도구 결과를 받아서 답변 생성
# 출력: "2와 3을 곱한 결과는 6입니다."

# return_direct=True: 도구 결과를 직접 반환
advanced_agent = create_agent(
    model=llm,
    tools=[advanced_calculator],
    system_prompt="당신은 수학 계산을 도와주는 AI 어시스턴트입니다."
)

result = advanced_agent.invoke({
    "messages": [{"role": "user", "content": "2와 3을 곱해줘"}]
})

# 도구 결과가 바로 반환됨
# 출력: "6"
```

### Step 3: Runnable을 도구로 변환

#### 3.1 이메일 작성 체인 생성

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 이메일 작성 체인
email_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 전문적인 이메일 작성 도우미입니다."),
    ("human", """
    다음 정보로 이메일을 작성해주세요:
    - 수신자: {recipient}
    - 제목: {subject}
    - 톤: {tone}
    - 추가 요청사항: {requirements}
    """)
])

email_chain = (
    email_prompt
    | ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)
    | StrOutputParser()
)
```

#### 3.2 체인을 도구로 변환

```python
# 이메일 작성 도구로 변환
email_tool = email_chain.as_tool(
    name="email_writer",
    description="전문적인 이메일 작성을 도와주는 도구입니다.",
)

# 도구 속성 변경
email_tool.return_direct = True

# 도구 속성 확인
print("도구 이름:", email_tool.name)
print("도구 설명:", email_tool.description)
print("도구 인자:", email_tool.args)
print("직접 반환:", email_tool.return_direct)
```

**출력 예시:**
```
도구 이름: email_writer
도구 설명: 전문적인 이메일 작성을 도와주는 도구입니다.
도구 인자: {'recipient': {...}, 'subject': {...}, 'tone': {...}, 'requirements': {...}}
직접 반환: True
```

#### 3.3 도구 실행

```python
# 도구 직접 호출
email_result = email_tool.invoke({
    "recipient": "team@example.com",
    "subject": "프로젝트 진행 현황 보고",
    "tone": "전문적",
    "requirements": "회의 일정 조율 요청 포함"
})

print(email_result)
```

**출력 예시:**
```
Subject: 프로젝트 진행 현황 보고 및 회의 일정 조율 요청

team@example.com 귀하,

안녕하세요.

현재 진행 중인 프로젝트의 현황을 아래와 같이 보고드립니다.

[프로젝트 진행 현황 요약]
- 주요 완료 사항:
- 진행 중인 작업:
- 예상 일정 및 향후 계획:

추가 논의가 필요한 사항이 있어 회의 일정을 조율하고자 합니다...
```

#### 3.4 LLM과 도구 바인딩

```python
# LLM과 도구 바인딩
llm_with_tools = llm.bind_tools([email_tool])

result = llm_with_tools.invoke(
    "팀에게 프로젝트 진행 현황을 보고하는 이메일을 작성해줘. "
    "(전문적 톤, 요구사항: 회의 일정 조율 요청 포함, 수신자: 'team@email.com')"
)

pprint(result.tool_calls)
```

**출력 예시:**
```python
[{'args': {'recipient': 'team@email.com',
           'requirements': '회의 일정 조율 요청 포함',
           'subject': '프로젝트 진행 현황 보고',
           'tone': '전문적'},
  'id': 'call_xxx',
  'name': 'email_writer',
  'type': 'tool_call'}]
```

#### 3.5 Agent와 통합

```python
from langchain.agents import create_agent

# 도구 실행 에이전트 생성
email_agent = create_agent(
    model=llm,
    tools=[email_tool],
    system_prompt="당신은 이메일 작성을 도와주는 AI 어시스턴트입니다."
)

# 도구 실행 에이전트 사용
result = email_agent.invoke({
    "messages": [{
        "role": "user",
        "content": "팀에게 프로젝트 진행 현황을 보고하는 이메일을 작성해줘. "
                  "(전문적 톤, 요구사항: 회의 일정 조율 요청 포함, 수신자: 'team@example.com')"
    }]
})

# 최종 메시지만 추출
final_message = result["messages"][-1]
print(final_message.content)
```

## 🎯 실습 문제

### 문제 1: 커스텀 검색 도구 만들기 ⭐⭐

@tool 데코레이터를 사용하여 도구 이름과 스키마를 직접 정의하는 데이터베이스 검색 도구를 생성하세요.

**요구사항:**
- 도구 이름: "Database-Search-Tool"
- 입력: query (str), k (int, 기본값 3)
- ChromaDB에서 검색 수행
- LLM과 바인딩하여 도구 호출 확인

```python
# TODO: 여기에 코드를 작성하세요.
```

### 문제 2: MMR 검색 비동기 도구 ⭐⭐⭐

MMR 검색 리트리버를 사용하여 비동기 방식으로 동작하는 도구를 생성하세요.

**요구사항:**
- 비동기 함수 (@tool + async def)
- MMR search_type 사용
- k=4, fetch_k=20 기본값
- 비동기 호출 (ainvoke) 테스트

```python
# TODO: 여기에 코드를 작성하세요.
```

### 문제 3: StructuredTool로 MMR 검색 도구 ⭐⭐⭐

StructuredTool을 이용하여 MMR 검색 리트리버를 사용하는 도구를 생성하세요.

**요구사항:**
- 입력 스키마 정의 (Pydantic)
- 동기/비동기 함수 모두 구현
- StructuredTool.from_function() 사용
- 도구 속성 확인 및 실행

```python
# TODO: 여기에 코드를 작성하세요.
```

### 문제 4: Reranking 체인을 도구로 변환 ⭐⭐⭐⭐

검색 결과에 CrossEncoderReranker를 적용하여 상위 결과를 반환하는 Runnable 체인을 도구로 변환하세요.

**요구사항:**
- k=10개 검색 → top_n=3개 선택
- 검색 결과를 포맷팅하여 출력
- Runnable 체인 구성
- as_tool()로 도구 변환

```python
# TODO: 여기에 코드를 작성하세요.
```

## ✅ 솔루션 예시

### 솔루션 1: 커스텀 검색 도구

```python
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# 1. 입력 스키마 정의
class DatabaseSearchInput(BaseModel):
    """ 데이터베이스 검색 도구 입력 스키마 """
    query: str = Field(description="검색할 텍스트 쿼리")
    k: int = Field(3, description="반환할 문서의 개수 (기본값: 3)")

# 2. 도구 생성
@tool("Database-Search-Tool", args_schema=DatabaseSearchInput)
def my_search_tool(query: str, k: int = 3) -> str:
    """
    ChromaDB 데이터베이스에서 주어진 쿼리로 문서를 검색합니다.

    Args:
        query: 검색할 텍스트 쿼리
        k: 반환할 결과의 개수 (기본값: 3)
    """
    retriever = chroma_db.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)

# 3. 도구 속성 확인
print("도구 이름:", my_search_tool.name)
print("도구 설명:", my_search_tool.description)
print("도구 인자:", my_search_tool.args)

# 4. 도구 실행
docs = my_search_tool.invoke({"query": "리비안은 언제 설립되었나요?", "k": 3})
print("\n도구 실행 결과:")
pprint(docs)

# 5. LLM과 바인딩
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
llm_with_tools = llm.bind_tools([my_search_tool])

result = llm_with_tools.invoke("리비안의 설립 연도를 알려주세요. (3개 문서 검색)")
print("\nLLM 도구 호출 결과:")
pprint(result.tool_calls)
```

### 솔루션 2: MMR 검색 비동기 도구

```python
from langchain_core.tools import tool

# MMR 검색 비동기 도구 생성
@tool
async def search_database_mmr(query: str, k: int = 4, fetch_k: int = 20) -> str:
    """
    MMR 검색을 사용하여 데이터베이스에서 다양성을 고려한 검색을 수행합니다.

    Args:
        query: 검색할 텍스트 쿼리
        k: 반환할 결과의 개수 (기본값: 4)
        fetch_k: MMR 알고리즘에 전달할 문서 개수 (기본값: 20)
    """
    retriever = chroma_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k}
    )
    return await retriever.ainvoke(query)

# 도구 속성 확인
print("도구 이름:", search_database_mmr.name)
print("도구 설명:", search_database_mmr.description)
print("도구 인자:", search_database_mmr.args)

# 비동기 실행
docs = await search_database_mmr.ainvoke({
    "query": "리비안은 언제 설립되었나요?",
    "k": 3,
    "fetch_k": 15
})
print("\nMMR 검색 결과:")
pprint(docs)
```

### 솔루션 3: StructuredTool로 MMR 검색 도구

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

# 1. 입력 스키마 정의
class MMRSearchInput(BaseModel):
    """MMR 검색 도구 입력 스키마"""
    query: str = Field(description="검색할 쿼리")
    k: int = Field(4, description="반환할 문서의 개수")
    fetch_k: int = Field(20, description="MMR 알고리즘에 전달할 문서 개수")

# 2. MMR 검색 수행 함수 (동기)
def mmr_search(query: str, k: int = 4, fetch_k: int = 20) -> list:
    """MMR 검색을 수행하여 다양성을 고려한 문서를 반환합니다."""
    retriever = chroma_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k}
    )
    return retriever.invoke(query)

# 3. MMR 검색 수행 함수 (비동기)
async def mmr_search_async(query: str, k: int = 4, fetch_k: int = 20) -> list:
    """MMR 검색의 비동기 버전입니다."""
    retriever = chroma_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k}
    )
    return await retriever.ainvoke(query)

# 4. StructuredTool로 도구 생성
mmr_search_tool = StructuredTool.from_function(
    func=mmr_search,
    name="MMR_Search",
    description="MMR 알고리즘을 사용하여 다양성을 고려한 문서 검색을 수행합니다.",
    args_schema=MMRSearchInput,
    coroutine=mmr_search_async,
    return_direct=False
)

# 5. 도구 속성 확인
print("도구 이름:", mmr_search_tool.name)
print("도구 설명:", mmr_search_tool.description)
print("도구 인자:", mmr_search_tool.args)

# 6. 동기 실행
result_sync = mmr_search_tool.invoke({
    "query": "리비안의 전기 트럭에 대해 알려주세요",
    "k": 3,
    "fetch_k": 15
})
print("\n동기 MMR 검색 결과:")
pprint(result_sync)

# 7. 비동기 실행
result_async = await mmr_search_tool.ainvoke({
    "query": "테슬라의 전기차 기술",
    "k": 3,
    "fetch_k": 15
})
print("\n비동기 MMR 검색 결과:")
pprint(result_async)
```

### 솔루션 4: Reranking 체인을 도구로 변환

```python
from langchain_core.runnables import RunnableLambda
from sentence_transformers import CrossEncoder

# 1. Cross-Encoder Reranker 구현
def search_and_rerank(query_input, k: int = 10, top_n: int = 3):
    """
    데이터베이스에서 k개 문서를 검색한 후 Cross-Encoder로 상위 top_n개를 선택합니다.
    """
    # 입력 처리
    query = query_input["query"] if isinstance(query_input, dict) else query_input

    # Cross-Encoder 모델 초기화
    model = CrossEncoder("BAAI/bge-reranker-base")

    # k개 문서 검색
    retriever = chroma_db.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)

    # Relevance score 계산
    pairs = [[query, doc.page_content] for doc in docs]
    scores = model.predict(pairs)

    # Score 기준 정렬 및 상위 top_n 선택
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in scored_docs[:top_n]]

    return top_docs

# 2. 검색 결과 포맷팅 함수
def format_docs(docs):
    """검색된 문서를 포맷팅하여 반환합니다."""
    formatted_result = f"검색 결과 ({len(docs)}개 문서):\n" + "=" * 100 + "\n\n"

    for idx, doc in enumerate(docs, 1):
        formatted_result += f"[문서 {idx}]\n"
        formatted_result += f"내용: {doc.page_content[:200]}...\n"

        if doc.metadata:
            formatted_result += f"메타데이터: {doc.metadata}\n"

        formatted_result += "-" * 100 + "\n\n"

    return formatted_result

# 3. Runnable 체인 구성
search_and_rerank_chain = (
    RunnableLambda(search_and_rerank)
    | RunnableLambda(format_docs)
)

# 4. 체인을 도구로 변환
search_rerank_tool = search_and_rerank_chain.as_tool(
    name="search_and_rerank",
    description="데이터베이스에서 10개 문서를 검색한 후 CrossEncoder를 사용하여 "
                "상위 3개를 선택하여 포맷팅된 결과를 반환합니다."
)

# 도구 속성 확인
print("도구 이름:", search_rerank_tool.name)
print("도구 설명:", search_rerank_tool.description)
print("도구 인자:", search_rerank_tool.args)

# 도구 실행
result = search_rerank_tool.invoke({"query": "리비안의 전기 트럭 기술에 대해 알려주세요"})
print("\n검색 및 Re-rank 결과:")
print(result)
```

## 🚀 실무 활용 예시

### 예시 1: 다중 도구 Agent 시스템

```python
from langchain.agents import create_agent

# 여러 도구를 조합한 Agent
multi_tool_agent = create_agent(
    model=llm,
    tools=[
        search_database,      # 기본 검색
        search_database_mmr,  # MMR 검색
        email_tool,           # 이메일 작성
        text_analyzer         # 텍스트 분석
    ],
    system_prompt="""당신은 다양한 도구를 활용하는 AI 어시스턴트입니다.

    사용 가능한 도구:
    - search_database: 일반 검색
    - search_database_mmr: 다양성 검색
    - email_writer: 이메일 작성
    - TextAnalyzer: 텍스트 분석

    사용자 요청에 가장 적합한 도구를 선택하여 사용하세요.
    """
)

# 복합 질의 실행
result = multi_tool_agent.invoke({
    "messages": [{
        "role": "user",
        "content": "리비안에 대해 검색하고, 결과를 요약해서 팀에게 이메일로 보내줘"
    }]
})
```

### 예시 2: 도구 체이닝

```python
# 검색 → 분석 → 이메일 작성 파이프라인
from langchain_core.runnables import RunnableSequence

def process_and_email(query: str) -> str:
    """검색 결과를 분석하고 이메일로 작성"""
    # 1. 검색
    docs = search_database.invoke({"query": query, "k": 5})

    # 2. 결과 텍스트 추출
    text = "\n\n".join([doc.page_content for doc in docs])

    # 3. 텍스트 분석
    analysis = text_analyzer.invoke({
        "text": text[:1000],  # 처음 1000자만
        "include_sentiment": True
    })

    # 4. 이메일 작성
    email = email_tool.invoke({
        "recipient": "team@company.com",
        "subject": f"{query} - 검색 결과 보고",
        "tone": "전문적",
        "requirements": f"검색 문서 {len(docs)}개, 감성: {analysis.get('sentiment', 'N/A')}"
    })

    return email

# 실행
result = process_and_email("리비안의 전기차 기술")
print(result)
```

### 예시 3: 조건부 도구 선택

```python
def smart_search(query: str, need_diversity: bool = False) -> list:
    """필요에 따라 일반 검색 또는 MMR 검색 선택"""
    if need_diversity:
        print("MMR 검색 수행 (다양성 고려)")
        return search_database_mmr.invoke({
            "query": query,
            "k": 5,
            "fetch_k": 20
        })
    else:
        print("일반 검색 수행")
        return search_database.invoke({
            "query": query,
            "k": 5
        })

# 사용
result1 = smart_search("전기차", need_diversity=False)
result2 = smart_search("전기차", need_diversity=True)
```

## 📖 참고 자료

### 공식 문서
- [LangChain Custom Tools Documentation](https://python.langchain.com/docs/modules/agents/tools/custom_tools/)
- [Tool Calling Guide](https://python.langchain.com/docs/concepts/tool_calling/)
- [Pydantic Models](https://docs.pydantic.dev/latest/)

### 추가 학습 자료
- StructuredTool vs @tool 데코레이터 선택 가이드
- 비동기 도구 성능 최적화
- Agent와 Tool 통합 베스트 프랙티스

### 관련 노트북
- `PRJ03_W1_001_ToolCalling_Agent_Intro.md` - Tool Calling 기초
- `PRJ03_W1_002_LangChain_BuiltIn_Tool.md` - Built-in Tools
- 다음: Custom Tool 고급 활용 (Part2)

---

**학습 완료 체크리스트:**
- [ ] @tool 데코레이터로 기본 도구 생성 이해
- [ ] 도구 스키마 커스터마이징 방법 숙지
- [ ] 비동기 도구 구현 및 활용
- [ ] StructuredTool 사용 시나리오 이해
- [ ] Runnable을 도구로 변환하는 방법 습득
- [ ] 실습 문제 4개 완료
- [ ] 실무 예시 코드 실행 및 이해
