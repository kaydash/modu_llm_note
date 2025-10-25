# PRJ03_W1_002_LangChain_BuiltIn_Tool

## 📚 학습 목표

이 학습 가이드를 완료하면 다음을 수행할 수 있습니다:

1. **LangChain Built-in Tool 이해**: LangChain에서 제공하는 기본 제공 도구의 종류와 활용법 학습
2. **SQLDatabaseToolkit 활용**: 데이터베이스 쿼리, 스키마 조회, SQL 검증 등 SQL 관련 도구 사용
3. **Tavily Search 통합**: 실시간 웹 검색 기능을 에이전트에 통합
4. **Wikipedia API 활용**: 백과사전 정보를 활용하는 도구 구현
5. **도구 조합**: 여러 Built-in Tool을 조합하여 강력한 에이전트 구축

## 🔑 핵심 개념

### 1. LangChain Built-in Tools

LangChain은 다양한 사전 구축된 도구를 제공합니다:

- **SQLDatabaseToolkit**: 데이터베이스 작업을 위한 도구 모음
- **Tavily Search**: 웹 검색 기능
- **Wikipedia**: 백과사전 정보 조회
- **ArXiv**: 학술 논문 검색
- **Python REPL**: 코드 실행 도구
- 그 외 수십 가지 도구들

### 2. SQLDatabaseToolkit

데이터베이스 작업을 위한 5가지 핵심 도구:

```python
# 1. QuerySQLDatabaseTool: SQL 쿼리 실행
# 2. InfoSQLDatabaseTool: 테이블 스키마 정보 조회
# 3. ListSQLDatabaseTool: 사용 가능한 테이블 목록 조회
# 4. QuerySQLCheckerTool: SQL 쿼리 검증
# 5. SQL Agent: 자연어를 SQL로 변환하여 실행
```

### 3. Tavily Search API

실시간 웹 검색을 위한 AI 친화적 검색 엔진:
- 최신 정보 검색
- 구조화된 검색 결과 제공
- LLM에 최적화된 응답 형식

### 4. Wikipedia API

Wikipedia 콘텐츠 접근:
- 문서 검색 및 요약
- 다국어 지원
- 신뢰할 수 있는 백과사전 정보 제공

## 🛠 환경 설정

### 필수 라이브러리 설치

```bash
pip install langchain langchain-openai langchain-community
pip install tavily-python wikipedia-api
pip install langchain-chroma chromadb
pip install python-dotenv
```

### 환경 변수 설정

`.env` 파일 생성:

```bash
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 기본 임포트

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import create_react_agent, AgentExecutor
from langchain_classic import hub

# 환경 변수 로드
load_dotenv()
```

## 💻 단계별 구현

### Step 1: SQLDatabaseToolkit 기본 사용

#### 1.1 데이터베이스 연결 및 Toolkit 생성

```python
from langchain_community.utilities import SQLDatabase

# SQLite 데이터베이스 연결 (예: ETF 데이터)
db = SQLDatabase.from_uri("sqlite:///etf_data.db")

# Toolkit 생성
from langchain_community.agent_toolkits import SQLDatabaseToolkit

toolkit = SQLDatabaseToolkit(
    db=db,
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0)
)

# 사용 가능한 도구 확인
tools = toolkit.get_tools()
for tool in tools:
    print(f"도구 이름: {tool.name}")
    print(f"설명: {tool.description}\n")
```

**출력 예시:**
```
도구 이름: sql_db_query
설명: Execute a SQL query against the database and get back the result.

도구 이름: sql_db_schema
설명: Get the schema and sample rows for the specified SQL tables.

도구 이름: sql_db_list_tables
설명: List the available tables in the database.

도구 이름: sql_db_query_checker
설명: Check if your query is correct before executing it.
```

#### 1.2 개별 도구 사용

```python
# 테이블 목록 조회
list_tables_tool = next(t for t in tools if t.name == "sql_db_list_tables")
tables = list_tables_tool.invoke("")
print(f"사용 가능한 테이블: {tables}")

# 스키마 정보 조회
schema_tool = next(t for t in tools if t.name == "sql_db_schema")
schema = schema_tool.invoke("etf_info")
print(f"ETF 테이블 스키마:\n{schema}")

# SQL 쿼리 실행
query_tool = next(t for t in tools if t.name == "sql_db_query")
result = query_tool.invoke("SELECT * FROM etf_info LIMIT 5")
print(f"쿼리 결과:\n{result}")
```

#### 1.3 SQL Agent 생성 및 실행

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_classic import hub

# ReAct 프롬프트 로드
prompt = hub.pull("hwchase17/react")

# LLM 설정
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Agent 생성
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# AgentExecutor 생성
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10
)

# 자연어 질의 실행
response = agent_executor.invoke({
    "input": "ETF 데이터베이스에서 상위 5개의 ETF 이름과 운용사를 알려주세요."
})

print(f"\n최종 답변: {response['output']}")
```

**실행 과정 예시:**
```
> Entering new AgentExecutor chain...
Action: sql_db_list_tables → Observation: etf_info
Action: sql_db_schema → Observation: CREATE TABLE etf_info (name TEXT, company TEXT, ...)
Action: sql_db_query → Observation: [('KODEX 200', 'Samsung'), ...]

Final Answer: 상위 5개 ETF는 다음과 같습니다:
1. KODEX 200 (Samsung)
2. TIGER 200 (Mirae Asset)
...
```

### Step 2: Tavily Search 통합

#### 2.1 Tavily Search 도구 설정

```python
from langchain_community.tools.tavily_search import TavilySearchResults

# Tavily Search 도구 생성
tavily_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",  # "basic" 또는 "advanced"
    include_answer=True,
    include_raw_content=False,
    include_images=False
)

# 검색 테스트
search_results = tavily_tool.invoke("LangChain 최신 업데이트")
print(search_results)
```

#### 2.2 Tavily를 포함한 Agent 생성

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_classic import hub

# 도구 리스트 (SQL + Tavily)
combined_tools = toolkit.get_tools() + [tavily_tool]

# Agent 생성
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm,
    tools=combined_tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=combined_tools,
    verbose=True,
    handle_parsing_errors=True
)

# 실시간 정보와 DB 정보를 결합한 질의
response = agent_executor.invoke({
    "input": "2024년 ETF 시장 동향과 우리 데이터베이스의 ETF 개수를 비교해주세요."
})

print(f"\n답변: {response['output']}")
```

#### 2.3 Tavily Search 파라미터 상세

```python
# 고급 설정 예시
tavily_advanced = TavilySearchResults(
    max_results=10,           # 최대 결과 수
    search_depth="advanced",  # 검색 깊이
    include_answer=True,      # AI 생성 답변 포함
    include_raw_content=True, # 원본 콘텐츠 포함
    include_images=True,      # 이미지 URL 포함
    topic="news"              # 주제: "general" 또는 "news"
)

# 뉴스 검색 예시
news_results = tavily_advanced.invoke("AI 에이전트 최신 뉴스")
for idx, result in enumerate(news_results, 1):
    print(f"\n{idx}. {result.get('title', 'N/A')}")
    print(f"   URL: {result.get('url', 'N/A')}")
    print(f"   내용: {result.get('content', 'N/A')[:200]}...")
```

#### 2.4 Tavily Search 실습 문제

**실습 1: 주제별 검색 결과 비교**

```python
"""
다양한 주제에 대해 Tavily Search로 검색하고 결과를 비교하세요.
"""

def compare_search_results(topics: list[str]) -> dict:
    """
    여러 주제에 대한 검색 결과를 비교합니다.

    Args:
        topics: 검색할 주제 리스트

    Returns:
        주제별 검색 결과 요약
    """
    # TODO: 구현하세요
    pass

# 테스트
topics = ["LangChain AI agents", "OpenAI GPT-4", "Vector databases"]
results = compare_search_results(topics)
```

**솔루션:**

```python
from langchain_community.tools.tavily_search import TavilySearchResults

def compare_search_results(topics: list[str]) -> dict:
    """여러 주제에 대한 검색 결과를 비교합니다."""

    tavily_tool = TavilySearchResults(
        max_results=3,
        search_depth="advanced",
        include_answer=True
    )

    results = {}

    for topic in topics:
        try:
            search_results = tavily_tool.invoke(topic)

            # 결과 요약
            results[topic] = {
                "result_count": len(search_results) if isinstance(search_results, list) else 1,
                "top_sources": [r.get('url', 'N/A') for r in search_results[:3]] if isinstance(search_results, list) else [],
                "summary": search_results[:300] + "..." if isinstance(search_results, str) and len(search_results) > 300 else search_results,
                "status": "success"
            }
        except Exception as e:
            results[topic] = {
                "status": "error",
                "error": str(e)
            }

    return results

# 테스트
topics = ["LangChain AI agents", "OpenAI GPT-4", "Vector databases"]
results = compare_search_results(topics)

for topic, data in results.items():
    print(f"\n{'='*60}")
    print(f"주제: {topic}")
    print(f"상태: {data['status']}")
    if data['status'] == 'success':
        print(f"결과 수: {data['result_count']}")
        print(f"주요 출처: {data['top_sources']}")
```

**실습 2: 뉴스 vs 일반 검색 비교**

```python
"""
Tavily의 'news' 모드와 'general' 모드를 비교하세요.
"""

def compare_search_modes(query: str) -> dict:
    """
    동일한 쿼리에 대해 뉴스 검색과 일반 검색을 비교합니다.

    Args:
        query: 검색 쿼리

    Returns:
        모드별 검색 결과 비교
    """
    # TODO: 구현하세요
    pass

# 테스트
result = compare_search_modes("AI industry trends 2024")
```

**솔루션:**

```python
from langchain_community.tools.tavily_search import TavilySearchResults

def compare_search_modes(query: str) -> dict:
    """동일한 쿼리에 대해 뉴스 검색과 일반 검색을 비교합니다."""

    # 뉴스 검색
    news_search = TavilySearchResults(
        max_results=5,
        topic="news",
        search_depth="advanced",
        include_answer=True
    )

    # 일반 검색
    general_search = TavilySearchResults(
        max_results=5,
        topic="general",
        search_depth="advanced",
        include_answer=True
    )

    try:
        news_results = news_search.invoke(query)
        general_results = general_search.invoke(query)

        comparison = {
            "query": query,
            "news_mode": {
                "result_count": len(news_results) if isinstance(news_results, list) else 1,
                "sample": str(news_results)[:400] + "...",
            },
            "general_mode": {
                "result_count": len(general_results) if isinstance(general_results, list) else 1,
                "sample": str(general_results)[:400] + "...",
            }
        }

        return comparison

    except Exception as e:
        return {"error": str(e)}

# 테스트
result = compare_search_modes("AI industry trends 2024")
print(f"쿼리: {result['query']}")
print(f"\n[뉴스 모드]")
print(f"결과 수: {result['news_mode']['result_count']}")
print(f"샘플: {result['news_mode']['sample']}")
print(f"\n[일반 모드]")
print(f"결과 수: {result['general_mode']['result_count']}")
print(f"샘플: {result['general_mode']['sample']}")
```

**실습 3: 검색 깊이 비교 (basic vs advanced)**

```python
"""
Tavily의 'basic'과 'advanced' 검색 깊이를 비교하세요.
"""

def compare_search_depth(query: str) -> dict:
    """
    동일한 쿼리에 대해 basic과 advanced 검색 깊이를 비교합니다.

    Args:
        query: 검색 쿼리

    Returns:
        검색 깊이별 결과 비교
    """
    # TODO: 구현하세요
    pass

# 테스트
result = compare_search_depth("RAG implementation best practices")
```

**솔루션:**

```python
import time
from langchain_community.tools.tavily_search import TavilySearchResults

def compare_search_depth(query: str) -> dict:
    """동일한 쿼리에 대해 basic과 advanced 검색 깊이를 비교합니다."""

    # Basic 검색
    basic_search = TavilySearchResults(
        max_results=5,
        search_depth="basic"
    )

    # Advanced 검색
    advanced_search = TavilySearchResults(
        max_results=5,
        search_depth="advanced"
    )

    try:
        # Basic 검색 측정
        start_time = time.time()
        basic_results = basic_search.invoke(query)
        basic_time = time.time() - start_time

        # Advanced 검색 측정
        start_time = time.time()
        advanced_results = advanced_search.invoke(query)
        advanced_time = time.time() - start_time

        comparison = {
            "query": query,
            "basic": {
                "execution_time": f"{basic_time:.2f}초",
                "result_count": len(basic_results) if isinstance(basic_results, list) else 1,
                "sample": str(basic_results)[:400] + "...",
            },
            "advanced": {
                "execution_time": f"{advanced_time:.2f}초",
                "result_count": len(advanced_results) if isinstance(advanced_results, list) else 1,
                "sample": str(advanced_results)[:400] + "...",
            }
        }

        return comparison

    except Exception as e:
        return {"error": str(e)}

# 테스트
result = compare_search_depth("RAG implementation best practices")
print(f"쿼리: {result['query']}")
print(f"\n[Basic 검색]")
print(f"실행 시간: {result['basic']['execution_time']}")
print(f"결과 수: {result['basic']['result_count']}")
print(f"\n[Advanced 검색]")
print(f"실행 시간: {result['advanced']['execution_time']}")
print(f"결과 수: {result['advanced']['result_count']}")
```

### Step 3: Wikipedia 도구 활용

#### 3.1 Wikipedia 도구 생성

```python
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Wikipedia API Wrapper 설정
wikipedia_wrapper = WikipediaAPIWrapper(
    top_k_results=2,      # 상위 결과 수
    doc_content_chars_max=4000,  # 최대 문자 수
    lang="ko"             # 언어 설정 (ko, en 등)
)

# Wikipedia 도구 생성
wikipedia_tool = WikipediaQueryRun(
    api_wrapper=wikipedia_wrapper,
    name="wikipedia",
    description="Wikipedia에서 정보를 검색합니다. 입력은 검색 쿼리여야 합니다."
)

# 검색 테스트
wiki_result = wikipedia_tool.invoke("인공지능")
print(wiki_result)
```

#### 3.2 Wikipedia + Agent 통합

```python
# 모든 도구 결합
all_tools = [
    *toolkit.get_tools(),
    tavily_tool,
    wikipedia_tool
]

# Agent 생성
from langchain_classic import hub

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm=llm, tools=all_tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=all_tools,
    verbose=True,
    max_iterations=15
)

# 복합 질의 실행
response = agent_executor.invoke({
    "input": """
    다음을 수행해주세요:
    1. Wikipedia에서 'ETF'에 대한 설명을 찾으세요
    2. 우리 데이터베이스에 있는 ETF 종류를 확인하세요
    3. 최신 ETF 트렌드를 웹에서 검색하세요
    4. 종합하여 리포트를 작성해주세요
    """
})

print(f"\n\n=== 최종 리포트 ===\n{response['output']}")
```

#### 3.3 다국어 Wikipedia 활용

```python
# 영어 Wikipedia
wikipedia_en = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(lang="en", top_k_results=1)
)

# 한국어 Wikipedia
wikipedia_ko = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(lang="ko", top_k_results=1)
)

# 비교 검색
topic = "Machine Learning"
en_result = wikipedia_en.invoke(topic)
ko_result = wikipedia_ko.invoke("머신러닝")

print(f"영어 결과:\n{en_result[:300]}...\n")
print(f"한국어 결과:\n{ko_result[:300]}...")
```

### Step 4: 통합 예제 - 금융 리서치 Agent

```python
from langchain.prompts import PromptTemplate

# 커스텀 프롬프트 생성
template = """당신은 금융 리서치 전문가입니다. 다음 도구를 활용하여 질문에 답변하세요:

사용 가능한 도구:
{tools}

도구 이름: {tool_names}

다음 형식을 사용하세요:

Question: 답변해야 할 질문
Thought: 무엇을 해야 할지 생각합니다
Action: 수행할 작업 [{tool_names} 중 하나]
Action Input: 작업에 대한 입력
Observation: 작업의 결과
... (이 Thought/Action/Action Input/Observation을 반복)
Thought: 이제 최종 답변을 알았습니다
Final Answer: 원래 질문에 대한 최종 답변

질문: {input}
{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template)

# 금융 도구 세트
financial_tools = [
    *toolkit.get_tools(),
    tavily_tool,
    wikipedia_tool
]

# Agent 생성
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = create_react_agent(llm=llm, tools=financial_tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=financial_tools,
    verbose=True,
    max_iterations=20,
    handle_parsing_errors=True
)

# 복잡한 금융 질의
query = """
S&P 500 ETF에 대해 다음을 조사해주세요:
1. Wikipedia에서 S&P 500의 정의와 역사
2. 최신 S&P 500 동향 (웹 검색)
3. 우리 DB에 S&P 500 관련 ETF가 있는지 확인
4. 종합 투자 리포트 작성
"""

result = agent_executor.invoke({"input": query})
print(f"\n\n{'='*60}\n최종 리포트\n{'='*60}\n{result['output']}")
```

## 🎯 실습 문제

### 문제 1: Wikipedia 검색 도구 만들기 ⭐⭐

여러 주제를 동시에 검색하는 Wikipedia 도구를 구현하세요.

**요구사항:**
- 입력: 검색 주제 리스트
- 출력: 각 주제에 대한 요약 정보
- 한국어와 영어 결과 모두 제공

```python
def multi_topic_wikipedia_search(topics: list[str]) -> dict:
    """
    여러 주제에 대한 Wikipedia 정보를 검색합니다.

    Args:
        topics: 검색할 주제 리스트

    Returns:
        주제별 검색 결과 딕셔너리
    """
    # TODO: 구현하세요
    pass

# 테스트
topics = ["Python", "LangChain", "OpenAI"]
results = multi_topic_wikipedia_search(topics)
```

### 문제 2: SQL + Tavily 통합 Agent ⭐⭐⭐

데이터베이스 정보와 실시간 웹 검색을 결합하는 Agent를 만드세요.

**요구사항:**
- SQLDatabaseToolkit 사용
- Tavily Search 통합
- 자연어 질의 처리
- 결과를 구조화된 형식으로 반환

```python
def create_hybrid_search_agent(db_uri: str, tavily_api_key: str):
    """
    DB 검색과 웹 검색을 결합한 Agent를 생성합니다.

    Args:
        db_uri: 데이터베이스 URI
        tavily_api_key: Tavily API 키

    Returns:
        AgentExecutor 객체
    """
    # TODO: 구현하세요
    pass

# 테스트
agent = create_hybrid_search_agent(
    db_uri="sqlite:///test.db",
    tavily_api_key=os.getenv("TAVILY_API_KEY")
)
```

### 문제 3: 도구 성능 모니터링 ⭐⭐⭐

각 도구의 호출 횟수와 실행 시간을 추적하는 시스템을 구현하세요.

**요구사항:**
- 도구 래퍼 클래스 작성
- 실행 시간 측정
- 호출 통계 저장
- 성능 리포트 생성

```python
from typing import Any
import time

class ToolMonitor:
    """도구 실행을 모니터링하는 래퍼 클래스"""

    def __init__(self, tool):
        # TODO: 구현하세요
        pass

    def invoke(self, input: Any) -> Any:
        # TODO: 실행 시간 측정 및 통계 저장
        pass

    def get_statistics(self) -> dict:
        # TODO: 통계 반환
        pass
```

### 문제 4: 멀티모달 검색 Agent ⭐⭐⭐⭐

SQL, Wikipedia, Tavily를 모두 활용하여 종합적인 리서치를 수행하는 Agent를 구현하세요.

**요구사항:**
- 3가지 도구 모두 활용
- 결과를 비교 분석
- 신뢰도 점수 계산
- Markdown 형식 리포트 생성

```python
class ResearchAgent:
    """멀티모달 리서치 Agent"""

    def __init__(self, db_uri: str, llm):
        # TODO: 도구 초기화
        pass

    def research(self, query: str) -> str:
        """
        종합 리서치를 수행하고 Markdown 리포트를 생성합니다.

        Args:
            query: 리서치 질문

        Returns:
            Markdown 형식의 리포트
        """
        # TODO: 구현하세요
        pass
```

### 문제 5: 도구 자동 선택 시스템 ⭐⭐⭐⭐⭐

질문 유형을 분석하여 최적의 도구를 자동으로 선택하는 시스템을 구현하세요.

**요구사항:**
- 질문 분류 로직
- 도구 우선순위 결정
- 동적 도구 조합
- 실패 시 대체 전략

```python
class SmartToolSelector:
    """질문에 따라 최적의 도구를 선택하는 시스템"""

    def __init__(self, all_tools: list):
        self.all_tools = all_tools
        # TODO: 초기화

    def classify_query(self, query: str) -> dict:
        """질문을 분류합니다"""
        # TODO: 구현하세요
        pass

    def select_tools(self, query: str) -> list:
        """질문에 최적화된 도구 세트를 반환합니다"""
        # TODO: 구현하세요
        pass

    def execute_with_fallback(self, query: str) -> str:
        """도구 실패 시 대체 전략을 사용합니다"""
        # TODO: 구현하세요
        pass
```

## ✅ 솔루션 예시

### 솔루션 1: Wikipedia 검색 도구

```python
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

def multi_topic_wikipedia_search(topics: list[str]) -> dict:
    """
    여러 주제에 대한 Wikipedia 정보를 검색합니다.
    """
    # 한국어 Wikipedia
    wiki_ko = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(lang="ko", top_k_results=1, doc_content_chars_max=1000)
    )

    # 영어 Wikipedia
    wiki_en = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(lang="en", top_k_results=1, doc_content_chars_max=1000)
    )

    results = {}

    for topic in topics:
        try:
            ko_result = wiki_ko.invoke(topic)
            en_result = wiki_en.invoke(topic)

            results[topic] = {
                "korean": ko_result[:500] + "..." if len(ko_result) > 500 else ko_result,
                "english": en_result[:500] + "..." if len(en_result) > 500 else en_result,
                "status": "success"
            }
        except Exception as e:
            results[topic] = {
                "korean": None,
                "english": None,
                "status": "error",
                "error_message": str(e)
            }

    return results

# 테스트
topics = ["Python", "LangChain", "OpenAI"]
results = multi_topic_wikipedia_search(topics)

for topic, data in results.items():
    print(f"\n{'='*60}")
    print(f"주제: {topic}")
    print(f"상태: {data['status']}")
    if data['status'] == 'success':
        print(f"\n[한국어]\n{data['korean']}")
        print(f"\n[English]\n{data['english']}")
    else:
        print(f"오류: {data['error_message']}")
```

### 솔루션 2: SQL + Tavily 통합 Agent

```python
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_react_agent, AgentExecutor
from langchain_classic import hub
from langchain_openai import ChatOpenAI

def create_hybrid_search_agent(db_uri: str, tavily_api_key: str):
    """
    DB 검색과 웹 검색을 결합한 Agent를 생성합니다.
    """
    # 데이터베이스 연결
    db = SQLDatabase.from_uri(db_uri)

    # LLM 설정
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # SQL Toolkit
    sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    sql_tools = sql_toolkit.get_tools()

    # Tavily Search
    tavily_tool = TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=True
    )

    # 도구 결합
    all_tools = sql_tools + [tavily_tool]

    # 프롬프트
    prompt = hub.pull("hwchase17/react")

    # Agent 생성
    agent = create_react_agent(llm=llm, tools=all_tools, prompt=prompt)

    # AgentExecutor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=all_tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=15
    )

    return agent_executor

# 테스트
agent = create_hybrid_search_agent(
    db_uri="sqlite:///etf_data.db",
    tavily_api_key=os.getenv("TAVILY_API_KEY")
)

# 하이브리드 질의
response = agent.invoke({
    "input": """
    다음을 조사하세요:
    1. 데이터베이스에서 ETF 총 개수 확인
    2. 웹에서 2024년 ETF 시장 규모 검색
    3. 비교 분석 리포트 작성
    """
})

print(f"\n최종 답변:\n{response['output']}")
```

### 솔루션 3: 도구 성능 모니터링

```python
import time
from typing import Any, Dict
from datetime import datetime

class ToolMonitor:
    """도구 실행을 모니터링하는 래퍼 클래스"""

    def __init__(self, tool):
        self.tool = tool
        self.call_count = 0
        self.total_time = 0.0
        self.call_history = []

    def invoke(self, input: Any) -> Any:
        """도구를 실행하고 성능을 측정합니다"""
        start_time = time.time()

        try:
            result = self.tool.invoke(input)
            status = "success"
            error = None
        except Exception as e:
            result = None
            status = "error"
            error = str(e)

        end_time = time.time()
        execution_time = end_time - start_time

        # 통계 업데이트
        self.call_count += 1
        self.total_time += execution_time

        # 히스토리 저장
        self.call_history.append({
            "timestamp": datetime.now().isoformat(),
            "input": str(input)[:100],
            "execution_time": execution_time,
            "status": status,
            "error": error
        })

        if status == "error":
            raise Exception(error)

        return result

    def get_statistics(self) -> Dict:
        """통계 정보를 반환합니다"""
        return {
            "tool_name": self.tool.name,
            "total_calls": self.call_count,
            "total_time": round(self.total_time, 3),
            "average_time": round(self.total_time / self.call_count, 3) if self.call_count > 0 else 0,
            "success_rate": sum(1 for h in self.call_history if h["status"] == "success") / len(self.call_history) if self.call_history else 0,
            "recent_calls": self.call_history[-5:]  # 최근 5개 호출
        }

    def print_report(self):
        """성능 리포트를 출력합니다"""
        stats = self.get_statistics()
        print(f"\n{'='*60}")
        print(f"도구 성능 리포트: {stats['tool_name']}")
        print(f"{'='*60}")
        print(f"총 호출 횟수: {stats['total_calls']}")
        print(f"총 실행 시간: {stats['total_time']}초")
        print(f"평균 실행 시간: {stats['average_time']}초")
        print(f"성공률: {stats['success_rate']*100:.1f}%")
        print(f"\n최근 호출 내역:")
        for idx, call in enumerate(stats['recent_calls'], 1):
            print(f"  {idx}. {call['timestamp']} - {call['status']} ({call['execution_time']:.3f}초)")

# 테스트
from langchain_community.tools.tavily_search import TavilySearchResults

tavily_tool = TavilySearchResults(max_results=3)
monitored_tavily = ToolMonitor(tavily_tool)

# 여러 번 호출
queries = ["LangChain", "OpenAI", "AI Agents"]
for query in queries:
    result = monitored_tavily.invoke(query)
    print(f"검색 완료: {query}")

# 리포트 출력
monitored_tavily.print_report()
```

### 솔루션 4: 멀티모달 검색 Agent

```python
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_classic import hub

class ResearchAgent:
    """멀티모달 리서치 Agent"""

    def __init__(self, db_uri: str, llm=None):
        # LLM 초기화
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # 데이터베이스 도구
        db = SQLDatabase.from_uri(db_uri)
        sql_toolkit = SQLDatabaseToolkit(db=db, llm=self.llm)
        self.sql_tools = sql_toolkit.get_tools()

        # Tavily 검색
        self.tavily_tool = TavilySearchResults(
            max_results=5,
            search_depth="advanced"
        )

        # Wikipedia
        self.wiki_tool = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(
                top_k_results=2,
                doc_content_chars_max=2000,
                lang="ko"
            )
        )

        # 모든 도구 결합
        self.all_tools = self.sql_tools + [self.tavily_tool, self.wiki_tool]

        # Agent 생성
        from langchain_classic import hub

        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(
            llm=self.llm,
            tools=self.all_tools,
            prompt=prompt
        )

        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.all_tools,
            verbose=True,
            max_iterations=20,
            handle_parsing_errors=True
        )

    def research(self, query: str) -> str:
        """
        종합 리서치를 수행하고 Markdown 리포트를 생성합니다.
        """
        # Agent 실행
        result = self.agent_executor.invoke({"input": query})

        # Markdown 리포트 생성
        report = f"""# 리서치 리포트

## 질문
{query}

## 조사 결과
{result['output']}

## 사용된 도구
- 데이터베이스 쿼리 (SQL)
- 웹 검색 (Tavily)
- 백과사전 (Wikipedia)

## 신뢰도 평가
- 데이터베이스: 높음 (내부 데이터)
- 웹 검색: 중간 (실시간 정보)
- Wikipedia: 높음 (검증된 정보)

---
생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return report

# 테스트
research_agent = ResearchAgent(db_uri="sqlite:///etf_data.db")

query = """
'ETF'에 대해 다음을 조사하세요:
1. Wikipedia에서 ETF의 정의
2. 최신 ETF 시장 동향 (웹 검색)
3. 데이터베이스의 ETF 통계
"""

report = research_agent.research(query)
print(report)
```

### 솔루션 5: 도구 자동 선택 시스템

```python
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

class SmartToolSelector:
    """질문에 따라 최적의 도구를 선택하는 시스템"""

    def __init__(self, all_tools: list):
        self.all_tools = all_tools
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # 도구 카테고리 매핑
        self.tool_categories = {
            "database": ["sql_db_query", "sql_db_schema", "sql_db_list_tables"],
            "web_search": ["tavily_search_results_json"],
            "encyclopedia": ["wikipedia"]
        }

    def classify_query(self, query: str) -> Dict:
        """질문을 분류합니다"""
        classification_prompt = ChatPromptTemplate.from_messages([
            ("system", """다음 질문을 분석하여 필요한 도구 카테고리를 JSON 형식으로 반환하세요.

카테고리:
- database: 내부 데이터베이스 조회가 필요한 경우
- web_search: 실시간 웹 검색이 필요한 경우
- encyclopedia: 일반적인 지식/정의가 필요한 경우

응답 형식:
{{"categories": ["category1", "category2"], "priority": "category1", "reasoning": "이유"}}
"""),
            ("user", "{query}")
        ])

        response = self.llm.invoke(
            classification_prompt.format_messages(query=query)
        )

        import json
        try:
            return json.loads(response.content)
        except:
            return {"categories": ["web_search"], "priority": "web_search", "reasoning": "기본값"}

    def select_tools(self, query: str) -> List:
        """질문에 최적화된 도구 세트를 반환합니다"""
        classification = self.classify_query(query)

        selected_tools = []
        for category in classification["categories"]:
            tool_names = self.tool_categories.get(category, [])
            for tool in self.all_tools:
                if tool.name in tool_names:
                    selected_tools.append(tool)

        # 중복 제거
        selected_tools = list({tool.name: tool for tool in selected_tools}.values())

        print(f"\n선택된 도구: {[t.name for t in selected_tools]}")
        print(f"이유: {classification['reasoning']}")

        return selected_tools

    def execute_with_fallback(self, query: str) -> str:
        """도구 실패 시 대체 전략을 사용합니다"""
        from langchain.agents import create_react_agent, AgentExecutor
        from langchain import hub

        # 1차 시도: 최적 도구 선택
        selected_tools = self.select_tools(query)

        if not selected_tools:
            return "적합한 도구를 찾을 수 없습니다."

        try:
            prompt = hub.pull("hwchase17/react")
            agent = create_react_agent(
                llm=self.llm,
                tools=selected_tools,
                prompt=prompt
            )

            agent_executor = AgentExecutor(
                agent=agent,
                tools=selected_tools,
                verbose=True,
                max_iterations=10,
                handle_parsing_errors=True
            )

            result = agent_executor.invoke({"input": query})
            return result["output"]

        except Exception as e:
            print(f"\n1차 시도 실패: {e}")
            print("대체 전략 실행: 모든 도구 사용")

            # 2차 시도: 모든 도구 사용
            try:
                agent = create_react_agent(
                    llm=self.llm,
                    tools=self.all_tools,
                    prompt=prompt
                )

                agent_executor = AgentExecutor(
                    agent=agent,
                    tools=self.all_tools,
                    verbose=True,
                    max_iterations=15
                )

                result = agent_executor.invoke({"input": query})
                return result["output"]

            except Exception as e2:
                return f"모든 시도 실패: {e2}"

# 테스트
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# 도구 준비
db = SQLDatabase.from_uri("sqlite:///test.db")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)

all_tools = [
    *sql_toolkit.get_tools(),
    TavilySearchResults(max_results=3),
    WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(lang="ko"))
]

# SmartToolSelector 생성
selector = SmartToolSelector(all_tools)

# 테스트 질의들
queries = [
    "데이터베이스에 있는 테이블을 알려주세요",
    "AI의 정의가 무엇인가요?",
    "최신 AI 뉴스를 검색해주세요"
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"질문: {query}")
    result = selector.execute_with_fallback(query)
    print(f"\n답변: {result}")
```

## 🚀 실무 활용 예시

### 예시 1: 금융 데이터 분석 시스템

```python
"""
실시간 금융 뉴스와 내부 포트폴리오 데이터를 결합한 분석 시스템
"""

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_classic import hub

class FinancialAnalysisAgent:
    def __init__(self, portfolio_db_uri: str):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # 포트폴리오 데이터베이스
        db = SQLDatabase.from_uri(portfolio_db_uri)
        toolkit = SQLDatabaseToolkit(db=db, llm=self.llm)

        # 뉴스 검색
        news_search = TavilySearchResults(
            max_results=5,
            topic="news",
            search_depth="advanced"
        )

        # 도구 결합
        tools = toolkit.get_tools() + [news_search]

        # Agent 생성
        from langchain_classic import hub

        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(self.llm, tools, prompt)
        self.executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=15
        )

    def analyze_portfolio_risk(self, stock_symbol: str) -> str:
        """특정 주식의 리스크를 분석합니다"""
        query = f"""
        {stock_symbol}에 대해:
        1. 최신 뉴스 검색
        2. 포트폴리오 내 보유 현황 확인
        3. 리스크 평가 리포트 작성
        """
        result = self.executor.invoke({"input": query})
        return result["output"]

# 사용 예시
agent = FinancialAnalysisAgent("sqlite:///portfolio.db")
report = agent.analyze_portfolio_risk("AAPL")
print(report)
```

### 예시 2: 고객 지원 챗봇

```python
"""
제품 데이터베이스와 실시간 웹 검색을 활용한 고객 지원 시스템
"""

class CustomerSupportBot:
    def __init__(self, product_db_uri: str):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

        # 제품 데이터베이스
        db = SQLDatabase.from_uri(product_db_uri)
        toolkit = SQLDatabaseToolkit(db=db, llm=self.llm)

        # 웹 검색 (최신 정보)
        web_search = TavilySearchResults(max_results=3)

        # Wikipedia (일반 지식)
        wiki = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(lang="ko", top_k_results=1)
        )

        tools = toolkit.get_tools() + [web_search, wiki]

        # 커스텀 프롬프트
        from langchain.prompts import PromptTemplate
        template = """당신은 친절한 고객 지원 상담원입니다.

사용 가능한 도구: {tools}
도구 이름: {tool_names}

질문: {input}
{agent_scratchpad}

항상 친절하고 명확하게 답변하세요."""

        prompt = PromptTemplate.from_template(template)
        agent = create_react_agent(self.llm, tools, prompt)

        self.executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            max_iterations=10
        )

    def handle_query(self, customer_query: str) -> str:
        """고객 질의를 처리합니다"""
        result = self.executor.invoke({"input": customer_query})
        return result["output"]

# 사용 예시
bot = CustomerSupportBot("sqlite:///products.db")
response = bot.handle_query("제품 A의 사양을 알려주세요")
print(response)
```

### 예시 3: 연구 보조 시스템

```python
"""
학술 연구를 위한 Wikipedia + 웹 검색 통합 시스템
"""

class ResearchAssistant:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Wikipedia (신뢰할 수 있는 정보)
        wiki_ko = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(lang="ko", top_k_results=2)
        )
        wiki_en = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(lang="en", top_k_results=2)
        )

        # 학술 검색
        academic_search = TavilySearchResults(
            max_results=10,
            search_depth="advanced",
            topic="general"
        )

        tools = [wiki_ko, wiki_en, academic_search]

        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(self.llm, tools, prompt)

        self.executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=20
        )

    def research_topic(self, topic: str, language: str = "both") -> dict:
        """주제에 대한 종합 연구를 수행합니다"""
        query = f"""
        '{topic}'에 대해:
        1. Wikipedia에서 정의와 배경 조사 ({language})
        2. 최신 연구 동향 검색
        3. 주요 발견사항 정리
        4. 참고문헌 리스트 작성
        """

        result = self.executor.invoke({"input": query})

        return {
            "topic": topic,
            "findings": result["output"],
            "language": language
        }

# 사용 예시
assistant = ResearchAssistant()
report = assistant.research_topic("Transformer Architecture", language="both")
print(report["findings"])
```

### 예시 4: 비즈니스 인텔리전스 대시보드

```python
"""
SQL 분석 + 시장 동향을 결합한 BI 시스템
"""

class BusinessIntelligenceAgent:
    def __init__(self, analytics_db_uri: str):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # 분석 데이터베이스
        db = SQLDatabase.from_uri(analytics_db_uri)
        toolkit = SQLDatabaseToolkit(db=db, llm=self.llm)

        # 시장 동향 검색
        market_search = TavilySearchResults(
            max_results=5,
            topic="news",
            search_depth="advanced"
        )

        tools = toolkit.get_tools() + [market_search]

        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(self.llm, tools, prompt)

        self.executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True
        )

    def generate_insights(self, metric: str, period: str = "monthly") -> str:
        """비즈니스 인사이트를 생성합니다"""
        query = f"""
        {metric}에 대해:
        1. 데이터베이스에서 {period} 트렌드 분석
        2. 시장 동향 검색
        3. 인사이트 및 권장사항 제공
        """

        result = self.executor.invoke({"input": query})
        return result["output"]

# 사용 예시
bi_agent = BusinessIntelligenceAgent("sqlite:///sales_analytics.db")
insights = bi_agent.generate_insights("매출", "quarterly")
print(insights)
```

## 📖 참고 자료

### 공식 문서
- [LangChain Tools Documentation](https://python.langchain.com/docs/modules/agents/tools/)
- [SQLDatabaseToolkit API](https://api.python.langchain.com/en/latest/agent_toolkits/langchain_community.agent_toolkits.sql.toolkit.SQLDatabaseToolkit.html)
- [Tavily Search API](https://docs.tavily.com/)
- [Wikipedia API](https://wikipedia-api.readthedocs.io/)

### 추가 학습 자료
- LangChain Agent 고급 패턴
- SQL Injection 방지 전략
- 웹 검색 결과 신뢰도 평가
- 멀티모달 도구 조합 베스트 프랙티스

### 관련 노트북
- `PRJ03_W1_001_ToolCalling_Agent_Intro.md` - Tool Calling 기초
- 다음: Custom Tool 개발 가이드

---

**학습 완료 체크리스트:**
- [ ] SQLDatabaseToolkit의 5가지 도구 이해
- [ ] Tavily Search API 설정 및 사용
- [ ] Wikipedia 도구 한국어/영어 활용
- [ ] 여러 도구를 조합한 Agent 생성
- [ ] 실습 문제 5개 완료
- [ ] 실무 예시 코드 실행 및 이해
