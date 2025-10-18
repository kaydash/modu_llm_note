# W3_003_ETF_Text2SQL_RAG - Text2SQL 기반 RAG 시스템 구현

## 학습 목표

이 가이드에서는 LangGraph와 ReAct 에이전트를 활용한 Text2SQL RAG 시스템 구축을 학습합니다:

- **Chain vs Agent 패턴**: 순차적 실행과 자율적 추론 방식의 차이 이해
- **LangGraph 상태 관리**: TypedDict 기반 상태 추적 및 워크플로우 구축
- **ReAct 에이전트**: 도구 사용과 반복적 추론을 통한 복잡한 쿼리 처리
- **테이블 선택 전략**: 대규모 데이터베이스에서 관련 테이블 자동 식별
- **오류 복구 메커니즘**: 쿼리 실패 시 자동 재시도 및 수정

### 선수 지식
- Text2SQL 기본 개념 (W3_002 가이드 참조)
- LangChain 체인 구성 경험
- SQLite 쿼리 작성 능력
- Python 타입 힌팅 및 TypedDict 이해

---

## 핵심 개념

### SQL QA 시스템이란?
구조화된 데이터베이스에 대해 자연어로 질문하고 답변을 받는 AI 시스템입니다.

**처리 과정**:
1. 🗣️ **자연어 질문** 수신
2. 🔄 **SQL 쿼리 변환** (Text2SQL)
3. ⚙️ **데이터베이스 실행** 및 결과 추출
4. 📝 **자연어 답변 생성** (RAG)

### Chain vs Agent 방식 비교

| 특징 | Chain 방식 | Agent 방식 |
|------|-----------|-----------|
| **실행 흐름** | 순차적, 예측 가능 | 자율적, 동적 결정 |
| **적용 사례** | 단순 쿼리, 단일 질문 | 복잡한 분석, 다단계 추론 |
| **오류 처리** | 수동 개입 필요 | 자동 복구 시도 |
| **도구 사용** | 고정된 순서 | 필요에 따라 선택 |
| **성능** | 빠름 (단일 실행) | 느림 (반복 실행) |
| **예측 가능성** | 높음 | 낮음 (추론 기반) |

**Chain 방식 예시**:
```
질문 → SQL 생성 → 쿼리 실행 → 답변 생성 → 완료
```

**Agent 방식 예시**:
```
질문 → [도구 선택] → 스키마 조회 → [도구 선택] → SQL 생성
     → [도구 선택] → 실행 → 오류 발생 → [도구 선택] → 수정
     → 재실행 → 성공 → 답변 생성
```

### LangGraph 상태 관리
상태 기반 워크플로우를 구축하는 LangChain의 그래프 프레임워크입니다.

**핵심 개념**:
- **State**: 노드 간 공유되는 데이터 구조
- **Node**: 상태를 변경하는 함수
- **Edge**: 노드 간 전환 로직

```python
from langgraph.graph import StateGraph, START

# 상태 정의
class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

# 그래프 구축
graph = StateGraph(State)
graph.add_node("node1", function1)
graph.add_node("node2", function2)
graph.add_edge(START, "node1")
graph.add_edge("node1", "node2")
```

### ReAct 에이전트
Reasoning + Acting을 결합한 에이전트 패턴입니다.

**동작 원리**:
1. **생각 (Reasoning)**: "무엇을 해야 할까?"
2. **행동 (Acting)**: 도구 실행
3. **관찰 (Observation)**: 결과 확인
4. **반복**: 목표 달성까지 1-3 반복

---

## 환경 설정

### 필수 라이브러리

```bash
# LangChain 핵심
pip install langchain langchain-openai langchain-google-genai
pip install langchain-community

# LangGraph (상태 관리)
pip install langgraph

# 데이터베이스
pip install sqlalchemy

# 환경 변수
pip install python-dotenv
```

### 환경 변수 설정

```python
from dotenv import load_dotenv
import os
import warnings

# 환경 변수 로드
load_dotenv()

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# LangSmith 추적 설정 (선택 사항)
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # 또는 "false"
os.environ["LANGCHAIN_PROJECT"] = "ETF_Text2SQL_RAG"
```

### 데이터베이스 연결

```python
from langchain_community.utilities import SQLDatabase

# SQLite 데이터베이스 연결
db = SQLDatabase.from_uri("sqlite:///etf_database.db")

# 데이터베이스 정보 확인
print(f"Dialect: {db.dialect}")
print(f"Tables: {db.get_usable_table_names()}")

# 샘플 데이터 확인
etfs = db.run("SELECT * FROM ETFs LIMIT 5;")
for etf in eval(etfs):
    print(etf)
```

---

## 단계별 구현: Chain 방식

### 1단계: State 상태 정의

LangGraph에서 노드 간 데이터를 공유하는 상태 구조를 정의합니다.

```python
from typing import TypedDict

class State(TypedDict):
    """SQL QA 시스템의 상태 정의"""
    question: str  # 사용자 입력 질문
    query: str     # 생성된 SQL 쿼리
    result: str    # 쿼리 실행 결과
    answer: str    # 최종 자연어 답변
```

**설계 포인트**:
- 각 노드는 State의 일부를 업데이트
- 이전 노드의 결과를 다음 노드가 참조
- 불변성 유지 (새 딕셔너리 반환)

---

### 2단계: 프롬프트 템플릿 구성

SQL 쿼리 생성을 위한 프롬프트를 작성합니다.

```python
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import Annotated

# SQL 쿼리 생성 프롬프트
query_prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
    Given an input question, create a syntactically correct {dialect} query to run to help find the answer.
    Unless the user specifies in his question a specific number of examples they wish to obtain,
    always limit your query to at most {top_k} results.

    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.
    Pay attention to use only the column names that you can see in the schema description.
    Be careful to not query for columns that do not exist.

    Also, pay attention to which column is in which table.
    Only use the following tables:
    {table_info}
    """),
    ("user", """
    Question:
    {input}
    """)
])

# 필요한 입력 필드 확인
print(query_prompt_template.input_variables)
# ['dialect', 'input', 'table_info', 'top_k']
```

**프롬프트 구성 요소**:
- `{dialect}`: 데이터베이스 방언 (sqlite, postgresql 등)
- `{top_k}`: 최대 결과 개수 제한
- `{table_info}`: 테이블 스키마 정보
- `{input}`: 사용자 질문

---

### 3단계: SQL 쿼리 생성 노드

```python
from langchain_openai import ChatOpenAI

# LLM 모델 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 출력 구조 정의
class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]

def write_query(state: State):
    """
    자연어 질문을 SQL 쿼리로 변환

    Parameters:
        state (State): 현재 상태 (question 포함)

    Returns:
        dict: 생성된 쿼리 {'query': 'SELECT ...'}
    """
    # 프롬프트에 필요한 값 전달
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )

    # 구조화된 출력으로 LLM 실행
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)

    return {"query": result["query"]}

# 테스트
test_state = {"question": "총보수가 0.1% 이하인 ETF는?"}
result = write_query(test_state)
print(f"생성된 쿼리: {result['query']}")
```

**핵심 기능**:
- `with_structured_output()`: JSON 스키마 기반 출력 강제
- 데이터베이스 스키마 자동 전달
- 쿼리 구문 검증 (Pydantic)

---

### 4단계: SQL 쿼리 실행 노드

```python
from langchain_community.tools import QuerySQLDatabaseTool

def execute_query(state: State):
    """
    생성된 SQL 쿼리를 데이터베이스에서 실행

    Parameters:
        state (State): 현재 상태 (query 포함)

    Returns:
        dict: 쿼리 실행 결과 {'result': '[(...), (...)]'}
    """
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    result = execute_query_tool.invoke(state["query"])

    return {"result": result}

# 테스트
test_state_with_query = {
    "question": "총보수가 0.1% 이하인 ETF는?",
    "query": "SELECT 종목코드, 종목명, 총보수 FROM ETFs WHERE 총보수 <= 0.1 LIMIT 5"
}
result = execute_query(test_state_with_query)
print(f"실행 결과: {result['result']}")
```

**오류 처리**:
- 구문 오류 발생 시 예외 발생
- Chain 방식에서는 수동 처리 필요
- Agent 방식에서는 자동 복구

---

### 5단계: RAG 답변 생성 노드

```python
def generate_answer(state: State):
    """
    SQL 결과를 바탕으로 자연어 답변 생성

    Parameters:
        state (State): 전체 상태 (question, query, result 포함)

    Returns:
        dict: 최종 답변 {'answer': '...'}
    """
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )

    response = llm.invoke(prompt)
    return {"answer": response.content}

# 테스트
full_state = {
    "question": "총보수가 0.1% 이하인 ETF는?",
    "query": "SELECT 종목코드, 종목명, 총보수 FROM ETFs WHERE 총보수 <= 0.1",
    "result": "[('069660', 'KOSEF 200', 0.05), ('102110', 'TIGER 200', 0.05)]"
}
result = generate_answer(full_state)
print(f"답변: {result['answer']}")
```

---

### 6단계: LangGraph 통합

```python
from langgraph.graph import START, StateGraph

# 그래프 빌더 초기화
graph_builder = StateGraph(State)

# 노드 추가
graph_builder.add_node("write_query", write_query)
graph_builder.add_node("execute_query", execute_query)
graph_builder.add_node("generate_answer", generate_answer)

# 엣지 연결 (순차 실행)
graph_builder.add_edge(START, "write_query")
graph_builder.add_edge("write_query", "execute_query")
graph_builder.add_edge("execute_query", "generate_answer")

# 그래프 컴파일
graph = graph_builder.compile()

print("✅ LangGraph 체인 생성 완료")
```

**그래프 시각화**:
```python
from IPython.display import Image, display

# 그래프 구조 이미지 생성
display(Image(graph.get_graph().draw_mermaid_png()))
```

**실행 예제**:
```python
# 스트리밍 실행
question = "총보수가 0.1% 이하인 ETF는 무엇인가요?"

for step in graph.stream(
    {"question": question},
    stream_mode="updates"  # 각 노드의 업데이트만 출력
):
    print(step)

# 최종 결과만 가져오기
result = graph.invoke({"question": question})
print(f"최종 답변: {result['answer']}")
```

---

## 단계별 구현: Agent 방식

### 1단계: SQLDatabaseToolkit 준비

ReAct 에이전트가 사용할 도구 모음을 준비합니다.

```python
from langchain_community.agent_toolkits import SQLDatabaseToolkit

# 도구킷 초기화
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# 사용 가능한 도구 목록
tools = toolkit.get_tools()

print(f"도구 개수: {len(tools)}")
for tool in tools:
    print(f"- {tool.name}: {tool.description}")
```

**기본 제공 도구**:
- `sql_db_query`: SQL 쿼리 실행
- `sql_db_schema`: 테이블 스키마 조회
- `sql_db_list_tables`: 테이블 목록 조회
- `sql_db_query_checker`: 쿼리 구문 검증

---

### 2단계: Agent 프롬프트 작성

```python
system_message = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer.

Unless the user specifies a specific number of examples they wish to obtain,
always limit your query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.

You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.

You MUST double check your query before executing it. If you get an error while executing a query,
rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

To start you should ALWAYS look at the tables in the database to see what you can query.
Do NOT skip this step.
Then you should query the schema of the most relevant tables.
"""
```

**프롬프트 설계 원칙**:
- 명확한 역할 정의
- 제약 조건 명시 (DML 금지)
- 단계별 지침 제공
- 오류 처리 방법 안내

---

### 3단계: ReAct Agent 초기화

```python
from langgraph.prebuilt import create_react_agent

# ReAct 에이전트 생성
agent_executor = create_react_agent(
    llm=llm,
    tools=tools,
    state_modifier=system_message
)

print("✅ ReAct Agent 생성 완료")
```

**create_react_agent 파라미터**:
- `llm`: 사용할 언어 모델
- `tools`: 에이전트가 사용할 도구 리스트
- `state_modifier`: 시스템 프롬프트 (선택)

---

### 4단계: Agent 실행 및 추적

```python
from langchain_core.messages import HumanMessage

question = "총보수가 0.1% 이하인 ETF는 모두 몇 개인가요?"

print(f"질문: {question}\n")
print("=" * 80)

# 스트리밍 실행으로 사고 과정 추적
for step in agent_executor.stream(
    {"messages": [HumanMessage(content=question)]},
    stream_mode="values"
):
    # 마지막 메시지 출력
    step["messages"][-1].pretty_print()
```

**출력 예시**:
```
================================ Human Message =================================
총보수가 0.1% 이하인 ETF는 모두 몇 개인가요?

================================== Ai Message ==================================
Tool Calls:
  sql_db_list_tables (call_xxx)
  Call ID: call_xxx
  Args: {}

================================= Tool Message =================================
Name: sql_db_list_tables
ETFs, ETFsInfo

================================== Ai Message ==================================
Tool Calls:
  sql_db_schema (call_yyy)
  Call ID: call_yyy
  Args:
    table_names: ETFs

================================= Tool Message =================================
Name: sql_db_schema
CREATE TABLE ETFs (
    종목코드 TEXT PRIMARY KEY,
    ...
    총보수 REAL,
    ...
)

================================== Ai Message ==================================
Tool Calls:
  sql_db_query (call_zzz)
  Call ID: call_zzz
  Args:
    query: SELECT COUNT(*) FROM ETFs WHERE 총보수 <= 0.1

================================= Tool Message =================================
Name: sql_db_query
[(42,)]

================================== Ai Message ==================================
총보수가 0.1% 이하인 ETF는 총 42개입니다.
```

---

## DB 테이블 관리

대규모 데이터베이스에서 관련 테이블만 선택하는 전략을 학습합니다.

### 특정 테이블만 선택

```python
from langchain_community.utilities import SQLDatabase

# 특정 테이블만 포함
db = SQLDatabase.from_uri(
    "sqlite:///etf_database.db",
    include_tables=['ETFs'],  # ETFs 테이블만 사용
    sample_rows_in_table_info=3
)

print(f"사용 가능한 테이블: {db.get_usable_table_names()}")
```

**장점**:
- 프롬프트 크기 감소
- 관련 없는 테이블 제외
- 쿼리 정확도 향상

---

### 복잡한 스키마 처리: 테이블 카테고리화

여러 테이블을 논리적 그룹으로 분류하여 관리합니다.

```python
from pydantic import BaseModel, Field
from typing import List

# 테이블 정의 모델
class Table(BaseModel):
    """SQL 데이터베이스의 테이블"""
    name: str = Field(description="테이블 이름")
    category: str = Field(description="테이블 카테고리")
    purpose: str = Field(description="테이블 목적")

class TableList(BaseModel):
    """테이블 목록"""
    tables: List[Table] = Field(description="관련 테이블들")

# 카테고리 정의 프롬프트
system = """
당신은 Sakila DVD 대여점 데이터베이스의 전문가입니다.
사용자 질문과 관련된 SQL 테이블들을 식별하고 적절한 카테고리로 분류하세요.

사용 가능한 카테고리:
- Film: 영화 관련 정보 (film, film_actor, film_category, actor, category, language)
- Customer: 고객 관련 정보 (customer, rental, payment)
- Location: 지역 관련 정보 (store, address, city, country)

질문과 관련된 모든 테이블을 반환하세요.
"""

category_chain = (
    ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{input}")
    ])
    | llm.with_structured_output(TableList)
)

# 테스트
question = "가장 많은 작품을 대여한 고객은 누구인가요?"
result = category_chain.invoke({"input": question})

for table in result.tables:
    print(f"- {table.name} ({table.category}): {table.purpose}")
```

---

### 동적 테이블 선택 체인

질문에 따라 필요한 테이블만 자동으로 선택합니다.

```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter

# 테이블 이름만 추출하는 체인
table_chain = (
    {"input": itemgetter("question")}
    | category_chain
    | RunnableLambda(lambda x: [t.name for t in x.tables])
)

# SQL 생성 체인과 결합
def get_table_info(table_names: list) -> str:
    """선택된 테이블의 스키마 정보 반환"""
    return db.get_table_info(table_names)

full_chain = (
    RunnablePassthrough.assign(tables=table_chain)
    | RunnablePassthrough.assign(
        table_info=lambda x: get_table_info(x["tables"])
    )
    | query_prompt_template
    | llm.with_structured_output(QueryOutput)
)

# 실행
question = "가장 많은 작품을 대여한 고객은?"
result = full_chain.invoke({"question": question})
print(f"생성된 쿼리: {result['query']}")
```

---

### Agent에 커스텀 테이블 선택 도구 추가

```python
from langchain_core.tools import tool

@tool
def select_relevant_tables(question: str) -> str:
    """질문과 관련된 테이블의 스키마 정보를 반환합니다."""
    try:
        # 테이블 선택
        selected_tables = table_chain.invoke({"question": question})

        # 스키마 정보 가져오기
        table_info = db.get_table_info(selected_tables)

        return f"관련 테이블: {', '.join(selected_tables)}\n\n{table_info}"
    except Exception as e:
        return f"테이블 선택 중 오류 발생: {str(e)}"

# 기존 SQL 도구와 결합
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_tools = toolkit.get_tools()

# 커스텀 도구 추가
custom_tools = [select_relevant_tables] + sql_tools

# Agent 생성
custom_agent = create_react_agent(
    llm=llm,
    tools=custom_tools,
    state_modifier="""
    You are an expert SQL agent with table selection capabilities.

    When given a question:
    1. FIRST use select_relevant_tables to identify relevant tables
    2. THEN use sql_db_query to execute queries on those tables
    3. Finally, provide a clear answer based on the results

    Answer all questions in Korean.
    """
)

# 실행
result = custom_agent.invoke({
    "messages": [HumanMessage(content="가장 많이 대여된 영화는?")]
})
print(result["messages"][-1].content)
```

---

## 실전 활용 예제

### 예제 1: 복잡한 집계 쿼리

```python
def analyze_etf_portfolio(question: str):
    """
    복잡한 ETF 포트폴리오 분석

    Agent 방식으로 다단계 분석 수행
    """
    result = agent_executor.invoke({
        "messages": [HumanMessage(content=question)]
    })

    # 최종 답변 추출
    final_answer = result["messages"][-1].content

    return final_answer

# 사용 예시
questions = [
    "운용사별 ETF 개수와 평균 총보수를 계산해주세요",
    "순자산총액 상위 10개 ETF의 수익률 분포는?",
    "분류체계가 '주식'인 ETF 중 총보수가 평균 이하인 것은?"
]

for question in questions:
    print(f"\n질문: {question}")
    answer = analyze_etf_portfolio(question)
    print(f"답변: {answer}")
    print("-" * 80)
```

### 예제 2: 조건부 필터링 시스템

```python
def filter_etfs_by_criteria(criteria: dict):
    """
    여러 조건으로 ETF 필터링

    Parameters:
        criteria (dict): {
            'max_cost': 0.2,
            'min_return': 5.0,
            'volatility': ['낮음', '매우낮음'],
            'limit': 10
        }

    Returns:
        str: 필터링 결과 답변
    """
    # 조건을 자연어 질문으로 변환
    conditions = []

    if 'max_cost' in criteria:
        conditions.append(f"총보수가 {criteria['max_cost']}% 이하")

    if 'min_return' in criteria:
        conditions.append(f"수익률이 {criteria['min_return']}% 이상")

    if 'volatility' in criteria:
        vol_str = ' 또는 '.join(criteria['volatility'])
        conditions.append(f"변동성이 {vol_str}")

    question = f"{' 그리고 '.join(conditions)}인 ETF를 "

    if 'limit' in criteria:
        question += f"{criteria['limit']}개만 알려주세요"
    else:
        question += "모두 알려주세요"

    # Agent로 실행
    return analyze_etf_portfolio(question)

# 실행
result = filter_etfs_by_criteria({
    'max_cost': 0.15,
    'min_return': 3.0,
    'volatility': ['낮음', '매우낮음'],
    'limit': 5
})
print(result)
```

### 예제 3: 비교 분석 시스템

```python
def compare_etf_groups(group_a: str, group_b: str):
    """
    두 그룹의 ETF 비교 분석

    Parameters:
        group_a (str): 첫 번째 그룹 조건
        group_b (str): 두 번째 그룹 조건

    Returns:
        dict: 비교 결과
    """
    question_a = f"{group_a}인 ETF들의 평균 총보수, 평균 수익률, 개수를 알려주세요"
    question_b = f"{group_b}인 ETF들의 평균 총보수, 평균 수익률, 개수를 알려주세요"

    result_a = analyze_etf_portfolio(question_a)
    result_b = analyze_etf_portfolio(question_b)

    return {
        'group_a': {
            'condition': group_a,
            'analysis': result_a
        },
        'group_b': {
            'condition': group_b,
            'analysis': result_b
        }
    }

# 실행
comparison = compare_etf_groups(
    group_a="삼성자산운용이 운용하는",
    group_b="미래에셋자산운용이 운용하는"
)

print("=== 그룹 A ===")
print(comparison['group_a']['analysis'])
print("\n=== 그룹 B ===")
print(comparison['group_b']['analysis'])
```

---

## 연습 문제

### 기본 문제

**문제 1**: Chain 방식 기본 구현
```python
# 과제: 다음 질문에 대한 Chain 방식 구현
# "수익률이 10% 이상인 ETF는 몇 개인가요?"

# State 정의
class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

# TODO: write_query, execute_query, generate_answer 함수 구현
# TODO: LangGraph 구성
```

**문제 2**: Agent 도구 확인
```python
# 과제: SQLDatabaseToolkit의 각 도구 역할 파악

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

# TODO: 각 도구를 실행해보고 출력 결과 확인
# 힌트: tool.invoke() 사용
```

**문제 3**: 프롬프트 수정
```python
# 과제: 한국어 답변을 생성하도록 프롬프트 수정

# TODO: generate_answer 함수의 프롬프트에
# "Answer in Korean (한국어로 답변)" 추가
```

### 중급 문제

**문제 4**: 조건부 엣지
```python
# 과제: 쿼리 실행 결과가 비어있으면 재시도하는 로직 추가

from langgraph.graph import END

def check_result(state: State):
    """결과가 비어있는지 확인"""
    # TODO: state["result"]가 비어있으면 "retry", 아니면 "continue" 반환
    pass

# TODO: 조건부 엣지 추가
# graph_builder.add_conditional_edges(
#     "execute_query",
#     check_result,
#     {"retry": "write_query", "continue": "generate_answer"}
# )
```

**문제 5**: 커스텀 도구 생성
```python
# 과제: ETF 추천 도구 만들기

from langchain_core.tools import tool

@tool
def recommend_low_cost_etfs(max_cost: float, limit: int = 5) -> str:
    """
    저비용 ETF 추천 도구

    Parameters:
        max_cost: 최대 총보수 (%)
        limit: 추천 개수

    Returns:
        str: 추천 ETF 목록
    """
    # TODO: SQL 쿼리 실행하여 추천 결과 반환
    pass

# TODO: Agent에 추가하여 테스트
```

**문제 6**: 오류 복구 메커니즘
```python
# 과제: Agent가 SQL 오류 발생 시 자동으로 재시도하는지 확인

# 의도적으로 잘못된 질문 실행
question = "존재하지_않는_컬럼으로 필터링해주세요"

# TODO: Agent 실행 과정 추적
# TODO: 오류 발생 → 재시도 → 수정 → 성공 과정 확인
```

### 고급 문제

**문제 7**: 다중 쿼리 Chain
```python
# 과제: 여러 쿼리를 순차적으로 실행하는 Chain 구현

class MultiQueryState(TypedDict):
    question: str
    queries: list[str]  # 여러 쿼리
    results: list[str]  # 각 쿼리 결과
    answer: str

# TODO:
# 1. 질문을 분석하여 여러 쿼리 생성
# 2. 각 쿼리를 순차 실행
# 3. 모든 결과를 종합하여 답변 생성

# 테스트 질문:
# "운용사별 ETF 개수와 각 운용사의 평균 총보수를 비교해주세요"
```

**문제 8**: 대화형 SQL Agent
```python
# 과제: 이전 질문의 컨텍스트를 유지하는 대화형 Agent

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

class ConversationalSQLAgent:
    def __init__(self, db, llm):
        self.db = db
        self.llm = llm
        self.history = []

    def chat(self, question: str) -> str:
        """
        대화 컨텍스트를 유지하며 질문 처리

        TODO:
        1. 이전 대화 내역을 프롬프트에 포함
        2. Agent 실행
        3. 대화 내역에 추가
        4. 답변 반환
        """
        pass

# 테스트:
# agent = ConversationalSQLAgent(db, llm)
# agent.chat("총보수가 낮은 ETF 5개 알려줘")
# agent.chat("그 중에서 수익률이 가장 높은 것은?")  # 이전 결과 참조
```

**문제 9**: 성능 모니터링
```python
# 과제: Chain과 Agent의 성능 비교 시스템 구현

import time

def benchmark_system(questions: list[str]):
    """
    Chain과 Agent의 성능 비교

    측정 항목:
    - 실행 시간
    - LLM 호출 횟수
    - 토큰 사용량
    - 정확도 (수동 검증)

    Returns:
        dict: 비교 결과
    """
    # TODO: 각 질문에 대해 Chain과 Agent 실행
    # TODO: 성능 지표 수집
    # TODO: 결과 비교 및 시각화
    pass

# 테스트 질문 세트
test_questions = [
    "총보수가 0.1% 이하인 ETF는?",  # 단순
    "운용사별 평균 수익률은?",  # 집계
    "수익률 상위 10개 중 총보수가 평균 이하인 ETF는?"  # 복잡
]
```

---

## 문제 해결 가이드

### Chain 방식 문제

#### 1. State 업데이트 오류
```python
# 문제: 노드가 State를 올바르게 업데이트하지 않음

# 잘못된 예시
def write_query(state: State):
    state["query"] = "SELECT ..."  # 직접 수정 (불가)
    return state

# 올바른 예시
def write_query(state: State):
    return {"query": "SELECT ..."}  # 새 딕셔너리 반환
```

#### 2. 순환 참조 오류
```python
# 문제: 엣지가 순환 구조를 형성

# 해결: END 노드 사용
from langgraph.graph import END

graph_builder.add_edge("generate_answer", END)
```

#### 3. 프롬프트 변수 누락
```python
# 문제: 프롬프트에 필요한 변수를 전달하지 않음

# 디버깅
print(query_prompt_template.input_variables)
# ['dialect', 'top_k', 'table_info', 'input']

# 모든 변수 전달 확인
prompt = query_prompt_template.invoke({
    "dialect": db.dialect,
    "top_k": 10,
    "table_info": db.get_table_info(),
    "input": state["question"]
})
```

### Agent 방식 문제

#### 1. 무한 루프
```python
# 문제: Agent가 종료 조건을 찾지 못함

# 해결: max_iterations 설정
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    llm=llm,
    tools=tools,
    state_modifier=system_message,
    max_iterations=10  # 최대 반복 횟수 제한
)
```

#### 2. 도구 사용 오류
```python
# 문제: Agent가 잘못된 인자로 도구 호출

# 해결: 도구 설명 개선
@tool
def my_tool(arg1: str, arg2: int) -> str:
    """
    명확한 설명 작성

    Parameters:
        arg1: 설명 (예시: "고객 이름")
        arg2: 설명 (예시: 개수, 1-10 범위)

    Returns:
        설명
    """
    pass
```

#### 3. 쿼리 구문 오류 반복
```python
# 문제: Agent가 같은 오류를 반복

# 해결: 프롬프트에 명시적 지침 추가
system_message = """
...
If you get an error:
1. Carefully read the error message
2. Check the table schema again
3. Rewrite the query with correct column names
4. DO NOT repeat the same mistake

Common mistakes to avoid:
- Using non-existent columns
- Incorrect table names
- Wrong data types in WHERE clause
...
"""
```

### 성능 최적화

#### 테이블 정보 캐싱
```python
# 문제: 매번 get_table_info() 호출로 느림

# 해결: 캐싱
from functools import lru_cache

@lru_cache(maxsize=10)
def get_cached_table_info(table_names_tuple: tuple):
    """테이블 정보 캐싱"""
    return db.get_table_info(list(table_names_tuple))

# 사용
table_info = get_cached_table_info(tuple(table_names))
```

#### 병렬 쿼리 실행
```python
# 독립적인 쿼리를 병렬로 실행

import asyncio
from langchain_community.tools import QuerySQLDatabaseTool

async def run_parallel_queries(queries: list[str]):
    """여러 쿼리를 병렬 실행"""
    tool = QuerySQLDatabaseTool(db=db)

    async def run_query(query):
        return await tool.ainvoke(query)

    results = await asyncio.gather(*[run_query(q) for q in queries])
    return results

# 사용
queries = [
    "SELECT COUNT(*) FROM ETFs",
    "SELECT AVG(총보수) FROM ETFs",
    "SELECT AVG(수익률_최근1년) FROM ETFs"
]

results = asyncio.run(run_parallel_queries(queries))
```

---

## 추가 학습 자료

### 공식 문서
- [LangChain SQL QA Tutorial](https://python.langchain.com/docs/tutorials/sql_qa/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [ReAct Agent Pattern](https://python.langchain.com/docs/modules/agents/agent_types/react/)
- [SQLDatabaseToolkit API](https://python.langchain.com/api_reference/community/agent_toolkits/langchain_community.agent_toolkits.sql.toolkit.SQLDatabaseToolkit.html)

### 예제 데이터베이스
- [Sakila Sample Database](https://www.kaggle.com/datasets/atanaskanev/sqlite-sakila-sample-database)
- [Chinook Database](https://github.com/lerocha/chinook-database)
- [Northwind Database](https://github.com/pthom/northwind_psql)

### 다음 단계
1. **스트리밍 응답**: 실시간으로 답변 생성 과정 표시
2. **대화형 인터페이스**: Gradio/Streamlit으로 UI 구축
3. **쿼리 최적화**: EXPLAIN PLAN 분석 및 인덱스 최적화
4. **보안 강화**: SQL Injection 방지 및 권한 관리
5. **멀티 데이터베이스**: PostgreSQL, MySQL 등 다양한 DB 지원

### 심화 주제
- **Semantic Layer**: 비즈니스 로직을 추상화한 의미론적 계층
- **Query Decomposition**: 복잡한 질문을 여러 단순 쿼리로 분해
- **Self-Correction**: Agent가 스스로 오류를 감지하고 수정
- **Multi-Agent Collaboration**: 여러 Agent가 협력하여 문제 해결
- **Hybrid Search**: SQL + Vector Search 결합

---

## 요약

이 가이드에서 학습한 핵심 내용:

✅ **Chain vs Agent 패턴 이해**
- Chain: 순차적, 예측 가능, 단순 쿼리에 적합
- Agent: 자율적, 복잡한 분석, 오류 자동 복구

✅ **LangGraph 상태 관리**
- TypedDict 기반 상태 정의
- 노드와 엣지로 워크플로우 구성
- 스트리밍 실행 및 추적

✅ **ReAct 에이전트 구현**
- SQLDatabaseToolkit 활용
- 도구 선택 및 반복적 추론
- 커스텀 도구 추가

✅ **테이블 선택 전략**
- 특정 테이블 필터링
- 카테고리 기반 그룹화
- 동적 테이블 선택 체인

✅ **실전 활용 패턴**
- 복잡한 집계 쿼리
- 조건부 필터링
- 비교 분석 시스템

이제 자연어로 데이터베이스와 대화하는 지능형 SQL QA 시스템을 구축할 수 있습니다!
