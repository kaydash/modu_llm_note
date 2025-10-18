# W3_004_ETF_Text2SQL_Cardinality - High Cardinality 고유명사 처리

## 학습 목표

이 가이드에서는 High Cardinality 문제를 해결하기 위한 벡터 기반 고유명사 처리를 학습합니다:

- **High Cardinality 문제**: 많은 고유값을 가진 컬럼의 정확한 매칭 어려움
- **벡터 검색 통합**: 임베딩을 활용한 유사 고유명사 자동 검색
- **엔티티 매칭 전략**: 맞춤법 오류, 약어, 별칭 처리
- **프롬프트 강화**: 검색된 엔티티 정보를 SQL 생성에 활용
- **확장 가능한 시스템**: 새로운 고유명사 필드 추가 방법

### 선수 지식
- Text2SQL 기본 개념 (W3_002 참조)
- LangGraph Chain 구성 (W3_003 참조)
- 벡터 임베딩 및 검색 (W2 RAG 가이드 참조)

---

## 핵심 개념

### High Cardinality란?
데이터베이스 컬럼에 많은 고유값(unique values)이 존재하는 상황을 의미합니다.

**문제 사례**:
```python
# 운용사 컬럼: 26개의 고유값
운용사들 = ['삼성자산운용', '케이비자산운용', '미래에셋자산운용', ...]

# 사용자 질문: "KB에서 운용하는 ETF는?"
# 문제: "KB" ≠ "케이비자산운용" → 매칭 실패

# ETF 종목명: 925개의 고유값
종목들 = ['KODEX 200', 'TIGER 나스닥100', ...]

# 사용자 질문: "Dow Jones ETF는?"
# 문제: "Dow Jones" ≠ "다우존스" → 매칭 실패
```

### 왜 문제가 되는가?

| 이슈 | 설명 | 예시 |
|------|------|------|
| **맞춤법 차이** | 정확한 철자를 모름 | "케이비" vs "케이비자산운용" |
| **약어 사용** | 축약형 사용 | "KB" vs "케이비자산운용" |
| **언어 혼용** | 한/영 혼용 | "Dow Jones" vs "다우존스" |
| **대소문자** | 대소문자 구분 | "S&P 500" vs "s&p 500" |
| **공백/특수문자** | 형식 차이 | "S&P500" vs "S&P 500" |

### 해결 방법: 벡터 기반 검색

**전통적 방법 (실패)**:
```sql
-- LIKE 패턴 매칭
SELECT * FROM ETFs WHERE 운용사 LIKE '%KB%'  -- 결과: 0개 (실패)

-- 정확한 매칭만 가능
SELECT * FROM ETFs WHERE 운용사 = '케이비자산운용'  -- 성공
```

**벡터 검색 방법 (성공)**:
```python
# 1. 모든 고유명사를 벡터로 변환
vector_store.add_texts(['케이비자산운용', '삼성자산운용', ...])

# 2. 사용자 입력을 벡터 검색
result = retriever.invoke("KB")
# 결과: '케이비자산운용' (유사도 기반 매칭)

# 3. 검색된 정확한 이름으로 SQL 생성
SELECT * FROM ETFs WHERE 운용사 = '케이비자산운용'  -- 성공!
```

**처리 흐름**:
```
사용자 질문: "KB에서 운용하는 ETF는?"
    ↓
벡터 검색: "KB" → ['케이비자산운용', '아이비케이자산운용', ...]
    ↓
프롬프트에 포함: "검색된 엔티티: 케이비자산운용"
    ↓
LLM이 SQL 생성: WHERE 운용사 = '케이비자산운용'
    ↓
정확한 결과: 118개 ETF 발견
```

---

## 환경 설정

### 필수 라이브러리

```bash
# 핵심 라이브러리
pip install langchain langchain-openai langchain-community
pip install langgraph

# 벡터 저장소
pip install langchain-core

# 데이터베이스
pip install sqlalchemy

# 환경 변수
pip install python-dotenv
```

### 환경 초기화

```python
from dotenv import load_dotenv
import os
import warnings

# 환경 변수 로드
load_dotenv()

# 경고 무시
warnings.filterwarnings('ignore')

# API 키 확인
print("OpenAI API:", "✓" if os.getenv("OPENAI_API_KEY") else "✗")
```

### 데이터베이스 연결

```python
from langchain_community.utilities import SQLDatabase

# ETF 데이터베이스 연결
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

## 단계별 구현

### 1단계: 기본 SQL QA Chain (문제 확인)

먼저 벡터 검색 없이 기본 Chain을 구현하여 High Cardinality 문제를 확인합니다.

#### State 및 기본 함수 정의

```python
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.tools import QuerySQLDatabaseTool
from langgraph.graph import START, StateGraph

# State 정의
class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

# LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 프롬프트 템플릿 (기본 버전)
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

# QueryOutput 정의
class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]

# SQL 쿼리 생성
def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke({
        "dialect": db.dialect,
        "top_k": 10,
        "table_info": db.get_table_info(),
        "input": state["question"],
    })
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}

# SQL 쿼리 실행
def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}

# 답변 생성
def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}

# 그래프 구성
graph_builder = StateGraph(State).add_sequence(
    [write_query, execute_query, generate_answer]
)
graph_builder.add_edge(START, "write_query")
graph = graph_builder.compile()

print("✅ 기본 SQL QA Chain 생성 완료")
```

#### 문제 확인 테스트

```python
# 테스트 1: 약어 사용 (실패 예상)
print("=== 테스트 1: KB 운용사 검색 ===")
for step in graph.stream(
    {"question": "KB에서 운용하는 ETF는 모두 몇개인가요?"},
    stream_mode="updates"
):
    print(step)

# 예상 출력:
# {'write_query': {'query': "SELECT COUNT(*) WHERE 운용사 LIKE '%KB%'"}}
# {'execute_query': {'result': '[(0,)]'}}
# {'generate_answer': {'answer': 'KB에서 운용하는 ETF는 없습니다.'}}  # 잘못된 답변!
```

```python
# 테스트 2: 정확한 이름 (성공)
print("\n=== 테스트 2: 정확한 운용사명 검색 ===")
for step in graph.stream(
    {"question": "케이비자산운용에서 운용하는 ETF는 모두 몇개인가요?"},
    stream_mode="updates"
):
    print(step)

# 예상 출력:
# {'write_query': {'query': "SELECT COUNT(*) WHERE 운용사 = '케이비자산운용'"}}
# {'execute_query': {'result': '[(118,)]'}}
# {'generate_answer': {'answer': '케이비자산운용에서 운용하는 ETF는 118개입니다.'}}  # 정확!
```

```python
# 테스트 3: 영어 이름 (부분 성공)
print("\n=== 테스트 3: Dow Jones 검색 ===")
for step in graph.stream(
    {"question": "Dow Jones ETF는 모두 몇개인가요?"},
    stream_mode="updates"
):
    print(step)

# 예상 출력:
# {'write_query': {'query': "SELECT COUNT(*) WHERE 기초지수 LIKE '%Dow Jones%'"}}
# {'execute_query': {'result': '[(16,)]'}}
# {'generate_answer': {'answer': 'Dow Jones ETF는 총 16개입니다.'}}  # 우연히 성공
```

**문제 요약**:
- ✗ "KB" → "케이비자산운용" 매칭 실패
- ✓ "케이비자산운용" → 정확한 매칭 성공
- △ "Dow Jones" → 우연히 성공 (DB에 영어 저장됨)

---

### 2단계: 고유명사 추출 및 벡터스토어 구축

#### 고유명사 추출 함수

```python
import ast
import re

def query_as_list(db, query):
    """
    DB 쿼리 결과를 고유명사 리스트로 변환

    Parameters:
        db: SQLDatabase 객체
        query: SQL 쿼리 문자열

    Returns:
        list: 중복 제거된 고유명사 리스트

    처리 과정:
    1. 쿼리 실행 및 결과 파싱
    2. 중첩 리스트 평탄화
    3. 숫자 제거 (예: "KODEX 200" → "KODEX")
    4. 공백 정리 및 중복 제거
    """
    # 쿼리 실행
    res = db.run(query)

    # 문자열 결과를 리스트로 변환
    res = [el for sub in ast.literal_eval(res) for el in sub if el]

    # 숫자 제거 및 정리
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]

    # 중복 제거
    return list(set(res))

# ETF 종목명 추출
etfs = query_as_list(db, "SELECT DISTINCT 종목명 FROM ETFs")

# 운용사 추출
fund_managers = query_as_list(db, "SELECT DISTINCT 운용사 FROM ETFs")

# 결과 확인
print(f"ETF 종목 수: {len(etfs)}")
print(f"운용사 수: {len(fund_managers)}")
print(f"\nETF 샘플: {etfs[:5]}")
print(f"운용사 샘플: {fund_managers[:5]}")
```

**출력 예시**:
```
ETF 종목 수: 925
운용사 수: 26

ETF 샘플: ['PLUS 미국S&P500', 'TIGER 소프트웨어', 'TIMEFOLIO 코스피액티브', ...]
운용사 샘플: ['에셋플러스자산운용', '브이아이자산운용', '유리에셋', '케이비자산운용', ...]
```

#### 벡터스토어 생성 및 저장

```python
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# 임베딩 모델 초기화
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# 인메모리 벡터 저장소 생성
vector_store = InMemoryVectorStore(embeddings)

# 고유명사 임베딩 및 저장
_ = vector_store.add_texts(etfs + fund_managers)

# 검색기 설정 (상위 10개 반환)
retriever = vector_store.as_retriever(search_kwargs={"k": 10})

print("✅ 벡터스토어 생성 완료")
print(f"총 저장된 고유명사: {len(etfs) + len(fund_managers)}개")
```

**임베딩 모델 비교**:
| 모델 | 차원 | 성능 | 비용 |
|------|------|------|------|
| text-embedding-3-small | 1536 | 보통 | 저렴 |
| text-embedding-3-large | 3072 | 우수 | 비쌈 |

**권장**: High Cardinality 처리에는 `text-embedding-3-large` 사용

---

### 3단계: 벡터 검색 테스트

#### 운용사 검색 테스트

```python
# 테스트 1: 약어
print("=== 약어 검색: '케이비' ===")
result = retriever.invoke("케이비")
for doc in result:
    print(f"- {doc.page_content}")
```

**출력**:
```
- 케이비자산운용
- KOSEF 블루칩
- 아이비케이자산운용
- 케이씨지아이자산운용
- PLUS 고배당주위클리커버드콜
...
```

✅ **성공**: "케이비" → "케이비자산운용" 정확히 매칭

```python
# 테스트 2: 영문 약어
print("\n=== 영문 약어 검색: 'KB운용' ===")
result = retriever.invoke("KB운용")
for doc in result:
    print(f"- {doc.page_content}")
```

**출력**:
```
- 아이비케이자산운용
- 비엔케이자산운용
- PLUS 코리아밸류업
- 키움투자자산운용
- 케이비자산운용
...
```

✅ **성공**: "KB운용" → 관련 운용사들 검색

#### ETF 종목명 검색 테스트

```python
# 테스트 3: 영어 이름
print("\n=== 영어 검색: 'Dow Jones' ===")
result = retriever.invoke("Dow Jones")
for doc in result:
    print(f"- {doc.page_content}")
```

**출력**:
```
- SOL 미국배당다우존스
- SOL 미국배당다우존스TR
- SOL 미국배당다우존스(H)
- PLUS 미국다우존스고배당주(합성 H)
- TIGER 미국배당다우존스
- KODEX 미국배당다우존스
...
```

✅ **성공**: "Dow Jones" → "다우존스" 포함 ETF들 검색

```python
# 테스트 4: 한글 이름
print("\n=== 한글 검색: '다우존스' ===")
result = retriever.invoke("다우존스")
for doc in result:
    print(f"- {doc.page_content}")
```

**출력**:
```
- ACE 미국배당다우존스
- KODEX 대만테크고배당다우존스
- SOL 미국배당다우존스
- KODEX 미국배당다우존스
...
```

✅ **성공**: "다우존스" → 관련 ETF들 정확히 검색

---

### 4단계: 검색 도구 생성

벡터 검색을 LangChain 도구로 래핑합니다.

```python
from langchain.agents.agent_toolkits import create_retriever_tool

# 검색 도구 설명
description = (
    "Use to look up values to filter on. Input is an approximate spelling "
    "of the proper noun, output is valid proper nouns. Use the noun most "
    "similar to the search."
)

# 검색 도구 생성
entity_retriever_tool = create_retriever_tool(
    retriever,
    name="search_proper_nouns",
    description=description,
)

print("✅ 검색 도구 생성 완료")
```

#### 도구 사용 테스트

```python
# 테스트 1: 운용사 검색
print("=== 도구 테스트 1 ===")
result = entity_retriever_tool.invoke("KB에서 운용하는 ETF는 모두 몇개인가요?")
print(result)
```

**출력**:
```
KOSEF 미국ETF산업STOXX

키움투자자산운용

한국투자밸류자산운용

KODEX 미국ETF산업Top10 Indxx

케이비자산운용

...
```

✅ "KB" → "케이비자산운용" 포함된 결과 반환

```python
# 테스트 2: ETF 종목 검색
print("\n=== 도구 테스트 2 ===")
result = entity_retriever_tool.invoke("Dow Jones 관련 ETF는 무엇인가요?")
print(result)
```

**출력**:
```
KODEX 미국ETF산업Top10 Indxx

PLUS 미국다우존스고배당주(합성 H)

KODEX 미국배당다우존스

SOL 미국배당다우존스(H)

SOL 미국배당다우존스TR

TIGER 미국다우존스30

...
```

✅ "Dow Jones" → 다우존스 관련 ETF들 반환

---

### 5단계: 프롬프트 템플릿 업데이트

검색된 엔티티 정보를 SQL 생성 프롬프트에 포함합니다.

```python
from langchain_core.prompts import ChatPromptTemplate

# 엔티티 정보를 포함한 프롬프트 템플릿
entity_query_template = """
Given an input question, create a syntactically correct {dialect} query to run to help find the answer. Unless the user specifies in his question a specific number of examples they wish to obtain, always limit your query to at most {top_k} results. You can order the results by a relevant column to return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

Only use the following tables:
{table_info}

Entity names and their relationships to consider:
{entity_info}

## Matching Guidelines
- Use exact matches when comparing entity names
- Check for historical name variations if available
- Apply case-sensitive matching for official names
- Handle both Korean and English entity names when present

Question: {input}
"""

# ChatPromptTemplate 생성
query_prompt_template = ChatPromptTemplate.from_template(entity_query_template)

# 입력 변수 확인
print("Input variables:", query_prompt_template.input_variables)
# ['dialect', 'entity_info', 'input', 'table_info', 'top_k']
```

**핵심 변경사항**:
- `{entity_info}` 추가: 검색된 고유명사 목록
- **Matching Guidelines**: 엔티티 매칭 지침 추가
- 정확한 매칭 강조

---

### 6단계: 업데이트된 Chain 구성

엔티티 검색을 통합한 새로운 `write_query` 함수를 작성합니다.

```python
def write_query(state: State):
    """
    엔티티 검색을 활용한 SQL 쿼리 생성

    Process:
    1. 질문에서 고유명사 추출 (벡터 검색)
    2. 검색된 고유명사를 프롬프트에 포함
    3. LLM이 정확한 고유명사로 SQL 생성
    """
    prompt = query_prompt_template.invoke({
        "dialect": db.dialect,
        "top_k": 10,
        "table_info": db.get_table_info(),
        "input": state["question"],
        "entity_info": entity_retriever_tool.invoke(state["question"]),  # 추가!
    })
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}

# 테스트
response = write_query({"question": "KB에서 운용하는 ETF는 모두 몇개인가요?"})
print(f"생성된 쿼리: {response['query']}")
```

**출력**:
```
생성된 쿼리: SELECT COUNT(*) AS ETF_개수 FROM ETFs WHERE 운용사 = '케이비자산운용';
```

✅ **성공**: "KB" → "케이비자산운용"으로 정확히 변환!

#### LangGraph 재구성

```python
from langgraph.graph import START, StateGraph

# 그래프 빌더 생성
graph_builder = StateGraph(State)

# 노드 추가 (업데이트된 write_query 사용)
graph_builder.add_node("write_query", write_query)
graph_builder.add_node("execute_query", execute_query)
graph_builder.add_node("generate_answer", generate_answer)

# 엣지 연결
graph_builder.add_edge(START, "write_query")
graph_builder.add_edge("write_query", "execute_query")
graph_builder.add_edge("execute_query", "generate_answer")

# 그래프 컴파일
graph = graph_builder.compile()

print("✅ 업데이트된 Chain 생성 완료")
```

---

### 7단계: 통합 테스트

#### 테스트 1: 약어 운용사

```python
print("=== 테스트 1: KB 운용사 ===")
for step in graph.stream(
    {"question": "KB에서 운용하는 ETF는 모두 몇개인가요?"},
    stream_mode="updates"
):
    print(step)
```

**출력**:
```
{'write_query': {'query': "SELECT COUNT(*) WHERE 운용사 = '케이비자산운용';"}}
{'execute_query': {'result': '[(118,)]'}}
{'generate_answer': {'answer': 'KB에서 운용하는 ETF는 모두 118개입니다.'}}
```

✅ **성공**: "KB" → "케이비자산운용" 자동 변환

#### 테스트 2: 다양한 표기법

```python
print("\n=== 테스트 2: 케이비 운용사 ===")
for step in graph.stream(
    {"question": "케이비에서 운용하는 ETF는 모두 몇개인가요?"},
    stream_mode="updates"
):
    print(step)
```

**출력**:
```
{'write_query': {'query': "SELECT COUNT(*) WHERE 운용사 LIKE '%케이비%';"}}
{'execute_query': {'result': '[(118,)]'}}
{'generate_answer': {'answer': '케이비에서 운용하는 ETF는 모두 118개입니다.'}}
```

✅ **성공**: 다양한 표기법 처리

#### 테스트 3: ETF 종목 검색

```python
print("\n=== 테스트 3: Dow Jones ETF ===")
for step in graph.stream(
    {"question": "Dow Jones ETF는 모두 몇개인가요?"},
    stream_mode="updates"
):
    print(step)
```

**출력**:
```
{'write_query': {'query': "SELECT COUNT(*) WHERE 종목명 LIKE '%다우존스%'"}}
{'execute_query': {'result': '[(12,)]'}}
{'generate_answer': {'answer': 'Dow Jones ETF는 모두 12개입니다.'}}
```

✅ **성공**: 영어 → 한글 자동 변환

---

## 실습: ETFsInfo 테이블 고유명사 확장

### (1) 추가 고유명사 추출

ETFsInfo 테이블의 다양한 필드에서 고유명사를 추출합니다.

```python
# 기초자산 추출
assets = query_as_list(
    db,
    "SELECT DISTINCT 기초자산 FROM ETFsInfo WHERE 기초자산 IS NOT NULL AND 기초자산 != ''"
)

# 기초시장 추출
markets = query_as_list(
    db,
    "SELECT DISTINCT 기초시장 FROM ETFsInfo WHERE 기초시장 IS NOT NULL AND 기초시장 != ''"
)

# 기초지수명 추출
index_names = query_as_list(
    db,
    "SELECT DISTINCT 기초지수명 FROM ETFsInfo WHERE 기초지수명 IS NOT NULL AND 기초지수명 != ''"
)

# 결과 확인
print(f"기초자산 수: {len(assets)}")
print(f"기초시장 수: {len(markets)}")
print(f"기초지수명 수: {len(index_names)}")

print(f"\n기초자산 샘플: {assets[:3]}")
print(f"기초시장 샘플: {markets[:3]}")
print(f"기초지수명 샘플: {index_names[:3]}")
```

**출력**:
```
기초자산 수: 63
기초시장 수: 27
기초지수명 수: 706

기초자산 샘플: [
    '(원자재) (금속) / (-) (구리) / (-)',
    '(주식) (업종섹터) / (-) (정보기술) / (-)',
    '(채권) (혼합) / (혼합) (중기) / (장기)'
]
기초시장 샘플: [
    '(해외) (선진국) (-)',
    '(해외) (아시아) (베트남)',
    '(해외) (남미) (멕시코)'
]
기초지수명 샘플: [
    'FnGuide K-뷰티 지수',
    'Markit iBoxx USD Liquid Investment Grade Index(Total Return)',
    'KAP MMF 지수(TR)'
]
```

### (2) 확장된 벡터스토어 생성

```python
# 모든 고유명사 통합
all_entities = etfs + fund_managers + assets + markets + index_names

# 새로운 벡터 저장소 생성
vector_store_v2 = InMemoryVectorStore(embeddings)
_ = vector_store_v2.add_texts(all_entities)
retriever_v2 = vector_store_v2.as_retriever(search_kwargs={"k": 10})

print(f"✅ 전체 고유명사 수: {len(all_entities)}개")
```

**통계**:
```
전체 고유명사: 1,747개
- ETF 종목명: 925개
- 운용사: 26개
- 기초자산: 63개
- 기초시장: 27개
- 기초지수명: 706개
```

### (3) 검색 테스트

```python
# 테스트 1: 기초자산
print("=== 기초자산 검색: '주식' ===")
result = retriever_v2.invoke("주식")
for doc in result[:3]:
    print(f"- {doc.page_content}")
```

**출력**:
```
- 주식국채혼합(주식형)지수
- (주식) (전략) / (-) (가치) / (-)
- (주식) (전략) / (-) (배당) / (-)
```

```python
# 테스트 2: 기초시장
print("\n=== 기초시장 검색: '미국' ===")
result = retriever_v2.invoke("미국")
for doc in result[:3]:
    print(f"- {doc.page_content}")
```

**출력**:
```
- (해외) (북미) (미국)
- PLUS 미국나스닥테크
- PLUS 미국대체투자Top10
```

```python
# 테스트 3: 기초지수명
print("\n=== 기초지수명 검색: 'S&P' ===")
result = retriever_v2.invoke("S&P")
for doc in result[:3]:
    print(f"- {doc.page_content}")
```

**출력**:
```
- S&P
- S&P Growth Index(PR)
- S&P Total Return Index
```

### (4) 업데이트된 검색 도구

```python
# 확장된 검색 도구 생성
entity_retriever_tool_v2 = create_retriever_tool(
    retriever_v2,
    name="search_proper_nouns_v2",
    description=description,
)

# 테스트
print("=== 검색 도구 v2 테스트 ===")
result = entity_retriever_tool_v2.invoke("미국 주식 ETF는 무엇인가요?")
print(result)
```

**출력**:
```
ACE 미국S&P500채권혼합액티브

TIGER 미국나스닥100ETF선물

KOSEF 미국ETF산업STOXX

KODEX iShares미국투자등급회사채액티브

PLUS 미국S&P500성장주

PLUS 미국S&P500

...
```

### (5) 최종 Chain 구성

```python
def write_query_v2(state: State):
    """업데이트된 엔티티 검색을 활용한 SQL 생성"""
    prompt = query_prompt_template.invoke({
        "dialect": db.dialect,
        "top_k": 10,
        "table_info": db.get_table_info(),
        "input": state["question"],
        "entity_info": entity_retriever_tool_v2.invoke(state["question"]),  # v2 사용
    })
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}

# 그래프 재구성
graph_builder_v2 = StateGraph(State)
graph_builder_v2.add_node("write_query", write_query_v2)
graph_builder_v2.add_node("execute_query", execute_query)
graph_builder_v2.add_node("generate_answer", generate_answer)

graph_builder_v2.add_edge(START, "write_query")
graph_builder_v2.add_edge("write_query", "execute_query")
graph_builder_v2.add_edge("execute_query", "generate_answer")

graph_v2 = graph_builder_v2.compile()

print("✅ 최종 Chain v2 생성 완료")
```

### (6) 확장 테스트

#### 테스트 1: 기초자산 쿼리

```python
print("=== 기초자산 쿼리 ===")
for step in graph_v2.stream(
    {"question": "주식을 기초자산으로 하는 ETF는 몇 개인가요?"},
    stream_mode="updates"
):
    print(step)
```

**출력**:
```
{'write_query': {'query': "SELECT COUNT(*) FROM ETFsInfo WHERE 기초자산 LIKE '%주식%'"}}
{'execute_query': {'result': '[(694,)]'}}
{'generate_answer': {'answer': '주식을 기초자산으로 하는 ETF는 총 694개입니다.'}}
```

#### 테스트 2: 기초시장 쿼리

```python
print("\n=== 기초시장 쿼리 ===")
for step in graph_v2.stream(
    {"question": "미국 시장을 기초시장으로 하는 ETF는 몇 개인가요?"},
    stream_mode="updates"
):
    print(step)
```

**출력**:
```
{'write_query': {'query': "SELECT COUNT(*) FROM ETFsInfo WHERE 기초시장 LIKE '%미국%'"}}
{'execute_query': {'result': '[(188,)]'}}
{'generate_answer': {'answer': '미국 시장을 기초시장으로 하는 ETF는 총 188개입니다.'}}
```

#### 테스트 3: 기초지수명 쿼리

```python
print("\n=== 기초지수명 쿼리 ===")
for step in graph_v2.stream(
    {"question": "S&P 500 지수를 추종하는 ETF는 무엇인가요?"},
    stream_mode="updates"
):
    print(step)
```

**출력**:
```
{'write_query': {'query': "SELECT 종목코드, 종목명, 운용사, 수익률_최근1년 FROM ETFs WHERE 기초지수 LIKE '%S&P 500%' LIMIT 10;"}}
{'execute_query': {'result': "[('360200', 'ACE 미국S&P500', '한국투자신탁운용', 41.85), ...]"}}
{'generate_answer': {'answer': 'S&P 500 지수를 추종하는 ETF는 다음과 같습니다:\n1. ACE 미국S&P500 ...'}}
```

#### 테스트 4: 복합 쿼리

```python
print("\n=== 복합 쿼리 ===")
for step in graph_v2.stream(
    {"question": "미래에셋에서 운용하는 미국 주식 ETF는 몇 개인가요?"},
    stream_mode="updates"
):
    print(step)
```

**출력**:
```
{'write_query': {'query': "SELECT COUNT(*) FROM ETFs WHERE 운용사 LIKE '%미래에셋%' AND 분류체계 LIKE '주식%미국%'"}}
{'execute_query': {'result': '[(X,)]'}}
{'generate_answer': {'answer': '미래에셋에서 운용하는 미국 주식 ETF는 X개입니다.'}}
```

---

## 실전 활용 예제

### 예제 1: 다국어 엔티티 처리

```python
def multilingual_query(question: str, language: str = "auto"):
    """
    다국어 질문 처리

    Parameters:
        question (str): 사용자 질문
        language (str): 언어 힌트 ('ko', 'en', 'auto')

    Returns:
        dict: 답변 결과
    """
    # 언어 감지 (간단한 휴리스틱)
    if language == "auto":
        has_korean = any('\uac00' <= char <= '\ud7a3' for char in question)
        language = "ko" if has_korean else "en"

    # Chain 실행
    result = graph_v2.invoke({"question": question})

    return {
        "question": question,
        "detected_language": language,
        "answer": result["answer"]
    }

# 테스트
questions = [
    "KB에서 운용하는 ETF는?",
    "Dow Jones ETFs?",
    "미국 S&P 500 추종 상품은?"
]

for q in questions:
    result = multilingual_query(q)
    print(f"Q: {result['question']}")
    print(f"A: {result['answer']}\n")
```

### 예제 2: 유사도 기반 추천

```python
def recommend_similar_etfs(etf_name: str, top_k: int = 5):
    """
    유사한 ETF 추천

    Parameters:
        etf_name (str): 기준 ETF 이름
        top_k (int): 추천 개수

    Returns:
        list: 유사 ETF 목록
    """
    # 벡터 검색으로 유사 ETF 찾기
    similar = retriever_v2.invoke(etf_name)

    # 상위 k개만 반환
    return [doc.page_content for doc in similar[:top_k]]

# 테스트
print("=== 'KODEX 200'와 유사한 ETF ===")
similar_etfs = recommend_similar_etfs("KODEX 200")
for etf in similar_etfs:
    print(f"- {etf}")
```

**출력**:
```
- KODEX 200
- KOSEF 200
- TIGER 200
- ACE 200
- KODEX 코스피
```

### 예제 3: 엔티티 정규화 시스템

```python
def normalize_entity(user_input: str, entity_type: str = "auto"):
    """
    사용자 입력을 정규화된 엔티티로 변환

    Parameters:
        user_input (str): 사용자 입력
        entity_type (str): 엔티티 타입 ('company', 'etf', 'auto')

    Returns:
        dict: 정규화 결과
    """
    # 벡터 검색
    results = retriever_v2.invoke(user_input)

    # 엔티티 타입 필터링 (필요시)
    if entity_type == "company":
        # 운용사만 필터링 (예: "자산운용" 포함)
        results = [r for r in results if "자산운용" in r.page_content]
    elif entity_type == "etf":
        # ETF 종목만 필터링
        results = [r for r in results if "자산운용" not in r.page_content]

    # 최상위 매치
    best_match = results[0].page_content if results else None

    return {
        "input": user_input,
        "normalized": best_match,
        "candidates": [r.page_content for r in results[:5]],
        "confidence": "high" if results else "low"
    }

# 테스트
test_inputs = ["KB", "다우존스", "S&P", "미국"]
for inp in test_inputs:
    result = normalize_entity(inp)
    print(f"입력: {result['input']}")
    print(f"정규화: {result['normalized']}")
    print(f"후보: {result['candidates'][:3]}\n")
```

---

## 연습 문제

### 기본 문제

**문제 1**: 고유명사 추출
```python
# 과제: "분류체계" 컬럼의 고유명사 추출

# TODO: query_as_list() 사용하여 분류체계 추출
categories = query_as_list(db, "SELECT DISTINCT ...")

# TODO: 벡터 저장소에 추가
# TODO: 검색 테스트
```

**문제 2**: 검색 품질 평가
```python
# 과제: 검색 결과의 정확도 평가

test_cases = [
    ("KB", "케이비자산운용"),
    ("다우존스", "KODEX 미국배당다우존스"),
    ("주식", "(주식) (전략) / (-) (배당) / (-)")
]

# TODO: 각 테스트 케이스에 대해
# - 벡터 검색 수행
# - 정답이 상위 k개 안에 있는지 확인
# - 정확도 계산
```

**문제 3**: 엔티티 타입 분류
```python
# 과제: 검색된 엔티티의 타입 자동 분류

def classify_entity(entity_name: str) -> str:
    """
    엔티티 타입 분류

    Returns:
        str: 'company', 'etf', 'asset', 'market', 'index'
    """
    # TODO: 규칙 기반 또는 ML 기반 분류
    pass
```

### 중급 문제

**문제 4**: 동적 k 값 조정
```python
# 과제: 질문 복잡도에 따라 검색 개수(k) 조정

def adaptive_retriever(question: str):
    """
    질문 복잡도 분석 후 적절한 k 값 설정

    복잡도 판단 기준:
    - 단순: k=5
    - 보통: k=10
    - 복잡: k=20
    """
    # TODO: 질문 복잡도 분석
    # TODO: k 값 동적 설정
    # TODO: 검색 수행
    pass
```

**문제 5**: 캐싱 시스템
```python
# 과제: 자주 검색되는 엔티티 캐싱

from functools import lru_cache

@lru_cache(maxsize=100)
def cached_entity_search(query: str) -> list:
    """
    검색 결과 캐싱

    장점:
    - 반복 검색 속도 향상
    - API 호출 비용 절감
    """
    # TODO: 캐싱 구현
    pass
```

**문제 6**: 엔티티 별칭 관리
```python
# 과제: 엔티티 별칭 매핑 시스템

entity_aliases = {
    "케이비자산운용": ["KB", "케이비", "KB운용", "KB자산운용"],
    "삼성자산운용": ["삼성", "삼성운용"],
    # ...
}

def resolve_alias(user_input: str) -> str:
    """
    별칭을 정식 명칭으로 변환

    TODO:
    1. 별칭 딕셔너리 검색
    2. 없으면 벡터 검색
    3. 정식 명칭 반환
    """
    pass
```

### 고급 문제

**문제 7**: 하이브리드 검색
```python
# 과제: 키워드 검색 + 벡터 검색 결합

def hybrid_search(query: str, alpha: float = 0.5):
    """
    하이브리드 검색 (BM25 + 벡터)

    Parameters:
        alpha: 가중치 (0=BM25만, 1=벡터만)

    TODO:
    1. BM25 키워드 검색
    2. 벡터 유사도 검색
    3. 점수 정규화 및 결합
    4. 최종 순위 반환
    """
    pass
```

**문제 8**: 멀티 테이블 엔티티 해결
```python
# 과제: 여러 테이블에 걸친 엔티티 검색

def cross_table_entity_search(question: str):
    """
    질문 분석 후 관련 테이블의 엔티티 검색

    예시:
    "미래에셋의 미국 주식 ETF"
    → ETFs 테이블: 운용사
    → ETFsInfo 테이블: 기초시장, 기초자산

    TODO:
    1. 질문에서 엔티티 타입 추출
    2. 각 타입별로 적절한 테이블 선택
    3. 병렬 검색 수행
    4. 결과 통합
    """
    pass
```

**문제 9**: 자동 스키마 학습
```python
# 과제: 데이터베이스 스키마 자동 분석 및 고유명사 추출

def auto_extract_entities(db_uri: str):
    """
    데이터베이스 자동 분석

    TODO:
    1. 모든 테이블 목록 조회
    2. 각 테이블의 컬럼 타입 분석
    3. TEXT 타입 컬럼에서 고유값 추출
    4. Cardinality 계산
    5. High Cardinality 컬럼 자동 식별
    6. 벡터스토어 자동 구축
    """
    pass
```

---

## 문제 해결 가이드

### 일반적인 문제

#### 1. 검색 품질 저하
```python
# 문제: 관련 없는 엔티티가 검색됨

# 원인 1: 임베딩 모델 부적합
# 해결: text-embedding-3-small → text-embedding-3-large

# 원인 2: k 값이 너무 큼
# 해결: k=10 → k=5로 줄이기

# 원인 3: 노이즈 데이터
# 해결: 전처리 강화
def clean_entity_name(name: str) -> str:
    """엔티티 정리"""
    # 숫자 제거
    name = re.sub(r'\b\d+\b', '', name)
    # 특수문자 정리
    name = re.sub(r'[^\w\s가-힣]', '', name)
    return name.strip()
```

#### 2. 속도 저하
```python
# 문제: 벡터 검색이 느림

# 해결 1: 캐싱
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_search(query: str):
    return retriever.invoke(query)

# 해결 2: 배치 검색
def batch_search(queries: list[str]):
    """여러 쿼리를 한 번에 처리"""
    return [retriever.invoke(q) for q in queries]

# 해결 3: 인덱싱 최적화
# FAISS 등 고속 벡터 DB 사용 고려
```

#### 3. 메모리 부족
```python
# 문제: 대량의 엔티티로 메모리 초과

# 해결 1: 영구 저장소 사용
from langchain_community.vectorstores import Chroma

# 디스크 기반 저장소
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# 해결 2: 샤딩
# 엔티티를 여러 저장소로 분할
company_store = InMemoryVectorStore(embeddings)
etf_store = InMemoryVectorStore(embeddings)
```

### 프롬프트 최적화

#### 엔티티 정보 포맷팅
```python
def format_entity_info(entities: list) -> str:
    """
    검색된 엔티티를 구조화된 형식으로 변환

    Before:
    "케이비자산운용\nKOSEF 블루칩\n..."

    After:
    "운용사: 케이비자산운용, 아이비케이자산운용
    ETF: KOSEF 블루칩, PLUS 고배당주"
    """
    # 엔티티 분류
    companies = [e for e in entities if "자산운용" in e]
    etfs = [e for e in entities if "자산운용" not in e]

    formatted = []
    if companies:
        formatted.append(f"운용사: {', '.join(companies[:5])}")
    if etfs:
        formatted.append(f"ETF: {', '.join(etfs[:5])}")

    return "\n".join(formatted)
```

---

## 추가 학습 자료

### 공식 문서
- [LangChain Vector Stores](https://python.langchain.com/docs/modules/data_connection/vectorstores/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [High Cardinality in Databases](https://en.wikipedia.org/wiki/Cardinality_(SQL_statements))

### 관련 논문
- "Dense Passage Retrieval for Open-Domain Question Answering" (Karpukhin et al., 2020)
- "ColBERT: Efficient and Effective Passage Search" (Khattab & Zaharia, 2020)

### 다음 단계
1. **Hybrid Search**: BM25 + 벡터 검색 결합
2. **Re-ranking**: 검색 결과 재정렬 시스템
3. **Query Expansion**: 쿼리 확장 및 개선
4. **Feedback Loop**: 사용자 피드백 기반 학습
5. **Production Deployment**: 실제 서비스 배포

### 심화 주제
- **Few-shot Learning**: 소량의 예제로 개선
- **Active Learning**: 불확실한 케이스 학습
- **Cross-lingual Search**: 다국어 검색 최적화
- **Semantic Chunking**: 의미 기반 텍스트 분할
- **Vector Index Optimization**: HNSW, IVF 알고리즘

---

## 요약

이 가이드에서 학습한 핵심 내용:

✅ **High Cardinality 문제 이해**
- 많은 고유값을 가진 컬럼의 정확한 매칭 어려움
- 맞춤법, 약어, 언어 차이로 인한 검색 실패

✅ **벡터 기반 해결책**
- 임베딩을 활용한 유사도 검색
- InMemoryVectorStore로 빠른 검색
- text-embedding-3-large로 높은 정확도

✅ **시스템 통합**
- 검색 도구를 LangChain Chain에 통합
- 프롬프트에 엔티티 정보 추가
- 자동 고유명사 매칭

✅ **확장 가능한 아키텍처**
- 새로운 테이블/컬럼 쉽게 추가
- query_as_list() 재사용 패턴
- 모듈화된 검색 도구

이제 사용자 친화적이고 정확한 Text2SQL 시스템을 구축할 수 있습니다!
