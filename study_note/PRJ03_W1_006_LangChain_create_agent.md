# LangChain create_agent - Agent 생성 및 관리

## 📚 학습 목표

- **create_agent 함수**를 사용하여 LangChain Agent를 생성할 수 있다
- **도구(Tools) 통합**을 통해 Agent에 외부 기능을 추가할 수 있다
- **Middleware**를 활용하여 Agent의 동작을 커스터마이징할 수 있다
- **Checkpointing**을 통해 대화 상태를 저장하고 복원할 수 있다
- **실전 프로젝트**를 통해 웹 리서치 및 데이터베이스 Agent를 구현할 수 있다

## 🔑 핵심 개념

### create_agent 함수

**create_agent**는 LangChain의 표준 Agent 생성 함수로, 복잡한 설정 없이 강력한 Agent를 빠르게 구축할 수 있습니다.

**주요 특징:**
- **간편한 설정**: 최소한의 파라미터로 Agent 생성
- **도구 통합**: 다양한 도구를 Agent에 연결
- **Middleware 지원**: Agent 동작을 커스터마이징
- **상태 관리**: Checkpointing을 통한 대화 지속성
- **LangGraph 기반**: 내부적으로 LangGraph를 사용하여 복잡한 워크플로우 자동 관리

### Agent vs ReAct Agent

| 비교 항목 | create_agent | ReAct Agent (수동 구현) |
|----------|-------------|------------------------|
| 구현 난이도 | 낮음 | 높음 |
| 설정 복잡도 | 간단 | 복잡 |
| 상태 관리 | 자동 | 수동 |
| Middleware | 지원 | 수동 구현 필요 |
| Checkpointing | 내장 | 별도 구현 필요 |
| 확장성 | 높음 | 중간 |

### 관련 기술 스택

```python
# LangChain 핵심
langchain              # Agent 기본 기능
langchain-openai       # OpenAI 모델 통합
langchain-tavily       # 웹 검색 도구
langgraph              # Agent 워크플로우 관리

# 도구 및 유틸리티
python-dotenv          # 환경 변수 관리
langchain-community    # 커뮤니티 도구 (SQL 등)
```

## 🛠 환경 설정

### 필요한 라이브러리 설치

```bash
pip install langchain langchain-openai langchain-tavily langgraph
pip install python-dotenv
pip install langchain-community  # SQL Agent용 (선택)
```

### API 키 설정

```.env
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 기본 설정 코드

```python
from dotenv import load_dotenv
import os

# 환경 변수 로드
load_dotenv()

# API 키 확인
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

if not os.getenv("TAVILY_API_KEY"):
    print("⚠️ TAVILY_API_KEY가 설정되지 않았습니다. 웹 검색 기능을 사용할 수 없습니다.")

print("✅ 환경 설정 완료!")
```

## 💻 단계별 구현

### 1단계: 기본 Agent 생성

#### 1.1 도구 없는 기본 Agent

```python
from langchain.agents import create_agent

# 가장 간단한 Agent 생성 (도구 없음)
agent = create_agent(
    model="openai:gpt-4.1-nano",           # 사용할 LLM 모델
    tools=[],                               # 도구 목록 (빈 리스트)
    system_prompt="You are a helpful assistant."  # 시스템 프롬프트
)

# Agent 실행
result = agent.invoke({
    "messages": [
        {"role": "user", "content": "안녕하세요!"}
    ]
})

# 최종 응답 출력
print(result["messages"][-1].content)
```

**출력:**
```
안녕하세요! 무엇을 도와드릴까요?
```

**주요 파라미터:**
- `model`: LLM 모델 지정 (`"openai:gpt-4.1-nano"`, `"anthropic:claude-3-sonnet"` 등)
- `tools`: Agent가 사용할 도구 목록
- `system_prompt`: Agent의 역할과 행동 방식 정의

#### 1.2 Agent 응답 구조

```python
# result 객체 구조 확인
print("응답 구조:")
print(f"- 타입: {type(result)}")
print(f"- 키: {result.keys()}")
print(f"- 메시지 수: {len(result['messages'])}")

# 모든 메시지 확인
for i, msg in enumerate(result["messages"]):
    print(f"\n메시지 {i+1}: {msg.__class__.__name__}")
    print(f"  내용: {msg.content[:100]}...")
```

**예상 출력:**
```
응답 구조:
- 타입: <class 'dict'>
- 키: dict_keys(['messages'])
- 메시지 수: 2

메시지 1: HumanMessage
  내용: 안녕하세요!...

메시지 2: AIMessage
  내용: 안녕하세요! 무엇을 도와드릴까요?...
```

### 2단계: 도구(Tools) 통합

#### 2.1 기본 Tavily 검색 도구

```python
from langchain.agents import create_agent
from langchain_tavily import TavilySearch

# Tavily 검색 도구 초기화
search_tool = TavilySearch(
    max_results=5,                    # 최대 검색 결과 수
    topic="general",                  # 검색 주제 (general, news)
)

# 검색 도구를 포함한 Agent 생성
agent = create_agent(
    model="openai:gpt-4.1-nano",
    tools=[search_tool],
    system_prompt="You are a helpful research assistant that can search the web."
)

# Agent 실행 (웹 검색이 필요한 질문)
result = agent.invoke({
    "messages": [
        {"role": "user", "content": "2024년 노벨 물리학상 수상자는 누구인가요?"}
    ]
})

# 모든 메시지 확인 (검색 과정 포함)
for msg in result["messages"]:
    msg.pretty_print()
```

**실행 흐름:**
```
사용자 질문
    ↓
Agent가 검색 필요 판단
    ↓
TavilySearch 도구 호출
    ↓
검색 결과 수신
    ↓
결과 기반 답변 생성
```

**예상 출력:**
```
================================ Human Message =================================
2024년 노벨 물리학상 수상자는 누구인가요?

================================== Ai Message ==================================
Tool Calls:
  tavily_search (call_abc123)
 Call ID: call_abc123
  Args:
    query: 2024 노벨 물리학상 수상자

================================= Tool Message =================================
Name: tavily_search

[검색 결과: John Hopfield와 Geoffrey Hinton이 인공 신경망 연구로 수상...]

================================== Ai Message ==================================
2024년 노벨 물리학상은 John Hopfield와 Geoffrey Hinton이 수상했습니다.
이들은 인공 신경망에 대한 기초 연구로 수상했습니다.
```

#### 2.2 동적 파라미터 설정

Agent가 상황에 따라 검색 파라미터를 자동으로 조정할 수 있습니다.

```python
from langchain.agents import create_agent
from langchain_tavily import TavilySearch

search_tool = TavilySearch(
    max_results=5,
)

# 시스템 프롬프트에서 파라미터 사용 지침 제공
agent = create_agent(
    model="openai:gpt-4.1-mini",
    tools=[search_tool],
    system_prompt="""You are a research assistant.
    When searching for academic content, use include_domains=['wikipedia.org'].
    For news, use topic='news'."""
)

# Wikipedia에서만 검색하도록 요청
result = agent.invoke({
    "messages": [
        {"role": "user",
         "content": "Find information about quantum computing from Wikipedia only."}
    ]
})

for msg in result["messages"]:
    msg.pretty_print()
```

**주요 관찰:**
- Agent가 `include_domains=['wikipedia.org']` 파라미터를 자동으로 적용
- 시스템 프롬프트의 지침을 따라 적절한 도구 호출 방식 선택

**뉴스 검색 예시:**

```python
# 최신 뉴스 검색
result = agent.invoke({
    "messages": [
        {"role": "user",
         "content": "Get me the latest news about artificial intelligence from the past week."}
    ]
})

for msg in result["messages"]:
    msg.pretty_print()
```

#### 2.3 순차적 도구 사용

여러 도구를 조합하여 복잡한 작업을 수행할 수 있습니다.

```python
from langchain_tavily import TavilySearch, TavilyExtract

# 검색 도구
search_tool = TavilySearch(
    max_results=5,
    topic="general"
)

# 콘텐츠 추출 도구
extract_tool = TavilyExtract(
    extract_depth="basic",      # 추출 깊이 (basic, advanced)
    include_images=False        # 이미지 포함 여부
)

# 두 도구를 모두 사용하는 Agent
agent = create_agent(
    model="openai:gpt-4.1-mini",
    tools=[search_tool, extract_tool],
    system_prompt="""You are a research assistant.
    - Use tavily_search to find relevant URLs
    - Use tavily_extract to get detailed content from specific URLs
    """
)

# 순차적 도구 사용이 필요한 작업
result = agent.invoke({
    "messages": [
        {"role": "user",
         "content": "최신 AI 연구 동향을 찾고, 가장 관련성 높은 기사의 전체 내용을 요약해주세요."}
    ]
})

for msg in result["messages"]:
    msg.pretty_print()
```

**실행 순서:**
1. **TavilySearch**: 관련 URL 검색
2. **TavilyExtract**: 선택된 URL에서 상세 콘텐츠 추출
3. **LLM**: 추출된 콘텐츠 요약

### 3단계: Middleware 설정

Middleware는 Agent의 동작을 가로채서 추가 기능을 제공합니다.

#### 3.1 대화 요약 Middleware (SummarizationMiddleware)

긴 대화를 자동으로 요약하여 컨텍스트 길이를 관리합니다.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_tavily import TavilySearch

# 검색 도구
search_tool = TavilySearch(max_results=5)

# 요약 Middleware를 포함한 Agent
agent = create_agent(
    model="openai:gpt-4.1-mini",
    tools=[search_tool],
    middleware=[
        SummarizationMiddleware(
            model="openai:gpt-4.1-nano",     # 요약에 사용할 모델
            max_tokens_before_summary=500,   # 요약 시작 임계값 (토큰 수)
            messages_to_keep=3,              # 요약 후에도 유지할 최근 메시지 수
        )
    ],
    checkpointer=InMemorySaver(),  # 대화 상태 저장
)

# 첫 번째 대화
result = agent.invoke(
    {"messages": [{"role": "user", "content": "최신 AI 연구 동향을 찾아서 설명해주세요."}]},
    config={"configurable": {"thread_id": "custom_thread_001"}}
)

for msg in result["messages"]:
    msg.pretty_print()
```

**두 번째 대화 (컨텍스트 누적):**

```python
# 대화 계속 (동일한 thread_id)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "LLM과 생성형 AI를 중심으로 자세하게 조사해주세요."}]},
    config={"configurable": {"thread_id": "custom_thread_001"}}
)

for msg in result["messages"]:
    msg.pretty_print()
```

**세 번째 대화 (요약 트리거):**

```python
# 토큰 임계값 초과 시 자동 요약
result = agent.invoke(
    {"messages": [{"role": "user", "content": "에너지 부족 문제에 대해 조사해주세요."}]},
    config={"configurable": {"thread_id": "custom_thread_001"}}
)

# 메시지 목록 확인 - 요약된 것을 확인 가능
print(f"\n현재 메시지 수: {len(result['messages'])}")
for msg in result["messages"]:
    msg.pretty_print()
```

**SummarizationMiddleware 작동 원리:**
1. 대화가 진행되면서 메시지가 누적됨
2. 토큰 수가 `max_tokens_before_summary`를 초과하면 자동 요약
3. 오래된 메시지들을 하나의 요약 메시지로 대체
4. `messages_to_keep` 개수만큼 최근 메시지는 유지

#### 3.2 커스텀 Middleware 생성

자신만의 Middleware를 만들어 Agent 동작을 커스터마이징할 수 있습니다.

```python
from langchain.agents.middleware import AgentMiddleware
from typing import Any, Dict

class LoggingMiddleware(AgentMiddleware):
    """모든 도구 호출을 로깅하는 Middleware"""

    def before_model(self, state: Dict[str, Any], runtime) -> Dict[str, Any] | None:
        """모델 호출 전에 실행"""
        print(f"🤖 모델 호출 전: {len(state['messages'])} 메시지")
        return None

    def after_model(self, state: Dict[str, Any], runtime) -> Dict[str, Any] | None:
        """모델 호출 후에 실행"""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            print(f"🔧 도구 호출: {[tc['name'] for tc in last_message.tool_calls]}")
        return None

    def after_tools(self, state: Dict[str, Any], runtime) -> Dict[str, Any] | None:
        """도구 실행 후에 실행"""
        print(f"✅ 도구 실행 완료")
        return None
```

**Middleware 적용:**

```python
# 커스텀 Middleware를 포함한 Agent
agent = create_agent(
    model="openai:gpt-4.1-mini",
    tools=[search_tool],
    middleware=[LoggingMiddleware()]
)

# Agent 실행 (로깅 확인)
result = agent.invoke({
    "messages": [{"role": "user", "content": "2024년 노벨 물리학상 수상자는 누구인가요?"}]
})

for msg in result["messages"]:
    msg.pretty_print()
```

**예상 출력:**
```
🤖 모델 호출 전: 1 메시지
🔧 도구 호출: ['tavily_search']
✅ 도구 실행 완료
🤖 모델 호출 전: 3 메시지

================================ Human Message =================================
2024년 노벨 물리학상 수상자는 누구인가요?

================================== Ai Message ==================================
Tool Calls:
  tavily_search (call_abc123)
  ...

================================= Tool Message =================================
[검색 결과]

================================== Ai Message ==================================
2024년 노벨 물리학상은 John Hopfield와 Geoffrey Hinton이 수상했습니다.
```

**에러 처리 Middleware:**

```python
class ErrorHandlingMiddleware(AgentMiddleware):
    """도구 실행 오류를 처리하는 Middleware"""

    def after_tools(self, state: Dict[str, Any], runtime) -> Dict[str, Any] | None:
        """도구 실행 후 오류 확인"""
        from langchain_core.messages import ToolMessage

        for msg in state["messages"]:
            if isinstance(msg, ToolMessage) and msg.status == "error":
                print(f"⚠️ 도구 실행 오류 발생: {msg.content}")
                # 여기서 오류 처리 로직 추가 가능

        return None

# 에러 처리 Middleware 적용
agent = create_agent(
    model="openai:gpt-4.1-mini",
    tools=[search_tool],
    middleware=[LoggingMiddleware(), ErrorHandlingMiddleware()]
)
```

#### 3.3 대화 영속성 (Checkpointing)

Checkpointing을 통해 대화의 상태를 저장하고 복원할 수 있습니다.

```python
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain_tavily import TavilySearch

# 메모리 체크포인터 생성
checkpointer = InMemorySaver()

search_tool = TavilySearch(max_results=5)

agent = create_agent(
    model="openai:gpt-4.1-nano",
    tools=[search_tool],
    checkpointer=checkpointer  # Checkpointing 활성화
)

# 첫 번째 대화
config1 = {"configurable": {"thread_id": "user-123-session-1"}}
result1 = agent.invoke(
    {"messages": [{"role": "user", "content": "내 이름은 김철수야"}]},
    config1
)
for msg in result1["messages"]:
    msg.pretty_print()
```

**두 번째 대화 (같은 thread_id로 이전 대화 기억):**

```python
# 같은 thread_id 사용 → 이전 대화 기억
result2 = agent.invoke(
    {"messages": [{"role": "user", "content": "내 이름이 뭐였지?"}]},
    config1
)
for msg in result2["messages"]:
    msg.pretty_print()
```

**예상 출력:**
```
================================== Ai Message ==================================
당신의 이름은 김철수입니다.
```

**다른 세션 (다른 thread_id):**

```python
# 다른 thread_id로 새로운 세션
config2 = {"configurable": {"thread_id": "user-123-session-2"}}
result3 = agent.invoke(
    {"messages": [{"role": "user", "content": "내 이름이 뭐야?"}]},
    config2
)
for msg in result3["messages"]:
    msg.pretty_print()
```

**예상 출력:**
```
================================== Ai Message ==================================
죄송하지만 저는 당신의 이름을 알지 못합니다. 알려주시겠어요?
```

**Checkpointing 주요 특징:**
- ✅ `thread_id`로 각 대화 세션 구분
- ✅ 같은 `thread_id` 사용 시 이전 대화 기억
- ✅ 다른 `thread_id` 사용 시 독립적인 새 세션
- ✅ `InMemorySaver`: 메모리에 저장 (프로세스 종료 시 삭제)
- ✅ `SqliteSaver`, `PostgresSaver`: 영구 저장 가능

### 4단계: 실전 프로젝트

#### 4.1 웹 리서치 Agent

복잡한 리서치 작업을 수행하는 Agent를 구현합니다.

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch, TavilyExtract
from langgraph.checkpoint.memory import InMemorySaver

# 도구 설정
search_tool = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced"  # 고급 검색 (더 많은 정보)
)

extract_tool = TavilyExtract(
    extract_depth="advanced"  # 고급 추출 (전체 콘텐츠)
)

# 모델 인스턴스를 직접 생성하여 세부 설정
model = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0.3,          # 창의성과 정확성 균형
    max_completion_tokens=5000
)

# 웹 리서치 Agent 생성
research_agent = create_agent(
    model=model,
    tools=[search_tool, extract_tool],
    system_prompt="""You are an expert web researcher. Your task is to:

1. Search for relevant information using tavily_search
2. Extract detailed content from the most relevant sources using tavily_extract
3. Synthesize the information into a comprehensive summary
4. Cite your sources with URLs

Always be thorough and accurate in your research.""",
    checkpointer=InMemorySaver()
)

# 복잡한 리서치 작업
config = {"configurable": {"thread_id": "research-001"}}

result = research_agent.invoke(
    {"messages": [
        {"role": "user",
         "content": """
         다음 주제에 대해 상세히 조사해주세요:
         1. 2024년 AI 분야의 주요 발전
         2. GPT-4와 Claude 3의 차이점
         3. 향후 AI 산업 전망

         각 주제별로 최소 3개 이상의 출처를 참고하고,
         출처 URL을 포함해서 정리해주세요.
         """}
    ]},
    config=config
)

# 결과 출력
print("\n" + "="*80)
print("웹 리서치 결과")
print("="*80)
final_answer = result["messages"][-1].content
print(final_answer)
```

**예상 출력:**
```
================================================================================
웹 리서치 결과
================================================================================
# 2024년 AI 분야 종합 조사 결과

## 1. 2024년 AI 분야의 주요 발전

### 1.1 대규모 언어 모델의 발전
- GPT-4.5와 Claude 3 Opus 출시로 추론 능력 대폭 향상
- 멀티모달 기능 강화 (텍스트, 이미지, 음성 통합)
출처: https://openai.com/research/gpt-4.5

### 1.2 AI 에이전트 시스템
- AutoGPT, BabyAGI 등 자율 에이전트 프레임워크 발전
- 도구 사용 능력 향상으로 실무 적용 증가
출처: https://arxiv.org/abs/2024.ai.agents

## 2. GPT-4와 Claude 3의 차이점

### 2.1 컨텍스트 윈도우
- GPT-4: 최대 128K 토큰
- Claude 3: 최대 200K 토큰
출처: https://anthropic.com/claude-3

### 2.2 특화 기능
- GPT-4: 코드 생성, 수학적 추론
- Claude 3: 긴 문서 분석, 안전성
출처: https://towardsdatascience.com/gpt4-vs-claude3

## 3. 향후 AI 산업 전망

### 3.1 산업 전반의 AI 도입 가속화
- 2025년까지 AI 시장 규모 $190B 예상
- 기업의 85%가 AI 도입 계획
출처: https://mckinsey.com/ai-outlook-2024

[추가 상세 내용...]
```

**리서치 Agent의 특징:**
- ✅ 여러 출처에서 정보 수집
- ✅ 상세 콘텐츠 추출 및 분석
- ✅ 종합적인 리포트 생성
- ✅ 출처 URL 포함

#### 4.2 SQL 데이터베이스 Agent

데이터베이스를 자연어로 쿼리하는 Agent를 구현합니다.

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_community.utilities import SQLDatabase

# 데이터베이스 연결 (SQLite 예시)
db = SQLDatabase.from_uri("sqlite:///etf_database.db")

# 데이터베이스 도구 정의
@tool
def list_tables() -> str:
    """데이터베이스의 모든 테이블을 나열합니다."""
    tables = db.get_table_names()
    return f"사용 가능한 테이블: {', '.join(tables)}"

@tool
def get_schema(table_name: str) -> str:
    """특정 테이블의 스키마를 가져옵니다."""
    return db.get_table_info([table_name])

@tool
def run_query(query: str) -> str:
    """SQL 쿼리를 실행합니다. SELECT 쿼리만 허용됩니다."""
    try:
        # 보안: SELECT 쿼리만 허용
        if not query.strip().upper().startswith("SELECT"):
            return "오류: SELECT 쿼리만 허용됩니다."

        result = db.run(query)
        return result
    except Exception as e:
        return f"쿼리 실행 오류: {str(e)}"

# SQL Agent 생성
sql_agent = create_agent(
    model=ChatOpenAI(model="gpt-4.1-mini", temperature=0),
    tools=[list_tables, get_schema, run_query],
    system_prompt="""You are a SQL database assistant.

When answering questions:
1. First, use list_tables to see available tables
2. Use get_schema to understand the table structure
3. Write appropriate SQL queries using run_query
4. Explain the results in Korean

Important:
- Always check table schema before querying
- Use proper JOIN operations when needed
- Provide clear explanations of the results"""
)

# SQL Agent 실행
result = sql_agent.invoke(
    {"messages": [{"role": "user", "content": "순자산총액이 가장 큰 ETF 5개를 알려주세요."}]},
    config={"recursion_limit": 15}  # Agent가 충분히 작업할 수 있도록 제한 설정
)

for msg in result["messages"]:
    msg.pretty_print()
```

**실행 흐름:**
```
사용자 질문: "순자산총액이 가장 큰 ETF 5개를 알려주세요."
    ↓
Agent: list_tables() 호출
    ↓
Agent: get_schema("etf_info") 호출
    ↓
Agent: SQL 쿼리 생성
SELECT name, company, 순자산총액
FROM etf_info
ORDER BY 순자산총액 DESC
LIMIT 5
    ↓
Agent: run_query() 호출
    ↓
Agent: 결과 해석 및 답변 생성
```

**예상 출력:**
```
================================ Human Message =================================
순자산총액이 가장 큰 ETF 5개를 알려주세요.

================================== Ai Message ==================================
Tool Calls:
  list_tables (call_123)

================================= Tool Message =================================
사용 가능한 테이블: etf_info

================================== Ai Message ==================================
Tool Calls:
  get_schema (call_124)
 Args:
    table_name: etf_info

================================= Tool Message =================================
CREATE TABLE etf_info (
    name TEXT,
    company TEXT,
    순자산총액 INTEGER,
    ...
)

================================== Ai Message ==================================
Tool Calls:
  run_query (call_125)
 Args:
    query: SELECT name, company, 순자산총액 FROM etf_info ORDER BY 순자산총액 DESC LIMIT 5

================================= Tool Message =================================
[('KODEX 200', 'Samsung', 5000000000000),
 ('TIGER 200', 'Mirae Asset', 3000000000000), ...]

================================== Ai Message ==================================
순자산총액이 가장 큰 ETF 5개는 다음과 같습니다:

1. KODEX 200 (Samsung): 5조원
2. TIGER 200 (Mirae Asset): 3조원
3. KODEX 레버리지 (Samsung): 2.5조원
4. TIGER 차이나전기차 (Mirae Asset): 2조원
5. KODEX 미국S&P500 (Samsung): 1.8조원
```

**SQL Agent의 안전장치:**
- ✅ SELECT 쿼리만 허용 (INSERT, UPDATE, DELETE 차단)
- ✅ 스키마 확인 후 쿼리 생성
- ✅ 에러 처리 및 사용자 친화적 메시지
- ✅ 결과 해석 및 한글 설명

## 🎯 실습 문제

### 실습 1: 뉴스 요약 Agent (⭐⭐⭐)

**문제:**
최신 뉴스를 검색하고 요약하는 Agent를 구현하세요.

**요구사항:**
1. Tavily 검색 도구를 `topic="news"`로 설정
2. 특정 주제에 대한 최신 뉴스 검색
3. 각 뉴스의 핵심 내용을 요약
4. 출처 URL 포함

**힌트:**
- `TavilySearch(topic="news")`
- `search_depth="advanced"` 사용
- 시스템 프롬프트에 요약 지침 포함

### 실습 2: 대화형 학습 도우미 Agent (⭐⭐⭐⭐)

**문제:**
대화 히스토리를 유지하며 학습을 도와주는 Agent를 구현하세요.

**요구사항:**
1. Checkpointing으로 대화 상태 유지
2. 이전 질문과 답변을 기억하여 연관 질문에 답변
3. 웹 검색을 통해 최신 정보 제공
4. 학습 진도를 추적하는 커스텀 Middleware 구현

**힌트:**
- `InMemorySaver()` 또는 `SqliteSaver()` 사용
- 학습 진도 추적을 위한 커스텀 Middleware 작성
- `thread_id`로 각 학습자 구분

### 실습 3: 멀티 에이전트 시스템 (⭐⭐⭐⭐⭐)

**문제:**
서로 다른 역할을 가진 여러 Agent를 조합하여 복잡한 작업을 수행하는 시스템을 구현하세요.

**요구사항:**
1. **리서치 Agent**: 웹 검색 및 정보 수집
2. **분석 Agent**: 수집된 정보 분석
3. **작성 Agent**: 최종 리포트 작성
4. 메인 Agent가 작업을 각 하위 Agent에게 분배

**힌트:**
- 각 Agent를 도구로 래핑: `agent.as_tool()`
- 메인 Agent가 하위 Agent 도구들을 사용
- 순차적 작업 흐름 관리

## ✅ 솔루션 예시

### 실습 1 솔루션: 뉴스 요약 Agent

```python
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI

# 뉴스 검색 도구
news_search = TavilySearch(
    max_results=5,
    topic="news",              # 뉴스 전용 검색
    search_depth="advanced",   # 상세 검색
    days=7                     # 최근 7일 이내 뉴스
)

# 뉴스 요약 Agent
news_agent = create_agent(
    model=ChatOpenAI(model="gpt-4.1-mini", temperature=0.3),
    tools=[news_search],
    system_prompt="""You are a news summarization assistant.

When summarizing news:
1. Search for the latest news on the given topic
2. Summarize each article in 2-3 sentences
3. Include the publication date if available
4. Provide the source URL
5. Present in Korean

Format:
## [뉴스 제목]
- **출처**: [출처명] ([날짜])
- **요약**: [2-3문장 요약]
- **링크**: [URL]
"""
)

# 실행
result = news_agent.invoke({
    "messages": [
        {"role": "user", "content": "인공지능 관련 최신 뉴스를 요약해주세요."}
    ]
})

print(result["messages"][-1].content)
```

**예상 출력:**
```
## 인공지능 최신 뉴스 요약

### 1. OpenAI, GPT-5 개발 중단 발표
- **출처**: TechCrunch (2024-10-25)
- **요약**: OpenAI가 GPT-5 개발을 중단하고 GPT-4의 최적화에 집중한다고 발표했습니다.
  대신 안전성과 신뢰성 향상에 초점을 맞춰 GPT-4의 여러 변형 모델을 출시할 예정입니다.
- **링크**: https://techcrunch.com/...

### 2. Google, Gemini Ultra 일반 공개
- **출처**: The Verge (2024-10-24)
- **요약**: Google이 최신 AI 모델 Gemini Ultra를 일반에 공개했습니다.
  멀티모달 기능이 강화되어 텍스트, 이미지, 동영상을 동시에 처리할 수 있습니다.
- **링크**: https://theverge.com/...

[추가 뉴스...]
```

### 실습 2 솔루션: 대화형 학습 도우미 Agent

```python
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from typing import Any, Dict
import json

# 학습 진도 추적 Middleware
class LearningProgressMiddleware(AgentMiddleware):
    """학습 주제와 질문 수를 추적하는 Middleware"""

    def __init__(self):
        self.topics = {}  # {thread_id: [topics]}
        self.question_counts = {}  # {thread_id: count}

    def after_model(self, state: Dict[str, Any], runtime) -> Dict[str, Any] | None:
        """학습 진도 업데이트"""
        thread_id = runtime.config.get("configurable", {}).get("thread_id")

        if thread_id:
            # 질문 수 증가
            self.question_counts[thread_id] = self.question_counts.get(thread_id, 0) + 1

            # 학습 진도 출력
            if self.question_counts[thread_id] % 3 == 0:
                print(f"\n📊 학습 진도: {self.question_counts[thread_id]}개 질문 완료")

        return None

# 학습 도우미 Agent
learning_middleware = LearningProgressMiddleware()

learning_agent = create_agent(
    model=ChatOpenAI(model="gpt-4.1-mini", temperature=0.5),
    tools=[TavilySearch(max_results=3, topic="general")],
    middleware=[learning_middleware],
    checkpointer=InMemorySaver(),
    system_prompt="""You are a patient and supportive learning assistant.

Your role:
1. Answer questions clearly and concisely
2. Remember previous questions and build upon them
3. Provide examples and analogies
4. Encourage the learner
5. Search for current information when needed

Always be supportive and educational."""
)

# 학습 세션 1
config = {"configurable": {"thread_id": "learner-001"}}

# 첫 번째 질문
result1 = learning_agent.invoke(
    {"messages": [{"role": "user", "content": "파이썬의 리스트란 무엇인가요?"}]},
    config=config
)
print("답변 1:", result1["messages"][-1].content[:200], "...")

# 두 번째 질문 (연관 질문)
result2 = learning_agent.invoke(
    {"messages": [{"role": "user", "content": "그럼 리스트에 요소를 추가하는 방법은?"}]},
    config=config
)
print("\n답변 2:", result2["messages"][-1].content[:200], "...")

# 세 번째 질문 (이전 맥락 활용)
result3 = learning_agent.invoke(
    {"messages": [{"role": "user", "content": "append와 extend의 차이점도 알려주세요."}]},
    config=config
)
print("\n답변 3:", result3["messages"][-1].content[:200], "...")
```

**예상 출력:**
```
답변 1: 파이썬의 리스트는 여러 개의 값을 순서대로 저장할 수 있는 자료구조입니다.
대괄호 []로 표현하며, 다양한 타입의 데이터를 함께 저장할 수 있습니다...

답변 2: 앞서 설명한 리스트에 요소를 추가하는 방법은 여러 가지가 있습니다:
1. append(): 리스트 끝에 하나의 요소를 추가합니다...

📊 학습 진도: 3개 질문 완료

답변 3: append()와 extend()의 차이점을 이전에 배운 리스트 개념과 함께 설명드리겠습니다:
- append(x): 리스트 끝에 x를 하나의 요소로 추가...
```

### 실습 3 솔루션: 멀티 에이전트 시스템

```python
from langchain.agents import create_agent
from langchain_tavily import TavilySearch, TavilyExtract
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

# 1. 리서치 Agent
research_agent = create_agent(
    model=ChatOpenAI(model="gpt-4.1-mini", temperature=0.3),
    tools=[
        TavilySearch(max_results=5, search_depth="advanced"),
        TavilyExtract(extract_depth="advanced")
    ],
    system_prompt="""You are a research specialist.
    Your job is to find comprehensive information on the given topic.
    Use search and extract tools to gather detailed content."""
)

# 리서치 Agent를 도구로 변환
research_tool = research_agent.as_tool(
    name="research_agent",
    description="Conducts in-depth web research on any topic and returns comprehensive information."
)

# 2. 분석 Agent
analysis_agent = create_agent(
    model=ChatOpenAI(model="gpt-4.1-mini", temperature=0.2),
    tools=[],  # 도구 없음 (순수 분석)
    system_prompt="""You are an analytical specialist.
    Your job is to analyze information and extract key insights.
    Identify patterns, trends, and important points."""
)

# 분석 Agent를 도구로 변환
analysis_tool = analysis_agent.as_tool(
    name="analysis_agent",
    description="Analyzes information and extracts key insights and patterns."
)

# 3. 작성 Agent
writing_agent = create_agent(
    model=ChatOpenAI(model="gpt-4.1-mini", temperature=0.7),
    tools=[],
    system_prompt="""You are a professional writer.
    Your job is to write clear, engaging, and well-structured reports.
    Use proper formatting and citations."""
)

# 작성 Agent를 도구로 변환
writing_tool = writing_agent.as_tool(
    name="writing_agent",
    description="Writes professional reports based on analyzed information."
)

# 4. 메인 Orchestrator Agent
orchestrator = create_agent(
    model=ChatOpenAI(model="gpt-4.1-mini", temperature=0.1),
    tools=[research_tool, analysis_tool, writing_tool],
    checkpointer=InMemorySaver(),
    system_prompt="""You are a project orchestrator managing a team of specialists.

Your workflow:
1. Use research_agent to gather information on the topic
2. Use analysis_agent to analyze the research findings
3. Use writing_agent to create a final report

Always follow this sequence for comprehensive results."""
)

# 멀티 에이전트 시스템 실행
result = orchestrator.invoke(
    {"messages": [
        {"role": "user",
         "content": """
         다음 주제에 대한 종합 리포트를 작성해주세요:
         '2024년 전기차 시장 동향과 주요 기업 분석'

         리포트에는 다음이 포함되어야 합니다:
         1. 시장 현황 및 규모
         2. 주요 기업 (Tesla, BYD, Rivian) 분석
         3. 향후 전망
         """}
    ]},
    config={
        "configurable": {"thread_id": "multi-agent-001"},
        "recursion_limit": 20  # 충분한 재귀 제한
    }
)

# 최종 리포트 출력
print("\n" + "="*80)
print("멀티 에이전트 시스템 최종 리포트")
print("="*80)
print(result["messages"][-1].content)
```

**실행 흐름:**
```
사용자 요청: "2024년 전기차 시장 동향과 주요 기업 분석 리포트 작성"
    ↓
Orchestrator: research_agent 호출
    ↓
Research Agent: TavilySearch + TavilyExtract로 정보 수집
    ↓
Orchestrator: analysis_agent 호출
    ↓
Analysis Agent: 수집된 정보 분석 및 인사이트 추출
    ↓
Orchestrator: writing_agent 호출
    ↓
Writing Agent: 전문적인 리포트 작성
    ↓
최종 리포트 반환
```

**예상 출력:**
```
================================================================================
멀티 에이전트 시스템 최종 리포트
================================================================================
# 2024년 전기차 시장 동향과 주요 기업 분석

## 요약
2024년 전기차 시장은 전년 대비 35% 성장하여 1,200만 대 판매를 기록했습니다.
Tesla, BYD, Rivian이 주요 플레이어로 부상하며 시장을 선도하고 있습니다.

## 1. 시장 현황 및 규모

### 1.1 글로벌 시장 규모
- 2024년 글로벌 전기차 판매: 약 1,200만 대
- 전년 대비 성장률: 35%
- 시장 가치: $450 billion
출처: IEA Global EV Outlook 2024

### 1.2 지역별 분포
- 중국: 전체 시장의 60% 점유
- 유럽: 25% 점유
- 북미: 10% 점유
출처: Bloomberg NEF

## 2. 주요 기업 분석

### 2.1 Tesla
**시장 지위**: 글로벌 전기차 시장 점유율 1위 (20%)

**핵심 경쟁력**:
- 자율주행 기술 FSD (Full Self-Driving)
- 슈퍼차저 네트워크
- 브랜드 가치

**2024년 실적**:
- 연간 판매: 240만 대 (전년 대비 15% 증가)
- 주요 모델: Model 3, Model Y

**전망**: 저가형 모델 출시로 시장 확대 예상

### 2.2 BYD
**시장 지위**: 중국 1위, 글로벌 2위 (18%)

**핵심 경쟁력**:
- 배터리 자체 생산 (Blade Battery)
- 가격 경쟁력
- 중국 시장 장악

**2024년 실적**:
- 연간 판매: 216만 대 (전년 대비 50% 증가)
- 주요 모델: Seagull, Dolphin, Han

**전망**: 해외 시장 진출 가속화

### 2.3 Rivian
**시장 지위**: 미국 전기 픽업트럭 시장 선도

**핵심 경쟁력**:
- 오프로드 성능
- 프리미엄 브랜딩
- 아마존 투자 및 파트너십

**2024년 실적**:
- 연간 판매: 5만 대 (스타트업 단계)
- 주요 모델: R1T (픽업), R1S (SUV)

**전망**: 대량 생산 체제 구축 중

## 3. 향후 전망

### 3.1 시장 성장 전망
- 2025년 예상 판매: 1,600만 대 (33% 성장)
- 2030년 전기차 비중: 신차 판매의 40%
출처: McKinsey Electric Vehicle Index

### 3.2 기술 발전 방향
1. **배터리 기술**: 고체 배터리 상용화 (2025-2027)
2. **충전 인프라**: 초고속 충전 (5분 이내) 확대
3. **자율주행**: Level 3-4 상용화

### 3.3 산업 트렌드
- 전통 자동차 제조사의 전기차 전환 가속
- 배터리 공급망 다변화
- 정부 보조금 정책 변화

## 4. 결론

2024년 전기차 시장은 급속한 성장세를 지속하고 있으며, Tesla, BYD, Rivian 등
주요 기업들이 각자의 강점을 바탕으로 시장을 선도하고 있습니다. 향후 배터리 기술
혁신과 충전 인프라 확충이 시장 성장의 핵심 요인이 될 것으로 전망됩니다.

---
*본 리포트는 2024년 10월 기준 최신 정보를 바탕으로 작성되었습니다.*
```

## 🚀 실무 활용 예시

### 예시 1: 고객 지원 챗봇

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langgraph.checkpoint.sqlite import SqliteSaver

# 제품 정보 조회 도구
@tool
def get_product_info(product_id: str) -> str:
    """제품 ID로 제품 정보를 조회합니다."""
    # 실제로는 데이터베이스에서 조회
    products = {
        "P001": "노트북 - 가격: 1,500,000원, 재고: 10개",
        "P002": "마우스 - 가격: 50,000원, 재고: 50개",
        "P003": "키보드 - 가격: 120,000원, 재고: 30개"
    }
    return products.get(product_id, "제품을 찾을 수 없습니다.")

@tool
def check_order_status(order_id: str) -> str:
    """주문 ID로 주문 상태를 확인합니다."""
    # 실제로는 주문 시스템에서 조회
    orders = {
        "O001": "배송 중 (도착 예정: 2024-10-30)",
        "O002": "배송 완료 (2024-10-25 도착)",
        "O003": "주문 확인 중"
    }
    return orders.get(order_id, "주문을 찾을 수 없습니다.")

@tool
def create_support_ticket(
    customer_name: str,
    issue_description: str,
    priority: str
) -> str:
    """고객 지원 티켓을 생성합니다."""
    ticket_id = f"T{hash(customer_name + issue_description) % 10000:04d}"
    return f"지원 티켓이 생성되었습니다. 티켓 번호: {ticket_id}"

# 고객 지원 챗봇
customer_support_agent = create_agent(
    model=ChatOpenAI(model="gpt-4.1-mini", temperature=0.3),
    tools=[
        get_product_info,
        check_order_status,
        create_support_ticket,
        TavilySearch(max_results=3)  # 일반 질문용
    ],
    checkpointer=SqliteSaver.from_conn_string("checkpoints.db"),
    system_prompt="""You are a friendly customer support assistant.

Your capabilities:
1. Check product information and availability
2. Track order status
3. Create support tickets for issues
4. Answer general questions using web search

Always:
- Be polite and helpful
- Ask for clarification if needed
- Provide accurate information
- Create tickets for unresolved issues

Respond in Korean."""
)

# 고객 대화 시뮬레이션
conversations = [
    "안녕하세요! P001 제품 정보를 알려주세요.",
    "O001 주문 상태를 확인하고 싶어요.",
    "제품이 파손되어 도착했어요. 교환하고 싶습니다."
]

config = {"configurable": {"thread_id": "customer-12345"}}

for i, msg in enumerate(conversations, 1):
    print(f"\n{'='*80}")
    print(f"대화 {i}")
    print('='*80)
    print(f"고객: {msg}")

    result = customer_support_agent.invoke(
        {"messages": [{"role": "user", "content": msg}]},
        config=config
    )

    print(f"상담원: {result['messages'][-1].content}")
```

**예상 출력:**
```
================================================================================
대화 1
================================================================================
고객: 안녕하세요! P001 제품 정보를 알려주세요.
상담원: 안녕하세요! P001 제품 정보를 확인해드리겠습니다.

P001 제품은 노트북으로, 가격은 1,500,000원이며 현재 재고가 10개 있습니다.
구매를 원하시면 주문하실 수 있습니다. 다른 궁금한 점이 있으신가요?

================================================================================
대화 2
================================================================================
고객: O001 주문 상태를 확인하고 싶어요.
상담원: O001 주문 상태를 확인해드렸습니다.

현재 배송 중이며, 도착 예정일은 2024년 10월 30일입니다.
배송 관련 추가 문의사항이 있으시면 말씀해주세요!

================================================================================
대화 3
================================================================================
고객: 제품이 파손되어 도착했어요. 교환하고 싶습니다.
상담원: 제품이 파손된 상태로 도착하셨다니 정말 죄송합니다.
바로 지원 티켓을 생성해드리겠습니다.

지원 티켓이 생성되었습니다. 티켓 번호: T3847

담당자가 24시간 이내에 연락드려 교환 절차를 안내해드리겠습니다.
불편을 드려 다시 한번 사과드립니다.
```

### 예시 2: 코드 리뷰 Agent

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
import ast

@tool
def analyze_python_code(code: str) -> str:
    """Python 코드를 분석하여 구조와 복잡도를 평가합니다."""
    try:
        tree = ast.parse(code)

        # 함수 수 계산
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

        # 클래스 수 계산
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

        # 코드 줄 수
        lines = len(code.split('\n'))

        return f"""
코드 분석 결과:
- 총 줄 수: {lines}
- 함수 수: {len(functions)}
- 클래스 수: {len(classes)}
- 함수 이름: {[f.name for f in functions]}
- 클래스 이름: {[c.name for c in classes]}
"""
    except Exception as e:
        return f"코드 분석 오류: {str(e)}"

@tool
def check_code_style(code: str) -> str:
    """코드 스타일 가이드 (PEP 8) 준수 여부를 확인합니다."""
    issues = []

    lines = code.split('\n')

    for i, line in enumerate(lines, 1):
        # 라인 길이 체크
        if len(line) > 79:
            issues.append(f"Line {i}: 라인이 79자를 초과합니다 ({len(line)}자)")

        # 들여쓰기 체크 (4칸 단위)
        indent = len(line) - len(line.lstrip())
        if indent % 4 != 0 and line.strip():
            issues.append(f"Line {i}: 들여쓰기가 4칸 단위가 아닙니다")

    if not issues:
        return "✅ 코드 스타일이 PEP 8 가이드를 준수합니다."
    else:
        return "⚠️ 스타일 이슈:\n" + "\n".join(issues[:10])  # 최대 10개만

@tool
def search_best_practices(topic: str) -> str:
    """특정 주제에 대한 Python 베스트 프랙티스를 검색합니다."""
    # 실제로는 웹 검색이나 지식베이스에서 조회
    practices = {
        "error_handling": "try-except 블록 사용, 구체적인 예외 처리, finally 활용",
        "naming": "소문자와 언더스코어, 의미 있는 이름, 약어 지양",
        "functions": "단일 책임 원칙, 짧은 함수, 명확한 매개변수명"
    }
    return practices.get(topic, "해당 주제의 베스트 프랙티스를 찾을 수 없습니다.")

# 코드 리뷰 Agent
code_review_agent = create_agent(
    model=ChatOpenAI(model="gpt-4.1-mini", temperature=0.2),
    tools=[analyze_python_code, check_code_style, search_best_practices],
    system_prompt="""You are an expert Python code reviewer.

Your review process:
1. Use analyze_python_code to understand code structure
2. Use check_code_style to verify PEP 8 compliance
3. Use search_best_practices for relevant best practices
4. Provide constructive feedback with specific improvements

Review criteria:
- Code readability
- Performance
- Error handling
- Best practices
- Maintainability

Provide feedback in Korean with code examples."""
)

# 코드 리뷰 실행
sample_code = """
def calculate_average(numbers):
  total = 0
  for num in numbers:
    total = total + num
  return total / len(numbers)

class DataProcessor:
  def __init__(self, data):
    self.data = data

  def process(self):
    result = []
    for item in self.data:
      if item > 0:
        result.append(item * 2)
    return result
"""

result = code_review_agent.invoke({
    "messages": [
        {"role": "user",
         "content": f"다음 Python 코드를 리뷰해주세요:\n\n```python\n{sample_code}\n```"}
    ]
})

print(result["messages"][-1].content)
```

**예상 출력:**
```
# 코드 리뷰 결과

## 📊 코드 분석
- 총 줄 수: 15
- 함수 수: 2 (calculate_average, process)
- 클래스 수: 1 (DataProcessor)

## 🎨 스타일 체크
⚠️ 스타일 이슈:
- Line 2: 들여쓰기가 4칸 단위가 아닙니다 (2칸 사용)
- Line 6: 들여쓰기가 4칸 단위가 아닙니다

## 💡 개선 제안

### 1. 들여쓰기 수정
**문제**: PEP 8 표준인 4칸 들여쓰기를 사용하지 않음

**개선 코드**:
```python
def calculate_average(numbers):
    total = 0  # 4칸 들여쓰기
    for num in numbers:
        total = total + num
    return total / len(numbers)
```

### 2. 에러 처리 추가
**문제**: 빈 리스트나 0으로 나누는 경우 처리 없음

**개선 코드**:
```python
def calculate_average(numbers):
    if not numbers:
        raise ValueError("리스트가 비어있습니다")
    return sum(numbers) / len(numbers)
```

### 3. 리스트 컴프리헨션 활용
**문제**: 반복문이 장황함

**개선 코드**:
```python
def process(self):
    return [item * 2 for item in self.data if item > 0]
```

### 4. Type Hints 추가
**베스트 프랙티스**: 함수 시그니처에 타입 힌트 추가

**개선 코드**:
```python
from typing import List

def calculate_average(numbers: List[float]) -> float:
    if not numbers:
        raise ValueError("리스트가 비어있습니다")
    return sum(numbers) / len(numbers)
```

## ✅ 최종 권장 코드
```python
from typing import List

def calculate_average(numbers: List[float]) -> float:
    """숫자 리스트의 평균을 계산합니다."""
    if not numbers:
        raise ValueError("리스트가 비어있습니다")
    return sum(numbers) / len(numbers)

class DataProcessor:
    """데이터 처리를 담당하는 클래스"""

    def __init__(self, data: List[int]) -> None:
        self.data = data

    def process(self) -> List[int]:
        """양수만 필터링하고 2배로 변환합니다."""
        return [item * 2 for item in self.data if item > 0]
```

## 📈 종합 평가
- **가독성**: 7/10 → 8/10 (들여쓰기 수정 후)
- **안정성**: 5/10 → 9/10 (에러 처리 추가 후)
- **효율성**: 7/10 → 9/10 (리스트 컴프리헨션 사용 후)
- **유지보수성**: 6/10 → 9/10 (타입 힌트 및 독스트링 추가 후)
```

## 📖 참고 자료

### 공식 문서
- [LangChain Agents 공식 문서](https://python.langchain.com/docs/modules/agents/)
- [create_agent API 레퍼런스](https://api.python.langchain.com/en/latest/agents/langchain.agents.create_agent.html)
- [LangGraph 문서](https://langchain-ai.github.io/langgraph/)
- [Tavily Search 문서](https://docs.tavily.com/)

### Agent 관련
- [Agent Types 비교](https://python.langchain.com/docs/modules/agents/agent_types/)
- [Custom Agent 구현](https://python.langchain.com/docs/modules/agents/how_to/custom_agent)
- [Agent Middleware](https://python.langchain.com/docs/modules/agents/middleware/)

### Checkpointing
- [LangGraph Checkpointing](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [SqliteSaver 사용법](https://langchain-ai.github.io/langgraph/reference/checkpoints/)

### 추가 학습 자료
- [Building Agentic Systems](https://www.deeplearning.ai/courses/building-agentic-systems/)
- [LangChain Agent 튜토리얼](https://python.langchain.com/docs/tutorials/agents/)
- [Multi-Agent Systems](https://www.langchain.com/multi-agent-systems)
