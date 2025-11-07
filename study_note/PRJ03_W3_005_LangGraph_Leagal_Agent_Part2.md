# 법률 문서 기반 검색 에이전트 Part2 - RAG 에이전트 및 통합 시스템

## 📚 학습 목표

이 학습 가이드를 완료하면 다음을 할 수 있습니다:

1. **Corrective RAG 패턴**을 활용하여 법률별 특화 에이전트를 구현할 수 있습니다
2. **Adaptive RAG 패턴**을 적용하여 질문 라우팅 시스템을 구축할 수 있습니다
3. **ReAct 에이전트**를 사용하여 답변 품질을 평가할 수 있습니다
4. **HITL (Human-in-the-Loop)**를 구현하여 사용자 피드백을 통합할 수 있습니다
5. **Gradio**를 활용하여 법률 상담 챗봇 UI를 구축할 수 있습니다
6. **LangGraph StateGraph**로 복잡한 에이전트 워크플로우를 설계할 수 있습니다
7. 실전 법률 상담 시스템을 처음부터 끝까지 구축할 수 있습니다

## 🔑 핵심 개념

### Corrective RAG vs Adaptive RAG

| 특징 | Corrective RAG | Adaptive RAG |
|------|----------------|--------------|
| **목적** | 검색 품질 개선 | 전략 선택 |
| **적용 단계** | 검색 후 문서 평가 | 검색 전 전략 결정 |
| **주요 동작** | 문서 재평가, 질문 재작성 | 질문 분석, 에이전트 라우팅 |
| **Part1 CRAG** | 문서 평가 → 재검색 | 질문 → 적절한 에이전트 선택 |

### 시스템 아키텍처

```
사용자 질문
   ↓
[질문 라우팅 에이전트] ← Adaptive RAG
   ↓
   ├─ 개인정보보호법 에이전트 ← Corrective RAG
   ├─ 근로기준법 에이전트 ← Corrective RAG
   └─ 주택임대차보호법 에이전트 ← Corrective RAG
   ↓
[답변 평가 에이전트] ← ReAct
   ↓
[HITL 확인] ← Human-in-the-Loop
   ↓
[Gradio 챗봇 UI]
   ↓
최종 답변
```

### Corrective RAG 워크플로우

```
질문 입력
   ↓
[문서 검색]
   ↓
[문서 관련성 평가] ← 3단계: relevant / not_relevant / ambiguous
   ↓
관련 문서 충분? ─ Yes → [답변 생성]
   ↓ No
[질문 재작성]
   ↓
[웹 검색]
   ↓
[답변 생성]
```

### ReAct 패턴

**ReAct (Reasoning + Acting)**는 추론과 행동을 결합한 에이전트 패턴입니다.

```
Thought (생각): "답변이 법률 조항을 명확히 인용하지 않았다"
   ↓
Action (행동): 답변 평가 도구 호출
   ↓
Observation (관찰): 평가 결과 확인
   ↓
Thought: "재생성이 필요하다"
   ↓
Final Answer: 평가 결과 반환
```

## 💻 단계별 구현

### 단계 1: Corrective RAG 에이전트 구현

각 법률 도메인에 특화된 Corrective RAG 에이전트를 구축합니다.

#### 1.1 State 정의

```python
from pydantic import BaseModel, Field
from typing import List, TypedDict, Optional, Literal
from langchain_core.documents import Document

class CorrectiveRagState(TypedDict):
    """Corrective RAG 기본 상태"""
    question: str                      # 원본 질문
    documents: List[Document]          # 검색된 문서
    relevance: str                     # 문서 관련성
    answer: str                        # 생성된 답변

class PersonalRagState(CorrectiveRagState):
    """개인정보보호법 RAG 상태"""
    rewritten_query: Optional[str]     # 재작성된 질문
```

#### 1.2 문서 관련성 평가

```python
from langchain_core.prompts import ChatPromptTemplate

class DocumentRelevance(BaseModel):
    """문서 관련성 평가 결과"""
    relevance: Literal["relevant", "not_relevant", "ambiguous"] = Field(
        description="문서가 질문과 관련이 있는지 평가"
    )
    reason: str = Field(description="평가 근거")

# LLM 초기화
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
structured_llm = llm.with_structured_output(DocumentRelevance)

# 평가 프롬프트
relevance_prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 문서 관련성 평가 전문가입니다.

    질문과 검색된 문서를 분석하여 관련성을 평가하세요.

    평가 기준:
    - relevant: 질문에 직접 답할 수 있는 정보 포함
    - not_relevant: 질문과 무관한 내용
    - ambiguous: 부분적으로 관련되어 있으나 불충분
    """),
    ("human", "질문: {question}\n\n문서:\n{documents}")
])

relevance_chain = relevance_prompt | structured_llm
```

#### 1.3 Corrective RAG 노드 함수

```python
from langgraph.graph import StateGraph, START, END

def retrieve_documents(state: PersonalRagState) -> PersonalRagState:
    """문서 검색"""
    question = state["question"]

    # 벡터 저장소에서 검색
    docs = personal_vectorstore.similarity_search(question, k=3)

    return {"documents": docs}

def evaluate_relevance(state: PersonalRagState) -> PersonalRagState:
    """문서 관련성 평가"""
    question = state["question"]
    docs = state["documents"]

    # 문서 내용 결합
    doc_content = "\n\n".join([doc.page_content for doc in docs])

    # 관련성 평가
    result = relevance_chain.invoke({
        "question": question,
        "documents": doc_content
    })

    return {"relevance": result.relevance}

def rewrite_query(state: PersonalRagState) -> PersonalRagState:
    """질문 재작성"""
    question = state["question"]

    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", "검색 결과가 불충분합니다. 질문을 더 구체적으로 재작성하세요."),
        ("human", "{question}")
    ])

    chain = rewrite_prompt | llm | StrOutputParser()
    rewritten = chain.invoke({"question": question})

    return {"rewritten_query": rewritten}

def web_search_docs(state: PersonalRagState) -> PersonalRagState:
    """웹 검색"""
    query = state.get("rewritten_query", state["question"])

    # Tavily 검색
    from langchain_community.tools.tavily_search import TavilySearchResults
    web_search = TavilySearchResults(max_results=3)

    results = web_search.invoke(query)

    # Document로 변환
    docs = [Document(page_content=str(result)) for result in results]

    return {"documents": docs}

def generate_answer(state: PersonalRagState) -> PersonalRagState:
    """답변 생성"""
    question = state["question"]
    docs = state["documents"]

    # RAG 프롬프트
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 개인정보보호법 전문가입니다.
        제공된 문서를 바탕으로 정확하고 상세한 답변을 제공하세요.
        """),
        ("human", "질문: {question}\n\n문서:\n{context}")
    ])

    context = "\n\n".join([doc.page_content for doc in docs])
    chain = rag_prompt | llm | StrOutputParser()

    answer = chain.invoke({"question": question, "context": context})

    return {"answer": answer}
```

#### 1.4 조건부 라우팅

```python
def decide_next_step(state: PersonalRagState) -> str:
    """다음 단계 결정"""
    relevance = state.get("relevance", "")

    if relevance == "relevant":
        return "generate"
    elif relevance == "not_relevant":
        return "rewrite"
    else:  # ambiguous
        return "generate"  # 일단 생성 시도
```

#### 1.5 그래프 구축

```python
from langgraph.graph import StateGraph, START, END

# 개인정보보호법 RAG 그래프
personal_graph = StateGraph(PersonalRagState)

# 노드 추가
personal_graph.add_node("retrieve", retrieve_documents)
personal_graph.add_node("evaluate", evaluate_relevance)
personal_graph.add_node("rewrite", rewrite_query)
personal_graph.add_node("web_search", web_search_docs)
personal_graph.add_node("generate", generate_answer)

# 엣지 연결
personal_graph.add_edge(START, "retrieve")
personal_graph.add_edge("retrieve", "evaluate")

# 조건부 엣지
personal_graph.add_conditional_edges(
    "evaluate",
    decide_next_step,
    {
        "generate": "generate",
        "rewrite": "rewrite"
    }
)

personal_graph.add_edge("rewrite", "web_search")
personal_graph.add_edge("web_search", "generate")
personal_graph.add_edge("generate", END)

# 컴파일
personal_agent = personal_graph.compile()

print("✅ 개인정보보호법 RAG 에이전트 생성 완료")
```

#### 1.6 에이전트 테스트

```python
# 테스트 실행
inputs = {"question": "개인정보 처리에 대한 동의를 받을 때 주의해야 할 점은 무엇인가요?"}

for output in personal_agent.stream(inputs):
    for key, value in output.items():
        print(f"\n[{key}]")
        if "answer" in value:
            from IPython.display import Markdown, display
            display(Markdown(value["answer"]))
```

**실행 결과 예시**:
```
[retrieve]
검색된 문서: 3개

[evaluate]
관련성 평가: relevant

[generate]
개인정보 처리에 대한 동의를 받을 때는 다음 사항을 주의해야 합니다:

1. **명시적 동의**: 정보주체의 명확한 의사표시를 받아야 합니다.
2. **동의 범위**: 개인정보의 수집·이용 목적, 수집하는 개인정보의 항목,
   개인정보의 보유 및 이용 기간을 명확히 고지해야 합니다.
3. **선택적 동의**: 필수 항목과 선택 항목을 구분하여 동의를 받아야 합니다.
...
```

### 단계 2: 질문 라우팅 에이전트 (Adaptive RAG)

사용자 질문을 분석하여 적절한 법률 에이전트로 라우팅합니다.

#### 2.1 라우팅 State 정의

```python
class QueryRouter(BaseModel):
    """질문 라우팅 결과"""
    law_type: Literal["personal_info", "labor", "housing", "general"] = Field(
        description="""질문이 관련된 법률 분야:
        - personal_info: 개인정보보호법
        - labor: 근로기준법
        - housing: 주택임대차보호법
        - general: 법률과 무관하거나 일반 질문
        """
    )
    confidence: float = Field(ge=0, le=1, description="라우팅 결정 신뢰도")
    reason: str = Field(description="라우팅 이유")

class RouterState(TypedDict):
    """라우터 상태"""
    question: str
    law_type: str
    answer: str
```

#### 2.2 질문 분석 및 라우팅

```python
# 라우팅 LLM
router_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
structured_router = router_llm.with_structured_output(QueryRouter)

# 라우팅 프롬프트
router_prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 법률 질문 분류 전문가입니다.

    사용자의 질문을 분석하여 관련 법률 분야를 판단하세요.

    분류 기준:
    - personal_info: 개인정보 보호, 수집, 이용, 정보주체 권리
    - labor: 근로 조건, 임금, 휴가, 해고, 퇴직금
    - housing: 임대차 계약, 전월세, 보증금, 임차인 권리
    - general: 위 분야와 무관하거나 일반적인 질문
    """),
    ("human", "{question}")
])

router_chain = router_prompt | structured_router
```

#### 2.3 라우팅 그래프 구축

```python
def route_question(state: RouterState) -> RouterState:
    """질문 라우팅"""
    question = state["question"]

    # 라우팅 결정
    result = router_chain.invoke({"question": question})

    return {"law_type": result.law_type}

def call_personal_agent(state: RouterState) -> RouterState:
    """개인정보보호법 에이전트 호출"""
    question = state["question"]

    # 에이전트 실행
    result = personal_agent.invoke({"question": question})

    return {"answer": result["answer"]}

def call_labor_agent(state: RouterState) -> RouterState:
    """근로기준법 에이전트 호출"""
    question = state["question"]

    # 에이전트 실행
    result = labor_agent.invoke({"question": question})

    return {"answer": result["answer"]}

def call_housing_agent(state: RouterState) -> RouterState:
    """주택임대차보호법 에이전트 호출"""
    question = state["question"]

    # 에이전트 실행
    result = housing_agent.invoke({"question": question})

    return {"answer": result["answer"]}

def handle_general_query(state: RouterState) -> RouterState:
    """일반 질문 처리"""
    question = state["question"]

    # 직접 LLM 답변
    answer = llm.invoke(question).content

    return {"answer": answer}

# 라우팅 그래프
router_graph = StateGraph(RouterState)

# 노드 추가
router_graph.add_node("route", route_question)
router_graph.add_node("personal", call_personal_agent)
router_graph.add_node("labor", call_labor_agent)
router_graph.add_node("housing", call_housing_agent)
router_graph.add_node("general", handle_general_query)

# 엣지 연결
router_graph.add_edge(START, "route")
router_graph.add_conditional_edges(
    "route",
    lambda state: state["law_type"],
    {
        "personal_info": "personal",
        "labor": "labor",
        "housing": "housing",
        "general": "general"
    }
)

# 모든 에이전트에서 END로
router_graph.add_edge("personal", END)
router_graph.add_edge("labor", END)
router_graph.add_edge("housing", END)
router_graph.add_edge("general", END)

# 컴파일
legal_assistant = router_graph.compile()

print("✅ 법률 상담 에이전트 생성 완료")
```

#### 2.4 통합 에이전트 테스트

```python
# 다양한 질문 테스트
test_queries = [
    "개인정보 유출 사고 시 대응 절차는?",
    "연차휴가를 사용하지 못한 경우 보상은?",
    "전세 계약 갱신 시 주의사항은?",
    "오늘 날씨는 어때요?"
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"질문: {query}")
    print('='*60)

    result = legal_assistant.invoke({"question": query})

    print(f"\n라우팅: {result['law_type']}")
    print(f"\n답변:\n{result['answer']}")
```

### 단계 3: 답변 평가 에이전트 (ReAct)

생성된 답변의 품질을 평가합니다.

#### 3.1 평가 도구 정의

```python
from langchain.tools import Tool

def evaluate_answer_quality(query: str, answer: str) -> str:
    """답변 품질 평가"""

    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 법률 답변 품질 평가 전문가입니다.

        다음 기준으로 답변을 평가하세요:
        1. 정확성: 법률 조항을 정확히 인용했는가?
        2. 완전성: 질문에 충분히 답변했는가?
        3. 명확성: 이해하기 쉽게 설명했는가?
        4. 근거: 법률적 근거를 제시했는가?

        평가 결과를 다음 형식으로 제공하세요:
        - 점수: X/10
        - 강점: ...
        - 약점: ...
        - 개선사항: ...
        """),
        ("human", "질문: {query}\n\n답변: {answer}")
    ])

    chain = eval_prompt | llm | StrOutputParser()
    evaluation = chain.invoke({"query": query, "answer": answer})

    return evaluation

# 도구로 등록
eval_tool = Tool(
    name="evaluate_answer",
    description="법률 답변의 품질을 평가합니다.",
    func=lambda x: evaluate_answer_quality(x["query"], x["answer"])
)
```

#### 3.2 ReAct 에이전트 구축

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate

# ReAct 프롬프트
react_prompt = PromptTemplate.from_template("""
당신은 법률 답변 평가 전문가입니다.

사용 가능한 도구:
{tools}

도구 이름: {tool_names}

질문: {input}

생각 과정:
{agent_scratchpad}
""")

# ReAct 에이전트
react_agent = create_react_agent(
    llm=llm,
    tools=[eval_tool],
    prompt=react_prompt
)

agent_executor = AgentExecutor(
    agent=react_agent,
    tools=[eval_tool],
    verbose=True,
    max_iterations=3
)

print("✅ ReAct 평가 에이전트 생성 완료")
```

#### 3.3 평가 실행

```python
# 답변 생성
question = "연차휴가 부여 기준은?"
result = legal_assistant.invoke({"question": question})
answer = result["answer"]

# 답변 평가
evaluation_input = {
    "input": f"다음 법률 답변을 평가하세요.\n\n질문: {question}\n\n답변: {answer}"
}

evaluation = agent_executor.invoke(evaluation_input)

print("\n📊 평가 결과:")
print(evaluation["output"])
```

### 단계 4: HITL (Human-in-the-Loop) 구현

사용자 피드백을 수집하고 반영합니다.

#### 4.1 HITL State 정의

```python
class HITLState(TypedDict):
    """HITL 상태"""
    question: str
    answer: str
    evaluation: str
    user_feedback: Optional[str]
    final_answer: str
```

#### 4.2 HITL 노드

```python
from langgraph.types import interrupt

def get_user_feedback(state: HITLState) -> HITLState:
    """사용자 피드백 요청"""

    # 중단점 생성 (사용자 입력 대기)
    feedback = interrupt({
        "question": state["question"],
        "answer": state["answer"],
        "evaluation": state["evaluation"],
        "prompt": "답변이 만족스러우신가요? (yes/no 또는 피드백 입력)"
    })

    return {"user_feedback": feedback}

def process_feedback(state: HITLState) -> HITLState:
    """피드백 처리"""
    feedback = state["user_feedback"]
    answer = state["answer"]

    if feedback and feedback.lower() not in ["yes", "y"]:
        # 피드백을 반영하여 답변 재생성
        refine_prompt = ChatPromptTemplate.from_messages([
            ("system", "사용자 피드백을 반영하여 답변을 개선하세요."),
            ("human", "원래 답변: {answer}\n\n피드백: {feedback}\n\n개선된 답변:")
        ])

        chain = refine_prompt | llm | StrOutputParser()
        improved = chain.invoke({"answer": answer, "feedback": feedback})

        return {"final_answer": improved}
    else:
        return {"final_answer": answer}
```

#### 4.3 HITL 그래프

```python
hitl_graph = StateGraph(HITLState)

hitl_graph.add_node("feedback", get_user_feedback)
hitl_graph.add_node("process", process_feedback)

hitl_graph.add_edge(START, "feedback")
hitl_graph.add_edge("feedback", "process")
hitl_graph.add_edge("process", END)

hitl_agent = hitl_graph.compile()
```

### 단계 5: Gradio 챗봇 UI

사용자 친화적인 웹 인터페이스를 구축합니다.

#### 5.1 Gradio 챗봇 구현

```python
import gradio as gr
from typing import List, Tuple

class LegalChatbot:
    """법률 상담 챗봇"""

    def __init__(self, assistant):
        self.assistant = assistant
        self.history = []

    def chat(self, message: str, history: List[Tuple[str, str]]) -> str:
        """채팅 처리"""

        # 에이전트 호출
        result = self.assistant.invoke({"question": message})
        answer = result["answer"]

        # 이력 저장
        self.history.append((message, answer))

        return answer

# 챗봇 인스턴스
chatbot = LegalChatbot(legal_assistant)

# Gradio 인터페이스
demo = gr.ChatInterface(
    fn=chatbot.chat,
    title="🏛️ 법률 상담 AI 챗봇",
    description="""
    법률 관련 질문을 입력하세요.
    개인정보보호법, 근로기준법, 주택임대차보호법에 대해 답변합니다.
    """,
    examples=[
        "개인정보 수집 시 동의를 받아야 하는 경우는?",
        "연차휴가 부여 기준에 대해 설명해주세요.",
        "전세 계약 시 임차인 권리는 무엇인가요?"
    ],
    theme=gr.themes.Soft(),
    retry_btn="🔄 다시 시도",
    undo_btn="↩️ 실행 취소",
    clear_btn="🗑️ 대화 초기화"
)

# 실행
if __name__ == "__main__":
    demo.launch(share=True)
```

## 🎯 실습 문제

### 실습: 완전한 법률 상담 시스템 구축

**과제**: Part 1과 Part 2의 모든 개념을 통합하여 완전한 법률 상담 시스템을 구축하세요.

**요구사항**:

1. **벡터 저장소 구축** (Part 1)
   - 3개 법률 PDF 로드 및 벡터 저장소 생성

2. **Corrective RAG 에이전트** (Part 2)
   - 각 법률에 대한 개별 에이전트 구현
   - 문서 관련성 평가 및 질문 재작성

3. **Adaptive RAG 라우팅** (Part 2)
   - 질문 분석 및 에이전트 자동 선택

4. **답변 품질 관리** (Part 2)
   - ReAct 평가 에이전트
   - HITL 피드백 수집

5. **Gradio UI** (Part 2)
   - 사용자 친화적인 챗봇 인터페이스

## ✅ 솔루션 예시

솔루션은 노트북의 실습 셀을 참고하세요. 주요 포인트:

```python
# 1. 모든 에이전트 구축
personal_agent = create_corrective_rag_agent(personal_vectorstore)
labor_agent = create_corrective_rag_agent(labor_vectorstore)
housing_agent = create_corrective_rag_agent(housing_vectorstore)

# 2. 라우팅 통합
legal_assistant = create_routing_system([
    personal_agent,
    labor_agent,
    housing_agent
])

# 3. 평가 레이어 추가
evaluated_assistant = add_evaluation_layer(legal_assistant)

# 4. HITL 통합
hitl_assistant = add_hitl(evaluated_assistant)

# 5. Gradio 배포
demo = create_gradio_interface(hitl_assistant)
demo.launch()
```

## 📖 참고 자료

### 공식 문서
- [LangGraph 문서](https://langchain-ai.github.io/langgraph/)
- [ReAct 논문](https://arxiv.org/abs/2210.03629)
- [Gradio 문서](https://www.gradio.app/docs/)

### 관련 논문
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884)

---

**완료**: Part 1과 Part 2를 통해 완전한 법률 문서 기반 검색 에이전트 시스템을 구축했습니다. 이제 실제 법률 상담 서비스에 적용할 수 있는 프로덕션급 시스템을 구현할 수 있습니다.
