# W3_004_Chat_History.md

## 학습 목표
- LangChain에서 채팅 히스토리 관리 방법 이해하기
- 메시지 전달 방식과 RunnableWithMessageHistory 구현하기
- 다양한 히스토리 저장 방식 (인메모리, SQLite, 요약 저장) 학습하기
- 메시지 트리밍과 대화 요약 기법 실습하기
- 실제 챗봇에서 활용할 수 있는 효율적인 메모리 관리 시스템 구축하기

## 주요 개념

### 1. 채팅 히스토리 관리의 중요성
채팅 히스토리 관리는 대화형 AI의 핵심 구성 요소로서, 사용자와의 이전 대화 맥락을 유지하여 연속성 있는 대화를 가능하게 합니다. 이를 통해 AI는 이전 대화를 참조하여 더 정확하고 개인화된 응답을 제공할 수 있습니다.

### 2. LangChain 메모리 관리 방식
LangChain에서는 크게 두 가지 메모리 관리 방식을 제공합니다:
- **메시지 전달 방식**: 이전 대화 기록을 체인에 직접 전달하는 기본적인 방법
- **RunnableWithMessageHistory**: 대화 기록을 자동으로 저장하고 검색하는 고급 방법

### 3. 대화 기록 저장소
- **인메모리 저장**: 실행 중에만 유지되는 빠른 저장 방식
- **SQLite 저장**: 데이터베이스를 활용한 영구 저장 방식
- **요약 저장**: 대화를 요약하여 컨텍스트 길이를 관리하는 방식

### 4. 메모리 최적화 기법
- **메시지 트리밍**: 토큰 제한을 관리하기 위한 메시지 수 조정
- **대화 요약**: 긴 대화 내용을 핵심으로 압축하는 기법
- **트리밍 + 요약**: 두 기법을 결합한 효율적인 메모리 관리

## 환경 설정

### 필수 라이브러리 설치
```bash
pip install langchain langchain-openai langchain-community python-dotenv pydantic
```

### 환경 변수 설정
```python
from dotenv import load_dotenv
import os

load_dotenv()

# OpenAI API 키 설정
openai_api_key = os.getenv("OPENAI_API_KEY")
```

## 핵심 구현

### 1. 기본 메시지 전달 방식

메시지 전달 방식은 가장 기본적인 채팅 히스토리 관리 방법입니다:

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# LLM 초기화
llm = ChatOpenAI(model='gpt-4.1-mini', temperature=0.5)

# 프롬프트 템플릿 구성 (MessagesPlaceholder를 사용한 이전 메시지 포함)
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절한 여행 가이드입니다. 사용자의 여행 계획을 도와주세요."),
    MessagesPlaceholder(variable_name="messages"),  # 이전 대화 기록을 여기에 삽입
    ("human", "{input}")
])

# 체인 구성
chain = prompt | llm

# 대화 기록 저장을 위한 리스트
messages = []

# 첫 번째 대화
response = chain.invoke({
    "messages": messages,
    "input": "안녕하세요, 저는 김민수입니다."
})

# AI 응답을 대화 기록에 추가
messages.extend([
    HumanMessage(content="안녕하세요, 저는 김민수입니다."),
    AIMessage(content=response.content)
])

# 두 번째 대화 - 이름을 기억하는지 확인
response = chain.invoke({
    "messages": messages + [HumanMessage(content="제 이름을 기억하나요?")]
})
```

### 2. RunnableWithMessageHistory 구현

더 고급 방식인 RunnableWithMessageHistory를 사용한 자동 메모리 관리:

```python
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from pydantic import BaseModel, Field
from typing import List

# 메모리 기반 히스토리 구현
class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

# 세션별 히스토리 저장소
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]

# 프롬프트 템플릿 구성
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절한 여행 가이드입니다."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm

# 히스토리 관리 추가
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 세션 ID를 사용한 대화
response = chain_with_history.invoke(
    {"input": "제주도 여행 계획을 도와주세요."},
    config={"configurable": {"session_id": "tourist_1"}}
)
```

### 3. SQLite 기반 영구 저장

SQLite 데이터베이스를 활용한 영구 저장 방식:

```python
import sqlite3
import json
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

class SQLiteChatMessageHistory(BaseChatMessageHistory):
    """SQLite 데이터베이스를 사용하는 채팅 메시지 히스토리"""

    def __init__(self, session_id: str, db_path: str = "chat_history.db"):
        self.session_id = session_id
        self.db_path = db_path
        self._create_tables()

    def _create_tables(self):
        """데이터베이스 테이블 생성"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                message_type TEXT,
                content TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def add_message(self, message: BaseMessage) -> None:
        """단일 메시지 추가"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        message_type = type(message).__name__
        content = message.content

        cursor.execute(
            "INSERT INTO chat_messages (session_id, message_type, content) VALUES (?, ?, ?)",
            (self.session_id, message_type, content)
        )
        conn.commit()
        conn.close()

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """여러 메시지 추가"""
        for message in messages:
            self.add_message(message)

    def clear(self) -> None:
        """세션의 모든 메시지 삭제"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_messages WHERE session_id = ?", (self.session_id,))
        conn.commit()
        conn.close()

    @property
    def messages(self) -> List[BaseMessage]:
        """저장된 메시지 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT message_type, content FROM chat_messages WHERE session_id = ? ORDER BY timestamp",
            (self.session_id,)
        )

        messages = []
        for message_type, content in cursor.fetchall():
            if message_type == "HumanMessage":
                messages.append(HumanMessage(content=content))
            elif message_type == "AIMessage":
                messages.append(AIMessage(content=content))

        conn.close()
        return messages

# SQLite 히스토리 사용
def get_chat_history(session_id: str) -> BaseChatMessageHistory:
    return SQLiteChatMessageHistory(session_id=session_id)
```

### 4. 메시지 트리밍 시스템

토큰 제한을 관리하기 위한 메시지 트리밍 구현:

```python
from langchain_core.messages import trim_messages

class TrimmedInMemoryHistory(BaseChatMessageHistory, BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)
    max_tokens: int = Field(default=2)

    def __init__(self, max_tokens: int = 2, **kwargs):
        """
        max_tokens: 유지할 최대 토큰 수 (실제로는 메시지 쌍 수)
        """
        super().__init__(max_tokens=max_tokens, **kwargs)
        self.max_tokens = max_tokens

    def add_messages(self, messages: List[BaseMessage]) -> None:
        self.messages.extend(messages)
        # 메시지 추가 후 자동으로 트리밍 수행
        self.messages = trim_messages(
            self.messages,
            max_tokens=self.max_tokens,
            strategy="last",
            token_counter=len,  # 실제 토큰 수 대신 메시지 수 사용
            include_system=True,
            start_on="human"
        )

    def clear(self) -> None:
        self.messages = []

# 트리밍된 히스토리 사용
def get_trimmed_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = TrimmedInMemoryHistory(max_tokens=2)
    return store[session_id]
```

### 5. 대화 요약 저장 시스템

긴 대화를 요약하여 컨텍스트를 관리하는 시스템:

```python
from langchain_core.messages import HumanMessage, SystemMessage

class SummarizedInMemoryHistory(BaseChatMessageHistory, BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)
    summary_threshold: int = Field(default=6)  # 요약을 시작할 메시지 수
    llm: ChatOpenAI = Field(default_factory=lambda: ChatOpenAI(model="gpt-4.1-mini", temperature=0.1))

    def add_messages(self, new_messages: List[BaseMessage]) -> None:
        self.messages.extend(new_messages)

        # 메시지 수가 임계값을 초과하면 요약 수행
        if len(self.messages) > self.summary_threshold:
            summary = self._summarize_conversation()

            # 요약으로 대화 내용을 압축
            system_summary = SystemMessage(content=f"이전 대화 요약: {summary}")
            # 최근 2개 메시지는 유지
            recent_messages = self.messages[-2:]
            self.messages = [system_summary] + recent_messages

    def _summarize_conversation(self) -> str:
        """대화 내용을 요약"""
        conversation_text = ""
        for msg in self.messages:
            if isinstance(msg, HumanMessage):
                conversation_text += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                conversation_text += f"Assistant: {msg.content}\n"

        summary_prompt = f"""
        다음 대화를 3-4줄로 요약해주세요. 주요 주제와 핵심 정보를 포함하세요:

        {conversation_text}
        """

        summary_response = self.llm.invoke([HumanMessage(content=summary_prompt)])
        return summary_response.content

    def clear(self) -> None:
        self.messages = []

# 요약 히스토리 사용
def get_summarized_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = SummarizedInMemoryHistory()
    return store[session_id]
```

### 6. 트리밍 + 요약 결합 시스템

메시지 트리밍과 대화 요약을 결합한 최적화된 시스템:

```python
class TrimmedAndSummarizedHistory(BaseChatMessageHistory, BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)
    max_tokens: int = Field(default=4)  # 트리밍 기준
    summary_threshold: int = Field(default=6)  # 요약 기준
    llm: ChatOpenAI = Field(default_factory=lambda: ChatOpenAI(model="gpt-4.1-mini", temperature=0.1))
    summarized_messages: List[BaseMessage] = Field(default_factory=list)  # 요약된 메시지 저장

    def add_messages(self, new_messages: List[BaseMessage]) -> None:
        # 새 메시지 추가
        self.messages.extend(new_messages)

        # 메시지 수가 요약 임계값을 초과하면 요약 수행
        if len(self.messages) >= self.summary_threshold:
            summary = self._summarize_conversation()
            summary_message = SystemMessage(content=f"이전 대화 요약: {summary}")

            # 요약된 메시지 저장
            self.summarized_messages.append(summary_message)

            # 최근 메시지만 유지 (트리밍 효과)
            self.messages = self.messages[-2:]

        # 트리밍 적용 (토큰 제한 관리)
        all_messages = self.summarized_messages + self.messages
        if len(all_messages) > self.max_tokens:
            # 요약 메시지는 유지하고 최근 대화만 트리밍
            trimmed_messages = trim_messages(
                all_messages,
                max_tokens=self.max_tokens,
                strategy="last",
                token_counter=len,
                include_system=True,
                start_on="human"
            )

            # 요약 메시지와 일반 메시지 분리
            self.summarized_messages = [msg for msg in trimmed_messages if isinstance(msg, SystemMessage)]
            self.messages = [msg for msg in trimmed_messages if not isinstance(msg, SystemMessage)]

    def _summarize_conversation(self) -> str:
        """현재 대화 내용을 요약"""
        conversation_text = ""
        for msg in self.messages:
            if isinstance(msg, HumanMessage):
                conversation_text += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                conversation_text += f"Assistant: {msg.content}\n"

        summary_prompt = f"""
        다음 대화를 간결하게 요약해주세요 (2-3줄):

        {conversation_text}
        """

        summary_response = self.llm.invoke([HumanMessage(content=summary_prompt)])
        return summary_response.content

    def clear(self) -> None:
        self.messages = []
        self.summarized_messages = []

    def get_all_messages(self) -> List[BaseMessage]:
        # 요약된 메시지와 현재 메시지를 결합하여 반환
        return self.summarized_messages + self.messages

# 트리밍 + 요약 히스토리 사용
def get_trimmed_summarized_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = TrimmedAndSummarizedHistory()
    return store[session_id]
```

## 실습 문제

### 실습 1: 개인화된 쇼핑 어시스턴트 구현
사용자의 구매 이력과 선호도를 기억하는 쇼핑 어시스턴트를 구현하세요.

**요구사항:**
1. 사용자의 이름, 선호 브랜드, 예산 범위를 기억
2. 이전 구매 상품을 바탕으로 추천 제공
3. SQLite를 사용하여 대화 내용 영구 저장
4. 메시지 트리밍으로 성능 최적화

### 실습 2: 의료 상담 챗봇 메모리 시스템
환자의 증상과 상담 이력을 관리하는 의료 상담 챗봇을 구현하세요.

**요구사항:**
1. 환자별 증상 기록과 상담 이력 관리
2. 민감한 의료 정보는 요약하지 않고 원본 유지
3. 세션별 독립적인 메모리 관리
4. 트리밍 임계값을 높게 설정하여 충분한 컨텍스트 유지

### 실습 3: 멀티 언어 학습 튜터 시스템
여러 언어를 동시에 학습할 수 있는 언어 튜터 챗봇을 구현하세요.

**요구사항:**
1. 언어별 학습 진도와 취약 부분 기록
2. 대화 요약 시 언어별로 분리하여 관리
3. 학습자의 수준에 따른 맞춤형 메모리 관리
4. 트리밍과 요약을 결합한 효율적인 시스템

## 실습 해답

### 실습 1 해답: 개인화된 쇼핑 어시스턴트

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
import sqlite3
import json
from typing import Dict, Any

class ShoppingAssistantHistory(SQLiteChatMessageHistory):
    """쇼핑 어시스턴트용 확장된 SQLite 히스토리"""

    def __init__(self, session_id: str, db_path: str = "shopping_history.db"):
        super().__init__(session_id, db_path)
        self._create_user_profile_table()

    def _create_user_profile_table(self):
        """사용자 프로필 테이블 생성"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profiles (
                session_id TEXT PRIMARY KEY,
                name TEXT,
                preferred_brands TEXT,
                budget_range TEXT,
                purchase_history TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def update_user_profile(self, profile: Dict[str, Any]):
        """사용자 프로필 업데이트"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO user_profiles
            (session_id, name, preferred_brands, budget_range, purchase_history)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            self.session_id,
            profile.get('name', ''),
            json.dumps(profile.get('preferred_brands', [])),
            profile.get('budget_range', ''),
            json.dumps(profile.get('purchase_history', []))
        ))
        conn.commit()
        conn.close()

    def get_user_profile(self) -> Dict[str, Any]:
        """사용자 프로필 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT name, preferred_brands, budget_range, purchase_history FROM user_profiles WHERE session_id = ?",
            (self.session_id,)
        )

        result = cursor.fetchone()
        conn.close()

        if result:
            return {
                'name': result[0],
                'preferred_brands': json.loads(result[1] or '[]'),
                'budget_range': result[2],
                'purchase_history': json.loads(result[3] or '[]')
            }
        return {}

# 트리밍이 적용된 쇼핑 어시스턴트 히스토리
class TrimmedShoppingHistory(ShoppingAssistantHistory):
    def __init__(self, session_id: str, max_messages: int = 10):
        super().__init__(session_id)
        self.max_messages = max_messages

    def add_messages(self, messages: List[BaseMessage]) -> None:
        # 기본 메시지 추가
        super().add_messages(messages)

        # 메시지 수 제한 적용
        if len(self.messages) > self.max_messages:
            # 최근 메시지만 유지
            recent_messages = self.messages[-self.max_messages:]

            # 데이터베이스 업데이트
            self.clear()
            super().add_messages(recent_messages)

# 쇼핑 어시스턴트 체인 구성
def create_shopping_assistant():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 개인화된 쇼핑 어시스턴트입니다.
        사용자의 선호도와 구매 이력을 바탕으로 맞춤형 추천을 제공합니다.

        사용자 정보:
        - 이름: {user_name}
        - 선호 브랜드: {preferred_brands}
        - 예산 범위: {budget_range}
        - 구매 이력: {purchase_history}
        """),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    return prompt | llm

def get_shopping_history(session_id: str) -> TrimmedShoppingHistory:
    return TrimmedShoppingHistory(session_id, max_messages=20)

# 사용 예제
shopping_chain = create_shopping_assistant()
shopping_chain_with_history = RunnableWithMessageHistory(
    shopping_chain,
    get_shopping_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 사용자 프로필 설정
history = get_shopping_history("user_123")
history.update_user_profile({
    'name': '김쇼핑',
    'preferred_brands': ['나이키', '아디다스'],
    'budget_range': '10-20만원',
    'purchase_history': ['운동화', '후드티', '조거 팬츠']
})

# 개인화된 쇼핑 상담
profile = history.get_user_profile()
response = shopping_chain_with_history.invoke({
    "input": "새로운 운동복을 추천해주세요",
    "user_name": profile.get('name', '고객님'),
    "preferred_brands": ', '.join(profile.get('preferred_brands', [])),
    "budget_range": profile.get('budget_range', '미설정'),
    "purchase_history": ', '.join(profile.get('purchase_history', []))
}, config={"configurable": {"session_id": "user_123"}})

print(response.content)
```

### 실습 2 해답: 의료 상담 챗봇

```python
class MedicalConsultationHistory(BaseChatMessageHistory, BaseModel):
    """의료 상담용 특수 메모리 시스템"""

    messages: List[BaseMessage] = Field(default_factory=list)
    medical_records: List[BaseMessage] = Field(default_factory=list)  # 의료 기록용 별도 저장
    max_context_messages: int = Field(default=30)  # 높은 컨텍스트 유지

    def add_messages(self, messages: List[BaseMessage]) -> None:
        for message in messages:
            self.messages.append(message)

            # 의료 관련 키워드가 포함된 메시지는 별도 보관
            if self._is_medical_record(message):
                self.medical_records.append(message)

        # 일반 대화만 트리밍 적용 (의료 기록은 유지)
        if len(self.messages) > self.max_context_messages:
            # 의료 기록 메시지는 제외하고 트리밍
            non_medical = [msg for msg in self.messages if not self._is_medical_record(msg)]
            medical_in_context = [msg for msg in self.messages if self._is_medical_record(msg)]

            # 최근 일반 대화만 유지
            recent_non_medical = non_medical[-(self.max_context_messages - len(medical_in_context)):]
            self.messages = medical_in_context + recent_non_medical

    def _is_medical_record(self, message: BaseMessage) -> bool:
        """의료 기록 여부 판단"""
        medical_keywords = [
            '증상', '아프다', '통증', '열', '기침', '두통', '복통',
            '진료', '병원', '약물', '처방', '검사', '진단'
        ]

        content = message.content.lower()
        return any(keyword in content for keyword in medical_keywords)

    def get_medical_summary(self) -> str:
        """의료 기록 요약"""
        if not self.medical_records:
            return "기록된 의료 정보가 없습니다."

        summary = "=== 의료 상담 기록 ===\n"
        for i, record in enumerate(self.medical_records[-10:], 1):  # 최근 10개
            summary += f"{i}. {record.content}\n"

        return summary

    def clear(self) -> None:
        self.messages = []
        # 의료 기록은 유지
        self.messages.extend(self.medical_records)

# 의료 상담 체인 구성
def create_medical_consultant():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 의료 상담 어시스턴트입니다.

        중요 지침:
        1. 의료 진단은 제공하지 않고 일반적인 정보만 제공
        2. 심각한 증상의 경우 병원 방문을 권장
        3. 이전 상담 내용을 참고하여 연관성 있는 조언 제공
        4. 환자의 프라이버시 보호

        의료 기록 요약:
        {medical_summary}
        """),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    return prompt | llm

def get_medical_history(session_id: str) -> MedicalConsultationHistory:
    if session_id not in medical_store:
        medical_store[session_id] = MedicalConsultationHistory()
    return medical_store[session_id]

medical_store = {}
medical_chain = create_medical_consultant()

# 의료 상담 체인에 히스토리 추가
medical_chain_with_history = RunnableWithMessageHistory(
    lambda x: medical_chain.invoke({
        **x,
        "medical_summary": get_medical_history(x.get("session_id", "")).get_medical_summary()
    }),
    get_medical_history,
    input_messages_key="input",
    history_messages_key="history",
)
```

### 실습 3 해답: 멀티 언어 학습 튜터

```python
class MultiLanguageTutorHistory(BaseChatMessageHistory, BaseModel):
    """다국어 학습용 메모리 시스템"""

    messages: List[BaseMessage] = Field(default_factory=list)
    language_progress: Dict[str, Dict] = Field(default_factory=dict)  # 언어별 학습 진도
    current_language: str = Field(default="")
    max_messages_per_language: int = Field(default=20)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        for message in messages:
            self.messages.append(message)

            # 현재 언어 감지 및 학습 기록 업데이트
            detected_lang = self._detect_language(message)
            if detected_lang:
                self.current_language = detected_lang
                self._update_language_progress(message, detected_lang)

        # 언어별 메시지 관리
        self._manage_language_messages()

    def _detect_language(self, message: BaseMessage) -> str:
        """메시지에서 언어 감지"""
        content = message.content.lower()

        # 간단한 키워드 기반 언어 감지
        if any(word in content for word in ['영어', 'english', 'speak english']):
            return 'english'
        elif any(word in content for word in ['일본어', 'japanese', '일본']):
            return 'japanese'
        elif any(word in content for word in ['중국어', 'chinese', '중국']):
            return 'chinese'
        elif any(word in content for word in ['프랑스어', 'french', '프랑스']):
            return 'french'

        return self.current_language

    def _update_language_progress(self, message: BaseMessage, language: str):
        """언어별 학습 진도 업데이트"""
        if language not in self.language_progress:
            self.language_progress[language] = {
                'total_messages': 0,
                'vocabulary_learned': [],
                'grammar_points': [],
                'weak_areas': []
            }

        self.language_progress[language]['total_messages'] += 1

        # 어휘나 문법 포인트 추출 (실제로는 더 정교한 NLP 분석 필요)
        content = message.content
        if '단어' in content or 'vocabulary' in content:
            # 어휘 학습으로 분류
            pass
        elif '문법' in content or 'grammar' in content:
            # 문법 학습으로 분류
            pass

    def _manage_language_messages(self):
        """언어별 메시지 수 관리"""
        if len(self.messages) > self.max_messages_per_language * 3:  # 3개 언어 기준
            # 언어별로 최근 메시지만 유지
            language_messages = {}

            for msg in self.messages:
                lang = self._detect_language(msg) or 'general'
                if lang not in language_messages:
                    language_messages[lang] = []
                language_messages[lang].append(msg)

            # 각 언어별로 최근 메시지만 유지
            trimmed_messages = []
            for lang, msgs in language_messages.items():
                trimmed_messages.extend(msgs[-self.max_messages_per_language:])

            self.messages = sorted(trimmed_messages, key=lambda x: hash(x.content))  # 간단한 정렬

    def get_language_summary(self, language: str = None) -> str:
        """특정 언어의 학습 진도 요약"""
        if language and language in self.language_progress:
            progress = self.language_progress[language]
            return f"""
            {language} 학습 진도:
            - 총 대화 수: {progress['total_messages']}
            - 학습한 어휘: {len(progress['vocabulary_learned'])}개
            - 학습한 문법: {len(progress['grammar_points'])}개
            """

        # 전체 언어 요약
        summary = "=== 다국어 학습 현황 ===\n"
        for lang, progress in self.language_progress.items():
            summary += f"{lang}: {progress['total_messages']}개 대화\n"

        return summary

    def clear(self) -> None:
        self.messages = []
        # 학습 진도는 유지

    def switch_language(self, new_language: str):
        """학습 언어 전환"""
        self.current_language = new_language

# 다국어 튜터 체인 구성
def create_multilingual_tutor():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 다국어 학습 튜터입니다.

        현재 학습 언어: {current_language}

        학습 진도 요약:
        {language_summary}

        지침:
        1. 학습자의 수준에 맞는 적절한 난이도로 대화
        2. 실수를 교정하고 개선점 제시
        3. 이전 학습 내용과 연관지어 설명
        4. 각 언어별 특성에 맞는 학습 방법 제시
        """),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    return prompt | llm

def get_multilingual_history(session_id: str) -> MultiLanguageTutorHistory:
    if session_id not in tutor_store:
        tutor_store[session_id] = MultiLanguageTutorHistory()
    return tutor_store[session_id]

tutor_store = {}
tutor_chain = create_multilingual_tutor()

# 다국어 튜터 체인에 히스토리 추가
tutor_chain_with_history = RunnableWithMessageHistory(
    lambda x: tutor_chain.invoke({
        **x,
        "current_language": get_multilingual_history(x.get("session_id", "")).current_language,
        "language_summary": get_multilingual_history(x.get("session_id", "")).get_language_summary()
    }),
    get_multilingual_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 사용 예제
# 영어 학습 시작
history = get_multilingual_history("student_1")
history.switch_language("english")

response = tutor_chain_with_history.invoke({
    "input": "영어 회화를 연습하고 싶어요",
    "session_id": "student_1"
}, config={"configurable": {"session_id": "student_1"}})

print(response.content)
```

## 참고 자료

### 공식 문서
- [LangChain Memory Documentation](https://python.langchain.com/docs/modules/memory/)
- [RunnableWithMessageHistory Guide](https://python.langchain.com/docs/expression_language/how_to/message_history)
- [Message Trimming Documentation](https://python.langchain.com/docs/how_to/trim_messages/)

### 심화 학습
- [ChatMessageHistory Implementations](https://python.langchain.com/docs/integrations/memory/)
- [Custom Memory Classes](https://python.langchain.com/docs/modules/memory/custom_memory)
- [Session Management Best Practices](https://python.langchain.com/docs/how_to/manage_memory)

### 관련 라이브러리
- SQLite3: Python 내장 데이터베이스
- Pydantic: 데이터 검증 및 모델링
- Redis: 고성능 메모리 저장소 (선택적)

이 가이드를 통해 다양한 채팅 히스토리 관리 방법을 학습하고, 실제 애플리케이션에서 요구사항에 맞는 최적의 메모리 시스템을 구축할 수 있습니다.