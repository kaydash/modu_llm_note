# LangChain Neo4j 통합 - Part 1

**학습 자료**: PRJ04_W3_002 - LangChain과 Neo4j 그래프 데이터베이스 통합

---

## 📚 학습 목표

이 실습을 완료하면 다음을 할 수 있습니다:

1. **Neo4jGraph로 그래프 데이터베이스 연결**: LangChain을 통한 Neo4j 연결 및 Cypher 쿼리 실행
2. **대화 이력을 그래프로 관리**: Neo4jChatMessageHistory를 활용한 구조화된 대화 저장
3. **대화 패턴 분석**: 그래프 쿼리를 통한 대화 흐름 및 통계 분석
4. **실무 활용 시나리오 이해**: 채팅 기록 관리 및 세션 기반 분석 구현

---

## 🔑 핵심 개념

### LangChain Neo4j 통합

- **Neo4jGraph**: Neo4j 데이터베이스와의 연결 및 Cypher 쿼리 실행을 위한 래퍼 클래스
- **Neo4jChatMessageHistory**: 대화 메시지를 그래프 구조로 저장하고 관리하는 컴포넌트
- **구조화된 대화 저장**: 메시지를 노드와 관계로 표현하여 패턴 분석 가능

### 주요 활용 사례

- **지식 그래프 기반 RAG**: 복잡한 관계 정보를 활용한 검색 증강 생성
- **대화형 AI 시스템**: 문맥을 유지하는 챗봇 및 가상 어시스턴트
- **대화 패턴 분석**: 세션별 상호작용 통계 및 흐름 추적

---

## 🛠 환경 설정

### 필수 라이브러리 설치

```bash
pip install langchain-neo4j langchain-openai python-dotenv
```

### Neo4j 데이터베이스 준비

- Neo4j AuraDB 클라우드 인스턴스 또는 로컬 Neo4j 서버 필요
- 연결 정보를 `.env` 파일에 설정:

```bash
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=neo4j
OPENAI_API_KEY=your-openai-key
```

---

## 💻 단계별 구현


---

## **기본 개요**

- **LangChain과 Neo4j 그래프 데이터베이스의 통합** 기능 제공
- Neo4j 데이터베이스와 LangChain을 함께 사용하여 **AI 기반 지식 그래프 애플리케이션** 구축
- **설치**: `pip install -U langchain-neo4j`
- **공식 문서**: [LangChain Neo4j Integration](https://python.langchain.com/docs/integrations/graphs/neo4j_cypher/)

- **주요 활용 사례**
    - **지식 그래프 기반 RAG**: 복잡한 관계 정보를 활용한 검색 증강 생성
    - **대화형 AI 시스템**: 문맥을 유지하는 챗봇 및 가상 어시스턴트
    - **의미론적 검색**: 벡터 임베딩과 그래프 구조를 결합한 하이브리드 검색
    - **추천 시스템**: 관계 기반 개인화 추천 엔진

- **핵심 컴포넌트 소개**
    1. **Neo4jGraph**: 그래프 데이터베이스 연결 및 Cypher 쿼리 실행
    2. **Neo4jChatMessageHistory**: 대화 이력 그래프 저장
    3. **Neo4jVector**: 벡터 기반 의미론적 검색
    4. **GraphCypherQAChain**: 자연어-Cypher 변환 및 QA

## **1. Neo4jGraph**

- **Neo4jGraph** 클래스는 Neo4j Python 드라이버를 래핑하여 **간단한 인터페이스** 제공
- Neo4j 데이터베이스와의 상호 작용 및 기본적인 **Cypher 쿼리** 실행이 가능

### **1.1 기본 연결 설정**

```python
import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

# 환경 변수 로드
load_dotenv()

# 최신 Neo4jGraph 연결 
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"), 
    username=os.getenv("NEO4J_USERNAME"), 
    password=os.getenv("NEO4J_PASSWORD"),
    database=os.getenv("NEO4J_DATABASE", "neo4j"),  # 명시적 데이터베이스 지정
    refresh_schema=True,  # 스키마 자동 갱신
    sanitize=True,  # 쿼리 검증 강화
    enhanced_schema=True  # 향상된 스키마 정보 제공
)

# 연결 상태 확인
try:
    result = graph.query("RETURN 'Neo4j 연결 성공!' as message")
    print(f"연결 결과: {result[0]}")
except Exception as e:
    print(f"연결 실패: {e}")
```

### **1.2 쿼리 실행**

```python
# 성능 모니터링과 함께 쿼리 실행
def execute_with_monitoring(graph, query, parameters=None):
    """모니터링 기능이 포함된 쿼리 실행"""
    import time
    
    start_time = time.time()
    try:
        result = graph.query(query, parameters or {})
        execution_time = time.time() - start_time
        
        return {
            "success": True,
            "data": result,
            "execution_time": execution_time,
            "query": query
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "execution_time": time.time() - start_time,
            "query": query
        }

# 사용 예제
result = execute_with_monitoring(graph, "MATCH (n) RETURN count(n) as node_count")
if result["success"]:
    print(f"노드 개수: {result['data'][0]['node_count']}")
    print(f"실행 시간: {result['execution_time']:.3f}초")
else:
    print(f"오류: {result['error']}")
```

### **[실습 1]** 

- Neo4j 데이터베이스에 연결하고, 간단한 Cypher 쿼리를 실행하여 전체 노드 개수를 반환하는 코드를 작성하세요.

```python
# 여기에 코드를 작성하세요. 

from langchain_neo4j import Neo4jGraph
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE
)

query = "MATCH (n) RETURN count(n)"
results = graph.query(query)
print(f"노드 개수: {results[0]['count(n)']}")
```

## **2. Neo4jChatMessageHistory - 대화 이력 관리**

- **Neo4jChatMessageHistory**는 채팅 메시지를 **구조화된 그래프**로 저장
- **대화 패턴 분석**과 **컨텍스트 추적**이 가능


![image-3.png](attachment:image-3.png)

### **2.1 대화 이력 관리**

```python
from langchain_neo4j import Neo4jChatMessageHistory
from datetime import datetime

# 고급 설정으로 대화 이력 관리자 초기화
history = Neo4jChatMessageHistory( 
    session_id="session_001",
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database="neo4j",                    # 기본값: "neo4j"
    node_label="Session",                # 기본값: "Session" 
    window=3                             # 기본값: 3 (메시지 조회 시 가져올 메시지 쌍의 개수) -> window=3 이면 최근 6개 메시지 (사용자 3개 + AI 3개)를 조회
)


# 대화 이력 출력
history.messages
```

```python
from langchain_core.messages import HumanMessage, AIMessage

# 대화 이력 추가
history.add_user_message(
    HumanMessage(
        content="hi",
        additional_kwargs={"language": "en"}
    )
)
history.add_ai_message(
    AIMessage(
        content="whats up?",
        additional_kwargs={"language": "en"}
    )
)

# 대화 이력 출력
history.messages
```

```python
# 대화 이력 추가 
history.add_user_message("Please tell me about yourself.")
history.add_ai_message("I am an AI assistant.")

# 대화 이력 출력
history.messages
```

### **[실습 2]** 

- 다음 대화 이력을 기존의 Neo4j 데이터베이스에 추가하세요.

- 대화 이력:
    ```
    User: "안녕하세요, 오늘 날씨가 어떤가요?"
    AI: "안녕하세요! 오늘은 맑고 화창한 날씨가 예상됩니다."
    User: "그럼 산책하기 좋겠네요. 근처 공원을 추천해 줄 수 있나요?"
    AI: "네, 물론입니다. 근처에 '중앙 공원'이 산책하기 아주 좋습니다."
    ```

```python
# 여기에 코드를 작성하세요

# 대화 이력 추가
history.add_user_message("안녕하세요, 오늘 날씨가 어떤가요?")
history.add_ai_message("안녕하세요! 오늘은 맑고 화창한 날씨가 예상됩니다.")
history.add_user_message("그럼 산책하기 좋겠네요. 근처 공원을 추천해 줄 수 있나요?")
history.add_ai_message("네, 물론입니다. 근처에 '중앙 공원'이 산책하기 아주 좋습니다.")

# 대화 이력 출력
history.messages
```

### **2.2 대화 패턴 분석**

```python
# 그래프 스키마 갱신
graph.refresh_schema()
```

```python
# 세션 ID 확인
history._session_id
```

```python
# 세션 기반 분석 (마지막 메시지 추적)
session_analysis_query = """
// 세션 ID로 세션 노드와 마지막 메시지 노드를 찾는 쿼리
MATCH (s:Session {id: $session_id})
MATCH (s)-[:LAST_MESSAGE]->(last:Message)  // 세션 노드와 마지막 메시지 노드를 연결하는 관계를 찾음
RETURN 
    s.id as session_id,
    last as last_message
"""

result = graph.query(
    query=session_analysis_query, 
    params={"session_id": history._session_id}
    )
    
result
```

```python
# 세션의 모든 메시지 가져오기 (연결된 체인 따라가기)
conversation_analysis_query = """
// 세션 ID로 세션 노드와 마지막 메시지 노드를 찾고, 
MATCH (s:Session {id: $session_id})-[:LAST_MESSAGE]->(last:Message)

// 마지막 메시지에서 시작하여 연결된 모든 메시지를 가져오는 쿼리
MATCH path = (first:Message)-[:NEXT*0..]->(last)    // *0..은 0개 이상의 관계를 의미
WHERE NOT EXISTS(()-[:NEXT]->(first))    // 이때 first는 이전 메시지가 없는 첫번째 메시지

// 메시지 노드의 속성을 가져오는 쿼리
WITH nodes(path) as messages  // path를 nodes로 변환 (WITH 구문: 쿼리 결과를 다음 쿼리로 전달)

// 메시지 노드를 UNWIND하여 각 메시지의 속성을 반환 (UNWIND 구문: 리스트를 개별 요소로 분리)
UNWIND messages as msg

// 반환할 메시지의 속성
RETURN 
    msg.content as content,
    msg.role as role,
    msg.createdAt as created_at
ORDER BY msg.createdAt ASC
"""

result = graph.query(
    query=conversation_analysis_query,
    params={"session_id": history._session_id}
)

result
```

```python
# 대화 이력 출력
for i, record in enumerate(result, 1):
    print(f"{i}. [{record['role'].upper()}] {record['content']} ({record['created_at']})")
```

```python
# 대화 통계 계산
conversation_stats_query = """
// 세션 ID로 세션 노드와 마지막 메시지 노드를 찾고,
MATCH (s:Session {id: $session_id})-[:LAST_MESSAGE]->(last:Message)

// 마지막 메시지에서 시작하여 연결된 모든 메시지를 가져오는 쿼리
MATCH path = (first:Message)-[:NEXT*0..]->(last)
WHERE NOT EXISTS(()-[:NEXT]->(first))   

// 메시지 노드들을 가져오고,
WITH nodes(path) as messages

// 메시지 노드들을 가져오고,
RETURN 
    size(messages) as total_messages,   // 총 메시지 수
    size([msg IN messages WHERE msg.role = 'human']) as user_messages,   // 사용자 메시지 수
    size([msg IN messages WHERE msg.role = 'ai']) as ai_messages,   // AI 메시지 수
    [msg IN messages | msg.role] as message_sequence,   // 메시지 순서
    min([msg IN messages | msg.createdAt]) as conversation_start,   // 대화 시작 시간
    max([msg IN messages | msg.createdAt]) as conversation_end   // 대화 종료 시간
"""

result = graph.query(
    query=conversation_stats_query, 
    params={"session_id": history._session_id}
)

result
```

```python
stats = result[0]
print(f"총 메시지 수: {stats['total_messages']}")
print(f"사용자 메시지: {stats['user_messages']}")
print(f"AI 메시지: {stats['ai_messages']}")
print(f"대화 시작: {stats['conversation_start'][0].to_clock_time()}")
print(f"대화 종료: {stats['conversation_end'][0]}")
print(f"메시지 순서: {' -> '.join(stats['message_sequence'][-10:])}")  # 마지막 10개
```

### **[실습 3]** 

- `Neo4jChatMessageHistory`를 사용하여 간단한 챗봇의 대화 기록을 저장하고 불러오는 코드를 작성하세요. 

```python
# 여기에 코드를 작성하세요.

from langchain_neo4j import Neo4jChatMessageHistory
import os

# 대화 이력 관리자 초기화
history = Neo4jChatMessageHistory(
    session_id="chatbot_session",
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database="neo4j"
)

# 대화 이력 초기화 (실습을 위해 기존 이력 삭제)
history.clear()

# 메시지 추가
history.add_user_message("안녕하세요!")
history.add_ai_message("안녕하세요. 무엇을 도와드릴까요?")
history.add_user_message("LangChain Neo4j에 대해 알려주세요.")

# 대화 기록 불러오기 및 출력
messages = history.messages
print("대화 기록:")
for message in messages:
    print(f"{message.type}: {message.content}")
```

### **[실습 4]** 

- `Neo4jChatMessageHistory`를 사용하여 다음 기능을 구현해보세요:

```python
# 1. 여러 세션의 대화 이력 생성

sessions = ["tech_support", "product_inquiry", "general_chat"]

conversation_data = {
    "tech_support": [
        ("user", "시스템에 오류가 발생했습니다."),
        ("ai", "어떤 종류의 오류인지 더 자세히 설명해주시겠어요?"),
        ("user", "로그인이 안 됩니다."),
        ("ai", "브라우저 캐시를 삭제하고 다시 시도해보세요.")
    ],
    "product_inquiry": [
        ("user", "신제품 출시 일정을 알고 싶습니다."),
        ("ai", "다음 분기에 새로운 제품 라인업이 출시될 예정입니다."),
        ("user", "가격대는 어떻게 되나요?"),
        ("ai", "가격은 기본 모델 기준 $299부터 시작합니다.")
    ],
    "general_chat": [
        ("user", "안녕하세요!"),
        ("ai", "안녕하세요! 무엇을 도와드릴까요?"),
        ("user", "오늘 날씨가 좋네요."),
        ("ai", "네, 정말 좋은 날씨입니다. 다른 도움이 필요하시면 언제든 말씀하세요.")
    ]
}

# 여기에 코드를 작성하세요.

# 여러 세션의 대화 이력 생성
session_histories = {}

for session_id in sessions:
    # 세션별 대화 이력 관리자 초기화
    session_history = Neo4jChatMessageHistory(
        session_id=session_id,
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        database="neo4j"
    )
    
    # 기존 이력 초기화
    session_history.clear()
    
    # 대화 데이터 추가
    for role, content in conversation_data[session_id]:
        if role == "user":
            session_history.add_user_message(content)
        else:
            session_history.add_ai_message(content)
    
    session_histories[session_id] = session_history
    print(f"✅ {session_id} 세션 생성 완료 - {len(session_history.messages)}개 메시지")

print(f"\n총 {len(session_histories)}개 세션 생성 완료")
```

```python
# 2. 세션별 활동 비교
activity_stats = {}

for session_id, history in session_histories.items():
    # TODO: 메시지 개수 계산 
    message_count = len(history.messages)
    
    # TODO: 사용자 메시지 개수 계산
    user_messages = len([msg for msg in history.messages if msg.type == "human"])
    
    # TODO: AI 메시지 개수 계산  
    ai_messages = len([msg for msg in history.messages if msg.type == "ai"])
    
    activity_stats[session_id] = {
        'total_messages': message_count,
        'user_messages': user_messages,
        'ai_messages': ai_messages,
        'interaction_ratio': user_messages / message_count if message_count > 0 else 0
    }

# 활동 통계 출력 포맷 개선
for session_id, stats in activity_stats.items():
    print(f"세션 ID: {session_id}")
    print(f"  총 메시지 수: {stats['total_messages']}")
    print(f"  사용자 메시지: {stats['user_messages']}")
    print(f"  AI 메시지: {stats['ai_messages']}")
    print(f"  상호작용 비율: {stats['interaction_ratio']:.2f}\n")
```
