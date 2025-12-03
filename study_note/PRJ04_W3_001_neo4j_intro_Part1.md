# **PRJ04_W3_001: Neo4j Knowledge Graph 입문 Part 1 - 기초와 Cypher 쿼리**

**Part 1 학습 가이드**

이 문서는 Neo4j 그래프 데이터베이스의 기초부터 Cypher 쿼리 언어를 활용한 데이터 CRUD 작업까지 다룹니다.

---

## **📚 학습 목표**

이 실습을 완료하면 다음을 할 수 있습니다:

1. **Knowledge Graph와 Neo4j의 핵심 개념 이해**: 그래프 데이터 모델의 장점과 활용 사례 파악
2. **Neo4j AuraDB 클라우드 환경 구축**: 무료 인스턴스 생성 및 Python 환경 설정
3. **Cypher 쿼리로 노드와 관계 생성**: CREATE와 MERGE를 활용한 데이터 모델링
4. **Cypher 쿼리로 그래프 데이터 조회**: MATCH와 WHERE를 사용한 패턴 매칭 및 필터링
5. **복잡한 그래프 관계 설계 및 구현**: 다중 노드 간 의미있는 관계 구축

---

## **🔑 핵심 개념**

### **1. Knowledge Graph (지식 그래프)**

**정의**: 실세계의 개체(Entity)와 그들 간의 관계(Relationship)를 그래프 구조로 표현한 데이터 모델

**핵심 구성요소**:
1. **노드(Node)**: 개체(Entity) - 사람, 제품, 개념 등
2. **엣지(Edge)**: 관계(Relationship) - 노드들 간의 연결
3. **속성(Property)**: 노드와 엣지의 특성 정보

**장점**:
- **직관적 데이터 표현**: 실세계 관계를 자연스럽게 모델링
- **유연한 스키마**: 새로운 관계 타입 추가 용이
- **추론 능력**: 숨겨진 패턴과 관계 발견
- **연결성**: 다양한 데이터 소스 통합

**예시 구조**:
```
(사람: 젠슨 황) -[WORKS_FOR]-> (회사: 엔비디아)
(회사: 엔비디아) -[MANUFACTURES]-> (제품: GPU)
(기사: AI 반도체 뉴스) -[MENTIONS]-> (회사: 엔비디아)
```

### **2. Neo4j 그래프 데이터베이스**

**특징**:
- 가장 널리 사용되는 그래프 데이터베이스
- 네이티브 그래프 저장 및 처리 (index-free adjacency)
- 강력한 Cypher 쿼리 언어: SQL과 유사한 직관적 문법
- 확장성: 수십억 노드와 관계 처리 가능
- LangChain 통합: AI/ML 파이프라인 완벽 지원

**전통적 RDBMS vs Neo4j**:
| 특성 | RDBMS | Neo4j |
|------|-------|-------|
| 데이터 모델 | 테이블 (행/열) | 그래프 (노드/관계) |
| 조인 성능 | 조인 증가 시 성능 저하 | 관계 탐색 시 일정한 성능 |
| 스키마 | 엄격한 스키마 | 유연한 스키마 |
| 쿼리 언어 | SQL | Cypher |
| 적합한 데이터 | 정형 데이터 | 연결된 데이터 |

### **3. LangChain + Neo4j 통합**

**LangChain-Neo4j 패키지 주요 기능**:
1. **Neo4jGraph**: 그래프 데이터베이스 래퍼 (Python ↔ Neo4j 연결)
2. **Neo4jVector**: 벡터 검색 지원 (임베딩 저장 및 검색)
3. **GraphCypherQAChain**: 자연어 → Cypher 쿼리 자동 변환
4. **Neo4jChatMessageHistory**: 채팅 기록 관리

**AI 기반 그래프 활용 시나리오**:
- **RAG (Retrieval Augmented Generation)**: 그래프 구조 + 벡터 검색 결합
- **자연어 쿼리**: "엔비디아와 관련된 모든 기사 찾아줘" → Cypher 쿼리 자동 생성
- **지능형 추천**: 그래프 경로 분석을 통한 추천 시스템

### **4. Cypher 쿼리 언어**

**기본 구조**:
```cypher
// 노드 매칭
MATCH (변수:레이블 {속성: 값})

// 관계 매칭
MATCH (a)-[r:관계타입]->(b)

// 노드 생성
CREATE (변수:레이블 {속성1: 값1, 속성2: 값2})

// 조건부 생성 (중복 방지)
MERGE (변수:레이블 {속성: 값})

// 결과 반환
RETURN 변수
```

**CREATE vs MERGE**:
- **CREATE**: 항상 새로운 노드/관계 생성, 중복 허용, 빠른 실행
- **MERGE**: 존재하면 재사용, 없으면 생성, 중복 방지 (PRIMARY KEY 역할)

---

## **🛠 환경 설정**

### **Step 1: Neo4j AuraDB 인스턴스 생성**

Neo4j AuraDB는 완전 관리형 클라우드 서비스로 제공되는 Neo4j 그래프 데이터베이스입니다.

#### **(1) 회원가입**
1. https://neo4j.com/product/auradb/ 방문
2. "Start Free" 버튼 클릭
3. 이메일 또는 Google 계정으로 가입

#### **(2) AuraDB 인스턴스 생성**
1. 로그인 후 "New Instance" 선택
2. "Aura DB Free" 선택
   - 무료 티어: 200k 노드, 400k 관계까지 지원
   - 영구 무료 사용 가능
3. 인스턴스 이름 입력 (예: "langchain-kg")
4. 리전 선택 (가장 가까운 리전 선택)

#### **(3) 자격 증명 저장**

인스턴스 생성 완료 시 **단 한 번만** 표시되는 정보:
- **Username**: neo4j (기본값)
- **Password**: 자동 생성된 긴 비밀번호 (반드시 복사하여 안전하게 보관!)
- **Connection URL**: `neo4j+s://xxxxxxx.databases.neo4j.io`

⚠️ **주의**: 비밀번호를 잃어버리면 인스턴스를 삭제하고 다시 생성해야 합니다.

#### **(4) Neo4j Browser로 연결 확인**
1. 대시보드에서 "Open" 버튼 클릭
2. Neo4j Browser 열림
3. 자격 증명으로 로그인
4. 테스트 쿼리 실행:
   ```cypher
   CREATE (n:Test {name: "Hello AuraDB"})
   RETURN n
   ```

### **Step 2: Python 환경 설정**

#### **(1) 필수 라이브러리 설치**

```bash
# 최신 LangChain Neo4j 통합 라이브러리
pip install -U langchain-neo4j

# 기본 Neo4j Python 드라이버 (선택사항)
pip install neo4j

# 환경 변수 관리
pip install python-dotenv

# OpenAI API (LangChain 통합용)
pip install openai langchain-openai
```

#### **(2) 환경 변수 설정**

프로젝트 루트에 `.env` 파일 생성:

```bash
# Neo4j AuraDB 연결 정보
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=복사해서_저장한_Password
NEO4J_DATABASE=neo4j

# OpenAI API 키 (LangChain 통합 시 필요)
OPENAI_API_KEY=your_openai_api_key
```

⚠️ **보안**: `.env` 파일은 `.gitignore`에 추가하여 버전 관리에서 제외하세요!

### **Step 3: 연결 테스트**

#### **(1) LangChain Neo4jGraph 사용 (권장)**

```python
import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

# 환경 변수 로드
load_dotenv()

# Neo4j 연결 객체 초기화 (최신 방식)
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database=os.getenv("NEO4J_DATABASE", "neo4j")
)

# 그래프 스키마 최신화
graph.refresh_schema()

# 연결 테스트
try:
    result = graph.query("RETURN 'Hello Neo4j!' as message")
    print("연결 성공:", result)
except Exception as e:
    print("연결 실패:", e)
```

**출력 예시**:
```
연결 성공: [{'message': 'Hello Neo4j!'}]
```

**설명**:
- `Neo4jGraph`: LangChain에서 제공하는 Neo4j 래퍼 클래스
- `refresh_schema()`: 그래프 스키마 정보를 메모리에 캐싱
- `query()`: Cypher 쿼리 실행 및 결과 반환

#### **(2) 기본 Neo4j Python 드라이버 사용**

```python
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

# Python 드라이버 직접 사용
URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    # execute_query 메서드 사용
    records, summary, keys = driver.execute_query(
        "RETURN 'Hello Neo4j!' as message",
        database_=os.getenv("NEO4J_DATABASE", "neo4j")
    )
    for record in records:
        print(record["message"])
```

**LangChain vs 기본 드라이버**:
| 특성 | LangChain Neo4jGraph | 기본 Neo4j 드라이버 |
|------|----------------------|----------------------|
| 사용 편의성 | 높음 (추상화) | 중간 (직접 제어) |
| LangChain 통합 | 완벽 지원 | 수동 구현 필요 |
| 성능 | 약간 느림 (래퍼) | 빠름 (직접 연결) |
| 권장 사항 | AI 파이프라인 구축 | 순수 그래프 작업 |

---

## **💻 단계별 구현**

### **Step 1: 노드 생성 (CREATE와 MERGE)**

#### **(1) Article 노드 생성**

**CREATE 사용 예시** (항상 새 노드 생성):

```python
# 첫 번째 뉴스 기사 생성 (CREATE)
cypher_query = """
CREATE (a:Article {
    id: "tech-001",
    title: "엔비디아, AI 반도체 수요 급증으로 실적 고공행진",
    content: "엔비디아가 2024년 1분기 실적 발표에서 AI 반도체 수요 증가에 힘입어 전년 대비 200% 성장을 기록했다고 발표했다. 특히 데이터센터용 GPU 판매가 크게 증가했으며, 젠슨 황 CEO는 AI 수요가 앞으로도 지속적으로 증가할 것으로 전망했다.",
    publishDate: date("2024-03-15"),
    source: "테크경제신문",
    author: "김기자",
    category: "technology",
    wordCount: 245
})
RETURN a
"""
result = graph.query(cypher_query)
print("첫 번째 기사 생성:", result)
```

**출력 예시**:
```python
첫 번째 기사 생성: [{'a': {'id': 'tech-001', 'title': '엔비디아, AI 반도체 수요 급증...', ...}}]
```

**MERGE 사용 예시** (중복 방지, 없으면 생성):

```python
# 두 번째 뉴스 기사 생성 (MERGE)
cypher_query = """
MERGE (a:Article {
    id: "tech-002",
    title: "삼성전자, 차세대 AI 반도체 개발 계획 발표",
    content: "삼성전자가 차세대 AI 반도체 개발 로드맵을 공개했다. 2025년까지 1000억원을 투자해 GAA 공정 기반의 새로운 AI 가속기를 개발할 계획이며, 엔비디아와의 기술 격차를 줄이겠다는 목표다.",
    publishDate: date("2024-03-14"),
    source: "테크경제신문",
    author: "이기자",
    category: "technology",
    wordCount: 198
})
RETURN a
"""
result = graph.query(cypher_query)
print("두 번째 기사 생성:", result)
```

**CREATE vs MERGE 실전 활용**:
- **CREATE**: 로그 데이터, 이벤트 기록 (중복 허용)
- **MERGE**: 마스터 데이터, 참조 데이터 (중복 방지 필수)

#### **(2) Topic 노드 생성**

```python
cypher_query = """
MERGE (t1:Topic {name: "AI 반도체", description: "인공지능 가속기 및 관련 기술"})
MERGE (t2:Topic {name: "기업 실적", description: "회사의 재무 성과 및 실적"})
MERGE (t3:Topic {name: "기술 투자", description: "R&D 투자 및 기술 개발"})
RETURN t1, t2, t3
"""
graph.query(cypher_query)
```

**설명**:
- 여러 노드를 한 번에 생성 가능
- `MERGE`를 사용하여 같은 쿼리를 여러 번 실행해도 중복 생성 방지

---

### **Step 2: 데이터 조회 (READ)**

#### **(1) 기본 조회 (MATCH)**

```python
# 모든 Article 노드 조회
cypher_query = """
MATCH (a:Article)
RETURN a.id, a.title, a.publishDate, a.author
ORDER BY a.publishDate DESC
"""
result = graph.query(cypher_query)
print(result)
```

**출력 예시**:
```python
[
    {'a.id': 'tech-001', 'a.title': '엔비디아, AI 반도체...', 'a.publishDate': '2024-03-15', 'a.author': '김기자'},
    {'a.id': 'tech-002', 'a.title': '삼성전자, 차세대...', 'a.publishDate': '2024-03-14', 'a.author': '이기자'}
]
```

**Cypher 구문 분석**:
- `MATCH (a:Article)`: Article 레이블을 가진 모든 노드를 `a` 변수에 할당
- `RETURN a.id, a.title`: 특정 속성만 반환
- `ORDER BY a.publishDate DESC`: 발행일 기준 내림차순 정렬

#### **(2) 필터링 (WHERE)**

```python
# 특정 조건 필터링 - 김기자 작성, 기술 카테고리 기사
cypher_query = """
MATCH (a:Article)
WHERE a.author = "김기자" AND a.category = "technology"
RETURN a.title, a.content
"""
result = graph.query(cypher_query)
print(result)
```

**WHERE 조건 예시**:
```cypher
# 문자열 포함 검색
WHERE a.title CONTAINS "AI"

# 숫자 범위 검색
WHERE a.wordCount >= 200 AND a.wordCount <= 300

# 날짜 범위 검색
WHERE a.publishDate >= date("2024-01-01")

# NULL 체크
WHERE a.content IS NOT NULL

# 여러 값 중 하나
WHERE a.category IN ["technology", "business"]
```

---

### **Step 3: 관계 생성 및 조회**

#### **(1) 복합 노드와 관계 생성**

**인물-회사 관계 생성**:

```python
# 젠슨 황 CEO 노드 생성 및 엔비디아와의 관계 설정
cypher_query = """
MERGE (p:Person {
    name: "젠슨 황",
    role: "CEO",
    nationality: "대만계 미국인",
    birthYear: 1963
})
MERGE (c:Company {name: "엔비디아"})
MERGE (p)-[r:WORKS_FOR {
    since: 1993,
    position: "CEO",
    isFounder: true
}]->(c)
RETURN p, c, r
"""
result = graph.query(cypher_query)
```

**그래프 구조**:
```
(Person: 젠슨 황) -[WORKS_FOR {since: 1993, position: "CEO"}]-> (Company: 엔비디아)
```

**제품-회사 관계 생성**:

```python
# GPU 제품 노드 생성 및 엔비디아와의 관계 설정
cypher_query = """
MERGE (p:Product {
    name: "GPU",
    category: "Hardware",
    type: "Graphics Processing Unit"
})
MERGE (c:Company {name: "엔비디아"})
MERGE (p)-[r:MANUFACTURED_BY {
    since: 1999,
    primaryUse: "AI/ML Computing"
}]->(c)
RETURN p, c, r
"""
result = graph.query(cypher_query)
```

**관계 설계 원칙**:
1. **의미있는 관계 이름**: `WORKS_FOR`, `MANUFACTURED_BY` (동사 형태)
2. **방향성 고려**: 화살표 방향이 의미를 가짐 (`->`는 "~를 향하여")
3. **관계 속성 활용**: 관계에 `since`, `position` 등 추가 정보 저장
4. **다중 관계 허용**: 같은 노드 간 여러 관계 타입 가능

#### **(2) 복잡한 관계 네트워크 생성**

```python
# 첫 번째 기사의 다중 관계 생성
cypher_query = """
// 노드 매칭
MATCH (a:Article {id: "tech-001"})
MATCH (t1:Topic {name: "AI 반도체"})
MATCH (t2:Topic {name: "기업 실적"})
MATCH (c:Company {name: "엔비디아"})
MATCH (p:Person {name: "젠슨 황"})

// 관계 생성
MERGE (a)-[:COVERS {relevance: 0.9}]->(t1)
MERGE (a)-[:COVERS {relevance: 0.8}]->(t2)
MERGE (a)-[:MENTIONS {
    context: "실적 발표 기업",
    sentiment: "positive"
}]->(c)
MERGE (a)-[:QUOTES {
    statement: "AI 수요가 지속적으로 증가할 것",
    context: "미래 전망"
}]->(p)

RETURN a.title, "관계 생성 완료" as status
"""
result = graph.query(cypher_query)
print(result)
```

**생성된 그래프 구조**:
```
(Article: tech-001)
├── [COVERS {relevance: 0.9}] → (Topic: AI 반도체)
├── [COVERS {relevance: 0.8}] → (Topic: 기업 실적)
├── [MENTIONS {context: "실적 발표 기업"}] → (Company: 엔비디아)
└── [QUOTES {statement: "..."}] → (Person: 젠슨 황)
```

#### **(3) 관계 기반 조회 (패턴 매칭)**

```python
# 특정 토픽을 다루는 기사 찾기
cypher_query = """
MATCH (a:Article)-[:COVERS]->(t:Topic)
WHERE t.name CONTAINS "AI"
RETURN a.title
"""
result = graph.query(cypher_query)
```

**고급 패턴 매칭**:

```python
# 특정 토픽과 관련된 모든 기사와 회사 찾기 (다중 홉)
cypher_query = """
MATCH path = (a:Article)-[:COVERS]->(t:Topic {name: "AI 반도체"})<-[:COVERS]-(a2:Article)-[:MENTIONS]->(c:Company)
RETURN DISTINCT a.title, a2.title, c.name, length(path) as pathLength
ORDER BY pathLength
"""
result = graph.query(cypher_query)
```

**패턴 매칭 설명**:
- `path = (a)-[r]->(b)`: 경로 전체를 변수에 할당
- `<-[:COVERS]-`: 역방향 관계 매칭
- `length(path)`: 경로 길이 (홉 수) 계산
- `DISTINCT`: 중복 결과 제거

---

## **🎯 실습 문제**

### **실습 1: Company 노드 생성 (기초)**

**문제**: 다음 기업 정보를 바탕으로 Company 노드를 생성하세요.

| 기업명 | 국가 | 설립연도 | 업종 | 시가총액 |
|--------|------|----------|------|----------|
| 엔비디아 | 미국 | 1993 | 반도체 | 1.8조달러 |
| 삼성전자 | 한국 | 1969 | 전자/반도체 | 300조원 |

**요구사항**:
- `MERGE`를 사용하여 중복 방지
- 모든 속성 정보를 포함할 것
- 두 회사를 하나의 쿼리로 생성

**힌트**:
```cypher
MERGE (c1:Company {
    name: "?",
    country: "?",
    founded: ?,
    ...
})
```

---

### **실습 2: Topic 노드 조회 (기초)**

**문제**: 모든 Topic 노드를 조회하여 이름과 설명을 출력하세요.

**요구사항**:
- `MATCH`와 `RETURN` 사용
- name과 description 속성만 반환

---

### **실습 3: 특정 Topic 필터링 (중급)**

**문제**: name 속성이 "기술 투자"인 Topic 노드를 조회하세요.

**요구사항**:
- `WHERE` 절 사용
- 정확히 일치하는 노드만 반환

---

### **실습 4: 복잡한 관계 네트워크 생성 (고급)**

**문제**: "tech-002" ID를 가진 기사에 다음 관계들을 생성하세요:

**기사 관계 구조**:
```
기사(tech-002)
├── COVERS → AI 반도체 토픽 (관련도: 0.95)
├── COVERS → 기술 투자 토픽 (관련도: 0.9)
├── MENTIONS → 삼성전자 (맥락: "기술 개발 주체", 투자금: "1000억원")
├── DISCUSSES → GAA 기술 (맥락: "핵심 기술", 개발단계: "계획")
└── 삼성전자 INVESTS_IN GAA 기술 (금액: "1000억원", 기간: "2025년까지")
```

**요구사항**:
1. 먼저 Technology 노드 생성 (GAA 기술)
2. 모든 관계를 한 쿼리로 생성
3. 관계 속성 정확히 포함

**힌트**:
```cypher
// 1. GAA 기술 노드 생성
MERGE (t:Technology {
    name: "GAA",
    fullName: "Gate-All-Around",
    ...
})

// 2. 관계 생성
MATCH (a:Article {id: "tech-002"})
MATCH (t1:Topic {name: "AI 반도체"})
...
MERGE (a)-[:COVERS {relevance: ?}]->(t1)
...
```

---

### **실습 5: 관계 기반 검색 (고급)**

**문제**: 특정 기술(GAA)을 다루는 기사를 찾고, 관계 타입도 함께 반환하세요.

**요구사항**:
- Technology 노드와 연결된 Article 찾기
- `type(r)` 함수로 관계 타입 반환
- 기사 제목, ID, 관계 타입, 기술명 모두 반환

**힌트**:
```cypher
MATCH (a:Article)-[r]->(tech:Technology {name: "GAA"})
RETURN a.title, type(r) as relationship_type, tech.name
```

---

## **✅ 솔루션 예시**

### **실습 1 솔루션**

```python
# Company 노드 생성
cypher_query = """
MERGE (c1:Company {
    name: "엔비디아",
    country: "미국",
    founded: 1993,
    sector: "반도체",
    marketCap: "1.8조달러"
})
MERGE (c2:Company {
    name: "삼성전자",
    country: "한국",
    founded: 1969,
    sector: "전자/반도체",
    marketCap: "300조원"
})
RETURN c1, c2
"""
result = graph.query(cypher_query)
print("기업 노드 생성 완료:", result)
```

**설명**:
- `MERGE` 사용으로 중복 방지
- 두 회사를 하나의 트랜잭션으로 처리
- 모든 속성 정보 포함

### **실습 2 솔루션**

```python
# 모든 Topic 조회
cypher_query = """
MATCH (t:Topic)
RETURN t.name, t.description
"""
result = graph.query(cypher_query)
print("Topic 목록:", result)
```

**출력 예시**:
```python
[
    {'t.name': 'AI 반도체', 't.description': '인공지능 가속기 및 관련 기술'},
    {'t.name': '기업 실적', 't.description': '회사의 재무 성과 및 실적'},
    {'t.name': '기술 투자', 't.description': 'R&D 투자 및 기술 개발'}
]
```

### **실습 3 솔루션**

```python
# 특정 Topic 필터링
cypher_query = """
MATCH (t:Topic)
WHERE t.name = "기술 투자"
RETURN t.name, t.description
"""
result = graph.query(cypher_query)
print("기술 투자 Topic:", result)
```

### **실습 4 솔루션**

```python
# Step 1: GAA 기술 노드 생성
cypher_query = """
MERGE (t:Technology {
    name: "GAA",
    fullName: "Gate-All-Around",
    description: "차세대 트랜지스터 기술",
    processNode: "3nm"
})
RETURN t
"""
graph.query(cypher_query)

# Step 2: 복잡한 관계 네트워크 생성
cypher_query = """
// 기본 노드들 매칭
MATCH (a:Article {id: "tech-002"})
MATCH (t1:Topic {name: "AI 반도체"})
MATCH (t2:Topic {name: "기술 투자"})
MATCH (c:Company {name: "삼성전자"})
MATCH (tech:Technology {name: "GAA"})

// 관계 생성
MERGE (a)-[:COVERS {relevance: 0.95}]->(t1)
MERGE (a)-[:COVERS {relevance: 0.9}]->(t2)
MERGE (a)-[:MENTIONS {
    context: "기술 개발 주체",
    investment: "1000억원"
}]->(c)
MERGE (a)-[:DISCUSSES {
    context: "핵심 기술",
    developmentStage: "계획"
}]->(tech)
MERGE (c)-[:INVESTS_IN {
    amount: "1000억원",
    timeline: "2025년까지"
}]->(tech)

RETURN a.title, "완료" as status
"""
result = graph.query(cypher_query)
print("관계 네트워크 생성:", result)
```

**생성된 그래프 시각화**:
```
(Article: tech-002)
├── [COVERS] → (Topic: AI 반도체)
├── [COVERS] → (Topic: 기술 투자)
├── [MENTIONS] → (Company: 삼성전자)
│   └── [INVESTS_IN] → (Technology: GAA)
└── [DISCUSSES] → (Technology: GAA)
```

### **실습 5 솔루션**

```python
# 특정 기술을 다루는 기사 찾기
cypher_query = """
MATCH (a:Article)-[r]->(tech:Technology {name: "GAA"})
RETURN a.title, a.id, type(r) as relationship_type, tech.name
"""
result = graph.query(cypher_query)
print("GAA 기술 관련 기사:", result)
```

**출력 예시**:
```python
[
    {
        'a.title': '삼성전자, 차세대 AI 반도체 개발 계획 발표',
        'a.id': 'tech-002',
        'relationship_type': 'DISCUSSES',
        'tech.name': 'GAA'
    }
]
```

---

## **🚀 실무 활용 예시**

### **1. 뉴스 추천 시스템**

**시나리오**: 사용자가 읽은 기사와 유사한 주제의 기사를 추천

```python
def recommend_similar_articles(article_id: str, limit: int = 5):
    """
    특정 기사와 같은 토픽을 다루는 다른 기사 추천
    """
    cypher_query = f"""
    MATCH (a1:Article {{id: $article_id}})-[:COVERS]->(t:Topic)<-[:COVERS]-(a2:Article)
    WHERE a1 <> a2
    RETURN DISTINCT a2.title, a2.id, count(t) as common_topics
    ORDER BY common_topics DESC
    LIMIT $limit
    """

    result = graph.query(
        cypher_query,
        params={"article_id": article_id, "limit": limit}
    )
    return result

# 사용 예시
recommendations = recommend_similar_articles("tech-001", limit=3)
print("추천 기사:", recommendations)
```

**출력**:
```python
[
    {'a2.title': '삼성전자, 차세대 AI 반도체...', 'a2.id': 'tech-002', 'common_topics': 1}
]
```

### **2. 기업 영향력 분석**

**시나리오**: 특정 기업이 언급된 모든 기사와 관련 인물 파악

```python
def analyze_company_coverage(company_name: str):
    """
    특정 기업의 미디어 노출도 및 관련 인물 분석
    """
    cypher_query = """
    MATCH (c:Company {name: $company_name})
    OPTIONAL MATCH (a:Article)-[:MENTIONS]->(c)
    OPTIONAL MATCH (p:Person)-[:WORKS_FOR]->(c)

    RETURN
        c.name as company,
        count(DISTINCT a) as article_count,
        collect(DISTINCT a.title)[0..5] as sample_articles,
        collect(DISTINCT p.name) as key_people
    """

    result = graph.query(cypher_query, params={"company_name": company_name})
    return result[0] if result else None

# 사용 예시
analysis = analyze_company_coverage("엔비디아")
print(f"기업: {analysis['company']}")
print(f"관련 기사 수: {analysis['article_count']}")
print(f"주요 인물: {analysis['key_people']}")
```

### **3. 토픽 트렌드 분석**

**시나리오**: 시간별 토픽 언급 빈도 분석

```python
def topic_trend_analysis(topic_name: str):
    """
    특정 토픽의 시간별 기사 발행 추이 분석
    """
    cypher_query = """
    MATCH (a:Article)-[:COVERS]->(t:Topic {name: $topic_name})
    WITH a.publishDate as date, count(a) as article_count
    RETURN date, article_count
    ORDER BY date DESC
    """

    result = graph.query(cypher_query, params={"topic_name": topic_name})
    return result

# 사용 예시
trend = topic_trend_analysis("AI 반도체")
print("토픽 트렌드:", trend)
```

### **4. 데이터 정제 및 중복 제거**

**시나리오**: 같은 회사명의 다양한 표기 통합

```python
def merge_duplicate_companies():
    """
    유사한 회사명을 하나로 통합
    """
    # 삼성전자의 다양한 표기 통합
    cypher_query = """
    MATCH (c1:Company)
    WHERE c1.name IN ["Samsung Electronics", "삼성전자", "Samsung"]

    WITH collect(c1) as companies
    WHERE size(companies) > 1

    // 첫 번째 노드를 메인으로 사용
    WITH companies[0] as main, companies[1..] as duplicates

    // 중복 노드의 관계를 메인 노드로 이전
    UNWIND duplicates as dup
    MATCH (dup)-[r]->(target)
    MERGE (main)-[r2:typeof(r)]->(target)
    SET r2 = properties(r)
    DELETE r, dup

    RETURN main.name as merged_company
    """

    result = graph.query(cypher_query)
    return result
```

---

## **📖 참고 자료**

### **공식 문서**
- [Neo4j 공식 문서](https://neo4j.com/docs/)
- [Cypher 쿼리 언어 가이드](https://neo4j.com/docs/cypher-manual/current/)
- [LangChain Neo4j 통합](https://python.langchain.com/docs/integrations/graphs/neo4j_cypher)
- [Neo4j AuraDB 문서](https://neo4j.com/docs/aura/)

### **학습 자료**
- [Neo4j GraphAcademy](https://graphacademy.neo4j.com/) - 무료 온라인 강좌
- [Cypher Query Language Tutorial](https://neo4j.com/developer/cypher/) - 공식 튜토리얼
- [Graph Data Science Library](https://neo4j.com/docs/graph-data-science/current/) - 그래프 알고리즘

### **커뮤니티**
- [Neo4j Community Forum](https://community.neo4j.com/)
- [Neo4j Discord](https://discord.gg/neo4j)
- [Stack Overflow - Neo4j 태그](https://stackoverflow.com/questions/tagged/neo4j)

### **실전 예제**
- [Neo4j Use Cases](https://neo4j.com/use-cases/) - 산업별 활용 사례
- [Neo4j Sandbox](https://sandbox.neo4j.com/) - 무료 샘플 데이터셋

---

## **🎓 학습 정리**

### **Part 1에서 배운 내용**

1. **Knowledge Graph 개념**: 노드, 관계, 속성으로 구성된 그래프 데이터 모델
2. **Neo4j AuraDB 설정**: 클라우드 인스턴스 생성 및 Python 연결
3. **Cypher 기본 문법**: CREATE, MERGE, MATCH, WHERE, RETURN
4. **관계 모델링**: 의미있는 관계 설계 및 복잡한 그래프 구조 구축
5. **패턴 매칭**: 관계 기반 검색 및 다중 홉 쿼리

### **다음 단계 (Part 2 예고)**

1. **데이터 수정 및 삭제**: UPDATE, DELETE, DETACH DELETE
2. **LangChain 통합**: 자연어 → Cypher 쿼리 자동 변환
3. **배치 데이터 처리**: 대량 데이터 로딩 및 최적화
4. **그래프 분석**: 중심성, 커뮤니티 탐지, 경로 찾기
5. **벡터 검색 통합**: 임베딩 기반 유사도 검색

---

**Part 1 완료!** 🎉

이제 Neo4j의 기초와 Cypher 쿼리를 활용한 데이터 모델링을 할 수 있습니다. Part 2에서는 더 고급 기능과 LangChain 통합을 다룹니다!
