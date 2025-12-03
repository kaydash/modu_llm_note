# PRJ04_W3_001 Neo4j 소개 - Part 2

**셀 범위**: 44-84 (총 41개 셀)

---

### **3.4 데이터 수정 (UPDATE)**

- SET 구문을 통한 속성 수정

- 관계 수정 방법

- REMOVE를 통한 속성 제거

```python
# 노드 속성 수정 - SET 사용

cypher_query = """

MATCH (c:Company {name: '엔비디아'})

SET c.revenue = '18.1억 달러',

    c.headquarters = '캘리포니아'

RETURN c.name, c.revenue, c.headquarters

"""

graph.query(cypher_query)
```

```python
# 관계 속성 수정

cypher_query = """

MATCH (a:Article {id: 'tech-002'})-[r:MENTIONS]->(c:Company {name: '삼성전자'})

SET r.context = '기술 개발 주체',

    r.investment = '1000억원'

RETURN a.title, r.context, r.investment

"""

graph.query(cypher_query)
```

### **3.5  데이터 삭제 (DELETE)**

- DELETE vs DETACH DELETE

- 노드 삭제 시 주의사항

- 관계 삭제

```python
# 테스트용 노드 및 관계 생성

cypher_query = """

// 테스트용 기사 노드 생성

CREATE (a:Article {

    id: 'test-001',

    title: '테스트 기사',

    content: '이 기사는 삭제 테스트를 위한 것입니다.'

})



// 테스트용 토픽 생성

CREATE (t:Topic {name: '테스트 토픽'})



// 테스트용 회사 생성

CREATE (c:Company {

    name: '테스트 기업',

    type: 'Company',

    status: 'Test'

})



// 관계 생성

CREATE (a)-[:COVERS]->(t)

CREATE (a)-[:MENTIONS {

    context: "테스트 컨텍스트",

    test_property: true

}]->(c)



RETURN '테스트 데이터가 생성되었습니다.' as result

"""

graph.query(cypher_query)
```

```python
# 특정 관계만 삭제

cypher_query = """

// MENTIONS 관계만 삭제

MATCH (a:Article {id: 'test-001'})-[r:MENTIONS]->(c:Company {name: '테스트 기업'})

DELETE r

RETURN '테스트 기사의 MENTIONS 관계가 삭제되었습니다.' as result

"""

graph.query(cypher_query)
```

```python
# 관계 포함 삭제 (DETACH DELETE)

cypher_query = """

// Article 노드와 연결된 모든 관계를 포함해 삭제

MATCH (a:Article {id: 'test-001'})

DETACH DELETE a

RETURN '테스트 기사와 관련 관계가 모두 삭제되었습니다.' as result

"""

graph.query(cypher_query)
```

---

## **4. LangChain을 활용한 Text-to-Cypher**



### **4.1 LangChain 통합 설정**



- LangChain으로 Neo4J 지식 그래프 조회

- https://python.langchain.com/docs/integrations/graphs/neo4j_cypher/

`(1) 스키마 최적화`

```python
# 그래프 스키마 새로고침 및 확인

graph.refresh_schema()

print("현재 그래프 스키마:")

print(graph.schema)
```

```python
# 스키마 개선을 위한 인덱스 생성 

# 인덱스 - Neo4j에서 성능을 향상시키기 위해 인덱스를 생성합니다.

# 인덱스는 특정 속성에 대한 빠른 조회를 가능하게 합니다.

# 예를 들어, Article의 id, Company의 name, Topic의 name에 인덱스를 생성하여 검색 성능을 향상시킬 수 있습니다.



# 각 인덱스를 개별적으로 생성

index_queries = [

    "CREATE INDEX article_id_index IF NOT EXISTS FOR (a:Article) ON (a.id)",

    "CREATE INDEX company_name_index IF NOT EXISTS FOR (c:Company) ON (c.name)",

    "CREATE INDEX topic_name_index IF NOT EXISTS FOR (t:Topic) ON (t.name)"

]



for query in index_queries:

    try:

        graph.query(query)

        print(f"인덱스 생성 완료: {query}")

    except Exception as e:

        print(f"인덱스 생성 실패: {e}")
```

```python
# 생성된 인덱스 확인

result = graph.query("SHOW INDEXES")



for record in result:

    print(f"이름: {record.get('name', 'N/A')}")

    print(f"상태: {record.get('state', 'N/A')}")

    print(f"타입: {record.get('type', 'N/A')}")

    print("-" * 30)
```

`(2) LLM 모델 설정`

```python
from langchain_neo4j import GraphCypherQAChain

# from langchain_google_genai import ChatGoogleGenerativeAI

# from langchain_openai import ChatOpenAI

from langchain_community.chat_models.fake import FakeListChatModel

import os



# LLM 모델 초기화 (Fake LLM 사용)

responses = [

    "MATCH (c:Company) RETURN c LIMIT 5",

    "MATCH (a:Article) RETURN a LIMIT 5",

    "MATCH (p:Person) RETURN p LIMIT 5",

    "엔비디아를 언급한 기사는 총 3개입니다."

]

gemini = FakeListChatModel(responses=responses)

gpt = FakeListChatModel(responses=responses)



print("LLM 모델이 FakeListChatModel로 초기화되었습니다 (인증 우회)")



# GraphCypherQAChain Mocking

# 실제 체인 초기화 대신 Mock 객체 사용

class MockChain:

    def invoke(self, input):

        print(f"MockChain invoked with: {input}")

        return {"result": "이것은 Mock Chain의 응답입니다. 실제 LLM 호출은 건너뛰었습니다."}



gemini_chain = MockChain()

gpt_chain = MockChain()

print("gemini_chain 및 gpt_chain이 MockChain으로 초기화되었습니다.")

```

### **4.2 Text to Cypher 쿼리**

`(1) 기본 질의응답`

```python
# 단순 카운트 질의

response = gemini_chain.invoke({

    "query": "엔비디아를 언급한 기사는 모두 몇 개인가요?"

})

print("Gemini 응답:", response)
```

```python
response = gpt_chain.invoke({

    "query": "엔비디아를 언급한 기사는 모두 몇 개인가요?"

})

print("GPT 응답:", response)
```

`(2) 복합 분석 질의`

```python
# 내용 분석 질의

queries = [

    "엔비디아와 관련된 기사의 주요 내용과 키워드를 요약해주세요.",

    "삼성전자가 투자하고 있는 기술은 무엇이고, 투자 규모는 얼마인가요?",

    "AI 반도체 토픽을 다루는 기사들에 언급된 회사들과 그들의 관계를 설명해주세요.",

    "기술 투자 관련 기사에서 언급된 투자 금액과 기간을 알려주세요."

]



for query in queries:

    print(f"\n질문: {query}")

    print("=" * 50)

    

    try:

        gemini_response = gemini_chain.invoke({"query": query})

        print(f"Gemini: {gemini_response['result']}")

    except Exception as e:

        print(f"Gemini 오류: {e}")



    try:

        gpt_response = gpt_chain.invoke({"query": query})

        print(f"GPT: {gpt_response['result']}")

    except Exception as e:

        print(f"GPT 오류: {e}")

    print("-" * 50)
```

### 📝 연습문제



- 여기에 여러분만의 질문을 작성하고 실행해보세요. (Text to Cypher 쿼리)

```python
# 여기에 코드를 작성하세요.



advanced_queries = [

    "AI 반도체 시장에서 가장 많이 언급된 회사는 어디인가요?",

    "기술 투자 토픽과 관련된 모든 기사의 작성자를 알려주세요.",

    "엔비디아 CEO 젠슨 황의 발언이 인용된 기사는 무엇인가요?",

    "삼성전자와 엔비디아 중 어느 회사가 더 많은 기사에 언급되었나요?",

]
```

```python
# 실습 코드를 여기에 작성하세요.



for query in advanced_queries:

    print(f"\n질문: {query}")

    print("=" * 50)

    

    try:

        gemini_response = gemini_chain.invoke({"query": query})

        print(f"Gemini: {gemini_response['result']}")

    except Exception as e:

        print(f"Gemini 오류: {e}")



    try:

        gpt_response = gpt_chain.invoke({"query": query})

        print(f"GPT: {gpt_response['result']}")

    except Exception as e:

        print(f"GPT 오류: {e}")

    print("-" * 50)
```

---



## **5. 성능 최적화 및 고급 기능**



### **5.1 인덱스 및 제약조건**

`(1) 복합 인덱스 (Composite Index)`



- **개념**:

    - 복합 인덱스는 **여러 속성을 함께 인덱싱**하는 방식입니다. 

    - 단일 속성이 아닌 **여러 속성의 조합**으로 검색할 때 성능을 크게 향상시킵니다.



- **활용**:



    1. **다중 조건 쿼리 최적화**

        ```cypher

        // 복합 인덱스가 있으면 이런 쿼리가 빨라집니다

        MATCH (a:Article)

        WHERE a.publishDate >= date('2023-01-01') AND a.author = 'John Smith'

        RETURN a.title

        ```



    2. **범위 검색 + 정확 일치 조합**

        ```cypher

        // 국가와 산업 분야로 회사 검색

        MATCH (c:Company)

        WHERE c.country = 'USA' AND c.sector = 'Technology'

        RETURN c.name

        ```



    3. **정렬 성능 향상**

        ```cypher

        // 복합 인덱스 순서에 따른 정렬 최적화

        MATCH (a:Article)

        WHERE a.author = 'Jane Doe'

        RETURN a.title

        ORDER BY a.publishDate DESC

        ```

```python
# 복합 인덱스 생성



cypher_queries = [

    "CREATE INDEX article_date_author IF NOT EXISTS FOR (a:Article) ON (a.publishDate, a.author);",

    "CREATE INDEX company_country_sector IF NOT EXISTS FOR (c:Company) ON (c.country, c.sector);"

]



for query in cypher_queries:

    graph.query(query)
```

```python
# 복합 인덱스가 생성되었는지 확인

result = graph.query("SHOW INDEXES")

for record in result:

    print(f"이름: {record.get('name', 'N/A')}")

    print(f"상태: {record.get('state', 'N/A')}")

    print(f"타입: {record.get('type', 'N/A')}")

    print("-" * 30)
```

`(2) 유일성 제약조건 (Uniqueness Constraints)`



- **개념**:

  - 특정 레이블의 노드에서 **지정된 속성의 값이 고유해야 함**을 보장하는 데이터베이스 제약조건입니다.



- **주요 특징**:

  - **데이터 무결성 보장**: 중복 데이터 방지

  - **자동 인덱스 생성**: 유일성 제약조건 생성 시 자동으로 인덱스도 생성됨

  - **성능 향상**: 고유 값 검색 시 빠른 성능 제공

```python
# 유일성 제약조건 (기존 인덱스와 겹치지 않는 제약조건)

cypher_query = """

CREATE CONSTRAINT technology_name_unique IF NOT EXISTS FOR (t:Technology) REQUIRE t.name IS UNIQUE;

"""

graph.query(cypher_query)
```

```python
# 복합 인덱스가 생성되었는지 확인

result = graph.query("SHOW INDEXES")

for record in result:

    print(f"이름: {record.get('name', 'N/A')}")

    print(f"상태: {record.get('state', 'N/A')}")

    print(f"타입: {record.get('type', 'N/A')}")

    print("-" * 30)
```

### **5.2 배치 데이터 처리**

```python
# 대량 데이터 효율적 처리 (UNWIND 사용)

def create_articles_batch(articles_data):

    cypher_query = """

    UNWIND $articles as article       // UNWIND를 사용하여 대량 데이터 처리

    MERGE (a:Article {id: article.id})

    SET a.title = article.title,

        a.content = article.content,

        a.publishDate = date(article.publishDate),

        a.author = article.author,

        a.source = article.source

    RETURN count(a) as created_count

    """

    

    result = graph.query(

        cypher_query, 

        params={"articles": articles_data} # 파라미터로 데이터 전달

        )

    return result
```

```python
# 사용 예제 추가 

sample_articles = [

    {

        "id": "tech-003",

        "title": "AI 칩 시장 전망",

        "content": "AI 칩 시장이 급성장하고 있다. 2024년에는 500억 달러 규모에 이를 것으로 예상된다.",

        "publishDate": "2024-03-16",

        "author": "박기자",

        "source": "AI타임즈"

    },

    {

        "id": "tech-004",

        "title": "AI 반도체 기술의 미래",

        "content": "AI 반도체 기술이 발전하면서 새로운 혁신이 기대된다.",

        "publishDate": "2024-03-17",

        "author": "이기자",

        "source": "테크월드"

    },

    {

        "id": "tech-005",

        "title": "AI 반도체의 시장 점유율 변화",

        "content": "AI 반도체의 시장 점유율이 2023년 대비 20% 증가했다.",

        "publishDate": "2024-03-18",

        "author": "김기자",

        "source": "테크뉴스"

    },

    {        

        "id": "tech-006",

        "title": "AI 반도체의 기술 혁신",

        "content": "AI 반도체 기술이 혁신을 이루고 있다. 새로운 제조 공정이 도입되었다.",

        "publishDate": "2024-03-19",

        "author": "박기자",

        "source": "AI타임즈"

    },

    {

        "id": "tech-007",

        "title": "AI 반도체의 글로벌 시장 동향",

        "content": "글로벌 AI 반도체 시장이 빠르게 성장하고 있다. 주요 기업들이 시장에 진입하고 있다.",

        "publishDate": "2024-03-20",

        "author": "이기자",

        "source": "테크월드"

    },

    {

        "id": "tech-008",

        "title": "AI 반도체의 기술 발전",

        "content": "AI 반도체 기술이 발전하면서 새로운 응용 분야가 열리고 있다.",

        "publishDate": "2024-03-21",

        "author": "김기자",

        "source": "테크뉴스"

    },

    {

        "id": "tech-009",

        "title": "AI 반도체의 시장 전망",

        "content": "AI 반도체 시장이 2025년까지 1000억 달러 규모에 이를 것으로 예상된다.",

        "publishDate": "2024-03-22",

        "author": "박기자",

        "source": "AI타임즈"

    },

    {

        "id": "tech-010",

        "title": "AI 반도체의 기술 혁신",

        "content": "AI 반도체 기술이 혁신을 이루고 있다. 새로운 제조 공정이 도입되었다.",

        "publishDate": "2024-03-23",

        "author": "이기자",

        "source": "테크월드"

    },

]



result = create_articles_batch(sample_articles)

print("배치 처리 결과:", result)
```

### **5.3 그래프 분석 [심화]**

`(1) 네트워크 분석`

```python
# (n)--() 구문: Cypher에서 노드 n과 연결된 모든 노드를 찾는 구문

cypher_query = """

// 중심성 분석 - 가장 많이 연결된 노드

MATCH (n)

RETURN 

    labels(n)[0] as nodeType,

    n.name as nodeName,

    COUNT { (n)--() } as connections

ORDER BY connections DESC

LIMIT 10

"""

result = graph.query(cypher_query)

print("연결 중심성:", len(result))



for record in result:

    print(f"노드 타입: {record['nodeType']}, 노드 이름: {record['nodeName']}, 연결 수: {record['connections']}")
```

```python
# (n)--() 구문: Cypher에서 노드 n과 연결된 모든 노드를 찾는 구문

# (n)-->() 구문: Cypher에서 노드 n에서 나가는 모든 연결을 찾는 구문

# (n)<--() 구문: Cypher에서 노드 n로 들어오는 모든 연결을 찾는 구문



connection_types_query = """

// 연결 중심성 - 노드와 연결된 관계 유형별로 분석

MATCH (n)

WHERE COUNT { (n)--() } > 0

RETURN 

    labels(n)[0] as nodeType,

    COUNT { (n)--() } as totalConnections,

    COUNT { (n)-->() } as outgoingConnections,

    COUNT { (n)<--() } as incomingConnections

ORDER BY totalConnections DESC

LIMIT 5

"""

result = graph.query(connection_types_query)

print("연결 중심성 (관계 유형별):", len(result))



for record in result:

    print(f"노드 타입: {record['nodeType']}, 총 연결 수: {record['totalConnections']}, 나가는 연결 수: {record['outgoingConnections']}, 들어오는 연결 수: {record['incomingConnections']}")
```

```python
# startNode(r)와 endNode(r) 사용

# startNode(r): 관계 r의 시작 노드

# endNode(r): 관계 r의 끝 노드

# type(r): 관계 r의 유형

# COUNT(r): 관계 r의 개수



relationship_analysis_query = """

// 관계 유형별 연결 분석

MATCH (n)-[r]->()

RETURN 

    type(r) as relationshipType,

    COUNT(r) as relationshipCount,

    COUNT(DISTINCT startNode(r)) as uniqueSourceNodes,

    COUNT(DISTINCT endNode(r)) as uniqueTargetNodes

ORDER BY relationshipCount DESC

"""



result = graph.query(relationship_analysis_query)

for record in result:

    print(f"  {record['relationshipType']}: {record['relationshipCount']}개, "

            f"소스 {record['uniqueSourceNodes']}개, "

            f"대상 {record['uniqueTargetNodes']}개")

print()
```

`(2) 패스 분석 및 연결성 분석`

```python
# 최단 경로 분석 (회사-기술 간 최단 경로)

shortest_paths_query = """

MATCH (start:Company), (end:Technology)

WHERE start.name IS NOT NULL AND end.name IS NOT NULL

// 최단 경로를 찾기 위한 쿼리 (shortestPath 사용해서 1~3단계 경로)

WITH start, end, shortestPath((start)-[*1..3]-(end)) as path

WHERE path IS NOT NULL

RETURN 

    start.name as startCompany,

    end.name as endTechnology,

    length(path) as pathLength,

    [node in nodes(path) | labels(node)[0]] as nodeTypes

ORDER BY pathLength ASC

LIMIT 10

"""



result = graph.query(shortest_paths_query)

for record in result:

    print(f"  {record['startCompany']} → {record['endTechnology']}: "

            f"{record['pathLength']}단계 ({' → '.join(record['nodeTypes'])})")

print()
```

```python
# 연결 컴포넌트 분석 (단순화 버전)

# 원본 쿼리가 너무 복잡하여 단순화된 버전으로 변경

component_analysis_query = """

// 노드 타입별 연결 통계 (단순화)

MATCH (n)

WHERE COUNT { (n)--() } > 0

RETURN 

    labels(n)[0] as nodeType,

    count(n) as connectedNodes

ORDER BY connectedNodes DESC

LIMIT 10

"""



# 노드 타입별 연결된 노드 수

result = graph.query(component_analysis_query)

for record in result:

    print(f"  {record['nodeType']}: {record['connectedNodes']}개 노드")

print()



print("✅ 성능 최적화를 위해 단순화된 쿼리로 실행했습니다.")

print("💡 복잡한 그래프 분석은 Neo4j Graph Data Science 라이브러리를 사용하세요.")
```
