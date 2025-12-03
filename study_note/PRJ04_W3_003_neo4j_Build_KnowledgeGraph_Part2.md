# Knowledge Graph 구축 - Part 2: Text-to-Cypher와 벡터 검색

**학습 자료**: PRJ04_W3_003 - 자연어 질의와 의미론적 검색 구현

---

## 📚 학습 목표

이 실습을 완료하면 다음을 할 수 있습니다:

1. **Text-to-Cypher 구현**: 자연어 질문을 Cypher 쿼리로 자동 변환
2. **프롬프트 엔지니어링**: 정확한 쿼리 생성을 위한 프롬프트 최적화
3. **벡터 검색 구현**: 임베딩 기반 의미론적 검색 시스템 구축
4. **하이브리드 검색**: 그래프 검색과 벡터 검색 결합
5. **실무 QA 시스템**: 완전한 질의응답 시스템 구현

---

## 🔑 핵심 개념

### Text-to-Cypher

**Text-to-Cypher**는 자연어 질문을 Cypher 쿼리 언어로 자동 변환하는 기술입니다.

#### 작동 원리

1. **스키마 이해**: 그래프 데이터베이스의 구조 파악
2. **질문 분석**: LLM을 통한 자연어 질문 해석
3. **쿼리 생성**: 질문을 Cypher 쿼리로 변환
4. **실행 및 검증**: 생성된 쿼리 실행 및 결과 확인
5. **답변 생성**: 쿼리 결과를 자연어 답변으로 변환

#### Text-to-Cypher의 장점

- **접근성**: 비기술자도 그래프 데이터 활용 가능
- **생산성**: 복잡한 쿼리 작성 시간 단축
- **정확성**: LLM의 언어 이해 능력 활용

### 벡터 검색 (Vector Search)

**벡터 검색**은 텍스트를 임베딩 벡터로 변환하여 의미론적 유사도를 계산하는 검색 방식입니다.

#### 작동 원리

1. **임베딩 생성**: 텍스트를 고차원 벡터로 변환
2. **벡터 인덱스**: 효율적인 유사도 검색을 위한 인덱스 생성
3. **유사도 계산**: 코사인 유사도 등으로 관련성 측정
4. **결과 반환**: 가장 유사한 상위 k개 결과 반환

#### 하이브리드 검색

- **벡터 검색**: 의미론적 유사도 기반
- **그래프 검색**: 관계 기반 탐색
- **결합**: 두 방식의 장점을 모두 활용

---

## 🛠 환경 설정

### Part 1 선행 학습 필요

이 Part 2는 Part 1의 내용을 기반으로 합니다:
- Neo4j Desktop 환경 구축
- 정형 데이터를 지식 그래프로 변환
- 기본 인덱스 생성

### 추가 준비 사항

Part 1에서 생성한 지식 그래프가 Neo4j에 저장되어 있어야 합니다.

---

## 💻 단계별 구현


## 5. Text to Cypher
- LangChain으로 Neo4J 지식 그래프 조회
- Ollama 모델 사용

`(1) GraphCypherQAChain 설정`

```python
from langchain_openai import ChatOpenAI
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph

# LangChain 도구 활용 - LLM 및 그래프 객체 초기화
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database=os.getenv("NEO4J_DATABASE"),
    enhanced_schema=True,
    refresh_schema=True,  # 스키마를 최신 상태로 유지
)

# LangChain 도구 활용 - GraphCypherQAChain 객체 초기화
cypher_chain = GraphCypherQAChain.from_llm(
    llm=llm, 
    graph=graph, 
    allow_dangerous_requests=True,
    verbose=True,)
```

`(2) Text to Cypher - DB 조회`

```python
cypher_chain.invoke({"query": "삼성자산운용의 KODEX 관련 상품은 무엇인가요?"})
```

`(3) 출력 갯수를 지정 (top k)`

```python
cypher_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph, 
    allow_dangerous_requests=True,
    verbose=True,
    top_k=3,
)

cypher_chain.invoke({"query": "삼성자산운용의 KODEX 관련 상품은 무엇인가요?"})
```

`(4) 중간 결과를 포함하여 출력`

```python
cypher_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph, 
    allow_dangerous_requests=True,
    verbose=True,
    top_k=3,
    return_intermediate_steps=True
)

cypher_chain.invoke({"query": "삼성자산운용의 KODEX 관련 상품은 무엇인가요?"})
```

`(5) cypher 쿼리 결과를 직접 출력 (LLM 답변 생성하지 않음)`

```python
cypher_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph, 
    allow_dangerous_requests=True,
    verbose=True,
    top_k=3,
    return_intermediate_steps=True,
    return_direct=True
)

cypher_chain.invoke({"query": "삼성자산운용의 KODEX 관련 상품은 무엇인가요?"})
```

`(6) cypher 쿼리 생성하는 모델과 최종 답변 생성 모델을 별도 적용`

```python
# 원격 Ollama 서버 연결 (test_ollama.py 방식)
from langchain_openai import ChatOpenAI

# Ollama 서버 설정
OLLAMA_BASE_URL = "http://littletask.kro.kr:1410/v1"
OLLAMA_API_KEY = "Task123!"
OLLAMA_MODEL = "phi4-mini:3.8b"

# OpenAI 호환 API를 사용하여 원격 Ollama 서버에 연결
second_llm = ChatOpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key=OLLAMA_API_KEY,
    model=OLLAMA_MODEL,
    temperature=0.0
)

print("✅ second_llm 정의 완료")

# 연결 테스트 (선택사항)
try:
    test_response = second_llm.invoke("안녕하세요")
    print(f"✅ Ollama 서버 연결 성공: {test_response.content[:50]}")
except Exception as e:
    print(f"⚠️ Ollama 연결 테스트 실패: {str(e)[:100]}")
    print("💡 second_llm은 정의되었지만, 실제 사용 시 오류가 발생할 수 있습니다.")
```

`(7) 사용자 프롬프트를 직접 지정`

```python
from langchain_core.prompts.prompt import PromptTemplate

# Cypher 생성을 위한 프롬프트
CYPHER_GENERATION_TEMPLATE = """Task: Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}

Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

Examples: Here are a few examples of generated Cypher statements for particular questions:
# 수익률이 가장 높은 5개 ETF는 무엇인가요?
MATCH (e:ETF)
WHERE e.yearReturn IS NOT NULL
RETURN e.name, e.code, e.yearReturn
ORDER BY e.yearReturn DESC
LIMIT 5

# 미래에셋자산운용에서 운용하는 모든 ETF를 보여주세요.
MATCH (c:Company {{name: '미래에셋자산운용'}})-[:MANAGES]->(e:ETF)
RETURN e.name, e.code, e.category, e.yearReturn
ORDER BY e.yearReturn DESC

# 국내 주식형 ETF 중 순자산총액이 가장 큰 상위 10개는 무엇인가요?
MATCH (c:Category {{name: '국내 주식형'}})<-[:BELONGS_TO]-(e:ETF)
WHERE e.netAsset IS NOT NULL
RETURN e.name, e.code, e.netAsset, e.yearReturn
ORDER BY e.netAsset DESC
LIMIT 10

# 순자산총액이 1조원 이상인 ETF의 평균 수익률은 얼마인가요?
MATCH (e:ETF)
WHERE e.netAsset >= 1000000000000 AND e.yearReturn IS NOT NULL
RETURN AVG(e.yearReturn) AS averageReturn, COUNT(e) AS etfCount

# 운용사별 평균 ETF 수익률은 어떻게 되나요?
MATCH (c:Company)-[:MANAGES]->(e:ETF)
WHERE e.yearReturn IS NOT NULL
WITH c.name AS company, AVG(e.yearReturn) AS avgReturn, COUNT(e) AS etfCount
ORDER BY avgReturn DESC
RETURN company, avgReturn, etfCount

The question is:
{question}"""

# 결과 처리를 위한 QA 프롬프트
QA_TEMPLATE = """
You are a financial assistant providing information about ETF databases in an easy-to-understand manner in Korean.
Based on the information obtained from the ETF database, answer the question in a clear and informative way.

Question: {question}
Search Results: {context}

Respond only with relevant information in a natural, conversational tone. Convert numerical data such as returns and net asset values into appropriate units (%, million USD, billion USD, etc.) to make them easily readable.
Avoid content that could be interpreted as investment advice or recommendations, and only convey objective facts.
"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], 
    template=CYPHER_GENERATION_TEMPLATE
)

QA_PROMPT = PromptTemplate(
    input_variables=["question", "context"], 
    template=QA_TEMPLATE
)

# Chain 생성 - input_key와 output_key를 명시적으로 설정
cypher_chain = GraphCypherQAChain.from_llm(
    cypher_llm=llm,
    qa_llm=second_llm,
    graph=graph, 
    allow_dangerous_requests=True,
    verbose=True,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
    qa_prompt=QA_PROMPT,
    input_key="question",  
    output_key="result"
)

# 쿼리 실행 - 입력 키를 "question"으로 변경
cypher_chain.invoke({"question": "삼성자산운용의 KODEX 관련 상품은 무엇인가요?"})
```

## 6. 벡터 검색(Semantic Search) 

`(1) Graph DB를 초기화`

```python
from langchain_openai import OpenAIEmbeddings
from langchain_neo4j import Neo4jVector

embeddings = OpenAIEmbeddings(model="text-embedding-3-small") 

existing_graph = Neo4jVector.from_existing_index(
    embeddings,
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    index_name="etf_vector_index",  # 기존에 생성한 벡터 인덱스 이름
    node_label="ETF",  # 노드 레이블
    text_node_property="name",  # 텍스트 속성 (ETF 이름)
    embedding_node_property="embedding",  # 임베딩 속성
    retrieval_query="""
    WITH node, score
    WITH node, score,
         "ETF 종목: " + node.name + 
         CASE WHEN node.company IS NOT NULL THEN "\n- 운용사: " + node.company ELSE "" END +
         CASE WHEN node.baseIndex IS NOT NULL THEN "\n- 기초지수: " + node.baseIndex ELSE "" END +
         CASE WHEN node.category IS NOT NULL THEN "\n- 카테고리: " + node.category ELSE "" END AS combined_text
         
    RETURN combined_text AS text,   // 텍스트 (LangChain Document 객체의 page_content 속성)
           score,  // 유사도 점수 
           {
               etf_name: node.name,
               company: node.company,
               base_index: node.baseIndex,
               category: node.category,
               listing_date: node.listingDate,
               total_fee: node.totalFee,
               etf_code: node.code
           } AS metadata    // 메타데이터 (LangChain Document 객체의 metadata 속성)
    """
)

# 벡터 검색 쿼리 실행
query = "삼성자산운용의 KODEX 관련 상품은 무엇인가요?" # 검색할 쿼리
similar_docs = existing_graph.similarity_search_with_score(query, k=5) # 유사도 상위 5개 문서 검색

for doc, score in similar_docs:
    print(f"문서 내용: {doc.page_content}, 유사도 점수: {score}")
```

`(2) Graph DB 검색을 수행하는 함수`

```python
# Cypher 쿼리 실행 및 관련 ETF 노드 검색을 처리하는 함수 
def execute_query_and_get_etf_data(query, k=5):
    # 벡터 검색 쿼리 실행
    similar_docs = existing_graph.similarity_search_with_score(query, k) # 유사도 상위 k개 문서 검색
    similar_doc_names = [doc.metadata['etf_name'] for doc, _ in similar_docs] # 유사한 문서의 종목명 추출

    # 유사한 문서의 종목명을 기반으로 ETF 노드 검색
    etf_query = """
    MATCH (e:ETF)   // 모든 ETF 노드 검색
    WHERE e.name IN $similar_doc_names   // 유사한 문서의 종목명과 일치하는 ETF 노드 검색
    RETURN e
    """
    etf_results = graph.query(
        etf_query, 
        params={"similar_doc_names": similar_doc_names}
    ) # ETF 노드 검색
    etf_results = [etf['e'] for etf in etf_results] # 검색된 ETF 노드 반환

    # 문자열 포맷팅
    result = ""
    for etf in etf_results:
        result += f"# 종목명: {etf['name']} (운용사: {etf['company']})\n"
        result += f"- 기초지수: {etf['baseIndex']}, 수익률: {etf['yearReturn']}, 순자산총액: {etf['netAsset']}\n"
        result += f"- 추적오차: {etf['trackingError']}, 변동성: {etf['volatility']}, 복제방법: {etf['replicationMethod']}, 총보수: {etf['totalFee']}, 과세유형: {etf['taxType']}\n"
        result += "-" * 3 + "\n"
    return result


# 쿼리 실행 및 관련 ETF 노드 검색
query = "삼성자산운용의 KODEX 관련 상품은 무엇인가요?" 
etf_data = execute_query_and_get_etf_data(query, k=5) # ETF 노드 검색
print(etf_data)
```

`(3) LCEL 사용하여 RAG 체인을 구성`

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# LLM
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)

# Prompt
template = '''당신은 ETF 데이터 분석 전문가로서 오직 주어진 정보에 기반하여 객관적이고 정확한 답변을 제공합니다.

주어진 정보:
{context}

질문: {question}

답변 작성 지침:
1. 제공된 정보에 명시된 사실만 사용하세요.
2. 수익률은 '%', 순자산총액은 '억 원' 또는 '조 원' 단위로 표시하세요.
3. 투자 조언이나 추천으로 해석될 수 있는 표현은 사용하지 마세요.
4. 제공된 정보에 없는 내용은 "제공된 정보에서 해당 내용을 찾을 수 없습니다"라고 답하세요.
5. 한국어로 자연스럽고 이해하기 쉽게 답변하세요.
'''

prompt = ChatPromptTemplate.from_template(template)

# RAG Chain 연결
rag_chain = (
    {'context': RunnableLambda(execute_query_and_get_etf_data), 'question': RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Chain 실행
query = "삼성자산운용의 KODEX 관련 상품은 무엇인가요?" 
answer = rag_chain.invoke(query)

print(answer)
```

### **[실습]** 

- S&P 500 회사 데이터(data/sp500_companies.csv)를 가져와서, Knowledge Graph로 구현합니다. 
- 관계 구조: Company → Sector → Industry (또는 Company → Exchange)
- 지식 그래프 검색의 다양한 기법을 적용합니다. 

- 출처: https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks?resource=download

#### 1. 데이터 로드 및 전처리

```python
sp500_df = pd.read_csv("./data/sp500_companies.csv")
sp500_df.head()
```

#### 2. 제약 조건 설정

```python
# ==========================================
# 데이터베이스 초기화 (S&P 500 실습 시작 전)
# ==========================================
print("⚠️ 데이터베이스 초기화 중...")
graph.query("MATCH (n) DETACH DELETE n")
print("✅ 데이터베이스 초기화 완료")
```

```python
# 여기에 코드를 작성하세요.

# Company 노드의 Symbol이 고유하도록 유일성 제약조건 설정
unique_symbol_constraint = """
CREATE CONSTRAINT company_symbol_unique IF NOT EXISTS
FOR (c:Company)
REQUIRE c.Symbol IS UNIQUE
"""
graph.query(unique_symbol_constraint)

# Company 노드의 Symbol이 반드시 존재하도록 설정
exists_symbol_constraint = """
CREATE CONSTRAINT company_symbol_exists IF NOT EXISTS
FOR (c:Company)
REQUIRE c.Symbol IS NOT NULL
"""
graph.query(exists_symbol_constraint)

# Sector 노드의 name이 고유하도록 유일성 제약조건 설정
unique_sector_constraint = """
CREATE CONSTRAINT sector_name_unique IF NOT EXISTS
FOR (s:Sector)
REQUIRE s.name IS UNIQUE
"""
graph.query(unique_sector_constraint)

# Industry 노드의 name이 고유하도록 유일성 제약조건 설정
unique_industry_constraint = """
CREATE CONSTRAINT industry_name_unique IF NOT EXISTS
FOR (i:Industry)
REQUIRE i.name IS UNIQUE
"""
graph.query(unique_industry_constraint)

# Exchange 노드의 name이 고유하도록 유일성 제약조건 설정
unique_exchange_constraint = """
CREATE CONSTRAINT exchange_name_unique IF NOT EXISTS
FOR (e:Exchange)
REQUIRE e.name IS UNIQUE
"""
graph.query(unique_exchange_constraint)

print("✅ 제약 조건 설정 완료")
graph.query("SHOW CONSTRAINTS")
```

### 3. 노드 및 관계 생성

```python
# 여기에 코드를 작성하세요.

# DataFrame을 딕셔너리 리스트로 변환
def clean_value(value):
    """NaN 값을 None으로 변환하는 헬퍼 함수"""
    if pd.isna(value):
        return None
    return value

# 회사 데이터 변환
company_data = []
for record in sp500_df.to_dict('records'):
    cleaned_record = {k: clean_value(v) for k, v in record.items()}
    company_data.append(cleaned_record)

print(f"총 {len(company_data)}개의 회사 데이터 준비 완료")

# 1. Company 노드 생성
company_batch_query = """
UNWIND $company_list AS company
CREATE (:Company {
    Symbol: company.Symbol,
    Security: company.Security,
    Sector: company.`GICS Sector`,
    SubIndustry: company.`GICS Sub-Industry`,
    HeadquartersLocation: company.`Headquarters Location`,
    DateAdded: company.`Date added`,
    CIK: company.CIK,
    Founded: company.Founded
})
"""

BATCH_SIZE = 100
from tqdm import tqdm

for i in tqdm(range(0, len(company_data), BATCH_SIZE), desc="Company 노드 생성"):
    batch = company_data[i:i + BATCH_SIZE]
    graph.query(company_batch_query, params={'company_list': batch})

print("✅ Company 노드 생성 완료")

# 2. Sector 노드 생성
sector_query = """
MATCH (c:Company)
WITH DISTINCT c.Sector AS sectorName
WHERE sectorName IS NOT NULL
MERGE (s:Sector {name: sectorName})
RETURN count(s) AS sector_count
"""
result = graph.query(sector_query)
print(f"✅ Sector 노드 생성 완료: {result[0]['sector_count']}개")

# 3. Industry 노드 생성
industry_query = """
MATCH (c:Company)
WITH DISTINCT c.SubIndustry AS industryName
WHERE industryName IS NOT NULL
MERGE (i:Industry {name: industryName})
RETURN count(i) AS industry_count
"""
result = graph.query(industry_query)
print(f"✅ Industry 노드 생성 완료: {result[0]['industry_count']}개")

# 4. Exchange 노드 생성 (Symbol에서 추출)
exchange_query = """
MATCH (c:Company)
WITH DISTINCT 
    CASE 
        WHEN c.Symbol CONTAINS '.' THEN split(c.Symbol, '.')[1]
        ELSE 'NYSE'
    END AS exchangeName
MERGE (e:Exchange {name: exchangeName})
RETURN count(e) AS exchange_count
"""
result = graph.query(exchange_query)
print(f"✅ Exchange 노드 생성 완료: {result[0]['exchange_count']}개")

# 5. Company와 Sector 관계 생성
company_sector_rel = """
MATCH (c:Company), (s:Sector)
WHERE c.Sector = s.name
MERGE (c)-[r:BELONGS_TO_SECTOR]->(s)
RETURN count(r) AS rel_count
"""
result = graph.query(company_sector_rel)
print(f"✅ Company-Sector 관계 생성 완료: {result[0]['rel_count']}개")

# 6. Company와 Industry 관계 생성
company_industry_rel = """
MATCH (c:Company), (i:Industry)
WHERE c.SubIndustry = i.name
MERGE (c)-[r:IN_INDUSTRY]->(i)
RETURN count(r) AS rel_count
"""
result = graph.query(company_industry_rel)
print(f"✅ Company-Industry 관계 생성 완료: {result[0]['rel_count']}개")

# 7. Industry와 Sector 관계 생성
industry_sector_rel = """
MATCH (c:Company)-[:IN_INDUSTRY]->(i:Industry)
MATCH (c)-[:BELONGS_TO_SECTOR]->(s:Sector)
WITH DISTINCT i, s
MERGE (i)-[r:PART_OF_SECTOR]->(s)
RETURN count(r) AS rel_count
"""
result = graph.query(industry_sector_rel)
print(f"✅ Industry-Sector 관계 생성 완료: {result[0]['rel_count']}개")

# 8. Company와 Exchange 관계 생성
company_exchange_rel = """
MATCH (c:Company), (e:Exchange)
WHERE (c.Symbol CONTAINS '.' AND split(c.Symbol, '.')[1] = e.name)
   OR (NOT c.Symbol CONTAINS '.' AND e.name = 'NYSE')
MERGE (c)-[r:LISTED_ON]->(e)
RETURN count(r) AS rel_count
"""
result = graph.query(company_exchange_rel)
print(f"✅ Company-Exchange 관계 생성 완료: {result[0]['rel_count']}개")

print("\n✅ 모든 노드 및 관계 생성 완료")
```

### 4. 데이터 확인 및 검색

```python
# 여기에 코드를 작성하세요.

# 1. 전체 노드 개수 확인
node_count_query = """
MATCH (n)
RETURN labels(n)[0] AS nodeType, count(n) AS count
ORDER BY count DESC
"""
result = graph.query(node_count_query)
print("=== 노드 타입별 개수 ===")
for record in result:
    print(f"{record['nodeType']}: {record['count']}개")

print("\n" + "="*50 + "\n")

# 2. 특정 회사 검색 (예: Apple)
company_search_query = """
MATCH (c:Company {Symbol: 'AAPL'})
RETURN c.Symbol AS Symbol, 
       c.Security AS Name,
       c.Sector AS Sector,
       c.SubIndustry AS Industry,
       c.HeadquartersLocation AS Location,
       c.Founded AS Founded
"""
result = graph.query(company_search_query)
print("=== Apple Inc. 정보 ===")
if result:
    record = result[0]
    for key, value in record.items():
        print(f"{key}: {value}")

print("\n" + "="*50 + "\n")

# 3. 섹터별 회사 수 통계
sector_stats_query = """
MATCH (s:Sector)<-[:BELONGS_TO_SECTOR]-(c:Company)
RETURN s.name AS Sector, count(c) AS CompanyCount
ORDER BY CompanyCount DESC
"""
result = graph.query(sector_stats_query)
print("=== 섹터별 회사 수 ===")
for record in result:
    print(f"{record['Sector']}: {record['CompanyCount']}개")

print("\n" + "="*50 + "\n")

# 4. 특정 섹터의 회사 목록 조회 (예: Information Technology)
sector_companies_query = """
MATCH (c:Company)-[:BELONGS_TO_SECTOR]->(s:Sector {name: 'Information Technology'})
RETURN c.Symbol AS Symbol, c.Security AS Name
ORDER BY c.Symbol
LIMIT 10
"""
result = graph.query(sector_companies_query)
print("=== Information Technology 섹터 회사 (상위 10개) ===")
for record in result:
    print(f"{record['Symbol']}: {record['Name']}")

print("\n" + "="*50 + "\n")

# 5. 관계 패턴 검색: Company -> Industry -> Sector 경로
relationship_pattern_query = """
MATCH (c:Company)-[:IN_INDUSTRY]->(i:Industry)-[:PART_OF_SECTOR]->(s:Sector)
WHERE c.Symbol = 'AAPL'
RETURN c.Security AS Company,
       i.name AS Industry,
       s.name AS Sector
"""
result = graph.query(relationship_pattern_query)
print("=== Apple의 관계 구조 ===")
if result:
    record = result[0]
    print(f"회사: {record['Company']}")
    print(f"산업: {record['Industry']}")
    print(f"섹터: {record['Sector']}")

print("\n" + "="*50 + "\n")

# 6. 거래소별 회사 수
exchange_stats_query = """
MATCH (e:Exchange)<-[:LISTED_ON]-(c:Company)
RETURN e.name AS Exchange, count(c) AS CompanyCount
ORDER BY CompanyCount DESC
"""
result = graph.query(exchange_stats_query)
print("=== 거래소별 회사 수 ===")
for record in result:
    print(f"{record['Exchange']}: {record['CompanyCount']}개")

print("\n" + "="*50 + "\n")

# 7. 특정 산업의 회사 목록
industry_companies_query = """
MATCH (c:Company)-[:IN_INDUSTRY]->(i:Industry {name: 'Internet & Direct Marketing Retail'})
RETURN c.Symbol AS Symbol, c.Security AS Name, c.HeadquartersLocation AS Location
ORDER BY c.Symbol
"""
result = graph.query(industry_companies_query)
print("=== Internet & Direct Marketing Retail 산업 회사 ===")
for record in result:
    print(f"{record['Symbol']}: {record['Name']} ({record['Location']})")

print("\n✅ 데이터 확인 및 검색 완료")
```

### 5. 벡터 인덱스 생성

```python
# 여기에 코드를 작성하세요.

from langchain_openai import OpenAIEmbeddings
from langchain_neo4j import Neo4jVector

# OpenAI 임베딩 모델 초기화
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 1. Company 노드 데이터 조회
companies = graph.query("""
MATCH (c:Company)
RETURN c.Symbol AS Symbol,
       c.Security AS Security,
       c.Sector AS Sector,
       c.SubIndustry AS SubIndustry,
       c.HeadquartersLocation AS Location
""")

print(f"총 {len(companies)}개의 회사 데이터 조회 완료")

# 2. 각 Company에 대해 임베딩 생성 및 업데이트
from tqdm import tqdm

for company in tqdm(companies, desc="임베딩 생성 중"):
    # Company 정보를 텍스트로 결합
    combined_text = f"""회사명: {company['Security']}
심볼: {company['Symbol']}
섹터: {company['Sector']}
산업: {company['SubIndustry']}
본사 위치: {company['Location']}"""
    
    # 임베딩 생성
    embedding_vector = embeddings.embed_query(combined_text)
    
    # Company 노드에 임베딩 속성 추가
    graph.query("""
    MATCH (c:Company {Symbol: $symbol})
    CALL db.create.setNodeVectorProperty(c, 'embedding', $embedding)
    """, params={
        'symbol': company['Symbol'],
        'embedding': embedding_vector
    })

print("✅ 임베딩 생성 완료")

# 3. 벡터 인덱스 생성
graph.query("""
CREATE VECTOR INDEX company_vector_index IF NOT EXISTS
FOR (c:Company) ON (c.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
  }
}
""")

print("✅ 벡터 인덱스 생성 완료")

# 4. 벡터 검색 테스트
vector_db = Neo4jVector.from_existing_index(
    embeddings,
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    index_name="company_vector_index",
    node_label="Company",
    text_node_property="Security",
    embedding_node_property="embedding",
        retrieval_query="""
    WITH node, score
    WITH node, score,
         "회사명: " + COALESCE(node.Security, "Unknown") + 
         "
심볼: " + COALESCE(node.Symbol, "N/A") +
         CASE WHEN node.Sector IS NOT NULL THEN "
섹터: " + node.Sector ELSE "" END +
         CASE WHEN node.SubIndustry IS NOT NULL THEN "
산업: " + node.SubIndustry ELSE "" END +
         CASE WHEN node.HeadquartersLocation IS NOT NULL THEN "
본사: " + node.HeadquartersLocation ELSE "" END
         AS combined_text
         
    RETURN combined_text AS text,
           score,
           {
               symbol: node.Symbol,
               security: node.Security,
               sector: node.Sector,
               subIndustry: node.SubIndustry,
               location: node.HeadquartersLocation
           } AS metadata
    """)

# 5. 유사도 검색 테스트
query = "technology companies in California"
results = vector_db.similarity_search(query, k=5)

print(f"\n검색 질의: '{query}'\n")
print("="*50)
for i, result in enumerate(results, 1):
    print(f"\n{i}. {result.page_content}")
    print(f"   메타데이터: {result.metadata}")

print("\n✅ 벡터 인덱스 생성 및 검색 테스트 완료")
```

---

## 🚀 실무 활용 예시

### 1. 금융 상품 추천 시스템

```python
# 사용자 질문에 기반한 ETF 추천
def recommend_etf(user_query: str, k: int = 5):
    """
    사용자 질문에 적합한 ETF 추천
    
    Args:
        user_query: 사용자 질문 (예: '안정적인 배당 수익을 원합니다')
        k: 추천할 ETF 개수
    """
    # 1. 벡터 검색으로 유사한 ETF 찾기
    vector_results = vector_store.similarity_search(user_query, k=k*2)
    
    # 2. 그래프 검색으로 관련 정보 확장
    for result in vector_results:
        etf_name = result.metadata['name']
        
        # 관련 회사 및 카테고리 정보 조회
        query = """
        MATCH (e:ETF {name: $etf_name})
        MATCH (c:Company)-[:MANAGES]->(e)
        MATCH (e)-[:BELONGS_TO]->(cat:Category)
        RETURN e, c.name as company, collect(cat.name) as categories
        """
        
        graph_info = graph.query(query, params={'etf_name': etf_name})
        
        # 종합 정보 생성
        print(f"ETF: {etf_name}")
        print(f"운용사: {graph_info[0]['company']}")
        print(f"카테고리: {', '.join(graph_info[0]['categories'])}")
        print(f"설명: {result.page_content[:100]}...\n")
```

### 2. 경쟁사 분석 시스템

```python
# 특정 회사의 경쟁사 및 시장 포지션 분석
def analyze_company_position(company_name: str):
    """회사의 시장 포지션 분석"""
    
    analysis_query = """
    // 1. 대상 회사가 운용하는 ETF
    MATCH (c:Company {name: $company})-[:MANAGES]->(e:ETF)
    
    // 2. 같은 카테고리의 다른 회사 ETF
    MATCH (e)-[:BELONGS_TO]->(cat:Category)
    MATCH (other_etf:ETF)-[:BELONGS_TO]->(cat)
    MATCH (competitor:Company)-[:MANAGES]->(other_etf)
    WHERE competitor.name <> $company
    
    // 3. 통계 계산
    RETURN 
        c.name as company,
        count(DISTINCT e) as my_etf_count,
        count(DISTINCT cat) as category_count,
        collect(DISTINCT competitor.name) as competitors,
        count(DISTINCT competitor) as competitor_count
    """
    
    result = graph.query(analysis_query, params={'company': company_name})
    return result[0]
```

### 3. 트렌드 분석 및 인사이트

```python
# 카테고리별 ETF 분포 및 트렌드 분석
def analyze_market_trends():
    """시장 트렌드 분석"""
    
    trend_query = """
    // 카테고리별 ETF 개수 및 운용사 분포
    MATCH (cat:Category)<-[:BELONGS_TO]-(e:ETF)<-[:MANAGES]-(c:Company)
    
    WITH cat, 
         count(DISTINCT e) as etf_count,
         count(DISTINCT c) as company_count,
         collect(DISTINCT c.name) as companies
    
    RETURN 
        cat.name as category,
        etf_count,
        company_count,
        companies
    ORDER BY etf_count DESC
    LIMIT 10
    """
    
    trends = graph.query(trend_query)
    
    print("📊 시장 트렌드 분석\n")
    for trend in trends:
        print(f"카테고리: {trend['category']}")
        print(f"  ETF 개수: {trend['etf_count']}")
        print(f"  참여 운용사: {trend['company_count']}개")
        print(f"  주요 운용사: {', '.join(trend['companies'][:3])}\n")
```

---

## 🎯 실습 문제

### 문제 1: 커스텀 Text-to-Cypher 프롬프트

다음 요구사항을 만족하는 커스텀 프롬프트를 작성하세요:

- ETF 이름이 포함된 질문에 특화
- 카테고리 정보를 자동으로 포함
- 운용사 정보도 함께 반환

### 문제 2: 하이브리드 검색 함수

벡터 검색과 그래프 검색을 결합한 하이브리드 검색 함수를 구현하세요:

```python
def hybrid_search(query: str, k: int = 5):
    """
    하이브리드 검색 구현
    
    Args:
        query: 검색 질의
        k: 반환할 결과 개수
    
    Returns:
        통합 검색 결과
    """
    # 여기에 코드를 작성하세요
    pass
```

### 문제 3: 추천 시스템 고도화

사용자 프로필(리스크 성향, 투자 목적 등)을 고려한 ETF 추천 시스템을 구현하세요.

---

## ✅ 솔루션 예시

### 문제 2 솔루션: 하이브리드 검색

```python
def hybrid_search(query: str, k: int = 5):
    """하이브리드 검색: 벡터 + 그래프"""
    
    # 1. 벡터 검색
    vector_results = vector_store.similarity_search_with_score(query, k=k*2)
    
    results = []
    for doc, score in vector_results:
        etf_name = doc.metadata.get('name')
        
        # 2. 그래프 검색으로 관계 정보 확장
        graph_query = """
        MATCH (e:ETF {name: $etf_name})
        MATCH (c:Company)-[:MANAGES]->(e)
        MATCH (e)-[:BELONGS_TO]->(cat:Category)
        
        OPTIONAL MATCH (e)-[:BELONGS_TO]->(cat)<-[:BELONGS_TO]-(similar:ETF)
        WHERE similar.name <> e.name
        
        RETURN 
            e.name as etf,
            c.name as company,
            collect(DISTINCT cat.name) as categories,
            count(DISTINCT similar) as similar_count
        """
        
        graph_info = graph.query(graph_query, params={'etf_name': etf_name})
        
        if graph_info:
            info = graph_info[0]
            results.append({
                'etf': info['etf'],
                'company': info['company'],
                'categories': info['categories'],
                'similar_count': info['similar_count'],
                'vector_score': score,
                'content': doc.page_content
            })
    
    # 3. 종합 점수로 재정렬
    # 벡터 점수 + 유사 ETF 개수 고려
    for result in results:
        result['final_score'] = (
            result['vector_score'] * 0.7 + 
            (result['similar_count'] / 10) * 0.3
        )
    
    results.sort(key=lambda x: x['final_score'], reverse=True)
    
    return results[:k]
```

---

## 📖 참고 자료

### 공식 문서

- [LangChain GraphCypherQAChain](https://python.langchain.com/docs/use_cases/graph/graph_cypher_qa)
- [Neo4j Vector Search](https://neo4j.com/docs/cypher-manual/current/indexes-for-vector-search/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)

### 추가 학습 자료

- **Neo4j Cypher Manual**: 쿼리 언어 완벽 가이드
- **LangChain Use Cases**: 그래프 데이터베이스 활용 패턴
- **Vector Search Best Practices**: 효율적인 벡터 검색 구현

---

## ✅ 학습 마무리

이 Part 2를 완료하면서 다음을 배웠습니다:

1. ✅ Text-to-Cypher로 자연어 질의응답 시스템 구현
2. ✅ 벡터 검색을 활용한 의미론적 검색
3. ✅ 하이브리드 검색으로 검색 품질 향상
4. ✅ 실무 활용 시나리오 (금융 상품 추천, 경쟁사 분석)

**다음 단계**: 이제 실제 비즈니스 시나리오에 지식 그래프 기반 QA 시스템을 적용할 수 있습니다!

---

**Part 2 완료!** 🎉

**전체 과정 완료 축하합니다!** 이제 여러분은 Neo4j로 정형 데이터를 지식 그래프로 변환하고, 자연어 질의응답 시스템을 구축할 수 있습니다.
