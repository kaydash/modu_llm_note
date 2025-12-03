# Knowledge Graph 구축 - Part 1: 기초와 인덱싱

**학습 자료**: PRJ04_W3_003 - Neo4j로 정형 데이터를 지식 그래프로 변환하기

---

## 📚 학습 목표

이 실습을 완료하면 다음을 할 수 있습니다:

1. **Neo4j Desktop 환경 구축**: 로컬 개발 환경 설정 및 데이터베이스 생성
2. **정형 데이터를 지식 그래프로 변환**: CSV/DataFrame을 노드와 관계로 모델링
3. **GraphCypherQAChain 활용**: LangChain을 통한 자연어 질의응답 구현
4. **인덱스 최적화**: 검색 성능 향상을 위한 다양한 인덱스 생성 및 활용
5. **복합 인덱스 및 Full-Text 검색**: 고급 검색 기능 구현

---

## 🔑 핵심 개념

### Knowledge Graph (지식 그래프)

**지식 그래프**는 현실 세계의 개체(Entity)와 그들 간의 관계(Relationship)를 그래프 구조로 표현한 것입니다.

#### 구성 요소

1. **노드(Node)**: 개체를 나타냄 (예: 회사, ETF, 카테고리)
2. **관계(Relationship)**: 노드 간의 연결 (예: 회사-[운용]->ETF)
3. **속성(Property)**: 노드나 관계의 특성 (예: 회사명, ETF 이름)

#### 지식 그래프의 장점

- **관계 중심 데이터 모델**: 복잡한 관계를 자연스럽게 표현
- **유연한 스키마**: 새로운 관계나 속성을 쉽게 추가
- **강력한 쿼리**: 그래프 순회를 통한 복잡한 질의 가능
- **의미론적 검색**: 컨텍스트 기반 검색 및 추론

### Neo4j 인덱스

인덱스는 데이터베이스 검색 성능을 향상시키는 핵심 기능입니다.

#### 인덱스 종류

1. **단일 속성 인덱스**: 하나의 속성에 대한 인덱스
2. **복합 인덱스**: 여러 속성을 조합한 인덱스
3. **Full-Text 인덱스**: 텍스트 검색을 위한 특수 인덱스
4. **벡터 인덱스**: 임베딩 기반 유사도 검색용 인덱스

---

## 🛠 환경 설정

### Neo4j Desktop 설치

1. [Neo4j Desktop](https://neo4j.com/download/) 다운로드 및 설치
2. 새 프로젝트 생성
3. 로컬 데이터베이스 인스턴스 생성
4. 데이터베이스 시작

### 필수 라이브러리 설치

```bash
# 핵심 라이브러리
pip install langchain langchain-community langchain-openai langchain-neo4j

# Neo4j 드라이버
pip install neo4j

# 데이터 처리
pip install pandas

# 유틸리티
pip install python-dotenv
```

### 환경 변수 설정

`.env` 파일에 다음 정보를 설정합니다:

```bash
# OpenAI API
OPENAI_API_KEY=your-openai-key

# Neo4j Local (Desktop)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=neo4j
```

---

## 💻 단계별 구현


## 1. Neo4J Desktop 환경 설정

- 버전 1.6.2 설치: https://neo4j.com/deployment-center/

- Neo4J Desktop 설치 후, Neo4J Desktop에서 새로운 프로젝트 생성
    - 프로젝트 이름: `test`
    - 데이터베이스 버전: `5.24.0`

- 데이터베이스에서 플러그인 설치
    - `APOC`

- 데이터베이스 Settings에서 다음 설정 추가 (apoc 검색해서 기존 설정에 추가)
    - `dbms.security.procedures.unrestricted=jwt.security.*,apoc.*,apoc.meta.*`

- Neo4J Desktop 사용(.env)
    ```
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USERNAME=neo4j
    NEO4J_PASSWORD=modulab1234
    NEO4J_DATABASE=neo4j
    ```

`(1) Env 환경변수`

```python
from dotenv import load_dotenv
load_dotenv()
```

`(2) 라이브러리`

```python
import os
from glob import glob

from pprint import pprint
import json

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')
```

`(3) Neo4j 설정`

```python
import os
from langchain_neo4j import Neo4jGraph

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")


# LangChain 도구 활용 - DB 연결 객체 초기화 
graph = Neo4jGraph(  
    url=NEO4J_URI, 
    username=NEO4J_USERNAME, 
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
    )

graph.query("MATCH (n) RETURN n LIMIT 5;")
```

`(4) 기존 DB의 모든 내용 삭제`

```python
def reset_database(graph):
    """
    데이터베이스 초기화하기
    """
    # 모든 노드와 관계 삭제
    graph.query("MATCH (n) DETACH DELETE n")
    
    # 모든 제약조건 삭제
    constraints = graph.query("SHOW CONSTRAINTS")
    for constraint in constraints:
        constraint_name = constraint.get("name")
        if constraint_name:
            graph.query(f"DROP CONSTRAINT {constraint_name}")
    
    # 모든 인덱스 삭제
    indexes = graph.query("SHOW INDEXES")
    for index in indexes:
        index_name = index.get("name")
        index_type = index.get("type")
        if index_name and index_type != "CONSTRAINT":
            graph.query(f"DROP INDEX {index_name}")
    
    print("데이터베이스가 초기화되었습니다.")

# 데이터베이스 초기화
reset_database(graph)
```

```python
# 그래프 스키마 조회
graph.refresh_schema()
print(graph.schema)
```

## 2. 정형 데이터를 KG로 변환

`(1) Load CSV Data`

- etf info 데이터 활용 (data/etf_list.csv)

```python
# CSV 파일 읽기
df = pd.read_csv('data/etf_list.csv', encoding='cp949')

df.shape
```

```python
df.head()
```

`(2) 제약 조건 생성`

```python
# 종목코드(code)는 ETF마다 고유하므로 유일성 제약조건을 설정 
unique_code_constraint = """
CREATE CONSTRAINT etf_code_unique IF NOT EXISTS
FOR (e:ETF) 
REQUIRE e.code IS UNIQUE
"""
graph.query(unique_code_constraint)
```

```python
graph.query("SHOW CONSTRAINTS")
```

```python
# 종목코드와 종목명은 반드시 존재해야 하는 필수 속성으로 설정 

exists_code_constraint = """
// 종목코드가 반드시 존재해야 하는 필수 속성으로 설정
CREATE CONSTRAINT etf_code_exists IF NOT EXISTS
FOR (e:ETF)
REQUIRE e.code IS NOT NULL
"""
graph.query(exists_code_constraint)

exists_name_constraint = """
// 종목명이 반드시 존재해야 하는 필수 속성으로 설정
CREATE CONSTRAINT etf_name_exists IF NOT EXISTS
FOR (e:ETF)
REQUIRE e.name IS NOT NULL
"""
graph.query(exists_name_constraint)
```

```python
graph.query("SHOW INDEXES")
```

`(3) Neo4J 그래프 DB에서 CSV파일 업로드`

```python
df.head(1)
```

```python
# pandas의 to_dict를 활용한 데이터 변환
def clean_value(value):
    """NaN 값을 None으로 변환하는 헬퍼 함수"""
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)) and pd.notna(value):
        return value
    return value

# DataFrame을 딕셔너리 리스트로 변환
etf_data = []
for record in df.to_dict('records'):
    cleaned_record = {k: clean_value(v) for k, v in record.items()}

    # 컬럼명을 영어로 매핑
    etf_record = {
        'code': cleaned_record['종목코드'],
        'name': cleaned_record['종목명'],
        'listingDate': cleaned_record['상장일'],
        'category': cleaned_record['분류체계'],
        'company': cleaned_record['운용사'],
        'yearReturn': cleaned_record['수익률(최근 1년)'],
        'baseIndex': cleaned_record['기초지수'],
        'trackingError': cleaned_record['추적오차'],
        'netAsset': cleaned_record['순자산총액'],
        'disparityRatio': cleaned_record['괴리율'],
        'volatility': cleaned_record['변동성'],
        'replicationMethod': cleaned_record['복제방법'],
        'totalFee': cleaned_record['총보수'],
        'taxType': cleaned_record['과세유형']
    }
    etf_data.append(etf_record)

# 배치 처리 실행
BATCH_SIZE = 100
batch_query = """
// ETF 노드 생성
UNWIND $etf_list AS etf    // $etf_list를 etf로 풀어헤치기
CREATE (:ETF {
    code: etf.code, // 종목코드
    name: etf.name, // 종목명
    listingDate: etf.listingDate, // 상장일
    category: etf.category, // 분류체계
    company: etf.company, // 운용사
    yearReturn: etf.yearReturn, // 수익률(최근 1년)
    baseIndex: etf.baseIndex,  // 기초지수
    trackingError: etf.trackingError, // 추적오차
    netAsset: etf.netAsset, // 순자산총액
    disparityRatio: etf.disparityRatio, // 괴리율
    volatility: etf.volatility, // 변동성
    replicationMethod: etf.replicationMethod, // 복제방법
    totalFee: etf.totalFee, // 총보수
    taxType: etf.taxType // 과세유형
})
"""

from tqdm import tqdm

# 배치 처리
for i in tqdm(range(0, len(etf_data), BATCH_SIZE)):
    batch = etf_data[i:i + BATCH_SIZE]
    graph.query(batch_query, params={'etf_list': batch})

print("ETF 데이터 배치 로드 완료")
```

```python
# ETF 노드 개수 확인
count_query = """
MATCH (e:ETF)   // 모든 ETF 노드 검색
RETURN count(e) AS count   // 노드 개수 반환
"""
graph.query(count_query)
```

```python
# ETF 노드 속성으로 조건을 주고 검색
query = """
MATCH (e:ETF {code: '451060'})   // 종목코드가 451060인 ETF 노드 검색
RETURN e   // 노드 반환
"""
graph.query(query)
```

```python
# 운용사(Company) 노드 생성 및 관계 설정
company_query = """
MATCH (e:ETF)       // ETF 노드에서
WITH DISTINCT e.company AS companyName    // 회사 이름을 가져옴
WHERE companyName IS NOT NULL             // 회사 이름이 NULL이 아닌 경우
MERGE (c:Company {name: companyName})     // Company 노드 생성
RETURN count(c) AS company_count          // 생성된 Company 노드 개수 반환
"""

result = graph.query(company_query)
print(result)
```

```python
# ETF와 운용사 간의 관계 생성
relationship_query = """
MATCH (e:ETF), (c:Company)     // ETF와 Company 노드 모두 선택
WHERE e.company = c.name       // 회사 이름이 일치하는 경우
MERGE (c)-[r:MANAGES]->(e)      // Company 노드에서 ETF 노드로 MANAGES 관계 생성
RETURN count(r) AS relationship_count   // 생성된 관계 개수 반환
"""
graph.query(relationship_query)
```

`(4) 관계 추가`

```python
# 분류체계(Category) 유일성 제약조건 설정
unique_category_constraint = """
CREATE CONSTRAINT category_name_unique IF NOT EXISTS
FOR (c:Category)
REQUIRE c.name IS UNIQUE
"""
graph.query(unique_category_constraint)
```

```python
# 분류체계(Category) 노드 생성
category_query = """
MATCH (e:ETF)           // ETF 노드에서
WITH DISTINCT e.category AS categoryName    // 분류체계 이름을 가져옴
WHERE categoryName IS NOT NULL              // 분류체계 이름이 NULL이 아닌 경우
MERGE (c:Category {name: categoryName})     // Category 노드 생성
RETURN count(c) as CategoryCount            // 생성된 Category 노드 수 반환
"""
graph.query(category_query)
```

```python
# ETF와 분류체계(Category) 간의 관계 생성
relationship_query = """
MATCH (e:ETF), (c:Category)     // ETF와 Category 노드 모두 선택
WHERE e.category = c.name       // 분류체계 이름이 일치하는 경우
MERGE (e)-[r:BELONGS_TO]->(c)    // ETF 노드에서 Category 노드로 BELONGS_TO 관계 생성
RETURN count(r) as RelationshipCount   // 생성된 관계 수 반환
"""
graph.query(relationship_query)
```

```python
# 특정 카테코리의 ETF 노드 조회
category_query = """
MATCH (c:Category {name: '채권-혼합-단기'})<-[:BELONGS_TO]-(etf:ETF)  // '채권-혼합-단기' 카테고리에 속하는 ETF 노드 검색
RETURN etf.name, etf.yearReturn, c.name   // ETF 노드의 이름과 수익률 반환, 카테고리 반환
ORDER BY etf.yearReturn DESC // 수익률 기준 내림차순 정렬
LIMIT 5  // 상위 5개 노드만 반환
"""

graph.query(category_query)
```

```python
# 카테코리별 평균 수익률 등의 통계 데이터를 계산
category_stats_query = """
MATCH (c:Category)<-[:BELONGS_TO]-(etf:ETF)  // 모든 카테고리와 그에 속하는 ETF 노드 검색
RETURN c.name AS category,      // 카테고리 이름
       COUNT(etf) AS etf_count,    // ETF 개수
       AVG(etf.yearReturn) AS avg_yearReturn,   // 평균 수익률
       SUM(etf.netAsset) AS total_netAsset,     // 총 순자산총액
       AVG(etf.trackingError) AS avg_trackingError   // 평균 추적오차
ORDER BY avg_yearReturn DESC // 평균 수익률 기준 내림차순 정렬  
"""

category_stats = graph.query(category_stats_query)

# 결과를 DataFrame으로 변환
category_stats_df = pd.DataFrame(category_stats)
category_stats_df.head(10)  # 상위 10개 카테고리 출력
```

## 3. GraphCypherQAChain 활용

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
)

# LangChain 도구 활용 - GraphCypherQAChain 객체 초기화
chain = GraphCypherQAChain.from_llm(
    llm=llm, 
    graph=graph, 
    allow_dangerous_requests=True,
    verbose=True,)

result = chain.run("케이비자산운용은 모두 몇개의 ETF를 운용하고 있나요?")
```

```python
# 결과 출력
print(result)
```

케이비, 운용, ETF

```python
chain.run("케이비운용은 모두 몇개의 ETF를 운용하고 있나요?")
```

```python
result = chain.invoke("코스피 200을 기초 지수로 하는 ETF는 무엇인가요?")
```

```python
# 결과 출력
print(result)
```

```python
print(result['result'])
```

## 4. 인덱스 생성
- 자주 쿼리하는 속성에 대해 인덱스를 생성하여 쿼리 성능 개선

`(1) 기본 인덱스`
- 유일성 제약조건이 있는 속성은 자동으로 인덱스가 생성되므로 별도 생성 불필요


```python
# 인덱스 확인
graph.query("SHOW INDEXES")
```

`(2) 인덱스 활용한 검색`

- 인덱스를 생성한 속성을 WHERE 절에서 사용하면 Neo4j가 자동으로 인덱스를 활용

```python
# 종목코드로 ETF 검색 (정확한 일치 검색)
code_search_query = """
MATCH (e:ETF)
WHERE e.code = '451060'
RETURN e AS etf
"""
graph.query(code_search_query)
```

```python
# 종목코드로 ETF 검색 (범위 검색)
code_search_query = """
MATCH (e:ETF)
WHERE toFloat(e.code) >= 453000 AND toFloat(e.code) < 454000   // 코드가 문자열 순으로 '453'와 '454' 사이에 있는 ETF 검색
RETURN e.code AS etf_code, e.name AS etf_name  // 종목코드와 종목명 반환     
ORDER BY e.code ASC       // 종목코드 기준으로 정렬 (ASC/DESC)
LIMIT 10     // 최대 10개 결과 반환
"""
results = graph.query(code_search_query)

results = pd.DataFrame(results)
results
```

```python
print(graph.schema)
```

```python
# 종목명으로 ETF 검색 (접두사 검색)
name_search_query = """
MATCH (e:ETF)  
WHERE e.name STARTS WITH 'TIGER'     // 'TIGER'로 시작하는 종목명 검색
RETURN e.name
"""

graph.query(name_search_query)
```

```python
# 운용사 이름으로 ETF 검색 (IN 연산자 활용)
company_search_query = """
MATCH (e:ETF)  
WHERE e.company IN ['삼성자산운용', '미래에셋자산운용']  // 삼성자산운용 또는 미래에셋자산운용이 운용하는 ETF 검색" 
RETURN e.name, e.company  // 종목명과 운용사 이름 반환
LIMIT 10   // 최대 10개 결과 반환
"""

graph.query(company_search_query)
```

```python
# 카테고리 노드와 관련된 관계 검색
relationship_query = """
MATCH (c:Category)      
WHERE c.name = '주식-시장대표'   // 카테고리 이름이 '주식-시장대표'인 카테고리 노드 검색
MATCH (e:ETF)-[r:BELONGS_TO]->(c)   // ETF 노드와 카테고리 노드 간의 BELONGS_TO 관계 검색
RETURN c.name AS category_name,  // 카테고리 이름 반환
       collect(e.name) AS etf_names,  // 관련된 ETF 노드의 이름을 리스트로 반환
       count(e) AS etf_count  // 관련된 ETF 개수 반환
"""

graph.query(relationship_query)
```

`(3) 인덱스 생성`

- 대량의 데이터를 처리하거나 자주 쿼리하는 속성에 대해 인덱스를 생성하면 쿼리 성능이 크게 향상됨

```python
# 수익률에 대한 인덱스 생성 - 수익률 기반 검색 및 정렬 시 성능 향상
return_index_query = """
CREATE INDEX etf_yearreturn_idx IF NOT EXISTS
FOR (e:ETF) ON (e.yearReturn)
"""
graph.query(return_index_query)
```

```python
# 수익률 기준으로 ETF 검색 (정렬)
return_search_query = """
MATCH (e:ETF)
WHERE e.yearReturn > 0.1   // 수익률이 10% 이상인 ETF 검색
RETURN e.name AS Name, e.yearReturn AS Return   // 종목명과 수익률 반환
ORDER BY e.yearReturn DESC   // 수익률 기준으로 내림차순 정렬
LIMIT 10   // 최대 10개 결과 반환
"""

graph.query(return_search_query)
```

```python
# 순자산총액에 대한 인덱스 생성 - 규모별 검색 및 정렬 시 성능 향상
netasset_index_query = """
CREATE INDEX etf_netasset_idx IF NOT EXISTS
FOR (e:ETF) ON (e.netAsset)
"""
graph.query(netasset_index_query)
```

```python
# 순자산총액 기준으로 ETF 검색 (정렬)
netasset_search_query = """
MATCH (e:ETF)
WHERE e.netAsset > 1000000000   // 순자산총액이 10억 이상인 ETF 검색
RETURN e.name, e.netAsset   // 종목명과 순자산총액 반환
ORDER BY e.netAsset DESC   // 순자산총액 기준으로 내림차순 정렬
LIMIT 10   // 최대 10개 결과 반환
"""
graph.query(netasset_search_query)
```

`(4) 복합 인덱스`

- 여러 속성을 함께 검색하는 경우가 많다면 복합 인덱스를 고려

```python
# 복합 인덱스 생성 - 카테고리와 수익률을 동시에 검색 및 정렬 시 성능 향상
compound_index_query = """
CREATE INDEX etf_category_return_idx IF NOT EXISTS
FOR (e:ETF) ON (e.category, e.yearReturn)
"""
graph.query(compound_index_query)
```

```python
# 카테고리와 수익률 기준으로 ETF 검색 (정렬)
compound_search_query = """
MATCH (e:ETF)
WHERE e.category = '주식-시장대표' AND e.yearReturn > 0.1   // 카테고리가 '주식-시장대표'이고 수익률이 10% 이상인 ETF 검색
RETURN e.name, e.yearReturn   // 종목명과 수익률 반환
ORDER BY e.yearReturn DESC   // 수익률 기준으로 내림차순 정렬
LIMIT 10   // 최대 10개 결과 반환
"""
graph.query(compound_search_query)
```

`(5) Full Text 검색 인덱스`

- 텍스트 검색이 필요한 경우  Full Text 인덱스를 고려

```python
# Full Text 없이 종목명 검색 ("액티브" 포함)
name_search_query = """
MATCH (e:ETF)  
WHERE e.name CONTAINS '액티브'   // 종목명에 '액티브'가 포함된 ETF 검색
RETURN e.name   // 종목명 반환
LIMIT 10   // 최대 10개 결과 반환
"""
graph.query(name_search_query)
```

```python
# Full Text 인덱스 생성 - 종목명에 대한 Full Text 검색을 위한 인덱스 생성
fulltext_index_query = """
CREATE FULLTEXT INDEX etf_name_fulltext IF NOT EXISTS
FOR (e:ETF) ON EACH [e.name]
"""
graph.query(fulltext_index_query)
```

```python
# Full Text 검색 - 종목명에 "액티브" 포함
fulltext_search_query = """
CALL db.index.fulltext.queryNodes('etf_name_fulltext', '액티브') YIELD node  // Full Text 인덱스 검색
RETURN node.name   // 종목명 반환
LIMIT 10   // 최대 10개 결과 반환
"""
graph.query(fulltext_search_query)
```

```python
graph.query("SHOW FULLTEXT INDEXES")  # 생성된 Full Text 인덱스 확인
```

```python
# # 인덱스 삭제
# drop_index_query = """
# DROP INDEX etf_name_baseindex_fulltext IF EXISTS
# """
# graph.query(drop_index_query)
```

```python
# Full Text 인덱스 생성 - 종목명, 기초지수에 대한 Full Text 검색을 위한 인덱스 생성 (cjk 형태소 분석기 사용)
fulltext_index_query = """
CREATE FULLTEXT INDEX etf_name_baseindex_fulltext IF NOT EXISTS
FOR (e:ETF) ON EACH [e.name, e.baseIndex]
OPTIONS {
    indexConfig: {
        `fulltext.analyzer`: 'cjk',   // cjk 형태소 분석기 사용 
        `fulltext.eventually_consistent`: true  // 성능 최적화를 위한 설정 활성화
    }
}
"""
graph.query(fulltext_index_query)
```

```python
graph.query("SHOW FULLTEXT INDEXES")  # 생성된 Full Text 인덱스 확인
```

```python
# Full Text 검색 - 종목명 또는 기초지수에 "클린 에너지" 포함
fulltext_search_query = """
CALL db.index.fulltext.queryNodes('etf_name_baseindex_fulltext', '클린 에너지') YIELD node  // Full Text 인덱스 검색
RETURN node.name, node.baseIndex   // 종목명과 기초지수 반환
LIMIT 10   // 최대 10개 결과 반환
"""
graph.query(fulltext_search_query)
```

`(6) 벡터 인덱스 검색`

- 벡터 검색이 필요한 경우 벡터 인덱스를 고려
- 단일 속성에 대해서만 벡터 인덱스를 지원

```python
from langchain_core.documents import Document

etfs = graph.query("""
MATCH (e:ETF)   // 모든 ETF 노드 검색
RETURN e.name AS name,   // 종목명 반환
       e.company AS company,   // 운용사 반환
       e.baseIndex AS baseIndex,   // 기초지수 반환
       e.category AS category,   // 카테고리 반환
       e.listingDate AS listingDate   // 상장일 반환
""")

# 검색 결과를 DataFrame으로 변환
etfs_df = pd.DataFrame(etfs)

# 종목명, 운용사, 기초지수 속성을 결합한 문서를 생성
docs = [
    Document(
        page_content=f"종목명: {row['name']}, 운용사: {row['company']}, 기초지수: {row['baseIndex']}",
        metadata={"name": row['name'], "company": row['company'], "baseIndex": row['baseIndex'], "category": row['category'], "listingDate": row['listingDate']}
    )
    for _, row in etfs_df.iterrows()
]

# 문서의 속성 확인
for doc in docs[:5]:  # 상위 5개 문서 출력
    print(doc.page_content)
    print(doc.metadata)
    print()
```

```python
from langchain_openai import OpenAIEmbeddings

# OpenAI 임베딩 모델 초기화
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 기존 ETF 데이터 조회
etfs = graph.query("""
MATCH (e:ETF)
RETURN e.name AS name,
       e.company AS company,
       e.baseIndex AS baseIndex,
       e.category AS category,
       e.listingDate AS listingDate
""")

# 각 ETF에 대해 임베딩 생성 및 업데이트
for etf in etfs:
    # ETF 정보를 텍스트로 결합
    combined_text = f"이 종목은 {etf['name']}이며, 운용사는 {etf['company']}입니다. 기초지수는 {etf['baseIndex']}입니다."
    
    # 임베딩 생성
    embedding_vector = embeddings.embed_query(combined_text)
    
    # 기존 ETF 노드에 임베딩 속성 추가
    graph.query("""
    MATCH (e:ETF {name: $name})
    CALL db.create.setNodeVectorProperty(e, 'embedding', $embedding)
    """, params={
        'name': etf['name'],
        'embedding': embedding_vector
    })
```

```python
# 벡터 인덱스 생성
graph.query("""
CREATE VECTOR INDEX etf_vector_index IF NOT EXISTS
FOR (e:ETF) ON (e.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
  }
}
""")
```

```python
# 기존 벡터 인덱스에 연결
from langchain_openai import OpenAIEmbeddings
from langchain_neo4j import Neo4jVector

embeddings = OpenAIEmbeddings(model="text-embedding-3-small") 

vector_db = Neo4jVector.from_existing_index(
    embeddings,
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    index_name="etf_vector_index",  # 위에서 생성한 인덱스 이름
    node_label="ETF",  # 노드 레이블
    text_node_property="name",  # 텍스트 속성 
    embedding_node_property="embedding"  # 임베딩 속성
)

# 유사도 검색 테스트
results = vector_db.similarity_search("KODEX", k=5)
for result in results:
    print(f"ETF: {result.page_content}")
    print(f"메타데이터: {result.metadata}")
    print("---")
```

```python
# from langchain_openai import OpenAIEmbeddings
# from langchain_neo4j import Neo4jVector

# embeddings = OpenAIEmbeddings(model="text-embedding-3-small") 

# vector_db = Neo4jVector.from_documents(
#     docs,
#     embeddings,
#     url=NEO4J_URI,
#     username=NEO4J_USERNAME,
#     password=NEO4J_PASSWORD,
#     index_name="etf_name_company_asset_vector_index" # 인덱스 이름 설정 (선택 사항)
# )
```

```python
query = "삼성자산운용의 국내 주식형 상품은 무엇인가요?" # 검색할 쿼리
similar_docs = vector_db.similarity_search_with_score(query, k=5) # 유사도 상위 5개 문서 검색

print(f"'{query}'와 유사한 문서:")
for doc, score in similar_docs:
    print(f"문서 내용: {doc.page_content}, 유사도 점수: {score}")
    print(f"메타데이터: {doc.metadata}")
    print("-" * 50)  # 구분선 출력
```

```python
# 유사한 문서들의 메타데이터에서 카테고리와 운용사 정보 추출
categories = set()
companies = set()

for doc, score in similar_docs:
    # 벡터 검색된 문서의 메타데이터에서 정보 추출
    if 'category' in doc.metadata:
        categories.add(doc.metadata['category'])
    if 'company' in doc.metadata:
        companies.add(doc.metadata['company'])

# 쿼리에서 언급된 조건들도 직접 추가
target_company = "삼성자산운용"
target_categories = ["통화-미국달러", "채권-혼합-단기"]  # 가능한 카테고리 표현들

# ETF 노드에서 조건에 맞는 상품 검색 (특정 운용사가 관리하는 ETF 중 카테고리가 일치하는 상품)
etf_query = """
MATCH (e:ETF)
WHERE e.company = $target_company  // 자산운용사 매칭
  AND e.category IN $target_categories  // 카테고리 매칭
WITH e
ORDER BY toFloat(e.netAsset) DESC
LIMIT 10
RETURN 
    e.name AS name,
    e.company AS company,
    e.category AS category,
    e.baseIndex AS baseIndex,
    e.yearReturn AS yearReturn,
    e.totalFee AS totalFee,
    toFloat(e.netAsset) AS netAsset
"""

params = {
    "target_company": target_company,
    "target_categories": target_categories,
}

etf_results = graph.query(etf_query, params=params)
etf_df = pd.DataFrame(etf_results)
print(f"검색된 ETF 상품: {len(etf_df)}개")
etf_df.head(10)
```

```python
# ETF 전용 벡터 검색 설정
from langchain_openai import OpenAIEmbeddings
from langchain_neo4j import Neo4jVector

embeddings = OpenAIEmbeddings(model="text-embedding-3-small") 

vector_db = Neo4jVector.from_existing_index(
    embeddings,
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    index_name="etf_vector_index",  # ETF 벡터 인덱스
    node_label="ETF",  # ETF 노드 레이블
    text_node_property="name",  # 기본 텍스트 속성 (ETF 이름)
    embedding_node_property="embedding",  # 임베딩 속성
    retrieval_query="""
    // 기본 ETF 노드와 점수
    WITH node, score
    
    // ETF와 관련된 모든 정보 수집
    OPTIONAL MATCH (node)-[:BELONGS_TO]->(category:Category)
    OPTIONAL MATCH (node)-[:MANAGED_BY]->(company:Company)
    
    // 관련 ETF들도 찾기 (같은 카테고리에 속하는)
    OPTIONAL MATCH (node)-[:BELONGS_TO]->(cat:Category)<-[:BELONGS_TO]-(related_etf:ETF)
    WHERE related_etf <> node       // 같은 ETF 제외
    
    // 집계 함수를 사용하여 관련 ETF들 수집
    WITH node, score,
         collect(DISTINCT related_etf.name)[0..3] AS related_etfs   // 관련 ETF 3개만 (중복 제외)
    
    // 검색용 종합 텍스트 생성
    WITH node, score, related_etfs,
         "ETF 종목: " + node.name + "\n" + 
         "- 운용사: " + COALESCE(node.company, "N/A") + "\n" +
         "- 기초지수: " + COALESCE(node.baseIndex, "N/A") + "\n" +
         "- 카테고리: " + COALESCE(node.category, "N/A") + "\n" +
         "- 관련 ETF:" + apoc.text.join(related_etfs, ", ") + "\n" AS combined_text

    RETURN combined_text AS text,   // 검색 결과 텍스트 (Langchain Document의 page_content 속성)
           score,   // 벡터 유사도 점수
           {
               etf_name: node.name,
               company: node.company,
               base_index: node.baseIndex,
               category: node.category,
               listing_date: node.listingDate,
               expense_ratio: node.expenseRatio,
               nav: node.nav,
               related_etfs: related_etfs,
               etf_code: node.code
           } AS metadata    // 메타데이터 (Langchain Document의 metadata 속성)
    """
)


# 유사도 검색 테스트
def search_etfs(query, k=5):
    """ETF 검색 함수"""
    results = vector_db.similarity_search(query, k=k)
    
    print(f"검색어: '{query}'")
    print(f"검색 결과 ({len(results)}개):")
    print("-" * 50)
    
    for i, result in enumerate(results, 1):
        metadata = result.metadata
        print(f"{i}.")
        print(f"{result.page_content}") # 페이지 내용 출력           
        print(f"상장일: {metadata.get('listing_date', 'N/A')}")
        print()


# 일반적인 카테고리 검색
search_etfs("국내 주식 ETF")
```

```python
# 특정 운용사 검색  
search_etfs("삼성자산운용 ETF")
```

```python
# 특정 지수 추적 ETF 검색
search_etfs("코스피200 추적 ETF")
```

```python
# 배당 관련 ETF 검색
search_etfs("배당 ETF")
```
