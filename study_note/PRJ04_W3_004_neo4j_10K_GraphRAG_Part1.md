# GraphRAG with Neo4j - Part 1: 기초와 환경 설정

**학습 자료**: PRJ04_W3_004 - SEC 10-K 문서로 지식 그래프 기반 RAG 시스템 구축

---

## 📚 학습 목표

이 실습을 완료하면 다음을 할 수 있습니다:

1. **GraphRAG 개념 이해**: 전통적 RAG와 GraphRAG의 차이점 및 장점 파악
2. **Neo4j 환경 구축**: AuraDB 클라우드 인스턴스 생성 및 LangChain 연동
3. **문서 처리 파이프라인 구현**: PDF 문서 로딩, 청킹, 임베딩 생성
4. **지식 그래프 구축**: 문서를 노드와 관계로 변환하여 Neo4j에 저장
5. **기본 검색 시스템 구현**: 벡터 검색 및 그래프 검색 결합

---

## 🔑 핵심 개념

### GraphRAG란?

**GraphRAG (Graph Retrieval-Augmented Generation)**는 지식 그래프를 활용하여 검색 증강 생성을 수행하는 고급 RAG 기법입니다.

#### 전통적 RAG vs GraphRAG

| 구분 | 전통적 RAG | GraphRAG |
|------|-----------|----------|
| **데이터 구조** | 벡터 임베딩만 저장 | 벡터 + 그래프 관계 |
| **검색 방식** | 벡터 유사도 검색 | 벡터 + 그래프 순회 |
| **컨텍스트 이해** | 단편적 정보 | 관계 기반 맥락 |
| **복잡한 질문** | 제한적 | 다단계 추론 가능 |

#### GraphRAG의 장점

1. **관계 기반 검색**: 문서 간 연결을 활용하여 더 풍부한 컨텍스트 제공
2. **다단계 추론**: 그래프 순회를 통한 복잡한 질의응답
3. **설명 가능성**: 검색 경로를 그래프로 시각화 가능
4. **확장성**: 새로운 정보를 그래프에 쉽게 추가

### SEC 10-K 문서

- **10-K 보고서**: 미국 증권거래위원회(SEC)에 제출하는 연간 재무 보고서
- **내용**: 기업의 재무 상태, 사업 개요, 리스크 요인, 경영진 논의 등
- **활용**: 기업 분석, 투자 결정, 리스크 평가에 사용

---

## 🛠 환경 설정

### 필수 라이브러리 설치

```bash
# 핵심 라이브러리
pip install langchain langchain-community langchain-openai langchain-neo4j

# Neo4j 및 벡터 검색
pip install neo4j

# 문서 처리
pip install pypdf unstructured

# 유틸리티
pip install python-dotenv
```

### 환경 변수 설정

`.env` 파일에 다음 정보를 설정합니다:

```bash
# OpenAI API
OPENAI_API_KEY=your-openai-key

# Neo4j AuraDB
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=neo4j
```

### Neo4j AuraDB 클라우드 인스턴스 생성

1. [Neo4j Aura](https://neo4j.com/cloud/aura/) 접속
2. 무료 인스턴스 생성 (Free Tier 사용)
3. 연결 정보 저장 (URI, Username, Password)
4. `.env` 파일에 연결 정보 입력

---

## 💻 단계별 구현


 # **테슬라 10-K 보고서 RAG**



 이 노트북은 테슬라의 10-K 재무 보고서 데이터를 Neo4j 지식 그래프(Knowledge Graph)에 저장하고,

 이를 기반으로 벡터 검색(Vector Search)과 그래프 탐색을 결합한 RAG(Retrieval-Augmented Generation) 시스템을 구축하는 과정을 다룹니다.



 1. **지식 그래프 (Knowledge Graph)**: 데이터를 노드(Node)와 관계(Relationship)로 표현하여 데이터 간의 맥락과 구조를 보존합니다.

 2. **벡터 검색 (Vector Search)**: 텍스트의 의미를 벡터로 변환하여 유사한 의미를 가진 데이터를 찾습니다.

 3. **Graph RAG**: 단순 벡터 검색을 넘어, 그래프의 관계(예: 이전/다음 청크, 상위 섹션 정보)를 활용하여 더 풍부한 문맥을 LLM에 제공합니다.



 ---

 ## 1. 환경 설정



 필요한 라이브러리를 로드하고, Neo4j 데이터베이스 연결을 설정합니다.

 `(1) Env 환경변수`

 - `.env` 파일에서 API 키(OpenAI, Neo4j 등)를 로드

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

# 불필요한 경고 메시지 숨기기
warnings.filterwarnings('ignore')
```

 `(3) Neo4j 설정`

 - `LangChain`의 `Neo4jGraph` 래퍼를 사용하여 Neo4j 데이터베이스에 연결합니다.

 - 이 객체를 통해 Cypher 쿼리를 실행하고 그래프 데이터를 관리합니다.

```python
from langchain_neo4j import Neo4jGraph

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# LangChain 도구 활용 - DB 연결 객체 초기화 
graph = Neo4jGraph( 
    url=NEO4J_URI, 
    username=NEO4J_USERNAME, 
    password=NEO4J_PASSWORD,
    database="neo4j",
    refresh_schema=True,  # 스키마 자동 갱신
    sanitize=True,  # 쿼리 검증 강화
    enhanced_schema=True  # 향상된 스키마 정보 제공
)

# 연결 확인을 위해 임의의 노드 5개 조회
try:
    graph.query("MATCH (n) RETURN n LIMIT 5;")
    print("Neo4j 연결 성공")
except Exception as e:
    print(f"Neo4j 연결 실패: {e}")
```

 `(4) 기존 DB의 모든 내용 삭제`

 - 실습을 위해 기존 데이터베이스를 깨끗하게 초기화합니다.

 - **주의**: 실제 운영 환경에서는 절대 함부로 실행하면 안 됩니다.

 - `APOC` 라이브러리 의존성을 없애기 위해 순수 Cypher 쿼리로 구현되었습니다.

```python
def reset_database(graph):
    """
    데이터베이스 초기화 함수
    1. 모든 노드와 관계 삭제 (DETACH DELETE)
    2. 모든 제약조건(Constraints) 삭제
    3. 모든 인덱스(Indexes) 삭제
    """
    print("데이터베이스 초기화를 시작합니다...")
    
    # 1. 모든 노드와 관계 삭제
    graph.query("MATCH (n) DETACH DELETE n")
    
    # 2. 모든 제약조건 삭제 (제약조건과 함께 관련된 인덱스도 삭제됨)
    constraints = graph.query("SHOW CONSTRAINTS")
    for constraint in constraints:
        constraint_name = constraint.get("name")
        if constraint_name:
            graph.query(f"DROP CONSTRAINT {constraint_name}")
    
    # 3. 모든 인덱스 삭제 (제약조건 타입이 아닌 경우)
    indexes = graph.query("SHOW INDEXES")
    for index in indexes:
        index_name = index.get("name")
        index_type = index.get("type")
        if index_name and index_type != "CONSTRAINT":
            graph.query(f"DROP INDEX {index_name}")
    
    print("데이터베이스가 초기화되었습니다.")

# 데이터베이스 초기화 실행
reset_database(graph)
```

```python
# 그래프 스키마 조회 (초기화 확인)
graph.refresh_schema()
print(graph.schema)
```

 ## 2. 지식그래프 스키마 설계



 데이터를 구조화하여 저장하기 위한 스키마를 정의합니다.


 * **주요 엔티티 (노드)**:

    1. **Document (문서)**: 10-K 보고서 전체를 나타내는 최상위 노드.

    2. **Section (섹션)**: 문서의 주요 챕터 (예: 'Business', 'Risk Factors').

    3. **Chunk (청크)**: 실제 텍스트 데이터가 담긴 최소 단위. 벡터 검색의 대상.



 * **관계 (Relationships)**:

    1. **HAS_SECTION**: Document -> Section (포함 관계)

    2. **CONTAINS**: Section -> Chunk (포함 관계)

    3. **NEXT**: Chunk -> Chunk (순서 관계). 문맥 파악을 위해 중요.



 * **제약조건 (Constraints)**:

    - 데이터 무결성을 보장하고 검색 속도를 높이기 위해 고유 키(Unique Key) 등을 설정.

```python
# 1. Document 노드 제약조건: id는 고유해야 함
cypher_query = """
CREATE CONSTRAINT IF NOT EXISTS
FOR (d:Document)
REQUIRE d.id IS UNIQUE;
"""
graph.query(cypher_query)

# 2. Section 노드 제약조건: (name, document_id) 조합이 고유해야 함 (복합키)
# 같은 이름의 섹션이라도 문서가 다르면 다른 섹션으로 취급
cypher_query = """
CREATE CONSTRAINT IF NOT EXISTS
FOR (s:Section)
REQUIRE (s.name, s.document_id) IS NODE KEY;
"""
graph.query(cypher_query)

# 3. Chunk 노드 제약조건: chunk_id는 고유해야 함
cypher_query = """
CREATE CONSTRAINT IF NOT EXISTS
FOR (c:Chunk)
REQUIRE c.chunk_id IS UNIQUE;
"""
graph.query(cypher_query)
```

```python
# 4. 벡터 인덱스 생성
# Chunk 노드의 'embedding' 속성에 대해 벡터 인덱스를 생성합니다.
# 이를 통해 코사인 유사도 기반의 시맨틱 검색이 가능해집니다.
cypher_query = """
CREATE VECTOR INDEX chunk_content_index IF NOT EXISTS
FOR (c:Chunk) 
ON (c.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1536,      // OpenAI text-embedding-3-small 차원 수
    `vector.similarity_function`: 'cosine'
  }
}
"""
graph.query(cypher_query)
```

 ## 3. 데이터 구조화 및 저장



 전처리된 데이터를 로드하고, 정의한 스키마에 맞춰 Neo4j에 노드와 관계를 생성합니다.


 * **노드 구조**:
   - **Document 노드**: 전체 10-K 문서 표현
   - **Section 노드**: 문서의 각 섹션 표현 (Business, Risk Factors 등)
   - **Chunk 노드**: 분할된 텍스트 청크

* **관계 구조**:
   - `(:Document)-[:HAS_SECTION]->(:Section)`
   - `(:Section)-[:CONTAINS]->(:Chunk)`
   - `(:Chunk)-[:NEXT]->(:Chunk)` 

```python
import pickle

# 저장된 섹션 데이터 로드 (사전에 전처리된 pickle 파일)
# 데이터 형식: { "SectionName": [Chunk1, Chunk2, ...], ... }
data_path = "data/tesla_10k_sections_split.pkl"
if os.path.exists(data_path):
    with open(data_path, "rb") as f: 
        section_docs_split = pickle.load(f)
    print(f"Number of sections: {len(section_docs_split)}")
else:
    print(f"Error: {data_path} 파일을 찾을 수 없습니다.")
    # 더미 데이터나 예외 처리가 필요할 수 있음
```

```python
# 데이터 샘플 확인
if 'section_docs_split' in locals():
    print("Sections:", section_docs_split.keys())
    if "Business" in section_docs_split:
        print("Sample Chunk Metadata:", section_docs_split["Business"][0].metadata)
```

```python
# 문서 ID 추출 (파일명 등에서 추출)
# 모든 섹션이 동일한 문서에서 왔다고 가정
if 'section_docs_split' in locals():
    first_doc = next(iter(section_docs_split.values()))[0]
    doc_id = os.path.split(first_doc.metadata['source'])[-1].split(".")[0]
    print(f"Document ID: {doc_id}")
```

```python
import uuid
from langchain_openai import OpenAIEmbeddings

# OpenAI Embeddings 객체 생성
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# --- 노드 및 관계 생성 함수 정의 ---

def create_document_node(
        graph: Neo4jGraph,   # Neo4j 그래프 객체
        doc_id: str,         # 문서 ID
        source: str          # 문서 소스
    ):
    """Document 노드 생성 (MERGE를 사용하여 중복 방지)"""
    query = """
    MERGE (d:Document {id: $doc_id})
    SET d.source = $source
    RETURN d
    """
    return graph.query(query, params={"doc_id": doc_id, "source": source})

def create_section_node(
        graph: Neo4jGraph,   # Neo4j 그래프 객체
        section_name: str,   # 섹션 이름
        doc_id: str          # 문서 ID
    ):
    """Section 노드 생성 및 Document와의 관계 연결"""
    query = """
    MATCH (d:Document {id: $doc_id})
    MERGE (s:Section {name: $section_name, document_id: $doc_id})
    MERGE (d)-[:HAS_SECTION]->(s)
    RETURN s
    """
    return graph.query(query, params={"section_name": section_name, "doc_id": doc_id})

def create_chunk_node(
        graph: Neo4jGraph,   # Neo4j 그래프 객체
        section_name: str,   # 섹션 이름
        doc_id: str,         # 문서 ID
        chunk_id: str,       # 청크 ID
        content: str,        # 청크 내용
        embedding: np.ndarray, # 임베딩 벡터
        metadata: dict       # 청크 메타데이터
    ):
    """Chunk 노드 생성, 벡터 저장, Section과의 관계 연결"""
    if hasattr(embedding, "tolist"):  
        embedding = embedding.tolist()
    
    # db.create.setNodeVectorProperty 프로시저를 사용하여 벡터를 저장 (벡터 임베딩 생성)
    query = """
    MATCH (s:Section {name: $section_name, document_id: $doc_id})
    CREATE (c:Chunk {
        chunk_id: $chunk_id,  // 청크 ID
        content: $content,    // 청크 내용
        order: $order,        // 청크 순서
        element_id: $element_id, // 청크 요소 ID
        parent_id: $parent_id, // 청크 부모 ID
        document_id: $doc_id, // 문서 ID
        section_name: $section_name, // 섹션 이름
        section_start_page: $page_number // 섹션 시작 페이지
    })
    WITH c, s  // c: 청크 노드, s: 섹션 노드
    CALL db.create.setNodeVectorProperty(c, 'embedding', $embedding)  // 벡터 임베딩 생성
    CREATE (s)-[:CONTAINS]->(c)  // 섹션과 청크의 관계 연결
    RETURN c
    """
    return graph.query(
        query,
        params={
            "chunk_id": chunk_id,
            "content": content,
            "embedding": embedding,
            "order": metadata.get('order'),
            "element_id": metadata.get('element_id'),
            "parent_id": metadata.get('parent_id'),
            "page_number": metadata.get('page_number'),
            "section_name": section_name,
            "doc_id": doc_id
        }
    )

def create_next_relationship(
    graph: Neo4jGraph,   # Neo4j 그래프 객체
    prev_chunk_id: str,  # 이전 청크 ID
    next_chunk_id: str   # 다음 청크 ID
    ):
    """Chunk 간의 순서(NEXT) 관계 생성"""
    query = """
    MATCH (prev:Chunk {chunk_id: $prev_chunk_id})
    MATCH (next:Chunk {chunk_id: $next_chunk_id})
    MERGE (prev)-[:NEXT]->(next)
    """
    return graph.query(query, params={"prev_chunk_id": prev_chunk_id, "next_chunk_id": next_chunk_id})
```

```python
# --- 데이터 적재 실행 ---
# 각 섹션과 청크를 순회하며 그래프 DB에 저장

if 'section_docs_split' in locals():
    for section_name, chunks in section_docs_split.items():
        # 1. Document 노드 (이미 존재하면 건너뜀)
        doc_id = chunks[0].metadata['source'] # 파일 경로 전체를 ID로 사용
        create_document_node(graph, doc_id, doc_id)

        # 2. Section 노드
        create_section_node(graph, section_name, doc_id)    
        print(f"Processing section: {section_name}")

        prev_chunk_id = None     # 청크 ID 추적을 위한 변수
        
        # 3. Chunk 노드 및 관계
        for chunk in chunks:
            chunk_id = str(uuid.uuid4())    # 청크 ID 생성
            
            # 임베딩 생성 
            embedding = embeddings.embed_query(chunk.page_content)
            
            # Chunk 노드 생성
            create_chunk_node(
                graph, section_name, doc_id, chunk_id, 
                chunk.page_content, embedding, chunk.metadata
            )  
            
            # 이전 청크와 NEXT 관계 연결 (Linked List 구조)
            if prev_chunk_id:
                create_next_relationship(graph, prev_chunk_id, chunk_id)
            
            # 현재 청크 ID를 이전 청크 ID로 저장
            prev_chunk_id = chunk_id

        # 저장 확인 (저장된 청크 수 확인)
        count = graph.query(
            "MATCH (c:Chunk {section_name: $section_name}) RETURN COUNT(c) AS count",
            params={"section_name": section_name}
        )[0]['count']
        print(f"  - Saved {count} chunks.")

    print("모든 데이터 적재 완료.")
```

```python
# 전체 노드 수 확인
print("Total Chunks:", graph.query("MATCH (c:Chunk) RETURN COUNT(c) AS count")[0]['count'])
print("Total Sections:", graph.query("MATCH (s:Section) RETURN COUNT(s) AS count")[0]['count'])
```

 ## 4. 벡터 검색 및 RAG 구현



 구축된 지식 그래프를 활용하여 질문에 답하는 RAG 시스템을 구현합니다.

 단순 벡터 검색뿐만 아니라, 그래프 구조(윈도우 검색)를 활용하여 문맥을 보강합니다.

```python
from langchain_neo4j import Neo4jVector
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Neo4jVector 초기화
# retrieval_query: 벡터 검색 후 실행할 추가 Cypher 쿼리입니다.
# 여기서 그래프의 강점이 드러납니다. 단순히 유사한 청크 하나만 가져오는 것이 아니라,
# 그 청크의 '이전'과 '다음' 청크(window)를 함께 가져와 문맥을 풍부하게 만듭니다.
vector_store = Neo4jVector.from_existing_index(
    embedding=embeddings,
    url=NEO4J_URI,
    username=NEO4J_USERNAME, 
    password=NEO4J_PASSWORD,
    index_name="chunk_content_index",
    node_label="Chunk",
    text_node_property="content",
    embedding_node_property="embedding",
    retrieval_query="""
    MATCH (c:Chunk {chunk_id: node.chunk_id})   // 청크 ID 기반 매칭
    OPTIONAL MATCH (c)<-[:CONTAINS]-(s:Section)  // 섹션 매칭
    OPTIONAL MATCH window = (prev:Chunk)-[:NEXT*0..1]->(c)-[:NEXT*0..1]->(next:Chunk)  // 이전/다음 청크 매칭 (가변 길이 경로 탐색)
    WITH DISTINCT c, s, score, nodes(window) AS context_nodes  // 고유 청크 노드 추출
    WITH 
      // 청크 노드의 내용을 결합하여 문맥 텍스트 생성
      apoc.text.join([chunk IN context_nodes | chunk.content], '\n\n') AS context_text,
      s.name AS section_name,  // 섹션 이름
      c.section_start_page AS section_page,  // 섹션 시작 페이지
      c.order AS chunk_order,  // 청크 순서
      score  // 점수
    RETURN DISTINCT context_text AS text, score, {
      section: section_name,  // 섹션 이름
      section_page: section_page,  // 섹션 시작 페이지
      chunk_order: chunk_order  // 청크 순서
    } AS metadata
    """
)
```

```python
# 중복 제거 함수
# 윈도우 검색을 사용하면 인접한 청크들이 여러 번 검색될 수 있으므로 중복을 제거합니다.
def remove_duplicates(docs):
    unique_results = []
    seen_ids = set()
    for doc in docs:
        # 섹션과 청크 순서를 조합하여 고유 ID로 사용
        unique_id = (doc.metadata.get('section'), doc.metadata.get('chunk_order'))
        if unique_id not in seen_ids:
            seen_ids.add(unique_id)
            unique_results.append(doc)
    return unique_results
```

```python
# RAG 체인 구성
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) # 최신 모델 사용 권장

template = """
당신은 테슬라 10-K 보고서 분석 전문가입니다. 
아래 제공된 [보고서 내용]을 바탕으로 [질문]에 대해 상세하고 정확하게 답변해 주세요.
답변 시 정보의 출처(섹션, 페이지 등)를 명시하면 신뢰도가 높아집니다.

<보고서 내용>
{context}
</보고서 내용>

<질문>
{question}
</질문>
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

def format_docs(docs):
    return "\n\n".join([
        f"--- 출처: {d.metadata.get('section', 'N/A')} (p.{d.metadata.get('section_page', 'N/A')}) ---\n{d.page_content}" 
        for d in docs
    ])

# Retriever 설정 (k=5)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

rag_chain = (
    {
        "context": retriever | remove_duplicates | format_docs, 
        "question": RunnablePassthrough()
    }
    | prompt 
    | llm 
    | StrOutputParser()
)
```

```python
# 테스트 실행
query = "What recognition did Tesla receive in the 2024 American Opportunity Index?"
print(f"질문: {query}")
response = rag_chain.invoke(query)
print("\n답변:")
print(response)
```

```python
pprint(section_docs_split['Business'][13].page_content)
```

```python
# 질문에 대한 답변 생성
test_query = "테슬라의 2024년 HR 정책에 대해서 설명해 주세요."
response = rag_chain.invoke(test_query)

print(response)
```
