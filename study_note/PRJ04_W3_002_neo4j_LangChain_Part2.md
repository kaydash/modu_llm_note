# LangChain Neo4j 통합 - Part 2

**학습 자료**: PRJ04_W3_002 - Neo4j 벡터 검색 및 Text-to-Cypher

---

## 📚 학습 목표

이 실습을 완료하면 다음을 할 수 있습니다:

1. **Neo4jVector로 벡터 검색 구현**: 그래프 데이터베이스에서 의미론적 검색 수행
2. **하이브리드 검색 활용**: 벡터 유사도와 그래프 관계를 결합한 고급 검색
3. **GraphCypherQAChain 사용**: 자연어를 Cypher 쿼리로 자동 변환
4. **실무 RAG 시스템 구축**: 지식 그래프 기반 검색 증강 생성 구현

---

## 🔑 핵심 개념

### Neo4jVector - 벡터 검색

- **의미론적 검색**: 텍스트 임베딩을 활용한 유사도 기반 검색
- **하이브리드 검색**: 벡터 유사도 + 그래프 관계를 결합하여 더 정확한 검색
- **메타데이터 필터링**: 그래프 속성을 활용한 조건부 검색

### GraphCypherQAChain - Text-to-Cypher

- **자연어-쿼리 변환**: LLM을 활용하여 자연어 질문을 Cypher 쿼리로 변환
- **컨텍스트 인식 답변**: 쿼리 결과를 기반으로 자연어 답변 생성
- **그래프 QA 시스템**: 복잡한 관계 정보를 활용한 질의응답

---

## 🛠 환경 설정

### Part 1 선행 학습 필요

이 Part 2는 Part 1의 내용을 기반으로 합니다:
- Neo4jGraph 연결 설정
- 기본 Cypher 쿼리 실행
- 환경 변수 설정

### 추가 준비 사항

- Part 1에서 생성한 Neo4j 데이터베이스 연결 유지
- OpenAI API 키 설정 (임베딩 및 LLM 사용)

---

## 💻 단계별 구현


---

## **3. Neo4jVector - 벡터 검색**

**Neo4jVector**는 **하이브리드 검색**(벡터 + 그래프 관계)을 지원하여 더욱 정확한 의미론적 검색을 제공합니다.


```python
# Article 노드를 조회하는 쿼리

query = "MATCH (n:Article) RETURN n"

articles = graph.query(query)

print(f"Article 노드 개수: {len(articles)}")

if len(articles) > 0:
    print(articles[0])
else:
    print("⚠️ Article 노드가 없습니다. 첫 번째 노트북(PRJ04_W3_001_neo4j_intro.ipynb)을 먼저 실행하여 데이터를 생성하세요.")
```

### **3.1 기존 데이터를 활용한 벡터 스토어 구축**

```python
from langchain_core.documents import Document

# 랭체인 문서로 변환 
print(f"Article 노드 개수: {len(articles)}")

if len(articles) > 0:
    docs = [
        Document(
            page_content=f"{article['n'].get('title', '')}\n\n{article['n']['content']}",
            metadata={
                "author": article['n']['author'] or "Unknown",
                "publishDate": article['n']['publishDate'] or "Unknown",
                "source": article['n']['source'] or "Unknown",
            },
        )
        for article in articles
    ]

    print(f"문서 개수: {len(docs)}")
    print(docs[0].page_content)
    print(docs[0].metadata)
else:
    print("⚠️ Article 노드가 없어 문서를 생성할 수 없습니다.")
    docs = []
```

```python
from langchain_openai import OpenAIEmbeddings
from langchain_neo4j import Neo4jVector

if len(docs) > 0:
    # 임베딩 모델 초기화
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",  # 최신 모델
        show_progress_bar=True  # 진행률 표시
    )

    # LangChain 도구 활용 - Neo4j 벡터 스토어 생성
    db = Neo4jVector.from_documents(
        docs,
        embeddings,
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        index_name="article_embeddings",  # 인덱스 이름 명시
        text_node_property="content",  # 텍스트 속성명
        embedding_node_property="embedding",  # 임베딩 속성명
    )
    print("✅ 벡터 스토어 생성 완료")
else:
    print("⚠️ 문서가 없어 벡터 스토어를 생성하지 않습니다. Cell 35-38은 건너뛰세요.")
    db = None
```

### **3.2 다양한 검색 전략**

```python
# 1. 기본 유사도 검색
if db is not None:
    query = "AI의 발전 방향에 대한 기사"

    basic_results = db.similarity_search(query, k=2)

    for doc in basic_results:
        print(f"문서 내용: {doc.page_content[:100]}...")  # 내용 일부 출력
        print(f"메타데이터: {doc.metadata}")  # 메타데이터 출력
        print("-" * 40)  # 구분선 출력
else:
    print("⚠️ 벡터 스토어가 생성되지 않아 검색을 수행할 수 없습니다.")
```

```python
# 2. 점수 포함 유사도 검색
if db is not None:
    scored_results = db.similarity_search_with_score(query, k=2)    

    for doc, score in scored_results:
        print(f"문서 내용: {doc.page_content[:100]}...")  # 내용 일부 출력
        print(f"점수: {score}")  # 점수 출력
        print("-" * 40)  # 구분선 출력
else:
    print("⚠️ 벡터 스토어가 생성되지 않아 검색을 수행할 수 없습니다.")
```

```python
# 3. 메타데이터 필터링
if db is not None:
    filtered_results = db.similarity_search(
        query,  
        k=2,
        filter={"source": "테크경제신문"}  # 특정 출처로 필터링
    )

    for doc in filtered_results:
        print(f"문서 내용: {doc.page_content[:100]}...")  # 내용 일부 출력
        print(f"메타데이터: {doc.metadata}")  # 메타데이터 출력
        print("-" * 40)  # 구분선 출력
else:
    print("⚠️ 벡터 스토어가 생성되지 않아 검색을 수행할 수 없습니다.")
```

```python
# 4. MMR (Maximal Marginal Relevance) 검색 - 다양성 고려
if db is not None:
    mmr_results = db.max_marginal_relevance_search(query, k=2, fetch_k=4)

    for doc in mmr_results:
        print(f"문서 내용: {doc.page_content[:100]}...")  # 내용 일부 출력
        print(f"메타데이터: {doc.metadata}")  # 메타데이터 출력
        print("-" * 40)  # 구분선 출력
else:
    print("⚠️ 벡터 스토어가 생성되지 않아 검색을 수행할 수 없습니다.")
```

### **[실습 5] Neo4jVector 벡터 스토어 구축** 

- 다음 문서 목록을 이용하여 `Neo4jVector` 벡터 스토어를 구축하고, "LangChain Neo4j의 장점"과 유사한 문서를 검색하는 코드를 작성하세요. 
- OpenAI Embeddings 모델을 사용하고, 유사도 검색 결과를 문서 내용과 함께 출력하세요.

```python
from langchain_core.documents import Document

documents = [
    Document(page_content="LangChain Neo4j는 Neo4j 그래프 데이터베이스와 LangChain을 통합합니다."),
    Document(page_content="Neo4j는 그래프 데이터 모델을 사용하는 NoSQL 데이터베이스입니다."),
    Document(page_content="LangChain은 LLM 기반 애플리케이션 개발을 위한 프레임워크입니다."),
    Document(page_content="Neo4jVector는 LangChain과 Neo4j를 연결하여 벡터 검색 기능을 제공합니다."),
    Document(page_content="LangChain Neo4j를 사용하면 지식 그래프 기반 RAG 시스템을 쉽게 구축할 수 있습니다."),
    Document(page_content="그래프 데이터베이스는 관계형 데이터베이스와 다르게 노드와 관계로 데이터를 표현합니다."),
    Document(page_content="RAG (Retrieval-Augmented Generation)는 검색 증강 생성 모델로, 외부 지식 기반을 활용합니다."),
    Document(page_content="Neo4j Cypher는 그래프 데이터베이스 쿼리 언어로, SQL과 유사하지만 그래프 구조에 최적화되어 있습니다."),
    Document(page_content="LangChain의 Agent 기능은 LLM이 도구와 상호작용하여 복잡한 태스크를 해결할 수 있게 합니다."),
    Document(page_content="지식 그래프는 정보를 구조화된 방식으로 표현하여 컨텍스트 기반 추론을 가능하게 합니다."),
    Document(page_content="벡터 임베딩은 텍스트를 수치적 벡터로 변환하여 의미적 유사성을 계산할 수 있게 합니다."),
    Document(page_content="하이브리드 검색은 키워드 기반 검색과 의미론적 검색을 결합하여 검색 품질을 향상시킵니다."),
    Document(page_content="Neo4j APOC는 그래프 데이터베이스 작업을 위한 확장 라이브러리입니다."),
    Document(page_content="LangChain 체인은 여러 LLM 컴포넌트를 연결하여 복잡한 워크플로우를 구성할 수 있습니다."),
    Document(page_content="그래프 RAG는 단순 벡터 검색보다 관계 정보를 활용하여 더 정확한 정보 검색이 가능합니다."),
    Document(page_content="Neo4j Bloom은 그래프 데이터를 시각화하고 탐색할 수 있는 도구입니다."),
    Document(page_content="프롬프트 엔지니어링은 LLM에게 효과적인 지시를 제공하는 기술입니다."),
    Document(page_content="LangChain 메모리 컴포넌트는 대화 이력을 관리하여 문맥을 유지합니다."),
    Document(page_content="Neo4j GraphQL은 그래프 데이터베이스와 GraphQL을 통합하여 API 개발을 단순화합니다."),
    Document(page_content="지식 그래프 임베딩은 그래프 구조를 벡터 공간에 매핑하는 기술입니다."),
    Document(page_content="LLM 추론은 대규모 언어 모델이 생성한 출력의 신뢰성과 정확성을 검증하는 과정입니다."),
    Document(page_content="Neo4j Aura는 클라우드 기반 그래프 데이터베이스 서비스입니다."),
    Document(page_content="멀티모달 RAG는 텍스트뿐만 아니라 이미지, 오디오 등 다양한 데이터 형식을 처리할 수 있습니다."),
    Document(page_content="LangChain 문서 로더는 다양한 소스에서 데이터를 가져오는 기능을 제공합니다."),
    Document(page_content="그래프 알고리즘은 관계 패턴을 분석하여 중요한 인사이트를 도출합니다."),
]

# 여기에 코드를 작성하세요.

from langchain_openai import OpenAIEmbeddings
from langchain_neo4j import Neo4jVector
import os

# 임베딩 모델 초기화
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=True
)

# Neo4jVector 벡터 스토어 구축
vector_store = Neo4jVector.from_documents(
    documents,
    embeddings,
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    index_name="practice_embeddings",
    text_node_property="text",
    embedding_node_property="embedding"
)

# 유사도 검색 실행
query = "LangChain Neo4j의 장점"
results = vector_store.similarity_search(query, k=3)

# 결과 출력
print(f"검색 질의: {query}\n")
print("검색 결과:")
for i, doc in enumerate(results, 1):
    print(f"\n{i}. {doc.page_content}")
```

---

## **4. GraphCypherQAChain**

**GraphCypherQAChain**은 자연어를 정확한 Cypher 쿼리로 변환하고, **컨텍스트 인식** 답변을 생성합니다.

```python
from langchain_openai import ChatOpenAI
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph

# LangChain 도구 활용 - LLM 및 그래프 객체 초기화
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)

# LangChain 도구 활용 - GraphCypherQAChain 객체 초기화
chain = GraphCypherQAChain.from_llm(
    llm=llm, 
    graph=graph, 
    allow_dangerous_requests=True,
    verbose=True,)

result = chain.run("엔비디아 관련 기사를 작성한 기자는 누구인가요?")
```

```python
# 결과 출력
print(result)
```

### **[실습 6] GraphCypherQAChain** 

- `GraphCypherQAChain`을 사용하여 영화에 대한 자연어 질문에 답변하는 코드를 작성하세요. 
- 예시 질문: 삼성전자 관련 기사를 작성한 기자와 소속 언론사는 어디인가요?

```python
# 여기에 코드를 작성하세요.

from langchain_openai import ChatOpenAI
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
import os

# LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# Neo4jGraph 초기화
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)

# GraphCypherQAChain 초기화
chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    allow_dangerous_requests=True,
    verbose=True
)

# 자연어 질문 실행
question = "삼성전자 관련 기사를 작성한 기자와 소속 언론사는 어디인가요?"
result = chain.run(question)

# 결과 출력
print(f"\n질문: {question}")
print(f"\n답변: {result}")
```

---

## 🚀 실무 활용 예시

### 1. 지식 그래프 기반 RAG 시스템

```python
from langchain_neo4j import Neo4jVector, GraphCypherQAChain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 벡터 스토어를 리트리버로 설정
retriever = vector_store.as_retriever(search_kwargs={'k': 3})

# RAG 체인 구성
template = """다음 컨텍스트를 바탕으로 질문에 답변하세요:

컨텍스트: {context}

질문: {question}

답변:"""

prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model="gpt-4o-mini")

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 실행
result = rag_chain.invoke("LangChain Neo4j의 주요 장점은 무엇인가요?")
print(result)
```

### 2. 하이브리드 검색 시스템

```python
# 벡터 검색 + 그래프 관계 결합
hybrid_query = """
// 1단계: 벡터 유사도 검색
CALL db.index.vector.queryNodes('article_embeddings', 5, $embedding)
YIELD node, score

// 2단계: 관련 노드 확장 (그래프 관계 활용)
MATCH (node)-[:MENTIONS]->(company:Company)
MATCH (node)-[:COVERS]->(topic:Topic)

// 3단계: 결과 집계
RETURN 
    node.content as content,
    score,
    collect(DISTINCT company.name) as companies,
    collect(DISTINCT topic.name) as topics
ORDER BY score DESC
"""
```

### 3. 대화형 QA 시스템

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# 메모리와 리트리버를 결합한 대화형 체인
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    retriever=retriever,
    memory=memory
)

# 연속적인 대화
print(qa_chain.invoke({"question": "Neo4j의 주요 기능은?"})["answer"])
print(qa_chain.invoke({"question": "그것을 LangChain과 어떻게 통합하나요?"})["answer"])
```

---

## 📖 참고 자료

### 공식 문서

- [LangChain Neo4j Integration](https://python.langchain.com/docs/integrations/graphs/neo4j_cypher/)
- [Neo4j Vector Search Documentation](https://neo4j.com/docs/cypher-manual/current/indexes-for-vector-search/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)

### 추가 학습 자료

- **Neo4j Graph Data Science Library**: 고급 그래프 알고리즘 및 머신러닝
- **LangChain RAG Tutorial**: 검색 증강 생성 시스템 구축
- **Vector Search Best Practices**: 효율적인 벡터 검색 구현 가이드

### 관련 블로그 및 튜토리얼

- Neo4j Blog: "Building Knowledge Graphs with LangChain"
- LangChain Documentation: "Integrating Vector Stores"
- Medium: "Hybrid Search with Neo4j and LangChain"

---

## ✅ 학습 마무리

이 Part 2를 완료하면서 다음을 배웠습니다:

1. ✅ Neo4jVector를 활용한 의미론적 검색 구현
2. ✅ 다양한 검색 전략 (기본, MMR, 메타데이터 필터링) 활용
3. ✅ GraphCypherQAChain으로 자연어 질의응답 시스템 구축
4. ✅ 실무 RAG 시스템 설계 및 구현 방법 이해

**다음 단계**: 이제 Part 1과 Part 2의 지식을 결합하여 완전한 지식 그래프 기반 RAG 애플리케이션을 구축할 수 있습니다!

---

**Part 2 완료!** 🎉
