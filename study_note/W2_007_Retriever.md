# W2_007_Retriever.md - RAG 체인과 검색기 구현

## 🎯 학습 목표

- RAG 기반 질의응답 시스템의 구성 요소와 동작 원리를 이해합니다
- 다양한 검색 전략(Top-K, 임계값, MMR, 필터링)을 활용한 검색기를 구현합니다
- LangChain의 LCEL 문법을 사용한 RAG 체인을 구성합니다
- Gradio를 활용한 스트리밍 챗봇 인터페이스를 구현합니다
- 실무에서 활용할 수 있는 Naive RAG 시스템을 완성합니다

## 📚 핵심 개념

### 1. RAG(Retrieval-Augmented Generation) 체계

RAG는 정보 검색과 생성을 결합한 AI 시스템으로, 다음 세 단계로 구성됩니다:

```python
# RAG 파이프라인 구조
query → retrieval → augmentation → generation
```

**주요 구성 요소:**
- **Retriever**: 질의와 관련된 문서를 검색하는 구성 요소
- **Prompt Template**: 검색된 컨텍스트와 질의를 구조화
- **LLM**: 컨텍스트를 바탕으로 응답을 생성
- **Output Parser**: 응답을 원하는 형식으로 변환

### 2. 검색기(Retriever) 유형

#### 2.1 Top-K 검색
가장 유사한 상위 K개 문서를 반환하는 기본적인 검색 방식입니다.

```python
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}  # 상위 5개 문서 검색
)
```

#### 2.2 Similarity Score Threshold
유사도 점수가 임계값 이상인 문서만 검색하는 방식입니다.

```python
retriever = vectorstore.as_retriever(
    search_type='similarity_score_threshold',
    search_kwargs={'score_threshold': 0.7, 'k': 10}
)
```

#### 2.3 MMR(Maximal Marginal Relevance)
관련성과 다양성을 동시에 고려하여 검색하는 고급 기법입니다.

```python
retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={
        'k': 5,              # 최종 검색 문서 수
        'fetch_k': 20,       # 후보 문서 수
        'lambda_mult': 0.5   # 관련성 vs 다양성 균형 (0~1)
    }
)
```

**MMR 수식:**
```
MMR = argmax[D_i ∈ R\S] [λ * Sim₁(D_i, Q) - (1-λ) * max[D_j ∈ S] Sim₂(D_i, D_j)]
```

#### 2.4 메타데이터 필터링
문서의 메타데이터를 기반으로 검색 범위를 제한하는 방식입니다.

```python
retriever = vectorstore.as_retriever(
    search_kwargs={
        'filter': {'source': 'document.pdf', 'category': 'technical'},
        'k': 5
    }
)
```

### 3. LCEL(LangChain Expression Language)

LangChain의 파이프라인 구성 문법으로, 체인을 직관적으로 연결할 수 있습니다.

```python
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

## 🔧 환경 설정

### 1. 필수 라이브러리 설치

```bash
# uv 사용 (권장)
uv add langchain langchain-community langchain-openai langchain-chroma langchain-huggingface
uv add faiss-cpu gradio python-dotenv pypdf2

# pip 사용
pip install langchain langchain-community langchain-openai langchain-chroma langchain-huggingface
pip install faiss-cpu gradio python-dotenv pypdf2
```

### 2. 환경 변수 설정

```bash
# .env 파일에 추가
OPENAI_API_KEY=your_openai_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
```

### 3. 기본 임포트

```python
import os
from dotenv import load_dotenv
from pprint import pprint
import json
import uuid
from typing import Iterator, List, Dict, Any

# LangChain 관련
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 외부 라이브러리
import faiss
import gradio as gr

# 환경 변수 로드
load_dotenv()
```

## 💻 코드 예제

### 1. 문서 로딩 및 전처리

```python
class DocumentProcessor:
    def __init__(self, embedding_model: str = "BAAI/bge-m3"):
        """문서 처리 클래스 초기화"""
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.tokenizer = self.embeddings._client.tokenizer

    def count_tokens(self, text: str) -> int:
        """텍스트의 토큰 수 계산"""
        return len(self.tokenizer(text)['input_ids'])

    def load_and_split_pdf(
        self,
        pdf_path: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ) -> List[Dict[str, Any]]:
        """PDF 문서 로딩 및 청크 분할"""

        # PDF 로더 초기화
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # 텍스트 분할기 생성
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self.count_tokens,
            separators=["\n\n", "\n", " ", ""]
        )

        # 문서 분할
        chunks = splitter.split_documents(documents)

        print(f"원본 문서 수: {len(documents)}")
        print(f"생성된 청크 수: {len(chunks)}")
        print(f"평균 청크 토큰 수: {sum(self.count_tokens(chunk.page_content) for chunk in chunks) / len(chunks):.1f}")

        return chunks

# 사용 예시
processor = DocumentProcessor()
chunks = processor.load_and_split_pdf('./data/transformer.pdf')
```

### 2. 다양한 검색기 구현

```python
class AdvancedRetrieverManager:
    def __init__(self, chunks: List[Dict[str, Any]], embedding_model: str = "BAAI/bge-m3"):
        """고급 검색기 관리 클래스"""
        self.chunks = chunks
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstores = {}
        self.retrievers = {}

    def create_chroma_vectorstore(self, collection_name: str = "documents") -> None:
        """Chroma 벡터 스토어 생성"""
        self.vectorstores['chroma'] = Chroma.from_documents(
            documents=self.chunks,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory="./chroma_db",
            collection_metadata={'hnsw:space': 'cosine'}
        )

    def create_faiss_vectorstore(self) -> None:
        """FAISS 벡터 스토어 생성"""
        # 임베딩 차원 확인
        test_embedding = self.embeddings.embed_query("test")
        dim = len(test_embedding)

        # FAISS 인덱스 생성 (유클리드 거리)
        faiss_index = faiss.IndexFlatL2(dim)

        # FAISS 벡터 스토어 생성
        faiss_db = FAISS(
            embedding_function=self.embeddings,
            index=faiss_index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )

        # 문서 추가
        doc_ids = [str(uuid.uuid4()) for _ in range(len(self.chunks))]
        faiss_db.add_documents(self.chunks, ids=doc_ids)

        self.vectorstores['faiss'] = faiss_db

    def create_top_k_retriever(self, vectorstore_type: str = "chroma", k: int = 5) -> None:
        """Top-K 검색기 생성"""
        if vectorstore_type not in self.vectorstores:
            raise ValueError(f"벡터 스토어 '{vectorstore_type}'가 초기화되지 않았습니다.")

        self.retrievers[f'{vectorstore_type}_top_k'] = self.vectorstores[vectorstore_type].as_retriever(
            search_kwargs={"k": k}
        )

    def create_threshold_retriever(
        self,
        vectorstore_type: str = "chroma",
        threshold: float = 0.5,
        k: int = 10
    ) -> None:
        """임계값 기반 검색기 생성"""
        if vectorstore_type not in self.vectorstores:
            raise ValueError(f"벡터 스토어 '{vectorstore_type}'가 초기화되지 않았습니다.")

        self.retrievers[f'{vectorstore_type}_threshold'] = self.vectorstores[vectorstore_type].as_retriever(
            search_type='similarity_score_threshold',
            search_kwargs={'score_threshold': threshold, 'k': k}
        )

    def create_mmr_retriever(
        self,
        vectorstore_type: str = "chroma",
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5
    ) -> None:
        """MMR 검색기 생성"""
        if vectorstore_type not in self.vectorstores:
            raise ValueError(f"벡터 스토어 '{vectorstore_type}'가 초기화되지 않았습니다.")

        self.retrievers[f'{vectorstore_type}_mmr'] = self.vectorstores[vectorstore_type].as_retriever(
            search_type='mmr',
            search_kwargs={
                'k': k,
                'fetch_k': fetch_k,
                'lambda_mult': lambda_mult
            }
        )

    def create_metadata_retriever(
        self,
        vectorstore_type: str = "chroma",
        filter_dict: Dict[str, Any] = None,
        k: int = 5
    ) -> None:
        """메타데이터 필터링 검색기 생성"""
        if vectorstore_type not in self.vectorstores:
            raise ValueError(f"벡터 스토어 '{vectorstore_type}'가 초기화되지 않았습니다.")

        if filter_dict is None:
            filter_dict = {}

        self.retrievers[f'{vectorstore_type}_metadata'] = self.vectorstores[vectorstore_type].as_retriever(
            search_kwargs={'filter': filter_dict, 'k': k}
        )

    def test_retrievers(self, query: str) -> Dict[str, List[Dict[str, Any]]]:
        """모든 검색기 테스트"""
        results = {}

        for retriever_name, retriever in self.retrievers.items():
            try:
                docs = retriever.invoke(query)
                results[retriever_name] = [
                    {
                        'content': doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
                        'metadata': doc.metadata,
                        'full_content_length': len(doc.page_content)
                    }
                    for doc in docs
                ]
            except Exception as e:
                results[retriever_name] = f"오류: {str(e)}"

        return results

    def get_retriever(self, name: str):
        """특정 검색기 반환"""
        if name not in self.retrievers:
            raise ValueError(f"검색기 '{name}'를 찾을 수 없습니다. 사용 가능한 검색기: {list(self.retrievers.keys())}")
        return self.retrievers[name]

# 사용 예시
retriever_manager = AdvancedRetrieverManager(chunks)

# 벡터 스토어 생성
retriever_manager.create_chroma_vectorstore()
retriever_manager.create_faiss_vectorstore()

# 다양한 검색기 생성
retriever_manager.create_top_k_retriever("chroma", k=3)
retriever_manager.create_mmr_retriever("faiss", k=3, lambda_mult=0.3)
retriever_manager.create_threshold_retriever("chroma", threshold=0.6)

# 검색 테스트
test_query = "대표적인 시퀀스 모델은 어떤 것들이 있나요?"
results = retriever_manager.test_retrievers(test_query)

for retriever_name, result in results.items():
    print(f"\n=== {retriever_name} ===")
    if isinstance(result, str):
        print(result)
    else:
        for i, doc in enumerate(result, 1):
            print(f"{i}. {doc['content']}")
```

### 3. RAG 체인 구성

```python
class RAGChain:
    def __init__(self, retriever, model_name: str = "gpt-4.1-mini", temperature: float = 0.0):
        """RAG 체인 클래스"""
        self.retriever = retriever
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.prompt = self._create_prompt()
        self.chain = self._build_chain()

    def _create_prompt(self) -> ChatPromptTemplate:
        """프롬프트 템플릿 생성"""
        template = """다음 컨텍스트를 기반으로 질문에 답하세요. 컨텍스트에서 찾을 수 없는 내용은 추측하지 말고, 모르겠다고 답하세요.

[작업 지침]
- 주어진 컨텍스트의 정보만을 사용하여 답변하세요
- 외부 지식을 사용하지 마세요
- 불확실한 정보는 추측하지 말고 명확히 표시하세요
- 답변할 수 없다면 솔직히 말하세요

[컨텍스트]
{context}

[질문]
{question}

[답변]
위 컨텍스트를 바탕으로 답변드리겠습니다:

**핵심 답변:**

**근거:**

**추가 설명 (해당되는 경우):**

답변은 한국어로 작성하며, 사실에 기반하여 명확하게 제시하겠습니다."""

        return ChatPromptTemplate.from_template(template)

    def _format_docs(self, docs) -> str:
        """검색된 문서들을 텍스트로 포맷팅"""
        return "\n\n".join([f"문서 {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])

    def _build_chain(self):
        """LCEL을 사용한 체인 구성"""
        return (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def invoke(self, query: str) -> str:
        """동기 실행"""
        return self.chain.invoke(query)

    def stream(self, query: str):
        """스트리밍 실행"""
        return self.chain.stream(query)

    async def ainvoke(self, query: str) -> str:
        """비동기 실행"""
        return await self.chain.ainvoke(query)

    async def astream(self, query: str):
        """비동기 스트리밍 실행"""
        return self.chain.astream(query)

    def get_context_and_answer(self, query: str) -> Dict[str, Any]:
        """컨텍스트와 답변을 함께 반환"""
        # 검색된 문서들 가져오기
        retrieved_docs = self.retriever.invoke(query)

        # 체인 실행
        answer = self.invoke(query)

        return {
            'question': query,
            'context_docs': [
                {
                    'content': doc.page_content,
                    'metadata': doc.metadata
                }
                for doc in retrieved_docs
            ],
            'answer': answer,
            'num_context_docs': len(retrieved_docs)
        }

# 사용 예시
# 검색기 선택 (앞서 생성한 검색기 사용)
selected_retriever = retriever_manager.get_retriever("faiss_mmr")

# RAG 체인 생성
rag_chain = RAGChain(
    retriever=selected_retriever,
    model_name="gpt-4.1-mini",
    temperature=0.0
)

# 질의응답 테스트
test_queries = [
    "대표적인 시퀀스 모델은 어떤 것들이 있나요?",
    "Transformer의 주요 특징은 무엇인가요?",
    "어텐션 메커니즘이란 무엇인가요?"
]

for query in test_queries:
    print(f"\n질문: {query}")
    result = rag_chain.get_context_and_answer(query)
    print(f"답변: {result['answer']}")
    print(f"참조 문서 수: {result['num_context_docs']}")
```

### 4. Gradio 스트리밍 인터페이스

```python
class RAGChatInterface:
    def __init__(self, rag_chain: RAGChain):
        """RAG 채팅 인터페이스"""
        self.rag_chain = rag_chain

    def streaming_response(self, message: str, history) -> Iterator[str]:
        """스트리밍 응답 생성"""
        response = ""
        try:
            for chunk in self.rag_chain.stream(message):
                if isinstance(chunk, str):
                    response += chunk
                    yield response
        except Exception as e:
            yield f"오류가 발생했습니다: {str(e)}"

    def create_interface(self) -> gr.ChatInterface:
        """Gradio 인터페이스 생성"""
        interface = gr.ChatInterface(
            fn=self.streaming_response,
            title="🤖 RAG 기반 질의응답 시스템",
            description="PDF 문서를 기반으로 질문에 답변하는 AI 어시스턴트입니다.",
            examples=[
                "대표적인 시퀀스 모델은 어떤 것들이 있나요?",
                "Transformer의 주요 특징은 무엇인가요?",
                "어텐션 메커니즘이란 무엇인가요?",
                "인코더-디코더 구조에 대해 설명해주세요.",
                "RNN과 CNN의 한계점은 무엇인가요?"
            ],
            cache_examples=False,
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 800px !important;
                margin: auto !important;
            }
            .chat-message {
                font-size: 14px !important;
            }
            """
        )
        return interface

    def launch(self, **kwargs):
        """인터페이스 실행"""
        interface = self.create_interface()
        return interface.launch(**kwargs)

# 사용 예시
chat_interface = RAGChatInterface(rag_chain)

# 인터페이스 실행 (개발 환경)
if __name__ == "__main__":
    demo = chat_interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # 공개 링크 생성 여부
        debug=True    # 디버그 모드
    )
```

## 🚀 실습해보기

### 실습 1: 다중 검색 전략 비교 시스템

다양한 검색 전략의 성능을 비교하는 시스템을 구현해보세요.

```python
# 여러 검색기의 성능을 비교하는 시스템 구현
class RetrieverComparison:
    def __init__(self, chunks, test_queries):
        # 초기화 코드 작성
        pass

    def compare_retrievers(self, query):
        # 여러 검색기 결과 비교 코드 작성
        pass

    def evaluate_relevance(self, query, docs):
        # 검색 결과의 관련성 평가 코드 작성
        pass
```

### 실습 2: 적응형 검색기 구현

질문의 특성에 따라 최적의 검색 전략을 자동 선택하는 시스템을 구현해보세요.

```python
# 질문 분석 후 최적 검색기 선택 시스템
class AdaptiveRetriever:
    def __init__(self, retriever_manager):
        # 초기화 코드 작성
        pass

    def analyze_query_type(self, query):
        # 질문 유형 분석 코드 작성
        pass

    def select_best_retriever(self, query):
        # 최적 검색기 선택 코드 작성
        pass
```

### 실습 3: 멀티모달 RAG 시스템

텍스트와 이미지를 모두 처리할 수 있는 RAG 시스템을 구현해보세요.

```python
# 텍스트와 이미지를 함께 처리하는 RAG 시스템
class MultimodalRAG:
    def __init__(self):
        # 초기화 코드 작성 (텍스트, 이미지 임베딩 모델)
        pass

    def process_multimodal_documents(self, documents):
        # 멀티모달 문서 처리 코드 작성
        pass

    def hybrid_search(self, query, query_type="text"):
        # 하이브리드 검색 코드 작성
        pass
```

## 📋 해답

### 실습 1: 다중 검색 전략 비교 시스템

```python
class RetrieverComparison:
    def __init__(self, chunks: List[Dict[str, Any]], test_queries: List[str]):
        """검색기 비교 시스템 초기화"""
        self.chunks = chunks
        self.test_queries = test_queries
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

        # 여러 검색기 초기화
        self.retriever_manager = AdvancedRetrieverManager(chunks)
        self.retriever_manager.create_chroma_vectorstore()
        self.retriever_manager.create_faiss_vectorstore()

        # 다양한 검색기 생성
        self.retriever_manager.create_top_k_retriever("chroma", k=3)
        self.retriever_manager.create_mmr_retriever("chroma", k=3, lambda_mult=0.5)
        self.retriever_manager.create_mmr_retriever("faiss", k=3, lambda_mult=0.3)
        self.retriever_manager.create_threshold_retriever("chroma", threshold=0.6)

    def calculate_cosine_similarity(self, query: str, doc_content: str) -> float:
        """코사인 유사도 계산"""
        query_embedding = self.embeddings.embed_query(query)
        doc_embedding = self.embeddings.embed_query(doc_content)

        # 코사인 유사도 계산
        dot_product = sum(a * b for a, b in zip(query_embedding, doc_embedding))
        norm_a = sum(a * a for a in query_embedding) ** 0.5
        norm_b = sum(b * b for b in doc_embedding) ** 0.5

        return dot_product / (norm_a * norm_b) if norm_a * norm_b > 0 else 0.0

    def evaluate_relevance(self, query: str, docs: List[Any]) -> Dict[str, float]:
        """검색 결과의 관련성 평가"""
        if not docs:
            return {'avg_similarity': 0.0, 'max_similarity': 0.0, 'num_docs': 0}

        similarities = []
        for doc in docs:
            similarity = self.calculate_cosine_similarity(query, doc.page_content)
            similarities.append(similarity)

        return {
            'avg_similarity': sum(similarities) / len(similarities),
            'max_similarity': max(similarities),
            'min_similarity': min(similarities),
            'num_docs': len(docs),
            'similarity_std': (sum((s - sum(similarities)/len(similarities))**2 for s in similarities) / len(similarities))**0.5
        }

    def compare_retrievers(self, query: str) -> Dict[str, Dict[str, Any]]:
        """여러 검색기 결과 비교"""
        results = {}

        for retriever_name, retriever in self.retriever_manager.retrievers.items():
            try:
                # 검색 실행
                docs = retriever.invoke(query)

                # 관련성 평가
                relevance_metrics = self.evaluate_relevance(query, docs)

                # 결과 저장
                results[retriever_name] = {
                    'retrieved_docs': [
                        {
                            'content': doc.page_content[:150] + "...",
                            'metadata': doc.metadata,
                            'similarity': self.calculate_cosine_similarity(query, doc.page_content)
                        }
                        for doc in docs
                    ],
                    'metrics': relevance_metrics,
                    'execution_time': None  # 실제 구현에서는 시간 측정 추가
                }

            except Exception as e:
                results[retriever_name] = {
                    'error': str(e),
                    'retrieved_docs': [],
                    'metrics': {'avg_similarity': 0.0}
                }

        return results

    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """종합 평가 실행"""
        overall_results = {}

        for query in self.test_queries:
            print(f"\n평가 중인 질문: {query}")
            query_results = self.compare_retrievers(query)
            overall_results[query] = query_results

        # 전체 성능 요약
        summary = self._generate_summary(overall_results)

        return {
            'detailed_results': overall_results,
            'summary': summary
        }

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, float]:
        """전체 결과 요약 생성"""
        retriever_scores = {}

        for query, query_results in results.items():
            for retriever_name, retriever_result in query_results.items():
                if 'metrics' in retriever_result:
                    if retriever_name not in retriever_scores:
                        retriever_scores[retriever_name] = []
                    retriever_scores[retriever_name].append(
                        retriever_result['metrics']['avg_similarity']
                    )

        # 평균 성능 계산
        summary = {}
        for retriever_name, scores in retriever_scores.items():
            summary[retriever_name] = {
                'avg_performance': sum(scores) / len(scores) if scores else 0,
                'query_count': len(scores)
            }

        return summary

# 실습 1 테스트
test_queries = [
    "대표적인 시퀀스 모델은 어떤 것들이 있나요?",
    "Transformer의 주요 특징은 무엇인가요?",
    "어텐션 메커니즘이란 무엇인가요?"
]

comparison_system = RetrieverComparison(chunks, test_queries)
evaluation_results = comparison_system.run_comprehensive_evaluation()

# 결과 출력
print("\n=== 검색기 성능 비교 결과 ===")
for retriever_name, performance in evaluation_results['summary'].items():
    print(f"{retriever_name}: {performance['avg_performance']:.3f}")
```

### 실습 2: 적응형 검색기 구현

```python
class AdaptiveRetriever:
    def __init__(self, retriever_manager: AdvancedRetrieverManager):
        """적응형 검색기 초기화"""
        self.retriever_manager = retriever_manager
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

        # 질문 유형별 최적 검색기 매핑
        self.strategy_mapping = {
            'factual': 'chroma_top_k',      # 사실적 질문
            'comparative': 'faiss_mmr',     # 비교 질문
            'conceptual': 'chroma_mmr',     # 개념적 질문
            'specific': 'chroma_threshold', # 특정 정보 질문
            'broad': 'faiss_mmr'           # 광범위한 질문
        }

        # 질문 유형 분류를 위한 키워드
        self.type_keywords = {
            'factual': ['무엇인', '누구', '언제', '어디서', '몇', '얼마나'],
            'comparative': ['차이', '비교', '대비', 'vs', '다른', '같은', '유사한'],
            'conceptual': ['개념', '원리', '이론', '방식', '방법', '과정'],
            'specific': ['구체적', '자세히', '정확히', '예시', '사례'],
            'broad': ['전반적', '일반적', '대략', '대체로', '종합적']
        }

    def analyze_query_type(self, query: str) -> str:
        """질문 유형 분석"""
        query_lower = query.lower()

        # 키워드 기반 분류
        type_scores = {}
        for query_type, keywords in self.type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                type_scores[query_type] = score

        # 가장 높은 점수의 유형 반환
        if type_scores:
            return max(type_scores, key=type_scores.get)

        # 기본값: 개념적 질문으로 분류
        return 'conceptual'

    def analyze_query_complexity(self, query: str) -> str:
        """질문 복잡도 분석"""
        # 질문 길이 기반 복잡도 추정
        if len(query) > 50:
            return 'complex'
        elif len(query) > 20:
            return 'medium'
        else:
            return 'simple'

    def select_best_retriever(self, query: str) -> tuple:
        """최적 검색기 선택"""
        query_type = self.analyze_query_type(query)
        complexity = self.analyze_query_complexity(query)

        # 기본 검색기 선택
        base_retriever_name = self.strategy_mapping.get(query_type, 'chroma_mmr')

        # 복잡도에 따른 조정
        if complexity == 'complex' and 'mmr' not in base_retriever_name:
            # 복잡한 질문은 MMR 사용
            base_retriever_name = base_retriever_name.replace('top_k', 'mmr')

        # 검색기 반환
        try:
            retriever = self.retriever_manager.get_retriever(base_retriever_name)
            return retriever, base_retriever_name, query_type
        except ValueError:
            # 기본 검색기로 폴백
            fallback_name = 'chroma_top_k'
            retriever = self.retriever_manager.get_retriever(fallback_name)
            return retriever, fallback_name, query_type

    def adaptive_search(self, query: str) -> Dict[str, Any]:
        """적응형 검색 실행"""
        # 최적 검색기 선택
        retriever, retriever_name, query_type = self.select_best_retriever(query)

        # 검색 실행
        docs = retriever.invoke(query)

        # 결과 반환
        return {
            'query': query,
            'selected_retriever': retriever_name,
            'query_type': query_type,
            'retrieved_docs': [
                {
                    'content': doc.page_content,
                    'metadata': doc.metadata
                }
                for doc in docs
            ],
            'num_docs': len(docs)
        }

    def batch_adaptive_search(self, queries: List[str]) -> List[Dict[str, Any]]:
        """여러 질문에 대한 적응형 검색"""
        results = []

        for query in queries:
            result = self.adaptive_search(query)
            results.append(result)

            print(f"질문: {query}")
            print(f"선택된 검색기: {result['selected_retriever']}")
            print(f"질문 유형: {result['query_type']}")
            print(f"검색된 문서 수: {result['num_docs']}")
            print("-" * 50)

        return results

# 실습 2 테스트
adaptive_retriever = AdaptiveRetriever(retriever_manager)

test_queries = [
    "RNN과 CNN의 차이점은 무엇인가요?",  # comparative
    "어텐션 메커니즘이란?",              # conceptual
    "Transformer에서 사용되는 구체적인 수식을 알려주세요",  # specific
    "딥러닝에서 시퀀스 처리 방법들을 전반적으로 설명해주세요"  # broad
]

adaptive_results = adaptive_retriever.batch_adaptive_search(test_queries)
```

### 실습 3: 멀티모달 RAG 시스템

```python
class MultimodalRAG:
    def __init__(self):
        """멀티모달 RAG 시스템 초기화"""
        # 텍스트 임베딩 모델
        self.text_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

        # 실제 구현에서는 CLIP 같은 멀티모달 모델 사용
        # 여기서는 텍스트 기반 시뮬레이션

        self.text_vectorstore = None
        self.image_metadata_store = []
        self.multimodal_index = {}

    def extract_image_descriptions(self, image_path: str) -> str:
        """이미지에서 텍스트 설명 추출 (시뮬레이션)"""
        # 실제 구현에서는 OCR, 이미지 캡셔닝 모델 사용
        import os
        filename = os.path.basename(image_path)

        # 파일명 기반 설명 생성 (실제로는 AI 모델 사용)
        descriptions = {
            'attention_diagram.png': 'Attention mechanism diagram showing query, key, value matrices',
            'transformer_architecture.png': 'Complete Transformer architecture with encoder and decoder',
            'positional_encoding.png': 'Positional encoding visualization with sinusoidal patterns'
        }

        return descriptions.get(filename, f"Image content from {filename}")

    def process_multimodal_documents(self, text_documents: List[Any], image_paths: List[str]) -> None:
        """멀티모달 문서 처리"""
        all_documents = []

        # 텍스트 문서 처리
        for doc in text_documents:
            doc.metadata['content_type'] = 'text'
            all_documents.append(doc)

        # 이미지 문서 처리
        for img_path in image_paths:
            description = self.extract_image_descriptions(img_path)

            # 이미지를 텍스트 기반 문서로 변환
            from langchain.schema import Document
            img_document = Document(
                page_content=f"이미지 설명: {description}",
                metadata={
                    'source': img_path,
                    'content_type': 'image',
                    'image_path': img_path
                }
            )
            all_documents.append(img_document)

            # 이미지 메타데이터 저장
            self.image_metadata_store.append({
                'path': img_path,
                'description': description,
                'doc_id': len(all_documents) - 1
            })

        # 통합 벡터 스토어 생성
        self.text_vectorstore = FAISS.from_documents(
            all_documents,
            self.text_embeddings
        )

        print(f"처리된 문서 수: {len(all_documents)}")
        print(f"텍스트 문서: {len(text_documents)}")
        print(f"이미지 문서: {len(image_paths)}")

    def hybrid_search(self, query: str, query_type: str = "text", k: int = 5) -> Dict[str, Any]:
        """하이브리드 검색 실행"""
        if self.text_vectorstore is None:
            raise ValueError("먼저 process_multimodal_documents()를 호출하여 문서를 처리하세요.")

        # 기본 유사도 검색
        retriever = self.text_vectorstore.as_retriever(
            search_type='mmr',
            search_kwargs={'k': k, 'lambda_mult': 0.5}
        )

        docs = retriever.invoke(query)

        # 결과를 텍스트와 이미지로 분류
        text_results = []
        image_results = []

        for doc in docs:
            if doc.metadata.get('content_type') == 'text':
                text_results.append(doc)
            elif doc.metadata.get('content_type') == 'image':
                image_results.append(doc)

        return {
            'query': query,
            'query_type': query_type,
            'text_results': [
                {
                    'content': doc.page_content[:200] + "...",
                    'metadata': doc.metadata
                }
                for doc in text_results
            ],
            'image_results': [
                {
                    'description': doc.page_content,
                    'image_path': doc.metadata.get('image_path'),
                    'metadata': doc.metadata
                }
                for doc in image_results
            ],
            'total_results': len(docs)
        }

    def create_multimodal_rag_chain(self, model_name: str = "gpt-4.1-mini"):
        """멀티모달 RAG 체인 생성"""
        from langchain.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough

        # 멀티모달 프롬프트 템플릿
        template = """다음 텍스트 컨텍스트와 이미지 설명을 기반으로 질문에 답하세요.

[텍스트 컨텍스트]
{text_context}

[이미지 설명]
{image_context}

[질문]
{question}

[답변]
텍스트와 이미지 정보를 종합하여 답변드리겠습니다:

**핵심 답변:**

**텍스트 기반 근거:**

**이미지 기반 근거:**

**종합 설명:**
"""

        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatOpenAI(model=model_name, temperature=0)

        def format_multimodal_context(query: str) -> Dict[str, str]:
            """멀티모달 컨텍스트 포맷팅"""
            search_results = self.hybrid_search(query)

            text_context = "\n\n".join([
                result['content'] for result in search_results['text_results']
            ])

            image_context = "\n\n".join([
                f"이미지: {result['description']}"
                for result in search_results['image_results']
            ])

            return {
                'text_context': text_context or "관련 텍스트를 찾을 수 없습니다.",
                'image_context': image_context or "관련 이미지를 찾을 수 없습니다.",
                'question': query
            }

        # LCEL 체인 구성
        multimodal_chain = (
            RunnablePassthrough()
            | (lambda x: format_multimodal_context(x))
            | prompt
            | llm
            | StrOutputParser()
        )

        return multimodal_chain

# 실습 3 테스트
multimodal_rag = MultimodalRAG()

# 시뮬레이션을 위한 이미지 경로
image_paths = [
    './images/attention_diagram.png',
    './images/transformer_architecture.png',
    './images/positional_encoding.png'
]

# 멀티모달 문서 처리
multimodal_rag.process_multimodal_documents(chunks, image_paths)

# 하이브리드 검색 테스트
hybrid_results = multimodal_rag.hybrid_search("어텐션 메커니즘의 구조를 시각적으로 보여주세요")

print("=== 하이브리드 검색 결과 ===")
print(f"텍스트 결과: {len(hybrid_results['text_results'])}개")
print(f"이미지 결과: {len(hybrid_results['image_results'])}개")

for img_result in hybrid_results['image_results']:
    print(f"이미지: {img_result['image_path']}")
    print(f"설명: {img_result['description']}")

# 멀티모달 RAG 체인 테스트
multimodal_chain = multimodal_rag.create_multimodal_rag_chain()
multimodal_answer = multimodal_chain.invoke("어텐션 메커니즘을 시각적 자료와 함께 설명해주세요")

print("\n=== 멀티모달 RAG 답변 ===")
print(multimodal_answer)
```

## 🔍 참고 자료

### 공식 문서
- [LangChain Retrievers](https://python.langchain.com/docs/modules/data_connection/retrievers/)
- [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/)
- [Chroma Vector Database](https://docs.trychroma.com/)
- [FAISS Documentation](https://faiss.ai/cpp_api/)
- [Gradio Documentation](https://gradio.app/docs/)

### 학술 자료
- Karpukhin, V., et al. (2020). "Dense Passage Retrieval for Open-Domain Question Answering"
- Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- Zhu, F., et al. (2021). "Retrieving and Reading: A Comprehensive Survey on Open-domain Question Answering"

### 실무 가이드
- [RAG System Design Patterns](https://docs.google.com/document/d/1example)
- [Vector Database Performance Comparison](https://example.com/vector-db-comparison)
- [Production RAG Deployment Guide](https://example.com/rag-deployment)

### 추가 학습 자료
- [Hugging Face Embeddings Models](https://huggingface.co/models?pipeline_tag=feature-extraction)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Advanced RAG Techniques](https://example.com/advanced-rag)

---

**다음 학습**: W3_001_Prompt_Engineering_Basic.md - 프롬프트 엔지니어링 기초와 효과적인 프롬프트 작성법을 학습합니다.