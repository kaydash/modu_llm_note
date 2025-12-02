# PRJ04_W1_006 - Unstructured & LangChain 통합 Part2: 법률 문서 RAG 실습

## 📚 학습 목표

이 가이드를 완료하면 다음을 할 수 있습니다:

1. **다중 파일 형식 RAG 구축**: PDF, DOCX, TXT 등 다양한 형식을 통합 처리하는 RAG 시스템
2. **Chroma DB 활용**: 벡터 데이터베이스를 사용한 효율적인 문서 검색
3. **LCEL 기반 체인 구성**: LangChain Expression Language로 최적화된 RAG 파이프라인 구축
4. **법률 특화 프롬프트**: 전문 도메인에 맞는 맞춤형 프롬프트 엔지니어링
5. **고급 기능 구현**: 법적 근거 하이라이팅, 용어 자동 설명 등 실무 기능 추가
6. **실전 배포**: 프로덕션 환경을 고려한 RAG 시스템 설계

---

## 🎯 핵심 개념

### 1. RAG (Retrieval-Augmented Generation)

**RAG**는 외부 지식을 검색하여 LLM의 응답을 보강하는 기법입니다.

**기본 흐름:**
```
사용자 질문 → 관련 문서 검색 → 검색된 문서 + 질문 → LLM → 답변
```

**장점:**
- 최신 정보 반영 (모델 재학습 불필요)
- 출처 명시 가능 (신뢰성 향상)
- 환각(Hallucination) 감소
- 도메인 특화 가능

### 2. Vector Database (벡터 데이터베이스)

문서를 **고차원 벡터로 변환하여 저장**하고 유사도 검색을 수행합니다.

**Chroma DB:**
- 경량 벡터 DB
- 로컬 및 영구 저장 지원
- LangChain 네이티브 통합
- 무료 오픈소스

**작동 원리:**
```
1. 문서를 청크로 분할
2. 각 청크를 임베딩 벡터로 변환
3. Chroma DB에 저장
4. 쿼리를 벡터로 변환
5. 코사인 유사도로 검색
6. 가장 유사한 청크 반환
```

### 3. LCEL (LangChain Expression Language)

파이프라인을 **선언적으로 구성**하는 LangChain의 표현 언어입니다.

**기본 구조:**
```python
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | output_parser
)
```

**특징:**
- 파이프(`|`) 연산자로 연결
- 자동 병렬 처리
- 스트리밍 지원
- 에러 처리 내장

### 4. Retrieval 전략

**유사도 검색 (Similarity Search):**
- 가장 기본적인 방법
- 코사인 유사도 기반
- Top-K 문서 반환

**최대 한계 관련성 (MMR):**
- 다양성과 관련성 균형
- 중복 문서 제거

**매개변수:**
- `k`: 반환할 문서 수 (기본 4)
- `score_threshold`: 최소 유사도 점수
- `fetch_k`: 초기 검색 문서 수 (MMR용)

---

## 💻 실습: 법률 문서 RAG 시스템 구축

### 단계 1: 기본 RAG 시스템 설계

#### 1.1 클래스 구조 설계

```python
from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_unstructured import UnstructuredLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import os
from typing import List, Optional, Dict
import logging

class LegalDocumentRAG:
    """
    법률 문서 RAG 시스템
    - 다중 파일 형식 지원 (PDF, DOCX, TXT)
    - Chroma DB 벡터 검색
    - LCEL 기반 RAG 체인
    """

    def __init__(self, collection_name: str, persist_directory: str = "./chroma_db"):
        """
        RAG 시스템 초기화

        Args:
            collection_name: Chroma 컬렉션 이름
            persist_directory: 벡터 DB 저장 경로
        """
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # 기본 설정
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # 임베딩 모델 설정
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # 텍스트 분할기 설정
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,              # 청크 크기
            chunk_overlap=200,            # 중복 영역
            length_function=len,
            separators=["\\n\\n", "\\n", ".", " ", ""]  # 분할 우선순위
        )

        # 벡터 DB (나중에 초기화)
        self.vectorstore = None
```

#### 1.2 문서 로더 구현

```python
    def _load_document(self, file_path: str) -> List[Document]:
        """
        파일 형식에 따라 적절한 로더 선택

        Args:
            file_path: 문서 파일 경로

        Returns:
            Document 객체 리스트
        """
        self.logger.info(f"Loading document: {file_path}")

        try:
            # 파일 확장자에 따라 로더 선택
            if file_path.endswith('.pdf'):
                loader = UnstructuredPDFLoader(file_path)
            elif file_path.endswith('.docx'):
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file_path.endswith('.txt'):
                loader = UnstructuredLoader(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")

            # 문서 로드
            documents = loader.load()

            # 메타데이터 정제 및 파일명 추가
            for doc in documents:
                # Chroma는 str, int, float, bool만 지원
                doc.metadata = {
                    k: v for k, v in doc.metadata.items()
                    if isinstance(v, (str, int, float, bool))
                }
                # 파일명 추가
                doc.metadata["source"] = os.path.basename(file_path)

            return documents

        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {str(e)}")
            raise
```

#### 1.3 디렉토리 전체 문서 로드

```python
    def load_documents(self, directory: str):
        """
        디렉토리의 모든 문서를 로드하고 벡터 DB에 저장

        Args:
            directory: 문서가 있는 디렉토리 경로

        Returns:
            int: 생성된 청크 수
        """
        all_documents = []

        # 지원하는 파일 확장자
        supported_extensions = ('.pdf', '.docx', '.txt')

        # 디렉토리 내 모든 파일 처리
        for file in os.listdir(directory):
            if file.endswith(supported_extensions):
                file_path = os.path.join(directory, file)
                documents = self._load_document(file_path)
                all_documents.extend(documents)

        self.logger.info(f"Loaded {len(all_documents)} documents")

        # 문서 분할
        splits = self.text_splitter.split_documents(all_documents)
        self.logger.info(f"Created {len(splits)} document chunks")

        # Chroma DB에 저장
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory
        )

        return len(splits)
```

---

### 단계 2: LCEL 기반 RAG 체인 구성

#### 2.1 프롬프트 템플릿 설계

```python
    def setup_rag_chain(
        self,
        llm: Optional[ChatOpenAI] = None,
        retriever_kwargs: Optional[Dict] = None
    ):
        """
        LCEL을 사용한 RAG 체인 설정

        Args:
            llm: 사용할 LLM 모델
            retriever_kwargs: 검색기 설정

        Returns:
            RAG 체인
        """

        # 법률 전문가 프롬프트 템플릿
        template = """당신은 한국 법률 전문가입니다.
주어진 컨텍스트를 기반으로 질문에 대해 정확하고 객관적으로 답변해주세요.
관련 법 조항이나 근거가 있다면 함께 인용해주세요.

컨텍스트: {context}

질문: {question}

답변:"""

        prompt = ChatPromptTemplate.from_template(template)

        # LLM 설정 (기본값)
        if llm is None:
            llm = ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0  # 일관된 답변을 위해 0
            )

        # 검색기 설정
        if retriever_kwargs is None:
            retriever_kwargs = {"search_kwargs": {"k": 4}}

        retriever = self.vectorstore.as_retriever(**retriever_kwargs)

        # 문서 포맷팅 함수
        def format_documents(documents: List[Document]) -> str:
            """Document 리스트를 하나의 문자열로 변환"""
            return "\\n\\n".join([doc.page_content for doc in documents])

        # RAG 체인 구성 (LCEL)
        chain = (
            {
                "context": retriever | format_documents,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        return chain
```

#### 2.2 RAG 체인 실행

```python
# RAG 시스템 초기화
rag = LegalDocumentRAG(
    collection_name="legal_docs_collection",
    persist_directory="./chroma_db"
)

# 문서 로드 및 인덱싱
docs_path = "./data/legal_documents"
num_chunks = rag.load_documents(docs_path)
print(f"\n✅ 총 {num_chunks}개의 문서 청크가 처리되었습니다.")

# RAG 체인 설정
chain = rag.setup_rag_chain(
    llm=ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0
    ),
    retriever_kwargs={
        "search_type": "similarity",
        "search_kwargs": {"k": 4}
    }
)

# 질문 및 답변
question = "개인정보보호법에 따르면 개인정보의 수집에 대한 동의는 어떻게 해야 하나요?"
answer = chain.invoke(question)

print(f"\n📝 질문: {question}")
print(f"💬 답변: {answer}")
```

**출력 예시:**
```
✅ 총 45개의 문서 청크가 처리되었습니다.

📝 질문: 개인정보보호법에 따르면 개인정보의 수집에 대한 동의는 어떻게 해야 하나요?
💬 답변: 개인정보보호법 제22조에 따르면, 개인정보처리자는 다음 각 호의 어느 하나에
해당하는 경우에는 정보주체의 동의를 받아야 합니다:
1. 개인정보의 수집·이용 목적
2. 수집하려는 개인정보의 항목
3. 개인정보의 보유 및 이용 기간
4. 동의를 거부할 권리가 있다는 사실 및 동의 거부에 따른 불이익이 있는 경우에는 그 불이익의 내용

동의는 서면, 전자우편, 팩스 등의 방법으로 받을 수 있으며, 명확하고 구체적으로 이루어져야 합니다.
```

---

### 단계 3: 고급 기능 추가

#### 3.1 법적 근거 하이라이팅

```python
from langchain_core.runnables import RunnableParallel

class LegalDocumentRAG:
    # ... (기존 코드 생략)

    def setup_rag_chain_advanced(
        self,
        llm: Optional[ChatOpenAI] = None,
        retriever_kwargs: Optional[Dict] = None
    ):
        """
        고급 기능이 포함된 RAG 체인
        - 법적 근거 강조 (볼드체)
        - 법률 용어 설명
        """

        # 1. 답변 생성 프롬프트 (법적 근거 강조)
        answer_template = """당신은 한국 법률 전문가입니다.
주어진 컨텍스트를 기반으로 질문에 대해 정확하고 객관적으로 답변해주세요.
**답변의 근거가 되는 법 조항 또는 판례 등을 명시하고, 해당 부분을 `**볼드체**`로 강조**하여 사용자가 명확히 알 수 있도록 해주세요.
만약 컨텍스트에서 답을 찾을 수 없다면, '제공된 정보 내에서는 답변을 찾을 수 없습니다.' 라고 답변해주세요.

컨텍스트: {context}

질문: {question}

답변:"""
        answer_prompt = ChatPromptTemplate.from_template(answer_template)

        # 2. 법률 용어 설명 프롬프트
        term_explanation_template = """주어진 텍스트에서 법률 용어를 찾아서, 각 용어에 대한 간결하고 쉬운 설명을 제공해주세요.
설명은 일반인이 이해하기 쉬운 한국어로 작성하며, 필요한 경우 비유 또는 예시를 들어주세요.
만약 법률 용어가 없다면, \"법률 용어 설명이 필요하지 않습니다.\" 라고 답변해주세요.

텍스트: {legal_answer}

법률 용어 설명:"""
        term_explanation_prompt = ChatPromptTemplate.from_template(term_explanation_template)

        # LLM 설정
        if llm is None:
            llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

        # 검색기 설정
        if retriever_kwargs is None:
            retriever_kwargs = {"search_kwargs": {"k": 4}}
        retriever = self.vectorstore.as_retriever(**retriever_kwargs)

        # 문서 포맷팅
        def format_documents(documents: List[Document]) -> str:
            return "\\n\\n".join([doc.page_content for doc in documents])

        # 답변 생성 체인
        answer_chain = (
            {
                "context": retriever | format_documents,
                "question": RunnablePassthrough()
            }
            | answer_prompt
            | llm
            | StrOutputParser()
        )

        # 법률 용어 설명 체인
        term_explanation_chain = (
            {"legal_answer": RunnablePassthrough()}
            | term_explanation_prompt
            | llm
            | StrOutputParser()
        )

        # 병렬 실행 (답변 생성 + 용어 설명)
        rag_chain = RunnableParallel(
            legal_answer=answer_chain,
            term_explanation=term_explanation_chain
        )

        self.logger.info("고급 RAG chain 설정 완료")
        return rag_chain
```

#### 3.2 질의응답 메서드

```python
from typing import Tuple

def ask_question(
    self,
    question: str,
    rag_chain,
    explain_terms: bool = True
) -> Tuple[str, Optional[str]]:
    """
    질문에 답변하고, 필요에 따라 법률 용어 설명도 제공

    Args:
        question: 사용자 질문
        rag_chain: RAG 체인
        explain_terms: 법률 용어 설명 여부

    Returns:
        (답변, 용어 설명) 튜플
    """
    self.logger.info(f"질문: {question}")

    if not rag_chain:
        rag_chain = self.setup_rag_chain_advanced()

    try:
        # RAG 체인 실행 (답변 및 용어 설명 동시 생성)
        response = rag_chain.invoke(question)
        legal_answer = response.get("legal_answer", "답변 생성 실패")
        term_explanation = response.get("term_explanation", None)

        self.logger.info(f"답변: {legal_answer}")
        if explain_terms:
            self.logger.info(f"법률 용어 설명: {term_explanation}")

        if explain_terms:
            return legal_answer, term_explanation
        else:
            return legal_answer, None

    except Exception as e:
        self.logger.error(f"RAG 체인 실행 중 오류 발생: {e}")
        return "답변 생성 중 오류가 발생했습니다.", None
```

#### 3.3 고급 RAG 시스템 실행

```python
# RAG 시스템 초기화
legal_rag = LegalDocumentRAG(
    collection_name="legal_documents",
    persist_directory="./chroma_db"
)

# 문서 로드 (최초 실행 시)
DOCUMENTS_DIR = "./data/legal_documents"
if not os.path.exists(os.path.join("./chroma_db", "legal_documents")):
    document_count = legal_rag.load_documents(DOCUMENTS_DIR)
    print(f"{document_count}개의 문서가 로드 및 Chroma DB에 저장되었습니다.")
else:
    print("기존 Chroma DB를 사용합니다.")

# 고급 RAG 체인 설정
rag_chain = legal_rag.setup_rag_chain_advanced(
    llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
    retriever_kwargs={
        "search_type": "similarity",
        "search_kwargs": {"k": 4}
    }
)

# 질문 및 답변 (법률 용어 설명 활성화)
question1 = "상가건물 임대차보호법의 적용 범위는 어떻게 되나요?"
answer1, term_explanation1 = legal_rag.ask_question(question1, rag_chain, explain_terms=True)

print(f"\n📝 질문: {question1}")
print(f"💬 답변: {answer1}")
if term_explanation1:
    print(f"📚 법률 용어 설명: {term_explanation1}")
```

**출력 예시:**
```
📝 질문: 상가건물 임대차보호법의 적용 범위는 어떻게 되나요?
💬 답변: **상가건물 임대차보호법 제2조**에 따르면, 이 법은 **상가건물의 임대차**에
적용됩니다. 구체적으로:
1. **보증금액이 일정 금액 이하**인 경우 (대통령령으로 정함)
2. **영업용 건축물**의 임대차
3. **일부 주거용 건축물의 일부분**을 영업용으로 사용하는 경우

단, 다음의 경우는 제외됩니다:
- 임대차 목적물의 주된 부분을 영업용으로 사용하지 않는 경우
- 일시 사용을 위한 임대차

📚 법률 용어 설명:
- **상가건물**: 영업을 목적으로 사용하는 건축물로, 상점, 사무실, 음식점 등이 포함됩니다.
- **임대차**: 임대인이 임차인에게 일정한 물건을 사용·수익하게 하고, 임차인이 이에 대하여 차임을 지급하는 계약입니다.
- **보증금**: 임대차 계약 시 임차인이 임대인에게 지급하는 금액으로, 계약 종료 시 반환받는 돈입니다.
```

---

## 🎯 실습 문제

### 문제 1: 검색 전략 최적화

다양한 검색 전략(similarity, MMR)의 성능을 비교 분석하세요.

**요구사항:**
- similarity 검색과 MMR 검색 구현
- 같은 질문에 대한 검색 결과 비교
- 검색된 문서의 다양성 측정

<details>
<summary>솔루션 보기</summary>

```python
from langchain_community.vectorstores import Chroma
from sentence_transformers.util import cos_sim
import numpy as np

def compare_retrieval_strategies(vectorstore, question, k=4):
    """
    검색 전략 비교

    Args:
        vectorstore: Chroma 벡터스토어
        question: 검색 쿼리
        k: 검색할 문서 수
    """

    print(f"📝 질문: {question}\n")
    print("="*80)

    # 1. Similarity Search
    print("\n🔍 Strategy 1: Similarity Search")
    similarity_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    similarity_docs = similarity_retriever.get_relevant_documents(question)

    print(f"검색된 문서 수: {len(similarity_docs)}")
    for i, doc in enumerate(similarity_docs, 1):
        print(f"\n문서 {i} (from {doc.metadata.get('source', 'unknown')}):")
        print(doc.page_content[:200])

    # 2. MMR (Maximum Marginal Relevance)
    print("\n" + "="*80)
    print("\n🔍 Strategy 2: MMR (다양성 중시)")
    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 20,  # 초기 검색 문서 수
            "lambda_mult": 0.5  # 관련성(1.0) vs 다양성(0.0)
        }
    )
    mmr_docs = mmr_retriever.get_relevant_documents(question)

    print(f"검색된 문서 수: {len(mmr_docs)}")
    for i, doc in enumerate(mmr_docs, 1):
        print(f"\n문서 {i} (from {doc.metadata.get('source', 'unknown')}):")
        print(doc.page_content[:200])

    # 3. 다양성 측정 (임베딩 기반)
    print("\n" + "="*80)
    print("\n📊 다양성 분석:")

    # 임베딩 가져오기
    similarity_texts = [doc.page_content for doc in similarity_docs]
    mmr_texts = [doc.page_content for doc in mmr_docs]

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    similarity_embs = model.encode(similarity_texts, convert_to_tensor=True)
    mmr_embs = model.encode(mmr_texts, convert_to_tensor=True)

    # 평균 유사도 계산 (다양성의 역수)
    def calculate_avg_similarity(embeddings):
        sims = cos_sim(embeddings, embeddings)
        # 대각선 제거 (자기 자신과의 유사도)
        np.fill_diagonal(sims.cpu().numpy(), 0)
        return sims.mean().item()

    similarity_avg_sim = calculate_avg_similarity(similarity_embs)
    mmr_avg_sim = calculate_avg_similarity(mmr_embs)

    print(f"Similarity Search 평균 유사도: {similarity_avg_sim:.3f}")
    print(f"MMR 평균 유사도: {mmr_avg_sim:.3f}")
    print(f"MMR 다양성 개선: {(similarity_avg_sim - mmr_avg_sim)/similarity_avg_sim*100:.1f}%")

# 테스트
compare_retrieval_strategies(
    legal_rag.vectorstore,
    "임대차 계약 갱신 요구권은 언제 행사할 수 있나요?"
)
```

</details>

---

### 문제 2: 출처 추적 기능 추가

답변에 사용된 문서의 출처를 명시하는 기능을 구현하세요.

**요구사항:**
- 답변 생성에 사용된 문서 추적
- 문서명과 관련 텍스트 표시
- 중복 제거

<details>
<summary>솔루션 보기</summary>

```python
from typing import List, Dict

def create_rag_with_sources():
    """출처 추적 기능이 있는 RAG 체인"""

    # 프롬프트 템플릿 (출처 포함)
    template = """당신은 한국 법률 전문가입니다.
주어진 컨텍스트를 기반으로 질문에 대해 정확하고 객관적으로 답변해주세요.
답변 끝에 참조한 법률 문서를 명시해주세요.

컨텍스트: {context}

질문: {question}

답변 (마지막에 [출처: 문서명] 형식으로 출처 표기):"""

    prompt = ChatPromptTemplate.from_template(template)

    # 검색기
    retriever = legal_rag.vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    # LLM
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # 문서와 출처를 함께 포맷팅
    def format_documents_with_sources(documents: List[Document]) -> Dict[str, str]:
        """문서를 포맷팅하고 출처 정보도 함께 반환"""
        # 문서 텍스트 결합
        context = "\\n\\n".join([
            f"[문서 {i+1}: {doc.metadata.get('source', 'unknown')}]\\n{doc.page_content}"
            for i, doc in enumerate(documents)
        ])

        # 출처 목록 (중복 제거)
        sources = list(set([
            doc.metadata.get('source', 'unknown')
            for doc in documents
        ]))

        return {
            "context": context,
            "sources": sources,
            "documents": documents
        }

    # RAG 체인
    chain = (
        {
            "docs": retriever,
            "question": RunnablePassthrough()
        }
        | RunnablePassthrough.assign(
            formatted=lambda x: format_documents_with_sources(x["docs"])
        )
        | {
            "answer": (
                {
                    "context": lambda x: x["formatted"]["context"],
                    "question": lambda x: x["question"]
                }
                | prompt
                | llm
                | StrOutputParser()
            ),
            "sources": lambda x: x["formatted"]["sources"],
            "documents": lambda x: x["formatted"]["documents"]
        }
    )

    return chain

# 사용 예시
chain_with_sources = create_rag_with_sources()
result = chain_with_sources.invoke("근로자의 연차 유급휴가 일수는 어떻게 계산되나요?")

print(f"💬 답변:\n{result['answer']}")
print(f"\n📚 참조 문서:")
for source in result['sources']:
    print(f"  - {source}")

print(f"\n📄 사용된 문서 상세:")
for i, doc in enumerate(result['documents'], 1):
    print(f"\n문서 {i} ({doc.metadata.get('source')}):")
    print(doc.page_content[:150])
```

</details>

---

### 문제 3: 대화 기록 관리

이전 대화 내용을 고려하는 대화형 RAG 시스템을 구현하세요.

**요구사항:**
- 대화 기록 저장
- 컨텍스트 윈도우 관리 (최근 N개 대화)
- 대화 흐름 유지

<details>
<summary>솔루션 보기</summary>

```python
from typing import List, Tuple

class ConversationalRAG:
    """대화 기록을 관리하는 RAG 시스템"""

    def __init__(self, rag_chain, max_history: int = 5):
        """
        Args:
            rag_chain: 기본 RAG 체인
            max_history: 유지할 최대 대화 기록 수
        """
        self.rag_chain = rag_chain
        self.max_history = max_history
        self.conversation_history: List[Tuple[str, str]] = []

    def format_history(self) -> str:
        """대화 기록을 문자열로 포맷팅"""
        if not self.conversation_history:
            return ""

        history_text = "이전 대화:\\n"
        for q, a in self.conversation_history[-self.max_history:]:
            history_text += f"Q: {q}\\nA: {a}\\n\\n"
        return history_text

    def ask(self, question: str) -> str:
        """
        질문하고 답변 받기 (대화 기록 포함)

        Args:
            question: 사용자 질문

        Returns:
            답변
        """
        # 대화 기록과 함께 질문 구성
        if self.conversation_history:
            full_question = f"{self.format_history()}\\n현재 질문: {question}"
        else:
            full_question = question

        # RAG 체인 실행
        answer = self.rag_chain.invoke(full_question)

        # 대화 기록 저장
        self.conversation_history.append((question, answer))

        # 최대 기록 수 유지
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

        return answer

    def clear_history(self):
        """대화 기록 초기화"""
        self.conversation_history = []
        print("✅ 대화 기록이 초기화되었습니다.")

    def show_history(self):
        """대화 기록 출력"""
        if not self.conversation_history:
            print("대화 기록이 없습니다.")
            return

        print("📜 대화 기록:")
        for i, (q, a) in enumerate(self.conversation_history, 1):
            print(f"\\n대화 {i}:")
            print(f"Q: {q}")
            print(f"A: {a[:100]}...")

# 사용 예시
conversational_rag = ConversationalRAG(chain, max_history=5)

# 연속 대화
print("대화 1:")
answer1 = conversational_rag.ask("개인정보보호법의 주요 내용은 무엇인가요?")
print(f"답변: {answer1}\\n")

print("대화 2 (이전 맥락 참조):")
answer2 = conversational_rag.ask("위반 시 벌칙은 어떻게 되나요?")
print(f"답변: {answer2}\\n")

print("대화 3 (이전 맥락 참조):")
answer3 = conversational_rag.ask("개인정보 유출 사고 발생 시 조치는?")
print(f"답변: {answer3}\\n")

# 대화 기록 확인
conversational_rag.show_history()
```

</details>

---

### 문제 4: 성능 모니터링 대시보드

RAG 시스템의 성능을 모니터링하는 대시보드를 구축하세요.

**요구사항:**
- 응답 시간 측정
- 검색된 문서 수 추적
- 평균 유사도 점수
- 시각화 (matplotlib)

<details>
<summary>솔루션 보기</summary>

```python
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False

class RAGMonitor:
    """RAG 시스템 성능 모니터링"""

    def __init__(self):
        self.metrics = defaultdict(list)

    def track_query(self, question, answer, retrieval_time, generation_time, num_docs, avg_score):
        """쿼리 메트릭 기록"""
        self.metrics['questions'].append(question)
        self.metrics['retrieval_times'].append(retrieval_time)
        self.metrics['generation_times'].append(generation_time)
        self.metrics['total_times'].append(retrieval_time + generation_time)
        self.metrics['num_docs'].append(num_docs)
        self.metrics['avg_scores'].append(avg_score)

    def monitored_query(self, rag_chain, retriever, question):
        """모니터링이 포함된 쿼리 실행"""

        # 1. 검색 시간 측정
        retrieval_start = time.time()
        docs = retriever.get_relevant_documents(question)
        retrieval_time = time.time() - retrieval_start

        # 2. 생성 시간 측정
        generation_start = time.time()
        answer = rag_chain.invoke(question)
        generation_time = time.time() - generation_start

        # 3. 메트릭 계산
        num_docs = len(docs)
        # 유사도 점수는 벡터스토어에서 가져와야 하지만 간소화
        avg_score = 0.85  # 예시 값

        # 4. 기록
        self.track_query(
            question, answer,
            retrieval_time, generation_time,
            num_docs, avg_score
        )

        return answer

    def visualize_performance(self):
        """성능 지표 시각화"""

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 응답 시간 분포
        axes[0, 0].hist(self.metrics['total_times'], bins=20, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('응답 시간 (초)')
        axes[0, 0].set_ylabel('빈도')
        axes[0, 0].set_title('응답 시간 분포')
        axes[0, 0].axvline(
            sum(self.metrics['total_times'])/len(self.metrics['total_times']),
            color='red', linestyle='--', label=f"평균: {sum(self.metrics['total_times'])/len(self.metrics['total_times']):.2f}초"
        )
        axes[0, 0].legend()

        # 2. 검색 vs 생성 시간
        queries = list(range(1, len(self.metrics['retrieval_times'])+1))
        axes[0, 1].plot(queries, self.metrics['retrieval_times'], label='검색 시간', marker='o')
        axes[0, 1].plot(queries, self.metrics['generation_times'], label='생성 시간', marker='s')
        axes[0, 1].set_xlabel('쿼리 번호')
        axes[0, 1].set_ylabel('시간 (초)')
        axes[0, 1].set_title('쿼리별 시간 비교')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # 3. 검색된 문서 수
        axes[1, 0].bar(queries, self.metrics['num_docs'], color='lightcoral')
        axes[1, 0].set_xlabel('쿼리 번호')
        axes[1, 0].set_ylabel('문서 수')
        axes[1, 0].set_title('검색된 문서 수')
        axes[1, 0].grid(axis='y', alpha=0.3)

        # 4. 평균 유사도 점수
        axes[1, 1].plot(queries, self.metrics['avg_scores'], marker='D', color='green')
        axes[1, 1].set_xlabel('쿼리 번호')
        axes[1, 1].set_ylabel('평균 유사도')
        axes[1, 1].set_title('검색 품질 (평균 유사도)')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 통계 출력
        print("\\n📊 성능 통계:")
        print(f"  - 총 쿼리 수: {len(self.metrics['questions'])}")
        print(f"  - 평균 응답 시간: {sum(self.metrics['total_times'])/len(self.metrics['total_times']):.3f}초")
        print(f"  - 평균 검색 시간: {sum(self.metrics['retrieval_times'])/len(self.metrics['retrieval_times']):.3f}초")
        print(f"  - 평균 생성 시간: {sum(self.metrics['generation_times'])/len(self.metrics['generation_times']):.3f}초")
        print(f"  - 평균 검색 문서: {sum(self.metrics['num_docs'])/len(self.metrics['num_docs']):.1f}개")

# 사용 예시
monitor = RAGMonitor()

test_questions = [
    "개인정보보호법의 주요 내용은?",
    "임대차 계약 갱신 요구권은?",
    "근로자의 연차 휴가는?",
    "상가건물 임대차보호법 적용 범위는?",
    "개인정보 유출 시 조치는?"
]

retriever = legal_rag.vectorstore.as_retriever(search_kwargs={"k": 4})

for question in test_questions:
    answer = monitor.monitored_query(chain, retriever, question)
    print(f"Q: {question}")
    print(f"A: {answer[:100]}...\\n")

# 성능 시각화
monitor.visualize_performance()
```

</details>

---

## 🚀 실무 활용 예시

### 예시 1: 법률 자문 챗봇

**시나리오**: 고객이 법률 상담을 받을 수 있는 웹 챗봇 서비스

```python
import gradio as gr

def create_legal_chatbot():
    """법률 자문 챗봇 UI"""

    # RAG 시스템 초기화
    rag = LegalDocumentRAG("legal_docs", "./chroma_db")
    rag_chain = rag.setup_rag_chain_advanced()

    def chatbot_response(message, history):
        """챗봇 응답 함수"""
        try:
            # 질문 처리
            answer, terms = rag.ask_question(message, rag_chain, explain_terms=True)

            # 답변 포맷팅
            response = f"💬 {answer}"
            if terms and "법률 용어 설명이 필요하지 않습니다" not in terms:
                response += f"\\n\\n📚 법률 용어 설명:\\n{terms}"

            return response

        except Exception as e:
            return f"⚠️ 오류 발생: {str(e)}"

    # Gradio UI
    demo = gr.ChatInterface(
        chatbot_response,
        chatbot=gr.Chatbot(height=500),
        textbox=gr.Textbox(placeholder="법률 관련 질문을 입력하세요...", container=False, scale=7),
        title="⚖️ AI 법률 자문 챗봇",
        description="한국 법률(개인정보보호법, 근로기준법, 주택임대차보호법)에 대해 질문하세요.",
        theme="soft",
        examples=[
            "개인정보 수집 시 고지해야 할 사항은?",
            "임대차 계약 갱신 요구권은 언제 행사하나요?",
            "근로자의 연차 유급휴가는 어떻게 계산되나요?"
        ],
        cache_examples=False
    )

    return demo

# 실행
# demo = create_legal_chatbot()
# demo.launch()
```

---

### 예시 2: 계약서 분석 시스템

**시나리오**: 계약서를 업로드하면 법적 리스크를 자동 분석

```python
def analyze_contract(contract_file_path: str) -> Dict[str, any]:
    """
    계약서 분석 시스템

    Args:
        contract_file_path: 계약서 파일 경로

    Returns:
        분석 결과 딕셔너리
    """

    # 1. 계약서 로드
    loader = UnstructuredPDFLoader(contract_file_path)
    contract_docs = loader.load()
    contract_text = "\\n".join([doc.page_content for doc in contract_docs])

    # 2. RAG 시스템 초기화
    rag = LegalDocumentRAG("legal_docs", "./chroma_db")
    rag_chain = rag.setup_rag_chain()

    # 3. 분석 항목별 질문
    analysis_questions = {
        "권리_의무": "이 계약서에서 각 당사자의 주요 권리와 의무는 무엇인가요?",
        "해지_조건": "계약 해지 조건과 절차는 어떻게 되나요?",
        "분쟁_해결": "분쟁 발생 시 해결 방법은 무엇인가요?",
        "법적_준수": "이 계약서가 관련 법률을 준수하고 있는지 검토해주세요.",
        "위험_요소": "법적 위험 요소나 불리한 조항이 있나요?"
    }

    results = {}
    for key, question in analysis_questions.items():
        full_question = f"{question}\\n\\n계약서 내용:\\n{contract_text[:2000]}"
        answer = rag_chain.invoke(full_question)
        results[key] = answer

    # 4. 종합 평가
    risk_score = calculate_risk_score(results)

    return {
        "분석_결과": results,
        "위험도_점수": risk_score,
        "권장_조치": generate_recommendations(results, risk_score)
    }

def calculate_risk_score(results: Dict[str, str]) -> float:
    """위험도 점수 계산 (0~100)"""
    # 간단한 키워드 기반 점수 (실제로는 더 정교한 분석 필요)
    risk_keywords = ["위험", "불리", "주의", "문제", "위반"]
    total_risk = 0

    for answer in results.values():
        risk_count = sum(1 for keyword in risk_keywords if keyword in answer)
        total_risk += risk_count

    # 정규화
    return min(total_risk * 10, 100)

def generate_recommendations(results: Dict[str, str], risk_score: float) -> List[str]:
    """권장 조치 생성"""
    recommendations = []

    if risk_score > 70:
        recommendations.append("⚠️ 고위험: 법률 전문가 검토 필수")
    elif risk_score > 40:
        recommendations.append("⚠️ 중위험: 일부 조항 재검토 권장")
    else:
        recommendations.append("✅ 저위험: 전반적으로 양호")

    # 결과 기반 구체적 권장사항
    if "위험" in results.get("위험_요소", ""):
        recommendations.append("⚠️ 위험 요소 발견: 해당 조항 수정 필요")

    return recommendations

# 사용 예시
# contract_analysis = analyze_contract("./data/sample_contract.pdf")
# print("📄 계약서 분석 결과:")
# for key, value in contract_analysis["분석_결과"].items():
#     print(f"\\n{key}:\\n{value}")
# print(f"\\n위험도: {contract_analysis['위험도_점수']}/100")
# print(f"권장 조치:\\n" + "\\n".join(contract_analysis['권장_조치']))
```

---

### 예시 3: 법률 문서 비교 시스템

**시나리오**: 여러 버전의 법률 문서를 비교하여 변경사항 분석

```python
def compare_legal_documents(doc1_path: str, doc2_path: str) -> Dict[str, any]:
    """
    법률 문서 비교 시스템

    Args:
        doc1_path: 첫 번째 문서 경로 (구버전)
        doc2_path: 두 번째 문서 경로 (신버전)

    Returns:
        비교 결과
    """

    # 1. 문서 로드
    loader1 = UnstructuredLoader(doc1_path)
    loader2 = UnstructuredLoader(doc2_path)

    doc1 = "\\n".join([d.page_content for d in loader1.load()])
    doc2 = "\\n".join([d.page_content for d in loader2.load()])

    # 2. LLM으로 차이점 분석
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    comparison_prompt = ChatPromptTemplate.from_template("""
다음 두 법률 문서의 차이점을 분석해주세요.

구버전:
{old_version}

신버전:
{new_version}

다음 항목별로 분석해주세요:
1. 주요 변경사항
2. 추가된 조항
3. 삭제된 조항
4. 수정된 조항
5. 법적 영향 및 주의사항

분석 결과:
""")

    chain = comparison_prompt | llm | StrOutputParser()

    result = chain.invoke({
        "old_version": doc1[:3000],  # 토큰 제한 고려
        "new_version": doc2[:3000]
    })

    return {
        "비교_분석": result,
        "구버전_길이": len(doc1),
        "신버전_길이": len(doc2),
        "변경_비율": abs(len(doc2) - len(doc1)) / len(doc1) * 100
    }

# 사용 예시
# comparison = compare_legal_documents(
#     "./data/개인정보보호법_v1.txt",
#     "./data/개인정보보호법_v2.txt"
# )
# print("📊 문서 비교 결과:")
# print(comparison["비교_분석"])
# print(f"\\n변경 비율: {comparison['변경_비율']:.1f}%")
```

---

## 📌 핵심 요약

### 1. RAG 시스템 구축 체크리스트

**환경 설정:**
- [ ] Unstructured & LangChain 설치
- [ ] Chroma DB 설정
- [ ] OpenAI API 키 설정

**문서 처리:**
- [ ] 다중 파일 형식 지원
- [ ] 적절한 청킹 전략
- [ ] 메타데이터 관리

**검색 최적화:**
- [ ] 임베딩 모델 선택
- [ ] 검색 전략 설정 (similarity/MMR)
- [ ] Top-K 조정

**응답 생성:**
- [ ] 도메인 특화 프롬프트
- [ ] 출처 추적
- [ ] 오류 처리

### 2. 성능 최적화 팁

**메모리 효율:**
- `lazy_load()` 사용
- 청크 크기 조정 (500-1500자)
- 불필요한 메타데이터 제거

**검색 품질:**
- 적절한 임베딩 모델 선택
- Top-K 값 최적화 (3-5개)
- MMR로 다양성 확보

**응답 속도:**
- Chroma DB 로컬 캐싱
- 배치 처리
- 비동기 처리

### 3. 프로덕션 고려사항

**확장성:**
- 벡터 DB 분산 처리
- 로드 밸런싱
- 캐싱 전략

**보안:**
- API 키 관리
- 접근 제어
- 데이터 암호화

**모니터링:**
- 응답 시간 추적
- 에러 로깅
- 사용량 분석

---

## 🔗 참고 자료

- **Unstructured**: https://unstructured.io/
- **LangChain**: https://python.langchain.com/
- **Chroma DB**: https://www.trychroma.com/
- **LCEL 가이드**: https://python.langchain.com/docs/expression_language/

---

이 가이드를 통해 **Unstructured와 LangChain을 통합한 실전 RAG 시스템**을 구축할 수 있습니다!
