# AI 개발 실무 가이드

## 🏗 프로젝트 구조 및 아키텍처

### 권장 프로젝트 구조

```
ai_project/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py
│   │   └── logging.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── llm.py
│   │   ├── embeddings.py
│   │   ├── vectorstore.py
│   │   └── chains.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── document_service.py
│   │   ├── rag_service.py
│   │   └── chat_service.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── requests.py
│   │   └── responses.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── helpers.py
│   │   └── validators.py
│   └── api/
│       ├── __init__.py
│       ├── main.py
│       └── routers/
│           ├── __init__.py
│           ├── chat.py
│           └── documents.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── data/
│   ├── raw/
│   ├── processed/
│   └── vectorstores/
├── notebooks/
│   └── experiments/
├── scripts/
│   ├── setup.py
│   ├── ingest_data.py
│   └── deploy.py
└── docs/
    ├── api.md
    ├── deployment.md
    └── usage.md
```

### 핵심 설정 파일들

#### `src/config/settings.py`
```python
from pydantic import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # API Keys
    openai_api_key: str
    langsmith_api_key: Optional[str] = None

    # LLM Settings
    model_name: str = "gpt-4.1-mini"
    temperature: float = 0.3
    max_tokens: int = 2000

    # Embedding Settings
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536

    # Vector Store Settings
    vectorstore_type: str = "chroma"  # chroma, faiss, pinecone
    vectorstore_path: str = "./data/vectorstores"
    collection_name: str = "documents"

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False

    # Database
    database_url: Optional[str] = None

    # Redis (for caching)
    redis_url: Optional[str] = None

    # Monitoring
    enable_tracing: bool = True
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# 전역 설정 인스턴스
settings = Settings()
```

#### `src/config/logging.py`
```python
import logging
import sys
from pathlib import Path

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """로깅 설정"""

    # 로그 포맷 설정
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    handlers = [console_handler]

    # 파일 핸들러 (옵션)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # 루트 로거 설정
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=handlers,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 외부 라이브러리 로그 레벨 조정
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
```

### 핵심 서비스 구현

#### `src/core/llm.py`
```python
from functools import lru_cache
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from src.config.settings import settings
import logging

logger = logging.getLogger(__name__)

@lru_cache()
def get_llm() -> BaseChatModel:
    """LLM 인스턴스 생성 (캐싱됨)"""

    try:
        llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.model_name,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            request_timeout=30,
            max_retries=3
        )

        logger.info(f"LLM initialized: {settings.model_name}")
        return llm

    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise

class LLMManager:
    """LLM 관리 클래스"""

    def __init__(self):
        self.llm = get_llm()
        self._request_count = 0
        self._error_count = 0

    def invoke(self, prompt: str, **kwargs) -> str:
        """안전한 LLM 호출"""
        try:
            self._request_count += 1
            response = self.llm.invoke(prompt, **kwargs)
            return response.content

        except Exception as e:
            self._error_count += 1
            logger.error(f"LLM invocation failed: {e}")
            raise

    def get_stats(self) -> dict:
        """LLM 사용 통계"""
        return {
            "total_requests": self._request_count,
            "total_errors": self._error_count,
            "error_rate": self._error_count / max(self._request_count, 1)
        }
```

#### `src/core/vectorstore.py`
```python
from functools import lru_cache
from typing import List, Optional, Dict, Any
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from src.core.embeddings import get_embeddings
from src.config.settings import settings
import logging

logger = logging.getLogger(__name__)

class VectorStoreFactory:
    """벡터 스토어 팩토리"""

    @staticmethod
    def create_vectorstore(vectorstore_type: str = None) -> VectorStore:
        """벡터 스토어 생성"""

        store_type = vectorstore_type or settings.vectorstore_type
        embeddings = get_embeddings()

        if store_type == "chroma":
            return Chroma(
                collection_name=settings.collection_name,
                embedding_function=embeddings,
                persist_directory=settings.vectorstore_path
            )

        elif store_type == "faiss":
            # FAISS는 디렉토리가 존재하는 경우만 로드
            faiss_path = f"{settings.vectorstore_path}/faiss_index"

            try:
                return FAISS.load_local(faiss_path, embeddings)
            except:
                # 인덱스가 없으면 새로 생성 (빈 문서로)
                empty_docs = [Document(page_content="초기화", metadata={})]
                vectorstore = FAISS.from_documents(empty_docs, embeddings)
                vectorstore.save_local(faiss_path)
                return vectorstore

        else:
            raise ValueError(f"Unsupported vectorstore type: {store_type}")

class VectorStoreManager:
    """벡터 스토어 관리 클래스"""

    def __init__(self):
        self.vectorstore = VectorStoreFactory.create_vectorstore()
        self._document_count = 0

    def add_documents(self, documents: List[Document]) -> List[str]:
        """문서 추가"""
        try:
            doc_ids = self.vectorstore.add_documents(documents)
            self._document_count += len(documents)

            logger.info(f"Added {len(documents)} documents to vectorstore")
            return doc_ids

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """유사도 검색"""
        try:
            if filter:
                results = self.vectorstore.similarity_search(
                    query, k=k, filter=filter
                )
            else:
                results = self.vectorstore.similarity_search(query, k=k)

            logger.debug(f"Retrieved {len(results)} documents for query: {query[:50]}...")
            return results

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise

    def get_retriever(self, **kwargs):
        """Retriever 반환"""
        search_kwargs = {
            "k": kwargs.get("k", 5),
            "search_type": kwargs.get("search_type", "similarity"),
        }

        if "filter" in kwargs:
            search_kwargs["filter"] = kwargs["filter"]

        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)

    def get_stats(self) -> Dict[str, Any]:
        """벡터 스토어 통계"""
        return {
            "document_count": self._document_count,
            "vectorstore_type": type(self.vectorstore).__name__
        }

@lru_cache()
def get_vectorstore_manager() -> VectorStoreManager:
    """벡터 스토어 매니저 싱글톤"""
    return VectorStoreManager()
```

#### `src/services/rag_service.py`
```python
from typing import Dict, List, Optional
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from src.core.llm import get_llm
from src.core.vectorstore import get_vectorstore_manager
from src.models.requests import RAGRequest
from src.models.responses import RAGResponse
import logging
import time

logger = logging.getLogger(__name__)

class RAGService:
    """RAG 서비스 클래스"""

    def __init__(self):
        self.llm = get_llm()
        self.vectorstore_manager = get_vectorstore_manager()
        self._setup_chains()

    def _setup_chains(self):
        """RAG 체인 설정"""

        # 시스템 프롬프트
        system_prompt = """
        당신은 도움이 되고 정확한 AI 어시스턴트입니다.
        제공된 컨텍스트를 바탕으로 질문에 답변해주세요.

        컨텍스트에서 답변할 수 없는 질문이라면 솔직히 모른다고 답변하세요.

        컨텍스트: {context}
        """

        # 프롬프트 템플릿
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        # 문서 결합 체인
        self.document_chain = create_stuff_documents_chain(
            self.llm, self.prompt
        )

        # 검색기
        self.retriever = self.vectorstore_manager.get_retriever(k=5)

        # RAG 체인
        self.rag_chain = create_retrieval_chain(
            self.retriever, self.document_chain
        )

    def query(self, request: RAGRequest) -> RAGResponse:
        """RAG 쿼리 처리"""

        start_time = time.time()

        try:
            # 메타데이터 필터 적용 (있는 경우)
            if request.filters:
                retriever = self.vectorstore_manager.get_retriever(
                    k=request.k or 5,
                    filter=request.filters
                )

                # 필터가 적용된 새로운 체인 생성
                filtered_chain = create_retrieval_chain(
                    retriever, self.document_chain
                )

                result = filtered_chain.invoke({"input": request.question})
            else:
                result = self.rag_chain.invoke({"input": request.question})

            # 실행 시간 계산
            execution_time = time.time() - start_time

            # 응답 구성
            response = RAGResponse(
                answer=result["answer"],
                sources=[
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": getattr(doc, "score", 0.0)
                    }
                    for doc in result.get("context", [])
                ],
                execution_time=execution_time,
                retrieved_docs=len(result.get("context", []))
            )

            logger.info(f"RAG query processed in {execution_time:.2f}s")
            return response

        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            raise

    def add_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """문서 추가"""
        try:
            doc_ids = self.vectorstore_manager.add_documents(documents)

            # 체인 재설정 (새로운 문서 반영)
            self._setup_chains()

            return {
                "status": "success",
                "added_documents": len(documents),
                "document_ids": doc_ids
            }

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    def get_service_stats(self) -> Dict[str, Any]:
        """서비스 통계"""
        llm_stats = self.llm.get_stats() if hasattr(self.llm, 'get_stats') else {}
        vectorstore_stats = self.vectorstore_manager.get_stats()

        return {
            "llm": llm_stats,
            "vectorstore": vectorstore_stats,
            "status": "healthy"
        }
```

#### `src/services/chat_service.py`
```python
from typing import Dict, List, Optional
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
import redis
import json
from datetime import datetime, timedelta
from src.core.llm import get_llm
from src.services.rag_service import RAGService
from src.models.requests import ChatRequest
from src.models.responses import ChatResponse
import logging

logger = logging.getLogger(__name__)

class RedisChatHistory(BaseChatMessageHistory, BaseModel):
    """Redis 기반 채팅 히스토리"""

    session_id: str
    ttl: int = 3600  # 1시간

    def __init__(self, session_id: str, redis_client=None, **kwargs):
        super().__init__(session_id=session_id, **kwargs)
        self.redis_client = redis_client or redis.Redis(
            host='localhost', port=6379, db=0, decode_responses=True
        )
        self._key = f"chat_history:{session_id}"

    @property
    def messages(self) -> List[BaseMessage]:
        """메시지 로드"""
        try:
            messages_data = self.redis_client.lrange(self._key, 0, -1)
            messages = []

            for msg_json in messages_data:
                msg_data = json.loads(msg_json)

                if msg_data["type"] == "human":
                    messages.append(HumanMessage(content=msg_data["content"]))
                elif msg_data["type"] == "ai":
                    messages.append(AIMessage(content=msg_data["content"]))

            return messages

        except Exception as e:
            logger.warning(f"Failed to load chat history: {e}")
            return []

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """메시지 추가"""
        try:
            for message in messages:
                msg_data = {
                    "type": "human" if isinstance(message, HumanMessage) else "ai",
                    "content": message.content,
                    "timestamp": datetime.now().isoformat()
                }

                self.redis_client.rpush(self._key, json.dumps(msg_data, ensure_ascii=False))

            # TTL 설정
            self.redis_client.expire(self._key, self.ttl)

        except Exception as e:
            logger.error(f"Failed to save chat history: {e}")

    def clear(self) -> None:
        """히스토리 클리어"""
        self.redis_client.delete(self._key)

class ChatService:
    """채팅 서비스"""

    def __init__(self, enable_rag: bool = True):
        self.llm = get_llm()
        self.enable_rag = enable_rag

        if enable_rag:
            self.rag_service = RAGService()

        # Redis 연결
        try:
            self.redis_client = redis.Redis(
                host='localhost', port=6379, db=0, decode_responses=True
            )
            self.redis_client.ping()  # 연결 테스트

        except Exception as e:
            logger.warning(f"Redis not available, using in-memory storage: {e}")
            self.redis_client = None

        self._setup_chains()

    def _setup_chains(self):
        """채팅 체인 설정"""

        if self.enable_rag:
            # RAG 기반 채팅 프롬프트
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", """당신은 도움이 되는 AI 어시스턴트입니다.

                이전 대화 내용을 참고하여 일관성 있게 대화하세요.
                질문에 대해 관련 문서가 있다면 해당 정보를 활용하여 답변하세요.

                관련 문서: {context}
                """),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}")
            ])

        else:
            # 일반 채팅 프롬프트
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", "당신은 친근하고 도움이 되는 AI 어시스턴트입니다."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}")
            ])

        self.chat_chain = chat_prompt | self.llm

        # 히스토리 관리 체인
        self.chain_with_history = RunnableWithMessageHistory(
            self.chat_chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """세션 히스토리 가져오기"""
        if self.redis_client:
            return RedisChatHistory(session_id, self.redis_client)
        else:
            # 인메모리 폴백 (실제 운영에서는 권장하지 않음)
            if not hasattr(self, '_memory_store'):
                self._memory_store = {}

            if session_id not in self._memory_store:
                self._memory_store[session_id] = InMemoryChatHistory()

            return self._memory_store[session_id]

    def chat(self, request: ChatRequest) -> ChatResponse:
        """채팅 처리"""

        start_time = time.time()

        try:
            # RAG 컨텍스트 준비 (RAG 활성화시)
            context = ""
            sources = []

            if self.enable_rag:
                # 문서 검색
                relevant_docs = self.rag_service.vectorstore_manager.similarity_search(
                    request.message, k=3
                )

                context = "\n".join([doc.page_content for doc in relevant_docs])
                sources = [
                    {
                        "content": doc.page_content[:200],
                        "metadata": doc.metadata
                    }
                    for doc in relevant_docs
                ]

            # 채팅 실행
            response = self.chain_with_history.invoke(
                {
                    "input": request.message,
                    "context": context if self.enable_rag else ""
                },
                config={"configurable": {"session_id": request.session_id}}
            )

            execution_time = time.time() - start_time

            return ChatResponse(
                response=response.content,
                session_id=request.session_id,
                sources=sources if self.enable_rag else [],
                execution_time=execution_time
            )

        except Exception as e:
            logger.error(f"Chat processing failed: {e}")
            raise

    def get_chat_history(self, session_id: str) -> List[Dict[str, str]]:
        """채팅 히스토리 조회"""
        try:
            history = self._get_session_history(session_id)

            formatted_history = []
            for message in history.messages:
                formatted_history.append({
                    "type": "human" if isinstance(message, HumanMessage) else "ai",
                    "content": message.content,
                    "timestamp": datetime.now().isoformat()  # 실제로는 메시지에서 추출
                })

            return formatted_history

        except Exception as e:
            logger.error(f"Failed to get chat history: {e}")
            return []

    def clear_chat_history(self, session_id: str) -> bool:
        """채팅 히스토리 삭제"""
        try:
            history = self._get_session_history(session_id)
            history.clear()
            return True

        except Exception as e:
            logger.error(f"Failed to clear chat history: {e}")
            return False

class InMemoryChatHistory(BaseChatMessageHistory, BaseModel):
    """인메모리 채팅 히스토리 (폴백용)"""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []
```

### API 엔드포인트

#### `src/api/main.py`
```python
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from src.config.settings import settings
from src.config.logging import setup_logging
from src.api.routers import chat, documents
from src.services.rag_service import RAGService
from src.services.chat_service import ChatService

# 로깅 설정
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)

# 서비스 인스턴스 (전역)
rag_service = None
chat_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 생명주기 관리"""
    # 시작 시
    global rag_service, chat_service

    logger.info("Initializing services...")

    try:
        rag_service = RAGService()
        chat_service = ChatService(enable_rag=True)

        logger.info("Services initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

    yield

    # 종료 시
    logger.info("Shutting down services...")

# FastAPI 앱 생성
app = FastAPI(
    title="AI Assistant API",
    description="LangChain 기반 RAG 및 채팅 API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 운영에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(documents.router, prefix="/api/v1/documents", tags=["documents"])

# 의존성 주입
def get_rag_service() -> RAGService:
    if rag_service is None:
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    return rag_service

def get_chat_service() -> ChatService:
    if chat_service is None:
        raise HTTPException(status_code=500, detail="Chat service not initialized")
    return chat_service

@app.get("/")
async def root():
    return {"message": "AI Assistant API", "status": "healthy"}

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    try:
        # 서비스 상태 확인
        rag_stats = rag_service.get_service_stats() if rag_service else {}

        return {
            "status": "healthy",
            "services": {
                "rag": rag_stats,
                "chat": {"status": "ready"} if chat_service else {"status": "not_initialized"}
            }
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Service unhealthy")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_debug,
        log_level=settings.log_level.lower()
    )
```

#### `src/api/routers/chat.py`
```python
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict
import uuid

from src.services.chat_service import ChatService
from src.models.requests import ChatRequest
from src.models.responses import ChatResponse
from src.api.main import get_chat_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service)
):
    """채팅 메시지 처리"""
    try:
        # 세션 ID가 없으면 생성
        if not request.session_id:
            request.session_id = str(uuid.uuid4())

        response = chat_service.chat(request)
        return response

    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{session_id}")
async def get_chat_history(
    session_id: str,
    chat_service: ChatService = Depends(get_chat_service)
) -> List[Dict[str, str]]:
    """채팅 히스토리 조회"""
    try:
        history = chat_service.get_chat_history(session_id)
        return history

    except Exception as e:
        logger.error(f"Failed to get chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/history/{session_id}")
async def clear_chat_history(
    session_id: str,
    chat_service: ChatService = Depends(get_chat_service)
):
    """채팅 히스토리 삭제"""
    try:
        success = chat_service.clear_chat_history(session_id)

        if success:
            return {"message": "Chat history cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear chat history")

    except Exception as e:
        logger.error(f"Failed to clear chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

### 모델 정의

#### `src/models/requests.py`
```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class RAGRequest(BaseModel):
    question: str = Field(..., description="검색할 질문")
    k: Optional[int] = Field(5, description="검색할 문서 수")
    filters: Optional[Dict[str, Any]] = Field(None, description="메타데이터 필터")

class ChatRequest(BaseModel):
    message: str = Field(..., description="채팅 메시지")
    session_id: Optional[str] = Field(None, description="세션 ID")

class DocumentUploadRequest(BaseModel):
    documents: List[Dict[str, Any]] = Field(..., description="업로드할 문서들")

    class Config:
        schema_extra = {
            "example": {
                "documents": [
                    {
                        "content": "문서 내용",
                        "metadata": {
                            "source": "example.pdf",
                            "page": 1
                        }
                    }
                ]
            }
        }
```

#### `src/models/responses.py`
```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class DocumentSource(BaseModel):
    content: str = Field(..., description="문서 내용")
    metadata: Dict[str, Any] = Field(..., description="문서 메타데이터")
    score: Optional[float] = Field(None, description="유사도 점수")

class RAGResponse(BaseModel):
    answer: str = Field(..., description="생성된 답변")
    sources: List[DocumentSource] = Field(..., description="참조한 문서들")
    execution_time: float = Field(..., description="실행 시간 (초)")
    retrieved_docs: int = Field(..., description="검색된 문서 수")

class ChatResponse(BaseModel):
    response: str = Field(..., description="채팅 응답")
    session_id: str = Field(..., description="세션 ID")
    sources: List[DocumentSource] = Field(default_factory=list, description="참조 문서 (RAG 활성화 시)")
    execution_time: float = Field(..., description="실행 시간 (초)")

class DocumentUploadResponse(BaseModel):
    status: str = Field(..., description="업로드 상태")
    added_documents: int = Field(..., description="추가된 문서 수")
    document_ids: List[str] = Field(..., description="문서 ID 목록")
```

## 🧪 테스트 구조

### `tests/conftest.py`
```python
import pytest
import tempfile
import shutil
from pathlib import Path

from src.config.settings import Settings
from src.core.llm import get_llm
from src.core.vectorstore import VectorStoreManager
from src.services.rag_service import RAGService

@pytest.fixture(scope="session")
def test_settings():
    """테스트용 설정"""
    return Settings(
        openai_api_key="test_key",
        vectorstore_type="chroma",
        vectorstore_path="./test_vectorstore",
        api_debug=True
    )

@pytest.fixture
def temp_vectorstore():
    """임시 벡터스토어"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_documents():
    """테스트용 샘플 문서"""
    from langchain_core.documents import Document

    return [
        Document(
            page_content="Python은 프로그래밍 언어입니다.",
            metadata={"source": "python.txt", "page": 1}
        ),
        Document(
            page_content="머신러닝은 AI의 한 분야입니다.",
            metadata={"source": "ml.txt", "page": 1}
        )
    ]

@pytest.fixture
def mock_llm(monkeypatch):
    """Mock LLM"""
    class MockLLM:
        def invoke(self, prompt, **kwargs):
            class MockResponse:
                content = "테스트 응답"
            return MockResponse()

    def mock_get_llm():
        return MockLLM()

    monkeypatch.setattr("src.core.llm.get_llm", mock_get_llm)
    return mock_get_llm()

@pytest.fixture
def rag_service(mock_llm, temp_vectorstore, sample_documents, monkeypatch):
    """RAG 서비스 픽스처"""
    # 설정 오버라이드
    monkeypatch.setenv("VECTORSTORE_PATH", temp_vectorstore)

    service = RAGService()
    service.add_documents(sample_documents)

    return service
```

### `tests/unit/test_rag_service.py`
```python
import pytest
from src.models.requests import RAGRequest
from src.services.rag_service import RAGService

class TestRAGService:
    """RAG 서비스 단위 테스트"""

    def test_query_basic(self, rag_service):
        """기본 쿼리 테스트"""
        request = RAGRequest(question="Python이 무엇인가요?")
        response = rag_service.query(request)

        assert response.answer is not None
        assert len(response.sources) > 0
        assert response.execution_time > 0
        assert response.retrieved_docs > 0

    def test_query_with_filter(self, rag_service):
        """필터 적용 쿼리 테스트"""
        request = RAGRequest(
            question="프로그래밍에 대해 알려주세요",
            filters={"source": "python.txt"}
        )
        response = rag_service.query(request)

        assert response.answer is not None
        # 필터된 문서만 반환되는지 확인
        for source in response.sources:
            assert source.metadata.get("source") == "python.txt"

    def test_add_documents(self, rag_service, sample_documents):
        """문서 추가 테스트"""
        new_docs = sample_documents[:1]  # 첫 번째 문서만

        result = rag_service.add_documents(new_docs)

        assert result["status"] == "success"
        assert result["added_documents"] == 1
        assert len(result["document_ids"]) == 1

    def test_service_stats(self, rag_service):
        """서비스 통계 테스트"""
        stats = rag_service.get_service_stats()

        assert "vectorstore" in stats
        assert "status" in stats
        assert stats["status"] == "healthy"
```

### `tests/integration/test_api.py`
```python
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

class TestAPIIntegration:
    """API 통합 테스트"""

    def test_health_check(self):
        """헬스 체크 테스트"""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_chat_endpoint(self):
        """채팅 엔드포인트 테스트"""
        chat_request = {
            "message": "안녕하세요",
            "session_id": "test_session"
        }

        response = client.post("/api/v1/chat/", json=chat_request)

        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert data["session_id"] == "test_session"

    def test_chat_history(self):
        """채팅 히스토리 테스트"""
        session_id = "test_session_history"

        # 먼저 채팅 메시지 전송
        chat_request = {
            "message": "테스트 메시지",
            "session_id": session_id
        }
        client.post("/api/v1/chat/", json=chat_request)

        # 히스토리 조회
        response = client.get(f"/api/v1/chat/history/{session_id}")

        assert response.status_code == 200
        history = response.json()
        assert len(history) >= 2  # 사용자 메시지 + AI 응답

    def test_clear_chat_history(self):
        """채팅 히스토리 삭제 테스트"""
        session_id = "test_session_clear"

        # 채팅 메시지 전송
        chat_request = {
            "message": "삭제될 메시지",
            "session_id": session_id
        }
        client.post("/api/v1/chat/", json=chat_request)

        # 히스토리 삭제
        response = client.delete(f"/api/v1/chat/history/{session_id}")

        assert response.status_code == 200
        assert response.json()["message"] == "Chat history cleared successfully"

        # 히스토리가 비워졌는지 확인
        history_response = client.get(f"/api/v1/chat/history/{session_id}")
        history = history_response.json()
        assert len(history) == 0
```

## 🐳 Docker 및 배포

### `Dockerfile`
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 환경변수 설정
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 포트 노출
EXPOSE 8000

# 헬스체크
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 애플리케이션 실행
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `docker-compose.yml`
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/aiassistant
    depends_on:
      - redis
      - postgres
    volumes:
      - ./data:/app/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=aiassistant
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
```

### `scripts/deploy.py`
```python
#!/usr/bin/env python3
"""
배포 스크립트
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, check=True):
    """명령 실행"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)

    print(result.stdout)
    return result

def deploy_local():
    """로컬 배포"""
    print("=== Local Deployment ===")

    # Docker 이미지 빌드
    run_command("docker-compose build")

    # 서비스 시작
    run_command("docker-compose up -d")

    # 서비스 상태 확인
    run_command("docker-compose ps")

    print("Local deployment completed!")
    print("API available at: http://localhost:8000")

def deploy_production():
    """운영 배포"""
    print("=== Production Deployment ===")

    # 환경변수 확인
    required_vars = ["OPENAI_API_KEY", "DATABASE_URL", "REDIS_URL"]
    for var in required_vars:
        if not os.getenv(var):
            print(f"Error: {var} environment variable not set")
            sys.exit(1)

    # 테스트 실행
    run_command("python -m pytest tests/")

    # Docker 이미지 빌드 및 태깅
    image_tag = os.getenv("IMAGE_TAG", "latest")
    run_command(f"docker build -t ai-assistant:{image_tag} .")

    # 컨테이너 레지스트리에 푸시 (예: AWS ECR)
    if os.getenv("ECR_REGISTRY"):
        registry = os.getenv("ECR_REGISTRY")
        run_command(f"docker tag ai-assistant:{image_tag} {registry}/ai-assistant:{image_tag}")
        run_command(f"docker push {registry}/ai-assistant:{image_tag}")

    print("Production deployment completed!")

def rollback():
    """롤백"""
    print("=== Rollback ===")

    previous_tag = os.getenv("PREVIOUS_TAG")
    if not previous_tag:
        print("Error: PREVIOUS_TAG environment variable not set")
        sys.exit(1)

    # 이전 버전으로 롤백
    run_command(f"docker-compose pull ai-assistant:{previous_tag}")
    run_command("docker-compose up -d")

    print(f"Rolled back to version: {previous_tag}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python deploy.py [local|production|rollback]")
        sys.exit(1)

    action = sys.argv[1]

    if action == "local":
        deploy_local()
    elif action == "production":
        deploy_production()
    elif action == "rollback":
        rollback()
    else:
        print("Invalid action. Use: local, production, or rollback")
        sys.exit(1)
```

## 📊 모니터링 및 로깅

### `src/utils/monitoring.py`
```python
import time
import logging
from functools import wraps
from typing import Dict, Any, Optional
import psutil
import threading
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """성능 모니터링 클래스"""

    def __init__(self):
        self.metrics = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_error": 0,
            "response_times": [],
            "memory_usage": [],
            "cpu_usage": [],
        }
        self._start_monitoring()

    def _start_monitoring(self):
        """시스템 모니터링 시작"""
        def monitor_system():
            while True:
                try:
                    # 메모리 사용량
                    memory_percent = psutil.virtual_memory().percent
                    self.metrics["memory_usage"].append({
                        "timestamp": datetime.now(),
                        "value": memory_percent
                    })

                    # CPU 사용량
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.metrics["cpu_usage"].append({
                        "timestamp": datetime.now(),
                        "value": cpu_percent
                    })

                    # 오래된 데이터 정리 (1시간 이상)
                    cutoff_time = datetime.now() - timedelta(hours=1)

                    self.metrics["memory_usage"] = [
                        m for m in self.metrics["memory_usage"]
                        if m["timestamp"] > cutoff_time
                    ]

                    self.metrics["cpu_usage"] = [
                        m for m in self.metrics["cpu_usage"]
                        if m["timestamp"] > cutoff_time
                    ]

                    time.sleep(60)  # 1분마다 수집

                except Exception as e:
                    logger.error(f"System monitoring error: {e}")
                    time.sleep(60)

        monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        monitor_thread.start()

    def record_request(self, execution_time: float, success: bool = True):
        """요청 기록"""
        self.metrics["requests_total"] += 1

        if success:
            self.metrics["requests_success"] += 1
        else:
            self.metrics["requests_error"] += 1

        # 최근 1000개 응답시간만 보관
        self.metrics["response_times"].append({
            "timestamp": datetime.now(),
            "value": execution_time
        })

        if len(self.metrics["response_times"]) > 1000:
            self.metrics["response_times"] = self.metrics["response_times"][-1000:]

    def get_metrics(self) -> Dict[str, Any]:
        """메트릭 조회"""
        # 응답시간 통계
        response_times = [m["value"] for m in self.metrics["response_times"]]

        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
        else:
            avg_response_time = min_response_time = max_response_time = 0

        # 현재 시스템 상태
        current_memory = psutil.virtual_memory().percent
        current_cpu = psutil.cpu_percent()

        return {
            "requests": {
                "total": self.metrics["requests_total"],
                "success": self.metrics["requests_success"],
                "error": self.metrics["requests_error"],
                "success_rate": (
                    self.metrics["requests_success"] / max(self.metrics["requests_total"], 1) * 100
                )
            },
            "response_times": {
                "average": avg_response_time,
                "min": min_response_time,
                "max": max_response_time,
                "count": len(response_times)
            },
            "system": {
                "memory_percent": current_memory,
                "cpu_percent": current_cpu,
                "disk_usage": psutil.disk_usage('/').percent
            }
        }

# 전역 모니터 인스턴스
monitor = PerformanceMonitor()

def track_performance(func):
    """성능 추적 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        success = True

        try:
            result = func(*args, **kwargs)
            return result

        except Exception as e:
            success = False
            logger.error(f"Function {func.__name__} failed: {e}")
            raise

        finally:
            execution_time = time.time() - start_time
            monitor.record_request(execution_time, success)

    return wrapper

def get_system_metrics() -> Dict[str, Any]:
    """시스템 메트릭 조회"""
    return monitor.get_metrics()
```

이 개발 가이드를 통해 실제 운영 환경에서 안정적이고 확장 가능한 AI 애플리케이션을 구축할 수 있습니다. 각 컴포넌트는 모듈화되어 있어 필요에 따라 독립적으로 개발하고 테스트할 수 있습니다.