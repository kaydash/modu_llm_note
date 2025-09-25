# 주요 언어 모델 공급자 및 활용 - 정리노트

## 📖 이 수업에서 배운 것

### 🎯 핵심 개념
- **LLM 공급자**: 대규모 언어 모델을 개발하고 서비스를 제공하는 기업들
- **RAG (Retrieval-Augmented Generation)**: 검색 기반 생성 AI 시스템
- **벡터 검색**: 문서를 임베딩으로 변환하여 의미적 유사도로 검색하는 방법

---

## 🏢 LLM 공급자 4가지

### 1. Google Gemini
- **특징**: 구글이 만든 멀티모달 AI (텍스트 + 이미지 + 오디오)
- **장점**: 무료 체험판 제공, 다양한 입력 형식 지원
- **사용법**: `ChatGoogleGenerativeAI(model="gemini-1.5-flash")`
- **API 키**: `GOOGLE_API_KEY` 환경변수 필요

### 2. Groq
- **특징**: 초고속 추론 속도 (1초 미만 응답)
- **장점**: 빠른 속도, 오픈소스 모델 지원
- **사용법**: `ChatGroq(model="llama-3.3-70b-versatile")`
- **API 키**: `GROQ_API_KEY` 환경변수 필요

### 3. Ollama
- **특징**: 로컬 컴퓨터에서 AI 모델 실행
- **장점**: 인터넷 없이 사용 가능, 무료
- **사용법**: `ChatOllama(model="qwen3:1.7b", base_url="...")`
- **주의**: 서버 설정과 인증 헤더 필요

### 4. OpenAI
- **특징**: GPT 시리즈 제공, 업계 표준
- **장점**: 안정적 성능, 높은 품질
- **사용법**: `ChatOpenAI(model="gpt-4o-mini")`
- **API 키**: `OPENAI_API_KEY` 환경변수 필요

---

## 🔧 실습에서 사용한 주요 기술

### 환경 설정
```python
from dotenv import load_dotenv
load_dotenv()  # .env 파일에서 API 키 불러오기
```

### ChromaDB 벡터 데이터베이스
```python
# 벡터 저장소 생성
chroma_db = Chroma(
    collection_name="db_korean_cosine_metadata",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="./chroma_db"
)

# 검색기 생성 (상위 4개 문서 검색)
retriever = chroma_db.as_retriever(search_kwargs={"k": 4})
```

### RAG 체인 구현
```python
def create_rag_chain(retriever, llm):
    template = """질문에 답하세요. 관련 정보가 없으면
    '답변에 필요한 근거를 찾지 못했습니다.'라고 답하세요.

    [Context] {context}
    [Question] {question}
    [Answer]"""

    prompt = ChatPromptTemplate.from_template(template)

    # 문서들을 하나의 텍스트로 합치는 함수
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    # RAG 체인 구성: 검색 → 포맷팅 → 프롬프트 → LLM → 출력
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

    return rag_chain
```

---

## 💡 실습에서 겪은 문제와 해결

### 문제 1: SQLite 버전 오류
**증상**: ChromaDB 사용 시 SQLite 버전 오류
**해결**:
```python
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
```

### 문제 2: Ollama 인증 오류
**증상**: 401 Unauthorized 오류
**해결**: headers에 Authorization 추가
```python
llm = ChatOllama(
    model="qwen3:1.7b",
    base_url="http://littletask.kro.kr:1410",
    headers={"Authorization": "Bearer Task123!"}
)
```

### 문제 3: LangFuse 추적 설정
**증상**: `from langfuse.callback import CallbackHandler` 모듈 오류
**해결**: LangFuse 3.x 버전에서는 import 경로 변경됨
```python
from langfuse import Langfuse

# Langfuse 클라이언트 생성
langfuse = Langfuse()
print("Langfuse 연결 성공")

# 또는 간단한 확인
print("langsmith 추적 여부: ", os.getenv('LANGCHAIN_TRACING_V2'))
```

### 문제 4: "답변에 필요한 근거를 찾지 못했습니다"
**원인**: 벡터 데이터베이스에 관련 문서가 없음
**해결**: 적절한 문서를 벡터 DB에 추가하거나 검색 쿼리 조정

---

## 📊 각 공급자별 특징 비교

| 공급자 | 속도 | 품질 | 비용 | 특징 |
|--------|------|------|------|------|
| **Groq** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 가장 빠름 |
| **OpenAI** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | 가장 안정적 |
| **Gemini** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 멀티모달 |
| **Ollama** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 로컬 실행 |

---

## 🤔 복습 질문

1. **RAG 시스템의 구성 요소는?**
   - 답: 검색기(Retriever) + 생성기(LLM) + 프롬프트

2. **벡터 검색과 키워드 검색의 차이는?**
   - 벡터: 의미적 유사도로 검색
   - 키워드: 단어 일치로 검색

3. **각 LLM 공급자를 언제 사용하면 좋을까?**
   - **빠른 응답이 필요할 때**: Groq (1초 미만 응답)
   - **높은 품질이 필요할 때**: OpenAI (안정적이고 정확한 답변)
   - **비용을 절약하고 싶을 때**: Ollama (로컬 실행으로 API 비용 없음)
   - **이미지도 함께 처리할 때**: Gemini (멀티모달 지원)

4. **API 키 설정할 때 주의할 점은?**
   - `.env` 파일에 저장하고 `load_dotenv()` 호출
   - 각 공급자별 환경변수명이 다름 (OPENAI_API_KEY, GOOGLE_API_KEY 등)
   - Ollama는 API 키 + base_url + headers 설정 모두 필요

---

## 💻 내가 구현해본 코드 패턴

### 기본 패턴
1. 환경변수 로드 (`load_dotenv()`)
2. 임베딩 모델 설정 (`OpenAIEmbeddings`)
3. 벡터 DB 연결 (`Chroma`)
4. LLM 초기화 (각 공급자별)
5. RAG 체인 생성 (`create_rag_chain`)
6. 질문하고 답변 받기 (`rag_chain.invoke()`)

### 주의할 점
- API 키는 반드시 `.env` 파일에 보관
- 각 공급자별로 import 구문이 다름
- Ollama는 서버 URL과 인증 헤더 설정 필요
- ChromaDB 사용 전 SQLite 버전 문제 해결 필요

---

## 🎯 다음에 공부할 것
- [ ] 더 복잡한 RAG 시스템 (하이브리드 검색, 재순위 등)
- [ ] 각 공급자의 더 다양한 모델들 테스트
- [ ] 성능 비교 및 평가 지표 학습
- [ ] 실제 프로젝트에 RAG 시스템 적용해보기

---

*이 노트는 'PRJ02_W2_001_LLM_Providers.ipynb' 수업 내용을 정리한 것입니다.*