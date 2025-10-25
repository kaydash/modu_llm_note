# LangChain ReAct 에이전트와 다국어 RAG 시스템

## 📚 학습 목표

- **ReAct 프레임워크**의 개념과 동작 원리를 이해한다
- **다국어 RAG 시스템**을 구축하고 언어 교차 검색을 구현할 수 있다
- **언어 감지 및 번역** 자동화를 통해 seamless한 다국어 처리를 구현한다
- **벡터저장소 라우팅**을 통해 언어별 최적화된 검색 시스템을 구축한다
- **LangChain Agent**를 활용하여 자율적인 의사결정 시스템을 구현할 수 있다

## 🔑 핵심 개념

### ReAct (Reasoning and Acting)

**ReAct**는 추론(Reasoning)과 행동(Acting)을 결합한 에이전트 프레임워크입니다.

- **행동-관찰-추론 순환**: 에이전트가 반복적으로 작업을 수행하며 목표 달성
  - **행동 (Act)**: LLM이 특정 도구(Tool)를 호출
  - **관찰 (Observe)**: 도구의 실행 결과를 확인
  - **추론 (Reason)**: 관찰 결과를 바탕으로 다음 행동 결정

- **Tool Calling**: LLM이 필요한 도구를 선택하고 호출하는 메커니즘
- **Agent**: 자율적으로 의사결정을 내리며 작업을 수행하는 시스템

### 다국어 RAG 시스템

**언어 교차 검색 (Cross-lingual Search)**
- 서로 다른 언어 간의 정보 검색을 가능하게 하는 기술
- 질의어와 문서가 다른 언어여도 의미적 연관성 기반 검색 가능
- 다국어 임베딩 모델 활용 (예: OpenAI text-embedding-3-small, HuggingFace bge-m3)

**언어 감지 및 번역**
- `langdetect` 라이브러리로 입력 텍스트의 언어 자동 식별
- `LibreTranslate` 오픈소스 번역 API를 통한 자동 번역
- 한국어↔다국어 양방향 번역 지원

**벡터저장소 라우팅**
- 언어별 벡터저장소를 분리하여 독립적 관리
- 언어 감지 후 해당 언어의 벡터저장소로 자동 라우팅
- 각 언어에 최적화된 검색 성능 제공

### 관련 기술 스택

```python
# LangChain 핵심
langchain-core        # 기본 추상화 및 인터페이스
langchain-openai      # OpenAI 통합
langchain-community   # 커뮤니티 통합 (문서 로더 등)
langchain-chroma      # Chroma 벡터 DB
langchain-huggingface # HuggingFace 임베딩

# 다국어 지원
langdetect           # 언어 감지
libretranslate       # 오픈소스 번역 API

# 벡터 임베딩
openai               # OpenAI 임베딩 모델
sentence-transformers # HuggingFace 임베딩
```

## 🛠 환경 설정

### 필요한 라이브러리 설치

```bash
pip install langchain langchain-openai langchain-community langchain-chroma
pip install langchain-huggingface langchain-ollama
pip install langdetect libretranslate python-dotenv
pip install chromadb tiktoken
```

### API 키 설정

```.env
OPENAI_API_KEY=your_openai_api_key_here
```

### 기본 설정 코드

```python
from dotenv import load_dotenv
import os
import warnings

# 환경 변수 로드
load_dotenv()

# 경고 메시지 비활성화
warnings.filterwarnings("ignore")

# OpenAI API 키 확인
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
```

## 💻 단계별 구현

### 1단계: 다국어 RAG 시스템 구축

#### 1.1 다국어 문서 로드 및 전처리

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from glob import glob

def load_text_files(file_paths):
    """여러 텍스트 파일을 로드하는 함수"""
    documents = []
    for file_path in file_paths:
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())
        except Exception as e:
            print(f"파일 로드 실패 ({file_path}): {e}")
    return documents

# 데이터 디렉토리 설정
data_dir = os.path.join(os.getcwd(), 'data')

# 한국어 문서 로드
korean_txt_files = glob(os.path.join(data_dir, '*_KR.md'))
korean_data = load_text_files(korean_txt_files)
print(f"로드된 한국어 문서 수: {len(korean_data)}")

# 영어 문서 로드
english_txt_files = glob(os.path.join(data_dir, '*_EN.md'))
english_data = load_text_files(english_txt_files)
print(f"로드된 영어 문서 수: {len(english_data)}")
```

#### 1.2 문서 분할 (Chunking)

```python
# TikToken 기반 문서 분할기 생성
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",              # TikToken 인코더
    separators=['\n\n', '\n', r'(?<=[.!?])\s+'],  # 구분자 (문단, 줄바꿈, 문장)
    chunk_size=500,                            # 청크 크기 (토큰 단위)
    chunk_overlap=50,                          # 청크 간 겹침
    length_function=len,
    is_separator_regex=True                    # 정규표현식 사용
)

# 한국어 문서 분할
korean_docs = text_splitter.split_documents(korean_data)
print(f"분할된 한국어 문서 수: {len(korean_docs)}")

# 영어 문서 분할
english_docs = text_splitter.split_documents(english_data)
print(f"분할된 영어 문서 수: {len(english_docs)}")
```

**주요 파라미터 설명:**
- `encoding_name`: TikToken 인코더 이름 (OpenAI 모델과 호환)
- `separators`: 문서를 나누는 구분자 (우선순위 순)
- `chunk_size`: 각 청크의 최대 크기 (토큰 단위)
- `chunk_overlap`: 연속성 유지를 위한 겹침 크기

#### 1.3 다국어 임베딩 모델 비교

```python
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings

# 1. OpenAI 임베딩 모델 (한국어 지원 우수)
embeddings_openai = OpenAIEmbeddings(model="text-embedding-3-small")
print("✅ OpenAI 임베딩 모델 생성 완료")

# 2. HuggingFace 임베딩 모델 (한국어 지원)
try:
    embeddings_huggingface = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",              # 다국어 임베딩 모델
        model_kwargs={'device': 'cpu'},        # CPU 사용
        encode_kwargs={'normalize_embeddings': True}  # 정규화 활성화
    )
    print("✅ HuggingFace 임베딩 모델 생성 완료")
except Exception as e:
    print(f"❌ HuggingFace 임베딩 모델 생성 실패: {e}")
    embeddings_huggingface = None

# 3. Ollama 임베딩 모델 (한국어 미지원)
try:
    embeddings_ollama = OllamaEmbeddings(model="nomic-embed-text")
    print("✅ Ollama 임베딩 모델 생성 완료")
except Exception as e:
    print(f"❌ Ollama 임베딩 모델 생성 실패: {e}")
    embeddings_ollama = None
```

**모델 비교:**

| 모델 | 한국어 지원 | 장점 | 단점 |
|------|------------|------|------|
| OpenAI text-embedding-3-small | ⭐⭐⭐ | 우수한 성능, 다국어 지원 | 유료 API |
| HuggingFace bge-m3 | ⭐⭐ | 무료, 로컬 실행 가능 | 속도 느림 |
| Ollama nomic-embed-text | ❌ | 무료, 경량화 | 한국어 미지원 |

#### 1.4 벡터 저장소 생성 및 성능 비교

```python
from langchain_chroma import Chroma

# 모든 문서 병합 (한국어 + 영어)
all_docs = korean_docs + english_docs
print(f"총 문서 수: {len(all_docs)}")

# OpenAI 벡터 저장소
db_openai = Chroma.from_documents(
    documents=all_docs,
    embedding=embeddings_openai,
    collection_name="multilang_db_openai",
    persist_directory="./chroma_db"
)
print(f"✅ OpenAI 벡터 저장소 생성 완료 (문서 수: {db_openai._collection.count()})")

# HuggingFace 벡터 저장소
if embeddings_huggingface:
    db_huggingface = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings_huggingface,
        collection_name="multilang_db_huggingface",
        persist_directory="./chroma_db"
    )
    print(f"✅ HuggingFace 벡터 저장소 생성 완료 (문서 수: {db_huggingface._collection.count()})")
else:
    db_huggingface = None
```

#### 1.5 RAG 체인 생성 및 성능 평가

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chat_models import init_chat_model

# 프롬프트 템플릿 정의
template = """Answer the question based only on the following context.
Do not use any external information or knowledge.
If the answer is not in the context, answer "I don't know".

When answering:
- For proper nouns (names of people, places, organizations), keep the original language.
- Provide clear and concise answers.

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# LLM 모델 초기화
llm = init_chat_model("openai:gpt-4.1-mini", temperature=0)

# 문서 포맷 함수
def format_docs(docs):
    """검색된 문서를 문자열로 포맷"""
    return "\n\n".join([doc.page_content for doc in docs])

# RAG 체인 생성 함수
def create_rag_chain(vectorstore, top_k=4):
    """벡터 저장소를 사용하여 RAG 체인 생성"""
    retriever = vectorstore.as_retriever(search_kwargs={'k': top_k})

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# 각 임베딩 모델별 RAG 체인 생성
rag_chain_openai = create_rag_chain(db_openai) if db_openai else None
rag_chain_huggingface = create_rag_chain(db_huggingface) if db_huggingface else None
```

**성능 평가 - 한국어 쿼리:**

```python
query_ko = "테슬라 창업자는 누구인가요?"

print("="*80)
print(f"질문: {query_ko}")
print("="*80)

# OpenAI
if rag_chain_openai:
    try:
        output_openai = rag_chain_openai.invoke(query_ko)
        print("✅ OpenAI:", output_openai)
    except Exception as e:
        print(f"❌ OpenAI 오류: {str(e)[:150]}")

# HuggingFace
if rag_chain_huggingface:
    try:
        output_hf = rag_chain_huggingface.invoke(query_ko)
        print("✅ HuggingFace:", output_hf)
    except Exception as e:
        print(f"❌ HuggingFace 오류: {str(e)[:150]}")
```

**예상 출력:**
```
================================================================================
질문: 테슬라 창업자는 누구인가요?
================================================================================
✅ OpenAI: Elon Musk
✅ HuggingFace: Elon Musk
```

**성능 평가 - 영어 쿼리:**

```python
query_en = "Who is the founder of Tesla?"

print("="*80)
print(f"Question: {query_en}")
print("="*80)

# OpenAI
if rag_chain_openai:
    output_openai = rag_chain_openai.invoke(query_en)
    print("✅ OpenAI:", output_openai)

# HuggingFace
if rag_chain_huggingface:
    output_hf = rag_chain_huggingface.invoke(query_en)
    print("✅ HuggingFace:", output_hf)
```

### 2단계: 언어 감지 및 번역 자동화

#### 2.1 언어 감지 라이브러리 설정

```python
from langdetect import detect

# 언어 감지 테스트
test_texts = {
    "한국어": "테슬라 창업자는 누구인가요?",
    "영어": "Who is the founder of Tesla?",
    "일본어": "テスラの創業者は誰ですか？",
    "중국어": "特斯拉的创始人是谁？"
}

for lang, text in test_texts.items():
    detected = detect(text)
    print(f"{lang}: {text} → 감지된 언어: {detected}")
```

**예상 출력:**
```
한국어: 테슬라 창업자는 누구인가요? → 감지된 언어: ko
영어: Who is the founder of Tesla? → 감지된 언어: en
일본어: テスラの創業者は誰ですか？ → 감지된 언어: ja
중국어: 特斯拉的创始人是谁？ → 감지된 언어: zh-cn
```

#### 2.2 LibreTranslate 번역 설정

```python
from langchain_community.tools import LibreTranslateAPI

# LibreTranslate 서버 연결 (로컬 또는 원격)
try:
    translator = LibreTranslateAPI(
        url="https://libretranslate.com/translate",  # 공개 서버
        # url="http://localhost:5000/translate",      # 로컬 서버
    )
    print("✅ LibreTranslate 서버 연결 성공")
except Exception as e:
    print(f"❌ LibreTranslate 서버 연결 실패: {e}")
    translator = None
```

**로컬 LibreTranslate 서버 실행 (선택사항):**
```bash
# Docker를 사용한 로컬 서버 실행
docker run -ti --rm -p 5000:5000 libretranslate/libretranslate
```

#### 2.3 언어 감지 기반 RAG 체인

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import chain

# 한국어 벡터 저장소 로드
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    collection_name="db_korean_cosine_metadata",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

if translator:
    # 벡터 저장소 retriever 생성
    retriever = vectorstore.as_retriever(search_kwargs={'k': 4})

    # RAG 체인 생성
    lang_rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 언어 감지 및 번역 기반 RAG 실행 함수
    @chain
    def run_lang_rag_chain(query):
        """
        1. 입력 쿼리의 언어 감지
        2. 한국어가 아닌 경우 한국어로 번역
        3. RAG 체인 실행
        4. 응답을 원래 언어로 번역
        """
        # 언어 감지
        original_lang = detect(query)
        print(f"감지된 언어: {original_lang}")

        # 한국어가 아닌 경우 번역
        if original_lang.upper() != 'KO':
            print(f"번역 중: {original_lang} → 한국어")
            query_ko = translator.run(
                query=query,
                source=original_lang,
                target='ko'
            )
            print(f"번역된 질문: {query_ko}")
        else:
            query_ko = query

        # RAG 체인 실행
        answer_ko = lang_rag_chain.invoke(query_ko)
        print(f"한국어 답변: {answer_ko}")

        # 원래 언어로 답변 번역
        if original_lang.upper() != 'KO':
            print(f"번역 중: 한국어 → {original_lang}")
            answer = translator.run(
                query=answer_ko,
                source='ko',
                target=original_lang
            )
            return answer
        else:
            return answer_ko
else:
    run_lang_rag_chain = None
    print("⚠️ LibreTranslate를 사용할 수 없어 언어 감지 RAG를 생성하지 않습니다.")
```

**실행 예시:**

```python
if run_lang_rag_chain:
    # 한국어 쿼리
    query_ko = "테슬라 창업자는 누구인가요?"
    output_ko = run_lang_rag_chain.invoke(query_ko)
    print(f"\n최종 답변: {output_ko}")

    # 영어 쿼리
    query_en = "Who is the founder of Tesla?"
    output_en = run_lang_rag_chain.invoke(query_en)
    print(f"\n최종 답변: {output_en}")

    # 일본어 쿼리
    query_ja = "テスラの創業者は誰ですか？"
    output_ja = run_lang_rag_chain.invoke(query_ja)
    print(f"\n最終答え: {output_ja}")
```

### 3단계: 언어별 벡터저장소 라우팅

#### 3.1 언어별 벡터 저장소 생성

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 한국어 벡터 저장소 로드
db_korean = Chroma(
    collection_name="db_korean_cosine_metadata",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
print(f"✅ 한국어 문서 수: {db_korean._collection.count()}")

# 영어 벡터 저장소 생성
db_english = Chroma.from_documents(
    documents=english_docs,
    embedding=embeddings,
    collection_name="eng_db_openai",
    persist_directory="./chroma_db"
)
print(f"✅ 영어 문서 수: {db_english._collection.count()}")
```

#### 3.2 언어 감지 기반 라우팅 RAG 체인

```python
from langdetect import detect
from langchain_core.runnables import chain

# 각 언어별 RAG 체인 생성
rag_chain_korean = create_rag_chain(db_korean)
rag_chain_english = create_rag_chain(db_english)

# 라우팅 RAG 체인
@chain
def run_route_rag_chain(query):
    """
    언어를 감지하고 해당 언어의 벡터 저장소를 사용하여 RAG 실행
    """
    # 언어 감지
    original_lang = detect(query)
    print(f"감지된 언어: {original_lang}")

    # 한국어인 경우 한국어 RAG 체인 실행
    if original_lang.upper() == 'KO':
        print("→ 한국어 벡터 저장소 사용")
        return rag_chain_korean.invoke(query)

    # 영어인 경우 영어 RAG 체인 실행
    elif 'EN' in original_lang.upper():
        print("→ 영어 벡터 저장소 사용")
        return rag_chain_english.invoke(query)

    # 지원하지 않는 언어
    else:
        return f"지원하지 않는 언어입니다: {original_lang}"
```

**실행 및 결과:**

```python
# 한국어 쿼리 테스트
query_ko = "테슬라 창업자는 누구인가요?"
output_ko = run_route_rag_chain.invoke(query_ko)
print(f"답변: {output_ko}\n")

# 영어 쿼리 테스트
query_en = "Who is the founder of Tesla?"
output_en = run_route_rag_chain.invoke(query_en)
print(f"Answer: {output_en}\n")
```

**예상 출력:**
```
감지된 언어: ko
→ 한국어 벡터 저장소 사용
답변: Elon Musk

감지된 언어: en
→ 영어 벡터 저장소 사용
Answer: Elon Musk
```

**라우팅 방식의 장점:**
- ✅ 언어별 최적화된 검색 성능
- ✅ 번역 없이 직접 검색으로 정확도 향상
- ✅ 각 언어별 독립적인 벡터 저장소 관리
- ✅ 확장 가능한 다국어 지원 (저장소 추가만으로 언어 추가 가능)

### 4단계: ReAct - 도구(Tool) 정의하기

#### 4.1 메타데이터 포함 RAG 체인 생성

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.chat_models import init_chat_model

def create_rag_chain_with_metadata(vectorstore, top_k=4):
    """
    메타데이터를 포함한 RAG 체인 생성
    - 검색된 문서의 출처(source) 정보를 함께 반환
    """
    template = """Answer the question based only on the following context.
Do not use any external information or knowledge.
If the answer is not in the context, answer "I don't know".

- For proper nouns (names of people, places, organizations), keep the original language.
- Provide the sources of information when available.

Context:
{context}

Question: {question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)
    llm = init_chat_model("openai:gpt-4.1-mini", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={'k': top_k})

    def format_docs_with_metadata(docs):
        """문서와 메타데이터를 함께 포맷"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content
            formatted.append(f"[Source {i}: {source}]\n{content}")
        return "\n\n".join(formatted)

    rag_chain = (
        {
            "context": retriever | format_docs_with_metadata,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

# 한국어 및 영어 RAG 체인 생성
rag_chain_korean = create_rag_chain_with_metadata(db_korean, top_k=4)
rag_chain_english = create_rag_chain_with_metadata(db_english, top_k=4)
```

**실행 테스트:**

```python
# 영어 RAG 체인 테스트
response = rag_chain_english.invoke({"question": "Who is the founder of Tesla?"})
print(response)
```

**예상 출력:**
```
Elon Musk is the founder of Tesla. [Source: data/tesla_EN.md]
```

#### 4.2 RAG 체인을 Tool 객체로 변환

```python
# 한국어 RAG 도구 생성
rag_tool_korean = rag_chain_korean.as_tool(
    name="rag_korean_db",
    description="한국어 질문에 대한 리비안, 테슬라 관련 문서를 벡터 저장소에서 검색하고, 그 결과와 함께 답변을 생성합니다."
)

print(f"Tool 이름: {rag_tool_korean.name}")
print(f"Tool 설명: {rag_tool_korean.description}")
print(f"Tool 입력 파라미터:")
from pprint import pprint
pprint(rag_tool_korean.args)
```

**출력:**
```
Tool 이름: rag_korean_db
Tool 설명: 한국어 질문에 대한 리비안, 테슬라 관련 문서를 벡터 저장소에서 검색하고, 그 결과와 함께 답변을 생성합니다.
Tool 입력 파라미터:
{'properties': {'question': {'title': 'Question', 'type': 'string'}},
 'required': ['question'],
 'title': 'rag_korean_db',
 'type': 'object'}
```

```python
# 영어 RAG 도구 생성
rag_tool_english = rag_chain_english.as_tool(
    name="rag_english_db",
    description="Retrieve and generate answers from the vector store for English questions related to Rivian and Tesla."
)

print(f"Tool 이름: {rag_tool_english.name}")
print(f"Tool 설명: {rag_tool_english.description}")
```

**Tool 객체의 구조:**
- `name`: 도구의 고유 식별자
- `description`: LLM이 도구를 선택할 때 참고하는 설명
- `args`: 도구의 입력 스키마 (Pydantic 모델로 정의)
- `invoke()`: 도구를 실행하는 메서드

### 5단계: ReAct - 도구(Tool) 호출하기

#### 5.1 LLM에 도구 바인딩

```python
from langchain.chat_models import init_chat_model

# 도구 목록
tools = [rag_tool_korean, rag_tool_english]

# LLM 모델 초기화
llm = init_chat_model("openai:gpt-4.1-mini", temperature=0)

# LLM에 도구 바인딩
llm_with_tools = llm.bind_tools(tools=tools)

# 한국어 질문으로 테스트
query = "테슬라 창업자는 누구인가요?"
response = llm_with_tools.invoke(query)

pprint(response)
```

**출력 (AIMessage 객체):**
```python
AIMessage(
    content='',
    additional_kwargs={
        'tool_calls': [
            {
                'id': 'call_abc123',
                'function': {
                    'arguments': '{"question":"테슬라 창업자는 누구인가요?"}',
                    'name': 'rag_korean_db'
                },
                'type': 'function'
            }
        ]
    },
    tool_calls=[
        {
            'name': 'rag_korean_db',
            'args': {'question': '테슬라 창업자는 누구인가요?'},
            'id': 'call_abc123'
        }
    ]
)
```

#### 5.2 ToolCall 객체 확인

```python
# ToolCall 리스트 확인
print("도구 호출 정보:")
pprint(response.tool_calls)
```

**출력:**
```python
[{
    'name': 'rag_korean_db',
    'args': {'question': '테슬라 창업자는 누구인가요?'},
    'id': 'call_abc123'
}]
```

**영어 질문 테스트:**

```python
query_en = "Who is the founder of Tesla?"
response_en = llm_with_tools.invoke(query_en)

print("도구 호출 정보:")
pprint(response_en.tool_calls)
```

**출력:**
```python
[{
    'name': 'rag_english_db',
    'args': {'question': 'Who is the founder of Tesla?'},
    'id': 'call_def456'
}]
```

**도구와 무관한 질문 테스트:**

```python
query_test = "오늘 날씨는 어떤가요?"
response_test = llm_with_tools.invoke(query_test)

print("Tool calls:", response_test.tool_calls)
print("Content:", response_test.content)
```

**출력:**
```python
Tool calls: []
Content: 죄송하지만 저는 날씨 정보에 접근할 수 없습니다. 날씨 정보를 확인하시려면 날씨 앱이나 웹사이트를 이용해주세요.
```

**주요 관찰:**
- ✅ LLM이 자동으로 적절한 도구 선택 (한국어 → rag_korean_db, 영어 → rag_english_db)
- ✅ 도구가 필요하지 않은 질문은 일반 응답 생성
- ✅ `tool_calls` 속성을 통해 어떤 도구를 호출할지 확인 가능

### 6단계: ReAct - 도구(Tool) 실행하기

#### 6.1 도구 매핑 생성

```python
# 도구 이름을 기준으로 도구 객체 매핑
tool_map = {
    "rag_korean_db": rag_tool_korean,
    "rag_english_db": rag_tool_english
}

# 도구 실행 테스트
result = tool_map["rag_korean_db"].invoke({"question": "테슬라 창업자는 누구인가요?"})
print(result)
```

**출력:**
```
Elon Musk [Source: data/tesla_KR.md]
```

#### 6.2 도구 호출 함수 정의

```python
from langchain_core.messages import AIMessage

def call_tools(msg: AIMessage):
    """
    AIMessage의 tool_calls를 실행하고 결과를 반환하는 헬퍼 함수

    Args:
        msg: AIMessage 객체 (tool_calls 속성 포함)

    Returns:
        실행 결과가 포함된 tool_calls 리스트
    """
    tool_calls = msg.tool_calls.copy()

    for tool_call in tool_calls:
        # 도구 이름으로 도구 객체 찾기
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        # 도구 실행
        tool_output = tool_map[tool_name].invoke(tool_args)

        # 결과를 tool_call에 추가
        tool_call["output"] = tool_output

    return tool_calls

# 도구 호출 함수 테스트
print("ToolCall 객체:")
pprint(response.tool_calls[0])
print("-" * 80)

tool_calls = call_tools(response)
print("\n실행 결과:")
pprint(tool_calls)
```

**출력:**
```
ToolCall 객체:
{'args': {'question': '테슬라 창업자는 누구인가요?'},
 'id': 'call_abc123',
 'name': 'rag_korean_db'}
--------------------------------------------------------------------------------

실행 결과:
[{'args': {'question': '테슬라 창업자는 누구인가요?'},
  'id': 'call_abc123',
  'name': 'rag_korean_db',
  'output': 'Elon Musk [Source: data/tesla_KR.md]'}]
```

#### 6.3 도구 호출 체인 생성

```python
# LLM과 도구 호출 함수를 파이프라인으로 연결
search_tool_chain = llm_with_tools | call_tools

# 한국어 쿼리 실행
query = "테슬라 창업자는 누구인가요?"
search_response = search_tool_chain.invoke(query)

pprint(search_response)
```

**출력:**
```python
[{'args': {'question': '테슬라 창업자는 누구인가요?'},
  'id': 'call_abc123',
  'name': 'rag_korean_db',
  'output': 'Elon Musk [Source: data/tesla_KR.md]'}]
```

**영어 쿼리 실행:**

```python
query = "Who is the founder of Tesla?"
search_response = search_tool_chain.invoke(query)

pprint(search_response)
```

**출력:**
```python
[{'args': {'question': 'Who is the founder of Tesla?'},
  'id': 'call_def456',
  'name': 'rag_english_db',
  'output': 'Elon Musk [Source: data/tesla_EN.md]'}]
```

**도구 실행 흐름:**

```
사용자 질문
    ↓
llm_with_tools.invoke()  ← LLM이 적절한 도구 선택
    ↓
AIMessage (tool_calls 포함)
    ↓
call_tools()  ← 도구 실제 실행
    ↓
실행 결과 (output 포함)
```

### 7단계: Agent 구현

#### 7.1 Agent 생성 및 실행

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# LLM 초기화
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# 도구 목록
tools = [rag_tool_korean, rag_tool_english]

# Agent 생성
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="당신은 사용자의 요청을 처리하는 AI Assistant입니다."
)

print("✅ Agent 생성 완료")
```

**Agent의 역할:**
- **자율적 의사결정**: 필요한 도구를 자동으로 선택하고 실행
- **계획-실행-관찰 순환**: ReAct 패턴 자동 관리
- **상태 관리**: 대화 히스토리 및 중간 결과 추적

#### 7.2 Agent 실행 - 한국어 쿼리

```python
# 한국어 쿼리로 Agent 실행
response = agent.invoke(
    {"messages": [{"role": "user", "content": "테슬라 창업자는 누구인가요?"}]}
)

# 결과 출력
pprint(response)
```

**출력 (축약):**
```python
{
    'messages': [
        HumanMessage(content='테슬라 창업자는 누구인가요?'),
        AIMessage(
            content='',
            tool_calls=[{
                'name': 'rag_korean_db',
                'args': {'question': '테슬라 창업자는 누구인가요?'},
                'id': 'call_abc123'
            }]
        ),
        ToolMessage(
            content='Elon Musk [Source: data/tesla_KR.md]',
            tool_call_id='call_abc123'
        ),
        AIMessage(
            content='테슬라의 창업자는 Elon Musk입니다.'
        )
    ]
}
```

**실행 흐름:**
1. **HumanMessage**: 사용자 질문
2. **AIMessage (tool_calls)**: Agent가 도구 선택
3. **ToolMessage**: 도구 실행 결과
4. **AIMessage (final)**: Agent의 최종 답변

#### 7.3 Agent 실행 - 영어 쿼리

```python
# 영어 쿼리로 Agent 실행
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Who is the founder of Tesla?"}]}
)

# 최종 답변만 추출
final_message = response["messages"][-1]
print("최종 답변:", final_message.content)
```

**출력:**
```
최종 답변: The founder of Tesla is Elon Musk.
```

**Agent vs 수동 Tool Calling:**

| 비교 항목 | Agent | 수동 Tool Calling |
|----------|-------|------------------|
| 도구 선택 | 자동 | 수동 지정 |
| 상태 관리 | 자동 | 수동 관리 |
| 반복 실행 | 자동 순환 | 명시적 반복 필요 |
| 에러 처리 | 내장 | 직접 구현 |
| 코드 복잡도 | 낮음 | 높음 |

## 🎯 실습 문제

### 실습 1: 다국어 RAG 시스템 확장 (⭐⭐⭐)

**문제:**
언어 감지 및 번역 자동화 방식의 다국어 RAG 시스템을 구현하세요.

**요구사항:**
1. 한국어, 영어, 중국어, 일본어 지원
2. 사용자 언어 감지 후 한국어 벡터 저장소에서 검색
3. 언어별 번역 도구를 별도로 구현
4. 최종 답변을 원래 언어로 반환

**힌트:**
- `langdetect`로 언어 감지
- `LibreTranslate`로 번역
- `@tool` 데코레이터로 번역 도구 정의
- Agent를 사용하여 자동 라우팅

### 실습 2: 메타데이터 기반 필터링 RAG (⭐⭐⭐⭐)

**문제:**
문서의 메타데이터(출처, 날짜, 카테고리 등)를 활용한 필터링 RAG 시스템을 구현하세요.

**요구사항:**
1. 문서에 메타데이터 추가 (source, date, category)
2. 메타데이터 기반 필터링 retriever 구현
3. 특정 출처나 날짜 범위의 문서만 검색하는 도구 생성
4. Agent가 질문에 따라 적절한 필터 자동 적용

**힌트:**
- `Document` 객체의 `metadata` 속성 활용
- `self_query` retriever 사용
- 필터 조건을 파라미터로 받는 도구 구현

### 실습 3: 멀티홉 질문 답변 시스템 (⭐⭐⭐⭐⭐)

**문제:**
여러 단계의 추론이 필요한 복잡한 질문에 답변하는 ReAct Agent를 구현하세요.

**요구사항:**
1. 첫 번째 도구로 기본 정보 검색
2. 검색된 정보를 바탕으로 두 번째 도구로 추가 정보 검색
3. 여러 검색 결과를 종합하여 최종 답변 생성
4. Agent가 자동으로 다단계 추론 수행

**예시 질문:**
- "테슬라와 리비안의 창업자를 각각 찾고, 두 사람의 공통점을 설명해주세요."
- "테슬라의 최신 모델과 가격을 찾고, 경쟁사와 비교해주세요."

**힌트:**
- 여러 개의 RAG 도구 정의 (회사별, 주제별)
- Agent의 `intermediate_steps` 확인
- `AgentExecutor`의 `max_iterations` 설정

### 실습 4: 실시간 데이터 통합 RAG (⭐⭐⭐⭐)

**문제:**
정적 문서와 실시간 API 데이터를 결합한 하이브리드 RAG 시스템을 구현하세요.

**요구사항:**
1. 벡터 저장소에서 기본 정보 검색하는 도구
2. 실시간 API에서 최신 정보 가져오는 도구 (예: 주식 가격, 날씨)
3. Agent가 질문에 따라 적절한 도구 조합 선택
4. 정적 정보와 실시간 정보를 통합한 답변 생성

**힌트:**
- `yfinance` API로 주식 정보 조회 도구 구현
- `@tool` 데코레이터로 API 래핑
- 여러 도구의 결과를 결합하는 로직 구현

## ✅ 솔루션 예시

### 실습 1 솔루션: 다국어 RAG 시스템

```python
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langdetect import detect
from typing import Literal

# LibreTranslate 연결 확인
if translator is None:
    print("⚠️ LibreTranslate 서버가 연결되지 않았습니다!")
else:
    # 1. 번역 도구 정의
    @tool
    def translate_to_korean(
        text: str,
        source_lang: Literal["en", "zh-cn", "ja"]
    ) -> str:
        """다국어 텍스트를 한국어로 번역합니다."""
        try:
            result = translator.run(
                query=text,
                source=source_lang,
                target='ko'
            )
            return result
        except Exception as e:
            return f"번역 실패: {str(e)}"

    @tool
    def translate_from_korean(
        text: str,
        target_lang: Literal["en", "zh-cn", "ja"]
    ) -> str:
        """한국어 텍스트를 다른 언어로 번역합니다."""
        try:
            result = translator.run(
                query=text,
                source='ko',
                target=target_lang
            )
            return result
        except Exception as e:
            return f"번역 실패: {str(e)}"

    # 2. 한국어 RAG 도구 (기존 사용)
    korean_rag_tool = rag_tool_korean

    # 3. Agent 생성
    multilang_tools = [
        translate_to_korean,
        translate_from_korean,
        korean_rag_tool
    ]

    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    multilang_agent = create_agent(
        model=llm,
        tools=multilang_tools,
        system_prompt="""당신은 다국어 질문에 답변하는 AI Assistant입니다.

작업 순서:
1. 사용자의 질문 언어를 확인합니다
2. 한국어가 아닌 경우, translate_to_korean 도구로 한국어로 번역합니다
3. rag_korean_db 도구로 한국어 벡터 저장소에서 검색합니다
4. 원래 언어가 한국어가 아닌 경우, translate_from_korean 도구로 원래 언어로 번역합니다
5. 최종 답변을 반환합니다"""
    )

    # 4. 테스트 실행
    test_queries = {
        "한국어": "테슬라 창업자는 누구인가요?",
        "영어": "Who is the founder of Tesla?",
        "중국어": "特斯拉的创始人是谁？",
        "일본어": "テスラの創業者は誰ですか？"
    }

    for lang, query in test_queries.items():
        print(f"\n{'='*80}")
        print(f"{lang} 질문: {query}")
        print('='*80)

        response = multilang_agent.invoke(
            {"messages": [{"role": "user", "content": query}]}
        )

        # 최종 답변 추출
        final_answer = response["messages"][-1].content
        print(f"답변: {final_answer}")
```

**예상 출력:**
```
================================================================================
한국어 질문: 테슬라 창업자는 누구인가요?
================================================================================
답변: 테슬라의 창업자는 Elon Musk입니다.

================================================================================
영어 질문: Who is the founder of Tesla?
================================================================================
답변: The founder of Tesla is Elon Musk.

================================================================================
중국어 질문: 特斯拉的创始人是谁？
================================================================================
답변: 特斯拉的创始人是Elon Musk。

================================================================================
일본어 질문: テスラの創業者は誰ですか？
================================================================================
답변: テスラの創業者はElon Muskです。
```

**작동 원리:**
1. Agent가 질문 언어 감지 (내부 추론)
2. 한국어가 아니면 `translate_to_korean` 도구 호출
3. `rag_korean_db` 도구로 한국어 벡터 저장소 검색
4. 원래 언어로 `translate_from_korean` 도구 호출
5. 최종 답변 반환

### 실습 2 솔루션: 메타데이터 기반 필터링 RAG

```python
from langchain.tools import tool
from langchain_core.documents import Document
from typing import Optional, List
from datetime import datetime

# 1. 메타데이터가 포함된 문서 생성
documents_with_metadata = [
    Document(
        page_content="Tesla was founded by Elon Musk in 2003.",
        metadata={"source": "tesla_history.md", "date": "2003-07-01", "category": "company"}
    ),
    Document(
        page_content="Tesla Model 3 was released in 2017.",
        metadata={"source": "tesla_products.md", "date": "2017-07-28", "category": "product"}
    ),
    Document(
        page_content="Rivian was founded by RJ Scaringe in 2009.",
        metadata={"source": "rivian_history.md", "date": "2009-06-01", "category": "company"}
    ),
    Document(
        page_content="Rivian R1T truck was announced in 2018.",
        metadata={"source": "rivian_products.md", "date": "2018-11-27", "category": "product"}
    )
]

# 2. 메타데이터 포함 벡터 저장소 생성
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

db_with_metadata = Chroma.from_documents(
    documents=documents_with_metadata,
    embedding=embeddings,
    collection_name="db_with_metadata",
    persist_directory="./chroma_db"
)

# 3. 메타데이터 필터링 도구 정의
@tool
def search_by_category(
    question: str,
    category: Literal["company", "product"]
) -> str:
    """특정 카테고리의 문서만 검색합니다."""
    retriever = db_with_metadata.as_retriever(
        search_kwargs={
            'k': 4,
            'filter': {'category': category}
        }
    )
    docs = retriever.invoke(question)

    if not docs:
        return f"'{category}' 카테고리에서 관련 문서를 찾을 수 없습니다."

    return "\n\n".join([
        f"[{doc.metadata['source']}] {doc.page_content}"
        for doc in docs
    ])

@tool
def search_by_source(
    question: str,
    source: str
) -> str:
    """특정 출처의 문서만 검색합니다."""
    retriever = db_with_metadata.as_retriever(
        search_kwargs={
            'k': 4,
            'filter': {'source': source}
        }
    )
    docs = retriever.invoke(question)

    if not docs:
        return f"'{source}' 출처에서 관련 문서를 찾을 수 없습니다."

    return "\n\n".join([doc.page_content for doc in docs])

@tool
def search_by_date_range(
    question: str,
    start_date: str,
    end_date: str
) -> str:
    """특정 날짜 범위의 문서만 검색합니다. (날짜 형식: YYYY-MM-DD)"""
    # 모든 문서 검색 후 날짜 필터링 (Chroma의 날짜 필터 제한 우회)
    retriever = db_with_metadata.as_retriever(search_kwargs={'k': 10})
    all_docs = retriever.invoke(question)

    # 날짜 범위 필터링
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    filtered_docs = []
    for doc in all_docs:
        doc_date = datetime.strptime(doc.metadata['date'], "%Y-%m-%d")
        if start <= doc_date <= end:
            filtered_docs.append(doc)

    if not filtered_docs:
        return f"{start_date}부터 {end_date} 사이에 관련 문서를 찾을 수 없습니다."

    return "\n\n".join([
        f"[{doc.metadata['date']}] {doc.page_content}"
        for doc in filtered_docs
    ])

# 4. Agent 생성
metadata_tools = [search_by_category, search_by_source, search_by_date_range]

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

metadata_agent = create_agent(
    model=llm,
    tools=metadata_tools,
    system_prompt="""당신은 메타데이터 기반 검색을 수행하는 AI Assistant입니다.

사용자의 질문을 분석하여 적절한 필터를 자동으로 적용합니다:
- 카테고리 언급 시: search_by_category 사용
- 출처 파일 언급 시: search_by_source 사용
- 날짜/기간 언급 시: search_by_date_range 사용
"""
)

# 5. 테스트 실행
test_queries = [
    "회사 정보만 검색해서 창업자를 알려주세요",
    "제품 정보만 검색해서 언제 출시되었는지 알려주세요",
    "2010년부터 2020년 사이의 정보를 검색해주세요"
]

for query in test_queries:
    print(f"\n{'='*80}")
    print(f"질문: {query}")
    print('='*80)

    response = metadata_agent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )

    final_answer = response["messages"][-1].content
    print(f"답변: {final_answer}")
```

**예상 출력:**
```
================================================================================
질문: 회사 정보만 검색해서 창업자를 알려주세요
================================================================================
답변: 회사 정보를 검색한 결과:
- Tesla의 창업자는 Elon Musk입니다 (2003년 설립)
- Rivian의 창업자는 RJ Scaringe입니다 (2009년 설립)

================================================================================
질문: 제품 정보만 검색해서 언제 출시되었는지 알려주세요
================================================================================
답변: 제품 출시 정보:
- Tesla Model 3: 2017년 7월 28일 출시
- Rivian R1T: 2018년 11월 27일 발표

================================================================================
질문: 2010년부터 2020년 사이의 정보를 검색해주세요
================================================================================
답변: 2010년부터 2020년 사이의 정보:
- 2017년: Tesla Model 3 출시
- 2018년: Rivian R1T 발표
```

### 실습 3 솔루션: 멀티홉 질문 답변 시스템

```python
from langchain.agents import create_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.tools import tool

# 1. 회사별 RAG 도구 정의
@tool
def search_tesla_info(question: str) -> str:
    """테슬라 관련 정보를 검색합니다."""
    # 테슬라 문서만 필터링
    retriever = db_with_metadata.as_retriever(
        search_kwargs={
            'k': 3,
            'filter': {'source': {'$regex': 'tesla'}}
        }
    )
    docs = retriever.invoke(question)
    return "\n".join([doc.page_content for doc in docs])

@tool
def search_rivian_info(question: str) -> str:
    """리비안 관련 정보를 검색합니다."""
    # 리비안 문서만 필터링
    retriever = db_with_metadata.as_retriever(
        search_kwargs={
            'k': 3,
            'filter': {'source': {'$regex': 'rivian'}}
        }
    )
    docs = retriever.invoke(question)
    return "\n".join([doc.page_content for doc in docs])

@tool
def compare_companies(company1_info: str, company2_info: str, aspect: str) -> str:
    """두 회사의 정보를 비교합니다."""
    prompt = f"""다음 두 회사의 {aspect}을(를) 비교하세요:

회사 1 정보:
{company1_info}

회사 2 정보:
{company2_info}

비교 결과를 구체적으로 작성해주세요."""

    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    result = llm.invoke(prompt)
    return result.content

# 2. Agent 생성 (max_iterations 설정)
multihop_tools = [search_tesla_info, search_rivian_info, compare_companies]

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

multihop_agent = create_agent(
    model=llm,
    tools=multihop_tools,
    system_prompt="""당신은 다단계 추론을 수행하는 AI Assistant입니다.

복잡한 질문의 처리 절차:
1. 질문을 하위 질문으로 분해합니다
2. 각 하위 질문에 대해 적절한 도구를 사용합니다
3. 수집한 정보를 바탕으로 비교나 종합을 수행합니다
4. 최종 답변을 생성합니다

여러 단계의 도구 호출이 필요할 수 있습니다."""
)

# 3. 멀티홉 질문 테스트
multihop_queries = [
    "테슬라와 리비안의 창업자를 각각 찾고, 두 사람의 공통점을 설명해주세요.",
    "테슬라와 리비안이 각각 언제 설립되었는지 찾고, 어느 회사가 더 오래되었는지 비교해주세요."
]

for query in multihop_queries:
    print(f"\n{'='*80}")
    print(f"질문: {query}")
    print('='*80)

    response = multihop_agent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )

    # 중간 단계 출력 (디버깅용)
    print("\n[중간 단계]")
    for i, msg in enumerate(response["messages"][1:-1], 1):
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            print(f"  단계 {i}: {msg.tool_calls[0]['name']} 호출")

    # 최종 답변
    final_answer = response["messages"][-1].content
    print(f"\n[최종 답변]\n{final_answer}")
```

**예상 출력:**
```
================================================================================
질문: 테슬라와 리비안의 창업자를 각각 찾고, 두 사람의 공통점을 설명해주세요.
================================================================================

[중간 단계]
  단계 1: search_tesla_info 호출
  단계 2: search_rivian_info 호출
  단계 3: compare_companies 호출

[최종 답변]
테슬라의 창업자는 Elon Musk이고, 리비안의 창업자는 RJ Scaringe입니다.

두 창업자의 공통점:
1. 전기차 산업의 선구자: 두 사람 모두 전기차 산업을 혁신하려는 비전을 가지고 있습니다.
2. 지속 가능성 추구: 환경 보호와 지속 가능한 교통수단 개발에 집중하고 있습니다.
3. 혁신적 접근: 기존 자동차 산업의 관행을 깨고 새로운 방식으로 접근합니다.
```

### 실습 4 솔루션: 실시간 데이터 통합 RAG

```python
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
import yfinance as yf
from datetime import datetime

# 1. 실시간 주식 정보 도구
@tool
def get_realtime_stock_price(symbol: str) -> str:
    """실시간 주식 가격 정보를 가져옵니다. (예: TSLA, RIVN)"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info

        current_price = info.get('currentPrice', 'N/A')
        market_cap = info.get('marketCap', 'N/A')
        pe_ratio = info.get('trailingPE', 'N/A')

        # 숫자 포맷팅
        if isinstance(market_cap, (int, float)):
            market_cap_b = market_cap / 1_000_000_000
            market_cap = f"${market_cap_b:.2f}B"

        result = f"""
{symbol} 실시간 주식 정보 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}):
- 현재가: ${current_price}
- 시가총액: {market_cap}
- P/E Ratio: {pe_ratio}
"""
        return result.strip()
    except Exception as e:
        return f"주식 정보 조회 실패: {str(e)}"

@tool
def get_stock_history(symbol: str, period: str = "1mo") -> str:
    """주식의 과거 가격 정보를 가져옵니다. period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max"""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)

        if hist.empty:
            return f"{symbol}의 {period} 기간 데이터가 없습니다."

        latest = hist.iloc[-1]
        earliest = hist.iloc[0]

        price_change = latest['Close'] - earliest['Close']
        price_change_pct = (price_change / earliest['Close']) * 100

        result = f"""
{symbol} 주식 {period} 기간 변화:
- 시작가: ${earliest['Close']:.2f}
- 종가: ${latest['Close']:.2f}
- 변화: ${price_change:.2f} ({price_change_pct:+.2f}%)
- 최고가: ${hist['High'].max():.2f}
- 최저가: ${hist['Low'].min():.2f}
"""
        return result.strip()
    except Exception as e:
        return f"주식 히스토리 조회 실패: {str(e)}"

# 2. 정적 문서 검색 도구 (기존)
static_rag_tool = rag_tool_english  # 영어 RAG 도구 재사용

# 3. 하이브리드 Agent 생성
hybrid_tools = [
    get_realtime_stock_price,
    get_stock_history,
    static_rag_tool
]

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

hybrid_agent = create_agent(
    model=llm,
    tools=hybrid_tools,
    system_prompt="""당신은 정적 문서와 실시간 데이터를 통합하여 답변하는 AI Assistant입니다.

질문에 따라 적절한 도구를 선택합니다:
- 회사 역사, 창업자 등 정적 정보: rag_english_db 사용
- 현재 주가, 시가총액 등 실시간 정보: get_realtime_stock_price 사용
- 주가 변화 추이: get_stock_history 사용

여러 도구를 조합하여 종합적인 답변을 제공합니다."""
)

# 4. 하이브리드 질문 테스트
hybrid_queries = [
    "테슬라의 창업자는 누구이며, 현재 주가는 얼마인가요?",
    "테슬라와 리비안의 현재 시가총액을 비교해주세요",
    "테슬라의 지난 1개월 주가 변화를 알려주세요"
]

for query in hybrid_queries:
    print(f"\n{'='*80}")
    print(f"질문: {query}")
    print('='*80)

    response = hybrid_agent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )

    final_answer = response["messages"][-1].content
    print(f"답변:\n{final_answer}")
```

**예상 출력:**
```
================================================================================
질문: 테슬라의 창업자는 누구이며, 현재 주가는 얼마인가요?
================================================================================
답변:
테슬라의 창업자는 Elon Musk입니다.

현재 주가 정보 (2025-10-25 14:30:00):
- 현재가: $242.50
- 시가총액: $771.23B
- P/E Ratio: 78.34

테슬라는 2003년에 설립되었으며, 전기차 시장을 선도하는 기업입니다.

================================================================================
질문: 테슬라와 리비안의 현재 시가총액을 비교해주세요
================================================================================
답변:
현재 시가총액 비교:

테슬라 (TSLA):
- 시가총액: $771.23B
- 현재가: $242.50

리비안 (RIVN):
- 시가총액: $12.45B
- 현재가: $11.23

테슬라의 시가총액이 리비안보다 약 62배 더 큽니다. 이는 테슬라가 전기차 시장에서
훨씬 더 큰 규모와 시장 지배력을 가지고 있음을 나타냅니다.
```

## 🚀 실무 활용 예시

### 예시 1: 고객 지원 챗봇 시스템

```python
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# 1. 도구 정의
@tool
def search_faq(question: str) -> str:
    """자주 묻는 질문(FAQ) 데이터베이스에서 검색합니다."""
    # FAQ 벡터 저장소에서 검색
    faq_retriever = faq_vectorstore.as_retriever(search_kwargs={'k': 3})
    docs = faq_retriever.invoke(question)

    if not docs:
        return "관련 FAQ를 찾을 수 없습니다."

    return "\n\n".join([
        f"Q: {doc.metadata['question']}\nA: {doc.page_content}"
        for doc in docs
    ])

@tool
def search_product_manual(product: str, question: str) -> str:
    """제품 매뉴얼에서 정보를 검색합니다."""
    # 제품별 매뉴얼 벡터 저장소에서 검색
    manual_retriever = product_manuals[product].as_retriever(search_kwargs={'k': 3})
    docs = manual_retriever.invoke(question)

    return "\n\n".join([doc.page_content for doc in docs])

@tool
def create_support_ticket(
    customer_name: str,
    issue_description: str,
    priority: Literal["low", "medium", "high"]
) -> str:
    """고객 지원 티켓을 생성합니다."""
    ticket_id = f"TICKET-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # 티켓 생성 로직 (데이터베이스 저장 등)
    ticket = {
        "ticket_id": ticket_id,
        "customer_name": customer_name,
        "issue": issue_description,
        "priority": priority,
        "status": "open",
        "created_at": datetime.now().isoformat()
    }

    return f"지원 티켓이 생성되었습니다. 티켓 번호: {ticket_id}"

# 2. 고객 지원 Agent 생성
support_tools = [search_faq, search_product_manual, create_support_ticket]

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

support_agent = create_agent(
    model=llm,
    tools=support_tools,
    system_prompt="""당신은 고객 지원 AI Assistant입니다.

고객 질문 처리 절차:
1. FAQ에서 유사한 질문 검색
2. FAQ에 없으면 제품 매뉴얼 검색
3. 해결되지 않으면 지원 티켓 생성 제안
4. 친절하고 명확한 답변 제공

고객의 만족을 최우선으로 합니다."""
)

# 3. 고객 대화 시뮬레이션
customer_queries = [
    "제품 A의 배터리 수명은 얼마나 되나요?",
    "제품 B가 켜지지 않습니다. 어떻게 해야 하나요?",
    "제품이 파손되어 교환하고 싶습니다."
]

for query in customer_queries:
    print(f"\n{'='*80}")
    print(f"고객: {query}")
    print('='*80)

    response = support_agent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )

    answer = response["messages"][-1].content
    print(f"상담원: {answer}")
```

### 예시 2: 법률 문서 분석 시스템

```python
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma

# 1. 법률 문서 도구
@tool
def search_legal_precedents(case_description: str, jurisdiction: str) -> str:
    """유사한 법적 판례를 검색합니다."""
    # 판례 벡터 저장소에서 검색 (관할 지역 필터링)
    precedent_retriever = legal_db.as_retriever(
        search_kwargs={
            'k': 5,
            'filter': {'jurisdiction': jurisdiction}
        }
    )
    docs = precedent_retriever.invoke(case_description)

    return "\n\n".join([
        f"판례 {i+1}: {doc.metadata['case_name']} ({doc.metadata['year']})\n{doc.page_content[:200]}..."
        for i, doc in enumerate(docs)
    ])

@tool
def search_statutes(legal_issue: str) -> str:
    """관련 법률 조항을 검색합니다."""
    statute_retriever = statute_db.as_retriever(search_kwargs={'k': 3})
    docs = statute_retriever.invoke(legal_issue)

    return "\n\n".join([
        f"{doc.metadata['statute_name']} 제{doc.metadata['article']}조:\n{doc.page_content}"
        for doc in docs
    ])

@tool
def analyze_contract_clause(clause_text: str) -> str:
    """계약서 조항을 분석하고 잠재적 리스크를 평가합니다."""
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    prompt = f"""다음 계약서 조항을 분석하세요:

조항:
{clause_text}

분석 항목:
1. 주요 내용 요약
2. 잠재적 법적 리스크
3. 수정 제안
4. 주의사항
"""

    result = llm.invoke(prompt)
    return result.content

# 2. 법률 분석 Agent
legal_tools = [search_legal_precedents, search_statutes, analyze_contract_clause]

llm = ChatOpenAI(model="gpt-4o", temperature=0)  # 더 강력한 모델 사용

legal_agent = create_agent(
    model=llm,
    tools=legal_tools,
    system_prompt="""당신은 법률 분석 AI Assistant입니다.

분석 절차:
1. 법률 이슈 파악
2. 관련 법률 조항 검색
3. 유사 판례 검색
4. 종합 분석 제공

주의: 법적 조언이 아닌 정보 제공 목적임을 명시합니다."""
)

# 3. 법률 분석 실행
legal_query = """
부동산 매매 계약서에 "매수인은 계약금 지급 후 7일 이내에 잔금을 지급하며,
지급하지 않을 경우 계약금을 포기하고 계약이 자동 해지된다"는 조항이 있습니다.
이 조항의 법적 유효성과 리스크를 분석해주세요.
"""

response = legal_agent.invoke(
    {"messages": [{"role": "user", "content": legal_query}]}
)

print(response["messages"][-1].content)
```

**예상 출력:**
```
[법률 분석 결과]

1. 관련 법률 조항:
   - 민법 제565조 (해제권의 행사)
   - 민법 제551조 (위약금의 약정)

2. 유사 판례:
   - 대법원 2015다12345: 계약금 포기 약정의 유효성 인정
   - 대법원 2018다67890: 과도한 위약금 조항의 제한

3. 조항 분석:
   - 7일 잔금 지급 기한은 매우 짧아 매수인에게 불리함
   - 계약금 포기 조항은 일반적으로 유효하나, 계약금이 과도한 경우 법원이 감액 가능
   - "자동 해지" 조항은 별도의 해지 통지 없이 해지되는 것으로 해석될 수 있음

4. 리스크 및 제안:
   - 리스크: 짧은 기한으로 인한 매수인의 채무불이행 가능성
   - 제안: 잔금 지급 기한을 최소 1개월로 연장
   - 제안: 자동 해지가 아닌 최고 절차 추가

**면책사항: 이는 정보 제공 목적이며 법적 조언이 아닙니다.**
```

## 📖 참고 자료

### 공식 문서
- [LangChain ReAct Agent 가이드](https://python.langchain.com/docs/modules/agents/agent_types/react)
- [LangChain Tools 문서](https://python.langchain.com/docs/modules/agents/tools/)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Chroma Vector Database](https://docs.trychroma.com/)

### 다국어 지원 관련
- [langdetect 라이브러리](https://pypi.org/project/langdetect/)
- [LibreTranslate API](https://libretranslate.com/)
- [OpenAI 다국어 임베딩](https://platform.openai.com/docs/guides/embeddings)

### ReAct 프레임워크
- [ReAct 논문 (Yao et al., 2022)](https://arxiv.org/abs/2210.03629)
- [Tool Use and Agent Development](https://www.anthropic.com/research/tool-use)

### 추가 학습 자료
- [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/)
- [Agent Executor 심화](https://python.langchain.com/docs/modules/agents/agent_executors/)
- [Custom Agent 구현](https://python.langchain.com/docs/modules/agents/how_to/custom_agent)
