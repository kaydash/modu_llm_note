# PRJ04_W2_006: 멀티모달 RAG 구현 (금융 분석 보고서) - Part 2

## 🎯 학습 목표

이 실습을 완료하면 다음을 수행할 수 있습니다:

1. **타입별 요약 체인 구현**: Table/Text/Image 각각에 대한 LLM 요약 체인을 구축할 수 있습니다
2. **MultiVectorRetriever 활용**: 요약과 원본을 분리 저장하는 벡터 스토어 시스템을 구현할 수 있습니다
3. **RAG 파이프라인 구축**: 검색-생성 전체 흐름을 LangChain으로 연결할 수 있습니다
4. **페이지 기반 요약**: 메타데이터의 page_number를 활용한 페이지별 요약 시스템을 설계할 수 있습니다
5. **실전 RAG 시스템**: 학술 논문(Transformer) RAG 시스템을 처음부터 구축할 수 있습니다

---

## 📚 핵심 개념

### 1. 타입별 요약 전략

| 타입 | 요약 방법 | 프롬프트 전략 | 출력 형식 |
|------|----------|-------------|----------|
| **Text** | 텍스트 기반 LLM | 핵심 정보 추출, 간결한 요약 | 한국어 요약 |
| **Table** | 텍스트 기반 LLM | 수치 데이터 유지, 패턴 강조 | 한국어 요약 |
| **Image** | 멀티모달 LLM | 차트/그래프 데이터 분석, 구조 설명 | 한국어 상세 설명 |

**핵심 차이점:**
- Text/Table: 텍스트만 입력 (`ChatOpenAI` + text prompt)
- Image: Base64 이미지 + 텍스트 입력 (`ChatOpenAI` + multimodal prompt)

### 2. MultiVectorRetriever 아키텍처

```
검색 요청
    ↓
[VectorStore: 요약 검색]
    ↓
관련 요약 발견 → doc_id 추출
    ↓
[DocStore: 원본 조회]
    ↓
doc_id로 원본 문서 반환
    ↓
LLM에 원본 전달
```

**장점:**
- 짧은 요약으로 빠른 검색
- 상세한 원본으로 정확한 답변 생성
- 메모리 효율적 (벡터 DB는 요약만 저장)

### 3. 옵션 2 RAG 파이프라인 전체 흐름

```
[1] PDF 파싱
    ↓
[2] 타입별 분류 (Text/Table/Image)
    ↓
[3] LLM 요약 생성
    ├─ Text/Table → TextLLM
    └─ Image → MultimodalLLM
    ↓
[4] MultiVectorRetriever 구축
    ├─ VectorStore: 요약 임베딩
    └─ DocStore: 원본 저장
    ↓
[5] 검색 & 답변 생성
    ├─ 질의 → 요약 검색 → 원본 조회
    └─ 원본 컨텍스트 → LLM → 최종 답변
```

---

## 📝 단계별 구현

### 1단계: 타입별 구분

Part 1에서 파싱한 `pdf_chunks`를 타입별로 분류:

```python
# 타입별 분류
tables = []
texts = []
images = []

for chunk in pdf_chunks:
    # 테이블 청크
    if "Table" in str(type(chunk)):
        tables.append(chunk)

    # 텍스트 청크 (CompositeElement)
    elif "CompositeElement" in str(type(chunk)):
        texts.append(chunk)

# 이미지 추출 (metadata에서)
for chunk in pdf_chunks:
    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
        for el in chunk.metadata.orig_elements:
            # 이미지 요소가 있는 경우만 추출
            if hasattr(el.metadata, 'image_base64') and el.metadata.image_base64:
                images.append(el.metadata.image_base64)

print(f"📊 테이블 청크: {len(tables)}개")
print(f"📝 텍스트 청크: {len(texts)}개")
print(f"🖼️ 이미지: {len(images)}개")
```

**확인:**
```python
# 테이블 청크 확인
tables[0]

# 텍스트 청크 확인
texts[0]

# 이미지 청크 확인
plt_img_base64(images[0])
```

### 2단계: Table/Text 요약 체인

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 프롬프트 템플릿 설정
prompt_text = """
다음 텍스트나 표를 한국어로 간결하게 요약하세요.
핵심 정보와 수치 데이터를 포함하되, 불필요한 설명은 제외하세요.

요약할 내용:
{element}

요약:
"""
prompt = ChatPromptTemplate.from_template(prompt_text)

# 요약 체인 생성
model = ChatOpenAI(temperature=0, model="gpt-4o-mini")
summarize_chain = prompt | model | StrOutputParser()
```

**단일 요약 테스트:**
```python
# 텍스트 요약 테스트
text_summary = summarize_chain.invoke(texts[0])
print(text_summary)

# 테이블 요약 테스트
table_summary = summarize_chain.invoke(tables[0])
print(table_summary)
```

**배치 요약 생성:**
```python
# 텍스트 요약 (병렬 처리)
text_summaries = summarize_chain.batch(
    texts,
    {"max_concurrency": 3}     # 동시 실행 개수 설정
)

# 테이블 요약 (병렬 처리)
table_summaries = summarize_chain.batch(
    tables,
    {"max_concurrency": 3}     # 동시 실행 개수 설정
)

print(f"Text 요약 개수: {len(text_summaries)}")
print(f"Table 요약 개수: {len(table_summaries)}")
```

**핵심 파라미터:**
- `max_concurrency=3`: 최대 3개 요청 동시 실행 (API rate limit 고려)
- `temperature=0`: 결정론적 요약 생성

### 3단계: 이미지 캡션 요약 체인

```python
from langchain_openai import ChatOpenAI

# 이미지 요약 프롬프트 템플릿 설정
messages = [
    (
        "user",
        [
            {
                "type": "text",
                "text": """
                이 이미지를 자세히 분석하고 한국어로 설명하세요.
                특히 다음 사항에 주의하세요:

                1. 차트나 그래프의 경우: 데이터 트렌드, 수치, 비교 포인트
                2. 다이어그램의 경우: 구조, 연결관계, 프로세스 흐름
                3. 표의 경우: 주요 데이터 포인트와 패턴
                4. 일반 이미지의 경우: 주요 객체, 텍스트, 맥락

                분석 결과를 구체적이고 정확하게 기술하세요.
                """
            },
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,{image}"},
            },
        ],
    )
]

prompt = ChatPromptTemplate.from_messages(messages)

# 멀티모달 LLM 체인
image_chain = prompt | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()

# 이미지 요약 테스트
image_summary = image_chain.invoke(images[0])
print(image_summary)
```

**배치 이미지 요약:**
```python
# 이미지 요약 (병렬 처리)
image_summaries = image_chain.batch(
    images,
    {"max_concurrency": 3}
)

print(f"Image 요약 개수: {len(image_summaries)}")
```

**요약 결과 저장:**
```python
# 추출한 이미지와 요약 결과를 저장할 폴더 경로
output_dir = "data/analyst_reports/summaries"
os.makedirs(output_dir, exist_ok=True)

# 요약 결과를 JSON 파일로 저장
summary_data = {
    "text_summaries": text_summaries,
    "table_summaries": table_summaries,
    "image_summaries": image_summaries
}
with open(os.path.join(output_dir, "summaries.json"), "w", encoding="utf-8") as f:
    json.dump(summary_data, f, ensure_ascii=False, indent=4)

print(f"💾 요약 저장 완료: {os.path.join(output_dir, 'summaries.json')}")
```

### 4단계: 벡터스토어 구축

#### 4-1. 저장/로드 유틸리티 함수

```python
from langchain_core.stores import InMemoryStore
import pickle

def save_store_to_disk(store: InMemoryStore, path: str):
    """InMemoryStore를 디스크에 저장"""
    data = dict(store.store)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"💾 Store 저장: {path}")


def load_store_from_disk(path: str):
    """디스크에서 InMemoryStore 로드"""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    new_store = InMemoryStore()
    new_store.store = data
    return new_store
```

#### 4-2. MultiVectorRetriever 초기화

```python
import uuid

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.stores import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# 자식 청크를 인덱싱하기 위한 벡터 저장소
vectorstore = Chroma(
    collection_name="mm_summaries",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="./chroma_db",  # 벡터 저장소 경로
)

# 부모 문서를 위한 저장 레이어
store = InMemoryStore()
id_key = "doc_id"

# 검색기 생성
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)
```

**기존 컬렉션 삭제 (필요시):**
```python
# retriever.vectorstore.delete_collection()  # 기존 컬렉션 삭제
```

#### 4-3. 텍스트 추가

```python
### 텍스트 추가 ###

# 각 텍스트에 대한 고유 ID 생성
doc_ids = [str(uuid.uuid4()) for _ in texts]

# 텍스트 요약 문서 생성 (검색용)
summary_texts = [
    Document(page_content=s, metadata={id_key: doc_ids[i], "source": "text"})
    for i, s in enumerate(text_summaries)
]

# 원본 텍스트 문서 생성 (답변 생성용)
original_texts = [
    Document(page_content=t.text, metadata={id_key: doc_ids[i], "source": "text"})
    for i, t in enumerate(texts)
]

# 벡터 저장소에 텍스트 요약 추가
retriever.vectorstore.add_documents(summary_texts)

# 문서 저장소에 원본 텍스트 추가
retriever.docstore.mset(list(zip(doc_ids, original_texts)))

print(f"📄 텍스트 요약 추가: {len(summary_texts)}개")
```

**검색 테스트:**
```python
# 검색 테스트
query = "2023년 2분기 매출은 얼마인가요?"
docs = retriever.invoke(query)
print(f"🔍 검색 결과 개수: {len(docs)}개")

for doc in docs:
    print(f"ID: {doc.metadata[id_key]}")
    print(f"Source: {doc.metadata.get('source', 'unknown')}")
    print(f"Content: {doc.page_content[:200]}...")  # 내용 일부만 출력
    print("=" * 50)
```

#### 4-4. 테이블 추가

```python
### 테이블 요약 추가 ###

# 각 테이블에 대한 고유 ID 생성
table_ids = [str(uuid.uuid4()) for _ in table_summaries]

# 테이블 요약 문서 생성
summary_tables = [
    Document(page_content=s, metadata={id_key: table_ids[i], "source": "table"})
    for i, s in enumerate(table_summaries)
]

# 원본 테이블 문서 생성 (metadata에서 HTML 추출)
original_tables = [
    Document(
        page_content=t.metadata.orig_elements[0].metadata.text_as_html,
        metadata={id_key: table_ids[i], "source": "table"}
    )
    for i, t in enumerate(tables)
]

# 벡터 저장소에 테이블 요약 추가
retriever.vectorstore.add_documents(summary_tables)

# 문서 저장소에 원본 테이블 추가
retriever.docstore.mset(list(zip(table_ids, original_tables)))

print(f"📄 테이블 요약 추가: {len(summary_tables)}개")
```

**검색 테스트:**
```python
# 검색 테스트
query = "2023년 2분기 매출은 얼마인가요?"
docs = retriever.invoke(query)
print(f"🔍 검색 결과 개수: {len(docs)}개")

for doc in docs:
    print(f"ID: {doc.metadata[id_key]}")
    print(f"Source: {doc.metadata.get('source', 'unknown')}")
    print(f"Content: {doc.page_content[:200]}...")
    print("=" * 50)
```

#### 4-5. 이미지 요약 추가

```python
### 이미지 요약 추가 ###

# 각 이미지에 대한 고유 ID 생성
img_ids = [str(uuid.uuid4()) for _ in image_summaries]

# 이미지 요약 문서 생성
summary_images = [
    Document(page_content=s, metadata={id_key: img_ids[i], "source": "image"})
    for i, s in enumerate(image_summaries)
]

# 벡터 저장소에 이미지 요약 추가
retriever.vectorstore.add_documents(summary_images)

# 문서 저장소에 이미지 요약 추가
retriever.docstore.mset(list(zip(img_ids, summary_images)))

print(f"📄 이미지 요약 추가: {len(summary_images)}개")
```

**검색 테스트:**
```python
# 검색 테스트
query = "삼성전기의 과거 2년간 투자 의견은 무엇인가요?"
docs = retriever.invoke(query)
print(f"🔍 검색 결과 개수: {len(docs)}개")

for doc in docs:
    print(f"ID: {doc.metadata[id_key]}")
    print(f"Source: {doc.metadata.get('source', 'unknown')}")
    print(f"Content: {doc.page_content[:200]}...")
    print("=" * 50)
```

#### 4-6. Store 저장 및 로드

```python
# store 저장
save_store_to_disk(store, "mm_summaries.pkl")

# store 로드
loaded_store = load_store_from_disk("mm_summaries.pkl")

# 벡터 저장소 로드
vectorstore = Chroma(
    collection_name="mm_summaries",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="./chroma_db",  # 벡터 저장소 경로
)

# 로드한 저장소로 새 검색기 만들기
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=loaded_store,
    id_key=id_key,
)

# 벡터 스토어에서 문서 검색
docs = retriever.invoke("삼성전기의 2024년 실적은 어떻게 전망하고 있나요?")
print(f"검색된 문서 개수: {len(docs)}")

for doc in docs:
    print(f"ID: {doc.metadata[id_key]}")
    print(f"Source: {doc.metadata.get('source', 'unknown')}")
    print(f"Content: {doc.page_content[:200]}...")
    print("=" * 50)
```

### 5단계: RAG 파이프라인

#### 5-1. 프롬프트 및 체인 구성

```python
from operator import itemgetter

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

# 프롬프트 템플릿
template = """Answer the question based only on the following context, which can include text and tables:

<Context>
{context}
</Context>

<Question>
{question}
</Question>

Answer in 한국어:
"""
prompt = ChatPromptTemplate.from_template(template)

# 텍스트 기반 LLM
model = ChatOpenAI(temperature=0, model="gpt-4o-mini")

# RAG 파이프라인 구성
chain = (
    {"context": retriever, "question": RunnablePassthrough()}  # retriever로 컨텍스트 검색 및 질문 그대로 전달
    | prompt  # 프롬프트 템플릿에 컨텍스트와 질문 통합
    | model  # LLM을 통한 응답 생성
    | StrOutputParser()  # 출력을 문자열로 변환
)
```

**파이프라인 구조:**
1. `retriever`: 질문으로 관련 문서 검색
2. `prompt`: 검색된 문서와 질문을 템플릿에 삽입
3. `model`: LLM이 컨텍스트 기반 답변 생성
4. `StrOutputParser`: 문자열 형식으로 변환

#### 5-2. RAG 체인 실행

```python
# chain 실행
result = chain.invoke("삼성전기의 2024년 실적은 어떻게 전망하고 있나요?")

# 결과 출력
print(result)
```

**출력 예시:**
```
삼성전기의 2024년 실적은 반도체 업황 회복과 함께 개선될 것으로 전망됩니다.
특히 MLCC 수요 증가와 FC-BGA 기판 사업 확대로 영업이익률이 전년 대비
2.3%p 상승한 8.5%를 기록할 것으로 예상됩니다.
```

```python
# chain 실행
result = chain.invoke("삼성전기의 과거 2년간 투자 의견은 무엇인가요?")

# 결과 출력
print(result)
```

**특징:**
- **텍스트만 활용**: 이미지 정보는 텍스트 요약으로 변환되어 검색
- **컨텍스트 제한**: 검색된 청크만 사용 (hallucination 방지)
- **한국어 답변**: 프롬프트에서 한국어 응답 지정

---

## 🎯 실습 문제

### 실습 1 (기본): metadata 기반 페이지별 요약 시스템

**문제:**
`metadata.page_number`를 활용하여 페이지별로 요약을 생성하고, 이를 기반으로 질문에 답변하는 시스템을 구축하세요.

**요구사항:**
1. 각 청크에서 `page_number` 추출
2. 페이지별로 텍스트/테이블/이미지 그룹화
3. 페이지 단위로 통합 요약 생성
4. 페이지 요약을 벡터 DB에 저장
5. 페이지 기반 검색 및 답변

**힌트:**
```python
from collections import defaultdict

# 페이지별 분류
page_data = defaultdict(lambda: {
    "texts": [],
    "tables": [],
    "images": [],
    "doc_name": "",
    "page_num": -1
})

for chunk in all_chunks:
    page_num = chunk.metadata.page_number
    # 페이지별로 분류...
```

### 실습 2 (중급): Transformer 논문 멀티모달 RAG

**문제:**
`data/transformer.pdf` 파일에 대한 완전한 멀티모달 RAG 시스템을 구현하세요 (옵션 2 방식).

**요구사항:**
1. PDF 파싱 (hi_res 전략)
2. 타입별 분류 (Text/Table/Image)
3. 영어 요약 생성 (학술 논문 스타일)
4. MultiVectorRetriever 구축
5. RAG 체인 구현
6. 테스트 질문 3개로 검증

**테스트 질문:**
- "What is the Transformer architecture?"
- "What is self-attention mechanism?"
- "What are the key components of the Transformer model?"

### 실습 3 (고급): 하이브리드 RAG 시스템

**문제:**
옵션 2 방식을 개선하여 중요 이미지는 base64로 직접 LLM에 전달하는 하이브리드 시스템을 구현하세요.

**요구사항:**
1. 이미지 중요도 점수 계산 (크기, 포함된 텍스트 양 등)
2. 상위 20% 이미지는 base64 직접 활용
3. 나머지 80% 이미지는 텍스트 요약 활용
4. 검색 시 이미지 타입에 따라 다른 처리
5. 비용과 품질 트레이드오프 분석

```python
class HybridMultimodalRAG:
    def __init__(self, chunks, importance_threshold=0.8):
        self.chunks = chunks
        self.threshold = importance_threshold

    def calculate_importance(self, image_base64: str) -> float:
        """이미지 중요도 계산"""
        pass

    def classify_images(self) -> dict:
        """중요 이미지 vs 일반 이미지 분류"""
        pass

    def build_retriever(self) -> MultiVectorRetriever:
        """하이브리드 retriever 구축"""
        pass

    def generate_answer(self, query: str) -> dict:
        """이미지 타입에 따른 답변 생성"""
        pass
```

---

## 💡 솔루션 예시

### 솔루션 1: metadata 기반 페이지별 요약 시스템

```python
from collections import defaultdict
import os
import uuid
import json
from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.stores import InMemoryStore
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever

# 1. PDF 로드 및 파티셔닝
data_path = "data/analyst_reports/"
pdf_files = glob(os.path.join(data_path, "*.pdf"))

all_chunks = []
for pdf_file in pdf_files:
    print(f"   처리 중: {os.path.basename(pdf_file)}")

    chunks = partition_pdf(
        filename=pdf_file,
        strategy="hi_res",
        infer_table_structure=True,
        languages=["eng", "kor"],
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=True,
    )
    all_chunks.extend(chunks)

print(f"총 청크 개수: {len(all_chunks)}개")

# 2. 페이지별 분류 (문서별 + 페이지별)
page_data = defaultdict(lambda: {
    "texts": [],
    "tables": [],
    "images": [],
    "doc_name": "",
    "page_num": -1
})

# 페이지별 통합 요소 카운터
page_element_counters = defaultdict(int)

for chunk_idx, chunk in enumerate(all_chunks):
    doc_name = chunk.metadata.filename.split(".")[0] if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'filename') else "unknown"
    page_num = int(chunk.metadata.page_number) if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'page_number') else -1

    # 청크 키 생성
    chunk_key = f"{doc_name}*page*{page_num}"

    # 문서명과 페이지 번호 정보 저장
    page_data[chunk_key]["doc_name"] = doc_name
    page_data[chunk_key]["page_num"] = page_num

    # 해당 페이지에서의 통합 요소 순서
    element_idx = page_element_counters[chunk_key]

    # 청크 타입별 분류 및 인덱스 추가
    if "Table" in str(type(chunk)):
        page_data[chunk_key]["tables"].append({
            "content": chunk.metadata.text_as_html if hasattr(chunk.metadata, 'text_as_html') else chunk.text,
            "page_index": element_idx,
            "global_index": chunk_idx
        })
    elif "Image" in str(type(chunk)):
        if hasattr(chunk.metadata, 'image_base64') and chunk.metadata.image_base64:
            page_data[chunk_key]["images"].append({
                "content": chunk.metadata.image_base64,
                "page_index": element_idx,
                "global_index": chunk_idx
            })
    else:
        page_data[chunk_key]["texts"].append({
            "content": chunk.text if hasattr(chunk, 'text') else str(chunk),
            "page_index": element_idx,
            "global_index": chunk_idx
        })

    # 페이지 내 통합 요소 카운터 증가
    page_element_counters[chunk_key] += 1

# 문서별로 그룹화
doc_groups = defaultdict(list)
for chunk_key, data in page_data.items():
    doc_name = data["doc_name"]
    doc_groups[doc_name].append((chunk_key, data))

# 3. 페이지별 요약 생성
# 텍스트/테이블 요약 체인
text_prompt = """다음 내용을 한국어로 간결하게 요약하세요:

<Source>
문서: {doc_name}, 페이지: {page_num}
</Source>

<Content>
{content}
</Content>
"""
text_chain = (
    ChatPromptTemplate.from_template(text_prompt)
    | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    | StrOutputParser()
)

# 이미지 캡션 체인
image_messages = [
    ("user", [
        {
            "type": "text",
            "text": """이 이미지를 한국어로 상세히 설명하세요.
문서: {doc_name}, 페이지: {page_num}
차트, 그래프, 표, 다이어그램의 경우 핵심 데이터와 트렌드를 포함하세요."""
        },
        {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,{image}"}
        }
    ])
]
image_chain = (
    ChatPromptTemplate.from_messages(image_messages)
    | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    | StrOutputParser()
)

# 각 문서별로 페이지 요약 생성
page_summaries = {}
for doc_name, doc_pages in doc_groups.items():
    print(f"\n📄 문서 '{doc_name}' 처리 중...")

    # 페이지 번호로 정렬
    doc_pages.sort(key=lambda x: x[1]["page_num"])

    for chunk_key, data in doc_pages:
        page_num = data["page_num"]
        print(f"  페이지 {page_num} 처리 중...")

        # 페이지 내 모든 요소를 순서대로 정렬
        all_elements = []

        # 텍스트 요소들 추가
        for text_item in data["texts"]:
            all_elements.append({
                'type': 'text',
                'content': text_item['content'],
                'page_index': text_item['page_index'],
                'global_index': text_item['global_index']
            })

        # 테이블 요소들 추가
        for table_item in data["tables"]:
            all_elements.append({
                'type': 'table',
                'content': table_item['content'],
                'page_index': table_item['page_index'],
                'global_index': table_item['global_index']
            })

        # 이미지 요소들 추가
        for image_item in data["images"]:
            all_elements.append({
                'type': 'image',
                'content': image_item['content'],
                'page_index': image_item['page_index'],
                'global_index': image_item['global_index']
            })

        # 페이지 내 순서로 정렬
        all_elements.sort(key=lambda x: x['page_index'])

        # 순서대로 요약 생성
        page_content = []

        for element in all_elements:
            element_type = element['type']
            page_idx = element['page_index']

            try:
                if element_type == 'text':
                    text_summary = text_chain.invoke({
                        "doc_name": doc_name,
                        "page_num": page_num,
                        "content": element['content']
                    })
                    page_content.append(f"[{page_idx}] 텍스트: {text_summary}")

                elif element_type == 'table':
                    table_summary = text_chain.invoke({
                        "doc_name": doc_name,
                        "page_num": page_num,
                        "content": element['content']
                    })
                    page_content.append(f"[{page_idx}] 표: {table_summary}")

                elif element_type == 'image':
                    image_summary = image_chain.invoke({
                        "doc_name": doc_name,
                        "page_num": page_num,
                        "image": element['content']
                    })
                    page_content.append(f"[{page_idx}] 이미지: {image_summary}")
            except Exception as e:
                print(f"      요약 실패 ({element_type}, 순서 {page_idx}): {e}")

        # 페이지 전체 요약
        if page_content:
            full_content = "\n\n".join(page_content)
            page_summary = f"문서: {doc_name}, 페이지 {page_num}:\n{full_content}"
            page_summaries[chunk_key] = page_summary
        else:
            page_summaries[chunk_key] = f"문서: {doc_name}, 페이지 {page_num}: 내용 없음"

print(f"\n✅ 총 {len(page_summaries)}개 페이지 요약 완료")

# 4. 요약 결과 저장
summary_file = os.path.join(data_path, "page_summaries.json")
with open(summary_file, "w", encoding="utf-8") as f:
    json.dump(page_summaries, f, ensure_ascii=False, indent=2)
print(f"📂 페이지 요약 결과 저장: {summary_file}")

# 5. 벡터 스토어 구축
vectorstore = Chroma(
    collection_name="page_summaries",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="./chroma_db"
)

store = InMemoryStore()
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key="doc_id"
)

# 페이지 요약을 벡터 스토어에 추가
for chunk_key, summary in page_summaries.items():
    doc_id = str(uuid.uuid4())
    doc_name = page_data[chunk_key]["doc_name"]
    page_num = page_data[chunk_key]["page_num"]

    summary_doc = Document(
        page_content=summary,
        metadata={
            "doc_id": doc_id,
            "chunk_key": chunk_key,
            "doc_name": doc_name,
            "page_number": page_num,
            "source": "page_summary",
        }
    )

    retriever.vectorstore.add_documents([summary_doc])
    retriever.docstore.mset([(doc_id, summary_doc)])

print(f"✅ {len(page_summaries)}개 페이지 요약 벡터화 완료")

# 6. 검색 테스트
docs = retriever.invoke("삼성전기의 2024년 실적은 어떻게 전망하고 있나요?")
print(f"\n🔍 검색된 문서 개수: {len(docs)}")

for doc in docs[:3]:
    print(f"\n문서명: {doc.metadata.get('doc_name', 'unknown')}")
    print(f"페이지: {doc.metadata.get('page_number', 'unknown')}")
    print(f"내용: {doc.page_content[:300]}...")
    print("=" * 50)
```

**출력 예시:**
```
총 청크 개수: 156개

📄 문서 '삼성전기_2024_1Q' 처리 중...
  페이지 1 처리 중...
  페이지 2 처리 중...
...

✅ 총 45개 페이지 요약 완료
📂 페이지 요약 결과 저장: data/analyst_reports/page_summaries.json
✅ 45개 페이지 요약 벡터화 완료

🔍 검색된 문서 개수: 4

문서명: 삼성전기_2024_1Q
페이지: 5
내용: [0] 텍스트: 2024년 1분기 실적 전망 - 매출 2.8조원(+12% YoY), 영업이익 2,400억원(+45% YoY)

[1] 표: 2024년 전체 실적 전망 - 매출 12.5조원, 영업이익 1.1조원, 영업이익률 8.8%...
==================================================
```

### 솔루션 2: Transformer 논문 멀티모달 RAG

완전한 Transformer 논문 RAG 시스템 (앞서 제공된 실습 2 코드 참조):

```python
# 실습: Transformer 논문 멀티모달 RAG (옵션 2 방식)
import os
import uuid
from glob import glob
from dotenv import load_dotenv
from collections import defaultdict

from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.stores import InMemoryStore
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever

# 1. 환경 설정
load_dotenv()

# 2. PDF 파싱
pdf_path = "data/transformer.pdf"
print(f"Processing {pdf_path}...")

transformer_chunks = partition_pdf(
    filename=pdf_path,
    strategy="hi_res",
    infer_table_structure=True,
    languages=["eng"],
    extract_image_block_types=["Image", "Table"],
    extract_image_block_to_payload=True,
    chunking_strategy="by_title",
    max_characters=1500,
    new_after_n_chars=1200,
    combine_text_under_n_chars=800,
)
print(f"Extracted {len(transformer_chunks)} chunks.")

# 3. 타입별 분류
tables = []
texts = []
images = []

for chunk in transformer_chunks:
    if "Table" in str(type(chunk)):
        tables.append(chunk)
    elif "CompositeElement" in str(type(chunk)):
        texts.append(chunk)

# 이미지 추출
for chunk in transformer_chunks:
    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
        for el in chunk.metadata.orig_elements:
            if hasattr(el, 'metadata') and hasattr(el.metadata, 'image_base64') and el.metadata.image_base64:
                images.append(el.metadata.image_base64)

print(f"📊 테이블: {len(tables)}개")
print(f"📝 텍스트: {len(texts)}개")
print(f"🖼️ 이미지: {len(images)}개")

# 4. 요약 체인 설정
text_prompt = """Summarize the following content concisely in English.
Include key information and numerical data:

{element}

Summary:
"""
text_summarize_chain = (
    ChatPromptTemplate.from_template(text_prompt)
    | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    | StrOutputParser()
)

image_messages = [
    ("user", [
        {"type": "text", "text": """Analyze this image and describe it in detail.
For charts/graphs: data trends, numbers, comparisons.
For diagrams: structure, connections, process flow."""},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image}"}}
    ])
]
image_summarize_chain = (
    ChatPromptTemplate.from_messages(image_messages)
    | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    | StrOutputParser()
)

# 5. 요약 생성
print("\n텍스트 요약 생성 중...")
text_summaries = text_summarize_chain.batch(
    [{"element": t.text} for t in texts],
    {"max_concurrency": 3}
)
print(f"텍스트 요약 완료: {len(text_summaries)}개")

if tables:
    print("테이블 요약 생성 중...")
    table_summaries = text_summarize_chain.batch(
        [{"element": t.text if hasattr(t, 'text') else str(t)} for t in tables],
        {"max_concurrency": 3}
    )
    print(f"테이블 요약 완료: {len(table_summaries)}개")
else:
    table_summaries = []

if images:
    print("이미지 요약 생성 중... (처음 10개만)")
    image_summaries = image_summarize_chain.batch(
        [{"image": img} for img in images[:10]],
        {"max_concurrency": 3}
    )
    print(f"이미지 요약 완료: {len(image_summaries)}개")
else:
    image_summaries = []

# 6. 벡터 스토어 구축
print("\n벡터 스토어 구축 중...")
vectorstore = Chroma(
    collection_name="transformer_summaries",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
)

store = InMemoryStore()
id_key = "doc_id"

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)

# 텍스트 추가
doc_ids = [str(uuid.uuid4()) for _ in texts]
summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i], "source": "text"})
    for i, s in enumerate(text_summaries)
]
original_docs = [
    Document(page_content=t.text, metadata={id_key: doc_ids[i], "source": "text"})
    for i, t in enumerate(texts)
]
retriever.vectorstore.add_documents(summary_docs)
retriever.docstore.mset(list(zip(doc_ids, original_docs)))
print(f"📄 텍스트 추가: {len(summary_docs)}개")

# 테이블 추가
if table_summaries:
    table_ids = [str(uuid.uuid4()) for _ in table_summaries]
    table_summary_docs = [
        Document(page_content=s, metadata={id_key: table_ids[i], "source": "table"})
        for i, s in enumerate(table_summaries)
    ]
    table_original_docs = [
        Document(
            page_content=t.metadata.text_as_html if hasattr(t.metadata, 'text_as_html') else t.text,
            metadata={id_key: table_ids[i], "source": "table"}
        )
        for i, t in enumerate(tables)
    ]
    retriever.vectorstore.add_documents(table_summary_docs)
    retriever.docstore.mset(list(zip(table_ids, table_original_docs)))
    print(f"📊 테이블 추가: {len(table_summary_docs)}개")

# 이미지 추가
if image_summaries:
    img_ids = [str(uuid.uuid4()) for _ in image_summaries]
    img_summary_docs = [
        Document(page_content=s, metadata={id_key: img_ids[i], "source": "image"})
        for i, s in enumerate(image_summaries)
    ]
    retriever.vectorstore.add_documents(img_summary_docs)
    retriever.docstore.mset(list(zip(img_ids, img_summary_docs)))
    print(f"🖼️ 이미지 추가: {len(img_summary_docs)}개")

# 7. RAG 체인 구축
rag_template = """Answer the question based only on the following context:

<Context>
{context}
</Context>

<Question>
{question}
</Question>

Answer:
"""
rag_prompt = ChatPromptTemplate.from_template(rag_template)
rag_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | rag_model
    | StrOutputParser()
)

# 8. 테스트
print("\n" + "=" * 50)
print("RAG 시스템 테스트")
print("=" * 50)

test_questions = [
    "What is the Transformer architecture?",
    "What is self-attention mechanism?",
    "What are the key components of the Transformer model?",
]

for q in test_questions:
    print(f"\n❓ Question: {q}")
    answer = rag_chain.invoke(q)
    print(f"💡 Answer: {answer[:500]}..." if len(answer) > 500 else f"💡 Answer: {answer}")
    print("-" * 50)

print("\n✅ Transformer 논문 멀티모달 RAG 시스템 구축 완료!")
```

**출력 예시:**
```
Processing data/transformer.pdf...
Extracted 78 chunks.
📊 테이블: 5개
📝 텍스트: 65개
🖼️ 이미지: 8개

텍스트 요약 생성 중...
텍스트 요약 완료: 65개
테이블 요약 생성 중...
테이블 요약 완료: 5개
이미지 요약 생성 중... (처음 10개만)
이미지 요약 완료: 8개

벡터 스토어 구축 중...
📄 텍스트 추가: 65개
📊 테이블 추가: 5개
🖼️ 이미지 추가: 8개

==================================================
RAG 시스템 테스트
==================================================

❓ Question: What is the Transformer architecture?
💡 Answer: The Transformer is a novel neural network architecture based entirely on attention mechanisms, dispensing with recurrence and convolutions entirely. It consists of an encoder-decoder structure where the encoder maps an input sequence to a continuous representation, and the decoder generates an output sequence from that representation. Both encoder and decoder are composed of stacked self-attention and point-wise, fully connected layers.
--------------------------------------------------

❓ Question: What is self-attention mechanism?
💡 Answer: Self-attention is a mechanism that relates different positions of a single sequence in order to compute a representation of that sequence. In the Transformer, it allows each position in the encoder to attend to all positions in the previous layer, and each position in the decoder to attend to all positions in the decoder up to and including that position.
--------------------------------------------------

✅ Transformer 논문 멀티모달 RAG 시스템 구축 완료!
```

### 솔루션 3: 하이브리드 RAG 시스템

```python
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple


class HybridMultimodalRAG:
    """하이브리드 멀티모달 RAG: 중요 이미지는 직접 활용, 나머지는 요약"""

    def __init__(self, chunks, importance_threshold=0.8):
        self.chunks = chunks
        self.threshold = importance_threshold
        self.important_images = []
        self.normal_images = []

    def calculate_importance(self, image_base64: str) -> float:
        """이미지 중요도 계산"""
        try:
            # Base64 디코딩
            image_data = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_data))

            # 1. 이미지 크기 (큰 이미지 = 중요할 가능성)
            width, height = image.size
            size_score = min((width * height) / (1000 * 1000), 1.0)  # 정규화 (1M pixels = 1.0)

            # 2. 색상 복잡도 (색상 다양성 = 정보량)
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[2]), axis=0))
                color_score = min(unique_colors / 10000, 1.0)  # 정규화
            else:
                color_score = 0.3  # 흑백 이미지는 낮은 점수

            # 3. 파일 크기 (큰 파일 = 상세 정보)
            file_size = len(image_base64)
            filesize_score = min(file_size / (500 * 1024), 1.0)  # 정규화 (500KB = 1.0)

            # 가중 평균
            importance = (size_score * 0.4 + color_score * 0.3 + filesize_score * 0.3)

            return importance

        except Exception as e:
            print(f"이미지 중요도 계산 실패: {e}")
            return 0.0

    def classify_images(self) -> Dict[str, List]:
        """중요 이미지 vs 일반 이미지 분류"""
        all_images = []

        # 모든 이미지 수집
        for chunk in self.chunks:
            if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
                for el in chunk.metadata.orig_elements:
                    if hasattr(el, 'metadata') and hasattr(el.metadata, 'image_base64'):
                        img_base64 = el.metadata.image_base64
                        if img_base64:
                            importance = self.calculate_importance(img_base64)
                            all_images.append({
                                'base64': img_base64,
                                'importance': importance,
                                'element': el
                            })

        # 중요도 기준 정렬
        all_images.sort(key=lambda x: x['importance'], reverse=True)

        # 상위 20% → 중요 이미지
        threshold_idx = int(len(all_images) * 0.2)
        self.important_images = all_images[:threshold_idx]
        self.normal_images = all_images[threshold_idx:]

        print(f"📊 이미지 분류 완료:")
        print(f"   중요 이미지 (직접 활용): {len(self.important_images)}개")
        print(f"   일반 이미지 (요약 활용): {len(self.normal_images)}개")

        return {
            'important': self.important_images,
            'normal': self.normal_images
        }

    def build_retriever(self) -> MultiVectorRetriever:
        """하이브리드 retriever 구축"""
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        from langchain.retrievers.multi_vector import MultiVectorRetriever
        from langchain_core.stores import InMemoryStore
        from langchain_core.documents import Document
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        import uuid

        # 벡터 스토어 초기화
        vectorstore = Chroma(
            collection_name="hybrid_mm_rag",
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        )

        store = InMemoryStore()
        id_key = "doc_id"

        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=store,
            id_key=id_key,
        )

        # 요약 체인 (일반 이미지용)
        image_messages = [
            ("user", [
                {"type": "text", "text": "이 이미지를 한국어로 상세히 요약하세요."},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image}"}}
            ])
        ]
        summarize_chain = (
            ChatPromptTemplate.from_messages(image_messages)
            | ChatOpenAI(model="gpt-4o-mini", temperature=0)
            | StrOutputParser()
        )

        # 중요 이미지: Base64 직접 저장
        print("\n중요 이미지 처리 중...")
        for img_data in self.important_images:
            doc_id = str(uuid.uuid4())

            # 요약용 문서 (메타데이터에 base64 포함)
            summary_doc = Document(
                page_content=f"중요 이미지 (중요도: {img_data['importance']:.2f})",
                metadata={
                    id_key: doc_id,
                    "source": "important_image",
                    "importance": img_data['importance'],
                    "image_base64": img_data['base64']  # Base64 저장
                }
            )

            # 원본 문서도 base64 포함
            original_doc = Document(
                page_content=f"[IMAGE] 중요 이미지 (중요도: {img_data['importance']:.2f})",
                metadata={
                    id_key: doc_id,
                    "source": "important_image",
                    "image_base64": img_data['base64']
                }
            )

            retriever.vectorstore.add_documents([summary_doc])
            retriever.docstore.mset([(doc_id, original_doc)])

        # 일반 이미지: 텍스트 요약 저장
        print("일반 이미지 요약 생성 중...")
        normal_summaries = summarize_chain.batch(
            [{"image": img['base64']} for img in self.normal_images[:50]],  # 최대 50개
            {"max_concurrency": 3}
        )

        for i, (img_data, summary) in enumerate(zip(self.normal_images[:50], normal_summaries)):
            doc_id = str(uuid.uuid4())

            summary_doc = Document(
                page_content=summary,
                metadata={
                    id_key: doc_id,
                    "source": "normal_image_summary",
                    "importance": img_data['importance']
                }
            )

            original_doc = Document(
                page_content=summary,
                metadata={
                    id_key: doc_id,
                    "source": "normal_image_summary"
                }
            )

            retriever.vectorstore.add_documents([summary_doc])
            retriever.docstore.mset([(doc_id, original_doc)])

        print(f"✅ 하이브리드 retriever 구축 완료")
        return retriever

    def generate_answer(self, query: str, retriever: MultiVectorRetriever) -> Dict:
        """이미지 타입에 따른 답변 생성"""
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI
        from langchain_core.output_parsers import StrOutputParser

        # 검색
        docs = retriever.invoke(query)

        # 중요 이미지가 있는지 확인
        has_important_image = any(
            doc.metadata.get('source') == 'important_image' and
            'image_base64' in doc.metadata
            for doc in docs
        )

        if has_important_image:
            # 멀티모달 프롬프트 (이미지 직접 활용)
            important_imgs = [
                doc.metadata['image_base64']
                for doc in docs
                if doc.metadata.get('source') == 'important_image' and 'image_base64' in doc.metadata
            ]

            messages = [
                ("user", [
                    {"type": "text", "text": f"""다음 이미지들을 참고하여 질문에 답하세요.

질문: {query}

답변:"""},
                    *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
                      for img in important_imgs[:3]]  # 최대 3개 이미지
                ])
            ]

            chain = (
                ChatPromptTemplate.from_messages(messages)
                | ChatOpenAI(model="gpt-4o-mini", temperature=0)
                | StrOutputParser()
            )

            answer = chain.invoke({})
            answer_type = "multimodal"

        else:
            # 텍스트 기반 프롬프트 (요약 활용)
            context = "\n\n".join([doc.page_content for doc in docs])

            template = """다음 컨텍스트를 바탕으로 질문에 답하세요.

<Context>
{context}
</Context>

<Question>
{question}
</Question>

답변:
"""

            chain = (
                ChatPromptTemplate.from_template(template)
                | ChatOpenAI(model="gpt-4o-mini", temperature=0)
                | StrOutputParser()
            )

            answer = chain.invoke({"context": context, "question": query})
            answer_type = "text_only"

        return {
            'answer': answer,
            'type': answer_type,
            'retrieved_docs': len(docs),
            'has_important_image': has_important_image
        }


# 사용 예시
hybrid_rag = HybridMultimodalRAG(pdf_chunks, importance_threshold=0.8)

# 이미지 분류
classification = hybrid_rag.classify_images()

# Retriever 구축
retriever = hybrid_rag.build_retriever()

# 답변 생성
result = hybrid_rag.generate_answer("2024년 매출 전망은?", retriever)

print(f"\n🤖 답변 타입: {result['type']}")
print(f"📊 검색된 문서: {result['retrieved_docs']}개")
print(f"🖼️ 중요 이미지 포함: {result['has_important_image']}")
print(f"\n💡 답변:\n{result['answer']}")
```

**출력 예시:**
```
📊 이미지 분류 완료:
   중요 이미지 (직접 활용): 9개
   일반 이미지 (요약 활용): 36개

중요 이미지 처리 중...
일반 이미지 요약 생성 중...
✅ 하이브리드 retriever 구축 완료

🤖 답변 타입: multimodal
📊 검색된 문서: 4개
🖼️ 중요 이미지 포함: True

💡 답변:
2024년 매출 전망은 12.5조원으로 전년 대비 8% 증가할 것으로 예상됩니다.
이미지에 나타난 분기별 매출 추이를 보면 1Q 2.8조원, 2Q 3.1조원, 3Q 3.2조원,
4Q 3.4조원으로 점진적인 성장세가 예상됩니다.
```

---

## 🌟 실무 활용 예시

### 예시 1: 금융 보고서 QA 챗봇

```python
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class FinancialReportChatbot:
    """금융 분석 보고서 QA 챗봇 시스템"""

    def __init__(self, retriever):
        self.retriever = retriever
        self.chain = self._build_chain()
        self.chat_history = []

    def _build_chain(self):
        """RAG 체인 구축"""
        template = """당신은 금융 분석 전문가입니다. 다음 문서를 바탕으로 질문에 답하세요.

<Context>
{context}
</Context>

<Chat History>
{history}
</Chat History>

<Question>
{question}
</Question>

답변 시 다음을 고려하세요:
1. 구체적인 수치와 날짜를 포함하세요
2. 출처 페이지나 섹션을 명시하세요
3. 불확실한 경우 명확히 밝히세요
4. 이전 대화 맥락을 고려하세요

답변:
"""

        prompt = ChatPromptTemplate.from_template(template)
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        chain = (
            {
                "context": lambda x: self.retriever.invoke(x["question"]),
                "history": lambda x: self._format_history(),
                "question": lambda x: x["question"]
            }
            | prompt
            | model
            | StrOutputParser()
        )

        return chain

    def _format_history(self, max_turns=3):
        """대화 히스토리 포맷팅"""
        if not self.chat_history:
            return "없음"

        recent = self.chat_history[-max_turns:]
        formatted = []
        for q, a in recent:
            formatted.append(f"Q: {q}\nA: {a}")

        return "\n\n".join(formatted)

    def query(self, question: str) -> str:
        """질문 처리"""
        answer = self.chain.invoke({"question": question})
        self.chat_history.append((question, answer))
        return answer

    def clear_history(self):
        """대화 히스토리 초기화"""
        self.chat_history = []

    def launch_ui(self):
        """Gradio UI 실행"""
        def respond(message, history):
            answer = self.query(message)
            return answer

        with gr.Blocks(title="금융 보고서 QA 챗봇") as demo:
            gr.Markdown("# 📊 금융 분석 보고서 QA 챗봇")
            gr.Markdown("증권사 분석 보고서에 대해 질문하세요.")

            chatbot = gr.Chatbot(height=500)
            msg = gr.Textbox(
                placeholder="예: 2024년 1분기 영업이익은 얼마인가요?",
                label="질문"
            )

            with gr.Row():
                submit = gr.Button("전송", variant="primary")
                clear = gr.Button("대화 초기화")

            gr.Examples(
                examples=[
                    "2024년 매출 전망은 얼마인가요?",
                    "주요 투자 포인트는 무엇인가요?",
                    "최근 2년간 투자 의견 변화는?",
                    "경쟁사 대비 강점은 무엇인가요?"
                ],
                inputs=msg
            )

            def user(user_message, history):
                return "", history + [[user_message, None]]

            def bot(history):
                user_message = history[-1][0]
                bot_message = respond(user_message, history)
                history[-1][1] = bot_message
                return history

            msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot, chatbot, chatbot
            )
            submit.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot, chatbot, chatbot
            )
            clear.click(lambda: (None, self.clear_history()), None, chatbot, queue=False)

        demo.launch(share=True)


# 사용 예시
chatbot = FinancialReportChatbot(retriever)

# UI 실행
chatbot.launch_ui()

# 또는 프로그래매틱 사용
answer = chatbot.query("2024년 1분기 영업이익은?")
print(answer)
```

**활용 시나리오:**
- 증권사 애널리스트 보고서 분석
- 실시간 투자 의사결정 지원
- 재무제표 데이터 질의응답
- 경쟁사 비교 분석

### 예시 2: 자동 보고서 생성 시스템

```python
from typing import List, Dict
import json
from datetime import datetime


class AutoReportGenerator:
    """RAG 기반 자동 보고서 생성 시스템"""

    def __init__(self, retriever):
        self.retriever = retriever
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    def generate_report(self, company_name: str, report_type: str = "quarterly") -> Dict:
        """자동 보고서 생성"""

        # 보고서 타입별 질문 템플릿
        question_templates = {
            "quarterly": [
                f"{company_name}의 최근 분기 실적은?",
                f"{company_name}의 주요 사업 동향은?",
                f"{company_name}의 향후 전망은?",
                f"{company_name}의 주요 리스크는?"
            ],
            "annual": [
                f"{company_name}의 연간 실적 추이는?",
                f"{company_name}의 사업 구조 변화는?",
                f"{company_name}의 중장기 전략은?",
                f"{company_name}의 재무 건전성은?"
            ]
        }

        questions = question_templates.get(report_type, question_templates["quarterly"])

        # 각 질문에 대한 답변 수집
        sections = {}
        for question in questions:
            docs = self.retriever.invoke(question)
            context = "\n\n".join([doc.page_content for doc in docs[:5]])

            prompt = f"""다음 컨텍스트를 바탕으로 질문에 대한 보고서 섹션을 작성하세요.

<Context>
{context}
</Context>

<Question>
{question}
</Question>

보고서 섹션 (불렛 포인트 3-5개, 구체적 수치 포함):
"""

            section = self.model.invoke(prompt).content
            section_title = question.replace(f"{company_name}의 ", "").replace("?", "")
            sections[section_title] = section

        # 전체 요약 생성
        all_sections = "\n\n".join([f"## {title}\n{content}" for title, content in sections.items()])

        summary_prompt = f"""다음 섹션들을 종합하여 3-5문장의 핵심 요약을 작성하세요.

{all_sections}

핵심 요약:
"""

        summary = self.model.invoke(summary_prompt).content

        # 보고서 생성
        report = {
            "company": company_name,
            "report_type": report_type,
            "generated_at": datetime.now().isoformat(),
            "summary": summary,
            "sections": sections
        }

        return report

    def export_markdown(self, report: Dict, output_path: str):
        """마크다운 형식으로 내보내기"""
        md_content = []

        # 헤더
        md_content.append(f"# {report['company']} {report['report_type'].upper()} 보고서")
        md_content.append(f"\n**생성일시:** {report['generated_at']}\n")

        # 요약
        md_content.append("## 📌 핵심 요약\n")
        md_content.append(report['summary'])
        md_content.append("\n---\n")

        # 섹션
        for title, content in report['sections'].items():
            md_content.append(f"## {title}\n")
            md_content.append(content)
            md_content.append("\n")

        # 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(md_content))

        print(f"📄 보고서 저장: {output_path}")

    def export_json(self, report: Dict, output_path: str):
        """JSON 형식으로 내보내기"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"💾 JSON 저장: {output_path}")


# 사용 예시
generator = AutoReportGenerator(retriever)

# 분기 보고서 생성
quarterly_report = generator.generate_report("삼성전기", report_type="quarterly")

# 마크다운 내보내기
generator.export_markdown(quarterly_report, "reports/삼성전기_Q1_2024.md")

# JSON 내보내기
generator.export_json(quarterly_report, "reports/삼성전기_Q1_2024.json")

# 보고서 출력
print("\n" + "=" * 60)
print(quarterly_report['summary'])
print("=" * 60)
```

**출력 예시:**
```
📄 보고서 저장: reports/삼성전기_Q1_2024.md
💾 JSON 저장: reports/삼성전기_Q1_2024.json

============================================================
삼성전기는 2024년 1분기 매출 2.8조원(+12% YoY), 영업이익 2,400억원(+45% YoY)을
기록하며 실적 개선세를 보였습니다. MLCC 수요 증가와 FC-BGA 기판 사업 확대가
주요 동력이며, 2024년 연간 매출 12.5조원, 영업이익 1.1조원을 전망합니다.
다만 환율 변동성과 중국 경기 둔화가 주요 리스크 요인으로 작용할 것으로 보입니다.
============================================================
```

**활용 시나리오:**
- 증권사 애널리스트 보고서 자동 초안 작성
- 정기 투자 리포트 생성
- 경쟁사 비교 분석 보고서
- 산업 동향 모니터링 리포트

---

## 🎓 Part 2 요약

### 핵심 내용
1. **타입별 요약 체인**: Text/Table (TextLLM), Image (MultimodalLLM) 각각 구현
2. **MultiVectorRetriever**: 요약으로 검색, 원본으로 답변 생성하는 2단계 구조
3. **RAG 파이프라인**: Retriever → Prompt → LLM → Answer 전체 흐름 연결
4. **페이지 기반 요약**: metadata.page_number 활용한 페이지별 통합 요약
5. **실전 시스템**: Transformer 논문 RAG, 금융 챗봇, 자동 보고서 생성

### 옵션 2 방식의 특징
- ✅ **비용 효율적**: 이미지를 텍스트로 변환하여 임베딩 비용 절감
- ✅ **기존 인프라 활용**: 텍스트 RAG 시스템 재사용 가능
- ✅ **확장성**: 대량 문서 처리에 유리
- ⚠️ **정보 손실**: 이미지의 시각적 정보 일부 손실
- ⚠️ **품질 제한**: 옵션 1(CLIP) 대비 답변 품질 낮을 수 있음

### 다음 단계
- **옵션 3 구현**: 이미지 요약 + 원본 참조 하이브리드 방식
- **성능 최적화**: 캐싱, 배치 처리, 병렬화
- **평가 시스템**: RAG 품질 자동 평가 (retrieval accuracy, answer quality)
- **프로덕션 배포**: FastAPI 서버, 모니터링, 로깅

---

**관련 파일: PRJ04_W2_006_Multimodal_RAG_Financial_Part2_Part1.md**
