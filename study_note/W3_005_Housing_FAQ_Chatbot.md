# W3_005_Housing_FAQ_Chatbot.md

## 학습 목표
- RAG 기반 FAQ 시스템의 전체 구현 과정 이해하기
- 문서 전처리와 청킹, 메타데이터 활용 방법 학습하기
- 벡터 데이터베이스를 활용한 효율적인 검색 시스템 구축하기
- Gradio를 사용한 실용적인 챗봇 인터페이스 개발하기
- 메타데이터 필터링과 관련성 평가를 통한 검색 품질 향상하기

## 주요 개념

### 1. FAQ 시스템의 구조
FAQ(Frequently Asked Questions) 시스템은 사용자의 질문에 대해 기존에 정리된 답변을 제공하는 시스템입니다. RAG 방식을 활용하면 정확한 답변과 함께 근거가 되는 원본 문서를 제시할 수 있어 신뢰성을 높일 수 있습니다.

### 2. 문서 전처리 과정
- **데이터 로딩**: TextLoader를 사용한 원본 텍스트 파일 읽기
- **구조 분석**: 정규표현식을 활용한 Q&A 쌍 자동 추출
- **메타데이터 생성**: LLM을 활용한 키워드와 요약 정보 자동 생성
- **문서 포맷팅**: 검색 최적화를 위한 문서 구조화

### 3. 검색 성능 최적화
- **MMR(Maximal Marginal Relevance)**: 관련성과 다양성의 균형을 맞춘 검색
- **메타데이터 필터링**: 구조화된 조건을 통한 정밀 검색
- **관련성 평가**: LLM을 활용한 검색 결과의 품질 검증
- **임계값 설정**: 낮은 품질의 답변 필터링

### 4. 사용자 경험 최적화
- **스트리밍 응답**: 실시간 응답 생성으로 체감 속도 향상
- **소스 문서 제공**: 답변의 근거가 되는 원본 문서 표시
- **오류 처리**: 관련 문서가 없을 때의 적절한 안내

## 환경 설정

### 필수 라이브러리 설치
```bash
pip install langchain langchain-openai langchain-chroma langchain-community gradio python-dotenv pydantic
```

### 환경 변수 설정
```python
from dotenv import load_dotenv
import os

load_dotenv()

# OpenAI API 키 설정
openai_api_key = os.getenv("OPENAI_API_KEY")
```

## 핵심 구현

### 1. 문서 로딩 및 전처리

FAQ 텍스트 파일에서 질문-답변 쌍을 자동으로 추출합니다:

```python
from langchain_community.document_loaders import TextLoader
import re

def extract_qa_pairs(text):
    """텍스트에서 Q&A 쌍을 추출하는 함수"""
    qa_pairs = []
    lines = text.strip().split('\n')

    current_question = None
    current_answer = []
    current_number = None
    in_answer = False

    for line in lines:
        line = line.strip()

        # 빈 줄 처리
        if not line:
            if current_question and current_answer:
                # 다음 질문이 시작되기 전 빈 줄이면 현재 QA 쌍 저장
                qa_pairs.append({
                    'number': current_number,
                    'question': current_question,
                    'answer': ' '.join(current_answer).strip()
                })
                current_question = None
                current_answer = []
                current_number = None
                in_answer = False
            continue

        # 새로운 질문 확인 (Q 다음에 숫자가 오는 패턴)
        q_match = re.match(r'Q(\d+)\s+(.*)', line)
        if q_match:
            # 이전 QA 쌍이 있으면 저장
            if current_question and current_answer:
                qa_pairs.append({
                    'number': current_number,
                    'question': current_question,
                    'answer': ' '.join(current_answer).strip()
                })
                current_answer = []

            # 새로운 질문 시작
            current_number = int(q_match.group(1))
            current_question = q_match.group(2).strip().rstrip('?') + '?'
            current_answer = []
            in_answer = False

        # 답변 시작 확인
        elif line.startswith('A ') or (current_question and not current_answer and line):
            in_answer = True
            current_answer.append(line.lstrip('A '))

        # 기존 답변에 내용 추가
        elif current_question is not None and (in_answer or not line.startswith('Q')):
            if in_answer or (current_answer and not line.startswith('Q')):
                current_answer.append(line)

    # 마지막 QA 쌍 저장
    if current_question and current_answer:
        qa_pairs.append({
            'number': current_number,
            'question': current_question,
            'answer': ' '.join(current_answer).strip()
        })

    return qa_pairs

# 파일 읽기 및 QA 쌍 추출
faq_text_file = "data/housing_faq.txt"

with open(faq_text_file, 'r') as f:
    content = f.read()

qa_pairs = extract_qa_pairs(content)
print(f"추출된 QA 쌍 개수: {len(qa_pairs)}")
```

### 2. LLM을 활용한 메타데이터 생성

각 Q&A 쌍에서 키워드와 요약을 자동으로 추출합니다:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# LLM 초기화
llm = ChatOpenAI(model='gpt-4.1-mini', temperature=0.3)

# 출력 형식 정의
class KeywordOutput(BaseModel):
    keyword: str = Field(description="텍스트에서 추출한 가장 중요한 키워드(법률용어, 주제 등)")
    summary: str = Field(description="텍스트의 간단한 요약")

# 프롬프트 템플릿
template = """
다음 주택청약 FAQ 텍스트를 분석하여 키워드와 요약을 추출해주세요.

텍스트:
{text}

키워드는 법률용어, 핵심 개념, 지역명 등 검색에 유용한 단어들을 쉼표로 구분하여 나열해주세요.
요약은 2-3문장으로 핵심 내용을 간결하게 정리해주세요.
"""

# LCEL 체인 구성
prompt = ChatPromptTemplate.from_template(template)
llm_with_structure = llm.with_structured_output(KeywordOutput)
keyword_chain = prompt | llm_with_structure

# 각 QA 쌍에 메타데이터 추가
for pair in qa_pairs:
    full_text = f"질문: {pair['question']}\n답변: {pair['answer']}"

    try:
        result = keyword_chain.invoke({"text": full_text})
        pair['keyword'] = result.keyword
        pair['summary'] = result.summary
    except Exception as e:
        print(f"Error processing pair {pair['number']}: {e}")
        pair['keyword'] = ""
        pair['summary'] = ""
```

### 3. 문서 객체 생성 및 포맷팅

검색에 최적화된 문서 구조를 생성합니다:

```python
from langchain_core.documents import Document

def format_qa_pairs_with_summary(qa_pairs):
    """QA 쌍을 요약과 함께 문서 객체로 변환"""
    processed_docs = []

    for pair in qa_pairs:
        # 요약문을 메인 콘텐츠로 사용 (검색 최적화)
        doc = Document(
            page_content=pair['summary'],
            metadata={
                'question_id': pair['number'],
                'question': pair['question'],
                'answer': pair['answer'],
                'keyword': pair['keyword'],
                'full_content': f"[{pair['number']}]\n질문: {pair['question']}\n답변: {pair['answer']}"
            }
        )
        processed_docs.append(doc)

    return processed_docs

# 문서 포맷팅
summary_formatted_docs = format_qa_pairs_with_summary(qa_pairs)
print(f"생성된 문서 객체 수: {len(summary_formatted_docs)}")
```

### 4. 벡터 데이터베이스 구축

Chroma 벡터 데이터베이스에 문서를 저장합니다:

```python
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 임베딩 모델 초기화
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 요약 문서용 벡터 스토어 생성
summary_vectorstore = Chroma.from_documents(
    documents=summary_formatted_docs,
    embedding=embeddings,
    collection_name="housing_faq_summary_db",
    persist_directory="./chroma_summary_db",
)

print("요약 문서 벡터 스토어 생성 완료")

# 벡터 스토어 로드 (재사용 시)
loaded_summary_vectorstore = Chroma(
    collection_name="housing_faq_summary_db",
    persist_directory="./chroma_summary_db",
    embedding_function=embeddings,
)
```

### 5. 고급 검색기 구현

MMR(Maximal Marginal Relevance)을 활용한 다양성 기반 검색:

```python
# MMR 검색기 생성
mmr_retriever = loaded_summary_vectorstore.as_retriever(
    search_type="mmr",  # 다양성 기반 검색
    search_kwargs={
        "k": 3,         # 최종적으로 반환할 문서 수
        "fetch_k": 10,  # 초기에 가져올 문서 수
        "lambda_mult": 0.5,  # 다양성 가중치 (0: 다양성 최대, 1: 관련성 최대)
    }
)

# 테스트
query = "수원시의 주택건설지역은 어디에 해당하나요?"
results = mmr_retriever.invoke(query)

for i, result in enumerate(results, 1):
    print(f"문서 {i}:")
    print(f"질문 ID: {result.metadata['question_id']}")
    print(f"질문: {result.metadata['question']}")
    print(f"답변: {result.metadata['answer']}")
    print("="*50)
```

### 6. 메타데이터 필터링 시스템

구조화된 검색 조건을 자동으로 생성하는 시스템:

```python
from typing import Optional, Dict

class ImprovedMetadataFilter(BaseModel):
    keyword: Optional[str] = Field(description="검색할 키워드")
    keyword_expression: Optional[str] = Field(description="키워드 검색 표현식")
    question_id_min: Optional[int] = Field(description="질문 ID 최소값")
    question_id_max: Optional[int] = Field(description="질문 ID 최대값")
    question_contains: Optional[str] = Field(description="질문에 포함된 문자열")
    answer_contains: Optional[str] = Field(description="답변에 포함된 문자열")
    operation: Optional[str] = Field(description="조건간 연산 ($and, $or)")

def create_metadata_filter(query: str, llm) -> Dict:
    """사용자 쿼리를 분석하여 메타데이터 필터를 생성"""

    improved_system_prompt = """사용자 쿼리에서 검색 조건을 추출하여 구조화된 필터를 생성합니다.

다음 필드들을 활용합니다:
1. 키워드 검색:
   - keyword: 검색할 키워드
   - keyword_expression: 검색 표현식 ($eq, $ne, $in)

2. 질문 ID 범위 검색:
   - question_id_min: 최소 질문 ID
   - question_id_max: 최대 질문 ID

3. 텍스트 내용 검색:
   - question_contains: 질문에 포함된 문자열
   - answer_contains: 답변에 포함된 문자열

4. 조건 조합:
   - operation: 여러 조건을 조합하는 연산자 ($and, $or)

예시:
- "청약통장 관련 문서" → keyword: "청약통장", keyword_expression: "$eq"
- "질문 ID 5번부터 15번 사이의 청약통장 관련 문서"
  → keyword: "청약통장", keyword_expression: "$eq", question_id_min: 5, question_id_max: 15, operation: "$and"
"""

    improved_prompt = ChatPromptTemplate.from_messages([
        ("system", improved_system_prompt),
        ("human", "쿼리: {query}")
    ])

    improved_metadata_chain = improved_prompt | llm.with_structured_output(ImprovedMetadataFilter)
    filter_params = improved_metadata_chain.invoke({"query": query})

    # 필터 딕셔너리 생성
    filter_dict = {}
    conditions = []

    # 키워드 조건
    if filter_params.keyword and filter_params.keyword_expression:
        if filter_params.keyword_expression == "$eq":
            conditions.append({"keyword": {"$regex": f".*{filter_params.keyword}.*"}})
        elif filter_params.keyword_expression == "$in":
            conditions.append({"keyword": {"$in": [filter_params.keyword]}})

    # 질문 ID 범위 처리
    id_conditions = []
    if filter_params.question_id_min is not None:
        id_conditions.append({"question_id": {"$gte": filter_params.question_id_min}})
    if filter_params.question_id_max is not None:
        id_conditions.append({"question_id": {"$lte": filter_params.question_id_max}})

    if id_conditions:
        if len(id_conditions) == 1:
            conditions.extend(id_conditions)
        else:
            conditions.append({"$and": id_conditions})

    # 텍스트 내용 조건
    if filter_params.question_contains:
        conditions.append({"question": {"$regex": f".*{filter_params.question_contains}.*"}})
    if filter_params.answer_contains:
        conditions.append({"answer": {"$regex": f".*{filter_params.answer_contains}.*"}})

    # 최종 필터 구성
    if conditions:
        if len(conditions) == 1:
            filter_dict = conditions[0]
        else:
            operation = filter_params.operation if filter_params.operation in ["$and", "$or"] else "$and"
            filter_dict = {operation: conditions}

    return filter_dict

# 사용 예제
query = "주택건설지역 관련 문서를 질문 ID 10번 이하인 문서중에서 검색해주세요"
filter_dict = create_metadata_filter(query, llm)

# 필터 적용된 검색
filtered_mmr_retriever = loaded_summary_vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10,
        "lambda_mult": 0.5,
        "filter": filter_dict
    }
)
```

### 7. RAG 체인 구성

문서 검색부터 답변 생성까지의 전체 파이프라인:

```python
from langchain.schema import format_document
from langchain_core.runnables import RunnableLambda
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser

def get_context_and_docs(question: str) -> Dict:
    """문서와 포맷팅된 컨텍스트를 함께 반환"""

    def format_docs(docs):
        return '\n\n'.join([d.page_content for d in docs])

    docs = mmr_retriever.invoke(question)

    return {
        "question": question,
        "context": format_docs(docs),
        "source_documents": docs
    }

def prompt_and_generate_answer(input_data: Dict) -> Dict:
    """컨텍스트와 질문을 입력으로 받아 답변을 생성"""

    template = """다음 컨텍스트를 기반으로 질문에 답변해주세요.

컨텍스트:
{context}

질문:
{question}

답변:"""

    prompt_template = ChatPromptTemplate.from_template(template)
    answer_chain = prompt_template | llm | StrOutputParser()

    answer = answer_chain.invoke({
        "context": input_data["context"],
        "question": input_data["question"]
    })

    return {
        "answer": answer,
        "source_documents": input_data["source_documents"]
    }

# RAG 체인 구성
rag_chain_with_sources = {
    'question_and_context': RunnableLambda(get_context_and_docs),
    'response': RunnableLambda(prompt_and_generate_answer),
    'question': itemgetter("question"),
    "source_documents": itemgetter("source_documents")
}

# 테스트
query = "수원시의 주택건설지역은 어디에 해당하나요?"
result = rag_chain_with_sources.invoke({"question": query})
print("답변:", result["response"]["answer"])
```

### 8. 관련성 평가 시스템

검색된 문서의 품질을 평가하여 부적절한 답변을 필터링:

```python
class RelevanceOutput(BaseModel):
    is_relevant: bool = Field(description="문서가 질문과 관련있는지 여부")
    confidence: float = Field(description="관련성에 대한 확신도 (0.0-1.0)")
    reason: str = Field(description="판단 근거")

def create_relevance_checker(llm):
    """문서 관련성 평가 체인 생성"""

    relevance_prompt = ChatPromptTemplate.from_messages([
        ("system", """주어진 컨텍스트가 질문에 답변하는데 필요한 정보를 포함하고 있는지 평가하세요.

평가 기준:
1. 컨텍스트가 질문에 답변하는데 필요한 정보를 직접적으로 포함하고 있는가?
2. 컨텍스트의 정보로부터 답변에 필요한 내용을 논리적으로 추론할 수 있는가?
3. 컨텍스트의 정보가 질문에 대한 답변을 제공할 수 있는가?

is_relevant: True/False로 판단
confidence: 0.0-1.0 사이의 확신도
reason: 판단 근거를 간단히 설명
"""),
        ("human", """[컨텍스트]
{context}

[질문]
{question}""")
    ])

    return relevance_prompt | llm.with_structured_output(RelevanceOutput)

relevance_checker = create_relevance_checker(llm)

def check_document_relevance(doc, question):
    """문서의 관련성을 평가하는 함수"""
    return relevance_checker.invoke({
        "context": doc.page_content,
        "question": question
    })
```

### 9. Gradio 챗봇 인터페이스

실용적인 웹 기반 챗봇 인터페이스 구현:

```python
import gradio as gr
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class SearchResult:
    context: str
    source_documents: Optional[List]

class RAGSystem:
    def __init__(self, llm, retriever, relevance_checker=None):
        self.llm = llm
        self.retriever = retriever
        self.relevance_checker = relevance_checker
        self.min_confidence_threshold = 0.6

    def _format_docs(self, docs: List) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    def _format_source_documents(self, docs: Optional[List]) -> str:
        if not docs:
            return "\n\nℹ️ 관련 문서를 찾을 수 없습니다."

        formatted_docs = []
        for i, doc in enumerate(docs, 1):
            formatted_docs.append(
                f"📄 **참조 문서 {i}** (질문 ID: {doc.metadata.get('question_id', 'N/A')})\n"
                f"**질문**: {doc.metadata.get('question', 'N/A')}\n"
                f"**답변**: {doc.metadata.get('answer', doc.page_content)}"
            )

        return "\n\n" + "\n\n".join(formatted_docs)

    def _check_relevance(self, docs: List, question: str) -> List:
        """문서의 관련성 확인"""
        if not self.relevance_checker or not docs:
            return docs

        relevant_docs = []
        for doc in docs:
            try:
                relevance_result = self.relevance_checker.invoke({
                    "context": doc.page_content,
                    "question": question
                })

                if (relevance_result.is_relevant and
                    relevance_result.confidence >= self.min_confidence_threshold):
                    relevant_docs.append(doc)

            except Exception as e:
                print(f"관련성 평가 중 오류 발생: {e}")
                relevant_docs.append(doc)  # 오류 시 문서 유지

        return relevant_docs

    def search_and_generate(self, question: str) -> str:
        """질문에 대한 답변을 생성하고 소스 문서를 포함하여 반환"""
        try:
            # 문서 검색
            docs = self.retriever.invoke(question)

            # 관련성 확인
            relevant_docs = self._check_relevance(docs, question)

            if not relevant_docs:
                return ("죄송합니다. 질문과 관련된 정보를 찾을 수 없습니다. "
                       "다른 방식으로 질문을 다시 해보시거나, 구체적인 키워드를 포함해 주세요.")

            # 컨텍스트 생성
            context = self._format_docs(relevant_docs)

            # 답변 생성
            template = """다음 컨텍스트를 기반으로 질문에 정확하고 상세하게 답변해주세요.

컨텍스트:
{context}

질문: {question}

답변:"""

            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.llm | StrOutputParser()

            answer = chain.invoke({
                "context": context,
                "question": question
            })

            # 소스 문서 정보 추가
            source_info = self._format_source_documents(relevant_docs)

            return f"{answer}{source_info}"

        except Exception as e:
            return f"답변 생성 중 오류가 발생했습니다: {str(e)}"

# RAG 시스템 초기화
rag_system = RAGSystem(
    llm=llm,
    retriever=mmr_retriever,
    relevance_checker=relevance_checker
)

def respond_to_user(message, history):
    """Gradio 챗봇 응답 함수 (스트리밍 지원)"""
    try:
        # RAG 시스템으로 답변 생성
        response = rag_system.search_and_generate(message)

        # 스트리밍 효과를 위한 점진적 응답
        for i in range(0, len(response), 10):
            yield response[:i+10]

    except Exception as e:
        yield f"오류가 발생했습니다: {str(e)}"

# Gradio 인터페이스 생성
def create_gradio_interface():
    iface = gr.ChatInterface(
        respond_to_user,
        title="🏠 주택청약 FAQ 챗봇",
        description="주택청약 관련 질문에 대해 정확한 답변과 근거 문서를 제공합니다.",
        examples=[
            "무주택세대구성원이란 무엇인가요?",
            "경기도 과천시의 주택건설지역은 어디인가요?",
            "청약통장과 관련된 정보를 알려주세요.",
            "주택 공급 관련 규정을 설명해주세요."
        ],
        theme=gr.themes.Soft(),
        retry_btn="🔄 다시 시도",
        undo_btn="↩️ 이전",
        clear_btn="🗑️ 대화 초기화"
    )
    return iface

# 인터페이스 실행
if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
```

## 실습 문제

### 실습 1: 다중 도메인 FAQ 시스템
서로 다른 도메인의 FAQ 데이터를 통합하여 관리하는 시스템을 구현하세요.

**요구사항:**
1. 부동산, 세금, 금융 등 3개 도메인의 FAQ 데이터 통합
2. 도메인별 메타데이터 분류 및 필터링
3. 도메인 간 유사 질문 탐지 및 크로스 레퍼런스
4. 도메인별 전문 용어 처리 및 동의어 확장

### 실습 2: 개인화된 FAQ 추천 시스템
사용자의 질문 패턴을 학습하여 개인화된 FAQ를 추천하는 시스템을 구현하세요.

**요구사항:**
1. 사용자별 질문 히스토리 저장 및 분석
2. 질문 유형별 선호도 학습
3. 유사 사용자 그룹 기반 협업 필터링
4. 실시간 추천 업데이트 및 A/B 테스트

### 실습 3: 멀티모달 FAQ 시스템
텍스트뿐만 아니라 이미지, 표, 도표가 포함된 FAQ 시스템을 구현하세요.

**요구사항:**
1. 이미지 OCR을 통한 텍스트 추출
2. 표와 도표 데이터의 구조화 및 검색
3. 시각적 요소를 고려한 답변 생성
4. 멀티모달 콘텐츠의 관련성 평가

## 실습 해답

### 실습 1 해답: 다중 도메인 FAQ 시스템

```python
from enum import Enum
from typing import Dict, List, Set
import sqlite3
from datetime import datetime

class Domain(Enum):
    REAL_ESTATE = "부동산"
    TAX = "세금"
    FINANCE = "금융"
    GENERAL = "일반"

class MultiDomainFAQSystem:
    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings
        self.domain_vectorstores = {}
        self.domain_synonyms = self._load_domain_synonyms()
        self.cross_references = {}

    def _load_domain_synonyms(self) -> Dict[str, List[str]]:
        """도메인별 전문 용어 동의어 사전"""
        return {
            Domain.REAL_ESTATE.value: {
                "주택청약": ["아파트 청약", "주택 분양", "분양권"],
                "전용면적": ["전용 면적", "전용평", "평수"],
                "주택건설지역": ["해당지역", "공급지역", "건설 지역"]
            },
            Domain.TAX.value: {
                "취득세": ["취득 세금", "부동산 취득세"],
                "양도소득세": ["양도세", "양도 소득세"],
                "종합부동산세": ["종부세", "보유세"]
            },
            Domain.FINANCE.value: {
                "주택담보대출": ["주담대", "아파트 대출", "주택 대출"],
                "전세자금대출": ["전세 대출", "전세금 대출"],
                "LTV": ["주택담보인정비율", "대출한도비율"]
            }
        }

    def process_domain_documents(self, documents: List[Dict], domain: Domain):
        """도메인별 문서 처리 및 벡터 스토어 생성"""

        # 도메인별 메타데이터 강화
        enhanced_docs = []
        for doc in documents:
            enhanced_doc = Document(
                page_content=doc['content'],
                metadata={
                    **doc.get('metadata', {}),
                    'domain': domain.value,
                    'domain_keywords': self._extract_domain_keywords(doc['content'], domain),
                    'processed_date': datetime.now().isoformat(),
                    'cross_domain_tags': self._generate_cross_domain_tags(doc['content'])
                }
            )
            enhanced_docs.append(enhanced_doc)

        # 도메인별 벡터 스토어 생성
        self.domain_vectorstores[domain] = Chroma.from_documents(
            documents=enhanced_docs,
            embedding=self.embeddings,
            collection_name=f"faq_{domain.value}",
            persist_directory=f"./chroma_{domain.value}"
        )

        # 크로스 레퍼런스 생성
        self._build_cross_references(enhanced_docs, domain)

    def _extract_domain_keywords(self, content: str, domain: Domain) -> List[str]:
        """도메인별 키워드 추출"""
        domain_prompt = f"""
        다음은 {domain.value} 분야의 FAQ 내용입니다.
        이 분야의 전문 용어와 핵심 키워드를 추출해주세요.

        내용: {content}

        키워드를 쉼표로 구분하여 나열해주세요.
        """

        response = self.llm.invoke(domain_prompt)
        keywords = [k.strip() for k in response.content.split(',')]

        # 동의어 확장
        expanded_keywords = set(keywords)
        for keyword in keywords:
            if keyword in self.domain_synonyms.get(domain.value, {}):
                expanded_keywords.update(self.domain_synonyms[domain.value][keyword])

        return list(expanded_keywords)

    def _generate_cross_domain_tags(self, content: str) -> List[str]:
        """다른 도메인과의 연관성 태그 생성"""
        cross_domain_prompt = f"""
        다음 내용이 부동산, 세금, 금융 분야 중 어느 분야와 관련이 있는지 분석해주세요.

        내용: {content}

        관련 분야를 모두 나열해주세요 (부동산, 세금, 금융 중에서).
        """

        response = self.llm.invoke(cross_domain_prompt)
        domains = [d.strip() for d in response.content.split(',') if d.strip() in ["부동산", "세금", "금융"]]
        return domains

    def _build_cross_references(self, documents: List[Document], domain: Domain):
        """도메인 간 크로스 레퍼런스 구축"""
        for doc in documents:
            cross_tags = doc.metadata.get('cross_domain_tags', [])
            doc_id = f"{domain.value}_{doc.metadata.get('question_id', 'unknown')}"

            self.cross_references[doc_id] = {
                'related_domains': cross_tags,
                'similarity_threshold': 0.8,
                'related_questions': []
            }

    def search_across_domains(self, query: str, target_domains: List[Domain] = None, k: int = 5) -> Dict:
        """다중 도메인 검색"""
        if target_domains is None:
            target_domains = list(self.domain_vectorstores.keys())

        all_results = {}

        for domain in target_domains:
            if domain not in self.domain_vectorstores:
                continue

            # 도메인별 검색 수행
            retriever = self.domain_vectorstores[domain].as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "lambda_mult": 0.7}
            )

            domain_results = retriever.invoke(query)
            all_results[domain.value] = domain_results

        # 결과 통합 및 랭킹
        unified_results = self._unify_and_rank_results(all_results, query)

        return {
            'unified_results': unified_results,
            'domain_results': all_results,
            'cross_references': self._find_cross_references(unified_results)
        }

    def _unify_and_rank_results(self, domain_results: Dict, query: str) -> List[Document]:
        """도메인별 결과를 통합하고 재랭킹"""
        all_docs = []

        for domain, docs in domain_results.items():
            for doc in docs:
                # 도메인 가중치 적용
                domain_weight = self._get_domain_weight(query, domain)
                doc.metadata['domain_score'] = domain_weight
                all_docs.append(doc)

        # 유사도와 도메인 점수를 결합하여 재랭킹
        ranked_docs = sorted(all_docs,
                           key=lambda x: x.metadata.get('domain_score', 0),
                           reverse=True)

        return ranked_docs[:10]  # 상위 10개 결과 반환

    def _get_domain_weight(self, query: str, domain: str) -> float:
        """쿼리에 대한 도메인 가중치 계산"""
        domain_keywords = {
            "부동산": ["주택", "아파트", "청약", "분양", "임대", "전세", "매매"],
            "세금": ["세금", "세율", "공제", "신고", "납부", "환급", "과세"],
            "금융": ["대출", "금리", "은행", "대출한도", "상환", "이자", "신용"]
        }

        query_lower = query.lower()
        domain_words = domain_keywords.get(domain, [])

        matches = sum(1 for word in domain_words if word in query_lower)
        return matches / len(domain_words) if domain_words else 0.1

    def _find_cross_references(self, results: List[Document]) -> Dict:
        """관련 질문 크로스 레퍼런스 찾기"""
        cross_refs = {}

        for doc in results:
            doc_id = f"{doc.metadata['domain']}_{doc.metadata.get('question_id', 'unknown')}"
            if doc_id in self.cross_references:
                related_domains = self.cross_references[doc_id]['related_domains']
                cross_refs[doc_id] = {
                    'current_domain': doc.metadata['domain'],
                    'related_domains': related_domains,
                    'suggested_searches': self._generate_related_searches(doc.page_content, related_domains)
                }

        return cross_refs

    def _generate_related_searches(self, content: str, related_domains: List[str]) -> Dict[str, List[str]]:
        """관련 도메인에서의 추천 검색어 생성"""
        related_searches = {}

        for domain in related_domains:
            prompt = f"""
            다음 내용과 관련하여 {domain} 분야에서 유용한 검색 키워드를 3개 제안해주세요.

            내용: {content}

            검색어를 쉼표로 구분하여 나열해주세요.
            """

            response = self.llm.invoke(prompt)
            searches = [s.strip() for s in response.content.split(',')[:3]]
            related_searches[domain] = searches

        return related_searches

# 사용 예제
multi_domain_system = MultiDomainFAQSystem(llm, embeddings)

# 각 도메인별 데이터 처리
real_estate_docs = [
    {"content": "주택청약 관련 FAQ...", "metadata": {"question_id": 1}},
    # ... 더 많은 부동산 문서
]

tax_docs = [
    {"content": "취득세 관련 FAQ...", "metadata": {"question_id": 1}},
    # ... 더 많은 세금 문서
]

finance_docs = [
    {"content": "주택담보대출 관련 FAQ...", "metadata": {"question_id": 1}},
    # ... 더 많은 금융 문서
]

# 도메인별 처리
multi_domain_system.process_domain_documents(real_estate_docs, Domain.REAL_ESTATE)
multi_domain_system.process_domain_documents(tax_docs, Domain.TAX)
multi_domain_system.process_domain_documents(finance_docs, Domain.FINANCE)

# 다중 도메인 검색
query = "아파트 구매 시 필요한 세금과 대출 정보"
results = multi_domain_system.search_across_domains(query)

print("통합 검색 결과:")
for i, doc in enumerate(results['unified_results'][:5], 1):
    print(f"{i}. [{doc.metadata['domain']}] {doc.metadata.get('question', doc.page_content[:100])}")
```

### 실습 2 해답: 개인화된 FAQ 추천 시스템

```python
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class PersonalizedFAQRecommender:
    def __init__(self, llm, embeddings, db_path="user_history.db"):
        self.llm = llm
        self.embeddings = embeddings
        self.db_path = db_path
        self.user_profiles = {}
        self.question_categories = {}
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='korean')
        self._init_database()

    def _init_database(self):
        """사용자 히스토리 데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                question TEXT,
                category TEXT,
                satisfaction_score REAL,
                timestamp DATETIME,
                session_id TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY,
                preferred_categories TEXT,
                question_complexity_level REAL,
                avg_session_length REAL,
                last_active DATETIME
            )
        ''')

        conn.commit()
        conn.close()

    def log_user_interaction(self, user_id: str, question: str, satisfaction_score: float, session_id: str):
        """사용자 상호작용 로깅"""
        category = self._classify_question(question)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO user_history
            (user_id, question, category, satisfaction_score, timestamp, session_id)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, question, category, satisfaction_score, datetime.now(), session_id))

        conn.commit()
        conn.close()

        # 사용자 프로필 업데이트
        self._update_user_profile(user_id)

    def _classify_question(self, question: str) -> str:
        """질문 분류"""
        classification_prompt = f"""
        다음 질문을 주택청약 FAQ의 주요 카테고리로 분류해주세요:

        카테고리: 청약자격, 청약통장, 주택공급, 당첨규칙, 계약절차, 기타

        질문: {question}

        가장 적합한 카테고리 하나만 답해주세요.
        """

        response = self.llm.invoke(classification_prompt)
        category = response.content.strip()

        # 캐시에 저장
        self.question_categories[question] = category
        return category

    def _update_user_profile(self, user_id: str):
        """사용자 프로필 업데이트"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 사용자의 최근 활동 분석
        cursor.execute('''
            SELECT category, satisfaction_score, question, timestamp
            FROM user_history
            WHERE user_id = ? AND timestamp > datetime('now', '-30 days')
            ORDER BY timestamp DESC
        ''', (user_id,))

        recent_history = cursor.fetchall()

        if not recent_history:
            return

        # 선호 카테고리 분석
        category_scores = defaultdict(list)
        for category, score, question, timestamp in recent_history:
            category_scores[category].append(score)

        # 평균 만족도가 높은 카테고리들을 선호 카테고리로 설정
        preferred_categories = []
        for category, scores in category_scores.items():
            if np.mean(scores) >= 3.5:  # 5점 만점 기준
                preferred_categories.append(category)

        # 질문 복잡도 레벨 분석
        question_lengths = [len(q.split()) for _, _, q, _ in recent_history]
        avg_complexity = np.mean(question_lengths) if question_lengths else 5.0

        # 세션 길이 분석
        sessions = defaultdict(list)
        for _, _, _, timestamp in recent_history:
            date = timestamp[:10]  # YYYY-MM-DD
            sessions[date].append(timestamp)

        session_lengths = [len(timestamps) for timestamps in sessions.values()]
        avg_session_length = np.mean(session_lengths) if session_lengths else 1.0

        # 프로필 업데이트
        cursor.execute('''
            INSERT OR REPLACE INTO user_preferences
            (user_id, preferred_categories, question_complexity_level, avg_session_length, last_active)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            user_id,
            ','.join(preferred_categories),
            avg_complexity,
            avg_session_length,
            datetime.now()
        ))

        conn.commit()
        conn.close()

        # 메모리에 프로필 저장
        self.user_profiles[user_id] = {
            'preferred_categories': preferred_categories,
            'question_complexity_level': avg_complexity,
            'avg_session_length': avg_session_length,
            'last_active': datetime.now()
        }

    def get_personalized_recommendations(self, user_id: str, current_question: str = None, k: int = 5) -> List[Dict]:
        """개인화된 FAQ 추천"""

        # 사용자 프로필 로드
        if user_id not in self.user_profiles:
            self._load_user_profile(user_id)

        user_profile = self.user_profiles.get(user_id, {})

        # 추천 전략들
        recommendations = []

        # 1. 선호 카테고리 기반 추천
        category_recs = self._get_category_based_recommendations(user_profile, k//2)
        recommendations.extend(category_recs)

        # 2. 유사 사용자 기반 추천
        collaborative_recs = self._get_collaborative_recommendations(user_id, k//2)
        recommendations.extend(collaborative_recs)

        # 3. 현재 질문 기반 유사 질문 추천 (있는 경우)
        if current_question:
            similar_recs = self._get_similar_question_recommendations(current_question, k//2)
            recommendations.extend(similar_recs)

        # 4. 트렌딩 질문 추천
        trending_recs = self._get_trending_recommendations(k//2)
        recommendations.extend(trending_recs)

        # 중복 제거 및 개인화 점수 계산
        unique_recs = self._deduplicate_and_score(recommendations, user_profile)

        return sorted(unique_recs, key=lambda x: x['personalized_score'], reverse=True)[:k]

    def _load_user_profile(self, user_id: str):
        """데이터베이스에서 사용자 프로필 로드"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT preferred_categories, question_complexity_level, avg_session_length, last_active
            FROM user_preferences
            WHERE user_id = ?
        ''', (user_id,))

        result = cursor.fetchone()
        conn.close()

        if result:
            self.user_profiles[user_id] = {
                'preferred_categories': result[0].split(',') if result[0] else [],
                'question_complexity_level': result[1],
                'avg_session_length': result[2],
                'last_active': datetime.fromisoformat(result[3])
            }

    def _get_category_based_recommendations(self, user_profile: Dict, k: int) -> List[Dict]:
        """선호 카테고리 기반 추천"""
        preferred_categories = user_profile.get('preferred_categories', [])
        if not preferred_categories:
            return []

        # 선호 카테고리의 인기 질문들 추출
        recommendations = []
        for category in preferred_categories:
            # 실제로는 FAQ 데이터베이스에서 해당 카테고리의 인기 질문들을 가져옴
            category_questions = self._get_popular_questions_by_category(category, k//len(preferred_categories))
            recommendations.extend(category_questions)

        return recommendations

    def _get_collaborative_recommendations(self, user_id: str, k: int) -> List[Dict]:
        """협업 필터링 기반 추천"""
        # 사용자의 질문 히스토리를 벡터화
        user_questions = self._get_user_questions(user_id)
        if not user_questions:
            return []

        # 유사한 사용자들 찾기
        similar_users = self._find_similar_users(user_id, user_questions)

        # 유사 사용자들이 만족한 질문들 추천
        recommendations = []
        for similar_user, similarity in similar_users[:5]:
            user_satisfied_questions = self._get_satisfied_questions(similar_user)
            for question in user_satisfied_questions:
                recommendations.append({
                    'question': question['question'],
                    'category': question['category'],
                    'source': 'collaborative',
                    'similarity_score': similarity,
                    'base_score': question['satisfaction_score']
                })

        return recommendations[:k]

    def _find_similar_users(self, user_id: str, user_questions: List[str]) -> List[tuple]:
        """유사한 사용자들 찾기"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 다른 사용자들의 질문 히스토리 가져오기
        cursor.execute('''
            SELECT DISTINCT user_id FROM user_history WHERE user_id != ?
        ''', (user_id,))

        other_users = [row[0] for row in cursor.fetchall()]
        conn.close()

        user_similarities = []
        user_vector = self._vectorize_questions(user_questions)

        for other_user in other_users:
            other_questions = self._get_user_questions(other_user)
            if other_questions:
                other_vector = self._vectorize_questions(other_questions)
                similarity = cosine_similarity([user_vector], [other_vector])[0][0]
                user_similarities.append((other_user, similarity))

        return sorted(user_similarities, key=lambda x: x[1], reverse=True)

    def _vectorize_questions(self, questions: List[str]) -> np.ndarray:
        """질문들을 벡터로 변환"""
        combined_text = ' '.join(questions)
        try:
            vector = self.vectorizer.fit_transform([combined_text])
            return vector.toarray()[0]
        except:
            return np.zeros(1000)  # 기본 벡터 크기

    def _get_similar_question_recommendations(self, current_question: str, k: int) -> List[Dict]:
        """현재 질문과 유사한 질문들 추천"""
        # 임베딩을 사용한 유사 질문 검색
        # 실제 구현에서는 FAQ 벡터 스토어에서 유사 질문들을 검색
        recommendations = []

        # 예시 구현
        similar_questions = self._semantic_search(current_question, k)
        for question in similar_questions:
            recommendations.append({
                'question': question['content'],
                'category': question['category'],
                'source': 'semantic_similarity',
                'base_score': question.get('popularity_score', 0.5)
            })

        return recommendations

    def _get_trending_recommendations(self, k: int) -> List[Dict]:
        """트렌딩 질문 추천"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 최근 7일간 인기 질문들
        cursor.execute('''
            SELECT question, category, COUNT(*) as frequency, AVG(satisfaction_score) as avg_satisfaction
            FROM user_history
            WHERE timestamp > datetime('now', '-7 days')
            GROUP BY question
            ORDER BY frequency DESC, avg_satisfaction DESC
            LIMIT ?
        ''', (k,))

        trending_questions = cursor.fetchall()
        conn.close()

        recommendations = []
        for question, category, frequency, avg_satisfaction in trending_questions:
            recommendations.append({
                'question': question,
                'category': category,
                'source': 'trending',
                'base_score': (frequency * 0.3 + avg_satisfaction * 0.7) / 5.0
            })

        return recommendations

    def _deduplicate_and_score(self, recommendations: List[Dict], user_profile: Dict) -> List[Dict]:
        """중복 제거 및 개인화 점수 계산"""
        seen_questions = set()
        unique_recs = []
        preferred_categories = user_profile.get('preferred_categories', [])

        for rec in recommendations:
            question = rec['question']
            if question in seen_questions:
                continue

            seen_questions.add(question)

            # 개인화 점수 계산
            base_score = rec.get('base_score', 0.5)
            category_bonus = 0.2 if rec['category'] in preferred_categories else 0
            source_weight = {
                'collaborative': 0.3,
                'category_based': 0.25,
                'semantic_similarity': 0.2,
                'trending': 0.15
            }.get(rec['source'], 0.1)

            personalized_score = base_score + category_bonus + source_weight
            rec['personalized_score'] = personalized_score

            unique_recs.append(rec)

        return unique_recs

# 사용 예제
recommender = PersonalizedFAQRecommender(llm, embeddings)

# 사용자 상호작용 로깅
recommender.log_user_interaction(
    user_id="user_123",
    question="무주택세대구성원이 되려면 어떤 조건이 필요한가요?",
    satisfaction_score=4.5,
    session_id="session_001"
)

# 개인화된 추천 받기
recommendations = recommender.get_personalized_recommendations(
    user_id="user_123",
    current_question="청약통장을 만들려면 어떻게 해야 하나요?",
    k=5
)

print("개인화된 FAQ 추천:")
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. [{rec['category']}] {rec['question']}")
    print(f"   점수: {rec['personalized_score']:.2f} (출처: {rec['source']})")
```

### 실습 3 해답: 멀티모달 FAQ 시스템

```python
import pytesseract
from PIL import Image
import cv2
import pandas as pd
from langchain.schema import Document
import base64
import io

class MultimodalFAQSystem:
    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings
        self.ocr_config = r'--oem 3 --psm 6 -l kor+eng'

    def process_multimodal_document(self, file_path: str, doc_type: str) -> List[Document]:
        """다양한 형태의 문서 처리"""

        if doc_type == "image":
            return self._process_image_document(file_path)
        elif doc_type == "table":
            return self._process_table_document(file_path)
        elif doc_type == "mixed":
            return self._process_mixed_document(file_path)
        else:
            raise ValueError(f"Unsupported document type: {doc_type}")

    def _process_image_document(self, image_path: str) -> List[Document]:
        """이미지 문서 처리 (OCR)"""

        # 이미지 전처리
        processed_image = self._preprocess_image(image_path)

        # OCR 수행
        extracted_text = pytesseract.image_to_string(processed_image, config=self.ocr_config)

        # 텍스트 후처리
        cleaned_text = self._clean_ocr_text(extracted_text)

        # 이미지에서 구조 정보 추출
        layout_info = self._analyze_image_layout(processed_image)

        # 문서 객체 생성
        documents = []

        # 전체 텍스트 문서
        main_doc = Document(
            page_content=cleaned_text,
            metadata={
                'source': image_path,
                'type': 'image_ocr',
                'layout_info': layout_info,
                'confidence': self._estimate_ocr_confidence(extracted_text),
                'image_base64': self._image_to_base64(processed_image)
            }
        )
        documents.append(main_doc)

        # 섹션별 분할 (레이아웃 기반)
        if layout_info.get('sections'):
            section_docs = self._create_section_documents(cleaned_text, layout_info, image_path)
            documents.extend(section_docs)

        return documents

    def _preprocess_image(self, image_path: str):
        """이미지 전처리"""
        image = cv2.imread(image_path)

        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 노이즈 제거
        denoised = cv2.medianBlur(gray, 3)

        # 대비 향상
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)

        # 이진화
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def _clean_ocr_text(self, text: str) -> str:
        """OCR 텍스트 정제"""
        import re

        # 불필요한 공백 제거
        cleaned = re.sub(r'\s+', ' ', text)

        # 특수문자 정리
        cleaned = re.sub(r'[^\w\s가-힣.,!?()-]', '', cleaned)

        # 의미없는 단어 제거
        meaningless_patterns = [
            r'\b[a-zA-Z]{1,2}\b',  # 1-2글자 영문
            r'\b\d{1,2}\b(?!\d)',  # 단독 1-2자리 숫자
        ]

        for pattern in meaningless_patterns:
            cleaned = re.sub(pattern, '', cleaned)

        return cleaned.strip()

    def _analyze_image_layout(self, image) -> Dict:
        """이미지 레이아웃 분석"""
        # 컨투어를 사용한 텍스트 블록 검출
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        text_blocks = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 50 and h > 20:  # 최소 크기 필터
                text_blocks.append({
                    'bbox': (x, y, w, h),
                    'area': w * h,
                    'aspect_ratio': w / h
                })

        # 텍스트 블록을 크기와 위치로 정렬
        text_blocks.sort(key=lambda x: (x['bbox'][1], x['bbox'][0]))  # y축 우선, x축 보조

        return {
            'sections': text_blocks,
            'total_blocks': len(text_blocks),
            'image_size': image.shape
        }

    def _process_table_document(self, file_path: str) -> List[Document]:
        """표 데이터 처리"""

        # 파일 형식에 따른 로드
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported table format")

        documents = []

        # 전체 테이블 요약
        table_summary = self._generate_table_summary(df)
        main_doc = Document(
            page_content=table_summary,
            metadata={
                'source': file_path,
                'type': 'table',
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'table_data': df.to_dict('records')[:100]  # 처음 100행만 저장
            }
        )
        documents.append(main_doc)

        # 행별 문서 생성 (중요한 행만)
        important_rows = self._identify_important_rows(df)
        for idx, row in important_rows.iterrows():
            row_doc = Document(
                page_content=self._format_row_as_text(row, df.columns),
                metadata={
                    'source': file_path,
                    'type': 'table_row',
                    'row_index': idx,
                    'row_data': row.to_dict()
                }
            )
            documents.append(row_doc)

        # 컬럼별 통계 문서
        if df.select_dtypes(include=[np.number]).shape[1] > 0:
            stats_doc = self._create_statistics_document(df, file_path)
            documents.append(stats_doc)

        return documents

    def _generate_table_summary(self, df: pd.DataFrame) -> str:
        """테이블 요약 생성"""
        summary_parts = []

        # 기본 정보
        summary_parts.append(f"이 테이블은 {df.shape[0]}행 {df.shape[1]}열의 데이터를 포함합니다.")

        # 컬럼 정보
        summary_parts.append(f"컬럼: {', '.join(df.columns.tolist())}")

        # 수치형 데이터 요약
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols[:3]:  # 처음 3개 수치형 컬럼만
                mean_val = df[col].mean()
                summary_parts.append(f"{col}의 평균값: {mean_val:.2f}")

        # 범주형 데이터 요약
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols[:3]:  # 처음 3개 범주형 컬럼만
                top_value = df[col].value_counts().index[0]
                summary_parts.append(f"{col}에서 가장 많은 값: {top_value}")

        return ' '.join(summary_parts)

    def _identify_important_rows(self, df: pd.DataFrame, max_rows: int = 10) -> pd.DataFrame:
        """중요한 행 식별"""
        # 간단한 휴리스틱: 수치값이 극값인 행들
        important_indices = set()

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # 최대값, 최소값을 가진 행
            max_idx = df[col].idxmax()
            min_idx = df[col].idxmin()
            if pd.notna(max_idx):
                important_indices.add(max_idx)
            if pd.notna(min_idx):
                important_indices.add(min_idx)

        # 무작위로 몇 개 더 선택
        remaining_indices = set(df.index) - important_indices
        if len(remaining_indices) > 0:
            additional_count = min(max_rows - len(important_indices), len(remaining_indices))
            additional_indices = np.random.choice(list(remaining_indices), additional_count, replace=False)
            important_indices.update(additional_indices)

        return df.loc[list(important_indices)]

    def _format_row_as_text(self, row: pd.Series, columns: List[str]) -> str:
        """행을 자연어 텍스트로 변환"""
        text_parts = []
        for col, val in row.items():
            if pd.notna(val):
                text_parts.append(f"{col}: {val}")

        return ", ".join(text_parts)

    def multimodal_search(self, query: str, content_types: List[str] = None) -> List[Dict]:
        """멀티모달 검색"""
        if content_types is None:
            content_types = ['text', 'image', 'table']

        search_results = []

        # 텍스트 기반 검색
        if 'text' in content_types:
            text_results = self._text_search(query)
            search_results.extend(text_results)

        # 이미지 기반 검색 (OCR 텍스트 대상)
        if 'image' in content_types:
            image_results = self._image_search(query)
            search_results.extend(image_results)

        # 테이블 기반 검색
        if 'table' in content_types:
            table_results = self._table_search(query)
            search_results.extend(table_results)

        # 멀티모달 관련성 평가
        evaluated_results = self._evaluate_multimodal_relevance(search_results, query)

        return sorted(evaluated_results, key=lambda x: x['relevance_score'], reverse=True)

    def _evaluate_multimodal_relevance(self, results: List[Dict], query: str) -> List[Dict]:
        """멀티모달 콘텐츠의 관련성 평가"""

        for result in results:
            content_type = result['metadata']['type']

            # 콘텐츠 타입별 관련성 평가
            if content_type == 'image_ocr':
                relevance = self._evaluate_image_relevance(result, query)
            elif content_type == 'table':
                relevance = self._evaluate_table_relevance(result, query)
            else:
                relevance = self._evaluate_text_relevance(result, query)

            result['relevance_score'] = relevance
            result['confidence'] = result['metadata'].get('confidence', 0.8)

        return results

    def _evaluate_image_relevance(self, result: Dict, query: str) -> float:
        """이미지 콘텐츠 관련성 평가"""

        # OCR 신뢰도 고려
        ocr_confidence = result['metadata'].get('confidence', 0.5)

        # 텍스트 유사도
        text_similarity = self._calculate_text_similarity(result['page_content'], query)

        # 레이아웃 정보 고려 (구조화된 문서일수록 가중치)
        layout_info = result['metadata'].get('layout_info', {})
        structure_bonus = min(0.2, layout_info.get('total_blocks', 0) * 0.05)

        return (text_similarity * ocr_confidence) + structure_bonus

    def _evaluate_table_relevance(self, result: Dict, query: str) -> float:
        """테이블 콘텐츠 관련성 평가"""

        # 텍스트 유사도 (요약문 기준)
        text_similarity = self._calculate_text_similarity(result['page_content'], query)

        # 테이블 메타데이터 매칭
        metadata = result['metadata']
        columns = metadata.get('columns', [])

        # 쿼리의 키워드가 컬럼명에 포함되는지 확인
        query_words = query.lower().split()
        column_matches = sum(1 for col in columns
                           for word in query_words
                           if word in col.lower())

        column_bonus = min(0.3, column_matches * 0.1)

        return text_similarity + column_bonus

    def generate_multimodal_answer(self, query: str, search_results: List[Dict]) -> str:
        """멀티모달 검색 결과를 기반으로 답변 생성"""

        # 콘텐츠 타입별 정보 분리
        text_content = []
        image_content = []
        table_content = []

        for result in search_results[:5]:  # 상위 5개 결과만 사용
            content_type = result['metadata']['type']

            if content_type == 'image_ocr':
                image_content.append({
                    'text': result['page_content'],
                    'confidence': result['metadata'].get('confidence', 0.8),
                    'source': result['metadata']['source']
                })
            elif content_type in ['table', 'table_row']:
                table_content.append({
                    'summary': result['page_content'],
                    'data': result['metadata'].get('table_data', {}),
                    'source': result['metadata']['source']
                })
            else:
                text_content.append(result['page_content'])

        # 멀티모달 답변 생성 프롬프트
        multimodal_prompt = f"""
        다음 다양한 형태의 정보를 종합하여 질문에 답변해주세요:

        질문: {query}

        텍스트 정보:
        {' '.join(text_content) if text_content else '없음'}

        이미지에서 추출한 정보:
        {self._format_image_content(image_content) if image_content else '없음'}

        표 데이터 정보:
        {self._format_table_content(table_content) if table_content else '없음'}

        위의 정보들을 종합하여 정확하고 포괄적인 답변을 제공해주세요.
        각 정보 출처의 신뢰도도 고려하여 답변해주세요.
        """

        response = self.llm.invoke(multimodal_prompt)

        return response.content

    def _format_image_content(self, image_content: List[Dict]) -> str:
        """이미지 콘텐츠 포맷팅"""
        formatted = []
        for img in image_content:
            confidence_level = "높음" if img['confidence'] > 0.8 else "중간" if img['confidence'] > 0.6 else "낮음"
            formatted.append(f"- 출처: {img['source']} (신뢰도: {confidence_level})\n  내용: {img['text']}")

        return '\n'.join(formatted)

    def _format_table_content(self, table_content: List[Dict]) -> str:
        """테이블 콘텐츠 포맷팅"""
        formatted = []
        for table in table_content:
            formatted.append(f"- 출처: {table['source']}\n  요약: {table['summary']}")

            # 구체적인 데이터 포함 (처음 3개 행만)
            if table.get('data'):
                data_sample = table['data'][:3]
                for i, row in enumerate(data_sample, 1):
                    row_text = ', '.join([f"{k}: {v}" for k, v in row.items() if v is not None])
                    formatted.append(f"  데이터 {i}: {row_text}")

        return '\n'.join(formatted)

# 사용 예제
multimodal_system = MultimodalFAQSystem(llm, embeddings)

# 이미지 문서 처리
image_docs = multimodal_system.process_multimodal_document("housing_chart.png", "image")

# 테이블 문서 처리
table_docs = multimodal_system.process_multimodal_document("housing_stats.xlsx", "table")

# 멀티모달 검색
query = "아파트 가격 동향과 청약 경쟁률은 어떻게 되나요?"
search_results = multimodal_system.multimodal_search(query)

# 멀티모달 답변 생성
answer = multimodal_system.generate_multimodal_answer(query, search_results)
print("멀티모달 답변:", answer)
```

## 참고 자료

### 공식 문서
- [LangChain Document Loaders](https://python.langchain.com/docs/modules/data_connection/document_loaders/)
- [Chroma Vector Database](https://docs.trychroma.com/)
- [Gradio ChatInterface](https://gradio.app/docs/#chatinterface)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)

### 심화 학습
- [RAG 시스템 최적화](https://python.langchain.com/docs/use_cases/question_answering/)
- [메타데이터 필터링](https://python.langchain.com/docs/modules/data_connection/vectorstores/)
- [MMR 검색 알고리즘](https://python.langchain.com/docs/modules/data_connection/retrievers/)
- [Gradio 고급 기능](https://gradio.app/guides/)

### 관련 기술
- Tesseract OCR: 이미지 텍스트 추출
- OpenCV: 이미지 전처리
- Pandas: 데이터 테이블 처리
- Scikit-learn: 머신러닝 알고리즘

이 가이드를 통해 실제 운영 가능한 수준의 FAQ 챗봇 시스템을 구축할 수 있으며, 다양한 고급 기능들을 통해 사용자 경험을 크게 향상시킬 수 있습니다.