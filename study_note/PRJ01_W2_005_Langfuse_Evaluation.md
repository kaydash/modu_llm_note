# W2_005 Langfuse Evaluation을 사용한 RAG 답변 평가

## 학습 목표
- Langfuse 플랫폼을 활용한 체계적인 RAG 시스템 평가 방법 학습
- 데이터셋 기반 자동화 평가 파이프라인 구축
- ROUGE 점수와 LLM-as-Judge 평가 지표 통합 활용
- Langfuse 대시보드를 통한 실시간 모니터링 및 분석

## 핵심 개념

### 1. Langfuse란?
- **정의**: LLM 애플리케이션을 위한 오픈소스 관측성(Observability) 플랫폼
- **목적**: RAG 시스템의 성능을 체계적으로 평가하고 지속적으로 모니터링
- **특징**: 자동화된 추적, 데이터셋 관리, 평가 지표 수집, 시각적 대시보드

### 2. Langfuse의 주요 장점
- 🔄 **자동화된 추적 및 로깅**: CallbackHandler를 통한 투명한 실행 기록
- 📊 **시각적 대시보드**: 직관적인 성능 분석 인터페이스
- 🔍 **다양한 평가 지표**: ROUGE, LLM-as-Judge 등 통합 지원
- 🚀 **확장 가능한 파이프라인**: 대규모 평가 자동화 가능

### 3. 평가 프로세스
1. **환경 설정**: Langfuse 클라이언트 초기화 및 인증
2. **데이터셋 업로드**: `create_dataset()` 및 `create_dataset_item()` 사용
3. **RAG 체인 실행**: CallbackHandler로 자동 추적 설정
4. **평가 실행**: 데이터셋 기반 체계적 평가
5. **결과 분석**: 대시보드를 통한 성능 분석

## 환경 설정

### 1. 필수 라이브러리 설치
```bash
pip install langfuse langchain langchain-openai langchain-chroma
pip install korouge-score krag pandas openpyxl
```

### 2. 환경 변수 설정
```python
# .env 파일
OPENAI_API_KEY=your_openai_api_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
LANGSMITH_TRACING=true
```

### 3. 기본 설정
```python
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import json
from pprint import pprint
import warnings

# 환경 변수 로드
load_dotenv()
warnings.filterwarnings("ignore")

# Langsmith 추적 확인
print("langsmith 추적 여부:", os.getenv('LANGSMITH_TRACING'))
```

## 1단계: 테스트 데이터 준비

### 테스트 데이터셋 로드
```python
# 테스트 데이터셋 로드
df_qa_test = pd.read_excel("data/testset.xlsx")
print(f"테스트셋: {df_qa_test.shape[0]}개 문서")

# 데이터 구조 확인
df_qa_test.head(2)
```

**데이터셋 구조:**
- `user_input`: 사용자 질문
- `reference`: 정답 참조
- `reference_contexts`: 참조 문맥
- `synthesizer_name`: 데이터 생성 방식

## 2단계: 검색 도구 정의

### 1. 벡터 스토어 로드
```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# 임베딩 모델 설정
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Chroma DB 로드
chroma_db = Chroma(
    collection_name="db_korean_cosine_metadata",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

# 벡터 검색기 생성
chroma_k = chroma_db.as_retriever(search_kwargs={'k': 4})

# 검색 테스트
query = "테슬라의 회장은 누구인가요?"
retrieved_docs = chroma_k.invoke(query)

for doc in retrieved_docs:
    print(f"- {doc.page_content} [출처: {doc.metadata['source']}]")
```

### 2. BM25 검색기 준비
```python
from krag.tokenizers import KiwiTokenizer
from krag.retrievers import KiWiBM25RetrieverWithScore
from langchain.schema import Document

# 문서 로드 함수
def load_jsonlines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        docs = [json.loads(line) for line in f]
    return docs

# 한국어 문서 로드
korean_docs = load_jsonlines('data/korean_docs_final.jsonl')
print(f"로드된 문서: {len(korean_docs)}개")

# Document 객체로 변환
documents = []
for data in korean_docs:
    if isinstance(data, str):
        doc_data = json.loads(data)
    else:
        doc_data = data

    documents.append(Document(
        page_content=doc_data['page_content'],
        metadata=doc_data['metadata']
    ))

print(f"변환된 문서: {len(documents)}개")

# BM25 검색기 설정
kiwi_tokenizer = KiwiTokenizer(
    model_type='knlm',    # Kiwi 언어 모델 타입
    typos='basic'         # 기본 오타교정
)

bm25_db = KiWiBM25RetrieverWithScore(
    documents=documents,
    kiwi_tokenizer=kiwi_tokenizer,
    k=4,
)

# BM25 검색 테스트
retrieved_docs = bm25_db.invoke(query)
for doc in retrieved_docs:
    print(f"BM25 점수: {doc.metadata['bm25_score']:.2f}")
    print(f"{doc.page_content}")
```

### 3. 하이브리드 검색 구성
```python
from langchain.retrievers import EnsembleRetriever

# 하이브리드 검색기 초기화
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_db, chroma_k],
    weights=[0.5, 0.5],  # BM25와 벡터 검색의 가중치
)

# 하이브리드 검색 테스트
retrieved_docs = hybrid_retriever.invoke(query)
for doc in retrieved_docs:
    print(f"{doc.page_content}\n[출처: {doc.metadata['source']}]")
```

## 3단계: RAG Chain 정의

### RAG 체인 함수 생성
```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

def create_rag_chain(retriever, llm):
    """RAG 체인 생성 함수"""

    template = """Answer the following question based on this context.
    If the context is not relevant to the question, just answer with '답변에 필요한 근거를 찾지 못했습니다.'

    [Context]
    {context}

    [Question]
    {question}

    [Answer]
    """

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join([f"{doc.page_content}" for doc in docs])

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

# RAG 체인 생성 및 테스트
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.5)
openai_rag_chain = create_rag_chain(hybrid_retriever, llm)

# 테스트 실행
question = "테슬라의 회장은 누구인가요?"
answer = openai_rag_chain.invoke(question)

print(f"쿼리: {question}")
print(f"답변: {answer}")
```

## 4단계: Langfuse 환경 설정

### 1. Langfuse 클라이언트 초기화
```python
from langfuse.langchain import CallbackHandler
from langfuse import get_client

# Langfuse 콜백 핸들러 생성
langfuse_handler = CallbackHandler()

# Langfuse 클라이언트 초기화
langfuse_client = get_client()

# 인증 확인
print("Langfuse 인증 상태:", langfuse_client.auth_check())
```

## 5단계: 평가용 데이터셋 업로드

### 데이터셋 생성 및 아이템 추가
```python
# 데이터셋 생성
dataset_name = "RAG_Evaluation_Dataset_Test"
dataset = langfuse_client.create_dataset(name=dataset_name)
print(f"생성된 데이터셋: {dataset.name}")

# 평가용 데이터셋 변환
data = [
    {
        "user_input": row["user_input"],
        "reference": row["reference"],
        "reference_contexts": row["reference_contexts"],
    }
    for _, row in df_qa_test.iterrows()
]

print(f"평가용 데이터셋 아이템 수: {len(data)}개")

# 데이터셋 아이템 추가
for item in data:
    langfuse_client.create_dataset_item(
        dataset_name=dataset_name,
        input=item.get("user_input", ""),
        expected_output=item.get("reference", ""),
        metadata={
            "reference_contexts": item.get("reference_contexts", ""),
        }
    )

# Langfuse에 저장
langfuse_client.flush()

# 데이터셋 확인
dataset = langfuse_client.get_dataset(name=dataset_name)
print(f"생성된 데이터셋: {dataset.name}")
print(f"데이터셋 아이템 수: {len(dataset.items)}개")
```

### 데이터셋 아이템 출력
```python
# 처음 5개 아이템 확인
for item in dataset.items[:5]:
    print(f"입력: {item.input}")
    print(f"기대 출력: {item.expected_output}")
    print(f"메타데이터: {item.metadata}")
    print("-" * 200)
```

## 6단계: 평가 지표 설정

### ROUGE 스코어 및 간결성 평가자 설정
```python
from langchain.evaluation import load_evaluator
from korouge_score import rouge_scorer
from krag.tokenizers import KiwiTokenizer

# Kiwi 토크나이저 사용하여 토큰화하는 클래스 정의
class CustomKiwiTokenizer(KiwiTokenizer):
    def tokenize(self, text):
        return [t.form for t in super().tokenize(text)]

# 토크나이저 생성
kiwi_tokenizer = CustomKiwiTokenizer(model_type='knlm', typos='basic')

# ROUGE 스코어 계산기
scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    tokenizer=kiwi_tokenizer
)

# 간결성 평가자 로드
conciseness_evaluator = load_evaluator(
    evaluator="labeled_criteria",
    criteria="conciseness",
    llm=llm
)
```

## 7단계: 데이터셋 기반 평가 실행

### 평가 결과 데이터 클래스 정의
```python
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class EvaluationResult:
    """평가 결과를 담는 데이터 클래스"""
    item_id: str
    input: Any
    output: str
    expected_output: str
    scores: Dict[str, float]
    details: Dict[str, Any]
    trace_id: Optional[str] = None
    error: Optional[str] = None
```

### 평가 실행 함수
```python
def run_dataset_evaluation(
    dataset_name: str,
    rag_chain,
    run_name: str
) -> List[EvaluationResult]:
    """데이터셋 전체에 대한 평가 실행"""

    # 데이터셋 가져오기
    langfuse_client = get_client()
    dataset = langfuse_client.get_dataset(name=dataset_name)

    if not dataset:
        raise ValueError(f"데이터셋 '{dataset_name}'이(가) 존재하지 않습니다.")

    print(f"📊 RAG 평가 시작: {dataset_name} ({len(dataset.items)}개 항목)")

    results = []
    successful = 0
    failed = 0

    for idx, item in enumerate(dataset.items, 1):
        try:
            print(f"\n🔄 아이템 {idx}/{len(dataset.items)} 처리 중...")

            # Langfuse 트레이싱 설정
            with item.run(run_name=run_name) as root_span:

                # RAG 체인 실행
                output = rag_chain.invoke(
                    item.input,
                    config={"callbacks": [CallbackHandler()]}
                )

                # 평가 수행
                scores, details = {}, {}

                # 1. ROUGE 점수 평가
                try:
                    rouge_results = scorer.score(
                        str(item.expected_output),
                        str(output)
                    )
                    rouge_scores = {
                        "rouge1": rouge_results['rouge1'].fmeasure,
                        "rouge2": rouge_results['rouge2'].fmeasure,
                        "rougeL": rouge_results['rougeL'].fmeasure
                    }
                    scores["rouge"] = sum(rouge_scores.values()) / len(rouge_scores)
                    details["rouge"] = rouge_scores
                except Exception as e:
                    scores["rouge"] = 0.0
                    details["rouge"] = {"error": str(e)}

                # 2. 간결성 평가
                try:
                    conciseness_result = conciseness_evaluator.evaluate_strings(
                        input=str(item.input),
                        prediction=str(output),
                        reference=str(item.expected_output)
                    )
                    scores["conciseness"] = float(conciseness_result.get('score', 0))
                    details["conciseness"] = {
                        "reasoning": conciseness_result.get('reasoning', ''),
                        "score": conciseness_result.get('score', 0)
                    }
                except Exception as e:
                    scores["conciseness"] = 0.0
                    details["conciseness"] = {"error": str(e)}

                # 전체 점수 계산 및 기록
                overall_score = sum(scores.values()) / len(scores)
                root_span.score(name="overall", value=overall_score)

                # 각 평가 점수 기록
                for score_name, score_value in scores.items():
                    root_span.score(name=score_name, value=score_value)

                # 결과 저장
                result = EvaluationResult(
                    item_id=item.id,
                    input=item.input,
                    output=str(output),
                    expected_output=str(item.expected_output) if item.expected_output else "",
                    scores=scores,
                    details=details,
                    trace_id=getattr(root_span, 'trace_id', None)
                )
                results.append(result)
                successful += 1

                print(f"   ✅ 완료 (종합 점수: {overall_score:.2f})")
                print(f"   🔍 세부 정보: {details}")

        except Exception as e:
            failed += 1
            print(f"   ❌ 실패: {str(e)}")

            # 실패해도 결과에 기록
            results.append(EvaluationResult(
                item_id=item.id,
                input=item.input,
                output="",
                expected_output=str(item.expected_output) if item.expected_output else "",
                scores={},
                details={},
                error=str(e)
            ))

    # 결과 요약
    print(f"\n📋 평가 완료: 성공 {successful}개, 실패 {failed}개")

    return results
```

### 평가 실행
```python
# 평가 실행
results = run_dataset_evaluation(
    dataset_name="RAG_Evaluation_Dataset_Test",
    rag_chain=openai_rag_chain,
    run_name="simple_evaluation_v1"
)

print(f"\n평가 완료: {len(results)}개 항목")
```

## 8단계: 결과 분석

### 평가 결과 통계
```python
# 성공한 결과만 필터링
successful_results = [r for r in results if not r.error]

# ROUGE 점수 분석
rouge_scores = [r.scores.get('rouge', 0) for r in successful_results]
conciseness_scores = [r.scores.get('conciseness', 0) for r in successful_results]

print("📊 평가 결과 통계:")
print(f"   - 전체 평가: {len(results)}개")
print(f"   - 성공: {len(successful_results)}개")
print(f"   - 실패: {len(results) - len(successful_results)}개")
print(f"\n   - ROUGE 평균 점수: {np.mean(rouge_scores):.3f}")
print(f"   - ROUGE 표준편차: {np.std(rouge_scores):.3f}")
print(f"\n   - 간결성 평균 점수: {np.mean(conciseness_scores):.3f}")
print(f"   - 간결성 표준편차: {np.std(conciseness_scores):.3f}")
print(f"\n   - 전체 평균 점수: {np.mean([np.mean([r, c]) for r, c in zip(rouge_scores, conciseness_scores)]):.3f}")
```

### 상위/하위 성능 분석
```python
# 종합 점수 계산
for result in successful_results:
    if result.scores:
        result.overall_score = np.mean(list(result.scores.values()))

# 점수 기준 정렬
sorted_results = sorted(
    successful_results,
    key=lambda x: getattr(x, 'overall_score', 0),
    reverse=True
)

# 상위 3개
print("\n🏆 상위 3개 결과:")
for i, result in enumerate(sorted_results[:3], 1):
    print(f"\n{i}. 질문: {result.input}")
    print(f"   답변: {result.output}")
    print(f"   종합 점수: {result.overall_score:.3f}")

# 하위 3개
print("\n⚠️ 하위 3개 결과:")
for i, result in enumerate(sorted_results[-3:], 1):
    print(f"\n{i}. 질문: {result.input}")
    print(f"   답변: {result.output}")
    print(f"   종합 점수: {result.overall_score:.3f}")
```

## 실습 과제

### 기본 실습
1. **Langfuse 데이터셋 생성**
   - 자신만의 테스트 데이터셋 생성 (최소 5개 항목)
   - RAG 체인으로 답변 생성
   - Langfuse 대시보드에서 결과 확인

2. **평가 지표 추가**
   - Helpfulness 평가자 추가
   - Relevance 평가자 추가
   - 다중 평가 지표로 종합 평가

### 응용 실습
3. **사용자 정의 평가 기준**
   - 한국어 특화 평가 기준 개발
   - 도메인 특화 평가자 구현
   - 기존 지표와 성능 비교

4. **다양한 검색기 비교**
   - Vector Search, BM25, Hybrid Search 각각 평가
   - 검색기별 성능 차이 분석
   - 최적 검색기 조합 탐색

### 심화 실습
5. **모델 비교 평가**
   - 여러 LLM 모델(GPT-4, Gemini 등) 성능 비교
   - 동일 데이터셋으로 A/B 테스트
   - 비용 대비 성능 분석

6. **실시간 모니터링 시스템**
   - 성능 임계값 설정 및 모니터링
   - 성능 저하 시 자동 알림
   - 지속적 개선 파이프라인 구축

## 문제 해결 가이드

### 일반적인 오류들
1. **Langfuse 인증 오류**
   ```python
   # 환경 변수 확인
   print("LANGFUSE_PUBLIC_KEY:", bool(os.getenv("LANGFUSE_PUBLIC_KEY")))
   print("LANGFUSE_SECRET_KEY:", bool(os.getenv("LANGFUSE_SECRET_KEY")))
   print("LANGFUSE_HOST:", os.getenv("LANGFUSE_HOST"))
   ```

2. **데이터셋 생성 오류**
   ```python
   # 기존 데이터셋 확인
   try:
       existing_dataset = langfuse_client.get_dataset(name=dataset_name)
       print(f"데이터셋 '{dataset_name}'이 이미 존재합니다.")
   except:
       print(f"데이터셋 '{dataset_name}'을 생성합니다.")
   ```

3. **평가 실행 오류**
   ```python
   # 개별 아이템 테스트
   test_item = dataset.items[0]
   try:
       test_output = rag_chain.invoke(test_item.input)
       print("테스트 성공:", test_output)
   except Exception as e:
       print("테스트 실패:", e)
   ```

## 참고 자료
- [Langfuse 공식 문서](https://langfuse.com/docs)
- [Langfuse LangChain 통합](https://langfuse.com/docs/integrations/langchain)
- [ROUGE Score 계산](https://github.com/neural-dialogue-metrics/rouge)
- [한국어 ROUGE 구현](https://github.com/gucci-j/korouge-score)
- [LangChain 평가 가이드](https://python.langchain.com/docs/guides/evaluation/)

이 학습 가이드를 통해 Langfuse를 활용한 체계적인 RAG 평가 시스템을 구축하고, 데이터 기반으로 시스템 성능을 지속적으로 개선할 수 있습니다.