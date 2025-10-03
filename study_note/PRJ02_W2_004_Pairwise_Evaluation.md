# W2_004 LLM 성능평가 개요 - A/B 테스트를 통한 비교 평가

## 학습 목표
- A/B 테스트를 수행하여 LLM 애플리케이션의 성능 평가를 적용한다
- Reference-free와 Reference-based 평가 방법론을 이해하고 구현한다
- 다양한 모델과 프롬프트를 비교 평가하여 최적화 방안을 도출한다
- LangChain 평가기를 활용한 체계적인 성능 분석을 수행한다

## 핵심 개념

### 1. LLM 애플리케이션 평가의 핵심 요소
- **데이터셋**: 평가를 위한 고품질 예제 (초기 10-20개부터 시작)
- **평가자**: 인간 평가와 자동화 평가의 적절한 조합
- **평가 방법론**: 상황에 맞는 평가 기준과 지표 선택

### 2. 평가 방식의 분류
- **인간 평가**: 주관적 판단이 필요한 초기 단계
- **자동화 평가**: 확장이 필요한 경우 휴리스틱 기반 평가
- **오프라인 평가**: 벤치마킹, 테스트
- **온라인 평가**: 실시간 모니터링

### 3. A/B 테스트 평가 유형
#### Reference-free 평가
- **특징**: 참조 답변 없이 두 RAG 답변 직접 비교
- **평가 요소**: 사실성, 관련성, 일관성 등 상대 비교
- **장점**: 절대 기준 없이도 RAG 시스템 간 성능 차이 판단 가능

#### Reference-based 평가
- **특징**: 참조 답안과 RAG 응답을 비교 평가
- **평가 방식**: 자동화된 A/B 테스트로 객관적 성능 측정
- **주요 지표**: 정확도, 완성도, 관련성 등 정량적 평가

## 환경 설정

### 1. 필수 라이브러리 설치
```bash
pip install langchain langchain-openai langchain-google-genai
pip install langchain-chroma langchain-community
pip install langfuse pandas openpyxl
pip install krag  # 한국어 BM25 검색기
```

### 2. 환경 변수 설정
```python
# .env 파일
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 3. 기본 설정
```python
from dotenv import load_dotenv
import os
import pandas as pd
import json
from pprint import pprint

# 환경 변수 로드
load_dotenv()

# Langfuse 콜백 핸들러 설정
from langfuse.langchain import CallbackHandler
langfuse_handler = CallbackHandler()

print("환경 설정 완료!")
```

## 1단계: 데이터 준비

### 테스트 데이터셋 로드
```python
# 테스트 데이터셋 로드
df_qa_test = pd.read_excel("data/testset.xlsx")
print(f"테스트셋: {df_qa_test.shape[0]}개 문서")

# 데이터 구조 확인
df_qa_test.head(2)
```

### 데이터셋 구조
- `user_input`: 사용자 질문
- `reference`: 참조 답변
- `reference_contexts`: 참조 문맥
- `synthesizer_name`: 데이터 생성 방식

## 2단계: 검색 시스템 구성

### 1. 벡터 스토어 설정
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
query = "Elon Musk는 Tesla의 초기 자금 조달과 경영 변화에 어떻게 관여했으며, 그 과정에서 어떤 논란에 직면했나요?"
retrieved_docs = chroma_k.invoke(query)

for doc in retrieved_docs:
    print(f"- {doc.page_content} [출처: {doc.metadata['source']}]")
```

### 2. BM25 검색기 구성
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

# BM25 검색기 설정
kiwi_tokenizer = KiwiTokenizer(model_type='knlm', typos='basic')
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

### 3. 하이브리드 검색기 구성
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

## 3단계: RAG 체인 구현

### RAG 봇 함수 정의
```python
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig, RunnablePassthrough, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict

def rag_bot(
    question: str,
    retriever: BaseRetriever,
    llm: BaseChatModel,
    config: RunnableConfig | None = None,
) -> Dict[str, str | List[Document]]:
    """
    문서 검색 기반 질의응답 수행
    """
    # 문서 검색
    docs = retriever.invoke(question)
    context = "\n".join(doc.page_content for doc in docs)

    # 시스템 프롬프트 설정
    system_prompt = f"""문서 기반 질의응답 어시스턴트입니다.
- 제공된 문서만 참고하여 답변
- 불확실할 경우 '모르겠습니다' 라고 응답
- 3문장 이내로 답변

[문서]
{context}"""

    # 프롬프트 템플릿 생성
    prompt = ChatPromptTemplate.from_messages([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n[질문]{question}\n\n[답변]\n"},
    ])

    # RAG 체인 구성
    docqa_chain = {
        "context": lambda x: context,
        "question": RunnablePassthrough(),
        "docs": lambda x: docs,
    } | RunnableParallel({
        "answer": prompt | llm | StrOutputParser(),
        "documents": lambda x: x["docs"],
    })

    return docqa_chain.invoke(question, config=config)

# 기본 테스트
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
result = rag_bot(
    question="Elon Musk는 Tesla의 초기 자금 조달과 경영 변화에 어떻게 관여했으며, 그 과정에서 어떤 논란에 직면했나요?",
    retriever=hybrid_retriever,
    llm=llm,
    config={"callbacks": [langfuse_handler]},
)

print("답변:", result["answer"])
```

## 4단계: Reference-free A/B 테스트

### 1. 기본 모델 비교
```python
from langchain.evaluation import load_evaluator
from langchain_google_genai import ChatGoogleGenerativeAI

# 평가용 LLM 설정
evaluator_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    api_key=os.getenv("GOOGLE_API_KEY")
)

# 비교 평가자 로드
comparison_evaluator = load_evaluator(
    "pairwise_string",
    llm=evaluator_llm
)

# 두 모델 설정
gpt_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0)

# 같은 질문에 대한 두 모델의 답변 생성
question = "Tesla의 주요 제품과 서비스는 무엇인가요?"

# GPT 답변
gpt_response = rag_bot(
    question=question,
    retriever=chroma_k,
    llm=gpt_llm,
    config={"callbacks": [langfuse_handler]}
)

# Gemini 답변
gemini_response = rag_bot(
    question=question,
    retriever=chroma_k,
    llm=gemini_llm,
    config={"callbacks": [langfuse_handler]}
)

# A/B 테스트 평가
evaluation_result = comparison_evaluator.evaluate_string_pairs(
    prediction=gpt_response["answer"],
    prediction_b=gemini_response["answer"],
    input=question
)

print("평가 결과:")
print(f"선호되는 응답: {evaluation_result['value']}")
print(f"점수: {evaluation_result['score']}")
print(f"평가 근거: {evaluation_result['reasoning']}")
```

### 2. 프롬프트 비교 평가
```python
# 다양한 스타일의 RAG 봇 함수들
def rag_bot_korean_style(question: str, retriever: BaseRetriever, llm: BaseChatModel, config=None):
    """한국어 친화적 스타일의 RAG 봇"""
    docs = retriever.invoke(question)
    context = "\n".join(doc.page_content for doc in docs)

    system_prompt = f"""안녕하세요! 저는 문서 기반 질의응답 도우미입니다.
- 주어진 문서 내용을 바탕으로 정중하고 친근하게 답변드립니다
- 확실하지 않은 내용은 "잘 모르겠습니다"라고 솔직하게 말씀드립니다
- 이해하기 쉽게 2-3문장으로 설명드리겠습니다

[참고 문서]
{context}"""

    prompt = ChatPromptTemplate.from_messages([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"질문: {question}\n\n답변을 부탁드립니다."},
    ])

    chain = prompt | llm | StrOutputParser()
    return {"answer": chain.invoke({"question": question}, config=config)}

def rag_bot_business_style(question: str, retriever: BaseRetriever, llm: BaseChatModel, config=None):
    """비즈니스 전문적 스타일의 RAG 봇"""
    docs = retriever.invoke(question)
    context = "\n".join(doc.page_content for doc in docs)

    system_prompt = f"""전문 비즈니스 컨설턴트로서 답변합니다.
- 제공된 문서를 근거로 정확하고 간결한 분석을 제공합니다
- 데이터가 불충분한 경우 명시적으로 언급합니다
- 핵심 포인트를 명확하게 전달합니다

[분석 자료]
{context}"""

    prompt = ChatPromptTemplate.from_messages([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"[Query] {question}\n\n[Analysis]"},
    ])

    chain = prompt | llm | StrOutputParser()
    return {"answer": chain.invoke({"question": question}, config=config)}

# 프롬프트 스타일 비교
question = "Tesla의 경쟁 우위는 무엇인가요?"

korean_response = rag_bot_korean_style(question, hybrid_retriever, gpt_llm, {"callbacks": [langfuse_handler]})
business_response = rag_bot_business_style(question, hybrid_retriever, gpt_llm, {"callbacks": [langfuse_handler]})

# 프롬프트 스타일 평가
style_evaluation = comparison_evaluator.evaluate_string_pairs(
    prediction=korean_response["answer"],
    prediction_b=business_response["answer"],
    input=question
)

print("프롬프트 스타일 비교 결과:")
print(f"선호되는 응답: {style_evaluation['value']}")
print(f"평가 근거: {style_evaluation['reasoning']}")
```

## 5단계: Reference-based A/B 테스트

### 1. 참조 답변 기반 평가
```python
# 테스트셋에서 샘플 선택
sample_idx = 0
sample = df_qa_test.iloc[sample_idx]

question = sample["user_input"]
reference = sample["reference"]

print(f"질문: {question}")
print(f"참조 답변: {reference}")

# 두 모델의 답변 생성
gpt_response = rag_bot(question, hybrid_retriever, gpt_llm, {"callbacks": [langfuse_handler]})
gemini_response = rag_bot(question, hybrid_retriever, gemini_llm, {"callbacks": [langfuse_handler]})

# Reference-based 평가
ref_based_evaluator = load_evaluator(
    "labeled_pairwise_string",
    criteria="correctness",
    llm=evaluator_llm
)

ref_evaluation = ref_based_evaluator.evaluate_string_pairs(
    prediction=gpt_response["answer"],
    prediction_b=gemini_response["answer"],
    reference=reference,
    input=question
)

print("Reference-based 평가 결과:")
print(f"선호되는 응답: {ref_evaluation['value']}")
print(f"점수: {ref_evaluation['score']}")
print(f"평가 근거: {ref_evaluation['reasoning']}")
```

### 2. 사용자 정의 기준 평가
```python
from langchain_core.prompts import PromptTemplate

# 사용자 정의 평가 기준
custom_criteria = {
    "conciseness": "답변이 간결하고 핵심을 잘 전달하는가? 불필요한 반복이나 장황함이 없는가?",
    "helpfulness": "답변이 사용자에게 실질적인 도움이 되는가? 유용한 정보를 제공하는가?",
    "accuracy": "답변이 사실적으로 정확한가? 제공된 문서와 일치하는가?"
}

# 사용자 정의 평가자 생성
def create_custom_evaluator(criteria_name, criteria_description):
    """사용자 정의 평가자 생성"""

    # 사용자 정의 프롬프트 템플릿
    custom_prompt = PromptTemplate(
        input_variables=["input", "prediction", "prediction_b", "reference"],
        template="""다음 질문에 대한 두 개의 답변을 비교 평가해주세요.

평가 기준: {criteria_description}

질문: {input}

참조 답변: {reference}

답변 A: {prediction}

답변 B: {prediction_b}

평가 결과를 다음 형식으로 제공해주세요:
- 선택: A 또는 B (더 나은 답변을 선택)
- 점수: 0 (B가 더 좋음) 또는 1 (A가 더 좋음)
- 근거: 상세한 평가 근거

선택: [A/B]
점수: [0/1]
근거: [평가 근거]"""
    )

    # 평가자 로드
    evaluator = load_evaluator(
        "labeled_pairwise_string",
        criteria=criteria_description,
        llm=evaluator_llm,
        prompt=custom_prompt
    )

    return evaluator

# 사용자 정의 기준으로 평가
for criteria_name, criteria_desc in custom_criteria.items():
    print(f"\n=== {criteria_name.upper()} 평가 ===")

    custom_evaluator = create_custom_evaluator(criteria_name, criteria_desc)

    custom_result = custom_evaluator.evaluate_string_pairs(
        prediction=gpt_response["answer"],
        prediction_b=gemini_response["answer"],
        reference=reference,
        input=question
    )

    print(f"기준: {criteria_desc}")
    print(f"선호되는 응답: {custom_result['value']}")
    print(f"점수: {custom_result['score']}")
    print(f"평가 근거: {custom_result['reasoning']}")
```

## 6단계: 대규모 A/B 테스트 구현

### 종합 평가 시스템
```python
import time
import random
from typing import List, Dict, Any

class ComprehensiveABTester:
    def __init__(self, evaluator_llm, langfuse_handler):
        self.evaluator_llm = evaluator_llm
        self.langfuse_handler = langfuse_handler
        self.evaluation_results = []

    def run_comprehensive_evaluation(
        self,
        test_dataset: pd.DataFrame,
        model_a_config: Dict[str, Any],
        model_b_config: Dict[str, Any],
        retriever,
        evaluation_criteria: List[str] = ["conciseness", "helpfulness", "accuracy"],
        sample_size: int = None
    ) -> Dict[str, Any]:
        """
        종합적인 A/B 테스트 수행
        """

        # 샘플 데이터 선택
        if sample_size and sample_size < len(test_dataset):
            sample_indices = random.sample(range(len(test_dataset)), sample_size)
            test_data = test_dataset.iloc[sample_indices]
        else:
            test_data = test_dataset

        print(f"🚀 A/B 테스트 시작: {len(test_data)}개 샘플")

        results = {
            "model_a_wins": 0,
            "model_b_wins": 0,
            "ties": 0,
            "detailed_results": [],
            "criteria_scores": {criterion: {"a_wins": 0, "b_wins": 0, "ties": 0}
                             for criterion in evaluation_criteria}
        }

        for idx, row in test_data.iterrows():
            try:
                print(f"\n📊 평가 진행: {idx+1}/{len(test_data)}")

                question = row["user_input"]
                reference = row.get("reference", "")

                # 모델 A 답변 생성
                response_a = self._generate_response(
                    question, retriever,
                    model_a_config["llm"],
                    model_a_config.get("rag_function", rag_bot)
                )

                # 모델 B 답변 생성
                response_b = self._generate_response(
                    question, retriever,
                    model_b_config["llm"],
                    model_b_config.get("rag_function", rag_bot)
                )

                # 각 기준별 평가
                item_results = {
                    "question": question,
                    "reference": reference,
                    "response_a": response_a["answer"],
                    "response_b": response_b["answer"],
                    "evaluations": {}
                }

                overall_a_score = 0
                overall_b_score = 0

                for criterion in evaluation_criteria:
                    criterion_result = self._evaluate_responses(
                        question, response_a["answer"], response_b["answer"],
                        reference, criterion
                    )

                    item_results["evaluations"][criterion] = criterion_result

                    # 점수 집계
                    if criterion_result["score"] == 1:  # A가 더 좋음
                        overall_a_score += 1
                        results["criteria_scores"][criterion]["a_wins"] += 1
                    elif criterion_result["score"] == 0:  # B가 더 좋음
                        overall_b_score += 1
                        results["criteria_scores"][criterion]["b_wins"] += 1
                    else:  # 무승부
                        results["criteria_scores"][criterion]["ties"] += 1

                # 전체 승부 결정
                if overall_a_score > overall_b_score:
                    results["model_a_wins"] += 1
                    item_results["winner"] = "A"
                elif overall_b_score > overall_a_score:
                    results["model_b_wins"] += 1
                    item_results["winner"] = "B"
                else:
                    results["ties"] += 1
                    item_results["winner"] = "Tie"

                results["detailed_results"].append(item_results)

                # 진행률 출력
                if (idx + 1) % 5 == 0:
                    self._print_progress(results, idx + 1, len(test_data))

                # API 호출 제한을 위한 대기
                time.sleep(1)

            except Exception as e:
                print(f"❌ 평가 실패 (인덱스 {idx}): {e}")
                continue

        # 최종 결과 계산
        total_evaluations = results["model_a_wins"] + results["model_b_wins"] + results["ties"]
        results["model_a_win_rate"] = results["model_a_wins"] / total_evaluations if total_evaluations > 0 else 0
        results["model_b_win_rate"] = results["model_b_wins"] / total_evaluations if total_evaluations > 0 else 0
        results["tie_rate"] = results["ties"] / total_evaluations if total_evaluations > 0 else 0

        return results

    def _generate_response(self, question: str, retriever, llm, rag_function):
        """응답 생성"""
        return rag_function(
            question=question,
            retriever=retriever,
            llm=llm,
            config={"callbacks": [self.langfuse_handler]}
        )

    def _evaluate_responses(self, question: str, response_a: str, response_b: str,
                          reference: str, criterion: str) -> Dict[str, Any]:
        """응답 평가"""
        try:
            evaluator = load_evaluator(
                "labeled_pairwise_string",
                criteria=criterion,
                llm=self.evaluator_llm
            )

            result = evaluator.evaluate_string_pairs(
                prediction=response_a,
                prediction_b=response_b,
                reference=reference,
                input=question
            )

            return {
                "criterion": criterion,
                "value": result["value"],
                "score": result["score"],
                "reasoning": result["reasoning"]
            }

        except Exception as e:
            print(f"⚠️ 평가 실패 ({criterion}): {e}")
            return {
                "criterion": criterion,
                "value": "C",
                "score": 0.5,
                "reasoning": f"평가 실패: {e}"
            }

    def _print_progress(self, results: Dict, current: int, total: int):
        """진행률 출력"""
        print(f"\n📈 진행률: {current}/{total}")
        print(f"   모델 A 승리: {results['model_a_wins']}")
        print(f"   모델 B 승리: {results['model_b_wins']}")
        print(f"   무승부: {results['ties']}")

    def generate_evaluation_report(self, results: Dict, model_a_name: str, model_b_name: str) -> str:
        """평가 리포트 생성"""
        report = []
        report.append("# A/B 테스트 평가 리포트")
        report.append(f"**모델 A**: {model_a_name}")
        report.append(f"**모델 B**: {model_b_name}")
        report.append("")

        # 전체 결과
        report.append("## 📊 전체 결과")
        report.append(f"- 모델 A 승리: {results['model_a_wins']} ({results['model_a_win_rate']:.1%})")
        report.append(f"- 모델 B 승리: {results['model_b_wins']} ({results['model_b_win_rate']:.1%})")
        report.append(f"- 무승부: {results['ties']} ({results['tie_rate']:.1%})")
        report.append("")

        # 기준별 결과
        report.append("## 🎯 기준별 성능")
        for criterion, scores in results["criteria_scores"].items():
            total = scores["a_wins"] + scores["b_wins"] + scores["ties"]
            if total > 0:
                a_rate = scores["a_wins"] / total
                b_rate = scores["b_wins"] / total
                report.append(f"### {criterion.title()}")
                report.append(f"- 모델 A: {scores['a_wins']} ({a_rate:.1%})")
                report.append(f"- 모델 B: {scores['b_wins']} ({b_rate:.1%})")
                report.append(f"- 무승부: {scores['ties']} ({(1-a_rate-b_rate):.1%})")
                report.append("")

        # 권장사항
        report.append("## 💡 권장사항")
        if results["model_a_win_rate"] > results["model_b_win_rate"]:
            report.append(f"- {model_a_name}이 전반적으로 우수한 성능을 보입니다")
        elif results["model_b_win_rate"] > results["model_a_win_rate"]:
            report.append(f"- {model_b_name}이 전반적으로 우수한 성능을 보입니다")
        else:
            report.append("- 두 모델의 성능이 비슷합니다")

        return "\n".join(report)

# 종합 평가 실행
ab_tester = ComprehensiveABTester(evaluator_llm, langfuse_handler)

# 모델 설정
model_configs = {
    "gpt-4.1-mini": {
        "llm": ChatOpenAI(model="gpt-4.1-mini", temperature=0),
        "rag_function": rag_bot
    },
    "gemini-2.0-flash": {
        "llm": ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0),
        "rag_function": rag_bot
    }
}

# A/B 테스트 실행
evaluation_results = ab_tester.run_comprehensive_evaluation(
    test_dataset=df_qa_test,
    model_a_config=model_configs["gpt-4.1-mini"],
    model_b_config=model_configs["gemini-2.0-flash"],
    retriever=hybrid_retriever,
    evaluation_criteria=["conciseness", "helpfulness", "accuracy"],
    sample_size=10  # 테스트용으로 10개 샘플만
)

# 결과 리포트 생성
report = ab_tester.generate_evaluation_report(
    evaluation_results,
    "GPT-4.1-mini",
    "Gemini-2.0-flash"
)

print(report)

# 리포트 파일 저장
with open("ab_test_report.md", "w", encoding="utf-8") as f:
    f.write(report)

print("\n📄 평가 리포트가 'ab_test_report.md'에 저장되었습니다.")
```

## 실습 과제

### 기본 실습
1. **단일 모델 비교**
   - GPT-4.1-mini와 Gemini-2.0-flash 모델 비교
   - Reference-free 방식으로 5개 질문 평가
   - 결과를 Langfuse UI에서 확인

2. **프롬프트 스타일 비교**
   - 친근한 스타일 vs 전문적 스타일 프롬프트 작성
   - 동일 모델로 두 프롬프트 성능 비교
   - 사용자 선호도 분석

### 응용 실습
3. **다중 기준 평가**
   - Conciseness, Helpfulness, Accuracy 기준으로 평가
   - 기준별 가중치 적용한 종합 점수 계산
   - 기준별 성능 차이 분석

4. **오픈소스 모델 성능 비교**
   - Ollama에서 다운로드한 모델과 상용 모델 비교
   - 전체 테스트셋(df_qa_test)으로 comprehensive evaluation
   - Reference-based와 Reference-free 평가 모두 수행

### 심화 실습
5. **검색기 성능 비교**
   - Vector Search vs BM25 vs Hybrid Search 비교
   - 검색 정확도가 최종 답변 품질에 미치는 영향 분석
   - 검색기별 최적 모델 조합 탐색

6. **실시간 A/B 테스트 시스템**
   - 사용자 피드백을 반영한 지속적 평가 시스템
   - 성능 저하 감지 및 자동 알림 기능
   - 모델 배포 전 자동 검증 파이프라인

## 문제 해결 가이드

### 일반적인 오류들
1. **API 호출 제한 오류**
   ```python
   # 요청 간 대기 시간 추가
   import time
   time.sleep(1)  # 1초 대기
   ```

2. **Langfuse 연결 오류**
   ```python
   # 환경 변수 확인
   print("LANGFUSE_PUBLIC_KEY:", bool(os.getenv("LANGFUSE_PUBLIC_KEY")))
   print("LANGFUSE_SECRET_KEY:", bool(os.getenv("LANGFUSE_SECRET_KEY")))
   ```

3. **메모리 부족 오류**
   ```python
   # 배치 처리 및 가비지 컬렉션
   import gc
   gc.collect()
   ```

## 참고 자료
- [LangChain 평가 가이드](https://python.langchain.com/docs/guides/evaluation/)
- [Langfuse A/B 테스트 문서](https://langfuse.com/docs/evaluation)
- [LLM 애플리케이션 평가 Best Practices](https://docs.smith.langchain.com/evaluation)
- [Pairwise Evaluation 방법론](https://python.langchain.com/api_reference/langchain/evaluation.html)

이 가이드를 통해 체계적인 A/B 테스트를 구현하여 LLM 애플리케이션의 성능을 객관적으로 평가하고 최적화할 수 있습니다.