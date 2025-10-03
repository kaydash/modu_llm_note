# Langfuse를 활용한 RAG 시스템 평가 - 체계적 모니터링과 평가 자동화 가이드

## 📚 학습 목표
- Langfuse의 핵심 기능과 RAG 시스템 모니터링 방법을 이해한다
- 데이터셋 기반의 체계적인 평가 파이프라인을 구축할 수 있다
- CallbackHandler를 활용한 자동 추적 및 로깅 시스템을 구현한다
- ROUGE 점수와 LLM-as-Judge 방식의 통합 평가를 수행할 수 있다
- Langfuse 대시보드를 통한 성능 분석과 개선점 도출 방법을 습득한다

## 🔑 핵심 개념

### Langfuse란?
- **정의**: LLM 애플리케이션을 위한 오픈소스 관측성 및 분석 플랫폼
- **목적**: 프로덕션 환경에서 LLM 애플리케이션의 성능을 모니터링하고 개선
- **특징**: 실시간 추적, 자동 평가, 비용 분석, 성능 대시보드 제공

### 주요 기능
1. **트레이싱(Tracing)**: LLM 호출과 체인 실행 과정의 상세 추적
2. **평가(Evaluation)**: 자동화된 품질 평가 및 메트릭 수집
3. **데이터셋 관리**: 평가용 데이터셋 생성 및 버전 관리
4. **대시보드**: 시각적 성능 분석 및 인사이트 제공
5. **비용 추적**: API 사용량 및 비용 모니터링

### Langfuse의 장점
- **🔄 자동화된 추적**: CallbackHandler를 통한 투명한 로깅
- **📊 시각적 대시보드**: 직관적인 성능 분석 인터페이스
- **🔍 다양한 평가 지표**: ROUGE, LLM-as-Judge 등 통합 지원
- **🚀 확장 가능한 파이프라인**: 대규모 평가 자동화 가능

## 🛠 환경 설정

### 필수 라이브러리 설치
```bash
# Langfuse 관련 라이브러리
pip install langfuse langfuse-langchain

# 기본 LangChain 라이브러리
pip install langchain langchain-openai langchain-chroma

# 평가 메트릭 라이브러리
pip install korouge-score
pip install krag  # 한국어 토크나이저 및 검색기

# 데이터 처리 라이브러리
pip install pandas numpy openpyxl
```

### 환경 변수 설정
```python
# .env 파일
OPENAI_API_KEY=your_openai_api_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com  # 또는 자체 호스팅 URL

# LangSmith 설정 (선택사항)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
```

### 기본 설정
```python
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# 환경 변수 로드
load_dotenv()
warnings.filterwarnings("ignore")

# Langfuse 라이브러리
from langfuse import get_client
from langfuse.langchain import CallbackHandler

# LangChain 라이브러리
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.evaluation import load_evaluator
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 평가 메트릭
from korouge_score import rouge_scorer
from krag.tokenizers import KiwiTokenizer
```

## 💻 단계별 구현

### 1단계: Langfuse 클라이언트 초기화 및 인증

```python
class LangfuseManager:
    def __init__(self):
        """Langfuse 클라이언트 초기화"""
        self.client = get_client()
        self.callback_handler = CallbackHandler()

        # 인증 상태 확인
        auth_status = self.client.auth_check()
        if auth_status:
            print("✅ Langfuse 인증 성공")
        else:
            raise Exception("❌ Langfuse 인증 실패 - API 키를 확인하세요")

    def get_client(self):
        """Langfuse 클라이언트 반환"""
        return self.client

    def get_callback_handler(self):
        """CallbackHandler 반환"""
        return self.callback_handler

    def list_datasets(self) -> List[str]:
        """사용 가능한 데이터셋 목록 반환"""
        try:
            # Langfuse API로 데이터셋 목록 조회
            # 실제 구현은 Langfuse API 문서 참고
            return []
        except Exception as e:
            print(f"데이터셋 목록 조회 실패: {e}")
            return []

    def flush(self):
        """보류 중인 작업을 Langfuse 서버에 전송"""
        self.client.flush()

# Langfuse 매니저 초기화
langfuse_manager = LangfuseManager()
langfuse_client = langfuse_manager.get_client()
langfuse_handler = langfuse_manager.get_callback_handler()

print("Langfuse 초기화 완료!")
```

### 2단계: 평가용 데이터셋 생성 및 관리

```python
class DatasetManager:
    def __init__(self, langfuse_client):
        self.client = langfuse_client

    def create_dataset_from_excel(self, excel_path: str, dataset_name: str) -> str:
        """
        Excel 파일로부터 Langfuse 데이터셋 생성

        Args:
            excel_path: Excel 파일 경로
            dataset_name: 생성할 데이터셋 이름

        Returns:
            생성된 데이터셋 이름
        """
        # Excel 파일 읽기
        df_qa_test = pd.read_excel(excel_path)
        print(f"📊 Excel 데이터 로드: {df_qa_test.shape[0]}개 항목")

        # 데이터셋 생성
        try:
            dataset = self.client.create_dataset(name=dataset_name)
            print(f"📂 데이터셋 생성: {dataset_name}")
        except Exception as e:
            print(f"⚠️ 데이터셋 생성 실패 (이미 존재할 수 있음): {e}")

        # 데이터 변환 및 추가
        successful_items = 0
        failed_items = 0

        for idx, row in df_qa_test.iterrows():
            try:
                self.client.create_dataset_item(
                    dataset_name=dataset_name,
                    input=row["user_input"],
                    expected_output=row["reference"],
                    metadata={
                        "reference_contexts": row.get("reference_contexts", ""),
                        "synthesizer_name": row.get("synthesizer_name", ""),
                        "item_index": idx
                    }
                )
                successful_items += 1

                if (idx + 1) % 10 == 0:
                    print(f"📝 데이터 추가 진행: {idx + 1}/{len(df_qa_test)}")

            except Exception as e:
                failed_items += 1
                print(f"❌ 항목 {idx} 추가 실패: {e}")

        # 변경사항 저장
        self.client.flush()

        print(f"✅ 데이터셋 생성 완료:")
        print(f"   - 성공: {successful_items}개")
        print(f"   - 실패: {failed_items}개")

        return dataset_name

    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """데이터셋 정보 조회"""
        try:
            dataset = self.client.get_dataset(name=dataset_name)

            return {
                "name": dataset.name,
                "item_count": len(dataset.items),
                "created_at": getattr(dataset, 'created_at', 'Unknown'),
                "items_preview": dataset.items[:3] if dataset.items else []
            }
        except Exception as e:
            print(f"데이터셋 정보 조회 실패: {e}")
            return {}

    def validate_dataset(self, dataset_name: str) -> bool:
        """데이터셋 유효성 검증"""
        try:
            dataset = self.client.get_dataset(name=dataset_name)

            if not dataset:
                print(f"❌ 데이터셋 '{dataset_name}'이 존재하지 않습니다")
                return False

            if len(dataset.items) == 0:
                print(f"❌ 데이터셋 '{dataset_name}'에 항목이 없습니다")
                return False

            # 샘플 항목 검증
            sample_item = dataset.items[0]
            required_fields = ['input', 'expected_output']

            for field in required_fields:
                if not hasattr(sample_item, field) or not getattr(sample_item, field):
                    print(f"❌ 필수 필드 '{field}'가 누락되었습니다")
                    return False

            print(f"✅ 데이터셋 '{dataset_name}' 검증 완료")
            return True

        except Exception as e:
            print(f"❌ 데이터셋 검증 실패: {e}")
            return False

# 데이터셋 매니저 사용 예시
dataset_manager = DatasetManager(langfuse_client)

# Excel 파일로부터 데이터셋 생성
dataset_name = dataset_manager.create_dataset_from_excel(
    excel_path="data/testset.xlsx",
    dataset_name="RAG_Evaluation_Dataset_Tesla"
)

# 데이터셋 정보 확인
dataset_info = dataset_manager.get_dataset_info(dataset_name)
print(f"\n📊 데이터셋 정보:")
for key, value in dataset_info.items():
    if key != 'items_preview':
        print(f"   - {key}: {value}")
```

### 3단계: RAG 시스템 구성

```python
class RAGSystemBuilder:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.1)

    def setup_retrievers(self, chroma_path: str = "./chroma_db",
                        documents_path: str = "data/korean_docs_final.jsonl"):
        """검색기 설정"""
        # 벡터 저장소 설정
        chroma_db = Chroma(
            collection_name="db_korean_cosine_metadata",
            embedding_function=self.embeddings,
            persist_directory=chroma_path,
        )

        vector_retriever = chroma_db.as_retriever(search_kwargs={'k': 4})

        # BM25 검색기 설정 (한국어 문서용)
        try:
            from krag.tokenizers import KiwiTokenizer
            from krag.retrievers import KiWiBM25RetrieverWithScore
            import json
            from langchain.schema import Document

            # 한국어 문서 로드
            with open(documents_path, 'r', encoding='utf-8') as f:
                korean_docs = [json.loads(line) for line in f]

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

            # BM25 검색기 초기화
            kiwi_tokenizer = KiwiTokenizer(model_type='knlm', typos='basic')
            bm25_retriever = KiWiBM25RetrieverWithScore(
                documents=documents,
                kiwi_tokenizer=kiwi_tokenizer,
                k=4
            )

            # 하이브리드 검색기 (Ensemble)
            from langchain.retrievers import EnsembleRetriever

            hybrid_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.5, 0.5]
            )

            print("✅ 하이브리드 검색기 설정 완료")
            return hybrid_retriever

        except ImportError:
            print("⚠️ BM25 검색기를 사용할 수 없습니다. 벡터 검색기만 사용합니다.")
            return vector_retriever

    def create_rag_chain(self, retriever):
        """RAG 체인 생성"""
        template = """다음 맥락을 바탕으로 질문에 답하세요.
맥락이 질문과 관련이 없다면 '답변에 필요한 근거를 찾지 못했습니다.'라고 답하세요.

[맥락]
{context}

[질문]
{question}

[답변]
"""

        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return rag_chain

# RAG 시스템 구축
rag_builder = RAGSystemBuilder()
retriever = rag_builder.setup_retrievers()
rag_chain = rag_builder.create_rag_chain(retriever)

# 테스트 실행
test_question = "테슬라의 CEO는 누구인가요?"
test_answer = rag_chain.invoke(test_question)
print(f"\n🧪 RAG 시스템 테스트:")
print(f"질문: {test_question}")
print(f"답변: {test_answer}")
```

### 4단계: 종합 평가 시스템 구현

```python
@dataclass
class EvaluationResult:
    """평가 결과를 담는 데이터 클래스"""
    item_id: str
    input: str
    output: str
    expected_output: str
    scores: Dict[str, float]
    details: Dict[str, Any]
    trace_id: Optional[str] = None
    execution_time: Optional[float] = None
    error: Optional[str] = None

class ComprehensiveEvaluator:
    def __init__(self, llm):
        self.llm = llm
        self.setup_evaluators()

    def setup_evaluators(self):
        """평가자 초기화"""
        # ROUGE 평가자
        from korouge_score import rouge_scorer
        from krag.tokenizers import KiwiTokenizer

        class CustomKiwiTokenizer(KiwiTokenizer):
            def tokenize(self, text):
                return [t.form for t in super().tokenize(text)]

        self.kiwi_tokenizer = CustomKiwiTokenizer(model_type='knlm', typos='basic')
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            tokenizer=self.kiwi_tokenizer
        )

        # LLM-as-Judge 평가자들
        self.evaluators = {
            "relevance": load_evaluator(
                evaluator="labeled_criteria",
                criteria="relevance",
                llm=self.llm
            ),
            "helpfulness": load_evaluator(
                evaluator="labeled_criteria",
                criteria="helpfulness",
                llm=self.llm
            ),
            "conciseness": load_evaluator(
                evaluator="labeled_criteria",
                criteria="conciseness",
                llm=self.llm
            ),
            "correctness": load_evaluator(
                evaluator="labeled_criteria",
                criteria="correctness",
                llm=self.llm
            )
        }

        print("✅ 평가자 초기화 완료")

    def evaluate_rouge_scores(self, prediction: str, reference: str) -> Dict[str, float]:
        """ROUGE 점수 계산"""
        try:
            rouge_results = self.rouge_scorer.score(reference, prediction)
            return {
                "rouge1": rouge_results['rouge1'].fmeasure,
                "rouge2": rouge_results['rouge2'].fmeasure,
                "rougeL": rouge_results['rougeL'].fmeasure
            }
        except Exception as e:
            print(f"ROUGE 평가 실패: {e}")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    def evaluate_llm_criteria(self, input_text: str, prediction: str,
                             reference: str) -> Dict[str, Dict[str, Any]]:
        """LLM-as-Judge 평가"""
        results = {}

        for criterion_name, evaluator in self.evaluators.items():
            try:
                evaluation_result = evaluator.evaluate_strings(
                    input=input_text,
                    prediction=prediction,
                    reference=reference
                )

                results[criterion_name] = {
                    "score": float(evaluation_result.get('score', 0)),
                    "reasoning": evaluation_result.get('reasoning', ''),
                    "status": "success"
                }

            except Exception as e:
                results[criterion_name] = {
                    "score": 0.0,
                    "reasoning": f"평가 실패: {str(e)}",
                    "status": "error"
                }

        return results

    def calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """종합 점수 계산"""
        if not scores:
            return 0.0

        # 가중치 설정
        weights = {
            "rouge1": 0.2,
            "rouge2": 0.1,
            "rougeL": 0.1,
            "relevance": 0.25,
            "helpfulness": 0.15,
            "conciseness": 0.1,
            "correctness": 0.1
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for metric, score in scores.items():
            if metric in weights:
                weight = weights[metric]
                weighted_sum += score * weight
                total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

class LangfuseEvaluationPipeline:
    def __init__(self, langfuse_client, rag_chain, evaluator):
        self.client = langfuse_client
        self.rag_chain = rag_chain
        self.evaluator = evaluator

    def run_evaluation(self, dataset_name: str, run_name: str,
                      max_items: Optional[int] = None) -> List[EvaluationResult]:
        """전체 평가 파이프라인 실행"""

        # 데이터셋 가져오기
        dataset = self.client.get_dataset(name=dataset_name)
        if not dataset:
            raise ValueError(f"데이터셋 '{dataset_name}'을 찾을 수 없습니다")

        items_to_process = dataset.items
        if max_items:
            items_to_process = items_to_process[:max_items]

        print(f"🚀 평가 시작: {len(items_to_process)}개 항목")

        results = []
        successful = 0
        failed = 0

        for idx, item in enumerate(items_to_process, 1):
            try:
                print(f"\n📊 항목 {idx}/{len(items_to_process)} 처리 중...")

                result = self._evaluate_single_item(item, run_name)
                results.append(result)

                if result.error:
                    failed += 1
                    print(f"   ❌ 실패: {result.error}")
                else:
                    successful += 1
                    overall_score = self.evaluator.calculate_overall_score(result.scores)
                    print(f"   ✅ 완료 (종합 점수: {overall_score:.3f})")

                # 진행률 출력
                if idx % 10 == 0 or idx == len(items_to_process):
                    print(f"\n📈 진행률: {idx}/{len(items_to_process)} "
                          f"(성공: {successful}, 실패: {failed})")

            except Exception as e:
                failed += 1
                print(f"   ❌ 예외 발생: {e}")

                # 실패 케이스도 결과에 포함
                results.append(EvaluationResult(
                    item_id=item.id,
                    input=str(item.input),
                    output="",
                    expected_output=str(item.expected_output) if item.expected_output else "",
                    scores={},
                    details={},
                    error=str(e)
                ))

        print(f"\n🎯 평가 완료:")
        print(f"   - 총 항목: {len(items_to_process)}")
        print(f"   - 성공: {successful}")
        print(f"   - 실패: {failed}")
        print(f"   - 성공률: {successful/len(items_to_process)*100:.1f}%")

        return results

    def _evaluate_single_item(self, item, run_name: str) -> EvaluationResult:
        """단일 항목 평가"""
        import time

        start_time = time.time()

        # Langfuse 트레이싱과 함께 RAG 실행
        with item.run(run_name=run_name) as root_span:
            try:
                # RAG 체인 실행
                output = self.rag_chain.invoke(
                    item.input,
                    config={"callbacks": [CallbackHandler()]}
                )

                # 평가 수행
                rouge_scores = self.evaluator.evaluate_rouge_scores(
                    str(output), str(item.expected_output)
                )

                llm_criteria_results = self.evaluator.evaluate_llm_criteria(
                    str(item.input), str(output), str(item.expected_output)
                )

                # 점수 통합
                all_scores = rouge_scores.copy()
                for criterion, result in llm_criteria_results.items():
                    all_scores[criterion] = result["score"]

                # 종합 점수 계산
                overall_score = self.evaluator.calculate_overall_score(all_scores)

                # Langfuse에 점수 기록
                root_span.score(name="overall", value=overall_score)
                for score_name, score_value in all_scores.items():
                    root_span.score(name=score_name, value=score_value)

                execution_time = time.time() - start_time

                return EvaluationResult(
                    item_id=item.id,
                    input=str(item.input),
                    output=str(output),
                    expected_output=str(item.expected_output) if item.expected_output else "",
                    scores=all_scores,
                    details={
                        "rouge": rouge_scores,
                        "llm_criteria": llm_criteria_results,
                        "overall_score": overall_score
                    },
                    trace_id=getattr(root_span, 'trace_id', None),
                    execution_time=execution_time
                )

            except Exception as e:
                execution_time = time.time() - start_time

                # 실패해도 Langfuse에 기록
                root_span.score(name="overall", value=0.0)

                return EvaluationResult(
                    item_id=item.id,
                    input=str(item.input),
                    output="",
                    expected_output=str(item.expected_output) if item.expected_output else "",
                    scores={},
                    details={},
                    execution_time=execution_time,
                    error=str(e)
                )

# 종합 평가 시스템 구성
evaluator = ComprehensiveEvaluator(ChatOpenAI(model="gpt-4.1-mini", temperature=0.1))
evaluation_pipeline = LangfuseEvaluationPipeline(langfuse_client, rag_chain, evaluator)

# 평가 실행
results = evaluation_pipeline.run_evaluation(
    dataset_name="RAG_Evaluation_Dataset_Tesla",
    run_name="comprehensive_evaluation_v1",
    max_items=10  # 테스트용으로 처음 10개만
)

print(f"\n📊 평가 결과 요약:")
successful_results = [r for r in results if not r.error]
if successful_results:
    avg_overall_score = np.mean([
        evaluator.calculate_overall_score(r.scores)
        for r in successful_results
    ])
    avg_execution_time = np.mean([r.execution_time for r in successful_results])

    print(f"   - 평균 종합 점수: {avg_overall_score:.3f}")
    print(f"   - 평균 실행 시간: {avg_execution_time:.2f}초")
```

## 🎯 실습 문제

### 기초 실습
1. **Langfuse 설정 및 연결**
   - Langfuse 계정 생성 후 API 키 설정
   - 간단한 CallbackHandler를 사용한 LLM 호출 추적
   - 대시보드에서 호출 로그 확인

2. **기본 데이터셋 생성**
   - 5개의 질문-답변 쌍으로 소규모 데이터셋 생성
   - RAG 체인으로 답변 생성 후 Langfuse에 기록

### 응용 실습
3. **커스텀 평가 메트릭 구현**
   - 한국어 특화 평가 기준 개발
   - 도메인 특화 평가자 (예: 기술 정확성) 구현
   - 기존 메트릭과 성능 비교

4. **A/B 테스트 시스템**
   - 서로 다른 RAG 설정으로 두 시스템 구성
   - 동일 데이터셋으로 성능 비교 평가
   - Langfuse에서 결과 분석 및 시각화

### 심화 실습
5. **프로덕션 모니터링 시스템**
   - 실시간 성능 임계값 모니터링
   - 성능 저하 시 자동 알림 시스템
   - 지속적 개선을 위한 피드백 루프 구현

## ✅ 솔루션 예시

### 실습 1: 실시간 성능 모니터링 시스템
```python
class RealTimeMonitor:
    def __init__(self, langfuse_client, threshold_scores: Dict[str, float]):
        self.client = langfuse_client
        self.thresholds = threshold_scores
        self.alerts = []

    def monitor_performance(self, run_name: str,
                          check_interval: int = 100) -> Dict[str, Any]:
        """실시간 성능 모니터링"""

        # 최근 실행 결과 가져오기 (실제 구현 시 Langfuse API 활용)
        recent_traces = self._get_recent_traces(run_name, check_interval)

        if not recent_traces:
            return {"status": "no_data", "message": "모니터링할 데이터가 없습니다."}

        # 성능 지표 계산
        performance_metrics = self._calculate_performance_metrics(recent_traces)

        # 임계값 체크
        alerts = self._check_thresholds(performance_metrics)

        # 알림 생성
        if alerts:
            self._generate_alerts(alerts, performance_metrics)

        return {
            "status": "monitored",
            "metrics": performance_metrics,
            "alerts": alerts,
            "trace_count": len(recent_traces)
        }

    def _get_recent_traces(self, run_name: str, limit: int) -> List[Dict]:
        """최근 트레이스 데이터 조회 (모의 구현)"""
        # 실제로는 Langfuse API를 통해 데이터 조회
        # 여기서는 예시 데이터 반환
        return [
            {
                "id": f"trace_{i}",
                "scores": {
                    "overall": np.random.uniform(0.6, 0.9),
                    "relevance": np.random.uniform(0.7, 0.95),
                    "correctness": np.random.uniform(0.65, 0.85)
                },
                "execution_time": np.random.uniform(1.0, 3.0),
                "timestamp": "2024-01-01T00:00:00Z"
            }
            for i in range(limit)
        ]

    def _calculate_performance_metrics(self, traces: List[Dict]) -> Dict[str, float]:
        """성능 지표 계산"""
        if not traces:
            return {}

        metrics = {}

        # 점수별 평균 계산
        score_names = ["overall", "relevance", "correctness"]
        for score_name in score_names:
            scores = [
                trace["scores"].get(score_name, 0)
                for trace in traces
                if "scores" in trace
            ]
            if scores:
                metrics[f"avg_{score_name}"] = np.mean(scores)
                metrics[f"min_{score_name}"] = np.min(scores)

        # 실행 시간 통계
        execution_times = [trace.get("execution_time", 0) for trace in traces]
        if execution_times:
            metrics["avg_execution_time"] = np.mean(execution_times)
            metrics["max_execution_time"] = np.max(execution_times)

        return metrics

    def _check_thresholds(self, metrics: Dict[str, float]) -> List[Dict]:
        """임계값 확인"""
        alerts = []

        for metric_name, value in metrics.items():
            if metric_name in self.thresholds:
                threshold = self.thresholds[metric_name]

                # 점수는 임계값보다 높아야 함
                if "score" in metric_name or "avg_" in metric_name:
                    if value < threshold:
                        alerts.append({
                            "type": "low_performance",
                            "metric": metric_name,
                            "value": value,
                            "threshold": threshold,
                            "severity": "high" if value < threshold * 0.8 else "medium"
                        })

                # 실행 시간은 임계값보다 낮아야 함
                elif "time" in metric_name:
                    if value > threshold:
                        alerts.append({
                            "type": "slow_performance",
                            "metric": metric_name,
                            "value": value,
                            "threshold": threshold,
                            "severity": "high" if value > threshold * 1.5 else "medium"
                        })

        return alerts

    def _generate_alerts(self, alerts: List[Dict], metrics: Dict[str, float]):
        """알림 생성 및 저장"""
        for alert in alerts:
            alert_message = self._format_alert_message(alert)

            self.alerts.append({
                "timestamp": pd.Timestamp.now(),
                "message": alert_message,
                "alert_data": alert,
                "metrics_snapshot": metrics.copy()
            })

            print(f"🚨 알림: {alert_message}")

    def _format_alert_message(self, alert: Dict) -> str:
        """알림 메시지 포맷팅"""
        metric = alert["metric"]
        value = alert["value"]
        threshold = alert["threshold"]
        severity = alert["severity"]

        severity_emoji = "🔴" if severity == "high" else "🟡"

        if alert["type"] == "low_performance":
            return (f"{severity_emoji} {metric} 성능 저하 감지: "
                   f"{value:.3f} < {threshold:.3f} (임계값)")
        else:
            return (f"{severity_emoji} {metric} 응답 지연 감지: "
                   f"{value:.2f}초 > {threshold:.2f}초 (임계값)")

    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """알림 요약 조회"""
        cutoff_time = pd.Timestamp.now() - pd.Timedelta(hours=hours)
        recent_alerts = [
            alert for alert in self.alerts
            if alert["timestamp"] > cutoff_time
        ]

        if not recent_alerts:
            return {"period": f"최근 {hours}시간", "alert_count": 0}

        # 알림 유형별 집계
        alert_types = {}
        severity_counts = {"high": 0, "medium": 0}

        for alert in recent_alerts:
            alert_type = alert["alert_data"]["type"]
            severity = alert["alert_data"]["severity"]

            alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
            severity_counts[severity] += 1

        return {
            "period": f"최근 {hours}시간",
            "alert_count": len(recent_alerts),
            "alert_types": alert_types,
            "severity_distribution": severity_counts,
            "latest_alerts": recent_alerts[-5:]  # 최근 5개
        }

# 실시간 모니터링 시스템 사용 예시
monitor = RealTimeMonitor(
    langfuse_client=langfuse_client,
    threshold_scores={
        "avg_overall": 0.75,      # 종합 점수 75% 이상
        "avg_relevance": 0.80,    # 관련성 80% 이상
        "avg_correctness": 0.70,  # 정확성 70% 이상
        "avg_execution_time": 5.0, # 평균 실행 시간 5초 이하
        "max_execution_time": 10.0 # 최대 실행 시간 10초 이하
    }
)

# 성능 모니터링 실행
monitoring_result = monitor.monitor_performance("comprehensive_evaluation_v1")
print("모니터링 결과:", monitoring_result)

# 알림 요약 확인
alert_summary = monitor.get_alert_summary(hours=24)
print("알림 요약:", alert_summary)
```

### 실습 2: 자동화된 성능 비교 시스템
```python
class AutomatedComparisonSystem:
    def __init__(self, langfuse_client):
        self.client = langfuse_client
        self.comparison_results = []

    def compare_rag_systems(self, systems_config: Dict[str, Dict],
                           dataset_name: str,
                           comparison_name: str) -> Dict[str, Any]:
        """
        여러 RAG 시스템 자동 비교

        Args:
            systems_config: {"system_name": {"rag_chain": chain, "description": str}, ...}
            dataset_name: 비교에 사용할 데이터셋
            comparison_name: 비교 실험 이름
        """

        print(f"🔄 RAG 시스템 비교 시작: {len(systems_config)}개 시스템")

        # 각 시스템별 평가 실행
        system_results = {}

        for system_name, config in systems_config.items():
            print(f"\n📊 {system_name} 시스템 평가 중...")

            # 평가 파이프라인 생성
            evaluator = ComprehensiveEvaluator(
                ChatOpenAI(model="gpt-4.1-mini", temperature=0.1)
            )
            pipeline = LangfuseEvaluationPipeline(
                self.client, config["rag_chain"], evaluator
            )

            # 평가 실행
            run_name = f"{comparison_name}_{system_name}"
            results = pipeline.run_evaluation(
                dataset_name=dataset_name,
                run_name=run_name,
                max_items=20  # 비교용으로 20개 항목
            )

            # 결과 분석
            system_analysis = self._analyze_system_results(results, system_name)
            system_results[system_name] = {
                "config": config,
                "results": results,
                "analysis": system_analysis
            }

        # 전체 비교 분석
        comparison_analysis = self._compare_systems(system_results)

        # 결과 저장
        comparison_result = {
            "comparison_name": comparison_name,
            "dataset_name": dataset_name,
            "systems": system_results,
            "comparison_analysis": comparison_analysis,
            "timestamp": pd.Timestamp.now()
        }

        self.comparison_results.append(comparison_result)

        return comparison_result

    def _analyze_system_results(self, results: List[EvaluationResult],
                               system_name: str) -> Dict[str, Any]:
        """단일 시스템 결과 분석"""
        successful_results = [r for r in results if not r.error]

        if not successful_results:
            return {"error": "성공한 평가가 없습니다"}

        # 평균 점수 계산
        evaluator = ComprehensiveEvaluator(
            ChatOpenAI(model="gpt-4.1-mini", temperature=0.1)
        )

        overall_scores = [
            evaluator.calculate_overall_score(r.scores)
            for r in successful_results
        ]

        # 메트릭별 점수
        metric_scores = {}
        metric_names = ["rouge1", "rouge2", "rougeL", "relevance", "helpfulness", "correctness", "conciseness"]

        for metric in metric_names:
            scores = [r.scores.get(metric, 0) for r in successful_results]
            if scores:
                metric_scores[metric] = {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "min": np.min(scores),
                    "max": np.max(scores)
                }

        # 실행 시간 분석
        execution_times = [r.execution_time for r in successful_results if r.execution_time]
        time_analysis = {}
        if execution_times:
            time_analysis = {
                "mean": np.mean(execution_times),
                "std": np.std(execution_times),
                "min": np.min(execution_times),
                "max": np.max(execution_times)
            }

        return {
            "system_name": system_name,
            "total_evaluations": len(results),
            "successful_evaluations": len(successful_results),
            "success_rate": len(successful_results) / len(results),
            "overall_score": {
                "mean": np.mean(overall_scores),
                "std": np.std(overall_scores),
                "scores": overall_scores
            },
            "metric_scores": metric_scores,
            "execution_time": time_analysis
        }

    def _compare_systems(self, system_results: Dict[str, Dict]) -> Dict[str, Any]:
        """시스템 간 비교 분석"""
        comparison = {
            "ranking": [],
            "performance_comparison": {},
            "statistical_significance": {},
            "recommendations": []
        }

        # 전체 점수 기준 순위
        system_scores = []
        for system_name, data in system_results.items():
            analysis = data["analysis"]
            if "overall_score" in analysis:
                system_scores.append({
                    "system": system_name,
                    "score": analysis["overall_score"]["mean"],
                    "std": analysis["overall_score"]["std"]
                })

        # 점수 기준 정렬
        system_scores.sort(key=lambda x: x["score"], reverse=True)
        comparison["ranking"] = system_scores

        # 메트릭별 최고 성능 시스템
        metric_names = ["rouge1", "relevance", "helpfulness", "correctness"]
        for metric in metric_names:
            best_system = None
            best_score = -1

            for system_name, data in system_results.items():
                analysis = data["analysis"]
                if metric in analysis.get("metric_scores", {}):
                    score = analysis["metric_scores"][metric]["mean"]
                    if score > best_score:
                        best_score = score
                        best_system = system_name

            if best_system:
                comparison["performance_comparison"][metric] = {
                    "best_system": best_system,
                    "score": best_score
                }

        # 권장사항 생성
        if system_scores:
            best_system = system_scores[0]
            comparison["recommendations"].append({
                "type": "best_overall",
                "recommendation": f"{best_system['system']} 시스템이 종합 성능이 가장 우수합니다 (점수: {best_system['score']:.3f})"
            })

            # 실행 시간 고려
            fastest_system = None
            fastest_time = float('inf')

            for system_name, data in system_results.items():
                analysis = data["analysis"]
                if "execution_time" in analysis and "mean" in analysis["execution_time"]:
                    time = analysis["execution_time"]["mean"]
                    if time < fastest_time:
                        fastest_time = time
                        fastest_system = system_name

            if fastest_system:
                comparison["recommendations"].append({
                    "type": "fastest",
                    "recommendation": f"{fastest_system} 시스템이 가장 빠른 응답 시간을 보입니다 ({fastest_time:.2f}초)"
                })

        return comparison

    def generate_comparison_report(self, comparison_result: Dict[str, Any]) -> str:
        """비교 결과 리포트 생성"""
        report = []
        report.append(f"# RAG 시스템 비교 리포트")
        report.append(f"**실험명**: {comparison_result['comparison_name']}")
        report.append(f"**데이터셋**: {comparison_result['dataset_name']}")
        report.append(f"**실행 시간**: {comparison_result['timestamp']}")
        report.append("")

        # 순위 정보
        ranking = comparison_result["comparison_analysis"]["ranking"]
        report.append("## 📊 종합 성능 순위")
        for i, system in enumerate(ranking, 1):
            report.append(f"{i}. **{system['system']}**: {system['score']:.3f} (±{system['std']:.3f})")
        report.append("")

        # 메트릭별 최고 성능
        performance = comparison_result["comparison_analysis"]["performance_comparison"]
        if performance:
            report.append("## 🏆 메트릭별 최고 성능")
            for metric, data in performance.items():
                report.append(f"- **{metric}**: {data['best_system']} ({data['score']:.3f})")
            report.append("")

        # 권장사항
        recommendations = comparison_result["comparison_analysis"]["recommendations"]
        if recommendations:
            report.append("## 💡 권장사항")
            for rec in recommendations:
                report.append(f"- {rec['recommendation']}")
            report.append("")

        # 상세 시스템 분석
        report.append("## 📈 상세 분석")
        for system_name, data in comparison_result["systems"].items():
            analysis = data["analysis"]
            report.append(f"### {system_name}")
            report.append(f"- 성공률: {analysis['success_rate']:.1%}")

            if "overall_score" in analysis:
                report.append(f"- 평균 점수: {analysis['overall_score']['mean']:.3f}")

            if "execution_time" in analysis and "mean" in analysis["execution_time"]:
                report.append(f"- 평균 실행 시간: {analysis['execution_time']['mean']:.2f}초")

            report.append("")

        return "\n".join(report)

# 비교 시스템 사용 예시
comparison_system = AutomatedComparisonSystem(langfuse_client)

# 여러 RAG 시스템 구성
systems_to_compare = {
    "Standard_RAG": {
        "rag_chain": rag_chain,  # 기본 RAG 시스템
        "description": "기본 하이브리드 검색 RAG"
    },
    "High_Temp_RAG": {
        "rag_chain": rag_builder.create_rag_chain(retriever), # 다른 설정의 RAG
        "description": "높은 창의성 설정 RAG"
    }
}

# 비교 실행
comparison_result = comparison_system.compare_rag_systems(
    systems_config=systems_to_compare,
    dataset_name="RAG_Evaluation_Dataset_Tesla",
    comparison_name="rag_system_comparison_v1"
)

# 리포트 생성
report = comparison_system.generate_comparison_report(comparison_result)
print(report)

# 리포트 파일로 저장
with open("rag_comparison_report.md", "w", encoding="utf-8") as f:
    f.write(report)

print("\n📄 비교 리포트가 'rag_comparison_report.md'에 저장되었습니다.")
```

## 🚀 실무 활용 예시

### 1. 지속적 통합(CI) 평가 시스템

```python
class ContinuousEvaluationSystem:
    def __init__(self, langfuse_client, config_path: str = "evaluation_config.json"):
        self.client = langfuse_client
        self.config = self._load_config(config_path)
        self.baseline_scores = {}

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """평가 설정 로드"""
        default_config = {
            "evaluation_frequency": "daily",  # daily, weekly, on_commit
            "datasets": ["RAG_Evaluation_Dataset_Tesla"],
            "quality_thresholds": {
                "overall": 0.75,
                "relevance": 0.80,
                "correctness": 0.70
            },
            "regression_threshold": 0.05,  # 5% 성능 저하시 실패
            "notification_channels": ["console", "email"]
        }

        try:
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
            default_config.update(custom_config)
        except FileNotFoundError:
            print(f"설정 파일 '{config_path}'를 찾을 수 없습니다. 기본 설정을 사용합니다.")

        return default_config

    def run_ci_evaluation(self, rag_chain, commit_id: str = None) -> Dict[str, Any]:
        """CI 환경에서 평가 실행"""
        evaluation_id = f"ci_eval_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"

        if commit_id:
            evaluation_id += f"_{commit_id[:8]}"

        print(f"🔄 CI 평가 시작: {evaluation_id}")

        results = {
            "evaluation_id": evaluation_id,
            "commit_id": commit_id,
            "timestamp": pd.Timestamp.now(),
            "dataset_results": {},
            "overall_status": "unknown",
            "recommendations": []
        }

        # 각 데이터셋에 대해 평가 실행
        for dataset_name in self.config["datasets"]:
            print(f"\n📊 데이터셋 '{dataset_name}' 평가 중...")

            try:
                dataset_result = self._evaluate_dataset(
                    rag_chain, dataset_name, evaluation_id
                )
                results["dataset_results"][dataset_name] = dataset_result

            except Exception as e:
                print(f"❌ 데이터셋 '{dataset_name}' 평가 실패: {e}")
                results["dataset_results"][dataset_name] = {
                    "status": "failed",
                    "error": str(e)
                }

        # 전체 결과 분석
        overall_analysis = self._analyze_ci_results(results)
        results.update(overall_analysis)

        # 베이스라인과 비교
        regression_analysis = self._check_regression(results)
        results["regression_analysis"] = regression_analysis

        # 최종 상태 결정
        results["overall_status"] = self._determine_ci_status(results)

        # 알림 발송
        if results["overall_status"] == "failed":
            self._send_notifications(results)

        print(f"\n🎯 CI 평가 완료: {results['overall_status']}")
        return results

    def _evaluate_dataset(self, rag_chain, dataset_name: str,
                         evaluation_id: str) -> Dict[str, Any]:
        """단일 데이터셋 평가"""
        evaluator = ComprehensiveEvaluator(
            ChatOpenAI(model="gpt-4.1-mini", temperature=0.1)
        )
        pipeline = LangfuseEvaluationPipeline(self.client, rag_chain, evaluator)

        # 평가 실행 (CI에서는 샘플링)
        results = pipeline.run_evaluation(
            dataset_name=dataset_name,
            run_name=f"{evaluation_id}_{dataset_name}",
            max_items=30  # CI에서는 30개 샘플만
        )

        # 결과 분석
        successful_results = [r for r in results if not r.error]

        if not successful_results:
            return {"status": "failed", "reason": "모든 평가가 실패했습니다"}

        # 평균 점수 계산
        overall_scores = [
            evaluator.calculate_overall_score(r.scores)
            for r in successful_results
        ]

        # 메트릭별 평균
        metric_averages = {}
        for metric in ["relevance", "correctness", "helpfulness"]:
            scores = [r.scores.get(metric, 0) for r in successful_results]
            if scores:
                metric_averages[metric] = np.mean(scores)

        # 품질 임계값 체크
        quality_checks = {}
        thresholds = self.config["quality_thresholds"]

        overall_avg = np.mean(overall_scores)
        quality_checks["overall"] = {
            "score": overall_avg,
            "threshold": thresholds.get("overall", 0.75),
            "passed": overall_avg >= thresholds.get("overall", 0.75)
        }

        for metric, avg_score in metric_averages.items():
            threshold = thresholds.get(metric, 0.70)
            quality_checks[metric] = {
                "score": avg_score,
                "threshold": threshold,
                "passed": avg_score >= threshold
            }

        # 전체 통과 여부
        all_passed = all(check["passed"] for check in quality_checks.values())

        return {
            "status": "passed" if all_passed else "failed",
            "total_evaluations": len(results),
            "successful_evaluations": len(successful_results),
            "success_rate": len(successful_results) / len(results),
            "overall_score": overall_avg,
            "metric_scores": metric_averages,
            "quality_checks": quality_checks,
            "detailed_results": results
        }

    def _analyze_ci_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """CI 결과 전체 분석"""
        dataset_results = results["dataset_results"]
        successful_datasets = [
            name for name, result in dataset_results.items()
            if result.get("status") == "passed"
        ]

        failed_datasets = [
            name for name, result in dataset_results.items()
            if result.get("status") == "failed"
        ]

        # 전체 평균 점수 계산
        overall_scores = []
        for dataset_result in dataset_results.values():
            if "overall_score" in dataset_result:
                overall_scores.append(dataset_result["overall_score"])

        analysis = {
            "total_datasets": len(dataset_results),
            "passed_datasets": len(successful_datasets),
            "failed_datasets": len(failed_datasets),
            "dataset_pass_rate": len(successful_datasets) / len(dataset_results) if dataset_results else 0,
        }

        if overall_scores:
            analysis["aggregate_score"] = np.mean(overall_scores)

        return analysis

    def _check_regression(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """성능 회귀 체크"""
        regression_analysis = {
            "baseline_available": False,
            "regression_detected": False,
            "performance_change": {}
        }

        # 베이스라인 점수가 있는 경우 비교
        if self.baseline_scores and "aggregate_score" in results:
            current_score = results["aggregate_score"]
            baseline_score = self.baseline_scores.get("aggregate_score")

            if baseline_score:
                regression_analysis["baseline_available"] = True
                performance_change = (current_score - baseline_score) / baseline_score
                regression_analysis["performance_change"] = {
                    "absolute": current_score - baseline_score,
                    "relative": performance_change,
                    "current_score": current_score,
                    "baseline_score": baseline_score
                }

                # 회귀 임계값 체크
                regression_threshold = self.config["regression_threshold"]
                if performance_change < -regression_threshold:  # 5% 이상 성능 저하
                    regression_analysis["regression_detected"] = True

        return regression_analysis

    def _determine_ci_status(self, results: Dict[str, Any]) -> str:
        """최종 CI 상태 결정"""
        # 회귀 감지시 실패
        if results.get("regression_analysis", {}).get("regression_detected", False):
            return "failed"

        # 데이터셋 통과율 확인
        pass_rate = results.get("dataset_pass_rate", 0)
        if pass_rate < 0.8:  # 80% 미만 통과시 실패
            return "failed"

        # 모든 데이터셋이 실패한 경우
        if results.get("passed_datasets", 0) == 0:
            return "failed"

        return "passed"

    def _send_notifications(self, results: Dict[str, Any]):
        """실패 시 알림 발송"""
        channels = self.config.get("notification_channels", ["console"])

        failure_message = self._format_failure_message(results)

        for channel in channels:
            if channel == "console":
                print(f"\n🚨 CI 평가 실패 알림:\n{failure_message}")
            elif channel == "email":
                # 실제 구현에서는 이메일 발송 로직 추가
                print(f"📧 이메일 알림 발송 (구현 필요): {failure_message}")

    def _format_failure_message(self, results: Dict[str, Any]) -> str:
        """실패 메시지 포맷팅"""
        message = [
            f"CI 평가 실패: {results['evaluation_id']}",
            f"시간: {results['timestamp']}",
            f"상태: {results['overall_status']}"
        ]

        if results.get("commit_id"):
            message.append(f"커밋: {results['commit_id']}")

        # 실패 원인
        if results.get("regression_analysis", {}).get("regression_detected"):
            change = results["regression_analysis"]["performance_change"]
            message.append(f"성능 회귀 감지: {change['relative']:.1%} 감소")

        # 실패한 데이터셋
        failed_datasets = [
            name for name, result in results["dataset_results"].items()
            if result.get("status") == "failed"
        ]

        if failed_datasets:
            message.append(f"실패한 데이터셋: {', '.join(failed_datasets)}")

        return "\n".join(message)

    def update_baseline(self, results: Dict[str, Any]):
        """베이스라인 점수 업데이트"""
        if results.get("overall_status") == "passed" and "aggregate_score" in results:
            self.baseline_scores["aggregate_score"] = results["aggregate_score"]
            self.baseline_scores["updated_at"] = results["timestamp"]
            print(f"✅ 베이스라인 업데이트: {results['aggregate_score']:.3f}")

# CI 평가 시스템 사용 예시
ci_system = ContinuousEvaluationSystem(langfuse_client)

# CI 평가 실행 (예: GitHub Actions에서)
ci_results = ci_system.run_ci_evaluation(
    rag_chain=rag_chain,
    commit_id="abc123def"  # Git 커밋 ID
)

# 성공한 경우 베이스라인 업데이트
if ci_results["overall_status"] == "passed":
    ci_system.update_baseline(ci_results)

print(f"\nCI 평가 결과: {ci_results['overall_status']}")
if ci_results["overall_status"] == "failed":
    print("❌ 빌드 실패 - 성능 기준을 충족하지 못했습니다")
    exit(1)  # CI 파이프라인 실패
else:
    print("✅ 빌드 성공 - 모든 품질 기준을 충족했습니다")
```

## 📖 참고 자료

### Langfuse 관련
- [Langfuse 공식 문서](https://langfuse.com/docs)
- [Langfuse LangChain 통합 가이드](https://langfuse.com/docs/integrations/langchain)
- [Langfuse Python SDK](https://langfuse.com/docs/sdk/python)

### 평가 방법론
- [LLM 애플리케이션 평가 best practices](https://langfuse.com/docs/evaluation)
- [프로덕션 LLM 모니터링 전략](https://docs.smith.langchain.com/monitoring)

### 메트릭과 평가 도구
- [ROUGE 점수 계산 라이브러리](https://github.com/neural-dialogue-metrics/rouge)
- [한국어 ROUGE 구현](https://github.com/gucci-j/korouge-score)
- [LangChain 평가 도구](https://python.langchain.com/docs/guides/evaluation/)

### 실무 모니터링
- [MLOps for LLM Applications](https://neptune.ai/blog/mlops-for-llm)
- [LLM 애플리케이션 CI/CD](https://docs.smith.langchain.com/evaluation)

이 가이드를 통해 Langfuse를 활용한 체계적인 RAG 시스템 평가와 모니터링 시스템을 구축할 수 있습니다. 실무에서는 지속적인 모니터링과 개선을 통해 LLM 애플리케이션의 품질을 유지하고 발전시키는 것이 중요합니다.