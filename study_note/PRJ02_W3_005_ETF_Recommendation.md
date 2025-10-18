# W3_005_ETF_Recommendation - LLM과 RAG 기반 ETF 추천 시스템

## 학습 목표

이 가이드에서는 LLM과 RAG를 결합한 지능형 ETF 추천 시스템 구축을 학습합니다:

- **전통적 추천 시스템 vs RAG**: 데이터 희소성, 콜드 스타트 문제 해결
- **하이브리드 검색**: BM25 키워드 검색 + 벡터 검색 결합
- **사용자 프로필 분석**: 투자 성향, 위험 선호도, 목표 자동 추출
- **다단계 추천 파이프라인**: 프로필 → 검색 → 랭킹 → 설명 생성
- **Gradio 인터페이스**: 대화형 웹 UI 구축 및 Hugging Face 배포

### 선수 지식
- Text2SQL 및 RAG 시스템 (W3_002, W3_003 참조)
- High Cardinality 처리 (W3_004 참조)
- LangGraph 상태 관리
- Pydantic 모델 정의

---

## 핵심 개념

### 전통적 추천 시스템의 문제점

#### 1. 데이터 희소성 (Data Sparsity)
사용자-아이템 상호작용 매트릭스가 대부분 빈 값으로 구성됩니다.

**문제**:
```python
# 사용자-ETF 평점 매트릭스
       ETF1  ETF2  ETF3  ETF4  ETF5
User1   5.0   NaN   NaN   NaN   NaN
User2   NaN   NaN   4.0   NaN   NaN
User3   NaN   NaN   NaN   NaN   3.0

# 희소도: 93% (28/30 값이 비어있음)
```

**영향**:
- 협업 필터링 정확도 저하
- 유사한 사용자/아이템 찾기 어려움
- 추천 다양성 감소

#### 2. 콜드 스타트 (Cold Start)
신규 사용자나 아이템에 대한 데이터가 없어 추천이 불가능합니다.

**문제 사례**:
- **신규 사용자**: 투자 이력 없음 → 추천 불가
- **신규 ETF**: 평점/리뷰 없음 → 추천 대상에서 제외
- **틈새 ETF**: 상호작용 부족 → 절대 추천되지 않음

#### 3. 명시적 피드백 부족
평점이나 리뷰 수집이 제한적입니다.

**현실**:
- 대부분 사용자는 평가하지 않음
- 극단적 경험만 리뷰 작성 (편향)
- ETF 투자는 장기적 → 즉각적 피드백 없음

---

### RAG 기반 추천 시스템의 강점

#### 1. 콘텐츠 기반 추천
사용자 이력 없이도 상품 특성으로 추천 가능합니다.

**접근법**:
```python
# 사용자 질문: "안정적인 배당 ETF 추천해주세요"

# RAG 프로세스:
1. 질문 분석 → 투자 성향 추출
2. 벡터 검색 → "배당", "안정적" 키워드 매칭
3. LLM 설명 → 각 ETF의 배당 특성 설명
```

#### 2. 자연어 기반 상호작용
복잡한 투자 선호도를 자연어로 표현 가능합니다.

**예시**:
```
전통적: [고위험 ✓] [성장형 ✓] [IT섹터 ✓]
RAG: "기술 혁신에 투자하고 싶지만 너무 변동성이 크지 않았으면 좋겠어요.
     AI와 클라우드 분야에 관심이 있고, 월 50만원씩 10년 이상 투자할 계획입니다."
```

#### 3. 풍부한 설명 생성
추천 이유를 자세히 설명할 수 있습니다.

**전통적 시스템**:
```
추천: KODEX 200 (평점 4.5/5.0)
```

**RAG 시스템**:
```
추천: KODEX 200

추천 이유:
1. 투자 전략: 코스피 200 지수 추종으로 한국 대표 기업에 분산 투자
2. 안정성: 낮은 변동성 (베타 0.95)과 높은 순자산 (10조원)
3. 비용 효율: 총보수 0.15%로 업계 평균 대비 저렴
4. 적합성: 귀하의 중위험 선호도와 장기 투자 계획에 부합
```

---

### 하이브리드 검색 전략

BM25 키워드 검색과 벡터 검색을 결합하여 검색 품질을 향상시킵니다.

#### BM25 검색 (Keyword-based)
**강점**:
- 정확한 용어 매칭 (예: "다우존스")
- 빠른 속도
- 해석 가능성

**약점**:
- 동의어 처리 어려움
- 맞춤법 오류 민감
- 의미적 유사성 무시

#### 벡터 검색 (Semantic)
**강점**:
- 의미적 유사성 포착
- 맞춤법 오류 허용
- 다국어 지원

**약점**:
- 정확한 매칭 부족 가능
- 느린 속도
- 블랙박스 특성

#### 앙상블 검색 (Hybrid)
두 방법의 장점을 결합합니다.

```python
from langchain.retrievers import EnsembleRetriever

ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]  # 동등한 가중치
)
```

**결과**:
- BM25: 정확한 키워드 매칭
- Vector: 의미적 유사성
- Ensemble: 두 결과의 가중 평균

---

## 환경 설정

### 필수 라이브러리

```bash
# LangChain 핵심
pip install langchain langchain-openai langchain-community
pip install langgraph

# 벡터 검색
pip install langchain-core

# BM25 검색
pip install rank-bm25

# 한국어 토크나이저
pip install kiwipiepy

# 웹 인터페이스
pip install gradio

# 환경 변수
pip install python-dotenv
```

### 한국어 토크나이저 설정

```python
from kiwipiepy import Kiwi
import re

# Kiwi 토크나이저 초기화
kiwi = Kiwi()

def korean_tokenizer(text: str) -> list[str]:
    """
    kiwipiepy를 사용한 한국어 토크나이징

    Parameters:
        text (str): 토크나이징할 텍스트

    Returns:
        list[str]: 토큰 리스트

    처리 과정:
    1. 특수문자 제거
    2. 소문자 변환
    3. 형태소 분석
    4. 명사/동사/형용사만 추출
    """
    # 전처리
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()

    # 형태소 분석
    tokens = kiwi.tokenize(text)

    # 필터링 (명사, 형용사, 동사, 외국어, 한자)
    filtered_tokens = [
        token.form for token in tokens
        if token.tag in ['NNG', 'NNP', 'VA', 'VV', 'SL', 'SH']
    ]

    return filtered_tokens if filtered_tokens else [token.form for token in tokens]

# 테스트
print(korean_tokenizer("다우존스 관련 ETF는 무엇인가요?"))
# ['다우존스', '관련', 'etf', '무엇']
```

---

## 단계별 구현

### 준비: 데이터베이스 및 고유명사 추출

#### 데이터베이스 연결
```python
from langchain_community.utilities import SQLDatabase

# ETF 데이터베이스 연결
db = SQLDatabase.from_uri("sqlite:///etf_database.db")

print(f"Dialect: {db.dialect}")
print(f"Tables: {db.get_usable_table_names()}")
```

#### 고유명사 추출
```python
import ast
import re

def query_as_list(db, query: str) -> list[str]:
    """
    DB 쿼리 결과를 고유명사 리스트로 변환

    Parameters:
        db: SQLDatabase 객체
        query: SQL 쿼리

    Returns:
        list[str]: 중복 제거된 고유명사 리스트
    """
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    # 숫자 제거
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return list(set(res))

# 고유명사 추출
etfs = query_as_list(db, "SELECT DISTINCT 종목명 FROM ETFs")
fund_managers = query_as_list(db, "SELECT DISTINCT 운용사 FROM ETFs")
underlying_assets = query_as_list(db, "SELECT DISTINCT 기초지수 FROM ETFs")

print(f"ETFs: {len(etfs)}")
print(f"Fund Managers: {len(fund_managers)}")
print(f"Underlying Assets: {len(underlying_assets)}")
```

---

### 1단계: 하이브리드 검색 시스템 구축

#### 벡터 검색 설정
```python
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# 임베딩 모델 생성
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# 벡터 저장소 생성
vector_store = InMemoryVectorStore(embeddings)

# 고유명사 임베딩
all_entities = etfs + fund_managers + underlying_assets
_ = vector_store.add_texts(all_entities)

# 벡터 검색기
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 20})

print(f"✅ 벡터 저장소 생성 완료: {len(all_entities)}개 엔티티")
```

#### BM25 검색 설정
```python
from langchain_community.retrievers import BM25Retriever

def korean_bm25_from_texts(texts: list[str], **kwargs):
    """한국어 토크나이저를 사용하는 BM25 리트리버 생성"""
    return BM25Retriever.from_texts(
        texts,
        preprocess_func=korean_tokenizer,
        **kwargs
    )

# BM25 검색기
bm25_retriever = korean_bm25_from_texts(
    all_entities,
    k=20
)

print("✅ BM25 검색기 생성 완료")
```

#### 하이브리드 앙상블 검색
```python
from langchain.retrievers import EnsembleRetriever

# 앙상블 검색기 (50:50 가중치)
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)

# 테스트
query = "다우존스 관련 ETF는 무엇인가요?"
results = ensemble_retriever.get_relevant_documents(query)

print(f"Query: {query}")
print("-" * 60)
for i, result in enumerate(results[:5], 1):
    print(f"{i}. {result.page_content}")
```

**출력 예시**:
```
Query: 다우존스 관련 ETF는 무엇인가요?
------------------------------------------------------------
1. SOL 미국배당다우존스
2. KODEX 미국배당다우존스
3. TIGER 미국다우존스30
4. PLUS 미국다우존스고배당주(합성 H)
5. ACE 미국배당다우존스
```

#### 검색 도구 생성
```python
from langchain.agents.agent_toolkits import create_retriever_tool

description = (
    "Use to look up values to filter on. Input is an approximate spelling "
    "of the proper noun, output is valid proper nouns. Use the noun most "
    "similar to the search."
)

entity_retriever_tool = create_retriever_tool(
    ensemble_retriever,
    name="search_proper_nouns",
    description=description
)

print("✅ 검색 도구 생성 완료")
```

---

### 2단계: 상태(State) 관리 정의

추천 시스템의 전체 상태를 관리하는 TypedDict를 정의합니다.

```python
from typing import TypedDict

class State(TypedDict):
    """ETF 추천 시스템의 상태 정의"""
    question: str          # 사용자 입력 질문
    user_profile: dict     # 사용자 프로필 정보
    query: str             # 생성된 SQL 쿼리
    candidates: list       # 후보 ETF 목록
    rankings: list         # 순위가 매겨진 ETF 목록
    explanation: str       # 추천 이유 설명
    final_answer: str      # 최종 추천 답변
```

**상태 흐름**:
```
question → user_profile → query → candidates → rankings → explanation → final_answer
```

---

### 3단계: 사용자 프로필 분석

사용자의 질문에서 투자 성향을 자동으로 추출합니다.

#### Pydantic 모델 정의
```python
from enum import Enum
from typing import List
from pydantic import BaseModel, Field

class RiskTolerance(Enum):
    """위험 선호도"""
    CONSERVATIVE = "conservative"  # 보수적
    MODERATE = "moderate"          # 중립적
    AGGRESSIVE = "aggressive"      # 공격적

class InvestmentHorizon(Enum):
    """투자 기간"""
    SHORT = "short"    # 단기 (1년 미만)
    MEDIUM = "medium"  # 중기 (1-5년)
    LONG = "long"      # 장기 (5년 이상)

class InvestmentProfile(BaseModel):
    """투자자 프로필"""
    risk_tolerance: RiskTolerance = Field(
        description="투자자의 위험 성향 (conservative/moderate/aggressive)"
    )
    investment_horizon: InvestmentHorizon = Field(
        description="투자 기간 (short/medium/long)"
    )
    investment_goal: str = Field(
        description="투자의 주요 목적 설명"
    )
    preferred_sectors: List[str] = Field(
        description="선호하는 투자 섹터 목록"
    )
    excluded_sectors: List[str] = Field(
        description="투자를 원하지 않는 섹터 목록"
    )
    monthly_investment: int = Field(
        description="월 투자 가능 금액 (원)"
    )
```

#### 프로필 분석 함수
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 프롬프트 템플릿
PROFILE_TEMPLATE = """
사용자의 질문을 분석하여 투자 프로필을 생성해주세요.

사용자 질문: {question}

분석 지침:
1. 명시적으로 언급된 정보를 우선 사용
2. 언급되지 않은 경우 질문의 맥락에서 추론
3. 불확실한 경우 보수적으로 설정

출력: InvestmentProfile JSON
"""

profile_prompt = ChatPromptTemplate.from_template(PROFILE_TEMPLATE)

# LLM with structured output
llm = ChatOpenAI(model="gpt-4o-mini")
profile_llm = llm.with_structured_output(InvestmentProfile)

def analyze_profile(state: State) -> dict:
    """
    사용자 질문을 분석하여 투자 프로필 생성

    Parameters:
        state (State): 현재 상태 (question 포함)

    Returns:
        dict: 업데이트된 상태 {'user_profile': {...}}
    """
    prompt = profile_prompt.invoke({"question": state["question"]})
    response = profile_llm.invoke(prompt)
    return {"user_profile": dict(response)}

# 테스트
question = """
저는 30대 초반의 직장인입니다.
월 100만원 정도를 3년 이상 장기 투자하고 싶고,
기술 섹터와 헬스케어에 관심이 있습니다.
중위험 중수익을 추구하며, ESG 요소도 고려하고 싶습니다.
적합한 ETF를 추천해주세요.
"""

result = analyze_profile({"question": question})
print(result["user_profile"])
```

**출력**:
```python
{
    'risk_tolerance': 'moderate',
    'investment_horizon': 'long',
    'investment_goal': '기술과 헬스케어 섹터의 성장을 통한 장기 자산 증식',
    'preferred_sectors': ['기술', '헬스케어', 'ESG'],
    'excluded_sectors': [],
    'monthly_investment': 1000000
}
```

---

### 4단계: ETF 검색 쿼리 생성

사용자 프로필을 기반으로 SQL 쿼리를 생성합니다.

#### 쿼리 생성 프롬프트
```python
QUERY_TEMPLATE = """
Given an input question and investment profile, create a syntactically correct {dialect} query to run.
Unless specified, limit your query to at most {top_k} results.
Order the results by most relevant columns based on the investment profile.

Never query for all columns from a specific table, only ask for relevant columns given the question and investment criteria.

Available tables:
{table_info}

Entity relationships (from hybrid search):
{entity_info}

## Matching Guidelines
- Use exact matches when comparing entity names
- Handle both Korean and English entity names
- Consider risk tolerance when filtering by 변동성
- Match investment horizon with appropriate ETF types

## Investment Profile Considerations
User Profile: {user_profile}

Risk Tolerance Mapping:
- conservative → 변동성 = '매우낮음' OR '낮음'
- moderate → 변동성 = '낮음' OR '보통'
- aggressive → 변동성 = '보통' OR '높음' OR '매우높음'

Sector Mapping:
- 기술 → 분류체계 LIKE '%정보기술%' OR '%IT%' OR '%반도체%'
- 헬스케어 → 분류체계 LIKE '%헬스케어%' OR '%바이오%'

Question: {input}
"""

query_prompt = ChatPromptTemplate.from_template(QUERY_TEMPLATE)
```

#### 쿼리 생성 함수
```python
from typing import Annotated

class QueryOutput(TypedDict):
    """Generated SQL query"""
    query: Annotated[str, ..., "Syntactically valid SQL query."]

def write_query(state: State) -> dict:
    """
    사용자 프로필 기반 SQL 쿼리 생성

    Parameters:
        state (State): 현재 상태 (question, user_profile 포함)

    Returns:
        dict: {'query': 'SELECT ...'}
    """
    # 엔티티 검색
    entity_info = entity_retriever_tool.invoke(state["question"])

    # 프롬프트 생성
    prompt = query_prompt.invoke({
        "dialect": db.dialect,
        "top_k": 20,
        "table_info": db.get_table_info(),
        "entity_info": entity_info,
        "user_profile": state["user_profile"],
        "input": state["question"]
    })

    # SQL 생성
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)

    return {"query": result["query"]}

# 테스트
result = write_query({
    "question": question,
    "user_profile": result["user_profile"]
})
print(f"생성된 쿼리:\n{result['query']}")
```

**출력 예시**:
```sql
SELECT 종목코드, 종목명, 운용사, 분류체계, 수익률_최근1년, 총보수, 변동성, 순자산총액
FROM ETFs
WHERE (분류체계 LIKE '%정보기술%' OR 분류체계 LIKE '%헬스케어%')
  AND 변동성 IN ('낮음', '보통')
  AND 총보수 < 0.5
ORDER BY 수익률_최근1년 DESC, 순자산총액 DESC
LIMIT 20;
```

---

### 5단계: 쿼리 실행 및 후보 ETF 검색

```python
from langchain_community.tools import QuerySQLDatabaseTool

def execute_query(state: State) -> dict:
    """
    SQL 쿼리 실행하여 후보 ETF 검색

    Parameters:
        state (State): 현재 상태 (query 포함)

    Returns:
        dict: {'candidates': '[(...), (...), ...]'}
    """
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    results = execute_query_tool.invoke(state["query"])
    return {"candidates": results}

# 테스트
result = execute_query({"query": result["query"]})
print(f"후보 ETF 수: {len(eval(result['candidates']))}")
print(f"\n첫 3개:")
for etf in eval(result["candidates"])[:3]:
    print(etf)
```

---

### 6단계: 후보 ETF 랭킹 및 필터링

여러 지표를 종합하여 상위 ETF를 선정합니다.

#### 랭킹 프롬프트
```python
RANKING_TEMPLATE = """
Rank the following ETF candidates based on the user's investment profile and return the top 3 ETFs.

Consider these factors when ranking:
1. 수익률 (Return): 높을수록 좋음
2. 변동성 (Volatility): 사용자 위험 선호도와 일치
3. 순자산총액 (AUM): 클수록 안정적
4. 총보수 (Expense Ratio): 낮을수록 좋음
5. Profile matching score: 선호 섹터 일치도

Scoring Guidelines:
- Conservative: 변동성 가중치 높음, 수익률 가중치 낮음
- Moderate: 균형 잡힌 가중치
- Aggressive: 수익률 가중치 높음, 변동성 가중치 낮음

User Profile:
{user_profile}

Candidate ETFs:
{candidates}

Return the top 3 ETFs with:
- rank: 순위 (1-3)
- etf_code: 종목코드
- etf_name: 종목명
- score: 종합 점수 (0-100)
- ranking_reason: 선정 이유 (한국어)
"""

ranking_prompt = ChatPromptTemplate.from_template(RANKING_TEMPLATE)
```

#### 랭킹 모델 정의
```python
from typing import List

class ETFRanking(TypedDict):
    """Individual ETF ranking result"""
    rank: Annotated[int, ..., "Ranking position (1-3)"]
    etf_code: Annotated[str, ..., "ETF 종목코드"]
    etf_name: Annotated[str, ..., "ETF 종목명"]
    score: Annotated[float, ..., "Composite score (0-100)"]
    ranking_reason: Annotated[str, ..., "선정 이유 (한국어)"]

class ETFRankingResult(TypedDict):
    """Ranked ETFs"""
    rankings: List[ETFRanking]
```

#### 랭킹 함수
```python
def rank_etfs(state: State) -> dict:
    """
    후보 ETF를 랭킹하여 상위 3개 선정

    Parameters:
        state (State): 현재 상태 (user_profile, candidates 포함)

    Returns:
        dict: {'rankings': [ETFRanking, ...]}
    """
    prompt = ranking_prompt.invoke({
        "user_profile": state["user_profile"],
        "candidates": state["candidates"]
    })

    ranking_llm = llm.with_structured_output(ETFRankingResult)
    result = ranking_llm.invoke(prompt)

    return {"rankings": result["rankings"]}

# 테스트
result = rank_etfs({
    "user_profile": profile,
    "candidates": candidates
})

for ranking in result["rankings"]:
    print(f"Rank {ranking['rank']}: {ranking['etf_name']} (점수: {ranking['score']})")
    print(f"  이유: {ranking['ranking_reason']}\n")
```

**출력 예시**:
```
Rank 1: KODEX IT (점수: 92.5)
  이유: 정보기술 섹터 집중 투자로 사용자 선호도와 완벽히 일치하며,
       최근 1년 수익률 45%로 우수한 성과를 보임. 순자산 5조원으로 안정적.

Rank 2: TIGER 헬스케어 (점수: 88.3)
  이유: 헬스케어 섹터 투자 목표에 부합하며, 중간 변동성으로
       중위험 선호도와 일치. 총보수 0.25%로 저렴.

Rank 3: ACE ESG리더스 (점수: 85.1)
  이유: ESG 요소 고려 요구사항 충족. 기술과 헬스케어 포함된
       다각화 포트폴리오로 리스크 분산 효과.
```

---

### 7단계: 추천 설명 생성

상위 ETF에 대한 상세한 설명을 생성합니다.

#### 설명 생성 프롬프트
```python
EXPLANATION_TEMPLATE = """
Please provide a comprehensive explanation for the recommended ETFs based on the user's investment profile.

Structure your explanation as follows:

## 1. ETF 특성 분석
각 추천 ETF에 대해:
- 투자 전략 및 접근법
- 과거 성과 개요
- 수수료 구조 및 효율성
- 기초 자산 및 분산 투자 효과

## 2. 프로필 적합성 분석
- 위험 선호도와의 일치도
- 투자 기간과의 부합성
- 선호 섹터와의 연관성
- 투자 목표 달성 기여도

## 3. 포트폴리오 구성안
- 권장 배분 비율 (%)
- 분산 투자 효과
- 리밸런싱 고려사항
- 실행 전략

## 4. 리스크 고려사항
- 시장 위험 요소
- 개별 ETF 리스크
- 경제 시나리오별 영향
- 모니터링 요구사항

---

User Profile:
{user_profile}

Ranked ETFs:
{rankings}

Generate a detailed explanation in Korean.
"""

explanation_prompt = ChatPromptTemplate.from_template(EXPLANATION_TEMPLATE)
```

#### 설명 생성 함수
```python
def generate_explanation(state: State) -> dict:
    """
    추천 ETF에 대한 상세 설명 생성

    Parameters:
        state (State): 현재 상태 (user_profile, rankings 포함)

    Returns:
        dict: {'explanation': '...', 'final_answer': '...'}
    """
    prompt = explanation_prompt.invoke({
        "user_profile": state["user_profile"],
        "rankings": state["rankings"]
    })

    response = llm.invoke(prompt)
    explanation = response.content

    # 최종 답변 포맷팅
    final_answer = f"""
# ETF 투자 추천

{explanation}

---

**추천 요약**:
"""
    for ranking in state["rankings"]:
        final_answer += f"\n{ranking['rank']}. {ranking['etf_name']} ({ranking['etf_code']}) - 점수: {ranking['score']}"

    return {
        "explanation": explanation,
        "final_answer": final_answer
    }

# 테스트
result = generate_explanation({
    "user_profile": profile,
    "rankings": rankings
})
print(result["final_answer"])
```

---

### 8단계: LangGraph 통합

모든 노드를 연결하여 완전한 파이프라인을 구성합니다.

```python
from langgraph.graph import StateGraph, START, END

# 상태 그래프 생성
graph_builder = StateGraph(State)

# 노드 추가
graph_builder.add_node("analyze_profile", analyze_profile)
graph_builder.add_node("write_query", write_query)
graph_builder.add_node("execute_query", execute_query)
graph_builder.add_node("rank_etfs", rank_etfs)
graph_builder.add_node("generate_explanation", generate_explanation)

# 엣지 연결
graph_builder.add_edge(START, "analyze_profile")
graph_builder.add_edge("analyze_profile", "write_query")
graph_builder.add_edge("write_query", "execute_query")
graph_builder.add_edge("execute_query", "rank_etfs")
graph_builder.add_edge("rank_etfs", "generate_explanation")
graph_builder.add_edge("generate_explanation", END)

# 그래프 컴파일
graph = graph_builder.compile()

print("✅ ETF 추천 시스템 그래프 생성 완료")
```

#### 그래프 시각화
```python
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))
```

#### 전체 파이프라인 실행
```python
# 사용자 질문
question = """
40대 중반이고 은퇴 자금 마련이 목표입니다.
월 200만원씩 15년 이상 투자할 계획이고,
안정적인 배당 수익을 원합니다.
변동성은 낮았으면 좋겠습니다.
"""

# 전체 파이프라인 실행
final_state = graph.invoke({"question": question})

# 결과 출력
print("=" * 80)
print("사용자 프로필:")
print(final_state["user_profile"])
print("\n" + "=" * 80)
print("생성된 SQL 쿼리:")
print(final_state["query"])
print("\n" + "=" * 80)
print("최종 추천:")
print(final_state["final_answer"])
```

---

## Gradio 인터페이스 구현

### 1단계: Gradio 앱 작성

`app.py` 파일 생성:

```python
import gradio as gr
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# ETF 추천 시스템 초기화 (이전 코드)
# ... (db, retriever, graph 등 초기화)

def recommend_etf(user_question: str, chat_history: list) -> tuple:
    """
    사용자 질문에 대한 ETF 추천 생성

    Parameters:
        user_question (str): 사용자 질문
        chat_history (list): 대화 이력

    Returns:
        tuple: (chat_history, "")
    """
    # ETF 추천 실행
    result = graph.invoke({"question": user_question})

    # 대화 이력에 추가
    chat_history.append((user_question, result["final_answer"]))

    return chat_history, ""

# Gradio 인터페이스
with gr.Blocks(title="ETF 투자 추천 시스템") as demo:
    gr.Markdown("# 🤖 AI 기반 ETF 투자 추천 시스템")
    gr.Markdown("""
    투자 목표, 위험 선호도, 선호 섹터를 자연어로 설명해주세요.
    AI가 맞춤형 ETF를 추천해드립니다.

    **예시 질문**:
    - "안정적인 배당 수익을 위한 ETF 추천해주세요"
    - "기술 섹터에 투자하고 싶은데, 변동성은 낮았으면 좋겠어요"
    - "은퇴 자금 마련을 위해 15년 이상 장기 투자할 계획입니다"
    """)

    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(
        placeholder="투자 목표와 선호도를 자유롭게 설명해주세요...",
        label="질문 입력"
    )

    with gr.Row():
        submit_btn = gr.Button("추천 받기", variant="primary")
        clear_btn = gr.Button("대화 초기화")

    # 이벤트 핸들러
    submit_btn.click(
        recommend_etf,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg]
    )
    msg.submit(
        recommend_etf,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg]
    )
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])

if __name__ == "__main__":
    demo.launch()
```

### 2단계: 로컬 테스트

```bash
# 가상 환경에서 실행
uv run app.py

# 또는
python app.py
```

브라우저에서 `http://localhost:7860` 접속

### 3단계: 배포 준비

#### `requirements.txt` 생성:
```
gradio>=5.34.2
langchain>=0.3.18
langchain-openai>=0.3.3
langchain-community>=0.3.16
langgraph>=0.3.9
python-dotenv>=1.0.0
kiwipiepy>=0.18.1
rank-bm25>=0.2.2
```

#### `.gitignore` 생성:
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/

# Environment
.env
.venv
venv/

# IDE
.vscode/
.idea/

# Data
*.db
*.csv
```

### 4단계: Hugging Face Spaces 배포

```bash
# Gradio CLI로 배포
uv run gradio deploy

# 또는
gradio deploy
```

**배포 과정**:
1. Hugging Face 토큰 입력
2. Space 이름 입력
3. Secrets 설정 (OPENAI_API_KEY)
4. 자동 배포 시작
5. URL 제공

**배포 후 관리**:
- Hugging Face Spaces 대시보드에서 관리
- Settings → Secrets에서 환경 변수 수정
- Files → app.py에서 코드 수정
- Logs 탭에서 오류 확인

---

## 실전 활용 예제

### 예제 1: 다양한 투자 시나리오

```python
# 시나리오 1: 보수적 투자자
question_conservative = """
50대 후반이고 은퇴가 가까워졌습니다.
월 300만원씩 5년 정도 투자하려고 하는데,
원금 손실이 거의 없고 안정적인 배당 수익을 원합니다.
"""

# 시나리오 2: 공격적 투자자
question_aggressive = """
20대 초반이고 장기 투자 가능합니다.
월 50만원씩 20년 이상 투자할 계획이고,
높은 수익률을 위해 변동성을 감수할 수 있습니다.
기술주와 성장주에 관심이 많습니다.
"""

# 시나리오 3: ESG 투자자
question_esg = """
환경과 사회적 책임을 중요하게 생각합니다.
ESG 등급이 높고 지속가능한 기업에 투자하고 싶습니다.
월 100만원씩 10년 정도 투자할 계획입니다.
"""

# 각 시나리오 실행
for scenario_name, question in [
    ("보수적", question_conservative),
    ("공격적", question_aggressive),
    ("ESG", question_esg)
]:
    print(f"\n{'=' * 80}")
    print(f"시나리오: {scenario_name} 투자자")
    print('=' * 80)

    result = graph.invoke({"question": question})
    print(result["final_answer"])
```

### 예제 2: 포트폴리오 리밸런싱 조언

```python
def portfolio_rebalancing_advice(current_portfolio: dict, question: str):
    """
    현재 포트폴리오를 고려한 추천

    Parameters:
        current_portfolio (dict): {
            'etf_code': allocation_percentage
        }
        question (str): 리밸런싱 질문

    Returns:
        str: 리밸런싱 조언
    """
    # 질문에 현재 포트폴리오 정보 추가
    enhanced_question = f"""
    현재 제 포트폴리오는 다음과 같습니다:
    {', '.join([f'{k}: {v}%' for k, v in current_portfolio.items()])}

    {question}
    """

    result = graph.invoke({"question": enhanced_question})
    return result["final_answer"]

# 사용 예시
current = {
    "KODEX 200": 40,
    "TIGER 미국S&P500": 30,
    "ACE 채권혼합": 30
}

advice = portfolio_rebalancing_advice(
    current,
    "기술 섹터 비중을 늘리고 싶은데 어떻게 조정하면 좋을까요?"
)
print(advice)
```

### 예제 3: 정기 투자 시뮬레이션

```python
def investment_simulation(
    monthly_amount: int,
    years: int,
    etf_recommendations: list
):
    """
    정기 투자 시뮬레이션

    Parameters:
        monthly_amount (int): 월 투자 금액
        years (int): 투자 기간
        etf_recommendations (list): 추천 ETF 목록

    Returns:
        dict: 시뮬레이션 결과
    """
    total_investment = monthly_amount * 12 * years

    simulation = {
        "total_investment": total_investment,
        "projected_returns": {}
    }

    for etf in etf_recommendations:
        # 각 ETF의 과거 평균 수익률 가정
        avg_return = etf.get("avg_return", 0.07)  # 기본 7%

        # 복리 계산
        future_value = monthly_amount * (
            ((1 + avg_return/12) ** (years * 12) - 1) / (avg_return/12)
        )

        simulation["projected_returns"][etf["etf_name"]] = {
            "future_value": future_value,
            "total_return": future_value - total_investment,
            "return_rate": (future_value / total_investment - 1) * 100
        }

    return simulation

# 사용 예시
recommendations = graph.invoke({"question": question})["rankings"]
simulation = investment_simulation(
    monthly_amount=1000000,
    years=10,
    etf_recommendations=recommendations
)

print(f"총 투자 금액: {simulation['total_investment']:,}원")
for etf_name, projection in simulation["projected_returns"].items():
    print(f"\n{etf_name}:")
    print(f"  예상 자산: {projection['future_value']:,.0f}원")
    print(f"  수익: {projection['total_return']:,.0f}원")
    print(f"  수익률: {projection['return_rate']:.1f}%")
```

---

## 연습 문제

### 기본 문제

**문제 1**: 프로필 분석 개선
```python
# 과제: 나이대별 기본 프로필 추가

# TODO: 나이를 추출하여 기본 프로필 설정
# - 20대: aggressive, long
# - 30대: moderate, long
# - 40대: moderate, medium
# - 50대+: conservative, short
```

**문제 2**: 섹터 매핑 확장
```python
# 과제: 더 많은 섹터 키워드 매핑

sector_mappings = {
    '기술': ['%IT%', '%반도체%', '%소프트웨어%'],
    '헬스케어': ['%헬스케어%', '%바이오%', '%제약%'],
    # TODO: 추가 섹터 정의
}
```

**문제 3**: 검색 품질 평가
```python
# 과제: BM25 vs Vector vs Hybrid 비교

def compare_retrievers(query: str):
    """
    세 가지 검색 방법 비교

    TODO:
    1. BM25 검색 결과
    2. Vector 검색 결과
    3. Hybrid 검색 결과
    4. 비교 분석
    """
    pass
```

### 중급 문제

**문제 4**: 동적 가중치 조정
```python
# 과제: 사용자 피드백 기반 가중치 조정

class AdaptiveEnsemble:
    def __init__(self):
        self.bm25_weight = 0.5
        self.vector_weight = 0.5

    def adjust_weights(self, feedback: dict):
        """
        사용자 피드백으로 가중치 조정

        TODO:
        1. 피드백 분석
        2. 가중치 업데이트
        3. 새로운 앙상블 생성
        """
        pass
```

**문제 5**: 멀티 모델 랭킹
```python
# 과제: 여러 LLM 모델의 랭킹 결합

def multi_model_ranking(candidates: list):
    """
    GPT, Claude, Gemini 랭킹 결합

    TODO:
    1. 각 모델로 랭킹 생성
    2. 결과 정규화
    3. 앙상블 랭킹
    """
    pass
```

**문제 6**: A/B 테스트 시스템
```python
# 과제: 추천 알고리즘 A/B 테스트

class ABTest:
    def __init__(self, variant_a, variant_b):
        self.variant_a = variant_a
        self.variant_b = variant_b
        self.results = {"a": [], "b": []}

    def run_test(self, user_question: str):
        """
        두 변형 동시 실행 및 비교

        TODO:
        1. 변형 A, B 실행
        2. 결과 저장
        3. 통계 분석
        """
        pass
```

### 고급 문제

**문제 7**: 강화 학습 기반 추천
```python
# 과제: 사용자 피드백으로 개선하는 시스템

class RLRecommender:
    def __init__(self):
        self.policy = {}  # 상태 → 액션 매핑

    def get_recommendation(self, state: dict):
        """
        현재 상태에서 최적 추천

        TODO:
        1. 상태 표현 정의
        2. Q-learning 구현
        3. 피드백 학습
        """
        pass

    def update_policy(self, state, action, reward):
        """정책 업데이트"""
        pass
```

**문제 8**: 설명 가능한 AI
```python
# 과제: 추천 결정 과정 시각화

def explain_recommendation(ranking: dict):
    """
    추천 이유를 상세히 설명

    TODO:
    1. 각 요소의 기여도 계산
    2. SHAP 값 또는 LIME 적용
    3. 시각화 생성
    """
    pass
```

**문제 9**: 실시간 데이터 통합
```python
# 과제: 실시간 시장 데이터로 추천 개선

import yfinance as yf

class RealTimeRecommender:
    def update_market_data(self):
        """
        실시간 시장 데이터 가져오기

        TODO:
        1. yfinance로 실시간 가격 조회
        2. 데이터베이스 업데이트
        3. 캐시 무효화
        """
        pass

    def adjust_for_market_conditions(self, recommendations):
        """
        시장 상황 고려 조정

        TODO:
        1. 현재 시장 동향 분석
        2. 추천 가중치 조정
        3. 리스크 경고 추가
        """
        pass
```

---

## 문제 해결 가이드

### 일반적인 문제

#### 1. 프로필 추출 실패
```python
# 문제: 질문에서 충분한 정보를 추출하지 못함

# 해결: 프롬프트에 예제 추가
PROFILE_TEMPLATE = """
... (기존 내용)

예시:
질문: "안정적인 배당 수익을 원합니다"
→ risk_tolerance: conservative
→ preferred_sectors: ["배당"]

질문: "기술주에 장기 투자하고 싶습니다"
→ risk_tolerance: aggressive
→ investment_horizon: long
→ preferred_sectors: ["기술"]
"""
```

#### 2. SQL 쿼리 오류
```python
# 문제: 생성된 SQL이 실행 오류 발생

# 해결 1: 쿼리 검증 추가
def validate_query(query: str) -> bool:
    """SQL 쿼리 사전 검증"""
    try:
        # 읽기 전용 쿼리 확인
        if not query.strip().upper().startswith("SELECT"):
            return False

        # 위험한 키워드 확인
        dangerous = ["DROP", "DELETE", "UPDATE", "INSERT"]
        if any(kw in query.upper() for kw in dangerous):
            return False

        return True
    except:
        return False

# 해결 2: 재시도 메커니즘
def write_query_with_retry(state: State, max_retries: int = 3):
    """쿼리 생성 재시도"""
    for attempt in range(max_retries):
        result = write_query(state)
        if validate_query(result["query"]):
            return result

        print(f"재시도 {attempt + 1}/{max_retries}")

    raise ValueError("유효한 쿼리 생성 실패")
```

#### 3. 검색 결과 없음
```python
# 문제: 후보 ETF가 0개

# 해결: 조건 완화
def execute_query_with_fallback(state: State):
    """조건 완화 재검색"""
    result = execute_query(state)

    if len(eval(result["candidates"])) == 0:
        # 조건 완화된 쿼리 생성
        relaxed_query = state["query"].replace("AND", "OR")
        result = execute_query({"query": relaxed_query})

    return result
```

### 성능 최적화

#### 캐싱 전략
```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=100)
def cached_profile_analysis(question_hash: str):
    """프로필 분석 캐싱"""
    # 실제로는 question을 사용
    return analyze_profile({"question": question})

def analyze_profile_cached(state: State):
    """캐시를 활용한 프로필 분석"""
    question_hash = hashlib.md5(
        state["question"].encode()
    ).hexdigest()

    return cached_profile_analysis(question_hash)
```

#### 배치 처리
```python
def batch_recommendations(questions: list[str]):
    """여러 질문을 배치 처리"""
    # 프로필 분석 병렬 실행
    profiles = [analyze_profile({"question": q}) for q in questions]

    # 쿼리 생성 병렬 실행
    queries = [
        write_query({"question": q, "user_profile": p})
        for q, p in zip(questions, profiles)
    ]

    # 결과 수집
    return queries
```

---

## 추가 학습 자료

### 공식 문서
- [LangChain Retrievers](https://python.langchain.com/docs/modules/data_connection/retrievers/)
- [Gradio Documentation](https://www.gradio.app/docs/)
- [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces)
- [Pydantic Models](https://docs.pydantic.dev/)

### 추천 시스템 논문
- "Deep Learning based Recommender System" (Zhang et al., 2019)
- "Neural Collaborative Filtering" (He et al., 2017)
- "BERT4Rec: Sequential Recommendation with BERT" (Sun et al., 2019)

### 다음 단계
1. **사용자 피드백 루프**: 평가 시스템 구축
2. **실시간 업데이트**: 시장 데이터 자동 갱신
3. **멀티 모달 추천**: 차트, 뉴스 통합
4. **개인화 강화**: 사용자별 학습
5. **프로덕션 배포**: 스케일링 및 모니터링

### 심화 주제
- **Contextual Bandits**: 탐색-활용 트레이드오프
- **Sequential Recommendation**: 시간 순서 고려
- **Cross-domain Recommendation**: 도메인 간 지식 전이
- **Explainable AI**: 추천 이유 설명
- **Fairness in Recommendation**: 편향 제거

---

## 요약

이 가이드에서 학습한 핵심 내용:

✅ **전통적 추천 vs RAG 추천**
- 데이터 희소성, 콜드 스타트 문제 해결
- 자연어 기반 풍부한 상호작용
- LLM으로 설명 가능한 추천

✅ **하이브리드 검색 시스템**
- BM25 키워드 검색 + 벡터 검색
- 한국어 토크나이저 (kiwipiepy)
- 앙상블 리트리버로 정확도 향상

✅ **다단계 추천 파이프라인**
- 프로필 분석 → 쿼리 생성 → 검색 → 랭킹 → 설명
- LangGraph로 상태 관리
- Pydantic으로 구조화된 출력

✅ **웹 인터페이스 구축**
- Gradio 대화형 UI
- Hugging Face Spaces 배포
- 사용자 친화적 경험

이제 실제 사용 가능한 AI 기반 ETF 추천 시스템을 구축할 수 있습니다!
