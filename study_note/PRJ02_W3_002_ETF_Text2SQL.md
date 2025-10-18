# W3_002_ETF_Text2SQL - ETF 데이터베이스 기반 Text2SQL 구현

## 학습 목표

이 가이드에서는 ETF 데이터를 활용한 Text2SQL 시스템 구현을 학습합니다:

- **SQLite 데이터베이스 설계**: ETF 목록과 상세 정보를 위한 관계형 데이터베이스 스키마 설계
- **데이터 전처리 및 저장**: Pandas를 활용한 데이터 타입 변환 및 DB 적재
- **LangChain SQL 체인**: 자연어 질의를 SQL 쿼리로 자동 변환하는 체인 구성
- **다중 모델 비교**: GPT와 Gemini 모델의 SQL 생성 성능 비교
- **쿼리 추출 및 정리**: 정규식을 활용한 SQL 쿼리 파싱 및 정제

### 선수 지식
- Python 기본 문법 및 Pandas 데이터 처리
- SQLite 기본 쿼리 작성 능력
- LangChain 기초 개념 이해
- 환경 변수 설정 경험

---

## 핵심 개념

### Text2SQL이란?
자연어 질문을 SQL 쿼리로 자동 변환하는 AI 기술입니다. 비개발자도 데이터베이스에 직접 접근하여 원하는 정보를 조회할 수 있도록 합니다.

**주요 특징**:
- 🔍 **자연어 인터페이스**: SQL 문법 몰라도 데이터베이스 조회 가능
- 🤖 **LLM 기반 변환**: GPT/Gemini 등 대규모 언어 모델이 쿼리 생성
- 📊 **스키마 인식**: 데이터베이스 구조를 이해하고 적절한 쿼리 작성
- ⚡ **실시간 응답**: 질문 즉시 SQL 생성 및 실행

### SQLite 데이터베이스
경량 파일 기반 관계형 데이터베이스로, 별도 서버 설치 없이 사용 가능합니다.

**장점**:
- 📦 설치 불필요 (Python 기본 포함)
- 💾 단일 파일로 데이터베이스 관리
- 🚀 빠른 속도와 낮은 메모리 사용량
- 🔧 프로토타입 및 소규모 프로젝트에 적합

### LangChain SQL Query Chain
LangChain이 제공하는 SQL 쿼리 자동 생성 체인입니다.

**구성 요소**:
```python
from langchain.chains import create_sql_query_chain

chain = create_sql_query_chain(
    llm=llm,           # 사용할 언어 모델
    db=db              # SQLDatabase 객체
)
```

**동작 원리**:
1. 데이터베이스 스키마 정보를 프롬프트에 포함
2. 사용자 질문과 함께 LLM에 전달
3. LLM이 적절한 SQL 쿼리 생성
4. 생성된 쿼리 반환 (자동 실행 안 함)

---

## 환경 설정

### 필수 라이브러리

```bash
# 핵심 라이브러리
pip install langchain langchain-openai langchain-google-genai
pip install langchain-community

# 데이터 처리
pip install pandas numpy

# 환경 변수 관리
pip install python-dotenv
```

### API 키 설정

`.env` 파일을 프로젝트 루트에 생성:

```bash
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

### 환경 변수 로드

```python
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

# API 키 확인
print("OpenAI API Key:", "설정됨" if os.getenv("OPENAI_API_KEY") else "미설정")
print("Google API Key:", "설정됨" if os.getenv("GOOGLE_API_KEY") else "미설정")
```

---

## 단계별 구현

### 1단계: ETF 데이터 로드 및 전처리

#### ETF 목록 데이터 로드
```python
import pandas as pd
import numpy as np

# ETF 목록 CSV 파일 로드
etf_data = pd.read_csv('data/etf_list.csv', encoding='cp949')

# 컬럼명 정리 (공백 제거 및 일관성 확보)
etf_data.columns = [
    '종목코드', '종목명', '상장일', '분류체계', '운용사', '수익률_최근1년',
    '기초지수', '추적오차', '순자산총액', '괴리율', '변동성',
    '복제방법', '총보수', '과세유형'
]

print(f"ETF 목록 데이터: {etf_data.shape}")
etf_data.head()
```

#### 데이터 타입 변환
```python
def convert_to_numeric_safely(value):
    """
    안전한 숫자 변환 함수
    변환 실패 시 None 반환
    """
    try:
        return pd.to_numeric(value)
    except:
        return None

# 숫자형 컬럼 변환
etf_data['종목코드'] = etf_data['종목코드'].apply(lambda x: str(x).strip())
etf_data['수익률_최근1년'] = etf_data['수익률_최근1년'].apply(convert_to_numeric_safely)
etf_data['추적오차'] = etf_data['추적오차'].apply(convert_to_numeric_safely)
etf_data['순자산총액'] = etf_data['순자산총액'].apply(convert_to_numeric_safely)
etf_data['괴리율'] = etf_data['괴리율'].apply(convert_to_numeric_safely)
etf_data['총보수'] = etf_data['총보수'].apply(convert_to_numeric_safely)

# 문자열 컬럼 정리
string_columns = ['종목명', '상장일', '분류체계', '운용사', '기초지수',
                  '변동성', '복제방법', '과세유형']
for col in string_columns:
    etf_data[col] = etf_data[col].astype(str).apply(lambda x: x.strip())

# 데이터 타입 확인
etf_data.info()
```

#### ETF 상세 정보 통합
```python
from glob import glob

# 개별 CSV 파일 로드
existing_csv_files = glob('data/etf_info/etf_info_*.csv')
print(f"발견된 CSV 파일: {len(existing_csv_files)}개")

# 모든 파일 통합
df_list = []
for file in existing_csv_files:
    df = pd.read_csv(file)
    df = df.set_index('항목').T

    # 필요한 컬럼만 선택
    required_columns = [
        '한글명', '영문명', '종목코드', '상장일', '펀드형태', '기초지수명',
        '추적배수', '자산운용사', '지정참가회사(AP)', '총보수(%)', '회계기간',
        '과세유형', '분배금 지급일', '홈페이지', '기초 시장', '기초 자산',
        '기본 정보', '투자유의사항'
    ]

    # 존재하는 컬럼만 선택
    available_columns = [col for col in required_columns if col in df.columns]
    df = df[available_columns]
    df_list.append(df)

# 통합 DataFrame 생성
etf_info = pd.concat(df_list)
etf_info = etf_info.dropna(axis=1, how='all')

print(f"ETF 상세 정보: {etf_info.shape}")
etf_info.head()
```

---

### 2단계: SQLite 데이터베이스 생성

#### ETFs 테이블 생성
```python
import sqlite3

# 데이터베이스 연결
conn = sqlite3.connect('etf_database.db')
cursor = conn.cursor()

# 기존 테이블 삭제
cursor.execute("DROP TABLE IF EXISTS ETFs")

# ETFs 테이블 생성
cursor.execute("""
CREATE TABLE ETFs (
    종목코드 TEXT PRIMARY KEY,
    종목명 TEXT,
    상장일 TEXT,
    분류체계 TEXT,
    운용사 TEXT,
    수익률_최근1년 REAL,
    기초지수 TEXT,
    추적오차 REAL,
    순자산총액 REAL,
    괴리율 REAL,
    변동성 TEXT,
    복제방법 TEXT,
    총보수 REAL,
    과세유형 TEXT
)
""")

# 데이터 삽입
for _, row in etf_data.iterrows():
    try:
        cursor.execute("""
        INSERT INTO ETFs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(row['종목코드']),
            str(row['종목명']),
            str(row['상장일']),
            str(row['분류체계']),
            str(row['운용사']),
            float(row['수익률_최근1년']) if pd.notna(row['수익률_최근1년']) else None,
            str(row['기초지수']),
            float(row['추적오차']) if pd.notna(row['추적오차']) else None,
            float(row['순자산총액']) if pd.notna(row['순자산총액']) else None,
            float(row['괴리율']) if pd.notna(row['괴리율']) else None,
            str(row['변동성']),
            str(row['복제방법']),
            float(row['총보수']) if pd.notna(row['총보수']) else None,
            str(row['과세유형'])
        ))
    except Exception as e:
        print(f"Error inserting row: {row['종목코드']}, {str(e)}")
        continue

# 변경사항 저장
conn.commit()

# 결과 확인
cursor.execute("SELECT COUNT(*) FROM ETFs")
print(f"ETF 개수: {cursor.fetchone()[0]}")
```

#### ETFsInfo 테이블 생성
```python
def create_etfs_info_table(conn, etf_info):
    """
    ETF 상세 정보 테이블 생성 함수

    Parameters:
        conn: SQLite 연결 객체
        etf_info: ETF 상세 정보 DataFrame

    Returns:
        conn: 업데이트된 연결 객체
    """
    cursor = conn.cursor()

    # 기존 테이블 삭제
    cursor.execute("DROP TABLE IF EXISTS ETFsInfo")

    # 테이블 생성
    cursor.execute("""
    CREATE TABLE ETFsInfo (
        한글명 TEXT,
        영문명 TEXT,
        종목코드 TEXT PRIMARY KEY,
        상장일 TEXT,
        펀드형태 TEXT,
        기초지수명 TEXT,
        추적배수 TEXT,
        자산운용사 TEXT,
        지정참가회사 TEXT,
        총보수 TEXT,
        회계기간 TEXT,
        과세유형 TEXT,
        분배금지급일 TEXT,
        홈페이지 TEXT,
        기초시장 TEXT,
        기초자산 TEXT,
        기본정보 TEXT,
        투자유의사항 TEXT
    )
    """)

    # 데이터 삽입
    for _, row in etf_info.iterrows():
        try:
            cursor.execute("""
            INSERT INTO ETFsInfo VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, tuple(str(row[col]) for col in etf_info.columns))
        except Exception as e:
            print(f"오류 - 종목코드: {row['종목코드']}, {str(e)}")
            continue

    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM ETFsInfo")
    print(f"ETFsInfo 테이블 생성 완료: {cursor.fetchone()[0]}개")

    return conn

# 테이블 생성 실행
conn = create_etfs_info_table(conn, etf_info)
conn.close()
```

---

### 3단계: LangChain과 데이터베이스 연동

#### 데이터베이스 스키마 확인
```python
from langchain_community.utilities import SQLDatabase

# SQLite 데이터베이스 연결
db = SQLDatabase.from_uri("sqlite:///etf_database.db")

# 사용 가능한 테이블 목록
tables = db.get_usable_table_names()
print(f"테이블 목록: {tables}")

# 테이블 스키마 정보
print("\n=== 데이터베이스 스키마 ===")
print(db.get_table_info())
```

**출력 예시**:
```
테이블 목록: ['ETFs', 'ETFsInfo']

=== 데이터베이스 스키마 ===
CREATE TABLE "ETFs" (
    "종목코드" TEXT PRIMARY KEY,
    "종목명" TEXT,
    "상장일" TEXT,
    "분류체계" TEXT,
    "운용사" TEXT,
    "수익률_최근1년" REAL,
    ...
)
```

#### 기본 쿼리 실행 테스트
```python
# 간단한 SELECT 쿼리 실행
query = "SELECT * FROM ETFs LIMIT 5"
result = db.run(query)
print(result)
```

---

### 4단계: SQL Query Chain 구성

#### LLM 모델 설정
```python
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_sql_query_chain

# GPT 모델 설정
gpt_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Gemini 모델 설정
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0
)

# SQL Query Chain 생성
gpt_sql = create_sql_query_chain(llm=gpt_llm, db=db)
gemini_sql = create_sql_query_chain(llm=gemini_llm, db=db)

print("SQL Query Chain 생성 완료")
```

#### 쿼리 생성 테스트
```python
# 테스트 질문
test_question = "상위 5개 운용사별 ETF 개수는 몇 개인가요?"

# GPT로 SQL 생성
gpt_generated_sql = gpt_sql.invoke({'question': test_question})
print(f"GPT 생성 쿼리:\n{gpt_generated_sql}")

# Gemini로 SQL 생성
gemini_generated_sql = gemini_sql.invoke({'question': test_question})
print(f"\nGemini 생성 쿼리:\n{gemini_generated_sql}")
```

**출력 예시**:
```
GPT 생성 쿼리:
```sql
SELECT "운용사", COUNT(*) AS "ETF_개수"
FROM "ETFs"
GROUP BY "운용사"
ORDER BY "ETF_개수" DESC
LIMIT 5;
```

Gemini 생성 쿼리:
```sqlite
SELECT "운용사", COUNT(*) AS "ETF_개수"
FROM ETFs
GROUP BY "운용사"
ORDER BY "ETF_개수" DESC
LIMIT 5
```
```

---

### 5단계: SQL 쿼리 추출 및 정리

#### 정규식 기반 SQL 추출 함수
```python
import re
from typing import Optional

def extract_sql(text: str) -> Optional[str]:
    """
    LLM 응답에서 SQL 쿼리를 추출하고 정리

    처리 단계:
    1. 마크다운 코드 블록 제거 (```sql ... ```)
    2. SQLQuery: 패턴 제거
    3. 줄바꿈 및 공백 정리
    4. 세미콜론 제거

    Parameters:
        text (str): SQL이 포함된 텍스트

    Returns:
        Optional[str]: 정리된 SQL 쿼리 또는 None

    Examples:
        >>> extract_sql('```sql\\nSELECT * FROM table\\n```')
        'SELECT * FROM table'

        >>> extract_sql('SQLQuery: SELECT * FROM table;')
        'SELECT * FROM table'
    """
    if text is None:
        return None

    # 1. SQL 마크다운 코드 블록 제거
    markdown_pattern = r'```sql[ite]*\s*(.*?)\s*```'
    markdown_match = re.search(markdown_pattern, text, re.DOTALL)
    if markdown_match:
        text = markdown_match.group(1)

    # 2. SQLQuery: 패턴 처리
    sql_pattern = r'SQLQuery:\s*(.*?)(?=SQLQuery:|$)'
    sql_match = re.search(sql_pattern, text, re.DOTALL)
    if sql_match:
        text = sql_match.group(1)

    # 3. 쿼리 정리
    if text:
        # 줄바꿈을 공백으로
        cleaned = text.replace('\n', ' ')
        # 연속 공백을 하나로
        cleaned = re.sub(r'\s+', ' ', cleaned)
        # 양쪽 공백 제거
        cleaned = cleaned.strip()
        # 마지막 세미콜론 제거
        cleaned = re.sub(r';$', '', cleaned)

        return cleaned

    return None

# 테스트
gpt_cleaned_query = extract_sql(gpt_generated_sql)
gemini_cleaned_query = extract_sql(gemini_generated_sql)

print(f"GPT 정리된 쿼리:\n{gpt_cleaned_query}")
print(f"\nGemini 정리된 쿼리:\n{gemini_cleaned_query}")
```

#### 쿼리 실행 및 결과 비교
```python
# GPT 쿼리 실행
gpt_result = db.run(gpt_cleaned_query)
print(f"GPT 결과: {gpt_result}")

# Gemini 쿼리 실행
gemini_result = db.run(gemini_cleaned_query)
print(f"Gemini 결과: {gemini_result}")
```

**출력 예시**:
```
GPT 결과: [('삼성자산운용', 202), ('미래에셋자산운용', 200), ('케이비자산운용', 118), ('한국투자신탁운용', 88), ('한화자산운용', 65)]

Gemini 결과: [('삼성자산운용', 202), ('미래에셋자산운용', 200), ('케이비자산운용', 118), ('한국투자신탁운용', 88), ('한화자산운용', 65)]
```

---

### 6단계: 다중 질문 처리 함수

#### 비교 함수 구현
```python
import time

def compare_answer_question(question: str):
    """
    GPT와 Gemini 모델의 SQL 생성 및 실행 결과 비교

    Parameters:
        question (str): 자연어 질문
    """
    try:
        # 1. SQL 생성
        gpt_generated_sql = gpt_sql.invoke({'question': question})
        gemini_generated_sql = gemini_sql.invoke({'question': question})

        # 2. SQL 정리
        gpt_cleaned_query = extract_sql(gpt_generated_sql)
        gemini_cleaned_query = extract_sql(gemini_generated_sql)

        # 3. 쿼리 실행
        gpt_result = db.run(gpt_cleaned_query)
        gemini_result = db.run(gemini_cleaned_query)

        # 4. 결과 출력
        print(f"Question: {question}")
        print(f"GPT SQL: {gpt_cleaned_query}")
        print(f"GPT Result: {gpt_result}")
        print(f"Gemini SQL: {gemini_cleaned_query}")
        print(f"Gemini Result: {gemini_result}")
        print("-" * 100)

    except Exception as e:
        print(f"Error processing question: {question}")
        print(f"Error message: {str(e)}")
        print("-" * 100)

# 테스트 질문들
sample_questions = [
    "평균 총보수가 가장 높은 운용사는 어디인가요?",
    "순자산총액이 가장 큰 ETF는 무엇인가요?",
    "수익률이 10% 이상인 ETF는 몇 개인가요?"
]

# 질문 처리
for question in sample_questions:
    compare_answer_question(question)
    time.sleep(2)  # API 호출 제한 고려
```

---

### 7단계: 고급 쿼리 - 조인 처리

#### 두 테이블 조인 쿼리
```python
# 복잡한 질문 (두 테이블 조인 필요)
complex_question = "추적배수가 일반 유형이고, 총보수가 0.1보다 작은 ETF 상품은 무엇인가요?"

# GPT 쿼리 생성
gpt_generated_sql = gpt_sql.invoke({'question': complex_question})
gpt_cleaned_query = extract_sql(gpt_generated_sql)

print(f"생성된 SQL:\n{gpt_cleaned_query}")

# 쿼리 실행
try:
    result = db.run(gpt_cleaned_query)
    print(f"\n실행 결과:\n{result}")
except Exception as e:
    print(f"Error: {str(e)}")
```

**GPT 생성 예시 (조인 쿼리)**:
```sql
SELECT "종목코드", "종목명", "총보수"
FROM "ETFs"
WHERE "총보수" < 0.1 AND "종목코드" IN (
    SELECT "종목코드"
    FROM "ETFsInfo"
    WHERE "추적배수" = '일반 (1)'
)
LIMIT 5
```

---

## 실전 활용 예제

### 예제 1: ETF 추천 시스템 기본

```python
def recommend_etf(user_query: str):
    """
    사용자 질문에 맞는 ETF 추천

    Parameters:
        user_query (str): 사용자 자연어 질문

    Returns:
        list: 추천 ETF 목록
    """
    # SQL 생성
    generated_sql = gpt_sql.invoke({'question': user_query})
    cleaned_sql = extract_sql(generated_sql)

    # 쿼리 실행
    try:
        result = db.run(cleaned_sql)
        return result
    except Exception as e:
        return f"오류 발생: {str(e)}"

# 사용 예시
queries = [
    "수익률이 가장 높은 ETF 5개를 알려주세요",
    "총보수가 0.1% 이하인 저비용 ETF를 찾아주세요",
    "삼성자산운용이 운용하는 ETF 중 순자산총액 상위 3개는?"
]

for query in queries:
    print(f"질문: {query}")
    result = recommend_etf(query)
    print(f"결과: {result}\n")
```

### 예제 2: 통계 분석 자동화

```python
def analyze_etf_statistics(analysis_type: str):
    """
    ETF 통계 분석 자동 실행

    Parameters:
        analysis_type (str): 분석 유형
            - 'distribution': 운용사별 분포
            - 'performance': 수익률 분석
            - 'cost': 비용 분석

    Returns:
        dict: 분석 결과
    """
    questions = {
        'distribution': "운용사별 ETF 개수와 비중을 계산해주세요",
        'performance': "분류체계별 평균 수익률을 보여주세요",
        'cost': "총보수 구간별 ETF 개수를 집계해주세요"
    }

    if analysis_type not in questions:
        return "지원하지 않는 분석 유형입니다"

    query = questions[analysis_type]
    generated_sql = gpt_sql.invoke({'question': query})
    cleaned_sql = extract_sql(generated_sql)

    try:
        result = db.run(cleaned_sql)
        return {
            'analysis_type': analysis_type,
            'query': query,
            'sql': cleaned_sql,
            'result': result
        }
    except Exception as e:
        return {'error': str(e)}

# 실행
for analysis in ['distribution', 'performance', 'cost']:
    result = analyze_etf_statistics(analysis)
    print(f"\n{analysis} 분석 결과:")
    print(result)
```

### 예제 3: 비교 분석 시스템

```python
def compare_etfs(etf_codes: list):
    """
    여러 ETF 비교 분석

    Parameters:
        etf_codes (list): ETF 종목코드 리스트

    Returns:
        pd.DataFrame: 비교 결과
    """
    codes_str = "', '".join(etf_codes)
    query = f"""
    SELECT
        종목코드, 종목명, 운용사,
        수익률_최근1년, 총보수, 순자산총액
    FROM ETFs
    WHERE 종목코드 IN ('{codes_str}')
    """

    result = db.run(query)

    # DataFrame으로 변환
    df = pd.DataFrame(
        result,
        columns=['종목코드', '종목명', '운용사', '수익률', '총보수', '순자산총액']
    )

    return df

# 사용 예시
etf_list = ['069500', '069660', '091160']
comparison = compare_etfs(etf_list)
print(comparison)
```

---

## 연습 문제

### 기본 문제

**문제 1**: 기본 쿼리 생성
- 과제: "변동성이 '매우낮음'인 ETF 개수를 세어주세요" 질문에 대한 SQL 생성
- 힌트: COUNT() 함수와 WHERE 절 사용

**문제 2**: 정렬 쿼리
- 과제: "순자산총액 기준 상위 10개 ETF 목록" 생성
- 힌트: ORDER BY와 LIMIT 절 활용

**문제 3**: 집계 함수
- 과제: "분류체계별 평균 총보수 계산" 쿼리 생성
- 힌트: GROUP BY와 AVG() 함수 사용

### 중급 문제

**문제 4**: 조건 필터링
```python
# 다음 조건을 만족하는 ETF 찾기
# - 수익률이 5% 이상
# - 총보수가 0.2% 이하
# - 순자산총액이 1000억 이상

question = "수익률 5% 이상, 총보수 0.2% 이하, 순자산 1000억 이상 ETF는?"
# SQL 생성 및 실행 코드 작성
```

**문제 5**: 서브쿼리
```python
# ETFsInfo 테이블에서 펀드형태가 '수익증권형'인 ETF 중
# ETFs 테이블에서 수익률 상위 5개 찾기

question = "수익증권형 펀드 중 수익률 상위 5개 ETF는?"
# 조인 또는 서브쿼리 사용
```

**문제 6**: 여러 테이블 조인
```python
# 두 테이블을 조인하여 상세 정보 통합
# 조건: 총보수가 평균보다 낮은 ETF의 상세 정보

question = "평균보다 총보수가 낮은 ETF의 기본정보를 보여주세요"
# 두 테이블의 정보를 결합
```

### 고급 문제

**문제 7**: 복잡한 집계
```python
# 운용사별로 다음 정보 산출:
# - ETF 개수
# - 평균 수익률
# - 평균 총보수
# - 전체 순자산총액 합계

def get_company_analysis():
    """운용사별 종합 분석"""
    # 구현
    pass
```

**문제 8**: 순위 계산
```python
# 분류체계별 수익률 순위 매기기
# 각 분류체계 내에서 TOP 3 ETF 찾기

def rank_by_category():
    """분류체계별 순위 계산"""
    # 구현
    pass
```

**문제 9**: 동적 쿼리 생성기
```python
def build_dynamic_query(filters: dict):
    """
    사용자가 제공한 필터로 동적 쿼리 생성

    Parameters:
        filters (dict): {
            'min_return': 3.0,
            'max_cost': 0.3,
            'companies': ['삼성자산운용', '미래에셋자산운용'],
            'volatility': ['매우낮음', '낮음']
        }

    Returns:
        str: 생성된 자연어 질문
    """
    # 필터를 자연어 질문으로 변환
    # Text2SQL로 처리
    pass
```

---

## 문제 해결 가이드

### 일반적인 오류

#### 1. 데이터베이스 연결 실패
```python
# 문제: sqlite3.OperationalError: unable to open database file

# 해결방법:
import os

# 현재 디렉토리 확인
print("현재 디렉토리:", os.getcwd())

# 절대 경로 사용
db_path = os.path.join(os.getcwd(), 'etf_database.db')
db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
```

#### 2. 한글 인코딩 문제
```python
# 문제: UnicodeDecodeError 발생

# 해결방법:
etf_data = pd.read_csv('data/etf_list.csv', encoding='cp949')  # Windows
# 또는
etf_data = pd.read_csv('data/etf_list.csv', encoding='euc-kr')  # Linux/Mac
```

#### 3. SQL 쿼리 파싱 실패
```python
# 문제: extract_sql()이 None 반환

# 디버깅:
def extract_sql_debug(text: str):
    print(f"원본 텍스트: {text}")

    # 마크다운 패턴 확인
    markdown_match = re.search(r'```sql[ite]*\s*(.*?)\s*```', text, re.DOTALL)
    print(f"마크다운 매치: {markdown_match}")

    # SQLQuery 패턴 확인
    sql_match = re.search(r'SQLQuery:\s*(.*?)(?=SQLQuery:|$)', text, re.DOTALL)
    print(f"SQL 패턴 매치: {sql_match}")

    return extract_sql(text)
```

#### 4. 데이터 타입 불일치
```python
# 문제: TypeError: 'str' object cannot be interpreted as an integer

# 해결방법:
# ETFsInfo 테이블의 총보수가 TEXT 타입일 때
cursor.execute("""
SELECT * FROM ETFsInfo
WHERE CAST(총보수 AS REAL) < 0.1
""")
```

### 성능 최적화

#### 인덱스 생성
```python
# 자주 조회되는 컬럼에 인덱스 추가
conn = sqlite3.connect('etf_database.db')
cursor = conn.cursor()

cursor.execute("CREATE INDEX idx_company ON ETFs(운용사)")
cursor.execute("CREATE INDEX idx_return ON ETFs(수익률_최근1년)")
cursor.execute("CREATE INDEX idx_category ON ETFs(분류체계)")

conn.commit()
conn.close()
```

#### 쿼리 최적화
```python
# 비효율적:
query = "SELECT * FROM ETFs WHERE 운용사 = '삼성자산운용'"

# 효율적 (필요한 컬럼만 선택):
query = "SELECT 종목코드, 종목명, 수익률_최근1년 FROM ETFs WHERE 운용사 = '삼성자산운용'"
```

#### 배치 처리
```python
def batch_questions(questions: list, batch_size: int = 5):
    """
    여러 질문을 배치로 처리

    Parameters:
        questions (list): 질문 리스트
        batch_size (int): 배치 크기
    """
    results = []

    for i in range(0, len(questions), batch_size):
        batch = questions[i:i+batch_size]

        for question in batch:
            result = recommend_etf(question)
            results.append({
                'question': question,
                'result': result
            })

        # API 제한 고려
        if i + batch_size < len(questions):
            time.sleep(5)

    return results
```

---

## 추가 학습 자료

### 관련 문서
- [LangChain SQL Database](https://python.langchain.com/docs/use_cases/sql/)
- [SQLite 공식 문서](https://www.sqlite.org/docs.html)
- [Pandas to_sql 메서드](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_sql.html)
- [정규표현식 가이드](https://docs.python.org/3/library/re.html)

### 다음 단계
1. **RAG와 Text2SQL 통합**: 벡터 검색과 SQL 쿼리 결합
2. **실시간 데이터 업데이트**: 스케줄링으로 ETF 데이터 자동 갱신
3. **대화형 인터페이스**: Gradio/Streamlit으로 웹 UI 구축
4. **쿼리 검증 시스템**: 생성된 SQL의 안전성 및 정확성 검증
5. **멀티 모달 분석**: 차트 생성 및 시각화 통합

### 심화 주제
- **SQL Injection 방지**: 파라미터화된 쿼리 사용
- **복잡한 조인 쿼리**: INNER/LEFT/RIGHT JOIN 활용
- **Window Functions**: RANK(), ROW_NUMBER() 등 고급 집계
- **트랜잭션 관리**: BEGIN, COMMIT, ROLLBACK 처리
- **데이터베이스 최적화**: 정규화, 인덱싱 전략

---

## 요약

이 가이드에서 학습한 핵심 내용:

✅ **SQLite 데이터베이스 설계 및 구축**
- Pandas DataFrame을 SQLite 테이블로 변환
- 적절한 데이터 타입 선택 및 PRIMARY KEY 설정

✅ **Text2SQL 시스템 구현**
- LangChain SQL Query Chain을 활용한 자동 SQL 생성
- GPT와 Gemini 모델 통합 및 비교

✅ **SQL 쿼리 추출 및 정리**
- 정규식을 활용한 마크다운 코드 블록 파싱
- 쿼리 정제 및 실행 가능한 형태로 변환

✅ **실전 활용 사례**
- ETF 추천 시스템 구축
- 통계 분석 자동화
- 다중 모델 비교 분석

이제 자연어 인터페이스를 통해 데이터베이스에 접근하는 실용적인 AI 시스템을 구축할 수 있습니다!
