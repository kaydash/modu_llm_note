# W3_001 추천 시스템 - ETF 데이터 수집 및 전처리

## 학습 목표
- Crawl4AI를 활용한 고성능 웹 크롤링 기술 습득
- ETF 데이터 수집 및 전처리 방법 학습
- 비동기 크롤링과 추출 전략 구현
- 구조화된 데이터 추출 및 CSV 저장

## 핵심 개념

### 1. Crawl4AI란?
- **정의**: LLM, AI 에이전트, 데이터 파이프라인을 위한 고성능 웹 크롤링 솔루션
- **특징**: 오픈소스 기반, 실시간 성능, 유연한 구현
- **장점**: 비동기 처리, 다양한 추출 전략, 캐시 지원

### 2. 주요 구성 요소

#### BrowserConfig
- 브라우저 타입, 헤드리스 모드, 뷰포트 크기 등 기본 설정 관리
- 프록시 설정과 디버깅 옵션을 통한 고급 크롤링 환경 구성

#### CrawlerRunConfig
- 단어 수 제한, 추출 전략, 캐시 설정 등 크롤링 옵션 관리
- js_code와 wait_for 옵션으로 동적 콘텐츠 처리
- screenshot과 rate_limiting 기능으로 크롤링 과정 제어

#### CrawlResult
- 크롤링된 URL, HTML, 성공 여부 등 기본 정보 포함
- cleaned_html와 markdown 필드로 정제된 데이터 접근
- extracted_content를 통해 구조화된 형태의 추출 데이터 확인

### 3. ETF 데이터
- **출처**: 한국거래소(KRX) ETF 상세 정보 페이지
- **수집 항목**: 한글명, 영문명, 종목코드, 상장일, 펀드형태, 기초지수명 등
- **활용**: 추천 시스템, 데이터 분석, 투자 전략 개발

## 환경 설정

### 1. 필수 라이브러리 설치
```bash
# 기본 설치 - 코어 라이브러리만 설치
pip install crawl4ai

# 초기 설정 - Playwright 브라우저 설치 및 환경 점검
crawl4ai-setup

# 진단 도구 - 시스템 호환성 확인
crawl4ai-doctor

# 기타 라이브러리
pip install pandas numpy nest-asyncio python-dotenv
```

### 2. 환경 변수 설정
```python
from dotenv import load_dotenv
import os
from glob import glob
from pprint import pprint
import json
import pandas as pd
import numpy as np

# 환경 변수 로드
load_dotenv()
```

### 3. Jupyter 환경 설정
```python
# nest_asyncio는 Jupyter에서 이미 실행 중인 이벤트 루프 위에
# 중첩된 이벤트 루프를 실행할 수 있게 해주는 패키지

import nest_asyncio
nest_asyncio.apply()

# Windows 환경용 추가 설정
import asyncio
import sys

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# 새 이벤트 루프 생성 및 설정
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
nest_asyncio.apply(loop)
```

## 1단계: ETF 데이터 준비

### CSV 데이터 로드
```python
# ETF 목록 CSV 파일 로드
etf_list = pd.read_csv('data/etf_list.csv', encoding='cp949')

# 데이터 확인
print(f"ETF 목록: {etf_list.shape[0]}개")
etf_list.head()
```

**데이터 출처:**
- 한국거래소(KRX) ETF 정보 페이지
- http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC020103010901

### ETF 상세 페이지 URL 생성
```python
# 종목 코드로 상세 페이지 URL 생성
strIsurCd = '45979'  # ETF 고유 코드
etf_url = f"https://kind.krx.co.kr/disclosure/etfisudetail.do?method=searchEtfIsuSummary&strIsurCd={strIsurCd}"

print(f"ETF 상세 페이지: {etf_url}")
```

## 2단계: 기본 크롤링 구현

### 기본 크롤링 함수
```python
import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

async def main():
    """
    기본 크롤링 실행 함수
    """
    browser_config = BrowserConfig()  # 브라우저 설정
    run_config = CrawlerRunConfig()   # 크롤러 실행 설정

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url=etf_url,
            config=run_config
        )

    return result

# 크롤링 실행 및 결과 출력
result = loop.run_until_complete(main())
print(result.markdown)  # 크롤링 결과 (정제된 markdown 출력)
```

### CrawlResult 응답 객체 확인
```python
# 크롤링 성공 여부 확인
print("성공 여부:", result.success)

# HTML 접근
print("원본 HTML (처음 100자):")
print(result.html[:100])

# 정제된 HTML 출력
print("\n정제된 HTML (처음 100자):")
print(result.cleaned_html[:100])

# 마크다운 변환 결과
print("\n마크다운 변환 결과:")
print(result.markdown)

# 구조화된 데이터 (Optional)
if result.extracted_content:
    data = json.loads(result.extracted_content)
    print("\n추출된 데이터:")
    print(data)

# 전체 결과 객체 확인
result_dict = result.model_dump()
print("\n결과 객체 키:", result_dict.keys())
```

## 3단계: BrowserConfig 설정

### 기본 브라우저 설정
```python
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

# 기본 설정
base_config = BrowserConfig(
    browser_type="chromium",       # 브라우저 엔진 선택 (chromium, firefox, webkit)
    viewport_width=1080,           # 뷰포트 너비
    viewport_height=600,           # 뷰포트 높이
    text_mode=True,                # 텍스트 모드 (이미지 비활성화로 속도 향상)
    use_persistent_context=True,   # 세션을 유지 (쿠키, 로컬 스토리지 등)
    headless=True,                 # 헤드리스 모드 실행 (브라우저 UI 없음)
)

print("기본 브라우저 설정 완료")
```

### 디버깅용 설정
```python
# 디버깅용 설정 (브라우저 UI 표시)
debug_config = base_config.clone(
    headless=False,  # 헤드리스 모드 비활성화
    verbose=True     # 디버깅용 로그 출력
)

print("디버깅 설정 완료")
```

### 프록시 설정
```python
# 프록시 설정 (필요 시)
proxy_config_dict = {
    "server": "http://proxy.example.com:8080",
    "username": "user",
    "password": "pass"
}

proxy_config = base_config.clone(
    proxy_config=proxy_config_dict,
)

print("프록시 설정 완료")
```

### 범용 크롤링 함수
```python
async def extract_by_url(
    url: str,
    browser_config: BrowserConfig,
    run_config: CrawlerRunConfig
):
    """
    URL로부터 데이터를 크롤링하는 범용 함수
    """
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url=url,
            config=run_config
        )

    return result

# 사용 예시
result = loop.run_until_complete(
    extract_by_url(etf_url, base_config, CrawlerRunConfig())
)
print("크롤링 성공:", result.success)
print(result.markdown)
```

## 4단계: CrawlerRunConfig 설정

### 기본 추출 설정
```python
from crawl4ai import CacheMode

run_config = CrawlerRunConfig(
    cache_mode=CacheMode.ENABLED,   # 캐시 활성화 (동일 URL 재요청 시 캐시 사용)
    word_count_threshold=200,       # 컨텐츠 블록의 최소 단어 수 기준
                                   # 짧은 문단이 많은 사이트는 값을 낮춰야 함
)

# 크롤링 실행
result = loop.run_until_complete(
    extract_by_url(etf_url, base_config, run_config)
)
print("크롤링 성공:", result.success)
print(result.markdown)
```

### CSS 선택자 기반 추출
```python
# 특정 영역만 추출하는 설정
run_config = CrawlerRunConfig(
    css_selector="main.content",         # 크롤링 대상 컨텐츠 선택자
    word_count_threshold=10,             # 최소 단어 수 기준
    excluded_tags=["nav", "footer"],     # 제외할 태그
    exclude_external_links=True,         # 외부 링크 제외
    exclude_social_media_links=True,     # 소셜 미디어 링크 제외
    exclude_domains=["ads.com", "spammytrackers.net"],  # 제외할 도메인
    exclude_external_images=True,        # 외부 이미지 제외
    cache_mode=CacheMode.BYPASS          # 캐시 비활성화 (항상 최신 데이터)
)

# 크롤링 실행
result = loop.run_until_complete(
    extract_by_url(etf_url, base_config, run_config)
)
print(result.markdown)
```

## 5단계: 추출 전략 (Extraction Strategy)

### CSS 기반 추출 전략
```python
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy

# 뉴스 사이트 예시 스키마
news_url = "https://news.ycombinator.com/newest"

news_schema = {
    "name": "HN Stories",
    "baseSelector": "tr.athing",  # 반복되는 항목의 기본 선택자
    "fields": [
        {
            "name": "title",
            "selector": "td.title span.titleline a:first-child",
            "type": "text"
        },
        {
            "name": "url",
            "selector": "td.title span.titleline a:first-child",
            "type": "attribute",
            "attribute": "href"
        },
        {
            "name": "rank",
            "selector": "td.title span.rank",
            "type": "text",
            "transform": lambda x: int(x.strip("."))  # 변환 함수 적용
        }
    ]
}

# 추출 전략 설정
extraction_config = CrawlerRunConfig(
    extraction_strategy=JsonCssExtractionStrategy(news_schema),
    cache_mode=CacheMode.BYPASS  # 항상 최신 데이터 가져오기
)

# 크롤링 실행
result = loop.run_until_complete(
    extract_by_url(news_url, base_config, extraction_config)
)

# 구조화된 데이터 확인
if result.extracted_content:
    data = json.loads(result.extracted_content)
    print("추출된 데이터:")
    print(json.dumps(data, indent=2, ensure_ascii=False))

    # DataFrame으로 변환
    df = pd.DataFrame(data)
    print(f"\n추출된 항목 수: {len(df)}")
    print(df.head())
```

**참고 문서:**
- Crawl4AI Content Selection: https://docs.crawl4ai.com/core/content-selection/

## 6단계: ETF 상세정보 수집 실습

### ETF 데이터 추출 함수
```python
async def extract_etf_details(etf_code: str) -> dict:
    """
    ETF 상세 정보를 추출하는 함수

    Args:
        etf_code: ETF 고유 코드 (5자리 숫자)

    Returns:
        ETF 상세 정보 딕셔너리
    """
    # URL 생성
    url = f"https://kind.krx.co.kr/disclosure/etfisudetail.do?method=searchEtfIsuSummary&strIsurCd={etf_code}"

    # 브라우저 설정
    browser_config = BrowserConfig(
        browser_type="chromium",
        headless=True,
        text_mode=True
    )

    # 크롤러 실행 설정
    run_config = CrawlerRunConfig(
        word_count_threshold=10,
        cache_mode=CacheMode.BYPASS
    )

    # 크롤링 실행
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=run_config)

    if not result.success:
        print(f"크롤링 실패: {etf_code}")
        return None

    # HTML에서 테이블 추출
    try:
        # pandas.read_html()로 테이블 데이터 추출
        tables = pd.read_html(result.html)

        # ETF 정보가 포함된 테이블 찾기
        etf_info = {}

        # 테이블 순회하며 필요한 정보 추출
        for table in tables:
            if '한글명' in str(table.columns) or '한글명' in str(table.values):
                # 테이블에서 필드명과 값 추출
                for idx, row in table.iterrows():
                    if len(row) >= 2:
                        field_name = str(row[0]).strip()
                        field_value = str(row[1]).strip()
                        etf_info[field_name] = field_value

        # 종목코드 추가
        etf_info['종목코드'] = etf_code

        return etf_info

    except Exception as e:
        print(f"데이터 추출 실패 ({etf_code}): {e}")
        return None

# 단일 ETF 테스트
test_code = '45979'
test_result = loop.run_until_complete(extract_etf_details(test_code))
print("추출된 정보:")
print(json.dumps(test_result, indent=2, ensure_ascii=False))
```

### 다수 ETF 데이터 수집
```python
async def collect_multiple_etfs(etf_codes: list) -> pd.DataFrame:
    """
    여러 ETF의 상세 정보를 수집하는 함수

    Args:
        etf_codes: ETF 고유 코드 리스트

    Returns:
        ETF 정보 DataFrame
    """
    etf_data_list = []

    for idx, code in enumerate(etf_codes, 1):
        print(f"수집 중 ({idx}/{len(etf_codes)}): {code}")

        # ETF 정보 추출
        etf_info = await extract_etf_details(code)

        if etf_info:
            etf_data_list.append(etf_info)

        # API 부하 방지를 위한 대기
        await asyncio.sleep(1)

    # DataFrame으로 변환
    df = pd.DataFrame(etf_data_list)

    return df

# 5개 ETF 샘플 수집
sample_codes = ['45979', '46400', '46500', '46600', '46700']

# 데이터 수집 실행
etf_df = loop.run_until_complete(collect_multiple_etfs(sample_codes))

# 결과 확인
print(f"\n수집된 ETF: {len(etf_df)}개")
print(etf_df.head())

# CSV 저장
etf_df.to_csv('data/etf_details.csv', index=False, encoding='utf-8-sig')
print("\nCSV 파일 저장 완료: data/etf_details.csv")
```

### 수집 대상 필드
```python
# ETF 상세 정보 필드 목록
target_fields = [
    '한글명',
    '영문명',
    '종목코드',
    '상장일',
    '펀드형태',
    '기초지수명',
    '추적배수',
    '자산운용사',
    '지정참가회사(AP)',
    '총보수(%)',
    '회계기간',
    '과세유형',
    '분배금 지급일',
    '홈페이지',
    '기초 시장',
    '기초 자산',
    '기본 정보',
    '투자유의사항'
]

print("수집 대상 필드:", target_fields)
```

## 7단계: 데이터 전처리 및 분석

### 데이터 정제
```python
# 결측값 확인
print("결측값 확인:")
print(etf_df.isnull().sum())

# 중복 데이터 확인
print(f"\n중복 데이터: {etf_df.duplicated().sum()}개")

# 데이터 타입 확인
print("\n데이터 타입:")
print(etf_df.dtypes)

# 필요한 전처리 수행
# 예: 날짜 형식 변환, 숫자 형식 변환 등
if '상장일' in etf_df.columns:
    etf_df['상장일'] = pd.to_datetime(etf_df['상장일'], errors='coerce')

if '총보수(%)' in etf_df.columns:
    etf_df['총보수(%)'] = pd.to_numeric(
        etf_df['총보수(%)'].str.replace('%', ''),
        errors='coerce'
    )

print("\n전처리 완료")
print(etf_df.info())
```

### 기본 통계 분석
```python
# 수치형 데이터 통계
print("수치형 데이터 통계:")
print(etf_df.describe())

# 범주형 데이터 분포
if '펀드형태' in etf_df.columns:
    print("\n펀드형태 분포:")
    print(etf_df['펀드형태'].value_counts())

if '자산운용사' in etf_df.columns:
    print("\n자산운용사 분포:")
    print(etf_df['자산운용사'].value_counts())
```

## 실습 과제

### 기본 실습
1. **추가 ETF 데이터 수집**
   - 10개 이상의 ETF 데이터 수집
   - 모든 필드 추출 및 검증
   - CSV 파일로 저장

2. **다양한 추출 전략 테스트**
   - CSS 선택자를 사용한 데이터 추출
   - XPath를 사용한 데이터 추출
   - 정규표현식을 사용한 데이터 정제

### 응용 실습
3. **에러 처리 및 재시도 로직**
   - 네트워크 오류 시 재시도 구현
   - 타임아웃 설정 및 처리
   - 로깅 시스템 구축

4. **병렬 크롤링 구현**
   - asyncio.gather()를 사용한 병렬 처리
   - 동시 요청 수 제한 (Semaphore)
   - 성능 비교 (순차 vs 병렬)

### 심화 실습
5. **동적 콘텐츠 크롤링**
   - JavaScript 실행이 필요한 페이지 처리
   - 무한 스크롤 페이지 크롤링
   - 로그인이 필요한 페이지 크롤링

6. **데이터 파이프라인 구축**
   - 주기적 데이터 수집 스케줄러
   - 데이터 변경 감지 및 알림
   - 데이터베이스 자동 저장

## 문제 해결 가이드

### 일반적인 오류들
1. **이벤트 루프 오류**
   ```python
   # Jupyter에서 asyncio 오류 발생 시
   import nest_asyncio
   nest_asyncio.apply()
   ```

2. **타임아웃 오류**
   ```python
   # 타임아웃 설정 증가
   run_config = CrawlerRunConfig(
       page_timeout=60000,  # 60초
       wait_until="networkidle"
   )
   ```

3. **인코딩 오류**
   ```python
   # CSV 저장 시 인코딩 지정
   df.to_csv('output.csv', encoding='utf-8-sig', index=False)
   ```

4. **봇 감지 우회**
   ```python
   # User-Agent 설정
   browser_config = BrowserConfig(
       user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
   )
   ```

## 참고 자료
- [Crawl4AI GitHub](https://github.com/unclecode/crawl4ai)
- [Crawl4AI 공식 문서](https://docs.crawl4ai.com/)
- [한국거래소 ETF 정보](http://data.krx.co.kr/)
- [Playwright Python 문서](https://playwright.dev/python/)
- [asyncio 공식 문서](https://docs.python.org/3/library/asyncio.html)

이 학습 가이드를 통해 Crawl4AI를 활용한 효율적인 웹 크롤링과 ETF 데이터 수집 방법을 익힐 수 있습니다.