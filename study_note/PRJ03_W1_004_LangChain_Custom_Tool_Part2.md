# PRJ03_W1_004_LangChain_Custom_Tool_Part2

## 📚 학습 목표

이 학습 가이드를 완료하면 다음을 수행할 수 있습니다:

1. **외부 API 연동**: Yahoo Finance 등 실제 API와 LangChain 도구 통합
2. **도구 에러 처리**: handle_tool_error를 사용한 체계적인 에러 관리
3. **BaseTool 상속**: 복잡한 커스텀 도구 구현을 위한 고급 패턴
4. **QA 체인 구성**: 도구와 체인을 결합한 질의응답 시스템 구축
5. **실무 통합**: Naver API 등 실제 서비스와의 통합 패턴

## 🔑 핵심 개념

### 1. 외부 API 연동

LangChain 도구를 통해 외부 API를 LLM에 연결:

```python
# 외부 API 호출 함수
def call_external_api(params):
    # API 호출 로직
    return api_response

# 도구로 변환
@tool
def api_tool(query: str) -> dict:
    """외부 API를 호출하는 도구"""
    return call_external_api(query)
```

**핵심 포인트:**
- API 응답을 LLM이 이해할 수 있는 형식으로 변환
- 에러 처리 및 타임아웃 관리
- API 키 보안 (환경 변수 사용)

### 2. 도구 에러 처리

3가지 에러 처리 방법:

```python
# 1. 기본 에러 메시지 반환
StructuredTool.from_function(
    func=my_func,
    handle_tool_error=True  # ToolException 메시지 그대로 반환
)

# 2. 커스텀 에러 메시지
StructuredTool.from_function(
    func=my_func,
    handle_tool_error="사용자 정의 에러 메시지"
)

# 3. 에러 처리 함수
def custom_error_handler(error):
    return f"처리된 에러: {error}"

StructuredTool.from_function(
    func=my_func,
    handle_tool_error=custom_error_handler
)
```

### 3. BaseTool 상속

복잡한 도구 구현을 위한 클래스 기반 접근:

```python
from langchain_core.tools import BaseTool

class MyCustomTool(BaseTool):
    name: str = "my_tool"
    description: str = "도구 설명"
    args_schema: type[BaseModel] = MyInputSchema

    def _run(self, arg1: str, arg2: int) -> str:
        """동기 실행 로직"""
        return process(arg1, arg2)

    async def _arun(self, arg1: str, arg2: int) -> str:
        """비동기 실행 로직"""
        return await async_process(arg1, arg2)
```

**BaseTool vs StructuredTool:**
- BaseTool: 복잡한 상태 관리, 초기화 로직 필요 시
- StructuredTool: 간단한 함수 기반 도구

## 🛠 환경 설정

### 필수 라이브러리 설치

```bash
pip install langchain langchain-openai langchain-community
pip install yfinance  # Yahoo Finance API
pip install python-dotenv
pip install requests  # Naver API
```

### 환경 변수 설정

`.env` 파일 생성:

```bash
OPENAI_API_KEY=your_openai_api_key_here

# Yahoo Finance는 API 키 불필요

# Naver 개발자 API (선택)
NAVER_CLIENT_ID=your_naver_client_id
NAVER_CLIENT_SECRET=your_naver_client_secret
```

### 기본 임포트

```python
import os
from dotenv import load_dotenv
from typing import Optional, Dict
from datetime import datetime, timedelta
from pprint import pprint

import yfinance as yf
from langchain_core.tools import tool, StructuredTool, BaseTool, ToolException
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# 환경 변수 로드
load_dotenv()
```

## 💻 단계별 구현

### Step 1: 외부 API 연동 - Yahoo Finance

#### 1.1 Yahoo Finance API 기본 사용

```python
import yfinance as yf

# Ticker 객체 생성
stock = yf.Ticker("MSFT")

# 기업 정보
info = stock.info
print(f"회사명: {info.get('longName')}")
print(f"섹터: {info.get('sector')}")

# 주요 일정
calendar = stock.calendar
print("주요 일정:", calendar)

# 특정 기간 데이터 조회
hist = stock.history(start="2022-01-03", end="2022-01-05")
print(hist)
```

**출력 예시:**
```
회사명: Microsoft Corporation
섹터: Technology

            Open    High     Low   Close      Volume
Date
2022-01-03  335.0  337.5   332.2   334.75   25678900
2022-01-04  333.5  336.8   331.0   335.44   23456700
```

#### 1.2 날짜 검증 및 데이터 처리 함수

```python
from datetime import datetime, timedelta

def is_valid_date(date_str: str) -> bool:
    """날짜 형식 검증 (YYYY-MM-DD)"""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False

def get_stock_price(symbol: str, date: Optional[str] = None) -> Dict:
    """
    yfinance 사용하여 특정 날짜의 주식 가격 정보를 조회합니다.

    Args:
        symbol: 주식 티커 심볼 (예: "AAPL", "MSFT")
        date: 조회 날짜 (YYYY-MM-DD), None이면 최근 영업일

    Returns:
        주식 가격 정보 딕셔너리
    """
    # 날짜 검증
    if date and not is_valid_date(date):
        raise ToolException(f"잘못된 날짜 형식입니다: {date}")

    try:
        stock = yf.Ticker(symbol)

        # 날짜 지정 시
        if date:
            # 해당 날짜 +1일까지 조회 (해당 날짜 데이터 포함을 위해)
            start_date = datetime.strptime(date, "%Y-%m-%d")
            end_date = start_date + timedelta(days=1)

            hist = stock.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d")
            )

            if hist.empty:
                return {
                    "symbol": symbol,
                    "date": date,
                    "message": "해당 날짜에 거래 데이터가 없습니다. (주말 또는 휴장일)"
                }

            # 인덱스(날짜) 초기화 및 날짜 문자열로 변환
            hist = hist.reset_index()
            hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')

            # 딕셔너리로 변환
            result = hist.to_dict('records')[0]
            result['symbol'] = symbol

            return result

        # 날짜 미지정 시 최근 영업일
        else:
            hist = stock.history(period="5d")  # 최근 5일

            if hist.empty:
                raise ToolException(f"주식 데이터를 찾을 수 없습니다: {symbol}")

            hist = hist.reset_index()
            hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
            result = hist.to_dict('records')[-1]  # 가장 최근
            result['symbol'] = symbol

            return result

    except Exception as e:
        raise ToolException(f"주식 데이터 조회 중 오류 발생: {str(e)}")
```

#### 1.3 함수 테스트

```python
# 특정 날짜 조회
result1 = get_stock_price("AAPL", "2024-01-03")
print("특정 날짜:", result1)

# 최근 영업일 조회
result2 = get_stock_price("MSFT")
print("최근 영업일:", result2)

# 거래 없는 날짜 (주말)
result3 = get_stock_price("AAPL", "2024-01-06")  # 토요일
print("주말:", result3)
```

**출력 예시:**
```python
특정 날짜: {
    'Date': '2024-01-03',
    'Open': 184.22,
    'High': 185.88,
    'Low': 183.43,
    'Close': 184.25,
    'Volume': 58414200,
    'symbol': 'AAPL'
}

최근 영업일: {...}

주말: {
    'symbol': 'AAPL',
    'date': '2024-01-06',
    'message': '해당 날짜에 거래 데이터가 없습니다. (주말 또는 휴장일)'
}
```

#### 1.4 StructuredTool로 변환

```python
# 도구 생성
stock_tool = StructuredTool.from_function(
    func=get_stock_price,
    name="stock_price",
    description="yfinance를 사용하여 주식 가격 정보를 조회하는 도구입니다.",
)

# 도구 실행
result = stock_tool.invoke({"symbol": "AAPL", "date": "2024-01-03"})
pprint(result)
```

#### 1.5 LLM과 통합

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
llm_with_tools = llm.bind_tools([stock_tool])

# LLM이 도구 호출
response = llm_with_tools.invoke("2024년 1월 3일 Apple 주식 가격을 알려줘")
pprint(response.tool_calls)

# 도구 실행
if response.tool_calls:
    tool_call = response.tool_calls[0]
    result = stock_tool.invoke(tool_call)
    print("\n주식 가격 정보:")
    pprint(result)
```

**출력 예시:**
```python
[{'args': {'date': '2024-01-03', 'symbol': 'AAPL'},
  'id': 'call_xxx',
  'name': 'stock_price',
  'type': 'tool_call'}]

주식 가격 정보:
{'Close': 184.25,
 'Date': '2024-01-03',
 'High': 185.88,
 'Low': 183.43,
 'Open': 184.22,
 'Volume': 58414200,
 'symbol': 'AAPL'}
```

#### 1.6 Agent와 통합

```python
from langchain.agents import create_agent

# Agent 생성
stock_agent = create_agent(
    model=llm,
    tools=[stock_tool],
    system_prompt="당신은 주식 정보 조회를 도와주는 AI 어시스턴트입니다."
)

# Agent 실행
result = stock_agent.invoke({
    "messages": [{
        "role": "user",
        "content": "2024년 1월 3일 Microsoft와 Apple의 종가를 비교해줘"
    }]
})

# 최종 답변
final_message = result["messages"][-1]
print(final_message.content)
```

### Step 2: 도구 에러 처리

#### 2.1 기본 에러 처리

```python
# handle_tool_error=True: ToolException 메시지 그대로 반환
stock_tool_basic = StructuredTool.from_function(
    func=get_stock_price,
    name="stock_price_basic",
    handle_tool_error=True  # 기본 에러 메시지 반환
)

# 정상 실행
result1 = stock_tool_basic.invoke({"symbol": "AAPL"})
print("정상:", result1)

# 에러 발생 (잘못된 날짜 형식)
result2 = stock_tool_basic.invoke({"symbol": "AAPL", "date": "2024/01/03"})
print("에러:", result2)
```

**출력 예시:**
```
정상: {'Date': '2024-01-05', 'Open': 181.99, ...}
에러: 잘못된 날짜 형식입니다: 2024/01/03
```

#### 2.2 커스텀 에러 메시지

```python
# 모든 에러에 대해 동일한 메시지 반환
stock_tool_custom_msg = StructuredTool.from_function(
    func=get_stock_price,
    name="stock_price_custom",
    handle_tool_error="주식 정보 조회 중 오류가 발생했습니다. 티커 심볼과 날짜 형식(YYYY-MM-DD)을 확인해주세요."
)

# 에러 발생
result = stock_tool_custom_msg.invoke({"symbol": "INVALID", "date": "2024/01/03"})
print(result)
```

**출력 예시:**
```
주식 정보 조회 중 오류가 발생했습니다. 티커 심볼과 날짜 형식(YYYY-MM-DD)을 확인해주세요.
```

#### 2.3 커스텀 에러 처리 함수

```python
def custom_error_handler(error: ToolException) -> str:
    """
    에러 타입에 따라 다른 메시지 반환

    Args:
        error: ToolException 객체

    Returns:
        사용자 친화적인 에러 메시지
    """
    error_msg = str(error)

    # 날짜 형식 오류
    if "날짜 형식" in error_msg:
        return "❌ 날짜 형식이 올바르지 않습니다. YYYY-MM-DD 형식으로 입력해주세요. (예: 2024-01-03)"

    # 데이터 없음
    elif "거래 데이터가 없습니다" in error_msg:
        return "⚠️ 해당 날짜는 거래일이 아닙니다. 주말이나 공휴일이 아닌지 확인해주세요."

    # 주식 티커 오류
    elif "주식 데이터를 찾을 수 없습니다" in error_msg:
        return "❌ 유효하지 않은 주식 티커입니다. 올바른 티커 심볼을 입력해주세요. (예: AAPL, MSFT)"

    # 기타 오류
    else:
        return f"⚠️ 주식 정보 조회 중 오류가 발생했습니다: {error_msg}"

# 도구 생성
stock_tool_handler = StructuredTool.from_function(
    func=get_stock_price,
    name="stock_price_handler",
    handle_tool_error=custom_error_handler
)

# 테스트
print("날짜 형식 오류:")
print(stock_tool_handler.invoke({"symbol": "AAPL", "date": "2024/01/03"}))

print("\n거래 없는 날짜:")
print(stock_tool_handler.invoke({"symbol": "AAPL", "date": "2024-01-06"}))

print("\n잘못된 티커:")
print(stock_tool_handler.invoke({"symbol": "INVALID123"}))
```

**출력 예시:**
```
날짜 형식 오류:
❌ 날짜 형식이 올바르지 않습니다. YYYY-MM-DD 형식으로 입력해주세요. (예: 2024-01-03)

거래 없는 날짜:
⚠️ 해당 날짜는 거래일이 아닙니다. 주말이나 공휴일이 아닌지 확인해주세요.

잘못된 티커:
❌ 유효하지 않은 주식 티커입니다. 올바른 티커 심볼을 입력해주세요. (예: AAPL, MSFT)
```

#### 2.4 Agent에서 에러 처리 활용

```python
# 에러 처리가 적용된 도구로 Agent 생성
agent_with_error_handling = create_agent(
    model=llm,
    tools=[stock_tool_handler],
    system_prompt="""당신은 주식 정보 조회를 도와주는 AI 어시스턴트입니다.

    도구 실행 시 에러가 발생하면, 에러 메시지를 사용자에게 전달하고
    올바른 형식으로 다시 시도하도록 안내하세요."""
)

# 잘못된 입력으로 실행
result = agent_with_error_handling.invoke({
    "messages": [{
        "role": "user",
        "content": "2024/01/03 날짜의 Apple 주식 가격 알려줘"
    }]
})

# Agent가 에러를 받아서 사용자에게 안내
final_message = result["messages"][-1]
print(final_message.content)
```

**출력 예시:**
```
날짜 형식이 올바르지 않습니다. 날짜는 YYYY-MM-DD 형식으로 입력해주세요.
예를 들어 "2024-01-03"과 같이 입력하시면 됩니다.
```

### Step 3: BaseTool 상속

#### 3.1 입력 스키마 정의

```python
from pydantic import BaseModel, Field

class StockPriceInput(BaseModel):
    """주식 가격 조회 도구 입력 스키마"""
    symbol: str = Field(description="주식 티커 심볼 (예: AAPL, MSFT)")
    date: str = Field(description="조회 날짜 (YYYY-MM-DD 형식)")
```

#### 3.2 BaseTool 상속하여 도구 구현

```python
from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from typing import Optional, Dict

class StockPriceTool(BaseTool):
    """주식 가격 조회 도구 (BaseTool 상속)"""

    name: str = "StockPrice"
    description: str = "yfinance를 사용하여 특정 날짜의 주식 가격 정보를 조회합니다."
    args_schema: type[BaseModel] = StockPriceInput
    return_direct: bool = False

    def _run(
        self,
        symbol: str,
        date: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Dict:
        """
        동기 실행 로직

        Args:
            symbol: 주식 티커 심볼
            date: 조회 날짜
            run_manager: 콜백 매니저 (선택)

        Returns:
            주식 가격 정보 딕셔너리
        """
        # 기존 함수 재사용
        return get_stock_price(symbol, date)

    async def _arun(
        self,
        symbol: str,
        date: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> Dict:
        """
        비동기 실행 로직

        Args:
            symbol: 주식 티커 심볼
            date: 조회 날짜
            run_manager: 비동기 콜백 매니저 (선택)

        Returns:
            주식 가격 정보 딕셔너리
        """
        # 동기 함수 호출 (yfinance는 비동기 미지원)
        # 실제 비동기 구현이 필요한 경우 aiohttp 등 사용
        return self._run(symbol, date, run_manager=None)
```

#### 3.3 도구 실행

```python
# 도구 생성
stock_tool_base = StockPriceTool()

# 도구 속성 확인
print("도구 이름:", stock_tool_base.name)
print("도구 설명:", stock_tool_base.description)
print("입력 스키마:", stock_tool_base.args_schema.model_json_schema())

# 동기 실행
result1 = stock_tool_base.invoke({"symbol": "AAPL", "date": "2024-01-03"})
print("\n동기 실행 결과:")
pprint(result1)

# 비동기 실행
result2 = await stock_tool_base.ainvoke({"symbol": "MSFT", "date": "2024-01-03"})
print("\n비동기 실행 결과:")
pprint(result2)
```

#### 3.4 LLM과 통합

```python
llm_with_base_tool = llm.bind_tools([stock_tool_base])

# LLM 호출
response = llm_with_base_tool.invoke("2024년 1월 3일 Tesla 주식 가격을 알려줘")
pprint(response.tool_calls)

# 도구 실행
if response.tool_calls:
    tool_call = response.tool_calls[0]
    result = stock_tool_base.invoke(tool_call)
    print("\n결과:")
    pprint(result)
```

#### 3.5 BaseTool의 장점 활용 - 상태 관리

```python
class StockPriceToolWithCache(BaseTool):
    """캐시 기능이 있는 주식 가격 조회 도구"""

    name: str = "StockPriceWithCache"
    description: str = "주식 가격 조회 (캐시 지원)"
    args_schema: type[BaseModel] = StockPriceInput

    # 클래스 속성으로 캐시 관리
    _cache: Dict[str, Dict] = {}

    def _run(self, symbol: str, date: str, run_manager=None) -> Dict:
        """캐시를 활용한 실행 로직"""
        cache_key = f"{symbol}_{date}"

        # 캐시 확인
        if cache_key in self._cache:
            print(f"✅ 캐시에서 가져옴: {cache_key}")
            return self._cache[cache_key]

        # API 호출
        print(f"🔄 API 호출: {cache_key}")
        result = get_stock_price(symbol, date)

        # 캐시 저장
        self._cache[cache_key] = result

        return result

    async def _arun(self, symbol: str, date: str, run_manager=None) -> Dict:
        return self._run(symbol, date)

# 도구 생성 및 테스트
cached_tool = StockPriceToolWithCache()

# 첫 호출 (API 호출)
result1 = cached_tool.invoke({"symbol": "AAPL", "date": "2024-01-03"})

# 두 번째 호출 (캐시 사용)
result2 = cached_tool.invoke({"symbol": "AAPL", "date": "2024-01-03"})

# 다른 심볼 (API 호출)
result3 = cached_tool.invoke({"symbol": "MSFT", "date": "2024-01-03"})
```

**출력 예시:**
```
🔄 API 호출: AAPL_2024-01-03
✅ 캐시에서 가져옴: AAPL_2024-01-03
🔄 API 호출: MSFT_2024-01-03
```

### Step 4: QA 체인 구성

#### 4.1 기본 QA 체인

```python
from langchain_core.prompts import ChatPromptTemplate

# 프롬프트 템플릿
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 주식 분석 전문가입니다.

    주어진 주식 데이터를 분석하여 다음 정보를 제공하세요:
    1. 주요 가격 변동
    2. 거래량 분석
    3. 간단한 투자 인사이트

    데이터: {stock_data}
    """),
    ("user", "{query}")
])

# QA 체인 구성
qa_chain = qa_prompt | llm

# 사용 예시
stock_data = stock_tool.invoke({"symbol": "AAPL", "date": "2024-01-03"})

response = qa_chain.invoke({
    "stock_data": stock_data,
    "query": "이 주식에 대해 분석해줘"
})

print(response.content)
```

#### 4.2 도구 통합 QA 체인

```python
from langchain_core.runnables import RunnableLambda

def stock_qa_pipeline(query: str) -> str:
    """
    주식 정보 조회 → 분석 파이프라인

    Args:
        query: 사용자 질문 (예: "2024년 1월 3일 Apple 주식 분석해줘")

    Returns:
        분석 결과
    """
    # 1. LLM으로 티커와 날짜 추출
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", "사용자 질문에서 주식 티커 심볼과 날짜를 추출하여 JSON 형식으로 반환하세요."),
        ("user", "{query}")
    ])

    class ExtractionOutput(BaseModel):
        symbol: str = Field(description="주식 티커 심볼")
        date: Optional[str] = Field(description="날짜 (YYYY-MM-DD)")

    extraction_chain = (
        extraction_prompt
        | llm.with_structured_output(ExtractionOutput)
    )

    extracted = extraction_chain.invoke({"query": query})
    print(f"추출된 정보: {extracted}")

    # 2. 주식 데이터 조회
    stock_data = stock_tool.invoke({
        "symbol": extracted.symbol,
        "date": extracted.date
    })
    print(f"주식 데이터: {stock_data}")

    # 3. 분석
    analysis = qa_chain.invoke({
        "stock_data": stock_data,
        "query": query
    })

    return analysis.content

# 실행
result = stock_qa_pipeline("2024년 1월 3일 Apple 주식을 분석해줘")
print("\n최종 분석:")
print(result)
```

**출력 예시:**
```
추출된 정보: symbol='AAPL' date='2024-01-03'
주식 데이터: {'Date': '2024-01-03', 'Open': 184.22, ...}

최종 분석:
Apple 주식 (2024-01-03) 분석:

1. 주요 가격 변동:
   - 시가: $184.22
   - 종가: $184.25
   - 일일 변동: +$0.03 (+0.02%)
   - 안정적인 횡보 거래

2. 거래량 분석:
   - 거래량: 58,414,200주
   - 평균 대비 보통 수준

3. 투자 인사이트:
   - 변동성이 낮은 안정적인 거래일
   - 추가 분석을 위해 전후 거래일 데이터 필요
```

## 🎯 실습 문제

### 문제 1: 기간별 주식 데이터 조회 도구 ⭐⭐

yfinance의 history 메소드를 사용하여 시작일~종료일 기간의 거래 데이터를 가져오는 도구를 생성하세요.

**요구사항:**
- 입력: symbol (str), start_date (str), end_date (str)
- start와 end 파라미터 사용
- 데이터프레임을 딕셔너리 리스트로 변환
- StructuredTool 사용

```python
# TODO: 여기에 코드를 작성하세요.
```

### 문제 2: 에러 처리 강화 ⭐⭐⭐

다양한 에러 상황을 처리하는 커스텀 에러 핸들러를 구현하세요.

**요구사항:**
- 날짜 형식 오류
- 미래 날짜 오류
- 잘못된 티커 심볼
- 네트워크 오류
- 각 에러별로 다른 사용자 친화적 메시지 반환

```python
# TODO: 여기에 코드를 작성하세요.
```

### 문제 3: BaseTool로 복잡한 도구 구현 ⭐⭐⭐⭐

BaseTool을 상속하여 여러 주식을 비교하는 도구를 구현하세요.

**요구사항:**
- 입력: 여러 심볼 리스트, 날짜
- 모든 심볼의 데이터 조회
- 종가 기준 비교 분석
- 캐싱 기능 추가
- 비동기 실행 지원

```python
# TODO: 여기에 코드를 작성하세요.
```

### 문제 4: Naver 뉴스 검색 통합 ⭐⭐⭐⭐⭐

Naver 개발자 API를 사용하여 뉴스 검색 도구를 만들고, 주식 정보와 결합한 분석 체인을 구성하세요.

**요구사항:**
- Naver 뉴스 검색 API 도구 구현
- 주식 정보 + 뉴스 검색 결합
- 종합 분석 리포트 생성
- Agent 패턴 사용

```python
# TODO: 여기에 코드를 작성하세요.
# Hint: https://developers.naver.com/docs/serviceapi/search/news/news.md
```

## ✅ 솔루션 예시

### 솔루션 1: 기간별 주식 데이터 조회 도구

```python
from pydantic import BaseModel, Field
from typing import List, Dict

class StockPeriodInput(BaseModel):
    """기간별 주식 조회 입력 스키마"""
    symbol: str = Field(description="주식 티커 심볼")
    start_date: str = Field(description="시작 날짜 (YYYY-MM-DD)")
    end_date: str = Field(description="종료 날짜 (YYYY-MM-DD)")

def get_stock_period_data(symbol: str, start_date: str, end_date: str) -> List[Dict]:
    """
    특정 기간의 주식 데이터를 조회합니다.

    Args:
        symbol: 주식 티커 심볼
        start_date: 시작 날짜
        end_date: 종료 날짜

    Returns:
        기간별 주식 데이터 리스트
    """
    # 날짜 검증
    if not is_valid_date(start_date) or not is_valid_date(end_date):
        raise ToolException("날짜 형식이 올바르지 않습니다. YYYY-MM-DD 형식을 사용하세요.")

    # 시작일이 종료일보다 늦은 경우
    if start_date > end_date:
        raise ToolException("시작 날짜는 종료 날짜보다 빠를 수 없습니다.")

    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(start=start_date, end=end_date)

        if hist.empty:
            return {
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "message": "해당 기간에 거래 데이터가 없습니다.",
                "data": []
            }

        # 데이터프레임 처리
        hist = hist.reset_index()
        hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
        data_list = hist.to_dict('records')

        return {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "count": len(data_list),
            "data": data_list
        }

    except Exception as e:
        raise ToolException(f"데이터 조회 중 오류: {str(e)}")

# StructuredTool 생성
stock_period_tool = StructuredTool.from_function(
    func=get_stock_period_data,
    name="stock_period_data",
    description="특정 기간의 주식 거래 데이터를 조회합니다.",
    args_schema=StockPeriodInput,
    handle_tool_error=True
)

# 테스트
result = stock_period_tool.invoke({
    "symbol": "AAPL",
    "start_date": "2024-01-02",
    "end_date": "2024-01-05"
})

print(f"조회 기간: {result['start_date']} ~ {result['end_date']}")
print(f"데이터 개수: {result['count']}")
print("\n데이터 샘플:")
pprint(result['data'][:2])
```

### 솔루션 2: 에러 처리 강화

```python
from datetime import datetime

def enhanced_error_handler(error: ToolException) -> str:
    """강화된 에러 처리 핸들러"""
    error_msg = str(error)

    # 날짜 형식 오류
    if "날짜 형식" in error_msg:
        return ("❌ **날짜 형식 오류**\n"
                "올바른 형식: YYYY-MM-DD\n"
                "예시: 2024-01-03")

    # 시작일/종료일 순서 오류
    elif "시작 날짜는 종료 날짜보다 빠를 수 없습니다" in error_msg:
        return ("❌ **날짜 순서 오류**\n"
                "시작 날짜가 종료 날짜보다 늦습니다.\n"
                "날짜 순서를 확인해주세요.")

    # 미래 날짜 (추가 구현)
    elif "미래" in error_msg:
        return ("⚠️ **미래 날짜**\n"
                "미래 날짜의 주식 데이터는 조회할 수 없습니다.\n"
                f"현재 날짜: {datetime.now().strftime('%Y-%m-%d')}")

    # 데이터 없음
    elif "거래 데이터가 없습니다" in error_msg or "data" in error_msg.lower():
        return ("⚠️ **데이터 없음**\n"
                "해당 날짜/기간에 거래 데이터가 없습니다.\n"
                "주말, 공휴일, 휴장일이 아닌지 확인해주세요.")

    # 잘못된 티커
    elif "주식 데이터를 찾을 수 없습니다" in error_msg or "invalid" in error_msg.lower():
        return ("❌ **잘못된 티커 심볼**\n"
                "유효하지 않은 주식 티커입니다.\n"
                "예시: AAPL (Apple), MSFT (Microsoft), TSLA (Tesla)")

    # 네트워크 오류
    elif "connection" in error_msg.lower() or "timeout" in error_msg.lower():
        return ("🌐 **네트워크 오류**\n"
                "데이터 서버에 연결할 수 없습니다.\n"
                "인터넷 연결을 확인하고 잠시 후 다시 시도해주세요.")

    # 기타 오류
    else:
        return f"⚠️ **오류 발생**\n{error_msg}\n\n고객 지원팀에 문의해주세요."

# 미래 날짜 체크 추가 버전
def get_stock_price_enhanced(symbol: str, date: Optional[str] = None) -> Dict:
    """미래 날짜 체크가 추가된 주식 조회 함수"""

    if date:
        # 날짜 검증
        if not is_valid_date(date):
            raise ToolException(f"잘못된 날짜 형식입니다: {date}")

        # 미래 날짜 체크
        query_date = datetime.strptime(date, "%Y-%m-%d")
        if query_date > datetime.now():
            raise ToolException(f"미래 날짜는 조회할 수 없습니다: {date}")

    return get_stock_price(symbol, date)

# 도구 생성
enhanced_stock_tool = StructuredTool.from_function(
    func=get_stock_price_enhanced,
    name="stock_price_enhanced",
    description="강화된 에러 처리가 적용된 주식 가격 조회 도구",
    handle_tool_error=enhanced_error_handler
)

# 테스트
test_cases = [
    {"symbol": "AAPL", "date": "2024/01/03"},  # 형식 오류
    {"symbol": "AAPL", "date": "2025-12-31"},  # 미래 날짜
    {"symbol": "INVALID123"},                  # 잘못된 티커
    {"symbol": "AAPL", "date": "2024-01-06"},  # 주말
]

for test in test_cases:
    print(f"\n테스트: {test}")
    print(enhanced_stock_tool.invoke(test))
    print("-" * 80)
```

### 솔루션 3: BaseTool로 복잡한 도구 구현

```python
from typing import List

class StockComparisonInput(BaseModel):
    """주식 비교 도구 입력 스키마"""
    symbols: List[str] = Field(description="비교할 주식 티커 심볼 리스트")
    date: str = Field(description="비교 날짜 (YYYY-MM-DD)")

class StockComparisonTool(BaseTool):
    """여러 주식을 비교하는 도구"""

    name: str = "StockComparison"
    description: str = "여러 주식의 가격을 비교 분석합니다."
    args_schema: type[BaseModel] = StockComparisonInput

    # 캐시
    _cache: Dict[str, Dict] = {}

    def _run(
        self,
        symbols: List[str],
        date: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Dict:
        """
        여러 주식을 비교 분석

        Args:
            symbols: 주식 티커 리스트
            date: 비교 날짜

        Returns:
            비교 분석 결과
        """
        results = []
        errors = []

        for symbol in symbols:
            cache_key = f"{symbol}_{date}"

            try:
                # 캐시 확인
                if cache_key in self._cache:
                    data = self._cache[cache_key]
                else:
                    data = get_stock_price(symbol, date)
                    self._cache[cache_key] = data

                results.append({
                    "symbol": symbol,
                    "data": data,
                    "status": "success"
                })

            except Exception as e:
                errors.append({
                    "symbol": symbol,
                    "error": str(e),
                    "status": "error"
                })

        # 성공한 데이터만 분석
        if results:
            # 종가 기준 정렬
            sorted_results = sorted(
                [r for r in results if r["status"] == "success"],
                key=lambda x: x["data"].get("Close", 0),
                reverse=True
            )

            return {
                "date": date,
                "total_symbols": len(symbols),
                "success_count": len(results),
                "error_count": len(errors),
                "ranked_by_close": sorted_results,
                "errors": errors,
                "highest": sorted_results[0]["symbol"] if sorted_results else None,
                "lowest": sorted_results[-1]["symbol"] if sorted_results else None
            }

        return {
            "date": date,
            "total_symbols": len(symbols),
            "success_count": 0,
            "error_count": len(errors),
            "errors": errors
        }

    async def _arun(
        self,
        symbols: List[str],
        date: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> Dict:
        """비동기 실행 (동기 함수 호출)"""
        return self._run(symbols, date)

# 도구 생성 및 테스트
comparison_tool = StockComparisonTool()

result = comparison_tool.invoke({
    "symbols": ["AAPL", "MSFT", "GOOGL", "TSLA"],
    "date": "2024-01-03"
})

print(f"비교 날짜: {result['date']}")
print(f"성공: {result['success_count']}, 실패: {result['error_count']}")
print(f"\n최고가: {result['highest']}")
print(f"최저가: {result['lowest']}")

print("\n순위:")
for i, stock in enumerate(result['ranked_by_close'], 1):
    data = stock['data']
    print(f"{i}. {stock['symbol']}: ${data.get('Close', 0):.2f}")
```

### 솔루션 4: Naver 뉴스 검색 통합

```python
import requests
from typing import Dict, List

@tool
def naver_news_search(query: str, display: int = 5) -> Dict:
    """
    네이버 검색 API를 사용하여 뉴스 검색 결과를 조회합니다.

    Args:
        query: 검색어
        display: 검색 결과 개수 (기본 5개)

    Returns:
        검색 결과 딕셔너리
    """
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id": os.getenv("NAVER_CLIENT_ID"),
        "X-Naver-Client-Secret": os.getenv("NAVER_CLIENT_SECRET")
    }
    params = {
        "query": query,
        "display": display,
        "sort": "date"  # 최신순
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        # HTML 태그 제거
        import re
        for item in data.get('items', []):
            item['title'] = re.sub('<[^<]+?>', '', item['title'])
            item['description'] = re.sub('<[^<]+?>', '', item['description'])

        return {
            "total": data.get('total', 0),
            "items": data.get('items', []),
            "query": query
        }

    except Exception as e:
        raise ToolException(f"뉴스 검색 중 오류: {str(e)}")

# 주식 분석 Agent (주식 정보 + 뉴스)
stock_analysis_agent = create_agent(
    model=llm,
    tools=[stock_tool, naver_news_search],
    system_prompt="""당신은 주식 분석 전문가입니다.

    사용자가 주식에 대해 질문하면:
    1. stock_price 도구로 주식 가격 정보 조회
    2. naver_news_search 도구로 관련 뉴스 검색
    3. 두 정보를 종합하여 분석 제공

    분석 시 다음을 포함하세요:
    - 현재 주가 정보
    - 최근 뉴스 요약 (최대 3개)
    - 종합 의견
    """
)

# 실행
result = stock_analysis_agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Apple 주식에 대해 분석해줘 (2024년 1월 3일 기준)"
    }]
})

# 최종 분석 출력
final_message = result["messages"][-1]
print(final_message.content)
```

**출력 예시:**
```
Apple (AAPL) 종합 분석 (2024-01-03 기준)

📊 주가 정보:
- 시가: $184.22
- 종가: $184.25
- 거래량: 58,414,200주

📰 최근 뉴스:
1. "애플, AI 기능 강화한 새 아이폰 출시 예정"
2. "애플 앱스토어 규제 관련 EU와 협상 중"
3. "애플 실적 발표 앞두고 주가 안정세"

💡 종합 의견:
주가는 안정적이며 변동성이 낮은 상태입니다.
AI 기능 강화 소식은 긍정적이나, EU 규제 이슈는 주의가 필요합니다.
```

## 🚀 실무 활용 예시

### 예시 1: 포트폴리오 분석 시스템

```python
from langchain_core.runnables import RunnablePassthrough

# 포트폴리오 분석 체인
def analyze_portfolio(portfolio: List[Dict]) -> str:
    """
    포트폴리오 전체 분석

    Args:
        portfolio: [{"symbol": "AAPL", "shares": 10}, ...]

    Returns:
        분석 리포트
    """
    # 1. 모든 주식 현재가 조회
    symbols = [item["symbol"] for item in portfolio]
    comparison = comparison_tool.invoke({
        "symbols": symbols,
        "date": datetime.now().strftime("%Y-%m-%d")
    })

    # 2. 포트폴리오 가치 계산
    total_value = 0
    holdings = []

    for item in portfolio:
        symbol = item["symbol"]
        shares = item["shares"]

        # 해당 심볼 데이터 찾기
        stock_data = next(
            (s for s in comparison["ranked_by_close"] if s["symbol"] == symbol),
            None
        )

        if stock_data:
            price = stock_data["data"]["Close"]
            value = price * shares
            total_value += value

            holdings.append({
                "symbol": symbol,
                "shares": shares,
                "price": price,
                "value": value,
                "weight": 0  # 나중에 계산
            })

    # 비중 계산
    for holding in holdings:
        holding["weight"] = (holding["value"] / total_value) * 100

    # 3. LLM으로 분석
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 포트폴리오 분석 전문가입니다."),
        ("user", """다음 포트폴리오를 분석해주세요:

총 가치: ${total_value:,.2f}

보유 종목:
{holdings}

분석 내용:
1. 포트폴리오 구성 평가
2. 분산 투자 수준
3. 개선 제안
""")
    ])

    analysis = (analysis_prompt | llm).invoke({
        "total_value": total_value,
        "holdings": "\n".join([
            f"- {h['symbol']}: {h['shares']}주, ${h['value']:,.2f} ({h['weight']:.1f}%)"
            for h in holdings
        ])
    })

    return analysis.content

# 사용
my_portfolio = [
    {"symbol": "AAPL", "shares": 10},
    {"symbol": "MSFT", "shares": 15},
    {"symbol": "GOOGL", "shares": 5}
]

report = analyze_portfolio(my_portfolio)
print(report)
```

### 예시 2: 자동 알림 시스템

```python
def price_alert_system(symbol: str, target_price: float, check_interval: int = 60):
    """
    주가 알림 시스템

    Args:
        symbol: 주식 티커
        target_price: 목표 가격
        check_interval: 확인 주기 (초)
    """
    import time

    print(f"📊 {symbol} 주가 모니터링 시작")
    print(f"목표가: ${target_price}")

    while True:
        try:
            # 현재가 조회
            data = stock_tool.invoke({"symbol": symbol})
            current_price = data.get("Close", 0)

            print(f"\r현재가: ${current_price:.2f}", end="")

            # 목표가 도달 시
            if current_price >= target_price:
                print(f"\n\n🎯 목표가 도달! ${current_price:.2f}")

                # 뉴스 검색
                news = naver_news_search.invoke(f"{symbol} 주식")

                # 알림 메시지 생성
                alert_msg = f"""
                ⚠️ 주가 알림
                종목: {symbol}
                현재가: ${current_price:.2f}
                목표가: ${target_price:.2f}

                최근 뉴스:
                {news['items'][0]['title'] if news['items'] else 'N/A'}
                """

                print(alert_msg)
                break

            time.sleep(check_interval)

        except KeyboardInterrupt:
            print("\n\n모니터링 중단")
            break

# 사용 (주의: 실제 운영 시 백그라운드 작업 필요)
# price_alert_system("AAPL", 185.0, check_interval=300)
```

## 📖 참고 자료

### 공식 문서
- [LangChain Error Handling](https://python.langchain.com/docs/how_to/tool_error_handling/)
- [BaseTool API](https://api.python.langchain.com/en/latest/tools/langchain_core.tools.base.BaseTool.html)
- [Yahoo Finance API](https://pypi.org/project/yfinance/)
- [Naver 개발자 API](https://developers.naver.com/)

### 추가 학습 자료
- 외부 API 통합 베스트 프랙티스
- 에러 처리 및 재시도 전략
- BaseTool 고급 활용 패턴
- 실시간 데이터 처리 및 캐싱

### 관련 노트북
- `PRJ03_W1_003_LangChain_Custom_Tool_Part1.md` - Custom Tool 기초
- `PRJ03_W1_001_ToolCalling_Agent_Intro.md` - Tool Calling 개념
- `PRJ03_W1_002_LangChain_BuiltIn_Tool.md` - Built-in Tools

---

**학습 완료 체크리스트:**
- [ ] Yahoo Finance API 연동 이해
- [ ] 3가지 에러 처리 방법 숙지
- [ ] BaseTool 상속 패턴 이해
- [ ] QA 체인 구성 방법 습득
- [ ] 실습 문제 4개 완료
- [ ] 실무 예시 코드 실행 및 이해
