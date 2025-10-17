# W2_006 RunnableConfig와 Fallback 처리

## 학습 목표
- LangChain RunnableConfig를 활용한 런타임 동작 제어 방법 학습
- 성능 모니터링 및 알림 콜백 핸들러 구현
- Fallback 메커니즘을 통한 안정적인 LLM 애플리케이션 구축
- 동적 모델 선택 및 프롬프트 전환 기법 습득

## 핵심 개념

### 1. RunnableConfig란?
- **정의**: LangChain에서 런타임에 Runnable(실행 가능한 컴포넌트)의 동작을 세밀하게 제어하기 위한 설정 객체
- **역할**: 체인, 툴, 모델 등 다양한 Runnable에 전달되어 실행 시 동작을 조정
- **특징**: 실행 중인 Runnable과 하위 호출들에 설정을 전달하는 컨텍스트 역할

### 2. RunnableConfig의 주요 속성

#### configurable
- **용도**: 런타임에 조정 가능한 속성 값 전달
- **예시**: 모델의 온도, 세션 ID, 프롬프트 템플릿 등

#### callbacks
- **용도**: 실행 과정에서 이벤트를 처리할 콜백 핸들러 지정
- **예시**: 로깅, 모니터링, 이벤트 추적

#### tags
- **용도**: 실행에 태그를 붙여 추적 및 필터링
- **예시**: 실험 버전, 사용자 그룹, 요청 타입 등

#### metadata
- **용도**: 실행 관련 추가 메타데이터 전달
- **예시**: 요청 ID, 사용자 정보, 세션 데이터 등

### 3. Fallback이란?
- **정의**: 주요 실행 경로가 실패했을 때 자동으로 대체 경로를 실행하는 메커니즘
- **필요성**: API 안정성 문제, 비용 최적화, 성능 최적화, 컨텍스트 길이 제한 대응

## 환경 설정

### 필수 라이브러리 설치
```bash
pip install langchain langchain-openai langchain-google-genai
pip install python-dotenv
```

### 환경 변수 설정
```python
from dotenv import load_dotenv
import os

# 환경 변수 로드
load_dotenv()

# Langsmith 추적 확인
print("Langsmith 추적:", os.getenv('LANGSMITH_TRACING'))
```

## 1단계: 성능 모니터링 콜백 핸들러 구현

### PerformanceMonitoringCallback 클래스
```python
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class PerformanceMonitoringCallback(BaseCallbackHandler):
    """LLM 호출 성능을 모니터링하는 콜백 핸들러"""

    def __init__(self):
        self.start_time: Optional[float] = None       # LLM 호출 시작 시간
        self.token_usage: Dict[str, Any] = {}         # 토큰 사용량 정보
        self.call_count: int = 0                      # LLM 호출 횟수

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any
    ) -> None:
        """LLM 호출이 시작될 때 호출"""
        self.start_time = time.time()
        self.call_count += 1
        print(f"🚀 LLM 호출 #{self.call_count} 시작 - {datetime.now().strftime('%H:%M:%S')}")

        # 첫 번째 프롬프트의 길이 확인
        if prompts:
            print(f"📝 프롬프트 길이: {len(prompts[0])} 문자")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """LLM 호출이 완료될 때 호출"""
        if self.start_time:
            duration = time.time() - self.start_time
            print(f"✅ LLM 호출 완료 - 소요시간: {duration:.2f}초")

            # 토큰 사용량 추적
            if response.generations:
                generation = response.generations[0][0]
                usage = response.llm_output.get('token_usage', {})

                if usage:
                    print(f"🔢 토큰 사용량: {usage}")
                    self.token_usage = usage

                # 응답 길이 체크
                if hasattr(generation, 'text'):
                    response_text = generation.text
                    print(f"📊 응답 길이: {len(response_text)} 문자")

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """LLM 호출에서 오류가 발생할 때 호출"""
        print(f"❌ LLM 호출 오류: {str(error)}")

    def get_statistics(self) -> Dict[str, Any]:
        """현재까지의 통계 정보를 반환"""
        return {
            "total_calls": self.call_count,
            "last_token_usage": self.token_usage
        }

# 사용 예시
performance_callback = PerformanceMonitoringCallback()
print("성능 모니터링 콜백 핸들러 생성 완료")
```

## 2단계: 실시간 알림 콜백 핸들러 구현

### AlertCallback 클래스
```python
class AlertCallback(BaseCallbackHandler):
    """특정 조건에서 알림을 보내는 콜백 핸들러"""

    def __init__(
        self,
        cost_threshold: float = 1.0,            # 비용 임계값 (달러 단위)
        response_time_threshold: float = 10.0,  # 응답 시간 임계값 (초 단위)
        token_threshold: int = 4000             # 긴 프롬프트 토큰 임계값
    ):
        self.cost_threshold = cost_threshold
        self.response_time_threshold = response_time_threshold
        self.token_threshold = token_threshold
        self.start_time: Optional[float] = None
        self.cumulative_cost: float = 0.0

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any
    ) -> None:
        """LLM 호출이 시작될 때 호출"""
        self.start_time = time.time()

        # 긴 프롬프트 경고
        if prompts and len(prompts[0]) > self.token_threshold:
            self._send_alert(f"⚠️ 긴 프롬프트 감지: {len(prompts[0])} 문자")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """LLM 호출이 완료될 때 호출"""
        # 응답 시간 체크
        if self.start_time:
            duration = time.time() - self.start_time
            if duration > self.response_time_threshold:
                self._send_alert(f"🐌 느린 응답: {duration:.2f}초")

        # 비용 체크
        if response.generations:
            usage = response.llm_output.get('token_usage', {})

            if usage:
                # 토큰 사용량 계산
                total_tokens = usage.get('total_tokens', 0)
                if total_tokens == 0:
                    total_tokens = usage.get('input_tokens', 0) + usage.get('output_tokens', 0)

                # 간단한 비용 계산 (실제로는 모델별 가격 적용 필요)
                estimated_cost = (total_tokens / 1000) * 0.002
                self.cumulative_cost += estimated_cost

                if self.cumulative_cost > self.cost_threshold:
                    self._send_alert(f"💸 비용 임계값 초과: ${self.cumulative_cost:.4f}")

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """LLM 호출에서 오류가 발생할 때 호출"""
        self._send_alert(f"🚨 LLM 오류 발생: {str(error)}")

    def _send_alert(self, message: str) -> None:
        """실제 환경에서는 Slack, Discord, 이메일 등으로 알림 전송"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        alert_message = f"[ALERT] {timestamp} - {message}"
        print("🔔 알림 전송:", alert_message)

        # 로깅 시스템에 기록
        logging.warning(alert_message)

        # 실제 구현에서는 외부 서비스에 알림 전송
        # self._send_slack_notification(message)
        # self._send_email_notification(message)

    def reset_cost_tracking(self) -> None:
        """누적 비용 추적을 리셋"""
        self.cumulative_cost = 0.0

# 알림 콜백 핸들러 생성
alert_callback = AlertCallback(
    cost_threshold=0.01,           # $0.01 이상 시 알림
    response_time_threshold=5.0,   # 5초 이상 시 알림
    token_threshold=3000           # 3000 문자 이상 시 알림
)
print("알림 콜백 핸들러 생성 완료")
```

## 3단계: RunnableConfig 기본 사용

### configurable_fields를 통한 동적 설정
```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import ConfigurableField

# 동적으로 설정 가능한 모델 생성
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
).configurable_fields(
    temperature=ConfigurableField(
        id="llm_temperature",
        name="LLM Temperature",
        description="The temperature of the LLM"
    ),
    model=ConfigurableField(
        id="llm_model",
        name="LLM Model",
        description="The model to use"
    )
)

# 프롬프트 템플릿 정의
prompt = PromptTemplate.from_template(
    "다음 질문에 답변해주세요: {question}"
)

# 체인 구성
chain = prompt | llm | StrOutputParser()

# RunnableConfig를 사용한 실행
from langchain_core.runnables import RunnableConfig

# 기본 실행
response1 = chain.invoke(
    {"question": "인공지능이란 무엇인가요?"},
    config=RunnableConfig(
        callbacks=[performance_callback, alert_callback],
        tags=["basic_query", "ai_question"],
        metadata={"user_id": "user_001", "session_id": "session_123"}
    )
)

print("\n기본 실행 결과:")
print(response1)

# 온도 조정한 실행
response2 = chain.invoke(
    {"question": "인공지능이란 무엇인가요?"},
    config=RunnableConfig(
        configurable={"llm_temperature": 0.2},  # 더 일관된 답변
        callbacks=[performance_callback],
        tags=["low_temperature"]
    )
)

print("\n낮은 온도 실행 결과:")
print(response2)

# 모델 변경한 실행
response3 = chain.invoke(
    {"question": "인공지능이란 무엇인가요?"},
    config=RunnableConfig(
        configurable={"llm_model": "gpt-4o"},  # 더 강력한 모델
        callbacks=[performance_callback],
        tags=["high_quality_model"]
    )
)

print("\nGPT-4 실행 결과:")
print(response3)
```

## 4단계: 동적 모델 및 프롬프트 선택

### 1. 동적 모델 선택
```python
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# 여러 모델을 대안으로 설정
llm_dynamic = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
).configurable_alternatives(
    ConfigurableField(id="llm_model"),
    default_key="openai",
    gemini=ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7),
    gpt4=ChatOpenAI(model="gpt-4o", temperature=0.7),
)

chain_dynamic = prompt | llm_dynamic | StrOutputParser()

# OpenAI 모델 사용 (기본값)
result_openai = chain_dynamic.invoke(
    {"question": "파이썬의 장점은?"},
    config={"configurable": {"llm_model": "openai"}}
)

print("OpenAI 결과:", result_openai)

# Gemini 모델 사용
result_gemini = chain_dynamic.invoke(
    {"question": "파이썬의 장점은?"},
    config={"configurable": {"llm_model": "gemini"}}
)

print("\nGemini 결과:", result_gemini)

# GPT-4 모델 사용
result_gpt4 = chain_dynamic.invoke(
    {"question": "파이썬의 장점은?"},
    config={"configurable": {"llm_model": "gpt4"}}
)

print("\nGPT-4 결과:", result_gpt4)
```

### 2. 동적 프롬프트 선택
```python
from langchain_core.prompts import PromptTemplate

# 기본 프롬프트
default_prompt = PromptTemplate.from_template(
    "다음 질문에 답변해주세요: {question}"
)

# 상세 프롬프트
detailed_prompt = PromptTemplate.from_template(
    """당신은 전문가입니다. 다음 질문에 상세하고 구조적으로 답변해주세요.

질문: {question}

답변 형식:
1. 개요
2. 핵심 내용
3. 예시
4. 결론
"""
)

# 간결한 프롬프트
concise_prompt = PromptTemplate.from_template(
    "한 문장으로 답변: {question}"
)

# 동적 프롬프트 설정
dynamic_prompt = default_prompt.configurable_alternatives(
    ConfigurableField(id="prompt_style"),
    default_key="default",
    detailed=detailed_prompt,
    concise=concise_prompt,
)

chain_with_dynamic_prompt = dynamic_prompt | llm | StrOutputParser()

# 기본 프롬프트 사용
print("=== 기본 프롬프트 ===")
result1 = chain_with_dynamic_prompt.invoke(
    {"question": "머신러닝이란?"},
    config={"configurable": {"prompt_style": "default"}}
)
print(result1)

# 상세 프롬프트 사용
print("\n=== 상세 프롬프트 ===")
result2 = chain_with_dynamic_prompt.invoke(
    {"question": "머신러닝이란?"},
    config={"configurable": {"prompt_style": "detailed"}}
)
print(result2)

# 간결한 프롬프트 사용
print("\n=== 간결한 프롬프트 ===")
result3 = chain_with_dynamic_prompt.invoke(
    {"question": "머신러닝이란?"},
    config={"configurable": {"prompt_style": "concise"}}
)
print(result3)
```

## 5단계: Fallback 처리 구현

### Fallback이 필요한 이유
1. **API 안정성 문제**: 요금 제한, 서버 다운타임, 네트워크 오류
2. **비용 최적화**: 저렴한 모델을 먼저 시도, 실패 시 비싼 모델 사용
3. **성능 최적화**: 빠른 모델 우선 사용, 복잡한 작업에만 고성능 모델
4. **컨텍스트 길이 제한**: 짧은 컨텍스트 모델 먼저 시도

### Fallback 체인 구성
```python
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# 주 모델 (저렴하고 빠름)
primary_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    request_timeout=5  # 5초 타임아웃
)

# 백업 모델 1
backup_llm_1 = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7
)

# 백업 모델 2 (가장 강력하지만 비쌈)
backup_llm_2 = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7
)

# Fallback 체인 구성
llm_with_fallback = primary_llm.with_fallbacks(
    [backup_llm_1, backup_llm_2]
)

# 프롬프트 및 체인
prompt = PromptTemplate.from_template("다음 질문에 답변: {question}")
fallback_chain = prompt | llm_with_fallback | StrOutputParser()

# Fallback 테스트
try:
    result = fallback_chain.invoke(
        {"question": "양자 컴퓨팅의 미래는?"},
        config=RunnableConfig(
            callbacks=[performance_callback, alert_callback]
        )
    )
    print("결과:", result)
except Exception as e:
    print(f"모든 Fallback 실패: {e}")
```

### 조건부 Fallback
```python
from langchain_core.runnables import RunnableLambda

def check_response_quality(response):
    """응답 품질을 검사하는 함수"""
    # 응답이 너무 짧으면 False 반환
    if len(response) < 50:
        raise ValueError("응답이 너무 짧습니다")
    return response

# 품질 검사를 포함한 체인
quality_checked_chain = (
    prompt
    | primary_llm
    | StrOutputParser()
    | RunnableLambda(check_response_quality)
).with_fallbacks([
    prompt | backup_llm_1 | StrOutputParser(),
    prompt | backup_llm_2 | StrOutputParser()
])

# 실행
result = quality_checked_chain.invoke(
    {"question": "블록체인 기술을 설명해주세요"}
)
print("품질 검증된 결과:", result)
```

## 6단계: 알림 조건 테스트

### 다양한 알림 시나리오
```python
# 1. 긴 프롬프트 테스트
long_question = "인공지능" * 1000  # 긴 프롬프트 생성
try:
    chain.invoke(
        {"question": long_question},
        config=RunnableConfig(callbacks=[alert_callback])
    )
except Exception as e:
    print(f"긴 프롬프트 오류: {e}")

# 2. 다수의 요청으로 비용 임계값 테스트
alert_callback.reset_cost_tracking()
for i in range(10):
    chain.invoke(
        {"question": f"질문 {i+1}: AI란 무엇인가?"},
        config=RunnableConfig(callbacks=[alert_callback])
    )
    print(f"누적 비용: ${alert_callback.cumulative_cost:.4f}")

# 3. 통계 확인
stats = performance_callback.get_statistics()
print("\n최종 통계:")
print(f"총 호출 횟수: {stats['total_calls']}")
print(f"마지막 토큰 사용량: {stats['last_token_usage']}")
```

## 실습 과제

### 기본 실습
1. **커스텀 콜백 핸들러 작성**
   - 특정 키워드가 포함된 응답을 필터링하는 콜백
   - 응답 시간이 일정 시간 이상이면 로그 저장
   - 에러 발생 시 재시도 로직 포함

2. **동적 설정 활용**
   - 사용자 등급에 따라 다른 모델 사용
   - 시간대에 따라 다른 프롬프트 스타일 적용
   - 질문 복잡도에 따라 온도 자동 조정

### 응용 실습
3. **다단계 Fallback 시스템**
   - 3개 이상의 모델로 Fallback 체인 구성
   - 각 모델의 실패 원인 분석 및 로깅
   - 비용과 성능을 고려한 최적 Fallback 전략 수립

4. **실시간 모니터링 대시보드**
   - 콜백 데이터를 수집하여 시각화
   - 성능 지표 추적 (응답 시간, 토큰 사용량, 비용)
   - 알림 발생 이력 관리

### 심화 실습
5. **적응형 모델 선택 시스템**
   - 질문 유형 자동 분류
   - 질문 유형에 따라 최적 모델 자동 선택
   - 성능 피드백을 반영한 모델 선택 개선

6. **프로덕션 레벨 에러 처리**
   - 다양한 에러 시나리오 대응
   - Circuit Breaker 패턴 구현
   - 그레이스풀 디그레이데이션(Graceful Degradation) 구현

## 문제 해결 가이드

### 일반적인 오류들
1. **콜백 핸들러 작동 안 함**
   ```python
   # config에 callbacks를 명시적으로 전달했는지 확인
   config = RunnableConfig(callbacks=[your_callback])
   result = chain.invoke(input_data, config=config)
   ```

2. **Fallback이 작동하지 않음**
   ```python
   # Fallback은 예외 발생 시에만 작동
   # 타임아웃 설정 확인
   llm = ChatOpenAI(request_timeout=5)
   ```

3. **설정 값이 적용되지 않음**
   ```python
   # configurable_fields의 id와 config의 키가 일치하는지 확인
   llm = ChatOpenAI().configurable_fields(
       temperature=ConfigurableField(id="temp")
   )
   # 사용 시
   config = {"configurable": {"temp": 0.5}}
   ```

## 참고 자료
- [LangChain RunnableConfig 공식 문서](https://python.langchain.com/docs/concepts/runnables/)
- [LangChain Fallbacks 가이드](https://python.langchain.com/docs/how_to/fallbacks/)
- [LangSmith Alert 설정](https://docs.smith.langchain.com/observability/how_to_guides/alerts)
- [Callback Handlers 가이드](https://python.langchain.com/docs/modules/callbacks/)

이 학습 가이드를 통해 RunnableConfig와 Fallback 메커니즘을 활용하여 안정적이고 효율적인 LLM 애플리케이션을 구축할 수 있습니다.