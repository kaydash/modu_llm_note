# OpenAI Chat Completion API 활용 가이드

## 📚 학습 목표
LLM의 기본 원리를 이해하고 OpenAI Chat Completion API를 활용하여 다양한 AI 응용 프로그램을 개발할 수 있다.

## 📖 주요 내용
1. LLM(대규모 언어 모델) 기본 개념
2. OpenAI API 핵심 구성 요소
3. Chat Completion API 활용법
4. 멀티모달(텍스트, 이미지, 오디오) 처리
5. 매개변수 최적화 전략

---

# 1. LLM 기본 개념

## 1.1 LLM(Large Language Model)의 생성 원리

### 핵심 개념
**LLM은 어떻게 작동하나요?**
- **트랜스포머 구조**: 대화형 AI의 핵심 아키텍처
- **토큰 예측**: 다음에 올 가장 적절한 단어를 예측
- **학습 방식**: 인터넷의 방대한 텍스트 데이터로 사전 훈련

### 핵심 프로세스
1. **토큰화**: 텍스트를 작은 단위(토큰)로 분할
2. **확률 계산**: 각 토큰이 다음에 올 확률 계산
3. **토큰 생성**: 확률 분포에 따라 토큰 선택
4. **반복**: 종료 조건까지 과정 반복

### 트랜스포머 아키텍처
- **인코더-디코더 구조**: 입력과 출력을 동시에 처리
- **어텐션 메커니즘**: 입력의 모든 부분을 동시에 고려하여 중요한 정보에 집중

![트랜스포머 구조](https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Transformer%2C_full_architecture.png/440px-Transformer%2C_full_architecture.png)

---

# 2. OpenAI API 핵심 개념

## 2.1 주요 구성 요소

### 메시지 형식
```python
messages = [
    {"role": "system", "content": "당신은 도움이 되는 AI 어시스턴트입니다."},
    {"role": "user", "content": "파이썬에서 리스트를 정렬하는 방법을 알려주세요."},
    {"role": "assistant", "content": "sort() 메서드나 sorted() 함수를 사용할 수 있습니다."}
]
```

### 현재 사용 가능한 주요 모델 (2025년 기준)
- **gpt-4.1**: 최고 성능, 복잡한 작업용
- **gpt-4.1-mini**: 빠른 속도, 비용 효율적
- **gpt-4.1-nano**: 초고속, 최저 비용
- **o3, o4-mini**: 복잡한 추론 작업용
- **gpt-4o**: 멀티모달 (텍스트, 이미지, 오디오)

### API 응답 구조
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "gpt-4.1-mini",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "생성된 텍스트"
      }
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 50,
    "total_tokens": 60
  }
}
```

## 2.2 환경 설정

### 프로젝트 설정
```bash
# uv 사용 (권장)
uv init my_ai_project
uv venv --python=3.12
uv add langchain langchain_openai python-dotenv ipykernel

# pip 사용
pip install langchain langchain_openai python-dotenv ipykernel
```

### API 키 설정
```python
# .env 파일 생성
OPENAI_API_KEY=your_api_key_here

# Python에서 로드
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

---

# 3. Chat Completion API 활용법

## 3.1 기본 텍스트 생성

### 클라이언트 설정 및 요청
```python
from openai import OpenAI

# 클라이언트 생성
client = OpenAI()  # API 키는 환경변수에서 자동 로드

# Chat Completion 요청
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "system", "content": "You are a helpful programming assistant."},
        {"role": "user", "content": "파이썬에서 파일을 읽는 방법을 알려주세요."},
    ],
    temperature=0.7,
    max_tokens=1000,
)

# 응답 처리
print("생성된 텍스트:")
print(response.choices[0].message.content)
print(f"토큰 사용량: {response.usage.total_tokens}")
```

### 응답 구조 이해
```python
# 응답의 주요 요소
response_id = response.id                           # 응답 고유 ID
model_used = response.model                         # 사용된 모델
generated_text = response.choices[0].message.content # 생성된 텍스트
token_usage = response.usage                        # 토큰 사용량 정보
```

## 3.2 구조화된 JSON 출력

### JSON Schema를 활용한 구조화된 데이터 추출
```python
from typing import Dict, Any
import json

def extract_product_info(product_text: str) -> Dict[str, Any]:
    """
    상품 정보를 구조화된 JSON으로 추출하는 함수

    Args:
        product_text: 상품 정보가 포함된 텍스트

    Returns:
        구조화된 상품 정보 딕셔너리
    """
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": "상품 정보를 구조화된 형태로 추출하고 분석합니다."
            },
            {
                "role": "user",
                "content": product_text
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "product_schema",
                "description": "상품의 상세 정보를 구조화하기 위한 스키마",
                "schema": {
                    "type": "object",
                    "properties": {
                        "brand": {
                            "type": "string",
                            "description": "제조사 또는 브랜드 이름"
                        },
                        "model": {
                            "type": "string",
                            "description": "제품의 모델명 또는 시리즈명"
                        },
                        "capacity": {
                            "type": "string",
                            "description": "저장 용량 또는 규격"
                        },
                        "color": {
                            "type": "string",
                            "description": "제품의 색상"
                        },
                        "price": {
                            "type": "number",
                            "description": "제품의 가격 (단위: 원)",
                            "minimum": 0
                        },
                        "category": {
                            "type": "string",
                            "description": "제품의 카테고리"
                        }
                    },
                    "required": ["brand", "model", "price"],
                    "additionalProperties": False
                }
            }
        }
    )

    return json.loads(response.choices[0].message.content)

# 사용 예시
product_text = "애플 아이폰 15 프로 256GB (블랙) - 1,500,000원"
product_info = extract_product_info(product_text)
print(product_info)
# 출력: {'brand': '애플', 'model': '아이폰 15 프로', 'capacity': '256GB', 'color': '블랙', 'price': 1500000, 'category': '스마트폰'}
```

---

# 4. 멀티모달 처리

## 4.1 이미지 분석

### URL을 통한 이미지 분석
```python
from typing import Optional
import httpx
from PIL import Image
from io import BytesIO

async def analyze_image_from_url(image_url: str, question: str = "이미지에 대해 설명해주세요.") -> str:
    """
    URL의 이미지를 분석하는 함수

    Args:
        image_url: 분석할 이미지 URL
        question: 이미지에 대한 질문

    Returns:
        이미지 분석 결과
    """
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{question} 한국어로 답변해주세요."},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]
    )

    return response.choices[0].message.content

# 사용 예시
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
result = await analyze_image_from_url(image_url, "이 이미지에서 볼 수 있는 자연 요소들을 설명해주세요.")
print(result)
```

### Base64 인코딩을 통한 로컬 이미지 분석
```python
import base64
from pathlib import Path

def encode_image(image_path: str) -> str:
    """이미지 파일을 Base64로 인코딩"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_local_image(image_path: str, analysis_prompt: str) -> str:
    """
    로컬 이미지 파일을 분석하는 함수

    Args:
        image_path: 로컬 이미지 파일 경로
        analysis_prompt: 분석을 위한 프롬프트

    Returns:
        이미지 분석 결과
    """
    client = OpenAI()

    # 이미지를 Base64로 인코딩
    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": analysis_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
    )

    return response.choices[0].message.content

# 사용 예시
chart_analysis = analyze_local_image(
    "data/sales_chart.jpg",
    "이 차트에서 보이는 주요 트렌드와 패턴을 분석해주세요."
)
print(chart_analysis)
```

## 4.2 오디오 생성 (Text-to-Speech)

### 음성 생성 및 저장
```python
import base64
from pathlib import Path

def generate_speech(text: str, output_path: str = "output.wav", voice: str = "alloy") -> None:
    """
    텍스트를 음성으로 변환하여 파일로 저장하는 함수

    Args:
        text: 음성으로 변환할 텍스트
        output_path: 저장할 파일 경로
        voice: 음성 종류 ("alloy", "echo", "fable", "onyx", "nova", "shimmer")
    """
    client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-4o-mini-audio-preview",
        modalities=["text", "audio"],
        audio={"voice": voice, "format": "wav"},
        messages=[
            {
                "role": "user",
                "content": text
            }
        ]
    )

    # 오디오 데이터 디코딩 및 저장
    wav_bytes = base64.b64decode(completion.choices[0].message.audio.data)
    with open(output_path, "wb") as f:
        f.write(wav_bytes)

    print(f"음성 파일이 저장되었습니다: {output_path}")

    # 텍스트 응답도 반환
    return completion.choices[0].message.content

# 사용 예시
response_text = generate_speech(
    "안녕하세요. OpenAI의 음성 생성 기능을 테스트하고 있습니다.",
    "greeting.wav",
    "alloy"
)
print(f"텍스트 응답: {response_text}")
```

---

# 5. 매개변수 최적화

## 5.1 핵심 매개변수 가이드

| 매개변수 | 범위 | 용도 | 추천값 |
|---------|------|------|--------|
| `temperature` | 0~2 | 창의성 조절 | 0.3 (정확성), 0.7 (균형), 1.2 (창의성) |
| `top_p` | 0~1 | 응답 다양성 | 0.9 (기본), 0.3 (집중적) |
| `max_tokens` | 1~8192+ | 최대 길이 | 작업에 따라 조절 |
| `frequency_penalty` | -2~2 | 반복 억제 | 0.3~0.6 |
| `presence_penalty` | -2~2 | 새 주제 도입 | 0.3~0.6 |

## 5.2 시나리오별 설정

### 정확한 정보 제공
```python
def get_factual_response(question: str) -> str:
    """정확한 정보 제공을 위한 설정"""
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": question}],
        temperature=0.2,    # 낮은 창의성
        top_p=0.3,         # 집중적 응답
        max_tokens=500
    )

    return response.choices[0].message.content

# 사용 예시
factual_info = get_factual_response("파이썬 딕셔너리 메서드들을 설명해주세요.")
print(factual_info)
```

### 창의적 글쓰기
```python
def generate_creative_content(prompt: str) -> str:
    """창의적 콘텐츠 생성을 위한 설정"""
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=1.1,           # 높은 창의성
        top_p=0.9,                # 다양한 표현
        max_tokens=1000,
        frequency_penalty=0.5     # 반복 방지
    )

    return response.choices[0].message.content

# 사용 예시
creative_story = generate_creative_content("우주 정거장에서의 하루를 소설로 써주세요.")
print(creative_story)
```

### 코드 생성
```python
def generate_code(description: str) -> str:
    """코드 생성을 위한 최적화된 설정"""
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": description}],
        temperature=0.4,    # 약간의 창의성
        max_tokens=800
    )

    return response.choices[0].message.content

# 사용 예시
code_example = generate_code("웹 스크래핑을 위한 Python 함수를 만들어주세요.")
print(code_example)
```

---

# 6. 실습 문제

## 문제 1: 언어 번역기 만들기

### 실습해보기
```python
from typing import Optional
from openai import OpenAI

def translator(text: str, target_language: str) -> Optional[str]:
    """
    텍스트를 지정된 언어로 번역하는 함수

    Args:
        text: 번역할 텍스트
        target_language: 목표 언어

    Returns:
        번역된 텍스트 또는 None (오류 시)
    """
    # 여기에 코드를 작성하세요
    pass

# 테스트 코드
result = translator("안녕하세요, 오늘 날씨가 좋네요!", "영어")
print(result)  # 예상 출력: Hello, the weather is nice today!
```

### 해답
```python
from typing import Optional
from openai import OpenAI

def translator(text: str, target_language: str) -> Optional[str]:
    """
    텍스트를 지정된 언어로 번역하는 함수

    Args:
        text: 번역할 텍스트
        target_language: 목표 언어

    Returns:
        번역된 텍스트 또는 None (오류 시)
    """
    try:
        client = OpenAI()

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a professional translator. Translate the given text to {target_language}. Only return the translated text, no explanations."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0.3,
            max_tokens=500
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"번역 중 오류 발생: {e}")
        return None

# 테스트 코드
result = translator("안녕하세요, 오늘 날씨가 좋네요!", "영어")
print(result)  # 출력: Hello, the weather is nice today!
```

## 문제 2: 감정 분석기

### 실습해보기
```python
from typing import Dict, Optional
import json

def analyze_sentiment(text: str) -> Optional[Dict[str, any]]:
    """
    텍스트의 감정을 분석하여 JSON 형태로 반환하는 함수

    Args:
        text: 분석할 텍스트

    Returns:
        {"sentiment": "positive/negative/neutral", "confidence": float} 형태의 딕셔너리
    """
    # 여기에 코드를 작성하세요
    # 힌트: response_format과 json_schema를 활용하세요
    pass

# 테스트 코드
result = analyze_sentiment("오늘 시험을 잘 봤어요! 정말 기쁩니다.")
print(result)  # 예상 출력: {'sentiment': 'positive', 'confidence': 0.95}
```

### 해답
```python
from typing import Dict, Optional
import json

def analyze_sentiment(text: str) -> Optional[Dict[str, any]]:
    """
    텍스트의 감정을 분석하여 JSON 형태로 반환하는 함수

    Args:
        text: 분석할 텍스트

    Returns:
        {"sentiment": "positive/negative/neutral", "confidence": float} 형태의 딕셔너리
    """
    try:
        client = OpenAI()

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a sentiment analysis expert. Analyze the sentiment of the given text and provide confidence score."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "sentiment_analysis",
                    "description": "감정 분석 결과",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "sentiment": {
                                "type": "string",
                                "enum": ["positive", "negative", "neutral"],
                                "description": "감정 분류"
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                                "description": "신뢰도 (0~1)"
                            }
                        },
                        "required": ["sentiment", "confidence"],
                        "additionalProperties": False
                    }
                }
            }
        )

        return json.loads(response.choices[0].message.content)

    except Exception as e:
        print(f"감정 분석 중 오류 발생: {e}")
        return None

# 테스트 코드
result = analyze_sentiment("오늘 시험을 잘 봤어요! 정말 기쁩니다.")
print(result)  # 출력: {'sentiment': 'positive', 'confidence': 0.95}
```

---

# 🎯 핵심 요약

## Chat Completion API 활용 체크리스트
- [ ] 환경변수를 통한 안전한 API 키 관리
- [ ] 적절한 모델 선택 (작업 복잡도에 따라)
- [ ] 메시지 역할(system, user, assistant) 구분
- [ ] 매개변수 최적화 (temperature, max_tokens 등)
- [ ] 구조화된 출력을 위한 JSON Schema 활용
- [ ] 멀티모달 기능 (이미지, 오디오) 활용
- [ ] 예외 처리 및 오류 관리

## 실무 적용 팁
1. **토큰 관리**: 비용 최적화를 위한 토큰 사용량 모니터링
2. **응답 품질**: 적절한 temperature 설정으로 일관성 확보
3. **구조화**: JSON Schema를 활용한 안정적인 데이터 추출
4. **보안**: API 키는 절대 코드에 직접 포함하지 않기
5. **성능**: 배치 처리와 비동기 처리 활용

---

# 📚 참고 자료

## 공식 문서
- [OpenAI API 공식 문서](https://platform.openai.com/docs)
- [Chat Completions API 가이드](https://platform.openai.com/docs/guides/text-generation)
- [OpenAI 토큰 계산기](https://platform.openai.com/tokenizer)

## 추가 학습 자료
- [프롬프트 엔지니어링 가이드](https://platform.openai.com/docs/guides/prompt-engineering)
- [OpenAI Python 라이브러리](https://github.com/openai/openai-python)
- [OpenAI 모델 비교](https://platform.openai.com/docs/models)

## 도구 및 라이브러리
- **HTTP 클라이언트**: `httpx` (비동기), `requests` (동기)
- **이미지 처리**: `Pillow` (PIL)
- **환경 변수**: `python-dotenv`
- **비동기 처리**: `asyncio`

---

**📝 다음 학습:** W1_003_Langchain_Components.md