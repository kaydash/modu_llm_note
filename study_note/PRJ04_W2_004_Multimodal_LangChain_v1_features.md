# LangChain v1.0 멀티모달 표준 콘텐츠 블록

## 학습 목표

이 실습을 완료하면 다음을 할 수 있습니다:

1. **표준 콘텐츠 블록 이해**: LangChain v1.0의 표준화된 멀티모달 콘텐츠 블록 시스템을 이해합니다
2. **content_blocks 활용**: `content_blocks` 속성을 사용하여 타입 안전한 멀티모달 처리를 구현합니다
3. **다양한 미디어 처리**: 이미지, 오디오, 비디오, PDF 등 여러 형태의 데이터를 통합 처리합니다
4. **표준 직렬화 구현**: `output_version="v1"` 설정으로 프로바이더 독립적인 응답 형식을 구현합니다
5. **추론 과정 추출**: `reasoning` 블록을 활용하여 모델의 사고 과정을 확인하고 분석합니다

---

## 핵심 개념

### 1. LangChain v1.0 멀티모달리티

**멀티모달리티(Multimodality)**는 텍스트, 이미지, 오디오, 비디오 등 다양한 형태의 데이터를 처리하는 기술입니다.

**LangChain의 주요 멀티모달 컴포넌트:**

| 컴포넌트 | 역할 | 예시 |
|---------|------|------|
| **채팅 모델** | 다양한 입출력 형식 처리 | 이미지 + 텍스트 → 텍스트 응답 |
| **임베딩 모델** | 다양한 데이터 타입의 벡터 표현 | CLIP으로 이미지-텍스트 임베딩 |
| **벡터 저장소** | 멀티모달 임베딩 검색 | 이미지 벡터로 유사 이미지 검색 |

### 2. 표준 콘텐츠 블록 (Standard Content Blocks)

LangChain은 세 가지 `content` 형식을 지원합니다:

| 형식 | 설명 | 장점 | 단점 |
|------|------|------|------|
| **문자열** | 단순 텍스트 메시지 | 간단함 | 멀티모달 불가 |
| **프로바이더 네이티브** | 프로바이더별 형식 | 프로바이더 최적화 | 이식성 낮음 |
| **표준 콘텐츠 블록** ✅ | LangChain 표준 형식 | 프로바이더 독립적 | 약간의 오버헤드 |

**표준 콘텐츠 블록을 사용하는 이유:**
- ✅ 프로바이더 간 전환이 쉬움 (OpenAI ↔ Gemini ↔ Claude)
- ✅ 타입 안전성 보장
- ✅ 일관된 API 인터페이스
- ✅ 미래 호환성

### 3. 지원하는 콘텐츠 블록 타입

**Core 블록:**

| 타입 | 설명 | 주요 필드 |
|------|------|-----------|
| `text` | 표준 텍스트 출력 | `text`, `annotations` |
| `reasoning` | 모델 추론 단계 | `reasoning` |

**Multimodal 블록:**

| 타입 | 설명 | 주요 필드 | 지원 모델 |
|------|------|-----------|----------|
| `image` | 이미지 데이터 | `url`, `base64`, `file_id`, `mime_type` | GPT-4o, Gemini, Claude |
| `audio` | 오디오 데이터 | `url`, `base64`, `file_id`, `mime_type` | Gemini, Whisper |
| `video` | 비디오 데이터 | `url`, `base64`, `file_id`, `mime_type` | Gemini |
| `file` | 일반 파일 (PDF 등) | `url`, `base64`, `file_id`, `mime_type` | GPT-4o, Claude |
| `text-plain` | 문서 텍스트 (.txt, .md) | `text`, `mime_type` | 모든 모델 |

**Tool Calling 블록:**

| 타입 | 설명 |
|------|------|
| `tool_call` | 함수 호출 |
| `tool_call_chunk` | 스트리밍 도구 호출 단편 |
| `invalid_tool_call` | 잘못된 도구 호출 |

### 4. 데이터 전달 방식 비교

| 방식 | 장점 | 단점 | 사용 시나리오 |
|------|------|------|--------------|
| **URL** | - 간단하고 직관적<br>- 대용량 파일에 유리<br>- 네트워크 트래픽 절약 | - 공개 URL 필요<br>- 네트워크 의존성 | 웹 이미지<br>공개 데이터셋 |
| **Base64** ✅ | - 가장 안정적<br>- URL 불필요<br>- 모든 프로바이더 지원<br>- 네트워크 독립적 | - 토큰 사용량 증가<br>- 페이로드 크기 증가 | 로컬 파일<br>비공개 데이터<br>프로덕션 환경 |
| **File ID** | - 토큰 효율적<br>- 재사용 가능 | - 프로바이더별 사전 업로드<br>- 제한적 지원 | 반복 사용<br>프로바이더 관리 파일 |

### 5. content vs content_blocks

```python
# 기본 동작 (output_version 미설정)
response.content         → 프로바이더 네이티브 형식 (예: 문자열)
response.content_blocks  → LangChain 표준 형식 (항상)

# 직렬화 활성화 (output_version="v1")
response.content         → LangChain 표준 형식 ✅
response.content_blocks  → LangChain 표준 형식 ✅

# 결과: 두 속성이 동일
response.content == response.content_blocks  # True
```

**직렬화가 필요한 경우:**
- API 엔드포인트 구현
- 프론트엔드와의 통신
- 멀티 프로바이더 지원
- 로깅/분석 시스템
- JSON 직렬화 필요

**직렬화가 불필요한 경우:**
- 내부 처리만 수행
- 단일 프로바이더만 사용
- 최대 성능 필요

---

## 환경 설정

### 필요한 라이브러리 설치

```bash
# 핵심 라이브러리
pip install langchain langchain-openai langchain-google-genai
pip install python-dotenv pillow requests matplotlib

# FastAPI (API 서버 구현 시)
pip install fastapi uvicorn

# 모니터링 (선택사항)
pip install langfuse
```

### 환경 변수 설정

`.env` 파일을 생성하고 API 키를 설정합니다:

```env
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
```

### 기본 import

```python
from dotenv import load_dotenv
load_dotenv()

import os
import base64
import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

import warnings
warnings.filterwarnings('ignore')
```

---

## 단계별 구현

### 1단계: 표준 콘텐츠 블록으로 이미지 처리 (URL 방식)

v1.0의 표준 `content_blocks` 형식을 사용합니다.

```python
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

# 모델 초기화 (init_chat_model 권장)
model = init_chat_model(
    "gpt-4o-mini",  # 멀티모달 지원 모델
    temperature=0
)

def process_image_from_url(image_url: str, prompt: str):
    """URL에서 직접 이미지 처리 (v1.0 표준 콘텐츠 블록)"""

    # content_blocks 사용 (표준 방식)
    message = HumanMessage(content_blocks=[
        {"type": "text", "text": prompt},
        {"type": "image", "url": image_url, "mime_type": "image/jpeg"}
    ])

    return model.invoke([message])

# 실행 예시
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
result = process_image_from_url(image_url, "이미지를 상세히 설명해주세요")
print(result.content)
```

**출력 예시:**
```
이미지에는 두 마리의 고양이가 소파 위에서 편안하게 잠을 자고 있는 모습이 담겨 있습니다.
소파는 밝은 핑크색으로, 고양이들은 서로 다른 자세로 누워 있습니다.
왼쪽에 있는 고양이는 길고 날씬한 몸매를 가지고 있으며,
오른쪽에 있는 고양이는 좀 더 통통한 체형으로 옆으로 누워 있습니다.
```

### 2단계: Base64 인코딩 방식으로 로컬 이미지 처리

로컬 파일을 안정적으로 처리하는 권장 방식입니다.

```python
import base64
import requests

def encode_image_to_base64(image_path: str) -> str:
    """로컬 이미지를 base64로 인코딩"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def encode_image_url_to_base64(image_url: str) -> str:
    """URL 이미지를 base64로 인코딩"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    response = requests.get(image_url, headers=headers)
    response.raise_for_status()
    return base64.b64encode(response.content).decode('utf-8')

def process_image_with_base64(image_source: str, prompt: str):
    """Base64로 이미지 처리 (v1.0 표준 콘텐츠 블록)"""

    # 이미지를 base64로 변환
    if image_source.startswith("http"):
        image_data = encode_image_url_to_base64(image_source)
    else:
        image_data = encode_image_to_base64(image_source)

    message = HumanMessage(content_blocks=[
        {"type": "text", "text": prompt},
        {
            "type": "image",
            "base64": image_data,
            "mime_type": "image/jpeg"
        }
    ])

    return model.invoke([message])

# 실행 예시
result = process_image_with_base64("portrait.jpg", "이 초상화의 특징을 분석해주세요")
print(result.content)
```

**출력 예시:**
```
이 초상화는 빈센트 반 고흐의 자화상으로, 여러 가지 특징이 있습니다.

1. 색채 사용: 강렬하고 대조적인 색상을 사용하여 감정을 표현합니다.
   배경의 푸른색과 인물의 따뜻한 피부톤 및 붉은 수염이 뚜렷한 대비를 이룹니다.

2. 붓질: 독특한 붓질이 잘 드러나며, 짧고 굵은 터치로 질감을 강조합니다.

3. 표정: 강렬하고 집중된 느낌을 주며, 고흐의 내면의 갈등을 반영합니다.
```

### 3단계: 여러 이미지 동시 처리

여러 이미지를 한 번에 분석하고 비교합니다.

```python
def process_multiple_images(image_sources: list, prompt: str):
    """여러 이미지를 동시에 처리 (v1.0 표준 콘텐츠 블록)"""

    # 콘텐츠 블록 리스트 시작 (프롬프트)
    content_blocks = [{"type": "text", "text": prompt}]

    # 각 이미지를 콘텐츠 블록에 추가
    for image_source in image_sources:
        if image_source.startswith("http"):
            # URL 처리
            image_data = encode_image_url_to_base64(image_source)
        else:
            # Base64 직접 처리
            image_data = encode_image_to_base64(image_source)

        content_blocks.append({
            "type": "image",
            "base64": image_data,
            "mime_type": "image/jpeg"
        })

    message = HumanMessage(content_blocks=content_blocks)
    return model.invoke([message])

# 실행 예시
images = [
    "portrait.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/600px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"
]

result = process_multiple_images(
    images,
    "이 두 작품을 비교 분석해주세요. 화풍, 색채, 구도의 차이점을 중심으로 설명해주세요."
)
print(result.content)
```

**출력 예시:**
```
두 작품은 빈센트 반 고흐의 대표작입니다.

화풍:
- 자화상: 강렬한 붓질과 점묘법으로 감정을 표현
- 별이 빛나는 밤: 유동적이고 역동적인 화풍, 소용돌이치는 구름

색채:
- 자화상: 따뜻한 색조 (붉은 머리, 어두운 외투)와 차가운 배경의 대비
- 별이 빛나는 밤: 차가운 파란색과 밝은 노란색의 강렬한 대비

구도:
- 자화상: 인물 중심, 중앙 배치로 감정에 집중
- 별이 빛나는 밤: 복잡한 구도, 하늘과 땅의 조화, 깊이감 제공
```

### 4단계: 다양한 멀티모달 타입 처리

PDF, 오디오, 비디오 등 다양한 미디어를 처리합니다.

**PDF 문서 처리:**

```python
def process_pdf_document(pdf_source: str, prompt: str):
    """PDF 문서 처리 (v1.0 표준 콘텐츠 블록)"""

    if pdf_source.startswith("http"):
        # URL 방식
        content_blocks = [
            {"type": "text", "text": prompt},
            {"type": "file", "url": pdf_source, "mime_type": "application/pdf"}
        ]
    else:
        # Base64 방식
        with open(pdf_source, "rb") as pdf_file:
            pdf_data = base64.b64encode(pdf_file.read()).decode("utf-8")

        content_blocks = [
            {"type": "text", "text": prompt},
            {
                "type": "file",
                "base64": pdf_data,
                "mime_type": "application/pdf",
                "extras": {"filename": "document.pdf"}
            }
        ]

    message = HumanMessage(content_blocks=content_blocks)
    return model.invoke([message])

# 실행 예시
result = process_pdf_document(
    "data/report.pdf",
    "이 문서의 주요 내용을 요약해주세요"
)
print(result.content)
```

**오디오 처리:**

```python
from langchain.chat_models import init_chat_model

# 오디오 지원 모델 (Gemini)
gemini_model = init_chat_model(
    "google_genai:gemini-2.0-flash-exp",
    temperature=0,
)

def process_audio(audio_source: str, prompt: str = "이 오디오를 텍스트로 변환해주세요"):
    """오디오 처리 (v1.0 표준 콘텐츠 블록)"""

    if audio_source.startswith("http"):
        content_blocks = [
            {"type": "text", "text": prompt},
            {"type": "audio", "url": audio_source, "mime_type": "audio/mpeg"}
        ]
    else:
        with open(audio_source, "rb") as audio_file:
            audio_data = base64.b64encode(audio_file.read()).decode("utf-8")
        content_blocks = [
            {"type": "text", "text": prompt},
            {
                "type": "audio",
                "base64": audio_data,
                "mime_type": "audio/mpeg"
            }
        ]

    message = HumanMessage(content_blocks=content_blocks)
    return gemini_model.invoke([message])

# 실행 예시
result = process_audio("data/meeting_sample.mp3")
print(result.content)
```

**비디오 처리:**

```python
def process_video(video_source: str, prompt: str):
    """비디오 처리 (v1.0 표준 콘텐츠 블록)"""

    if video_source.startswith("http"):
        content_blocks = [
            {"type": "text", "text": prompt},
            {"type": "video", "url": video_source, "mime_type": "video/mp4"}
        ]
    else:
        with open(video_source, "rb") as video_file:
            video_data = base64.b64encode(video_file.read()).decode("utf-8")

        content_blocks = [
            {"type": "text", "text": prompt},
            {
                "type": "video",
                "base64": video_data,
                "mime_type": "video/mp4"
            }
        ]

    message = HumanMessage(content_blocks=content_blocks)
    return gemini_model.invoke([message])

# 실행 예시
result = process_video(
    "data/bikes.mp4",
    "이 비디오에서 일어나는 주요 이벤트를 시간순으로 설명해주세요"
)
print(result.content)
```

### 5단계: 표준 콘텐츠 직렬화와 API 서버 구현

`output_version="v1"`을 설정하여 표준 형식으로 응답을 받습니다.

```python
from langchain.chat_models import init_chat_model

# 기본 동작 (직렬화 없음)
model_basic = init_chat_model("gpt-4o-mini")
response = model_basic.invoke("AI를 설명해주세요")
print("content:", type(response.content))  # <class 'str'>
print("content_blocks:", type(response.content_blocks))  # <class 'list'>

# 직렬화 활성화
model_serialized = init_chat_model("gpt-4o-mini", output_version="v1")
response = model_serialized.invoke("AI를 설명해주세요")
print("content:", type(response.content))  # <class 'list'> ✅
print("content_blocks:", type(response.content_blocks))  # <class 'list'>

# 두 속성이 동일
print(response.content == response.content_blocks)  # True
```

**FastAPI 서버 구현:**

```python
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chat_models import init_chat_model

app = FastAPI()

# 표준 직렬화 활성화
model = init_chat_model("gpt-4o-mini", output_version="v1")

@app.post("/query")
async def query(text: str):
    response = model.invoke(text)

    return {
        "content": response.content,  # ✅ 표준 형식 보장
        "model": "gpt-4o-mini",
        "tokens": response.usage_metadata["total_tokens"]
    }

# 서버 실행: uvicorn fastapi_example:app --reload
```

**클라이언트 테스트:**

```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    params={"text": "인공지능에 대해 간단히 설명해주세요"}
)

print("Status Code:", response.status_code)
print("Response:", response.json())
```

**출력 예시:**
```json
{
  "content": [
    {
      "type": "text",
      "text": "AI는 컴퓨터 시스템이 인간의 지능을 모방하여 학습, 추론, 문제 해결 등을 수행하는 기술입니다..."
    }
  ],
  "model": "gpt-4o-mini",
  "tokens": 221
}
```

### 6단계: content_blocks로 추론 과정 추출

Google Gemini의 `reasoning` 블록을 활용합니다.

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    include_thoughts=True,  # reasoning 블록 포함
)

response = llm.invoke("How many 'r's are in the word 'strawberry'?")

# content_blocks를 순회하며 텍스트와 추론 분리
for block in response.content_blocks:
    if block["type"] == "text":
        print(f"✅ 답변: {block['text']}")

    elif block["type"] == "reasoning":
        print(f"🧠 추론 과정:\n{block['reasoning']}")
```

**출력 예시:**
```
🧠 추론 과정:
**Calculating the 'R' Count in "Strawberry"**

Okay, so I see the request. It's a pretty straightforward query: count the occurrences
of the letter 'r' in the word "strawberry." Let me break this down systematically.

First, I need to isolate the target letter, which is 'r'. Then, I scan through the word:
s-t-r (one), a-w-b-e-r (two), r (three), y.

Three 'r's! The word "strawberry" contains three instances of the letter 'r'.

✅ 답변: There are **3** 'r's in the word 'strawberry'.
```

---

## 실습 문제

### 문제 1: 표준 콘텐츠 블록으로 이미지 분석 파이프라인 구축 (기본)

**난이도**: ⭐⭐☆☆☆

**문제**:
LangChain v1.0의 표준 `content_blocks` 형식을 사용하여 이미지 분석 파이프라인을 구축하세요.

**요구사항**:
1. 로컬 이미지 또는 URL 이미지를 모두 지원
2. Base64 인코딩 방식 사용
3. 이미지의 주요 특징 3가지 이상 추출
4. 결과를 구조화된 JSON 형식으로 반환

**입력 예시**:
```python
image_path = "sample_image.jpg"
```

**출력 예시**:
```json
{
  "image_source": "sample_image.jpg",
  "features": [
    "주요 객체: 고양이 2마리",
    "색상 톤: 따뜻한 핑크와 베이지",
    "분위기: 평화롭고 아늑함"
  ],
  "content_blocks_used": true
}
```

### 문제 2: 멀티 미디어 타입 통합 처리 시스템 (중급)

**난이도**: ⭐⭐⭐☆☆

**문제**:
이미지, PDF, 오디오를 모두 처리할 수 있는 통합 멀티미디어 분석 시스템을 구현하세요.

**요구사항**:
1. 파일 확장자를 자동으로 감지하여 적절한 `mime_type` 설정
2. 각 미디어 타입에 최적화된 프롬프트 자동 생성
3. `content_blocks` 표준 형식 사용
4. 처리 결과를 통합 데이터베이스 형식으로 저장

**지원 파일 형식**:
- 이미지: `.jpg`, `.jpeg`, `.png`
- 문서: `.pdf`
- 오디오: `.mp3`, `.wav`

**출력 예시**:
```json
{
  "file_name": "report.pdf",
  "file_type": "document",
  "mime_type": "application/pdf",
  "analysis": "이 문서는 2024년 분기별 실적을 요약합니다...",
  "processed_at": "2025-06-01T10:30:00",
  "processing_method": "content_blocks"
}
```

### 문제 3: 표준 직렬화 기반 멀티 프로바이더 API 서버 (고급)

**난이도**: ⭐⭐⭐⭐☆

**문제**:
`output_version="v1"` 직렬화를 활용하여 여러 LLM 프로바이더를 지원하는 API 서버를 구현하세요.

**요구사항**:
1. OpenAI, Google Gemini, Anthropic Claude 지원
2. 사용자가 프로바이더를 선택할 수 있는 엔드포인트
3. 모든 응답을 표준 `content_blocks` 형식으로 반환
4. `reasoning` 블록이 있으면 추가로 추출하여 반환
5. 에러 처리 및 로깅

**API 엔드포인트**:
```
POST /analyze
{
  "provider": "openai|gemini|claude",
  "media_type": "image|audio|pdf",
  "media_source": "파일 경로 또는 URL",
  "prompt": "분석 요청"
}
```

**응답 예시**:
```json
{
  "provider": "gemini",
  "content_blocks": [
    {
      "type": "reasoning",
      "reasoning": "이미지를 분석하기 위해..."
    },
    {
      "type": "text",
      "text": "이 이미지에는..."
    }
  ],
  "standard_format": true,
  "tokens_used": 450
}
```

---

## 솔루션 예시

### 문제 1 솔루션: 표준 콘텐츠 블록 이미지 분석 파이프라인

```python
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
import base64
import json
import os

class StandardImageAnalyzer:
    """LangChain v1.0 표준 콘텐츠 블록을 사용하는 이미지 분석기"""

    def __init__(self, model_name="gpt-4o-mini"):
        self.model = init_chat_model(model_name, temperature=0)

    def encode_image_to_base64(self, image_path: str) -> str:
        """로컬 이미지를 base64로 인코딩"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def encode_url_to_base64(self, image_url: str) -> str:
        """URL 이미지를 base64로 인코딩"""
        import requests
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(image_url, headers=headers)
        response.raise_for_status()
        return base64.b64encode(response.content).decode("utf-8")

    def analyze_image(self, image_source: str) -> dict:
        """이미지를 분석하여 구조화된 정보 반환"""

        # 이미지 소스 타입 판별
        if image_source.startswith("http"):
            image_data = self.encode_url_to_base64(image_source)
            source_type = "url"
        else:
            if not os.path.exists(image_source):
                return {"error": f"파일을 찾을 수 없습니다: {image_source}"}
            image_data = self.encode_image_to_base64(image_source)
            source_type = "local_file"

        # 표준 콘텐츠 블록 형식으로 메시지 생성
        prompt = """이 이미지를 분석하여 JSON 형식으로 반환하세요:
        {
            "main_objects": ["주요 객체 목록"],
            "color_tone": "색상 톤 설명",
            "mood": "분위기 설명",
            "additional_features": ["기타 특징"]
        }

        반드시 유효한 JSON 형식으로만 응답하세요."""

        message = HumanMessage(content_blocks=[
            {"type": "text", "text": prompt},
            {
                "type": "image",
                "base64": image_data,
                "mime_type": "image/jpeg"
            }
        ])

        # 모델 호출
        response = self.model.invoke([message])

        # JSON 파싱
        try:
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]

            analysis = json.loads(content.strip())

            # 결과 구조화
            return {
                "image_source": image_source,
                "source_type": source_type,
                "features": [
                    f"주요 객체: {', '.join(analysis.get('main_objects', []))}",
                    f"색상 톤: {analysis.get('color_tone', 'N/A')}",
                    f"분위기: {analysis.get('mood', 'N/A')}"
                ],
                "additional_features": analysis.get('additional_features', []),
                "content_blocks_used": True,
                "raw_analysis": analysis
            }

        except json.JSONDecodeError:
            return {
                "image_source": image_source,
                "source_type": source_type,
                "features": ["분석 실패"],
                "content_blocks_used": True,
                "error": "JSON 파싱 실패",
                "raw_response": response.content
            }

# 사용 예시
analyzer = StandardImageAnalyzer()

# 로컬 이미지 분석
result1 = analyzer.analyze_image("portrait.jpg")
print("=== 로컬 이미지 분석 결과 ===")
print(json.dumps(result1, indent=2, ensure_ascii=False))

# URL 이미지 분석
result2 = analyzer.analyze_image("http://images.cocodataset.org/val2017/000000039769.jpg")
print("\n=== URL 이미지 분석 결과 ===")
print(json.dumps(result2, indent=2, ensure_ascii=False))
```

**출력**:
```json
{
  "image_source": "portrait.jpg",
  "source_type": "local_file",
  "features": [
    "주요 객체: 사람, 초상화, 예술작품",
    "색상 톤: 따뜻한 오렌지와 푸른 배경의 대비",
    "분위기: 진지하고 사색적인"
  ],
  "additional_features": [
    "후기 인상주의 화풍",
    "강렬한 붓터치",
    "자화상"
  ],
  "content_blocks_used": true
}
```

### 문제 2 솔루션: 멀티 미디어 통합 처리 시스템

```python
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage
import base64
import os
from datetime import datetime
from typing import Literal
import mimetypes

class MultiMediaProcessor:
    """다양한 미디어 타입을 통합 처리하는 시스템"""

    def __init__(self):
        # 이미지/PDF용 모델
        self.vision_model = init_chat_model("gpt-4o-mini", temperature=0)
        # 오디오/비디오용 모델
        self.gemini_model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0
        )

        # 파일 확장자별 MIME 타입 매핑
        self.mime_mapping = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.pdf': 'application/pdf',
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.mp4': 'video/mp4'
        }

        # 미디어 타입별 프롬프트 템플릿
        self.prompt_templates = {
            'image': "이 이미지의 주요 내용을 자세히 설명해주세요. 객체, 색상, 구도를 포함해주세요.",
            'document': "이 문서의 주요 내용을 요약해주세요. 핵심 포인트와 주요 정보를 중심으로 설명해주세요.",
            'audio': "이 오디오 내용을 텍스트로 변환하고 요약해주세요.",
            'video': "이 비디오에서 일어나는 주요 이벤트를 시간순으로 설명해주세요."
        }

    def detect_media_type(self, file_path: str) -> tuple:
        """파일 확장자로부터 미디어 타입과 MIME 타입 감지"""
        _, ext = os.path.splitext(file_path.lower())

        mime_type = self.mime_mapping.get(ext)

        if not mime_type:
            # mimetypes 모듈로 fallback
            mime_type, _ = mimetypes.guess_type(file_path)

        if ext in ['.jpg', '.jpeg', '.png']:
            media_type = 'image'
        elif ext == '.pdf':
            media_type = 'document'
        elif ext in ['.mp3', '.wav']:
            media_type = 'audio'
        elif ext == '.mp4':
            media_type = 'video'
        else:
            media_type = 'unknown'

        return media_type, mime_type

    def encode_file_to_base64(self, file_path: str) -> str:
        """파일을 base64로 인코딩"""
        with open(file_path, "rb") as file:
            return base64.b64encode(file.read()).decode("utf-8")

    def process_file(self, file_path: str, custom_prompt: str = None) -> dict:
        """파일을 분석하여 통합 데이터베이스 형식으로 반환"""

        # 파일 존재 확인
        if not os.path.exists(file_path):
            return {"error": f"파일을 찾을 수 없습니다: {file_path}"}

        # 미디어 타입 감지
        media_type, mime_type = self.detect_media_type(file_path)

        if media_type == 'unknown':
            return {"error": f"지원하지 않는 파일 형식입니다: {file_path}"}

        # Base64 인코딩
        file_data = self.encode_file_to_base64(file_path)

        # 프롬프트 선택
        prompt = custom_prompt or self.prompt_templates[media_type]

        # 콘텐츠 블록 타입 결정
        if media_type == 'image':
            block_type = 'image'
            model = self.vision_model
        elif media_type == 'document':
            block_type = 'file'
            model = self.vision_model
        elif media_type in ['audio', 'video']:
            block_type = media_type
            model = self.gemini_model

        # 표준 콘텐츠 블록으로 메시지 생성
        content_blocks = [
            {"type": "text", "text": prompt},
            {
                "type": block_type,
                "base64": file_data,
                "mime_type": mime_type
            }
        ]

        if media_type == 'document':
            # PDF의 경우 파일명 추가
            content_blocks[1]["extras"] = {"filename": os.path.basename(file_path)}

        message = HumanMessage(content_blocks=content_blocks)

        # 모델 호출
        try:
            response = model.invoke([message])

            # 통합 데이터베이스 형식으로 결과 구조화
            return {
                "file_name": os.path.basename(file_path),
                "file_path": file_path,
                "file_type": media_type,
                "mime_type": mime_type,
                "analysis": response.content,
                "processed_at": datetime.now().isoformat(),
                "processing_method": "content_blocks",
                "success": True
            }

        except Exception as e:
            return {
                "file_name": os.path.basename(file_path),
                "file_path": file_path,
                "file_type": media_type,
                "mime_type": mime_type,
                "error": str(e),
                "processed_at": datetime.now().isoformat(),
                "success": False
            }

    def batch_process(self, file_paths: list) -> list:
        """여러 파일을 일괄 처리"""
        results = []

        print(f"총 {len(file_paths)}개 파일 처리 시작...\n")

        for idx, file_path in enumerate(file_paths, 1):
            print(f"[{idx}/{len(file_paths)}] {file_path} 처리 중...")
            result = self.process_file(file_path)
            results.append(result)

            if result.get('success'):
                print(f"  ✅ 성공: {result['file_type']}")
            else:
                print(f"  ❌ 실패: {result.get('error', 'Unknown error')}")

        return results

# 사용 예시
processor = MultiMediaProcessor()

# 단일 파일 처리
result = processor.process_file("portrait.jpg")
print("=== 단일 파일 처리 결과 ===")
print(json.dumps(result, indent=2, ensure_ascii=False))

# 일괄 처리
files = [
    "portrait.jpg",
    "data/report.pdf",
    "data/meeting_sample.mp3"
]

batch_results = processor.batch_process(files)

# 결과 저장
with open("processing_results.json", 'w', encoding='utf-8') as f:
    json.dump(batch_results, f, indent=2, ensure_ascii=False)

print("\n✅ 일괄 처리 완료. 결과가 processing_results.json에 저장되었습니다.")
```

### 문제 3 솔루션: 멀티 프로바이더 API 서버

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage
import base64
import os
from typing import Literal, Optional
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Provider Multimodal API")

# 요청 모델
class AnalyzeRequest(BaseModel):
    provider: Literal["openai", "gemini", "claude"]
    media_type: Literal["image", "audio", "pdf"]
    media_source: str
    prompt: str

# 응답 모델
class AnalyzeResponse(BaseModel):
    provider: str
    content_blocks: list
    standard_format: bool
    tokens_used: int
    success: bool
    error: Optional[str] = None

class MultiProviderAnalyzer:
    """멀티 프로바이더 분석기"""

    def __init__(self):
        # 프로바이더별 모델 초기화 (표준 직렬화 활성화)
        self.models = {
            "openai": init_chat_model("gpt-4o-mini", output_version="v1", temperature=0),
            "claude": init_chat_model("claude-3-5-sonnet-20241022", output_version="v1", temperature=0),
            "gemini": ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                include_thoughts=True,
                temperature=0
            )
        }

        # MIME 타입 매핑
        self.mime_types = {
            "image": "image/jpeg",
            "audio": "audio/mpeg",
            "pdf": "application/pdf"
        }

    def encode_file(self, file_path: str) -> str:
        """파일을 base64로 인코딩"""
        with open(file_path, "rb") as file:
            return base64.b64encode(file.read()).decode("utf-8")

    def encode_url(self, url: str) -> str:
        """URL을 base64로 인코딩"""
        import requests
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return base64.b64encode(response.content).decode("utf-8")

    def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        """미디어 분석 실행"""

        try:
            # 모델 선택
            model = self.models.get(request.provider)
            if not model:
                raise HTTPException(status_code=400, detail=f"지원하지 않는 프로바이더: {request.provider}")

            # 미디어 소스 인코딩
            if request.media_source.startswith("http"):
                media_data = self.encode_url(request.media_source)
            else:
                if not os.path.exists(request.media_source):
                    raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없습니다: {request.media_source}")
                media_data = self.encode_file(request.media_source)

            # 콘텐츠 블록 타입 결정
            block_type_map = {
                "image": "image",
                "audio": "audio",
                "pdf": "file"
            }
            block_type = block_type_map[request.media_type]

            # 표준 콘텐츠 블록 생성
            content_blocks = [
                {"type": "text", "text": request.prompt},
                {
                    "type": block_type,
                    "base64": media_data,
                    "mime_type": self.mime_types[request.media_type]
                }
            ]

            if request.media_type == "pdf":
                content_blocks[1]["extras"] = {"filename": "document.pdf"}

            message = HumanMessage(content_blocks=content_blocks)

            # 모델 호출
            response = model.invoke([message])

            # 토큰 사용량 추출
            tokens_used = response.usage_metadata.get("total_tokens", 0) if hasattr(response, 'usage_metadata') else 0

            # content_blocks 추출 (표준 형식)
            result_blocks = response.content_blocks if hasattr(response, 'content_blocks') else []

            logger.info(f"성공: {request.provider} - {request.media_type} - {tokens_used} tokens")

            return AnalyzeResponse(
                provider=request.provider,
                content_blocks=result_blocks,
                standard_format=True,
                tokens_used=tokens_used,
                success=True
            )

        except Exception as e:
            logger.error(f"오류: {request.provider} - {str(e)}")
            return AnalyzeResponse(
                provider=request.provider,
                content_blocks=[],
                standard_format=False,
                tokens_used=0,
                success=False,
                error=str(e)
            )

# 전역 분석기 인스턴스
analyzer = MultiProviderAnalyzer()

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_media(request: AnalyzeRequest):
    """미디어 분석 엔드포인트"""
    return analyzer.analyze(request)

@app.get("/providers")
async def list_providers():
    """지원하는 프로바이더 목록"""
    return {
        "providers": ["openai", "gemini", "claude"],
        "media_types": ["image", "audio", "pdf"],
        "standard_format": "v1.0 content_blocks"
    }

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {"status": "healthy", "version": "1.0.0"}

# 서버 실행: uvicorn solution_3:app --reload --port 8000
```

**클라이언트 테스트:**

```python
import requests
import json

API_URL = "http://localhost:8000"

# 이미지 분석 (OpenAI)
request_data = {
    "provider": "openai",
    "media_type": "image",
    "media_source": "portrait.jpg",
    "prompt": "이 이미지를 자세히 분석해주세요"
}

response = requests.post(f"{API_URL}/analyze", json=request_data)
result = response.json()

print("=== OpenAI 이미지 분석 결과 ===")
print(json.dumps(result, indent=2, ensure_ascii=False))

# 이미지 분석 (Gemini with reasoning)
request_data["provider"] = "gemini"
response = requests.post(f"{API_URL}/analyze", json=request_data)
result = response.json()

print("\n=== Gemini 이미지 분석 결과 (reasoning 포함) ===")
for block in result["content_blocks"]:
    if block["type"] == "reasoning":
        print(f"🧠 추론: {block['reasoning'][:200]}...")
    elif block["type"] == "text":
        print(f"✅ 답변: {block['text'][:200]}...")

# 지원 프로바이더 확인
providers = requests.get(f"{API_URL}/providers").json()
print("\n=== 지원하는 기능 ===")
print(json.dumps(providers, indent=2))
```

---

## 실무 활용 예시

### 예시 1: 멀티모달 콘텐츠 관리 시스템

다양한 미디어 파일을 자동으로 분류하고 메타데이터를 생성하는 시스템입니다.

```python
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
import base64
import os
import json
from datetime import datetime
from typing import List, Dict
import hashlib

class ContentManagementSystem:
    """멀티모달 콘텐츠 관리 시스템"""

    def __init__(self, model_name="gpt-4o-mini"):
        # 표준 직렬화 활성화
        self.model = init_chat_model(model_name, output_version="v1", temperature=0)
        self.metadata_db = {}  # 간단한 메모리 DB

    def calculate_file_hash(self, file_path: str) -> str:
        """파일 해시 계산 (중복 방지)"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            hasher.update(f.read())
        return hasher.hexdigest()

    def encode_file_to_base64(self, file_path: str) -> str:
        """파일을 base64로 인코딩"""
        with open(file_path, "rb") as file:
            return base64.b64encode(file.read()).decode("utf-8")

    def generate_metadata(self, file_path: str) -> Dict:
        """파일의 메타데이터 자동 생성"""

        # 파일 기본 정보
        file_stat = os.stat(file_path)
        file_hash = self.calculate_file_hash(file_path)
        _, ext = os.path.splitext(file_path)

        # 중복 확인
        if file_hash in self.metadata_db:
            print(f"⚠️ 중복 파일: {file_path}")
            return self.metadata_db[file_hash]

        # 파일 인코딩
        file_data = self.encode_file_to_base64(file_path)

        # 미디어 타입 결정
        mime_mapping = {
            '.jpg': ('image', 'image/jpeg'),
            '.jpeg': ('image', 'image/jpeg'),
            '.png': ('image', 'image/png'),
            '.pdf': ('file', 'application/pdf')
        }

        media_type, mime_type = mime_mapping.get(ext.lower(), ('unknown', 'application/octet-stream'))

        if media_type == 'unknown':
            return {"error": "지원하지 않는 파일 형식"}

        # AI로 콘텐츠 분석
        prompt = """이 파일을 분석하여 JSON 형식으로 메타데이터를 생성하세요:
        {
            "title": "간단한 제목 (20자 이내)",
            "description": "상세 설명 (100자 이내)",
            "tags": ["태그1", "태그2", "태그3"],
            "category": "카테고리",
            "keywords": ["키워드1", "키워드2"]
        }

        반드시 유효한 JSON으로만 응답하세요."""

        message = HumanMessage(content_blocks=[
            {"type": "text", "text": prompt},
            {
                "type": media_type,
                "base64": file_data,
                "mime_type": mime_type
            }
        ])

        response = self.model.invoke([message])

        # JSON 파싱
        try:
            # content는 표준 형식 (list)
            content_text = response.content[0]['text'] if isinstance(response.content, list) else response.content

            if content_text.startswith("```json"):
                content_text = content_text[7:-3]
            elif content_text.startswith("```"):
                content_text = content_text[3:-3]

            ai_metadata = json.loads(content_text.strip())

            # 최종 메타데이터 통합
            metadata = {
                "file_id": file_hash,
                "file_name": os.path.basename(file_path),
                "file_path": file_path,
                "file_size": file_stat.st_size,
                "file_type": media_type,
                "mime_type": mime_type,
                "created_at": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                "indexed_at": datetime.now().isoformat(),
                **ai_metadata,
                "standard_format": True
            }

            # 메모리 DB에 저장
            self.metadata_db[file_hash] = metadata

            return metadata

        except json.JSONDecodeError:
            return {
                "error": "메타데이터 생성 실패",
                "raw_response": response.content
            }

    def index_directory(self, directory_path: str, pattern: str = "*.*") -> List[Dict]:
        """디렉토리 내 모든 파일 인덱싱"""
        from glob import glob

        file_paths = glob(os.path.join(directory_path, pattern))
        results = []

        print(f"📂 {directory_path} 디렉토리 인덱싱 시작...")
        print(f"   총 {len(file_paths)}개 파일 발견\n")

        for idx, file_path in enumerate(file_paths, 1):
            print(f"[{idx}/{len(file_paths)}] {os.path.basename(file_path)} 처리 중...")
            metadata = self.generate_metadata(file_path)
            results.append(metadata)

            if "error" not in metadata:
                print(f"  ✅ 제목: {metadata.get('title', 'N/A')}")
                print(f"  🏷️ 태그: {', '.join(metadata.get('tags', []))}")
            else:
                print(f"  ❌ 오류: {metadata['error']}")

        return results

    def search_by_tag(self, tag: str) -> List[Dict]:
        """태그로 콘텐츠 검색"""
        results = []
        for file_hash, metadata in self.metadata_db.items():
            if tag in metadata.get('tags', []):
                results.append(metadata)
        return results

    def export_metadata(self, output_path: str):
        """메타데이터를 JSON 파일로 내보내기"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(list(self.metadata_db.values()), f, indent=2, ensure_ascii=False)
        print(f"✅ 메타데이터가 {output_path}에 저장되었습니다.")

# 사용 예시
cms = ContentManagementSystem()

# 디렉토리 인덱싱
results = cms.index_directory("./content_library/")

# 태그로 검색
art_content = cms.search_by_tag("예술")
print(f"\n🎨 '예술' 태그로 검색된 콘텐츠: {len(art_content)}개")

# 메타데이터 내보내기
cms.export_metadata("content_metadata.json")
```

**특징**:
- 파일 해시로 중복 방지
- AI 자동 메타데이터 생성
- 태그 기반 검색
- 표준 직렬화로 일관된 출력

### 예시 2: 멀티모달 교육 콘텐츠 분석 플랫폼

교육 자료(이미지, PDF, 비디오)를 분석하여 학습 효과를 평가하는 시스템입니다.

```python
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage
import base64
import os
from datetime import datetime
from typing import Dict, List
import json

class EducationalContentAnalyzer:
    """교육 콘텐츠 분석 플랫폼"""

    def __init__(self):
        # 이미지/PDF용 모델
        self.vision_model = init_chat_model("gpt-4o-mini", output_version="v1", temperature=0)
        # 비디오용 모델 (reasoning 포함)
        self.video_model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            include_thoughts=True,
            temperature=0
        )

        # 평가 기준
        self.evaluation_criteria = {
            "clarity": "내용의 명확성 (1-5)",
            "engagement": "학습 참여도 (1-5)",
            "completeness": "내용 완성도 (1-5)",
            "accessibility": "접근성 (1-5)",
            "pedagogical_value": "교육적 가치 (1-5)"
        }

    def encode_file(self, file_path: str) -> str:
        """파일을 base64로 인코딩"""
        with open(file_path, "rb") as file:
            return base64.b64encode(file.read()).decode("utf-8")

    def analyze_content(self, file_path: str, content_type: str, target_audience: str = "일반") -> Dict:
        """교육 콘텐츠 분석 및 평가"""

        if not os.path.exists(file_path):
            return {"error": f"파일을 찾을 수 없습니다: {file_path}"}

        # 파일 인코딩
        file_data = self.encode_file(file_path)

        # 미디어 타입 결정
        _, ext = os.path.splitext(file_path)
        mime_mapping = {
            '.jpg': ('image', 'image/jpeg', self.vision_model),
            '.jpeg': ('image', 'image/jpeg', self.vision_model),
            '.png': ('image', 'image/png', self.vision_model),
            '.pdf': ('file', 'application/pdf', self.vision_model),
            '.mp4': ('video', 'video/mp4', self.video_model)
        }

        block_type, mime_type, model = mime_mapping.get(ext.lower(), (None, None, None))

        if not block_type:
            return {"error": "지원하지 않는 파일 형식"}

        # 평가 프롬프트
        criteria_desc = "\n".join([f"- {key}: {desc}" for key, desc in self.evaluation_criteria.items()])

        prompt = f"""이 교육 콘텐츠를 분석하고 평가해주세요.

대상 학습자: {target_audience}
콘텐츠 타입: {content_type}

다음 기준으로 평가하여 JSON 형식으로 반환하세요:
{criteria_desc}

{{
    "summary": "콘텐츠 요약 (100자 이내)",
    "learning_objectives": ["학습 목표1", "학습 목표2"],
    "key_concepts": ["핵심 개념1", "핵심 개념2"],
    "evaluation": {{
        "clarity": 점수,
        "engagement": 점수,
        "completeness": 점수,
        "accessibility": 점수,
        "pedagogical_value": 점수
    }},
    "strengths": ["강점1", "강점2"],
    "improvements": ["개선점1", "개선점2"],
    "overall_score": 평균 점수,
    "recommendation": "종합 의견"
}}

반드시 유효한 JSON으로만 응답하세요."""

        # 콘텐츠 블록 생성
        content_blocks = [
            {"type": "text", "text": prompt},
            {
                "type": block_type,
                "base64": file_data,
                "mime_type": mime_type
            }
        ]

        if block_type == "file":
            content_blocks[1]["extras"] = {"filename": os.path.basename(file_path)}

        message = HumanMessage(content_blocks=content_blocks)

        # 모델 호출
        try:
            response = model.invoke([message])

            # 추론 과정 추출 (Gemini의 경우)
            reasoning_text = None
            if hasattr(response, 'content_blocks'):
                for block in response.content_blocks:
                    if block.get("type") == "reasoning":
                        reasoning_text = block.get("reasoning")
                        break

            # JSON 파싱
            content_text = response.content[0]['text'] if isinstance(response.content, list) else response.content

            if content_text.startswith("```json"):
                content_text = content_text[7:-3]
            elif content_text.startswith("```"):
                content_text = content_text[3:-3]

            evaluation = json.loads(content_text.strip())

            # 최종 결과
            result = {
                "file_name": os.path.basename(file_path),
                "file_path": file_path,
                "content_type": content_type,
                "target_audience": target_audience,
                "analyzed_at": datetime.now().isoformat(),
                **evaluation,
                "standard_format": True
            }

            if reasoning_text:
                result["reasoning_process"] = reasoning_text

            return result

        except Exception as e:
            return {
                "error": str(e),
                "file_path": file_path
            }

    def batch_analyze(self, files: List[tuple]) -> List[Dict]:
        """여러 콘텐츠 일괄 분석

        Args:
            files: [(file_path, content_type, target_audience), ...]
        """
        results = []

        print(f"📚 총 {len(files)}개 교육 콘텐츠 분석 시작...\n")

        for idx, (file_path, content_type, target_audience) in enumerate(files, 1):
            print(f"[{idx}/{len(files)}] {os.path.basename(file_path)} 분석 중...")
            result = self.analyze_content(file_path, content_type, target_audience)
            results.append(result)

            if "error" not in result:
                print(f"  ✅ 종합 점수: {result.get('overall_score', 'N/A')}/5")
                print(f"  📝 요약: {result.get('summary', 'N/A')[:50]}...")
            else:
                print(f"  ❌ 오류: {result['error']}")

        return results

    def generate_report(self, results: List[Dict], output_path: str):
        """분석 결과 리포트 생성"""

        report = f"""
{'='*80}
교육 콘텐츠 분석 리포트
{'='*80}

분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
총 콘텐츠 수: {len(results)}

"""

        for idx, result in enumerate(results, 1):
            if "error" in result:
                report += f"\n[{idx}] {result.get('file_name', 'N/A')} - 분석 실패\n"
                continue

            report += f"""
{'='*80}
[{idx}] {result['file_name']}
{'='*80}

콘텐츠 타입: {result['content_type']}
대상 학습자: {result['target_audience']}

📝 요약: {result.get('summary', 'N/A')}

🎯 학습 목표:
"""
            for obj in result.get('learning_objectives', []):
                report += f"  - {obj}\n"

            report += f"\n💡 핵심 개념:\n"
            for concept in result.get('key_concepts', []):
                report += f"  - {concept}\n"

            report += f"\n📊 평가 점수:\n"
            evaluation = result.get('evaluation', {})
            for criterion, score in evaluation.items():
                report += f"  - {criterion}: {score}/5\n"

            report += f"\n⭐ 종합 점수: {result.get('overall_score', 'N/A')}/5\n"

            report += f"\n✅ 강점:\n"
            for strength in result.get('strengths', []):
                report += f"  - {strength}\n"

            report += f"\n🔄 개선점:\n"
            for improvement in result.get('improvements', []):
                report += f"  - {improvement}\n"

            report += f"\n💬 종합 의견:\n  {result.get('recommendation', 'N/A')}\n"

        # 통계
        successful = [r for r in results if "error" not in r]
        if successful:
            avg_score = sum(r.get('overall_score', 0) for r in successful) / len(successful)
            report += f"""
{'='*80}
통계 요약
{'='*80}

분석 성공: {len(successful)}/{len(results)}
평균 종합 점수: {avg_score:.2f}/5
"""

        # 파일 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n✅ 리포트가 {output_path}에 저장되었습니다.")

# 사용 예시
analyzer = EducationalContentAnalyzer()

# 교육 콘텐츠 목록
educational_files = [
    ("course_intro.jpg", "강의 소개 이미지", "대학생"),
    ("lecture_slides.pdf", "강의 자료", "대학생"),
    ("demo_video.mp4", "실습 데모 영상", "대학생")
]

# 일괄 분석
results = analyzer.batch_analyze(educational_files)

# 결과 저장
with open("educational_analysis.json", 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# 리포트 생성
analyzer.generate_report(results, "educational_report.txt")
```

**특징**:
- 5가지 기준으로 콘텐츠 평가
- 학습 목표 및 핵심 개념 자동 추출
- 추론 과정 포함 (Gemini)
- 상세한 텍스트 리포트 생성
- 다양한 미디어 타입 통합 처리

---

## 정리

이번 실습에서 학습한 내용:

### 핵심 개념
1. **표준 콘텐츠 블록**: LangChain v1.0의 프로바이더 독립적인 표준 형식
2. **content_blocks 속성**: 타입 안전한 멀티모달 처리를 위한 핵심 API
3. **다양한 미디어 처리**: 이미지, 오디오, 비디오, PDF 통합 처리
4. **표준 직렬화**: `output_version="v1"`로 일관된 출력 형식 보장
5. **추론 과정 추출**: `reasoning` 블록으로 모델의 사고 과정 확인

### 주요 장점
- ✅ **프로바이더 독립성**: OpenAI ↔ Gemini ↔ Claude 간 쉬운 전환
- ✅ **타입 안전성**: 표준화된 블록 구조로 오류 감소
- ✅ **일관된 API**: 모든 미디어 타입에 동일한 인터페이스
- ✅ **미래 호환성**: LangChain 표준을 따르므로 버전 업그레이드 용이

### 실무 적용
- **콘텐츠 관리**: 자동 메타데이터 생성 및 태그 기반 검색
- **교육 플랫폼**: 학습 자료 자동 평가 및 품질 관리
- **API 서버**: 멀티 프로바이더 지원 멀티모달 분석 서비스
- **미디어 분석**: 통합 파이프라인으로 다양한 미디어 처리

### 다음 단계
- 스트리밍 방식 멀티모달 처리
- File ID 방식 활용 (토큰 최적화)
- 대용량 미디어 처리 최적화
- 프로덕션 환경 모니터링 및 로깅

LangChain v1.0의 표준 콘텐츠 블록은 멀티모달 애플리케이션 개발을 더욱 견고하고 유지보수하기 쉽게 만듭니다.
