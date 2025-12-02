# LangChain 멀티모달 Chat Model 활용

## 학습 목표

이 실습을 완료하면 다음을 할 수 있습니다:

1. **멀티모달 입력 처리**: LangChain Chat Model에서 이미지, 텍스트를 동시에 처리하는 방법을 이해합니다
2. **인코딩 방식 선택**: URL 직접 전달과 base64 인코딩 방식의 차이점과 활용 시나리오를 학습합니다
3. **Tool Calling 구현**: 멀티모달 모델에 도구 기능을 바인딩하여 구조화된 출력을 생성합니다
4. **멀티 프로바이더 활용**: OpenAI, Google Gemini, Groq 등 다양한 LLM 제공자의 멀티모달 API를 비교하고 활용합니다
5. **실무 시스템 구축**: 이미지 분석 파이프라인을 설계하고 프로덕션 환경에 적용합니다

---

## 핵심 개념

### 1. 멀티모달리티(Multimodality)란?

**멀티모달리티**는 텍스트, 이미지, 오디오, 비디오 등 여러 형태의 데이터를 동시에 처리하는 AI 기술입니다.

| 구성 요소 | 설명 | 활용 예시 |
|----------|------|----------|
| **채팅 모델** | 다양한 입출력 형식 처리 | 이미지 + 텍스트 질문 → 텍스트 답변 |
| **임베딩 모델** | 다양한 데이터 타입의 벡터 표현 | 이미지와 텍스트를 동일한 벡터 공간에 표현 |
| **벡터 저장소** | 멀티모달 데이터의 임베딩 검색 | 텍스트 쿼리로 유사 이미지 검색 |

### 2. 멀티모달 입력 방식 비교

| 방식 | 장점 | 단점 | 적용 시나리오 |
|------|------|------|--------------|
| **URL 직접 전달** | 간단하고 빠른 구현<br>네트워크 트래픽 절약 | 공개 URL 필요<br>네트워크 의존성 | 웹 이미지 분석<br>공개 데이터셋 처리 |
| **Base64 인코딩** | 로컬 파일 처리 가능<br>네트워크 독립적<br>안정적 전달 | 페이로드 크기 증가<br>인코딩 오버헤드 | 비공개 이미지 처리<br>서버 간 데이터 전송 |

### 3. HumanMessage 구조

LangChain의 `HumanMessage`는 멀티모달 입력을 구조화된 형태로 전달합니다:

```python
HumanMessage(
    content=[
        {"type": "text", "text": "프롬프트 텍스트"},
        {"type": "image_url", "image_url": {"url": "이미지_URL_또는_base64"}}
    ]
)
```

### 4. 모델별 멀티모달 호환성

| 모델 제공자 | 지원 모델 | 입력 형식 | 특징 |
|------------|----------|----------|------|
| **OpenAI** | gpt-4o, gpt-4o-mini | 이미지, 텍스트 | 높은 정확도, 안정적 API |
| **Google Gemini** | gemini-2.5-flash | 이미지, 텍스트, 비디오 | 빠른 응답, 다국어 지원 |
| **Anthropic Claude** | claude-3-opus | 이미지, 텍스트, PDF | 긴 컨텍스트, 상세한 분석 |
| **Groq** | llama-3.2-11b-vision | 이미지, 텍스트 | 빠른 추론 속도 |
| **Ollama** | llava, bakllava | 이미지, 텍스트 | 로컬 실행, 프라이버시 |

---

## 환경 설정

### 필요한 라이브러리 설치

```bash
# 핵심 라이브러리
pip install langchain langchain-openai langchain-google-genai langchain-groq langchain-ollama
pip install python-dotenv pillow requests

# 모니터링 (선택사항)
pip install langfuse
```

### 환경 변수 설정

`.env` 파일을 생성하고 API 키를 설정합니다:

```env
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
GROQ_API_KEY=your_groq_key
```

### 기본 import

```python
from dotenv import load_dotenv
load_dotenv()

import os
import base64
import requests
from PIL import Image
from io import BytesIO

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
```

---

## 단계별 구현

### 1단계: OpenAI 모델 초기화 및 URL 방식 이미지 처리

가장 간단한 방식으로 공개 URL의 이미지를 직접 전달합니다.

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 이미지 처리용 모델 초기화
image_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

def process_image_url(image_url: str, prompt: str):
    """URL 방식으로 이미지 처리"""
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    )
    return image_model.invoke([message])

# 예시 이미지 URL (COCO 데이터셋)
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"

# 이미지 설명 생성
result = process_image_url(image_url, "이 이미지를 자세히 설명해주세요")
print(result.content)
```

**출력 예시:**
```
이미지에는 두 마리의 고양이가 소파 위에서 편안하게 잠을 자고 있는 모습이 담겨 있습니다.
소파는 분홍색이며, 고양이들 사이에는 리모컨이 놓여 있습니다.
왼쪽의 고양이는 길고 날씬한 모습이며, 오른쪽의 고양이는 더 통통하고 둥근 형태입니다.
```

### 2단계: Base64 인코딩 방식으로 로컬 이미지 처리

로컬 파일이나 비공개 이미지를 처리할 때는 base64 인코딩이 필요합니다.

```python
import base64
import requests

def get_base64_image(image_path: str) -> str:
    """로컬 이미지를 base64로 인코딩"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string

def get_image_base64_from_url(image_url: str) -> str:
    """URL 이미지를 다운로드하여 base64로 인코딩"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    response = requests.get(image_url, headers=headers)
    response.raise_for_status()
    return base64.b64encode(response.content).decode('utf-8')

def process_image_base64(image_path: str, prompt: str):
    """Base64 방식으로 이미지 처리"""
    # 로컬 파일 또는 URL 판단
    if not image_path.startswith("http"):
        image_data = get_base64_image(image_path)
    else:
        image_data = get_image_base64_from_url(image_path)

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            },
        ]
    )

    return image_model.invoke([message])

# 로컬 파일 처리
result = process_image_base64("portrait.jpg", "이 초상화의 화풍과 특징을 분석해주세요")
print(result.content)
```

**출력 예시:**
```
이 이미지는 유명한 화가의 자화상으로 보입니다.
화가는 붉은 머리와 수염을 가지고 있으며, 진한 색상의 외투를 입고 있습니다.
배경은 다양한 색상의 점들로 구성되어 있어, 화가의 독특한 화풍을 잘 보여줍니다.
```

### 3단계: 여러 이미지 동시 처리

여러 이미지를 비교하거나 동시에 분석해야 할 때 사용합니다.

```python
def process_multiple_images(image_paths: list, prompt: str):
    """여러 이미지를 동시에 처리"""
    content = [{"type": "text", "text": prompt}]

    for image_path in image_paths:
        # 이미지를 base64 문자열로 변환
        if not image_path.startswith("http"):
            image_data = get_base64_image(image_path)
        else:
            image_data = get_image_base64_from_url(image_path)

        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
        })

    message = HumanMessage(content=content)
    return image_model.invoke([message])

# 여러 이미지 비교 분석
image_paths = [
    "portrait.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/600px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"
]

result = process_multiple_images(
    image_paths,
    "이 두 작품을 비교하여 화풍, 색채, 구도의 차이점을 설명해주세요"
)
print(result.content)
```

**출력 예시:**
```
두 이미지는 모두 빈센트 반 고흐의 작품입니다.

첫 번째는 자화상으로, 인물 중심의 구도와 차분한 색채가 특징입니다.
두 번째는 '별이 빛나는 밤'으로, 소용돌이치는 하늘과 역동적인 붓터치가 돋보입니다.

공통점: 두터운 임파스토 기법, 강렬한 색채 대비
차이점: 자화상은 내면 표현, 별이 빛나는 밤은 자연의 웅장함 표현
```

### 4단계: Tool Calling과 멀티모달 통합

멀티모달 모델에 도구 기능을 추가하여 구조화된 출력을 생성합니다.

```python
from typing import Literal
from langchain_core.tools import tool

@tool
def weather_tool(weather: Literal["sunny", "cloudy", "rainy"]) -> str:
    """이미지에서 감지된 날씨 상태를 설명하는 도구"""
    weather_descriptions = {
        "sunny": "날씨가 맑습니다. ☀️",
        "cloudy": "날씨가 흐립니다. ☁️",
        "rainy": "비가 옵니다. 🌧️"
    }
    return weather_descriptions.get(weather, "날씨를 알 수 없습니다.")

def setup_model_with_tools(model):
    """모델에 도구 바인딩"""
    return model.bind_tools([weather_tool])

# 도구가 바인딩된 모델 생성
model_with_tools = setup_model_with_tools(image_model)

# 이미지 URL
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
image_data = get_image_base64_from_url(image_url)

message = HumanMessage(
    content=[
        {"type": "text", "text": "이 이미지의 날씨를 분석해주세요"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
    ],
)

# 도구 호출
response = model_with_tools.invoke([message])
print("Tool Calls:", response.tool_calls)

# 도구 실행
if response.tool_calls:
    result = weather_tool.invoke(response.tool_calls[0])
    print("결과:", result)
```

**출력 예시:**
```
Tool Calls: [{'name': 'weather_tool', 'args': {'weather': 'sunny'}, 'id': 'call_xyz', 'type': 'tool_call'}]
결과: ToolMessage(content='날씨가 맑습니다. ☀️', name='weather_tool', ...)
```

### 5단계: 다른 LLM 제공자 활용 (Google Gemini)

다양한 LLM 제공자의 멀티모달 기능을 활용할 수 있습니다.

```python
from langchain_google_genai import ChatGoogleGenerativeAI

# Google Gemini 모델 초기화
gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

def process_image_with_gemini(image_paths: list, prompt: str):
    """Gemini로 이미지 처리"""
    content = [{"type": "text", "text": prompt}]

    for image_path in image_paths:
        if not image_path.startswith("http"):
            image_data = get_base64_image(image_path)
            url = f"data:image/jpeg;base64,{image_data}"
        else:
            url = image_path

        content.append({"type": "image_url", "image_url": {"url": url}})

    message = HumanMessage(content=content)
    return gemini_model.invoke([message])

# Gemini로 이미지 분석
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
result = process_image_with_gemini([image_url], "이 이미지를 한국어로 자세히 설명해주세요")
print(result.content)
```

---

## 실습 문제

### 문제 1: URL 방식 이미지 분석 시스템 (기본)

**난이도**: ⭐⭐☆☆☆

**문제**:
공개 이미지 URL을 입력받아 다음 정보를 추출하는 함수를 작성하세요:
- 이미지의 주요 객체
- 색상 톤
- 분위기
- 추천 사용 용도

**입력 예시**:
```python
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
```

**출력 예시**:
```python
{
    "objects": ["고양이", "소파", "리모컨"],
    "color_tone": "따뜻한 색상 (핑크, 베이지)",
    "mood": "편안하고 아늑한",
    "recommended_use": "반려동물 관련 콘텐츠, 휴식 테마"
}
```

### 문제 2: Base64 방식 다중 이미지 비교 (중급)

**난이도**: ⭐⭐⭐☆☆

**문제**:
로컬 폴더에 있는 여러 제품 이미지를 비교하여 가장 마케팅에 적합한 이미지를 선정하는 시스템을 구현하세요.

**요구사항**:
1. 폴더 내 모든 이미지를 base64로 인코딩
2. 각 이미지의 구도, 조명, 선명도 평가
3. 점수화하여 최적 이미지 선정
4. 선정 이유를 상세히 설명

**평가 기준**:
- 구도 (30점): 중심 배치, 여백 활용
- 조명 (30점): 밝기, 그림자 처리
- 선명도 (20점): 초점, 디테일
- 전체 인상 (20점): 고급스러움, 매력도

### 문제 3: Tool Calling 기반 이미지 자동 태깅 시스템 (고급)

**난이도**: ⭐⭐⭐⭐☆

**문제**:
이미지를 분석하여 자동으로 태그를 생성하는 시스템을 Tool Calling을 활용해 구현하세요.

**요구사항**:
1. 카테고리별 도구 정의:
   - `object_detection_tool`: 객체 감지
   - `scene_classification_tool`: 장면 분류
   - `emotion_analysis_tool`: 감정 분석
   - `color_palette_tool`: 색상 팔레트 추출

2. 각 도구는 구조화된 출력 반환
3. 모든 도구 결과를 통합하여 최종 메타데이터 생성
4. JSON 형식으로 저장

**출력 예시**:
```json
{
    "image_id": "img_001",
    "objects": ["cat", "sofa", "remote"],
    "scene": "indoor_living_room",
    "emotions": ["relaxed", "cozy"],
    "color_palette": ["#FFC0CB", "#F5F5DC", "#808080"],
    "tags": ["pet", "home", "comfort", "lifestyle"],
    "confidence": 0.92
}
```

---

## 솔루션 예시

### 문제 1 솔루션: URL 방식 이미지 분석 시스템

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import json

class ImageAnalyzer:
    def __init__(self, model_name="gpt-4o-mini"):
        self.model = ChatOpenAI(model=model_name, temperature=0)

    def analyze_image(self, image_url: str) -> dict:
        """이미지를 분석하여 구조화된 정보 반환"""
        prompt = """이 이미지를 분석하여 다음 정보를 JSON 형식으로 반환하세요:
        {
            "objects": ["주요 객체 목록"],
            "color_tone": "색상 톤 설명",
            "mood": "분위기 설명",
            "recommended_use": "추천 사용 용도"
        }

        반드시 유효한 JSON 형식으로만 응답하세요."""

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        )

        response = self.model.invoke([message])

        # JSON 파싱
        try:
            # 코드 블록 제거
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            return json.loads(content.strip())
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 기본 구조 반환
            return {
                "objects": [],
                "color_tone": "분석 실패",
                "mood": "분석 실패",
                "recommended_use": "분석 실패",
                "raw_response": response.content
            }

# 사용 예시
analyzer = ImageAnalyzer()
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
result = analyzer.analyze_image(image_url)

print("=== 이미지 분석 결과 ===")
print(json.dumps(result, indent=2, ensure_ascii=False))
```

**출력**:
```json
{
  "objects": ["고양이 2마리", "분홍색 소파", "리모컨 2개"],
  "color_tone": "따뜻한 파스텔 톤 (핑크, 베이지)",
  "mood": "편안하고 평화로운, 아늑한 가정 분위기",
  "recommended_use": "반려동물 용품 광고, 가정용 가구 마케팅, 휴식/힐링 테마 콘텐츠"
}
```

### 문제 2 솔루션: Base64 방식 다중 이미지 비교

```python
import base64
import os
from glob import glob
from typing import List, Dict
import json

class ProductImageSelector:
    def __init__(self, model_name="gpt-4o-mini"):
        self.model = ChatOpenAI(model=model_name, temperature=0)

    def get_base64_image(self, image_path: str) -> str:
        """로컬 이미지를 base64로 인코딩"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def evaluate_single_image(self, image_path: str) -> Dict:
        """단일 이미지 평가"""
        image_data = self.get_base64_image(image_path)

        prompt = """이 제품 이미지를 마케팅 관점에서 평가하여 JSON으로 반환하세요:
        {
            "composition_score": 0-30 (구도 점수),
            "composition_comment": "구도 평가 설명",
            "lighting_score": 0-30 (조명 점수),
            "lighting_comment": "조명 평가 설명",
            "sharpness_score": 0-20 (선명도 점수),
            "sharpness_comment": "선명도 평가 설명",
            "overall_impression_score": 0-20 (전체 인상 점수),
            "overall_comment": "전체 평가 설명",
            "total_score": 합계 (0-100)
        }"""

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                }
            ]
        )

        response = self.model.invoke([message])

        # JSON 파싱
        try:
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]

            evaluation = json.loads(content.strip())
            evaluation["image_path"] = image_path
            evaluation["filename"] = os.path.basename(image_path)
            return evaluation
        except json.JSONDecodeError:
            return {
                "image_path": image_path,
                "filename": os.path.basename(image_path),
                "total_score": 0,
                "error": "평가 실패"
            }

    def select_best_image(self, folder_path: str, pattern: str = "*.jpg") -> Dict:
        """폴더 내 최적 이미지 선정"""
        # 이미지 파일 목록
        image_paths = glob(os.path.join(folder_path, pattern))

        if not image_paths:
            return {"error": "이미지 파일을 찾을 수 없습니다"}

        print(f"총 {len(image_paths)}개의 이미지를 평가합니다...\n")

        # 각 이미지 평가
        evaluations = []
        for idx, image_path in enumerate(image_paths, 1):
            print(f"[{idx}/{len(image_paths)}] {os.path.basename(image_path)} 평가 중...")
            evaluation = self.evaluate_single_image(image_path)
            evaluations.append(evaluation)

        # 점수순 정렬
        evaluations.sort(key=lambda x: x.get("total_score", 0), reverse=True)

        # 결과 정리
        best_image = evaluations[0]

        return {
            "best_image": best_image["filename"],
            "best_image_path": best_image["image_path"],
            "score": best_image.get("total_score", 0),
            "evaluation_detail": best_image,
            "all_rankings": [
                {
                    "rank": idx + 1,
                    "filename": eval["filename"],
                    "score": eval.get("total_score", 0)
                }
                for idx, eval in enumerate(evaluations)
            ]
        }

# 사용 예시
selector = ProductImageSelector()

# 폴더 내 이미지 평가
result = selector.select_best_image("./product_images/")

print("\n=== 최적 이미지 선정 결과 ===")
print(f"선정된 이미지: {result['best_image']}")
print(f"총점: {result['score']}/100")
print(f"\n상세 평가:")
print(json.dumps(result['evaluation_detail'], indent=2, ensure_ascii=False))
print(f"\n전체 순위:")
for ranking in result['all_rankings']:
    print(f"  {ranking['rank']}위: {ranking['filename']} ({ranking['score']}점)")
```

### 문제 3 솔루션: Tool Calling 기반 이미지 자동 태깅 시스템

```python
from typing import Literal, List
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import json
import base64

# 도구 정의
@tool
def object_detection_tool(objects: List[str]) -> dict:
    """이미지에서 감지된 객체 목록"""
    return {
        "tool": "object_detection",
        "detected_objects": objects,
        "count": len(objects)
    }

@tool
def scene_classification_tool(
    scene: Literal[
        "indoor_living_room", "indoor_kitchen", "indoor_bedroom",
        "outdoor_nature", "outdoor_urban", "outdoor_beach",
        "workplace_office", "workplace_factory",
        "commercial_store", "commercial_restaurant"
    ]
) -> dict:
    """이미지의 장면 분류"""
    return {
        "tool": "scene_classification",
        "scene_type": scene
    }

@tool
def emotion_analysis_tool(emotions: List[Literal[
    "happy", "sad", "relaxed", "energetic", "cozy",
    "professional", "playful", "elegant", "rustic"
]]) -> dict:
    """이미지에서 느껴지는 감정/분위기 분석"""
    return {
        "tool": "emotion_analysis",
        "emotions": emotions
    }

@tool
def color_palette_tool(colors: List[str]) -> dict:
    """이미지의 주요 색상 팔레트 (HEX 코드)"""
    return {
        "tool": "color_palette",
        "palette": colors
    }

class AutoTaggingSystem:
    def __init__(self, model_name="gpt-4o-mini"):
        self.model = ChatOpenAI(model=model_name, temperature=0)
        # 모든 도구 바인딩
        self.model_with_tools = self.model.bind_tools([
            object_detection_tool,
            scene_classification_tool,
            emotion_analysis_tool,
            color_palette_tool
        ])

    def get_base64_image(self, image_path: str) -> str:
        """로컬 이미지를 base64로 인코딩"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def generate_tags(self, image_path: str, image_id: str = None) -> dict:
        """이미지 자동 태깅"""
        if image_id is None:
            image_id = os.path.basename(image_path)

        # 이미지 로드
        image_data = self.get_base64_image(image_path)

        # 프롬프트
        prompt = """이 이미지를 분석하여 다음 도구들을 모두 호출하세요:

        1. object_detection_tool: 이미지 내 모든 객체 감지
        2. scene_classification_tool: 장면 분류
        3. emotion_analysis_tool: 감정/분위기 분석
        4. color_palette_tool: 주요 색상 5개 (HEX 코드)

        모든 도구를 순차적으로 호출하세요."""

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                }
            ]
        )

        # 도구 호출
        response = self.model_with_tools.invoke([message])

        # 도구 결과 수집
        metadata = {
            "image_id": image_id,
            "image_path": image_path,
            "objects": [],
            "scene": None,
            "emotions": [],
            "color_palette": [],
            "tags": [],
            "confidence": 0.0
        }

        if response.tool_calls:
            total_confidence = 0
            tool_count = 0

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                if tool_name == "object_detection_tool":
                    metadata["objects"] = tool_args.get("objects", [])
                    total_confidence += 0.95
                    tool_count += 1

                elif tool_name == "scene_classification_tool":
                    metadata["scene"] = tool_args.get("scene", "unknown")
                    total_confidence += 0.90
                    tool_count += 1

                elif tool_name == "emotion_analysis_tool":
                    metadata["emotions"] = tool_args.get("emotions", [])
                    total_confidence += 0.85
                    tool_count += 1

                elif tool_name == "color_palette_tool":
                    metadata["color_palette"] = tool_args.get("colors", [])
                    total_confidence += 0.90
                    tool_count += 1

            # 평균 신뢰도 계산
            if tool_count > 0:
                metadata["confidence"] = round(total_confidence / tool_count, 2)

            # 태그 생성 (객체 + 장면 + 감정)
            metadata["tags"] = (
                metadata["objects"] +
                [metadata["scene"]] +
                metadata["emotions"]
            )

        return metadata

    def batch_tagging(self, image_folder: str, pattern: str = "*.jpg") -> List[dict]:
        """폴더 내 모든 이미지 일괄 태깅"""
        image_paths = glob(os.path.join(image_folder, pattern))
        results = []

        print(f"총 {len(image_paths)}개의 이미지를 태깅합니다...\n")

        for idx, image_path in enumerate(image_paths, 1):
            filename = os.path.basename(image_path)
            print(f"[{idx}/{len(image_paths)}] {filename} 태깅 중...")

            metadata = self.generate_tags(image_path, image_id=f"img_{idx:03d}")
            results.append(metadata)

        return results

    def save_metadata(self, metadata_list: List[dict], output_path: str):
        """메타데이터를 JSON 파일로 저장"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, indent=2, ensure_ascii=False)
        print(f"\n메타데이터 저장 완료: {output_path}")

# 사용 예시
tagging_system = AutoTaggingSystem()

# 단일 이미지 태깅
single_result = tagging_system.generate_tags("portrait.jpg", "img_001")
print("=== 단일 이미지 태깅 결과 ===")
print(json.dumps(single_result, indent=2, ensure_ascii=False))

# 폴더 내 일괄 태깅
batch_results = tagging_system.batch_tagging("./images/")
tagging_system.save_metadata(batch_results, "image_metadata.json")

# 결과 요약
print("\n=== 태깅 결과 요약 ===")
for result in batch_results:
    print(f"{result['image_id']}: {len(result['objects'])}개 객체, "
          f"{result['scene']}, 신뢰도 {result['confidence']}")
```

**출력 예시**:
```json
{
  "image_id": "img_001",
  "image_path": "portrait.jpg",
  "objects": ["person", "portrait", "painting", "face"],
  "scene": "indoor_living_room",
  "emotions": ["elegant", "professional"],
  "color_palette": ["#4A7C59", "#D4A574", "#8B4513", "#2F4F4F", "#F5DEB3"],
  "tags": ["person", "portrait", "painting", "face", "indoor_living_room", "elegant", "professional"],
  "confidence": 0.90
}
```

---

## 실무 활용 예시

### 예시 1: 의료 영상 분석 시스템

의료 이미지(X-ray, MRI, CT)를 분석하여 이상 소견을 보조적으로 탐지하는 시스템입니다.

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from typing import Literal
import base64
import json
from datetime import datetime

class MedicalImageAnalyzer:
    """의료 영상 분석 시스템"""

    def __init__(self, model_name="gpt-4o"):
        # 의료 분석에는 더 정확한 모델 사용
        self.model = ChatOpenAI(model=model_name, temperature=0)

    def get_base64_image(self, image_path: str) -> str:
        """이미지 base64 인코딩"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def analyze_medical_image(
        self,
        image_path: str,
        image_type: Literal["xray", "mri", "ct", "ultrasound"],
        body_part: str,
        patient_id: str = None
    ) -> dict:
        """의료 영상 분석"""

        image_data = self.get_base64_image(image_path)

        prompt = f"""당신은 의료 영상 분석 보조 AI입니다.

이미지 타입: {image_type.upper()}
신체 부위: {body_part}

다음 사항을 분석하여 JSON 형식으로 반환하세요:
{{
    "image_quality": "양호/보통/불량",
    "quality_issues": ["품질 문제 목록"],
    "visible_structures": ["확인 가능한 해부학적 구조"],
    "potential_findings": [
        {{
            "finding": "발견 사항",
            "location": "위치",
            "severity": "경미함/보통/심각함",
            "confidence": 0.0-1.0
        }}
    ],
    "recommendations": ["추가 검사 또는 확인 필요 사항"],
    "disclaimer": "이 분석은 보조적 참고용이며 전문의 판독을 대체할 수 없습니다"
}}

중요: 이 분석은 의료 전문가의 최종 판단을 보조하는 참고 자료입니다."""

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                }
            ]
        )

        response = self.model.invoke([message])

        try:
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]

            analysis = json.loads(content.strip())

            # 메타데이터 추가
            analysis["metadata"] = {
                "patient_id": patient_id,
                "image_path": image_path,
                "image_type": image_type,
                "body_part": body_part,
                "analysis_timestamp": datetime.now().isoformat(),
                "model_used": model_name
            }

            return analysis

        except json.JSONDecodeError:
            return {
                "error": "분석 실패",
                "raw_response": response.content,
                "metadata": {
                    "patient_id": patient_id,
                    "image_path": image_path,
                    "analysis_timestamp": datetime.now().isoformat()
                }
            }

    def generate_report(self, analysis: dict) -> str:
        """분석 결과를 읽기 쉬운 리포트로 변환"""
        report = f"""
========================================
의료 영상 분석 리포트
========================================

[환자 정보]
- 환자 ID: {analysis['metadata']['patient_id']}
- 분석 일시: {analysis['metadata']['analysis_timestamp']}

[영상 정보]
- 영상 타입: {analysis['metadata']['image_type'].upper()}
- 신체 부위: {analysis['metadata']['body_part']}
- 영상 품질: {analysis.get('image_quality', 'N/A')}

[품질 문제]
"""
        if analysis.get('quality_issues'):
            for issue in analysis['quality_issues']:
                report += f"  - {issue}\n"
        else:
            report += "  - 없음\n"

        report += "\n[확인된 해부학적 구조]\n"
        if analysis.get('visible_structures'):
            for structure in analysis['visible_structures']:
                report += f"  - {structure}\n"

        report += "\n[잠재적 발견 사항]\n"
        if analysis.get('potential_findings'):
            for finding in analysis['potential_findings']:
                report += f"""
  발견: {finding['finding']}
  위치: {finding['location']}
  중요도: {finding['severity']}
  신뢰도: {finding['confidence']:.2f}
"""
        else:
            report += "  - 특이사항 없음\n"

        report += "\n[권장사항]\n"
        if analysis.get('recommendations'):
            for rec in analysis['recommendations']:
                report += f"  - {rec}\n"

        report += f"\n[면책사항]\n  {analysis.get('disclaimer', '')}\n"
        report += "========================================\n"

        return report

# 사용 예시
analyzer = MedicalImageAnalyzer()

# X-ray 이미지 분석
analysis = analyzer.analyze_medical_image(
    image_path="chest_xray.jpg",
    image_type="xray",
    body_part="chest",
    patient_id="P12345"
)

# 분석 결과 출력
print(json.dumps(analysis, indent=2, ensure_ascii=False))

# 리포트 생성
report = analyzer.generate_report(analysis)
print(report)

# 리포트 저장
with open(f"medical_report_{analysis['metadata']['patient_id']}.txt", 'w', encoding='utf-8') as f:
    f.write(report)
```

**특징**:
- 의료 영상의 품질 평가
- 해부학적 구조 확인
- 잠재적 이상 소견 탐지
- 신뢰도 수치 제공
- 추가 검사 권장사항
- 면책사항 명시 (보조 도구임을 강조)

### 예시 2: 제품 품질 검사 시스템

제조업 현장에서 제품의 외관 불량을 자동으로 감지하는 시스템입니다.

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from typing import Literal, List
import base64
import json
from datetime import datetime
import os

@tool
def defect_detection_tool(defects: List[dict]) -> dict:
    """제품 불량 감지 도구

    Args:
        defects: 불량 목록, 각 항목은 {type, location, severity} 포함
    """
    return {
        "tool": "defect_detection",
        "defects": defects,
        "defect_count": len(defects)
    }

@tool
def quality_score_tool(
    score: int,
    pass_fail: Literal["pass", "fail", "review_required"]
) -> dict:
    """품질 점수 평가 도구

    Args:
        score: 품질 점수 (0-100)
        pass_fail: 합격/불합격/재검토
    """
    return {
        "tool": "quality_score",
        "score": score,
        "decision": pass_fail
    }

class ProductQualityInspector:
    """제품 품질 검사 시스템"""

    def __init__(self, model_name="gpt-4o-mini"):
        self.model = ChatOpenAI(model=model_name, temperature=0)
        self.model_with_tools = self.model.bind_tools([
            defect_detection_tool,
            quality_score_tool
        ])

    def get_base64_image(self, image_path: str) -> str:
        """이미지 base64 인코딩"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def inspect_product(
        self,
        image_path: str,
        product_id: str,
        product_type: str,
        quality_criteria: dict = None
    ) -> dict:
        """제품 품질 검사"""

        if quality_criteria is None:
            quality_criteria = {
                "surface_defects": ["scratch", "dent", "discoloration"],
                "dimensional_accuracy": True,
                "assembly_defects": ["misalignment", "missing_parts"],
                "pass_threshold": 85
            }

        image_data = self.get_base64_image(image_path)

        prompt = f"""당신은 제조업 품질 관리 AI입니다.

제품 타입: {product_type}
검사 기준: {json.dumps(quality_criteria, ensure_ascii=False)}

이 제품 이미지를 검사하여 다음 도구들을 호출하세요:

1. defect_detection_tool: 발견된 모든 불량 목록
   - 각 불량은 다음 형식: {{"type": "불량 유형", "location": "위치", "severity": "경미함/보통/심각함"}}

2. quality_score_tool: 전체 품질 점수 (0-100)와 합격/불합격 판정
   - {quality_criteria['pass_threshold']}점 이상: pass
   - {quality_criteria['pass_threshold']}점 미만: fail
   - 애매한 경우: review_required

모든 도구를 호출하세요."""

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                }
            ]
        )

        response = self.model_with_tools.invoke([message])

        # 도구 결과 수집
        inspection_result = {
            "product_id": product_id,
            "product_type": product_type,
            "image_path": image_path,
            "timestamp": datetime.now().isoformat(),
            "defects": [],
            "quality_score": 0,
            "decision": "unknown",
            "quality_criteria": quality_criteria
        }

        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                if tool_name == "defect_detection_tool":
                    inspection_result["defects"] = tool_args.get("defects", [])

                elif tool_name == "quality_score_tool":
                    inspection_result["quality_score"] = tool_args.get("score", 0)
                    inspection_result["decision"] = tool_args.get("pass_fail", "unknown")

        return inspection_result

    def batch_inspection(
        self,
        image_folder: str,
        product_type: str,
        pattern: str = "*.jpg"
    ) -> dict:
        """일괄 품질 검사"""
        from glob import glob

        image_paths = glob(os.path.join(image_folder, pattern))
        results = []

        print(f"총 {len(image_paths)}개 제품 검사 시작...\n")

        pass_count = 0
        fail_count = 0
        review_count = 0

        for idx, image_path in enumerate(image_paths, 1):
            filename = os.path.basename(image_path)
            product_id = f"PROD_{datetime.now().strftime('%Y%m%d')}_{idx:04d}"

            print(f"[{idx}/{len(image_paths)}] {filename} 검사 중...")

            result = self.inspect_product(image_path, product_id, product_type)
            results.append(result)

            if result["decision"] == "pass":
                pass_count += 1
            elif result["decision"] == "fail":
                fail_count += 1
            else:
                review_count += 1

        # 통계 요약
        summary = {
            "total_inspected": len(results),
            "pass_count": pass_count,
            "fail_count": fail_count,
            "review_required_count": review_count,
            "pass_rate": round(pass_count / len(results) * 100, 2) if results else 0,
            "inspection_results": results,
            "batch_timestamp": datetime.now().isoformat()
        }

        return summary

    def generate_inspection_report(self, inspection_result: dict) -> str:
        """검사 리포트 생성"""
        report = f"""
========================================
제품 품질 검사 리포트
========================================

[제품 정보]
- 제품 ID: {inspection_result['product_id']}
- 제품 타입: {inspection_result['product_type']}
- 검사 일시: {inspection_result['timestamp']}

[검사 결과]
- 품질 점수: {inspection_result['quality_score']}/100
- 판정: {inspection_result['decision'].upper()}
- 발견된 불량 수: {len(inspection_result['defects'])}

[불량 상세]
"""
        if inspection_result['defects']:
            for idx, defect in enumerate(inspection_result['defects'], 1):
                report += f"""
  [{idx}] {defect['type']}
      위치: {defect['location']}
      심각도: {defect['severity']}
"""
        else:
            report += "  - 불량 없음\n"

        report += "\n[검사 기준]\n"
        for key, value in inspection_result['quality_criteria'].items():
            report += f"  - {key}: {value}\n"

        report += "========================================\n"

        return report

# 사용 예시
inspector = ProductQualityInspector()

# 단일 제품 검사
result = inspector.inspect_product(
    image_path="product_sample.jpg",
    product_id="PROD_20250601_0001",
    product_type="스마트폰 케이스"
)

print("=== 검사 결과 ===")
print(json.dumps(result, indent=2, ensure_ascii=False))

# 리포트 생성
report = inspector.generate_inspection_report(result)
print(report)

# 일괄 검사
batch_summary = inspector.batch_inspection(
    image_folder="./product_images/",
    product_type="스마트폰 케이스"
)

print("\n=== 일괄 검사 요약 ===")
print(f"전체 검사: {batch_summary['total_inspected']}개")
print(f"합격: {batch_summary['pass_count']}개 ({batch_summary['pass_rate']}%)")
print(f"불합격: {batch_summary['fail_count']}개")
print(f"재검토 필요: {batch_summary['review_required_count']}개")

# 결과 저장
with open("quality_inspection_summary.json", 'w', encoding='utf-8') as f:
    json.dump(batch_summary, f, indent=2, ensure_ascii=False)
```

**특징**:
- Tool Calling으로 구조화된 불량 정보 추출
- 자동 합격/불합격 판정
- 일괄 검사 지원
- 통계 요약 제공
- 검사 이력 저장

**활용 분야**:
- 전자제품 외관 검사
- 자동차 부품 품질 검사
- 식품 포장 상태 검사
- 섬유/의류 불량 검사

---

## 정리

이번 실습에서 학습한 내용:

### 핵심 개념
1. **멀티모달 입력 처리**: LangChain Chat Model에서 텍스트와 이미지를 동시에 처리
2. **인코딩 방식**: URL 직접 전달 vs Base64 인코딩의 장단점과 활용
3. **Tool Calling**: 멀티모달 모델에 도구 바인딩하여 구조화된 출력 생성
4. **멀티 프로바이더**: OpenAI, Gemini, Groq 등 다양한 LLM 활용

### 실무 적용
- **의료 영상 분석**: 보조 진단 도구로 이상 소견 탐지
- **품질 검사**: 제조업 현장의 자동 불량 검사
- **이미지 자동 태깅**: 콘텐츠 관리 시스템의 메타데이터 생성
- **마케팅 자동화**: 제품 이미지 평가 및 최적 이미지 선정

### 다음 단계
- 비디오 입력 처리 (프레임 추출 + 분석)
- 오디오 입력과 통합 (음성 + 이미지 동시 분석)
- 스트리밍 방식 멀티모달 처리
- 프로덕션 환경 배포 (API 서버화, 모니터링)

멀티모달 AI는 단일 모달리티로는 해결하기 어려웠던 복잡한 문제들을 해결할 수 있는 강력한 도구입니다. 실무에 적용할 때는 모델의 한계와 윤리적 고려사항을 충분히 검토해야 합니다.
