# W1_005_Gradio_Chatbot.md - Gradio 챗봇 UI 구현

## 🎯 학습 목표
- Gradio ChatInterface를 활용한 대화형 UI 구축 능력 습득
- LangChain과 Gradio 통합을 통한 AI 챗봇 개발
- 메모리 기능과 채팅 히스토리 관리 기법 학습
- 멀티모달 및 고급 UI 컴포넌트 활용법 이해

## 📚 핵심 개념

### Gradio ChatInterface
- **대화형 인터페이스**: 채팅 형태의 사용자 친화적 UI 제공
- **실시간 상호작용**: 스트리밍 응답과 즉시 피드백 지원
- **확장성**: 추가 입력 컴포넌트와 멀티모달 기능 통합 가능
- **커스터마이징**: 제목, 설명, 예시 질문 등 다양한 설정 옵션

### 채팅 히스토리 관리
- **메시지 형식**: OpenAI 스타일의 role/content 구조
- **컨텍스트 보존**: 이전 대화 내용을 활용한 연속적 대화
- **메모리 통합**: LangChain 메모리 시스템과의 원활한 연동
- **상태 관리**: 세션별 대화 상태 유지 및 관리

## 🔧 환경 설정

### 필수 라이브러리 설치
```bash
# 기본 설치
pip install gradio langchain langchain-openai

# UV 패키지 매니저 사용 시
uv add gradio langchain langchain-openai langchain-google-genai

# 추가 기능용 패키지
pip install gradio-pdf  # PDF 뷰어 기능
```

### Langfuse 통합 설정
```python
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv
from langchain_core.runnables.base import Runnable
import functools

# 환경변수 로드
load_dotenv()

# 전역 Langfuse handler 생성
_langfuse_handler = CallbackHandler()

# 기존 invoke 메서드를 래핑
_original_invoke = Runnable.invoke

@functools.wraps(_original_invoke)
def _invoke_with_langfuse(self, input, config=None, **kwargs):
    if config is None:
        config = {}
    if "callbacks" not in config:
        config["callbacks"] = []
    config["callbacks"].append(_langfuse_handler)
    return _original_invoke(self, input, config, **kwargs)

# Monkey patch 적용
Runnable.invoke = _invoke_with_langfuse
```

## 💻 코드 예제

### 1. 기본 ChatInterface 구조

#### Simple QA Chain
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 프롬프트 템플릿 정의
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 파이썬(Python) 코드 작성을 도와주는 AI 어시스턴트입니다."),
    ("human", "{user_input}")
])

# LLM 모델 정의
model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.3
)

# 체인 생성
chain = prompt | model | StrOutputParser()

# 체인 실행 테스트
response = chain.invoke({
    "user_input": "파이썬에서 리스트를 정렬하는 방법은 무엇인가요?"
})

print(response)
```

#### 기본 Gradio 인터페이스
```python
import gradio as gr

# 챗봇 함수 정의
def chat_function(message, history):
    print(f"입력 메시지: {message}")
    print("-" * 40)
    print(f"채팅 히스토리:")
    for chat in history:
        print(f"사용자: {chat['role']}, 메시지: {chat['content']}")
    return "응답 메시지"

# 챗봇 인터페이스 생성
demo = gr.ChatInterface(
    fn=chat_function,
    analytics_enabled=False,
    type="messages"  # OpenAI 스타일 메시지 형식
)

# 인터페이스 실행
demo.launch()
```

### 2. Echo 챗봇 예제

```python
def echo_bot(message, history):
    return f"당신이 입력한 메시지: {message}"

demo = gr.ChatInterface(
    fn=echo_bot,
    title="Echo 챗봇",
    description="입력한 메시지를 그대로 되돌려주는 챗봇입니다.",
    analytics_enabled=False
)

demo.launch()
```

### 3. 스트리밍 응답

```python
import time

def streaming_bot(message, history):
    response = f"처리 중인 메시지: {message}"
    for i in range(len(response)):
        time.sleep(0.1)  # 0.1초 대기
        yield response[:i+1]

demo = gr.ChatInterface(
    fn=streaming_bot,
    title="스트리밍 챗봇",
    description="입력한 메시지를 한 글자씩 처리하는 챗봇입니다.",
    analytics_enabled=False
)

demo.launch()
```

### 4. 추가 입력 컴포넌트

```python
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

def chat_function(message, history, model, temperature):
    if model == "gpt-4.1-mini":
        model = ChatOpenAI(model=model, temperature=temperature)
    elif model == "gemini-2.0-flash":
        model = ChatGoogleGenerativeAI(model=model, temperature=temperature)

    chain = prompt | model | StrOutputParser()

    response = chain.invoke({"user_input": message})
    return response

# 추가 입력 컴포넌트가 있는 인터페이스
with gr.Blocks() as demo:
    model_selector = gr.Dropdown(
        ["gpt-4.1-mini", "gemini-2.0-flash"],
        label="모델 선택"
    )
    slider = gr.Slider(
        0.0, 1.0,
        label="Temperature",
        value=0.3,
        step=0.1,
        render=False
    )

    gr.ChatInterface(
        fn=chat_function,
        additional_inputs=[model_selector, slider],
        analytics_enabled=False
    )

demo.launch()
```

### 5. 예시 질문 설정

```python
demo = gr.ChatInterface(
    fn=streaming_bot,
    title="스트리밍 챗봇",
    description="입력한 메시지를 한 글자씩 처리하는 챗봇입니다.",
    analytics_enabled=False,
    examples=[
        "파이썬 코드를 작성하는 방법을 알려주세요",
        "파이썬에서 리스트를 정렬하는 방법은 무엇인가요?",
    ]
)

demo.launch()
```

### 6. 멀티모달 기능

```python
import base64
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

def convert_to_url(image_path):
    """이미지를 URL 형식으로 변환"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded_string}"

def multimodal_bot(message, history):
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    if isinstance(message, dict):
        text = message.get("text", "")
        files = message.get("files", [])

        if files:
            # 이미지 처리
            image_urls = []
            for file_path in files:
                try:
                    image_url = convert_to_url(file_path)
                    image_urls.append({
                        "type": "image_url",
                        "image_url": image_url
                    })
                except Exception as e:
                    print(f"이미지 처리 중 오류 발생: {e}")
                    continue

            if image_urls:
                content = [
                    {"type": "text", "text": text if text else "이 이미지에 대해 설명해주세요."},
                    *image_urls
                ]

                try:
                    response = model.invoke([HumanMessage(content=content)])
                    return response.content
                except Exception as e:
                    return f"모델 응답 생성 중 오류가 발생했습니다: {str(e)}"

        return text if text else "이미지를 업로드해주세요."

    return "텍스트나 이미지를 입력해주세요."

# 멀티모달 인터페이스
demo = gr.ChatInterface(
    fn=multimodal_bot,
    type="messages",
    multimodal=True,
    title="멀티모달 챗봇",
    description="텍스트와 이미지를 함께 처리할 수 있는 챗봇입니다.",
    analytics_enabled=False,
    textbox=gr.MultimodalTextbox(
        placeholder="텍스트를 입력하거나 이미지를 업로드해주세요.",
        file_count="multiple",
        file_types=["image"]
    )
)

demo.launch()
```

### 7. PDF 뷰어 통합

```python
from gradio_pdf import PDF

def answer_invoke(message, history):
    return message

with gr.Blocks(analytics_enabled=False) as demo:
    with gr.Row():
        api_key_input = gr.Textbox(
            label="Enter OpenAI API Key",
            type="password",
            placeholder="sk-..."
        )

    with gr.Row():
        with gr.Column(scale=2):
            pdf_file = PDF(
                label="Upload PDF File",
                height=600
            )
        with gr.Column(scale=1):
            chatbot = gr.ChatInterface(
                fn=answer_invoke,
                type="messages",
                title="PDF-based Chatbot",
                description="Upload a PDF file and ask questions about its contents."
            )

demo.launch()
```

### 8. 메모리 통합 챗봇

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 메시지 플레이스홀더가 있는 프롬프트 템플릿
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 파이썬(Python) 코드 작성을 도와주는 AI 어시스턴트입니다."),
    MessagesPlaceholder("chat_history"),
    ("system", "이전 대화 내용을 참고하여 질문에 대해서 친절하게 답변합니다."),
    ("human", "{user_input}")
])

# 체인 생성
chain = prompt | model | StrOutputParser()

def answer_invoke(message, history):
    # 히스토리를 LangChain 메시지 형식으로 변환
    history_messages = []
    for msg in history:
        if msg['role'] == "user":
            history_messages.append(HumanMessage(content=msg['content']))
        elif msg['role'] == "assistant":
            history_messages.append(AIMessage(content=msg['content']))

    response = chain.invoke({
        "chat_history": history_messages,
        "user_input": message
    })
    return response

# 메모리 기능이 있는 챗봇 인터페이스
demo = gr.ChatInterface(
    fn=answer_invoke,
    type="messages",
    title="파이썬 코드 어시스턴트"
)

demo.launch()
```

## 🚀 실습해보기

### 실습: 맞춤형 여행 일정 계획 어시스턴트

**목표**: Gradio ChatInterface를 활용한 대화형 여행 계획 어시스턴트 구현

#### 요구사항
1. **기본 기능**
   - OpenAI Chat Completion API와 LangChain 활용
   - LCEL을 사용한 단계별 프롬프트 체인 구성
   - 채팅 히스토리를 활용한 연속적 대화

2. **모델 매개변수 최적화**
   - temperature=0.7: 적당한 창의성과 일관성 균형
   - top_p=0.9: 고품질 응답 생성
   - presence_penalty와 frequency_penalty: 반복 줄이고 다양성 증대

3. **프롬프트 설계**
   - 여행 플래너 역할 정의
   - 구체적인 정보 포함 지시
   - 한국어 응답 명시

#### 구현 코드
```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import gradio as gr

# 여행 계획 전문 프롬프트
prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 전문적인 여행 일정 계획 AI 어시스턴트입니다.

다음 가이드라인을 따라 응답해주세요:
1. 사용자의 여행 목적, 기간, 예산, 선호도를 파악하여 맞춤형 일정을 제안합니다
2. 구체적인 장소, 시간, 교통편, 비용 정보를 포함합니다
3. 현지 문화와 특색을 반영한 추천을 제공합니다
4. 안전 정보와 유용한 팁을 함께 제공합니다
5. 모든 응답은 친근하고 전문적인 톤으로 한국어로 작성합니다

이전 대화 내용을 참고하여 연속성 있는 조언을 제공하세요."""),
    MessagesPlaceholder("chat_history"),
    ("system", "사용자의 질문에 대해 여행 전문가로서 상세하고 실용적인 답변을 제공합니다."),
    ("human", "{user_input}")
])

# 최적화된 LLM 모델 설정
model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.7,        # 창의성과 일관성 균형
    top_p=0.9,             # 고품질 토큰 선택
    presence_penalty=0.3,   # 새로운 주제 도입 촉진
    frequency_penalty=0.3,  # 반복 표현 방지
    max_tokens=1500        # 충분한 응답 길이
)

# LCEL 체인 구성
chain = prompt | model | StrOutputParser()

def travel_planner(message, history):
    """여행 계획 어시스턴트 메인 함수"""
    # 채팅 히스토리를 LangChain 메시지 형식으로 변환
    history_messages = []
    for msg in history:
        if msg['role'] == "user":
            history_messages.append(HumanMessage(content=msg['content']))
        elif msg['role'] == "assistant":
            history_messages.append(AIMessage(content=msg['content']))

    try:
        # 체인 실행
        response = chain.invoke({
            "chat_history": history_messages,
            "user_input": message
        })
        return response
    except Exception as e:
        return f"죄송합니다. 처리 중 오류가 발생했습니다: {str(e)}"

# Gradio ChatInterface 생성
demo = gr.ChatInterface(
    fn=travel_planner,
    type="messages",
    title="🌍 맞춤형 여행 일정 계획 어시스턴트",
    description="당신만의 특별한 여행을 계획해드립니다. 여행지, 기간, 예산, 선호도를 알려주세요!",
    analytics_enabled=False,
    examples=[
        "제주도 2박 3일 여행 계획을 세워주세요",
        "부산에서 먹을거리 위주로 1박 2일 여행하고 싶어요",
        "서울 근교에서 가족과 함께 당일치기 여행 추천해주세요",
        "유럽 배낭여행 3주 일정을 계획해주세요"
    ]
)

# 인터페이스 실행
demo.launch()
```

## 📋 해답

### 완전한 여행 계획 어시스턴트 구현

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import gradio as gr

# 전문적인 여행 계획 프롬프트 템플릿
prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 경험 많은 여행 전문가이자 맞춤형 여행 일정 계획 AI 어시스턴트입니다.

🎯 역할과 목표:
- 사용자의 니즈에 완벽히 맞는 개인화된 여행 계획 수립
- 실용적이고 실현 가능한 일정 제안
- 현지 문화와 특색을 반영한 진정성 있는 추천

📋 응답 가이드라인:
1. **정보 수집**: 여행 목적, 기간, 예산, 인원, 선호 활동, 숙박 스타일 파악
2. **맞춤 제안**: 수집된 정보를 바탕으로 개인화된 일정 구성
3. **구체적 정보**: 장소명, 운영시간, 예상 비용, 교통 정보, 소요 시간 명시
4. **실용적 팁**: 예약 방법, 할인 정보, 현지 에티켓, 주의사항 포함
5. **유연성**: 대안 옵션과 날씨/상황별 백업 플랜 제시

💡 특별 고려사항:
- 계절과 날씨 고려한 추천
- 현지 축제와 이벤트 정보
- 교통비와 입장료 등 비용 투명성
- 접근성과 안전 정보
- 포토 스팟과 인스타그램 명소

이전 대화를 참고하여 연속성 있고 발전적인 조언을 제공하세요."""),
    MessagesPlaceholder("chat_history"),
    ("system", "사용자의 여행 관련 질문에 대해 전문가 수준의 상세하고 실용적인 답변을 한국어로 제공합니다."),
    ("human", "{user_input}")
])

# 창의성과 일관성이 균형잡힌 모델 설정
model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.7,        # 적당한 창의성 유지
    top_p=0.9,             # 높은 품질의 토큰 선택
    presence_penalty=0.3,   # 새로운 주제와 아이디어 도입
    frequency_penalty=0.3,  # 반복적 표현 줄이기
    max_tokens=2000        # 충분한 응답 길이
)

# LCEL을 활용한 체인 구성
planning_chain = prompt | model | StrOutputParser()

def travel_planning_assistant(message, history):
    """
    여행 계획 어시스턴트 메인 함수

    Args:
        message (str): 사용자 입력 메시지
        history (list): 채팅 히스토리 (OpenAI 형식)

    Returns:
        str: AI 어시스턴트의 응답
    """
    # 채팅 히스토리를 LangChain 메시지 객체로 변환
    history_messages = []
    for msg in history:
        if msg['role'] == "user":
            history_messages.append(HumanMessage(content=msg['content']))
        elif msg['role'] == "assistant":
            history_messages.append(AIMessage(content=msg['content']))

    try:
        # LCEL 체인을 통한 응답 생성
        response = planning_chain.invoke({
            "chat_history": history_messages,
            "user_input": message
        })
        return response

    except Exception as e:
        error_message = f"""죄송합니다. 요청을 처리하는 중 오류가 발생했습니다.

🔧 **오류 정보**: {str(e)}

📞 **해결 방법**:
1. 잠시 후 다시 시도해 주세요
2. 메시지를 더 간단하게 작성해 보세요
3. 문제가 지속되면 새로고침 후 다시 시도해 주세요

💡 **도움이 필요하시면** 구체적인 여행 계획 요청을 다시 해주세요!"""
        return error_message

# Gradio ChatInterface 구성
demo = gr.ChatInterface(
    fn=travel_planning_assistant,
    type="messages",
    title="🌟 맞춤형 여행 일정 계획 어시스턴트",
    description="""
    🗺️ **나만의 특별한 여행을 계획해드립니다!**

    여행지, 기간, 예산, 선호도를 알려주시면 완벽한 맞춤형 일정을 제안해드립니다.
    현지 문화, 숨은 명소, 실용적인 팁까지 모든 것을 고려한 전문적인 여행 계획을 경험해보세요.
    """,
    analytics_enabled=False,
    examples=[
        "제주도 2박 3일 힐링 여행을 계획하고 싶어요. 자연 풍경과 맛집을 중심으로 추천해주세요.",
        "부산에서 친구들과 1박 2일 먹방 여행! 현지 맛집과 야시장을 중심으로 일정을 짜주세요.",
        "서울 근교에서 아이들과 함께하는 가족 당일치기 여행을 추천해주세요.",
        "유럽 배낭여행 3주 일정을 계획 중입니다. 동유럽 위주로 예산은 300만원 정도예요.",
        "일본 도쿄 4박 5일 여행인데, 애니메이션과 팝컬처에 관심이 많아요.",
        "강릉에서 바다가 보이는 카페와 해변을 중심으로 한 로맨틱한 1박 2일 여행을 원해요."
    ],
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    """
)

# 인터페이스 실행
if __name__ == "__main__":
    demo.launch()
```

## 🔍 고급 기능과 최적화

### 1. 응답 품질 향상 기법

#### 토큰 사용량 최적화
```python
# 토큰 사용량 모니터링
def monitor_token_usage(response_obj):
    if hasattr(response_obj, 'response_metadata'):
        usage = response_obj.response_metadata.get('token_usage', {})
        print(f"토큰 사용량 - 입력: {usage.get('prompt_tokens', 0)}, "
              f"출력: {usage.get('completion_tokens', 0)}, "
              f"총계: {usage.get('total_tokens', 0)}")
```

#### 스트리밍 응답 구현
```python
def streaming_travel_planner(message, history):
    # 스트리밍 체인 구성
    streaming_chain = prompt | model.stream | StrOutputParser()

    history_messages = []
    for msg in history:
        if msg['role'] == "user":
            history_messages.append(HumanMessage(content=msg['content']))
        elif msg['role'] == "assistant":
            history_messages.append(AIMessage(content=msg['content']))

    # 스트리밍 응답 생성
    partial_response = ""
    for chunk in streaming_chain.stream({
        "chat_history": history_messages,
        "user_input": message
    }):
        partial_response += chunk
        yield partial_response
```

### 2. 에러 처리와 사용성 개선

#### 포괄적 에러 처리
```python
def robust_travel_planner(message, history):
    try:
        # 입력 검증
        if not message.strip():
            return "여행 계획에 대해 궁금한 것을 알려주세요!"

        # 체인 실행
        response = planning_chain.invoke({
            "chat_history": history_messages,
            "user_input": message
        })

        # 응답 검증
        if not response.strip():
            return "죄송합니다. 다시 한 번 질문해 주세요."

        return response

    except Exception as e:
        logger.error(f"Travel planner error: {e}")
        return "일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
```

### 3. 성능 모니터링

#### Langfuse 통합 모니터링
```python
# 성능 메트릭 수집
def collect_metrics(start_time, response, user_message):
    end_time = time.time()
    processing_time = end_time - start_time

    metrics = {
        "processing_time": processing_time,
        "message_length": len(user_message),
        "response_length": len(response),
        "timestamp": datetime.now().isoformat()
    }

    # Langfuse로 메트릭 전송
    langfuse_handler.log_metrics(metrics)
```

## 📚 참고 자료

### 공식 문서
- [Gradio ChatInterface](https://gradio.app/docs/chatinterface) - 공식 ChatInterface 가이드
- [LangChain Memory](https://python.langchain.com/docs/concepts/memory/) - 메모리 관리 문서
- [OpenAI API](https://platform.openai.com/docs/api-reference/chat) - Chat Completions API

### 학습 자료
- [Gradio 쿡북](https://github.com/gradio-app/gradio/tree/main/demo) - 다양한 예제 모음
- [LangChain Gradio 통합](https://python.langchain.com/docs/integrations/tools/gradio_tools/) - 통합 가이드
- [멀티모달 챗봇 구현](https://huggingface.co/spaces) - HuggingFace 예제

### 개발 도구
- [Gradio Hub](https://gradio.app/) - 온라인 배포 플랫폼
- [HuggingFace Spaces](https://huggingface.co/spaces) - 무료 호스팅
- [Langfuse](https://langfuse.com/) - LLM 옵저빌리티

### 추가 학습
- UI/UX 디자인 패턴과 사용자 경험 최적화
- 실시간 채팅과 WebSocket 통신
- 멀티모달 AI와 이미지/문서 처리
- 프로덕션 배포와 스케일링 전략