# PRJ04_W1_005 - LLM 파인튜닝 실습 Part1: Runpod 환경 설정 및 Unsloth 파인튜닝

## 📚 학습 목표

이 가이드를 완료하면 다음을 할 수 있습니다:

1. **Runpod GPU 서버 설정**: VSCode/Cursor로 원격 GPU 서버 연결 및 작업 환경 구축
2. **GPU 환경 이해**: CUDA, VRAM 등 GPU 학습 환경의 핵심 개념 파악
3. **Unsloth 활용**: 메모리 효율적인 LLM 파인튜닝 라이브러리 사용
4. **LoRA 파인튜닝**: 적은 리소스로 대규모 언어 모델을 특정 도메인에 맞게 학습
5. **모델 배포**: 학습된 모델을 Hugging Face Hub에 업로드하여 공유
6. **성능 비교**: 파인튜닝 전후 모델의 응답 품질 차이 확인

---

## 🎯 핵심 개념

### 1. Runpod와 클라우드 GPU

**Runpod**는 시간 단위로 GPU를 임대할 수 있는 클라우드 서비스입니다.

**장점:**
- 강력한 GPU (RTX 4090, A6000 등)를 저렴하게 사용
- 시간당 $0.2~$1 정도 (개인 GPU 구매 대비 훨씬 저렴)
- 설치 없이 즉시 사용 가능
- 다양한 GPU 선택 가능

**Pod (포드):**
- Runpod에서 제공하는 가상 서버
- GPU가 장착된 컴퓨터를 시간 단위로 임대
- 딥러닝 학습에 필요한 연산 능력 제공

### 2. SSH (Secure Shell)

**SSH**는 원격 서버에 안전하게 접속하기 위한 프로토콜입니다.

**SSH 키 페어:**
- **공개 키(Public Key)**: 서버에 등록 (자물쇠)
- **개인 키(Private Key)**: 본인만 보관 (열쇠)

**작동 원리:**
```
[내 컴퓨터] --- 개인 키로 인증 ---> [Runpod 서버]
                                      (공개 키로 검증)
```

### 3. GPU와 CUDA

**GPU (Graphics Processing Unit):**
- 원래 그래픽 처리용으로 설계
- 수천 개의 작은 코어가 병렬로 작동
- 딥러닝의 행렬 연산에 최적화
- CPU 대비 10~100배 빠른 학습 속도

**CUDA:**
- NVIDIA GPU를 프로그래밍하기 위한 플랫폼
- PyTorch, TensorFlow가 CUDA를 사용
- GPU 메모리(VRAM)가 클수록 큰 모델 학습 가능

**VRAM (Video RAM):**
- GPU 전용 메모리
- 모델 크기와 배치 크기를 결정하는 중요한 요소
- 예: RTX 4090 (24GB), A6000 (48GB)

### 4. 파인튜닝 개념

**사전 학습(Pre-training):**
- 대규모 텍스트 데이터로 모델을 학습
- 일반적인 언어 패턴과 지식 습득
- 예: GPT, Llama는 인터넷의 방대한 텍스트로 학습

**파인튜닝(Fine-tuning):**
- 사전 학습된 모델을 특정 작업/도메인에 맞게 추가 학습
- 적은 데이터로도 효과적
- 예: ETF 정보에 특화된 모델 만들기

### 5. LoRA (Low-Rank Adaptation)

**전통적 파인튜닝의 문제:**
- 모든 파라미터 업데이트 → 메모리 많이 필요
- 학습 시간 오래 걸림
- 모델 크기만큼의 저장 공간 필요

**LoRA의 해결책:**
- 원본 모델은 고정(freeze)
- 작은 어댑터(adapter) 레이어만 학습
- 메모리 사용량 80% 감소
- 학습 속도 2-3배 향상
- 어댑터만 저장 (수십 MB vs 수십 GB)

**LoRA 원리:**
```
원본 가중치: W (고정)
LoRA 업데이트: ΔW = B × A (학습)
최종 가중치: W' = W + ΔW

여기서 B와 A는 낮은 rank의 행렬
```

**LoRA 파라미터:**
- `r` (rank): 낮을수록 파라미터 적음, 일반적으로 8-64
- `lora_alpha`: 학습률 스케일링
- `target_modules`: 어떤 레이어에 LoRA 적용할지

### 6. Quantization (양자화)

모델 가중치를 낮은 정밀도로 저장하는 기법입니다.

**정밀도 비교:**
- FP32 (32비트 부동소수점): 기본
- FP16 (16비트): 메모리 1/2
- INT8 (8비트 정수): 메모리 1/4
- INT4 (4비트 정수): 메모리 1/8

**장점:**
- 메모리 사용량 대폭 감소
- 추론 속도 향상
- 성능 손실 최소 (1-2%)
- 더 큰 모델을 같은 GPU에서 실행 가능

---

## 🔧 환경 설정

### 1. Runpod Pod에 VSCode/Cursor로 연결

#### 준비물
- [ ] Windows 10/11, Mac, 또는 Linux
- [ ] VSCode 또는 Cursor 설치
- [ ] Runpod 계정 생성 (https://runpod.io)

#### 전체 과정 (5단계)

```
1단계: SSH 키 만들기
2단계: Runpod에 키 등록
3단계: Pod 배포
4단계: VSCode/Cursor 연결
5단계: 작업 시작
```

#### 1단계: SSH 키 만들기

**Windows PowerShell:**
```powershell
# PowerShell 열기: 시작 메뉴 → "PowerShell" 검색

# SSH 키 생성
ssh-keygen -t ed25519 -f C:\Users\$env:USERNAME\.ssh\id_ed25519

# 질문이 나오면 그냥 Enter 두 번 누르기
# Enter passphrase: [Enter]
# Enter same passphrase again: [Enter]

# SSH 키 확인
dir C:\Users\$env:USERNAME\.ssh\
```

**Mac/Linux Terminal:**
```bash
# 터미널 열기
# Mac: Cmd + Space → "Terminal"
# Linux: Ctrl + Alt + T

# SSH 키 생성
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519

# 질문이 나오면 그냥 Enter 두 번 누르기

# SSH 키 확인
ls -la ~/.ssh/
```

**생성된 파일:**
- `id_ed25519`: 개인 키 (절대 공유 금지!)
- `id_ed25519.pub`: 공개 키 (서버에 등록)

#### 2단계: Runpod에 키 등록

**공개 키 복사:**

Windows:
```powershell
type C:\Users\$env:USERNAME\.ssh\id_ed25519.pub
```

Mac/Linux:
```bash
cat ~/.ssh/id_ed25519.pub
```

**출력 예시:**
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAA... your_email@example.com
```

**Runpod에 등록:**
1. https://www.runpod.io/console/user/settings 접속
2. **SSH Public Keys** 섹션 찾기
3. 복사한 공개 키 붙여넣기
4. **Update public key** 클릭

#### 3단계: Pod 배포

1. https://www.runpod.io/console/pods 접속
2. **+ Deploy** 클릭
3. 템플릿 선택 (예: **RunPod PyTorch**)
4. **SSH Terminal Access** 체크 확인
5. GPU 선택 (예: RTX 4090, A6000)
6. **Deploy** 클릭
7. 1-2분 대기 (상태가 **Running**이 될 때까지)

**GPU 선택 가이드:**
| GPU | VRAM | 권장 모델 크기 | 시간당 비용 |
|-----|------|-------------|----------|
| RTX 4090 | 24GB | 7B-13B | $0.3-0.5 |
| A6000 | 48GB | 13B-30B | $0.5-0.8 |
| A100 | 80GB | 30B-70B | $1.5-2.0 |

#### 4단계: VSCode 연결

**Remote-SSH 확장 설치:**
1. VSCode 실행
2. `Ctrl+Shift+X` (Mac: `Cmd+Shift+X`)
3. **"Remote - SSH"** 검색
4. **Install** 클릭

**Pod 정보 확인:**
1. 배포한 Pod 클릭
2. **Connect** 버튼 클릭
3. **SSH** 탭 선택
4. 명령어 복사

**명령어 예시:**
```bash
# Mac/Linux
ssh root@123.456.789.80 -p 12345 -i ~/.ssh/id_ed25519

# Windows (경로 수정 필요)
ssh root@123.456.789.80 -p 12345 -i C:\Users\YourName\.ssh\id_ed25519
```

**VSCode에서 연결:**
1. `Ctrl+Shift+P` (Mac: `Cmd+Shift+P`)
2. **Remote-SSH: Connect to Host** 입력
3. **Add New SSH Host** 선택
4. 복사한 명령어 붙여넣기
5. Enter
6. 다시 `Ctrl+Shift+P` → **Remote-SSH: Connect to Host**
7. 방금 추가한 IP 주소 선택
8. **Linux** 선택
9. **Continue** 클릭
10. 연결 대기 (처음에는 1-2분 소요)

#### 5단계: 작업 시작

1. VSCode 좌측 **Open Folder** 클릭
2. `/workspace` 입력
3. **OK** 클릭
4. 작업 시작!

---

## 💻 단계별 구현 가이드

### 단계 1: GPU 및 환경 확인

#### 1.1 GPU 정보 확인

```python
# GPU 확인
# 현재 시스템에 사용 가능한 GPU 정보를 확인합니다.
!nvidia-smi
```

**출력 해석:**
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 565.57.01              Driver Version: 565.57.01      CUDA Version: 12.7     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                              | Bus-Id        | Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX A6000                  | 00000000:0D:00.0 Off |                  Off |
| 30%   36C    P8             20W / 300W |       2MiB / 49140MiB |      0%      Default |
+-----------------------------------------------------------------------------------------+
```

**주요 정보:**
- **GPU Name**: RTX A6000
- **VRAM**: 49140MiB (약 48GB)
- **GPU-Util**: 0% (현재 사용률)
- **Memory-Usage**: 2MiB / 49140MiB (거의 비어있음)

#### 1.2 필수 패키지 설치

**Unsloth**: LLM 파인튜닝 최적화 라이브러리
- 메모리 사용량 감소 (최대 80%)
- 학습 속도 향상 (최대 2배)
- LoRA, QLoRA 등 효율적인 파인튜닝 지원

```bash
# Runpod 환경에서 Unsloth 설치
pip install unsloth pandas python-dotenv ipykernel
```

**설치되는 주요 라이브러리:**
- `unsloth`: LLM 파인튜닝
- `pandas`: 데이터 처리
- `python-dotenv`: 환경 변수 관리
- `ipykernel`: Jupyter 커널

#### 1.3 Hugging Face 인증

**Hugging Face Hub**는 머신러닝 모델과 데이터셋을 공유하는 플랫폼입니다.

**Token 생성 방법:**
1. https://huggingface.co 접속
2. Settings → Access Tokens
3. New Token → Write 권한 선택
4. 생성된 토큰 복사

```python
import os
from dotenv import load_dotenv
from huggingface_hub import login

# 환경 변수 로드
load_dotenv()

# 토큰 설정 (직접 입력 또는 .env 파일에서)
if "HUGGINGFACE_TOKEN" not in os.environ or not os.environ["HUGGINGFACE_TOKEN"]:
    os.environ["HUGGINGFACE_TOKEN"] = "your_token_here"  # 실제 토큰으로 변경
    print("✅ 토큰 직접 설정")
else:
    print("✅ .env에서 토큰 로드")

# Hugging Face 로그인
try:
    login(token=os.environ["HUGGINGFACE_TOKEN"], add_to_git_credential=True)
    print("✅ Hugging Face 로그인 완료!")
except Exception as e:
    print(f"⚠️ Git credential 저장 실패 (무시 가능): {e}")
    login(token=os.environ["HUGGINGFACE_TOKEN"], add_to_git_credential=False)
    print("✅ Hugging Face 로그인 완료 (git credential 제외)!")
```

#### 1.4 기본 라이브러리 및 PyTorch 확인

```python
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
import torch
import warnings
warnings.filterwarnings('ignore')  # 경고 메시지 숨기기

# PyTorch 및 CUDA 정보 출력
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    # Compute Capability: GPU의 기능 수준
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    # VRAM: GPU 메모리
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```

**출력 예시:**
```
PyTorch: 2.9.0+cu128
CUDA available: True
GPU: NVIDIA RTX A6000
CUDA Capability: (8, 6)
VRAM: 47.43 GB
```

---

### 단계 2: 데이터 준비

#### 2.1 작업 디렉토리 설정

```python
# 작업 디렉토리 설정 (Runpod의 영구 스토리지)
WORKSPACE = "/workspace"
DATA_DIR = f"{WORKSPACE}/etf_data"
MODEL_DIR = f"{WORKSPACE}/models"

import os
# exist_ok=True: 디렉토리가 이미 존재해도 에러 발생하지 않음
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"작업 디렉토리: {WORKSPACE}")
print(f"데이터: {DATA_DIR}")
print(f"모델: {MODEL_DIR}")
```

**Runpod 디렉토리 구조:**
- `/workspace`: 영구 스토리지 (Pod 재시작 후에도 유지)
- `/root`: 임시 스토리지 (Pod 종료 시 삭제)

#### 2.2 Hugging Face 데이터셋 로드

**Train/Test Set:**
- **Train Set**: 모델이 패턴을 학습하는 데 사용
- **Test Set**: 모델의 일반화 성능을 평가

```python
from datasets import load_dataset

# Hugging Face username 설정
username = 'your_username'  # 실제 사용자명으로 변경

# LLM 데이터셋 로드
alpaca_dataset = load_dataset(f"{username}/etf-alpaca-llm-v1")

print(f"✅ LLM 데이터셋 로드 완료!")
print(f"   Train: {len(alpaca_dataset['train'])}개")
print(f"   Test: {len(alpaca_dataset['test'])}개")

# 샘플 확인
print("\nLLM 샘플:")
print(alpaca_dataset['train'][0])
```

**데이터셋 형식 (Alpaca):**
```python
{
    'instruction': '다음 ETF의 기본 정보를 제공해주세요.',
    'input': 'KB RISE 국채선물10년인버스증권상장지수투자신탁(채권-파생형) (종목코드: 295020)',
    'output': 'KB RISE 국채선물10년인버스증권상장지수투자신탁(채권-파생형)은 케이비자산운용에서 운용하는 ETF입니다...'
}
```

---

### 단계 3: LLM 모델 로드 및 설정

#### 3.1 Unsloth로 모델 로드

```python
from unsloth import FastLanguageModel

# 모델 선택 (사전 양자화 버전)
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"  # 4bit 양자화 모델
MAX_SEQ_LENGTH = 4096  # 최대 토큰 길이

# 모델 로드
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit = True,      # 4비트 양자화 활성화
    full_finetuning = False,  # LoRA 사용 (전체 파인튜닝 비활성화)
)
```

**모델 선택 가이드:**
| 모델 | 파라미터 | VRAM | 특징 |
|------|---------|------|-----|
| Llama-3.1-8B-4bit | 8B | ~6GB | 일반 작업에 적합 |
| Llama-3.1-13B-4bit | 13B | ~10GB | 복잡한 작업 |
| Llama-3.1-70B-4bit | 70B | ~40GB | 최고 성능 |

**`bnb-4bit` 모델:**
- `bnb`: bitsandbytes (양자화 라이브러리)
- `4bit`: 4비트 정밀도
- 사전 양자화되어 빠른 로딩

#### 3.2 프롬프트 템플릿 및 데이터 전처리

**Alpaca 프롬프트 형식:**
```
### Instruction:
[작업 지시]

### Input:
[구체적인 입력]

### Response:
[모델의 출력]
```

**EOS Token (End of Sequence):**
- 문장/대화의 끝을 표시하는 특수 토큰
- 추가하지 않으면 모델이 계속 생성함
- 학습 시 반드시 포함

```python
# 프롬프트 템플릿 정의
alpaca_prompt = """Below is an instruction that describes a task, paired with an input. Write a response.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token  # 문장 종료 토큰

def formatting_prompts_func(examples):
    """
    데이터를 프롬프트 형식으로 변환

    Args:
        examples: instruction, input, output을 포함한 배치 데이터

    Returns:
        프롬프트 형식의 텍스트 리스트
    """
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []

    for instruction, input, output in zip(instructions, inputs, outputs):
        # EOS_TOKEN을 반드시 추가
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)

    return {"text": texts}

# 데이터셋 전처리 (batched=True: 배치 단위 처리)
train_data = alpaca_dataset['train'].map(formatting_prompts_func, batched=True)
test_data  = alpaca_dataset['test'].map(formatting_prompts_func, batched=True)

print(f"✅ 데이터 전처리 완료")
print(f"   Train: {len(train_data)}개")
print(f"   Test: {len(test_data)}개")
```

---

### 단계 4: 학습 전 모델 테스트

파인튜닝 전 모델의 성능을 확인합니다.

**Inference(추론) 모드:**
- 학습이 아닌 예측/생성 모드
- 그래디언트 계산 비활성화 (메모리 절약)
- 드롭아웃 등 학습용 기능 비활성화

```python
# 모델을 추론 모드로 전환
FastLanguageModel.for_inference(model)

def test_model(instruction, input_text=""):
    """
    모델 테스트 함수

    Args:
        instruction: 작업 지시
        input_text: 입력 데이터

    Returns:
        모델의 응답
    """
    # 프롬프트 생성 (Response 부분은 비워둠)
    prompt = alpaca_prompt.format(instruction, input_text, "")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,       # 최대 512 토큰 생성
        temperature=0.7,          # 다양성 (0.0=결정적, 1.0=창의적)
        top_p=0.9,                # nucleus sampling
        repetition_penalty=1.1    # 반복 억제
    )

    # 생성된 텍스트 디코딩
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Response 부분만 추출
    return response.split("### Response:")[-1].strip()

# 테스트 케이스
test_cases = [
    ("다음 ETF의 기본 정보를 제공해주세요.", "TIGER 200"),
    ("이 ETF의 투자 정보를 알려주세요.", "KODEX 레버리지"),
    ("어떤 ETF를 추천하나요?", "안정적인 투자를 원합니다")
]

print("\n" + "="*60)
print("학습 전 모델 테스트")
print("="*60)
for inst, inp in test_cases:
    print(f"\n[질문] {inst}")
    print(f"[입력] {inp}")
    print(f"[답변] {test_model(inst, inp)}")
    print("-"*60)
```

**예상 출력 (학습 전):**
```
============================================================
학습 전 모델 테스트
============================================================

[질문] 다음 ETF의 기본 정보를 제공해주세요.
[입력] TIGER 200
[답변] TIGER 200 is a mutual fund that tracks the KOSPI 200 index...
(영어로 답변하거나 ETF 정보를 정확히 알지 못함)
------------------------------------------------------------
```

---

### 단계 5: LoRA 어댑터 추가 및 학습 설정

#### 5.1 학습 모드 전환 및 LoRA 추가

```python
# 학습 모드로 전환
FastLanguageModel.for_training(model)

# LoRA 어댑터 추가
model = FastLanguageModel.get_peft_model(
    model,
    r = 64,  # LoRA rank (파라미터 수 결정)
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention 레이어
        "gate_proj", "up_proj", "down_proj",     # Feed-Forward 레이어
    ],
    lora_alpha = 64,  # 스케일링 계수 (보통 r과 같게)
    lora_dropout = 0,  # 드롭아웃 (0 = 최적화)
    bias = "none",     # 바이어스 학습 안 함
    use_gradient_checkpointing = "unsloth",  # 메모리 절약 (30%)
    random_state = 3407,  # 재현성을 위한 시드
)

print("✅ LoRA 어댑터 추가 완료")
```

**LoRA 파라미터 설명:**

| 파라미터 | 설명 | 권장값 |
|---------|------|--------|
| `r` | LoRA 행렬의 차원 (낮을수록 파라미터 감소) | 8-64 |
| `lora_alpha` | 스케일링 계수 (보통 r과 동일) | r과 같게 |
| `target_modules` | LoRA를 적용할 레이어 | q/k/v/o_proj, gate/up/down_proj |
| `lora_dropout` | 과적합 방지 드롭아웃 | 0 (Unsloth 권장) |
| `use_gradient_checkpointing` | 메모리 절약 기법 | "unsloth" |

#### 5.2 학습 파라미터 설정

**주요 하이퍼파라미터:**

**배치(Batch):**
- `per_device_train_batch_size`: GPU당 배치 크기
- `gradient_accumulation_steps`: 그래디언트 누적 스텝
- 실제 배치 크기 = batch_size × accumulation_steps

**에포크(Epoch):**
- 전체 데이터를 한 번 학습하는 단위
- `num_train_epochs`: 총 에포크 수

**학습률(Learning Rate):**
- 파라미터 업데이트 크기
- LoRA: 2e-4 ~ 5e-4 권장

**Warmup:**
- 초반에 학습률을 서서히 증가
- 학습 안정성 향상

```python
from trl import SFTTrainer
from transformers import TrainingArguments

# 학습 설정
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_data,
    eval_dataset=test_data,
    dataset_text_field="text",  # 텍스트가 저장된 필드명
    max_seq_length=MAX_SEQ_LENGTH,
    args=TrainingArguments(
        # 배치 크기
        per_device_train_batch_size=4,   # GPU 메모리에 따라 조정
        gradient_accumulation_steps=4,   # 실제 배치 = 4 × 4 = 16

        # 학습 기간
        warmup_steps=10,        # 초반 워밍업
        num_train_epochs=2,     # 총 2 에포크

        # 최적화
        learning_rate=2e-4,            # 학습률
        optim="adamw_8bit",            # 8비트 AdamW (메모리 절약)
        weight_decay=0.01,             # 가중치 감쇠 (정규화)
        lr_scheduler_type="cosine",    # 코사인 스케줄러

        # Mixed Precision
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),

        # 로깅 및 저장
        logging_steps=1,              # 매 스텝마다 로그
        save_strategy="epoch",        # 에포크마다 저장
        eval_strategy="epoch",        # 에포크마다 평가
        load_best_model_at_end=True,  # 최고 성능 모델 로드

        # 출력 디렉토리
        output_dir=f"{MODEL_DIR}/llm_outputs",
        seed=3407,
        report_to="none",  # wandb 등 비활성화
    ),
)

print("🚀 학습 시작...")
```

---

### 단계 6: 모델 학습 실행

```python
import time
start_time = time.time()

# 학습 실행
trainer.train()

elapsed = time.time() - start_time
print(f"✅ 학습 완료! (소요 시간: {elapsed/60:.2f}분)")
```

**학습 과정 모니터링:**
```
Epoch 1/2:
Step 100/372: loss=1.234, lr=0.0002
Step 200/372: loss=0.987, lr=0.00015
...

Epoch 2/2:
Step 300/372: loss=0.756, lr=0.0001
Step 372/372: loss=0.523, lr=0.00005

Training completed!
```

**학습 시간 예상:**
- RTX 4090 (24GB): 약 10-15분
- A6000 (48GB): 약 15-20분
- 데이터셋 크기와 에포크 수에 따라 변동

---

### 단계 7: 학습 후 모델 테스트

```python
# 모델을 추론 모드로 전환
FastLanguageModel.for_inference(model)

# 동일한 테스트 케이스로 재테스트
print("\n" + "="*60)
print("학습 후 모델 테스트")
print("="*60)
for inst, inp in test_cases:
    print(f"\n[질문] {inst}")
    print(f"[입력] {inp}")
    print(f"[답변] {test_model(inst, inp)}")
    print("-"*60)
```

**예상 출력 (학습 후):**
```
============================================================
학습 후 모델 테스트
============================================================

[질문] 다음 ETF의 기본 정보를 제공해주세요.
[입력] TIGER 200
[답변] TIGER 200은 삼성자산운용에서 운용하는 ETF입니다.
기초지수는 KOSPI 200이며, 총보수는 연 0.15%입니다.
------------------------------------------------------------
```

**개선 사항:**
- ✅ 한국어로 정확하게 답변
- ✅ ETF 정보를 구체적으로 제공
- ✅ 문장 구조가 자연스러움

---

### 단계 8: 모델 저장 및 배포

#### 8.1 로컬 저장

**저장 방식:**

1. **LoRA 어댑터만 저장** (권장)
   - 가장 작은 크기 (수십 MB)
   - 원본 모델과 함께 로드 필요

2. **병합된 16-bit 모델**
   - 원본 + LoRA를 하나로 병합
   - 독립적으로 사용 가능
   - 크기 증가 (수 GB)

3. **GGUF 형식**
   - llama.cpp, Ollama 등에서 사용
   - CPU에서도 실행 가능

```python
# 모델 저장 디렉토리
save_dir = f"{MODEL_DIR}/etf_llama_model"

# 1. LoRA 어댑터만 저장 (권장)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"✅ LoRA 어댑터 저장: {save_dir}")

# 2. 병합된 16-bit 모델 (선택사항)
# model.save_pretrained_merged(
#     f"{save_dir}_merged_16bit",
#     tokenizer,
#     save_method="merged_16bit"
# )

# 3. GGUF 형식 (Ollama용, 선택사항)
# model.save_pretrained_gguf(
#     f"{save_dir}_gguf",
#     tokenizer,
#     quantization_method=["q4_k_m", "q8_0"]
# )
```

#### 8.2 Hugging Face Hub 업로드

```python
# Hugging Face 업로드
repo_name = f"{username}/llama-8b-etf-finetuned"

# LoRA 어댑터 업로드
model.push_to_hub(
    repo_name,
    token=os.environ["HUGGINGFACE_TOKEN"],
    private=True  # 비공개 리포지토리
)
tokenizer.push_to_hub(
    repo_name,
    token=os.environ["HUGGINGFACE_TOKEN"],
    private=True
)

print(f"✅ 모델 업로드 완료: https://huggingface.co/{repo_name}")
```

**업로드된 파일:**
```
your_username/llama-8b-etf-finetuned/
├── adapter_config.json     # LoRA 설정
├── adapter_model.safetensors  # LoRA 가중치 (수십 MB)
├── tokenizer_config.json   # 토크나이저 설정
├── tokenizer.json          # 토크나이저
└── special_tokens_map.json # 특수 토큰
```

---

## 🎯 핵심 요약

### 1. Runpod 사용 플로우

```
SSH 키 생성 → Runpod 등록 → Pod 배포 → VSCode 연결 → 학습 시작
```

### 2. 파인튜닝 플로우

```
모델 로드 → 데이터 전처리 → LoRA 추가 → 학습 → 평가 → 저장/배포
```

### 3. 주요 개념 정리

| 개념 | 설명 | 핵심 값 |
|------|------|--------|
| **LoRA rank** | 어댑터 크기 | 8-64 |
| **Batch size** | GPU 메모리에 맞춰 조정 | 4-16 |
| **Learning rate** | LoRA 학습률 | 2e-4 ~ 5e-4 |
| **Quantization** | 4bit 양자화 | 메모리 1/8 |

### 4. 메모리 최적화 기법

- ✅ 4-bit 양자화: 메모리 75% 절감
- ✅ LoRA: 학습 파라미터 98% 감소
- ✅ Gradient Checkpointing: 메모리 30% 절감
- ✅ Gradient Accumulation: 작은 배치로 큰 배치 효과

### 5. 체크리스트

**학습 전:**
- [ ] GPU 확인 (VRAM 충분한지)
- [ ] 데이터셋 로드 및 확인
- [ ] 학습 전 모델 테스트

**학습 중:**
- [ ] Loss 감소 확인
- [ ] 메모리 사용량 모니터링
- [ ] 주기적인 체크포인트 저장

**학습 후:**
- [ ] 학습 후 모델 테스트
- [ ] 성능 비교 (before vs after)
- [ ] 모델 저장 및 업로드

---

## 📌 다음 단계

Part2에서는 **임베딩 모델 파인튜닝**을 다룹니다:
- Sentence Transformers 파인튜닝
- Matryoshka Loss
- 임베딩 유사도 테스트
- 실습 문제 및 실무 활용 예시

---

이 가이드를 통해 **Runpod GPU 서버에서 Unsloth를 활용한 LLM 파인튜닝**의 전체 과정을 익힐 수 있습니다!
