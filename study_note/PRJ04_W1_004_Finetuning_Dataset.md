# PRJ04_W1_004 - 파인튜닝(Fine-tuning) 데이터셋 생성 가이드

## 📚 학습 목표

이 가이드를 완료하면 다음을 할 수 있습니다:

1. **파인튜닝 개념 이해**: LLM과 임베딩 모델의 파인튜닝 차이점 파악
2. **Alpaca 형식 데이터셋 생성**: instruction-input-output 구조의 LLM 학습 데이터 제작
3. **임베딩 데이터셋 생성**: positive pairs와 hard negatives를 포함한 유사도 학습 데이터 구축
4. **데이터 증강 기법 활용**: LLM 생성, 역번역, 패러프레이징으로 데이터 확장
5. **DPO 데이터셋 생성**: Direct Preference Optimization용 prompt-chosen-rejected 데이터 제작
6. **Hugging Face 활용**: 데이터셋 업로드 및 버전 관리

---

## 🎯 핵심 개념

### 1. 파인튜닝(Fine-tuning)이란?

파인튜닝은 **사전 학습된 모델을 특정 도메인이나 작업에 맞게 추가 학습**시키는 과정입니다.

**장점:**
- 도메인 특화 지식 습득 (예: 금융, 의료, 법률)
- 응답 스타일 및 포맷 커스터마이징
- 일반 모델보다 높은 정확도
- 처음부터 학습하는 것보다 비용 효율적

### 2. LLM vs 임베딩 모델 파인튜닝

| 특징 | LLM 파인튜닝 | 임베딩 모델 파인튜닝 |
|------|-------------|-------------------|
| **목적** | 텍스트 생성, 질의응답 | 의미적 유사도 계산, 검색 |
| **출력** | 자연어 텍스트 | 벡터 임베딩 (고차원 실수 벡터) |
| **주요 사용처** | 챗봇, 요약, 번역, 문서 생성 | RAG 시스템, 시맨틱 검색, 추천 시스템 |
| **데이터 형식** | Q&A 쌍, instruction-output | 유사 문장 쌍, triplets |
| **학습 방법** | Next token prediction | Contrastive learning, Triplet loss |
| **하드웨어 요구사항** | 고성능 GPU (16GB+ VRAM) | 중급 GPU (8GB+ VRAM) |

### 3. 주요 데이터셋 형식

#### (1) Alpaca 형식 (LLM용)

```python
{
    "instruction": "다음 ETF의 기본 정보를 제공해주세요.",
    "input": "ETF 이름 (종목코드: XXXX)",
    "output": "상세한 정보 및 설명..."
}
```

#### (2) 임베딩 형식

```python
{
    "sentence1": "검색 쿼리 또는 질문",
    "sentence2": "관련 문서 또는 답변",
    "label": 1  # 1=유사, 0=유사하지 않음
}
```

#### (3) DPO 형식 (Direct Preference Optimization)

```python
{
    "prompt": "사용자 질문",
    "chosen": "선호되는 고품질 답변",
    "rejected": "거부되는 저품질 답변"
}
```

---

## 🔧 환경 설정

### 1. 필수 라이브러리 설치

```bash
# LLM 파인튜닝용 (Unsloth - 메모리 효율적인 파인튜닝 프레임워크)
pip install unsloth
# 또는 uv 사용
uv pip install unsloth

# 임베딩 모델 파인튜닝용
pip install -U "sentence-transformers>=3.0"

# 공통 라이브러리
pip install datasets accelerate torch pandas scikit-learn
pip install python-dotenv  # 환경변수 관리
pip install langchain-openai  # 데이터 증강용

# 역번역용
pip install -U deep-translator
```

### 2. 하드웨어 요구사항

**LLM 파인튜닝 (Unsloth)**
- GPU: NVIDIA GPU 16GB+ VRAM 권장
- 4-bit 양자화로 메모리 4배 절감 가능
- 무료 옵션: Google Colab (무료 T4), Kaggle Notebooks
- 유료 옵션: Runpod, Lambda Labs, AWS/GCP

**임베딩 모델 파인튜닝**
- GPU: 8GB+ VRAM
- CPU에서도 가능하지만 속도 느림
- Colab 무료 티어로도 충분

### 3. Hugging Face 설정

```python
import os
from dotenv import load_dotenv
from getpass import getpass
from huggingface_hub import login

# 환경 변수 로드
load_dotenv()

# Hugging Face 토큰 설정
if "HUGGINGFACE_TOKEN" not in os.environ:
    os.environ["HUGGINGFACE_TOKEN"] = getpass("Hugging Face Token: ")

# OpenAI API 키 설정 (데이터 증강용)
if "OPENAI_API_KEY" not in os.environ:
    use_openai = input("OpenAI API 사용? (y/n): ")
    if use_openai.lower() == 'y':
        os.environ["OPENAI_API_KEY"] = getpass("OpenAI API Key: ")

# Hugging Face 로그인
login(token=os.environ["HUGGINGFACE_TOKEN"], add_to_git_credential=True)

print("✅ 환경 설정 완료!")
```

---

## 📊 단계별 구현 가이드

### 단계 1: 데이터 수집 및 전처리

#### 1.1 데이터 로드 및 확인

```python
import pandas as pd

# CSV 파일에서 ETF 데이터 로드
df = pd.read_csv('data/etf_info.csv', encoding='cp949')

# 데이터 확인
print(f"데이터 크기: {df.shape}")
print(f"컬럼: {df.columns.tolist()}")
df.head()
```

**출력 예시:**
```
데이터 크기: (930, 17)
컬럼: ['표준코드', '단축코드', '한글종목명', ...]
```

#### 1.2 컬럼명 영문 변환

```python
# 한글 컬럼명을 영문으로 매핑
column_mapping = {
    '표준코드': 'standard_code',
    '단축코드': 'ticker',
    '한글종목명': 'name_kr',
    '한글종목약명': 'short_name_kr',
    '영문종목명': 'name_en',
    '상장일': 'listing_date',
    '기초지수명': 'base_index',
    '지수산출기관': 'index_provider',
    '추적배수': 'tracking_multiplier',
    '복제방법': 'replication_method',
    '기초시장분류': 'base_market_category',
    '기초자산분류': 'base_asset_category',
    '상장좌수': 'listed_shares',
    '운용사': 'manager',
    'CU수량': 'cu_quantity',
    '총보수': 'total_expense_ratio',
    '과세유형': 'tax_type'
}

# 컬럼명 변경
df = df.rename(columns=column_mapping)
```

#### 1.3 Train/Test 분할

```python
from sklearn.model_selection import train_test_split

# 80/20 분할 (과세 유형별 균등 분할)
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=1207,
    stratify=df['tax_type']  # 계층화 샘플링
)

print(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
print(f"Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

# 저장
train_df.to_csv('data/etf_train.csv', index=False)
test_df.to_csv('data/etf_test.csv', index=False)
```

---

### 단계 2: Alpaca 형식 데이터셋 생성

Alpaca 형식은 **instruction-input-output 구조**로, LLM이 사용자의 지시(instruction)와 입력(input)을 이해하고 적절한 출력(output)을 생성하도록 학습합니다.

#### 2.1 기본 데이터셋 생성 함수

```python
from datasets import Dataset

def create_alpaca_dataset(df):
    """
    ETF 데이터를 Alpaca 형식으로 변환
    4가지 패턴: 기본 정보, 투자 특징, 운용사 정보, 투자 적합성
    """
    alpaca_data = []

    for _, row in df.iterrows():
        # 패턴 1: 기본 정보
        alpaca_data.append({
            "instruction": "다음 ETF의 기본 정보를 제공해주세요.",
            "input": f"{row['name_kr']} (종목코드: {row['ticker']})",
            "output": f"{row['name_kr']}은 {row['manager']}에서 운용하는 ETF입니다. "
                     f"기초지수는 {row['base_index']}이며, "
                     f"총보수는 연 {row['total_expense_ratio']:.2f}%입니다. "
                     f"{pd.to_datetime(row['listing_date']).strftime('%Y년 %m월')}에 상장되었습니다."
        })

        # 패턴 2: 투자 특징
        alpaca_data.append({
            "instruction": "이 ETF의 특징을 알려주세요.",
            "input": f"{row['name_kr']}",
            "output": f"{row['name_kr']}는 {row['base_index']}를 추종하는 ETF로, "
                     f"{row['replication_method']} 방식으로 운용됩니다. "
                     f"기초자산은 {row['base_asset_category']}이며, "
                     f"과세 유형은 {row['tax_type']}입니다."
        })

        # 패턴 3: 운용사 정보
        alpaca_data.append({
            "instruction": "이 ETF의 운용사 정보를 알려주세요.",
            "input": f"{row['name_kr']}",
            "output": f"{row['name_kr']}는 {row['manager']}에서 운용하고 있습니다. "
                     f"총보수는 연 {row['total_expense_ratio']:.2f}%입니다."
        })

        # 패턴 4: 투자 적합성 (자산 분류별)
        suitability = {
            '주식': '시장 상승기에 유리하며, 변동성이 높아 리스크 관리가 필요합니다.',
            '채권': '안정적인 수익을 추구하며, 금리 변동에 영향을 받습니다.',
            '원자재': '인플레이션 헤지 수단으로 활용 가능하며, 시장 변동성이 큽니다.',
        }

        advice = suitability.get(
            row['base_asset_category'],
            '해당 자산군의 특성을 충분히 이해하고 투자하시기 바랍니다.'
        )

        alpaca_data.append({
            "instruction": "이 ETF는 어떤 투자자에게 적합한가요?",
            "input": f"{row['name_kr']}",
            "output": f"{row['name_kr']}는 {row['base_asset_category']} 자산에 투자하는 ETF입니다. "
                     f"{advice}"
        })

    return Dataset.from_list(alpaca_data)

# 데이터셋 생성
train_alpaca = create_alpaca_dataset(train_df)
test_alpaca = create_alpaca_dataset(test_df)

print(f"Alpaca 데이터: Train {len(train_alpaca)}, Test {len(test_alpaca)}")
```

**출력 예시:**
```
Alpaca 데이터: Train 2976, Test 744
```

#### 2.2 데이터셋 저장 및 로드

```python
import os
from pathlib import Path

# 저장 디렉토리 생성
output_dir = "datasets"
os.makedirs(output_dir, exist_ok=True)

# 저장
train_alpaca.save_to_disk(f"{output_dir}/train_alpaca")
test_alpaca.save_to_disk(f"{output_dir}/test_alpaca")

print("✅ 데이터셋 저장 완료")

# 로드
from datasets import load_from_disk

train_alpaca = load_from_disk(f"{output_dir}/train_alpaca")
test_alpaca = load_from_disk(f"{output_dir}/test_alpaca")
```

#### 2.3 Hugging Face에 업로드

```python
from datasets import DatasetDict

# Train과 Test를 하나의 데이터셋으로 구성
llm_alpaca_dataset = DatasetDict({
    'train': train_alpaca,
    'test': test_alpaca
})

# Hugging Face Hub에 업로드
llm_alpaca_dataset.push_to_hub(
    "your_username/etf-alpaca-llm-v1",  # 사용자명 변경 필요
    private=True,  # 비공개 설정
)

print("✅ Hugging Face 업로드 완료!")
```

#### 2.4 Hugging Face에서 다운로드

```python
from datasets import load_dataset

# 데이터셋 다운로드
train_alpaca = load_dataset("your_username/etf-alpaca-llm-v1", split="train")
test_alpaca = load_dataset("your_username/etf-alpaca-llm-v1", split="test")

print(f"Train: {len(train_alpaca)}, Test: {len(test_alpaca)}")
```

---

### 단계 3: 임베딩 데이터셋 생성

임베딩 모델은 **문장 간 유사도**를 학습합니다. Positive pairs(유사한 쌍)와 Negative pairs(유사하지 않은 쌍)를 제공하여 학습합니다.

#### 3.1 Hard Negatives를 포함한 데이터셋 생성

**Hard Negatives**란 겉보기에는 유사해 보이지만 실제로는 다른 예시로, 모델이 미묘한 차이를 학습하도록 돕습니다.

```python
from itertools import combinations

def create_embedding_dataset(df):
    """
    임베딩 학습용 데이터셋 생성
    - Positive pairs: 같은 ETF의 다른 표현
    - Hard negatives: 같은 운용사/지수지만 다른 ETF
    - Easy negatives: 완전히 다른 카테고리
    """
    pairs = []

    # 1. Positive Pairs: 같은 ETF의 다른 표현
    for _, row in df.iterrows():
        # ETF 이름 ↔ 기초지수 설명
        pairs.append({
            "sentence1": f"{row['name_kr']}",
            "sentence2": f"{row['base_index']}를 추종하는 {row['manager']} 운용 ETF",
            "label": 1  # 유사함
        })

        # 종목코드 ↔ ETF 이름
        pairs.append({
            "sentence1": f"종목코드 {row['ticker']}",
            "sentence2": f"{row['name_kr']} ETF",
            "label": 1
        })

    # 2. Hard Negatives: 같은 운용사지만 다른 ETF
    for manager in df['manager'].unique():
        manager_etfs = df[df['manager'] == manager]

        if len(manager_etfs) > 1:
            # 같은 운용사의 ETF 조합 생성
            for (idx1, row1), (idx2, row2) in combinations(manager_etfs.iterrows(), 2):
                pairs.append({
                    "sentence1": f"{row1['name_kr']}",
                    "sentence2": f"{row2['name_kr']}",
                    "label": 0  # 유사하지 않음 (같은 운용사지만 다른 상품)
                })

    # 3. Hard Negatives: 같은 기초지수지만 다른 ETF
    for index in df['base_index'].unique():
        index_etfs = df[df['base_index'] == index]

        if len(index_etfs) > 1:
            for (idx1, row1), (idx2, row2) in combinations(index_etfs.iterrows(), 2):
                pairs.append({
                    "sentence1": f"종목코드 {row1['ticker']}",
                    "sentence2": f"{row2['name_kr']}",
                    "label": 0  # 다른 ETF
                })

    # 4. Easy Negatives: 완전히 다른 카테고리
    for _, row1 in df.iterrows():
        # 다른 자산군 선택
        diff_category = df[df['base_asset_category'] != row1['base_asset_category']]

        if len(diff_category) > 0:
            row2 = diff_category.sample(1).iloc[0]
            pairs.append({
                "sentence1": f"{row1['name_kr']}",
                "sentence2": f"{row2['name_kr']}",
                "label": 0
            })

    return Dataset.from_list(pairs)

# 데이터셋 생성
train_embedding = create_embedding_dataset(train_df)
test_embedding = create_embedding_dataset(test_df)

print(f"임베딩 데이터: Train {len(train_embedding)}, Test {len(test_embedding)}")
```

**출력 예시:**
```
임베딩 데이터: Train 39442, Test 3035
```

#### 3.2 Hugging Face에 업로드

```python
from datasets import DatasetDict

# 데이터셋 구성
embedding_dataset = DatasetDict({
    'train': train_embedding,
    'test': test_embedding
})

# 업로드
embedding_dataset.push_to_hub(
    "your_username/etf-embedding-v1",
    private=True,
)

print("✅ 임베딩 데이터셋 업로드 완료!")
```

---

### 단계 4: 데이터 증강 기법

데이터가 부족할 때 **고품질 데이터를 인공적으로 생성**하는 기법입니다.

#### 4.1 LLM을 활용한 Q&A 생성

OpenAI API를 사용하여 **전문가 수준의 질문-답변 쌍**을 자동 생성합니다.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List
import time

# Pydantic 모델 정의
class QnAPair(BaseModel):
    """질문-답변 쌍"""
    question: str = Field(description="투자자의 자연스러운 질문")
    answer: str = Field(description="전문가의 상세한 답변")

class ETFQnASet(BaseModel):
    """5가지 질문-답변 세트"""
    qna_pairs: List[QnAPair] = Field(description="질문-답변 목록")

# LLM 초기화
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.2
)

# 구조화된 출력
structured_llm = llm.with_structured_output(ETFQnASet)

# 프롬프트 템플릿
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 ETF 투자 전문가입니다.
    제공된 ETF 정보를 바탕으로 투자자가 물어볼 수 있는 자연스럽고 다양한 질문 5가지와
    각 질문에 대한 전문적이고 상세한 답변을 생성해주세요.

    답변은 다음을 포함해야 합니다:
    - 정확한 데이터 기반 정보
    - 투자자가 이해하기 쉬운 설명
    - 주의사항 및 리스크 안내
    - 실용적인 조언"""),
    ("human", """
    ETF 정보:
    {etf_info}

    다음 5가지 카테고리의 질문과 답변을 생성하세요:
    1. 기본 정보 (이름, 종목코드, 상장일 등)
    2. 투자 전략 (기초지수, 추종 방식, 포트폴리오)
    3. 비용 및 수익 (총보수, 배당, 세금)
    4. 위험 관리 (변동성, 추적오차, 주의사항)
    5. 투자 적합성 (목표 수익률, 투자 기간, 투자자 유형)
    """)
])

def generate_qa_data(df, batch_size=10):
    """OpenAI API로 고품질 Q&A 생성"""
    enhanced_data = []
    total_batches = (len(df) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx]

        print(f"배치 {batch_idx+1}/{total_batches} 처리 중...")

        # 배치 프롬프트 생성
        batch_prompts = []
        for _, row in batch_df.iterrows():
            etf_info = f"""
종목코드: {row['ticker']}
ETF 이름: {row['name_kr']}
운용사: {row['manager']}
기초지수: {row['base_index']}
총보수: {row['total_expense_ratio']:.2f}%
과세유형: {row['tax_type']}
상장일: {row['listing_date']}
            """.strip()

            batch_prompts.append(qa_prompt.format(etf_info=etf_info))

        # 배치 처리
        try:
            batch_results = structured_llm.batch(batch_prompts)

            # 결과 저장
            for i, qa_set in enumerate(batch_results):
                for qa in qa_set.qna_pairs:
                    enhanced_data.append({
                        "instruction": "다음 ETF 관련 질문에 전문가답게 답변해주세요.",
                        "input": qa.question,
                        "output": qa.answer,
                        "source": "openai_generated",
                        "etf_ticker": batch_df.iloc[i]['ticker']
                    })

            # Rate limit 방지
            time.sleep(1)

        except Exception as e:
            print(f"⚠️ 오류 발생: {e}")

    return Dataset.from_list(enhanced_data)

# 샘플 데이터로 테스트
sample_df = df.sample(n=10, random_state=1207)
enhanced_dataset = generate_qa_data(sample_df, batch_size=5)

print(f"OpenAI 생성 데이터: {len(enhanced_dataset)}개")
```

#### 4.2 역번역(Back-Translation) 데이터 증강

텍스트를 **다른 언어로 번역했다가 다시 원래 언어로 번역**하여 표현을 다양화합니다.

```python
from deep_translator import GoogleTranslator
import time

def back_translate_deep(text, intermediate_lang='en'):
    """
    역번역을 통한 패러프레이징
    한국어 → 영어 → 한국어
    """
    try:
        # 한국어 → 중간 언어
        translated = GoogleTranslator(source='ko', target=intermediate_lang).translate(text)

        # 중간 언어 → 한국어
        back_translated = GoogleTranslator(source=intermediate_lang, target='ko').translate(translated)

        return back_translated
    except Exception as e:
        print(f"역번역 오류: {e}")
        return text

def augment_by_back_translation(dataset, languages=['en', 'ja', 'zh-CN']):
    """여러 언어로 역번역하여 데이터 증강"""
    augmented_data = []

    print(f"역번역 데이터 증강 중... (언어: {languages})")

    for idx, example in enumerate(dataset):
        # 원본 추가
        augmented_data.append(example)

        # 각 언어로 역번역
        for lang in languages:
            try:
                aug_input = back_translate_deep(example['input'], lang)
                aug_output = back_translate_deep(example['output'], lang)

                # 너무 유사하면 스킵
                if aug_input == example['input']:
                    continue

                augmented_data.append({
                    'instruction': example['instruction'],
                    'input': aug_input,
                    'output': aug_output,
                    'source': f'back_translation_{lang}'
                })

                # Rate limit 방지
                time.sleep(0.5)

            except Exception as e:
                print(f"오류: {e}")
                continue

        if (idx + 1) % 10 == 0:
            print(f"진행: {idx+1}/{len(dataset)}")

    return Dataset.from_list(augmented_data)

# 샘플 증강 (영어만 사용)
sample_dataset = train_alpaca.select(range(20))
augmented_dataset = augment_by_back_translation(
    sample_dataset,
    languages=['en']
)

print(f"원본: {len(sample_dataset)}개")
print(f"증강 후: {len(augmented_dataset)}개")
print(f"증강 비율: {len(augmented_dataset)/len(sample_dataset):.1f}x")
```

#### 4.3 패러프레이징(Paraphrasing)

LLM을 사용하여 **같은 의미를 다른 방식으로 표현**합니다.

```python
def paraphrase_with_llm(text, num_variations=3):
    """LLM으로 패러프레이즈 생성"""

    prompt = f"""다음 텍스트를 {num_variations}가지 다른 방식으로 표현해주세요.
의미는 동일하게 유지하되, 문장 구조와 단어 선택을 다양하게 해주세요.

원문: {text}

다음 형식으로 응답하세요:
1. [첫 번째 패러프레이즈]
2. [두 번째 패러프레이즈]
3. [세 번째 패러프레이즈]
"""

    try:
        response = llm.invoke(prompt)
        # 번호 제거 및 분리
        variations = []
        for line in response.content.strip().split('\n'):
            line = line.strip()
            if line and line[0].isdigit():
                text = line.split('.', 1)[1].strip()
                variations.append(text)

        return variations[:num_variations]
    except Exception as e:
        print(f"패러프레이즈 오류: {e}")
        return [text]

def augment_by_paraphrasing(dataset, num_variations=2):
    """패러프레이징으로 데이터 증강"""
    augmented = []

    print("패러프레이징 데이터 증강 중...")

    for idx, example in enumerate(dataset):
        # 원본
        augmented.append(example)

        # 질문 패러프레이즈
        question_variations = paraphrase_with_llm(
            example['input'],
            num_variations
        )

        for var in question_variations:
            if var != example['input']:
                augmented.append({
                    'instruction': example['instruction'],
                    'input': var,
                    'output': example['output'],
                    'source': 'paraphrase'
                })

        if (idx + 1) % 5 == 0:
            print(f"진행: {idx+1}/{len(dataset)}")
            time.sleep(1)

    return Dataset.from_list(augmented)

# 샘플 증강
sample_for_para = train_alpaca.select(range(5))
paraphrased_dataset = augment_by_paraphrasing(sample_for_para, num_variations=2)

print(f"원본: {len(sample_for_para)}")
print(f"증강 후: {len(paraphrased_dataset)}")
```

---

### 단계 5: DPO 데이터셋 생성

DPO(Direct Preference Optimization)는 **선호되는 답변과 거부되는 답변을 비교 학습**하여 모델의 응답 품질을 향상시킵니다.

#### 5.1 추적배수 파싱 함수

```python
import re
import pandas as pd

def parse_tracking_multiplier(value):
    """
    추적배수 데이터 파싱

    입력:
    - '일반' → 1.0
    - '2X 레버리지' → 2.0
    - '2X 인버스' → -2.0
    - '1X 인버스' → -1.0
    """
    if pd.isna(value):
        return 1.0

    value_str = str(value).strip()

    # '일반'인 경우
    if value_str == '일반':
        return 1.0

    # 인버스 체크
    is_inverse = '인버스' in value_str

    # 숫자 추출 (1X, 2X, 1.5X 등)
    numbers = re.findall(r'(\d+\.?\d*)X?', value_str)

    if numbers:
        multiplier = float(numbers[0])
        return -multiplier if is_inverse else multiplier

    return 1.0

def get_etf_type_info(value):
    """ETF 타입 정보 반환"""
    multiplier = parse_tracking_multiplier(value)
    value_str = str(value).strip()

    if '인버스' in value_str:
        return {
            'type': '인버스',
            'multiplier': multiplier,
            'display_name': value_str,
            'abs_multiplier': abs(multiplier)
        }
    elif '레버리지' in value_str:
        return {
            'type': '레버리지',
            'multiplier': multiplier,
            'display_name': value_str,
            'abs_multiplier': multiplier
        }
    else:
        return {
            'type': '일반',
            'multiplier': 1.0,
            'display_name': '일반형',
            'abs_multiplier': 1.0
        }
```

#### 5.2 DPO 데이터셋 생성 함수

```python
def create_dpo_dataset(df):
    """
    ETF 타입별 맞춤형 DPO 데이터셋 생성
    - 패턴 1: 기본 설명 (모든 ETF)
    - 패턴 2: 인버스 ETF 주의사항
    - 패턴 3: 레버리지 ETF 적합성
    - 패턴 4: 일반형 ETF 장점
    - 패턴 5: 비용 비교 (모든 ETF)
    """
    dpo_data = []

    print("DPO 데이터셋 생성 중...")

    for idx, row in df.iterrows():
        # ETF 타입 정보 추출
        etf_info = get_etf_type_info(row['tracking_multiplier'])
        etf_type = etf_info['type']
        display_name = etf_info['display_name']
        abs_mult = etf_info['abs_multiplier']

        # ===== 패턴 1: 기본 설명 (모든 ETF) =====
        prompt1 = f"{row['name_kr']} ETF에 대해 설명해주세요."

        # 레버리지 특성 문구
        if etf_type == '인버스':
            leverage_desc = f"{display_name} 상품으로 기초지수와 반대 방향({abs_mult}배)으로 움직이며 높은 변동성을 보입니다"
        elif etf_type == '레버리지':
            leverage_desc = f"{display_name} 상품으로 기초지수 움직임의 {abs_mult}배 변동성을 보입니다"
        else:
            leverage_desc = "일반형 상품으로 기초지수를 안정적으로 추종합니다"

        chosen1 = f"""{row['name_kr']}(종목코드: {row['ticker']})는 {row['manager']}에서 운용하는 ETF입니다.

**기본 정보**
- 기초지수: {row['base_index']}
- 추적 방식: {display_name}, {row['replication_method']} 복제
- 상장일: {row['listing_date']}

**비용 및 과세**
- 총보수: 연 {row['total_expense_ratio']:.2f}%
- 과세 유형: {row['tax_type']}

**투자 특성**
- 기초 시장: {row['base_market_category']}
- 기초 자산: {row['base_asset_category']}

이 ETF는 {leverage_desc}."""

        rejected1 = f"""{row['name_kr']}는 ETF입니다. 종목코드는 {row['ticker']}이고, {row['manager']}에서 만들었습니다.
더 자세한 정보는 증권사 홈페이지를 참고하세요."""

        dpo_data.append({
            "prompt": prompt1,
            "chosen": chosen1,
            "rejected": rejected1,
            "etf_ticker": row['ticker'],
            "etf_type": etf_type
        })

        # ===== 패턴 2: 인버스 ETF 주의사항 =====
        if etf_type == '인버스':
            prompt2 = f"{row['name_kr']} ETF 투자 시 주의사항은 무엇인가요?"

            chosen2 = f"""{row['name_kr']}는 {display_name} ETF로, 다음 사항에 각별히 주의해야 합니다:

**1. 반대 방향 움직임**
기초지수({row['base_index']})가 상승하면 이 ETF는 하락하고, 지수가 하락하면 상승합니다.

**2. 복리 효과의 함정**
일간 수익률을 추종하므로, 2일 이상 보유 시 복리 효과로 인해 기초지수와 큰 괴리가 발생할 수 있습니다.

**3. 적합한 투자 방식**
- ✅ 적합: 단기(1일) 하락장 대응, 헤지 목적
- ❌ 부적합: 장기 투자, 매수 후 보유 전략

**4. 높은 비용**
연 {row['total_expense_ratio']:.2f}%의 총보수가 발생합니다.

시장 방향성을 정확히 예측할 수 있고, 단기 트레이딩 경험이 있는 투자자만 신중히 투자하시기 바랍니다."""

            rejected2 = f"{row['name_kr']}는 인버스 ETF라서 위험합니다. 초보자는 투자하지 마세요."

            dpo_data.append({
                "prompt": prompt2,
                "chosen": chosen2,
                "rejected": rejected2,
                "etf_ticker": row['ticker'],
                "etf_type": etf_type
            })

        # ===== 패턴 3: 레버리지 ETF 적합성 =====
        elif etf_type == '레버리지':
            prompt3 = f"{row['name_kr']} ETF는 어떤 투자자에게 적합한가요?"

            chosen3 = f"""{row['name_kr']}는 {display_name} ETF로, 다음과 같은 투자자에게 적합합니다:

**적합한 투자자**
- 단기(1일~수일) 트레이딩 경험이 있는 투자자
- 시장 방향성에 대한 확신이 있는 투자자
- 높은 위험을 감수할 수 있는 투자자

**주의사항**
1. **높은 변동성**: 기초지수 변동의 {abs_mult}배 영향
2. **복리 효과**: 장기 보유 시 기초지수와 괴리 발생
3. **적절한 보유 기간**: 1일~수일 단기 트레이딩용
4. **손절 필수**: 예상과 다른 방향 시 즉시 손절

초보 투자자나 장기 투자자에게는 일반형 ETF를 권장합니다."""

            rejected3 = f"{row['name_kr']}는 레버리지 ETF입니다. 위험하니 투자하지 마세요."

            dpo_data.append({
                "prompt": prompt3,
                "chosen": chosen3,
                "rejected": rejected3,
                "etf_ticker": row['ticker'],
                "etf_type": etf_type
            })

        # ===== 패턴 4: 일반형 ETF 장점 =====
        else:
            prompt4 = f"{row['name_kr']} ETF의 장점은 무엇인가요?"

            chosen4 = f"""{row['name_kr']}는 일반형 ETF로, 다음과 같은 장점이 있습니다:

**1. 안정적인 추종**
{row['base_index']}를 1:1로 추종하여 복리 효과로 인한 괴리가 거의 없습니다.

**2. 합리적인 비용**
연 {row['total_expense_ratio']:.2f}%의 총보수로 장기 투자 시에도 비용 부담이 적습니다.

**3. 장기 투자 적합**
레버리지나 인버스 상품과 달리 장기 보유가 가능하며, 중장기 자산 배분 전략에 적합합니다.

**4. 안정적인 운용**
{row['manager']}의 {row['replication_method']} 방식으로 안정적으로 운용됩니다.

{row['base_asset_category']} 자산에 장기적으로 투자하고자 하는 투자자에게 적합한 상품입니다."""

            rejected4 = f"{row['name_kr']}는 그냥 평범한 ETF입니다. 특별한 점이 없습니다."

            dpo_data.append({
                "prompt": prompt4,
                "chosen": chosen4,
                "rejected": rejected4,
                "etf_ticker": row['ticker'],
                "etf_type": etf_type
            })

        if (idx + 1) % 50 == 0:
            print(f"진행: {idx+1}/{len(df)}")

    return Dataset.from_list(dpo_data)

# DPO 데이터셋 생성
dpo_dataset = create_dpo_dataset(df)
dpo_split = dpo_dataset.train_test_split(test_size=0.1, seed=1207)

print(f"총 샘플: {len(dpo_dataset)}개")
print(f"Train: {len(dpo_split['train'])}개")
print(f"Test: {len(dpo_split['test'])}개")
```

#### 5.3 Hugging Face에 업로드

```python
from datasets import DatasetDict

# DPO DatasetDict 생성
dpo_dataset_dict = DatasetDict({
    'train': dpo_split['train'],
    'test': dpo_split['test']
})

# 업로드
dpo_dataset_dict.push_to_hub(
    "your_username/etf-dpo-v1",
    private=True,
)

print("✅ DPO 데이터셋 업로드 완료!")
```

---

## 💡 실습 문제

### 문제 1: 새로운 Alpaca 패턴 추가

기존 `create_alpaca_dataset` 함수에 **"ETF 비교"** 패턴을 추가하세요.

**요구사항:**
- 같은 운용사의 다른 ETF와 총보수를 비교
- instruction: "다음 ETF를 비교해주세요."
- input: "ETF A vs ETF B"
- output: 총보수, 기초지수, 과세 유형 비교

<details>
<summary>솔루션 보기</summary>

```python
def create_alpaca_with_comparison(df):
    """ETF 비교 패턴이 추가된 Alpaca 데이터셋"""
    alpaca_data = []

    # 기존 4가지 패턴 (생략)
    # ...

    # 새로운 패턴 5: ETF 비교
    for manager in df['manager'].unique():
        manager_etfs = df[df['manager'] == manager]

        if len(manager_etfs) >= 2:
            # 첫 2개 ETF 비교
            row1 = manager_etfs.iloc[0]
            row2 = manager_etfs.iloc[1]

            alpaca_data.append({
                "instruction": "다음 ETF를 비교해주세요.",
                "input": f"{row1['name_kr']} vs {row2['name_kr']}",
                "output": f"""두 ETF 모두 {manager}에서 운용하지만 차이가 있습니다:

**{row1['short_name_kr']}** (종목코드: {row1['ticker']})
- 기초지수: {row1['base_index']}
- 총보수: 연 {row1['total_expense_ratio']:.2f}%
- 과세 유형: {row1['tax_type']}

**{row2['short_name_kr']}** (종목코드: {row2['ticker']})
- 기초지수: {row2['base_index']}
- 총보수: 연 {row2['total_expense_ratio']:.2f}%
- 과세 유형: {row2['tax_type']}

비용 측면에서는 {"첫 번째" if row1['total_expense_ratio'] < row2['total_expense_ratio'] else "두 번째"} ETF가 더 유리합니다."""
            })

    return Dataset.from_list(alpaca_data)

# 테스트
comparison_dataset = create_alpaca_with_comparison(train_df)
print(f"비교 패턴 포함 데이터: {len(comparison_dataset)}개")
print(comparison_dataset[0])
```

</details>

---

### 문제 2: 임베딩 데이터셋 Hard Negative 비율 분석

생성된 `train_embedding` 데이터셋에서 **Positive pairs와 Negative pairs의 비율**을 계산하세요.

**요구사항:**
- label=1 (Positive) 개수
- label=0 (Negative) 개수
- 비율 출력

<details>
<summary>솔루션 보기</summary>

```python
def analyze_label_distribution(dataset):
    """임베딩 데이터셋의 레이블 분포 분석"""
    labels = [example['label'] for example in dataset]

    positive_count = sum(1 for label in labels if label == 1)
    negative_count = sum(1 for label in labels if label == 0)
    total_count = len(labels)

    print(f"총 샘플 수: {total_count:,}개")
    print(f"\nPositive pairs (label=1): {positive_count:,}개 ({positive_count/total_count*100:.1f}%)")
    print(f"Negative pairs (label=0): {negative_count:,}개 ({negative_count/total_count*100:.1f}%)")
    print(f"\nPositive:Negative 비율 = 1:{negative_count/positive_count:.2f}")

    # 불균형 검사
    if negative_count / positive_count > 3:
        print("\n⚠️ 경고: Negative 샘플이 너무 많습니다. 균형을 맞춰주세요.")
    elif positive_count / negative_count > 3:
        print("\n⚠️ 경고: Positive 샘플이 너무 많습니다. Hard Negatives를 추가하세요.")
    else:
        print("\n✅ 레이블 분포가 적절합니다.")

# 분석 실행
analyze_label_distribution(train_embedding)
```

**출력 예시:**
```
총 샘플 수: 39,442개

Positive pairs (label=1): 1,488개 (3.8%)
Negative pairs (label=0): 37,954개 (96.2%)

Positive:Negative 비율 = 1:25.51

⚠️ 경고: Negative 샘플이 너무 많습니다. 균형을 맞춰주세요.
```

</details>

---

### 문제 3: 역번역 품질 검증

역번역된 텍스트가 **원본과 얼마나 다른지** 유사도를 계산하세요.

**요구사항:**
- 원본 텍스트와 역번역 텍스트의 단어 중복률 계산
- 5개 샘플에 대해 검증

<details>
<summary>솔루션 보기</summary>

```python
def calculate_word_overlap(text1, text2):
    """두 텍스트 간 단어 중복률 계산"""
    words1 = set(text1.split())
    words2 = set(text2.split())

    if len(words1) == 0 or len(words2) == 0:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2

    jaccard_similarity = len(intersection) / len(union)
    return jaccard_similarity

def validate_back_translation(original_dataset, augmented_dataset, num_samples=5):
    """역번역 품질 검증"""
    print("역번역 품질 검증\n" + "="*60)

    # 역번역된 샘플만 필터링
    back_translated = [
        (i, ex) for i, ex in enumerate(augmented_dataset)
        if ex.get('source', '').startswith('back_translation')
    ]

    if len(back_translated) == 0:
        print("역번역 샘플이 없습니다.")
        return

    # 랜덤 샘플 선택
    import random
    samples = random.sample(back_translated, min(num_samples, len(back_translated)))

    for idx, (aug_idx, aug_example) in enumerate(samples, 1):
        # 원본 찾기 (original_idx 활용)
        original_idx = aug_example.get('original_idx', 0)
        original_text = original_dataset[original_idx]['input']
        back_translated_text = aug_example['input']

        similarity = calculate_word_overlap(original_text, back_translated_text)

        print(f"\n샘플 {idx}:")
        print(f"원본: {original_text}")
        print(f"역번역: {back_translated_text}")
        print(f"단어 중복률: {similarity*100:.1f}%")
        print(f"언어: {aug_example.get('source', 'unknown')}")

# 검증 실행
validate_back_translation(sample_dataset, augmented_dataset, num_samples=3)
```

**출력 예시:**
```
역번역 품질 검증
============================================================

샘플 1:
원본: KB RISE 국채선물10년인버스증권상장지수투자신탁(채권-파생형) (종목코드: 295020)
역번역: KB RISE 국채선물 10년 인버스증권거래소지수투자신탁(채권파생상품) (제목코드: 295020)
단어 중복률: 42.9%
언어: back_translation_en

샘플 2:
...
```

</details>

---

### 문제 4: DPO 데이터셋 ETF 타입별 분포 시각화

DPO 데이터셋에서 **일반, 레버리지, 인버스 ETF의 비율**을 막대 그래프로 시각화하세요.

**요구사항:**
- matplotlib 사용
- ETF 타입별 개수와 비율 표시
- 막대 그래프에 값 레이블 추가

<details>
<summary>솔루션 보기</summary>

```python
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
# plt.rcParams['font.family'] = 'AppleGothic'  # macOS
plt.rcParams['axes.unicode_minus'] = False

def visualize_etf_type_distribution(dpo_dataset):
    """ETF 타입별 분포 시각화"""
    # 타입별 카운트
    etf_types = [example['etf_type'] for example in dpo_dataset]

    type_counts = {}
    for etf_type in ['일반', '레버리지', '인버스']:
        count = etf_types.count(etf_type)
        type_counts[etf_type] = count

    # 그래프 생성
    fig, ax = plt.subplots(figsize=(10, 6))

    types = list(type_counts.keys())
    counts = list(type_counts.values())
    total = sum(counts)
    percentages = [count/total*100 for count in counts]

    # 막대 그래프
    bars = ax.bar(types, counts, color=['#4CAF50', '#FF9800', '#F44336'])

    # 값 레이블 추가
    for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
                f'{count:,}개\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_xlabel('ETF 타입', fontsize=14)
    ax.set_ylabel('샘플 수', fontsize=14)
    ax.set_title('DPO 데이터셋 ETF 타입별 분포', fontsize=16, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 통계 출력
    print("ETF 타입별 통계:")
    print("="*50)
    for etf_type, count in type_counts.items():
        print(f"{etf_type:8s}: {count:5,}개 ({count/total*100:5.1f}%)")
    print("="*50)
    print(f"총계      : {total:5,}개 (100.0%)")

# 시각화 실행
visualize_etf_type_distribution(dpo_dataset)
```

</details>

---

## 🚀 실무 활용 예시

### 예시 1: ETF 챗봇용 통합 데이터셋 구축

**시나리오**: ETF 투자 상담 챗봇을 위한 종합 학습 데이터셋 생성

```python
def create_comprehensive_chatbot_dataset(df):
    """
    ETF 챗봇용 통합 데이터셋
    - Alpaca 기본 데이터
    - LLM 생성 Q&A (전문가 수준)
    - 역번역 증강 (표현 다양화)
    - DPO 데이터 (응답 품질 향상)
    """
    from datasets import concatenate_datasets, DatasetDict

    print("📊 1단계: 기본 Alpaca 데이터셋 생성")
    base_alpaca = create_alpaca_dataset(df)
    print(f"   생성: {len(base_alpaca)}개")

    print("\n🤖 2단계: LLM 생성 고품질 Q&A")
    sample_df = df.sample(n=50, random_state=1207)  # 비용 고려하여 샘플만
    llm_qa = generate_qa_data(sample_df, batch_size=10)
    print(f"   생성: {len(llm_qa)}개")

    print("\n🌐 3단계: 역번역 증강")
    sample_for_aug = base_alpaca.select(range(100))
    augmented = augment_by_back_translation(sample_for_aug, languages=['en'])
    print(f"   생성: {len(augmented)}개")

    print("\n📈 4단계: 데이터셋 병합")
    # instruction, input, output 컬럼만 유지
    combined = concatenate_datasets([
        base_alpaca.select_columns(['instruction', 'input', 'output']),
        llm_qa.select_columns(['instruction', 'input', 'output']),
        augmented.select_columns(['instruction', 'input', 'output'])
    ])

    print(f"   총 {len(combined)}개 샘플")

    print("\n🎯 5단계: Train/Validation 분할")
    split = combined.train_test_split(test_size=0.1, seed=1207)

    final_dataset = DatasetDict({
        'train': split['train'],
        'validation': split['test']
    })

    print(f"   Train: {len(final_dataset['train'])}개")
    print(f"   Validation: {len(final_dataset['validation'])}개")

    print("\n☁️ 6단계: Hugging Face 업로드")
    final_dataset.push_to_hub(
        "your_username/etf-chatbot-dataset-v1",
        private=True
    )

    print("✅ 통합 데이터셋 구축 완료!")
    return final_dataset

# 실행
comprehensive_dataset = create_comprehensive_chatbot_dataset(df)
```

---

### 예시 2: 임베딩 모델 파인튜닝 파이프라인

**시나리오**: 한국 ETF 시맨틱 검색을 위한 임베딩 모델 학습

```python
def train_embedding_model_pipeline():
    """
    임베딩 모델 파인튜닝 완전 파이프라인
    """
    from sentence_transformers import SentenceTransformer, losses, InputExample
    from torch.utils.data import DataLoader

    print("🔧 1단계: 기본 모델 로드")
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    print(f"   모델: {model}")

    print("\n📊 2단계: 임베딩 데이터셋 준비")
    embedding_dataset = create_embedding_dataset(df)

    # InputExample 변환
    train_examples = []
    for example in embedding_dataset:
        train_examples.append(InputExample(
            texts=[example['sentence1'], example['sentence2']],
            label=float(example['label'])
        ))

    print(f"   학습 샘플: {len(train_examples)}개")

    print("\n🎓 3단계: DataLoader 생성")
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=16
    )

    print("\n📉 4단계: Loss 함수 설정")
    train_loss = losses.CosineSimilarityLoss(model)

    print("\n🚀 5단계: 파인튜닝 실행")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        warmup_steps=100,
        output_path='./output/etf-embedding-model'
    )

    print("\n💾 6단계: 모델 저장 및 업로드")
    model.save('./output/etf-embedding-model-final')

    # Hugging Face 업로드 (선택)
    # model.push_to_hub("your_username/etf-embedding-ko")

    print("\n✅ 임베딩 모델 파인튜닝 완료!")

    print("\n🧪 7단계: 모델 테스트")
    test_queries = [
        "삼성전자에 투자하는 ETF",
        "미국 S&P500 추종 상품",
        "금 관련 ETF"
    ]

    for query in test_queries:
        embedding = model.encode(query)
        print(f"\n쿼리: {query}")
        print(f"임베딩 차원: {len(embedding)}")
        print(f"임베딩 샘플: {embedding[:5]}")

# 실행
# train_embedding_model_pipeline()  # 실제 학습은 주석 처리
```

---

### 예시 3: DPO 기반 안전한 ETF 상담 시스템

**시나리오**: 고위험 ETF(레버리지/인버스)에 대한 안전한 응답 시스템

```python
def create_safe_etf_advisor_dataset(df):
    """
    안전한 ETF 상담을 위한 DPO 데이터셋
    - 위험 상품에 대한 정확한 경고
    - 투자자 보호 중심의 응답
    - 규제 준수 답변
    """
    from datasets import DatasetDict

    print("🛡️ 안전한 ETF 상담 시스템 데이터셋 생성\n")

    # DPO 데이터셋 생성
    dpo_dataset = create_dpo_dataset(df)

    # 고위험 상품 필터링
    high_risk = [
        ex for ex in dpo_dataset
        if ex['etf_type'] in ['레버리지', '인버스']
    ]

    normal_risk = [
        ex for ex in dpo_dataset
        if ex['etf_type'] == '일반'
    ]

    print(f"📊 데이터 분포:")
    print(f"   고위험 (레버리지/인버스): {len(high_risk)}개")
    print(f"   일반 위험 (일반형): {len(normal_risk)}개")

    # 고위험 샘플 비율 증가 (안전 교육 강화)
    from datasets import Dataset

    # 고위험 샘플 2배 복제
    safety_focused = Dataset.from_list(high_risk * 2 + normal_risk)

    print(f"\n🎯 안전 강화 후 총 샘플: {len(safety_focused)}개")

    # Train/Test 분할
    split = safety_focused.train_test_split(test_size=0.1, seed=1207)

    final_dataset = DatasetDict({
        'train': split['train'],
        'test': split['test']
    })

    # 업로드
    final_dataset.push_to_hub(
        "your_username/safe-etf-advisor-dpo-v1",
        private=True
    )

    print("\n✅ 안전한 ETF 상담 시스템 데이터셋 완성!")
    print(f"   URL: https://huggingface.co/datasets/your_username/safe-etf-advisor-dpo-v1")

    return final_dataset

# 실행
safe_advisor_dataset = create_safe_etf_advisor_dataset(df)
```

---

## 📌 핵심 요약

### 1. 데이터셋 형식 선택 가이드

| 목적 | 데이터셋 형식 | 사용 사례 |
|------|-------------|----------|
| 질의응답 시스템 | Alpaca (instruction-input-output) | 챗봇, 상담 시스템 |
| 시맨틱 검색 | 임베딩 (sentence pairs) | RAG, 문서 검색 |
| 응답 품질 향상 | DPO (prompt-chosen-rejected) | 안전한 응답, 고품질 답변 |

### 2. 데이터 증강 기법 비교

| 기법 | 비용 | 품질 | 속도 | 적합한 경우 |
|------|------|------|------|----------|
| LLM 생성 | 높음 (API 비용) | 최고 | 느림 | 전문 도메인, 고품질 필요 |
| 역번역 | 낮음 (무료 API) | 중간 | 중간 | 표현 다양화, 언어 로버스트 |
| 패러프레이징 | 중간 (LLM 사용) | 높음 | 느림 | 자연스러운 변형 |

### 3. Hugging Face 워크플로우

```python
# 업로드
dataset.push_to_hub("username/dataset-name", private=True)

# 다운로드
from datasets import load_dataset
dataset = load_dataset("username/dataset-name", split="train")

# 버전 관리
dataset.push_to_hub("username/dataset-name", revision="v2")
```

### 4. 체크리스트

**Alpaca 데이터셋**
- [ ] instruction, input, output 필드 포함
- [ ] 다양한 질문 패턴 (4가지 이상)
- [ ] 도메인 특화 정보 포함
- [ ] Train/Test 분할 (80/20)

**임베딩 데이터셋**
- [ ] Positive pairs 충분 (전체의 20% 이상)
- [ ] Hard negatives 포함 (겉보기 유사하지만 다름)
- [ ] Easy negatives 포함 (완전히 다름)
- [ ] 레이블 균형 확인 (1:3 비율 권장)

**DPO 데이터셋**
- [ ] chosen 답변: 전문적, 상세, 안전
- [ ] rejected 답변: 불충분, 위험, 부정확
- [ ] 명확한 품질 차이
- [ ] 도메인별 맞춤형 패턴

---

## 🔗 참고 자료

- **Unsloth**: https://github.com/unslothai/unsloth
- **Sentence Transformers**: https://www.sbert.net/
- **Hugging Face Datasets**: https://huggingface.co/docs/datasets
- **Alpaca 형식**: https://github.com/tatsu-lab/stanford_alpaca
- **DPO 논문**: https://arxiv.org/abs/2305.18290
- **LangChain**: https://python.langchain.com/

---

이 가이드를 통해 **파인튜닝용 고품질 데이터셋**을 체계적으로 생성하고 관리할 수 있습니다. 도메인 특화 LLM 및 임베딩 모델 개발에 활용하세요!
