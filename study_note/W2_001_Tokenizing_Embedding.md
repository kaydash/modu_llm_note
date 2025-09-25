# W2_001_Tokenizing_Embedding.md - 토큰화와 임베딩 마스터하기

## 🎯 학습 목표
- 토큰화(Tokenization)의 개념과 다양한 기법 이해
- 임베딩(Embedding)의 원리와 활용법 학습
- 한국어 NLP를 위한 실용적 도구 습득
- 텍스트 유사도 계산과 비교 분석 능력 개발

## 📚 핵심 개념

### 토큰화(Tokenization)
- **개념**: 텍스트를 분석 가능한 작은 단위로 나누는 과정
- **목적**: 컴퓨터가 자연어를 이해하기 위한 첫 번째 단계
- **의미 단위**: 의미 있는 최소 단위로 텍스트 분할

### 임베딩(Embedding)
- **개념**: 텍스트를 컴퓨터가 이해할 수 있는 수치 벡터로 변환
- **목적**: 의미적으로 유사한 텍스트를 벡터 공간에서 가까운 거리에 배치
- **응용**: 문서 검색, 텍스트 분류, 감정 분석, 기계 번역

## 🔧 환경 설정

### 필수 라이브러리 설치
```bash
# 기본 NLP 도구
pip install kiwipiepy transformers sentence-transformers

# 과학 계산 및 시각화
pip install numpy pandas matplotlib seaborn scikit-learn

# 전통적 임베딩 모델
pip install gensim

# UV 패키지 매니저 사용 시
uv add kiwipiepy transformers sentence-transformers numpy pandas matplotlib seaborn scikit-learn gensim
```

### 기본 임포트 및 설정
```python
import time
from functools import wraps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정 (Windows)
import matplotlib.font_manager as fm
font_path = "c:/Windows/Fonts/malgun.ttf"  # Windows 경로
matplotlib.rc('font', family=fm.FontProperties(fname=font_path).get_name())

# 한글 폰트 설정 (Mac)
matplotlib.rc('font', family='Gulim')

# 마이너스 부호 처리
matplotlib.rc("axes", unicode_minus=False)

def measure_time(func):
    """실행 시간 측정 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} 실행 시간: {end-start:.4f}초")
        return result
    return wrapper
```

## 💻 토큰화(Tokenization)

### 1. 단어 단위 토큰화 (Word Tokenization)

#### 개념과 특징
- **형태소 분석 기반**: 의미 있는 최소 단위로 분리
- **문법 요소 분리**: 조사, 어미 등 개별 토큰으로 처리
- **한국어 특화**: 교착어 특성을 잘 반영
- **활용 분야**: 문장 분석, 품사 태깅, 텍스트 분류

#### Kiwi 형태소 분석기 활용
```python
from kiwipiepy import Kiwi

# Kiwi 형태소 분석기 초기화
kiwi = Kiwi()

@measure_time
def tokenize_with_kiwi(text):
    """토큰화 함수"""
    try:
        return kiwi.tokenize(text)
    except Exception as e:
        print(f"토큰화 오류: {e}")
        return []

# 토큰화 실행
text = "자연어처리를 공부하는 것은 정말 흥미롭고 유용합니다!"
tokens = tokenize_with_kiwi(text)

print("토큰화 결과:")
for i, token in enumerate(tokens, 1):
    print(f"{i:>3} {token.form:^8} {token.tag:^8} ({token.start}-{token.start + token.len})")
```

#### 품사 태그 매핑
```python
# 품사 태그 한국어 매핑
pos_dict = {
    # 체언
    'NNG': '일반명사', 'NNP': '고유명사', 'NNB': '의존명사',
    'NP': '대명사', 'NR': '수사',
    # 용언
    'VV': '동사', 'VA': '형용사', 'VX': '보조용언',
    'VCP': '긍정지정사', 'VCN': '부정지정사',
    # 관형사/부사
    'MM': '관형사', 'MAG': '일반부사', 'MAJ': '접속부사',
    # 조사
    'JKS': '주격조사', 'JKO': '목적격조사', 'JKG': '관형격조사',
    'JKB': '부사격조사', 'JX': '보조사', 'JC': '접속조사',
    # 어미
    'EF': '종결어미', 'EC': '연결어미', 'ETM': '관형사형전성어미',
    # 접사
    'XSV': '동사파생접미사', 'XSA': '형용사파생접미사', 'XR': '어근',
    # 기호
    'SF': '마침표류', 'SP': '쉼표류', 'SN': '숫자'
}

# 결과 출력
print(f"{'번호':>3} {'단어':^8} {'품사태그':^8} {'한국어품사':^12} {'위치':^6}")
print("-" * 45)
for i, token in enumerate(tokens, 1):
    korean_pos = pos_dict.get(token.tag, token.tag)
    pos_range = f"{token.start}-{token.start + token.len}"
    print(f"{i:>3} {token.form:^8} {token.tag:^8} {korean_pos:^12} {pos_range:^6}")
```

### 2. 서브워드 토큰화 (Subword Tokenization)

#### 개념과 특징
- **서브워드 단위**: 형태소보다 작은 의미 단위 사용
- **패턴 기반**: 자주 등장하는 문자열 패턴을 토큰으로 활용
- **OOV 해결**: Out-of-Vocabulary 문제 효과적 해결
- **다국어 지원**: 언어에 관계없이 일관된 처리

#### 주요 알고리즘 비교

| 알고리즘 | 특징 | 대표 모델 | 장점 |
|---------|------|----------|------|
| BPE (Byte Pair Encoding) | 빈도 기반 병합 | GPT 시리즈 | 단순하고 효율적 |
| WordPiece | 확률 기반 분할 | BERT, ELECTRA | 언어학적 의미 보존 |
| SentencePiece | 언어 독립적 | T5, mT5, XLM-R | 다국어 지원 우수 |

#### BERT 토크나이저 (WordPiece) 활용
```python
from transformers import AutoTokenizer

# KcBERT 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained('beomi/kcbert-base')

# 토큰화 수행
text = "자연어처리를 공부합니다"
tokens = tokenizer.tokenize(text)

print(f"원문: {text}")
print(f"토큰: {tokens}")
# 출력: ['자연', '##어', '##처리', '##를', '공부', '##합니다']

# 토큰화 분석 함수
def analyze_tokenization(texts, tokenizer):
    """토큰화 분석 함수"""
    results = []

    for text in texts:
        tokens = tokenizer.tokenize(text)
        subword_tokens = [token for token in tokens if token.startswith('##')]

        results.append({
            'text': text,
            'token_count': len(tokens),
            'subword_count': len(subword_tokens),
            'tokens': tokens
        })

    return results

# 테스트 문장들
test_texts = [
    "자연어처리를 공부합니다",
    "인공지능과 머신러닝은 미래기술이다",
    "COVID-19로 인한 비대면 수업이 증가했다",
    "스마트폰의 음성인식 기능이 발전했다"
]

# 분석 결과 출력
results = analyze_tokenization(test_texts, tokenizer)
for i, result in enumerate(results, 1):
    print(f"\n{i}. 원문: {result['text']}")
    print(f"   토큰 수: {result['token_count']}개")
    print(f"   서브워드: {result['subword_count']}개")
    print(f"   토큰들: {result['tokens']}")
```

#### SentencePiece 토크나이저 활용
```python
from transformers import AutoTokenizer

# BGE-M3 토크나이저 (SentencePiece 기반)
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

text = "자연어처리를 공부합니다"
tokens = tokenizer.tokenize(text)
print(f"SentencePiece 토큰: {tokens}")
# 출력: ['▁자연', '어', '처리', '를', '▁공부', '합니다']
# ▁ 는 단어 시작을 나타내는 특수 기호
```

## 🎯 임베딩(Embedding)

### 1. 단어 임베딩 (Word Embedding)

#### 1.1 Bag of Words (BoW)

**개념과 특징**
- **빈도 기반**: 단어의 출현 빈도를 벡터로 표현
- **순서 무시**: 단어 순서 정보는 고려하지 않음
- **희소 벡터**: 대부분의 값이 0인 sparse 벡터 생성

```python
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

def create_bow_vectors(texts):
    # 토큰화된 텍스트로 변환
    tokenized_texts = []
    for text in texts:
        tokens = tokenize_with_kiwi(text)
        token_words = [token.form for token in tokens if token.tag not in ['SF', 'SP']]
        tokenized_texts.append(' '.join(token_words))

    # BoW 벡터화
    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform(tokenized_texts)

    return bow_matrix, vectorizer

# 예제 텍스트
texts = [
    "자연어 처리를 공부합니다",
    "자연어는 컴퓨터 언어가 아니라 인간의 언어입니다",
    "자연어 수업 시간에 자연어 처리 방법을 배우고 있습니다"
]

bow_matrix, vectorizer = create_bow_vectors(texts)

# 결과를 DataFrame으로 표시
feature_names = vectorizer.get_feature_names_out()
bow_df = pd.DataFrame(
    bow_matrix.toarray(),
    columns=feature_names,
    index=[f'문서{i+1}' for i in range(len(texts))]
)

print("BoW 매트릭스:")
print(bow_df)
```

#### 1.2 TF-IDF (Term Frequency-Inverse Document Frequency)

**개념과 특징**
- **가중치 부여**: 단어의 중요도를 고려한 가중치 적용
- **희귀성 고려**: 희귀한 단어에 높은 가중치 부여
- **상대적 중요성**: 문서 집합 내에서의 상대적 중요도 측정

```python
from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf_vectors(texts):
    # 토큰화
    tokenized_texts = []
    for text in texts:
        tokens = tokenize_with_kiwi(text)
        token_words = [token.form for token in tokens if token.tag not in ['SF', 'SP']]
        tokenized_texts.append(' '.join(token_words))

    # TF-IDF 벡터화
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(tokenized_texts)

    return tfidf_matrix, tfidf

tfidf_matrix, tfidf = create_tfidf_vectors(texts)

# 결과를 DataFrame으로 표시
feature_names = tfidf.get_feature_names_out()
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=feature_names,
    index=[f'문서{i+1}' for i in range(len(texts))]
)

print("TF-IDF 매트릭스:")
print(tfidf_df.round(3))
```

#### 1.3 Word2Vec

**개념과 특징**
- **의미적 관계**: 단어의 의미적 관계를 벡터 공간에 표현
- **문맥 고려**: 주변 단어의 문맥을 고려한 단어 표현
- **두 가지 모델**: CBOW(Continuous Bag of Words)와 Skip-gram
- **벡터 연산**: 단어 간 의미적 관계를 벡터 연산으로 표현

```python
from gensim.models import Word2Vec

def train_word2vec_model(texts, vector_size=100, window=3, min_count=1):
    """Word2Vec 모델 훈련"""
    # 토큰화
    tokenized_corpus = []
    for text in texts:
        tokens = tokenize_with_kiwi(text)
        token_words = [token.form for token in tokens if len(token.form) > 1]
        tokenized_corpus.append(token_words)

    # Skip-gram 모델 훈련
    model = Word2Vec(
        sentences=tokenized_corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        sg=1,  # 1: Skip-gram, 0: CBOW
        epochs=100
    )

    return model, tokenized_corpus

# 더 큰 코퍼스로 모델 훈련
word2vec_texts = [
    "자연어 처리는 인공지능의 중요한 분야입니다",
    "머신러닝과 딥러닝이 자연어 처리에 활용됩니다",
    "컴퓨터가 인간의 언어를 이해하는 기술입니다",
    "텍스트 데이터를 분석하고 처리하는 방법을 학습합니다",
    "단어의 의미와 문맥을 파악하는 것이 중요합니다"
]

word2vec_model, tokenized_corpus = train_word2vec_model(word2vec_texts)

print(f"어휘 사전 크기: {len(word2vec_model.wv.key_to_index)}개")
print(f"벡터 차원: {word2vec_model.vector_size}차원")

# 단어 벡터 확인
test_words = ['자연어 처리', '언어', '데이터', '컴퓨터']
available_words = [word for word in test_words if word in word2vec_model.wv]

for word in available_words:
    vector = word2vec_model.wv[word]
    print(f"'{word}' 벡터 (처음 10차원): {vector[:10].round(3)}")

# 유사 단어 찾기
def find_similar_words(word, model, topn=3):
    try:
        similar_words = model.wv.most_similar(word, topn=topn)
        print(f"\n'{word}'와 유사한 단어들:")
        for similar_word, similarity in similar_words:
            print(f"  {similar_word}: {similarity:.3f}")
    except:
        print(f"'{word}'의 유사 단어를 찾을 수 없습니다.")

for word in available_words:
    find_similar_words(word, word2vec_model)
```

### 2. 문장 임베딩 (Sentence Embedding)

#### 개념과 특징
- **전체 문장 표현**: 문장 전체를 하나의 벡터로 표현
- **의미 보존**: 문장의 의미적 특성을 벡터에 보존
- **고정 크기**: 문장 길이와 관계없이 고정된 크기의 벡터
- **유사도 계산**: 문장 간 의미적 유사도 계산 가능

#### 2.1 평균 기반 문장 임베딩

```python
def create_sentence_embedding_avg(sentence, word2vec_model):
    """Word2Vec 평균을 이용한 문장 임베딩"""
    tokens = tokenize_with_kiwi(sentence)
    token_words = [token.form for token in tokens if token.form in word2vec_model.wv]

    if not token_words:
        return np.zeros(word2vec_model.vector_size)

    # 단어 벡터들의 평균 계산
    sentence_vector = np.zeros(word2vec_model.vector_size)
    for word in token_words:
        sentence_vector += word2vec_model.wv[word]

    return sentence_vector / len(token_words)

# 문장 임베딩 생성
test_sentence = "자연어 처리를 공부합니다"
sentence_embed = create_sentence_embedding_avg(test_sentence, word2vec_model)

print(f"문장: '{test_sentence}'")
print(f"임베딩 차원: {len(sentence_embed)}")
print(f"임베딩 벡터 (처음 10차원): {sentence_embed[:10].round(3)}")
```

#### 2.2 SBERT (Sentence-BERT)

```python
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# 한국어 최적화 모델들
korean_models = {
    'ko-sroberta': 'jhgan/ko-sroberta-multitask',
    'bge-m3': 'BAAI/bge-m3'
}

def load_sentence_transformer(model_key='ko-sroberta'):
    """SBERT 모델 로딩"""
    try:
        model_name = korean_models[model_key]
        model = SentenceTransformer(model_name)
        print(f"모델 로드 완료: {model_name}")
        return model
    except Exception as e:
        print(f"모델 로딩 오류: {e}")
        return None

@measure_time
def create_sentence_embeddings(sentences, model):
    """배치 문장 임베딩 생성"""
    try:
        embeddings = model.encode(sentences, convert_to_tensor=False)
        return embeddings
    except Exception as e:
        print(f"임베딩 생성 오류: {e}")
        return None

# ko-sroberta 모델 로드
sbert_model = load_sentence_transformer('ko-sroberta')

# 테스트 문장들
test_sentences = [
    "자연어 처리는 인공지능의 핵심 기술입니다",
    "머신러닝을 이용해서 텍스트를 분석합니다",
    "오늘 날씨가 정말 좋네요",
    "컴퓨터가 인간의 언어를 이해하는 방법을 연구합니다"
]

# 임베딩 생성
sbert_embeddings = create_sentence_embeddings(test_sentences, sbert_model)

print(f"임베딩 형태: {sbert_embeddings.shape}")
print(f"임베딩 차원: {sbert_embeddings.shape[1]}")

# 첫 번째 문장의 임베딩 일부 출력
print(f"\n첫 번째 문장 임베딩 (처음 10차원):")
print(f"'{test_sentences[0]}'")
print(f"{sbert_embeddings[0][:10].round(3)}")

# BGE-M3 모델도 동일하게 활용 가능
bge_model = load_sentence_transformer('bge-m3')
bge_embeddings = create_sentence_embeddings(test_sentences, bge_model)
print(f"\nBGE-M3 임베딩 차원: {bge_embeddings.shape[1]}")
```

## 📊 텍스트 유사도 비교

### 유사도 메트릭 비교

| 메트릭 | 수식 | 범위 | 해석 | 특징 | 주요 용도 |
|--------|------|------|------|------|----------|
| 유클리드 거리 | $\sqrt{\sum(a_i-b_i)^2}$ | [0, ∞) | 낮을수록 유사 | 절대 거리 | 클러스터링 |
| 코사인 유사도 | $\frac{a \cdot b}{\\|a\\|\\|b\\|}$ | [-1, 1] | 1에 가까울수록 유사 | 방향성 중시 | 문서 검색 |
| 내적 | $\sum a_i \cdot b_i$ | (-∞, ∞) | 클수록 유사 | 크기+방향 | 추천 시스템 |

### 유사도 계산 시스템

```python
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

class SimilarityCalculator:
    """텍스트 유사도 계산 클래스"""

    def __init__(self, sbert_model):
        self.model = sbert_model

    def calculate_all_similarities(self, sentence1, sentence2):
        """모든 유사도 메트릭 계산"""
        # 임베딩 생성
        embeddings = self.model.encode([sentence1, sentence2])
        emb1, emb2 = embeddings[0], embeddings[1]

        # 각종 유사도 계산
        euclidean_dist = euclidean_distances([emb1], [emb2])[0][0]
        cosine_sim = cosine_similarity([emb1], [emb2])[0][0]
        dot_product = np.dot(emb1, emb2)

        return {
            'euclidean_distance': euclidean_dist,
            'cosine_similarity': cosine_sim,
            'dot_product': dot_product
        }

    def compare_sentence_pairs(self, sentence_pairs):
        """여러 문장 쌍 비교"""
        results = []

        for pair in sentence_pairs:
            sent1, sent2 = pair
            similarities = self.calculate_all_similarities(sent1, sent2)

            result = {
                'sentence1': sent1,
                'sentence2': sent2,
                'euclidean': similarities['euclidean_distance'],
                'cosine': similarities['cosine_similarity'],
                'dot_product': similarities['dot_product']
            }
            results.append(result)

        return results

# 유사도 계산기 생성
similarity_calc = SimilarityCalculator(sbert_model)

# 테스트 문장 쌍들
sentence_pairs = [
    ("학생이 학교에서 공부한다", "학생이 도서관에서 공부한다"),    # 유사
    ("자연어 처리를 배운다", "컴퓨터로 언어를 분석한다"),        # 유사
    ("오늘 날씨가 좋다", "내일 비가 올 예정이다"),             # 관련
    ("영화가 재미있다", "수학 문제를 푼다")                   # 무관
]

# 유사도 비교 실행
comparison_results = similarity_calc.compare_sentence_pairs(sentence_pairs)

# 결과 출력
for i, result in enumerate(comparison_results, 1):
    print(f"\n{i}. 문장 비교:")
    print(f"   A: {result['sentence1']}")
    print(f"   B: {result['sentence2']}")
    print(f"   유클리드 거리: {result['euclidean']:.4f}")
    print(f"   코사인 유사도: {result['cosine']:.4f}")
    print(f"   내적: {result['dot_product']:.4f}")
```

### 유사도 시각화

```python
def plot_similarity_comparison(results, title="유사도 비교"):
    """유사도 비교 시각화"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 데이터 준비
    labels = [f"쌍{i+1}" for i in range(len(results))]
    euclidean_scores = [r['euclidean'] for r in results]
    cosine_scores = [r['cosine'] for r in results]
    dot_scores = [r['dot_product'] for r in results]

    # 유클리드 거리 (낮을수록 유사)
    axes[0].bar(labels, euclidean_scores, color='skyblue', alpha=0.7)
    axes[0].set_title('유클리드 거리 (낮을수록 유사)')
    axes[0].set_ylabel('거리')
    axes[0].grid(True, alpha=0.3)

    # 코사인 유사도 (높을수록 유사)
    axes[1].bar(labels, cosine_scores, color='lightgreen', alpha=0.7)
    axes[1].set_title('코사인 유사도 (높을수록 유사)')
    axes[1].set_ylabel('유사도')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    # 내적 (높을수록 유사)
    axes[2].bar(labels, dot_scores, color='salmon', alpha=0.7)
    axes[2].set_title('내적 (높을수록 유사)')
    axes[2].set_ylabel('내적값')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# 시각화 실행
plot_similarity_comparison(comparison_results, "SBERT 모델 유사도 비교")
```

## 🚀 실습해보기

### 실습: 문서 유사도 비교 시스템 구현

**목표**: 주어진 문서들을 토큰화하고 임베딩한 후 유사도를 비교하는 시스템 구현

#### 구현해야 할 기능
1. 문서를 토큰화하고 BoW 벡터로 변환
2. 문서를 임베딩으로 변환
3. 문서들 간의 유사도 계산
4. 가장 유사한 문서 쌍 찾기

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer

# 테스트용 샘플 문서
documents = [
    "인공지능은 컴퓨터 과학의 중요한 분야입니다.",
    "머신러닝은 인공지능의 하위 분야입니다.",
    "딥러닝은 머신러닝의 한 종류입니다.",
    "자연어 처리는 텍스트 데이터를 다룹니다."
]

# 문제 1: 문서를 토큰화하고 BoW 벡터로 변환하시오
def tokenize_documents(docs):
    """문서들을 토큰화하고 BoW 벡터로 변환"""
    # 한국어 토큰화
    tokenized_docs = []
    for doc in docs:
        tokens = tokenize_with_kiwi(doc)
        token_words = [token.form for token in tokens
                      if token.tag not in ['SF', 'SP', 'SS'] and len(token.form) > 1]
        tokenized_docs.append(' '.join(token_words))

    # BoW 벡터화
    vectorizer = CountVectorizer()
    bow_vectors = vectorizer.fit_transform(tokenized_docs)

    return bow_vectors, vectorizer

# 문제 2: 문서를 임베딩으로 변환하시오
def create_embeddings(docs, model):
    """SBERT를 사용하여 문서를 임베딩으로 변환"""
    embeddings = model.encode(docs)
    return embeddings

# 문제 3: 문서들 간의 유사도를 계산하시오
def calculate_similarity_matrix(vectors):
    """유사도 매트릭스 계산"""
    similarity_matrix = cosine_similarity(vectors)
    return similarity_matrix

# 문제 4: 가장 유사한 문서 쌍을 찾으시오
def find_most_similar_pair(similarity_matrix, docs):
    """가장 유사한 문서 쌍 찾기"""
    n = len(docs)
    max_similarity = 0
    most_similar_pair = (0, 1)

    # 대각선 제외하고 최대 유사도 찾기
    for i in range(n):
        for j in range(i+1, n):
            if similarity_matrix[i][j] > max_similarity:
                max_similarity = similarity_matrix[i][j]
                most_similar_pair = (i, j)

    return {
        'most_similar_pair': (docs[most_similar_pair[0]], docs[most_similar_pair[1]]),
        'similarity_score': max_similarity,
        'similarity_matrix': similarity_matrix.tolist()
    }

# 실행 예제
def run_document_similarity_analysis():
    """문서 유사도 분석 실행"""
    # 1. BoW 기반 분석
    bow_vectors, vectorizer = tokenize_documents(documents)
    bow_similarity = calculate_similarity_matrix(bow_vectors)
    bow_result = find_most_similar_pair(bow_similarity, documents)

    print("=== BoW 기반 분석 결과 ===")
    print(f"가장 유사한 문서 쌍:")
    print(f"  문서 1: {bow_result['most_similar_pair'][0]}")
    print(f"  문서 2: {bow_result['most_similar_pair'][1]}")
    print(f"  유사도: {bow_result['similarity_score']:.4f}")

    # 2. SBERT 기반 분석
    sbert_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    sbert_embeddings = create_embeddings(documents, sbert_model)
    sbert_similarity = calculate_similarity_matrix(sbert_embeddings)
    sbert_result = find_most_similar_pair(sbert_similarity, documents)

    print("\n=== SBERT 기반 분석 결과 ===")
    print(f"가장 유사한 문서 쌍:")
    print(f"  문서 1: {sbert_result['most_similar_pair'][0]}")
    print(f"  문서 2: {sbert_result['most_similar_pair'][1]}")
    print(f"  유사도: {sbert_result['similarity_score']:.4f}")

    return bow_result, sbert_result

# 분석 실행
bow_result, sbert_result = run_document_similarity_analysis()
```

## 📋 해답

### 완전한 문서 유사도 비교 시스템

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

class DocumentSimilarityAnalyzer:
    """문서 유사도 분석 클래스"""

    def __init__(self):
        self.kiwi = Kiwi()
        self.sbert_model = None

    def load_sbert_model(self, model_name='jhgan/ko-sroberta-multitask'):
        """SBERT 모델 로드"""
        try:
            self.sbert_model = SentenceTransformer(model_name)
            print(f"SBERT 모델 로드 완료: {model_name}")
        except Exception as e:
            print(f"SBERT 모델 로드 실패: {e}")

    def preprocess_documents(self, docs):
        """문서 전처리"""
        processed_docs = []
        for doc in docs:
            tokens = self.kiwi.tokenize(doc)
            # 의미있는 토큰만 추출 (명사, 동사, 형용사)
            meaningful_tokens = [
                token.form for token in tokens
                if token.tag.startswith(('NN', 'VV', 'VA', 'XR')) and len(token.form) > 1
            ]
            processed_docs.append(' '.join(meaningful_tokens))
        return processed_docs

    def create_bow_vectors(self, docs):
        """BoW 벡터 생성"""
        processed_docs = self.preprocess_documents(docs)
        vectorizer = CountVectorizer()
        vectors = vectorizer.fit_transform(processed_docs)
        return vectors.toarray(), vectorizer

    def create_tfidf_vectors(self, docs):
        """TF-IDF 벡터 생성"""
        processed_docs = self.preprocess_documents(docs)
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(processed_docs)
        return vectors.toarray(), vectorizer

    def create_sbert_embeddings(self, docs):
        """SBERT 임베딩 생성"""
        if self.sbert_model is None:
            raise ValueError("SBERT 모델이 로드되지 않았습니다.")
        return self.sbert_model.encode(docs)

    def calculate_similarity_matrix(self, vectors):
        """유사도 매트릭스 계산"""
        return cosine_similarity(vectors)

    def find_most_similar_pair(self, similarity_matrix, docs):
        """가장 유사한 문서 쌍 찾기"""
        n = len(docs)
        max_similarity = 0
        most_similar_indices = (0, 1)

        for i in range(n):
            for j in range(i+1, n):
                if similarity_matrix[i][j] > max_similarity:
                    max_similarity = similarity_matrix[i][j]
                    most_similar_indices = (i, j)

        return {
            'most_similar_pair': (docs[most_similar_indices[0]], docs[most_similar_indices[1]]),
            'indices': most_similar_indices,
            'similarity_score': max_similarity,
            'similarity_matrix': similarity_matrix
        }

    def comprehensive_analysis(self, docs):
        """종합적인 문서 유사도 분석"""
        results = {}

        # 1. BoW 분석
        bow_vectors, bow_vectorizer = self.create_bow_vectors(docs)
        bow_similarity = self.calculate_similarity_matrix(bow_vectors)
        results['bow'] = self.find_most_similar_pair(bow_similarity, docs)

        # 2. TF-IDF 분석
        tfidf_vectors, tfidf_vectorizer = self.create_tfidf_vectors(docs)
        tfidf_similarity = self.calculate_similarity_matrix(tfidf_vectors)
        results['tfidf'] = self.find_most_similar_pair(tfidf_similarity, docs)

        # 3. SBERT 분석
        if self.sbert_model:
            sbert_embeddings = self.create_sbert_embeddings(docs)
            sbert_similarity = self.calculate_similarity_matrix(sbert_embeddings)
            results['sbert'] = self.find_most_similar_pair(sbert_similarity, docs)

        return results

    def visualize_similarity_matrix(self, similarity_matrix, docs, method_name):
        """유사도 매트릭스 시각화"""
        plt.figure(figsize=(10, 8))

        # 히트맵 생성
        sns.heatmap(
            similarity_matrix,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            xticklabels=[f"문서{i+1}" for i in range(len(docs))],
            yticklabels=[f"문서{i+1}" for i in range(len(docs))],
            cbar_kws={'label': '코사인 유사도'}
        )

        plt.title(f'{method_name} 문서 유사도 매트릭스')
        plt.xlabel('문서')
        plt.ylabel('문서')
        plt.tight_layout()
        plt.show()

# 사용 예제
analyzer = DocumentSimilarityAnalyzer()
analyzer.load_sbert_model()

# 테스트 문서들
test_documents = [
    "인공지능은 컴퓨터 과학의 중요한 분야입니다.",
    "머신러닝은 인공지능의 하위 분야입니다.",
    "딥러닝은 머신러닝의 한 종류입니다.",
    "자연어 처리는 텍스트 데이터를 다룹니다.",
    "컴퓨터 비전은 이미지를 분석하는 기술입니다."
]

# 종합 분석 실행
comprehensive_results = analyzer.comprehensive_analysis(test_documents)

# 결과 출력
for method, result in comprehensive_results.items():
    print(f"\n=== {method.upper()} 분석 결과 ===")
    print(f"가장 유사한 문서 쌍 (인덱스 {result['indices']}):")
    print(f"  문서 A: {result['most_similar_pair'][0]}")
    print(f"  문서 B: {result['most_similar_pair'][1]}")
    print(f"  유사도: {result['similarity_score']:.4f}")

    # 유사도 매트릭스 시각화
    analyzer.visualize_similarity_matrix(
        result['similarity_matrix'],
        test_documents,
        method.upper()
    )

# 상세 분석 테이블 생성
def create_detailed_comparison_table(results, docs):
    """상세 비교 테이블 생성"""
    comparison_data = []

    for method, result in results.items():
        indices = result['indices']
        comparison_data.append({
            '방법': method.upper(),
            '문서1 인덱스': indices[0],
            '문서2 인덱스': indices[1],
            '유사도 점수': f"{result['similarity_score']:.4f}",
            '문서1 내용': result['most_similar_pair'][0][:30] + "...",
            '문서2 내용': result['most_similar_pair'][1][:30] + "..."
        })

    return pd.DataFrame(comparison_data)

comparison_table = create_detailed_comparison_table(comprehensive_results, test_documents)
print("\n=== 방법별 비교 결과 ===")
print(comparison_table.to_string(index=False))
```

## 🔍 참고 자료

### 공식 문서
- [Kiwi 형태소 분석기](https://github.com/bab2min/Kiwi) - 한국어 형태소 분석
- [Transformers](https://huggingface.co/docs/transformers/) - 토크나이저와 사전 훈련 모델
- [Sentence Transformers](https://www.sbert.net/) - 문장 임베딩 모델
- [Gensim](https://radimrehurek.com/gensim/) - Word2Vec 구현

### 학습 자료
- [한국어 NLP 가이드](https://wikidocs.net/book/2155) - 한국어 자연어처리
- [SBERT 논문](https://arxiv.org/abs/1908.10084) - Sentence-BERT 원리
- [Word2Vec 튜토리얼](https://radimrehurek.com/gensim/models/word2vec.html) - 단어 임베딩

### 개발 도구
- [HuggingFace Model Hub](https://huggingface.co/models) - 사전 훈련된 모델
- [Scikit-learn](https://scikit-learn.org/) - 머신러닝 도구
- [Matplotlib](https://matplotlib.org/) - 데이터 시각화

### 추가 학습
- FastText, GloVe 등 다른 단어 임베딩 모델
- Transformer 기반 최신 언어 모델 이해
- 대용량 텍스트 데이터 처리 기법
- 다국어 임베딩과 크로스링구얼 모델