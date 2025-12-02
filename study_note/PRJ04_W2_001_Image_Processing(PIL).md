# PRJ04_W2_001: PIL을 활용한 파이썬 이미지 처리 기초

## 📚 학습 목표

이 가이드를 완료하면 다음을 수행할 수 있습니다:

1. **디지털 이미지의 구조 이해**: 픽셀, 채널, 이미지 포맷의 개념과 특성 파악
2. **PIL/Pillow 라이브러리 활용**: 이미지 로드, 저장, 변환 등 기본 조작 수행
3. **이미지 변형 기법 적용**: 크기 조절, 회전, 자르기 등 다양한 변형 기법 구현
4. **이미지 필터 및 색상 조정**: 블러, 샤프닝, 엣지 검출 등 필터 적용 및 색상 보정
5. **실무 이미지 처리 파이프라인 구축**: 여러 이미지 처리 기법을 조합한 실전 프로그램 작성

## 🔑 핵심 개념

### 디지털 이미지의 구조

**픽셀 (Pixel)**
- 이미지의 최소 단위로 "Picture Element"의 줄임말
- 각 픽셀은 색상 정보를 담고 있으며, 수백만 개의 픽셀이 모여 하나의 이미지를 형성
- 이미지 크기는 가로 × 세로 픽셀 수로 표현 (예: 1920×1080)

**채널 (Channel)**
색상 정보를 담는 단위로 다음과 같은 모드가 있습니다:

| 모드 | 채널 수 | 설명 | 용도 |
|------|---------|------|------|
| RGB | 3 | Red, Green, Blue | 일반 컬러 이미지 |
| RGBA | 4 | RGB + Alpha(투명도) | 투명도가 필요한 이미지 |
| L | 1 | Grayscale (흑백) | 흑백 이미지, 성능 최적화 |
| CMYK | 4 | Cyan, Magenta, Yellow, Black | 인쇄용 이미지 |

### 이미지 포맷

| 포맷 | 압축 방식 | 특징 | 사용 사례 |
|------|-----------|------|-----------|
| JPEG/JPG | 손실 압축 | 작은 파일 크기, 투명도 미지원 | 사진, 웹 이미지 |
| PNG | 무손실 압축 | 투명도 지원, 중간 파일 크기 | 로고, 아이콘, 그래픽 |
| BMP | 무압축 | 큰 파일 크기, 빠른 처리 | 간단한 그래픽 |
| TIFF | 무손실 | 고품질, 큰 파일 크기 | 전문가용 사진 |
| GIF | 무손실 | 256색 제한, 애니메이션 지원 | 단순한 애니메이션 |

### PIL/Pillow 라이브러리

- **PIL (Python Imaging Library)**: 파이썬의 원조 이미지 처리 라이브러리
- **Pillow**: PIL의 현대적인 포크 버전으로 현재 활발히 유지보수됨
- **주요 기능**:
  - 이미지 파일 읽기/쓰기
  - 이미지 크기 조절, 회전, 자르기
  - 색상 모드 변환 (RGB ↔ Grayscale 등)
  - 필터 적용 (블러, 샤프닝, 엣지 검출)
  - 이미지 합성 및 레이어 작업
  - 텍스트 및 도형 그리기

## 🛠 환경 설정

### 1. 필수 라이브러리 설치

```bash
# Pillow 설치 (PIL의 현대적 버전)
pip install pillow

# 이미지 시각화를 위한 matplotlib
pip install matplotlib

# 배열 연산을 위한 numpy
pip install numpy
```

### 2. 기본 Import

```python
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

# Matplotlib 한글 폰트 설정 (선택사항)
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
# plt.rcParams['font.family'] = 'AppleGothic'  # macOS
```

### 3. 샘플 이미지 다운로드

실습을 위한 샘플 이미지를 다운로드합니다:

```python
# User-Agent 헤더 추가 (일부 서버에서 요구)
opener = urllib.request.build_opener()
opener.addheaders = [
    ('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
]
urllib.request.install_opener(opener)

# 반 고흐 자화상 이미지 다운로드
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/VanGogh_1887_Selbstbildnis.jpg/507px-VanGogh_1887_Selbstbildnis.jpg"
urllib.request.urlretrieve(url, 'portrait.jpg')

print("✅ 샘플 이미지 다운로드 완료: portrait.jpg")
```

## 💻 단계별 구현

### 1단계: 이미지 열기와 정보 확인

#### 이미지 열기

```python
from PIL import Image

# 이미지 파일 열기
img = Image.open('portrait.jpg')

# 이미지 기본 정보 출력
print(f"이미지 크기: {img.size}")      # (width, height)
print(f"이미지 형식: {img.format}")    # JPEG, PNG 등
print(f"이미지 모드: {img.mode}")      # RGB, RGBA, L 등
```

**출력 예시:**
```
이미지 크기: (507, 640)
이미지 형식: JPEG
이미지 모드: RGB
```

**💡 핵심 포인트:**
- `Image.open()`: 이미지 파일을 메모리에 로드
- `size`: (width, height) 튜플 형태로 반환
- `format`: 원본 파일 형식 (저장 시 다른 형식으로 변환 가능)
- `mode`: 색상 모드 (RGB가 가장 일반적)

#### 이미지 시각화

```python
# Matplotlib으로 이미지 표시
plt.figure(figsize=(8, 10))
plt.imshow(img)
plt.axis('off')  # 축 숨기기
plt.title('원본 이미지')
plt.show()
```

### 2단계: 이미지 저장하기

#### 다른 형식으로 저장

```python
# PNG 형식으로 저장
img.save('portrait_saved.png')

# JPEG 품질 설정 (1-95, 높을수록 고품질)
img.save('portrait_saved_95.jpg', quality=95)

# 웹 최적화 JPEG 저장
img.save('portrait_web.jpg', quality=85, optimize=True)

print("✅ 이미지 저장 완료")
```

**품질 설정 가이드:**
- `quality=95`: 최고 품질 (파일 크기 큼)
- `quality=85`: 웹용 권장 (품질과 크기의 균형)
- `quality=60`: 저용량 (눈에 띄는 품질 저하)
- `optimize=True`: 파일 크기 추가 최적화

### 3단계: 이미지 크기 조정

#### 특정 크기로 조정

```python
print(f"원본 이미지 크기: {img.size}")

# 800×600 픽셀로 조정 (비율 무시)
resized_img = img.resize((800, 600))

print(f"조정된 이미지 크기: {resized_img.size}")

# 시각화
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title(f'원본 ({img.size[0]}×{img.size[1]})')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(resized_img)
plt.title(f'조정 ({resized_img.size[0]}×{resized_img.size[1]})')
plt.axis('off')

plt.tight_layout()
plt.show()
```

#### 비율 유지하면서 조정

```python
# 가로 800px로 조정하면서 세로는 비율에 맞게 자동 계산
width, height = img.size
ratio = height / width
new_width = 800
new_height = int(new_width * ratio)

proportional_img = img.resize((new_width, new_height))

print(f"비율 유지 조정된 이미지 크기: {proportional_img.size}")
```

**💡 실무 팁:**
- 썸네일 생성 시 `thumbnail()` 메서드 사용:
  ```python
  img.thumbnail((200, 200))  # 최대 크기 지정, 비율 자동 유지
  ```

### 4단계: 이미지 회전

#### 90도 단위 회전

```python
# 90도 회전
rotated_90 = img.rotate(90)

# 시각화
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('원본')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(rotated_90)
plt.title('90도 회전')
plt.axis('off')

plt.tight_layout()
plt.show()
```

#### 임의 각도 회전

```python
# 45도 회전 (배경색 지정, 캔버스 확장)
rotated_45 = img.rotate(45, expand=True, fillcolor='white')

plt.imshow(rotated_45)
plt.title('45도 회전 (캔버스 확장)')
plt.axis('off')
plt.show()
```

**파라미터 설명:**
- `angle`: 회전 각도 (반시계 방향)
- `expand`: `True`면 전체 이미지가 보이도록 캔버스 확장
- `fillcolor`: 빈 공간을 채울 색상

### 5단계: 이미지 자르기

#### 특정 영역 자르기

```python
# crop((left, top, right, bottom))
# 좌표는 픽셀 단위로 지정
cropped = img.crop((100, 100, 400, 400))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('원본')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cropped)
plt.title('자른 이미지 (100,100)-(400,400)')
plt.axis('off')

plt.tight_layout()
plt.show()
```

**💡 실무 활용:**
```python
# 중앙 자르기 함수
def center_crop(image, crop_width, crop_height):
    """이미지 중앙을 기준으로 자르기"""
    img_width, img_height = image.size
    left = (img_width - crop_width) // 2
    top = (img_height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height
    return image.crop((left, top, right, bottom))

# 사용 예시
center_cropped = center_crop(img, 300, 300)
plt.imshow(center_cropped)
plt.title('중앙 300×300 자르기')
plt.axis('off')
plt.show()
```

### 6단계: 이미지 필터 적용

#### 블러 필터 (흐림 효과)

```python
from PIL import ImageFilter

# 블러 필터 적용
blurred = img.filter(ImageFilter.BLUR)

# 가우시안 블러 (더 자연스러운 흐림)
gaussian_blurred = img.filter(ImageFilter.GaussianBlur(radius=5))

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img)
axes[0].set_title('원본')
axes[0].axis('off')

axes[1].imshow(blurred)
axes[1].set_title('일반 블러')
axes[1].axis('off')

axes[2].imshow(gaussian_blurred)
axes[2].set_title('가우시안 블러 (radius=5)')
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

#### 샤프닝 필터 (선명하게)

```python
# 샤프닝 필터 적용
sharpened = img.filter(ImageFilter.SHARPEN)

# 더 강한 샤프닝
unsharp_mask = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150))

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img)
axes[0].set_title('원본')
axes[0].axis('off')

axes[1].imshow(sharpened)
axes[1].set_title('샤프닝')
axes[1].axis('off')

axes[2].imshow(unsharp_mask)
axes[2].set_title('언샤프 마스크')
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

#### 엣지 검출 필터

```python
# 엣지 검출 필터
edges = img.filter(ImageFilter.FIND_EDGES)

# 윤곽선 검출
contour = img.filter(ImageFilter.CONTOUR)

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img)
axes[0].set_title('원본')
axes[0].axis('off')

axes[1].imshow(edges)
axes[1].set_title('엣지 검출')
axes[1].axis('off')

axes[2].imshow(contour)
axes[2].set_title('윤곽선 검출')
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

**주요 필터 정리:**

| 필터 | 효과 | 용도 |
|------|------|------|
| `BLUR` | 이미지 흐림 | 노이즈 제거, 부드러운 효과 |
| `GaussianBlur` | 자연스러운 흐림 | 배경 흐림, 포커스 효과 |
| `SHARPEN` | 이미지 선명화 | 디테일 강조, 품질 개선 |
| `FIND_EDGES` | 엣지 검출 | 객체 윤곽선 추출 |
| `CONTOUR` | 윤곽선 검출 | 경계선 강조 |
| `EMBOSS` | 양각 효과 | 예술적 효과 |

### 7단계: 이미지 색상 조정

#### 흑백 변환

```python
# 그레이스케일로 변환
grayscale = img.convert('L')

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(img)
axes[0].set_title('컬러 이미지')
axes[0].axis('off')

axes[1].imshow(grayscale, cmap='gray')
axes[1].set_title('흑백 이미지')
axes[1].axis('off')

plt.tight_layout()
plt.show()
```

#### 밝기 조정

```python
from PIL import ImageEnhance

# 밝기 조정 (1.0 = 원본, >1.0 = 밝게, <1.0 = 어둡게)
enhancer = ImageEnhance.Brightness(img)
brightened = enhancer.enhance(1.5)  # 50% 더 밝게
darkened = enhancer.enhance(0.7)    # 30% 더 어둡게

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(darkened)
axes[0].set_title('어두움 (0.7)')
axes[0].axis('off')

axes[1].imshow(img)
axes[1].set_title('원본 (1.0)')
axes[1].axis('off')

axes[2].imshow(brightened)
axes[2].set_title('밝음 (1.5)')
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

#### 대비 조정

```python
# 대비 조정
contrast_enhancer = ImageEnhance.Contrast(img)
high_contrast = contrast_enhancer.enhance(2.0)  # 대비 증가
low_contrast = contrast_enhancer.enhance(0.5)   # 대비 감소

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(low_contrast)
axes[0].set_title('낮은 대비 (0.5)')
axes[0].axis('off')

axes[1].imshow(img)
axes[1].set_title('원본 (1.0)')
axes[1].axis('off')

axes[2].imshow(high_contrast)
axes[2].set_title('높은 대비 (2.0)')
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

#### 채도 조정

```python
# 채도 조정
color_enhancer = ImageEnhance.Color(img)
desaturated = color_enhancer.enhance(0.3)  # 채도 낮춤
saturated = color_enhancer.enhance(2.0)    # 채도 높임

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(desaturated)
axes[0].set_title('낮은 채도 (0.3)')
axes[0].axis('off')

axes[1].imshow(img)
axes[1].set_title('원본 (1.0)')
axes[1].axis('off')

axes[2].imshow(saturated)
axes[2].set_title('높은 채도 (2.0)')
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

**ImageEnhance 모듈 정리:**

| Enhancer | 조정 항목 | 권장 범위 |
|----------|-----------|-----------|
| `Brightness` | 밝기 | 0.5 ~ 1.5 |
| `Contrast` | 대비 | 0.5 ~ 2.0 |
| `Color` | 채도 | 0.0 ~ 2.0 |
| `Sharpness` | 선명도 | 0.0 ~ 2.0 |

### 8단계: 여러 이미지 합치기

#### 가로로 이미지 합치기

```python
def combine_images_horizontal(images):
    """여러 이미지를 가로로 합치기"""
    # 총 너비 계산
    total_width = sum(img.size[0] for img in images)
    # 최대 높이 찾기
    max_height = max(img.size[1] for img in images)

    # 새 캔버스 생성
    combined = Image.new('RGB', (total_width, max_height))

    # 이미지 순서대로 붙이기
    x_offset = 0
    for img in images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.size[0]

    return combined

# 사용 예시
img1 = img.resize((200, 250))
img2 = grayscale.convert('RGB').resize((200, 250))
img3 = edges.convert('RGB').resize((200, 250))

combined_horizontal = combine_images_horizontal([img1, img2, img3])

plt.figure(figsize=(12, 4))
plt.imshow(combined_horizontal)
plt.title('가로로 합친 이미지')
plt.axis('off')
plt.show()
```

#### 세로로 이미지 합치기

```python
def combine_images_vertical(images):
    """여러 이미지를 세로로 합치기"""
    # 최대 너비 찾기
    max_width = max(img.size[0] for img in images)
    # 총 높이 계산
    total_height = sum(img.size[1] for img in images)

    # 새 캔버스 생성
    combined = Image.new('RGB', (max_width, total_height))

    # 이미지 순서대로 붙이기
    y_offset = 0
    for img in images:
        combined.paste(img, (0, y_offset))
        y_offset += img.size[1]

    return combined

# 사용 예시
combined_vertical = combine_images_vertical([img1, img2, img3])

plt.figure(figsize=(4, 12))
plt.imshow(combined_vertical)
plt.title('세로로 합친 이미지')
plt.axis('off')
plt.show()
```

## 🎯 실습 문제

### 실습 1: 이미지 처리 파이프라인 (기초)

**문제**: 사용자로부터 이미지 파일을 입력받아 다음 작업을 수행하는 프로그램을 작성하세요:

1. 이미지 크기를 50% 축소
2. 이미지를 180도 회전
3. 결과 이미지를 PNG 형식으로 저장

**입력**: 이미지 파일 경로 (JPG, PNG 등)

**출력**: 처리된 이미지를 'output.png'로 저장

**제약사항**:
- PIL/Pillow 라이브러리 사용
- 이미지 품질 유지

**힌트**:
```python
def process_image(image_path):
    # 1. 이미지 열기
    # 2. 크기 50% 축소
    # 3. 180도 회전
    # 4. PNG로 저장
    pass
```

### 실습 2: 썸네일 생성기 (응용)

**문제**: 원본 이미지의 썸네일을 생성하는 함수를 작성하세요:

**요구사항**:
- 비율을 유지하면서 최대 크기 200×200 픽셀
- 배경이 흰색인 정사각형 캔버스 (200×200)
- 이미지를 중앙에 배치
- 파일명: 원본명_thumbnail.jpg

**입력**: 이미지 경로

**출력**: 썸네일 이미지 파일

**힌트**:
```python
def create_thumbnail(image_path, size=(200, 200)):
    # 1. 이미지 로드
    # 2. 비율 유지하면서 축소
    # 3. 흰색 정사각형 캔버스 생성
    # 4. 이미지를 중앙에 붙여넣기
    # 5. 저장
    pass
```

### 실습 3: 이미지 필터 비교 도구 (심화)

**문제**: 하나의 이미지에 여러 필터를 적용하고 2×3 그리드로 비교하는 프로그램을 작성하세요:

**요구사항**:
- 원본, 블러, 샤프닝, 엣지 검출, 흑백, 밝게(1.5배) 총 6개 버전 생성
- Matplotlib으로 2×3 그리드 시각화
- 각 이미지에 제목 표시
- 하나의 이미지로 저장 (comparison.png)

**입력**: 이미지 경로

**출력**: comparison.png 파일

### 실습 4: 사진 일괄 처리 (실전)

**문제**: 폴더 내 모든 이미지를 일괄 처리하는 프로그램을 작성하세요:

**요구사항**:
- 지정된 폴더의 모든 JPG, PNG 파일 검색
- 각 이미지를 800×600으로 리사이즈 (비율 유지)
- 파일명에 '_resized' 추가
- 'output' 폴더에 저장
- 처리된 파일 수 출력

**입력**: 입력 폴더 경로, 출력 폴더 경로

**출력**: 리사이즈된 이미지들

## ✅ 솔루션 예시

### 실습 1 솔루션

```python
def process_image(image_path):
    """이미지 처리 파이프라인"""
    from PIL import Image

    # 1. 이미지 열기
    img = Image.open(image_path)

    # 2. 이미지 크기를 50% 축소
    width, height = img.size
    new_width = int(width * 0.5)
    new_height = int(height * 0.5)
    resized_img = img.resize((new_width, new_height))

    # 3. 이미지를 180도 회전
    rotated_img = resized_img.rotate(180)

    # 4. 결과 이미지를 PNG 형식으로 저장
    rotated_img.save('output.png')

    print(f"✅ 이미지 처리 완료!")
    print(f"   원본 크기: {img.size}")
    print(f"   축소된 크기: {resized_img.size}")
    print(f"   출력 파일: output.png")

    return rotated_img

# 실행 예시
result = process_image("portrait.jpg")
plt.imshow(result)
plt.title('처리된 이미지')
plt.axis('off')
plt.show()
```

### 실습 2 솔루션

```python
def create_thumbnail(image_path, size=(200, 200)):
    """썸네일 생성 함수"""
    from PIL import Image
    import os

    # 1. 이미지 로드
    img = Image.open(image_path)

    # 2. 비율 유지하면서 축소
    img.thumbnail(size, Image.Resampling.LANCZOS)

    # 3. 흰색 정사각형 캔버스 생성
    canvas = Image.new('RGB', size, 'white')

    # 4. 이미지를 중앙에 붙여넣기
    x = (size[0] - img.size[0]) // 2
    y = (size[1] - img.size[1]) // 2
    canvas.paste(img, (x, y))

    # 5. 파일명 생성 및 저장
    filename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f"{filename}_thumbnail.jpg"
    canvas.save(output_path, quality=90)

    print(f"✅ 썸네일 생성 완료: {output_path}")
    print(f"   원본 크기: {Image.open(image_path).size}")
    print(f"   썸네일 크기: {canvas.size}")

    return canvas

# 실행 예시
thumbnail = create_thumbnail("portrait.jpg")
plt.imshow(thumbnail)
plt.title('썸네일')
plt.axis('off')
plt.show()
```

### 실습 3 솔루션

```python
def compare_filters(image_path):
    """여러 필터를 적용하고 비교 시각화"""
    from PIL import Image, ImageFilter, ImageEnhance
    import matplotlib.pyplot as plt

    # 이미지 로드
    img = Image.open(image_path)

    # 다양한 필터 적용
    filters = {
        '원본': img,
        '블러': img.filter(ImageFilter.BLUR),
        '샤프닝': img.filter(ImageFilter.SHARPEN),
        '엣지 검출': img.filter(ImageFilter.FIND_EDGES),
        '흑백': img.convert('L'),
        '밝게 (1.5배)': ImageEnhance.Brightness(img).enhance(1.5)
    }

    # 2×3 그리드로 시각화
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (title, filtered_img) in enumerate(filters.items()):
        if filtered_img.mode == 'L':  # 흑백 이미지
            axes[idx].imshow(filtered_img, cmap='gray')
        else:
            axes[idx].imshow(filtered_img)
        axes[idx].set_title(title, fontsize=12)
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("✅ 필터 비교 이미지 생성 완료: comparison.png")

# 실행 예시
compare_filters("portrait.jpg")
```

### 실습 4 솔루션

```python
def batch_resize_images(input_folder, output_folder, max_size=(800, 600)):
    """폴더 내 모든 이미지를 일괄 리사이즈"""
    from PIL import Image
    import os
    import glob

    # 출력 폴더 생성
    os.makedirs(output_folder, exist_ok=True)

    # 지원하는 이미지 확장자
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']

    # 모든 이미지 파일 찾기
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))

    if not image_files:
        print(f"⚠️ {input_folder}에서 이미지를 찾을 수 없습니다.")
        return

    print(f"🔄 {len(image_files)}개의 이미지 처리 중...")

    processed_count = 0
    for image_path in image_files:
        try:
            # 이미지 열기
            img = Image.open(image_path)

            # 비율 유지하면서 리사이즈
            img.thumbnail(max_size, Image.Resampling.LANCZOS)

            # 출력 파일명 생성
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_resized{ext}"
            output_path = os.path.join(output_folder, output_filename)

            # 저장
            img.save(output_path, quality=90)
            processed_count += 1

            print(f"  ✅ {filename} → {output_filename}")

        except Exception as e:
            print(f"  ❌ {filename} 처리 실패: {e}")

    print(f"\n✅ 총 {processed_count}개의 이미지 처리 완료")
    print(f"   출력 폴더: {output_folder}")

# 실행 예시
batch_resize_images(
    input_folder="./images",
    output_folder="./output",
    max_size=(800, 600)
)
```

## 🚀 실무 활용 예시

### 예시 1: 프로필 사진 자동 처리

웹사이트나 앱의 사용자 프로필 사진을 자동으로 처리하는 시스템:

```python
from PIL import Image, ImageFilter, ImageEnhance
import os

class ProfileImageProcessor:
    """프로필 사진 자동 처리 클래스"""

    def __init__(self, output_dir="./profile_images"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def process_profile_image(self, image_path, user_id):
        """
        프로필 사진 처리 파이프라인
        - 얼굴 중심 자르기 (정사각형)
        - 여러 크기의 썸네일 생성
        - 약간의 보정 적용
        """
        # 1. 이미지 로드
        img = Image.open(image_path)

        # 2. RGB 모드로 변환 (RGBA나 다른 모드 처리)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # 3. 중앙 정사각형으로 자르기
        width, height = img.size
        min_dimension = min(width, height)

        left = (width - min_dimension) // 2
        top = (height - min_dimension) // 2
        right = left + min_dimension
        bottom = top + min_dimension

        img = img.crop((left, top, right, bottom))

        # 4. 이미지 품질 향상 (약간의 보정)
        # 선명도 향상
        img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150))
        # 약간 밝게
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.1)

        # 5. 여러 크기로 저장
        sizes = {
            'large': 400,
            'medium': 200,
            'small': 100,
            'thumbnail': 50
        }

        saved_files = {}
        for size_name, size in sizes.items():
            # 크기 조정
            resized = img.resize((size, size), Image.Resampling.LANCZOS)

            # 파일명 생성
            filename = f"user_{user_id}_{size_name}.jpg"
            filepath = os.path.join(self.output_dir, filename)

            # 저장
            resized.save(filepath, quality=90, optimize=True)
            saved_files[size_name] = filepath

        return saved_files

    def create_placeholder_avatar(self, user_id, initials, size=200):
        """
        사용자가 이미지를 업로드하지 않은 경우 기본 아바타 생성
        """
        from PIL import ImageDraw, ImageFont

        # 원형 배경 생성
        img = Image.new('RGB', (size, size), color='#3498db')
        draw = ImageDraw.Draw(img)

        # 원형 마스크
        mask = Image.new('L', (size, size), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.ellipse((0, 0, size, size), fill=255)

        # 이니셜 텍스트 추가 (간단한 버전)
        draw.text(
            (size//2, size//2),
            initials.upper(),
            fill='white',
            anchor='mm'
        )

        # 마스크 적용
        output = Image.new('RGB', (size, size), (255, 255, 255))
        output.paste(img, (0, 0), mask)

        # 저장
        filename = f"user_{user_id}_avatar.jpg"
        filepath = os.path.join(self.output_dir, filename)
        output.save(filepath, quality=90)

        return filepath

# 사용 예시
processor = ProfileImageProcessor()

# 프로필 사진 처리
saved_files = processor.process_profile_image("portrait.jpg", user_id=12345)

print("✅ 프로필 사진 처리 완료:")
for size_name, filepath in saved_files.items():
    print(f"   {size_name}: {filepath}")

# 기본 아바타 생성
placeholder = processor.create_placeholder_avatar(user_id=67890, initials="JD")
print(f"\n✅ 기본 아바타 생성 완료: {placeholder}")
```

### 예시 2: 상품 이미지 자동 최적화

이커머스 사이트의 상품 이미지를 자동으로 최적화:

```python
from PIL import Image, ImageEnhance
import os

class ProductImageOptimizer:
    """상품 이미지 최적화 클래스"""

    def __init__(self, output_dir="./product_images"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def optimize_product_image(self, image_path, product_id):
        """
        상품 이미지 최적화
        - 배경 흰색으로 통일
        - 여러 크기 생성 (상세, 목록, 썸네일)
        - 웹 최적화 (파일 크기 최소화)
        """
        # 이미지 로드
        img = Image.open(image_path)

        # RGB 모드로 변환
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # 이미지 품질 향상
        # 1. 약간의 선명도 향상
        img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120))

        # 2. 약간의 채도 향상 (상품이 더 매력적으로 보임)
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.1)

        # 3. 약간 밝게
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.05)

        # 여러 크기로 저장
        sizes = {
            'detail': (1200, 1200),    # 상세 페이지
            'list': (600, 600),         # 목록 페이지
            'thumbnail': (300, 300),    # 썸네일
            'small': (150, 150)         # 작은 썸네일
        }

        saved_files = {}
        for size_name, max_size in sizes.items():
            # 비율 유지하면서 리사이즈
            img_copy = img.copy()
            img_copy.thumbnail(max_size, Image.Resampling.LANCZOS)

            # 흰색 배경의 정사각형 캔버스에 중앙 배치
            canvas = Image.new('RGB', max_size, 'white')
            x = (max_size[0] - img_copy.size[0]) // 2
            y = (max_size[1] - img_copy.size[1]) // 2
            canvas.paste(img_copy, (x, y))

            # 파일명 생성
            filename = f"product_{product_id}_{size_name}.jpg"
            filepath = os.path.join(self.output_dir, filename)

            # 크기별 최적화 품질 설정
            quality = 95 if size_name == 'detail' else 85

            # 저장
            canvas.save(filepath, quality=quality, optimize=True)
            saved_files[size_name] = filepath

        return saved_files

    def create_image_with_watermark(self, image_path, product_id, watermark_text="© MyShop"):
        """워터마크가 있는 이미지 생성"""
        from PIL import ImageDraw, ImageFont

        # 이미지 로드
        img = Image.open(image_path)

        # RGB 모드로 변환
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # 복사본 생성 (원본 보존)
        img_with_watermark = img.copy()
        draw = ImageDraw.Draw(img_with_watermark)

        # 워터마크 위치 (오른쪽 하단)
        width, height = img_with_watermark.size
        text_position = (width - 150, height - 30)

        # 워터마크 텍스트 추가 (폰트가 없으면 기본 폰트 사용)
        draw.text(
            text_position,
            watermark_text,
            fill=(200, 200, 200, 128)  # 반투명 회색
        )

        # 저장
        filename = f"product_{product_id}_watermarked.jpg"
        filepath = os.path.join(self.output_dir, filename)
        img_with_watermark.save(filepath, quality=90)

        return filepath

# 사용 예시
optimizer = ProductImageOptimizer()

# 상품 이미지 최적화
saved_files = optimizer.optimize_product_image("portrait.jpg", product_id=98765)

print("✅ 상품 이미지 최적화 완료:")
for size_name, filepath in saved_files.items():
    file_size = os.path.getsize(filepath) / 1024  # KB
    print(f"   {size_name}: {filepath} ({file_size:.1f} KB)")

# 워터마크 추가
watermarked = optimizer.create_image_with_watermark("portrait.jpg", product_id=98765)
print(f"\n✅ 워터마크 이미지 생성 완료: {watermarked}")
```

### 예시 3: 이미지 품질 자동 검사

업로드된 이미지의 품질을 자동으로 검사하는 시스템:

```python
from PIL import Image
import numpy as np

class ImageQualityChecker:
    """이미지 품질 자동 검사 클래스"""

    def __init__(self):
        self.min_width = 800
        self.min_height = 600
        self.max_file_size_mb = 5
        self.min_file_size_kb = 50

    def check_image_quality(self, image_path):
        """
        이미지 품질 종합 검사
        반환: (is_valid, issues, metrics)
        """
        issues = []
        metrics = {}

        try:
            # 파일 크기 확인
            file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
            file_size_kb = os.path.getsize(image_path) / 1024
            metrics['file_size_mb'] = round(file_size_mb, 2)

            if file_size_mb > self.max_file_size_mb:
                issues.append(f"파일 크기가 너무 큼 ({file_size_mb:.1f}MB > {self.max_file_size_mb}MB)")

            if file_size_kb < self.min_file_size_kb:
                issues.append(f"파일 크기가 너무 작음 ({file_size_kb:.1f}KB < {self.min_file_size_kb}KB)")

            # 이미지 로드
            img = Image.open(image_path)

            # 이미지 크기 확인
            width, height = img.size
            metrics['width'] = width
            metrics['height'] = height

            if width < self.min_width:
                issues.append(f"너비가 너무 작음 ({width}px < {self.min_width}px)")

            if height < self.min_height:
                issues.append(f"높이가 너무 작음 ({height}px < {self.min_height}px)")

            # 이미지 모드 확인
            metrics['mode'] = img.mode
            if img.mode not in ['RGB', 'RGBA', 'L']:
                issues.append(f"지원하지 않는 색상 모드: {img.mode}")

            # 이미지 품질 분석 (선명도)
            sharpness_score = self._calculate_sharpness(img)
            metrics['sharpness'] = round(sharpness_score, 2)

            if sharpness_score < 50:
                issues.append(f"이미지가 흐릿함 (선명도: {sharpness_score:.1f})")

            # 밝기 분석
            brightness_score = self._calculate_brightness(img)
            metrics['brightness'] = round(brightness_score, 2)

            if brightness_score < 30:
                issues.append("이미지가 너무 어두움")
            elif brightness_score > 220:
                issues.append("이미지가 너무 밝음")

            # 최종 판정
            is_valid = len(issues) == 0

            return is_valid, issues, metrics

        except Exception as e:
            return False, [f"이미지 처리 오류: {str(e)}"], {}

    def _calculate_sharpness(self, img):
        """이미지 선명도 계산 (라플라시안 분산)"""
        # 그레이스케일로 변환
        if img.mode != 'L':
            img_gray = img.convert('L')
        else:
            img_gray = img

        # NumPy 배열로 변환
        img_array = np.array(img_gray)

        # 라플라시안 필터 적용 (엣지 검출)
        laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

        # 간단한 컨볼루션 (scipy 없이)
        variance = np.var(img_array)

        return variance

    def _calculate_brightness(self, img):
        """이미지 평균 밝기 계산"""
        # 그레이스케일로 변환
        if img.mode != 'L':
            img_gray = img.convert('L')
        else:
            img_gray = img

        # NumPy 배열로 변환
        img_array = np.array(img_gray)

        # 평균 밝기
        return np.mean(img_array)

    def generate_quality_report(self, image_path):
        """이미지 품질 리포트 생성"""
        is_valid, issues, metrics = self.check_image_quality(image_path)

        print("=" * 60)
        print(f"이미지 품질 검사 리포트: {os.path.basename(image_path)}")
        print("=" * 60)

        print("\n📊 측정 지표:")
        for key, value in metrics.items():
            print(f"   {key}: {value}")

        if is_valid:
            print("\n✅ 품질 검사 통과")
        else:
            print("\n❌ 품질 검사 실패")
            print("\n⚠️ 문제점:")
            for issue in issues:
                print(f"   - {issue}")

        print("=" * 60)

        return is_valid, issues, metrics

# 사용 예시
checker = ImageQualityChecker()

# 이미지 품질 검사
is_valid, issues, metrics = checker.generate_quality_report("portrait.jpg")

if is_valid:
    print("\n이미지를 업로드할 수 있습니다.")
else:
    print("\n이미지를 수정한 후 다시 업로드해주세요.")
```

## 📖 참고 자료

### 공식 문서
- [Pillow 공식 문서](https://pillow.readthedocs.io/)
- [PIL/Pillow Tutorial](https://pillow.readthedocs.io/en/stable/handbook/tutorial.html)
- [Image Module Reference](https://pillow.readthedocs.io/en/stable/reference/Image.html)

### 추가 학습 자료
- [Real Python - Image Processing with Pillow](https://realpython.com/image-processing-with-the-python-pillow-library/)
- [Python Image Processing Cookbook](https://github.com/PacktPublishing/Python-Image-Processing-Cookbook)

### 관련 라이브러리
- **OpenCV**: 고급 컴퓨터 비전 및 이미지 처리
- **scikit-image**: 과학적 이미지 처리
- **imageio**: 다양한 이미지 형식 입출력
- **imgaug**: 이미지 증강 (딥러닝용)

---

이 가이드를 완료하셨다면 PIL/Pillow를 활용한 기본적인 이미지 처리를 수행할 수 있습니다. 다음 단계로 OpenCV를 활용한 고급 이미지 처리를 학습하시길 권장합니다! 🎉
