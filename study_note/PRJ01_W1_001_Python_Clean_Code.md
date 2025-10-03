# Python 클린 코드 가이드

## 📚 학습 목표
Python 코드 컨벤션과 클린코드 원칙을 적용하여 읽기 쉽고 유지보수가 용이한 코드를 작성할 수 있다.

## 📖 주요 내용
1. Python 기초 문법 복습
2. 클린 코드 작성 원칙
3. 실무에서 적용할 수 있는 코딩 스타일
4. 타입 힌트와 문서화 방법

---

# 1단계: 파이썬 기초 개념

## 1.1 변수와 데이터 타입

### 핵심 개념
- **변수**는 메모리 공간의 이름으로, 파이썬의 _동적 타이핑_ 특성상 타입 선언 없이 자유롭게 값 할당 가능
- 기본 데이터 타입은 **숫자형**, **문자열**, **불리언**, **리스트**, **튜플**, **딕셔너리**, **집합** 등이 있으며 각각 고유한 특성 보유
- `type()` 함수로 데이터 타입을 확인하고, **타입 변환 함수**(`int()`, `str()`, `float()`)를 통해 데이터 타입 변환 가능

### 코드 예시
```python
# 기본 데이터 타입 예제
number = 42          # 정수형(int)
decimal = 3.14       # 실수형(float)
text = "Hello"       # 문자열(str)
is_valid = True      # 불리언(bool)

# 데이터 타입 확인하기
print(type(number))  # <class 'int'>
print(type(text))    # <class 'str'>
```

## 1.2 기본 연산자

### 연산자 종류
- **산술 연산자**(`+`, `-`, `*`, `/`, `%`, `**`)로 기본 수학 연산을 수행하며, _정수 나눗셈_은 `//` 연산자 사용
- **비교 연산자**(`==`, `!=`, `>`, `<`, `>=`, `<=`)와 **논리 연산자**(`and`, `or`, `not`)를 조합하여 조건문 구성 가능
- **할당 연산자**(`=`, `+=`, `-=`, `*=`, `/=`)를 통해 변수에 값을 할당하거나 연산과 할당을 동시 수행

### 코드 예시
```python
# 산술 연산자
result1 = 10 + 5    # 덧셈: 15
result2 = 10 - 5    # 뺄셈: 5
result3 = 10 * 5    # 곱셈: 50
result4 = 10 / 5    # 나눗셈: 2.0
result5 = 10 % 3    # 나머지: 1
result6 = 10 ** 2   # 거듭제곱: 100

# 비교 연산자
is_equal = 10 == 5      # False
is_greater = 10 > 5     # True

# 논리 연산자
result1 = True and False   # False
result2 = True or False    # True

# 할당 연산자
number = 10
number += 5     # number = number + 5 → 15
```

---

# 2단계: 제어문

## 2.1 조건문

### 핵심 개념
- **조건문**은 `if`, `elif`, `else`를 사용하며, 들여쓰기로 코드 블록을 구분
- **조건식**에서는 비교/논리 연산자 사용 가능하며, 빈 컨테이너(`[]`, `()`, `{}`), `0`, `None`은 거짓으로 평가
- **삼항 연산자**(`결과 = A if 조건 else B`)로 간단한 조건문을 한 줄로 표현 가능

### 코드 예시
```python
# 조건문 사용 - 점수에 따른 등급을 출력
score = 85

if score >= 90:
    print("A")
elif score >= 80:
    print("B")
elif score >= 70:
    print("C")
else:
    print("F")

# 삼항 연산자
score = 85
result = "pass" if score >= 80 else "fail"
print(result)   # pass
```

## 2.2 반복문

### 핵심 개념
- **반복문**은 `for`와 `while` 두 종류가 있으며, `for`는 시퀀스 순회에, `while`은 조건부 반복에 사용
- **제어 키워드** `break`, `continue`, `else`를 통해 반복문의 세밀한 흐름 제어 가능
- **내장 함수** `enumerate()`, `zip()` 등과 함께 사용하여 다양한 반복 패턴 구현 가능

### 코드 예시
```python
# for 반복문 - 5번 반복
for i in range(5):
    print(f"반복 {i+1}번째")

# while 반복문 - 조건을 충족할 때까지 반복
count = 0
while count < 5:
    print(f"현재 카운트: {count}")
    count += 1

# break 키워드 사용 - 반복문 종료
for i in range(10):
    if i == 5:
        break
    print(i)  # 0, 1, 2, 3, 4 출력

# continue 키워드 사용 - 나머지 부분 건너뛰기
for i in range(10):
    if i % 2 == 0:
        continue
    print(i)  # 1, 3, 5, 7, 9 출력

# enumerate 함수 - 인덱스와 원소를 함께 사용
fruits = ["apple", "banana", "cherry"]
for index, fruit in enumerate(fruits):
    print(f"인덱스: {index}, 과일: {fruit}")

# zip 함수 - 여러 리스트를 같은 인덱스끼리 묶기
fruits = ["apple", "banana", "cherry"]
colors = ["red", "yellow", "pink"]
for fruit, color in zip(fruits, colors):
    print(f"과일: {fruit}, 색상: {color}")
```

---

# 3단계: 자료구조

## 3.1 리스트 (List)

### 핵심 개념
- **리스트**는 순서가 있고 변경 가능한 시퀀스로, `[]`로 생성하며 다양한 타입의 요소 저장 가능
- **주요 메서드**로 `append()`, `extend()`, `insert()`, `remove()`, `pop()`, `sort()`를 제공하여 **데이터 조작** 가능
- **리스트 컴프리헨션**을 통해 반복문과 조건문을 한 줄로 표현하여 새로운 리스트를 효율적으로 생성

### 코드 예시
```python
# 리스트 생성과 조작
numbers = [1, 2, 3, 4, 5]
fruits = ["사과", "바나나", "오렌지"]

# 인덱싱 - 특정 위치의 원소 가져오기
print(numbers[0])   # 1
print(fruits[1])    # 바나나

# 슬라이싱 - 리스트의 일부분 가져오기
print(numbers[1:3]) # [2, 3]
print(fruits[:2])   # ['사과', '바나나']

# 요소 추가 (append) : 리스트 끝에 요소 추가
numbers.append(6)
print(numbers)  # [1, 2, 3, 4, 5, 6]

# 요소 추가 (insert) : 특정 위치에 요소 추가
fruits.insert(1, "배")
print(fruits)   # ['사과', '배', '바나나', '오렌지']

# 마지막 요소 제거 (pop)
numbers.pop()
print(numbers)  # [1, 2, 3, 4, 5]

# 리스트 컴프리헨션 - 리스트를 간단하게 생성
odd_numbers = [i for i in range(10) if i % 2 == 1]
print(odd_numbers)  # [1, 3, 5, 7, 9]
```

## 3.2 딕셔너리(Dictionary)

### 핵심 개념
- **딕셔너리**는 키-값 쌍을 저장하는 해시 테이블 구조로, `{}`로 생성하며 키는 반드시 **고유하고 변경 불가능**해야 함
- **핵심 메서드**로 `keys()`, `values()`, `items()`, `get()`, `update()`, `pop()`을 제공하여 데이터 관리
- **딕셔너리 컴프리헨션**으로 간결한 문법을 통해 새로운 딕셔너리를 효율적으로 생성 가능

### 코드 예시
```python
# 딕셔너리 생성과 활용
student = {
    "name": "김철수",
    "age": 20,
    "scores": {
        "국어": 85,
        "수학": 90,
        "영어": 88
    }
}

# 딕셔너리 데이터 접근
print(student["name"])          # 김철수
print(student["scores"]["수학"]) # 90

# 딕셔너리 데이터 수정
student["age"] = 21
student["scores"]["국어"] = 90

# 주요 메서드 사용
keys = student.keys()     # 키 목록
values = student.values() # 값 목록
items = student.items()   # (키, 값) 튜플 목록

# get 메서드 - 안전한 값 가져오기
name = student.get("name")  # 김철수
address = student.get("address", "주소 없음")  # 기본값 설정
```

---

# 4단계: 함수와 모듈

## 4.1 함수 정의와 호출

### 핵심 개념
- **함수**는 `def` 키워드로 정의하며, 코드의 **재사용성**과 **모듈화**를 위한 핵심 요소
- **매개변수** 설정이 유연하며, 기본값 설정과 `*args`, `**kwargs`를 통한 가변 인자 지원
- **다중 반환값**과 **중첩 함수**를 지원하여 복잡한 함수 로직 구현 가능

### 코드 예시
```python
# 가변인자 사용하기
def calculate_average(*numbers):
    """
    여러 숫자의 평균을 계산하는 함수
    Args:
        *numbers: 가변 인자로 받는 숫자들
    Returns:
        float: 평균값
    """
    return sum(numbers) / len(numbers)

# 함수 호출
avg = calculate_average(10, 20, 30, 40, 50)
print(f"평균: {avg}")  # 평균: 30.0

# 키워드 인자 사용
def print_student_info(name, age, score):
    """
    학생 정보를 출력하는 함수
    Args:
        name (str): 학생 이름
        age (int): 학생 나이
        score (int): 학생 점수
    """
    print(f"이름: {name}, 나이: {age}, 점수: {score}")

# 키워드 인자로 함수 호출
print_student_info(name="김철수", age=20, score=90)
```

## 4.2 모듈 사용

### 핵심 개념
- **모듈**은 `import` 문으로 가져오며, 전체 모듈, 특정 요소, 별칭 사용 등 **다양한 임포트 방식** 지원
- **표준 라이브러리**에서 `math`, `random`, `datetime`, `os`, `sys` 등 **자주 사용되는 모듈** 제공
- **사용자 정의 모듈**은 작업 디렉토리나 `PYTHONPATH`에 위치해야 하며, **패키지**로 체계적 관리 가능

### 코드 예시
```python
# 모듈 사용 방법들
import random
import datetime
from datetime import datetime
import random as rd

# 사용 예시
random_number = random.randint(1, 100)
current_time = datetime.now()
random_number2 = rd.randint(1, 100)

# 사용자 정의 모듈 예시
# calculator.py 파일
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

if __name__ == "__main__":
    print(add(10, 5))  # 15
```

---

# 클린 코드 원칙

## 1. 의미 있는 이름 짓기

### 원칙
- **의미 있는 변수명**은 snake_case로 작성하며, 데이터의 특성을 정확히 반영해야 합니다
- **불리언 변수**는 `is_`, `has_`, `can_` 접두사를 사용하여 의미를 명확히 합니다
- **함수명**은 동사로 시작하며 기능을 직관적으로 표현해야 합니다
- 모호한 이름(`x`, `temp`, `data`)은 사용을 피해야 합니다

### 예시
```python
# ❌ 나쁜 예제
def f(x, l):
    r = []
    for i in range(l):
        if x[i] > 0:
            r.append(x[i])
    return r

# ✅ 좋은 예제
def filter_positive_numbers(numbers: list, length: int) -> list:
    """양수만 필터링하는 함수"""
    positive_numbers = []
    for index in range(length):
        if numbers[index] > 0:
            positive_numbers.append(numbers[index])
    return positive_numbers
```

## 2. 함수 작성 원칙

### 단일 책임 원칙 (Single Responsibility Principle)
- 함수가 하나의 작업만 수행해야 하는 원칙
- 각 함수는 **명확한 목적**을 가지며 다른 기능과 분리되어야 합니다
- **코드 유지보수**와 **재사용성**이 향상됩니다

### 예시
```python
# ❌ 나쁜 예제 - 여러 책임이 혼재
def process_user_data(user_data):
    # 데이터 검증
    if not user_data.get('name') or not user_data.get('email'):
        raise ValueError("Invalid user data")

    # 데이터베이스 저장
    save_to_db(user_data)

    # 이메일 발송
    send_welcome_email(user_data['email'])

# ✅ 좋은 예제 - 책임 분리
def validate_user_data(user_data):
    """사용자 데이터 검증"""
    if not user_data.get('name') or not user_data.get('email'):
        raise ValueError("Invalid user data")

def save_user_data(user_data):
    """사용자 데이터 저장"""
    return save_to_db(user_data)

def send_user_welcome_email(email):
    """환영 이메일 발송"""
    send_welcome_email(email)

def process_user_registration(user_data):
    """사용자 등록 처리 함수 (책임 분리)"""
    validate_user_data(user_data)
    save_user_data(user_data)
    send_user_welcome_email(user_data['email'])
```

## 3. 주석 작성 원칙

### 원칙
- 주석은 `#`(한 줄) 또는 `"""`/`'''`(여러 줄)을 사용하여 코드의 **목적**과 **동작 원리**를 설명
- **독스트링(docstring)**은 함수와 클래스에 필수이며 매개변수, 반환값, 예외를 명확히 기술
- 코드는 **자체 문서화(self-documenting)** 원칙을 따르고, 불필요한 주석은 제거

### 예시
```python
# ❌ 나쁜 예제 - 불필요한 주석
# 사용자의 나이를 계산한다
def calculate_age(birth_year):
    # 현재 연도에서 출생연도를 뺀다
    return 2024 - birth_year

# ✅ 좋은 예제 - 필요한 설명만 포함
def calculate_age(birth_year: int) -> int:
    """
    주어진 출생연도로부터 현재 나이를 계산합니다.

    Args:
        birth_year: 출생연도 (4자리 숫자)

    Returns:
        현재 나이

    Raises:
        ValueError: 출생연도가 올바르지 않은 경우
    """
    if birth_year < 1900 or birth_year > 2024:
        raise ValueError("올바르지 않은 출생연도입니다")
    return 2024 - birth_year
```

## 4. 예외 처리

### 원칙
- **예외 처리**는 `try-except` 구문을 사용해 프로그램 실행 중 발생하는 오류를 체계적으로 관리
- 파이썬은 `TypeError`, `ValueError`, `FileNotFoundError` 등 다양한 **내장 예외 클래스** 제공
- **구체적인 예외 처리**를 통해 문제의 원인을 명확히 파악

### 예시
```python
# ❌ 나쁜 예제 - 모든 예외를 동일하게 처리
def process_file(filename):
    try:
        with open(filename) as f:
            data = f.read()
            process_data(data)
    except Exception as e:
        print(f"Error: {e}")

# ✅ 좋은 예제 - 구체적인 예외 처리
def process_file(filename: str) -> None:
    try:
        with open(filename) as f:
            data = f.read()
            process_data(data)
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {filename}")
        raise
    except PermissionError:
        print(f"파일 접근 권한이 없습니다: {filename}")
        raise
    except Exception as e:
        print(f"예상치 못한 에러 발생: {str(e)}")
        raise
```

## 5. 상수 관리

### 원칙
- 파이썬에서 상수는 모듈 레벨의 **대문자 변수**로 정의
- **매직 넘버** 제거를 위해 상수를 사용하여 코드의 의미를 명확히 함
- 관련 상수는 **Enum 클래스**로 그룹화하여 체계적으로 관리

### 예시
```python
# ❌ 나쁜 예제 - 매직 넘버 사용
def calculate_price(quantity, price):
    if quantity > 100:
        return quantity * price * 0.9
    return quantity * price

# ✅ 좋은 예제 - 상수 정의
BULK_ORDER_THRESHOLD = 100
BULK_ORDER_DISCOUNT = 0.9

def calculate_price(quantity: int, price: float) -> float:
    if quantity > BULK_ORDER_THRESHOLD:
        return quantity * price * BULK_ORDER_DISCOUNT
    return quantity * price

# Enum 사용 예시
from enum import Enum

class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

print(Color.RED.name)   # "RED"
print(Color.RED.value)  # 1
```

## 6. 타입 힌트(Type Hints)

### 원칙
- **타입 힌트**는 코드의 가독성을 높이고 타입 관련 버그를 조기 발견하는 기능
- `typing` 모듈로 `List`, `Dict`, `Optional` 등 **복잡한 타입**을 명시 가능
- Python 3.5부터 도입된 기능으로 코드의 안정성과 가독성을 향상

### 예시
```python
from typing import List, Dict, Optional

def process_user_list(
    users: List[Dict[str, str]],           # 사용자 정보 딕셔너리 리스트
    department: Optional[str] = None       # 선택적 부서명 필터
) -> List[str]:                            # 이메일 주소 리스트 반환
    """
    사용자 목록을 처리하여 이메일 주소 리스트를 반환합니다.

    Args:
        users: 사용자 정보가 담긴 딕셔너리 리스트
        department: 필터링할 부서명 (선택사항)

    Returns:
        이메일 주소 리스트
    """
    if department:
        filtered_users = [
            user for user in users
            if user.get('department') == department
        ]
    else:
        filtered_users = users

    return [user['email'] for user in filtered_users]
```

### VS Code 타입 체킹 설정
1. **필요한 확장 프로그램 설치**:
   - Python extension (Microsoft)
   - Pylance (Microsoft) - Python의 언어 서버로, 타입 체킹 기능 제공

2. **Pylance 설정**:
   - VS Code 설정(Settings)에서 "Python Type Checking Mode" 검색
   - 모드 선택:
     - `off`: 타입 체킹 비활성화
     - `basic`: 기본적인 타입 체킹
     - `strict`: 엄격한 타입 체킹

---

# 실습 문제

## 문제 1: 가장 큰 수 찾기
다음과 같은 함수를 작성하고 테스트 결과를 출력하세요.

**요구사항:**
- 입력: 정수 배열 nums
- 출력: 배열에서 가장 큰 수와 그 위치 (인덱스)
- 제약조건: 배열 길이 1 이상, 중복 가능

### 실습해보기
```python
from typing import List, Tuple

def find_max_number(nums: List[int]) -> Tuple[int, int]:
    """
    배열에서 가장 큰 수와 그 인덱스를 찾는 함수

    Args:
        nums: 정수 배열

    Returns:
        (최대값, 인덱스) 튜플
    """
    # 여기에 코드를 작성하세요
    pass

# 테스트 코드
test_nums = [3, 1, 4, 1, 5, 9, 2, 6]
max_val, max_idx = find_max_number(test_nums)
print(f"최대값: {max_val}, 인덱스: {max_idx}")
```

### 해답
```python
from typing import List, Tuple

def find_max_number(nums: List[int]) -> Tuple[int, int]:
    """
    배열에서 가장 큰 수와 그 인덱스를 찾는 함수

    Args:
        nums: 정수 배열

    Returns:
        (최대값, 인덱스) 튜플
    """
    max_value = nums[0]
    max_index = 0

    for i, num in enumerate(nums):
        if num > max_value:
            max_value = num
            max_index = i

    return max_value, max_index

# 테스트
test_nums = [3, 1, 4, 1, 5, 9, 2, 6]
max_val, max_idx = find_max_number(test_nums)
print(f"최대값: {max_val}, 인덱스: {max_idx}")  # 최대값: 9, 인덱스: 5
```

## 문제 2: 리스트에서 중복 숫자 찾기
다음과 같은 함수를 작성하고 테스트 결과를 출력하세요.

**요구사항:**
- 주어진 정수 리스트에서 중복되는 숫자 찾기
- 결과는 오름차순으로 정렬된 리스트로 반환
- 중복이 없으면 빈 리스트 반환

**제약조건:**
- 입력 리스트 길이: 1 이상 100 이하
- 각 정수 범위: -100 이상 100 이하

### 실습해보기
```python
from typing import List

def find_duplicates(nums: List[int]) -> List[int]:
    """
    리스트에서 중복되는 숫자를 찾아 정렬된 리스트로 반환

    Args:
        nums: 정수 리스트

    Returns:
        중복 숫자들의 정렬된 리스트
    """
    # 여기에 코드를 작성하세요
    pass

# 테스트 코드
test_nums = [1, 2, 3, 2, 4, 3, 5, 1]
duplicates = find_duplicates(test_nums)
print(f"중복 숫자: {duplicates}")
```

### 해답
```python
from typing import List

def find_duplicates(nums: List[int]) -> List[int]:
    """
    리스트에서 중복되는 숫자를 찾아 정렬된 리스트로 반환

    Args:
        nums: 정수 리스트

    Returns:
        중복 숫자들의 정렬된 리스트
    """
    count_dict = {}
    duplicates = []

    # 각 숫자의 개수 세기
    for num in nums:
        count_dict[num] = count_dict.get(num, 0) + 1

    # 중복된 숫자 찾기
    for num, count in count_dict.items():
        if count > 1:
            duplicates.append(num)

    return sorted(duplicates)

# 테스트
test_nums = [1, 2, 3, 2, 4, 3, 5, 1]
duplicates = find_duplicates(test_nums)
print(f"중복 숫자: {duplicates}")  # 중복 숫자: [1, 2, 3]
```

---

# 🎯 핵심 요약

## 클린 코드 체크리스트
- [ ] 변수명과 함수명이 의미가 명확한가?
- [ ] 함수가 하나의 책임만 가지고 있는가?
- [ ] 독스트링과 타입 힌트가 작성되어 있는가?
- [ ] 매직 넘버가 상수로 대체되었는가?
- [ ] 예외 처리가 구체적으로 되어 있는가?
- [ ] 코드가 자체 문서화되어 있는가?

## 실무 적용 팁
1. **일관성**: 프로젝트 전체에서 동일한 네이밍 컨벤션 사용
2. **가독성**: 코드를 읽는 사람을 고려한 명확한 구조
3. **유지보수성**: 나중에 수정하기 쉬운 모듈화된 코드
4. **테스트 용이성**: 각 함수를 독립적으로 테스트할 수 있는 구조

---

# 📚 참고 자료

## 공식 문서
- [PEP 8 - Python 스타일 가이드](https://peps.python.org/pep-0008/)
- [PEP 484 - 타입 힌트](https://peps.python.org/pep-0484/)
- [Python 공식 문서 - typing 모듈](https://docs.python.org/3/library/typing.html)

## 추가 학습 자료
- [Clean Code in Python (실전 파이썬 클린 코드)](https://github.com/zedr/clean-code-python)
- [Python 코딩 컨벤션 가이드](https://wikidocs.net/7896)
- [VS Code Python 개발 환경 설정](https://code.visualstudio.com/docs/python/python-tutorial)

## 도구 및 라이브러리
- **코드 포맷터**: `black`, `autopep8`
- **린터**: `pylint`, `flake8`
- **타입 체커**: `mypy`, `pyright` (Pylance)

---

**📝 다음 학습:** W1_002_OpenAI_Chat_Completion.md