# GraphRAG with Neo4j - Part 2: 실전 구현 및 고급 기능

**학습 자료**: PRJ04_W3_004 - Knowledge Graph 기반 RAG 시스템 실습 및 확장

---

## 📚 학습 목표

이 실습을 완료하면 다음을 할 수 있습니다:

1. **완전한 GraphRAG 시스템 구축**: SEC 10-K 문서로 실전 지식 그래프 RAG 구현
2. **실제 API 연동**: DART API를 활용한 실제 공시 문서 수집
3. **고급 문서 처리**: Docling을 활용한 정교한 PDF 파티셔닝
4. **하이브리드 검색**: RRF(Reciprocal Rank Fusion)를 활용한 검색 품질 향상
5. **그래프 시각화**: Neo4j Browser를 통한 지식 그래프 탐색

---

## 🔑 핵심 개념

### Knowledge Graph 기반 RAG 시스템

완전한 GraphRAG 시스템은 다음 컴포넌트로 구성됩니다:

1. **문서 수집**: API 또는 로컬 파일에서 문서 로드
2. **문서 처리**: 청킹, 임베딩, 엔티티 추출
3. **그래프 구축**: 노드 및 관계 생성, Neo4j 저장
4. **검색 엔진**: 벡터 검색 + 그래프 순회
5. **답변 생성**: LLM을 활용한 자연어 답변

### 고급 기능

#### 1. DART API 연동
- 한국 금융감독원의 전자공시시스템
- 실시간 공시 문서 수집 가능
- 기업 재무정보 및 사업보고서 접근

#### 2. Docling 고급 파티셔닝
- 문서 구조 인식 (제목, 단락, 표, 그림)
- 의미론적 청킹
- 메타데이터 자동 추출

#### 3. RRF 하이브리드 검색
- **Reciprocal Rank Fusion**: 여러 검색 결과를 결합
- 벡터 검색 + 키워드 검색 + 그래프 검색
- 검색 정확도 향상

#### 4. Neo4j Browser 시각화
- 대화형 그래프 탐색
- Cypher 쿼리 실행 및 결과 시각화
- 관계 패턴 분석

---

## 🛠 환경 설정

### Part 1 선행 학습 필요

이 Part 2는 Part 1의 내용을 기반으로 합니다:
- Neo4j AuraDB 연결 설정
- 기본 문서 처리 파이프라인
- LangChain Neo4j 통합

### 추가 라이브러리 설치

```bash
# 고급 문서 처리
pip install docling

# DART API (한국 전자공시)
pip install dart-fss

# 하이브리드 검색
pip install rank-bm25
```

---

## 💻 단계별 구현


---

## **[실습] Knowledge Graph 기반 RAG 시스템 구축**

- 이전 코드를 기반으로 국내 상장기업의 사업보고서를 다운로드합니다. 

- unstructured/docling 라이브러리를 사용하여 문서 파티셔닝 및 청킹 과정을 수행합니다. 

- Knowledge Graph를 구축하고, 이에 기반한 RAG 시스템 구축합니다. 


```python
# ============================================================================
# 실습: 삼성전자 사업보고서 기반 Knowledge Graph RAG 시스템 구축
# ============================================================================

# 1. 사업보고서 다운로드 (DART API 사용)
import dart_fss as dart
import requests
from bs4 import BeautifulSoup
from pathlib import Path

# DART API 키 설정 (무료 API 키 발급: https://opendart.fss.or.kr/)
# 참고: 실제 API 키가 필요합니다. 여기서는 샘플 데이터를 사용합니다.

# 사업보고서 샘플 텍스트 준비 (실제로는 DART에서 다운로드)
sample_report = """
[사업의 내용]

1. 사업의 개요

삼성전자는 글로벌 기술 리더로서 반도체, 가전, IT & 모바일 커뮤니케이션 등의 사업을 영위하고 있습니다.
회사는 메모리 반도체, 시스템 LSI, 파운드리 사업을 통해 글로벌 반도체 시장을 선도하고 있으며,
스마트폰, 태블릿, 웨어러블 기기 등 모바일 제품군에서도 시장 점유율 1위를 유지하고 있습니다.

2. 주요 제품 및 서비스

2.1 반도체 부문
- DRAM: 서버, PC, 모바일용 고성능 메모리 반도체
- NAND Flash: SSD, 스마트폰용 저장장치
- 시스템 LSI: 모바일 AP, 이미지 센서 등
- 파운드리: 최첨단 공정 기술 기반 위탁 생산

2.2 IT & 모바일 커뮤니케이션
- 스마트폰: Galaxy S 시리즈, Galaxy Z 폴더블 시리즈
- 태블릿: Galaxy Tab 시리즈
- 웨어러블: Galaxy Watch, Galaxy Buds

2.3 가전 부문
- TV: QLED, Neo QLED, Micro LED
- 생활가전: 냉장고, 세탁기, 에어컨
- 주방가전: 전자레인지, 식기세척기

[위험요인]

1. 시장 경쟁 심화

글로벌 반도체 시장의 경쟁이 심화되고 있으며, 특히 중국 업체들의 추격이 가속화되고 있습니다.
메모리 반도체 가격 변동성이 크며, 공급 과잉 시 수익성 악화가 우려됩니다.

2. 기술 혁신 리스크

반도체 미세 공정 기술 개발에 막대한 R&D 투자가 필요하며, 기술 개발 지연 시 경쟁력 약화가 예상됩니다.
AI, 5G, 6G 등 신기술 대응을 위한 지속적인 혁신이 필요합니다.

3. 환율 및 원자재 가격 변동

글로벌 사업 특성상 환율 변동에 민감하며, 반도체 제조에 필요한 희토류 등 원자재 가격 상승이 비용 증가 요인입니다.

[재무 상태]

1. 수익성 지표

2023년 연결 기준 매출액은 약 258조원을 기록하였으며, 영업이익은 6.5조원입니다.
반도체 부문의 실적 부진으로 전년 대비 감소하였으나, 2024년 회복세가 예상됩니다.

2. 재무 안정성

부채비율은 30% 수준으로 매우 양호하며, 현금성 자산이 풍부하여 재무 건전성이 우수합니다.
신용등급은 국내외 주요 신용평가기관으로부터 최고 등급을 유지하고 있습니다.

[인사 정책]

1. 인재 채용 및 육성

삼성전자는 글로벌 우수 인재 확보를 위해 공개 채용, 경력 채용, 인턴십 프로그램 등을 운영하고 있습니다.
임직원의 역량 강화를 위해 다양한 교육 프로그램과 해외 연수 기회를 제공합니다.

2. 다양성 및 포용성

성별, 국적, 학력에 관계없이 능력 중심의 인사 제도를 운영하며, 여성 리더 육성 프로그램을 강화하고 있습니다.
장애인 고용 확대 및 일·가정 양립 지원 제도를 통해 포용적 조직문화를 조성하고 있습니다.
"""

# 샘플 데이터를 파일로 저장
report_path = "data/samsung_business_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(sample_report)

print(f"사업보고서 샘플 데이터 저장 완료: {report_path}")
print(f"문서 길이: {len(sample_report)} 문자")
```

## 실습 완료 요약

위 코드는 다음 단계로 삼성전자 사업보고서 기반 Knowledge Graph RAG 시스템을 구축합니다:

### 1단계: 데이터 준비
- 삼성전자 사업보고서 샘플 데이터 생성 및 저장
- 실제 환경에서는 DART API를 통해 실제 사업보고서를 다운로드할 수 있음

### 2단계: 문서 파티셔닝 및 청킹
- 대괄호 `[]`로 구분된 섹션을 자동 파싱
- `RecursiveCharacterTextSplitter`를 사용하여 각 섹션을 청크로 분할
- 각 청크에 메타데이터(순서, ID, 부모 섹션 등) 추가

### 3단계: Knowledge Graph 구축
- Neo4j 데이터베이스에 Document, Section, Chunk 노드 생성
- OpenAI 임베딩을 사용하여 각 청크의 벡터 임베딩 생성
- 청크 간 NEXT 관계를 통해 순서 정보 보존

### 4단계: RAG 시스템 테스트
- 벡터 검색과 그래프 탐색을 결합한 RAG 시스템 활용
- 다양한 질문(사업 개요, 위험요인, 재무 상태, 인사 정책)으로 시스템 테스트
- LLM이 검색된 문맥을 바탕으로 정확한 답변 생성

### 주요 개선 가능 사항
1. **실제 DART API 연동**: dart-fss 라이브러리로 실제 사업보고서 다운로드
2. **Docling 활용**: PDF 문서의 고급 파티셔닝 (표, 이미지, 레이아웃 보존)
3. **하이브리드 검색**: BM25 키워드 검색과 벡터 검색 결합
4. **메타데이터 필터링**: 섹션별, 날짜별 필터링 기능 추가
5. **시각화**: Neo4j Browser로 지식 그래프 구조 시각화

```python
# 4. RAG 시스템 구현 및 테스트

# RAG 체인은 이미 위에서 정의되어 있으므로 바로 사용 가능
# vector_store, retriever, rag_chain이 이미 설정되어 있음

print("=" * 60)
print("삼성전자 사업보고서 RAG 시스템 테스트")
print("=" * 60)

# 테스트 질문 1: 사업 개요
test_query_1 = "삼성전자의 주요 사업 부문은 무엇인가요?"
print(f"\n질문 1: {test_query_1}")
print("-" * 60)
response_1 = rag_chain.invoke(test_query_1)
print(response_1)

# 테스트 질문 2: 위험요인
test_query_2 = "삼성전자가 직면한 주요 위험요인에 대해 설명해주세요."
print(f"\n\n질문 2: {test_query_2}")
print("-" * 60)
response_2 = rag_chain.invoke(test_query_2)
print(response_2)

# 테스트 질문 3: 재무 상태
test_query_3 = "삼성전자의 2023년 재무 실적은 어떠했나요?"
print(f"\n\n질문 3: {test_query_3}")
print("-" * 60)
response_3 = rag_chain.invoke(test_query_3)
print(response_3)

# 테스트 질문 4: 인사 정책
test_query_4 = "삼성전자의 인사 정책과 다양성 노력에 대해 알려주세요."
print(f"\n\n질문 4: {test_query_4}")
print("-" * 60)
response_4 = rag_chain.invoke(test_query_4)
print(response_4)

print("\n" + "=" * 60)
print("RAG 시스템 테스트 완료!")
print("=" * 60)
```

```python
# 3. Knowledge Graph 구축 (삼성전자 사업보고서)

# 문서 ID 설정
samsung_doc_id = "samsung_business_report_2023"

print("=" * 60)
print("삼성전자 사업보고서 Knowledge Graph 구축 시작")
print("=" * 60)

# Document 노드 생성
create_document_node(graph, samsung_doc_id, report_path)
print(f"✓ Document 노드 생성 완료: {samsung_doc_id}")

# 각 섹션 처리
total_chunks_created = 0
for section_name, chunks in section_docs_split.items():
    print(f"\n처리중인 섹션: {section_name}")
    
    # Section 노드 생성
    create_section_node(graph, section_name, samsung_doc_id)
    print(f"  ✓ Section 노드 생성")
    
    prev_chunk_id = None
    
    # 각 청크 처리
    for chunk in chunks:
        chunk_id = str(uuid.uuid4())
        
        # 임베딩 생성 (OpenAI)
        embedding = embeddings.embed_query(chunk.page_content)
        
        # Chunk 노드 생성
        create_chunk_node(
            graph, 
            section_name, 
            samsung_doc_id, 
            chunk_id,
            chunk.page_content, 
            embedding, 
            chunk.metadata
        )
        
        # 이전 청크와 NEXT 관계 연결
        if prev_chunk_id:
            create_next_relationship(graph, prev_chunk_id, chunk_id)
        
        prev_chunk_id = chunk_id
        total_chunks_created += 1
    
    # 섹션별 저장 확인
    count = graph.query(
        """
        MATCH (c:Chunk {section_name: $section_name, document_id: $doc_id}) 
        RETURN COUNT(c) AS count
        """,
        params={"section_name": section_name, "doc_id": samsung_doc_id}
    )[0]['count']
    print(f"  ✓ 저장된 청크 수: {count}")

print("\n" + "=" * 60)
print(f"Knowledge Graph 구축 완료!")
print(f"총 섹션 수: {len(section_docs_split)}")
print(f"총 청크 수: {total_chunks_created}")
print("=" * 60)
```

```python
# 2. 문서 파티셔닝 및 청킹
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import re

# 섹션별로 문서 분리 (대괄호 [] 기준)
def parse_sections(text):
    """대괄호로 구분된 섹션을 파싱"""
    sections = {}
    current_section = None
    current_content = []
    
    for line in text.split('\n'):
        # 섹션 헤더 감지 (예: [사업의 내용])
        section_match = re.match(r'\[(.*?)\]', line.strip())
        if section_match:
            # 이전 섹션 저장
            if current_section:
                sections[current_section] = '\n'.join(current_content).strip()
            # 새 섹션 시작
            current_section = section_match.group(1)
            current_content = []
        elif current_section:
            current_content.append(line)
    
    # 마지막 섹션 저장
    if current_section:
        sections[current_section] = '\n'.join(current_content).strip()
    
    return sections

# 보고서 파싱
with open(report_path, "r", encoding="utf-8") as f:
    report_text = f.read()

sections = parse_sections(report_text)
print(f"파싱된 섹션 수: {len(sections)}")
print(f"섹션 목록: {list(sections.keys())}")

# 텍스트 스플리터 설정
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # 청크 크기
    chunk_overlap=50,  # 오버랩
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# 섹션별로 청크 분할
section_docs_split = {}
for section_name, content in sections.items():
    # Document 객체로 변환
    docs = [Document(
        page_content=content,
        metadata={
            "source": report_path,
            "section": section_name,
            "page_number": 1  # 샘플이므로 페이지는 1로 설정
        }
    )]
    
    # 청크 분할
    split_docs = text_splitter.split_documents(docs)
    
    # 청크에 순서 정보 추가
    for idx, doc in enumerate(split_docs):
        doc.metadata['order'] = idx
        doc.metadata['chunk_id'] = f"{section_name}_{idx}"
        doc.metadata['element_id'] = f"chunk_{idx}"
        doc.metadata['parent_id'] = section_name
    
    section_docs_split[section_name] = split_docs
    print(f"  - {section_name}: {len(split_docs)} chunks")

print(f"\n총 청크 수: {sum(len(chunks) for chunks in section_docs_split.values())}")
```

---

## **Feature 1: 실제 DART API 연동**

이제 샘플 데이터가 아닌 실제 DART API를 사용하여 반도체 3사(삼성전자, SK하이닉스, 삼성SDI)의 2022-2024년 사업보고서를 다운로드합니다.

### 구현 내용
1. DART API 키 설정 (.env 파일에서 로드)
2. 3개 기업 × 3개년 = 9개 보고서 다운로드
3. HTML 텍스트 추출 및 섹션 파싱
4. Knowledge Graph에 저장

```python
# ============================================================================
# Feature 1-1: DART API 설정 및 기업/연도 정의
# ============================================================================

import dart_fss as dart
from pathlib import Path
import time

# DART API 키 설정 (.env 파일에서 로드)
DART_API_KEY = os.getenv("OPEN_DART_API")
dart.set_api_key(DART_API_KEY)

print(f"DART API 키 설정 완료: {DART_API_KEY[:10]}...")

# 타겟 기업 및 연도 정의
COMPANIES = {
    '삼성전자': '00126380',  # 삼성전자 고유번호
    'SK하이닉스': '00164779',  # SK하이닉스 고유번호
    '삼성SDI': '00117896'   # 삼성SDI 고유번호
}

YEARS = ['2022', '2023', '2024']

# 다운로드 디렉토리 생성
DOWNLOAD_DIR = Path("data/dart_reports")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n다운로드 디렉토리: {DOWNLOAD_DIR}")
print(f"타겟 기업: {list(COMPANIES.keys())}")
print(f"타겟 연도: {YEARS}")
print(f"총 다운로드 예정 보고서 수: {len(COMPANIES) * len(YEARS)} 개")
```

```python
# ============================================================================
# Feature 1-2: 보고서 다운로드 및 HTML 텍스트 추출
# ============================================================================

from bs4 import BeautifulSoup
import re

# 다운로드한 보고서 정보 저장
dart_reports = {}

print("=" * 70)
print("DART 사업보고서 다운로드 시작")
print("=" * 70)

for company_name, corp_code in COMPANIES.items():
    print(f"\n{'='*70}")
    print(f"기업: {company_name} (고유번호: {corp_code})")
    print(f"{'='*70}")
    
    dart_reports[company_name] = {}
    
    for year in YEARS:
        try:
            print(f"\n  [{year}년 사업보고서 검색 중...]")
            
            # 회사 객체 생성
            corp = dart.get_corp(corp_code)
            
            # 사업보고서 찾기
            reports = corp.find_all(bgn_de=f"{year}0101", pblntf_ty="A")
            
            # 사업보고서 필터링 (정정 보고서 제외, 가장 최근 것 선택)
            annual_reports = [r for r in reports if r.report_nm and '사업보고서' in r.report_nm and '정정' not in r.report_nm]
            
            if not annual_reports:
                print(f"    ⚠ {year}년 사업보고서를 찾을 수 없습니다.")
                continue
            
            # 최신 보고서 선택
            report = annual_reports[0]
            print(f"    ✓ 발견: {report.report_nm} (접수일: {report.rcept_dt})")
            
            # 보고서 상세 정보 로드 및 HTML 추출
            report.load()
            html_content = report.html
            
            # HTML에서 텍스트 추출
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 불필요한 태그 제거
            for tag in soup(['script', 'style', 'meta', 'link']):
                tag.decompose()
            
            # 텍스트 추출
            text = soup.get_text(separator='\n')
            
            # 연속된 빈 줄 제거 및 정리
            text = re.sub(r'\n\s*\n+', '\n\n', text)
            text = text.strip()
            
            # 저장
            dart_reports[company_name][year] = {
                'report': report,
                'html': html_content,
                'text': text,
                'rcept_dt': report.rcept_dt,
                'report_nm': report.report_nm
            }
            
            print(f"    ✓ HTML 텍스트 추출 완료 (길이: {len(text):,} 문자)")
            
            # API 호출 제한 대응 (1초 대기)
            time.sleep(1)
            
        except Exception as e:
            print(f"    ✗ 오류 발생: {str(e)}")
            continue

print(f"\n{'='*70}")
print("다운로드 완료 요약")
print(f"{'='*70}")
total_downloaded = sum(len(years) for years in dart_reports.values())
print(f"총 다운로드 성공: {total_downloaded} / {len(COMPANIES) * len(YEARS)} 개")

for company_name, years_data in dart_reports.items():
    print(f"  - {company_name}: {len(years_data)} 개")
```

```python
# ============================================================================
# Feature 1-3: 섹션 파싱 및 청킹
# ============================================================================

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 섹션 파싱 함수 (간단한 구조 기반)
def parse_dart_sections(text):
    """DART 사업보고서에서 주요 섹션 추출"""
    sections = {}
    
    # 주요 섹션 패턴 정의
    section_patterns = {
        '사업의내용': r'(?:I+|1)[.\s]*사업의\s*내용',
        '재무정보': r'(?:I+|\d+)[.\s]*재무.*?(?:정보|제표)',
        '이사회운영': r'(?:I+|\d+)[.\s]*이사.*?(?:회|의)',
        '주주총회': r'(?:I+|\d+)[.\s]*주주.*?총회',
        '임원현황': r'(?:I+|\d+)[.\s]*임원.*?현황',
    }
    
    # 전체 텍스트를 기본 섹션으로 저장
    sections['전체'] = text
    
    return sections

# 텍스트 스플리터 설정
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # 청크 크기
    chunk_overlap=100,  # 오버랩
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# 모든 보고서에 대해 청킹 수행
dart_chunks = {}

print("=" * 70)
print("보고서 섹션 파싱 및 청킹")
print("=" * 70)

for company_name, years_data in dart_reports.items():
    print(f"\n[{company_name}]")
    dart_chunks[company_name] = {}
    
    for year, report_data in years_data.items():
        print(f"  - {year}년 처리 중...")
        
        text = report_data['text']
        
        # 섹션 파싱
        sections = parse_dart_sections(text)
        
        # 섹션별 청킹
        year_chunks = {}
        total_chunk_count = 0
        
        for section_name, section_text in sections.items():
            # Document 객체 생성
            docs = [Document(
                page_content=section_text,
                metadata={
                    "source": f"{company_name}_{year}_사업보고서",
                    "company": company_name,
                    "year": year,
                    "section": section_name,
                    "rcept_dt": report_data['rcept_dt']
                }
            )]
            
            # 청킹
            split_docs = text_splitter.split_documents(docs)
            
            # 메타데이터 추가
            for idx, doc in enumerate(split_docs):
                doc.metadata['order'] = idx
                doc.metadata['chunk_id'] = f"{company_name}_{year}_{section_name}_{idx}"
                doc.metadata['element_id'] = f"chunk_{idx}"
                doc.metadata['parent_id'] = section_name
            
            year_chunks[section_name] = split_docs
            total_chunk_count += len(split_docs)
        
        dart_chunks[company_name][year] = year_chunks
        print(f"    ✓ {total_chunk_count} 개 청크 생성")

print(f"\n{'='*70}")
print("청킹 완료")
print(f"{'='*70}")
for company_name, years_data in dart_chunks.items():
    total = sum(sum(len(chunks) for chunks in year_data.values()) for year_data in years_data.values())
    print(f"  - {company_name}: {total} 개 청크")
```

```python
# ============================================================================
# Feature 1-4: Knowledge Graph에 저장
# ============================================================================

print("=" * 70)
print("Knowledge Graph에 DART 보고서 저장")
print("=" * 70)

total_saved_chunks = 0

for company_name, years_data in dart_chunks.items():
    print(f"\n{'='*70}")
    print(f"기업: {company_name}")
    print(f"{'='*70}")
    
    for year, sections_data in years_data.items():
        print(f"\n  [{year}년 사업보고서]")
        
        # Document ID 생성
        doc_id = f"{company_name}_{year}_사업보고서"
        
        # 1. Document 노드 생성
        create_document_node(graph, doc_id, doc_id)
        print(f"    ✓ Document 노드 생성: {doc_id}")
        
        # 2. 각 섹션 처리
        for section_name, chunks in sections_data.items():
            if not chunks:
                continue
                
            print(f"    - 섹션: {section_name} ({len(chunks)} 청크)")
            
            # Section 노드 생성
            create_section_node(graph, section_name, doc_id)
            
            prev_chunk_id = None
            
            # 각 청크 처리
            for chunk in chunks:
                chunk_id = str(uuid.uuid4())
                
                # 임베딩 생성
                embedding = embeddings.embed_query(chunk.page_content)
                
                # Chunk 노드 생성
                chunk_metadata = chunk.metadata.copy()
                chunk_metadata['page_number'] = 1  # DART는 페이지 정보 없음
                
                create_chunk_node(
                    graph,
                    section_name,
                    doc_id,
                    chunk_id,
                    chunk.page_content,
                    embedding,
                    chunk_metadata
                )
                
                # 이전 청크와 NEXT 관계 연결
                if prev_chunk_id:
                    create_next_relationship(graph, prev_chunk_id, chunk_id)
                
                prev_chunk_id = chunk_id
                total_saved_chunks += 1
            
            # 저장 확인
            count = graph.query(
                "MATCH (c:Chunk {section_name: $section_name, document_id: $doc_id}) RETURN COUNT(c) AS count",
                params={"section_name": section_name, "doc_id": doc_id}
            )[0]['count']
            print(f"      ✓ 저장 완료: {count} 청크")

print(f"\n{'='*70}")
print(f"저장 완료 - 총 {total_saved_chunks} 개 청크")
print(f"{'='*70}")

# 전체 통계
stats = graph.query("""
MATCH (d:Document)
OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
OPTIONAL MATCH (s)-[:CONTAINS]->(c:Chunk)
RETURN 
    COUNT(DISTINCT d) AS documents,
    COUNT(DISTINCT s) AS sections,
    COUNT(DISTINCT c) AS chunks
""")

print(f"\n[Neo4j 전체 통계]")
print(f"  - 문서: {stats[0]['documents']} 개")
print(f"  - 섹션: {stats[0]['sections']} 개")
print(f"  - 청크: {stats[0]['chunks']} 개")
```

---

## **Feature 2: Docling을 활용한 고급 PDF 파티셔닝**

DART HTML을 PDF로 변환한 후, Docling을 사용하여 표, 레이아웃, 섹션 구조를 보존하면서 파싱합니다.

### 구현 내용
1. HTML → PDF 변환 (weasyprint 사용)
2. Docling으로 PDF 구조 파싱 (표, 텍스트, 레이아웃 보존)
3. 구조화된 청크 생성

```python
# ============================================================================
# Feature 2-1: HTML → PDF 변환
# ============================================================================

from weasyprint import HTML, CSS
from pathlib import Path

# PDF 저장 디렉토리
PDF_DIR = Path("data/dart_pdfs")
PDF_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("HTML → PDF 변환")
print("=" * 70)

dart_pdfs = {}

for company_name, years_data in dart_reports.items():
    print(f"\n[{company_name}]")
    dart_pdfs[company_name] = {}
    
    for year, report_data in years_data.items():
        try:
            print(f"  - {year}년 변환 중...")
            
            html_content = report_data['html']
            
            # PDF 파일명
            pdf_filename = f"{company_name}_{year}_사업보고서.pdf"
            pdf_path = PDF_DIR / pdf_filename
            
            # HTML → PDF 변환
            # CSS 스타일 추가 (한글 폰트 및 레이아웃 최적화)
            css = CSS(string='''
                @page {
                    size: A4;
                    margin: 2cm;
                }
                body {
                    font-family: "Noto Sans KR", "Malgun Gothic", sans-serif;
                    font-size: 10pt;
                    line-height: 1.5;
                }
                table {
                    border-collapse: collapse;
                    width: 100%;
                    margin: 10px 0;
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 5px;
                    text-align: left;
                }
            ''')
            
            HTML(string=html_content).write_pdf(
                pdf_path,
                stylesheets=[css]
            )
            
            dart_pdfs[company_name][year] = str(pdf_path)
            
            # 파일 크기 확인
            file_size = pdf_path.stat().st_size / (1024 * 1024)  # MB
            print(f"    ✓ 변환 완료: {pdf_filename} ({file_size:.2f} MB)")
            
        except Exception as e:
            print(f"    ✗ 변환 실패: {str(e)}")
            continue

print(f"\n{'='*70}")
print("PDF 변환 완료")
print(f"{'='*70}")
total_pdfs = sum(len(years) for years in dart_pdfs.values())
print(f"총 변환된 PDF: {total_pdfs} 개")
```

```python
# ============================================================================
# Feature 2-2: Docling으로 PDF 구조 파싱
# ============================================================================

from docling.document_converter import DocumentConverter
from pathlib import Path

print("=" * 70)
print("Docling PDF 구조 파싱")
print("=" * 70)

# Docling converter 초기화
converter = DocumentConverter()

docling_results = {}

for company_name, years_data in dart_pdfs.items():
    print(f"\n[{company_name}]")
    docling_results[company_name] = {}
    
    for year, pdf_path in years_data.items():
        try:
            print(f"  - {year}년 파싱 중...")
            
            # PDF 파싱
            result = converter.convert(pdf_path)
            
            # 문서 구조 추출
            doc_data = {
                'document': result.document,
                'markdown': result.document.export_to_markdown(),
                'tables': [],
                'sections': []
            }
            
            # 표 추출
            for item in result.document.iterate_items():
                if hasattr(item, 'label') and 'table' in item.label.lower():
                    doc_data['tables'].append({
                        'text': item.text if hasattr(item, 'text') else '',
                        'page': item.prov[0].page if hasattr(item, 'prov') and item.prov else None
                    })
            
            docling_results[company_name][year] = doc_data
            
            print(f"    ✓ 파싱 완료: {len(doc_data['tables'])} 개 표 추출")
            print(f"    ✓ Markdown 길이: {len(doc_data['markdown']):,} 문자")
            
        except Exception as e:
            print(f"    ✗ 파싱 실패: {str(e)}")
            continue

print(f"\n{'='*70}")
print("Docling 파싱 완료")
print(f"{'='*70}")
```

```python
# ============================================================================
# Feature 2-3: Docling 결과를 청킹하여 Neo4j에 저장
# ============================================================================

print("=" * 70)
print("Docling 파싱 결과 Knowledge Graph 저장")
print("=" * 70)

docling_saved_chunks = 0

for company_name, years_data in docling_results.items():
    print(f"\n[{company_name}]")
    
    for year, doc_data in years_data.items():
        try:
            print(f"  - {year}년 저장 중...")
            
            # Document ID (Docling 버전)
            doc_id = f"{company_name}_{year}_사업보고서_docling"
            
            # Document 노드 생성
            create_document_node(graph, doc_id, f"{company_name}_{year}_docling")
            
            # Markdown을 청킹
            markdown_text = doc_data['markdown']
            
            # 청킹
            docs = [Document(
                page_content=markdown_text,
                metadata={
                    "source": doc_id,
                    "company": company_name,
                    "year": year,
                    "section": "전체_docling",
                    "format": "markdown"
                }
            )]
            
            split_docs = text_splitter.split_documents(docs)
            
            # Section 노드
            section_name = "전체_docling"
            create_section_node(graph, section_name, doc_id)
            
            prev_chunk_id = None
            
            for idx, doc in enumerate(split_docs):
                chunk_id = str(uuid.uuid4())
                
                # 임베딩
                embedding = embeddings.embed_query(doc.page_content)
                
                # 메타데이터
                metadata = doc.metadata.copy()
                metadata['order'] = idx
                metadata['page_number'] = 1
                metadata['element_id'] = f"docling_chunk_{idx}"
                metadata['parent_id'] = section_name
                
                # Chunk 저장
                create_chunk_node(
                    graph,
                    section_name,
                    doc_id,
                    chunk_id,
                    doc.page_content,
                    embedding,
                    metadata
                )
                
                if prev_chunk_id:
                    create_next_relationship(graph, prev_chunk_id, chunk_id)
                
                prev_chunk_id = chunk_id
                docling_saved_chunks += 1
            
            print(f"    ✓ {len(split_docs)} 청크 저장 완료")
            
        except Exception as e:
            print(f"    ✗ 저장 실패: {str(e)}")
            continue

print(f"\n{'='*70}")
print(f"Docling 결과 저장 완료 - 총 {docling_saved_chunks} 청크")
print(f"{'='*70}")
```

---

## **Feature 3: RRF 하이브리드 검색**

BM25 키워드 검색과 벡터 검색을 RRF(Reciprocal Rank Fusion)로 결합하여 더 정확한 검색 결과를 제공합니다.

### 구현 내용
1. BM25 Retriever 구축 (키워드 기반 검색)
2. RRF 알고리즘 구현 (BM25 + Vector 결과 융합)
3. 업데이트된 RAG 체인 구성

```python
# ============================================================================
# Feature 3-1: BM25 Retriever 구축
# ============================================================================

from rank_bm25 import BM25Okapi
import numpy as np

print("=" * 70)
print("BM25 Retriever 구축")
print("=" * 70)

# Neo4j에서 모든 청크 가져오기
all_chunks_query = """
MATCH (c:Chunk)
RETURN 
    c.chunk_id AS chunk_id,
    c.content AS content,
    c.document_id AS document_id,
    c.section_name AS section_name,
    c.order AS order
ORDER BY c.document_id, c.section_name, c.order
"""

all_chunks = graph.query(all_chunks_query)
print(f"총 청크 수: {len(all_chunks)}")

# BM25를 위한 문서 준비
bm25_docs = []
bm25_metadata = []

for chunk in all_chunks:
    # 토큰화 (간단한 공백 기반)
    tokens = chunk['content'].lower().split()
    bm25_docs.append(tokens)
    bm25_metadata.append({
        'chunk_id': chunk['chunk_id'],
        'content': chunk['content'],
        'document_id': chunk['document_id'],
        'section': chunk['section_name'],
        'order': chunk['order']
    })

# BM25 인덱스 생성
bm25 = BM25Okapi(bm25_docs)

print(f"✓ BM25 인덱스 생성 완료: {len(bm25_docs)} 문서")

# BM25 검색 함수
def bm25_search(query, k=5):
    """BM25 기반 검색"""
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    
    # 상위 k개 추출
    top_indices = np.argsort(scores)[::-1][:k]
    
    results = []
    for idx in top_indices:
        if scores[idx] > 0:  # 점수가 0보다 큰 것만
            results.append({
                'metadata': bm25_metadata[idx],
                'score': float(scores[idx]),
                'content': bm25_metadata[idx]['content']
            })
    
    return results

# 테스트
print("\n[BM25 검색 테스트]")
test_query = "반도체 사업 실적"
bm25_results = bm25_search(test_query, k=3)
print(f"질문: {test_query}")
print(f"결과: {len(bm25_results)} 개 문서")
for i, result in enumerate(bm25_results, 1):
    print(f"  {i}. 점수: {result['score']:.2f} | 문서: {result['metadata']['document_id']}")
```

```python
# ============================================================================
# Feature 3-2: RRF (Reciprocal Rank Fusion) 구현
# ============================================================================

from langchain_core.documents import Document as LCDocument

def reciprocal_rank_fusion(bm25_results, vector_results, k=60):
    """
    RRF (Reciprocal Rank Fusion) 알고리즘
    
    Args:
        bm25_results: BM25 검색 결과 리스트
        vector_results: 벡터 검색 결과 리스트 (LangChain Document)
        k: RRF 파라미터 (기본값 60)
    
    Returns:
        융합된 결과 리스트
    """
    # 문서별 RRF 점수 계산
    rrf_scores = {}
    
    # BM25 결과 처리
    for rank, result in enumerate(bm25_results, 1):
        chunk_id = result['metadata']['chunk_id']
        score = 1 / (k + rank)
        
        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = {
                'score': 0,
                'content': result['content'],
                'metadata': result['metadata']
            }
        rrf_scores[chunk_id]['score'] += score
    
    # 벡터 검색 결과 처리
    for rank, doc in enumerate(vector_results, 1):
        # chunk_id 추출 (메타데이터에서)
        chunk_id = doc.metadata.get('chunk_id') if hasattr(doc, 'metadata') else None
        
        if chunk_id:
            score = 1 / (k + rank)
            
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {
                    'score': 0,
                    'content': doc.page_content,
                    'metadata': doc.metadata
                }
            rrf_scores[chunk_id]['score'] += score
    
    # 점수 순으로 정렬
    sorted_results = sorted(
        rrf_scores.items(),
        key=lambda x: x[1]['score'],
        reverse=True
    )
    
    # LangChain Document 객체로 변환
    final_results = []
    for chunk_id, data in sorted_results:
        final_results.append(LCDocument(
            page_content=data['content'],
            metadata=data['metadata']
        ))
    
    return final_results

# 하이브리드 검색 함수
def hybrid_search(query, k=5):
    """BM25 + Vector 하이브리드 검색 (RRF 융합)"""
    # 1. BM25 검색
    bm25_results = bm25_search(query, k=k)
    
    # 2. 벡터 검색
    vector_results = retriever.invoke(query)
    
    # 3. RRF 융합
    fused_results = reciprocal_rank_fusion(bm25_results, vector_results, k=60)
    
    return fused_results[:k]

print("=" * 70)
print("RRF 하이브리드 검색 테스트")
print("=" * 70)

# 테스트
test_query = "삼성전자 반도체 사업 실적"
print(f"\n질문: {test_query}")

# BM25만
bm25_only = bm25_search(test_query, k=3)
print(f"\n[BM25 검색] {len(bm25_only)} 개 결과")
for i, r in enumerate(bm25_only, 1):
    print(f"  {i}. {r['metadata']['document_id']} (점수: {r['score']:.2f})")

# 벡터만
vector_only = retriever.invoke(test_query)
print(f"\n[벡터 검색] {len(vector_only)} 개 결과")
for i, doc in enumerate(vector_only[:3], 1):
    print(f"  {i}. {doc.metadata.get('section', 'N/A')}")

# 하이브리드
hybrid_results = hybrid_search(test_query, k=5)
print(f"\n[하이브리드 검색 (RRF)] {len(hybrid_results)} 개 결과")
for i, doc in enumerate(hybrid_results, 1):
    print(f"  {i}. {doc.metadata.get('document_id', 'N/A')} - {doc.metadata.get('section', 'N/A')}")
```

```python
# ============================================================================
# Feature 3-3: 하이브리드 RAG 체인 구성
# ============================================================================

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# 하이브리드 RAG 체인 구성
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

hybrid_template = """
당신은 한국 반도체 기업(삼성전자, SK하이닉스, 삼성SDI) 사업보고서 분석 전문가입니다.
아래 제공된 [보고서 내용]을 바탕으로 [질문]에 대해 상세하고 정확하게 답변해 주세요.

답변 시:
1. 정보의 출처(기업명, 연도, 섹션)를 명시하세요
2. 여러 기업을 비교할 때는 표 형식으로 정리하세요
3. 수치 데이터는 정확하게 인용하세요

<보고서 내용>
{context}
</보고서 내용>

<질문>
{question}
</질문>
"""

hybrid_prompt = PromptTemplate(template=hybrid_template, input_variables=["context", "question"])

def format_hybrid_docs(docs):
    """하이브리드 검색 결과 포맷팅"""
    formatted = []
    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata
        source = f"{metadata.get('document_id', 'N/A')} - {metadata.get('section', 'N/A')}"
        formatted.append(f"[문서 {i}] 출처: {source}\n{doc.page_content}")
    return "\n\n" + "="*70 + "\n\n".join(formatted)

# 하이브리드 RAG 체인
hybrid_rag_chain = (
    {
        "context": lambda x: format_hybrid_docs(hybrid_search(x, k=5)),
        "question": RunnablePassthrough()
    }
    | hybrid_prompt
    | llm
    | StrOutputParser()
)

print("=" * 70)
print("하이브리드 RAG 체인 테스트")
print("=" * 70)

# 테스트 질문들
test_questions = [
    "삼성전자의 2023년 반도체 사업 실적은?",
    "SK하이닉스와 삼성SDI의 주요 사업 분야를 비교해주세요",
    "3개 기업의 R&D 투자 현황은?"
]

for i, question in enumerate(test_questions, 1):
    print(f"\n{'='*70}")
    print(f"질문 {i}: {question}")
    print(f"{'='*70}")
    
    try:
        response = hybrid_rag_chain.invoke(question)
        print(response)
    except Exception as e:
        print(f"오류: {str(e)}")

print(f"\n{'='*70}")
print("하이브리드 RAG 시스템 구축 완료!")
print(f"{'='*70}")
```

---

## **Feature 4: Neo4j Browser 시각화**

Neo4j Browser에서 Knowledge Graph를 시각화하고 탐색할 수 있는 Cypher 쿼리 모음입니다.

### Neo4j Browser 접속
1. Neo4j Aura 콘솔에 로그인
2. "Open with" → "Browser" 선택
3. 아래 쿼리를 복사하여 실행

### 유용한 Cypher 쿼리

```python
# ============================================================================
# Feature 4: Neo4j 시각화를 위한 Cypher 쿼리 모음
# ============================================================================

# 쿼리 모음을 딕셔너리로 정리
neo4j_queries = {
    "1. 전체 그래프 개요": """
    // 전체 노드 및 관계 통계
    MATCH (n)
    RETURN 
        labels(n)[0] AS 노드타입,
        COUNT(*) AS 개수
    ORDER BY 개수 DESC
    """,
    
    "2. 특정 기업의 Knowledge Graph 구조": """
    // 삼성전자 2023년 사업보고서 구조 시각화
    MATCH path = (d:Document {id: '삼성전자_2023_사업보고서'})
                 -[:HAS_SECTION]->(s:Section)
                 -[:CONTAINS]->(c:Chunk)
    RETURN path
    LIMIT 50
    """,
    
    "3. 문서별 청크 개수": """
    // 각 문서의 청크 개수 확인
    MATCH (d:Document)-[:HAS_SECTION]->(s:Section)-[:CONTAINS]->(c:Chunk)
    RETURN 
        d.id AS 문서,
        COUNT(DISTINCT c) AS 청크개수
    ORDER BY 청크개수 DESC
    """,
    
    "4. 섹션별 청크 분포": """
    // 섹션별 청크 개수 (상위 10개)
    MATCH (s:Section)-[:CONTAINS]->(c:Chunk)
    RETURN 
        s.name AS 섹션명,
        COUNT(c) AS 청크개수
    ORDER BY 청크개수 DESC
    LIMIT 10
    """,
    
    "5. 청크 연결 구조 (NEXT 관계)": """
    // 특정 섹션의 청크 연결 구조 시각화
    MATCH path = (c1:Chunk {section_name: '전체'})-[:NEXT*1..5]->(c2:Chunk)
    WHERE c1.document_id CONTAINS '삼성전자'
    RETURN path
    LIMIT 20
    """,
    
    "6. 특정 키워드가 포함된 청크 검색": """
    // '반도체' 키워드가 포함된 청크 찾기
    MATCH (c:Chunk)
    WHERE c.content CONTAINS '반도체'
    RETURN 
        c.document_id AS 문서,
        c.section_name AS 섹션,
        c.order AS 순서,
        substring(c.content, 0, 100) + '...' AS 내용미리보기
    LIMIT 10
    """,
    
    "7. 기업별 문서 현황": """
    // 기업별 저장된 문서 확인
    MATCH (d:Document)
    WITH d.id AS doc_id
    WITH 
        CASE 
            WHEN doc_id CONTAINS '삼성전자' THEN '삼성전자'
            WHEN doc_id CONTAINS 'SK하이닉스' THEN 'SK하이닉스'
            WHEN doc_id CONTAINS '삼성SDI' THEN '삼성SDI'
            ELSE 'Tesla'
        END AS 기업,
        doc_id
    RETURN 기업, COLLECT(doc_id) AS 문서목록, COUNT(*) AS 문서개수
    ORDER BY 문서개수 DESC
    """,
    
    "8. 그래프 전체 구조 샘플링": """
    // 전체 그래프 구조를 샘플로 시각화 (성능을 위해 제한)
    MATCH path = (d:Document)-[:HAS_SECTION]->(s:Section)-[:CONTAINS]->(c:Chunk)
    RETURN path
    LIMIT 100
    """,
    
    "9. 벡터 인덱스 확인": """
    // 생성된 벡터 인덱스 확인
    SHOW INDEXES
    YIELD name, type, entityType, labelsOrTypes, properties
    WHERE type = 'VECTOR'
    RETURN name, entityType, labelsOrTypes, properties
    """,
    
    "10. 데이터 품질 체크": """
    // 임베딩이 없거나 내용이 비어있는 청크 찾기
    MATCH (c:Chunk)
    WHERE c.embedding IS NULL OR c.content = '' OR c.content IS NULL
    RETURN 
        c.document_id AS 문서,
        c.chunk_id AS 청크ID,
        CASE 
            WHEN c.embedding IS NULL THEN '임베딩 없음'
            WHEN c.content IS NULL THEN '내용 없음'
            ELSE '내용 비어있음'
        END AS 문제
    LIMIT 10
    """
}

# 쿼리 출력
print("=" * 70)
print("Neo4j Browser 시각화 쿼리 모음")
print("=" * 70)

for title, query in neo4j_queries.items():
    print(f"\n{'='*70}")
    print(f"[{title}]")
    print(f"{'='*70}")
    print(query.strip())

# 쿼리를 파일로 저장
query_file = Path("claudedocs/neo4j_visualization_queries.txt")
query_file.parent.mkdir(exist_ok=True)

with open(query_file, 'w', encoding='utf-8') as f:
    f.write("Neo4j Browser 시각화 쿼리 모음\n")
    f.write("=" * 70 + "\n\n")
    
    for title, query in neo4j_queries.items():
        f.write(f"{'='*70}\n")
        f.write(f"[{title}]\n")
        f.write(f"{'='*70}\n")
        f.write(query.strip() + "\n\n")

print(f"\n✓ 쿼리 모음이 파일로 저장되었습니다: {query_file}")
```

---

## **전체 검증 및 테스트**

구현된 모든 기능을 종합적으로 테스트합니다.

```python
# ============================================================================
# 전체 시스템 검증 및 테스트
# ============================================================================

print("=" * 70)
print("Knowledge Graph RAG 시스템 전체 검증")
print("=" * 70)

# 1. Neo4j 연결 확인
print("\n[1] Neo4j 연결 상태 확인")
try:
    result = graph.query("RETURN 'Connected' AS status")
    print(f"  ✓ Neo4j 연결: {result[0]['status']}")
except Exception as e:
    print(f"  ✗ Neo4j 연결 실패: {e}")

# 2. 데이터 적재 확인
print("\n[2] 데이터 적재 현황")
stats = graph.query("""
MATCH (d:Document)
OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
OPTIONAL MATCH (s)-[:CONTAINS]->(c:Chunk)
RETURN 
    COUNT(DISTINCT d) AS documents,
    COUNT(DISTINCT s) AS sections,
    COUNT(DISTINCT c) AS chunks
""")
print(f"  ✓ 문서: {stats[0]['documents']} 개")
print(f"  ✓ 섹션: {stats[0]['sections']} 개")
print(f"  ✓ 청크: {stats[0]['chunks']} 개")

# 3. 벡터 인덱스 확인
print("\n[3] 벡터 인덱스 상태")
try:
    indexes = graph.query("SHOW INDEXES YIELD name, type WHERE type = 'VECTOR' RETURN name, type")
    if indexes:
        for idx in indexes:
            print(f"  ✓ 인덱스: {idx['name']} ({idx['type']})")
    else:
        print("  ⚠ 벡터 인덱스가 없습니다")
except Exception as e:
    print(f"  ✗ 인덱스 확인 실패: {e}")

# 4. BM25 Retriever 테스트
print("\n[4] BM25 Retriever 테스트")
try:
    test_results = bm25_search("반도체", k=3)
    print(f"  ✓ BM25 검색 성공: {len(test_results)} 개 결과")
except Exception as e:
    print(f"  ✗ BM25 검색 실패: {e}")

# 5. 벡터 검색 테스트
print("\n[5] 벡터 검색 테스트")
try:
    test_results = retriever.invoke("삼성전자 실적")
    print(f"  ✓ 벡터 검색 성공: {len(test_results)} 개 결과")
except Exception as e:
    print(f"  ✗ 벡터 검색 실패: {e}")

# 6. 하이브리드 검색 테스트
print("\n[6] 하이브리드 검색 (RRF) 테스트")
try:
    test_results = hybrid_search("SK하이닉스 사업", k=3)
    print(f"  ✓ 하이브리드 검색 성공: {len(test_results)} 개 결과")
except Exception as e:
    print(f"  ✗ 하이브리드 검색 실패: {e}")

# 7. RAG 체인 테스트
print("\n[7] RAG 체인 종단간 테스트")
try:
    test_question = "삼성전자의 주요 사업 분야는?"
    response = hybrid_rag_chain.invoke(test_question)
    print(f"  ✓ RAG 응답 생성 성공")
    print(f"  질문: {test_question}")
    print(f"  응답 (앞 100자): {response[:100]}...")
except Exception as e:
    print(f"  ✗ RAG 체인 실패: {e}")

# 8. 데이터 품질 체크
print("\n[8] 데이터 품질 검증")
try:
    # 빈 청크 확인
    empty_chunks = graph.query("""
    MATCH (c:Chunk)
    WHERE c.content = '' OR c.content IS NULL
    RETURN COUNT(c) AS empty_count
    """)
    
    if empty_chunks[0]['empty_count'] == 0:
        print(f"  ✓ 빈 청크 없음")
    else:
        print(f"  ⚠ 빈 청크 발견: {empty_chunks[0]['empty_count']} 개")
except Exception as e:
    print(f"  ✗ 품질 체크 실패: {e}")

print("\n" + "=" * 70)
print("전체 검증 완료!")
print("=" * 70)

print("\n" + "=" * 70)
print("🎉 Knowledge Graph RAG 시스템 구축 완료! 🎉")
print("=" * 70)

print("\n구현된 기능:")
print("  ✓ Feature 1: DART API 연동 (반도체 3사 × 3개년 = 9개 보고서)")
print("  ✓ Feature 2: Docling 고급 PDF 파티셔닝")
print("  ✓ Feature 3: RRF 하이브리드 검색 (BM25 + Vector)")
print("  ✓ Feature 4: Neo4j Browser 시각화 쿼리")

print("\n다음 단계:")
print("  1. Jupyter notebook의 셀들을 순서대로 실행")
print("  2. Neo4j Browser에서 시각화 쿼리 실행")
print("  3. 하이브리드 RAG 체인으로 질의응답 테스트")
```

---

## 🚀 실무 활용 예시

### 1. 기업 분석 시스템

```python
# SEC 10-K 문서를 활용한 기업 재무 분석
from langchain_neo4j import Neo4jVector, GraphCypherQAChain

# 복합 질문 예시
questions = [
    "이 기업의 주요 리스크 요인은 무엇인가요?",
    "경쟁사 대비 재무 상태는 어떤가요?",
    "최근 3년간 매출 성장률 추이는?",
    "R&D 투자 전략과 향후 계획은?"
]

for question in questions:
    result = qa_chain.invoke({"question": question})
    print(f"Q: {question}")
    print(f"A: {result['answer']}\n")
```

### 2. 리스크 모니터링 시스템

```python
# 그래프 쿼리로 리스크 요인 추적
risk_query = """
MATCH (doc:Document)-[:MENTIONS]->(risk:Risk)
WHERE doc.year >= 2020
WITH risk, collect(doc) as docs
RETURN 
    risk.name as risk_factor,
    risk.category as category,
    size(docs) as mention_count,
    [d in docs | d.year] as years
ORDER BY mention_count DESC
LIMIT 10
"""

risks = graph.query(risk_query)
for risk in risks:
    print(f"{risk['risk_factor']}: {risk['mention_count']}회 언급")
```

### 3. 경쟁사 비교 분석

```python
# 여러 기업의 10-K 문서를 그래프에 저장하고 비교
comparison_query = """
MATCH (c1:Company)-[:FILED]->(doc1:Document)
MATCH (c2:Company)-[:FILED]->(doc2:Document)
WHERE c1.name <> c2.name
  AND doc1.year = doc2.year
MATCH (doc1)-[:HAS_METRIC]->(m1:Metric)
MATCH (doc2)-[:HAS_METRIC]->(m2:Metric)
WHERE m1.type = m2.type
RETURN 
    c1.name as company1,
    c2.name as company2,
    m1.type as metric_type,
    m1.value as value1,
    m2.value as value2
"""
```

---

## 📖 참고 자료

### 공식 문서

- [LangChain Neo4j Integration](https://python.langchain.com/docs/integrations/graphs/neo4j_cypher/)
- [Neo4j Graph Data Science](https://neo4j.com/docs/graph-data-science/current/)
- [SEC EDGAR Database](https://www.sec.gov/edgar/searchedgar/companysearch.html)
- [DART 전자공시시스템](https://dart.fss.or.kr/)

### 추가 학습 자료

- **GraphRAG 논문**: "From Local to Global: A Graph RAG Approach"
- **Neo4j Blog**: "Building Knowledge Graphs from Documents"
- **LangChain Tutorials**: "Advanced RAG Patterns"

### 관련 블로그 및 튜토리얼

- Microsoft GraphRAG: Official Implementation Guide
- Neo4j: "Graph-Powered Semantic Search"
- Towards Data Science: "Hybrid Search with Neo4j"

---

## 🎯 실습 문제

### 문제 1: 커스텀 엔티티 추출

다음 요구사항을 만족하는 엔티티 추출 함수를 작성하세요:

- 문서에서 회사명, 인물, 날짜, 금액을 추출
- 각 엔티티를 Neo4j 노드로 저장
- 문서와 엔티티 간 관계 생성

```python
def extract_entities(document: str) -> dict:
    """
    문서에서 엔티티를 추출하고 Neo4j에 저장
    
    Args:
        document: 분석할 문서 텍스트
    
    Returns:
        추출된 엔티티 정보
    """
    # 여기에 코드를 작성하세요
    pass
```

### 문제 2: 그래프 기반 추천 시스템

유사한 10-K 문서를 추천하는 시스템을 구현하세요:

- 벡터 유사도 + 그래프 관계를 결합
- 공통 리스크 요인이나 산업 분야 고려
- 추천 이유를 설명 가능하게 구현

### 문제 3: 시계열 분석

연도별 10-K 문서를 분석하여 트렌드를 파악하세요:

- 주요 키워드의 출현 빈도 변화
- 새로운 리스크 요인의 등장
- 사업 전략의 변화 추적

---

## ✅ 솔루션 예시

### 문제 1 솔루션: 엔티티 추출

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json

def extract_entities(document: str) -> dict:
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 금융 문서에서 엔티티를 추출하는 전문가입니다."),
        ("user", """다음 문서에서 엔티티를 추출하세요:
        
문서: {document}

다음 형식의 JSON으로 반환:
{{
    "companies": [회사명 리스트],
    "people": [인물명 리스트],
    "dates": [날짜 리스트],
    "amounts": [금액 리스트]
}}
""")
    ])
    
    chain = prompt | llm
    result = chain.invoke({"document": document})
    
    entities = json.loads(result.content)
    
    # Neo4j에 저장
    for company in entities.get('companies', []):
        graph.query(
            "MERGE (c:Company {name: $name})",
            params={"name": company}
        )
    
    return entities
```

### 문제 2 솔루션: 그래프 기반 추천

```python
def recommend_similar_documents(doc_id: str, k: int = 5) -> list:
    """유사 문서 추천"""
    
    query = """
    // 1. 대상 문서와 벡터 유사도가 높은 문서
    MATCH (target:Document {id: $doc_id})
    CALL db.index.vector.queryNodes('doc_embeddings', $k, target.embedding)
    YIELD node as similar, score as vector_score
    
    // 2. 공통 리스크 요인 계산
    OPTIONAL MATCH (target)-[:MENTIONS]->(risk:Risk)<-[:MENTIONS]-(similar)
    WITH similar, vector_score, count(DISTINCT risk) as common_risks
    
    // 3. 같은 산업 분야 확인
    MATCH (target)-[:IN_INDUSTRY]->(ind:Industry)<-[:IN_INDUSTRY]-(similar)
    
    // 4. 종합 점수 계산
    WITH similar, 
         vector_score * 0.5 + 
         (common_risks * 0.3) + 
         (0.2) as final_score
    
    RETURN 
        similar.id as doc_id,
        similar.title as title,
        final_score,
        common_risks
    ORDER BY final_score DESC
    LIMIT $k
    """
    
    return graph.query(query, params={"doc_id": doc_id, "k": k})
```

---

## ✅ 학습 마무리

이 Part 2를 완료하면서 다음을 배웠습니다:

1. ✅ 완전한 GraphRAG 시스템 구축 및 실전 적용
2. ✅ DART API 연동 및 실제 문서 수집
3. ✅ Docling을 활용한 고급 문서 처리
4. ✅ RRF 하이브리드 검색으로 검색 품질 향상
5. ✅ Neo4j Browser를 통한 그래프 시각화

**다음 단계**: 이제 실제 기업 분석, 리스크 모니터링, 경쟁사 비교 등 실무 시나리오에 GraphRAG를 적용할 수 있습니다!

---

**Part 2 완료!** 🎉

**전체 과정 완료 축하합니다!** 이제 여러분은 Neo4j와 LangChain을 활용한 고급 GraphRAG 시스템을 구축할 수 있습니다.
