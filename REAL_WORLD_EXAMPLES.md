# 실무 활용 예시 모음집

## 🏢 엔터프라이즈 급 시스템 예시

### 1. 대기업 내부 지식관리 시스템

```python
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

class EnterpriseKnowledgeSystem:
    """대기업용 지식관리 시스템"""

    def __init__(self, config_path: str = "config/enterprise.json"):
        self.config = self._load_config(config_path)
        self.llm = ChatOpenAI(
            model=self.config["model"],
            temperature=self.config["temperature"],
            api_key=self.config["openai_api_key"]
        )
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # 부서별 벡터 스토어
        self.department_stores = {}
        self.access_control = AccessController()
        self.audit_logger = AuditLogger()

    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        with open(config_path, 'r') as f:
            return json.load(f)

    def setup_department_stores(self, departments: List[str]):
        """부서별 벡터 스토어 설정"""
        for dept in departments:
            collection_name = f"knowledge_{dept.lower()}"
            persist_dir = f"./vectorstores/{dept.lower()}"

            self.department_stores[dept] = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=persist_dir
            )

    async def ingest_documents(self, dept: str, file_paths: List[str], user_id: str):
        """문서 일괄 등록"""
        if not self.access_control.can_upload(user_id, dept):
            raise PermissionError(f"User {user_id} cannot upload to {dept}")

        # 문서 처리 파이프라인
        documents = []

        for file_path in file_paths:
            # 파일 타입별 로더 선택
            loader = self._get_loader(file_path)
            docs = loader.load()

            # 메타데이터 강화
            for doc in docs:
                doc.metadata.update({
                    "department": dept,
                    "uploaded_by": user_id,
                    "upload_date": datetime.now().isoformat(),
                    "file_path": file_path,
                    "security_level": self._classify_security_level(doc.page_content)
                })

            documents.extend(docs)

        # 청킹 및 벡터화
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

        split_docs = text_splitter.split_documents(documents)

        # 벡터 스토어에 저장
        await self.department_stores[dept].aadd_documents(split_docs)

        # 감사 로그
        self.audit_logger.log_upload(user_id, dept, len(split_docs))

        return len(split_docs)

    def search_knowledge(self, query: str, user_id: str, departments: Optional[List[str]] = None) -> Dict:
        """지식 검색 (권한 기반)"""

        # 사용자 권한 확인
        accessible_depts = self.access_control.get_accessible_departments(user_id)

        if departments:
            departments = [d for d in departments if d in accessible_depts]
        else:
            departments = accessible_depts

        if not departments:
            return {"error": "No accessible departments"}

        # 부서별 검색 수행
        all_results = {}

        for dept in departments:
            if dept in self.department_stores:
                retriever = self.department_stores[dept].as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 5, "lambda_mult": 0.7}
                )

                docs = retriever.invoke(query)

                # 보안 레벨 필터링
                filtered_docs = self._filter_by_security_level(docs, user_id)

                all_results[dept] = filtered_docs

        # 통합 응답 생성
        response = self._generate_unified_response(query, all_results, user_id)

        # 검색 로그
        self.audit_logger.log_search(user_id, query, departments)

        return response

    def _get_loader(self, file_path: str):
        """파일 확장자별 로더 선택"""
        suffix = Path(file_path).suffix.lower()

        if suffix == '.pdf':
            return PyPDFLoader(file_path)
        elif suffix in ['.docx', '.doc']:
            return Docx2txtLoader(file_path)
        elif suffix == '.csv':
            return CSVLoader(file_path)
        elif suffix in ['.txt', '.md']:
            return TextLoader(file_path)
        else:
            return UnstructuredFileLoader(file_path)

    def _classify_security_level(self, content: str) -> str:
        """콘텐츠 보안 등급 분류"""
        sensitive_keywords = ["기밀", "confidential", "내부전용", "임원진", "재무제표"]

        content_lower = content.lower()
        if any(keyword in content_lower for keyword in sensitive_keywords):
            return "confidential"
        else:
            return "general"

    def _filter_by_security_level(self, docs: List[Document], user_id: str) -> List[Document]:
        """사용자 권한에 따른 문서 필터링"""
        user_clearance = self.access_control.get_security_clearance(user_id)

        filtered = []
        for doc in docs:
            doc_level = doc.metadata.get("security_level", "general")

            if self._can_access_level(user_clearance, doc_level):
                filtered.append(doc)

        return filtered

    def _can_access_level(self, user_clearance: str, doc_level: str) -> bool:
        """보안 등급 접근 권한 확인"""
        clearance_hierarchy = {
            "general": 1,
            "confidential": 2,
            "secret": 3
        }

        user_level = clearance_hierarchy.get(user_clearance, 0)
        required_level = clearance_hierarchy.get(doc_level, 1)

        return user_level >= required_level

class AccessController:
    """접근 권한 제어"""

    def __init__(self):
        self.user_permissions = self._load_permissions()

    def _load_permissions(self) -> Dict:
        """사용자 권한 정보 로드 (실제로는 데이터베이스에서)"""
        return {
            "john.doe": {
                "departments": ["IT", "HR", "General"],
                "security_clearance": "confidential"
            },
            "jane.smith": {
                "departments": ["Finance", "General"],
                "security_clearance": "secret"
            }
        }

    def can_upload(self, user_id: str, department: str) -> bool:
        """업로드 권한 확인"""
        user_perms = self.user_permissions.get(user_id, {})
        return department in user_perms.get("departments", [])

    def get_accessible_departments(self, user_id: str) -> List[str]:
        """접근 가능한 부서 목록 반환"""
        user_perms = self.user_permissions.get(user_id, {})
        return user_perms.get("departments", [])

    def get_security_clearance(self, user_id: str) -> str:
        """사용자 보안 등급 반환"""
        user_perms = self.user_permissions.get(user_id, {})
        return user_perms.get("security_clearance", "general")

class AuditLogger:
    """감사 로그 시스템"""

    def __init__(self, log_file: str = "logs/audit.log"):
        self.log_file = log_file
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    def log_upload(self, user_id: str, department: str, doc_count: int):
        """업로드 로그"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": "upload",
            "user_id": user_id,
            "department": department,
            "doc_count": doc_count
        }
        self._write_log(log_entry)

    def log_search(self, user_id: str, query: str, departments: List[str]):
        """검색 로그"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": "search",
            "user_id": user_id,
            "query": query[:100],  # 쿼리 일부만 로그
            "departments": departments
        }
        self._write_log(log_entry)

    def _write_log(self, entry: Dict):
        """로그 파일에 기록"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

# 사용 예시
async def main():
    # 시스템 초기화
    knowledge_system = EnterpriseKnowledgeSystem()

    # 부서별 스토어 설정
    departments = ["IT", "HR", "Finance", "General"]
    knowledge_system.setup_department_stores(departments)

    # 문서 업로드
    hr_docs = ["docs/employee_handbook.pdf", "docs/benefits_guide.docx"]
    await knowledge_system.ingest_documents("HR", hr_docs, "john.doe")

    # 지식 검색
    result = knowledge_system.search_knowledge(
        "휴가 정책에 대해 알려주세요",
        user_id="john.doe",
        departments=["HR"]
    )

    print(result["answer"])

# 실행
if __name__ == "__main__":
    asyncio.run(main())
```

### 2. 실시간 고객 지원 시스템

```python
import streamlit as st
import redis
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import uuid

class RealTimeCustomerSupport:
    """실시간 고객 지원 시스템"""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # 실시간 데이터 저장소
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)

        # FAQ 벡터 스토어
        self.faq_store = Chroma(
            collection_name="customer_faq",
            embedding_function=self.embeddings,
            persist_directory="./customer_faq_store"
        )

        # 대화 분류기
        self.intent_classifier = self._setup_intent_classifier()

        # 에스컬레이션 규칙
        self.escalation_rules = self._load_escalation_rules()

    def _setup_intent_classifier(self):
        """대화 의도 분류기 설정"""
        intent_prompt = ChatPromptTemplate.from_template("""
        고객 메시지의 의도를 다음 중 하나로 분류해주세요:

        의도 목록:
        - question: 일반적인 질문
        - complaint: 불만사항
        - technical_issue: 기술적 문제
        - billing: 결제/요금 관련
        - account: 계정 관련
        - urgent: 긴급 상황

        고객 메시지: {message}

        분류 결과를 JSON 형식으로 반환해주세요:
        {{
            "intent": "분류된 의도",
            "confidence": 신뢰도(0.0-1.0),
            "urgency": 긴급도(1-5)
        }}
        """)

        return intent_prompt | self.llm

    def _load_escalation_rules(self) -> Dict:
        """에스컬레이션 규칙 로드"""
        return {
            "urgent": {"threshold": 0.8, "target": "senior_agent"},
            "complaint": {"threshold": 0.7, "target": "supervisor"},
            "technical_issue": {"threshold": 0.6, "target": "tech_support"},
            "billing": {"threshold": 0.5, "target": "billing_team"}
        }

    def handle_customer_message(self, session_id: str, message: str, customer_info: Dict) -> Dict:
        """고객 메시지 처리"""

        # 1. 의도 분류
        intent_result = self._classify_intent(message)

        # 2. 세션 컨텍스트 로드
        session_context = self._get_session_context(session_id)

        # 3. FAQ 검색
        relevant_faqs = self._search_faq(message)

        # 4. 개인화된 응답 생성
        response = self._generate_personalized_response(
            message, intent_result, relevant_faqs, customer_info, session_context
        )

        # 5. 에스컬레이션 필요성 판단
        escalation_needed = self._check_escalation(intent_result, session_context)

        # 6. 세션 상태 업데이트
        self._update_session_context(session_id, message, response, intent_result)

        # 7. 실시간 메트릭 업데이트
        self._update_metrics(intent_result, escalation_needed)

        return {
            "response": response,
            "intent": intent_result,
            "escalation_needed": escalation_needed,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }

    def _classify_intent(self, message: str) -> Dict:
        """메시지 의도 분류"""
        result = self.intent_classifier.invoke({"message": message})

        try:
            return json.loads(result.content)
        except:
            return {"intent": "question", "confidence": 0.5, "urgency": 1}

    def _get_session_context(self, session_id: str) -> Dict:
        """세션 컨텍스트 로드"""
        context_key = f"session:{session_id}"
        context_data = self.redis_client.get(context_key)

        if context_data:
            return json.loads(context_data.decode())
        else:
            return {
                "messages": [],
                "start_time": datetime.now().isoformat(),
                "interaction_count": 0,
                "sentiment_history": [],
                "issues": []
            }

    def _search_faq(self, message: str, k: int = 3) -> List[Document]:
        """FAQ 검색"""
        return self.faq_store.similarity_search(message, k=k)

    def _generate_personalized_response(self, message: str, intent: Dict,
                                      faqs: List[Document], customer_info: Dict,
                                      session_context: Dict) -> str:
        """개인화된 응답 생성"""

        # 고객 정보 기반 개인화
        customer_tier = customer_info.get("tier", "standard")
        interaction_history = session_context.get("interaction_count", 0)

        # FAQ 컨텍스트 구성
        faq_context = "\n".join([doc.page_content for doc in faqs])

        personalized_prompt = ChatPromptTemplate.from_template("""
        당신은 {customer_tier} 등급 고객을 담당하는 전문 고객지원 담당자입니다.

        고객 정보:
        - 이름: {customer_name}
        - 등급: {customer_tier}
        - 이번 세션 상호작용 횟수: {interaction_count}
        - 감지된 의도: {intent}
        - 긴급도: {urgency}/5

        관련 FAQ 정보:
        {faq_context}

        이전 대화 맥락:
        {previous_messages}

        고객 메시지: {message}

        다음 지침을 따라 응답해주세요:
        1. 고객 등급에 맞는 적절한 톤 사용
        2. FAQ 정보를 기반으로 정확한 답변 제공
        3. 필요시 추가 질문 유도
        4. 긍정적이고 해결 지향적인 자세 유지
        """)

        # 이전 대화 요약
        previous_messages = self._summarize_previous_messages(session_context.get("messages", []))

        response_chain = personalized_prompt | self.llm

        response = response_chain.invoke({
            "customer_name": customer_info.get("name", "고객님"),
            "customer_tier": customer_tier,
            "interaction_count": interaction_history,
            "intent": intent.get("intent"),
            "urgency": intent.get("urgency", 1),
            "faq_context": faq_context,
            "previous_messages": previous_messages,
            "message": message
        })

        return response.content

    def _check_escalation(self, intent_result: Dict, session_context: Dict) -> Dict:
        """에스컬레이션 필요성 판단"""
        intent = intent_result.get("intent")
        confidence = intent_result.get("confidence", 0)
        urgency = intent_result.get("urgency", 1)
        interaction_count = session_context.get("interaction_count", 0)

        escalation = {"needed": False, "reason": "", "target": ""}

        # 규칙 기반 에스컬레이션
        if intent in self.escalation_rules:
            rule = self.escalation_rules[intent]
            if confidence >= rule["threshold"]:
                escalation = {
                    "needed": True,
                    "reason": f"High confidence {intent}",
                    "target": rule["target"]
                }

        # 긴급도 기반 에스컬레이션
        if urgency >= 4:
            escalation = {
                "needed": True,
                "reason": "High urgency",
                "target": "senior_agent"
            }

        # 반복 상호작용 기반 에스컬레이션
        if interaction_count >= 5:
            escalation = {
                "needed": True,
                "reason": "Multiple interactions",
                "target": "supervisor"
            }

        return escalation

    def _update_session_context(self, session_id: str, message: str,
                              response: str, intent: Dict):
        """세션 컨텍스트 업데이트"""
        context_key = f"session:{session_id}"
        context = self._get_session_context(session_id)

        # 메시지 추가
        context["messages"].append({
            "timestamp": datetime.now().isoformat(),
            "customer_message": message,
            "bot_response": response,
            "intent": intent
        })

        # 상호작용 횟수 증가
        context["interaction_count"] += 1

        # 감정 히스토리 업데이트 (실제로는 감정 분석 모델 사용)
        sentiment = self._analyze_sentiment(message)
        context["sentiment_history"].append(sentiment)

        # Redis에 저장 (1시간 TTL)
        self.redis_client.setex(
            context_key,
            3600,  # 1 hour
            json.dumps(context, ensure_ascii=False)
        )

    def _analyze_sentiment(self, message: str) -> str:
        """감정 분석 (단순화된 버전)"""
        positive_words = ["좋다", "감사", "만족", "훌륭", "완벽"]
        negative_words = ["싫다", "화나", "최악", "실망", "문제"]

        message_lower = message.lower()

        positive_count = sum(1 for word in positive_words if word in message_lower)
        negative_count = sum(1 for word in negative_words if word in message_lower)

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"

    def _update_metrics(self, intent_result: Dict, escalation_needed: Dict):
        """실시간 메트릭 업데이트"""
        current_hour = datetime.now().strftime("%Y%m%d%H")

        # 의도별 카운트
        intent = intent_result.get("intent", "unknown")
        intent_key = f"metrics:intent:{intent}:{current_hour}"
        self.redis_client.incr(intent_key)
        self.redis_client.expire(intent_key, 86400)  # 24시간 보관

        # 에스컬레이션 카운트
        if escalation_needed.get("needed"):
            escalation_key = f"metrics:escalation:{current_hour}"
            self.redis_client.incr(escalation_key)
            self.redis_client.expire(escalation_key, 86400)

    def get_real_time_dashboard_data(self) -> Dict:
        """실시간 대시보드 데이터 제공"""
        current_hour = datetime.now().strftime("%Y%m%d%H")

        # 지난 24시간 데이터
        hours = []
        for i in range(24):
            hour = (datetime.now() - timedelta(hours=i)).strftime("%Y%m%d%H")
            hours.append(hour)

        dashboard_data = {
            "hourly_interactions": {},
            "intent_distribution": {},
            "escalation_rate": {},
            "active_sessions": 0
        }

        # 시간별 상호작용
        for hour in hours:
            total_interactions = 0
            for intent in ["question", "complaint", "technical_issue", "billing", "account", "urgent"]:
                key = f"metrics:intent:{intent}:{hour}"
                count = int(self.redis_client.get(key) or 0)
                total_interactions += count

                if intent not in dashboard_data["intent_distribution"]:
                    dashboard_data["intent_distribution"][intent] = 0
                dashboard_data["intent_distribution"][intent] += count

            dashboard_data["hourly_interactions"][hour] = total_interactions

            # 에스컬레이션 비율
            escalation_key = f"metrics:escalation:{hour}"
            escalation_count = int(self.redis_client.get(escalation_key) or 0)

            if total_interactions > 0:
                dashboard_data["escalation_rate"][hour] = escalation_count / total_interactions
            else:
                dashboard_data["escalation_rate"][hour] = 0

        # 활성 세션 수
        active_sessions = len(self.redis_client.keys("session:*"))
        dashboard_data["active_sessions"] = active_sessions

        return dashboard_data

# Streamlit 웹 앱
def create_support_app():
    st.set_page_config(page_title="고객지원 시스템", layout="wide")

    support_system = RealTimeCustomerSupport()

    # 사이드바 - 대시보드
    with st.sidebar:
        st.title("📊 실시간 대시보드")

        dashboard_data = support_system.get_real_time_dashboard_data()

        st.metric("활성 세션", dashboard_data["active_sessions"])

        # 의도 분포 차트
        if dashboard_data["intent_distribution"]:
            st.subheader("의도 분포")
            st.bar_chart(dashboard_data["intent_distribution"])

        # 시간별 상호작용
        if dashboard_data["hourly_interactions"]:
            st.subheader("시간별 상호작용")
            st.line_chart(dashboard_data["hourly_interactions"])

    # 메인 화면 - 챗봇 인터페이스
    st.title("🤖 AI 고객지원 시스템")

    # 세션 ID 생성
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    # 고객 정보 입력
    with st.expander("고객 정보"):
        customer_name = st.text_input("고객명", value="김고객")
        customer_tier = st.selectbox("등급", ["standard", "premium", "vip"])
        customer_info = {"name": customer_name, "tier": customer_tier}

    # 채팅 인터페이스
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 메시지 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and "metadata" in message:
                metadata = message["metadata"]

                with st.expander("상세 정보"):
                    st.json(metadata)

    # 사용자 입력
    if prompt := st.chat_input("메시지를 입력하세요..."):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("응답 생성 중..."):
                result = support_system.handle_customer_message(
                    st.session_state.session_id,
                    prompt,
                    customer_info
                )

            response = result["response"]
            st.markdown(response)

            # 에스컬레이션 알림
            if result["escalation_needed"]["needed"]:
                st.warning(f"🚨 에스컬레이션 필요: {result['escalation_needed']['reason']}")
                st.info(f"담당팀: {result['escalation_needed']['target']}")

            # 메타데이터 표시
            with st.expander("분석 결과"):
                st.json({
                    "의도": result["intent"]["intent"],
                    "신뢰도": result["intent"]["confidence"],
                    "긴급도": result["intent"]["urgency"],
                    "에스컬레이션": result["escalation_needed"]
                })

        # 응답 저장
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "metadata": result
        })

# 앱 실행
if __name__ == "__main__":
    create_support_app()
```

### 3. 코드 리뷰 자동화 플랫폼

```python
import ast
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import git
from github import Github

class AutomatedCodeReviewPlatform:
    """자동화된 코드 리뷰 플랫폼"""

    def __init__(self, github_token: str):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.1)  # 코드 분석용 모델
        self.github = Github(github_token)

        # 코드 분석 도구들
        self.static_analyzers = {
            "python": ["flake8", "pylint", "black", "mypy"],
            "javascript": ["eslint", "prettier"],
            "java": ["checkstyle", "spotbugs"],
            "go": ["golint", "go vet"]
        }

        # 리뷰 템플릿
        self.review_templates = self._load_review_templates()

        # 보안 취약점 패턴
        self.security_patterns = self._load_security_patterns()

    def _load_review_templates(self) -> Dict:
        """리뷰 템플릿 로드"""
        return {
            "code_quality": """
            다음 코드의 품질을 분석하고 개선사항을 제안해주세요:

            코드:
            ```{language}
            {code}
            ```

            분석 항목:
            1. 가독성 및 명명 규칙
            2. 코드 구조 및 설계 패턴
            3. 성능 최적화 가능성
            4. 테스트 가능성
            5. 유지보수성

            구체적인 개선사항과 수정된 코드를 제공해주세요.
            """,

            "security_review": """
            다음 코드의 보안 취약점을 분석해주세요:

            코드:
            ```{language}
            {code}
            ```

            확인 항목:
            1. 입력 값 검증
            2. SQL 인젝션 가능성
            3. XSS 취약점
            4. 인증/인가 이슈
            5. 데이터 노출 위험
            6. 암호화 관련 이슈

            발견된 취약점과 해결방안을 제시해주세요.
            """,

            "performance_review": """
            다음 코드의 성능을 분석하고 최적화 방안을 제안해주세요:

            코드:
            ```{language}
            {code}
            ```

            분석 항목:
            1. 시간 복잡도 분석
            2. 공간 복잡도 분석
            3. 병목 구간 식별
            4. 캐싱 최적화
            5. 데이터베이스 쿼리 최적화
            6. 메모리 사용량 최적화

            구체적인 성능 개선 코드를 제공해주세요.
            """
        }

    def _load_security_patterns(self) -> Dict:
        """보안 취약점 패턴 로드"""
        return {
            "python": [
                r"eval\s*\(",  # eval() 사용
                r"exec\s*\(",  # exec() 사용
                r"__import__\s*\(",  # 동적 import
                r"pickle\.loads?\s*\(",  # pickle 역직렬화
                r"yaml\.load\s*\(",  # YAML 로드 (안전하지 않은)
                r"shell=True",  # 셸 명령 실행
            ],
            "javascript": [
                r"eval\s*\(",  # eval() 사용
                r"innerHTML\s*=",  # innerHTML 직접 설정
                r"document\.write\s*\(",  # document.write 사용
                r"setTimeout\s*\(\s*[\"']",  # setTimeout with string
            ],
            "sql": [
                r"SELECT\s+.*\+.*FROM",  # SQL 문자열 결합
                r"WHERE\s+.*\+",  # WHERE 절 문자열 결합
                r"'.*\+.*'",  # 문자열 결합된 쿼리
            ]
        }

    def review_pull_request(self, repo_name: str, pr_number: int) -> Dict:
        """Pull Request 자동 리뷰"""

        # GitHub에서 PR 정보 가져오기
        repo = self.github.get_repo(repo_name)
        pr = repo.get_pull(pr_number)

        # 변경된 파일들 분석
        files = pr.get_files()
        review_results = []

        for file in files:
            if file.status in ["added", "modified"]:
                file_review = self._review_single_file(
                    file.filename,
                    file.patch,
                    file.raw_url
                )
                review_results.append(file_review)

        # 전체적인 PR 분석
        overall_review = self._analyze_pr_overall(pr, review_results)

        # 리뷰 결과 종합
        final_review = {
            "pr_number": pr_number,
            "title": pr.title,
            "files_reviewed": len(review_results),
            "overall_score": overall_review["score"],
            "recommendations": overall_review["recommendations"],
            "file_reviews": review_results,
            "approval_status": self._determine_approval_status(review_results)
        }

        # GitHub에 리뷰 코멘트 작성
        self._post_review_to_github(pr, final_review)

        return final_review

    def _review_single_file(self, filename: str, patch: str, raw_url: str) -> Dict:
        """단일 파일 리뷰"""

        # 파일 언어 감지
        language = self._detect_language(filename)

        # 전체 파일 내용 가져오기
        file_content = self._fetch_file_content(raw_url)

        # 정적 분석 실행
        static_analysis = self._run_static_analysis(filename, file_content, language)

        # AI 코드 리뷰 실행
        ai_reviews = {}

        # 품질 리뷰
        ai_reviews["quality"] = self._ai_code_review(
            file_content, language, "code_quality"
        )

        # 보안 리뷰
        ai_reviews["security"] = self._ai_code_review(
            file_content, language, "security_review"
        )

        # 성능 리뷰
        ai_reviews["performance"] = self._ai_code_review(
            file_content, language, "performance_review"
        )

        # 복잡도 분석
        complexity_analysis = self._analyze_complexity(file_content, language)

        # 테스트 커버리지 분석
        test_coverage = self._analyze_test_coverage(filename, file_content)

        return {
            "filename": filename,
            "language": language,
            "static_analysis": static_analysis,
            "ai_reviews": ai_reviews,
            "complexity": complexity_analysis,
            "test_coverage": test_coverage,
            "overall_score": self._calculate_file_score(static_analysis, ai_reviews, complexity_analysis),
            "critical_issues": self._extract_critical_issues(static_analysis, ai_reviews)
        }

    def _detect_language(self, filename: str) -> str:
        """파일 확장자로 언어 감지"""
        suffix = Path(filename).suffix.lower()

        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".go": "go",
            ".cpp": "cpp",
            ".c": "c",
            ".cs": "csharp",
            ".rb": "ruby",
            ".php": "php"
        }

        return language_map.get(suffix, "unknown")

    def _fetch_file_content(self, raw_url: str) -> str:
        """원격 파일 내용 가져오기"""
        import requests
        response = requests.get(raw_url)
        return response.text

    def _run_static_analysis(self, filename: str, content: str, language: str) -> Dict:
        """정적 분석 도구 실행"""
        if language not in self.static_analyzers:
            return {"tools": [], "issues": []}

        issues = []
        tools_used = []

        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(mode='w', suffix=Path(filename).suffix, delete=False) as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
            for tool in self.static_analyzers[language]:
                try:
                    tool_issues = self._run_single_analyzer(tool, tmp_file_path, language)
                    issues.extend(tool_issues)
                    tools_used.append(tool)
                except Exception as e:
                    print(f"Error running {tool}: {e}")
        finally:
            # 임시 파일 삭제
            Path(tmp_file_path).unlink()

        return {
            "tools": tools_used,
            "issues": issues
        }

    def _run_single_analyzer(self, tool: str, file_path: str, language: str) -> List[Dict]:
        """개별 정적 분석 도구 실행"""
        issues = []

        try:
            if tool == "flake8" and language == "python":
                result = subprocess.run(
                    ["flake8", file_path, "--format=json"],
                    capture_output=True, text=True
                )
                if result.stdout:
                    flake8_issues = json.loads(result.stdout)
                    for issue in flake8_issues:
                        issues.append({
                            "tool": "flake8",
                            "line": issue.get("line_number"),
                            "column": issue.get("column_number"),
                            "message": issue.get("text"),
                            "code": issue.get("code"),
                            "severity": "warning"
                        })

            elif tool == "pylint" and language == "python":
                result = subprocess.run(
                    ["pylint", file_path, "--output-format=json"],
                    capture_output=True, text=True
                )
                if result.stdout:
                    pylint_issues = json.loads(result.stdout)
                    for issue in pylint_issues:
                        issues.append({
                            "tool": "pylint",
                            "line": issue.get("line"),
                            "column": issue.get("column"),
                            "message": issue.get("message"),
                            "code": issue.get("message-id"),
                            "severity": issue.get("type")
                        })

        except Exception as e:
            print(f"Error running {tool}: {e}")

        return issues

    def _ai_code_review(self, content: str, language: str, review_type: str) -> Dict:
        """AI 기반 코드 리뷰"""

        template = self.review_templates.get(review_type, self.review_templates["code_quality"])

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm

        try:
            response = chain.invoke({
                "code": content[:4000],  # 토큰 제한 고려
                "language": language
            })

            return {
                "review_type": review_type,
                "analysis": response.content,
                "recommendations": self._extract_recommendations(response.content),
                "severity": self._assess_severity(response.content)
            }

        except Exception as e:
            return {
                "review_type": review_type,
                "analysis": f"Error during AI review: {e}",
                "recommendations": [],
                "severity": "unknown"
            }

    def _analyze_complexity(self, content: str, language: str) -> Dict:
        """코드 복잡도 분석"""

        if language == "python":
            return self._analyze_python_complexity(content)
        else:
            return {"complexity": "unknown", "metrics": {}}

    def _analyze_python_complexity(self, content: str) -> Dict:
        """Python 코드 복잡도 분석"""
        try:
            tree = ast.parse(content)

            complexity_analyzer = PythonComplexityAnalyzer()
            complexity_analyzer.visit(tree)

            return {
                "cyclomatic_complexity": complexity_analyzer.cyclomatic_complexity,
                "function_count": complexity_analyzer.function_count,
                "class_count": complexity_analyzer.class_count,
                "max_nesting_depth": complexity_analyzer.max_nesting_depth,
                "lines_of_code": len(content.split('\n')),
                "complexity_score": complexity_analyzer.get_complexity_score()
            }

        except Exception as e:
            return {"error": str(e)}

    def _analyze_test_coverage(self, filename: str, content: str) -> Dict:
        """테스트 커버리지 분석"""

        is_test_file = any(pattern in filename.lower() for pattern in ["test_", "_test", ".test."])

        if is_test_file:
            return {
                "is_test_file": True,
                "test_methods": self._count_test_methods(content),
                "assertions": self._count_assertions(content)
            }
        else:
            return {
                "is_test_file": False,
                "has_corresponding_test": self._check_test_file_exists(filename),
                "testable_functions": self._count_testable_functions(content)
            }

    def _calculate_file_score(self, static_analysis: Dict, ai_reviews: Dict, complexity: Dict) -> float:
        """파일 전체 점수 계산"""
        score = 100.0

        # 정적 분석 이슈로 점수 차감
        error_count = len([issue for issue in static_analysis.get("issues", [])
                          if issue.get("severity") == "error"])
        warning_count = len([issue for issue in static_analysis.get("issues", [])
                            if issue.get("severity") == "warning"])

        score -= (error_count * 10 + warning_count * 5)

        # 복잡도로 점수 차감
        cyclomatic = complexity.get("cyclomatic_complexity", 0)
        if cyclomatic > 10:
            score -= (cyclomatic - 10) * 2

        # AI 리뷰 결과로 점수 차감
        for review_type, review in ai_reviews.items():
            severity = review.get("severity", "low")
            if severity == "high":
                score -= 15
            elif severity == "medium":
                score -= 10
            elif severity == "low":
                score -= 5

        return max(0, score)

    def _post_review_to_github(self, pr, review_result: Dict):
        """GitHub에 리뷰 결과 포스팅"""

        # 리뷰 코멘트 생성
        review_body = self._generate_review_comment(review_result)

        # PR에 리뷰 작성
        pr.create_review(
            body=review_body,
            event="COMMENT"  # 또는 "APPROVE", "REQUEST_CHANGES"
        )

        # 심각한 이슈가 있으면 개별 코멘트도 추가
        for file_review in review_result["file_reviews"]:
            if file_review["critical_issues"]:
                for issue in file_review["critical_issues"]:
                    pr.create_review_comment(
                        body=f"🚨 Critical Issue: {issue['message']}",
                        path=file_review["filename"],
                        line=issue.get("line", 1)
                    )

    def _generate_review_comment(self, review_result: Dict) -> str:
        """리뷰 코멘트 생성"""

        comment = f"""
## 🤖 자동 코드 리뷰 결과

### 📊 전체 요약
- **검토 파일 수**: {review_result['files_reviewed']}
- **전체 점수**: {review_result['overall_score']:.1f}/100
- **승인 상태**: {review_result['approval_status']}

### 📋 주요 권장사항
"""

        for rec in review_result['recommendations'][:5]:  # 상위 5개만
            comment += f"- {rec}\n"

        comment += "\n### 📁 파일별 상세 분석\n"

        for file_review in review_result['file_reviews']:
            filename = file_review['filename']
            score = file_review['overall_score']
            critical_count = len(file_review['critical_issues'])

            comment += f"- **{filename}**: {score:.1f}/100"
            if critical_count > 0:
                comment += f" ⚠️ {critical_count}개 심각한 이슈"
            comment += "\n"

        comment += "\n---\n*이 리뷰는 AI 도구에 의해 자동 생성되었습니다. 사람의 최종 검토가 권장됩니다.*"

        return comment

class PythonComplexityAnalyzer(ast.NodeVisitor):
    """Python 코드 복잡도 분석기"""

    def __init__(self):
        self.cyclomatic_complexity = 1  # 기본 복잡도
        self.function_count = 0
        self.class_count = 0
        self.max_nesting_depth = 0
        self.current_nesting_depth = 0

    def visit_FunctionDef(self, node):
        self.function_count += 1
        self.current_nesting_depth += 1
        self.max_nesting_depth = max(self.max_nesting_depth, self.current_nesting_depth)

        # 함수 내부의 복잡도 증가 요소들 방문
        self.generic_visit(node)

        self.current_nesting_depth -= 1

    def visit_ClassDef(self, node):
        self.class_count += 1
        self.generic_visit(node)

    def visit_If(self, node):
        self.cyclomatic_complexity += 1
        self.current_nesting_depth += 1
        self.max_nesting_depth = max(self.max_nesting_depth, self.current_nesting_depth)
        self.generic_visit(node)
        self.current_nesting_depth -= 1

    def visit_While(self, node):
        self.cyclomatic_complexity += 1
        self.current_nesting_depth += 1
        self.max_nesting_depth = max(self.max_nesting_depth, self.current_nesting_depth)
        self.generic_visit(node)
        self.current_nesting_depth -= 1

    def visit_For(self, node):
        self.cyclomatic_complexity += 1
        self.current_nesting_depth += 1
        self.max_nesting_depth = max(self.max_nesting_depth, self.current_nesting_depth)
        self.generic_visit(node)
        self.current_nesting_depth -= 1

    def visit_ExceptHandler(self, node):
        self.cyclomatic_complexity += 1
        self.generic_visit(node)

    def get_complexity_score(self) -> str:
        """복잡도 점수 반환"""
        if self.cyclomatic_complexity <= 5:
            return "low"
        elif self.cyclomatic_complexity <= 10:
            return "medium"
        else:
            return "high"

# 사용 예시
def main():
    # GitHub 토큰 설정
    github_token = "your_github_token_here"

    # 코드 리뷰 플랫폼 초기화
    review_platform = AutomatedCodeReviewPlatform(github_token)

    # Pull Request 리뷰 실행
    repo_name = "organization/repository"
    pr_number = 123

    review_result = review_platform.review_pull_request(repo_name, pr_number)

    # 결과 출력
    print(f"리뷰 완료: {review_result['overall_score']:.1f}/100")
    print(f"승인 상태: {review_result['approval_status']}")

    for file_review in review_result['file_reviews']:
        print(f"\n파일: {file_review['filename']}")
        print(f"점수: {file_review['overall_score']:.1f}/100")

        if file_review['critical_issues']:
            print("심각한 이슈:")
            for issue in file_review['critical_issues']:
                print(f"  - {issue['message']} (라인 {issue.get('line', '?')})")

if __name__ == "__main__":
    main()
```

### 4. 개인화된 학습 도우미

```python
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

class PersonalizedLearningAssistant:
    """개인화된 학습 도우미"""

    def __init__(self, db_path: str = "learning_assistant.db"):
        self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.5)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.db_path = db_path

        # 학습 스타일 분류기
        self.learning_style_classifier = self._setup_learning_style_classifier()

        # 지식 그래프
        self.knowledge_graph = self._build_knowledge_graph()

        # 개념별 벡터 스토어
        self.concept_vectorstores = {}

        # 데이터베이스 초기화
        self._init_database()

    def _init_database(self):
        """학습 데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 학습자 프로필 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learner_profiles (
                user_id TEXT PRIMARY KEY,
                learning_style TEXT,
                knowledge_level TEXT,
                preferred_pace TEXT,
                strengths TEXT,
                weaknesses TEXT,
                goals TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        ''')

        # 학습 세션 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                concept TEXT,
                session_type TEXT,
                duration_minutes INTEGER,
                performance_score REAL,
                difficulty_level INTEGER,
                questions_asked INTEGER,
                correct_answers INTEGER,
                session_date TIMESTAMP,
                notes TEXT
            )
        ''')

        # 개념 관계 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS concept_relationships (
                concept_a TEXT,
                concept_b TEXT,
                relationship_type TEXT,
                strength REAL,
                PRIMARY KEY (concept_a, concept_b, relationship_type)
            )
        ''')

        # 학습 경로 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_paths (
                path_id TEXT PRIMARY KEY,
                user_id TEXT,
                path_name TEXT,
                concepts TEXT,
                estimated_duration INTEGER,
                difficulty_progression TEXT,
                created_at TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

    def _setup_learning_style_classifier(self):
        """학습 스타일 분류기 설정"""

        learning_style_prompt = ChatPromptTemplate.from_template("""
        다음 학습자의 응답과 행동 패턴을 분석하여 학습 스타일을 판단해주세요:

        학습자 정보:
        - 질문 응답: {responses}
        - 선호하는 학습 방식: {preferences}
        - 과거 성과 패턴: {performance_history}

        학습 스타일 분류:
        1. Visual (시각적): 다이어그램, 차트, 이미지 선호
        2. Auditory (청각적): 설명, 토론, 음성 학습 선호
        3. Kinesthetic (체감각적): 실습, 실험, 손으로 하는 활동 선호
        4. Reading/Writing (읽기/쓰기): 텍스트 기반 학습 선호

        분석 결과를 JSON 형식으로 반환해주세요:
        {{
            "primary_style": "주요 학습 스타일",
            "secondary_style": "보조 학습 스타일",
            "confidence": "신뢰도 (0.0-1.0)",
            "recommendations": ["추천 학습 방법 리스트"]
        }}
        """)

        return learning_style_prompt | self.llm

    def _build_knowledge_graph(self) -> Dict:
        """지식 그래프 구축"""

        # 기본 개념들과 관계 정의
        knowledge_graph = {
            "programming": {
                "prerequisites": [],
                "subtopics": ["variables", "functions", "loops", "conditionals", "data_structures"],
                "difficulty": 2,
                "estimated_hours": 40
            },
            "variables": {
                "prerequisites": [],
                "subtopics": ["data_types", "scope", "naming_conventions"],
                "difficulty": 1,
                "estimated_hours": 4
            },
            "functions": {
                "prerequisites": ["variables"],
                "subtopics": ["parameters", "return_values", "recursion", "lambdas"],
                "difficulty": 2,
                "estimated_hours": 8
            },
            "data_structures": {
                "prerequisites": ["variables", "functions"],
                "subtopics": ["lists", "dictionaries", "sets", "tuples"],
                "difficulty": 3,
                "estimated_hours": 12
            },
            "machine_learning": {
                "prerequisites": ["programming", "statistics", "linear_algebra"],
                "subtopics": ["supervised_learning", "unsupervised_learning", "deep_learning"],
                "difficulty": 4,
                "estimated_hours": 80
            }
        }

        return knowledge_graph

    def create_learner_profile(self, user_id: str, initial_assessment: Dict) -> Dict:
        """학습자 프로필 생성"""

        # 학습 스타일 분석
        style_result = self.learning_style_classifier.invoke({
            "responses": initial_assessment.get("responses", ""),
            "preferences": initial_assessment.get("preferences", ""),
            "performance_history": initial_assessment.get("performance_history", "")
        })

        try:
            style_analysis = json.loads(style_result.content)
        except:
            style_analysis = {
                "primary_style": "Visual",
                "secondary_style": "Reading/Writing",
                "confidence": 0.7,
                "recommendations": ["다양한 학습 방법 시도"]
            }

        # 지식 수준 평가
        knowledge_level = self._assess_knowledge_level(user_id, initial_assessment)

        # 학습 목표 설정
        learning_goals = self._set_learning_goals(initial_assessment, knowledge_level)

        # 프로필 저장
        profile = {
            "user_id": user_id,
            "learning_style": style_analysis["primary_style"],
            "knowledge_level": knowledge_level,
            "preferred_pace": initial_assessment.get("preferred_pace", "medium"),
            "strengths": json.dumps(initial_assessment.get("strengths", [])),
            "weaknesses": json.dumps(initial_assessment.get("weaknesses", [])),
            "goals": json.dumps(learning_goals),
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }

        self._save_learner_profile(profile)

        return profile

    def generate_personalized_lesson(self, user_id: str, concept: str) -> Dict:
        """개인화된 레슨 생성"""

        # 학습자 프로필 로드
        profile = self._get_learner_profile(user_id)

        # 개념 정보 가져오기
        concept_info = self.knowledge_graph.get(concept, {})

        # 선수 지식 확인
        prerequisites_met = self._check_prerequisites(user_id, concept)

        if not prerequisites_met["all_met"]:
            return {
                "status": "prerequisites_needed",
                "missing_prerequisites": prerequisites_met["missing"],
                "recommended_path": self._generate_prerequisite_path(
                    user_id, prerequisites_met["missing"]
                )
            }

        # 현재 지식 수준 평가
        current_knowledge = self._assess_concept_knowledge(user_id, concept)

        # 학습 스타일에 맞는 레슨 생성
        lesson_content = self._create_adaptive_lesson(
            concept, profile, current_knowledge, concept_info
        )

        # 연습 문제 생성
        practice_problems = self._generate_practice_problems(
            concept, current_knowledge["level"], profile["learning_style"]
        )

        # 학습 세션 기록 준비
        session_id = self._create_learning_session(user_id, concept)

        return {
            "status": "ready",
            "session_id": session_id,
            "concept": concept,
            "lesson_content": lesson_content,
            "practice_problems": practice_problems,
            "estimated_duration": concept_info.get("estimated_hours", 2),
            "difficulty_level": current_knowledge["level"],
            "learning_objectives": self._define_learning_objectives(concept, current_knowledge)
        }

    def _create_adaptive_lesson(self, concept: str, profile: Dict,
                              current_knowledge: Dict, concept_info: Dict) -> Dict:
        """적응형 레슨 콘텐츠 생성"""

        learning_style = profile["learning_style"]
        knowledge_level = current_knowledge["level"]

        # 학습 스타일별 프롬프트 템플릿
        style_prompts = {
            "Visual": """
            {concept}에 대한 시각적 학습자를 위한 레슨을 만들어주세요.

            학습자 정보:
            - 현재 지식 수준: {knowledge_level}/5
            - 개념: {concept}
            - 학습 목표: {objectives}

            다음 요소를 포함해주세요:
            1. 개념을 설명하는 다이어그램 또는 차트 제안
            2. 시각적 예시와 비유
            3. 단계별 시각적 프로세스
            4. 인포그래픽 스타일의 요약
            5. 마인드맵 구조 제안

            실제 시각 자료를 생성할 수는 없지만, 자세한 설명과 ASCII 아트를 활용해주세요.
            """,

            "Auditory": """
            {concept}에 대한 청각적 학습자를 위한 레슨을 만들어주세요.

            학습자 정보:
            - 현재 지식 수준: {knowledge_level}/5
            - 개념: {concept}
            - 학습 목표: {objectives}

            다음 요소를 포함해주세요:
            1. 대화형 설명 (질문과 답변 형식)
            2. 운율이 있는 기억법
            3. 스토리텔링을 통한 개념 설명
            4. 토론 주제 제안
            5. "소리내어 생각하기" 연습

            설명은 마치 강의를 듣는 것처럼 자연스럽게 작성해주세요.
            """,

            "Kinesthetic": """
            {concept}에 대한 체감각적 학습자를 위한 레슨을 만들어주세요.

            학습자 정보:
            - 현재 지식 수준: {knowledge_level}/5
            - 개념: {concept}
            - 학습 목표: {objectives}

            다음 요소를 포함해주세요:
            1. 실습 활동과 코딩 예제
            2. 단계별 따라하기 튜토리얼
            3. 실험과 시행착오 학습
            4. 프로젝트 기반 학습 제안
            5. 몸짓이나 동작을 활용한 기억법

            "직접 해보기"에 중점을 둔 실용적인 접근법을 사용해주세요.
            """,

            "Reading/Writing": """
            {concept}에 대한 읽기/쓰기 학습자를 위한 레슨을 만들어주세요.

            학습자 정보:
            - 현재 지식 수준: {knowledge_level}/5
            - 개념: {concept}
            - 학습 목표: {objectives}

            다음 요소를 포함해주세요:
            1. 구조화된 텍스트 설명
            2. 핵심 용어 정의와 용어집
            3. 요약 정리와 체크리스트
            4. 노트 작성 가이드
            5. 추가 읽기 자료 추천

            논리적이고 체계적인 텍스트 구조로 작성해주세요.
            """
        }

        # 선택된 학습 스타일에 맞는 프롬프트 사용
        prompt_template = style_prompts.get(learning_style, style_prompts["Visual"])

        lesson_prompt = ChatPromptTemplate.from_template(prompt_template)
        lesson_chain = lesson_prompt | self.llm

        objectives = self._define_learning_objectives(concept, current_knowledge)

        lesson_response = lesson_chain.invoke({
            "concept": concept,
            "knowledge_level": knowledge_level,
            "objectives": objectives
        })

        return {
            "content": lesson_response.content,
            "style": learning_style,
            "difficulty": knowledge_level,
            "estimated_reading_time": len(lesson_response.content.split()) // 200  # 분 단위
        }

    def _generate_practice_problems(self, concept: str, difficulty_level: int,
                                  learning_style: str) -> List[Dict]:
        """연습 문제 생성"""

        problem_prompt = ChatPromptTemplate.from_template("""
        {concept}에 대한 연습 문제를 생성해주세요.

        요구사항:
        - 난이도: {difficulty_level}/5
        - 학습 스타일: {learning_style}
        - 문제 개수: 5개

        각 문제는 다음 형식으로 작성해주세요:

        문제 1:
        [문제 설명]

        선택지 (객관식인 경우):
        A) 선택지 1
        B) 선택지 2
        C) 선택지 3
        D) 선택지 4

        정답: [정답]
        해설: [상세 해설]

        다양한 유형의 문제 (객관식, 주관식, 실습)를 포함해주세요.
        """)

        problem_chain = problem_prompt | self.llm

        problems_response = problem_chain.invoke({
            "concept": concept,
            "difficulty_level": difficulty_level,
            "learning_style": learning_style
        })

        # 응답을 파싱하여 문제별로 분리
        problems = self._parse_problems(problems_response.content)

        return problems

    def evaluate_learning_session(self, session_id: str, answers: List[Dict],
                                time_spent: int) -> Dict:
        """학습 세션 평가"""

        # 세션 정보 가져오기
        session_info = self._get_session_info(session_id)

        # 답안 채점
        grading_results = self._grade_answers(answers, session_info["concept"])

        # 성과 분석
        performance_analysis = self._analyze_performance(grading_results, time_spent)

        # 개인화된 피드백 생성
        feedback = self._generate_personalized_feedback(
            session_info["user_id"], session_info["concept"],
            grading_results, performance_analysis
        )

        # 다음 학습 추천
        next_recommendations = self._recommend_next_steps(
            session_info["user_id"], session_info["concept"], performance_analysis
        )

        # 세션 결과 저장
        self._save_session_results(session_id, grading_results, performance_analysis)

        # 학습자 프로필 업데이트
        self._update_learner_progress(session_info["user_id"], session_info["concept"], performance_analysis)

        return {
            "session_id": session_id,
            "score": grading_results["score"],
            "correct_answers": grading_results["correct_count"],
            "total_questions": grading_results["total_count"],
            "time_spent_minutes": time_spent,
            "performance_level": performance_analysis["level"],
            "strengths": performance_analysis["strengths"],
            "areas_for_improvement": performance_analysis["weaknesses"],
            "feedback": feedback,
            "next_recommendations": next_recommendations,
            "achievement_unlocked": self._check_achievements(session_info["user_id"], performance_analysis)
        }

    def generate_learning_analytics(self, user_id: str, timeframe: str = "last_month") -> Dict:
        """학습 분석 대시보드 생성"""

        # 시간 범위 설정
        end_date = datetime.now()
        if timeframe == "last_week":
            start_date = end_date - timedelta(weeks=1)
        elif timeframe == "last_month":
            start_date = end_date - timedelta(days=30)
        elif timeframe == "last_quarter":
            start_date = end_date - timedelta(days=90)
        else:
            start_date = end_date - timedelta(days=30)

        # 학습 세션 데이터 수집
        sessions_data = self._get_sessions_data(user_id, start_date, end_date)

        # 성과 분석
        performance_analytics = self._calculate_performance_metrics(sessions_data)

        # 학습 패턴 분석
        learning_patterns = self._analyze_learning_patterns(sessions_data)

        # 개념별 숙련도 분석
        concept_mastery = self._analyze_concept_mastery(user_id, sessions_data)

        # 학습 목표 진도 추적
        goal_progress = self._track_goal_progress(user_id)

        # 시각화 데이터 준비
        visualizations = self._prepare_visualizations(sessions_data, performance_analytics)

        return {
            "timeframe": timeframe,
            "summary": {
                "total_sessions": len(sessions_data),
                "total_study_time": sum(s["duration_minutes"] for s in sessions_data),
                "average_score": performance_analytics["average_score"],
                "improvement_rate": performance_analytics["improvement_rate"]
            },
            "performance_trends": performance_analytics["trends"],
            "learning_patterns": learning_patterns,
            "concept_mastery": concept_mastery,
            "goal_progress": goal_progress,
            "visualizations": visualizations,
            "recommendations": self._generate_analytics_recommendations(
                performance_analytics, learning_patterns, concept_mastery
            )
        }

    def create_adaptive_learning_path(self, user_id: str, target_concept: str,
                                    deadline: Optional[datetime] = None) -> Dict:
        """적응형 학습 경로 생성"""

        # 현재 지식 상태 평가
        current_state = self._assess_comprehensive_knowledge(user_id)

        # 목표 개념까지의 경로 계산
        learning_path = self._calculate_optimal_path(current_state, target_concept)

        # 개인화된 일정 생성
        schedule = self._generate_personalized_schedule(
            user_id, learning_path, deadline
        )

        # 마일스톤 설정
        milestones = self._set_learning_milestones(learning_path, schedule)

        # 학습 경로 저장
        path_id = self._save_learning_path(user_id, learning_path, schedule, milestones)

        return {
            "path_id": path_id,
            "target_concept": target_concept,
            "total_concepts": len(learning_path["concepts"]),
            "estimated_duration_weeks": schedule["total_weeks"],
            "daily_study_time_minutes": schedule["daily_minutes"],
            "learning_path": learning_path,
            "schedule": schedule,
            "milestones": milestones,
            "success_probability": self._predict_success_probability(user_id, learning_path)
        }

# Streamlit 웹 앱
def create_learning_assistant_app():
    st.set_page_config(
        page_title="AI 학습 도우미",
        page_icon="🎓",
        layout="wide"
    )

    assistant = PersonalizedLearningAssistant()

    # 사이드바 - 학습자 정보
    with st.sidebar:
        st.title("👤 학습자 프로필")

        user_id = st.text_input("사용자 ID", value="learner_001")

        # 프로필이 없으면 초기 평가 진행
        profile = assistant._get_learner_profile(user_id)

        if not profile:
            st.warning("프로필을 먼저 생성해주세요.")

            with st.expander("초기 평가"):
                preferred_pace = st.selectbox("선호하는 학습 속도", ["slow", "medium", "fast"])
                learning_goals = st.multiselect("학습 목표", [
                    "programming", "machine_learning", "data_science",
                    "web_development", "mobile_development"
                ])
                strengths = st.multiselect("강점 분야", [
                    "logical_thinking", "creativity", "problem_solving",
                    "attention_to_detail", "persistence"
                ])

                if st.button("프로필 생성"):
                    initial_assessment = {
                        "preferred_pace": preferred_pace,
                        "goals": learning_goals,
                        "strengths": strengths,
                        "responses": "기초부터 차근차근 배우고 싶습니다.",
                        "preferences": "실습 위주의 학습을 선호합니다."
                    }

                    profile = assistant.create_learner_profile(user_id, initial_assessment)
                    st.success("프로필이 생성되었습니다!")
                    st.rerun()

        else:
            st.success(f"학습 스타일: {profile['learning_style']}")
            st.info(f"지식 수준: {profile['knowledge_level']}")

    # 메인 화면
    if profile:
        # 탭 구성
        tab1, tab2, tab3, tab4 = st.tabs(["📚 학습하기", "📊 분석", "🎯 학습 경로", "🏆 성과"])

        with tab1:
            st.title("📚 개인화된 학습")

            # 개념 선택
            concept = st.selectbox("학습할 개념 선택", [
                "programming", "variables", "functions", "data_structures",
                "machine_learning", "statistics"
            ])

            if st.button("레슨 시작"):
                with st.spinner("개인화된 레슨 생성 중..."):
                    lesson = assistant.generate_personalized_lesson(user_id, concept)

                if lesson["status"] == "prerequisites_needed":
                    st.warning("선수 지식이 필요합니다.")
                    st.write("부족한 개념:", lesson["missing_prerequisites"])

                    if st.button("선수 과정 시작"):
                        for prereq in lesson["missing_prerequisites"]:
                            prereq_lesson = assistant.generate_personalized_lesson(user_id, prereq)
                            st.write(f"## {prereq}")
                            st.write(prereq_lesson["lesson_content"]["content"])

                else:
                    # 레슨 표시
                    st.write(f"## {concept} 학습")
                    st.write(lesson["lesson_content"]["content"])

                    # 연습 문제
                    if lesson["practice_problems"]:
                        st.write("### 연습 문제")

                        answers = []
                        for i, problem in enumerate(lesson["practice_problems"]):
                            st.write(f"**문제 {i+1}:**")
                            st.write(problem["question"])

                            if problem["type"] == "multiple_choice":
                                answer = st.radio(
                                    f"선택하세요 (문제 {i+1})",
                                    problem["choices"],
                                    key=f"q_{i}"
                                )
                                answers.append({"question_id": i, "answer": answer})

                            elif problem["type"] == "short_answer":
                                answer = st.text_input(
                                    f"답변을 입력하세요 (문제 {i+1})",
                                    key=f"q_{i}"
                                )
                                answers.append({"question_id": i, "answer": answer})

                        if st.button("답안 제출"):
                            time_spent = 30  # 실제로는 타이머 구현 필요

                            with st.spinner("답안 채점 중..."):
                                evaluation = assistant.evaluate_learning_session(
                                    lesson["session_id"], answers, time_spent
                                )

                            # 결과 표시
                            st.write("### 학습 결과")

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("점수", f"{evaluation['score']:.1f}%")
                            with col2:
                                st.metric("정답", f"{evaluation['correct_answers']}/{evaluation['total_questions']}")
                            with col3:
                                st.metric("소요 시간", f"{evaluation['time_spent_minutes']}분")

                            # 피드백
                            st.write("### 개인화된 피드백")
                            st.write(evaluation["feedback"])

                            # 다음 학습 추천
                            st.write("### 다음 학습 추천")
                            for rec in evaluation["next_recommendations"]:
                                st.write(f"- {rec}")

        with tab2:
            st.title("📊 학습 분석")

            # 분석 기간 선택
            timeframe = st.selectbox("분석 기간", [
                "last_week", "last_month", "last_quarter"
            ])

            if st.button("분석 생성"):
                with st.spinner("학습 데이터 분석 중..."):
                    analytics = assistant.generate_learning_analytics(user_id, timeframe)

                # 요약 메트릭
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("총 학습 세션", analytics["summary"]["total_sessions"])
                with col2:
                    st.metric("총 학습 시간", f"{analytics['summary']['total_study_time']}분")
                with col3:
                    st.metric("평균 점수", f"{analytics['summary']['average_score']:.1f}%")
                with col4:
                    st.metric("향상률", f"{analytics['summary']['improvement_rate']:.1f}%")

                # 성과 트렌드
                if analytics["performance_trends"]:
                    st.subheader("성과 트렌드")
                    st.line_chart(analytics["performance_trends"])

                # 개념별 숙련도
                if analytics["concept_mastery"]:
                    st.subheader("개념별 숙련도")
                    st.bar_chart(analytics["concept_mastery"])

                # 학습 패턴
                st.subheader("학습 패턴")
                st.write(analytics["learning_patterns"])

                # 추천사항
                st.subheader("개선 추천사항")
                for rec in analytics["recommendations"]:
                    st.write(f"- {rec}")

        with tab3:
            st.title("🎯 개인화된 학습 경로")

            # 목표 설정
            target_concept = st.selectbox("학습 목표", [
                "machine_learning", "web_development", "data_science",
                "mobile_development", "devops"
            ])

            deadline = st.date_input("목표 완료 일자 (선택사항)")

            if st.button("학습 경로 생성"):
                with st.spinner("최적 학습 경로 계산 중..."):
                    learning_path = assistant.create_adaptive_learning_path(
                        user_id, target_concept, deadline
                    )

                # 경로 정보
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("총 개념 수", learning_path["total_concepts"])
                with col2:
                    st.metric("예상 기간", f"{learning_path['estimated_duration_weeks']}주")
                with col3:
                    st.metric("일일 학습 시간", f"{learning_path['daily_study_time_minutes']}분")

                # 성공 예측
                success_prob = learning_path["success_probability"]
                st.progress(success_prob)
                st.write(f"성공 확률: {success_prob:.1%}")

                # 학습 경로 시각화
                st.subheader("학습 경로")

                for i, concept in enumerate(learning_path["learning_path"]["concepts"], 1):
                    st.write(f"{i}. **{concept}** ({learning_path['learning_path']['durations'][i-1]}시간)")

                # 마일스톤
                st.subheader("마일스톤")

                for milestone in learning_path["milestones"]:
                    st.write(f"📅 {milestone['date']} - {milestone['title']}")
                    st.write(f"   목표: {milestone['goal']}")

        with tab4:
            st.title("🏆 학습 성과")

            # 배지 시스템 (예시)
            achievements = [
                {"name": "첫 걸음", "description": "첫 학습 세션 완료", "earned": True},
                {"name": "꾸준함", "description": "7일 연속 학습", "earned": False},
                {"name": "완벽주의자", "description": "100% 점수 달성", "earned": True},
                {"name": "탐험가", "description": "5개 이상 개념 학습", "earned": False}
            ]

            st.subheader("획득한 배지")

            cols = st.columns(4)
            for i, achievement in enumerate(achievements):
                with cols[i % 4]:
                    if achievement["earned"]:
                        st.success(f"🏆 {achievement['name']}")
                    else:
                        st.info(f"🔒 {achievement['name']}")
                    st.caption(achievement["description"])

# 앱 실행
if __name__ == "__main__":
    create_learning_assistant_app()
```

## 🚀 클라우드 배포 예시

### AWS Lambda + API Gateway
```python
import json
import boto3
from mangum import Mangum
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="AI Assistant API")

# Lambda 핸들러
handler = Mangum(app)

class QueryRequest(BaseModel):
    question: str
    user_id: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    session_id: str
    confidence: float

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    try:
        # S3에서 벡터 스토어 로드
        s3_client = boto3.client('s3')
        vectorstore = load_vectorstore_from_s3(s3_client, "my-vectorstore-bucket")

        # RAG 체인 실행
        result = rag_chain.invoke({
            "input": request.question,
            "user_id": request.user_id
        })

        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            session_id=request.session_id or generate_session_id(),
            confidence=result.get("confidence", 0.8)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def load_vectorstore_from_s3(s3_client, bucket_name):
    # S3에서 벡터 스토어 데이터 로드하는 로직
    pass

# 배포 설정 (serverless.yml 예시)
"""
service: ai-assistant-api

provider:
  name: aws
  runtime: python3.9
  region: us-east-1
  timeout: 30
  memorySize: 1024
  environment:
    OPENAI_API_KEY: ${env:OPENAI_API_KEY}

functions:
  api:
    handler: main.handler
    events:
      - http:
          path: /{proxy+}
          method: ANY
          cors: true

plugins:
  - serverless-python-requirements
"""
```

이 예시 모음집을 통해 실제 기업 환경에서 활용할 수 있는 다양한 AI 시스템을 구축할 수 있습니다. 각 예시는 실제 운영 환경을 고려한 확장성, 보안성, 유지보수성을 갖추고 있습니다.