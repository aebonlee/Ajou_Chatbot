import os
import pytest
from fastapi.testclient import TestClient

# ==== 공용 픽스처 ====
@pytest.fixture(scope="session", autouse=True)
def _env():
    os.environ.setdefault("LLM_MODEL", "gpt-4o-mini")
    os.environ.setdefault("TEMPERATURE", "0.0")
    os.environ.setdefault("MAX_TOKENS", "512")
    yield

@pytest.fixture()
def client():
    from app.api.server import app
    return TestClient(app)


# ==== 유틸 LLM 스텁 ====
class _LLMResp:
    def __init__(self, content: str):
        self.content = content

class FakeLLMIncludeOnRetry:
    """1차에 항목 하나 누락 → 2차 재시도에서 전부 포함"""
    def __init__(self, **_):
        self.calls = 0
    def invoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            # 일부 누락(AR·VR)
            return _LLMResp(
                "디지털미디어학과의 마이크로전공은 다음과 같습니다:\n"
                "- 메타버스기획 마이크로전공\n"
                "- 디지털휴먼 마이크로전공\n"
                "\n출처: [SOURCE 1], [SOURCE 2], [SOURCE 3]"
            )
        # 재시도: 모두 포함
        return _LLMResp(
            "디지털미디어학과의 마이크로전공은 다음과 같습니다:\n"
            "- 메타버스기획 마이크로전공\n"
            "- 디지털휴먼 마이크로전공\n"
            "- 메타버스(AR·VR)지식재산 마이크로전공\n"
            "\n출처: [SOURCE 1], [SOURCE 2], [SOURCE 3]"
        )

class FakeLLMAlwaysMissing:
    """항상 하나 누락 → 결국 규칙기반 폴백으로 가야 함"""
    def __init__(self, **_): pass
    def invoke(self, messages):
        return _LLMResp(
            "디지털미디어학과의 마이크로전공은 다음과 같습니다:\n"
            "- 메타버스기획 마이크로전공\n"
            "- 디지털휴먼 마이크로전공\n"
            "\n출처: [SOURCE 1], [SOURCE 2]"
        )


# ==== 공용 더미 hits ====
def _hits_three_micro():
    return [
        {
            "id": "d1",
            "score": 0.9,
            "document": "[PATH] 소프트웨어융합대학 > 디지털미디어학과 > 메타버스기획 마이크로전공 > 졸업요건 및 교육과정 > 2학년 1학기\n* **메타버스콘텐츠기획:** 전공필수, 3학점 3시간.",
            "metadata": {"dept": "디지털미디어학과", "major": "메타버스기획 마이크로전공", "path": ""},
            "path": "소프트웨어융합대학 > 디지털미디어학과 > 메타버스기획 마이크로전공 > 졸업요건 및 교육과정 > 2학년 1학기",
        },
        {
            "id": "d2",
            "score": 0.85,
            "document": "[PATH] 소프트웨어융합대학 > 디지털미디어학과 > 디지털휴먼 마이크로전공 > 졸업요건 및 교육과정 > 3학년 1학기\n* **디지털휴먼파이프라인:** 전공선택, 3학점 3시간.",
            "metadata": {"dept": "디지털미디어학과", "major": "디지털휴먼 마이크로전공", "path": ""},
            "path": "소프트웨어융합대학 > 디지털미디어학과 > 디지털휴먼 마이크로전공 > 졸업요건 및 교육과정 > 3학년 1학기",
        },
        {
            "id": "d3",
            "score": 0.83,
            "document": "[PATH] 소프트웨어융합대학 > 디지털미디어학과 > 메타버스(AR·VR)지식재산 마이크로전공 > 졸업요건 및 교육과정 > 2학년 2학기\n* **지식재산과 연구개발:** 전공필수, 3학점 3시간.",
            "metadata": {"dept": "디지털미디어학과", "major": "메타버스(AR·VR)지식재산 마이크로전공", "path": ""},
            "path": "소프트웨어융합대학 > 디지털미디어학과 > 메타버스(AR·VR)지식재산 마이크로전공 > 졸업요건 및 교육과정 > 2학년 2학기",
        },
    ]


# ==== 테스트 1) /health ====
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("ok") is True


# ==== 테스트 2) 고정 라우트(포탈/캡스톤) ====
@pytest.mark.parametrize("q, expect", [
    ("복수전공 신청 어디서 하나요?", "아주대학교 포탈"),
    ("이번 학기 캡스톤 신청 어떻게 하죠?", "학기 시작 전 사전 신청"),
])
def test_fixed_routes(client, q, expect):
    r = client.post("/yoram", json={"question": q, "use_llm": False}).json()
    assert expect in r["answer"]
    assert r["context"] == ""
    assert r["sources"] == []


# ==== 테스트 3) clarification 필요 ====
def test_needs_clarification_when_no_dept(client):
    r = client.post("/yoram", json={"question": "졸업요건 알려줘", "use_llm": False}).json()
    assert r["clarification"] is not None
    assert r["context"] == ""


# ==== 테스트 4) must_include: 재시도 성공 ====
def test_micro_list_must_include_retry_success(client, monkeypatch):
    # retriever → 세 마이크로전공 모두 포함
    from app.services import retriever as R
    monkeypatch.setattr(R, "retrieve", lambda *a, **k: _hits_three_micro())
    # LLM → 1차 누락, 2차 포함
    import app.graphs.nodes as N
    monkeypatch.setattr(N, "_make_llm", lambda **kw: FakeLLMIncludeOnRetry())

    payload = {
        "question": "디지털미디어학과 마이크로전공에는 뭐가 있어?",
        "departments": ["디지털미디어학과"],
        "use_llm": True,
        "topk": 8,
    }
    r = client.post("/yoram", json=payload).json()
    ans = r["answer"]
    for name in ["메타버스기획 마이크로전공", "디지털휴먼 마이크로전공", "메타버스(AR·VR)지식재산 마이크로전공"]:
        assert name in ans
    assert "출처" in ans


# ==== 테스트 5) must_include: 항상 누락 → 규칙기반 폴백 ====
def test_micro_list_fallback_list_answer(client, monkeypatch):
    from app.services import retriever as R
    monkeypatch.setattr(R, "retrieve", lambda *a, **k: _hits_three_micro())
    import app.graphs.nodes as N
    monkeypatch.setattr(N, "_make_llm", lambda **kw: FakeLLMAlwaysMissing())

    r = client.post("/yoram", json={
        "question": "디지털미디어학과 마이크로전공에는 뭐가 있어?",
        "departments": ["디지털미디어학과"],
        "use_llm": True
    }).json()
    ans = r["answer"]
    for name in ["메타버스기획 마이크로전공", "디지털휴먼 마이크로전공", "메타버스(AR·VR)지식재산 마이크로전공"]:
        assert name in ans
    assert "출처" in ans


# ==== 테스트 6) use_llm=False 폴백 응답 형태 ====
def test_use_llm_false_returns_summary_and_sources(client, monkeypatch):
    from app.services import retriever as R
    monkeypatch.setattr(R, "retrieve", lambda *a, **k: [{
        "id": "x1",
        "score": 0.9,
        "document": "본문",
        "metadata": {"dept": "디지털미디어학과"},
        "path": "소프트웨어융합대학 > 디지털미디어학과 > 졸업요건 및 교육과정",
    }])
    r = client.post("/yoram", json={
        "question": "디지털미디어학과 마이크로전공에는 뭐가 있어?",
        "departments": ["디지털미디어학과"],
        "use_llm": False
    }).json()
    assert "검색된 문서 요약" in r["answer"]
    assert any("디지털미디어학과" in s for s in r["sources"])


# ==== 테스트 7) 카테고리 오버라이드가 적용되어도 에러 없이 흐름 유지 ====
def test_category_overrides_smoke(client, monkeypatch):
    # retrieve 결과 없어도, 파이프라인이 죽지 않고 응답
    from app.services import retriever as R
    monkeypatch.setattr(R, "retrieve", lambda *a, **k: [])
    r = client.post("/yoram", json={
        "question": "디지털미디어학과 마이크로전공에는 뭐가 있어?",
        "departments": ["디지털미디어학과"],
        "use_llm": False
    }).json()
    # 최소한 정상 응답이어야 하고, 마이크로 관련 질문이라 micro_mode가 include로 설정될 가능성 高
    assert r.get("error") is None
    assert r.get("answer") is not None