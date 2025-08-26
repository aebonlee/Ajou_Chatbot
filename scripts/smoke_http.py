#!/usr/bin/env python3
# Python requests 기반 스모크 (OS 무관)
# 사용법:
#   python scripts/smoke_http.py                 # 통과/실패만 표기
#   python scripts/smoke_http.py --verbose       # 각 케이스의 응답 요약/본문 출력
#   python scripts/smoke_http.py --verbose --save responses.jsonl  # 응답 로그 저장
#
# 환경변수:
#   HOST  (기본: http://127.0.0.1:8000)
#   MODEL (예: claude-3-5-sonnet-20240620)
#   DM_DEPT (기본: 디지털미디어학과)
#   DEBUG=true/false (기본: false)
#
import os
import json
import argparse
import requests
from datetime import datetime

HOST = os.getenv("HOST", "http://127.0.0.1:8000")
MODEL = os.getenv("MODEL", "")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
DM_DEPT = os.getenv("DM_DEPT", "디지털미디어학과")

def post_yoram(question, departments=None, use_llm=True, topk=8):
    payload = {
        "question": question,
        "departments": departments or [],
        "use_llm": use_llm,
        "topk": topk,
        "debug": DEBUG,
    }
    if MODEL:
        payload["model_name"] = MODEL
    r = requests.post(f"{HOST}/yoram", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

def assert_true(cond, msg):
    if not cond:
        raise AssertionError(msg)

def pretty(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2)

def print_case(title, req, resp, verbose=False):
    print(f"\n=== {title} ===")
    print(f"- Q: {req['question']}")
    if verbose:
        # 응답 요약
        ans = (resp.get("answer") or "")[:300].replace("\n", " ") + ("..." if (resp.get("answer") and len(resp["answer"]) > 300) else "")
        ctx_len = len(resp.get("context") or "")
        srcs = resp.get("sources") or []
        print(f"- ANSWER (preview): {ans}")
        print(f"- SOURCES: {len(srcs)} → {srcs[:3]}{' ...' if len(srcs)>3 else ''}")
        print(f"- CONTEXT length: {ctx_len}")
        if resp.get("error"):
            print(f"- ERROR: {resp['error']}")
        # 전체 응답
        print("\n--- Raw Response ---")
        print(pretty(resp))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="각 테스트의 응답 상세 출력")
    ap.add_argument("--save", type=str, default="", help="모든 응답을 JSONL로 저장할 파일 경로")
    args = ap.parse_args()

    save_path = args.save
    writer = None
    if save_path:
        writer = open(save_path, "a", encoding="utf-8")
        # 세션 헤더
        writer.write(json.dumps({"_session": datetime.now().isoformat(), "host": HOST, "model": MODEL}, ensure_ascii=False) + "\n")

    def record(title, req, resp):
        if writer:
            writer.write(json.dumps({"title": title, "request": req, "response": resp}, ensure_ascii=False) + "\n")

    # 0) /health
    r = requests.get(f"{HOST}/health", timeout=10).json()
    assert_true(r.get("ok") is True, f"/health 실패: {r}")

    # 1) track_rules 고정
    req = {"question": "복수전공 신청 어디서 해요?", "departments": [], "use_llm": False}
    resp = post_yoram(**req)
    assert_true("아주대학교 포탈" in (resp.get("answer") or ""), f"track_rules 실패: {resp}")
    print_case("track_rules (fixed)", req, resp, args.verbose)
    record("track_rules (fixed)", req, resp)

    # 2) practice_capstone 고정
    req = {"question": "이번 학기 캡스톤 신청 어떻게 하죠?", "departments": [], "use_llm": False}
    resp = post_yoram(**req)
    assert_true("학기 시작 전 사전 신청" in (resp.get("answer") or ""), f"practice_capstone 실패: {resp}")
    print_case("practice_capstone (fixed)", req, resp, args.verbose)
    record("practice_capstone (fixed)", req, resp)

    # 3) clarification
    req = {"question": "졸업요건 알려줘", "departments": [], "use_llm": False}
    resp = post_yoram(**req)
    assert_true(bool(resp.get("clarification")), f"clarification 실패: {resp}")
    print_case("clarification", req, resp, args.verbose)
    record("clarification", req, resp)

    # 4) micro_list (LLM)
    req = {"question": "디지털미디어학과 마이크로전공에는 뭐가 있어?", "departments": [DM_DEPT], "use_llm": True}
    resp = post_yoram(**req)
    ans = resp.get("answer") or ""
    assert_true(("마이크로전공" in ans) and ("출처:" in ans), f"micro_list 기본 체크 실패: {resp}")
    print_case("micro_list (LLM)", req, resp, args.verbose)
    record("micro_list (LLM)", req, resp)

    # 5) use_llm=false 폴백
    req = {"question": "디지털미디어학과 마이크로전공에는 뭐가 있어?", "departments": [DM_DEPT], "use_llm": False}
    resp = post_yoram(**req)
    assert_true("검색된 문서 요약" in (resp.get("answer") or ""), f"use_llm=false 폴백 실패: {resp}")
    print_case("micro_list (fallback)", req, resp, args.verbose)
    record("micro_list (fallback)", req, resp)

    print("\n✅ 모든 스모크 테스트 통과!")
    if writer:
        writer.close()
        print(f"💾 저장됨: {save_path}")

if __name__ == "__main__":
    main()