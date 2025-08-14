def confidence_from_scores(scores: list[float]) -> float:
    if not scores: return 0.0
    s = sorted(scores, reverse=True)
    top = s[0]; second = s[1] if len(s)>1 else 0.0
    # 이 부분은 휴리스틱하게 결정하기
    return max(0.0, min(1.0, top - 0.4*max(0.0, top-second)))

def fallback_message(q: str) -> str:
    return "죄송하지만 설명에 참고할 만한 정보를 제가 갖고 있지 않아요. 질문을 이렇게 바꿔보실래요?\n- 더 구체적으로 학과/전공/마이크로전공 명시\n- 학년·학기 포함\n- 표/섹션 이름 포함"