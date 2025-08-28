import re
from typing import List, Optional, Tuple

# ------------ 공통 정규식(헤더/학년/학기) ------------
HDR_RE = re.compile(r"^(#{1,6})\s+(.*)\s*$")
YEAR_RE = re.compile(r"([0-9])\s*학\s*년")
SEM_RE  = re.compile(r"([0-9])\s*학\s*기")
TERM_H4_RE = re.compile(r"([0-9])\s*학\s*년.*?([0-9])\s*학\s*기")

SEM_ORDER = {"1학기": 1, "2학기": 2}
YEAR_ORDER = {"1학년": 1, "2학년": 2, "3학년": 3, "4학년": 4}

def normalize_numbers(q: str) -> str:
    """3-2, 3학년-2학기, 3학년2학기 → '3학년 2학기'로 정규화."""
    def _fmt(y, s): return f"{y}학년 {s}학기 "
    q = re.sub(r"([0-9])\s*[-~]\s*([0-9])\s*학?\s*기?", lambda m: _fmt(m.group(1), m.group(2)), q or "")
    q = re.sub(r"([0-9])\s*학\s*년\s*[-~]\s*([0-9])\s*학\s*기", lambda m: _fmt(m.group(1), m.group(2)), q)
    q = re.sub(r"([0-9])\s*학\s*년\s*([0-9])\s*학\s*기", lambda m: _fmt(m.group(1), m.group(2)), q)
    q = re.sub(r"(학기)(?=\S)", r"\1 ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q

def parse_year_semester(text: str) -> Tuple[Optional[str], Optional[str]]:
    y = YEAR_RE.search(text or "")
    s = SEM_RE.search(text or "")
    return (f"{y.group(1)}학년" if y else None, f"{s.group(1)}학기" if s else None)

def looks_like_term_header(title: str) -> bool:
    return bool(TERM_H4_RE.search(title or "") or (YEAR_RE.search(title or "") and SEM_RE.search(title or "")))

def detect_year_semester_in_query(q: str) -> Tuple[Optional[str], Optional[str]]:
    qn = normalize_numbers(q or "")
    y = YEAR_RE.search(qn)
    s = SEM_RE.search(qn)
    return (f"{y.group(1)}학년" if y else None, f"{s.group(1)}학기" if s else None)

def term_sort_key(year: Optional[str], semester: Optional[str]) -> Tuple[int, int]:
    return (YEAR_ORDER.get(year or "", 99), SEM_ORDER.get(semester or "", 99))

# ---------------- BM25용 토크나이저 ----------------
_KIWI = None
try:
    from kiwipiepy import Kiwi
    _KIWI = Kiwi()
except Exception:
    _KIWI = None

def _ngrams(s: str, n: int) -> List[str]:
    return [s[i:i+n] for i in range(len(s)-n+1)] if len(s) >= n else []

def tokenize_ko(text: str) -> List[str]:
    import re as _re
    text = (text or "").lower()
    text = normalize_numbers(text)
    text = _re.sub(r"[·•/／\-\–—]", " ", text)
    text = _re.sub(r"[‘’“”‟‚‛′″´`]", "", text)
    if _KIWI:
        morphs = [t.form for t in _KIWI.tokenize(text) if t.form.strip()]
        out: List[str] = []
        for m in morphs:
            out.append(m)
            if _re.fullmatch(r"[가-힣]+", m):
                out.extend(_ngrams(m, 2)); out.extend(_ngrams(m, 3))
        return out
    segs = _re.findall(r"[가-힣]+|[a-z]+|\d+", text)
    out: List[str] = []
    for s in segs:
        out.append(s)
        if _re.fullmatch(r"[가-힣]+", s):
            out.extend(_ngrams(s, 2)); out.extend(_ngrams(s, 3))
    return out