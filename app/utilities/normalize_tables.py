from __future__ import annotations
import re, hashlib
from typing import List, Dict, Any, Tuple
import pandas as pd

# ---------- 공통 유틸 ----------
def _nz(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()

def _sig_header(df: pd.DataFrame, n: int = 2) -> str:
    """상위 n개 헤더행을 이어붙여 시그니처 생성 (롱테이블 병합용)"""
    n = min(n, len(df))
    head = ["|".join(_nz(x) for x in df.iloc[i].tolist()) for i in range(n)]
    sig = " / ".join(head)
    sig = re.sub(r"\s+", " ", sig)
    return hashlib.md5(sig.encode("utf-8")).hexdigest()

def _is_bullet(x: Any) -> bool:
    s = str(x or "").strip()
    return s in {"●", "•", "●표시", "● 표시"} or bool(re.fullmatch(r"[●•]", s))

def _to_num(x: Any):
    s = _nz(x).replace(",", "")
    if not s: return None
    try:
        if "." in s: return float(s)
        return int(s)
    except: return s

# ---------- 테이블 타입 판별 ----------
def detect_table_type(df: pd.DataFrame) -> str:
    cols = [ _nz(c) for c in df.columns ]
    joined = " ".join(cols)
    # 권장이수표: 중앙 '이수구분' + 좌우 1/2학기 세트
    if "이수구분" in joined and cols.count("과목명")>=1 and ("1학기" in joined and "2학기" in joined):
        return "plan"       # 권장 이수 순서표
    # 교육과정표: "개설 학년 및 학기" + "학점구성(이론/설계/실험실습/학점 합계)"
    if ("개설" in joined and "학년" in joined and "학기" in joined) and ("학점구성" in joined or "이론" in joined):
        return "curriculum" # 교육과정
    return "generic"

# ---------- 롱테이블 병합 ----------
def stitch_long_tables(dfs: List[Tuple[int, int, pd.DataFrame]]):
    """
    dfs: [(page, area_idx, df), ...]  # 같은 페이지 내 여러 area(좌/우 컬럼)에서 뽑힌 DF들
    헤더 시그니처와 컬럼 개수/순서를 기준으로 연속된 DF를 병합.
    """
    grouped: Dict[Tuple[str, int], List[Tuple[int,int,pd.DataFrame]]] = {}
    for (page, aidx, df) in dfs:
        # 상단 1~2행이 헤더인 경우가 많으므로 그대로 시그니처로 사용
        sig = (_sig_header(df, n=min(2, len(df))), len(df.columns))
        grouped.setdefault(sig, []).append((page, aidx, df))

    stitched = []
    for sig, items in grouped.items():
        items.sort(key=lambda x: (x[0], x[1]))  # page→area 순
        # 첫 DF의 헤더를 유지하고, 이후 DF는 body만 붙임
        base_page, base_aidx, base_df = items[0]
        header = base_df.iloc[0:1] if len(base_df)>0 else base_df.head(0)
        body_list = [ base_df.iloc[1:] ]  # 1행 헤더 가정
        for (pg, ai, df) in items[1:]:
            body_list.append(df.iloc[1:])
        merged = pd.concat([header]+body_list, axis=0, ignore_index=True)
        stitched.append((base_page, items[-1][0], merged))  # (start_page, end_page, df)
    return stitched

# ---------- 권장이수표 정규화 ----------
def normalize_plan_table(df_raw: pd.DataFrame, meta: Dict[str,Any]) -> List[Dict[str,Any]]:
    """
    중앙 '이수구분'이 좌/우 1/2학기 모두에 적용되는 권장이수표를
    학기 단위 레코드로 펴기.
    """
    df = df_raw.copy().fillna("")
    # 컬럼명 정리 (중복 접미사 표준화)
    cols = [ _nz(c) for c in df.columns ]
    df.columns = cols

    # 좌/우 매핑 자동 추론
    base_keys = ["과목명","학점","시간","선수과목","외국어강의여부"]
    left_map, right_map = {}, {}
    for k in base_keys:
        # 왼쪽은 첫 등장, 오른쪽은 그 다음 등장
        idxs = [i for i,c in enumerate(cols) if c.startswith(k)]
        left_map[k]  = cols[idxs[0]] if idxs else None
        right_map[k] = cols[idxs[1]] if len(idxs)>1 else None

    mid_col = next((c for c in cols if "이수구분" in c), None)

    out = []
    for ridx in range(1, len(df)):  # 0행: 헤더
        row = df.iloc[ridx]
        # 중앙 이수구분 ffill(병합셀 보정)
        isugubun = _nz(row.get(mid_col))
        if not isugubun and ridx>1:
            isugubun = _nz(df.iloc[ridx-1].get(mid_col))

        def make(side: str, semester: int):
            m = left_map if side=="L" else right_map
            if not m.get("과목명"): return None
            subj = _nz(row.get(m["과목명"]))
            if subj in {"", "-", "계"}: return None
            rec = {
                "학년": None,            # 원 표에 별도 열이 있으면 후처리에서 보강
                "학기": semester,         # 1 or 2
                "과목명": subj,
                "학점": _to_num(row.get(m["학점"])),
                "시간": _to_num(row.get(m["시간"])),
                "선수과목": _nz(row.get(m["선수과목"])),
                "외국어강의여부": _nz(row.get(m["외국어강의여부"])),
                "이수구분": isugubun,
                "metadata": {**meta}
            }
            return rec

        L = make("L", 1)
        if L: out.append(L)
        R = make("R", 2)
        if R: out.append(R)
    return out

# ---------- 교육과정표 정규화 ----------
def normalize_curriculum_table(df_raw: pd.DataFrame, meta: Dict[str,Any]) -> List[Dict[str,Any]]:
    """
    '개설 학년 및 학기(●)' + '학점구성(이론/설계/실험실습/학점 합계)' 구조를
    학기 단위 레코드로 변환.
    """
    df = df_raw.copy().fillna("")
    cols = [ _nz(c) for c in df.columns ]
    df.columns = cols

    # 좌측 블록(이수구분/학수구분/과목명), 중간(학년·학기), 우측(학점구성) 구간 추정
    # 간단히 키워드 위치 기반으로 분할
    # 중간: 1학년~4학년 * 1/2학기
    mid_idxs = [i for i,c in enumerate(cols) if re.search(r"(1|2|3|4)\s*학년", c) or c in {"1학기","2학기"}]
    right_idxs = [i for i,c in enumerate(cols) if c in {"이론","설계","실험실습","학점합계","학점 수합계","학점수합계"}]
    left_end = min(mid_idxs) if mid_idxs else (len(cols)-len(right_idxs))
    right_start = min(right_idxs) if right_idxs else len(cols)

    left_cols = cols[:left_end]
    mid_cols  = cols[left_end:right_start]
    right_cols= cols[right_start:]

    def parse_left(row):
        return {
            "이수구분": _nz(row.get(next((c for c in left_cols if "이수구분" in c), ""))),
            "학수구분": _nz(row.get(next((c for c in left_cols if "학수구분" in c), ""))),
            "과목명":   _nz(row.get(next((c for c in left_cols if "과목명"   in c), ""))),
        }

    def parse_right(row):
        return {
            "이론": _to_num(row.get(next((c for c in right_cols if "이론" in c), ""))),
            "설계": _to_num(row.get(next((c for c in right_cols if "설계" in c), ""))),
            "실험실습": _to_num(row.get(next((c for c in right_cols if "실험" in c), ""))),
            "학점합계": _to_num(row.get(next((c for c in right_cols if "합계" in c), ""))),
        }

    # 학년/학기 열 식별 (● 마크가 들어가는 곳)
    sem_keys = []
    for c in mid_cols:
        if re.search(r"(1|2|3|4)\s*학년", c) or c in {"1학기","2학기"} or re.search(r"학기", c):
            sem_keys.append(c)

    out = []
    for ridx in range(1, len(df)):  # 0행 헤더
        row = df.iloc[ridx]
        left = parse_left(row)
        right = parse_right(row)

        # 무의미/합계 행 스킵
        subj = left["과목명"]
        if subj in {"", "-", "계"}:
            continue

        # 학기 마크를 읽어 각각 레코드 생성
        for sk in sem_keys:
            if _is_bullet(row.get(sk)):
                # 학년/학기 파싱
                y = re.search(r"([1-4])\s*학년", sk)
                s = 1 if "1학기" in sk else (2 if "2학기" in sk else None)
                rec = {
                    "과목명": subj,
                    "학년": int(y.group(1)) if y else None,
                    "학기": s,
                    "이수구분": left["이수구분"] or left["학수구분"],
                    "학수구분": left["학수구분"],
                    "이론": right["이론"],
                    "설계": right["설계"],
                    "실험실습": right["실험실습"],
                    "학점합계": right["학점합계"],
                    "metadata": {**meta, "semester_col": sk}
                }
                out.append(rec)
    return out

# ---------- generic ----------
def normalize_generic_rows(df_raw: pd.DataFrame, meta: Dict[str,Any]) -> List[Dict[str,Any]]:
    df = df_raw.copy().fillna("")
    header = [ _nz(c) for c in df.columns ]
    out = []
    for ridx in range(1, len(df)):
        row = df.iloc[ridx]
        if not any(_nz(row.get(c)) for c in header):
            continue
        rec = { h: _nz(row.get(h)) for h in header }
        out.append({
            "content": "; ".join(f"{k}={v}" for k,v in rec.items() if _nz(v)),
            "structured": rec,
            "metadata": meta
        })
    return out