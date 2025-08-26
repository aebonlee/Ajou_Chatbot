# app/services/storage_inspect.py
"""
Chroma 스토리지 상태 점검 유틸(임시).
- 전체/필터(dept/college/텍스트) 집계
- dept/college별 건수 랭킹
- 샘플 PATH/본문 미리보기
- 옵션: 간단 쿼리 테스트

사용 예)
  python -m app.services.storage_inspect --persist storage/chroma-acad
  python -m app.services.storage_inspect --persist storage/chroma-acad --collection acad_docs_bge_m3_clean --dept 사이버보안학과
  python -m app.services.storage_inspect --persist storage/chroma-acad --grep "졸업요건"
  python -m app.services.storage_inspect --persist storage/chroma-acad --query "사보 졸업" --topk 5
"""
from __future__ import annotations
import argparse
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

# 내부 chroma 래퍼 사용
from app.services.storage import get_client, get_collection, get_all

DEFAULT_PERSIST = os.getenv("PERSIST_DIR", "storage/chroma-acad")
DEFAULT_COLLECTION = os.getenv("COLLECTION", "acad_docs_bge_m3_clean")
DEFAULT_EMBEDDING = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")


def _print_kv(title: str, kv: Dict[str, int], topn: int = 20) -> None:
    print(f"\n== {title} (top {topn}) ==")
    for k, v in sorted(kv.items(), key=lambda x: x[1], reverse=True)[:topn]:
        print(f"{v:6d}  {k}")


def _preview(items: List[Tuple[str, str, Dict[str, Any]]], n: int = 5) -> None:
    print(f"\n== 샘플 {min(n, len(items))}개 미리보기 ==")
    for i, (doc, _id, meta) in enumerate(items[:n], 1):
        path = (meta or {}).get("path") or "(path 없음)"
        print(f"\n[{i}] id={_id}")
        print(f"PATH: {path}")
        body = doc
        if body.startswith("[PATH]") and "\n" in body:
            body = body.split("\n", 1)[1]
        body = body.strip()
        if len(body) > 400:
            body = body[:400] + " …"
        print(body or "(본문 없음)")


def _filter_mask(
    docs: List[str],
    metas: List[Dict[str, Any]],
    *,
    college: str | None,
    dept: str | None,
    grep: str | None,
) -> List[int]:
    idxs = []
    for i, (d, m) in enumerate(zip(docs, metas)):
        if college and (m or {}).get("college") != college:
            continue
        if dept and (m or {}).get("dept") != dept:
            continue
        if grep:
            txt = (d or "") + "\n" + str(m or "")
            if grep not in txt:
                continue
        idxs.append(i)
    return idxs


def _simple_query(col, all_ids: List[str], q: str, n: int = 5, where: Dict[str, Any] | None = None) -> List[Tuple[int, float]]:
    """Dense-only 간단 질의(embedding_function 부착 가정)."""
    try:
        res = col.query(
            query_texts=[q],
            n_results=n,
            where=where or None,  # ✅ 필터 전달
            include=["distances", "metadatas", "documents"],  # ✅ "ids" 제거
        )
    except Exception as e:
        print(f"[WARN] dense query 실패: {e}")
        return []

    # ids는 include에 넣지 않아도 항상 반환됨
    ids = res.get("ids", [[]])[0] if res else []
    dists = res.get("distances", [[]])[0] if res else []
    out: List[Tuple[int, float]] = []
    for cid, dist in zip(ids, dists):
        try:
            gi = all_ids.index(cid)  # all_ids는 이미 필터링/limit 반영된 배열
        except ValueError:
            continue
        sim = 1.0 - float(dist)
        out.append((gi, sim))
    return out

def main():
    ap = argparse.ArgumentParser(description="Chroma 저장소 점검")
    ap.add_argument("--persist", default=DEFAULT_PERSIST, help="Persistent dir (기본: %(default)s)")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION, help="Collection name (기본: %(default)s)")
    ap.add_argument("--embedding", default=DEFAULT_EMBEDDING, help="Embedding model (기본: %(default)s)")
    ap.add_argument("--college", help="필터: 단과대")
    ap.add_argument("--dept", help="필터: 학과")
    ap.add_argument("--grep", help="본문/메타에 포함되는 텍스트 필터")
    ap.add_argument("--limit", type=int, default=0, help="상위 N개만 로직에 사용(0=전체)")
    ap.add_argument("--preview", type=int, default=5, help="샘플 미리보기 개수")
    ap.add_argument("--query", help="간단 dense 쿼리 테스트 문장")
    ap.add_argument("--topk", type=int, default=5, help="--query 시 상위 k")
    args = ap.parse_args()

    print("== 설정 ==")
    print(f"persist   : {args.persist}")
    print(f"collection: {args.collection}")
    print(f"embedding : {args.embedding}")

    client = get_client(args.persist)
    col = get_collection(client, args.collection, args.embedding)
    all_ids, all_docs, all_metas = get_all(col)

    total = len(all_ids)
    print(f"\n총 청크 수: {total}")
    if total == 0:
        print("데이터가 없습니다. 인덱싱 여부를 먼저 확인하세요.")
        return

    # 선택적 제한
    if args.limit and args.limit > 0:
        all_ids = all_ids[:args.limit]
        all_docs = all_docs[:args.limit]
        all_metas = all_metas[:args.limit]
        print(f"(limit 적용) 검사 대상: {len(all_ids)}")

    # 필터 마스크
    mask = _filter_mask(all_docs, all_metas, college=args.college, dept=args.dept, grep=args.grep)
    if mask and len(mask) != len(all_ids):
        all_ids = [all_ids[i] for i in mask]
        all_docs = [all_docs[i] for i in mask]
        all_metas = [all_metas[i] for i in mask]
        print(f"필터 적용 후 청크 수: {len(all_ids)}")

    # 집계
    by_college = Counter((m or {}).get("college", "(없음)") for m in all_metas)
    by_dept = Counter((m or {}).get("dept", "(없음)") for m in all_metas)
    by_program = Counter((m or {}).get("program_type", "(없음)") for m in all_metas)
    by_is_micro = Counter((m or {}).get("is_micro", "(없음)") for m in all_metas)
    by_section = Counter((m or {}).get("section", "(없음)") for m in all_metas)

    _print_kv("단과대별 청크 수", dict(by_college))
    _print_kv("학과별 청크 수", dict(by_dept))
    _print_kv("program_type(major/micro)", dict(by_program))
    _print_kv("is_micro(Y/N)", dict(by_is_micro))
    _print_kv("section 상위", dict(by_section), topn=15)

    # PATH 빈도
    path_count = Counter((m or {}).get("path", "(없음)") for m in all_metas)
    _print_kv("PATH 상위", dict(path_count), topn=10)

    # 샘플 프리뷰
    _preview(list(zip(all_docs, all_ids, all_metas)), n=max(1, args.preview))

    # 간단 dense 질의 테스트
    if args.query:
        print(f"\n== Dense 질의 테스트: “{args.query}” (top {args.topk}) ==")
        pairs = _simple_query(col, all_ids, args.query, n=args.topk)
        if not pairs:
            print("(결과 없음)")
        else:
            for rank, (gi, sim) in enumerate(pairs, 1):
                meta = all_metas[gi] or {}
                path = meta.get("path") or "(path 없음)"
                print(f"{rank:2d}. sim={sim:.4f}  id={all_ids[gi]}")
                print(f"    PATH: {path}")

if __name__ == "__main__":
    main()