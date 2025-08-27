"""
폴더 트리 인덱서:
- 입력: data/<college>/<dept>/*.md
- 처리: H1/H2/H4 단위로 청크 쪼개기, 메타필드 저장(클린 텍스트만 저장)
- 긴 섹션은 1200자 윈도우/200자 오버랩으로 분할
- (핵심) source_path + line_s/e + sec_seq/leaf_seq/chunk_idx + order_key 저장
  → 섹션을 "파일에서" 복원하고, 원문 순서대로 스티칭 가능
"""
import os, uuid, glob, re
from typing import List, Dict, Optional, Tuple, Any
from .storage import get_client, get_collection, add
from .textutil import HDR_RE, looks_like_term_header, parse_year_semester

# ------------------ 유틸 ------------------
def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _slice(lines: List[str], s: int, e: int) -> str:
    block = lines[s:e]
    while block and block[0].strip() == "":
        block.pop(0)
    return "\n".join(block).strip()

def _find_headers(lines: List[str]) -> List[Tuple[int, int, str]]:
    headers = []
    for i, ln in enumerate(lines):
        m = HDR_RE.match(ln)
        if m:
            headers.append((i, len(m.group(1)), m.group(2).strip()))
    return headers

def _make_path(college: str, dept: str, major: Optional[str], sec=None, leaf=None) -> str:
    parts = [college, dept]
    if major: parts.append(major)
    if sec: parts.append(sec)
    if leaf: parts.append(leaf)
    return " > ".join([p for p in parts if p])

def _dedup_adjacent(text: str) -> str:
    out, prev = [], None
    for ln in (text or "").splitlines():
        cur = ln.strip()
        if cur and cur == prev:
            continue
        out.append(ln)
        prev = cur
    return "\n".join(out)

def _compact(t: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", t or "").strip()

def _catalog_year_from_name(fname: str) -> Optional[str]:
    m = re.match(r"(\d{4})[_-]", fname or "")
    return f"{m.group(1)}학년도" if m else None

def _split_long_text(text: str, max_len: int = 1200, overlap: int = 200) -> List[str]:
    """
    문단 우선 누적 → 초과 시 글자 윈도우(오버랩 포함)
    """
    text = _compact(_dedup_adjacent(text or ""))
    if len(text) <= max_len:
        return [text]
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    cur = ""
    for p in paras:
        if len(cur) + len(p) + 2 <= max_len:
            cur = (cur + "\n\n" + p).strip() if cur else p
        else:
            if cur:
                chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)

    final: List[str] = []
    for c in chunks:
        if len(c) <= max_len:
            final.append(c)
        else:
            i = 0
            while i < len(c):
                final.append(c[i:i + max_len])
                if i + max_len >= len(c):
                    break
                i = max(0, i + max_len - overlap)
    return final

def _emit_chunk(
    chunks: List[Dict],
    *,
    base_id: str,
    path: str,
    meta: Dict,
    body: str,
    sec_seq: int,
    leaf_seq: int,
) -> None:
    """
    - 저장 문서는 '클린 텍스트'만 (문서 첫 줄에 [PATH] 절대 넣지 않음)
    - 순서 복원을 위해 chunk_idx와 order_key를 메타에 넣음
    """
    parts = _split_long_text(body, max_len=1200, overlap=200)
    for idx, part in enumerate(parts):
        order_key = f"{sec_seq:03d}.{leaf_seq:03d}.{idx:04d}"
        meta2 = {
            **meta,
            "path": path,
            "chunk_idx": idx,
            "sec_seq": sec_seq,
            "leaf_seq": leaf_seq,
            "order_key": order_key,
        }
        chunks.append({
            "id": f"{base_id}_{idx}" if len(parts) > 1 else base_id,
            "text": (part or "").strip(),   # 🟢 클린 텍스트
            "metadata": meta2,
        })

# ------------------ 청킹 ------------------
def _chunk_md(md_text: str, college: str, dept: str, source_file: str, catalog_year: Optional[str]) -> List[Dict]:
    lines = (md_text or "").splitlines()
    headers = _find_headers(lines)
    h1_idxs = [i for i, lvl, _ in headers if lvl == 1]
    bounds = [(h1, (h1_idxs[idx+1] if idx+1 < len(h1_idxs) else len(lines))) for idx, h1 in enumerate(h1_idxs)]

    chunks: List[Dict] = []
    for s, e in bounds:
        m = HDR_RE.match(lines[s])
        h1_title = m.group(2).strip() if m else ""
        major = None if h1_title.endswith("학과") else h1_title
        is_micro = "마이크로전공" in (major or "")
        program_type = "micro" if is_micro else "major"

        locals_ = [(i, lvl, t) for i, lvl, t in headers if s <= i < e]
        h2s = [(i, t) for i, lvl, t in locals_ if lvl == 2]

        # 섹션 시퀀스 카운터
        sec_seq = 0

        if not h2s:
            path = _make_path(college, dept, major)
            sec_id = f"doc_{uuid.uuid4().hex}"
            meta = {
                "college": college, "dept": dept, "major": major or "학과공통",
                "section": None, "year": None, "semester": None,
                "catalog_year": catalog_year, "chunk_type": "doc",
                "source": os.path.basename(source_file), "source_path": source_file,
                "parent_id": None, "section_id": sec_id,
                "line_s": s+1, "line_e": e,
                "program_type": program_type, "is_micro": "Y" if is_micro else "N",
            }
            body = _compact(_slice(lines, s+1, e))
            _emit_chunk(
                chunks, base_id=sec_id, path=path, meta=meta, body=body,
                sec_seq=sec_seq, leaf_seq=0
            )
            continue

        for idx, (h2_line, h2_title) in enumerate(h2s):
            sec_seq += 1
            h2_start, h2_end = h2_line, (h2s[idx+1][0] if idx+1 < len(h2s) else e)
            path = _make_path(college, dept, major, h2_title)
            sec_id = f"sec_{uuid.uuid4().hex}"
            meta_sec = {
                "college": college, "dept": dept, "major": major or "학과공통",
                "section": h2_title, "year": None, "semester": None,
                "catalog_year": catalog_year, "chunk_type": "sec",
                "source": os.path.basename(source_file), "source_path": source_file,
                "parent_id": None, "section_id": sec_id,
                "line_s": h2_start+1, "line_e": h2_end,
                "program_type": program_type, "is_micro": "Y" if is_micro else "N",
            }
            body_sec = _compact(_slice(lines, h2_start+1, h2_end))
            # leaf_seq=0 으로 섹션 본문 emit
            _emit_chunk(
                chunks, base_id=sec_id, path=path, meta=meta_sec, body=body_sec,
                sec_seq=sec_seq, leaf_seq=0
            )

            # 하위 학년/학기(H4)
            inner = [(i, lvl, t) for i, lvl, t in locals_ if h2_start < i < h2_end]
            h4s = [(i, t) for i, lvl, t in inner if lvl == 4]

            leaf_seq = 0
            for k, (h4_line, h4_title) in enumerate(h4s):
                if not looks_like_term_header(h4_title):
                    continue
                leaf_seq += 1
                leaf_s, leaf_e = h4_line, (h4s[k+1][0] if k+1 < len(h4s) else h2_end)
                year, sem = parse_year_semester(h4_title)
                path_leaf = _make_path(college, dept, major, h2_title, h4_title)
                term_id = f"term_{uuid.uuid4().hex}"
                meta_leaf = {
                    "college": college, "dept": dept, "major": major or "학과공통",
                    "section": h2_title, "year": year, "semester": sem,
                    "catalog_year": catalog_year, "chunk_type": "term",
                    "source": os.path.basename(source_file), "source_path": source_file,
                    "parent_id": sec_id, "section_id": term_id,
                    "line_s": leaf_s+1, "line_e": leaf_e,
                    "program_type": program_type, "is_micro": "Y" if is_micro else "N",
                }
                body_leaf = _compact(_slice(lines, leaf_s+1, leaf_e))
                _emit_chunk(
                    chunks, base_id=term_id, path=path_leaf, meta=meta_leaf, body=body_leaf,
                    sec_seq=sec_seq, leaf_seq=leaf_seq
                )
    return chunks

# ------------------ 엔트리 ------------------
def index_tree(root: str, persist_dir: str, collection: str, embedding_model: str) -> int:
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise FileNotFoundError(root)
    client = get_client(persist_dir)
    col = get_collection(client, collection, embedding_model)

    total = 0
    for college in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
        for dept in sorted(d for d in os.listdir(os.path.join(root, college)) if os.path.isdir(os.path.join(root, college, d))):
            md_glob = os.path.join(root, college, dept, "*.md")
            for md_path in glob.glob(md_glob):
                md = _read(md_path)
                cy = _catalog_year_from_name(os.path.basename(md_path))
                chunks = _chunk_md(md, college, dept, md_path, cy)
                if not chunks:
                    continue
                add(
                    col,
                    [c["id"] for c in chunks],
                    [c["text"] for c in chunks],
                    [c["metadata"] for c in chunks],
                )
                total += len(chunks)
                print(f"[INDEX] {college}/{dept}/{os.path.basename(md_path)} -> {len(chunks)} chunks")
    print(f"Done. Total chunks: {total}")
    return total