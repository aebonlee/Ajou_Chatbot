#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/ingest_tree.py
- data/<college>/<dept>/**/*.md 구조를 순회하며 Markdown을 청크로 나눠 Chroma에 인덱싱합니다.
- 메타데이터: college, dept, major(전공/트랙명), section(H2), year/semester(H4에서 추출), is_micro(Y/N), program_type(major/micro), path
- 본 스크립트는 프로젝트 내부 모듈에 의존하지 않으며, 단독으로 실행 가능합니다.

필요 라이브러리:
pip install chromadb sentence-transformers rank-bm25 langchain-huggingface

예시 실행:
python scripts/ingest_tree.py \
  --root data \
  --persist storage/chroma-acad \
  --collection acad_docs \
  --embedding sentence-transformers/all-MiniLM-L6-v2
"""
import os
import re
import glob
import uuid
import argparse
from typing import List, Dict, Optional, Tuple

import chromadb
from chromadb.utils import embedding_functions

# -------------------------------
# 간단 유틸
# -------------------------------
HDR_RE = re.compile(r"^(#{1,6})\s+(.*)\s*$")
YEAR_RE = re.compile(r"([0-9])\s*학\s*년")
SEM_RE  = re.compile(r"([0-9])\s*학\s*기")
TERM_H4_RE = re.compile(r"([0-9])\s*학\s*년.*?([0-9])\s*학\s*기")

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def slice_block(lines: List[str], start: int, end: int) -> str:
    block = lines[start:end]
    while block and block[0].strip() == "":
        block.pop(0)
    return "\n".join(block).strip()

def find_headers(lines: List[str]) -> List[Tuple[int, int, str]]:
    headers = []
    for i, ln in enumerate(lines):
        m = HDR_RE.match(ln)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            headers.append((i, level, title))
    return headers

def looks_like_term_header(title: str) -> bool:
    return bool(TERM_H4_RE.search(title) or (YEAR_RE.search(title) and SEM_RE.search(title)))

def parse_year_semester(text: str) -> Tuple[Optional[str], Optional[str]]:
    y = YEAR_RE.search(text)
    s = SEM_RE.search(text)
    return (f"{y.group(1)}학년" if y else None, f"{s.group(1)}학기" if s else None)

def make_path(college: str, dept: str, major: Optional[str], sec=None, subsec=None, leaf=None) -> str:
    parts = [college, dept]
    if major:
        parts.append(major)
    if sec:
        parts.append(sec)
    if subsec:
        parts.append(subsec)
    if leaf:
        parts.append(leaf)
    return " > ".join([p for p in parts if p])

def compact(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()

def sanitize_meta(meta: Dict) -> Dict:
    """Chroma 메타는 None 불가 / 복합타입 문자열화."""
    out = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out

# -------------------------------
# 인덱싱 핵심
# -------------------------------
def chunk_markdown(md_text: str, college: str, dept: str, source_file: str) -> List[Dict]:
    """
    Markdown을 H1/H2/H4 기준으로 나눔.
    - H1: '...학과'면 학과공통, 아니면 전공/트랙/마이크로전공명
    - H2: 큰 섹션(졸업요건/교육과정 등)
    - H4: 'n학년 m학기'가 보이면 term 청크 생성
    """
    lines = md_text.splitlines()
    headers = find_headers(lines)
    chunks: List[Dict] = []

    h1_idxs = [i for i, lvl, _ in headers if lvl == 1]
    h1_bounds = []
    for idx, h1_i in enumerate(h1_idxs):
        start = h1_i
        end = h1_idxs[idx + 1] if idx + 1 < len(h1_idxs) else len(lines)
        h1_bounds.append((start, end))

    for h1_start, h1_end in h1_bounds:
        m = HDR_RE.match(lines[h1_start])
        h1_title = m.group(2).strip() if m else ""
        major = None if h1_title.endswith("학과") else h1_title

        is_micro = "마이크로전공" in (major or "")
        program_type = "micro" if is_micro else "major"

        local_headers = [(i, lvl, t) for i, lvl, t in headers if h1_start <= i < h1_end]
        h2_list = [(i, t) for i, lvl, t in local_headers if lvl == 2]

        if not h2_list:
            block = slice_block(lines, h1_start + 1, h1_end)
            path = make_path(college, dept, major)
            text = f"[PATH] {path}\n\n{compact(block)}"
            chunks.append({
                "id": f"doc_{uuid.uuid4().hex}",
                "text": text,
                "metadata": sanitize_meta({
                    "college": college, "dept": dept, "major": major or "학과공통",
                    "section": None, "year": None, "semester": None,
                    "chunk_type": "doc", "path": path, "source": os.path.basename(source_file),
                    "program_type": program_type, "is_micro": "Y" if is_micro else "N",
                })
            })
            continue

        for idx2, (h2_line, h2_title) in enumerate(h2_list):
            h2_start = h2_line
            h2_end = h2_list[idx2 + 1][0] if idx2 + 1 < len(h2_list) else h1_end
            big_block = slice_block(lines, h2_start + 1, h2_end)
            path = make_path(college, dept, major, h2_title)
            big_text = f"[PATH] {path}\n\n{compact(big_block)}"
            h2_id = f"sec_{uuid.uuid4().hex}"
            chunks.append({
                "id": h2_id,
                "text": big_text,
                "metadata": sanitize_meta({
                    "college": college, "dept": dept, "major": major or "학과공통",
                    "section": h2_title, "year": None, "semester": None,
                    "chunk_type": "sec", "path": path, "source": os.path.basename(source_file),
                    "program_type": program_type, "is_micro": "Y" if is_micro else "N",
                })
            })

            inner_headers = [(i, lvl, t) for i, lvl, t in local_headers if h2_start < i < h2_end]
            h4_list = [(i, t) for i, lvl, t in inner_headers if lvl == 4]
            for idx4, (h4_line, h4_title) in enumerate(h4_list):
                if not looks_like_term_header(h4_title):
                    continue
                leaf_start = h4_line
                leaf_end = h4_list[idx4 + 1][0] if idx4 + 1 < len(h4_list) else h2_end
                leaf_block = slice_block(lines, leaf_start + 1, leaf_end)
                year, sem = parse_year_semester(h4_title)
                path = make_path(college, dept, major, h2_title, None, h4_title)
                small_text = f"[PATH] {path}\n\n{compact(leaf_block)}"
                chunks.append({
                    "id": f"term_{uuid.uuid4().hex}",
                    "text": small_text,
                    "metadata": sanitize_meta({
                        "college": college, "dept": dept, "major": major or "학과공통",
                        "section": h2_title, "year": year, "semester": sem,
                        "chunk_type": "term", "path": path, "source": os.path.basename(source_file),
                        "program_type": program_type, "is_micro": "Y" if is_micro else "N",
                    })
                })
    return chunks

def index_tree(root: str, persist_dir: str, collection: str, embedding_model: str) -> int:
    """
    data/<college>/<dept>/**/*.md 트리를 읽어 Chroma에 색인합니다.
    """
    client = chromadb.PersistentClient(path=persist_dir)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)
    try:
        col = client.get_collection(name=collection)
        # attach EF (query_texts 사용 가능하게)
        col._embedding_function = ef  # type: ignore[attr-defined]
    except Exception:
        col = client.create_collection(name=collection, embedding_function=ef, metadata={"hnsw:space": "cosine"})

    total = 0
    for college in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
        cdir = os.path.join(root, college)
        for dept in sorted(d for d in os.listdir(cdir) if os.path.isdir(os.path.join(cdir, d))):
            ddir = os.path.join(cdir, dept)
            files = sorted(glob.glob(os.path.join(ddir, "**", "*.md"), recursive=True))
            if not files:
                continue
            print(f"[INGEST] {college} / {dept} ({len(files)} files)")
            batch_ids, batch_docs, batch_meta = [], [], []
            def _flush():
                nonlocal batch_ids, batch_docs, batch_meta, total
                if not batch_ids:
                    return
                col.add(ids=batch_ids, documents=batch_docs, metadatas=batch_meta)
                total += len(batch_ids)
                batch_ids, batch_docs, batch_meta = [], [], []

            for fp in files:
                md = read_text(fp)
                for ch in chunk_markdown(md, college, dept, fp):
                    batch_ids.append(ch["id"])
                    batch_docs.append(ch["text"])
                    batch_meta.append(ch["metadata"])
                    if len(batch_ids) >= 128:
                        _flush()
            _flush()
    print(f"Indexed {total} chunks into '{collection}' at '{persist_dir}'.")
    return total

# -------------------------------
# CLI
# -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="data root (e.g., data)")
    ap.add_argument("--persist", required=True, help="Chroma persist dir")
    ap.add_argument("--collection", required=True, help="Chroma collection name")
    ap.add_argument("--embedding", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()
    index_tree(args.root, args.persist, args.collection, args.embedding)

if __name__ == "__main__":
    main()