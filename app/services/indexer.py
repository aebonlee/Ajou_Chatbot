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
from bs4 import BeautifulSoup, Tag
from langchain_core.documents import Document
from app.core.config import PDF_FILES

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


# --------------------------------------------
# 학사공통
# -------------------------------------------

def chunk_markdown_file(file_path: str, source_metadata: Dict[str, Any] = None) -> List[Document]:
    """
    Markdown/HTML 파일을 페이지별로 청킹하고, ##를 기준으로 나누되,
    페이지 경계에서 헤더 계층을 올바르게 전파하여 LangChain Document 리스트를 반환합니다.
    """
    if source_metadata is None:
        source_metadata = {}

    EXCLUDED_H1_HEADERS = {"학칙", "학사력", "대학생활안내", "대학생활", "총람"}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            full_content = f.read()
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다 - {file_path}")
        return []

    pages = full_content.split('---')
    all_chunks = []

    header_context = {"h1": "", "h2": "", "h3": "", "h4": ""}

    for i, page_content in enumerate(pages):
        page_number = i + 1
        if not page_content.strip():
            continue

        # 1. 정제
        cleaned_content = re.sub(r'2025 아주대학교 요람|AJOU UNIVERSITY', '', page_content)
        cleaned_content = re.sub(r'【?\[?\s*6\..*?\]?】?', '', cleaned_content)
        cleaned_content = re.sub(r'(?:아주대학교\s*\d+|\d+\s*AJOU UNIV\.?|AU \d+ .*)', '', cleaned_content)
        cleaned_content = re.sub(r'^\s*AU\s*$', '', cleaned_content, flags=re.MULTILINE)

        # 2. 테이블 변환
        soup = BeautifulSoup(f"<div>{cleaned_content}</div>", 'lxml')
        for table in soup.find_all('table'):
            table_text = _convert_table_to_text(table)
            table.replace_with(table_text)
        full_text_on_page = soup.get_text(separator='\n')

        # 3. 내용 없는 헤더 최종 삭제
        full_text_on_page = re.sub(r'^#+\s*$', '', full_text_on_page, flags=re.MULTILINE).strip()
        if not full_text_on_page:
            continue

        # 4. ## 기준으로 청크 분할
        text_blocks = re.split(r'(?=\n##\s)', full_text_on_page)

        for block_idx, block in enumerate(text_blocks):
            block = block.strip()
            if not block or len(block.split()) < 3:
                continue

            if "학 위 기" in block:
                continue

            final_content = block

            # 5. 고아 청크 처리
            if block_idx == 0 and not block.startswith('##'):
                prefixes = []
                path_parts = [v for k, v in sorted(header_context.items()) if v]
                if path_parts:
                    prefixes.append(f"[이전 문서 경로: {' > '.join(path_parts)}]")
                if prefixes:
                    final_content = "\n".join(prefixes) + f"\n\n{block}"

            # 6. 표준 청크 처리
            elif header_context.get("h1"):
                final_content = f"[대주제: {header_context['h1']}]\n\n{block}"

            metadata = {
                **source_metadata,
                'page': page_number,
                'section_title': block.split('\n')[0].strip().replace('## ', '')
            }
            all_chunks.append(Document(page_content=final_content, metadata=metadata))

        # 7. 다음 페이지를 위해 현재 페이지의 최종 헤더 계층 업데이트
        headers_on_page = re.findall(r'(#+)\s([^\n]+)', cleaned_content)
        for hashes, text in headers_on_page:
            level = len(hashes)
            header_key = f'h{level}'
            if header_key in header_context:
                if level == 1:
                    if text.strip() in EXCLUDED_H1_HEADERS:
                        header_context = {k: "" for k in header_context}
                    else:
                        header_context['h1'] = text.strip()
                        for l in range(2, 5): header_context[f'h{l}'] = ""
                else:
                    header_context[header_key] = text.strip()
                    for l in range(level + 1, 5): header_context[f'h{l}'] = ""

    return all_chunks


def _convert_table_to_text(table_tag: Tag) -> str:
    """
    모든 종류의 HTML 테이블을 안정적으로 자연어 텍스트로 변환합니다.
    - rowspan에 영향을 받지 않는 방식으로 텍스트화
    """
    thead = table_tag.find('thead')
    header_rows = thead.find_all('tr') if thead else []

    # thead가 없다면 테이블의 첫번째 tr을 헤더로 간주할 수 있음
    if not header_rows and not thead:
        first_row = table_tag.find('tr')
        if first_row and first_row.find('th'):
            header_rows = [first_row]

    header_matrix = [[None] * 50 for _ in range(len(header_rows))]
    for r_idx, row in enumerate(header_rows):
        cells = row.find_all(['th', 'td'])
        c_idx = 0
        for cell in cells:
            while header_matrix[r_idx][c_idx] is not None: c_idx += 1
            rowspan, colspan = int(cell.get('rowspan', 1)), int(cell.get('colspan', 1))
            text = ' '.join(cell.get_text(strip=True).split())
            for i in range(rowspan):
                for j in range(colspan):
                    if r_idx + i < len(header_rows) and c_idx + j < 50:
                        header_matrix[r_idx + i][c_idx + j] = text
            c_idx += colspan

    if not header_matrix:
        tr = table_tag.find('tr')
        num_cols = len(tr.find_all(['th', 'td'])) if tr else 0
        final_headers = [f"column_{i + 1}" for i in range(num_cols)]
    else:
        num_cols = 0
        for r in header_matrix:
            num_cols = max(num_cols, len(r))
        final_headers = [''] * num_cols
        for c in range(num_cols):
            header_parts = [header_matrix[r][c] for r in range(len(header_matrix)) if
                            c < len(header_matrix[r]) and header_matrix[r][c]]
            final_headers[c] = '_'.join(dict.fromkeys(header_parts))

    # tbody가 없어도 모든 tr을 대상으로 하되, 헤더로 사용된 행은 제외
    all_rows = table_tag.find_all('tr')
    body_rows = [row for row in all_rows if row not in header_rows]

    if not body_rows: return ""

    result_texts = ["\n[표 시작]"]
    row_context = {}  # rowspan 데이터를 기억하기 위한 변수

    for row in body_rows:
        cells = row.find_all('td')
        if not cells: continue

        row_data = [(' '.join(cell.get_text(strip=True).split()), int(cell.get('rowspan', 1))) for cell in cells]

        current_row_pairs = []
        cell_idx = 0
        col_idx = 0

        # rowspan으로 인해 누락된 셀 복원 및 데이터-헤더 매칭
        while col_idx < len(final_headers):
            if col_idx in row_context and row_context[col_idx]['remaining'] > 0:
                # 이전 행에서 병합된 셀 데이터를 가져옴
                current_row_pairs.append(f"{final_headers[col_idx]}: {row_context[col_idx]['text']}")
                row_context[col_idx]['remaining'] -= 1
                col_idx += 1
            elif cell_idx < len(row_data):
                text, rowspan = row_data[cell_idx]
                if rowspan > 1:
                    row_context[col_idx] = {'text': text, 'remaining': rowspan - 1}
                if text:
                    current_row_pairs.append(f"{final_headers[col_idx]}: {text}")
                cell_idx += 1
                col_idx += 1
            else:
                break

        if current_row_pairs:
            result_texts.append(", ".join(current_row_pairs))

    result_texts.append("[표 끝]\n")
    return "\n".join(result_texts)


def process_documents(pdf_paths: List[str]) -> List[Document]:
    """주어진 경로의 모든 문서를 읽고 청킹하여 Document 리스트로 반환합니다."""

    pdf_title_map = {
        "2025_rules.md": "2025년도_학칙_학생준칙_장학금_생활관_공학인증",
        "2025_overview.md": "2025년도_학사력_총람_기구표_구성_학교법인_대우학원",
        "2025_campus_life.md": "2025년도_대학생활_학사_장학_학생교류_복지_문화시설_고시준비_학생병사_예비군_민방위교육",
    }

    all_chunks = []
    for path in pdf_paths:
        file_name = os.path.basename(path)
        source_id = next((ctype.value for ctype, fpath in PDF_FILES.items() if fpath == path), "unknown")

        source_metadata = {
            "source": source_id,
            "title": pdf_title_map.get(file_name, file_name)  # title 맵에서 제목 조회
        }
        chunks = chunk_markdown_file(path, source_metadata)
        all_chunks.extend(chunks)
    return all_chunks