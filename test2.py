import os
from app.services.storage import get_client, get_collection
from app.core import config

# === 설정 ===
FILENAME = "2025학년도_공과대학_화학공학과.md"  # meta["source"] 값
PERSIST_DIR = config.PERSIST_DIR
COLLECTION   = config.COLLECTION
EMBED_MODEL  = config.EMBEDDING_MODEL
PRINT_LIMIT  = 2000   # 콘솔에 표시할 최대 글자수 (너무 길면 자름)
OUT_MD       = "chem_eng_chunks.md"

client = get_client(PERSIST_DIR)
col = get_collection(client, COLLECTION, EMBED_MODEL)

# chroma에서 해당 source만 긁어오기
res = col.get(
    where={"source": FILENAME},
    include=["documents", "metadatas", "embeddings"],
    limit=10000,  # 충분히 크게
    offset=0
)

ids    = res.get("ids", [])
docs   = res.get("documents", [])
metas  = res.get("metadatas", [])

# 정렬: order_key(없으면 'zzz') → id
def ord_key(meta, _id):
    ok = (meta or {}).get("order_key") or "zzz"
    return (ok, _id)

rows = sorted(zip(ids, docs, metas), key=lambda x: ord_key(x[2], x[0]))

# 콘솔 출력 + md 파일 저장
lines = []
for idx, (cid, doc, meta) in enumerate(rows, 1):
    ok = (meta or {}).get("order_key")
    path = (meta or {}).get("path")
    length = len(doc or "")
    header = f"{idx}. order_key={ok} len={length} path={path}"
    sep = "-" * 80
    print(header)
    print(sep)
    if not doc:
        print("[EMPTY DOC]\n")
        lines.append(header + "\n" + sep + "\n[EMPTY DOC]\n\n")
        continue

    # 콘솔엔 일부만
    print(doc[:PRINT_LIMIT])
    if length > PRINT_LIMIT:
        print(f"\n...[TRUNCATED {length-PRINT_LIMIT} chars]...\n")

    # 파일엔 전체
    lines.append(header + "\n" + sep + "\n" + doc + "\n\n")

# md로 덤프
with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write("# 화학공학과 청크 덤프 (원문 그대로)\n\n")
    f.writelines(lines)

print(f"\n=== DONE ===\n총 {len(rows)} chunks. 전체 내용은 '{OUT_MD}' 에 저장됨.")