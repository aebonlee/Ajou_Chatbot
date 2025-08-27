from app.services.storage import get_client, get_collection, get_where_all, get_all
from app.core import config

print(f"PERSIST_DIR: {config.PERSIST_DIR}")
print(f"COLLECTION: {config.COLLECTION}")
print(f"EMBEDDING_MODEL: {config.EMBEDDING_MODEL}")

client = get_client(config.PERSIST_DIR)
CORRECT_COLLECTION = "acad_docs_bge_m3_clean"  # 인덱싱할 때 사용한 컬렉션

try:
    col = get_collection(client, CORRECT_COLLECTION, config.EMBEDDING_MODEL)

    # 디지털미디어학과 전체 데이터 가져오기
    where = {"dept": "디지털미디어학과"}
    ids, docs, metas = get_where_all(col, where)

    print(f"총 {len(docs)}개의 디지털미디어학과 청크를 찾았습니다.\n")

    # order_key 기준 정렬
    rows = sorted(zip(metas, docs), key=lambda md: md[0].get("order_key") or "zzz")

    OUT_MD = "dm_chunks_debug.md"
    PRINT_LIMIT = 1500  # 콘솔에 출력할 길이 제한

    lines = []
    for idx, (meta, doc) in enumerate(rows, 1):
        ok = meta.get("order_key")
        path = meta.get("path")
        src = meta.get("source")
        section_id = meta.get("section_id")
        chunk_type = meta.get("chunk_type")
        length = len(doc or "")

        header = f"{idx}. order_key={ok} type={chunk_type} len={length} path={path}"
        sep = "-" * 80

        print(header)
        print(sep)

        if not doc:
            print("[EMPTY DOC]\n")
            lines.append(header + "\n" + sep + "\n[EMPTY DOC]\n\n")
            continue

        # 콘솔 출력
        print(doc[:PRINT_LIMIT])
        if length > PRINT_LIMIT:
            print(f"\n...[TRUNCATED {length - PRINT_LIMIT} chars]...\n")
        print("")

        # 파일 저장
        lines.append(header + "\n" + sep + "\n" + doc + "\n\n")

    # 전체 내용을 파일로 저장
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("# 디지털미디어학과 청크 덤프 (벡터DB에서 직접 조회)\n\n")
        f.writelines(lines)

    print(f"=== DONE ===")
    print(f"총 {len(rows)} chunks. 전체 내용은 '{OUT_MD}' 에 저장됨.")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()