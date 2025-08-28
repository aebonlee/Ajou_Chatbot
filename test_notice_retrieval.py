# scripts/inspect_chroma_info.py
import os, sys, json, collections
import chromadb
from chromadb.utils import embedding_functions

PERSIST_DIR = os.environ.get("PERSIST_DIR_INFO", "storage/chroma-info")
EMB_MODEL   = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")

def p(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2))

def main():
    print(f"📁 PERSIST_DIR_INFO = {PERSIST_DIR}")
    if not os.path.isdir(PERSIST_DIR):
        print("❌ 디렉터리가 없습니다. (최초 인덱싱이 안됐을 가능성)")
        sys.exit(0)

    print("\n📄 디렉터리 파일 목록:")
    for name in sorted(os.listdir(PERSIST_DIR)):
        print(" -", name)

    # Chroma 클라이언트
    client = chromadb.PersistentClient(path=PERSIST_DIR)

    # 컬렉션 나열
    cols = client.list_collections()
    print("\n🗂️  컬렉션 목록:")
    for c in cols:
        print(" -", c.name)

    if not cols:
        print("❌ 컬렉션이 없습니다. (빈 인덱스)")
        return

    # 보통 하나일 것. 필요하면 루프 돌면서 모두 확인.
    for c in cols:
        print("\n" + "="*80)
        print(f"📌 컬렉션: {c.name}")

        # 로드된 컬렉션에는 embedding_function이 비어있을 수 있어 붙여줌
        try:
            ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMB_MODEL)
            c._embedding_function = ef  # attach (query에 필요)
        except Exception as e:
            print(f"⚠️ 임베딩 함수 설정 실패: {e}")

        # 개수
        try:
            total = c.count()
            print(f"총 문서(청크) 수: {total}")
        except Exception as e:
            print(f"count() 실패: {e}")

        # 샘플 5건 가져오기
        res = c.get(include=["metadatas", "documents"], limit=5, offset=0)
        ids = res.get("ids", [])
        mets = res.get("metadatas", [])
        docs = res.get("documents", [])
        print("\n🔎 샘플(최대 5건):")
        for i, cid in enumerate(ids):
            meta = mets[i] if i < len(mets) else {}
            doc  = docs[i][:200].replace("\n", " ") + ("..." if docs[i] and len(docs[i])>200 else "")
            print(f"  [{i+1}] id={cid}")
            print(f"      meta.title / source / page: {meta.get('title')} / {meta.get('source')} / {meta.get('page')}")
            print(f"      path/section: {meta.get('path')} / {meta.get('section')}")
            print(f"      preview: {doc}")

        # 소스/타이틀 분포 상위
        print("\n📊 source(또는 title) 상위 분포:")
        counter = collections.Counter()
        for m in mets:
            k = m.get("title") or m.get("source") or "unknown"
            counter[k] += 1
        for k, v in counter.most_common(10):
            print(f"  - {k}: {v}")

        # 간단 쿼리 테스트
        try:
            print("\n🧪 쿼리 테스트 (query='장학'):")
            q = c.query(query_texts=["장학"], n_results=5, include=["metadatas","documents"])
            q_ids = q.get("ids", [[]])[0]
            q_metas = q.get("metadatas", [[]])[0]
            q_docs = q.get("documents", [[]])[0]
            if not q_ids:
                print("  (검색 결과 0건)")
            else:
                for i, cid in enumerate(q_ids):
                    meta = q_metas[i] if i < len(q_metas) else {}
                    doc  = q_docs[i][:160].replace("\n", " ") + ("..." if q_docs[i] and len(q_docs[i])>160 else "")
                    print(f"  [{i+1}] {meta.get('title') or meta.get('source')}  page={meta.get('page')}")
                    print(f"       path={meta.get('path')}")
                    print(f"       preview={doc}")
        except Exception as e:
            print(f"쿼리 실패: {e}")

if __name__ == "__main__":
    main()