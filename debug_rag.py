# 디버깅을 위한 스크립트

import sys
import os

sys.path.append('/Users/leesj/Desktop/dev/Project/ICT_11')

from app.services.storage import get_client, get_collection, get_all, get_where_all
from app.services.textutil import tokenize_ko, normalize_numbers
from rank_bm25 import BM25Okapi

# 설정
PERSIST_DIR = "storage/chroma-acad"
COLLECTION = "acad_docs_bge_m3_clean_v2"
EMBEDDING_MODEL = "BAAI/bge-m3"


def debug_retrieval():
    print("=== RAG 디버깅 시작 ===")

    # 1. 컬렉션 연결 확인
    client = get_client(PERSIST_DIR)
    col = get_collection(client, COLLECTION, EMBEDDING_MODEL)
    print(f"✓ 컬렉션 연결: {COLLECTION}")

    # 2. 전체 데이터 확인
    all_ids, all_docs, all_metas = get_all(col)
    print(f"✓ 전체 문서 수: {len(all_docs)}")

    # 3. 디지털미디어학과 필터링 확인
    target_dept = "디지털미디어학과"
    scope_idx = [i for i, m in enumerate(all_metas) if (m or {}).get("dept") == target_dept]
    print(f"✓ {target_dept} 문서 수: {len(scope_idx)}")

    if len(scope_idx) == 0:
        print("❌ 디지털미디어학과 문서가 없습니다!")
        # 실제 학과명들 확인
        depts = set()
        for m in all_metas:
            if m and m.get("dept"):
                depts.add(m.get("dept"))
        print(f"실제 학과명들: {sorted(depts)}")
        return

    # 4. 2학년 관련 문서 확인
    year_docs = []
    for i in scope_idx:
        meta = all_metas[i]
        doc = all_docs[i]
        if "2학년" in (doc or "") or meta.get("year") == "2학년":
            year_docs.append((i, meta, doc[:200]))

    print(f"✓ 2학년 관련 문서: {len(year_docs)}개")
    for i, (idx, meta, preview) in enumerate(year_docs[:3]):
        print(f"  {i + 1}. {meta.get('path', 'N/A')}")
        print(f"     Year: {meta.get('year', 'N/A')}, Section: {meta.get('section', 'N/A')}")
        print(f"     Preview: {preview}")
        print()

    # 5. BM25 검색 테스트
    question = "미디어과 2학년때 무슨 과목 들어야해?"
    normalized_q = normalize_numbers(question)
    print(f"✓ 원본 질문: {question}")
    print(f"✓ 정규화된 질문: {normalized_q}")

    # BM25 토큰화 확인
    q_tokens = tokenize_ko(normalized_q)
    print(f"✓ 질문 토큰: {q_tokens[:10]}...")  # 처음 10개만

    # BM25 코퍼스 구성
    corpus = [tokenize_ko(all_docs[i] or "") for i in scope_idx]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(q_tokens)

    # 상위 점수들 확인
    top_pairs = sorted(zip(scope_idx, scores), key=lambda x: x[1], reverse=True)[:5]
    print(f"✓ BM25 상위 5개 결과:")
    for rank, (idx, score) in enumerate(top_pairs, 1):
        meta = all_metas[idx]
        print(f"  {rank}. Score: {score:.4f}")
        print(f"     Path: {meta.get('path', 'N/A')}")
        print(f"     Year: {meta.get('year', 'N/A')}")
        print()

    # 6. Dense 검색 테스트
    try:
        where_dense = {"dept": target_dept}
        dense_result = col.query(
            query_texts=[normalized_q],
            n_results=5,
            where=where_dense,
            include=["distances", "metadatas"],
        )
        print(f"✓ Dense 검색 결과: {len(dense_result.get('ids', [[]])[0])}개")

        if dense_result.get('ids') and dense_result['ids'][0]:
            for i, (doc_id, dist) in enumerate(zip(dense_result['ids'][0], dense_result['distances'][0])):
                sim = 1.0 - dist
                print(f"  {i + 1}. ID: {doc_id}, Similarity: {sim:.4f}")

    except Exception as e:
        print(f"❌ Dense 검색 실패: {e}")

    # 7. where 조건 테스트
    try:
        where_test = {"dept": target_dept}
        test_ids, test_docs, test_metas = get_where_all(col, where_test)
        print(f"✓ where 조건 테스트: {len(test_docs)}개 문서")
    except Exception as e:
        print(f"❌ where 조건 테스트 실패: {e}")


if __name__ == "__main__":
    debug_retrieval()