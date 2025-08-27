#!/usr/bin/env python3
# app/scripts/inspect_chunks.py
import argparse, statistics
from collections import Counter
from app.core import config
from app.services.storage import get_client, get_collection, get_all

def quantiles(a):
    if not a: return (0,0,0,0,0)
    a = sorted(a)
    n = len(a)
    q = lambda p: a[max(0, min(n-1, int(p*(n-1))))]
    return (a[0], q(0.25), q(0.5), round(sum(a)/n, 1), q(0.75), a[-1])

def main():
    ap = argparse.ArgumentParser(description="Inspect chunks in Chroma")
    ap.add_argument("--dept", default="디지털미디어학과")
    ap.add_argument("--contains", default=None, help="Filter by substring in path or body")
    ap.add_argument("--limit", type=int, default=20, help="Print first N samples")
    args = ap.parse_args()

    client = get_client(config.PERSIST_DIR)
    col = get_collection(client, config.COLLECTION, config.EMBEDDING_MODEL)
    ids, docs, metas = get_all(col)

    rows = []
    for i in range(len(ids)):
        meta = metas[i] or {}
        if meta.get("dept") == args.dept:
            doc = docs[i] or ""
            if args.contains:
                if args.contains not in (meta.get("path") or "") and args.contains not in doc:
                    continue
            rows.append((i, ids[i], doc, meta))

    print(f"[INFO] dept={args.dept}  matched_chunks={len(rows)}  (collection={config.COLLECTION})")
    if not rows:
        return

    lens = [len(r[2]) for r in rows]
    mn, p25, med, mean, p75, mx = quantiles(lens)
    print(f"[LEN] min={mn}  p25={p25}  median={med}  mean={mean}  p75={p75}  max={mx}")

    paths = [r[3].get("path", "") for r in rows]
    top_paths = Counter(paths).most_common(30)
    print("\n[TOP PATHS by chunk count]")
    for p, n in top_paths:
        print(f"{n:4d}  {p}")

    print(f"\n[SAMPLES] first {min(args.limit, len(rows))} chunks\n" + "-"*80)
    for k, (gi, cid, doc, meta) in enumerate(rows[:args.limit], 1):
        path = meta.get("path", "")
        body = doc.split("\n", 1)[1] if doc.startswith("[PATH]") and "\n" in doc else doc
        print(f"#{k} gi={gi} id={cid} len={len(doc)}")
        print(f"path: {path}")
        print(body[:600].replace("\n", "⏎"))
        print("-"*80)

if __name__ == "__main__":
    main()