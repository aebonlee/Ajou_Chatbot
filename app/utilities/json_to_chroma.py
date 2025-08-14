import json, argparse
from langchain.schema import Document
from app.clients.chroma import get_collection
from app.utilities.utils import de_newline

def jsonl_to_docs(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            txt = obj.get("text") or obj.get("content") or ""
            txt = de_newline(txt)
            meta = {
                "section": obj.get("section"),
                "page": obj.get("page"),
                "printed_page_candidates": obj.get("printed_page_candidates"),
                "program_id": obj.get("meta",{}).get("program_id"),
                "program_name": obj.get("meta",{}).get("program_name"),
                "table_kind": obj.get("meta",{}).get("table_kind"),
                "content_type": obj.get("content_type"),
                "title": obj.get("title"),
                "source_id": obj.get("source_id"),
            }
            yield Document(page_content=txt, metadata=meta)

def run(jsonl: str, collection: str):
    vs = get_collection(collection)
    docs = list(jsonl_to_docs(jsonl))
    if docs:
        vs.add_documents(docs)
        vs.persist()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--collection", required=True)
    args = ap.parse_args()
    run(args.jsonl, args.collection)