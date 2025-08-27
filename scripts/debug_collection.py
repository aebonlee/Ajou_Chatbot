# scripts/debug_collection.py
from app.services.storage import get_client
from app.core import config

client = get_client(config.PERSIST_DIR)

# 내가 쓰는 컬렉션 이름이 실제로 존재하는지 확인
cols = client.list_collections()
print("== Collections ==")
for c in cols:
    print("-", c.name)

# 선택 컬렉션 count 확인
from app.services.storage import get_collection
col = get_collection(client, config.COLLECTION, config.EMBEDDING_MODEL)
try:
    print("count:", col.count())
except Exception as e:
    print("count error:", e)