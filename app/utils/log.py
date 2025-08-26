# app/utils/log.py
import json, time, uuid, logging
from functools import wraps

logger = logging.getLogger("acad")
handler = logging.StreamHandler()
fmt = logging.Formatter("%(message)s")
handler.setFormatter(fmt)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False

def jlog(**kw):
    logger.info(json.dumps(kw, ensure_ascii=False))

def timed(span: str):
    def deco(fn):
        @wraps(fn)
        def wrap(*a, **k):
            t0 = time.time()
            rid = k.get("request_id") or str(uuid.uuid4())
            try:
                res = fn(*a, **k)
                jlog(span=span, request_id=rid, ms=round((time.time()-t0)*1000,2), ok=True)
                return res
            except Exception as e:
                jlog(span=span, request_id=rid, ms=round((time.time()-t0)*1000,2), ok=False, error=str(e))
                raise
        return wrap
    return deco