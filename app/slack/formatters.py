def to_blocks(res: dict):
    blocks = [
        {"type":"section","text":{"type":"mrkdwn","text":res["answer"]}},
        {"type":"context","elements":[{"type":"mrkdwn","text":f"*intent:* {res['intent']}  •  *tool:* {res.get('tool')}  •  *conf:* {res.get('confidence',0):.2f}"}]}
    ]
    if res.get("sources"):
        s = "\n".join([f"- {x.get('title','')}, p.{x.get('page')}" for x in res["sources"]])
        blocks.append({"type":"context","elements":[{"type":"mrkdwn","text":"*sources:*\n"+s}]})
    return blocks