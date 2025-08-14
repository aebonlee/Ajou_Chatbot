import asyncio, ssl, certifi, re, os
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_sdk.web.async_client import AsyncWebClient
import httpx
from app.config.settings import settings
from app.slack.formatters import to_blocks

ssl_ctx = ssl.create_default_context(cafile=certifi.where())
web = AsyncWebClient(token=settings.SLACK_BOT_TOKEN, ssl=ssl_ctx)
app = AsyncApp(token=settings.SLACK_BOT_TOKEN, client=web)

API_URL = os.getenv("ORCH_URL", "http://localhost:8000/query")

def strip_mention(text, bot_id):
    return re.sub(rf"<@{bot_id}>\s*", "", text).strip()

@app.event("app_mention")
async def on_mention(body, say, logger):
    bot_id = body["authorizations"][0]["user_id"]
    text = strip_mention(body["event"].get("text",""), bot_id)
    payload = {"user_id": body["event"]["user"], "channel_id": body["event"]["channel"], "text": text}

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(API_URL, json=payload)
        data = resp.json()

    await say(blocks=to_blocks(data), text=data["answer"])

async def main():
    handler = AsyncSocketModeHandler(app, settings.SLACK_APP_TOKEN, web_client=web)
    await handler.start_async()

if __name__ == "__main__":
    asyncio.run(main())