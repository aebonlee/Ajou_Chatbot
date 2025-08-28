from __future__ import annotations
import pendulum
import os
import asyncio
import json
import re
import traceback
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from urllib.parse import urljoin, urlparse, parse_qs
from playwright.async_api import async_playwright
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator

# --- DB & Crawling Configuration ---
load_dotenv("/home/ma/ICT/.env")
DB_HOST = os.getenv("DB_HOST")
DB_DATABASE = os.getenv("DB_DATABASE")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")

BOARDS = ["notice", "notice_scholarship"]
LIST_LINK = ("div.b-title-box a[href*='articleNo='][href*='mode=view'], "
             "a[href*='articleNo='][href*='mode=view']")
BODY_PRIMARY = "div.b-content-box"
BODY_FALLBACK = [".fr-view", ".board-view .b-contents", ".board-view", ".view .contents", ".view"]
POLITE_DELAY_SEC = 0.2
HARD_CAP_PAGES = 300
STOP_PAGES_WITHOUT = 5

# --- Common Utilities ---
def kst_yesterday_str() -> str:
    return (datetime.now(ZoneInfo("Asia/Seoul")).date() - timedelta(days=1)).strftime("%Y-%m-%d")

def parse_article_no(href: str) -> str | None:
    qs = parse_qs(urlparse(href).query); vals = qs.get("articleNo")
    return vals[0] if vals else None

async def crawl_for_date(board: str, target_date: str) -> list[dict]:
    base = f"https://www.ajou.ac.kr/kr/ajou/{board}.do"
    results, seen = [], set()
    print(f"[START] board={board}, target_date={target_date}")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(
            locale="ko-KR", user_agent=("Mozilla/5.0"), viewport={"width": 1360, "height": 900}
        )
        page = await ctx.new_page()
        page_idx, consecutive_no_target = 0, 0
        while page_idx < HARD_CAP_PAGES:
            list_url = f"{base}?article.offset={page_idx*10}&articleLimit=10&mode=list"
            try:
                await page.goto(list_url, wait_until="domcontentloaded", timeout=60_000)
                await page.wait_for_load_state("networkidle")
                await page.wait_for_selector(LIST_LINK, state="attached", timeout=20_000)
            except Exception:
                print(f"[{board}][PAGE {page_idx+1}] no links → stop")
                break
            anchors = await page.locator(LIST_LINK).all()
            any_target_on_page = False
            page_items = []
            for a in anchors:
                href_rel = await a.get_attribute("href")
                if not href_rel: continue
                url = urljoin(list_url, href_rel)
                aid = parse_article_no(href_rel)
                if not aid or aid in seen: continue
                title_attr = await a.get_attribute("title"); title_text = (await a.inner_text()) or ""
                title = (title_attr or title_text).strip()
                row = a.locator("xpath=ancestor::tr[1]"); tds = row.locator("td"); ntd = await tds.count()
                date_text = (await tds.nth(ntd - 1).inner_text()).strip() if ntd > 0 else ""
                if date_text == target_date: any_target_on_page = True
                if date_text != target_date: continue
                seen.add(aid)
                page_items.append({"url": url, "id": aid, "title": title, "date": date_text})
            added = 0
            for m in page_items:
                d = await ctx.new_page()
                try:
                    await d.goto(m["url"], wait_until="domcontentloaded", timeout=60_000)
                    await d.wait_for_load_state("networkidle")
                    container = await d.locator(BODY_PRIMARY).first.count() > 0 and d.locator(BODY_PRIMARY).first or next((d.locator(sel).first for sel in BODY_FALLBACK if await d.locator(sel).count() > 0), None)
                    content_text, images = "", []
                    if container:
                        content_text = (await container.inner_text()).strip()
                        images = [urljoin(m["url"], await img.get_attribute("src")) for img in await container.locator("img[src]").all()]
                    results.append({"id": int(m["id"]), "title": m["title"], "date": m["date"], "url": m["url"], "content_text": content_text, "images": images})
                    added += 1
                finally: await d.close()
                await page.wait_for_timeout(int(POLITE_DELAY_SEC * 1000))
            if not any_target_on_page and added == 0:
                consecutive_no_target += 1
            else:
                consecutive_no_target = 0
            if consecutive_no_target >= STOP_PAGES_WITHOUT:
                print(f"[{board}] stop condition met")
                break
            page_idx += 1
        await browser.close()
    return results

def insert_general_data_to_db(data: list[dict], category: str):
    conn = None
    try:
        conn = psycopg2.connect(host=DB_HOST, database=DB_DATABASE, user=DB_USER, password=DB_PASSWORD, port=DB_PORT)
        cursor = conn.cursor()
        data_to_insert = [(row['id'], row['title'], datetime.strptime(row['date'], '%Y-%m-%d'), row['url'], row['content_text'], json.dumps(row['images']), category) for row in data]
        insert_query = """
        INSERT INTO notices (id, title, date, url, content_text, images, category)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE
        SET title = EXCLUDED.title, date = EXCLUDED.date, url = EXCLUDED.url,
            content_text = EXCLUDED.content_text, images = EXCLUDED.images, category = EXCLUDED.category;
        """
        execute_batch(cursor, insert_query, data_to_insert)
        conn.commit()
    except (Exception, psycopg2.Error) as error:
        traceback.print_exc()
    finally:
        if conn:
            cursor.close()
            conn.close()

# --- Airflow Task Function ---
def crawl_and_store_general(**kwargs):
    loop = asyncio.get_event_loop()
    target_date = kst_yesterday_str()
    for board in BOARDS:
        category = "장학공지" if "notice_scholarship" in board else "일반공지"
        async def run_crawl():
            return await crawl_for_date(board, target_date)
        scraped_data = loop.run_until_complete(run_crawl())
        insert_general_data_to_db(scraped_data, category)

# --- DAG Definition ---
with DAG(
    dag_id="crawl_general_notices_dag",
    start_date=pendulum.datetime(2025, 1, 1, tz="Asia/Seoul"),
    schedule="@daily",
    catchup=False,
    tags=["notice_crawler", "general"],
) as dag:
    crawl_task = PythonOperator(
        task_id="crawl_and_store_general_notices",
        python_callable=crawl_and_store_general,
    )