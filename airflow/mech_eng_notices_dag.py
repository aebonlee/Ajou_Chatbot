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

LIST_LINK = "div.b-title-box a[href*='articleNo=']"
BODY_PRIMARY = "div.notice-view"
BODY_FALLBACKS = [".board-view .b-contents", ".board-view", ".view .contents", ".view"]
POLITE_DELAY_SEC = 0.2
HARD_CAP_PAGES = 300
STOP_PAGES_WITHOUT = 5

# --- Common Utilities ---
def kst_yesterday_str() -> str:
    return (datetime.now(ZoneInfo("Asia/Seoul")).date() - timedelta(days=1)).strftime("%Y-%m-%d")

def parse_article_no(href: str) -> str | None:
    qs = parse_qs(urlparse(href).query); vals = qs.get("articleNo")
    return vals[0] if vals else None

async def crawl_mech_eng_notices(target_date: str):
    base_url = "https://me.ajou.ac.kr/me/board/under-notice.do"
    results, seen = [], set()
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1360, "height": 900})
        page = await ctx.new_page()
        page_idx, consecutive_no_target = 0, 0
        while page_idx < HARD_CAP_PAGES:
            list_url = f"{base_url}?article.offset={page_idx*10}&articleLimit=10&mode=list"
            try:
                await page.goto(list_url, wait_until="domcontentloaded", timeout=60_000)
                await page.wait_for_load_state("networkidle")
                await page.wait_for_selector(LIST_LINK, state="attached", timeout=20_000)
            except Exception:
                break
            anchors = await page.locator(LIST_LINK).all()
            for a in anchors:
                href_rel = await a.get_attribute("href")
                url = urljoin(list_url, href_rel)
                aid = parse_article_no(href_rel)
                row = a.locator("xpath=ancestor::tr[1]")
                date_text = (await row.locator("td:last-child").inner_text()).strip()
                if date_text != target_date: continue
                if not aid or aid in seen: continue
                seen.add(aid)
                title = (await a.inner_text()).strip()
                d = await ctx.new_page()
                try:
                    await d.goto(url, wait_until="domcontentloaded", timeout=60_000)
                    container = await d.locator(BODY_PRIMARY).first.count() > 0 and d.locator(BODY_PRIMARY).first or next((d.locator(sel).first for sel in BODY_FALLBACKS if await d.locator(sel).count() > 0), None)
                    content_text, images = "", []
                    if container:
                        content_text = (await container.inner_text()).strip()
                        images = [urljoin(url, await img.get_attribute("src")) for img in await container.locator("img[src]").all()]
                    results.append({"id": int(aid), "title": title, "date": date_text, "url": url, "content_text": content_text, "images": images})
                finally:
                    await d.close()
            page_idx += 1
        await browser.close()
        return results

def insert_department_data_to_db(data: list[dict], college: str, department: str):
    conn = None
    try:
        conn = psycopg2.connect(host=DB_HOST, database=DB_DATABASE, user=DB_USER, password=DB_PASSWORD, port=DB_PORT)
        cursor = conn.cursor()
        data_to_insert = [(row['id'], row['title'], datetime.strptime(row['date'], '%Y-%m-%d'), row['url'], row['content_text'], json.dumps(row['images']), college, department) for row in data]
        insert_query = """
        INSERT INTO department_notices (id, title, date, url, content_text, images, college_name, department_name)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE
        SET title = EXCLUDED.title, date = EXCLUDED.date, url = EXCLUDED.url,
            content_text = EXCLUDED.content_text, images = EXCLUDED.images,
            college_name = EXCLUDED.college_name, department_name = EXCLUDED.department_name;
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
def crawl_and_store_mech_eng(**kwargs):
    loop = asyncio.get_event_loop()
    target_date = kst_yesterday_str()
    async def run_crawl():
        return await crawl_mech_eng_notices(target_date)
    scraped_data = loop.run_until_complete(run_crawl())
    insert_department_data_to_db(scraped_data, '공과대학', '기계공학과')

# --- DAG Definition ---
with DAG(
    dag_id="crawl_mech_eng_notices_dag",
    start_date=pendulum.datetime(2025, 1, 1, tz="Asia/Seoul"),
    schedule="@daily",
    catchup=False,
    tags=["notice_crawler", "engineering", "mechanical"],
) as dag:
    crawl_task = PythonOperator(
        task_id="crawl_and_store_mech_eng_notices",
        python_callable=crawl_and_store_mech_eng,
    )