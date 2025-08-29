from __future__ import annotations
import pendulum
import os
import pandas as pd
from datetime import datetime, timedelta
from konlpy.tag import Okt
import re
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import psycopg2
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from app.core.config import PERSIST_DIR_NOTICE,EMBEDDING_MODEL

# .env 파일에서 환경 변수 불러오기
load_dotenv()

# 환경 변수 사용
DB_HOST = os.getenv("DB_HOST")
DB_DATABASE = os.getenv("DB_DATABASE")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")

okt = Okt()
def preprocess_text(text):
    if not isinstance(text, str): return ""
    clean_text = re.sub(r'<[^>]+>', '', text)
    clean_text = re.sub(r'\(.*?\)', '', clean_text)
    clean_text = re.sub(r'[\r\n\t]', ' ', clean_text)
    clean_text = re.sub(r'[^\s\w가-힣]', '', clean_text)
    nouns = okt.nouns(clean_text)
    return " ".join(nouns)

def get_yesterday_data_from_all_dbs():
    conn = None
    try:
        conn = psycopg2.connect(host=DB_HOST, database=DB_DATABASE, user=DB_USER, password=DB_PASSWORD, port=DB_PORT)
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        # 각 테이블에서 어제 날짜 데이터 불러오기
        df_notices = pd.read_sql_query(f"SELECT id, title, url, content_text, date, '일반/장학공지' as category, NULL as college_name, NULL as department_name FROM notices WHERE date >= '{yesterday}';", conn)
        df_college = pd.read_sql_query(f"SELECT id, title, url, content_text, date, '단과대공지' as category, college_name, NULL as department_name FROM college_notices WHERE date >= '{yesterday}';", conn)
        df_department = pd.read_sql_query(f"SELECT id, title, url, content_text, date, '학과공지' as category, college_name, department_name FROM department_notices WHERE date >= '{yesterday}';", conn)
        
        # 모든 데이터프레임 통합
        combined_df = pd.concat([df_notices, df_college, df_department], ignore_index=True)
        print(f"DB에서 어제 날짜 데이터 총 {len(combined_df)}건을 불러와 통합했습니다.")
        return combined_df
    except (Exception, psycopg2.Error) as error:
        print("DB 연결 또는 쿼리 오류:", error)
        return pd.DataFrame()
    finally:
        if conn: conn.close()

def embed_and_add_to_vector_db(**kwargs):
    df = get_yesterday_data_from_all_dbs()
    if df.empty:
        print("새로운 데이터가 없으므로 임베딩을 건너뜁니다.")
        return

    df['preprocessed_text'] = df['content_text'].apply(preprocess_text)
    documents = []
    for _, row in df.iterrows():
        date_obj = pd.to_datetime(row['date'])
        timestamp = int(date_obj.timestamp())
        page_content = f"제목: {row['title']} | 내용: {row['preprocessed_text']}"
        
        metadata = {
            "notice_id": row['id'], "title": row['title'], "url": row['url'],
            "date": timestamp, "notice_type": row['category'],
            "college_name": row['college_name'], "department_name": row['department_name']
        }
        documents.append(Document(page_content=page_content, metadata=metadata))

    embeddings = HuggingFaceBgeEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})
    vectorstore = Chroma(persist_directory=PERSIST_DIR_NOTICE, embedding_function=embeddings)
    
    # 중복 삽입을 방지하기 위해 ID 필터링
    ids = [f"{doc.metadata['notice_type']}_{doc.metadata['notice_id']}" for doc in documents]
    
    vectorstore.add_documents(documents=documents, ids=ids)
    print(f"벡터 DB에 {len(documents)}개의 문서가 성공적으로 추가되었습니다.")
    vectorstore.persist()

with DAG(
    dag_id="embed_all_notices_dag",
    start_date=pendulum.datetime(2025, 1, 1, tz="Asia/Seoul"),
    schedule="0 1 * * *", # 매일 새벽 1시에 실행
    catchup=False,
    tags=["embedding", "vector_db"],
) as dag:
    embed_task = PythonOperator(
        task_id="embed_and_add_to_vector_db",
        python_callable=embed_and_add_to_vector_db,
    )