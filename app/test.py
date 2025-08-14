import asyncio, ssl, certifi, re, os, glob
from typing import List, Sequence
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_sdk.web.async_client import AsyncWebClient

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from konlpy.tag import Okt
TOKENIZER = Okt()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

import logging

logging.basicConfig(level=logging.INFO)
rag_logger = logging.getLogger("rag")

OPENAI_API_KEY= "my_openai_api_key"
ANTHROPIC_API_KEY= "my_anthropic_key"
SLACK_BOT_TOKEN = "my_slack_bot_token"
SLACK_APP_TOKEN = "my_slack_app_token"

PDF_PATHS = glob.glob("data/pdfs/*.pdf") # pdf 경로
CHROMA_DIR = "./chroma_db" # Chroma 퍼시스턴스 경로

ssl_context = ssl.create_default_context(cafile=certifi.where())
async_web_client = AsyncWebClient(token=SLACK_BOT_TOKEN, ssl=ssl_context)
app_slack = AsyncApp(token=SLACK_BOT_TOKEN, client=async_web_client)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=OPENAI_API_KEY,)
# llm = ChatAnthropic(
#     model_name="claude-3-5-sonnet-20240620",
#     temperature=0.2,
#     api_key=ANTHROPIC_API_KEY,
#     timeout=None,
#     stop=None,
# )
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)

def load_chroma(pdf_paths: Sequence[str], persistent_dir: str) -> Chroma:
    # 이미 persistence directory가 있으면 그냥 가져오기
    if os.path.exists(persistent_dir) and len(os.listdir(persistent_dir)) > 0:
        return Chroma(persist_directory=persistent_dir, embedding_function=embeddings)
    # 없으면 빌드해서 가져오기
    all_docs = []
    for p in pdf_paths:
        if not os.path.isfile(p):
            raise ValueError(f"{p} is not a file")
        loader = PyPDFLoader(p)
        all_docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(all_docs)
    vs = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=persistent_dir)
    return vs

vector_vs = load_chroma(PDF_PATHS, CHROMA_DIR)
chroma_r = vector_vs.as_retriever(search_kwargs={"k": 10})

# 한국어 토크나이저
def korean_tokenizer(text: str) -> List[str]:
    return TOKENIZER.morphs(text)

# BM25 인덱스 구축하기
loader_docs = []
for p in PDF_PATHS:
    loader_docs.extend(PyPDFLoader(p).load())

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)
bm25_chunks = splitter.split_documents(loader_docs)

def ko_tokenize(x: str) -> List[str]:
    return TOKENIZER.morphs(x)

bm25_r = BM25Retriever.from_documents(
    bm25_chunks,
    preprocess_func=ko_tokenize
)
bm25_r.k = 10
ensemble = EnsembleRetriever(
    retrievers=[chroma_r, bm25_r],
    weights=[0.2, 0.8]  # 필요 시 가중치 조절
)



def format_docs(docs, max_chars: int = 1800) -> str:
    out = []
    for i, d in enumerate(docs, 1):
        if not hasattr(d, "metadata") or not hasattr(d, "page_content"):
            # 예상치 못한 타입 방어
            continue
        src = d.metadata.get("source", "doc")
        page = d.metadata.get("page", None)
        header = f"[{i}] source={src}" + (f", page={page}" if page is not None else "")
        body = d.page_content.replace("\u200b", "").strip()
        if max_chars:
            body = body[:max_chars]
        out.append(f"{header}\n{body}")
    return "\n\n---\n\n".join(out)

def preview(text: str, n: int = 120) -> str:
    return text[:n].replace("\n", " ")

def make_context(inputs: dict) -> str:
    query = inputs["user_input"]
    docs = ensemble.get_relevant_documents(query)
    rag_logger.info("🔎 RAG 검색 질의: %s", query)
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "doc")
        page = d.metadata.get("page", None)
        rag_logger.info("  [%d] src=%s page=%s | %s",
                        i, src, page, d.page_content[:120].replace("\n", " "))
    return format_docs(docs)




system_prompt = (
    "You're the Slack bot \"래기\". You must speak in Korean."
    "Answer with the following documents as the primary basis, and if you don't know, answer that you don't know."
    "Quote briefly a clause/page/section that may be the source."
)
user_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("history"),
    ("user", "The following are a reference documents: \n{context}"),
    ("user", "{user_input}"),
])


rag_chain = (
    RunnablePassthrough.assign(
        context=RunnableLambda(make_context)
    )
    | user_prompt
    | llm
)


# 테스트용 인메모리 대화 메모리 저장소 (나중에 redis나 DB 등으로 대체)
store: dict[str, ChatMessageHistory] = {}

def get_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

runnable = RunnableWithMessageHistory(
    rag_chain,
    get_history,
    input_messages_key="user_input",
    history_messages_key="history",
)

BOT_MENTION_RE = re.compile(r"<@([A-Z0-9]+)>\s*")
def strip_bot_mention(text: str, body: dict) -> str:
    # authorizations[0].user_id 가 봇 유저ID
    bot_id = None
    try:
        bot_id = body["authorizations"][0]["user_id"]
    except Exception:
        pass
    if bot_id:
        text = re.sub(rf"<@{bot_id}>\s*", "", text).strip()
    return text


# ===여기서부터 이벤트 핸들러===
@app_slack.event("app_mention")
async def handle_mentions(body, say, logger):
    try:
        text = body["event"].get("text", "")
        clean_text = strip_bot_mention(text, body)

        channel = body["event"].get("channel")
        user = body["event"].get("user")
        session_id = f"{channel}:{user}"

        # RAG+히스토리 호출
        result = await runnable.ainvoke(
            {"user_input": clean_text},
            config={"configurable": {"session_id": session_id}},
        )
        await say(result.content)
    except Exception as e:
        logger.exception(e)
        await say("에러가 발생했습니다. 잠시 후 다시 시도해 주세요.")

# 모든 메시지 로깅
@app_slack.event("message")
async def handle_all_messages(body, logger):
    logger.info(f"[message] {body}")


# 소켓모드로 봇 부팅
async def main():
    handler = AsyncSocketModeHandler(app_slack, SLACK_APP_TOKEN, web_client=async_web_client)
    await handler.start_async()

if __name__ == "__main__":
    asyncio.run(main())