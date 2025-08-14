from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

SYS = (
  "You're the Slack bot \"래기\". You must speak in Korean."
  "Answer with the following documents as the primary basis, and if you don't know, answer that you don't know."
  "Quote briefly a clause/page/section that may be the source."
)

PROMPT = ChatPromptTemplate.from_messages([
  ("system", SYS),
  ("user", "Here is query contexts:\n{context}\n\nUser query:\n{question}")
])

def make_rag_chain(llm: ChatOpenAI):
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough
    def format_docs(docs):
        items=[]
        for i,d in enumerate(docs,1):
            page = d.metadata.get("page") or d.metadata.get("printed_page_candidates")
            sec = d.metadata.get("section") or d.metadata.get("meta",{}).get("section")
            items.append(f"[{i}] page={page} sec={sec}\n{d.page_content[:1200]}")
        return "\n\n---\n\n".join(items)

    return (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
        | PROMPT
        | llm
        | StrOutputParser()
    )