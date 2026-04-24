import os
import sys
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent as create_react_agent
from langchain_core.tools.retriever import create_retriever_tool

# Load environment variables from .env
load_dotenv()

# Connect to Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "sentry-index"

# Ingestion mode: only runs when called with --ingest flag
# Run once to index your docs: python sentry_query.py --ingest
# After that, the index persists in Pinecone and this block is skipped
INGEST_MODE = "--ingest" in sys.argv

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

if INGEST_MODE:
    # Clear old vectors so stale data does not persist across ingestions
    existing_index = pc.Index(index_name)
    existing_index.delete(delete_all=True)
    print("Cleared existing vectors.")
    loader = PyPDFDirectoryLoader("./docs")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = PineconeVectorStore.from_documents(splits, embeddings, index_name=index_name)
    print("Ingestion complete.")
else:
    # Connect to existing Pinecone index without re-ingesting
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# Agentic AI pipeline using ReAct via LangGraph
# The LLM reasons about when and how to call the retriever as a tool
# rather than always retrieving on every query in a fixed pipeline
llm = ChatOpenAI(model="gpt-4o", temperature=0)

retriever_tool = create_retriever_tool(
    vectorstore.as_retriever(),
    "search_ingram_micro_docs",
    "Search Ingram Micro documents for information about the company, Xvantage platform, AI strategy, and operations."
)

tools = [retriever_tool]
agent = create_react_agent(llm, tools)

st.title("SentryQuery Agentic AI Assistant")
st.caption("Powered by LangChain, Pinecone, and GPT-4o")

query = st.text_input("Ask about Ingram Micro:")

if st.button("Run") and query:
    with st.spinner("Agent thinking..."):
        response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    st.write(response["messages"][-1].content)