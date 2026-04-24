# SentryQuery AI

An agentic AI assistant that answers questions over indexed documents using 
LangGraph, LangChain, Pinecone, and GPT-4o. Built with a Streamlit web UI.

## Architecture

Documents are indexed once into Pinecone using OpenAI embeddings. On each 
query, a LangGraph ReAct agent reasons about when and how to call the 
retriever tool, rather than following a fixed pipeline. The agent can call 
the retriever multiple times with different queries if needed.

## Stack

- LangGraph for agentic ReAct reasoning
- LangChain for retriever tooling
- Pinecone as the vector database
- OpenAI text-embedding-3-small for embeddings
- GPT-4o as the language model
- Streamlit for the web UI

## Setup

python3 -m venv venv
source venv/bin/activate
pip install langchain langchain-community langchain-openai langchain-pinecone langgraph streamlit pinecone-client python-dotenv pypdf

Create a .env file with:
OPENAI_API_KEY=your_key
PINECONE_API_KEY=your_key

## Usage

Index your documents once:
python sentry_query.py --ingest

Launch the web UI:
python -m streamlit run sentry_query.py
