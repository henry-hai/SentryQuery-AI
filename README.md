# SentryQuery AI

An agentic AI assistant that answers questions over indexed enterprise documents
using a LangGraph ReAct agent over Pinecone, with a Streamlit web UI.

## Architecture

Documents are indexed once into Pinecone using OpenAI embeddings. On each query,
a LangGraph ReAct agent (built via `langchain.agents.create_agent`, which
compiles a LangGraph state graph internally) reasons about whether and how to
call the retriever tool — rather than following a fixed retrieve-then-answer
pipeline. The agent can call the retriever multiple times with refined queries
if needed, and a system prompt restricts it to grounded, on-topic answers.

The Streamlit UI surfaces both the final answer and the source document chunks
the agent consulted, expanding by PDF and page.

## Stack

- LangGraph for agentic ReAct reasoning (via `langchain.agents.create_agent`)
- LangChain for retriever tooling
- Pinecone as the vector database
- OpenAI `text-embedding-3-small` for embeddings (1536-d)
- GPT-4o as the language model
- Tavily for live web search (optional second agent tool)
- Streamlit for the web UI

## Setup

```
python -m venv venv
source venv/bin/activate    # or: venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Create a `.env` file with:

```
OPENAI_API_KEY=your_key
PINECONE_API_KEY=your_key
TAVILY_API_KEY=your_key   # optional; enables the web_search tool
```

Pre-create a Pinecone index named `sentry-index` with dimension `1536` and
cosine similarity.

## Usage

Drop your PDFs into `./docs/` and index them once:

```
python sentry_query.py --ingest
```

Launch the web UI:

```
python -m streamlit run sentry_query.py
```

## Evaluation

A small eval harness lives in `evals/`. It runs each Q&A pair in `evals/qa.json`
through the agent and checks (1) the answer contains an expected keyword and
(2) the agent correctly used (or skipped) the retriever for the given question.
Off-topic prompts are expected to be refused per the system prompt.

```
python evals/eval.py
```
