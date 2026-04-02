# SentryQuery-AI

A minimal **Retrieval-Augmented Generation (RAG)** prototype for asking questions over a folder of PDFs. It indexes documents into **Pinecone** and uses **LangChain (LCEL)** + **OpenAI** to generate answers grounded in retrieved context.

## What it does
- Loads PDFs from `./docs/`
- Chunks text for retrieval
- Creates embeddings with OpenAI
- Stores/retrieves vectors with Pinecone
- Builds a context-grounded prompt and answers with GPT via LangChain

## Architecture (workflow)
**Ingest (indexing)**
1. Load environment variables from `.env`
2. Load PDFs from `./docs`
3. Split text into overlapping chunks
4. Embed chunks
5. Upsert embeddings to Pinecone index (`sentry-index`)

**Query (retrieval + generation)**
1. Take a user question
2. Retrieve top-matching chunks from Pinecone
3. Insert chunks + question into a prompt template
4. Generate an answer with the chat model
5. Print the result

## Requirements
- Python 3.11+
- An OpenAI API key
- A Pinecone API key (and an existing Pinecone project)

## Setup
```bash
python3 -m venv venv
source venv/bin/activate

pip install -U langchain langchain-community langchain-openai langchain-pinecone pypdf pinecone-client python-dotenv
```

## Configuration
Create a `.env` file in the repo root (do not commit it):

```bash
OPENAI_API_KEY=...
PINECONE_API_KEY=...
# Optional (see `.env.example`):
# PINECONE_ENV=us-east-1
```

You can copy `.env.example` as a starting point.

## Add documents
Place one or more PDFs into:
- `./docs/`

## Run
```bash
./venv/bin/python sentry_query.py
```

## Example queries
Edit the `query = "..."` line in `sentry_query.py` and try:
- "Summarize the main purpose of these documents."
- "List key requirements or controls and group them by category."
- "Extract an implementation checklist."
- "What access control or PII-related guidance is described?"

## Notes / limitations
- This script currently re-embeds and re-upserts on every run (good for a demo, inefficient for production).
- Data is sent to cloud services (OpenAI + Pinecone). Avoid indexing confidential data unless you have permission and governance controls.

## License
MIT (see `LICENSE`).
