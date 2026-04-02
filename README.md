# SentryQuery-AI

A minimal **Retrieval-Augmented Generation (RAG)** prototype for asking questions over a folder of PDFs. It indexes documents into **Pinecone** and uses **LangChain (LCEL)** + **OpenAI** to generate answers grounded in retrieved context.

## Features
- Loads PDFs from `./docs/`
- Chunks text for retrieval
- Creates embeddings with OpenAI
- Stores/retrieves vectors with Pinecone
- Builds a context-grounded prompt and answers with GPT via LangChain

## Architecture (Workflow)
**Ingest (Indexing)**
1. Load environment variables from `.env`
2. Load PDFs from `./docs`
3. Split text into overlapping chunks
4. Embed chunks
5. Upsert embeddings to Pinecone index (`sentry-index`)

**Query (Retrieval + Generation)**
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

In the project root (next to `sentry_query.py`), create a file named `.env`. It is gitignored; do not commit it.

Add exactly these two variables, each on its own line, with your real keys after the `=`:

```
OPENAI_API_KEY=your_openai_key_here
PINECONE_API_KEY=your_pinecone_key_here
```

`load_dotenv()` in `sentry_query.py` loads this file. LangChain’s OpenAI classes read `OPENAI_API_KEY`. Pinecone reads `PINECONE_API_KEY`.

## Add Documents
Place one or more PDFs into:
- `./docs/`

## Run
```bash
./venv/bin/python sentry_query.py
```

## Example Queries
Edit the `query = "..."` line in `sentry_query.py` and try:
- "Summarize the main purpose of these documents."
- "List key requirements or controls and group them by category."
- "Extract an implementation checklist."
- "What access control or PII-related guidance is described?"

## Notes / Limitations
- This script currently re-embeds and re-upserts on every run (good for a demo, inefficient for production).
- Data is sent to cloud services (OpenAI + Pinecone). Avoid indexing confidential data unless you have permission and governance controls.

## License
MIT (see `LICENSE`).
