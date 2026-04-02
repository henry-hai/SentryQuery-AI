# SentryQuery-AI

RAG over PDFs: chunk → OpenAI embeddings → Pinecone → retrieve → GPT answer via LangChain (LCEL). Index name: `sentry-index`.

**Flow:** `.env` + load `./docs` → split → embed → upsert Pinecone → question → retrieve chunks → prompt → print.

## Setup and run

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U langchain langchain-community langchain-openai langchain-pinecone pypdf pinecone-client python-dotenv
```

Root `.env` (gitignored):

```
OPENAI_API_KEY=...
PINECONE_API_KEY=...
```

Put PDFs in `./docs/`, then:

```bash
./venv/bin/python sentry_query.py
```

Edit the `query = "..."` line in `sentry_query.py` to try other questions.

**Note:** Every run re-embeds and re-upserts. OpenAI + Pinecone see your content; don’t index data you’re not allowed to send to vendors.

MIT — see `LICENSE`.
