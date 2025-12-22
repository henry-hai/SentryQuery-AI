# SentryQuery-AI

A Retrieval-Augmented Generation (RAG) engine built for technical document analysis. This system utilizes LangChain Expression Language (LCEL) to pipe vectorized data from Pinecone into GPT-4o for grounded answering.

## Setup and Execution
Run this entire block in your terminal to initialize the environment, install 2025-standard dependencies, and execute the program:

```bash
python3 -m venv venv

source venv/bin/activate

pip install -U langchain langchain-community langchain-openai langchain-pinecone pypdf pinecone-client python-dotenv

mkdir -p docs

./venv/bin/python sentry_query.py
```


