"""Shared configuration for SentryQuery.

Holds the settings more than one module needs: the Pinecone client and index,
the embedding model, the corpus-neutral system prompt, the model names, and the
Critic revision cap. Importing this module loads .env.
"""
import os

from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

# Load OPENAI_API_KEY, PINECONE_API_KEY, and TAVILY_API_KEY from .env
# so secrets stay out of source control.
load_dotenv()

# A single Pinecone client instance is shared across ingestion and query modes.
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# The Pinecone index must be pre-created with dimension 1536 (matches
# text-embedding-3-small) and cosine similarity.
INDEX_NAME = "sentry-index"

# GPT-4o for the Researcher (accurate, factual grounded Q&A); gpt-4o-mini for the
# Critic - groundedness checking is a narrower verification task, so the cheaper
# model suffices at a fraction of the per-call cost. Both run at temperature=0.
RESEARCHER_MODEL = "gpt-4o"
CRITIC_MODEL = "gpt-4o-mini"

# How many REVISE -> Researcher refinement passes the Critic loop may take.
MAX_REVISIONS = 1

# The system prompt scopes the agent to the indexed documents, instructs it
# which tool to prefer for which kind of question, and tells it to refuse
# off-topic queries. This is the primary prompt-engineering surface in the app.
# It is deliberately corpus-neutral: it never names a specific company or file,
# so the same code works over whatever PDFs are ingested into ./docs/.
SYSTEM_PROMPT = """You are SentryQuery, an assistant that answers questions grounded in a set of indexed enterprise documents.

Tool routing:
- For questions whose answer could plausibly be found in the indexed documents
  (the organizations they cover, their business, financials, strategy,
  operations, products, or policies): use search_documents FIRST.
- For questions that require live or recent information not contained in the
  documents (current news, today's events, current market data): use
  web_search. Do not answer from memory - always call the tool, and summarize
  the results that come back. Do not say "I couldn't find" if the tool returned
  any content - report what it returned.
- For questions unrelated to the indexed documents and their subject matter
  (general chit-chat, the weather, unrelated coding help, and so on): politely
  refuse and explain that you only answer questions about the indexed documents.

Answering style:
- Be concise. Ground every claim in retrieved content.
- If neither tool returns useful information, say
  "I don't have that information in the indexed documents or available web
  sources." Do not speculate or rely on outside knowledge.
- When quoting specifics, prefer short verbatim phrases over paraphrases.
"""

# text-embedding-3-small produces 1536-dim vectors and is ~5x cheaper
# than ada-002 with comparable retrieval quality on MTEB benchmarks.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
