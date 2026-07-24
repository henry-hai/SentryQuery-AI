"""Test bootstrap: make the top-level modules importable with no real keys.

The project modules (config, agent, graph, schema) live at the repo root and the
eval harness lives in evals/, so both paths go on sys.path before collection.

Importing config.py constructs the Pinecone client (and the OpenAI embeddings),
which now raise if no API key is present, even though these tests never make a
network call. So we set clearly-fake placeholder keys first, and only if the
environment does not already provide real ones. These are not credentials, and
no test in this suite contacts OpenAI, Pinecone, or Tavily.
"""
import os
import sys
from pathlib import Path

for var in ("PINECONE_API_KEY", "OPENAI_API_KEY", "TAVILY_API_KEY"):
    os.environ.setdefault(var, "ci-placeholder-not-a-real-key")

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "evals"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
