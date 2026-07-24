"""Test bootstrap: make the top-level modules importable.

The project modules (config, agent, graph, schema) live at the repo root and the
eval harness lives in evals/, so both paths go on sys.path before collection.
None of these tests hit OpenAI, Pinecone, or Tavily, so they run with no keys.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "evals"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
