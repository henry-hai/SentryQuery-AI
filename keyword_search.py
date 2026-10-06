"""In-memory BM25 search over the exported PDF chunks."""
import gzip
import json
import math
import re
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document

from chunks import CHUNK_PATH

K1 = 1.5
B = 0.75
TOKEN = re.compile(r"\d[\d,]*(?:\.\d+)?%?|[a-z]+(?:'[a-z]+)?", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Keep figures such as 8,099 and 10% together."""
    return TOKEN.findall(text.lower())


class KeywordIndex:
    """A small Okapi BM25 index built once from the local chunk file."""

    def __init__(self, documents: list[Document]):
        self.documents = documents
        self.lengths = []
        self.postings = defaultdict(list)
        for index, doc in enumerate(documents):
            counts = Counter(tokenize(doc.page_content))
            self.lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                self.postings[term].append((index, frequency))
        self.average_length = sum(self.lengths) / len(documents) if documents else 0

    def search(self, query: str, allowed_sources: list[str] | None = None,
               k: int = 8) -> list[Document]:
        """Return the highest scoring chunks within the optional source filter."""
        if not self.documents or k <= 0:
            return []
        allowed = set(allowed_sources) if allowed_sources is not None else None
        scores = defaultdict(float)
        total = len(self.documents)
        for term in set(tokenize(query)):
            postings = self.postings.get(term, [])
            if not postings:
                continue
            idf = math.log(1 + (total - len(postings) + 0.5) / (len(postings) + 0.5))
            for index, frequency in postings:
                if allowed is not None and self.documents[index].metadata["source"] not in allowed:
                    continue
                length = self.lengths[index]
                denominator = frequency + K1 * (1 - B + B * length / self.average_length)
                scores[index] += idf * frequency * (K1 + 1) / denominator
        ranked = sorted(scores, key=lambda index: (-scores[index], index))[:k]
        return [
            Document(
                page_content=self.documents[index].page_content,
                metadata=dict(self.documents[index].metadata),
            )
            for index in ranked
        ]


@lru_cache(maxsize=1)
def load_index(path: Path = CHUNK_PATH) -> KeywordIndex | None:
    """Load the chunk file on first search, with a single warning if absent."""
    if not path.exists():
        print(f"Warning: keyword chunk file missing at {path}", file=sys.stderr)
        return None
    with gzip.open(path, "rt", encoding="utf-8") as source:
        docs = [
            Document(
                page_content=row["text"],
                metadata={"source": row["source"], "page": row["page"]},
            )
            for row in (json.loads(line) for line in source)
        ]
    return KeywordIndex(docs)


def search(query: str, allowed_sources: list[str] | None = None,
           k: int = 8) -> list[Document]:
    """Search the local index, or return no hits when it is unavailable."""
    index = load_index()
    return index.search(query, allowed_sources, k) if index else []
