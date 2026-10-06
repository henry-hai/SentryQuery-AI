"""Load the indexed PDF chunks and export them for local keyword search."""
import gzip
import io
import json
from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_PATH = Path(__file__).resolve().parent / "data" / "chunks.jsonl.gz"


def load_chunks():
    """Use the same PDF loading and splitting settings as ingestion."""
    docs = PyPDFDirectoryLoader("./docs").load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)


def write_chunks(path: Path = CHUNK_PATH) -> int:
    """Write source, zero-based page, and text without contacting the index."""
    splits = load_chunks()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw, gzip.GzipFile(
        filename="", fileobj=raw, mode="wb", mtime=0
    ) as compressed, io.TextIOWrapper(compressed, encoding="utf-8") as output:
        for doc in splits:
            row = {
                "source": doc.metadata["source"],
                "page": int(doc.metadata["page"]),
                "text": doc.page_content,
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(splits)


if __name__ == "__main__":
    count = write_chunks()
    print(f"Wrote {count} chunks to {CHUNK_PATH}")
