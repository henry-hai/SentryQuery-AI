"""Which indexed filing belongs to which company.

Every chunk in Pinecone already carries the path of the PDF it came from in its
`source` metadata field, so a company filter needs no re-ingestion. This file
maps each company to the names a claim might use for it and to the source paths
of its filings. When a claim names an indexed company, retrieval is filtered to
those paths and a chunk from any other filing cannot come back.

To index a new filing, drop the PDF in ./docs/, re-ingest, and add an entry here.
A company with no entry is simply never filtered on.
"""
import re

COMPANY_SOURCES: dict[str, dict[str, list[str]]] = {
    "costco": {
        "names": ["costco"],
        "sources": ["docs/costco-10k-2025.pdf"],
    },
    "southwest": {
        "names": ["southwest airlines", "southwest"],
        "sources": ["docs/southwest-10k-2025.pdf"],
    },
    "deere": {
        "names": ["john deere", "deere"],
        "sources": ["docs/deere-10k-2025.pdf"],
    },
}


def companies_named(text: str, mapping: dict | None = None) -> list[str]:
    """The indexed companies a piece of text names, in mapping order.

    Matching is case-insensitive on whole words, so "Deere's" matches deere
    and "costcoish" does not match costco.
    """
    mapping = COMPANY_SOURCES if mapping is None else mapping
    found: list[str] = []
    for company, entry in mapping.items():
        for name in entry["names"]:
            if re.search(rf"\b{re.escape(name)}\b", text, flags=re.IGNORECASE):
                found.append(company)
                break
    return found


def source_filter(companies: list[str], mapping: dict | None = None) -> dict | None:
    """A Pinecone metadata filter restricting retrieval to these companies'
    filings, or None when no company is named, which means search everything."""
    mapping = COMPANY_SOURCES if mapping is None else mapping
    sources: list[str] = []
    for company in companies:
        for path in mapping.get(company, {}).get("sources", []):
            if path not in sources:
                sources.append(path)
    if not sources:
        return None
    return {"source": {"$in": sources}}


_STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "by", "for", "from", "in", "is",
    "of", "on", "or", "the", "to", "was", "were", "with",
}


def strip_company_names(text: str, companies: list[str],
                        mapping: dict | None = None) -> str:
    """Remove filtered company names unless the search would lose its meaning."""
    mapping = COMPANY_SOURCES if mapping is None else mapping
    names = [
        name for company in companies
        for name in mapping.get(company, {}).get("names", [])
    ]
    stripped = text
    for name in sorted(names, key=len, reverse=True):
        pattern = rf"\b{re.escape(name)}\b(?:['’]s|['’])?"
        stripped = re.sub(pattern, " ", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    words = re.findall(r"[a-z0-9]+", stripped.lower())
    return stripped if any(word not in _STOPWORDS for word in words) else text
