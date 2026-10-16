"""Persistent local vector store for code chunks (ChromaDB).

Local-first to match the rest of the stack — no hosted service, no API key,
data lives on disk at `persist_dir` and survives across runs so step 3 (the
SLM) can query it without re-indexing.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, persist_dir, collection_name: str = "code_chunks"):
        import chromadb

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"})
        logger.info("vector store ready: %s/%s (%d existing)",
                    self.persist_dir, collection_name, self.collection.count())

    def upsert(self, chunks, embeddings):
        if not chunks:
            return
        self.collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.code for c in chunks],
            metadatas=[_metadata(c) for c in chunks],
        )
        logger.info("upserted %d chunks into '%s' (collection now has %d)",
                    len(chunks), self.collection.name, self.collection.count())

    def query(self, query_embedding, k: int = 8, where: dict = None):
        return self.collection.query(
            query_embeddings=[query_embedding], n_results=k, where=where)

    def count(self) -> int:
        return self.collection.count()


def _metadata(chunk) -> dict:
    return {
        "repo_path": chunk.repo_path,
        "language": chunk.language or "",
        "kind": chunk.kind,
        "name": chunk.name,
        "qualified_name": chunk.qualified_name,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "parent_qualified_name": chunk.parent_qualified_name or "",
        "oversized": bool(chunk.metadata.get("oversized", False)),
    }