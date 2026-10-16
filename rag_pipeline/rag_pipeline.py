"""Builds the RAG index from the same shallow clone the graph stage uses.

Usage — single call, one clone, both outputs:

    from rag_pipeline.rag_pipeline import analyze_repo_with_rag

    graph, rag = analyze_repo_with_rag(
        "https://github.com/org/repo", progress=print, run_id="run123")

    # graph  -> exactly what analyze_repo() already returned (unchanged)
    # rag    -> RagIndexer, holding the persisted Chroma collection for
    #           this run; step 3 (SLM traversal) queries it directly.

Or, if you're calling analyze_repo() yourself already, just pass a hook:

    indexer = RagIndexer(persist_dir="./peach_rag_db")
    graph = analyze_repo(url, run_id=run_id, rag_hook=indexer.as_hook(run_id))
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from .chunker import chunk_repo
from .embedder import Embedder
from .vector_store import VectorStore

# analyzer/ and rag_pipeline/ are sibling packages under the project root,
# not the same package — so this is an absolute import, not a relative one.
# Requires both to be importable from the root (each has an __init__.py,
# and the process is run from/aware of the project root — e.g. `python -m
# app.main`, or the root is on PYTHONPATH).
from analyzer.pipeline import analyze_repo

logger = logging.getLogger(__name__)

DEFAULT_PERSIST_DIR = Path.cwd() / ".peach_rag_db"


class RagIndexer:
    def __init__(self, persist_dir=None, embed_model: str = None, batch_size: int = 64):
        self.persist_dir = Path(persist_dir or DEFAULT_PERSIST_DIR)
        self.embedder = Embedder(model_name=embed_model)
        self.batch_size = batch_size
        self._last_collection = None

    def index(self, repo_root, code_records, config_records=None, run_id: str = None,
              ctx=None) -> dict:
        """Chunk, embed, and store. Returns a summary dict (also folded
        into the pipeline's run-report counters when called via the hook)."""
        t0 = time.time()
        collection_name = _collection_name(run_id)
        logger.info("RAG index starting for run_id=%s -> collection '%s'",
                    run_id, collection_name)
        store = VectorStore(self.persist_dir, collection_name=collection_name)

        chunks = chunk_repo(repo_root, code_records, config_records)
        if not chunks:
            logger.warning("no chunks produced for run_id=%s — nothing to index", run_id)
            self._last_collection = collection_name
            return {"chunks": 0, "collection": collection_name}

        n_batches = (len(chunks) + self.batch_size - 1) // self.batch_size
        for i, start in enumerate(range(0, len(chunks), self.batch_size), start=1):
            batch = chunks[start:start + self.batch_size]
            logger.info("embedding batch %d/%d (%d chunks)", i, n_batches, len(batch))
            embeddings = self.embedder.embed([c.code for c in batch])
            store.upsert(batch, embeddings)

        by_kind: dict = {}
        for c in chunks:
            by_kind[c.kind] = by_kind.get(c.kind, 0) + 1

        self._last_collection = collection_name
        summary = {"chunks": len(chunks), "collection": collection_name}
        summary.update({f"kind_{k}": v for k, v in by_kind.items()})
        logger.info("RAG index done for run_id=%s: %d chunks in %.2fs (%s)",
                    run_id, len(chunks), time.time() - t0,
                    ", ".join(f"{k}={v}" for k, v in sorted(by_kind.items())))
        return summary

    def index_start(self, repo_root, code_records, config_records=None, run_id: str = None,
                     ctx=None, status_store: dict = None) -> dict:
        """Non-blocking version of index(). Chunking has to happen here,
        synchronously, because it's the only part that needs the repo files
        actually on disk — and pipeline.py deletes repo_root right after
        analyze_repo() returns. Embedding + storage don't need the repo at
        all (chunks already hold their source text in memory), so that part
        runs in a background thread and the caller gets control back
        immediately after chunking, instead of waiting ~seconds for
        embedding to finish.

        If `status_store` is given (a plain dict), this writes
        status_store[run_id] = {...} as it progresses — "embedding" as soon
        as chunking finishes, then "ready" or "error" once the background
        thread completes. app.py's /api/rag-status/<run_id> endpoint reads
        this dict directly to answer polling requests.
        """
        t0 = time.time()
        collection_name = _collection_name(run_id)
        chunks = chunk_repo(repo_root, code_records, config_records)
        logger.info("chunked %d chunks for run_id=%s in %.2fs, handing off to background embedding",
                    len(chunks), run_id, time.time() - t0)

        if not chunks:
            logger.warning("no chunks produced for run_id=%s — nothing to index", run_id)
            if status_store is not None:
                status_store[run_id] = {
                    "status": "ready", "chunks": 0, "collection": collection_name,
                    "by_kind": {}, "elapsed_seconds": round(time.time() - t0, 2), "error": None,
                }
            self._last_collection = collection_name
            return {"chunks": 0, "collection": collection_name, "status": "ready"}

        if status_store is not None:
            status_store[run_id] = {
                "status": "embedding", "chunks": len(chunks), "collection": collection_name,
                "by_kind": None, "elapsed_seconds": None, "error": None,
            }

        def _embed_and_store():
            try:
                store = VectorStore(self.persist_dir, collection_name=collection_name)
                n_batches = (len(chunks) + self.batch_size - 1) // self.batch_size
                for i, start in enumerate(range(0, len(chunks), self.batch_size), start=1):
                    batch = chunks[start:start + self.batch_size]
                    logger.info("embedding batch %d/%d (%d chunks) [run_id=%s]",
                                i, n_batches, len(batch), run_id)
                    embeddings = self.embedder.embed([c.code for c in batch])
                    store.upsert(batch, embeddings)

                by_kind: dict = {}
                for c in chunks:
                    by_kind[c.kind] = by_kind.get(c.kind, 0) + 1
                elapsed = time.time() - t0
                logger.info("RAG index ready for run_id=%s: %d chunks in %.2fs (%s)",
                            run_id, len(chunks), elapsed,
                            ", ".join(f"{k}={v}" for k, v in sorted(by_kind.items())))
                self._last_collection = collection_name
                if status_store is not None:
                    status_store[run_id] = {
                        "status": "ready", "chunks": len(chunks), "collection": collection_name,
                        "by_kind": by_kind, "elapsed_seconds": round(elapsed, 2), "error": None,
                    }
            except Exception as e:  # noqa: BLE001 - report via status_store, don't crash the thread silently
                logger.exception("RAG indexing failed for run_id=%s", run_id)
                if status_store is not None:
                    status_store[run_id] = {
                        "status": "error", "chunks": len(chunks), "collection": collection_name,
                        "by_kind": None, "elapsed_seconds": round(time.time() - t0, 2), "error": str(e),
                    }

        threading.Thread(target=_embed_and_store, daemon=True,
                          name=f"rag-embed-{run_id}").start()

        return {"chunks": len(chunks), "collection": collection_name, "status": "embedding"}

    def retrieve(self, run_id: str, query_text: str, k: int = 8) -> list[dict]:
        """Embed the question and pull the k nearest chunks for this run's
        collection. Returns [{id, code, metadata, distance}, ...], nearest
        first. Used by the /api/chat endpoint to ground the SLM's answer."""
        store = VectorStore(self.persist_dir, collection_name=_collection_name(run_id))
        query_vec = self.embedder.embed([query_text])[0]
        raw = store.query(query_vec, k=k)

        ids = (raw.get("ids") or [[]])[0]
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]

        results = [
            {"id": ids[i], "code": docs[i], "metadata": metas[i],
             "distance": dists[i] if i < len(dists) else None}
            for i in range(len(ids))
        ]
        logger.info("retrieved %d chunks for run_id=%s query=%r", len(results), run_id, query_text)
        return results

    def as_hook(self, run_id: str = None, status_store: dict = None):
        """Adapter matching the (repo_root, code_records, config_records,
        ctx=...) signature analyze_repo() calls rag_hook with. Uses the
        non-blocking index_start() so analyze_repo() (and therefore the
        /api/analyze response) doesn't wait on embedding."""
        def _hook(repo_root, code_records, config_records, ctx=None):
            return self.index_start(repo_root, code_records, config_records,
                                     run_id=run_id, ctx=ctx, status_store=status_store)
        return _hook

    def get_store(self, run_id: str = None) -> VectorStore:
        return VectorStore(self.persist_dir, collection_name=_collection_name(run_id))


def _collection_name(run_id: str = None) -> str:
    return f"repo_{run_id}" if run_id else "code_chunks_default"


def analyze_repo_with_rag(url: str, progress=None, run_id: str = None,
                           persist_dir=None, embed_model: str = None):
    """Runs the existing graph pipeline unchanged, and — off the same
    shallow clone, before it's cleaned up — builds the semantic RAG index
    alongside it. Returns (graph, indexer)."""
    indexer = RagIndexer(persist_dir=persist_dir, embed_model=embed_model)
    graph = analyze_repo(url, progress=progress, run_id=run_id,
                          rag_hook=indexer.as_hook(run_id))
    return graph, indexer