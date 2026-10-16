"""Embedding wrapper for the RAG index.

Defaults to a local, open-source, code-aware embedding model — consistent
with the rest of this pipeline (SCIP, ast-grep, Joern, Qwen2.5-Coder are all
local tooling already). No API keys, no network calls at query time.
"""

from __future__ import annotations

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "jinaai/jina-embeddings-v2-base-code"
FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # lighter, no trust_remote_code


class Embedder:
    def __init__(self, model_name: str = None, device: str = None, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or DEFAULT_MODEL
        self.batch_size = batch_size
        t0 = time.time()
        logger.info("loading embedding model: %s", self.model_name)
        try:
            self.model = SentenceTransformer(
                self.model_name, trust_remote_code=True, device=device)
        except Exception:
            # code-specific model failed to load (missing deps, no network
            # to fetch weights, etc.) — fall back rather than hard-fail the
            # whole indexing run.
            logger.warning("failed to load %s, falling back to %s",
                            self.model_name, FALLBACK_MODEL, exc_info=True)
            self.model_name = FALLBACK_MODEL
            self.model = SentenceTransformer(self.model_name, device=device)
        logger.info("embedding model ready: %s (%.1fs)", self.model_name, time.time() - t0)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        t0 = time.time()
        vectors = self.model.encode(
            texts, batch_size=self.batch_size, show_progress_bar=False,
            normalize_embeddings=True,
        )
        elapsed = time.time() - t0
        logger.info("embedded %d chunks in %.2fs (%.1f/s)",
                    len(texts), elapsed, len(texts) / elapsed if elapsed > 0 else 0)
        return vectors.tolist()