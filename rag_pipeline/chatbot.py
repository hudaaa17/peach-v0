"""Qwen2.5 chat wrapper for the RAG-grounded chatbot — via Ollama.

Previously this loaded raw fp16/bf16 weights through `transformers`, which
on a CPU with no CUDA and no hardware bf16 support (e.g. a Ryzen 5500U)
runs an unaccelerated fallback path — that's what produced the 843s-for-
10-tokens results. This version instead talks to a local Ollama server,
which runs a quantized GGUF build of the model. Same model family
(Qwen2.5-Coder), same public interface (QwenChat().answer(...)), just a
runtime that's actually usable on CPU-only hardware.

Setup, one-time:
  1. Install Ollama: https://ollama.com/download (Windows/Mac/Linux)
  2. It runs as a background service after install — nothing to launch
     manually. Confirm with `ollama list` in a terminal.
  3. The model is pulled automatically on first use by this module (see
     _ensure_model_available below), or pull it yourself ahead of time:
         ollama pull qwen2.5-coder:1.5b

"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen2.5-coder:1.5b"
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_KEEP_ALIVE = "30m"  # keep the model resident between chat turns instead
                             # of Ollama's 5-minute default, to avoid reload lag
                             # mid-conversation

SYSTEM_PROMPT = (
    "You are Peach's codebase assistant. Answer questions about this "
    "repository using ONLY the code snippets given to you as context. Each "
    "snippet is labeled with its file path, line range, and the "
    "function/class/method it came from. If the context doesn't contain "
    "enough information to answer confidently, say so explicitly instead "
    "of guessing. When you make a specific claim, name the file and "
    "function/class it came from. Keep answers concise."
)


class QwenChat:
    def __init__(self, model_name: str = None, host: str = None,
                 auto_pull: bool = True, keep_alive: str = None):
        import ollama

        self.model_name = model_name or DEFAULT_MODEL
        self.host = host or DEFAULT_HOST
        self.keep_alive = keep_alive or DEFAULT_KEEP_ALIVE
        self._client = ollama.Client(host=self.host)
        self._ensure_model_available(auto_pull=auto_pull)

    def _ensure_model_available(self, auto_pull: bool) -> None:
        try:
            local = self._client.list().get("models", [])
        except Exception as e:
            raise RuntimeError(
                f"Could not reach Ollama at {self.host}. Is it installed and "
                f"running? (https://ollama.com/download — it runs as a "
                f"background service once installed, nothing to launch "
                f"manually). Original error: {e}"
            ) from e

        local_names = {m.get("model") or m.get("name") for m in local}
        if self.model_name in local_names:
            logger.info("chat model already pulled: %s", self.model_name)
            return

        if not auto_pull:
            raise RuntimeError(
                f"Model '{self.model_name}' isn't pulled yet. Run: "
                f"ollama pull {self.model_name}"
            )

        logger.info("pulling chat model (first run only, ~1GB): %s", self.model_name)
        t0 = time.time()
        last_status = None
        for progress in self._client.pull(self.model_name, stream=True):
            status = progress.get("status")
            if status != last_status:
                logger.info("pull: %s", status)
                last_status = status
        logger.info("model pulled: %s (%.1fs)", self.model_name, time.time() - t0)

    def answer(self, question: str, context_chunks: list[dict], max_tokens: int = 512) -> str:
        context_block = _format_context(context_chunks)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {question}"},
        ]

        t0 = time.time()
        response = self._client.chat(
            model=self.model_name,
            messages=messages,
            options={"num_predict": max_tokens},
            keep_alive=self.keep_alive,
        )
        wall = time.time() - t0
        answer = (response.get("message") or {}).get("content", "").strip()

        # Ollama reports its own generation stats — more meaningful than our
        # wall-clock timing alone, since wall time also includes prompt
        # processing (prefill), which scales with how much context we sent.
        eval_count = response.get("eval_count")
        eval_ns = response.get("eval_duration")
        prompt_count = response.get("prompt_eval_count")
        if eval_count and eval_ns:
            tok_per_s = eval_count / (eval_ns / 1e9)
            logger.info(
                "chat generation done in %.2fs wall (%d tokens @ %.1f tok/s, "
                "%s prompt tokens, %d context chunks)",
                wall, eval_count, tok_per_s, prompt_count, len(context_chunks),
            )
        else:
            logger.info("chat generation done in %.2fs (%d context chunks)",
                        wall, len(context_chunks))
        return answer


def _format_context(chunks: list[dict]) -> str:
    if not chunks:
        return "(no relevant context found in this repo)"
    parts = []
    for c in chunks:
        meta = c.get("metadata", {}) or {}
        label = meta.get("qualified_name") or meta.get("repo_path") or c.get("id")
        loc = f"{meta.get('repo_path', '?')}:{meta.get('start_line', '?')}-{meta.get('end_line', '?')}"
        parts.append(f"### {label} ({loc})\n```\n{c.get('code', '')}\n```")
    return "\n\n".join(parts)