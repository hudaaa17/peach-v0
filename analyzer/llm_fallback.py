"""
Stage 8: LLM fallback — the last resort for external-call candidates that
literal_check, constprop_check, and the Joern escalation stage all failed
to resolve a host for.

A small local instruction-tuned code model (Qwen2.5-Coder-1.5B-Instruct,
per project spec) is shown the call site and a few surrounding lines and
asked to *infer* the likely downstream service from naming and context.
This is explicitly a guess, not a proven trace, so:

  * every edge this stage touches is marked `needs_review = True`
  * a short `review_reason` is attached explaining why, so the human
    reviewer isn't just told "trust me" — for a resolved guess, and for a
    still-unresolved edge, so it's clear the pipeline gave up cleanly
    rather than silently dropping it.

The model is loaded lazily and once per process. If `transformers`/`torch`
aren't installed, no weights are cached locally, or generation fails for
any reason, the stage degrades to flagging the edge for manual review
instead of raising — a demo/offline environment shouldn't crash the whole
analysis over a missing multi-GB model download.
"""
import json
import os
import re
import threading
import time

from .obs import get_logger, log

LOG = get_logger("slm")

MODEL_ID = os.environ.get("PEACH_SLM_MODEL_ID", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
MAX_EDGES_PER_RUN = int(os.environ.get("PEACH_SLM_MAX_EDGES", "25"))
MAX_NEW_TOKENS = 120
CONTEXT_LINES_BEFORE = 4
CONTEXT_LINES_AFTER = 2

_SYSTEM_PROMPT = (
    "You are a static-analysis assistant helping map microservice dependencies "
    "in a codebase. You will be shown one unresolved external call site: a call "
    "to a networking/HTTP client whose target could not be determined by static "
    "tracing. Using only naming and surrounding context, infer the most likely "
    "name of the downstream service or host being called.\n"
    "Respond with ONLY a single compact JSON object on one line, no prose, no "
    "markdown fences, matching exactly this shape:\n"
    '{"service_name": string or null, "confidence": "low" | "medium" | "high", '
    '"reasoning": short string}\n'
    "Set service_name to null if you cannot infer anything meaningful — a "
    "guess with no basis in the shown context is worse than admitting you "
    "don't know."
)

_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)

_lock = threading.Lock()
_pipe = None          # (tokenizer, model) once loaded
_load_error = None     # sticky error message if loading ever failed


def _get_pipe():
    """Lazily load the model once per process. Thread-safe, and memoizes
    failure too so we don't retry a doomed load on every edge."""
    global _pipe, _load_error
    if _pipe is not None or _load_error is not None:
        return _pipe
    with _lock:
        if _pipe is not None or _load_error is not None:
            return _pipe
        t0 = time.time()
        log(LOG, "info", "loading SLM weights (first run downloads ~3GB)", model=MODEL_ID)
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float32,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
            )
            model.eval()
            _pipe = (tokenizer, model)
            log(LOG, "info", "SLM ready", model=MODEL_ID,
                device=str(model.device), load_seconds=round(time.time() - t0, 2))
        except Exception as exc:  # noqa: BLE001 - any load failure degrades gracefully
            _load_error = f"{type(exc).__name__}: {exc}"
            _pipe = None
            log(LOG, "warning", "SLM unavailable; stage will flag for manual review",
                model=MODEL_ID, error=_load_error,
                load_seconds=round(time.time() - t0, 2))
    return _pipe


def _build_context(edge, records_by_path):
    rec = records_by_path.get(edge.file)
    if not rec or not rec.source_lines:
        return ""
    start = max(0, edge.line - 1 - CONTEXT_LINES_BEFORE)
    end = min(len(rec.source_lines), edge.line + CONTEXT_LINES_AFTER)
    snippet_lines = rec.source_lines[start:end]
    numbered = [f"{start + i + 1}: {line}" for i, line in enumerate(snippet_lines)]
    return "\n".join(numbered)


def _build_user_prompt(edge, records_by_path):
    snippet = _build_context(edge, records_by_path)
    return (
        f"File: {edge.file}\n"
        f"Enclosing function: {edge.caller}\n"
        f"Unresolved call: {edge.callee_expr}({edge.arg_text})\n"
        f"Detected client type: {edge.external_pattern or 'unknown'}\n"
        f"Surrounding code:\n{snippet or '(no source available)'}\n"
    )


def _extract_json(text: str):
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    name = data.get("service_name")
    if isinstance(name, str):
        name = name.strip().strip('"').strip("'")
        if name.lower() in ("null", "none", ""):
            name = None
    else:
        name = None
    data["service_name"] = name
    if data.get("confidence") not in ("low", "medium", "high"):
        data["confidence"] = "low"
    return data


def _infer(pipe, edge, records_by_path):
    tokenizer, model = pipe
    import torch

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(edge, records_by_path)},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        return _extract_json(text)
    except Exception as exc:  # noqa: BLE001 - a single bad generation shouldn't kill the batch
        log(LOG, "warning", "generation failed for edge",
            file=edge.file, line=edge.line, error=f"{type(exc).__name__}: {exc}")
        return None


def check_llm_fallback(edges, records_by_path, ctx=None):
    """Mutates edges in place. Returns the list of edges sent to this stage
    (whether or not the model could resolve them)."""
    candidates = [e for e in edges if e.status == "external_candidate" and not e.host]
    if not candidates:
        log(LOG, "info", "nothing reached the SLM fallback; earlier stages resolved everything")
        return []

    log(LOG, "info", "SLM fallback starting", unresolved=len(candidates), cap=MAX_EDGES_PER_RUN)
    if ctx is not None and len(candidates) > MAX_EDGES_PER_RUN:
        ctx.note("warning", "slm",
                 f"{len(candidates)} unresolved calls exceed the per-run cap of "
                 f"{MAX_EDGES_PER_RUN}; the remainder were flagged without inference.")

    pipe = _get_pipe()
    touched = []
    resolved = 0
    skipped = 0
    failed = 0
    infer_seconds = 0.0

    for i, edge in enumerate(candidates):
        edge.sent_to_llm = True
        edge.needs_review = True
        touched.append(edge)

        if i >= MAX_EDGES_PER_RUN:
            edge.review_reason = (
                f"Skipped SLM fallback (cap of {MAX_EDGES_PER_RUN} calls per run reached); "
                "needs manual review."
            )
            skipped += 1
            continue

        if pipe is None:
            edge.review_reason = (
                f"SLM ({MODEL_ID}) unavailable in this environment "
                f"({_load_error or 'not loaded'}); needs manual review."
            )
            skipped += 1
            continue

        t0 = time.time()
        result = _infer(pipe, edge, records_by_path)
        elapsed = time.time() - t0
        infer_seconds += elapsed
        log(LOG, "debug", "inference complete", file=edge.file, line=edge.line,
            seconds=round(elapsed, 2), got_result=bool(result))

        if result and result.get("service_name"):
            edge.host = result["service_name"]
            edge.literal_method = "llm_inferred"
            confidence = result.get("confidence", "low")
            reasoning = (result.get("reasoning") or "").strip()
            edge.review_reason = (
                f"SLM-inferred ({confidence} confidence)"
                + (f": {reasoning}" if reasoning else "")
                + " — not a proven trace, verify before trusting."
            )
            resolved += 1
            if ctx is not None:
                ctx.bump(f"slm.confidence.{confidence}")
            log(LOG, "info", "SLM inferred a target", file=edge.file, line=edge.line,
                host=edge.host, confidence=confidence)
        else:
            edge.review_reason = (
                "SLM could not confidently infer a target from context; needs manual review."
            )
            failed += 1

    if ctx is not None:
        ctx.bump("slm.sent", len(touched))
        ctx.bump("slm.resolved", resolved)
        ctx.bump("slm.skipped", skipped)
        ctx.bump("slm.no_inference", failed)
        ctx.bump("slm.inference_seconds", int(infer_seconds))
        if resolved:
            ctx.note("info", "slm",
                     f"{resolved} external call(s) were resolved only by SLM inference. "
                     "These are guesses flagged needs_review, not proven traces — a high "
                     "number here means the deterministic stages are underperforming.")

    log(LOG, "info", "SLM fallback complete", sent=len(touched), resolved=resolved,
        skipped=skipped, no_inference=failed,
        inference_seconds=round(infer_seconds, 2))
    return touched
