"""
Stage 8: SLM fallback — the last resort for external-call candidates that
the literal, const-prop, config and Joern stages all failed to resolve.

This stage is the only one in the pipeline whose output is a *guess*. That
does not make it the one stage where correctness is optional: a guess that
reaches the graph as a service node is indistinguishable, to everything
downstream, from a host proven by data flow. The previous version wrote
whatever string the model emitted straight into `edge.host` — prose,
`"unknown"`, a hostname the model invented that appears nowhere in the
repo — and the only defence was a `needs_review` flag. So the failure mode
wasn't "sometimes wrong", it was "confidently populates the dependency
graph with names that don't exist".

What this version does instead:

1. **Grounded, candidate-constrained inference.** The prompt no longer
   asks an open question. It carries the evidence the deterministic stages
   already produced — the identifiers involved, their in-file assignments,
   the imports, the client library detected, and why the trace stage gave
   up — plus a *candidate list* built from the repo itself: hosts already
   resolved on other edges, hosts and service names declared in config
   files, and env-var keys seen at this call site. The model is asked to
   pick a candidate, or to answer with a name it can point at in the shown
   context, or to return null.
2. **Every answer is verified before it is believed.** `_validate` rejects
   anything that isn't a syntactically usable host/service token, and
   `_ground` requires the answer to be a known candidate or to literally
   appear in the context the model was shown. An answer that survives
   neither is recorded as a hallucination (counted, logged) and the edge
   stays unresolved. This is the difference between "the model is allowed
   to guess" and "the model is allowed to invent".
3. **Real decoding discipline.** Prompts are deduplicated and batched with
   left padding, the input is truncated to the model's own context limit
   instead of overflowing it, generation is seeded and greedy, the
   assistant turn is pre-filled with `{"service_name":` so the model
   cannot open with prose, and it stops at the closing brace instead of
   always burning the full token budget.
4. **Honest bookkeeping.** `sent_to_llm` is now set only on edges actually
   sent to the model, and a `review_reason` written by an earlier stage
   (e.g. "Joern did not analyse this call site") is preserved rather than
   overwritten, so the reviewer sees the whole history and not just the
   last thing that touched the edge.

Degrading when `transformers`/`torch` aren't installed is still correct
behaviour here — unlike stage 7, this stage's product is an unproven
inference, so its absence costs nothing that was ever trusted. It is
counted and reported, never silent.
"""
import hashlib
import json
import os
import re
import threading
import time

from .obs import get_logger, log, TRACE_EDGES

LOG = get_logger("slm")

MODEL_ID = os.environ.get("PEACH_SLM_MODEL_ID", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
MAX_EDGES_PER_RUN = int(os.environ.get("PEACH_SLM_MAX_EDGES", "25"))
BATCH_SIZE = int(os.environ.get("PEACH_SLM_BATCH_SIZE", "4"))
MAX_NEW_TOKENS = 96
MAX_INPUT_TOKENS = int(os.environ.get("PEACH_SLM_MAX_INPUT_TOKENS", "2048"))
SEED = int(os.environ.get("PEACH_SLM_SEED", "0"))
CONTEXT_LINES_BEFORE = 12
CONTEXT_LINES_AFTER = 4
MAX_CANDIDATES = 40

#: The model is pre-filled with this, so its first emitted token is a JSON
#: value rather than "Sure! Here's the analysis:".
_PREFILL = '{"service_name":'

_SYSTEM_PROMPT = (
    "You are a static-analysis assistant mapping microservice dependencies. "
    "You are shown one external call site whose destination could not be "
    "determined by static tracing, the evidence collected about it, and a list "
    "of candidate destinations found elsewhere in the same repository.\n"
    "Rules:\n"
    "1. Prefer a candidate from the CANDIDATES list when one fits.\n"
    "2. Otherwise you may answer with a host, URL or service name that appears "
    "literally in the shown code or evidence.\n"
    "3. Never invent a name that is not present in what you were shown. If "
    "nothing in the context supports an answer, return null. A null is a "
    "correct answer here; an invented hostname is not.\n"
    "4. service_name must be a bare host, URL or service identifier - no "
    "sentences, no explanation inside the name.\n"
    "Respond with ONLY one compact JSON object on one line, no prose, no "
    "markdown fences:\n"
    '{"service_name": string or null, "confidence": "low"|"medium"|"high", '
    '"evidence": short string quoting what in the context supports it}'
)

#: A usable answer: a bare host, URL, or service identifier. Anything with
#: whitespace is a sentence, not a name.
_NAME_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._:/\-]{1,120}$')

#: Answers that are technically well-formed but carry no information; the
#: model reaching for one of these means "I don't know" and is treated as
#: such rather than becoming a node in the graph.
_NON_ANSWERS = {
    "unknown", "none", "null", "n/a", "na", "undefined", "service", "api",
    "host", "hostname", "server", "endpoint", "url", "uri", "external",
    "external-service", "externalservice", "localhost", "example.com",
    "example.org", "api.example.com", "your-service", "servicename",
    "service_name", "third-party", "thirdparty", "backend", "unknown-service",
}

_lock = threading.Lock()
_pipe = None            # (tokenizer, model) once loaded
_load_error = None      # sticky error message if loading ever failed
_result_cache = {}      # prompt hash -> parsed result, per process


# ------------------------------------------------------------- the model ----

def _get_pipe():
    """Lazily load the model once per process. Thread-safe, and memoizes
    failure too so a doomed load isn't retried per edge."""
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

            torch.manual_seed(SEED)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            # Decoder-only batched generation requires left padding: pad on
            # the right and every sequence's last token is a pad token, so
            # the model continues from padding instead of from the prompt.
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            cuda = torch.cuda.is_available()
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                # fp32 on a GPU doubles memory and halves throughput for no
                # accuracy gain on a 1.5B instruct model; fp32 on CPU is
                # correct, since CPU fp16 kernels are slower, not faster.
                torch_dtype=torch.float16 if cuda else torch.float32,
            )
            model.to("cuda" if cuda else "cpu")
            model.eval()
            _pipe = (tokenizer, model)
            log(LOG, "info", "SLM ready", model=MODEL_ID, device=str(model.device),
                dtype=str(model.dtype), load_seconds=round(time.time() - t0, 2))
        except Exception as exc:  # noqa: BLE001 - any load failure degrades to review flags
            _load_error = f"{type(exc).__name__}: {exc}"
            _pipe = None
            log(LOG, "warning", "SLM unavailable; stage will flag for manual review",
                model=MODEL_ID, error=_load_error,
                load_seconds=round(time.time() - t0, 2))
    return _pipe


def _max_input_tokens(tokenizer, model):
    limit = getattr(model.config, "max_position_embeddings", None) or 0
    model_max = getattr(tokenizer, "model_max_length", 0) or 0
    if 0 < model_max < 10 ** 6:
        limit = min(limit, model_max) if limit else model_max
    limit = limit or MAX_INPUT_TOKENS
    # Leave room for the answer; an input sized to the full window makes
    # generation fail or silently truncate the answer.
    return max(256, min(MAX_INPUT_TOKENS, limit - MAX_NEW_TOKENS - 8))


def _generate(pipe, prompts):
    """Greedy, seeded, batched generation. Returns one decoded string per
    prompt, or None for a batch that failed (so one bad batch doesn't lose
    the whole run)."""
    tokenizer, model = pipe
    import torch

    outputs = [None] * len(prompts)
    budget = _max_input_tokens(tokenizer, model)

    for start in range(0, len(prompts), BATCH_SIZE):
        chunk = prompts[start:start + BATCH_SIZE]
        try:
            inputs = tokenizer(chunk, return_tensors="pt", padding=True,
                               truncation=True, max_length=budget).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    stopping_criteria=_brace_stop(tokenizer, inputs["input_ids"].shape[1]),
                )
            prompt_len = inputs["input_ids"].shape[1]
            for i in range(len(chunk)):
                outputs[start + i] = tokenizer.decode(
                    out[i][prompt_len:], skip_special_tokens=True)
        except Exception as exc:  # noqa: BLE001 - keep the remaining batches
            log(LOG, "warning", "generation failed for a batch",
                batch_start=start, size=len(chunk), error=f"{type(exc).__name__}: {exc}")
    return outputs


def _brace_stop(tokenizer, prompt_len):
    """Stop as soon as every sequence has closed its JSON object. Without
    this every call pays for the full token budget generating trailing
    prose after the answer is complete."""
    try:
        from transformers import StoppingCriteria, StoppingCriteriaList

        class _ClosedBrace(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                for row in input_ids:
                    text = tokenizer.decode(row[prompt_len:], skip_special_tokens=True)
                    if "}" not in text:
                        return False
                return True

        return StoppingCriteriaList([_ClosedBrace()])
    except Exception:  # noqa: BLE001 - stopping early is an optimisation, not a requirement
        return None


# ------------------------------------------------------------- the prompt ----

def _build_context(edge, records_by_path):
    rec = records_by_path.get(edge.file)
    if not rec or not rec.source_lines:
        return ""
    start = max(0, edge.line - 1 - CONTEXT_LINES_BEFORE)
    end = min(len(rec.source_lines), edge.line + CONTEXT_LINES_AFTER)
    return "\n".join(f"{start + i + 1}: {line}"
                     for i, line in enumerate(rec.source_lines[start:end]))


def _evidence_lines(edge, records_by_path):
    """What the deterministic stages already established. The old prompt
    threw all of this away and showed the model six lines of source, which
    is both less information than the pipeline had and less than a human
    reviewer would be given."""
    out = []
    rec = records_by_path.get(edge.file)
    if edge.external_pattern:
        out.append(f"Client library detected: {edge.external_pattern}")
    if edge.external_via:
        out.append(f"Detected by: {edge.external_via}")
    if rec and rec.imports:
        raws = [i.raw for i in rec.imports][:12]
        out.append("Imports in this file: " + ", ".join(raws))
    kind = getattr(edge, "joern_source_kind", None)
    if kind:
        out.append(f"Data-flow analysis found a {kind} source but no usable host")
    conditions = getattr(edge, "joern_conditions", None)
    if conditions:
        out.append("Call is control-dependent on: " + "; ".join(conditions[:3]))
    if edge.escalated_to_joern and not edge.joern_resolved:
        out.append("Inter-procedural data-flow analysis did not resolve this call")
    if edge.review_reason:
        out.append(f"Earlier stage note: {edge.review_reason}")
    return out


def _build_candidates(edges, config_records):
    """Destinations that actually exist in this repo: hosts other edges
    resolved by proof, plus hosts and service names declared in config.
    This is what turns the question from 'invent a plausible service' into
    'pick the one this call is most likely talking to'."""
    from .config_check import _build_config_index, _normalize_host

    candidates = {}
    for e in edges:
        if e.host and e.literal_method in ("literal", "const_prop", "joern"):
            host = _normalize_host(e.host) or e.host
            candidates.setdefault(host, "resolved elsewhere in the repo")
    for rec in config_records or []:
        index = _build_config_index(rec)
        for host in index.hosts:
            candidates.setdefault(host, f"declared in {rec.path}")
        for name in index.services:
            candidates.setdefault(name, f"service declared in {rec.path}")
    return dict(list(candidates.items())[:MAX_CANDIDATES])


def _build_user_prompt(edge, records_by_path, candidates):
    snippet = _build_context(edge, records_by_path)
    evidence = _evidence_lines(edge, records_by_path)
    listed = "\n".join(f"- {name} ({why})" for name, why in candidates.items())
    return (
        f"File: {edge.file}\n"
        f"Enclosing function: {edge.caller}\n"
        f"Unresolved call: {edge.callee_expr}({edge.arg_text})\n"
        f"EVIDENCE:\n" + ("\n".join(f"- {e}" for e in evidence) or "- none") + "\n"
        f"CANDIDATES (destinations known to exist in this repo):\n"
        + (listed or "- none found") + "\n"
        f"CODE:\n{snippet or '(no source available)'}\n"
    )


def _render_prompt(pipe, edge, records_by_path, candidates):
    tokenizer, _ = pipe
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(edge, records_by_path, candidates)},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    return text + _PREFILL


# -------------------------------------------------------- parse & validate ----

def _json_objects(text: str):
    """Every balanced `{...}` run in `text`, brace-depth tracked and
    string-aware. The old `\\{.*?\\}` non-greedy regex stopped at the first
    `}`, so any answer containing a nested object — or a `}` inside the
    reasoning string — was truncated into invalid JSON and dropped."""
    out = []
    depth = 0
    start = None
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0 and start is not None:
                    out.append(text[start:i + 1])
    return out


def _extract_json(text: str):
    """Parse the model's answer. The prefill means the reply usually starts
    mid-object, so the prefill is re-attached before parsing."""
    if text is None:
        return None
    candidates = _json_objects(text)
    if not candidates:
        candidates = _json_objects(_PREFILL + text)
    for blob in candidates:
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict) or "service_name" not in data:
            continue
        name = data.get("service_name")
        if isinstance(name, str):
            name = name.strip().strip('"').strip("'").strip()
        else:
            name = None
        data["service_name"] = name or None
        if data.get("confidence") not in ("low", "medium", "high"):
            data["confidence"] = "low"
        data["evidence"] = str(data.get("evidence") or data.get("reasoning") or "").strip()[:300]
        return data
    return None


def _validate(name):
    """Is this a usable destination name at all? Returns (name, reason)."""
    if not name:
        return None, "model_returned_null"
    if len(name) > 200:
        return None, "answer_too_long_to_be_a_name"
    if any(c.isspace() for c in name):
        return None, "answer_was_prose_not_a_name"
    cleaned = name.strip().rstrip("/.,;")
    if cleaned.lower() in _NON_ANSWERS:
        return None, "answer_was_a_placeholder"
    if not _NAME_RE.match(cleaned):
        return None, "answer_is_not_a_valid_host_or_service_token"
    return cleaned, None


def _ground(name, candidates, prompt_text):
    """Can the answer be pointed at? Returns a grounding kind, or None if
    the model produced a name that appears nowhere in what it was shown —
    the definition of a hallucination, and the single most important check
    in this stage."""
    from .config_check import _normalize_host

    lowered = name.lower()
    normalized = _normalize_host(name) or lowered
    for cand in candidates:
        if cand.lower() in (lowered, normalized):
            return "candidate"
    haystack = prompt_text.lower()
    if re.search(r'(?<![\w.\-])' + re.escape(lowered) + r'(?![\w\-])', haystack):
        return "quoted_from_context"
    if normalized != lowered and re.search(
            r'(?<![\w.\-])' + re.escape(normalized) + r'(?![\w\-])', haystack):
        return "quoted_from_context"
    first = normalized.split(".")[0]
    if len(first) > 3 and re.search(r'(?<![\w.\-])' + re.escape(first) + r'(?![\w\-])', haystack):
        return "partial_context_match"
    return None


def _note_review(edge, text):
    """Append rather than overwrite: an edge arriving here often already
    carries a reason from stage 7, and losing it hides *why* this call
    needed guessing in the first place."""
    edge.needs_review = True
    existing = (edge.review_reason or "").strip()
    if existing and text not in existing:
        edge.review_reason = f"{existing} | {text}"
    else:
        edge.review_reason = existing or text


# -------------------------------------------------------------- the stage ----

def check_llm_fallback(edges, records_by_path, ctx=None, config_records=None):
    """Mutates edges in place. Returns the edges this stage handled.

    `config_records` is optional but strongly recommended: it is what the
    candidate list is built from, and a grounded answer is the only kind
    this stage will accept."""
    candidates_edges = [e for e in edges if e.status == "external_candidate" and not e.host]
    if not candidates_edges:
        log(LOG, "info", "nothing reached the SLM fallback; earlier stages resolved everything")
        return []

    log(LOG, "info", "SLM fallback starting",
        unresolved=len(candidates_edges), cap=MAX_EDGES_PER_RUN)

    handled = list(candidates_edges)
    to_infer = candidates_edges[:MAX_EDGES_PER_RUN]
    over_cap = candidates_edges[MAX_EDGES_PER_RUN:]

    for edge in over_cap:
        # Not sent, so not marked as sent. The old version set
        # `sent_to_llm = True` on these, which made the stage's own
        # counters disagree with what it actually did.
        _note_review(edge, f"Not sent to the SLM (per-run cap of {MAX_EDGES_PER_RUN} "
                           "reached); needs manual review.")
    if over_cap and ctx is not None:
        ctx.note("warning", "slm",
                 f"{len(over_cap)} unresolved call(s) exceeded the per-run cap of "
                 f"{MAX_EDGES_PER_RUN} and were flagged without inference. Raise "
                 "PEACH_SLM_MAX_EDGES or fix the upstream stages feeding this many here.")

    pipe = _get_pipe()
    if pipe is None:
        for edge in to_infer:
            _note_review(edge, f"SLM ({MODEL_ID}) unavailable in this environment "
                               f"({_load_error or 'not loaded'}); needs manual review.")
        if ctx is not None:
            ctx.bump("slm.unavailable", len(to_infer))
            ctx.note("warning", "slm",
                     f"{len(to_infer)} unresolved call(s) could not be inferred: the SLM "
                     f"is unavailable ({_load_error or 'not loaded'}). They are flagged "
                     "for manual review, not silently dropped.")
        log(LOG, "warning", "SLM fallback did no inference", unresolved=len(to_infer),
            reason=_load_error)
        return handled

    known = _build_candidates(edges, config_records)
    if ctx is not None:
        ctx.bump("slm.candidate_pool", len(known))
    if not known:
        log(LOG, "warning", "no grounding candidates available",
            note="no config hosts and no proven hosts elsewhere; answers can only be "
                 "grounded in the shown code")

    # Deduplicate: the same client wrapper called from twenty places
    # produces the same prompt twenty times.
    prompts, prompt_by_edge, order = {}, {}, []
    for edge in to_infer:
        text = _render_prompt(pipe, edge, records_by_path, known)
        key = hashlib.sha1(text.encode("utf-8")).hexdigest()
        prompt_by_edge[id(edge)] = (key, text)
        if key not in prompts:
            prompts[key] = text
            if key not in _result_cache:
                order.append(key)

    t0 = time.time()
    raw = _generate(pipe, [prompts[k] for k in order]) if order else []
    infer_seconds = time.time() - t0
    for key, text in zip(order, raw):
        _result_cache[key] = _extract_json(text)

    resolved = 0
    hallucinated = 0
    no_answer = 0
    by_grounding = {}
    by_confidence = {}

    for edge in to_infer:
        edge.sent_to_llm = True
        key, prompt_text = prompt_by_edge[id(edge)]
        result = _result_cache.get(key)

        if not result:
            no_answer += 1
            _note_review(edge, "SLM produced no parseable answer; needs manual review.")
            continue

        name, reject = _validate(result.get("service_name"))
        if not name:
            no_answer += 1
            _note_review(edge, f"SLM did not infer a usable target ({reject}); "
                               "needs manual review.")
            if TRACE_EDGES:
                log(LOG, "debug", "answer rejected", file=edge.file, line=edge.line,
                    reason=reject, raw=result.get("service_name"))
            continue

        grounding = _ground(name, known, prompt_text)
        if grounding is None:
            # The model named something that appears nowhere in the repo or
            # in the context it was shown. Previously this became a node in
            # the dependency graph.
            hallucinated += 1
            _note_review(edge, f"SLM proposed '{name}', which appears nowhere in the "
                               "shown context or in the repo's config; rejected as "
                               "ungrounded. Needs manual review.")
            log(LOG, "warning", "rejected ungrounded SLM answer",
                file=edge.file, line=edge.line, proposed=name,
                evidence=result.get("evidence"))
            continue

        confidence = result.get("confidence", "low")
        edge.host = name
        edge.literal_method = "llm_inferred"
        setattr(edge, "inferred_host", name)
        setattr(edge, "inference_grounding", grounding)
        setattr(edge, "inference_confidence", confidence)
        _note_review(edge, (
            f"SLM-inferred ({confidence} confidence, grounding: {grounding})"
            + (f": {result['evidence']}" if result.get("evidence") else "")
            + " — inferred, not traced; verify before trusting."))
        resolved += 1
        by_grounding[grounding] = by_grounding.get(grounding, 0) + 1
        by_confidence[confidence] = by_confidence.get(confidence, 0) + 1
        log(LOG, "info", "SLM inferred a target", file=edge.file, line=edge.line,
            host=name, confidence=confidence, grounding=grounding)

    if ctx is not None:
        ctx.bump("slm.sent", len(to_infer))
        ctx.bump("slm.resolved", resolved)
        ctx.bump("slm.rejected_ungrounded", hallucinated)
        ctx.bump("slm.no_usable_answer", no_answer)
        ctx.bump("slm.skipped_over_cap", len(over_cap))
        ctx.bump("slm.unique_prompts", len(prompts))
        ctx.bump("slm.inference_ms", int(infer_seconds * 1000))
        for kind, n in by_grounding.items():
            ctx.bump(f"slm.grounding.{kind}", n)
        for conf, n in by_confidence.items():
            ctx.bump(f"slm.confidence.{conf}", n)
        if resolved:
            ctx.note("info", "slm",
                     f"{resolved} external call(s) were resolved only by SLM inference "
                     "(grounded in repo config or the shown code, flagged needs_review). "
                     "These are not proven traces — a high number here means the "
                     "deterministic stages are underperforming.")
        if hallucinated:
            ctx.note("warning", "slm",
                     f"{hallucinated} SLM answer(s) named a service that exists nowhere in "
                     "the repo and were rejected rather than added to the graph.")

    log(LOG, "info", "SLM fallback complete",
        unresolved_in=len(candidates_edges), sent=len(to_infer),
        unique_prompts=len(prompts), resolved=resolved,
        rejected_ungrounded=hallucinated, no_usable_answer=no_answer,
        over_cap=len(over_cap), by_grounding=by_grounding,
        inference_seconds=round(infer_seconds, 2))
    return handled