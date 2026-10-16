"""
Observability core for the peach pipeline.

Everything the stages need to be monitorable lives here so the stage
modules stay readable: a process-wide logging setup, a per-analysis run
context (run id + stage timings + counters), and a `stage()` context
manager that every pipeline step wraps itself in.

Design notes
------------
* Logging is *structured*: every record carries a `run` id and, inside a
  stage, a `stage` name, so concurrent analyses interleaved in one log
  stream can still be pulled apart. Set `PEACH_LOG_FORMAT=json` to emit
  one JSON object per line for ingestion into a log store; the default
  `text` format stays human-readable in a dev terminal.
* Counters are cheap and additive. A stage bumps a counter instead of
  logging a line per item, so a 400-file repo doesn't produce 400 log
  lines at INFO — the per-item detail lives at DEBUG.
* "Bail reasons" are the important idea here. When a stage declines to
  resolve something, it records *why* as a counter key. The histogram of
  bail reasons is what turns "Joern resolved 0" from a mystery into a
  one-line answer.

Environment variables
---------------------
PEACH_LOG_LEVEL   DEBUG | INFO | WARNING | ERROR      (default INFO)
PEACH_LOG_FORMAT  text | json                          (default text)
PEACH_LOG_FILE    path to also write logs to           (default none)
PEACH_TRACE_EDGES 1 to log every per-edge decision     (default off)
"""
import contextvars
import json
import logging
import os
import sys
import time
import uuid
from collections import Counter, OrderedDict
from contextlib import contextmanager

_run_id_var = contextvars.ContextVar("peach_run_id", default="-")
_stage_var = contextvars.ContextVar("peach_stage", default="-")

_configured = False

#: When on, stages emit a DEBUG line for *every* edge decision rather than
#: only aggregate counters. Expensive on big repos, invaluable on a repo
#: where three specific edges are behaving oddly.
TRACE_EDGES = os.environ.get("PEACH_TRACE_EDGES", "").lower() in ("1", "true", "yes")


class _ContextFilter(logging.Filter):
    """Attach run/stage ids to every record so formatters can use them."""

    def filter(self, record):
        record.run = _run_id_var.get()
        record.stage = _stage_var.get()
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = OrderedDict(
            ts=self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            level=record.levelname,
            run=getattr(record, "run", "-"),
            stage=getattr(record, "stage", "-"),
            logger=record.name,
            msg=record.getMessage(),
        )
        extra = getattr(record, "fields", None)
        if extra:
            payload["fields"] = extra
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


#: Fields whose whole point is the detail they carry. Truncating these to
#: _INLINE_LIMIT chars decapitated exactly the part worth reading — a
#: four-element `traces` list would render as `[{'trace': 'env_var', ...}, {'trace': '…`,
#: showing the first entry's reason and hiding the other three. They get
#: one line each instead.
#:
#: This list is per-field-name, not per-stage, so it has to be kept in
#: sync by hand whenever a stage starts emitting a new multi-element
#: field — nothing enforces that automatically, and a field left off
#: this list degrades silently back to inline truncation rather than
#: raising anywhere. `conditions` (stage 7: the branch conditions
#: guarding a resolved call) and `by_kind`/`by_grounding`/`by_source`
#: (stages 6, 7, 8: per-kind match/answer breakdowns) were added for
#: exactly that reason — they carry real source text and were
#: truncating mid-list before this fix, which is the specific failure
#: this mechanism exists to prevent.
_INLINE_LIMIT = 110

_BLOCK_FIELDS = ("traces", "summary", "message", "bail_reasons", "arg_shapes", "traces_ran",
                 "traces_resolved", "by_pattern", "examples",
                 "top_unresolved_roots", "identifiers",
                 "conditions", "by_kind", "by_grounding", "by_source", "outcome",
                 "flows", "steps")


class _TextFormatter(logging.Formatter):
    def format(self, record):
        base = (
            f"{self.formatTime(record, '%H:%M:%S')} "
            f"{record.levelname:<7} "
            f"[{getattr(record, 'run', '-')}] "
            f"[{getattr(record, 'stage', '-'):<12}] "
            f"{record.getMessage()}"
        )
        extra = getattr(record, "fields", None)
        if extra:
            inline, block = [], []
            for k, v in extra.items():
                if not v and v != 0:
                    continue  # an empty list/dict says nothing; don't print a bare header
                rendered = str(v)
                blockable = (k in _BLOCK_FIELDS and isinstance(v, (list, tuple, dict))) \
                    or (isinstance(v, str) and len(rendered) > _INLINE_LIMIT)
                # Short values stay inline even if blockable — a one-element
                # list doesn't need three lines. Break only when inlining
                # would actually lose information to truncation.
                if blockable and len(rendered) > _INLINE_LIMIT:
                    block.append((k, v))
                else:
                    inline.append(f"{k}={_short(v)}")
            if inline:
                base = f"{base} | {' '.join(inline)}"
            for key, value in block:
                base += f"\n    {key}:"
                for line in _render_block(value):
                    base += f"\n      {line}"
        if record.exc_info:
            base = f"{base}\n{self.formatException(record.exc_info)}"
        return base


def _render_block(value):
    """One readable line per element, untruncated."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [f"{k}: {v}" for k, v in value.items()]
    lines = []
    for item in value:
        if isinstance(item, dict) and "trace" in item:
            # Legacy per-trace result shape (name/ran/resolved/reason/detail).
            # No current stage emits this — joern_check's stage 7 rewrite
            # replaced the regex "traces" it described with real CPG flow
            # results — but a still-installed integration or an older
            # cached log record could still carry it, so the readable
            # rendering stays rather than silently reverting those to
            # `str(item)`.
            verdict = ("RESOLVED" if item.get("resolved")
                       else "ran, no match" if item.get("ran")
                       else "skipped")
            line = f"{item['trace']:<22} {verdict}"
            note = item.get("detail") or item.get("reason")
            if note:
                line += f"  ({note})"
            lines.append(line)
        else:
            lines.append(str(item))
    return lines


def _short(value, limit=120):
    text = str(value)
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def configure_logging(force=False):
    """Idempotent logging setup for the whole process."""
    global _configured
    if _configured and not force:
        return
    level = os.environ.get("PEACH_LOG_LEVEL", "INFO").upper()
    fmt = os.environ.get("PEACH_LOG_FORMAT", "text").lower()
    formatter = _JsonFormatter() if fmt == "json" else _TextFormatter()

    root = logging.getLogger("peach")
    root.setLevel(getattr(logging, level, logging.INFO))
    root.handlers.clear()
    root.propagate = False

    handlers = [logging.StreamHandler(sys.stderr)]
    log_file = os.environ.get("PEACH_LOG_FILE")
    if log_file:
        try:
            handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
        except OSError as exc:
            print(f"peach: could not open PEACH_LOG_FILE={log_file}: {exc}", file=sys.stderr)

    ctx_filter = _ContextFilter()
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.addFilter(ctx_filter)
        root.addHandler(handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(f"peach.{name}")


def log(logger, level, msg, **fields):
    """Log with structured fields: log(LOG, 'info', 'parsed', files=18)."""
    getattr(logger, level)(msg, extra={"fields": fields} if fields else None)


# --------------------------------------------------------------- run ctx ----

class RunContext:
    """Per-analysis state: stage timings, counters, and bail-reason
    histograms. One instance is threaded through the whole pipeline and
    its `summary()` is attached to the graph meta, so whatever shows up in
    the logs is also visible to the caller/UI without re-reading logs."""

    def __init__(self, repo_url: str, run_id: str = None):
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self.repo_url = repo_url
        self.started = time.time()
        self.stages = OrderedDict()   # name -> {seconds, status, fields}
        self.counters = Counter()     # "stage.key" -> int
        self.notes = []               # surfaced diagnostics, see note()

    # -- counters ------------------------------------------------------
    def bump(self, key: str, n: int = 1):
        self.counters[key] += n

    def count(self, key: str) -> int:
        return self.counters.get(key, 0)

    def counters_with_prefix(self, prefix: str) -> dict:
        cut = len(prefix)
        return {
            k[cut:]: v for k, v in sorted(self.counters.items())
            if k.startswith(prefix)
        }

    # -- diagnostics ---------------------------------------------------
    def note(self, level: str, stage: str, message: str, **fields):
        """Record a human-readable diagnostic that should survive past the
        log stream — e.g. 'no config files found, so the config stage was
        a no-op'. These end up in meta.diagnostics."""
        self.notes.append({
            "level": level, "stage": stage, "message": message, "fields": fields,
        })

    # -- summary -------------------------------------------------------
    def summary(self) -> dict:
        return {
            "run_id": self.run_id,
            "total_seconds": round(time.time() - self.started, 2),
            "stages": [
                {"name": name, **data} for name, data in self.stages.items()
            ],
            "counters": dict(sorted(self.counters.items())),
            "diagnostics": self.notes,
        }


@contextmanager
def stage(ctx: RunContext, name: str, logger: logging.Logger, **start_fields):
    """Wrap a pipeline stage: logs entry/exit, records duration, and makes
    sure a raised exception is logged *with its stage* before it unwinds."""
    token_run = _run_id_var.set(ctx.run_id)
    token_stage = _stage_var.set(name)
    t0 = time.time()
    log(logger, "info", "stage start", **start_fields)
    record = {"seconds": 0.0, "status": "ok"}
    ctx.stages[name] = record
    try:
        yield record
    except Exception as exc:
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["seconds"] = round(time.time() - t0, 3)
        logger.exception("stage failed")
        ctx.note("error", name, f"stage raised {type(exc).__name__}: {exc}")
        raise
    else:
        record["seconds"] = round(time.time() - t0, 3)
        finish = {k: v for k, v in record.items() if k not in ("status",)}
        log(logger, "info", "stage done", **finish)
    finally:
        _stage_var.reset(token_stage)
        _run_id_var.reset(token_run)


@contextmanager
def run_context(ctx: RunContext):
    """Bind a run id outside of any particular stage."""
    token = _run_id_var.set(ctx.run_id)
    try:
        yield ctx
    finally:
        _run_id_var.reset(token)