# peach — repo system mapper (graph extraction slice)

A small Flask app that connects a public GitHub repo and builds the full
**graph data extraction** half of the PEACH pipeline end to end — every
stage in the architecture diagram, from tree-sitter parsing through the
Joern escalation and LLM fallback. No vector pipeline, no chat assistant —
just clone → parse → resolve → map.

## What it does

1. **Connect repo** — shallow `git clone --depth 1` into a scratch temp dir.
2. **Tree-sitter parse** — a real structural pass over every source file.
   Python is parsed with the `ast` module (a true CST); other common
   languages (JS/TS, Java, Go, Ruby, C#, Rust) get a lightweight
   language-agnostic regex structural pass in the same spirit — fast,
   no compiling or executing anything.
3. **SCIP symbol resolution** — every call is checked against a repo-wide
   symbol table; calls that resolve to a function/method defined in the
   repo become "internal" edges.
4. **ast-grep pattern match** — calls that *didn't* resolve internally are
   checked against known external-client call shapes (`requests.get`,
   `fetch(...)`, `axios.post`, `http.Get`, `RestTemplate`, etc.).
5. **Literal check** — if the flagged call's argument is a string literal,
   it's read directly (and parsed as a URL when possible).
6. **Constant-prop check** — if the argument is a bare variable, we trace
   backward through the file for the nearest `var = "literal"` assignment
   above the call site.
7. **Config check** — the resolved host (or its first label, as a stand-in
   service name) is cross-referenced against yaml/json/.env/Dockerfiles and
   other manifest-looking files in the repo, and linked if found.
8. **Escalate to Joern** — for external-call candidates that steps 5–7
   still couldn't resolve, a scoped, cross-function/cross-file static trace
   widens the search past the boundary of one file or function: backward
   through `self.attr` instance attributes set in a different method,
   through imported module-level constants defined in a different file,
   through environment-variable lookups (resolved to the key name and
   handed back to the config check above), and — as the most approximate,
   most heuristically-gated case — through the argument another function
   passes in when it calls this one elsewhere in the repo. We don't ship
   the real Joern binary (it's a heavy JVM code-property-graph engine);
   this stage plays the same *role* in the pipeline, at the same position,
   with the same "only for the hard cases" cost profile.
9. **LLM fallback** — anything still unresolved after Joern is shown to a
   small local instruction-tuned code model, **Qwen2.5-Coder-1.5B-Instruct**
   (loaded once per process via `transformers`), along with the call site
   and a few lines of surrounding context, and asked to infer the likely
   service name from naming/semantics alone. Every edge this stage touches
   — resolved or not — is flagged `needs_review` with a short reason,
   since an LLM guess is exactly that: a guess, not a proven trace. If the
   model/weights aren't available in a given environment (no `transformers`/
   `torch`, no cached weights, no network), the stage degrades to flagging
   the edge for manual review instead of crashing the analysis.

The result is rendered as an interactive node/edge graph (vis-network) with
files, functions/methods/classes, external services, and matched config
files as node types, and `contains` / `imports` / `calls` /
`calls_external` / `configured_in` as edge kinds. `calls_external` edges
carry a `method` (`literal` / `const_prop` / `joern` / `llm_inferred`) and,
once a call has been through Joern or the LLM fallback, a `needs_review`
flag with a human-readable reason. Click any node to see its file, line,
and connections in the side panel.

Everything is best-effort and heuristic — static analysis (even Joern-style
inter-procedural tracing) can only *approximate* values truly computed at
runtime, and the LLM fallback is explicitly a flagged guess rather than a
proven trace. That matches the limitations called out in the project
write-up (manifest drift, heuristic escalation, inferred edges needing
active review); this slice surfaces those caveats in the graph itself
rather than hiding them.

## Run it

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:5000`, paste a public repo (either a full URL
or `owner/repo` shorthand), and click **Map it**.

The first analysis that actually reaches the LLM fallback stage will
download the Qwen2.5-Coder-1.5B-Instruct weights from Hugging Face
(~3GB) and cache them locally; every run after that loads from cache.
If you'd rather skip that entirely (e.g. no network, no disk to spare, or
you just want the deterministic stages), don't install `transformers`/
`torch` — the app still runs fine and simply flags Joern-unresolved edges
as "SLM unavailable — needs manual review" instead of guessing.

Two environment variables tune stage 9 if you need to:

- `PEACH_SLM_MODEL_ID` — override the model id (defaults to
  `Qwen/Qwen2.5-Coder-1.5B-Instruct`).
- `PEACH_SLM_MAX_EDGES` — cap how many unresolved calls get sent to the
  model in one analysis run (defaults to `25`), so a repo with a lot of
  dynamic/indirected calls doesn't turn one "Map it" click into a very
  long CPU-bound generation loop.

## Logging & monitoring

Every stage is instrumented. Each one logs entry/exit with a wall-clock
duration, emits aggregate counters, and — importantly — records a *reason*
whenever it declines to resolve something. A stage reporting "0 resolved"
is otherwise ambiguous between "tried and missed" and "couldn't even try",
and those two need very different fixes.

```bash
PEACH_LOG_LEVEL=DEBUG PEACH_TRACE_EDGES=1 python app.py
```

| Variable | Default | What it does |
|---|---|---|
| `PEACH_LOG_LEVEL` | `INFO` | Standard levels. `DEBUG` adds per-file detail. |
| `PEACH_LOG_FORMAT` | `text` | `json` emits one JSON object per line for log ingestion. |
| `PEACH_LOG_FILE` | — | Also write logs to this path. |
| `PEACH_TRACE_EDGES` | off | Log every per-edge decision, not just aggregates. Noisy but decisive when a handful of specific calls are misbehaving. |

Every log record carries a short `run` id, so concurrent analyses can be
pulled apart from one stream. The same data is attached to the API
response under `meta.run` — `stages` (name, seconds, status), `counters`,
and `diagnostics` (human-readable notes) — so a surprising run can be
explained from the response alone, without server log access.

The counters worth watching:

- `parse.arg_repr.placeholder` — call arguments that couldn't be recovered
  as source text. **These are unresolvable by every downstream stage.**
  Non-zero here caps how well literal/const-prop/Joern can possibly do.
- `joern.trace_ran.*` vs `joern.bail.*` — which of the four traces
  actually executed, and why the others declined.
- `joern.arg_shape.*` — the shape of the arguments arriving at escalation.
- `external.uncovered_client_calls` — calls that look like network clients
  but match no entry in `EXTERNAL_PATTERNS`, i.e. catalogue gaps.
- `slm.resolved` — a high number means the deterministic stages are
  underperforming, since the SLM is meant to be the last resort.

## Tests

```bash
python -m unittest discover -s tests -t .
```

## Notes / limits

- Public repos only (no auth), shallow clone, capped at ~400 files, files
  over ~400KB are skipped — keeps things responsive for a demo.
- The repo is deleted from disk right after analysis; nothing is persisted
  server-side.
- The "constant-prop" backward trace only follows simple
  `name = "literal"`-style assignments in the same file — it won't follow
  through function calls, string concatenation, or f-strings. That's a
  deliberate scope cut, not a bug; the Joern escalation stage is what picks
  up several of those cross-function/cross-file cases instead.
- The call-site argument-propagation trace in the Joern stage only checks
  *one* call site and doesn't verify parameter position — it's gated
  behind a naming heuristic (`url`/`host`/`endpoint`/... in the parameter
  name) specifically because it's the least reliable of the four traces.
- There's no persistent human-review queue in this slice (the full project
  design keeps one in Postgres alongside the vector store, which is out of
  scope here) — `needs_review`/`review_reason` are surfaced per-edge in the
  graph JSON and the side panel instead, so nothing is silently dropped.
