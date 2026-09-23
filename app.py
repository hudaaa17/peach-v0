import logging
import threading
import uuid

from flask import Flask, request, jsonify, render_template

from analyzer.clone import CloneError
from analyzer.pipeline import analyze_repo
from rag_pipeline.rag_pipeline import RagIndexer
from rag_pipeline.chatbot import QwenChat

# Prototype-only: prints INFO+ logs (including the new rag_pipeline ones)
# to console. Remove/replace if you already configure logging elsewhere
# (e.g. gunicorn's logger).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOG = logging.getLogger("app")

app = Flask(__name__)

# Loads the embedding model once at process start, not on every request.
# persist_dir is where the Chroma collections live on disk, one per run_id.
RAG_INDEXER = RagIndexer(persist_dir="./.peach_rag_db")

# run_id -> {"status": "starting"|"embedding"|"ready"|"error", "chunks": int,
#            "by_kind": dict|None, "elapsed_seconds": float|None, "error": str|None}
# In-memory only — fine for a single-process prototype. A restart loses
# status (though the Chroma collections on disk survive), and this won't be
# correct if you run multiple worker processes (e.g. gunicorn -w N>1),
# since each process would have its own copy of this dict.
RAG_STATUS: dict = {}

# The chat model is loaded lazily on first /api/chat call, not at startup —
# it's a second model on top of the embedder, and no point paying for it
# before anyone's actually chatted.
_CHAT_MODEL = None
_CHAT_MODEL_LOCK = threading.Lock()


def get_chat_model() -> QwenChat:
    global _CHAT_MODEL
    if _CHAT_MODEL is None:
        with _CHAT_MODEL_LOCK:
            if _CHAT_MODEL is None:  # re-check inside the lock
                _CHAT_MODEL = QwenChat()
    return _CHAT_MODEL


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    url = (data.get("repo_url") or "").strip()
    if not url:
        return jsonify({"error": "Please enter a GitHub repo URL."}), 400
    run_id = uuid.uuid4().hex
    RAG_STATUS[run_id] = {"status": "starting", "chunks": 0, "by_kind": None,
                           "elapsed_seconds": None, "error": None}
    try:
        graph = analyze_repo(
            url, run_id=run_id,
            rag_hook=RAG_INDEXER.as_hook(run_id, status_store=RAG_STATUS))
    except CloneError as e:
        RAG_STATUS.pop(run_id, None)
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # noqa: BLE001 - surface a friendly message either way
        RAG_STATUS.pop(run_id, None)
        return jsonify({"error": f"Analysis failed: {e}"}), 500
    graph["meta"]["run_id"] = run_id  # frontend polls /api/rag-status/<run_id> with this
    return jsonify(graph)


@app.route("/api/rag-status/<run_id>")
def rag_status(run_id):
    """Polled by the frontend after the graph renders, to know when to
    reveal the chat. `status` is one of: starting, embedding, ready, error."""
    return jsonify(RAG_STATUS.get(run_id, {"status": "unknown"}))


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    run_id = (data.get("run_id") or "").strip()
    message = (data.get("message") or "").strip()
    if not run_id or not message:
        return jsonify({"error": "run_id and message are required."}), 400

    status = RAG_STATUS.get(run_id)
    if not status or status.get("status") != "ready":
        return jsonify({
            "error": "The knowledge base for this repo isn't ready yet.",
            "status": (status or {}).get("status", "unknown"),
        }), 409

    try:
        chunks = RAG_INDEXER.retrieve(run_id, message, k=8)
        model = get_chat_model()
        answer = model.answer(message, chunks)
    except Exception as e:  # noqa: BLE001 - surface a friendly message either way
        LOG.exception("chat failed for run_id=%s", run_id)
        return jsonify({"error": f"Chat failed: {e}"}), 500

    return jsonify({
        "answer": answer,
        "sources": [
            {
                "path": c["metadata"].get("repo_path"),
                "name": c["metadata"].get("qualified_name"),
                "start_line": c["metadata"].get("start_line"),
                "end_line": c["metadata"].get("end_line"),
            }
            for c in chunks
        ],
    })


if __name__ == "__main__":
    # threaded=True matters here: without it, Flask's dev server handles one
    # request at a time, so the frontend's /api/rag-status polling would
    # queue up behind whatever request is currently in flight instead of
    # actually getting an answer while indexing runs in the background.
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)