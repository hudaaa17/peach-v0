from flask import Flask, request, jsonify, render_template

from analyzer.clone import CloneError
from analyzer.pipeline import analyze_repo

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    url = (data.get("repo_url") or "").strip()
    if not url:
        return jsonify({"error": "Please enter a GitHub repo URL."}), 400
    try:
        graph = analyze_repo(url)
    except CloneError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # noqa: BLE001 - surface a friendly message either way
        return jsonify({"error": f"Analysis failed: {e}"}), 500
    return jsonify(graph)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
