"""
Flask app for the Drone Safety RAG frontend.

Run locally:
  python app.py
  # or: FLASK_DEBUG=1 python app.py

Deploy (production):
  gunicorn -w 4 -b 0.0.0.0:5000 app:app
  # or set PORT: gunicorn -w 4 -b 0.0.0.0:$PORT app:app

Set GEMINI_API_KEY (and optionally CHROMA_PERSIST_DIR) in the environment.
"""
import os
from pathlib import Path

from flask import Flask, render_template, request, jsonify

# Project root and frontend paths
ROOT = Path(__file__).resolve().parent
TEMPLATES = ROOT / "FrontEnd" / "Templates"
STATIC = ROOT / "FrontEnd" / "static"

app = Flask(
    __name__,
    template_folder=str(TEMPLATES),
    static_folder=str(STATIC) if STATIC.is_dir() else None,
)

# Lazy-load RAG to avoid slow startup until first request (optional for deployment)
_rag_system = None


def get_rag():
    global _rag_system
    if _rag_system is None:
        from RagPipeline.generation import RAGSystem
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        _rag_system = RAGSystem(key)
    return _rag_system


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    """For deployment health checks (e.g. load balancer)."""
    return jsonify({"status": "ok"})


@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json() or {}
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        rag = get_rag()
        result = rag.generate_answer(query)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    app.run(host="0.0.0.0", port=port, debug=debug)
