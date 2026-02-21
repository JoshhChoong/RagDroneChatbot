"""
Flask app for the Drone Safety RAG frontend.

Run locally:
  python app.py
  # or: FLASK_DEBUG=1 python app.py

Deploy (production):
  gunicorn -w 4 -b 0.0.0.0:5000 app:app
  # or set PORT: gunicorn -w 4 -b 0.0.0.0:$PORT app:app

Set OPENROUTER_API_KEY (recommended) or GEMINI_API_KEY in the environment. Optionally CHROMA_PERSIST_DIR.
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
        openrouter_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
        gemini_key = (os.getenv("GEMINI_API_KEY") or "").strip()
        if not openrouter_key and not gemini_key:
            raise RuntimeError(
                "Set OPENROUTER_API_KEY (recommended) or GEMINI_API_KEY in .env. "
                "Get an OpenRouter key at https://openrouter.ai/keys"
            )
        _rag_system = RAGSystem(openrouter_api_key=openrouter_key or None, gemini_api_key=gemini_key or None)
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
