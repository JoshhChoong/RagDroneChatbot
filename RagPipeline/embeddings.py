import os
import sys
from typing import List

# Cached local model so we don't reload on every get_embedding() call
_sentence_transformers_model = None


def _get_local_model():
    """Lazy-load and cache the sentence-transformers model (cheapest: small, local, free)."""
    global _sentence_transformers_model
    if _sentence_transformers_model is None:
        from sentence_transformers import SentenceTransformer
        # Suppress "LOAD REPORT" / UNEXPECTED key / progress bar (they use print, not logging)
        with open(os.devnull, "w") as devnull:
            _old_stdout, _old_stderr = sys.stdout, sys.stderr
            try:
                sys.stdout = sys.stderr = devnull
                # paraphrase-MiniLM-L3-v2 (~61MB), 384 dims
                _sentence_transformers_model = SentenceTransformer(
                    "sentence-transformers/paraphrase-MiniLM-L3-v2"
                )
            finally:
                sys.stdout, sys.stderr = _old_stdout, _old_stderr
    return _sentence_transformers_model


def get_embedding(text: str) -> List[float]:
    """Return embedding for `text`.

    Preference order (cheapest first):
    - Use local sentence-transformers (paraphrase-MiniLM-L3-v2) — free, no API.
    - If USE_GEMINI_EMBEDDINGS=1 and API key is set, try Google's embedding API.
    """
    # Prefer local (free) model first
    use_gemini = os.getenv("USE_GEMINI_EMBEDDINGS", "").strip().lower() in ("1", "true", "yes")
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if use_gemini and key:
        import requests
        from requests.exceptions import HTTPError

        def _parse_embedding_response(data):
            if not isinstance(data, dict):
                return None
            if "embedding" in data and "values" in data["embedding"]:
                return data["embedding"]["values"]
            if "embedding" in data and isinstance(data["embedding"], list):
                return data["embedding"]
            if "embeddings" in data and isinstance(data["embeddings"], list):
                emb = data["embeddings"][0]
                return emb.get("values", emb) if isinstance(emb, dict) else emb
            return None

        # Google AI embedding models (try in order)
        # Doc: https://ai.google.dev/gemini-api/docs/embeddings
        endpoints_to_try = [
            ("gemini-embedding-001", "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent", {"content": {"parts": [{"text": text}]}}),
            ("text-embedding-004", "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent", {"content": {"parts": [{"text": text}]}}),
            ("textembedding-gecko-001 (legacy)", "https://generativelanguage.googleapis.com/v1beta2/models/textembedding-gecko-001:embedText", {"text": text}),
        ]
        last_error = None
        for name, url_base, body in endpoints_to_try:
            url = f"{url_base}?key={key}"
            try:
                resp = requests.post(url, json=body, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                parsed = _parse_embedding_response(data)
                if parsed is not None:
                    return parsed
            except HTTPError as e:
                last_error = f"{name}: HTTP {resp.status_code} - {resp.text[:200] if resp.text else str(e)}"
                continue
            except Exception as e:
                last_error = f"{name}: {e}"
                continue
        print("Gemini embedding failed (tried available endpoints). Falling back to local sentence-transformers.")
        if last_error:
            print("  Last error:", last_error)
            if "403" in str(last_error) or "401" in str(last_error):
                print("  Tip: Get a key at https://aistudio.google.com/apikey and set GEMINI_API_KEY in .env")

    # Local fallback: sentence-transformers for offline embeddings (384 dims)
    try:
        model = _get_local_model()
        return model.encode(text).tolist()
    except Exception as e:
        raise RuntimeError(
            "No GEMINI_API_KEY set and local sentence-transformers unavailable. "
            "Set GEMINI_API_KEY in .env or install sentence-transformers."
        ) from e
