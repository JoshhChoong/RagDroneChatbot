"""
Heavily inspired from https://dev.to/paul_robertson_e844997d2b/build-a-rag-system-from-scratch-create-an-ai-that-answers-questions-about-your-codebase-pnb
"""

import os
import re
import time
import json
import requests
from pathlib import Path
from RagPipeline.retreival import CodebaseRetriever
from typing import Dict, List, Any
from dotenv import load_dotenv

# For fallback when metadata has no source_url (e.g. pre-existing DB)
_PROCESSED_DATA_DIR = Path(__file__).resolve().parent.parent / "Data" / "files" / "processed"

load_dotenv()

# Default: prefer faster models first (2.5-flash often has more quota than 2.0-flash)
# Set GEMINI_MODEL to override; GEMINI_FAST_MODE=1 = fewer retries, shorter waits
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-preview")
GEMINI_MODEL_FALLBACKS = os.getenv("GEMINI_MODEL_FALLBACKS", "gemini-2.0-flash,gemini-1.5-flash,gemini-1.5-pro").split(",")
GEMINI_FAST_MODE = os.getenv("GEMINI_FAST_MODE", "1").strip().lower() in ("1", "true", "yes")

# Official REST endpoint: https://ai.google.dev/gemini-api/docs/rest
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_GENERATE_PATH = f"models/{{model}}:generateContent"
GEMINI_LIST_MODELS_URL = f"{GEMINI_BASE}/models"

# Preferred order: try faster / higher-quota models first to avoid long 429 waits
PREFERRED_MODEL_ORDER = (
    "gemini-2.5-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro",
    "gemini-1.5-pro-latest",
    "gemini-1.0-pro",
)

_cached_available_models: list = []


def _get_available_gemini_models(api_key: str) -> list:
    """Return list of model names that support generateContent, in preferred order. Cached."""
    global _cached_available_models
    if _cached_available_models:
        return _cached_available_models
    try:
        r = requests.get(
            f"{GEMINI_LIST_MODELS_URL}?key={api_key}",
            timeout=15,
        )
        if not r.ok:
            return []
        data = r.json()
        models = data.get("models") or []
        supported = []
        for m in models:
            name = m.get("name", "")
            if not name.startswith("models/"):
                continue
            model_id = name.replace("models/", "", 1)
            if "tts" in model_id.lower():
                continue  # TTS models return AUDIO only, not TEXT
            methods = m.get("supportedGenerationMethods") or []
            if "generateContent" in methods:
                supported.append(model_id)
        # Sort by preferred order, then append any others
        ordered = []
        for preferred in PREFERRED_MODEL_ORDER:
            if preferred in supported:
                ordered.append(preferred)
        for s in supported:
            if s not in ordered:
                ordered.append(s)
        _cached_available_models = ordered
        return ordered
    except Exception:
        return []


def _parse_retry_seconds(err_detail) -> int:
    """Parse retry delay from Gemini 429. In fast mode skip wait and try next model immediately."""
    if GEMINI_FAST_MODE:
        return 0
    try:
        if isinstance(err_detail, dict):
            for d in err_detail.get("error", {}).get("details", []) or []:
                if isinstance(d, dict) and d.get("@type") == "type.googleapis.com/google.rpc.RetryInfo":
                    delay = d.get("retryDelay", "")
                    if isinstance(delay, str):
                        m = re.match(r"^(\d+(?:\.\d+)?)\s*s", delay.strip())
                        if m:
                            return min(120, max(10, int(float(m.group(1)))))
    except Exception:
        pass
    return 60


def _gemini_generate_content(api_key: str, prompt: str, temperature: float = 0.1, max_tokens: int = 500):
    """Call Gemini generateContent REST API and return the generated text.
    On 429 (quota), retries after waiting; tries fallback models. Returns None if all fail."""
    discovered = _get_available_gemini_models(api_key)
    if discovered:
        models_to_try = discovered
    else:
        models_to_try = [GEMINI_MODEL] + [m.strip() for m in GEMINI_MODEL_FALLBACKS if m.strip()]
    last_error = None

    for model in models_to_try:
        url = f"{GEMINI_BASE}/{GEMINI_GENERATE_PATH.format(model=model)}"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        max_attempts = 1 if GEMINI_FAST_MODE else 3
        for attempt in range(max_attempts):
            resp = requests.post(url, json=payload, headers=headers, timeout=90)
            if resp.ok:
                data = resp.json()
                candidates = data.get("candidates") or []
                if not candidates:
                    raise RuntimeError("Gemini returned no candidates")
                parts = (candidates[0].get("content") or {}).get("parts") or []
                if not parts:
                    return ""
                return (parts[0].get("text") or "").strip()

            try:
                err_detail = resp.json()
            except Exception:
                err_detail = resp.text

            if resp.status_code == 404:
                last_error = f"Gemini API 404 for {url}: {err_detail}"
                if model != models_to_try[-1]:
                    print(f"Model {model} not found (404). Trying next model...")
                break
            if resp.status_code == 400:
                err_msg = str(err_detail)
                if "modalities" in err_msg or "not supported" in err_msg.lower():
                    last_error = f"Gemini API 400 for {url}: model does not support text output"
                    if model != models_to_try[-1]:
                        print(f"Model {model} does not support text (400). Trying next model...")
                    break
            if resp.status_code not in (429, 404, 400):
                raise RuntimeError(
                    f"Gemini API {resp.status_code} for {url}: {err_detail}"
                ) from None
            if resp.status_code != 429:
                continue

            # 429: in fast mode skip long waits and try next model immediately after one short wait
            wait_sec = _parse_retry_seconds(err_detail)
            if attempt < max_attempts - 1:
                if not GEMINI_FAST_MODE:
                    print(f"Gemini quota exceeded (model={model}). Waiting {wait_sec}s before retry {attempt + 2}/{max_attempts}...")
                time.sleep(wait_sec)
            else:
                last_error = f"Gemini API 429 for {url}: {err_detail}"
                break

        if last_error and model == models_to_try[-1]:
            print("All Gemini models failed (quota or not found). Use local fallback or set GEMINI_MODEL_FALLBACKS.")
            return None
        if last_error:
            idx = models_to_try.index(model)
            next_model = models_to_try[idx + 1] if idx + 1 < len(models_to_try) else "?"
            print(f"Model {model} quota exceeded. Trying next model: {next_model}...")
            last_error = None
    return None


def _get_source_url_and_title(meta: dict) -> tuple:
    """Return (url, title) from metadata or by loading JSON from processed data dir."""
    url = (meta or {}).get("source_url")
    title = (meta or {}).get("source_title")
    if url:
        return (url, title or "")
    # Fallback: load JSON by file_name to get meta.url / meta.title
    fname = (meta or {}).get("file_name")
    if fname and _PROCESSED_DATA_DIR.is_dir():
        path = _PROCESSED_DATA_DIR / fname
        try:
            if path.is_file():
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                m = data.get("meta") or {}
                return (m.get("url") or "", m.get("title") or "")
        except Exception:
            pass
    return ("", title or "")


def _first_three_source_links(used_metadatas: List[dict]) -> List[Dict[str, str]]:
    """Deduplicate by URL, take first 3, return list of {url, title} for display."""
    seen = set()
    out = []
    for meta in used_metadatas:
        url, title = _get_source_url_and_title(meta)
        if url and url not in seen:
            seen.add(url)
            out.append({"url": url, "title": title or url})
        if len(out) >= 3:
            break
    return out


class RAGSystem:
    def __init__(self, gemini_api_key: str):
        self.gemini_api_key = gemini_api_key
        self.retriever = CodebaseRetriever(gemini_api_key)

    def generate_answer(self, query: str, max_context_length: int = 4500) -> Dict:
        """Generate answer using retrieved context and Gemini."""
        retrieved = self.retriever.retrieve_relevant_chunks(query, n_results=15)

        context_parts = []
        used_metadatas = []
        total_length = 0
        per_doc_limit = 1000

        for doc, metadata in zip(retrieved["documents"], retrieved["metadatas"]):
            if total_length >= max_context_length:
                break

            text = metadata.get("summary") if isinstance(metadata, dict) else None
            if not text:
                text = doc or ""

            if len(text) > per_doc_limit:
                text = text[:per_doc_limit] + "\n...[truncated]"

            if total_length + len(text) > max_context_length:
                remaining = max_context_length - total_length
                if remaining <= 0:
                    break
                text = text[:remaining]

            context_parts.append(f"File: {metadata.get('file_name', 'unknown')}\n{text}\n---")
            used_metadatas.append(metadata)
            total_length += len(text)

        context = "\n".join(context_parts)

        prompt = f"""You are a helpful assistant that answers questions about drone safety and regulations in Canada.
Use only the Context below to answer. Cite specific numbers and limits when they appear.

Answer in plain language: write full sentences that explain the actual rules or steps. Do NOT just list section titles, menu labels, or link names from the context (e.g. do not answer "where can I fly" with only "Search the interactive map" or "Prohibited areas"—instead explain where you can fly, e.g. in uncontrolled airspace, away from airports, and that you should check the map or CARs for specifics).

If the Context mentions "122 metres" or "400 feet" or "stay below" for where you can fly or BVLOS, that is the general maximum height—cite it for "how high" / "maximum height" questions. Do not say the context lacks it if that passage is present.

Give a direct answer first and always finish your answer. Only if the Context truly has no relevant information, say so in one sentence and suggest the source links.

Context:
{context}

Question: {query}

Answer:"""

        answer = _gemini_generate_content(
            self.gemini_api_key, prompt, temperature=0.1, max_tokens=2048
        )

        if answer is None:
            # Local fallback when all Gemini models fail (quota/404)
            summary_len = 800
            summary = (context.strip()[:summary_len] + "...") if len(context.strip()) > summary_len else context.strip()
            answer = (
                "I couldn't reach the AI (quota or model unavailable). "
                "Here are relevant excerpts from the documentation:\n\n" + summary
            )

        return {
            "answer": answer,
            "sources": _first_three_source_links(used_metadatas),
            "context_used": len(context_parts),
        }