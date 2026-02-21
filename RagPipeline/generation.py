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
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# For fallback when metadata has no source_url (e.g. pre-existing DB)
_PROCESSED_DATA_DIR = Path(__file__).resolve().parent.parent / "Data" / "files" / "processed"

load_dotenv()

# --- OpenRouter (preferred when OPENROUTER_API_KEY is set) ---
# https://openrouter.ai/docs/api-reference/chat-completions
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
OPENROUTER_MODEL_FALLBACKS = os.getenv(
    "OPENROUTER_MODEL_FALLBACKS",
    "google/gemini-2.0-flash-exp,anthropic/claude-3-haiku,meta-llama/llama-3.1-8b-instruct"
).split(",")

# --- Gemini direct (used when OPENROUTER_API_KEY is not set) ---
# Set GEMINI_MODEL to override; GEMINI_FAST_MODE=1 = fewer retries, shorter waits
# GEMINI_SKIP_GEMMA=1 (default): do not fall back to Gemma when Gemini hits quota
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-preview")
GEMINI_MODEL_FALLBACKS = os.getenv("GEMINI_MODEL_FALLBACKS", "gemini-2.0-flash,gemini-1.5-flash,gemini-1.5-pro").split(",")
GEMINI_FAST_MODE = os.getenv("GEMINI_FAST_MODE", "1").strip().lower() in ("1", "true", "yes")
GEMINI_SKIP_GEMMA = os.getenv("GEMINI_SKIP_GEMMA", "1").strip().lower() in ("1", "true", "yes")

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
        # When Gemini hits quota, API list can leave only Gemma (e.g. gemma-3-1b-it). Skip Gemma by default so we return context fallback instead of weak answers.
        if GEMINI_SKIP_GEMMA:
            ordered = [m for m in ordered if "gemma" not in m.lower()]
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


def _gemini_generate_content(api_key: str, prompt: str, temperature: float = 0.1, max_tokens: int = 500) -> tuple:
    """Call Gemini generateContent REST API. On 429, retries then tries fallback models. Returns (text, model_id) or (None, None)."""
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
            resp = requests.post(url, json=payload, headers=headers, timeout=120)
            if resp.ok:
                data = resp.json()
                candidates = data.get("candidates") or []
                if not candidates:
                    raise RuntimeError("Gemini returned no candidates")
                parts = (candidates[0].get("content") or {}).get("parts") or []
                if not parts:
                    return ("", model)
                return ((parts[0].get("text") or "").strip(), model)

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
            return (None, None)
        if last_error:
            idx = models_to_try.index(model)
            next_model = models_to_try[idx + 1] if idx + 1 < len(models_to_try) else "?"
            print(f"Model {model} quota exceeded. Trying next model: {next_model}...")
            last_error = None
    return (None, None)


def _openrouter_generate_content(api_key: str, prompt: str, temperature: float = 0.1, max_tokens: int = 500) -> tuple:
    """Call OpenRouter chat/completions (OpenAI-compatible). Tries OPENROUTER_MODEL then fallbacks. Returns (text, model_id) or (None, None)."""
    url = f"{OPENROUTER_BASE}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    models_to_try = [OPENROUTER_MODEL] + [m.strip() for m in OPENROUTER_MODEL_FALLBACKS if m.strip()]

    for model in models_to_try:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=120)
            if resp.ok:
                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    continue
                msg = choices[0].get("message") or {}
                text = (msg.get("content") or "").strip()
                if text:
                    return (text, model)
                continue
            try:
                err = resp.json()
            except Exception:
                err = resp.text
            if resp.status_code == 429:
                print(f"OpenRouter quota exceeded (model={model}). Trying next model...")
                continue
            if resp.status_code in (404, 400, 401):
                print(f"OpenRouter {resp.status_code} for {model}: {err}. Trying next model...")
                continue
            print(f"OpenRouter {resp.status_code}: {err}")
        except requests.RequestException as e:
            print(f"OpenRouter request failed for {model}: {e}")
    return (None, None)


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


def _is_comparison_query(query: str) -> bool:
    """True if the user is asking to compare/contrast two or more distinct concepts (e.g. SFOC vs RPAS)."""
    q = query.strip().lower()
    if any(
        p in q
        for p in (
            "difference between",
            "difference of",
            "compare",
            "contrast",
            " vs ",
            " versus ",
        )
    ):
        return True
    if " and " in q and ("difference" in q or "different" in q or "same" in q):
        return True
    return False


def _is_procedural_query(query: str) -> bool:
    """True if the user is asking HOW to do something (e.g. get a licence, register, become a pilot)—needs steps and key requirements."""
    q = query.strip().lower()
    if not ("how" in q or "what do i need" in q or "steps" in q or "process" in q):
        return False
    return any(
        w in q
        for w in (
            "licen", "certificate", "register", "pilot", "permit", "fly",
            "get ", "become", "apply", "requirement", "need to",
        )
    )


def _is_basic_advanced_certificate_comparison(query: str) -> bool:
    """True if the user is comparing Basic vs Advanced (pilot) certificate—answer from context or standard Canadian distinctions. Excludes exam-specific questions."""
    q = query.strip().lower()
    if "basic" not in q and "advanced" not in q:
        return False
    if "exam" in q or "test" in q or "assessment" in q:
        return False  # exam questions use exam_comparison_instruction instead
    return "certificate" in q or "licen" in q or "pilot" in q or _is_comparison_query(query)


def _is_basic_advanced_exam_comparison(query: str) -> bool:
    """True if the user is comparing Basic vs Advanced EXAM (not certificate privileges)—answer about exam content, difficulty, and post-exam flight review. Excludes preparation questions."""
    q = query.strip().lower()
    if "preparation" in q or "prep " in q or "prepare" in q or "studying" in q or "study " in q:
        return False  # preparation questions use preparation_instruction instead
    if ("basic" not in q and "advanced" not in q) or ("exam" not in q and "test" not in q and "assessment" not in q):
        return False
    return _is_comparison_query(query) or ("difference" in q or "different" in q or "between" in q)


def _is_preparation_for_exam_query(query: str) -> bool:
    """True if the user is asking about PREPARATION for the exam (what to study, how prep differs)—answer about prep, NOT weight thresholds or certificate requirement."""
    q = query.strip().lower()
    prep_words = ("preparation", "prep", "prepare", "preparing", "study", "studying", "ready", "apply to the exam")
    if not any(w in q for w in prep_words):
        return False
    return "exam" in q or "test" in q or "basic" in q or "advanced" in q or "differ" in q


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
    def __init__(self, openrouter_api_key: Optional[str] = None, gemini_api_key: Optional[str] = None):
        self.openrouter_api_key = (openrouter_api_key or "").strip() or None
        self.gemini_api_key = (gemini_api_key or "").strip() or None
        # Retriever/embeddings use env (GEMINI_API_KEY / USE_GEMINI_EMBEDDINGS); pass a dummy if no Gemini key so Chroma still inits
        self.retriever = CodebaseRetriever(self.gemini_api_key or self.openrouter_api_key or "dummy-key")

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

        comparison_instruction = ""
        if _is_comparison_query(query):
            comparison_instruction = (
                "\n\nIMPORTANT: The user is asking for a COMPARISON between two (or more) distinct concepts. "
                "In your answer: (1) Clearly define or describe EACH concept separately (e.g. what is SFOC, what is RPAS). "
                "(2) Then explain how they differ or how they relate. "
                "Do NOT describe a single combined or hybrid term (e.g. 'SFOC-RPAS') as if it were one of the things being compared—treat each term the user asked about as a distinct concept and compare those.\n"
            )

        procedural_instruction = ""
        if _is_procedural_query(query):
            procedural_instruction = (
                "\n\nIMPORTANT: The user is asking HOW to do something (e.g. get a licence, register). Give a genuinely helpful answer:\n"
                "(1) A clear step-by-step where the Context supports it (e.g. register drone if applicable → Basic vs Advanced → take exam → flight review if Advanced).\n"
                "(2) Key context from the Context: Basic vs Advanced certificate, age requirements, drone weight rules (e.g. 250g–25kg), registration, and when flight school or flight review is required vs optional.\n"
                "(3) Be precise: if you mention flight school or any option, say whether it is required or optional and for which path (Basic/Advanced).\n"
                "Use a short numbered list or short paragraphs if helpful. Reference the Drone Management Portal when relevant. Stay on topic; no filler.\n"
            )

        certificate_comparison_instruction = ""
        if _is_basic_advanced_certificate_comparison(query):
            certificate_comparison_instruction = (
                "\n\nIMPORTANT: The user is asking about Basic vs Advanced (pilot) certificate in Canada. Give a direct, helpful answer. "
                "Use the Context when it describes these certificates. If the Context does not spell out every difference, you may include the standard Canadian distinctions: "
                "controlled airspace access (Advanced can operate in controlled airspace with approval; Basic is more restricted), "
                "minimum distance from bystanders, flight review requirement (Advanced typically requires a flight review), and operational privileges. "
                "Cite Transport Canada or the Drone Management Portal for full details. Do NOT reply with 'the context does not contain' or 'I cannot answer'—answer the question.\n"
            )

        exam_comparison_instruction = ""
        if _is_basic_advanced_exam_comparison(query):
            exam_comparison_instruction = (
                "\n\nIMPORTANT: The user is asking about the **EXAMS** (Basic vs Advanced exam), NOT certificate privileges. Answer about the exams themselves:\n"
                "(1) **Exam content/difficulty:** Advanced exam is harder and covers more: complex airspace, NAV CANADA coordination, emergency procedures; Basic exam is more limited.\n"
                "(2) **After the exam:** Advanced requires a flight review after passing the exam; Basic does not require a flight review.\n"
                "(3) Do NOT answer about certificate privileges (controlled airspace, EVLOS, sheltered operations) unless you explain each term in one line. Prefer plain language; avoid dropping jargon without context. Focus on exam differences.\n"
                "Use the Context when it describes exam content or flight review. Cite Transport Canada or the Drone Management Portal for full details.\n"
            )

        preparation_instruction = ""
        if _is_preparation_for_exam_query(query):
            preparation_instruction = (
                "\n\nIMPORTANT: The user is asking about **PREPARATION** for the drone exam. Give **actionable** advice—concrete topics and resources. Do NOT give a vague answer like 'study resources are available' or 'a recommended learning path exists'.\n"
                "Include specific topics to study when relevant (from Context or standard Canadian prep): Transport Canada study guide, airspace rules, NOTAMs, weather, RPAS regulations, practice exams. Name where to find them (e.g. Drone Management Portal, Transport Canada website). For Basic exam: airspace, weather, RPAS rules, practice questions. For Advanced: add NAV CANADA, controlled airspace, emergency procedures.\n"
                "Do NOT answer about 250g or certificate requirement unless the user asked. If the Context names specific resources or study guides, list them. End with a clear next step (e.g. use the Transport Canada study guide, take practice exams on the Portal).\n"
            )

        prompt = f"""You are a helpful assistant for Canadian drone safety and regulations. Prefer the Context below when it answers the question.

Rules:
- Be direct and precise. No filler or vague hedging ("you may need to", "it depends") unless the Context actually says so.
- State only what the Context says. Cite specific numbers or limits when the Context mentions them (e.g. age, weight 250g–25kg). Do NOT list section titles or link names.
- For "how do I" questions (e.g. get a licence): give a structured answer with steps and key requirements (Basic vs Advanced, registration, exam, flight review when applicable). For simple factual questions, a few clear sentences are enough.
- If you mention something (e.g. flight school), say whether it is required or optional and for whom.
{comparison_instruction}
{procedural_instruction}
{certificate_comparison_instruction}
{exam_comparison_instruction}
{preparation_instruction}

Context:
{context}

Question: {query}

Answer:"""

        model_used = None
        if self.openrouter_api_key:
            answer, model_used = _openrouter_generate_content(
                self.openrouter_api_key, prompt, temperature=0.1, max_tokens=2048
            )
        elif self.gemini_api_key:
            answer, model_used = _gemini_generate_content(
                self.gemini_api_key, prompt, temperature=0.1, max_tokens=2048
            )
        else:
            answer = None

        if answer is None:
            model_used = None
            summary_len = 800
            summary = (context.strip()[:summary_len] + "...") if len(context.strip()) > summary_len else context.strip()
            answer = (
                "I couldn't reach the AI (quota or model unavailable). "
                "Try again later or check your API quota. Here are relevant excerpts from the documentation:\n\n" + summary
            )

        return {
            "answer": answer,
            "model": model_used,
            "sources": _first_three_source_links(used_metadatas),
            "context_used": len(context_parts),
        }