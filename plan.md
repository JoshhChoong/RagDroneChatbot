Plan: Canadian RAG Dataset & QA
TL;DR — Build a reproducible local RAG pipeline that ingests Transport Canada / Government of Canada sources, chunks and embeds text into a local Chroma DB, and answers questions with a local open-source LLM (CPU-friendly). Design a pytest-based QA harness that can inject poisoned evidence and assert retrieval/answering behavior so Phase 2 tests plug in cleanly later.

Steps

Add deps: create requirements.txt listing core packages: chromadb, sentence-transformers, transformers, llama-cpp-python (Windows/CPU fallback), langchain (optional helper), pytest.
Data ingestion
Add Data/download_transport_canada.py: scripts to download/normalize HTML/PDF/CSV from Government portals and save raw files to Data/raw/.
Add Data/convert_to_text.py: normalize PDFs/HTML/CSV → plain text/markdown into Data/processed/.
RAG pipeline code (create RagPipeline files)
RagPipeline/ingest.py: orchestrates reading Data/processed/ and creating document objects with metadata.
RagPipeline/chunker.py: deterministic chunker using character/token windows + overlap (configurable, default: 1000 token / 200 overlap).
RagPipeline/embeddings.py: wrapper to produce embeddings using sentence-transformers/all-MiniLM-L6-v2.
RagPipeline/vectorstore.py: create/restore Chroma DB with local persistence and metadata fields (source, url, doc_id, chunk_index).
RagPipeline/retriever.py: semantic search top-k + MMR fallback; returns scored doc slices and provenance.
RagPipeline/reader.py: adapter to local model via llama_cpp.Llama (use ggml quantized weights) with prompt template that includes provenance citations.
RagPipeline/pipeline.py: end-to-end runner for ingest → chunk → embed → store → simple QA example.
Add config: RagPipeline/config.yaml for chunk sizes, embed model, chroma path, model backend.
Main and examples
Update main.py to include CLI for ingest, serve-query, and demo QA query showing sources returned with answers.
Add README updates in README.MD documenting setup and CPU-only notes.
QA / Poisoning test design (Phase 2 — implementable now)
Tests layout:
tests/test_poisoning.py: pytest tests that build an in-memory chroma instance from small fixture docs, inject a poisoned doc, run retriever+reader, and assert behavior.
tests/fixtures/clean_doc.md and tests/fixtures/poisoned_doc.md: small docs that contain conflicting facts (poisoned doc contains a plausible but false assertion).
Helper API in code: RagPipeline/testing_helpers.py exposing build_local_store(docs), inject_poison(doc_id, content), simulate_query(query, top_k=5).
Test assertions (design-level):
Verify retrieved top-k contains provenance metadata including source.
Two test modes:
Baseline: with only clean docs, answer must match ground truth.
Poisoned: after injecting poisoned doc, assert whether reader either (a) correctly prefers authoritative source or (b) includes provenance and flags uncertainty. Tests should allow configurable expected behavior so you can choose strict or permissive models later.
Metrics to capture: precision@k, hallucination count (asserted facts without source), source-consistency rate.
Documentation & CI
Add docs/testing.md explaining the poisoning-test intent, how to add new poisoned fixtures, and how to run tests.
Add simple GitHub Actions workflow (optional later) to run pip install -r requirements.txt + pytest on push.