# RAG Context Compressor

A lightweight, sub-millisecond **Context Pruning & Compression API** for Retrieval-Augmented Generation (RAG) pipelines. It filters out irrelevant filler sentences from retrieved vector-DB chunks before passing them to an LLM.

## Why This Exists
Vector search retrieves fixed-size chunks (e.g., 512 tokens each), meaning **50%–70% of retrieved text is irrelevant noise**. Passing bloated context increases input token costs, slows down Time-To-First-Token (TTFT), and degrades reasoning accuracy (*"Lost in the Middle"*).

`RAG Context Compressor` scores every sentence against the user query using **Okapi BM25 + Query Concept Coverage**, keeps only high-signal sentences, restores original document order, and reports exact token savings.

## Key Capabilities
- **Okapi BM25 Sentence Scoring:** Ranks individual sentences across multiple retrieved documents with zero GPU or heavy model overhead.
- **Order-Preserving Reconstruction:** Reassembles selected sentences in their original narrative sequence so context coherence stays intact.
- **Token Savings Telemetry:** Returns `original_tokens`, `compressed_tokens`, `tokens_saved`, and `compression_pct` per request.

## Quick Start

```bash
pip install -r requirements.txt
uvicorn main:app --port 8001 --reloadcurl -X POST [http://127.0.0.1:8001/v1/rag/compress](http://127.0.0.1:8001/v1/rag/compress) \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the API rate limit for enterprise workspaces?",
    "chunks": [
      "Welcome to our portal. Enterprise workspaces allow 1,200 requests per minute.",
      "Invoices generate on the 1st. Exceeding the rate limit returns HTTP 429."
    ],
    "keep_ratio": 0.5
  }'
