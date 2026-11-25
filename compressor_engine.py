import re
import math
import time
from collections import Counter
from typing import List, Dict, Any


class RAGContextCompressor:
    """
    Sentence-level RAG Context Compression Engine.
    Ranks sentences across retrieved chunks using Okapi BM25 + query term coverage,
    prunes low-signal context within a target compression ratio or token budget,
    and preserves original document order for coherent LLM reasoning.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._stopwords = {
            "the", "is", "at", "which", "on", "and", "a", "an", "in", "to",
            "for", "of", "with", "as", "by", "that", "this", "it", "from",
            "are", "was", "were", "be", "or", "if", "into", "about", "can",
        }

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
        return [t for t in tokens if t not in self._stopwords and len(t) > 1]

    def _split_sentences(self, text: str) -> List[str]:
        raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        return [s.strip() for s in raw_sentences if len(s.strip()) >= 8]

    def _estimate_tokens(self, text: str) -> int:
        # Standard ~4 chars per token heuristic for English/technical text
        return max(1, math.ceil(len(text) / 4))

    def compress(
        self,
        query: str,
        chunks: List[str],
        keep_ratio: float = 0.5,
        min_score_threshold: float = 0.05,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()

        if not (0.1 <= keep_ratio <= 1.0):
            raise ValueError("keep_ratio must be between 0.1 and 1.0")

        query_tokens = self._tokenize(query)
        sentence_records: List[Dict[str, Any]] = []

        for doc_idx, chunk in enumerate(chunks):
            if not chunk or not chunk.strip():
                continue
            for sent in self._split_sentences(chunk):
                tokens = self._tokenize(sent)
                sentence_records.append({
                    "global_idx": len(sentence_records),
                    "doc_idx": doc_idx,
                    "text": sent,
                    "tokens": tokens,
                    "token_count": self._estimate_tokens(sent),
                })

        total_original_tokens = sum(r["token_count"] for r in sentence_records)

        if not sentence_records:
            return {
                "compressed_context": "",
                "selected_sentences": [],
                "metrics": {
                    "original_sentences": 0,
                    "retained_sentences": 0,
                    "original_tokens": 0,
                    "compressed_tokens": 0,
                    "tokens_saved": 0,
                    "compression_pct": 0.0,
                    "latency_ms": round((time.perf_counter() - start_time) * 1000, 3),
                },
            }

        # Fallback if query has no content tokens after stopword removal
        if not query_tokens:
            keep_count = max(1, math.ceil(len(sentence_records) * keep_ratio))
            selected = sentence_records[:keep_count]
            compressed_text = " ".join(s["text"] for s in selected)
            compressed_tokens = sum(s["token_count"] for s in selected)
            saved = max(0, total_original_tokens - compressed_tokens)
            return {
                "compressed_context": compressed_text,
                "selected_sentences": [
                    {"doc_index": s["doc_idx"], "score": 0.0, "text": s["text"]}
                    for s in selected
                ],
                "metrics": {
                    "original_sentences": len(sentence_records),
                    "retained_sentences": len(selected),
                    "original_tokens": total_original_tokens,
                    "compressed_tokens": compressed_tokens,
                    "tokens_saved": saved,
                    "compression_pct": round((saved / max(total_original_tokens, 1)) * 100, 2),
                    "latency_ms": round((time.perf_counter() - start_time) * 1000, 3),
                },
            }

        # Compute Okapi BM25 scores across all extracted sentences
        n_docs = len(sentence_records)
        avg_dl = sum(len(r["tokens"]) for r in sentence_records) / max(n_docs, 1)

        doc_freqs: Counter = Counter()
        for r in sentence_records:
            for unique_tok in set(r["tokens"]):
                doc_freqs[unique_tok] += 1

        query_set = set(query_tokens)

        for r in sentence_records:
            doc_tokens = r["tokens"]
            dl = len(doc_tokens)
            if dl == 0:
                r["score"] = 0.0
                continue

            tf_counts = Counter(doc_tokens)
            bm25_score = 0.0

            for q_tok in query_tokens:
                if q_tok not in tf_counts:
                    continue
                df = doc_freqs[q_tok]
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                tf = tf_counts[q_tok]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / max(avg_dl, 1.0)))
                bm25_score += idf * ((tf * (self.k1 + 1.0)) / denom)

            # Query coverage boost (rewards sentences matching distinct query concepts)
            overlap = len(query_set.intersection(set(doc_tokens))) / len(query_set)
            r["score"] = round(bm25_score * (1.0 + 0.5 * overlap), 4)

        # Rank by relevance, filter noise, and restore original reading order
        ranked = sorted(sentence_records, key=lambda x: x["score"], reverse=True)
        target_count = max(1, math.ceil(n_docs * keep_ratio))

        filtered = [r for r in ranked if r["score"] >= min_score_threshold][:target_count]
        if not filtered:
            filtered = ranked[:1]

        ordered_selection = sorted(filtered, key=lambda x: x["global_idx"])
        compressed_text = " ".join(s["text"] for s in ordered_selection)
        compressed_tokens = sum(s["token_count"] for s in ordered_selection)
        tokens_saved = max(0, total_original_tokens - compressed_tokens)

        return {
            "compressed_context": compressed_text,
            "selected_sentences": [
                {
                    "doc_index": s["doc_idx"],
                    "score": s["score"],
                    "text": s["text"],
                }
                for s in ordered_selection
            ],
            "metrics": {
                "original_sentences": n_docs,
                "retained_sentences": len(ordered_selection),
                "original_tokens": total_original_tokens,
                "compressed_tokens": compressed_tokens,
                "tokens_saved": tokens_saved,
                "compression_pct": round((tokens_saved / max(total_original_tokens, 1)) * 100, 2),
                "latency_ms": round((time.perf_counter() - start_time) * 1000, 3),
            },
        }
