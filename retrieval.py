import os
import json
import pickle
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from opentelemetry import trace
from openinference.semconv.trace import SpanAttributes
from opentelemetry.trace import Status, StatusCode

# =========================
# Tracer (NO register() here — handled in app.py)
# =========================
tracer = trace.get_tracer(__name__)

OUTPUT_DIR = "output"

class Retriever:
    def __init__(self):
        self.chunk_data = pickle.load(
            open(os.path.join(OUTPUT_DIR, "chunked_data.pkl"), "rb")
        )
        self.bm25 = pickle.load(
            open(os.path.join(OUTPUT_DIR, "bm25_index.pkl"), "rb")
        )
        self.faiss_index = faiss.read_index(
            os.path.join(OUTPUT_DIR, "vector_index.faiss")
        )
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    # =========================
    # Intent Router
    # =========================
    def query_router(self, query: str):
        q = query.lower()

        if any(term in q for term in ["contact", "email", "reach out", "coordinator"]):
            return "contacts"
        elif any(term in q for term in ["event", "this week", "seminar", "fair"]):
            return "events"
        elif any(term in q for term in ["credits", "minor", "requirement", "prerequisite"]):
            return "requirements"
        elif any(term in q for term in ["club", "organization", "society"]):
            return "student_life"
        return "general"

    # =========================
    # Reciprocal Rank Fusion
    # =========================
    def reciprocal_rank_fusion(self, dense_results, sparse_results, k=60):
        scores = {}

        for rank, idx in enumerate(dense_results):
            scores[idx] = scores.get(idx, 0.0) + 1 / (k + rank + 1)

        for rank, idx in enumerate(sparse_results):
            scores[idx] = scores.get(idx, 0.0) + 1 / (k + rank + 1)

        return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    # =========================
    # MAIN RETRIEVAL FUNCTION
    # =========================
    def retrieve(self, query: str, top_k: int = 5, router_override: bool = True):

        with tracer.start_as_current_span(
            "retriever.retrieve",
            attributes={
                SpanAttributes.INPUT_VALUE: query,
                "retrieval.top_k": top_k,
            },
        ) as span:

            try:
                # -------------------------
                # Intent routing
                # -------------------------
                intent = self.query_router(query)
                span.set_attribute("retrieval.intent", intent)

                # -------------------------
                # BM25 retrieval span
                # -------------------------
                with tracer.start_as_current_span("retriever.bm25") as bm25_span:
                    tokenized_query = query.lower().split()
                    bm25_scores = self.bm25.get_scores(tokenized_query)

                    if router_override and intent != "general":
                        for i, chunk in enumerate(self.chunk_data):
                            if chunk.get("category") == intent:
                                bm25_scores[i] *= 1.5

                    sparse_top_n = np.argsort(bm25_scores)[::-1][:15]
                    bm25_span.set_attribute("retrieval.sparse.top_n", len(sparse_top_n))

                # -------------------------
                # Vector search span
                # -------------------------
                with tracer.start_as_current_span("retriever.vector_search") as vec_span:
                    emb = self.model.encode([query])
                    _, dense_top_n = self.faiss_index.search(emb, 15)
                    dense_top_n = dense_top_n[0]
                    vec_span.set_attribute("retrieval.dense.top_n", len(dense_top_n))

                # -------------------------
                # Fusion span
                # -------------------------
                with tracer.start_as_current_span("retriever.fusion") as fusion_span:
                    fused_indices = self.reciprocal_rank_fusion(
                        dense_top_n, sparse_top_n
                    )
                    final_indices = fused_indices[:top_k]
                    fusion_span.set_attribute("retrieval.final_k", len(final_indices))

                # -------------------------
                # Build results
                # -------------------------
                results = []
                for idx in final_indices:
                    chunk = self.chunk_data[idx]
                    results.append(
                        {
                            "text": chunk.get("text", ""),
                            "metadata_prefix": chunk.get("metadata_prefix", ""),
                            "category": chunk.get("category", ""),
                            "index": idx,
                        }
                    )

                # -------------------------
                # Attach documents for Phoenix RAG eval
                # ✅ Serialized to JSON string — required by OpenTelemetry
                # -------------------------
                span.set_attribute(
                    "retrieval.documents",
                    json.dumps([
                        {
                            "id": str(i),
                            "text": r["text"],
                            "metadata": {
                                "category": r.get("category", ""),
                            },
                        }
                        for i, r in enumerate(results)
                    ])
                )

                span.set_status(Status(StatusCode.OK))
                return results, intent

            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                return [], "error"

if __name__ == "__main__":
    r = Retriever()
    res, intent = r.retrieve("Who is the contact for the MITA program?")
    print(f"Router intent: {intent}")
    print("Top result:", res[0]['text'])  # ✅ Fixed key name
