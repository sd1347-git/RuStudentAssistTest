import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import streamlit as st
from opentelemetry import trace
from phoenix.otel import register # Bring this back
import streamlit as st

# Force it to register itself specifically
tracer = register(
    project_name="RU_Student_Assistant_Test",
    endpoint="https://app.phoenix.arize.com/v1/traces",
    api_key=st.secrets.get("PHOENIX_API_KEY")
).get_tracer(__name__)

# 1. Define the directory where your pkl and faiss files live
OUTPUT_DIR = "output" 

class Retriever:
    def __init__(self):
        # Load indexes and data using the defined OUTPUT_DIR
        self.chunk_data = pickle.load(open(os.path.join(OUTPUT_DIR, "chunked_data.pkl"), "rb"))
        self.bm25 = pickle.load(open(os.path.join(OUTPUT_DIR, "bm25_index.pkl"), "rb"))
        self.faiss_index = faiss.read_index(os.path.join(OUTPUT_DIR, "vector_index.faiss"))
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def query_router(self, query):
        """Simplistic query router to detect specific intents."""
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

    def reciprocal_rank_fusion(self, dense_results, sparse_results, k=60):
        """Combines ranked lists using RRF."""
        scores = {}
        for rank, idx in enumerate(dense_results):
            if idx not in scores:
                scores[idx] = 0.0
            scores[idx] += 1 / (k + rank + 1)
            
        for rank, idx in enumerate(sparse_results):
            if idx not in scores:
                scores[idx] = 0.0
            scores[idx] += 1 / (k + rank + 1)
            
        sorted_indices = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return sorted_indices
      
    def retrieve(self, query, top_k=5, router_override=True):
        # This span nests under the Master Workflow in app.py
        with tracer.start_as_current_span("Retriever.retrieve") as span:
            span.set_attribute("input.value", query)
            
            intent = self.query_router(query)
            span.set_attribute("retrieval.intent", intent)
            
            # 1. Sparse Search (BM25)
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            
            if router_override and intent != "general":
                for i, chunk in enumerate(self.chunk_data):
                    if chunk["category"] == intent:
                        bm25_scores[i] *= 1.5 
            
            sparse_top_n = np.argsort(bm25_scores)[::-1][:15]

            # 2. Dense Search (FAISS)
            emb = self.model.encode([query])
            distances, dense_top_n = self.faiss_index.search(emb, 15)
            dense_top_n = dense_top_n[0]

            # 3. Combine via RRF
            fused_indices = self.reciprocal_rank_fusion(dense_top_n, sparse_top_n)
            final_indices = fused_indices[:top_k]

            results = [self.chunk_data[i] for i in final_indices]
            
            # Add results to attributes for the trace view
            span.set_attribute("retrieval.documents", "\n\n".join([c['text'] for c in results]))
            
            return results, intent

if __name__ == "__main__":
    r = Retriever()
    res, intent = r.retrieve("Who is the contact for the MITA program?")
    print(f"Router intent: {intent}")
    print("Top result:", res[0]['title'])
