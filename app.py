# ✅ Tracing FIRST — only from tracing.py, no re-registration
from tracing import tracer_provider, tracer

import streamlit as st
import uuid
from opentelemetry import trace
from openinference.instrumentation import using_session          # ✅ Added
from openinference.semconv.trace import SpanAttributes          # ✅ Added
from retrieval import Retriever
from generator import RAGGenerator

# ✅ Session ID — once per user session
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

session_id = st.session_state.session_id

# Initialize components
@st.cache_resource
def load_system():
    retriever = Retriever()
    generator = RAGGenerator()
    return retriever, generator

st.set_page_config(page_title="RBS Student Life Assistant", page_icon="🛡️")
st.title("Student Life Assistant for Rutgers Business School 🛡️")
st.markdown("Ask questions about RBS contacts, events, majors, and student life! Powered by Hybrid Retrieval (FAISS + BM25) and OpenAI.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("View Retrieved Sources"):
                for src in msg["sources"]:
                    st.caption(f"{src['metadata_prefix']} \n\n {src['text']}")

# Chat Input
query = st.chat_input("Ask a question (e.g. 'Who is the contact for MITA?')")

@tracer.chain
def run_rag_pipeline(query: str):
    current_span = trace.get_current_span()
    current_span.set_attribute(SpanAttributes.SESSION_ID, session_id)
    current_span.set_attribute(SpanAttributes.INPUT_VALUE, query)

    retriever, generator = load_system()                        # ✅ Use cached system

    with using_session(session_id):
        retrieved_chunks, intent = retriever.retrieve(query)
        answer = generator.generate_answer(query, retrieved_chunks)

    current_span.set_attribute(SpanAttributes.OUTPUT_VALUE, answer)
    return answer, retrieved_chunks, intent                     # ✅ Now returns 3 values

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with using_session(st.session_state.session_id):
            with tracer.start_as_current_span(
                "Rutgers_Assistant_Workflow",
                attributes={
                    "openinference.span.kind":"CHAIN",
                    "session.id": st.session_state.session_id
                }
    
            with st.spinner("Processing..."):
                try:
                    answer, retrieved_chunks, intent = run_rag_pipeline(query)  # ✅ 3 values
                except FileNotFoundError:
                    st.error("Missing index files. Please run `python ingest.py` first.")
                    st.stop()

    with st.chat_message("assistant"):
        st.markdown(answer)
        with st.expander("View Retrieved Sources"):
            for src in retrieved_chunks:
                st.caption(f"{src['metadata_prefix']} \n\n {src['text']}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": retrieved_chunks
    })

# Sidebar metrics
with st.sidebar:
    st.header("Pipeline Info")
    st.markdown("- **Embeddings:** all-MiniLM-L6-v2")
    st.markdown("- **Vector DB:** FAISS (Dense)")
    st.markdown("- **Keyword:** BM25 (Sparse)")
    st.markdown("- **Reranker:** Reciprocal Rank Fusion")
    st.markdown("- **LLM:** gpt-4o-mini")
    
