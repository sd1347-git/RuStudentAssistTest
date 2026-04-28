import streamlit as st
import os
import uuid
import time

from retrieval import Retriever
from generator import RAGGenerator

from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace as otel_trace
from opentelemetry.trace import Status, StatusCode

# =========================
# 1. Phoenix Initialization
# =========================
@st.cache_resource
def init_phoenix():
    try:
        api_key = st.secrets["PHOENIX_API_KEY"]  # Fails loudly if missing

        # Environment variables — most reliable for Phoenix Cloud
        os.environ["PHOENIX_API_KEY"] = api_key
        os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={api_key}"
        os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"

        tracer_provider = register(
            project_name="RU_Student_Assistant_Test",  # Must match Phoenix Cloud exactly
        )

        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
        return tracer_provider

    except Exception as e:
        st.error(f"❌ Phoenix initialization failed: {e}")
        raise

tracer_provider = init_phoenix()
tracer = otel_trace.get_tracer(__name__)

# =========================
# 2. Load RAG Components
# =========================
@st.cache_resource
def load_system():
    retriever = Retriever()
    generator = RAGGenerator()
    return retriever, generator

# =========================
# 3. Streamlit Setup
# =========================
st.set_page_config(page_title="RBS Student Life Assistant", page_icon="🛡️")

st.title("Student Life Assistant for Rutgers Business School 🛡️")
st.markdown("Ask questions about RBS contacts, events, majors, and student life! Powered by Hybrid Retrieval (FAISS + BM25) and OpenAI.")

# Persistent session ID for trace grouping
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# 4. Chat History UI
# =========================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("View Retrieved Sources"):
                for src in msg["sources"]:
                    st.caption(f"{src['metadata_prefix']} \n\n {src['text']}")

# =========================
# 5. Chat Input
# =========================
query = st.chat_input("Ask a question (e.g. 'Who is the contact for MITA?')")

# =========================
# 6. Main RAG Pipeline
# =========================
if query:
    # User message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # ROOT TRACE (full workflow)
    with tracer.start_as_current_span(
        "rag.workflow",
        attributes={
            "openinference.span.kind": "CHAIN",
            SpanAttributes.INPUT_VALUE: query,
            "session.id": st.session_state.session_id,
        },
    ) as span:

        try:
            # -------------------------
            # Retrieval Step
            # -------------------------
            with st.spinner("Searching specific knowledge base..."):
                try:
                    retriever, generator = load_system()
                except FileNotFoundError:
                    st.error("Error: Missing index files. Please run `python ingest.py` first.")
                    st.stop()

                retrieved_chunks, intent = retriever.retrieve(query, top_k=5)

            context_text = "\n\n".join([c["text"] for c in retrieved_chunks])

            span.set_attribute("retrieval.intent", intent)
            span.set_attribute("retrieval.context", context_text)

            # -------------------------
            # Generation Step
            # -------------------------
            with st.spinner(f"Generating answer (Router detected intent: {intent})..."):
                with tracer.start_as_current_span(
                    "llm.generate",
                    attributes={
                        "openinference.span.kind": "LLM",
                        SpanAttributes.INPUT_VALUE: query,
                        "retrieval.intent": intent,
                        "retrieval.context": context_text,
                    },
                ) as llm_span:

                    answer = generator.generate_answer(query, retrieved_chunks)
                    llm_span.set_attribute(SpanAttributes.OUTPUT_VALUE, answer)

            # Finalize root span
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, answer)
            span.set_status(Status(StatusCode.OK))

        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            answer = f"Error: {str(e)}"
            retrieved_chunks = []

    # -------------------------
    # Bot Response UI
    # -------------------------
    with st.chat_message("assistant"):
        st.markdown(answer)
        if retrieved_chunks:
            with st.expander("View Retrieved Sources"):
                for src in retrieved_chunks:
                    st.caption(f"{src['metadata_prefix']} \n\n {src['text']}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": retrieved_chunks,
    })

    # Flush traces — critical for Streamlit Cloud
    tracer_provider.force_flush(timeout_millis=10000)
    time.sleep(2)

# =========================
# 7. Sidebar
# =========================
with st.sidebar:
    st.header("Pipeline Info")
    st.markdown("- **Embeddings:** all-MiniLM-L6-v2")
    st.markdown("- **Vector DB:** FAISS (Dense)")
    st.markdown("- **Keyword:** BM25 (Sparse)")
    st.markdown("- **Reranker:** Reciprocal Rank Fusion")
    st.markdown("- **LLM:** gpt-4o-mini")
