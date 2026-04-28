import streamlit as st
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
# 1. Phoenix Initialization (ONLY ONCE)
# =========================
@st.cache_resource
def init_phoenix():
    api_key = st.secrets.get("PHOENIX_API_KEY")

    tracer_provider = register(
        project_name="RU_Student_Assistant_Test",
        endpoint="https://app.phoenix.arize.com/v1/traces",
        api_key=api_key,
    )

    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
    return tracer_provider


tracer_provider = init_phoenix()
tracer = otel_trace.get_tracer(__name__)


# =========================
# 2. Load RAG components
# =========================
@st.cache_resource
def load_system():
    return Retriever(), RAGGenerator()


retriever, generator = load_system()


# =========================
# 3. Streamlit setup
# =========================
st.set_page_config(page_title="RBS Student Life Assistant", page_icon="🛡️")
st.title("Student Life Assistant for Rutgers Business School 🛡️")


# Persistent session ID (CRITICAL FIX)
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================
# 4. Chat history UI
# =========================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


query = st.chat_input("Ask a question...")


# =========================
# 5. Main RAG pipeline
# =========================
def run_rag(user_query: str):
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    # ROOT TRACE (full workflow)
    with tracer.start_as_current_span(
        "rag.workflow",
        attributes={
            "openinference.span.kind": "CHAIN",
            SpanAttributes.INPUT_VALUE: user_query,
            "session.id": st.session_state.session_id,
        },
    ) as span:

        try:
            # -------------------------
            # Retrieval step
            # -------------------------
            with st.spinner("Searching Rutgers Knowledge Base..."):
                retrieved_chunks, intent = retriever.retrieve(user_query)

            context_text = "\n\n".join([c["text"] for c in retrieved_chunks])

            span.set_attribute("retrieval.intent", intent)
            span.set_attribute("retrieval.context", context_text)

            # -------------------------
            # Generation step
            # -------------------------
            with st.spinner("Generating Answer..."):
                with tracer.start_as_current_span(
                    "llm.generate",
                    attributes={
                        "openinference.span.kind": "LLM",
                        SpanAttributes.INPUT_VALUE: user_query,
                        "retrieval.intent": intent,
                        "retrieval.context": context_text,
                    },
                ) as llm_span:

                    answer = generator.generate_answer(user_query, retrieved_chunks)

                    llm_span.set_attribute(SpanAttributes.OUTPUT_VALUE, answer)

            # finalize root span
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, answer)
            span.set_status(Status(StatusCode.OK))

        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            answer = f"Error: {str(e)}"
            retrieved_chunks = []


    # =========================
    # UI response rendering
    # =========================
    with st.chat_message("assistant"):
        st.markdown(answer)

        if retrieved_chunks:
            with st.expander("View Retrieved Sources"):
                for i, src in enumerate(retrieved_chunks):
                    st.caption(f"Document {i+1}")
                    st.write(src.get("text", ""))


    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": retrieved_chunks,
        }
    )

    # Flush traces (important for Streamlit Cloud)
    tracer_provider.force_flush()
    time.sleep(1)


# =========================
# 6. Trigger pipeline
# =========================
if query:
    run_rag(query)


# =========================
# 7. Sidebar
# =========================
with st.sidebar:
    st.header("Pipeline Info")
    st.markdown("- **Embeddings:** all-MiniLM-L6-v2")
    st.markdown("- **Vector DB:** FAISS (Dense)")
    st.markdown("- **Keyword:** BM25 (Sparse)")
    st.markdown("- **Reranker:** Reciprocal Rank Fusion")
    st.markdown("- **LLM:** llama-3.1-8b-instant (Groq)")
