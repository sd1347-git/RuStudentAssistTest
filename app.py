import streamlit as st
import os
import uuid
import time
from retrieval import Retriever
from generator import RAGGenerator
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace as otel_trace
from opentelemetry.trace import StatusCode

# 1. Initialize Phoenix for the Cloud
api_key = st.secrets.get("PHOENIX_API_KEY")
project_name = "RU_Student_Assistant_Test"

# We use the register tool but keep a reference to the provider
tracer_provider = register(
    project_name=project_name,
    endpoint="https://app.phoenix.arize.com/v1/traces",
    api_key=api_key,
)

OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
tracer = otel_trace.get_tracer(__name__)

# --- Component Loading ---
@st.cache_resource
def load_system():
    return Retriever(), RAGGenerator()

# --- Page Config ---
st.set_page_config(page_title="RBS Student Life Assistant", page_icon="🛡️")
st.title("Student Life Assistant for Rutgers Business School 🛡️")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- UI Loop ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a question...")

def get_rutgers_answer(user_query: str):
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
  
        # START TRACE
        with tracer.start_as_current_span(
            "Rutgers_Assistant_Workflow", 
            attributes={
                "openinference.span.kind": "CHAIN",
                "input.value": user_query,
            }
        ) as span:
            try:
                retriever, generator = load_system()

                # Step A: Search
                with st.spinner("Searching Rutgers Knowledge Base..."):
                    retrieved_chunks, intent = retriever.retrieve(user_query)
                    context_text = "\n\n".join([c['text'] for c in retrieved_chunks])
                    span.set_attribute("retrieval.documents", context_text)
                
                # Step B: Generate
                with st.spinner("Generating Answer..."):
                    answer = generator.generate_answer(user_query, retrieved_chunks)
                    span.set_attribute("output.value", answer)
                
                span.set_status(StatusCode.OK)

            except Exception as e:
                span.set_status(StatusCode.ERROR, str(e))
                answer = f"I'm sorry, I encountered an error: {str(e)}"
                retrieved_chunks = []

        # UI Update
        with st.chat_message("assistant"):
            st.markdown(answer)
            if retrieved_chunks:
                with st.expander("View Retrieved Sources"):
                    for src in retrieved_chunks:
                        st.caption(f"{src.get('metadata_prefix', 'Source')} \n\n {src['text']}")
      
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": retrieved_chunks})
        
        # --- CRITICAL CLOUD FLUSH ---
        tracer_provider.force_flush()
        time.sleep(2) # Network grace period for Streamlit Cloud
        
    return answer

if query:
    get_rutgers_answer(query)
  
# Sidebar metrics
with st.sidebar:
    st.header("Pipeline Info")
    st.markdown("- **Embeddings:** all-MiniLM-L6-v2")
    st.markdown("- **Vector DB:** FAISS (Dense)")
    st.markdown("- **Keyword:** BM25 (Sparse)")
    st.markdown("- **Reranker:** Reciprocal Rank Fusion")
    st.markdown("- **LLM:** gpt-4o-mini")
    
