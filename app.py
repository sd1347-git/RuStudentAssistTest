import streamlit as st
import os
from retrieval import Retriever
from generator import RAGGenerator
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
import uuid
from openinference.instrumentation import using_session
from opentelemetry.trace import StatusCode
from opentelemetry import trace as otel_trace

# --- CLEAN TRACING SETUP (REPLACE YOUR TOP SECTION) ---

# 1. Initialize Phoenix
tracer_provider = register(
    project_name="RU_Student_Assistant_Test",
    endpoint="https://app.phoenix.arize.com/v1/traces" if st.secrets.get("PHOENIX_API_KEY") else "http://localhost:6006/v1/traces",
    api_key=st.secrets.get("PHOENIX_API_KEY"),
)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
tracer = otel_trace.get_tracer(__name__)

# Initialize a persistent session ID for the user
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# No more subprocess needed! 
# The app will just check if the folder you uploaded is there.
if not os.path.exists("output"):
    st.error("Index folder not found. Please ensure 'output' is uploaded to GitHub.")
else:
    # Initialize your retriever using the files in the output folder
    # retriever = Retriever(index_path="output")
    pass

# Initialize components
@st.cache_resource
def load_system():
    # Only loads once
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

def get_rutgers_answer(user_query: str):
    if user_query:
        # 1. UI Update
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
  
        # 2. Start the ONE span that matters for the Evaluator
        with tracer.start_as_current_span(
            "Rutgers_Assistant_Workflow", 
            attributes={
                "openinference.span.kind": "CHAIN",
                "input.value": user_query,
            }
        ) as root_span:
            try:
                # Load your components
                retriever, generator = load_system()

                # Step A: Get Documents
                with st.spinner("Searching Rutgers Knowledge Base..."):
                    retrieved_chunks, intent = retriever.retrieve(user_query)
                    
                    # Attach context to the ROOT so the judge can see it
                    context_text = "\n\n".join([c['text'] for c in retrieved_chunks])
                    root_span.set_attribute("retrieval.documents", context_text)
                
                # Step B: Get Answer
                with st.spinner("Generating Answer..."):
                    answer = generator.generate_answer(user_query, retrieved_chunks)
                    
                    # Attach answer to the ROOT so the judge can see it
                    root_span.set_attribute("output.value", answer)
                
                # Step C: Success!
                root_span.set_status(StatusCode.OK)

            except Exception as e:
                # If it breaks, record why
                root_span.set_status(StatusCode.ERROR, str(e))
                answer = f"Technical error: {str(e)}"
                retrieved_chunks = []

        # 3. Final UI Update
        with st.chat_message("assistant"):
            st.markdown(answer)
            if retrieved_chunks:
                with st.expander("View Retrieved Sources"):
                    for src in retrieved_chunks:
                        st.caption(f"{src.get('metadata_prefix', 'Source')} \n\n {src['text']}")
      
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": retrieved_chunks})
        
        # Force the data to Phoenix immediately
        tracer_provider.force_flush()
        
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
    
