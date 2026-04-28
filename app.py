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

# 1. Get your API Key from Streamlit secrets or environment variables
api_key = st.secrets.get("PHOENIX_API_KEY")

# 2. Register ONCE. If API key exists, it goes to Cloud. If not, it stays Local.
tracer_provider = register(
    project_name="RU_Student_Assistant_Test",
    endpoint="https://app.phoenix.arize.com/v1/traces" if api_key else "http://localhost:6006/v1/traces",
    api_key=api_key,
    auto_instrument=True
)

OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

# 3. Create the tracer tool
tracer = tracer_provider.get_tracer(__name__)

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
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
  
        # 1. START THE MASTER SPAN
        with tracer.start_as_current_span(
            "Rutgers_Assistant_Workflow", 
            attributes={
                "openinference.span.kind": "CHAIN",
                "input.value": user_query,
            }
        ) as root_span:
            
            # 2. GET THE CURRENT CONTEXT
            # This is the "glue" that tells the other files: "You belong to me!"
            current_context = otel_trace.set_span_in_context(root_span)

            try:
                retriever, generator = load_system()

                # 3. RUN RETRIEVER (Force it into the context)
                with otel_trace.use_span(root_span, context=current_context):
                    retrieved_chunks, intent = retriever.retrieve(user_query, top_k=5)
                
                context_text = "\n\n".join([c['text'] for c in retrieved_chunks])
                root_span.set_attribute("retrieval.documents", context_text)
                
                # 4. RUN GENERATOR (Force it into the context)
                with otel_trace.use_span(root_span, context=current_context):
                    answer = generator.generate_answer(user_query, retrieved_chunks)
                
                root_span.set_attribute("output.value", answer)
                root_span.set_status(StatusCode.OK) 

            except Exception as e:
                root_span.set_status(StatusCode.ERROR, str(e))
                answer = "Error occurred."
                retrieved_chunks = []

        # UI Logic
        with st.chat_message("assistant"):
            st.markdown(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": retrieved_chunks})
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
    
