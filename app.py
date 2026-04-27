import streamlit as st
import os
from retrieval import Retriever
from generator import RAGGenerator
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
import uuid
from openinference.instrumentation import using_session

# --- CLEAN TRACING SETUP (REPLACE YOUR TOP SECTION) ---

# 1. Get your API Key from Streamlit secrets or environment variables
api_key = st.secrets.get("PHOENIX_API_KEY")

# 2. Register ONCE. If API key exists, it goes to Cloud. If not, it stays Local.
tracer_provider = register(
    project_name="dataset-evaluator-ce7f0780fd136b682bafdec3",
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
  if query:
      # 1. User messages
      st.session_state.messages.append({"role": "user", "content": query})
      with st.chat_message("user"):
          st.markdown(query)
  
      # WRAP EVERYTHING IN THE SESSION CONTEXT
      # This ensures every trace in this 'visit' is grouped together in Arize
      with using_session(st.session_state.session_id):
          
          with tracer.start_as_current_span(
              "Rutgers_Assistant_Workflow", 
              attributes={
                  "openinference.span.kind": "CHAIN",
                  "session.id": st.session_state.session_id  # Manual session tag
              }
          ) as root_span:
              
              # 2. Process query
              with st.spinner("Searching specific knowledge base..."):
                  try:
                      retriever, generator = load_system()
                  except FileNotFoundError:
                      st.error("Error: Missing index files.")
                      st.stop()
                  
                  # Child Span 1 (The wall of text you saw earlier)
                  retrieved_chunks, intent = retriever.retrieve(query, top_k=5)
                  context_text = "\n\n".join([c['text'] for c in retrieved_chunks])
                  root_span.set_attribute("retrieval.documents", context_text)
              
              with st.spinner(f"Generating answer (Router detected intent: {intent})..."):
                  # Child Span 2 (The Chatbot Response)
                  answer = generator.generate_answer(query, retrieved_chunks)
                  
                  # Explicitly record the answer on the parent span so you can see it in the list!
                  root_span.set_attribute("output.value", answer)
  
      # 3. Bot response
      with st.chat_message("assistant"):
          st.markdown(answer)
          with st.expander("View Retrieved Sources"):
              for src in retrieved_chunks:
                  st.caption(f"{src['metadata_prefix']} \n\n {src['text']}")
      
      st.session_state.messages.append({"role": "assistant", "content": answer, "sources": retrieved_chunks})
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
    
