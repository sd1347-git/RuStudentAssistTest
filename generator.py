import os
from openai import OpenAI
from opentelemetry import trace # Changed from phoenix.otel import register
from phoenix.otel import register
import streamlit as st

tracer = register(
    project_name="RU_Student_Assistant_Test",
    endpoint="https://app.phoenix.arize.com/v1/traces",
    api_key=st.secrets.get("PHOENIX_API_KEY")
).get_tracer(__name__)

# Setup API Key for Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
) if GROQ_API_KEY else None

MODEL_NAME = 'llama-3.1-8b-instant'

PROMPT_TEMPLATE = """You are the "Student Life Assistant for Rutgers Business School".
Your task is to answer the user's question using ONLY the provided contextual documents below. 

Instructions:
1. Try to answer the question using ONLY the knowledge provided in the Context.
2. If the Context DOES NOT contain the answer, EXPLICITLY state: "I don't have information about that in my current database." Do NOT hallucinate.
3. Keep the answer concise and direct.
4. MUST INCLUDE CITATIONS: Cite the source URL or File that your answer came from at the end of the sentence or block, exactly as [Source: <url/file>].

Context:
{context_str}

User Question:
{query}
"""

class RAGGenerator:
    def __init__(self):
        self.model_name = MODEL_NAME

    def generate_answer(self, query, retrieved_chunks):
        # This will now correctly nest under the Workflow in app.py
        with tracer.start_as_current_span("Generator.generate") as span:
            if not client:
                return "Please set the GROQ_API_KEY environment variable."

            context_parts = []
            for i, chunk in enumerate(retrieved_chunks):
                context_parts.append(f"--- Document {i+1} ---\n{chunk['metadata_prefix']}{chunk['text']}\n")
            
            context_str = "\n".join(context_parts)
            prompt = PROMPT_TEMPLATE.format(context_str=context_str, query=query)

            span.set_attribute("input.value", query)
            span.set_attribute("llm.prompt", prompt) 

            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                answer = response.choices[0].message.content
                
                span.set_attribute("output.value", answer)
                return answer
                
            except Exception as e:
                span.record_exception(e)
                return f"Error during generation: {e}"

if __name__ == "__main__":
    print("Run app.py to interact with the generator.")
