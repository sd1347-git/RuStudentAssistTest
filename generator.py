from tracing import tracer
import os
from openai import OpenAI
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

# Setup API Key for Groq
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(
    api_key=OPENAI_API_KEY,
) if OPENAI_API_KEY else None

MODEL_NAME = 'gpt-4o-mini'

PROMPT_TEMPLATE = """You are the "Student Life Assistant for Rutgers Business School".
Your task is to answer the user's question using ONLY the provided contextual documents below. 

Instructions:
1. Try to answer the question using ONLY the knowledge provided in the Context.
2. If the Context DOES NOT contain the answer (e.g. if asked about an event that isn't listed, or a major that isn't mentioned), EXPLICITLY state: "I don't have information about that in my current database." Do NOT hallucinate.
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

    @tracer.chain
    def generate_answer(self, query, retrieved_chunks):
        if not client:
            return "Please set the OPENAI_API_KEY."

        context_str = "\n".join([f"--- Document {i+1} ---\n{c['text']}" for i, c in enumerate(retrieved_chunks)])
        prompt = PROMPT_TEMPLATE.format(context_str=context_str, query=query)

        current_span = trace.get_current_span()
        
        # 1. Use the standard attribute names Arize looks for
        current_span.set_attribute("llm.model_name", self.model_name)
        current_span.set_attribute("llm.provider", "openai")

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            answer = response.choices[0].message.content

            # 2. CAPTURE TOKENS (This is what enables Cost Tracking)
            usage = response.usage
            current_span.set_attribute("llm.token_count.prompt", usage.prompt_tokens)
            current_span.set_attribute("llm.token_count.completion", usage.completion_tokens)
            current_span.set_attribute("llm.token_count.total", usage.total_tokens)

            current_span.set_attribute("output.value", answer) 
            return answer
        except Exception as e:
            current_span.record_exception(e)
            return f"Error: {e}"

if __name__ == "__main__":
    print("Run app.py to interact with the generator.")
