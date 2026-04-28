import os

from openai import OpenAI
from opentelemetry import trace
from openinference.semconv.trace import SpanAttributes
from opentelemetry.trace import Status, StatusCode


# =========================
# Tracer (NO register HERE)
# =========================
tracer = trace.get_tracer(__name__)


# =========================
# Groq / OpenAI-compatible client
# =========================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = (
    OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )
    if GROQ_API_KEY
    else None
)

MODEL_NAME = "llama-3.1-8b-instant"


# =========================
# Prompt Template
# =========================
PROMPT_TEMPLATE = """
You are the "Student Life Assistant for Rutgers Business School".

You MUST follow these rules:
1. Use ONLY the provided context.
2. If the answer is not in the context, say:
   "I don't have information about that in my current database."
3. Be concise and factual.
4. Always cite sources like [Source: file/url].

Context:
{context_str}

Question:
{query}
"""


class RAGGenerator:
    def __init__(self):
        self.model_name = MODEL_NAME

    def generate_answer(self, query: str, retrieved_chunks: list):

        with tracer.start_as_current_span(
            "llm.generate_answer",
            attributes={
                SpanAttributes.INPUT_VALUE: query,
                "llm.model": self.model_name,
            },
        ) as span:

            try:
                if not client:
                    raise ValueError("GROQ_API_KEY is not set.")

                # =========================
                # Build context
                # =========================
                context_parts = []

                for i, chunk in enumerate(retrieved_chunks):
                    prefix = chunk.get("metadata_prefix", "")
                    text = chunk.get("text", "")

                    context_parts.append(
                        f"--- Document {i+1} ---\n{prefix}\n{text}\n"
                    )

                context_str = "\n".join(context_parts)

                prompt = PROMPT_TEMPLATE.format(
                    context_str=context_str,
                    query=query,
                )

                # =========================
                # Attach RAG context for Arize
                # =========================
                span.set_attribute("llm.prompt", prompt)
                span.set_attribute("retrieval.context", context_str)

                span.set_attribute(
                    "retrieval.documents",
                    [
                        {
                            "id": str(i),
                            "text": c.get("text", ""),
                            "metadata": {
                                "category": c.get("category", ""),
                            },
                        }
                        for i, c in enumerate(retrieved_chunks)
                    ],
                )

                # =========================
                # LLM Call
                # =========================
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )

                answer = response.choices[0].message.content

                # =========================
                # Output tracking
                # =========================
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, answer)
                span.set_status(Status(StatusCode.OK))

                return answer

            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                return f"Error during generation: {str(e)}"


if __name__ == "__main__":
    print("Run app.py to use the RAG system.")
