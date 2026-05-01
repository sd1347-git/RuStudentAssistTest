from tracing import tracer
import os
from openai import OpenAI
from opentelemetry import trace
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues

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
        # 1. Manually set the Span Kind to LLM so Phoenix knows to calculate cost
        current_span = trace.get_current_span()
        current_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.LLM.value)
        
        # 2. Use the correct semantic names for the model and provider
        current_span.set_attribute(SpanAttributes.LLM_MODEL_NAME, self.model_name)
        current_span.set_attribute(SpanAttributes.LLM_PROVIDER, "openai")
        current_span.set_attribute(SpanAttributes.INPUT_VALUE, query)

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            answer = response.choices[0].message.content
            usage = response.usage

            # 3. Capture Tokens (The "Fuel" for the cost math)
            current_span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, usage.prompt_tokens)
            current_span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, usage.completion_tokens)
            current_span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, usage.total_tokens)

            current_span.set_attribute(SpanAttributes.OUTPUT_VALUE, answer) 
            return answer
        except Exception as e:
            current_span.record_exception(e)
            return f"Error: {e}"
