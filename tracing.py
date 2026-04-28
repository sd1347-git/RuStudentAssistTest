# tracing.py
from phoenix.otel import register

tracer_provider = register(
    project_name="RU_Student_Assistant_Test",
)

tracer = tracer_provider.get_tracer(__name__)  # ✅ Phoenix-wrapped tracer with .chain
