"""OpenTelemetry tracing. Spans are written to logs/spans.jsonl and, if
OTEL_EXPORTER_OTLP_ENDPOINT is set (e.g. Langfuse's OTel endpoint), exported there too."""
from __future__ import annotations
import json
import os
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (BatchSpanProcessor, ConsoleSpanExporter,
                                            SimpleSpanProcessor)

from app.config import settings

_ready = False


def init_tracing() -> None:
    global _ready
    if _ready:
        return
    provider = TracerProvider(resource=Resource.create({"service.name": "mf-copilot"}))
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    out = open(settings.log_dir / "spans.jsonl", "a", buffering=1, encoding="utf-8")
    provider.add_span_processor(SimpleSpanProcessor(
        ConsoleSpanExporter(out=out, formatter=lambda s: s.to_json(indent=None) + "\n")))
    otlp_ep = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_ep and "host.docker.internal" in otlp_ep:
        import socket
        try:
            socket.gethostbyname("host.docker.internal")
        except (socket.gaierror, OSError):
            try:
                with socket.create_connection(("127.0.0.1", 3000), timeout=0.2):
                    otlp_ep = otlp_ep.replace("host.docker.internal", "localhost")
            except Exception:
                otlp_ep = None
    if otlp_ep:
        if not otlp_ep.rstrip("/").endswith("/v1/traces"):
            otlp_ep = otlp_ep.rstrip("/") + "/v1/traces"
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_ep)))
        except Exception:
            pass
    trace.set_tracer_provider(provider)
    _ready = True


@contextmanager
def span(name: str, **attrs):
    init_tracing()
    with trace.get_tracer("mf-copilot").start_as_current_span(name) as sp:
        for k, v in attrs.items():
            sp.set_attribute(k, v if isinstance(v, (str, int, float, bool))
                             else json.dumps(v, default=str)[:500])
        yield sp
