"""Prometheus instrumentation for the FastAPI backend.

Wraps `prometheus-fastapi-instrumentator` so request count, latency and
in-progress requests are exposed at `/metrics` with one call. Prometheus
(configured in `docker/prometheus/prometheus.yml`) scrapes that endpoint;
Grafana reads from Prometheus.
"""

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator


def instrument_app(app: FastAPI) -> Instrumentator:
    """Attach Prometheus instrumentation to a FastAPI app.

    Call once, right after the app is constructed. Exposes metrics at
    `GET /metrics` and excludes it from the metrics it collects about itself.

    Args:
        app: The FastAPI application to instrument.

    Returns:
        The `Instrumentator` instance, in case the caller wants to add
        custom metrics before `expose()` is called elsewhere.
    """
    instrumentator = Instrumentator(
        excluded_handlers=["/metrics", "/health"],
        should_group_status_codes=True,
        should_ignore_untemplated=True,
    )
    instrumentator.instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
    return instrumentator
