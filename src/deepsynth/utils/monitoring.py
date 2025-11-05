#!/usr/bin/env python3
"""Observability utilities for production deployments.

This module provides comprehensive monitoring infrastructure:
- OpenTelemetry distributed tracing
- Prometheus metrics collection
- Structured logging utilities
- Performance profiling helpers

Features:
    - Zero-overhead when monitoring is disabled
    - Automatic context propagation for distributed systems
    - GDPR-compliant with privacy controls
    - Integration with popular monitoring backends

Example:
    >>> from deepsynth.utils.monitoring import (
    ...     init_monitoring,
    ...     trace_function,
    ...     record_metric,
    ...     get_metrics_handler,
    ... )
    >>>
    >>> # Initialize monitoring
    >>> init_monitoring(
    ...     service_name="deepsynth-api",
    ...     enable_tracing=True,
    ...     enable_metrics=True,
    ... )
    >>>
    >>> # Trace a function
    >>> @trace_function("ocr.inference")
    >>> def run_ocr(image):
    ...     return model.predict(image)
    >>>
    >>> # Record metrics
    >>> record_metric("ocr.latency", 123.4, {"model": "deepseek"})
    >>>
    >>> # Get Prometheus metrics endpoint
    >>> metrics_handler = get_metrics_handler()
"""

from __future__ import annotations

import functools
import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, TypeVar
from dataclasses import dataclass, field

# Optional dependencies
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None

try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        Summary,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = None
    Histogram = None
    Gauge = None

LOGGER = logging.getLogger(__name__)

# Global state
_monitoring_initialized = False
_tracer: Optional[Any] = None
_metrics_registry: Optional[Any] = None
_metrics: Dict[str, Any] = {}


F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class MonitoringConfig:
    """Configuration for monitoring infrastructure.

    Args:
        service_name: Name of the service (e.g., "deepsynth-api")
        enable_tracing: Enable OpenTelemetry tracing (default: False)
        enable_metrics: Enable Prometheus metrics (default: False)
        enable_logging: Enable structured logging (default: True)
        otlp_endpoint: OTLP exporter endpoint (default: None for console export)
        privacy_mode: Enable GDPR-compliant privacy mode (default: True)
        sample_rate: Trace sampling rate 0-1 (default: 1.0 = all traces)

    Example:
        >>> config = MonitoringConfig(
        ...     service_name="deepsynth-api",
        ...     enable_tracing=True,
        ...     enable_metrics=True,
        ...     privacy_mode=True,
        ... )
    """

    service_name: str = "deepsynth"
    enable_tracing: bool = False
    enable_metrics: bool = False
    enable_logging: bool = True
    otlp_endpoint: Optional[str] = None
    privacy_mode: bool = True
    sample_rate: float = 1.0
    environment: str = "development"


def init_monitoring(
    service_name: str = "deepsynth",
    enable_tracing: bool = False,
    enable_metrics: bool = False,
    enable_logging: bool = True,
    otlp_endpoint: Optional[str] = None,
    privacy_mode: bool = True,
    sample_rate: float = 1.0,
    environment: str = "development",
) -> bool:
    """Initialize monitoring infrastructure.

    Args:
        service_name: Name of the service
        enable_tracing: Enable OpenTelemetry tracing
        enable_metrics: Enable Prometheus metrics
        enable_logging: Enable structured logging
        otlp_endpoint: OTLP endpoint (None = console export)
        privacy_mode: Enable GDPR privacy mode
        sample_rate: Trace sampling rate 0-1
        environment: Environment name (dev/staging/prod)

    Returns:
        True if successfully initialized

    Example:
        >>> init_monitoring(
        ...     service_name="deepsynth-api",
        ...     enable_tracing=True,
        ...     enable_metrics=True,
        ...     otlp_endpoint="http://localhost:4317",
        ... )
    """
    global _monitoring_initialized, _tracer, _metrics_registry

    if _monitoring_initialized:
        LOGGER.warning("Monitoring already initialized")
        return True

    config = MonitoringConfig(
        service_name=service_name,
        enable_tracing=enable_tracing,
        enable_metrics=enable_metrics,
        enable_logging=enable_logging,
        otlp_endpoint=otlp_endpoint,
        privacy_mode=privacy_mode,
        sample_rate=sample_rate,
        environment=environment,
    )

    # Initialize tracing
    if config.enable_tracing:
        if not OTEL_AVAILABLE:
            LOGGER.warning(
                "OpenTelemetry not available. Install with: "
                "pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp"
            )
        else:
            _init_tracing(config)

    # Initialize metrics
    if config.enable_metrics:
        if not PROMETHEUS_AVAILABLE:
            LOGGER.warning(
                "Prometheus client not available. Install with: "
                "pip install prometheus-client"
            )
        else:
            _init_metrics(config)

    # Configure logging
    if config.enable_logging:
        _init_logging(config)

    _monitoring_initialized = True
    LOGGER.info(
        f"✅ Monitoring initialized: service={service_name}, "
        f"tracing={enable_tracing}, metrics={enable_metrics}, "
        f"environment={environment}"
    )

    return True


def _init_tracing(config: MonitoringConfig):
    """Initialize OpenTelemetry tracing."""
    global _tracer

    # Create resource with service info
    resource = Resource.create(
        {
            "service.name": config.service_name,
            "service.version": "0.2.0",
            "deployment.environment": config.environment,
        }
    )

    # Create tracer provider
    provider = TracerProvider(resource=resource)

    # Configure exporter
    if config.otlp_endpoint:
        # OTLP exporter for production (Jaeger, Tempo, etc.)
        exporter = OTLPSpanExporter(endpoint=config.otlp_endpoint, insecure=True)
        LOGGER.info(f"Using OTLP exporter: {config.otlp_endpoint}")
    else:
        # Console exporter for development
        exporter = ConsoleSpanExporter()
        LOGGER.info("Using console span exporter (development mode)")

    # Add span processor
    provider.add_span_processor(BatchSpanProcessor(exporter))

    # Set as global tracer provider
    trace.set_tracer_provider(provider)

    # Get tracer
    _tracer = trace.get_tracer(__name__)

    # Auto-instrument requests library
    try:
        RequestsInstrumentor().instrument()
    except Exception as e:
        LOGGER.warning(f"Failed to instrument requests: {e}")

    LOGGER.info("✅ OpenTelemetry tracing initialized")


def _init_metrics(config: MonitoringConfig):
    """Initialize Prometheus metrics."""
    global _metrics_registry, _metrics

    # Create registry
    _metrics_registry = CollectorRegistry()

    # Create common metrics
    _metrics["requests_total"] = Counter(
        "deepsynth_requests_total",
        "Total number of requests",
        ["service", "endpoint", "status"],
        registry=_metrics_registry,
    )

    _metrics["request_duration_seconds"] = Histogram(
        "deepsynth_request_duration_seconds",
        "Request duration in seconds",
        ["service", "endpoint"],
        registry=_metrics_registry,
    )

    _metrics["inference_latency_ms"] = Histogram(
        "deepsynth_inference_latency_ms",
        "Inference latency in milliseconds",
        ["model", "batch_size"],
        buckets=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
        registry=_metrics_registry,
    )

    _metrics["batch_size"] = Histogram(
        "deepsynth_batch_size",
        "Batch size distribution",
        ["model"],
        buckets=[1, 2, 4, 8, 16, 32, 64],
        registry=_metrics_registry,
    )

    _metrics["errors_total"] = Counter(
        "deepsynth_errors_total",
        "Total number of errors",
        ["service", "error_type"],
        registry=_metrics_registry,
    )

    _metrics["active_requests"] = Gauge(
        "deepsynth_active_requests",
        "Number of active requests",
        ["service"],
        registry=_metrics_registry,
    )

    LOGGER.info("✅ Prometheus metrics initialized")


def _init_logging(config: MonitoringConfig):
    """Initialize structured logging."""
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Add privacy filter if enabled
    if config.privacy_mode:
        # Add filter to redact sensitive information
        class PrivacyFilter(logging.Filter):
            def filter(self, record):
                # Redact common sensitive fields
                if hasattr(record, "msg"):
                    msg = str(record.msg)
                    # Simple redaction (enhance as needed)
                    for sensitive in ["password", "token", "key", "secret"]:
                        if sensitive in msg.lower():
                            record.msg = msg.replace(
                                sensitive, f"<REDACTED_{sensitive.upper()}>"
                            )
                return True

        logging.getLogger().addFilter(PrivacyFilter())

    LOGGER.info("✅ Structured logging initialized")


def trace_function(span_name: Optional[str] = None) -> Callable[[F], F]:
    """Decorator to trace function execution.

    Args:
        span_name: Custom span name (default: function name)

    Example:
        >>> @trace_function("ocr.inference")
        >>> def run_ocr(image):
        ...     return model.predict(image)
        >>>
        >>> result = run_ocr(image)  # Automatically traced
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not _tracer or not OTEL_AVAILABLE:
                # Tracing not enabled, just run function
                return func(*args, **kwargs)

            name = span_name or f"{func.__module__}.{func.__name__}"

            with _tracer.start_as_current_span(name) as span:
                # Add function metadata
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)

                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("success", True)
                    return result
                except Exception as e:
                    span.set_attribute("success", False)
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e))
                    span.record_exception(e)
                    raise

        return wrapper

    return decorator


@contextmanager
def trace_context(span_name: str, attributes: Optional[Dict[str, Any]] = None):
    """Context manager for tracing code blocks.

    Args:
        span_name: Name for the span
        attributes: Optional attributes to add to span

    Example:
        >>> with trace_context("data.preprocessing", {"batch_size": 32}):
        ...     data = preprocess(raw_data)
    """
    if not _tracer or not OTEL_AVAILABLE:
        # Tracing not enabled, just yield
        yield
        return

    with _tracer.start_as_current_span(span_name) as span:
        # Add attributes
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        try:
            yield span
        except Exception as e:
            span.set_attribute("error", True)
            span.record_exception(e)
            raise


def record_metric(
    metric_name: str,
    value: float,
    labels: Optional[Dict[str, str]] = None,
    metric_type: str = "histogram",
):
    """Record a metric value.

    Args:
        metric_name: Name of the metric
        value: Metric value
        labels: Optional labels for the metric
        metric_type: Type of metric (histogram, counter, gauge)

    Example:
        >>> record_metric("ocr.latency", 123.4, {"model": "deepseek"})
        >>> record_metric("requests.total", 1, {"endpoint": "/api/ocr"}, "counter")
    """
    if not _metrics_registry or not PROMETHEUS_AVAILABLE:
        return

    # Map to existing metrics
    if metric_name in _metrics:
        metric = _metrics[metric_name]
        labels = labels or {}

        if metric_type == "counter":
            metric.labels(**labels).inc(value)
        elif metric_type == "histogram":
            metric.labels(**labels).observe(value)
        elif metric_type == "gauge":
            metric.labels(**labels).set(value)


def get_metrics_handler():
    """Get Prometheus metrics handler for HTTP endpoint.

    Returns:
        Function that returns (content, content_type) for metrics endpoint

    Example:
        >>> # Flask
        >>> @app.route("/metrics")
        >>> def metrics():
        ...     content, content_type = get_metrics_handler()()
        ...     return Response(content, mimetype=content_type)
        >>>
        >>> # FastAPI
        >>> @app.get("/metrics")
        >>> def metrics():
        ...     content, content_type = get_metrics_handler()()
        ...     return Response(content=content, media_type=content_type)
    """
    if not _metrics_registry or not PROMETHEUS_AVAILABLE:
        return lambda: (b"Metrics not available", "text/plain")

    def handler():
        content = generate_latest(_metrics_registry)
        return content, CONTENT_TYPE_LATEST

    return handler


class PerformanceTimer:
    """Context manager for timing code execution.

    Example:
        >>> with PerformanceTimer() as timer:
        ...     expensive_operation()
        >>> print(f"Took {timer.elapsed_ms:.2f}ms")
    """

    def __init__(self):
        self.start_time = 0.0
        self.end_time = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return (self.end_time - self.start_time) * 1000.0

    @property
    def elapsed_seconds(self) -> float:
        """Elapsed time in seconds."""
        return self.end_time - self.start_time


def log_performance(func: F, threshold_ms: float = 1000.0) -> F:
    """Decorator to log slow function executions.

    Args:
        func: Function to decorate
        threshold_ms: Log warning if execution exceeds this (ms)

    Example:
        >>> @log_performance
        >>> def slow_function():
        ...     time.sleep(2)
        >>>
        >>> slow_function()  # Logs: "slow_function took 2000ms (slow!)"
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with PerformanceTimer() as timer:
            result = func(*args, **kwargs)

        if timer.elapsed_ms > threshold_ms:
            LOGGER.warning(
                f"{func.__name__} took {timer.elapsed_ms:.2f}ms (slow!)"
            )
        else:
            LOGGER.debug(
                f"{func.__name__} took {timer.elapsed_ms:.2f}ms"
            )

        return result

    return wrapper


__all__ = [
    "MonitoringConfig",
    "init_monitoring",
    "trace_function",
    "trace_context",
    "record_metric",
    "get_metrics_handler",
    "PerformanceTimer",
    "log_performance",
]
