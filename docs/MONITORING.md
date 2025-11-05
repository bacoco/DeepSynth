# Monitoring & Observability Guide

Complete guide to monitoring DeepSynth OCR pipeline in production with OpenTelemetry, Prometheus, and structured logging.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [OpenTelemetry Tracing](#opentelemetry-tracing)
4. [Prometheus Metrics](#prometheus-metrics)
5. [Structured Logging](#structured-logging)
6. [Dashboard Setup](#dashboard-setup)
7. [Alerting](#alerting)
8. [Best Practices](#best-practices)

---

## Overview

DeepSynth provides comprehensive observability through three pillars:

1. **Traces** (OpenTelemetry) - Distributed request tracing
2. **Metrics** (Prometheus) - Time-series performance data
3. **Logs** (Structured logging) - Application events and errors

### Architecture

```
┌─────────────────┐
│   Application   │
│   (DeepSynth)   │
└────────┬────────┘
         │
    ┌────┴────┬────────────┬──────────┐
    │         │            │          │
    ▼         ▼            ▼          ▼
┌────────┐ ┌──────┐  ┌─────────┐ ┌──────┐
│ OTLP   │ │Prom  │  │  Logs   │ │ W&B  │
│Export  │ │Client│  │ Filter  │ │(opt) │
└───┬────┘ └───┬──┘  └────┬────┘ └──────┘
    │          │           │
    ▼          ▼           ▼
┌────────┐ ┌──────┐  ┌──────────┐
│ Jaeger │ │Prom  │  │ ELK/     │
│/Tempo  │ │Server│  │ Loki     │
└────────┘ └──────┘  └──────────┘
```

---

## Quick Start

### 1. Initialize Monitoring

```python
from deepsynth.utils.monitoring import init_monitoring

init_monitoring(
    service_name="deepsynth-api",
    enable_tracing=True,
    enable_metrics=True,
    otlp_endpoint="http://localhost:4317",
    environment="production",
)
```

### 2. Start Monitoring Stack (Docker)

```bash
# Start Jaeger (tracing)
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 4317:4317 \
  jaegertracing/all-in-one:latest

# Start Prometheus (metrics)
docker run -d --name prometheus \
  -p 9090:9090 \
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus

# Start Grafana (dashboards)
docker run -d --name grafana \
  -p 3000:3000 \
  grafana/grafana
```

### 3. Access UIs

- **Jaeger:** http://localhost:16686
- **Prometheus:** http://localhost:9090
- **Grafana:** http://localhost:3000 (admin/admin)
- **Metrics Endpoint:** http://localhost:9090/metrics

---

## OpenTelemetry Tracing

### Automatic Function Tracing

```python
from deepsynth.utils.monitoring import trace_function

@trace_function("ocr.inference")
def run_ocr(image):
    """This function is automatically traced."""
    result = model.predict(image)
    return result

# Usage
result = run_ocr(image)  # Trace appears in Jaeger
```

### Manual Context Tracing

```python
from deepsynth.utils.monitoring import trace_context

def process_batch(images):
    with trace_context("batch.processing", {"batch_size": len(images)}):
        # Preprocessing
        with trace_context("batch.preprocessing"):
            preprocessed = [preprocess(img) for img in images]

        # Inference
        with trace_context("batch.inference"):
            results = model.batch_predict(preprocessed)

        return results
```

### Adding Attributes to Spans

```python
from deepsynth.utils.monitoring import trace_context

with trace_context("ocr.inference") as span:
    span.set_attribute("model.name", "deepseek-vl2")
    span.set_attribute("batch.size", 8)
    span.set_attribute("image.width", 1024)
    span.set_attribute("image.height", 768)

    result = model.predict(image)

    span.set_attribute("prediction.length", len(result))
    span.set_attribute("success", True)
```

### Example Trace View

```
Service: deepsynth-api
Trace ID: 3fa85f6457174562b3fc2c963f66afa6

├─ api.request (200ms)
│  ├─ preprocessing (20ms)
│  │  ├─ image.decode (5ms)
│  │  └─ image.resize (15ms)
│  ├─ ocr.inference (150ms)
│  │  ├─ model.tokenize (10ms)
│  │  ├─ model.forward (120ms)
│  │  └─ model.decode (20ms)
│  └─ postprocessing (30ms)
```

### Distributed Tracing

Traces propagate across services automatically:

```python
# Service A
import requests

@trace_function("service_a.call_service_b")
def call_service_b(data):
    # Trace context automatically propagated
    response = requests.post("http://service-b/predict", json=data)
    return response.json()

# Service B (receives trace context)
@trace_function("service_b.predict")
def predict(data):
    # This span is child of service_a.call_service_b
    return model.predict(data)
```

---

## Prometheus Metrics

### Available Metrics

#### Request Metrics

```python
# deepsynth_requests_total
# Total number of requests
# Labels: service, endpoint, status

from deepsynth.utils.monitoring import record_metric

record_metric(
    "requests_total",
    1,
    {"service": "api", "endpoint": "/predict", "status": "200"},
    metric_type="counter",
)
```

#### Latency Metrics

```python
# deepsynth_inference_latency_ms
# Inference latency histogram
# Labels: model, batch_size

record_metric(
    "inference_latency_ms",
    123.4,
    {"model": "deepseek-vl2", "batch_size": "8"},
    metric_type="histogram",
)
```

#### Batch Size Distribution

```python
# deepsynth_batch_size
# Batch size histogram
# Labels: model

record_metric(
    "batch_size",
    8,
    {"model": "deepseek-vl2"},
    metric_type="histogram",
)
```

#### Error Tracking

```python
# deepsynth_errors_total
# Total errors counter
# Labels: service, error_type

record_metric(
    "errors_total",
    1,
    {"service": "api", "error_type": "OOMError"},
    metric_type="counter",
)
```

#### Active Requests

```python
# deepsynth_active_requests
# Current active requests gauge
# Labels: service

record_metric(
    "active_requests",
    5,
    {"service": "api"},
    metric_type="gauge",
)
```

### Metrics Endpoint

Expose metrics for Prometheus scraping:

```python
from flask import Flask, Response
from deepsynth.utils.monitoring import get_metrics_handler

app = Flask(__name__)

@app.route("/metrics")
def metrics():
    handler = get_metrics_handler()
    content, content_type = handler()
    return Response(content, mimetype=content_type)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9090)
```

### Prometheus Configuration

Create `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'deepsynth-api'
    static_configs:
      - targets: ['localhost:9090']
        labels:
          service: 'deepsynth-api'
          environment: 'production'

  - job_name: 'deepsynth-worker'
    static_configs:
      - targets: ['worker-1:9090', 'worker-2:9090']
        labels:
          service: 'deepsynth-worker'
          environment: 'production'
```

### PromQL Queries

#### Average Latency

```promql
# Average inference latency (last 5 minutes)
rate(deepsynth_inference_latency_ms_sum[5m]) /
rate(deepsynth_inference_latency_ms_count[5m])
```

#### Request Rate

```promql
# Requests per second
rate(deepsynth_requests_total[1m])
```

#### Error Rate

```promql
# Error percentage
(
  sum(rate(deepsynth_errors_total[5m]))
  /
  sum(rate(deepsynth_requests_total[5m]))
) * 100
```

#### P95 Latency

```promql
# 95th percentile latency
histogram_quantile(0.95,
  rate(deepsynth_inference_latency_ms_bucket[5m])
)
```

#### Throughput by Batch Size

```promql
# Requests per second by batch size
sum by (batch_size) (
  rate(deepsynth_batch_size_count[5m])
)
```

---

## Structured Logging

### Basic Logging

```python
import logging

logger = logging.getLogger(__name__)

logger.info("OCR inference started", extra={
    "model": "deepseek-vl2",
    "batch_size": 8,
    "image_size": (1024, 768),
})

logger.warning("High latency detected", extra={
    "latency_ms": 2500,
    "threshold_ms": 1000,
})

logger.error("Inference failed", extra={
    "error": "OOMError",
    "batch_size": 32,
}, exc_info=True)
```

### Performance Logging

```python
from deepsynth.utils.monitoring import log_performance

@log_performance
def slow_function():
    # If execution > 1000ms, logs warning
    time.sleep(2)

# Logs: "slow_function took 2000ms (slow!)"
```

### Privacy-Aware Logging

Logging automatically redacts PII when `REDACT_PII_IN_LOGS=true`:

```python
# Automatic PII redaction
logger.info(f"User password: {password}")
# Logs: "User password: <REDACTED_PASSWORD>"

logger.info(f"API token: {token}")
# Logs: "API token: <REDACTED_TOKEN>"
```

### Log Aggregation

#### ELK Stack

```yaml
# filebeat.yml
filebeat.inputs:
  - type: log
    enabled: true
    paths:
      - /var/log/deepsynth/*.log
    json.keys_under_root: true
    json.add_error_key: true

output.elasticsearch:
  hosts: ["localhost:9200"]
  index: "deepsynth-%{+yyyy.MM.dd}"
```

#### Loki

```yaml
# promtail-config.yaml
server:
  http_listen_port: 9080

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: deepsynth
    static_configs:
      - targets:
          - localhost
        labels:
          job: deepsynth-api
          __path__: /var/log/deepsynth/*.log
```

---

## Dashboard Setup

### Grafana Dashboard

#### 1. Add Data Sources

```bash
# Prometheus
curl -X POST http://admin:admin@localhost:3000/api/datasources \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Prometheus",
    "type": "prometheus",
    "url": "http://prometheus:9090",
    "access": "proxy"
  }'

# Jaeger
curl -X POST http://admin:admin@localhost:3000/api/datasources \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Jaeger",
    "type": "jaeger",
    "url": "http://jaeger:16686",
    "access": "proxy"
  }'
```

#### 2. Import DeepSynth Dashboard

Create `deepsynth-dashboard.json`:

```json
{
  "dashboard": {
    "title": "DeepSynth OCR Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [{
          "expr": "rate(deepsynth_requests_total[1m])"
        }]
      },
      {
        "title": "P95 Latency",
        "targets": [{
          "expr": "histogram_quantile(0.95, rate(deepsynth_inference_latency_ms_bucket[5m]))"
        }]
      },
      {
        "title": "Error Rate",
        "targets": [{
          "expr": "(sum(rate(deepsynth_errors_total[5m])) / sum(rate(deepsynth_requests_total[5m]))) * 100"
        }]
      }
    ]
  }
}
```

Import:

```bash
curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @deepsynth-dashboard.json
```

### Key Panels

#### Request Rate Over Time
```promql
rate(deepsynth_requests_total[1m])
```

#### Latency Percentiles
```promql
histogram_quantile(0.50, rate(deepsynth_inference_latency_ms_bucket[5m])) # P50
histogram_quantile(0.95, rate(deepsynth_inference_latency_ms_bucket[5m])) # P95
histogram_quantile(0.99, rate(deepsynth_inference_latency_ms_bucket[5m])) # P99
```

#### Error Rate Percentage
```promql
(sum(rate(deepsynth_errors_total[5m])) / sum(rate(deepsynth_requests_total[5m]))) * 100
```

#### Throughput by Model
```promql
sum by (model) (rate(deepsynth_requests_total[1m]))
```

---

## Alerting

### Prometheus Alerting Rules

Create `alerts.yml`:

```yaml
groups:
  - name: deepsynth_alerts
    interval: 30s
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: |
          (
            sum(rate(deepsynth_errors_total[5m]))
            /
            sum(rate(deepsynth_requests_total[5m]))
          ) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }}"

      # High latency
      - alert: HighLatency
        expr: |
          histogram_quantile(0.95,
            rate(deepsynth_inference_latency_ms_bucket[5m])
          ) > 2000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High inference latency"
          description: "P95 latency is {{ $value }}ms"

      # Service down
      - alert: ServiceDown
        expr: up{job="deepsynth-api"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "DeepSynth service is down"
          description: "Service {{ $labels.instance }} is unreachable"

      # OOM errors
      - alert: OutOfMemory
        expr: |
          rate(deepsynth_errors_total{error_type="OOMError"}[5m]) > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Out of memory errors detected"
          description: "OOM errors on {{ $labels.instance }}"
```

### Alertmanager Configuration

```yaml
# alertmanager.yml
global:
  resolve_timeout: 5m

route:
  group_by: ['alertname', 'cluster']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'slack'

receivers:
  - name: 'slack'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
        channel: '#alerts'
        title: 'DeepSynth Alert'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_KEY'
```

---

## Best Practices

### 1. Sampling Strategy

```python
# Sample 10% of traces in production
init_monitoring(
    service_name="deepsynth-api",
    enable_tracing=True,
    sample_rate=0.1,  # 10%
    environment="production",
)

# Sample 100% in development
init_monitoring(
    service_name="deepsynth-api",
    enable_tracing=True,
    sample_rate=1.0,  # 100%
    environment="development",
)
```

### 2. Privacy Compliance

```python
# Enable privacy mode (GDPR compliant)
init_monitoring(
    service_name="deepsynth-api",
    enable_tracing=True,
    privacy_mode=True,  # Redacts PII, anonymizes metrics
)
```

### 3. Performance Monitoring

```python
from deepsynth.utils.monitoring import PerformanceTimer

def process_request():
    timings = {}

    with PerformanceTimer() as timer:
        preprocess()
    timings["preprocess"] = timer.elapsed_ms

    with PerformanceTimer() as timer:
        infer()
    timings["infer"] = timer.elapsed_ms

    with PerformanceTimer() as timer:
        postprocess()
    timings["postprocess"] = timer.elapsed_ms

    return timings
```

### 4. Custom Metrics

```python
from prometheus_client import Counter, Histogram

# Custom counter
custom_counter = Counter(
    'deepsynth_custom_events_total',
    'Custom events',
    ['event_type'],
)

custom_counter.labels(event_type='user_action').inc()

# Custom histogram
custom_histogram = Histogram(
    'deepsynth_custom_duration_seconds',
    'Custom operation duration',
    ['operation'],
)

with custom_histogram.labels(operation='preprocessing').time():
    preprocess_data()
```

### 5. Health Checks

```python
from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/health")
def health():
    # Check dependencies
    checks = {
        "model_loaded": model is not None,
        "gpu_available": torch.cuda.is_available(),
        "memory_ok": get_memory_usage() < 0.9,
    }

    status = "healthy" if all(checks.values()) else "unhealthy"
    code = 200 if status == "healthy" else 503

    return jsonify({
        "status": status,
        "checks": checks,
        "timestamp": time.time(),
    }), code
```

---

## Additional Resources

- **OpenTelemetry Docs:** https://opentelemetry.io/docs/
- **Prometheus Docs:** https://prometheus.io/docs/
- **Grafana Docs:** https://grafana.com/docs/
- **Jaeger Docs:** https://www.jaegertracing.io/docs/

---

**Questions?** See [deepseek_ocr_pipeline.md](./deepseek_ocr_pipeline.md) or file an issue.
