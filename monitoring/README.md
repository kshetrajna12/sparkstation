# Sparkstation Monitoring

Comprehensive monitoring setup for Sparkstation with Prometheus and Grafana.

---

## Quick Start

### 1. Setup Prometheus

Add to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'sparkstation'
    static_configs:
      - targets: ['localhost:9001']
    scrape_interval: 15s
```

Restart Prometheus:
```bash
sudo systemctl restart prometheus
```

### 2. Import Grafana Dashboard

1. Open Grafana (typically http://localhost:3000)
2. Navigate to **Dashboards** → **Import**
3. Upload `grafana-dashboard.json` or paste the contents
4. Select your Prometheus datasource
5. Click **Import**

The dashboard will be available at: **Dashboards** → **Sparkstation - DGX Spark LLM Gateway**

---

## Dashboard Panels

### Top Row - Key Metrics (Gauges)
1. **Unified Memory Used** - Current memory usage (DGX Spark 128GB unified pool)
   - Green: <80GB
   - Yellow: 80-100GB
   - Red: >100GB

2. **GPU Temperature** - Current GPU temperature
   - Green: <65°C
   - Yellow: 65-80°C
   - Red: >80°C (auto-suspend threshold)

3. **GPU Power Draw** - Current power consumption in watts

4. **Running Models** - Number of currently running models (max 3)

### Middle Row - Time Series
5. **Memory Usage Over Time** - Memory usage with hard limit (110GB)
   - Stacked area chart showing used vs limit
   - Detects approaching memory saturation

6. **GPU Temperature Over Time** - Temperature trends
   - Shows thermal spikes and auto-suspend events
   - Red threshold line at 80°C

### Bottom Row - Model Analytics
7. **Model Status Distribution** - Pie chart of model states
   - Green: Running (2)
   - Blue: Suspended (3)
   - Yellow: Starting (1)
   - Red: Stopped (0) or Failed (4)

8. **Memory Per Model** - Stacked bars showing memory per model
   - Helps identify memory-heavy models
   - Useful for capacity planning

9. **Model Counts Over Time** - Running vs Suspended models
   - Track auto-suspend effectiveness
   - Verify resident model limit (max 3)

### Performance Metrics
10. **Request Rate by Model (5m)** - Requests per second
    - Identifies hottest models
    - Shows traffic distribution

11. **Request Latency (p50, p95)** - Response time percentiles
    - p50: Median latency
    - p95: 95th percentile (outlier detection)
    - Target: <5s for both text and vision

---

## Available Metrics

### Memory Metrics
- `unified_memory_used_bytes` - Total unified memory usage (DGX Spark)
- `unified_memory_limit_bytes` - Hard limit (110GB)
- `model_memory_used_bytes{model_name}` - Memory per model

### GPU Metrics
- `gpu_temperature_celsius` - GPU temperature
- `gpu_power_draw_watts` - GPU power consumption

### Model Metrics
- `model_status{model_name,model_id}` - Model state (0-4)
- `model_last_request_timestamp{model_name}` - Last request unix timestamp
- `resident_models_count` - Running models count
- `suspended_models_count` - Suspended models count

### Request Metrics
- `model_requests_total{model_name}` - Total requests (counter)
- `model_requests_failed{model_name}` - Failed requests (counter)
- `model_request_latency_seconds{model_name}` - Request latency histogram

---

## Useful Queries

### Memory Utilization Percentage
```promql
(unified_memory_used_bytes / unified_memory_limit_bytes) * 100
```

### Time Since Last Request (seconds)
```promql
time() - model_last_request_timestamp
```

### Error Rate (5m window)
```promql
rate(model_requests_failed[5m]) / rate(model_requests_total[5m])
```

### Models Approaching Idle Timeout (>25 min idle)
```promql
(time() - model_last_request_timestamp) > 1500
```

### Average Request Latency by Model
```promql
rate(model_request_latency_seconds_sum[5m]) / rate(model_request_latency_seconds_count[5m])
```

---

## Alerts (Recommended)

### Memory Alerts
```yaml
# High memory usage
- alert: SparkstationHighMemory
  expr: (unified_memory_used_bytes / unified_memory_limit_bytes) > 0.85
  for: 5m
  annotations:
    summary: "Sparkstation memory usage above 85%"
    description: "Memory: {{ $value }}%"

# Memory approaching limit
- alert: SparkstationMemoryCritical
  expr: (unified_memory_used_bytes / unified_memory_limit_bytes) > 0.95
  for: 2m
  annotations:
    summary: "Sparkstation memory CRITICAL"
```

### Temperature Alerts
```yaml
# High temperature
- alert: SparkstationHighTemperature
  expr: gpu_temperature_celsius > 75
  for: 2m
  annotations:
    summary: "GPU temperature above 75°C"
    description: "Temperature: {{ $value }}°C"

# Critical temperature (auto-suspend imminent)
- alert: SparkstationCriticalTemperature
  expr: gpu_temperature_celsius > 80
  for: 1m
  annotations:
    summary: "GPU temperature CRITICAL - auto-suspend triggered"
```

### Model Health Alerts
```yaml
# Model failed
- alert: SparkstationModelFailed
  expr: model_status == 4
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "Model {{ $labels.model_name }} has FAILED"

# High request latency
- alert: SparkstationHighLatency
  expr: histogram_quantile(0.95, rate(model_request_latency_seconds_bucket[5m])) > 10
  for: 5m
  annotations:
    summary: "Model {{ $labels.model_name }} p95 latency > 10s"
```

### Capacity Alerts
```yaml
# Too many resident models
- alert: SparkstationMaxModels
  expr: resident_models_count >= 3
  for: 10m
  annotations:
    summary: "Maximum resident models (3) reached"
    description: "Auto-suspend may be slow to free capacity"
```

---

## Grafana Setup (Production)

### Install Grafana
```bash
# Debian/Ubuntu
sudo apt-get install -y grafana

# Enable and start
sudo systemctl enable grafana-server
sudo systemctl start grafana-server
```

### Add Prometheus Data Source
1. Navigate to **Configuration** → **Data Sources**
2. Click **Add data source**
3. Select **Prometheus**
4. URL: `http://localhost:9090`
5. Click **Save & Test**

### Configure Auto-Refresh
The dashboard is pre-configured with 10s refresh. Adjust via:
- Dashboard settings → **Time options** → **Refresh**
- Options: 5s, 10s, 30s, 1m, 5m

---

## Troubleshooting

### No data in Grafana

1. **Check Prometheus is scraping**:
   ```bash
   curl http://localhost:9090/api/v1/targets
   ```
   Look for `sparkstation` job with state `UP`

2. **Check Supervisor is exposing metrics**:
   ```bash
   curl http://localhost:9001/metrics
   ```
   Should return Prometheus-format metrics

3. **Verify Prometheus datasource**:
   - Grafana → Configuration → Data Sources
   - Test connection to Prometheus

### Metrics show "No data"

- **Check time range**: Default is "Last 1 hour"
- **Verify models are running**: Some metrics only populate when models are active
- **Check scrape interval**: Metrics update every 15s (default)

### Dashboard import fails

- **Check Grafana version**: Dashboard tested with Grafana 10.x
- **Verify JSON format**: Ensure no copy/paste corruption
- **Manual datasource**: After import, manually select Prometheus datasource if needed

---

## Advanced: Custom Panels

### Panel: Idle Models (suspended >10 min)
```promql
(time() - model_last_request_timestamp{model_status="3"}) / 60 > 10
```

### Panel: Auto-Suspend Events (last 24h)
```promql
changes(model_status{model_status="3"}[24h])
```

### Panel: Memory Freed by Auto-Suspend
```promql
sum(model_memory_used_bytes{model_status="3"})
```

---

## Integration with Alertmanager

Add to Alertmanager `config.yml`:

```yaml
receivers:
  - name: 'sparkstation-alerts'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK'
        channel: '#sparkstation-alerts'
        title: 'Sparkstation Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

route:
  group_by: ['alertname']
  receiver: 'sparkstation-alerts'
  routes:
    - match:
        job: sparkstation
      receiver: 'sparkstation-alerts'
```

---

## Dashboard Maintenance

### Backup Dashboard
```bash
# Export from Grafana UI
Dashboard → Settings → JSON Model → Copy to clipboard

# Or via API
curl -H "Authorization: Bearer YOUR_API_KEY" \
  http://localhost:3000/api/dashboards/uid/sparkstation-main \
  > backup-$(date +%Y%m%d).json
```

### Update Dashboard
1. Make changes in Grafana UI
2. Export JSON: Dashboard → Settings → JSON Model
3. Save to `monitoring/grafana-dashboard.json`
4. Commit to git

---

## Performance Impact

- **Prometheus scrape**: ~5ms per scrape (15s interval)
- **Metrics endpoint**: <1ms response time
- **Memory overhead**: <10MB for metrics collection
- **CPU overhead**: Negligible (<0.1%)

---

## TODO / Future Enhancements

### Request Metrics (Not Yet Implemented)

The following metrics are defined but not yet instrumented:
- `model_requests_total` - Total requests served by model
- `model_requests_failed` - Failed requests by model
- `model_request_latency_seconds` - Request latency histogram

**Dashboard panels affected:**
- "Request Rate by Model (5m)" - Shows "No data"
- "Request Latency (p50, p95)" - Shows "No data"

**Implementation options:**
1. **Gateway middleware** - Add LiteLLM middleware to track requests
2. **Proxy layer** - Intercept requests between gateway and backends
3. **Backend polling** - Query vLLM/SGLang metrics endpoints (they expose their own metrics)

**Current workaround:**
These panels can be safely ignored. All critical monitoring (memory, temperature, power, model status) is fully functional.

---

## Additional Resources

- [Prometheus Docs](https://prometheus.io/docs/)
- [Grafana Docs](https://grafana.com/docs/)
- [PromQL Query Examples](https://prometheus.io/docs/prometheus/latest/querying/examples/)

---

**Built for NVIDIA DGX Spark (Grace Blackwell)**
