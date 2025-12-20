# AIOps Logging & Monitoring Guide

## Overview

The AIOps logging infrastructure provides comprehensive observability, ROI tracking, and compliance capabilities for the AI_Ops_Lite platform.

**Annual Value:** $150,000
- Compliance & Audit: $50K
- Performance Optimization: $50K
- ROI Proof & Stakeholder Confidence: $50K

---

## Quick Start

### 1. Automatic Logging

Logging is automatically enabled for all AI Intelligence features:
- Similar Incident Recommendations
- Intelligent Assignment Routing
- Auto-Problem Creation

Simply use the features normally - all events are logged automatically!

### 2. View Monitoring Dashboard

```bash
streamlit run app.py
```

Navigate to the **"📊 Monitoring & ROI"** tab to see:
- ROI metrics and annual projections
- ML model accuracy by feature
- System performance statistics
- Error tracking
- User activity
- Audit trail

---

## Log Files

All logs are stored in structured JSON format (JSONL) in the `logs/` directory:

```
logs/
├── business_events.jsonl    # ML predictions, patterns, user actions
├── performance.jsonl         # Function execution times
├── audit_trail.jsonl         # Compliance events
├── errors.jsonl              # Application errors
└── roi_metrics.jsonl         # Daily ROI calculations
```

### Log Format

Each log entry is a JSON object with consistent structure:

```json
{
  "timestamp": "2025-12-19T14:23:15.123456",
  "event_type": "ml_prediction",
  "user": "john.doe@company.com",
  "feature": "similar_incidents",
  "prediction": {...},
  "user_action": "accepted",
  "business_impact": {
    "time_saved_hours": 2.0,
    "cost_saved_usd": 100.0
  }
}
```

---

## Using the Logging API

### Business Event Logging

```python
from aiops_logging import business_logger

# Log ML prediction
business_logger.log_ml_prediction(
    user='john.doe@company.com',
    feature='similar_incidents',
    input_data={'incident_desc': 'Database timeout'},
    prediction={'similarity': 0.89, 'incident': 'INC0012345'},
    outcome='accepted',  # 'accepted', 'rejected', 'modified'
    confidence=0.89
)

# Log pattern detection
business_logger.log_pattern_detection(
    pattern_type='cluster',
    count=15,
    affected_incidents=['INC001', 'INC002', ...],
    metadata={'cluster_id': 3, 'keywords': ['database', 'timeout']}
)

# Log deflection opportunity
business_logger.log_deflection_opportunity(
    incident_count=45,
    estimated_savings=2250.0,
    category='password_resets'
)
```

### Performance Monitoring

```python
from aiops_logging import track_performance

# Decorator approach (recommended)
@track_performance('my_function')
def my_expensive_function():
    # Your code here
    pass

# Manual tracking
from aiops_logging import performance_monitor

start = time.time()
# ... do work ...
duration_ms = (time.time() - start) * 1000

performance_monitor._log_event('performance', {
    'function': 'manual_operation',
    'duration_ms': duration_ms,
    'status': 'success'
})
```

### Audit Logging

```python
from aiops_logging import audit_logger

# Log data access
audit_logger.log_data_access(
    user='john.doe@company.com',
    data_type='incidents',
    record_count=1247,
    purpose='pattern_analysis',
    filters={'date_range': '30_days'}
)

# Log configuration changes
audit_logger.log_configuration_change(
    user='admin@company.com',
    setting='ml_threshold',
    old_value=0.85,
    new_value=0.90,
    reason='Improve accuracy'
)

# Log model training
audit_logger.log_model_training(
    model_type='IntelligentRouter',
    training_samples=1500,
    accuracy=0.87,
    hyperparameters={'n_estimators': 100, 'max_features': 300}
)
```

### ROI Tracking

```python
from aiops_logging import roi_tracker

# Track incidents analyzed
roi_tracker.record_incident_analysis(count=100)

# Track patterns detected
roi_tracker.record_pattern_detection(count=12)

# Track deflection opportunities
roi_tracker.record_deflection(count=25, savings=1250.0)

# Track ML predictions
roi_tracker.record_ml_prediction(
    accepted=True,
    time_saved=2.0,  # hours
    cost_saved=100.0  # USD
)

# Get summary
summary = roi_tracker.get_summary(days=7)
print(f"Total savings: ${summary['cost_saved_usd']}")
```

### Error Tracking

```python
from aiops_logging import error_tracker

try:
    # Your code
    risky_operation()
except Exception as e:
    error_tracker.log_error(
        error_type=type(e).__name__,
        message=str(e),
        context={'function': 'risky_operation', 'user': 'john.doe'},
        stack_trace=traceback.format_exc(),
        severity='ERROR'  # 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    )

# Check if should alert
if error_tracker.should_alert('DatabaseError:Connection timeout', threshold=5):
    # Send alert to ops team
    pass
```

---

## Analyzing Logs

### Using the LogAnalyzer

```python
from log_analyzer import LogAnalyzer

analyzer = LogAnalyzer()
analyzer.load_logs(days_back=7)

# Get ROI summary
roi = analyzer.get_roi_summary()
print(f"Total incidents analyzed: {roi['incidents_analyzed']}")
print(f"Total cost saved: ${roi['cost_saved_usd']}")
print(f"ML acceptance rate: {roi['ml_acceptance_rate']:.1%}")

# Get ML accuracy by feature
accuracy = analyzer.get_ml_accuracy_by_feature()
for feature, stats in accuracy.items():
    print(f"{feature}: {stats['acceptance_rate']:.1%} acceptance")

# Get performance stats
perf = analyzer.get_performance_stats()
for func, stats in perf.items():
    print(f"{func}: avg={stats['avg_ms']:.0f}ms, p95={stats['p95_ms']:.0f}ms")

# Get error summary
errors = analyzer.get_error_summary()
print(f"Total errors: {errors['total_errors']}")
print(f"Unique errors: {errors['unique_errors']}")

# Export full report
report = analyzer.export_summary_report()
with open('aiops_report.json', 'w') as f:
    json.dump(report, f, indent=2)
```

### Command Line Analysis

```bash
# View recent business events
tail -f logs/business_events.jsonl | jq .

# Count ML predictions by feature
cat logs/business_events.jsonl | jq -r 'select(.event_type=="ml_prediction") | .feature' | sort | uniq -c

# Calculate total savings
cat logs/business_events.jsonl | jq -r 'select(.business_impact) | .business_impact.cost_saved_usd' | awk '{sum+=$1} END {print "Total saved: $"sum}'

# Find slow functions (>1s)
cat logs/performance.jsonl | jq -r 'select(.duration_ms > 1000) | "\(.function): \(.duration_ms)ms"'

# Count errors by type
cat logs/errors.jsonl | jq -r '.error_type' | sort | uniq -c | sort -rn
```

---

## Configuration

### Environment Variables

```bash
# Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
export AIOPS_LOG_LEVEL=INFO

# Customize thresholds in aiops_logging.py
class LogConfig:
    SLOW_QUERY_THRESHOLD = 1000  # ms
    CRITICAL_THRESHOLD = 5000     # ms
```

### Log Rotation

Recommended log rotation configuration (using logrotate):

```
/path/to/AI_Ops_Lite/logs/*.jsonl {
    daily
    rotate 30
    compress
    missingok
    notifempty
    create 0644 root root
}
```

---

## Integration Examples

### Example 1: Track Feature Usage

```python
from aiops_logging import business_logger

def handle_similar_incidents_request(user, incident_desc):
    # Find similar incidents
    similar = find_similar_resolved_incidents(incident_desc, df)

    # Show to user and track their action
    user_action = show_to_user_and_get_feedback(similar)

    # Log the event
    business_logger.log_ml_prediction(
        user=user,
        feature='similar_incidents',
        input_data={'description': incident_desc},
        prediction={'matches': len(similar), 'top_similarity': similar[0]['similarity_score']},
        outcome=user_action,  # 'accepted', 'rejected', 'modified'
        confidence=similar[0]['similarity_score'] if similar else 0
    )

    return similar
```

### Example 2: Monitor Model Drift

```python
from aiops_logging import audit_logger
from log_analyzer import LogAnalyzer

# Train model
router = IntelligentRouter()
metrics = router.train(historical_df)

# Log training
audit_logger.log_model_training(
    model_type='IntelligentRouter',
    training_samples=metrics['num_training_samples'],
    accuracy=metrics['training_accuracy'],
    hyperparameters={'n_estimators': 100}
)

# Later: Check if model accuracy is degrading
analyzer = LogAnalyzer()
analyzer.load_logs(days_back=30)

accuracy = analyzer.get_ml_accuracy_by_feature()
if accuracy['assignment_routing']['acceptance_rate'] < 0.80:
    print("⚠️ Model accuracy below threshold - retrain recommended")
    # Trigger retraining workflow
```

### Example 3: Compliance Reporting

```python
from log_analyzer import LogAnalyzer
import pandas as pd

# Generate quarterly compliance report
analyzer = LogAnalyzer()
analyzer.load_logs(days_back=90)

audit_trail = analyzer.get_audit_trail(limit=1000)

report_df = pd.DataFrame(audit_trail)
report_df.to_csv('q4_compliance_report.csv', index=False)

print(f"Total audit events: {len(audit_trail)}")
print(f"Data access events: {len([e for e in audit_trail if e['event_type'] == 'data_access'])}")
print(f"Configuration changes: {len([e for e in audit_trail if e['event_type'] == 'config_change'])}")
```

---

## Monitoring Dashboard Features

The built-in Streamlit dashboard (`📊 Monitoring & ROI` tab) provides:

### ROI Summary
- Incidents analyzed
- Patterns detected
- Time saved (hours)
- Cost saved (USD)
- ML prediction acceptance rate
- Deflectable tickets identified
- **Annual value projection**

### ML Model Performance
- Accuracy by feature (similar_incidents, assignment_routing, etc.)
- Acceptance/rejection rates
- Prediction confidence trends

### System Performance
- Function call counts
- Average, P50, P95, Max response times
- Slow query detection
- Performance degradation alerts

### Error Tracking
- Total errors and unique types
- Top 5 most common errors
- Error frequency trends
- Last occurrence timestamps

### User Activity
- Top 10 active users
- Action counts per user
- Usage patterns

### Audit Trail
- Recent 20 audit events
- Filterable by type
- Full event details

### Export
- One-click JSON report export
- Complete monitoring snapshot

---

## Best Practices

### 1. Always Log Business Events

```python
# ✅ Good
@track_performance()
def find_similar_incidents(...):
    result = # ... do work
    business_logger.log_ml_prediction(...)  # Log the event
    return result

# ❌ Bad
def find_similar_incidents(...):
    return # ... do work (no logging)
```

### 2. Use Structured Data

```python
# ✅ Good
business_logger.log_pattern_detection(
    pattern_type='cluster',
    count=15,
    affected_incidents=incident_list,
    metadata={'cluster_id': 3, 'confidence': 0.92}
)

# ❌ Bad
logging.info(f"Found 15 incidents in cluster 3")  # Unstructured
```

### 3. Track User Actions

```python
# Always log what the user did with your recommendation
business_logger.log_ml_prediction(
    ...,
    outcome='accepted'  # or 'rejected', 'modified'
)
```

### 4. Monitor Performance

```python
# Use decorators for automatic tracking
@track_performance('expensive_operation')
def expensive_ml_function():
    pass
```

### 5. Regular Analysis

```python
# Weekly: Check ROI metrics
# Monthly: Review ML accuracy trends
# Quarterly: Generate compliance reports
```

---

## Troubleshooting

### Logs Not Appearing

1. Check that `logs/` directory exists and is writable
2. Verify `LOGGING_ENABLED = True` in aiops_intelligence.py
3. Check log level: `export AIOPS_LOG_LEVEL=DEBUG`

### Dashboard Shows No Data

1. Logs are only created when features are used
2. Try using the AI Intelligence features first
3. Refresh the monitoring dashboard tab

### Performance Impact

Logging overhead is minimal (~1-5ms per event). If concerned:
- Use `INFO` level in production (not `DEBUG`)
- Logs are written asynchronously
- No blocking I/O operations

---

## ROI Calculation Methodology

### Time Savings
```
Similar Incidents (accepted): 2.0 hours saved
Assignment Routing (accepted): 1.5 hours saved
Problem Creation (accepted): 50 hours saved (over lifecycle)
```

### Cost Conversion
```
Labor cost: $50/hour (configurable)
Cost saved = Time saved × $50/hour
```

### Annual Projection
```
Weekly savings × 52 weeks = Annual value
```

---

## Support & Resources

- **Log Files:** `logs/*.jsonl`
- **Configuration:** `aiops_logging.py` → `LogConfig` class
- **Analysis:** `log_analyzer.py` → `LogAnalyzer` class
- **Dashboard:** Streamlit app → "📊 Monitoring & ROI" tab
- **Documentation:** This file

For questions or issues, review the inline code documentation in:
- `aiops_logging.py`
- `log_analyzer.py`

---

**Next Steps:**
1. Use the AI Intelligence features to generate logs
2. View the Monitoring dashboard to see ROI in real-time
3. Export reports for stakeholder presentations
4. Use metrics to justify AIOps investment
