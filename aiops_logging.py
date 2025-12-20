"""
AIOps Logging and Monitoring Infrastructure

This module provides comprehensive logging, monitoring, and audit capabilities for AI_Ops_Lite.

Features:
- Structured JSON logging
- Business event tracking (ML predictions, user actions)
- Performance monitoring
- ROI metrics calculation
- Audit trail
- Error tracking and alerting
"""

import logging
import json
import time
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
import pandas as pd
from typing import Dict, Any, Optional, List
from collections import defaultdict
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

class LogConfig:
    """Centralized logging configuration"""
    LOG_DIR = Path("logs")
    BUSINESS_LOG = LOG_DIR / "business_events.jsonl"
    PERFORMANCE_LOG = LOG_DIR / "performance.jsonl"
    AUDIT_LOG = LOG_DIR / "audit_trail.jsonl"
    ERROR_LOG = LOG_DIR / "errors.jsonl"
    METRICS_LOG = LOG_DIR / "roi_metrics.jsonl"

    # Ensure log directory exists
    LOG_DIR.mkdir(exist_ok=True)

    # Log levels
    LOG_LEVEL = os.getenv('AIOPS_LOG_LEVEL', 'INFO')

    # Performance thresholds (milliseconds)
    SLOW_QUERY_THRESHOLD = 1000  # 1 second
    CRITICAL_THRESHOLD = 5000     # 5 seconds


# ============================================================================
# STRUCTURED LOGGER
# ============================================================================

class StructuredLogger:
    """Base logger with JSON structured output"""

    def __init__(self, name: str, log_file: Path):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, LogConfig.LOG_LEVEL))

        # File handler with JSON formatting
        handler = logging.FileHandler(log_file, mode='a')
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)

        # Also log to console in development
        console = logging.StreamHandler()
        console.setLevel(logging.WARNING)
        self.logger.addHandler(console)

    def _log_event(self, event_type: str, data: Dict[str, Any], level: str = 'INFO'):
        """Log structured event as JSON"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            **data
        }

        log_method = getattr(self.logger, level.lower())
        log_method(json.dumps(log_entry))

        return log_entry


# ============================================================================
# BUSINESS EVENT LOGGING
# ============================================================================

class BusinessEventLogger(StructuredLogger):
    """Logs business-critical events like ML predictions and user actions"""

    def __init__(self):
        super().__init__('aiops.business', LogConfig.BUSINESS_LOG)

    def log_ml_prediction(
        self,
        user: str,
        feature: str,
        input_data: Dict[str, Any],
        prediction: Dict[str, Any],
        outcome: str,
        confidence: Optional[float] = None
    ):
        """
        Log ML model prediction and user response

        Args:
            user: User who triggered prediction
            feature: Feature name (similar_incidents, assignment_routing, etc.)
            input_data: Input to the model
            prediction: Model output
            outcome: User action (accepted, rejected, modified)
            confidence: Prediction confidence score
        """
        return self._log_event('ml_prediction', {
            'user': user,
            'feature': feature,
            'input': input_data,
            'prediction': prediction,
            'confidence': confidence,
            'user_action': outcome,
            'business_impact': self._calculate_prediction_impact(feature, outcome)
        })

    def log_pattern_detection(
        self,
        pattern_type: str,
        count: int,
        affected_incidents: List[str],
        metadata: Dict[str, Any]
    ):
        """Log pattern detection events"""
        return self._log_event('pattern_detected', {
            'pattern_type': pattern_type,
            'incident_count': count,
            'affected_incidents': affected_incidents[:10],  # Limit to first 10
            'metadata': metadata
        })

    def log_deflection_opportunity(
        self,
        incident_count: int,
        estimated_savings: float,
        category: str
    ):
        """Log identified deflection opportunities"""
        return self._log_event('deflection_identified', {
            'incident_count': incident_count,
            'estimated_savings_usd': estimated_savings,
            'category': category
        })

    def log_user_action(
        self,
        user: str,
        action: str,
        context: Dict[str, Any],
        result: str
    ):
        """Log user interactions with the system"""
        return self._log_event('user_action', {
            'user': user,
            'action': action,
            'context': context,
            'result': result
        })

    def _calculate_prediction_impact(self, feature: str, outcome: str) -> Dict[str, Any]:
        """Calculate business impact of ML prediction"""
        impact_map = {
            'similar_incidents': {'accepted': 2.0, 'rejected': 0, 'modified': 1.0},  # hours saved
            'assignment_routing': {'accepted': 1.5, 'rejected': 0, 'modified': 0.5},
            'problem_creation': {'accepted': 50.0, 'rejected': 0, 'modified': 25.0}  # potential savings
        }

        time_saved = impact_map.get(feature, {}).get(outcome, 0)
        cost_saved = time_saved * 50  # $50/hour labor cost

        return {
            'time_saved_hours': time_saved,
            'cost_saved_usd': cost_saved
        }


# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

class PerformanceMonitor(StructuredLogger):
    """Monitors system and function performance"""

    def __init__(self):
        super().__init__('aiops.performance', LogConfig.PERFORMANCE_LOG)
        self.metrics = defaultdict(list)

    def monitor(self, func_name: Optional[str] = None):
        """
        Decorator to monitor function performance

        Usage:
            @performance_monitor.monitor()
            def my_function():
                pass
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                name = func_name or func.__name__
                start_time = time.time()
                error = None
                result = None

                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    error = str(e)
                    raise
                finally:
                    duration_ms = (time.time() - start_time) * 1000

                    # Log performance event
                    log_data = {
                        'function': name,
                        'duration_ms': round(duration_ms, 2),
                        'status': 'error' if error else 'success',
                        'error': error
                    }

                    level = 'ERROR' if error else 'INFO'

                    # Warn on slow queries
                    if duration_ms > LogConfig.SLOW_QUERY_THRESHOLD:
                        level = 'WARNING'
                        log_data['performance_issue'] = 'slow_query'

                    if duration_ms > LogConfig.CRITICAL_THRESHOLD:
                        level = 'ERROR'
                        log_data['performance_issue'] = 'critical_slow'

                    self._log_event('performance', log_data, level)

                    # Store for analytics
                    self.metrics[name].append(duration_ms)

            return wrapper
        return decorator

    def get_stats(self, func_name: str) -> Dict[str, float]:
        """Get performance statistics for a function"""
        if func_name not in self.metrics or not self.metrics[func_name]:
            return {}

        timings = self.metrics[func_name]
        return {
            'count': len(timings),
            'avg_ms': sum(timings) / len(timings),
            'min_ms': min(timings),
            'max_ms': max(timings),
            'p50_ms': sorted(timings)[len(timings) // 2],
            'p95_ms': sorted(timings)[int(len(timings) * 0.95)] if len(timings) > 20 else max(timings)
        }


# ============================================================================
# AUDIT TRAIL
# ============================================================================

class AuditLogger(StructuredLogger):
    """Maintains compliance and audit trail"""

    def __init__(self):
        super().__init__('aiops.audit', LogConfig.AUDIT_LOG)

    def log_data_access(
        self,
        user: str,
        data_type: str,
        record_count: int,
        purpose: str,
        filters: Optional[Dict[str, Any]] = None
    ):
        """Log data access for compliance"""
        return self._log_event('data_access', {
            'user': user,
            'data_type': data_type,
            'record_count': record_count,
            'purpose': purpose,
            'filters': filters or {}
        })

    def log_configuration_change(
        self,
        user: str,
        setting: str,
        old_value: Any,
        new_value: Any,
        reason: str
    ):
        """Log system configuration changes"""
        return self._log_event('config_change', {
            'user': user,
            'setting': setting,
            'old_value': str(old_value),
            'new_value': str(new_value),
            'reason': reason
        })

    def log_model_training(
        self,
        model_type: str,
        training_samples: int,
        accuracy: float,
        hyperparameters: Dict[str, Any]
    ):
        """Log ML model training events"""
        return self._log_event('model_training', {
            'model_type': model_type,
            'training_samples': training_samples,
            'accuracy': accuracy,
            'hyperparameters': hyperparameters
        })


# ============================================================================
# ROI METRICS TRACKER
# ============================================================================

class ROIMetricsTracker(StructuredLogger):
    """Tracks and calculates ROI metrics"""

    def __init__(self):
        super().__init__('aiops.metrics', LogConfig.METRICS_LOG)
        self.daily_metrics = defaultdict(lambda: {
            'incidents_analyzed': 0,
            'patterns_detected': 0,
            'deflectable_tickets': 0,
            'ml_predictions': 0,
            'ml_accepted': 0,
            'time_saved_hours': 0,
            'cost_saved_usd': 0
        })

    def record_incident_analysis(self, count: int):
        """Record incidents analyzed"""
        today = datetime.now().date().isoformat()
        self.daily_metrics[today]['incidents_analyzed'] += count
        self._persist_metrics(today)

    def record_pattern_detection(self, count: int):
        """Record patterns detected"""
        today = datetime.now().date().isoformat()
        self.daily_metrics[today]['patterns_detected'] += count
        self._persist_metrics(today)

    def record_deflection(self, count: int, savings: float):
        """Record deflection opportunity"""
        today = datetime.now().date().isoformat()
        self.daily_metrics[today]['deflectable_tickets'] += count
        self.daily_metrics[today]['cost_saved_usd'] += savings
        self._persist_metrics(today)

    def record_ml_prediction(self, accepted: bool, time_saved: float = 0, cost_saved: float = 0):
        """Record ML prediction outcome"""
        today = datetime.now().date().isoformat()
        self.daily_metrics[today]['ml_predictions'] += 1
        if accepted:
            self.daily_metrics[today]['ml_accepted'] += 1
            self.daily_metrics[today]['time_saved_hours'] += time_saved
            self.daily_metrics[today]['cost_saved_usd'] += cost_saved
        self._persist_metrics(today)

    def _persist_metrics(self, date: str):
        """Persist metrics to log"""
        metrics = self.daily_metrics[date].copy()
        metrics['date'] = date

        # Calculate derived metrics
        if metrics['ml_predictions'] > 0:
            metrics['ml_acceptance_rate'] = metrics['ml_accepted'] / metrics['ml_predictions']

        self._log_event('daily_metrics', metrics)

    def get_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get summary metrics for last N days"""
        start_date = (datetime.now() - timedelta(days=days)).date().isoformat()

        total_metrics = {
            'incidents_analyzed': 0,
            'patterns_detected': 0,
            'deflectable_tickets': 0,
            'ml_predictions': 0,
            'ml_accepted': 0,
            'time_saved_hours': 0,
            'cost_saved_usd': 0
        }

        for date, metrics in self.daily_metrics.items():
            if date >= start_date:
                for key in total_metrics:
                    total_metrics[key] += metrics.get(key, 0)

        # Calculate rates
        if total_metrics['ml_predictions'] > 0:
            total_metrics['ml_acceptance_rate'] = total_metrics['ml_accepted'] / total_metrics['ml_predictions']

        if total_metrics['incidents_analyzed'] > 0:
            total_metrics['pattern_detection_rate'] = total_metrics['patterns_detected'] / total_metrics['incidents_analyzed']
            total_metrics['deflection_rate'] = total_metrics['deflectable_tickets'] / total_metrics['incidents_analyzed']

        total_metrics['period_days'] = days
        total_metrics['start_date'] = start_date
        total_metrics['end_date'] = datetime.now().date().isoformat()

        return total_metrics


# ============================================================================
# ERROR TRACKING
# ============================================================================

class ErrorTracker(StructuredLogger):
    """Tracks errors and exceptions"""

    def __init__(self):
        super().__init__('aiops.errors', LogConfig.ERROR_LOG)
        self.error_counts = defaultdict(int)

    def log_error(
        self,
        error_type: str,
        message: str,
        context: Dict[str, Any],
        stack_trace: Optional[str] = None,
        severity: str = 'ERROR'
    ):
        """Log application error"""
        error_key = f"{error_type}:{message[:50]}"
        self.error_counts[error_key] += 1

        return self._log_event('error', {
            'error_type': error_type,
            'message': message,
            'context': context,
            'stack_trace': stack_trace,
            'severity': severity,
            'occurrence_count': self.error_counts[error_key]
        }, level=severity)

    def should_alert(self, error_key: str, threshold: int = 5) -> bool:
        """Check if error rate warrants alerting"""
        return self.error_counts[error_key] >= threshold


# ============================================================================
# SINGLETON INSTANCES
# ============================================================================

# Global logger instances
business_logger = BusinessEventLogger()
performance_monitor = PerformanceMonitor()
audit_logger = AuditLogger()
roi_tracker = ROIMetricsTracker()
error_tracker = ErrorTracker()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def log_ml_prediction(user: str, feature: str, input_data: dict, prediction: dict, outcome: str, confidence: float = None):
    """Convenience function for ML prediction logging"""
    return business_logger.log_ml_prediction(user, feature, input_data, prediction, outcome, confidence)

def track_performance(func_name: str = None):
    """Convenience decorator for performance monitoring"""
    return performance_monitor.monitor(func_name)

def log_audit(user: str, action: str, **kwargs):
    """Convenience function for audit logging"""
    if action == 'data_access':
        return audit_logger.log_data_access(user, **kwargs)
    elif action == 'config_change':
        return audit_logger.log_configuration_change(user, **kwargs)
    elif action == 'model_training':
        return audit_logger.log_model_training(**kwargs)

def track_roi(metric_type: str, **kwargs):
    """Convenience function for ROI tracking"""
    if metric_type == 'incidents':
        roi_tracker.record_incident_analysis(kwargs['count'])
    elif metric_type == 'patterns':
        roi_tracker.record_pattern_detection(kwargs['count'])
    elif metric_type == 'deflection':
        roi_tracker.record_deflection(kwargs['count'], kwargs['savings'])
    elif metric_type == 'ml_prediction':
        roi_tracker.record_ml_prediction(kwargs['accepted'], kwargs.get('time_saved', 0), kwargs.get('cost_saved', 0))

def log_error(error_type: str, message: str, context: dict = None, stack_trace: str = None):
    """Convenience function for error logging"""
    return error_tracker.log_error(error_type, message, context or {}, stack_trace)
