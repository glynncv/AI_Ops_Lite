"""
Log Analysis Utilities

Reads and analyzes structured logs to generate insights and dashboards.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any
from collections import defaultdict

from aiops_logging import LogConfig


class LogAnalyzer:
    """Analyzes structured JSON logs"""

    def __init__(self):
        self.business_events = []
        self.performance_events = []
        self.audit_events = []
        self.error_events = []
        self.metrics_events = []

    def load_logs(self, days_back: int = 7):
        """Load logs from the last N days"""
        cutoff_date = datetime.now() - timedelta(days=days_back)

        self.business_events = self._load_log_file(LogConfig.BUSINESS_LOG, cutoff_date)
        self.performance_events = self._load_log_file(LogConfig.PERFORMANCE_LOG, cutoff_date)
        self.audit_events = self._load_log_file(LogConfig.AUDIT_LOG, cutoff_date)
        self.error_events = self._load_log_file(LogConfig.ERROR_LOG, cutoff_date)
        self.metrics_events = self._load_log_file(LogConfig.METRICS_LOG, cutoff_date)

    def _load_log_file(self, log_file: Path, cutoff_date: datetime) -> List[Dict]:
        """Load and filter a JSONL log file"""
        events = []

        if not log_file.exists():
            return events

        with open(log_file, 'r') as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                    event_time = datetime.fromisoformat(event['timestamp'])

                    if event_time >= cutoff_date:
                        events.append(event)
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

        return events

    def get_roi_summary(self) -> Dict[str, Any]:
        """Calculate ROI summary from metrics"""
        if not self.metrics_events:
            return self._empty_roi_summary()

        # Get latest metrics for each day
        daily_metrics = {}
        for event in self.metrics_events:
            if event['event_type'] == 'daily_metrics':
                date = event['date']
                daily_metrics[date] = event

        # Aggregate totals
        total = {
            'incidents_analyzed': 0,
            'patterns_detected': 0,
            'deflectable_tickets': 0,
            'ml_predictions': 0,
            'ml_accepted': 0,
            'time_saved_hours': 0,
            'cost_saved_usd': 0
        }

        for metrics in daily_metrics.values():
            for key in total:
                total[key] += metrics.get(key, 0)

        # Calculate rates
        if total['ml_predictions'] > 0:
            total['ml_acceptance_rate'] = round(total['ml_accepted'] / total['ml_predictions'], 2)
        else:
            total['ml_acceptance_rate'] = 0

        if total['incidents_analyzed'] > 0:
            total['pattern_detection_rate'] = round(total['patterns_detected'] / total['incidents_analyzed'], 2)
            total['deflection_rate'] = round(total['deflectable_tickets'] / total['incidents_analyzed'], 2)
        else:
            total['pattern_detection_rate'] = 0
            total['deflection_rate'] = 0

        return total

    def _empty_roi_summary(self) -> Dict[str, Any]:
        """Return empty ROI summary structure"""
        return {
            'incidents_analyzed': 0,
            'patterns_detected': 0,
            'deflectable_tickets': 0,
            'ml_predictions': 0,
            'ml_accepted': 0,
            'ml_acceptance_rate': 0,
            'time_saved_hours': 0,
            'cost_saved_usd': 0,
            'pattern_detection_rate': 0,
            'deflection_rate': 0
        }

    def get_ml_accuracy_by_feature(self) -> Dict[str, Dict[str, Any]]:
        """Calculate ML prediction accuracy by feature"""
        feature_stats = defaultdict(lambda: {'total': 0, 'accepted': 0, 'rejected': 0, 'modified': 0})

        for event in self.business_events:
            if event['event_type'] == 'ml_prediction':
                feature = event['feature']
                outcome = event['user_action']

                feature_stats[feature]['total'] += 1
                feature_stats[feature][outcome] += 1

        # Calculate acceptance rates
        results = {}
        for feature, stats in feature_stats.items():
            if stats['total'] > 0:
                results[feature] = {
                    **stats,
                    'acceptance_rate': round(stats['accepted'] / stats['total'], 2)
                }

        return results

    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics by function"""
        function_timings = defaultdict(list)

        for event in self.performance_events:
            if event['event_type'] == 'performance' and event['status'] == 'success':
                function = event['function']
                duration = event['duration_ms']
                function_timings[function].append(duration)

        stats = {}
        for func, timings in function_timings.items():
            if timings:
                sorted_timings = sorted(timings)
                stats[func] = {
                    'count': len(timings),
                    'avg_ms': round(sum(timings) / len(timings), 2),
                    'min_ms': round(min(timings), 2),
                    'max_ms': round(max(timings), 2),
                    'p50_ms': round(sorted_timings[len(sorted_timings) // 2], 2),
                    'p95_ms': round(sorted_timings[int(len(sorted_timings) * 0.95)], 2) if len(sorted_timings) > 20 else round(max(timings), 2)
                }

        return stats

    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary and top errors"""
        error_counts = defaultdict(int)
        error_details = defaultdict(lambda: {'count': 0, 'last_seen': None, 'severity': 'ERROR'})

        for event in self.error_events:
            if event['event_type'] == 'error':
                error_type = event['error_type']
                message = event['message'][:100]  # Truncate long messages
                key = f"{error_type}: {message}"

                error_counts[key] += 1
                error_details[key]['count'] += 1
                error_details[key]['last_seen'] = event['timestamp']
                error_details[key]['severity'] = event['severity']

        # Sort by count
        top_errors = sorted(error_details.items(), key=lambda x: x[1]['count'], reverse=True)[:10]

        return {
            'total_errors': len(self.error_events),
            'unique_errors': len(error_details),
            'top_errors': [{'error': k, **v} for k, v in top_errors]
        }

    def get_user_activity(self) -> Dict[str, int]:
        """Get user activity counts"""
        user_counts = defaultdict(int)

        for event in self.business_events:
            if 'user' in event:
                user_counts[event['user']] += 1

        for event in self.audit_events:
            if 'user' in event:
                user_counts[event['user']] += 1

        return dict(sorted(user_counts.items(), key=lambda x: x[1], reverse=True))

    def get_timeline_data(self, metric: str = 'all') -> pd.DataFrame:
        """Get time-series data for visualization"""
        timeline = defaultdict(lambda: {
            'ml_predictions': 0,
            'patterns_detected': 0,
            'errors': 0,
            'incidents_analyzed': 0
        })

        # Process all events
        all_events = (
            self.business_events +
            self.error_events +
            self.metrics_events
        )

        for event in all_events:
            timestamp = datetime.fromisoformat(event['timestamp'])
            hour_key = timestamp.replace(minute=0, second=0, microsecond=0)

            if event['event_type'] == 'ml_prediction':
                timeline[hour_key]['ml_predictions'] += 1
            elif event['event_type'] == 'pattern_detected':
                timeline[hour_key]['patterns_detected'] += 1
            elif event['event_type'] == 'error':
                timeline[hour_key]['errors'] += 1
            elif event['event_type'] == 'daily_metrics':
                # These are daily, not hourly
                pass

        # Convert to DataFrame
        df = pd.DataFrame([
            {'timestamp': k, **v}
            for k, v in sorted(timeline.items())
        ])

        return df

    def get_audit_trail(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent audit trail entries"""
        sorted_events = sorted(self.audit_events, key=lambda x: x['timestamp'], reverse=True)
        return sorted_events[:limit]

    def export_summary_report(self) -> Dict[str, Any]:
        """Export comprehensive summary report"""
        return {
            'report_generated': datetime.now().isoformat(),
            'period_days': 7,
            'roi_summary': self.get_roi_summary(),
            'ml_accuracy': self.get_ml_accuracy_by_feature(),
            'performance': self.get_performance_stats(),
            'errors': self.get_error_summary(),
            'user_activity': self.get_user_activity(),
            'total_events': {
                'business': len(self.business_events),
                'performance': len(self.performance_events),
                'audit': len(self.audit_events),
                'errors': len(self.error_events),
                'metrics': len(self.metrics_events)
            }
        }
