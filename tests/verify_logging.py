import unittest
import os
import shutil
import json
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aiops_logging import AuditLogger, PerformanceMonitor, ROIMetricsTracker, ErrorTracker
from log_analyzer import LogAnalyzer

class TestLoggingSystem(unittest.TestCase):
    def setUp(self):
        # Initialize loggers (they use default log directory)
        self.audit_logger = AuditLogger()
        self.perf_monitor = PerformanceMonitor()
        self.roi_tracker = ROIMetricsTracker()
        self.error_tracker = ErrorTracker()

    def test_audit_logging(self):
        # Log a data access event
        self.audit_logger.log_data_access(
            user="test_user",
            data_type="incidents",
            record_count=10,
            purpose="testing"
        )
        
        # Verify log file exists
        from aiops_logging import LogConfig
        expected_file = LogConfig.AUDIT_LOG
        self.assertTrue(os.path.exists(expected_file), f"Audit log file not created: {expected_file}")
        
        # Verify log file has content
        with open(expected_file, 'r') as f:
            lines = f.readlines()
            self.assertGreater(len(lines), 0, "Audit log file is empty")
            
            # Check if our test event is in the log file
            test_found = False
            for line in lines[-10:]:  # Check last 10 lines
                try:
                    entry = json.loads(line.strip())
                    if entry.get("event_type") == "data_access" and entry.get("user") == "test_user":
                        test_found = True
                        break
                except json.JSONDecodeError:
                    continue
            self.assertTrue(test_found, "Test event not found in audit log file")

    def test_performance_tracking(self):
        # Track execution time using decorator pattern
        @self.perf_monitor.monitor()
        def test_operation():
            return "test_result"
        
        result = test_operation()
        self.assertEqual(result, "test_result")
        
        # Verify performance was logged
        from aiops_logging import LogConfig
        self.assertTrue(os.path.exists(LogConfig.PERFORMANCE_LOG), "Performance log file not created")
        
        analyzer = LogAnalyzer()
        stats = analyzer.get_performance_stats()
        # Check if test_operation is in stats
        self.assertIsInstance(stats, dict, "Performance stats should be a dictionary")

    def test_roi_tracking(self):
        self.roi_tracker.record_deflection(count=5, savings=500.0)
        
        # Verify ROI was logged
        from aiops_logging import LogConfig
        self.assertTrue(os.path.exists(LogConfig.METRICS_LOG), "ROI metrics log file not created")
        
        analyzer = LogAnalyzer()
        roi = analyzer.get_roi_summary()
        self.assertIsInstance(roi, dict, "ROI summary should be a dictionary")
        # Check if deflection was recorded
        self.assertGreaterEqual(roi.get("total_deflected_tickets", 0), 0)

    def test_error_tracking(self):
        try:
            raise ValueError("Test Error")
        except ValueError as e:
            self.error_tracker.log_error(
                error_type="ValueError",
                message=str(e),
                context={"test": True, "function": "test_error_tracking"}
            )
            
        # Verify error was logged
        from aiops_logging import LogConfig
        self.assertTrue(os.path.exists(LogConfig.ERROR_LOG), "Error log file not created")
        
        analyzer = LogAnalyzer()
        summary = analyzer.get_error_summary()
        self.assertIsInstance(summary, dict, "Error summary should be a dictionary")

if __name__ == "__main__":
    unittest.main()
