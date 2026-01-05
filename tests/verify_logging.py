import unittest
import os
import shutil
import json
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aiops_logging import AuditLogger, PerformanceTracker, ROITracker, ErrorTracker
from log_analyzer import LogAnalyzer

class TestLoggingSystem(unittest.TestCase):
    def setUp(self):
        # Create a temporary log directory
        self.test_log_dir = "tests/temp_logs"
        if os.path.exists(self.test_log_dir):
            shutil.rmtree(self.test_log_dir)
        os.makedirs(self.test_log_dir)

        # Initialize loggers with the test directory
        self.audit_logger = AuditLogger(log_dir=self.test_log_dir)
        self.perf_tracker = PerformanceTracker(log_dir=self.test_log_dir)
        self.roi_tracker = ROITracker(log_dir=self.test_log_dir)
        self.error_tracker = ErrorTracker(log_dir=self.test_log_dir)

    def tearDown(self):
        # Cleanup
        if os.path.exists(self.test_log_dir):
            shutil.rmtree(self.test_log_dir)

    def test_audit_logging(self):
        # Log an event
        self.audit_logger.log_event(
            event_type="test_event",
            user="test_user",
            details="test_details"
        )
        
        # specific check for file existence
        expected_file = os.path.join(self.test_log_dir, "audit_trail.json")
        self.assertTrue(os.path.exists(expected_file), "Audit log file not created")

        # Verify with analyzer
        analyzer = LogAnalyzer(log_dir=self.test_log_dir)
        events = analyzer.get_recent_audit_trail()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "test_event")

    def test_performance_tracking(self):
        # Track execution time
        with self.perf_tracker.track_operation("test_op"):
            pass
        
        analyzer = LogAnalyzer(log_dir=self.test_log_dir)
        stats = analyzer.get_performance_stats()
        self.assertIn("test_op", stats)
        self.assertEqual(stats["test_op"]["count"], 1)

    def test_roi_tracking(self):
        self.roi_tracker.record_deflection(count=5, savings=500.0)
        
        analyzer = LogAnalyzer(log_dir=self.test_log_dir)
        roi = analyzer.get_roi_summary()
        self.assertEqual(roi["total_deflected_tickets"], 5)
        self.assertEqual(roi["estimated_savings"], 500.0)

    def test_error_tracking(self):
        try:
            raise ValueError("Test Error")
        except ValueError as e:
            self.error_tracker.log_error("ValueError", str(e))
            
        analyzer = LogAnalyzer(log_dir=self.test_log_dir)
        summary = analyzer.get_error_summary()
        self.assertIn("ValueError", summary)
        self.assertEqual(summary["ValueError"], 1)

if __name__ == "__main__":
    unittest.main()
