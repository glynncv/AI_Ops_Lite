import sys
import os
import pandas as pd
import pytest

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from aiops_intelligence import (
        find_similar_resolved_incidents,
        IntelligentRouter,
        suggest_problem_creation,
        batch_suggest_problems
    )
    print("Import successful")
except Exception as e:
    print(f"Import failed: {e}")
    sys.exit(1)

import test_aiops_intelligence as test_module

# Bind functions from module
test_find_similar_resolved_incidents_func = test_module.test_find_similar_resolved_incidents
test_find_similar_no_match = test_module.test_find_similar_no_match
test_intelligent_router = test_module.test_intelligent_router
test_router_insufficient_data = test_module.test_router_insufficient_data
test_suggest_problem_creation = test_module.test_suggest_problem_creation
test_no_problem_under_threshold = test_module.test_no_problem_under_threshold
test_batch_suggest_problems = test_module.test_batch_suggest_problems

def get_historical_incidents():
    data = {
        'number': ['INC001', 'INC002', 'INC003', 'INC004'],
        'short_description': [
            'Database connection timeout',
            'VPN connection failed',
            'Server slow response',
            'Database deadlock'
        ],
        'description': [
            'Cannot connect to DB',
            'VPN error 404',
            'CPU high',
            'Transaction deadlock'
        ],
        'state': ['Closed', 'Resolved', 'Closed', 'Closed'],
        'close_notes': [
            'Restarted DB service',
            'Reset VPN profile',
            'Cleared cache',
            'Killed process'
        ],
        'assignment_group': ['DBA', 'Network', 'Server', 'DBA'],
        'opened_at': ['2023-01-01 10:00:00', '2023-01-01 11:00:00', '2023-01-01 12:00:00', '2023-01-02 09:00:00'],
        'closed_at': ['2023-01-01 12:00:00', '2023-01-01 12:00:00', '2023-01-01 14:00:00', '2023-01-02 10:00:00'],
        'sys_updated_at': ['2023-01-01', '2023-01-01', '2023-01-01', '2023-01-02']
    }
    return pd.DataFrame(data)

def get_clustered_incidents():
    data = {
        'number': ['INC001', 'INC002', 'INC003', 'INC004'],
        'short_description': [
            'DB Connection Fail',
            'DB Timeout',
            'DB Error',
            'Network Slow'
        ],
        'Cluster_ID': [1, 1, 1, 2],
        'assignment_group': ['DBA', 'DBA', 'DBA', 'Network'],
        'priority': ['High', 'High', 'Medium', 'Low'],
        'opened_at': pd.to_datetime(['2023-01-01'] * 4)
    }
    return pd.DataFrame(data)

def run_tests():
    print("Setting up data...")
    hist_df = get_historical_incidents()
    clus_df = get_clustered_incidents()
    
    print("Running test_find_similar_resolved_incidents...")
    test_find_similar_resolved_incidents_func(hist_df)
    print("PASS")

    print("Running test_find_similar_no_match...")
    test_find_similar_no_match(hist_df)
    print("PASS")

    print("Running test_intelligent_router...")
    test_intelligent_router(hist_df)
    print("PASS")

    print("Running test_router_insufficient_data...")
    test_router_insufficient_data(hist_df)
    print("PASS")

    print("Running test_suggest_problem_creation...")
    test_suggest_problem_creation(clus_df)
    print("PASS")

    print("Running test_no_problem_under_threshold...")
    test_no_problem_under_threshold(clus_df)
    print("PASS")

    print("Running test_batch_suggest_problems...")
    test_batch_suggest_problems(clus_df)
    print("PASS")

if __name__ == "__main__":
    try:
        run_tests()
        print("All manual tests passed!")
    except Exception as e:
        print(f"Test FAILED: {e}")
        import traceback
        traceback.print_exc()
