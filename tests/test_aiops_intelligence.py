import pytest
import pandas as pd
import numpy as np
from aiops_intelligence import (
    find_similar_resolved_incidents,
    IntelligentRouter,
    suggest_problem_creation,
    batch_suggest_problems
)

# Test Data Fixtures
@pytest.fixture
def historical_incidents():
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

@pytest.fixture
def clustered_incidents():
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

# Test Similar Incidents
def test_find_similar_resolved_incidents(historical_incidents):
    # Case 1: Match DB issue
    new_desc = "Database is timing out"
    similar = find_similar_resolved_incidents(new_desc, historical_incidents, top_n=2)
    
    assert len(similar) > 0
    # Should match INC001 or INC004 (DB issues)
    assert any(s['incident_number'] in ['INC001', 'INC004'] for s in similar)
    assert similar[0]['similarity_score'] > 0

def test_find_similar_no_match(historical_incidents):
    # Case 2: Totally unrelated
    new_desc = "Printer is out of paper xyz123"
    similar = find_similar_resolved_incidents(new_desc, historical_incidents, top_n=1)
    # Might still find something if threshold is low, but score should be low. 
    # Current implementation threshold is 0.1
    if similar:
        assert similar[0]['similarity_score'] < 0.5

# Test Intelligent Router
def test_intelligent_router(historical_incidents):
    router = IntelligentRouter()
    
    # Train
    # Need enough data - implementation requires >= 10 samples
    # Let's mock a larger dataset
    large_df = pd.concat([historical_incidents] * 3, ignore_index=True)
    metrics = router.train(large_df)
    
    assert metrics['success'] is True
    assert router.trained is True
    
    # Predict
    preds = router.predict_assignment("Database is down", top_n=1)
    assert preds[0]['assignment_group'] == 'DBA'
    
    preds_net = router.predict_assignment("VPN not working", top_n=1)
    assert preds_net[0]['assignment_group'] == 'Network'

def test_router_insufficient_data(historical_incidents):
    router = IntelligentRouter()
    metrics = router.train(historical_incidents) # Only 4 rows
    assert 'error' in metrics
    assert not router.trained

# Test Problem Suggestions
def test_suggest_problem_creation(clustered_incidents):
    # Cluster 1 has 3 DB incidents. Threshold 3.
    suggestion = suggest_problem_creation(clustered_incidents, cluster_id=1, threshold=3)
    
    assert suggestion is not None
    assert suggestion['should_create'] is True
    assert suggestion['incident_count'] == 3
    assert suggestion['priority'] == 'High'
    assert 'DB' in suggestion['problem_title'] or 'Connection' in suggestion['problem_title']

def test_no_problem_under_threshold(clustered_incidents):
    # Cluster 1 has 3 incidents. If threshold is 4, should return None.
    suggestion = suggest_problem_creation(clustered_incidents, cluster_id=1, threshold=4)
    assert suggestion is None

def test_batch_suggest_problems(clustered_incidents):
    suggestions = batch_suggest_problems(clustered_incidents, threshold=3)
    assert len(suggestions) == 1
    assert suggestions[0]['cluster_id'] == 1
