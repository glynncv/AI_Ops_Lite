import pytest
import pandas as pd
import numpy as np
from analysis import (
    check_historical_recursion,
    perform_clustering,
    find_suspect_changes,
    detect_volume_spike,
    extract_entities,
    correlate_cluster_causes
)

# --- Fixtures ---
@pytest.fixture
def incident_df():
    data = {
        'number': ['INC001', 'INC002', 'INC003', 'INC004', 'INC005'],
        'short_description': [
            'Server web01 is down',
            'Database error on db-prod-02',
            'web01 not responding',
            'Network issue',
            'Login failure'
        ],
        'description': [
             'The server is not pinging',
             'Connection refused',
             'Timeout',
             'Switch failure',
             'Auth failed'
        ],
        'state': ['New', 'New', 'New', 'Closed', 'Resolved'],
        'opened_at': pd.to_datetime([
            '2023-01-01 10:00:00',
            '2023-01-01 10:05:00',
            '2023-01-01 10:10:00',
            '2023-01-01 11:00:00',
            '2023-01-02 09:00:00'
        ])
    }
    return pd.DataFrame(data)

@pytest.fixture
def changes_df():
    data = {
        'number': ['CHG001', 'CHG002'],
        'short_description': [
            'Update web01 firmware',
            'Database patch'
        ],
        'description': [
            'Applying security patch to web server',
            'Monthly DB maintenance'
        ],
        'closed_at': pd.to_datetime([
            '2023-01-01 09:00:00', # 1 hour before INC001
            '2022-12-31 10:00:00'
        ])
    }
    return pd.DataFrame(data)

# --- Tests ---

def test_extract_entities():
    text = "Error on server web-01 and IP 192.168.1.1 with ID 123456"
    entities = extract_entities(text)
    assert 'web-01' in entities
    assert '192.168.1.1' in entities
    assert '123456' in entities
    
    # Test emptiness
    assert extract_entities("") == []
    assert extract_entities(None) == []

def test_check_historical_recursion(incident_df):
    # web01 appears in INC001 and INC003
    repeaters = check_historical_recursion(incident_df)
    
    assert len(repeaters) > 0
    web_repeater = next((r for r in repeaters if 'web01' in r['Entity']), None)
    
    assert web_repeater is not None
    assert web_repeater['Incident Count'] == 2
    assert 'INC001' in web_repeater['Incidents']
    assert 'INC003' in web_repeater['Incidents']

def test_perform_clustering(incident_df):
    # Should cluster INC001 (Server web01) and INC003 (web01 not responding) potentially, 
    # depending on TF-IDF similarity.
    # Let's verify structure first.
    
    clustered = perform_clustering(incident_df)
    assert 'Cluster_ID' in clustered.columns
    # Ensure -1 is used for noise (or just exists)
    assert not clustered['Cluster_ID'].isnull().any()

def test_find_suspect_changes(incident_df, changes_df):
    # INC001 (web01 down) at 10:00. CHG001 (Update web01) at 09:00.
    # Should match based on time (1h diff) and keyword 'web01' (if entity extraction logic was used, 
    # but the function uses word intersection).
    # 'web01' is alphanumeric > 2 chars, so it should be tokenized.
    
    row = incident_df.iloc[0] # INC001
    suspects = find_suspect_changes(row, changes_df)
    
    assert len(suspects) > 0
    assert 'CHG001' in suspects[0]

def test_detect_volume_spike():
    # Create normal data
    dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
    counts = [10, 12, 11, 10, 13, 11, 10, 12, 11, 10]
    
    data = []
    for d, c in zip(dates, counts):
        for _ in range(c):
            data.append({'opened_at': d})
            
    df_normal = pd.DataFrame(data)
    is_spike, _ = detect_volume_spike(df_normal)
    assert not is_spike
    
    # Add a spike today (technically the last date in the range, let's pretend it is 'today' relative to the data)
    # The function reindexes to 'now' so strictly speaking we need to mock 'now' or ensure our data is recent.
    # However, the code: full_idx = pd.date_range(start=daily_counts.index.min(), end=pd.Timestamp.now().date(), freq='D')
    # This means if our data is old, it will fill 0s until today.
    # To test logic cleanly, we should trust the IsolationForest on the provided dataframe data mostly?
    # Actually, if we provide old data, it pads with 0s till today.
    
    # Let's construct data that ends TODAY.
    today = pd.Timestamp.now().normalize()
    dates = pd.date_range(end=today, periods=10, freq='D')
    
    data_spike = []
    for i, d in enumerate(dates):
        count = 10
        if i == 9: # Today
            count = 100 # SPIKE
        for _ in range(count):
             data_spike.append({'opened_at': d})
             
    df_spike = pd.DataFrame(data_spike)
    is_spike, series = detect_volume_spike(df_spike)
    
    # Note: IF might require more data or might fail on small samples without tuning.
    # The function has a fallback: if len < 5. Here len=10.
    # IF should pick it up.
    assert is_spike
    assert series.iloc[-1] == 100

def test_correlate_cluster_causes(incident_df, changes_df):
    # Mock a cluster df
    cluster_df = incident_df.copy()
    cluster_df['Cluster_ID'] = [1, 1, 1, 2, -1]
    
    # Incident 1,2,3 are cluster 1. Opened 10:00, 10:05, 10:10. Min is 10:00.
    # CHG001 closed 09:00. Keyword web01 checks out.
    
    matches = correlate_cluster_causes(cluster_df, changes_df)
    
    assert len(matches) > 0
    # Check if we matched cluster 1 to CHG001
    match = next((m for m in matches if m['Cluster_ID'] == 1), None)
    assert match is not None
    assert match['Suspect_Change'] == 'CHG001'
