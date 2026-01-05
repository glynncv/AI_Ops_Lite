import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from data_loader import DataLoader
import io

@pytest.fixture
def data_loader():
    return DataLoader(data_dir='test_data')

def test_load_incidents_no_files(data_loader):
    with patch('data_loader.glob.glob', return_value=[]):
        df = data_loader.load_incidents()
        assert df.empty

def test_load_incidents_success(data_loader):
    mock_csv = "number,short_description,opened_at\nINC123,Test,2023-01-01"
    
    with patch('data_loader.glob.glob', return_value=['test_IM.csv']):
        with patch('data_loader.pd.read_csv', return_value=pd.read_csv(io.StringIO(mock_csv))):
            df = data_loader.load_incidents()
            assert not df.empty
            assert 'number' in df.columns
            assert df.iloc[0]['number'] == 'INC123'
            # Helper columns should be added
            assert 'state' in df.columns

def test_load_changes_proxy_closed_at(data_loader):
    # CSV missing closed_at but has end_date
    mock_csv = "number,short_description,end_date\nCHG001,Change 1,2023-01-02"
    
    with patch('data_loader.glob.glob', return_value=['test_CHANGES.csv']):
        with patch('data_loader.pd.read_csv', return_value=pd.read_csv(io.StringIO(mock_csv))):
            df = data_loader.load_changes()
            assert not df.empty
            assert 'closed_at' in df.columns # Should be created from end_date
            assert pd.to_datetime(df.iloc[0]['closed_at']) == pd.to_datetime("2023-01-02")

def test_load_changes_missing_all_dates(data_loader):
    mock_csv = "number,short_description\nCHG001,Change 1"
    
    with patch('data_loader.glob.glob', return_value=['test_CHANGES.csv']):
        with patch('data_loader.pd.read_csv', return_value=pd.read_csv(io.StringIO(mock_csv))):
            df = data_loader.load_changes()
            assert 'closed_at' in df.columns
            # Should be NaT
            assert pd.isna(df.iloc[0]['closed_at'])

def test_load_problems_success(data_loader):
    mock_csv = "number,problem_id,u_resolved\nPRB001,P100,2023-01-03"
    
    with patch('data_loader.glob.glob', return_value=['test_PM.csv']):
        with patch('data_loader.pd.read_csv', return_value=pd.read_csv(io.StringIO(mock_csv))):
            df = data_loader.load_problems()
            assert not df.empty
            # u_resolved should fill closed_at
            assert 'closed_at' in df.columns
            assert pd.to_datetime(df.iloc[0]['closed_at']) == pd.to_datetime("2023-01-03")

def test_read_csv_encoding_fallback(data_loader):
    # This is harder to mock perfectly without creating files with bad encoding.
    # We can mock read_csv to raise UnicodeDecodeError on first call.
    
    side_effect = [UnicodeDecodeError('utf-8', b'', 1, 1, 'error'), pd.DataFrame({'col': [1]})]
    
    with patch('data_loader.pd.read_csv', side_effect=side_effect) as mock_read:
        # We need to pass a mock file object that has a seek method
        mock_file = MagicMock()
        
        df = data_loader._read_csv(mock_file)
        
        # Should have called read_csv twice
        assert mock_read.call_count == 2
        assert not df.empty
        # Verify seek was called
        mock_file.seek.assert_called_with(0)
