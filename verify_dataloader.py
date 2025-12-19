import sys
from unittest.mock import MagicMock

# Mock streamlit BEFORE importing data_loader
mock_st = MagicMock()
mock_st.error.side_effect = lambda x: print(f"[ST ERROR] {x}")
mock_st.warning.side_effect = lambda x: print(f"[ST WARNING] {x}")
mock_st.success.side_effect = lambda x: print(f"[ST SUCCESS] {x}")
# Also mock specific internal attributes if touched, but side_effect print is usually enough for top level functions
sys.modules["streamlit"] = mock_st

from data_loader import DataLoader
import pandas as pd
import os

def verify():
    print("Initializing DataLoader...")
    # Use the absolute path if needed, or relative if CWD is correct
    loader = DataLoader(data_dir=r'c:\Users\cglynn\myPython\SNOW_MI_Flight_Deck\AI_Ops_Lite\data\input')
    
    print("\n--- Loading Incidents ---")
    inc_df = loader.load_incidents()
    if not inc_df.empty:
        print(f"Success! Loaded {len(inc_df)} incidents.")
        print("Columns:", list(inc_df.columns))
        # Check date parsing
        if 'opened_at' in inc_df.columns:
            print(f"Date check (opened_at): {inc_df['opened_at'].dtype}")
            print(f"Sample date: {inc_df['opened_at'].iloc[0]}")
    else:
        print("FAILED to load Incidents.")

    print("\n--- Loading Changes ---")
    chg_df = loader.load_changes()
    if not chg_df.empty:
        print(f"Success! Loaded {len(chg_df)} changes.")
        print("Columns:", list(chg_df.columns))
    else:
        print("FAILED to load Changes.")

    print("\n--- Loading Problems ---")
    prb_df = loader.load_problems()
    if not prb_df.empty:
        print(f"Success! Loaded {len(prb_df)} problems.")
        print("Columns:", list(prb_df.columns))
    else:
        print("FAILED to load Problems.")

if __name__ == "__main__":
    verify()
