import pandas as pd
from data_processor import fetch_changes, process_snow_data
from analysis import find_suspect_changes
import json
import os

def debug():
    print("DEBUG: Checking pd.Timestamp.now()")
    print(f"Current Time: {pd.Timestamp.now()}")
    
    print("\nDEBUG: Fetching Changes...")
    try:
        changes_df = fetch_changes()
        print(f"Fetched {len(changes_df)} changes.")
        if not changes_df.empty:
            print("Changes Columns:", changes_df.columns)
            print("Changes Head:")
            print(changes_df[['number', 'closed_at', 'short_description']].head())
            print("Closed At Type:", changes_df['closed_at'].dtype)
        else:
            print("WARNING: changes_df is empty!")
            print("This might be due to the 30-day filter.")
            
    except Exception as e:
        print(f"ERROR calling fetch_changes: {e}")
        return

    print("\nDEBUG: Loading Incidents...")
    try:
        with open('data/input/incidents.json', 'r') as f:
            incidents = json.load(f)
        df_cleaned = process_snow_data(incidents)
        print(f"Loaded {len(df_cleaned)} incidents.")
    except Exception as e:
        print(f"ERROR loading incidents: {e}")
        return

    print("\nDEBUG: Correlating...")
    if not df_cleaned.empty and not changes_df.empty:
        try:
            results = []
            for _, row in df_cleaned.iterrows():
                res = find_suspect_changes(row, changes_df, lookback_hours=48)
                if res:
                    results.append((row['number'], res))
            
            print(f"Found {len(results)} incidents with suspect changes.")
            for r in results:
                print(r)
        except Exception as e:
            print(f"ERROR during correlation: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Skipping correlation due to empty df.")

if __name__ == "__main__":
    debug()
