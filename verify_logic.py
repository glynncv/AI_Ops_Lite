import json
import pandas as pd
import os
from analysis import find_suspect_changes

def verify():
    # Load Incidents
    try:
        with open('data/input/incidents.json', 'r') as f:
            incidents = json.load(f)
        inc_df = pd.DataFrame(incidents)
        # Ensure dates
        inc_df['opened_at'] = pd.to_datetime(inc_df['opened_at'])
    except Exception as e:
        print(f"Error loading incidents for verification: {e}")
        return
    
    # Load Changes
    try:
        with open('data/input/changes.json', 'r') as f:
            changes = json.load(f)
        chg_df = pd.DataFrame(changes)
        chg_df['closed_at'] = pd.to_datetime(chg_df['closed_at'])
    except Exception as e:
        print(f"Error loading changes for verification: {e}")
        return
    
    print("Running find_suspect_changes (48h lookback)...")
    try:
        matches = []
        for _, row in inc_df.iterrows():
            res = find_suspect_changes(row, chg_df, lookback_hours=48)
            if res:
                for r in res:
                    matches.append((row['number'], r))

        print(f"Found {len(matches)} matches.")
        for m in matches:
            print(f"MATCH: Inc {m[0]} -> {m[1]}")
            
            
        if len(matches) > 0:
            print("SUCCESS: Suspect changes detected.")
        else:
            print("FAILURE: No suspect changes detected (expected some based on mock data).")
            
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    verify()
