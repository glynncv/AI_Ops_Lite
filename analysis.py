import re
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN

def extract_entities(text):
    """
    Extracts potential entities (IPs, 6-digit IDs, Server Names) from text using Regex.
    """
    if not isinstance(text, str):
        return []
    
    entities = []
    
    # Regex Patterns
    
    # 1. IP Addresses (IPv4) - Simple pattern
    # Matches 4 groups of 1-3 digits separated by dots
    ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    entities.extend(re.findall(ip_pattern, text))
    
    # 2. 6-Digit Numbers (Potential IDs)
    # strictly 6 digits
    id_pattern = r'\b\d{6}\b'
    entities.extend(re.findall(id_pattern, text))
    
    # 3. Server Names / Asset IDs
    # Heuristic: Alphanumeric string (3+ chars) containing at least one digit and one letter.
    # Matches things like: "web01", "db-server-02", "NYC-09"
    # Excludes: "123456" (caught by id_pattern or just numbers), "Error" (just letters)
    server_pattern = r'\b(?=.*\d)(?=.*[a-zA-Z])[a-zA-Z0-9-]{3,}\b'
    entities.extend(re.findall(server_pattern, text))
    
    # Dedup within the single text string
    return list(set(entities))

def check_historical_recursion(df):
    """
    Scans the DataFrame for recurring entities in short_description.
    Returns a list of dictionaries for the repeat offenders.
    """
    entity_map = defaultdict(list)
    
    if 'short_description' not in df.columns or 'number' not in df.columns:
        return []
        
    for index, row in df.iterrows():
        desc = str(row.get('short_description', ''))
        incident_num = row.get('number', 'Unknown')
        
        found_entities = extract_entities(desc)
        
        for entity in found_entities:
            entity_map[entity].append(incident_num)
            
    # Filter for entities appearing in > 1 incident
    repeat_offenders = []
    for entity, incidents in entity_map.items():
        if len(incidents) > 1:
            repeat_offenders.append({
                'Entity': entity,
                'Incident Count': len(incidents),
                'Incidents': ", ".join(incidents)
            })
            
    # Sort by count descending
    repeat_offenders.sort(key=lambda x: x['Incident Count'], reverse=True)
    
    return repeat_offenders

def perform_clustering(df):
    """
    Performs DBSCAN clustering on incidents based on text similarity.
    Returns the DataFrame with an added 'Cluster_ID' column.
    """
    if df.empty:
        return df

    # Create a wrapper or copy to avoid SettingWithCopy warning on the original if it's a slice
    df_clustered = df.copy()
    
    # Create a combined text column for analysis
    # Ensure we are working with string data
    df_clustered['combined_text'] = (
        df_clustered['short_description'].fillna('') + " " + 
        df_clustered['description'].fillna('')
    )
    
    # Vectorize the text
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df_clustered['combined_text'])
    
    # Apply DBSCAN clustering
    # metric='cosine' is good for text similarity
    # eps=0.5 is a starting point, might need tuning
    dbscan = DBSCAN(eps=0.5, min_samples=2, metric='cosine')
    clusters = dbscan.fit_predict(tfidf_matrix)
    
    # Add cluster IDs to the DataFrame
    df_clustered['Cluster_ID'] = clusters
    
    return df_clustered



def find_suspect_changes(incident_row, changes_df, lookback_hours=48):
    """
    Identifies suspect changes for a single incident based on time and text overlap.
    Returns a list of strings describing the suspect changes.
    """
    suspects = []
    
    if changes_df.empty:
        return suspects
        
    inc_start = incident_row['opened_at']
    if pd.isnull(inc_start):
        return suspects
        
    # Time Filter
    window_start = inc_start - pd.Timedelta(hours=lookback_hours)
    
    # Filter changes in time window (closed before incident opened)
    # Using 'closed_at' as per prompt requirements
    time_matches = changes_df[
        (changes_df['closed_at'] >= window_start) & 
        (changes_df['closed_at'] <= inc_start)
    ]
    
    if time_matches.empty:
        return suspects
        
    # Text Filter keys
    inc_desc = str(incident_row.get('short_description', '')).lower()
    # Simple stop words list (can be expanded)
    stop_words = {'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'on', 'at', 'for', 'with', 'by', 'is', 'it', 'this', 'that', 'issue', 'error', 'problem', 'failed', 'failure'}
    
    # Tokenize and filter
    inc_words = set(re.findall(r'\w+', inc_desc))
    inc_words = {w for w in inc_words if w not in stop_words and len(w) > 2}
    
    for _, chg in time_matches.iterrows():
        chg_desc = str(chg.get('short_description', '')).lower()
        chg_words = set(re.findall(r'\w+', chg_desc))
        
        # Check for intersection
        common_words = inc_words.intersection(chg_words)
        
        if common_words:
            # Format: "Number: Description (Matches: kw, kw)"
            match_str = f"{chg['number']}: {chg['short_description']} (Matches: {', '.join(common_words)})"
            suspects.append(match_str)
            
    return suspects

def identiy_l0_tickets(df):
    """
    Identifies L0 (Deflection Potential) tickets based on keywords in close_notes and short_description.
    Returns the count of matching tickets.
    """
    if df.empty:
        return 0
        
    keywords = ["password", "reset", "training", "advice", "cache", "clear", "restart", "reboot"]
    
    # Create combined text for search
    combined_text = (
        df.get('short_description', '').fillna('').astype(str).str.lower() + " " +
        df.get('close_notes', '').fillna('').astype(str).str.lower()
    )
    
    # Check if any keyword matches
    # This is a simple check; can be optimized with regex for word boundaries if needed
    pattern = '|'.join(keywords)
    match_count = combined_text.str.contains(pattern, case=False, na=False).sum()
    
    return int(match_count)

def calculate_operational_risk(cluster_count, repeat_count):
    """
    Calculates Operational Risk Level based on cluster and repeat offender counts.
    Returns 'High', 'Medium', or 'Low'.
    """
    if cluster_count > 3 or repeat_count > 5:
        return 'High'
    elif cluster_count > 0 or repeat_count > 0:
        return 'Medium'
    else:
        return 'Low'

# --- Phase 2: Current Risks Logic ---

from sklearn.ensemble import IsolationForest

def detect_volume_spike(incidents_df):
    """
    Detects if the current daily incident volume is a spike (anomaly) using IsolationForest.
    Returns:
        is_spike (bool): True if spike detected
        daily_counts (pd.Series): Time series of counts for visualization (optional context)
    """
    if incidents_df.empty:
        return False, pd.Series()

    df = incidents_df.copy()
    
    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df['opened_at']):
        df['opened_at'] = pd.to_datetime(df['opened_at'], errors='coerce')
        
    # Drop rows with no date
    df = df.dropna(subset=['opened_at'])
    if df.empty:
        return False, pd.Series()
        
    # Normalize to date level
    df['date'] = df['opened_at'].dt.date
    
    # Count per day
    daily_counts = df.groupby('date').size()
    
    # Reindex to fill missing days (ensure continuous time series)
    if len(daily_counts) > 0:
        full_idx = pd.date_range(start=daily_counts.index.min(), end=pd.Timestamp.now().date(), freq='D')
        daily_counts.index = pd.DatetimeIndex(daily_counts.index)
        daily_counts = daily_counts.reindex(full_idx, fill_value=0)
    
    # Need enough data points to train?? Let's say at least 5 days, else just threshold?
    # For now, if < 5 days, we can't really do isolation forest well, return False or simple std dev check.
    # We will try IF anyway, it handles small data but might warn.
    if len(daily_counts) < 5:
        # Fallback: Is today > 2 * avg?
        avg = daily_counts.mean()
        if daily_counts.iloc[-1] > avg * 2 and daily_counts.iloc[-1] > 5:
            return True, daily_counts
        return False, daily_counts
        
    # Prepare for sklearn (n_samples, n_features)
    X = daily_counts.values.reshape(-1, 1)
    
    # Train Isolation Forest
    # contamination='auto' or low value like 0.05 to detect top outliers
    clf = IsolationForest(contamination=0.05, random_state=42)
    clf.fit(X)
    
    # Predict
    preds = clf.predict(X)
    
    # Check the last data point (Today/Current)
    # -1 is outlier, 1 is inlier
    is_spike = preds[-1] == -1
    
    # Additional check: Is it a HIGH outlier? (IF detects outliers in both directions usually, 
    # but for volume spike we typically care about High. though Low volume could be an issue too (monitoring broke).
    # For "Volume Spike Detected" prompt, we assume High.
    # Simple check: Is value > mean?
    if is_spike and daily_counts.iloc[-1] < daily_counts.mean():
        is_spike = False
        
    return is_spike, daily_counts

def cluster_open_incidents(incidents_df):
    """
    Clusters OPEN incidents using Description + TF-IDF + DBSCAN.
    """
    if incidents_df.empty:
        return pd.DataFrame()
        
    # Filter Open Incidents
    # Assuming 'state' column. Open usually means NOT Closed/Resolved/Canceled.
    # Adjust states based on your actual data values.
    # Based on incidents.json, we see 'New', 'Closed'. 
    # We'll treat anything NOT 'Closed', 'Resolved', 'Canceled' as Open.
    open_mask = ~incidents_df['state'].isin(['Closed', 'Resolved', 'Canceled', 'Cancelled'])
    open_incidents = incidents_df[open_mask].copy()
    
    if open_incidents.empty:
        return pd.DataFrame()

    # Reuse existing clustering logic or re-implement?
    # The existing perform_clustering takes a df and returns it with Cluster_ID.
    # reusing it is safer and cleaner as it uses the same params.
    # NOTE: perform_clustering uses 'short_description' + 'description' and DBSCAN(eps=0.5).
    
    # Helper from this file (analysis.py)
    # We can just call perform_clustering on the filtered subset.
    df_clustered = perform_clustering(open_incidents)
    
    return df_clustered

def correlate_cluster_causes(cluster_df, changes_df):
    """
    For each valid cluster (ID != -1), identify potential root cause changes.
    Lookback: 48h from Cluster Start Time (min opened_at of incident in cluster).
    Keyword Match: Intersection of words.
    """
    matches = []
    
    if cluster_df.empty or 'Cluster_ID' not in cluster_df.columns:
        return matches
        
    if changes_df.empty:
        return matches

    # Ensure change dates
    if 'closed_at' not in changes_df.columns:
        return matches
        
    if not pd.api.types.is_datetime64_any_dtype(changes_df['closed_at']):
        changes_df['closed_at'] = pd.to_datetime(changes_df['closed_at'], errors='coerce')
        
    # Group by Cluster
    clusters = cluster_df[cluster_df['Cluster_ID'] != -1].groupby('Cluster_ID')
    
    for cluster_id, group in clusters:
        # Cluster Start = Earliest Incident Open Time
        cluster_start = group['opened_at'].min()
        
        # Lookback 48h
        window_start = cluster_start - pd.Timedelta(hours=48)
        
        # Filter Changes
        potential_changes = changes_df[
            (changes_df['closed_at'] >= window_start) & 
            (changes_df['closed_at'] <= cluster_start)
        ]
        
        if potential_changes.empty:
            continue
            
        # Extract keywords from Cluster
        # We can combine all titles in the cluster to get a rich bag of words
        cluster_text = " ".join(group['short_description'].fillna('').astype(str))
        cluster_words = set(re.findall(r'\w{4,}', cluster_text.lower())) # distinct words > 3 chars
        
        # Stopwords (English) - quick minimal list
        stops = {'with', 'that', 'this', 'from', 'have', 'error', 'issue', 'fail', 'failed', 'problem', 'unable', 'cannot', 'when', 'what', 'code', 'data', 'please', 'help'}
        cluster_words = cluster_words - stops
        
        for _, chg in potential_changes.iterrows():
            chg_text = str(chg.get('short_description', '')) + " " + str(chg.get('description', ''))
            chg_words = set(re.findall(r'\w{4,}', chg_text.lower()))
            chg_words = chg_words - stops
            
            # Intersection
            common = cluster_words.intersection(chg_words)
            
            # Threshold? At least 1 or 2 meaningful words?
            # Let's say at least 1 strong keyword match for now.
            if len(common) >= 1:
                matches.append({
                    'Cluster_ID': cluster_id,
                    'Cluster_Size': len(group),
                    'Cluster_Start': cluster_start,
                    'Suspect_Change': chg['number'],
                    'Change_Title': chg.get('short_description', 'N/A'),
                    'Closed_At': chg['closed_at'],
                    'Matched_Keywords': ", ".join(list(common)[:5]) # Show top 5
                })
                
    return matches
