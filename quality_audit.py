import pandas as pd
from textblob import TextBlob
from datetime import datetime
import plotly.express as px

def calculate_sentiment(text):
    """
    Calculates sentiment polarity using TextBlob.
    Returns a float between -1.0 (Negative) and 1.0 (Positive).
    """
    if not isinstance(text, str):
        return 0.0
    return TextBlob(text).sentiment.polarity

def identify_toxic_tickets(df, threshold=-0.5):
    """
    Identifies tickets with negative sentiment in short_description.
    """
    if df.empty or 'short_description' not in df.columns:
        return pd.DataFrame()
    
    # Calculate sentiment
    df['sentiment'] = df['short_description'].apply(calculate_sentiment)
    
    # Filter
    toxic_df = df[df['sentiment'] < threshold].copy()
    
    return toxic_df

def identify_zombie_tickets(df, days_threshold=30):
    """
    Identifies tickets open > 30 days that are NOT 'On Hold' or 'Closed'/'Resolved'.
    """
    if df.empty or 'opened_at' not in df.columns or 'state' not in df.columns:
        return pd.DataFrame()
    
    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df['opened_at']):
        df['opened_at'] = pd.to_datetime(df['opened_at'], errors='coerce')
        
    now = pd.Timestamp.now()
    cutoff = now - pd.Timedelta(days=days_threshold)
    
    # Filter: Opened before cutoff AND state is NOT closed/resolved/on hold
    # Adjusted states based on common SNOW lifecycle, prompt specifically mentions "NOT 'On Hold'"
    # Ideally we should also exclude Closed/Resolved ones.
    
    zombies = df[
        (df['opened_at'] < cutoff) & 
        (~df['state'].isin(['Closed', 'Resolved', 'On Hold']))
    ].copy()
    
    # Calculate Age in Days for display
    zombies['Age (Days)'] = (now - zombies['opened_at']).dt.days
    
    return zombies

def get_ping_pong_data(df, min_reassignments=3):
    """
    Filters tickets with high reassignment counts for the Wall of Shame.
    """
    if df.empty or 'reassignment_count' not in df.columns:
        return pd.DataFrame()
        
    ping_pong_df = df[df['reassignment_count'] > min_reassignments].copy()
    return ping_pong_df
