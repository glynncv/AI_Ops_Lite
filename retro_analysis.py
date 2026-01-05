import plotly.graph_objects as go
import pandas as pd
from collections import defaultdict
import re

def create_timeline_fusion_chart(incidents_df, problems_df, site_name=None):
    """
    Creates a Plotly Scatter plot combining Incidents (Blue Dots) and Problems (Red Lines).
    If a site_name is provided, logic assumes dfs are already filtered or we filter here.
    Ideally, filtering happens before calling this to keep plotting pure, 
    but for the title/context, site_name is useful.
    
    Trace 1 (Blue Dots): Incidents (opened_at vs number).
    Trace 2 (Red Lines): Problem Records (opened_at to closed_at).
    """
    fig = go.Figure()

    # 1. Trace: Incidents (Blue Dots)
    if not incidents_df.empty and 'opened_at' in incidents_df.columns:
        # Filter for valid dates
        incs = incidents_df.dropna(subset=['opened_at']).copy()
        
        # Sort by opened_at for better plotting behavior
        incs = incs.sort_values('opened_at')
        
        fig.add_trace(go.Scatter(
            x=incs['opened_at'],
            y=incs['number'], 
            mode='markers',
            name='Incidents',
            marker=dict(color='#1E88E5', size=10, opacity=0.8), # Material Blue
            text=incs['short_description'],
            hovertemplate="<b>%{y}</b><br>%{x}<br>%{text}<extra></extra>"
        ))
    
    # 2. Trace: Problems (Red Lines)
    if not problems_df.empty and 'opened_at' in problems_df.columns and 'closed_at' in problems_df.columns:
        # Filter problems with valid start/end
        probs = problems_df.dropna(subset=['opened_at', 'closed_at']).copy()
        
        for idx, row in probs.iterrows():
            prb_num = row.get('number', f'PRB_{idx}')
            
            fig.add_trace(go.Scatter(
                x=[row['opened_at'], row['closed_at']],
                y=[prb_num, prb_num],
                mode='lines+markers',
                name=f"Fixed: {prb_num}",
                line=dict(color='#D32F2F', width=4), # Material Red
                marker=dict(color='#D32F2F', size=8, symbol='square'),
                text=f"{prb_num}: {row.get('short_description', '')}",
                hovertemplate="<b>%{text}</b><br>Start: %{x}<extra></extra>"
            ))
            
    title_text = "The 'Groundhog Day' Analysis: Failed Fixes"
    if site_name:
        title_text += f" - Site: {site_name}"

    fig.update_layout(
        title=title_text,
        xaxis_title="Timeline",
        yaxis_title="Record ID",
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        hovermode="closest"
    )
    
    return fig

def identify_zombie_problems(problems_df):
    """
    Identifies 'Zombie Problems': Recurring PRBs per site.
    Groups by 'location' (or 'location.name').
    Counts number of PRBs per site.
    Returns Sites with >1 Problem Record.
    """
    if problems_df.empty:
        return pd.DataFrame()
    
    # Normalized location column
    # Check for likely location columns
    loc_col = None
    cols = [c.lower() for c in problems_df.columns]
    
    if 'location' in problems_df.columns:
        loc_col = 'location'
    elif 'location.name' in problems_df.columns:
        loc_col = 'location.name'
    elif 'u_location' in problems_df.columns:
        loc_col = 'u_location'
        
    if not loc_col:
        return pd.DataFrame()
        
    # Group by Location
    # Filter out empty locations
    df_clean = problems_df[problems_df[loc_col].astype(str) != ''].copy()
    
    if df_clean.empty:
         return pd.DataFrame()

    stats = df_clean.groupby(loc_col).agg(
        Problem_Count=('number', 'nunique'),
        Problem_Records=('number', lambda x: ", ".join(x.unique()))
    ).reset_index()
    
    # Filter for > 1
    zombies = stats[stats['Problem_Count'] > 1].sort_values('Problem_Count', ascending=False)
    
    # Attempt to extract a cleaner 'Site Name'
    # The format might be "00274 - Gillingham - United Kingdom" OR just "10610" if data is dirty.
    # Goal: Remove leading digits and separators to find the Name.
    def extract_site_name(loc_str):
        s = str(loc_str).strip()
        # Regex: Replace leading (Digits + Spaces + Dashes) with empty string
        # e.g. "00274 - Gillingham" -> "Gillingham"
        # e.g. "10610" -> "10610" (No change if no letters found, to be safe?)
        
        # If we just strip leading non-letters?
        # "00274 - Gillingham" -> "Gillingham..."
        
        import re
        # Match pattern: Start of line, any digits, optional spaces/dashes
        clean = re.sub(r'^[\d\s-]+', '', s)
        
        if clean:
            # If we have a result, use it.
            # Only issue: "United Kingdom" might remain if it was "ID - Country". 
            # But usually it is "ID - Name - Country".
            # Let's try to just take the first part of the matcher if it was split by ' - '?
            
            # Let's keep existing split logic as primary but refine it
            parts = s.split(' - ')
            if len(parts) >= 2:
                # 00274 - Gillingham - UK
                # parts[0] = 00274
                # parts[1] = Gillingham
                return parts[1]
                
        # Fallback: if split didn't work, maybe it was "10610 Gillingham"?
        # Try the regex removal of leading numbers
        clean_fallback = re.sub(r'^[\d\s-]+', '', s)
        if clean_fallback and clean_fallback != s:
             return clean_fallback
             
        return s
        
    if not zombies.empty:
        zombies['Site'] = zombies[loc_col].apply(extract_site_name)
        # Drop the original redundant ID/Location column
        # Keep Site, Problem_Count, Problem_Records
        zombies = zombies[['Site', 'Problem_Count', 'Problem_Records']]
    
    return zombies

def calculate_deflection_opportunity(incidents_df):
    """
    Deflection Opportunity:
    Filter inc_df for 'Simple' keywords: ['password', 'reset', 'access', 'admin', 'install'].
    Display a metric: 'Potential Zero-Touch Tickets' (Count & Percentage of total).
    
    Returns:
        count (int): Number of matches
        percentage (float): Percentage of total incidents
        savings (float): Estimated cost savings ($50/ticket)
        df (pd.DataFrame): The filtered dataframe
    """
    if incidents_df.empty:
        return 0, 0.0, 0, pd.DataFrame()
        
    keywords = ['password', 'reset', 'access', 'admin', 'install']
    pattern = '|'.join(keywords)
    
    total_count = len(incidents_df)
    
    # Filter
    # Check short_description
    mask = incidents_df['short_description'].fillna('').str.contains(pattern, case=False, regex=True)
    deflectable = incidents_df[mask]
    
    count = len(deflectable)
    if total_count > 0:
        pct = (count / total_count)
    else:
        pct = 0.0
        
    estimated_cost_per_ticket = 50 # Assumption
    savings = count * estimated_cost_per_ticket
    
    return count, pct, savings, deflectable[['number', 'short_description', 'opened_at', 'assignment_group']]
