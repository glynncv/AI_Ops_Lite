import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from collections import defaultdict
import re

def create_timeline_fusion_chart(incidents_df, problems_df):
    """
    Creates a Plotly Scatter plot combining Incidents (Blue Dots) and Problems (Red Lines).
    X-Axis: Time
    Trace 1: Incidents (Blue Dots)
    Trace 2: Problems (Red Lines - Start to End)
    """
    fig = go.Figure()

    # 1. Trace: Incidents (Blue Dots)
    if not incidents_df.empty and 'opened_at' in incidents_df.columns:
        # Filter for valid dates
        incs = incidents_df.dropna(subset=['opened_at']).copy()
        
        fig.add_trace(go.Scatter(
            x=incs['opened_at'],
            y=incs['number'], # Or some arbitrary Y, or categorical Y (like Assignment Group)
            mode='markers',
            name='Incidents',
            marker=dict(color='blue', size=8, opacity=0.7),
            text=incs['short_description'], # Hover text
            hovertemplate="<b>%{y}</b><br>%{x}<br>%{text}<extra></extra>"
        ))
    
    # 2. Trace: Problems (Red Lines)
    if not problems_df.empty and 'opened_at' in problems_df.columns and 'closed_at' in problems_df.columns:
        # Filter problems with valid start/end
        probs = problems_df.dropna(subset=['opened_at', 'closed_at']).copy()
        
        for idx, row in probs.iterrows():
            # For each problem, draw a line from start to end
            # We need a Y value. To visualize "fusion", we might want them on the same axis.
            # If Y is just category, it's hard to overlay lines.
            # OPTION: Use a constant Y or random Y, or try to match grouping.
            # For this 'Retro' view, usually time is X, and Y is just 'Entity' or just stacked.
            # Let's try plotting them on a separate 'y-axis' space or just overlaying them arbitrarily
            # if we use 'number' on Y-axis for incidents, it's categorical.
            
            # IMPROVEMENT: Let's just use a clear visual like 'y=1' for Incidents and 'y=2' for Problems?
            # Or better: "The Timeline Fusion" implies seeing them together.
            # Let's use 'Assignment Group' as Y axis? 
            # If too many groups, maybe just use a simple linear progression or just categorical 'y'.
            
            # Let's try categorical Y = "Problem Records" vs "Incidents" is too simple.
            # Let's stick to using 'Assignment Group' if available, else just simple ID indices.
            
            # COMPROMISE for simple visual:
            # Map Y to a categorical value (e.g. 'All Events') or just use the IDs (if they are messy, logic breaks).
            # Let's try to map them to 'Assignment Group' if shared?
            
            y_val = row.get('assignment_group', 'Unknown Group')
            
            fig.add_trace(go.Scatter(
                x=[row['opened_at'], row['closed_at']],
                y=[y_val, y_val],
                mode='lines+markers',
                name=f"Prob: {row.get('number', 'Unknown')}",
                line=dict(color='red', width=3),
                marker=dict(color='red', size=6),
                showlegend=False, # Too many legends
                hoverinfo='text',
                text=f"{row.get('number', 'N/A')}: {row.get('short_description', '')}"
            ))
            
    fig.update_layout(
        title="The Timeline Fusion: Incidents (Blue) vs Problems (Red)",
        xaxis_title="Time",
        yaxis_title="Assignment Group / Category",
        height=600,
        showlegend=True
    )
    
    return fig

def identify_zombie_problems(problems_df):
    """
    Identifies 'Zombie Problems': Entities that have >1 Problem Record in 12 months.
    Groups by location or cmdb_ci if available, otherwise heuristics from description.
    """
    zombies = []
    
    if problems_df.empty:
        return zombies
        
    # Heuristic: Check for duplicate Entity IDs in short_description (similar to Recursion Check for Incidents)
    # Re-use logic or implement specific 'Problem' logic.
    # The prompt says: "Group by location or cmdb_ci. List Entities that have >1 Problem Record in 12 months."
    
    # 1. Group by CMDB_CI / Location if available
    # Check columns
    groupable_cols = []
    if 'u_ci_type' in problems_df.columns: # Found in file preview
        groupable_cols.append('u_ci_type') 
    if 'location' in problems_df.columns:
        groupable_cols.append('location')
        
    # If explicit columns exist, use them. Else falls back to text extraction.
    
    # We will use a hybrid approach:
    # A. Count duplicates in 'location'
    if 'location' in problems_df.columns:
        loc_counts = problems_df['location'].value_counts()
        for loc, count in loc_counts.items():
            if count > 1 and loc: # Ignore empty
                # Get details
                subset = problems_df[problems_df['location'] == loc]
                zombies.append({
                    'Type': 'Location',
                    'Entity': loc,
                    'Count': count,
                    'Records': ", ".join(subset['number'].unique())
                })
                
    # B. Text Extraction from Description for 'Entity'
    # Reuse extraction logic?
    # Let's copy simple regex logic here to avoid circular imports or complex refactors right now.
    
    entity_map = defaultdict(list)
    
    for idx, row in problems_df.iterrows():
        desc = str(row.get('short_description', ''))
        num = row.get('number', 'UNK')
        
        # Regex for Asset-like things (e.g. "Server01", "10.0.0.1")
        # Reuse patterns
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        server_pattern = r'\b(?=.*\d)(?=.*[a-zA-Z])[a-zA-Z0-9-]{3,}\b'
        
        found = set(re.findall(ip_pattern, desc) + re.findall(server_pattern, desc))
        
        for ent in found:
            entity_map[ent].append(num)
            
    for ent, nums in entity_map.items():
        if len(set(nums)) > 1:
             zombies.append({
                'Type': 'Entity (Text)',
                'Entity': ent,
                'Count': len(set(nums)),
                'Records': ", ".join(sorted(list(set(nums))))
            })
            
    return pd.DataFrame(zombies)

def calculate_deflection_opportunity(incidents_df):
    """
    Filter Incidents by keywords ('password', 'reset', 'access').
    Calculate potential cost savings (e.g. $50 per ticket).
    """
    if incidents_df.empty:
        return 0, 0, pd.DataFrame()
        
    keywords = ['password', 'reset', 'access', 'login', 'account', 'unlock']
    pattern = '|'.join(keywords)
    
    # creating a copy to avoid SettingWithCopy warnings
    df = incidents_df.copy()
    
    # Filter
    # Check short_description
    mask = df['short_description'].fillna('').str.contains(pattern, case=False, regex=True)
    deflectable = df[mask]
    
    count = len(deflectable)
    estimated_cost_per_ticket = 50 # Assumption
    savings = count * estimated_cost_per_ticket
    
    return count, savings, deflectable[['number', 'short_description', 'opened_at']]
