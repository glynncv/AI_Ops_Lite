"""
Vendor Audit Module
==================
Analysis functions for evaluating vendor ticket quality and detecting
noise, misconfigurations, and topology gaps in incident data.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, Dict, Any


def ensure_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure opened_at and u_resolved columns are datetime type.
    Returns a copy of the dataframe with converted columns.
    """
    df = df.copy()

    if 'opened_at' in df.columns:
        df['opened_at'] = pd.to_datetime(df['opened_at'], errors='coerce')

    if 'u_resolved' in df.columns:
        df['u_resolved'] = pd.to_datetime(df['u_resolved'], errors='coerce')

    return df


def detect_burst_rate(df: pd.DataFrame) -> Tuple[int, pd.DataFrame]:
    """
    Metric 1: Machine Gun Detector (Burst Analysis)

    Groups data by opened_at (down to the exact second) and counts
    incidents per second to detect burst ticket creation.

    Returns:
        - max_burst_rate: Maximum tickets created in a single second
        - burst_table: DataFrame of timestamps where >1 ticket was created per second
    """
    if df.empty or 'opened_at' not in df.columns:
        return 0, pd.DataFrame()

    df = ensure_datetime_columns(df)

    # Filter out rows with invalid datetime
    valid_df = df[df['opened_at'].notna()].copy()

    if valid_df.empty:
        return 0, pd.DataFrame()

    # Truncate to second precision
    valid_df['opened_second'] = valid_df['opened_at'].dt.floor('s')

    # Count incidents per second
    burst_counts = valid_df.groupby('opened_second').size().reset_index(name='tickets_per_second')

    # Get max burst rate
    max_burst_rate = burst_counts['tickets_per_second'].max() if not burst_counts.empty else 0

    # Filter for bursts (>1 ticket per second)
    burst_table = burst_counts[burst_counts['tickets_per_second'] > 1].copy()
    burst_table = burst_table.sort_values('tickets_per_second', ascending=False)
    burst_table.columns = ['Timestamp', 'Tickets/Second']

    return int(max_burst_rate), burst_table


def detect_flapping(df: pd.DataFrame, threshold_seconds: int = 120) -> Tuple[float, int, int, pd.DataFrame]:
    """
    Metric 2: Flapping Detector (Instant Close)

    Calculates duration = u_resolved - opened_at and filters for tickets
    closed in less than threshold_seconds (default 2 minutes).

    Returns:
        - noise_ratio: Percentage of tickets closed under threshold
        - noise_count: Number of "noise" tickets
        - total_resolved: Total tickets with valid resolution times
        - flapping_df: DataFrame of flapping tickets
    """
    if df.empty or 'opened_at' not in df.columns or 'u_resolved' not in df.columns:
        return 0.0, 0, 0, pd.DataFrame()

    df = ensure_datetime_columns(df)

    # Filter for resolved tickets with valid dates
    resolved_df = df[df['opened_at'].notna() & df['u_resolved'].notna()].copy()

    if resolved_df.empty:
        return 0.0, 0, 0, pd.DataFrame()

    # Calculate duration in seconds
    resolved_df['duration_seconds'] = (resolved_df['u_resolved'] - resolved_df['opened_at']).dt.total_seconds()

    # Filter for positive durations only (u_resolved should be after opened_at)
    resolved_df = resolved_df[resolved_df['duration_seconds'] >= 0]

    total_resolved = len(resolved_df)

    if total_resolved == 0:
        return 0.0, 0, 0, pd.DataFrame()

    # Filter for flapping tickets (closed in under threshold)
    flapping_df = resolved_df[resolved_df['duration_seconds'] < threshold_seconds].copy()
    noise_count = len(flapping_df)

    # Calculate noise ratio
    noise_ratio = (noise_count / total_resolved) * 100

    # Prepare output dataframe
    display_cols = ['number', 'short_description', 'duration_seconds', 'opened_at', 'u_resolved']
    available_cols = [c for c in display_cols if c in flapping_df.columns]

    if available_cols:
        flapping_output = flapping_df[available_cols].copy()
        if 'duration_seconds' in flapping_output.columns:
            flapping_output['duration_seconds'] = flapping_output['duration_seconds'].round(0).astype(int)
            flapping_output = flapping_output.rename(columns={'duration_seconds': 'Duration (sec)'})
    else:
        flapping_output = pd.DataFrame()

    return noise_ratio, noise_count, total_resolved, flapping_output


def detect_topology_gaps(df: pd.DataFrame) -> Tuple[float, int, int, pd.DataFrame]:
    """
    Metric 3: Ghost Hunter (Topology Gap)

    Filters for rows where cmdb_ci is Empty/Null OR u_ci_type contains 'Other'/'Non-IT'.
    These are tickets that cannot be properly attributed to infrastructure.

    Returns:
        - ghost_rate: Percentage of tickets with no valid CI
        - ghost_count: Number of ghost tickets
        - total_count: Total tickets analyzed
        - ghost_df: DataFrame of ghost tickets
    """
    if df.empty:
        return 0.0, 0, 0, pd.DataFrame()

    total_count = len(df)

    # Build ghost filter conditions
    ghost_mask = pd.Series([False] * len(df), index=df.index)

    # Condition 1: cmdb_ci is empty/null
    if 'cmdb_ci' in df.columns:
        cmdb_empty = df['cmdb_ci'].isna() | (df['cmdb_ci'].astype(str).str.strip() == '')
        ghost_mask = ghost_mask | cmdb_empty

    # Condition 2: u_ci_type contains 'Other' or 'Non-IT' (case insensitive)
    if 'u_ci_type' in df.columns:
        ci_type_str = df['u_ci_type'].astype(str).str.lower()
        ci_type_invalid = ci_type_str.str.contains('other|non-it|non it|unknown', na=False, regex=True)
        ghost_mask = ghost_mask | ci_type_invalid

    # Apply filter
    ghost_df = df[ghost_mask].copy()
    ghost_count = len(ghost_df)

    # Calculate ghost rate
    ghost_rate = (ghost_count / total_count) * 100 if total_count > 0 else 0.0

    # Prepare output dataframe
    display_cols = ['number', 'short_description', 'cmdb_ci', 'u_ci_type', 'assignment_group']
    available_cols = [c for c in display_cols if c in ghost_df.columns]

    if available_cols:
        ghost_output = ghost_df[available_cols].copy()
        # Replace empty values for display
        for col in ['cmdb_ci', 'u_ci_type']:
            if col in ghost_output.columns:
                ghost_output[col] = ghost_output[col].fillna('[EMPTY]')
                ghost_output[col] = ghost_output[col].replace('', '[EMPTY]')
    else:
        ghost_output = pd.DataFrame()

    return ghost_rate, ghost_count, total_count, ghost_output


def detect_severity_mismatch(df: pd.DataFrame) -> Tuple[int, pd.DataFrame]:
    """
    Metric 4: Cry Wolf Detector (Severity Mismatch)

    Filters for rows where severity == '1 - High' AND priority == '4 - Low'.
    These are false alarms where severity doesn't match priority.

    Note: If 'severity' column doesn't exist, returns 0 count.

    Returns:
        - mismatch_count: Number of mismatched tickets
        - mismatch_df: DataFrame of mismatched tickets
    """
    if df.empty:
        return 0, pd.DataFrame()

    # Check if required columns exist
    has_severity = 'severity' in df.columns
    has_priority = 'priority' in df.columns

    if not has_priority:
        return 0, pd.DataFrame()

    if not has_severity:
        # Severity column doesn't exist - return empty result
        return 0, pd.DataFrame()

    # Normalize column values for comparison
    severity_str = df['severity'].astype(str).str.lower().str.strip()
    priority_str = df['priority'].astype(str).str.lower().str.strip()

    # Look for high severity with low priority
    # Pattern: severity contains '1' or 'high' AND priority contains '4' or 'low'
    high_severity = severity_str.str.contains('1|high|critical', na=False, regex=True)
    low_priority = priority_str.str.contains('4|5|low|planning', na=False, regex=True)

    mismatch_mask = high_severity & low_priority

    mismatch_df = df[mismatch_mask].copy()
    mismatch_count = len(mismatch_df)

    # Prepare output dataframe
    display_cols = ['number', 'short_description', 'severity', 'priority', 'assignment_group']
    available_cols = [c for c in display_cols if c in mismatch_df.columns]

    if available_cols:
        mismatch_output = mismatch_df[available_cols].copy()
    else:
        mismatch_output = pd.DataFrame()

    return mismatch_count, mismatch_output


def create_human_vs_machine_chart(df: pd.DataFrame, threshold_seconds: int = 120) -> go.Figure:
    """
    Visual: Bar Chart showing Human vs Machine

    Compares tickets closed in < 2 mins (Machine/Auto) vs > 2 mins (Human).

    Returns:
        Plotly Figure object
    """
    if df.empty or 'opened_at' not in df.columns or 'u_resolved' not in df.columns:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for Human vs Machine analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig

    df = ensure_datetime_columns(df)

    # Filter for resolved tickets with valid dates
    resolved_df = df[df['opened_at'].notna() & df['u_resolved'].notna()].copy()

    if resolved_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No resolved tickets with valid timestamps",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig

    # Calculate duration in seconds
    resolved_df['duration_seconds'] = (resolved_df['u_resolved'] - resolved_df['opened_at']).dt.total_seconds()

    # Filter for positive durations
    resolved_df = resolved_df[resolved_df['duration_seconds'] >= 0]

    if resolved_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No valid resolution times found",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig

    # Categorize
    machine_count = len(resolved_df[resolved_df['duration_seconds'] < threshold_seconds])
    human_count = len(resolved_df[resolved_df['duration_seconds'] >= threshold_seconds])

    # Create bar chart
    categories = ['Machine (< 2 min)', 'Human (>= 2 min)']
    counts = [machine_count, human_count]
    colors = ['#FF6B6B', '#4ECDC4']  # Red for machine, Teal for human

    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=counts,
            marker_color=colors,
            text=counts,
            textposition='auto',
            textfont=dict(size=18, color='white')
        )
    ])

    fig.update_layout(
        title=dict(
            text='Human vs Machine: Ticket Resolution Analysis',
            font=dict(size=20)
        ),
        xaxis_title='Resolution Type',
        yaxis_title='Number of Tickets',
        showlegend=False,
        height=400,
        template='plotly_white'
    )

    return fig


def get_audit_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive audit summary of all metrics.

    Returns a dictionary with all key metrics for the vendor audit.
    """
    summary = {
        'total_tickets': len(df) if not df.empty else 0,
        'burst_rate': 0,
        'noise_ratio': 0.0,
        'ghost_rate': 0.0,
        'severity_mismatch': 0,
        'has_severity_column': 'severity' in df.columns if not df.empty else False
    }

    if df.empty:
        return summary

    # Metric 1: Burst Rate
    max_burst, _ = detect_burst_rate(df)
    summary['burst_rate'] = max_burst

    # Metric 2: Noise Ratio
    noise_ratio, _, _, _ = detect_flapping(df)
    summary['noise_ratio'] = noise_ratio

    # Metric 3: Ghost Rate
    ghost_rate, _, _, _ = detect_topology_gaps(df)
    summary['ghost_rate'] = ghost_rate

    # Metric 4: Severity Mismatch
    mismatch_count, _ = detect_severity_mismatch(df)
    summary['severity_mismatch'] = mismatch_count

    return summary
