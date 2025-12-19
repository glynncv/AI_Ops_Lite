import streamlit as st
import pandas as pd
import json
import os
import snow_connector
from dotenv import load_dotenv
from data_processor import process_snow_data, fetch_changes

# Load Environment Variables
load_dotenv()
# Updated imports including new attributes
from analysis import (
    perform_clustering, 
    check_historical_recursion, 
    check_historical_recursion, 
    find_suspect_changes,
    detect_volume_spike, 
    cluster_open_incidents, 
    correlate_cluster_causes
)
from utils import generate_communication_template
from data_loader import DataLoader
from retro_analysis import create_timeline_fusion_chart, identify_zombie_problems, calculate_deflection_opportunity

def main():
    st.set_page_config(page_title="AI_Ops Flight Deck", layout="wide")
    st.title("AIOps Lite: Flight Deck")
    
    # Sidebar Configuration
    st.sidebar.title("Data Source Config")
    data_mode = st.sidebar.selectbox("Select Data Source", ["Live API (Mock)", "Live API (Real)", "Offline Data"])

    st.sidebar.markdown("---")
    # Flash Report Trigger
    if st.sidebar.button("Generate Flash Report"):
        st.session_state['show_flash_report'] = True
    else:
        if 'show_flash_report' not in st.session_state:
            st.session_state['show_flash_report'] = False
    
    # Initialize DataFrames
    df_cleaned = pd.DataFrame()
    changes_df = pd.DataFrame()
    problems_df = pd.DataFrame()

    # --- Phase 1: Data Loading (Common) ---
    st.sidebar.header("Status")
    
    if data_mode == 'Live API (Mock)':
        st.sidebar.info("Mode: Live API (Mock JSON)")
        
        # Load Mock Incidents
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'input', 'incidents.json')
        try:
            with open(data_path, 'r') as f:
                incidents_data = json.load(f)
            if incidents_data:
                df_cleaned = process_snow_data(incidents_data)
                st.sidebar.success(f"Loaded {len(df_cleaned)} Incidents")
        except Exception as e:
            st.sidebar.error(f"Error loading mock incidents: {e}")
            
        # Load Mock Changes
        try:
            changes_df = fetch_changes()
            if not changes_df.empty:
                st.sidebar.success(f"Loaded {len(changes_df)} Changes")
        except Exception as e:
            st.sidebar.error(f"Error loading changes: {e}")

        # Load Mock Problems (NEW)
        data_path_prb = os.path.join(os.path.dirname(__file__), 'data', 'input', 'problems.json')
        try:
            with open(data_path_prb, 'r') as f:
                problems_data = json.load(f)
            
            if problems_data:
                # Convert to DataFrame
                problems_df = pd.DataFrame(problems_data)
                
                # Standardize Dates
                date_cols = ['opened_at', 'closed_at']
                for col in date_cols:
                    if col in problems_df.columns:
                        problems_df[col] = pd.to_datetime(problems_df[col], errors='coerce')
                
                st.sidebar.success(f"Loaded {len(problems_df)} Problems (Mock)")
        except Exception as e:
             # It's okay if file doesn't exist, just don't crash
             pass

    elif data_mode == 'Live API (Real)':
        st.sidebar.info("Mode: Live ServiceNow API")
        
        # Credentials Input
        with st.sidebar.expander("ServiceNow Credentials", expanded=False):
            default_url = os.getenv("SNOW_INSTANCE_URL", "https://phinia.service-now.com/")
            default_user = os.getenv("SNOW_USERNAME", "")
            default_pass = os.getenv("SNOW_PASSWORD", "")
            
            instance_url = st.text_input("Instance URL", default_url)
            username = st.text_input("Username", default_user)
            password = st.text_input("Password", default_pass, type="password")
            
            connect_btn = st.button("Reconnect / Fetch Manually")

        # Auto-connect logic: If env vars exist and we haven't loaded real data yet, or if button is clicked
        should_connect = False
        
        # Check if we have credentials (either from env default or manual input)
        has_creds = instance_url and username and password
        
        # Check if we need to auto-connect (first run in this mode)
        # We use a session state flag to track if we've tried auto-connecting to avoid loops or spam
        if 'auto_connected' not in st.session_state:
            st.session_state['auto_connected'] = False
            
        if has_creds and not st.session_state['auto_connected']:
             should_connect = True
        
        if connect_btn and has_creds:
             should_connect = True

        if should_connect:
            try:
                with st.spinner("Auto-Connecting to ServiceNow..."):
                    client = snow_connector.ServiceNowClient(instance_url, username, password)
                    
                    # 1. Incidents
                    inc_data = snow_connector.get_snow_data('incident', client)
                    if inc_data:
                        process_snow_data(inc_data) # Just verifying processing
                        st.sidebar.success(f"Fetched {len(inc_data)} Incidents")
                        st.session_state['inc_df'] = process_snow_data(inc_data)
                    
                    # 2. Problems
                    prb_data = snow_connector.get_snow_data('problem', client)
                    if prb_data:
                        problems_df = pd.DataFrame(prb_data)
                        # Normalize dates
                        for col in ['opened_at', 'closed_at']:
                            if col in problems_df.columns:
                                problems_df[col] = pd.to_datetime(problems_df[col], errors='coerce')
                        st.session_state['prb_df'] = problems_df
                        st.sidebar.success(f"Fetched {len(problems_df)} Problems")

                    # 3. Changes
                    chg_data = snow_connector.get_snow_data('change_request', client)
                    if chg_data:
                        changes_df = pd.DataFrame(chg_data)
                        # Normalize dates
                        if 'closed_at' in changes_df.columns:
                            changes_df['closed_at'] = pd.to_datetime(changes_df['closed_at'], errors='coerce')
                        st.session_state['chg_df'] = changes_df
                        st.sidebar.success(f"Fetched {len(changes_df)} Changes")
                    
                    st.session_state['data_loaded'] = True
                    st.session_state['auto_connected'] = True
                    st.session_state['data_mode'] = 'real'
                    
                    # Rerun to update main view
                    st.rerun()

            except Exception as e:
                st.sidebar.error(f"Connection Failed: {e}")
                st.session_state['auto_connected'] = True # Stop trying if failed

    elif data_mode == 'Offline Data':
        st.sidebar.info("Mode: Offline Data (CSV)")
        loader = DataLoader()
        
        with st.sidebar.expander("Upload Overrides"):
            up_inc = st.file_uploader("Upload Incidents (CSV)", type=['csv'])
            up_chg = st.file_uploader("Upload Changes (CSV)", type=['csv'])
            up_prob = st.file_uploader("Upload Problems (CSV)", type=['csv'])

        df_cleaned = loader.load_incidents(up_inc)
        changes_df = loader.load_changes(up_chg)
        problems_df = loader.load_problems(up_prob)
        
        if not df_cleaned.empty:
             st.sidebar.success(f"Loaded {len(df_cleaned)} Incidents")
        if not changes_df.empty:
             st.sidebar.success(f"Loaded {len(changes_df)} Changes")

    # --- Tabs Layout ---
    tab_risks, tab_dive = st.tabs(["🔴 Current Risks", "🔍 Investigation Deck"])

    # ==========================
    # TAB 1: Current Risks
    # ==========================
    with tab_risks:
        st.header("Real-Time Risk Monitor")
        
        col1, col2, col3 = st.columns(3)
        
        # 1. Spike Detector
        with col1:
            st.subheader("Volume Monitor")
            if not df_cleaned.empty:
                is_spike, daily_counts = detect_volume_spike(df_cleaned)
                if is_spike:
                     st.metric(label="Daily Volume", value="Spike Detected", delta="Alert", delta_color="inverse")
                     st.error("Abnormal High Volume Detected Today!")
                else:
                     st.metric(label="Daily Volume", value="Normal", delta="Stable")
                
                # Small chart
                if not daily_counts.empty:
                    st.line_chart(daily_counts, height=150)
            else:
                st.info("No data for spike detection.")
                
        # 2. Cluster Engine (Open Incidents)
        with col2:
            st.subheader("Hidden Clusters (Open)")
            open_clusters = pd.DataFrame()
            if not df_cleaned.empty:
                open_clusters = cluster_open_incidents(df_cleaned)
                
                if not open_clusters.empty and 'Cluster_ID' in open_clusters.columns:
                    valid_clusters = open_clusters[open_clusters['Cluster_ID'] != -1]
                    if not valid_clusters.empty:
                         cluster_counts = valid_clusters['Cluster_ID'].value_counts()
                         st.dataframe(cluster_counts, height=150)
                         st.warning(f"{len(valid_clusters)} open incidents in {len(cluster_counts)} clusters.")
                    else:
                         st.success("No clustered open incidents.")
                else:
                    st.success("No open incidents or clusters found.")
            else:
                st.info("Waiting for data...")

        # 3. Change Correlator
        with col3:
            st.subheader("Suspect Root Causes")
            if not open_clusters.empty and not changes_df.empty:
                matches = correlate_cluster_causes(open_clusters, changes_df)
                if matches:
                    st.error(f"Found {len(matches)} Suspect Changes!")
                    for m in matches:
                        st.markdown(f"**Cluster {m['Cluster_ID']}** linked to **{m['Suspect_Change']}**")
                        st.caption(f"Reason: {m['Matched_Keywords']}")
                else:
                    st.success("No correlation with recent changes.")
            else:
                st.info("Need Clusters + Change Data.")

        # Detailed View for Current Risks
        st.markdown("---")
        if not open_clusters.empty and 'Cluster_ID' in open_clusters.columns:
             valid = open_clusters[open_clusters['Cluster_ID'] != -1]
             if not valid.empty:
                 st.subheader("Active Clusters Detail")
                 st.dataframe(valid[['Cluster_ID', 'number', 'short_description', 'assignment_group', 'state']])

    # ==========================
    # TAB 2: Investigation Deck (Deep Dive)
    # ==========================
    with tab_dive:
        st.header("Deep Dive Analysis")

        # Phase 2: Clustering Analysis (All)
        with st.expander("Analysis: Full Clustering (All States)", expanded=True):
            if not df_cleaned.empty:
                try:
                    df_all_clustered = perform_clustering(df_cleaned)
                    if 'Cluster_ID' in df_all_clustered.columns:
                        clustered = df_all_clustered[df_all_clustered['Cluster_ID'] != -1]
                        if not clustered.empty:
                            st.warning('⚠️ Potential Duplicate Patterns (All Time)')
                            st.dataframe(clustered[['Cluster_ID', 'number', 'short_description']].sort_values('Cluster_ID'))
                        else:
                            st.success("No patterns in full history.")
                except Exception as e:
                    st.error(f"Clustering Error: {e}")

        # Phase 3: Historical Recursion
        with st.expander("Analysis: Repeat Offenders", expanded=True):
            if not df_cleaned.empty:
                try:
                    repeat_offenders = check_historical_recursion(df_cleaned)
                    if repeat_offenders:
                        st.warning("🔥 Recurring Assets/Entities Detected")
                        st.table(pd.DataFrame(repeat_offenders))
                    else:
                        st.success("No recurring entities detected.")
                except Exception as e:
                    st.error(f"Recursion Check Error: {e}")

        # Phase 4: Suspect Change Analysis (Row-wise)
        with st.expander("Analysis: Incident-Change Correlation (Row-by-Row)", expanded=False):
            if not df_cleaned.empty and not changes_df.empty:
                try:
                    st.info("Correlating 48h lookback...")
                    df_cleaned['suspect_root_cause'] = df_cleaned.apply(
                        lambda row: find_suspect_changes(row, changes_df, lookback_hours=48), 
                        axis=1
                    )
                    suspects = df_cleaned[df_cleaned['suspect_root_cause'].map(len) > 0]
                    if not suspects.empty:
                        st.warning(f"Found {len(suspects)} incidents with potential change correlations.")
                        # Format for display
                        display = suspects[['number', 'short_description', 'suspect_root_cause']].copy()
                        display['suspect_root_cause'] = display['suspect_root_cause'].apply(lambda x: str(x))
                        st.dataframe(display)
                    else:
                        st.success("No row-wise correlations found.")
                except Exception as e:
                    st.error(f"Correlation Error: {e}")

        # Draft Comms
        st.divider()
        st.header("Communication Assistant")
        if not df_cleaned.empty:
            mi_list = df_cleaned['number'].unique()
            selected_mi = st.selectbox("Select Incident for Comm Draft", mi_list)
            if selected_mi:
                row = df_cleaned[df_cleaned['number'] == selected_mi].iloc[0]
                impact = st.text_area("Impact Details")
                if st.button("Generate Template"):
                    tmpl = generate_communication_template(
                        selected_mi, row.get('short_description'), row.get('state'), row.get('assignment_group'), impact
                    )
                    st.code(tmpl)

    # Flash Report Overlay
    if st.session_state.get('show_flash_report'):
        st.divider()
        st.header("⚡ Executive Flash Report")
        if df_cleaned.empty:
            st.error("No Data")
        else:
            # Simple metrics for now
            total = len(df_cleaned)
            open_cnt = len(df_cleaned[~df_cleaned['state'].isin(['Closed', 'Resolved'])])
            st.markdown(f"""
            **Status**: {'🔴 High Risk' if open_cnt > 5 else '🟢 Stable'}
            - **Total Incidents**: {total}
            - **Open Incidents**: {open_cnt}
            """)
            
            # Generating dynamic flash report content
            risk = "High" if open_cnt > 5 else "Low"
            
            active_clusters_val = 0
            if not df_cleaned.empty:
                oc = cluster_open_incidents(df_cleaned)
                if not oc.empty and 'Cluster_ID' in oc.columns:
                     active_clusters_val = oc['Cluster_ID'].nunique()

            chronic_sites_val = []
            if not df_cleaned.empty:
                ro = check_historical_recursion(df_cleaned)
                if ro:
                     chronic_sites_val = [x['Entity'] for x in ro[:3]]
            
            deflect_val = 0
            if not df_cleaned.empty:
                 d_count, _, _ = calculate_deflection_opportunity(df_cleaned)
                 deflect_val = d_count

            template = f"""
EXECUTIVE FLASH REPORT
----------------------
⚠️ Operational Risk: {risk}
Active Clusters: {active_clusters_val}
Chronic Sites: {', '.join(chronic_sites_val) if chronic_sites_val else 'None'}
Deflection Potential: {deflect_val} tickets
            """
            # 4. Display this in a st.code block
            st.code(template, language='text')

    # --- Phase 5: Retro Audit (Back to the Future) ---
    st.header("Phase 5: Retro Audit (Back to the Future)")
    
    tab_audit, tab_zombies, tab_deflection = st.tabs(["The Timeline Fusion", "Zombie Problems", "Deflection Opportunity"])
    
    with tab_audit:
        st.subheader("The Timeline Fusion")
        st.write("Visualizing the relationship between Incidents (Blue) and Problem Records (Red).")
        
        if not df_cleaned.empty and not problems_df.empty:
            fig = create_timeline_fusion_chart(df_cleaned, problems_df)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data for Timeline Fusion. Ensure both Incidents and Problems are loaded.")

    with tab_zombies:
        st.subheader("Recursion Table: Zombie Problems")
        st.write("Entities with >1 Problem Record in the last 12 months.")
        
        if not problems_df.empty:
            zombies = identify_zombie_problems(problems_df)
            if not zombies.empty:
                st.warning(f"Found {len(zombies)} Zombie Entities!")
                st.dataframe(zombies, use_container_width=True)
            else:
                st.success("No Zombie Problems detected (Entities with multiple Problem records).")
        else:
            st.warning("No Problem data loaded.")
            
    with tab_deflection:
        st.subheader("Deflection Opportunity")
        st.write("Potential cost savings from automating keyword-matched incidents.")
        
        if not df_cleaned.empty:
            deflect_count, savings, deflect_df = calculate_deflection_opportunity(df_cleaned)
            
            col1, col2 = st.columns(2)
            col1.metric("Deflectable Tickets", deflect_count)
            col2.metric("Potential Savings", f"${savings:,}")
            
            if not deflect_df.empty:
                with st.expander("View Deflectable Candidates"):
                    st.dataframe(deflect_df)
        else:
            st.warning("No Incident data loaded.")

if __name__ == "__main__":
    main()

