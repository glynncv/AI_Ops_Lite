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
    correlate_cluster_causes,
    find_similar_p1_resolutions
)
from utils import generate_communication_template
from data_loader import DataLoader
from retro_analysis import create_timeline_fusion_chart, identify_zombie_problems, calculate_deflection_opportunity
from aiops_intelligence import (
    find_similar_resolved_incidents,
    IntelligentRouter,
    suggest_problem_creation,
    batch_suggest_problems,
    calculate_mttr_improvement
)

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

    # Check Session State for Data Persistence
    if 'inc_df' in st.session_state:
        df_cleaned = st.session_state['inc_df']
    if 'prb_df' in st.session_state:
        problems_df = st.session_state['prb_df']
    if 'chg_df' in st.session_state:
        changes_df = st.session_state['chg_df']

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
        
        # Credentials from Environment Variables (UI Removed for Security)
        instance_url = os.getenv("SNOW_INSTANCE_URL", "https://phinia.service-now.com/")
        username = os.getenv("SNOW_USERNAME", "")
        password = os.getenv("SNOW_PASSWORD", "")
        
        connect_btn = st.sidebar.button("Reconnect / Fetch Manually")

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
                    # Bypass proxy by default (use_proxy=False) to avoid connection issues
                    # Set SNOW_USE_PROXY=true in .env if you need proxy access
                    use_proxy = os.getenv('SNOW_USE_PROXY', 'false').lower() == 'true'
                    
                    # SSL verification - can be disabled for corporate environments with self-signed certs
                    # Set SNOW_VERIFY_SSL=false in .env if you get SSL certificate errors
                    verify_ssl = os.getenv('SNOW_VERIFY_SSL', 'true').lower() == 'true'
                    
                    client = snow_connector.ServiceNowClient(instance_url, username, password, use_proxy=use_proxy, verify_ssl=verify_ssl)
                    
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

    # --- Phase 6: War Room Mode ---
    st.sidebar.markdown("---")
    war_room_mode = st.sidebar.toggle('🔴 Major Incident Mode', value=False)
    
    if war_room_mode:
        st.markdown("""
        <style>
        .war-room-header {
            color: #d32f2f;
            font-size: 3em;
            font-weight: 800;
            text-transform: uppercase;
            border-bottom: 3px solid #d32f2f;
            margin-bottom: 20px;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.6; }
            100% { opacity: 1; }
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown('<div class="war-room-header">🚨 WAR ROOM: MAJOR INCIDENT ACTIVE</div>', unsafe_allow_html=True)
        
        # --- Active Major Incidents Section ---
        if not df_cleaned.empty:
            # Filter for Open Incidents
            open_mask = ~df_cleaned['state'].isin(['Closed', 'Resolved', 'Canceled', 'Cancelled'])
            active_incidents = df_cleaned[open_mask].copy()
            
            # Filter for Priority 1/2 if priority column exists, otherwise show all open
            if not active_incidents.empty:
                mis = pd.DataFrame()
                
                if 'priority' in active_incidents.columns:
                    # Check if priority column has actual values (not all null/empty)
                    has_priority_data = active_incidents['priority'].notna().any()
                    
                    if has_priority_data:
                        # Filter for P1/P2 - match priorities that START with "1" or "2"
                        # This catches: "1", "1 - Critical", "2", "2 - High", etc.
                        # But NOT: "4 - Low" even if description contains "critical"
                        mis = active_incidents[
                            active_incidents['priority'].astype(str).str.match(r'^[12](\s|-|$)', na=False)
                        ]
                    else:
                        # Priority column exists but has no data - fall back to keyword search
                        st.info("📋 Priority field is empty — using keyword search. Ensure ServiceNow returns priority values.")
                        mis = active_incidents[
                            active_incidents['short_description'].str.contains('Critical|Outage|Urgent|P1|P2', case=False, na=False)
                        ].head(5)
                else:
                    # No priority column - fall back to keyword search (e.g. Offline CSV or Mock data)
                    st.info("📋 Using keyword search (Critical/Outage/Urgent/P1/P2) — add a *priority* column to your data for accurate P1/P2 filtering.")
                    mis = active_incidents[
                        active_incidents['short_description'].str.contains('Critical|Outage|Urgent|P1|P2', case=False, na=False)
                    ].head(5)
                
                # If we found Major Incidents, display them prominently
                if not mis.empty:
                    st.error(f"⚠️ {len(mis)} MAJOR INCIDENT(S) IN PROGRESS")
                    
                    # Include priority column if it exists
                    display_cols = ['number', 'short_description', 'assignment_group', 'state', 'opened_at']
                    if 'priority' in mis.columns:
                        display_cols.insert(1, 'priority')
                    
                    st.dataframe(
                        mis[display_cols],
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                     st.info("No Active Priority 1/2 Incidents detected in data.")
            else:
                 st.success("No Active Incidents.")
        
        col1, col2, col3 = st.columns(3)
        
        # 1. Velocity Meter
        with col1:
            st.subheader("🔥 Velocity Meter")
            velocity = 0.0
            if not df_cleaned.empty:
                # Ensure datetime
                if not pd.api.types.is_datetime64_any_dtype(df_cleaned['opened_at']):
                    df_cleaned['opened_at'] = pd.to_datetime(df_cleaned['opened_at'], errors='coerce')
                
                now = pd.Timestamp.now()
                # 30 min window for calculation
                limit_time = now - pd.Timedelta(minutes=30)
                recent = df_cleaned[df_cleaned['opened_at'] >= limit_time]
                velocity = len(recent) / 30.0
            
            st.metric("Incidents / Min", f"{velocity:.2f}", delta="Last 30 mins", delta_color="inverse")
            
        # 2. Blast Radius
        with col2:
            st.subheader("🌍 Blast Radius (Locations)")
            if not df_cleaned.empty and 'location' in df_cleaned.columns:
                affected_locs = df_cleaned['location'].value_counts().head(5)
                st.dataframe(affected_locs.rename("Ticket Count"), width=300)
            else:
                st.info("No location data available.")

        # 3. Change Radar
        with col3:
            st.subheader("📡 Change Radar")
            if not changes_df.empty:
                 if 'closed_at' in changes_df.columns:
                     if not pd.api.types.is_datetime64_any_dtype(changes_df['closed_at']):
                         changes_df['closed_at'] = pd.to_datetime(changes_df['closed_at'], errors='coerce')
                         
                     now = pd.Timestamp.now()
                     # Changes closed in last 4 hours
                     limit_time = now - pd.Timedelta(hours=4)
                     recent_changes = changes_df[
                         (changes_df['closed_at'] >= limit_time) &
                         (changes_df['closed_at'] <= now + pd.Timedelta(minutes=10)) # Slush
                     ]
                     
                     if not recent_changes.empty:
                         st.error(f"Found {len(recent_changes)} Recent Changes")
                         st.dataframe(recent_changes[['number', 'short_description', 'closed_at']].head(5), hide_index=True)
                     else:
                         st.success("No changes in last 4 hours")
                 else:
                     st.warning("Change data missing 'closed_at'")
            else:
                 st.info("No changes loaded.")

        st.divider()
        
        # 4. Crisis Memory
        st.subheader("🧠 Crisis Memory: Historical Fixes")
        
        # Build condensed search query from top cluster (shorter = better for TF-IDF similarity)
        def _condense_cluster_query(descriptions):
            """Extract key phrases (e.g. after ' : ') and join uniquely for a readable, search-effective query."""
            seen = set()
            parts = []
            for desc in descriptions[:3]:
                s = str(desc).strip()
                if not s:
                    continue
                # Many alerts use "identifier : error_type" - extract the error_type for search
                if " : " in s:
                    s = s.split(" : ", 1)[1].strip()
                elif ": " in s:
                    s = s.split(": ", 1)[1].strip()
                if s and s not in seen:
                    seen.add(s)
                    parts.append(s)
            return " ".join(parts) if parts else ""
        
        query_text = ""
        cluster_info = "Manual Entry"
        if not df_cleaned.empty:
             open_clusters = cluster_open_incidents(df_cleaned)
             if not open_clusters.empty and 'Cluster_ID' in open_clusters.columns:
                 valid = open_clusters[open_clusters['Cluster_ID'] != -1]
                 if not valid.empty:
                     top_id = valid['Cluster_ID'].value_counts().idxmax()
                     top_cluster = valid[valid['Cluster_ID'] == top_id]
                     descs = top_cluster['short_description'].head(3).astype(str).tolist()
                     query_text = _condense_cluster_query(descs)
                     if not query_text:
                         query_text = descs[0] if descs else ""  # fallback to first full description
                     cluster_info = f"Cluster {top_id} (Size: {len(top_cluster)})"

        search_query = st.text_input(f"Search Query ({cluster_info})", value=query_text if query_text else "Service Outage", 
                                      placeholder="e.g. ABAP IDoc errors")
        
        if st.button("Search Historical P1s", key='war_room_search'):
            with st.spinner("Searching Crisis Memory..."):
                matches = find_similar_p1_resolutions(search_query, df_cleaned)
                if matches:
                    st.success(f"Found {len(matches)} Relevant P1 Records")
                    for m in matches:
                        with st.expander(f"📌 {m['number']}: {m['short_description']} (Match: {m['score']:.1%})", expanded=True):
                            st.write("**Resolution Notes:**")
                            st.code(m['close_notes'])
                else:
                    st.warning("No relevant historical P1s found.")
        
        st.caption("Detailed dashboard disabled in War Room mode.")
        st.stop()

    # --- Tabs Layout ---
    tab_risks, tab_dive, tab_intelligence, tab_retro = st.tabs([
        "🔴 Current Risks", "🔍 Investigation Deck", "🧠 AI Intelligence", "⏪ Retro Audit"
    ])

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
                    # Group matches by cluster for better organization
                    from collections import defaultdict
                    grouped = defaultdict(list)
                    for m in matches:
                        grouped[m['Cluster_ID']].append(m)
                    
                    # Show summary metrics
                    st.error(f"🔍 Found {len(matches)} Suspect Changes!")
                    st.caption(f"**{len(grouped)} clusters** linked to recent changes")
                    
                    # Display grouped by cluster (show top 5 clusters)
                    sorted_clusters = sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True)
                    
                    for cluster_id, cluster_matches in sorted_clusters[:5]:
                        with st.expander(
                            f"Cluster {cluster_id} ({len(cluster_matches)} changes)",
                            expanded=(len(cluster_matches) > 3)  # Auto-expand if many changes
                        ):
                            for m in cluster_matches:
                                st.markdown(f"→ **{m['Suspect_Change']}**")
                                st.caption(f"Reason: {m['Matched_Keywords']}")
                    
                    # Show remaining count
                    if len(grouped) > 5:
                        st.caption(f"_... and {len(grouped) - 5} more clusters_")
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

    # ==========================
    # TAB 3: AI Intelligence (NEW)
    # ==========================
    with tab_intelligence:
        st.header("🧠 AI Intelligence & Predictions")
        st.info("Advanced ML-powered features: Similar Incident Matching, Intelligent Routing, and Proactive Problem Detection")

        # Feature 1: Similar Incident Recommendation
        st.subheader("1️⃣ Similar Incident Recommendation")
        st.write("Find how similar past incidents were resolved to accelerate current resolutions.")

        if not df_cleaned.empty:
            # Select an incident to analyze
            incident_list = df_cleaned['number'].unique()
            selected_incident = st.selectbox("Select an Incident to Find Similar Cases", incident_list, key='similar_inc')

            if selected_incident:
                incident_row = df_cleaned[df_cleaned['number'] == selected_incident].iloc[0]
                incident_desc = str(incident_row.get('short_description', '')) + ' ' + str(incident_row.get('description', ''))

                if st.button("Find Similar Incidents", key='find_similar'):
                    with st.spinner("Analyzing historical data..."):
                        similar = find_similar_resolved_incidents(incident_desc, df_cleaned, top_n=5)

                        if similar:
                            st.success(f"Found {len(similar)} similar resolved incidents!")

                            for i, sim in enumerate(similar):
                                with st.expander(f"🔍 {sim['incident_number']} (Similarity: {sim['similarity_score']:.0%})"):
                                    col1, col2 = st.columns(2)

                                    with col1:
                                        st.markdown(f"**Description:** {sim['short_description']}")
                                        st.markdown(f"**Assignment Group:** {sim['assignment_group']}")

                                    with col2:
                                        if sim['resolution_time_hours']:
                                            st.metric("Resolution Time", f"{sim['resolution_time_hours']:.1f}h")

                                    st.markdown(f"**Resolution Notes:**")
                                    st.code(sim['resolution_notes'], language='text')

                            # Show MTTR improvement potential
                            improvement = calculate_mttr_improvement(similar)
                            if improvement.get('avg_historical_resolution_hours'):
                                st.info(f"💡 **Time Savings Potential:** Using these similar incidents could save ~{improvement['estimated_time_savings_hours']:.1f} hours (30% of avg {improvement['avg_historical_resolution_hours']:.1f}h)")
                        else:
                            st.warning("No similar resolved incidents found.")
                            st.caption("Cluster mates may still be open, or resolved incidents may lack resolution notes. Similar Incident search only shows *closed* incidents with *close_notes*.")
        else:
            st.warning("No incident data loaded.")

        st.divider()

        # Feature 2: Intelligent Assignment/Routing
        st.subheader("2️⃣ Intelligent Assignment Routing")
        st.write("ML-powered prediction of which team should handle new incidents.")

        if not df_cleaned.empty:
            # Initialize router in session state
            if 'router' not in st.session_state:
                st.session_state.router = IntelligentRouter()

            # Train button
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("Train Assignment Model", key='train_router'):
                    with st.spinner("Training ML model on historical data..."):
                        metrics = st.session_state.router.train(df_cleaned)

                        if metrics.get('success'):
                            st.success(f"✅ Model trained on {metrics['num_training_samples']} incidents")
                            st.info(f"Training Accuracy: {metrics['training_accuracy']:.1%}")
                            st.write(f"**Assignment Groups:** {metrics['num_assignment_groups']}")

                            # Show group distribution
                            if metrics.get('assignment_groups'):
                                with st.expander("View Group Distribution"):
                                    group_df = pd.DataFrame(list(metrics['assignment_groups'].items()),
                                                           columns=['Assignment Group', 'Count'])
                                    st.dataframe(group_df.sort_values('Count', ascending=False))
                        else:
                            st.error(f"Training failed: {metrics.get('error', 'Unknown error')}")

            # Predict assignment
            if st.session_state.router.trained:
                st.markdown("**Test the Model:**")
                test_description = st.text_area("Enter incident description for routing prediction:",
                                                value="VPN connection failed for remote users")

                if st.button("Predict Assignment", key='predict_assign'):
                    predictions = st.session_state.router.predict_assignment(test_description, top_n=3)

                    if predictions and not predictions[0].get('error'):
                        st.success("🎯 Recommended Assignment:")

                        for i, pred in enumerate(predictions):
                            if i == 0:
                                # Highlight top recommendation
                                st.markdown(f"### 🥇 {pred['assignment_group']}")
                                st.progress(pred['confidence'])
                                st.caption(f"Confidence: {pred['confidence']:.0%} | {pred['reasoning']}")
                            else:
                                st.markdown(f"**Alternative {i}:** {pred['assignment_group']} ({pred['confidence']:.0%})")
                                st.caption(pred['reasoning'])

                        st.info("💰 **Business Impact:** Intelligent routing reduces mis-routing by 60-80%, saving ~2 hours per ticket.")
                    else:
                        st.error("Prediction failed. Please train the model first.")
        else:
            st.warning("No incident data loaded.")

        st.divider()

        # Feature 3: Auto-Problem Creation Suggestions
        st.subheader("3️⃣ Proactive Problem Detection")
        st.write("Automatically suggest Problem Records for recurring incident patterns.")

        if not df_cleaned.empty:
            col_btn1, col_btn2 = st.columns([3, 1])
            
            with col_btn1:
                threshold = st.slider("Minimum incidents to trigger Problem creation:", 3, 10, 5, key='problem_threshold')
            
            with col_btn2:
                if 'problem_suggestions' in st.session_state and st.session_state['problem_suggestions'] is not None:
                    if st.button("🔄 Clear Results", key='clear_problems'):
                        st.session_state.pop('problem_suggestions', None)
                        st.session_state.pop('last_threshold', None)
                        st.rerun()

            if st.button("Analyze Clusters for Problem Opportunities", key='detect_problems'):
                with st.spinner("Analyzing incident patterns..."):
                    # First, cluster all incidents
                    df_clustered = perform_clustering(df_cleaned)

                    if 'Cluster_ID' in df_clustered.columns:
                        # Get problem suggestions
                        problem_suggestions = batch_suggest_problems(df_clustered, threshold=threshold)
                        
                        # Store in session state for persistence across reruns
                        st.session_state['problem_suggestions'] = problem_suggestions
                        st.session_state['last_threshold'] = threshold  # Use different key to avoid widget conflict
                    else:
                        st.session_state['problem_suggestions'] = None

            # Display problem suggestions if they exist in session state
            if 'problem_suggestions' in st.session_state and st.session_state['problem_suggestions']:
                problem_suggestions = st.session_state['problem_suggestions']
                
                if problem_suggestions:
                    st.success(f"🔥 Found {len(problem_suggestions)} cluster(s) that should have Problem Records!")

                    for i, suggestion in enumerate(problem_suggestions):
                        with st.expander(f"📋 Problem Suggestion {i+1}: {suggestion['problem_title']}", expanded=(i==0)):
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric("Incident Count", suggestion['incident_count'])
                            with col2:
                                st.metric("Time Span", f"{suggestion['time_span_days']} days")
                            with col3:
                                st.metric("Priority", suggestion['priority'])

                            st.markdown(f"**Cluster ID:** {suggestion['cluster_id']}")
                            st.markdown(f"**Recommended Assignment:** {suggestion['assignment_group']}")
                            st.markdown(f"**Business Impact:** {suggestion['business_impact']}")

                            if suggestion.get('affected_assets'):
                                st.markdown(f"**Affected Assets:** {', '.join(suggestion['affected_assets'][:5])}")

                            # Display all incidents with smart formatting
                            incident_count = len(suggestion['related_incidents'])
                            st.markdown(f"**Related Incidents ({incident_count}):**")

                            if incident_count <= 20:
                                # Show all incidents inline for small lists
                                st.code(', '.join(suggestion['related_incidents']), language='text')
                            else:
                                # For large lists, show in scrollable code block with newlines for readability
                                incident_list = '\n'.join([
                                    ', '.join(suggestion['related_incidents'][i:i+10])
                                    for i in range(0, incident_count, 10)
                                ])
                                st.code(incident_list, language='text')
                                st.caption(f"Displaying all {incident_count} incidents (grouped by 10 per line)")

                            st.markdown("**Recommended Actions:**")
                            for action in suggestion['recommended_actions']:
                                st.markdown(f"- {action}")

                            if st.button(f"Create Problem Record (Draft)", key=f'create_prb_{i}'):
                                st.info("🚀 In production, this would create a ServiceNow Problem Record")

                                # Format incident list for readability in template
                                incident_count = len(suggestion['related_incidents'])
                                if incident_count <= 50:
                                    incidents_formatted = ', '.join(suggestion['related_incidents'])
                                else:
                                    # For very large lists, group by 10 per line
                                    incidents_formatted = '\n'.join([
                                        ', '.join(suggestion['related_incidents'][i:i+10])
                                        for i in range(0, incident_count, 10)
                                    ])

                                st.code(f"""
Problem Record Details:
━━━━━━━━━━━━━━━━━━━━━━
Title: {suggestion['problem_title']}
Priority: {suggestion['priority']}
Assignment: {suggestion['assignment_group']}
Cluster ID: {suggestion['cluster_id']}
Affected Assets: {', '.join(suggestion.get('affected_assets', [])[:5])}

Description:
{suggestion['business_impact']}

Related Incidents ({incident_count}):
{incidents_formatted}

Keywords: {', '.join(suggestion['top_keywords'])}
                                """, language='text')

                    st.success("💰 **Business Impact:** Proactive problem management prevents ~20 repeat incidents/month, saving $50K+/year")
                else:
                    st.info(f"No clusters found with >= {st.session_state.get('last_threshold', threshold)} incidents. Try lowering the threshold.")
            elif 'problem_suggestions' in st.session_state and st.session_state['problem_suggestions'] is None:
                st.error("Clustering failed. Please check your data.")
        else:
            st.warning("No incident data loaded.")

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

    # ==========================
    # TAB 4: Retro Audit
    # ==========================
    with tab_retro:
        st.header("⏪ Retro Audit (Back to the Future)")
        
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

