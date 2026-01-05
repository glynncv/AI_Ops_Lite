import streamlit as st
import pandas as pd
from data_loader import DataLoader
from retro_analysis import (
    create_timeline_fusion_chart,
    identify_zombie_problems,
    calculate_deflection_opportunity
)

# Page Config
st.set_page_config(page_title="Retro Audit | AI_Ops Flight Deck", layout="wide", page_icon="🔙")

st.title("🔙 Phase 3: The 'Back to the Future' Retro (Historical Audit)")
st.markdown("### Proving the failure of the old model using specific historical data.")
st.info("This dedicated page is optimized for handling heavy CSV uploads for deep historical analysis.")

# --- Sidebar: Data Upload ---
st.sidebar.header("📂 Historical Data Upload")
st.sidebar.markdown("Upload your full-year datasets here.")

uploaded_inc = st.sidebar.file_uploader("Upload Historical Incidents (CSV)", type=['csv'], help="Required columns: number, opened_at, short_description, location")
uploaded_prb = st.sidebar.file_uploader("Upload Historical Problems (CSV)", type=['csv'], help="Required columns: number, opened_at, closed_at, short_description, location.name")

# Load Data
df_inc = pd.DataFrame()
df_prb = pd.DataFrame()
loader = DataLoader()

# Incidents Logic
if uploaded_inc:
    with st.spinner("Processing Uploaded Incidents..."):
        df_inc = loader.load_incidents(uploaded_inc)
        if not df_inc.empty:
            st.sidebar.success(f"✅ Loaded {len(df_inc):,} Incidents (Uploaded)")
else:
    # Try Default
    with st.spinner("Looking for default Incidents..."):
        df_inc = loader.load_incidents(None)
        if not df_inc.empty:
            st.sidebar.info(f"Loaded {len(df_inc):,} Incidents from default file")

# Problems Logic
if uploaded_prb:
    with st.spinner("Processing Uploaded Problems..."):
        df_prb = loader.load_problems(uploaded_prb)
        if not df_prb.empty:
            st.sidebar.success(f"✅ Loaded {len(df_prb):,} Problems (Uploaded)")
else:
    # Try Default
    with st.spinner("Looking for default Problems..."):
        df_prb = loader.load_problems(None)
        if not df_prb.empty:
            st.sidebar.info(f"Loaded {len(df_prb):,} Problems from default file")

# --- Main Analysis Area ---

if df_inc.empty and df_prb.empty:
    st.warning("👈 Please upload your **Historical Incidents** and **Problems** CSV files in the sidebar to generate the audit.")
    st.markdown("""
    **Required Data Schema:**
    *   **Incidents:** `number`, `opened_at`, `short_description`, `location`
    *   **Problems:** `number`, `opened_at`, `closed_at`, `short_description`, `location.name`
    """)

else:
    # Tabs
    tab_fusion, tab_zombies, tab_deflection = st.tabs(["The Timeline Fusion", "Zombie Problems", "Deflection Opportunity"])

    # 1. Timeline Fusion
    with tab_fusion:
        st.subheader("1. The Timeline Fusion")
        st.markdown("**Visual Goal:** Watch for **Blue Dots** (Incidents) appearing *after* a **Red Line** (Problem) ends, indicating a **Failed Fix**.")

        # Prepare Filter Lists
        sites = set()
        if not df_inc.empty and 'location' in df_inc.columns:
            # Extract just the site name if possible (e.g. "10610 - Warwick" -> "Warwick")
            # Logic: If string contains " - ", likely ID - Name - Country.
            # We will use the full string for Uniqueness but maybe display better?
            # Actually, let's keep full string for the selector to be precise, 
            # as user might know the ID or Name.
            sites.update(df_inc['location'].dropna().astype(str).unique())
        if not df_prb.empty and 'location' in df_prb.columns:
            sites.update(df_prb['location'].dropna().astype(str).unique())
            
        site_list = sorted(list(sites))
        
        col_sel, col_viz = st.columns([1, 4])
        
        filtered_inc = pd.DataFrame()
        filtered_prb = pd.DataFrame()
        
        with col_sel:
            st.markdown("##### Configuration")
            selected_site = "Unknown"
            if site_list:
                selected_site = st.selectbox("Select Site", site_list)
                # Filter Data
                if not df_inc.empty and 'location' in df_inc.columns:
                    filtered_inc = df_inc[df_inc['location'].astype(str) == selected_site]
                
                if not df_prb.empty and 'location' in df_prb.columns:
                    filtered_prb = df_prb[df_prb['location'].astype(str) == selected_site]
            else:
                st.warning("No location data found to filter by.")

        with col_viz:
            if not filtered_inc.empty or not filtered_prb.empty:
                fig = create_timeline_fusion_chart(filtered_inc, filtered_prb, site_name=selected_site)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Please select a site with data to view the Timeline Fusion.")

    # 2. Zombie Problems
    with tab_zombies:
        st.subheader("2. Recursion Table ('Zombie Problems')")
        st.markdown("Sites with **>1 Problem Record** (Recurring Failures).")
        
        if not df_prb.empty:
            zombies = identify_zombie_problems(df_prb)
            if not zombies.empty:
                st.error(f"⚠️ Found {len(zombies)} Sites with Recurring Problems")
                st.dataframe(zombies.rename(columns={
                    'Problem_Count': 'Problem Count', 
                    'Problem_Records': 'Problem IDs'
                }), use_container_width=True, hide_index=True)
            else:
                st.success("✅ No 'Zombie Problems' detected (No sites with multiple PRBs).")
        else:
            st.info("No Problem data loaded for Zombie analysis.")

    # 3. Deflection Opportunity
    with tab_deflection:
        st.subheader("3. Deflection Opportunity")
        st.markdown("Potential **Zero-Touch Tickets** (password, reset, admin, install, access).")
        
        if not df_inc.empty:
                d_count, d_pct, d_savings, d_df = calculate_deflection_opportunity(df_inc)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Potential Zero-Touch", d_count)
                c2.metric("% of Total", f"{d_pct:.1%}")
                c3.metric("Est. Savings", f"${d_savings:,}", help="Assuming $50/ticket cost")
                
                if not d_df.empty:
                    with st.expander("View Candidates"):
                        st.dataframe(d_df[['number', 'short_description', 'opened_at', 'assignment_group']].rename(columns={
                            'number': 'Incident #',
                            'short_description': 'Description',
                            'opened_at': 'Opened At',
                            'assignment_group': 'Assignment Group'
                        }), hide_index=True)
        else:
            st.info("No Incident data loaded for Deflection analysis.")
