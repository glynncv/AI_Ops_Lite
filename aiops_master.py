import streamlit as st
import pandas as pd
import snow_connector

# Page Config
st.set_page_config(page_title="AI Ops Flight Deck", layout="wide")

st.title("✈️ AI Ops Flight Deck")

# --- 1. Unified Data Loader ---
def load_unified_data(mode, uploaded_files=None):
    """
    Loads data based on the selected mode: 'Live Connection' or 'Offline Audit'.
    Returns inc_df, prb_df, chg_df.
    """
    inc_df = pd.DataFrame()
    prb_df = pd.DataFrame()
    chg_df = pd.DataFrame()

    if mode == 'Live Connection':
        with st.spinner('Fetching Live Data from ServiceNow (Mock)...'):
            # Fetch data from mock connector
            inc_data = snow_connector.get_snow_data('incident')
            prb_data = snow_connector.get_snow_data('problem')
            chg_data = snow_connector.get_snow_data('change_request')
            
            # Convert to DataFrames
            if inc_data: inc_df = pd.DataFrame(inc_data)
            if prb_data: prb_df = pd.DataFrame(prb_data)
            if chg_data: chg_df = pd.DataFrame(chg_data)

            st.success("Buffers Filled from Live Stream.")

    elif mode == 'Offline Audit (CSV)':
        if uploaded_files:
            inc_file = uploaded_files.get('incidents')
            prb_file = uploaded_files.get('problems')
            chg_file = uploaded_files.get('changes')

            if inc_file:
                inc_df = pd.read_csv(inc_file)
            if prb_file:
                prb_df = pd.read_csv(prb_file)
            if chg_file:
                chg_df = pd.read_csv(chg_file)
            
            if not inc_df.empty or not prb_df.empty or not chg_df.empty:
                st.success("Buffers Filled from CSV Uploads.")

    # Data Normalization (Date Conversion)
    for df in [inc_df, prb_df, chg_df]:
        if not df.empty:
            if 'opened_at' in df.columns:
                df['opened_at'] = pd.to_datetime(df['opened_at'], errors='coerce')
            if 'closed_at' in df.columns:
                df['closed_at'] = pd.to_datetime(df['closed_at'], errors='coerce')
    
    return inc_df, prb_df, chg_df

# Sidebar Logic
st.sidebar.header("Data Source Mode")
mode = st.sidebar.radio("Select Source:", ('Live Connection', 'Offline Audit (CSV)'))

uploaded_files_dict = {}

if mode == 'Offline Audit (CSV)':
    st.sidebar.subheader("Upload CSV Files")
    inc_upload = st.sidebar.file_uploader("Upload Incidents (incidents.csv)", type=['csv'])
    prb_upload = st.sidebar.file_uploader("Upload Problems (problems.csv)", type=['csv'])
    chg_upload = st.sidebar.file_uploader("Upload Changes (changes.csv)", type=['csv'])
    
    uploaded_files_dict = {
        'incidents': inc_upload,
        'problems': prb_upload,
        'changes': chg_upload
    }

# Load Button
if st.sidebar.button("Initialize Flight Deck"):
    inc_df, prb_df, chg_df = load_unified_data(mode, uploaded_files_dict)
    
    # Store in Session State
    st.session_state['inc_df'] = inc_df
    st.session_state['prb_df'] = prb_df
    st.session_state['chg_df'] = chg_df
    st.session_state['data_loaded'] = True
    st.rerun()

# --- Display Data if Loaded ---
if st.session_state.get('data_loaded'):
    st.write("### 📊 Operational Data Loaded")
    
    col1, col2, col3 = st.columns(3)
    
    inc_count = len(st.session_state['inc_df']) if 'inc_df' in st.session_state else 0
    prb_count = len(st.session_state['prb_df']) if 'prb_df' in st.session_state else 0
    chg_count = len(st.session_state['chg_df']) if 'chg_df' in st.session_state else 0

    col1.metric("Incidents", inc_count)
    col2.metric("Problems", prb_count)
    col3.metric("Changes", chg_count)

    with st.expander("View Incidents"):
        if 'inc_df' in st.session_state and not st.session_state['inc_df'].empty:
            st.dataframe(st.session_state['inc_df'])
        else:
            st.info("No Incident Data Available")

    with st.expander("View Problems"):
        if 'prb_df' in st.session_state and not st.session_state['prb_df'].empty:
            st.dataframe(st.session_state['prb_df'])
        else:
            st.info("No Problem Data Available")
            
    with st.expander("View Changes"):
        if 'chg_df' in st.session_state and not st.session_state['chg_df'].empty:
            st.dataframe(st.session_state['chg_df'])
        else:
             st.info("No Change Data Available")
else:
    st.info("👈 Please select a Data Source and click 'Initialize Flight Deck' to begin.")
