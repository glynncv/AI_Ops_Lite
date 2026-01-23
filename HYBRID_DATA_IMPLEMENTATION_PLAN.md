# Hybrid Data Strategy - Phase 1 Implementation Plan

**Date:** 2026-01-22
**Status:** In Progress
**Phase:** 1 of 3 (MVP - Foundation)
**Timeline:** 3 weeks
**Priority:** P0 - Critical

---

## Executive Summary

This document outlines the detailed implementation plan for Phase 1 of the Hybrid Data Strategy, which will enable AI_Ops_Lite to combine live (open) incidents with historical (closed) incidents for effective ML training and analysis.

**Phase 1 Goals:**
- ✅ Add "Hybrid (Live + Historical)" mode to data source selector
- ✅ Implement CSV historical data + Live incident merge logic
- ✅ Update Training Data Status panel with hybrid indicators
- ✅ Ensure ML features use correct datasets
- ✅ Memory optimization with efficient storage
- ✅ Manual testing with 1K, 5K, 10K datasets

---

## Current State Analysis

### Existing Data Modes

| Mode | Source | Data Type | ML Training | Issues |
|------|--------|-----------|-------------|--------|
| **Live API (Mock)** | JSON files | Open incidents | ❌ 0 resolved | Cannot train ML |
| **Live API (Real)** | ServiceNow API | Recent incidents | ❌ Few resolved | Insufficient training data |
| **Offline Data** | CSV upload | Historical data | ✅ Many resolved | No real-time updates |

### The Gap

**Problem:** None of the existing modes provide both:
1. Live open incidents for current triage
2. Historical closed incidents for ML training

**Impact:**
- ML features (routing, similar incidents) cannot train in live mode
- Users stuck in offline mode lose real-time capabilities

---

## Phase 1 Architecture

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────┐
│              Hybrid Data Mode                        │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌────────────────┐         ┌──────────────────┐   │
│  │  Live Source   │         │ Historical Source│   │
│  │                │         │                  │   │
│  │ • ServiceNow   │         │ • CSV Upload     │   │
│  │ • Mock JSON    │         │ • Cached Data    │   │
│  │                │         │                  │   │
│  │ Open Incidents │         │ Closed Incidents │   │
│  │ (500 recent)   │         │ (6,811 historic) │   │
│  └───────┬────────┘         └────────┬─────────┘   │
│          │                           │              │
│          └─────────┬─────────────────┘              │
│                    │                                │
│         ┌──────────▼───────────┐                    │
│         │  Merge & Dedupe      │                    │
│         │  Logic               │                    │
│         └──────────┬───────────┘                    │
│                    │                                │
│         ┌──────────▼───────────┐                    │
│         │  Unified DataFrame   │                    │
│         │  (Session State)     │                    │
│         └──────────┬───────────┘                    │
│                    │                                │
│      ┌─────────────┼─────────────┐                  │
│      │             │             │                  │
│ ┌────▼────┐  ┌─────▼─────┐ ┌───▼────┐             │
│ │ Triage  │  │ML Training│ │Pattern │             │
│ │ (Live)  │  │(Historical)│ │ Detect │             │
│ └─────────┘  └───────────┘ └────────┘             │
└─────────────────────────────────────────────────────┘
```

### Component Architecture

```python
# New module: data_merger.py

class HybridDataManager:
    """
    Manages hybrid data strategy combining live and historical data
    """

    def __init__(self):
        self.live_data = pd.DataFrame()
        self.historical_data = pd.DataFrame()
        self.merged_data = pd.DataFrame()

    def load_live_data(self, source='api'):
        """Load live open incidents"""
        pass

    def load_historical_data(self, source='csv', file_path=None):
        """Load historical closed incidents"""
        pass

    def merge_datasets(self):
        """Merge live + historical with deduplication"""
        pass

    def get_training_data(self):
        """Return data suitable for ML training (resolved only)"""
        pass

    def get_triage_data(self):
        """Return data for current incident triage (live only)"""
        pass

    def get_all_data(self):
        """Return complete merged dataset"""
        pass

    def get_data_stats(self):
        """Return statistics about data composition"""
        pass
```

---

## Implementation Tasks

### Task 1: Add Hybrid Mode Selector (2 hours)

**File:** `app.py`

**Current Code (Line 38):**
```python
data_mode = st.sidebar.selectbox("Select Data Source",
    ["Live API (Mock)", "Live API (Real)", "Offline Data"])
```

**New Code:**
```python
data_mode = st.sidebar.selectbox("Select Data Source",
    ["Live API (Mock)",
     "Live API (Real)",
     "Offline Data",
     "Hybrid (Live + Historical)"])  # NEW
```

**UI Design:**
```
┌────────────────────────────────┐
│ Data Source Config             │
├────────────────────────────────┤
│ Select Data Source:            │
│ [ Hybrid (Live + Historical) ▼]│
│                                │
│ Live Connection:               │
│ ✅ Connected to ServiceNow     │
│ Last refreshed: 2 min ago      │
│                                │
│ Historical Data:               │
│ 📁 CSV Upload: PYTHON_EMEA.csv │
│ 📊 6,811 incidents             │
│ Date range: Jan-Dec 2024       │
│                                │
│ [🔄 Refresh Live Data]         │
│ [📁 Upload New Historical]     │
└────────────────────────────────┘
```

**Success Criteria:**
- ✅ "Hybrid" option appears in dropdown
- ✅ Selecting hybrid mode doesn't crash
- ✅ UI shows data source indicators

---

### Task 2: Create HybridDataManager Module (6 hours)

**File:** `data_merger.py` (NEW)

**Implementation:**

```python
"""
Hybrid Data Management Module

Combines live and historical incident data for optimal AIOps performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class HybridDataManager:
    """
    Manages hybrid data strategy combining live and historical incidents.

    Key Features:
    - Merge live and historical data without duplicates
    - Track data provenance (live vs historical)
    - Provide filtered views for different use cases
    - Memory-efficient storage

    Usage:
        manager = HybridDataManager()
        manager.load_live_data(live_df)
        manager.load_historical_data(historical_df)
        manager.merge_datasets()

        # Get data for specific use cases
        training_data = manager.get_training_data()
        triage_data = manager.get_triage_data()
    """

    def __init__(self):
        self.live_data = pd.DataFrame()
        self.historical_data = pd.DataFrame()
        self.merged_data = pd.DataFrame()
        self.merge_stats = {}

    def load_live_data(self, df: pd.DataFrame) -> Dict:
        """
        Load live incident data (typically open incidents).

        Args:
            df: DataFrame with incident data

        Returns:
            dict: Load statistics
        """
        if df is None or df.empty:
            return {'success': False, 'error': 'Empty dataframe'}

        self.live_data = df.copy()
        self.live_data['data_source'] = 'live'

        return {
            'success': True,
            'count': len(df),
            'open_count': len(df[~df['state'].isin(['Closed', 'Resolved'])]),
            'closed_count': len(df[df['state'].isin(['Closed', 'Resolved'])])
        }

    def load_historical_data(self, df: pd.DataFrame) -> Dict:
        """
        Load historical incident data (typically closed/resolved).

        Args:
            df: DataFrame with historical incident data

        Returns:
            dict: Load statistics
        """
        if df is None or df.empty:
            return {'success': False, 'error': 'Empty dataframe'}

        self.historical_data = df.copy()
        self.historical_data['data_source'] = 'historical'

        # Filter to closed/resolved only for training purity
        closed_mask = self.historical_data['state'].str.lower().isin([
            'closed', 'resolved', 'complete', 'cancelled', 'canceled'
        ])
        self.historical_data = self.historical_data[closed_mask]

        return {
            'success': True,
            'count': len(self.historical_data),
            'date_range': self._get_date_range(self.historical_data)
        }

    def merge_datasets(self) -> Dict:
        """
        Merge live and historical datasets with deduplication.

        Merge Strategy:
        1. Identify duplicates by 'number' field
        2. For duplicates, prefer live data (source of truth)
        3. Keep historical state in separate column for analysis
        4. Add merge metadata for transparency

        Returns:
            dict: Merge statistics
        """
        if self.live_data.empty and self.historical_data.empty:
            return {'success': False, 'error': 'No data to merge'}

        # If only one dataset exists, use it
        if self.live_data.empty:
            self.merged_data = self.historical_data.copy()
            self.merge_stats = {
                'success': True,
                'total_count': len(self.merged_data),
                'live_count': 0,
                'historical_count': len(self.merged_data),
                'duplicates': 0
            }
            return self.merge_stats

        if self.historical_data.empty:
            self.merged_data = self.live_data.copy()
            self.merge_stats = {
                'success': True,
                'total_count': len(self.merged_data),
                'live_count': len(self.merged_data),
                'historical_count': 0,
                'duplicates': 0
            }
            return self.merge_stats

        # Find duplicates
        live_numbers = set(self.live_data['number'].unique())
        historical_numbers = set(self.historical_data['number'].unique())
        duplicates = live_numbers.intersection(historical_numbers)

        # Remove duplicates from historical (prefer live)
        historical_unique = self.historical_data[
            ~self.historical_data['number'].isin(duplicates)
        ]

        # Merge
        self.merged_data = pd.concat([
            self.live_data,
            historical_unique
        ], ignore_index=True)

        # Sort by date for chronological order
        if 'opened_at' in self.merged_data.columns:
            self.merged_data = self.merged_data.sort_values('opened_at', ascending=False)

        self.merge_stats = {
            'success': True,
            'total_count': len(self.merged_data),
            'live_count': len(self.live_data),
            'historical_count': len(historical_unique),
            'duplicates': len(duplicates),
            'duplicate_incidents': list(duplicates)[:10]  # Sample for logging
        }

        logger.info(f"Merged datasets: {self.merge_stats}")
        return self.merge_stats

    def get_training_data(self) -> pd.DataFrame:
        """
        Get data suitable for ML training (resolved/closed incidents only).

        Returns:
            DataFrame: Filtered for ML training
        """
        if self.merged_data.empty:
            return pd.DataFrame()

        # Filter to resolved/closed states
        training_mask = self.merged_data['state'].str.lower().isin([
            'closed', 'resolved', 'complete', 'cancelled', 'canceled'
        ])

        return self.merged_data[training_mask].copy()

    def get_triage_data(self) -> pd.DataFrame:
        """
        Get data for current incident triage (live open incidents).

        Returns:
            DataFrame: Live open incidents only
        """
        if self.merged_data.empty:
            return pd.DataFrame()

        # Filter to live open incidents
        triage_mask = (
            (self.merged_data['data_source'] == 'live') &
            (~self.merged_data['state'].str.lower().isin([
                'closed', 'resolved', 'complete', 'cancelled', 'canceled'
            ]))
        )

        return self.merged_data[triage_mask].copy()

    def get_all_data(self) -> pd.DataFrame:
        """Get complete merged dataset."""
        return self.merged_data.copy()

    def get_data_stats(self) -> Dict:
        """
        Get comprehensive statistics about data composition.

        Returns:
            dict: Data statistics
        """
        if self.merged_data.empty:
            return {'total': 0}

        stats = {
            'total': len(self.merged_data),
            'live': len(self.merged_data[self.merged_data['data_source'] == 'live']),
            'historical': len(self.merged_data[self.merged_data['data_source'] == 'historical']),
            'open': len(self.merged_data[~self.merged_data['state'].str.lower().isin([
                'closed', 'resolved', 'complete', 'cancelled', 'canceled'
            ])]),
            'closed': len(self.merged_data[self.merged_data['state'].str.lower().isin([
                'closed', 'resolved', 'complete', 'cancelled', 'canceled'
            ])]),
            'training_ready': self._check_training_readiness(),
            'date_range': self._get_date_range(self.merged_data),
            'merge_info': self.merge_stats
        }

        return stats

    def _check_training_readiness(self) -> Dict:
        """Check if data is sufficient for ML training."""
        training_data = self.get_training_data()

        min_required = 10  # Minimum incidents for training
        has_enough = len(training_data) >= min_required

        return {
            'ready': has_enough,
            'count': len(training_data),
            'required': min_required,
            'percentage': (len(training_data) / len(self.merged_data) * 100) if len(self.merged_data) > 0 else 0
        }

    def _get_date_range(self, df: pd.DataFrame) -> Dict:
        """Get date range from dataframe."""
        if df.empty or 'opened_at' not in df.columns:
            return {'start': None, 'end': None}

        try:
            df['opened_at'] = pd.to_datetime(df['opened_at'], errors='coerce')
            return {
                'start': df['opened_at'].min().strftime('%Y-%m-%d') if pd.notna(df['opened_at'].min()) else None,
                'end': df['opened_at'].max().strftime('%Y-%m-%d') if pd.notna(df['opened_at'].max()) else None
            }
        except:
            return {'start': None, 'end': None}


# Helper function for backward compatibility
def merge_live_and_historical(live_df: pd.DataFrame, historical_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to merge live and historical data.

    Args:
        live_df: Live incident data
        historical_df: Historical incident data

    Returns:
        tuple: (merged_df, stats_dict)
    """
    manager = HybridDataManager()
    manager.load_live_data(live_df)
    manager.load_historical_data(historical_df)
    merge_stats = manager.merge_datasets()

    return manager.get_all_data(), merge_stats
```

**Testing:**
```python
# Unit tests in tests/test_data_merger.py
def test_merge_no_duplicates():
    live = pd.DataFrame({'number': ['INC001', 'INC002'], 'state': ['New', 'Open'], 'short_description': ['A', 'B']})
    historical = pd.DataFrame({'number': ['INC003', 'INC004'], 'state': ['Closed', 'Resolved'], 'short_description': ['C', 'D']})

    manager = HybridDataManager()
    manager.load_live_data(live)
    manager.load_historical_data(historical)
    manager.merge_datasets()

    assert len(manager.get_all_data()) == 4
    assert manager.merge_stats['duplicates'] == 0

def test_merge_with_duplicates():
    live = pd.DataFrame({'number': ['INC001', 'INC002'], 'state': ['New', 'Open'], 'short_description': ['A', 'B']})
    historical = pd.DataFrame({'number': ['INC002', 'INC003'], 'state': ['Closed', 'Resolved'], 'short_description': ['B_old', 'C']})

    manager = HybridDataManager()
    manager.load_live_data(live)
    manager.load_historical_data(historical)
    stats = manager.merge_datasets()

    assert stats['duplicates'] == 1
    assert len(manager.get_all_data()) == 3  # INC001, INC002 (live), INC003

    # Verify live version is kept
    inc002 = manager.get_all_data()[manager.get_all_data()['number'] == 'INC002'].iloc[0]
    assert inc002['state'] == 'Open'  # Live state, not historical 'Closed'

def test_training_data_filter():
    live = pd.DataFrame({'number': ['INC001', 'INC002'], 'state': ['New', 'Closed'], 'short_description': ['A', 'B']})
    historical = pd.DataFrame({'number': ['INC003', 'INC004'], 'state': ['Closed', 'Resolved'], 'short_description': ['C', 'D']})

    manager = HybridDataManager()
    manager.load_live_data(live)
    manager.load_historical_data(historical)
    manager.merge_datasets()

    training = manager.get_training_data()
    assert len(training) == 3  # INC002, INC003, INC004 (all closed/resolved)
    assert all(training['state'].isin(['Closed', 'Resolved']))
```

**Success Criteria:**
- ✅ Module passes all unit tests
- ✅ Deduplication works correctly
- ✅ Data provenance tracked
- ✅ No memory leaks with 10K incidents

---

### Task 3: Integrate Hybrid Mode into app.py (8 hours)

**File:** `app.py`

**Implementation Steps:**

1. **Add import:**
```python
from data_merger import HybridDataManager
```

2. **Add hybrid mode handler (after line 106):**
```python
elif data_mode == 'Hybrid (Live + Historical)':
    st.sidebar.info("Mode: Hybrid (Live + Historical)")

    # Initialize HybridDataManager in session state
    if 'hybrid_manager' not in st.session_state:
        st.session_state['hybrid_manager'] = HybridDataManager()

    manager = st.session_state['hybrid_manager']

    # Section 1: Live Data
    st.sidebar.subheader("🔴 Live Data")

    live_source = st.sidebar.radio("Live Source", ["ServiceNow API", "Mock JSON"], key='live_source')

    if live_source == "ServiceNow API":
        # Use existing ServiceNow connection logic
        instance_url = os.getenv("SNOW_INSTANCE_URL", "")
        username = os.getenv("SNOW_USERNAME", "")
        password = os.getenv("SNOW_PASSWORD", "")

        if st.sidebar.button("🔄 Fetch Live Data"):
            with st.spinner("Fetching live incidents..."):
                try:
                    incidents = snow_connector.fetch_incidents_snow(instance_url, username, password)
                    if incidents:
                        live_df = process_snow_data(incidents)
                        load_stats = manager.load_live_data(live_df)
                        st.sidebar.success(f"✅ Loaded {load_stats['count']} live incidents")
                except Exception as e:
                    st.sidebar.error(f"❌ Error: {e}")
    else:
        # Mock JSON
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'input', 'incidents.json')
        if st.sidebar.button("📁 Load Mock Live Data"):
            try:
                with open(data_path, 'r') as f:
                    incidents_data = json.load(f)
                live_df = process_snow_data(incidents_data)
                load_stats = manager.load_live_data(live_df)
                st.sidebar.success(f"✅ Loaded {load_stats['count']} live incidents")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {e}")

    # Section 2: Historical Data
    st.sidebar.subheader("📊 Historical Data")

    historical_source = st.sidebar.radio("Historical Source",
        ["CSV Upload", "Use Cached"], key='hist_source')

    if historical_source == "CSV Upload":
        uploaded_file = st.sidebar.file_uploader("Upload Historical CSV",
            type=['csv'], key='hist_csv')

        if uploaded_file is not None:
            try:
                loader = DataLoader()
                historical_df = loader.load_incidents_csv(uploaded_file)
                load_stats = manager.load_historical_data(historical_df)
                st.sidebar.success(f"✅ Loaded {load_stats['count']} historical incidents")
                st.sidebar.info(f"📅 Date range: {load_stats['date_range']['start']} to {load_stats['date_range']['end']}")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {e}")
    else:
        # Load from default historical file
        default_path = 'data/input/PYTHON EMEA IM (2025).csv'
        if st.sidebar.button("📁 Load Default Historical Data"):
            try:
                loader = DataLoader()
                historical_df = loader.load_incidents_csv(default_path)
                load_stats = manager.load_historical_data(historical_df)
                st.sidebar.success(f"✅ Loaded {load_stats['count']} historical incidents")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {e}")

    # Section 3: Merge & Stats
    st.sidebar.markdown("---")
    if st.sidebar.button("🔀 Merge Datasets"):
        with st.spinner("Merging live and historical data..."):
            merge_stats = manager.merge_datasets()
            if merge_stats['success']:
                st.sidebar.success(f"✅ Merged successfully!")
                st.sidebar.info(f"""
                📊 Dataset Summary:
                - Total: {merge_stats['total_count']}
                - Live: {merge_stats['live_count']}
                - Historical: {merge_stats['historical_count']}
                - Duplicates removed: {merge_stats['duplicates']}
                """)

                # Store merged data in session state for other features
                df_cleaned = manager.get_all_data()
                st.session_state['inc_df'] = df_cleaned
            else:
                st.sidebar.error(f"❌ Merge failed: {merge_stats.get('error')}")

    # Display current data stats
    if not manager.merged_data.empty:
        stats = manager.get_data_stats()
        st.sidebar.metric("Total Incidents", stats['total'])

        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Live", stats['live'])
            st.metric("Open", stats['open'])
        with col2:
            st.metric("Historical", stats['historical'])
            st.metric("Closed", stats['closed'])

        # Training readiness indicator
        if stats['training_ready']['ready']:
            st.sidebar.success(f"✅ Training Ready ({stats['training_ready']['count']} resolved)")
        else:
            st.sidebar.warning(f"⚠️ Need {stats['training_ready']['required'] - stats['training_ready']['count']} more resolved incidents")
```

**Success Criteria:**
- ✅ Hybrid mode selectable
- ✅ Can load live + historical data separately
- ✅ Merge button works
- ✅ Stats displayed correctly
- ✅ Merged data available to all features

---

### Task 4: Update Training Data Status Panel (4 hours)

**Location:** AI Intelligence tab, before ML features

**New UI:**
```python
# In AI Intelligence tab (around line 350)
if not df_cleaned.empty:
    st.subheader("📊 Training Data Status")

    # Check if hybrid mode
    is_hybrid = data_mode == 'Hybrid (Live + Historical)'

    if is_hybrid and 'hybrid_manager' in st.session_state:
        manager = st.session_state['hybrid_manager']
        stats = manager.get_data_stats()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Incidents", stats['total'])
        with col2:
            st.metric("Live Open", stats['open'],
                     help="Current incidents for triage")
        with col3:
            st.metric("Historical Closed", stats['historical'],
                     help="Used for ML training")
        with col4:
            training_ready = stats['training_ready']['ready']
            st.metric("Training Status",
                     "✅ Ready" if training_ready else "⚠️ Insufficient",
                     help=f"{stats['training_ready']['count']} resolved incidents")

        # Data composition chart
        st.markdown("**Data Composition:**")
        composition_data = pd.DataFrame({
            'Source': ['Live', 'Historical'],
            'Count': [stats['live'], stats['historical']]
        })
        st.bar_chart(composition_data.set_index('Source'))

        # State distribution
        if 'state' in df_cleaned.columns:
            st.markdown("**State Distribution:**")
            state_dist = df_cleaned['state'].value_counts()
            st.dataframe(state_dist, use_container_width=True)

    else:
        # Non-hybrid mode: show basic stats
        total = len(df_cleaned)
        resolved_count = len(df_cleaned[df_cleaned['state'].str.lower().isin([
            'closed', 'resolved', 'complete', 'cancelled', 'canceled'
        ])])

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Incidents", total)
        with col2:
            st.metric("Resolved/Closed", resolved_count)

        if resolved_count < 10:
            st.warning(f"⚠️ Only {resolved_count} resolved incidents. ML features require at least 10. Consider using Hybrid mode.")
```

**Success Criteria:**
- ✅ Panel shows clear data composition
- ✅ Training readiness indicator accurate
- ✅ Visual charts display correctly
- ✅ Warnings shown when insufficient data

---

### Task 5: Update ML Features to Use Filtered Data (6 hours)

**Files to update:**
- `app.py` (ML feature sections)
- Documentation comments

**Changes needed:**

1. **Intelligent Assignment Routing (Training):**
```python
# Before training, filter to historical data
if data_mode == 'Hybrid (Live + Historical)' and 'hybrid_manager' in st.session_state:
    training_df = st.session_state['hybrid_manager'].get_training_data()
else:
    # Regular mode: use all resolved incidents
    training_df = df_cleaned[df_cleaned['state'].str.lower().isin([
        'closed', 'resolved', 'complete', 'cancelled', 'canceled'
    ])]

# Train router
router = IntelligentRouter()
metrics = router.train(training_df)
```

2. **Similar Incident Search:**
```python
# Use training data (historical) for similarity search
if data_mode == 'Hybrid (Live + Historical)' and 'hybrid_manager' in st.session_state:
    search_corpus = st.session_state['hybrid_manager'].get_training_data()
else:
    search_corpus = df_cleaned[df_cleaned['state'].str.lower().isin([
        'closed', 'resolved', 'complete', 'cancelled', 'canceled'
    ])]

similar = find_similar_resolved_incidents(incident_desc, search_corpus)
```

3. **Problem Detection:**
```python
# Use all data (live + historical) for clustering
if data_mode == 'Hybrid (Live + Historical)' and 'hybrid_manager' in st.session_state:
    cluster_df = st.session_state['hybrid_manager'].get_all_data()
else:
    cluster_df = df_cleaned

# Proceed with clustering
clusters = perform_clustering(cluster_df)
```

**Success Criteria:**
- ✅ ML features use correct data subsets
- ✅ Training uses only historical/resolved
- ✅ Clustering uses all data
- ✅ No degradation in ML accuracy

---

## Testing Strategy

### Unit Tests (data_merger.py)

**File:** `tests/test_data_merger.py` (NEW)

Tests:
1. ✅ Test merge with no duplicates
2. ✅ Test merge with duplicates (verify live preferred)
3. ✅ Test training data filter
4. ✅ Test triage data filter
5. ✅ Test data stats calculation
6. ✅ Test empty dataframes
7. ✅ Test large datasets (10K incidents)

### Integration Tests

**File:** `tests/test_hybrid_mode_integration.py` (NEW)

Tests:
1. ✅ Load live + historical data
2. ✅ Merge datasets
3. ✅ Train ML router with hybrid data
4. ✅ Search similar incidents
5. ✅ Perform clustering
6. ✅ Verify data stats accuracy

### Manual Testing Scenarios

**Scenario 1: Small Dataset (100 incidents)**
- Load 50 live open incidents
- Load 50 historical closed incidents
- Merge and verify 100 total
- Train ML features
- Verify training succeeds

**Scenario 2: Medium Dataset (1,000 incidents)**
- Load 200 live incidents
- Load 800 historical incidents
- Merge and verify deduplication
- Check memory usage (<100MB)
- Test all ML features

**Scenario 3: Large Dataset (10,000 incidents)**
- Load 500 live incidents
- Load 9,500 historical incidents
- Merge time <5 seconds
- Memory usage <500MB
- ML training time <30 seconds

**Scenario 4: Duplicate Handling**
- Create dataset with known duplicates
- Verify live version is kept
- Check merge stats show correct duplicate count

**Scenario 5: Edge Cases**
- Empty live data
- Empty historical data
- All duplicates
- Invalid CSV format
- Missing required columns

---

## Success Criteria (Phase 1)

### Functional Requirements
- ✅ Hybrid mode option available in UI
- ✅ Can load live data (API or mock)
- ✅ Can load historical data (CSV upload)
- ✅ Merge logic works correctly
- ✅ Deduplication removes duplicates, prefers live
- ✅ Training data filtered to resolved/closed only
- ✅ Data stats panel shows accurate metrics
- ✅ ML features use appropriate data subsets

### Non-Functional Requirements
- ✅ Load 10K incidents in <3 minutes
- ✅ Memory usage <500MB for 10K incidents
- ✅ Merge operation <5 seconds
- ✅ No performance degradation vs single-mode
- ✅ UI remains responsive during operations

### Quality Requirements
- ✅ Unit test coverage >80%
- ✅ No critical bugs in manual testing
- ✅ Documentation complete
- ✅ Code review passed

---

## Rollout Plan

### Week 1: Development
- Day 1-2: Create HybridDataManager module + tests
- Day 3-4: Integrate hybrid mode into app.py
- Day 5: Update Training Data Status panel

### Week 2: Testing & Refinement
- Day 1-2: Unit testing and bug fixes
- Day 3: Integration testing
- Day 4-5: Manual testing scenarios

### Week 3: Documentation & Deployment
- Day 1-2: User documentation
- Day 3: Code review
- Day 4: Deploy to test environment
- Day 5: User acceptance testing

---

## Known Limitations (Phase 1)

**What Phase 1 DOES NOT include:**
- ❌ Automatic historical data fetching from ServiceNow API
- ❌ Incremental updates (full reload only)
- ❌ Persistent cache across sessions
- ❌ Background refresh
- ❌ Data quality monitoring
- ❌ Advanced merge conflict resolution

**These features are planned for Phase 2 and Phase 3.**

---

## Dependencies

**Python Packages:**
- pandas (existing)
- numpy (existing)
- No new external dependencies

**Data Requirements:**
- CSV files must have 'number' field for deduplication
- CSV files must have 'state' field for filtering
- Date fields should be parseable by pd.to_datetime()

**Environment:**
- No changes to .env required
- No database setup needed

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Memory overflow with large CSVs** | Medium | High | Test with 10K, add loading limits |
| **Merge logic bugs** | Low | High | Comprehensive unit tests |
| **UI complexity confuses users** | Medium | Medium | Clear labels, help text, tooltips |
| **Performance degradation** | Low | Medium | Profile and optimize, lazy loading |
| **Data quality issues** | Medium | Low | Add validation, error messages |

---

## Next Steps After Phase 1

**Phase 2 Goals (4 weeks):**
- ServiceNow Archive API integration
- Automatic historical data fetching
- Incremental updates
- Background refresh

**Phase 3 Goals (6 weeks):**
- Persistent cache with versioning
- Data quality dashboard
- Advanced conflict resolution
- Performance optimization

---

## Appendix: Code Locations

### Files to Create
- `data_merger.py` - New module
- `tests/test_data_merger.py` - Unit tests
- `tests/test_hybrid_mode_integration.py` - Integration tests

### Files to Modify
- `app.py` - Lines 38, 106+ (add hybrid mode)
- `app.py` - AI Intelligence tab (update training data panel)
- `app.py` - ML feature sections (use filtered data)

### Files to Read
- `data_loader.py` - Understand CSV loading
- `data_processor.py` - Understand ServiceNow processing
- `aiops_intelligence.py` - Understand ML feature requirements

---

**Plan Created:** 2026-01-22
**Next Review:** After Phase 1 completion
**Owner:** Development Team
**Status:** ✅ Ready for Implementation

