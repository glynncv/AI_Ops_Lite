# ML Model Persistence Across Data Modes - Explained

## Your Question
> "So I switch to offline mode and run the ML training and then ML knowledge is available when I switch back to live or retro mode?"

## Short Answer
**Partially YES, but with important caveats.**

The **ML model itself** persists in session state, BUT the **data does NOT** persist when switching modes. Here's what actually happens:

---

## 🔄 Current Behavior (How It Works Today)

### Session State Storage

The app uses Streamlit's `session_state` to persist data between interactions:

```python
# app.py - Lines 54-59
if 'inc_df' in st.session_state:
    df_cleaned = st.session_state['inc_df']
if 'prb_df' in st.session_state:
    problems_df = st.session_state['prb_df']
if 'chg_df' in st.session_state:
    changes_df = st.session_state['chg_df']

# ML Router stored in session state
# app.py - Lines 396-397
if 'router' not in st.session_state:
    st.session_state.router = IntelligentRouter()
```

### Mode-Specific Behavior

| Data Mode | Data Persistence | ML Model Persistence |
|-----------|------------------|----------------------|
| **Live API (Real)** | ✅ Stored in session_state | ✅ Stored in session_state |
| **Live API (Mock)** | ❌ Reloaded each time | ✅ Stored in session_state |
| **Offline Data (CSV)** | ❌ Reloaded each time | ✅ Stored in session_state |

---

## 📊 Detailed Scenario Walkthrough

### Scenario 1: Train in Offline Mode, Switch to Mock Mode

```
Step 1: Select "Offline Data" mode
   ↓
   Loads: PYTHON EMEA IM (2025).csv (9,240 incidents)
   Creates: df_cleaned (local variable, NOT in session_state)
   ↓
Step 2: Go to AI Intelligence tab → Click "Train Assignment Model"
   ↓
   Trains: st.session_state.router on df_cleaned (9,240 incidents)
   Result: router.trained = True
           router.model = RandomForest (trained on 9K incidents)
           router.vectorizer = TfidfVectorizer (fitted on 9K incidents)
   ↓
Step 3: Switch to "Live API (Mock)" mode
   ↓
   Loads: incidents.json (30 incidents)
   Creates: df_cleaned (local variable with 30 incidents)
   Session State: st.session_state.router still exists (still trained!)
   ↓
Step 4: Try to use the model
   ↓
   Result: ✅ Model still works!
           BUT: It was trained on 9,240 EMEA incidents (offline)
                Now you have 30 mock incidents loaded
                Model predictions are based on EMEA training, not mock data
```

**What You Get:**
- ✅ Model predictions still work
- ⚠️ Model was trained on different data than currently loaded
- ⚠️ Can't retrain on mock data (only 30 incidents, model expects similar vocabulary)

---

### Scenario 2: Train in Offline Mode, Switch to Live API (Real)

```
Step 1: Select "Offline Data" mode
   ↓
   Trains model on 9,240 EMEA CSV incidents
   ↓
Step 2: Switch to "Live API (Real)" mode
   ↓
   Fetches: Last 30 days from ServiceNow (let's say 500 incidents)
   Stores: st.session_state['inc_df'] = live_incidents
   Session State: st.session_state.router still exists (trained on EMEA CSV)
   ↓
Step 3: Model behavior
   ↓
   Result: ✅ Model still works
           ⚠️ Trained on CSV data, predicting on live data
           ⚠️ If your CSV and live data are from same org/region → predictions likely good
           ⚠️ If different → predictions may be off
```

**What You Get:**
- ✅ Model predictions work on live data
- ⚠️ Model still reflects CSV training (not live data characteristics)
- ✅ Data DOES persist in session_state for Live API (Real) mode
- ⚠️ Switching back to Offline would reload CSV, losing live data

---

### Scenario 3: What Happens in Retro Audit

**Important**: Retro Audit is **NOT a separate data mode**, it's a **tab/section** that uses whatever data is currently loaded.

```
Retro Audit (Phase 5) uses:
- df_cleaned (incidents)
- problems_df (problems)
- changes_df (changes)

These come from whichever mode is selected:
- "Live API (Mock)" → JSON files
- "Live API (Real)" → ServiceNow API
- "Offline Data" → CSV files
```

**So if you:**
1. Train ML in Offline mode (CSV data)
2. Stay in Offline mode
3. Go to Retro Audit tab

**Result**: ✅ Retro Audit sees the CSV data (9,240 incidents, 122 problems)

**But if you:**
1. Train ML in Offline mode (CSV data)
2. Switch to Mock mode
3. Go to Retro Audit tab

**Result**: ⚠️ Retro Audit sees JSON data (30 incidents, 3 problems)
            ⚠️ ML model still trained on CSV data
            ⚠️ Mismatch between training and current data

---

## 🎯 The Key Issue: Data Persistence Gap

### Current Code Issue

```python
# Live API (Real) - DOES persist data
st.session_state['inc_df'] = process_snow_data(inc_data)  # ✅
st.session_state['prb_df'] = problems_df                  # ✅
st.session_state['chg_df'] = changes_df                   # ✅

# Live API (Mock) - Does NOT persist data
df_cleaned = process_snow_data(incidents_data)  # ❌ Local variable only
problems_df = pd.DataFrame(problems_data)       # ❌ Local variable only

# Offline Data - Does NOT persist data
df_cleaned = loader.load_incidents(up_inc)      # ❌ Local variable only
changes_df = loader.load_changes(up_chg)        # ❌ Local variable only
problems_df = loader.load_problems(up_prob)     # ❌ Local variable only
```

This means:
- **Switching modes** triggers a Streamlit rerun
- **Streamlit reruns** the entire script
- **Local variables** are lost
- **Only session_state** persists

---

## 💡 What This Means in Practice

### Use Case 1: Train Once, Use Everywhere (Current Behavior)

**Workflow:**
```
1. Switch to "Offline Data" (CSV with 9,240 incidents)
2. Train Intelligent Router model
3. Use model predictions while staying in Offline mode ✅
4. Switch to "Live API (Real)"
5. Model predictions still work ⚠️ (but trained on old data)
```

**Limitation**: Model trained on one dataset, applied to different dataset

---

### Use Case 2: Analyze Different Data Sources (Current Behavior)

**Workflow:**
```
1. Switch to "Offline Data" → See Retro Audit with 14 months CSV data ✅
2. Switch to "Live API (Mock)" → See Retro Audit with JSON sample data ✅
3. Switch to "Live API (Real)" → See Retro Audit with live 30-day data ✅
```

**Works**: Each mode shows analysis of its own data
**Limitation**: Switching modes loses previous data (except Live API Real)

---

### Use Case 3: Train on Historical, Predict on Current (Desired?)

**Ideal Workflow:**
```
1. Switch to "Offline Data" (14 months historical)
2. Train all ML models on historical data
3. Switch to "Live API (Real)" (current 30 days)
4. Use historical-trained models to predict on current incidents
```

**Current Status**: ⚠️ Partially works
- Model predictions work
- But model trained on offline data, not aware of current patterns
- No explicit "train on X, apply to Y" workflow

---

## 🔧 Recommended Usage Patterns

### For Training ML Models

**Best Practice**: Train and use models in the **same data mode**

```
GOOD ✅:
1. Select "Offline Data" (CSV)
2. Train Intelligent Router
3. Test predictions
4. Stay in Offline mode while using the model

RISKY ⚠️:
1. Select "Offline Data" (CSV)
2. Train Intelligent Router on EMEA data
3. Switch to "Live API (Mock)"
4. Try to predict on mock data (different vocabulary, different teams)
```

### For Retro Audit Analysis

**Best Practice**: Use **Offline Data** for comprehensive analysis

```
Offline Data mode gives you:
- 9,240 incidents (vs 30 mock or 500 live)
- 14 months of history (vs current snapshot)
- 122 problems (vs 3 mock)
- Best for clustering analysis
- Best for zombie detection
- Best for temporal patterns
```

**Why**: Retro Audit needs historical depth to show patterns

---

## 🚀 Proposed Enhancement (For Future)

### Add Explicit Data Persistence to All Modes

```python
# Proposed fix in app.py
elif data_mode == 'Offline Data':
    st.sidebar.info("Mode: Offline Data (CSV)")
    loader = DataLoader()

    # Load data
    df_cleaned = loader.load_incidents(up_inc)
    changes_df = loader.load_changes(up_chg)
    problems_df = loader.load_problems(up_prob)

    # PROPOSED: Store in session state (like Live API Real does)
    if not df_cleaned.empty:
        st.session_state['inc_df'] = df_cleaned        # ✅ NEW
        st.sidebar.success(f"Loaded {len(df_cleaned)} Incidents")
    if not changes_df.empty:
        st.session_state['chg_df'] = changes_df        # ✅ NEW
        st.sidebar.success(f"Loaded {len(changes_df)} Changes")
    if not problems_df.empty:
        st.session_state['prb_df'] = problems_df       # ✅ NEW
```

**Benefits of this change:**
- Data persists when switching modes
- Can train on one dataset, switch modes, data still available
- More consistent behavior across all modes

**Potential issue:**
- Users might get confused about which data is currently "active"
- Would need UI indicator showing "Data loaded from: Offline CSV" even after switching modes

---

## 📋 Summary Table

| Action | ML Model | Data | Works? |
|--------|----------|------|--------|
| Train in Offline, predict in Offline | ✅ Persists | ✅ Available | ✅ YES |
| Train in Offline, switch to Mock, predict | ✅ Persists | ❌ Reloaded | ⚠️ Model works but mismatch |
| Train in Offline, switch to Live Real, predict | ✅ Persists | ✅ Persists | ✅ YES |
| Train in Live Real, switch to Offline | ✅ Persists | ❌ Reloaded | ⚠️ Model works but mismatch |
| View Retro Audit in Offline | N/A | ✅ CSV loaded | ✅ YES |
| Train in Offline, switch to Mock, view Retro Audit | ✅ Model persists | ❌ Sees Mock JSON | ⚠️ Sees different data |

---

## 🎯 Direct Answer to Your Question

> "So I switch to offline mode and run the ML training and then ML knowledge is available when I switch back to live or retro mode?"

**Answer:**

**For ML Model (Intelligent Router):**
- ✅ YES - The trained model persists in `st.session_state.router`
- ✅ You can make predictions even after switching modes
- ⚠️ BUT the model was trained on offline data, so predictions reflect that training

**For Data (Incidents/Problems/Changes):**
- ❌ NO - Data does NOT persist when switching from Offline to Mock
- ✅ YES - Data DOES persist when switching from Offline to Live API (Real)... wait no, that's wrong. Let me recheck.

Actually, let me correct:
- ❌ Offline data does NOT persist when switching modes
- ✅ Live API (Real) data DOES persist when switching modes
- ❌ Mock data does NOT persist when switching modes

**For Retro Audit:**
- ⚠️ Retro Audit shows whatever data is currently loaded in the selected mode
- If you train in Offline then switch to Mock, Retro Audit will show Mock data (30 incidents)
- If you want Retro Audit to see the 9,240 incidents, stay in Offline mode

---

## 💡 Recommended Workflow

### For Best Results:

**Training and Using ML Models:**
```
1. Select "Offline Data" mode
2. Train Intelligent Router (on 9,240 incidents)
3. Test predictions
4. Keep using Offline mode for predictions
```

**Analyzing Historical Patterns (Retro Audit):**
```
1. Select "Offline Data" mode
2. Go to Retro Audit tab
3. See comprehensive 14-month analysis
4. Stay in Offline mode while analyzing
```

**Live Monitoring:**
```
1. Select "Live API (Real)" mode
2. Data loads from ServiceNow (persists in session_state)
3. Can train models on live data
4. Can switch modes and data persists
```

---

## 🐛 Known Limitation

**The current design does not support:**
- Training on historical (Offline) data
- Then applying to live (Real API) data
- While keeping both datasets available

**Workaround**: Train in the mode you'll be using for analysis.

**Future Enhancement**: Add data persistence to all modes, plus UI to show which dataset is "active"

---

**Created**: 2026-01-14
**Branch**: `claude/retro-audit-analysis-nWG3u`
**Status**: Documentation of current behavior
