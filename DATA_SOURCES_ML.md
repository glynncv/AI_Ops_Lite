# Data Sources for ML in AI_Ops_Lite

## Overview

The AI_Ops_Lite project uses **three primary data sources** for machine learning and analysis, designed to work in both **development/demo** and **production** environments.

---

## 📊 Data Source Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Data Source Layer                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Mock JSON Data (Development/Demo)                          │
│     ├─ incidents.json (185 lines, ~6KB)                        │
│     ├─ problems.json (23 lines, ~687B)                         │
│     └─ changes.json (47 lines, ~1.5KB)                         │
│                                                                  │
│  2. Real CSV Data (Production/Testing)                          │
│     ├─ PYTHON EMEA IM (2025).csv (9,241 lines, 3.9MB)         │
│     ├─ CHANGES EMEA 2025.csv (3,579 lines, 267KB)             │
│     ├─ PYTHON EMEA PM P1P2.csv (123 lines, 28KB)              │
│     └─ PYTHON EMEA TASK RCA.csv (62 lines, 13KB)              │
│                                                                  │
│  3. Live ServiceNow API (Production)                            │
│     ├─ GET /api/now/table/incident                             │
│     ├─ GET /api/now/table/problem                              │
│     └─ GET /api/now/table/change_request                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    ML/AI Processing Layer                        │
├─────────────────────────────────────────────────────────────────┤
│  • DBSCAN Clustering (TF-IDF + Cosine Similarity)              │
│  • RandomForest Classification (Intelligent Routing)            │
│  • Isolation Forest (Anomaly Detection)                         │
│  • TF-IDF Vectorization (Similar Incident Matching)            │
│  • Regex Entity Extraction (Zombie Detection)                   │
│  • TextBlob Sentiment Analysis (Quality Audit)                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. 🧪 Mock JSON Data (Development/Demo)

### Location
`data/input/incidents.json`, `problems.json`, `changes.json`

### Purpose
- Quick testing and development
- Demo presentations without real data
- Lightweight sample data for new users

### Schema Example (incidents.json)
```json
{
    "number": "INC7538368",
    "opened_at": "2025-12-15",
    "short_description": "SAP manufacturing issue reported by plant",
    "assignment_group": "SAP_MANUFACTURING",
    "state": "Closed",
    "close_notes": "User training provided"
}
```

### Key Fields Used by ML
- **short_description** + **description** → TF-IDF vectorization for clustering
- **assignment_group** → Training labels for Intelligent Routing
- **opened_at** / **closed_at** → Temporal analysis, MTTR calculation
- **state** → Filter for resolved incidents (training data)
- **close_notes** → Similar incident resolution matching

### Size
- **Incidents**: ~30 sample records (185 lines)
- **Problems**: ~3 sample records (23 lines)
- **Changes**: ~10 sample records (47 lines)

**Use Case**: "Live API (Mock)" mode in Streamlit UI

---

## 2. 📁 Real CSV Data (Production/Testing)

### Location
`data/input/` directory

### Files
| File | Size | Records | Purpose |
|------|------|---------|---------|
| **PYTHON EMEA IM (2025).csv** | 3.9MB | ~9,240 | Incident Management data |
| **CHANGES EMEA 2025 - Closed Complete.csv** | 267KB | ~3,578 | Change Request data |
| **PYTHON EMEA PM P1P2 (This Year).csv** | 28KB | ~122 | Problem Management data |
| **PYTHON EMEA TASK RCA (This Year).csv** | 13KB | ~61 | Root Cause Analysis tasks |

### Schema Example (PYTHON EMEA IM)
```csv
"number","sys_created_on","short_description","location","assignment_group",
"reassignment_count","opened_at","problem_id","priority","u_ci_type",
"u_resolved","close_code","incident_state","category","cmdb_ci",
"closed_at","location.city","location.country","location.u_region"
```

### Rich Data Features
This production data includes:
- **Location data**: City, country, region (for zombie analysis by geography)
- **Reassignment count**: For "ping pong" ticket detection
- **Priority levels**: For risk scoring
- **CMDB CI**: Configuration item tracking
- **Custom fields**: `u_ci_type`, `u_resolved`, `u_one_touch`

### Data Quality
- **Real production data** from EMEA region ServiceNow instance
- **2025 data** (current year) for relevant patterns
- **Closed/Resolved tickets** for training ML models
- **Multi-region**: UK, multiple European locations

### Loading Mechanism
```python
# data_loader.py
class DataLoader:
    def load_incidents(self, uploaded_file=None):
        # Pattern: *IM*.csv
        files = self.get_files('*IM*.csv')
        df = pd.read_csv(files[0])
        # Date conversion, normalization...
        return df
```

**Use Case**: "Offline Data" mode in Streamlit UI

---

## 3. 🌐 Live ServiceNow API (Production)

### Location
`snow_connector.py`

### API Endpoints
```python
# ServiceNow REST API v2 Table API
BASE_URL/api/now/table/{table_name}

Tables queried:
- incident
- problem
- change_request
```

### Authentication
- **HTTP Basic Auth** (username + password)
- Credentials from environment variables or UI input:
  ```python
  instance_url = os.getenv("SNOW_INSTANCE_URL")
  username = os.getenv("SNOW_USERNAME")
  password = os.getenv("SNOW_PASSWORD")
  ```

### Query Parameters
```python
params = {
    "sysparm_limit": 500,  # Max records per request
    "sysparm_display_value": "true",  # Readable values
    "sysparm_query": "opened_at>=javascript:gs.daysAgo(30)"  # Time filter
}
```

### Time Windows
- **Incidents**: Last 30 days
- **Problems**: Last 90 days
- **Changes**: Last 30 days (closed)

### Real-time Data Flow
```python
# snow_connector.py
class ServiceNowClient:
    def get_incidents(self, days_back=30, limit=500):
        query = f"opened_at>=javascript:gs.daysAgo({days_back})"
        return self.fetch_table_data('incident', limit, query)
```

**Use Case**: "Live API (Real)" mode in Streamlit UI

---

## 🧠 How ML Models Use This Data

### 1. **DBSCAN Clustering** (Pattern Detection)
**Data Source**: Incidents (all sources)
**Input Fields**: `short_description` + `description`
**Process**:
```python
# analysis.py - Lines 72-103
def perform_clustering(df):
    # 1. Combine text fields
    df['combined_text'] = df['short_description'] + " " + df['description']

    # 2. TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined_text'])

    # 3. DBSCAN clustering (cosine similarity)
    dbscan = DBSCAN(eps=0.5, min_samples=2, metric='cosine')
    clusters = dbscan.fit_predict(tfidf_matrix)

    return df.with_cluster_ids
```

**Output**: Groups similar incidents into clusters (-1 = noise, 0+ = cluster ID)

**Used In**:
- Investigation Deck: Full clustering analysis
- Current Risks: Open incident clustering
- AI Intelligence: Problem creation suggestions

---

### 2. **RandomForest Classification** (Intelligent Routing)
**Data Source**: Resolved/Closed incidents
**Input Fields**: `short_description` + `description` → TF-IDF features
**Target Label**: `assignment_group`

**Process**:
```python
# aiops_intelligence.py - Lines 135-192
def train(self, historical_df):
    # 1. Filter resolved incidents
    resolved = historical_df[historical_df['state'].isin(['Closed', 'Resolved'])]

    # 2. Text vectorization
    vectorizer = TfidfVectorizer(stop_words='english', max_features=300, min_df=2)
    X = vectorizer.fit_transform(resolved['combined_text'])

    # 3. Train model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, resolved['assignment_group'])

    return training_metrics
```

**Requirements**: Minimum 10 resolved tickets with assignment groups
**Output**: Predicts top 3 assignment groups with confidence scores

**Training Data Size**:
- Mock data: ~20 closed incidents
- EMEA CSV: ~9,000 incidents (plenty for training!)
- Live API: Last 30 days (typically 100-1000)

---

### 3. **Isolation Forest** (Anomaly Detection)
**Data Source**: Incidents (time-series daily counts)
**Input Fields**: `opened_at` (aggregated by day)

**Process**:
```python
# analysis.py - Lines 196-261
def detect_volume_spike(incidents_df):
    # 1. Group by day
    daily_counts = incidents_df.groupby('date').size()

    # 2. Train IsolationForest
    clf = IsolationForest(contamination=0.05)
    clf.fit(daily_counts.reshape(-1, 1))

    # 3. Detect today's spike
    is_spike = clf.predict(today_count) == -1

    return is_spike, daily_counts
```

**Output**: Boolean (spike detected or not) + time series for visualization

---

### 4. **TF-IDF Similarity** (Similar Incident Matching)
**Data Source**: Resolved incidents
**Input Fields**: `short_description` + `description` + `close_notes`

**Process**:
```python
# aiops_intelligence.py
def find_similar_resolved_incidents(incident_desc, incidents_df, top_n=5):
    # 1. Vectorize all resolved incidents
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(resolved['combined_text'])

    # 2. Vectorize query incident
    query_vec = vectorizer.transform([incident_desc])

    # 3. Calculate cosine similarity
    similarities = cosine_similarity(query_vec, vectors)

    # 4. Return top N matches
    return top_matches_with_resolutions
```

**Output**: Top 5 similar incidents with resolution notes and MTTR

---

### 5. **Regex Entity Extraction** (Zombie Detection)
**Data Source**: Incidents and Problems
**Input Fields**: `short_description` (text mining)

**Process**:
```python
# analysis.py - Lines 7-36
def extract_entities(text):
    # IP addresses: r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    # Server names: r'\b(?=.*\d)(?=.*[a-zA-Z])[a-zA-Z0-9-]{3,}\b'
    # 6-digit IDs: r'\b\d{6}\b'

    return list_of_entities
```

**No ML training required** - rule-based pattern matching

---

## 📊 Data Quality Requirements

### For ML Models to Work:

| Model | Min Records | Required Fields | Training Time |
|-------|-------------|-----------------|---------------|
| **DBSCAN Clustering** | 5+ incidents | short_description | ~1-5 seconds |
| **Intelligent Routing** | 10+ resolved | assignment_group, state | ~2-10 seconds |
| **Anomaly Detection** | 5+ days | opened_at | ~1 second |
| **Similar Incident** | 1+ resolved | short_description, close_notes | ~1-3 seconds |
| **Entity Extraction** | 1+ records | short_description | Instant (regex) |

### Current Data Availability:

✅ **Mock JSON**: Works for all models (small scale)
✅ **EMEA CSV**: Excellent for all models (9,000+ incidents!)
✅ **Live API**: Works if instance has sufficient history (30 days = ~100-1000 incidents typically)

---

## 🔄 Data Flow in Application

### User Selects Data Mode:

```python
# app.py - Lines 38-195
data_mode = st.sidebar.selectbox("Select Data Source", [
    "Live API (Mock)",     # → Uses incidents.json
    "Live API (Real)",     # → Calls ServiceNow API
    "Offline Data"         # → Loads CSV files
])
```

### Data Loading:
```
1. Mode Selection
   ↓
2. Data Loader executes
   ↓
3. DataFrames created:
   - df_cleaned (incidents)
   - problems_df (problems)
   - changes_df (changes)
   ↓
4. Stored in st.session_state
   ↓
5. Available to all ML models
```

### Data Persistence:
```python
# app.py - Lines 54-59
if 'inc_df' in st.session_state:
    df_cleaned = st.session_state['inc_df']
if 'prb_df' in st.session_state:
    problems_df = st.session_state['prb_df']
if 'chg_df' in st.session_state:
    changes_df = st.session_state['chg_df']
```

---

## 🎯 Which Data Source for Which Purpose?

### Development/Testing:
**Use**: Mock JSON (`incidents.json`)
- Fast iteration
- Known data for debugging
- Demo presentations

### Production POC:
**Use**: EMEA CSV files
- Real production data
- Large volume for accurate ML
- Offline analysis (no API dependency)

### Live Production:
**Use**: ServiceNow API
- Real-time monitoring
- Current operational data
- Requires credentials and network access

---

## 🔐 Data Security & Privacy

### Sensitive Data Handling:
- **No credentials in code**: Uses environment variables
- **CSV files**: Already exported (de-identified?)
- **API calls**: HTTPS with BasicAuth
- **Local processing**: All ML runs locally, no external APIs

### PII Considerations:
The current data includes:
- Ticket descriptions (may contain user names)
- Location data (cities, countries)
- Assignment groups (team names)

**For production**: Apply data masking/anonymization as needed.

---

## 📈 Data Volume Impact on Performance

| Data Source | Records | Clustering Time | UI Load Time | Recommendation |
|-------------|---------|-----------------|--------------|----------------|
| Mock JSON | ~30 | <1 sec | Instant | ✅ Fast demos |
| EMEA CSV | ~9,000 | 5-15 sec | 2-3 sec | ✅ Production POC |
| Live API | 100-1000 | 2-8 sec | 3-5 sec | ✅ Real-time monitoring |

**Optimization**: Results are cached in `st.session_state` to avoid re-clustering on every interaction.

---

## 🚀 Future Data Source Enhancements

### Proposed:
1. **Database integration** (PostgreSQL, MongoDB)
2. **Multiple ServiceNow instances** (multi-tenant)
3. **Historical data warehouse** (for trend analysis over years)
4. **Streaming data** (real-time incident ingestion)
5. **External data sources** (monitoring tools, logs)

---

## Summary

**Primary Data Source**: ServiceNow-formatted data (Incidents, Problems, Changes)

**Three Modes**:
1. **Mock JSON** - Development/demo (30 records)
2. **CSV Export** - Production POC (9,000+ records)
3. **Live API** - Real-time monitoring (configurable window)

**ML Training**: All models train on historical resolved/closed incidents with:
- Text descriptions for NLP/clustering
- Assignment groups for classification
- Timestamps for anomaly detection
- Resolution notes for similarity matching

**Current Status**: Fully functional with all three data sources. EMEA CSV provides the best training data volume for accurate ML models.

---

**For Gemini Implementation**: The clustering enhancements will use the same data sources without requiring new data. Just leverage existing `df_cleaned` and `problems_df` DataFrames!
