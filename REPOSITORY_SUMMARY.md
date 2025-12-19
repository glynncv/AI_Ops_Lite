# AI_Ops_Lite - Repository Summary

## 🎯 What This Repository Does

**AI_Ops_Lite** is an AI-powered IT Operations Analytics platform that helps identify, analyze, and prevent recurring IT incidents by using machine learning to find hidden patterns in ServiceNow data.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     AI_OPS_LITE FLIGHT DECK                     │
│                     (Streamlit Web Interface)                    │
└────────────────────┬────────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌───────────────┐         ┌──────────────┐
│  Live API     │         │  Offline CSV │
│  (ServiceNow) │         │  Upload Mode │
└───────┬───────┘         └──────┬───────┘
        │                        │
        └────────────┬───────────┘
                     ▼
        ┌────────────────────────┐
        │   Data Processor       │
        │  - Incidents           │
        │  - Problems            │
        │  - Change Requests     │
        └────────┬───────────────┘
                 │
                 ▼
    ┌────────────────────────────┐
    │   AI Analysis Engine       │
    ├────────────────────────────┤
    │ • ML Clustering (DBSCAN)   │
    │ • Anomaly Detection        │
    │ • Pattern Recognition      │
    │ • Root Cause Analysis      │
    │ • Deflection Opportunities │
    └────────┬───────────────────┘
                 │
                 ▼
    ┌────────────────────────────┐
    │   Output & Insights        │
    ├────────────────────────────┤
    │ • Flash Reports            │
    │ • Visual Dashboards        │
    │ • Communication Templates  │
    │ • Cost Savings Analysis    │
    └────────────────────────────┘
```

---

## 🔑 Core Features

### 1. 🔴 Real-Time Risk Monitor

```
┌──────────────────┬──────────────────┬──────────────────┐
│  Volume Spike    │  Hidden Clusters │  Root Cause      │
│  Detection       │  (Open Issues)   │  Correlation     │
├──────────────────┼──────────────────┼──────────────────┤
│ Uses ML to detect│ Groups similar   │ Links clusters   │
│ abnormal spikes  │ open incidents   │ to recent        │
│ in ticket volume │ using TF-IDF +   │ changes within   │
│                  │ DBSCAN           │ 48h window       │
└──────────────────┴──────────────────┴──────────────────┘
```

**Algorithms Used:**
- **Isolation Forest**: Detects volume anomalies
- **DBSCAN Clustering**: Groups similar incidents
- **TF-IDF Vectorization**: Text similarity analysis
- **Keyword Matching**: Correlates changes with incidents

### 2. 🔍 Investigation Deck (Deep Dive)

**Full Clustering Analysis**
- Analyzes ALL incidents (not just open ones)
- Identifies duplicate patterns across history
- Uses cosine similarity for text matching

**Repeat Offenders Detection**
- Extracts entities from incident descriptions:
  - IP Addresses (e.g., `192.168.1.1`)
  - Server Names (e.g., `web-server-01`)
  - Asset IDs (e.g., `NYC-DB-09`)
- Flags entities appearing in multiple incidents

**Change Correlation (Row-by-Row)**
- 48-hour lookback window
- Matches incident keywords with change descriptions
- Highlights suspect changes that may have caused incidents

### 3. ⚡ Executive Flash Report

```
╔═══════════════════════════════════════╗
║   EXECUTIVE FLASH REPORT              ║
╠═══════════════════════════════════════╣
║ ⚠️  Operational Risk: High/Medium/Low ║
║ 📊 Active Clusters: X                 ║
║ 🔥 Chronic Sites: [Entity1, Entity2]  ║
║ 💰 Deflection Potential: X tickets    ║
╚═══════════════════════════════════════╝
```

### 4. 📅 Retro Audit ("Back to the Future")

**The Timeline Fusion**
```
TIME ───────────────────────────────────────────────>

  ●           ●    ●         ●              ●     Incidents (Blue Dots)
    ━━━━━━━━           ━━━━━━━━━━━         ━━━   Problems (Red Lines)
```
- Visualizes incidents and problems on a timeline
- Shows temporal relationships
- Identifies problem resolution patterns

**Zombie Problems**
- Entities with >1 Problem Record in 12 months
- Groups by location or CMDB CI
- Highlights chronic infrastructure issues

**Deflection Opportunity**
```
┌─────────────────────────────────────────┐
│ Keywords: password, reset, access,      │
│          login, unlock, account         │
├─────────────────────────────────────────┤
│ Deflectable Tickets: X                  │
│ Potential Savings: $XX,XXX              │
│ (@ $50 per ticket)                      │
└─────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
AI_Ops_Lite/
│
├── app.py                      # Main Streamlit application
├── aiops_master.py             # Alternative simpler UI
│
├── analysis.py                 # Core ML/AI analysis functions
│   ├── perform_clustering()    # DBSCAN clustering
│   ├── detect_volume_spike()   # Anomaly detection
│   ├── cluster_open_incidents()
│   ├── correlate_cluster_causes()
│   └── find_suspect_changes()
│
├── retro_analysis.py           # Historical analysis
│   ├── create_timeline_fusion_chart()
│   ├── identify_zombie_problems()
│   └── calculate_deflection_opportunity()
│
├── data_loader.py              # CSV data loading
├── data_processor.py           # Data cleaning & processing
├── snow_connector.py           # ServiceNow API mock connector
├── utils.py                    # Communication templates
│
├── tests/
│   └── test_browser_nav.py
│
└── data/
    ├── input/
    │   ├── incidents.json
    │   ├── problems.json
    │   ├── changes.json
    │   └── *.csv files (EMEA data)
    └── output/
```

---

## 🛠️ Technology Stack

```
┌─────────────────────────────────────────┐
│ Frontend                                │
│ • Streamlit (Web UI)                    │
│ • Plotly (Interactive Charts)           │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Data Processing                         │
│ • Pandas (DataFrames)                   │
│ • NumPy (Numerical Operations)          │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Machine Learning                        │
│ • scikit-learn (Clustering, Anomalies)  │
│   - DBSCAN (Density-based clustering)   │
│   - IsolationForest (Anomaly detection) │
│   - TfidfVectorizer (Text analysis)     │
│ • TextBlob (NLP)                        │
└─────────────────────────────────────────┘
```

---

## 🎯 Use Cases

### 1. **Proactive Incident Prevention**
Identify patterns before they become major outages

### 2. **Root Cause Analysis**
Automatically correlate incidents with recent changes

### 3. **Cost Optimization**
Calculate deflection opportunities for L0 automation

### 4. **Executive Reporting**
Generate flash reports for stakeholders

### 5. **Quality Auditing**
Identify chronic infrastructure issues ("Zombie Problems")

---

## 📊 Sample Workflow

```
1. DATA INGESTION
   ↓
   Load incidents, problems, and changes
   (Live API or CSV upload)

2. REAL-TIME MONITORING
   ↓
   • Detect volume spikes
   • Cluster open incidents
   • Find suspect changes

3. DEEP ANALYSIS
   ↓
   • Full historical clustering
   • Identify repeat offenders
   • Row-by-row correlation

4. REPORTING
   ↓
   • Generate flash report
   • Create timeline visualizations
   • Calculate cost savings

5. ACTION
   ↓
   • Generate communication templates
   • Prioritize high-risk clusters
   • Plan preventive measures
```

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the main application
streamlit run app.py

# Or run the simpler version
streamlit run aiops_master.py
```

---

## 📈 Key Metrics Tracked

| Metric | Description | Impact |
|--------|-------------|--------|
| **Operational Risk** | High/Medium/Low based on clusters + repeat offenders | Strategic Planning |
| **Active Clusters** | Number of incident groups with similar patterns | Issue Prioritization |
| **Chronic Sites** | Entities with recurring problems | Infrastructure Investment |
| **Deflection Potential** | Tickets that could be automated | Cost Savings |
| **Volume Spikes** | Abnormal increase in incidents | Early Warning System |

---

## 💡 Intelligence Features

### Pattern Recognition
```
Input: 100 incidents about "database timeout"
   ↓
ML Analysis: Groups 85 similar incidents into 3 clusters
   ↓
Output: "Cluster #1: DB-PROD-01 timeout (42 incidents)"
```

### Anomaly Detection
```
Normal daily incidents: 10-15
Today's incidents: 45
   ↓
Isolation Forest: SPIKE DETECTED
   ↓
Alert: "Abnormal High Volume Detected Today!"
```

### Root Cause Correlation
```
Incident: "API Gateway timeout" @ 2PM
   ↓
48h Lookback: Find Change "API Gateway upgrade" @ 12PM
   ↓
Keyword Match: "gateway", "timeout", "api"
   ↓
Output: "Suspect Change: CHG0012345"
```

---

## 🎨 Visual Dashboard Components

1. **Risk Monitor Cards** - Real-time metrics with color-coded alerts
2. **Line Charts** - Daily incident volume trends
3. **Data Tables** - Clustered incidents with details
4. **Timeline Fusion** - Interactive Plotly scatter plot
5. **Flash Report** - Executive summary in text format

---

## 🔮 Future Enhancement Opportunities

- Real ServiceNow API integration (currently mock)
- Predictive incident forecasting
- Automated ticket classification
- Slack/Teams integration for alerts
- Custom ML model training per environment
- Knowledge base integration for deflection

---

**Built for IT Operations Teams to work smarter, not harder.**
