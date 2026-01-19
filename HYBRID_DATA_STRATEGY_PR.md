# Product Requirement: Hybrid Data Strategy for Enhanced AIOps

**Document Type:** Product Requirement  
**Version:** 1.0  
**Date:** 2026-01-19  
**Author:** AIOps Platform Team  
**Status:** Proposed

---

## Executive Summary

This document proposes a **hybrid data strategy** that combines offline historical data with live ServiceNow data to maximize the effectiveness of AIOps features, particularly ML model training and incident intelligence.

**Current State:** The system operates in either offline mode (CSV) OR live mode (ServiceNow API), but doesn't leverage both simultaneously.

**Proposed State:** A hybrid mode that uses historical closed incidents for training while displaying current open incidents for triage.

---

## Problem Statement

### Current Limitations

1. **Live Mode Weakness:**
   - Fetches only recent incidents (last 30 days by default)
   - May not include enough resolved incidents for ML training
   - Users see "0 Resolved/Closed" and cannot train Intelligent Assignment Routing

2. **Offline Mode Weakness:**
   - Users must manually export and upload CSVs
   - No real-time data refresh
   - Cannot fetch additional data on-demand

3. **Analysis Findings:**
   - Offline historical data (`PYTHON EMEA IM (2025).csv`): **6,811 resolved incidents** (94% closed/resolved)
   - Live mode data: **500 incidents**, **0 resolved** (100% open: New, Awaiting Problem)
   - **Neither mode alone provides optimal data coverage**

---

## Proposed Solution: Hybrid Data Mode

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   AIOps Platform                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────┐      ┌──────────────────┐        │
│  │  Live Data       │      │  Historical Data  │        │
│  │  (Real-time)     │      │  (Training Corpus)│        │
│  ├──────────────────┤      ├──────────────────┤        │
│  │ • Open Incidents │      │ • Closed Incidents│        │
│  │ • Recent Changes │      │ • Resolution Notes│        │
│  │ • Active Problems│      │ • Assignment Data │        │
│  │                  │      │ • 90-365 days     │        │
│  └────────┬─────────┘      └─────────┬────────┘        │
│           │                          │                  │
│           └──────────┬───────────────┘                  │
│                      │                                  │
│           ┌──────────▼───────────┐                      │
│           │  Unified Data Store  │                      │
│           │  (Session State)     │                      │
│           └──────────┬───────────┘                      │
│                      │                                  │
│        ┌─────────────┼─────────────┐                    │
│        │             │             │                    │
│   ┌────▼────┐  ┌─────▼─────┐  ┌──▼────┐               │
│   │Triage & │  │ML Training│  │Pattern│               │
│   │Analysis │  │  (Router) │  │Detect │               │
│   └─────────┘  └───────────┘  └───────┘               │
└─────────────────────────────────────────────────────────┘
```

---

## Requirements

### 1. New Data Mode: "Hybrid (Live + Historical)"

**Functional Requirements:**

- **FR-1.1:** Add a new data mode option: "Hybrid (Live + Historical)"
- **FR-1.2:** When selected, automatically:
  - Fetch live open incidents from ServiceNow
  - Load historical closed incidents from configured CSV or ServiceNow archive
- **FR-1.3:** Display data source indicators:
  - Badge showing: "Live: 500 open | Historical: 6,811 closed"

**User Interface:**
```
Data Source Config
├── Mode: [ Hybrid (Live + Historical) ▼ ]
├── Live Connection: [ ✅ Connected to ServiceNow ]
└── Historical Data:
    ├── Source: [ CSV Upload | ServiceNow Archive ▼ ]
    └── Date Range: [ Last 90 days ▼ ]
```

---

### 2. Smart Data Loading Strategy

**Functional Requirements:**

- **FR-2.1:** On initial load:
  1. Connect to ServiceNow → fetch open incidents (state: New, In Progress, Awaiting Problem)
  2. Load historical data → fetch/upload closed incidents (state: Closed, Resolved, Autoclosed)
  3. Merge datasets with duplicate detection by `number` field

- **FR-2.2:** Background refresh:
  - Live data: refresh every 5 minutes (configurable)
  - Historical data: cache for session duration

- **FR-2.3:** Data age indicators:
  - Show "Last refreshed: 2 minutes ago" for live data
  - Show "Historical data: Dec 2024 - Jan 2025" for training corpus

**Performance:**
- Target load time: < 10 seconds for initial hybrid load
- Memory optimization: lazy load historical data only when ML features are accessed

---

### 3. Feature-Specific Data Routing

**Functional Requirements:**

| Feature | Data Source | Rationale |
|---------|-------------|-----------|
| **Incident Cluster Lookup** | Live + Historical | Show both current and past patterns |
| **Similar Incident Recommendation** | Historical only | Need resolved incidents with solutions |
| **Intelligent Assignment Routing (Training)** | Historical only | Requires closed incidents with verified assignments |
| **Intelligent Assignment Routing (Prediction)** | Both | Train on historical, predict for new incidents |
| **Proactive Problem Detection** | Live + Historical | Detect patterns across all data |
| **Volume Spike Detection** | Live data | Real-time monitoring |
| **Investigation Deck** | Live data | Current incident triage |
| **Flash Report** | Live data | Current state reporting |

**FR-3.1:** Each feature UI should display which dataset it's using:
```
🔍 Similar Incident Recommendation
Using: Historical Data (6,811 resolved incidents)
```

---

### 4. Training Data Management

**Functional Requirements:**

- **FR-4.1:** Training Data Status Panel (Enhanced):
  ```
  📊 Training Data Status
  ┌─────────────────────────────────────────┐
  │ Live Open Incidents:          500       │
  │ Historical Closed Incidents:  6,811     │
  │ ─────────────────────────────────────   │
  │ Total Dataset:                7,311     │
  │ Training Ready:               ✅ Yes    │
  └─────────────────────────────────────────┘
  ```

- **FR-4.2:** Auto-select training dataset:
  - If historical data available → use for training
  - If only live data → check if sufficient resolved incidents exist
  - Display warning if training data insufficient

- **FR-4.3:** Manual override option:
  - Checkbox: "Include open incidents in analysis" (default: off)
  - Use case: Large enterprises wanting to detect patterns in in-progress incidents

---

### 5. Historical Data Source Options

**FR-5.1:** Support multiple historical data sources:

1. **CSV Upload** (Current)
   - Manual upload of exported ServiceNow data
   - Suitable for: Air-gapped environments, compliance requirements

2. **ServiceNow Archive Query** (New)
   - API query for closed incidents with custom date range
   - Suitable for: Live environments with API access

3. **Persistent Cache** (New)
   - Store previously fetched historical data locally
   - Auto-refresh monthly or on-demand
   - Suitable for: Reducing API calls, offline capability

**FR-5.2:** Historical Data Configuration UI:
```
Historical Data Settings
├── Source: [ ServiceNow Archive ▼ ]
├── Date Range: [ Last ⏹ 90 ] days
├── Max Records: [ 10,000 ]
├── States to Include:
│   ☑ Closed
│   ☑ Resolved  
│   ☑ Autoclosed
│   ☐ Cancelled
└── [ Fetch Historical Data ] [ Clear Cache ]
```

---

### 6. User Workflows

#### Workflow 1: New User - First Time Setup
```
1. User selects "Hybrid (Live + Historical)"
2. System prompts: "Connect to ServiceNow?"
3. User enters credentials → connects
4. System auto-fetches:
   - 500 open incidents (last 30 days)
   - 2,000 closed incidents (last 90 days)
5. Dashboard shows: "✅ Ready! 2,500 incidents loaded"
6. All features enabled
```

#### Workflow 2: Power User - Optimized Training
```
1. User already connected to ServiceNow (live mode)
2. User navigates to "AI Intelligence" → "Intelligent Assignment Routing"
3. System detects: only 12 resolved incidents in live data
4. System prompts: "⚠️ Limited training data. Fetch historical data?"
5. User clicks "Fetch Historical Data"
6. System loads 6,000 closed incidents from ServiceNow (last 180 days)
7. Training Data Status updates: "✅ Ready with 6,012 resolved incidents"
8. User trains model successfully
```

#### Workflow 3: Compliance/Offline Environment
```
1. User selects "Hybrid (Live + Historical)"
2. User uploads historical data CSV (6,811 closed incidents)
3. User connects to ServiceNow for live data (500 open incidents)
4. System merges datasets intelligently
5. ML features train on historical, predictions apply to live
6. User can disconnect from ServiceNow and continue analysis
```

---

## Benefits

### Business Benefits

1. **Improved ML Accuracy:**
   - Training on 6,811 resolved incidents vs. 12 → **567x more training data**
   - Better assignment predictions → reduced MTTR

2. **Faster Time-to-Value:**
   - No waiting for sufficient resolved incidents to accumulate
   - Immediate ML feature enablement

3. **Cost Reduction:**
   - Fewer API calls (cache historical data)
   - Reduced manual data management

### User Experience Benefits

1. **No Mode Switching:**
   - Single view of open + historical incidents
   - Seamless workflow

2. **Clear Data Indicators:**
   - Always know which dataset is being used
   - Transparency builds trust in ML recommendations

3. **Flexibility:**
   - Choose data sources based on needs
   - Offline capability maintained

---

## Technical Considerations

### Data Merge Strategy

**Deduplication:**
- Key: `number` field (incident number)
- Rule: Keep most recent record if duplicates found
- Log: Track merge statistics for debugging

**State Conflicts:**
- If incident exists in both live and historical with different states:
  - Prefer live data state (source of truth)
  - Preserve historical state in `state_historical` column
  - Use case: Incident was closed, then reopened

### Performance Optimization

**Memory Management:**
```python
# Lazy load historical data only when needed
if user_accessing_ml_features():
    load_historical_data()
else:
    load_only_live_data()
```

**Caching:**
- Session-level cache for historical data
- Browser localStorage for persistent cache (optional)
- TTL: 24 hours for historical, 5 minutes for live

### Security & Compliance

**Data Handling:**
- Historical data may contain PII → apply same encryption as live data
- GDPR compliance: respect data retention policies
- Audit: Log which datasets are merged and when

---

## Implementation Phases

### Phase 1: Basic Hybrid Mode (MVP)
**Timeline:** 2 weeks

- Add "Hybrid" mode selector
- Implement CSV + Live merge logic
- Update Training Data Status panel
- Test with sample datasets

**Success Criteria:**
- User can load CSV + connect to ServiceNow simultaneously
- ML features use correct datasets
- No performance degradation

### Phase 2: ServiceNow Archive Integration
**Timeline:** 3 weeks

- Add "Fetch Historical Data" button with date range selector
- Implement background refresh
- Add data source indicators to UI
- Performance optimization

**Success Criteria:**
- Fetch 10,000+ incidents in <15 seconds
- Auto-detect and prompt when training data insufficient
- Clear UX indicators

### Phase 3: Advanced Features
**Timeline:** 4 weeks

- Persistent cache with auto-refresh
- Data quality metrics (completeness, freshness)
- Smart dataset recommendations per feature
- Historical data versioning

**Success Criteria:**
- Zero manual intervention for optimal data loading
- 95%+ user satisfaction with data availability
- <5% API call increase vs. current live mode

---

## Success Metrics

### Quantitative Metrics

| Metric | Current (Offline/Live) | Target (Hybrid) |
|--------|------------------------|-----------------|
| Average resolved incidents for training | 12 | 2,000+ |
| Time to first successful ML training | Never (insufficient data) | <2 minutes |
| User complaints about "not enough data" | ~40% of users | <5% |
| ML model training accuracy | N/A (can't train) | >80% |
| API calls per session | 3-5 (live) | 4-6 (live + 1-2 historical) |

### Qualitative Metrics

- User feedback: "Finally able to use ML features!"
- Support tickets: Reduction in "how do I train the model?" queries
- Feature adoption: Increase in Intelligent Assignment Routing usage

---

## Risks & Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Historical data too large (>50K incidents) | Performance degradation | Medium | Implement pagination, lazy loading |
| State conflicts between live/historical | Incorrect analysis | Low | Clear merge rules, prefer live data |
| User confusion about which data to use | Poor UX | Medium | Clear UI indicators, smart defaults |
| API rate limiting (fetching historical) | Slow initial load | Low | Batch requests, caching |
| GDPR compliance issues (old data) | Legal risk | Low | Respect retention policies, audit logs |

---

## Open Questions

1. **Default historical data range:**
   - Should we default to 90 days, 180 days, or let user choose?
   - **Recommendation:** 90 days (balances data volume vs. relevance)

2. **Auto-refresh frequency:**
   - How often to refresh historical data?
   - **Recommendation:** Monthly, or when cache expires (24 hours)

3. **Data retention in session:**
   - Should historical data persist across browser sessions?
   - **Recommendation:** Optional with user consent (localStorage)

4. **Incident state prioritization:**
   - What if same incident appears as "In Progress" (live) and "Closed" (historical)?
   - **Recommendation:** Always use live state, log discrepancy

---

## Appendix: Real-World Example

### Before (Live Mode Only)
```
ServiceNow Query: Last 30 days, all incidents
Results: 500 incidents
- State: New (497), Awaiting Problem (3)
- Resolved: 0

ML Training Status: ❌ Cannot train (need 10+ resolved)
User Action: Give up or manually export historical data
```

### After (Hybrid Mode)
```
Live Query: Last 30 days, open incidents
Results: 500 incidents (New, Awaiting Problem)

Historical Load: CSV or API query for closed incidents
Results: 6,811 incidents (Closed, Resolved, Autoclosed)

Combined Dataset: 7,311 incidents
- Open: 500 (for current triage)
- Closed: 6,811 (for ML training)

ML Training Status: ✅ Ready (6,811 resolved incidents)
User Action: Click "Train Model" → Success!
```

---

## Conclusion

The **Hybrid Data Strategy** represents a significant enhancement to the AIOps platform by addressing the fundamental tension between real-time operational data and historical training data. By intelligently combining both, we enable:

1. **Immediate ML feature availability** (no waiting for data accumulation)
2. **Better model accuracy** (trained on large historical corpus)
3. **Real-time incident intelligence** (applied to current open incidents)
4. **Flexible deployment options** (works in connected and air-gapped environments)

**Recommendation:** Proceed with phased implementation, starting with MVP (Phase 1) to validate approach with users.

---

**Document End**
