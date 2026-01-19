# Manual Test Plan: `feature/ui-and-logic-enhancements` Branch

**Branch:** `feature/ui-and-logic-enhancements`  
**Test Date:** _______________  
**Tester:** _______________  
**Base Branch:** `origin/main`  
**Commits Ahead:** 6 commits

---

## Prerequisites

- [ ] Git branch checked out: `feature/ui-and-logic-enhancements`
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Test data files exist in `data/input/`:
  - [ ] `incidents.json`
  - [ ] `changes.json`
  - [ ] `problems.json`
- [ ] Logs directory exists: `logs/` (will be created automatically)
- [ ] Python environment: Python 3.13+ recommended

---

## Test Execution Instructions

1. Start the Streamlit app: `streamlit run app.py`
2. Open browser to: `http://localhost:8501`
3. Follow each test section in order
4. Check off items as you complete them
5. Note any issues or observations in the "Notes" column
6. Take screenshots of any issues found

---

## Test 1: Application Startup

**Objective:** Verify the app starts without errors and displays correctly

| Step | Action | Expected Result | Pass/Fail | Notes |
|------|--------|----------------|-----------|-------|
| 1.1 | Launch app with `streamlit run app.py` | App starts, no errors in console | ☐ | |
| 1.2 | Check browser loads | Page loads at `http://localhost:8501` | ☐ | |
| 1.3 | Verify title | Title shows "AIOps Lite: Flight Deck" | ☐ | |
| 1.4 | Check sidebar | Sidebar shows "Data Source Config" with selectbox | ☐ | |
| 1.5 | Verify no error messages | No red error messages displayed | ☐ | |
| 1.6 | Check console output | No Python tracebacks or critical errors | ☐ | |

**Result:** ☐ PASS  ☐ FAIL

---

## Test 2: Data Loading - Mock Mode

**Objective:** Verify data loads correctly in Mock mode

| Step | Action | Expected Result | Pass/Fail | Notes |
|------|--------|----------------|-----------|-------|
| 2.1 | Select "Live API (Mock)" from sidebar dropdown | Dropdown shows selected option | ☐ | |
| 2.2 | Wait for data to load | Sidebar shows success messages | ☐ | |
| 2.3 | Check incidents loaded | "Loaded X Incidents" message appears | ☐ | |
| 2.4 | Check changes loaded | "Loaded X Changes" message appears | ☐ | |
| 2.5 | Check problems loaded | "Loaded X Problems" message appears (if available) | ☐ | |
| 2.6 | Verify data appears in UI | Main content area shows data/metrics | ☐ | |
| 2.7 | Check session state | Data persists when switching tabs | ☐ | |

**Result:** ☐ PASS  ☐ FAIL

---

## Test 3: Current Risks Tab

**Objective:** Verify real-time risk monitoring features

| Step | Action | Expected Result | Pass/Fail | Notes |
|------|--------|----------------|-----------|-------|
| 3.1 | Click on "🔴 Current Risks" tab | Tab becomes active | ☐ | |
| 3.2 | Verify header | Header shows "Real-Time Risk Monitor" | ☐ | |
| 3.3 | Check Volume Monitor (Column 1) | Shows: | ☐ | |
| 3.3a | Subheader | "Volume Monitor" | ☐ | |
| 3.3b | Metric | Daily Volume metric with status | ☐ | |
| 3.3c | Chart | Line chart showing daily counts | ☐ | |
| 3.4 | Check Hidden Clusters (Column 2) | Shows: | ☐ | |
| 3.4a | Subheader | "Hidden Clusters (Open)" | ☐ | |
| 3.4b | Cluster table | DataFrame with Cluster ID and Count | ☐ | |
| 3.5 | Check Suspect Root Causes (Column 3) | Shows: | ☐ | |
| 3.5a | Subheader | "Suspect Root Causes" | ☐ | |
| 3.5b | Correlation results | Table with Cluster, Change, Reason | ☐ | |
| 3.6 | Check Active Clusters Detail | Section below shows detailed cluster data | ☐ | |
| 3.7 | Test Solution Recommender | Scroll to "💡 Agent Assist" section | ☐ | |
| 3.7a | Select incident dropdown | Dropdown shows list of open incidents | ☐ | |
| 3.7b | Select an incident | Incident details display | ☐ | |
| 3.7c | Click "🔍 Find Recommended Fixes" | Spinner appears, then results show | ☐ | |
| 3.7d | Verify recommendations | Shows similar incidents with resolution notes | ☐ | |

**Result:** ☐ PASS  ☐ FAIL

---

## Test 4: Investigation Deck Tab

**Objective:** Verify deep dive analysis features

| Step | Action | Expected Result | Pass/Fail | Notes |
|------|--------|----------------|-----------|-------|
| 4.1 | Click "🔍 Investigation Deck" tab | Tab becomes active | ☐ | |
| 4.2 | Verify header | Header shows "Deep Dive Analysis" | ☐ | |
| 4.3 | Check "Full Clustering" expander | Expandable section visible | ☐ | |
| 4.4 | Expand "Full Clustering" | Shows clustering results with details | ☐ | |
| 4.5 | Check "Repeat Offenders" expander | Expandable section visible | ☐ | |
| 4.6 | Expand "Repeat Offenders" | Shows recurring entities (sites/users) | ☐ | |
| 4.7 | Check "Incident-Change Correlation" expander | Expandable section visible | ☐ | |
| 4.8 | Expand "Incident-Change Correlation" | Shows correlation results | ☐ | |
| 4.9 | Verify data persistence | Switch tabs and return - data still visible | ☐ | |

**Result:** ☐ PASS  ☐ FAIL

---

## Test 5: AI Intelligence Tab

**Objective:** Verify AI-powered features work correctly

### 5.1 Similar Incident Recommendation

| Step | Action | Expected Result | Pass/Fail | Notes |
|------|--------|----------------|-----------|-------|
| 5.1.1 | Scroll to "1️⃣ Similar Incident Recommendation" | Section visible | ☐ | |
| 5.1.2 | Select incident from dropdown | Dropdown shows incident list | ☐ | |
| 5.1.3 | Click "Find Similar Incidents" button | Spinner appears | ☐ | |
| 5.1.4 | Wait for results | Results appear with: | ☐ | |
| 5.1.4a | Success message | "Found X similar resolved incidents!" | ☐ | |
| 5.1.4b | Expandable results | Each result shows in expander | ☐ | |
| 5.1.4c | Similarity score | Shows percentage match | ☐ | |
| 5.1.4d | Resolution notes | Shows resolution details | ☐ | |
| 5.1.5 | Check MTTR improvement | Info box shows time savings potential | ☐ | |

### 5.2 Intelligent Assignment Routing

| Step | Action | Expected Result | Pass/Fail | Notes |
|------|--------|----------------|-----------|-------|
| 5.2.1 | Scroll to "2️⃣ Intelligent Assignment Routing" | Section visible | ☐ | |
| 5.2.2 | Click "Train Assignment Model" button | Spinner appears | ☐ | |
| 5.2.3 | Wait for training | Training completes (may take 10-30 seconds) | ☐ | |
| 5.2.4 | Verify training success | Success message shows: | ☐ | |
| 5.2.4a | "✅ Model trained on X incidents" | Message appears | ☐ | |
| 5.2.4b | Training accuracy | Shows percentage | ☐ | |
| 5.2.4c | Assignment groups count | Shows number of groups | ☐ | |
| 5.2.5 | Enter test description | Type in text area: "VPN connection failed" | ☐ | |
| 5.2.6 | Click "Predict Assignment" button | Predictions appear | ☐ | |
| 5.2.7 | Verify predictions | Shows: | ☐ | |
| 5.2.7a | Top recommendation | Shows assignment group with 🥇 | ☐ | |
| 5.2.7b | Confidence score | Progress bar and percentage | ☐ | |
| 5.2.7c | Reasoning | Explanation text | ☐ | |
| 5.2.7d | Alternatives | Additional recommendations listed | ☐ | |

### 5.3 Proactive Problem Detection

| Step | Action | Expected Result | Pass/Fail | Notes |
|------|--------|----------------|-----------|-------|
| 5.3.1 | Scroll to "3️⃣ Proactive Problem Detection" | Section visible | ☐ | |
| 5.3.2 | Adjust threshold slider | Set to 5 incidents | ☐ | |
| 5.3.3 | Click "Analyze Clusters for Problem Opportunities" | Spinner appears | ☐ | |
| 5.3.4 | Wait for analysis | Analysis completes | ☐ | |
| 5.3.5 | Verify problem suggestions | If clusters found: | ☐ | |
| 5.3.5a | Success message | "🔥 Found X cluster(s) that should have Problem Records!" | ☐ | |
| 5.3.5b | Problem suggestion expanders | Each suggestion in expander | ☐ | |
| 5.3.5c | Problem details | Shows incident count, time span, priority | ☐ | |
| 5.3.5d | Related incidents | Lists incident numbers | ☐ | |
| 5.3.5e | Recommended actions | Shows action items | ☐ | |
| 5.3.6 | Click "Create Problem Record (Draft)" | Problem record details display | ☐ | |

### 5.4 Communication Assistant

| Step | Action | Expected Result | Pass/Fail | Notes |
|------|--------|----------------|-----------|-------|
| 5.4.1 | Scroll to "4️⃣ Communication Assistant" | Section visible | ☐ | |
| 5.4.2 | Select incident from dropdown | Dropdown shows incident list | ☐ | |
| 5.4.3 | Enter impact details | Type in text area: "Service outage affecting 50 users" | ☐ | |
| 5.4.4 | Click "Generate Template" button | Template appears | ☐ | |
| 5.4.5 | Verify template | Shows formatted communication template | ☐ | |

**Result:** ☐ PASS  ☐ FAIL

---

## Test 6: Monitoring & ROI Tab

**Objective:** Verify monitoring dashboard and ROI tracking

| Step | Action | Expected Result | Pass/Fail | Notes |
|------|--------|----------------|-----------|-------|
| 6.1 | Click "📊 Monitoring & ROI" tab | Tab becomes active | ☐ | |
| 6.2 | Verify header | Header shows "📊 Platform Monitoring & ROI Tracking" | ☐ | |
| 6.3 | Check ROI Summary section | Shows metrics: | ☐ | |
| 6.3a | Incidents Analyzed | Metric displayed | ☐ | |
| 6.3b | Patterns Detected | Metric displayed | ☐ | |
| 6.3c | Time Saved | Metric displayed in hours | ☐ | |
| 6.3d | Cost Saved | Metric displayed in USD | ☐ | |
| 6.4 | Check ML Model Performance | Section visible | ☐ | |
| 6.5 | Expand ML feature | Shows acceptance rates by feature | ☐ | |
| 6.6 | Check System Performance | Section visible | ☐ | |
| 6.7 | Verify performance metrics | Shows P50, P95, average, max | ☐ | |
| 6.8 | Check Error Tracking | Section visible | ☐ | |
| 6.9 | Verify error summary | Shows error counts by type | ☐ | |
| 6.10 | Check User Activity | Section visible | ☐ | |
| 6.11 | Check Audit Trail | Section visible with recent events | ☐ | |
| 6.12 | Click "Export Full Monitoring Report" | JSON report displays | ☐ | |
| 6.13 | Verify report content | Report contains all sections | ☐ | |

**Result:** ☐ PASS  ☐ FAIL

---

## Test 7: War Room Mode (Major Incident Mode)

**Objective:** Verify Major Incident Mode functionality

| Step | Action | Expected Result | Pass/Fail | Notes |
|------|--------|----------------|-----------|-------|
| 7.1 | Find sidebar toggle | Look for "🔴 Major Incident Mode" toggle | ☐ | |
| 7.2 | Toggle ON | Click toggle to enable | ☐ | |
| 7.3 | Verify visual change | Page changes to red theme | ☐ | |
| 7.4 | Check header | Large red header: "🚨 WAR ROOM: MAJOR INCIDENT ACTIVE" | ☐ | |
| 7.5 | Verify header animation | Header pulses/animated | ☐ | |
| 7.6 | Check Active Major Incidents | Section shows open P1/P2 incidents | ☐ | |
| 7.7 | Check Velocity Meter (Column 1) | Shows "Incidents / Min" metric | ☐ | |
| 7.8 | Verify velocity calculation | Shows incidents per minute (last 30 mins) | ☐ | |
| 7.9 | Check Blast Radius (Column 2) | Shows "Blast Radius (Locations)" | ☐ | |
| 7.10 | Verify location data | Shows affected locations with ticket counts | ☐ | |
| 7.11 | Check Change Radar (Column 3) | Shows "Change Radar" | ☐ | |
| 7.12 | Verify recent changes | Shows changes closed in last 4 hours | ☐ | |
| 7.13 | Check Crisis Memory | Section visible | ☐ | |
| 7.14 | Verify search query | Auto-populated from top cluster | ☐ | |
| 7.15 | Enter search query | Type in search box | ☐ | |
| 7.16 | Click "Search Historical P1s" | Results appear | ☐ | |
| 7.17 | Verify results | Shows historical P1 records with resolutions | ☐ | |
| 7.18 | Verify dashboard disabled | Normal tabs not visible | ☐ | |
| 7.19 | Toggle OFF | Disable War Room mode | ☐ | |
| 7.20 | Verify normal view | Returns to normal dashboard view | ☐ | |

**Result:** ☐ PASS  ☐ FAIL

---

## Test 8: Retro Audit Page

**Objective:** Verify multi-page structure and Retro Audit functionality

| Step | Action | Expected Result | Pass/Fail | Notes |
|------|--------|----------------|-----------|-------|
| 8.1 | Check sidebar navigation | Look for page navigation or use URL | ☐ | |
| 8.2 | Navigate to Retro Audit | Click on "1_Retro_Audit" or navigate to `/1_Retro_Audit` | ☐ | |
| 8.3 | Verify page loads | Page shows "🔙 Phase 3: The 'Back to the Future' Retro" | ☐ | |
| 8.4 | Check sidebar upload | File uploaders visible | ☐ | |
| 8.5 | Verify upload options | Three uploaders: Incidents, Problems, Changes | ☐ | |
| 8.6 | Check Timeline Fusion | Section visible | ☐ | |
| 8.7 | Expand Timeline Fusion | Shows timeline chart | ☐ | |
| 8.8 | Check Zombie Problems | Section visible | ☐ | |
| 8.9 | Expand Zombie Problems | Shows zombie problem analysis | ☐ | |
| 8.10 | Check Deflection Opportunity | Section visible | ☐ | |
| 8.11 | Expand Deflection Opportunity | Shows deflection metrics | ☐ | |
| 8.12 | Navigate back to main page | Return to main app | ☐ | |
| 8.13 | Verify navigation works | Can switch between pages | ☐ | |

**Result:** ☐ PASS  ☐ FAIL

---

## Test 9: Flash Report

**Objective:** Verify Flash Report generation

| Step | Action | Expected Result | Pass/Fail | Notes |
|------|--------|----------------|-----------|-------|
| 9.1 | Find sidebar button | Look for "Generate Flash Report" button | ☐ | |
| 9.2 | Click "Generate Flash Report" | Flash report appears at bottom | ☐ | |
| 9.3 | Verify report header | Shows "⚡ Executive Flash Report" | ☐ | |
| 9.4 | Check report content | Shows: | ☐ | |
| 9.4a | Status | "🔴 High Risk" or "🟢 Stable" | ☐ | |
| 9.4b | Total Incidents | Count displayed | ☐ | |
| 9.4c | Open Incidents | Count displayed | ☐ | |
| 9.4d | Operational Risk | Risk level shown | ☐ | |
| 9.4e | Active Clusters | Cluster count | ☐ | |
| 9.4f | Chronic Sites | List of sites | ☐ | |
| 9.4g | Deflection Potential | Ticket count | ☐ | |

**Result:** ☐ PASS  ☐ FAIL

---

## Test 10: Data Loading - Offline Mode

**Objective:** Verify CSV upload functionality works

| Step | Action | Expected Result | Pass/Fail | Notes |
|------|--------|----------------|-----------|-------|
| 10.1 | Select "Offline Data" from sidebar | Dropdown changes | ☐ | |
| 10.2 | Check sidebar message | Shows "Mode: Offline Data (CSV)" | ☐ | |
| 10.3 | Expand "Upload Overrides" | Expander opens | ☐ | |
| 10.4 | Verify file uploaders | Three uploaders visible: | ☐ | |
| 10.4a | Upload Incidents (CSV) | File uploader visible | ☐ | |
| 10.4b | Upload Changes (CSV) | File uploader visible | ☐ | |
| 10.4c | Upload Problems (CSV) | File uploader visible | ☐ | |
| 10.5 | Upload test CSV (optional) | If CSV available, upload | ☐ | |
| 10.6 | Verify data loads | Success messages appear | ☐ | |
| 10.7 | Verify data displays | Data appears in tabs | ☐ | |

**Result:** ☐ PASS  ☐ FAIL

---

## Test 11: Logging Infrastructure Verification

**Objective:** Verify logging files are created and written

| Step | Action | Expected Result | Pass/Fail | Notes |
|------|--------|----------------|-----------|-------|
| 11.1 | Navigate to logs directory | `cd logs/` or check folder | ☐ | |
| 11.2 | Check log files exist | All 5 files present: | ☐ | |
| 11.2a | audit_trail.jsonl | File exists | ☐ | |
| 11.2b | business_events.jsonl | File exists | ☐ | |
| 11.2c | errors.jsonl | File exists | ☐ | |
| 11.2d | performance.jsonl | File exists | ☐ | |
| 11.2e | roi_metrics.jsonl | File exists | ☐ | |
| 11.3 | Use app features | Perform actions in app (AI predictions, etc.) | ☐ | |
| 11.4 | Check log file sizes | Files should have content | ☐ | |
| 11.5 | Verify log entries | Open one file, verify JSONL format | ☐ | |
| 11.6 | Check audit trail | Verify audit entries logged | ☐ | |
| 11.7 | Check performance logs | Verify performance metrics logged | ☐ | |
| 11.8 | Check ROI metrics | Verify ROI calculations logged | ☐ | |

**Result:** ☐ PASS  ☐ FAIL

---

## Test 12: Error Handling

**Objective:** Verify graceful error handling

| Step | Action | Expected Result | Pass/Fail | Notes |
|------|--------|----------------|-----------|-------|
| 12.1 | Switch to Mock mode | Select "Live API (Mock)" | ☐ | |
| 12.2 | Verify no errors | App continues to work | ☐ | |
| 12.3 | Switch between tabs rapidly | Click tabs quickly | ☐ | |
| 12.4 | Verify no crashes | App remains stable | ☐ | |
| 12.5 | Check console | No Python errors in terminal | ☐ | |
| 12.6 | Test with empty data | If possible, test with empty dataset | ☐ | |
| 12.7 | Verify error messages | Errors displayed gracefully | ☐ | |
| 12.8 | Check error logging | Errors logged to errors.jsonl | ☐ | |

**Result:** ☐ PASS  ☐ FAIL

---

## Test 13: Performance Testing

**Objective:** Verify performance characteristics

| Step | Action | Expected Result | Pass/Fail | Notes |
|------|--------|----------------|-----------|-------|
| 13.1 | Measure app startup | Time from command to browser load | ☐ | |
| 13.2 | Measure data loading | Time to load mock data | ☐ | |
| 13.3 | Measure ML training | Time for assignment model training | ☐ | |
| 13.4 | Measure tab switching | Time to switch between tabs | ☐ | |
| 13.5 | Check responsiveness | UI responds quickly to interactions | ☐ | |
| 13.6 | Monitor memory usage | Check for memory leaks (optional) | ☐ | |

**Performance Targets:**
- App startup: < 3 seconds
- Data loading: < 2 seconds
- ML training: < 30 seconds
- Tab switching: < 1 second

**Result:** ☐ PASS  ☐ FAIL

---

## Test Summary

### Overall Test Results

| Test Section | Status | Notes |
|--------------|--------|-------|
| Test 1: Application Startup | ☐ PASS ☐ FAIL | |
| Test 2: Data Loading - Mock Mode | ☐ PASS ☐ FAIL | |
| Test 3: Current Risks Tab | ☐ PASS ☐ FAIL | |
| Test 4: Investigation Deck Tab | ☐ PASS ☐ FAIL | |
| Test 5: AI Intelligence Tab | ☐ PASS ☐ FAIL | |
| Test 6: Monitoring & ROI Tab | ☐ PASS ☐ FAIL | |
| Test 7: War Room Mode | ☐ PASS ☐ FAIL | |
| Test 8: Retro Audit Page | ☐ PASS ☐ FAIL | |
| Test 9: Flash Report | ☐ PASS ☐ FAIL | |
| Test 10: Data Loading - Offline | ☐ PASS ☐ FAIL | |
| Test 11: Logging Infrastructure | ☐ PASS ☐ FAIL | |
| Test 12: Error Handling | ☐ PASS ☐ FAIL | |
| Test 13: Performance Testing | ☐ PASS ☐ FAIL | |

### Issues Found

**Critical Issues:**
- None: ☐

**Minor Issues:**
1. _________________________________________________
2. _________________________________________________
3. _________________________________________________

**Enhancement Suggestions:**
1. _________________________________________________
2. _________________________________________________
3. _________________________________________________

### Performance Observations

- App startup time: _______________ seconds
- Data loading time: _______________ seconds
- ML training time: _______________ seconds
- Overall responsiveness: ☐ Excellent  ☐ Good  ☐ Fair  ☐ Poor

### Browser Information

- Browser: _______________ (Chrome/Firefox/Edge/etc.)
- Version: _______________
- OS: _______________

### Final Recommendation

☐ **APPROVE** - Branch ready for merge  
☐ **CONDITIONAL APPROVE** - Minor issues noted  
☐ **REJECT** - Critical issues found  

**Tester Signature:** _______________  
**Date:** _______________

---

## Quick Reference: Tab Structure

1. **🔴 Current Risks** - Real-time risk monitoring
   - Volume Monitor
   - Hidden Clusters
   - Suspect Root Causes
   - Agent Assist (Solution Recommender)

2. **🔍 Investigation Deck** - Deep dive analysis
   - Full Clustering
   - Repeat Offenders
   - Incident-Change Correlation

3. **🧠 AI Intelligence** - AI-powered features
   - Similar Incident Recommendation
   - Intelligent Assignment Routing
   - Proactive Problem Detection
   - Communication Assistant

4. **📊 Monitoring & ROI** - Platform monitoring
   - ROI Summary
   - ML Model Performance
   - System Performance
   - Error Tracking
   - User Activity
   - Audit Trail
   - Export Report

**Additional Pages:**
- **Retro Audit** (`pages/1_Retro_Audit.py`) - Historical analysis

---

## Tips for Testing

- **Take screenshots** of any issues found
- **Note browser** and version used
- **Check console** for JavaScript errors (F12)
- **Check terminal** for Python errors
- **Test on different screen sizes** if possible
- **Try rapid clicking** to test stability
- **Verify log files** after using features
- **Test all data modes** (Mock, Real, Offline)

---

## Known Issues / Notes

- Altair deprecation warning (cosmetic only, doesn't affect functionality)
- Logging verification script was updated to use correct API

---

**End of Test Plan**


