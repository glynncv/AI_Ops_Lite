# Manual Test Plan: `claude/integrated-workflow-ui-NjR6U` Branch

**Branch:** `claude/integrated-workflow-ui-NjR6U`  
**Test Date:** _______________  
**Tester:** _______________  

---

## Prerequisites

- [ ] Git worktree created at: `../AI_Ops_Lite-test-integrated-workflow-ui`
- [ ] Navigate to worktree directory: `cd ../AI_Ops_Lite-test-integrated-workflow-ui`
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Test data files exist in `data/input/`:
  - [ ] `incidents.json`
  - [ ] `changes.json`
  - [ ] `problems.json`

---

## Test Execution Instructions

1. Start the Streamlit app: `streamlit run app.py`
2. Open browser to: `http://localhost:8501`
3. Follow each test section in order
4. Check off items as you complete them
5. Note any issues or observations in the "Notes" column

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

**Result:** ☐ PASS  ☐ FAIL

---

## Test 2: Data Loading - Mock Mode

**Objective:** Verify data loads correctly in Mock mode

| Step | Action | Expected Result | Pass/Fail | Notes |
|------|--------|----------------|-----------|-------|
| 2.1 | Select "Live API (Mock)" from sidebar dropdown | Dropdown shows selected option | ☐ | |
| 2.2 | Wait for data to load | Sidebar shows success messages: | ☐ | |
| 2.3 | Check incidents loaded | "Loaded X Incidents" message appears | ☐ | |
| 2.4 | Check changes loaded | "Loaded X Changes" message appears | ☐ | |
| 2.5 | Check problems loaded | "Loaded X Problems (Mock)" message appears | ☐ | |
| 2.6 | Verify data appears in UI | Main content area shows data/metrics | ☐ | |

**Result:** ☐ PASS  ☐ FAIL

---

## Test 3: Dashboard Tab (New Feature)

**Objective:** Verify new Dashboard tab displays metrics and quick actions

| Step | Action | Expected Result | Pass/Fail | Notes |
|------|--------|----------------|-----------|-------|
| 3.1 | Click on "🏠 Dashboard" tab | Dashboard tab becomes active | ☐ | |
| 3.2 | Verify header | Header shows "📊 Operations Overview" | ☐ | |
| 3.3 | Check metrics row | Four metrics displayed: | ☐ | |
| 3.3a | Total Incidents | Shows total count | ☐ | |
| 3.3b | Open | Shows open count with percentage delta | ☐ | |
| 3.3c | Closed | Shows closed count | ☐ | |
| 3.3d | Active Clusters | Shows cluster count | ☐ | |
| 3.4 | Check Current Status section | Left column shows: | ☐ | |
| 3.4a | Operational Risk | Shows "🔴 High Risk" or "🟢 Stable" | ☐ | |
| 3.4b | Volume Status | Shows spike status or "✅ Normal volume" | ☐ | |
| 3.4c | Active Clusters | Shows cluster count or "None detected" | ☐ | |
| 3.5 | Check Recent Trend section | Right column shows line chart | ☐ | |
| 3.6 | Check Quick Actions | Three buttons displayed: | ☐ | |
| 3.6a | "🚨 View Active Incidents" | Button visible and clickable | ☐ | |
| 3.6b | "🧠 Use AI Assistant" | Button visible and clickable | ☐ | |
| 3.6c | "📊 View Monitoring" | Button visible and clickable | ☐ | |
| 3.7 | Click "View Active Incidents" button | Info message appears | ☐ | |

**Result:** ☐ PASS  ☐ FAIL

---

## Test 4: Active Incidents Tab

**Objective:** Verify Active Incidents tab shows real-time risk monitoring

| Step | Action | Expected Result | Pass/Fail | Notes |
|------|--------|----------------|-----------|-------|
| 4.1 | Click "🚨 Active Incidents" tab | Tab becomes active | ☐ | |
| 4.2 | Verify header | Header shows "Real-Time Risk Monitor" | ☐ | |
| 4.3 | Check Volume Monitor (Column 1) | Shows: | ☐ | |
| 4.3a | Subheader | "Volume Monitor" | ☐ | |
| 4.3b | Metric | Daily Volume metric with status | ☐ | |
| 4.3c | Chart | Line chart showing daily counts | ☐ | |
| 4.4 | Check Hidden Clusters (Column 2) | Shows: | ☐ | |
| 4.4a | Subheader | "Hidden Clusters (Open)" | ☐ | |
| 4.4b | Cluster table | DataFrame with Cluster ID and Count | ☐ | |
| 4.5 | Check Suspect Root Causes (Column 3) | Shows: | ☐ | |
| 4.5a | Subheader | "Suspect Root Causes" | ☐ | |
| 4.5b | Correlation results | Table with Cluster, Change, Reason | ☐ | |
| 4.6 | Check Active Clusters Detail | Section below shows detailed cluster data | ☐ | |
| 4.7 | Test Solution Recommender | Scroll to "💡 Agent Assist" section | ☐ | |
| 4.7a | Select incident dropdown | Dropdown shows list of open incidents | ☐ | |
| 4.7b | Select an incident | Incident details display | ☐ | |
| 4.7c | Click "🔍 Find Recommended Fixes" | Spinner appears, then results show | ☐ | |
| 4.7d | Verify recommendations | Shows similar incidents with resolution notes | ☐ | |

**Result:** ☐ PASS  ☐ FAIL

---

## Test 5: AI Assistant Tab - Similar Incident Recommendation

**Objective:** Verify AI Intelligence feature 1 works correctly

| Step | Action | Expected Result | Pass/Fail | Notes |
|------|--------|----------------|-----------|-------|
| 5.1 | Click "🧠 AI Assistant" tab | Tab becomes active | ☐ | |
| 5.2 | Verify header | Header shows "🧠 AI Intelligence & Predictions" | ☐ | |
| 5.3 | Find "1️⃣ Similar Incident Recommendation" | Section visible | ☐ | |
| 5.4 | Select incident from dropdown | Dropdown shows incident list | ☐ | |
| 5.5 | Click "Find Similar Incidents" button | Spinner appears | ☐ | |
| 5.6 | Wait for results | Results appear with: | ☐ | |
| 5.6a | Success message | "Found X similar resolved incidents!" | ☐ | |
| 5.6b | Expandable results | Each result shows in expander | ☐ | |
| 5.6c | Similarity score | Shows percentage match | ☐ | |
| 5.6d | Resolution notes | Shows resolution details | ☐ | |
| 5.7 | Check MTTR improvement | Info box shows time savings potential | ☐ | |

**Result:** ☐ PASS  ☐ FAIL

---

## Test 6: AI Assistant Tab - Intelligent Assignment Routing

**Objective:** Verify AI Intelligence feature 2 works correctly

| Step | Action | Expected Result | Pass/Fail | Notes |
|------|--------|----------------|-----------|-------|
| 6.1 | Scroll to "2️⃣ Intelligent Assignment Routing" | Section visible | ☐ | |
| 6.2 | Click "Train Assignment Model" button | Spinner appears | ☐ | |
| 6.3 | Wait for training | Training completes (may take 10-30 seconds) | ☐ | |
| 6.4 | Verify training success | Success message shows: | ☐ | |
| 6.4a | "✅ Model trained on X incidents" | Message appears | ☐ | |
| 6.4b | Training accuracy | Shows percentage | ☐ | |
| 6.4c | Assignment groups count | Shows number of groups | ☐ | |
| 6.5 | Enter test description | Type in text area: "VPN connection failed" | ☐ | |
| 6.6 | Click "Predict Assignment" button | Predictions appear | ☐ | |
| 6.7 | Verify predictions | Shows: | ☐ | |
| 6.7a | Top recommendation | Shows assignment group with 🥇 | ☐ | |
| 6.7b | Confidence score | Progress bar and percentage | ☐ | |
| 6.7c | Reasoning | Explanation text | ☐ | |
| 6.7d | Alternatives | Additional recommendations listed | ☐ | |

**Result:** ☐ PASS  ☐ FAIL

---

## Test 7: AI Assistant Tab - Proactive Problem Detection

**Objective:** Verify AI Intelligence feature 3 works correctly

| Step | Action | Expected Result | Pass/Fail | Notes |
|------|--------|----------------|-----------|-------|
| 7.1 | Scroll to "3️⃣ Proactive Problem Detection" | Section visible | ☐ | |
| 7.2 | Adjust threshold slider | Set to 5 incidents | ☐ | |
| 7.3 | Click "Analyze Clusters for Problem Opportunities" | Spinner appears | ☐ | |
| 7.4 | Wait for analysis | Analysis completes | ☐ | |
| 7.5 | Verify problem suggestions | If clusters found: | ☐ | |
| 7.5a | Success message | "🔥 Found X cluster(s) that should have Problem Records!" | ☐ | |
| 7.5b | Problem suggestion expanders | Each suggestion in expander | ☐ | |
| 7.5c | Problem details | Shows incident count, time span, priority | ☐ | |
| 7.5d | Related incidents | Lists incident numbers | ☐ | |
| 7.5e | Recommended actions | Shows action items | ☐ | |
| 7.6 | Click "Create Problem Record (Draft)" | Problem record details display | ☐ | |

**Result:** ☐ PASS  ☐ FAIL

---

## Test 8: AI Assistant Tab - Communication Assistant

**Objective:** Verify AI Intelligence feature 4 works correctly

| Step | Action | Expected Result | Pass/Fail | Notes |
|------|--------|----------------|-----------|-------|
| 8.1 | Scroll to "4️⃣ Communication Assistant" | Section visible (or check Historical Analysis tab) | ☐ | |
| 8.2 | Select incident from dropdown | Dropdown shows incident list | ☐ | |
| 8.3 | Enter impact details | Type in text area: "Service outage affecting 50 users" | ☐ | |
| 8.4 | Click "Generate Template" button | Template appears | ☐ | |
| 8.5 | Verify template | Shows formatted communication template | ☐ | |

**Result:** ☐ PASS  ☐ FAIL

---

## Test 9: Historical Analysis Tab

**Objective:** Verify historical analysis features work correctly

| Step | Action | Expected Result | Pass/Fail | Notes |
|------|--------|----------------|-----------|-------|
| 9.1 | Click "📈 Historical Analysis" tab | Tab becomes active | ☐ | |
| 9.2 | Verify header | Header shows "Deep Dive Analysis" | ☐ | |
| 9.3 | Check "Full Clustering" expander | Expandable section visible | ☐ | |
| 9.4 | Expand "Full Clustering" | Shows clustering results | ☐ | |
| 9.5 | Check "Repeat Offenders" expander | Expandable section visible | ☐ | |
| 9.6 | Expand "Repeat Offenders" | Shows recurring entities | ☐ | |
| 9.7 | Check "Incident-Change Correlation" expander | Expandable section visible | ☐ | |
| 9.8 | Expand "Incident-Change Correlation" | Shows correlation results | ☐ | |

**Result:** ☐ PASS  ☐ FAIL

---

## Test 10: Monitoring & ROI Tab

**Objective:** Verify monitoring and ROI tracking works correctly

| Step | Action | Expected Result | Pass/Fail | Notes |
|------|--------|----------------|-----------|-------|
| 10.1 | Click "📊 Monitoring & ROI" tab | Tab becomes active | ☐ | |
| 10.2 | Verify header | Header shows "📊 Platform Monitoring & ROI Tracking" | ☐ | |
| 10.3 | Check ROI Summary section | Shows metrics: | ☐ | |
| 10.3a | Incidents Analyzed | Metric displayed | ☐ | |
| 10.3b | Patterns Detected | Metric displayed | ☐ | |
| 10.3c | Time Saved | Metric displayed in hours | ☐ | |
| 10.3d | Cost Saved | Metric displayed in USD | ☐ | |
| 10.4 | Check ML Model Performance | Section visible | ☐ | |
| 10.5 | Expand ML feature | Shows acceptance rates | ☐ | |
| 10.6 | Check System Performance | Section visible | ☐ | |
| 10.7 | Check Error Tracking | Section visible | ☐ | |
| 10.8 | Check User Activity | Section visible | ☐ | |
| 10.9 | Check Audit Trail | Section visible with recent events | ☐ | |
| 10.10 | Click "Export Full Monitoring Report" | JSON report displays | ☐ | |

**Result:** ☐ PASS  ☐ FAIL

---

## Test 11: Reports Tab

**Objective:** Verify Reports tab and Flash Report generation

| Step | Action | Expected Result | Pass/Fail | Notes |
|------|--------|----------------|-----------|-------|
| 11.1 | Click "📄 Reports" tab | Tab becomes active | ☐ | |
| 11.2 | Verify header | Header shows "📄 Executive Reports & Summaries" | ☐ | |
| 11.3 | Check Flash Report section | Section visible | ☐ | |
| 11.4 | Click "Generate Flash Report" button | Report generates | ☐ | |
| 11.5 | Verify report content | Shows: | ☐ | |
| 11.5a | Status | "🔴 High Risk" or "🟢 Stable" | ☐ | |
| 11.5b | Total Incidents | Count displayed | ☐ | |
| 11.5c | Open Incidents | Count displayed | ☐ | |
| 11.5d | Executive Flash Report | Formatted report text | ☐ | |
| 11.5e | Operational Risk | Risk level shown | ☐ | |
| 11.5f | Active Clusters | Cluster count | ☐ | |
| 11.5g | Chronic Sites | List of sites | ☐ | |
| 11.5h | Deflection Potential | Ticket count | ☐ | |

**Result:** ☐ PASS  ☐ FAIL

---

## Test 12: War Room Mode

**Objective:** Verify Major Incident Mode (War Room) works correctly

| Step | Action | Expected Result | Pass/Fail | Notes |
|------|--------|----------------|-----------|-------|
| 12.1 | Find sidebar toggle | Look for "🔴 Major Incident Mode" toggle | ☐ | |
| 12.2 | Toggle ON | Click toggle to enable | ☐ | |
| 12.3 | Verify visual change | Page changes to red theme | ☐ | |
| 12.4 | Check header | Large red header: "🚨 WAR ROOM: MAJOR INCIDENT ACTIVE" | ☐ | |
| 12.5 | Check Active Major Incidents | Section shows open P1/P2 incidents | ☐ | |
| 12.6 | Check Velocity Meter (Column 1) | Shows "Incidents / Min" metric | ☐ | |
| 12.7 | Check Blast Radius (Column 2) | Shows "Blast Radius (Locations)" | ☐ | |
| 12.8 | Check Change Radar (Column 3) | Shows "Change Radar" with recent changes | ☐ | |
| 12.9 | Check Crisis Memory | Section visible | ☐ | |
| 12.10 | Enter search query | Type in search box | ☐ | |
| 12.11 | Click "Search Historical P1s" | Results appear | ☐ | |
| 12.12 | Verify results | Shows historical P1 records with resolutions | ☐ | |
| 12.13 | Toggle OFF | Disable War Room mode | ☐ | |
| 12.14 | Verify normal view | Returns to normal dashboard view | ☐ | |

**Result:** ☐ PASS  ☐ FAIL

---

## Test 13: Retro Audit Page

**Objective:** Verify multi-page structure and Retro Audit functionality

| Step | Action | Expected Result | Pass/Fail | Notes |
|------|--------|----------------|-----------|-------|
| 13.1 | Check sidebar navigation | Look for page navigation | ☐ | |
| 13.2 | Navigate to Retro Audit | Click on "1_Retro_Audit" or use URL | ☐ | |
| 13.3 | Verify page loads | Page shows "🔙 Phase 3: The 'Back to the Future' Retro" | ☐ | |
| 13.4 | Check sidebar upload | File uploaders visible | ☐ | |
| 13.5 | Check Timeline Fusion | Section visible | ☐ | |
| 13.6 | Check Zombie Problems | Section visible | ☐ | |
| 13.7 | Check Deflection Opportunity | Section visible | ☐ | |

**Result:** ☐ PASS  ☐ FAIL

---

## Test 14: Data Loading - Offline Mode

**Objective:** Verify CSV upload functionality works

| Step | Action | Expected Result | Pass/Fail | Notes |
|------|--------|----------------|-----------|-------|
| 14.1 | Select "Offline Data" from sidebar | Dropdown changes | ☐ | |
| 14.2 | Check sidebar message | Shows "Mode: Offline Data (CSV)" | ☐ | |
| 14.3 | Expand "Upload Overrides" | Expander opens | ☐ | |
| 14.4 | Verify file uploaders | Three uploaders visible: | ☐ | |
| 14.4a | Upload Incidents (CSV) | File uploader visible | ☐ | |
| 14.4b | Upload Changes (CSV) | File uploader visible | ☐ | |
| 14.4c | Upload Problems (CSV) | File uploader visible | ☐ | |
| 14.5 | Upload test CSV (optional) | If CSV available, upload | ☐ | |
| 14.6 | Verify data loads | Success messages appear | ☐ | |

**Result:** ☐ PASS  ☐ FAIL

---

## Test 15: Error Handling

**Objective:** Verify graceful error handling

| Step | Action | Expected Result | Pass/Fail | Notes |
|------|--------|----------------|-----------|-------|
| 15.1 | Switch to Mock mode | Select "Live API (Mock)" | ☐ | |
| 15.2 | Verify no errors | App continues to work | ☐ | |
| 15.3 | Switch between tabs rapidly | Click tabs quickly | ☐ | |
| 15.4 | Verify no crashes | App remains stable | ☐ | |
| 15.5 | Check console | No Python errors in terminal | ☐ | |

**Result:** ☐ PASS  ☐ FAIL

---

## Test 16: Logging Infrastructure

**Objective:** Verify logging files are created and written

| Step | Action | Expected Result | Pass/Fail | Notes |
|------|--------|----------------|-----------|-------|
| 16.1 | Navigate to worktree directory | `cd ../AI_Ops_Lite-test-integrated-workflow-ui` | ☐ | |
| 16.2 | Check logs directory | `ls logs/` or check folder | ☐ | |
| 16.3 | Verify log files exist | All 5 files present: | ☐ | |
| 16.3a | audit_trail.jsonl | File exists | ☐ | |
| 16.3b | business_events.jsonl | File exists | ☐ | |
| 16.3c | errors.jsonl | File exists | ☐ | |
| 16.3d | performance.jsonl | File exists | ☐ | |
| 16.3e | roi_metrics.jsonl | File exists | ☐ | |
| 16.4 | Use app features | Perform actions in app | ☐ | |
| 16.5 | Check log file sizes | Files should have content | ☐ | |
| 16.6 | Verify log entries | Open one file, verify JSONL format | ☐ | |

**Result:** ☐ PASS  ☐ FAIL

---

## Test Summary

### Overall Test Results

| Test Section | Status | Notes |
|--------------|--------|-------|
| Test 1: Application Startup | ☐ PASS ☐ FAIL | |
| Test 2: Data Loading - Mock Mode | ☐ PASS ☐ FAIL | |
| Test 3: Dashboard Tab | ☐ PASS ☐ FAIL | |
| Test 4: Active Incidents Tab | ☐ PASS ☐ FAIL | |
| Test 5: AI Assistant - Similar Incidents | ☐ PASS ☐ FAIL | |
| Test 6: AI Assistant - Routing | ☐ PASS ☐ FAIL | |
| Test 7: AI Assistant - Problem Detection | ☐ PASS ☐ FAIL | |
| Test 8: AI Assistant - Communication | ☐ PASS ☐ FAIL | |
| Test 9: Historical Analysis | ☐ PASS ☐ FAIL | |
| Test 10: Monitoring & ROI | ☐ PASS ☐ FAIL | |
| Test 11: Reports Tab | ☐ PASS ☐ FAIL | |
| Test 12: War Room Mode | ☐ PASS ☐ FAIL | |
| Test 13: Retro Audit Page | ☐ PASS ☐ FAIL | |
| Test 14: Data Loading - Offline | ☐ PASS ☐ FAIL | |
| Test 15: Error Handling | ☐ PASS ☐ FAIL | |
| Test 16: Logging Infrastructure | ☐ PASS ☐ FAIL | |

### Issues Found

**Critical Issues:**
- None: ☐

**Minor Issues:**
1. _________________________________________________
2. _________________________________________________
3. _________________________________________________

### Performance Observations

- App startup time: _______________ seconds
- Data loading time: _______________ seconds
- ML training time: _______________ seconds
- Overall responsiveness: ☐ Excellent  ☐ Good  ☐ Fair  ☐ Poor

### Final Recommendation

☐ **APPROVE** - Branch ready for merge  
☐ **CONDITIONAL APPROVE** - Minor issues noted  
☐ **REJECT** - Critical issues found  

**Tester Signature:** _______________  
**Date:** _______________

---

## Quick Reference: Tab Structure

1. 🏠 **Dashboard** - Overview metrics and quick actions
2. 🚨 **Active Incidents** - Real-time risk monitoring
3. 🧠 **AI Assistant** - 4 AI Intelligence features
4. 📈 **Historical Analysis** - Deep dive analysis
5. 📊 **Monitoring & ROI** - Platform performance tracking
6. 📄 **Reports** - Executive reports and summaries

---

## Tips for Testing

- **Take screenshots** of any issues found
- **Note browser** and version used
- **Check console** for JavaScript errors (F12)
- **Check terminal** for Python errors
- **Test on different screen sizes** if possible
- **Try rapid clicking** to test stability

---

**End of Test Plan**

