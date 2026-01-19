
Manual Test Plan: feature/ui-and-logic-enhancements Branch
Branch: feature/ui-and-logic-enhancements
Test Date: _
Tester: _
Base Branch: origin/main
Commits Ahead: 6 commits


Prerequisites

Git branch checked out: feature/ui-and-logic-enhancements
Dependencies installed: pip install -r requirements.txt
Test data files exist in data/input/:incidents.json
changes.json
problems.json
Logs directory exists: logs/ (will be created automatically)
Python environment: Python 3.13+ recommended


Test Execution Instructions

Start the Streamlit app: streamlit run app.py
Open browser to: http://localhost:8501
Follow each test section in order
Check off items as you complete them
Note any issues or observations in the "Notes" column
Take screenshots of any issues found


Test 1: Application Startup
Objective: Verify the app starts without errors and displays correctly

Step	Action	Expected Result	Pass/Fail	Notes
1.1	Launch app with streamlit run app.py	App starts, no errors in console	Pass	
1.2	Check browser loads	Page loads at http://localhost:8501	Pass	
1.3	Verify title	Title shows "AIOps Lite: Flight Deck"	Pass	
1.4	Check sidebar	Sidebar shows "Data Source Config" with selectbox	Pass	
1.5	Verify no error messages	No red error messages displayed	Pass	
1.6	Check console output	No Python tracebacks or critical errors	Pass	

Result: ☐ PASS  ☐ FAIL


Test 2: Data Loading - Mock Mode
Objective: Verify data loads correctly in Mock mode

Step	Action	Expected Result	Pass/Fail	Notes
2.1	Select "Live API (Mock)" from sidebar dropdown	Dropdown shows selected option	Pass	
2.2	Wait for data to load	Sidebar shows success messages	Pass	
2.3	Check incidents loaded	"Loaded X Incidents" message appears	Pass	
2.4	Check changes loaded	"Loaded X Changes" message appears	Pass	
2.5	Check problems loaded	"Loaded X Problems" message appears (if available)	Pass	
2.6	Verify data appears in UI	Main content area shows data/metrics	Pass	
2.7	Check session state	Data persists when switching tabs	Pass	

Result: ☐ PASS  ☐ FAIL


Test 3: Current Risks Tab
Objective: Verify real-time risk monitoring features

Step	Action	Expected Result	Pass/Fail	Notes
3.1	Click on "🔴 Current Risks" tab	Tab becomes active	Pass	
3.2	Verify header	Header shows "Real-Time Risk Monitor"	Pass	
3.3	Check Volume Monitor (Column 1)	Shows:	Pass	
3.3a	Subheader	"Volume Monitor"	Pass	
3.3b	Metric	Daily Volume metric with status	Pass	
3.3c	Chart	Line chart showing daily counts	Pass	
3.4	Check Hidden Clusters (Column 2)	Shows:	Pass	"No clustered open incidents."
3.4a	Subheader	"Hidden Clusters (Open)"	Pass	Would be better that mock data has example clusters
3.4b	Cluster table	DataFrame with Cluster ID and Count	Pass	No frame as no clusters
3.5	Check Suspect Root Causes (Column 3)	Shows:	Pass	No correlation with recent changes.
3.5a	Subheader	"Suspect Root Causes"	Pass	Would be better that mock data has example clusters
3.5b	Correlation results	Table with Cluster, Change, Reason	Pass	No data
3.6	Check Active Clusters Detail	Section below shows detailed cluster data	Pass	No data
3.7	Test Solution Recommender	Scroll to "💡 Agent Assist" section	Pass	
3.7a	Select incident dropdown	Dropdown shows list of open incidents	Pass	
3.7b	Select an incident	Incident details display	Pass	
3.7c	Click "🔍 Find Recommended Fixes"	Spinner appears, then results show	Pass	
3.7d	Verify recommendations	Shows similar incidents with resolution notes	Pass	Where is the data coming from?

Result: ☐ PASS  ☐ FAIL


Test 4: Investigation Deck Tab
Objective: Verify deep dive analysis features

Step	Action	Expected Result	Pass/Fail	Notes
4.1	Click "🔍 Investigation Deck" tab	Tab becomes active	Pass	
4.2	Verify header	Header shows "Deep Dive Analysis"	Pass	
4.3	Check "Full Clustering" expander	Expandable section visible	Pass	
4.4	Expand "Full Clustering"	Shows clustering results with details	Pass	
4.5	Check "Repeat Offenders" expander	Expandable section visible	Pass	
4.6	Expand "Repeat Offenders"	Shows recurring entities (sites/users)	Pass	
4.7	Check "Incident-Change Correlation" expander	Expandable section visible	Pass	
4.8	Expand "Incident-Change Correlation"	Shows correlation results	Pass	
4.9	Verify data persistence	Switch tabs and return - data still visible	Pass	

Result: PASS 


Test 5: AI Intelligence Tab
Objective: Verify AI-powered features work correctly
5.1 Similar Incident Recommendation

Step	Action	Expected Result	Pass/Fail	Notes
5.1.1	Scroll to "1️⃣ Similar Incident Recommendation"	Section visible	Pass	
5.1.2	Select incident from dropdown	Dropdown shows incident list	Pass	
5.1.3	Click "Find Similar Incidents" button	Spinner appears	Pass	
5.1.4	Wait for results	Results appear with:	Pass	
5.1.4a	Success message	"Found X similar resolved incidents!"	Pass	
5.1.4b	Expandable results	Each result shows in expander	Pass	
5.1.4c	Similarity score	Shows percentage match	Pass	
5.1.4d	Resolution notes	Shows resolution details	Pass	
5.1.5	Check MTTR improvement	Info box shows time savings potential	Pass	

5.2 Intelligent Assignment Routing

Step	Action	Expected Result	Pass/Fail	Notes
5.2.1	Scroll to "2️⃣ Intelligent Assignment Routing"	Section visible	Pass	
5.2.2	Click "Train Assignment Model" button	Spinner appears	Pass	
5.2.3	Wait for training	Training completes (may take 10-30 seconds)	Pass	
5.2.4	Verify training success	Success message shows:	Pass	
5.2.4a	"✅ Model trained on X incidents"	Message appears	Pass	
5.2.4b	Training accuracy	Shows percentage	Pass	
5.2.4c	Assignment groups count	Shows number of groups	Pass	
5.2.5	Enter test description	Type in text area: "VPN connection failed"	Pass	
5.2.6	Click "Predict Assignment" button	Predictions appear	Pass	
5.2.7	Verify predictions	Shows:	Pass	The Recommended Assignment looks odd - VPN issue is SAP issue
5.2.7a	Top recommendation	Shows assignment group with 🥇	Pass	
5.2.7b	Confidence score	Progress bar and percentage	Pass	
5.2.7c	Reasoning	Explanation text	Pass	
5.2.7d	Alternatives	Additional recommendations listed	Pass	

5.3 Proactive Problem Detection

Step	Action	Expected Result	Pass/Fail	Notes
5.3.1	Scroll to "3️⃣ Proactive Problem Detection"	Section visible	Pass	
5.3.2	Adjust threshold slider	Set to 5 incidents	Pass	
5.3.3	Click "Analyze Clusters for Problem Opportunities"	Spinner appears	Pass	
5.3.4	Wait for analysis	Analysis completes	Pass	
5.3.5	Verify problem suggestions	If clusters found:	Pass	
5.3.5a	Success message	"🔥 Found X cluster(s) that should have Problem Records!"	Pass	Only found related incidents when set to 3
5.3.5b	Problem suggestion expanders	Each suggestion in expander	Pass	
5.3.5c	Problem details	Shows incident count, time span, priority	Pass	
5.3.5d	Related incidents	Lists incident numbers	Pass	
5.3.5e	Recommended actions	Shows action items	Pass	
5.3.6	Click "Create Problem Record (Draft)"	Problem record details display	Fail	nothing displayed

5.4 Communication Assistant

Step	Action	Expected Result	Pass/Fail	Notes
5.4.1	Scroll to "4️⃣ Communication Assistant"	Section visible	Fail	
5.4.2	Select incident from dropdown	Dropdown shows incident list	Fail	
5.4.3	Enter impact details	Type in text area: "Service outage affecting 50 users"	Fail	
5.4.4	Click "Generate Template" button	Template appears	Fail	
5.4.5	Verify template	Shows formatted communication template	Fail	

Result:  FAIL


Test 6: Monitoring & ROI Tab
Objective: Verify monitoring dashboard and ROI tracking

Step	Action	Expected Result	Pass/Fail	Notes
6.1	Click "📊 Monitoring & ROI" tab	Tab becomes active	Pass	
6.2	Verify header	Header shows "📊 Platform Monitoring & ROI Tracking"	Pass	
6.3	Check ROI Summary section	Shows metrics:	Pass	shows "No metrics data available yet. Metrics will appear as you use the system."
6.3a	Incidents Analyzed	Metric displayed	Pass	No data
6.3b	Patterns Detected	Metric displayed	Pass	No data
6.3c	Time Saved	Metric displayed in hours	Pass	No data
6.3d	Cost Saved	Metric displayed in USD		No data
6.4	Check ML Model Performance	Section visible	Pass	
6.5	Expand ML feature	Shows acceptance rates by feature	Pass	No ML prediction data yet. Data will appear as you use AI Intelligence features.
6.6	Check System Performance	Section visible	Pass	
6.7	Verify performance metrics	Shows P50, P95, average, max		
6.8	Check Error Tracking	Section visible	Pass	
6.9	Verify error summary	Shows error counts by type	Pass	
6.10	Check User Activity	Section visible	Pass	
6.11	Check Audit Trail	Section visible with recent events	Pass	No user activity logged yet.
6.12	Click "Export Full Monitoring Report"	JSON report displays		
6.13	Verify report content	Report contains all sections		

Result: PASS 


Test 7: War Room Mode (Major Incident Mode)
Objective: Verify Major Incident Mode functionality

Step	Action	Expected Result	Pass/Fail	Notes
7.1	Find sidebar toggle	Look for "🔴 Major Incident Mode" toggle	Pass	
7.2	Toggle ON	Click toggle to enable	Pass	
7.3	Verify visual change	Page changes to red theme	Pass	
7.4	Check header	Large red header: "🚨 WAR ROOM: MAJOR INCIDENT ACTIVE"	Pass	
7.5	Verify header animation	Header pulses/animated	Pass	
7.6	Check Active Major Incidents	Section shows open P1/P2 incidents	Pass	
7.7	Check Velocity Meter (Column 1)	Shows "Incidents / Min" metric	Pass	
7.8	Verify velocity calculation	Shows incidents per minute (last 30 mins)	Pass	
7.9	Check Blast Radius (Column 2)	Shows "Blast Radius (Locations)"	Pass	
7.10	Verify location data	Shows affected locations with ticket counts	Pass	
7.11	Check Change Radar (Column 3)	Shows "Change Radar"	Pass	
7.12	Verify recent changes	Shows changes closed in last 4 hours	Pass	No changes loaded.?
7.13	Check Crisis Memory	Section visible	Pass	
7.14	Verify search query	Auto-populated from top cluster	Pass	
7.15	Enter search query	Type in search box	Pass	
7.16	Click "Search Historical P1s"	Results appear	Pass	
7.17	Verify results	Shows historical P1 records with resolutions	Pass	No relevant historical P1s found.?
7.18	Verify dashboard disabled	Normal tabs not visible	Pass	
7.19	Toggle OFF	Disable War Room mode	Pass	
7.20	Verify normal view	Returns to normal dashboard view	Pass	

Result: ☐ PASS  ☐ FAIL


Test 8: Retro Audit Page
Objective: Verify multi-page structure and Retro Audit functionality

Step	Action	Expected Result	Pass/Fail	Notes
8.1	Check sidebar navigation	Look for page navigation or use URL	Pass	
8.2	Navigate to Retro Audit	Click on "1_Retro_Audit" or navigate to /1_Retro_Audit	Pass	
8.3	Verify page loads	Page shows "🔙 Phase 3: The 'Back to the Future' Retro"	Pass	
8.4	Check sidebar upload	File uploaders visible	Pass	
8.5	Verify upload options	Three uploaders: Incidents, Problems, Changes	Pass	
8.6	Check Timeline Fusion	Section visible	Pass	
8.7	Expand Timeline Fusion	Shows timeline chart	Pass	
8.8	Check Zombie Problems	Section visible	Pass	
8.9	Expand Zombie Problems	Shows zombie problem analysis	Pass	
8.10	Check Deflection Opportunity	Section visible	Pass	
8.11	Expand Deflection Opportunity	Shows deflection metrics	Pass	
8.12	Navigate back to main page	Return to main app	Pass	
8.13	Verify navigation works	Can switch between pages	Pass	

Result: PASS


Test 9: Flash Report
Objective: Verify Flash Report generation

Step	Action	Expected Result	Pass/Fail	Notes
9.1	Find sidebar button	Look for "Generate Flash Report" button	Pass	
9.2	Click "Generate Flash Report"	Flash report appears at bottom	Pass	
9.3	Verify report header	Shows "⚡ Executive Flash Report"	Pass	
9.4	Check report content	Shows:	Fail	NameError: name 'calculate_deflection_opportunity' is not defined
File "C:\Users\cglynn\myPython\SNOW_MI_Flight_Deck\AI_Ops_Lite\app.py", line 997, in     main()    ~~~~^^ File "C:\Users\cglynn\myPython\SNOW_MI_Flight_Deck\AI_Ops_Lite\app.py", line 980, in main    d_count, _, _ = calculate_deflection_opportunity(df_cleaned)                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
9.4a	Status	"🔴 High Risk" or "🟢 Stable"	Fail	
9.4b	Total Incidents	Count displayed	Pass	
9.4c	Open Incidents	Count displayed	Pass	
9.4d	Operational Risk	Risk level shown	Fail	
9.4e	Active Clusters	Cluster count	Fail	
9.4f	Chronic Sites	List of sites	Fail	
9.4g	Deflection Potential	Ticket count	Fail	

Result: FAIL


Test 10: Data Loading - Offline Mode
Objective: Verify CSV upload functionality works

Step	Action	Expected Result	Pass/Fail	Notes
10.1	Select "Offline Data" from sidebar	Dropdown changes	Pass	
10.2	Check sidebar message	Shows "Mode: Offline Data (CSV)"	Pass	
10.3	Expand "Upload Overrides"	Expander opens	Pass	
10.4	Verify file uploaders	Three uploaders visible:	Pass	
10.4a	Upload Incidents (CSV)	File uploader visible	Pass	
10.4b	Upload Changes (CSV)	File uploader visible	Pass	
10.4c	Upload Problems (CSV)	File uploader visible	Pass	
10.5	Upload test CSV (optional)	If CSV available, upload	Pass	
10.6	Verify data loads	Success messages appear	Pass	
10.7	Verify data displays	Data appears in tabs	Pass	

Result: PASS 


Test 11: Logging Infrastructure Verification
Objective: Verify logging files are created and written

Step	Action	Expected Result	Pass/Fail	Notes
11.1	Navigate to logs directory	cd logs/ or check folder	Pass	
11.2	Check log files exist	All 5 files present:	Pass	
11.2a	audit_trail.jsonl	File exists	Pass	
11.2b	business_events.jsonl	File exists	Pass	
11.2c	errors.jsonl	File exists	Pass	
11.2d	performance.jsonl	File exists	Pass	
11.2e	roi_metrics.jsonl	File exists	Pass	
11.3	Use app features	Perform actions in app (AI predictions, etc.)	Pass	
11.4	Check log file sizes	Files should have content	Pass	
11.5	Verify log entries	Open one file, verify JSONL format	Pass	
11.6	Check audit trail	Verify audit entries logged	Pass	
11.7	Check performance logs	Verify performance metrics logged	Pass	
11.8	Check ROI metrics	Verify ROI calculations logged	Pass	

Result: PASS 


Test 12: Error Handling
Objective: Verify graceful error handling

Step	Action	Expected Result	Pass/Fail	Notes
12.1	Switch to Mock mode	Select "Live API (Mock)"	Pass	
12.2	Verify no errors	App continues to work	Pass	
12.3	Switch between tabs rapidly	Click tabs quickly	Pass	
12.4	Verify no crashes	App remains stable	Pass	
12.5	Check console	No Python errors in terminal	Pass	
12.6	Test with empty data	If possible, test with empty dataset	Pass	
12.7	Verify error messages	Errors displayed gracefully	Pass	
12.8	Check error logging	Errors logged to errors.jsonl	Pass	

Result: PASS 


Test 13: Performance Testing
Objective: Verify performance characteristics

Step	Action	Expected Result	Pass/Fail	Notes
13.1	Measure app startup	Time from command to browser load	Pass	
13.2	Measure data loading	Time to load mock data	Pass	
13.3	Measure ML training	Time for assignment model training	Pass	
13.4	Measure tab switching	Time to switch between tabs	Pass	
13.5	Check responsiveness	UI responds quickly to interactions	Pass	
13.6	Monitor memory usage	Check for memory leaks (optional)	Pass	

Performance Targets:

App startup: < 3 seconds
Data loading: < 2 seconds
ML training: < 30 seconds
Tab switching: < 1 second
Result: ☐ PASS  ☐ FAIL


Test Summary
Overall Test Results

Test Section	Status	Notes
Test 1: Application Startup		
Test 2: Data Loading - Mock Mode		
Test 3: Current Risks Tab		
Test 4: Investigation Deck Tab		
Test 5: AI Intelligence Tab	Fail	
Test 6: Monitoring & ROI Tab		
Test 7: War Room Mode		
Test 8: Retro Audit Page		
Test 9: Flash Report	Fail	
Test 10: Data Loading - Offline		
Test 11: Logging Infrastructure		
Test 12: Error Handling		
Test 13: Performance Testing		

Issues Found
Critical Issues:

None: ☐
Minor Issues:

_____
_____
_____
Enhancement Suggestions:

_____
_____
_____
Performance Observations

App startup time: _ seconds
Data loading time: _ seconds
ML training time: _ seconds
Overall responsiveness: ☐ Excellent ☐ Good ☐ Fair ☐ Poor
Browser Information

Browser: _ (Chrome/Firefox/Edge/etc.)
Version: _
OS: _
Final Recommendation
☐ APPROVE - Branch ready for merge
☐ CONDITIONAL APPROVE - Minor issues noted
☐ REJECT - Critical issues found
Tester Signature: _
Date: _


Quick Reference: Tab Structure

🔴 Current Risks - Real-time risk monitoringVolume Monitor
Hidden Clusters
Suspect Root Causes
Agent Assist (Solution Recommender)
🔍 Investigation Deck - Deep dive analysisFull Clustering
Repeat Offenders
Incident-Change Correlation
🧠 AI Intelligence - AI-powered featuresSimilar Incident Recommendation
Intelligent Assignment Routing
Proactive Problem Detection
Communication Assistant
📊 Monitoring & ROI - Platform monitoringROI Summary
ML Model Performance
System Performance
Error Tracking
User Activity
Audit Trail
Export Report
Additional Pages:

Retro Audit (pages/1_Retro_Audit.py) - Historical analysis


Tips for Testing

Take screenshots of any issues found
Note browser and version used
Check console for JavaScript errors (F12)
Check terminal for Python errors
Test on different screen sizes if possible
Try rapid clicking to test stability
Verify log files after using features
Test all data modes (Mock, Real, Offline)


Known Issues / Notes

Altair deprecation warning (cosmetic only, doesn't affect functionality)
Logging verification script was updated to use correct API


End of Test Plan
