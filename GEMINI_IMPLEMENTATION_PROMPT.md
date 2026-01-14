# Implementation Prompt for Gemini Antigravity

## Context

You are working on the **AI_Ops_Lite** project, a proof-of-concept AIOps system that demonstrates how to identify failures in traditional ITIL-based IT service management using AI and machine learning.

The project includes a **Retro Audit module** (aka "Back to the Future") designed to analyze historical incident, problem, and change data to expose reactive patterns and quantify opportunities for proactive problem management.

---

## Your Task

A comprehensive analysis has been completed that:
1. **Documents the current state** of the Retro Audit module
2. **Identifies a critical gap**: Lack of clustering analysis despite robust clustering capabilities existing elsewhere in the codebase
3. **Proposes detailed enhancements** with complete code implementations

Your job is to:
1. **Pull the analysis branch**
2. **Read and understand the two analysis documents**
3. **Create a detailed implementation plan**
4. **Begin implementation** of the highest-priority enhancements

---

## Step 1: Pull the Analysis Branch

```bash
# Checkout the analysis branch
git fetch origin
git checkout claude/retro-audit-analysis-nWG3u

# Verify you have the analysis documents
ls -la *.md
```

You should see:
- `RETRO_AUDIT_ANALYSIS.md` (643 lines)
- `CLUSTERING_FOR_RETRO_AUDIT.md` (624 lines)
- `REPOSITORY_SUMMARY.md` (existing project overview)

---

## Step 2: Read the Analysis Documents

### Document 1: `RETRO_AUDIT_ANALYSIS.md`

This document contains:
- **Current implementation summary** of the Retro Audit module (3 functions)
- **How it demonstrates ITIL failures** using historical data
- **10 enhancement proposals** prioritized by business impact
- **Implementation roadmap** (Phases 1-4)
- **Code quality observations** and refactoring suggestions
- **UI/UX improvement mockups**

**Key sections to focus on:**
- "Current Implementation Summary" (lines 11-100)
- "Suggested Enhancements" (lines 150-400)
- "Implementation Roadmap" (lines 650-700)

### Document 2: `CLUSTERING_FOR_RETRO_AUDIT.md`

This document addresses the **critical gap**:
- **Why clustering analysis is missing** from Retro Audit
- **Why this gap matters** for demonstrating ITIL failures
- **4 complete implementations** ready to integrate:
  1. `identify_problem_coverage_gaps()` - Find clusters without problem records
  2. `identify_zombie_patterns()` - Track patterns recurring post-resolution
  3. `analyze_cluster_temporal_evolution()` - Show patterns over time
  4. `calculate_cluster_based_itil_score()` - Single maturity metric (0-100)
- **Full UI integration code** for `app.py`
- **Expected outputs and business impact**

**Key sections to focus on:**
- "Current Gap" (lines 10-30)
- "Proposed Implementation" (lines 40-500)
- "UI Integration" (lines 510-600)

---

## Step 3: Understand the Codebase Structure

Before implementing, review these key files:

### Core Files:
```
AI_Ops_Lite/
├── app.py                  # Main Streamlit UI (613 lines)
│   └── Lines 566-611: Retro Audit section (Phase 5)
│
├── retro_analysis.py       # Retro Audit logic (175 lines)
│   ├── create_timeline_fusion_chart()
│   ├── identify_zombie_problems()
│   └── calculate_deflection_opportunity()
│
├── analysis.py             # Core ML/clustering (363 lines)
│   ├── perform_clustering() - DBSCAN + TF-IDF (LINES 72-103)
│   ├── check_historical_recursion()
│   └── cluster_open_incidents()
│
└── aiops_intelligence.py   # AI features (existing)
```

### Key Insight:
**Clustering already exists** in `analysis.py` but is **NOT used** in `retro_analysis.py`. The enhancement is to bridge this gap.

---

## Step 4: Create Your Implementation Plan

Based on the analysis documents, create a plan that addresses:

### A. **Scope Definition**
- Which enhancements from `RETRO_AUDIT_ANALYSIS.md` will you implement?
- Which clustering features from `CLUSTERING_FOR_RETRO_AUDIT.md` are highest priority?
- What is your definition of "done" for each component?

### B. **Implementation Strategy**
- Will you create a new file `retro_clustering.py` or extend `retro_analysis.py`?
- How will you integrate with existing `analysis.py` clustering functions?
- What dependencies/imports are needed?
- How will you handle edge cases (empty DataFrames, missing columns, etc.)?

### C. **Prioritized Task Breakdown**

Suggested priority (you can adjust):

**🔴 HIGH PRIORITY (Implement First):**
1. `identify_problem_coverage_gaps()` - Shows clusters without problem records
2. `identify_zombie_patterns()` - Pattern-based zombie detection
3. `calculate_cluster_based_itil_score()` - Executive dashboard metric
4. UI integration - Add 4th tab "Pattern Analysis" to app.py

**🟡 MEDIUM PRIORITY (Implement Second):**
5. `analyze_cluster_temporal_evolution()` - Recurring pattern timeline
6. Enhanced zombie detection with time-window enforcement (from RETRO_AUDIT_ANALYSIS.md)
7. Incident-to-Problem lag analysis

**🟢 LOW PRIORITY (Future Enhancement):**
8. Interactive timeline filtering
9. Zombie heatmap visualization
10. ML-based deflection classifier

### D. **Testing Strategy**
- How will you test with mock data (`data/input/incidents.json`, `problems.json`)?
- What edge cases need testing (empty data, single incident, no clusters, etc.)?
- How will you validate the ITIL Maturity Score calculation?

### E. **UI/UX Design**
- Review the proposed UI mockups in `CLUSTERING_FOR_RETRO_AUDIT.md` (lines 510-600)
- Plan the Streamlit component structure for the new "Pattern Analysis" tab
- Decide on metrics display format (st.metric, st.dataframe, st.plotly_chart, etc.)

### F. **Documentation Requirements**
- Update `REPOSITORY_SUMMARY.md` with new clustering features
- Add docstrings to all new functions
- Create usage examples for key functions

### G. **Integration Points**
- How will this integrate with existing "AI Intelligence" tab?
- Can zombie patterns link to "Auto-Problem Creation" suggestions?
- Should the ITIL Maturity Score appear in the Flash Report?

---

## Step 5: Deliverables Expected

Please provide:

### 1. **Implementation Plan Document** (Markdown)
Create: `IMPLEMENTATION_PLAN.md` with:
- [ ] Executive Summary (what you'll build, why, expected impact)
- [ ] Scope (in-scope, out-of-scope, assumptions)
- [ ] Architecture (file structure, dependencies, integration points)
- [ ] Task Breakdown (numbered tasks with effort estimates)
- [ ] Testing Plan (test cases, validation criteria)
- [ ] Timeline (days/hours per task)
- [ ] Risk Assessment (what could go wrong, mitigations)
- [ ] Success Metrics (how to measure if implementation succeeded)

### 2. **Code Implementation**
Implement the HIGH PRIORITY items:
- [ ] New functions in `retro_analysis.py` or new `retro_clustering.py`
- [ ] Helper functions (`extract_pattern_keywords()`, `keyword_similarity()`, etc.)
- [ ] UI integration in `app.py` (new tab or enhanced existing tabs)
- [ ] Error handling and edge case management
- [ ] Docstrings and inline comments

### 3. **Testing Validation**
- [ ] Test with mock data in `data/input/`
- [ ] Verify ITIL Maturity Score calculation
- [ ] Ensure UI renders correctly in Streamlit
- [ ] Screenshot or description of working UI

### 4. **Documentation Updates**
- [ ] Update `REPOSITORY_SUMMARY.md` with clustering features
- [ ] Create `CLUSTERING_USAGE_GUIDE.md` (how to use new features)
- [ ] Add examples to docstrings

---

## Step 6: Implementation Guidelines

### Code Quality Standards:
- **Follow existing patterns**: Match the style in `retro_analysis.py` and `analysis.py`
- **Defensive programming**: Check for empty DataFrames, missing columns, null values
- **Type hints**: Use where appropriate (pandas DataFrames, lists, dicts)
- **Error handling**: Try/except with meaningful error messages
- **Performance**: Avoid nested loops where possible, vectorize pandas operations

### Example Error Handling Pattern:
```python
def identify_problem_coverage_gaps(incidents_df, problems_df, min_cluster_size=3):
    """
    Identifies incident clusters that should have Problem Records but don't.

    Args:
        incidents_df: DataFrame with incident data
        problems_df: DataFrame with problem data
        min_cluster_size: Minimum incidents in cluster to consider (default 3)

    Returns:
        DataFrame with gaps, or empty DataFrame if none found
    """
    # Validate inputs
    if incidents_df.empty:
        return pd.DataFrame()

    if 'short_description' not in incidents_df.columns:
        raise ValueError("incidents_df missing required column 'short_description'")

    try:
        # Implementation here...
        pass
    except Exception as e:
        print(f"Error in problem coverage gap analysis: {e}")
        return pd.DataFrame()
```

### Streamlit UI Best Practices:
- Use `st.spinner()` for long-running operations
- Use `st.error()`, `st.warning()`, `st.success()`, `st.info()` for status messages
- Use `st.expander()` for detailed data to keep UI clean
- Use `st.metric()` for key numbers (ITIL Score, counts, etc.)
- Use `st.columns()` for side-by-side layouts
- Add helpful `st.caption()` explanations under complex metrics

---

## Step 7: Key Questions to Answer

As you read the analysis docs, consider:

1. **Architecture Decision**: Should clustering functions go in:
   - `retro_analysis.py` (extend existing)
   - `retro_clustering.py` (new module) ← **Recommended**
   - `analysis.py` (add to existing core logic)

2. **Performance**: The analysis involves clustering all incidents multiple times. How to optimize?
   - Cache clustering results in session state?
   - Allow user to trigger analysis vs auto-run?
   - Progressive disclosure (calculate on-demand per tab)?

3. **Data Requirements**: What if problems_df is empty?
   - Can we still show value (coverage gaps without comparison)?
   - Should we show warnings or just disable features?

4. **Integration Priority**: Should we integrate with:
   - Flash Report (show ITIL Score there)?
   - AI Intelligence tab (link zombie patterns to problem suggestions)?
   - Investigation Deck (share clustering results)?

5. **Configurability**: Should thresholds be user-adjustable?
   - `min_cluster_size` (default 3)
   - `months_after_resolution` for zombie detection (default 6)
   - `similarity_threshold` for keyword matching (default 0.4)

---

## Step 8: Reference the Existing Code

### Clustering Already Exists:
```python
# From analysis.py lines 72-103
def perform_clustering(df):
    """Performs DBSCAN clustering on incidents based on text similarity."""
    # ... TF-IDF vectorization
    # ... DBSCAN with eps=0.5, min_samples=2, metric='cosine'
    # ... Returns df with 'Cluster_ID' column added
```

**You should REUSE this function**, not rewrite clustering from scratch.

### Entity Extraction Pattern:
```python
# From analysis.py lines 7-36
def extract_entities(text):
    """Extracts IPs, IDs, server names using regex"""
    # ... IP pattern: r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    # ... Server pattern: r'\b(?=.*\d)(?=.*[a-zA-Z])[a-zA-Z0-9-]{3,}\b'
```

**You can extend this pattern** for keyword extraction in clustering analysis.

---

## Step 9: Success Criteria

Your implementation will be considered successful if:

### Functional Requirements:
- ✅ All HIGH priority features work with mock data
- ✅ ITIL Maturity Score calculates correctly (0-100 range, sensible results)
- ✅ Problem coverage gaps accurately identify clusters without problems
- ✅ Zombie pattern detection finds patterns recurring post-resolution
- ✅ UI renders without errors in Streamlit

### Code Quality:
- ✅ Follows existing code style and patterns
- ✅ Includes error handling for edge cases
- ✅ Has docstrings for all public functions
- ✅ No hardcoded values (use function parameters)

### Business Value:
- ✅ Demonstrates ITIL failures with concrete data
- ✅ Provides actionable insights (which patterns to address)
- ✅ Executive-ready metrics (single ITIL Score number)
- ✅ Clear ROI calculation (cost impact of gaps)

### Documentation:
- ✅ Implementation plan explains what was built and why
- ✅ Usage guide shows how to use new features
- ✅ Updated repository summary reflects new capabilities

---

## Step 10: Getting Started

### Immediate Actions:
1. **Fetch and checkout the branch**
   ```bash
   git fetch origin
   git checkout claude/retro-audit-analysis-nWG3u
   ```

2. **Read both analysis documents thoroughly**
   - Take notes on key insights
   - Flag any questions or unclear areas

3. **Explore the existing codebase**
   ```bash
   # Review current Retro Audit implementation
   cat retro_analysis.py

   # Review clustering implementation
   grep -A 30 "def perform_clustering" analysis.py

   # Review current UI
   grep -A 50 "Phase 5: Retro Audit" app.py
   ```

4. **Create your implementation plan**
   - Start with `IMPLEMENTATION_PLAN.md`
   - Break down into tasks with estimates
   - Identify dependencies and risks

5. **Begin implementation**
   - Start with highest priority items
   - Test incrementally with mock data
   - Commit frequently with clear messages

---

## Questions to Consider Before Starting

1. **What is your estimated timeline** for completing HIGH priority items?
2. **What technical risks** do you foresee (performance, data quality, UI complexity)?
3. **What additional context** do you need about the project?
4. **Are there any analysis recommendations** you disagree with or would modify?
5. **What dependencies or libraries** might be needed beyond what's in `requirements.txt`?

---

## Additional Resources

### Project Background:
- Read `REPOSITORY_SUMMARY.md` for full project context
- Understand the ITIL → AIOps transformation story
- Review existing AI Intelligence features in `aiops_intelligence.py`

### Technical References:
- **DBSCAN Clustering**: Used for pattern detection (scikit-learn)
- **TF-IDF**: Text vectorization for similarity (scikit-learn)
- **Streamlit**: UI framework (streamlit.io docs)
- **Pandas**: Data manipulation (pandas.pydata.org)
- **Plotly**: Interactive charts (plotly.com/python)

### Data Format:
- Mock data: `data/input/incidents.json`, `problems.json`, `changes.json`
- CSV data: `data/input/*.csv` (EMEA datasets)
- All data uses ServiceNow-like schema (number, short_description, opened_at, etc.)

---

## Final Notes

This is a **high-impact enhancement** that transforms the Retro Audit module from a visualization tool to a definitive ITIL failure detection system.

The analysis has already done the heavy lifting:
- ✅ Identified the gap
- ✅ Explained why it matters
- ✅ Provided complete code implementations
- ✅ Designed the UI integration
- ✅ Calculated business impact

Your job is to:
1. **Understand** the analysis
2. **Plan** the implementation
3. **Execute** with quality
4. **Validate** with testing
5. **Document** the results

**Expected Outcome:**
A working Retro Audit module with clustering-based pattern analysis that shows:
- ITIL Maturity Score: XX/100
- Problem Coverage: XX%
- Zombie Patterns: X detected
- Recurring Patterns: X identified
- Cost Impact: $XX,XXX quantified

This will provide **definitive, data-driven proof** of ITIL failures and AIOps opportunities.

---

## Ready to Begin?

1. Pull the branch: `claude/retro-audit-analysis-nWG3u`
2. Read: `RETRO_AUDIT_ANALYSIS.md` and `CLUSTERING_FOR_RETRO_AUDIT.md`
3. Create: `IMPLEMENTATION_PLAN.md`
4. Implement: Start with HIGH priority features
5. Test: Validate with mock data
6. Document: Update repository docs
7. Commit: Push to the branch or create new implementation branch

**Good luck! This is exciting work that will significantly enhance the AIOps demonstration value.**

---

**Branch:** `claude/retro-audit-analysis-nWG3u`
**Key Files:** `RETRO_AUDIT_ANALYSIS.md`, `CLUSTERING_FOR_RETRO_AUDIT.md`
**Priority:** 🔴 HIGH - Critical gap that limits ITIL failure demonstration
**Impact:** 🔥 Very High - Transforms module into executive-ready proof system
**Estimated Effort:** 3-5 days for HIGH priority features

---

_This prompt created: 2026-01-14_
_Analysis completed by: Claude (Sonnet 4.5)_
_Implementation by: Gemini Antigravity_
