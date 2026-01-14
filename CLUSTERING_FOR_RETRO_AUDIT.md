# Clustering Analysis for Retro Audit Module

## 🎯 Strategic Value

Adding **clustering analysis** to the Retro Audit module will transform it from a descriptive tool to a **pattern-detection powerhouse** that definitively proves ITIL's failure to prevent recurring issues.

---

## 📊 Current Gap

### What Exists Now:
```python
# analysis.py - Lines 72-103
def perform_clustering(df):
    """DBSCAN + TF-IDF clustering"""
    # Used in Investigation Deck & AI Intelligence
    # NOT used in Retro Audit ❌
```

### What's Missing:
- **Pattern-based zombie detection** (current is entity-based only)
- **Problem coverage gap analysis** (clusters with no problem records)
- **Post-resolution pattern tracking** (did the problem actually fix anything?)
- **Temporal cluster evolution** (same pattern recurring over time)

---

## 🚀 Proposed Implementation

### Enhancement 1: **Problem Coverage Gap Analysis**

```python
# Add to retro_analysis.py

from analysis import perform_clustering
import re
from collections import Counter

def identify_problem_coverage_gaps(incidents_df, problems_df, min_cluster_size=3):
    """
    Identifies incident clusters that should have Problem Records but don't.

    Returns:
        DataFrame with: Cluster_ID, Incident_Count, Pattern, Gap_Type, Business_Impact
    """
    if incidents_df.empty:
        return pd.DataFrame()

    # Cluster all incidents
    clustered = perform_clustering(incidents_df)

    gaps = []

    for cluster_id in clustered['Cluster_ID'].unique():
        if cluster_id == -1:  # Noise points
            continue

        cluster_incidents = clustered[clustered['Cluster_ID'] == cluster_id]

        # Skip small clusters
        if len(cluster_incidents) < min_cluster_size:
            continue

        # Extract keywords from cluster
        cluster_text = " ".join(cluster_incidents['short_description'].fillna(''))
        keywords = extract_pattern_keywords(cluster_text)

        # Check if Problem record exists for this pattern
        has_problem = False
        matched_problem = None

        if not problems_df.empty:
            for _, problem in problems_df.iterrows():
                problem_text = str(problem.get('short_description', ''))

                # Check keyword overlap
                if keyword_similarity(keywords, problem_text) > 0.3:
                    has_problem = True
                    matched_problem = problem['number']
                    break

        # Calculate time span
        time_span = (cluster_incidents['opened_at'].max() -
                     cluster_incidents['opened_at'].min()).days

        if not has_problem:
            # ITIL FAILURE: Recurring pattern, no problem record
            gaps.append({
                'Cluster_ID': cluster_id,
                'Incident_Count': len(cluster_incidents),
                'Time_Span_Days': time_span,
                'Pattern_Description': cluster_incidents['short_description'].iloc[0][:80],
                'First_Occurrence': cluster_incidents['opened_at'].min(),
                'Last_Occurrence': cluster_incidents['opened_at'].max(),
                'Gap_Type': '🔥 NO PROBLEM RECORD',
                'Business_Impact': f"{len(cluster_incidents)} incidents over {time_span} days",
                'Cost_Impact': f"${len(cluster_incidents) * 50:,}",
                'Related_Incidents': ", ".join(cluster_incidents['number'].head(5).tolist())
            })
        elif matched_problem:
            # Has problem, but did it help?
            # (Will be caught by zombie pattern analysis)
            pass

    return pd.DataFrame(gaps).sort_values('Incident_Count', ascending=False)


def extract_pattern_keywords(text, top_n=5):
    """Extract meaningful keywords from text"""
    # Remove stopwords
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'on', 'at',
        'for', 'with', 'by', 'is', 'it', 'this', 'that', 'issue',
        'error', 'problem', 'failed', 'failure', 'unable', 'not', 'cannot'
    }

    words = re.findall(r'\b\w{4,}\b', text.lower())
    filtered = [w for w in words if w not in stopwords]

    # Get top N most common
    common = Counter(filtered).most_common(top_n)
    return [word for word, count in common]


def keyword_similarity(keywords_list, text):
    """Calculate keyword match ratio"""
    text_lower = text.lower()
    matches = sum(1 for kw in keywords_list if kw in text_lower)
    return matches / len(keywords_list) if keywords_list else 0
```

---

### Enhancement 2: **Zombie Pattern Analysis**

```python
def identify_zombie_patterns(problems_df, incidents_df, months_after_resolution=6):
    """
    Identifies Problem Records that didn't actually fix the root cause.

    Method:
    1. For each closed Problem, extract pattern keywords
    2. Search for incidents AFTER problem closure with same pattern
    3. If pattern continues → Zombie Problem

    Returns:
        DataFrame with zombie problems and post-resolution incident counts
    """
    if problems_df.empty or incidents_df.empty:
        return pd.DataFrame()

    zombies = []

    for _, problem in problems_df.iterrows():
        # Skip open problems
        if pd.isnull(problem.get('closed_at')):
            continue

        problem_close_date = problem['closed_at']

        # Define analysis window (e.g., 6 months after problem closed)
        analysis_end = problem_close_date + pd.DateOffset(months=months_after_resolution)

        # Extract pattern from problem
        problem_keywords = extract_pattern_keywords(
            str(problem.get('short_description', '')) + " " +
            str(problem.get('description', ''))
        )

        if not problem_keywords:
            continue

        # Find incidents AFTER problem was closed with similar pattern
        post_resolution = incidents_df[
            (incidents_df['opened_at'] > problem_close_date) &
            (incidents_df['opened_at'] <= analysis_end)
        ].copy()

        if post_resolution.empty:
            continue

        # Check each incident for pattern match
        matching_incidents = []
        for _, incident in post_resolution.iterrows():
            incident_text = str(incident.get('short_description', ''))
            if keyword_similarity(problem_keywords, incident_text) > 0.4:
                matching_incidents.append(incident)

        if len(matching_incidents) >= 2:  # Threshold: at least 2 recurring incidents
            # ZOMBIE PROBLEM!
            zombie_df = pd.DataFrame(matching_incidents)

            zombies.append({
                'Problem_Number': problem['number'],
                'Problem_Description': problem.get('short_description', 'N/A')[:60],
                'Problem_Closed': problem_close_date,
                'Pattern_Keywords': ", ".join(problem_keywords),
                'Post_Resolution_Incidents': len(matching_incidents),
                'Time_Since_Closure_Days': (zombie_df['opened_at'].max() - problem_close_date).days,
                'Status': '🧟 ZOMBIE - Root cause not fixed',
                'Cost_Impact': f"${len(matching_incidents) * 50:,}",
                'Related_Incidents': ", ".join(zombie_df['number'].head(5).tolist()),
                'Severity': 'CRITICAL' if len(matching_incidents) > 5 else 'HIGH'
            })

    return pd.DataFrame(zombies).sort_values('Post_Resolution_Incidents', ascending=False)
```

---

### Enhancement 3: **Cluster Timeline Evolution**

```python
def analyze_cluster_temporal_evolution(incidents_df, time_window_months=1):
    """
    Shows how clusters evolve over time.

    Method:
    1. Cluster incidents by time windows (e.g., monthly)
    2. Track if same patterns recur across windows
    3. Identify "persistent patterns" that span multiple windows

    Returns:
        DataFrame showing pattern persistence and recurrence frequency
    """
    if incidents_df.empty:
        return pd.DataFrame()

    df = incidents_df.copy()

    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df['opened_at']):
        df['opened_at'] = pd.to_datetime(df['opened_at'], errors='coerce')

    df = df.dropna(subset=['opened_at'])

    # Create time windows
    df['time_window'] = df['opened_at'].dt.to_period(f'{time_window_months}M')

    # Dictionary to store patterns by window
    pattern_tracker = defaultdict(list)

    for window in df['time_window'].unique():
        window_data = df[df['time_window'] == window]

        # Cluster this window's data
        clustered = perform_clustering(window_data)

        for cluster_id in clustered['Cluster_ID'].unique():
            if cluster_id == -1:
                continue

            cluster = clustered[clustered['Cluster_ID'] == cluster_id]

            # Create pattern signature
            keywords = extract_pattern_keywords(
                " ".join(cluster['short_description'].fillna(''))
            )
            pattern_sig = "|".join(sorted(keywords))

            pattern_tracker[pattern_sig].append({
                'time_window': window,
                'incident_count': len(cluster),
                'incidents': cluster['number'].tolist()
            })

    # Identify recurring patterns (appear in multiple windows)
    recurring = []

    for pattern_sig, occurrences in pattern_tracker.items():
        if len(occurrences) > 1:  # Pattern recurs
            recurring.append({
                'Pattern_Signature': pattern_sig.replace('|', ', '),
                'Occurrence_Count': len(occurrences),
                'Time_Windows': ", ".join([str(occ['time_window']) for occ in occurrences]),
                'Total_Incidents': sum(occ['incident_count'] for occ in occurrences),
                'Status': '🔁 RECURRING PATTERN',
                'ITIL_Failure': 'Pattern never resolved proactively',
                'Business_Impact': f"Recurring over {len(occurrences)} periods"
            })

    return pd.DataFrame(recurring).sort_values('Occurrence_Count', ascending=False)
```

---

### Enhancement 4: **Cluster-Based ITIL Maturity Score**

```python
def calculate_cluster_based_itil_score(incidents_df, problems_df):
    """
    Enhanced ITIL maturity scoring using clustering analysis.

    Metrics:
    1. Problem coverage: % of clusters with problem records
    2. Zombie rate: % of problems that didn't prevent recurrence
    3. Pattern persistence: % of patterns that recur across time windows
    4. Proactive ratio: % of problems created BEFORE major cluster growth

    Returns:
        Score (0-100) and detailed breakdown
    """
    metrics = {
        'problem_coverage': 0,
        'zombie_rate': 0,
        'pattern_persistence': 0,
        'proactive_ratio': 0
    }

    # 1. Problem Coverage
    gaps = identify_problem_coverage_gaps(incidents_df, problems_df)
    clustered = perform_clustering(incidents_df)
    total_clusters = clustered[clustered['Cluster_ID'] != -1]['Cluster_ID'].nunique()

    if total_clusters > 0:
        gaps_count = len(gaps)
        metrics['problem_coverage'] = max(0, 100 - (gaps_count / total_clusters * 100))

    # 2. Zombie Rate
    zombies = identify_zombie_patterns(problems_df, incidents_df)
    total_problems = len(problems_df)

    if total_problems > 0:
        zombie_count = len(zombies)
        metrics['zombie_rate'] = (zombie_count / total_problems * 100)

    # 3. Pattern Persistence
    evolution = analyze_cluster_temporal_evolution(incidents_df)
    if not evolution.empty:
        persistent_count = len(evolution)
        metrics['pattern_persistence'] = persistent_count  # Lower is better

    # 4. Proactive Ratio (TODO: requires lag analysis)
    # For now, placeholder
    metrics['proactive_ratio'] = 30  # Assume 30% proactive (typical org)

    # Calculate final score
    # Weights: Coverage 30%, Zombie 30%, Persistence 20%, Proactive 20%
    score = (
        metrics['problem_coverage'] * 0.30 -
        metrics['zombie_rate'] * 0.30 -
        (min(metrics['pattern_persistence'], 10) * 2) * 0.20 +  # Cap at 10 patterns
        metrics['proactive_ratio'] * 0.20
    )

    score = max(0, min(100, score))

    return {
        'itil_maturity_score': round(score, 1),
        'grade': get_grade(score),
        'metrics': metrics,
        'interpretation': interpret_score(score)
    }


def get_grade(score):
    """Convert score to letter grade"""
    if score >= 90:
        return 'A - Excellent'
    elif score >= 80:
        return 'B - Good'
    elif score >= 70:
        return 'C - Fair'
    elif score >= 60:
        return 'D - Poor'
    else:
        return 'F - Failing'


def interpret_score(score):
    """Provide executive interpretation"""
    if score >= 80:
        return "Strong proactive problem management with minimal recurring patterns."
    elif score >= 60:
        return "Moderate ITIL maturity. Some patterns slip through. Improvement needed."
    elif score >= 40:
        return "Reactive problem management. Many recurring patterns. Significant gaps."
    else:
        return "ITIL process failures evident. High recurring patterns, poor problem coverage."
```

---

## 🎨 UI Integration (app.py)

### Add Fourth Tab to Retro Audit

```python
# In app.py, update Phase 5 section (around line 569)

tab_audit, tab_zombies, tab_deflection, tab_clusters = st.tabs([
    "The Timeline Fusion",
    "Zombie Problems",
    "Deflection Opportunity",
    "🔬 Pattern Analysis"  # NEW TAB
])

# ... existing tabs ...

with tab_clusters:
    st.subheader("🔬 Cluster-Based ITIL Failure Analysis")
    st.write("Advanced pattern detection to identify recurring issues and problem coverage gaps.")

    if not df_cleaned.empty:
        # 1. ITIL Maturity Score (Top Banner)
        if not problems_df.empty:
            with st.spinner("Calculating ITIL Maturity Score..."):
                score_result = calculate_cluster_based_itil_score(df_cleaned, problems_df)

                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    st.metric(
                        "ITIL Maturity Score",
                        f"{score_result['itil_maturity_score']}/100",
                        delta=score_result['grade']
                    )
                    st.caption(score_result['interpretation'])

                with col2:
                    st.metric(
                        "Problem Coverage",
                        f"{score_result['metrics']['problem_coverage']:.0f}%"
                    )

                with col3:
                    st.metric(
                        "Zombie Rate",
                        f"{score_result['metrics']['zombie_rate']:.0f}%",
                        delta_color="inverse"
                    )

        st.divider()

        # 2. Problem Coverage Gaps
        st.subheader("1️⃣ Problem Coverage Gaps")
        st.write("Incident clusters that should have Problem Records but don't.")

        gaps = identify_problem_coverage_gaps(df_cleaned, problems_df, min_cluster_size=3)

        if not gaps.empty:
            st.error(f"🔥 Found {len(gaps)} clusters without Problem Records!")

            # Display as expandable cards
            for idx, gap in gaps.iterrows():
                with st.expander(
                    f"Cluster {gap['Cluster_ID']}: {gap['Incident_Count']} incidents "
                    f"over {gap['Time_Span_Days']} days"
                ):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"**Pattern:** {gap['Pattern_Description']}")
                        st.markdown(f"**First/Last:** {gap['First_Occurrence'].date()} → {gap['Last_Occurrence'].date()}")

                    with col2:
                        st.markdown(f"**Business Impact:** {gap['Business_Impact']}")
                        st.markdown(f"**Cost Impact:** {gap['Cost_Impact']}")

                    st.code(f"Related Incidents: {gap['Related_Incidents']}", language='text')

                    if st.button(f"Suggest Problem Record", key=f'gap_{idx}'):
                        st.info("💡 Would create Problem Record draft here (integrate with AI Intelligence tab)")
        else:
            st.success("✅ All significant clusters have Problem Records!")

        st.divider()

        # 3. Zombie Pattern Analysis
        st.subheader("2️⃣ Zombie Pattern Analysis")
        st.write("Problem Records that failed to prevent pattern recurrence.")

        if not problems_df.empty:
            zombies = identify_zombie_patterns(problems_df, df_cleaned)

            if not zombies.empty:
                st.warning(f"🧟 Found {len(zombies)} Zombie Problems!")

                # Sort by severity
                zombies_sorted = zombies.sort_values('Post_Resolution_Incidents', ascending=False)

                st.dataframe(
                    zombies_sorted[[
                        'Problem_Number', 'Problem_Description',
                        'Post_Resolution_Incidents', 'Time_Since_Closure_Days',
                        'Cost_Impact', 'Severity'
                    ]],
                    use_container_width=True
                )

                # Show details
                selected = st.selectbox(
                    "Select Zombie Problem for details:",
                    zombies_sorted['Problem_Number'].tolist()
                )

                if selected:
                    zombie = zombies_sorted[zombies_sorted['Problem_Number'] == selected].iloc[0]

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Post-Resolution Incidents", zombie['Post_Resolution_Incidents'])
                    col2.metric("Days Since Closure", zombie['Time_Since_Closure_Days'])
                    col3.metric("Cost Impact", zombie['Cost_Impact'])

                    st.markdown(f"**Pattern Keywords:** {zombie['Pattern_Keywords']}")
                    st.code(f"Related Incidents: {zombie['Related_Incidents']}", language='text')
            else:
                st.success("✅ No zombie patterns detected! All problems effectively resolved.")
        else:
            st.info("Load Problem data to analyze zombie patterns.")

        st.divider()

        # 4. Temporal Evolution
        st.subheader("3️⃣ Recurring Pattern Timeline")
        st.write("Patterns that persist across multiple time windows (months).")

        evolution = analyze_cluster_temporal_evolution(df_cleaned, time_window_months=1)

        if not evolution.empty:
            st.warning(f"🔁 Found {len(evolution)} recurring patterns!")

            st.dataframe(
                evolution[[
                    'Pattern_Signature', 'Occurrence_Count',
                    'Total_Incidents', 'Time_Windows', 'ITIL_Failure'
                ]],
                use_container_width=True
            )
        else:
            st.success("✅ No patterns recurring across multiple time windows.")
    else:
        st.warning("Load incident data to perform cluster analysis.")
```

---

## 📊 Expected Outputs

### 1. Problem Coverage Gaps
```
🔥 Found 8 clusters without Problem Records!

Cluster 12: 15 incidents over 45 days
├─ Pattern: "VPN connection timeout for remote users"
├─ First/Last: 2025-10-01 → 2025-11-15
├─ Business Impact: 15 incidents over 45 days
├─ Cost Impact: $750
└─ Related Incidents: INC0012345, INC0012567, INC0012789...
```

### 2. Zombie Pattern Analysis
```
🧟 Found 5 Zombie Problems!

PRB0001234: Database connection pool exhaustion
├─ Problem Closed: 2025-09-15
├─ Post-Resolution Incidents: 8
├─ Time Since Closure: 67 days
├─ Cost Impact: $400
├─ Severity: HIGH
└─ Status: 🧟 ZOMBIE - Root cause not fixed
```

### 3. ITIL Maturity Score
```
╔════════════════════════════════════╗
║  ITIL Maturity Score: 58/100      ║
║  Grade: D - Poor                   ║
╠════════════════════════════════════╣
║  Problem Coverage: 65%             ║
║  Zombie Rate: 22%                  ║
║  Recurring Patterns: 12            ║
║  Interpretation:                   ║
║  "Reactive problem management.     ║
║   Many recurring patterns.         ║
║   Significant gaps."               ║
╚════════════════════════════════════╝
```

---

## 🎯 Business Impact

| Feature | ITIL Failure Demonstrated | Stakeholder Value |
|---------|---------------------------|-------------------|
| **Problem Coverage Gaps** | "We have recurring patterns but no Problem Records" | Shows $XX,XXX wasted on repeat fixes |
| **Zombie Patterns** | "We created Problems but pattern continues" | Proves root causes aren't being fixed |
| **ITIL Maturity Score** | Single metric: 58/100 | Executive dashboard number |
| **Temporal Evolution** | "Same issue every month for 6 months" | Timeline proof of reactive behavior |

---

## ✅ Implementation Priority

| Component | Lines of Code | Effort | Impact | Priority |
|-----------|---------------|--------|--------|----------|
| Problem Coverage Gaps | ~100 | Medium | 🔥 Very High | **HIGH** |
| Zombie Pattern Analysis | ~80 | Medium | 🔥 Very High | **HIGH** |
| ITIL Maturity Score | ~60 | Low-Medium | 🔥 Very High | **HIGH** |
| Temporal Evolution | ~100 | Medium | High | **MEDIUM** |
| UI Integration | ~150 | Medium | High | **HIGH** |

**Total Effort:** 3-5 days
**Total Impact:** Transforms Retro Audit into definitive ITIL failure proof

---

## 📝 Next Steps

1. **Create `retro_clustering.py`** - New module for clustering-specific retro analysis
2. **Update `retro_analysis.py`** - Import and integrate new functions
3. **Enhance `app.py`** - Add fourth tab "Pattern Analysis"
4. **Update `RETRO_AUDIT_ANALYSIS.md`** - Document new capabilities

---

**This enhancement answers the question:**
> "You say ITIL is reactive, but can you PROVE it with data?"

**Answer:**
✅ Yes - here are the exact clusters, zombie patterns, and coverage gaps
✅ Here's your ITIL Maturity Score: 58/100
✅ Here's exactly how much it's costing: $XX,XXX/month
