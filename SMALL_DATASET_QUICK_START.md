# Small Dataset ML - Quick Start Guide

## 🚨 Your Current Situation
- **Intelligent Routing:** Only 21 training samples
- **Solution Recommender:** Only 3 resolved incidents with notes
- **Problem:** Poor predictions, no search results, loss of trust

## ⚡ Quick Fix (Implement Today - 2-3 hours)

### Step 1: Add Data Quality Checks (30 min)
```python
# Copy from SMALL_DATASET_SOLUTIONS.md - Solution #1
from aiops_intelligence import DataQualityChecker

quality = DataQualityChecker.check_routing_data_quality(df_cleaned)

if not quality['sufficient']:
    st.error(f"⚠️ {quality['reason']}")
    st.info("Using rule-based fallback instead")
    # Use fallback system
```

### Step 2: Add Rule-Based Fallback (1 hour)
```python
# Copy from SMALL_DATASET_SOLUTIONS.md - Solution #2
class RuleBasedRouter:
    ROUTING_RULES = {
        'NETWORK_INFRASTRUCTURE': {
            'keywords': ['network', 'vpn', 'firewall', 'dns', 'outage'],
            'confidence': 0.85
        },
        # ... add your groups
    }
```

### Step 3: Add Confidence Thresholds (30 min)
```python
# Copy from SMALL_DATASET_SOLUTIONS.md - Solution #6
predictions = router.predict_assignment(desc, min_confidence=0.60)

for pred in predictions:
    if pred['confidence'] < 0.60:
        st.warning("⚠️ Low confidence - manual review recommended")
```

### Step 4: Add Demo Mode (1 hour)
```python
# Copy from SMALL_DATASET_SOLUTIONS.md - Solution #4
if data_insufficient:
    st.warning("🎭 Demo Mode: Using curated examples")
    df = DemoDataProvider.get_demo_incidents()
```

## 📋 3-Week Implementation Plan

### Week 1: Safety & Fallbacks
**Goal:** Never show bad predictions

- [ ] Day 1-2: Implement minimum thresholds (Solution #1)
- [ ] Day 3-4: Add rule-based routing (Solution #2)
- [ ] Day 5: Add confidence scores (Solution #6)

**Result:** ✅ Features work safely with any data size

### Week 2: User Experience
**Goal:** Users understand requirements

- [ ] Day 1-2: Add data quality dashboard (Solution #8)
- [ ] Day 3-4: Implement demo mode (Solution #4)
- [ ] Day 5: Testing and refinement

**Result:** ✅ Clear messaging, working demos

### Week 3: Long-term Foundation
**Goal:** System improves over time

- [ ] Day 1-3: Implement feedback collection (Solution #5)
- [ ] Day 4-5: Add progressive enhancement (Solution #7)

**Result:** ✅ Path to production-ready ML

## 🎯 Which Solution When?

| Scenario | Use Solutions | Priority |
|----------|---------------|----------|
| **Right Now** (< 20 incidents) | #1, #2, #4, #6 | 🔴 Critical |
| **This Month** (20-50 incidents) | #8, #5 | 🟡 High |
| **Next 90 Days** (50-100 incidents) | #7, #3 | 🟢 Medium |
| **Production** (100+ incidents) | All except #3, #4 | ✅ Maintain |

## 📊 Your Roadmap to 100+ Incidents

### Current State (Today)
- 21 incidents → **Demo Mode**
- Show curated examples
- Collect feedback
- Use rule-based routing

### Month 1 Target: 50 Incidents
- Enable **Advisory Mode**
- Show ML suggestions (read-only)
- Collect user corrections
- 70% have resolution notes

### Month 2 Target: 75 Incidents
- Enable **Assisted Mode**
- One-click apply suggestions
- Track acceptance rate
- 80% have resolution notes

### Month 3 Target: 100+ Incidents
- Enable **Automated Mode**
- Confident predictions auto-applied
- Human oversight
- >70% acceptance rate

## 🔑 Critical Success Factors

### Data Quality Requirements
```
✅ Minimum 50 resolved incidents
✅ At least 3 assignment groups
✅ 5+ examples per group
✅ 80%+ have resolution notes
✅ Consistent group naming
```

### For Similar Incidents Feature
```
✅ Minimum 20 resolved incidents
✅ 20+ with close_notes populated
✅ Diverse incident types
✅ Clear resolution descriptions
```

## 💡 Quick Wins (Do First)

1. **Make resolution notes mandatory** when closing incidents
   - Biggest impact for Similar Incidents feature
   - Simple process change

2. **Standardize assignment group names**
   - Critical for routing accuracy
   - Clean up existing data

3. **Add feedback buttons** to all ML predictions
   - "Was this helpful?" → Build training data
   - 10-20 corrections = noticeable improvement

4. **Show data quality progress** on Dashboard
   - Gamify data collection
   - "45/100 incidents to unlock Assisted Mode"

## 🚫 What NOT to Do

❌ **Don't** train ML with <20 incidents
   - Use rule-based instead

❌ **Don't** hide that data is limited
   - Be transparent: "Demo Mode" or "Limited Data"

❌ **Don't** auto-apply predictions with <60% confidence
   - Always require human review

❌ **Don't** rely on synthetic data alone
   - Use max 20-30% synthetic, rest real

❌ **Don't** skip confidence calibration
   - Raw model confidence ≠ real accuracy

## 📈 Expected Outcomes

### After Week 1
- ✅ No more bad predictions shown
- ✅ Features work with current data
- ✅ Clear error messages

### After Week 2
- ✅ Users understand requirements
- ✅ Demo mode shows value
- ✅ Data quality improving

### After Week 3
- ✅ Collecting user feedback
- ✅ Path to production clear
- ✅ Early improvements visible

### After 90 Days
- ✅ 100+ quality incidents
- ✅ ML accuracy >70%
- ✅ User trust established
- ✅ Production-ready

## 📚 Full Documentation

See **SMALL_DATASET_SOLUTIONS.md** for:
- Complete code implementations
- All 9 solutions in detail
- UI integration examples
- Best practices
- Decision matrices

## 🆘 Emergency Fixes

### If users complaining about bad predictions NOW:
```python
# Immediate: Disable ML, use rules only
USE_ML_ROUTING = False  # Set to False immediately

if USE_ML_ROUTING and len(training_data) >= 50:
    predictions = ml_router.predict()
else:
    predictions = rule_router.predict()  # Always reliable
```

### If features showing errors:
```python
# Add try/catch with friendly fallback
try:
    predictions = ml_router.predict()
except Exception as e:
    st.warning("ML temporarily unavailable - using rule-based routing")
    predictions = rule_router.predict()
```

### If users don't understand limitations:
```python
# Add prominent banner
st.warning("""
⚠️ **Limited Training Data Mode**

ML features require 50+ resolved incidents for reliable predictions.
Currently using rule-based routing (always accurate).

Progress: {current}/{target} incidents
""")
```

## 🎯 Start Here

1. ✅ Read this document (you just did!)
2. ✅ Review your current data:
   ```python
   resolved = len(df[df['state'].isin(['Closed', 'Resolved'])])
   with_notes = len(df[df['close_notes'].notna()])
   print(f"Resolved: {resolved}, With notes: {with_notes}")
   ```
3. ✅ Implement Week 1 fixes (Solutions #1, #2, #6)
4. ✅ Add data quality dashboard (Solution #8)
5. ✅ Create 90-day data collection plan

**Questions?** Review specific solutions in SMALL_DATASET_SOLUTIONS.md

**Need code?** All implementations included with copy-paste examples

**Ready to start?** Begin with Solution #1 (takes 30 minutes)
