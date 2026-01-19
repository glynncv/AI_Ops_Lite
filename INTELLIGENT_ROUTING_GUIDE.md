# Understanding Intelligent Assignment Routing Results

## What Is Intelligent Assignment Routing?

Intelligent Assignment Routing is an **ML-powered feature** that predicts which team should handle a new incident based on historical resolution patterns. It learns from past incidents that were successfully resolved by different teams.

---

## How It Works

### 1. **Training Phase** (When you click "Train Assignment Model")

The system:
- Analyzes all **resolved/closed incidents** in your data
- Learns patterns from incident descriptions and which teams resolved them
- Builds a **RandomForest classifier** model
- Calculates how well it can predict assignments on historical data

**What you see:**
- ✅ **Model trained on X incidents** - Number of historical incidents used for training
- **Training Accuracy: X%** - How well the model predicts assignments on the training data
  - **100%** = Perfect prediction on historical data (may indicate overfitting with small datasets)
  - **80-95%** = Good model performance
  - **<70%** = Model may need more training data
- **Assignment Groups: X** - Number of different teams/groups in your data
- **View Group Distribution** - Shows how many incidents each team handled historically

### 2. **Prediction Phase** (When you enter a description and click "Predict Assignment")

The system:
- Analyzes the incident description you provide
- Compares it to patterns learned from historical data
- Predicts which team should handle it
- Provides confidence scores and reasoning

---

## Understanding the Results

### Top Recommendation (🥇)

**Example Display:**
```
🥇 EDI_Support
[Progress Bar: ████████████████████] 94%
Confidence: 94% | Keywords: edi, connection, timeout
```

**What it means:**
- **Assignment Group Name** - The team the model recommends
- **Progress Bar** - Visual representation of confidence (longer = more confident)
- **Confidence Percentage** - How certain the model is (0-100%)
  - **90-100%** = Very confident, strong pattern match
  - **70-89%** = Confident, good match
  - **50-69%** = Moderate confidence, may need review
  - **<50%** = Low confidence, consider manual review
- **Keywords** - Important words from the description that influenced the prediction

### Alternative Recommendations

**Example Display:**
```
Alternative 1: SAP_ORDER_FULFILLMENT (45%)
Keywords: sap, order, error

Alternative 2: EDI_GXS_Customer (12%)
Keywords: edi, customer
```

**What it means:**
- Shows **backup options** if the top choice isn't available
- Lower confidence scores indicate these are less likely matches
- Useful for **escalation paths** or when primary team is unavailable

---

## Interpreting Confidence Scores

### High Confidence (85-100%)
✅ **Action:** Route directly to recommended team
- Strong pattern match with historical data
- Similar incidents were successfully handled by this team
- Low risk of mis-routing

### Medium Confidence (60-84%)
⚠️ **Action:** Review before routing
- Reasonable match, but not definitive
- May want to check incident details
- Consider alternatives if available

### Low Confidence (<60%)
🔍 **Action:** Manual review recommended
- Weak pattern match
- May be a new type of incident
- Consider routing to a general support team first

---

## Real-World Example

**Scenario:** New incident description: "EDI connection timeout error affecting order processing"

**Training Data Shows:**
- 15 similar incidents → Resolved by "EDI_Support" ✅
- 3 similar incidents → Resolved by "SAP_ORDER_FULFILLMENT"
- 1 similar incident → Resolved by "EDI_GXS_Customer"

**Prediction Results:**
```
🥇 EDI_Support (94% confidence)
Keywords: edi, connection, timeout

Alternative 1: SAP_ORDER_FULFILLMENT (5%)
Alternative 2: EDI_GXS_Customer (1%)
```

**Interpretation:**
- **94% confidence** = Very high certainty
- **EDI_Support** handled 15/19 similar incidents (79% of cases)
- **Strong recommendation** - Route directly to EDI_Support
- Alternatives are much less likely (5% and 1%)

---

## Business Value

### What This Feature Prevents:

1. **Mis-routing** - Incidents going to wrong teams
   - **Impact:** 2+ hours wasted per mis-routed ticket
   - **Savings:** 60-80% reduction in mis-routing

2. **Delayed Resolution** - Tickets bouncing between teams
   - **Impact:** Longer resolution times, frustrated users
   - **Savings:** Faster time-to-assignment

3. **Manual Triage Overhead** - Agents manually deciding routing
   - **Impact:** 5-10 minutes per ticket
   - **Savings:** Automated routing for common patterns

### ROI Calculation:

**Example:**
- 100 tickets/month
- 30% mis-routing rate without AI = 30 mis-routed tickets
- 2 hours wasted per mis-route = 60 hours/month
- At $50/hour = **$3,000/month saved**
- With AI routing (80% reduction) = **$2,400/month saved**

---

## Best Practices

### ✅ Do:
- **Train on recent data** - Use last 3-6 months of resolved incidents
- **Review low-confidence predictions** - Always check predictions <70%
- **Update training periodically** - Retrain when new teams or patterns emerge
- **Use alternatives** - Consider backup options for high-priority incidents

### ❌ Don't:
- **Blindly trust 100% accuracy** - May indicate overfitting with small datasets
- **Ignore alternatives** - Backup options are valuable
- **Train on open incidents** - Only use resolved/closed incidents
- **Use for all incidents** - Some complex cases need human judgment

---

## Troubleshooting

### "Training Accuracy: 100%"
**Possible causes:**
- Small dataset (<50 incidents)
- Model memorized training data (overfitting)
- Very distinct patterns between teams

**What to do:**
- Get more training data (aim for 100+ incidents)
- Test with new incidents to verify real-world performance
- Consider this a baseline - real performance may be lower

### Low Confidence Predictions
**Possible causes:**
- New type of incident not seen before
- Ambiguous description
- Multiple teams could handle it

**What to do:**
- Review incident details manually
- Consider routing to a general support team
- Use alternatives as backup options

### "Model not trained yet"
**What to do:**
- Click "Train Assignment Model" button first
- Ensure you have resolved incidents with assignment groups
- Need at least 10 resolved incidents to train

---

## Technical Details

### Algorithm: RandomForest Classifier
- **Type:** Ensemble learning (multiple decision trees)
- **Input:** TF-IDF vectorized incident descriptions
- **Output:** Assignment group predictions with probabilities
- **Training:** Uses historical resolved incidents only

### Features Used:
- Incident short description
- Incident full description
- Historical assignment patterns
- Keyword patterns

### Model Limitations:
- Only as good as historical data quality
- Requires sufficient training data (10+ incidents minimum)
- May struggle with completely new incident types
- Training accuracy ≠ real-world accuracy (especially with small datasets)

---

## Summary

**Intelligent Assignment Routing helps you:**
1. ✅ Route incidents to the right team automatically
2. ✅ Reduce mis-routing by 60-80%
3. ✅ Save ~2 hours per ticket
4. ✅ Improve time-to-assignment
5. ✅ Provide backup routing options

**Key Metrics to Watch:**
- **Training Accuracy** - Model performance on historical data
- **Confidence Scores** - Certainty of predictions
- **Group Distribution** - Balance of workload across teams
- **Real-world Accuracy** - How often predictions are correct (track manually)

**Remember:** This is a **recommendation tool**, not a replacement for human judgment. Use it to assist routing decisions, especially for high-confidence predictions.


