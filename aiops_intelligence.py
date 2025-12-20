"""
AIOps Intelligence Module

This module contains advanced AIOps capabilities including:
- Similar Incident Recommendation
- Intelligent Assignment/Routing
- Auto-Problem Creation Suggestions

These features demonstrate the predictive and prescriptive capabilities
that differentiate AIOps from traditional reactive ITSM.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import re

# Import logging infrastructure
try:
    from aiops_logging import track_performance, business_logger, audit_logger, track_roi
    LOGGING_ENABLED = True
except ImportError:
    LOGGING_ENABLED = False
    # Create no-op decorators if logging not available
    def track_performance(name=None):
        def decorator(func):
            return func
        return decorator


# ============================================================================
# 1. SIMILAR INCIDENT RECOMMENDATION
# ============================================================================

@track_performance('find_similar_incidents')
def find_similar_resolved_incidents(new_incident_desc, historical_df, top_n=3):
    """
    Find similar resolved incidents from historical data using TF-IDF and cosine similarity.

    This helps agents resolve tickets faster by showing how similar issues were resolved.

    Args:
        new_incident_desc (str): Description of the new incident
        historical_df (pd.DataFrame): Historical incidents with 'short_description', 'description',
                                      'state', 'number', 'close_notes', 'sys_updated_at'
        top_n (int): Number of similar incidents to return

    Returns:
        list of dict: Similar incidents with similarity scores and resolution info

    Example:
        >>> similar = find_similar_resolved_incidents("Database timeout error", historical_df)
        >>> print(similar[0])
        {
            'incident_number': 'INC0012345',
            'similarity_score': 0.89,
            'short_description': 'DB connection timeout',
            'resolution_notes': 'Cleared connection pool and restarted service',
            'resolution_time_hours': 2.5
        }
    """
    if historical_df.empty:
        return []

    # Filter to only resolved/closed incidents with resolution notes
    resolved_mask = historical_df['state'].isin(['Closed', 'Resolved'])
    resolved_df = historical_df[resolved_mask].copy()

    if resolved_df.empty:
        return []

    # Filter to incidents that have close_notes (actual resolutions)
    if 'close_notes' in resolved_df.columns:
        resolved_df = resolved_df[resolved_df['close_notes'].notna() & (resolved_df['close_notes'] != '')]

    if resolved_df.empty:
        return []

    # Combine description fields for better matching
    resolved_df['combined_text'] = (
        resolved_df['short_description'].fillna('') + ' ' +
        resolved_df.get('description', pd.Series([''] * len(resolved_df))).fillna('')
    )

    # Prepare corpus: historical incidents + new incident
    corpus = resolved_df['combined_text'].tolist() + [new_incident_desc]

    # Vectorize using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Calculate cosine similarity between new incident (last row) and all historical
    new_incident_vector = tfidf_matrix[-1]
    historical_vectors = tfidf_matrix[:-1]

    similarities = cosine_similarity(new_incident_vector, historical_vectors).flatten()

    # Get top N similar incidents
    top_indices = similarities.argsort()[-top_n:][::-1]

    results = []
    for idx in top_indices:
        if similarities[idx] > 0.1:  # Minimum threshold
            incident = resolved_df.iloc[idx]

            # Calculate resolution time if dates available
            resolution_time = None
            if 'opened_at' in incident and 'closed_at' in incident:
                try:
                    opened = pd.to_datetime(incident['opened_at'])
                    closed = pd.to_datetime(incident['closed_at'])
                    resolution_time = (closed - opened).total_seconds() / 3600  # hours
                except:
                    pass

            results.append({
                'incident_number': incident['number'],
                'similarity_score': float(similarities[idx]),
                'short_description': incident['short_description'],
                'resolution_notes': incident.get('close_notes', 'N/A'),
                'resolution_time_hours': resolution_time,
                'assignment_group': incident.get('assignment_group', 'N/A')
            })

    return results


# ============================================================================
# 2. INTELLIGENT ASSIGNMENT / ROUTING
# ============================================================================

class IntelligentRouter:
    """
    ML-based incident assignment router.

    Uses historical resolution data to predict which team should handle new incidents.
    """

    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.trained = False

    def train(self, historical_df):
        """
        Train the assignment model on historical incident data.

        Args:
            historical_df: DataFrame with 'short_description', 'description',
                          'assignment_group', 'state' (resolved/closed)

        Returns:
            dict: Training metrics (accuracy, coverage)
        """
        if historical_df.empty:
            return {'error': 'No training data'}

        # Filter to resolved incidents with assignment groups
        resolved_mask = historical_df['state'].isin(['Closed', 'Resolved'])
        train_df = historical_df[resolved_mask].copy()

        if 'assignment_group' not in train_df.columns:
            return {'error': 'No assignment_group column'}

        train_df = train_df[train_df['assignment_group'].notna()]

        if len(train_df) < 10:
            return {'error': 'Insufficient training data (need at least 10 resolved tickets)'}

        # Combine text fields
        train_df['combined_text'] = (
            train_df['short_description'].fillna('') + ' ' +
            train_df.get('description', pd.Series([''] * len(train_df))).fillna('')
        )

        # Vectorize text
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=300, min_df=2)
        X = self.vectorizer.fit_transform(train_df['combined_text'])

        # Encode labels
        y = train_df['assignment_group']

        # Train RandomForest
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)

        self.trained = True

        # Calculate training accuracy
        train_accuracy = self.model.score(X, y)

        # Get assignment group distribution
        group_counts = y.value_counts().to_dict()

        metrics = {
            'success': True,
            'training_accuracy': float(train_accuracy),
            'num_training_samples': len(train_df),
            'num_assignment_groups': len(group_counts),
            'assignment_groups': group_counts
        }

        # Log training event
        if LOGGING_ENABLED:
            audit_logger.log_model_training(
                model_type='IntelligentRouter',
                training_samples=len(train_df),
                accuracy=train_accuracy,
                hyperparameters={'n_estimators': 100, 'max_features': 300}
            )

        return metrics

    @track_performance('intelligent_router_predict')
    def predict_assignment(self, incident_description, top_n=3):
        """
        Predict the best assignment group for a new incident.

        Args:
            incident_description (str): Description of the incident
            top_n (int): Number of top predictions to return

        Returns:
            list of dict: Predictions with confidence scores

        Example:
            >>> predictions = router.predict_assignment("VPN connection failed")
            >>> print(predictions[0])
            {
                'assignment_group': 'Network Infrastructure',
                'confidence': 0.94,
                'reasoning': 'Keywords: vpn, connection, network'
            }
        """
        if not self.trained:
            return [{'error': 'Model not trained yet'}]

        # Vectorize new incident
        X_new = self.vectorizer.transform([incident_description])

        # Get prediction probabilities
        proba = self.model.predict_proba(X_new)[0]
        classes = self.model.classes_

        # Get top N predictions
        top_indices = proba.argsort()[-top_n:][::-1]

        results = []
        for idx in top_indices:
            # Extract important keywords (simple approach)
            feature_names = self.vectorizer.get_feature_names_out()
            feature_values = X_new.toarray()[0]
            top_features_idx = feature_values.argsort()[-5:][::-1]
            keywords = [feature_names[i] for i in top_features_idx if feature_values[i] > 0]

            results.append({
                'assignment_group': classes[idx],
                'confidence': float(proba[idx]),
                'reasoning': f"Keywords: {', '.join(keywords[:3])}" if keywords else 'Pattern match'
            })

        return results


# ============================================================================
# 3. AUTO-PROBLEM CREATION SUGGESTIONS
# ============================================================================

@track_performance('suggest_problem_creation')
def suggest_problem_creation(cluster_df, cluster_id, threshold=5):
    """
    Suggest creating a Problem Record when incident clusters reach critical mass.

    Args:
        cluster_df (pd.DataFrame): Clustered incidents with 'Cluster_ID' column
        cluster_id (int): The cluster to analyze
        threshold (int): Minimum incidents to trigger problem creation

    Returns:
        dict or None: Problem record suggestion with details

    Example:
        >>> suggestion = suggest_problem_creation(clustered_df, cluster_id=3)
        >>> print(suggestion)
        {
            'should_create': True,
            'problem_title': 'Recurring issue: database timeout connection',
            'incident_count': 8,
            'related_incidents': ['INC001', 'INC002', ...],
            'affected_assets': ['DB-PROD-01', '10.0.0.15'],
            'priority': 'High',
            'business_impact': 'Multiple database connection failures',
            'recommended_actions': [...]
        }
    """
    if cluster_df.empty or 'Cluster_ID' not in cluster_df.columns:
        return None

    # Get incidents in this cluster
    cluster = cluster_df[cluster_df['Cluster_ID'] == cluster_id].copy()

    if len(cluster) < threshold:
        return None  # Not enough incidents to warrant a problem record

    # Extract common keywords from descriptions
    all_text = ' '.join(cluster['short_description'].fillna('').astype(str))

    # Simple keyword extraction (remove stopwords)
    stopwords = {'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'on', 'at', 'for', 'with', 'by',
                 'is', 'it', 'this', 'that', 'issue', 'error', 'problem'}
    words = re.findall(r'\w+', all_text.lower())
    word_freq = Counter([w for w in words if w not in stopwords and len(w) > 3])
    top_keywords = [word for word, _ in word_freq.most_common(5)]

    # Extract affected assets/entities using regex
    from analysis import extract_entities  # Import from existing module

    all_entities = []
    for desc in cluster['short_description'].fillna(''):
        all_entities.extend(extract_entities(str(desc)))

    entity_freq = Counter(all_entities)
    affected_assets = [entity for entity, _ in entity_freq.most_common(10)]

    # Determine priority based on incident priorities
    if 'priority' in cluster.columns:
        priorities = cluster['priority'].value_counts()
        # If most incidents are high priority, problem is high priority
        if len(priorities) > 0 and priorities.index[0] in ['1', 'Critical', 'High', '1 - Critical', '2 - High']:
            priority = 'High'
        else:
            priority = 'Medium'
    else:
        priority = 'Medium'

    # Calculate time span
    if 'opened_at' in cluster.columns:
        cluster['opened_at'] = pd.to_datetime(cluster['opened_at'], errors='coerce')
        date_range = cluster['opened_at'].max() - cluster['opened_at'].min()
        days_span = date_range.days if pd.notna(date_range) else 0
    else:
        days_span = 0

    # Generate problem title
    problem_title = f"Recurring issue: {' '.join(top_keywords[:4])}"

    # Generate recommended actions
    recommended_actions = [
        "Conduct root cause analysis across all related incidents",
        f"Investigate common assets: {', '.join(affected_assets[:3])}" if affected_assets else "Investigate affected infrastructure",
        "Review recent changes that may have introduced this issue",
        "Implement permanent fix to prevent recurrence"
    ]

    return {
        'should_create': True,
        'problem_title': problem_title,
        'incident_count': len(cluster),
        'related_incidents': cluster['number'].tolist(),
        'affected_assets': affected_assets,
        'priority': priority,
        'business_impact': f"{len(cluster)} incidents over {days_span} days affecting similar systems",
        'top_keywords': top_keywords,
        'assignment_group': cluster['assignment_group'].mode()[0] if 'assignment_group' in cluster.columns and len(cluster['assignment_group'].mode()) > 0 else 'Problem Management',
        'recommended_actions': recommended_actions,
        'time_span_days': days_span
    }


@track_performance('batch_suggest_problems')
def batch_suggest_problems(cluster_df, threshold=5):
    """
    Analyze all clusters and suggest which ones warrant Problem Record creation.

    Args:
        cluster_df (pd.DataFrame): All clustered incidents
        threshold (int): Minimum incidents per cluster

    Returns:
        list of dict: Problem suggestions for all qualifying clusters
    """
    if cluster_df.empty or 'Cluster_ID' not in cluster_df.columns:
        return []

    # Get all valid clusters (exclude noise cluster -1)
    valid_clusters = cluster_df[cluster_df['Cluster_ID'] != -1]['Cluster_ID'].unique()

    suggestions = []
    for cluster_id in valid_clusters:
        suggestion = suggest_problem_creation(cluster_df, cluster_id, threshold)
        if suggestion:
            suggestion['cluster_id'] = int(cluster_id)
            suggestions.append(suggestion)

    # Sort by incident count (most critical first)
    suggestions.sort(key=lambda x: x['incident_count'], reverse=True)

    return suggestions


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_mttr_improvement(similar_incidents):
    """
    Calculate potential MTTR improvement from similar incident recommendations.

    Args:
        similar_incidents (list): List of similar incident recommendations

    Returns:
        dict: Improvement metrics
    """
    if not similar_incidents:
        return {'avg_resolution_time': None, 'potential_improvement': None}

    resolution_times = [inc['resolution_time_hours'] for inc in similar_incidents
                       if inc.get('resolution_time_hours') is not None]

    if not resolution_times:
        return {'avg_resolution_time': None, 'potential_improvement': None}

    avg_time = np.mean(resolution_times)

    # Estimate: Using similar incidents reduces resolution time by ~30%
    estimated_improvement = avg_time * 0.3

    return {
        'avg_historical_resolution_hours': round(avg_time, 2),
        'estimated_time_savings_hours': round(estimated_improvement, 2),
        'confidence': 'Based on historical similar incidents'
    }
