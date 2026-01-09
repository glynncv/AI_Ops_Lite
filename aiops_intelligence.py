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
# DATA QUALITY CHECKER (Solution #1: Minimum Data Thresholds)
# ============================================================================

class DataQualityChecker:
    """
    Checks if there's sufficient training data for ML features.
    Prevents poor predictions by enforcing minimum data thresholds.
    """

    MINIMUM_THRESHOLDS = {
        'intelligent_routing': {
            'min_total_incidents': 50,
            'min_per_group': 5,
            'min_groups': 3,
            'min_resolved': 50
        },
        'similar_incidents': {
            'min_resolved': 20,
            'min_with_notes': 10,
            'min_for_search': 5
        },
        'problem_detection': {
            'min_total': 30,
            'min_cluster_size': 3
        }
    }

    @staticmethod
    def check_routing_data_quality(df):
        """
        Check if there's enough data to train intelligent routing.

        Args:
            df: DataFrame with incident data

        Returns:
            dict: Quality assessment with recommendation
        """
        if df.empty:
            return {
                'sufficient': False,
                'reason': 'No incident data available',
                'confidence': 'none',
                'recommended_action': 'use_fallback',
                'metrics': {}
            }

        # Check resolved incidents
        resolved = df[df['state'].isin(['Closed', 'Resolved'])]
        num_resolved = len(resolved)

        if num_resolved < DataQualityChecker.MINIMUM_THRESHOLDS['intelligent_routing']['min_resolved']:
            return {
                'sufficient': False,
                'reason': f"Only {num_resolved} resolved incidents. Need at least {DataQualityChecker.MINIMUM_THRESHOLDS['intelligent_routing']['min_resolved']} for reliable ML predictions.",
                'confidence': 'low',
                'recommended_action': 'use_fallback',
                'metrics': {
                    'resolved_incidents': num_resolved,
                    'target': DataQualityChecker.MINIMUM_THRESHOLDS['intelligent_routing']['min_resolved'],
                    'progress_pct': round(num_resolved / DataQualityChecker.MINIMUM_THRESHOLDS['intelligent_routing']['min_resolved'] * 100, 1)
                }
            }

        # Check assignment group distribution
        if 'assignment_group' in resolved.columns:
            group_counts = resolved['assignment_group'].value_counts()
            num_groups = len(group_counts)
            min_group_size = group_counts.min() if len(group_counts) > 0 else 0

            if num_groups < DataQualityChecker.MINIMUM_THRESHOLDS['intelligent_routing']['min_groups']:
                return {
                    'sufficient': False,
                    'reason': f"Only {num_groups} assignment groups. Need at least {DataQualityChecker.MINIMUM_THRESHOLDS['intelligent_routing']['min_groups']}.",
                    'confidence': 'low',
                    'recommended_action': 'use_fallback',
                    'metrics': {
                        'num_groups': num_groups,
                        'target_groups': DataQualityChecker.MINIMUM_THRESHOLDS['intelligent_routing']['min_groups']
                    }
                }

            if min_group_size < DataQualityChecker.MINIMUM_THRESHOLDS['intelligent_routing']['min_per_group']:
                return {
                    'sufficient': False,
                    'reason': f"Smallest assignment group has only {min_group_size} incidents. Need at least {DataQualityChecker.MINIMUM_THRESHOLDS['intelligent_routing']['min_per_group']} per group.",
                    'confidence': 'medium',
                    'recommended_action': 'use_fallback',
                    'metrics': {
                        'min_group_size': min_group_size,
                        'target_per_group': DataQualityChecker.MINIMUM_THRESHOLDS['intelligent_routing']['min_per_group'],
                        'group_distribution': group_counts.to_dict()
                    }
                }

        # Data is sufficient
        return {
            'sufficient': True,
            'reason': f"Sufficient data: {num_resolved} resolved incidents across {num_groups} groups",
            'confidence': 'high',
            'recommended_action': 'use_ml',
            'metrics': {
                'resolved_incidents': num_resolved,
                'num_groups': num_groups,
                'min_group_size': min_group_size,
                'avg_group_size': round(num_resolved / num_groups, 1) if num_groups > 0 else 0
            }
        }

    @staticmethod
    def check_similar_incidents_data_quality(df):
        """
        Check if there's enough data for similar incident search.

        Args:
            df: DataFrame with incident data

        Returns:
            dict: Quality assessment
        """
        if df.empty:
            return {
                'sufficient': False,
                'reason': 'No incident data available',
                'recommended_action': 'show_message',
                'metrics': {}
            }

        # Check resolved incidents
        resolved = df[df['state'].isin(['Closed', 'Resolved'])]
        num_resolved = len(resolved)

        # Check incidents with resolution notes
        if 'close_notes' in resolved.columns:
            with_notes = resolved[resolved['close_notes'].notna() & (resolved['close_notes'] != '')]
            num_with_notes = len(with_notes)
        else:
            num_with_notes = 0

        if num_resolved < DataQualityChecker.MINIMUM_THRESHOLDS['similar_incidents']['min_resolved']:
            return {
                'sufficient': False,
                'reason': f"Only {num_resolved} resolved incidents. Need at least {DataQualityChecker.MINIMUM_THRESHOLDS['similar_incidents']['min_resolved']} for meaningful similarity search.",
                'recommended_action': 'show_message',
                'metrics': {
                    'resolved_incidents': num_resolved,
                    'target': DataQualityChecker.MINIMUM_THRESHOLDS['similar_incidents']['min_resolved'],
                    'progress_pct': round(num_resolved / DataQualityChecker.MINIMUM_THRESHOLDS['similar_incidents']['min_resolved'] * 100, 1)
                }
            }

        if num_with_notes < DataQualityChecker.MINIMUM_THRESHOLDS['similar_incidents']['min_with_notes']:
            return {
                'sufficient': False,
                'reason': f"Only {num_with_notes} incidents have resolution notes. Need at least {DataQualityChecker.MINIMUM_THRESHOLDS['similar_incidents']['min_with_notes']} for quality recommendations.",
                'recommended_action': 'partial_feature',
                'metrics': {
                    'incidents_with_notes': num_with_notes,
                    'target': DataQualityChecker.MINIMUM_THRESHOLDS['similar_incidents']['min_with_notes'],
                    'progress_pct': round(num_with_notes / DataQualityChecker.MINIMUM_THRESHOLDS['similar_incidents']['min_with_notes'] * 100, 1)
                }
            }

        # Data is sufficient
        return {
            'sufficient': True,
            'reason': f"Sufficient data: {num_with_notes} resolved incidents with resolution notes",
            'recommended_action': 'use_feature',
            'metrics': {
                'resolved_incidents': num_resolved,
                'incidents_with_notes': num_with_notes,
                'notes_coverage_pct': round(num_with_notes / num_resolved * 100, 1) if num_resolved > 0 else 0
            }
        }


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
# RULE-BASED ROUTER (Solution #2: Fallback System)
# ============================================================================

class RuleBasedRouter:
    """
    Keyword-based routing fallback for when ML model has insufficient training data.
    Always reliable, requires no training data.
    """

    # Define routing rules based on keyword patterns
    ROUTING_RULES = {
        'NETWORK_INFRASTRUCTURE': {
            'keywords': ['network', 'vpn', 'firewall', 'switch', 'router', 'dns',
                        'connection', 'connectivity', 'ping', 'traceroute', 'latency',
                        'bandwidth', 'port', 'gateway', 'subnet', 'wifi', 'ethernet'],
            'confidence': 0.85
        },
        'DATABASE_ADMIN': {
            'keywords': ['database', 'db', 'sql', 'oracle', 'mysql', 'postgres',
                        'query', 'timeout', 'deadlock', 'table', 'index', 'connection pool',
                        'schema', 'stored procedure', 'transaction'],
            'confidence': 0.85
        },
        'APPLICATION_SUPPORT': {
            'keywords': ['application', 'app', 'software', 'login', 'password', 'access',
                        'error message', 'crash', 'hang', 'slow', 'timeout', 'ui',
                        'button', 'screen', 'page load'],
            'confidence': 0.80
        },
        'SERVER_INFRASTRUCTURE': {
            'keywords': ['server', 'cpu', 'memory', 'disk', 'storage', 'virtual machine',
                        'vm', 'host', 'reboot', 'startup', 'shutdown', 'performance',
                        'resource', 'capacity'],
            'confidence': 0.85
        },
        'SAP_ORDER_FULFILLMENT': {
            'keywords': ['sap', 'order', 'fulfillment', 'procurement', 'inventory',
                        'purchase order', 'po', 'goods receipt', 'invoice', 'vendor',
                        'material', 'stock'],
            'confidence': 0.85
        },
        'SECURITY_OPERATIONS': {
            'keywords': ['security', 'malware', 'virus', 'breach', 'unauthorized',
                        'suspicious', 'phishing', 'encryption', 'certificate', 'ssl',
                        'authentication', 'authorization'],
            'confidence': 0.85
        },
        'END_USER_COMPUTING': {
            'keywords': ['desktop', 'laptop', 'workstation', 'printer', 'monitor',
                        'keyboard', 'mouse', 'windows', 'outlook', 'office', 'teams',
                        'zoom', 'browser'],
            'confidence': 0.80
        }
    }

    def predict_assignment(self, description, top_n=3, min_confidence=0.60):
        """
        Predict assignment group using keyword matching rules.

        Args:
            description (str): Incident description
            top_n (int): Number of predictions to return
            min_confidence (float): Minimum confidence threshold

        Returns:
            list of dict: Predictions with confidence and reasoning
        """
        if not description or not isinstance(description, str):
            return []

        description_lower = description.lower()
        scores = []

        for group, rule in self.ROUTING_RULES.items():
            # Find matching keywords
            matched_keywords = [kw for kw in rule['keywords'] if kw in description_lower]

            if matched_keywords:
                # Calculate confidence based on number of matches
                # More keyword matches = higher confidence
                match_ratio = len(matched_keywords) / len(rule['keywords'])
                base_confidence = rule['confidence']

                # Boost confidence for multiple matches (up to 0.95 max)
                confidence = min(base_confidence * (1 + match_ratio), 0.95)

                scores.append({
                    'assignment_group': group,
                    'confidence': round(confidence, 2),
                    'reasoning': f"Matched keywords: {', '.join(matched_keywords[:5])}",
                    'method': 'rule_based',
                    'num_keywords_matched': len(matched_keywords)
                })

        # Sort by confidence
        scores.sort(key=lambda x: x['confidence'], reverse=True)

        # Filter by minimum confidence
        scores = [s for s in scores if s['confidence'] >= min_confidence]

        # Return top N
        return scores[:top_n]

    def get_available_groups(self):
        """Return list of assignment groups this router can predict."""
        return list(self.ROUTING_RULES.keys())

    def add_custom_rule(self, group_name, keywords, confidence=0.80):
        """
        Add a custom routing rule.

        Args:
            group_name (str): Assignment group name
            keywords (list): List of keywords to match
            confidence (float): Base confidence for this rule
        """
        self.ROUTING_RULES[group_name] = {
            'keywords': keywords,
            'confidence': confidence
        }


# ============================================================================
# CONFIDENCE CALIBRATOR (Solution #6: Uncertainty Quantification)
# ============================================================================

class ConfidenceCalibrator:
    """
    Calibrates ML model confidence scores based on training data size and quality.
    Prevents overconfident predictions from under-trained models.
    """

    @staticmethod
    def calibrate_confidence(raw_confidence, num_training_samples, num_groups, min_group_size):
        """
        Adjust confidence score based on training data quality.

        Args:
            raw_confidence (float): Raw model confidence (0-1)
            num_training_samples (int): Total training samples
            num_groups (int): Number of assignment groups
            min_group_size (int): Smallest group size

        Returns:
            float: Calibrated confidence score
        """
        # Start with raw confidence
        calibrated = raw_confidence

        # Penalty for small training set
        if num_training_samples < 50:
            # Reduce confidence by up to 30% for very small datasets
            sample_penalty = 0.3 * (1 - num_training_samples / 50)
            calibrated *= (1 - sample_penalty)

        elif num_training_samples < 100:
            # Smaller penalty for medium datasets
            sample_penalty = 0.15 * (1 - num_training_samples / 100)
            calibrated *= (1 - sample_penalty)

        # Penalty for imbalanced groups
        if min_group_size < 5:
            # Reduce confidence for imbalanced training data
            balance_penalty = 0.2 * (1 - min_group_size / 5)
            calibrated *= (1 - balance_penalty)

        # Penalty for too many groups with little data
        avg_samples_per_group = num_training_samples / num_groups if num_groups > 0 else 0
        if avg_samples_per_group < 10:
            sparsity_penalty = 0.15 * (1 - avg_samples_per_group / 10)
            calibrated *= (1 - sparsity_penalty)

        # Ensure confidence stays in valid range
        return max(0.0, min(1.0, calibrated))

    @staticmethod
    def get_confidence_label(confidence):
        """
        Convert numeric confidence to human-readable label.

        Args:
            confidence (float): Confidence score (0-1)

        Returns:
            str: Confidence label
        """
        if confidence >= 0.90:
            return "Very High"
        elif confidence >= 0.75:
            return "High"
        elif confidence >= 0.60:
            return "Medium"
        elif confidence >= 0.40:
            return "Low"
        else:
            return "Very Low"

    @staticmethod
    def should_show_prediction(confidence, min_threshold=0.60):
        """
        Determine if prediction should be shown to user.

        Args:
            confidence (float): Calibrated confidence
            min_threshold (float): Minimum acceptable confidence

        Returns:
            bool: Whether to show prediction
        """
        return confidence >= min_threshold

    @staticmethod
    def get_recommendation_text(confidence):
        """
        Get recommendation text for different confidence levels.

        Args:
            confidence (float): Calibrated confidence

        Returns:
            str: Recommendation message
        """
        if confidence >= 0.90:
            return "High confidence prediction - safe to auto-apply"
        elif confidence >= 0.75:
            return "Good confidence - recommend applying with quick review"
        elif confidence >= 0.60:
            return "Moderate confidence - manual review recommended"
        elif confidence >= 0.40:
            return "Low confidence - use as suggestion only"
        else:
            return "Very low confidence - consider alternative routing methods"


# ============================================================================
# 2. INTELLIGENT ASSIGNMENT / ROUTING
# ============================================================================

class IntelligentRouter:
    """
    ML-based incident assignment router with data quality checks and confidence calibration.

    Uses historical resolution data to predict which team should handle new incidents.
    Integrates with RuleBasedRouter for graceful degradation when data is insufficient.
    """

    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.trained = False
        # Training metrics for confidence calibration
        self.num_training_samples = 0
        self.num_groups = 0
        self.min_group_size = 0
        self.training_quality = None

    def train(self, historical_df):
        """
        Train the assignment model on historical incident data with quality checks.

        Args:
            historical_df: DataFrame with 'short_description', 'description',
                          'assignment_group', 'state' (resolved/closed)

        Returns:
            dict: Training metrics (accuracy, coverage, quality assessment)
        """
        if historical_df.empty:
            return {'error': 'No training data', 'success': False}

        # Check data quality FIRST
        quality = DataQualityChecker.check_routing_data_quality(historical_df)
        self.training_quality = quality

        if not quality['sufficient']:
            # Data is insufficient - don't train ML model
            return {
                'success': False,
                'error': 'Insufficient training data',
                'reason': quality['reason'],
                'recommended_action': quality['recommended_action'],
                'metrics': quality['metrics'],
                'use_fallback': True  # Signal to use RuleBasedRouter instead
            }

        # Filter to resolved incidents with assignment groups
        resolved_mask = historical_df['state'].isin(['Closed', 'Resolved'])
        train_df = historical_df[resolved_mask].copy()

        if 'assignment_group' not in train_df.columns:
            return {'error': 'No assignment_group column', 'success': False}

        train_df = train_df[train_df['assignment_group'].notna()]

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
        group_counts = y.value_counts()

        # Store metrics for confidence calibration
        self.num_training_samples = len(train_df)
        self.num_groups = len(group_counts)
        self.min_group_size = int(group_counts.min()) if len(group_counts) > 0 else 0

        # Train RandomForest
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)

        self.trained = True

        # Calculate training accuracy
        train_accuracy = self.model.score(X, y)

        metrics = {
            'success': True,
            'training_accuracy': float(train_accuracy),
            'num_training_samples': self.num_training_samples,
            'num_assignment_groups': self.num_groups,
            'min_group_size': self.min_group_size,
            'assignment_groups': group_counts.to_dict(),
            'data_quality': quality['confidence'],
            'use_fallback': False
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
    def predict_assignment(self, incident_description, top_n=3, min_confidence=0.60):
        """
        Predict the best assignment group for a new incident with calibrated confidence.

        Args:
            incident_description (str): Description of the incident
            top_n (int): Number of top predictions to return
            min_confidence (float): Minimum confidence threshold to show predictions

        Returns:
            list of dict: Predictions with calibrated confidence scores

        Example:
            >>> predictions = router.predict_assignment("VPN connection failed")
            >>> print(predictions[0])
            {
                'assignment_group': 'Network Infrastructure',
                'confidence': 0.94,
                'calibrated_confidence': 0.87,
                'confidence_label': 'High',
                'reasoning': 'Keywords: vpn, connection, network',
                'recommendation': 'Good confidence - recommend applying with quick review'
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
            raw_confidence = float(proba[idx])

            # Calibrate confidence based on training data quality
            calibrated_confidence = ConfidenceCalibrator.calibrate_confidence(
                raw_confidence,
                self.num_training_samples,
                self.num_groups,
                self.min_group_size
            )

            # Skip predictions below minimum confidence threshold
            if calibrated_confidence < min_confidence:
                continue

            # Extract important keywords (simple approach)
            feature_names = self.vectorizer.get_feature_names_out()
            feature_values = X_new.toarray()[0]
            top_features_idx = feature_values.argsort()[-5:][::-1]
            keywords = [feature_names[i] for i in top_features_idx if feature_values[i] > 0]

            results.append({
                'assignment_group': classes[idx],
                'confidence': raw_confidence,
                'calibrated_confidence': round(calibrated_confidence, 2),
                'confidence_label': ConfidenceCalibrator.get_confidence_label(calibrated_confidence),
                'reasoning': f"Keywords: {', '.join(keywords[:3])}" if keywords else 'Pattern match',
                'recommendation': ConfidenceCalibrator.get_recommendation_text(calibrated_confidence),
                'method': 'ml_model'
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
