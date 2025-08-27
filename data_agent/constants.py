"""
Shared constants and API contracts for data agent components.
"""

# API Response Keys - Standardized across all components
class ResponseKeys:
    """Standard response keys used across all analysis components."""
    
    # Statistical Analysis
    AGGREGATION_FUNCTIONS = "aggregation_functions"
    FUNCTIONS = "functions"
    
    # Clustering Analysis
    CLUSTER_ASSIGNMENTS = "cluster_assignments"
    CLUSTER_LABELS = "cluster_labels"
    
    # Anomaly Detection
    ANOMALIES = "anomalies"
    OUTLIERS = "outliers"
    OUTLIER_INDICES = "outlier_indices"
    
    # Quality Assessment
    OVERALL_SCORE = "overall_score"
    QUALITY_SCORE = "quality_score"
    
    # Response Structure
    RESULTS = "results"
    ERROR = "error"
    CONFIDENCE = "confidence"
    METHOD = "method"
    EVIDENCE = "evidence"
    METHODOLOGY = "methodology"


# Analysis Method Constants
class AnalysisMethods:
    """Standard analysis methods."""
    DESCRIBE_DATASET = "describe_dataset"
    COUNT_ANALYSIS = "count_analysis"
    CORRELATION_ANALYSIS = "correlation_analysis"
    CLUSTERING_ANALYSIS = "clustering_analysis"
    OUTLIER_DETECTION = "outlier_detection"
    TREND_ANALYSIS = "trend_analysis"


# Clustering Algorithm Constants
class ClusteringAlgorithms:
    """Supported clustering algorithms."""
    KMEANS = "kmeans"
    DBSCAN = "dbscan"


# Default Thresholds
class Thresholds:
    """Default analysis thresholds."""
    MIN_CORRELATION = 0.3
    HIGH_CORRELATION = 0.7
    CONFIDENCE_HIGH = 0.8
    CONFIDENCE_MEDIUM = 0.5
    QUALITY_GOOD = 0.8
    QUALITY_FAIR = 0.6


# Error Messages
class ErrorMessages:
    """Standard error messages."""
    MISSING_COLUMNS = "Missing required columns"
    INSUFFICIENT_DATA = "Insufficient data for analysis"
    INVALID_PARAMETERS = "Invalid analysis parameters"
    ANALYSIS_FAILED = "Analysis failed due to data issues"