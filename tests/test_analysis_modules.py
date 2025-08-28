"""
Comprehensive tests for analysis modules including statistical analysis, 
pattern recognition, anomaly detection, and causal analysis.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import warnings
warnings.filterwarnings("ignore")

# Import analysis modules
from data_agent.analysis.statistics import StatisticalAnalyzer, get_default_statistical_analyzer
from data_agent.analysis.patterns import PatternAnalyzer, get_default_pattern_analyzer
from data_agent.analysis.anomalies import AnomalyDetector, get_default_anomaly_detector
from data_agent.analysis.causal import CausalAnalyzer, get_causal_analyzer


@pytest.fixture
def sample_dataset():
    """Create a comprehensive sample dataset for testing."""
    np.random.seed(42)
    n = 200
    
    data = {
        'revenue': np.random.uniform(1000, 5000, n),
        'costs': np.random.uniform(500, 2000, n),
        'profit': np.random.uniform(100, 1000, n),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n),
        'product_type': np.random.choice(['A', 'B', 'C'], n),
        'quarter': np.random.choice(['Q1', 'Q2', 'Q3', 'Q4'], n),
        'date': pd.date_range('2023-01-01', periods=n, freq='D'),
        'temperature': np.random.normal(20, 5, n),
        'sales_volume': np.random.poisson(50, n)
    }
    
    # Add some correlations
    data['profit'] = data['revenue'] - data['costs'] + np.random.normal(0, 50, n)
    
    # Add some outliers
    outlier_indices = np.random.choice(n, size=10, replace=False)
    data['revenue'] = np.array(data['revenue'])
    data['revenue'][outlier_indices] *= 3  # Create outliers
    
    return pd.DataFrame(data)


@pytest.fixture
def time_series_dataset():
    """Create a time series dataset for testing."""
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=365, freq='D')
    
    # Create seasonal pattern
    seasonal = 10 * np.sin(2 * np.pi * np.arange(365) / 365)
    trend = 0.01 * np.arange(365)
    noise = np.random.normal(0, 1, 365)
    
    values = 100 + seasonal + trend + noise
    
    return pd.DataFrame({
        'date': dates,
        'value': values,
        'category': np.random.choice(['A', 'B'], 365)
    })


class TestStatisticalAnalyzer:
    """Test statistical analysis functionality."""

    def test_get_default_analyzer(self):
        """Test getting default analyzer instance."""
        analyzer = get_default_statistical_analyzer()
        assert isinstance(analyzer, StatisticalAnalyzer)

    def test_describe_dataset_comprehensive(self, sample_dataset):
        """Test comprehensive dataset description."""
        analyzer = StatisticalAnalyzer()
        result = analyzer.describe_dataset(sample_dataset)
        
        # Check structure
        assert 'overview' in result
        assert 'numeric_summary' in result
        assert 'categorical_summary' in result
        
        # Check overview
        overview = result['overview']
        assert overview['total_rows'] == 200
        assert overview['total_columns'] == 9
        assert 'completeness_score' in overview
        assert 'memory_usage_mb' in overview
        assert 'duplicate_rows' in overview
        
        # Check numeric summary
        numeric_summary = result['numeric_summary']
        assert len(numeric_summary) > 0
        for col_stats in numeric_summary.values():
            assert 'count' in col_stats
            assert 'mean' in col_stats
            assert 'std' in col_stats
            assert 'min' in col_stats
            assert 'max' in col_stats

    def test_count_analysis_detailed(self, sample_dataset):
        """Test detailed count analysis."""
        analyzer = StatisticalAnalyzer()
        result = analyzer.count_analysis(sample_dataset, 'region')
        
        assert result['column'] == 'region'
        assert result['total_count'] == 200
        assert result['unique_count'] == 4  # North, South, East, West
        assert 'value_counts' in result
        assert 'percentage_distribution' in result
        
        # Check all regions are present
        regions = set(result['value_counts'].keys())
        assert regions == {'North', 'South', 'East', 'West'}

    def test_count_analysis_missing_column(self, sample_dataset):
        """Test count analysis with missing column."""
        analyzer = StatisticalAnalyzer()
        result = analyzer.count_analysis(sample_dataset, 'nonexistent_column')
        
        assert 'error' in result
        assert 'not found' in result['error'].lower()

    def test_aggregate_data_multiple_functions(self, sample_dataset):
        """Test data aggregation with multiple functions."""
        analyzer = StatisticalAnalyzer()
        result = analyzer.aggregate_data(
            sample_dataset, 'region', 'revenue', ['mean', 'sum', 'count', 'std']
        )
        
        assert result['group_by'] == ['region']
        assert result['agg_column'] == 'revenue'
        assert result['functions'] == ['mean', 'sum', 'count', 'std']
        
        # Check results structure
        assert 'results' in result
        for region_result in result['results']:
            assert 'region' in region_result
            assert 'mean' in region_result
            assert 'sum' in region_result
            assert 'count' in region_result
            assert 'std' in region_result

    def test_aggregate_data_invalid_column(self, sample_dataset):
        """Test aggregation with invalid column."""
        analyzer = StatisticalAnalyzer()
        result = analyzer.aggregate_data(
            sample_dataset, 'nonexistent', 'revenue', ['mean']
        )
        
        assert 'error' in result

    def test_trend_analysis_comprehensive(self, time_series_dataset):
        """Test comprehensive trend analysis."""
        analyzer = StatisticalAnalyzer()
        result = analyzer.trend_analysis(time_series_dataset, 'date', 'value')
        
        assert result['date_column'] == 'date'
        assert result['value_column'] == 'value'
        assert 'trend_statistics' in result
        
        trend_stats = result['trend_statistics']
        assert 'slope' in trend_stats
        assert 'intercept' in trend_stats
        assert 'r_squared' in trend_stats
        assert 'p_value' in trend_stats

    def test_filter_data_single_condition(self, sample_dataset):
        """Test data filtering with single condition."""
        analyzer = StatisticalAnalyzer()
        filters = [{'column': 'revenue', 'operator': '>', 'value': 3000}]
        
        result = analyzer.filter_data(sample_dataset, filters)
        
        assert 'filtered_data' in result
        assert 'original_count' in result
        assert 'filtered_count' in result
        assert result['original_count'] == 200
        assert result['filtered_count'] < 200  # Should be fewer after filtering
        assert result['filters_applied'] == filters

    def test_filter_data_multiple_conditions(self, sample_dataset):
        """Test data filtering with multiple conditions."""
        analyzer = StatisticalAnalyzer()
        filters = [
            {'column': 'revenue', 'operator': '>', 'value': 2000},
            {'column': 'region', 'operator': '==', 'value': 'North'}
        ]
        
        result = analyzer.filter_data(sample_dataset, filters)
        
        assert result['filtered_count'] < result['original_count']
        # All filtered rows should be from North region and have revenue > 2000
        filtered_df = pd.DataFrame(result['filtered_data'])
        if len(filtered_df) > 0:
            assert all(filtered_df['region'] == 'North')
            assert all(filtered_df['revenue'] > 2000)

    def test_group_analysis(self, sample_dataset):
        """Test group analysis functionality."""
        analyzer = StatisticalAnalyzer()
        result = analyzer.group_analysis(sample_dataset, 'region')
        
        assert result['grouping_column'] == 'region'
        assert 'group_statistics' in result
        assert len(result['group_statistics']) == 4  # 4 regions
        
        for group_stat in result['group_statistics']:
            assert 'group_value' in group_stat
            assert 'count' in group_stat
            assert 'numeric_summaries' in group_stat


class TestPatternAnalyzer:
    """Test pattern analysis functionality."""

    def test_get_default_analyzer(self):
        """Test getting default pattern analyzer."""
        analyzer = get_default_pattern_analyzer()
        assert isinstance(analyzer, PatternAnalyzer)

    def test_correlation_analysis_comprehensive(self, sample_dataset):
        """Test comprehensive correlation analysis."""
        analyzer = PatternAnalyzer()
        result = analyzer.correlation_analysis(sample_dataset, method='pearson')
        
        assert result['method'] == 'pearson'
        assert 'correlation_matrix' in result
        assert 'strong_correlations' in result
        
        # Check correlation matrix structure
        corr_matrix = result['correlation_matrix']
        assert isinstance(corr_matrix, dict)
        
        # Should find strong correlation between revenue and profit
        strong_corr = result['strong_correlations']
        assert len(strong_corr) > 0
        
        # Each correlation should have required fields
        for corr in strong_corr:
            assert 'variable_1' in corr
            assert 'variable_2' in corr
            assert 'correlation' in corr
            assert 'strength' in corr

    def test_correlation_analysis_spearman(self, sample_dataset):
        """Test Spearman correlation analysis."""
        analyzer = PatternAnalyzer()
        result = analyzer.correlation_analysis(sample_dataset, method='spearman')
        
        assert result['method'] == 'spearman'
        assert 'correlation_matrix' in result

    def test_clustering_analysis_kmeans(self, sample_dataset):
        """Test K-means clustering analysis."""
        analyzer = PatternAnalyzer()
        features = ['revenue', 'costs', 'profit']
        
        result = analyzer.clustering_analysis(
            sample_dataset, features=features, algorithm='kmeans'
        )
        
        assert result['algorithm'] == 'kmeans'
        assert result['features'] == features
        assert result['n_clusters'] == 3  # Default
        assert 'cluster_centers' in result
        assert 'cluster_assignments' in result
        assert 'evaluation_metrics' in result
        
        # Check evaluation metrics
        metrics = result['evaluation_metrics']
        assert 'silhouette_score' in metrics
        assert 'calinski_harabasz_score' in metrics
        assert 'davies_bouldin_score' in metrics

    def test_clustering_analysis_dbscan(self, sample_dataset):
        """Test DBSCAN clustering analysis."""
        analyzer = PatternAnalyzer()
        features = ['revenue', 'costs']
        
        result = analyzer.clustering_analysis(
            sample_dataset, features=features, algorithm='dbscan', eps=500
        )
        
        assert result['algorithm'] == 'dbscan'
        assert result['features'] == features
        assert 'eps' in result
        assert 'cluster_assignments' in result
        assert 'n_clusters' in result

    def test_clustering_invalid_algorithm(self, sample_dataset):
        """Test clustering with invalid algorithm."""
        analyzer = PatternAnalyzer()
        features = ['revenue', 'costs']
        
        result = analyzer.clustering_analysis(
            sample_dataset, features=features, algorithm='invalid_algo'
        )
        
        assert 'error' in result

    def test_time_series_patterns(self, time_series_dataset):
        """Test time series pattern analysis."""
        analyzer = PatternAnalyzer()
        result = analyzer.time_series_patterns(
            time_series_dataset, 'date', ['value']
        )
        
        assert result['date_column'] == 'date'
        assert result['value_columns'] == ['value']
        assert 'seasonal_analysis' in result
        assert 'trend_analysis' in result
        
        seasonal = result['seasonal_analysis']
        assert 'seasonality_detected' in seasonal
        assert 'seasonal_period' in seasonal

    def test_time_series_patterns_invalid_date_column(self, sample_dataset):
        """Test time series patterns with invalid date column."""
        analyzer = PatternAnalyzer()
        result = analyzer.time_series_patterns(
            sample_dataset, 'region', ['revenue']  # region is not a date column
        )
        
        assert 'error' in result


class TestAnomalyDetector:
    """Test anomaly detection functionality."""

    def test_get_default_detector(self):
        """Test getting default anomaly detector."""
        detector = get_default_anomaly_detector()
        assert isinstance(detector, AnomalyDetector)

    def test_detect_outliers_statistical(self, sample_dataset):
        """Test statistical outlier detection."""
        detector = AnomalyDetector()
        result = detector.detect_outliers(
            sample_dataset, columns=['revenue'], contamination=0.1
        )
        
        assert result['method'] == 'statistical'
        assert result['columns'] == ['revenue']
        assert 'outliers' in result
        assert 'outlier_count' in result
        assert 'outlier_percentage' in result
        
        # Should detect some outliers (we added them in the fixture)
        assert result['outlier_count'] > 0
        assert result['outlier_percentage'] > 0

    def test_detect_outliers_all_columns(self, sample_dataset):
        """Test outlier detection on all numeric columns."""
        detector = AnomalyDetector()
        result = detector.detect_outliers(sample_dataset, columns=None)
        
        assert result['method'] == 'statistical'
        assert len(result['columns']) > 1  # Should include all numeric columns
        assert 'outliers' in result

    def test_detect_multivariate_anomalies(self, sample_dataset):
        """Test multivariate anomaly detection."""
        detector = AnomalyDetector()
        features = ['revenue', 'costs', 'profit']
        
        result = detector.detect_multivariate_anomalies(
            sample_dataset, features=features, contamination=0.05
        )
        
        assert result['method'] == 'isolation_forest'
        assert result['features'] == features
        assert result['contamination'] == 0.05
        assert 'anomalies' in result
        assert 'anomaly_scores' in result
        assert 'anomaly_count' in result

    def test_detect_multivariate_anomalies_auto_features(self, sample_dataset):
        """Test multivariate anomaly detection with automatic feature selection."""
        detector = AnomalyDetector()
        
        result = detector.detect_multivariate_anomalies(
            sample_dataset, features=None, contamination=0.1
        )
        
        assert result['method'] == 'isolation_forest'
        assert len(result['features']) > 1  # Should auto-select numeric features
        assert 'anomalies' in result


class TestCausalAnalyzer:
    """Test causal analysis functionality."""

    def test_get_causal_analyzer(self):
        """Test getting causal analyzer instance."""
        analyzer = get_causal_analyzer()
        assert isinstance(analyzer, CausalAnalyzer)

    def test_analyze_pipeline_capacity_drivers(self, sample_dataset):
        """Test pipeline capacity drivers analysis."""
        analyzer = CausalAnalyzer()
        result = analyzer.analyze_pipeline_capacity_drivers(sample_dataset)
        
        assert result['analysis_type'] == 'capacity_drivers'
        assert result['methodology'] == 'Statistical correlation and regression analysis'
        assert 'drivers' in result
        assert 'summary' in result
        assert 'caveats' in result
        
        summary = result['summary']
        assert 'target_variable' in summary
        assert 'predictors_analyzed' in summary
        assert 'data_points' in summary

    def test_detect_infrastructure_bottlenecks(self, sample_dataset):
        """Test infrastructure bottleneck detection."""
        analyzer = CausalAnalyzer()
        result = analyzer.detect_infrastructure_bottlenecks(sample_dataset)
        
        assert result['analysis_type'] == 'bottleneck_detection'
        assert result['methodology'] == 'Statistical bottleneck detection using capacity and utilization metrics'
        assert 'bottlenecks' in result
        assert 'recommendations' in result
        assert 'summary' in result

    def test_analyze_seasonal_patterns_with_date(self, time_series_dataset):
        """Test seasonal pattern analysis with proper date data."""
        analyzer = CausalAnalyzer()
        result = analyzer.analyze_seasonal_patterns(time_series_dataset)
        
        assert result['analysis_type'] == 'seasonal_patterns'
        assert 'patterns' in result
        assert 'confidence' in result

    def test_analyze_seasonal_patterns_no_date(self, sample_dataset):
        """Test seasonal pattern analysis without date columns."""
        analyzer = CausalAnalyzer()
        result = analyzer.analyze_seasonal_patterns(sample_dataset)
        
        # Should handle gracefully even without proper date structure
        assert result['analysis_type'] == 'seasonal_patterns'
        assert 'patterns' in result

    def test_statistical_validation_methods(self, sample_dataset):
        """Test statistical validation methods."""
        analyzer = CausalAnalyzer()
        
        # Test with sample data
        sample1 = np.random.normal(0, 1, 50)
        sample2 = np.random.normal(0.5, 1, 50)
        
        # Test group comparison
        result = analyzer._validate_and_perform_group_test(
            [sample1, sample2], "Test Comparison"
        )
        
        assert 'test' in result
        assert 'statistic' in result
        assert 'p_value' in result
        assert 'significant' in result

    def test_effect_size_calculations(self, sample_dataset):
        """Test effect size calculation methods."""
        analyzer = CausalAnalyzer()
        
        # Test Cohen's d
        group1 = np.random.normal(0, 1, 50)
        group2 = np.random.normal(0.8, 1, 50)  # Medium effect size
        
        cohens_d = analyzer._calculate_cohens_d(group1, group2)
        
        assert isinstance(cohens_d, float)
        assert 0.3 < abs(cohens_d) < 1.2  # Should be medium effect size

    def test_correlation_confidence_interval(self, sample_dataset):
        """Test correlation confidence interval calculation."""
        analyzer = CausalAnalyzer()
        
        # Test with known correlation
        r = 0.5
        n = 100
        
        ci = analyzer._calculate_correlation_confidence_interval(r, n)
        
        assert 'lower' in ci
        assert 'upper' in ci
        assert 'confidence_level' in ci
        assert ci['lower'] < r < ci['upper']

    def test_multiple_testing_correction(self, sample_dataset):
        """Test multiple testing correction application."""
        analyzer = CausalAnalyzer()
        
        # Create mock results with p-values
        mock_results = {
            'statistical_tests': [
                {'p_value': 0.01, 'test': 'test1'},
                {'p_value': 0.03, 'test': 'test2'},
                {'p_value': 0.05, 'test': 'test3'},
                {'p_value': 0.08, 'test': 'test4'}
            ]
        }
        
        corrected = analyzer._apply_overall_multiple_testing_correction(mock_results)
        
        assert 'multiple_testing_correction' in corrected
        correction_info = corrected['multiple_testing_correction']
        assert 'method' in correction_info
        assert 'significant_before' in correction_info
        assert 'significant_after' in correction_info


class TestAnalysisIntegration:
    """Test integration between different analysis modules."""

    def test_all_analyzers_instantiate(self):
        """Test that all analyzers can be instantiated without errors."""
        stat_analyzer = get_default_statistical_analyzer()
        pattern_analyzer = get_default_pattern_analyzer()
        anomaly_detector = get_default_anomaly_detector()
        causal_analyzer = get_causal_analyzer()
        
        assert all(analyzer is not None for analyzer in [
            stat_analyzer, pattern_analyzer, anomaly_detector, causal_analyzer
        ])

    def test_cross_module_workflow(self, sample_dataset):
        """Test a workflow that uses multiple analysis modules."""
        # 1. Basic statistical analysis
        stat_analyzer = get_default_statistical_analyzer()
        stat_result = stat_analyzer.describe_dataset(sample_dataset)
        
        assert stat_result is not None
        assert 'numeric_summary' in stat_result
        
        # 2. Pattern analysis on numeric columns
        pattern_analyzer = get_default_pattern_analyzer()
        corr_result = pattern_analyzer.correlation_analysis(sample_dataset)
        
        assert corr_result is not None
        assert 'correlation_matrix' in corr_result
        
        # 3. Anomaly detection
        anomaly_detector = get_default_anomaly_detector()
        anomaly_result = anomaly_detector.detect_outliers(sample_dataset)
        
        assert anomaly_result is not None
        assert 'outliers' in anomaly_result
        
        # 4. Causal analysis
        causal_analyzer = get_causal_analyzer()
        causal_result = causal_analyzer.analyze_pipeline_capacity_drivers(sample_dataset)
        
        assert causal_result is not None
        assert 'drivers' in causal_result

    def test_error_handling_consistency(self, sample_dataset):
        """Test that all analyzers handle errors consistently."""
        analyzers = [
            get_default_statistical_analyzer(),
            get_default_pattern_analyzer(),
            get_default_anomaly_detector(),
            get_causal_analyzer()
        ]
        
        # Test with empty dataset
        empty_df = pd.DataFrame()
        
        for analyzer in analyzers:
            # Each analyzer should handle empty data gracefully
            if hasattr(analyzer, 'describe_dataset'):
                result = analyzer.describe_dataset(empty_df)
                # Should either succeed or return error dict
                assert isinstance(result, dict)
            
            if hasattr(analyzer, 'correlation_analysis'):
                result = analyzer.correlation_analysis(empty_df)
                assert isinstance(result, dict)
            
            if hasattr(analyzer, 'detect_outliers'):
                result = analyzer.detect_outliers(empty_df)
                assert isinstance(result, dict)


if __name__ == "__main__":
    # Run tests if this file is executed directly
    pytest.main([__file__, "-v"])