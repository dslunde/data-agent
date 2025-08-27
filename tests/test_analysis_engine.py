"""
Unit tests for the analysis engine components.
"""

import pytest
import pandas as pd
import numpy as np

from data_agent.analysis.statistics import StatisticalAnalyzer
from data_agent.analysis.patterns import PatternAnalyzer
from data_agent.analysis.anomalies import AnomalyDetector


class TestStatisticalAnalyzer:
    """Test StatisticalAnalyzer functionality."""

    @pytest.fixture
    def analyzer(self):
        """Create a StatisticalAnalyzer instance."""
        return StatisticalAnalyzer()

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "id": range(1, 101),
                "category": np.random.choice(["A", "B", "C"], 100),
                "value": np.random.normal(100, 20, 100),
                "count": np.random.randint(1, 50, 100),
                "date": pd.date_range("2023-01-01", periods=100, freq="D"),
                "score": np.random.uniform(0, 100, 100),
            }
        )

    def test_describe_dataset(self, analyzer, sample_df):
        """Test dataset description functionality."""
        result = analyzer.describe_dataset(sample_df)

        # Check required sections
        assert "overview" in result
        assert "numeric_summary" in result
        assert "categorical_summary" in result

        # Check overview content
        assert result["overview"]["total_rows"] == 100
        assert result["overview"]["total_columns"] == 6
        assert result["overview"]["memory_usage_mb"] > 0

        # Check numeric summary
        assert "value" in result["numeric_summary"]
        assert "count" in result["numeric_summary"]
        assert "score" in result["numeric_summary"]

        # Check categorical summary
        assert "category" in result["categorical_summary"]

        # Verify statistics are reasonable
        value_stats = result["numeric_summary"]["value"]
        assert 80 <= value_stats["mean"] <= 120  # Should be around 100
        assert value_stats["std"] > 0

    def test_count_analysis(self, analyzer, sample_df):
        """Test count analysis functionality."""
        result = analyzer.count_analysis(sample_df, "category")

        assert result["column"] == "category"
        assert result["total_count"] == 100
        assert result["unique_count"] == 3
        assert "top_values" in result

        # Check that result has value counts info
        assert "value_counts" in result or "top_values" in result

        # Get the counts (handle both possible keys)
        counts_data = result.get("top_values", result.get("value_counts", {}))
        if isinstance(counts_data, dict):
            assert len(counts_data) <= 3  # Should have A, B, C or fewer
            # Check that counts are reasonable
            total_from_counts = sum(counts_data.values()) if counts_data else 0
            assert total_from_counts <= 100

    def test_aggregate_data(self, analyzer, sample_df):
        """Test data aggregation functionality."""
        result = analyzer.aggregate_data(
            sample_df, "category", "value", ["count", "mean", "std"]
        )

        assert result["group_by"] == ["category"]
        assert result["agg_column"] == "value"
        assert result["aggregation_functions"] == ["count", "mean", "std"]
        assert "results" in result

        # Check results structure (pandas groupby returns dict structure)
        results = result["results"]
        assert isinstance(results, dict)
        
        # Should have the aggregation functions as keys
        for func in ["count", "mean", "std"]:
            assert func in results
            # Each function should have category groups
            func_results = results[func]
            assert isinstance(func_results, dict)
            # Should have results for categories A, B, C
            assert len(func_results) == 3

    def test_filter_data(self, analyzer, sample_df):
        """Test data filtering functionality."""
        filters = {"category": "A", "value": {"min": 90, "max": 110}}

        result = analyzer.filter_data(sample_df, filters)

        assert "original_count" in result
        assert "filtered_count" in result
        assert "filters_applied" in result
        assert "sample_data" in result

        assert result["original_count"] == 100
        assert result["filtered_count"] <= 100

        # Check that filters were applied correctly
        sample_data = result["sample_data"]
        if len(sample_data) > 0:
            # All rows should have category 'A'
            assert all(row["category"] == "A" for row in sample_data)
            # All values should be between 90 and 110
            assert all(90 <= row["value"] <= 110 for row in sample_data)

    def test_trend_analysis(self, analyzer, sample_df):
        """Test trend analysis functionality."""
        result = analyzer.trend_analysis(sample_df, "date", "value")

        assert result["date_column"] == "date"
        assert result["value_column"] == "value"
        assert "trend_statistics" in result
        assert "period_analysis" in result

        # Check trend statistics
        trend_stats = result["trend_statistics"]
        assert "slope" in trend_stats
        assert "correlation" in trend_stats
        assert "trend_strength" in trend_stats

        # Correlation should be between -1 and 1
        assert -1 <= trend_stats["correlation"] <= 1

    def test_group_analysis(self, analyzer, sample_df):
        """Test group analysis functionality."""
        result = analyzer.group_analysis(sample_df, "category")

        assert result["grouping_column"] == "category"
        assert "group_statistics" in result
        assert "comparison_results" in result

        # Should have statistics for each group
        group_stats = result["group_statistics"]
        assert len(group_stats) == 3  # A, B, C groups

        for group_stat in group_stats:
            assert "group_value" in group_stat
            assert "count" in group_stat
            assert "numeric_columns" in group_stat

    def test_invalid_column_handling(self, analyzer, sample_df):
        """Test handling of invalid column names."""
        # Test with non-existent column
        result = analyzer.count_analysis(sample_df, "nonexistent_column")
        assert "error" in result

        # Test trend analysis with invalid columns
        result = analyzer.trend_analysis(sample_df, "nonexistent_date", "value")
        assert "error" in result

        result = analyzer.trend_analysis(sample_df, "date", "nonexistent_value")
        assert "error" in result


class TestPatternAnalyzer:
    """Test PatternAnalyzer functionality."""

    @pytest.fixture
    def analyzer(self):
        """Create a PatternAnalyzer instance."""
        return PatternAnalyzer()

    @pytest.fixture
    def corr_df(self):
        """Create DataFrame with known correlations."""
        np.random.seed(42)
        x = np.random.normal(0, 1, 100)
        y = 0.8 * x + np.random.normal(0, 0.6, 100)  # Strong positive correlation
        z = np.random.normal(0, 1, 100)  # No correlation with x,y

        return pd.DataFrame(
            {"x": x, "y": y, "z": z, "category": np.random.choice(["A", "B"], 100)}
        )

    def test_correlation_analysis(self, analyzer, corr_df):
        """Test correlation analysis functionality."""
        result = analyzer.correlation_analysis(corr_df, method="pearson")

        assert "method" in result
        assert "correlation_matrix" in result
        assert "significant_correlations" in result
        assert result["method"] == "pearson"

        # Check correlation matrix structure
        corr_matrix = result["correlation_matrix"]
        assert isinstance(corr_matrix, dict)
        assert "x" in corr_matrix
        assert "y" in corr_matrix

        # x and y should be strongly correlated
        xy_correlation = corr_matrix["x"]["y"]
        assert abs(xy_correlation) > 0.7

        # Check significant correlations
        sig_corr = result["significant_correlations"]
        assert isinstance(sig_corr, list)
        # Should find x-y correlation
        xy_found = any(
            (pair["column1"] == "x" and pair["column2"] == "y")
            or (pair["column1"] == "y" and pair["column2"] == "x")
            for pair in sig_corr
        )
        assert xy_found

    def test_clustering_analysis_kmeans(self, analyzer, corr_df):
        """Test K-means clustering analysis."""
        # Use numeric columns for clustering
        features = ["x", "y", "z"]
        result = analyzer.clustering_analysis(
            corr_df, features, algorithm="kmeans", n_clusters=2
        )

        assert "algorithm" in result
        assert "n_clusters" in result
        assert "cluster_assignments" in result
        assert "cluster_centers" in result
        assert "metrics" in result

        assert result["algorithm"] == "kmeans"
        assert result["n_clusters"] == 2

        # Check metrics structure (silhouette_score is nested in metrics)
        metrics = result["metrics"]
        assert "silhouette_score" in metrics
        assert isinstance(metrics["silhouette_score"], float)

        # Check cluster assignments
        assignments = result["cluster_assignments"]
        assert len(assignments) == len(corr_df)
        assert all(0 <= cluster <= 1 for cluster in assignments)

        # Silhouette score should be reasonable (from metrics)
        assert -1 <= metrics["silhouette_score"] <= 1

    def test_clustering_analysis_dbscan(self, analyzer, corr_df):
        """Test DBSCAN clustering analysis."""
        features = ["x", "y"]
        result = analyzer.clustering_analysis(corr_df, features, algorithm="dbscan")

        assert result["algorithm"] == "dbscan"
        assert "cluster_assignments" in result
        assert "n_clusters" in result

        # DBSCAN can have noise points (-1)
        assignments = result["cluster_assignments"]
        assert len(assignments) == len(corr_df)
        assert all(cluster >= -1 for cluster in assignments)

    def test_time_series_patterns(self, analyzer):
        """Test time series pattern analysis."""
        # Create time series data
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        values = np.sin(np.arange(100) * 2 * np.pi / 30) + np.random.normal(
            0, 0.1, 100
        )  # Monthly pattern

        ts_df = pd.DataFrame(
            {
                "date": dates,
                "value": values,
                "category": np.random.choice(["A", "B"], 100),
            }
        )

        result = analyzer.time_series_patterns(ts_df, "date", ["value"])

        assert "date_column" in result
        assert "value_columns" in result
        assert "patterns" in result

        # Should detect some patterns
        patterns = result["patterns"]
        assert isinstance(patterns, dict)
        if "value" in patterns:
            assert "trend" in patterns["value"]
            assert "seasonality" in patterns["value"]

    def test_invalid_clustering_parameters(self, analyzer, corr_df):
        """Test handling of invalid clustering parameters."""
        # Test with non-numeric columns
        result = analyzer.clustering_analysis(corr_df, ["category"], algorithm="kmeans")
        assert "error" in result

        # Test with invalid algorithm
        result = analyzer.clustering_analysis(
            corr_df, ["x", "y"], algorithm="invalid_algo"
        )
        assert "error" in result


class TestAnomalyDetector:
    """Test AnomalyDetector functionality."""

    @pytest.fixture
    def detector(self):
        """Create an AnomalyDetector instance."""
        return AnomalyDetector()

    @pytest.fixture
    def outlier_df(self):
        """Create DataFrame with known outliers."""
        np.random.seed(42)
        # Normal data
        normal_data = np.random.normal(50, 10, 95)
        # Add clear outliers
        outliers = np.array([150, 200, -50, -100, 300])
        all_data = np.concatenate([normal_data, outliers])

        return pd.DataFrame(
            {
                "value": all_data,
                "category": ["normal"] * 95 + ["outlier"] * 5,
                "id": range(100),
            }
        )

    def test_detect_outliers_iqr(self, detector, outlier_df):
        """Test IQR-based outlier detection."""
        result = detector.detect_outliers(outlier_df, ["value"], methods=["iqr"])

        assert "methods_used" in result
        assert "detailed_results" in result
        assert "outlier_summary" in result

        assert "iqr" in result.get(
            "methods_used", result.get("methods", result.get("method", []))
        )

        # Should detect the outliers we added
        # Check detailed results structure
        assert "iqr" in result["detailed_results"]
        iqr_results = result["detailed_results"]["iqr"]
        assert "total_outliers" in iqr_results
        assert iqr_results["total_outliers"] > 0

        # Check that outliers were found
        assert "outlier_indices" in iqr_results
        outlier_indices = iqr_results["outlier_indices"]
        assert len(outlier_indices) > 0

    def test_detect_outliers_zscore(self, detector, outlier_df):
        """Test Z-score based outlier detection."""
        result = detector.detect_outliers(outlier_df, ["value"], methods=["zscore"])

        assert "zscore" in result.get(
            "methods_used", result.get("methods", result.get("method", []))
        )
        assert "detailed_results" in result or "outliers_detected" in result

        # Should detect outliers - check either structure
        if "detailed_results" in result:
            assert "zscore" in result["detailed_results"]
            zscore_results = result["detailed_results"]["zscore"]
            assert "total_outliers" in zscore_results
            total_outliers = zscore_results["total_outliers"]
        else:
            outliers = result["outliers_detected"]
            total_outliers = len(outliers)

        assert total_outliers > 0

        # Check outlier summary
        if "outlier_summary" in result:
            summary = result["outlier_summary"]
            assert "total_outliers" in summary
            assert "outlier_percentage" in summary

    def test_detect_multivariate_anomalies(self, detector):
        """Test multivariate anomaly detection."""
        # Create data with multivariate outliers
        np.random.seed(42)

        # Normal correlated data
        normal_size = 95
        x1 = np.random.normal(0, 1, normal_size)
        x2 = x1 + np.random.normal(0, 0.3, normal_size)  # Correlated with x1

        # Add multivariate outliers (far from the correlation pattern)
        outlier_x1 = np.array([5, -5, 6, -6, 7])
        outlier_x2 = np.array([-5, 5, -6, 6, -7])  # Opposite correlation

        all_x1 = np.concatenate([x1, outlier_x1])
        all_x2 = np.concatenate([x2, outlier_x2])

        multi_df = pd.DataFrame({"x1": all_x1, "x2": all_x2, "id": range(100)})

        result = detector.detect_multivariate_anomalies(multi_df, features=["x1", "x2"])

        assert "method" in result or "methods_used" in result
        assert "anomalies_detected" in result
        assert "anomaly_scores" in result

        # Should detect some anomalies
        anomalies = result["anomalies_detected"]
        assert len(anomalies) > 0

        # Check anomaly scores
        scores = result["anomaly_scores"]
        assert len(scores) == 100
        assert all(isinstance(score, (int, float)) for score in scores)

    def test_outlier_detection_multiple_columns(self, detector, outlier_df):
        """Test outlier detection across multiple columns."""
        # Add another column with outliers
        outlier_df["second_value"] = np.random.normal(100, 15, 100)
        outlier_df.loc[95:99, "second_value"] = [
            500,
            -200,
            600,
            -300,
            700,
        ]  # Add outliers

        result = detector.detect_outliers(
            outlier_df, ["value", "second_value"], methods=["iqr"]
        )

        # Should detect outliers in both columns
        if "detailed_results" in result:
            assert "iqr" in result["detailed_results"]
            iqr_results = result["detailed_results"]["iqr"]
            assert "column_results" in iqr_results
            column_results = iqr_results["column_results"]
            assert "value" in column_results
            assert "second_value" in column_results
        else:
            assert len(result.get("outliers_detected", [])) > 0

    def test_no_outliers_case(self, detector):
        """Test behavior when no outliers are present."""
        # Create normal data without outliers
        normal_df = pd.DataFrame(
            {"value": np.random.normal(50, 5, 100)}  # Tight distribution
        )

        result = detector.detect_outliers(normal_df, ["value"], methods=["iqr"])

        # Should have minimal outliers
        if "detailed_results" in result:
            iqr_results = result["detailed_results"]["iqr"]
            total_outliers = iqr_results.get("total_outliers", 0)
        else:
            total_outliers = len(result.get("outliers_detected", []))

        # Might have a few outliers due to random variation, but should be minimal
        assert total_outliers <= 10  # At most 10% outliers

    def test_invalid_method_handling(self, detector, outlier_df):
        """Test handling of invalid detection methods."""
        result = detector.detect_outliers(
            outlier_df, ["value"], methods=["invalid_method"]
        )
        assert "error" in result

    def test_invalid_column_handling(self, detector, outlier_df):
        """Test handling of invalid column names."""
        result = detector.detect_outliers(
            outlier_df, ["nonexistent_column"], methods=["iqr"]
        )
        assert "error" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
