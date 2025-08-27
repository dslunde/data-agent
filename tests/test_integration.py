"""
Integration tests for the complete data agent system.
These tests work with the actual dataset and test end-to-end functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
import tempfile
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_agent.core import DataAgentCore
from data_agent.data.loader import get_default_loader


class TestDatasetIntegration:
    """Test integration with actual dataset functionality."""

    @pytest.fixture
    def sample_parquet_file(self):
        """Create a sample parquet file for testing."""
        # Create realistic test data
        np.random.seed(42)
        n_rows = 1000

        data = {
            "id": range(1, n_rows + 1),
            "customer_id": np.random.randint(1, 100, n_rows),
            "product_category": np.random.choice(
                ["Electronics", "Clothing", "Books", "Home"], n_rows
            ),
            "sales_amount": np.random.gamma(2, 50, n_rows),  # Right-skewed distribution
            "profit_margin": np.random.beta(2, 5, n_rows),  # Values between 0 and 1
            "region": np.random.choice(["North", "South", "East", "West"], n_rows),
            "date": pd.date_range("2023-01-01", periods=n_rows, freq="H"),
            "is_return": np.random.choice([True, False], n_rows, p=[0.1, 0.9]),
        }

        df = pd.DataFrame(data)

        # Add some missing values to make it realistic
        df.loc[np.random.choice(n_rows, 50, replace=False), "profit_margin"] = np.nan

        # Create temporary parquet file
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
            df.to_parquet(tmp_file.name)
            yield tmp_file.name

        # Cleanup
        Path(tmp_file.name).unlink()

    def test_dataset_loading_integration(self, sample_parquet_file):
        """Test complete dataset loading pipeline."""
        loader = get_default_loader()

        # Test loading
        df = loader.load_dataset(sample_parquet_file)

        # Verify data was loaded correctly
        assert df is not None
        assert len(df) == 1000
        assert "customer_id" in df.columns
        assert "sales_amount" in df.columns
        assert "date" in df.columns

        # Verify data types were inferred correctly
        assert pd.api.types.is_numeric_dtype(df["sales_amount"])
        assert pd.api.types.is_datetime64_any_dtype(df["date"])
        assert df["product_category"].dtype.name in ["object", "category"]

        # Test schema inference
        schema_info = loader.get_schema_info(df)
        assert schema_info.shape == (1000, 8)
        assert "sales_amount" in schema_info.numeric_columns
        assert "product_category" in schema_info.categorical_columns
        assert "date" in schema_info.datetime_columns

    def test_data_quality_assessment(self, sample_parquet_file):
        """Test data quality assessment on actual dataset."""
        from data_agent.data.quality import get_default_quality_assessor

        loader = get_default_loader()
        assessor = get_default_quality_assessor()

        df = loader.load_dataset(sample_parquet_file)
        quality_report = assessor.assess_quality(df)

        # Verify quality report structure
        assert "overall_score" in quality_report
        assert "issues" in quality_report
        assert "column_quality" in quality_report
        assert "recommendations" in quality_report

        # Should detect missing values in profit_margin
        missing_issues = [
            issue
            for issue in quality_report["issues"]
            if issue["type"] == "high_missing_values"
        ]
        assert len(missing_issues) > 0

        # Overall score should be reasonable (not perfect due to missing values)
        assert 50 <= quality_report["overall_score"] <= 95


class TestAnalysisIntegration:
    """Test integration of analysis components with realistic data."""

    @pytest.fixture
    def analysis_dataset(self):
        """Create a dataset suitable for analysis testing."""
        np.random.seed(42)
        n_rows = 500

        # Create data with known patterns for testing
        x = np.random.normal(0, 1, n_rows)
        y = 2 * x + np.random.normal(0, 0.5, n_rows)  # Strong correlation
        z = np.random.normal(0, 1, n_rows)  # No correlation

        # Add some outliers
        outlier_indices = np.random.choice(n_rows, 25, replace=False)
        y[outlier_indices] = np.random.uniform(10, 15, 25)  # Clear outliers

        df = pd.DataFrame(
            {
                "feature_x": x,
                "feature_y": y,
                "feature_z": z,
                "category": np.random.choice(["A", "B", "C"], n_rows),
                "value": np.random.gamma(2, 3, n_rows),
                "date": pd.date_range("2023-01-01", periods=n_rows, freq="D"),
            }
        )

        return df

    def test_statistical_analysis_integration(self, analysis_dataset):
        """Test statistical analysis with realistic data."""
        from data_agent.analysis.statistics import get_default_statistical_analyzer

        analyzer = get_default_statistical_analyzer()

        # Test descriptive statistics
        desc_result = analyzer.describe_dataset(analysis_dataset)

        assert "overview" in desc_result
        assert desc_result["overview"]["total_rows"] == 500
        assert desc_result["overview"]["total_columns"] == 6

        # Test that numeric summaries are reasonable
        assert "feature_x" in desc_result["numeric_summary"]
        x_stats = desc_result["numeric_summary"]["feature_x"]
        assert abs(x_stats["mean"]) < 0.5  # Should be close to 0
        assert 0.8 < x_stats["std"] < 1.2  # Should be close to 1

        # Test correlation analysis would work
        assert "feature_y" in desc_result["numeric_summary"]
        assert "value" in desc_result["numeric_summary"]

    def test_correlation_analysis_integration(self, analysis_dataset):
        """Test correlation analysis with known relationships."""
        from data_agent.analysis.patterns import get_default_pattern_analyzer

        analyzer = get_default_pattern_analyzer()

        result = analyzer.correlation_analysis(analysis_dataset)

        assert "correlation_matrix" in result
        assert "significant_correlations" in result

        # Should detect strong correlation between feature_x and feature_y
        corr_matrix = result["correlation_matrix"]
        xy_correlation = corr_matrix["feature_x"]["feature_y"]
        assert abs(xy_correlation) > 0.8  # Should be strongly correlated

        # Should find this in significant correlations
        sig_corr = result["significant_correlations"]
        xy_found = any(
            (pair["column1"] == "feature_x" and pair["column2"] == "feature_y")
            or (pair["column1"] == "feature_y" and pair["column2"] == "feature_x")
            for pair in sig_corr
        )
        assert xy_found

    def test_clustering_integration(self, analysis_dataset):
        """Test clustering analysis with realistic data."""
        from data_agent.analysis.patterns import get_default_pattern_analyzer

        analyzer = get_default_pattern_analyzer()

        # Test k-means clustering
        features = ["feature_x", "feature_y", "value"]
        result = analyzer.clustering_analysis(
            analysis_dataset, features, algorithm="kmeans", n_clusters=3
        )

        assert result["algorithm"] == "kmeans"
        assert result["n_clusters"] == 3
        assert "cluster_assignments" in result
        assert "silhouette_score" in result

        # Should have reasonable clustering results
        assignments = result["cluster_assignments"]
        assert len(assignments) == 500
        assert len(set(assignments)) <= 3  # Should have at most 3 clusters

        # Silhouette score should be reasonable for this data
        assert result["silhouette_score"] > -0.5  # Should not be terrible

    def test_outlier_detection_integration(self, analysis_dataset):
        """Test outlier detection with known outliers."""
        from data_agent.analysis.anomalies import get_default_anomaly_detector

        detector = get_default_anomaly_detector()

        # Test IQR method
        result = detector.detect_outliers(
            analysis_dataset, columns=["feature_y"], method="iqr"
        )

        assert "outliers_detected" in result
        assert result["method"] == "iqr"

        # Should detect the outliers we added to feature_y
        outliers = result["outliers_detected"]
        assert len(outliers) > 10  # Should detect most of the 25 outliers we added

        # Outlier values should be high (we added values between 10-15)
        outlier_values = [outlier["feature_y"] for outlier in outliers]
        high_outliers = [val for val in outlier_values if val > 8]
        assert len(high_outliers) > 5  # Should detect several high outliers


class TestEndToEndIntegration:
    """Test complete end-to-end functionality."""

    @pytest.fixture
    def mock_llm_responses(self):
        """Create realistic mock LLM responses."""
        return {
            "describe_dataset": {
                "content": """
                {
                    "query_type": "descriptive",
                    "analysis_method": "describe_dataset",
                    "columns": [],
                    "filters": {},
                    "parameters": {},
                    "confidence": 0.9
                }
                """
            },
            "correlation": {
                "content": """
                {
                    "query_type": "exploratory", 
                    "analysis_method": "correlation_analysis",
                    "columns": ["feature_x", "feature_y"],
                    "filters": {},
                    "parameters": {"method": "pearson"},
                    "confidence": 0.85
                }
                """
            },
            "response_generation": {
                "content": """
                Based on the analysis results, this dataset shows interesting patterns. 
                The correlation analysis reveals a strong positive relationship between feature_x and feature_y,
                with a correlation coefficient of approximately 0.8. This suggests that as feature_x increases,
                feature_y tends to increase as well. The dataset contains 500 observations across 6 variables,
                with good data quality overall.
                """
            },
        }

    @pytest.mark.asyncio
    async def test_core_integration_describe_dataset(self, mock_llm_responses):
        """Test complete workflow for describing a dataset."""
        # Create test data
        test_df = pd.DataFrame(
            {
                "id": range(100),
                "value": np.random.normal(50, 10, 100),
                "category": np.random.choice(["A", "B", "C"], 100),
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
            test_df.to_parquet(tmp_file.name)

            try:
                # Mock the LLM manager
                with patch("data_agent.core.create_llm_manager") as mock_llm_factory:
                    mock_manager = Mock()
                    mock_manager.generate_response = AsyncMock()
                    mock_manager.current_client = Mock()
                    mock_manager.get_current_provider.return_value = "anthropic"
                    mock_manager.get_available_providers.return_value = ["anthropic"]

                    # Set up mock responses
                    mock_manager.generate_response.side_effect = [
                        mock_llm_responses["describe_dataset"],  # Query processing
                        mock_llm_responses[
                            "response_generation"
                        ],  # Response generation
                    ]

                    mock_llm_factory.return_value = mock_manager

                    # Create core and test
                    core = DataAgentCore(preferred_provider="anthropic")

                    # Load dataset
                    dataset_info = await core.load_local_dataset(tmp_file.name)
                    assert dataset_info["shape"] == (100, 3)

                    # Process query
                    result = await core.process_query("describe the dataset")

                    # Verify results
                    assert "query" in result
                    assert "response" in result
                    assert "analysis_method" in result
                    assert result["query"] == "describe the dataset"

                    # Verify LLM was called appropriately
                    assert mock_manager.generate_response.call_count >= 1

            finally:
                # Cleanup
                Path(tmp_file.name).unlink()

    @pytest.mark.asyncio
    async def test_core_integration_with_errors(self, mock_llm_responses):
        """Test error handling in complete workflow."""
        with patch("data_agent.core.create_llm_manager") as mock_llm_factory:
            mock_manager = Mock()
            mock_manager.generate_response = AsyncMock()
            mock_manager.current_client = Mock()
            mock_manager.get_current_provider.return_value = "anthropic"

            # Simulate LLM failure
            mock_manager.generate_response.return_value = {
                "error": "LLM service unavailable"
            }
            mock_llm_factory.return_value = mock_manager

            # Create core
            core = DataAgentCore(preferred_provider="anthropic")

            # Create and load test dataset
            test_df = pd.DataFrame({"id": [1, 2, 3], "value": [1, 2, 3]})
            with tempfile.NamedTemporaryFile(
                suffix=".parquet", delete=False
            ) as tmp_file:
                test_df.to_parquet(tmp_file.name)

                try:
                    await core.load_local_dataset(tmp_file.name)

                    # Process query that will trigger LLM failure
                    result = await core.process_query("analyze this dataset")

                    # Should handle error gracefully
                    assert "query" in result
                    # Should either have error info or fallback response
                    assert ("error" in result) or ("response" in result)

                finally:
                    Path(tmp_file.name).unlink()

    def test_data_agent_initialization(self):
        """Test DataAgentCore initialization and component setup."""
        # Test with mocked LLM manager
        with patch("data_agent.core.create_llm_manager") as mock_llm_factory:
            mock_manager = Mock()
            mock_manager.current_client = Mock()
            mock_manager.get_current_provider.return_value = "anthropic"
            mock_llm_factory.return_value = mock_manager

            core = DataAgentCore(
                preferred_provider="anthropic", cache_enabled=True, verbose=True
            )

            # Verify components were initialized
            assert core.downloader is not None
            assert core.loader is not None
            assert core.quality_assessor is not None
            assert core.statistical_analyzer is not None
            assert core.pattern_analyzer is not None
            assert core.anomaly_detector is not None
            assert core.llm_manager is not None
            assert core.query_processor is not None
            assert core.response_generator is not None

            # Verify cache is enabled
            assert core.cache is not None

    def test_get_llm_status(self):
        """Test LLM status reporting."""
        with patch("data_agent.core.create_llm_manager") as mock_llm_factory:
            mock_manager = Mock()
            mock_manager.get_current_provider.return_value = "anthropic"
            mock_manager.get_available_providers.return_value = ["anthropic"]
            mock_manager.get_usage_stats.return_value = {
                "anthropic": {"request_count": 5, "model": "claude-3-haiku-20240307"}
            }
            mock_llm_factory.return_value = mock_manager

            core = DataAgentCore()
            status = core.get_llm_status()

            assert "current_provider" in status
            assert "available_providers" in status
            assert "usage_stats" in status
            assert status["current_provider"] == "anthropic"
            assert "anthropic" in status["available_providers"]


class TestPerformanceIntegration:
    """Test performance characteristics of the integrated system."""

    def test_large_dataset_handling(self):
        """Test system behavior with larger datasets."""
        # Create a moderately large dataset
        n_rows = 10000
        np.random.seed(42)

        large_df = pd.DataFrame(
            {
                "id": range(n_rows),
                "category": np.random.choice(["A", "B", "C", "D", "E"], n_rows),
                "value1": np.random.normal(100, 20, n_rows),
                "value2": np.random.gamma(2, 5, n_rows),
                "date": pd.date_range("2023-01-01", periods=n_rows, freq="H"),
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
            large_df.to_parquet(tmp_file.name)

            try:
                # Test loading performance
                loader = get_default_loader()
                import time

                start_time = time.time()
                df = loader.load_dataset(tmp_file.name)
                load_time = time.time() - start_time

                # Should load reasonably quickly (< 5 seconds for 10k rows)
                assert load_time < 5.0
                assert len(df) == n_rows

                # Test analysis performance
                from data_agent.analysis.statistics import (
                    get_default_statistical_analyzer,
                )

                analyzer = get_default_statistical_analyzer()

                start_time = time.time()
                desc_result = analyzer.describe_dataset(df)
                analysis_time = time.time() - start_time

                # Analysis should complete quickly (< 3 seconds)
                assert analysis_time < 3.0
                assert "overview" in desc_result

            finally:
                Path(tmp_file.name).unlink()

    def test_memory_usage_reasonable(self):
        """Test that memory usage stays reasonable."""
        import psutil
        import os

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create and process multiple datasets
        for i in range(3):
            test_df = pd.DataFrame(
                {
                    "id": range(1000),
                    "value": np.random.randn(1000),
                    "category": np.random.choice(["A", "B", "C"], 1000),
                }
            )

            with tempfile.NamedTemporaryFile(
                suffix=".parquet", delete=False
            ) as tmp_file:
                test_df.to_parquet(tmp_file.name)

                try:
                    loader = get_default_loader()
                    df = loader.load_dataset(tmp_file.name)

                    # Do some analysis
                    from data_agent.analysis.statistics import (
                        get_default_statistical_analyzer,
                    )

                    analyzer = get_default_statistical_analyzer()
                    analyzer.describe_dataset(df)

                    # Clean up references
                    del df

                finally:
                    Path(tmp_file.name).unlink()

        # Check memory usage hasn't grown too much
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Should not increase by more than 100MB for this test
        assert memory_increase < 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
