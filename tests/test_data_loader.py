"""
Unit tests for data loading and schema inference functionality.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import json
from pathlib import Path

from data_agent.data.loader import DataLoader, SchemaInfo
from data_agent.data.quality import DataQualityAssessor


class TestSchemaInfo:
    """Test SchemaInfo class functionality."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "name": ["A", "B", "C", "D", "E"],
                "value": [10.5, 20.3, 30.1, 40.7, 50.2],
                "count": [1, 2, 3, 4, 5],
                "category": ["X", "Y", "X", "Y", "X"],
                "date": pd.date_range("2023-01-01", periods=5),
                "nullable_int": pd.Series([1, 2, None, 4, 5], dtype="Int64"),
            }
        )

    def test_schema_info_basic_properties(self, sample_df):
        """Test basic SchemaInfo properties."""
        schema = SchemaInfo(sample_df)

        assert schema.shape == (5, 7)
        assert len(schema.columns) == 7
        assert schema.memory_usage_mb > 0

    def test_column_type_detection(self, sample_df):
        """Test proper column type detection."""
        schema = SchemaInfo(sample_df)

        # Test numeric columns detection
        assert "value" in schema.numeric_columns
        assert "count" in schema.numeric_columns
        assert "nullable_int" in schema.numeric_columns

        # Test categorical columns detection
        assert "name" in schema.categorical_columns
        assert "category" in schema.categorical_columns

        # Test datetime columns detection
        assert "date" in schema.datetime_columns

    def test_missing_value_detection(self, sample_df):
        """Test missing value detection."""
        # Add some missing values
        df_with_missing = sample_df.copy()
        df_with_missing.loc[0:1, "value"] = np.nan

        schema = SchemaInfo(df_with_missing)

        assert schema.missing_values["value"] == 2
        assert schema.missing_values["nullable_int"] == 1
        assert schema.missing_values["name"] == 0

    def test_to_dict_conversion(self, sample_df):
        """Test conversion to dictionary."""
        schema = SchemaInfo(sample_df)
        schema_dict = schema.to_dict()

        assert isinstance(schema_dict, dict)
        assert "shape" in schema_dict
        assert "columns" in schema_dict
        assert "dtypes" in schema_dict
        assert "missing_values" in schema_dict


class TestDataLoader:
    """Test DataLoader functionality."""

    def test_data_loader_initialization(self):
        """Test DataLoader initialization."""
        loader = DataLoader()
        assert loader is not None
        assert hasattr(loader, "_optimize_dtypes")
        assert hasattr(loader, "_clean_column_names")

    def test_load_parquet_file(self):
        """Test loading parquet file (requires actual file)."""
        loader = DataLoader()

        # Create a temporary parquet file
        test_df = pd.DataFrame(
            {"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [1.1, 2.2, 3.3]}
        )

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
            test_df.to_parquet(tmp_file.name)

            # Test loading
            loaded_df = loader.load_dataset(tmp_file.name)
            assert loaded_df.shape == (3, 3)
            assert list(loaded_df.columns) == ["a", "b", "c"]

            # Cleanup
            Path(tmp_file.name).unlink()

    def test_load_csv_file(self):
        """Test loading CSV file."""
        loader = DataLoader()

        # Create a temporary CSV file
        test_df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "score": [95.5, 87.2, 92.1],
            }
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmp_file:
            test_df.to_csv(tmp_file.name, index=False)

            # Test loading
            loaded_df = loader.load_dataset(tmp_file.name)
            assert loaded_df.shape == (3, 3)
            assert "id" in loaded_df.columns
            assert "name" in loaded_df.columns

            # Cleanup
            Path(tmp_file.name).unlink()

    def test_load_json_file(self):
        """Test loading JSON file."""
        loader = DataLoader()

        # Create test data
        test_data = [
            {"id": 1, "name": "Alice", "score": 95.5},
            {"id": 2, "name": "Bob", "score": 87.2},
            {"id": 3, "name": "Charlie", "score": 92.1},
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            json.dump(test_data, tmp_file)

            # Test loading
            loaded_df = loader.load_dataset(tmp_file.name)
            assert loaded_df.shape == (3, 3)
            assert "id" in loaded_df.columns

            # Cleanup
            Path(tmp_file.name).unlink()

    def test_sample_size_limiting(self):
        """Test dataset sampling functionality."""
        loader = DataLoader()

        # Create a larger dataset
        large_df = pd.DataFrame({"id": range(1000), "value": np.random.randn(1000)})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
            large_df.to_parquet(tmp_file.name)

            # Test sampling
            sampled_df = loader.load_dataset(tmp_file.name, sample_size=100)
            assert sampled_df.shape[0] == 100
            assert sampled_df.shape[1] == 2

            # Cleanup
            Path(tmp_file.name).unlink()

    def test_datetime_inference(self):
        """Test datetime column inference."""
        loader = DataLoader()

        test_df = pd.DataFrame(
            {
                "date_string": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "timestamp": [
                    "2023-01-01 10:30:00",
                    "2023-01-02 11:45:00",
                    "2023-01-03 12:00:00",
                ],
                "not_date": ["abc", "def", "ghi"],
            }
        )

        result_df = loader._infer_datetime_columns(test_df)

        # Should convert date-like strings to datetime
        assert pd.api.types.is_datetime64_any_dtype(result_df["date_string"])
        assert pd.api.types.is_datetime64_any_dtype(result_df["timestamp"])
        assert not pd.api.types.is_datetime64_any_dtype(result_df["not_date"])

    def test_dtype_optimization(self):
        """Test data type optimization."""
        loader = DataLoader()

        test_df = pd.DataFrame(
            {
                "small_int": [1, 2, 3, 4, 5],
                "large_int": [1000000, 2000000, 3000000, 4000000, 5000000],
                "category_str": ["A", "B", "A", "B", "A"],
                "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
            }
        )

        original_memory = test_df.memory_usage(deep=True).sum()
        optimized_df = loader._optimize_dtypes(test_df)
        optimized_memory = optimized_df.memory_usage(deep=True).sum()

        # Memory should be reduced or equal
        assert optimized_memory <= original_memory

        # Category column should be converted to category type
        assert optimized_df["category_str"].dtype.name == "category"

    def test_get_schema_info(self):
        """Test schema info generation."""
        loader = DataLoader()

        test_df = pd.DataFrame(
            {"id": [1, 2, 3], "name": ["A", "B", "C"], "value": [10.5, 20.3, 30.1]}
        )

        schema_info = loader.get_schema_info(test_df)

        assert isinstance(schema_info, SchemaInfo)
        assert schema_info.shape == (3, 3)
        assert "id" in schema_info.columns
        assert "value" in schema_info.numeric_columns
        assert "name" in schema_info.categorical_columns

    def test_detect_issues(self):
        """Test data quality issue detection."""
        loader = DataLoader()

        # Create data with various issues
        problematic_df = pd.DataFrame(
            {
                "mostly_missing": [1, np.nan, np.nan, np.nan, np.nan],
                "constant_column": [5, 5, 5, 5, 5],
                "high_cardinality": [f"unique_{i}" for i in range(5)],
                "normal_column": [1, 2, 3, 4, 5],
            }
        )

        issues = loader.detect_issues(problematic_df)

        assert isinstance(issues, list)
        # Should detect high missing values and constant columns
        issue_types = [issue["type"] for issue in issues]
        assert "high_missing_values" in issue_types
        assert "constant_column" in issue_types

    def test_unsupported_file_format(self):
        """Test handling of unsupported file formats."""
        loader = DataLoader()

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_file:
            tmp_file.write(b"some text content")

            with pytest.raises(ValueError, match="Unsupported file format"):
                loader.load_dataset(tmp_file.name)

            # Cleanup
            Path(tmp_file.name).unlink()

    def test_nonexistent_file(self):
        """Test handling of nonexistent files."""
        loader = DataLoader()

        with pytest.raises(FileNotFoundError):
            loader.load_dataset("/nonexistent/file.parquet")


class TestDataQualityAssessor:
    """Test DataQualityAssessor functionality."""

    @pytest.fixture
    def quality_assessor(self):
        """Create a DataQualityAssessor instance."""
        return DataQualityAssessor()

    @pytest.fixture
    def test_df(self):
        """Create a test DataFrame with various quality issues."""
        return pd.DataFrame(
            {
                "good_column": [1, 2, 3, 4, 5],
                "missing_values": [1, np.nan, 3, np.nan, 5],
                "constant_column": [10, 10, 10, 10, 10],
                "high_cardinality": [f"value_{i}" for i in range(5)],
                "outlier_column": [1, 2, 3, 4, 1000],  # 1000 is an outlier
                "duplicate_prone": [1, 1, 2, 2, 3],
            }
        )

    def test_assess_quality_basic(self, quality_assessor, test_df):
        """Test basic quality assessment."""
        result = quality_assessor.assess_quality(test_df)

        assert "overview" in result
        assert "issues" in result
        assert "recommendations" in result
        assert "completeness" in result

        # Should have basic quality metrics
        assert isinstance(result["issues"], list)
        assert isinstance(result["recommendations"], list)

    def test_missing_values_detection(self, quality_assessor, test_df):
        """Test missing values detection."""
        result = quality_assessor.assess_quality(test_df)

        # Check that completeness section detects missing values
        assert "completeness" in result
        completeness = result["completeness"]
        assert "columns_with_missing" in completeness
        assert "missing_values" in completeness["columns_with_missing"]

    def test_constant_column_detection(self, quality_assessor, test_df):
        """Test constant column detection."""
        result = quality_assessor.assess_quality(test_df)

        # Should detect constant column
        constant_issues = [
            issue for issue in result["issues"] if issue["type"] == "constant_column"
        ]
        assert len(constant_issues) > 0
        assert "constant_column" in [issue["column"] for issue in constant_issues]

    def test_column_quality_scores(self, quality_assessor, test_df):
        """Test individual column quality assessment."""
        result = quality_assessor.assess_quality(test_df)

        # Should have quality metrics for different aspects
        assert "completeness" in result
        assert "consistency" in result
        assert "validity" in result
        assert "uniqueness" in result

        # Each section should have relevant metrics
        assert "completeness_score" in result["completeness"]
        assert "consistency_score" in result["consistency"]

    def test_recommendations_generation(self, quality_assessor, test_df):
        """Test that recommendations are generated."""
        result = quality_assessor.assess_quality(test_df)

        assert isinstance(result["recommendations"], list)
        # May or may not have recommendations depending on data quality
        # But should be a valid list

        # All recommendations should be strings if present
        for rec in result["recommendations"]:
            assert isinstance(rec, str)
            assert len(rec) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
