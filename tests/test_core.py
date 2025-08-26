"""
Basic tests for the data agent core functionality.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, AsyncMock
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_agent.data.loader import DataLoader, SchemaInfo
from data_agent.analysis.statistics import StatisticalAnalyzer
from data_agent.llm.query_processor import QueryProcessor, QueryType, AnalysisMethod


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    data = {
        'id': range(1, 101),
        'name': [f'Item_{i}' for i in range(1, 101)],
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'value': np.random.normal(100, 20, 100),
        'count': np.random.randint(1, 50, 100),
        'date': pd.date_range('2023-01-01', periods=100, freq='D')
    }
    return pd.DataFrame(data)


class TestDataLoader:
    """Test the data loader functionality."""
    
    def test_schema_info_creation(self, sample_dataframe):
        """Test SchemaInfo creation from DataFrame."""
        schema_info = SchemaInfo(sample_dataframe)
        
        assert schema_info.shape == (100, 6)
        assert len(schema_info.columns) == 6
        assert 'id' in schema_info.columns
        assert 'value' in schema_info.numeric_columns
        assert 'category' in schema_info.categorical_columns
        assert 'date' in schema_info.datetime_columns
    
    def test_data_loader_schema_info(self, sample_dataframe):
        """Test data loader schema info generation."""
        loader = DataLoader()
        schema_info = loader.get_schema_info(sample_dataframe)
        
        assert isinstance(schema_info, SchemaInfo)
        assert schema_info.shape[0] == 100
        
    def test_detect_issues(self, sample_dataframe):
        """Test issue detection in data."""
        loader = DataLoader()
        
        # Add some issues to test
        df_with_issues = sample_dataframe.copy()
        df_with_issues.loc[0:10, 'value'] = np.nan  # Add missing values
        df_with_issues.loc[50, 'category'] = 'CONSTANT'  # Add constant column
        
        issues = loader.detect_issues(df_with_issues)
        
        assert isinstance(issues, list)
        # Should detect high missing values if percentage is high enough


class TestStatisticalAnalyzer:
    """Test the statistical analyzer functionality."""
    
    def test_describe_dataset(self, sample_dataframe):
        """Test dataset description functionality."""
        analyzer = StatisticalAnalyzer()
        result = analyzer.describe_dataset(sample_dataframe)
        
        assert 'overview' in result
        assert 'numeric_summary' in result
        assert 'categorical_summary' in result
        assert result['overview']['total_rows'] == 100
        assert result['overview']['total_columns'] == 6
    
    def test_count_analysis(self, sample_dataframe):
        """Test count analysis functionality."""
        analyzer = StatisticalAnalyzer()
        result = analyzer.count_analysis(sample_dataframe, 'category')
        
        assert 'column' in result
        assert 'total_count' in result
        assert 'unique_count' in result
        assert result['column'] == 'category'
        assert result['total_count'] == 100
    
    def test_aggregate_data(self, sample_dataframe):
        """Test data aggregation functionality."""
        analyzer = StatisticalAnalyzer()
        result = analyzer.aggregate_data(
            sample_dataframe, 
            'category', 
            'value',
            ['count', 'mean']
        )
        
        assert 'group_by' in result
        assert 'agg_column' in result
        assert 'results' in result
        assert result['group_by'] == ['category']
        assert result['agg_column'] == 'value'
    
    def test_trend_analysis(self, sample_dataframe):
        """Test trend analysis functionality."""
        analyzer = StatisticalAnalyzer()
        result = analyzer.trend_analysis(
            sample_dataframe,
            'date',
            'value'
        )
        
        assert 'date_column' in result
        assert 'value_column' in result
        assert 'trend_statistics' in result


class TestQueryProcessor:
    """Test the query processor functionality."""
    
    @pytest.fixture
    def mock_llm_manager(self):
        """Create a mock LLM manager."""
        mock_manager = Mock()
        mock_manager.generate_response = AsyncMock()
        mock_manager.generate_response.return_value = {
            "content": '{"query_type": "descriptive", "analysis_method": "describe_dataset", "columns": [], "filters": [], "confidence": 0.8}'
        }
        return mock_manager
    
    def test_query_processor_initialization(self, mock_llm_manager):
        """Test query processor initialization."""
        processor = QueryProcessor(mock_llm_manager)
        
        assert processor.llm_manager is mock_llm_manager
        assert processor.available_columns == []
        assert len(processor.query_patterns) > 0
    
    def test_update_schema_info(self, mock_llm_manager, sample_dataframe):
        """Test schema info update."""
        processor = QueryProcessor(mock_llm_manager)
        processor.update_schema_info(sample_dataframe)
        
        assert len(processor.available_columns) == 6
        assert 'category' in processor.available_columns
        assert 'value' in processor.available_columns
    
    def test_pattern_classification(self, mock_llm_manager):
        """Test pattern-based query classification."""
        processor = QueryProcessor(mock_llm_manager)
        
        # Test descriptive query
        result = processor._classify_query_patterns("describe the dataset")
        assert result['query_type'] == QueryType.DESCRIPTIVE
        assert result['method'] == AnalysisMethod.DESCRIBE_DATASET
        
        # Test correlation query
        result = processor._classify_query_patterns("find correlations between variables")
        assert result['query_type'] == QueryType.EXPLORATORY
        assert result['method'] == AnalysisMethod.CORRELATION_ANALYSIS
        
        # Test outlier query
        result = processor._classify_query_patterns("detect outliers in the data")
        assert result['query_type'] == QueryType.DIAGNOSTIC
        assert result['method'] == AnalysisMethod.OUTLIER_DETECTION


class TestIntegration:
    """Test integration between components."""
    
    def test_end_to_end_basic_flow(self, sample_dataframe):
        """Test basic end-to-end flow without LLM calls."""
        # Test that components can work together
        loader = DataLoader()
        analyzer = StatisticalAnalyzer()
        
        # Get schema info
        schema_info = loader.get_schema_info(sample_dataframe)
        assert schema_info is not None
        
        # Run analysis
        description = analyzer.describe_dataset(sample_dataframe)
        assert 'overview' in description
        assert description['overview']['total_rows'] == 100
        
        # Test count analysis
        count_result = analyzer.count_analysis(sample_dataframe, 'category')
        assert count_result['total_count'] == 100


if __name__ == "__main__":
    # Run tests if this file is executed directly
    pytest.main([__file__, "-v"])
