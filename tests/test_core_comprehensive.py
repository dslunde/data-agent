"""
Comprehensive tests for DataAgentCore class covering all major functionality.
"""

import pytest
import pandas as pd
import numpy as np
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from data_agent.core import DataAgentCore
from data_agent.llm.query_processor import AnalysisMethod
import tempfile
import os


@pytest.fixture
def sample_data():
    """Create sample dataset for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'revenue': np.random.uniform(1000, 5000, 100),
        'costs': np.random.uniform(500, 2000, 100),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
        'date': pd.date_range('2023-01-01', periods=100, freq='D'),
        'profit': np.random.uniform(100, 1000, 100)
    })


@pytest.fixture
def core_instance():
    """Create DataAgentCore instance for testing."""
    return DataAgentCore(cache_enabled=True, verbose=False)


@pytest.fixture
def mock_intent():
    """Create mock intent object."""
    intent = Mock()
    intent.analysis_method = AnalysisMethod.DESCRIBE_DATASET
    intent.columns = ["revenue", "costs"]
    intent.filters = []
    intent.parameters = {}
    intent.confidence = 0.9
    return intent


class TestDataAgentCoreInitialization:
    """Test DataAgentCore initialization and configuration."""

    def test_basic_initialization(self):
        """Test basic core initialization."""
        core = DataAgentCore()
        assert core.dataset is None
        assert core.dataset_info is None
        assert hasattr(core, 'llm_manager')
        assert hasattr(core, 'query_processor')
        assert hasattr(core, 'response_generator')
        assert hasattr(core, 'statistical_analyzer')
        assert hasattr(core, 'pattern_analyzer')
        assert hasattr(core, 'anomaly_detector')
        assert hasattr(core, 'causal_analyzer')

    def test_initialization_with_options(self):
        """Test initialization with various options."""
        core = DataAgentCore(
            preferred_provider="anthropic",
            model="claude-3-haiku",
            cache_enabled=False,
            verbose=True
        )
        assert core.verbose is True
        assert core.cache is None

    def test_components_initialization(self):
        """Test all components are properly initialized."""
        core = DataAgentCore()
        
        # Check all components exist
        components = [
            'downloader', 'loader', 'quality_assessor', 'profiler',
            'statistical_analyzer', 'pattern_analyzer', 'anomaly_detector',
            'dataset_optimizer', 'causal_analyzer', 'llm_manager',
            'query_processor', 'response_generator'
        ]
        
        for component in components:
            assert hasattr(core, component)
            assert getattr(core, component) is not None


class TestDatasetLoading:
    """Test dataset loading functionality."""

    @pytest.mark.asyncio
    async def test_load_local_dataset_csv(self, core_instance, sample_data):
        """Test loading local CSV dataset."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            
            try:
                result = await core_instance.load_local_dataset(f.name)
                
                assert core_instance.dataset is not None
                assert result['shape'] == (100, 5)
                assert set(result['columns']) == {'revenue', 'costs', 'region', 'date', 'profit'}
                assert result['memory_usage_mb'] > 0
                assert result['completeness_score'] == 100.0
                
            finally:
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_load_local_dataset_with_sampling(self, core_instance, sample_data):
        """Test loading dataset with sampling."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            
            try:
                result = await core_instance.load_local_dataset(f.name, sample_size=50)
                
                assert core_instance.dataset is not None
                assert result['shape'][0] <= 50  # Should be sampled
                
            finally:
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_load_nonexistent_file(self, core_instance):
        """Test loading non-existent file raises appropriate error."""
        with pytest.raises(Exception):
            await core_instance.load_local_dataset("/nonexistent/file.csv")

    def test_get_dataset_info_no_dataset(self, core_instance):
        """Test getting dataset info when no dataset is loaded."""
        result = core_instance.get_dataset_info()
        assert result['error'] == 'No dataset loaded'

    def test_get_dataset_info_with_dataset(self, core_instance, sample_data):
        """Test getting dataset info with loaded dataset."""
        core_instance.dataset = sample_data
        result = core_instance._get_dataset_info()
        
        assert result['shape'] == (100, 5)
        assert len(result['columns']) == 5
        assert 'column_types' in result
        assert 'memory_usage_mb' in result
        assert 'missing_values' in result
        assert 'completeness_score' in result


class TestCacheKeyGeneration:
    """Test the improved cache key generation system."""

    def test_generate_dataset_signature_basic(self, core_instance, sample_data):
        """Test basic dataset signature generation."""
        core_instance.dataset = sample_data
        signature = core_instance._generate_dataset_signature()
        
        assert isinstance(signature, str)
        assert len(signature) == 16  # Expected length
        assert signature != "no_dataset"

    def test_generate_dataset_signature_consistency(self, core_instance, sample_data):
        """Test dataset signature consistency."""
        core_instance.dataset = sample_data
        signature1 = core_instance._generate_dataset_signature()
        signature2 = core_instance._generate_dataset_signature()
        
        assert signature1 == signature2

    def test_generate_dataset_signature_different_data(self, core_instance):
        """Test signature changes with different data."""
        # First dataset
        data1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        core_instance.dataset = data1
        signature1 = core_instance._generate_dataset_signature()
        
        # Different dataset
        data2 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        core_instance.dataset = data2
        signature2 = core_instance._generate_dataset_signature()
        
        assert signature1 != signature2

    def test_generate_dataset_signature_no_dataset(self, core_instance):
        """Test signature when no dataset is loaded."""
        signature = core_instance._generate_dataset_signature()
        assert signature == "no_dataset"

    def test_cache_key_generation(self, core_instance, sample_data, mock_intent):
        """Test complete cache key generation."""
        core_instance.dataset = sample_data
        cache_key = core_instance._generate_cache_key(mock_intent)
        
        assert isinstance(cache_key, str)
        assert len(cache_key) > 0

    def test_cache_key_consistency(self, core_instance, sample_data, mock_intent):
        """Test cache key consistency."""
        core_instance.dataset = sample_data
        key1 = core_instance._generate_cache_key(mock_intent)
        key2 = core_instance._generate_cache_key(mock_intent)
        
        assert key1 == key2

    def test_hash_filters_empty(self, core_instance):
        """Test filter hashing with empty filters."""
        result = core_instance._hash_filters([])
        assert result == "no_filters"

    def test_hash_filters_with_data(self, core_instance):
        """Test filter hashing with data."""
        filters = [{"column": "revenue", "operator": ">", "value": 1000}]
        result = core_instance._hash_filters(filters)
        assert isinstance(result, str)
        assert len(result) == 8  # Expected hash length


class TestQueryProcessing:
    """Test query processing with improved error handling."""

    @pytest.mark.asyncio
    async def test_process_query_no_dataset(self, core_instance):
        """Test processing query when no dataset is loaded."""
        result = await core_instance.process_query("test query")
        
        assert 'error' in result
        assert result['query'] == "test query"
        assert result['error'] == "No dataset loaded. Please load a dataset first."

    @pytest.mark.asyncio
    async def test_process_query_with_mock_components(self, core_instance, sample_data):
        """Test query processing with mocked components."""
        core_instance.dataset = sample_data
        core_instance.dataset_info = {'shape': (100, 5)}
        
        # Mock query processor
        mock_intent = Mock()
        mock_intent.analysis_method = AnalysisMethod.DESCRIBE_DATASET
        mock_intent.columns = []
        mock_intent.filters = []
        mock_intent.parameters = {}
        mock_intent.confidence = 0.9
        
        core_instance.query_processor.process_query = AsyncMock(return_value=mock_intent)
        
        # Mock analysis execution
        core_instance._execute_analysis = AsyncMock(return_value={
            "analysis_type": "descriptive_statistics",
            "results": {"mean": 100, "std": 50}
        })
        
        # Mock response generation
        core_instance.response_generator.generate_response = AsyncMock(return_value={
            "query": "test query",
            "response": "Test response",
            "analysis_results": {"mean": 100, "std": 50}
        })
        
        result = await core_instance.process_query("describe the dataset")
        
        assert 'error' not in result
        assert result['response'] == "Test response"

    @pytest.mark.asyncio
    async def test_process_query_value_error(self, core_instance, sample_data):
        """Test query processing with ValueError (query parsing error)."""
        core_instance.dataset = sample_data
        core_instance.dataset_info = {'shape': (100, 5)}
        
        # Mock query processor to raise ValueError
        core_instance.query_processor.process_query = AsyncMock(
            side_effect=ValueError("Invalid query format")
        )
        
        result = await core_instance.process_query("invalid query")
        
        assert result['error_type'] == "QUERY_PARSING_ERROR"
        assert "Invalid query format" in result['error']
        assert result['query'] == "invalid query"
        assert 'suggestions' in result

    @pytest.mark.asyncio
    async def test_process_query_connection_error(self, core_instance, sample_data):
        """Test query processing with ConnectionError (LLM connection issue)."""
        core_instance.dataset = sample_data
        core_instance.dataset_info = {'shape': (100, 5)}
        
        # Mock query processor to raise ConnectionError
        core_instance.query_processor.process_query = AsyncMock(
            side_effect=ConnectionError("LLM service unavailable")
        )
        
        result = await core_instance.process_query("test query")
        
        assert result['error_type'] == "LLM_CONNECTION_ERROR"
        assert "LLM service unavailable" in result['error']
        assert 'suggestions' in result

    @pytest.mark.asyncio
    async def test_process_query_analysis_error(self, core_instance, sample_data):
        """Test query processing with analysis execution error."""
        core_instance.dataset = sample_data
        core_instance.dataset_info = {'shape': (100, 5)}
        
        # Mock successful query processing
        mock_intent = Mock()
        mock_intent.analysis_method = AnalysisMethod.DESCRIBE_DATASET
        mock_intent.columns = []
        mock_intent.filters = []
        mock_intent.parameters = {}
        mock_intent.confidence = 0.9
        
        core_instance.query_processor.process_query = AsyncMock(return_value=mock_intent)
        
        # Mock analysis execution to return error
        core_instance._execute_analysis = AsyncMock(return_value={
            "error": "Missing required column"
        })
        
        result = await core_instance.process_query("test query")
        
        assert result['error_type'] == "ANALYSIS_EXECUTION_ERROR"
        assert "Missing required column" in result['error']
        assert result['analysis_method'] == AnalysisMethod.DESCRIBE_DATASET.value
        assert 'suggestions' in result

    @pytest.mark.asyncio
    async def test_process_query_memory_error(self, core_instance, sample_data):
        """Test query processing with MemoryError."""
        core_instance.dataset = sample_data
        core_instance.dataset_info = {'shape': (100, 5)}
        
        # Mock successful query processing
        mock_intent = Mock()
        mock_intent.analysis_method = AnalysisMethod.DESCRIBE_DATASET
        mock_intent.columns = []
        mock_intent.filters = []
        mock_intent.parameters = {}
        mock_intent.confidence = 0.9
        
        core_instance.query_processor.process_query = AsyncMock(return_value=mock_intent)
        
        # Mock analysis execution to raise MemoryError
        core_instance._execute_analysis = AsyncMock(side_effect=MemoryError("Out of memory"))
        
        result = await core_instance.process_query("test query")
        
        assert result['error_type'] == "MEMORY_ERROR"
        assert "Insufficient memory" in result['error']
        assert 'suggestions' in result

    @pytest.mark.asyncio
    async def test_process_query_key_error(self, core_instance, sample_data):
        """Test query processing with KeyError (missing data)."""
        core_instance.dataset = sample_data
        core_instance.dataset_info = {'shape': (100, 5)}
        
        # Mock successful query processing
        mock_intent = Mock()
        mock_intent.analysis_method = AnalysisMethod.DESCRIBE_DATASET
        mock_intent.columns = []
        mock_intent.filters = []
        mock_intent.parameters = {}
        mock_intent.confidence = 0.9
        
        core_instance.query_processor.process_query = AsyncMock(return_value=mock_intent)
        
        # Mock analysis execution to raise KeyError
        core_instance._execute_analysis = AsyncMock(side_effect=KeyError("missing_column"))
        
        result = await core_instance.process_query("test query")
        
        assert result['error_type'] == "DATA_MISSING_ERROR"
        assert "missing_column" in result['error']
        assert 'suggestions' in result

    @pytest.mark.asyncio
    async def test_process_query_response_generation_error(self, core_instance, sample_data):
        """Test query processing with response generation error."""
        core_instance.dataset = sample_data
        core_instance.dataset_info = {'shape': (100, 5)}
        
        # Mock successful query processing and analysis
        mock_intent = Mock()
        mock_intent.analysis_method = AnalysisMethod.DESCRIBE_DATASET
        mock_intent.columns = []
        mock_intent.filters = []
        mock_intent.parameters = {}
        mock_intent.confidence = 0.9
        
        core_instance.query_processor.process_query = AsyncMock(return_value=mock_intent)
        core_instance._execute_analysis = AsyncMock(return_value={
            "analysis_type": "descriptive_statistics"
        })
        
        # Mock response generation to raise error
        core_instance.response_generator.generate_response = AsyncMock(
            side_effect=Exception("Response generation failed")
        )
        
        result = await core_instance.process_query("test query")
        
        assert result['error_type'] == "RESPONSE_GENERATION_ERROR"
        assert "Response generation failed" in result['error']
        assert 'analysis_results' in result  # Should include raw analysis


class TestAnalysisExecution:
    """Test analysis execution functionality."""

    @pytest.mark.asyncio
    async def test_execute_analysis_describe_dataset(self, core_instance, sample_data, mock_intent):
        """Test executing describe dataset analysis."""
        core_instance.dataset = sample_data
        mock_intent.analysis_method = AnalysisMethod.DESCRIBE_DATASET
        
        # Mock statistical analyzer
        mock_result = {"mean": 100, "std": 50}
        core_instance.statistical_analyzer.describe_dataset = Mock(return_value=mock_result)
        
        result = await core_instance._execute_analysis(mock_intent)
        
        assert result == mock_result
        core_instance.statistical_analyzer.describe_dataset.assert_called_once_with(sample_data)

    @pytest.mark.asyncio
    async def test_execute_analysis_with_caching(self, core_instance, sample_data, mock_intent):
        """Test analysis execution with caching."""
        core_instance.dataset = sample_data
        mock_intent.analysis_method = AnalysisMethod.DESCRIBE_DATASET
        
        # Mock cache
        mock_cache_result = {"cached": True, "mean": 100}
        core_instance.cache = Mock()
        core_instance.cache.get = Mock(return_value=mock_cache_result)
        
        result = await core_instance._execute_analysis(mock_intent)
        
        assert result == mock_cache_result
        assert core_instance.cache.get.called

    @pytest.mark.asyncio
    async def test_execute_analysis_cache_miss_then_store(self, core_instance, sample_data, mock_intent):
        """Test analysis execution with cache miss, then store result."""
        core_instance.dataset = sample_data
        mock_intent.analysis_method = AnalysisMethod.DESCRIBE_DATASET
        
        # Mock cache miss, then successful analysis
        mock_analysis_result = {"mean": 100, "std": 50}
        core_instance.cache = Mock()
        core_instance.cache.get = Mock(return_value=None)  # Cache miss
        core_instance.cache.set = Mock()
        core_instance.statistical_analyzer.describe_dataset = Mock(return_value=mock_analysis_result)
        
        result = await core_instance._execute_analysis(mock_intent)
        
        assert result == mock_analysis_result
        assert core_instance.cache.get.called
        assert core_instance.cache.set.called


class TestLLMIntegration:
    """Test LLM integration functionality."""

    def test_get_llm_status(self, core_instance):
        """Test LLM status retrieval."""
        # Mock LLM manager
        mock_status = {
            "current_provider": "anthropic",
            "available_providers": ["anthropic"],
            "usage_stats": {"tokens_used": 1000}
        }
        core_instance.llm_manager.get_current_provider = Mock(return_value="anthropic")
        core_instance.llm_manager.get_available_providers = Mock(return_value=["anthropic"])
        core_instance.llm_manager.get_usage_stats = Mock(return_value={"tokens_used": 1000})
        
        status = core_instance.get_llm_status()
        
        assert status['current_provider'] == "anthropic"
        assert status['available_providers'] == ["anthropic"]
        assert status['usage_stats']['tokens_used'] == 1000

    def test_switch_llm_provider(self, core_instance):
        """Test switching LLM provider."""
        # Mock LLM manager
        core_instance.llm_manager.switch_provider = Mock(return_value=True)
        
        result = core_instance.switch_llm_provider("openai")
        
        assert result is True
        core_instance.llm_manager.switch_provider.assert_called_once_with("openai")


class TestCacheManagement:
    """Test cache management functionality."""

    def test_clear_cache_with_cache_enabled(self, core_instance):
        """Test clearing cache when cache is enabled."""
        # Mock cache
        core_instance.cache = Mock()
        core_instance.cache.clear = Mock()
        
        core_instance.clear_cache()
        
        core_instance.cache.clear.assert_called_once()

    def test_clear_cache_with_cache_disabled(self):
        """Test clearing cache when cache is disabled."""
        core_instance = DataAgentCore(cache_enabled=False)
        # Should not raise error
        core_instance.clear_cache()


class TestFactoryFunction:
    """Test the factory function."""

    def test_create_data_agent_core(self):
        """Test factory function creates proper instance."""
        from data_agent.core import create_data_agent_core
        
        core = create_data_agent_core(verbose=True, cache_enabled=False)
        
        assert isinstance(core, DataAgentCore)
        assert core.verbose is True
        assert core.cache is None