"""
Core application integration that ties all components together.
"""

import logging
import hashlib
import json
import numpy as np
from typing import Optional, Dict, Any

from .data.downloader import get_default_downloader
from .data.loader import get_default_loader
from .data.quality import get_default_quality_assessor
from .data.profiler import get_default_profiler, get_default_cache
from .analysis.statistics import get_default_statistical_analyzer
from .analysis.patterns import get_default_pattern_analyzer
from .analysis.anomalies import get_default_anomaly_detector
from .analysis.optimizations import get_dataset_optimizer
from .analysis.causal import get_causal_analyzer
from .llm.clients import create_llm_manager
from .llm.query_processor import create_query_processor, AnalysisMethod
from .llm.response_generator import create_response_generator

logger = logging.getLogger(__name__)


class DataAgentCore:
    """Core application that integrates all data agent components."""

    def __init__(
        self,
        preferred_provider: str = "auto",
        model: Optional[str] = None,
        cache_enabled: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize the data agent core.

        Args:
            preferred_provider: LLM provider preference
            model: Specific model to use
            cache_enabled: Whether to enable caching
            verbose: Enable verbose logging
        """
        self.verbose = verbose

        # Dataset state
        self.dataset = None
        self.dataset_info = None

        # Initialize components
        self._initialize_components(preferred_provider, model, cache_enabled)

        logger.info("Data Agent Core initialized successfully")

    def _initialize_components(
        self, preferred_provider: str, model: Optional[str], cache_enabled: bool
    ):
        """Initialize all data agent components."""

        # Data components
        self.downloader = get_default_downloader()
        self.loader = get_default_loader()
        self.quality_assessor = get_default_quality_assessor()
        self.profiler = get_default_profiler()

        # Cache
        if cache_enabled:
            self.cache = get_default_cache()
        else:
            self.cache = None

        # Analysis components
        self.statistical_analyzer = get_default_statistical_analyzer()
        self.pattern_analyzer = get_default_pattern_analyzer()
        self.anomaly_detector = get_default_anomaly_detector()

        # Advanced analytics components
        self.dataset_optimizer = get_dataset_optimizer()
        self.causal_analyzer = get_causal_analyzer()

        # LLM components
        self.llm_manager = create_llm_manager(preferred_provider)
        if model and self.llm_manager.current_client:
            self.llm_manager.current_client.model = model

        self.query_processor = create_query_processor(self.llm_manager)
        self.response_generator = create_response_generator(self.llm_manager)

    async def download_and_load_dataset(
        self, url: Optional[str] = None, sample_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Download and load dataset from URL.

        Args:
            url: URL to download from (uses default if None)
            sample_size: Limit dataset size

        Returns:
            Dataset info dictionary
        """
        logger.info("Downloading and loading dataset")

        try:
            # Download dataset
            dataset_path = self.downloader.download_dataset(source=url)

            # Load dataset
            self.dataset = self.loader.load_dataset(
                dataset_path, sample_size=sample_size, optimize_dtypes=True
            )

            # Apply dataset optimizations
            self.dataset = self.dataset_optimizer.optimize_dataset_loading(self.dataset)

            # Update schema information for query processor
            self.query_processor.update_schema_info(self.dataset)

            # Get dataset info
            self.dataset_info = self._get_dataset_info()

            if self.verbose:
                logger.info(
                    f"Dataset loaded: {self.dataset.shape[0]} rows, {self.dataset.shape[1]} columns"
                )

            return self.dataset_info

        except Exception as e:
            logger.error(f"Error downloading/loading dataset: {e}")
            raise

    async def load_local_dataset(
        self, file_path: str, sample_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Load dataset from local file.

        Args:
            file_path: Path to local dataset file
            sample_size: Limit dataset size

        Returns:
            Dataset info dictionary
        """
        logger.info(f"Loading local dataset from {file_path}")

        try:
            # Load dataset
            self.dataset = self.loader.load_dataset(
                file_path, sample_size=sample_size, optimize_dtypes=True
            )

            # Apply dataset optimizations
            self.dataset = self.dataset_optimizer.optimize_dataset_loading(self.dataset)

            # Update schema information for query processor
            self.query_processor.update_schema_info(self.dataset)

            # Get dataset info
            self.dataset_info = self._get_dataset_info()

            if self.verbose:
                logger.info(
                    f"Dataset loaded: {self.dataset.shape[0]} rows, {self.dataset.shape[1]} columns"
                )

            return self.dataset_info

        except Exception as e:
            logger.error(f"Error loading local dataset: {e}")
            raise

    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process natural language query end-to-end.

        Args:
            query: Natural language query

        Returns:
            Complete response dictionary
        """
        if self.dataset is None:
            return {
                "error": "No dataset loaded. Please load a dataset first.",
                "query": query,
            }

        logger.info(f"Processing query: {query}")

        try:
            # Step 1: Process query to understand intent
            intent = await self.query_processor.process_query(query)

            if self.verbose:
                logger.info(
                    f"Query intent: {intent.analysis_method.value} (confidence: {intent.confidence:.2f})"
                )

            # Step 2: Execute analysis based on intent
            analysis_results = await self._execute_analysis(intent)

            # Step 3: Generate response
            response = await self.response_generator.generate_response(
                intent=intent,
                analysis_results=analysis_results,
                dataset_info=self.dataset_info,
            )

            return response

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "error": str(e),
                "query": query,
                "response": "I encountered an error while processing your query. Please try rephrasing it.",
            }

    async def _execute_analysis(self, intent) -> Dict[str, Any]:
        """Execute the appropriate analysis based on intent."""

        method = intent.analysis_method

        try:
            # Check cache first
            if self.cache:
                cache_key = self._generate_cache_key(intent)
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    if self.verbose:
                        logger.info("Using cached analysis result")
                    return cached_result

            # Execute analysis
            if method == AnalysisMethod.DESCRIBE_DATASET:
                result = self.statistical_analyzer.describe_dataset(self.dataset)

            elif method == AnalysisMethod.COUNT_VALUES:
                column = intent.columns[0] if intent.columns else None
                if column:
                    result = self.statistical_analyzer.count_analysis(
                        self.dataset, column
                    )
                else:
                    result = {"error": "No column specified for count analysis"}

            elif method == AnalysisMethod.AGGREGATE_DATA:
                if len(intent.columns) >= 2:
                    group_by = intent.columns[0]
                    agg_column = intent.columns[1]
                    result = self.statistical_analyzer.aggregate_data(
                        self.dataset, group_by, agg_column
                    )
                else:
                    result = {"error": "Need at least 2 columns for aggregation"}

            elif method == AnalysisMethod.FILTER_DATA:
                if intent.filters:
                    result = self.statistical_analyzer.filter_data(
                        self.dataset, intent.filters
                    )
                else:
                    result = {"error": "No filters specified"}

            elif method == AnalysisMethod.CORRELATION_ANALYSIS:
                result = self.pattern_analyzer.correlation_analysis(
                    self.dataset, method=intent.parameters.get("method", "pearson")
                )

            elif method == AnalysisMethod.CLUSTERING_ANALYSIS:
                result = self.pattern_analyzer.clustering_analysis(
                    self.dataset,
                    features=intent.columns,
                    algorithm=intent.parameters.get("algorithm", "kmeans"),
                    eps=intent.parameters.get("eps", 0.5),
                    min_samples=intent.parameters.get("min_samples", None),
                )

            elif method == AnalysisMethod.PATTERN_RECOGNITION:
                result = self.pattern_analyzer.correlation_analysis(self.dataset)

            elif method == AnalysisMethod.OUTLIER_DETECTION:
                result = self.anomaly_detector.detect_outliers(
                    self.dataset, 
                    columns=intent.columns or None,
                    contamination=intent.parameters.get("contamination", 0.1)
                )

            elif method == AnalysisMethod.ANOMALY_DETECTION:
                result = self.anomaly_detector.detect_multivariate_anomalies(
                    self.dataset, 
                    features=intent.columns or None,
                    contamination=intent.parameters.get("contamination", 0.1)
                )

            elif method == AnalysisMethod.TREND_ANALYSIS:
                if len(intent.columns) >= 2:
                    date_col = intent.columns[0]
                    value_col = intent.columns[1]
                    result = self.statistical_analyzer.trend_analysis(
                        self.dataset, date_col, value_col
                    )
                else:
                    # Try to find date column automatically
                    date_cols = self.dataset.select_dtypes(
                        include=["datetime64"]
                    ).columns
                    numeric_cols = self.dataset.select_dtypes(
                        include=["number"]
                    ).columns

                    if len(date_cols) > 0 and len(numeric_cols) > 0:
                        result = self.statistical_analyzer.trend_analysis(
                            self.dataset, date_cols[0], numeric_cols[0]
                        )
                    else:
                        result = {
                            "error": "Could not find suitable columns for trend analysis"
                        }

            elif method == AnalysisMethod.TIME_SERIES_PATTERNS:
                date_cols = self.dataset.select_dtypes(include=["datetime64"]).columns
                if len(date_cols) > 0:
                    result = self.pattern_analyzer.time_series_patterns(
                        self.dataset, date_cols[0], intent.columns
                    )
                else:
                    result = {
                        "error": "No datetime columns found for time series analysis"
                    }

            elif method == AnalysisMethod.GROUP_COMPARISON:
                if intent.columns:
                    result = self.statistical_analyzer.group_analysis(
                        self.dataset, intent.columns[0]
                    )
                else:
                    result = {"error": "No grouping column specified"}

            elif method == AnalysisMethod.DATA_QUALITY_CHECK:
                result = self.quality_assessor.assess_quality(self.dataset)

            # Causal analysis methods
            elif method == AnalysisMethod.CAUSAL_DRIVERS:
                result = self.causal_analyzer.analyze_pipeline_capacity_drivers(self.dataset)

            elif method == AnalysisMethod.BOTTLENECK_ANALYSIS:
                result = self.causal_analyzer.detect_infrastructure_bottlenecks(self.dataset)

            elif method == AnalysisMethod.SEASONAL_PATTERNS:
                result = self.causal_analyzer.analyze_seasonal_patterns(self.dataset)

            else:
                # Fallback to basic description
                result = self.statistical_analyzer.describe_dataset(self.dataset)

            # Cache result if caching enabled
            if self.cache and "error" not in result:
                cache_key = self._generate_cache_key(intent)
                self.cache.set(cache_key, result)

            return result

        except Exception as e:
            logger.error(f"Error executing analysis {method.value}: {e}")
            return {"error": str(e)}

    def _generate_cache_key(self, intent) -> str:
        """Generate robust cache key based on dataset schema and content hash."""
        # Create a hash of filters to avoid cache invalidation on minor filtering
        filters_hash = self._hash_filters(intent.filters)
        
        # Generate dataset signature based on schema and content
        dataset_signature = self._generate_dataset_signature()
        
        key_data = {
            "method": intent.analysis_method.value,
            "columns": sorted(intent.columns),
            "filters_hash": filters_hash,
            "parameters": intent.parameters,
            "dataset_signature": dataset_signature,
        }

        return self.cache.get_cache_key(key_data) if self.cache else ""

    def _generate_dataset_signature(self) -> str:
        """Generate a robust dataset signature based on schema and content sampling."""
        if self.dataset is None:
            return "no_dataset"
        
        try:
            signature_components = []
            
            # 1. Schema signature (column names and types)
            schema_info = {
                "columns": list(self.dataset.columns),
                "dtypes": [str(dtype) for dtype in self.dataset.dtypes],
                "shape": self.dataset.shape  # Keep shape for now as it's still useful
            }
            schema_str = json.dumps(schema_info, sort_keys=True)
            schema_hash = hashlib.md5(schema_str.encode()).hexdigest()[:8]
            signature_components.append(f"schema:{schema_hash}")
            
            # 2. Content sample signature (hash of first/last few rows + statistical summary)
            # This catches structural changes while being resistant to small data additions
            content_sample = []
            
            # Sample from beginning, middle, and end of dataset
            sample_size = min(10, len(self.dataset))  # Sample up to 10 rows
            if len(self.dataset) > 0:
                # First few rows
                head_sample = self.dataset.head(sample_size // 3 + 1)
                
                # Middle rows
                mid_idx = len(self.dataset) // 2
                mid_start = max(0, mid_idx - sample_size // 3)
                mid_end = min(len(self.dataset), mid_idx + sample_size // 3 + 1)
                mid_sample = self.dataset.iloc[mid_start:mid_end]
                
                # Last few rows  
                tail_sample = self.dataset.tail(sample_size // 3 + 1)
                
                # Combine samples and create hash
                for sample_df in [head_sample, mid_sample, tail_sample]:
                    if len(sample_df) > 0:
                        # Convert to string representation for hashing
                        sample_str = sample_df.to_string()
                        sample_hash = hashlib.md5(sample_str.encode()).hexdigest()[:6]
                        content_sample.append(sample_hash)
            
            if content_sample:
                content_hash = hashlib.md5("_".join(content_sample).encode()).hexdigest()[:8]
                signature_components.append(f"content:{content_hash}")
            
            # 3. Statistical summary signature (for numeric columns)
            numeric_cols = self.dataset.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                # Use basic statistics that are less sensitive to small row additions
                stats_data = {}
                for col in numeric_cols[:5]:  # Limit to first 5 numeric columns for performance
                    col_data = self.dataset[col].dropna()
                    if len(col_data) > 0:
                        # Use percentiles which are more stable than mean/std for caching
                        percentiles = col_data.quantile([0.25, 0.5, 0.75]).round(6)
                        stats_data[col] = {
                            "q25": float(percentiles.iloc[0]),
                            "q50": float(percentiles.iloc[1]), 
                            "q75": float(percentiles.iloc[2]),
                            "nunique": int(col_data.nunique()) if col_data.nunique() < 1000 else 1000
                        }
                
                if stats_data:
                    stats_str = json.dumps(stats_data, sort_keys=True)
                    stats_hash = hashlib.md5(stats_str.encode()).hexdigest()[:8]
                    signature_components.append(f"stats:{stats_hash}")
            
            # 4. Categorical summary signature
            categorical_cols = self.dataset.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                cat_data = {}
                for col in categorical_cols[:3]:  # Limit to first 3 categorical columns
                    unique_vals = self.dataset[col].value_counts().head(10)  # Top 10 categories
                    cat_data[col] = {
                        "top_categories": [str(x) for x in unique_vals.index],
                        "counts": [int(x) for x in unique_vals.values]
                    }
                
                if cat_data:
                    cat_str = json.dumps(cat_data, sort_keys=True)
                    cat_hash = hashlib.md5(cat_str.encode()).hexdigest()[:8]
                    signature_components.append(f"categorical:{cat_hash}")
            
            # Combine all signature components
            final_signature = "_".join(signature_components)
            
            # Create final hash of reasonable length
            return hashlib.md5(final_signature.encode()).hexdigest()[:16]
            
        except Exception as e:
            logger.warning(f"Error generating dataset signature: {e}")
            # Fallback to basic schema-based signature
            fallback_data = {
                "columns": list(self.dataset.columns) if self.dataset is not None else [],
                "shape": self.dataset.shape if self.dataset is not None else (0, 0)
            }
            fallback_str = json.dumps(fallback_data, sort_keys=True)
            return hashlib.md5(fallback_str.encode()).hexdigest()[:16]

    def _hash_filters(self, filters) -> str:
        """Create a hash of filters for consistent caching."""
        if not filters:
            return "no_filters"
        
        try:
            # Sort filters for consistent hashing
            if isinstance(filters, list):
                sorted_filters = sorted(filters, key=lambda x: str(x))
            else:
                sorted_filters = sorted(filters.items()) if isinstance(filters, dict) else filters
            
            filters_str = json.dumps(sorted_filters, sort_keys=True)
            return hashlib.md5(filters_str.encode()).hexdigest()[:8]
        except (TypeError, ValueError):
            # Fallback for unhashable types
            return str(hash(str(filters)))[:8]

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get current dataset information."""
        if self.dataset is None:
            return {"error": "No dataset loaded"}

        return self._get_dataset_info()

    def _get_dataset_info(self) -> Dict[str, Any]:
        """Internal method to get dataset info."""
        if self.dataset is None:
            return {}

        schema_info = self.loader.get_schema_info(self.dataset)

        return {
            "shape": self.dataset.shape,
            "columns": list(self.dataset.columns),
            "column_types": {
                col: str(dtype) for col, dtype in self.dataset.dtypes.items()
            },
            "memory_usage_mb": self.dataset.memory_usage(deep=True).sum() / 1024 / 1024,
            "missing_values": self.dataset.isnull().sum().to_dict(),
            "completeness_score": (
                (self.dataset.size - self.dataset.isnull().sum().sum())
                / self.dataset.size
            )
            * 100,
            "schema_info": schema_info.to_dict() if schema_info else {},
        }

    def get_llm_status(self) -> Dict[str, Any]:
        """Get LLM connection status."""
        return {
            "current_provider": self.llm_manager.get_current_provider(),
            "available_providers": self.llm_manager.get_available_providers(),
            "usage_stats": self.llm_manager.get_usage_stats(),
        }

    def switch_llm_provider(self, provider: str) -> bool:
        """Switch to a different LLM provider."""
        return self.llm_manager.switch_provider(provider)

    def clear_cache(self):
        """Clear analysis cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Analysis cache cleared")


def create_data_agent_core(**kwargs) -> DataAgentCore:
    """Create and return data agent core instance."""
    return DataAgentCore(**kwargs)
