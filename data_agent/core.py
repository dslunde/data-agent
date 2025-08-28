"""
Core application integration that ties all components together.
"""

import logging
import hashlib
import json
import numpy as np
import pandas as pd
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
            try:
                intent = await self.query_processor.process_query(query)
            except ValueError as e:
                logger.error(f"Query parsing error: {e}")
                return {
                    "error": f"Query parsing failed: {str(e)}",
                    "error_type": "QUERY_PARSING_ERROR",
                    "query": query,
                    "response": "I couldn't understand your query. Please try rephrasing it more clearly or check for typos.",
                    "suggestions": [
                        "Make sure your query is in English",
                        "Try asking about specific columns or metrics",
                        "Use simpler language and avoid complex nested questions"
                    ]
                }
            except ConnectionError as e:
                logger.error(f"LLM connection error: {e}")
                return {
                    "error": f"LLM service unavailable: {str(e)}",
                    "error_type": "LLM_CONNECTION_ERROR", 
                    "query": query,
                    "response": "I'm having trouble connecting to the language model service. Please try again later.",
                    "suggestions": [
                        "Check your internet connection",
                        "Verify API keys are properly configured",
                        "Try again in a few moments"
                    ]
                }
            except Exception as e:
                logger.error(f"Unexpected query processing error: {e}")
                return {
                    "error": f"Query processing failed: {str(e)}",
                    "error_type": "QUERY_PROCESSING_ERROR",
                    "query": query,
                    "response": "There was an unexpected error processing your query.",
                }

            if self.verbose:
                logger.info(
                    f"Query intent: {intent.analysis_method.value} (confidence: {intent.confidence:.2f})"
                )

            # Step 2: Execute analysis based on intent
            try:
                analysis_results = await self._execute_analysis(intent)
                
                # Check if analysis returned an error
                if "error" in analysis_results:
                    logger.warning(f"Analysis execution failed: {analysis_results['error']}")
                    return {
                        "error": analysis_results["error"],
                        "error_type": "ANALYSIS_EXECUTION_ERROR",
                        "query": query,
                        "analysis_method": intent.analysis_method.value,
                        "response": f"The {intent.analysis_method.value} analysis encountered an issue. This might be due to data incompatibility or missing required columns.",
                        "suggestions": [
                            "Try a different type of analysis",
                            "Check if your dataset has the required columns",
                            "Consider filtering or cleaning your data first"
                        ]
                    }
                    
            except KeyError as e:
                logger.error(f"Missing data/column error during analysis: {e}")
                return {
                    "error": f"Required data not found: {str(e)}",
                    "error_type": "DATA_MISSING_ERROR",
                    "query": query,
                    "analysis_method": intent.analysis_method.value,
                    "response": f"The analysis requires data or columns that aren't available in your dataset.",
                    "suggestions": [
                        f"Check if column {str(e)} exists in your dataset",
                        "Try a different analysis method",
                        "Load a dataset with the required structure"
                    ]
                }
            except MemoryError as e:
                logger.error(f"Memory error during analysis: {e}")
                return {
                    "error": "Insufficient memory for analysis",
                    "error_type": "MEMORY_ERROR",
                    "query": query,
                    "analysis_method": intent.analysis_method.value,
                    "response": "The dataset is too large for this analysis. Try reducing the dataset size or using sampling.",
                    "suggestions": [
                        "Use the --sample-size parameter to limit dataset size",
                        "Try a simpler analysis method",
                        "Consider aggregating your data before analysis"
                    ]
                }
            except Exception as e:
                logger.error(f"Analysis execution error: {e}")
                return {
                    "error": f"Analysis failed: {str(e)}",
                    "error_type": "ANALYSIS_ERROR",
                    "query": query,
                    "analysis_method": intent.analysis_method.value,
                    "response": f"The {intent.analysis_method.value} analysis encountered an unexpected error.",
                }

            # Step 3: Generate response
            try:
                response = await self.response_generator.generate_response(
                    intent=intent,
                    analysis_results=analysis_results,
                    dataset_info=self.dataset_info,
                )
                return response
                
            except ConnectionError as e:
                logger.error(f"LLM connection error during response generation: {e}")
                return {
                    "error": f"Response generation failed: {str(e)}",
                    "error_type": "RESPONSE_GENERATION_ERROR",
                    "query": query,
                    "analysis_results": analysis_results,  # Include results so user can see raw analysis
                    "response": "Analysis completed successfully, but I couldn't generate a natural language response.",
                    "suggestions": [
                        "The analysis results are available in the 'analysis_results' field",
                        "Try the query again to get a formatted response",
                        "Check your LLM service connection"
                    ]
                }
            except Exception as e:
                logger.error(f"Response generation error: {e}")
                return {
                    "error": f"Response generation failed: {str(e)}",
                    "error_type": "RESPONSE_GENERATION_ERROR",
                    "query": query,
                    "analysis_results": analysis_results,  # Include results so user can see raw analysis
                    "response": "Analysis completed, but response formatting failed.",
                }

        except Exception as e:
            # Final catch-all for truly unexpected errors
            logger.error(f"Unexpected error in process_query: {e}", exc_info=True)
            return {
                "error": f"Unexpected system error: {str(e)}",
                "error_type": "SYSTEM_ERROR", 
                "query": query,
                "response": "An unexpected system error occurred. Please contact support if this persists.",
                "debug_info": str(e) if self.verbose else None
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

            # Apply filters to get the target dataset for analysis
            filtered_df = self._apply_filters(intent)

            # Execute analysis
            if method == AnalysisMethod.DESCRIBE_DATASET:
                result = self.statistical_analyzer.describe_dataset(filtered_df)

            elif method == AnalysisMethod.COUNT_VALUES:
                column = intent.columns[0] if intent.columns else None
                if column:
                    result = self.statistical_analyzer.count_analysis(
                        filtered_df, column
                    )
                else:
                    result = {"error": "No column specified for count analysis"}

            elif method == AnalysisMethod.AGGREGATE_DATA:
                if len(intent.columns) >= 2:
                    group_by = intent.columns[0]
                    agg_column = intent.columns[1]
                    result = self.statistical_analyzer.aggregate_data(
                        filtered_df, group_by, agg_column
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
                    filtered_df, method=intent.parameters.get("method", "pearson")
                )

            elif method == AnalysisMethod.CLUSTERING_ANALYSIS:
                result = self.pattern_analyzer.clustering_analysis(
                    filtered_df,
                    features=intent.columns,
                    algorithm=intent.parameters.get("algorithm", "kmeans"),
                    eps=intent.parameters.get("eps", 0.5),
                    min_samples=intent.parameters.get("min_samples", None),
                )

            elif method == AnalysisMethod.PATTERN_RECOGNITION:
                result = self.pattern_analyzer.correlation_analysis(filtered_df)

            elif method == AnalysisMethod.OUTLIER_DETECTION:
                result = self.anomaly_detector.detect_outliers(
                    filtered_df,
                    columns=intent.columns or None,
                    contamination=intent.parameters.get("contamination", 0.1)
                )

            elif method == AnalysisMethod.ANOMALY_DETECTION:
                result = self.anomaly_detector.detect_multivariate_anomalies(
                    filtered_df,
                    features=intent.columns or None,
                    contamination=intent.parameters.get("contamination", 0.1)
                )

            elif method == AnalysisMethod.TREND_ANALYSIS:
                if len(intent.columns) >= 2:
                    date_col = intent.columns[0]
                    value_col = intent.columns[1]
                    
                    # Validate columns before analysis
                    if date_col not in filtered_df.columns:
                        result = {"error": f"Date column '{date_col}' not found in dataset"}
                    elif value_col not in filtered_df.columns:
                        result = {"error": f"Value column '{value_col}' not found in dataset"}
                    else:
                        result = self.statistical_analyzer.trend_analysis(
                            filtered_df, date_col, value_col
                        )
                else:
                    # Try to find suitable columns automatically
                    result = self._find_and_analyze_trends()

            elif method == AnalysisMethod.TIME_SERIES_PATTERNS:
                date_cols = filtered_df.select_dtypes(include=["datetime64"]).columns
                if len(date_cols) > 0:
                    result = self.pattern_analyzer.time_series_patterns(
                        filtered_df, date_cols[0], intent.columns
                    )
                else:
                    result = {
                        "error": "No datetime columns found for time series analysis"
                    }

            elif method == AnalysisMethod.GROUP_COMPARISON:
                if intent.columns:
                    # Determine target columns for analysis
                    target_columns = None
                    if hasattr(intent, 'target_columns') and intent.target_columns:
                        target_columns = intent.target_columns
                    else:
                        # Default to numeric columns for comparison
                        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
                        # Remove the grouping column from targets
                        target_columns = [col for col in numeric_cols if col != intent.columns[0]]
                    
                    result = self.statistical_analyzer.group_analysis(
                        filtered_df, 
                        group_column=intent.columns[0],
                        target_columns=target_columns
                    )
                else:
                    result = {"error": "No grouping column specified"}

            elif method == AnalysisMethod.DATA_QUALITY_CHECK:
                result = self.quality_assessor.assess_quality(filtered_df)

            # Causal analysis methods
            elif method == AnalysisMethod.CAUSAL_DRIVERS:
                result = self.causal_analyzer.analyze_pipeline_capacity_drivers(filtered_df)

            elif method == AnalysisMethod.BOTTLENECK_ANALYSIS:
                result = self.causal_analyzer.detect_infrastructure_bottlenecks(filtered_df)

            elif method == AnalysisMethod.SEASONAL_PATTERNS:
                result = self.causal_analyzer.analyze_seasonal_patterns(filtered_df)

            else:
                # Fallback to basic description
                result = self.statistical_analyzer.describe_dataset(filtered_df)

            # Cache result if caching enabled
            if self.cache and "error" not in result:
                cache_key = self._generate_cache_key(intent)
                self.cache.set(cache_key, result)

            return result

        except Exception as e:
            logger.error(f"Error executing analysis {method.value}: {e}")
            return {"error": str(e)}

    def _apply_filters(self, intent) -> pd.DataFrame:
        """
        Apply filters from the intent to the dataset, with enhanced logic for pipeline data.
        Includes smart handling for state filtering, company matching, and fallback strategies.
        """
        if not intent.filters:
            if self.verbose:
                logger.info("No filters to apply, returning full dataset")
            return self.dataset

        if self.verbose:
            logger.info(f"Applying {len(intent.filters)} filters: {intent.filters}")

        # Consolidate filters for comparison queries
        equals_filters = {}
        other_filters = []
        
        for f in intent.filters:
            if f.get("operator") == "equals":
                col = f.get("column")
                if col not in equals_filters:
                    equals_filters[col] = []
                equals_filters[col].append(f.get("value"))
            else:
                other_filters.append(f)

        consolidated_filters = other_filters
        for col, values in equals_filters.items():
            if len(values) > 1:
                consolidated_filters.append({"column": col, "operator": "isin", "value": values})
            else:
                consolidated_filters.append({"column": col, "operator": "equals", "value": values[0]})

        # Start with full dataset
        filtered_df = self.dataset.copy()
        original_size = len(filtered_df)
        
        # Apply each filter with enhanced pipeline data handling
        for i, filter_def in enumerate(consolidated_filters):
            column = filter_def.get("column")
            operator = filter_def.get("operator")
            value = filter_def.get("value")
            
            if self.verbose:
                logger.info(f"Applying filter {i+1}/{len(consolidated_filters)}: {column} {operator} {value}")

            # Check if column exists
            if column not in filtered_df.columns:
                # Try to map common pipeline terms to actual columns
                mapped_column = self._map_pipeline_column(column)
                if mapped_column and mapped_column in filtered_df.columns:
                    logger.info(f"Mapped column '{column}' to '{mapped_column}'")
                    column = mapped_column
                else:
                    logger.warning(f"Filter column '{column}' not found in dataset. Available columns: {list(filtered_df.columns)}")
                    continue

            # Apply the filter with pipeline data specific enhancements
            try:
                before_size = len(filtered_df)
                
                if operator == "equals":
                    # Enhanced equals handling for pipeline data
                    filtered_df = self._apply_equals_filter(filtered_df, column, value)
                    
                elif operator == "not_equals":
                    filtered_df = filtered_df[filtered_df[column] != value]
                    
                elif operator == "greater_than":
                    filtered_df = filtered_df[pd.to_numeric(filtered_df[column], errors='coerce') > pd.to_numeric(value, errors='coerce')]
                    
                elif operator == "less_than":
                    filtered_df = filtered_df[pd.to_numeric(filtered_df[column], errors='coerce') < pd.to_numeric(value, errors='coerce')]
                    
                elif operator == "greater_equal":
                    filtered_df = filtered_df[pd.to_numeric(filtered_df[column], errors='coerce') >= pd.to_numeric(value, errors='coerce')]
                    
                elif operator == "less_equal":
                    filtered_df = filtered_df[pd.to_numeric(filtered_df[column], errors='coerce') <= pd.to_numeric(value, errors='coerce')]
                    
                elif operator == "contains" and filtered_df[column].dtype == "object":
                    # Case-insensitive contains for text fields
                    filtered_df = filtered_df[
                        filtered_df[column].str.contains(str(value), case=False, na=False)
                    ]
                    
                elif operator == "isin" and isinstance(value, list):
                    # Enhanced isin handling with case normalization
                    filtered_df = self._apply_isin_filter(filtered_df, column, value)
                    
                else:
                    logger.warning(f"Unknown operator: {operator}")
                    continue
                
                after_size = len(filtered_df)
                
                if self.verbose:
                    logger.info(f"Filter applied: {before_size} -> {after_size} rows ({after_size/before_size*100:.1f}% retained)")
                
                # If filter eliminated all data, implement fallback strategy
                if after_size == 0 and before_size > 0:
                    logger.warning(f"Filter eliminated all data. Attempting fallback strategy for {column} {operator} {value}")
                    filtered_df = self._apply_fallback_filter(self.dataset.copy(), column, operator, value)
                    
                    if len(filtered_df) > 0:
                        logger.info(f"Fallback strategy recovered {len(filtered_df)} rows")
                    else:
                        logger.error(f"Even fallback strategy failed for filter: {filter_def}")
                        # Return partial result rather than empty dataset
                        return self.dataset.copy()
                        
            except Exception as e:
                logger.error(f"Failed to apply filter {filter_def}: {e}")
                # Continue with previous filtered result rather than fail completely
                continue

        final_size = len(filtered_df)
        
        if self.verbose:
            logger.info(f"All filters applied. Final result: {original_size} -> {final_size} rows ({final_size/original_size*100:.1f}% retained)")
        
        # Ensure we don't return an empty dataset unless absolutely necessary
        if final_size == 0 and original_size > 0:
            logger.warning("All filters resulted in empty dataset. Returning full dataset for analysis.")
            return self.dataset
            
        return filtered_df

    def _map_pipeline_column(self, column_name: str) -> str:
        """Map natural language column names to actual pipeline dataset columns."""
        column_mappings = {
            # State mappings
            "state": "state_abb",
            "states": "state_abb", 
            "state_name": "state_abb",
            "location": "state_abb",
            
            # Company mappings
            "company": "pipeline_name",
            "companies": "pipeline_name",
            "pipeline_company": "pipeline_name",
            "operator": "pipeline_name",
            
            # Category mappings
            "category": "category_short",
            "business_category": "category_short",
            "type": "category_short",
            
            # Quantity mappings
            "volume": "scheduled_quantity",
            "quantity": "scheduled_quantity",
            "flow": "scheduled_quantity",
            "throughput": "scheduled_quantity",
            
            # Date mappings
            "date": "eff_gas_day",
            "time": "eff_gas_day",
            "day": "eff_gas_day",
            
            # Location mappings
            "county": "county_name",
            "region": "state_abb",  # Fallback to state for region queries
        }
        
        return column_mappings.get(column_name.lower(), column_name)
    
    def _apply_equals_filter(self, df: pd.DataFrame, column: str, value) -> pd.DataFrame:
        """Apply equals filter with enhanced matching for pipeline data."""
        
        # Special handling for state abbreviations
        if column == "state_abb":
            # Handle both full names and abbreviations
            if isinstance(value, str):
                value_upper = value.upper()
                # Try exact match first
                result = df[df[column] == value_upper]
                if len(result) > 0:
                    return result
                
                # Try to convert state names to abbreviations
                state_name_map = {
                    "TEXAS": "TX", "LOUISIANA": "LA", "OKLAHOMA": "OK",
                    "CALIFORNIA": "CA", "NEW YORK": "NY", "FLORIDA": "FL",
                    # Add more as needed
                }
                
                if value_upper in state_name_map:
                    return df[df[column] == state_name_map[value_upper]]
                    
                # Fallback: partial match
                return df[df[column].str.contains(value_upper, na=False)]
        
        # Special handling for pipeline companies (case-insensitive partial matching)
        elif column == "pipeline_name":
            if isinstance(value, str):
                # Try exact match first
                exact_match = df[df[column].str.lower() == value.lower()]
                if len(exact_match) > 0:
                    return exact_match
                
                # Try contains match
                return df[df[column].str.contains(value, case=False, na=False)]
        
        # Default equals behavior
        return df[df[column] == value]
    
    def _apply_isin_filter(self, df: pd.DataFrame, column: str, values: list) -> pd.DataFrame:
        """Apply isin filter with enhanced matching."""
        
        if column == "state_abb":
            # Normalize state values to uppercase
            normalized_values = [str(v).upper() for v in values]
            return df[df[column].isin(normalized_values)]
        
        # Default isin behavior
        return df[df[column].isin(values)]
    
    def _apply_fallback_filter(self, df: pd.DataFrame, column: str, operator: str, value) -> pd.DataFrame:
        """Apply fallback filtering when primary filter fails."""
        
        logger.info(f"Applying fallback strategy for {column} {operator} {value}")
        
        # For state filtering, if exact match fails, try partial matching
        if column == "state_abb" and operator == "equals":
            if isinstance(value, str):
                # Try case-insensitive partial match
                partial_matches = df[df[column].str.contains(value, case=False, na=False)]
                if len(partial_matches) > 0:
                    return partial_matches
        
        # For company filtering, try broader matching
        elif column == "pipeline_name" and operator == "equals":
            if isinstance(value, str):
                # Try partial company name matching
                partial_matches = df[df[column].str.contains(value, case=False, na=False)]
                if len(partial_matches) > 0:
                    return partial_matches
        
        # For numeric filters, try with null handling
        elif operator in ["greater_than", "less_than", "greater_equal", "less_equal"]:
            try:
                # Remove nulls and try again
                non_null_df = df[df[column].notna()]
                numeric_col = pd.to_numeric(non_null_df[column], errors='coerce')
                non_null_df = non_null_df[numeric_col.notna()]
                
                if operator == "greater_than":
                    return non_null_df[pd.to_numeric(non_null_df[column], errors='coerce') > pd.to_numeric(value, errors='coerce')]
                elif operator == "less_than":
                    return non_null_df[pd.to_numeric(non_null_df[column], errors='coerce') < pd.to_numeric(value, errors='coerce')]
                # Add other operators as needed
                    
            except Exception:
                pass
        
        # If all fallback strategies fail, return empty DataFrame
        return df.iloc[0:0]

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

    def _find_and_analyze_trends(self):
        """Find suitable columns for trend analysis and perform the analysis."""
        # First, try to find actual datetime columns
        date_cols = self.dataset.select_dtypes(include=["datetime64"]).columns
        
        # Also look for columns that might be convertible to datetime
        potential_date_cols = []
        for col in self.dataset.columns:
            if col not in date_cols:
                # Skip categorical columns and those that are clearly not dates
                if pd.api.types.is_categorical_dtype(self.dataset[col]):
                    continue
                    
                # Try a small sample to see if it converts to datetime
                try:
                    sample = self.dataset[col].dropna().head(100)
                    if len(sample) > 0:
                        converted = pd.to_datetime(sample, errors="coerce")
                        # If more than 50% of values convert successfully, consider it a date column
                        if converted.notna().sum() / len(sample) > 0.5:
                            potential_date_cols.append(col)
                except Exception:
                    continue
        
        # Combine actual and potential date columns
        all_date_cols = list(date_cols) + potential_date_cols
        
        # Find numeric columns (excluding categorical)
        numeric_cols = []
        for col in self.dataset.select_dtypes(include=["number"]).columns:
            if not pd.api.types.is_categorical_dtype(self.dataset[col]):
                numeric_cols.append(col)
        
        if len(all_date_cols) > 0 and len(numeric_cols) > 0:
            # Use the first suitable combination
            return self.statistical_analyzer.trend_analysis(
                self.dataset, all_date_cols[0], numeric_cols[0]
            )
        else:
            date_info = f"Found {len(all_date_cols)} potential date columns: {all_date_cols[:3]}" if len(all_date_cols) > 0 else "No date columns found"
            numeric_info = f"Found {len(numeric_cols)} numeric columns: {numeric_cols[:3]}" if len(numeric_cols) > 0 else "No numeric columns found"
            return {
                "error": f"Could not find suitable columns for trend analysis. {date_info}. {numeric_info}."
            }


def create_data_agent_core(**kwargs) -> DataAgentCore:
    """Create and return data agent core instance."""
    return DataAgentCore(**kwargs)
