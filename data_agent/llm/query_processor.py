"""
Query understanding and intent recognition system.
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import pandas as pd

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries the system can handle."""

    DESCRIPTIVE = "descriptive"  # Basic stats, counts, descriptions
    EXPLORATORY = "exploratory"  # Correlations, patterns, distributions
    DIAGNOSTIC = "diagnostic"  # Anomalies, outliers, data quality
    PREDICTIVE = "predictive"  # Trends, forecasting (basic)
    COMPARATIVE = "comparative"  # Group comparisons, A/B testing
    TEMPORAL = "temporal"  # Time series analysis
    CAUSAL = "causal"  # Causal analysis and hypothesis


class AnalysisMethod(Enum):
    """Specific analysis methods."""

    # Descriptive methods
    DESCRIBE_DATASET = "describe_dataset"
    COUNT_VALUES = "count_values"
    AGGREGATE_DATA = "aggregate_data"
    FILTER_DATA = "filter_data"
    SUMMARY_STATS = "summary_stats"

    # Exploratory methods
    CORRELATION_ANALYSIS = "correlation_analysis"
    CLUSTERING_ANALYSIS = "clustering_analysis"
    DISTRIBUTION_ANALYSIS = "distribution_analysis"
    PATTERN_RECOGNITION = "pattern_recognition"

    # Diagnostic methods
    OUTLIER_DETECTION = "outlier_detection"
    ANOMALY_DETECTION = "anomaly_detection"
    DATA_QUALITY_CHECK = "data_quality_check"

    # Temporal methods
    TREND_ANALYSIS = "trend_analysis"
    TIME_SERIES_PATTERNS = "time_series_patterns"
    SEASONALITY_DETECTION = "seasonality_detection"

    # Comparative methods
    GROUP_COMPARISON = "group_comparison"
    SEGMENT_ANALYSIS = "segment_analysis"


@dataclass
class QueryIntent:
    """Represents the understood intent of a user query."""

    query_type: QueryType
    analysis_method: AnalysisMethod
    columns: List[str]
    filters: List[Dict[str, Any]]
    parameters: Dict[str, Any]
    confidence: float
    raw_query: str


class QueryProcessor:
    """Processes natural language queries and extracts analysis intent."""

    def __init__(self, llm_manager):
        """Initialize query processor with LLM manager."""
        self.llm_manager = llm_manager

        # Query patterns for initial classification
        self.query_patterns = self._initialize_patterns()

        # Column name cache (updated when dataset is loaded)
        self.available_columns = []
        self.column_types = {}

    def update_schema_info(self, df: pd.DataFrame):
        """Update available columns and types from loaded dataset."""
        self.available_columns = list(df.columns)
        self.column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
        logger.info(f"Updated schema info with {len(self.available_columns)} columns")

    async def process_query(self, query: str) -> QueryIntent:
        """
        Process natural language query and extract intent.

        Args:
            query: Natural language query from user

        Returns:
            QueryIntent object with extracted information
        """
        logger.info(f"Processing query: {query}")

        # Step 1: Initial pattern-based classification
        initial_intent = self._classify_query_patterns(query)

        # Step 2: Use LLM for detailed intent extraction
        llm_intent = await self._extract_intent_with_llm(query, initial_intent)

        # Step 3: Validate and refine the extracted intent
        final_intent = self._validate_and_refine_intent(llm_intent, query)

        logger.info(
            f"Extracted intent: {final_intent.analysis_method.value} with confidence {final_intent.confidence}"
        )

        return final_intent

    def _initialize_patterns(
        self,
    ) -> Dict[str, List[Tuple[str, QueryType, AnalysisMethod]]]:
        """Initialize regex patterns for query classification."""
        patterns = {
            "descriptive": [
                (
                    r"\b(describe|summary|overview|basic stats?)\b",
                    QueryType.DESCRIPTIVE,
                    AnalysisMethod.DESCRIBE_DATASET,
                ),
                (
                    r"\b(count|how many|number of)\b",
                    QueryType.DESCRIPTIVE,
                    AnalysisMethod.COUNT_VALUES,
                ),
                (
                    r"\b(average|mean|median|sum|total|aggregate)\b",
                    QueryType.DESCRIPTIVE,
                    AnalysisMethod.AGGREGATE_DATA,
                ),
                (
                    r"\b(filter|where|select|show me)\b.*\b(with|having|equals?|greater|less)\b",
                    QueryType.DESCRIPTIVE,
                    AnalysisMethod.FILTER_DATA,
                ),
            ],
            "exploratory": [
                (
                    r"\b(correlat\w*|relationship|associated?)\b",
                    QueryType.EXPLORATORY,
                    AnalysisMethod.CORRELATION_ANALYSIS,
                ),
                (
                    r"\b(cluster\w*|group\w*|segment\w*)\b",
                    QueryType.EXPLORATORY,
                    AnalysisMethod.CLUSTERING_ANALYSIS,
                ),
                (
                    r"\b(pattern\w*|trend\w*|distribution)\b",
                    QueryType.EXPLORATORY,
                    AnalysisMethod.PATTERN_RECOGNITION,
                ),
            ],
            "diagnostic": [
                (
                    r"\b(outlier\w*|anomal\w*|unusual|strange)\b",
                    QueryType.DIAGNOSTIC,
                    AnalysisMethod.OUTLIER_DETECTION,
                ),
                (
                    r"\b(quality|issue\w*|problem\w*|missing|incomplete)\b",
                    QueryType.DIAGNOSTIC,
                    AnalysisMethod.DATA_QUALITY_CHECK,
                ),
            ],
            "temporal": [
                (
                    r"\b(over time|time series|temporal|trend\w*)\b",
                    QueryType.TEMPORAL,
                    AnalysisMethod.TREND_ANALYSIS,
                ),
                (
                    r"\b(seasonal\w*|cyclic\w*|periodic)\b",
                    QueryType.TEMPORAL,
                    AnalysisMethod.SEASONALITY_DETECTION,
                ),
            ],
            "comparative": [
                (
                    r"\b(compare|comparison|difference|vs|versus)\b",
                    QueryType.COMPARATIVE,
                    AnalysisMethod.GROUP_COMPARISON,
                ),
                (
                    r"\b(segment\w*|group by|breakdown)\b",
                    QueryType.COMPARATIVE,
                    AnalysisMethod.SEGMENT_ANALYSIS,
                ),
            ],
            "causal": [
                (
                    r"\b(cause\w*|reason\w*|why|explain\w*|impact|effect)\b",
                    QueryType.CAUSAL,
                    AnalysisMethod.PATTERN_RECOGNITION,
                ),
            ],
        }

        # Compile regex patterns
        compiled_patterns = {}
        for category, pattern_list in patterns.items():
            compiled_patterns[category] = [
                (re.compile(pattern, re.IGNORECASE), query_type, method)
                for pattern, query_type, method in pattern_list
            ]

        return compiled_patterns

    def _classify_query_patterns(self, query: str) -> Dict[str, Any]:
        """Use regex patterns for initial query classification."""
        matches = []

        for category, pattern_list in self.query_patterns.items():
            for regex, query_type, method in pattern_list:
                if regex.search(query):
                    matches.append(
                        {
                            "category": category,
                            "query_type": query_type,
                            "method": method,
                            "confidence": 0.7,  # Base confidence for pattern matching
                        }
                    )

        # Return the first match or default
        if matches:
            return matches[0]
        else:
            return {
                "category": "descriptive",
                "query_type": QueryType.DESCRIPTIVE,
                "method": AnalysisMethod.DESCRIBE_DATASET,
                "confidence": 0.3,
            }

    async def _extract_intent_with_llm(
        self, query: str, initial_intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use LLM to extract detailed intent from the query."""

        system_prompt = self._create_intent_extraction_prompt()

        user_prompt = f"""
Query: "{query}"

Available columns: {', '.join(self.available_columns) if self.available_columns else 'Unknown'}

Initial classification: {initial_intent['method'].value}

Please analyze this query and extract the analysis intent in the specified JSON format.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = await self.llm_manager.generate_response(
                messages=messages, temperature=0.1, max_tokens=800
            )

            # Parse LLM response
            if "error" in response:
                logger.warning(f"LLM error: {response['error']}")
                return self._fallback_intent_extraction(query, initial_intent)

            # Extract JSON from response
            content = response.get("content", "")
            intent_data = self._parse_intent_response(content)

            if intent_data:
                return intent_data
            else:
                logger.warning("Could not parse LLM intent response")
                return self._fallback_intent_extraction(query, initial_intent)

        except Exception as e:
            logger.error(f"Error in LLM intent extraction: {e}")
            return self._fallback_intent_extraction(query, initial_intent)

    def _create_intent_extraction_prompt(self) -> str:
        """Create system prompt for intent extraction."""
        return """You are a data analysis intent extraction system. Your job is to understand user queries about datasets and extract the analysis intent.

Given a user query, you must determine:
1. The type of analysis needed
2. Which columns to analyze
3. Any filters or conditions
4. Analysis parameters

Available analysis methods:
- describe_dataset: Basic dataset overview and statistics
- count_values: Count occurrences of values
- aggregate_data: Sum, mean, median, etc. by groups
- filter_data: Filter data based on conditions
- correlation_analysis: Find relationships between variables
- clustering_analysis: Group similar records
- outlier_detection: Find unusual data points
- anomaly_detection: Detect anomalous patterns
- trend_analysis: Analyze changes over time
- time_series_patterns: Find temporal patterns
- group_comparison: Compare different groups
- pattern_recognition: Find general patterns

Respond with a JSON object in this exact format:
{
    "query_type": "descriptive|exploratory|diagnostic|temporal|comparative|causal",
    "analysis_method": "one of the methods above",
    "columns": ["column1", "column2"],
    "filters": [{"column": "col", "operator": "equals|greater_than|less_than|contains", "value": "val"}],
    "parameters": {"param1": "value1"},
    "confidence": 0.8
}

Extract column names that best match the user's intent. If no specific columns are mentioned, use an empty list.
For filters, extract any conditions mentioned in the query.
Set confidence between 0.1 and 1.0 based on how clear the intent is."""

    def _parse_intent_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response from LLM."""
        try:
            # Extract JSON from content (handle markdown code blocks)
            json_match = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL
            )
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON object directly
                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    return None

            # Parse JSON
            intent_data = json.loads(json_str)

            # Validate required fields
            required_fields = ["query_type", "analysis_method", "columns", "confidence"]
            for field in required_fields:
                if field not in intent_data:
                    logger.warning(f"Missing required field: {field}")
                    return None

            return intent_data

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing intent response: {e}")
            return None

    def _fallback_intent_extraction(
        self, query: str, initial_intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback intent extraction using simple heuristics."""

        # Extract potential column names from query
        columns = self._extract_column_names_heuristic(query)

        # Extract simple filters
        filters = self._extract_filters_heuristic(query)

        return {
            "query_type": initial_intent["query_type"].value,
            "analysis_method": initial_intent["method"].value,
            "columns": columns,
            "filters": filters,
            "parameters": {},
            "confidence": max(0.2, initial_intent["confidence"] - 0.2),
        }

    def _extract_column_names_heuristic(self, query: str) -> List[str]:
        """Extract column names using simple heuristics."""
        if not self.available_columns:
            return []

        found_columns = []
        query_lower = query.lower()

        for col in self.available_columns:
            col_lower = col.lower()

            # Direct match
            if col_lower in query_lower:
                found_columns.append(col)
                continue

            # Partial match (for columns with underscores or similar)
            col_parts = col_lower.replace("_", " ").split()
            if any(part in query_lower for part in col_parts if len(part) > 3):
                found_columns.append(col)

        return found_columns[:5]  # Limit to 5 columns

    def _extract_filters_heuristic(self, query: str) -> List[Dict[str, Any]]:
        """Extract simple filters using heuristics."""
        filters = []

        # Simple patterns for filters
        filter_patterns = [
            (r'(\w+)\s*=\s*(["\']?)([^"\'>\s]+)\2', "equals"),
            (r"(\w+)\s*>\s*(\d+(?:\.\d+)?)", "greater_than"),
            (r"(\w+)\s*<\s*(\d+(?:\.\d+)?)", "less_than"),
            (r'(\w+)\s+contains?\s+(["\']?)([^"\'>\s]+)\2', "contains"),
        ]

        for pattern, operator in filter_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                column = match.group(1)
                value = match.group(3) if len(match.groups()) >= 3 else match.group(2)

                # Convert numeric values
                try:
                    if "." in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass  # Keep as string

                filters.append({"column": column, "operator": operator, "value": value})

        return filters

    def _validate_and_refine_intent(
        self, intent_data: Dict[str, Any], query: str
    ) -> QueryIntent:
        """Validate and refine the extracted intent."""

        try:
            # Map string values to enums
            query_type = QueryType(intent_data["query_type"])
            analysis_method = AnalysisMethod(intent_data["analysis_method"])
        except ValueError as e:
            logger.warning(f"Invalid enum value: {e}")
            # Fallback to descriptive
            query_type = QueryType.DESCRIPTIVE
            analysis_method = AnalysisMethod.DESCRIBE_DATASET

        # Validate columns
        columns = intent_data.get("columns", [])
        if self.available_columns:
            validated_columns = [
                col for col in columns if col in self.available_columns
            ]
            # If no valid columns found but some were specified, try fuzzy matching
            if not validated_columns and columns:
                validated_columns = self._fuzzy_match_columns(columns)
        else:
            validated_columns = columns

        # Validate filters
        filters = intent_data.get("filters", [])
        validated_filters = []
        for f in filters:
            if (
                isinstance(f, dict)
                and "column" in f
                and "operator" in f
                and "value" in f
            ):
                validated_filters.append(f)

        # Get parameters
        parameters = intent_data.get("parameters", {})

        # Ensure confidence is in valid range
        confidence = max(0.1, min(1.0, intent_data.get("confidence", 0.5)))

        return QueryIntent(
            query_type=query_type,
            analysis_method=analysis_method,
            columns=validated_columns,
            filters=validated_filters,
            parameters=parameters,
            confidence=confidence,
            raw_query=query,
        )

    def _fuzzy_match_columns(self, columns: List[str]) -> List[str]:
        """Perform fuzzy matching for column names."""
        matched = []

        for col in columns:
            col_lower = col.lower()

            # Find closest match
            best_match = None
            best_score = 0

            for available_col in self.available_columns:
                available_lower = available_col.lower()

                # Simple similarity score
                if col_lower in available_lower or available_lower in col_lower:
                    score = (
                        len(col_lower) / len(available_lower) if available_lower else 0
                    )
                    if score > best_score:
                        best_score = score
                        best_match = available_col

                # Check if words match
                col_words = set(col_lower.replace("_", " ").split())
                available_words = set(available_lower.replace("_", " ").split())

                common_words = col_words.intersection(available_words)
                if common_words and len(common_words) > best_score:
                    best_score = len(common_words)
                    best_match = available_col

            if best_match and best_score > 0.3:
                matched.append(best_match)

        return matched


def create_query_processor(llm_manager) -> QueryProcessor:
    """Create and return query processor instance."""
    return QueryProcessor(llm_manager)
