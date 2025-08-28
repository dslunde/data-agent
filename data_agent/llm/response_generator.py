"""
Response generation system with methodology explanation and evidence compilation.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from .query_processor import QueryIntent, AnalysisMethod

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Generates natural language responses from analysis results."""

    def __init__(self, llm_manager):
        """Initialize response generator."""
        self.llm_manager = llm_manager

    async def generate_response(
        self,
        intent: QueryIntent,
        analysis_results: Dict[str, Any],
        dataset_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive response from analysis results.

        Args:
            intent: Original query intent
            analysis_results: Results from analysis
            dataset_info: Information about the dataset

        Returns:
            Response dictionary with natural language explanation
        """
        logger.info(f"Generating response for {intent.analysis_method.value}")

        try:
            # Generate response based on analysis method
            if "error" in analysis_results:
                return await self._generate_error_response(intent, analysis_results)

            # Create evidence summary
            evidence = self._compile_evidence(intent, analysis_results)

            # Generate methodology explanation
            methodology = self._generate_methodology(intent, analysis_results)

            # Generate main response
            main_response = await self._generate_main_response(
                intent, analysis_results, evidence
            )

            # Add statistical caveats
            caveats = self._generate_caveats(intent, analysis_results)

            # Compile final response
            response = {
                "query": intent.raw_query,
                "analysis_method": intent.analysis_method.value,
                "confidence": intent.confidence,
                "response": main_response,
                "methodology": methodology,
                "evidence": evidence,
                "caveats": caveats,
                "timestamp": datetime.now().isoformat(),
                "raw_results": analysis_results,  # Keep for debugging
            }

            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "query": intent.raw_query,
                "error": str(e),
                "response": "I encountered an error while analyzing your data. Please try rephrasing your question.",
                "timestamp": datetime.now().isoformat(),
            }

    async def _generate_error_response(
        self, intent: QueryIntent, analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate response for analysis errors."""
        error_message = analysis_results.get("error", "Unknown error occurred")

        # Generate helpful error response using LLM
        system_prompt = """You are a data analysis assistant. The user asked a question but there was an error during analysis. 
        
        Provide a helpful response that:
        1. Explains what went wrong in simple terms
        2. Suggests how the user might rephrase their question
        3. Offers alternative approaches if possible
        
        Be empathetic and constructive."""

        user_prompt = f"""
        User query: "{intent.raw_query}"
        Analysis method attempted: {intent.analysis_method.value}
        Error encountered: {error_message}
        
        Please provide a helpful response to the user.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            llm_response = await self.llm_manager.generate_response(
                messages, temperature=0.3
            )
            response_text = llm_response.get(
                "content", "I encountered an error analyzing your data."
            )
        except Exception as e:
            logger.error(f"Error generating error response: {e}")
            response_text = f"I encountered an error: {error_message}. Please try rephrasing your question."

        return {
            "query": intent.raw_query,
            "error": error_message,
            "response": response_text,
            "timestamp": datetime.now().isoformat(),
        }

    def _compile_evidence(
        self, intent: QueryIntent, results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compile supporting evidence from analysis results."""
        evidence = {
            "data_points_analyzed": 0,
            "columns_used": intent.columns,
            "filters_applied": intent.filters,
            "statistical_measures": {},
            "key_findings": [],
        }

        # Extract key metrics based on analysis method
        if intent.analysis_method == AnalysisMethod.DESCRIBE_DATASET:
            if "overview" in results:
                evidence["data_points_analyzed"] = results["overview"].get(
                    "total_rows", 0
                )
                evidence["statistical_measures"] = results.get("numeric_summary", {})

        elif intent.analysis_method == AnalysisMethod.CORRELATION_ANALYSIS:
            if "strong_correlations" in results:
                evidence["key_findings"] = results["strong_correlations"][:5]
                evidence["statistical_measures"]["correlation_count"] = results.get(
                    "total_correlations", 0
                )

        elif intent.analysis_method == AnalysisMethod.OUTLIER_DETECTION:
            if "outlier_summary" in results:
                evidence["statistical_measures"]["outliers_found"] = results[
                    "outlier_summary"
                ].get("total_outliers", 0)
                evidence["statistical_measures"]["outlier_percentage"] = results[
                    "outlier_summary"
                ].get("outlier_percentage", 0)

        elif intent.analysis_method == AnalysisMethod.CLUSTERING_ANALYSIS:
            if "n_clusters" in results:
                evidence["statistical_measures"]["clusters_found"] = results[
                    "n_clusters"
                ]
                if "metrics" in results:
                    evidence["statistical_measures"]["silhouette_score"] = results[
                        "metrics"
                    ].get("silhouette_score", 0)

        elif intent.analysis_method == AnalysisMethod.TREND_ANALYSIS:
            if "trend_statistics" in results:
                evidence["statistical_measures"] = results["trend_statistics"]
                evidence["data_points_analyzed"] = results.get("data_points", 0)

        # Add method-specific evidence
        self._add_method_specific_evidence(intent, results, evidence)

        return evidence

    def _add_method_specific_evidence(
        self, intent: QueryIntent, results: Dict[str, Any], evidence: Dict[str, Any]
    ):
        """Add method-specific evidence details."""

        if intent.analysis_method == AnalysisMethod.COUNT_VALUES:
            if "total_count" in results:
                evidence["data_points_analyzed"] = results["total_count"]
                evidence["statistical_measures"]["unique_values"] = results.get(
                    "unique_count", 0
                )

        elif intent.analysis_method == AnalysisMethod.AGGREGATE_DATA:
            if "total_groups" in results:
                evidence["statistical_measures"]["groups_analyzed"] = results[
                    "total_groups"
                ]

        elif intent.analysis_method == AnalysisMethod.TIME_SERIES_PATTERNS:
            if "date_range" in results:
                evidence["statistical_measures"]["time_span_days"] = results[
                    "date_range"
                ].get("span_days", 0)

    def _generate_methodology(
        self, intent: QueryIntent, results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate methodology explanation."""
        methodology = {
            "analysis_type": intent.analysis_method.value,
            "approach": "",
            "parameters": intent.parameters,
            "assumptions": [],
            "limitations": [],
        }

        # Method-specific methodology
        if intent.analysis_method == AnalysisMethod.CORRELATION_ANALYSIS:
            methodology["approach"] = (
                "Pearson correlation analysis to measure linear relationships between variables"
            )
            methodology["assumptions"] = [
                "Linear relationships between variables",
                "Variables are approximately normally distributed",
                "No extreme outliers affecting correlation",
            ]
            methodology["limitations"] = [
                "Only captures linear relationships",
                "Correlation does not imply causation",
            ]

        elif intent.analysis_method == AnalysisMethod.CLUSTERING_ANALYSIS:
            algorithm = results.get("algorithm", "unknown")
            methodology["approach"] = (
                f"Unsupervised clustering using {algorithm} algorithm"
            )

            if algorithm == "kmeans":
                methodology["assumptions"] = [
                    "Clusters are roughly spherical",
                    "Clusters have similar sizes",
                    "Features are scaled appropriately",
                ]
                methodology["limitations"] = [
                    "Assumes spherical clusters",
                    "Sensitive to initialization",
                    "Number of clusters must be specified",
                ]
            elif algorithm == "dbscan":
                methodology["assumptions"] = [
                    "Clusters have varying densities",
                    "Noise points can be identified",
                ]
                methodology["limitations"] = [
                    "Sensitive to parameter selection",
                    "Struggles with varying densities",
                ]

        elif intent.analysis_method == AnalysisMethod.OUTLIER_DETECTION:
            methods = results.get("methods_used", [])
            methodology["approach"] = (
                f"Multi-method outlier detection using {', '.join(methods)}"
            )
            methodology["assumptions"] = [
                "Outliers are significantly different from normal data",
                "Normal data follows expected distributions",
            ]
            methodology["limitations"] = [
                "May flag legitimate extreme values",
                "Sensitive to data distribution assumptions",
            ]

        elif intent.analysis_method == AnalysisMethod.TREND_ANALYSIS:
            methodology["approach"] = "Linear regression trend analysis over time"
            methodology["assumptions"] = [
                "Linear trend over time",
                "Independent observations",
                "Consistent measurement intervals",
            ]
            methodology["limitations"] = [
                "Assumes linear trends only",
                "May not capture seasonal patterns",
                "Sensitive to outliers",
            ]

        else:
            # Generic methodology for other methods
            methodology["approach"] = (
                f"Statistical analysis using {intent.analysis_method.value}"
            )
            methodology["assumptions"] = ["Standard statistical assumptions apply"]
            methodology["limitations"] = [
                "Results depend on data quality and completeness"
            ]

        return methodology

    async def _generate_main_response(
        self, intent: QueryIntent, results: Dict[str, Any], evidence: Dict[str, Any]
    ) -> str:
        """Generate the main natural language response."""

        # Create structured summary of results
        results_summary = self._create_results_summary(intent, results)

        system_prompt = """You are an expert data analyst providing insights to business users. Your job is to explain analysis results in clear, actionable language.

        Guidelines:
        1. Start with a direct answer to the user's question
        2. Provide specific numbers and percentages where relevant
        3. Highlight the most important insights
        4. Use business-friendly language, not technical jargon
        5. Be precise but accessible
        6. Focus on actionable insights
        7. Keep response concise but comprehensive
        
        Avoid overstating findings or making claims beyond what the data shows."""

        user_prompt = f"""
        User asked: "{intent.raw_query}"
        
        Analysis performed: {intent.analysis_method.value}
        
        Key results: {json.dumps(results_summary, indent=2)}
        
        Supporting evidence: {json.dumps(evidence, indent=2)}
        
        Please provide a clear, insightful response that directly answers the user's question.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            llm_response = await self.llm_manager.generate_response(
                messages=messages, temperature=0.2, max_tokens=1000
            )

            return llm_response.get(
                "content", "I was unable to generate a response for your analysis."
            )

        except Exception as e:
            logger.error(f"Error generating main response: {e}")
            return f"Based on the analysis, I found some interesting patterns in your data, but encountered an issue generating the full response. The analysis did complete successfully with the following key findings: {results_summary}"

    def _create_results_summary(
        self, intent: QueryIntent, results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a structured summary of analysis results."""
        summary = {}

        if intent.analysis_method == AnalysisMethod.DESCRIBE_DATASET:
            summary = {
                "total_rows": results.get("overview", {}).get("total_rows", 0),
                "total_columns": results.get("overview", {}).get("total_columns", 0),
                "completeness": results.get("overview", {}).get(
                    "completeness_score", 0
                ),
                "key_stats": results.get("numeric_summary", {}),
            }

        elif intent.analysis_method == AnalysisMethod.CORRELATION_ANALYSIS:
            correlations = results.get("strong_correlations", [])
            summary = {
                "strong_correlations_found": len(correlations),
                "top_correlations": correlations[:3],
                "correlation_method": results.get("method", "pearson"),
            }

        elif intent.analysis_method == AnalysisMethod.OUTLIER_DETECTION:
            outlier_summary = results.get("outlier_summary", {})
            summary = {
                "outliers_found": outlier_summary.get("total_outliers", 0),
                "outlier_percentage": outlier_summary.get("outlier_percentage", 0),
                "methods_used": results.get("methods_used", []),
            }

        elif intent.analysis_method == AnalysisMethod.CLUSTERING_ANALYSIS:
            summary = {
                "clusters_identified": results.get("n_clusters", 0),
                "algorithm_used": results.get("algorithm", "unknown"),
                "quality_score": results.get("metrics", {}).get("silhouette_score", 0),
            }

        elif intent.analysis_method == AnalysisMethod.TREND_ANALYSIS:
            trend_stats = results.get("trend_statistics", {})
            summary = {
                "trend_direction": trend_stats.get("direction", "unknown"),
                "trend_strength": trend_stats.get("strength", 0),
                "time_periods": results.get("data_points", 0),
            }

        elif intent.analysis_method == AnalysisMethod.COUNT_VALUES:
            summary = {
                "total_records": results.get("total_count", 0),
                "unique_values": results.get("unique_count", 0),
                "most_frequent": results.get("value_distribution", {}).get(
                    "most_frequent", "N/A"
                ),
            }

        elif intent.analysis_method == AnalysisMethod.AGGREGATE_DATA:
            summary = {
                "groups_analyzed": results.get("total_groups", 0),
                "aggregation_functions": results.get("functions", []),
                "group_column": results.get("group_by", []),
            }

        elif intent.analysis_method == AnalysisMethod.GROUP_COMPARISON:
            # Extract rich statistical analysis results for group comparison
            group_stats = results.get("group_statistics", {})
            overview = results.get("overview", {})
            
            summary = {
                "total_rows_analyzed": overview.get("total_rows", 0),
                "groups_compared": results.get("group_count", 0),
                "group_column": results.get("group_column", "unknown"),
                "target_columns": results.get("target_columns", []),
                "group_sizes": results.get("group_sizes", {}),
                "statistical_findings": {},
            }
            
            # Extract statistical findings from each target column
            for col_name, col_stats in group_stats.items():
                if isinstance(col_stats, dict) and "group_statistics" in col_stats:
                    group_data = col_stats["group_statistics"]
                    statistical_test = col_stats.get("statistical_comparison", {})
                    
                    # Extract key statistics for each group
                    group_summaries = {}
                    for group_name, stats in group_data.items():
                        group_summaries[group_name] = {
                            "mean": stats.get("mean", 0),
                            "median": stats.get("median", 0),
                            "count": stats.get("count", 0),
                            "std": stats.get("std", 0)
                        }
                    
                    summary["statistical_findings"][col_name] = {
                        "group_statistics": group_summaries,
                        "statistical_test": statistical_test.get("test", "unknown"),
                        "p_value": statistical_test.get("p_value", None),
                        "significant": statistical_test.get("significant", False),
                        "effect_size": statistical_test.get("effect_size", {})
                    }

        else:
            # Generic summary
            summary = {
                "analysis_completed": True,
                "method": intent.analysis_method.value,
            }

        return summary

    def _generate_caveats(
        self, intent: QueryIntent, results: Dict[str, Any]
    ) -> List[str]:
        """Generate statistical caveats and limitations."""
        caveats = []

        # General caveats
        if intent.confidence < 0.7:
            caveats.append(
                "The query interpretation had moderate confidence. Results should be verified."
            )

        # Sample size caveats
        data_points = results.get("overview", {}).get("total_rows", 0)
        if data_points < 30:
            caveats.append(
                f"Small sample size (n={data_points}) limits the reliability of statistical conclusions."
            )

        # Method-specific caveats
        if intent.analysis_method == AnalysisMethod.CORRELATION_ANALYSIS:
            caveats.extend(
                [
                    "Correlation does not imply causation",
                    "Results assume linear relationships between variables",
                ]
            )

            # Check for low correlation counts
            if results.get("summary", {}).get("significant_count", 0) == 0:
                caveats.append(
                    "No significant correlations found - variables may be independent"
                )

        elif intent.analysis_method == AnalysisMethod.CLUSTERING_ANALYSIS:
            caveats.append(
                "Clustering results may vary with different parameters or algorithms"
            )

            # Quality-based caveats
            silhouette_score = results.get("metrics", {}).get("silhouette_score", 0)
            if silhouette_score < 0.5:
                caveats.append(
                    "Clustering quality is moderate - clusters may not be well-separated"
                )

        elif intent.analysis_method == AnalysisMethod.OUTLIER_DETECTION:
            caveats.extend(
                [
                    "Outliers may represent valid extreme values rather than errors",
                    "Multiple detection methods used - consensus outliers are more reliable",
                ]
            )

        elif intent.analysis_method == AnalysisMethod.TREND_ANALYSIS:
            caveats.extend(
                [
                    "Linear trend analysis may not capture complex patterns",
                    "Past trends do not guarantee future performance",
                ]
            )

            # Check trend significance
            p_value = results.get("trend_statistics", {}).get("p_value", 1.0)
            if p_value > 0.05:
                caveats.append("Trend is not statistically significant (p > 0.05)")

        # Data quality caveats
        if "missing_data" in results:
            missing_pct = results["missing_data"].get("missing_percentage", 0)
            if missing_pct > 10:
                caveats.append(
                    f"Dataset has {missing_pct:.1f}% missing values which may affect results"
                )

        # Filter caveats
        if intent.filters:
            caveats.append(
                f"Results are based on filtered data ({len(intent.filters)} filter(s) applied)"
            )

        return caveats


def create_response_generator(llm_manager) -> ResponseGenerator:
    """Create and return response generator instance."""
    return ResponseGenerator(llm_manager)
