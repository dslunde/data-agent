"""
Core statistical analysis functions for data agent.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union, Optional, Tuple
import logging
from scipy import stats
from scipy.stats import t, norm
from sklearn.utils import resample
import warnings

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Core statistical analysis functionality."""

    def __init__(self):
        """Initialize statistical analyzer."""
        pass

    def describe_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive descriptive statistics for dataset.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary containing descriptive statistics
        """
        logger.info("Generating descriptive statistics")

        result = {
            "overview": self._get_overview_stats(df),
            "numeric_summary": self._describe_numeric_columns(df),
            "categorical_summary": self._describe_categorical_columns(df),
            "datetime_summary": self._describe_datetime_columns(df),
            "missing_data": self._analyze_missing_data(df),
            "data_types": self._analyze_data_types(df),
        }

        return result

    def aggregate_data(
        self,
        df: pd.DataFrame,
        group_by: Union[str, List[str]],
        agg_column: str,
        agg_functions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate data by specified columns.

        Args:
            df: DataFrame to aggregate
            group_by: Column(s) to group by
            agg_column: Column to aggregate
            agg_functions: List of aggregation functions

        Returns:
            Aggregation results
        """
        if agg_functions is None:
            agg_functions = ["count", "mean", "median", "std", "min", "max"]

        try:
            # Validate columns exist
            if isinstance(group_by, str):
                group_by = [group_by]

            missing_cols = [
                col for col in group_by + [agg_column] if col not in df.columns
            ]
            if missing_cols:
                return {"error": f"Missing columns: {missing_cols}"}

            # Perform aggregation
            valid_agg_funcs = []
            for func in agg_functions:
                if func == "count":
                    valid_agg_funcs.append("count")
                elif func in ["mean", "median", "std", "min", "max", "sum"] and df[
                    agg_column
                ].dtype in [np.number]:
                    valid_agg_funcs.append(func)

            if not valid_agg_funcs:
                return {
                    "error": f"No valid aggregation functions for column {agg_column}"
                }

            grouped = df.groupby(group_by)[agg_column].agg(valid_agg_funcs)

            result = {
                "group_by": group_by,
                "agg_column": agg_column,
                "aggregation_functions": valid_agg_funcs,  # Use standard API key
                "functions": valid_agg_funcs,  # Keep for backward compatibility
                "results": (
                    grouped.to_dict()
                    if len(group_by) == 1
                    else grouped.to_dict("index")
                ),
                "total_groups": len(grouped),
            }

            return result

        except Exception as e:
            logger.error(f"Error in aggregation: {e}")
            return {"error": str(e)}

    def filter_data(
        self, df: pd.DataFrame, filters: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Filter data based on specified conditions.

        Args:
            df: DataFrame to filter
            filters: Dictionary of column filters or list of filter conditions
                    Dict format: {"column": value, "column2": {"min": x, "max": y}}
                    List format: [{"column": "col", "operator": "equals", "value": x}]

        Returns:
            Filter results and statistics
        """
        try:
            filtered_df = df.copy()
            applied_filters = []
            
            # Convert dictionary format to list format for processing
            if isinstance(filters, dict):
                filter_list = []
                for column, condition in filters.items():
                    if isinstance(condition, dict):
                        # Handle range conditions like {"min": 90, "max": 110}
                        if "min" in condition:
                            filter_list.append({"column": column, "operator": "greater_equal", "value": condition["min"]})
                        if "max" in condition:
                            filter_list.append({"column": column, "operator": "less_equal", "value": condition["max"]})
                    else:
                        # Handle direct equality
                        filter_list.append({"column": column, "operator": "equals", "value": condition})
                filters = filter_list

            for filter_spec in filters:
                column = filter_spec.get("column")
                operator = filter_spec.get("operator")
                value = filter_spec.get("value")

                if column not in df.columns:
                    continue

                original_count = len(filtered_df)

                if operator == "equals":
                    filtered_df = filtered_df[filtered_df[column] == value]
                elif operator == "not_equals":
                    filtered_df = filtered_df[filtered_df[column] != value]
                elif operator == "greater_than":
                    filtered_df = filtered_df[filtered_df[column] > value]
                elif operator == "less_than":
                    filtered_df = filtered_df[filtered_df[column] < value]
                elif operator == "greater_equal":
                    filtered_df = filtered_df[filtered_df[column] >= value]
                elif operator == "less_equal":
                    filtered_df = filtered_df[filtered_df[column] <= value]
                elif operator == "contains" and df[column].dtype == "object":
                    filtered_df = filtered_df[
                        filtered_df[column]
                        .astype(str)
                        .str.contains(str(value), na=False)
                    ]
                elif operator == "isin" and isinstance(value, list):
                    filtered_df = filtered_df[filtered_df[column].isin(value)]

                new_count = len(filtered_df)
                applied_filters.append(
                    {
                        "column": column,
                        "operator": operator,
                        "value": value,
                        "rows_before": original_count,
                        "rows_after": new_count,
                        "rows_filtered": original_count - new_count,
                    }
                )

            result = {
                "original_count": len(df),
                "filtered_count": len(filtered_df),
                "reduction_percentage": ((len(df) - len(filtered_df)) / len(df)) * 100,
                "filters_applied": applied_filters,  # Use standard API key
                "applied_filters": applied_filters,  # Keep for backward compatibility
                "sample_data": (
                    filtered_df.head(10).to_dict("records")
                    if len(filtered_df) > 0
                    else []
                ),
            }

            return result

        except Exception as e:
            logger.error(f"Error in filtering: {e}")
            return {"error": str(e)}

    def count_analysis(
        self, df: pd.DataFrame, column: str, top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Perform count analysis on a column.

        Args:
            df: DataFrame to analyze
            column: Column to count
            top_n: Number of top values to return

        Returns:
            Count analysis results
        """
        try:
            if column not in df.columns:
                return {"error": f"Column {column} not found"}

            value_counts = df[column].value_counts(dropna=False)
            null_count = df[column].isnull().sum()

            result = {
                "column": column,
                "total_count": len(df),
                "non_null_count": len(df) - null_count,
                "null_count": int(null_count),
                "unique_count": df[column].nunique(),
                "top_values": value_counts.head(top_n).to_dict(),
                "value_distribution": {
                    "most_frequent": (
                        str(value_counts.index[0]) if len(value_counts) > 0 else None
                    ),
                    "most_frequent_count": (
                        int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
                    ),
                    "least_frequent": (
                        str(value_counts.index[-1]) if len(value_counts) > 0 else None
                    ),
                    "least_frequent_count": (
                        int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0
                    ),
                },
            }

            return result

        except Exception as e:
            logger.error(f"Error in count analysis: {e}")
            return {"error": str(e)}

    def group_analysis(
        self, df: pd.DataFrame, group_column: str, target_columns: List[str] = None
    ) -> Dict[str, Any]:
        """
        Perform group-based analysis.

        Args:
            df: DataFrame to analyze
            group_column: Column to group by
            target_columns: Columns to analyze within groups

        Returns:
            Group analysis results
        """
        try:
            if group_column not in df.columns:
                return {"error": f"Group column {group_column} not found"}

            if target_columns is None:
                target_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                target_columns = [col for col in target_columns if col != group_column]

            # Validate target columns
            target_columns = [col for col in target_columns if col in df.columns]

            if not target_columns:
                return {"error": "No valid target columns for analysis"}

            grouped = df.groupby(group_column)

            result = {
                "group_column": group_column,
                "target_columns": target_columns,
                "group_count": grouped.ngroups,
                "group_sizes": grouped.size().to_dict(),
                "group_statistics": {},
            }

            for target_col in target_columns:
                if df[target_col].dtype in [np.number]:
                    # Calculate basic statistics
                    group_stats = (
                        grouped[target_col]
                        .agg(["count", "mean", "median", "std", "min", "max"])
                        .fillna(0)
                    )
                    
                    # Add confidence intervals and statistical testing
                    enhanced_stats = self._enhance_group_statistics(
                        df, group_column, target_col, group_stats
                    )
                    
                    result["group_statistics"][target_col] = enhanced_stats

            return result

        except Exception as e:
            logger.error(f"Error in group analysis: {e}")
            return {"error": str(e)}

    def trend_analysis(
        self, df: pd.DataFrame, date_column: str, value_column: str, period: str = "ME"
    ) -> Dict[str, Any]:
        """
        Analyze trends over time.

        Args:
            df: DataFrame to analyze
            date_column: Date/datetime column
            value_column: Value column to analyze trends
            period: Aggregation period ('D', 'W', 'ME', 'QE', 'YE')

        Returns:
            Trend analysis results
        """
        try:
            if date_column not in df.columns or value_column not in df.columns:
                return {"error": "Required columns not found"}

            # Map deprecated frequency aliases to new ones
            period_mapping = {
                'M': 'ME',   # Month end
                'Q': 'QE',   # Quarter end  
                'Y': 'YE',   # Year end
                'A': 'YE',   # Annual (year end)
                'H': 'h',    # Hour
                'T': 'min',  # Minute
                'S': 's'     # Second
            }
            period = period_mapping.get(period, period)

            # Validate that we have appropriate data types
            df_copy = df.copy()
            
            # Check if date column can be converted to datetime
            if not pd.api.types.is_datetime64_any_dtype(df_copy[date_column]):
                try:
                    # Handle categorical columns explicitly
                    if pd.api.types.is_categorical_dtype(df_copy[date_column]):
                        # Convert categorical to regular series first to avoid CategoricalIndex issues
                        df_copy[date_column] = df_copy[date_column].astype(str)
                    
                    df_copy[date_column] = pd.to_datetime(
                        df_copy[date_column], errors="coerce"
                    )
                except Exception as e:
                    return {"error": f"Cannot convert '{date_column}' to datetime format: {str(e)}"}
                    
            # Check if we have any valid datetime values
            if df_copy[date_column].isna().all():
                return {"error": f"No valid datetime values found in '{date_column}' column"}
            
            # Check if value column is numeric
            if not pd.api.types.is_numeric_dtype(df_copy[value_column]):
                try:
                    df_copy[value_column] = pd.to_numeric(df_copy[value_column], errors="coerce")
                except Exception:
                    return {"error": f"Cannot convert '{value_column}' to numeric format"}
            
            # Remove rows with invalid dates or values
            valid_data = df_copy.dropna(subset=[date_column, value_column])

            if len(valid_data) == 0:
                return {"error": "No valid data for trend analysis"}
                
            # Check if we have enough data points for meaningful analysis
            if len(valid_data) < 3:
                return {"error": "Not enough data points for trend analysis (minimum 3 required)"}

            # Set date as index and resample
            valid_data = valid_data.set_index(date_column)
            
            # Additional safety check: ensure the index is not categorical
            if hasattr(valid_data.index, 'categories'):
                # Force convert the index to proper datetime if it's still categorical
                valid_data.index = pd.to_datetime(valid_data.index, errors="coerce")
            
            resampled = (
                valid_data[value_column]
                .resample(period)
                .agg(["count", "mean", "sum"])
                .fillna(0)
            )

            # Calculate trend statistics
            values = resampled["mean"].values
            if len(values) > 1:
                # Linear trend
                x = np.arange(len(values))
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    x, values
                )

                trend_direction = (
                    "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"
                )
                trend_strength = abs(r_value)
            else:
                slope, r_value, trend_direction, trend_strength = (
                    0,
                    0,
                    "insufficient_data",
                    0,
                )

            result = {
                "date_column": date_column,
                "value_column": value_column,
                "period": period,
                "data_points": len(resampled),
                "date_range": {
                    "start": valid_data.index.min().isoformat(),
                    "end": valid_data.index.max().isoformat(),
                },
                "trend_statistics": {
                    "slope": float(slope),
                    "correlation": float(r_value),
                    "direction": trend_direction,
                    "strength": float(trend_strength),
                },
                "time_series_data": resampled.to_dict("index"),
                "summary": {
                    "total_periods": len(resampled),
                    "average_value": float(resampled["mean"].mean()),
                    "total_sum": float(resampled["sum"].sum()),
                    "peak_period": (
                        resampled["mean"].idxmax().isoformat()
                        if len(resampled) > 0
                        else None
                    ),
                    "peak_value": (
                        float(resampled["mean"].max()) if len(resampled) > 0 else 0
                    ),
                },
            }

            return result

        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return {"error": str(e)}

    def _get_overview_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get overview statistics for dataset."""
        return {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "completeness_score": ((df.size - df.isnull().sum().sum()) / df.size) * 100,
            "duplicate_rows": int(df.duplicated().sum()),
        }

    def _describe_numeric_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Describe numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return {"message": "No numeric columns found"}

        numeric_stats = {}

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            try:
                stats_dict = {
                    "count": len(series),
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "skewness": float(series.skew()),
                    "kurtosis": float(series.kurtosis()),
                    "quartiles": {
                        "q25": float(series.quantile(0.25)),
                        "q50": float(series.quantile(0.50)),
                        "q75": float(series.quantile(0.75)),
                    },
                }

                numeric_stats[col] = stats_dict

            except Exception as e:
                logger.warning(f"Error calculating stats for {col}: {e}")
                numeric_stats[col] = {"error": str(e)}

        return numeric_stats

    def _describe_categorical_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Describe categorical columns."""
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns

        if len(categorical_cols) == 0:
            return {"message": "No categorical columns found"}

        categorical_stats = {}

        for col in categorical_cols:
            try:
                value_counts = df[col].value_counts()

                stats_dict = {
                    "count": int(df[col].count()),
                    "unique": int(df[col].nunique()),
                    "top_value": (
                        str(value_counts.index[0]) if len(value_counts) > 0 else None
                    ),
                    "top_freq": (
                        int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
                    ),
                    "cardinality_ratio": (
                        df[col].nunique() / len(df) if len(df) > 0 else 0
                    ),
                }

                categorical_stats[col] = stats_dict

            except Exception as e:
                logger.warning(f"Error calculating stats for {col}: {e}")
                categorical_stats[col] = {"error": str(e)}

        return categorical_stats

    def _describe_datetime_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Describe datetime columns."""
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns

        if len(datetime_cols) == 0:
            return {"message": "No datetime columns found"}

        datetime_stats = {}

        for col in datetime_cols:
            try:
                series = df[col].dropna()
                if len(series) == 0:
                    continue

                stats_dict = {
                    "count": len(series),
                    "min_date": series.min().isoformat(),
                    "max_date": series.max().isoformat(),
                    "range_days": (series.max() - series.min()).days,
                    "unique_dates": int(series.nunique()),
                }

                datetime_stats[col] = stats_dict

            except Exception as e:
                logger.warning(f"Error calculating stats for {col}: {e}")
                datetime_stats[col] = {"error": str(e)}

        return datetime_stats

    def _analyze_missing_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns."""
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100

        return {
            "total_missing": int(missing_counts.sum()),
            "columns_with_missing": int((missing_counts > 0).sum()),
            "missing_by_column": missing_counts[missing_counts > 0].to_dict(),
            "missing_percentages": missing_percentages[
                missing_percentages > 0
            ].to_dict(),
            "most_missing_column": (
                missing_counts.idxmax() if missing_counts.sum() > 0 else None
            ),
            "most_missing_percentage": (
                float(missing_percentages.max()) if missing_counts.sum() > 0 else 0
            ),
        }

    def _analyze_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data types distribution."""
        dtype_counts = df.dtypes.value_counts()

        return {
            "type_distribution": dtype_counts.to_dict(),
            "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(
                df.select_dtypes(include=["object", "category"]).columns
            ),
            "datetime_columns": len(df.select_dtypes(include=["datetime64"]).columns),
            "boolean_columns": len(df.select_dtypes(include=[bool]).columns),
        }

    def _enhance_group_statistics(self, df: pd.DataFrame, group_column: str, 
                                 target_column: str, basic_stats: pd.DataFrame) -> Dict[str, Any]:
        """
        Enhance group statistics with confidence intervals and statistical testing.
        
        Args:
            df: DataFrame with data
            group_column: Column used for grouping
            target_column: Target variable column
            basic_stats: Basic statistics from pandas groupby
            
        Returns:
            Enhanced statistics with confidence intervals and tests
        """
        try:
            enhanced_stats = {}
            
            # Convert basic stats to dict format
            basic_stats_dict = basic_stats.to_dict("index")
            
            # Get group data for statistical testing
            groups = []
            group_names = []
            
            for group_name in basic_stats.index:
                group_data = df[df[group_column] == group_name][target_column].dropna()
                if len(group_data) >= 3:  # Minimum for meaningful CI
                    groups.append(group_data.values)
                    group_names.append(group_name)
            
            # Perform statistical tests if we have multiple groups
            statistical_test_results = None
            if len(groups) >= 2:
                statistical_test_results = self._perform_group_comparison_test(
                    groups, group_names
                )
            
            # Calculate confidence intervals and enhanced stats for each group
            for group_name, stats_row in basic_stats_dict.items():
                group_data = df[df[group_column] == group_name][target_column].dropna()
                
                # Calculate confidence intervals
                confidence_intervals = self._calculate_mean_confidence_interval(group_data)
                
                # Calculate additional robust statistics
                robust_stats = self._calculate_robust_statistics(group_data)
                
                enhanced_stats[group_name] = {
                    **stats_row,  # Include original statistics
                    "confidence_intervals": confidence_intervals,
                    "robust_statistics": robust_stats,
                    "sample_size": len(group_data)
                }
            
            # Add overall test results
            result = {
                "group_statistics": enhanced_stats,
                "statistical_comparison": statistical_test_results,
                "methodology": {
                    "confidence_level": 0.95,
                    "confidence_method": "t-distribution for means",
                    "robust_statistics": "median, IQR, MAD",
                    "group_testing": "ANOVA with assumption validation"
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error enhancing group statistics: {e}")
            # Fallback to basic statistics
            return basic_stats.to_dict("index")
    
    def _calculate_mean_confidence_interval(self, data: pd.Series, confidence: float = 0.95) -> Dict[str, Any]:
        """
        Calculate confidence interval for the mean.
        
        Args:
            data: Data series
            confidence: Confidence level (default 0.95)
            
        Returns:
            Confidence interval information
        """
        try:
            if len(data) < 2:
                return {"error": "Insufficient data for confidence interval"}
            
            mean_val = float(data.mean())
            std_val = float(data.std(ddof=1))
            n = len(data)
            
            # Use t-distribution for small samples
            alpha = 1 - confidence
            t_critical = t.ppf(1 - alpha/2, df=n-1)
            
            margin_error = t_critical * std_val / np.sqrt(n)
            
            return {
                "mean": mean_val,
                "lower": float(mean_val - margin_error),
                "upper": float(mean_val + margin_error),
                "margin_error": float(margin_error),
                "confidence_level": confidence,
                "method": "t-distribution"
            }
            
        except Exception as e:
            logger.error(f"Error calculating confidence interval: {e}")
            return {"error": str(e)}
    
    def _calculate_robust_statistics(self, data: pd.Series) -> Dict[str, Any]:
        """
        Calculate robust statistical measures.
        
        Args:
            data: Data series
            
        Returns:
            Robust statistics
        """
        try:
            if len(data) == 0:
                return {"error": "No data for robust statistics"}
            
            # Calculate robust measures
            median_val = float(data.median())
            q25 = float(data.quantile(0.25))
            q75 = float(data.quantile(0.75))
            iqr = q75 - q25
            
            # Median Absolute Deviation (MAD)
            mad = float(np.median(np.abs(data - median_val)))
            
            # Robust coefficient of variation
            robust_cv = mad / median_val if median_val != 0 else np.inf
            
            return {
                "median": median_val,
                "q25": q25,
                "q75": q75,
                "iqr": iqr,
                "mad": mad,
                "robust_cv": float(robust_cv),
                "outlier_bounds": {
                    "lower": q25 - 1.5 * iqr,
                    "upper": q75 + 1.5 * iqr
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating robust statistics: {e}")
            return {"error": str(e)}
    
    def _perform_group_comparison_test(self, groups: List[np.ndarray], 
                                     group_names: List[str]) -> Dict[str, Any]:
        """
        Perform statistical test to compare groups.
        
        Args:
            groups: List of group data arrays
            group_names: Names of the groups
            
        Returns:
            Statistical test results
        """
        try:
            if len(groups) < 2:
                return {"error": "Need at least 2 groups for comparison"}
            
            # Test normality assumption
            normality_results = []
            all_normal = True
            
            for i, group in enumerate(groups):
                if len(group) >= 3 and len(group) <= 5000:
                    try:
                        stat, p_val = stats.shapiro(group)
                        is_normal = p_val > 0.05
                        normality_results.append({
                            "group": group_names[i],
                            "statistic": float(stat),
                            "p_value": float(p_val),
                            "normal": is_normal
                        })
                        if not is_normal:
                            all_normal = False
                    except Exception as e:
                        logger.warning(f"Normality test failed for group {group_names[i]}: {e}")
                        all_normal = False
            
            # Test homogeneity of variance
            homogeneity_result = None
            homogeneous = True
            
            if len(groups) >= 2:
                try:
                    stat, p_val = stats.levene(*groups)
                    homogeneous = p_val > 0.05
                    homogeneity_result = {
                        "statistic": float(stat),
                        "p_value": float(p_val),
                        "homogeneous": homogeneous
                    }
                except Exception as e:
                    logger.warning(f"Homogeneity test failed: {e}")
                    homogeneous = False
            
            # Choose appropriate test
            if all_normal and homogeneous:
                # Use ANOVA
                f_stat, p_value = stats.f_oneway(*groups)
                test_name = "ANOVA (Parametric)"
                statistic = f_stat
                
                # Calculate eta-squared effect size
                effect_size = self._calculate_anova_effect_size(groups, f_stat)
                
            else:
                # Use Kruskal-Wallis test
                h_stat, p_value = stats.kruskal(*groups)
                test_name = "Kruskal-Wallis (Non-parametric)"
                statistic = h_stat
                
                # Calculate epsilon-squared effect size
                effect_size = self._calculate_kruskal_effect_size(groups, h_stat)
            
            # Pairwise comparisons if significant
            pairwise_results = None
            if p_value < 0.05 and len(groups) > 2:
                pairwise_results = self._perform_pairwise_comparisons(groups, group_names)
            
            result = {
                "test": test_name,
                "statistic": float(statistic),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "effect_size": effect_size,
                "assumptions": {
                    "normality_tests": normality_results,
                    "homogeneity_test": homogeneity_result,
                    "assumptions_met": all_normal and homogeneous
                },
                "pairwise_comparisons": pairwise_results,
                "sample_sizes": [len(group) for group in groups],
                "group_names": group_names
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in group comparison test: {e}")
            return {"error": str(e)}
    
    def _calculate_anova_effect_size(self, groups: List[np.ndarray], f_stat: float) -> Dict[str, Any]:
        """Calculate eta-squared effect size for ANOVA."""
        try:
            k = len(groups)  # number of groups
            n_total = sum(len(group) for group in groups)
            df_between = k - 1
            df_within = n_total - k
            
            eta_squared = (f_stat * df_between) / (f_stat * df_between + df_within)
            
            # Interpret effect size
            if eta_squared < 0.01:
                interpretation = "negligible"
            elif eta_squared < 0.06:
                interpretation = "small"
            elif eta_squared < 0.14:
                interpretation = "medium"
            else:
                interpretation = "large"
            
            return {
                "eta_squared": float(eta_squared),
                "interpretation": interpretation,
                "type": "eta-squared"
            }
            
        except Exception as e:
            logger.error(f"Error calculating ANOVA effect size: {e}")
            return {"error": str(e)}
    
    def _calculate_kruskal_effect_size(self, groups: List[np.ndarray], h_stat: float) -> Dict[str, Any]:
        """Calculate epsilon-squared effect size for Kruskal-Wallis."""
        try:
            n_total = sum(len(group) for group in groups)
            k = len(groups)
            
            epsilon_squared = (h_stat - k + 1) / (n_total - k)
            epsilon_squared = max(0, epsilon_squared)  # Ensure non-negative
            
            # Interpret effect size (similar to eta-squared)
            if epsilon_squared < 0.01:
                interpretation = "negligible"
            elif epsilon_squared < 0.06:
                interpretation = "small"
            elif epsilon_squared < 0.14:
                interpretation = "medium"
            else:
                interpretation = "large"
            
            return {
                "epsilon_squared": float(epsilon_squared),
                "interpretation": interpretation,
                "type": "epsilon-squared"
            }
            
        except Exception as e:
            logger.error(f"Error calculating Kruskal-Wallis effect size: {e}")
            return {"error": str(e)}
    
    def _perform_pairwise_comparisons(self, groups: List[np.ndarray], 
                                    group_names: List[str]) -> List[Dict[str, Any]]:
        """
        Perform pairwise comparisons between groups.
        
        Args:
            groups: List of group data arrays
            group_names: Names of the groups
            
        Returns:
            List of pairwise comparison results
        """
        try:
            comparisons = []
            
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    group1, group2 = groups[i], groups[j]
                    name1, name2 = group_names[i], group_names[j]
                    
                    # Perform Mann-Whitney U test (non-parametric)
                    try:
                        statistic, p_value = stats.mannwhitneyu(
                            group1, group2, alternative="two-sided"
                        )
                        
                        # Calculate effect size (rank-biserial correlation)
                        n1, n2 = len(group1), len(group2)
                        r = 1 - (2 * statistic) / (n1 * n2)
                        
                        # Effect size interpretation
                        abs_r = abs(r)
                        if abs_r < 0.1:
                            effect_interpretation = "negligible"
                        elif abs_r < 0.3:
                            effect_interpretation = "small"
                        elif abs_r < 0.5:
                            effect_interpretation = "medium"
                        else:
                            effect_interpretation = "large"
                        
                        comparisons.append({
                            "group1": name1,
                            "group2": name2,
                            "statistic": float(statistic),
                            "p_value": float(p_value),
                            "significant": p_value < 0.05,
                            "effect_size": {
                                "rank_biserial_correlation": float(r),
                                "interpretation": effect_interpretation
                            },
                            "medians": {
                                name1: float(np.median(group1)),
                                name2: float(np.median(group2))
                            }
                        })
                        
                    except Exception as e:
                        logger.warning(f"Pairwise comparison failed for {name1} vs {name2}: {e}")
                        continue
            
            return comparisons
            
        except Exception as e:
            logger.error(f"Error in pairwise comparisons: {e}")
            return []


def get_default_statistical_analyzer() -> StatisticalAnalyzer:
    """Get default statistical analyzer instance."""
    return StatisticalAnalyzer()
