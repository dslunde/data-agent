"""
Core statistical analysis functions for data agent.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from scipy import stats
from datetime import datetime, timedelta
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
            "data_types": self._analyze_data_types(df)
        }
        
        return result
    
    def aggregate_data(
        self, 
        df: pd.DataFrame, 
        group_by: Union[str, List[str]], 
        agg_column: str,
        agg_functions: List[str] = None
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
            agg_functions = ['count', 'mean', 'median', 'std', 'min', 'max']
        
        try:
            # Validate columns exist
            if isinstance(group_by, str):
                group_by = [group_by]
            
            missing_cols = [col for col in group_by + [agg_column] if col not in df.columns]
            if missing_cols:
                return {"error": f"Missing columns: {missing_cols}"}
            
            # Perform aggregation
            valid_agg_funcs = []
            for func in agg_functions:
                if func == 'count':
                    valid_agg_funcs.append('count')
                elif func in ['mean', 'median', 'std', 'min', 'max', 'sum'] and df[agg_column].dtype in [np.number]:
                    valid_agg_funcs.append(func)
            
            if not valid_agg_funcs:
                return {"error": f"No valid aggregation functions for column {agg_column}"}
            
            grouped = df.groupby(group_by)[agg_column].agg(valid_agg_funcs)
            
            result = {
                "group_by": group_by,
                "agg_column": agg_column,
                "functions": valid_agg_funcs,
                "results": grouped.to_dict() if len(group_by) == 1 else grouped.to_dict('index'),
                "total_groups": len(grouped)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in aggregation: {e}")
            return {"error": str(e)}
    
    def filter_data(
        self, 
        df: pd.DataFrame, 
        filters: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Filter data based on specified conditions.
        
        Args:
            df: DataFrame to filter
            filters: List of filter conditions
            
        Returns:
            Filter results and statistics
        """
        try:
            filtered_df = df.copy()
            applied_filters = []
            
            for filter_spec in filters:
                column = filter_spec.get('column')
                operator = filter_spec.get('operator')
                value = filter_spec.get('value')
                
                if column not in df.columns:
                    continue
                
                original_count = len(filtered_df)
                
                if operator == 'equals':
                    filtered_df = filtered_df[filtered_df[column] == value]
                elif operator == 'not_equals':
                    filtered_df = filtered_df[filtered_df[column] != value]
                elif operator == 'greater_than':
                    filtered_df = filtered_df[filtered_df[column] > value]
                elif operator == 'less_than':
                    filtered_df = filtered_df[filtered_df[column] < value]
                elif operator == 'greater_equal':
                    filtered_df = filtered_df[filtered_df[column] >= value]
                elif operator == 'less_equal':
                    filtered_df = filtered_df[filtered_df[column] <= value]
                elif operator == 'contains' and df[column].dtype == 'object':
                    filtered_df = filtered_df[filtered_df[column].astype(str).str.contains(str(value), na=False)]
                elif operator == 'isin' and isinstance(value, list):
                    filtered_df = filtered_df[filtered_df[column].isin(value)]
                
                new_count = len(filtered_df)
                applied_filters.append({
                    "column": column,
                    "operator": operator,
                    "value": value,
                    "rows_before": original_count,
                    "rows_after": new_count,
                    "rows_filtered": original_count - new_count
                })
            
            result = {
                "original_count": len(df),
                "filtered_count": len(filtered_df),
                "reduction_percentage": ((len(df) - len(filtered_df)) / len(df)) * 100,
                "applied_filters": applied_filters,
                "sample_data": filtered_df.head(10).to_dict('records') if len(filtered_df) > 0 else []
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in filtering: {e}")
            return {"error": str(e)}
    
    def count_analysis(
        self, 
        df: pd.DataFrame, 
        column: str,
        top_n: int = 10
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
                    "most_frequent": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    "most_frequent_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    "least_frequent": str(value_counts.index[-1]) if len(value_counts) > 0 else None,
                    "least_frequent_count": int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in count analysis: {e}")
            return {"error": str(e)}
    
    def group_analysis(
        self, 
        df: pd.DataFrame, 
        group_column: str,
        target_columns: List[str] = None
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
                "group_statistics": {}
            }
            
            for target_col in target_columns:
                if df[target_col].dtype in [np.number]:
                    group_stats = grouped[target_col].agg(['count', 'mean', 'median', 'std', 'min', 'max']).fillna(0)
                    result["group_statistics"][target_col] = group_stats.to_dict('index')
            
            return result
            
        except Exception as e:
            logger.error(f"Error in group analysis: {e}")
            return {"error": str(e)}
    
    def trend_analysis(
        self, 
        df: pd.DataFrame, 
        date_column: str,
        value_column: str,
        period: str = 'M'
    ) -> Dict[str, Any]:
        """
        Analyze trends over time.
        
        Args:
            df: DataFrame to analyze
            date_column: Date/datetime column
            value_column: Value column to analyze trends
            period: Aggregation period ('D', 'W', 'M', 'Q', 'Y')
            
        Returns:
            Trend analysis results
        """
        try:
            if date_column not in df.columns or value_column not in df.columns:
                return {"error": "Required columns not found"}
            
            # Convert to datetime if necessary
            df_copy = df.copy()
            if not pd.api.types.is_datetime64_any_dtype(df_copy[date_column]):
                df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors='coerce')
            
            # Remove rows with invalid dates or values
            valid_data = df_copy.dropna(subset=[date_column, value_column])
            
            if len(valid_data) == 0:
                return {"error": "No valid data for trend analysis"}
            
            # Set date as index and resample
            valid_data = valid_data.set_index(date_column)
            resampled = valid_data[value_column].resample(period).agg(['count', 'mean', 'sum']).fillna(0)
            
            # Calculate trend statistics
            values = resampled['mean'].values
            if len(values) > 1:
                # Linear trend
                x = np.arange(len(values))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                
                trend_direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"
                trend_strength = abs(r_value)
            else:
                slope, r_value, trend_direction, trend_strength = 0, 0, "insufficient_data", 0
            
            result = {
                "date_column": date_column,
                "value_column": value_column,
                "period": period,
                "data_points": len(resampled),
                "date_range": {
                    "start": valid_data.index.min().isoformat(),
                    "end": valid_data.index.max().isoformat()
                },
                "trend_statistics": {
                    "slope": float(slope),
                    "correlation": float(r_value),
                    "direction": trend_direction,
                    "strength": float(trend_strength)
                },
                "time_series_data": resampled.to_dict('index'),
                "summary": {
                    "total_periods": len(resampled),
                    "average_value": float(resampled['mean'].mean()),
                    "total_sum": float(resampled['sum'].sum()),
                    "peak_period": resampled['mean'].idxmax().isoformat() if len(resampled) > 0 else None,
                    "peak_value": float(resampled['mean'].max()) if len(resampled) > 0 else 0
                }
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
            "duplicate_rows": int(df.duplicated().sum())
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
                        "q75": float(series.quantile(0.75))
                    }
                }
                
                numeric_stats[col] = stats_dict
                
            except Exception as e:
                logger.warning(f"Error calculating stats for {col}: {e}")
                numeric_stats[col] = {"error": str(e)}
        
        return numeric_stats
    
    def _describe_categorical_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Describe categorical columns."""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            return {"message": "No categorical columns found"}
        
        categorical_stats = {}
        
        for col in categorical_cols:
            try:
                value_counts = df[col].value_counts()
                
                stats_dict = {
                    "count": int(df[col].count()),
                    "unique": int(df[col].nunique()),
                    "top_value": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    "top_freq": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    "cardinality_ratio": df[col].nunique() / len(df) if len(df) > 0 else 0
                }
                
                categorical_stats[col] = stats_dict
                
            except Exception as e:
                logger.warning(f"Error calculating stats for {col}: {e}")
                categorical_stats[col] = {"error": str(e)}
        
        return categorical_stats
    
    def _describe_datetime_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Describe datetime columns."""
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        
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
                    "unique_dates": int(series.nunique())
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
            "missing_percentages": missing_percentages[missing_percentages > 0].to_dict(),
            "most_missing_column": missing_counts.idxmax() if missing_counts.sum() > 0 else None,
            "most_missing_percentage": float(missing_percentages.max()) if missing_counts.sum() > 0 else 0
        }
    
    def _analyze_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data types distribution."""
        dtype_counts = df.dtypes.value_counts()
        
        return {
            "type_distribution": dtype_counts.to_dict(),
            "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(df.select_dtypes(include=['object', 'category']).columns),
            "datetime_columns": len(df.select_dtypes(include=['datetime64']).columns),
            "boolean_columns": len(df.select_dtypes(include=[bool]).columns)
        }


def get_default_statistical_analyzer() -> StatisticalAnalyzer:
    """Get default statistical analyzer instance."""
    return StatisticalAnalyzer()
