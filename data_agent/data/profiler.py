"""
Data profiling and caching functionality.
"""

import pandas as pd
import numpy as np
import pickle
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)


class DataProfiler:
    """Profiles datasets to extract comprehensive statistics and insights."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize profiler with optional cache directory."""
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./data/.cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def profile_dataset(
        self, 
        df: pd.DataFrame, 
        cache_key: Optional[str] = None,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Generate comprehensive data profile.
        
        Args:
            df: DataFrame to profile
            cache_key: Optional cache key for storing results
            force_refresh: Force recalculation even if cached
            
        Returns:
            Comprehensive data profile
        """
        # Try to load from cache first
        if cache_key and not force_refresh:
            cached_profile = self._load_from_cache(f"profile_{cache_key}")
            if cached_profile:
                logger.info("Loaded profile from cache")
                return cached_profile
        
        logger.info("Generating data profile...")
        
        profile = {
            "metadata": self._get_metadata(df),
            "overview": self._get_overview(df),
            "columns": self._profile_columns(df),
            "correlations": self._analyze_correlations(df),
            "distributions": self._analyze_distributions(df),
            "patterns": self._identify_patterns(df),
            "relationships": self._analyze_relationships(df),
            "summary_stats": self._get_summary_statistics(df)
        }
        
        # Cache the results if cache_key provided
        if cache_key:
            self._save_to_cache(f"profile_{cache_key}", profile)
        
        return profile
    
    def _get_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get dataset metadata."""
        return {
            "shape": df.shape,
            "size": df.size,
            "memory_usage_bytes": df.memory_usage(deep=True).sum(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "index_type": str(type(df.index)),
            "created_at": datetime.now().isoformat()
        }
    
    def _get_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get high-level overview."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        
        return {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(categorical_cols),
            "datetime_columns": len(datetime_cols),
            "missing_cells": df.isnull().sum().sum(),
            "duplicate_rows": df.duplicated().sum(),
            "completeness_score": (1 - df.isnull().sum().sum() / df.size) * 100
        }
    
    def _profile_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Profile each column individually."""
        column_profiles = {}
        
        for col in df.columns:
            column_profiles[col] = self._profile_single_column(df[col])
        
        return column_profiles
    
    def _profile_single_column(self, series: pd.Series) -> Dict[str, Any]:
        """Profile a single column."""
        profile = {
            "name": series.name,
            "dtype": str(series.dtype),
            "count": len(series),
            "non_null_count": series.count(),
            "null_count": series.isnull().sum(),
            "null_percentage": (series.isnull().sum() / len(series)) * 100,
            "unique_count": series.nunique(),
            "unique_percentage": (series.nunique() / len(series)) * 100 if len(series) > 0 else 0
        }
        
        # Type-specific profiling
        if series.dtype in [np.number]:
            profile.update(self._profile_numeric_column(series))
        elif series.dtype in ['object', 'category']:
            profile.update(self._profile_categorical_column(series))
        elif 'datetime' in str(series.dtype):
            profile.update(self._profile_datetime_column(series))
        
        return profile
    
    def _profile_numeric_column(self, series: pd.Series) -> Dict[str, Any]:
        """Profile numeric column."""
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return {"numeric_stats": "No valid numeric data"}
        
        try:
            stats = {
                "min": float(clean_series.min()),
                "max": float(clean_series.max()),
                "mean": float(clean_series.mean()),
                "median": float(clean_series.median()),
                "std": float(clean_series.std()),
                "var": float(clean_series.var()),
                "skewness": float(clean_series.skew()),
                "kurtosis": float(clean_series.kurtosis()),
                "range": float(clean_series.max() - clean_series.min())
            }
            
            # Quantiles
            quantiles = clean_series.quantile([0.01, 0.05, 0.25, 0.75, 0.95, 0.99])
            stats["quantiles"] = {
                f"q_{int(q*100)}": float(v) for q, v in quantiles.items()
            }
            
            # Outlier detection (IQR method)
            Q1, Q3 = clean_series.quantile([0.25, 0.75])
            IQR = Q3 - Q1
            if IQR > 0:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = clean_series[(clean_series < lower_bound) | (clean_series > upper_bound)]
                stats["outlier_count"] = len(outliers)
                stats["outlier_percentage"] = (len(outliers) / len(clean_series)) * 100
            else:
                stats["outlier_count"] = 0
                stats["outlier_percentage"] = 0.0
            
            # Zero and negative value counts
            stats["zero_count"] = int((clean_series == 0).sum())
            stats["negative_count"] = int((clean_series < 0).sum())
            stats["positive_count"] = int((clean_series > 0).sum())
            
            return {"numeric_stats": stats}
            
        except Exception as e:
            logger.warning(f"Error profiling numeric column {series.name}: {e}")
            return {"numeric_stats": "Error calculating statistics"}
    
    def _profile_categorical_column(self, series: pd.Series) -> Dict[str, Any]:
        """Profile categorical column."""
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return {"categorical_stats": "No valid categorical data"}
        
        try:
            # Value counts
            value_counts = clean_series.value_counts()
            
            stats = {
                "most_frequent": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                "most_frequent_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                "least_frequent": str(value_counts.index[-1]) if len(value_counts) > 0 else None,
                "least_frequent_count": int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0,
                "cardinality": len(value_counts)
            }
            
            # Top categories (up to 10)
            top_categories = value_counts.head(10).to_dict()
            stats["top_categories"] = {str(k): int(v) for k, v in top_categories.items()}
            
            # String length statistics if applicable
            if series.dtype == 'object':
                str_lengths = clean_series.astype(str).str.len()
                stats["string_lengths"] = {
                    "min_length": int(str_lengths.min()),
                    "max_length": int(str_lengths.max()),
                    "avg_length": float(str_lengths.mean()),
                    "median_length": float(str_lengths.median())
                }
            
            return {"categorical_stats": stats}
            
        except Exception as e:
            logger.warning(f"Error profiling categorical column {series.name}: {e}")
            return {"categorical_stats": "Error calculating statistics"}
    
    def _profile_datetime_column(self, series: pd.Series) -> Dict[str, Any]:
        """Profile datetime column."""
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return {"datetime_stats": "No valid datetime data"}
        
        try:
            stats = {
                "min_date": clean_series.min().isoformat(),
                "max_date": clean_series.max().isoformat(),
                "date_range_days": (clean_series.max() - clean_series.min()).days,
            }
            
            # Frequency analysis
            if len(clean_series) > 1:
                # Infer frequency if possible
                try:
                    freq = pd.infer_freq(clean_series.sort_values())
                    stats["inferred_frequency"] = freq
                except:
                    stats["inferred_frequency"] = None
            
            return {"datetime_stats": stats}
            
        except Exception as e:
            logger.warning(f"Error profiling datetime column {series.name}: {e}")
            return {"datetime_stats": "Error calculating statistics"}
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {"message": "Need at least 2 numeric columns for correlation analysis"}
        
        try:
            # Pearson correlation
            corr_matrix = df[numeric_cols].corr()
            
            # Find strongest correlations
            strong_correlations = []
            for i, col1 in enumerate(corr_matrix.columns):
                for col2 in corr_matrix.columns[i+1:]:
                    corr_value = corr_matrix.loc[col1, col2]
                    if not pd.isna(corr_value) and abs(corr_value) > 0.5:
                        strong_correlations.append({
                            "column1": col1,
                            "column2": col2,
                            "correlation": float(corr_value),
                            "strength": "strong" if abs(corr_value) > 0.8 else "moderate"
                        })
            
            # Sort by absolute correlation value
            strong_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            
            return {
                "correlation_matrix": corr_matrix.to_dict(),
                "strong_correlations": strong_correlations[:10],  # Top 10
                "total_correlations": len(strong_correlations)
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing correlations: {e}")
            return {"error": "Could not calculate correlations"}
    
    def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze distributions of numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {"message": "No numeric columns for distribution analysis"}
        
        distributions = {}
        
        for col in numeric_cols:
            clean_data = df[col].dropna()
            if len(clean_data) == 0:
                continue
                
            try:
                # Basic distribution properties
                dist_info = {
                    "is_normal": self._test_normality(clean_data),
                    "is_uniform": self._test_uniformity(clean_data),
                    "histogram_bins": self._get_histogram_data(clean_data)
                }
                
                distributions[col] = dist_info
                
            except Exception as e:
                logger.warning(f"Error analyzing distribution for {col}: {e}")
        
        return distributions
    
    def _identify_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify patterns in the data."""
        patterns = {
            "time_patterns": self._identify_time_patterns(df),
            "categorical_patterns": self._identify_categorical_patterns(df),
            "numeric_patterns": self._identify_numeric_patterns(df)
        }
        
        return patterns
    
    def _analyze_relationships(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze relationships between columns."""
        relationships = []
        
        # Numeric-categorical relationships
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for num_col in numeric_cols:
            for cat_col in categorical_cols:
                try:
                    # Group statistics
                    grouped = df.groupby(cat_col)[num_col].agg(['count', 'mean', 'std']).fillna(0)
                    if len(grouped) > 1:
                        relationships.append({
                            "type": "numeric_by_categorical",
                            "numeric_column": num_col,
                            "categorical_column": cat_col,
                            "group_statistics": grouped.to_dict()
                        })
                except:
                    pass
        
        return {"relationships": relationships[:10]}  # Limit to first 10
    
    def _get_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics for the entire dataset."""
        return {
            "data_types_summary": df.dtypes.value_counts().to_dict(),
            "missing_data_summary": {
                "columns_with_missing": (df.isnull().sum() > 0).sum(),
                "total_missing_cells": df.isnull().sum().sum(),
                "missing_percentage": (df.isnull().sum().sum() / df.size) * 100
            },
            "uniqueness_summary": {
                "highly_unique_columns": sum(df.nunique() > len(df) * 0.9 for col in df.columns),
                "low_cardinality_columns": sum(df.nunique() < 10 for col in df.columns)
            }
        }
    
    def _test_normality(self, data: pd.Series) -> bool:
        """Test if data follows normal distribution (simple test)."""
        if len(data) < 8:
            return False
        
        try:
            from scipy.stats import shapiro
            _, p_value = shapiro(data.sample(min(5000, len(data))))
            return p_value > 0.05
        except:
            # Fallback to simple skewness/kurtosis check
            return abs(data.skew()) < 2 and abs(data.kurtosis()) < 7
    
    def _test_uniformity(self, data: pd.Series) -> bool:
        """Test if data follows uniform distribution (simple test)."""
        if len(data) < 10:
            return False
        
        try:
            # Simple test: check if standard deviation is close to theoretical uniform std
            theoretical_std = (data.max() - data.min()) / (12**0.5)
            actual_std = data.std()
            return abs(actual_std - theoretical_std) / theoretical_std < 0.2
        except:
            return False
    
    def _get_histogram_data(self, data: pd.Series, bins: int = 20) -> Dict[str, Any]:
        """Get histogram data for visualization."""
        try:
            counts, bin_edges = np.histogram(data, bins=min(bins, len(data.unique())))
            return {
                "counts": counts.tolist(),
                "bin_edges": bin_edges.tolist(),
                "bins": len(counts)
            }
        except:
            return {"error": "Could not generate histogram"}
    
    def _identify_time_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify patterns in datetime columns."""
        patterns = []
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        
        for col in datetime_cols:
            clean_data = df[col].dropna()
            if len(clean_data) > 0:
                patterns.append({
                    "column": col,
                    "pattern_type": "temporal",
                    "date_range": f"{clean_data.min()} to {clean_data.max()}",
                    "span_days": (clean_data.max() - clean_data.min()).days
                })
        
        return patterns
    
    def _identify_categorical_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify patterns in categorical columns."""
        patterns = []
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            if len(value_counts) > 0:
                # Check for dominance patterns
                top_percentage = value_counts.iloc[0] / len(df) * 100
                if top_percentage > 80:
                    patterns.append({
                        "column": col,
                        "pattern_type": "dominant_category",
                        "dominant_value": str(value_counts.index[0]),
                        "percentage": top_percentage
                    })
        
        return patterns
    
    def _identify_numeric_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify patterns in numeric columns."""
        patterns = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            clean_data = df[col].dropna()
            if len(clean_data) > 0:
                # Check for specific patterns
                if (clean_data == clean_data.round()).all():
                    patterns.append({
                        "column": col,
                        "pattern_type": "integer_values",
                        "description": "All values are integers"
                    })
                
                if (clean_data >= 0).all():
                    patterns.append({
                        "column": col,
                        "pattern_type": "non_negative",
                        "description": "All values are non-negative"
                    })
        
        return patterns
    
    def _save_to_cache(self, key: str, data: Any) -> None:
        """Save data to cache."""
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"Saved to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
    
    def _load_from_cache(self, key: str) -> Optional[Any]:
        """Load data from cache."""
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                # Check if cache is not too old (24 hours)
                if datetime.now().timestamp() - cache_file.stat().st_mtime < 24 * 3600:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    logger.debug(f"Loaded from cache: {cache_file}")
                    return data
                else:
                    # Remove old cache
                    cache_file.unlink()
            return None
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
            return None


class DataCache:
    """Simple caching system for analysis results."""
    
    def __init__(self, cache_dir: Optional[str] = None, ttl_hours: int = 24):
        """
        Initialize cache.
        
        Args:
            cache_dir: Cache directory path
            ttl_hours: Time to live for cache entries in hours
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./data/.cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600
    
    def get_cache_key(self, data: Union[pd.DataFrame, str, dict]) -> str:
        """Generate cache key from data."""
        if isinstance(data, pd.DataFrame):
            # Use shape, columns, and sample of data for hash
            key_data = {
                "shape": data.shape,
                "columns": list(data.columns),
                "dtypes": data.dtypes.to_dict(),
                "sample_hash": hashlib.md5(
                    str(data.head().values.tobytes()).encode()
                ).hexdigest()
            }
            key_str = json.dumps(key_data, sort_keys=True)
        else:
            key_str = str(data)
        
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        cache_file = self.cache_dir / f"{key}.pkl"
        
        if not cache_file.exists():
            return None
        
        # Check TTL
        if datetime.now().timestamp() - cache_file.stat().st_mtime > self.ttl_seconds:
            cache_file.unlink()  # Remove expired cache
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache {key}: {e}")
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache."""
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.warning(f"Failed to save cache {key}: {e}")
    
    def clear(self) -> None:
        """Clear all cache files."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove cache file {cache_file}: {e}")
    
    def clear_expired(self) -> int:
        """Clear expired cache files and return count of removed files."""
        removed_count = 0
        current_time = datetime.now().timestamp()
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                if current_time - cache_file.stat().st_mtime > self.ttl_seconds:
                    cache_file.unlink()
                    removed_count += 1
            except Exception as e:
                logger.warning(f"Failed to check/remove cache file {cache_file}: {e}")
        
        return removed_count


def get_default_profiler(cache_dir: Optional[str] = None) -> DataProfiler:
    """Get default data profiler instance."""
    return DataProfiler(cache_dir)


def get_default_cache(cache_dir: Optional[str] = None, ttl_hours: int = 24) -> DataCache:
    """Get default data cache instance."""
    return DataCache(cache_dir, ttl_hours)
