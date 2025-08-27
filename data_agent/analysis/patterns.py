"""
Pattern recognition, clustering, and correlation analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


class PatternAnalyzer:
    """Pattern recognition and clustering analysis."""

    def __init__(self):
        """Initialize pattern analyzer."""
        pass

    def correlation_analysis(
        self, df: pd.DataFrame, method: str = "pearson", min_correlation: float = 0.3
    ) -> Dict[str, Any]:
        """
        Perform comprehensive correlation analysis.

        Args:
            df: DataFrame to analyze
            method: Correlation method ('pearson', 'spearman', 'kendall')
            min_correlation: Minimum correlation threshold for reporting

        Returns:
            Correlation analysis results
        """
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            if len(numeric_cols) < 2:
                return {
                    "error": "Need at least 2 numeric columns for correlation analysis"
                }

            # Calculate correlation matrix
            corr_matrix = df[numeric_cols].corr(method=method)

            # Find significant correlations
            significant_correlations = []
            strong_correlations = []

            for i, col1 in enumerate(corr_matrix.columns):
                for col2 in corr_matrix.columns[i + 1 :]:
                    corr_value = corr_matrix.loc[col1, col2]

                    if pd.isna(corr_value):
                        continue

                    abs_corr = abs(corr_value)

                    if abs_corr >= min_correlation:
                        correlation_info = {
                            "column1": col1,
                            "column2": col2,
                            "correlation": float(corr_value),
                            "abs_correlation": float(abs_corr),
                            "strength": self._classify_correlation_strength(abs_corr),
                            "direction": "positive" if corr_value > 0 else "negative",
                        }

                        significant_correlations.append(correlation_info)

                        if abs_corr > 0.7:
                            strong_correlations.append(correlation_info)

            # Sort by absolute correlation
            significant_correlations.sort(
                key=lambda x: x["abs_correlation"], reverse=True
            )
            strong_correlations.sort(key=lambda x: x["abs_correlation"], reverse=True)

            result = {
                "method": method,
                "numeric_columns": numeric_cols,
                "correlation_matrix": corr_matrix.round(3).to_dict(),
                "significant_correlations": significant_correlations,
                "strong_correlations": strong_correlations,
                "summary": {
                    "total_pairs": len(corr_matrix.columns)
                    * (len(corr_matrix.columns) - 1)
                    // 2,
                    "significant_count": len(significant_correlations),
                    "strong_count": len(strong_correlations),
                    "average_correlation": float(
                        np.abs(
                            corr_matrix.values[
                                np.triu_indices_from(corr_matrix.values, k=1)
                            ]
                        ).mean()
                    ),
                },
            }

            return result

        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            return {"error": str(e)}

    def clustering_analysis(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        n_clusters: Optional[int] = None,
        algorithm: str = "kmeans",
    ) -> Dict[str, Any]:
        """
        Perform clustering analysis.

        Args:
            df: DataFrame to cluster
            features: List of feature columns to use
            n_clusters: Number of clusters (auto-detected if None)
            algorithm: Clustering algorithm ('kmeans', 'dbscan')

        Returns:
            Clustering results
        """
        try:
            if features is None:
                features = df.select_dtypes(include=[np.number]).columns.tolist()

            # Validate features
            features = [col for col in features if col in df.columns]

            if len(features) < 2:
                return {"error": "Need at least 2 numeric features for clustering"}

            # Prepare data
            data = df[features].dropna()

            if len(data) < 10:
                return {"error": "Insufficient data points for clustering"}

            # Scale the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)

            if algorithm == "kmeans":
                return self._kmeans_clustering(data, scaled_data, features, n_clusters)
            elif algorithm == "dbscan":
                return self._dbscan_clustering(data, scaled_data, features)
            else:
                return {"error": f"Unknown clustering algorithm: {algorithm}"}

        except Exception as e:
            logger.error(f"Error in clustering analysis: {e}")
            return {"error": str(e)}

    def time_series_patterns(
        self,
        df: pd.DataFrame,
        date_column: str,
        value_columns: Optional[List[str]] = None,
        detect_seasonality: bool = True,
    ) -> Dict[str, Any]:
        """
        Detect patterns in time series data.

        Args:
            df: DataFrame with time series data
            date_column: Date/datetime column
            value_columns: Value columns to analyze
            detect_seasonality: Whether to detect seasonal patterns

        Returns:
            Time series pattern analysis
        """
        try:
            if date_column not in df.columns:
                return {"error": f"Date column {date_column} not found"}

            if value_columns is None:
                value_columns = df.select_dtypes(include=[np.number]).columns.tolist()

            # Validate value columns
            value_columns = [col for col in value_columns if col in df.columns]

            if not value_columns:
                return {"error": "No valid numeric columns for time series analysis"}

            # Convert to datetime
            df_copy = df.copy()
            if not pd.api.types.is_datetime64_any_dtype(df_copy[date_column]):
                df_copy[date_column] = pd.to_datetime(
                    df_copy[date_column], errors="coerce"
                )

            # Remove invalid dates
            df_clean = df_copy.dropna(subset=[date_column])

            if len(df_clean) == 0:
                return {"error": "No valid dates found"}

            # Sort by date
            df_clean = df_clean.sort_values(date_column)

            patterns = {}

            for col in value_columns:
                col_patterns = self._analyze_time_series_column(
                    df_clean, date_column, col, detect_seasonality
                )
                patterns[col] = col_patterns

            result = {
                "date_column": date_column,
                "value_columns": value_columns,
                "date_range": {
                    "start": df_clean[date_column].min().isoformat(),
                    "end": df_clean[date_column].max().isoformat(),
                    "span_days": (
                        df_clean[date_column].max() - df_clean[date_column].min()
                    ).days,
                },
                "patterns": patterns,
            }

            return result

        except Exception as e:
            logger.error(f"Error in time series pattern analysis: {e}")
            return {"error": str(e)}

    def association_rules(
        self,
        df: pd.DataFrame,
        categorical_columns: Optional[List[str]] = None,
        min_support: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Find association rules in categorical data.

        Args:
            df: DataFrame to analyze
            categorical_columns: Categorical columns to analyze
            min_support: Minimum support threshold

        Returns:
            Association rules
        """
        try:
            if categorical_columns is None:
                categorical_columns = df.select_dtypes(
                    include=["object", "category"]
                ).columns.tolist()

            # Validate columns
            categorical_columns = [
                col for col in categorical_columns if col in df.columns
            ]

            if len(categorical_columns) < 2:
                return {
                    "error": "Need at least 2 categorical columns for association analysis"
                }

            # Find frequent patterns
            associations = []

            for i, col1 in enumerate(categorical_columns):
                for col2 in categorical_columns[i + 1 :]:
                    association_info = self._calculate_association(
                        df, col1, col2, min_support
                    )
                    if association_info:
                        associations.append(association_info)

            # Sort by lift
            associations.sort(key=lambda x: x.get("lift", 0), reverse=True)

            result = {
                "categorical_columns": categorical_columns,
                "min_support": min_support,
                "associations": associations[:20],  # Top 20
                "total_associations": len(associations),
            }

            return result

        except Exception as e:
            logger.error(f"Error in association rule mining: {e}")
            return {"error": str(e)}

    def _classify_correlation_strength(self, abs_corr: float) -> str:
        """Classify correlation strength."""
        if abs_corr >= 0.9:
            return "very_strong"
        elif abs_corr >= 0.7:
            return "strong"
        elif abs_corr >= 0.5:
            return "moderate"
        elif abs_corr >= 0.3:
            return "weak"
        else:
            return "very_weak"

    def _kmeans_clustering(
        self,
        data: pd.DataFrame,
        scaled_data: np.ndarray,
        features: List[str],
        n_clusters: Optional[int],
    ) -> Dict[str, Any]:
        """Perform K-means clustering."""

        # Determine optimal number of clusters if not provided
        if n_clusters is None:
            n_clusters = self._find_optimal_clusters(
                scaled_data, max_k=min(10, len(data) // 2)
            )

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)

        # Calculate metrics
        if n_clusters > 1:
            silhouette_avg = silhouette_score(scaled_data, cluster_labels)
            inertia = kmeans.inertia_
        else:
            silhouette_avg = 0
            inertia = 0

        # Analyze clusters
        cluster_analysis = self._analyze_clusters(data, cluster_labels, features)

        result = {
            "algorithm": "kmeans",
            "n_clusters": n_clusters,
            "features": features,
            "cluster_assignments": cluster_labels.tolist(),  # Use standard API key
            "cluster_labels": cluster_labels.tolist(),  # Keep for backward compatibility
            "metrics": {
                "silhouette_score": float(silhouette_avg),
                "inertia": float(inertia),
            },
            "cluster_analysis": cluster_analysis,
            "cluster_centers": kmeans.cluster_centers_.tolist(),
        }

        return result

    def _dbscan_clustering(
        self, data: pd.DataFrame, scaled_data: np.ndarray, features: List[str]
    ) -> Dict[str, Any]:
        """Perform DBSCAN clustering."""

        # Use default parameters that work well for most datasets
        eps = 0.5
        min_samples = max(2, len(features))

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(scaled_data)

        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)

        # Calculate silhouette score if there are clusters
        if n_clusters > 1:
            # Exclude noise points from silhouette calculation
            non_noise_mask = cluster_labels != -1
            if np.sum(non_noise_mask) > 1:
                silhouette_avg = silhouette_score(
                    scaled_data[non_noise_mask], cluster_labels[non_noise_mask]
                )
            else:
                silhouette_avg = 0
        else:
            silhouette_avg = 0

        # Analyze clusters
        cluster_analysis = self._analyze_clusters(data, cluster_labels, features)

        result = {
            "algorithm": "dbscan",
            "parameters": {"eps": eps, "min_samples": min_samples},
            "features": features,
            "cluster_assignments": cluster_labels.tolist(),  # Use standard API key
            "cluster_labels": cluster_labels.tolist(),  # Keep for backward compatibility
            "metrics": {
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "silhouette_score": float(silhouette_avg),
            },
            "cluster_analysis": cluster_analysis,
        }

        return result

    def _find_optimal_clusters(self, data: np.ndarray, max_k: int = 10) -> int:
        """Find optimal number of clusters using elbow method."""
        if len(data) < 4:
            return 2

        max_k = min(max_k, len(data) - 1)
        inertias = []

        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)

        # Simple elbow detection
        if len(inertias) < 2:
            return 2

        # Calculate rate of change
        rates = []
        for i in range(1, len(inertias)):
            rate = inertias[i - 1] - inertias[i]
            rates.append(rate)

        # Find the elbow (where rate of improvement decreases most)
        if len(rates) < 2:
            return 2

        max_rate_decrease = 0
        best_k = 2

        for i in range(1, len(rates)):
            rate_decrease = rates[i - 1] - rates[i]
            if rate_decrease > max_rate_decrease:
                max_rate_decrease = rate_decrease
                best_k = i + 2  # +2 because we started from k=2

        return min(best_k, 6)  # Cap at 6 clusters for interpretability

    def _analyze_clusters(
        self, data: pd.DataFrame, labels: np.ndarray, features: List[str]
    ) -> Dict[str, Any]:
        """Analyze cluster characteristics."""
        cluster_stats = {}

        unique_labels = np.unique(labels)

        for label in unique_labels:
            cluster_data = data[labels == label]

            if label == -1:  # Noise points (DBSCAN)
                cluster_name = "noise"
            else:
                cluster_name = f"cluster_{label}"

            stats = {
                "size": len(cluster_data),
                "percentage": (len(cluster_data) / len(data)) * 100,
                "features": {},
            }

            # Calculate statistics for each feature
            for feature in features:
                if feature in cluster_data.columns:
                    feature_data = cluster_data[feature].dropna()
                    if len(feature_data) > 0:
                        stats["features"][feature] = {
                            "mean": float(feature_data.mean()),
                            "median": float(feature_data.median()),
                            "std": float(feature_data.std()),
                            "min": float(feature_data.min()),
                            "max": float(feature_data.max()),
                        }

            cluster_stats[cluster_name] = stats

        return cluster_stats

    def _analyze_time_series_column(
        self, df: pd.DataFrame, date_col: str, value_col: str, detect_seasonality: bool
    ) -> Dict[str, Any]:
        """Analyze patterns in a single time series column."""

        # Remove missing values
        clean_data = df[[date_col, value_col]].dropna()

        if len(clean_data) < 3:
            return {"error": "Insufficient data points"}

        patterns = {
            "data_points": len(clean_data),
            "trend": self._detect_trend(clean_data[value_col].values),
            "volatility": float(clean_data[value_col].std()),
            "range": {
                "min": float(clean_data[value_col].min()),
                "max": float(clean_data[value_col].max()),
                "span": float(
                    clean_data[value_col].max() - clean_data[value_col].min()
                ),
            },
        }

        # Detect seasonality if requested
        if (
            detect_seasonality and len(clean_data) >= 24
        ):  # Need enough data for seasonality
            seasonality = self._detect_seasonality(clean_data, date_col, value_col)
            patterns["seasonality"] = seasonality

        return patterns

    def _detect_trend(self, values: np.ndarray) -> Dict[str, Any]:
        """Detect trend in time series values."""
        if len(values) < 2:
            return {"direction": "insufficient_data", "strength": 0}

        # Linear regression to detect trend
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)

        # Classify trend
        if abs(slope) < std_err:
            direction = "flat"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"

        return {
            "direction": direction,
            "slope": float(slope),
            "strength": float(abs(r_value)),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
        }

    def _detect_seasonality(
        self, df: pd.DataFrame, date_col: str, value_col: str
    ) -> Dict[str, Any]:
        """Detect seasonal patterns (simplified version)."""

        # Extract time components
        df_copy = df.copy()
        df_copy["month"] = df_copy[date_col].dt.month
        df_copy["day_of_week"] = df_copy[date_col].dt.dayofweek
        df_copy["hour"] = (
            df_copy[date_col].dt.hour
            if df_copy[date_col].dt.hour.nunique() > 1
            else None
        )

        seasonality = {}

        # Monthly seasonality
        if df_copy["month"].nunique() > 1:
            monthly_avg = df_copy.groupby("month")[value_col].mean()
            monthly_var = monthly_avg.var()
            seasonality["monthly"] = {
                "variance": float(monthly_var),
                "pattern": monthly_avg.to_dict(),
                "has_pattern": monthly_var > df_copy[value_col].var() * 0.1,
            }

        # Weekly seasonality
        if df_copy["day_of_week"].nunique() > 1:
            weekly_avg = df_copy.groupby("day_of_week")[value_col].mean()
            weekly_var = weekly_avg.var()
            seasonality["weekly"] = {
                "variance": float(weekly_var),
                "pattern": weekly_avg.to_dict(),
                "has_pattern": weekly_var > df_copy[value_col].var() * 0.1,
            }

        # Hourly seasonality (if hourly data available)
        if df_copy["hour"].nunique() > 1:
            hourly_avg = df_copy.groupby("hour")[value_col].mean()
            hourly_var = hourly_avg.var()
            seasonality["hourly"] = {
                "variance": float(hourly_var),
                "pattern": hourly_avg.to_dict(),
                "has_pattern": hourly_var > df_copy[value_col].var() * 0.1,
            }

        return seasonality

    def _calculate_association(
        self, df: pd.DataFrame, col1: str, col2: str, min_support: float
    ) -> Optional[Dict[str, Any]]:
        """Calculate association between two categorical variables."""

        # Create contingency table
        try:
            contingency = pd.crosstab(df[col1], df[col2])

            # Find associations above minimum support
            total_records = len(df)
            associations = []

            for val1 in contingency.index:
                for val2 in contingency.columns:
                    count = contingency.loc[val1, val2]
                    support = count / total_records

                    if support >= min_support:
                        # Calculate confidence and lift
                        col1_support = (df[col1] == val1).sum() / total_records
                        col2_support = (df[col2] == val2).sum() / total_records

                        confidence = count / (df[col1] == val1).sum()
                        lift = (
                            support / (col1_support * col2_support)
                            if col1_support * col2_support > 0
                            else 0
                        )

                        associations.append(
                            {
                                "rule": f"{col1}={val1} → {col2}={val2}",
                                "support": float(support),
                                "confidence": float(confidence),
                                "lift": float(lift),
                                "count": int(count),
                            }
                        )

            if associations:
                # Return the best association (highest lift)
                best_association = max(associations, key=lambda x: x["lift"])
                best_association["column1"] = col1
                best_association["column2"] = col2
                best_association["all_rules"] = associations
                return best_association

            return None

        except Exception as e:
            logger.warning(
                f"Error calculating association between {col1} and {col2}: {e}"
            )
            return None


def get_default_pattern_analyzer() -> PatternAnalyzer:
    """Get default pattern analyzer instance."""
    return PatternAnalyzer()
