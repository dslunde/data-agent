"""
Pattern recognition, clustering, and correlation analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.utils import resample
from statsmodels.stats.multitest import multipletests
import warnings

from ..constants import ResponseKeys

logger = logging.getLogger(__name__)


class PatternAnalyzer:
    """Pattern recognition and clustering analysis."""

    def __init__(self):
        """Initialize pattern analyzer."""
        pass

    def _calculate_correlation_confidence_interval(self, r: float, n: int, confidence: float = 0.95) -> Dict[str, float]:
        """Calculate confidence interval for correlation coefficient using Fisher's z-transformation."""
        try:
            if abs(r) >= 1.0 or n < 3:
                return {"ci_lower": np.nan, "ci_upper": np.nan, "confidence_level": confidence}
            
            # Fisher's z-transformation
            z = 0.5 * np.log((1 + r) / (1 - r))
            
            # Standard error
            se_z = 1 / np.sqrt(n - 3)
            
            # Critical value for given confidence level
            alpha = 1 - confidence
            z_critical = stats.norm.ppf(1 - alpha / 2)
            
            # Confidence interval in z space
            z_lower = z - z_critical * se_z
            z_upper = z + z_critical * se_z
            
            # Transform back to correlation space
            r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
            r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
            
            return {
                "ci_lower": float(r_lower),
                "ci_upper": float(r_upper),
                "confidence_level": confidence
            }
            
        except Exception as e:
            logger.warning(f"Error calculating correlation confidence interval: {e}")
            return {"ci_lower": np.nan, "ci_upper": np.nan, "confidence_level": confidence}

    def _test_correlation_significance(self, data1: np.ndarray, data2: np.ndarray, method: str = "pearson") -> Dict[str, Any]:
        """Test statistical significance of correlation with proper statistical testing."""
        result = {"method": method}
        
        try:
            # Remove pairs with missing values
            mask = ~(np.isnan(data1) | np.isnan(data2))
            clean_data1 = data1[mask]
            clean_data2 = data2[mask]
            
            n = len(clean_data1)
            if n < 3:
                return {"error": "Insufficient data points for correlation test", "n": n}
            
            # Calculate correlation and p-value based on method
            if method == "pearson":
                r, p_value = pearsonr(clean_data1, clean_data2)
            elif method == "spearman":
                r, p_value = spearmanr(clean_data1, clean_data2)
            elif method == "kendall":
                r, p_value = kendalltau(clean_data1, clean_data2)
            else:
                return {"error": f"Unknown correlation method: {method}"}
            
            # Calculate confidence interval
            ci = self._calculate_correlation_confidence_interval(r, n)
            
            # Interpret correlation strength
            abs_r = abs(r)
            if abs_r >= 0.9:
                strength = "very_strong"
            elif abs_r >= 0.7:
                strength = "strong"
            elif abs_r >= 0.5:
                strength = "moderate"
            elif abs_r >= 0.3:
                strength = "weak"
            else:
                strength = "negligible"
            
            result.update({
                "correlation": float(r),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "n": n,
                "strength": strength,
                "confidence_interval": ci,
                "test_statistic": f"{method}_r = {r:.3f}"
            })
            
        except Exception as e:
            logger.error(f"Error in correlation significance test: {e}")
            result["error"] = str(e)
        
        return result

    def _enhanced_clustering_validation(self, data: np.ndarray, labels: np.ndarray, 
                                      algorithm: str = "kmeans") -> Dict[str, Any]:
        """Enhanced clustering validation with multiple metrics."""
        validation = {
            "algorithm": algorithm,
            "n_samples": len(data),
            "n_features": data.shape[1] if len(data.shape) > 1 else 1
        }
        
        try:
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)
            n_noise = np.sum(labels == -1) if -1 in unique_labels else 0
            
            validation.update({
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_ratio": n_noise / len(labels)
            })
            
            # Only calculate metrics if we have clusters
            if n_clusters > 1 and n_clusters < len(data):
                # Silhouette Score
                try:
                    if n_noise > 0:
                        # For DBSCAN, exclude noise points
                        non_noise_mask = labels != -1
                        if np.sum(non_noise_mask) > 1:
                            sil_score = silhouette_score(
                                data[non_noise_mask], 
                                labels[non_noise_mask]
                            )
                        else:
                            sil_score = -1
                    else:
                        sil_score = silhouette_score(data, labels)
                    
                    validation["silhouette_score"] = float(sil_score)
                    
                    # Interpret silhouette score
                    if sil_score > 0.7:
                        validation["silhouette_interpretation"] = "strong"
                    elif sil_score > 0.5:
                        validation["silhouette_interpretation"] = "reasonable"
                    elif sil_score > 0.25:
                        validation["silhouette_interpretation"] = "weak"
                    else:
                        validation["silhouette_interpretation"] = "poor"
                        
                except Exception as e:
                    logger.warning(f"Error calculating silhouette score: {e}")
                    validation["silhouette_score"] = None
                
                # Calinski-Harabasz Score (Variance Ratio Criterion)
                try:
                    from sklearn.metrics import calinski_harabasz_score
                    if n_noise == 0:  # Only for non-noisy clusters
                        ch_score = calinski_harabasz_score(data, labels)
                        validation["calinski_harabasz_score"] = float(ch_score)
                except Exception as e:
                    logger.warning(f"Error calculating Calinski-Harabasz score: {e}")
                
                # Davies-Bouldin Score
                try:
                    from sklearn.metrics import davies_bouldin_score
                    if n_noise == 0:  # Only for non-noisy clusters
                        db_score = davies_bouldin_score(data, labels)
                        validation["davies_bouldin_score"] = float(db_score)
                        # Lower values are better for DB score
                        if db_score < 1.0:
                            validation["davies_bouldin_interpretation"] = "good"
                        elif db_score < 2.0:
                            validation["davies_bouldin_interpretation"] = "acceptable"
                        else:
                            validation["davies_bouldin_interpretation"] = "poor"
                except Exception as e:
                    logger.warning(f"Error calculating Davies-Bouldin score: {e}")
                
                # Inertia (for K-means)
                if algorithm.lower() == "kmeans":
                    try:
                        # Calculate within-cluster sum of squares
                        inertia = 0
                        for cluster_id in unique_labels:
                            if cluster_id != -1:
                                cluster_points = data[labels == cluster_id]
                                if len(cluster_points) > 0:
                                    centroid = np.mean(cluster_points, axis=0)
                                    inertia += np.sum((cluster_points - centroid) ** 2)
                        validation["inertia"] = float(inertia)
                    except Exception as e:
                        logger.warning(f"Error calculating inertia: {e}")
            
            # Overall clustering quality assessment
            if "silhouette_score" in validation:
                sil = validation["silhouette_score"]
                if sil is not None:
                    if sil > 0.5:
                        validation["overall_quality"] = "good"
                    elif sil > 0.25:
                        validation["overall_quality"] = "fair"
                    else:
                        validation["overall_quality"] = "poor"
            
        except Exception as e:
            logger.error(f"Error in clustering validation: {e}")
            validation["error"] = str(e)
        
        return validation

    def _calculate_detailed_correlation(self, data1: pd.Series, data2: pd.Series, 
                                      col1: str, col2: str, method: str = "pearson") -> Dict[str, Any]:
        """Calculate detailed correlation statistics with confidence intervals and significance tests."""
        try:
            # Remove missing values
            clean_data = pd.DataFrame({col1: data1, col2: data2}).dropna()
            if len(clean_data) < 3:
                return {"error": "Insufficient data points after removing missing values"}
            
            arr1 = clean_data[col1].values
            arr2 = clean_data[col2].values
            
            # Calculate correlation with significance test
            correlation_result = self._test_correlation_significance(arr1, arr2, method)
            
            # Add column names
            correlation_result.update({
                "variable1": col1,
                "variable2": col2,
                "pair": f"{col1} vs {col2}"
            })
            
            return correlation_result
            
        except Exception as e:
            logger.error(f"Error calculating detailed correlation for {col1} vs {col2}: {e}")
            return {"error": str(e), "variable1": col1, "variable2": col2}

    def _apply_correlation_multiple_testing_correction(self, correlations: List[Dict[str, Any]], 
                                                     method: str = "fdr_bh") -> List[Dict[str, Any]]:
        """Apply multiple testing correction to correlation p-values."""
        try:
            if not correlations or len(correlations) <= 1:
                return correlations
            
            # Extract p-values
            p_values = [corr.get("p_value", 1.0) for corr in correlations if corr.get("p_value") is not None]
            
            if not p_values:
                return correlations
            
            # Apply correction
            rejected, corrected_pvals, alpha_sidak, alpha_bonf = multipletests(
                p_values, alpha=0.05, method=method
            )
            
            # Update correlations with corrected p-values
            corrected_correlations = []
            p_idx = 0
            
            for corr in correlations:
                if corr.get("p_value") is not None and p_idx < len(corrected_pvals):
                    updated_corr = corr.copy()
                    updated_corr.update({
                        "p_value_corrected": float(corrected_pvals[p_idx]),
                        "significant_corrected": bool(rejected[p_idx]),
                        "multiple_testing_method": method
                    })
                    corrected_correlations.append(updated_corr)
                    p_idx += 1
                else:
                    corrected_correlations.append(corr)
            
            return corrected_correlations
            
        except Exception as e:
            logger.error(f"Error applying multiple testing correction to correlations: {e}")
            return correlations

    def _calculate_categorical_association(self, data1: pd.Series, data2: pd.Series) -> Dict[str, Any]:
        """Calculate association for two categorical variables using Chi-square test."""
        try:
            contingency_table = pd.crosstab(data1, data2)
            chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
            
            # Calculate Cramér's V for effect size
            n = contingency_table.sum().sum()
            phi2 = chi2 / n
            r, k = contingency_table.shape
            cramers_v = np.sqrt(phi2 / min(k-1, r-1))

            return {
                "test": "chi_square",
                "chi2_statistic": float(chi2),
                "p_value": float(p),
                "degrees_of_freedom": int(dof),
                "cramers_v": float(cramers_v),
                "significant": p < 0.05
            }
        except Exception as e:
            logger.warning(f"Could not calculate categorical association: {e}")
            return {"error": str(e)}

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
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

            all_associations = []

            # Numeric vs Numeric
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i + 1 :]:
                    correlation_info = self._calculate_detailed_correlation(
                        df[col1], df[col2], col1, col2, method
                    )
                    if correlation_info and not correlation_info.get("error"):
                        all_associations.append(correlation_info)

            # Categorical vs Categorical
            for i, col1 in enumerate(categorical_cols):
                for col2 in categorical_cols[i + 1 :]:
                    association_info = self._calculate_categorical_association(df[col1], df[col2])
                    if association_info and not association_info.get("error"):
                        association_info.update({"variable1": col1, "variable2": col2})
                        all_associations.append(association_info)
            
            # Numeric vs Categorical (ANOVA)
            for num_col in numeric_cols:
                for cat_col in categorical_cols:
                    try:
                        groups = [df[num_col][df[cat_col] == cat] for cat in df[cat_col].unique() if pd.notna(cat)]
                        if len(groups) > 1:
                            f_val, p_val = stats.f_oneway(*groups)
                            if p_val < 0.05:
                                all_associations.append({
                                    "test": "anova",
                                    "variable1": num_col,
                                    "variable2": cat_col,
                                    "f_statistic": float(f_val),
                                    "p_value": float(p_val),
                                    "significant": True
                                })
                    except Exception as e:
                        logger.warning(f"ANOVA for {num_col} vs {cat_col} failed: {e}")


            significant_correlations = [assoc for assoc in all_associations if assoc.get("significant")]
            strong_correlations = [assoc for assoc in significant_correlations if assoc.get("abs_correlation", 0) > 0.7 or assoc.get("cramers_v", 0) > 0.5]


            result = {
                "method": method,
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "significant_associations": significant_correlations,
                "strong_associations": strong_correlations,
                "summary": {
                    "total_pairs_analyzed": len(all_associations),
                    "significant_count": len(significant_correlations),
                    "strong_count": len(strong_correlations),
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
        eps: float = 0.5,
        min_samples: Optional[int] = None,
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
                return self._dbscan_clustering(data, scaled_data, features, eps, min_samples)
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

        # Enhanced clustering validation with multiple metrics
        validation_results = self._enhanced_clustering_validation(
            scaled_data, cluster_labels, "kmeans"
        )
        
        # Legacy metrics for backwards compatibility
        silhouette_avg = validation_results.get("silhouette_score", 0)
        inertia = validation_results.get("inertia", kmeans.inertia_)

        # Analyze clusters
        cluster_analysis = self._analyze_clusters(data, cluster_labels, features)

        result = {
            "algorithm": "kmeans",
            "n_clusters": n_clusters,
            "features": features,
            ResponseKeys.CLUSTER_ASSIGNMENTS: cluster_labels.tolist(),  # Use standard API key
            ResponseKeys.CLUSTER_LABELS: cluster_labels.tolist(),  # Keep for backward compatibility
            "metrics": {
                "silhouette_score": float(silhouette_avg),
                "inertia": float(inertia),
            },
            "cluster_analysis": cluster_analysis,
            "cluster_centers": kmeans.cluster_centers_.tolist(),
            "validation": validation_results,  # Enhanced validation metrics
        }

        return result

    def _dbscan_clustering(
        self, data: pd.DataFrame, scaled_data: np.ndarray, features: List[str],
        eps: float = 0.5, min_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """Perform DBSCAN clustering."""

        # Use provided parameters or sensible defaults
        if min_samples is None:
            min_samples = max(2, len(features))

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(scaled_data)

        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)

        # Enhanced clustering validation with multiple metrics
        validation_results = self._enhanced_clustering_validation(
            scaled_data, cluster_labels, "dbscan"
        )
        
        # Legacy metrics for backwards compatibility
        silhouette_avg = validation_results.get("silhouette_score", 0)

        # Analyze clusters
        cluster_analysis = self._analyze_clusters(data, cluster_labels, features)

        result = {
            "algorithm": "dbscan",
            "parameters": {"eps": eps, "min_samples": min_samples},
            "features": features,
            "n_clusters": n_clusters,  # Add at top level for consistency
            ResponseKeys.CLUSTER_ASSIGNMENTS: cluster_labels.tolist(),  # Use standard API key
            ResponseKeys.CLUSTER_LABELS: cluster_labels.tolist(),  # Keep for backward compatibility
            "metrics": {
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "silhouette_score": float(silhouette_avg),
            },
            "cluster_analysis": cluster_analysis,
            "validation": validation_results,  # Enhanced validation metrics
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

    def _calculate_detailed_correlation(self, x: pd.Series, y: pd.Series, 
                                      col1: str, col2: str, method: str) -> Dict[str, Any]:
        """
        Calculate detailed correlation statistics with confidence intervals and significance testing.
        
        Args:
            x: First variable series
            y: Second variable series
            col1: Name of first column
            col2: Name of second column
            method: Correlation method
            
        Returns:
            Detailed correlation statistics
        """
        try:
            # Remove missing values
            valid_data = pd.DataFrame({"x": x, "y": y}).dropna()
            
            if len(valid_data) < 3:
                return {"error": "Insufficient data for correlation"}
            
            x_clean = valid_data["x"]
            y_clean = valid_data["y"]
            
            # Calculate correlation and p-value using appropriate method
            if method == "pearson":
                corr_coef, p_value = pearsonr(x_clean, y_clean)
            elif method == "spearman":
                corr_coef, p_value = spearmanr(x_clean, y_clean)
            elif method == "kendall":
                corr_coef, p_value = kendalltau(x_clean, y_clean)
            else:
                return {"error": f"Unknown correlation method: {method}"}
            
            # Calculate confidence interval (for Pearson, use Fisher's z-transformation)
            confidence_interval = self._calculate_correlation_confidence_interval(
                corr_coef, len(valid_data), method
            )
            
            # Calculate effect size interpretation
            abs_corr = abs(corr_coef)
            strength = self._classify_correlation_strength(abs_corr)
            
            result = {
                "column1": col1,
                "column2": col2,
                "correlation": float(corr_coef),
                "abs_correlation": float(abs_corr),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "strength": strength,
                "direction": "positive" if corr_coef > 0 else "negative",
                "sample_size": len(valid_data),
                "confidence_interval": confidence_interval,
                "method": method
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating correlation between {col1} and {col2}: {e}")
            return {"error": str(e)}
    
    def _calculate_correlation_confidence_interval(self, corr: float, n: int, method: str) -> Dict[str, Any]:
        """
        Calculate confidence interval for correlation coefficient.
        
        Args:
            corr: Correlation coefficient
            n: Sample size
            method: Correlation method
            
        Returns:
            Confidence interval information
        """
        try:
            if method == "pearson" and n > 3:
                # Fisher's z-transformation for Pearson correlation
                z = 0.5 * np.log((1 + corr) / (1 - corr))
                se_z = 1 / np.sqrt(n - 3)
                z_critical = 1.96  # 95% CI
                
                z_lower = z - z_critical * se_z
                z_upper = z + z_critical * se_z
                
                # Transform back to correlation scale
                ci_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
                ci_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
                
                return {
                    "lower": float(ci_lower),
                    "upper": float(ci_upper),
                    "confidence_level": 0.95,
                    "method": "Fisher's z-transformation"
                }
            else:
                # Bootstrap confidence interval for non-parametric methods
                return self._bootstrap_correlation_ci(corr, n, method)
                
        except Exception as e:
            logger.error(f"Error calculating correlation CI: {e}")
            return {"error": str(e)}
    
    def _bootstrap_correlation_ci(self, corr: float, n: int, method: str) -> Dict[str, Any]:
        """
        Calculate bootstrap confidence interval for correlation.
        
        Args:
            corr: Original correlation
            n: Sample size
            method: Correlation method
            
        Returns:
            Bootstrap confidence interval
        """
        try:
            # For bootstrap, we need a rough estimate
            # Using Fisher's transformation as approximation for all methods
            if n > 10:
                se_approx = 1 / np.sqrt(n - 3) if method == "pearson" else 1.2 / np.sqrt(n - 3)
                margin_error = 1.96 * se_approx
                
                ci_lower = max(-1, corr - margin_error)
                ci_upper = min(1, corr + margin_error)
                
                return {
                    "lower": float(ci_lower),
                    "upper": float(ci_upper),
                    "confidence_level": 0.95,
                    "method": f"approximate bootstrap for {method}"
                }
            else:
                return {
                    "lower": float(corr),
                    "upper": float(corr),
                    "confidence_level": 0.95,
                    "method": "insufficient_data_for_ci"
                }
                
        except Exception as e:
            return {"error": str(e)}
    
    def _apply_correlation_multiple_testing_correction(self, correlations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply multiple testing correction to correlation p-values.
        
        Args:
            correlations: List of correlation dictionaries
            
        Returns:
            Updated correlations with corrected p-values
        """
        try:
            if not correlations:
                return correlations
            
            # Extract p-values
            p_values = []
            for corr in correlations:
                if "p_value" in corr and not corr.get("error"):
                    p_values.append(corr["p_value"])
                else:
                    p_values.append(1.0)  # Non-significant for missing/error cases
            
            if not p_values:
                return correlations
            
            # Apply Benjamini-Hochberg FDR correction
            rejected, corrected_p_values, _, _ = multipletests(
                p_values, alpha=0.05, method="fdr_bh"
            )
            
            # Update correlation results
            for i, corr in enumerate(correlations):
                if i < len(corrected_p_values):
                    corr["p_value_corrected"] = float(corrected_p_values[i])
                    corr["significant_corrected"] = bool(rejected[i])
                    corr["multiple_testing_method"] = "Benjamini-Hochberg FDR"
            
            return correlations
            
        except Exception as e:
            logger.error(f"Error applying multiple testing correction to correlations: {e}")
            return correlations


def get_default_pattern_analyzer() -> PatternAnalyzer:
    """Get default pattern analyzer instance."""
    return PatternAnalyzer()
