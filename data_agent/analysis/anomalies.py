"""
Anomaly detection capabilities including statistical outliers and ML-based detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
import warnings
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Comprehensive anomaly detection using multiple methods."""
    
    def __init__(self):
        """Initialize anomaly detector."""
        pass
    
    def detect_outliers(
        self, 
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        methods: List[str] = None,
        contamination: float = 0.1
    ) -> Dict[str, Any]:
        """
        Detect outliers using multiple statistical methods.
        
        Args:
            df: DataFrame to analyze
            columns: Columns to analyze (defaults to all numeric)
            methods: Detection methods to use
            contamination: Expected proportion of outliers
            
        Returns:
            Outlier detection results
        """
        if methods is None:
            methods = ['iqr', 'zscore', 'isolation_forest']
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Validate columns
        columns = [col for col in columns if col in df.columns]
        
        if not columns:
            return {"error": "No valid numeric columns for outlier detection"}
        
        try:
            results = {
                "analyzed_columns": columns,
                "methods_used": methods,
                "contamination": contamination,
                "outlier_summary": {},
                "detailed_results": {}
            }
            
            all_outliers = set()
            method_results = {}
            
            for method in methods:
                if method == 'iqr':
                    outliers = self._detect_outliers_iqr(df, columns)
                elif method == 'zscore':
                    outliers = self._detect_outliers_zscore(df, columns)
                elif method == 'isolation_forest':
                    outliers = self._detect_outliers_isolation_forest(df, columns, contamination)
                elif method == 'elliptic_envelope':
                    outliers = self._detect_outliers_elliptic_envelope(df, columns, contamination)
                else:
                    logger.warning(f"Unknown method: {method}")
                    continue
                
                method_results[method] = outliers
                all_outliers.update(outliers['outlier_indices'])
            
            # Combine results
            results["detailed_results"] = method_results
            results["outlier_summary"] = {
                "total_outliers": len(all_outliers),
                "outlier_percentage": (len(all_outliers) / len(df)) * 100,
                "outlier_indices": sorted(list(all_outliers)),
                "consensus_outliers": self._find_consensus_outliers(method_results, min_methods=2)
            }
            
            # Add sample outlier data
            if all_outliers:
                outlier_sample = df.iloc[sorted(list(all_outliers))[:10]]
                results["outlier_sample"] = outlier_sample.to_dict('records')
            
            return results
            
        except Exception as e:
            logger.error(f"Error in outlier detection: {e}")
            return {"error": str(e)}
    
    def detect_multivariate_anomalies(
        self, 
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        contamination: float = 0.1,
        method: str = 'isolation_forest'
    ) -> Dict[str, Any]:
        """
        Detect multivariate anomalies considering feature interactions.
        
        Args:
            df: DataFrame to analyze
            features: Features to use for detection
            contamination: Expected proportion of anomalies
            method: Detection method ('isolation_forest', 'elliptic_envelope')
            
        Returns:
            Multivariate anomaly detection results
        """
        try:
            if features is None:
                features = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Validate features
            features = [col for col in features if col in df.columns]
            
            if len(features) < 2:
                return {"error": "Need at least 2 features for multivariate anomaly detection"}
            
            # Prepare data
            data = df[features].dropna()
            
            if len(data) < 10:
                return {"error": "Insufficient data for multivariate anomaly detection"}
            
            # Scale the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            
            if method == 'isolation_forest':
                detector = IsolationForest(contamination=contamination, random_state=42)
            elif method == 'elliptic_envelope':
                detector = EllipticEnvelope(contamination=contamination, random_state=42)
            else:
                return {"error": f"Unknown method: {method}"}
            
            # Fit and predict
            predictions = detector.fit_predict(scaled_data)
            anomaly_scores = detector.score_samples(scaled_data)
            
            # Identify anomalies (prediction = -1)
            anomaly_mask = predictions == -1
            anomaly_indices = data.index[anomaly_mask].tolist()
            
            results = {
                "method": method,
                "features": features,
                "contamination": contamination,
                "total_samples": len(data),
                "anomalies_detected": int(np.sum(anomaly_mask)),
                "anomaly_percentage": float(np.sum(anomaly_mask) / len(data) * 100),
                "anomaly_indices": anomaly_indices,
                "anomaly_scores": {
                    "min_score": float(np.min(anomaly_scores)),
                    "max_score": float(np.max(anomaly_scores)),
                    "mean_score": float(np.mean(anomaly_scores)),
                    "threshold": float(np.percentile(anomaly_scores, (1-contamination)*100))
                }
            }
            
            # Add detailed anomaly information
            if len(anomaly_indices) > 0:
                anomaly_data = data.loc[anomaly_indices]
                results["anomaly_statistics"] = self._analyze_anomaly_characteristics(
                    data, anomaly_data, features
                )
                results["anomaly_sample"] = anomaly_data.head(10).to_dict('records')
            
            return results
            
        except Exception as e:
            logger.error(f"Error in multivariate anomaly detection: {e}")
            return {"error": str(e)}
    
    def detect_rule_based_anomalies(
        self, 
        df: pd.DataFrame,
        rules: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Detect anomalies based on business rules.
        
        Args:
            df: DataFrame to analyze
            rules: List of rule dictionaries
            
        Returns:
            Rule-based anomaly detection results
        """
        try:
            results = {
                "rules_applied": rules,
                "violations": [],
                "summary": {}
            }
            
            all_violations = set()
            
            for i, rule in enumerate(rules):
                rule_violations = self._apply_rule(df, rule, rule_id=i)
                results["violations"].append(rule_violations)
                all_violations.update(rule_violations["violation_indices"])
            
            results["summary"] = {
                "total_violations": len(all_violations),
                "violation_percentage": (len(all_violations) / len(df)) * 100,
                "violation_indices": sorted(list(all_violations))
            }
            
            # Add sample violation data
            if all_violations:
                violation_sample = df.iloc[sorted(list(all_violations))[:10]]
                results["violation_sample"] = violation_sample.to_dict('records')
            
            return results
            
        except Exception as e:
            logger.error(f"Error in rule-based anomaly detection: {e}")
            return {"error": str(e)}
    
    def detect_time_series_anomalies(
        self, 
        df: pd.DataFrame,
        date_column: str,
        value_column: str,
        method: str = 'statistical'
    ) -> Dict[str, Any]:
        """
        Detect anomalies in time series data.
        
        Args:
            df: DataFrame with time series data
            date_column: Date/datetime column
            value_column: Value column to analyze
            method: Detection method ('statistical', 'isolation_forest')
            
        Returns:
            Time series anomaly detection results
        """
        try:
            if date_column not in df.columns or value_column not in df.columns:
                return {"error": "Required columns not found"}
            
            # Prepare time series data
            df_copy = df.copy()
            if not pd.api.types.is_datetime64_any_dtype(df_copy[date_column]):
                df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors='coerce')
            
            # Remove invalid data
            clean_data = df_copy.dropna(subset=[date_column, value_column])
            
            if len(clean_data) < 10:
                return {"error": "Insufficient data for time series anomaly detection"}
            
            # Sort by date
            clean_data = clean_data.sort_values(date_column).reset_index(drop=True)
            
            if method == 'statistical':
                return self._detect_ts_anomalies_statistical(clean_data, date_column, value_column)
            elif method == 'isolation_forest':
                return self._detect_ts_anomalies_isolation_forest(clean_data, date_column, value_column)
            else:
                return {"error": f"Unknown method: {method}"}
                
        except Exception as e:
            logger.error(f"Error in time series anomaly detection: {e}")
            return {"error": str(e)}
    
    def _detect_outliers_iqr(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
        """Detect outliers using Interquartile Range method."""
        outlier_indices = set()
        column_results = {}
        
        for col in columns:
            data = df[col].dropna()
            if len(data) == 0:
                continue
                
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
                outlier_indices.update(col_outliers)
                
                column_results[col] = {
                    "outlier_count": len(col_outliers),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                    "outlier_indices": col_outliers
                }
        
        return {
            "method": "iqr",
            "outlier_indices": list(outlier_indices),
            "total_outliers": len(outlier_indices),
            "column_results": column_results
        }
    
    def _detect_outliers_zscore(self, df: pd.DataFrame, columns: List[str], threshold: float = 3.0) -> Dict[str, Any]:
        """Detect outliers using Z-score method."""
        outlier_indices = set()
        column_results = {}
        
        for col in columns:
            data = df[col].dropna()
            if len(data) == 0:
                continue
                
            z_scores = np.abs(stats.zscore(data))
            col_outliers = df.iloc[data.index[z_scores > threshold]].index.tolist()
            outlier_indices.update(col_outliers)
            
            column_results[col] = {
                "outlier_count": len(col_outliers),
                "threshold": threshold,
                "max_zscore": float(np.max(z_scores)),
                "outlier_indices": col_outliers
            }
        
        return {
            "method": "zscore",
            "outlier_indices": list(outlier_indices),
            "total_outliers": len(outlier_indices),
            "column_results": column_results
        }
    
    def _detect_outliers_isolation_forest(
        self, 
        df: pd.DataFrame, 
        columns: List[str], 
        contamination: float
    ) -> Dict[str, Any]:
        """Detect outliers using Isolation Forest."""
        data = df[columns].dropna()
        
        if len(data) < 10:
            return {
                "method": "isolation_forest",
                "outlier_indices": [],
                "total_outliers": 0,
                "error": "Insufficient data for Isolation Forest"
            }
        
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Apply Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        predictions = iso_forest.fit_predict(scaled_data)
        
        # Get outlier indices
        outlier_mask = predictions == -1
        outlier_indices = data.index[outlier_mask].tolist()
        
        return {
            "method": "isolation_forest",
            "outlier_indices": outlier_indices,
            "total_outliers": len(outlier_indices),
            "contamination": contamination
        }
    
    def _detect_outliers_elliptic_envelope(
        self, 
        df: pd.DataFrame, 
        columns: List[str], 
        contamination: float
    ) -> Dict[str, Any]:
        """Detect outliers using Elliptic Envelope."""
        data = df[columns].dropna()
        
        if len(data) < len(columns) + 1:
            return {
                "method": "elliptic_envelope",
                "outlier_indices": [],
                "total_outliers": 0,
                "error": "Insufficient data for Elliptic Envelope"
            }
        
        try:
            # Scale the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            
            # Apply Elliptic Envelope
            envelope = EllipticEnvelope(contamination=contamination, random_state=42)
            predictions = envelope.fit_predict(scaled_data)
            
            # Get outlier indices
            outlier_mask = predictions == -1
            outlier_indices = data.index[outlier_mask].tolist()
            
            return {
                "method": "elliptic_envelope",
                "outlier_indices": outlier_indices,
                "total_outliers": len(outlier_indices),
                "contamination": contamination
            }
            
        except Exception as e:
            return {
                "method": "elliptic_envelope",
                "outlier_indices": [],
                "total_outliers": 0,
                "error": str(e)
            }
    
    def _find_consensus_outliers(self, method_results: Dict[str, Any], min_methods: int = 2) -> List[int]:
        """Find outliers detected by multiple methods."""
        outlier_counts = {}
        
        for method, results in method_results.items():
            for idx in results.get("outlier_indices", []):
                outlier_counts[idx] = outlier_counts.get(idx, 0) + 1
        
        consensus_outliers = [idx for idx, count in outlier_counts.items() if count >= min_methods]
        return consensus_outliers
    
    def _analyze_anomaly_characteristics(
        self, 
        normal_data: pd.DataFrame,
        anomaly_data: pd.DataFrame,
        features: List[str]
    ) -> Dict[str, Any]:
        """Analyze characteristics of detected anomalies."""
        characteristics = {}
        
        for feature in features:
            if feature in normal_data.columns and feature in anomaly_data.columns:
                normal_stats = {
                    "mean": float(normal_data[feature].mean()),
                    "std": float(normal_data[feature].std()),
                    "median": float(normal_data[feature].median())
                }
                
                anomaly_stats = {
                    "mean": float(anomaly_data[feature].mean()),
                    "std": float(anomaly_data[feature].std()),
                    "median": float(anomaly_data[feature].median())
                }
                
                # Calculate deviations
                mean_deviation = abs(anomaly_stats["mean"] - normal_stats["mean"])
                std_deviation = abs(anomaly_stats["std"] - normal_stats["std"])
                
                characteristics[feature] = {
                    "normal": normal_stats,
                    "anomaly": anomaly_stats,
                    "mean_deviation": float(mean_deviation),
                    "std_deviation": float(std_deviation),
                    "is_extreme": mean_deviation > 2 * normal_stats["std"]
                }
        
        return characteristics
    
    def _apply_rule(self, df: pd.DataFrame, rule: Dict[str, Any], rule_id: int) -> Dict[str, Any]:
        """Apply a single business rule to detect violations."""
        try:
            rule_type = rule.get("type")
            column = rule.get("column")
            
            if column not in df.columns:
                return {
                    "rule_id": rule_id,
                    "rule": rule,
                    "violation_indices": [],
                    "violation_count": 0,
                    "error": f"Column {column} not found"
                }
            
            violations = []
            
            if rule_type == "range":
                min_val = rule.get("min")
                max_val = rule.get("max")
                if min_val is not None:
                    violations.extend(df[df[column] < min_val].index.tolist())
                if max_val is not None:
                    violations.extend(df[df[column] > max_val].index.tolist())
            
            elif rule_type == "categorical":
                valid_values = rule.get("valid_values", [])
                violations = df[~df[column].isin(valid_values)].index.tolist()
            
            elif rule_type == "null_check":
                if rule.get("allow_null", False):
                    violations = []
                else:
                    violations = df[df[column].isnull()].index.tolist()
            
            elif rule_type == "pattern":
                pattern = rule.get("pattern")
                if pattern and df[column].dtype == 'object':
                    violations = df[~df[column].astype(str).str.match(pattern, na=False)].index.tolist()
            
            elif rule_type == "custom":
                condition = rule.get("condition")
                if condition:
                    # This would require careful implementation to avoid security issues
                    # For now, we'll skip custom conditions
                    violations = []
            
            return {
                "rule_id": rule_id,
                "rule": rule,
                "violation_indices": violations,
                "violation_count": len(violations)
            }
            
        except Exception as e:
            return {
                "rule_id": rule_id,
                "rule": rule,
                "violation_indices": [],
                "violation_count": 0,
                "error": str(e)
            }
    
    def _detect_ts_anomalies_statistical(
        self, 
        df: pd.DataFrame, 
        date_col: str, 
        value_col: str
    ) -> Dict[str, Any]:
        """Detect time series anomalies using statistical methods."""
        
        # Calculate rolling statistics
        window_size = min(30, len(df) // 4) if len(df) > 30 else max(3, len(df) // 3)
        
        df_copy = df.copy()
        df_copy['rolling_mean'] = df_copy[value_col].rolling(window=window_size, center=True).mean()
        df_copy['rolling_std'] = df_copy[value_col].rolling(window=window_size, center=True).std()
        
        # Detect anomalies as points beyond 2 standard deviations from rolling mean
        df_copy['z_score'] = (df_copy[value_col] - df_copy['rolling_mean']) / df_copy['rolling_std']
        df_copy['is_anomaly'] = np.abs(df_copy['z_score']) > 2
        
        # Get anomaly information
        anomaly_mask = df_copy['is_anomaly'].fillna(False)
        anomaly_indices = df_copy.index[anomaly_mask].tolist()
        
        results = {
            "method": "statistical",
            "window_size": window_size,
            "anomaly_count": int(np.sum(anomaly_mask)),
            "anomaly_percentage": float(np.sum(anomaly_mask) / len(df) * 100),
            "anomaly_indices": anomaly_indices,
            "anomaly_timestamps": df_copy.loc[anomaly_mask, date_col].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            "statistics": {
                "mean_value": float(df_copy[value_col].mean()),
                "std_value": float(df_copy[value_col].std()),
                "max_z_score": float(np.max(np.abs(df_copy['z_score'].fillna(0))))
            }
        }
        
        if anomaly_indices:
            anomaly_data = df_copy.loc[anomaly_mask, [date_col, value_col, 'z_score']]
            results["anomaly_details"] = anomaly_data.head(10).to_dict('records')
        
        return results
    
    def _detect_ts_anomalies_isolation_forest(
        self, 
        df: pd.DataFrame, 
        date_col: str, 
        value_col: str
    ) -> Dict[str, Any]:
        """Detect time series anomalies using Isolation Forest with temporal features."""
        
        # Create temporal features
        df_copy = df.copy()
        df_copy['hour'] = df_copy[date_col].dt.hour
        df_copy['day_of_week'] = df_copy[date_col].dt.dayofweek
        df_copy['day_of_month'] = df_copy[date_col].dt.day
        df_copy['month'] = df_copy[date_col].dt.month
        
        # Create lag features
        for lag in [1, 7, 30]:
            if len(df_copy) > lag:
                df_copy[f'lag_{lag}'] = df_copy[value_col].shift(lag)
        
        # Create rolling features
        window_size = min(7, len(df) // 4) if len(df) > 7 else 3
        df_copy['rolling_mean'] = df_copy[value_col].rolling(window=window_size).mean()
        df_copy['rolling_std'] = df_copy[value_col].rolling(window=window_size).std()
        
        # Select features for anomaly detection
        feature_cols = [value_col, 'hour', 'day_of_week', 'day_of_month', 'month']
        
        # Add lag and rolling features if available
        for col in df_copy.columns:
            if col.startswith('lag_') or col.startswith('rolling_'):
                feature_cols.append(col)
        
        # Prepare data (remove rows with NaN values)
        analysis_data = df_copy[feature_cols].dropna()
        
        if len(analysis_data) < 10:
            return {"error": "Insufficient data for Isolation Forest analysis"}
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(analysis_data)
        
        # Apply Isolation Forest
        contamination = min(0.1, 0.05 + 0.05 * (len(analysis_data) > 100))
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        predictions = iso_forest.fit_predict(scaled_features)
        
        # Get anomaly information
        anomaly_mask = predictions == -1
        anomaly_indices = analysis_data.index[anomaly_mask].tolist()
        
        results = {
            "method": "isolation_forest",
            "features_used": feature_cols,
            "contamination": contamination,
            "anomaly_count": int(np.sum(anomaly_mask)),
            "anomaly_percentage": float(np.sum(anomaly_mask) / len(analysis_data) * 100),
            "anomaly_indices": anomaly_indices,
            "anomaly_timestamps": df_copy.loc[anomaly_indices, date_col].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
        }
        
        if anomaly_indices:
            anomaly_data = df_copy.loc[anomaly_indices, [date_col, value_col]]
            results["anomaly_details"] = anomaly_data.head(10).to_dict('records')
        
        return results


def get_default_anomaly_detector() -> AnomalyDetector:
    """Get default anomaly detector instance."""
    return AnomalyDetector()
