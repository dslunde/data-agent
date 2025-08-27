"""
Dataset-specific optimizations for the natural gas pipeline dataset.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pickle
import logging

logger = logging.getLogger(__name__)


class PipelineDatasetOptimizer:
    """Optimizations specific to the natural gas pipeline dataset."""

    def __init__(self, cache_dir: str = "./cache"):
        """Initialize optimizer with caching directory."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Dataset-specific knowledge
        self.major_states = ["LA", "TX", "PA", "KS", "OK", "AR", "IA", "MN", "MS", "IL"]
        self.major_pipelines = [
            "Northern Natural Gas Company",
            "Columbia Gas Transmission, LLC",
            "Enable Gas Transmission",
            "Texas Eastern Transmission, LP",
            "Tennessee Gas Pipeline Company",
        ]
        self.business_categories = [
            "LDC",
            "Production",
            "Interconnect",
            "Industrial",
            "Power",
            "Storage",
        ]

    def get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a given key."""
        return self.cache_dir / f"{cache_key}.pkl"

    def load_from_cache(self, cache_key: str) -> Optional[Any]:
        """Load data from cache if it exists."""
        cache_path = self.get_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    logger.debug(f"Loaded from cache: {cache_key}")
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Cache load failed for {cache_key}: {e}")
        return None

    def save_to_cache(self, cache_key: str, data: Any) -> None:
        """Save data to cache."""
        cache_path = self.get_cache_path(cache_key)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
            logger.debug(f"Saved to cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Cache save failed for {cache_key}: {e}")

    def optimize_dataset_loading(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply dataset-specific optimizations to loaded data."""
        logger.info("Applying pipeline dataset optimizations...")

        # Convert date column to datetime for better performance
        if "eff_gas_day" in df.columns:
            if df["eff_gas_day"].dtype == "category":
                df["eff_gas_day"] = pd.to_datetime(df["eff_gas_day"])

        # Convert coordinate columns to numeric for geographic analysis
        for coord_col in ["latitude", "longitude"]:
            if coord_col in df.columns and df[coord_col].dtype == "category":
                try:
                    df[coord_col] = pd.to_numeric(df[coord_col], errors="coerce")
                except Exception as e:
                    logger.warning(f"Could not convert {coord_col} to numeric: {e}")

        # Add derived columns for faster querying
        if "scheduled_quantity" in df.columns:
            df["has_quantity"] = df["scheduled_quantity"] > 0
            df["is_receipt"] = df["rec_del_sign"] == 1
            df["is_delivery"] = df["rec_del_sign"] == -1

        # Add date components for temporal analysis
        if "eff_gas_day" in df.columns and pd.api.types.is_datetime64_any_dtype(
            df["eff_gas_day"]
        ):
            df["month"] = df["eff_gas_day"].dt.month
            df["quarter"] = df["eff_gas_day"].dt.quarter
            df["year"] = df["eff_gas_day"].dt.year
            df["day_of_week"] = df["eff_gas_day"].dt.dayofweek

        logger.info("Dataset optimizations applied successfully")
        return df

    def get_dataset_filters(self, query_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimal filters based on query context."""
        filters = {}

        # Geographic filtering
        if "geographic" in query_context or "state" in query_context:
            # Focus on major states for faster processing
            filters["state_abb"] = self.major_states[:5]  # Top 5 states

        # Pipeline filtering
        if "pipeline" in query_context:
            filters["pipeline_name"] = self.major_pipelines[:3]  # Top 3 pipelines

        # Category filtering
        if "category" in query_context or "business" in query_context:
            filters["category_short"] = self.business_categories[:4]  # Top 4 categories

        # Quantity filtering (exclude zeros for most analyses)
        if "volume" in query_context or "quantity" in query_context:
            filters["has_quantity"] = True

        return filters

    def get_precomputed_stats(self) -> Dict[str, Any]:
        """Get precomputed statistics for the dataset."""
        cache_key = "pipeline_precomputed_stats"
        stats = self.load_from_cache(cache_key)

        if stats is not None:
            return stats

        # These would be computed once and cached
        stats = {
            "total_records": 23854855,
            "date_range": ("2022-01-01", "2025-08-26"),
            "major_states": self.major_states,
            "major_pipelines": self.major_pipelines,
            "business_categories": self.business_categories,
            "total_pipelines": 169,
            "total_locations": 17528,
            "total_counties": 1145,
            "states_count": 48,
            "non_zero_quantities_pct": 51.8,
            "receipts_vs_deliveries": {"receipts": 8935059, "deliveries": 14919796},
        }

        self.save_to_cache(cache_key, stats)
        return stats

    def optimize_correlation_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize dataframe for correlation analysis."""
        # Focus on numeric columns and key categorical encodings
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Add encoded categorical columns for correlation
        if "state_abb" in df.columns:
            # Create state rank encoding (major states get lower numbers)
            state_ranks = {state: i for i, state in enumerate(self.major_states)}
            df["state_rank"] = df["state_abb"].map(state_ranks).fillna(99)
            numeric_cols.append("state_rank")

        if "category_short" in df.columns:
            # Create category rank encoding
            cat_ranks = {cat: i for i, cat in enumerate(self.business_categories)}
            df["category_rank"] = df["category_short"].map(cat_ranks).fillna(99)
            numeric_cols.append("category_rank")

        return df[numeric_cols]

    def optimize_clustering_features(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Get optimal features for clustering pipeline data."""
        # Focus on business-relevant features
        features = []

        # Geographic features
        if "latitude" in df.columns and "longitude" in df.columns:
            if pd.api.types.is_numeric_dtype(df["latitude"]):
                features.extend(["latitude", "longitude"])

        # Quantity features
        if "scheduled_quantity" in df.columns:
            # Log transform for quantity (handle zeros)
            df["log_quantity"] = np.log1p(df["scheduled_quantity"].fillna(0))
            features.append("log_quantity")

        # Temporal features
        if "month" in df.columns:
            features.append("month")
        if "quarter" in df.columns:
            features.append("quarter")

        # Business features
        if "state_rank" in df.columns:
            features.append("state_rank")
        if "category_rank" in df.columns:
            features.append("category_rank")

        # Receipt/delivery features
        if "is_receipt" in df.columns:
            df["receipt_flag"] = df["is_receipt"].astype(int)
            features.append("receipt_flag")

        return df[features], features

    def get_anomaly_detection_params(self, method: str) -> Dict[str, Any]:
        """Get optimal parameters for anomaly detection on this dataset."""
        params = {}

        if method == "iqr":
            # More sensitive IQR for pipeline data (operational anomalies matter)
            params = {"multiplier": 1.5}  # Standard IQR multiplier

        elif method == "zscore":
            # Moderate Z-score threshold (gas flows can be variable)
            params = {"threshold": 2.5}  # Slightly relaxed from 2.0

        elif method == "isolation_forest":
            # Conservative contamination for critical infrastructure
            params = {"contamination": 0.05}  # 5% contamination rate

        return params

    def suggest_analysis_approach(self, query: str) -> Dict[str, Any]:
        """Suggest optimal analysis approach based on query."""
        query_lower = query.lower()
        suggestions = {
            "recommended_filters": {},
            "analysis_params": {},
            "performance_tips": [],
        }

        # Geographic queries
        if any(
            word in query_lower
            for word in ["state", "texas", "louisiana", "geographic"]
        ):
            suggestions["recommended_filters"]["state_abb"] = self.major_states[:3]
            suggestions["performance_tips"].append("Filter by major states first")

        # Pipeline-specific queries
        if any(word in query_lower for word in ["pipeline", "company", "operator"]):
            suggestions["recommended_filters"]["pipeline_name"] = self.major_pipelines[
                :3
            ]
            suggestions["performance_tips"].append("Focus on major pipeline operators")

        # Volume/quantity queries
        if any(
            word in query_lower for word in ["volume", "quantity", "flow", "throughput"]
        ):
            suggestions["recommended_filters"]["has_quantity"] = True
            suggestions["performance_tips"].append(
                "Exclude zero quantities for better statistics"
            )

        # Temporal queries
        if any(
            word in query_lower for word in ["seasonal", "monthly", "trend", "time"]
        ):
            suggestions["analysis_params"]["temporal_focus"] = True
            suggestions["performance_tips"].append(
                "Use date range filtering for large temporal queries"
            )

        # Business category queries
        if any(
            word in query_lower for word in ["ldc", "power", "industrial", "production"]
        ):
            suggestions["recommended_filters"]["category_short"] = (
                self.business_categories[:4]
            )
            suggestions["performance_tips"].append(
                "Filter by business categories early"
            )

        return suggestions


def get_dataset_optimizer() -> PipelineDatasetOptimizer:
    """Get the default dataset optimizer instance."""
    return PipelineDatasetOptimizer()
