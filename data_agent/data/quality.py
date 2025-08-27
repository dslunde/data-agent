"""
Data quality assessment and preprocessing functionality.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class DataQualityAssessor:
    """Assesses and reports data quality issues."""

    def __init__(self):
        """Initialize quality assessor."""
        pass

    def assess_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive data quality assessment.

        Args:
            df: DataFrame to assess

        Returns:
            Dictionary containing quality assessment results
        """
        assessment = {
            "overview": self._get_overview(df),
            "completeness": self._assess_completeness(df),
            "consistency": self._assess_consistency(df),
            "validity": self._assess_validity(df),
            "uniqueness": self._assess_uniqueness(df),
            "accuracy": self._assess_accuracy(df),
            "issues": self._identify_issues(df),
            "recommendations": [],
        }

        # Generate recommendations based on findings
        assessment["recommendations"] = self._generate_recommendations(assessment)

        return assessment

    def _get_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic overview of the dataset."""
        return {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
            "text_columns": len(df.select_dtypes(include=["object"]).columns),
            "datetime_columns": len(df.select_dtypes(include=["datetime64"]).columns),
        }

    def _assess_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data completeness (missing values)."""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()

        missing_by_column = df.isnull().sum()
        missing_pct_by_column = (missing_by_column / len(df) * 100).round(2)

        completeness_score = ((total_cells - missing_cells) / total_cells * 100).round(
            2
        )

        return {
            "completeness_score": completeness_score,
            "total_missing_cells": int(missing_cells),
            "missing_percentage": round((missing_cells / total_cells) * 100, 2),
            "columns_with_missing": missing_by_column[missing_by_column > 0].to_dict(),
            "missing_percentage_by_column": missing_pct_by_column[
                missing_pct_by_column > 0
            ].to_dict(),
            "columns_mostly_missing": missing_pct_by_column[
                missing_pct_by_column > 80
            ].index.tolist(),
        }

    def _assess_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data consistency issues."""
        issues = []

        # Check for inconsistent data types within columns
        for col in df.select_dtypes(include=["object"]).columns:
            sample = df[col].dropna().astype(str)
            if len(sample) > 0:
                # Check for mixed numeric/text
                numeric_count = sum(sample.str.match(r"^-?\d+\.?\d*$").fillna(False))
                text_count = len(sample) - numeric_count

                if numeric_count > 0 and text_count > 0:
                    issues.append(
                        {
                            "column": col,
                            "type": "mixed_types",
                            "description": f"Column has {numeric_count} numeric and {text_count} text values",
                        }
                    )

        # Check for inconsistent formats in potential date columns
        for col in df.columns:
            if df[col].dtype == "object":
                sample = df[col].dropna().astype(str).head(100)
                if self._looks_like_dates(sample):
                    formats = self._detect_date_formats(sample)
                    if len(formats) > 1:
                        issues.append(
                            {
                                "column": col,
                                "type": "inconsistent_date_formats",
                                "description": f"Multiple date formats detected: {formats}",
                            }
                        )

        # Check for case inconsistency
        for col in df.select_dtypes(include=["object"]).columns:
            sample = df[col].dropna().head(1000)
            if len(sample) > 0:
                case_variants = {}
                for value in sample:
                    lower_val = str(value).lower()
                    if lower_val not in case_variants:
                        case_variants[lower_val] = set()
                    case_variants[lower_val].add(str(value))

                inconsistent_cases = {
                    k: list(v) for k, v in case_variants.items() if len(v) > 1
                }
                if (
                    inconsistent_cases
                    and len(inconsistent_cases) < len(case_variants) * 0.5
                ):
                    issues.append(
                        {
                            "column": col,
                            "type": "case_inconsistency",
                            "description": f"Inconsistent casing in {len(inconsistent_cases)} values",
                        }
                    )

        return {
            "consistency_issues": issues,
            "consistency_score": max(0, 100 - len(issues) * 10),  # Simple scoring
        }

    def _assess_validity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data validity (outliers, invalid ranges)."""
        validity_issues = []

        # Check numeric columns for extreme outliers
        for col in df.select_dtypes(include=[np.number]).columns:
            data = df[col].dropna()
            if len(data) > 0:
                Q1, Q3 = data.quantile([0.25, 0.75])
                IQR = Q3 - Q1

                if IQR > 0:
                    lower_bound = Q1 - 3 * IQR
                    upper_bound = Q3 + 3 * IQR

                    outliers = data[(data < lower_bound) | (data > upper_bound)]
                    if len(outliers) > 0:
                        validity_issues.append(
                            {
                                "column": col,
                                "type": "extreme_outliers",
                                "count": len(outliers),
                                "percentage": round(len(outliers) / len(data) * 100, 2),
                                "description": f"{len(outliers)} extreme outliers detected",
                            }
                        )

        # Check for impossible values (e.g., negative ages, future dates)
        for col in df.columns:
            col_lower = col.lower()

            # Age-like columns shouldn't be negative or > 150
            if any(keyword in col_lower for keyword in ["age", "years_old"]):
                if df[col].dtype in [np.number]:
                    invalid_ages = df[(df[col] < 0) | (df[col] > 150)]
                    if len(invalid_ages) > 0:
                        validity_issues.append(
                            {
                                "column": col,
                                "type": "invalid_range",
                                "count": len(invalid_ages),
                                "description": "Invalid age values (< 0 or > 150)",
                            }
                        )

            # Date columns shouldn't have future dates if they represent past events
            if "datetime" in str(df[col].dtype):
                future_dates = df[df[col] > pd.Timestamp.now()]
                if (
                    len(future_dates) > 0
                    and "created" in col_lower
                    or "born" in col_lower
                ):
                    validity_issues.append(
                        {
                            "column": col,
                            "type": "future_dates",
                            "count": len(future_dates),
                            "description": "Future dates in historical data",
                        }
                    )

        return {
            "validity_issues": validity_issues,
            "validity_score": max(0, 100 - len(validity_issues) * 15),
        }

    def _assess_uniqueness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data uniqueness issues."""
        duplicate_rows = df.duplicated().sum()
        duplicate_percentage = (duplicate_rows / len(df) * 100).round(2)

        # Check for duplicate columns (same name or same content)
        duplicate_columns = []
        for i, col1 in enumerate(df.columns):
            for col2 in df.columns[i + 1 :]:
                if df[col1].equals(df[col2]):
                    duplicate_columns.append((col1, col2))

        # Check uniqueness of potential ID columns
        id_column_issues = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ["id", "key", "index", "uuid"]):
                unique_count = df[col].nunique()
                total_count = df[col].count()  # Non-null count

                if unique_count < total_count:
                    id_column_issues.append(
                        {
                            "column": col,
                            "unique_count": unique_count,
                            "total_count": total_count,
                            "description": "ID column has non-unique values",
                        }
                    )

        uniqueness_score = 100 - duplicate_percentage - len(duplicate_columns) * 10
        uniqueness_score = max(0, uniqueness_score)

        return {
            "duplicate_rows": int(duplicate_rows),
            "duplicate_percentage": duplicate_percentage,
            "duplicate_columns": duplicate_columns,
            "id_column_issues": id_column_issues,
            "uniqueness_score": uniqueness_score,
        }

    def _assess_accuracy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess potential accuracy issues."""
        accuracy_issues = []

        # Check for suspicious patterns in text data
        for col in df.select_dtypes(include=["object"]).columns:
            sample = df[col].dropna().head(1000)
            if len(sample) > 0:
                # Check for too many 'unknown', 'null', 'n/a' values
                placeholder_values = [
                    "unknown",
                    "null",
                    "n/a",
                    "na",
                    "none",
                    "",
                    "missing",
                ]
                placeholder_count = sum(
                    sample.astype(str).str.lower().isin(placeholder_values)
                )

                if placeholder_count > len(sample) * 0.1:  # More than 10%
                    accuracy_issues.append(
                        {
                            "column": col,
                            "type": "placeholder_values",
                            "count": placeholder_count,
                            "percentage": round(
                                placeholder_count / len(sample) * 100, 2
                            ),
                            "description": "High number of placeholder values",
                        }
                    )

        # Check for unrealistic correlations (might indicate data errors)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            correlation_matrix = df[numeric_cols].corr()

            # Find suspiciously perfect correlations (might be duplicated data)
            perfect_correlations = []
            for i, col1 in enumerate(correlation_matrix.columns):
                for col2 in correlation_matrix.columns[i + 1 :]:
                    corr_value = correlation_matrix.loc[col1, col2]
                    if abs(corr_value) > 0.99 and not pd.isna(corr_value):
                        perfect_correlations.append((col1, col2, corr_value))

            if perfect_correlations:
                accuracy_issues.append(
                    {
                        "type": "perfect_correlations",
                        "correlations": perfect_correlations,
                        "description": "Suspiciously perfect correlations detected",
                    }
                )

        accuracy_score = max(0, 100 - len(accuracy_issues) * 20)

        return {"accuracy_issues": accuracy_issues, "accuracy_score": accuracy_score}

    def _identify_issues(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify high-priority data quality issues."""
        issues = []

        # Critical issues that affect analysis

        # 1. Columns with >90% missing data
        missing_pct = df.isnull().sum() / len(df) * 100
        critical_missing = missing_pct[missing_pct > 90]
        for col in critical_missing.index:
            issues.append(
                {
                    "severity": "critical",
                    "type": "mostly_missing",
                    "column": col,
                    "description": f"Column '{col}' is {missing_pct[col]:.1f}% missing",
                    "impact": "May not be useful for analysis",
                }
            )

        # 2. All rows are duplicates
        if df.duplicated().all():
            issues.append(
                {
                    "severity": "critical",
                    "type": "all_duplicates",
                    "description": "All rows in dataset are duplicates",
                    "impact": "No unique information available",
                }
            )

        # 3. Single-value columns
        for col in df.columns:
            if df[col].nunique() <= 1:
                issues.append(
                    {
                        "severity": "high",
                        "type": "constant_column",
                        "column": col,
                        "description": f"Column '{col}' has only one unique value",
                        "impact": "Provides no discriminatory information",
                    }
                )

        # 4. Memory usage issues
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        if memory_mb > 1000:  # > 1GB
            issues.append(
                {
                    "severity": "medium",
                    "type": "large_memory",
                    "description": f"Dataset uses {memory_mb:.1f} MB of memory",
                    "impact": "May cause performance issues",
                }
            )

        return issues

    def _generate_recommendations(self, assessment: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on assessment."""
        recommendations = []

        # Completeness recommendations
        completeness = assessment["completeness"]
        if completeness["completeness_score"] < 80:
            recommendations.append(
                "Consider handling missing values through imputation or removal"
            )

        if completeness["columns_mostly_missing"]:
            recommendations.append(
                f"Consider removing columns with >80% missing data: {completeness['columns_mostly_missing']}"
            )

        # Uniqueness recommendations
        uniqueness = assessment["uniqueness"]
        if uniqueness["duplicate_rows"] > 0:
            recommendations.append(
                f"Remove {uniqueness['duplicate_rows']} duplicate rows"
            )

        if uniqueness["duplicate_columns"]:
            recommendations.append(
                f"Consider removing duplicate columns: {uniqueness['duplicate_columns']}"
            )

        # Consistency recommendations
        consistency = assessment["consistency"]
        if consistency["consistency_issues"]:
            recommendations.append("Address data consistency issues before analysis")

        # Issues-based recommendations
        for issue in assessment["issues"]:
            if issue["severity"] == "critical":
                if issue["type"] == "mostly_missing":
                    recommendations.append(
                        f"Remove or impute column '{issue['column']}'"
                    )
                elif issue["type"] == "constant_column":
                    recommendations.append(
                        f"Remove constant column '{issue['column']}'"
                    )

        return recommendations

    def _looks_like_dates(self, sample: pd.Series) -> bool:
        """Check if string series looks like dates."""
        date_indicators = ["-", "/", ":", "T", "20", "19"]  # Common date patterns

        matches = 0
        for value in sample.head(50):
            value_str = str(value)
            if any(indicator in value_str for indicator in date_indicators):
                matches += 1

        return matches > len(sample.head(50)) * 0.5

    def _detect_date_formats(self, sample: pd.Series) -> List[str]:
        """Detect different date formats in string series."""
        formats = set()

        for value in sample.head(20):
            value_str = str(value).strip()

            # Common patterns
            if "/" in value_str and len(value_str.split("/")) == 3:
                formats.add("MM/DD/YYYY or DD/MM/YYYY")
            elif "-" in value_str and len(value_str.split("-")) == 3:
                formats.add("YYYY-MM-DD")
            elif "T" in value_str:
                formats.add("ISO datetime")
            elif ":" in value_str:
                formats.add("HH:MM:SS")

        return list(formats)


class DataPreprocessor:
    """Handles data preprocessing and cleaning."""

    def __init__(self):
        """Initialize preprocessor."""
        pass

    def clean_dataset(
        self, df: pd.DataFrame, options: Dict[str, Any] = None
    ) -> pd.DataFrame:
        """
        Clean dataset based on options.

        Args:
            df: DataFrame to clean
            options: Cleaning options

        Returns:
            Cleaned DataFrame
        """
        if options is None:
            options = {}

        cleaned_df = df.copy()

        # Remove duplicate rows
        if options.get("remove_duplicates", True):
            initial_shape = cleaned_df.shape
            cleaned_df = cleaned_df.drop_duplicates()
            removed = initial_shape[0] - cleaned_df.shape[0]
            if removed > 0:
                logger.info(f"Removed {removed} duplicate rows")

        # Remove columns with too many missing values
        missing_threshold = options.get("missing_threshold", 0.9)
        if missing_threshold < 1.0:
            missing_pct = cleaned_df.isnull().sum() / len(cleaned_df)
            cols_to_drop = missing_pct[missing_pct > missing_threshold].index
            if len(cols_to_drop) > 0:
                cleaned_df = cleaned_df.drop(columns=cols_to_drop)
                logger.info(
                    f"Removed columns with >{missing_threshold*100}% missing: {list(cols_to_drop)}"
                )

        # Remove constant columns
        if options.get("remove_constant_columns", True):
            constant_cols = []
            for col in cleaned_df.columns:
                if cleaned_df[col].nunique() <= 1:
                    constant_cols.append(col)

            if constant_cols:
                cleaned_df = cleaned_df.drop(columns=constant_cols)
                logger.info(f"Removed constant columns: {constant_cols}")

        # Handle missing values
        missing_strategy = options.get("missing_strategy", "none")
        if missing_strategy != "none":
            cleaned_df = self._handle_missing_values(cleaned_df, missing_strategy)

        return cleaned_df

    def _handle_missing_values(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Handle missing values using specified strategy."""
        if strategy == "drop_rows":
            return df.dropna()

        elif strategy == "forward_fill":
            return df.fillna(method="ffill")

        elif strategy == "mean_median":
            # Fill numeric columns with mean, categorical with mode
            filled_df = df.copy()

            for col in df.columns:
                if df[col].dtype in [np.number]:
                    filled_df[col] = df[col].fillna(df[col].mean())
                else:
                    mode_val = df[col].mode()
                    if not mode_val.empty:
                        filled_df[col] = df[col].fillna(mode_val[0])

            return filled_df

        else:
            return df


def get_default_quality_assessor() -> DataQualityAssessor:
    """Get default quality assessor instance."""
    return DataQualityAssessor()


def get_default_preprocessor() -> DataPreprocessor:
    """Get default preprocessor instance."""
    return DataPreprocessor()
