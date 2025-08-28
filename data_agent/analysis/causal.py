"""
Advanced causal analysis for pipeline dataset.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from scipy import stats
from scipy.stats import shapiro, levene, kruskal
import logging
import warnings
from statsmodels.stats.multitest import multipletests
from sklearn.utils import resample

logger = logging.getLogger(__name__)


class CausalAnalyzer:
    """Advanced causal analysis for natural gas pipeline data."""

    def __init__(self):
        """Initialize causal analyzer."""
        pass

    def _calculate_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size for two groups."""
        try:
            n1, n2 = len(group1), len(group2)
            if n1 < 2 or n2 < 2:
                return np.nan
            
            # Calculate pooled standard deviation
            pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + 
                                 (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
            
            if pooled_std == 0:
                return np.nan
            
            return (np.mean(group1) - np.mean(group2)) / pooled_std
        except Exception:
            return np.nan

    def _calculate_eta_squared(self, f_stat: float, df_between: int, df_within: int) -> float:
        """Calculate eta-squared effect size for ANOVA."""
        try:
            if f_stat <= 0 or df_between <= 0:
                return np.nan
            return (f_stat * df_between) / (f_stat * df_between + df_within)
        except Exception:
            return np.nan

    def _interpret_effect_size(self, effect_size: float, measure: str = "cohens_d") -> str:
        """Interpret effect size magnitude."""
        if np.isnan(effect_size):
            return "unknown"
        
        abs_effect = abs(effect_size)
        
        if measure == "cohens_d":
            if abs_effect < 0.2:
                return "negligible"
            elif abs_effect < 0.5:
                return "small"
            elif abs_effect < 0.8:
                return "medium"
            else:
                return "large"
        elif measure == "eta_squared":
            if abs_effect < 0.01:
                return "negligible"
            elif abs_effect < 0.06:
                return "small"
            elif abs_effect < 0.14:
                return "medium"
            else:
                return "large"
        
        return "unknown"

    def _test_assumptions(self, samples: List[np.ndarray], test_name: str = "ANOVA") -> Dict[str, Any]:
        """Test statistical assumptions for parametric tests."""
        assumptions = {
            "normality_met": True,
            "homogeneity_met": True,
            "assumptions_valid": True,
            "normality_tests": [],
            "homogeneity_test": {},
            "recommended_test": test_name
        }
        
        try:
            # Test normality for each group
            normality_pvals = []
            for i, sample in enumerate(samples):
                if len(sample) >= 3 and len(sample) <= 5000:  # Shapiro-Wilk limitations
                    stat, p_val = shapiro(sample)
                    assumptions["normality_tests"].append({
                        "group": i,
                        "statistic": float(stat),
                        "p_value": float(p_val),
                        "normal": p_val > 0.05
                    })
                    normality_pvals.append(p_val)
            
            # Overall normality assessment
            if normality_pvals and min(normality_pvals) <= 0.05:
                assumptions["normality_met"] = False
                assumptions["recommended_test"] = "Kruskal-Wallis" if test_name == "ANOVA" else "Mann-Whitney U"
            
            # Test homogeneity of variance (Levene's test)
            if len(samples) >= 2:
                stat, p_val = levene(*samples, center='median')
                assumptions["homogeneity_test"] = {
                    "statistic": float(stat),
                    "p_value": float(p_val),
                    "homogeneous": p_val > 0.05
                }
                
                if p_val <= 0.05:
                    assumptions["homogeneity_met"] = False
            
            # Overall assumption validity
            assumptions["assumptions_valid"] = (
                assumptions["normality_met"] and assumptions["homogeneity_met"]
            )
            
        except Exception as e:
            logger.warning(f"Error testing assumptions: {e}")
            assumptions["error"] = str(e)
        
        return assumptions

    def _apply_multiple_testing_correction(self, p_values: List[float], 
                                         method: str = "fdr_bh") -> Dict[str, Any]:
        """Apply multiple testing correction to p-values."""
        try:
            if not p_values or len(p_values) <= 1:
                return {
                    "corrected_p_values": p_values,
                    "significant_corrected": [p <= 0.05 for p in p_values],
                    "method": method,
                    "n_tests": len(p_values)
                }
            
            # Apply correction
            rejected, corrected_pvals, alpha_sidak, alpha_bonf = multipletests(
                p_values, alpha=0.05, method=method
            )
            
            return {
                "corrected_p_values": corrected_pvals.tolist(),
                "significant_corrected": rejected.tolist(),
                "method": method,
                "n_tests": len(p_values),
                "alpha_bonferroni": float(alpha_bonf),
                "alpha_sidak": float(alpha_sidak)
            }
            
        except Exception as e:
            logger.warning(f"Error applying multiple testing correction: {e}")
            return {
                "corrected_p_values": p_values,
                "significant_corrected": [p <= 0.05 for p in p_values],
                "method": method,
                "error": str(e)
            }

    def _validate_and_perform_group_test(self, samples: List[pd.Series], test_name: str) -> Dict[str, Any]:
        """Validate assumptions and perform appropriate statistical test."""
        result = {
            "test": test_name,
            "n_groups": len(samples),
            "sample_sizes": [len(s) for s in samples]
        }
        
        try:
            # Convert to numpy arrays for statistical testing
            np_samples = [np.array(s.dropna()) for s in samples]
            
            # Test statistical assumptions
            assumptions = self._test_assumptions(np_samples, "ANOVA")
            result["assumptions"] = assumptions
            
            # Perform appropriate statistical test based on assumptions
            if assumptions["assumptions_valid"]:
                # Use parametric ANOVA
                f_stat, p_value = stats.f_oneway(*np_samples)
                result.update({
                    "statistic": float(f_stat),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                    "test_type": "parametric"
                })
                
                # Calculate effect size (eta-squared)
                df_between = len(samples) - 1
                df_within = sum(len(s) - 1 for s in np_samples)
                eta_squared = self._calculate_eta_squared(f_stat, df_between, df_within)
                result["effect_size"] = {
                    "eta_squared": float(eta_squared) if not np.isnan(eta_squared) else None,
                    "interpretation": self._interpret_effect_size(eta_squared, "eta_squared"),
                    "measure": "eta_squared"
                }
                
            else:
                # Use non-parametric Kruskal-Wallis test
                h_stat, p_value = kruskal(*np_samples)
                result.update({
                    "statistic": float(h_stat),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                    "test_type": "non_parametric",
                    "test": test_name.replace("ANOVA", "Kruskal-Wallis")
                })
                
                # For Kruskal-Wallis, calculate epsilon-squared as effect size
                n_total = sum(len(s) for s in np_samples)
                epsilon_squared = (h_stat - len(samples) + 1) / (n_total - len(samples))
                result["effect_size"] = {
                    "epsilon_squared": float(epsilon_squared),
                    "interpretation": self._interpret_effect_size(epsilon_squared, "eta_squared"),
                    "measure": "epsilon_squared"
                }
            
            # Add descriptive statistics for each group
            result["descriptive_stats"] = []
            for i, sample in enumerate(np_samples):
                if len(sample) > 0:
                    result["descriptive_stats"].append({
                        "group": i,
                        "n": len(sample),
                        "mean": float(np.mean(sample)),
                        "std": float(np.std(sample, ddof=1)),
                        "median": float(np.median(sample)),
                        "min": float(np.min(sample)),
                        "max": float(np.max(sample))
                    })
            
        except Exception as e:
            logger.error(f"Error in group statistical test: {e}")
            result["error"] = str(e)
        
        return result

    def _perform_pairwise_comparisons(self, samples: List[pd.Series], group_names: List[str] = None) -> Dict[str, Any]:
        """Perform pairwise comparisons between groups with effect sizes."""
        if group_names is None:
            group_names = [f"Group_{i}" for i in range(len(samples))]
        
        comparisons = []
        p_values = []
        
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                try:
                    sample1 = np.array(samples[i].dropna())
                    sample2 = np.array(samples[j].dropna())
                    
                    if len(sample1) < 3 or len(sample2) < 3:
                        continue
                    
                    # Test assumptions for this pair
                    assumptions = self._test_assumptions([sample1, sample2], "t-test")
                    
                    # Perform appropriate test
                    if assumptions["assumptions_valid"]:
                        # Independent t-test
                        statistic, p_value = stats.ttest_ind(sample1, sample2)
                        test_type = "parametric (t-test)"
                    else:
                        # Mann-Whitney U test
                        statistic, p_value = stats.mannwhitneyu(
                            sample1, sample2, alternative='two-sided'
                        )
                        test_type = "non_parametric (Mann-Whitney U)"
                    
                    # Calculate effect size
                    cohens_d = self._calculate_cohens_d(sample1, sample2)
                    
                    comparison = {
                        "group1": group_names[i],
                        "group2": group_names[j],
                        "statistic": float(statistic),
                        "p_value": float(p_value),
                        "significant": p_value < 0.05,
                        "test_type": test_type,
                        "effect_size": {
                            "cohens_d": float(cohens_d) if not np.isnan(cohens_d) else None,
                            "interpretation": self._interpret_effect_size(cohens_d, "cohens_d"),
                            "measure": "cohens_d"
                        },
                        "assumptions": assumptions
                    }
                    
                    comparisons.append(comparison)
                    p_values.append(p_value)
                    
                except Exception as e:
                    logger.warning(f"Error in pairwise comparison {group_names[i]} vs {group_names[j]}: {e}")
        
        # Apply multiple testing correction
        if p_values:
            correction = self._apply_multiple_testing_correction(p_values)
            
            # Update comparisons with corrected p-values
            for i, comparison in enumerate(comparisons):
                if i < len(correction["corrected_p_values"]):
                    comparison["p_value_corrected"] = correction["corrected_p_values"][i]
                    comparison["significant_corrected"] = correction["significant_corrected"][i]
            
            return {
                "comparisons": comparisons,
                "multiple_testing_correction": correction,
                "n_comparisons": len(comparisons)
            }
        
        return {"comparisons": comparisons, "n_comparisons": len(comparisons)}

    def _perform_two_group_comparison(self, sample1: pd.Series, sample2: pd.Series, 
                                    comparison_name: str) -> Dict[str, Any]:
        """Perform statistical comparison between two groups with effect sizes and confidence intervals."""
        result = {
            "test": comparison_name,
            "n1": len(sample1.dropna()),
            "n2": len(sample2.dropna())
        }
        
        try:
            arr1 = np.array(sample1.dropna())
            arr2 = np.array(sample2.dropna())
            
            if len(arr1) < 3 or len(arr2) < 3:
                result["error"] = "Insufficient sample sizes for comparison"
                return result
            
            # Test statistical assumptions
            assumptions = self._test_assumptions([arr1, arr2], "t-test")
            result["assumptions"] = assumptions
            
            # Perform appropriate statistical test
            if assumptions["assumptions_valid"]:
                # Independent t-test
                statistic, p_value = stats.ttest_ind(arr1, arr2)
                result.update({
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                    "test_type": "parametric (t-test)"
                })
                
                # Calculate confidence interval for mean difference
                mean_diff = np.mean(arr1) - np.mean(arr2)
                pooled_se = np.sqrt(np.var(arr1, ddof=1) / len(arr1) + np.var(arr2, ddof=1) / len(arr2))
                df = len(arr1) + len(arr2) - 2
                t_critical = stats.t.ppf(0.975, df)
                ci_lower = mean_diff - t_critical * pooled_se
                ci_upper = mean_diff + t_critical * pooled_se
                
                result["confidence_intervals"] = {
                    "mean_difference": float(mean_diff),
                    "ci_lower": float(ci_lower),
                    "ci_upper": float(ci_upper),
                    "confidence_level": 0.95
                }
                
            else:
                # Mann-Whitney U test
                statistic, p_value = stats.mannwhitneyu(
                    arr1, arr2, alternative='two-sided'
                )
                result.update({
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                    "test_type": "non_parametric (Mann-Whitney U)"
                })
                
                # For non-parametric test, use bootstrap for confidence interval of median difference
                try:
                    n_bootstrap = 1000
                    bootstrap_diffs = []
                    for _ in range(n_bootstrap):
                        boot1 = resample(arr1, n_samples=len(arr1))
                        boot2 = resample(arr2, n_samples=len(arr2))
                        bootstrap_diffs.append(np.median(boot1) - np.median(boot2))
                    
                    ci_lower = np.percentile(bootstrap_diffs, 2.5)
                    ci_upper = np.percentile(bootstrap_diffs, 97.5)
                    median_diff = np.median(arr1) - np.median(arr2)
                    
                    result["confidence_intervals"] = {
                        "median_difference": float(median_diff),
                        "ci_lower": float(ci_lower),
                        "ci_upper": float(ci_upper),
                        "confidence_level": 0.95,
                        "method": "bootstrap"
                    }
                except Exception as e:
                    logger.warning(f"Bootstrap CI calculation failed: {e}")
            
            # Calculate effect size
            cohens_d = self._calculate_cohens_d(arr1, arr2)
            result["effect_size"] = {
                "cohens_d": float(cohens_d) if not np.isnan(cohens_d) else None,
                "interpretation": self._interpret_effect_size(cohens_d, "cohens_d"),
                "measure": "cohens_d"
            }
            
            # Add descriptive statistics
            result["descriptive_stats"] = {
                "group1": {
                    "n": len(arr1),
                    "mean": float(np.mean(arr1)),
                    "std": float(np.std(arr1, ddof=1)),
                    "median": float(np.median(arr1)),
                    "min": float(np.min(arr1)),
                    "max": float(np.max(arr1))
                },
                "group2": {
                    "n": len(arr2),
                    "mean": float(np.mean(arr2)),
                    "std": float(np.std(arr2, ddof=1)),
                    "median": float(np.median(arr2)),
                    "min": float(np.min(arr2)),
                    "max": float(np.max(arr2))
                }
            }
            
        except Exception as e:
            logger.error(f"Error in two-group comparison: {e}")
            result["error"] = str(e)
        
        return result

    def _apply_overall_multiple_testing_correction(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply multiple testing correction to all statistical tests in the results."""
        try:
            if not results.get("statistical_tests"):
                return results
            
            # Extract p-values from all statistical tests
            p_values = []
            test_indices = []
            
            for i, test in enumerate(results["statistical_tests"]):
                if "p_value" in test and test["p_value"] is not None:
                    p_values.append(test["p_value"])
                    test_indices.append(i)
            
            if len(p_values) > 1:
                # Apply multiple testing correction
                correction = self._apply_multiple_testing_correction(p_values)
                
                # Update each test with corrected p-values
                for i, test_idx in enumerate(test_indices):
                    if i < len(correction["corrected_p_values"]):
                        results["statistical_tests"][test_idx]["p_value_corrected"] = correction["corrected_p_values"][i]
                        results["statistical_tests"][test_idx]["significant_corrected"] = correction["significant_corrected"][i]
                
                # Add correction summary to results
                results["multiple_testing_correction"] = {
                    "method": correction["method"],
                    "n_tests": correction["n_tests"],
                    "significant_before_correction": sum(p <= 0.05 for p in p_values),
                    "significant_after_correction": sum(correction["significant_corrected"]),
                    "family_wise_error_controlled": True
                }
                
                # Update confidence based on corrected results
                significant_corrected = len([
                    t for t in results["statistical_tests"] 
                    if t.get("significant_corrected", t.get("significant", False))
                ])
                
                if significant_corrected >= 2:
                    results["confidence"] = "high"
                elif significant_corrected >= 1:
                    results["confidence"] = "medium"
                else:
                    results["confidence"] = "low"
            
        except Exception as e:
            logger.error(f"Error applying multiple testing correction: {e}")
            results["multiple_testing_correction"] = {"error": str(e)}
        
        return results

    def analyze_pipeline_capacity_drivers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze what drives capacity utilization using statistical relationships."""
        results = {
            "analysis_type": "capacity_drivers",
            "methodology": "Statistical correlation and regression analysis",
            "drivers": [],
            "statistical_tests": [],
            "confidence": "medium",
            "caveats": [],
        }

        try:
            # Find numeric columns that could represent capacity/utilization metrics
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                results["error"] = "Need at least 2 numeric columns for causal analysis"
                return results
            
            # Look for capacity/utilization related columns
            capacity_cols = [col for col in numeric_cols if any(keyword in col.lower() 
                           for keyword in ['capacity', 'util', 'load', 'demand', 'pipeline'])]
            
            if not capacity_cols:
                # If no capacity-specific columns, use the first numeric column as target
                target_col = numeric_cols[0]
                predictor_cols = numeric_cols[1:]
            else:
                # Use first capacity column as target
                target_col = capacity_cols[0]
                predictor_cols = [col for col in numeric_cols if col != target_col]
            
            if len(predictor_cols) == 0:
                results["error"] = "No predictor variables available"
                return results

            # Analyze correlation relationships between target and predictors
            correlation_drivers = []
            for predictor in predictor_cols:
                # Remove missing values for correlation calculation
                clean_data = df[[target_col, predictor]].dropna()
                if len(clean_data) >= 30:  # Minimum sample size for correlation
                    correlation = clean_data[target_col].corr(clean_data[predictor])
                    
                    if abs(correlation) >= 0.3:  # Moderate correlation threshold
                        # Calculate confidence interval for correlation
                        conf_int = self._calculate_correlation_confidence_interval(
                            correlation, len(clean_data)
                        )
                        
                        driver = {
                            "factor": predictor,
                            "correlation": float(correlation),
                            "relationship": "positive" if correlation > 0 else "negative",
                            "strength": "strong" if abs(correlation) >= 0.7 else "moderate",
                            "sample_size": len(clean_data),
                            "confidence_interval": conf_int,
                            "evidence": f"r = {correlation:.3f}, 95% CI [{conf_int['lower']:.3f}, {conf_int['upper']:.3f}]"
                        }
                        correlation_drivers.append(driver)
            
            results["drivers"] = correlation_drivers
            
            # Look for categorical variables that might group data differently
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            for cat_col in categorical_cols:
                if df[cat_col].nunique() <= 10 and df[cat_col].nunique() >= 2:  # Reasonable number of groups
                    # Group analysis
                    groups = []
                    for category in df[cat_col].unique():
                        if pd.notna(category):
                            group_data = df[df[cat_col] == category][target_col].dropna()
                            if len(group_data) >= 10:  # Minimum group size
                                groups.append(group_data)
                    
                    if len(groups) >= 2:
                        # Perform group comparison test
                        test_result = self._validate_and_perform_group_test(
                            groups, f"Group Analysis ({cat_col})"
                        )
                        results["statistical_tests"].append(test_result)
                        
                        if test_result.get("significant", False):
                            group_means = [float(g.mean()) for g in groups]
                            driver = {
                                "factor": cat_col,
                                "effect": "Significant group differences",
                                "group_means": group_means,
                                "test_statistic": test_result.get("statistic"),
                                "p_value": test_result.get("p_value"),
                                "effect_size": test_result.get("effect_size", {}),
                                "evidence": f"Test: {test_result.get('test', 'N/A')}, p = {test_result.get('p_value', 'N/A'):.4f}"
                            }
                            results["drivers"].append(driver)

            # Set confidence based on findings
            if len(results["drivers"]) >= 3:
                results["confidence"] = "high"
            elif len(results["drivers"]) >= 1:
                results["confidence"] = "medium"
            else:
                results["confidence"] = "low"

            # Apply multiple testing correction if we have statistical tests
            if results["statistical_tests"]:
                results = self._apply_overall_multiple_testing_correction(results)

            # Add analysis summary
            results["summary"] = {
                "target_variable": target_col,
                "predictors_analyzed": predictor_cols,
                "significant_drivers": len(results["drivers"]),
                "data_points": len(df)
            }

            # Add appropriate caveats
            results["caveats"] = [
                "Correlation does not imply causation",
                "Results are based on statistical associations in the data",
                "External factors not in dataset may influence relationships",
                "Temporal relationships not considered in this analysis"
            ]

        except Exception as e:
            logger.error(f"Error in causal analysis: {e}")
            results["error"] = f"Analysis failed: {str(e)}"

        return results

    def _calculate_correlation_confidence_interval(self, r: float, n: int, confidence: float = 0.95) -> Dict[str, float]:
        """Calculate confidence interval for correlation coefficient using Fisher's z-transformation."""
        try:
            if abs(r) >= 1.0 or n < 3:
                return {"lower": np.nan, "upper": np.nan, "confidence_level": confidence}
            
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
                "lower": float(r_lower),
                "upper": float(r_upper),
                "confidence_level": confidence
            }
            
        except Exception as e:
            logger.warning(f"Error calculating correlation confidence interval: {e}")
            return {"lower": np.nan, "upper": np.nan, "confidence_level": confidence}

    def detect_infrastructure_bottlenecks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify potential infrastructure bottlenecks using statistical analysis."""
        results = {
            "analysis_type": "bottleneck_detection",
            "methodology": "Statistical bottleneck detection using capacity and utilization metrics",
            "bottlenecks": [],
            "recommendations": [],
            "confidence": "medium",
            "columns_analyzed": [],
            "data_summary": {},
        }

        try:
            # Find numeric columns that could represent capacity, utilization, or throughput
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                results["error"] = "No numeric columns found for bottleneck analysis"
                return results
            
            # Look for columns that might represent infrastructure metrics
            capacity_keywords = [
                'capacity', 'util', 'load', 'throughput', 'pipeline', 'flow', 'rate', 'volume',
                'demand', 'usage', 'consumption', 'quantity', 'amount', 'traffic', 'bandwidth',
                'pressure', 'temperature', 'efficiency', 'performance', 'speed', 'power',
                'production', 'output', 'input', 'delivery', 'supply', 'peak', 'max', 'min'
            ]
            
            capacity_cols = [col for col in numeric_cols if any(keyword in col.lower() 
                           for keyword in capacity_keywords)]
            
            if not capacity_cols:
                # Use all numeric columns if no infrastructure-specific ones found
                # But prioritize columns with higher variance (more likely to show bottlenecks)
                col_variances = []
                for col in numeric_cols:
                    if df[col].std() > 0:  # Skip constant columns
                        variance = df[col].var()
                        col_variances.append((col, variance))
                
                # Sort by variance and take top columns
                col_variances.sort(key=lambda x: x[1], reverse=True)
                capacity_cols = [col for col, _ in col_variances[:8]]  # Take top 8 most variable columns
            
            # Calculate statistical bottlenecks for each metric
            for col in capacity_cols:
                if df[col].std() == 0:  # Skip constant columns
                    continue
                
                # Identify outliers (potential bottlenecks) using statistical methods
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define bottleneck as values significantly above Q3 (high utilization/load)
                bottleneck_threshold = Q3 + 1.5 * IQR
                high_values = df[df[col] > bottleneck_threshold]
                
                if len(high_values) > 0:
                    # Calculate bottleneck statistics
                    mean_normal = df[df[col] <= Q3][col].mean()
                    mean_bottleneck = high_values[col].mean()
                    severity = (mean_bottleneck - mean_normal) / mean_normal if mean_normal != 0 else 0
                    
                    bottleneck = {
                        "metric": col,
                        "bottleneck_instances": len(high_values),
                        "threshold": float(bottleneck_threshold),
                        "severity_ratio": float(severity),
                        "mean_normal": float(mean_normal),
                        "mean_bottleneck": float(mean_bottleneck),
                        "max_observed": float(high_values[col].max()),
                        "frequency_pct": float(len(high_values) / len(df) * 100),
                        "bottleneck_score": float(severity * len(high_values) / len(df))
                    }
                    
                    results["bottlenecks"].append(bottleneck)

            # Generate recommendations and insights
            if len(results["bottlenecks"]) > 0:
                results["recommendations"] = [
                    "Monitor metrics with high bottleneck scores for capacity constraints",
                    "Consider infrastructure investments at identified bottleneck points", 
                    "Implement load balancing to distribute traffic more evenly",
                    "Analyze patterns to predict when bottlenecks are most likely to occur"
                ]
                results["confidence"] = "high"
            else:
                # Provide insights even when no major bottlenecks found
                results["recommendations"] = [
                    f"Analyzed {len(capacity_cols)} metrics across {len(df)} data points",
                    "No critical bottlenecks detected - system appears to be operating within normal parameters",
                    "Continue monitoring capacity utilization trends for early warning signs",
                    "Consider analyzing longer time periods or additional metrics for comprehensive assessment"
                ]
                results["confidence"] = "medium"
                
                # Add information about the metrics that were analyzed
                high_variance_cols = [col for col, variance in col_variances[:3]] if 'col_variances' in locals() else capacity_cols[:3]
                results["analysis_insights"] = {
                    "highest_variance_metrics": high_variance_cols,
                    "analysis_approach": "Statistical outlier detection using IQR method",
                    "threshold_method": "Q3 + 1.5 * IQR for each metric"
                }

            # Add analysis metadata
            results["columns_analyzed"] = capacity_cols
            results["data_summary"] = {
                "total_records": len(df),
                "total_numeric_columns": len(numeric_cols),
                "columns_examined": len(capacity_cols)
            }

            results["summary"] = {
                "total_metrics_analyzed": len(capacity_cols),
                "potential_bottlenecks_found": len(results["bottlenecks"]),
                "analysis_columns": capacity_cols,
                "data_points": len(df)
            }

        except Exception as e:
            logger.error(f"Error in bottleneck analysis: {e}")
            results["error"] = f"Analysis failed: {str(e)}"

        return results

    def analyze_seasonal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonal patterns in gas pipeline usage."""
        results = {
            "analysis_type": "seasonal_patterns",
            "methodology": "Time series decomposition and statistical testing",
            "patterns": [],
            "statistical_tests": [],
            "confidence": "medium",
            "date_column": None,
            "value_column": None,
            "data_summary": {},
        }

        month_names = {
            1: "January", 2: "February", 3: "March", 4: "April",
            5: "May", 6: "June", 7: "July", 8: "August",
            9: "September", 10: "October", 11: "November", 12: "December",
        }

        try:
            # Find date columns (look for datetime columns or columns that can be converted)
            date_col = None
            value_col = None
            
            # First, look for explicit datetime columns
            datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            
            # Also look for columns that might be dates
            potential_date_cols = []
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['date', 'time', 'day', 'month', 'year']):
                    potential_date_cols.append(col)
            
            # Try the most specific date column first
            date_candidates = []
            if "eff_gas_day" in df.columns:
                date_candidates.append("eff_gas_day")
            date_candidates.extend(datetime_cols)
            date_candidates.extend(potential_date_cols)
            
            # Try to convert and use the first valid date column
            for candidate in date_candidates:
                if candidate in df.columns:
                    try:
                        if df[candidate].dtype == "category" or not pd.api.types.is_datetime64_any_dtype(df[candidate]):
                            test_df = df.copy()
                            if pd.api.types.is_categorical_dtype(test_df[candidate]):
                                test_df[candidate] = test_df[candidate].astype(str)
                            test_df[candidate] = pd.to_datetime(test_df[candidate], errors="coerce")
                            
                            # Check if conversion was successful (more than 50% valid dates)
                            valid_dates = test_df[candidate].notna().sum()
                            if valid_dates / len(test_df) > 0.5:
                                df = test_df
                                date_col = candidate
                                break
                        else:
                            date_col = candidate
                            break
                    except Exception:
                        continue
            
            if date_col is None:
                results["error"] = "No valid date column found for seasonal analysis"
                return results
            
            # Find numeric columns for analysis (look for quantity, demand, volume, etc.)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            value_candidates = []
            if "scheduled_quantity" in df.columns:
                value_candidates.append("scheduled_quantity")
            
            # Look for columns that might represent values/quantities
            for col in numeric_cols:
                if any(keyword in col.lower() for keyword in [
                    'quantity', 'amount', 'volume', 'demand', 'consumption', 'usage', 
                    'flow', 'rate', 'power', 'energy', 'gas', 'production', 'delivery', 'supply'
                ]):
                    value_candidates.append(col)
            
            # If no specific candidates, use the first few numeric columns
            if not value_candidates:
                value_candidates = numeric_cols[:3]
            
            if not value_candidates:
                results["error"] = "No numeric columns found for seasonal analysis"
                return results
                
            value_col = value_candidates[0]
            
            # Add time components
            df["month"] = df[date_col].dt.month
            df["quarter"] = df[date_col].dt.quarter
            df["year"] = df[date_col].dt.year

            # Monthly analysis for non-zero values
            active_flows = df[df[value_col] > 0] if df[value_col].min() >= 0 else df[df[value_col] != 0]
            monthly_stats = (
                active_flows.groupby("month")[value_col]
                .agg(["mean", "count", "std"])
                .round(2)
            )

            if len(monthly_stats) >= 12:  # Full year of data
                # Test for seasonal differences
                monthly_samples = []
                for month in range(1, 13):
                    month_data = active_flows[active_flows["month"] == month][value_col]
                    if len(month_data) >= 30:  # Reduced threshold for more datasets
                        monthly_samples.append(
                            month_data.sample(min(500, len(month_data)))  # Smaller sample size
                        )

                    if len(monthly_samples) >= 12:
                        # Validate ANOVA assumptions and apply appropriate test
                        test_result = self._validate_and_perform_group_test(
                            monthly_samples, "ANOVA (Monthly Seasonality)"
                        )
                        results["statistical_tests"].append(test_result)

                        if test_result.get("significant_corrected", test_result.get("significant", False)):
                            # Identify peak and low months
                            peak_month = monthly_stats["mean"].idxmax()
                            low_month = monthly_stats["mean"].idxmin()

                            results["patterns"].append(
                                {
                                    "pattern_type": "Monthly Seasonality",
                                    "peak_month": month_names[peak_month],
                                    "low_month": month_names[low_month],
                                    "peak_volume": float(
                                        monthly_stats.loc[peak_month, "mean"]
                                    ),
                                    "low_volume": float(
                                        monthly_stats.loc[low_month, "mean"]
                                    ),
                                    "seasonal_variation": float(
                                        (
                                            monthly_stats["mean"].max()
                                            - monthly_stats["mean"].min()
                                        )
                                        / monthly_stats["mean"].mean()
                                        * 100
                                    ),
                                    "evidence": f"Test: {test_result['test']}, Statistic: {test_result.get('statistic', 'N/A'):.2f}, p-value: {test_result.get('p_value_corrected', test_result.get('p_value', 'N/A')):.4f}",
                                    "effect_size": test_result.get('effect_size', {}),
                                }
                            )

                # Category-specific seasonal analysis
                if "category_short" in df.columns:
                    for category in [
                        "Power",
                        "LDC",
                        "Industrial",
                    ]:  # Focus on key categories
                        cat_data = active_flows[
                            active_flows["category_short"] == category
                        ]
                        if len(cat_data) >= 1000:  # Minimum sample
                            cat_monthly = cat_data.groupby("month")[
                                "scheduled_quantity"
                            ].mean()
                            if len(cat_monthly) >= 12:
                                peak_month = cat_monthly.idxmax()
                                low_month = cat_monthly.idxmin()
                                seasonal_var = (
                                    (cat_monthly.max() - cat_monthly.min())
                                    / cat_monthly.mean()
                                    * 100
                                )

                                if seasonal_var > 20:  # Significant seasonal variation
                                    results["patterns"].append(
                                        {
                                            "pattern_type": f"{category} Seasonal Pattern",
                                            "peak_month": month_names.get(
                                                peak_month, str(peak_month)
                                            ),
                                            "low_month": month_names.get(
                                                low_month, str(low_month)
                                            ),
                                            "seasonal_variation": float(seasonal_var),
                                            "business_context": self._get_seasonal_context(
                                                category, peak_month
                                            ),
                                        }
                                    )

            # Set confidence based on findings
            if len(results["patterns"]) >= 2:
                results["confidence"] = "high"
            elif len(results["patterns"]) >= 1:
                results["confidence"] = "medium"
            elif len(results["statistical_tests"]) > 0:
                results["confidence"] = "medium"
                # Add insights even when no clear patterns found
                results["analysis_insights"] = {
                    "statistical_tests_performed": len(results["statistical_tests"]),
                    "data_examined": f"{date_col} vs {value_col}" if date_col and value_col else "Date and value columns",
                    "records_analyzed": len(active_flows) if 'active_flows' in locals() else len(df),
                    "methodology": "ANOVA and post-hoc testing for seasonal differences",
                    "interpretation": "No statistically significant seasonal patterns detected"
                }
            else:
                results["confidence"] = "low"
                # Provide context about why no patterns were found
                results["analysis_insights"] = {
                    "reason": "Insufficient data for comprehensive seasonal analysis",
                    "data_examined": f"{date_col} vs {value_col}" if date_col and value_col else "Date and value columns",
                    "records_available": len(df),
                    "recommendations": [
                        "Collect data spanning at least 12 months for seasonal analysis",
                        "Ensure sufficient sample sizes per time period",
                        "Consider other temporal patterns (weekly, quarterly)"
                    ]
                }

            # Add analysis metadata
            results["date_column"] = date_col
            results["value_column"] = value_col
            results["data_summary"] = {
                "total_records": len(df),
                "records_with_values": len(active_flows),
                "date_range_days": (df[date_col].max() - df[date_col].min()).days if date_col else 0,
                "months_with_data": len(monthly_stats) if 'monthly_stats' in locals() else 0
            }
            
            # Apply multiple testing correction for seasonal analysis
            if results["statistical_tests"]:
                results = self._apply_multiple_testing_correction(results)

        except Exception as e:
            logger.error(f"Error in seasonal analysis: {e}")
            results["error"] = f"Analysis failed: {str(e)}"

        return results

    def _get_seasonal_context(self, category: str, peak_month: int) -> str:
        """Get business context for seasonal patterns."""
        if category == "Power":
            if peak_month in [12, 1, 2, 6, 7, 8]:  # Winter/Summer peaks
                return "Power plants increase gas usage during heating/cooling seasons"
            else:
                return "Moderate power generation gas demand"
        elif category == "LDC":
            if peak_month in [12, 1, 2]:  # Winter peak
                return "Local distribution companies peak during heating season"
            else:
                return "Lower residential/commercial heating demand"
        elif category == "Industrial":
            return "Industrial demand may reflect production cycles"
        else:
            return "Seasonal pattern specific to business operations"

    def _validate_and_perform_group_test(self, samples: List[np.ndarray], test_name: str) -> Dict[str, Any]:
        """
        Validate statistical assumptions and perform appropriate group test.
        
        Args:
            samples: List of sample arrays for comparison
            test_name: Name of the test for reporting
            
        Returns:
            Test results with assumptions, statistics, and effect sizes
        """
        try:
            if len(samples) < 2:
                return {
                    "test": test_name,
                    "error": "Need at least 2 groups for comparison",
                    "significant": False
                }
            
            # Test assumptions
            assumptions = self._test_anova_assumptions(samples)
            
            if assumptions["normality_passed"] and assumptions["homogeneity_passed"]:
                # Use parametric ANOVA
                f_stat, p_value = stats.f_oneway(*samples)
                test_used = "ANOVA (Parametric)"
                statistic = f_stat
                
                # Calculate eta-squared (effect size for ANOVA)
                effect_size = self._calculate_eta_squared(samples, f_stat)
                
            else:
                # Use non-parametric Kruskal-Wallis test
                h_stat, p_value = stats.kruskal(*samples)
                test_used = "Kruskal-Wallis (Non-parametric)"
                statistic = h_stat
                
                # Calculate epsilon-squared (effect size for Kruskal-Wallis)
                effect_size = self._calculate_epsilon_squared(samples, h_stat)
            
            # Calculate confidence intervals for pairwise comparisons
            confidence_intervals = self._calculate_group_confidence_intervals(samples)
            
            result = {
                "test": f"{test_name} - {test_used}",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "assumptions": assumptions,
                "effect_size": effect_size,
                "confidence_intervals": confidence_intervals,
                "sample_sizes": [len(sample) for sample in samples]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in group test validation: {e}")
            return {
                "test": test_name,
                "error": str(e),
                "significant": False
            }
    
    def _test_anova_assumptions(self, samples: List[np.ndarray]) -> Dict[str, Any]:
        """
        Test ANOVA assumptions: normality and homogeneity of variance.
        
        Args:
            samples: List of sample arrays
            
        Returns:
            Dictionary with assumption test results
        """
        assumptions = {
            "normality_passed": True,
            "homogeneity_passed": True,
            "normality_tests": [],
            "homogeneity_test": {}
        }
        
        try:
            # Test normality for each sample (if sample size appropriate)
            for i, sample in enumerate(samples):
                if len(sample) >= 3 and len(sample) <= 5000:  # Shapiro-Wilk limits
                    try:
                        stat, p_val = shapiro(sample)
                        normality_result = {
                            "group": i,
                            "statistic": float(stat),
                            "p_value": float(p_val),
                            "normal": p_val > 0.05
                        }
                        assumptions["normality_tests"].append(normality_result)
                        
                        if p_val <= 0.05:
                            assumptions["normality_passed"] = False
                    except Exception as e:
                        logger.warning(f"Normality test failed for group {i}: {e}")
                        assumptions["normality_passed"] = False
            
            # Test homogeneity of variance (Levene's test)
            if len(samples) >= 2:
                try:
                    stat, p_val = levene(*samples)
                    assumptions["homogeneity_test"] = {
                        "statistic": float(stat),
                        "p_value": float(p_val),
                        "homogeneous": p_val > 0.05
                    }
                    
                    if p_val <= 0.05:
                        assumptions["homogeneity_passed"] = False
                        
                except Exception as e:
                    logger.warning(f"Homogeneity test failed: {e}")
                    assumptions["homogeneity_passed"] = False
            
        except Exception as e:
            logger.error(f"Error testing ANOVA assumptions: {e}")
            assumptions["normality_passed"] = False
            assumptions["homogeneity_passed"] = False
        
        return assumptions
    
    def _calculate_eta_squared(self, samples: List[np.ndarray], f_stat: float) -> Dict[str, Any]:
        """
        Calculate eta-squared effect size for ANOVA.
        
        Args:
            samples: List of sample arrays
            f_stat: F-statistic from ANOVA
            
        Returns:
            Effect size information
        """
        try:
            # Calculate degrees of freedom
            k = len(samples)  # number of groups
            n_total = sum(len(sample) for sample in samples)
            df_between = k - 1
            df_within = n_total - k
            
            # Calculate eta-squared
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
                "type": "eta-squared (ANOVA effect size)"
            }
            
        except Exception as e:
            logger.error(f"Error calculating eta-squared: {e}")
            return {"error": str(e)}
    
    def _calculate_epsilon_squared(self, samples: List[np.ndarray], h_stat: float) -> Dict[str, Any]:
        """
        Calculate epsilon-squared effect size for Kruskal-Wallis test.
        
        Args:
            samples: List of sample arrays
            h_stat: H-statistic from Kruskal-Wallis test
            
        Returns:
            Effect size information
        """
        try:
            n_total = sum(len(sample) for sample in samples)
            
            # Calculate epsilon-squared
            epsilon_squared = (h_stat - len(samples) + 1) / (n_total - len(samples))
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
                "type": "epsilon-squared (Kruskal-Wallis effect size)"
            }
            
        except Exception as e:
            logger.error(f"Error calculating epsilon-squared: {e}")
            return {"error": str(e)}
    
    def _calculate_group_confidence_intervals(self, samples: List[np.ndarray]) -> Dict[str, Any]:
        """
        Calculate confidence intervals for group means/medians.
        
        Args:
            samples: List of sample arrays
            
        Returns:
            Confidence interval information
        """
        try:
            confidence_intervals = {}
            
            for i, sample in enumerate(samples):
                if len(sample) >= 10:  # Minimum for bootstrap
                    # Bootstrap confidence interval for median
                    bootstrap_medians = []
                    for _ in range(1000):
                        bootstrap_sample = resample(sample, n_samples=len(sample))
                        bootstrap_medians.append(np.median(bootstrap_sample))
                    
                    ci_lower = np.percentile(bootstrap_medians, 2.5)
                    ci_upper = np.percentile(bootstrap_medians, 97.5)
                    
                    confidence_intervals[f"group_{i}"] = {
                        "median": float(np.median(sample)),
                        "ci_lower": float(ci_lower),
                        "ci_upper": float(ci_upper),
                        "method": "bootstrap"
                    }
                else:
                    # Fallback to simple statistics
                    confidence_intervals[f"group_{i}"] = {
                        "median": float(np.median(sample)),
                        "method": "insufficient_data_for_ci"
                    }
            
            return confidence_intervals
            
        except Exception as e:
            logger.error(f"Error calculating group confidence intervals: {e}")
            return {"error": str(e)}
    
    def _perform_mann_whitney_test(self, sample1: np.ndarray, sample2: np.ndarray, test_name: str) -> Dict[str, Any]:
        """
        Perform Mann-Whitney U test with effect size and confidence intervals.
        
        Args:
            sample1: First sample
            sample2: Second sample
            test_name: Name of the test
            
        Returns:
            Test results with effect size and confidence intervals
        """
        try:
            # Perform Mann-Whitney U test
            statistic, p_value = stats.mannwhitneyu(
                sample1, sample2, alternative="two-sided"
            )
            
            # Calculate effect size (rank-biserial correlation)
            effect_size = self._calculate_rank_biserial_correlation(sample1, sample2, statistic)
            
            # Calculate confidence intervals for medians
            confidence_intervals = self._calculate_median_confidence_intervals(
                sample1, sample2
            )
            
            result = {
                "test": f"Mann-Whitney U ({test_name})",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "effect_size": effect_size,
                "confidence_intervals": confidence_intervals,
                "sample_sizes": [len(sample1), len(sample2)]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Mann-Whitney test: {e}")
            return {
                "test": f"Mann-Whitney U ({test_name})",
                "error": str(e),
                "significant": False
            }
    
    def _calculate_rank_biserial_correlation(self, sample1: np.ndarray, sample2: np.ndarray, u_statistic: float) -> Dict[str, Any]:
        """
        Calculate rank-biserial correlation as effect size for Mann-Whitney U test.
        
        Args:
            sample1: First sample
            sample2: Second sample
            u_statistic: U statistic from Mann-Whitney test
            
        Returns:
            Effect size information
        """
        try:
            n1, n2 = len(sample1), len(sample2)
            
            # Calculate rank-biserial correlation
            r = 1 - (2 * u_statistic) / (n1 * n2)
            
            # Interpret effect size
            abs_r = abs(r)
            if abs_r < 0.1:
                interpretation = "negligible"
            elif abs_r < 0.3:
                interpretation = "small"
            elif abs_r < 0.5:
                interpretation = "medium"
            else:
                interpretation = "large"
            
            return {
                "rank_biserial_correlation": float(r),
                "interpretation": interpretation,
                "type": "rank-biserial correlation (Mann-Whitney effect size)"
            }
            
        except Exception as e:
            logger.error(f"Error calculating rank-biserial correlation: {e}")
            return {"error": str(e)}
    
    def _calculate_median_confidence_intervals(self, sample1: np.ndarray, sample2: np.ndarray) -> Dict[str, Any]:
        """
        Calculate confidence intervals for medians using bootstrap.
        
        Args:
            sample1: First sample
            sample2: Second sample
            
        Returns:
            Confidence interval information
        """
        try:
            confidence_intervals = {}
            
            for i, sample in enumerate([sample1, sample2], 1):
                if len(sample) >= 10:  # Minimum for bootstrap
                    # Bootstrap confidence interval for median
                    bootstrap_medians = []
                    for _ in range(1000):
                        bootstrap_sample = resample(sample, n_samples=len(sample))
                        bootstrap_medians.append(np.median(bootstrap_sample))
                    
                    ci_lower = np.percentile(bootstrap_medians, 2.5)
                    ci_upper = np.percentile(bootstrap_medians, 97.5)
                    
                    confidence_intervals[f"sample_{i}"] = {
                        "median": float(np.median(sample)),
                        "ci_lower": float(ci_lower),
                        "ci_upper": float(ci_upper),
                        "method": "bootstrap"
                    }
                else:
                    confidence_intervals[f"sample_{i}"] = {
                        "median": float(np.median(sample)),
                        "method": "insufficient_data_for_ci"
                    }
            
            return confidence_intervals
            
        except Exception as e:
            logger.error(f"Error calculating median confidence intervals: {e}")
            return {"error": str(e)}
    
    def _apply_multiple_testing_correction(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply multiple testing correction to statistical test results.
        
        Args:
            results: Results dictionary containing statistical tests
            
        Returns:
            Updated results with corrected p-values
        """
        try:
            statistical_tests = results.get("statistical_tests", [])
            
            if not statistical_tests:
                return results
            
            # Extract p-values
            p_values = []
            for test in statistical_tests:
                if "p_value" in test and not test.get("error"):
                    p_values.append(test["p_value"])
                else:
                    p_values.append(1.0)  # Non-significant for missing/error cases
            
            if not p_values:
                return results
            
            # Apply Benjamini-Hochberg FDR correction
            rejected, corrected_p_values, _, _ = multipletests(
                p_values, alpha=0.05, method="fdr_bh"
            )
            
            # Update test results with corrected p-values
            for i, test in enumerate(statistical_tests):
                if i < len(corrected_p_values):
                    test["p_value_corrected"] = float(corrected_p_values[i])
                    test["significant_corrected"] = bool(rejected[i])
                    test["multiple_testing_method"] = "Benjamini-Hochberg FDR"
            
            # Add multiple testing summary
            results["multiple_testing_summary"] = {
                "method": "Benjamini-Hochberg FDR",
                "alpha_level": 0.05,
                "total_tests": len(p_values),
                "significant_before_correction": sum(p < 0.05 for p in p_values),
                "significant_after_correction": sum(rejected)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error applying multiple testing correction: {e}")
            return results


def get_causal_analyzer() -> CausalAnalyzer:
    """Get default causal analyzer instance."""
    return CausalAnalyzer()
