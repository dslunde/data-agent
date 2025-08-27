"""
Advanced causal analysis for pipeline dataset.
"""

import pandas as pd
from typing import Dict, Any
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class CausalAnalyzer:
    """Advanced causal analysis for natural gas pipeline data."""

    def __init__(self):
        """Initialize causal analyzer."""
        pass

    def analyze_pipeline_capacity_drivers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze what drives pipeline capacity utilization."""
        results = {
            "analysis_type": "capacity_drivers",
            "methodology": "Multi-factor causal analysis",
            "findings": [],
            "statistical_tests": [],
            "confidence": "medium",
            "caveats": [],
        }

        try:
            # Focus on non-zero quantities
            active_flows = df[df["scheduled_quantity"] > 0].copy()

            if len(active_flows) == 0:
                results["error"] = "No active flows found"
                return results

            # Test geographic influence
            if "state_abb" in active_flows.columns:
                state_groups = active_flows.groupby("state_abb")[
                    "scheduled_quantity"
                ].agg(["mean", "count"])
                state_groups = state_groups[
                    state_groups["count"] >= 100
                ]  # Minimum sample size

                if len(state_groups) > 1:
                    # ANOVA test for state differences
                    state_samples = []
                    for state in state_groups.index:
                        state_data = active_flows[active_flows["state_abb"] == state][
                            "scheduled_quantity"
                        ]
                        if len(state_data) >= 100:  # Minimum sample size
                            state_samples.append(
                                state_data.sample(min(1000, len(state_data)))
                            )  # Sample for performance

                    if len(state_samples) >= 2:
                        f_stat, p_value = stats.f_oneway(*state_samples)
                        results["statistical_tests"].append(
                            {
                                "test": "ANOVA (State Effects)",
                                "f_statistic": float(f_stat),
                                "p_value": float(p_value),
                                "significant": p_value < 0.05,
                            }
                        )

                        if p_value < 0.05:
                            results["findings"].append(
                                {
                                    "factor": "Geographic Location (State)",
                                    "effect": "Significant regional differences in pipeline utilization",
                                    "evidence": f"F-statistic: {f_stat:.2f}, p-value: {p_value:.4f}",
                                    "business_impact": "Different states show distinct pipeline usage patterns",
                                }
                            )

            # Test business category influence
            if "category_short" in active_flows.columns:
                cat_groups = active_flows.groupby("category_short")[
                    "scheduled_quantity"
                ].agg(["mean", "count"])
                cat_groups = cat_groups[cat_groups["count"] >= 100]

                if len(cat_groups) > 1:
                    # Test for category differences
                    cat_samples = []
                    for category in cat_groups.index:
                        cat_data = active_flows[
                            active_flows["category_short"] == category
                        ]["scheduled_quantity"]
                        if len(cat_data) >= 100:
                            cat_samples.append(
                                cat_data.sample(min(1000, len(cat_data)))
                            )

                    if len(cat_samples) >= 2:
                        f_stat, p_value = stats.f_oneway(*cat_samples)
                        results["statistical_tests"].append(
                            {
                                "test": "ANOVA (Category Effects)",
                                "f_statistic": float(f_stat),
                                "p_value": float(p_value),
                                "significant": p_value < 0.05,
                            }
                        )

                        if p_value < 0.05:
                            results["findings"].append(
                                {
                                    "factor": "Business Category",
                                    "effect": "Significant differences in volume by customer type",
                                    "evidence": f"F-statistic: {f_stat:.2f}, p-value: {p_value:.4f}",
                                    "business_impact": "Different customer categories drive distinct usage patterns",
                                }
                            )

            # Test receipt vs delivery patterns
            if "rec_del_sign" in active_flows.columns:
                receipts = active_flows[active_flows["rec_del_sign"] == 1][
                    "scheduled_quantity"
                ]
                deliveries = active_flows[active_flows["rec_del_sign"] == -1][
                    "scheduled_quantity"
                ]

                if len(receipts) >= 100 and len(deliveries) >= 100:
                    # Sample for performance
                    receipts_sample = receipts.sample(min(2000, len(receipts)))
                    deliveries_sample = deliveries.sample(min(2000, len(deliveries)))

                    # Mann-Whitney U test (non-parametric)
                    statistic, p_value = stats.mannwhitneyu(
                        receipts_sample, deliveries_sample, alternative="two-sided"
                    )
                    results["statistical_tests"].append(
                        {
                            "test": "Mann-Whitney U (Receipt vs Delivery)",
                            "statistic": float(statistic),
                            "p_value": float(p_value),
                            "significant": p_value < 0.05,
                        }
                    )

                    if p_value < 0.05:
                        receipt_median = receipts.median()
                        delivery_median = deliveries.median()
                        results["findings"].append(
                            {
                                "factor": "Flow Direction (Receipt vs Delivery)",
                                "effect": "Receipt volumes differ significantly from delivery volumes",
                                "evidence": f"Receipt median: {receipt_median:.0f}, Delivery median: {delivery_median:.0f}, p-value: {p_value:.4f}",
                                "business_impact": "Pipeline networks show asymmetric flow patterns",
                            }
                        )

            # Add business context
            results["business_context"] = {
                "domain": "Natural Gas Pipeline Transportation",
                "key_factors": "Geographic location, customer type, flow direction",
                "implications": "Understanding capacity drivers helps optimize infrastructure investment",
            }

            # Set confidence based on number of significant findings
            significant_tests = len(
                [t for t in results["statistical_tests"] if t.get("significant", False)]
            )
            if significant_tests >= 2:
                results["confidence"] = "high"
            elif significant_tests >= 1:
                results["confidence"] = "medium"
            else:
                results["confidence"] = "low"

            # Add appropriate caveats
            results["caveats"] = [
                "Analysis based on scheduled quantities, not actual flows",
                "Results may vary by time period and market conditions",
                "Geographic effects may be confounded with pipeline infrastructure",
                "Sample sizes vary across categories and may affect statistical power",
            ]

        except Exception as e:
            logger.error(f"Error in causal analysis: {e}")
            results["error"] = f"Analysis failed: {str(e)}"

        return results

    def detect_infrastructure_bottlenecks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify potential infrastructure bottlenecks in the pipeline network."""
        results = {
            "analysis_type": "bottleneck_detection",
            "methodology": "Network flow analysis with capacity constraints",
            "bottlenecks": [],
            "recommendations": [],
            "confidence": "medium",
        }

        try:
            # Focus on locations with both receipts and deliveries (interconnection points)
            location_flows = (
                df.groupby("loc_name")
                .agg(
                    {
                        "scheduled_quantity": ["sum", "count", "mean", "std"],
                        "rec_del_sign": lambda x: (x == 1).sum()
                        - (x == -1).sum(),  # Net flow
                    }
                )
                .round(2)
            )

            location_flows.columns = [
                "total_volume",
                "transaction_count",
                "avg_volume",
                "volume_std",
                "net_flow",
            ]
            location_flows = location_flows[
                location_flows["transaction_count"] >= 50
            ]  # Minimum activity

            if len(location_flows) == 0:
                results["error"] = "No locations with sufficient activity found"
                return results

            # Identify high-volume, high-variability locations (potential bottlenecks)
            volume_threshold = location_flows["total_volume"].quantile(
                0.8
            )  # Top 20% by volume
            variability_threshold = location_flows["volume_std"].quantile(
                0.8
            )  # Top 20% variability

            potential_bottlenecks = location_flows[
                (location_flows["total_volume"] >= volume_threshold)
                & (location_flows["volume_std"] >= variability_threshold)
            ].sort_values("total_volume", ascending=False)

            for loc_name, row in potential_bottlenecks.head(10).iterrows():
                # Get location details
                loc_data = df[df["loc_name"] == loc_name]
                states = loc_data["state_abb"].unique()
                pipelines = loc_data["pipeline_name"].unique()
                categories = loc_data["category_short"].value_counts()

                bottleneck = {
                    "location": loc_name,
                    "total_volume": float(row["total_volume"]),
                    "avg_volume": float(row["avg_volume"]),
                    "volume_variability": float(row["volume_std"]),
                    "net_flow": float(row["net_flow"]),
                    "transaction_count": int(row["transaction_count"]),
                    "states": states.tolist(),
                    "pipelines": pipelines.tolist()[:3],  # Top 3
                    "primary_categories": categories.head(3).to_dict(),
                    "bottleneck_score": float(
                        row["total_volume"] * row["volume_std"] / 1e6
                    ),  # Composite score
                }

                results["bottlenecks"].append(bottleneck)

            # Generate recommendations
            if len(results["bottlenecks"]) > 0:
                results["recommendations"] = [
                    "Monitor high-variability locations for capacity constraints",
                    "Consider infrastructure investments at identified bottlenecks",
                    "Implement dynamic scheduling to optimize flow through constrained points",
                    "Analyze seasonal patterns to predict capacity needs",
                ]

            results["summary"] = {
                "total_locations_analyzed": len(location_flows),
                "potential_bottlenecks_found": len(results["bottlenecks"]),
                "volume_threshold_used": float(volume_threshold),
                "variability_threshold_used": float(variability_threshold),
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
        }

        try:
            # Convert date column if needed
            if "eff_gas_day" in df.columns:
                if df["eff_gas_day"].dtype == "category":
                    df = df.copy()
                    df["eff_gas_day"] = pd.to_datetime(df["eff_gas_day"])

                df["month"] = df["eff_gas_day"].dt.month
                df["quarter"] = df["eff_gas_day"].dt.quarter

                # Monthly analysis for active flows
                active_flows = df[df["scheduled_quantity"] > 0]
                monthly_stats = (
                    active_flows.groupby("month")["scheduled_quantity"]
                    .agg(["mean", "count", "std"])
                    .round(2)
                )

                if len(monthly_stats) >= 12:  # Full year of data
                    # Test for seasonal differences
                    monthly_samples = []
                    for month in range(1, 13):
                        month_data = active_flows[active_flows["month"] == month][
                            "scheduled_quantity"
                        ]
                        if len(month_data) >= 100:
                            monthly_samples.append(
                                month_data.sample(min(1000, len(month_data)))
                            )

                    if len(monthly_samples) >= 12:
                        f_stat, p_value = stats.f_oneway(*monthly_samples)
                        results["statistical_tests"].append(
                            {
                                "test": "ANOVA (Monthly Seasonality)",
                                "f_statistic": float(f_stat),
                                "p_value": float(p_value),
                                "significant": p_value < 0.05,
                            }
                        )

                        if p_value < 0.05:
                            # Identify peak and low months
                            peak_month = monthly_stats["mean"].idxmax()
                            low_month = monthly_stats["mean"].idxmin()

                            month_names = {
                                1: "January",
                                2: "February",
                                3: "March",
                                4: "April",
                                5: "May",
                                6: "June",
                                7: "July",
                                8: "August",
                                9: "September",
                                10: "October",
                                11: "November",
                                12: "December",
                            }

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
                                    "evidence": f"F-statistic: {f_stat:.2f}, p-value: {p_value:.4f}",
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
            else:
                results["confidence"] = "low"

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


def get_causal_analyzer() -> CausalAnalyzer:
    """Get default causal analyzer instance."""
    return CausalAnalyzer()
