# Natural Gas Pipeline Dataset Profile

## Dataset Overview
- **Domain**: Natural gas pipeline transportation and scheduling
- **Size**: 23,854,855 records (~23.8M rows)
- **Dimensions**: 13 columns
- **Memory**: 594 MB optimized (likely ~2GB unoptimized)
- **Time Period**: 2022-01-01 to 2025-08-26 (1,334 days, ~3.7 years)
- **Data Source**: Likely FERC or EIA regulatory reporting

## Business Context
This dataset represents **scheduled gas transportation** across the US pipeline network:
- **Receipts** (+1): Gas flowing INTO pipeline locations
- **Deliveries** (-1): Gas flowing OUT OF pipeline locations  
- **Scheduled Quantity**: Volume in standard units (likely Mcf or MMcf)

## Key Dimensions & Cardinalities

### Pipeline Infrastructure
- **169 unique pipeline companies** (Northern Natural Gas is largest)
- **17,528 unique locations** across the network
- **1,145 counties** in **48 states** (Louisiana, Texas, Pennsylvania lead)

### Business Categories (by volume)
1. **LDC** (7.68M records) - Local Distribution Companies
2. **Production** (5.98M) - Gas production facilities  
3. **Interconnect** (4.72M) - Pipeline interconnections
4. **Industrial** (2.26M) - Industrial customers
5. **Power** (1.13M) - Power generation facilities
6. **Storage** (0.79M) - Gas storage facilities

### Geographic Concentration
**Top 5 states by activity:**
1. Louisiana (3.19M records) - Gulf Coast hub
2. Texas (2.25M records) - Major production state
3. Pennsylvania (1.66M records) - Marcellus shale region
4. Kansas (1.02M records) - Central hub
5. Oklahoma (0.94M records) - Production state

## Data Quality Insights

### Strengths
- **Well-structured categorical data** (optimized memory usage)
- **Complete temporal coverage** (no gaps in daily reporting)
- **Rich geographic metadata** (state, county, coordinates)
- **Clear business semantics** (receipt/delivery flags)

### Challenges
- **48% zero/null quantities** (scheduled vs actual reporting)
- **High cardinality in locations** (17K+ unique)
- **Temporal data as categories** (needs conversion for analysis)
- **Coordinate data as categories** (needs numeric conversion)

## Performance Optimization Opportunities

### 1. Caching Strategy
- **Pre-compute state-level aggregations** (top 10 states = 70% of data)
- **Cache pipeline-level statistics** (top 10 pipelines = 50% of data)  
- **Store monthly/seasonal patterns** for time series queries

### 2. Query Optimization Patterns
- **Filter by state first** (reduces data by 10-90%)
- **Filter by major pipelines** (top 20 = 80% of records)
- **Pre-aggregate zero vs non-zero quantities**
- **Index on date ranges** for temporal analysis

### 3. Analysis-Specific Optimizations
- **Clustering**: Focus on geographic clusters (state/county)
- **Anomaly Detection**: Optimize for quantity outliers and pipeline disruptions
- **Correlation**: Pre-compute pipeline-to-pipeline relationships
- **Time Series**: Monthly/seasonal patterns in gas flow

## Key Business Questions Dataset Can Answer

### Operational Analytics
- "Which pipelines have the highest capacity utilization?"
- "What are seasonal patterns in gas flow to power plants?"
- "Where are the major pipeline bottlenecks?"

### Geographic Analysis  
- "How does gas flow from production areas to consumption centers?"
- "Which states are net importers vs exporters of gas?"
- "Where are the critical pipeline interconnection points?"

### Anomaly Detection
- "Which scheduled flows are unusually high or low?"
- "Are there pipeline disruptions or maintenance patterns?"
- "Which locations show irregular flow patterns?"

### Network Analysis
- "How interconnected is the pipeline network?"
- "What are the critical chokepoints in gas transportation?"
- "Which pipeline companies dominate specific regions?"

## Recommended Query Patterns for Evaluation

### High-Performance Queries (< 2 seconds)
1. "Show gas flow patterns for Texas pipelines"
2. "Find the largest LDC deliveries by state"  
3. "What are seasonal trends in power plant gas demand?"

### Complex Analytics (2-10 seconds)
4. "Cluster pipeline locations by flow similarity"
5. "Detect unusual scheduling patterns in the Gulf Coast"
6. "Analyze pipeline network connectivity"

### Advanced Insights (10+ seconds)
7. "Identify potential pipeline capacity constraints"
8. "Find correlations between production and delivery regions"
9. "Detect seasonal storage injection/withdrawal patterns"

## Dataset-Specific Optimizations Implemented

### Memory Optimizations
- Categorical encoding for high-cardinality strings
- Int8 for receipt/delivery flags  
- Float64 only where needed for quantities

### Computation Optimizations
- State-based filtering reduces dataset by 90%
- Date range filtering for temporal analysis
- Pipeline-based filtering for network analysis
- Non-zero quantity filtering for volume analysis

This dataset represents a rich, realistic business scenario perfect for demonstrating advanced analytics capabilities on infrastructure and energy data.