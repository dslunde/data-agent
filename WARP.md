# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a Python-based CLI chat agent that analyzes datasets and answers user questions through natural language interaction. The agent handles everything from simple data retrieval to advanced analytics including pattern recognition, anomaly detection, and causal hypothesis generation.

**Key Requirements:**
- CLI-only interface (no GUI or web API required)
- Python 3.10+ compatible
- Dataset ingestion with schema inference and missing value handling
- Natural language query understanding
- Advanced analytics capabilities (clustering, correlations, trends, anomalies)
- Evidence-backed responses with methodology transparency

## Development Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables for LLM APIs
export OPENAI_API_KEY="your_key_here"
export ANTHROPIC_API_KEY="your_key_here"
```

## Dataset Handling

**Important**: Do not commit dataset files to the repository. The code should:
- Prompt for local dataset path, OR
- Download from provided URL at runtime and store under `./data/` (ensure it's in .gitignore)
- Handle various data formats with automatic schema inference
- Gracefully handle missing values and data quality issues

## Core Architecture

The agent should be organized around these key components:

- **Data Ingestion Layer**: Schema inference, type detection, missing value handling
- **Query Understanding**: Natural language processing and intent recognition
- **Analysis Engine**: Statistical analysis, pattern recognition, anomaly detection
- **LLM Integration**: OpenAI/Anthropic API integration for query understanding and response generation
- **Evidence Generation**: Methodology tracking and supporting evidence compilation
- **CLI Interface**: User interaction and response formatting

## Common Development Commands

```bash
# Run the agent
python -m data_agent

# Run with specific dataset path
python -m data_agent --data-path /path/to/dataset.parquet

# Run tests
python -m pytest

# Run tests with coverage
python -m pytest --cov=data_agent

# Format code
black .

# Lint code
ruff check .
ruff format .

# Type checking
mypy data_agent/

# Run a single test file
python -m pytest tests/test_analysis.py

# Run specific test
python -m pytest tests/test_analysis.py::test_anomaly_detection
```

## Technology Stack

**Required Libraries:**
- **Data Processing**: pandas, numpy, scipy for core data analysis
- **LLM Integration**: openai, anthropic for API access
- **Statistical Analysis**: scikit-learn for ML/pattern recognition
- **CLI**: click or argparse for command-line interface
- **Data Validation**: pydantic for data models
- **Testing**: pytest, pytest-cov for testing framework

**Optional but Recommended:**
- **Advanced Analytics**: plotly/matplotlib for data visualization
- **Performance**: polars or pyarrow for large dataset handling
- **Async**: asyncio/aiohttp for concurrent API calls
- **Logging**: structlog or loguru for comprehensive logging

## Performance Considerations

- Optimize for query response speed (30% of evaluation score)
- Implement caching for repeated analyses
- Use efficient data structures for large datasets
- Consider lazy evaluation for expensive computations
- Profile and optimize bottlenecks

## Evaluation Criteria

The agent will be evaluated on:
- **Accuracy (70%)**: Correct numbers, sound methodology, evidence-backed hypotheses
- **Speed (30%)**: Lower latency per query
- **Bonus**: Insightful findings, data quality detection, business interpretation

## Development Best Practices

- Implement comprehensive error handling for data ingestion failures
- Log analysis steps for debugging and transparency
- Write unit tests for core analysis functions
- Document assumptions and limitations clearly
- Validate statistical methods and interpretations
- Handle edge cases in data (empty datasets, single values, etc.)
- Implement proper API key management and error handling
