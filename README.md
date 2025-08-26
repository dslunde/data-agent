# Data Agent

A Python CLI chat agent that analyzes datasets and answers user questions through natural language interaction. The agent handles everything from simple data retrieval to advanced analytics including pattern recognition, anomaly detection, and causal hypothesis generation.

## Features

- **Natural Language Queries**: Ask questions about your data in plain English
- **Advanced Analytics**: Pattern recognition, clustering, correlations, trend analysis
- **Anomaly Detection**: Statistical outliers, rule violations, and anomalies
- **Causal Analysis**: Evidence-backed causal hypothesis generation with caveats
- **Multiple Data Sources**: Local files or automatic dataset download
- **Schema Inference**: Automatic type detection and missing value handling
- **Performance Optimized**: Caching and efficient processing for fast responses

## Installation

### Prerequisites

- Python 3.10 or higher
- LLM API keys (OpenAI and/or Anthropic)

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd data-agent
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   
   # On macOS/Linux:
   source venv/bin/activate
   
   # On Windows:
   # venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   export OPENAI_API_KEY="your_openai_key_here"
   export ANTHROPIC_API_KEY="your_anthropic_key_here"
   ```
   
   Or create a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_key_here
   ANTHROPIC_API_KEY=your_anthropic_key_here
   ```

## Usage

### Basic Usage

```bash
# Run with automatic dataset download
python -m data_agent

# Run with local dataset
python -m data_agent --data-path /path/to/your/dataset.parquet

# Run with verbose output
python -m data_agent --verbose
```

### Dataset Options

**Option 1: Automatic Download (Recommended)**
The agent will automatically download the dataset from the provided Google Drive link on first run.

**Option 2: Local Dataset**
Provide a path to your own parquet file:
```bash
python -m data_agent --data-path /path/to/dataset.parquet
```

### Example Queries

Once running, you can ask questions like:

#### Basic Data Retrieval
- "How many records are in this dataset?"
- "What are the column names and types?"
- "Show me the first 5 rows"
- "What's the average value of column X?"

#### Pattern Recognition
- "Find clusters in the data based on customer behavior"
- "What correlations exist between sales and marketing spend?"
- "Show me trends in revenue over time"
- "Are there any seasonal patterns?"

#### Anomaly Detection
- "Find outliers in the sales data"
- "Detect unusual patterns in customer transactions"
- "What data points violate typical business rules?"
- "Identify anomalous time periods"

#### Causal Analysis
- "What factors might cause high customer churn?"
- "Analyze potential reasons for sales spikes"
- "What could explain the correlation between X and Y?"

### Interactive Mode

The agent runs in interactive mode by default. Type your questions and receive detailed analyses with supporting evidence and methodology explanations.

Type `exit`, `quit`, or `q` to end the session.

## Development

### Development Commands

```bash
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
```

### Project Structure

```
data-agent/
├── data_agent/           # Main package
│   ├── __init__.py
│   ├── cli.py           # CLI interface
│   ├── data/            # Data loading and processing
│   ├── analysis/        # Statistical analysis engines
│   ├── llm/            # LLM integration
│   └── utils/          # Utilities
├── tests/              # Test suite
├── data/              # Runtime dataset storage (gitignored)
├── docs/              # Documentation
├── requirements.txt   # Dependencies
└── README.md         # This file
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `DATA_AGENT_CACHE_DIR`: Custom cache directory (optional)
- `DATA_AGENT_LOG_LEVEL`: Log level (DEBUG, INFO, WARNING, ERROR)

### Performance Tuning

For large datasets, consider:
- Using the `--cache` option to enable aggressive caching
- Setting `DATA_AGENT_CACHE_DIR` to a fast disk location
- Using the `--sample` option to work with data samples during exploration

## Assumptions and Limitations

### Data Assumptions
- Dataset is provided as a parquet file or convertible format
- Data fits in memory (for large datasets, sampling is used)
- Column names are meaningful and well-structured

### Analysis Limitations
- Causal analysis provides hypotheses, not definitive causal claims
- Statistical tests assume standard assumptions (normality, independence, etc.)
- Complex domain-specific business logic may require additional context
- LLM responses are generated based on statistical evidence but may require expert validation

### Technical Limitations
- Requires internet connection for LLM API calls
- Performance depends on dataset size and query complexity
- Some analyses may take several seconds for large datasets

## Troubleshooting

### Common Issues

**"API key not found" error:**
- Verify your API keys are set as environment variables
- Check that the keys are valid and have sufficient credits

**"Dataset not found" error:**
- Ensure the dataset path is correct
- Check that automatic download completed successfully
- Verify the dataset is a valid parquet file

**Slow performance:**
- Enable caching with `--cache`
- Use data sampling for exploration with `--sample`
- Check your internet connection for LLM API calls

### Getting Help

- Use `python -m data_agent --help` for command-line options
- Enable verbose mode with `--verbose` for detailed logging
- Check the logs in `data_agent.log` for debugging information

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code passes linting and type checks
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
