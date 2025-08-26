# CLAUDE.md

This file provides specific guidance for Claude AI assistant when working on the Data Agent project.

## Project Context

You are working on a Python-based CLI chat agent that analyzes datasets and answers user questions through natural language interaction. This is a technical assessment project with specific requirements and evaluation criteria.

**Critical Project Requirements:**
- CLI-only interface (no GUI/web API)
- Python 3.10+ compatibility
- OpenAI/Anthropic LLM integration only
- Dataset must NOT be committed to repository
- Due: Friday, August 29, 2025 at 5:00 PM CST

## Core Capabilities to Implement

The agent must handle:

1. **Data Ingestion**
   - Schema inference and type detection
   - Missing value handling strategies
   - Data quality validation
   - Support for dataset path input or runtime download

2. **Natural Language Understanding**
   - Parse user questions into actionable analysis plans
   - Distinguish between deterministic queries and analytical tasks
   - Handle ambiguous or incomplete requests

3. **Analysis Engine**
   - **Pattern Recognition**: Clustering, correlations, trend analysis
   - **Anomaly Detection**: Statistical outliers, rule violations
   - **Causal Hypothesis**: Evidence-backed relationship explanations
   - **Simple Retrieval**: Basic counting, filtering, aggregation

4. **Response Generation**
   - Concise answers with supporting evidence
   - Methodology transparency (methods used, columns selected, filters applied)
   - Appropriate caveats and limitations

## Technical Architecture Guidance

### Recommended Project Structure
```
data_agent/
├── __init__.py
├── cli.py              # Command-line interface
├── data_loader.py      # Dataset ingestion and schema inference
├── query_parser.py     # Natural language understanding
├── analysis_engine.py  # Statistical analysis and ML
├── llm_integration.py  # OpenAI/Anthropic API handling
├── response_builder.py # Evidence compilation and formatting
└── utils.py           # Common utilities and helpers

tests/
├── test_data_loader.py
├── test_analysis_engine.py
├── test_query_parser.py
└── fixtures/          # Test data samples
```

### Key Design Patterns

1. **Modular Architecture**: Separate concerns clearly between data processing, analysis, and LLM integration
2. **Strategy Pattern**: Use for different analysis types (pattern recognition, anomaly detection, etc.)
3. **Factory Pattern**: For creating appropriate analyzers based on query type
4. **Observer Pattern**: For progress tracking and logging during long-running analyses

## Implementation Priorities

### Phase 1: Core Infrastructure
- Data loading with schema inference
- Basic CLI interface with argument parsing
- LLM API integration (OpenAI/Anthropic)
- Simple query-response loop

### Phase 2: Analysis Capabilities
- Descriptive statistics and basic aggregations
- Pattern recognition (correlations, trends)
- Anomaly detection algorithms
- Statistical testing frameworks

### Phase 3: Advanced Features
- Causal hypothesis generation
- Complex multi-step analysis chains
- Performance optimization
- Comprehensive error handling

### Phase 4: Polish & Testing
- Unit test coverage
- Integration testing
- Performance benchmarking
- Documentation and examples

## Development Guidelines

### Code Quality Standards
- Use type hints throughout (Python 3.10+ features)
- Follow PEP 8 with black formatting
- Implement comprehensive error handling
- Add docstrings for all public methods
- Use dataclasses or Pydantic for data models

### Statistical Analysis Best Practices
- Validate assumptions before applying statistical tests
- Report confidence intervals and p-values where appropriate
- Handle small sample sizes gracefully
- Detect and report multicollinearity issues
- Implement proper cross-validation for ML models

### Performance Optimization
- Profile code to identify bottlenecks
- Use vectorized operations (numpy/pandas)
- Implement caching for expensive computations
- Consider lazy evaluation for large datasets
- Optimize LLM API calls (batching, async)

### Error Handling Strategy
- Graceful degradation when data is missing/corrupted
- Clear error messages for invalid user queries
- Fallback strategies when primary analysis fails
- API rate limiting and retry logic
- Memory management for large datasets

## LLM Integration Guidelines

### Query Understanding
- Use structured prompts with clear examples
- Implement few-shot learning for query classification
- Parse structured responses (JSON) for analysis parameters
- Handle ambiguous queries with clarifying questions

### Response Generation
- Template-based responses for consistency
- Include methodology explanations
- Provide appropriate statistical caveats
- Format numerical results clearly
- Suggest follow-up analyses when relevant

### API Best Practices
- Implement proper rate limiting
- Use environment variables for API keys
- Handle API errors gracefully
- Log API usage for debugging
- Consider cost optimization strategies

## Testing Strategy

### Unit Tests
- Test data loading with various file formats
- Validate statistical calculations
- Mock LLM responses for consistent testing
- Test edge cases (empty data, single values, etc.)

### Integration Tests
- End-to-end query processing
- LLM integration with real APIs
- Performance benchmarks with sample data
- Error handling scenarios

### Test Data
- Create synthetic datasets for testing
- Include edge cases (missing values, outliers)
- Test with different data types and distributions
- Validate against known statistical results

## Evaluation Preparation

### Accuracy Focus (70% weight)
- Implement robust statistical validation
- Test calculations against known benchmarks
- Ensure methodology soundness
- Provide proper evidence for causal claims

### Speed Optimization (30% weight)
- Profile and optimize critical paths
- Implement intelligent caching strategies
- Use efficient data structures
- Minimize LLM API calls where possible

### Bonus Opportunities
- Detect and report data quality issues
- Identify non-obvious patterns or segments
- Provide business-relevant interpretations
- Implement robustness checks for findings

## Common Pitfalls to Avoid

1. **Statistical Errors**
   - Correlation vs. causation confusion
   - Multiple testing without correction
   - Inappropriate statistical test selection
   - Ignoring sample size limitations

2. **Performance Issues**
   - Loading entire dataset into memory unnecessarily
   - Repeated expensive computations
   - Excessive LLM API calls
   - Inefficient data structures

3. **User Experience Problems**
   - Unclear error messages
   - Long waits without progress indicators
   - Inconsistent response formatting
   - Poor handling of ambiguous queries

4. **Data Handling Issues**
   - Not validating data quality
   - Poor missing value strategies
   - Incorrect data type inference
   - Memory leaks with large datasets

## Debugging and Development Tips

- Use structured logging for analysis steps
- Implement verbose mode for detailed output
- Create reproducible analysis pipelines
- Save intermediate results for debugging
- Use meaningful variable names and comments
- Profile memory usage with large datasets

## Success Metrics

- Accurate statistical results
- Fast query response times
- Robust error handling
- Clear, evidence-backed explanations
- Insightful pattern discovery
- Professional code quality
- Comprehensive test coverage

Remember: This is an assessment project. Focus on demonstrating technical competence, statistical knowledge, and software engineering best practices while meeting the specific requirements outlined in the PRD.
