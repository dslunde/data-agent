"""
Command-line interface for the Data Agent.
"""

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv not available, skip

import os
import sys
import asyncio
import logging
import click
from typing import Optional
import json

# Setup logging - will be configured based on verbosity
logger = logging.getLogger(__name__)


def configure_logging(verbose: bool, debug: bool):
    """Configure logging levels and handlers based on verbosity."""
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    
    # Only log to file, not console unless debug mode
    handlers = [logging.FileHandler("data_agent.log")]
    
    if debug:
        # In debug mode, also log to console
        handlers.append(logging.StreamHandler(sys.stdout))
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True  # Override any existing configuration
    )


@click.command()
@click.option(
    "--data-path",
    type=click.Path(exists=True),
    help="Path to local dataset file (parquet, csv, xlsx, json)",
)
@click.option(
    "--download-url",
    type=str,
    help="URL to download dataset from (overrides default Google Drive URL)",
)
@click.option(
    "--provider",
    type=click.Choice(["openai", "anthropic", "auto"]),
    default="auto",
    help="LLM provider to use (default: auto)",
)
@click.option(
    "--model",
    type=str,
    help="Specific model to use (e.g., gpt-4o-mini, claude-3-haiku-20240307)",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--debug", is_flag=True, help="Enable debug mode with detailed logging")
@click.option(
    "--cache/--no-cache",
    default=True,
    help="Enable/disable result caching (default: enabled)",
)
@click.option(
    "--sample-size", type=int, help="Limit dataset to sample size for large datasets"
)
@click.option("--batch-mode", is_flag=True, help="Run in batch mode (non-interactive)")
@click.option("--query", type=str, help="Single query to execute in batch mode")
@click.option(
    "--output-format",
    type=click.Choice(["text", "json", "detailed"]),
    default="text",
    help="Output format for responses (default: text)",
)
@click.version_option(version="0.1.0", prog_name="data-agent")
def main(
    data_path: Optional[str],
    download_url: Optional[str],
    provider: str,
    model: Optional[str],
    verbose: bool,
    debug: bool,
    cache: bool,
    sample_size: Optional[int],
    batch_mode: bool,
    query: Optional[str],
    output_format: str,
):
    """
    Data Agent - Natural language interface for dataset analysis.

    Analyze datasets through natural language queries with advanced analytics
    including pattern recognition, anomaly detection, and causal analysis.

    Examples:

    \b
    # Interactive mode with auto-downloaded dataset
    python -m data_agent

    \b
    # Use local dataset file
    python -m data_agent --data-path /path/to/dataset.parquet

    \b
    # Batch mode with single query
    python -m data_agent --batch-mode --query "Show me basic statistics"

    \b
    # Use specific LLM provider
    python -m data_agent --provider openai --model gpt-4
    """

    # Configure logging properly
    configure_logging(verbose, debug)

    # Validate batch mode
    if batch_mode and not query:
        click.echo("Error: --query is required when using --batch-mode", err=True)
        sys.exit(1)

    # Welcome message
    if not batch_mode:
        click.echo("Welcome to Data Agent!")
        click.echo("Natural language interface for dataset analysis\n")

    try:
        # Run the application
        asyncio.run(
            run_data_agent(
                data_path=data_path,
                download_url=download_url,
                provider=provider,
                model=model,
                cache_enabled=cache,
                sample_size=sample_size,
                batch_mode=batch_mode,
                single_query=query,
                output_format=output_format,
                verbose=verbose,
            )
        )

    except KeyboardInterrupt:
        click.echo("\nThanks for using Data Agent!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application error: {e}")
        click.echo(f"Error: {e}", err=True)
        if debug:
            import traceback

            traceback.print_exc()
        sys.exit(1)


async def run_data_agent(
    data_path: Optional[str],
    download_url: Optional[str],
    provider: str,
    model: Optional[str],
    cache_enabled: bool,
    sample_size: Optional[int],
    batch_mode: bool,
    single_query: Optional[str],
    output_format: str,
    verbose: bool,
):
    """Main application logic."""

    from .core import DataAgentCore
    from .llm.clients import test_llm_connectivity

    # Test LLM connectivity first
    if verbose:
        click.echo("Testing LLM connectivity...")

    connectivity = test_llm_connectivity()
    available_providers = [p for p, info in connectivity.items() if info["available"]]

    if not available_providers:
        click.echo("No LLM providers available. Please check your API keys:", err=True)
        for provider_name, info in connectivity.items():
            click.echo(f"  - {provider_name}: {info['error']}", err=True)
        click.echo("\nSet environment variables:")
        click.echo("  export OPENAI_API_KEY=your_key_here")
        click.echo("  export ANTHROPIC_API_KEY=your_key_here")
        return

    if verbose:
        click.echo(f"Available providers: {', '.join(available_providers)}")

    # Initialize the core application
    try:
        # Add loading message for initialization
        if not batch_mode:
            click.echo("🔄 Initializing Data Agent... this may take a moment...")
        
        core = DataAgentCore(
            preferred_provider=provider,
            model=model,
            cache_enabled=cache_enabled,
            verbose=verbose,
        )

        # Load dataset with enhanced progress messages
        await load_dataset(core, data_path, download_url, sample_size, verbose, batch_mode)

        if batch_mode:
            # Run single query
            await run_batch_query(core, single_query, output_format, verbose)
        else:
            # Run interactive mode
            await run_interactive_mode(core, output_format, verbose)

    except Exception as e:
        logger.error(f"Core application error: {e}")
        raise


async def load_dataset(
    core,
    data_path: Optional[str],
    download_url: Optional[str],
    sample_size: Optional[int],
    verbose: bool,
    batch_mode: bool = False,
):
    """Load dataset into the core application."""

    try:
        # Add progress messages for better user experience
        if not batch_mode:
            if data_path:
                click.echo(f"📂 Loading dataset from {data_path}...")
                click.echo("⚙️ Processing and optimizing dataset... please wait...")
            else:
                click.echo("📥 Downloading and processing dataset...")
                click.echo("📊 Applying dataset optimizations... this may take up to 30 seconds...")
        
        if verbose:
            with click.progressbar(length=100, label="Loading dataset") as bar:
                # Simulate progress for user feedback
                bar.update(20)

                if data_path:
                    # Load local file
                    await core.load_local_dataset(data_path, sample_size)
                    bar.update(80)
                else:
                    # Download dataset
                    bar.update(40)
                    await core.download_and_load_dataset(download_url, sample_size)
                    bar.update(80)

                bar.update(20)  # Complete
        else:
            # Silent loading for clean output
            if data_path:
                await core.load_local_dataset(data_path, sample_size)
            else:
                await core.download_and_load_dataset(download_url, sample_size)

        # Add completion message
        if not batch_mode:
            click.echo("✅ Data Agent ready! You can now ask questions about your data.")
            click.echo()

        # Show dataset info based on verbosity
        dataset_info = core.get_dataset_info()

        if verbose:
            click.echo("Dataset loaded successfully!")
            click.echo(
                f"   Shape: {dataset_info['shape'][0]:,} rows x {dataset_info['shape'][1]} columns"
            )
            click.echo(f"   💾 Memory: {dataset_info['memory_usage_mb']:.1f} MB")
            click.echo(f"   Completeness: {dataset_info['completeness_score']:.1f}%")
            click.echo(f"   Columns: {', '.join(dataset_info['columns'][:5])}")
            if len(dataset_info["columns"]) > 5:
                click.echo(
                    f"              ... and {len(dataset_info['columns']) - 5} more"
                )
            click.echo()
        else:
            # Clean, minimal output
            click.echo(f"📊 Dataset: {dataset_info['shape'][0]:,} rows x {dataset_info['shape'][1]} columns")

    except Exception as e:
        click.echo(f"Failed to load dataset: {e}", err=True)
        raise


async def run_batch_query(core, query: str, output_format: str, verbose: bool = False):
    """Run a single query in batch mode."""

    if verbose:
        with click.progressbar(length=100, label="Analyzing") as bar:
            bar.update(20)
            response = await core.process_query(query)
            bar.update(80)
    else:
        # Clean output without progress bar
        response = await core.process_query(query)

    # Format and display response
    format_and_display_response(response, output_format)


async def run_interactive_mode(core, output_format: str, verbose: bool):
    """Run interactive query mode."""

    # Show help
    show_interactive_help()

    query_count = 0

    while True:
        try:
            # Get user input
            query = click.prompt(
                click.style("Ask me anything about your data", fg="cyan"), type=str
            ).strip()

            # Handle special commands
            if query.lower() in ["exit", "quit", "q"]:
                click.echo("Thanks for using Data Agent!")
                break

            elif query.lower() in ["help", "h", "?"]:
                show_interactive_help()
                continue

            elif query.lower() in ["info", "dataset", "schema"]:
                show_dataset_info(core)
                continue

            elif query.lower() in ["examples", "ex"]:
                show_example_queries()
                continue

            elif query.lower() in ["clear", "cls"]:
                os.system("clear" if os.name != "nt" else "cls")
                continue

            elif query.lower().startswith("set "):
                handle_settings_command(query, core, output_format)
                continue

            # Process data query
            if query:
                query_count += 1
                click.echo(f"\nQuery #{query_count}: {query}")

                try:
                    with click.progressbar(length=100, label="Analyzing") as bar:
                        bar.update(25)
                        response = await core.process_query(query)
                        bar.update(75)

                    click.echo()
                    format_and_display_response(response, output_format)

                except Exception as e:
                    click.echo(f"Query failed: {e}", err=True)
                    if verbose:
                        import traceback

                        traceback.print_exc()

                click.echo()

        except (EOFError, KeyboardInterrupt):
            click.echo("\nThanks for using Data Agent!")
            break
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            continue


def show_interactive_help():
    """Show interactive mode help."""
    click.echo("\nAvailable commands:")
    click.echo("  • Ask any question about your data in natural language")
    click.echo("  • help, h, ?     - Show this help")
    click.echo("  • info, dataset  - Show dataset information")
    click.echo("  • examples, ex   - Show example queries")
    click.echo("  • clear, cls     - Clear screen")
    click.echo("  • exit, quit, q  - Exit the application")
    click.echo("\nTry asking things like:")
    click.echo('  • "Show me basic statistics"')
    click.echo('  • "Find correlations between variables"')
    click.echo('  • "Detect outliers in the data"')
    click.echo('  • "What are the trends over time?"')
    click.echo()


def show_dataset_info(core):
    """Show detailed dataset information."""
    try:
        info = core.get_dataset_info()

        click.echo("\nDataset Information:")
        click.echo(f"   Shape: {info['shape'][0]:,} rows x {info['shape'][1]} columns")
        click.echo(f"   Memory: {info['memory_usage_mb']:.1f} MB")
        click.echo(f"   Completeness: {info['completeness_score']:.1f}%")

        click.echo(f"\nColumns ({len(info['columns'])}):")
        for i, col in enumerate(info["columns"]):
            if i < 20:  # Show first 20 columns
                dtype = info.get("column_types", {}).get(col, "unknown")
                click.echo(f"     {col} ({dtype})")
            elif i == 20:
                click.echo(f"     ... and {len(info['columns']) - 20} more columns")
                break

        click.echo()
    except Exception as e:
        click.echo(f"Error getting dataset info: {e}", err=True)


def show_example_queries():
    """Show example queries."""
    click.echo("\nExample queries you can try:")
    click.echo("\nBasic Analysis:")
    click.echo('  • "Describe the dataset"')
    click.echo('  • "Show summary statistics"')
    click.echo('  • "What are the column types?"')

    click.echo("\nExploratory Analysis:")
    click.echo('  • "Find correlations between variables"')
    click.echo('  • "Group customers into clusters"')
    click.echo('  • "Show the distribution of prices"')

    click.echo("\nAnomaly Detection:")
    click.echo('  • "Find outliers in sales data"')
    click.echo('  • "Detect unusual patterns"')
    click.echo('  • "What data points look strange?"')

    click.echo("\nComparative Analysis:")
    click.echo('  • "Compare performance by region"')
    click.echo('  • "Show differences between groups"')
    click.echo('  • "Analyze segments separately"')

    click.echo("\n⏰ Time Series:")
    click.echo('  • "What are the trends over time?"')
    click.echo('  • "Show seasonal patterns"')
    click.echo('  • "Analyze growth rates"')
    click.echo()


def handle_settings_command(command: str, core, output_format: str):
    """Handle settings commands."""
    # Simple settings commands
    if "format" in command.lower():
        click.echo(f"Current output format: {output_format}")
        click.echo("Available formats: text, json, detailed")
    else:
        click.echo("Available settings commands:")
        click.echo("  • set format - Show output format options")


def format_and_display_response(response: dict, output_format: str):
    """Format and display the response based on output format."""

    if output_format == "json":
        click.echo(json.dumps(response, indent=2, default=str))
        return

    # Text format (default)
    if "error" in response:
        # Display error with better formatting
        error_type = response.get("error_type", "UNKNOWN_ERROR")
        error_msg = response.get("error", "Unknown error occurred")
        
        click.echo(f"❌ {error_type.replace('_', ' ').title()}")
        click.echo(f"   {error_msg}")
        
        # Show suggestions if available
        suggestions = response.get("suggestions", [])
        if suggestions:
            click.echo("\n💡 Suggestions:")
            for suggestion in suggestions:
                click.echo(f"   • {suggestion}")
        
        # Show analysis results if available (for partial failures)
        if "analysis_results" in response:
            click.echo("\n📊 Raw Analysis Results:")
            click.echo(json.dumps(response["analysis_results"], indent=2, default=str))
        
        return

    # Main response
    main_response = response.get("response", "No response generated")
    click.echo("Analysis Result:")
    click.echo(f"   {main_response}")

    if output_format == "detailed":
        # Show additional details

        # Evidence
        evidence = response.get("evidence", {})
        if evidence:
            click.echo("\nSupporting Evidence:")
            if evidence.get("data_points_analyzed"):
                click.echo(f"   • Data points: {evidence['data_points_analyzed']:,}")
            if evidence.get("columns_used"):
                click.echo(f"   • Columns used: {', '.join(evidence['columns_used'])}")

            stats = evidence.get("statistical_measures", {})
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    if isinstance(value, float):
                        click.echo(f"   • {key.replace('_', ' ').title()}: {value:.3f}")
                    else:
                        click.echo(f"   • {key.replace('_', ' ').title()}: {value:,}")

        # Methodology
        methodology = response.get("methodology", {})
        if methodology and methodology.get("approach"):
            click.echo(f"\nMethodology: {methodology['approach']}")

        # Caveats
        caveats = response.get("caveats", [])
        if caveats:
            click.echo("\nImportant Notes:")
            for caveat in caveats[:3]:  # Show top 3 caveats
                click.echo(f"   • {caveat}")

        # Confidence
        confidence = response.get("confidence", 0)
        if confidence:
            confidence_pct = confidence * 100
            confidence_emoji = (
                "HIGH" if confidence > 0.8 else "MEDIUM" if confidence > 0.5 else "LOW"
            )
            click.echo(f"\n{confidence_emoji} Confidence: {confidence_pct:.0f}%")


# Additional CLI utility functions expected by tests
def format_response(response: dict, output_format: str) -> str:
    """
    Format response for display (wrapper for compatibility).
    
    Args:
        response: Response dictionary
        output_format: Format type ('text', 'json', 'detailed')
        
    Returns:
        Formatted response string
    """
    if output_format == "json":
        return json.dumps(response, indent=2, default=str)
    
    if "error" in response:
        return f"Error: {response['error']}"
    
    # Text format
    main_response = response.get("response", "No response generated")
    result = f"Analysis Result:\n   {main_response}"
    
    if output_format == "detailed":
        # Evidence
        evidence = response.get("evidence", {})
        if evidence:
            result += "\n\nSupporting Evidence:"
            if evidence.get("data_points_analyzed"):
                result += f"\n   • Data points: {evidence['data_points_analyzed']:,}"
            if evidence.get("columns_used"):
                result += f"\n   • Columns used: {', '.join(evidence['columns_used'])}"
        
        # Methodology
        methodology = response.get("methodology", {})
        if methodology:
            result += "\n\nMethodology:"
            if methodology.get("approach"):
                result += f"\n   • Approach: {methodology['approach']}"
            if methodology.get("parameters"):
                for key, value in methodology["parameters"].items():
                    result += f"\n   • {key.replace('_', ' ').title()}: {value}"
        
        # Important Notes - placeholder for any caveats or limitations
        result += "\n\nImportant Notes:"
        result += "\n   • Results are based on available data and selected methodology"
        if response.get("analysis_method"):
            result += f"\n   • Analysis method: {response['analysis_method']}"
        
        # Confidence
        confidence = response.get("confidence", 0)
        if confidence:
            confidence_pct = confidence * 100
            confidence_level = "HIGH" if confidence > 0.8 else "MEDIUM" if confidence > 0.5 else "LOW"
            result += f"\n\nConfidence: {confidence_level} ({confidence_pct:.0f}%)"
    
    return result


def get_confidence_indicator(confidence: float) -> str:
    """
    Get confidence level indicator.
    
    Args:
        confidence: Confidence score (0-1)
        
    Returns:
        Confidence indicator string
    """
    if confidence > 0.8:
        return "HIGH"
    elif confidence > 0.5:
        return "MEDIUM"
    else:
        return "LOW"


def display_dataset_info(dataset_info: dict, verbose: bool = False) -> None:
    """
    Display dataset information using click.echo.
    
    Args:
        dataset_info: Dataset information dictionary
        verbose: Whether to show detailed information
    """
    if not dataset_info or "error" in dataset_info:
        click.echo("Dataset information not available")
        return
    
    click.echo("Dataset Information:")
    
    # Basic info
    if "shape" in dataset_info:
        rows, cols = dataset_info["shape"]
        click.echo(f"   • Size: {rows:,} rows x {cols} columns")
    
    if "memory_usage_mb" in dataset_info:
        click.echo(f"   • Memory: {dataset_info['memory_usage_mb']:.2f} MB")
    
    if "completeness_score" in dataset_info:
        click.echo(f"   • Completeness: {dataset_info['completeness_score']:.1f}%")
    
    # Column types
    if "dtypes" in dataset_info:
        dtypes = dataset_info["dtypes"]
        type_counts = {}
        for dtype in dtypes.values():
            type_counts[dtype] = type_counts.get(dtype, 0) + 1
        
        type_strs = [f"{count} {dtype}" for dtype, count in type_counts.items()]
        click.echo(f"   • Column Types: {', '.join(type_strs)}")
    
    # Missing data
    if "missing_percentages" in dataset_info:
        missing = dataset_info["missing_percentages"]
        cols_with_missing = [(col, pct) for col, pct in missing.items() if pct > 0]
        if cols_with_missing:
            click.echo(f"   • Missing Data: {len(cols_with_missing)} columns with missing values")
    
    if verbose and "columns" in dataset_info:
        columns = dataset_info["columns"]
        click.echo(f"   • Columns: {', '.join(columns[:5])}")
        if len(columns) > 5:
            click.echo(f"            ... and {len(columns) - 5} more")


if __name__ == "__main__":
    main()
