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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("data_agent.log"), logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


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

    # Configure logging
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        verbose = True
    elif verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)

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
        core = DataAgentCore(
            preferred_provider=provider,
            model=model,
            cache_enabled=cache_enabled,
            verbose=verbose,
        )

        # Load dataset
        await load_dataset(core, data_path, download_url, sample_size, verbose)

        if batch_mode:
            # Run single query
            await run_batch_query(core, single_query, output_format)
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
):
    """Load dataset into the core application."""

    if verbose:
        click.echo("Loading dataset...")

    try:
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

        # Show dataset info
        dataset_info = core.get_dataset_info()

        click.echo("Dataset loaded successfully!")
        click.echo(
            f"   Shape: {dataset_info['shape'][0]:,} rows x {dataset_info['shape'][1]} columns"
        )
        click.echo(f"   💾 Memory: {dataset_info['memory_usage_mb']:.1f} MB")
        click.echo(f"   Completeness: {dataset_info['completeness_score']:.1f}%")

        if verbose:
            click.echo(f"   Columns: {', '.join(dataset_info['columns'][:5])}")
            if len(dataset_info["columns"]) > 5:
                click.echo(
                    f"              ... and {len(dataset_info['columns']) - 5} more"
                )

        click.echo()

    except Exception as e:
        click.echo(f"Failed to load dataset: {e}", err=True)
        raise


async def run_batch_query(core, query: str, output_format: str):
    """Run a single query in batch mode."""

    click.echo(f"Processing query: {query}")

    try:
        with click.progressbar(length=100, label="Analyzing") as bar:
            bar.update(20)
            response = await core.process_query(query)
            bar.update(80)

        # Format and display response
        format_and_display_response(response, output_format)

    except Exception as e:
        click.echo(f"Query failed: {e}", err=True)
        raise


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
        click.echo(f"Error: {response['error']}")
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


if __name__ == "__main__":
    main()
