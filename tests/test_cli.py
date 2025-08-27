"""
Unit tests for CLI interface functionality.
"""

import pytest
import tempfile
import json
import pandas as pd
from pathlib import Path
from unittest.mock import patch, Mock, AsyncMock
from click.testing import CliRunner
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_agent.cli import main, run_data_agent


class TestCLIArguments:
    """Test CLI argument parsing and validation."""

    def test_cli_help(self):
        """Test CLI help display."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "Data Agent" in result.output
        assert "--data-path" in result.output
        assert "--provider" in result.output
        assert "--verbose" in result.output

    def test_cli_version(self):
        """Test version display."""
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])

        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_batch_mode_requires_query(self):
        """Test that batch mode requires a query parameter."""
        runner = CliRunner()
        result = runner.invoke(main, ["--batch-mode"])

        assert result.exit_code == 1
        assert "Error: --query is required when using --batch-mode" in result.output

    @patch("data_agent.cli.run_data_agent")
    def test_cli_argument_parsing(self, mock_run_data_agent):
        """Test that CLI arguments are properly parsed."""
        runner = CliRunner()

        # Mock the async function to avoid actual execution
        async def mock_async_run(*args, **kwargs):
            pass

        mock_run_data_agent.return_value = None

        with patch("asyncio.run", side_effect=lambda x: None):
            result = runner.invoke(
                main,
                [
                    "--provider",
                    "anthropic",
                    "--verbose",
                    "--cache",
                    "--sample-size",
                    "1000",
                    "--output-format",
                    "json",
                ],
            )

        # Should not error on argument parsing
        assert result.exit_code == 0
        mock_run_data_agent.assert_called_once()

        # Check that arguments were passed correctly
        call_kwargs = mock_run_data_agent.call_args[1]
        assert call_kwargs["provider"] == "anthropic"
        assert call_kwargs["verbose"]
        assert call_kwargs["cache_enabled"]
        assert call_kwargs["sample_size"] == 1000
        assert call_kwargs["output_format"] == "json"

    def test_invalid_provider(self):
        """Test handling of invalid provider choice."""
        runner = CliRunner()
        result = runner.invoke(main, ["--provider", "invalid_provider"])

        assert result.exit_code == 2  # Click argument error
        assert "Invalid value for" in result.output

    def test_invalid_output_format(self):
        """Test handling of invalid output format."""
        runner = CliRunner()
        result = runner.invoke(main, ["--output-format", "invalid_format"])

        assert result.exit_code == 2
        assert "Invalid value for" in result.output


class TestCLIDataLoading:
    """Test CLI data loading functionality."""

    @pytest.mark.asyncio
    async def test_run_data_agent_with_local_file(self):
        """Test running data agent with local file."""
        # Create a temporary CSV file
        test_df = pd.DataFrame(
            {"id": [1, 2, 3], "value": [10.5, 20.3, 30.1], "category": ["A", "B", "A"]}
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmp_file:
            test_df.to_csv(tmp_file.name, index=False)

            # Mock the DataAgentCore and its methods
            with patch("data_agent.core.DataAgentCore") as mock_core_class:
                mock_core = Mock()
                mock_core.load_local_dataset = AsyncMock(return_value={"shape": (3, 3)})
                mock_core.get_dataset_info = Mock(return_value={
                    "shape": (3, 3), 
                    "columns": ["id", "value", "category"],
                    "memory_usage_mb": 0.1,
                    "completeness_score": 100.0
                })
                mock_core.process_query = AsyncMock(return_value={
                    "query": "describe the data",
                    "response": "This dataset contains 3 rows and 3 columns with categories A and B.",
                    "analysis_method": "descriptive_statistics",
                    "confidence": 0.95
                })
                mock_core_class.return_value = mock_core

                with patch("data_agent.llm.clients.test_llm_connectivity") as mock_connectivity:
                    mock_connectivity.return_value = {
                        "anthropic": {"available": True, "error": None}
                    }

                    # Test the function
                    await run_data_agent(
                        data_path=tmp_file.name,
                        download_url=None,
                        provider="auto",
                        model=None,
                        cache_enabled=True,
                        sample_size=None,
                        batch_mode=True,
                        single_query="describe the data",
                        output_format="text",
                        verbose=False,
                    )

                    # Verify core was initialized and dataset was loaded
                    mock_core_class.assert_called_once()
                    mock_core.load_local_dataset.assert_called_once()

            # Cleanup
            Path(tmp_file.name).unlink()

    @pytest.mark.asyncio
    async def test_run_data_agent_download_mode(self):
        """Test running data agent with dataset download."""
        with patch("data_agent.core.DataAgentCore") as mock_core_class:
            mock_core = Mock()
            mock_core.download_and_load_dataset = AsyncMock(
                return_value={"shape": (100, 5)}
            )
            mock_core.get_dataset_info = Mock(return_value={
                "shape": (100, 5), 
                "columns": ["col1", "col2", "col3", "col4", "col5"],
                "memory_usage_mb": 1.2,
                "completeness_score": 95.0
            })
            mock_core.process_query = AsyncMock(return_value={
                "query": "show statistics",
                "response": "Statistics for 100 rows and 5 columns.",
                "analysis_method": "statistical_summary",
                "confidence": 0.85
            })
            mock_core_class.return_value = mock_core

            with patch("data_agent.llm.clients.test_llm_connectivity") as mock_connectivity:
                mock_connectivity.return_value = {
                    "openai": {"available": True, "error": None}
                }

                await run_data_agent(
                    data_path=None,
                    download_url="https://example.com/data.csv",
                    provider="openai",
                    model="gpt-4",
                    cache_enabled=True,
                    sample_size=50,
                    batch_mode=True,
                    single_query="show statistics",
                    output_format="json",
                    verbose=True,
                )

                # Verify download was attempted
                mock_core.download_and_load_dataset.assert_called_once_with(
                    "https://example.com/data.csv", 50
                )

    @pytest.mark.asyncio
    async def test_no_llm_providers_available(self):
        """Test handling when no LLM providers are available."""
        with patch("data_agent.llm.clients.test_llm_connectivity") as mock_connectivity:
            mock_connectivity.return_value = {
                "openai": {"available": False, "error": "No API key"},
                "anthropic": {"available": False, "error": "No API key"},
            }

            with patch("click.echo") as mock_echo:
                await run_data_agent(
                    data_path=None,
                    download_url=None,
                    provider="auto",
                    model=None,
                    cache_enabled=True,
                    sample_size=None,
                    batch_mode=False,
                    single_query=None,
                    output_format="text",
                    verbose=False,
                )

                # Should print error message about no providers
                error_calls = [
                    call
                    for call in mock_echo.call_args_list
                    if "No LLM providers available" in str(call)
                ]
                assert len(error_calls) > 0


class TestCLIInteractiveMode:
    """Test CLI interactive mode functionality."""

    @pytest.mark.asyncio
    async def test_interactive_query_processing(self):
        """Test interactive query processing."""
        # Mock the core components
        with patch("data_agent.core.DataAgentCore") as mock_core_class:
            mock_core = Mock()
            mock_core.download_and_load_dataset = AsyncMock(
                return_value={
                    "shape": (100, 5),
                    "columns": ["id", "value", "category", "date", "score"],
                    "completeness_score": 95.0,
                }
            )
            mock_core.get_dataset_info = Mock(return_value={
                "shape": (100, 5),
                "columns": ["id", "value", "category", "date", "score"],
                "completeness_score": 95.0,
                "memory_usage_mb": 2.1
            })
            mock_core.process_query = AsyncMock(
                return_value={
                    "query": "test query",
                    "response": "This is a test response",
                    "analysis_method": "describe_dataset",
                    "confidence": 0.9,
                }
            )
            mock_core_class.return_value = mock_core

            # Mock connectivity
            with patch("data_agent.llm.clients.test_llm_connectivity") as mock_connectivity:
                mock_connectivity.return_value = {
                    "anthropic": {"available": True, "error": None}
                }

                # Mock input to simulate user interaction
                with patch("click.prompt", side_effect=["test query", "exit"]):
                    with patch("click.echo") as mock_echo:
                        await run_data_agent(
                            data_path=None,
                            download_url=None,
                            provider="auto",
                            model=None,
                            cache_enabled=True,
                            sample_size=None,
                            batch_mode=False,  # Interactive mode
                            single_query=None,
                            output_format="text",
                            verbose=False,
                        )

                        # Should have processed the query
                        mock_core.process_query.assert_called_with("test query")

                        # Should have printed the response
                        response_calls = [
                            call
                            for call in mock_echo.call_args_list
                            if "This is a test response" in str(call)
                        ]
                        assert len(response_calls) > 0


class TestCLIOutputFormatting:
    """Test different output formatting options."""

    def create_sample_response(self):
        """Create a sample response for testing."""
        return {
            "query": "describe the dataset",
            "response": "The dataset contains 100 rows and 5 columns.",
            "analysis_method": "describe_dataset",
            "confidence": 0.9,
            "evidence": {
                "key_findings": ["100 rows", "5 columns"],
                "supporting_statistics": ["Mean: 50.0", "Std: 15.0"],
            },
            "methodology": {
                "approach": "Descriptive statistics",
                "parameters": {"method": "pandas.describe()"},
            },
        }

    def test_format_response_text(self):
        """Test text response formatting."""
        from data_agent.cli import format_response

        response = self.create_sample_response()
        formatted = format_response(response, "text")

        assert "Analysis Result:" in formatted
        assert "The dataset contains 100 rows and 5 columns." in formatted
        # Text format should be simple, detailed info is in "detailed" format
        assert "Supporting Evidence:" not in formatted
        assert "Methodology:" not in formatted

    def test_format_response_json(self):
        """Test JSON response formatting."""
        from data_agent.cli import format_response

        response = self.create_sample_response()
        formatted = format_response(response, "json")

        # Should be valid JSON
        parsed = json.loads(formatted)
        assert parsed["query"] == "describe the dataset"
        assert parsed["analysis_method"] == "describe_dataset"
        assert parsed["confidence"] == 0.9

    def test_format_response_detailed(self):
        """Test detailed response formatting."""
        from data_agent.cli import format_response

        response = self.create_sample_response()
        formatted = format_response(response, "detailed")

        # Should include all sections
        assert "Analysis Result:" in formatted
        assert "Supporting Evidence:" in formatted
        assert "Methodology:" in formatted
        assert "Important Notes:" in formatted
        assert "Confidence: HIGH" in formatted

    def test_format_error_response(self):
        """Test error response formatting."""
        from data_agent.cli import format_response

        error_response = {
            "query": "invalid query",
            "error": "Analysis failed",
            "response": "Sorry, the analysis could not be completed.",
        }

        formatted = format_response(error_response, "text")
        assert "Error:" in formatted
        assert "Analysis failed" in formatted


class TestCLIUtilities:
    """Test CLI utility functions."""

    def test_confidence_indicator(self):
        """Test confidence level indicator."""
        from data_agent.cli import get_confidence_indicator

        assert get_confidence_indicator(0.9) == "HIGH"
        assert get_confidence_indicator(0.7) == "MEDIUM"
        assert get_confidence_indicator(0.4) == "LOW"
        assert get_confidence_indicator(1.0) == "HIGH"
        assert get_confidence_indicator(0.0) == "LOW"

    def test_dataset_info_display(self):
        """Test dataset information display formatting."""
        from data_agent.cli import display_dataset_info

        dataset_info = {
            "shape": (100, 5),
            "columns": ["id", "value", "category", "date", "score"],
            "completeness_score": 95.5,
            "memory_usage_mb": 2.3,
        }

        with patch("click.echo") as mock_echo:
            display_dataset_info(dataset_info, verbose=True)

            # Check that key information was displayed
            calls = [str(call) for call in mock_echo.call_args_list]
            dataset_info_found = any("Dataset Information:" in call for call in calls)
            shape_found = any("100 rows x 5 columns" in call for call in calls)
            completeness_found = any("95.5%" in call for call in calls)

            assert dataset_info_found
            assert shape_found
            assert completeness_found

    def test_show_example_queries(self):
        """Test example queries display."""
        from data_agent.cli import show_example_queries

        with patch("click.echo") as mock_echo:
            show_example_queries()

            # Check that examples were shown
            calls = [str(call) for call in mock_echo.call_args_list]
            examples_found = any("Example queries" in call for call in calls)
            correlation_found = any("correlation" in call.lower() for call in calls)
            outliers_found = any("outlier" in call.lower() for call in calls)

            assert examples_found
            assert correlation_found
            assert outliers_found


class TestCLIErrorHandling:
    """Test CLI error handling."""

    def test_keyboard_interrupt_handling(self):
        """Test graceful handling of keyboard interrupt."""
        runner = CliRunner()

        with patch("data_agent.cli.run_data_agent", side_effect=KeyboardInterrupt()):
            result = runner.invoke(main, [])

            assert result.exit_code == 0
            assert "Thanks for using Data Agent!" in result.output

    def test_general_exception_handling(self):
        """Test general exception handling."""
        runner = CliRunner()

        with patch(
            "data_agent.cli.run_data_agent", side_effect=Exception("Test error")
        ):
            result = runner.invoke(main, [])

            assert result.exit_code == 1
            assert "Error: Test error" in result.output

    def test_debug_mode_exception_handling(self):
        """Test exception handling in debug mode."""
        runner = CliRunner()

        with patch(
            "data_agent.cli.run_data_agent", side_effect=Exception("Test error")
        ):
            result = runner.invoke(main, ["--debug"])

            # In debug mode, should show full traceback
            assert result.exit_code == 1
            assert "Error: Test error" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
