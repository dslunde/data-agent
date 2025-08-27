#!/usr/bin/env python3
"""
Run core functionality tests for the data agent.
This script runs essential tests to validate that all components work correctly.
"""

import pytest
import sys
from pathlib import Path

def main():
    """Run core tests and report results."""
    test_files = [
        "tests/test_data_loader.py::TestSchemaInfo",
        "tests/test_data_loader.py::TestDataLoader::test_data_loader_initialization",
        "tests/test_data_loader.py::TestDataLoader::test_load_parquet_file",
        "tests/test_data_loader.py::TestDataLoader::test_load_csv_file",
        "tests/test_analysis_engine.py::TestStatisticalAnalyzer::test_describe_dataset",
        "tests/test_analysis_engine.py::TestStatisticalAnalyzer::test_count_analysis",
        "tests/test_analysis_engine.py::TestPatternAnalyzer::test_correlation_analysis",
        "tests/test_analysis_engine.py::TestAnomalyDetector::test_detect_outliers_iqr",
        "tests/test_llm_integration.py::TestLLMManager::test_llm_manager_no_keys",
        "tests/test_llm_integration.py::TestQueryProcessor::test_pattern_classification_descriptive",
        "tests/test_integration.py::TestDatasetIntegration::test_dataset_loading_integration"
    ]
    
    print("Running core functionality tests...")
    print("=" * 60)
    
    # Run tests
    args = ["-v", "--tb=short"] + test_files
    
    try:
        result = pytest.main(args)
        
        if result == 0:
            print("\n" + "=" * 60)
            print("✓ All core tests PASSED!")
            print("The data agent components are working correctly.")
            return True
        else:
            print("\n" + "=" * 60)
            print("✗ Some tests FAILED!")
            print("Please check the test output above for details.")
            return False
            
    except Exception as e:
        print(f"\n✗ Error running tests: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)