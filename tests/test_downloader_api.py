"""
Tests for the Google Drive API-based downloader implementation.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from data_agent.data.downloader import DatasetDownloader


class TestDatasetDownloader:
    """Test the new Google Drive API-based downloader."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.downloader = DatasetDownloader(data_dir=self.temp_dir)

    def test_extract_google_drive_id(self):
        """Test Google Drive ID extraction from various URL formats."""
        test_cases = [
            (
                "https://drive.google.com/file/d/1Gtb6XcXZRzI4fy8VASUtXaZXF530JO4C/view?usp=sharing",
                "1Gtb6XcXZRzI4fy8VASUtXaZXF530JO4C"
            ),
            (
                "https://drive.google.com/open?id=1Gtb6XcXZRzI4fy8VASUtXaZXF530JO4C",
                "1Gtb6XcXZRzI4fy8VASUtXaZXF530JO4C"
            ),
            (
                "https://drive.google.com/uc?id=1Gtb6XcXZRzI4fy8VASUtXaZXF530JO4C&export=download",
                "1Gtb6XcXZRzI4fy8VASUtXaZXF530JO4C"
            )
        ]
        
        for url, expected_id in test_cases:
            assert self.downloader.extract_google_drive_id(url) == expected_id

    def test_extract_google_drive_id_invalid(self):
        """Test that invalid URLs raise appropriate errors."""
        with pytest.raises(ValueError, match="Could not extract Google Drive ID"):
            self.downloader.extract_google_drive_id("https://example.com/invalid")

    @patch('data_agent.data.downloader.GOOGLE_API_AVAILABLE', False)
    @patch('requests.get')
    def test_fallback_download_without_api(self, mock_get):
        """Test fallback download when Google API is not available."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/octet-stream", "content-length": "1000"}
        mock_response.iter_content.return_value = [b"test data" * 100]
        mock_get.return_value = mock_response
        
        # Test URL
        test_url = "https://drive.google.com/file/d/1Gtb6XcXZRzI4fy8VASUtXaZXF530JO4C/view"
        
        result = self.downloader.download_from_google_drive(test_url, "test.parquet")
        
        assert result.exists()
        assert result.name == "test.parquet"
        mock_get.assert_called_once()

    @patch('data_agent.data.downloader.GOOGLE_API_AVAILABLE', False)
    @patch('requests.get')
    def test_fallback_download_html_response(self, mock_get):
        """Test handling of HTML responses in fallback mode."""
        # Mock HTML response (Google's error page)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_get.return_value = mock_response
        
        test_url = "https://drive.google.com/file/d/1Gtb6XcXZRzI4fy8VASUtXaZXF530JO4C/view"
        
        with pytest.raises(ValueError, match="Received HTML response"):
            self.downloader.download_from_google_drive(test_url, "test.parquet")

    @patch('data_agent.data.downloader.GOOGLE_API_AVAILABLE', True)
    def test_google_api_preference(self):
        """Test that Google API is tried first when available."""
        def mock_api_download(file_id, file_path):
            # Simulate successful download by creating the file
            file_path.write_bytes(b"test data")
            return True
            
        with patch.object(self.downloader, '_download_with_google_api', side_effect=mock_api_download) as mock_api:
            test_url = "https://drive.google.com/file/d/1Gtb6XcXZRzI4fy8VASUtXaZXF530JO4C/view"
            result = self.downloader.download_from_google_drive(test_url, "test.parquet")
            
            mock_api.assert_called_once()
            assert result.exists()

    def test_existing_file_skipped(self):
        """Test that existing files are not re-downloaded unless forced."""
        # Create existing file
        test_file = Path(self.temp_dir) / "existing.parquet"
        test_file.write_bytes(b"existing data")
        
        test_url = "https://drive.google.com/file/d/1Gtb6XcXZRzI4fy8VASUtXaZXF530JO4C/view"
        
        with patch.object(self.downloader, '_download_with_google_api') as mock_api:
            result = self.downloader.download_from_google_drive(test_url, "existing.parquet")
            
            # Should not call download methods
            mock_api.assert_not_called()
            assert result == test_file

    @patch('data_agent.data.downloader.GOOGLE_API_AVAILABLE', True)
    def test_force_download_overwrites(self):
        """Test that force_download=True overwrites existing files."""
        # Create existing file
        test_file = Path(self.temp_dir) / "existing.parquet"
        test_file.write_bytes(b"old data")
        
        test_url = "https://drive.google.com/file/d/1Gtb6XcXZRzI4fy8VASUtXaZXF530JO4C/view"
        
        def mock_api_download(file_id, file_path):
            # Simulate successful download by creating the file with new data
            file_path.write_bytes(b"new data")
            return True
            
        with patch.object(self.downloader, '_download_with_google_api', side_effect=mock_api_download):
            result = self.downloader.download_from_google_drive(
                test_url, "existing.parquet", force_download=True
            )
            assert result == test_file
            assert result.read_bytes() == b"new data"

    def test_get_dataset_info(self):
        """Test dataset info functionality."""
        # Test non-existent file
        info = self.downloader.get_dataset_info("nonexistent.parquet")
        assert info["exists"] is False
        
        # Test existing file
        test_file = Path(self.temp_dir) / "test.parquet"
        test_data = b"test data for info"
        test_file.write_bytes(test_data)
        
        info = self.downloader.get_dataset_info("test.parquet")
        assert info["exists"] is True
        assert info["size_bytes"] == len(test_data)
        assert info["size_mb"] == round(len(test_data) / (1024 * 1024), 2)
        assert "path" in info
        assert "modified_time" in info

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)