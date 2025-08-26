"""
Dataset download functionality for Google Drive and other sources.
"""

import os
import re
import requests
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class DatasetDownloader:
    """Downloads and manages datasets from various sources."""
    
    def __init__(self, data_dir: str = "./data"):
        """Initialize downloader with data directory."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def extract_google_drive_id(self, url: str) -> str:
        """Extract file ID from Google Drive URL."""
        patterns = [
            r'/file/d/([a-zA-Z0-9_-]+)',
            r'id=([a-zA-Z0-9_-]+)',
            r'/d/([a-zA-Z0-9_-]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        raise ValueError(f"Could not extract Google Drive ID from URL: {url}")
    
    def download_from_google_drive(
        self, 
        url: str, 
        filename: Optional[str] = None,
        force_download: bool = False
    ) -> Path:
        """
        Download file from Google Drive.
        
        Args:
            url: Google Drive URL
            filename: Local filename (defaults to dataset.parquet)
            force_download: Force re-download even if file exists
            
        Returns:
            Path to downloaded file
        """
        if filename is None:
            filename = "dataset.parquet"
            
        file_path = self.data_dir / filename
        
        # Check if file already exists
        if file_path.exists() and not force_download:
            logger.info(f"Dataset already exists at {file_path}")
            return file_path
        
        # Extract file ID from URL
        file_id = self.extract_google_drive_id(url)
        
        # Google Drive direct download URL
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        logger.info(f"Downloading dataset from Google Drive to {file_path}")
        
        try:
            # Start download session
            session = requests.Session()
            response = session.get(download_url, stream=True)
            
            # Handle large file download confirmation
            if "download_warning" in response.cookies:
                params = {'id': file_id, 'confirm': response.cookies["download_warning"]}
                response = session.get(download_url, params=params, stream=True)
            
            response.raise_for_status()
            
            # Get file size for progress bar
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress bar
            with open(file_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            logger.info(f"Successfully downloaded dataset to {file_path}")
            return file_path
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading dataset: {e}")
            # Clean up partial download
            if file_path.exists():
                file_path.unlink()
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error during download: {e}")
            # Clean up partial download
            if file_path.exists():
                file_path.unlink()
            raise
    
    def download_dataset(
        self,
        source: Optional[str] = None,
        filename: Optional[str] = None,
        force_download: bool = False
    ) -> Path:
        """
        Download dataset from configured source or provided URL.
        
        Args:
            source: URL to download from (uses default if None)
            filename: Local filename
            force_download: Force re-download
            
        Returns:
            Path to downloaded file
        """
        # Default Google Drive URL from PRD
        default_url = "https://drive.google.com/file/d/1Gtb6XcXZRzI4fy8VASUtXaZXF530JO4C/view?usp=sharing"
        
        url = source or default_url
        
        if "drive.google.com" in url:
            return self.download_from_google_drive(url, filename, force_download)
        else:
            raise NotImplementedError("Only Google Drive URLs are currently supported")
    
    def get_dataset_path(self, filename: str = "dataset.parquet") -> Path:
        """Get path to local dataset file."""
        return self.data_dir / filename
    
    def dataset_exists(self, filename: str = "dataset.parquet") -> bool:
        """Check if dataset file exists locally."""
        return self.get_dataset_path(filename).exists()
    
    def get_dataset_info(self, filename: str = "dataset.parquet") -> dict:
        """Get information about local dataset file."""
        file_path = self.get_dataset_path(filename)
        
        if not file_path.exists():
            return {"exists": False}
        
        stat = file_path.stat()
        
        return {
            "exists": True,
            "path": str(file_path),
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified_time": stat.st_mtime
        }


def get_default_downloader() -> DatasetDownloader:
    """Get default dataset downloader instance."""
    return DatasetDownloader()
