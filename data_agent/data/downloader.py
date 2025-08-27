"""
Dataset download functionality for Google Drive and other sources.
"""

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
            r"/file/d/([a-zA-Z0-9_-]+)",
            r"id=([a-zA-Z0-9_-]+)",
            r"/d/([a-zA-Z0-9_-]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        raise ValueError(f"Could not extract Google Drive ID from URL: {url}")

    def _try_download_method(
        self,
        session: requests.Session,
        method_name: str,
        url: str,
        headers: dict = None,
    ) -> requests.Response:
        """Try a specific download method."""
        logger.info(f"Trying download method: {method_name}")
        response = session.get(url, stream=True, headers=headers or {})

        # Check if we got HTML (virus scan warning or error page)
        content_type = response.headers.get("content-type", "").lower()
        if "text/html" in content_type:
            logger.debug(f"Method {method_name} returned HTML content")
            return None

        if response.status_code == 200:
            logger.info(f"Method {method_name} succeeded")
            return response

        logger.debug(f"Method {method_name} failed with status {response.status_code}")
        return None

    def download_from_google_drive(
        self, url: str, filename: Optional[str] = None, force_download: bool = False
    ) -> Path:
        """
        Download file from Google Drive using multiple fallback methods.

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

        logger.info(f"Downloading dataset from Google Drive to {file_path}")

        try:
            session = requests.Session()

            # Multiple download methods to try
            download_methods = [
                # Method 1: Direct download with confirm=t
                {
                    "name": "direct_confirm",
                    "url": f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t",
                    "headers": {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    },
                },
                # Method 2: Alternative direct download
                {
                    "name": "alt_direct",
                    "url": f"https://drive.google.com/u/0/uc?id={file_id}&export=download&confirm=t&uuid=12345",
                    "headers": {
                        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
                    },
                },
                # Method 3: Traditional method with token parsing
                {
                    "name": "token_method",
                    "url": f"https://drive.google.com/uc?export=download&id={file_id}",
                    "headers": {},
                },
                # Method 4: Mobile user agent
                {
                    "name": "mobile_agent",
                    "url": f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t",
                    "headers": {
                        "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15"
                    },
                },
            ]

            response = None

            # Try each method
            for method in download_methods:
                try:
                    test_response = self._try_download_method(
                        session, method["name"], method["url"], method["headers"]
                    )

                    if test_response:
                        response = test_response
                        break

                except Exception as e:
                    logger.debug(f"Method {method['name']} failed: {e}")
                    continue

            # If direct methods failed, try token extraction method
            if not response:
                logger.info("Direct methods failed, trying token extraction...")
                initial_response = session.get(
                    f"https://drive.google.com/uc?export=download&id={file_id}"
                )

                if (
                    initial_response.status_code == 200
                    and "text/html" in initial_response.headers.get("content-type", "")
                ):
                    html_content = initial_response.text

                    # Look for various token patterns
                    token_patterns = [
                        r'confirm=([^&"\']+)',
                        r'"confirm":"([^"]+)"',
                        r'confirm%3D([^&"\']+)',
                        r'uuid=([^&"\']+)',
                    ]

                    for pattern in token_patterns:
                        match = re.search(pattern, html_content)
                        if match:
                            token = match.group(1)
                            logger.info(
                                f"Found token with pattern {pattern}: {token[:10]}..."
                            )

                            confirmed_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={token}"
                            response = self._try_download_method(
                                session, "token_confirmed", confirmed_url
                            )
                            if response:
                                break

            if not response:
                raise ValueError(
                    "All download methods failed - unable to bypass Google Drive restrictions"
                )

            # Get file size for progress bar
            total_size = int(response.headers.get("content-length", 0))
            if total_size > 0:
                logger.info(f"Downloading {total_size:,} bytes...")
            else:
                logger.info("Downloading (size unknown)...")

            # Download with progress bar
            downloaded_size = 0
            with open(file_path, "wb") as f:
                if total_size > 0:
                    with tqdm(
                        total=total_size, unit="B", unit_scale=True, desc="Downloading"
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded_size += len(chunk)
                                pbar.update(len(chunk))
                else:
                    # No content-length header, show basic progress
                    with tqdm(unit="B", unit_scale=True, desc="Downloading") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded_size += len(chunk)
                                pbar.update(len(chunk))

            # Verify the downloaded file
            actual_size = file_path.stat().st_size
            if actual_size < 1000:
                logger.error(f"Downloaded file is too small ({actual_size} bytes)")
                # Check if it's HTML
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        first_content = f.read(500)
                        if any(
                            tag in first_content.lower()
                            for tag in ["<html", "<!doctype", "<head", "<body"]
                        ):
                            logger.error(
                                "Downloaded file appears to be HTML, not the expected file"
                            )
                            file_path.unlink()
                            raise ValueError(
                                "Downloaded HTML page instead of file - the file may be too large for direct download or requires different permissions"
                            )
                except Exception:
                    pass  # If we can't read as text, it's probably binary (good)

            logger.info(
                f"Successfully downloaded dataset to {file_path} ({actual_size:,} bytes)"
            )
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
        force_download: bool = False,
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
            try:
                return self.download_from_google_drive(url, filename, force_download)
            except Exception as e:
                logger.error(f"Automated download failed: {e}")
                self._show_manual_download_instructions(
                    url, filename or "dataset.parquet"
                )
                raise
        else:
            raise NotImplementedError("Only Google Drive URLs are currently supported")

    def _show_manual_download_instructions(self, url: str, filename: str):
        """Show manual download instructions when automated methods fail."""
        file_id = self.extract_google_drive_id(url)
        data_dir = self.data_dir.absolute()

        print(f"\n{'='*60}")
        print("MANUAL DOWNLOAD REQUIRED")
        print(f"{'='*60}")
        print("The automated download failed. Please manually download the file:")
        print("\n1. Open this URL in your browser:")
        print(f"   https://drive.google.com/file/d/{file_id}/view")
        print("\n2. Click 'Download' button")
        print(f"3. Save the file as: {filename}")
        print(f"4. Move the file to: {data_dir}/")
        print("\nAlternatively, try these direct download links:")
        print(f"   • https://drive.google.com/uc?export=download&id={file_id}")
        print(f"   • https://drive.google.com/u/0/uc?id={file_id}&export=download")
        print("\nOnce downloaded, place the file at:")
        print(f"   {data_dir}/{filename}")
        print("\nThen restart the application.")
        print(f"{'='*60}\n")

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
            "modified_time": stat.st_mtime,
        }


def get_default_downloader() -> DatasetDownloader:
    """Get default dataset downloader instance."""
    return DatasetDownloader()
