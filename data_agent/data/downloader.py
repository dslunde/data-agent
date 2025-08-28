"""
Dataset download functionality for Google Drive and other sources.
"""

import re
import requests
import os
import io
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import logging

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False

logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Downloads and manages datasets from various sources."""

    # OAuth 2.0 scopes for Google Drive
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

    def __init__(self, data_dir: str = "./data"):
        """Initialize downloader with data directory."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self._drive_service = None

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

    def _get_google_drive_service(self):
        """Get authenticated Google Drive service."""
        if not GOOGLE_API_AVAILABLE:
            raise ImportError(
                "Google API client libraries not installed. "
                "Run: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
            )

        if self._drive_service:
            return self._drive_service

        creds = None
        token_path = self.data_dir / "token.json"
        
        # Try to get credentials path from environment variable first
        credentials_path_env = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if credentials_path_env:
            credentials_path = Path(credentials_path_env)
        else:
            credentials_path = self.data_dir / "credentials.json"

        # Load existing credentials if available
        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), self.SCOPES)

        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    logger.warning(f"Token refresh failed: {e}")
                    creds = None

            if not creds:
                if not credentials_path.exists():
                    # Check if we have an API key for limited access
                    api_key = os.getenv('GOOGLE_DRIVE_API_KEY')
                    if api_key:
                        logger.info("Using Google Drive API key for limited access")
                        self._drive_service = build('drive', 'v3', developerKey=api_key)
                        return self._drive_service
                    else:
                        logger.warning(
                            f"Google API credentials not found at {credentials_path}. "
                            "Falling back to unauthenticated access."
                        )
                        # Try without authentication (public files only)
                        self._drive_service = build('drive', 'v3', developerKey=None)
                        return self._drive_service

                flow = InstalledAppFlow.from_client_secrets_file(
                    str(credentials_path), self.SCOPES
                )
                creds = flow.run_local_server(port=0)

            # Save the credentials for the next run
            with open(token_path, 'w') as token:
                token.write(creds.to_json())

        self._drive_service = build('drive', 'v3', credentials=creds)
        return self._drive_service

    def _download_with_google_api(self, file_id: str, file_path: Path) -> bool:
        """Download file using Google Drive API."""
        try:
            service = self._get_google_drive_service()
            
            # Get file metadata
            file_metadata = service.files().get(fileId=file_id).execute()
            logger.info(f"Downloading: {file_metadata.get('name', 'Unknown')}")
            
            # Request the file content
            request = service.files().get_media(fileId=file_id)
            
            # Use BytesIO to handle the download in memory first
            file_content = io.BytesIO()
            
            # Download in chunks
            from googleapiclient.http import MediaIoBaseDownload
            downloader = MediaIoBaseDownload(file_content, request)
            
            done = False
            with tqdm(desc="Downloading", unit="B", unit_scale=True) as pbar:
                while done is False:
                    status, done = downloader.next_chunk()
                    if status:
                        pbar.total = status.total_size
                        pbar.n = status.resumable_progress
                        pbar.refresh()
            
            # Write to file
            with open(file_path, 'wb') as f:
                f.write(file_content.getvalue())
                
            logger.info(f"Successfully downloaded using Google Drive API: {file_path}")
            return True
            
        except HttpError as error:
            if error.resp.status == 404:
                logger.error(f"File not found: {file_id}")
            elif error.resp.status == 403:
                logger.error(f"Permission denied for file: {file_id}")
            else:
                logger.error(f"Google API error: {error}")
            return False
        except Exception as e:
            logger.error(f"Error downloading with Google API: {e}")
            return False

    def download_from_google_drive(
        self, url: str, filename: Optional[str] = None, force_download: bool = False
    ) -> Path:
        """
        Download file from Google Drive using the official Google Drive API.

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
            # Try Google Drive API first (most reliable)
            if GOOGLE_API_AVAILABLE:
                logger.info("Attempting download with Google Drive API...")
                if self._download_with_google_api(file_id, file_path):
                    # Verify downloaded file
                    actual_size = file_path.stat().st_size
                    logger.info(
                        f"Successfully downloaded dataset to {file_path} ({actual_size:,} bytes)"
                    )
                    return file_path
                else:
                    logger.warning("Google Drive API failed, trying direct download as fallback...")
            else:
                logger.info("Google API libraries not available, using direct download...")

            # Fallback to simple direct download (only as backup)
            return self._fallback_direct_download(file_id, file_path)

        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            # Clean up partial download
            if file_path.exists():
                file_path.unlink()
            raise

    def _fallback_direct_download(self, file_id: str, file_path: Path) -> Path:
        """Simple fallback download method for public files."""
        try:
            # Try simple direct download URL (works for public files)
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
            
            logger.info("Attempting direct download...")
            response = requests.get(download_url, stream=True)
            
            # Check if we got HTML (usually means the file requires authentication)
            content_type = response.headers.get("content-type", "").lower()
            if "text/html" in content_type:
                raise ValueError(
                    "Received HTML response - file may require authentication or may not be public"
                )
            
            if response.status_code != 200:
                raise ValueError(f"Download failed with status code: {response.status_code}")
            
            # Get file size for progress bar
            total_size = int(response.headers.get("content-length", 0))
            
            # Download with progress bar
            with open(file_path, "wb") as f:
                if total_size > 0:
                    with tqdm(
                        total=total_size, unit="B", unit_scale=True, desc="Downloading"
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    # No content-length header, show basic progress
                    with tqdm(unit="B", unit_scale=True, desc="Downloading") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
            
            # Basic verification
            actual_size = file_path.stat().st_size
            if actual_size < 1000:
                # Check if it's HTML error page
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        first_content = f.read(500)
                        if any(
                            tag in first_content.lower()
                            for tag in ["<html", "<!doctype", "<head", "<body"]
                        ):
                            file_path.unlink()
                            raise ValueError(
                                "Downloaded HTML page instead of file. "
                                "The file may require authentication or manual download."
                            )
                except Exception:
                    pass  # If we can't read as text, it's probably binary
            
            logger.info(f"Direct download completed: {actual_size:,} bytes")
            return file_path
            
        except Exception as e:
            logger.error(f"Direct download failed: {e}")
            # Clean up
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
        """Show manual download and Google API setup instructions when automated methods fail."""
        file_id = self.extract_google_drive_id(url)
        data_dir = self.data_dir.absolute()

        print(f"\n{'='*60}")
        print("DOWNLOAD FAILED - SETUP REQUIRED")
        print(f"{'='*60}")
        print("The Google Drive download failed. Here are your options:")
        
        print("\n🔧 OPTION 1: Set up Google Drive API (RECOMMENDED)")
        print("This provides the most reliable downloads:")
        print("1. Go to: https://console.developers.google.com/")
        print("2. Create a new project or select existing one")
        print("3. Enable the Google Drive API")
        print("4. Create credentials (OAuth 2.0 Client ID for Desktop app)")
        print("5. Download the credentials JSON file")
        print(f"6. Save it as: {data_dir}/credentials.json")
        print("7. Run the application again - it will guide you through OAuth")
        
        print("\n📁 OPTION 2: Manual download")
        print("If you prefer to download manually:")
        print("1. Open this URL in your browser:")
        print(f"   https://drive.google.com/file/d/{file_id}/view")
        print("2. Click 'Download' button")
        print(f"3. Save the file as: {filename}")
        print(f"4. Move the file to: {data_dir}/")
        
        print("\n🔗 OPTION 3: Try direct links")
        print("These may work for public files:")
        print(f"   • https://drive.google.com/uc?export=download&id={file_id}")
        print(f"   • https://drive.google.com/u/0/uc?id={file_id}&export=download")
        
        print(f"\n📂 Target location: {data_dir}/{filename}")
        print("Once the file is in place, restart the application.")
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
