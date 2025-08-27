"""
Main entry point for running the data agent as a module.

Usage: python -m data_agent [options]
"""

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not available, skip

from .cli import main

if __name__ == "__main__":
    main()
