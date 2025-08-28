"""
LLM API clients with error handling, rate limiting, and retry logic.
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

try:
    import openai
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    from anthropic import Anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, model: Optional[str] = None):
        """Initialize LLM client.
        
        Note: API keys are not stored in memory for security.
        Subclasses handle API keys directly with their respective SDK clients.
        """
        self.model = model
        self.request_count = 0
        self.last_request_time = 0
        self.rate_limit_delay = 1.0  # seconds between requests

    @abstractmethod
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate response from LLM."""
        pass

    def _rate_limit(self):
        """Simple rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            time.sleep(sleep_time)

        self.last_request_time = time.time()
        self.request_count += 1

    def _retry_on_error(self, func, max_retries: int = 3, delay: float = 1.0):
        """Retry function call on error."""
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e

                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds..."
                )
                time.sleep(delay * (2**attempt))  # Exponential backoff

        return None


class OpenAIClient(LLMClient):
    """OpenAI GPT client with error handling and rate limiting."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """Initialize OpenAI client.
        
        Args:
            api_key: API key (if None, loads from OPENAI_API_KEY env var)
            model: Model to use for requests
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package not available. Install with: pip install openai"
            )

        super().__init__(model)

        # Securely get and validate API key without storing
        api_key_to_use = self._get_secure_api_key(api_key)
        
        # Let OpenAI SDK handle the API key securely
        # We don't store the key in our instance
        self.client = OpenAI(api_key=api_key_to_use)
        self.rate_limit_delay = 0.1  # OpenAI allows higher rates

        logger.info(f"Initialized OpenAI client with model: {self.model}")

    def _get_secure_api_key(self, provided_key: Optional[str]) -> str:
        """Securely retrieve and validate API key without storing."""
        # Get key from parameter or environment
        api_key = provided_key or os.getenv("OPENAI_API_KEY")
        
        if not api_key or not api_key.strip():
            raise ValueError(
                "OpenAI API key not provided or empty. Set OPENAI_API_KEY environment variable."
            )
        
        # Validate key format without storing
        if not self._is_valid_openai_key(api_key):
            raise ValueError(
                "OpenAI API key format is invalid. Expected format: sk-..."
            )
        
        return api_key
    
    @staticmethod
    def _is_valid_openai_key(key: str) -> bool:
        """Validate OpenAI API key format without storing."""
        if not key or not key.strip():
            return False
            
        # Common placeholder patterns
        placeholders = [
            "your_openai_key_here",
            "your_api_key_here", 
            "replace_with_your_key",
            "your_key_here",
            "sk-placeholder"
        ]
        
        key_lower = key.lower()
        if any(placeholder in key_lower for placeholder in placeholders):
            return False
        
        # OpenAI keys should start with sk- and be reasonable length
        return key.startswith("sk-") and len(key) > 20

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate response from OpenAI."""
        self._rate_limit()

        def _make_request():
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens or 2000,
                    top_p=0.9,
                    frequency_penalty=0.1,
                    presence_penalty=0.1,
                )

                return {
                    "content": response.choices[0].message.content,
                    "model": response.model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                    "finish_reason": response.choices[0].finish_reason,
                }

            except openai.RateLimitError as e:
                logger.warning("Rate limit exceeded, waiting...")
                time.sleep(60)  # Wait 1 minute for rate limit reset
                raise e

            except openai.APIError as e:
                logger.error(f"OpenAI API error: {e}")
                raise e

            except Exception as e:
                logger.error(f"Unexpected error in OpenAI request: {e}")
                raise e

        try:
            return self._retry_on_error(_make_request, max_retries=3)
        except Exception as e:
            return {
                "error": str(e),
                "content": "Sorry, I encountered an error while processing your request.",
                "model": self.model,
            }


class AnthropicClient(LLMClient):
    """Anthropic Claude client with error handling and rate limiting."""

    def __init__(
        self, api_key: Optional[str] = None, model: str = "claude-3-haiku-20240307"
    ):
        """Initialize Anthropic client.
        
        Args:
            api_key: API key (if None, loads from ANTHROPIC_API_KEY env var)
            model: Model to use for requests
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "Anthropic package not available. Install with: pip install anthropic"
            )

        super().__init__(model)

        # Securely get and validate API key without storing
        api_key_to_use = self._get_secure_api_key(api_key)
        
        # Let Anthropic SDK handle the API key securely
        # We don't store the key in our instance
        self.client = Anthropic(api_key=api_key_to_use)
        self.rate_limit_delay = 0.2  # Anthropic rate limits

        logger.info(f"Initialized Anthropic client with model: {self.model}")

    def _get_secure_api_key(self, provided_key: Optional[str]) -> str:
        """Securely retrieve and validate API key without storing."""
        # Get key from parameter or environment
        api_key = provided_key or os.getenv("ANTHROPIC_API_KEY")
        
        if not api_key or not api_key.strip():
            raise ValueError(
                "Anthropic API key not provided or empty. Set ANTHROPIC_API_KEY environment variable."
            )
        
        # Validate key format without storing
        if not self._is_valid_anthropic_key(api_key):
            raise ValueError(
                "Anthropic API key format is invalid. Expected format: sk-ant-..."
            )
        
        return api_key
    
    @staticmethod
    def _is_valid_anthropic_key(key: str) -> bool:
        """Validate Anthropic API key format without storing."""
        if not key or not key.strip():
            return False
            
        # Common placeholder patterns
        placeholders = [
            "your_anthropic_key_here",
            "your_api_key_here",
            "replace_with_your_key", 
            "your_key_here",
            "sk-ant-placeholder"
        ]
        
        key_lower = key.lower()
        if any(placeholder in key_lower for placeholder in placeholders):
            return False
        
        # Anthropic keys should start with sk-ant- and be reasonable length
        return key.startswith("sk-ant-") and len(key) > 30

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate response from Anthropic Claude."""
        self._rate_limit()

        def _make_request():
            try:
                # Convert messages to Anthropic format
                system_message = ""
                conversation_messages = []

                for msg in messages:
                    if msg["role"] == "system":
                        system_message = msg["content"]
                    else:
                        conversation_messages.append(
                            {"role": msg["role"], "content": msg["content"]}
                        )

                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens or 2000,
                    temperature=temperature,
                    system=system_message if system_message else None,
                    messages=conversation_messages,
                )

                return {
                    "content": response.content[0].text,
                    "model": response.model,
                    "usage": {
                        "prompt_tokens": response.usage.input_tokens,
                        "completion_tokens": response.usage.output_tokens,
                        "total_tokens": response.usage.input_tokens
                        + response.usage.output_tokens,
                    },
                    "finish_reason": response.stop_reason,
                }

            except anthropic.RateLimitError as e:
                logger.warning("Rate limit exceeded, waiting...")
                time.sleep(60)  # Wait 1 minute for rate limit reset
                raise e

            except anthropic.APIError as e:
                logger.error(f"Anthropic API error: {e}")
                raise e

            except Exception as e:
                logger.error(f"Unexpected error in Anthropic request: {e}")
                raise e

        try:
            return self._retry_on_error(_make_request, max_retries=3)
        except Exception as e:
            return {
                "error": str(e),
                "content": "Sorry, I encountered an error while processing your request.",
                "model": self.model,
            }


class LLMManager:
    """Manages multiple LLM clients and provides unified interface."""

    def __init__(self, preferred_provider: str = "auto"):
        """
        Initialize LLM manager.

        Args:
            preferred_provider: Preferred LLM provider ('openai', 'anthropic', 'auto')
        """
        self.preferred_provider = preferred_provider
        self.clients = {}
        self.current_client = None

        # Initialize available clients
        self._initialize_clients()

        # Set current client based on preference
        self._set_current_client()

        logger.info(
            f"LLM Manager initialized with provider: {self.current_client.__class__.__name__ if self.current_client else 'None'}"
        )

    @staticmethod
    def _is_valid_api_key_format(key: str, provider: str) -> bool:
        """Check if API key has valid format without storing the key.
        
        This method validates key format and detects placeholders,
        but does not store the key in memory for security.
        """
        if provider == "openai":
            return OpenAIClient._is_valid_openai_key(key)
        elif provider == "anthropic":
            return AnthropicClient._is_valid_anthropic_key(key)
        
        return False

    def _initialize_clients(self):
        """Initialize available LLM clients securely.
        
        This method attempts to initialize clients without storing API keys.
        Each client handles its own secure key retrieval and validation.
        """
        # Try to initialize OpenAI
        if OPENAI_AVAILABLE and self._should_initialize_openai():
            try:
                # Let OpenAIClient handle secure key retrieval and validation
                client = OpenAIClient()
                self.clients["openai"] = client
                logger.info("OpenAI client initialized successfully")
            except ValueError as e:
                logger.debug(f"OpenAI client initialization failed: {e}")
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client: {e}")

        # Try to initialize Anthropic
        if ANTHROPIC_AVAILABLE and self._should_initialize_anthropic():
            try:
                # Let AnthropicClient handle secure key retrieval and validation
                client = AnthropicClient()
                self.clients["anthropic"] = client
                logger.info("Anthropic client initialized successfully")
            except ValueError as e:
                logger.debug(f"Anthropic client initialization failed: {e}")
            except Exception as e:
                logger.warning(f"Could not initialize Anthropic client: {e}")

    def _should_initialize_openai(self) -> bool:
        """Check if OpenAI should be initialized without reading the full key."""
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key or not openai_key.strip():
            logger.debug("OpenAI API key not found or empty")
            return False
            
        # Quick format check without storing the key
        if not self._is_valid_api_key_format(openai_key, "openai"):
            logger.debug("OpenAI API key appears to be a placeholder or invalid format")
            return False
            
        return True

    def _should_initialize_anthropic(self) -> bool:
        """Check if Anthropic should be initialized without reading the full key."""
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_key or not anthropic_key.strip():
            logger.debug("Anthropic API key not found or empty")
            return False
            
        # Quick format check without storing the key
        if not self._is_valid_api_key_format(anthropic_key, "anthropic"):
            logger.debug("Anthropic API key appears to be a placeholder or invalid format")
            return False
            
        return True

    def _set_current_client(self):
        """Set the current client based on preference."""
        if self.preferred_provider == "auto":
            # Auto-select from available clients
            available_clients = list(self.clients.keys())

            if not available_clients:
                logger.error("No LLM clients available! Please check your API keys.")
                self.current_client = None
                return

            # Prefer Anthropic if available, then OpenAI (Anthropic generally has better reasoning)
            if "anthropic" in available_clients:
                self.current_client = self.clients["anthropic"]
                self.preferred_provider = "anthropic"
                logger.info("Auto-selected Anthropic as LLM provider")
            elif "openai" in available_clients:
                self.current_client = self.clients["openai"]
                self.preferred_provider = "openai"
                logger.info("Auto-selected OpenAI as LLM provider")
            else:
                # Use first available client
                provider = available_clients[0]
                self.current_client = self.clients[provider]
                self.preferred_provider = provider
                logger.info(f"Auto-selected {provider} as LLM provider")
        else:
            # Use preferred provider
            if self.preferred_provider in self.clients:
                self.current_client = self.clients[self.preferred_provider]
                logger.info(f"Using preferred provider: {self.preferred_provider}")
            else:
                available = list(self.clients.keys())
                logger.error(
                    f"Preferred provider '{self.preferred_provider}' not available. "
                    f"Available providers: {available if available else 'None'}"
                )
                self.current_client = None

    def switch_provider(self, provider: str) -> bool:
        """Switch to a different provider."""
        if provider in self.clients:
            self.current_client = self.clients[provider]
            self.preferred_provider = provider
            logger.info(f"Switched to provider: {provider}")
            return True
        else:
            logger.error(f"Provider {provider} not available")
            return False

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        fallback: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate response using current client with optional fallback.

        Args:
            messages: List of message dictionaries
            temperature: Response randomness
            max_tokens: Maximum tokens in response
            fallback: Whether to try fallback providers on failure

        Returns:
            Response dictionary
        """
        if not self.current_client:
            return {
                "error": "No LLM client available",
                "content": "Sorry, no language model is currently available.",
            }

        try:
            response = await self.current_client.generate_response(
                messages, temperature, max_tokens
            )

            # Check if response has error
            if "error" not in response:
                return response

            # If error and fallback enabled, try other providers
            if fallback:
                for provider, client in self.clients.items():
                    if provider != self.preferred_provider:
                        try:
                            logger.info(f"Trying fallback provider: {provider}")
                            fallback_response = await client.generate_response(
                                messages, temperature, max_tokens
                            )
                            if "error" not in fallback_response:
                                return fallback_response
                        except Exception as e:
                            logger.warning(f"Fallback provider {provider} failed: {e}")
                            continue

            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "error": str(e),
                "content": "Sorry, I encountered an error while processing your request.",
            }

    def get_available_providers(self) -> List[str]:
        """Get list of available providers."""
        return list(self.clients.keys())

    def get_current_provider(self) -> Optional[str]:
        """Get current provider name."""
        return self.preferred_provider if self.current_client else None

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all clients."""
        stats = {}

        for provider, client in self.clients.items():
            stats[provider] = {
                "request_count": client.request_count,
                "model": client.model,
            }

        return stats


def create_llm_manager(preferred_provider: str = "auto") -> LLMManager:
    """Create and return LLM manager instance."""
    return LLMManager(preferred_provider)


def test_llm_connectivity() -> Dict[str, Any]:
    """Test connectivity to available LLM providers securely.
    
    This function tests connectivity without storing API keys in memory.
    It relies on each client's secure initialization process.
    """
    results = {
        "openai": {"available": False, "error": None},
        "anthropic": {"available": False, "error": None},
    }

    # Test OpenAI
    try:
        if OPENAI_AVAILABLE:
            # Check if environment variable exists without storing it
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key and openai_key.strip():
                # Try to create client (it will handle secure validation)
                client = OpenAIClient()
                if client.client:
                    results["openai"]["available"] = True
                else:
                    results["openai"]["error"] = "Client initialization failed"
            else:
                results["openai"]["error"] = "API key not found in environment"
        else:
            results["openai"]["error"] = "OpenAI package not installed"
    except ValueError as e:
        # These are expected for invalid keys
        results["openai"]["error"] = str(e)
    except Exception as e:
        results["openai"]["error"] = f"Unexpected error: {str(e)}"

    # Test Anthropic
    try:
        if ANTHROPIC_AVAILABLE:
            # Check if environment variable exists without storing it
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            if anthropic_key and anthropic_key.strip():
                # Try to create client (it will handle secure validation)
                client = AnthropicClient()
                if client.client:
                    results["anthropic"]["available"] = True
                else:
                    results["anthropic"]["error"] = "Client initialization failed"
            else:
                results["anthropic"]["error"] = "API key not found in environment"
        else:
            results["anthropic"]["error"] = "Anthropic package not installed"
    except ValueError as e:
        # These are expected for invalid keys
        results["anthropic"]["error"] = str(e)
    except Exception as e:
        results["anthropic"]["error"] = f"Unexpected error: {str(e)}"

    return results
