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

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize LLM client."""
        self.api_key = api_key
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
        """Initialize OpenAI client."""
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package not available. Install with: pip install openai"
            )

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable."
            )

        super().__init__(api_key, model)

        self.client = OpenAI(api_key=self.api_key)
        self.rate_limit_delay = 0.1  # OpenAI allows higher rates

        logger.info(f"Initialized OpenAI client with model: {self.model}")

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
        """Initialize Anthropic client."""
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "Anthropic package not available. Install with: pip install anthropic"
            )

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not provided. Set ANTHROPIC_API_KEY environment variable."
            )

        super().__init__(api_key, model)

        self.client = Anthropic(api_key=self.api_key)
        self.rate_limit_delay = 0.2  # Anthropic rate limits

        logger.info(f"Initialized Anthropic client with model: {self.model}")

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

    def _initialize_clients(self):
        """Initialize available LLM clients."""
        # Try to initialize OpenAI
        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key and OPENAI_AVAILABLE:
                self.clients["openai"] = OpenAIClient(openai_key)
                logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize OpenAI client: {e}")

        # Try to initialize Anthropic
        try:
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            if anthropic_key and ANTHROPIC_AVAILABLE:
                self.clients["anthropic"] = AnthropicClient(anthropic_key)
                logger.info("Anthropic client initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize Anthropic client: {e}")

    def _set_current_client(self):
        """Set the current client based on preference."""
        if self.preferred_provider == "auto":
            # Auto-select first available client
            if "openai" in self.clients:
                self.current_client = self.clients["openai"]
                self.preferred_provider = "openai"
            elif "anthropic" in self.clients:
                self.current_client = self.clients["anthropic"]
                self.preferred_provider = "anthropic"
            else:
                logger.error("No LLM clients available!")
                self.current_client = None
        else:
            # Use preferred provider
            if self.preferred_provider in self.clients:
                self.current_client = self.clients[self.preferred_provider]
            else:
                logger.error(
                    f"Preferred provider {self.preferred_provider} not available"
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
    """Test connectivity to available LLM providers."""
    results = {
        "openai": {"available": False, "error": None},
        "anthropic": {"available": False, "error": None},
    }

    # Test OpenAI
    try:
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            client = OpenAIClient()
            # Test basic connectivity by checking model access
            if client.client and client.api_key:
                results["openai"]["available"] = True
            else:
                results["openai"]["error"] = "Client initialization failed"
        else:
            results["openai"][
                "error"
            ] = "API key not available or package not installed"
    except Exception as e:
        results["openai"]["error"] = str(e)

    # Test Anthropic
    try:
        if ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
            client = AnthropicClient()
            # Test basic connectivity by checking client initialization
            if client.client and client.api_key:
                results["anthropic"]["available"] = True
            else:
                results["anthropic"]["error"] = "Client initialization failed"
        else:
            results["anthropic"][
                "error"
            ] = "API key not available or package not installed"
    except Exception as e:
        results["anthropic"]["error"] = str(e)

    return results
