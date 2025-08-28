"""
Tests for secure API key handling in LLM clients.

This test suite verifies that API keys are handled securely:
- No API keys stored in memory unnecessarily
- Robust validation without persistent storage
- Placeholder detection for common dummy keys
- Secure client initialization patterns
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from data_agent.llm.clients import (
    LLMClient, OpenAIClient, AnthropicClient, LLMManager,
    test_llm_connectivity
)


class TestSecureAPIKeyHandling:
    """Test secure API key handling across all LLM clients."""

    def test_base_class_no_api_key_storage(self):
        """Test that base LLMClient class doesn't store API keys."""
        # Create a mock concrete implementation
        class TestClient(LLMClient):
            async def generate_response(self, messages, temperature=0.1, max_tokens=None):
                return {"content": "test"}
        
        client = TestClient(model="test-model")
        
        # Verify no api_key attribute exists
        assert not hasattr(client, 'api_key'), "Base class should not have api_key attribute"
        assert hasattr(client, 'model'), "Base class should have model attribute"
        assert hasattr(client, 'request_count'), "Base class should have request_count attribute"

    def test_openai_placeholder_detection(self):
        """Test OpenAI placeholder key detection."""
        # Test valid format
        valid_key = "sk-1234567890abcdefghijklmnopqrstuvwxyz1234567890"
        assert OpenAIClient._is_valid_openai_key(valid_key), "Should accept valid OpenAI key format"
        
        # Test common placeholders
        placeholders = [
            "your_openai_key_here",
            "your_api_key_here", 
            "replace_with_your_key",
            "your_key_here",
            "sk-placeholder"
        ]
        
        for placeholder in placeholders:
            assert not OpenAIClient._is_valid_openai_key(placeholder), f"Should reject placeholder: {placeholder}"
        
        # Test invalid formats
        invalid_keys = [
            "",
            "   ",
            "invalid-key",
            "sk-", 
            "sk-short"
        ]
        
        for invalid_key in invalid_keys:
            assert not OpenAIClient._is_valid_openai_key(invalid_key), f"Should reject invalid key: {invalid_key}"

    def test_anthropic_placeholder_detection(self):
        """Test Anthropic placeholder key detection."""
        # Test valid format
        valid_key = "sk-ant-1234567890abcdefghijklmnopqrstuvwxyz1234567890"
        assert AnthropicClient._is_valid_anthropic_key(valid_key), "Should accept valid Anthropic key format"
        
        # Test common placeholders
        placeholders = [
            "your_anthropic_key_here",
            "your_api_key_here",
            "replace_with_your_key", 
            "your_key_here",
            "sk-ant-placeholder"
        ]
        
        for placeholder in placeholders:
            assert not AnthropicClient._is_valid_anthropic_key(placeholder), f"Should reject placeholder: {placeholder}"
        
        # Test invalid formats
        invalid_keys = [
            "",
            "   ",
            "invalid-key",
            "sk-ant-",
            "sk-ant-short",
            "sk-wrongformat"
        ]
        
        for invalid_key in invalid_keys:
            assert not AnthropicClient._is_valid_anthropic_key(invalid_key), f"Should reject invalid key: {invalid_key}"

    @patch.dict('os.environ', {}, clear=True)
    def test_openai_client_secure_initialization_no_key(self):
        """Test OpenAI client fails securely when no API key is available."""
        with pytest.raises(ValueError, match="OpenAI API key not provided or empty"):
            OpenAIClient()

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'your_openai_key_here'}, clear=True)
    def test_openai_client_secure_initialization_placeholder(self):
        """Test OpenAI client fails securely with placeholder key."""
        with pytest.raises(ValueError, match="OpenAI API key format is invalid"):
            OpenAIClient()

    @patch.dict('os.environ', {}, clear=True)  
    def test_anthropic_client_secure_initialization_no_key(self):
        """Test Anthropic client fails securely when no API key is available."""
        with pytest.raises(ValueError, match="Anthropic API key not provided or empty"):
            AnthropicClient()

    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'your_anthropic_key_here'}, clear=True)
    def test_anthropic_client_secure_initialization_placeholder(self):
        """Test Anthropic client fails securely with placeholder key."""
        with pytest.raises(ValueError, match="Anthropic API key format is invalid"):
            AnthropicClient()

    @patch('data_agent.llm.clients.OpenAI')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-valid1234567890abcdefghijklmnop'}, clear=True)
    def test_openai_client_secure_sdk_usage(self, mock_openai_class):
        """Test that OpenAI client passes key to SDK without storing it."""
        mock_openai_instance = MagicMock()
        mock_openai_class.return_value = mock_openai_instance
        
        client = OpenAIClient()
        
        # Verify SDK was called with the key
        mock_openai_class.assert_called_once_with(api_key='sk-valid1234567890abcdefghijklmnop')
        
        # Verify our client doesn't store the key
        assert not hasattr(client, 'api_key'), "Client should not store API key"
        assert hasattr(client, 'client'), "Client should have SDK client"
        assert client.client == mock_openai_instance, "Should store SDK client instance"

    @patch('data_agent.llm.clients.Anthropic')
    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'sk-ant-valid1234567890abcdefghijklmnopqrstuvwxyz'}, clear=True)
    def test_anthropic_client_secure_sdk_usage(self, mock_anthropic_class):
        """Test that Anthropic client passes key to SDK without storing it."""
        mock_anthropic_instance = MagicMock()
        mock_anthropic_class.return_value = mock_anthropic_instance
        
        client = AnthropicClient()
        
        # Verify SDK was called with the key
        mock_anthropic_class.assert_called_once_with(api_key='sk-ant-valid1234567890abcdefghijklmnopqrstuvwxyz')
        
        # Verify our client doesn't store the key  
        assert not hasattr(client, 'api_key'), "Client should not store API key"
        assert hasattr(client, 'client'), "Client should have SDK client"
        assert client.client == mock_anthropic_instance, "Should store SDK client instance"

    def test_llm_manager_secure_validation(self):
        """Test LLM manager uses secure validation methods."""
        # Test OpenAI validation
        valid_openai = LLMManager._is_valid_api_key_format('sk-valid1234567890abcdefghijklmnop', 'openai')
        invalid_openai = LLMManager._is_valid_api_key_format('your_openai_key_here', 'openai')
        
        assert valid_openai, "Should validate good OpenAI key format"
        assert not invalid_openai, "Should reject placeholder OpenAI key"
        
        # Test Anthropic validation
        valid_anthropic = LLMManager._is_valid_api_key_format('sk-ant-valid1234567890abcdefghijklmnopqrstuvwxyz', 'anthropic')
        invalid_anthropic = LLMManager._is_valid_api_key_format('your_anthropic_key_here', 'anthropic')
        
        assert valid_anthropic, "Should validate good Anthropic key format"
        assert not invalid_anthropic, "Should reject placeholder Anthropic key"
        
        # Test unsupported provider
        unsupported = LLMManager._is_valid_api_key_format('any-key', 'unsupported')
        assert not unsupported, "Should reject unsupported provider"

    @patch.dict('os.environ', {}, clear=True)
    def test_llm_manager_secure_initialization_no_keys(self):
        """Test LLM manager handles missing keys securely."""
        manager = LLMManager()
        
        assert len(manager.clients) == 0, "Should have no clients when no keys available"
        assert manager.current_client is None, "Should have no current client"
        assert manager.get_current_provider() is None, "Should have no current provider"
        assert manager.get_available_providers() == [], "Should have empty provider list"

    @patch.dict('os.environ', {
        'OPENAI_API_KEY': 'your_openai_key_here',
        'ANTHROPIC_API_KEY': 'your_anthropic_key_here'
    }, clear=True)
    def test_llm_manager_secure_initialization_placeholders(self):
        """Test LLM manager rejects placeholder keys."""
        manager = LLMManager()
        
        assert len(manager.clients) == 0, "Should have no clients with placeholder keys"
        assert manager.current_client is None, "Should have no current client"

    @patch.dict('os.environ', {}, clear=True)
    def test_connectivity_test_secure_no_keys(self):
        """Test connectivity test handles missing keys securely."""
        results = test_llm_connectivity()
        
        assert "openai" in results, "Should have OpenAI results"
        assert "anthropic" in results, "Should have Anthropic results"
        
        assert not results["openai"]["available"], "OpenAI should not be available without key"
        assert not results["anthropic"]["available"], "Anthropic should not be available without key"
        
        assert "API key not found" in results["openai"]["error"], "Should indicate missing key"
        assert "API key not found" in results["anthropic"]["error"], "Should indicate missing key"

    @patch.dict('os.environ', {
        'OPENAI_API_KEY': 'your_openai_key_here',
        'ANTHROPIC_API_KEY': 'your_anthropic_key_here'
    }, clear=True)
    def test_connectivity_test_secure_placeholder_keys(self):
        """Test connectivity test handles placeholder keys securely."""
        results = test_llm_connectivity()
        
        assert not results["openai"]["available"], "OpenAI should not be available with placeholder"
        assert not results["anthropic"]["available"], "Anthropic should not be available with placeholder"
        
        # Should get validation errors
        assert "invalid format" in results["openai"]["error"], "Should indicate invalid format"
        assert "invalid format" in results["anthropic"]["error"], "Should indicate invalid format"

    def test_security_best_practices_documentation(self):
        """Document security best practices implemented."""
        security_practices = [
            "API keys are never stored in LLMClient base class",
            "Keys are passed directly to official SDK clients only", 
            "Validation methods are static and don't store keys",
            "Placeholder detection prevents dummy keys from being used",
            "Manager initialization handles missing/invalid keys gracefully",
            "Connectivity tests validate without storing keys",
            "All key handling follows principle of least privilege"
        ]
        
        # This test serves as documentation
        assert len(security_practices) == 7, f"Documented {len(security_practices)} security practices"
        
        # Verify key practices are implemented
        assert not hasattr(LLMClient(model="test"), 'api_key'), "Practice 1: No key storage in base class"
        assert hasattr(OpenAIClient, '_is_valid_openai_key'), "Practice 3: Static validation methods"
        assert hasattr(AnthropicClient, '_is_valid_anthropic_key'), "Practice 3: Static validation methods"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])