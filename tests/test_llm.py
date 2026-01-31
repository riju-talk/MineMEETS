"""
Unit tests for LLM interface
"""
import pytest
from unittest.mock import Mock, patch
from agents.llm import LLM


class TestLLM:
    """Test cases for LLM wrapper."""

    def test_initialization(self):
        """Test LLM initialization."""
        llm = LLM(model="llama3.1", temperature=0.0)
        
        assert llm.model == "llama3.1"
        assert llm.temperature == 0.0
        assert "localhost" in llm.base_url

    @patch('agents.llm.requests.post')
    def test_generate(self, mock_post):
        """Test synchronous generation."""
        mock_response = Mock()
        mock_response.json.return_value = {"response": "Test answer"}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        llm = LLM()
        result = llm.generate("Test prompt")
        
        assert result == "Test answer"
        mock_post.assert_called_once()

    @patch('agents.llm.requests.post')
    def test_generate_error(self, mock_post):
        """Test generation error handling."""
        mock_post.side_effect = Exception("Connection error")
        
        llm = LLM()
        
        with pytest.raises(RuntimeError):
            llm.generate("Test prompt")
