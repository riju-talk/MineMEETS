"""
Unit tests for document processor
"""
import pytest
from agents.document_processor import DocumentProcessor


class TestDocumentProcessor:
    """Test cases for DocumentProcessor."""

    def setup_method(self):
        """Setup test fixtures."""
        self.processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)

    def test_split_text_simple(self):
        """Test simple text splitting."""
        text = "This is a test. " * 20  # 80 words
        chunks = self.processor._split_text(text)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert all(len(chunk) <= self.processor.chunk_size for chunk in chunks)

    def test_split_empty_text(self):
        """Test splitting empty text."""
        chunks = self.processor._split_text("")
        assert len(chunks) == 0

    def test_chunk_metadata(self):
        """Test that chunks have proper metadata."""
        text = "Test content"
        result = self.processor.split_text_simple(text, meeting_id="test_meeting")
        
        assert len(result) > 0
        assert "text" in result[0]
        assert "metadata" in result[0]
        assert result[0]["metadata"]["meeting_id"] == "test_meeting"

    def test_supported_extensions(self):
        """Test that supported extensions are defined."""
        assert ".txt" in self.processor.supported_extensions
        assert ".pdf" in self.processor.supported_extensions
        assert ".docx" in self.processor.supported_extensions
