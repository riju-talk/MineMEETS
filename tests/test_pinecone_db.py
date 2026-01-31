"""
Unit tests for Pinecone database interface
"""
import pytest
from unittest.mock import Mock, patch
from agents.pinecone_db import PineconeDB


class TestPineconeDB:
    """Test cases for PineconeDB."""

    @patch('agents.pinecone_db.pinecone.Pinecone')
    @patch('agents.pinecone_db.SentenceTransformer')
    def test_initialization(self, mock_st, mock_pinecone):
        """Test PineconeDB initialization."""
        mock_pinecone.return_value.list_indexes.return_value = [{'name': 'test-index'}]
        
        with patch.dict('os.environ', {'PINECONE_API_KEY': 'test-key'}):
            db = PineconeDB(index_name='test-index')
            
            assert db.index_name == 'test-index'
            assert db.api_key == 'test-key'

    @patch('agents.pinecone_db.pinecone.Pinecone')
    @patch('agents.pinecone_db.SentenceTransformer')
    def test_embed_text(self, mock_st, mock_pinecone):
        """Test text embedding."""
        mock_pinecone.return_value.list_indexes.return_value = [{'name': 'test-index'}]
        mock_st.return_value.encode.return_value = [[0.1] * 512]
        
        with patch.dict('os.environ', {'PINECONE_API_KEY': 'test-key'}):
            db = PineconeDB(index_name='test-index')
            
            embedding = db.embed_text("test text")
            
            assert isinstance(embedding, list)
            assert len(embedding) == 512
