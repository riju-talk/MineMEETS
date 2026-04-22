"""Unit tests for Pinecone database interface."""

from unittest.mock import Mock, patch
import pytest

from agents.pinecone_db import PineconeDB


class TestPineconeDB:
    """Test cases for PineconeDB."""

    @patch("agents.pinecone_db.Pinecone")
    @patch("agents.pinecone_db.SentenceTransformer")
    def test_initialization(self, mock_st, mock_pinecone):
        """PineconeDB initializes with env API key and existing index."""
        mock_pinecone.return_value.list_indexes.return_value = [{"name": "test-index"}]

        with patch.dict("os.environ", {"PINECONE_API_KEY": "test-key-12345"}):
            db = PineconeDB(index_name="test-index")

            assert db.index_name == "test-index"
            assert db.api_key == "test-key-12345"
            mock_st.assert_called_once()

    @patch("agents.pinecone_db.Pinecone")
    @patch("agents.pinecone_db.SentenceTransformer")
    def test_embed_text(self, mock_st, mock_pinecone):
        """Text embedding returns expected vector shape."""
        mock_pinecone.return_value.list_indexes.return_value = [{"name": "test-index"}]

        fake_embedding = Mock()
        fake_embedding.tolist.return_value = [0.1] * 512
        mock_st.return_value.encode.return_value = [fake_embedding]

        with patch.dict("os.environ", {"PINECONE_API_KEY": "test-key-12345"}):
            db = PineconeDB(index_name="test-index")
            embedding = db.embed_text("test text")

            assert isinstance(embedding, list)
            assert len(embedding) == 512

    @patch("agents.pinecone_db.Pinecone")
    @patch("agents.pinecone_db.SentenceTransformer")
    def test_upsert_vectors_rejects_wrong_dimension(self, mock_st, mock_pinecone):
        """Upsert should enforce strict embedding dimension checks."""
        mock_pinecone.return_value.list_indexes.return_value = [{"name": "test-index"}]

        with patch.dict(
            "os.environ", {"PINECONE_API_KEY": "test-key-12345", "EMBEDDING_DIM": "512"}
        ):
            db = PineconeDB(index_name="test-index")

            bad_vectors = [{"id": "v1", "values": [0.1] * 64, "metadata": {"type": "text_chunk"}}]

            with pytest.raises(ValueError, match="Invalid embedding dimension"):
                db.upsert_vectors(bad_vectors, namespace="meeting_1")
