import os
import logging
from typing import Any, Dict, List, Optional
import pinecone
from sentence_transformers import SentenceTransformer

# LangChain imports
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings

logger = logging.getLogger(__name__)

class CustomSentenceTransformerEmbeddings(SentenceTransformerEmbeddings):
    """Custom wrapper for SentenceTransformer to work with LangChain."""

    def __init__(self, model_name: str = "sentence-transformers/clip-ViT-B-32"):
        super().__init__(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self.model.encode([text])[0].tolist()

class PineconeDB:
    def __init__(self, api_key: Optional[str] = None, index_name: str = "mine-meets"):
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY must be set")
        self.index_name = index_name
        self.pc = pinecone.Pinecone(api_key=self.api_key)

        # Check if index exists, create if not
        existing_indexes = self.pc.list_indexes()
        if self.index_name not in [idx['name'] for idx in existing_indexes]:
            logger.info(f"Creating new Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=512,  # CLIP ViT-B/32 dimension
                metric="cosine",
                spec=pinecone.ServerlessSpec(cloud="aws", region="us-west-2")
            )
            # Wait for index to be ready
            import time
            time.sleep(10)

        # Initialize LangChain components
        self.index = self.pc.Index(self.index_name)
        self.embeddings = CustomSentenceTransformerEmbeddings("sentence-transformers/clip-ViT-B-32")
        self.vector_store = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings,
            text_key="text"
        )

        logger.info(f"PineconeDB initialized with index: {self.index_name}")

    def upsert_documents(self, documents: List[Dict[str, Any]], namespace: str = "") -> None:
        """Upsert text documents with embeddings using LangChain."""
        try:
            # Convert to LangChain documents
            lc_documents = []
            for doc in documents:
                text = doc.get("text", "").strip()
                metadata = doc.get("metadata", {})
                if text:
                    lc_documents.append(Document(
                        page_content=text,
                        metadata=metadata
                    ))

            if lc_documents:
                # Use LangChain's add_documents with namespace support
                # Note: LangChain PineconeVectorStore doesn't directly support namespaces in add_documents
                # So we need to handle this differently
                if namespace:
                    # For namespaced documents, we need to add namespace to metadata
                    for doc in lc_documents:
                        doc.metadata["namespace"] = namespace

                self.vector_store.add_documents(lc_documents)
                logger.info(f"Successfully upserted {len(lc_documents)} documents to namespace '{namespace}'")
            else:
                logger.warning("No valid documents to upsert")

        except Exception as e:
            logger.error(f"Error upserting documents: {str(e)}")
            raise

    def upsert_vectors(self, vectors: List[Dict[str, Any]], namespace: str = "") -> None:
        """Upsert pre-computed vectors."""
        try:
            if not vectors:
                logger.warning("No vectors provided for upsert")
                return

            # Upsert in batches using direct Pinecone API for backward compatibility
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch, namespace=namespace)

            logger.info(f"Successfully upserted {len(vectors)} vectors to namespace '{namespace}'")

        except Exception as e:
            logger.error(f"Error upserting vectors: {str(e)}")
            raise

    def query_text(self, query: str, namespace: str = "", top_k: int = 5) -> List[Dict[str, Any]]:
        """Query by text using embeddings with LangChain."""
        try:
            # Use LangChain's similarity_search_with_score
            docs_and_scores = self.vector_store.similarity_search_with_score(
                query,
                k=top_k,
                namespace=namespace
            )

            # Convert back to original format
            results = []
            for doc, score in docs_and_scores:
                results.append({
                    "id": doc.metadata.get("chunk_id", doc.id),
                    "score": score,
                    "metadata": doc.metadata,
                    "text": doc.page_content
                })

            return results

        except Exception as e:
            logger.error(f"Error querying text: {str(e)}")
            return []

    def similarity_search(self, query: str, namespace: str = "", top_k: int = 5, **kwargs) -> List[Document]:
        """LangChain-style similarity search."""
        return self.vector_store.similarity_search(
            query,
            k=top_k,
            namespace=namespace,
            **kwargs
        )

    def similarity_search_with_score(self, query: str, namespace: str = "", top_k: int = 5, **kwargs) -> List[tuple]:
        """LangChain-style similarity search with scores."""
        return self.vector_store.similarity_search_with_score(
            query,
            k=top_k,
            namespace=namespace,
            **kwargs
        )

    def delete_vectors(self, ids: Optional[List[str]] = None, namespace: str = "", delete_all: bool = False) -> None:
        """Delete vectors by IDs or all vectors with proper error handling."""
        try:
            if delete_all:
                # Delete all vectors in the namespace
                self.index.delete(delete_all=True, namespace=namespace)
                logger.info(f"Deleted all vectors from namespace '{namespace}'")
            elif ids:
                # Delete specific vectors
                self.index.delete(ids=ids, namespace=namespace)
                logger.info(f"Deleted {len(ids)} vectors from namespace '{namespace}'")
            else:
                logger.warning("No deletion criteria provided")

        except Exception as e:
            logger.error(f"Error deleting vectors: {str(e)}")
            raise

    def flush_database(self) -> bool:
        """Flush the entire database (all namespaces)."""
        try:
            # Get all namespaces
            stats = self.get_index_stats()
            namespaces = stats.get('namespaces', {})

            # Delete from each namespace
            for namespace in namespaces.keys():
                self.index.delete(delete_all=True, namespace=namespace)
                logger.info(f"Flushed namespace: {namespace}")

            # Also delete from default namespace
            self.index.delete(delete_all=True, namespace="")

            logger.info("Database flushed successfully")
            return True

        except Exception as e:
            logger.error(f"Error flushing database: {str(e)}")
            return False

    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        try:
            return self.index.describe_index_stats()
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {}

    def get_total_vector_count(self) -> int:
        """Get total number of vectors across all namespaces."""
        try:
            stats = self.get_index_stats()
            return stats.get('total_vector_count', 0)
        except Exception as e:
            logger.error(f"Error getting total vector count: {str(e)}")
            return 0