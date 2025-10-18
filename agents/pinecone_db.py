import os
import logging
from typing import Any, Dict, List, Optional
import pinecone
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

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
        
        self.index = self.pc.Index(self.index_name)
        self.embedding_model = SentenceTransformer("sentence-transformers/clip-ViT-B-32")
        logger.info(f"PineconeDB initialized with index: {self.index_name}")

    def upsert_documents(self, documents: List[Dict[str, Any]], namespace: str = "") -> None:
        """Upsert text documents with embeddings."""
        try:
            vectors = []
            for doc in documents:
                text = doc.get("text", "").strip()
                metadata = doc.get("metadata", {})
                if text:
                    embedding = self.embedding_model.encode([text])[0].tolist()
                    vectors.append({
                        "id": metadata.get("chunk_id", f"doc_{len(vectors)}"),
                        "values": embedding,
                        "metadata": metadata
                    })
            
            if vectors:
                # Upsert in batches to avoid payload limits
                batch_size = 100
                for i in range(0, len(vectors), batch_size):
                    batch = vectors[i:i + batch_size]
                    self.index.upsert(vectors=batch, namespace=namespace)
                
                logger.info(f"Successfully upserted {len(vectors)} documents to namespace '{namespace}'")
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
                
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch, namespace=namespace)
                
            logger.info(f"Successfully upserted {len(vectors)} vectors to namespace '{namespace}'")
            
        except Exception as e:
            logger.error(f"Error upserting vectors: {str(e)}")
            raise

    def query_text(self, query: str, namespace: str = "", top_k: int = 5) -> List[Dict[str, Any]]:
        """Query by text using embeddings."""
        try:
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True
            )
            return response.get("matches", [])
        except Exception as e:
            logger.error(f"Error querying text: {str(e)}")
            return []

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