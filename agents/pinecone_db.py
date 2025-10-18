import os
import time
import logging
import threading
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field, validator
from pinecone import Pinecone  # Use the new Pinecone client
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class PineconeConfig(BaseModel):
    api_key: str = Field(..., description="Pinecone API key")
    environment: Optional[str] = Field(None, description="Pinecone environment")
    index_name: str = Field("meetings", description="Name of the Pinecone index")
    dimension: int = Field(512, description="Dimension of the embeddings")
    metric: str = Field("cosine", description="Distance metric")
    serverless: bool = Field(False, description="Use serverless index")
    cloud: Optional[str] = Field("aws", description="Cloud provider for serverless")
    region: Optional[str] = Field("us-west-2", description="Region for serverless")
    pod_type: Optional[str] = Field("starter", description="Pod type for pod-based index")
    timeout: int = Field(60, description="Timeout for index readiness in seconds")
    max_retries: int = Field(3, description="Number of retries for connecting")

    @validator('metric')
    def check_metric(cls, v):
        valid = {"cosine", "dotproduct", "euclidean"}
        if v.lower() not in valid:
            raise ValueError(f"metric must be one of {valid}")
        return v.lower()


class PineconeDB:
    _instance: Optional['PineconeDB'] = None
    _lock = threading.Lock()

    def __new__(cls: Type['PineconeDB'], *args, **kwargs) -> 'PineconeDB':
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
                cls._instance._config = PineconeConfig(
                    api_key=os.getenv("PINECONE_API_KEY", ""),
                    environment=os.getenv("PINECONE_ENVIRONMENT"),
                    index_name=os.getenv("PINECONE_INDEX", "mine-meets"),  # from screenshot
                    dimension=int(os.getenv("EMBEDDING_DIM", "512")),    # match CLIP ViT-B/32
                    metric=os.getenv("PINECONE_METRIC", "cosine"),
                    serverless=True,  # serverless index per screenshot
                    cloud="aws",
                    region="us-east-1",  # match screenshot
                    pod_type=None,
                    timeout=int(os.getenv("PINECONE_TIMEOUT", "60")),
                    max_retries=int(os.getenv("PINECONE_MAX_RETRIES", "3")),
                )
        return cls._instance

    def __init__(self) -> None:
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._init_client_and_index()
                    self._initialized = True

    def _init_client_and_index(self) -> None:
        if not self._config.api_key:
            raise ValueError("Missing PINECONE_API_KEY")

        # Use new Pinecone client API
        self._pc = Pinecone(api_key=self._config.api_key)

        # List existing indexes
        existing = self._pc.list_indexes()
        index_names = [idx.get("name") for idx in existing]

        if self._config.index_name not in index_names:
            logger.info(f"Creating index: {self._config.index_name}")
            self._pc.create_index(
                name=self._config.index_name,
                dimension=self._config.dimension,
                metric=self._config.metric,
                spec={
                    "serverless": {
                        "cloud": self._config.cloud,
                        "region": self._config.region,
                        "capacity": "starter"
                    }
                }
            )
            # Wait for readiness
            start = time.time()
            while time.time() - start < self._config.timeout:
                desc = self._pc.describe_index(self._config.index_name)
                status = desc.get("status", {})
                if status.get("ready", False):
                    logger.info(f"Index {self._config.index_name} ready")
                    break
                time.sleep(5)
            else:
                raise TimeoutError(f"Index {self._config.index_name} not ready in time")

        self._index = self._pc.Index(self._config.index_name)
        self._embeddings = None

    def _ensure_embeddings(self):
        if self._embeddings is None:
            model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/clip-ViT-B-32")
            # Ensure Pinecone index dimension matches this model (512)
            self._embeddings = SentenceTransformer(model_name)
        return self._embeddings

    def upsert_documents(self, documents: List[Dict[str, Any]], namespace: Optional[str] = None, batch_size: int = 64) -> List[str]:
        """
        Embed each document's text and upsert to Pinecone with metadata including the original text.
        documents: List of {"text": str, "metadata": dict, (optional) "id": str}
        Returns list of upserted IDs.
        """
        texts = [d.get("text", "") for d in documents]
        metas = [d.get("metadata", {}) or {} for d in documents]
        ids = [d.get("metadata", {}).get("chunk_id") or d.get("id") or str(i) for i, d in enumerate(documents)]
        model = self._ensure_embeddings()
        vectors = model.encode(texts, normalize_embeddings=True)
        payload = []
        for i, (vid, vec, meta, text) in enumerate(zip(ids, vectors, metas, texts)):
            m = dict(meta)
            # store text to enable showing sources
            if "text" not in m:
                m["text"] = text
            payload.append({"id": vid, "values": vec.tolist(), "metadata": m})
            # batch later
        # upsert in chunks
        out_ids = []
        for start in range(0, len(payload), batch_size):
            chunk = payload[start:start+batch_size]
            self._index.upsert(vectors=chunk, namespace=namespace)
            out_ids.extend([item["id"] for item in chunk])
        return out_ids

    def delete_vectors(self, ids: Optional[List[str]] = None, namespace: Optional[str] = None, delete_all: bool = False, filter: Optional[Dict[str, Any]] = None) -> None:
        self._index.delete(ids=ids, namespace=namespace, deleteAll=delete_all, filter=filter)

    def upsert_vectors(self, items: List[Dict[str, Any]], namespace: Optional[str] = None) -> None:
        """
        Upsert precomputed vectors directly to Pinecone.
        items: List of dicts with keys: id (str), values (List[float]), metadata (dict)
        """
        vectors = []
        for it in items:
            vectors.append({
                "id": it["id"],
                "values": it["values"],
                "metadata": it.get("metadata", {})
            })
        self._index.upsert(vectors=vectors, namespace=namespace)

    def query_text(self, query: str, namespace: Optional[str] = None, top_k: int = 5, filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        model = self._ensure_embeddings()
        qvec = model.encode([query], normalize_embeddings=True)[0].tolist()
        res = self._index.query(vector=qvec, namespace=namespace, top_k=top_k, include_values=False, include_metadata=True, filter=filter)
        matches = res.get("matches", [])
        out = []
        for m in matches:
            mid = m.get("id")
            score = m.get("score")
            meta = m.get("metadata", {})
            out.append({"id": mid, "score": score, "metadata": meta})
        return out

    def get_index_stats(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        try:
            return self._index.describe_index_stats(namespace=namespace)
        except Exception as e:
            logger.warning(f"Failed to get index stats: {e}")
            return {}

    def __del__(self) -> None:
        try:
            del self._index
            del self._pc
        except Exception:
            pass
        finally:
            self._initialized = False
