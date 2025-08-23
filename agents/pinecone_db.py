import os
import time
import logging
import threading
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field, validator
import pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.schema import BaseRetriever, Document
from langchain.embeddings.base import Embeddings

logger = logging.getLogger(__name__)


class PineconeConfig(BaseModel):
    api_key: str = Field(..., description="Pinecone API key")
    environment: Optional[str] = Field(None, description="Pinecone environment")
    index_name: str = Field("meetings", description="Name of the Pinecone index")
    dimension: int = Field(1536, description="Dimension of the embeddings")
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
                    dimension=int(os.getenv("EMBEDDING_DIM", "1024")),    # from screenshot or pinecone model config
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

        pinecone.init(api_key=self._config.api_key, environment=self._config.environment)

        # Index create/connect (v3+ usage: always handle existing)
        existing = pinecone.list_indexes()
        if self._config.index_name not in existing:
            logger.info(f"Creating index: {self._config.index_name}")
            pinecone.create_index(
                name=self._config.index_name,
                dimension=self._config.dimension,
                metric=self._config.metric,
                spec={
                    "serverless": {
                        "cloud": self._config.cloud,
                        "region": self._config.region,
                        "capacity": "starter"  # safely default starter plan
                    },
                    "model": "llama-text-embed-v2"  # from your screenshot
                }
            )
            # Wait for readiness
            start = time.time()
            while time.time() - start < self._config.timeout:
                desc = pinecone.describe_index(self._config.index_name)
                status = desc.get("status", {})
                if status.get("ready", False):
                    logger.info(f"Index {self._config.index_name} ready")
                    break
                time.sleep(5)
            else:
                raise TimeoutError(f"Index {self._config.index_name} not ready in time")

        self._index = pinecone.Index(self._config.index_name)

    def get_retriever(self, embeddings: Embeddings, namespace: Optional[str] = None, k: int = 4, score_threshold: Optional[float] = None) -> BaseRetriever:
        vs = PineconeVectorStore(
            index=self._index,
            embedding=embeddings,
            namespace=namespace
        )
        if score_threshold is not None:
            return vs.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": k, "score_threshold": score_threshold})
        else:
            return vs.as_retriever(search_kwargs={ "k": k })

    def upsert_documents(self, documents: List[Document], embeddings: Embeddings, namespace: Optional[str] = None, batch_size: int = 64) -> List[str]:
        vs = PineconeVectorStore(
            index=self._index,
            embedding=embeddings,
            namespace=namespace
        )
        return vs.add_documents(documents=documents, batch_size=batch_size)

    def delete_vectors(self, ids: Optional[List[str]] = None, namespace: Optional[str] = None, delete_all: bool = False, filter: Optional[Dict[str, Any]] = None) -> None:
        vs = PineconeVectorStore(index=self._index, embedding=None, namespace=namespace)
        vs.delete(ids=ids, delete_all=delete_all, filter=filter)

    def get_index_stats(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        return self._client.describe_index_stats(name=self._config.index_name, namespace=namespace)

    def __del__(self) -> None:
        try:
            del self._index
            del self._client
        except Exception:
            pass
        finally:
            self._initialized = False
