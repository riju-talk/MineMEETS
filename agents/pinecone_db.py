# backend/vector_store/pinecone_store.py
import os
import time
from typing import List, Dict, Optional, Iterable, Tuple
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings

# ---- Singleton metaclass ----
class _Singleton(type):
    _instances: Dict[Tuple[str, str], "PineconeDB"] = {}
    def __call__(cls, *args, **kwargs):
        # one instance per (index_name, namespace)
        key = (kwargs.get("index_name") or os.getenv("PINECONE_INDEX_NAME", "mine-meets"),
               kwargs.get("namespace") or os.getenv("PINECONE_NAMESPACE", "default"))
        if key not in cls._instances:
            cls._instances[key] = super().__call__(*args, **kwargs)
        return cls._instances[key]

class PineconeDB(metaclass=_Singleton):
    """
    Pinecone DB wrapper:
    - Creates index if missing (serverless).
    - Uses high-dim embeddings (default BGE-large 1024-d).
    - Upsert is idempotent (skips existing ids).
    - Provides LangChain VectorStore + retriever helper.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        namespace: Optional[str] = None,
        cloud: str = None,
        region: str = None,
        embedding_model: str = None,
        embedding_device: str = "cpu",
        metric: str = "cosine",
        dim: Optional[int] = None,
    ):
        # --- Config from env w/ sane defaults ---
        self.api_key     = api_key     or os.getenv("PINECONE_API_KEY")
        self.index_name  = index_name  or os.getenv("PINECONE_INDEX_NAME", "mine-meets")
        self.namespace   = namespace   or os.getenv("PINECONE_NAMESPACE", "default")
        self.cloud       = cloud       or os.getenv("PINECONE_CLOUD", "aws")
        self.region      = region      or os.getenv("PINECONE_REGION", "ap-southeast-1")  # SG is good for India
        self.metric      = metric      or os.getenv("PINECONE_METRIC", "cosine")

        # High-dimensional default (1024)
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
        self.dim             = int(dim or os.getenv("EMBEDDING_DIM", "1024"))

        if not self.api_key:
            raise ValueError("PINECONE_API_KEY is required")

        # --- Embeddings (HF, high-dim) ---
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": embedding_device},
            encode_kwargs={"normalize_embeddings": True},  # cosine works best when normalized
        )

        # --- Pinecone client (v5) ---
        self.pc = Pinecone(api_key=self.api_key)

        # --- Create index if needed ---
        self._ensure_index()

        # --- Connected index handle ---
        self.index = self.pc.Index(self.index_name)

        # --- LangChain VectorStore helper (lazy)
        self._vector_store = None

    # ---------- Index lifecycle ----------
    def _ensure_index(self):
        names = self.pc.list_indexes().names()
        if self.index_name in names:
            # already exists; sanity check dimension & metric?
            return

        self.pc.create_index(
            name=self.index_name,
            dimension=self.dim,
            metric=self.metric,
            spec=ServerlessSpec(cloud=self.cloud, region=self.region),
        )

        # Wait until ready
        while True:
            desc = self.pc.describe_index(self.index_name)
            if getattr(desc, "status", {}).get("ready"):
                break
            time.sleep(1)

    # ---------- LangChain VectorStore ----------
    def get_vector_store(self) -> PineconeVectorStore:
        if self._vector_store is None:
            self._vector_store = PineconeVectorStore(
                index_name=self.index_name,
                namespace=self.namespace,
                embedding=self.embeddings,
                text_key="text",  # we will store the raw text under metadata["text"]
            )
        return self._vector_store

    def as_retriever(self, k: int = 5):
        return self.get_vector_store().as_retriever(search_kwargs={"k": k})

    # ---------- Upsert (idempotent) ----------
    def upsert_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 128,
        skip_if_exists: bool = True,
    ) -> int:
        """
        Upsert texts as vectors with optional metadata and IDs.
        - If `skip_if_exists=True`, existing IDs are detected and skipped (no overwrite).
        Returns number of new records actually upserted.
        """
        if not texts:
            return 0
        if metadatas and len(metadatas) != len(texts):
            raise ValueError("metadatas length must match texts length")
        if ids and len(ids) != len(texts):
            raise ValueError("ids length must match texts length")

        # build default metadatas and ids
        if metadatas is None:
            metadatas = [{} for _ in texts]
        if ids is None:
            # deterministic ids from namespace + ordinal
            ids = [f"{self.namespace}-{i}" for i in range(len(texts))]

        # optional dedupe by checking existing IDs
        new_texts, new_metas, new_ids = texts, metadatas, ids
        if skip_if_exists:
            fetch = self.index.fetch(ids=ids, namespace=self.namespace)
            existing = set((fetch or {}).get("vectors", {}).keys())
            if existing:
                new_texts, new_metas, new_ids = [], [], []
                for t, m, _id in zip(texts, metadatas, ids):
                    if _id not in existing:
                        new_texts.append(t)
                        m = dict(m or {})
                        m.setdefault("text", t)  # ensure text is present for LangChain text_key
                        new_metas.append(m)
                        new_ids.append(_id)
            else:
                # add text to metadata
                new_metas = [dict(m, **{"text": t}) for t, m in zip(texts, metadatas)]

        if not new_texts:
            return 0

        # embed and upsert in batches
        total = 0
        for i in range(0, len(new_texts), batch_size):
            chunk_texts = new_texts[i : i + batch_size]
            chunk_ids   = new_ids[i : i + batch_size]
            chunk_meta  = new_metas[i : i + batch_size]
            vecs = self.embeddings.embed_documents(chunk_texts)  # List[List[float]]

            # Pinecone upsert format
            vectors = [
                {"id": _id, "values": v, "metadata": m}
                for _id, v, m in zip(chunk_ids, vecs, chunk_meta)
            ]
            self.index.upsert(vectors=vectors, namespace=self.namespace)
            total += len(vectors)

        return total

    # ---------- Query (raw) ----------
    def query(self, query_text: str, top_k: int = 5):
        qvec = self.embeddings.embed_query(query_text)
        res = self.index.query(
            vector=qvec,
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace,
        )
        # normalize a convenient structure
        matches = []
        for m in getattr(res, "matches", []):
            matches.append({
                "id": m.id,
                "score": m.score,
                "text": (m.metadata or {}).get("text", ""),
                "metadata": m.metadata or {},
            })
        return matches
