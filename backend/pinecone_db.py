import os
import time
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as PineconeStore
from langchain_community.embeddings import HuggingFaceEmbeddings
import pinecone

class PineconeDB:
    def __init__(self):
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "mine-meets")
        self.namespace = os.getenv("PINECONE_NAMESPACE", "default")

        # HuggingFace embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

        # Create Pinecone client
        self.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

        # Initialize index if not present
        self._init_pinecone()

        # Setup vector store for LangChain
        self.vector_store = pinecone.Pinecone.Index(self.index_name)

    def _init_pinecone(self):
        """Create Pinecone index if it doesn't exist."""
        if self.index_name not in [index.name for index in self.pc.list_indexes()]:
            # Create index if doesn't exist
            self.pc.create_index(
                name=self.index_name,
                dimension=384,  # For all-MiniLM-L6-v2
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-west-2"
                )
            )
            
            # Wait for index to initialize
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)

    def _get_vector_store(self):
        """Return a LangChain-compatible Pinecone vector store."""
        return PineconeStore.from_existing_index(
            index_name=self.index_name,
            embedding=self.embeddings,
            namespace=self.namespace,
            text_key="text"
        )