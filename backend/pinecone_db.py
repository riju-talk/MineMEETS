"""
Pinecone Database Handler

This module provides functions to interact with Pinecone vector database
for storing and retrieving document embeddings.
"""
import os
from typing import List, Dict, Any, Optional, Union
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as PineconeStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

class PineconeDB:
    """A class to handle Pinecone vector database operations."""
    
    def __init__(self, index_name: str = "minemeets", environment: str = "gcp-starter"):
        """Initialize the Pinecone database handler.
        
        Args:
            index_name: Name of the Pinecone index
            environment: Pinecone environment (e.g., 'gcp-starter')
        """
        self.index_name = index_name
        self.environment = environment
        # Use all-MiniLM-L6-v2 model for embeddings (small but effective)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have a GPU
            encode_kwargs={'normalize_embeddings': False}
        )
        self._init_pinecone()
        self.vector_store = self._get_vector_store()
    
    def _init_pinecone(self) -> None:
        """Initialize Pinecone with API key from environment variables."""
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
        
        self.pc = Pinecone(api_key=api_key)
    
    def _get_vector_store(self) -> PineconeStore:
        """Get or create the Pinecone vector store."""
        if self.index_name not in [index.name for index in self.pc.list_indexes()]:
            self.pc.create_index(
                name=self.index_name,
                dimension=384,  # Dimension for all-MiniLM-L6-v2
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'  # Using us-east-1 which is available in free tier
                )
            )
            
        return PineconeStore.from_existing_index(
            index_name=self.index_name,
            embedding=self.embeddings,
            pinecone_api_key=os.getenv("PINECONE_API_KEY")
        )
    
    def _create_index(self, dimension: int = 1536, metric: str = "cosine") -> None:
        """Create a new Pinecone index.
        
        Args:
            dimension: Dimension of the embeddings (default: 1536 for OpenAI)
            metric: Distance metric for similarity search (default: cosine)
        """
        pinecone.create_index(
            name=self.index_name,
            dimension=dimension,
            metric=metric
        )
    
    def add_documents(
        self,
        documents: List[Union[str, Document]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 100
    ) -> List[str]:
        """Add documents to the vector store.
        
        Args:
            documents: List of text documents or Document objects
            metadatas: Optional list of metadata dicts for each document
            batch_size: Number of documents to add in each batch
            
        Returns:
            List of document IDs
        """
        # Convert strings to Document objects if needed
        if documents and isinstance(documents[0], str):
            if metadatas is None:
                metadatas = [{}] * len(documents)
            documents = [
                Document(page_content=text, metadata=meta)
                for text, meta in zip(documents, metadatas)
            ]
        
        # Add documents to the vector store
        doc_ids = self.vector_store.add_documents(
            documents=documents,
            batch_size=batch_size
        )
        return doc_ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Optional filter to apply to the search
            
        Returns:
            List of matching documents
        """
        return self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter
        )
    
    def get_retriever(self, **kwargs):
        """Get a retriever for the vector store.
        
        Args:
            **kwargs: Additional arguments for the retriever
            
        Returns:
            A retriever object
        """
        return self.vector_store.as_retriever(**kwargs)
    
    def delete_index(self) -> None:
        """Delete the Pinecone index."""
        if self.index_name in pinecone.list_indexes():
            pinecone.delete_index(self.index_name)
    
    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[str]:
        """Split text into chunks.
        
        Args:
            text: Input text to chunk
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_text(text)


# Example usage
if __name__ == "__main__":
    # Initialize the Pinecone database
    pinecone_db = PineconeDB()
    
    # Example: Add documents
    documents = ["This is a test document.", "Another test document."]
    doc_ids = pinecone_db.add_documents(documents)
    print(f"Added documents with IDs: {doc_ids}")
    
    # Example: Search for similar documents
    results = pinecone_db.similarity_search("test", k=2)
    for doc in results:
        print(f"Found document: {doc.page_content}")
    
    # Example: Get a retriever
    retriever = pinecone_db.get_retriever(search_kwargs={"k": 3})
    print(f"Retriever created: {retriever is not None}")
