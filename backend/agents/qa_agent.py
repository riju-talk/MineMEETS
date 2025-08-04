"""QA Agent for answering questions about meetings using vector database."""
from typing import Dict, Any, List, Optional
from ..pinecone_db import PineconeDB
from .base_agent import BaseAgent, AgentResponse
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
import os

class QAAgent(BaseAgent):
    """Agent for handling question-answering about meetings."""
    
    def __init__(self, pinecone_db: PineconeDB):
        """Initialize the QA agent.
        
        Args:
            pinecone_db: Initialized PineconeDB instance
        """
        super().__init__(
            name="qa_agent",
            description="Answers questions about meeting content using vector database"
        )
        self.pinecone_db = pinecone_db
        self.llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0.2,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.retriever = self._create_retriever()
        self.qa_chain = self._create_qa_chain()
    
    def _create_retriever(self):
        """Create a retriever with specific search parameters."""
        return self.pinecone_db.get_retriever(
            search_type="similarity",
            search_kwargs={"k": 5, "score_threshold": 0.7}
        )
    
    def _create_qa_chain(self):
        """Create a QA chain with a custom prompt."""
        prompt_template = """You are a helpful meeting assistant. Use the following meeting context to answer the question at the end. If you don't know the answer, say you don't know, don't try to make up an answer.

        Context:
        {context}

        Question: {question}
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
    
    async def process(self, question: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Process a question and return an answer.
        
        Args:
            question: The question to answer
            context: Optional context including meeting_id, filters, etc.
            
        Returns:
            AgentResponse with the answer and sources
        """
        try:
            # Apply any filters from context
            if context and "meeting_id" in context:
                self.retriever.search_kwargs["filter"] = {"meeting_id": context["meeting_id"]}
            
            # Get the answer
            result = self.qa_chain({"query": question})
            
            return AgentResponse(
                success=True,
                content={
                    "answer": result["result"],
                    "sources": [
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata
                        }
                        for doc in result["source_documents"]
                    ]
                }
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                content=f"Error processing question: {str(e)}"
            )
    
    def add_meeting_context(self, meeting_id: str, documents: List[Dict[str, Any]]):
        """Add meeting documents to the vector store.
        
        Args:
            meeting_id: ID of the meeting
            documents: List of document dictionaries with 'text' and 'metadata'
        """
        texts = [doc["text"] for doc in documents]
        metadatas = [{"meeting_id": meeting_id, **doc.get("metadata", {})} 
                    for doc in documents]
        
        self.pinecone_db.add_documents(
            documents=texts,
            metadatas=metadatas
        )
