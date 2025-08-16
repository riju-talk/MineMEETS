# agents/qa_agent.py
from typing import Optional, Dict, Any
from .base_agent import BaseAgent, AgentResponse
from agents.pinecone_db import PineconeDB
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from agents.llm import get_llm_provider

class QAAgent(BaseAgent):
    def __init__(self, pinecone_db: PineconeDB):
        super().__init__(name="qa_agent", description="RAG QA agent")
        self.pinecone_db = pinecone_db
        self.llm = get_llm_provider()
        self.retriever = self.pinecone_db.get_retriever(k=5)

        # Try to load best prompt template from /prompts, fallback to default
        prompt_path = 'prompts/user_centric_prompt.txt' if os.path.exists('prompts/user_centric_prompt.txt') else None
        if prompt_path:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt_text = f.read()
            prompt_text += "\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        else:
            prompt_text = (
                "You are a concise assistant. Use only the provided context to answer. "
                "If you don't know, say you don't know.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
            )
        PROMPT = PromptTemplate(template=prompt_text, input_variables=["context", "question"])
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

    async def process(self, question: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        try:
            if context and "meeting_id" in context:
                self.retriever.search_kwargs["filter"] = {"meeting_id": context["meeting_id"]}
            result = self.qa_chain({"query": question})
            return AgentResponse(success=True, content={
                "answer": result["result"],
                "sources": [{"content": d.page_content, "metadata": d.metadata} for d in result["source_documents"]]
            })
        except Exception as e:
            return AgentResponse(success=False, content=f"QA failed: {e}")
