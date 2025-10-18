# agents/qa_agent.py
import os
from typing import Optional, Dict, Any, List
from .base_agent import BaseAgent, AgentResponse
from agents.pinecone_db import PineconeDB
from langchain_ollama import ChatOllama

class QAAgent(BaseAgent):
    def __init__(self, pinecone_db: PineconeDB):
        super().__init__(name="qa_agent", description="RAG QA agent")
        self.pinecone_db = pinecone_db
        self.llm = ChatOllama(
            model="llama3.1",
            temperature=0.0,
        )
        # Prompt template (string format)
        base = (
            "You are a concise assistant. Use only the provided context to answer. "
            "If you don't know, say you don't know."
        )
        self.prompt_template = (
            f"{base}\n\nContext:\n{{context}}\n\nQuestion: {{question}}\nAnswer:"
        )

    async def process(self, question: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        try:
            meeting_id = context.get("meeting_id") if context else None
            namespace = meeting_id if meeting_id else None
            # Retrieve top contexts
            hits = self.pinecone_db.query_text(question, namespace=namespace, top_k=5)
            contexts: List[str] = []
            sources: List[Dict[str, Any]] = []
            for h in hits:
                meta = h.get("metadata", {}) or {}
                text = meta.get("text", "")
                if text:
                    contexts.append(text)
                sources.append({"content": text, "metadata": meta})

            context_block = "\n\n---\n".join(contexts) if contexts else ""
            prompt = self.prompt_template.format(context=context_block, question=question)

            # Use ChatOllama directly
            from langchain_core.messages import HumanMessage
            message = HumanMessage(content=prompt)
            response = self.llm.invoke([message])
            answer = response.content.strip()

            return AgentResponse(success=True, content={
                "answer": answer,
                "sources": sources
            })
        except Exception as e:
            return AgentResponse(success=False, content=f"QA failed: {e}")
