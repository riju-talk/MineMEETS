# agents/qa_agent.py
import asyncio
from typing import Optional, Dict, Any, List
from agents.pinecone_db import PineconeDB
from agents.llm import LLM

class QAAgent:
    def __init__(self, pinecone_db: PineconeDB):
        self.pinecone_db = pinecone_db
        self.llm = LLM(model="llama3.1", temperature=0.0)
        # Prompt template (string format)
        base = (
            "You are a concise assistant. Use only the provided context to answer. "
            "If you don't know, say you don't know."
        )
        self.prompt_template = (
            f"{base}\n\nContext:\n{{context}}\n\nQuestion: {{question}}\nAnswer:"
        )

    async def process(self, question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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

            # Use LLM directly
            answer = await self.llm.generate_async(prompt)

            return {"success": True, "content": {
                "answer": answer.strip(),
                "sources": sources
            }}
        except Exception as e:
            return {"success": False, "content": f"QA failed: {e}"}
