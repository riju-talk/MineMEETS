# agents/internet_agent.py
from typing import Dict, Any, Optional, List
import asyncio
import logging
from duckduckgo_search import DDGS  # Correct import for duckduckgo_search
from agents.llm import LLM

class InternetAgent:
    """Internet agent using DuckDuckGo for search and LLM for summarization."""

    def __init__(self):
        self.llm = LLM(model="llama3.1", temperature=0.0)
        self.logger = logging.getLogger(__name__)

    def _ddg_search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Synchronous DuckDuckGo search using DDGS."""
        results = []
        try:
            with DDGS() as ddg:
                for result in ddg.text(query, max_results=max_results):
                    results.append(result)
        except Exception as e:
            self.logger.error(f"DuckDuckGo search failed: {e}")
        return results

    async def _run_gen(self, prompt: str) -> str:
        """Run generation using LLM."""
        return await self.llm.generate_async(prompt)

    async def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform a web search with DuckDuckGo and summarize top results.

        Returns:
          Dict with fields: answer (string), sources (list of dicts with title/url/snippet)
        """
        try:
            # Build a search query; optionally include meeting context for more targeted search
            if context and "meeting_context" in context:
                meeting_meta = context["meeting_context"]
                query = f"{query} (Meeting context: {meeting_meta})"

            # DuckDuckGo search (synchronous, wrapped in thread)
            results = await asyncio.to_thread(self._ddg_search, query, max_results=5)
            if not results:
                return {"success": True, "content": {"answer": "", "sources": []}}

            # Build a context string containing titles and snippets
            snippet_texts = []
            sources = []
            for r in results:
                title = r.get("title") or ""
                href = r.get("href") or r.get("url") or ""
                body = r.get("body") or ""
                snippet = f"{title}\n{body}\nURL: {href}"
                snippet_texts.append(snippet)
                sources.append({"title": title, "url": href, "snippet": body})

            context_for_model = "\n\n---\n\n".join(snippet_texts)
            prompt = (
                "You are an assistant that summarizes search results. "
                "Given the following search snippets, produce a concise answer to the user's query and list top sources.\n\n"
                f"Search snippets:\n{context_for_model}\n\nQuestion: {query}\n\nAnswer (brief):"
            )

            answer = await self._run_gen(prompt)
            return {"success": True, "content": {"answer": answer.strip(), "sources": sources}}

        except Exception as e:
            self.logger.exception("InternetAgent failed")
            return {"success": False, "content": f"Error performing internet search: {e}"}
