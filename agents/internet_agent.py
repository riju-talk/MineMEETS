# agents/internet_agent.py
from typing import Dict, Any, Optional, List
import asyncio
import logging
from duckduckgo_search import DDGS
from agents.llm import LLM

logger = logging.getLogger(__name__)


class InternetAgent:
    """Enhanced Internet agent using DuckDuckGo for search and LLM for intelligent summarization."""

    def __init__(self):
        self.llm = LLM(model="llama3.1", temperature=0.0)
        self.search_timeout = 30  # seconds
        self.max_results = 5

    async def _ddg_search_async(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Asynchronous DuckDuckGo search with timeout handling."""
        try:
            # Run synchronous search in thread pool with timeout
            return await asyncio.wait_for(
                asyncio.to_thread(self._ddg_search_sync, query, max_results),
                timeout=self.search_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Search timeout for query: {query}")
            return []
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {str(e)}")
            return []

    def _ddg_search_sync(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Synchronous DuckDuckGo search implementation."""
        results = []
        try:
            with DDGS() as ddg:
                for result in ddg.text(query, max_results=max_results):
                    # Validate and clean results
                    if result.get("title") and result.get("body"):
                        cleaned_result = {
                            "title": result["title"].strip(),
                            "url": result.get("href", ""),
                            "snippet": result["body"].strip()[:300],  # Limit snippet length
                        }
                        results.append(cleaned_result)

                    if len(results) >= max_results:
                        break

            logger.info(f"Found {len(results)} results for query: {query}")

        except Exception as e:
            logger.error(f"DuckDuckGo search error: {str(e)}")

        return results

    def _build_intelligent_prompt(
        self, query: str, search_results: List[Dict], context: Optional[Dict] = None
    ) -> str:
        """Build an intelligent prompt for summarizing search results."""

        # Build context from search results
        search_context = ""
        for i, result in enumerate(search_results, 1):
            search_context += f"""
SOURCE {i}:
Title: {result.get('title', 'N/A')}
URL: {result.get('url', 'N/A')}
Content: {result.get('snippet', 'N/A')}
"""

        # Enhanced prompt template
        prompt = f"""You are a research assistant that analyzes and synthesizes web search results. 

USER QUERY: {query}

SEARCH RESULTS:
{search_context}

INSTRUCTIONS:
1. Analyze the search results and provide a comprehensive answer to the user's query
2. Synthesize information from multiple sources when relevant
3. Be factual and cite specific sources when making claims
4. If the search results are insufficient or contradictory, acknowledge this
5. Provide a balanced perspective if there are conflicting viewpoints
6. Structure your response clearly with main points first

ADDITIONAL CONTEXT:
{context.get('meeting_context', 'No additional context provided') if context else 'No additional context'}

RESPONSE FORMAT:
- Start with a direct answer to the query
- Provide supporting details and evidence
- Mention source reliability when relevant
- End with a brief summary

ANSWER:"""

        return prompt

    async def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform intelligent web search and synthesis.

        Returns:
            Dict with fields:
            - answer (string): synthesized answer
            - sources (list): formatted source information
            - search_metadata (dict): search performance metrics
        """
        try:
            # Enhance query with context if available
            enhanced_query = self._enhance_query(query, context)

            # Perform search
            search_start = asyncio.get_event_loop().time()
            results = await self._ddg_search_async(enhanced_query, self.max_results)
            search_duration = asyncio.get_event_loop().time() - search_start

            if not results:
                return {
                    "success": True,
                    "content": {
                        "answer": "I searched for information but couldn't find relevant results for your query. Please try rephrasing or check your internet connection.",
                        "sources": [],
                        "search_metadata": {
                            "results_found": 0,
                            "search_time": round(search_duration, 2),
                            "query_used": enhanced_query,
                        },
                    },
                }

            # Generate intelligent summary
            prompt = self._build_intelligent_prompt(query, results, context)
            answer = await self.llm.generate_async(prompt)

            # Format sources for better presentation
            formatted_sources = self._format_sources(results)

            return {
                "success": True,
                "content": {
                    "answer": answer.strip(),
                    "sources": formatted_sources,
                    "search_metadata": {
                        "results_found": len(results),
                        "search_time": round(search_duration, 2),
                        "query_used": enhanced_query,
                        "sources_used": len(formatted_sources),
                    },
                },
            }

        except Exception as e:
            logger.error(f"InternetAgent processing failed: {str(e)}")
            return {
                "success": False,
                "content": f"Search service temporarily unavailable. Please try again later. Error: {str(e)}",
            }

    def _enhance_query(self, query: str, context: Optional[Dict] = None) -> str:
        """Enhance search query with context and specificity."""
        base_query = query

        if context and context.get("meeting_id"):
            # Add meeting context to make search more relevant
            base_query = f"{query} (meeting context)"

        # Add recency preference for time-sensitive topics
        time_sensitive_keywords = ["current", "recent", "latest", "today", "2024", "2025"]
        if any(keyword in query.lower() for keyword in time_sensitive_keywords):
            base_query = f"{base_query} latest information"

        return base_query

    def _format_sources(self, results: List[Dict]) -> List[Dict[str, Any]]:
        """Format search results for clean presentation."""
        formatted_sources = []

        for i, result in enumerate(results, 1):
            source = {
                "source_id": f"web_{i}",
                "title": result.get("title", "Unknown Title"),
                "url": result.get("url", ""),
                "snippet": result.get("snippet", ""),
                "type": "web_search",
                "relevance_score": 1.0 - (i * 0.1),  # Simple relevance scoring
            }
            formatted_sources.append(source)

        return formatted_sources

    async def quick_fact_check(
        self, statement: str, context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Quick fact-checking functionality."""
        fact_check_query = f"fact check: {statement}"
        return await self.process(fact_check_query, context)

    async def get_recent_news(self, topic: str, max_results: int = 3) -> Dict[str, Any]:
        """Get recent news on a specific topic."""
        news_query = f"recent news about {topic} 2024 2025"
        return await self.process(news_query, {"max_results": max_results})
