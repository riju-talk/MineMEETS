# agents/multimodal_rag.py
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple

from agents.pinecone_db import PineconeDB
from agents.llm import LLM

logger = logging.getLogger(__name__)


class MultimodalRetriever:
    """Production multimodal retriever with hybrid search strategies."""

    def __init__(self, pinecone_db: PineconeDB, content_types: List[str] = None):
        self.pinecone_db = pinecone_db
        self.content_types = content_types or ["text", "audio", "image"]

    def retrieve(self, query: str, meeting_id: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Retrieve documents from multiple modalities using hybrid search."""
        all_results = []

        try:
            # Strategy 1: Direct semantic search
            semantic_results = self.pinecone_db.query_text(query, namespace=meeting_id, top_k=10)
            for result in semantic_results:
                result["metadata"]["search_type"] = "semantic"
                all_results.append(result)

            # Strategy 2: Keyword-based search
            keywords = self._extract_keywords(query)
            for keyword in keywords[:3]:
                keyword_results = self.pinecone_db.query_text(
                    keyword, namespace=meeting_id, top_k=5
                )
                for result in keyword_results:
                    result["metadata"]["search_type"] = "keyword"
                    result["metadata"]["keyword"] = keyword
                    all_results.append(result)

            # Strategy 3: Query expansion for general questions
            if self._is_general_question(query):
                expanded_results = self.pinecone_db.query_text(
                    "meeting discussion topics conversation content", namespace=meeting_id, top_k=8
                )
                for result in expanded_results:
                    result["metadata"]["search_type"] = "expanded"
                    all_results.append(result)

            # Deduplicate and rank
            unique_results = self._deduplicate_results(all_results)
            ranked_results = self._rank_results(unique_results, query)

            return ranked_results[:top_k]

        except Exception as e:
            logger.error(f"Error in multimodal retrieval: {str(e)}")
            return []

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from query."""
        stop_words = {
            "what",
            "how",
            "why",
            "when",
            "where",
            "who",
            "which",
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "do",
            "does",
            "did",
            "can",
            "could",
            "would",
            "should",
            "will",
            "shall",
            "may",
            "might",
            "must",
            "about",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
        }
        words = text.lower().split()
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords[:5]

    def _is_general_question(self, question: str) -> bool:
        """Check if this is a general question."""
        general_indicators = [
            "what was discussed",
            "summary",
            "overview",
            "main points",
            "key topics",
            "what happened",
            "meeting about",
            "agenda",
        ]
        question_lower = question.lower()
        return any(indicator in question_lower for indicator in general_indicators)

    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on ID."""
        seen = set()
        unique = []

        for result in results:
            result_id = result.get("id")
            if result_id not in seen:
                seen.add(result_id)
                unique.append(result)

        return unique

    def _rank_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Rank results by hybrid score."""
        for result in results:
            score = self._calculate_score(result, query)
            result["hybrid_score"] = score

        # Sort by hybrid score
        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return results

    def _calculate_score(self, result: Dict[str, Any], query: str) -> float:
        """Calculate hybrid ranking score."""
        original_score = result.get("score", 0)
        metadata = result.get("metadata", {})

        # Boost by search type
        search_type = metadata.get("search_type", "semantic")
        type_boost = {"semantic": 1.0, "keyword": 0.8, "expanded": 0.6}.get(search_type, 0.5)

        # Boost by content type
        content_type = metadata.get("type", "text")
        content_boost = {"text_chunk": 1.0, "audio_segment": 0.9}.get(content_type, 0.7)

        # Position boost (earlier content may be more important)
        position = metadata.get("position", 0)
        position_boost = max(0.5, 1.0 - (position * 0.01))

        # Keyword overlap boost
        query_keywords = set(self._extract_keywords(query))
        text = result.get("text", "")
        content_keywords = set(self._extract_keywords(text))
        overlap = len(query_keywords.intersection(content_keywords))
        keyword_boost = 1 + (0.1 * overlap)

        return original_score * type_boost * content_boost * position_boost * keyword_boost


class MultimodalRAGChain:
    """Production RAG chain for multimodal content."""

    def __init__(self, pinecone_db: PineconeDB, llm_model: str = "llama3.1"):
        self.pinecone_db = pinecone_db
        self.llm = LLM(model=llm_model, temperature=0.1)
        self.retriever = MultimodalRetriever(pinecone_db)

    async def query(self, question: str, meeting_id: Optional[str] = None) -> Dict[str, Any]:
        """Query the multimodal RAG system."""
        try:
            if not meeting_id:
                return {
                    "success": False,
                    "answer": "No meeting ID provided",
                    "sources": [],
                    "context_stats": {"sources_used": 0, "modalities": []},
                }

            # Retrieve relevant context
            results = self.retriever.retrieve(question, meeting_id, top_k=20)

            if not results:
                return {
                    "success": True,
                    "answer": "I couldn't find relevant information in the meeting content to answer your question.",
                    "sources": [],
                    "context_stats": {"sources_used": 0, "modalities": []},
                }

            # Categorize by modality
            modality_contexts = self._categorize_by_modality(results)

            # Format context
            context_text = self._format_context(modality_contexts)

            # Create prompt
            prompt = self._create_prompt(context_text, question, modality_contexts)

            # Generate answer
            answer = await self.llm.generate_async(prompt)

            # Extract modalities and sources
            modalities = list(set([r.get("metadata", {}).get("type", "text") for r in results]))
            sources = self._format_sources(results)

            return {
                "success": True,
                "answer": answer.strip(),
                "sources": sources,
                "context_stats": {
                    "sources_used": len(results),
                    "modalities": modalities,
                    "modality_breakdown": self._get_modality_breakdown(results),
                    "total_context_length": len(context_text),
                },
            }

        except Exception as e:
            logger.error(f"RAG query failed: {str(e)}")
            return {
                "success": False,
                "answer": f"Error processing question: {str(e)}",
                "sources": [],
                "context_stats": {"sources_used": 0, "modalities": []},
            }

    def _categorize_by_modality(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize results by modality."""
        categories = {"text": [], "audio": [], "image": [], "other": []}

        for result in results:
            doc_type = result.get("metadata", {}).get("type", "text")
            if doc_type == "audio_segment":
                categories["audio"].append(result)
            elif doc_type in ["text_chunk", "transcript"]:
                categories["text"].append(result)
            elif "image" in doc_type:
                categories["image"].append(result)
            else:
                categories["other"].append(result)

        return categories

    def _format_context(self, modality_contexts: Dict[str, List[Dict[str, Any]]]) -> str:
        """Format context with modality indicators."""
        sections = []

        # Text content
        if modality_contexts["text"]:
            text_items = [
                f"[Text - Position {r.get('metadata', {}).get('position', 'N/A')}]\n{r.get('text', '')}"
                for r in modality_contexts["text"][:8]
            ]
            sections.append(f"TEXT TRANSCRIPTS:\n" + "\n\n".join(text_items))

        # Audio content
        if modality_contexts["audio"]:
            audio_items = [
                f"[Audio - {r.get('metadata', {}).get('start', '0')}s to {r.get('metadata', {}).get('end', '0')}s]\n{r.get('text', '')}"
                for r in modality_contexts["audio"][:5]
            ]
            sections.append(f"AUDIO TRANSCRIPTION:\n" + "\n\n".join(audio_items))

        # Image content
        if modality_contexts["image"]:
            image_items = [
                f"[Image - {r.get('metadata', {}).get('file_path', 'Unknown')}]\n{r.get('text', '')}"
                for r in modality_contexts["image"]
            ]
            sections.append(f"IMAGE DESCRIPTIONS:\n" + "\n\n".join(image_items))

        return "\n\n=== MODALITY SEPARATOR ===\n\n".join(sections)

    def _create_prompt(
        self, context: str, question: str, modality_contexts: Dict[str, List[Dict]]
    ) -> str:
        """Create prompt for LLM."""
        modality_info = {k: f"{len(v)} items" for k, v in modality_contexts.items() if v}

        return f"""You are an intelligent meeting assistant analyzing multimodal content.

AVAILABLE MODALITIES: {', '.join(modality_info.keys())}
{', '.join([f"{k}: {v}" for k, v in modality_info.items()])}

MEETING CONTEXT:
{context}

USER'S QUESTION: {question}

INSTRUCTIONS:
- Answer using ONLY the meeting context provided above
- Synthesize information from all relevant modalities
- If context contains relevant info, provide a detailed answer
- If not, clearly state what information is missing
- Be specific about which modality provided each piece of information

ANSWER:"""

    def _get_modality_breakdown(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get breakdown by modality."""
        breakdown = {}
        for result in results:
            modality = result.get("metadata", {}).get("type", "text")
            breakdown[modality] = breakdown.get(modality, 0) + 1
        return breakdown

    def _format_sources(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format sources with metadata."""
        sources = []

        for i, result in enumerate(results):
            metadata = result.get("metadata", {})
            content_type = metadata.get("type", "text")
            search_type = metadata.get("search_type", "semantic")
            text = result.get("text", "")

            # Create title
            if content_type == "audio_segment":
                start = metadata.get("start", "0")
                end = metadata.get("end", "0")
                title = f"Audio ({start}s-{end}s) - {search_type}"
            elif content_type == "text_chunk":
                position = metadata.get("position", i)
                title = f"Text Chunk {position} - {search_type}"
            else:
                title = f"Content {i+1} - {search_type}"

            sources.append(
                {
                    "title": title,
                    "content": text,
                    "preview": text[:200] + "..." if len(text) > 200 else text,
                    "content_type": content_type,
                    "search_type": search_type,
                    "hybrid_score": result.get("hybrid_score", 0),
                    "metadata": metadata,
                }
            )

        # Sort by score
        sources.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return sources

    async def get_meeting_summary(self, meeting_id: str) -> Dict[str, Any]:
        """Generate meeting summary."""
        try:
            results = self.retriever.retrieve(
                "meeting summary overview key points discussion topics decisions",
                meeting_id,
                top_k=20,
            )

            if not results:
                return {
                    "success": True,
                    "summary": "No meeting content available.",
                    "modalities": [],
                    "stats": {"total_chunks": 0},
                }

            # Create context
            context = "\n\n---\n\n".join(
                [
                    f"[{r.get('metadata', {}).get('type', 'content')}]\n{r.get('text', '')}"
                    for r in results
                ]
            )

            # Generate summary
            prompt = f"""Analyze this meeting content and provide a comprehensive summary:

{context}

Provide a detailed summary covering:
1. Main topics and themes discussed
2. Key decisions and action items
3. Important discussion points
4. Overall meeting purpose and effectiveness

Structure your response with clear sections:"""

            summary = await self.llm.generate_async(prompt)

            modalities = list(set([r.get("metadata", {}).get("type", "text") for r in results]))

            return {
                "success": True,
                "summary": summary.strip(),
                "modalities": modalities,
                "stats": {
                    "total_chunks": len(results),
                    "context_length": len(context),
                    "modalities_count": len(modalities),
                },
            }

        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}")
            return {
                "success": False,
                "summary": f"Failed to generate summary: {str(e)}",
                "modalities": [],
                "stats": {"total_chunks": 0},
            }
