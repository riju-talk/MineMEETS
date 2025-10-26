# agents/multimodal_rag.py
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from agents.pinecone_db import PineconeDB
from agents.llm import LLM

logger = logging.getLogger(__name__)

class MultimodalRetriever(BaseRetriever):
    """Custom retriever that can handle multimodal queries across different content types with hybrid search."""

    def __init__(self, pinecone_db: PineconeDB, content_types: List[str] = None):
        self.pinecone_db = pinecone_db
        self.content_types = content_types or ["text", "audio", "image"]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs
    ) -> List[Document]:
        """Retrieve documents from multiple modalities using hybrid search."""
        all_docs = []
        meeting_id = kwargs.get("meeting_id")

        try:
            # Hybrid search strategy 1: Direct semantic search
            if "text" in self.content_types:
                semantic_docs = self.pinecone_db.similarity_search(
                    query,
                    namespace=meeting_id,
                    k=10
                )
                # Add search type metadata
                for doc in semantic_docs:
                    doc.metadata["search_type"] = "semantic"
                    doc.metadata["original_score"] = doc.metadata.get("score", 0)
                all_docs.extend(semantic_docs)

            # Hybrid search strategy 2: Keyword-based search for better recall
            if "text" in self.content_types or "audio" in self.content_types:
                keywords = self._extract_keywords(query)
                for keyword in keywords[:3]:  # Use top 3 keywords
                    keyword_docs = self.pinecone_db.similarity_search(
                        keyword,
                        namespace=meeting_id,
                        k=5
                    )
                    # Add search type metadata
                    for doc in keyword_docs:
                        doc.metadata["search_type"] = "keyword"
                        doc.metadata["keyword"] = keyword
                        doc.metadata["original_score"] = doc.metadata.get("score", 0)
                    all_docs.extend(keyword_docs)

            # Hybrid search strategy 3: Query expansion for general questions
            if self._is_general_question(query):
                expanded_docs = self.pinecone_db.similarity_search(
                    "meeting discussion topics conversation content",
                    namespace=meeting_id,
                    k=8
                )
                # Add search type metadata
                for doc in expanded_docs:
                    doc.metadata["search_type"] = "expanded"
                    doc.metadata["original_score"] = doc.metadata.get("score", 0)
                all_docs.extend(expanded_docs)

            # Filter for audio content if requested
            if "audio" in self.content_types:
                audio_docs = [doc for doc in all_docs if doc.metadata.get("type") == "audio_segment"]
                # Keep audio docs in results
            else:
                # Remove audio docs if not requested
                all_docs = [doc for doc in all_docs if doc.metadata.get("type") != "audio_segment"]

            # Remove duplicates and sort by relevance
            unique_docs = self._deduplicate_documents(all_docs)
            return self._rank_documents(unique_docs, query)[:20]  # Return top 20 most relevant documents

        except Exception as e:
            logger.error(f"Error in multimodal hybrid retrieval: {str(e)}")
            return []

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract potential keywords from the query."""
        stop_words = {'what', 'how', 'why', 'when', 'where', 'who', 'which', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'do', 'does', 'did', 'can', 'could', 'would', 'should', 'will', 'shall', 'may', 'might', 'must', 'can', 'about', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'down', 'out', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'say', 'says', 'said', 'shall', 'should'}
        words = text.lower().split()
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords[:5]  # Return top 5 keywords

    def _is_general_question(self, question: str) -> bool:
        """Check if this is a general question about the meeting."""
        general_indicators = [
            'what was discussed', 'summary', 'overview', 'main points',
            'key topics', 'what happened', 'meeting about', 'agenda',
            'what did they talk about', 'what were the main topics',
            'give me a summary', 'what are the key points'
        ]
        question_lower = question.lower()
        return any(indicator in question_lower for indicator in general_indicators)

    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """Remove duplicate documents based on content similarity."""
        seen = set()
        unique_docs = []

        for doc in documents:
            # Create a hash based on content and metadata
            content_hash = hash(doc.page_content[:100])  # First 100 chars
            metadata_hash = hash(str(sorted(doc.metadata.items())))

            doc_key = (content_hash, metadata_hash)

            if doc_key not in seen:
                seen.add(doc_key)
                unique_docs.append(doc)

        return unique_docs

    def _rank_documents(self, documents: List[Document], query: str) -> List[Document]:
        """Rank documents by relevance using multiple factors."""
        scored_docs = []

        for doc in documents:
            score = self._calculate_document_score(doc, query)
            doc.metadata["hybrid_score"] = score
            scored_docs.append((doc, score))

        # Sort by hybrid score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs]

    def _calculate_document_score(self, doc: Document, query: str) -> float:
        """Calculate a hybrid score for document ranking."""
        original_score = doc.metadata.get("original_score", 0)

        # Boost score based on search type
        search_type = doc.metadata.get("search_type", "semantic")
        if search_type == "semantic":
            type_boost = 1.0
        elif search_type == "keyword":
            type_boost = 0.8
        else:  # expanded
            type_boost = 0.6

        # Boost score based on content type
        content_type = doc.metadata.get("type", "text")
        if content_type == "text_chunk":
            content_boost = 1.0
        elif content_type == "audio_segment":
            content_boost = 0.9
        else:
            content_boost = 0.7

        # Boost score based on position (earlier chunks might be more important)
        position = doc.metadata.get("position", 0)
        position_boost = max(0.5, 1.0 - (position * 0.01))  # Diminish by 1% per position

        # Combine scores
        hybrid_score = original_score * type_boost * content_boost * position_boost

        # Boost if query keywords are found in content
        query_keywords = set(self._extract_keywords(query))
        content_keywords = set(self._extract_keywords(doc.page_content))
        keyword_overlap = len(query_keywords.intersection(content_keywords))
        if keyword_overlap > 0:
            hybrid_score *= (1 + 0.1 * keyword_overlap)

        return hybrid_score

class MultimodalRAGChain:
    """Enhanced RAG chain that handles multimodal content using LangChain."""

    def __init__(self, pinecone_db: PineconeDB, llm_model: str = "llama3.1"):
        self.pinecone_db = pinecone_db
        self.llm = LLM(model=llm_model, temperature=0.1)

        # Create multimodal retriever
        self.retriever = MultimodalRetriever(pinecone_db)

        # Enhanced prompt template for multimodal content
        self.rag_prompt_template = """You are an intelligent meeting assistant analyzing multimodal meeting content including text transcripts, audio transcriptions, and visual information.

CONTEXT FROM MEETING:
{context}

USER'S QUESTION: {question}

ANALYSIS GUIDELINES:
- Synthesize information from all available modalities (text, audio, images)
- If the context contains relevant information from multiple sources, integrate them logically
- Consider temporal aspects: audio segments have timestamps, text chunks have positions
- For visual content, describe what's shown and how it relates to the discussion
- If information appears in multiple modalities, highlight the consistency or discrepancies
- Be comprehensive but concise in your answer
- If the context doesn't contain relevant information, clearly state what's missing
- Structure your answer to be easily parsable: use sections if appropriate

ANSWER:"""

        self.prompt = PromptTemplate(
            template=self.rag_prompt_template,
            input_variables=["context", "question"]
        )

        # Create the RAG chain
        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm.generate_async
            | StrOutputParser()
        )

    async def query(self, question: str, meeting_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Query the multimodal RAG system with enhanced context handling."""
        try:
            # Add meeting_id to kwargs for the retriever
            if meeting_id:
                kwargs["meeting_id"] = meeting_id

            # Get context documents using hybrid search
            context_docs = self.retriever._get_relevant_documents(question, **kwargs)

            if not context_docs:
                return {
                    "success": True,
                    "answer": "I couldn't find relevant information in the meeting content to answer your question. The meeting might not have been processed yet or may not contain information about this topic.",
                    "sources": [],
                    "context_stats": {"sources_used": 0, "modalities": []}
                }

            # Analyze and categorize context by modality
            modality_contexts = self._categorize_context_by_modality(context_docs)

            # Extract context text with modality indicators
            context_text = self._format_multimodal_context(modality_contexts, question)

            # Format enhanced prompt for multimodal understanding
            enhanced_prompt = self._create_enhanced_prompt(context_text, question, modality_contexts)

            answer = await self.llm.generate_async(enhanced_prompt)

            # Analyze sources and modalities
            modalities = list(set([doc.metadata.get("type", "text") for doc in context_docs]))
            sources = self._format_sources(context_docs)

            return {
                "success": True,
                "answer": answer.strip(),
                "sources": sources,
                "context_stats": {
                    "sources_used": len(context_docs),
                    "modalities": modalities,
                    "modality_breakdown": self._get_modality_breakdown(context_docs),
                    "total_context_length": len(context_text),
                    "search_strategies_used": self._get_search_strategies_used(context_docs)
                }
            }

        except Exception as e:
            logger.error(f"Multimodal RAG query failed: {str(e)}")
            return {
                "success": False,
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "context_stats": {"sources_used": 0, "modalities": []}
            }

    def _categorize_context_by_modality(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """Categorize documents by their modality/type."""
        categories = {
            "text": [],
            "audio": [],
            "image": [],
            "other": []
        }

        for doc in documents:
            doc_type = doc.metadata.get("type", "text")
            if doc_type == "audio_segment":
                categories["audio"].append(doc)
            elif doc_type in ["text_chunk", "transcript"]:
                categories["text"].append(doc)
            elif doc_type.startswith("image"):
                categories["image"].append(doc)
            else:
                categories["other"].append(doc)

        return categories

    def _format_multimodal_context(self, modality_contexts: Dict[str, List[Document]], query: str) -> str:
        """Format context with clear modality indicators."""
        formatted_sections = []

        # Format text content
        if modality_contexts["text"]:
            text_content = "\n\n".join([
                f"[Text Content - Position {doc.metadata.get('position', 'N/A')}]\n{doc.page_content}"
                for doc in modality_contexts["text"][:8]  # Limit to top 8 text chunks
            ])
            formatted_sections.append(f"TEXT TRANSCRIPTS:\n{text_content}")

        # Format audio content
        if modality_contexts["audio"]:
            audio_content = "\n\n".join([
                f"[Audio Content - {doc.metadata.get('start', '0')}s to {doc.metadata.get('end', '0')}s]\n{doc.page_content}"
                for doc in modality_contexts["audio"][:5]  # Limit to top 5 audio segments
            ])
            formatted_sections.append(f"AUDIO TRANSCRIPTION:\n{audio_content}")

        # Format image content (if any)
        if modality_contexts["image"]:
            image_content = "\n\n".join([
                f"[Image Content - {doc.metadata.get('file_path', 'Unknown')}]\n{doc.page_content}"
                for doc in modality_contexts["image"]
            ])
            formatted_sections.append(f"IMAGE DESCRIPTIONS:\n{image_content}")

        # Format other content
        if modality_contexts["other"]:
            other_content = "\n\n".join([
                f"[Other Content - {doc.metadata.get('type', 'unknown')}]\n{doc.page_content}"
                for doc in modality_contexts["other"]
            ])
            formatted_sections.append(f"OTHER CONTENT:\n{other_content}")

        return "\n\n=== MODALITY SEPARATOR ===\n\n".join(formatted_sections)

    def _create_enhanced_prompt(self, context: str, question: str, modality_contexts: Dict[str, List[Document]]) -> str:
        """Create an enhanced prompt that understands multimodal context."""
        modality_info = self._get_modality_info(modality_contexts)

        enhanced_template = f"""You are an intelligent multimodal meeting assistant that can analyze and synthesize information from text transcripts, audio recordings, and visual content.

MULTIMODAL CONTEXT INFORMATION:
- Available modalities: {', '.join(modality_info.keys())}
- Text content: {modality_info.get('text', 'None')} chunks
- Audio content: {modality_info.get('audio', 'None')} segments
- Image content: {modality_info.get('image', 'None')} descriptions
- Other content: {modality_info.get('other', 'None')} items

MEETING CONTEXT BY MODALITY:
{{context}}

USER'S QUESTION: {{question}}

ANALYSIS INSTRUCTIONS:
1. **Cross-Modal Analysis**: Look for patterns, consistency, or discrepancies across different modalities
2. **Temporal Understanding**: Consider the timeline - audio segments have timestamps, text chunks have positions
3. **Contextual Relevance**: Prioritize information that directly addresses the question
4. **Multimodal Insights**: If the same information appears in multiple modalities, note the reinforcement
5. **Visual Context**: If images are referenced, consider how they support or illustrate the discussion
6. **Audio Nuances**: Pay attention to audio segments that might contain tone, emphasis, or speaker emotions

RESPONSE GUIDELINES:
- Synthesize information from all relevant modalities
- Be specific about which modality provided each piece of information
- If information conflicts between modalities, note the discrepancy
- Structure your answer for clarity: use sections if multiple modalities contribute
- If no relevant information is found, clearly state what's available in the meeting
- Provide comprehensive but concise answers

ANSWER:"""

        return enhanced_template.format(context=context, question=question)

    def _get_modality_info(self, modality_contexts: Dict[str, List[Document]]) -> Dict[str, str]:
        """Get summary information about available modalities."""
        info = {}
        for modality, docs in modality_contexts.items():
            if docs:
                info[modality] = f"{len(docs)} {'chunks' if modality != 'audio' else 'segments'}"
        return info

    def _get_modality_breakdown(self, documents: List[Document]) -> Dict[str, int]:
        """Get breakdown of documents by modality."""
        breakdown = {}
        for doc in documents:
            modality = doc.metadata.get("type", "text")
            breakdown[modality] = breakdown.get(modality, 0) + 1
        return breakdown

    def _get_search_strategies_used(self, documents: List[Document]) -> List[str]:
        """Get list of search strategies that found results."""
        strategies = set()
        for doc in documents:
            strategy = doc.metadata.get("search_type", "unknown")
            strategies.add(strategy)
        return list(strategies)

    def _format_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Format retrieved documents as sources with enhanced metadata."""
        formatted_sources = []

        for i, doc in enumerate(documents):
            metadata = doc.metadata
            content_type = metadata.get("type", "text")

            # Create descriptive title based on content type and search strategy
            search_type = metadata.get("search_type", "semantic")
            if content_type == "audio_segment":
                start_time = metadata.get("start", "0")
                end_time = metadata.get("end", "0")
                title = f"Audio ({start_time}s-{end_time}s) - {search_type} search"
            elif content_type == "text_chunk":
                position = metadata.get("position", i)
                title = f"Text Chunk {position} - {search_type} search"
            else:
                title = f"Content Segment {i+1} - {search_type} search"

            formatted_source = {
                "title": title,
                "content": doc.page_content,
                "preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "content_type": content_type,
                "search_type": search_type,
                "hybrid_score": metadata.get("hybrid_score", 0),
                "metadata": metadata
            }
            formatted_sources.append(formatted_source)

        # Sort by hybrid score
        formatted_sources.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return formatted_sources

    async def get_meeting_summary(self, meeting_id: str) -> Dict[str, Any]:
        """Generate a comprehensive multimodal summary of a meeting."""
        try:
            # Query for overview content across all modalities
            context_docs = self.retriever._get_relevant_documents(
                "meeting summary overview key points discussion topics decisions action items",
                meeting_id=meeting_id
            )

            if not context_docs:
                return {
                    "success": True,
                    "summary": "No meeting content available for summarization.",
                    "modalities": [],
                    "stats": {"total_chunks": 0}
                }

            # Create comprehensive context
            context_text = "\n\n---\n\n".join([
                f"[{doc.metadata.get('type', 'content')} - {doc.metadata.get('chunk_id', 'N/A')}]\n{doc.page_content}"
                for doc in context_docs[:20]  # Limit to prevent token overflow
            ])

            # Generate multimodal summary
            summary_prompt = f"""Analyze this multimodal meeting content and provide a comprehensive summary:

{context_text}

Please provide a detailed summary covering:
1. Main topics and themes discussed across all modalities
2. Key decisions, action items, and outcomes
3. Important discussion points from text, audio, and visual content
4. Overall meeting purpose, tone, and effectiveness
5. Any notable patterns or insights from the multimodal data

Structure your response with clear sections:"""

            summary = await self.llm.generate_async(summary_prompt)

            # Analyze modalities used
            modalities = list(set([doc.metadata.get("type", "text") for doc in context_docs]))

            return {
                "success": True,
                "summary": summary.strip(),
                "modalities": modalities,
                "stats": {
                    "total_chunks": len(context_docs),
                    "context_length": len(context_text),
                    "modalities_count": len(modalities)
                }
            }

        except Exception as e:
            logger.error(f"Meeting summary failed: {str(e)}")
            return {
                "success": False,
                "summary": f"Failed to generate meeting summary: {str(e)}",
                "modalities": [],
                "stats": {"total_chunks": 0}
            }
