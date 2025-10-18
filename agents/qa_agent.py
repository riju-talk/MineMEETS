# agents/qa_agent.py
import asyncio
import logging
from typing import Optional, Dict, Any, List
from agents.pinecone_db import PineconeDB
from agents.llm import LLM

logger = logging.getLogger(__name__)

class QAAgent:
    def __init__(self, pinecone_db: PineconeDB):
        self.pinecone_db = pinecone_db
        self.llm = LLM(model="llama3.1", temperature=0.0)
        
        # More flexible and comprehensive prompt templates
        self.qa_prompt_template = """You are an intelligent meeting assistant analyzing meeting transcripts. Use the provided context to answer the question thoroughly.

CONTEXT FROM MEETING:
{context}

USER'S QUESTION: {question}

GUIDELINES:
- Answer using ONLY the meeting context provided above
- If the context contains relevant information, provide a detailed answer with specific details
- If the context doesn't mention the specific topic, say so clearly but suggest what related information is available
- Be helpful and extract as much value as possible from the available context
- If you can infer or connect information from different parts of the context, do so logically
- Structure your answer to be clear and easy to understand

ANSWER:"""

        self.fallback_prompt_template = """Based on the following meeting context, provide any relevant information that might help answer the question, even if it's not a direct answer:

MEETING CONTEXT:
{context}

QUESTION: {question}

Please provide any potentially relevant information from the meeting, or explain why the meeting doesn't cover this topic:"""

    async def process(self, question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            meeting_id = context.get("meeting_id") if context else None
            namespace = meeting_id if meeting_id else None
            
            # Get more context with multiple query strategies
            hits = await self._smart_query(question, namespace)
            
            if not hits:
                return await self._handle_no_context(question, namespace)

            # Process contexts with better filtering
            contexts, sources = self._process_hits(hits)
            
            if not contexts:
                return await self._handle_insufficient_context(question, namespace)

            context_block = "\n\n---\n\n".join(contexts)
            
            # Use appropriate prompt based on context relevance
            max_relevance = max([s.get('similarity_score', 0) for s in sources]) if sources else 0
            if max_relevance < 0.6:  # Low relevance threshold
                prompt = self.fallback_prompt_template.format(
                    context=context_block, 
                    question=question
                )
            else:
                prompt = self.qa_prompt_template.format(
                    context=context_block, 
                    question=question
                )

            # Generate answer with retry logic
            answer = await self._generate_answer_with_fallback(prompt, question, context_block)

            return {
                "success": True, 
                "content": {
                    "answer": answer.strip(),
                    "sources": self._format_sources(sources),
                    "context_stats": {
                        "sources_used": len(sources),
                        "total_context_length": sum(len(ctx) for ctx in contexts),
                        "average_relevance": round(sum(s.get('similarity_score', 0) for s in sources) / len(sources), 3) if sources else 0,
                        "max_relevance": max_relevance
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"QA processing error: {str(e)}")
            return {
                "success": False, 
                "content": f"I encountered an error while processing your question. Please try again."
            }

    async def _smart_query(self, question: str, namespace: str = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """Use multiple query strategies to find relevant context."""
        all_hits = []
        
        try:
            # Strategy 1: Direct semantic search
            direct_hits = self.pinecone_db.query_text(question, namespace=namespace, top_k=top_k)
            all_hits.extend(direct_hits)
            
            # Strategy 2: Keyword-based search for better recall
            keywords = self._extract_keywords(question)
            if keywords:
                for keyword in keywords[:3]:  # Use top 3 keywords
                    keyword_hits = self.pinecone_db.query_text(keyword, namespace=namespace, top_k=3)
                    all_hits.extend(keyword_hits)
            
            # Strategy 3: Broader context search for general questions
            if self._is_general_question(question):
                general_hits = self.pinecone_db.query_text("meeting discussion topics", namespace=namespace, top_k=5)
                all_hits.extend(general_hits)
            
            # Remove duplicates and sort by relevance
            unique_hits = self._deduplicate_hits(all_hits)
            return sorted(unique_hits, key=lambda x: x.get('score', 0), reverse=True)[:top_k]
            
        except Exception as e:
            logger.error(f"Smart query failed: {str(e)}")
            # Fallback to simple query
            return self.pinecone_db.query_text(question, namespace=namespace, top_k=top_k)

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract potential keywords from the question."""
        # Simple keyword extraction - can be enhanced with NLP
        stop_words = {'what', 'how', 'why', 'when', 'where', 'who', 'which', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'do', 'does', 'did'}
        words = text.lower().split()
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords[:5]  # Return top 5 keywords

    def _is_general_question(self, question: str) -> bool:
        """Check if this is a general question about the meeting."""
        general_indicators = [
            'what was discussed', 'summary', 'overview', 'main points',
            'key topics', 'what happened', 'meeting about', 'agenda'
        ]
        question_lower = question.lower()
        return any(indicator in question_lower for indicator in general_indicators)

    def _deduplicate_hits(self, hits: List[Dict]) -> List[Dict]:
        """Remove duplicate hits based on chunk_id or very similar content."""
        seen_ids = set()
        unique_hits = []
        
        for hit in hits:
            metadata = hit.get('metadata', {})
            chunk_id = metadata.get('chunk_id')
            
            if chunk_id and chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_hits.append(hit)
            elif not chunk_id:  # If no chunk_id, keep it
                unique_hits.append(hit)
                
        return unique_hits

    def _process_hits(self, hits: List[Dict]) -> tuple[List[str], List[Dict]]:
        """Process search hits into context and source objects."""
        contexts = []
        sources = []
        
        for i, hit in enumerate(hits):
            metadata = hit.get('metadata', {}) or {}
            text = metadata.get('text', '').strip()
            score = hit.get('score', 0)
            
            if text and score > 0.3:  # Minimum relevance threshold
                # Format context with relevance indicator
                relevance_indicator = f"[Relevance: {score:.2f}]"
                formatted_context = f"{relevance_indicator}\n{text}"
                contexts.append(formatted_context)
                
                source_info = {
                    "source_id": f"source_{i+1}",
                    "content": text,
                    "full_content": text,
                    "metadata": metadata,
                    "similarity_score": score
                }
                sources.append(source_info)
        
        return contexts, sources

    async def _generate_answer_with_fallback(self, prompt: str, question: str, context: str) -> str:
        """Generate answer with fallback strategies."""
        try:
            answer = await self.llm.generate_async(prompt)
            
            # Check if answer is too generic or indicates no information
            if self._is_generic_answer(answer):
                # Try with more permissive prompt
                permissive_prompt = f"""Given this meeting context, provide ANY information that might be relevant to the question, even loosely:

Context: {context}

Question: {question}

Please extract and share any information from the meeting that could be helpful:"""
                
                answer = await self.llm.generate_async(permissive_prompt)
                
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            return "I'm having trouble processing the meeting content right now. Please try again or rephrase your question."

    def _is_generic_answer(self, answer: str) -> bool:
        """Check if the answer is too generic or indicates no information."""
        generic_phrases = [
            "i don't know", "no information", "not mentioned", "not discussed",
            "the context doesn't", "based on the provided", "limited information",
            "i cannot answer", "unable to provide"
        ]
        answer_lower = answer.lower()
        return any(phrase in answer_lower for phrase in generic_phrases)

    async def _handle_no_context(self, question: str, namespace: str) -> Dict[str, Any]:
        """Handle case where no context is found."""
        # Try a broader search to see if there's any content at all
        try:
            test_hits = self.pinecone_db.query_text("meeting", namespace=namespace, top_k=3)
            if test_hits:
                return {
                    "success": True,
                    "content": {
                        "answer": f"I searched through the meeting content but couldn't find specific information about '{question}'. The meeting appears to cover different topics. Try asking about general themes or specific terms mentioned in the meeting.",
                        "sources": []
                    }
                }
            else:
                return {
                    "success": True,
                    "content": {
                        "answer": "This meeting doesn't appear to have any processed content yet, or the content may be empty. Please check if the meeting was processed correctly.",
                        "sources": []
                    }
                }
        except Exception as e:
            return {
                "success": True,
                "content": {
                    "answer": "I couldn't access the meeting content. The meeting might not be properly processed or there might be a connection issue.",
                    "sources": []
                }
            }

    async def _handle_insufficient_context(self, question: str, namespace: str) -> Dict[str, Any]:
        """Handle case where context exists but relevance is too low."""
        # Get some sample content to understand what the meeting is about
        sample_hits = self.pinecone_db.query_text("", namespace=namespace, top_k=5)
        if sample_hits:
            sample_texts = [hit.get('metadata', {}).get('text', '') for hit in sample_hits[:2]]
            sample_preview = "... ".join([text[:100] + "..." for text in sample_texts if text])
            
            answer = f"I couldn't find specific information about '{question}' in this meeting. "
            answer += f"The meeting appears to discuss topics like: {sample_preview} "
            answer += "Try asking about general themes or rephrasing your question using terms that might appear in the meeting transcript."
        else:
            answer = f"The meeting content doesn't contain specific information about '{question}'. The meeting might cover different topics entirely."

        return {
            "success": True,
            "content": {
                "answer": answer,
                "sources": []
            }
        }

    def _format_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format sources for better presentation."""
        formatted_sources = []
        
        for source in sources:
            meta = source.get("metadata", {})
            content = source.get("content", "")
            
            # Create meaningful title
            title = self._generate_source_title(content, meta)
            
            formatted_source = {
                "title": title,
                "content": source.get("full_content", ""),
                "preview": content[:200] + "..." if len(content) > 200 else content,
                "similarity_score": source.get("similarity_score", 0),
                "metadata": meta
            }
            formatted_sources.append(formatted_source)
        
        # Sort by relevance
        formatted_sources.sort(key=lambda x: x["similarity_score"], reverse=True)
        return formatted_sources

    def _generate_source_title(self, content: str, metadata: Dict[str, Any]) -> str:
        """Generate descriptive source titles."""
        # Use first meaningful sentence
        sentences = content.split('.')
        first_sentence = next((s.strip() for s in sentences if s.strip()), "Meeting Content")
        
        # Clean up
        first_sentence = first_sentence.replace('\n', ' ').strip()
        if len(first_sentence) > 80:
            first_sentence = first_sentence[:77] + "..."
        
        # Add position info if available
        chunk_id = metadata.get("chunk_id", "")
        if "chunk" in chunk_id:
            chunk_num = chunk_id.split("_chunk_")[-1] if "_chunk_" in chunk_id else "?"
            return f"Segment {chunk_num}: {first_sentence}"
        
        return first_sentence

    async def get_meeting_overview(self, meeting_id: str) -> Dict[str, Any]:
        """Generate comprehensive meeting overview."""
        try:
            # Get diverse content samples
            hits = self.pinecone_db.query_text("meeting discussion conversation topics", 
                                             namespace=meeting_id, top_k=20)
            
            if not hits:
                return {
                    "success": True,
                    "content": {
                        "overview": "No meeting content available for analysis.",
                        "topics": [],
                        "stats": {"total_chunks": 0, "content_samples": 0}
                    }
                }

            # Extract and combine content
            all_content = []
            for hit in hits:
                meta = hit.get("metadata", {}) or {}
                text = meta.get("text", "").strip()
                if text:
                    all_content.append(text)

            combined_content = "\n\n".join(all_content[:10])  # Use first 10 for overview
            
            # Generate comprehensive overview
            overview_prompt = f"""Analyze this meeting content and provide a detailed overview:

{combined_content[:3500]}

Please provide a comprehensive summary covering:
1. Main topics and themes discussed
2. Key decisions, action items, or outcomes
3. Important discussion points and highlights
4. Overall purpose and tone of the meeting

Structure your response to be informative and useful:"""

            overview = await self.llm.generate_async(overview_prompt)
            
            return {
                "success": True,
                "content": {
                    "overview": overview.strip(),
                    "topics": await self._extract_topics(combined_content),
                    "stats": {
                        "total_chunks": len(all_content),
                        "content_samples": len(hits),
                        "total_content_length": sum(len(content) for content in all_content)
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Meeting overview failed: {str(e)}")
            return {
                "success": False,
                "content": f"Failed to generate meeting overview: {str(e)}"
            }

    async def _extract_topics(self, content: str) -> List[str]:
        """Extract main topics from meeting content."""
        try:
            topic_prompt = f"""Review this meeting content and identify the main topics discussed. Return a clean list of topics:

{content[:1500]}

List the main topics (max 8):"""

            topics_response = await self.llm.generate_async(topic_prompt)
            
            # Parse topics from response
            topics = []
            for line in topics_response.split('\n'):
                line = line.strip()
                # Remove numbering and bullets
                for prefix in ['-', '*', '•', '1.', '2.', '3.', '4.', '5.']:
                    if line.startswith(prefix):
                        line = line[len(prefix):].strip()
                if line and len(line) > 3 and len(line) < 100:
                    topics.append(line)
            
            return topics[:8]  # Limit to top 8 topics
            
        except Exception as e:
            logger.error(f"Topic extraction failed: {str(e)}")
            # Fallback: extract keywords
            words = content.lower().split()
            common_words = {'the', 'and', 'for', 'with', 'that', 'this', 'have', 'from', 'they', 'were'}
            unique_words = [w for w in words if len(w) > 4 and w not in common_words]
            return list(set(unique_words))[:6]