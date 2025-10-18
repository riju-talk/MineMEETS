# agents/coordinator.py
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone
import os
import asyncio
import PyPDF2
import docx
from agents.audio_agent import AudioAgent
from agents.qa_agent import QAAgent
from agents.internet_agent import InternetAgent
from agents.image_agent import ImageAgent
from agents.pinecone_db import PineconeDB
import logging

logger = logging.getLogger(__name__)

class MeetingCoordinator:
    """
    Enhanced meeting coordinator with improved error handling, 
    progress tracking, and comprehensive meeting management.
    """
    
    def __init__(self):
        self.pinecone_db = PineconeDB()
        self.audio_agent = AudioAgent(model_size="base")
        self.image_agent = ImageAgent()
        self.qa_agent = QAAgent(self.pinecone_db)
        self.internet_agent = InternetAgent()
        self.active_meetings = {}  # {meeting_id: meeting_metadata}
        
        # Configuration
        self.max_file_size_mb = 50
        self.max_chunks_per_meeting = 10000
        self.chunk_size = 1000
        self.chunk_overlap = 200

    def list_meetings(self) -> List[Dict[str, Any]]:
        """Return a comprehensive list of all meetings with metadata."""
        meetings = []
        for meeting in self.active_meetings.values():
            meetings.append({
                "id": meeting["id"],
                "title": meeting.get("title", meeting["id"]),
                "date": meeting.get("date", "Unknown"),
                "participants": meeting.get("participants", []),
                "chunk_count": meeting.get("chunk_count", 0),
                "type": meeting.get("type", "unknown"),
                "status": "processed"
            })
        
        # Sort by date (newest first)
        return sorted(meetings, key=lambda x: x["date"], reverse=True)

    def get_meeting(self, meeting_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve comprehensive meeting information."""
        meeting = self.active_meetings.get(meeting_id)
        if meeting:
            # Add additional computed fields
            meeting["stats"] = {
                "transcript_length": len(meeting.get("transcript", "")),
                "chunk_count": len(meeting.get("chunks", [])),
                "participant_count": len(meeting.get("participants", [])),
                "processing_time": meeting.get("processing_time", "Unknown")
            }
        return meeting

    async def process_meeting(self, meeting_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a meeting with enhanced error handling and progress tracking.
        """
        processing_start = datetime.now(timezone.utc)
        meeting_id = meeting_data["id"]
        
        try:
            # Validate input
            self._validate_meeting_data(meeting_data)
            
            # Initialize meeting metadata
            meeting_meta = self._initialize_meeting_metadata(meeting_data)
            self.active_meetings[meeting_id] = meeting_meta

            logger.info(f"Starting processing for meeting: {meeting_id}")

            # Process based on content type
            processing_result = await self._process_meeting_content(meeting_data, meeting_id)
            
            if not processing_result["success"]:
                raise RuntimeError(processing_result["error"])

            # Update meeting metadata with results
            meeting_meta.update({
                "transcript": processing_result["raw_content"],
                "chunks": processing_result["chunks"],
                "chunk_count": len(processing_result["chunks"]),
                "processing_time": str(datetime.now(timezone.utc) - processing_start),
                "type": meeting_data.get("type", "unknown"),
                "status": "processed"
            })

            logger.info(f"Successfully processed meeting {meeting_id} with {len(processing_result['chunks'])} chunks")

            return {
                "meeting_id": meeting_id,
                "status": "processed",
                "chunks_processed": len(processing_result["chunks"]),
                "processing_time": meeting_meta["processing_time"],
                "content_type": meeting_data.get("type")
            }

        except MemoryError as e:
            error_msg = f"Memory error: Document too large. Try splitting into smaller files."
            logger.error(f"Memory error processing {meeting_id}: {str(e)}")
            return self._create_error_response(meeting_id, error_msg, "memory_error")
            
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            logger.error(f"Error processing meeting {meeting_id}: {str(e)}")
            return self._create_error_response(meeting_id, error_msg, "processing_error")

    def _validate_meeting_data(self, meeting_data: Dict[str, Any]) -> None:
        """Validate meeting data before processing."""
        required_fields = ["id", "type"]
        for field in required_fields:
            if field not in meeting_data:
                raise ValueError(f"Missing required field: {field}")
                
        valid_types = ["transcript", "file", "audio", "image"]
        if meeting_data.get("type") not in valid_types:
            raise ValueError(f"Invalid meeting type. Must be one of: {valid_types}")

    def _initialize_meeting_metadata(self, meeting_data: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize meeting metadata structure."""
        return {
            "id": meeting_data["id"],
            "title": meeting_data.get("title", meeting_data["id"]),
            "date": meeting_data.get("date", datetime.now(timezone.utc).isoformat()),
            "participants": meeting_data.get("participants", []),
            "type": meeting_data.get("type", "unknown"),
            "status": "processing"
        }

    async def _process_meeting_content(self, meeting_data: Dict[str, Any], meeting_id: str) -> Dict[str, Any]:
        """Process meeting content based on type."""
        content_type = meeting_data["type"]
        
        if content_type == "transcript":
            return await self._process_transcript(meeting_data, meeting_id)
        elif content_type == "file":
            return await self._process_file(meeting_data, meeting_id)
        elif content_type == "audio":
            return await self._process_audio(meeting_data, meeting_id)
        elif content_type == "image":
            return await self._process_image(meeting_data, meeting_id)
        else:
            raise ValueError(f"Unsupported content type: {content_type}")

    async def _process_transcript(self, meeting_data: Dict[str, Any], meeting_id: str) -> Dict[str, Any]:
        """Process transcript content."""
        if "content" in meeting_data:
            raw_content = meeting_data["content"]
        else:
            with open(meeting_data["content_path"], "r", encoding="utf-8") as f:
                raw_content = f.read()
                
        chunks = await self._chunk_text_safe(raw_content, meeting_id)
        return {"success": True, "raw_content": raw_content, "chunks": chunks}

    async def _process_file(self, meeting_data: Dict[str, Any], meeting_id: str) -> Dict[str, Any]:
        """Process document files."""
        file_path = meeting_data["content_path"]
        raw_content = self._extract_text_from_file(file_path)
        chunks = await self._chunk_text_safe(raw_content, meeting_id)
        return {"success": True, "raw_content": raw_content, "chunks": chunks}

    async def _process_audio(self, meeting_data: Dict[str, Any], meeting_id: str) -> Dict[str, Any]:
        """Process audio files."""
        audio_resp = await self.audio_agent.process(
            {"file_path": meeting_data["content_path"]}, 
            context={"meeting_id": meeting_id}
        )
        
        if not audio_resp["success"]:
            raise RuntimeError(f"Audio processing failed: {audio_resp['content']}")
        
        raw_content = audio_resp["content"]["text"]
        chunks = audio_resp["content"].get("chunks") or await self._chunk_text_safe(raw_content, meeting_id)
        
        # Normalize audio chunks
        if chunks and isinstance(chunks[0], dict) and "metadata" not in chunks[0]:
            chunks = self._normalize_audio_chunks(chunks, meeting_id)
            
        return {"success": True, "raw_content": raw_content, "chunks": chunks}

    async def _process_image(self, meeting_data: Dict[str, Any], meeting_id: str) -> Dict[str, Any]:
        """Process image files."""
        img_resp = await self.image_agent.process(
            {"file_path": meeting_data["content_path"]}, 
            context={"meeting_id": meeting_id}
        )
        
        if not img_resp["success"]:
            raise RuntimeError(f"Image processing failed: {img_resp['content']}")
            
        vectors = img_resp["content"]
        
        # Process vectors in optimized batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.pinecone_db.upsert_vectors(batch, namespace=meeting_id)
            await asyncio.sleep(0.05)  # Reduced delay for better performance
            
        return {
            "success": True, 
            "raw_content": meeting_data.get("caption", ""), 
            "chunks": []  # Images don't produce text chunks
        }

    def _normalize_audio_chunks(self, segments: List[Dict], meeting_id: str) -> List[Dict]:
        """Normalize audio segments to standard chunk format."""
        normalized_chunks = []
        
        for i, seg in enumerate(segments):
            if len(normalized_chunks) >= self.max_chunks_per_meeting:
                logger.warning(f"Reached chunk limit for meeting {meeting_id}")
                break
                
            text = seg.get("text", "").strip()
            if not text:
                continue
                
            normalized_chunks.append({
                "text": text,
                "metadata": {
                    "meeting_id": meeting_id,
                    "chunk_id": f"{meeting_id}_audio_chunk_{i}",
                    "position": i,
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "length": len(text),
                    "type": "audio_segment"
                }
            })
            
        return normalized_chunks

    def _extract_text_from_file(self, file_path: str) -> str:
        """Enhanced text extraction with better error handling and format support."""
        try:
            self._validate_file(file_path)
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext == ".pdf":
                return self._extract_pdf_text(file_path)
            elif ext == ".docx":
                return self._extract_docx_text(file_path)
            elif ext == ".txt":
                return self._extract_txt_text(file_path)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to extract text from {file_path}: {str(e)}")

    def _validate_file(self, file_path: str) -> None:
        """Validate file before processing."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        if file_size > self.max_file_size_mb:
            raise ValueError(f"File too large: {file_size:.2f}MB. Maximum is {self.max_file_size_mb}MB.")

    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF with improved formatting."""
        text_parts = []
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(reader.pages):
                if page_num >= 1000:  # Safety limit
                    logger.warning(f"PDF page limit reached at page {page_num}")
                    break
                text = page.extract_text()
                if text.strip():
                    text_parts.append(f"Page {page_num + 1}:\n{text}")
        return "\n\n".join(text_parts)

    def _extract_docx_text(self, file_path: str) -> str:
        """Extract text from DOCX files."""
        doc = docx.Document(file_path)
        text_parts = []
        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)
        return "\n".join(text_parts)

    def _extract_txt_text(self, file_path: str) -> str:
        """Extract text from TXT files with encoding detection."""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode text file with supported encodings")

    async def _chunk_text_safe(self, text: str, meeting_id: str, chunk_size: int = None, chunk_overlap: int = None) -> List[Dict]:
        """Enhanced text chunking with intelligent boundary detection."""
        if not text or not text.strip():
            return []
            
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length and len(chunks) < self.max_chunks_per_meeting:
            end = min(text_length, start + chunk_size)
            
            # Intelligent boundary detection
            if end < text_length:
                end = self._find_optimal_boundary(text, start, end)
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "meeting_id": meeting_id,
                        "chunk_id": f"{meeting_id}_chunk_{len(chunks)}",
                        "position": len(chunks),
                        "length": len(chunk_text),
                        "type": "text_chunk"
                    }
                })
            
            # Move to next chunk with overlap
            start = max(start + 1, end - chunk_overlap)
            
        logger.info(f"Created {len(chunks)} chunks for meeting {meeting_id}")
        return chunks

    def _find_optimal_boundary(self, text: str, start: int, current_end: int) -> int:
        """Find optimal chunk boundary near sentence or paragraph end."""
        boundary_priority = ['\n\n', '.\n', '.\r\n', '.\r', '. ', '! ', '? ', '\n', ' ']
        
        for boundary in boundary_priority:
            boundary_pos = text.rfind(boundary, start, current_end)
            if boundary_pos != -1 and boundary_pos > start + (current_end - start) // 2:
                return boundary_pos + len(boundary)
                
        return current_end

    async def ask_question(self, question: str, meeting_id: Optional[str] = None) -> Dict[str, Any]:
        """Enhanced question answering with better context and fallback handling."""
        try:
            context = {"meeting_id": meeting_id} if meeting_id else {}
            
            # Get primary answer from meeting content
            qa_response = await self.qa_agent.process(question, context=context)
            
            if qa_response["success"]:
                answer_data = qa_response["content"]
                
                # Check if we need internet augmentation
                if self._needs_internet_augmentation(answer_data.get("answer", "")):
                    internet_response = await self.internet_agent.process(question, context=context)
                    
                    if internet_response["success"] and internet_response["content"].get("answer"):
                        # Merge responses intelligently
                        merged_response = self._merge_responses(answer_data, internet_response["content"])
                        return merged_response
                
                return answer_data
            else:
                # Fallback to internet search if QA fails
                internet_response = await self.internet_agent.process(question, context=context)
                if internet_response["success"]:
                    return internet_response["content"]
                else:
                    return {"answer": "I couldn't find relevant information to answer your question.", "sources": []}
                    
        except Exception as e:
            logger.error(f"Question answering failed: {str(e)}")
            return {
                "answer": "I encountered an error while processing your question. Please try again.",
                "sources": [],
                "error": str(e)
            }

    def _needs_internet_augmentation(self, answer: str) -> bool:
        """Determine if internet search should augment the answer."""
        if not answer:
            return True
            
        answer_lower = answer.lower()
        uncertainty_indicators = [
            "i don't know", "not sure", "no information", "not mentioned",
            "i cannot", "unable to", "no context", "not provided",
            "the context doesn't", "based on the provided", "limited information"
        ]
        
        return any(indicator in answer_lower for indicator in uncertainty_indicators)

    def _merge_responses(self, meeting_data: Dict, internet_data: Dict) -> Dict[str, Any]:
        """Intelligently merge meeting and internet responses."""
        meeting_answer = meeting_data.get("answer", "")
        internet_answer = internet_data.get("answer", "")
        
        if meeting_answer and internet_answer:
            merged_answer = f"{meeting_answer}\n\nAdditional information from web search:\n{internet_answer}"
        elif meeting_answer:
            merged_answer = meeting_answer
        else:
            merged_answer = internet_answer
            
        # Combine sources
        meeting_sources = meeting_data.get("sources", [])
        internet_sources = internet_data.get("sources", [])
        combined_sources = meeting_sources + internet_sources
        
        return {
            "answer": merged_answer,
            "sources": combined_sources,
            "response_type": "combined",
            "meeting_sources_used": len(meeting_sources),
            "internet_sources_used": len(internet_sources)
        }

    def _create_error_response(self, meeting_id: str, error_message: str, error_type: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            "meeting_id": meeting_id,
            "status": "failed",
            "error": error_message,
            "error_type": error_type,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def get_meeting_overview(self, meeting_id: str) -> Dict[str, Any]:
        """Get comprehensive overview of a meeting."""
        meeting = self.get_meeting(meeting_id)
        if not meeting:
            return {"success": False, "content": "Meeting not found"}
            
        return await self.qa_agent.get_meeting_overview(meeting_id)

    def delete_meeting(self, meeting_id: str) -> bool:
        """Delete a meeting and its associated vectors."""
        try:
            # Remove from active meetings
            if meeting_id in self.active_meetings:
                del self.active_meetings[meeting_id]
            
            # Delete vectors from Pinecone
            self.pinecone_db.delete_vectors(delete_all=True, namespace=meeting_id)
            
            logger.info(f"Deleted meeting: {meeting_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete meeting {meeting_id}: {str(e)}")
            return False

# Module-level coordinator instance
coordinator = MeetingCoordinator()