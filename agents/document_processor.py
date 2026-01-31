# agents/document_processor.py
import os
import logging
import re
from typing import Dict, Any, List, Optional
from pathlib import Path
import PyPDF2
import docx

logger = logging.getLogger(__name__)


class DocumentChunk:
    """Represents a document chunk with text and metadata."""
    
    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.page_content = text  # Keep compatible with old code
        self.text = text
        self.metadata = metadata


class DocumentProcessor:
    """Production document processor with deterministic chunking."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Separators in order of preference
        self.separators = [
            "\n\n",  # Double newlines (paragraphs)
            "\n",    # Single newlines
            ". ",    # Sentences
            "! ",    # Exclamation sentences
            "? ",    # Question sentences
            "; ",    # Semi-colon sentences
            ": ",    # Colon sentences
            ", ",    # Commas
            " ",     # Words
            "",      # Characters
        ]

        # Supported file extensions
        self.supported_extensions = {
            '.pdf': self._load_pdf,
            '.docx': self._load_docx,
            '.txt': self._load_text,
            '.md': self._load_text,
        }

    def load_and_split_document(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """Load a document and split it into chunks."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            extension = file_path.suffix.lower()

            if extension not in self.supported_extensions:
                raise ValueError(f"Unsupported file type: {extension}")

            # Load document using appropriate loader
            loader_func = self.supported_extensions[extension]
            text = loader_func(str(file_path))

            if not text:
                logger.warning(f"No content found in {file_path}")
                return []

            # Split into chunks
            chunks = self._split_text(text)
            
            # Create document chunks with metadata
            doc_chunks = []
            base_metadata = metadata or {}
            for i, chunk_text in enumerate(chunks):
                chunk_metadata = {
                    **base_metadata,
                    "chunk_id": f"{file_path.stem}_chunk_{i}",
                    "chunk_index": i,
                    "source_file": str(file_path),
                    "file_extension": extension,
                    "total_chunks": len(chunks)
                }
                doc_chunks.append(DocumentChunk(chunk_text, chunk_metadata))

            logger.info(f"Processed {file_path} into {len(doc_chunks)} chunks")
            return doc_chunks

        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise

    def load_and_split_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """Split raw text into chunks."""
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided for processing")
                return []

            # Split into chunks
            chunks = self._split_text(text)
            
            # Create document chunks with metadata
            doc_chunks = []
            base_metadata = metadata or {}
            for i, chunk_text in enumerate(chunks):
                chunk_metadata = {
                    **base_metadata,
                    "chunk_id": f"text_chunk_{i}",
                    "chunk_index": i,
                    "source_type": "raw_text",
                    "total_chunks": len(chunks)
                }
                doc_chunks.append(DocumentChunk(chunk_text, chunk_metadata))

            logger.info(f"Split text into {len(doc_chunks)} chunks")
            return doc_chunks

        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            raise

    def create_meeting_chunks(self, transcript: str, meeting_metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Create optimized chunks for meeting transcripts."""
        try:
            # Split with meeting-specific settings
            chunk_size_backup = self.chunk_size
            overlap_backup = self.chunk_overlap
            
            self.chunk_size = 800  # Smaller chunks for conversations
            self.chunk_overlap = 150
            
            chunks = self._split_text(transcript)
            
            # Restore original settings
            self.chunk_size = chunk_size_backup
            self.chunk_overlap = overlap_backup
            
            # Create document chunks with meeting metadata
            doc_chunks = []
            for i, chunk_text in enumerate(chunks):
                chunk_metadata = {
                    "chunk_id": f"{meeting_metadata.get('id', 'meeting')}_transcript_chunk_{i}",
                    "chunk_index": i,
                    "chunk_type": "transcript",
                    "meeting_id": meeting_metadata.get("id"),
                    "type": "meeting_transcript",
                    "position": i,
                    "total_chunks": len(chunks),
                    **meeting_metadata
                }
                doc_chunks.append(DocumentChunk(chunk_text, chunk_metadata))

            logger.info(f"Created {len(doc_chunks)} meeting transcript chunks")
            return doc_chunks

        except Exception as e:
            logger.error(f"Error creating meeting chunks: {str(e)}")
            raise

    def _load_pdf(self, file_path: str) -> str:
        """Load text from PDF file."""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = []
                for page in reader.pages:
                    text.append(page.extract_text())
                return "\n\n".join(text)
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            raise

    def _load_docx(self, file_path: str) -> str:
        """Load text from DOCX file."""
        try:
            doc = docx.Document(file_path)
            return "\n\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            logger.error(f"Error loading DOCX {file_path}: {str(e)}")
            raise

    def _load_text(self, file_path: str) -> str:
        """Load text from plain text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {str(e)}")
            raise

    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks using recursive character splitting."""
        if not text:
            return []
        
        chunks = []
        self._recursive_split(text, chunks)
        return [c for c in chunks if c.strip()]  # Filter empty chunks

    def _recursive_split(self, text: str, chunks: List[str], separator_index: int = 0) -> None:
        """Recursively split text using separators in order of preference."""
        if not text or not text.strip():
            return

        # If text is already smaller than chunk size, add it
        if len(text) <= self.chunk_size:
            chunks.append(text.strip())
            return

        # Try current separator
        if separator_index >= len(self.separators):
            # Reached end of separators, force split at chunk_size
            chunks.append(text[:self.chunk_size].strip())
            if len(text) > self.chunk_size:
                self._recursive_split(text[self.chunk_size - self.chunk_overlap:], chunks, 0)
            return

        separator = self.separators[separator_index]
        
        # Split by separator
        if separator:
            splits = text.split(separator)
        else:
            # Character-level split
            splits = [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        # If we got only one split, try next separator
        if len(splits) == 1:
            self._recursive_split(text, chunks, separator_index + 1)
            return

        # Group splits into chunks
        current_chunk = []
        current_size = 0

        for split in splits:
            split_size = len(split) + len(separator)
            
            if current_size + split_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = separator.join(current_chunk)
                if chunk_text.strip():
                    chunks.append(chunk_text.strip())
                
                # Start new chunk with overlap
                if len(chunk_text) > self.chunk_overlap:
                    overlap_text = chunk_text[-self.chunk_overlap:]
                    current_chunk = [overlap_text] if overlap_text.strip() else []
                    current_size = len(overlap_text)
                else:
                    current_chunk = []
                    current_size = 0

            current_chunk.append(split)
            current_size += split_size

        # Add remaining chunk
        if current_chunk:
            chunk_text = separator.join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)

    def split_text_simple(self, text: str, meeting_id: str = "", chunk_id_prefix: str = "") -> List[Dict[str, Any]]:
        """Simple text splitting for backward compatibility."""
        chunks = self._split_text(text)
        
        result = []
        for i, chunk_text in enumerate(chunks):
            result.append({
                "text": chunk_text,
                "metadata": {
                    "chunk_id": f"{chunk_id_prefix}_{i}" if chunk_id_prefix else f"chunk_{i}",
                    "chunk_index": i,
                    "meeting_id": meeting_id,
                    "type": "text",
                    "total_chunks": len(chunks)
                }
            })
        
        return result

    def validate_document(self, file_path: str) -> Dict[str, Any]:
        """Validate document and return metadata."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return {"valid": False, "error": "File not found"}

            if not file_path.is_file():
                return {"valid": False, "error": "Path is not a file"}

            extension = file_path.suffix.lower()
            if extension not in self.supported_extensions:
                return {
                    "valid": False,
                    "error": f"Unsupported format: {extension}",
                    "supported_formats": list(self.supported_extensions.keys())
                }

            # Get file stats
            stat = file_path.stat()
            file_size_mb = stat.st_size / (1024 * 1024)

            return {
                "valid": True,
                "file_path": str(file_path),
                "extension": extension,
                "file_size_mb": round(file_size_mb, 2),
                "modified_time": stat.st_mtime,
                "can_process": file_size_mb <= 50  # 50MB limit
            }

        except Exception as e:
            return {"valid": False, "error": str(e)}
