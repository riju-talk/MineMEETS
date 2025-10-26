# agents/document_processor.py
import os
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

# LangChain imports for document processing
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Enhanced document processor using LangChain loaders and splitters."""

    def __init__(self):
        # Configure text splitters
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
            separators=[
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
        )

        # Markdown splitter for structured content
        self.markdown_splitter = MarkdownTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        # Supported file extensions
        self.supported_extensions = {
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.txt': TextLoader,
            '.md': TextLoader,  # Markdown treated as text
        }

    def load_and_split_document(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Load a document and split it into chunks using LangChain."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            extension = file_path.suffix.lower()

            if extension not in self.supported_extensions:
                raise ValueError(f"Unsupported file type: {extension}")

            # Load document using appropriate loader
            loader_class = self.supported_extensions[extension]
            loader = loader_class(str(file_path))
            documents = loader.load()

            if not documents:
                logger.warning(f"No content found in {file_path}")
                return []

            # Add metadata to all documents
            if metadata:
                for doc in documents:
                    doc.metadata.update(metadata)

            # Split documents into chunks
            if extension == '.md':
                # Use markdown splitter for markdown files
                split_docs = self.markdown_splitter.split_documents(documents)
            else:
                # Use recursive character splitter for other formats
                split_docs = self.text_splitter.split_documents(documents)

            # Add chunk-specific metadata
            for i, chunk in enumerate(split_docs):
                chunk.metadata.update({
                    "chunk_id": f"{file_path.stem}_chunk_{i}",
                    "chunk_index": i,
                    "source_file": str(file_path),
                    "file_extension": extension,
                    "total_chunks": len(split_docs)
                })

            logger.info(f"Processed {file_path} into {len(split_docs)} chunks")
            return split_docs

        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise

    def load_and_split_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Split raw text into chunks using LangChain."""
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided for processing")
                return []

            # Create a document from text
            document = Document(page_content=text, metadata=metadata or {})

            # Split into chunks
            split_docs = self.text_splitter.split_documents([document])

            # Add chunk-specific metadata
            for i, chunk in enumerate(split_docs):
                chunk.metadata.update({
                    "chunk_id": f"text_chunk_{i}",
                    "chunk_index": i,
                    "source_type": "raw_text",
                    "total_chunks": len(split_docs)
                })

            logger.info(f"Split text into {len(split_docs)} chunks")
            return split_docs

        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            raise

    def load_pdf_with_metadata(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Load PDF with enhanced metadata extraction."""
        try:
            file_path = Path(file_path)
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()

            # Extract PDF-specific metadata
            pdf_metadata = {
                "total_pages": len(documents),
                "file_size": file_path.stat().st_size,
                "file_modified": file_path.stat().st_mtime
            }

            # Add metadata to documents
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    "page_number": i + 1,
                    "page_content_length": len(doc.page_content),
                    **pdf_metadata,
                    **(metadata or {})
                })

            return documents

        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            raise

    def create_meeting_chunks(self, transcript: str, meeting_metadata: Dict[str, Any]) -> List[Document]:
        """Create optimized chunks for meeting transcripts."""
        try:
            # Use smaller chunks for meetings to preserve conversational context
            meeting_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,  # Smaller chunks for conversations
                chunk_overlap=150,
                length_function=len,
                separators=[
                    "\n\n",  # Speaker turns
                    ". ",    # Sentences
                    "! ", "? ", "; ",
                    ", ", " ", ""
                ]
            )

            # Create document with meeting metadata
            document = Document(
                page_content=transcript,
                metadata={
                    "meeting_id": meeting_metadata.get("id"),
                    "type": "meeting_transcript",
                    **meeting_metadata
                }
            )

            # Split and enhance metadata
            chunks = meeting_splitter.split_documents([document])

            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_id": f"{meeting_metadata.get('id', 'meeting')}_transcript_chunk_{i}",
                    "chunk_index": i,
                    "chunk_type": "transcript",
                    "position": i,
                    "total_chunks": len(chunks)
                })

            logger.info(f"Created {len(chunks)} meeting transcript chunks")
            return chunks

        except Exception as e:
            logger.error(f"Error creating meeting chunks: {str(e)}")
            raise

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
