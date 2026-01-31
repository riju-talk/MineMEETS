from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import asyncio

from PIL import Image
from sentence_transformers import SentenceTransformer


class ImageEmbedding:
    def __init__(self, id: str, values: List[float], metadata: Dict[str, Any]):
        self.id = id
        self.values = values
        self.metadata = metadata


class ImageAgent:
    """Agent for embedding screenshots/images using CLIP ViT-B/32."""

    SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    def __init__(self, model_name: str = "sentence-transformers/clip-ViT-B-32"):
        self.model_name = model_name
        self._model = None
        self._model_lock = asyncio.Lock()

    async def setup(self) -> None:
        if self._model is not None:
            return
        async with self._model_lock:
            if self._model is not None:
                return
            self._model = SentenceTransformer(self.model_name)

    async def process(
        self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Embed images using CLIP ViT-B/32.

        Args:
            input_data: Must contain 'file_path' key with path to image file
            context: Optional context with 'meeting_id' and other metadata

        Returns:
            Dict with success, content (list of ImageEmbedding)
        """
        try:
            if not input_data or "file_path" not in input_data:
                return {"success": False, "content": "Missing 'file_path' in input data"}

            file_path = input_data["file_path"]

            # Validate file
            try:
                file_path = self._validate_image_file(file_path)
            except ValueError as e:
                return {"success": False, "content": str(e)}

            # Load model if not already loaded
            try:
                await self.setup()
            except Exception as e:
                return {"success": False, "content": "Failed to initialize embedding model"}

            # Embed image
            try:
                image = Image.open(file_path)
                embeddings = self._model.encode([image], convert_to_numpy=True)
                vector = embeddings[0].tolist()

                # Create embedding object
                embedding = ImageEmbedding(
                    id=file_path.stem,  # Use filename as ID
                    values=vector,
                    metadata={"file_path": str(file_path)},
                )

                return {"success": True, "content": [embedding.__dict__]}

            except Exception as e:
                return {"success": False, "content": f"Embedding failed: {str(e)}"}

        except Exception as e:
            return {"success": False, "content": f"Image processing error: {str(e)}"}

    def _validate_image_file(self, file_path: Union[str, Path]) -> Path:
        """Validate image file exists and is in supported format."""
        try:
            path = Path(file_path).resolve()
            if not path.exists():
                raise ValueError(f"File not found: {file_path}")

            if not path.is_file():
                raise ValueError(f"Not a file: {file_path}")

            if path.suffix.lower() not in self.SUPPORTED_FORMATS:
                raise ValueError(
                    f"Unsupported file format: {path.suffix}. "
                    f"Supported formats: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
                )

            return path

        except (OSError, ValueError) as e:
            raise ValueError(f"Invalid image file: {str(e)}")
