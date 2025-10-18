from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import asyncio

from PIL import Image
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from .base_agent import BaseAgent, AgentResponse


class ImageEmbedding(BaseModel):
    id: str
    values: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ImageAgent(BaseAgent):
    """Agent for embedding screenshots/images using CLIP ViT-B/32."""

    SUPPORTED_FORMATS = {
        ".png", ".jpg", ".jpeg", ".webp", ".bmp"
    }

    def __init__(self, model_name: str = "sentence-transformers/clip-ViT-B-32"):
        super().__init__(
            name="image_agent",
            description="Embeds screenshots/images using CLIP ViT-B/32"
        )
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self._model_lock = asyncio.Lock()

    async def setup(self) -> None:
        if self._model is not None:
            return
        async with self._model_lock:
            if self._model is not None:
                return
            self._model = await asyncio.to_thread(SentenceTransformer, self.model_name)

    async def process(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        try:
            await self.setup()
            meeting_id = (context or {}).get("meeting_id", "unknown")

            paths: List[Union[str, Path]] = []
            if "file_path" in input_data:
                paths = [input_data["file_path"]]
            elif "file_paths" in input_data:
                paths = list(input_data["file_paths"]) or []
            else:
                return self.failure_response("Missing 'file_path' or 'file_paths' in input data")

            items: List[ImageEmbedding] = []
            for i, p in enumerate(paths):
                path = Path(p).resolve()
                if not path.exists() or path.suffix.lower() not in self.SUPPORTED_FORMATS:
                    return self.failure_response(f"Unsupported or missing file: {path}")
                img = await asyncio.to_thread(Image.open, path)
                img = img.convert("RGB")
                vec = await asyncio.to_thread(self._model.encode, img, normalize_embeddings=True)
                items.append(ImageEmbedding(
                    id=f"{meeting_id}_img_{i}",
                    values=vec.tolist(),
                    metadata={
                        "meeting_id": meeting_id,
                        "chunk_id": f"{meeting_id}_img_{i}",
                        "type": "image",
                        "filename": path.name,
                    }
                ))
            return self.success_response({
                "vectors": [it.dict() for it in items]
            })
        except Exception as e:
            self.logger.error("Image embedding failed", exc_info=True)
            return self.failure_response(f"Image embedding failed: {e}")
