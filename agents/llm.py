# agents/llm.py
import asyncio
import requests
import os
from typing import Optional


class LLM:
    """Simple LLM wrapper for Ollama HTTP API."""

    def __init__(self, model: str = "llama3.1", temperature: float = 0.0, **kwargs):
        self.model = model
        self.temperature = temperature
        self.base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.timeout = kwargs.get("timeout", 300)

    def generate(self, prompt: str) -> str:
        """Synchronous generation via Ollama HTTP API."""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "stream": False,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except Exception as e:
            raise RuntimeError(f"LLM generation failed: {str(e)}")

    async def generate_async(self, prompt: str) -> str:
        """Asynchronous generation."""
        return await asyncio.to_thread(self.generate, prompt)
