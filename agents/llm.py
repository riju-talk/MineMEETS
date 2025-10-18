# agents/llm.py
import asyncio
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

class LLM:
    """Simple LLM wrapper using ChatOllama."""
    def __init__(self, model: str = "llama3.1", temperature: float = 0.0, **kwargs):
        self.model = model
        self.temperature = temperature
        self.llm = ChatOllama(model=model, temperature=temperature, **kwargs)

    def generate(self, prompt: str) -> str:
        """Synchronous generation."""
        message = HumanMessage(content=prompt)
        response = self.llm.invoke([message])
        return response.content.strip()

    async def generate_async(self, prompt: str) -> str:
        """Asynchronous generation."""
        return await asyncio.to_thread(self.generate, prompt)
