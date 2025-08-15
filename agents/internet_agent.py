# agents/internet_agent.py
from typing import Dict, Any, Optional, List
from .base_agent import BaseAgent, AgentResponse
import os
import asyncio
import logging
from duckduckgo_search import ddg  # pip install duckduckgo_search
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

class InternetAgent(BaseAgent):
    """Internet agent using DuckDuckGo for search and a local HF LLM for summarization."""

    def __init__(self):
        super().__init__(name="internet_agent", description="Performs web searches using DuckDuckGo and summarizes results with a local LLM.")
        # model config
        model_id = os.getenv("LLM_MODEL_ID", "unsloth/Meta-Llama-3-8B-Instruct-bnb-4bit")
        max_new_tokens = int(os.getenv("LLM_MAX_NEW_TOKENS", "256"))
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Load tokenizer & model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=bnb_cfg,
            torch_dtype=torch.bfloat16
        )
        self.gen_pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False,
            return_full_text=False
        )

    def _run_gen_sync(self, prompt: str) -> str:
        res = self.gen_pipe(prompt)
        return res[0].get("generated_text", "") if res else ""

    async def _run_gen(self, prompt: str) -> str:
        return await asyncio.to_thread(self._run_gen_sync, prompt)

    async def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Perform a web search with DuckDuckGo and summarize top results.

        Returns:
          AgentResponse with fields: answer (string), sources (list of dicts with title/url/snippet)
        """
        try:
            # Build a search query; optionally include meeting context for more targeted search
            if context and "meeting_context" in context:
                meeting_meta = context["meeting_context"]
                query = f"{query} (Meeting context: {meeting_meta})"

            # DuckDuckGo search (synchronous)
            # ddg returns list of dicts: {"title","href","body"}
            results = ddg(query, max_results=5) or []
            if not results:
                return AgentResponse(success=True, content={"answer": "", "sources": []})

            # Build a context string containing titles and snippets
            snippet_texts = []
            sources = []
            for r in results:
                title = r.get("title") or ""
                href = r.get("href") or r.get("url") or ""
                body = r.get("body") or ""
                snippet = f"{title}\n{body}\nURL: {href}"
                snippet_texts.append(snippet)
                sources.append({"title": title, "url": href, "snippet": body})

            context_for_model = "\n\n---\n\n".join(snippet_texts)
            prompt = (
                "You are an assistant that summarizes search results. "
                "Given the following search snippets, produce a concise answer to the user's query and list top sources.\n\n"
                f"Search snippets:\n{context_for_model}\n\nQuestion: {query}\n\nAnswer (brief):"
            )

            answer = await self._run_gen(prompt)
            return AgentResponse(success=True, content={"answer": answer.strip(), "sources": sources})

        except Exception as e:
            self.logger.exception("InternetAgent failed")
            return AgentResponse(success=False, content=f"Error performing internet search: {e}", metadata={"error": str(e)})
