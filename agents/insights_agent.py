# agents/insights_agent.py
from typing import Dict, Any, Optional, List
from .base_agent import BaseAgent, AgentResponse
import os
import asyncio
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from textwrap import shorten

class InsightsAgent(BaseAgent):
    """Agent that extracts summaries, decisions, and action items using a local quantized HF model."""

    def __init__(self):
        super().__init__(name="insights_agent", description="Extracts summary, key decisions, and action items from transcripts.")
        # Load model configuration from env
        model_id = os.getenv("LLM_MODEL_ID", "unsloth/Meta-Llama-3-8B-Instruct-bnb-4bit")
        max_new_tokens = int(os.getenv("LLM_MAX_NEW_TOKENS", "256"))
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))

        # 4-bit quantization config (GPU preferred)
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Tokenizer + model + pipeline - these loads are blocking so we do in sync init
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

        # Pre-built prompts
        self.prompts = {
            "summary": "Summarize the following meeting transcript into a concise paragraph:\n\n{transcript}\n\nSummary:",
            "key_decisions": "List the key decisions made during the meeting. Provide short bullets:\n\n{transcript}\n\nDecisions:",
            "action_items": "List action items mentioned in the transcript. For each action item, include owner (if available) and due date (if available). Use JSON list format:\n\n{transcript}\n\nAction items:"
        }

    def _generate_sync(self, prompt: str) -> str:
        """Blocking generation call used inside async wrapper."""
        out = self.gen_pipe(prompt)
        if not out:
            return ""
        # pipeline returns a list of dicts with 'generated_text'
        return out[0].get("generated_text", "")

    async def _generate(self, prompt: str) -> str:
        """Run generation in thread to avoid blocking event loop."""
        return await asyncio.to_thread(self._generate_sync, prompt)

    async def process(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        try:
            transcript = input_data.get("transcript", "")
            if not transcript:
                return AgentResponse(success=False, content="Transcript missing", metadata={})

            # Run each prompt and collect results
            summary_prompt = self.prompts["summary"].format(transcript=transcript)
            decisions_prompt = self.prompts["key_decisions"].format(transcript=transcript)
            actions_prompt = self.prompts["action_items"].format(transcript=transcript)

            # Call generator concurrently
            summary_task = asyncio.create_task(self._generate(summary_prompt))
            decisions_task = asyncio.create_task(self._generate(decisions_prompt))
            actions_task = asyncio.create_task(self._generate(actions_prompt))

            summary = await summary_task
            decisions = await decisions_task
            actions_raw = await actions_task

            # Try to parse action items as JSON if model returned JSON-like output, else keep raw
            action_items = []
            try:
                import json
                candidate = actions_raw.strip()
                if candidate.startswith("[") or candidate.startswith("{"):
                    action_items = json.loads(candidate)
                else:
                    # fallback: split lines into bullets (best-effort)
                    action_items = [line.strip("-• ") for line in actions_raw.splitlines() if line.strip()]
            except Exception:
                action_items = [line.strip() for line in actions_raw.splitlines() if line.strip()]

            content = {
                "summary": summary.strip(),
                "key_points": [line.strip() for line in decisions.splitlines() if line.strip()] or [decisions.strip()],
                "action_items": action_items
            }

            return AgentResponse(success=True, content=content, metadata={"source": "insights_agent", **(context or {})})

        except Exception as e:
            self.logger.exception("Insights extraction failed")
            return AgentResponse(success=False, content=f"Failed to extract insights: {e}", metadata={"error": str(e)})
