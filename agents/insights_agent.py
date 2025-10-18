from typing import Dict, Any, Optional, List
import asyncio
from agents.llm import LLM

class InsightsAgent:
    def __init__(self):
        self.llm = LLM(model="llama3.1", temperature=0.0)
        # Pre-built prompts
        self.prompts = {
            "summary": "Summarize the following meeting transcript into a concise paragraph:\n\n{transcript}\n\nSummary:",
            "key_decisions": "List the key decisions made during the meeting. Provide short bullets:\n\n{transcript}\n\nDecisions:",
            "action_items": "List action items mentioned in the transcript. For each action item, include owner (if available) and due date (if available). Use JSON list format:\n\n{transcript}\n\nAction items:"
        }

    async def _generate(self, prompt: str) -> str:
        """Run generation using LLM."""
        return await self.llm.generate_async(prompt)

    async def process(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            transcript = input_data.get("transcript", "")
            if not transcript:
                return {"success": False, "content": "Transcript missing"}

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

            return {"success": True, "content": content}

        except Exception as e:
            return {"success": False, "content": f"Failed to extract insights: {e}"}
