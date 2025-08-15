# backend/agents/base_agent.py
"""Base agent class for all specialized agents with lifecycle, logging, and sync helpers."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import asyncio
import logging
from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    """Standardized response format for all agents."""
    success: bool = Field(..., description="Whether the operation was successful")
    content: Any = Field(..., description="The response content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BaseAgent(ABC):
    """
    Base class for agents.

    Features:
      - lifecycle hooks: setup() and shutdown()
      - async abstract process() method (implementations should be async)
      - run_sync() helper to call process() from synchronous code (uses asyncio.run)
      - simple timeout support for process calls
      - per-agent logger
      - convenience factories for success/failure AgentResponse
    """

    def __init__(self, name: str, description: str, timeout: Optional[float] = None, logger: Optional[logging.Logger] = None):
        self.name = name
        self.description = description
        self.is_active = True
        self._timeout = timeout  # default per-agent timeout in seconds (can be overridden per-call)

        # Logger: if not provided, create one scoped to agent name
        self.logger = logger or logging.getLogger(f"minemeets.agent.{self.name}")
        if not self.logger.handlers:
            # Minimal default config if the app hasn't configured logging
            handler = logging.StreamHandler()
            fmt = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
            handler.setFormatter(fmt)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    # ---------------- lifecycle hooks ----------------
    async def setup(self) -> None:
        """
        Optional async setup hook.
        Implementations can override to load models, open DB connections, etc.
        Called by external orchestrator before using the agent.
        """
        self.logger.debug("Default setup() called - no action.")

    async def shutdown(self) -> None:
        """
        Optional async shutdown hook.
        Implementations can override to free resources.
        """
        self.logger.debug("Default shutdown() called - no action.")

    # ---------------- core abstract ----------------
    @abstractmethod
    async def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Process the input and return an AgentResponse.

        Subclasses MUST implement this as an async method.
        """
        raise NotImplementedError()

    # ---------------- convenience runners ----------------
    async def _process_with_timeout(self, *args, timeout: Optional[float] = None, **kwargs) -> AgentResponse:
        """Internal wrapper to run process() with optional timeout."""
        t = timeout if timeout is not None else self._timeout
        if t:
            return await asyncio.wait_for(self.process(*args, **kwargs), timeout=t)
        return await self.process(*args, **kwargs)

    def run_sync(self, *args, timeout: Optional[float] = None, **kwargs) -> AgentResponse:
        """
        Synchronous wrapper for calling the async process().

        - If an asyncio event loop is already running, this method raises a clear error:
          you must `await agent.process(...)` instead in an async context.
        - Otherwise it will use asyncio.run(...) and return AgentResponse.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We cannot block inside an existing running loop; caller should await process()
            raise RuntimeError(
                f"Cannot call run_sync() while an asyncio event loop is running. "
                f"Use 'await {self.__class__.__name__}.process(...)' instead."
            )

        return asyncio.run(self._process_with_timeout(*args, timeout=timeout, **kwargs))

    # ---------------- helper factories ----------------
    def success_response(self, content: Any, metadata: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Convenience to create a successful AgentResponse."""
        return AgentResponse(success=True, content=content, metadata=metadata or {})

    def failure_response(self, content: Any, metadata: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Convenience to create a failed AgentResponse."""
        return AgentResponse(success=False, content=content, metadata=metadata or {})

    # ---------------- small utilities ----------------
    def validate_context(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Normalize / validate context. Returns a dict (possibly empty)."""
        if context is None:
            return {}
        if not isinstance(context, dict):
            self.logger.warning("Context provided is not a dict, converting to dict with 'value' key.")
            return {"value": context}
        return context

    def get_status(self) -> Dict[str, Any]:
        """Return the current status of the agent (for health endpoints / dashboards)."""
        return {
            "name": self.name,
            "description": self.description,
            "is_active": self.is_active
        }

    # synchronous call operator not provided intentionally to avoid ambiguity;
    # keep __call__ async so agents behave predictably in async code.
    async def __call__(self, *args, **kwargs) -> AgentResponse:
        """Make the agent awaitable/callable in async code."""
        return await self.process(*args, **kwargs)
