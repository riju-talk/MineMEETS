# agents/llm.py
import gc
import logging
import threading
import time
from typing import Optional, Dict, Any, List, TypeVar, cast

import torch
from pydantic import BaseModel, Field, validator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
    TextGenerationPipeline,
    PreTrainedModel,
    PreTrainedTokenizer
)
from langchain.llms import HuggingFacePipeline
from langchain.llms.base import LLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

T = TypeVar('T')

class LLMConfig(BaseModel):
    """Configuration for LLM initialization with validation."""
    model_id: str = Field(
        default="TheBloke/Mistral-7B-Instruct-v0.1-GPTQ",
        description="HuggingFace model ID or path"
    )
    device_map: str = Field(
        default="auto",
        description="Device map for model loading (auto, cuda, cpu, etc.)"
    )
    max_new_tokens: int = Field(
        default=512,
        description="Maximum number of new tokens to generate",
        gt=0
    )
    temperature: float = Field(
        default=0.7,
        description="Sampling temperature (0.0 to 1.0)",
        ge=0.0,
        le=2.0
    )
    top_p: float = Field(
        default=0.9,
        description="Nucleus sampling top-p parameter",
        gt=0.0,
        le=1.0
    )
    top_k: int = Field(
        default=50,
        description="Top-k sampling parameter",
        gt=0
    )
    repetition_penalty: float = Field(
        default=1.1,
        description="Penalty for repeating tokens",
        ge=1.0
    )
    load_in_4bit: bool = Field(
        default=True,
        description="Whether to load the model in 4-bit precision"
    )
    bnb_4bit_compute_dtype: str = Field(
        default="float16",
        description="Compute dtype for 4-bit quantization (float16 or bfloat16)"
    )
    bnb_4bit_quant_type: str = Field(
        default="nf4",
        description="Quantization type for 4-bit (nf4 or fp4)"
    )
    bnb_4bit_use_double_quant: bool = Field(
        default=True,
        description="Whether to use double quantization for 4-bit"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries for generation",
        ge=1
    )
    request_timeout: int = Field(
        default=300,
        description="Timeout in seconds for model requests",
        gt=0
    )

    @validator('bnb_4bit_compute_dtype')
    def validate_compute_dtype(cls, v):
        if v not in ['float16', 'bfloat16']:
            raise ValueError("bnb_4bit_compute_dtype must be 'float16' or 'bfloat16'")
        return v
    
    @validator('bnb_4bit_quant_type')
    def validate_quant_type(cls, v):
        if v not in ['nf4', 'fp4']:
            raise ValueError("bnb_4bit_quant_type must be 'nf4' or 'fp4'")
        return v

class LLMFactory:
    """Thread-safe factory for creating and managing LLM instances with resource management."""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, config: Optional[LLMConfig] = None) -> 'LLMFactory':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    logger.info("Initializing LLMFactory singleton instance")
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize(config or LLMConfig())
        return cls._instance
    
    def _initialize(self, config: LLMConfig) -> None:
        """Initialize the LLM factory with the given configuration."""
        logger.info(f"Initializing LLM with config: {config}")
        self.config = config
        self._model: Optional[PreTrainedModel] = None
        self._tokenizer: Optional[PreTrainedTokenizer] = None
        self._pipeline: Optional[TextGenerationPipeline] = None
        self._langchain_llm: Optional[LLM] = None
        self._load_lock = threading.RLock()
        self._is_loaded = False
    
    def _load_model(self) -> None:
        """Safely load the model and tokenizer with proper error handling."""
        with self._load_lock:
            if self._is_loaded:
                return
                
            logger.info(f"Loading model: {self.config.model_id}")
            
            try:
                # Configure quantization if enabled
                bnb_config = None
                if self.config.load_in_4bit:
                    logger.debug("Configuring 4-bit quantization")
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                        bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                        bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
                    )
                
                # Load tokenizer first (faster to fail if there's an issue)
                logger.debug("Loading tokenizer")
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_id,
                    use_fast=True,
                    trust_remote_code=True
                )
                
                # Load model with error handling for CUDA OOM
                logger.debug("Loading model")
                try:
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.config.model_id,
                        device_map=self.config.device_map,
                        quantization_config=bnb_config,
                        trust_remote_code=True,
                        torch_dtype=torch.float16 if not self.config.load_in_4bit else None,
                        low_cpu_mem_usage=True
                    )
                except torch.cuda.OutOfMemoryError as e:
                    logger.error("CUDA out of memory. Try reducing model size or enabling 4-bit.")
                    raise RuntimeError("CUDA out of memory. Try a smaller model or enable 4-bit quantization.") from e
                
                # Create text generation pipeline
                logger.debug("Creating text generation pipeline")
                self._pipeline = pipeline(
                    "text-generation",
                    model=self._model,
                    tokenizer=self._tokenizer,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    repetition_penalty=self.config.repetition_penalty,
                    pad_token_id=self._tokenizer.eos_token_id,
                    device_map=self.config.device_map
                )
                
                # Create LangChain compatible LLM
                logger.debug("Creating LangChain LLM wrapper")
                self._langchain_llm = HuggingFacePipeline(pipeline=self._pipeline)
                self._is_loaded = True
                logger.info("Model loaded successfully")
                
            except Exception as e:
                self._cleanup()
                logger.exception(f"Failed to load model {self.config.model_id}")
                raise RuntimeError(f"Failed to load model {self.config.model_id}: {str(e)}") from e
    
    def _cleanup(self) -> None:
        """Clean up model resources."""
        with self._load_lock:
            logger.debug("Cleaning up model resources")
            if hasattr(self, '_pipeline') and self._pipeline is not None:
                del self._pipeline
                self._pipeline = None
            
            if hasattr(self, '_model') and self._model is not None:
                del self._model
                self._model = None
            
            if hasattr(self, '_tokenizer') and self._tokenizer is not None:
                del self._tokenizer
                self._tokenizer = None
            
            self._langchain_llm = None
            self._is_loaded = False
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def get_langchain_llm(self) -> LLM:
        """Get a thread-safe LangChain compatible LLM instance."""
        self._load_model()
        if self._langchain_llm is None:
            raise RuntimeError("Failed to initialize LangChain LLM")
        return self._langchain_llm
    
    def generate(
        self, 
        prompt: str, 
        max_retries: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate text from a prompt with retry logic and error handling.
        
        Args:
            prompt: The input prompt text
            max_retries: Maximum number of retry attempts
            **kwargs: Additional generation parameters
            
        Returns:
            The generated text
            
        Raises:
            RuntimeError: If generation fails after all retries
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")
            
        max_retries = max_retries or self.config.max_retries
        last_error = None
        
        for attempt in range(max_retries):
            try:
                self._load_model()
                if self._pipeline is None:
                    raise RuntimeError("Text generation pipeline not available")
                
                # Generate with timeout
                result = self._pipeline(
                    prompt,
                    max_new_tokens=kwargs.get('max_new_tokens', self.config.max_new_tokens),
                    temperature=kwargs.get('temperature', self.config.temperature),
                    top_p=kwargs.get('top_p', self.config.top_p),
                    top_k=kwargs.get('top_k', self.config.top_k),
                    repetition_penalty=kwargs.get('repetition_penalty', self.config.repetition_penalty),
                    **{k: v for k, v in kwargs.items() if k not in [
                        'max_new_tokens', 'temperature', 'top_p', 'top_k', 'repetition_penalty'
                    ]}
                )
                
                # Extract and validate the generated text
                if not result or not isinstance(result, list) or not result[0].get('generated_text'):
                    raise ValueError("Unexpected response format from model")
                    
                return result[0]['generated_text'].strip()
                
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Generation attempt {attempt + 1}/{max_retries} failed: {str(e)}",
                    exc_info=attempt == max_retries - 1  # Log full trace on last attempt
                )
                
                # On CUDA errors, try to recover by cleaning up
                if "CUDA" in str(e):
                    self._cleanup()
                    # Wait before retry
                    time.sleep(1)
        
        # If we get here, all retries failed
        error_msg = f"Failed to generate text after {max_retries} attempts"
        logger.error(error_msg, exc_info=last_error)
        raise RuntimeError(f"{error_msg}: {str(last_error)}")
    
    def __del__(self):
        """Ensure resources are cleaned up when the factory is garbage collected."""
        self._cleanup()

# --- Ollama support ---
import requests

class OllamaLLM:
    """Lightweight LangChain-compatible LLM for local Ollama server (REST API)."""
    def __init__(self, model: str, host: str = "http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")
    def __call__(self, prompt: str, **kwargs) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        params = kwargs.copy()
        payload.update(params)
        resp = requests.post(f"{self.host}/api/generate", json=payload, timeout=60)
        resp.raise_for_status()
        out = resp.json()
        if 'response' in out:
            return out['response'].strip()
        if 'message' in out:
            return out['message'].strip()
        return str(out)


def get_llm_provider():
    """Dynamically choose Ollama or HuggingFace LLM instance."""
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL")
    if ollama_model:
        try:
            # Test Ollama connection
            resp = requests.get(f"{ollama_host}/api/tags", timeout=3)
            if resp.ok:
                return OllamaLLM(ollama_model, ollama_host)
        except Exception as e:
            logger.warning(f"Ollama not available: {e}. Falling back to HF model.")
    # Default to huggingface model via factory
    return llm_factory.get_langchain_llm()

# Global instance for easy access
llm_factory = LLMFactory()
