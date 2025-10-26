# agents/config.py
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mine_meets.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class Config:
    """Configuration management with validation and error handling."""

    def __init__(self):
        self.required_env_vars = [
            'PINECONE_API_KEY',
            'OLLAMA_HOST',
            'OLLAMA_MODEL'
        ]

        self.optional_env_vars = {
            'PINECONE_ENVIRONMENT': 'us-west-2',
            'PINECONE_INDEX': 'mine-meets',
            'WHISPER_MODEL': 'base',
            'WHISPER_CACHE_DIR': '.cache/whisper',
            'EMBEDDING_MODEL': 'sentence-transformers/clip-ViT-B-32',
            'EMBEDDING_DIM': '512'
        }

        # Only validate if explicitly requested
        self._validated = False

    def validate_config(self) -> None:
        """Validate configuration and provide helpful error messages."""
        missing_vars = []

        for var in self.required_env_vars:
            if not os.getenv(var):
                missing_vars.append(var)

        if missing_vars:
            error_msg = f"""
Missing required environment variables: {', '.join(missing_vars)}

Please set these in your .env file or environment:

Required variables:
"""
            for var in missing_vars:
                error_msg += f"  {var}=your-{var.lower()}\n"

            error_msg += """
Optional variables (with defaults):
"""
            for var, default in self.optional_env_vars.items():
                error_msg += f"  {var}={default}  # Optional\n"

            raise ValueError(error_msg)

        # Validate specific configurations
        self._validate_pinecone_config()
        self._validate_ollama_config()

    def _validate_pinecone_config(self) -> None:
        """Validate Pinecone configuration."""
        try:
            api_key = os.getenv('PINECONE_API_KEY')
            if not api_key or len(api_key) < 10:
                raise ValueError("PINECONE_API_KEY appears invalid")

            # Test connection if possible
            logger.info("Pinecone configuration validated")

        except Exception as e:
            logger.warning(f"Pinecone validation warning: {str(e)}")

    def _validate_ollama_config(self) -> None:
        """Validate Ollama configuration."""
        try:
            import requests

            ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
            ollama_model = os.getenv('OLLAMA_MODEL', 'llama3.1')

            # Test connection (optional, don't fail if Ollama not running yet)
            try:
                response = requests.get(f"{ollama_host}/api/tags", timeout=5)
                if response.status_code == 200:
                    models = response.json().get('models', [])
                    model_names = [model['name'] for model in models]

                    if ollama_model not in model_names:
                        logger.warning(f"Ollama model '{ollama_model}' not found. Available models: {model_names}")
                    else:
                        logger.info(f"Ollama model '{ollama_model}' validated")
                else:
                    logger.warning("Could not connect to Ollama API")

            except requests.exceptions.RequestException:
                logger.warning("Could not connect to Ollama (may not be running yet)")

        except Exception as e:
            logger.warning(f"Ollama validation warning: {str(e)}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return os.getenv(key, self.optional_env_vars.get(key, default))

    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration as dictionary."""
        config = {}

        # Add all required vars
        for var in self.required_env_vars:
            config[var] = os.getenv(var)

        # Add all optional vars with defaults
        for var, default in self.optional_env_vars.items():
            config[var] = os.getenv(var, default)

        return config

    def create_env_template(self, file_path: str = '.env.example') -> None:
        """Create an environment template file."""
        template_content = """# MineMEETS Configuration Template
# Copy this to .env and fill in your values

# Required Variables
PINECONE_API_KEY=your-pinecone-api-key-here
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1

# Optional Variables
PINECONE_ENVIRONMENT=us-west-2
PINECONE_INDEX=mine-meets
WHISPER_MODEL=base
WHISPER_CACHE_DIR=.cache/whisper
EMBEDDING_MODEL=sentence-transformers/clip-ViT-B-32
EMBEDDING_DIM=512

# Instructions:
# 1. Get Pinecone API key from https://app.pinecone.io/
# 2. Install and start Ollama from https://ollama.com/
# 3. Run: ollama pull llama3.1
"""

        try:
            Path(file_path).write_text(template_content)
            logger.info(f"Created environment template: {file_path}")
        except Exception as e:
            logger.error(f"Failed to create environment template: {str(e)}")

# Global configuration instance
config = Config()
