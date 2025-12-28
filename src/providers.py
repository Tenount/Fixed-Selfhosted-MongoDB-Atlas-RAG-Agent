"""Model providers for Semantic Search Agent."""

import re
import httpx
from typing import Optional
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from src.settings import load_settings


class SafeHttpxClient(httpx.AsyncClient):
    """Custom httpx client that cleans surrogate characters from JSON requests."""
    
    def _clean_json(self, obj):
        if isinstance(obj, str):
            return re.sub(r'[\ud800-\udfff]', '�', obj)
        elif isinstance(obj, dict):
            return {k: self._clean_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_json(i) for i in obj]
        return obj
    
    def build_request(self, method, url, *, content=None, **kwargs):
        # Clean JSON payloads before encoding
        if 'json' in kwargs:
            kwargs['json'] = self._clean_json(kwargs['json'])
        return super().build_request(method, url, content=content, **kwargs)

def get_llm_model(model_choice: Optional[str] = None) -> OpenAIModel:
    """
    Get LLM model configuration based on environment variables.
    Supports any OpenAI-compatible API provider.

    Args:
        model_choice: Optional override for model choice

    Returns:
        Configured OpenAI-compatible model
    """
    settings = load_settings()

    llm_choice = model_choice or settings.llm_model
    base_url = settings.llm_base_url
    api_key = settings.llm_api_key

    # Create safe httpx client
    safe_client = SafeHttpxClient(timeout=30.0)

    provider = OpenAIProvider(
        base_url=base_url,
        api_key=api_key,
        http_client=safe_client,
    )

    return OpenAIModel(llm_choice, provider=provider)


def get_embedding_model() -> OpenAIModel:
    """
    Get embedding model configuration.
    Uses OpenAI embeddings API (or compatible provider).

    Returns:
        Configured embedding model
    """
    settings = load_settings()
    
    safe_client = SafeHttpxClient(timeout=30.0)
    
    provider = OpenAIProvider(
        base_url=settings.llm_base_url, 
        api_key=settings.llm_api_key,
        http_client=safe_client  # Pass custom client here too
    )

    return OpenAIModel(settings.embedding_model, provider=provider)


def get_model_info() -> dict:
    """
    Get information about current model configuration.

    Returns:
        Dictionary with model configuration info
    """
    settings = load_settings()

    return {
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "llm_base_url": settings.llm_base_url,
        "embedding_model": settings.embedding_model,
    }


def validate_llm_configuration() -> bool:
    """
    Validate that LLM configuration is properly set.

    Returns:
        True if configuration is valid
    """
    try:
        # Check if we can create a model instance
        get_llm_model()
        return True
    except Exception as e:
        print(f"LLM configuration validation failed: {e}")
        return False
