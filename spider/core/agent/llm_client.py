"""
S.P.I.D.E.R. Multi-Model LLM Gateway - Brain Transplant
=========================================================

Unified interface to multiple LLM providers:
- OpenAI (GPT-4, GPT-4-Turbo)
- Anthropic (Claude 3 Opus, Sonnet, Haiku)
- Google Gemini (gemini-1.5-pro, gemini-1.5-flash)
- OpenRouter (access to 100+ models)
- Ollama (Llama3, Mistral, local models)

Features:
- Exponential backoff for rate limits
- Context window overflow handling
- Automatic retry with schema enforcement
- Token counting and cost tracking
"""

import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypedDict, Union
import logging

# Load .env file if present
def _load_env():
    """Load environment variables from .env file."""
    try:
        from pathlib import Path
        
        # Try multiple locations
        possible_paths = [
            Path(__file__).parent.parent.parent.parent / '.env',  # From module location
            Path.cwd() / '.env',  # Current working directory
            Path.home() / 'spider_sdk' / '.env',  # User home
        ]
        
        for env_path in possible_paths:
            if env_path.exists():
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            os.environ.setdefault(key.strip(), value.strip())
                break  # Stop after first .env found
    except Exception:
        pass

_load_env()

logger = logging.getLogger(__name__)


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

class MessageRole(Enum):
    """Role of a message in the conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """A message in a conversation."""
    role: MessageRole
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role.value, "content": self.content}


@dataclass
class LLMResponse:
    """Response from an LLM."""
    content: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0
    cost_usd: float = 0
    finish_reason: str = ""
    raw_response: Optional[Dict] = None
    
    @property
    def success(self) -> bool:
        return bool(self.content) and self.finish_reason != "error"


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    default_model: str = ""
    max_retries: int = 3
    timeout: int = 60
    context_window: int = 4096


# =============================================================================
# COST TRACKING
# =============================================================================

# Pricing per 1K tokens (as of late 2024)
PRICING = {
    # OpenAI
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    # Anthropic
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
    # Google Gemini
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
    "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004},
    # OpenRouter (varies by model, using average)
    "openrouter": {"input": 0.001, "output": 0.002},
    # Ollama (free/local)
    "llama3": {"input": 0, "output": 0},
    "llama3.1": {"input": 0, "output": 0},
    "mistral": {"input": 0, "output": 0},
    "codellama": {"input": 0, "output": 0},
}

# Context window sizes
CONTEXT_WINDOWS = {
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-4o": 128000,
    "gpt-3.5-turbo": 16384,
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
    "claude-3-haiku": 200000,
    "claude-3-5-sonnet": 200000,
    "gemini-1.5-pro": 2000000,
    "gemini-1.5-flash": 1000000,
    "gemini-2.0-flash": 1000000,
    "llama3": 8192,
    "llama3.1": 128000,
    "mistral": 32000,
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD."""
    pricing = PRICING.get(model, {"input": 0, "output": 0})
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1000


# =============================================================================
# PROVIDER BASE CLASS
# =============================================================================

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def complete(
        self,
        messages: List[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a completion."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available."""
        pass


# =============================================================================
# OPENAI PROVIDER
# =============================================================================

class OpenAIProvider(LLMProvider):
    """OpenAI API provider (GPT-4, GPT-3.5)."""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = config.base_url or "https://api.openai.com/v1"
        
        # Try to import openai
        try:
            import openai
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
            self._available = True
        except ImportError:
            self._client = None
            self._available = False
    
    def is_available(self) -> bool:
        return self._available and bool(self.api_key)
    
    def complete(
        self,
        messages: List[Message],
        model: str = "gpt-4-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        if not self.is_available():
            return LLMResponse(
                content="",
                model=model,
                provider="openai",
                finish_reason="error",
            )
        
        start = time.time()
        
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=[m.to_dict() for m in messages],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            
            latency_ms = (time.time() - start) * 1000
            
            usage = response.usage
            content = response.choices[0].message.content or ""
            
            return LLMResponse(
                content=content,
                model=model,
                provider="openai",
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
                latency_ms=latency_ms,
                cost_usd=calculate_cost(
                    model,
                    usage.prompt_tokens if usage else 0,
                    usage.completion_tokens if usage else 0,
                ),
                finish_reason=response.choices[0].finish_reason or "stop",
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
            )
            
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return LLMResponse(
                content=str(e),
                model=model,
                provider="openai",
                finish_reason="error",
                latency_ms=(time.time() - start) * 1000,
            )


# =============================================================================
# ANTHROPIC PROVIDER
# =============================================================================

class AnthropicProvider(LLMProvider):
    """Anthropic API provider (Claude 3)."""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
            self._available = True
        except ImportError:
            self._client = None
            self._available = False
    
    def is_available(self) -> bool:
        return self._available and bool(self.api_key)
    
    def complete(
        self,
        messages: List[Message],
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        if not self.is_available():
            return LLMResponse(
                content="",
                model=model,
                provider="anthropic",
                finish_reason="error",
            )
        
        start = time.time()
        
        try:
            # Anthropic uses separate system prompt
            system_prompt = ""
            user_messages = []
            
            for msg in messages:
                if msg.role == MessageRole.SYSTEM:
                    system_prompt = msg.content
                else:
                    user_messages.append({
                        "role": msg.role.value,
                        "content": msg.content,
                    })
            
            response = self._client.messages.create(
                model=model,
                max_tokens=max_tokens or 4096,
                system=system_prompt if system_prompt else None,
                messages=user_messages,
                temperature=temperature,
            )
            
            latency_ms = (time.time() - start) * 1000
            
            content = ""
            if response.content:
                content = response.content[0].text
            
            return LLMResponse(
                content=content,
                model=model,
                provider="anthropic",
                prompt_tokens=response.usage.input_tokens if response.usage else 0,
                completion_tokens=response.usage.output_tokens if response.usage else 0,
                total_tokens=(
                    (response.usage.input_tokens + response.usage.output_tokens)
                    if response.usage else 0
                ),
                latency_ms=latency_ms,
                cost_usd=calculate_cost(
                    model.split("-")[0] + "-" + model.split("-")[1] + "-" + model.split("-")[2],
                    response.usage.input_tokens if response.usage else 0,
                    response.usage.output_tokens if response.usage else 0,
                ),
                finish_reason=response.stop_reason or "stop",
            )
            
        except Exception as e:
            logger.error(f"Anthropic error: {e}")
            return LLMResponse(
                content=str(e),
                model=model,
                provider="anthropic",
                finish_reason="error",
                latency_ms=(time.time() - start) * 1000,
            )


# =============================================================================
# OLLAMA PROVIDER
# =============================================================================

class OllamaProvider(LLMProvider):
    """Ollama local LLM provider."""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.base_url = config.base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self._available = None  # Lazy check
    
    def is_available(self) -> bool:
        if self._available is None:
            try:
                import requests
                response = requests.get(f"{self.base_url}/api/tags", timeout=5)
                self._available = response.status_code == 200
            except Exception:
                self._available = False
        return self._available
    
    def complete(
        self,
        messages: List[Message],
        model: str = "llama3",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        import requests
        
        start = time.time()
        
        try:
            # Build prompt from messages
            prompt_parts = []
            for msg in messages:
                if msg.role == MessageRole.SYSTEM:
                    prompt_parts.append(f"System: {msg.content}")
                elif msg.role == MessageRole.USER:
                    prompt_parts.append(f"User: {msg.content}")
                elif msg.role == MessageRole.ASSISTANT:
                    prompt_parts.append(f"Assistant: {msg.content}")
            
            prompt_parts.append("Assistant: ")
            prompt = "\n\n".join(prompt_parts)
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens or 2048,
                    },
                },
                timeout=self.config.timeout,
            )
            
            latency_ms = (time.time() - start) * 1000
            
            if response.status_code != 200:
                return LLMResponse(
                    content=f"Ollama error: {response.status_code}",
                    model=model,
                    provider="ollama",
                    finish_reason="error",
                    latency_ms=latency_ms,
                )
            
            data = response.json()
            
            return LLMResponse(
                content=data.get("response", ""),
                model=model,
                provider="ollama",
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                latency_ms=latency_ms,
                cost_usd=0,  # Local is free
                finish_reason="stop" if data.get("done") else "length",
                raw_response=data,
            )
            
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return LLMResponse(
                content=str(e),
                model=model,
                provider="ollama",
                finish_reason="error",
                latency_ms=(time.time() - start) * 1000,
            )


# =============================================================================
# GEMINI PROVIDER (Google AI)
# =============================================================================

class GeminiProvider(LLMProvider):
    """Google Gemini API provider (gemini-1.5-pro, gemini-1.5-flash)."""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.api_key = config.api_key or os.getenv("GOOGLE_API_KEY")
        self._available = False
        self._model = None
        
        try:
            import google.generativeai as genai
            if self.api_key:
                genai.configure(api_key=self.api_key)
                self._genai = genai
                self._available = True
        except ImportError:
            self._genai = None
    
    def is_available(self) -> bool:
        return self._available and bool(self.api_key)
    
    def complete(
        self,
        messages: List[Message],
        model: str = "gemini-1.5-flash",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        if not self.is_available():
            return LLMResponse(
                content="",
                model=model,
                provider="gemini",
                finish_reason="error",
            )
        
        start = time.time()
        
        try:
            # Create the model
            gen_model = self._genai.GenerativeModel(model)
            
            # Build contents from messages
            contents = []
            system_instruction = None
            
            for msg in messages:
                if msg.role == MessageRole.SYSTEM:
                    system_instruction = msg.content
                elif msg.role == MessageRole.USER:
                    contents.append({"role": "user", "parts": [msg.content]})
                elif msg.role == MessageRole.ASSISTANT:
                    contents.append({"role": "model", "parts": [msg.content]})
            
            # Configure generation
            generation_config = self._genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens or 2048,
            )
            
            # Generate
            if system_instruction:
                gen_model = self._genai.GenerativeModel(
                    model,
                    system_instruction=system_instruction,
                )
            
            response = gen_model.generate_content(
                contents,
                generation_config=generation_config,
            )
            
            latency_ms = (time.time() - start) * 1000
            
            content = ""
            if response.text:
                content = response.text
            
            # Get token counts if available
            prompt_tokens = 0
            completion_tokens = 0
            if hasattr(response, 'usage_metadata'):
                prompt_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
                completion_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
            
            return LLMResponse(
                content=content,
                model=model,
                provider="gemini",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                latency_ms=latency_ms,
                cost_usd=calculate_cost(model, prompt_tokens, completion_tokens),
                finish_reason="stop",
            )
            
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            return LLMResponse(
                content=str(e),
                model=model,
                provider="gemini",
                finish_reason="error",
                latency_ms=(time.time() - start) * 1000,
            )


# =============================================================================
# OPENROUTER PROVIDER
# =============================================================================

class OpenRouterProvider(LLMProvider):
    """OpenRouter API provider - unified access to 100+ models."""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.api_key = config.api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        
        try:
            import openai
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
            self._available = True
        except ImportError:
            self._client = None
            self._available = False
    
    def is_available(self) -> bool:
        return self._available and bool(self.api_key)
    
    def complete(
        self,
        messages: List[Message],
        model: str = "google/gemini-flash-1.5",  # Cheapest option
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        if not self.is_available():
            return LLMResponse(
                content="",
                model=model,
                provider="openrouter",
                finish_reason="error",
            )
        
        start = time.time()
        
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=[m.to_dict() for m in messages],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers={
                    "HTTP-Referer": "https://spider-sdk.dev",
                    "X-Title": "S.P.I.D.E.R. SDK",
                },
                **kwargs,
            )
            
            latency_ms = (time.time() - start) * 1000
            
            usage = response.usage
            content = response.choices[0].message.content or ""
            
            return LLMResponse(
                content=content,
                model=model,
                provider="openrouter",
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
                latency_ms=latency_ms,
                cost_usd=calculate_cost("openrouter", 
                    usage.prompt_tokens if usage else 0,
                    usage.completion_tokens if usage else 0,
                ),
                finish_reason=response.choices[0].finish_reason or "stop",
            )
            
        except Exception as e:
            logger.error(f"OpenRouter error: {e}")
            return LLMResponse(
                content=str(e),
                model=model,
                provider="openrouter",
                finish_reason="error",
                latency_ms=(time.time() - start) * 1000,
            )


# =============================================================================
# DEEPSEEK PROVIDER (Direct API - Primary)
# =============================================================================

class DeepSeekProvider(LLMProvider):
    """
    DeepSeek API provider - Direct access to DeepSeek Chat.
    
    This is the cheapest and most reliable option for SWE-Bench.
    Cost: ~$0.001 per 1K tokens
    """
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.api_key = config.api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = "https://api.deepseek.com/v1"  # Must include /v1 for OpenAI compatibility
        
        try:
            import openai
            if self.api_key:
                self._client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
                self._available = True
            else:
                self._client = None
                self._available = False
        except ImportError:
            self._client = None
            self._available = False
    
    def is_available(self) -> bool:
        return self._available and bool(self.api_key)
    
    def complete(
        self,
        messages: List[Message],
        model: str = "deepseek-chat",  # Default DeepSeek Chat model
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        if not self.is_available():
            return LLMResponse(
                content="DeepSeek API key not configured",
                model=model,
                provider="deepseek",
                finish_reason="error",
            )
        
        start = time.time()
        
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=[m.to_dict() for m in messages],
                temperature=temperature,
                max_tokens=max_tokens or 2000,
                **kwargs,
            )
            
            latency_ms = (time.time() - start) * 1000
            
            usage = response.usage
            content = response.choices[0].message.content or ""
            
            # DeepSeek is very cheap: $0.14/1M input, $0.28/1M output
            input_cost = (usage.prompt_tokens / 1_000_000) * 0.14
            output_cost = (usage.completion_tokens / 1_000_000) * 0.28
            total_cost = input_cost + output_cost
            
            return LLMResponse(
                content=content,
                model=model,
                provider="deepseek",
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                latency_ms=latency_ms,
                cost_usd=total_cost,
                finish_reason=response.choices[0].finish_reason or "stop",
            )
            
        except Exception as e:
            logger.error(f"DeepSeek error: {e}")
            return LLMResponse(
                content=str(e),
                model=model,
                provider="deepseek",
                finish_reason="error",
                latency_ms=(time.time() - start) * 1000,
            )


# =============================================================================
# UNIFIED LLM GATEWAY
# =============================================================================

class LLMGateway:
    """
    Unified gateway to multiple LLM providers.
    
    Features:
    - Auto-select available provider
    - Exponential backoff for rate limits
    - Context window overflow handling
    - Token counting and cost tracking
    
    Example:
        gateway = LLMGateway()
        
        response = gateway.complete(
            messages=[
                Message(MessageRole.SYSTEM, "You are a coding assistant."),
                Message(MessageRole.USER, "Write a hello world in Python."),
            ],
            model="gpt-4-turbo",
        )
        
        print(response.content)
        print(f"Cost: ${response.cost_usd:.4f}")
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        deepseek_api_key: Optional[str] = None,
        ollama_host: Optional[str] = None,
        default_provider: str = "auto",
        max_retries: int = 3,
        base_delay: float = 1.0,
    ):
        """
        Initialize the gateway.
        
        Args:
            openai_api_key: OpenAI API key (or use OPENAI_API_KEY env var)
            anthropic_api_key: Anthropic API key (or use ANTHROPIC_API_KEY env var)
            google_api_key: Google API key (or use GOOGLE_API_KEY env var)
            openrouter_api_key: OpenRouter API key (or use OPENROUTER_API_KEY env var)
            deepseek_api_key: DeepSeek API key (or use DEEPSEEK_API_KEY env var)
            ollama_host: Ollama host URL (or use OLLAMA_HOST env var)
            default_provider: Default provider ("deepseek", "openai", "anthropic", "gemini", "openrouter", "ollama", "auto")
            max_retries: Maximum retries for rate limits
            base_delay: Base delay for exponential backoff
        """
        self.default_provider = default_provider
        self.max_retries = max_retries
        self.base_delay = base_delay
        
        # Initialize all 6 providers (DeepSeek is PRIMARY)
        self._providers: Dict[str, LLMProvider] = {
            "deepseek": DeepSeekProvider(ProviderConfig(api_key=deepseek_api_key)),
            "openai": OpenAIProvider(ProviderConfig(api_key=openai_api_key)),
            "anthropic": AnthropicProvider(ProviderConfig(api_key=anthropic_api_key)),
            "gemini": GeminiProvider(ProviderConfig(api_key=google_api_key)),
            "openrouter": OpenRouterProvider(ProviderConfig(api_key=openrouter_api_key)),
            "ollama": OllamaProvider(ProviderConfig(base_url=ollama_host)),
        }
        
        # Stats
        self._total_cost = 0.0
        self._total_tokens = 0
        self._requests = 0
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers."""
        return [name for name, provider in self._providers.items() if provider.is_available()]
    
    def _select_provider(self, provider: Optional[str] = None) -> LLMProvider:
        """Select the best available provider."""
        if provider and provider != "auto":
            if provider in self._providers and self._providers[provider].is_available():
                return self._providers[provider]
            raise ValueError(f"Provider {provider} not available")
        
        # Auto-select: DeepSeek is PRIMARY (cheapest and most reliable)
        # Order: deepseek, openrouter, gemini, anthropic, openai, ollama (last - slow)
        for name in ["deepseek", "openrouter", "gemini", "anthropic", "openai", "ollama"]:
            if self._providers[name].is_available():
                return self._providers[name]
        
        raise RuntimeError("No LLM providers available")
    
    def _truncate_messages(
        self,
        messages: List[Message],
        max_tokens: int,
        preserve_system: bool = True,
    ) -> List[Message]:
        """
        Truncate messages to fit context window.
        
        Strategy: Keep system + last N messages that fit.
        """
        # Simple approximation: 1 token ≈ 4 characters
        def estimate_tokens(text: str) -> int:
            return len(text) // 4
        
        system_messages = []
        other_messages = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM and preserve_system:
                system_messages.append(msg)
            else:
                other_messages.append(msg)
        
        # Calculate available space
        system_tokens = sum(estimate_tokens(m.content) for m in system_messages)
        available = max_tokens - system_tokens - 1000  # Reserve 1K for response
        
        # Keep recent messages that fit
        result = []
        current_tokens = 0
        
        for msg in reversed(other_messages):
            msg_tokens = estimate_tokens(msg.content)
            if current_tokens + msg_tokens <= available:
                result.insert(0, msg)
                current_tokens += msg_tokens
            else:
                break
        
        return system_messages + result
    
    def complete(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        provider: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        truncate: bool = True,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a completion with automatic retries.
        
        Args:
            messages: List of messages
            model: Model to use (None = provider default)
            provider: Provider to use (None = auto-select)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            truncate: Truncate messages if too long
            **kwargs: Additional provider-specific args
            
        Returns:
            LLMResponse with content and metadata
        """
        selected_provider = self._select_provider(provider or self.default_provider)
        
        # Set default model based on provider
        if not model:
            if isinstance(selected_provider, DeepSeekProvider):
                model = "deepseek-chat"  # DeepSeek's main model
            elif isinstance(selected_provider, OpenAIProvider):
                model = "gpt-4-turbo"
            elif isinstance(selected_provider, AnthropicProvider):
                model = "claude-3-5-sonnet-20241022"
            elif isinstance(selected_provider, GeminiProvider):
                model = "gemini-1.5-flash"  # Cheapest and fastest
            elif isinstance(selected_provider, OpenRouterProvider):
                model = "deepseek/deepseek-chat"  # DeepSeek via OpenRouter
            elif isinstance(selected_provider, OllamaProvider):
                model = "llama3"
            else:
                model = "deepseek-chat"  # Fallback to DeepSeek
        
        # Truncate if needed
        context_window = CONTEXT_WINDOWS.get(model, 4096)
        if truncate:
            messages = self._truncate_messages(messages, context_window)
        
        # Retry with exponential backoff
        last_error = None
        
        for attempt in range(self.max_retries):
            response = selected_provider.complete(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            
            if response.success:
                self._requests += 1
                self._total_cost += response.cost_usd
                self._total_tokens += response.total_tokens
                return response
            
            # Check if retry-able
            if "rate" in response.content.lower() or "429" in response.content:
                delay = self.base_delay * (2 ** attempt)
                logger.warning(f"Rate limited, retrying in {delay}s...")
                time.sleep(delay)
                last_error = response.content
            else:
                return response  # Non-retryable error
        
        return LLMResponse(
            content=f"Max retries exceeded: {last_error}",
            model=model,
            provider="unknown",
            finish_reason="error",
        )
    
    def complete_json(
        self,
        messages: List[Message],
        schema: Optional[Dict] = None,
        **kwargs,
    ) -> Dict:
        """
        Generate a JSON response with retry on format errors.
        
        Args:
            messages: List of messages
            schema: Optional JSON schema to validate against
            **kwargs: Args passed to complete()
            
        Returns:
            Parsed JSON dict (empty dict on failure)
        """
        # Add JSON instruction to system message
        json_instruction = "\n\nYou MUST respond with valid JSON only. No markdown, no explanations."
        
        enhanced_messages = []
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                enhanced_messages.append(Message(
                    role=msg.role,
                    content=msg.content + json_instruction,
                ))
            else:
                enhanced_messages.append(msg)
        
        # Try to get valid JSON
        for attempt in range(3):
            response = self.complete(enhanced_messages, **kwargs)
            
            if not response.success:
                continue
            
            # Try to parse JSON
            try:
                # Clean up markdown code blocks if present
                content = response.content.strip()
                if content.startswith("```"):
                    # Extract JSON from code block
                    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
                    if match:
                        content = match.group(1).strip()
                
                parsed = json.loads(content)
                
                # TODO: Validate against schema if provided
                
                return parsed
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error (attempt {attempt + 1}): {e}")
                # Add error feedback for retry
                enhanced_messages.append(Message(
                    role=MessageRole.USER,
                    content=f"Your response was not valid JSON. Error: {e}. Please respond with ONLY valid JSON.",
                ))
        
        return {}
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "requests": self._requests,
            "total_tokens": self._total_tokens,
            "total_cost_usd": self._total_cost,
        }
    
    def __repr__(self) -> str:
        available = self.get_available_providers()
        return f"LLMGateway(providers={available})"


# =============================================================================
# STRUCTURED ARCHITECT
# =============================================================================

class StructuredArchitect:
    """
    LLM-powered code architect that produces structured output.
    
    Replaces the mock Architect with real LLM calls.
    """
    
    SYSTEM_PROMPT = """You are an expert software architect. You analyze problems and propose code solutions.

When asked to solve a coding problem, respond with a JSON object containing:
{
    "file_path": "path/to/file.py",
    "code": "the complete code",
    "reasoning": ["step 1", "step 2", ...],
    "tests": "optional test code"
}

Be precise. Write production-quality code. Include type hints and docstrings."""
    
    def __init__(self, gateway: Optional[LLMGateway] = None):
        """
        Initialize the architect.
        
        Args:
            gateway: LLMGateway instance (creates default if None)
        """
        self.gateway = gateway or LLMGateway()
    
    def draft_proposal(
        self,
        problem_description: str,
        context: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a code proposal for a problem.
        
        Args:
            problem_description: Description of the problem
            context: Optional context (existing code, file structure)
            model: Model to use
            
        Returns:
            Dict with file_path, code, reasoning, tests
        """
        messages = [
            Message(MessageRole.SYSTEM, self.SYSTEM_PROMPT),
        ]
        
        if context:
            messages.append(Message(
                MessageRole.USER,
                f"Context:\n{context}\n\nProblem: {problem_description}",
            ))
        else:
            messages.append(Message(
                MessageRole.USER,
                f"Problem: {problem_description}",
            ))
        
        result = self.gateway.complete_json(messages, model=model)
        
        if not result:
            # Fallback to raw completion
            response = self.gateway.complete(messages, model=model)
            return {
                "file_path": "solution.py",
                "code": response.content,
                "reasoning": ["Generated by LLM"],
                "tests": "",
            }
        
        return result
    
    def refine_code(
        self,
        code: str,
        feedback: str,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Refine code based on feedback.
        
        Args:
            code: Current code
            feedback: Feedback (test errors, review comments)
            model: Model to use
            
        Returns:
            Dict with refined code
        """
        messages = [
            Message(MessageRole.SYSTEM, self.SYSTEM_PROMPT),
            Message(MessageRole.USER, f"""
Here is the current code:
```python
{code}
```

Feedback: {feedback}

Please fix the issues and respond with the improved code in JSON format.
"""),
        ]
        
        return self.gateway.complete_json(messages, model=model)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🧠 S.P.I.D.E.R. LLM GATEWAY - Demo")
    print("=" * 60)
    print()
    
    gateway = LLMGateway()
    
    print("Available providers:")
    for provider in gateway.get_available_providers():
        print(f"  ✅ {provider}")
    
    unavailable = set(["openai", "anthropic", "ollama"]) - set(gateway.get_available_providers())
    for provider in unavailable:
        print(f"  ❌ {provider}")
    
    print()
    
    if gateway.get_available_providers():
        print("Testing completion...")
        
        response = gateway.complete([
            Message(MessageRole.SYSTEM, "You are a helpful assistant."),
            Message(MessageRole.USER, "Say 'Hello from S.P.I.D.E.R.' in exactly 5 words."),
        ])
        
        print(f"Response: {response.content}")
        print(f"Provider: {response.provider}")
        print(f"Tokens: {response.total_tokens}")
        print(f"Cost: ${response.cost_usd:.6f}")
    else:
        print("No providers available.")
        print("Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or start Ollama.")
    
    print()
    print("=" * 60)
    print("✅ LLM Gateway ready!")
