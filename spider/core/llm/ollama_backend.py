"""
S.P.I.D.E.R. Ollama LLM Backend - FREE LOCAL AI
=================================================

The Cost-Free Solution: Run LLMs locally with Ollama.

No API keys. No credits. No rate limits. Unlimited usage.

Supported Models (all FREE):
- deepseek-r1:7b (reasoning, similar to o1)
- qwen2.5-coder:7b (code specialist)
- codellama:7b (code specialist)
- llama3.2:latest (general purpose)
- mistral:7b (fast, good quality)

Setup:
1. Install Ollama: https://ollama.ai/download
2. Run: ollama pull qwen2.5-coder:7b
3. Use this module!
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class OllamaConfig:
    """Configuration for Ollama."""
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5-coder:7b"
    temperature: float = 0.7
    timeout: int = 300  # Local models can be slower


@dataclass
class OllamaResponse:
    """Response from Ollama."""
    content: str
    model: str
    eval_count: int  # Tokens generated
    latency: float
    done: bool


# =============================================================================
# OLLAMA CLIENT
# =============================================================================

class OllamaClient:
    """
    Ollama Local LLM Client - 100% FREE.
    
    Usage:
        # First, install Ollama and pull a model:
        # ollama pull qwen2.5-coder:7b
        
        client = OllamaClient()
        response = client.generate("Write a Python function")
        print(response.content)
    """
    
    # Recommended models for coding
    RECOMMENDED_MODELS = {
        "qwen2.5-coder:7b": "Best for code generation (7B)",
        "deepseek-r1:7b": "Reasoning model like o1 (7B)", 
        "codellama:7b": "Meta's code model (7B)",
        "llama3.2:latest": "General purpose",
        "mistral:7b": "Fast and capable",
        "qwen2.5-coder:1.5b": "Lightweight, fast",
    }
    
    def __init__(self, config: OllamaConfig = None):
        """Initialize Ollama client."""
        self.config = config or OllamaConfig()
        
        self._stats = {
            "requests": 0,
            "tokens_generated": 0,
            "errors": 0,
            "total_latency": 0.0,
        }
    
    def is_running(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = requests.get(
                f"{self.config.base_url}/api/tags",
                timeout=5,
            )
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        """List available models."""
        try:
            response = requests.get(
                f"{self.config.base_url}/api/tags",
                timeout=10,
            )
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except:
            pass
        return []
    
    def pull_model(self, model: str) -> bool:
        """Pull a model (download it)."""
        try:
            print(f"Pulling {model}... (this may take a while)")
            response = requests.post(
                f"{self.config.base_url}/api/pull",
                json={"name": model},
                timeout=1800,  # 30 min for large models
                stream=True,
            )
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    status = data.get("status", "")
                    if "pulling" in status:
                        print(f"  {status}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        system: str = None,
        temperature: float = None,
    ) -> OllamaResponse:
        """
        Generate a response.
        
        Args:
            prompt: User prompt
            system: System prompt
            temperature: Override temperature
            
        Returns:
            OllamaResponse
        """
        self._stats["requests"] += 1
        start_time = time.time()
        
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature or self.config.temperature,
            },
        }
        
        if system:
            payload["system"] = system
        
        try:
            response = requests.post(
                f"{self.config.base_url}/api/generate",
                json=payload,
                timeout=self.config.timeout,
            )
            
            latency = time.time() - start_time
            self._stats["total_latency"] += latency
            
            if response.status_code != 200:
                self._stats["errors"] += 1
                raise Exception(f"Ollama error: {response.text[:200]}")
            
            data = response.json()
            
            tokens = data.get("eval_count", 0)
            self._stats["tokens_generated"] += tokens
            
            return OllamaResponse(
                content=data.get("response", ""),
                model=data.get("model", self.config.model),
                eval_count=tokens,
                latency=latency,
                done=data.get("done", True),
            )
            
        except requests.exceptions.ConnectionError:
            self._stats["errors"] += 1
            raise Exception(
                "Cannot connect to Ollama. Is it running?\n"
                "Start with: ollama serve"
            )
        except requests.exceptions.Timeout:
            self._stats["errors"] += 1
            raise Exception("Request timed out. Model might be loading.")
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
    ) -> OllamaResponse:
        """
        Chat completion (multi-turn).
        
        Args:
            messages: List of {"role": "...", "content": "..."}
            temperature: Override temperature
            
        Returns:
            OllamaResponse
        """
        self._stats["requests"] += 1
        start_time = time.time()
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature or self.config.temperature,
            },
        }
        
        try:
            response = requests.post(
                f"{self.config.base_url}/api/chat",
                json=payload,
                timeout=self.config.timeout,
            )
            
            latency = time.time() - start_time
            self._stats["total_latency"] += latency
            
            if response.status_code != 200:
                self._stats["errors"] += 1
                raise Exception(f"Ollama error: {response.text[:200]}")
            
            data = response.json()
            
            tokens = data.get("eval_count", 0)
            self._stats["tokens_generated"] += tokens
            
            return OllamaResponse(
                content=data.get("message", {}).get("content", ""),
                model=data.get("model", self.config.model),
                eval_count=tokens,
                latency=latency,
                done=data.get("done", True),
            )
            
        except requests.exceptions.ConnectionError:
            self._stats["errors"] += 1
            raise Exception("Cannot connect to Ollama.")
    
    def get_callback(self) -> Callable[[str], str]:
        """
        Get callback function for S.P.I.D.E.R. modules.
        
        This is the key integration - use this with any module!
        """
        def callback(prompt: str) -> str:
            response = self.generate(prompt)
            return response.content
        
        return callback
    
    def get_stats(self) -> Dict[str, Any]:
        avg_latency = (
            self._stats["total_latency"] / max(self._stats["requests"], 1)
        )
        return {
            **self._stats,
            "avg_latency": round(avg_latency, 2),
            "cost": "$0.00",  # ALWAYS FREE!
        }


# =============================================================================
# S.P.I.D.E.R. INTEGRATION
# =============================================================================

class SpiderLocalLLM:
    """
    S.P.I.D.E.R. Local LLM Interface - FREE & UNLIMITED.
    
    Usage:
        llm = SpiderLocalLLM()
        
        # Check setup
        if not llm.is_ready():
            llm.setup()  # Downloads model
        
        # Use directly
        code = llm.generate_code("Fix the bug in auth.py")
        
        # Or get callback for modules
        callback = llm.get_callback()
        sampler = EpistemicSampler(llm_callback=callback)
    """
    
    SYSTEM_PROMPT = """You are S.P.I.D.E.R., a highly capable software engineering AI.

Rules:
1. Think step by step
2. Handle ALL edge cases  
3. Write tests when appropriate
4. Never leave TODOs or placeholders
5. Output clean, production-ready code
"""
    
    def __init__(self, model: str = "qwen2.5-coder:7b"):
        """Initialize with a model."""
        self.config = OllamaConfig(model=model)
        self.client = OllamaClient(self.config)
    
    def is_ready(self) -> bool:
        """Check if Ollama is ready with the model."""
        if not self.client.is_running():
            return False
        
        models = self.client.list_models()
        return any(self.config.model in m for m in models)
    
    def setup(self) -> bool:
        """Set up Ollama with required model."""
        if not self.client.is_running():
            print("=" * 60)
            print("Ollama is not running!")
            print("")
            print("To install Ollama:")
            print("  1. Download from: https://ollama.ai/download")
            print("  2. Install and run it")
            print("  3. Or run: winget install Ollama.Ollama")
            print("")
            print("Then start it with: ollama serve")
            print("=" * 60)
            return False
        
        models = self.client.list_models()
        
        if not any(self.config.model in m for m in models):
            print(f"Downloading {self.config.model}...")
            return self.client.pull_model(self.config.model)
        
        return True
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
    ) -> str:
        """Generate response."""
        response = self.client.generate(
            prompt=prompt,
            system=self.SYSTEM_PROMPT,
            temperature=temperature,
        )
        return response.content
    
    def generate_code(
        self,
        task: str,
        context: str = "",
        language: str = "python",
    ) -> str:
        """Generate code for a task."""
        prompt = f"""Task: {task}

{f'Context:{chr(10)}{context[:2000]}' if context else ''}

Write {language} code. Return ONLY the code, no explanations.
Handle edge cases. Add error handling.
"""
        response = self.generate(prompt, temperature=0.3)
        
        # Extract code block if present
        if "```" in response:
            lines = response.split("\n")
            in_code = False
            code_lines = []
            for line in lines:
                if line.startswith("```"):
                    in_code = not in_code
                elif in_code:
                    code_lines.append(line)
            if code_lines:
                return "\n".join(code_lines)
        
        return response
    
    def analyze_bug(
        self,
        code: str,
        error: str,
    ) -> str:
        """Analyze a bug."""
        prompt = f"""Analyze this bug:

CODE:
```
{code[:2000]}
```

ERROR: {error}

Provide:
1. Root cause
2. Specific fix with code
3. Prevention tips
"""
        return self.generate(prompt, temperature=0.3)
    
    def get_callback(self) -> Callable[[str], str]:
        """Get callback for S.P.I.D.E.R. modules."""
        return self.client.get_callback()
    
    def get_stats(self) -> Dict[str, Any]:
        return self.client.get_stats()


# =============================================================================
# SETUP HELPER
# =============================================================================

def setup_ollama() -> bool:
    """
    Interactive setup helper.
    
    Usage:
        from spider.core.llm.ollama_backend import setup_ollama
        setup_ollama()
    """
    print("=" * 60)
    print("S.P.I.D.E.R. Local LLM Setup (FREE)")
    print("=" * 60)
    
    llm = SpiderLocalLLM()
    
    if not llm.client.is_running():
        print("\n[!] Ollama is not running.")
        print("\nInstall Ollama:")
        print("  Windows: winget install Ollama.Ollama")
        print("  macOS:   brew install ollama")
        print("  Linux:   curl -fsSL https://ollama.ai/install.sh | sh")
        print("\nThen run: ollama serve")
        return False
    
    print("\n[+] Ollama is running!")
    
    models = llm.client.list_models()
    print(f"\n[*] Available models: {models or 'None'}")
    
    if not llm.is_ready():
        print(f"\n[*] Downloading {llm.config.model}...")
        if llm.setup():
            print("[+] Model downloaded!")
        else:
            print("[-] Download failed")
            return False
    
    print("\n[+] Testing...")
    try:
        response = llm.generate("Say 'Hello' in one word.")
        print(f"[+] Response: {response[:50]}")
        print("\n[+] Setup complete! S.P.I.D.E.R. is ready.")
        return True
    except Exception as e:
        print(f"[-] Test failed: {e}")
        return False


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "OllamaClient",
    "OllamaConfig",
    "OllamaResponse",
    "SpiderLocalLLM",
    "setup_ollama",
]


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("[*] S.P.I.D.E.R. Local LLM (Ollama) - Demo")
    print("=" * 70)
    
    llm = SpiderLocalLLM()
    
    print("\n[1] Checking Ollama status...")
    if not llm.client.is_running():
        print("   [-] Ollama is not running!")
        print("   Install: https://ollama.ai/download")
        print("   Then run: ollama serve")
        exit(1)
    
    print("   [+] Ollama is running!")
    
    print("\n[2] Available models:")
    models = llm.client.list_models()
    for m in models:
        print(f"   - {m}")
    
    if not models:
        print("   No models found. Run: ollama pull qwen2.5-coder:7b")
        exit(1)
    
    print(f"\n[3] Using model: {llm.config.model}")
    
    if llm.is_ready():
        print("   [+] Model is ready!")
        
        print("\n[4] Testing generation...")
        try:
            code = llm.generate_code("Write a function to add two numbers")
            print(f"   Generated:\n{code[:300]}")
            print(f"\n[5] Stats: {llm.get_stats()}")
        except Exception as e:
            print(f"   [-] Error: {e}")
    else:
        print(f"   [-] Model not found. Run: ollama pull {llm.config.model}")
    
    print("\n" + "=" * 70)
    print("[*] FREE LOCAL AI - No API costs, unlimited usage!")
    print("=" * 70)
