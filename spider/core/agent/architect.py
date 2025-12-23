"""
S.P.I.D.E.R. Architect - The Creative Brain
=============================================

The Architect is responsible for generating code proposals by:
1. Querying the Bloom Filter for relevant context
2. Prompting Llama-3 (via Ollama) for code generation
3. Parsing responses into structured Proposals

This is the "Creative Brain" of the Leader Node.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests

# S.P.I.D.E.R. imports
from spider.core.dsa.bloom import CodebaseIndexer
from spider.core.distributed.protocol import Proposal

# Chameleon Engine (Adaptive Brain)
try:
    from spider.core.adaptive.config import ChameleonEngine, ChameleonConfig
    from spider.core.adaptive.projector import TaskDomain
    CHAMELEON_AVAILABLE = True
except ImportError:
    CHAMELEON_AVAILABLE = False
    ChameleonEngine = None
    ChameleonConfig = None
    TaskDomain = None


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ArchitectConfig:
    """Configuration for the Architect."""
    model_name: str = "llama3"
    ollama_url: str = "http://localhost:11434"
    timeout_seconds: int = 60
    max_retries: int = 3
    temperature: float = 0.7
    mock_mode: bool = False  # Use stub proposals when Ollama unreachable


# =============================================================================
# PROMPTS
# =============================================================================

SYSTEM_PROMPT = """You are a Principal Software Engineer. Your task is to generate a code solution.

IMPORTANT: You MUST output a valid JSON object with this EXACT structure:
{
    "file_path": "path/to/file.py",
    "diff": "the unified diff or new code",
    "reasoning": ["step 1 explanation", "step 2 explanation", "..."]
}

Rules:
1. Output ONLY the JSON object, no markdown code blocks
2. The "diff" should be valid Python code
3. The "reasoning" should explain your approach step-by-step
4. Be concise but thorough
"""

USER_PROMPT_TEMPLATE = """
Problem: {problem}

Relevant files in codebase: {file_list}

Generate a solution as a JSON object with file_path, diff, and reasoning fields.
"""


# =============================================================================
# ARCHITECT CLASS
# =============================================================================

class Architect:
    """
    The Creative Brain of S.P.I.D.E.R.
    
    Generates code proposals by:
    1. Finding relevant context via Bloom Filter
    2. Prompting LLM for code generation
    3. Parsing into structured Proposals
    
    Usage:
        architect = Architect()
        proposal = architect.draft_proposal(
            "Implement thread-safe Singleton",
            context_index=indexer
        )
    """

    def __init__(
        self,
        model_name: str = "llama3",
        ollama_url: str = "http://localhost:11434",
        mock_mode: bool = False,
        log_level: str = "INFO",
    ):
        """
        Initialize the Architect.
        
        Args:
            model_name: Ollama model to use (default: llama3).
            ollama_url: Ollama API base URL.
            mock_mode: If True, return stub proposals without calling Ollama.
            log_level: Logging level.
        """
        self.config = ArchitectConfig(
            model_name=model_name,
            ollama_url=ollama_url,
            mock_mode=mock_mode,
        )
        self._logger = self._setup_logger(log_level)
        
        # Chameleon Engine for adaptive brain switching
        self._chameleon: Optional[ChameleonEngine] = None
        if CHAMELEON_AVAILABLE:
            try:
                self._chameleon = ChameleonEngine()
                self._logger.info("🦎 Chameleon Engine initialized - adaptive mode enabled")
            except Exception as e:
                self._logger.warning(f"Chameleon Engine unavailable: {e}")
        
        # Statistics
        self._stats = {
            'proposals_generated': 0,
            'ollama_calls': 0,
            'retries': 0,
            'mock_proposals': 0,
            'parse_failures': 0,
            'chameleon_adaptations': 0,
        }

    def _setup_logger(self, level: str) -> logging.Logger:
        """Set up logging."""
        logger = logging.getLogger("Architect")
        logger.setLevel(getattr(logging, level))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s [%(name)s] %(levelname)s: %(message)s',
                datefmt='%H:%M:%S'
            ))
            logger.addHandler(handler)
        
        return logger

    def draft_proposal(
        self,
        problem_desc: str,
        context_index: Optional[CodebaseIndexer] = None,
        code_context: str = "",
    ) -> Proposal:
        """
        Generate a code proposal for the given problem.
        
        Uses Chameleon Engine for adaptive configuration based on task type.
        
        Args:
            problem_desc: Description of the problem to solve.
            context_index: Optional CodebaseIndexer for context lookup.
            code_context: Optional code snippet for better analysis.
            
        Returns:
            A Proposal object with the generated code.
        """
        self._logger.info(f"Drafting proposal for: {problem_desc[:50]}...")
        
        # Step 0: Adapt configuration using Chameleon Engine
        chameleon_config = None
        if self._chameleon:
            try:
                chameleon_config = self._chameleon.adapt(problem_desc, code_context)
                self._stats['chameleon_adaptations'] += 1
                
                # Apply adaptive configuration
                self.config.temperature = chameleon_config.solver_config.temperature
                
                self._logger.info(
                    f"🦎 Adapted: {chameleon_config.fingerprint.primary_domain.name} | "
                    f"temp={chameleon_config.solver_config.temperature:.1f} | "
                    f"persona={chameleon_config.persona}"
                )
            except Exception as e:
                self._logger.warning(f"Chameleon adaptation failed: {e}")
        
        # Step 1: Find relevant files
        relevant_files = self._find_relevant_files(problem_desc, context_index)
        self._logger.debug(f"Found {len(relevant_files)} relevant files")
        
        # Step 2: Check if we should use mock mode
        if self.config.mock_mode or not self._is_ollama_available():
            self._logger.warning("Using mock mode - generating stub proposal")
            proposal = self._generate_stub_proposal(problem_desc, relevant_files)
            self._record_chameleon_outcome(chameleon_config, True)
            return proposal
        
        # Step 3: Generate with LLM
        try:
            response = self._call_ollama(problem_desc, relevant_files, chameleon_config)
            proposal = self._parse_response(response, problem_desc)
            self._stats['proposals_generated'] += 1
            self._record_chameleon_outcome(chameleon_config, True)
            return proposal
        
        except Exception as e:
            self._logger.error(f"Failed to generate proposal: {e}")
            self._record_chameleon_outcome(chameleon_config, False)
            return self._generate_stub_proposal(problem_desc, relevant_files)
    
    def _record_chameleon_outcome(
        self,
        config: Optional['ChameleonConfig'],
        success: bool,
    ) -> None:
        """Record outcome for Chameleon learning."""
        if config and self._chameleon:
            try:
                self._chameleon.learn(config.task_id, success)
            except Exception:
                pass  # Non-critical

    def _find_relevant_files(
        self,
        problem_desc: str,
        context_index: Optional[CodebaseIndexer],
    ) -> List[str]:
        """
        Find relevant files based on keywords in the problem description.
        
        Uses simple keyword extraction and Bloom Filter lookup.
        """
        if context_index is None:
            return []
        
        # Extract keywords from problem description
        keywords = self._extract_keywords(problem_desc)
        self._logger.debug(f"Extracted keywords: {keywords}")
        
        relevant_files = []
        
        # Check each keyword against the Bloom Filter
        for keyword in keywords:
            # Check if keyword might exist in codebase
            if context_index.bloom_filter.check(keyword):
                # In a real implementation, we'd look up the actual file
                # For now, we note that the symbol exists
                pass
        
        # Return top files from the index
        if hasattr(context_index, 'file_list'):
            relevant_files = context_index.file_list[:5]
        
        return relevant_files

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract potential symbol/keyword names from text."""
        # Common programming terms to look for
        patterns = [
            r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b',  # CamelCase
            r'\b[a-z]+_[a-z_]+\b',                # snake_case
            r'\b(?:class|def|function|method|implement|create)\s+(\w+)',
        ]
        
        keywords = []
        text_lower = text.lower()
        
        # Extract CamelCase words
        for match in re.finditer(patterns[0], text):
            keywords.append(match.group())
        
        # Extract snake_case words
        for match in re.finditer(patterns[1], text):
            keywords.append(match.group())
        
        # Common keywords from the problem
        common_terms = ['singleton', 'thread', 'safe', 'pattern', 'class', 
                       'instance', 'lock', 'mutex', 'init', 'new']
        for term in common_terms:
            if term in text_lower:
                keywords.append(term)
        
        return list(set(keywords))

    def _is_ollama_available(self) -> bool:
        """Check if Ollama API is reachable."""
        try:
            response = requests.get(
                f"{self.config.ollama_url}/api/tags",
                timeout=5,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _call_ollama(
        self,
        problem_desc: str,
        relevant_files: List[str],
        chameleon_config: Optional['ChameleonConfig'] = None,
    ) -> str:
        """
        Call Ollama API to generate code.
        
        Uses Chameleon config for adaptive persona and temperature.
        Retries up to max_retries times on failure.
        """
        file_list = ", ".join(relevant_files) if relevant_files else "No specific files"
        
        # Build adaptive system prompt based on chameleon config
        if chameleon_config:
            persona = chameleon_config.persona.replace("_", " ").title()
            domain = chameleon_config.fingerprint.primary_domain.name.lower()
            system_prompt = f"""You are a {persona} specializing in {domain} development.
Your task is to generate a precise, production-ready code solution.

IMPORTANT: Output a valid JSON object with this EXACT structure:
{{
    "file_path": "path/to/file.py",
    "diff": "the unified diff or new code",
    "reasoning": ["step 1 explanation", "step 2 explanation", "..."]
}}

Rules:
1. Output ONLY the JSON object, no markdown code blocks
2. The "diff" should be valid, executable code
3. Think step-by-step in the "reasoning" array
4. Be thorough but concise
"""
        else:
            system_prompt = SYSTEM_PROMPT
        
        prompt = USER_PROMPT_TEMPLATE.format(
            problem=problem_desc,
            file_list=file_list,
        )
        
        # Use adaptive temperature from chameleon or default
        temperature = (
            chameleon_config.solver_config.temperature 
            if chameleon_config 
            else self.config.temperature
        )
        
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                self._stats['ollama_calls'] += 1
                self._logger.debug(f"Calling Ollama (attempt {attempt + 1})")
                
                response = requests.post(
                    f"{self.config.ollama_url}/api/generate",
                    json=payload,
                    timeout=self.config.timeout_seconds,
                )
                response.raise_for_status()
                
                result = response.json()
                return result.get("response", "")
                
            except requests.RequestException as e:
                last_error = e
                self._stats['retries'] += 1
                self._logger.warning(f"Retry {attempt + 1}: {e}")
                time.sleep(1)
        
        raise RuntimeError(f"Ollama call failed after {self.config.max_retries} retries: {last_error}")

    def _parse_response(self, response: str, problem_desc: str) -> Proposal:
        """
        Parse LLM response into a Proposal object.
        
        Handles markdown code blocks and invalid JSON gracefully.
        """
        # Clean markdown code blocks
        cleaned = self._clean_markdown(response)
        
        try:
            data = json.loads(cleaned)
            
            file_path = data.get("file_path", "generated.py")
            diff = data.get("diff", "")
            reasoning = data.get("reasoning", [])
            
            if isinstance(reasoning, str):
                reasoning = [reasoning]
            
            # Create unified diff format
            code_diff = f"--- a/{file_path}\n+++ b/{file_path}\n"
            for line in diff.split('\n'):
                if line and not line.startswith(('+', '-', '@')):
                    code_diff += f"+ {line}\n"
                else:
                    code_diff += f"{line}\n"
            
            return Proposal(
                code_diff=code_diff,
                merkle_root_hash="generated",
                reasoning_chain=reasoning,
            )
            
        except json.JSONDecodeError as e:
            self._stats['parse_failures'] += 1
            self._logger.warning(f"JSON parse failed: {e}")
            
            # Fallback: treat entire response as code
            return Proposal(
                code_diff=f"# Generated code\n{response}",
                merkle_root_hash="generated",
                reasoning_chain=[f"Problem: {problem_desc}", "Direct generation"],
            )

    def _clean_markdown(self, text: str) -> str:
        """Remove markdown code block wrappers from LLM output."""
        # Remove ```json ... ``` blocks
        patterns = [
            r'```json\s*(.*?)\s*```',
            r'```python\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # If no code block, try to find JSON object
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            return json_match.group()
        
        return text.strip()

    def _generate_stub_proposal(
        self,
        problem_desc: str,
        relevant_files: List[str],
    ) -> Proposal:
        """
        Generate a smart stub proposal when Ollama is unavailable.
        
        Supports common programming patterns for offline demos.
        """
        self._stats['mock_proposals'] += 1
        problem_lower = problem_desc.lower()
        
        # Pattern matching for common programming tasks
        if "singleton" in problem_lower and "thread" in problem_lower:
            code = '''import threading

class Singleton:
    """Thread-safe Singleton implementation using double-checked locking."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        pass
'''
            file_path = "singleton.py"
            reasoning = [
                "Using double-checked locking pattern for thread safety",
                "Lock ensures only one instance is created",
                "First check avoids lock overhead when instance exists",
            ]
            
        elif "fibonacci" in problem_lower:
            if "memoization" in problem_lower or "memo" in problem_lower:
                code = '''from functools import lru_cache

def fibonacci(n: int) -> int:
    """
    Calculate Fibonacci number using memoization.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Args:
        n: The position in Fibonacci sequence (0-indexed)
        
    Returns:
        The n-th Fibonacci number
        
    Examples:
        >>> fibonacci(0)
        0
        >>> fibonacci(10)
        55
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    
    @lru_cache(maxsize=None)
    def fib_memo(k: int) -> int:
        if k <= 1:
            return k
        return fib_memo(k - 1) + fib_memo(k - 2)
    
    return fib_memo(n)


# Alternative: Manual memoization with dictionary
def fibonacci_dict(n: int, memo: dict = None) -> int:
    """Fibonacci with explicit dictionary memoization."""
    if memo is None:
        memo = {}
    
    if n in memo:
        return memo[n]
    
    if n <= 1:
        return n
    
    memo[n] = fibonacci_dict(n - 1, memo) + fibonacci_dict(n - 2, memo)
    return memo[n]
'''
                file_path = "fibonacci.py"
                reasoning = [
                    "Using functools.lru_cache for automatic memoization",
                    "Time complexity reduced from O(2^n) to O(n)",
                    "Also included manual dictionary-based memoization variant",
                    "Added input validation for negative numbers",
                ]
            else:
                code = '''def fibonacci(n: int) -> int:
    """
    Calculate Fibonacci number recursively.
    
    Args:
        n: The position in Fibonacci sequence
        
    Returns:
        The n-th Fibonacci number
    """
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
'''
                file_path = "fibonacci.py"
                reasoning = [
                    "Simple recursive Fibonacci implementation",
                    "Base case: fib(0) = 0, fib(1) = 1",
                    "Note: O(2^n) time complexity without memoization",
                ]
                
        elif "binary search" in problem_lower or "binarysearch" in problem_lower:
            code = '''from typing import List, Optional

def binary_search(arr: List[int], target: int) -> int:
    """
    Perform binary search on a sorted array.
    
    Time Complexity: O(log n)
    Space Complexity: O(1)
    
    Args:
        arr: Sorted list of integers
        target: Value to search for
        
    Returns:
        Index of target if found, -1 otherwise
        
    Examples:
        >>> binary_search([1, 3, 5, 7, 9], 5)
        2
        >>> binary_search([1, 3, 5, 7, 9], 4)
        -1
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2  # Avoid overflow
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1


def binary_search_recursive(arr: List[int], target: int, 
                           left: int = 0, right: int = None) -> int:
    """Recursive binary search implementation."""
    if right is None:
        right = len(arr) - 1
        
    if left > right:
        return -1
    
    mid = left + (right - left) // 2
    
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)
'''
            file_path = "binary_search.py"
            reasoning = [
                "Iterative binary search with O(log n) time complexity",
                "Using mid = left + (right - left) // 2 to avoid integer overflow",
                "Also included recursive variant for comparison",
            ]
            
        elif "sort" in problem_lower:
            if "quick" in problem_lower:
                code = '''from typing import List

def quicksort(arr: List[int]) -> List[int]:
    """
    Quicksort implementation with Lomuto partition.
    
    Time Complexity: O(n log n) average, O(n²) worst
    Space Complexity: O(log n) for recursion stack
    """
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quicksort(left) + middle + quicksort(right)
'''
                file_path = "quicksort.py"
                reasoning = ["Quicksort with middle element as pivot", "O(n log n) average case"]
            elif "merge" in problem_lower:
                code = '''from typing import List

def merge_sort(arr: List[int]) -> List[int]:
    """
    Merge sort implementation.
    
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left: List[int], right: List[int]) -> List[int]:
    """Merge two sorted arrays."""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result
'''
                file_path = "merge_sort.py"
                reasoning = ["Merge sort with O(n log n) guaranteed", "Stable sorting algorithm"]
            else:
                code = '''from typing import List

def bubble_sort(arr: List[int]) -> List[int]:
    """Simple bubble sort implementation."""
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
'''
                file_path = "sort.py"
                reasoning = ["Basic bubble sort for educational purposes", "O(n²) time complexity"]
                
        elif "factorial" in problem_lower:
            code = '''from functools import lru_cache

def factorial(n: int) -> int:
    """
    Calculate factorial with memoization.
    
    Args:
        n: Non-negative integer
        
    Returns:
        n! (n factorial)
        
    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    
    @lru_cache(maxsize=None)
    def fact_memo(k: int) -> int:
        if k <= 1:
            return 1
        return k * fact_memo(k - 1)
    
    return fact_memo(n)
'''
            file_path = "factorial.py"
            reasoning = ["Recursive factorial with memoization", "Input validation for negative numbers"]
            
        elif "linked list" in problem_lower or "linkedlist" in problem_lower:
            code = '''from typing import Optional, Any

class ListNode:
    """Node for singly linked list."""
    def __init__(self, val: Any = 0, next: 'ListNode' = None):
        self.val = val
        self.next = next

class LinkedList:
    """Singly linked list implementation."""
    
    def __init__(self):
        self.head: Optional[ListNode] = None
    
    def append(self, val: Any) -> None:
        """Add element to end of list."""
        if not self.head:
            self.head = ListNode(val)
            return
        
        curr = self.head
        while curr.next:
            curr = curr.next
        curr.next = ListNode(val)
    
    def prepend(self, val: Any) -> None:
        """Add element to beginning of list."""
        new_node = ListNode(val, self.head)
        self.head = new_node
    
    def delete(self, val: Any) -> bool:
        """Delete first occurrence of value."""
        if not self.head:
            return False
        
        if self.head.val == val:
            self.head = self.head.next
            return True
        
        curr = self.head
        while curr.next:
            if curr.next.val == val:
                curr.next = curr.next.next
                return True
            curr = curr.next
        return False
    
    def find(self, val: Any) -> Optional[ListNode]:
        """Find node with given value."""
        curr = self.head
        while curr:
            if curr.val == val:
                return curr
            curr = curr.next
        return None
    
    def to_list(self) -> list:
        """Convert to Python list."""
        result = []
        curr = self.head
        while curr:
            result.append(curr.val)
            curr = curr.next
        return result
'''
            file_path = "linked_list.py"
            reasoning = [
                "Complete singly linked list implementation",
                "Includes append, prepend, delete, find operations",
                "O(1) prepend, O(n) append and search",
            ]
            
        elif "decorator" in problem_lower:
            code = '''import functools
import time
from typing import Callable, Any

def timer(func: Callable) -> Callable:
    """Decorator that measures execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.4f}s")
        return result
    return wrapper

def retry(max_attempts: int = 3, delay: float = 1.0):
    """Decorator that retries function on failure."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay)
        return wrapper
    return decorator

def cache(func: Callable) -> Callable:
    """Simple caching decorator."""
    cache_dict = {}
    
    @functools.wraps(func)
    def wrapper(*args) -> Any:
        if args in cache_dict:
            return cache_dict[args]
        result = func(*args)
        cache_dict[args] = result
        return result
    return wrapper
'''
            file_path = "decorators.py"
            reasoning = [
                "Common utility decorators: timer, retry, cache",
                "Uses functools.wraps to preserve function metadata",
                "Retry decorator supports configurable attempts and delay",
            ]
            
        elif "divide" in problem_lower or "division" in problem_lower:
            code = '''from typing import Optional, Union

def safe_divide(a: Union[int, float], b: Union[int, float]) -> Optional[float]:
    """
    Safely divide two numbers, handling division by zero.
    
    Args:
        a: Dividend
        b: Divisor
        
    Returns:
        Result of a/b, or None if b is zero
        
    Examples:
        >>> safe_divide(10, 2)
        5.0
        >>> safe_divide(10, 0)
        None
    """
    if b == 0:
        return None
    return a / b
'''
            file_path = "safe_divide.py"
            reasoning = [
                "Safe division with explicit zero check",
                "Returns None instead of raising ZeroDivisionError",
                "Z3 can prove this is safe: pre(b != 0) -> no exception",
            ]
        
        else:
            # Generic stub
            code = f'''"""
Solution for: {problem_desc}

Generated by S.P.I.D.E.R. Architect (Mock Mode)
"""

def solution(*args, **kwargs):
    """
    Placeholder implementation.
    
    TODO: Implement actual solution for:
    {problem_desc}
    """
    raise NotImplementedError("Replace with actual implementation")
'''
            file_path = "solution.py"
            reasoning = [
                f"Generic stub for: {problem_desc}",
                "Mock mode - Ollama unavailable",
                "Replace with actual implementation",
            ]
        
        diff = f"--- /dev/null\n+++ b/{file_path}\n"
        for line in code.strip().split('\n'):
            diff += f"+{line}\n"
        
        return Proposal(
            code_diff=diff,
            merkle_root_hash="stub",
            reasoning_chain=reasoning,
        )

    @property
    def stats(self) -> Dict[str, int]:
        """Get generation statistics."""
        return self._stats.copy()

    def print_stats(self) -> None:
        """Print generation statistics."""
        print("\n📊 Architect Statistics:")
        print(f"   Proposals generated:  {self._stats['proposals_generated']}")
        print(f"   Ollama calls:         {self._stats['ollama_calls']}")
        print(f"   Retries:              {self._stats['retries']}")
        print(f"   Mock proposals:       {self._stats['mock_proposals']}")
        print(f"   Parse failures:       {self._stats['parse_failures']}")


# =============================================================================
# DEMO
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("S.P.I.D.E.R. Architect Demo")
    print("=" * 60)
    
    # Create architect in mock mode for demo
    architect = Architect(mock_mode=True, log_level="DEBUG")
    
    # Test 1: Singleton pattern
    print("\n--- Test 1: Singleton Pattern ---")
    proposal = architect.draft_proposal(
        "Implement a thread-safe Singleton pattern in Python"
    )
    print(f"File: {proposal.code_diff.split('+++ ')[1].split()[0] if '+++ ' in proposal.code_diff else 'unknown'}")
    print(f"Reasoning: {proposal.reasoning_chain}")
    print(f"Diff preview:\n{proposal.code_diff[:500]}...")
    
    # Test 2: Generic problem
    print("\n--- Test 2: Generic Problem ---")
    proposal = architect.draft_proposal(
        "Create a binary search function"
    )
    print(f"Reasoning: {proposal.reasoning_chain}")
    
    # Print stats
    architect.print_stats()
    
    print("\n" + "=" * 60)
    print("Demo complete")
    print("=" * 60)
