"""
S.P.I.D.E.R. Retro Context Engine - JIT Rolling Memory
========================================================

Born from: Assumption-11 (RETRO - DeepMind)

The Scientific Finding:
"Standard RAG is 'One-Shot Retrieval'—you fetch documents BEFORE generation.
RETRO proves this is flawed. It introduces Chunked Cross-Attention, where
the model retrieves new information every 64 tokens."

The Insight:
Intelligence is not about having all context at the start. It's about
fetching the RIGHT context EXACTLY when you need it during the thought process.
This solves the "Lost in the Middle" phenomenon.

The Solution:
We move from "Static RAG" to "JIT (Just-In-Time) Retrieval."

Flow:
1. Monitor generation buffer
2. Every 64 tokens or logical block, PAUSE
3. Retrieve fresh context based on current position
4. Inject into context window, overriding stale context

Result: The LLM never "forgets" because it constantly refreshes memory
based on what it is currently typing.
"""

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# RETRO TYPES
# =============================================================================

class TriggerType(Enum):
    """Types of retrieval triggers."""
    TOKEN_COUNT = auto()      # Every N tokens
    LOGICAL_BLOCK = auto()    # New function, class, etc.
    KEYWORD = auto()          # Specific keyword detected
    UNCERTAINTY = auto()      # Hedging language detected


@dataclass
class ContextChunk:
    """A chunk of retrieved context."""
    chunk_id: str
    content: str
    source: str = ""
    relevance_score: float = 0.0
    token_count: int = 0
    timestamp: float = field(default_factory=time.time)
    
    def __hash__(self):
        return hash(self.chunk_id)


@dataclass
class RetrievalEvent:
    """Record of a retrieval event."""
    trigger_type: TriggerType
    trigger_text: str                  # What triggered retrieval
    position: int                      # Token position
    query: str                         # What we searched for
    chunks_retrieved: List[ContextChunk]
    timestamp: float = field(default_factory=time.time)


@dataclass
class RetroState:
    """Current state of the Retro engine."""
    total_tokens: int = 0
    active_chunks: List[ContextChunk] = field(default_factory=list)
    context_window_size: int = 4096
    retrieval_history: List[RetrievalEvent] = field(default_factory=list)
    last_retrieval_position: int = 0


# =============================================================================
# LOGICAL BLOCK DETECTOR
# =============================================================================

class LogicalBlockDetector:
    """
    Detects logical block boundaries in code/text.
    
    Triggers retrieval when entering new:
    - Function definitions
    - Class definitions
    - Comment blocks
    - Import sections
    - Major structural changes
    """
    
    # Patterns for Python code
    PYTHON_PATTERNS = [
        (r'^def\s+(\w+)\s*\(', 'function_def'),
        (r'^class\s+(\w+)', 'class_def'),
        (r'^import\s+', 'import'),
        (r'^from\s+\w+\s+import', 'from_import'),
        (r'^#\s*={3,}', 'section_comment'),
        (r'^"""', 'docstring_start'),
        (r"^'''", 'docstring_start'),
        (r'^\s*@\w+', 'decorator'),
        (r'^if\s+__name__\s*==', 'main_block'),
    ]
    
    # Patterns for general text
    TEXT_PATTERNS = [
        (r'^#{1,6}\s+', 'heading'),
        (r'^\d+\.\s+', 'numbered_list'),
        (r'^-\s+', 'bullet_list'),
        (r'^>\s+', 'blockquote'),
        (r'^```', 'code_block'),
    ]
    
    def __init__(self, language: str = "python"):
        self.language = language
        self.patterns = self._get_patterns()
    
    def _get_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """Get compiled patterns for language."""
        raw_patterns = (
            self.PYTHON_PATTERNS if self.language == "python" 
            else self.TEXT_PATTERNS
        )
        return [(re.compile(p, re.MULTILINE), name) for p, name in raw_patterns]
    
    def detect(self, text: str, position: int = 0) -> Optional[Tuple[str, str, int]]:
        """
        Detect if we're at a logical block boundary.
        
        Returns:
            Tuple of (block_type, matched_text, position) or None
        """
        # Get lines after position
        lines = text[position:].split('\n')
        
        for i, line in enumerate(lines[:10]):  # Check first 10 lines
            for pattern, block_type in self.patterns:
                match = pattern.match(line)
                if match:
                    return (block_type, match.group(0), position + sum(len(l)+1 for l in lines[:i]))
        
        return None
    
    def extract_signature(self, text: str) -> str:
        """Extract function/class signature for query."""
        # Function definition
        func_match = re.search(r'def\s+(\w+)\s*\([^)]*\)', text)
        if func_match:
            return func_match.group(0)
        
        # Class definition
        class_match = re.search(r'class\s+(\w+)', text)
        if class_match:
            return class_match.group(0)
        
        # Return first meaningful line
        for line in text.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                return line[:100]
        
        return text[:100]


# =============================================================================
# CONTEXT STORE
# =============================================================================

class ContextStore:
    """
    Vector store for context chunks.
    
    In production, this would connect to:
    - FAISS
    - Pinecone
    - Weaviate
    - ChromaDB
    
    For now, uses simple keyword matching.
    """
    
    def __init__(self, embedder: Optional[Callable[[str], List[float]]] = None):
        self.chunks: Dict[str, ContextChunk] = {}
        self.embedder = embedder
        self.keyword_index: Dict[str, Set[str]] = {}  # keyword -> chunk_ids
    
    def add(self, chunk_id: str, content: str, source: str = "") -> ContextChunk:
        """Add a chunk to the store."""
        chunk = ContextChunk(
            chunk_id=chunk_id,
            content=content,
            source=source,
            token_count=len(content.split()),
        )
        self.chunks[chunk_id] = chunk
        
        # Index keywords
        words = set(content.lower().split())
        for word in words:
            if len(word) > 3:  # Skip short words
                if word not in self.keyword_index:
                    self.keyword_index[word] = set()
                self.keyword_index[word].add(chunk_id)
        
        return chunk
    
    def search(
        self,
        query: str,
        max_results: int = 5,
        min_relevance: float = 0.1,
    ) -> List[ContextChunk]:
        """
        Search for relevant chunks.
        
        Uses keyword matching (would use embeddings in production).
        """
        query_words = set(query.lower().split())
        scores: Dict[str, float] = {}
        
        # Score chunks by keyword overlap
        for word in query_words:
            if word in self.keyword_index:
                for chunk_id in self.keyword_index[word]:
                    scores[chunk_id] = scores.get(chunk_id, 0) + 1
        
        # Normalize scores
        if scores:
            max_score = max(scores.values())
            for chunk_id in scores:
                scores[chunk_id] /= max_score
        
        # Filter and sort
        results = []
        for chunk_id, score in sorted(scores.items(), key=lambda x: -x[1]):
            if score >= min_relevance:
                chunk = self.chunks[chunk_id]
                chunk.relevance_score = score
                results.append(chunk)
                if len(results) >= max_results:
                    break
        
        return results
    
    def get(self, chunk_id: str) -> Optional[ContextChunk]:
        return self.chunks.get(chunk_id)
    
    def clear(self) -> None:
        self.chunks.clear()
        self.keyword_index.clear()


# =============================================================================
# RETRO CONTEXT ENGINE
# =============================================================================

class RetroContextEngine:
    """
    The Retro-Active Context Engine - JIT Rolling Memory.
    
    Unlike static RAG that retrieves once at the start, this engine:
    1. Monitors generation in real-time
    2. Detects retrieval trigger points (every 64 tokens or logical blocks)
    3. Performs micro-retrieval of relevant context
    4. Injects fresh context, evicting stale chunks
    
    From Assumption-11 (RETRO):
    "Chunked Cross-Attention retrieves every 64 tokens."
    
    Usage:
        engine = RetroContextEngine(chunk_interval=64)
        
        # Add knowledge base
        engine.add_document("stripe_api", stripe_docs)
        engine.add_document("payment_guide", payment_guide)
        
        # Process generation with JIT retrieval
        for token in generation_stream:
            engine.feed(token)
            
            # Engine may trigger retrieval
            if engine.should_retrieve():
                fresh_context = engine.retrieve()
                # Inject into LLM context
    """
    
    def __init__(
        self,
        chunk_interval: int = 64,
        context_window_size: int = 4096,
        max_active_chunks: int = 10,
        detect_logical_blocks: bool = True,
        language: str = "python",
    ):
        """
        Initialize Retro Context Engine.
        
        Args:
            chunk_interval: Tokens between retrieval (default: 64 from RETRO)
            context_window_size: Maximum context window in tokens
            max_active_chunks: Maximum chunks in active context
            detect_logical_blocks: Trigger on function/class boundaries
            language: Code language for block detection
        """
        self.chunk_interval = chunk_interval
        self.context_window_size = context_window_size
        self.max_active_chunks = max_active_chunks
        self.detect_logical_blocks = detect_logical_blocks
        
        self.store = ContextStore()
        self.block_detector = LogicalBlockDetector(language)
        
        # State
        self.state = RetroState(context_window_size=context_window_size)
        self.generation_buffer = ""
        self.pending_retrieval = False
        self.last_trigger: Optional[Tuple[TriggerType, str]] = None
        
        self._stats = {
            "tokens_processed": 0,
            "retrievals_triggered": 0,
            "chunks_retrieved": 0,
            "logical_blocks_detected": 0,
        }
    
    def add_document(
        self,
        doc_id: str,
        content: str,
        chunk_size: int = 256,
    ) -> int:
        """
        Add a document to the knowledge base.
        
        Splits document into retrievable chunks.
        
        Returns:
            Number of chunks created
        """
        # Split into chunks by sentences/paragraphs
        chunks = self._chunk_document(content, chunk_size)
        
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            self.store.add(chunk_id, chunk_text, source=doc_id)
        
        return len(chunks)
    
    def _chunk_document(self, content: str, chunk_size: int) -> List[str]:
        """Split document into chunks."""
        chunks = []
        words = content.split()
        
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunks.append(" ".join(chunk_words))
        
        return chunks
    
    def feed(self, text: str) -> None:
        """
        Feed generated text to the engine.
        
        Monitors for retrieval triggers.
        """
        self.generation_buffer += text
        token_count = len(text.split())
        self.state.total_tokens += token_count
        self._stats["tokens_processed"] += token_count
        
        # Check for triggers
        self._check_triggers()
    
    def _check_triggers(self) -> None:
        """Check if retrieval should be triggered."""
        # Trigger 1: Token count interval
        tokens_since_retrieval = self.state.total_tokens - self.state.last_retrieval_position
        if tokens_since_retrieval >= self.chunk_interval:
            self.pending_retrieval = True
            self.last_trigger = (TriggerType.TOKEN_COUNT, f"{tokens_since_retrieval} tokens")
            return
        
        # Trigger 2: Logical block boundary
        if self.detect_logical_blocks:
            block = self.block_detector.detect(
                self.generation_buffer,
                max(0, len(self.generation_buffer) - 200)
            )
            if block:
                self.pending_retrieval = True
                self.last_trigger = (TriggerType.LOGICAL_BLOCK, block[1])
                self._stats["logical_blocks_detected"] += 1
                return
        
        # Trigger 3: Uncertainty markers
        if self._detect_uncertainty():
            self.pending_retrieval = True
            self.last_trigger = (TriggerType.UNCERTAINTY, "hedging detected")
            return
    
    def _detect_uncertainty(self) -> bool:
        """Detect uncertainty markers in recent generation."""
        recent = self.generation_buffer[-200:].lower()
        markers = ["i think", "maybe", "not sure", "possibly", "TODO", "FIXME"]
        return any(m in recent for m in markers)
    
    def should_retrieve(self) -> bool:
        """Check if retrieval is pending."""
        return self.pending_retrieval
    
    def retrieve(self, custom_query: str = None) -> List[ContextChunk]:
        """
        Perform retrieval and return fresh context.
        
        Args:
            custom_query: Optional custom query (default: uses buffer)
            
        Returns:
            List of relevant context chunks
        """
        if not self.pending_retrieval and not custom_query:
            return []
        
        self._stats["retrievals_triggered"] += 1
        self.pending_retrieval = False
        
        # Build query from recent generation
        query = custom_query or self._build_query()
        
        # Search knowledge base
        chunks = self.store.search(query, max_results=self.max_active_chunks)
        self._stats["chunks_retrieved"] += len(chunks)
        
        # Record event
        event = RetrievalEvent(
            trigger_type=self.last_trigger[0] if self.last_trigger else TriggerType.TOKEN_COUNT,
            trigger_text=self.last_trigger[1] if self.last_trigger else "",
            position=self.state.total_tokens,
            query=query,
            chunks_retrieved=chunks,
        )
        self.state.retrieval_history.append(event)
        
        # Update active chunks (evict old, add new)
        self._update_active_chunks(chunks)
        
        self.state.last_retrieval_position = self.state.total_tokens
        self.last_trigger = None
        
        return chunks
    
    def _build_query(self) -> str:
        """Build retrieval query from generation buffer."""
        # Use last 128 tokens + detected signature
        recent_text = " ".join(self.generation_buffer.split()[-128:])
        
        # Extract any signature
        signature = self.block_detector.extract_signature(recent_text)
        
        # Combine
        return f"{signature} {recent_text[-200:]}"
    
    def _update_active_chunks(self, new_chunks: List[ContextChunk]) -> None:
        """Update active context, evicting old chunks."""
        # Add new chunks
        for chunk in new_chunks:
            if chunk not in self.state.active_chunks:
                self.state.active_chunks.append(chunk)
        
        # Evict oldest if over limit
        while len(self.state.active_chunks) > self.max_active_chunks:
            # Remove oldest (lowest score)
            self.state.active_chunks.sort(key=lambda c: c.relevance_score, reverse=True)
            self.state.active_chunks.pop()
    
    def get_active_context(self) -> str:
        """Get current active context as string."""
        return "\n\n---\n\n".join(
            f"[Source: {c.source}]\n{c.content}"
            for c in self.state.active_chunks
        )
    
    def get_context_for_injection(self, max_tokens: int = 1000) -> str:
        """
        Get context formatted for injection into LLM prompt.
        
        Prioritizes by relevance and recency.
        """
        # Sort by relevance
        sorted_chunks = sorted(
            self.state.active_chunks,
            key=lambda c: c.relevance_score,
            reverse=True,
        )
        
        context_parts = []
        total_tokens = 0
        
        for chunk in sorted_chunks:
            if total_tokens + chunk.token_count > max_tokens:
                break
            context_parts.append(f"[{chunk.source}]: {chunk.content}")
            total_tokens += chunk.token_count
        
        return "\n\n".join(context_parts)
    
    def reset(self) -> None:
        """Reset generation state (keep knowledge base)."""
        self.state = RetroState(context_window_size=self.context_window_size)
        self.generation_buffer = ""
        self.pending_retrieval = False
        self.last_trigger = None
    
    def get_stats(self) -> Dict[str, int]:
        return {
            **self._stats,
            "active_chunks": len(self.state.active_chunks),
            "total_retrievals": len(self.state.retrieval_history),
        }
    
    def print_status(self) -> None:
        """Print current engine status."""
        print("\n" + "=" * 60)
        print("📚 RETRO CONTEXT ENGINE STATUS")
        print("=" * 60)
        print(f"\n📊 Generation Progress:")
        print(f"   Tokens Processed: {self.state.total_tokens}")
        print(f"   Retrievals: {len(self.state.retrieval_history)}")
        print(f"   Active Chunks: {len(self.state.active_chunks)}")
        
        if self.state.active_chunks:
            print(f"\n📎 Active Context:")
            for chunk in self.state.active_chunks[:3]:
                print(f"   [{chunk.source}] {chunk.content[:50]}... (rel: {chunk.relevance_score:.0%})")
        
        if self.state.retrieval_history:
            last = self.state.retrieval_history[-1]
            print(f"\n🔍 Last Retrieval:")
            print(f"   Trigger: {last.trigger_type.name} - {last.trigger_text}")
            print(f"   Position: {last.position} tokens")
            print(f"   Query: {last.query[:50]}...")
        
        print("=" * 60)


# =============================================================================
# STREAMING WRAPPER
# =============================================================================

class RetroStreamWrapper:
    """
    Wraps an LLM generation stream with Retro retrieval.
    
    Intercepts generation, triggers retrieval, and injects context.
    
    Usage:
        wrapper = RetroStreamWrapper(engine)
        
        for token in llm.stream("Write a Stripe integration"):
            # Wrapper may augment with fresh context
            augmented = wrapper.process(token)
            yield augmented
    """
    
    def __init__(
        self,
        engine: RetroContextEngine,
        context_injector: Optional[Callable[[str], None]] = None,
    ):
        self.engine = engine
        self.context_injector = context_injector
    
    def process(self, token: str) -> str:
        """
        Process a generated token.
        
        May trigger retrieval and context injection.
        """
        self.engine.feed(token)
        
        if self.engine.should_retrieve():
            chunks = self.engine.retrieve()
            
            if chunks and self.context_injector:
                fresh_context = self.engine.get_context_for_injection()
                self.context_injector(fresh_context)
                
                logger.info(
                    f"RETRO: Injected {len(chunks)} chunks at position "
                    f"{self.engine.state.total_tokens}"
                )
        
        return token
    
    def process_batch(self, text: str) -> Tuple[str, bool]:
        """
        Process a batch of text.
        
        Returns:
            Tuple of (text, retrieval_occurred)
        """
        self.engine.feed(text)
        
        if self.engine.should_retrieve():
            chunks = self.engine.retrieve()
            return (text, len(chunks) > 0)
        
        return (text, False)


# =============================================================================
# PACKAGE INIT
# =============================================================================

__all__ = [
    "RetroContextEngine",
    "RetroStreamWrapper",
    "ContextStore",
    "ContextChunk",
    "RetrievalEvent",
    "TriggerType",
    "LogicalBlockDetector",
]


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("📚 S.P.I.D.E.R. Retro Context Engine - Demo")
    print("=" * 70)
    
    engine = RetroContextEngine(chunk_interval=64)
    
    # Add knowledge base
    engine.add_document("stripe_api", """
    The Stripe API allows you to create charges, customers, and subscriptions.
    To create a charge, use stripe.Charge.create() with amount, currency, source.
    Always include idempotency_key for safe retries.
    The amount is in cents (e.g., 1000 = $10.00).
    """)
    
    engine.add_document("python_best_practices", """
    Always use try-except blocks for API calls.
    Handle rate limiting with exponential backoff.
    Use type hints for better code documentation.
    Never hardcode API keys in source code.
    """)
    
    # Simulate generation
    generation = """
    import stripe
    
    def create_payment(amount, customer_id):
        # Create a charge using Stripe API
        charge = stripe.Charge.create(
            amount=amount,
            currency='usd',
            customer=customer_id,
    """
    
    print("\n📝 Simulating generation...")
    
    # Feed generation in chunks (simulating streaming)
    words = generation.split()
    for i in range(0, len(words), 10):
        chunk = " ".join(words[i:i+10]) + " "
        engine.feed(chunk)
        
        if engine.should_retrieve():
            print(f"\n🔍 Retrieval triggered at token {engine.state.total_tokens}")
            chunks = engine.retrieve()
            print(f"   Retrieved {len(chunks)} chunks")
            for chunk in chunks[:2]:
                print(f"   - [{chunk.source}]: {chunk.content[:60]}...")
    
    engine.print_status()
    print(f"\n📊 Stats: {engine.get_stats()}")
