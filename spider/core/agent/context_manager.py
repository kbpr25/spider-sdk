"""
S.P.I.D.E.R. Hierarchical Context Manager
==========================================

Manages context like Opus 4.5's extended thinking capability.
Enables small LLMs to punch above their weight through smart context allocation.

Context Levels:
1. IMMEDIATE: Current file, function being modified (60% tokens)
2. RELATED: Imports, callers, callees (25% tokens)  
3. SUMMARY: Codebase overview, AST structure (10% tokens)
4. ARCHIVE: Compressed history of past observations (5% tokens)

Mathematical Foundation:
- TF-IDF for chunk relevance scoring
- Sliding window with overlap for continuity
- Dynamic token budget allocation based on task complexity

Key Insight: 
Small LLMs fail not because they can't reason, but because they lack
the right context. This module ensures every token counts.
"""

import hashlib
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# CONTEXT LEVELS
# =============================================================================

class ContextLevel(Enum):
    """Hierarchical context levels with default token allocations."""
    IMMEDIATE = auto()   # Current focus: 60%
    RELATED = auto()     # Related code: 25%
    SUMMARY = auto()     # High-level overview: 10%
    ARCHIVE = auto()     # Compressed history: 5%


@dataclass
class ContextAllocation:
    """Token budget allocation across context levels."""
    immediate: float = 0.60
    related: float = 0.25
    summary: float = 0.10
    archive: float = 0.05
    
    def get_tokens(self, level: ContextLevel, total_budget: int) -> int:
        """Get token allocation for a specific level."""
        allocations = {
            ContextLevel.IMMEDIATE: self.immediate,
            ContextLevel.RELATED: self.related,
            ContextLevel.SUMMARY: self.summary,
            ContextLevel.ARCHIVE: self.archive,
        }
        return int(total_budget * allocations.get(level, 0))


# =============================================================================
# CONTEXT CHUNKS
# =============================================================================

@dataclass
class ContextChunk:
    """A chunk of context with metadata for smart retrieval."""
    content: str
    source: str  # File path or source identifier
    level: ContextLevel
    relevance_score: float = 0.0
    token_count: int = 0
    start_line: int = 0
    end_line: int = 0
    chunk_type: str = "code"  # code, docstring, comment, error, output
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.token_count == 0:
            # Rough estimate: 1 token ≈ 4 characters
            self.token_count = len(self.content) // 4 + 1
    
    def __hash__(self):
        return hash((self.source, self.start_line, self.end_line))


# =============================================================================
# TF-IDF SCORER
# =============================================================================

class TFIDFScorer:
    """
    TF-IDF based relevance scoring for context chunks.
    
    Term Frequency (TF): How often a term appears in the chunk
    Inverse Document Frequency (IDF): How rare a term is across all chunks
    
    Score = TF * IDF
    
    This helps prioritize chunks that contain rare, important terms
    (like the specific function name being debugged).
    """
    
    def __init__(self):
        self.document_frequencies: Counter = Counter()
        self.total_documents: int = 0
        self._tokenize_pattern = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b')
    
    def add_document(self, content: str) -> None:
        """Add a document to update IDF calculations."""
        tokens = set(self._tokenize(content))
        for token in tokens:
            self.document_frequencies[token] += 1
        self.total_documents += 1
    
    def _tokenize(self, content: str) -> List[str]:
        """Tokenize content into words/identifiers."""
        return self._tokenize_pattern.findall(content.lower())
    
    def compute_tf(self, content: str) -> Counter:
        """Compute term frequency for content."""
        tokens = self._tokenize(content)
        total = len(tokens) or 1
        return Counter({t: count / total for t, count in Counter(tokens).items()})
    
    def compute_idf(self, term: str) -> float:
        """Compute inverse document frequency for a term."""
        if self.total_documents == 0:
            return 1.0
        df = self.document_frequencies.get(term.lower(), 0)
        if df == 0:
            return math.log(self.total_documents + 1)
        return math.log(self.total_documents / df) + 1
    
    def score(self, content: str, query: str) -> float:
        """
        Score content relevance to query using TF-IDF.
        
        Args:
            content: The text chunk to score
            query: The query/search terms
            
        Returns:
            Relevance score (higher = more relevant)
        """
        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            return 0.0
        
        content_tf = self.compute_tf(content)
        
        score = 0.0
        for term in query_tokens:
            tf = content_tf.get(term.lower(), 0)
            idf = self.compute_idf(term)
            score += tf * idf
        
        # Normalize by query length
        return score / len(query_tokens)
    
    def batch_score(
        self, 
        chunks: List[ContextChunk], 
        query: str
    ) -> List[Tuple[ContextChunk, float]]:
        """Score multiple chunks and return sorted by relevance."""
        scored = [(chunk, self.score(chunk.content, query)) for chunk in chunks]
        return sorted(scored, key=lambda x: x[1], reverse=True)


# =============================================================================
# CODE ELEMENT EXTRACTOR
# =============================================================================

class CodeElementExtractor:
    """
    Extracts structural elements from Python code for context building.
    
    Extracts:
    - Function definitions with signatures and docstrings
    - Class definitions with methods
    - Import statements
    - Global variables
    """
    
    # Regex patterns for code elements
    FUNCTION_PATTERN = re.compile(
        r'^(\s*)def\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*[^:]+)?:(.+?)(?=\n\s*(?:def|class|\Z))',
        re.MULTILINE | re.DOTALL
    )
    CLASS_PATTERN = re.compile(
        r'^(\s*)class\s+(\w+)(?:\([^)]*\))?\s*:(.+?)(?=\n(?:class|\Z))',
        re.MULTILINE | re.DOTALL
    )
    IMPORT_PATTERN = re.compile(
        r'^(?:from\s+[\w.]+\s+)?import\s+.+$',
        re.MULTILINE
    )
    DOCSTRING_PATTERN = re.compile(
        r'^\s*["\'][\"\']["\'](.+?)["\'][\"\']["\']',
        re.DOTALL
    )
    
    def extract_functions(self, content: str, file_path: str = "") -> List[ContextChunk]:
        """Extract function definitions as context chunks."""
        chunks = []
        lines = content.split('\n')
        
        for match in self.FUNCTION_PATTERN.finditer(content):
            indent, name, args, body = match.groups()
            start_pos = match.start()
            end_pos = match.end()
            
            # Calculate line numbers
            start_line = content[:start_pos].count('\n') + 1
            end_line = content[:end_pos].count('\n') + 1
            
            # Extract docstring if present
            docstring = ""
            doc_match = self.DOCSTRING_PATTERN.match(body.strip())
            if doc_match:
                docstring = doc_match.group(1).strip()
            
            chunks.append(ContextChunk(
                content=match.group(0),
                source=file_path,
                level=ContextLevel.RELATED,
                start_line=start_line,
                end_line=end_line,
                chunk_type="function",
                metadata={
                    "name": name,
                    "args": args,
                    "docstring": docstring,
                }
            ))
        
        return chunks
    
    def extract_classes(self, content: str, file_path: str = "") -> List[ContextChunk]:
        """Extract class definitions as context chunks."""
        chunks = []
        
        for match in self.CLASS_PATTERN.finditer(content):
            indent, name, body = match.groups()
            start_pos = match.start()
            end_pos = match.end()
            
            start_line = content[:start_pos].count('\n') + 1
            end_line = content[:end_pos].count('\n') + 1
            
            # Extract method names
            methods = re.findall(r'def\s+(\w+)\s*\(', body)
            
            chunks.append(ContextChunk(
                content=match.group(0),
                source=file_path,
                level=ContextLevel.RELATED,
                start_line=start_line,
                end_line=end_line,
                chunk_type="class",
                metadata={
                    "name": name,
                    "methods": methods,
                }
            ))
        
        return chunks
    
    def extract_imports(self, content: str, file_path: str = "") -> List[ContextChunk]:
        """Extract import statements as context chunks."""
        imports = []
        for match in self.IMPORT_PATTERN.finditer(content):
            imports.append(match.group(0))
        
        if imports:
            return [ContextChunk(
                content="\n".join(imports),
                source=file_path,
                level=ContextLevel.SUMMARY,
                chunk_type="imports",
            )]
        return []
    
    def extract_all(self, content: str, file_path: str = "") -> List[ContextChunk]:
        """Extract all code elements from content."""
        chunks = []
        chunks.extend(self.extract_imports(content, file_path))
        chunks.extend(self.extract_functions(content, file_path))
        chunks.extend(self.extract_classes(content, file_path))
        return chunks


# =============================================================================
# SLIDING WINDOW CHUNKER
# =============================================================================

class SlidingWindowChunker:
    """
    Chunks code with overlapping windows for continuity.
    
    Uses sliding window with configurable:
    - Window size (in tokens or lines)
    - Overlap (for context continuity)
    - Boundary detection (avoid breaking mid-function)
    """
    
    def __init__(
        self,
        window_size: int = 500,  # tokens
        overlap: int = 100,      # tokens of overlap
        prefer_boundaries: bool = True,
    ):
        self.window_size = window_size
        self.overlap = overlap
        self.prefer_boundaries = prefer_boundaries
    
    def chunk(
        self, 
        content: str, 
        file_path: str = "",
        level: ContextLevel = ContextLevel.IMMEDIATE,
    ) -> List[ContextChunk]:
        """
        Chunk content using sliding window with overlap.
        
        Args:
            content: The text to chunk
            file_path: Source file path for metadata
            level: Context level for the chunks
            
        Returns:
            List of ContextChunks with overlap
        """
        lines = content.split('\n')
        chunks = []
        
        # Estimate tokens per line (rough: 10 tokens per line average)
        tokens_per_line = 10
        lines_per_window = max(1, self.window_size // tokens_per_line)
        overlap_lines = max(1, self.overlap // tokens_per_line)
        
        start_line = 0
        while start_line < len(lines):
            end_line = min(start_line + lines_per_window, len(lines))
            
            # Adjust to natural boundaries if enabled
            if self.prefer_boundaries and end_line < len(lines):
                end_line = self._find_boundary(lines, end_line)
            
            chunk_content = '\n'.join(lines[start_line:end_line])
            
            chunks.append(ContextChunk(
                content=chunk_content,
                source=file_path,
                level=level,
                start_line=start_line + 1,
                end_line=end_line,
            ))
            
            # Move window with overlap
            start_line = end_line - overlap_lines
            if start_line >= len(lines) - overlap_lines:
                break
        
        return chunks
    
    def _find_boundary(self, lines: List[str], target: int) -> int:
        """Find a natural code boundary near target line."""
        # Look for function/class definitions or blank lines
        search_range = 5
        
        for offset in range(search_range):
            for delta in [offset, -offset]:
                check_line = target + delta
                if 0 <= check_line < len(lines):
                    line = lines[check_line].strip()
                    if (line == "" or 
                        line.startswith("def ") or 
                        line.startswith("class ") or
                        line.startswith("#")):
                        return check_line
        
        return target


# =============================================================================
# COMPRESSION ENGINE
# =============================================================================

class ContextCompressor:
    """
    Compresses context for the ARCHIVE level.
    
    Techniques:
    1. Summary extraction: Keep only first line of functions
    2. Symbol extraction: Keep only names, not implementations
    3. Hash deduplication: Remove duplicate content
    """
    
    def __init__(self):
        self._seen_hashes: Set[str] = set()
    
    def compress(self, chunks: List[ContextChunk]) -> List[ContextChunk]:
        """Compress chunks for archive storage."""
        compressed = []
        
        for chunk in chunks:
            # Hash for deduplication
            content_hash = hashlib.md5(chunk.content.encode()).hexdigest()[:8]
            if content_hash in self._seen_hashes:
                continue
            self._seen_hashes.add(content_hash)
            
            # Compress based on type
            if chunk.chunk_type == "function":
                compressed.append(self._compress_function(chunk))
            elif chunk.chunk_type == "class":
                compressed.append(self._compress_class(chunk))
            else:
                # Keep first N tokens for other types
                compressed.append(self._truncate(chunk, max_tokens=50))
        
        return compressed
    
    def _compress_function(self, chunk: ContextChunk) -> ContextChunk:
        """Compress a function to signature + first line of docstring."""
        lines = chunk.content.split('\n')
        summary_lines = [lines[0]] if lines else []
        
        # Add docstring first line if available
        if len(lines) > 1:
            for line in lines[1:5]:
                if '"""' in line or "'''" in line:
                    summary_lines.append(line.strip())
                    break
        
        return ContextChunk(
            content='\n'.join(summary_lines),
            source=chunk.source,
            level=ContextLevel.ARCHIVE,
            start_line=chunk.start_line,
            end_line=chunk.start_line,
            chunk_type="function_summary",
            metadata=chunk.metadata,
        )
    
    def _compress_class(self, chunk: ContextChunk) -> ContextChunk:
        """Compress a class to definition + method names."""
        lines = chunk.content.split('\n')
        summary = lines[0] if lines else ""
        
        methods = chunk.metadata.get("methods", [])
        if methods:
            summary += f"\n  # Methods: {', '.join(methods[:5])}"
        
        return ContextChunk(
            content=summary,
            source=chunk.source,
            level=ContextLevel.ARCHIVE,
            start_line=chunk.start_line,
            end_line=chunk.start_line,
            chunk_type="class_summary",
            metadata=chunk.metadata,
        )
    
    def _truncate(self, chunk: ContextChunk, max_tokens: int) -> ContextChunk:
        """Truncate chunk to max tokens."""
        max_chars = max_tokens * 4
        truncated = chunk.content[:max_chars]
        if len(chunk.content) > max_chars:
            truncated += "..."
        
        return ContextChunk(
            content=truncated,
            source=chunk.source,
            level=ContextLevel.ARCHIVE,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            chunk_type=chunk.chunk_type,
            metadata=chunk.metadata,
        )


# =============================================================================
# HIERARCHICAL CONTEXT MANAGER
# =============================================================================

class HierarchicalContextManager:
    """
    The main context manager that orchestrates hierarchical context retrieval.
    
    This is what enables small LLMs to match Opus 4.5's extended thinking.
    
    Usage:
        manager = HierarchicalContextManager(token_budget=4000)
        
        # Add code files
        manager.add_file("/path/to/main.py", main_content)
        manager.add_file("/path/to/utils.py", utils_content)
        
        # Get optimized context for a query
        context = manager.get_context(
            query="Fix the bug in calculate_total function",
            focus_file="/path/to/main.py",
            focus_lines=(42, 60),
        )
    """
    
    def __init__(
        self,
        token_budget: int = 4000,
        allocation: Optional[ContextAllocation] = None,
    ):
        """
        Initialize the context manager.
        
        Args:
            token_budget: Total token budget for context
            allocation: Custom token allocation across levels
        """
        self.token_budget = token_budget
        self.allocation = allocation or ContextAllocation()
        
        # Components
        self.scorer = TFIDFScorer()
        self.extractor = CodeElementExtractor()
        self.chunker = SlidingWindowChunker()
        self.compressor = ContextCompressor()
        
        # Storage
        self.chunks: Dict[ContextLevel, List[ContextChunk]] = defaultdict(list)
        self.files: Dict[str, str] = {}  # path -> content
        
        # Statistics
        self.stats = {
            "files_indexed": 0,
            "chunks_created": 0,
            "queries_served": 0,
            "tokens_allocated": defaultdict(int),
        }
        
        logger.info(f"HierarchicalContextManager initialized with {token_budget} token budget")
    
    def add_file(
        self, 
        file_path: str, 
        content: str,
        as_immediate: bool = False,
    ) -> None:
        """
        Add a file to the context index.
        
        Args:
            file_path: Path to the file
            content: File content
            as_immediate: If True, treat as immediate context (current focus)
        """
        self.files[file_path] = content
        self.stats["files_indexed"] += 1
        
        # Add to TF-IDF scorer
        self.scorer.add_document(content)
        
        # Extract structural elements
        extracted = self.extractor.extract_all(content, file_path)
        
        # Add to appropriate levels
        level = ContextLevel.IMMEDIATE if as_immediate else ContextLevel.RELATED
        for chunk in extracted:
            chunk.level = level
            self.chunks[level].append(chunk)
            self.stats["chunks_created"] += 1
        
        # Create summary for high-level view
        summary = self._create_file_summary(file_path, content, extracted)
        self.chunks[ContextLevel.SUMMARY].append(summary)
        
        logger.debug(f"Added file {file_path}: {len(extracted)} chunks")
    
    def _create_file_summary(
        self, 
        file_path: str, 
        content: str,
        chunks: List[ContextChunk],
    ) -> ContextChunk:
        """Create a high-level summary of a file."""
        # Extract key info
        functions = [c.metadata.get("name", "") for c in chunks if c.chunk_type == "function"]
        classes = [c.metadata.get("name", "") for c in chunks if c.chunk_type == "class"]
        
        summary_parts = [f"# {Path(file_path).name}"]
        if classes:
            summary_parts.append(f"Classes: {', '.join(classes[:5])}")
        if functions:
            summary_parts.append(f"Functions: {', '.join(functions[:10])}")
        summary_parts.append(f"Lines: {len(content.splitlines())}")
        
        return ContextChunk(
            content="\n".join(summary_parts),
            source=file_path,
            level=ContextLevel.SUMMARY,
            chunk_type="file_summary",
        )
    
    def set_focus(
        self,
        file_path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> None:
        """
        Set the focus area for context retrieval.
        
        Args:
            file_path: The file being edited
            start_line: Start line of focus area
            end_line: End line of focus area
        """
        if file_path not in self.files:
            logger.warning(f"Focus file {file_path} not indexed")
            return
        
        content = self.files[file_path]
        
        # Clear previous immediate context
        self.chunks[ContextLevel.IMMEDIATE] = []
        
        # Add focus area as immediate context
        if start_line and end_line:
            lines = content.split('\n')
            focus_content = '\n'.join(lines[start_line-1:end_line])
            self.chunks[ContextLevel.IMMEDIATE].append(ContextChunk(
                content=focus_content,
                source=file_path,
                level=ContextLevel.IMMEDIATE,
                start_line=start_line,
                end_line=end_line,
                chunk_type="focus",
            ))
        
        # Add chunks from sliding window around focus
        window_chunks = self.chunker.chunk(content, file_path, ContextLevel.IMMEDIATE)
        self.chunks[ContextLevel.IMMEDIATE].extend(window_chunks)
    
    def get_context(
        self,
        query: str,
        focus_file: Optional[str] = None,
        focus_lines: Optional[Tuple[int, int]] = None,
        include_levels: Optional[List[ContextLevel]] = None,
    ) -> str:
        """
        Get optimized context for a query.
        
        Args:
            query: The query/task description
            focus_file: Optional file to focus on
            focus_lines: Optional (start, end) line numbers
            include_levels: Optional list of levels to include
            
        Returns:
            Optimized context string within token budget
        """
        self.stats["queries_served"] += 1
        
        # Set focus if specified
        if focus_file:
            self.set_focus(focus_file, 
                          focus_lines[0] if focus_lines else None,
                          focus_lines[1] if focus_lines else None)
        
        # Determine which levels to include
        levels = include_levels or list(ContextLevel)
        
        # Collect context from each level
        context_parts = []
        
        for level in levels:
            level_budget = self.allocation.get_tokens(level, self.token_budget)
            
            if level not in self.chunks or not self.chunks[level]:
                continue
            
            # Score and rank chunks by relevance
            scored = self.scorer.batch_score(self.chunks[level], query)
            
            # Select top chunks within budget
            selected = self._select_within_budget(scored, level_budget)
            
            if selected:
                level_parts = [f"# {level.name} CONTEXT"]
                for chunk in selected:
                    level_parts.append(f"## {chunk.source}:{chunk.start_line}-{chunk.end_line}")
                    level_parts.append(chunk.content)
                context_parts.append('\n'.join(level_parts))
                
                # Track allocation
                used_tokens = sum(c.token_count for c in selected)
                self.stats["tokens_allocated"][level.name] += used_tokens
        
        return '\n\n'.join(context_parts)
    
    def _select_within_budget(
        self,
        scored_chunks: List[Tuple[ContextChunk, float]],
        budget: int,
    ) -> List[ContextChunk]:
        """Select chunks within token budget, prioritizing by relevance."""
        selected = []
        used_tokens = 0
        
        for chunk, score in scored_chunks:
            if used_tokens + chunk.token_count <= budget:
                chunk.relevance_score = score
                selected.append(chunk)
                used_tokens += chunk.token_count
            
            if used_tokens >= budget * 0.95:  # Leave 5% buffer
                break
        
        return selected
    
    def add_observation(self, content: str, source: str = "observation") -> None:
        """
        Add an observation (tool output, error message, etc.) to context.
        
        These go to the archive level for reference.
        """
        chunk = ContextChunk(
            content=content[:1000],  # Limit observation size
            source=source,
            level=ContextLevel.ARCHIVE,
            chunk_type="observation",
        )
        self.chunks[ContextLevel.ARCHIVE].append(chunk)
        
        # Compress archive periodically
        if len(self.chunks[ContextLevel.ARCHIVE]) > 20:
            self.chunks[ContextLevel.ARCHIVE] = self.compressor.compress(
                self.chunks[ContextLevel.ARCHIVE]
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get context manager statistics."""
        chunk_counts = {level.name: len(chunks) 
                       for level, chunks in self.chunks.items()}
        
        return {
            **self.stats,
            "chunk_counts": chunk_counts,
            "total_chunks": sum(chunk_counts.values()),
        }
    
    def print_stats(self) -> None:
        """Print context manager statistics."""
        stats = self.get_stats()
        print("\n" + "=" * 60)
        print("HIERARCHICAL CONTEXT MANAGER STATISTICS")
        print("=" * 60)
        print(f"Files Indexed:    {stats['files_indexed']}")
        print(f"Total Chunks:     {stats['total_chunks']}")
        print(f"Queries Served:   {stats['queries_served']}")
        print("\nChunks by Level:")
        for level, count in stats['chunk_counts'].items():
            print(f"  {level}: {count}")
        print("\nToken Allocation History:")
        for level, tokens in stats['tokens_allocated'].items():
            print(f"  {level}: {tokens} tokens")
        print("=" * 60)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_context_manager(
    token_budget: int = 4000,
    **kwargs
) -> HierarchicalContextManager:
    """Factory function to create a configured context manager."""
    return HierarchicalContextManager(token_budget=token_budget, **kwargs)


def quick_context(
    files: Dict[str, str],
    query: str,
    token_budget: int = 4000,
) -> str:
    """Quick function to get context from files for a query."""
    manager = create_context_manager(token_budget)
    for path, content in files.items():
        manager.add_file(path, content)
    return manager.get_context(query)


# =============================================================================
# MAIN (DEMO)
# =============================================================================

if __name__ == "__main__":
    print("S.P.I.D.E.R. Hierarchical Context Manager Demo")
    print("=" * 50)
    
    # Sample code
    sample_code = '''
def calculate_total(items):
    """Calculate the total price of items."""
    total = 0
    for item in items:
        total += item.price
    return total

class ShoppingCart:
    """A shopping cart implementation."""
    
    def __init__(self):
        self.items = []
    
    def add_item(self, item):
        """Add an item to the cart."""
        self.items.append(item)
    
    def get_total(self):
        """Get the total price."""
        return calculate_total(self.items)
'''
    
    manager = HierarchicalContextManager(token_budget=2000)
    manager.add_file("cart.py", sample_code)
    
    context = manager.get_context("Fix calculate_total to handle None items")
    
    print("\nGenerated Context:")
    print("-" * 50)
    print(context)
    print("-" * 50)
    
    manager.print_stats()
