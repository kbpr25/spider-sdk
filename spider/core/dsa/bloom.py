"""
ContextBloom Indexer for S.P.I.D.E.R. SDK.

A lightweight, probabilistic data structure for rapid membership testing
of code symbols (functions, classes, variables) across a Python codebase.

Uses only Python Standard Library - no external dependencies.
"""

import ast
import hashlib
import json
import math
import os
import pickle
from dataclasses import dataclass, field
from typing import Iterator, Optional, Set, Tuple, Union


class BloomFilter:
    """
    A space-efficient probabilistic data structure for membership testing.
    
    Uses double hashing technique with SHA-256 and MD5 to generate k hash
    indices without external dependencies like mmh3.
    
    Attributes:
        size: Number of bits in the filter.
        hash_count: Number of hash functions (k).
        bit_array: The underlying bit storage as an integer.
    """

    def __init__(
        self,
        size: Optional[int] = None,
        hash_count: Optional[int] = None,
        expected_items: int = 10000,
        false_positive_rate: float = 0.01
    ):
        """
        Initialize the Bloom Filter.
        
        Args:
            size: Explicit bit array size. If None, calculated optimally.
            hash_count: Explicit number of hash functions. If None, calculated optimally.
            expected_items: Expected number of items to be inserted.
            false_positive_rate: Desired false positive probability (default 1%).
        """
        if size is None:
            size = self._optimal_size(expected_items, false_positive_rate)
        if hash_count is None:
            hash_count = self._optimal_hash_count(size, expected_items)
        
        self.size = size
        self.hash_count = hash_count
        self.bit_array = 0  # Use int for arbitrary-precision bit manipulation
        self._item_count = 0

    @staticmethod
    def _optimal_size(n: int, p: float) -> int:
        """
        Calculate optimal bit array size.
        
        Formula: m = -(n * ln(p)) / (ln(2)^2)
        
        Args:
            n: Expected number of items.
            p: Desired false positive rate.
            
        Returns:
            Optimal size in bits.
        """
        if p <= 0 or p >= 1:
            raise ValueError("False positive rate must be between 0 and 1 (exclusive)")
        if n <= 0:
            raise ValueError("Expected items must be positive")
        
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return max(int(math.ceil(m)), 64)  # Minimum 64 bits

    @staticmethod
    def _optimal_hash_count(m: int, n: int) -> int:
        """
        Calculate optimal number of hash functions.
        
        Formula: k = (m/n) * ln(2)
        
        Args:
            m: Bit array size.
            n: Expected number of items.
            
        Returns:
            Optimal number of hash functions.
        """
        if n <= 0:
            return 1
        k = (m / n) * math.log(2)
        return max(int(math.ceil(k)), 1)  # Minimum 1 hash

    def _get_hash_indices(self, item: str) -> Iterator[int]:
        """
        Generate k hash indices using double hashing technique.
        
        Uses: h(i) = (h1 + i * h2) mod m
        Where h1 = SHA-256, h2 = MD5
        
        Args:
            item: The string item to hash.
            
        Yields:
            k distinct indices in range [0, size).
        """
        item_bytes = item.encode('utf-8')
        
        # Primary hash: SHA-256
        h1 = int(hashlib.sha256(item_bytes).hexdigest(), 16)
        
        # Secondary hash: MD5
        h2 = int(hashlib.md5(item_bytes).hexdigest(), 16)
        
        # Ensure h2 is odd for better distribution (to avoid cycles)
        if h2 % 2 == 0:
            h2 += 1
        
        for i in range(self.hash_count):
            yield (h1 + i * h2) % self.size

    def add(self, item: str) -> None:
        """
        Add an item to the Bloom Filter.
        
        Args:
            item: The string item to add.
        """
        for index in self._get_hash_indices(item):
            self.bit_array |= (1 << index)
        self._item_count += 1

    def check(self, item: str) -> bool:
        """
        Check if an item might be in the Bloom Filter.
        
        Args:
            item: The string item to check.
            
        Returns:
            True if the item might be present (possible false positive).
            False if the item is definitely not present.
        """
        for index in self._get_hash_indices(item):
            if not (self.bit_array & (1 << index)):
                return False
        return True

    def __contains__(self, item: str) -> bool:
        """Enable `in` operator: `if 'func_name' in bloom_filter:`"""
        return self.check(item)

    def __len__(self) -> int:
        """Return the number of items added."""
        return self._item_count

    @property
    def estimated_false_positive_rate(self) -> float:
        """
        Calculate the current estimated false positive rate.
        
        Formula: (1 - e^(-kn/m))^k
        """
        if self._item_count == 0:
            return 0.0
        exponent = -self.hash_count * self._item_count / self.size
        return (1 - math.exp(exponent)) ** self.hash_count

    @property
    def fill_ratio(self) -> float:
        """Return the ratio of set bits to total bits."""
        if self.size == 0:
            return 0.0
        set_bits = bin(self.bit_array).count('1')
        return set_bits / self.size

    def union(self, other: 'BloomFilter') -> 'BloomFilter':
        """
        Create a new BloomFilter that is the union of this and another.
        
        Args:
            other: Another BloomFilter with the same size and hash_count.
            
        Returns:
            A new BloomFilter containing items from both.
            
        Raises:
            ValueError: If filters have different parameters.
        """
        if self.size != other.size or self.hash_count != other.hash_count:
            raise ValueError("Cannot union BloomFilters with different parameters")
        
        result = BloomFilter(size=self.size, hash_count=self.hash_count)
        result.bit_array = self.bit_array | other.bit_array
        result._item_count = self._item_count + other._item_count  # Approximate
        return result

    def intersection(self, other: 'BloomFilter') -> 'BloomFilter':
        """
        Create a new BloomFilter that is the intersection of this and another.
        
        Args:
            other: Another BloomFilter with the same size and hash_count.
            
        Returns:
            A new BloomFilter containing items possibly in both.
        """
        if self.size != other.size or self.hash_count != other.hash_count:
            raise ValueError("Cannot intersect BloomFilters with different parameters")
        
        result = BloomFilter(size=self.size, hash_count=self.hash_count)
        result.bit_array = self.bit_array & other.bit_array
        result._item_count = min(self._item_count, other._item_count)  # Approximate
        return result

    def clear(self) -> None:
        """Reset the filter to empty state."""
        self.bit_array = 0
        self._item_count = 0

    def copy(self) -> 'BloomFilter':
        """Create a copy of this BloomFilter."""
        result = BloomFilter(size=self.size, hash_count=self.hash_count)
        result.bit_array = self.bit_array
        result._item_count = self._item_count
        return result

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_bytes(self) -> bytes:
        """
        Serialize the BloomFilter to bytes.
        
        Returns:
            Bytes representation of the filter.
        """
        # Calculate minimum bytes needed
        byte_count = (self.size + 7) // 8
        bit_bytes = self.bit_array.to_bytes(byte_count, byteorder='big')
        
        # Pack metadata + bit array
        metadata = {
            'size': self.size,
            'hash_count': self.hash_count,
            'item_count': self._item_count,
            'version': 1
        }
        return pickle.dumps((metadata, bit_bytes))

    @classmethod
    def from_bytes(cls, data: bytes) -> 'BloomFilter':
        """
        Deserialize a BloomFilter from bytes.
        
        Args:
            data: Bytes produced by to_bytes().
            
        Returns:
            A reconstructed BloomFilter.
        """
        metadata, bit_bytes = pickle.loads(data)
        
        bf = cls(size=metadata['size'], hash_count=metadata['hash_count'])
        bf.bit_array = int.from_bytes(bit_bytes, byteorder='big')
        bf._item_count = metadata['item_count']
        return bf

    def save(self, filepath: str) -> None:
        """
        Save the BloomFilter to a file.
        
        Args:
            filepath: Path to save the filter.
        """
        with open(filepath, 'wb') as f:
            f.write(self.to_bytes())

    @classmethod
    def load(cls, filepath: str) -> 'BloomFilter':
        """
        Load a BloomFilter from a file.
        
        Args:
            filepath: Path to the saved filter.
            
        Returns:
            The loaded BloomFilter.
        """
        with open(filepath, 'rb') as f:
            return cls.from_bytes(f.read())

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            'size': self.size,
            'hash_count': self.hash_count,
            'item_count': self._item_count,
            'bit_array_hex': hex(self.bit_array),
            'version': 1
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'BloomFilter':
        """Reconstruct from a dictionary."""
        bf = cls(size=data['size'], hash_count=data['hash_count'])
        bf.bit_array = int(data['bit_array_hex'], 16)
        bf._item_count = data['item_count']
        return bf

    def __repr__(self) -> str:
        return (
            f"BloomFilter(size={self.size}, hash_count={self.hash_count}, "
            f"items={self._item_count}, fill={self.fill_ratio:.2%}, "
            f"est_fpr={self.estimated_false_positive_rate:.4f})"
        )


@dataclass
class IndexStats:
    """Statistics about the codebase index."""
    total_files: int = 0
    total_functions: int = 0
    total_classes: int = 0
    total_variables: int = 0
    total_methods: int = 0
    parse_errors: int = 0
    
    @property
    def total_symbols(self) -> int:
        return self.total_functions + self.total_classes + self.total_variables + self.total_methods


class SymbolExtractor(ast.NodeVisitor):
    """AST visitor that extracts function, class, and variable names."""

    def __init__(self):
        self.functions: Set[str] = set()
        self.classes: Set[str] = set()
        self.variables: Set[str] = set()
        self.methods: Set[str] = set()
        self._current_class: Optional[str] = None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self._current_class:
            # It's a method
            self.methods.add(f"{self._current_class}.{node.name}")
            self.methods.add(node.name)  # Also add bare method name
        else:
            self.functions.add(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if self._current_class:
            self.methods.add(f"{self._current_class}.{node.name}")
            self.methods.add(node.name)
        else:
            self.functions.add(node.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.classes.add(node.name)
        # Track class context for methods
        old_class = self._current_class
        self._current_class = node.name
        self.generic_visit(node)
        self._current_class = old_class

    def visit_Assign(self, node: ast.Assign) -> None:
        # Only capture module-level assignments
        if self._current_class is None:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    # Skip private/dunder names
                    if not target.id.startswith('_'):
                        self.variables.add(target.id)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # Annotated assignment: `x: int = 5`
        if self._current_class is None:
            if isinstance(node.target, ast.Name):
                if not node.target.id.startswith('_'):
                    self.variables.add(node.target.id)
        self.generic_visit(node)


class CodebaseIndexer:
    """
    Indexes a Python codebase into a BloomFilter for rapid symbol lookup.
    
    Extracts function names, class names, method names, and global variables
    from all .py files in the specified directory.
    """

    def __init__(
        self,
        root_path: str,
        expected_symbols: int = 10000,
        false_positive_rate: float = 0.01
    ):
        """
        Initialize the CodebaseIndexer.
        
        Args:
            root_path: Root directory to index.
            expected_symbols: Expected number of symbols (for BloomFilter sizing).
            false_positive_rate: Desired false positive rate.
        """
        self.root_path = os.path.abspath(root_path)
        self.expected_symbols = expected_symbols
        self.false_positive_rate = false_positive_rate
        
        self._bloom: Optional[BloomFilter] = None
        self._stats = IndexStats()
        self._indexed = False
        
        # Keep track of all symbols for debugging/inspection
        self._all_symbols: Set[str] = set()

    def index(self) -> 'CodebaseIndexer':
        """
        Build the index by walking the codebase.
        
        Returns:
            self (for method chaining).
            
        Raises:
            FileNotFoundError: If root_path doesn't exist.
            NotADirectoryError: If root_path is not a directory.
        """
        if not os.path.exists(self.root_path):
            raise FileNotFoundError(f"Path does not exist: {self.root_path}")
        if not os.path.isdir(self.root_path):
            raise NotADirectoryError(f"Path is not a directory: {self.root_path}")

        # Collect all symbols first
        symbols: Set[str] = set()
        self._stats = IndexStats()

        for dirpath, dirnames, filenames in os.walk(self.root_path):
            # Skip hidden and cache directories
            dirnames[:] = [d for d in dirnames if not d.startswith('.') and d != '__pycache__']
            
            for filename in filenames:
                if not filename.endswith('.py'):
                    continue
                
                filepath = os.path.join(dirpath, filename)
                file_symbols = self._extract_symbols_from_file(filepath)
                symbols.update(file_symbols)

        # Create optimally-sized BloomFilter
        if symbols:
            self._bloom = BloomFilter(
                expected_items=max(len(symbols), self.expected_symbols),
                false_positive_rate=self.false_positive_rate
            )
            for symbol in symbols:
                self._bloom.add(symbol)
        else:
            self._bloom = BloomFilter(
                expected_items=self.expected_symbols,
                false_positive_rate=self.false_positive_rate
            )

        self._all_symbols = symbols
        self._indexed = True
        return self

    def _extract_symbols_from_file(self, filepath: str) -> Set[str]:
        """
        Extract symbols from a single Python file.
        
        Args:
            filepath: Path to the Python file.
            
        Returns:
            Set of symbol names found in the file.
        """
        self._stats.total_files += 1
        symbols: Set[str] = set()

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()
        except (IOError, UnicodeDecodeError):
            self._stats.parse_errors += 1
            return symbols

        try:
            tree = ast.parse(source, filename=filepath)
        except SyntaxError:
            self._stats.parse_errors += 1
            return symbols

        extractor = SymbolExtractor()
        extractor.visit(tree)

        # Update stats
        self._stats.total_functions += len(extractor.functions)
        self._stats.total_classes += len(extractor.classes)
        self._stats.total_variables += len(extractor.variables)
        self._stats.total_methods += len(extractor.methods)

        # Collect all symbols
        symbols.update(extractor.functions)
        symbols.update(extractor.classes)
        symbols.update(extractor.variables)
        symbols.update(extractor.methods)

        # Also add file-qualified names
        relative_path = os.path.relpath(filepath, self.root_path)
        module_path = relative_path.replace(os.sep, '.').replace('.py', '')
        
        for func in extractor.functions:
            symbols.add(f"{module_path}.{func}")
        for cls in extractor.classes:
            symbols.add(f"{module_path}.{cls}")

        return symbols

    def __contains__(self, symbol: str) -> bool:
        """
        Check if a symbol might exist in the codebase.
        
        Args:
            symbol: The symbol name to check.
            
        Returns:
            True if the symbol might exist, False if definitely not.
        """
        if not self._indexed or self._bloom is None:
            raise RuntimeError("Index not built. Call index() first.")
        return symbol in self._bloom

    def check(self, symbol: str) -> bool:
        """Alias for __contains__ for explicit API."""
        return symbol in self

    @property
    def bloom_filter(self) -> Optional[BloomFilter]:
        """Access the underlying BloomFilter."""
        return self._bloom

    @property
    def stats(self) -> IndexStats:
        """Get indexing statistics."""
        return self._stats

    @property
    def symbols(self) -> Set[str]:
        """Get all indexed symbols (for debugging)."""
        return self._all_symbols.copy()

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def save(self, filepath: str) -> None:
        """
        Save the index to disk.
        
        Args:
            filepath: Path to save the index.
        """
        if not self._indexed or self._bloom is None:
            raise RuntimeError("Index not built. Call index() first.")

        data = {
            'root_path': self.root_path,
            'expected_symbols': self.expected_symbols,
            'false_positive_rate': self.false_positive_rate,
            'bloom': self._bloom.to_dict(),
            'stats': {
                'total_files': self._stats.total_files,
                'total_functions': self._stats.total_functions,
                'total_classes': self._stats.total_classes,
                'total_variables': self._stats.total_variables,
                'total_methods': self._stats.total_methods,
                'parse_errors': self._stats.parse_errors,
            },
            'version': 1
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'CodebaseIndexer':
        """
        Load an index from disk.
        
        Args:
            filepath: Path to the saved index.
            
        Returns:
            A reconstructed CodebaseIndexer.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        indexer = cls(
            root_path=data['root_path'],
            expected_symbols=data['expected_symbols'],
            false_positive_rate=data['false_positive_rate']
        )

        indexer._bloom = BloomFilter.from_dict(data['bloom'])
        indexer._stats = IndexStats(
            total_files=data['stats']['total_files'],
            total_functions=data['stats']['total_functions'],
            total_classes=data['stats']['total_classes'],
            total_variables=data['stats']['total_variables'],
            total_methods=data['stats']['total_methods'],
            parse_errors=data['stats']['parse_errors'],
        )
        indexer._indexed = True

        return indexer

    def save_binary(self, filepath: str) -> None:
        """Save in compact binary format using pickle."""
        if not self._indexed or self._bloom is None:
            raise RuntimeError("Index not built. Call index() first.")

        data = {
            'root_path': self.root_path,
            'bloom_bytes': self._bloom.to_bytes(),
            'stats': self._stats,
            'version': 1
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load_binary(cls, filepath: str) -> 'CodebaseIndexer':
        """Load from compact binary format."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        indexer = cls(root_path=data['root_path'])
        indexer._bloom = BloomFilter.from_bytes(data['bloom_bytes'])
        indexer._stats = data['stats']
        indexer._indexed = True

        return indexer

    def __repr__(self) -> str:
        if not self._indexed:
            return f"CodebaseIndexer(root={self.root_path!r}, indexed=False)"
        return (
            f"CodebaseIndexer(root={os.path.basename(self.root_path)!r}, "
            f"files={self._stats.total_files}, "
            f"symbols={self._stats.total_symbols}, "
            f"bloom={self._bloom})"
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_index(root_path: str, **kwargs) -> CodebaseIndexer:
    """
    Convenience function to create and build an index.
    
    Args:
        root_path: Root directory to index.
        **kwargs: Additional arguments for CodebaseIndexer.
        
    Returns:
        A built CodebaseIndexer.
    """
    return CodebaseIndexer(root_path, **kwargs).index()


def quick_check(root_path: str, symbol: str) -> bool:
    """
    Quickly check if a symbol exists in a codebase.
    
    Note: This rebuilds the index each time. For repeated lookups,
    use CodebaseIndexer.index() and reuse the instance.
    
    Args:
        root_path: Root directory.
        symbol: Symbol to check.
        
    Returns:
        True if symbol might exist, False if definitely not.
    """
    indexer = create_index(root_path)
    return symbol in indexer


# =============================================================================
# DEMO
# =============================================================================

if __name__ == '__main__':
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()

    print("=" * 60)
    print("ContextBloom Indexer - S.P.I.D.E.R. SDK")
    print("=" * 60)
    print(f"\nIndexing: {target}\n")

    indexer = CodebaseIndexer(target).index()

    print(f"Index Statistics:")
    print(f"  - Files scanned:    {indexer.stats.total_files}")
    print(f"  - Functions found:  {indexer.stats.total_functions}")
    print(f"  - Classes found:    {indexer.stats.total_classes}")
    print(f"  - Methods found:    {indexer.stats.total_methods}")
    print(f"  - Variables found:  {indexer.stats.total_variables}")
    print(f"  - Parse errors:     {indexer.stats.parse_errors}")
    print(f"  - Total symbols:    {indexer.stats.total_symbols}")

    print(f"\nBloom Filter Stats:")
    print(f"  - Size (bits):      {indexer.bloom_filter.size:,}")
    print(f"  - Hash functions:   {indexer.bloom_filter.hash_count}")
    print(f"  - Fill ratio:       {indexer.bloom_filter.fill_ratio:.2%}")
    print(f"  - Est. FPR:         {indexer.bloom_filter.estimated_false_positive_rate:.6f}")

    print("\n--- Symbol Lookup Demo ---")
    test_symbols = ['BloomFilter', 'CodebaseIndexer', 'add', 'check', 'nonexistent_xyz']
    for sym in test_symbols:
        result = "✓ FOUND" if sym in indexer else "✗ NOT FOUND"
        print(f"  '{sym}': {result}")

    print("\n--- Sample Indexed Symbols ---")
    sample = list(indexer.symbols)[:10]
    for sym in sorted(sample):
        print(f"  - {sym}")

    print("=" * 60)
