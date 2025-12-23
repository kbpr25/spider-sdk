"""
S.P.I.D.E.R. Sleep Test Dataset
================================

50 Synthetic Engineering Tasks that stress-test all layers:
- Logic & Algorithms (Layer 4: Memory)
- Security & Safety (Layer 3: Shield/Z3)
- Concurrency & SRE (Layer 2: Immune System + Layer 1: Hive Mind)

Each task is designed to trigger specific S.P.I.D.E.R. components.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any


class Difficulty(Enum):
    """Task difficulty levels."""
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"
    EXPERT = "Expert"


class Layer(Enum):
    """S.P.I.D.E.R. layer being tested."""
    MEMORY = "DSA (Bloom/Merkle)"
    SHIELD = "Z3 Verification"
    IMMUNE = "Phi Failure Detection"
    HIVEMIND = "Distributed Consensus"
    ALL = "Full Stack"


@dataclass
class Task:
    """A synthetic engineering task."""
    id: str
    problem: str
    difficulty: Difficulty
    layer: Layer
    repo_url: str = "https://github.com/TheAlgorithms/Python.git"
    expected_patterns: List[str] = None  # Keywords expected in solution
    
    def __post_init__(self):
        if self.expected_patterns is None:
            self.expected_patterns = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "problem": self.problem,
            "difficulty": self.difficulty.value,
            "layer": self.layer.value,
            "repo_url": self.repo_url,
            "expected_patterns": self.expected_patterns,
        }


def get_sleep_test_dataset() -> List[Dict[str, Any]]:
    """
    Returns 50 distinct engineering challenges varying in complexity.
    
    Distribution:
    - 20 Algorithm tasks (Layer 4: Memory)
    - 15 Security tasks (Layer 3: Shield)
    - 15 Concurrency tasks (Layer 1+2: Hive + Immune)
    
    Returns:
        List of task dictionaries for batch processing.
    """
    tasks = []
    
    # =========================================================================
    # LAYER 4: MEMORY (DSA) - 20 Tasks
    # These test the Bloom Filter and Merkle Tree integration
    # =========================================================================
    
    algorithms = [
        ("Binary Search", ["binary", "search", "left", "right", "mid"]),
        ("QuickSort", ["pivot", "partition", "recursive"]),
        ("MergeSort", ["merge", "divide", "conquer"]),
        ("Dijkstra", ["graph", "distance", "shortest", "path"]),
        ("A* Search", ["heuristic", "open", "closed", "path"]),
        ("Dynamic Knapsack", ["dp", "weight", "value", "capacity"]),
        ("Levenshtein Distance", ["edit", "distance", "insert", "delete"]),
        ("Trie Implementation", ["trie", "node", "insert", "search"]),
        ("LRU Cache", ["cache", "lru", "capacity", "get", "put"]),
        ("Heap Implementation", ["heap", "heapify", "parent", "child"]),
        ("Graph BFS", ["bfs", "queue", "visited", "neighbor"]),
        ("Graph DFS", ["dfs", "stack", "visited", "recursive"]),
        ("Red-Black Tree", ["tree", "node", "rotate", "color"]),
        ("AVL Tree", ["avl", "balance", "height", "rotate"]),
        ("Segment Tree", ["segment", "range", "query", "update"]),
        ("Fenwick Tree", ["fenwick", "prefix", "sum", "update"]),
        ("Floyd-Warshall", ["floyd", "distance", "path", "matrix"]),
        ("Bellman-Ford", ["bellman", "relax", "negative", "cycle"]),
        ("Kruskal MST", ["kruskal", "mst", "union", "find"]),
        ("Prim MST", ["prim", "mst", "priority", "queue"]),
    ]
    
    for i, (algo, patterns) in enumerate(algorithms):
        tasks.append(Task(
            id=f"ALGO-{i+1:03d}",
            problem=f"Implement {algo} with detailed comments, type hints, and O(n) complexity analysis.",
            difficulty=Difficulty.MEDIUM if i < 10 else Difficulty.HARD,
            layer=Layer.MEMORY,
            expected_patterns=patterns,
        ))
    
    # =========================================================================
    # LAYER 3: SHIELD (Z3 Verification) - 15 Tasks
    # These trigger Z3's formal verification capabilities
    # =========================================================================
    
    security_tasks = [
        ("safe division function", ["if", "zero", "none", "return"], "Must handle division by zero"),
        ("SQL injection sanitizer", ["escape", "quote", "sanitize"], "Must prevent injection attacks"),
        ("XSS sanitizer function", ["escape", "html", "script", "sanitize"], "Must escape HTML entities"),
        ("path traversal prevention", ["path", "normalize", "check", "safe"], "Must prevent directory traversal"),
        ("rate limiter with token bucket", ["token", "bucket", "rate", "limit"], "Must implement token bucket algorithm"),
        ("input validator", ["validate", "type", "check", "error"], "Must validate all input types"),
        ("bounds checker for arrays", ["bounds", "index", "check", "raise"], "Must prevent buffer overflow"),
        ("null-safe getter", ["none", "default", "get", "safe"], "Must handle null values"),
        ("integer overflow checker", ["overflow", "max", "min", "check"], "Must detect integer overflow"),
        ("safe type coercion", ["type", "convert", "safe", "error"], "Must handle type conversion safely"),
        ("memory-safe buffer", ["buffer", "size", "check", "bounds"], "Must prevent buffer overread"),
        ("safe file reader", ["file", "exists", "check", "error"], "Must handle file errors gracefully"),
        ("secure random generator", ["random", "secure", "crypto", "seed"], "Must use cryptographic randomness"),
        ("safe JSON parser", ["json", "parse", "error", "safe"], "Must handle malformed JSON"),
        ("authenticated wrapper", ["auth", "token", "verify", "secure"], "Must verify authentication"),
    ]
    
    for i, (task, patterns, desc) in enumerate(security_tasks):
        tasks.append(Task(
            id=f"SEC-{i+1:03d}",
            problem=f"Implement a {task} that passes formal verification. {desc}.",
            difficulty=Difficulty.HARD,
            layer=Layer.SHIELD,
            expected_patterns=patterns,
        ))
    
    # =========================================================================
    # LAYER 1+2: HIVEMIND + IMMUNE (Consensus + SRE) - 15 Tasks
    # These trigger distributed consensus and failure detection
    # =========================================================================
    
    concurrency_tasks = [
        ("thread-safe Singleton", ["lock", "instance", "thread", "safe"], "Double-checked locking pattern"),
        ("producer-consumer queue", ["queue", "producer", "consumer", "lock"], "Thread-safe bounded queue"),
        ("read-write lock", ["read", "write", "lock", "acquire"], "Multiple readers, single writer"),
        ("async HTTP crawler", ["async", "await", "fetch", "gather"], "Concurrent web requests"),
        ("retry decorator with exponential backoff", ["retry", "backoff", "attempt", "delay"], "Configurable retry logic"),
        ("connection pool manager", ["pool", "acquire", "release", "limit"], "Thread-safe connection pooling"),
        ("circuit breaker pattern", ["circuit", "breaker", "open", "closed"], "Fault tolerance pattern"),
        ("semaphore implementation", ["semaphore", "acquire", "release", "count"], "Counting semaphore"),
        ("barrier synchronization", ["barrier", "wait", "parties", "sync"], "Thread synchronization"),
        ("future/promise implementation", ["future", "promise", "result", "callback"], "Async result container"),
        ("actor model skeleton", ["actor", "message", "receive", "send"], "Message-passing concurrency"),
        ("leader election algorithm", ["leader", "election", "vote", "term"], "Distributed leader election"),
        ("heartbeat monitor", ["heartbeat", "interval", "timeout", "alive"], "Node health monitoring"),
        ("distributed lock", ["lock", "acquire", "release", "distributed"], "Cross-process locking"),
        ("event-driven state machine", ["state", "event", "transition", "machine"], "FSM implementation"),
    ]
    
    for i, (task, patterns, desc) in enumerate(concurrency_tasks):
        tasks.append(Task(
            id=f"SRE-{i+1:03d}",
            problem=f"Implement a {task} handling all edge cases. {desc}.",
            difficulty=Difficulty.HARD if i < 10 else Difficulty.EXPERT,
            layer=Layer.HIVEMIND if i < 10 else Layer.IMMUNE,
            expected_patterns=patterns,
        ))
    
    # Convert to dictionaries for batch processing
    return [task.to_dict() for task in tasks]


def get_quick_test_dataset() -> List[Dict[str, Any]]:
    """
    Returns 5 quick tasks for rapid testing.
    One from each difficulty/layer combination.
    """
    full_dataset = get_sleep_test_dataset()
    
    # Pick representative samples
    quick = [
        full_dataset[0],   # ALGO-001: Binary Search (Medium)
        full_dataset[10],  # ALGO-011: Graph BFS (Hard)
        full_dataset[20],  # SEC-001: Safe division (Hard/Shield)
        full_dataset[35],  # SRE-001: Thread-safe Singleton (Hard/Hive)
        full_dataset[45],  # SRE-011: Leader election (Expert/Immune)
    ]
    
    return quick


def print_dataset_summary():
    """Print a summary of the dataset."""
    dataset = get_sleep_test_dataset()
    
    print("\n" + "=" * 60)
    print("🌙 S.P.I.D.E.R. SLEEP TEST DATASET")
    print("=" * 60)
    print(f"\nTotal Tasks: {len(dataset)}")
    
    # Count by difficulty
    difficulty_counts = {}
    layer_counts = {}
    
    for task in dataset:
        d = task['difficulty']
        l = task['layer']
        difficulty_counts[d] = difficulty_counts.get(d, 0) + 1
        layer_counts[l] = layer_counts.get(l, 0) + 1
    
    print("\nBy Difficulty:")
    for d, count in sorted(difficulty_counts.items()):
        print(f"  {d}: {count}")
    
    print("\nBy Layer:")
    for l, count in sorted(layer_counts.items()):
        print(f"  {l}: {count}")
    
    print("\nSample Tasks:")
    for task in dataset[:3]:
        print(f"  [{task['id']}] {task['problem'][:50]}...")
    
    print()


if __name__ == "__main__":
    print_dataset_summary()