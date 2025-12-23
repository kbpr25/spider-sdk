"""
S.P.I.D.E.R. Reflexion Buffer - Episodic Learning Memory
=========================================================

Born from: Debug-5 (Reflexion)

The Scientific Finding:
"Standard agents have no memory. If they fix a SQL injection in Task A,
they will make the same mistake in Task B. Reflexion converts feedback
into textual summaries of 'What I learned' stored in a sliding buffer."

The Solution:
We give the SDK a "Long-Term Trauma Memory."

1. Post-Mortem: After fixing a bug, ask "What was the root cause?"
2. Storage: Save the lesson to persistent memory
3. Retrieval: Before new tasks, inject relevant lessons as "THINGS NOT TO DO"

Result: The Agent gets SMARTER with every bug it fixes. It EVOLVES.
"""

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# LESSON TYPES
# =============================================================================

@dataclass
class Lesson:
    """A learned lesson from debugging."""
    lesson_id: str
    summary: str                    # One-sentence summary
    root_cause: str                 # What caused the bug
    fix_applied: str                # How it was fixed
    domain: str = ""                # e.g., "Django ORM", "async", "SQL"
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    accessed_count: int = 0
    relevance_score: float = 1.0
    
    def to_prompt(self) -> str:
        """Format lesson for injection into prompt."""
        return f"⚠️ LESSON [{self.domain}]: {self.summary}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "lesson_id": self.lesson_id,
            "summary": self.summary,
            "root_cause": self.root_cause,
            "fix_applied": self.fix_applied,
            "domain": self.domain,
            "tags": self.tags,
            "created_at": self.created_at,
            "accessed_count": self.accessed_count,
            "relevance_score": self.relevance_score,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Lesson":
        """Create lesson from dictionary."""
        return cls(
            lesson_id=data["lesson_id"],
            summary=data["summary"],
            root_cause=data["root_cause"],
            fix_applied=data["fix_applied"],
            domain=data.get("domain", ""),
            tags=data.get("tags", []),
            created_at=data.get("created_at", time.time()),
            accessed_count=data.get("accessed_count", 0),
            relevance_score=data.get("relevance_score", 1.0),
        )


@dataclass
class LessonQuery:
    """Query for retrieving relevant lessons."""
    keywords: List[str] = field(default_factory=list)
    domain: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    max_results: int = 5
    min_relevance: float = 0.1


# =============================================================================
# LESSON STORE
# =============================================================================

class LessonStore:
    """
    Persistent storage for debugging lessons.
    
    Uses JSON file storage with keyword indexing.
    """
    
    def __init__(self, storage_path: str = None):
        self.storage_path = Path(storage_path or ".spider_lessons.json")
        self.lessons: Dict[str, Lesson] = {}
        self.keyword_index: Dict[str, Set[str]] = {}  # keyword -> lesson_ids
        self.domain_index: Dict[str, Set[str]] = {}   # domain -> lesson_ids
        
        self._load()
    
    def _load(self) -> None:
        """Load lessons from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                
                for item in data.get("lessons", []):
                    lesson = Lesson.from_dict(item)
                    self.lessons[lesson.lesson_id] = lesson
                    self._index_lesson(lesson)
                    
            except Exception as e:
                logger.warning(f"Failed to load lessons: {e}")
    
    def _save(self) -> None:
        """Save lessons to disk."""
        try:
            data = {
                "lessons": [l.to_dict() for l in self.lessons.values()],
                "version": 1,
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save lessons: {e}")
    
    def _index_lesson(self, lesson: Lesson) -> None:
        """Index a lesson for fast retrieval."""
        # Index keywords from summary
        words = re.findall(r'\w+', lesson.summary.lower())
        for word in words:
            if len(word) > 3:
                if word not in self.keyword_index:
                    self.keyword_index[word] = set()
                self.keyword_index[word].add(lesson.lesson_id)
        
        # Index domain
        if lesson.domain:
            domain_lower = lesson.domain.lower()
            if domain_lower not in self.domain_index:
                self.domain_index[domain_lower] = set()
            self.domain_index[domain_lower].add(lesson.lesson_id)
        
        # Index tags
        for tag in lesson.tags:
            tag_lower = tag.lower()
            if tag_lower not in self.keyword_index:
                self.keyword_index[tag_lower] = set()
            self.keyword_index[tag_lower].add(lesson.lesson_id)
    
    def add(self, lesson: Lesson) -> None:
        """Add a lesson to the store."""
        self.lessons[lesson.lesson_id] = lesson
        self._index_lesson(lesson)
        self._save()
    
    def get(self, lesson_id: str) -> Optional[Lesson]:
        """Get a lesson by ID."""
        lesson = self.lessons.get(lesson_id)
        if lesson:
            lesson.accessed_count += 1
            self._save()
        return lesson
    
    def search(self, query: LessonQuery) -> List[Lesson]:
        """Search for relevant lessons."""
        scores: Dict[str, float] = {}
        
        # Score by keywords
        for keyword in query.keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in self.keyword_index:
                for lesson_id in self.keyword_index[keyword_lower]:
                    scores[lesson_id] = scores.get(lesson_id, 0) + 1
        
        # Score by domain
        if query.domain:
            domain_lower = query.domain.lower()
            if domain_lower in self.domain_index:
                for lesson_id in self.domain_index[domain_lower]:
                    scores[lesson_id] = scores.get(lesson_id, 0) + 2
        
        # Score by tags
        for tag in query.tags:
            tag_lower = tag.lower()
            if tag_lower in self.keyword_index:
                for lesson_id in self.keyword_index[tag_lower]:
                    scores[lesson_id] = scores.get(lesson_id, 0) + 1.5
        
        # Normalize and filter
        if not scores:
            return []
        
        max_score = max(scores.values())
        results = []
        
        for lesson_id, score in sorted(scores.items(), key=lambda x: -x[1]):
            normalized_score = score / max_score
            if normalized_score >= query.min_relevance:
                lesson = self.lessons.get(lesson_id)
                if lesson:
                    lesson.relevance_score = normalized_score
                    results.append(lesson)
                
                if len(results) >= query.max_results:
                    break
        
        return results
    
    def get_all(self) -> List[Lesson]:
        """Get all lessons."""
        return list(self.lessons.values())
    
    def clear(self) -> None:
        """Clear all lessons."""
        self.lessons.clear()
        self.keyword_index.clear()
        self.domain_index.clear()
        self._save()
    
    def stats(self) -> Dict[str, int]:
        """Get store statistics."""
        return {
            "total_lessons": len(self.lessons),
            "domains": len(self.domain_index),
            "keywords_indexed": len(self.keyword_index),
        }


# =============================================================================
# REFLEXION BUFFER
# =============================================================================

class ReflexionBuffer:
    """
    The Reflexive Episodic Memory System.
    
    Stores debugging lessons and injects them into future tasks:
    1. After successful fix: Generate lesson summary
    2. Before new task: Retrieve relevant lessons
    3. Inject as "THINGS NOT TO DO" in system prompt
    
    From Debug-5 (Reflexion):
    "The agent learns from its mistakes and evolves."
    
    Usage:
        buffer = ReflexionBuffer()
        
        # After fixing a bug
        buffer.learn_from_fix(
            root_cause="Django ORM update() doesn't trigger save() signals",
            fix_applied="Used save() on individual objects instead",
            domain="Django",
            tags=["orm", "signals"],
        )
        
        # Before new task
        lessons = buffer.get_relevant_lessons(
            task_description="Update user profiles in Django",
            keywords=["django", "update", "orm"],
        )
        
        # Inject into prompt
        prompt += buffer.format_lessons_for_prompt(lessons)
    """
    
    LESSON_GENERATION_PROMPT = """You just successfully fixed a bug. Generate a concise lesson.

ROOT CAUSE: {root_cause}
FIX APPLIED: {fix_applied}
CODE CONTEXT: {code_context}

Generate a ONE SENTENCE lesson that a future engineer should remember.
Format: "When [situation], always [do X] because [reason]."

Example: "When using Django ORM update(), always use individual save() calls if you need signals to fire because update() bypasses the model layer."

YOUR LESSON:"""
    
    def __init__(
        self,
        storage_path: str = None,
        max_lessons: int = 1000,
        max_prompt_lessons: int = 5,
    ):
        """
        Initialize Reflexion Buffer.
        
        Args:
            storage_path: Path to store lessons
            max_lessons: Maximum lessons to store (FIFO eviction)
            max_prompt_lessons: Maximum lessons to inject per task
        """
        self.store = LessonStore(storage_path)
        self.max_lessons = max_lessons
        self.max_prompt_lessons = max_prompt_lessons
        
        self._stats = {
            "lessons_created": 0,
            "lessons_retrieved": 0,
            "lessons_applied": 0,
        }
    
    def learn_from_fix(
        self,
        root_cause: str,
        fix_applied: str,
        domain: str = "",
        tags: List[str] = None,
        code_context: str = "",
        llm_callback: Optional[Callable[[str], str]] = None,
    ) -> Lesson:
        """
        Create a lesson from a successful bug fix.
        
        Args:
            root_cause: What caused the bug
            fix_applied: How it was fixed
            domain: Domain (e.g., "Django", "async")
            tags: Relevant tags
            code_context: Optional code snippet
            llm_callback: LLM for generating summary
            
        Returns:
            Created lesson
        """
        tags = tags or []
        
        # Generate lesson summary
        if llm_callback:
            prompt = self.LESSON_GENERATION_PROMPT.format(
                root_cause=root_cause,
                fix_applied=fix_applied,
                code_context=code_context[:500],
            )
            summary = llm_callback(prompt).strip()
        else:
            # Auto-generate summary
            summary = f"When encountering {root_cause[:50]}..., fix by {fix_applied[:50]}..."
        
        # Create lesson
        lesson_id = hashlib.md5(
            f"{root_cause}{fix_applied}{time.time()}".encode()
        ).hexdigest()[:12]
        
        lesson = Lesson(
            lesson_id=lesson_id,
            summary=summary,
            root_cause=root_cause,
            fix_applied=fix_applied,
            domain=domain,
            tags=tags,
        )
        
        # Store lesson
        self.store.add(lesson)
        self._stats["lessons_created"] += 1
        
        # Evict old lessons if necessary
        self._evict_if_needed()
        
        logger.info(f"📚 Learned lesson: {summary[:60]}...")
        
        return lesson
    
    def get_relevant_lessons(
        self,
        task_description: str = "",
        keywords: List[str] = None,
        domain: str = "",
        tags: List[str] = None,
        max_results: int = None,
    ) -> List[Lesson]:
        """
        Retrieve lessons relevant to a task.
        
        Args:
            task_description: Description of the task
            keywords: Keywords to search for
            domain: Domain filter
            tags: Tag filters
            max_results: Maximum lessons to return
            
        Returns:
            List of relevant lessons
        """
        max_results = max_results or self.max_prompt_lessons
        
        # Extract keywords from task description
        search_keywords = keywords or []
        if task_description:
            # Extract significant words
            words = re.findall(r'\b\w{4,}\b', task_description.lower())
            search_keywords.extend(words[:10])
        
        query = LessonQuery(
            keywords=search_keywords,
            domain=domain,
            tags=tags or [],
            max_results=max_results,
        )
        
        lessons = self.store.search(query)
        self._stats["lessons_retrieved"] += len(lessons)
        
        return lessons
    
    def format_lessons_for_prompt(
        self,
        lessons: List[Lesson],
        header: str = "⚠️ LESSONS LEARNED (Do not repeat these mistakes):",
    ) -> str:
        """
        Format lessons for injection into LLM prompt.
        
        Returns:
            Formatted string for prompt injection
        """
        if not lessons:
            return ""
        
        self._stats["lessons_applied"] += len(lessons)
        
        lines = [header, ""]
        for i, lesson in enumerate(lessons, 1):
            lines.append(f"{i}. {lesson.to_prompt()}")
            lines.append(f"   Root cause: {lesson.root_cause[:100]}")
            lines.append(f"   Fix: {lesson.fix_applied[:100]}")
            lines.append("")
        
        return "\n".join(lines)
    
    def get_lessons_for_task(
        self,
        task_description: str,
        domain: str = "",
    ) -> str:
        """
        One-step method: Get formatted lessons for a task.
        
        Returns:
            Formatted lesson string for prompt injection
        """
        lessons = self.get_relevant_lessons(
            task_description=task_description,
            domain=domain,
        )
        return self.format_lessons_for_prompt(lessons)
    
    def _evict_if_needed(self) -> None:
        """Evict oldest lessons if over capacity."""
        all_lessons = self.store.get_all()
        if len(all_lessons) > self.max_lessons:
            # Sort by created_at and remove oldest
            all_lessons.sort(key=lambda l: l.created_at)
            to_remove = len(all_lessons) - self.max_lessons
            for lesson in all_lessons[:to_remove]:
                del self.store.lessons[lesson.lesson_id]
            self.store._save()
    
    def get_stats(self) -> Dict[str, int]:
        return {
            **self._stats,
            **self.store.stats(),
        }
    
    def print_lessons(self) -> None:
        """Print all stored lessons."""
        lessons = self.store.get_all()
        
        print("\n" + "=" * 60)
        print("📚 REFLEXION BUFFER - STORED LESSONS")
        print("=" * 60)
        print(f"\nTotal Lessons: {len(lessons)}")
        
        for lesson in lessons[:10]:
            print(f"\n🔸 [{lesson.domain}] {lesson.summary[:60]}...")
            print(f"   Tags: {', '.join(lesson.tags)}")
            print(f"   Accessed: {lesson.accessed_count} times")
        
        if len(lessons) > 10:
            print(f"\n... and {len(lessons) - 10} more lessons")
        
        print("=" * 60)


# =============================================================================
# PACKAGE INIT
# =============================================================================

__all__ = [
    "ReflexionBuffer",
    "Lesson",
    "LessonQuery",
    "LessonStore",
]


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("📚 S.P.I.D.E.R. Reflexion Buffer - Demo")
    print("=" * 70)
    
    # Use temp storage for demo
    buffer = ReflexionBuffer(storage_path=".demo_lessons.json")
    buffer.store.clear()
    
    # Simulate learning from fixes
    lessons_data = [
        {
            "root_cause": "Django ORM update() doesn't trigger save() signals",
            "fix_applied": "Used save() on individual objects instead of bulk update()",
            "domain": "Django",
            "tags": ["orm", "signals", "bulk"],
        },
        {
            "root_cause": "Async function called without await",
            "fix_applied": "Added await keyword before async function calls",
            "domain": "Python Async",
            "tags": ["async", "await", "coroutine"],
        },
        {
            "root_cause": "SQL injection via string formatting",
            "fix_applied": "Used parameterized queries with placeholders",
            "domain": "SQL",
            "tags": ["security", "injection", "sql"],
        },
        {
            "root_cause": "Race condition in concurrent access",
            "fix_applied": "Added mutex lock around critical section",
            "domain": "Concurrency",
            "tags": ["threading", "mutex", "race"],
        },
    ]
    
    print("\n📝 Learning from fixes...")
    for data in lessons_data:
        lesson = buffer.learn_from_fix(**data)
        print(f"   ✓ Learned: {lesson.summary[:50]}...")
    
    # Retrieve relevant lessons
    print("\n🔍 Retrieving lessons for 'Django user profile update'...")
    prompt_addition = buffer.get_lessons_for_task(
        task_description="Update user profile in Django using ORM",
        domain="Django",
    )
    
    print(prompt_addition)
    
    buffer.print_lessons()
    
    print(f"\n📊 Stats: {buffer.get_stats()}")
    
    # Cleanup
    os.remove(".demo_lessons.json")
