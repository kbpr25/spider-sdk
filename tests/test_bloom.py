"""
Bloom Filter Test Suite
========================

Tests for the CodebaseIndexer and BloomFilter components.
"""

import pytest
import tempfile
import os
from pathlib import Path

from spider.core.dsa.bloom import BloomFilter, CodebaseIndexer


class TestBloomFilter:
    """Tests for the BloomFilter class."""

    def test_basic_add_and_check(self):
        """Test basic add and membership check."""
        bf = BloomFilter(size=1000, hash_count=3)
        
        bf.add("hello")
        bf.add("world")
        
        assert bf.check("hello") is True
        assert bf.check("world") is True
        assert bf.count == 2

    def test_false_positives_rare(self):
        """Test that false positives are rare for properly sized filter."""
        bf = BloomFilter(size=10000, hash_count=7)
        
        # Add 100 items
        for i in range(100):
            bf.add(f"item_{i}")
        
        # Check for items that were never added
        false_positives = 0
        for i in range(100, 1000):
            if bf.check(f"item_{i}"):
                false_positives += 1
        
        # False positive rate should be very low
        assert false_positives < 50  # Less than 5%

    def test_optimal_sizing(self):
        """Test optimal filter creation."""
        bf = BloomFilter.optimal(expected_items=1000, false_positive_rate=0.01)
        
        assert bf.size > 0
        assert bf.hash_count > 0

    def test_merge_filters(self):
        """Test merging two bloom filters."""
        bf1 = BloomFilter(size=1000, hash_count=3)
        bf2 = BloomFilter(size=1000, hash_count=3)
        
        bf1.add("a")
        bf2.add("b")
        
        bf1.merge(bf2)
        
        assert bf1.check("a") is True
        assert bf1.check("b") is True


class TestCodebaseIndexer:
    """Tests for the CodebaseIndexer class."""

    def test_index_directory(self, tmp_path):
        """Test indexing a directory."""
        # Create test files
        (tmp_path / "test.py").write_text("def hello(): pass")
        (tmp_path / "other.py").write_text("class Foo: pass")
        
        indexer = CodebaseIndexer(str(tmp_path), expected_symbols=100)
        indexer.index()
        
        assert indexer.stats.total_files >= 2

    def test_symbol_lookup(self, tmp_path):
        """Test symbol lookup after indexing."""
        (tmp_path / "test.py").write_text("def my_function(): pass")
        
        indexer = CodebaseIndexer(str(tmp_path), expected_symbols=100)
        indexer.index()
        
        # Check that the file was indexed
        assert indexer.stats.total_files >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
