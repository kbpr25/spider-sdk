"""
Merkle Tree Test Suite
======================

Tests for the MerkleTree and CodebaseMerkleTree components.
"""

import pytest
import tempfile
from pathlib import Path

from spider.core.dsa.merkle import MerkleTree, CodebaseMerkleTree


class TestMerkleTree:
    """Tests for the MerkleTree class."""

    def test_single_leaf(self):
        """Test tree with single leaf."""
        tree = MerkleTree(["hello"])
        
        assert tree.root is not None
        assert len(tree.root) == 64  # SHA-256 hex

    def test_multiple_leaves(self):
        """Test tree with multiple leaves."""
        tree = MerkleTree(["a", "b", "c", "d"])
        
        assert tree.root is not None

    def test_proof_verification(self):
        """Test proof generation and verification."""
        leaves = ["item1", "item2", "item3", "item4"]
        tree = MerkleTree(leaves)
        
        # Get proof for item2
        proof = tree.get_proof("item2")
        
        # Verify the proof
        assert tree.verify_proof("item2", proof, tree.root) is True

    def test_tamper_detection(self):
        """Test that tampering is detected."""
        leaves = ["a", "b", "c", "d"]
        tree = MerkleTree(leaves)
        original_root = tree.root
        
        # Create tampered tree
        tampered = MerkleTree(["a", "TAMPERED", "c", "d"])
        
        assert tampered.root != original_root

    def test_empty_tree(self):
        """Test empty tree handling."""
        tree = MerkleTree([])
        
        assert tree.root == ""

    def test_deterministic(self):
        """Test that same inputs produce same root."""
        leaves = ["x", "y", "z"]
        
        tree1 = MerkleTree(leaves)
        tree2 = MerkleTree(leaves)
        
        assert tree1.root == tree2.root


class TestCodebaseMerkleTree:
    """Tests for the CodebaseMerkleTree class."""

    def test_index_directory(self, tmp_path):
        """Test indexing a directory."""
        # Create test files
        (tmp_path / "file1.py").write_text("content1")
        (tmp_path / "file2.py").write_text("content2")
        
        tree = CodebaseMerkleTree(str(tmp_path))
        
        assert tree.root is not None
        assert len(tree.files) >= 2

    def test_change_detection(self, tmp_path):
        """Test that file changes are detected."""
        test_file = tmp_path / "test.py"
        test_file.write_text("original content")
        
        tree1 = CodebaseMerkleTree(str(tmp_path))
        root1 = tree1.root
        
        # Modify file
        test_file.write_text("modified content")
        
        tree2 = CodebaseMerkleTree(str(tmp_path))
        root2 = tree2.root
        
        assert root1 != root2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
