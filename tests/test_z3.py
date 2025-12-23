"""
Z3 Symbolic Verifier Test Suite
================================

Tests for the SymbolicVerifier and Z3 integration.
"""

import pytest

from spider.core.verifier.symbolic import (
    SymbolicVerifier,
    VerificationResult,
    VerificationStatus,
    prove,
    find_bug,
)


class TestSymbolicVerifier:
    """Tests for the SymbolicVerifier class."""

    @pytest.fixture
    def verifier(self):
        return SymbolicVerifier(timeout_ms=5000, log_level="WARNING")

    def test_prove_correct_code(self, verifier):
        """Test proving correct code."""
        result = verifier.verify_contract(
            code_str="y = x + 5",
            pre_condition="x > 0",
            post_condition="y > 5",
        )
        
        assert result.status == VerificationStatus.PROVEN
        assert result.proven is True

    def test_find_buggy_code(self, verifier):
        """Test finding bugs in incorrect code."""
        result = verifier.verify_contract(
            code_str="y = x + 1",
            pre_condition="x >= 0",
            post_condition="y > 5",
        )
        
        assert result.status == VerificationStatus.DISPROVEN
        assert result.proven is False
        assert result.counter_example is not None

    def test_division_safety(self, verifier):
        """Test division safety with precondition."""
        result = verifier.verify_contract(
            code_str="result = x / y",
            pre_condition="y != 0",
            post_condition="True",
        )
        
        assert result.proven is True

    def test_complex_arithmetic(self, verifier):
        """Test complex arithmetic expressions."""
        result = verifier.verify_contract(
            code_str="result = (a + b) * 2",
            pre_condition="a > 0 and b > 0",
            post_condition="result > a and result > b",
        )
        
        assert result.status == VerificationStatus.PROVEN

    def test_stats_tracking(self, verifier):
        """Test that statistics are tracked."""
        verifier.verify_contract("y = x + 1", "x > 0", "y > 1")
        
        assert verifier.stats['total_verifications'] == 1


class TestConvenienceFunctions:
    """Tests for the prove() and find_bug() helpers."""

    def test_prove_helper(self):
        """Test the prove() convenience function."""
        assert prove("y = x + 5", "x > 0", "y > 5") is True
        assert prove("y = x + 1", "x >= 0", "y > 10") is False

    def test_find_bug_helper(self):
        """Test the find_bug() convenience function."""
        bug = find_bug("y = x + 1", "x >= 0", "y > 5")
        
        assert bug is not None
        assert 'x' in bug.inputs


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def verifier(self):
        return SymbolicVerifier(log_level="WARNING")

    def test_off_by_one(self, verifier):
        """Test off-by-one error detection."""
        result = verifier.verify_contract(
            code_str="y = x + 5",
            pre_condition="x >= 0",  # Bug: allows x=0
            post_condition="y > 5",  # Fails when x=0
        )
        
        assert result.proven is False

    def test_subtraction_underflow(self, verifier):
        """Test subtraction underflow detection."""
        result = verifier.verify_contract(
            code_str="result = x - 5",
            pre_condition="x > 0",
            post_condition="result >= 0",
        )
        
        assert result.proven is False  # x=1 gives result=-4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
