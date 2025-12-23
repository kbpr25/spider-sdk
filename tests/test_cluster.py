"""
Cluster & Distributed Consensus Test Suite
===========================================

Tests for SpiderNode, SpiderCluster, and Phi Failure Detection.
"""

import pytest
import time
import multiprocessing
from unittest.mock import MagicMock

from spider.core.sre.failure_detector import PhiFailureDetector, ClusterHealthMonitor


class TestPhiFailureDetector:
    """Tests for the Phi Accrual Failure Detector."""

    def test_healthy_node(self):
        """Test that regular heartbeats keep phi low."""
        detector = PhiFailureDetector(threshold=8.0)
        
        # Simulate regular heartbeats
        for _ in range(10):
            detector.heartbeat()
            time.sleep(0.05)
        
        phi = detector.phi()
        assert phi < 2.0  # Should be healthy

    def test_dead_node(self):
        """Test that missing heartbeats increase phi."""
        detector = PhiFailureDetector(threshold=8.0)
        
        # Establish baseline
        for _ in range(5):
            detector.heartbeat()
            time.sleep(0.05)
        
        # Simulate silence
        time.sleep(0.5)
        
        phi = detector.phi()
        assert phi > 2.0  # Should be suspicious

    def test_threshold(self):
        """Test is_available based on threshold."""
        detector = PhiFailureDetector(threshold=8.0)
        
        # With heartbeats
        for _ in range(5):
            detector.heartbeat()
            time.sleep(0.05)
        
        assert detector.is_available() is True

    def test_phi_increases_monotonically(self):
        """Test that phi increases with silence."""
        detector = PhiFailureDetector(threshold=8.0)
        
        # Establish baseline
        for _ in range(5):
            detector.heartbeat()
            time.sleep(0.05)
        
        phi_values = []
        for _ in range(5):
            time.sleep(0.1)
            phi_values.append(detector.phi())
        
        # Phi should generally increase (allowing some noise)
        assert phi_values[-1] > phi_values[0]


class TestClusterHealthMonitor:
    """Tests for the ClusterHealthMonitor."""

    def test_register_nodes(self):
        """Test registering nodes."""
        monitor = ClusterHealthMonitor()
        
        monitor.register_node("node-1")
        monitor.register_node("node-2")
        
        assert "node-1" in monitor.get_all_nodes()
        assert "node-2" in monitor.get_all_nodes()

    def test_heartbeat_recording(self):
        """Test recording heartbeats."""
        monitor = ClusterHealthMonitor()
        monitor.register_node("node-1")
        
        for _ in range(5):
            monitor.heartbeat("node-1")
            time.sleep(0.05)
        
        health = monitor.get_node_status("node-1")
        assert health is not None
        assert health.phi < 5.0

    def test_available_nodes(self):
        """Test getting available nodes."""
        monitor = ClusterHealthMonitor(threshold=8.0)
        monitor.register_node("node-1")
        monitor.register_node("node-2")
        
        # Only node-1 has heartbeats
        for _ in range(5):
            monitor.heartbeat("node-1")
            time.sleep(0.05)
        
        available = monitor.get_available_nodes()
        assert "node-1" in [n[0] for n in available]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
