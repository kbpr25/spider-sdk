"""
S.P.I.D.E.R. Security Package
==============================

Provides security layers for safe code execution.

Components:
- CommandFirewall: Shell command safety layer
- AuditLogger: Command audit logging
"""

from .firewall import CommandFirewall, CommandAnalysis, ThreatLevel, AuditLogger

__all__ = ["CommandFirewall", "CommandAnalysis", "ThreatLevel", "AuditLogger"]
