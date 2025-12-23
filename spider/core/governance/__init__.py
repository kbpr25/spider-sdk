"""
S.P.I.D.E.R. Governance Package
===============================

Provides AI-supervised code governance.

Components:
- ConstitutionalGovernor: Code review against engineering principles
"""

from .constitutional import (
    ConstitutionalGovernor,
    Constitution,
    ConstitutionalCritic,
    Principle,
    Violation,
    CritiqueResult,
    RevisionResult,
    PrincipleCategory,
    ViolationSeverity,
)

__all__ = [
    "ConstitutionalGovernor",
    "Constitution",
    "ConstitutionalCritic",
    "Principle",
    "Violation",
    "CritiqueResult",
    "RevisionResult",
    "PrincipleCategory",
    "ViolationSeverity",
]
