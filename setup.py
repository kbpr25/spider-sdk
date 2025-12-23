"""
S.P.I.D.E.R. - Strategic Protocol for Intelligent Distributed Execution and Reasoning
======================================================================================

The Verification Layer That AI Agents Are Missing.

This is a production-grade SDK for building reliable, self-healing AI agents
that use mathematical verification instead of probabilistic guessing.

Components:
    - Scout (Bloom Filter): O(1) codebase indexing
    - Council (Distributed): Raft-inspired multi-agent consensus
    - Shield (Z3): SMT-based formal verification
    - Watchdog (Phi): Probabilistic failure detection

Installation:
    pip install spider-core

    Or from source:
    pip install -e .

Usage:
    # CLI
    spider demo                  # Run demonstration
    spider solve "fix auth.py"   # Solve a problem
    spider status                # Check system status

    # Python API
    from spider import SpiderEngine
    engine = SpiderEngine()
    engine.start()
    result = engine.solve("implement safe division")

Author: Solo Founder
License: MIT
"""

import os
import sys
from pathlib import Path

from setuptools import setup, find_packages

# Read version from package
VERSION = "0.1.0-alpha"

# Read long description from README
HERE = Path(__file__).parent
README_PATH = HERE / "README.md"
if README_PATH.exists():
    LONG_DESCRIPTION = README_PATH.read_text(encoding="utf-8")
else:
    LONG_DESCRIPTION = __doc__

# Core dependencies
INSTALL_REQUIRES = [
    "z3-solver>=4.12.0",      # Microsoft Z3 Theorem Prover
    "requests>=2.31.0",       # HTTP for Ollama API
]

# Optional dependencies for extended features
EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=23.0.0",
        "isort>=5.12.0",
        "mypy>=1.0.0",
    ],
    "ollama": [
        "requests>=2.31.0",
    ],
    "gpu": [
        "torch>=2.0.0",        # For local LLM inference
        "transformers>=4.30.0",
    ],
    "all": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=23.0.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
    ],
}

# Package metadata
setup(
    # =========================================================================
    # PACKAGE IDENTITY
    # =========================================================================
    name="spider-core",
    version=VERSION,
    description=(
        "S.P.I.D.E.R: Strategic Protocol for Intelligent Distributed "
        "Execution and Reasoning - The Verification Layer for AI Agents"
    ),
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    
    # =========================================================================
    # AUTHOR & PROJECT INFO
    # =========================================================================
    author="Solo Founder",
    author_email="founder@spider.ai",
    url="https://github.com/spider-ai/spider-core",
    project_urls={
        "Documentation": "https://spider.ai/docs",
        "Source": "https://github.com/spider-ai/spider-core",
        "Issues": "https://github.com/spider-ai/spider-core/issues",
        "Changelog": "https://github.com/spider-ai/spider-core/blob/main/CHANGELOG.md",
    },
    license="MIT",
    
    # =========================================================================
    # PACKAGE DISCOVERY
    # =========================================================================
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    include_package_data=True,
    zip_safe=False,
    
    # =========================================================================
    # DEPENDENCIES
    # =========================================================================
    python_requires=">=3.9",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    
    # =========================================================================
    # ENTRY POINTS (CLI)
    # =========================================================================
    entry_points={
        "console_scripts": [
            "spider=spider.main:main",
        ],
    },
    
    # =========================================================================
    # CLASSIFIERS (for PyPI)
    # =========================================================================
    classifiers=[
        # Development Status
        "Development Status :: 3 - Alpha",
        
        # Intended Audience
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        
        # Topic
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Quality Assurance",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
        
        # License
        "License :: OSI Approved :: MIT License",
        
        # Python Versions
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        
        # OS
        "Operating System :: OS Independent",
        
        # Typing
        "Typing :: Typed",
    ],
    
    # =========================================================================
    # KEYWORDS (for PyPI search)
    # =========================================================================
    keywords=[
        "ai",
        "agents",
        "formal-verification",
        "z3",
        "theorem-prover",
        "distributed-systems",
        "consensus",
        "bloom-filter",
        "merkle-tree",
        "reliability",
        "self-healing",
        "llm",
        "code-review",
        "swe-bench",
    ],
)