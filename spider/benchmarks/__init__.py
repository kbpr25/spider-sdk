"""S.P.I.D.E.R. Benchmarks Package - Production SWE-Bench Pipeline."""

from spider.benchmarks.harness import BenchmarkRunner, TrialResult, BenchmarkCase
from spider.benchmarks.swe_pipeline import (
    SWEBenchTask,
    SWEBenchSolver,
    SolverConfig,
    TestResultParser,
    TestResult,
    TestStatus,
    PatchGenerator,
)
from spider.benchmarks.swe_runner import SWEBenchRunner

__all__ = [
    # Original harness
    'BenchmarkRunner',
    'TrialResult',
    'BenchmarkCase',
    # New SWE-Bench pipeline
    'SWEBenchTask',
    'SWEBenchSolver',
    'SolverConfig',
    'TestResultParser',
    'TestResult',
    'TestStatus',
    'PatchGenerator',
    'SWEBenchRunner',
]
