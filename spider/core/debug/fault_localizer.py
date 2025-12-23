"""
S.P.I.D.E.R. Fault Localizer - Spectra-Based Fault Localization
================================================================

Born from: Debug-3 (Automated Program Repair)

The Scientific Finding:
"LLMs are great at rewriting logic but bad at pinpointing WHERE to edit.
Traditional APR tools (statistical fault localization) are great at finding
the buggy line but bad at fixing it. Combining them beats either alone."

The Solution:
Implement Spectra-Based Fault Localization (SBFL):

1. Coverage Matrix: Track which lines run in passing vs failing tests
2. Suspiciousness Score: Calculate Ochiai/Tarantula score per line
3. Surgical Prompting: Tell LLM "Line 45 has 95% suspiciousness"

Result: Drastic reduction in token costs and "hallucinated fixes."
"""

import ast
import hashlib
import logging
import re
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# SBFL TYPES
# =============================================================================

@dataclass
class LineCoverage:
    """Coverage information for a single line."""
    line_no: int
    code: str = ""
    passed_tests: int = 0        # Tests that passed and executed this line
    failed_tests: int = 0        # Tests that failed and executed this line
    total_executions: int = 0
    suspiciousness: float = 0.0


@dataclass
class FaultLocation:
    """A suspicious fault location."""
    line_no: int
    code: str
    suspiciousness: float
    rank: int
    context: str = ""            # Surrounding lines
    variables: List[str] = field(default_factory=list)


@dataclass
class LocalizationResult:
    """Result of fault localization."""
    file_path: str
    total_lines: int
    suspicious_lines: List[FaultLocation]
    coverage_summary: Dict[str, int]
    tests_passed: int
    tests_failed: int
    top_suspects: List[int]      # Top 5 line numbers


class SBFLMetric(Enum):
    """SBFL suspiciousness metrics."""
    OCHIAI = auto()
    TARANTULA = auto()
    JACCARD = auto()
    DSTAR = auto()


# =============================================================================
# COVERAGE COLLECTOR
# =============================================================================

class CoverageCollector:
    """
    Collects line coverage during test execution.
    
    Uses sys.settrace to track which lines are executed.
    """
    
    def __init__(self, target_file: str = "<string>"):
        self.target_file = target_file
        self.covered_lines: Set[int] = set()
        self.line_hits: Dict[int, int] = {}
    
    def start(self) -> None:
        """Start coverage collection."""
        self.covered_lines.clear()
        self.line_hits.clear()
        sys.settrace(self._trace_callback)
    
    def stop(self) -> None:
        """Stop coverage collection."""
        sys.settrace(None)
    
    def _trace_callback(self, frame, event, arg):
        """Trace callback for coverage."""
        if event != "line":
            return self._trace_callback
        
        if frame.f_code.co_filename == self.target_file:
            line_no = frame.f_lineno
            self.covered_lines.add(line_no)
            self.line_hits[line_no] = self.line_hits.get(line_no, 0) + 1
        
        return self._trace_callback
    
    def get_coverage(self) -> Set[int]:
        """Get set of covered line numbers."""
        return self.covered_lines.copy()


# =============================================================================
# SBFL CALCULATOR
# =============================================================================

class SBFLCalculator:
    """
    Calculates suspiciousness scores using SBFL metrics.
    
    Metrics implemented:
    - Ochiai: ef / sqrt((ef + nf) * (ef + ep))
    - Tarantula: (ef/nf) / ((ef/nf) + (ep/np))
    - Jaccard: ef / (ef + nf + ep)
    - D*: ef^2 / (nf + ep)
    
    Where:
    - ef = failing tests that execute the statement
    - ep = passing tests that execute the statement
    - nf = failing tests that don't execute the statement
    - np = passing tests that don't execute the statement
    """
    
    def __init__(self, metric: SBFLMetric = SBFLMetric.OCHIAI):
        self.metric = metric
    
    def calculate(
        self,
        ef: int,    # Failed tests executing line
        ep: int,    # Passed tests executing line
        nf: int,    # Failed tests NOT executing line
        np_: int,   # Passed tests NOT executing line
    ) -> float:
        """Calculate suspiciousness score."""
        if self.metric == SBFLMetric.OCHIAI:
            return self._ochiai(ef, ep, nf, np_)
        elif self.metric == SBFLMetric.TARANTULA:
            return self._tarantula(ef, ep, nf, np_)
        elif self.metric == SBFLMetric.JACCARD:
            return self._jaccard(ef, ep, nf, np_)
        elif self.metric == SBFLMetric.DSTAR:
            return self._dstar(ef, ep, nf, np_)
        else:
            return self._ochiai(ef, ep, nf, np_)
    
    def _ochiai(self, ef: int, ep: int, nf: int, np_: int) -> float:
        """Ochiai metric - best overall performance."""
        denominator = ((ef + nf) * (ef + ep)) ** 0.5
        if denominator == 0:
            return 0.0
        return ef / denominator
    
    def _tarantula(self, ef: int, ep: int, nf: int, np_: int) -> float:
        """Tarantula metric."""
        total_failed = ef + nf
        total_passed = ep + np_
        
        if total_failed == 0 or total_passed == 0:
            return 0.0
        
        fail_ratio = ef / total_failed
        pass_ratio = ep / total_passed
        
        if fail_ratio + pass_ratio == 0:
            return 0.0
        
        return fail_ratio / (fail_ratio + pass_ratio)
    
    def _jaccard(self, ef: int, ep: int, nf: int, np_: int) -> float:
        """Jaccard metric."""
        denominator = ef + nf + ep
        if denominator == 0:
            return 0.0
        return ef / denominator
    
    def _dstar(self, ef: int, ep: int, nf: int, np_: int, star: int = 2) -> float:
        """D* metric with star=2."""
        denominator = nf + ep
        if denominator == 0:
            return float('inf') if ef > 0 else 0.0
        return (ef ** star) / denominator


# =============================================================================
# FAULT LOCALIZER
# =============================================================================

class FaultLocalizer:
    """
    Spectra-Based Fault Localization for S.P.I.D.E.R.
    
    Combines APR precision with LLM flexibility:
    1. Run tests and collect coverage
    2. Calculate suspiciousness scores
    3. Identify top suspicious lines
    4. Generate surgical prompt for LLM
    
    From Debug-3:
    "Use APR to find the line, LLM to fix it."
    
    Usage:
        localizer = FaultLocalizer()
        
        # Add test results with coverage
        localizer.add_test_result(
            test_name="test_valid_input",
            passed=True,
            covered_lines={1, 2, 3, 5, 6}
        )
        localizer.add_test_result(
            test_name="test_edge_case",
            passed=False,
            covered_lines={1, 2, 3, 4, 6}  # Line 4 only in failing!
        )
        
        # Localize fault
        result = localizer.localize(code)
        
        # Get surgical prompt
        prompt = localizer.get_surgical_prompt(result)
    """
    
    def __init__(
        self,
        metric: SBFLMetric = SBFLMetric.OCHIAI,
        top_n: int = 5,
        context_lines: int = 3,
    ):
        """
        Initialize Fault Localizer.
        
        Args:
            metric: SBFL metric to use
            top_n: Number of top suspects to return
            context_lines: Lines of context around suspects
        """
        self.metric = metric
        self.top_n = top_n
        self.context_lines = context_lines
        
        self.calculator = SBFLCalculator(metric)
        
        # Test results
        self.test_results: List[Dict[str, Any]] = []
        self.passing_coverage: List[Set[int]] = []
        self.failing_coverage: List[Set[int]] = []
        
        self._stats = {
            "tests_analyzed": 0,
            "faults_localized": 0,
            "avg_rank": 0.0,
        }
    
    def add_test_result(
        self,
        test_name: str,
        passed: bool,
        covered_lines: Set[int],
    ) -> None:
        """
        Add a test result with coverage.
        
        Args:
            test_name: Name of the test
            passed: Whether the test passed
            covered_lines: Set of line numbers executed
        """
        self.test_results.append({
            "name": test_name,
            "passed": passed,
            "covered": covered_lines,
        })
        
        if passed:
            self.passing_coverage.append(covered_lines)
        else:
            self.failing_coverage.append(covered_lines)
        
        self._stats["tests_analyzed"] += 1
    
    def run_test_and_collect(
        self,
        code: str,
        test_code: str,
        test_name: str = "test",
    ) -> bool:
        """
        Run a single test and collect coverage.
        
        Args:
            code: Source code to test
            test_code: Test code to execute
            test_name: Name for this test
            
        Returns:
            Whether the test passed
        """
        collector = CoverageCollector()
        
        exec_globals = {'__builtins__': __builtins__}
        exec_locals = {}
        
        passed = True
        try:
            # Execute source code first
            exec(compile(code, '<string>', 'exec'), exec_globals, exec_locals)
            
            # Execute test with coverage
            collector.start()
            try:
                exec(compile(test_code, '<string>', 'exec'), exec_globals, exec_locals)
            finally:
                collector.stop()
        except Exception as e:
            passed = False
            collector.stop()
        
        self.add_test_result(test_name, passed, collector.get_coverage())
        return passed
    
    def localize(
        self,
        code: str,
        file_path: str = "<string>",
    ) -> LocalizationResult:
        """
        Perform fault localization on code.
        
        Args:
            code: Source code to analyze
            file_path: File path for reporting
            
        Returns:
            LocalizationResult with suspicious lines
        """
        self._stats["faults_localized"] += 1
        
        lines = code.split('\n')
        total_lines = len(lines)
        
        # Get all executed lines
        all_lines = set()
        for test in self.test_results:
            all_lines.update(test["covered"])
        
        # Calculate suspiciousness for each line
        line_coverage: Dict[int, LineCoverage] = {}
        
        total_passed = len(self.passing_coverage)
        total_failed = len(self.failing_coverage)
        
        for line_no in range(1, total_lines + 1):
            # Count test categories
            ef = sum(1 for cov in self.failing_coverage if line_no in cov)
            ep = sum(1 for cov in self.passing_coverage if line_no in cov)
            nf = total_failed - ef
            np_ = total_passed - ep
            
            # Calculate suspiciousness
            suspiciousness = self.calculator.calculate(ef, ep, nf, np_)
            
            line_coverage[line_no] = LineCoverage(
                line_no=line_no,
                code=lines[line_no - 1] if line_no <= len(lines) else "",
                passed_tests=ep,
                failed_tests=ef,
                total_executions=ef + ep,
                suspiciousness=suspiciousness,
            )
        
        # Rank by suspiciousness
        ranked = sorted(
            line_coverage.values(),
            key=lambda x: x.suspiciousness,
            reverse=True,
        )
        
        # Build suspicious locations
        suspicious_lines = []
        for rank, cov in enumerate(ranked[:self.top_n], 1):
            if cov.suspiciousness > 0:
                context = self._get_context(lines, cov.line_no)
                suspicious_lines.append(FaultLocation(
                    line_no=cov.line_no,
                    code=cov.code,
                    suspiciousness=cov.suspiciousness,
                    rank=rank,
                    context=context,
                ))
        
        return LocalizationResult(
            file_path=file_path,
            total_lines=total_lines,
            suspicious_lines=suspicious_lines,
            coverage_summary={
                "lines_covered": len(all_lines),
                "lines_total": total_lines,
                "coverage_pct": len(all_lines) / total_lines * 100 if total_lines > 0 else 0,
            },
            tests_passed=total_passed,
            tests_failed=total_failed,
            top_suspects=[loc.line_no for loc in suspicious_lines],
        )
    
    def _get_context(self, lines: List[str], line_no: int) -> str:
        """Get surrounding context for a line."""
        start = max(0, line_no - 1 - self.context_lines)
        end = min(len(lines), line_no + self.context_lines)
        
        context_lines = []
        for i in range(start, end):
            prefix = ">>>" if i == line_no - 1 else "   "
            context_lines.append(f"{prefix} {i + 1}: {lines[i]}")
        
        return "\n".join(context_lines)
    
    def get_surgical_prompt(
        self,
        result: LocalizationResult,
        code: str,
        error: str = "",
    ) -> str:
        """
        Generate a surgical prompt for LLM to fix the bug.
        
        Instead of showing the whole file, focuses on suspicious lines.
        
        Args:
            result: Localization result
            code: Full source code
            error: Error message
            
        Returns:
            Focused prompt for LLM
        """
        if not result.suspicious_lines:
            return f"No suspicious lines identified. Full code:\n```python\n{code}\n```"
        
        top = result.suspicious_lines[0]
        
        prompt = f"""SURGICAL DEBUG REQUEST

The automated fault localizer has identified the most likely bug location.

FILE: {result.file_path}
ERROR: {error}

[*] MOST SUSPICIOUS (Rank #{top.rank}, {top.suspiciousness:.0%} suspiciousness):
Line {top.line_no}: {top.code.strip()}

CONTEXT:
{top.context}

COVERAGE ANALYSIS:
- This line was executed by {result.tests_failed} FAILING tests
- This line was executed by {result.tests_passed - result.tests_failed} PASSING tests
- Total tests: {result.tests_passed + result.tests_failed}

OTHER SUSPECTS:
"""
        for loc in result.suspicious_lines[1:]:
            prompt += f"  Line {loc.line_no} ({loc.suspiciousness:.0%}): {loc.code.strip()[:50]}\n"
        
        prompt += """
YOUR TASK:
1. Focus on Line {top_line} - this is the most likely bug location
2. Explain what's wrong with this specific line
3. Provide the FIXED version of lines {start}-{end} only

Respond with:
ISSUE: [What's wrong with line {top_line}]
FIX:
```python
[Fixed code for lines {start}-{end}]
```
""".format(
            top_line=top.line_no,
            start=max(1, top.line_no - 2),
            end=min(result.total_lines, top.line_no + 2),
        )
        
        return prompt
    
    def clear(self) -> None:
        """Clear all test results."""
        self.test_results.clear()
        self.passing_coverage.clear()
        self.failing_coverage.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        return self._stats.copy()
    
    def print_result(self, result: LocalizationResult) -> None:
        """Print localization result."""
        print("\n" + "=" * 60)
        print("[*] FAULT LOCALIZATION RESULT")
        print("=" * 60)
        
        print(f"\n[F] File: {result.file_path}")
        print(f"[%] Coverage: {result.coverage_summary['coverage_pct']:.1f}%")
        print(f"[+] Passed: {result.tests_passed} | [X] Failed: {result.tests_failed}")
        
        print(f"\n[?] TOP {len(result.suspicious_lines)} SUSPICIOUS LINES:")
        for loc in result.suspicious_lines:
            bar = "#" * int(loc.suspiciousness * 20)
            print(f"\n   #{loc.rank} Line {loc.line_no} [{loc.suspiciousness:.0%}] {bar}")
            print(f"      {loc.code.strip()[:60]}")
        
        print("\n" + "=" * 60)


# =============================================================================
# PACKAGE INIT
# =============================================================================

__all__ = [
    "FaultLocalizer",
    "LocalizationResult",
    "FaultLocation",
    "LineCoverage",
    "SBFLMetric",
    "SBFLCalculator",
    "CoverageCollector",
]


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("[*] S.P.I.D.E.R. Fault Localizer - Demo")
    print("=" * 70)
    
    localizer = FaultLocalizer(metric=SBFLMetric.OCHIAI)
    
    # Buggy code
    code = '''
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    average = total / len(numbers)  # Bug: no check for empty list
    return average

def process_data(data):
    filtered = [x for x in data if x > 0]
    return calculate_average(filtered)
'''
    
    print("\n[i] CODE WITH BUG:")
    for i, line in enumerate(code.split('\n'), 1):
        print(f"   {i}: {line}")
    
    # Simulate test coverage
    # Test 1: passes (non-empty list)
    localizer.add_test_result(
        "test_with_data",
        passed=True,
        covered_lines={2, 3, 4, 5, 6, 7, 9, 10, 11}
    )
    
    # Test 2: passes (positive numbers)
    localizer.add_test_result(
        "test_positive",
        passed=True,
        covered_lines={2, 3, 4, 5, 6, 7, 9, 10, 11}
    )
    
    # Test 3: FAILS (empty after filter)
    localizer.add_test_result(
        "test_all_negative",
        passed=False,
        covered_lines={2, 3, 6, 9, 10, 11}  # Skips loop, crashes at line 6
    )
    
    # Test 4: FAILS (empty input)
    localizer.add_test_result(
        "test_empty_input",
        passed=False,
        covered_lines={2, 3, 6, 9, 10, 11}  # Same pattern
    )
    
    # Localize fault
    result = localizer.localize(code)
    localizer.print_result(result)
    
    # Generate surgical prompt
    print("\n[>] SURGICAL PROMPT FOR LLM:")
    print("-" * 60)
    prompt = localizer.get_surgical_prompt(result, code, "ZeroDivisionError: division by zero")
    print(prompt[:800])
    print("-" * 60)
    
    print(f"\n[%] Stats: {localizer.get_stats()}")
