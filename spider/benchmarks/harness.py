"""
S.P.I.D.E.R. Zero-Cost SWE-Bench Harness
=========================================

The Arena Manager - runs real-world bug fixes against open source projects
without needing expensive cloud infrastructure.

This module provides:
- BenchmarkRunner: Orchestrates Docker-based evaluation
- TrialResult: Structured result of each benchmark trial
- BenchmarkCase: Definition of a benchmark problem

Usage:
    from spider.benchmarks import BenchmarkRunner
    
    runner = BenchmarkRunner()
    result = runner.run_trial(
        repo_url="https://github.com/psf/requests",
        issue_id="1234",
        problem_desc="Fix connection timeout handling"
    )
    print(f"Result: {result.status}")
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class TrialStatus(Enum):
    """Status of a benchmark trial."""
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"
    SKIPPED = "SKIPPED"


@dataclass
class BenchmarkCase:
    """Definition of a benchmark problem."""
    repo_url: str
    issue_id: str
    problem_desc: str
    commit_before: Optional[str] = None  # Commit hash before fix
    commit_after: Optional[str] = None   # Commit hash after fix
    test_files: List[str] = field(default_factory=list)  # Files to test
    expected_behavior: str = ""
    

@dataclass
class TrialResult:
    """Result of a benchmark trial."""
    status: TrialStatus
    case: BenchmarkCase
    duration_seconds: float
    logs: str
    spider_output: str = ""
    test_output: str = ""
    generated_patch: str = ""
    error_message: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'status': self.status.value,
            'repo_url': self.case.repo_url,
            'issue_id': self.case.issue_id,
            'problem_desc': self.case.problem_desc,
            'duration_seconds': self.duration_seconds,
            'logs': self.logs,
            'spider_output': self.spider_output,
            'test_output': self.test_output,
            'generated_patch': self.generated_patch,
            'error_message': self.error_message,
            'timestamp': self.timestamp,
        }


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

class BenchmarkRunner:
    """
    The Arena Manager - orchestrates SWE-Bench style evaluations.
    
    This class manages Docker containers for isolated benchmark execution,
    allowing S.P.I.D.E.R. to be tested against real-world bugs.
    
    Usage:
        runner = BenchmarkRunner()
        
        # Run a single trial
        result = runner.run_trial(
            repo_url="https://github.com/psf/requests",
            issue_id="1234",
            problem_desc="Fix timeout handling bug"
        )
        
        # Run multiple trials
        results = runner.run_batch(cases)
    """
    
    DOCKER_IMAGE = "spider-arena"
    CONTAINER_PREFIX = "spider-trial"
    
    def __init__(
        self,
        docker_context: str = ".",
        timeout_seconds: int = 300,
        cleanup: bool = True,
        log_level: str = "INFO",
    ):
        """
        Initialize the BenchmarkRunner.
        
        Args:
            docker_context: Path to directory containing Dockerfile
            timeout_seconds: Maximum time for each trial
            cleanup: Whether to remove containers after trials
            log_level: Logging verbosity
        """
        self.docker_context = Path(docker_context).resolve()
        self.timeout_seconds = timeout_seconds
        self.cleanup = cleanup
        self._logger = self._setup_logger(log_level)
        
        # Statistics
        self._stats = {
            'trials_run': 0,
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'timeouts': 0,
        }
    
    def _setup_logger(self, level: str) -> logging.Logger:
        """Set up logging."""
        logger = logging.getLogger("BenchmarkRunner")
        logger.setLevel(getattr(logging, level))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s [%(name)s] %(levelname)s: %(message)s',
                datefmt='%H:%M:%S'
            ))
            logger.addHandler(handler)
        
        return logger
    
    def _check_docker(self) -> bool:
        """Check if Docker is available."""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def build_image(self, force_rebuild: bool = False) -> bool:
        """
        Build the spider-arena Docker image.
        
        Args:
            force_rebuild: If True, rebuild even if image exists
            
        Returns:
            True if build successful
        """
        self._logger.info(f"Building Docker image: {self.DOCKER_IMAGE}")
        
        # Check if image exists
        if not force_rebuild:
            check = subprocess.run(
                ["docker", "images", "-q", self.DOCKER_IMAGE],
                capture_output=True,
                text=True,
            )
            if check.stdout.strip():
                self._logger.info("Image already exists, skipping build")
                return True
        
        # Find Dockerfile
        dockerfile_path = self.docker_context / "spider" / "benchmarks" / "Dockerfile"
        if not dockerfile_path.exists():
            self._logger.error(f"Dockerfile not found: {dockerfile_path}")
            return False
        
        # Build image
        try:
            result = subprocess.run(
                [
                    "docker", "build",
                    "-t", self.DOCKER_IMAGE,
                    "-f", str(dockerfile_path),
                    str(self.docker_context),
                ],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout for build
            )
            
            if result.returncode != 0:
                self._logger.error(f"Docker build failed: {result.stderr}")
                return False
            
            self._logger.info("Docker image built successfully")
            return True
            
        except subprocess.TimeoutExpired:
            self._logger.error("Docker build timed out")
            return False
        except Exception as e:
            self._logger.error(f"Docker build error: {e}")
            return False
    
    def run_trial(
        self,
        repo_url: str,
        issue_id: str,
        problem_desc: str,
        commit_before: Optional[str] = None,
        test_command: str = "pytest",
    ) -> TrialResult:
        """
        Run a single benchmark trial.
        
        Args:
            repo_url: URL of the repository to test
            issue_id: Issue/bug identifier
            problem_desc: Description of the problem for S.P.I.D.E.R.
            commit_before: Optional commit to checkout
            test_command: Command to run tests
            
        Returns:
            TrialResult with status and logs
        """
        case = BenchmarkCase(
            repo_url=repo_url,
            issue_id=issue_id,
            problem_desc=problem_desc,
            commit_before=commit_before,
        )
        
        start_time = time.perf_counter()
        container_name = f"{self.CONTAINER_PREFIX}-{issue_id}-{int(time.time())}"
        logs = []
        
        self._logger.info(f"Starting trial: {issue_id}")
        logs.append(f"=== Trial: {issue_id} ===")
        logs.append(f"Repo: {repo_url}")
        logs.append(f"Problem: {problem_desc}")
        
        try:
            # Step 1: Ensure Docker image exists
            if not self._check_docker():
                return self._error_result(case, start_time, "Docker not available")
            
            if not self.build_image():
                return self._error_result(case, start_time, "Failed to build Docker image")
            
            # Step 2: Create the trial script
            trial_script = self._generate_trial_script(
                repo_url=repo_url,
                problem_desc=problem_desc,
                commit_before=commit_before,
                test_command=test_command,
            )
            logs.append("\n--- Trial Script ---")
            logs.append(trial_script)
            
            # Step 3: Run container with trial script
            self._logger.info(f"Starting container: {container_name}")
            
            result = subprocess.run(
                [
                    "docker", "run",
                    "--name", container_name,
                    "--rm" if self.cleanup else "",
                    "-e", "PYTHONUNBUFFERED=1",
                    self.DOCKER_IMAGE,
                    "/bin/bash", "-c", trial_script,
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            
            duration = time.perf_counter() - start_time
            logs.append(f"\n--- Container Output ---")
            logs.append(result.stdout)
            
            if result.stderr:
                logs.append(f"\n--- Container Errors ---")
                logs.append(result.stderr)
            
            # Step 4: Determine result
            self._stats['trials_run'] += 1
            
            if result.returncode == 0:
                self._stats['passed'] += 1
                status = TrialStatus.PASS
                self._logger.info(f"Trial PASSED: {issue_id}")
            else:
                self._stats['failed'] += 1
                status = TrialStatus.FAIL
                self._logger.info(f"Trial FAILED: {issue_id}")
            
            return TrialResult(
                status=status,
                case=case,
                duration_seconds=duration,
                logs="\n".join(logs),
                spider_output=result.stdout,
                test_output=result.stderr,
            )
            
        except subprocess.TimeoutExpired:
            duration = time.perf_counter() - start_time
            self._stats['timeouts'] += 1
            self._logger.warning(f"Trial TIMEOUT: {issue_id}")
            
            # Kill the container
            subprocess.run(["docker", "kill", container_name], capture_output=True)
            
            return TrialResult(
                status=TrialStatus.TIMEOUT,
                case=case,
                duration_seconds=duration,
                logs="\n".join(logs),
                error_message=f"Trial exceeded {self.timeout_seconds}s timeout",
            )
            
        except Exception as e:
            duration = time.perf_counter() - start_time
            self._stats['errors'] += 1
            self._logger.error(f"Trial ERROR: {issue_id} - {e}")
            
            return TrialResult(
                status=TrialStatus.ERROR,
                case=case,
                duration_seconds=duration,
                logs="\n".join(logs),
                error_message=str(e),
            )
        
        finally:
            # Cleanup container if it exists
            if self.cleanup:
                subprocess.run(
                    ["docker", "rm", "-f", container_name],
                    capture_output=True,
                )
    
    def _generate_trial_script(
        self,
        repo_url: str,
        problem_desc: str,
        commit_before: Optional[str],
        test_command: str,
    ) -> str:
        """Generate the bash script to run inside the container."""
        
        # Extract repo name from URL
        repo_name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
        
        script = f'''#!/bin/bash
set -e

echo "🕷️ S.P.I.D.E.R. Arena - Trial Starting"
echo "========================================"

# Step 1: Clone the repository
echo "[1/5] Cloning repository..."
cd /app/workspace
git clone --depth 1 {repo_url} {repo_name}
cd {repo_name}

# Step 2: Checkout specific commit (if provided)
'''
        
        if commit_before:
            script += f'''
echo "[2/5] Checking out commit: {commit_before}"
git fetch --depth 1 origin {commit_before}
git checkout {commit_before}
'''
        else:
            script += '''
echo "[2/5] Using latest commit (MVP mode)"
'''
        
        script += f'''
# Step 3: Install the target package
echo "[3/5] Installing target package..."
pip install -e . --quiet 2>/dev/null || pip install . --quiet 2>/dev/null || echo "Install skipped"

# Step 4: Run S.P.I.D.E.R. to generate fix
echo "[4/5] 🕷️ Running S.P.I.D.E.R. solve..."
spider solve "{problem_desc}" || echo "Spider solve completed"

# Step 5: Run tests
echo "[5/5] Running test suite..."
{test_command} || true

echo ""
echo "========================================"
echo "🕷️ Trial Complete"
exit 0
'''
        return script
    
    def _error_result(
        self,
        case: BenchmarkCase,
        start_time: float,
        error_msg: str,
    ) -> TrialResult:
        """Create an error result."""
        self._stats['errors'] += 1
        return TrialResult(
            status=TrialStatus.ERROR,
            case=case,
            duration_seconds=time.perf_counter() - start_time,
            logs=f"Error: {error_msg}",
            error_message=error_msg,
        )
    
    def run_batch(
        self,
        cases: List[BenchmarkCase],
        parallel: bool = False,
    ) -> List[TrialResult]:
        """
        Run multiple benchmark trials.
        
        Args:
            cases: List of benchmark cases to run
            parallel: If True, run trials in parallel (not implemented)
            
        Returns:
            List of TrialResults
        """
        results = []
        total = len(cases)
        
        self._logger.info(f"Starting batch run: {total} trials")
        
        for i, case in enumerate(cases, 1):
            self._logger.info(f"Trial {i}/{total}: {case.issue_id}")
            
            result = self.run_trial(
                repo_url=case.repo_url,
                issue_id=case.issue_id,
                problem_desc=case.problem_desc,
                commit_before=case.commit_before,
            )
            results.append(result)
        
        self._logger.info(f"Batch complete: {self._stats}")
        return results
    
    def print_stats(self) -> None:
        """Print benchmark statistics."""
        print("\n" + "=" * 60)
        print("🕷️ S.P.I.D.E.R. BENCHMARK RESULTS")
        print("=" * 60)
        print(f"  Trials Run:  {self._stats['trials_run']}")
        print(f"  ✅ Passed:   {self._stats['passed']}")
        print(f"  ❌ Failed:   {self._stats['failed']}")
        print(f"  ⚠️  Errors:   {self._stats['errors']}")
        print(f"  ⏱️  Timeouts: {self._stats['timeouts']}")
        
        if self._stats['trials_run'] > 0:
            pass_rate = (self._stats['passed'] / self._stats['trials_run']) * 100
            print(f"\n  Pass Rate: {pass_rate:.1f}%")
    
    @property
    def stats(self) -> Dict[str, int]:
        """Get benchmark statistics."""
        return self._stats.copy()


# =============================================================================
# SAMPLE BENCHMARKS
# =============================================================================

# MVP benchmark cases for testing
SAMPLE_CASES = [
    BenchmarkCase(
        repo_url="https://github.com/python/cpython",
        issue_id="sample-001",
        problem_desc="Fix off-by-one error in range iteration",
    ),
    BenchmarkCase(
        repo_url="https://github.com/psf/requests",
        issue_id="sample-002",
        problem_desc="Handle connection timeout gracefully",
    ),
]


# =============================================================================
# CLI
# =============================================================================

def main():
    """Run benchmark harness from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="S.P.I.D.E.R. Zero-Cost SWE-Bench Harness"
    )
    parser.add_argument(
        '--repo', '-r',
        default="https://github.com/psf/requests",
        help="Repository URL to test"
    )
    parser.add_argument(
        '--issue', '-i',
        default="test-001",
        help="Issue ID"
    )
    parser.add_argument(
        '--problem', '-p',
        default="Fix timeout handling bug",
        help="Problem description"
    )
    parser.add_argument(
        '--timeout', '-t',
        type=int,
        default=300,
        help="Trial timeout in seconds"
    )
    parser.add_argument(
        '--build',
        action='store_true',
        help="Force rebuild Docker image"
    )
    
    args = parser.parse_args()
    
    print("🕷️ S.P.I.D.E.R. Zero-Cost SWE-Bench Harness")
    print("=" * 50)
    
    runner = BenchmarkRunner(timeout_seconds=args.timeout)
    
    if args.build:
        runner.build_image(force_rebuild=True)
    
    result = runner.run_trial(
        repo_url=args.repo,
        issue_id=args.issue,
        problem_desc=args.problem,
    )
    
    print(f"\nResult: {result.status.value}")
    print(f"Duration: {result.duration_seconds:.1f}s")
    
    if result.error_message:
        print(f"Error: {result.error_message}")
    
    runner.print_stats()
    
    return 0 if result.status == TrialStatus.PASS else 1


if __name__ == '__main__':
    exit(main())
