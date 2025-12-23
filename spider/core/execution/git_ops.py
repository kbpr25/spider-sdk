"""
S.P.I.D.E.R. Git Surgeon - Production-Grade Git Operations
============================================================

The Iron Interface between LLM-generated patches and real Git repos.

Key Challenges:
- LLMs mess up line numbers
- Patches fail on whitespace issues
- Working directories can be dirty
- Tests can hang indefinitely

This module handles ALL of these edge cases.
"""

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging


# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# RESULT TYPES
# =============================================================================

class PatchResult(Enum):
    """Result of a patch application attempt."""
    SUCCESS = "success"
    PARTIAL = "partial"        # Applied with rejects
    FAILED = "failed"
    CONFLICT = "conflict"      # Working directory dirty
    INVALID = "invalid"        # Malformed patch


@dataclass
class OperationResult:
    """Result of a Git operation."""
    success: bool
    message: str = ""
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    duration_ms: float = 0
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __bool__(self) -> bool:
        return self.success


@dataclass
class TestResult:
    """Result of running tests."""
    passed: bool
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    errors: int = 0
    stdout: str = ""
    stderr: str = ""
    duration_ms: float = 0
    timeout: bool = False


# =============================================================================
# GIT OPERATOR
# =============================================================================

class GitOperator:
    """
    Production-grade Git operations for S.P.I.D.E.R.
    
    Handles the messy reality of:
    - Fuzzy patch application
    - Dirty working directories
    - Branch management
    - Test execution with timeouts
    
    Example:
        git = GitOperator("/path/to/repo")
        
        # Create a feature branch
        git.checkout_new_branch("spider/fix-bug-123")
        
        # Apply LLM-generated patch
        result = git.apply_patch(diff_content)
        if result.success:
            git.commit("Fix: Handle edge case in parser")
            
            # Run tests
            test_result = git.run_tests("pytest tests/")
            if test_result.passed:
                print("✅ All tests pass!")
    """
    
    def __init__(
        self,
        repo_path: str,
        timeout: int = 60,
        git_binary: str = "git",
    ):
        """
        Initialize the Git operator.
        
        Args:
            repo_path: Path to the Git repository
            timeout: Default timeout for operations (seconds)
            git_binary: Path to git executable
        """
        self.repo_path = Path(repo_path).resolve()
        self.timeout = timeout
        self.git_binary = git_binary
        
        # Validate repo
        if not (self.repo_path / ".git").exists():
            logger.warning(f"No .git directory found at {self.repo_path}")
    
    def _run_git(
        self,
        args: List[str],
        timeout: Optional[int] = None,
        check: bool = False,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess:
        """
        Run a git command.
        
        Args:
            args: Git subcommand and arguments
            timeout: Timeout in seconds (None = use default)
            check: Raise on non-zero exit
            capture_output: Capture stdout/stderr
            
        Returns:
            CompletedProcess instance
        """
        cmd = [self.git_binary] + args
        timeout = timeout or self.timeout
        
        logger.debug(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.repo_path),
                capture_output=capture_output,
                text=True,
                timeout=timeout,
                check=check,
            )
            return result
        except subprocess.TimeoutExpired as e:
            logger.error(f"Git command timed out: {' '.join(cmd)}")
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {e.stderr}")
            raise
    
    def _run_command(
        self,
        cmd: List[str],
        timeout: Optional[int] = None,
        cwd: Optional[str] = None,
    ) -> subprocess.CompletedProcess:
        """Run an arbitrary command."""
        timeout = timeout or self.timeout
        cwd = cwd or str(self.repo_path)
        
        logger.debug(f"Running: {' '.join(cmd)}")
        
        return subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=True if os.name == 'nt' else False,
        )
    
    # =========================================================================
    # REPOSITORY STATE
    # =========================================================================
    
    def is_repo(self) -> bool:
        """Check if the path is a valid Git repository."""
        try:
            result = self._run_git(["rev-parse", "--git-dir"])
            return result.returncode == 0
        except Exception:
            return False
    
    def is_dirty(self) -> bool:
        """Check if the working directory has uncommitted changes."""
        try:
            result = self._run_git(["status", "--porcelain"])
            return bool(result.stdout.strip())
        except Exception:
            return True
    
    def current_branch(self) -> str:
        """Get the current branch name."""
        result = self._run_git(["rev-parse", "--abbrev-ref", "HEAD"])
        return result.stdout.strip()
    
    def stash(self, message: str = "spider-auto-stash") -> OperationResult:
        """Stash current changes."""
        start = time.time()
        
        if not self.is_dirty():
            return OperationResult(
                success=True,
                message="Nothing to stash",
            )
        
        result = self._run_git(["stash", "push", "-m", message])
        
        return OperationResult(
            success=result.returncode == 0,
            message="Stashed changes" if result.returncode == 0 else "Failed to stash",
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
            duration_ms=(time.time() - start) * 1000,
        )
    
    def stash_pop(self) -> OperationResult:
        """Pop the last stash."""
        start = time.time()
        result = self._run_git(["stash", "pop"])
        
        return OperationResult(
            success=result.returncode == 0,
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
            duration_ms=(time.time() - start) * 1000,
        )
    
    # =========================================================================
    # BRANCH OPERATIONS
    # =========================================================================
    
    def checkout_new_branch(self, branch_name: str) -> OperationResult:
        """
        Create and switch to a new branch.
        
        Handles dirty working directory by stashing first.
        
        Args:
            branch_name: Name of the new branch
            
        Returns:
            OperationResult with success/failure info
        """
        start = time.time()
        
        # Handle dirty state
        was_dirty = self.is_dirty()
        if was_dirty:
            stash_result = self.stash(f"spider-before-{branch_name}")
            if not stash_result.success:
                return OperationResult(
                    success=False,
                    message=f"Failed to stash changes: {stash_result.stderr}",
                )
        
        # Create and checkout branch
        result = self._run_git(["checkout", "-b", branch_name])
        
        if result.returncode != 0:
            # Maybe branch already exists? Try just checkout
            result = self._run_git(["checkout", branch_name])
        
        # Restore stash if we had one
        if was_dirty:
            self.stash_pop()
        
        return OperationResult(
            success=result.returncode == 0,
            message=f"Switched to branch {branch_name}" if result.returncode == 0 else f"Failed to checkout {branch_name}",
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
            duration_ms=(time.time() - start) * 1000,
        )
    
    def checkout(self, ref: str) -> OperationResult:
        """Checkout an existing branch or commit."""
        start = time.time()
        result = self._run_git(["checkout", ref])
        
        return OperationResult(
            success=result.returncode == 0,
            message=f"Checked out {ref}" if result.returncode == 0 else f"Failed to checkout {ref}",
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
            duration_ms=(time.time() - start) * 1000,
        )
    
    def delete_branch(self, branch_name: str, force: bool = False) -> OperationResult:
        """Delete a branch."""
        flag = "-D" if force else "-d"
        result = self._run_git(["branch", flag, branch_name])
        
        return OperationResult(
            success=result.returncode == 0,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    
    # =========================================================================
    # PATCH APPLICATION (THE FUZZY PATCHER)
    # =========================================================================
    
    def apply_patch(
        self,
        diff_content: str,
        allow_partial: bool = True,
    ) -> OperationResult:
        """
        Apply a patch with fuzzy matching and error recovery.
        
        Strategy:
        1. Try exact application
        2. Try with fuzz factor
        3. Try with whitespace fixing
        4. Try with --reject (partial application)
        
        Args:
            diff_content: The diff/patch content as a string
            allow_partial: Allow partial application with .rej files
            
        Returns:
            OperationResult with details about what worked
        """
        start = time.time()
        
        # Write patch to temp file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.patch',
            delete=False,
            encoding='utf-8',
        ) as f:
            f.write(diff_content)
            patch_file = f.name
        
        try:
            # Strategy 1: Try exact application with dry-run first
            check_result = self._run_git(["apply", "--check", patch_file])
            
            if check_result.returncode == 0:
                # Patch will apply cleanly
                apply_result = self._run_git(["apply", patch_file])
                
                return OperationResult(
                    success=apply_result.returncode == 0,
                    message="Patch applied successfully",
                    stdout=apply_result.stdout,
                    stderr=apply_result.stderr,
                    return_code=apply_result.returncode,
                    duration_ms=(time.time() - start) * 1000,
                    details={"method": "exact"},
                )
            
            # Strategy 2: Try with whitespace fixing
            logger.info("Exact patch failed, trying with whitespace fix...")
            ws_result = self._run_git([
                "apply",
                "--whitespace=fix",
                "--ignore-space-change",
                patch_file,
            ])
            
            if ws_result.returncode == 0:
                return OperationResult(
                    success=True,
                    message="Patch applied with whitespace fixes",
                    stdout=ws_result.stdout,
                    stderr=ws_result.stderr,
                    return_code=ws_result.returncode,
                    duration_ms=(time.time() - start) * 1000,
                    details={"method": "whitespace_fix"},
                )
            
            # Strategy 3: Try with fuzz factor (git apply doesn't support this, but patch does)
            # We'll try with 3-way merge instead
            logger.info("Whitespace fix failed, trying 3-way merge...")
            three_way_result = self._run_git([
                "apply",
                "--3way",
                patch_file,
            ])
            
            if three_way_result.returncode == 0:
                return OperationResult(
                    success=True,
                    message="Patch applied with 3-way merge",
                    stdout=three_way_result.stdout,
                    stderr=three_way_result.stderr,
                    return_code=three_way_result.returncode,
                    duration_ms=(time.time() - start) * 1000,
                    details={"method": "3way"},
                )
            
            # Strategy 4: Force application with rejects
            if allow_partial:
                logger.info("3-way failed, trying with rejects...")
                reject_result = self._run_git([
                    "apply",
                    "--reject",
                    "--whitespace=fix",
                    patch_file,
                ])
                
                # Check for .rej files
                rej_files = list(self.repo_path.rglob("*.rej"))
                
                if reject_result.returncode == 0 or len(rej_files) > 0:
                    return OperationResult(
                        success=len(rej_files) == 0,
                        message=f"Patch partially applied ({len(rej_files)} rejects)",
                        stdout=reject_result.stdout,
                        stderr=reject_result.stderr,
                        return_code=reject_result.returncode,
                        duration_ms=(time.time() - start) * 1000,
                        details={
                            "method": "reject",
                            "rej_files": [str(f) for f in rej_files],
                        },
                    )
            
            # All strategies failed
            return OperationResult(
                success=False,
                message="Failed to apply patch",
                stdout=check_result.stdout,
                stderr=check_result.stderr,
                return_code=check_result.returncode,
                duration_ms=(time.time() - start) * 1000,
                details={"method": "none"},
            )
            
        finally:
            # Cleanup temp file
            try:
                os.unlink(patch_file)
            except Exception:
                pass
    
    def apply_patch_manual(
        self,
        file_path: str,
        old_content: str,
        new_content: str,
    ) -> OperationResult:
        """
        Apply a change by direct file manipulation.
        
        Fallback when git apply fails completely.
        
        Args:
            file_path: Path to file (relative to repo)
            old_content: Expected old content
            new_content: New content to write
            
        Returns:
            OperationResult
        """
        start = time.time()
        target = self.repo_path / file_path
        
        try:
            # Check if file exists and matches expected content
            if target.exists():
                current = target.read_text(encoding='utf-8')
                if old_content and current != old_content:
                    # Try fuzzy matching
                    if old_content.strip() not in current:
                        return OperationResult(
                            success=False,
                            message="File content doesn't match expected",
                            details={"method": "manual_fuzzy_fail"},
                        )
            
            # Write new content
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(new_content, encoding='utf-8')
            
            return OperationResult(
                success=True,
                message=f"Manually applied to {file_path}",
                duration_ms=(time.time() - start) * 1000,
                details={"method": "manual"},
            )
            
        except Exception as e:
            return OperationResult(
                success=False,
                message=f"Manual patch failed: {e}",
                details={"method": "manual", "error": str(e)},
            )
    
    def clean_rejects(self) -> int:
        """Remove all .rej files from the repository."""
        rej_files = list(self.repo_path.rglob("*.rej"))
        for f in rej_files:
            f.unlink()
        return len(rej_files)
    
    # =========================================================================
    # COMMIT OPERATIONS
    # =========================================================================
    
    def add(self, paths: Optional[List[str]] = None) -> OperationResult:
        """Stage files for commit."""
        if paths is None:
            paths = ["."]
        
        result = self._run_git(["add"] + paths)
        
        return OperationResult(
            success=result.returncode == 0,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    
    def commit(self, message: str, add_all: bool = True) -> OperationResult:
        """
        Create a commit.
        
        Args:
            message: Commit message
            add_all: Stage all changes before committing
            
        Returns:
            OperationResult with commit info
        """
        start = time.time()
        
        if add_all:
            self.add(["-A"])
        
        result = self._run_git(["commit", "-m", message])
        
        # Get commit hash if successful
        commit_hash = ""
        if result.returncode == 0:
            hash_result = self._run_git(["rev-parse", "HEAD"])
            commit_hash = hash_result.stdout.strip()[:8]
        
        return OperationResult(
            success=result.returncode == 0,
            message=f"Committed: {commit_hash}" if result.returncode == 0 else "Commit failed",
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
            duration_ms=(time.time() - start) * 1000,
            details={"commit_hash": commit_hash},
        )
    
    def reset(self, mode: str = "hard", ref: str = "HEAD") -> OperationResult:
        """Reset the repository."""
        result = self._run_git(["reset", f"--{mode}", ref])
        
        return OperationResult(
            success=result.returncode == 0,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    
    # =========================================================================
    # TEST EXECUTION
    # =========================================================================
    
    def run_tests(
        self,
        test_cmd: str = "pytest",
        timeout: int = 300,
        env: Optional[Dict[str, str]] = None,
    ) -> TestResult:
        """
        Run tests in the repository.
        
        Handles:
        - Timeout (tests that hang)
        - Output capture
        - Basic result parsing
        
        Args:
            test_cmd: Command to run tests (e.g., "pytest tests/")
            timeout: Maximum time in seconds
            env: Additional environment variables
            
        Returns:
            TestResult with pass/fail info
        """
        start = time.time()
        
        # Build environment
        run_env = os.environ.copy()
        if env:
            run_env.update(env)
        
        try:
            result = subprocess.run(
                test_cmd,
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=True,
                env=run_env,
            )
            
            duration_ms = (time.time() - start) * 1000
            
            # Parse pytest output (basic)
            passed = result.returncode == 0
            total_tests = 0
            passed_tests = 0
            failed_tests = 0
            
            # Try to extract pytest summary
            output = result.stdout + result.stderr
            
            # Look for "X passed" pattern
            import re
            passed_match = re.search(r'(\d+) passed', output)
            failed_match = re.search(r'(\d+) failed', output)
            error_match = re.search(r'(\d+) error', output)
            
            if passed_match:
                passed_tests = int(passed_match.group(1))
            if failed_match:
                failed_tests = int(failed_match.group(1))
            
            total_tests = passed_tests + failed_tests
            
            return TestResult(
                passed=passed,
                total_tests=total_tests,
                passed_tests=passed_tests,
                failed_tests=failed_tests,
                errors=int(error_match.group(1)) if error_match else 0,
                stdout=result.stdout,
                stderr=result.stderr,
                duration_ms=duration_ms,
                timeout=False,
            )
            
        except subprocess.TimeoutExpired:
            return TestResult(
                passed=False,
                stdout="",
                stderr=f"Test command timed out after {timeout}s",
                duration_ms=timeout * 1000,
                timeout=True,
            )
        except Exception as e:
            return TestResult(
                passed=False,
                stdout="",
                stderr=str(e),
                duration_ms=(time.time() - start) * 1000,
            )
    
    # =========================================================================
    # UTILITY
    # =========================================================================
    
    def get_diff(self, ref: str = "HEAD") -> str:
        """Get the current diff."""
        result = self._run_git(["diff", ref])
        return result.stdout
    
    def get_log(self, count: int = 10) -> str:
        """Get recent commit log."""
        result = self._run_git([
            "log",
            f"-{count}",
            "--oneline",
            "--decorate",
        ])
        return result.stdout
    
    def __repr__(self) -> str:
        return f"GitOperator({self.repo_path})"


# =============================================================================
# DEMO
# =============================================================================

if __name__ == '__main__':
    import tempfile
    
    print("=" * 60)
    print("🔧 S.P.I.D.E.R. GIT SURGEON - Demo")
    print("=" * 60)
    print()
    
    # Create a temp repo for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize repo
        subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "spider@test.com"],
            cwd=tmpdir, capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Spider"],
            cwd=tmpdir, capture_output=True,
        )
        
        # Create initial file
        test_file = Path(tmpdir) / "hello.py"
        test_file.write_text("print('Hello')\n")
        
        subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial"],
            cwd=tmpdir, capture_output=True,
        )
        
        print(f"Created test repo at: {tmpdir}")
        print()
        
        # Test GitOperator
        git = GitOperator(tmpdir)
        
        # Check status
        print(f"Is repo: {git.is_repo()}")
        print(f"Is dirty: {git.is_dirty()}")
        print(f"Current branch: {git.current_branch()}")
        print()
        
        # Create branch
        result = git.checkout_new_branch("spider/feature-1")
        print(f"Branch creation: {result.message}")
        print()
        
        # Apply a patch
        patch = '''diff --git a/hello.py b/hello.py
--- a/hello.py
+++ b/hello.py
@@ -1 +1,2 @@
 print('Hello')
+print('World')
'''
        
        result = git.apply_patch(patch)
        print(f"Patch application: {result.message}")
        print(f"Method used: {result.details.get('method', 'unknown')}")
        print()
        
        # Commit
        result = git.commit("Add greeting")
        print(f"Commit: {result.message}")
        print()
        
        # Show log
        print("Git log:")
        print(git.get_log(5))
        
        print("=" * 60)
        print("✅ Git Surgeon operational!")
