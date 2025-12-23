"""
S.P.I.D.E.R. Docker Executor - Isolated Test Execution
========================================================

Run tests in isolated Docker containers with:
- Timeout handling
- OOM (Out of Memory) detection
- Output capture
- Resource limits

This is the "Iron Sandbox" that makes SWE-Bench evaluation possible.
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
import json


logger = logging.getLogger(__name__)


# =============================================================================
# RESULT TYPES
# =============================================================================

class ContainerStatus(Enum):
    """Status of a container execution."""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    OOM = "oom"  # Out of Memory
    ERROR = "error"


@dataclass
class ContainerResult:
    """Result of running code in a container."""
    status: ContainerStatus
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    duration_ms: float = 0
    container_id: str = ""
    tests_passed: int = 0
    tests_failed: int = 0
    tests_total: int = 0
    
    @property
    def success(self) -> bool:
        return self.status == ContainerStatus.SUCCESS and self.exit_code == 0


# =============================================================================
# DOCKER EXECUTOR
# =============================================================================

class DockerExecutor:
    """
    Execute tests in isolated Docker containers.
    
    Features:
    - Container pooling for faster execution
    - Timeout handling (kill hanging tests)
    - OOM detection
    - Resource limits (CPU, memory)
    - Volume mounting for code
    
    Example:
        executor = DockerExecutor(image="python:3.11-slim")
        
        result = executor.run_tests(
            repo_path="/path/to/repo",
            test_cmd="pytest tests/",
            timeout=300,
        )
        
        if result.success:
            print(f"✅ {result.tests_passed}/{result.tests_total} passed")
    """
    
    DEFAULT_IMAGE = "python:3.11-slim"
    
    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        memory_limit: str = "1g",
        cpu_limit: float = 1.0,
        docker_binary: str = "docker",
        network: str = "none",  # Isolated by default
    ):
        """
        Initialize the Docker executor.
        
        Args:
            image: Docker image to use
            memory_limit: Memory limit (e.g., "1g", "512m")
            cpu_limit: CPU limit (cores)
            docker_binary: Path to docker executable
            network: Network mode ("none", "bridge", "host")
        """
        self.image = image
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.docker_binary = docker_binary
        self.network = network
        
        # Pool of ready containers
        self._pool: List[str] = []
        self._max_pool_size = 3
    
    def is_available(self) -> bool:
        """Check if Docker is available."""
        try:
            result = subprocess.run(
                [self.docker_binary, "version"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def pull_image(self) -> bool:
        """Pull the Docker image if not present."""
        try:
            result = subprocess.run(
                [self.docker_binary, "pull", self.image],
                capture_output=True,
                timeout=300,
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Failed to pull image: {e}")
            return False
    
    def _create_container(
        self,
        repo_path: str,
        working_dir: str = "/workspace",
        env: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """
        Create a container (but don't start it).
        
        Returns:
            Container ID or None on failure
        """
        cmd = [
            self.docker_binary, "create",
            "--memory", self.memory_limit,
            "--cpus", str(self.cpu_limit),
            "--network", self.network,
            "-v", f"{os.path.abspath(repo_path)}:{working_dir}",
            "-w", working_dir,
        ]
        
        # Add environment variables
        if env:
            for key, value in env.items():
                cmd.extend(["-e", f"{key}={value}"])
        
        # Add the image and a sleep command to keep it running
        cmd.extend([self.image, "tail", "-f", "/dev/null"])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                return result.stdout.strip()[:12]  # Short container ID
            else:
                logger.error(f"Failed to create container: {result.stderr}")
                return None
        except Exception as e:
            logger.error(f"Exception creating container: {e}")
            return None
    
    def _start_container(self, container_id: str) -> bool:
        """Start a container."""
        try:
            result = subprocess.run(
                [self.docker_binary, "start", container_id],
                capture_output=True,
                timeout=30,
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def _exec_in_container(
        self,
        container_id: str,
        command: str,
        timeout: int = 300,
    ) -> Tuple[int, str, str]:
        """
        Execute a command inside a running container.
        
        Returns:
            (exit_code, stdout, stderr)
        """
        cmd = [
            self.docker_binary, "exec",
            container_id,
            "/bin/sh", "-c", command,
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            # Kill the container
            subprocess.run(
                [self.docker_binary, "kill", container_id],
                capture_output=True,
            )
            return -1, "", "TIMEOUT"
        except Exception as e:
            return -1, "", str(e)
    
    def _stop_container(self, container_id: str, timeout: int = 10):
        """Stop a container."""
        try:
            subprocess.run(
                [self.docker_binary, "stop", "-t", str(timeout), container_id],
                capture_output=True,
                timeout=timeout + 5,
            )
        except Exception:
            # Force kill
            subprocess.run(
                [self.docker_binary, "kill", container_id],
                capture_output=True,
            )
    
    def _remove_container(self, container_id: str):
        """Remove a container."""
        try:
            subprocess.run(
                [self.docker_binary, "rm", "-f", container_id],
                capture_output=True,
                timeout=10,
            )
        except Exception:
            pass
    
    def _check_oom(self, container_id: str) -> bool:
        """Check if container was killed due to OOM."""
        try:
            result = subprocess.run(
                [self.docker_binary, "inspect", container_id, "--format", "{{.State.OOMKilled}}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout.strip().lower() == "true"
        except Exception:
            return False
    
    def run_tests(
        self,
        repo_path: str,
        test_cmd: str = "pytest",
        timeout: int = 300,
        setup_cmd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> ContainerResult:
        """
        Run tests in an isolated container.
        
        Args:
            repo_path: Path to the repository to test
            test_cmd: Command to run tests
            timeout: Maximum execution time in seconds
            setup_cmd: Optional setup command (e.g., "pip install -r requirements.txt")
            env: Environment variables
            
        Returns:
            ContainerResult with test outcomes
        """
        start = time.time()
        
        # Check Docker availability
        if not self.is_available():
            return ContainerResult(
                status=ContainerStatus.ERROR,
                stderr="Docker is not available",
            )
        
        container_id = None
        
        try:
            # Create container
            container_id = self._create_container(repo_path, env=env)
            if not container_id:
                return ContainerResult(
                    status=ContainerStatus.ERROR,
                    stderr="Failed to create container",
                )
            
            # Start container
            if not self._start_container(container_id):
                return ContainerResult(
                    status=ContainerStatus.ERROR,
                    stderr="Failed to start container",
                    container_id=container_id,
                )
            
            # Run setup if provided
            if setup_cmd:
                exit_code, stdout, stderr = self._exec_in_container(
                    container_id, setup_cmd, timeout=120,
                )
                if exit_code != 0:
                    return ContainerResult(
                        status=ContainerStatus.FAILED,
                        exit_code=exit_code,
                        stdout=stdout,
                        stderr=f"Setup failed: {stderr}",
                        container_id=container_id,
                        duration_ms=(time.time() - start) * 1000,
                    )
            
            # Run tests
            exit_code, stdout, stderr = self._exec_in_container(
                container_id, test_cmd, timeout=timeout,
            )
            
            duration_ms = (time.time() - start) * 1000
            
            # Check for special conditions
            if stderr == "TIMEOUT":
                return ContainerResult(
                    status=ContainerStatus.TIMEOUT,
                    exit_code=-1,
                    stderr=f"Test execution timed out after {timeout}s",
                    container_id=container_id,
                    duration_ms=duration_ms,
                )
            
            if self._check_oom(container_id):
                return ContainerResult(
                    status=ContainerStatus.OOM,
                    exit_code=exit_code,
                    stdout=stdout,
                    stderr="Container killed: Out of Memory",
                    container_id=container_id,
                    duration_ms=duration_ms,
                )
            
            # Parse test results
            tests_passed, tests_failed, tests_total = self._parse_pytest_output(stdout + stderr)
            
            return ContainerResult(
                status=ContainerStatus.SUCCESS if exit_code == 0 else ContainerStatus.FAILED,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                container_id=container_id,
                duration_ms=duration_ms,
                tests_passed=tests_passed,
                tests_failed=tests_failed,
                tests_total=tests_total,
            )
            
        finally:
            # Cleanup
            if container_id:
                self._stop_container(container_id)
                self._remove_container(container_id)
    
    def _parse_pytest_output(self, output: str) -> Tuple[int, int, int]:
        """Parse pytest output to extract test counts."""
        import re
        
        passed = 0
        failed = 0
        
        # Look for "X passed" pattern
        passed_match = re.search(r'(\d+) passed', output)
        failed_match = re.search(r'(\d+) failed', output)
        
        if passed_match:
            passed = int(passed_match.group(1))
        if failed_match:
            failed = int(failed_match.group(1))
        
        return passed, failed, passed + failed
    
    def run_command(
        self,
        repo_path: str,
        command: str,
        timeout: int = 60,
        env: Optional[Dict[str, str]] = None,
    ) -> ContainerResult:
        """
        Run an arbitrary command in a container.
        
        Args:
            repo_path: Path to mount as /workspace
            command: Command to execute
            timeout: Maximum execution time
            env: Environment variables
            
        Returns:
            ContainerResult
        """
        return self.run_tests(
            repo_path=repo_path,
            test_cmd=command,
            timeout=timeout,
            env=env,
        )
    
    def build_image(
        self,
        dockerfile_path: str,
        image_name: str,
        build_args: Optional[Dict[str, str]] = None,
    ) -> bool:
        """
        Build a Docker image from a Dockerfile.
        
        Args:
            dockerfile_path: Path to directory containing Dockerfile
            image_name: Name for the built image
            build_args: Build arguments
            
        Returns:
            True if build succeeded
        """
        cmd = [
            self.docker_binary, "build",
            "-t", image_name,
        ]
        
        if build_args:
            for key, value in build_args.items():
                cmd.extend(["--build-arg", f"{key}={value}"])
        
        cmd.append(dockerfile_path)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=600,  # 10 minute build timeout
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Build failed: {e}")
            return False
    
    def cleanup_all(self):
        """Remove all S.P.I.D.E.R. containers."""
        try:
            # List containers with spider label
            result = subprocess.run(
                [
                    self.docker_binary, "ps", "-aq",
                    "--filter", "ancestor=" + self.image,
                ],
                capture_output=True,
                text=True,
            )
            
            container_ids = result.stdout.strip().split('\n')
            for cid in container_ids:
                if cid:
                    self._remove_container(cid)
                    
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    def __repr__(self) -> str:
        return f"DockerExecutor(image={self.image}, memory={self.memory_limit})"


# =============================================================================
# QUICK EXECUTOR (No Docker fallback)
# =============================================================================

class LocalExecutor:
    """
    Fallback executor that runs tests locally (without Docker).
    
    Use when Docker is not available or for fast iteration.
    """
    
    def run_tests(
        self,
        repo_path: str,
        test_cmd: str = "pytest",
        timeout: int = 300,
        env: Optional[Dict[str, str]] = None,
    ) -> ContainerResult:
        """Run tests locally using subprocess."""
        start = time.time()
        
        run_env = os.environ.copy()
        if env:
            run_env.update(env)
        
        try:
            result = subprocess.run(
                test_cmd,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=True,
                env=run_env,
            )
            
            duration_ms = (time.time() - start) * 1000
            
            # Parse output
            import re
            passed = 0
            failed = 0
            
            passed_match = re.search(r'(\d+) passed', result.stdout + result.stderr)
            failed_match = re.search(r'(\d+) failed', result.stdout + result.stderr)
            
            if passed_match:
                passed = int(passed_match.group(1))
            if failed_match:
                failed = int(failed_match.group(1))
            
            return ContainerResult(
                status=ContainerStatus.SUCCESS if result.returncode == 0 else ContainerStatus.FAILED,
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                duration_ms=duration_ms,
                tests_passed=passed,
                tests_failed=failed,
                tests_total=passed + failed,
            )
            
        except subprocess.TimeoutExpired:
            return ContainerResult(
                status=ContainerStatus.TIMEOUT,
                stderr=f"Timed out after {timeout}s",
                duration_ms=timeout * 1000,
            )
        except Exception as e:
            return ContainerResult(
                status=ContainerStatus.ERROR,
                stderr=str(e),
                duration_ms=(time.time() - start) * 1000,
            )


# =============================================================================
# FACTORY
# =============================================================================

def get_executor(prefer_docker: bool = True) -> DockerExecutor | LocalExecutor:
    """
    Get the best available executor.
    
    Args:
        prefer_docker: Try Docker first
        
    Returns:
        DockerExecutor if available, otherwise LocalExecutor
    """
    if prefer_docker:
        docker = DockerExecutor()
        if docker.is_available():
            return docker
        logger.warning("Docker not available, falling back to local executor")
    
    return LocalExecutor()


# =============================================================================
# DEMO
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🐳 S.P.I.D.E.R. DOCKER EXECUTOR - Demo")
    print("=" * 60)
    print()
    
    executor = DockerExecutor()
    
    print(f"Docker available: {executor.is_available()}")
    print(f"Image: {executor.image}")
    print(f"Memory limit: {executor.memory_limit}")
    print(f"CPU limit: {executor.cpu_limit}")
    print()
    
    if executor.is_available():
        print("Docker is available!")
        print("Run with: executor.run_tests('/path/to/repo', 'pytest')")
    else:
        print("Docker not available. Using LocalExecutor as fallback.")
        executor = LocalExecutor()
    
    print()
    print("=" * 60)
    print("✅ Executor ready!")
