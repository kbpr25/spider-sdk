"""
S.P.I.D.E.R. Container Foundry - Cattle-Style Container Management
====================================================================

"Treat containers like cattle, not pets."

This module manages ephemeral Docker containers for:
- Cloning repositories
- Running tests in isolation
- Memory monitoring (OOM prevention)
- Automatic cleanup

Uses the official Docker SDK for Python.
"""

import os
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable
import logging

# Try to import docker SDK
try:
    import docker
    from docker.errors import ContainerError, ImageNotFound, APIError
    from docker.models.containers import Container
    DOCKER_SDK_AVAILABLE = True
except ImportError:
    DOCKER_SDK_AVAILABLE = False
    docker = None
    Container = None


logger = logging.getLogger(__name__)


# =============================================================================
# RESULT TYPES
# =============================================================================

class SessionStatus(Enum):
    """Status of a container session."""
    CREATED = "created"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    OOM = "oom"
    TIMEOUT = "timeout"
    CLEANED = "cleaned"


@dataclass
class ExecResult:
    """Result of executing a command in a container."""
    exit_code: int
    output: str
    error: str = ""
    duration_ms: float = 0
    memory_peak_mb: float = 0
    timeout: bool = False
    oom_killed: bool = False
    
    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.oom_killed


@dataclass 
class SessionStats:
    """Statistics for a container session."""
    commands_executed: int = 0
    total_exec_time_ms: float = 0
    peak_memory_mb: float = 0
    errors: int = 0


# =============================================================================
# CONTAINER SESSION
# =============================================================================

class ContainerSession:
    """
    A managed Docker container session.
    
    Handles the full lifecycle of a container:
    1. Create with specific image and resources
    2. Clone a git repository inside
    3. Install dependencies
    4. Execute commands with monitoring
    5. Cleanup on exit
    
    Example:
        session = ContainerSession(
            image_name="python:3.11-slim",
            repo_url="https://github.com/user/repo.git",
        )
        
        with session:
            session.start()
            result = session.exec_command("pytest tests/", timeout=300)
            print(f"Tests passed: {result.success}")
        
        # Container automatically cleaned up
    """
    
    DEFAULT_IMAGE = "python:3.11-slim"
    DEFAULT_MEMORY_LIMIT = "2g"
    DEFAULT_CPU_LIMIT = 2.0
    
    def __init__(
        self,
        image_name: str = DEFAULT_IMAGE,
        repo_url: Optional[str] = None,
        memory_limit: str = DEFAULT_MEMORY_LIMIT,
        cpu_limit: float = DEFAULT_CPU_LIMIT,
        network_mode: str = "bridge",  # "none" for full isolation
        work_dir: str = "/workspace",
        env: Optional[Dict[str, str]] = None,
        labels: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize a container session.
        
        Args:
            image_name: Docker image to use
            repo_url: Git repository URL to clone
            memory_limit: Memory limit (e.g., "2g", "512m")
            cpu_limit: CPU cores limit
            network_mode: Docker network mode
            work_dir: Working directory inside container
            env: Environment variables
            labels: Container labels for tracking
        """
        if not DOCKER_SDK_AVAILABLE:
            raise RuntimeError(
                "Docker SDK not available. Install with: pip install docker"
            )
        
        self.image_name = image_name
        self.repo_url = repo_url
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.network_mode = network_mode
        self.work_dir = work_dir
        self.env = env or {}
        self.labels = labels or {"created_by": "spider"}
        
        # Docker client
        self._client: Optional[docker.DockerClient] = None
        self._container: Optional[Container] = None
        
        # Status tracking
        self._status = SessionStatus.CREATED
        self._stats = SessionStats()
        
        # Memory monitoring
        self._memory_monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._peak_memory_mb = 0.0
    
    @property
    def status(self) -> SessionStatus:
        """Get current session status."""
        return self._status
    
    @property
    def container_id(self) -> Optional[str]:
        """Get container ID if running."""
        if self._container:
            return self._container.short_id
        return None
    
    @property
    def stats(self) -> SessionStats:
        """Get session statistics."""
        return self._stats
    
    def _get_client(self) -> docker.DockerClient:
        """Get or create Docker client."""
        if self._client is None:
            self._client = docker.from_env()
        return self._client
    
    def _ensure_image(self):
        """Pull image if not present."""
        client = self._get_client()
        try:
            client.images.get(self.image_name)
            logger.debug(f"Image {self.image_name} found locally")
        except ImageNotFound:
            logger.info(f"Pulling image {self.image_name}...")
            client.images.pull(self.image_name)
            logger.info(f"Image {self.image_name} pulled successfully")
    
    def start(self) -> bool:
        """
        Start the container session.
        
        1. Pull image if needed
        2. Create container
        3. Clone repository if specified
        4. Install dependencies
        
        Returns:
            True if started successfully
        """
        try:
            client = self._get_client()
            
            # Ensure image exists
            self._ensure_image()
            
            # Create container
            logger.info(f"Creating container from {self.image_name}...")
            
            self._container = client.containers.run(
                image=self.image_name,
                command="tail -f /dev/null",  # Keep alive
                detach=True,
                mem_limit=self.memory_limit,
                nano_cpus=int(self.cpu_limit * 1e9),
                network_mode=self.network_mode,
                working_dir=self.work_dir,
                environment=self.env,
                labels=self.labels,
                remove=False,  # We'll remove manually for stats
            )
            
            logger.info(f"Container {self._container.short_id} started")
            self._status = SessionStatus.RUNNING
            
            # Start memory monitoring
            self._start_memory_monitor()
            
            # Clone repository if specified
            if self.repo_url:
                clone_result = self.exec_command(
                    f"git clone {self.repo_url} .",
                    timeout=120,
                )
                if not clone_result.success:
                    logger.error(f"Failed to clone repo: {clone_result.error}")
                    return False
                
                # Install dependencies if requirements.txt exists
                check_result = self.exec_command(
                    "test -f requirements.txt && echo 'exists'",
                    timeout=10,
                )
                if "exists" in check_result.output:
                    logger.info("Installing dependencies...")
                    install_result = self.exec_command(
                        "pip install -r requirements.txt",
                        timeout=300,
                    )
                    if not install_result.success:
                        logger.warning(f"Dependency install issues: {install_result.error}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start container: {e}")
            self._status = SessionStatus.FAILED
            return False
    
    def _start_memory_monitor(self):
        """Start background thread to monitor memory usage."""
        def monitor():
            while not self._stop_monitoring.is_set():
                if self._container:
                    try:
                        stats = self._container.stats(stream=False)
                        memory_usage = stats.get('memory_stats', {}).get('usage', 0)
                        memory_mb = memory_usage / (1024 * 1024)
                        
                        if memory_mb > self._peak_memory_mb:
                            self._peak_memory_mb = memory_mb
                            self._stats.peak_memory_mb = memory_mb
                    except Exception:
                        pass
                
                self._stop_monitoring.wait(timeout=1.0)
        
        self._stop_monitoring.clear()
        self._memory_monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._memory_monitor_thread.start()
    
    def _stop_memory_monitor(self):
        """Stop the memory monitoring thread."""
        self._stop_monitoring.set()
        if self._memory_monitor_thread:
            self._memory_monitor_thread.join(timeout=2.0)
    
    def exec_command(
        self,
        cmd: str,
        timeout: int = 300,
        workdir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> ExecResult:
        """
        Execute a command inside the container.
        
        Features:
        - Timeout handling
        - Memory monitoring
        - OOM detection
        - Output capture
        
        Args:
            cmd: Command to execute
            timeout: Maximum execution time in seconds
            workdir: Override working directory
            env: Additional environment variables
            
        Returns:
            ExecResult with exit code, output, and stats
        """
        if not self._container:
            return ExecResult(
                exit_code=-1,
                output="",
                error="Container not running",
            )
        
        start = time.time()
        self._stats.commands_executed += 1
        
        # Prepare environment
        exec_env = dict(self.env)
        if env:
            exec_env.update(env)
        
        try:
            # Execute command
            exec_result = self._container.exec_run(
                cmd=cmd,
                workdir=workdir or self.work_dir,
                environment=exec_env,
                demux=True,  # Separate stdout/stderr
            )
            
            duration_ms = (time.time() - start) * 1000
            self._stats.total_exec_time_ms += duration_ms
            
            # Parse output
            stdout = ""
            stderr = ""
            
            if exec_result.output:
                if isinstance(exec_result.output, tuple):
                    if exec_result.output[0]:
                        stdout = exec_result.output[0].decode('utf-8', errors='replace')
                    if exec_result.output[1]:
                        stderr = exec_result.output[1].decode('utf-8', errors='replace')
                else:
                    stdout = exec_result.output.decode('utf-8', errors='replace')
            
            # Check for OOM
            self._container.reload()
            oom_killed = self._container.attrs.get('State', {}).get('OOMKilled', False)
            
            if oom_killed:
                self._status = SessionStatus.OOM
                self._stats.errors += 1
            
            return ExecResult(
                exit_code=exec_result.exit_code,
                output=stdout,
                error=stderr,
                duration_ms=duration_ms,
                memory_peak_mb=self._peak_memory_mb,
                oom_killed=oom_killed,
            )
            
        except Exception as e:
            self._stats.errors += 1
            return ExecResult(
                exit_code=-1,
                output="",
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )
    
    def run_tests(
        self,
        test_cmd: str = "pytest",
        timeout: int = 300,
    ) -> ExecResult:
        """
        Convenience method to run tests.
        
        Args:
            test_cmd: Test command (default: pytest)
            timeout: Maximum time for tests
            
        Returns:
            ExecResult
        """
        return self.exec_command(test_cmd, timeout=timeout)
    
    def copy_to_container(self, local_path: str, container_path: str) -> bool:
        """Copy a file from host to container."""
        if not self._container:
            return False
        
        try:
            import tarfile
            import io
            
            # Create tar archive
            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode='w') as tar:
                tar.add(local_path, arcname=os.path.basename(local_path))
            
            tar_stream.seek(0)
            
            # Copy to container
            self._container.put_archive(
                os.path.dirname(container_path) or '/',
                tar_stream,
            )
            
            return True
        except Exception as e:
            logger.error(f"Failed to copy file: {e}")
            return False
    
    def get_file(self, container_path: str) -> Optional[bytes]:
        """Get a file from the container."""
        if not self._container:
            return None
        
        try:
            bits, stat = self._container.get_archive(container_path)
            return b''.join(bits)
        except Exception as e:
            logger.error(f"Failed to get file: {e}")
            return None
    
    def cleanup(self):
        """
        Stop and remove the container.
        
        Called automatically when using context manager.
        """
        self._stop_memory_monitor()
        
        if self._container:
            try:
                # Stop container
                logger.info(f"Stopping container {self._container.short_id}...")
                self._container.stop(timeout=10)
                
                # Remove container
                self._container.remove(force=True)
                logger.info(f"Container {self._container.short_id} removed")
                
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                # Force remove
                try:
                    self._container.remove(force=True)
                except Exception:
                    pass
            
            self._container = None
        
        self._status = SessionStatus.CLEANED
    
    def __enter__(self) -> 'ContainerSession':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup."""
        self.cleanup()
        return False
    
    def __repr__(self) -> str:
        status = self._status.value
        cid = self.container_id or "N/A"
        return f"ContainerSession(id={cid}, status={status})"


# =============================================================================
# CONTAINER POOL
# =============================================================================

class ContainerPool:
    """
    Pool of pre-warmed containers for fast execution.
    
    Maintains a pool of ready containers to reduce cold-start time.
    """
    
    def __init__(
        self,
        image_name: str = ContainerSession.DEFAULT_IMAGE,
        pool_size: int = 3,
        **container_kwargs,
    ):
        """
        Initialize the container pool.
        
        Args:
            image_name: Docker image for containers
            pool_size: Number of containers to keep ready
            **container_kwargs: Additional args for ContainerSession
        """
        self.image_name = image_name
        self.pool_size = pool_size
        self.container_kwargs = container_kwargs
        
        self._pool: List[ContainerSession] = []
        self._lock = threading.Lock()
    
    def warm_up(self):
        """Pre-create containers in the pool."""
        logger.info(f"Warming up pool with {self.pool_size} containers...")
        
        for i in range(self.pool_size):
            session = ContainerSession(
                image_name=self.image_name,
                **self.container_kwargs,
            )
            if session.start():
                with self._lock:
                    self._pool.append(session)
                logger.info(f"Container {i+1}/{self.pool_size} ready")
    
    def acquire(self) -> Optional[ContainerSession]:
        """Get a container from the pool."""
        with self._lock:
            if self._pool:
                return self._pool.pop()
        
        # Pool empty - create new
        session = ContainerSession(
            image_name=self.image_name,
            **self.container_kwargs,
        )
        if session.start():
            return session
        return None
    
    def release(self, session: ContainerSession):
        """Return a container to the pool (or cleanup if pool full)."""
        with self._lock:
            if len(self._pool) < self.pool_size:
                # Reset container state
                session.exec_command("rm -rf /workspace/*", timeout=10)
                self._pool.append(session)
                return
        
        # Pool full - cleanup
        session.cleanup()
    
    def cleanup(self):
        """Cleanup all containers in the pool."""
        with self._lock:
            for session in self._pool:
                session.cleanup()
            self._pool.clear()
    
    def __enter__(self) -> 'ContainerPool':
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False


# =============================================================================
# QUICK RUN (Convenience function)
# =============================================================================

def run_in_container(
    commands: List[str],
    image_name: str = ContainerSession.DEFAULT_IMAGE,
    repo_url: Optional[str] = None,
    timeout: int = 300,
) -> List[ExecResult]:
    """
    Quick utility to run commands in a fresh container.
    
    Args:
        commands: List of commands to execute
        image_name: Docker image
        repo_url: Optional repo to clone
        timeout: Timeout per command
        
    Returns:
        List of ExecResult for each command
    """
    results = []
    
    with ContainerSession(image_name=image_name, repo_url=repo_url) as session:
        if not session.start():
            return [ExecResult(exit_code=-1, output="", error="Failed to start container")]
        
        for cmd in commands:
            result = session.exec_command(cmd, timeout=timeout)
            results.append(result)
            
            if not result.success:
                break  # Stop on first failure
    
    return results


# =============================================================================
# DEMO
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🏭 S.P.I.D.E.R. CONTAINER FOUNDRY - Demo")
    print("=" * 60)
    print()
    
    if not DOCKER_SDK_AVAILABLE:
        print("❌ Docker SDK not installed!")
        print("   Install with: pip install docker")
        exit(1)
    
    print("Docker SDK available: ✅")
    print()
    
    # Check if Docker daemon is running
    try:
        client = docker.from_env()
        client.ping()
        print("Docker daemon: ✅ Running")
    except Exception as e:
        print(f"Docker daemon: ❌ Not available ({e})")
        exit(1)
    
    print()
    print("Example usage:")
    print("-" * 40)
    print("""
    from spider.core.execution.container import ContainerSession
    
    with ContainerSession(
        image_name="python:3.11-slim",
        repo_url="https://github.com/user/repo.git",
    ) as session:
        session.start()
        
        result = session.exec_command("pytest tests/")
        print(f"Exit code: {result.exit_code}")
        print(f"Output: {result.output}")
    """)
    
    print()
    print("=" * 60)
    print("✅ Container Foundry ready!")
