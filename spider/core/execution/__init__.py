"""
S.P.I.D.E.R. Execution Module - Iron Interface Components
==========================================================

Production-grade integration with Git, Docker, and Test Runners.
"""

from .git_ops import GitOperator, OperationResult, TestResult, PatchResult
from .docker_exec import DockerExecutor, LocalExecutor, ContainerResult, ContainerStatus, get_executor

# Container module (requires docker SDK)
try:
    from .container import ContainerSession, ContainerPool, ExecResult, SessionStatus, run_in_container
    CONTAINER_AVAILABLE = True
except ImportError:
    CONTAINER_AVAILABLE = False
    ContainerSession = None
    ContainerPool = None

__all__ = [
    # Git
    'GitOperator',
    'OperationResult',
    'TestResult',
    'PatchResult',
    # Docker (subprocess-based)
    'DockerExecutor',
    'LocalExecutor',
    'ContainerResult',
    'ContainerStatus',
    'get_executor',
    # Container (SDK-based)
    'ContainerSession',
    'ContainerPool',
    'ExecResult',
    'SessionStatus',
    'run_in_container',
    'CONTAINER_AVAILABLE',
]

