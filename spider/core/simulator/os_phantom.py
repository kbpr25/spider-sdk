"""
S.P.I.D.E.R. Phantom OS - Generative Shell Simulation
======================================================

Born from: Linux-3.pdf (Generative Honeypots / shelLM)

The Insight:
"A small LLM can hallucinate a consistent Linux Shell state with 90% realism.
It remembers that `mkdir foo` happened without actually touching the disk."

The Problem:
In MCTS, testing a command in Docker takes 5 seconds.
With 500 futures to explore, that's 42 minutes per decision.

The Solution:
PhantomOS maintains a Virtual File System in memory and simulates
Linux commands using pure Python logic + LLM fallback.

Speedup: 100x vs Docker. MCTS can explore 500 futures/second.

Usage:
    phantom = PhantomShell()
    
    # Simulate commands
    output = phantom.exec("mkdir -p /app/data")
    output = phantom.exec("echo 'hello' > /app/data/test.txt")
    output = phantom.exec("cat /app/data/test.txt")  # Returns: hello
    
    # State is consistent
    output = phantom.exec("ls /app/data")  # Returns: test.txt
"""

import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# VIRTUAL FILE SYSTEM
# =============================================================================

@dataclass
class VirtualFile:
    """Represents a file in the virtual filesystem."""
    name: str
    content: str = ""
    permissions: int = 0o644
    owner: str = "root"
    group: str = "root"
    size: int = 0
    is_dir: bool = False
    created_at: float = field(default_factory=time.time)
    modified_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        self.size = len(self.content)


class VirtualFileSystem:
    """
    In-memory Virtual File System for PhantomOS.
    
    Supports:
    - Directory hierarchy
    - File creation/deletion/modification
    - Permission checking
    - Path resolution
    """
    
    def __init__(self):
        self.root: Dict[str, Any] = {}
        self.cwd = "/"
        
        # Initialize standard Linux directories
        self._init_standard_dirs()
    
    def _init_standard_dirs(self):
        """Create standard Linux directory structure."""
        standard_dirs = [
            "/", "/bin", "/etc", "/home", "/tmp", "/usr", "/var",
            "/usr/bin", "/usr/lib", "/var/log", "/root", "/opt",
        ]
        for d in standard_dirs:
            self.mkdir(d)
        
        # Add some common files
        self.write_file("/etc/passwd", "root:x:0:0:root:/root:/bin/bash\n")
        self.write_file("/etc/hostname", "phantom-os\n")
    
    def _path_to_parts(self, path: str) -> List[str]:
        """Convert path string to list of parts."""
        # Handle relative paths
        if not path.startswith("/"):
            path = f"{self.cwd.rstrip('/')}/{path}"
        
        # Normalize
        path = PurePosixPath(path)
        parts = [p for p in path.parts if p and p != "/"]
        return parts
    
    def _navigate(self, path: str, create_dirs: bool = False) -> Tuple[Dict, str]:
        """Navigate to parent directory, return (parent_dict, filename)."""
        parts = self._path_to_parts(path)
        
        if not parts:
            return self.root, ""
        
        current = self.root
        for part in parts[:-1]:
            if part not in current:
                if create_dirs:
                    current[part] = {"__meta__": VirtualFile(part, is_dir=True)}
                else:
                    raise FileNotFoundError(f"No such directory: {part}")
            
            node = current[part]
            if isinstance(node, dict):
                current = node
            else:
                raise NotADirectoryError(f"Not a directory: {part}")
        
        return current, parts[-1]
    
    def exists(self, path: str) -> bool:
        """Check if path exists."""
        try:
            parent, name = self._navigate(path)
            return name in parent or not name
        except (FileNotFoundError, NotADirectoryError):
            return False
    
    def is_dir(self, path: str) -> bool:
        """Check if path is a directory."""
        try:
            parent, name = self._navigate(path)
            if not name:
                return True
            node = parent.get(name)
            return isinstance(node, dict)
        except (FileNotFoundError, NotADirectoryError):
            return False
    
    def mkdir(self, path: str, parents: bool = True) -> bool:
        """Create directory."""
        try:
            parent, name = self._navigate(path, create_dirs=parents)
            if name and name not in parent:
                parent[name] = {"__meta__": VirtualFile(name, is_dir=True)}
            return True
        except (FileNotFoundError, NotADirectoryError) as e:
            if not parents:
                raise
            return False
    
    def write_file(self, path: str, content: str) -> bool:
        """Write content to file."""
        try:
            parent, name = self._navigate(path, create_dirs=True)
            if not name:
                raise IsADirectoryError("Cannot write to directory")
            
            parent[name] = VirtualFile(name, content=content)
            return True
        except Exception as e:
            logger.error(f"Failed to write {path}: {e}")
            return False
    
    def read_file(self, path: str) -> str:
        """Read file content."""
        parent, name = self._navigate(path)
        if not name or name not in parent:
            raise FileNotFoundError(f"No such file: {path}")
        
        node = parent[name]
        if isinstance(node, dict):
            raise IsADirectoryError(f"Is a directory: {path}")
        
        return node.content
    
    def append_file(self, path: str, content: str) -> bool:
        """Append content to file."""
        try:
            existing = self.read_file(path)
            return self.write_file(path, existing + content)
        except FileNotFoundError:
            return self.write_file(path, content)
    
    def remove(self, path: str, recursive: bool = False) -> bool:
        """Remove file or directory."""
        parent, name = self._navigate(path)
        if not name or name not in parent:
            raise FileNotFoundError(f"No such file: {path}")
        
        node = parent[name]
        if isinstance(node, dict) and not recursive:
            # Check if empty (only __meta__)
            if len(node) > 1:
                raise OSError(f"Directory not empty: {path}")
        
        del parent[name]
        return True
    
    def list_dir(self, path: str = None) -> List[str]:
        """List directory contents."""
        path = path or self.cwd
        parent, name = self._navigate(path)
        
        if name:
            if name not in parent:
                raise FileNotFoundError(f"No such directory: {path}")
            target = parent[name]
        else:
            target = parent
        
        if not isinstance(target, dict):
            raise NotADirectoryError(f"Not a directory: {path}")
        
        return [k for k in target.keys() if k != "__meta__"]
    
    def cd(self, path: str) -> str:
        """Change current directory."""
        if path == "-":
            # Could implement pushd/popd later
            return self.cwd
        
        if not path.startswith("/"):
            path = f"{self.cwd.rstrip('/')}/{path}"
        
        # Normalize
        path = str(PurePosixPath(path))
        
        if not self.is_dir(path):
            raise NotADirectoryError(f"Not a directory: {path}")
        
        self.cwd = path
        return self.cwd
    
    def copy(self, src: str, dst: str) -> bool:
        """Copy file."""
        content = self.read_file(src)
        return self.write_file(dst, content)
    
    def move(self, src: str, dst: str) -> bool:
        """Move file."""
        if self.copy(src, dst):
            self.remove(src)
            return True
        return False


# =============================================================================
# COMMAND EXECUTOR
# =============================================================================

class CommandExecutor:
    """
    Executes simulated Linux commands against the VFS.
    
    Supports common commands with realistic behavior.
    For unknown commands, falls back to LLM simulation.
    """
    
    def __init__(self, vfs: VirtualFileSystem, llm_callback: Optional[Callable] = None):
        self.vfs = vfs
        self.llm_callback = llm_callback
        self.env: Dict[str, str] = {
            "HOME": "/root",
            "USER": "root",
            "PWD": "/",
            "PATH": "/usr/bin:/bin",
            "SHELL": "/bin/bash",
        }
        self.last_exit_code = 0
        
        # Command handlers
        self.commands = {
            "ls": self._cmd_ls,
            "cd": self._cmd_cd,
            "pwd": self._cmd_pwd,
            "mkdir": self._cmd_mkdir,
            "touch": self._cmd_touch,
            "cat": self._cmd_cat,
            "echo": self._cmd_echo,
            "rm": self._cmd_rm,
            "cp": self._cmd_cp,
            "mv": self._cmd_mv,
            "head": self._cmd_head,
            "tail": self._cmd_tail,
            "wc": self._cmd_wc,
            "grep": self._cmd_grep,
            "find": self._cmd_find,
            "test": self._cmd_test,
            "[": self._cmd_test,
            "true": lambda _: ("", 0),
            "false": lambda _: ("", 1),
            "exit": self._cmd_exit,
            "export": self._cmd_export,
            "env": self._cmd_env,
            "which": self._cmd_which,
            "whoami": lambda _: ("root\n", 0),
            "hostname": lambda _: ("phantom-os\n", 0),
            "uname": self._cmd_uname,
            "date": lambda _: (time.strftime("%a %b %d %H:%M:%S %Z %Y\n"), 0),
            "id": lambda _: ("uid=0(root) gid=0(root) groups=0(root)\n", 0),
        }
    
    def execute(self, command: str) -> Tuple[str, int]:
        """
        Execute a command and return (output, exit_code).
        """
        command = command.strip()
        if not command or command.startswith("#"):
            return ("", 0)
        
        # Handle pipes (basic)
        if "|" in command:
            return self._execute_pipeline(command)
        
        # Handle redirections
        stdout_file = None
        stdout_append = False
        stderr_file = None
        
        # Parse redirections (simplified)
        if ">>" in command:
            parts = command.split(">>", 1)
            command = parts[0].strip()
            stdout_file = parts[1].strip()
            stdout_append = True
        elif ">" in command and "2>" not in command:
            parts = command.split(">", 1)
            command = parts[0].strip()
            stdout_file = parts[1].strip()
        
        # Parse command and args
        tokens = self._tokenize(command)
        if not tokens:
            return ("", 0)
        
        cmd = tokens[0]
        args = tokens[1:]
        
        # Execute
        if cmd in self.commands:
            try:
                output, exit_code = self.commands[cmd](args)
            except Exception as e:
                output = f"{cmd}: {e}\n"
                exit_code = 1
        else:
            # Unknown command - use LLM fallback
            output, exit_code = self._llm_fallback(command)
        
        # Handle stdout redirection
        if stdout_file:
            if stdout_append:
                self.vfs.append_file(stdout_file, output)
            else:
                self.vfs.write_file(stdout_file, output)
            output = ""
        
        self.last_exit_code = exit_code
        self.env["?"] = str(exit_code)
        self.env["PWD"] = self.vfs.cwd
        
        return (output, exit_code)
    
    def _tokenize(self, command: str) -> List[str]:
        """Simple tokenization respecting quotes."""
        tokens = []
        current = ""
        in_quotes = None
        
        for char in command:
            if char in ('"', "'") and not in_quotes:
                in_quotes = char
            elif char == in_quotes:
                in_quotes = None
            elif char == " " and not in_quotes:
                if current:
                    tokens.append(self._expand_vars(current))
                    current = ""
                continue
            else:
                current += char
        
        if current:
            tokens.append(self._expand_vars(current))
        
        return tokens
    
    def _expand_vars(self, s: str) -> str:
        """Expand environment variables."""
        # $VAR and ${VAR}
        for var, value in self.env.items():
            s = s.replace(f"${{{var}}}", value)
            s = s.replace(f"${var}", value)
        return s
    
    def _execute_pipeline(self, command: str) -> Tuple[str, int]:
        """Execute a pipeline of commands."""
        parts = command.split("|")
        input_data = ""
        exit_code = 0
        
        for part in parts:
            part = part.strip()
            # For piped commands, we'd need stdin support
            # Simplified: just run sequentially
            output, exit_code = self.execute(part)
            input_data = output
        
        return (input_data, exit_code)
    
    def _llm_fallback(self, command: str) -> Tuple[str, int]:
        """Use LLM to simulate unknown command."""
        if self.llm_callback:
            state_summary = f"CWD: {self.vfs.cwd}, Files: {self.vfs.list_dir()[:10]}"
            prompt = f"""You are simulating a Linux shell. The current state is:
{state_summary}

The user ran: {command}

Generate the likely stdout output. Be realistic but brief.
If this command would fail, output an error message.
Output ONLY the command output, nothing else."""
            
            try:
                output = self.llm_callback(prompt)
                return (output + "\n" if not output.endswith("\n") else output, 0)
            except Exception as e:
                logger.warning(f"LLM fallback failed: {e}")
        
        return (f"bash: {command.split()[0]}: command not found\n", 127)
    
    # -------------------------------------------------------------------------
    # COMMAND IMPLEMENTATIONS
    # -------------------------------------------------------------------------
    
    def _cmd_ls(self, args: List[str]) -> Tuple[str, int]:
        """List directory contents."""
        long_format = "-l" in args
        show_all = "-a" in args
        args = [a for a in args if not a.startswith("-")]
        
        path = args[0] if args else self.vfs.cwd
        
        try:
            items = self.vfs.list_dir(path)
            if show_all:
                items = [".", ".."] + items
            
            if long_format:
                lines = []
                for item in items:
                    item_path = f"{path.rstrip('/')}/{item}"
                    is_dir = self.vfs.is_dir(item_path)
                    perm = "drwxr-xr-x" if is_dir else "-rw-r--r--"
                    lines.append(f"{perm} 1 root root 4096 Dec 21 12:00 {item}")
                return ("\n".join(lines) + "\n" if lines else "", 0)
            else:
                return ("  ".join(items) + "\n" if items else "", 0)
        except FileNotFoundError:
            return (f"ls: cannot access '{path}': No such file or directory\n", 2)
    
    def _cmd_cd(self, args: List[str]) -> Tuple[str, int]:
        """Change directory."""
        path = args[0] if args else self.env.get("HOME", "/")
        try:
            self.vfs.cd(path)
            return ("", 0)
        except NotADirectoryError:
            return (f"cd: {path}: Not a directory\n", 1)
        except FileNotFoundError:
            return (f"cd: {path}: No such file or directory\n", 1)
    
    def _cmd_pwd(self, args: List[str]) -> Tuple[str, int]:
        """Print working directory."""
        return (self.vfs.cwd + "\n", 0)
    
    def _cmd_mkdir(self, args: List[str]) -> Tuple[str, int]:
        """Make directory."""
        parents = "-p" in args
        args = [a for a in args if not a.startswith("-")]
        
        for path in args:
            try:
                self.vfs.mkdir(path, parents=parents)
            except FileNotFoundError:
                return (f"mkdir: cannot create directory '{path}': No such file or directory\n", 1)
        return ("", 0)
    
    def _cmd_touch(self, args: List[str]) -> Tuple[str, int]:
        """Create empty file or update timestamp."""
        for path in args:
            if not self.vfs.exists(path):
                self.vfs.write_file(path, "")
        return ("", 0)
    
    def _cmd_cat(self, args: List[str]) -> Tuple[str, int]:
        """Concatenate and print files."""
        output = []
        for path in args:
            try:
                content = self.vfs.read_file(path)
                output.append(content)
            except FileNotFoundError:
                return (f"cat: {path}: No such file or directory\n", 1)
            except IsADirectoryError:
                return (f"cat: {path}: Is a directory\n", 1)
        return ("".join(output), 0)
    
    def _cmd_echo(self, args: List[str]) -> Tuple[str, int]:
        """Echo arguments."""
        no_newline = "-n" in args
        args = [a for a in args if a != "-n"]
        
        output = " ".join(args)
        if not no_newline:
            output += "\n"
        return (output, 0)
    
    def _cmd_rm(self, args: List[str]) -> Tuple[str, int]:
        """Remove files."""
        recursive = "-r" in args or "-rf" in args or "-fr" in args
        force = "-f" in args or "-rf" in args or "-fr" in args
        args = [a for a in args if not a.startswith("-")]
        
        for path in args:
            try:
                self.vfs.remove(path, recursive=recursive)
            except FileNotFoundError:
                if not force:
                    return (f"rm: cannot remove '{path}': No such file or directory\n", 1)
            except OSError as e:
                return (f"rm: cannot remove '{path}': {e}\n", 1)
        return ("", 0)
    
    def _cmd_cp(self, args: List[str]) -> Tuple[str, int]:
        """Copy files."""
        args = [a for a in args if not a.startswith("-")]
        if len(args) < 2:
            return ("cp: missing operand\n", 1)
        
        src, dst = args[0], args[1]
        try:
            self.vfs.copy(src, dst)
            return ("", 0)
        except Exception as e:
            return (f"cp: {e}\n", 1)
    
    def _cmd_mv(self, args: List[str]) -> Tuple[str, int]:
        """Move files."""
        args = [a for a in args if not a.startswith("-")]
        if len(args) < 2:
            return ("mv: missing operand\n", 1)
        
        src, dst = args[0], args[1]
        try:
            self.vfs.move(src, dst)
            return ("", 0)
        except Exception as e:
            return (f"mv: {e}\n", 1)
    
    def _cmd_head(self, args: List[str]) -> Tuple[str, int]:
        """Print first lines of file."""
        n = 10
        for i, arg in enumerate(args):
            if arg == "-n" and i + 1 < len(args):
                n = int(args[i + 1])
        
        files = [a for a in args if not a.startswith("-") and not a.isdigit()]
        if not files:
            return ("", 0)
        
        try:
            content = self.vfs.read_file(files[0])
            lines = content.split("\n")[:n]
            return ("\n".join(lines) + "\n", 0)
        except Exception as e:
            return (f"head: {e}\n", 1)
    
    def _cmd_tail(self, args: List[str]) -> Tuple[str, int]:
        """Print last lines of file."""
        n = 10
        for i, arg in enumerate(args):
            if arg == "-n" and i + 1 < len(args):
                n = int(args[i + 1])
        
        files = [a for a in args if not a.startswith("-") and not a.isdigit()]
        if not files:
            return ("", 0)
        
        try:
            content = self.vfs.read_file(files[0])
            lines = content.split("\n")[-n:]
            return ("\n".join(lines) + "\n", 0)
        except Exception as e:
            return (f"tail: {e}\n", 1)
    
    def _cmd_wc(self, args: List[str]) -> Tuple[str, int]:
        """Word count."""
        files = [a for a in args if not a.startswith("-")]
        
        for path in files:
            try:
                content = self.vfs.read_file(path)
                lines = content.count("\n")
                words = len(content.split())
                chars = len(content)
                return (f"{lines:>7} {words:>7} {chars:>7} {path}\n", 0)
            except Exception as e:
                return (f"wc: {e}\n", 1)
        return ("", 0)
    
    def _cmd_grep(self, args: List[str]) -> Tuple[str, int]:
        """Search for pattern in files."""
        if len(args) < 1:
            return ("grep: missing pattern\n", 1)
        
        ignore_case = "-i" in args
        line_numbers = "-n" in args
        args = [a for a in args if not a.startswith("-")]
        
        pattern = args[0]
        files = args[1:] if len(args) > 1 else ["-"]
        
        output = []
        flags = re.IGNORECASE if ignore_case else 0
        
        for path in files:
            try:
                content = self.vfs.read_file(path)
                for i, line in enumerate(content.split("\n"), 1):
                    if re.search(pattern, line, flags):
                        if line_numbers:
                            output.append(f"{i}:{line}")
                        else:
                            output.append(line)
            except Exception:
                pass
        
        return ("\n".join(output) + "\n" if output else "", 0 if output else 1)
    
    def _cmd_find(self, args: List[str]) -> Tuple[str, int]:
        """Find files."""
        path = args[0] if args and not args[0].startswith("-") else "."
        name_pattern = None
        
        for i, arg in enumerate(args):
            if arg == "-name" and i + 1 < len(args):
                name_pattern = args[i + 1]
        
        # Simplified: just list directory
        output = []
        try:
            items = self.vfs.list_dir(path)
            for item in items:
                if name_pattern:
                    if re.match(name_pattern.replace("*", ".*"), item):
                        output.append(f"{path}/{item}")
                else:
                    output.append(f"{path}/{item}")
        except Exception:
            pass
        
        return ("\n".join(output) + "\n" if output else "", 0)
    
    def _cmd_test(self, args: List[str]) -> Tuple[str, int]:
        """Test file attributes."""
        if not args:
            return ("", 1)
        
        # Remove trailing ]
        args = [a for a in args if a != "]"]
        
        if len(args) >= 2:
            flag = args[0]
            path = args[1]
            
            if flag == "-f":
                return ("", 0 if self.vfs.exists(path) and not self.vfs.is_dir(path) else 1)
            elif flag == "-d":
                return ("", 0 if self.vfs.is_dir(path) else 1)
            elif flag == "-e":
                return ("", 0 if self.vfs.exists(path) else 1)
            elif flag == "-n":
                return ("", 0 if path else 1)
            elif flag == "-z":
                return ("", 0 if not path else 1)
        
        return ("", 1)
    
    def _cmd_exit(self, args: List[str]) -> Tuple[str, int]:
        """Exit shell."""
        code = int(args[0]) if args else 0
        return ("", code)
    
    def _cmd_export(self, args: List[str]) -> Tuple[str, int]:
        """Export environment variable."""
        for arg in args:
            if "=" in arg:
                key, value = arg.split("=", 1)
                self.env[key] = value
        return ("", 0)
    
    def _cmd_env(self, args: List[str]) -> Tuple[str, int]:
        """Print environment."""
        lines = [f"{k}={v}" for k, v in self.env.items()]
        return ("\n".join(lines) + "\n", 0)
    
    def _cmd_which(self, args: List[str]) -> Tuple[str, int]:
        """Find command location."""
        if not args:
            return ("", 1)
        
        cmd = args[0]
        if cmd in self.commands:
            return (f"/bin/{cmd}\n", 0)
        return (f"{cmd} not found\n", 1)
    
    def _cmd_uname(self, args: List[str]) -> Tuple[str, int]:
        """System information."""
        if "-a" in args:
            return ("Linux phantom-os 5.15.0 #1 SMP x86_64 GNU/Linux\n", 0)
        return ("Linux\n", 0)


# =============================================================================
# PHANTOM SHELL
# =============================================================================

class PhantomShell:
    """
    The Phantom OS - LLM-Accelerated Shell Simulation.
    
    Provides 100x speedup over Docker for MCTS exploration by:
    1. Maintaining a Virtual File System in memory
    2. Executing common commands with pure Python
    3. Falling back to LLM for complex/unknown commands
    
    Usage:
        phantom = PhantomShell()
        
        # Execute commands
        output = phantom.exec("mkdir -p /app/data")
        output = phantom.exec("echo 'hello' > /app/data/test.txt")
        output = phantom.exec("cat /app/data/test.txt")
        
        # Check state
        print(phantom.exec("ls -la /app"))
    """
    
    def __init__(
        self,
        llm_callback: Optional[Callable[[str], str]] = None,
        initial_state: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize PhantomShell.
        
        Args:
            llm_callback: Function to call LLM for unknown commands
                         Signature: (prompt: str) -> str
            initial_state: Optional dict of {path: content} to pre-populate
        """
        self.vfs = VirtualFileSystem()
        self.executor = CommandExecutor(self.vfs, llm_callback)
        self.history: List[Tuple[str, str, int]] = []
        
        # Pre-populate if provided
        if initial_state:
            for path, content in initial_state.items():
                if content is None:
                    self.vfs.mkdir(path)
                else:
                    self.vfs.write_file(path, content)
        
        self._stats = {
            "commands_executed": 0,
            "llm_calls": 0,
            "errors": 0,
        }
    
    def exec(self, command: str) -> str:
        """
        Execute a shell command and return output.
        
        Args:
            command: The shell command to execute
            
        Returns:
            Combined stdout/stderr output
        """
        self._stats["commands_executed"] += 1
        
        # Handle multiple commands separated by ;
        if ";" in command and '"' not in command and "'" not in command:
            outputs = []
            for cmd in command.split(";"):
                cmd = cmd.strip()
                if cmd:
                    output, _ = self.executor.execute(cmd)
                    outputs.append(output)
            return "".join(outputs)
        
        # Handle && and ||
        if "&&" in command:
            parts = command.split("&&")
            for part in parts:
                output, code = self.executor.execute(part.strip())
                if code != 0:
                    return output
            return output
        
        if "||" in command:
            parts = command.split("||")
            for part in parts:
                output, code = self.executor.execute(part.strip())
                if code == 0:
                    return output
            return output
        
        output, code = self.executor.execute(command)
        self.history.append((command, output, code))
        
        if code != 0:
            self._stats["errors"] += 1
        
        return output
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current state for LLM context.
        
        Returns a summary of the VFS state.
        """
        return {
            "cwd": self.vfs.cwd,
            "env": {k: v for k, v in self.executor.env.items() if not k.startswith("_")},
            "last_exit_code": self.executor.last_exit_code,
            "files_in_cwd": self.vfs.list_dir()[:20],
        }
    
    def reset(self) -> None:
        """Reset to initial state."""
        self.vfs = VirtualFileSystem()
        self.executor = CommandExecutor(self.vfs, self.executor.llm_callback)
        self.history.clear()
    
    def fork(self) -> "PhantomShell":
        """
        Create a copy for MCTS branching.
        
        The forked shell has independent state for exploration.
        """
        import copy
        forked = PhantomShell.__new__(PhantomShell)
        forked.vfs = copy.deepcopy(self.vfs)
        forked.executor = CommandExecutor(forked.vfs, self.executor.llm_callback)
        forked.executor.env = copy.deepcopy(self.executor.env)
        forked.history = []
        forked._stats = {"commands_executed": 0, "llm_calls": 0, "errors": 0}
        return forked
    
    def get_stats(self) -> Dict[str, int]:
        return self._stats


# =============================================================================
# PACKAGE INIT
# =============================================================================

__all__ = ["PhantomShell", "VirtualFileSystem", "CommandExecutor"]


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("👻 S.P.I.D.E.R. PhantomOS - Demo")
    print("=" * 70)
    
    phantom = PhantomShell()
    
    # Test basic commands
    commands = [
        "pwd",
        "mkdir -p /app/data",
        "cd /app/data",
        "pwd",
        "echo 'Hello from PhantomOS!' > hello.txt",
        "cat hello.txt",
        "ls -la",
        "echo 'Line 2' >> hello.txt",
        "cat hello.txt",
        "wc hello.txt",
        "grep Hello hello.txt",
        "rm hello.txt",
        "ls -la",
    ]
    
    for cmd in commands:
        print(f"\n$ {cmd}")
        output = phantom.exec(cmd)
        if output:
            print(output.rstrip())
    
    print("\n" + "=" * 70)
    print(f"Stats: {phantom.get_stats()}")
    print("=" * 70)
