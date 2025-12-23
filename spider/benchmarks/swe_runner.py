#!/usr/bin/env python3
"""
S.P.I.D.E.R. SWE-Bench Runner - Production Mode
=================================================

Run S.P.I.D.E.R. on SWE-Bench tasks with full production pipeline.

Usage:
    # Single task
    python swe_runner.py --task tasks/django-12345.json
    
    # Multiple tasks
    python swe_runner.py --tasks-dir tasks/ --output predictions.json
    
    # With budget limit
    python swe_runner.py --tasks-dir tasks/ --max-cost 1.00
    
    # Quick demo
    python swe_runner.py --demo
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from spider.benchmarks.swe_pipeline import (
    SWEBenchTask,
    SWEBenchSolver,
    SolverConfig,
    TestResultParser,
    PatchGenerator,
)


# =============================================================================
# COLORS
# =============================================================================

class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"


def style(text: str, *styles: str) -> str:
    return f"{''.join(styles)}{text}{Colors.RESET}"


def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   🕷️  S.P.I.D.E.R. SWE-Bench Runner  🕷️                                     ║
║                                                                              ║
║   Speculative Planning with Iterative Deep Exploration & Refinement         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(style(banner, Colors.MAGENTA))


# =============================================================================
# DEMO TASKS
# =============================================================================

DEMO_TASKS = [
    SWEBenchTask(
        instance_id="demo__fix-division-by-zero",
        repo="demo/calculator",
        base_commit="abc123",
        problem_statement="""
The calculator crashes when dividing by zero.

File: calculator.py
```python
def divide(a, b):
    return a / b
```

Expected: Return None or raise a friendly error when b is 0.
Actual: Crashes with ZeroDivisionError.
""",
    ),
    SWEBenchTask(
        instance_id="demo__fix-null-check",
        repo="demo/userservice",
        base_commit="def456",
        problem_statement="""
The user service crashes when user is None.

File: user_service.py
```python
def get_username(user):
    return user.name
```

Expected: Return "Anonymous" when user is None.
Actual: Crashes with AttributeError.
""",
    ),
    SWEBenchTask(
        instance_id="demo__add-validation",
        repo="demo/api",
        base_commit="ghi789",
        problem_statement="""
The API endpoint doesn't validate email format.

File: validators.py
```python
def validate_email(email):
    return True  # TODO: implement
```

Expected: Return True only for valid email format (contains @ and .).
Actual: Returns True for everything.
""",
    ),
]


# =============================================================================
# RUNNER
# =============================================================================

class SWEBenchRunner:
    """
    Production runner for SWE-Bench evaluation.
    
    Features:
    - Progress tracking
    - Cost monitoring
    - Result aggregation
    - Checkpoint saving
    """
    
    def __init__(
        self,
        config: Optional[SolverConfig] = None,
        output_path: str = "predictions.json",
        checkpoint_interval: int = 5,
    ):
        self.config = config or SolverConfig()
        self.output_path = output_path
        self.checkpoint_interval = checkpoint_interval
        
        self.solver = SWEBenchSolver(self.config)
        self.results = []
        self.start_time = None
    
    def run_tasks(self, tasks: List[SWEBenchTask]) -> dict:
        """Run solver on a list of tasks."""
        
        print_banner()
        
        print(style(f"Running {len(tasks)} tasks...", Colors.CYAN, Colors.BOLD))
        print(f"  Model: {self.config.model}")
        print(f"  Budget: ${self.config.max_cost_usd}/task")
        print(f"  Max iterations: {self.config.max_iterations}")
        print()
        
        self.start_time = time.time()
        solved = 0
        
        for i, task in enumerate(tasks):
            print(style(f"\n[{i+1}/{len(tasks)}] ", Colors.CYAN) + task.instance_id)
            print("-" * 60)
            
            try:
                success, patch = self.solver.solve(task)
                
                self.results.append({
                    "instance_id": task.instance_id,
                    "success": success,
                    "patch": patch,
                    "cost": self.solver.total_cost,
                })
                
                if success:
                    solved += 1
                    print(style("  ✅ Solved!", Colors.GREEN))
                    if patch:
                        print(f"  Patch: {len(patch)} chars")
                else:
                    print(style("  ❌ Failed", Colors.RED))
                
            except Exception as e:
                print(style(f"  ❌ Error: {e}", Colors.RED))
                self.results.append({
                    "instance_id": task.instance_id,
                    "success": False,
                    "error": str(e),
                })
            
            # Checkpoint
            if (i + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint()
            
            # Cost check
            if self.solver.total_cost >= self.config.max_cost_usd * len(tasks):
                print(style("\n⚠️ Budget exhausted!", Colors.YELLOW))
                break
        
        # Final save
        self._save_results()
        
        # Summary
        duration = time.time() - self.start_time
        print()
        print(style("=" * 60, Colors.MAGENTA))
        print(style("  RESULTS SUMMARY", Colors.MAGENTA, Colors.BOLD))
        print(style("=" * 60, Colors.MAGENTA))
        print(f"  Tasks:    {solved}/{len(tasks)} solved ({100*solved/len(tasks):.1f}%)")
        print(f"  Cost:     ${self.solver.total_cost:.4f}")
        print(f"  Duration: {duration:.1f}s")
        print(f"  Output:   {self.output_path}")
        print()
        
        return {
            "solved": solved,
            "total": len(tasks),
            "accuracy": solved / len(tasks) if tasks else 0,
            "cost": self.solver.total_cost,
            "duration": duration,
        }
    
    def _save_checkpoint(self):
        """Save intermediate results."""
        checkpoint_path = self.output_path.replace(".json", "_checkpoint.json")
        with open(checkpoint_path, "w") as f:
            json.dump(self.results, f, indent=2)
    
    def _save_results(self):
        """Save final predictions in SWE-Bench format."""
        predictions = {}
        for r in self.results:
            predictions[r["instance_id"]] = {
                "model_patch": r.get("patch", ""),
                "model_name_or_path": "S.P.I.D.E.R.",
            }
        
        with open(self.output_path, "w") as f:
            json.dump(predictions, f, indent=2)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="S.P.I.D.E.R. SWE-Bench Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run on demo tasks (no cost)",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Path to single task JSON file",
    )
    parser.add_argument(
        "--tasks-dir",
        type=str,
        help="Directory containing task JSON files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.json",
        help="Output predictions file (default: predictions.json)",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=0.50,
        help="Max cost per task in USD (default: 0.50)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Max refinement iterations (default: 5)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek/deepseek-chat",
        help="Model to use (default: deepseek/deepseek-chat)",
    )
    
    args = parser.parse_args()
    
    # Load tasks
    if args.demo:
        tasks = DEMO_TASKS
    elif args.task:
        tasks = SWEBenchTask.from_json_file(args.task)
    elif args.tasks_dir:
        tasks = []
        for f in Path(args.tasks_dir).glob("*.json"):
            tasks.extend(SWEBenchTask.from_json_file(str(f)))
    else:
        print("Usage: python swe_runner.py --demo")
        print("       python swe_runner.py --task path/to/task.json")
        print("       python swe_runner.py --tasks-dir path/to/tasks/")
        sys.exit(1)
    
    # Configure
    config = SolverConfig(
        max_cost_usd=args.max_cost,
        max_iterations=args.max_iterations,
        model=args.model,
    )
    
    # Run
    runner = SWEBenchRunner(config, args.output)
    results = runner.run_tasks(tasks)
    
    return 0 if results["solved"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
