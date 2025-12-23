#!/usr/bin/env python3
"""
S.P.I.D.E.R. CLI - Command Line Interface
==========================================

Usage:
    spider solve "Fix the bug in calculator.py"
    spider solve --file bug_report.txt
    spider demo
    spider status
"""

import argparse
import sys
from pathlib import Path


def cmd_solve(args):
    """Solve a coding problem."""
    from spider.core.agent.ultimate import UltimateSolver, UltimateSolverConfig, SolverMode
    from spider.benchmarks.swe_pipeline import SWEBenchTask
    
    # Get problem description
    if args.file:
        problem = Path(args.file).read_text()
    else:
        problem = args.problem
    
    if not problem:
        print("Error: Provide a problem description or --file")
        return 1
    
    print("🕷️ S.P.I.D.E.R. Solving...")
    print("-" * 50)
    
    # Create task
    task = SWEBenchTask(
        instance_id="cli-task",
        repo="local",
        base_commit="HEAD",
        problem_statement=problem,
    )
    
    # Solve
    mode_map = {
        "simple": SolverMode.SIMPLE,
        "agentic": SolverMode.AGENTIC,
        "multi": SolverMode.MULTI_AGENT,
        "full": SolverMode.FULL,
    }
    
    config = UltimateSolverConfig(
        mode=mode_map.get(args.mode, SolverMode.AGENTIC),
        max_cost_usd=args.max_cost,
    )
    
    solver = UltimateSolver(config, args.repo)
    success, patch, meta = solver.solve(task)
    
    print("\n" + "=" * 50)
    if success:
        print("✅ SOLVED!")
        print(f"\nPatch:\n```\n{patch}\n```")
    else:
        print("❌ Could not solve")
    
    print(f"\nCost: ${meta.get('cost', 0):.4f}")
    return 0 if success else 1


def cmd_demo(args):
    """Run demo to test installation."""
    print("🕷️ S.P.I.D.E.R. Demo")
    print("=" * 50)
    
    # Test imports
    tests = []
    
    try:
        from spider.core.agent import UltimateSolver
        tests.append(("UltimateSolver", True))
    except Exception as e:
        tests.append(("UltimateSolver", False))
    
    try:
        from spider.core.agent import ReActAgent
        tests.append(("ReActAgent", True))
    except Exception as e:
        tests.append(("ReActAgent", False))
    
    try:
        from spider.core.agent import LLMGateway
        gw = LLMGateway()
        providers = gw.get_available_providers()
        tests.append((f"LLMGateway ({len(providers)} providers)", True))
    except Exception as e:
        tests.append(("LLMGateway", False))
    
    for name, passed in tests:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
    
    all_passed = all(p for _, p in tests)
    print("\n" + ("✅ All systems go!" if all_passed else "❌ Some issues found"))
    return 0 if all_passed else 1


def cmd_status(args):
    """Show S.P.I.D.E.R. status."""
    print("🕷️ S.P.I.D.E.R. SDK v1.0.0")
    print("=" * 50)
    
    # Check components
    components = [
        "spider.core.agent.agentic",
        "spider.core.agent.multiagent",
        "spider.core.agent.ultimate",
        "spider.core.agent.llm_client",
        "spider.benchmarks.swe_pipeline",
    ]
    
    for comp in components:
        try:
            __import__(comp)
            print(f"  ✅ {comp.split('.')[-1]}")
        except:
            print(f"  ❌ {comp.split('.')[-1]}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="spider",
        description="🕷️ S.P.I.D.E.R. - AI-Powered Code Fixing SDK",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # solve command
    solve_parser = subparsers.add_parser("solve", help="Solve a coding problem")
    solve_parser.add_argument("problem", nargs="?", help="Problem description")
    solve_parser.add_argument("--file", "-f", help="Read problem from file")
    solve_parser.add_argument("--repo", "-r", default=".", help="Repository path")
    solve_parser.add_argument("--mode", "-m", default="agentic", 
                             choices=["simple", "agentic", "multi", "full"],
                             help="Solving mode")
    solve_parser.add_argument("--max-cost", type=float, default=0.10,
                             help="Max cost in USD")
    
    # demo command
    demo_parser = subparsers.add_parser("demo", help="Run demo")
    
    # status command
    status_parser = subparsers.add_parser("status", help="Show status")
    
    args = parser.parse_args()
    
    if args.command == "solve":
        return cmd_solve(args)
    elif args.command == "demo":
        return cmd_demo(args)
    elif args.command == "status":
        return cmd_status(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
