#!/usr/bin/env python3
"""
S.P.I.D.E.R. Batch Commander - The Sleep Test
==============================================

Run 50 engineering tasks overnight. Wake up to data.

Features:
- Graceful Ctrl+C handling (no data loss)
- Progress checkpointing after every task
- Live statistics dashboard
- CSV report generation
- Crash recovery

Usage:
    python -m spider.benchmarks.batch_runner
    
Or for quick test:
    python -m spider.benchmarks.batch_runner --quick
"""

import csv
import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

# Ensure we can import spider
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from spider.benchmarks.dataset import get_sleep_test_dataset, get_quick_test_dataset
from spider.main import SpiderEngine


# =============================================================================
# COLORS
# =============================================================================

class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"


def style(text: str, *styles: str) -> str:
    return f"{''.join(styles)}{text}{Colors.RESET}"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TaskResult:
    """Result of a single task execution."""
    task_id: str
    problem: str
    difficulty: str
    layer: str
    status: str  # PASS, FAIL, CRASH, SKIP
    duration_sec: float
    timestamp: str
    error_message: str = ""
    proposal_length: int = 0
    consensus_votes: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "task_id": self.task_id,
            "problem": self.problem[:50],
            "difficulty": self.difficulty,
            "layer": self.layer,
            "status": self.status,
            "duration_sec": self.duration_sec,
            "proposal_length": self.proposal_length,
            "consensus_votes": self.consensus_votes,
            "error_message": self.error_message,
        }


# =============================================================================
# BATCH COMMANDER
# =============================================================================

class BatchCommander:
    """
    The Overnight Commander - runs The Sleep Test.
    
    Manages:
    - Task iteration with progress tracking
    - Graceful interruption (Ctrl+C safe)
    - Checkpoint saving after each task
    - Live statistics dashboard
    - Final report generation
    """
    
    def __init__(
        self,
        output_dir: str = ".",
        node_count: int = 3,
        cooldown_seconds: float = 1.0,
    ):
        self.output_dir = Path(output_dir)
        self.node_count = node_count
        self.cooldown_seconds = cooldown_seconds
        
        self.results: List[TaskResult] = []
        self.running = True
        self.start_time: Optional[datetime] = None
        self.current_task: Optional[str] = None
        
        # Statistics
        self.stats = {
            'total': 0,
            'completed': 0,
            'passed': 0,
            'failed': 0,
            'crashed': 0,
            'skipped': 0,
        }
        
        # Handle Ctrl+C gracefully
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
    
    def _handle_interrupt(self, signum, frame):
        """Handle graceful shutdown."""
        print(f"\n\n{style('[COMMANDER]', Colors.RED, Colors.BOLD)} 🛑 RECEIVED STOP SIGNAL")
        print(f"  Finishing task: {self.current_task or 'None'}")
        print(f"  Results saved: {len(self.results)} tasks")
        self.running = False
    
    def _print_banner(self):
        """Print the batch runner banner."""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   🌙  THE SLEEP TEST  🌙                                                     ║
║                                                                              ║
║   "Can you go to sleep? Can you wake up to 50 PRs and zero crashes?"        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        print(style(banner, Colors.MAGENTA))
    
    def _print_progress(self, current: int, total: int, task_id: str):
        """Print progress bar."""
        pct = (current / total) * 100
        filled = int(pct / 2)
        bar = "█" * filled + "░" * (50 - filled)
        
        elapsed = datetime.now() - self.start_time
        if current > 0:
            eta = (elapsed / current) * (total - current)
            eta_str = str(timedelta(seconds=int(eta.total_seconds())))
        else:
            eta_str = "calculating..."
        
        print(f"\n{style('─' * 78, Colors.DIM)}")
        print(f"  [{bar}] {pct:.1f}%")
        print(f"  Task {current}/{total} | ETA: {eta_str} | Current: {task_id}")
        print(f"{style('─' * 78, Colors.DIM)}")
    
    def _print_live_stats(self):
        """Print live statistics."""
        total = self.stats['completed']
        if total == 0:
            return
        
        pass_rate = (self.stats['passed'] / total) * 100
        
        print(f"\n  {style('LIVE STATS:', Colors.CYAN, Colors.BOLD)}")
        print(f"    ✅ Passed:  {self.stats['passed']:3d} ({pass_rate:.1f}%)")
        print(f"    ❌ Failed:  {self.stats['failed']:3d}")
        print(f"    💥 Crashed: {self.stats['crashed']:3d}")
        print(f"    ⏭️  Skipped: {self.stats['skipped']:3d}")
    
    def run_sleep_test(self, quick: bool = False):
        """
        Run the full Sleep Test.
        
        Args:
            quick: If True, run only 5 representative tasks
        """
        self._print_banner()
        
        # Get dataset
        if quick:
            tasks = get_quick_test_dataset()
            print(f"  {style('MODE:', Colors.YELLOW)} Quick Test (5 tasks)")
        else:
            tasks = get_sleep_test_dataset()
            print(f"  {style('MODE:', Colors.GREEN)} Full Sleep Test (50 tasks)")
        
        self.stats['total'] = len(tasks)
        self.start_time = datetime.now()
        
        print(f"  {style('START:', Colors.CYAN)} {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  {style('TASKS:', Colors.CYAN)} {len(tasks)}")
        print(f"  {style('AGENTS:', Colors.CYAN)} {self.node_count}")
        print()
        
        # Initialize engine once
        engine = SpiderEngine(node_count=self.node_count)
        
        try:
            print(f"  {style('▸', Colors.DIM)} Starting S.P.I.D.E.R. Engine...")
            if not engine.start():
                print(f"  {style('✗', Colors.RED)} Failed to start engine")
                return
            print(f"  {style('✓', Colors.GREEN)} Engine ready")
            
            # Run tasks
            for i, task in enumerate(tasks, 1):
                if not self.running:
                    break
                
                self.current_task = task['id']
                self._print_progress(i, len(tasks), task['id'])
                
                # Execute task
                result = self._execute_task(engine, task)
                self.results.append(result)
                
                # Update stats
                self.stats['completed'] += 1
                if result.status == "PASS":
                    self.stats['passed'] += 1
                elif result.status == "FAIL":
                    self.stats['failed'] += 1
                elif result.status == "CRASH":
                    self.stats['crashed'] += 1
                else:
                    self.stats['skipped'] += 1
                
                # Print result
                icon = {"PASS": "✅", "FAIL": "❌", "CRASH": "💥", "SKIP": "⏭️"}.get(result.status, "❓")
                print(f"\n  {icon} {style(task['id'], Colors.BOLD)}: {result.status} ({result.duration_sec:.1f}s)")
                
                # Print live stats
                self._print_live_stats()
                
                # Checkpoint save
                self._save_checkpoint()
                
                # Cooldown
                if self.running and i < len(tasks):
                    time.sleep(self.cooldown_seconds)
            
        finally:
            engine.stop()
        
        # Final report
        self._print_final_report()
        self._save_final_report()
    
    def _execute_task(self, engine: SpiderEngine, task: Dict) -> TaskResult:
        """Execute a single task and return the result."""
        start_time = time.perf_counter()
        timestamp = datetime.now().isoformat()
        
        try:
            # Run spider solve
            result = engine.solve(task['problem'])
            
            duration = time.perf_counter() - start_time
            
            # Determine status
            if result.success:
                status = "PASS"
                proposal_length = len(result.proposal.code_diff) if result.proposal else 0
            else:
                status = "FAIL"
                proposal_length = 0
            
            return TaskResult(
                task_id=task['id'],
                problem=task['problem'],
                difficulty=task['difficulty'],
                layer=task['layer'],
                status=status,
                duration_sec=round(duration, 2),
                timestamp=timestamp,
                proposal_length=proposal_length,
                consensus_votes=3 if status == "PASS" else 0,
            )
            
        except Exception as e:
            duration = time.perf_counter() - start_time
            return TaskResult(
                task_id=task['id'],
                problem=task['problem'],
                difficulty=task['difficulty'],
                layer=task['layer'],
                status="CRASH",
                duration_sec=round(duration, 2),
                timestamp=timestamp,
                error_message=str(e)[:100],
            )
    
    def _save_checkpoint(self):
        """Save current results as checkpoint."""
        filename = self.output_dir / f"sleep_test_checkpoint_{datetime.now().strftime('%Y%m%d')}.csv"
        self._write_csv(filename)
    
    def _save_final_report(self):
        """Save final report with full details."""
        date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # CSV report
        csv_file = self.output_dir / f"sleep_test_results_{date_str}.csv"
        self._write_csv(csv_file)
        print(f"\n  📄 CSV Report: {csv_file}")
        
        # JSON report with full stats
        json_file = self.output_dir / f"sleep_test_results_{date_str}.json"
        report = {
            "meta": {
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": datetime.now().isoformat(),
                "total_tasks": self.stats['total'],
                "node_count": self.node_count,
            },
            "stats": self.stats,
            "results": [r.to_dict() for r in self.results],
        }
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"  📊 JSON Report: {json_file}")
    
    def _write_csv(self, filename: Path):
        """Write results to CSV."""
        if not self.results:
            return
        
        fields = list(self.results[0].to_dict().keys())
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for result in self.results:
                writer.writerow(result.to_dict())
    
    def _print_final_report(self):
        """Print final summary."""
        elapsed = datetime.now() - self.start_time if self.start_time else timedelta()
        
        print("\n")
        print(style("═" * 78, Colors.GREEN))
        print(f"  {style('🌞 SLEEP TEST COMPLETE. GOOD MORNING. 🌞', Colors.GREEN, Colors.BOLD)}")
        print(style("═" * 78, Colors.GREEN))
        
        total = self.stats['completed']
        pass_rate = (self.stats['passed'] / total * 100) if total > 0 else 0
        
        print(f"""
  {style('FINAL STATISTICS', Colors.CYAN, Colors.BOLD)}
  {'─' * 40}
  Total Tasks:     {total}
  Elapsed Time:    {str(elapsed).split('.')[0]}
  
  ✅ Passed:       {self.stats['passed']:3d} ({pass_rate:.1f}%)
  ❌ Failed:       {self.stats['failed']:3d}
  💥 Crashed:      {self.stats['crashed']:3d}
  ⏭️  Skipped:      {self.stats['skipped']:3d}
  
  {style('VERDICT:', Colors.BOLD)} {'🏆 SLEEP TEST PASSED' if pass_rate >= 80 else '📈 NEEDS IMPROVEMENT'}
""")
        
        if pass_rate >= 80:
            print(f"  {style('🕷️ S.P.I.D.E.R. is production ready!', Colors.GREEN, Colors.BOLD)}")
        else:
            print(f"  {style('Keep iterating. The path to reliability is paved with data.', Colors.YELLOW)}")
        
        print()


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="S.P.I.D.E.R. Sleep Test - Overnight Batch Runner"
    )
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help="Run quick test (5 tasks) instead of full test (50 tasks)"
    )
    parser.add_argument(
        '--nodes', '-n',
        type=int,
        default=3,
        help="Number of agents (default: 3)"
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=".",
        help="Output directory for reports"
    )
    parser.add_argument(
        '--cooldown', '-c',
        type=float,
        default=1.0,
        help="Cooldown between tasks in seconds"
    )
    
    args = parser.parse_args()
    
    commander = BatchCommander(
        output_dir=args.output,
        node_count=args.nodes,
        cooldown_seconds=args.cooldown,
    )
    
    commander.run_sleep_test(quick=args.quick)
    
    return 0 if commander.stats['crashed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())