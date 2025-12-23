"""
S.P.I.D.E.R. Self-Correction Engine
====================================

Implements the key insight from Opus 4.5:
"Tool use and agentic coding - ability to learn from execution feedback"

This is the MOST CRITICAL component for achieving 90%+ Pass@1 on SWE-Bench.

Algorithm:
1. Generate initial solution
2. Execute tests to verify
3. If failed: Analyze error → Generate hypothesis → Retry with refined approach
4. Track successful patterns for future use
5. Max iterations configurable (default: 5)

Mathematical Foundation:
- Bayesian Error Correction: P(fix|error) = P(error|fix) * P(fix) / P(error)
- Each iteration refines prior based on observed test results
"""

import logging
import re
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# ERROR TAXONOMY
# =============================================================================

class ErrorCategory(Enum):
    """Classification of errors for targeted correction."""
    SYNTAX = auto()           # SyntaxError, IndentationError
    TYPE = auto()             # TypeError, AttributeError
    VALUE = auto()            # ValueError, KeyError, IndexError
    IMPORT = auto()           # ImportError, ModuleNotFoundError
    ASSERTION = auto()        # AssertionError from tests
    RUNTIME = auto()          # RuntimeError, generic exceptions
    TIMEOUT = auto()          # Execution timeout
    UNKNOWN = auto()          # Unclassified errors


@dataclass
class ErrorAnalysis:
    """Detailed analysis of an error for targeted correction."""
    category: ErrorCategory
    message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    function_name: Optional[str] = None
    stack_trace: str = ""
    suggested_fix: str = ""
    confidence: float = 0.0
    
    def to_prompt(self) -> str:
        """Convert to prompt-friendly format for LLM."""
        parts = [
            f"Error Type: {self.category.name}",
            f"Message: {self.message}",
        ]
        if self.file_path:
            parts.append(f"File: {self.file_path}")
        if self.line_number:
            parts.append(f"Line: {self.line_number}")
        if self.function_name:
            parts.append(f"Function: {self.function_name}")
        if self.stack_trace:
            parts.append(f"Stack Trace:\n{self.stack_trace}")
        if self.suggested_fix:
            parts.append(f"Suggested Fix Direction: {self.suggested_fix}")
        return "\n".join(parts)


# =============================================================================
# CORRECTION STRATEGIES
# =============================================================================

@dataclass
class CorrectionStrategy:
    """A strategy for correcting a specific type of error."""
    name: str
    description: str
    applies_to: List[ErrorCategory]
    prompt_template: str
    priority: int = 0  # Higher = try first
    
    def matches(self, error: ErrorAnalysis) -> bool:
        """Check if this strategy applies to the given error."""
        return error.category in self.applies_to


# Built-in correction strategies
CORRECTION_STRATEGIES = [
    CorrectionStrategy(
        name="syntax_fix",
        description="Fix syntax errors by analyzing the exact error location",
        applies_to=[ErrorCategory.SYNTAX],
        prompt_template="""
The code has a syntax error at {file_path}:{line_number}.
Error: {message}

Common causes:
1. Missing parentheses, brackets, or quotes
2. Incorrect indentation
3. Missing colons after if/for/def/class

Please fix the syntax error and provide the corrected code.
""",
        priority=100,
    ),
    CorrectionStrategy(
        name="type_fix",
        description="Fix type errors by checking variable types and conversions",
        applies_to=[ErrorCategory.TYPE],
        prompt_template="""
The code has a type error: {message}

This usually means:
1. Calling a method on None (check for null values)
2. Wrong argument types (check function signatures)
3. Missing type conversions (str vs int vs float)

Stack trace:
{stack_trace}

Please analyze the type mismatch and fix the code.
""",
        priority=90,
    ),
    CorrectionStrategy(
        name="assertion_fix",
        description="Fix test assertion failures by understanding expected vs actual",
        applies_to=[ErrorCategory.ASSERTION],
        prompt_template="""
A test assertion failed: {message}

This is a CRITICAL error indicating the fix doesn't produce the expected result.

Stack trace:
{stack_trace}

Analyze:
1. What is the expected value?
2. What is the actual value?
3. Why is there a difference?
4. What change to the fix will produce the correct result?

Please provide a corrected patch.
""",
        priority=95,
    ),
    CorrectionStrategy(
        name="import_fix",
        description="Fix import errors by checking module availability",
        applies_to=[ErrorCategory.IMPORT],
        prompt_template="""
Import error: {message}

Common solutions:
1. Check if the module name is spelled correctly
2. Check if the module is installed
3. Check if using relative vs absolute imports
4. Check if the module path is correct

Please fix the import statement.
""",
        priority=80,
    ),
    CorrectionStrategy(
        name="value_fix",
        description="Fix value errors like missing keys, out-of-range indices",
        applies_to=[ErrorCategory.VALUE],
        prompt_template="""
Value error: {message}

This typically means:
1. Accessing a key that doesn't exist (add default or check existence)
2. Index out of range (check list/string length)
3. Invalid value for a function (check constraints)

Stack trace:
{stack_trace}

Please fix the value handling.
""",
        priority=85,
    ),
    CorrectionStrategy(
        name="generic_fix",
        description="Generic fix for unclassified errors",
        applies_to=[ErrorCategory.RUNTIME, ErrorCategory.UNKNOWN, ErrorCategory.TIMEOUT],
        prompt_template="""
An error occurred: {message}

Stack trace:
{stack_trace}

Please analyze what went wrong and provide a fix.
Focus on:
1. The exact line where the error occurred
2. What conditions led to this error
3. How to handle this case properly
""",
        priority=50,
    ),
]


# =============================================================================
# ERROR ANALYZER
# =============================================================================

class ErrorAnalyzer:
    """
    Analyzes errors from test execution to categorize and extract actionable info.
    
    Uses regex patterns and heuristics to parse Python error messages.
    """
    
    # Patterns for extracting error information
    TRACEBACK_FILE_PATTERN = re.compile(
        r'File "([^"]+)", line (\d+), in (\w+)'
    )
    ERROR_TYPE_PATTERN = re.compile(
        r'^(\w+Error|\w+Exception): (.+)$', re.MULTILINE
    )
    ASSERTION_PATTERN = re.compile(
        r'assert\w* (.+?) == (.+)'
    )
    
    def __init__(self):
        self.stats = {
            "total_analyzed": 0,
            "by_category": {cat.name: 0 for cat in ErrorCategory},
        }
    
    def analyze(self, error_output: str, test_output: str = "") -> ErrorAnalysis:
        """
        Analyze error output and return structured analysis.
        
        Args:
            error_output: The stderr or exception message
            test_output: Optional stdout from test execution
            
        Returns:
            ErrorAnalysis with categorization and extracted details
        """
        self.stats["total_analyzed"] += 1
        combined = error_output + "\n" + test_output
        
        # Determine error category
        category = self._categorize_error(combined)
        self.stats["by_category"][category.name] += 1
        
        # Extract error message
        message = self._extract_message(combined)
        
        # Extract location info
        file_path, line_number, function_name = self._extract_location(combined)
        
        # Generate fix suggestion
        suggested_fix = self._suggest_fix(category, message, combined)
        
        # Calculate confidence based on how much info we extracted
        confidence = self._calculate_confidence(
            category, message, file_path, line_number
        )
        
        return ErrorAnalysis(
            category=category,
            message=message,
            file_path=file_path,
            line_number=line_number,
            function_name=function_name,
            stack_trace=combined[-2000:] if len(combined) > 2000 else combined,
            suggested_fix=suggested_fix,
            confidence=confidence,
        )
    
    def _categorize_error(self, text: str) -> ErrorCategory:
        """Categorize the error based on keywords."""
        text_lower = text.lower()
        
        if "syntaxerror" in text_lower or "indentationerror" in text_lower:
            return ErrorCategory.SYNTAX
        elif "typeerror" in text_lower or "attributeerror" in text_lower:
            return ErrorCategory.TYPE
        elif any(e in text_lower for e in ["valueerror", "keyerror", "indexerror"]):
            return ErrorCategory.VALUE
        elif "importerror" in text_lower or "modulenotfounderror" in text_lower:
            return ErrorCategory.IMPORT
        elif "assertionerror" in text_lower or "assert" in text_lower:
            return ErrorCategory.ASSERTION
        elif "timeout" in text_lower or "timed out" in text_lower:
            return ErrorCategory.TIMEOUT
        elif "runtimeerror" in text_lower:
            return ErrorCategory.RUNTIME
        else:
            return ErrorCategory.UNKNOWN
    
    def _extract_message(self, text: str) -> str:
        """Extract the main error message."""
        match = self.ERROR_TYPE_PATTERN.search(text)
        if match:
            return f"{match.group(1)}: {match.group(2)[:200]}"
        
        # Fallback: last non-empty line
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if lines:
            return lines[-1][:200]
        return "Unknown error"
    
    def _extract_location(self, text: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
        """Extract file, line number, and function name from traceback."""
        matches = list(self.TRACEBACK_FILE_PATTERN.finditer(text))
        if matches:
            # Get the last match (usually the actual error location)
            last = matches[-1]
            return last.group(1), int(last.group(2)), last.group(3)
        return None, None, None
    
    def _suggest_fix(self, category: ErrorCategory, message: str, text: str) -> str:
        """Generate a suggested fix direction based on error analysis."""
        suggestions = {
            ErrorCategory.SYNTAX: "Check the line for missing symbols or incorrect indentation",
            ErrorCategory.TYPE: "Verify the object type before calling methods; add null checks",
            ErrorCategory.VALUE: "Add bounds checking or default values",
            ErrorCategory.IMPORT: "Verify module name and import path",
            ErrorCategory.ASSERTION: "Review expected vs actual values; fix logic error",
            ErrorCategory.TIMEOUT: "Optimize algorithm or add early termination",
            ErrorCategory.RUNTIME: "Add defensive error handling",
            ErrorCategory.UNKNOWN: "Review the stack trace for the root cause",
        }
        return suggestions.get(category, "Review the error and apply appropriate fix")
    
    def _calculate_confidence(
        self,
        category: ErrorCategory,
        message: str,
        file_path: Optional[str],
        line_number: Optional[int],
    ) -> float:
        """Calculate confidence score for the analysis (0-1)."""
        score = 0.3  # Base score
        
        if category != ErrorCategory.UNKNOWN:
            score += 0.2
        if file_path:
            score += 0.2
        if line_number:
            score += 0.2
        if len(message) > 10:
            score += 0.1
        
        return min(score, 1.0)


# =============================================================================
# SELF-CORRECTION ENGINE
# =============================================================================

@dataclass
class CorrectionAttempt:
    """Record of a correction attempt."""
    iteration: int
    error_analysis: ErrorAnalysis
    strategy_used: str
    patch_generated: str
    success: bool
    duration_ms: float
    

@dataclass
class CorrectionResult:
    """Final result of the self-correction process."""
    success: bool
    final_patch: str
    iterations: int
    attempts: List[CorrectionAttempt]
    total_duration_ms: float
    error_progression: List[str]  # How errors changed over iterations
    lessons_learned: List[str]  # Insights that could help future fixes


class SelfCorrectionEngine:
    """
    The core self-correction engine that implements iterative refinement.
    
    This is what makes Opus 4.5-level performance possible with smaller models.
    
    Key Features:
    1. Error Analysis: Deep understanding of what went wrong
    2. Strategy Selection: Choose the right fix approach
    3. Iterative Refinement: Learn from each attempt
    4. Pattern Learning: Remember what works
    
    Usage:
        engine = SelfCorrectionEngine(llm_gateway)
        result = engine.correct(
            initial_patch="...",
            test_command="pytest test_file.py",
            context="... problem description ..."
        )
        if result.success:
            final_patch = result.final_patch
    """
    
    def __init__(
        self,
        llm_gateway=None,
        max_iterations: int = 5,
        strategies: Optional[List[CorrectionStrategy]] = None,
        enable_learning: bool = True,
    ):
        """
        Initialize the self-correction engine.
        
        Args:
            llm_gateway: LLMGateway for generating corrections
            max_iterations: Maximum correction attempts before giving up
            strategies: Custom correction strategies (or use built-in)
            enable_learning: Whether to learn from successful corrections
        """
        self.gateway = llm_gateway
        self.max_iterations = max_iterations
        self.strategies = strategies or CORRECTION_STRATEGIES
        self.analyzer = ErrorAnalyzer()
        self.enable_learning = enable_learning
        
        # Learning memory
        self.successful_patterns: List[Dict[str, Any]] = []
        
        # Statistics
        self.stats = {
            "total_corrections": 0,
            "successful_corrections": 0,
            "total_iterations": 0,
            "by_category": {cat.name: {"attempts": 0, "successes": 0} 
                          for cat in ErrorCategory},
        }
        
        logger.info(f"SelfCorrectionEngine initialized with {len(self.strategies)} strategies")
    
    def correct(
        self,
        initial_patch: str,
        test_runner: Callable[[str], Tuple[bool, str, str]],
        context: str,
        problem_description: str = "",
        file_content: str = "",
    ) -> CorrectionResult:
        """
        Attempt to correct a patch through iterative refinement.
        
        Args:
            initial_patch: The initial patch to test and refine
            test_runner: Function that runs tests, returns (success, stdout, stderr)
            context: Additional context (problem description, file content, etc.)
            problem_description: Original problem statement
            file_content: Original file content being modified
            
        Returns:
            CorrectionResult with the final patch and metadata
        """
        start_time = time.time()
        self.stats["total_corrections"] += 1
        
        current_patch = initial_patch
        attempts: List[CorrectionAttempt] = []
        error_progression: List[str] = []
        lessons: List[str] = []
        
        for iteration in range(self.max_iterations):
            self.stats["total_iterations"] += 1
            iter_start = time.time()
            
            logger.info(f"Correction iteration {iteration + 1}/{self.max_iterations}")
            
            # Run tests
            success, stdout, stderr = test_runner(current_patch)
            
            if success:
                # Success! Record the pattern and return
                self.stats["successful_corrections"] += 1
                lessons.append(f"Fixed after {iteration + 1} iterations")
                
                if self.enable_learning and iteration > 0:
                    self._learn_pattern(attempts, current_patch)
                
                return CorrectionResult(
                    success=True,
                    final_patch=current_patch,
                    iterations=iteration + 1,
                    attempts=attempts,
                    total_duration_ms=(time.time() - start_time) * 1000,
                    error_progression=error_progression,
                    lessons_learned=lessons,
                )
            
            # Analyze the error
            error_analysis = self.analyzer.analyze(stderr, stdout)
            error_progression.append(error_analysis.message)
            
            self.stats["by_category"][error_analysis.category.name]["attempts"] += 1
            
            # Select best strategy
            strategy = self._select_strategy(error_analysis)
            
            # Generate correction prompt
            correction_prompt = self._build_correction_prompt(
                current_patch=current_patch,
                error_analysis=error_analysis,
                strategy=strategy,
                context=context,
                problem_description=problem_description,
                file_content=file_content,
                previous_attempts=attempts,
            )
            
            # Get LLM to generate corrected patch
            corrected_patch = self._generate_correction(correction_prompt)
            
            if not corrected_patch or corrected_patch == current_patch:
                # LLM couldn't generate a different patch
                lessons.append(f"Iteration {iteration + 1}: No new patch generated")
                continue
            
            # Record attempt
            attempt = CorrectionAttempt(
                iteration=iteration + 1,
                error_analysis=error_analysis,
                strategy_used=strategy.name,
                patch_generated=corrected_patch,
                success=False,  # Will update on next iteration if successful
                duration_ms=(time.time() - iter_start) * 1000,
            )
            attempts.append(attempt)
            
            # Update current patch for next iteration
            current_patch = corrected_patch
            lessons.append(f"Iteration {iteration + 1}: Applied {strategy.name} strategy")
        
        # Exhausted iterations without success
        return CorrectionResult(
            success=False,
            final_patch=current_patch,
            iterations=self.max_iterations,
            attempts=attempts,
            total_duration_ms=(time.time() - start_time) * 1000,
            error_progression=error_progression,
            lessons_learned=lessons + ["Exhausted max iterations without success"],
        )
    
    def _select_strategy(self, error: ErrorAnalysis) -> CorrectionStrategy:
        """Select the best correction strategy for the given error."""
        matching = [s for s in self.strategies if s.matches(error)]
        if not matching:
            # Fallback to generic strategy
            return self.strategies[-1]
        
        # Sort by priority and return highest
        return sorted(matching, key=lambda s: s.priority, reverse=True)[0]
    
    def _build_correction_prompt(
        self,
        current_patch: str,
        error_analysis: ErrorAnalysis,
        strategy: CorrectionStrategy,
        context: str,
        problem_description: str,
        file_content: str,
        previous_attempts: List[CorrectionAttempt],
    ) -> str:
        """Build the prompt for the LLM to generate a correction."""
        
        # Build previous attempts summary
        attempts_summary = ""
        if previous_attempts:
            attempts_list = []
            for i, att in enumerate(previous_attempts[-3:], 1):  # Last 3 attempts
                attempts_list.append(
                    f"  Attempt {att.iteration}: {att.strategy_used} - "
                    f"{att.error_analysis.category.name}: {att.error_analysis.message[:100]}"
                )
            attempts_summary = "\n".join(attempts_list)
        
        prompt = f"""You are debugging a code change. The current patch is failing tests.

## Original Problem
{problem_description[:1000] if problem_description else 'Fix the bug described below.'}

## Current Patch Attempt
```diff
{current_patch[:2000]}
```

## Error Analysis
{error_analysis.to_prompt()}

## Correction Strategy: {strategy.name}
{strategy.description}

{strategy.prompt_template.format(
    message=error_analysis.message,
    file_path=error_analysis.file_path or "unknown",
    line_number=error_analysis.line_number or "unknown",
    stack_trace=error_analysis.stack_trace[-1000:],
    function_name=error_analysis.function_name or "unknown",
)}

## Previous Attempts (if any)
{attempts_summary or "This is the first correction attempt."}

## Your Task
Analyze why the current patch is failing and generate a CORRECTED patch.
Think step by step:
1. What is the error telling us?
2. What is wrong with the current patch?
3. What specific change will fix it?

Provide ONLY the corrected unified diff patch. No explanations.
Start with --- a/ and end with the complete patch.
"""
        return prompt
    
    def _generate_correction(self, prompt: str) -> Optional[str]:
        """Use LLM to generate a corrected patch."""
        if not self.gateway:
            logger.warning("No LLM gateway configured, cannot generate correction")
            return None
        
        try:
            from spider.core.agent.llm_client import Message, MessageRole
            
            messages = [
                Message(MessageRole.SYSTEM, 
                       "You are an expert debugger. Generate only the corrected patch, nothing else."),
                Message(MessageRole.USER, prompt),
            ]
            
            response = self.gateway.complete(
                messages=messages,
                max_tokens=2000,
                temperature=0.3,  # Lower temperature for more focused fixes
            )
            
            if response.success:
                return self._extract_patch(response.content)
            else:
                logger.warning(f"LLM failed to generate correction: {response.error}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating correction: {e}")
            return None
    
    def _extract_patch(self, content: str) -> Optional[str]:
        """Extract patch from LLM response."""
        # Look for diff format
        if "--- a/" in content or "--- " in content:
            # Find the start of the diff
            lines = content.split("\n")
            diff_lines = []
            in_diff = False
            
            for line in lines:
                if line.startswith("--- ") or line.startswith("diff --git"):
                    in_diff = True
                if in_diff:
                    diff_lines.append(line)
            
            if diff_lines:
                return "\n".join(diff_lines)
        
        # Look for code blocks
        if "```" in content:
            blocks = re.findall(r'```(?:\w+)?\n(.*?)```', content, re.DOTALL)
            if blocks:
                return blocks[0].strip()
        
        return content.strip() if content.strip() else None
    
    def _learn_pattern(
        self,
        attempts: List[CorrectionAttempt],
        final_patch: str,
    ) -> None:
        """Learn from a successful correction for future use."""
        if not attempts:
            return
        
        pattern = {
            "error_category": attempts[0].error_analysis.category.name,
            "strategies_tried": [a.strategy_used for a in attempts],
            "iterations_needed": len(attempts),
            "final_strategy": attempts[-1].strategy_used,
            "timestamp": time.time(),
        }
        
        self.successful_patterns.append(pattern)
        
        # Keep only last 100 patterns
        if len(self.successful_patterns) > 100:
            self.successful_patterns = self.successful_patterns[-100:]
        
        logger.debug(f"Learned pattern: {pattern}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        success_rate = (
            self.stats["successful_corrections"] / self.stats["total_corrections"]
            if self.stats["total_corrections"] > 0 else 0
        )
        avg_iterations = (
            self.stats["total_iterations"] / self.stats["total_corrections"]
            if self.stats["total_corrections"] > 0 else 0
        )
        
        return {
            **self.stats,
            "success_rate": success_rate,
            "average_iterations": avg_iterations,
            "patterns_learned": len(self.successful_patterns),
        }
    
    def print_stats(self) -> None:
        """Print statistics."""
        stats = self.get_stats()
        print("\n" + "=" * 60)
        print("SELF-CORRECTION ENGINE STATISTICS")
        print("=" * 60)
        print(f"Total Corrections Attempted: {stats['total_corrections']}")
        print(f"Successful Corrections:      {stats['successful_corrections']}")
        print(f"Success Rate:                {stats['success_rate']:.1%}")
        print(f"Average Iterations:          {stats['average_iterations']:.1f}")
        print(f"Patterns Learned:            {stats['patterns_learned']}")
        print("\nErrors by Category:")
        for cat, data in stats["by_category"].items():
            if data["attempts"] > 0:
                print(f"  {cat}: {data['attempts']} attempts, {data['successes']} successes")
        print("=" * 60)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_engine(gateway=None, **kwargs) -> SelfCorrectionEngine:
    """Factory function to create a configured SelfCorrectionEngine."""
    return SelfCorrectionEngine(llm_gateway=gateway, **kwargs)


def analyze_error(error_text: str) -> ErrorAnalysis:
    """Quick function to analyze an error without full engine."""
    analyzer = ErrorAnalyzer()
    return analyzer.analyze(error_text)


# =============================================================================
# MAIN (DEMO)
# =============================================================================

if __name__ == "__main__":
    # Demo the error analyzer
    print("S.P.I.D.E.R. Self-Correction Engine Demo")
    print("=" * 50)
    
    # Sample error
    sample_error = '''
Traceback (most recent call last):
  File "test_example.py", line 42, in test_add_numbers
    assert add(2, 3) == 5
AssertionError: assert 6 == 5
'''
    
    analyzer = ErrorAnalyzer()
    analysis = analyzer.analyze(sample_error)
    
    print("\nError Analysis:")
    print(analysis.to_prompt())
    print(f"\nConfidence: {analysis.confidence:.0%}")
