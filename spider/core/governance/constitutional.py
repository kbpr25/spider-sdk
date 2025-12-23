"""
S.P.I.D.E.R. Constitutional Governor - AI-Supervised Code Governance
=====================================================================

Born from: Assumption-12 (Constitutional AI - Anthropic)

The Scientific Finding:
"RLHF is messy and biased. RLAIF (RL from AI Feedback) gives the AI a
Constitution—a set of principles—and forces it to critique and revise
its own work until it aligns with the law."

The Insight:
You don't need a smarter model; you need a stricter "Constitution"
that forces the model to self-correct before the user sees output.

The Solution:
We act as "Founding Fathers" of code. We define a S.P.I.D.E.R. Constitution—
a YAML file of non-negotiable Engineering Principles.

The SL-CAI Loop:
1. Draft: Agent generates solution
2. Critique: Critic Agent checks against Constitution
3. Revision: Generator fixes violations
4. Ratification: Only release when Violations: None

Result: "Senior Engineer Standards" baked into the pipeline.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTITUTION TYPES
# =============================================================================

class PrincipleCategory(Enum):
    """Categories of constitutional principles."""
    SECURITY = auto()       # No hardcoded creds, no injection
    RELIABILITY = auto()    # Error handling, idempotency
    PERFORMANCE = auto()    # No N+1 queries, caching
    MAINTAINABILITY = auto()  # Type hints, documentation
    CORRECTNESS = auto()    # Logic errors, edge cases
    STYLE = auto()          # PEP8, naming conventions


class ViolationSeverity(Enum):
    """Severity of principle violations."""
    CRITICAL = auto()   # Must fix before release
    MAJOR = auto()      # Should fix
    MINOR = auto()      # Nice to fix
    INFO = auto()       # Suggestion only


@dataclass
class Principle:
    """A constitutional principle."""
    id: str
    name: str
    description: str
    category: PrincipleCategory
    severity: ViolationSeverity = ViolationSeverity.MAJOR
    check_pattern: Optional[str] = None     # Regex to detect violation
    fix_guidance: str = ""                   # How to fix
    examples: List[str] = field(default_factory=list)
    enabled: bool = True


@dataclass
class Violation:
    """A detected principle violation."""
    principle: Principle
    location: str = ""              # Line number or code snippet
    evidence: str = ""              # What triggered the violation
    suggested_fix: str = ""
    auto_fixable: bool = False


@dataclass
class CritiqueResult:
    """Result of constitution critique."""
    code: str
    violations: List[Violation]
    passed: bool
    score: float                    # 0-100 constitutional compliance
    critique_text: str = ""
    
    @property
    def critical_violations(self) -> List[Violation]:
        return [v for v in self.violations if v.principle.severity == ViolationSeverity.CRITICAL]
    
    @property
    def major_violations(self) -> List[Violation]:
        return [v for v in self.violations if v.principle.severity == ViolationSeverity.MAJOR]


@dataclass 
class RevisionResult:
    """Result of code revision."""
    original_code: str
    revised_code: str
    violations_fixed: List[Violation]
    violations_remaining: List[Violation]
    iterations: int
    final_score: float


# =============================================================================
# DEFAULT CONSTITUTION
# =============================================================================

DEFAULT_PRINCIPLES = [
    # SECURITY
    Principle(
        id="SEC001",
        name="No Hardcoded Credentials",
        description="Never hardcode API keys, passwords, or secrets in source code.",
        category=PrincipleCategory.SECURITY,
        severity=ViolationSeverity.CRITICAL,
        check_pattern=r'(?:api_key|password|secret|token)\s*=\s*["\'][^"\']{8,}["\']',
        fix_guidance="Use environment variables or secret managers.",
        examples=["api_key = 'sk-1234...'", "password = 'mysecret'"],
    ),
    Principle(
        id="SEC002",
        name="No SQL Injection",
        description="Never use string formatting for SQL queries.",
        category=PrincipleCategory.SECURITY,
        severity=ViolationSeverity.CRITICAL,
        check_pattern=r'(?:execute|query)\s*\(\s*f?["\'].*\{.*\}.*["\']',
        fix_guidance="Use parameterized queries with placeholders.",
        examples=["cursor.execute(f'SELECT * FROM users WHERE id = {user_id}')"],
    ),
    Principle(
        id="SEC003",
        name="No Command Injection",
        description="Never pass user input directly to shell commands.",
        category=PrincipleCategory.SECURITY,
        severity=ViolationSeverity.CRITICAL,
        check_pattern=r'(?:os\.system|subprocess\.(?:call|run|Popen))\s*\([^)]*?\+',
        fix_guidance="Use subprocess with shell=False and explicit arguments.",
    ),
    Principle(
        id="SEC004",
        name="No Eval/Exec on User Input",
        description="Never use eval() or exec() on user-provided data.",
        category=PrincipleCategory.SECURITY,
        severity=ViolationSeverity.CRITICAL,
        check_pattern=r'(?:eval|exec)\s*\(',
        fix_guidance="Use ast.literal_eval for safe parsing, or avoid entirely.",
    ),
    
    # RELIABILITY
    Principle(
        id="REL001",
        name="Exception Handling",
        description="API calls and I/O operations must have try-except blocks.",
        category=PrincipleCategory.RELIABILITY,
        severity=ViolationSeverity.MAJOR,
        check_pattern=r'(?:requests\.|urllib|open\()(?!.*try)',
        fix_guidance="Wrap I/O operations in try-except with specific exceptions.",
    ),
    Principle(
        id="REL002",
        name="Idempotency Keys",
        description="Payment and mutation operations should use idempotency keys.",
        category=PrincipleCategory.RELIABILITY,
        severity=ViolationSeverity.MAJOR,
        check_pattern=r'(?:Charge\.create|Payment\.create)(?!.*idempotency)',
        fix_guidance="Include idempotency_key parameter for safe retries.",
    ),
    Principle(
        id="REL003",
        name="Resource Cleanup",
        description="File handles and connections must be properly closed.",
        category=PrincipleCategory.RELIABILITY,
        severity=ViolationSeverity.MAJOR,
        check_pattern=r'open\([^)]+\)(?!.*(?:with|\.close))',
        fix_guidance="Use 'with' statement for automatic resource cleanup.",
    ),
    Principle(
        id="REL004",
        name="Timeout on Network Calls",
        description="Network requests should have explicit timeouts.",
        category=PrincipleCategory.RELIABILITY,
        severity=ViolationSeverity.MAJOR,
        check_pattern=r'requests\.(?:get|post|put|delete)\([^)]+\)(?!.*timeout)',
        fix_guidance="Add timeout parameter: requests.get(url, timeout=30)",
    ),
    
    # CORRECTNESS
    Principle(
        id="COR001",
        name="Division By Zero Check",
        description="Check for zero before division operations.",
        category=PrincipleCategory.CORRECTNESS,
        severity=ViolationSeverity.MAJOR,
        check_pattern=r'\/\s*(?!\d)[a-zA-Z_]\w*(?!\s*(?:!=|==|>|<)\s*0)',
        fix_guidance="Add guard clause: if divisor != 0",
    ),
    Principle(
        id="COR002",
        name="None Check Before Access",
        description="Check for None before attribute access.",
        category=PrincipleCategory.CORRECTNESS,
        severity=ViolationSeverity.MAJOR,
        fix_guidance="Use 'if obj is not None' or Optional chaining.",
    ),
    Principle(
        id="COR003",
        name="Empty Collection Check",
        description="Check if collection is empty before accessing first/last.",
        category=PrincipleCategory.CORRECTNESS,
        severity=ViolationSeverity.MAJOR,
        check_pattern=r'\[\s*0\s*\]|\[\s*-1\s*\](?!.*(?:if|len))',
        fix_guidance="Check 'if collection:' or 'if len(collection) > 0'.",
    ),
    
    # MAINTAINABILITY
    Principle(
        id="MNT001",
        name="Type Hints",
        description="Function parameters and returns should have type hints.",
        category=PrincipleCategory.MAINTAINABILITY,
        severity=ViolationSeverity.MINOR,
        check_pattern=r'def\s+\w+\s*\([^:)]+\)(?!\s*->)',
        fix_guidance="Add type hints: def func(arg: str) -> int:",
    ),
    Principle(
        id="MNT002",
        name="Docstrings",
        description="Public functions should have docstrings.",
        category=PrincipleCategory.MAINTAINABILITY,
        severity=ViolationSeverity.MINOR,
        check_pattern=r'def\s+(?!_)\w+\([^)]*\):[^"\']*(?:\n\s+[^"\'])',
        fix_guidance="Add docstring after function definition.",
    ),
    Principle(
        id="MNT003",
        name="Magic Numbers",
        description="Avoid magic numbers; use named constants.",
        category=PrincipleCategory.MAINTAINABILITY,
        severity=ViolationSeverity.MINOR,
        check_pattern=r'(?<![.\d])\b(?!0|1|-1)\d{2,}\b(?![.\d])',
        fix_guidance="Define as constant: MAX_RETRIES = 5",
    ),
    
    # PERFORMANCE
    Principle(
        id="PRF001",
        name="N+1 Query Prevention",
        description="Avoid database queries in loops.",
        category=PrincipleCategory.PERFORMANCE,
        severity=ViolationSeverity.MAJOR,
        check_pattern=r'for\s+\w+\s+in\s+\w+:.*(?:\.query|\.filter|\.get)',
        fix_guidance="Use prefetch/select_related or batch queries.",
    ),
    Principle(
        id="PRF002",
        name="String Concatenation in Loop",
        description="Avoid += for string building in loops.",
        category=PrincipleCategory.PERFORMANCE,
        severity=ViolationSeverity.MINOR,
        check_pattern=r'for\s+.*:\s*\n(?:[^}]*?\+=\s*["\'])',
        fix_guidance="Use list append and join, or StringIO.",
    ),
]


# =============================================================================
# CONSTITUTION
# =============================================================================

class Constitution:
    """
    The S.P.I.D.E.R. Constitution - Engineering Principles.
    
    Can be loaded from YAML or defined in code.
    """
    
    def __init__(self, principles: List[Principle] = None):
        self.principles = {p.id: p for p in (principles or DEFAULT_PRINCIPLES)}
    
    @classmethod
    def from_yaml(cls, path: str) -> "Constitution":
        """Load constitution from YAML file."""
        import yaml
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        principles = []
        for p in data.get("principles", []):
            principles.append(Principle(
                id=p["id"],
                name=p["name"],
                description=p.get("description", ""),
                category=PrincipleCategory[p.get("category", "CORRECTNESS").upper()],
                severity=ViolationSeverity[p.get("severity", "MAJOR").upper()],
                check_pattern=p.get("pattern"),
                fix_guidance=p.get("fix", ""),
                examples=p.get("examples", []),
            ))
        
        return cls(principles)
    
    def get_principle(self, principle_id: str) -> Optional[Principle]:
        return self.principles.get(principle_id)
    
    def get_by_category(self, category: PrincipleCategory) -> List[Principle]:
        return [p for p in self.principles.values() if p.category == category]
    
    def get_critical(self) -> List[Principle]:
        return [p for p in self.principles.values() 
                if p.severity == ViolationSeverity.CRITICAL]
    
    def to_prompt(self) -> str:
        """Convert constitution to prompt format for LLM."""
        lines = ["# S.P.I.D.E.R. Constitution - Engineering Principles\n"]
        
        for category in PrincipleCategory:
            category_principles = self.get_by_category(category)
            if category_principles:
                lines.append(f"\n## {category.name}\n")
                for p in category_principles:
                    lines.append(f"- **{p.id}: {p.name}** [{p.severity.name}]")
                    lines.append(f"  {p.description}")
                    if p.fix_guidance:
                        lines.append(f"  Fix: {p.fix_guidance}")
        
        return "\n".join(lines)


# =============================================================================
# CRITIC AGENT
# =============================================================================

class ConstitutionalCritic:
    """
    The Critic Agent - Evaluates code against the Constitution.
    
    Uses pattern matching for known issues and LLM for complex analysis.
    """
    
    def __init__(
        self,
        constitution: Constitution,
        llm_callback: Optional[Callable[[str], str]] = None,
    ):
        self.constitution = constitution
        self.llm_callback = llm_callback
        
        # Compile patterns
        self.patterns: Dict[str, re.Pattern] = {}
        for p in constitution.principles.values():
            if p.check_pattern:
                try:
                    self.patterns[p.id] = re.compile(p.check_pattern, re.MULTILINE | re.IGNORECASE)
                except re.error:
                    logger.warning(f"Invalid pattern for {p.id}")
    
    def critique(self, code: str) -> CritiqueResult:
        """
        Critique code against the constitution.
        
        Args:
            code: Source code to analyze
            
        Returns:
            CritiqueResult with violations
        """
        violations = []
        
        # Pattern-based checks
        for principle_id, pattern in self.patterns.items():
            principle = self.constitution.get_principle(principle_id)
            if not principle or not principle.enabled:
                continue
            
            matches = pattern.finditer(code)
            for match in matches:
                # Find line number
                line_num = code[:match.start()].count('\n') + 1
                
                violations.append(Violation(
                    principle=principle,
                    location=f"Line {line_num}",
                    evidence=match.group(0)[:100],
                    suggested_fix=principle.fix_guidance,
                    auto_fixable=False,
                ))
        
        # LLM-based checks (for complex principles)
        if self.llm_callback:
            llm_violations = self._llm_critique(code)
            violations.extend(llm_violations)
        
        # Calculate score
        if not violations:
            score = 100.0
        else:
            # Deduct points based on severity
            deductions = sum(
                20 if v.principle.severity == ViolationSeverity.CRITICAL
                else 10 if v.principle.severity == ViolationSeverity.MAJOR
                else 5 if v.principle.severity == ViolationSeverity.MINOR
                else 1
                for v in violations
            )
            score = max(0.0, 100.0 - deductions)
        
        passed = not any(
            v.principle.severity in (ViolationSeverity.CRITICAL, ViolationSeverity.MAJOR)
            for v in violations
        )
        
        return CritiqueResult(
            code=code,
            violations=violations,
            passed=passed,
            score=score,
            critique_text=self._generate_critique_text(violations),
        )
    
    def _llm_critique(self, code: str) -> List[Violation]:
        """Use LLM for complex critique."""
        if not self.llm_callback:
            return []
        
        prompt = f"""You are a Senior Code Reviewer. Analyze this code against our Constitution.

CONSTITUTION:
{self.constitution.to_prompt()}

CODE TO REVIEW:
```python
{code}
```

For each violation found, respond with JSON:
[
  {{"principle_id": "SEC001", "location": "line 5", "evidence": "...", "fix": "..."}},
  ...
]

If no violations, respond with: []
"""
        
        try:
            response = self.llm_callback(prompt)
            
            # Parse JSON
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                items = json.loads(json_match.group())
                violations = []
                for item in items:
                    principle = self.constitution.get_principle(item.get("principle_id", ""))
                    if principle:
                        violations.append(Violation(
                            principle=principle,
                            location=item.get("location", ""),
                            evidence=item.get("evidence", ""),
                            suggested_fix=item.get("fix", principle.fix_guidance),
                        ))
                return violations
        except Exception as e:
            logger.warning(f"LLM critique failed: {e}")
        
        return []
    
    def _generate_critique_text(self, violations: List[Violation]) -> str:
        """Generate human-readable critique."""
        if not violations:
            return "✅ Code passes all constitutional checks."
        
        lines = [f"❌ Found {len(violations)} violation(s):\n"]
        
        for v in violations:
            severity_emoji = {
                ViolationSeverity.CRITICAL: "🔴",
                ViolationSeverity.MAJOR: "🟠",
                ViolationSeverity.MINOR: "🟡",
                ViolationSeverity.INFO: "ℹ️",
            }.get(v.principle.severity, "❓")
            
            lines.append(f"{severity_emoji} [{v.principle.id}] {v.principle.name}")
            lines.append(f"   Location: {v.location}")
            lines.append(f"   Evidence: {v.evidence[:60]}...")
            lines.append(f"   Fix: {v.suggested_fix}")
            lines.append("")
        
        return "\n".join(lines)


# =============================================================================
# CONSTITUTIONAL GOVERNOR
# =============================================================================

class ConstitutionalGovernor:
    """
    The Legislative Supervisor - Constitutional AI for Code.
    
    Implements the SL-CAI (Self-Learning Constitutional AI) Loop:
    1. Draft: Agent generates solution
    2. Critique: Check against Constitution
    3. Revision: Fix violations
    4. Ratification: Only release when Violations: None
    
    From Assumption-12:
    "Give the AI a Constitution and force it to self-correct."
    
    Usage:
        governor = ConstitutionalGovernor()
        
        # Submit code for review
        result = governor.govern(
            code=generated_code,
            llm_callback=my_llm,
        )
        
        if result.final_score >= 80:
            print("Code approved!")
            print(result.revised_code)
        else:
            print("Code rejected:", result.violations_remaining)
    """
    
    def __init__(
        self,
        constitution: Constitution = None,
        max_iterations: int = 3,
        min_score: float = 80.0,
        require_zero_critical: bool = True,
    ):
        """
        Initialize Constitutional Governor.
        
        Args:
            constitution: The constitution to enforce
            max_iterations: Maximum revision cycles
            min_score: Minimum score for ratification
            require_zero_critical: Block if any critical violations remain
        """
        self.constitution = constitution or Constitution()
        self.max_iterations = max_iterations
        self.min_score = min_score
        self.require_zero_critical = require_zero_critical
        
        self.critic = ConstitutionalCritic(self.constitution)
        
        self._stats = {
            "codes_reviewed": 0,
            "codes_approved": 0,
            "codes_rejected": 0,
            "total_violations": 0,
            "violations_fixed": 0,
        }
    
    def govern(
        self,
        code: str,
        llm_callback: Optional[Callable[[str], str]] = None,
        auto_revise: bool = True,
    ) -> RevisionResult:
        """
        Govern code through the constitutional review process.
        
        Args:
            code: Source code to review
            llm_callback: LLM for revision (optional)
            auto_revise: Attempt automatic revision
            
        Returns:
            RevisionResult with final code and status
        """
        self._stats["codes_reviewed"] += 1
        
        original_code = code
        current_code = code
        all_violations_fixed = []
        iteration = 0
        
        for iteration in range(1, self.max_iterations + 1):
            # Critique current code
            critique = self.critic.critique(current_code)
            self._stats["total_violations"] += len(critique.violations)
            
            # Check if passed
            if critique.passed and critique.score >= self.min_score:
                self._stats["codes_approved"] += 1
                return RevisionResult(
                    original_code=original_code,
                    revised_code=current_code,
                    violations_fixed=all_violations_fixed,
                    violations_remaining=[],
                    iterations=iteration,
                    final_score=critique.score,
                )
            
            # Check for critical violations
            if self.require_zero_critical and critique.critical_violations:
                if not auto_revise or not llm_callback:
                    self._stats["codes_rejected"] += 1
                    return RevisionResult(
                        original_code=original_code,
                        revised_code=current_code,
                        violations_fixed=all_violations_fixed,
                        violations_remaining=critique.violations,
                        iterations=iteration,
                        final_score=critique.score,
                    )
            
            # Attempt revision
            if auto_revise and llm_callback and critique.violations:
                revised = self._revise(current_code, critique, llm_callback)
                if revised != current_code:
                    # Track fixed violations
                    new_critique = self.critic.critique(revised)
                    fixed = [v for v in critique.violations 
                             if v.principle.id not in [nv.principle.id for nv in new_critique.violations]]
                    all_violations_fixed.extend(fixed)
                    self._stats["violations_fixed"] += len(fixed)
                    current_code = revised
                else:
                    # No progress made
                    break
            else:
                break
        
        # Final check
        final_critique = self.critic.critique(current_code)
        
        if final_critique.score >= self.min_score:
            self._stats["codes_approved"] += 1
        else:
            self._stats["codes_rejected"] += 1
        
        return RevisionResult(
            original_code=original_code,
            revised_code=current_code,
            violations_fixed=all_violations_fixed,
            violations_remaining=final_critique.violations,
            iterations=iteration,
            final_score=final_critique.score,
        )
    
    def _revise(
        self,
        code: str,
        critique: CritiqueResult,
        llm_callback: Callable[[str], str],
    ) -> str:
        """Revise code to fix violations."""
        violation_text = "\n".join(
            f"- {v.principle.id}: {v.principle.name} at {v.location}\n"
            f"  Evidence: {v.evidence}\n"
            f"  Fix: {v.suggested_fix}"
            for v in critique.violations
        )
        
        prompt = f"""You are a Senior Engineer. Fix ALL constitutional violations in this code.

VIOLATIONS FOUND:
{violation_text}

ORIGINAL CODE:
```python
{code}
```

Provide the COMPLETE fixed code. Fix ALL violations while preserving functionality.
Respond with only the fixed code, no explanations.
"""
        
        try:
            response = llm_callback(prompt)
            
            # Extract code block
            code_match = re.search(r'```(?:python)?\n?(.*?)```', response, re.DOTALL)
            if code_match:
                return code_match.group(1).strip()
            
            # If no code block, assume entire response is code
            if "def " in response or "class " in response:
                return response.strip()
                
        except Exception as e:
            logger.warning(f"Revision failed: {e}")
        
        return code
    
    def quick_check(self, code: str) -> Tuple[bool, float, List[str]]:
        """
        Quick check without revision.
        
        Returns:
            Tuple of (passed, score, violation_summaries)
        """
        critique = self.critic.critique(code)
        summaries = [
            f"{v.principle.id}: {v.principle.name}"
            for v in critique.violations
        ]
        return (critique.passed, critique.score, summaries)
    
    def get_stats(self) -> Dict[str, int]:
        return self._stats.copy()
    
    def print_result(self, result: RevisionResult) -> None:
        """Print governance result."""
        print("\n" + "=" * 60)
        print("⚖️ CONSTITUTIONAL GOVERNOR RESULT")
        print("=" * 60)
        
        status = "✅ APPROVED" if result.final_score >= self.min_score else "❌ REJECTED"
        print(f"\n{status}")
        print(f"Score: {result.final_score:.0f}/100")
        print(f"Iterations: {result.iterations}")
        print(f"Violations Fixed: {len(result.violations_fixed)}")
        print(f"Violations Remaining: {len(result.violations_remaining)}")
        
        if result.violations_remaining:
            print("\n⚠️ REMAINING VIOLATIONS:")
            for v in result.violations_remaining:
                severity = v.principle.severity.name
                print(f"   [{severity}] {v.principle.id}: {v.principle.name}")
        
        if result.violations_fixed:
            print("\n✅ FIXED VIOLATIONS:")
            for v in result.violations_fixed:
                print(f"   ✓ {v.principle.id}: {v.principle.name}")
        
        print("=" * 60)


# =============================================================================
# PACKAGE INIT
# =============================================================================

__all__ = [
    "ConstitutionalGovernor",
    "Constitution",
    "ConstitutionalCritic",
    "Principle",
    "Violation",
    "CritiqueResult",
    "RevisionResult",
    "PrincipleCategory",
    "ViolationSeverity",
]


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("⚖️ S.P.I.D.E.R. Constitutional Governor - Demo")
    print("=" * 70)
    
    governor = ConstitutionalGovernor()
    
    # Code with violations
    bad_code = '''
def process_payment(user_id, amount):
    api_key = "sk_live_12345678901234567890"  # SEC001: Hardcoded credential
    
    # No type hints (MNT001)
    # No docstring (MNT002)
    
    conn = db.connect()
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")  # SEC002: SQL injection
    
    result = requests.post(
        "https://api.stripe.com/charge",  # REL004: No timeout
        data={"amount": amount}
    )
    
    return result
'''
    
    print("\n📝 CODE WITH VIOLATIONS:")
    print(bad_code)
    
    # Quick check (no revision)
    passed, score, violations = governor.quick_check(bad_code)
    
    print(f"\n🔍 QUICK CHECK:")
    print(f"   Passed: {passed}")
    print(f"   Score: {score:.0f}/100")
    print(f"   Violations: {len(violations)}")
    for v in violations:
        print(f"   - {v}")
    
    # Full governance (without LLM for demo)
    result = governor.govern(bad_code, auto_revise=False)
    governor.print_result(result)
    
    print(f"\n📊 Stats: {governor.get_stats()}")
