"""
S.P.I.D.E.R. Reasoning Core
============================

The cognitive engine for evaluating code proposals using a multi-stage
verification pipeline:
1. Fast Check: Bloom filter for instant file existence validation
2. Integrity Check: Merkle tree for codebase state verification
3. Deep Check: LLM-powered semantic code review via Ollama

Designed to minimize compute by failing fast on obvious issues.
"""

import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

# External dependency
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from spider.core.distributed.protocol import Proposal, VoteDecision
from spider.core.dsa.bloom import CodebaseIndexer
from spider.core.dsa.merkle import CodebaseMerkleTree


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ReasoningConfig:
    """Configuration for the ReasoningCore."""
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "llama3"
    ollama_timeout: float = 60.0
    risk_threshold: int = 7
    simulation_mode: bool = False
    simulation_approval_rate: float = 0.85
    enable_deep_check: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    log_level: str = "INFO"


# =============================================================================
# ANALYSIS RESULT
# =============================================================================

class CheckStage(Enum):
    """Stages in the reasoning pipeline."""
    FAST_CHECK = auto()
    INTEGRITY_CHECK = auto()
    DEEP_CHECK = auto()


@dataclass
class LLMReviewResult:
    """Result from LLM code review."""
    approved: bool
    risk_score: int
    reason: str
    raw_response: str = ""
    parse_success: bool = True


@dataclass
class AnalysisResult:
    """Complete analysis result for a proposal."""
    proposal_id: str
    decision: VoteDecision
    stage_reached: CheckStage
    reasoning: str
    confidence: float = 1.0
    llm_review: Optional[LLMReviewResult] = None
    duration_ms: float = 0.0
    checks_performed: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'proposal_id': self.proposal_id,
            'decision': self.decision.name,
            'stage_reached': self.stage_reached.name,
            'reasoning': self.reasoning,
            'confidence': self.confidence,
            'duration_ms': self.duration_ms,
            'checks_performed': self.checks_performed,
        }


# =============================================================================
# REASONING CORE
# =============================================================================

class ReasoningCore:
    """
    The cognitive engine for S.P.I.D.E.R. code proposal evaluation.
    
    Uses a multi-stage verification pipeline:
    1. Fast Check: Bloom filter for file existence
    2. Integrity Check: Merkle tree for state verification
    3. Deep Check: LLM-powered semantic review
    
    Attributes:
        config: Configuration settings.
        indexer: CodebaseIndexer for fast symbol lookup.
        merkle_tree: CodebaseMerkleTree for integrity verification.
    """

    def __init__(
        self,
        indexer: Optional[CodebaseIndexer] = None,
        merkle_tree: Optional[CodebaseMerkleTree] = None,
        config: Optional[ReasoningConfig] = None,
    ):
        """
        Initialize the ReasoningCore.
        
        Args:
            indexer: Pre-built CodebaseIndexer, or None to skip fast check.
            merkle_tree: Pre-built CodebaseMerkleTree, or None to skip integrity.
            config: Configuration settings.
        """
        self.config = config or ReasoningConfig()
        self.indexer = indexer
        self.merkle_tree = merkle_tree
        self._logger = self._setup_logger()
        
        # Track if Ollama is available
        self._ollama_available: Optional[bool] = None
        
        # Statistics
        self._stats = {
            'total_analyzed': 0,
            'fast_check_rejects': 0,
            'integrity_check_rejects': 0,
            'deep_check_rejects': 0,
            'approvals': 0,
            'simulation_mode_uses': 0,
        }

    def _setup_logger(self) -> logging.Logger:
        """Set up logging."""
        logger = logging.getLogger(f"ReasoningCore")
        logger.setLevel(getattr(logging, self.config.log_level))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s [%(name)s] %(levelname)s: %(message)s',
                datefmt='%H:%M:%S'
            ))
            logger.addHandler(handler)
        
        return logger

    # =========================================================================
    # LOAD FROM DISK
    # =========================================================================

    @classmethod
    def from_disk(
        cls,
        indexer_path: str,
        merkle_root_path: str,
        config: Optional[ReasoningConfig] = None,
    ) -> 'ReasoningCore':
        """
        Load ReasoningCore with pre-built indexes from disk.
        
        Args:
            indexer_path: Path to saved CodebaseIndexer JSON.
            merkle_root_path: Path to codebase root for Merkle tree.
            config: Optional configuration.
            
        Returns:
            Initialized ReasoningCore.
        """
        # Load Bloom indexer
        indexer = CodebaseIndexer.load(indexer_path)
        
        # Build Merkle tree
        merkle_tree = CodebaseMerkleTree(merkle_root_path)
        merkle_tree.build()
        
        return cls(indexer=indexer, merkle_tree=merkle_tree, config=config)

    @classmethod
    def from_codebase(
        cls,
        codebase_path: str,
        config: Optional[ReasoningConfig] = None,
    ) -> 'ReasoningCore':
        """
        Create ReasoningCore by indexing a codebase directory.
        
        Args:
            codebase_path: Path to the codebase root.
            config: Optional configuration.
            
        Returns:
            Initialized ReasoningCore with fresh indexes.
        """
        # Build Bloom indexer
        indexer = CodebaseIndexer(codebase_path).index()
        
        # Build Merkle tree
        merkle_tree = CodebaseMerkleTree(codebase_path)
        merkle_tree.build()
        
        return cls(indexer=indexer, merkle_tree=merkle_tree, config=config)

    # =========================================================================
    # MAIN ANALYSIS
    # =========================================================================

    def analyze_proposal(self, proposal: Proposal) -> VoteDecision:
        """
        Analyze a proposal and return a vote decision.
        
        Pipeline:
        1. Fast Check: Verify files exist via Bloom filter
        2. Integrity Check: Verify Merkle root matches
        3. Deep Check: LLM semantic review
        
        Args:
            proposal: The proposal to analyze.
            
        Returns:
            VoteDecision.APPROVE or VoteDecision.REJECT
        """
        result = self.analyze_proposal_detailed(proposal)
        return result.decision

    def analyze_proposal_detailed(self, proposal: Proposal) -> AnalysisResult:
        """
        Analyze a proposal with detailed results.
        
        Returns full AnalysisResult with reasoning and metrics.
        """
        start_time = time.perf_counter()
        self._stats['total_analyzed'] += 1
        
        self._logger.info(f"Analyzing proposal {proposal.proposal_id[:8]}...")
        
        checks_performed = {}
        
        # Stage 1: Fast Check (Bloom Filter)
        if self.indexer:
            fast_check_passed, fast_reason = self._fast_check(proposal)
            checks_performed['fast_check'] = fast_check_passed
            
            if not fast_check_passed:
                self._stats['fast_check_rejects'] += 1
                self._logger.warning(f"Fast check FAILED: {fast_reason}")
                return AnalysisResult(
                    proposal_id=proposal.proposal_id,
                    decision=VoteDecision.REJECT,
                    stage_reached=CheckStage.FAST_CHECK,
                    reasoning=fast_reason,
                    confidence=1.0,
                    duration_ms=(time.perf_counter() - start_time) * 1000,
                    checks_performed=checks_performed,
                )
            
            self._logger.info("Fast check PASSED")
        
        # Stage 2: Integrity Check (Merkle Tree)
        if self.merkle_tree and self.merkle_tree.root_hash:
            integrity_passed, integrity_reason = self._integrity_check(proposal)
            checks_performed['integrity_check'] = integrity_passed
            
            if not integrity_passed:
                self._stats['integrity_check_rejects'] += 1
                self._logger.warning(f"Integrity check FAILED: {integrity_reason}")
                return AnalysisResult(
                    proposal_id=proposal.proposal_id,
                    decision=VoteDecision.REJECT,
                    stage_reached=CheckStage.INTEGRITY_CHECK,
                    reasoning=integrity_reason,
                    confidence=1.0,
                    duration_ms=(time.perf_counter() - start_time) * 1000,
                    checks_performed=checks_performed,
                )
            
            self._logger.info("Integrity check PASSED")
        
        # Stage 3: Deep Check (LLM Review)
        if self.config.enable_deep_check:
            llm_result = self._deep_check(proposal)
            checks_performed['deep_check'] = llm_result.approved
            
            decision = VoteDecision.APPROVE if llm_result.approved else VoteDecision.REJECT
            
            if not llm_result.approved:
                self._stats['deep_check_rejects'] += 1
            else:
                self._stats['approvals'] += 1
            
            self._logger.info(
                f"Deep check: {'APPROVED' if llm_result.approved else 'REJECTED'} "
                f"(risk={llm_result.risk_score})"
            )
            
            return AnalysisResult(
                proposal_id=proposal.proposal_id,
                decision=decision,
                stage_reached=CheckStage.DEEP_CHECK,
                reasoning=llm_result.reason,
                confidence=1.0 - (llm_result.risk_score / 10),
                llm_review=llm_result,
                duration_ms=(time.perf_counter() - start_time) * 1000,
                checks_performed=checks_performed,
            )
        
        # All checks passed, no deep check
        self._stats['approvals'] += 1
        return AnalysisResult(
            proposal_id=proposal.proposal_id,
            decision=VoteDecision.APPROVE,
            stage_reached=CheckStage.INTEGRITY_CHECK,
            reasoning="All checks passed",
            confidence=0.8,  # Lower confidence without LLM review
            duration_ms=(time.perf_counter() - start_time) * 1000,
            checks_performed=checks_performed,
        )

    # =========================================================================
    # STAGE 1: FAST CHECK (BLOOM FILTER)
    # =========================================================================

    def _fast_check(self, proposal: Proposal) -> Tuple[bool, str]:
        """
        Fast check using Bloom filter.
        
        Verifies that files mentioned in the proposal exist in the codebase.
        
        Returns:
            Tuple of (passed, reason).
        """
        if not self.indexer:
            return True, "No indexer configured, skipping fast check"
        
        # Extract files from proposal
        target_files = proposal.target_files
        
        # Also try to extract files from diff
        diff_files = self._extract_files_from_diff(proposal.code_diff)
        all_files = set(target_files) | set(diff_files)
        
        if not all_files:
            # No files specified, can't validate
            return True, "No files specified in proposal"
        
        # Check each file
        missing_files = []
        for filepath in all_files:
            # Normalize path for checking
            normalized = filepath.replace('\\', '/').strip('/')
            basename = normalized.split('/')[-1].replace('.py', '')
            
            # Check if file or module exists in index
            if normalized not in self.indexer and basename not in self.indexer:
                missing_files.append(filepath)
        
        if missing_files:
            return False, f"Unknown files: {', '.join(missing_files[:3])}"
        
        return True, "All files exist in codebase"

    def _extract_files_from_diff(self, diff: str) -> List[str]:
        """Extract file paths from a unified diff."""
        files = []
        
        # Match --- a/path/to/file or +++ b/path/to/file
        pattern = r'^[+-]{3}\s+[ab]/(.+)$'
        for line in diff.split('\n'):
            match = re.match(pattern, line)
            if match:
                filepath = match.group(1)
                if filepath != '/dev/null':
                    files.append(filepath)
        
        return files

    # =========================================================================
    # STAGE 2: INTEGRITY CHECK (MERKLE TREE)
    # =========================================================================

    def _integrity_check(self, proposal: Proposal) -> Tuple[bool, str]:
        """
        Integrity check using Merkle tree.
        
        Verifies that the proposal's merkle_root_hash matches current state.
        
        Returns:
            Tuple of (passed, reason).
        """
        if not self.merkle_tree or not self.merkle_tree.root_hash:
            return True, "No Merkle tree configured, skipping integrity check"
        
        proposal_hash = proposal.merkle_root_hash
        current_hash = self.merkle_tree.root_hash
        
        # Check if hashes match (or if proposal hash is a placeholder)
        if not proposal_hash or proposal_hash.startswith('merkle_'):
            # Placeholder hash, accept for simulation
            self._logger.debug("Proposal has placeholder Merkle hash, accepting")
            return True, "Placeholder Merkle hash accepted"
        
        if proposal_hash != current_hash:
            return False, (
                f"State mismatch: proposal based on {proposal_hash[:12]}... "
                f"but current is {current_hash[:12]}..."
            )
        
        return True, "Merkle root matches current state"

    # =========================================================================
    # STAGE 3: DEEP CHECK (LLM REVIEW)
    # =========================================================================

    def _deep_check(self, proposal: Proposal) -> LLMReviewResult:
        """
        Deep semantic check using LLM via Ollama.
        
        Prompts the LLM to review the code diff and assess risk.
        Falls back to simulation mode if Ollama is unavailable.
        
        Returns:
            LLMReviewResult with approval decision and risk score.
        """
        # Check if we should use simulation mode
        if self.config.simulation_mode or not self._check_ollama_available():
            return self._simulate_llm_review(proposal)
        
        # Build the prompt
        prompt = self._build_review_prompt(proposal)
        
        # Call Ollama
        try:
            response = self._call_ollama(prompt)
            return self._parse_llm_response(response)
        except Exception as e:
            self._logger.error(f"LLM call failed: {e}")
            # Fallback to simulation
            self._stats['simulation_mode_uses'] += 1
            return self._simulate_llm_review(proposal, fallback=True)

    def _check_ollama_available(self) -> bool:
        """Check if Ollama is available."""
        if self._ollama_available is not None:
            return self._ollama_available
        
        if not HAS_REQUESTS:
            self._logger.warning("requests library not installed, using simulation mode")
            self._ollama_available = False
            return False
        
        try:
            # Quick health check
            response = requests.get(
                self.config.ollama_url.replace('/api/generate', '/api/tags'),
                timeout=5.0
            )
            self._ollama_available = response.status_code == 200
        except Exception:
            self._ollama_available = False
        
        if not self._ollama_available:
            self._logger.warning("Ollama not available, using simulation mode")
        
        return self._ollama_available

    def _build_review_prompt(self, proposal: Proposal) -> str:
        """Build the code review prompt for the LLM."""
        reasoning_text = "\n".join(f"- {r}" for r in proposal.reasoning_chain)
        
        prompt = f"""You are a Senior Site Reliability Engineer (SRE) reviewing a code change proposal.

## Code Diff
```diff
{proposal.code_diff}
```

## Author's Reasoning
{reasoning_text if reasoning_text else "No reasoning provided."}

## Target Files
{', '.join(proposal.target_files) if proposal.target_files else "Not specified"}

## Your Task
Review this code change for:
1. Security vulnerabilities
2. Performance issues
3. Error handling
4. Code quality
5. Potential bugs

Output ONLY valid JSON in exactly this format (no other text):
{{"approved": true/false, "risk_score": 0-10, "reason": "brief explanation"}}

Where:
- approved: true if safe to merge, false if risky
- risk_score: 0 (safe) to 10 (critical risk)
- reason: one sentence explaining your decision"""
        
        return prompt

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API and return the response text."""
        if not HAS_REQUESTS:
            raise RuntimeError("requests library not available")
        
        payload = {
            "model": self.config.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for consistent output
                "num_predict": 256,  # Limit response length
            }
        }
        
        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(
                    self.config.ollama_url,
                    json=payload,
                    timeout=self.config.ollama_timeout,
                )
                response.raise_for_status()
                
                data = response.json()
                return data.get('response', '')
                
            except requests.exceptions.Timeout:
                self._logger.warning(f"Ollama timeout (attempt {attempt + 1})")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
            except requests.exceptions.RequestException as e:
                self._logger.error(f"Ollama request failed: {e}")
                raise
        
        raise RuntimeError("Ollama call failed after all retries")

    def _parse_llm_response(self, response: str) -> LLMReviewResult:
        """Parse the LLM response JSON."""
        # Try to extract JSON from the response
        json_match = re.search(r'\{[^}]+\}', response)
        
        if not json_match:
            self._logger.warning(f"No JSON found in LLM response: {response[:100]}")
            return LLMReviewResult(
                approved=False,
                risk_score=8,
                reason="LLM response format invalid, rejecting for safety",
                raw_response=response,
                parse_success=False,
            )
        
        try:
            data = json.loads(json_match.group())
            
            approved = bool(data.get('approved', False))
            risk_score = int(data.get('risk_score', 5))
            reason = str(data.get('reason', 'No reason provided'))
            
            # Clamp risk score
            risk_score = max(0, min(10, risk_score))
            
            # Override approval if risk is too high
            if risk_score > self.config.risk_threshold:
                approved = False
                reason = f"Risk score {risk_score} exceeds threshold {self.config.risk_threshold}. {reason}"
            
            return LLMReviewResult(
                approved=approved,
                risk_score=risk_score,
                reason=reason,
                raw_response=response,
                parse_success=True,
            )
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self._logger.warning(f"Failed to parse LLM JSON: {e}")
            return LLMReviewResult(
                approved=False,
                risk_score=8,
                reason=f"LLM response parse error: {e}",
                raw_response=response,
                parse_success=False,
            )

    def _simulate_llm_review(
        self,
        proposal: Proposal,
        fallback: bool = False
    ) -> LLMReviewResult:
        """
        Simulate LLM review for testing without Ollama.
        
        Uses heuristics based on the proposal content.
        """
        self._stats['simulation_mode_uses'] += 1
        
        # Simulate processing time
        time.sleep(random.uniform(0.1, 0.3))
        
        # Heuristic-based risk assessment
        risk_score = 3  # Base risk
        reasons = []
        
        diff = proposal.code_diff.lower()
        
        # Check for risky patterns
        if 'delete' in diff or 'drop' in diff or 'remove' in diff:
            risk_score += 2
            reasons.append("destructive operation")
        
        if 'password' in diff or 'secret' in diff or 'key' in diff:
            risk_score += 2
            reasons.append("sensitive data handling")
        
        if 'exec(' in diff or 'eval(' in diff or 'system(' in diff:
            risk_score += 3
            reasons.append("dynamic code execution")
        
        if 'sql' in diff and ('format' in diff or '%s' not in diff):
            risk_score += 2
            reasons.append("potential SQL injection")
        
        # Positive signals
        if 'test' in diff:
            risk_score -= 1
            reasons.append("includes tests")
        
        if proposal.reasoning_chain:
            risk_score -= 1
            reasons.append("has reasoning")
        
        # Clamp risk score
        risk_score = max(0, min(10, risk_score))
        
        # Random factor for simulation variety
        if random.random() > self.config.simulation_approval_rate:
            risk_score = max(risk_score, 8)
        
        approved = risk_score <= self.config.risk_threshold
        
        reason_text = f"[SIMULATION] Risk assessment: {', '.join(reasons) if reasons else 'standard change'}"
        if fallback:
            reason_text = f"[FALLBACK] {reason_text}"
        
        return LLMReviewResult(
            approved=approved,
            risk_score=risk_score,
            reason=reason_text,
            raw_response="<simulation>",
            parse_success=True,
        )

    # =========================================================================
    # UTILITIES
    # =========================================================================

    @property
    def stats(self) -> Dict[str, int]:
        """Get analysis statistics."""
        return self._stats.copy()

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        for key in self._stats:
            self._stats[key] = 0

    def print_stats(self) -> None:
        """Print analysis statistics."""
        print("\n📊 ReasoningCore Statistics:")
        print(f"   Total analyzed:        {self._stats['total_analyzed']}")
        print(f"   Fast check rejects:    {self._stats['fast_check_rejects']}")
        print(f"   Integrity rejects:     {self._stats['integrity_check_rejects']}")
        print(f"   Deep check rejects:    {self._stats['deep_check_rejects']}")
        print(f"   Approvals:             {self._stats['approvals']}")
        print(f"   Simulation mode uses:  {self._stats['simulation_mode_uses']}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_reasoning_core(
    codebase_path: str,
    simulation_mode: bool = False,
    **config_kwargs
) -> ReasoningCore:
    """
    Convenience function to create a fully-configured ReasoningCore.
    
    Args:
        codebase_path: Path to the codebase root.
        simulation_mode: If True, skip LLM calls.
        **config_kwargs: Additional ReasoningConfig options.
        
    Returns:
        Initialized ReasoningCore.
    """
    config = ReasoningConfig(
        simulation_mode=simulation_mode,
        **config_kwargs
    )
    return ReasoningCore.from_codebase(codebase_path, config=config)


def quick_analyze(proposal: Proposal, simulation: bool = True) -> VoteDecision:
    """
    Quick analysis without pre-built indexes.
    
    For testing purposes only.
    """
    config = ReasoningConfig(simulation_mode=simulation)
    core = ReasoningCore(config=config)
    return core.analyze_proposal(proposal)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == '__main__':
    import sys
    
    print("=" * 60)
    print("S.P.I.D.E.R. ReasoningCore Demo")
    print("=" * 60)
    
    # Create a test proposal
    proposal = Proposal(
        code_diff="""
--- a/utils.py
+++ b/utils.py
@@ -10,6 +10,10 @@ def process_data(data):
+    if data is None:
+        raise ValueError("Data cannot be None")
     return data.value
""",
        merkle_root_hash="merkle_test_1234",
        reasoning_chain=[
            "Identified null pointer risk in production logs",
            "Added defensive null check",
            "Includes error message for debugging",
        ],
        target_files=["utils.py"],
        author_id="agent-001",
    )
    
    print(f"\n📋 Test Proposal: {proposal.proposal_id[:12]}...")
    print(f"   Files: {proposal.target_files}")
    print(f"   Reasoning steps: {len(proposal.reasoning_chain)}")
    
    # Create ReasoningCore in simulation mode
    config = ReasoningConfig(
        simulation_mode=True,
        log_level="DEBUG",
    )
    core = ReasoningCore(config=config)
    
    print("\n🔍 Analyzing proposal (simulation mode)...")
    
    result = core.analyze_proposal_detailed(proposal)
    
    print(f"\n📊 Analysis Result:")
    print(f"   Decision:      {result.decision.name}")
    print(f"   Stage Reached: {result.stage_reached.name}")
    print(f"   Confidence:    {result.confidence:.1%}")
    print(f"   Reasoning:     {result.reasoning}")
    print(f"   Duration:      {result.duration_ms:.1f}ms")
    
    if result.llm_review:
        print(f"\n🤖 LLM Review:")
        print(f"   Risk Score:    {result.llm_review.risk_score}/10")
        print(f"   Approved:      {result.llm_review.approved}")
    
    # Print stats
    core.print_stats()
    
    print("\n" + "=" * 60)
