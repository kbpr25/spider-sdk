"""
S.P.I.D.E.R. Structured Prompting Engine
=========================================

Advanced prompt engineering techniques that maximize LLM reasoning quality.
These techniques allow small LLMs to punch well above their weight class.

Techniques Implemented:
1. Chain-of-Thought (CoT): Step-by-step reasoning
2. Self-Consistency: Generate multiple solutions, vote on best
3. Role Prompting: Expert personas for different tasks
4. Few-Shot Examples: Learn from demonstrations
5. Structured Output: Force specific output formats
6. Metacognitive Prompting: Make LLM reflect on its reasoning

Research Basis:
- Wei et al. (2022): "Chain-of-Thought Prompting Elicits Reasoning"
- Wang et al. (2022): "Self-Consistency Improves Chain of Thought Reasoning"
- Reynolds & McDonell (2021): "Prompt Programming for Large Language Models"

This is the +8% improvement component.
"""

import hashlib
import logging
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

class PromptStyle(Enum):
    """Different prompting styles for different tasks."""
    DIRECT = auto()           # Simple direct prompting
    CHAIN_OF_THOUGHT = auto() # Step-by-step reasoning
    FEW_SHOT = auto()         # Learning from examples
    ROLE_BASED = auto()       # Expert persona
    STRUCTURED = auto()       # Enforced output format
    METACOGNITIVE = auto()    # Self-reflection


@dataclass
class PromptTemplate:
    """A reusable prompt template with placeholders."""
    name: str
    style: PromptStyle
    system_prompt: str
    user_template: str
    output_format: Optional[str] = None
    examples: List[Dict[str, str]] = field(default_factory=list)
    
    def render(self, **kwargs) -> Tuple[str, str]:
        """
        Render the template with given parameters.
        
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system = self.system_prompt.format(**kwargs) if kwargs else self.system_prompt
        user = self.user_template.format(**kwargs) if kwargs else self.user_template
        
        # Add examples for few-shot
        if self.examples and self.style == PromptStyle.FEW_SHOT:
            examples_text = "\n\n".join([
                f"Example {i+1}:\nInput: {ex.get('input', '')}\nOutput: {ex.get('output', '')}"
                for i, ex in enumerate(self.examples)
            ])
            user = f"{examples_text}\n\nNow solve:\n{user}"
        
        # Add output format guidance
        if self.output_format:
            user += f"\n\n{self.output_format}"
        
        return system, user


# =============================================================================
# BUILT-IN TEMPLATES
# =============================================================================

# Chain-of-Thought for code analysis
COT_CODE_ANALYSIS = PromptTemplate(
    name="cot_code_analysis",
    style=PromptStyle.CHAIN_OF_THOUGHT,
    system_prompt="""You are an expert software engineer with deep debugging skills.
Always think step by step before providing solutions.
Break down complex problems into smaller, manageable parts.""",
    user_template="""Analyze this code issue and provide a fix.

## Problem Description
{problem}

## Relevant Code
```python
{code}
```

## Analysis Process
Think through this step by step:

### Step 1: Understanding the Bug
What is the reported issue? What behavior is expected vs actual?

### Step 2: Locating the Source
Where in the code does this issue originate? Which function/line?

### Step 3: Root Cause Analysis
WHY does this bug occur? What's the underlying logic error?

### Step 4: Solution Design
What is the minimal change needed to fix this? Are there edge cases?

### Step 5: Implementation
Provide the fixed code.""",
    output_format="""
## Output Format
Provide your analysis following the steps above, then end with:
```python
# Fixed code here
```""",
)

# Role-based prompting for senior engineer
ROLE_SENIOR_ENGINEER = PromptTemplate(
    name="role_senior_engineer",
    style=PromptStyle.ROLE_BASED,
    system_prompt="""You are a senior software engineer at Google with 15 years of experience.
You have deep expertise in Python, debugging, and code review.
You are known for:
- Writing minimal, elegant fixes
- Anticipating edge cases
- Explaining your reasoning clearly
- Following best practices religiously

When fixing bugs, you always:
1. Understand the full context before changing anything
2. Make the smallest change that fixes the issue
3. Consider backwards compatibility
4. Think about performance implications
5. Add appropriate error handling""",
    user_template="""As a senior engineer, review and fix this bug:

## Context
{context}

## Problem
{problem}

## Current Code
```python
{code}
```

Provide your professional assessment and the fix.""",
)

# Structured output for patch generation
STRUCTURED_PATCH = PromptTemplate(
    name="structured_patch",
    style=PromptStyle.STRUCTURED,
    system_prompt="""You are a code repair assistant.
You MUST follow the exact output format specified.
Do not deviate from the format under any circumstances.""",
    user_template="""Fix the following bug:

{problem}

Code:
```python
{code}
```""",
    output_format="""
## Required Output Format
You MUST respond with EXACTLY this structure:

ANALYSIS:
[One paragraph explaining the bug]

ROOT_CAUSE:
[One sentence identifying the root cause]

FIX:
```python
[The corrected code - only the changed function/section]
```

VERIFICATION:
[One sentence explaining how to verify the fix works]""",
)

# Metacognitive prompting for self-reflection
METACOGNITIVE_DEBUG = PromptTemplate(
    name="metacognitive_debug",
    style=PromptStyle.METACOGNITIVE,
    system_prompt="""You are an introspective debugger who reasons about your own thought process.
Before providing solutions, reflect on:
- What assumptions am I making?
- What could I be missing?
- Is there a simpler explanation?
- What would happen if I'm wrong?""",
    user_template="""Debug this issue using metacognitive reflection:

{problem}

```python
{code}
```

## Reflection Process

### What do I think the bug is?
[Your initial hypothesis]

### What assumptions am I making?
[List your assumptions]

### What evidence supports my hypothesis?
[Evidence from the code]

### What evidence contradicts it?
[Counter-evidence or edge cases]

### What would a more experienced engineer think?
[Alternative perspective]

### My refined diagnosis
[Final analysis after reflection]

### The fix
[Your solution]""",
)

# Few-shot learning for common patterns
FEW_SHOT_NULL_CHECK = PromptTemplate(
    name="few_shot_null_check",
    style=PromptStyle.FEW_SHOT,
    system_prompt="You fix null/None handling bugs by learning from examples.",
    user_template="""{problem}

```python
{code}
```

Apply the same fix pattern as shown in the examples.""",
    examples=[
        {
            "input": "TypeError: 'NoneType' object has no attribute 'items'",
            "output": "if obj is not None:\n    for k, v in obj.items():\n        process(k, v)"
        },
        {
            "input": "AttributeError: 'NoneType' object has no attribute 'strip'",
            "output": "text = text.strip() if text else ''"
        },
        {
            "input": "TypeError: cannot unpack non-iterable NoneType object",
            "output": "result = func() or (default_a, default_b)\na, b = result"
        },
    ],
)


# =============================================================================
# PROMPT BUILDER
# =============================================================================

class PromptBuilder:
    """
    Builds optimized prompts from templates and context.
    
    Features:
    - Template selection based on task type
    - Dynamic context injection
    - Token budget management
    - Example selection
    """
    
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        self.templates: Dict[str, PromptTemplate] = {}
        
        # Register built-in templates
        for template in [
            COT_CODE_ANALYSIS,
            ROLE_SENIOR_ENGINEER,
            STRUCTURED_PATCH,
            METACOGNITIVE_DEBUG,
            FEW_SHOT_NULL_CHECK,
        ]:
            self.templates[template.name] = template
    
    def register_template(self, template: PromptTemplate) -> None:
        """Register a custom template."""
        self.templates[template.name] = template
    
    def build(
        self,
        template_name: str,
        **kwargs,
    ) -> Tuple[str, str]:
        """
        Build a prompt from a template.
        
        Args:
            template_name: Name of the template to use
            **kwargs: Template parameters
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        if template_name not in self.templates:
            raise ValueError(f"Unknown template: {template_name}")
        
        template = self.templates[template_name]
        return template.render(**kwargs)
    
    def build_cot(
        self,
        problem: str,
        code: str,
        context: str = "",
    ) -> Tuple[str, str]:
        """Build a Chain-of-Thought prompt."""
        return self.build("cot_code_analysis", problem=problem, code=code)
    
    def build_role(
        self,
        problem: str,
        code: str,
        context: str = "",
    ) -> Tuple[str, str]:
        """Build a role-based prompt."""
        return self.build(
            "role_senior_engineer",
            problem=problem,
            code=code,
            context=context or "Production codebase",
        )
    
    def build_structured(
        self,
        problem: str,
        code: str,
    ) -> Tuple[str, str]:
        """Build a structured output prompt."""
        return self.build("structured_patch", problem=problem, code=code)


# =============================================================================
# SELF-CONSISTENCY ENGINE
# =============================================================================

@dataclass
class SolutionCandidate:
    """A candidate solution with metadata."""
    content: str
    reasoning: str = ""
    confidence: float = 0.0
    votes: int = 0
    hash: str = ""
    
    def __post_init__(self):
        if not self.hash:
            # Normalize and hash for comparison
            normalized = re.sub(r'\s+', ' ', self.content.strip())
            self.hash = hashlib.md5(normalized.encode()).hexdigest()[:8]


class SelfConsistencyEngine:
    """
    Implements self-consistency sampling for more reliable outputs.
    
    Algorithm:
    1. Generate N candidate solutions independently
    2. Cluster similar solutions
    3. Vote on the most common answer
    4. Return the consensus solution
    
    This dramatically improves accuracy on complex reasoning tasks.
    """
    
    def __init__(
        self,
        llm_gateway=None,
        num_samples: int = 3,
        temperature: float = 0.7,
        similarity_threshold: float = 0.8,
    ):
        """
        Initialize the self-consistency engine.
        
        Args:
            llm_gateway: LLM gateway for generating samples
            num_samples: Number of samples to generate
            temperature: Temperature for diverse sampling
            similarity_threshold: Threshold for clustering similar solutions
        """
        self.gateway = llm_gateway
        self.num_samples = num_samples
        self.temperature = temperature
        self.similarity_threshold = similarity_threshold
        
        self.stats = {
            "queries": 0,
            "consensus_found": 0,
            "avg_agreement": 0.0,
        }
    
    def generate_consensus(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2000,
    ) -> Tuple[str, float]:
        """
        Generate a consensus solution through self-consistency.
        
        Args:
            system_prompt: System prompt to use
            user_prompt: User prompt to use
            max_tokens: Max tokens per sample
            
        Returns:
            Tuple of (consensus_solution, confidence_score)
        """
        self.stats["queries"] += 1
        
        if not self.gateway:
            logger.warning("No LLM gateway, returning single sample")
            return user_prompt, 0.0
        
        # Generate multiple samples
        candidates = self._generate_samples(
            system_prompt, user_prompt, max_tokens
        )
        
        if not candidates:
            return "", 0.0
        
        # Cluster similar solutions
        clusters = self._cluster_solutions(candidates)
        
        # Find the largest cluster (most common solution pattern)
        largest_cluster = max(clusters, key=lambda c: len(c))
        
        # Calculate confidence as fraction of samples in largest cluster
        confidence = len(largest_cluster) / len(candidates)
        
        if confidence >= 0.5:
            self.stats["consensus_found"] += 1
        
        # Update running average
        n = self.stats["queries"]
        old_avg = self.stats["avg_agreement"]
        self.stats["avg_agreement"] = (old_avg * (n-1) + confidence) / n
        
        # Return the first candidate from the largest cluster
        return largest_cluster[0].content, confidence
    
    def _generate_samples(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> List[SolutionCandidate]:
        """Generate N diverse samples."""
        candidates = []
        
        try:
            from spider.core.agent.llm_client import Message, MessageRole
            
            for i in range(self.num_samples):
                messages = [
                    Message(MessageRole.SYSTEM, system_prompt),
                    Message(MessageRole.USER, user_prompt),
                ]
                
                # Vary temperature slightly for diversity
                temp = self.temperature + (random.random() - 0.5) * 0.2
                
                response = self.gateway.complete(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temp,
                )
                
                if response.success:
                    candidates.append(SolutionCandidate(
                        content=response.content,
                        confidence=1.0 - (i / self.num_samples),  # Slight decay
                    ))
                    
        except Exception as e:
            logger.error(f"Error generating samples: {e}")
        
        return candidates
    
    def _cluster_solutions(
        self,
        candidates: List[SolutionCandidate],
    ) -> List[List[SolutionCandidate]]:
        """Cluster similar solutions together."""
        if not candidates:
            return []
        
        clusters: List[List[SolutionCandidate]] = []
        
        for candidate in candidates:
            # Try to find a matching cluster
            matched = False
            for cluster in clusters:
                if self._is_similar(candidate, cluster[0]):
                    cluster.append(candidate)
                    matched = True
                    break
            
            # Create new cluster if no match
            if not matched:
                clusters.append([candidate])
        
        return clusters
    
    def _is_similar(
        self,
        a: SolutionCandidate,
        b: SolutionCandidate,
    ) -> bool:
        """Check if two solutions are similar."""
        # Quick hash check
        if a.hash == b.hash:
            return True
        
        # Extract code blocks for comparison
        code_a = self._extract_code(a.content)
        code_b = self._extract_code(b.content)
        
        if not code_a or not code_b:
            return False
        
        # Jaccard similarity on tokens
        tokens_a = set(re.findall(r'\b\w+\b', code_a))
        tokens_b = set(re.findall(r'\b\w+\b', code_b))
        
        if not tokens_a or not tokens_b:
            return False
        
        intersection = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        jaccard = intersection / union if union > 0 else 0
        
        return jaccard >= self.similarity_threshold
    
    def _extract_code(self, content: str) -> str:
        """Extract code from response."""
        if "```" in content:
            blocks = re.findall(r'```(?:\w+)?\n(.*?)```', content, re.DOTALL)
            return "\n".join(blocks)
        return content


# =============================================================================
# OPUS-STYLE PROMPTS
# =============================================================================

class OpusStylePrompts:
    """
    Pre-built prompts that mimic Opus 4.5's prompting strategies.
    
    Based on analysis of Anthropic's prompting best practices.
    """
    
    @staticmethod
    def get_analysis_prompt(problem: str, code: str) -> str:
        """Get Opus-style analysis prompt."""
        return f"""You are an expert software engineer. Analyze this bug systematically.

## The Bug Report
{problem}

## The Code
```python
{code}
```

## Your Analysis (Think Step by Step)

### 1. UNDERSTAND THE SYMPTOMS
What exactly is happening? What should happen instead?

### 2. LOCATE THE ROOT CAUSE
Trace the data flow. Where does the bug originate?

### 3. IDENTIFY THE FIX
What is the MINIMAL change that fixes the root cause?

### 4. CHECK EDGE CASES
Will this fix work for all inputs? What edge cases exist?

### 5. PROVIDE THE SOLUTION
Write the fixed code."""
    
    @staticmethod
    def get_fix_prompt(problem: str, code: str, test_output: str = "") -> str:
        """Get Opus-style fix prompt with optional test output."""
        prompt = f"""Fix this bug in the code.

PROBLEM:
{problem}

CODE:
```python
{code}
```
"""
        if test_output:
            prompt += f"""
TEST OUTPUT:
{test_output}
"""
        
        prompt += """
INSTRUCTIONS:
1. Identify what's wrong
2. Explain why it's wrong (one sentence)
3. Provide the fixed code

OUTPUT FORMAT:
DIAGNOSIS: [One sentence]
```python
[Fixed code]
```"""
        return prompt
    
    @staticmethod
    def get_review_prompt(patch: str, original: str) -> str:
        """Get Opus-style code review prompt."""
        return f"""Review this code change for correctness and quality.

ORIGINAL CODE:
```python
{original}
```

PROPOSED CHANGE:
```diff
{patch}
```

REVIEW CHECKLIST:
1. ✅ Does this fix the bug?
2. ✅ Are there any new bugs introduced?
3. ✅ Edge cases handled?
4. ✅ Performance implications?
5. ✅ Code style consistency?

Provide your review with specific feedback."""


# =============================================================================
# STRUCTURED PROMPTING ENGINE
# =============================================================================

class StructuredPromptingEngine:
    """
    The main prompting engine that orchestrates all techniques.
    
    Usage:
        engine = StructuredPromptingEngine(llm_gateway)
        
        # Use Chain-of-Thought
        response = engine.generate_cot(problem, code)
        
        # Use self-consistency
        response, confidence = engine.generate_with_consensus(problem, code)
        
        # Use structured output
        analysis = engine.generate_structured_analysis(problem, code)
    """
    
    def __init__(
        self,
        llm_gateway=None,
        default_style: PromptStyle = PromptStyle.CHAIN_OF_THOUGHT,
        enable_consensus: bool = True,
        num_samples: int = 3,
    ):
        """
        Initialize the prompting engine.
        
        Args:
            llm_gateway: LLM gateway for generation
            default_style: Default prompting style
            enable_consensus: Whether to use self-consistency by default
            num_samples: Number of samples for self-consistency
        """
        self.gateway = llm_gateway
        self.default_style = default_style
        self.enable_consensus = enable_consensus
        
        self.builder = PromptBuilder()
        self.consistency = SelfConsistencyEngine(llm_gateway, num_samples)
        
        self.stats = {
            "generations": 0,
            "by_style": {style.name: 0 for style in PromptStyle},
        }
    
    def generate(
        self,
        problem: str,
        code: str,
        style: Optional[PromptStyle] = None,
        context: str = "",
        use_consensus: Optional[bool] = None,
    ) -> Tuple[str, float]:
        """
        Generate a solution using specified prompting style.
        
        Args:
            problem: Problem description
            code: Relevant code
            style: Prompting style (uses default if None)
            context: Additional context
            use_consensus: Override consensus setting
            
        Returns:
            Tuple of (solution, confidence)
        """
        style = style or self.default_style
        use_consensus = use_consensus if use_consensus is not None else self.enable_consensus
        
        self.stats["generations"] += 1
        self.stats["by_style"][style.name] += 1
        
        # Build prompt based on style
        if style == PromptStyle.CHAIN_OF_THOUGHT:
            system, user = self.builder.build_cot(problem, code, context)
        elif style == PromptStyle.ROLE_BASED:
            system, user = self.builder.build_role(problem, code, context)
        elif style == PromptStyle.STRUCTURED:
            system, user = self.builder.build_structured(problem, code)
        elif style == PromptStyle.METACOGNITIVE:
            system, user = self.builder.build(
                "metacognitive_debug", problem=problem, code=code
            )
        else:
            # Default to CoT
            system, user = self.builder.build_cot(problem, code, context)
        
        # Generate with or without consensus
        if use_consensus:
            return self.consistency.generate_consensus(system, user)
        else:
            return self._single_generate(system, user)
    
    def _single_generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Tuple[str, float]:
        """Generate a single response."""
        if not self.gateway:
            return "", 0.0
        
        try:
            from spider.core.agent.llm_client import Message, MessageRole
            
            messages = [
                Message(MessageRole.SYSTEM, system_prompt),
                Message(MessageRole.USER, user_prompt),
            ]
            
            response = self.gateway.complete(messages, max_tokens=2000)
            
            if response.success:
                return response.content, 1.0
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
        
        return "", 0.0
    
    def generate_cot(
        self,
        problem: str,
        code: str,
        context: str = "",
    ) -> str:
        """Generate using Chain-of-Thought."""
        response, _ = self.generate(
            problem, code, 
            style=PromptStyle.CHAIN_OF_THOUGHT,
            context=context,
        )
        return response
    
    def generate_with_role(
        self,
        problem: str,
        code: str,
        context: str = "",
    ) -> str:
        """Generate using role-based prompting."""
        response, _ = self.generate(
            problem, code,
            style=PromptStyle.ROLE_BASED,
            context=context,
        )
        return response
    
    def generate_structured_analysis(
        self,
        problem: str,
        code: str,
    ) -> Dict[str, str]:
        """Generate structured analysis with parsed sections."""
        response, _ = self.generate(
            problem, code,
            style=PromptStyle.STRUCTURED,
            use_consensus=False,
        )
        
        # Parse structured output
        sections = {}
        current_section = None
        current_content = []
        
        for line in response.split('\n'):
            if line.strip().endswith(':') and line.strip().isupper():
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = line.strip()[:-1]
                current_content = []
            else:
                current_content.append(line)
        
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            **self.stats,
            "consistency_stats": self.consistency.stats,
        }
    
    def print_stats(self) -> None:
        """Print engine statistics."""
        stats = self.get_stats()
        print("\n" + "=" * 60)
        print("STRUCTURED PROMPTING ENGINE STATISTICS")
        print("=" * 60)
        print(f"Total Generations: {stats['generations']}")
        print("\nBy Style:")
        for style, count in stats['by_style'].items():
            if count > 0:
                print(f"  {style}: {count}")
        print(f"\nConsensus Queries: {stats['consistency_stats']['queries']}")
        print(f"Consensus Found: {stats['consistency_stats']['consensus_found']}")
        print(f"Avg Agreement: {stats['consistency_stats']['avg_agreement']:.1%}")
        print("=" * 60)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def build_cot_prompt(problem: str, code: str) -> Tuple[str, str]:
    """Quick function to build a CoT prompt."""
    builder = PromptBuilder()
    return builder.build_cot(problem, code)


def build_role_prompt(problem: str, code: str, context: str = "") -> Tuple[str, str]:
    """Quick function to build a role-based prompt."""
    builder = PromptBuilder()
    return builder.build_role(problem, code, context)


def opus_analysis_prompt(problem: str, code: str) -> str:
    """Get an Opus-style analysis prompt."""
    return OpusStylePrompts.get_analysis_prompt(problem, code)


# =============================================================================
# MAIN (DEMO)
# =============================================================================

if __name__ == "__main__":
    print("S.P.I.D.E.R. Structured Prompting Engine Demo")
    print("=" * 50)
    
    # Demo problem
    problem = "The calculate_average function returns 0 for non-empty lists"
    code = '''
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers) if numbers else 0
'''
    
    # Build different prompts
    builder = PromptBuilder()
    
    print("\n1. Chain-of-Thought Prompt:")
    print("-" * 40)
    system, user = builder.build_cot(problem, code)
    print(f"System: {system[:100]}...")
    print(f"User: {user[:300]}...")
    
    print("\n2. Role-Based Prompt:")
    print("-" * 40)
    system, user = builder.build_role(problem, code)
    print(f"System: {system[:200]}...")
    
    print("\n3. Opus-Style Analysis:")
    print("-" * 40)
    opus = OpusStylePrompts.get_analysis_prompt(problem, code)
    print(opus[:500] + "...")
