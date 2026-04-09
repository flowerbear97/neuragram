"""Automatic memory classification.

Given raw text content, automatically determines:
- MemoryType (fact, preference, episode, procedure, plan_state)
- Importance score [0, 1]
- Confidence score [0, 1]
- Suggested tags

Supports two modes:
1. **LLM-based**: Uses an LLM for high-accuracy classification
2. **Rule-based**: Fast heuristic classification without LLM dependency

Usage::

    # With LLM
    classifier = MemoryClassifier(llm_provider=my_llm)
    result = await classifier.classify("User prefers dark mode")
    # result.memory_type == MemoryType.PREFERENCE

    # Without LLM (rule-based fallback)
    classifier = MemoryClassifier()
    result = await classifier.classify("User prefers dark mode")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from neuragram.core.models import MemoryType
from neuragram.processing.llm import BaseLLMProvider, LLMError

_CLASSIFICATION_PROMPT = """You are a memory classification engine. Given a piece of text, classify it into the most appropriate memory type and assign quality scores.

Memory types:
- **fact**: Factual information about a person, system, or environment
- **preference**: Opinions, likes, dislikes, style preferences
- **episode**: Events, incidents, experiences with temporal context
- **procedure**: Processes, workflows, step-by-step instructions
- **plan_state**: Current task status, in-progress work, temporary state

Respond with a JSON object:
{
  "type": "fact|preference|episode|procedure|plan_state",
  "importance": 0.7,
  "confidence": 0.9,
  "tags": ["tag1", "tag2"],
  "reasoning": "brief explanation"
}"""


@dataclass
class ClassificationResult:
    """Result of classifying a memory."""

    memory_type: MemoryType = MemoryType.FACT
    importance: float = 0.5
    confidence: float = 0.8
    tags: list[str] = field(default_factory=list)
    reasoning: str = ""
    method: str = "rule"  # "rule" or "llm"


class MemoryClassifier:
    """Classifies memory content by type, importance, and confidence.

    Args:
        llm_provider: Optional LLM for high-accuracy classification.
            If None, falls back to rule-based heuristics.
    """

    def __init__(self, llm_provider: BaseLLMProvider | None = None) -> None:
        self._llm = llm_provider

    async def classify(self, content: str) -> ClassificationResult:
        """Classify a piece of text.

        Tries LLM-based classification first (if available),
        falls back to rule-based heuristics.

        Args:
            content: The text to classify.

        Returns:
            ClassificationResult with type, scores, and tags.
        """
        if self._llm is not None:
            try:
                return await self._classify_with_llm(content)
            except (LLMError, Exception):
                pass  # Fall through to rule-based

        return self._classify_with_rules(content)

    async def _classify_with_llm(self, content: str) -> ClassificationResult:
        """Classify using LLM."""
        assert self._llm is not None

        raw = await self._llm.complete_json(
            system_prompt=_CLASSIFICATION_PROMPT,
            user_message=content,
        )

        type_str = str(raw.get("type", "fact")).lower()
        try:
            memory_type = MemoryType(type_str)
        except ValueError:
            memory_type = MemoryType.FACT

        importance = max(0.0, min(1.0, float(raw.get("importance", 0.5))))
        confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.8))))

        tags = raw.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        tags = [str(t) for t in tags if t]

        return ClassificationResult(
            memory_type=memory_type,
            importance=importance,
            confidence=confidence,
            tags=tags,
            reasoning=str(raw.get("reasoning", "")),
            method="llm",
        )

    @staticmethod
    def _classify_with_rules(content: str) -> ClassificationResult:
        """Classify using keyword-based heuristics.

        This is a fast fallback when no LLM is available. It uses
        pattern matching to guess the memory type and assign scores.
        """
        lower = content.lower()

        # Preference patterns
        preference_patterns = [
            r"\b(prefer|like|dislike|hate|love|enjoy|favor|want)\b",
            r"\b(style|mode|theme|format)\b",
        ]
        for pattern in preference_patterns:
            if re.search(pattern, lower):
                return ClassificationResult(
                    memory_type=MemoryType.PREFERENCE,
                    importance=0.6,
                    confidence=0.7,
                    tags=["auto-classified"],
                    reasoning="Matched preference keyword pattern",
                    method="rule",
                )

        # Episode patterns (temporal markers)
        episode_patterns = [
            r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b",
            r"\b(yesterday|today|last week|last month|ago)\b",
            r"\b(happened|occurred|incident|resolved|fixed|debugged|encountered)\b",
        ]
        for pattern in episode_patterns:
            if re.search(pattern, lower):
                return ClassificationResult(
                    memory_type=MemoryType.EPISODE,
                    importance=0.5,
                    confidence=0.7,
                    tags=["auto-classified"],
                    reasoning="Matched episode/temporal keyword pattern",
                    method="rule",
                )

        # Procedure patterns
        procedure_patterns = [
            r"\b(step\s*\d|first.*then|workflow|process|pipeline)\b",
            r"\b(how to|instructions|guide)\b",
            r"\b(deploy|build|install|configure|setup)\b",
        ]
        for pattern in procedure_patterns:
            if re.search(pattern, lower):
                return ClassificationResult(
                    memory_type=MemoryType.PROCEDURE,
                    importance=0.6,
                    confidence=0.7,
                    tags=["auto-classified"],
                    reasoning="Matched procedure/workflow keyword pattern",
                    method="rule",
                )

        # Plan state patterns
        plan_state_patterns = [
            r"\b(currently|working on|in progress|debugging|investigating)\b",
            r"\b(task|todo|blocked|waiting)\b",
        ]
        for pattern in plan_state_patterns:
            if re.search(pattern, lower):
                return ClassificationResult(
                    memory_type=MemoryType.PLAN_STATE,
                    importance=0.4,
                    confidence=0.6,
                    tags=["auto-classified"],
                    reasoning="Matched plan/state keyword pattern",
                    method="rule",
                )

        # Default: fact
        return ClassificationResult(
            memory_type=MemoryType.FACT,
            importance=0.5,
            confidence=0.6,
            tags=["auto-classified"],
            reasoning="No specific pattern matched, defaulting to fact",
            method="rule",
        )
