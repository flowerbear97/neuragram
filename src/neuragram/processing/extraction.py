"""Memory extraction from raw conversation text."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from neuragram.core.models import Memory, MemoryType
from neuragram.processing.llm import BaseLLMProvider, LLMError

_EXTRACTION_SYSTEM_PROMPT = """You are a memory extraction engine. Your job is to analyze conversations and extract important, persistent memories that would be useful to recall in future interactions.

Extract memories in these categories:
- **fact**: Factual information about the user or their environment (e.g., "User is a Python developer", "User works at Company X")
- **preference**: User preferences and opinions (e.g., "User prefers concise explanations", "User likes dark mode")
- **episode**: Notable events or experiences (e.g., "User debugged a Redis timeout issue on 2024-03-15")
- **procedure**: Processes, workflows, or how-to knowledge (e.g., "Deployment steps: run tests, build, push to main")

Rules:
1. Only extract information that is worth remembering long-term
2. Each memory should be a single, self-contained statement
3. Do NOT extract trivial greetings, acknowledgments, or transient information
4. Prefer specific facts over vague statements
5. If the conversation contains no extractable memories, return an empty list
6. Assign importance (0.0-1.0): how critical is this information?
7. Assign confidence (0.0-1.0): how certain are you this is accurate?
8. Include relevant tags for categorization

Respond with a JSON object:
{
  "memories": [
    {
      "content": "the memory text",
      "type": "fact|preference|episode|procedure",
      "importance": 0.7,
      "confidence": 0.9,
      "tags": ["tag1", "tag2"]
    }
  ]
}"""

_EXTRACTION_FROM_TEXT_PROMPT = """You are a memory extraction engine. Analyze the following text and extract important, persistent memories.

Extract memories in these categories:
- **fact**: Factual information
- **preference**: Preferences and opinions
- **episode**: Notable events or experiences
- **procedure**: Processes, workflows, or how-to knowledge

Rules:
1. Only extract information worth remembering long-term
2. Each memory should be a single, self-contained statement
3. Assign importance (0.0-1.0) and confidence (0.0-1.0)
4. Include relevant tags

Respond with a JSON object:
{
  "memories": [
    {
      "content": "the memory text",
      "type": "fact|preference|episode|procedure",
      "importance": 0.7,
      "confidence": 0.9,
      "tags": ["tag1", "tag2"]
    }
  ]
}"""


@dataclass
class ExtractionResult:
    """Result of a memory extraction operation."""

    memories: list[Memory] = field(default_factory=list)
    raw_response: dict[str, Any] = field(default_factory=dict)
    model_used: str = ""


class MemoryExtractor:
    """Extracts structured memories from conversations using an LLM.

    Args:
        llm_provider: The LLM provider to use for extraction.
    """

    def __init__(self, llm_provider: BaseLLMProvider) -> None:
        self._llm = llm_provider

    async def extract_from_conversation(
        self,
        messages: list[dict[str, str]],
        user_id: str = "",
        agent_id: str = "",
        namespace: str = "default",
    ) -> ExtractionResult:
        """Extract memories from a conversation message list.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."} dicts.
            user_id: Owner of the extracted memories.
            agent_id: Agent that participated in the conversation.
            namespace: Logical grouping.

        Returns:
            ExtractionResult containing parsed Memory objects.
        """
        conversation_text = self._format_conversation(messages)

        try:
            raw = await self._llm.complete_json(
                system_prompt=_EXTRACTION_SYSTEM_PROMPT,
                user_message=conversation_text,
            )
        except LLMError:
            return ExtractionResult(model_used=self._llm.model_name)

        memories = self._parse_extraction_response(
            raw, user_id=user_id, agent_id=agent_id, namespace=namespace
        )

        return ExtractionResult(
            memories=memories,
            raw_response=raw,
            model_used=self._llm.model_name,
        )

    async def extract_from_text(
        self,
        text: str,
        user_id: str = "",
        agent_id: str = "",
        namespace: str = "default",
    ) -> ExtractionResult:
        """Extract memories from arbitrary text.

        Args:
            text: Raw text to extract memories from.
            user_id: Owner of the extracted memories.
            agent_id: Agent context.
            namespace: Logical grouping.

        Returns:
            ExtractionResult containing parsed Memory objects.
        """
        try:
            raw = await self._llm.complete_json(
                system_prompt=_EXTRACTION_FROM_TEXT_PROMPT,
                user_message=text,
            )
        except LLMError:
            return ExtractionResult(model_used=self._llm.model_name)

        memories = self._parse_extraction_response(
            raw, user_id=user_id, agent_id=agent_id, namespace=namespace
        )

        return ExtractionResult(
            memories=memories,
            raw_response=raw,
            model_used=self._llm.model_name,
        )

    @staticmethod
    def _format_conversation(messages: list[dict[str, str]]) -> str:
        """Format a message list into a readable conversation string."""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _parse_extraction_response(
        raw: dict[str, Any],
        user_id: str = "",
        agent_id: str = "",
        namespace: str = "default",
    ) -> list[Memory]:
        """Parse the LLM JSON response into Memory objects."""
        memories_data = raw.get("memories", [])
        if not isinstance(memories_data, list):
            return []

        results: list[Memory] = []
        for item in memories_data:
            if not isinstance(item, dict) or "content" not in item:
                continue

            content = str(item["content"]).strip()
            if not content:
                continue

            # Parse type with fallback
            type_str = str(item.get("type", "fact")).lower()
            try:
                memory_type = MemoryType(type_str)
            except ValueError:
                memory_type = MemoryType.FACT

            # Parse scores with bounds
            importance = max(0.0, min(1.0, float(item.get("importance", 0.5))))
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.8))))

            # Parse tags
            tags = item.get("tags", [])
            if not isinstance(tags, list):
                tags = []
            tags = [str(t) for t in tags if t]

            memory = Memory(
                content=content,
                memory_type=memory_type,
                user_id=user_id,
                agent_id=agent_id,
                namespace=namespace,
                importance=importance,
                confidence=confidence,
                tags=tags,
                source="llm_extraction",
            )
            results.append(memory)

        return results
