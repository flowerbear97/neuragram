"""LangChain integration for Engram.

Provides an adapter that implements LangChain's BaseMemory interface,
allowing Engram to be used as a drop-in memory backend for LangChain
chains and agents.

Usage::

    from engram.integrations.langchain import EngramMemory

    memory = EngramMemory(db_path="./memory.db", user_id="u1")

    # Use with LangChain
    from langchain.chains import ConversationChain
    chain = ConversationChain(memory=memory, llm=my_llm)

Requires: pip install neuragram[langchain]
"""

from __future__ import annotations

from typing import Any


class EngramMemory:
    """LangChain-compatible memory backed by Engram.

    Implements the LangChain BaseMemory interface pattern:
    - memory_variables: list of output keys
    - load_memory_variables(): retrieve relevant context
    - save_context(): store new information
    - clear(): reset memory

    Args:
        db_path: SQLite database path.
        user_id: User ID for memory scoping.
        agent_id: Agent ID for memory scoping.
        namespace: Namespace for memory grouping.
        memory_key: Key name for the memory variable in chain context.
        input_key: Key for the human input in save_context.
        output_key: Key for the AI output in save_context.
        top_k: Number of memories to retrieve per query.
        embedding: Embedding provider name.
        llm: LLM provider name (optional, for smart features).
        **kwargs: Additional options passed to AgentMemory.
    """

    memory_key: str = "history"

    def __init__(
        self,
        db_path: str = "./engram.db",
        user_id: str = "",
        agent_id: str = "",
        namespace: str = "default",
        memory_key: str = "history",
        input_key: str = "input",
        output_key: str = "output",
        top_k: int = 5,
        embedding: str = "none",
        llm: str | None = None,
        **kwargs: Any,
    ) -> None:
        from engram.client import AgentMemory

        self._memory = AgentMemory(
            db_path=db_path,
            embedding=embedding,
            llm=llm,
            **kwargs,
        )
        self._user_id = user_id
        self._agent_id = agent_id
        self._namespace = namespace
        self.memory_key = memory_key
        self._input_key = input_key
        self._output_key = output_key
        self._top_k = top_k

    @property
    def memory_variables(self) -> list[str]:
        """Keys this memory will inject into the chain context."""
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        """Retrieve relevant memories based on the current input.

        Args:
            inputs: The current chain inputs (must contain input_key).

        Returns:
            Dict with memory_key mapped to formatted memory string.
        """
        query = inputs.get(self._input_key, "")
        if not query:
            return {self.memory_key: ""}

        results = self._memory.recall(
            query=str(query),
            user_id=self._user_id or None,
            namespace=self._namespace or None,
            top_k=self._top_k,
        )

        if not results:
            return {self.memory_key: ""}

        formatted_memories = []
        for scored in results:
            memory = scored.memory
            formatted_memories.append(
                f"[{memory.memory_type.value}] {memory.content}"
            )

        return {self.memory_key: "\n".join(formatted_memories)}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """Save the current interaction to memory.

        Stores both the human input and AI output as episode memories.

        Args:
            inputs: The chain inputs.
            outputs: The chain outputs.
        """
        human_input = inputs.get(self._input_key, "")
        ai_output = outputs.get(self._output_key, "")

        if human_input:
            self._memory.smart_remember(
                content=f"User said: {human_input}",
                user_id=self._user_id,
                agent_id=self._agent_id,
                namespace=self._namespace,
            )

        if ai_output:
            self._memory.smart_remember(
                content=f"Assistant responded: {ai_output}",
                user_id=self._user_id,
                agent_id=self._agent_id,
                namespace=self._namespace,
            )

    def clear(self) -> None:
        """Clear all memories for this user/namespace scope."""
        if self._user_id:
            self._memory.forget(user_id=self._user_id)

    async def aload_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        """Async version of load_memory_variables."""
        query = inputs.get(self._input_key, "")
        if not query:
            return {self.memory_key: ""}

        results = await self._memory.arecall(
            query=str(query),
            user_id=self._user_id or None,
            namespace=self._namespace or None,
            top_k=self._top_k,
        )

        if not results:
            return {self.memory_key: ""}

        formatted_memories = []
        for scored in results:
            memory = scored.memory
            formatted_memories.append(
                f"[{memory.memory_type.value}] {memory.content}"
            )

        return {self.memory_key: "\n".join(formatted_memories)}

    async def asave_context(
        self, inputs: dict[str, Any], outputs: dict[str, str]
    ) -> None:
        """Async version of save_context."""
        human_input = inputs.get(self._input_key, "")
        ai_output = outputs.get(self._output_key, "")

        if human_input:
            await self._memory.asmart_remember(
                content=f"User said: {human_input}",
                user_id=self._user_id,
                agent_id=self._agent_id,
                namespace=self._namespace,
            )

        if ai_output:
            await self._memory.asmart_remember(
                content=f"Assistant responded: {ai_output}",
                user_id=self._user_id,
                agent_id=self._agent_id,
                namespace=self._namespace,
            )

    async def aclear(self) -> None:
        """Async version of clear."""
        if self._user_id:
            await self._memory.aforget(user_id=self._user_id)

    def close(self) -> None:
        """Release resources."""
        self._memory.close()

    async def aclose(self) -> None:
        """Async release resources."""
        await self._memory.aclose()
