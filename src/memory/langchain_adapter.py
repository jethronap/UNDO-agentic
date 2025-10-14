"""
LangChain Memory Adapter Module.

This module provides adapters between the existing SQLite-based memory system
and LangChain's memory interfaces, enabling LangChain agents to use the
existing memory store while maintaining all caching functionality.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.memory import BaseMemory

from src.config.logger import logger
from src.config.settings import DatabaseSettings, LangChainSettings
from src.memory.store import MemoryStore
from src.memory.models import Memory


class SQLiteChatMessageHistory(BaseChatMessageHistory):
    """
    LangChain chat message history implementation using SQLite storage.

    This adapter allows LangChain agents to use the existing SQLite memory
    system for conversation history while maintaining compatibility with
    LangChain's memory interfaces.
    """

    def __init__(
        self,
        session_id: str,
        memory_store: Optional[MemoryStore] = None,
        database_settings: Optional[DatabaseSettings] = None,
    ):
        """
        Initialize SQLite chat message history.

        :param session_id: Unique identifier for this conversation session
        :param memory_store: Existing memory store instance (optional)
        :param database_settings: Database settings for creating new store
        """
        self.session_id = session_id

        if memory_store is not None:
            self.memory_store = memory_store
        elif database_settings is not None:
            self.memory_store = MemoryStore(database_settings)
        else:
            # Use default settings
            self.memory_store = MemoryStore(DatabaseSettings())

        logger.debug(f"Initialized SQLite chat history for session: {session_id}")

    @property
    def messages(self) -> List[BaseMessage]:
        """Get all messages for this session."""
        return self.get_messages()

    def get_messages(self) -> List[BaseMessage]:
        """
        Retrieve all messages for this session from SQLite.

        :return: List of LangChain message objects
        """
        try:
            # Load memories for this session
            memories = self.memory_store.load(agent_id=self.session_id)

            messages = []
            for memory in memories:
                if memory.step.startswith("message_"):
                    message = self._parse_message_from_memory(memory)
                    if message:
                        messages.append(message)

            logger.debug(
                f"Retrieved {len(messages)} messages for session {self.session_id}"
            )
            return messages

        except Exception as e:
            logger.error(f"Failed to get messages for session {self.session_id}: {e}")
            return []

    def add_message(self, message: BaseMessage) -> None:
        """
        Add a new message to the conversation history.

        :param message: LangChain message object to store
        """
        try:
            # Determine message type and create step identifier
            if isinstance(message, HumanMessage):
                step = "message_human"
            elif isinstance(message, AIMessage):
                step = "message_ai"
            elif isinstance(message, SystemMessage):
                step = "message_system"
            else:
                step = f"message_{message.__class__.__name__.lower()}"

            # Serialize message content with metadata
            content = self._serialize_message(message)

            # Store in memory system
            self.memory_store.store(
                agent_id=self.session_id, step=step, content=content
            )

            logger.debug(f"Added {step} to session {self.session_id}")

        except Exception as e:
            logger.error(f"Failed to add message to session {self.session_id}: {e}")
            raise

    def clear(self) -> None:
        """
        Clear all messages for this session.

        Note: This implementation doesn't actually delete from SQLite,
        but marks the session as cleared to avoid breaking existing caching.
        """
        try:
            # Add a clear marker instead of deleting existing memories
            self.memory_store.store(
                agent_id=self.session_id,
                step="session_cleared",
                content=f"Session cleared at {datetime.now().isoformat()}",
            )

            logger.info(f"Marked session {self.session_id} as cleared")

        except Exception as e:
            logger.error(f"Failed to clear session {self.session_id}: {e}")
            raise

    @staticmethod
    def _serialize_message(self, message: BaseMessage) -> str:
        """
        Serialize a LangChain message to string format for storage.

        :param message: LangChain message object
        :return: Serialized string representation
        """
        import json

        serialized = {
            "type": message.__class__.__name__,
            "content": message.content,
            "additional_kwargs": getattr(message, "additional_kwargs", {}),
        }

        # Add response metadata if present (for AI messages)
        if hasattr(message, "response_metadata"):
            serialized["response_metadata"] = message.response_metadata

        return json.dumps(serialized)

    @staticmethod
    def _parse_message_from_memory(self, memory: Memory) -> Optional[BaseMessage]:
        """
        Parse a stored memory back into a LangChain message object.

        :param memory: Memory object from SQLite
        :return: LangChain message object or None if parsing fails
        """
        try:
            import json

            # Skip clear markers and non-message entries
            if memory.step == "session_cleared":
                return None

            data = json.loads(memory.content)
            message_type = data.get("type", "")
            content = data.get("content", "")

            # Create appropriate message type
            if message_type == "HumanMessage":
                return HumanMessage(content=content)
            elif message_type == "AIMessage":
                additional_kwargs = data.get("additional_kwargs", {})
                response_metadata = data.get("response_metadata", {})
                return AIMessage(
                    content=content,
                    additional_kwargs=additional_kwargs,
                    response_metadata=response_metadata,
                )
            elif message_type == "SystemMessage":
                return SystemMessage(content=content)
            else:
                logger.warning(f"Unknown message type: {message_type}")
                return None

        except Exception as e:
            logger.warning(f"Failed to parse message from memory {memory.id}: {e}")
            return None


class SQLiteAgentMemory(BaseMemory):
    """
    LangChain memory implementation for agents using SQLite storage.

    This memory class maintains agent context and conversation history
    while preserving the existing caching behavior for tools and actions.
    """

    def __init__(
        self,
        session_id: str,
        memory_store: Optional[MemoryStore] = None,
        langchain_settings: Optional[LangChainSettings] = None,
        return_messages: bool = False,
    ):
        """
        Initialize agent memory with SQLite backend.

        :param session_id: Unique identifier for this memory session
        :param memory_store: Existing memory store instance
        :param langchain_settings: LangChain configuration
        :param return_messages: Whether to return messages or string summary
        """
        self.session_id = session_id
        self.return_messages = return_messages

        if langchain_settings is None:
            langchain_settings = LangChainSettings()
        self.settings = langchain_settings

        # Initialize chat history
        self.chat_memory = SQLiteChatMessageHistory(
            session_id=session_id, memory_store=memory_store
        )

        logger.info(f"Initialized SQLite agent memory for session: {session_id}")

    @property
    def memory_variables(self) -> List[str]:
        """Return list of memory variables this memory class provides."""
        return ["history"]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load memory variables for the current conversation.

        :param inputs: Current input variables (not used for history)
        :return: Dictionary with conversation history
        """
        try:
            messages = self.chat_memory.get_messages()

            if self.return_messages:
                # Return messages as objects
                history = messages
            else:
                # Return as formatted string
                history = self._format_messages_as_string(messages)

            # Apply token limit if configured
            if self.settings.memory_max_tokens > 0:
                history = self._truncate_to_token_limit(history)

            return {"history": history}

        except Exception as e:
            logger.error(
                f"Failed to load memory variables for session {self.session_id}: {e}"
            )
            return {"history": "" if not self.return_messages else []}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """
        Save the current interaction context to memory.

        :param inputs: Input variables from current interaction
        :param outputs: Output variables from current interaction
        """
        try:
            # Extract user input
            user_input = inputs.get("input", inputs.get("question", ""))
            if user_input:
                self.chat_memory.add_message(HumanMessage(content=user_input))

            # Extract AI response
            ai_output = outputs.get("output", outputs.get("answer", ""))
            if ai_output:
                self.chat_memory.add_message(AIMessage(content=ai_output))

            logger.debug(f"Saved context for session {self.session_id}")

        except Exception as e:
            logger.error(f"Failed to save context for session {self.session_id}: {e}")

    def clear(self) -> None:
        """Clear all memory for this session."""
        try:
            self.chat_memory.clear()
            logger.info(f"Cleared memory for session {self.session_id}")
        except Exception as e:
            logger.error(f"Failed to clear memory for session {self.session_id}: {e}")
            raise

    @staticmethod
    def _format_messages_as_string(self, messages: List[BaseMessage]) -> str:
        """
        Format messages as a human-readable string.

        :param messages: List of LangChain message objects
        :return: Formatted conversation string
        """
        if not messages:
            return ""

        formatted_lines = []
        for message in messages:
            if isinstance(message, HumanMessage):
                formatted_lines.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                formatted_lines.append(f"AI: {message.content}")
            elif isinstance(message, SystemMessage):
                formatted_lines.append(f"System: {message.content}")
            else:
                formatted_lines.append(
                    f"{message.__class__.__name__}: {message.content}"
                )

        return "\n".join(formatted_lines)

    def _truncate_to_token_limit(self, history: Any) -> Any:
        """
        Truncate history to respect token limits.

        This is a simplified implementation that truncates by character count.
        A more sophisticated implementation would use actual token counting.

        :param history: History to truncate (string or list)
        :return: Truncated history
        """
        if not self.settings.memory_max_tokens:
            return history

        # Rough approximation: 4 characters per token
        char_limit = self.settings.memory_max_tokens * 4

        if isinstance(history, str):
            if len(history) <= char_limit:
                return history
            # Keep the most recent part
            return "..." + history[-char_limit:]

        elif isinstance(history, list) and history:
            # For message lists, remove oldest messages until under limit
            total_chars = sum(
                len(str(msg.content)) for msg in history if hasattr(msg, "content")
            )

            if total_chars <= char_limit:
                return history

            # Keep most recent messages
            truncated = []
            current_chars = 0

            for message in reversed(history):
                msg_chars = (
                    len(str(message.content)) if hasattr(message, "content") else 0
                )
                if current_chars + msg_chars <= char_limit:
                    truncated.insert(0, message)
                    current_chars += msg_chars
                else:
                    break

            return truncated

        return history


# ============================================================================
# Memory Factory Functions
# ============================================================================


def create_agent_memory(
    agent_name: str,
    session_id: Optional[str] = None,
    memory_store: Optional[MemoryStore] = None,
    langchain_settings: Optional[LangChainSettings] = None,
) -> SQLiteAgentMemory:
    """
    Factory function to create agent memory with appropriate configuration.

    :param agent_name: Name of the agent (used in session ID)
    :param session_id: Explicit session ID (optional)
    :param memory_store: Existing memory store instance
    :param langchain_settings: LangChain configuration
    :return: Configured SQLiteAgentMemory instance
    """
    if session_id is None:
        session_id = f"{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return SQLiteAgentMemory(
        session_id=session_id,
        memory_store=memory_store,
        langchain_settings=langchain_settings,
    )


def create_chat_history(
    session_id: str,
    memory_store: Optional[MemoryStore] = None,
    database_settings: Optional[DatabaseSettings] = None,
) -> SQLiteChatMessageHistory:
    """
    Factory function to create chat message history.

    :param session_id: Unique session identifier
    :param memory_store: Existing memory store instance
    :param database_settings: Database configuration
    :return: Configured SQLiteChatMessageHistory instance
    """
    return SQLiteChatMessageHistory(
        session_id=session_id,
        memory_store=memory_store,
        database_settings=database_settings,
    )


def create_pipeline_memory(
    pipeline_name: str = "surveillance_pipeline",
    memory_store: Optional[MemoryStore] = None,
) -> SQLiteAgentMemory:
    """
    Factory function to create memory for the overall pipeline.

    :param pipeline_name: Name of the pipeline
    :param memory_store: Existing memory store instance
    :return: Configured pipeline memory
    """
    session_id = f"{pipeline_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return SQLiteAgentMemory(
        session_id=session_id,
        memory_store=memory_store,
        return_messages=True,  # Pipeline needs access to full message objects
    )
