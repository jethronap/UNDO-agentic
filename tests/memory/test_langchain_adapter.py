"""Tests for LangChain memory adapter module."""

from unittest.mock import patch, MagicMock
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.memory.langchain_adapter import (
    SQLiteChatMessageHistory,
    SQLiteAgentMemory,
    create_agent_memory,
    create_chat_history,
    create_pipeline_memory,
)
from src.memory.models import Memory
from src.config.settings import DatabaseSettings, LangChainSettings


class TestSQLiteChatMessageHistory:
    """Test suite for SQLiteChatMessageHistory."""

    @patch("src.memory.langchain_adapter.MemoryStore")
    def test_initialization(self, mock_memory_store):
        """Test initialization of SQLite chat message history."""
        mock_store_instance = MagicMock()
        mock_memory_store.return_value = mock_store_instance

        history = SQLiteChatMessageHistory(
            session_id="test_session", database_settings=DatabaseSettings()
        )

        assert history.session_id == "test_session"
        assert history.memory_store == mock_store_instance

    @patch("src.memory.langchain_adapter.MemoryStore")
    def test_initialization_with_existing_store(self, mock_memory_store):
        """Test initialization with existing memory store."""
        existing_store = MagicMock()

        history = SQLiteChatMessageHistory(
            session_id="test_session", memory_store=existing_store
        )

        assert history.memory_store == existing_store
        # Should not create new store
        mock_memory_store.assert_not_called()

    def test_add_human_message(self):
        """Test adding human message to history."""
        mock_store = MagicMock()
        history = SQLiteChatMessageHistory("test_session", memory_store=mock_store)

        message = HumanMessage(content="Hello!")
        history.add_message(message)

        # Verify store was called with correct parameters
        mock_store.store.assert_called_once()
        call_args = mock_store.store.call_args

        assert call_args[1]["agent_id"] == "test_session"
        assert call_args[1]["step"] == "message_human"
        assert "Hello!" in call_args[1]["content"]

    def test_add_ai_message(self):
        """Test adding AI message to history."""
        mock_store = MagicMock()
        history = SQLiteChatMessageHistory("test_session", memory_store=mock_store)

        message = AIMessage(
            content="Hi there!",
            additional_kwargs={"model": "test"},
            response_metadata={"tokens": 100},
        )
        history.add_message(message)

        # Verify store was called with correct parameters
        mock_store.store.assert_called_once()
        call_args = mock_store.store.call_args

        assert call_args[1]["agent_id"] == "test_session"
        assert call_args[1]["step"] == "message_ai"
        # Content should be JSON serialized
        content = call_args[1]["content"]
        assert "Hi there!" in content
        assert "model" in content
        assert "tokens" in content

    def test_add_system_message(self):
        """Test adding system message to history."""
        mock_store = MagicMock()
        history = SQLiteChatMessageHistory("test_session", memory_store=mock_store)

        message = SystemMessage(content="System initialized")
        history.add_message(message)

        mock_store.store.assert_called_once()
        call_args = mock_store.store.call_args
        assert call_args[1]["step"] == "message_system"

    def test_get_messages_empty(self):
        """Test getting messages when none exist."""
        mock_store = MagicMock()
        mock_store.load.return_value = []

        history = SQLiteChatMessageHistory("test_session", memory_store=mock_store)
        messages = history.get_messages()

        assert messages == []
        mock_store.load.assert_called_once_with(agent_id="test_session")

    def test_get_messages_with_data(self):
        """Test getting messages with stored data."""
        import json

        # Mock stored memories
        mock_memories = [
            Memory(
                id=1,
                agent_id="test_session",
                step="message_human",
                content=json.dumps(
                    {
                        "type": "HumanMessage",
                        "content": "Hello!",
                        "additional_kwargs": {},
                    }
                ),
                timestamp=datetime.now(),
            ),
            Memory(
                id=2,
                agent_id="test_session",
                step="message_ai",
                content=json.dumps(
                    {
                        "type": "AIMessage",
                        "content": "Hi there!",
                        "additional_kwargs": {},
                        "response_metadata": {"tokens": 100},
                    }
                ),
                timestamp=datetime.now(),
            ),
        ]

        mock_store = MagicMock()
        mock_store.load.return_value = mock_memories

        history = SQLiteChatMessageHistory("test_session", memory_store=mock_store)
        messages = history.get_messages()

        assert len(messages) == 2
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "Hello!"
        assert isinstance(messages[1], AIMessage)
        assert messages[1].content == "Hi there!"
        assert messages[1].response_metadata == {"tokens": 100}

    def test_get_messages_filters_non_messages(self):
        """Test that get_messages filters out non-message entries."""
        import json

        mock_memories = [
            Memory(
                id=1,
                agent_id="test_session",
                step="some_action",  # Not a message
                content="some content",
                timestamp=datetime.now(),
            ),
            Memory(
                id=2,
                agent_id="test_session",
                step="message_human",
                content=json.dumps(
                    {
                        "type": "HumanMessage",
                        "content": "Hello!",
                        "additional_kwargs": {},
                    }
                ),
                timestamp=datetime.now(),
            ),
        ]

        mock_store = MagicMock()
        mock_store.load.return_value = mock_memories

        history = SQLiteChatMessageHistory("test_session", memory_store=mock_store)
        messages = history.get_messages()

        # Should only return the actual message
        assert len(messages) == 1
        assert isinstance(messages[0], HumanMessage)

    def test_clear_session(self):
        """Test clearing session."""
        mock_store = MagicMock()
        history = SQLiteChatMessageHistory("test_session", memory_store=mock_store)

        history.clear()

        # Should store a clear marker
        mock_store.store.assert_called_once()
        call_args = mock_store.store.call_args
        assert call_args[1]["step"] == "session_cleared"


class TestSQLiteAgentMemory:
    """Test suite for SQLiteAgentMemory."""

    @patch("src.memory.langchain_adapter.SQLiteChatMessageHistory")
    def test_initialization(self, mock_chat_history):
        """Test initialization of SQLite agent memory."""
        mock_chat_instance = MagicMock()
        mock_chat_history.return_value = mock_chat_instance

        memory = SQLiteAgentMemory("test_session")

        assert memory.session_id == "test_session"
        assert memory.chat_memory == mock_chat_instance
        assert "history" in memory.memory_variables

    @patch("src.memory.langchain_adapter.SQLiteChatMessageHistory")
    def test_load_memory_variables_as_string(self, mock_chat_history):
        """Test loading memory variables as formatted string."""
        # Mock messages
        mock_messages = [HumanMessage(content="Hello!"), AIMessage(content="Hi there!")]

        mock_chat_instance = MagicMock()
        mock_chat_instance.get_messages.return_value = mock_messages
        mock_chat_history.return_value = mock_chat_instance

        memory = SQLiteAgentMemory("test_session", return_messages=False)
        variables = memory.load_memory_variables({})

        assert "history" in variables
        history_str = variables["history"]
        assert "Human: Hello!" in history_str
        assert "AI: Hi there!" in history_str

    @patch("src.memory.langchain_adapter.SQLiteChatMessageHistory")
    def test_load_memory_variables_as_messages(self, mock_chat_history):
        """Test loading memory variables as message objects."""
        mock_messages = [HumanMessage(content="Hello!"), AIMessage(content="Hi there!")]

        mock_chat_instance = MagicMock()
        mock_chat_instance.get_messages.return_value = mock_messages
        mock_chat_history.return_value = mock_chat_instance

        memory = SQLiteAgentMemory("test_session", return_messages=True)
        variables = memory.load_memory_variables({})

        assert "history" in variables
        history_messages = variables["history"]
        assert len(history_messages) == 2
        assert isinstance(history_messages[0], HumanMessage)
        assert isinstance(history_messages[1], AIMessage)

    @patch("src.memory.langchain_adapter.SQLiteChatMessageHistory")
    def test_save_context(self, mock_chat_history):
        """Test saving interaction context."""
        mock_chat_instance = MagicMock()
        mock_chat_history.return_value = mock_chat_instance

        memory = SQLiteAgentMemory("test_session")

        inputs = {"input": "What is surveillance?"}
        outputs = {"output": "Surveillance is monitoring..."}

        memory.save_context(inputs, outputs)

        # Should add both human and AI messages
        assert mock_chat_instance.add_message.call_count == 2

        # Check the calls
        calls = mock_chat_instance.add_message.call_args_list
        human_msg = calls[0][0][0]  # First call, first argument
        ai_msg = calls[1][0][0]  # Second call, first argument

        assert isinstance(human_msg, HumanMessage)
        assert human_msg.content == "What is surveillance?"
        assert isinstance(ai_msg, AIMessage)
        assert ai_msg.content == "Surveillance is monitoring..."

    @patch("src.memory.langchain_adapter.SQLiteChatMessageHistory")
    def test_save_context_alternative_keys(self, mock_chat_history):
        """Test saving context with alternative input/output keys."""
        mock_chat_instance = MagicMock()
        mock_chat_history.return_value = mock_chat_instance

        memory = SQLiteAgentMemory("test_session")

        inputs = {"question": "What is surveillance?"}
        outputs = {"answer": "Surveillance is monitoring..."}

        memory.save_context(inputs, outputs)

        # Should still work with alternative keys
        assert mock_chat_instance.add_message.call_count == 2

    @patch("src.memory.langchain_adapter.SQLiteChatMessageHistory")
    def test_clear_memory(self, mock_chat_history):
        """Test clearing memory."""
        mock_chat_instance = MagicMock()
        mock_chat_history.return_value = mock_chat_instance

        memory = SQLiteAgentMemory("test_session")
        memory.clear()

        mock_chat_instance.clear.assert_called_once()

    @patch("src.memory.langchain_adapter.SQLiteChatMessageHistory")
    def test_token_limit_string_truncation(self, mock_chat_history):
        """Test token limit truncation for string history."""
        long_messages = [
            HumanMessage(content="A" * 1000),  # Long message
            AIMessage(content="B" * 1000),  # Another long message
        ]

        mock_chat_instance = MagicMock()
        mock_chat_instance.get_messages.return_value = long_messages
        mock_chat_history.return_value = mock_chat_instance

        # Set low token limit
        settings = LangChainSettings(memory_max_tokens=100)  # Very small limit
        memory = SQLiteAgentMemory(
            "test_session", langchain_settings=settings, return_messages=False
        )

        variables = memory.load_memory_variables({})
        history_str = variables["history"]

        # Should be truncated (rough estimate: 100 tokens * 4 chars = 400 chars + "...")
        assert len(history_str) <= 403  # 400 + "..."
        assert history_str.startswith("...")

    @patch("src.memory.langchain_adapter.SQLiteChatMessageHistory")
    def test_token_limit_message_truncation(self, mock_chat_history):
        """Test token limit truncation for message list history."""
        # Create many messages
        long_messages = []
        for i in range(20):
            long_messages.append(HumanMessage(content=f"Message {i}" + "X" * 100))

        mock_chat_instance = MagicMock()
        mock_chat_instance.get_messages.return_value = long_messages
        mock_chat_history.return_value = mock_chat_instance

        # Set low token limit
        settings = LangChainSettings(memory_max_tokens=200)  # Small limit
        memory = SQLiteAgentMemory(
            "test_session", langchain_settings=settings, return_messages=True
        )

        variables = memory.load_memory_variables({})
        history_messages = variables["history"]

        # Should have fewer messages than the original 20
        assert len(history_messages) < 20
        assert len(history_messages) > 0

        # Should keep the most recent messages
        for i, msg in enumerate(history_messages):
            # Later messages should have higher numbers
            assert "Message" in msg.content


class TestFactoryFunctions:
    """Test suite for memory factory functions."""

    @patch("src.memory.langchain_adapter.SQLiteAgentMemory")
    def test_create_agent_memory(self, mock_memory_class):
        """Test create_agent_memory factory function."""
        mock_instance = MagicMock()
        mock_memory_class.return_value = mock_instance

        result = create_agent_memory("TestAgent")

        # Should create memory with generated session ID
        mock_memory_class.assert_called_once()
        call_args = mock_memory_class.call_args

        assert call_args[1]["session_id"].startswith("TestAgent_")
        assert result == mock_instance

    @patch("src.memory.langchain_adapter.SQLiteAgentMemory")
    def test_create_agent_memory_with_session_id(self, mock_memory_class):
        """Test create_agent_memory with explicit session ID."""
        mock_instance = MagicMock()
        mock_memory_class.return_value = mock_instance

        result = create_agent_memory("TestAgent", session_id="custom_session")

        # Should use provided session ID
        mock_memory_class.assert_called_once()
        call_args = mock_memory_class.call_args

        assert call_args[1]["session_id"] == "custom_session"
        assert result == mock_instance

    @patch("src.memory.langchain_adapter.SQLiteChatMessageHistory")
    def test_create_chat_history(self, mock_history_class):
        """Test create_chat_history factory function."""
        mock_instance = MagicMock()
        mock_history_class.return_value = mock_instance

        result = create_chat_history("test_session")

        mock_history_class.assert_called_once_with(
            session_id="test_session", memory_store=None, database_settings=None
        )
        assert result == mock_instance

    @patch("src.memory.langchain_adapter.SQLiteAgentMemory")
    def test_create_pipeline_memory(self, mock_memory_class):
        """Test create_pipeline_memory factory function."""
        mock_instance = MagicMock()
        mock_memory_class.return_value = mock_instance

        result = create_pipeline_memory()

        # Should create memory with generated session ID and return_messages=True
        mock_memory_class.assert_called_once()
        call_args = mock_memory_class.call_args

        assert call_args[1]["session_id"].startswith("surveillance_pipeline_")
        assert call_args[1]["return_messages"] is True
        assert result == mock_instance


class TestMessageSerialization:
    """Test message serialization and deserialization."""

    def test_serialize_human_message(self):
        """Test serialization of human message."""
        mock_store = MagicMock()
        history = SQLiteChatMessageHistory("test", memory_store=mock_store)

        message = HumanMessage(content="Hello world!")
        serialized = history._serialize_message(message)

        import json

        data = json.loads(serialized)

        assert data["type"] == "HumanMessage"
        assert data["content"] == "Hello world!"
        assert "additional_kwargs" in data

    def test_serialize_ai_message_with_metadata(self):
        """Test serialization of AI message with metadata."""
        mock_store = MagicMock()
        history = SQLiteChatMessageHistory("test", memory_store=mock_store)

        message = AIMessage(
            content="Hello!",
            additional_kwargs={"model": "gpt-4"},
            response_metadata={"tokens": 100, "finish_reason": "stop"},
        )
        serialized = history._serialize_message(message)

        import json

        data = json.loads(serialized)

        assert data["type"] == "AIMessage"
        assert data["content"] == "Hello!"
        assert data["additional_kwargs"]["model"] == "gpt-4"
        assert data["response_metadata"]["tokens"] == 100

    def test_parse_human_message_from_memory(self):
        """Test parsing human message from stored memory."""
        import json

        mock_store = MagicMock()
        history = SQLiteChatMessageHistory("test", memory_store=mock_store)

        memory = Memory(
            id=1,
            agent_id="test",
            step="message_human",
            content=json.dumps(
                {
                    "type": "HumanMessage",
                    "content": "Test message",
                    "additional_kwargs": {},
                }
            ),
            timestamp=datetime.now(),
        )

        message = history._parse_message_from_memory(memory)

        assert isinstance(message, HumanMessage)
        assert message.content == "Test message"

    def test_parse_invalid_message_returns_none(self):
        """Test parsing invalid message returns None."""
        mock_store = MagicMock()
        history = SQLiteChatMessageHistory("test", memory_store=mock_store)

        memory = Memory(
            id=1,
            agent_id="test",
            step="message_human",
            content="invalid json",  # Not valid JSON
            timestamp=datetime.now(),
        )

        message = history._parse_message_from_memory(memory)
        assert message is None

    def test_parse_session_cleared_returns_none(self):
        """Test parsing session cleared marker returns None."""
        mock_store = MagicMock()
        history = SQLiteChatMessageHistory("test", memory_store=mock_store)

        memory = Memory(
            id=1,
            agent_id="test",
            step="session_cleared",
            content="Session cleared",
            timestamp=datetime.now(),
        )

        message = history._parse_message_from_memory(memory)
        assert message is None
