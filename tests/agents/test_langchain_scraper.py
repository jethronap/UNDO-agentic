import json
from unittest.mock import Mock, patch

import pytest

from src.agents.langchain_scraper import SurveillanceScraperAgent
from src.config.settings import LangChainSettings


# Sample payloads
PAYLOAD_OK = {
    "elements": [
        {"id": 1, "lat": 55.6, "lon": 13.0},
        {"id": 2, "lat": 55.7, "lon": 13.1},
    ]
}
PAYLOAD_EMPTY = {"elements": []}


@pytest.fixture
def mock_langchain_settings():
    """Create mock LangChain settings for testing."""
    return LangChainSettings(
        ollama_base_url="http://localhost:11434",
        ollama_model="test-model",
        agent_verbose=False,
        agent_max_iterations=5,
        agent_max_execution_time=30.0,
    )


@pytest.fixture
def mock_llm():
    """Create a mock LLM that returns predictable responses."""
    mock = Mock()
    # Mock the invoke method to return agent-like responses
    mock.invoke.return_value = "Task completed successfully"
    return mock


class TestSurveillanceScraperAgent:
    """Test suite for LangChain-based SurveillanceScraperAgent."""

    @patch("src.agents.langchain_scraper.create_surveillance_llm")
    @patch("src.agents.langchain_scraper.create_scraper_tools")
    def test_initialization(
        self,
        mock_create_tools,
        mock_create_llm,
        mem_fake,
        mock_langchain_settings,
    ):
        """Test that agent initializes correctly with all components."""
        # Setup mocks
        mock_llm_instance = Mock()
        mock_llm_instance.llm = Mock()
        mock_create_llm.return_value = mock_llm_instance
        mock_create_tools.return_value = []

        # Initialize agent
        agent = SurveillanceScraperAgent(
            name="TestScraperAgent",
            memory=mem_fake,
            settings=mock_langchain_settings,
        )

        # Verify initialization
        assert agent.name == "TestScraperAgent"
        assert agent.memory == mem_fake
        assert agent.settings == mock_langchain_settings
        mock_create_llm.assert_called_once_with(mock_langchain_settings)
        mock_create_tools.assert_called_once_with(mem_fake)

    @patch("src.agents.langchain_scraper.create_surveillance_llm")
    @patch("src.agents.langchain_scraper.create_scraper_tools")
    @patch("src.agents.langchain_scraper.create_react_agent")
    @patch("src.agents.langchain_scraper.AgentExecutor")
    def test_scrape_with_cache_hit(
        self,
        mock_executor_class,
        mock_create_react_agent,
        mock_create_tools,
        mock_create_llm,
        mem_fake,
        tmp_path,
    ):
        """Test scraping when data exists in cache."""
        # Setup
        cached_file = tmp_path / "lund.json"
        cached_file.write_text(json.dumps(PAYLOAD_OK))

        # Mock LLM
        mock_llm_instance = Mock()
        mock_llm_instance.llm = Mock()
        mock_create_llm.return_value = mock_llm_instance

        # Mock tools
        mock_tools = []
        mock_create_tools.return_value = mock_tools

        # Mock agent and executor
        mock_agent = Mock()
        mock_create_react_agent.return_value = mock_agent

        mock_executor = Mock()
        # Simulate cache hit response
        mock_executor.invoke.return_value = {
            "output": "Cache hit! Found 2 cameras in cached data.",
            "intermediate_steps": [
                (
                    Mock(tool="check_query_cache"),
                    {
                        "cache_hit": True,
                        "data": PAYLOAD_OK,
                        "filepath": str(cached_file),
                        "elements_count": 2,
                    },
                ),
            ],
        }
        mock_executor_class.return_value = mock_executor

        # Create agent and scrape
        agent = SurveillanceScraperAgent("TestAgent", mem_fake)
        result = agent.scrape({"city": "Lund", "overpass_dir": str(tmp_path)})

        # Verify cache hit
        assert result["success"] is True
        assert result["cache_hit"] is True
        assert result["cached_path"] == str(cached_file)
        assert result["elements_count"] == 2
        assert result["data"] == PAYLOAD_OK
        assert "Cache hit" in result["agent_output"]

    @patch("src.agents.langchain_scraper.create_surveillance_llm")
    @patch("src.agents.langchain_scraper.create_scraper_tools")
    @patch("src.agents.langchain_scraper.create_react_agent")
    @patch("src.agents.langchain_scraper.AgentExecutor")
    def test_scrape_with_fresh_data(
        self,
        mock_executor_class,
        mock_create_react_agent,
        mock_create_tools,
        mock_create_llm,
        mem_fake,
        tmp_path,
    ):
        """Test scraping when no cache exists and fresh data is fetched."""
        # Mock LLM
        mock_llm_instance = Mock()
        mock_llm_instance.llm = Mock()
        mock_create_llm.return_value = mock_llm_instance

        # Mock tools
        mock_tools = []
        mock_create_tools.return_value = mock_tools

        # Mock agent and executor
        mock_agent = Mock()
        mock_create_react_agent.return_value = mock_agent

        mock_executor = Mock()
        saved_path = tmp_path / "malmo" / "malmo.json"
        # Simulate cache miss and successful save
        mock_executor.invoke.return_value = {
            "output": "Successfully collected 2 surveillance cameras for Malmo",
            "intermediate_steps": [
                (Mock(tool="check_query_cache"), {"cache_hit": False, "data": None}),
                (Mock(tool="run_overpass_query"), PAYLOAD_OK),
                (
                    Mock(tool="save_overpass_data"),
                    {
                        "saved": True,
                        "empty": False,
                        "filepath": str(saved_path),
                        "elements_count": 2,
                    },
                ),
            ],
        }
        mock_executor_class.return_value = mock_executor

        # Create agent and scrape
        agent = SurveillanceScraperAgent("TestAgent", mem_fake)
        result = agent.scrape({"city": "Malmo", "overpass_dir": str(tmp_path)})

        # Verify fresh data fetch
        assert result["success"] is True
        assert result["cache_hit"] is False
        assert result["filepath"] == str(saved_path)
        assert result["elements_count"] == 2
        assert result["empty"] is False

    @patch("src.agents.langchain_scraper.create_surveillance_llm")
    @patch("src.agents.langchain_scraper.create_scraper_tools")
    @patch("src.agents.langchain_scraper.create_react_agent")
    @patch("src.agents.langchain_scraper.AgentExecutor")
    def test_scrape_with_empty_result(
        self,
        mock_executor_class,
        mock_create_react_agent,
        mock_create_tools,
        mock_create_llm,
        mem_fake,
        tmp_path,
    ):
        """Test scraping when city has no surveillance cameras."""
        # Mock LLM
        mock_llm_instance = Mock()
        mock_llm_instance.llm = Mock()
        mock_create_llm.return_value = mock_llm_instance

        # Mock tools
        mock_tools = []
        mock_create_tools.return_value = mock_tools

        # Mock agent and executor
        mock_agent = Mock()
        mock_create_react_agent.return_value = mock_agent

        mock_executor = Mock()
        # Simulate empty result
        mock_executor.invoke.return_value = {
            "output": "No surveillance data found for Nowhere",
            "intermediate_steps": [
                (Mock(tool="check_query_cache"), {"cache_hit": False, "data": None}),
                (Mock(tool="run_overpass_query"), PAYLOAD_EMPTY),
                (
                    Mock(tool="save_overpass_data"),
                    {
                        "saved": False,
                        "empty": True,
                        "message": "No surveillance data found for Nowhere",
                    },
                ),
            ],
        }
        mock_executor_class.return_value = mock_executor

        # Create agent and scrape
        agent = SurveillanceScraperAgent("TestAgent", mem_fake)
        result = agent.scrape({"city": "Nowhere", "overpass_dir": str(tmp_path)})

        # Verify empty result handling
        assert result["success"] is True
        assert result["cache_hit"] is False
        assert result["empty"] is True
        assert "No surveillance data" in result["agent_output"]

    @patch("src.agents.langchain_scraper.create_surveillance_llm")
    @patch("src.agents.langchain_scraper.create_scraper_tools")
    @patch("src.agents.langchain_scraper.create_react_agent")
    @patch("src.agents.langchain_scraper.AgentExecutor")
    def test_scrape_with_country_code(
        self,
        mock_executor_class,
        mock_create_react_agent,
        mock_create_tools,
        mock_create_llm,
        mem_fake,
        tmp_path,
    ):
        """Test scraping with country code for disambiguation."""
        # Mock setup
        mock_llm_instance = Mock()
        mock_llm_instance.llm = Mock()
        mock_create_llm.return_value = mock_llm_instance
        mock_create_tools.return_value = []
        mock_create_react_agent.return_value = Mock()

        mock_executor = Mock()
        mock_executor.invoke.return_value = {
            "output": "Successfully collected data for Cambridge, UK",
            "intermediate_steps": [],
        }
        mock_executor_class.return_value = mock_executor

        # Create agent and scrape with country
        agent = SurveillanceScraperAgent("TestAgent", mem_fake)
        result = agent.scrape(
            {"city": "Cambridge", "country": "UK", "overpass_dir": str(tmp_path)}
        )

        # Verify country was included
        assert result["city"] == "Cambridge"
        assert result["country"] == "UK"
        assert result["success"] is True

        # Check that agent input included country
        call_args = mock_executor.invoke.call_args[0][0]
        assert "Cambridge" in call_args["input"]
        assert "UK" in call_args["input"]

    @patch("src.agents.langchain_scraper.create_surveillance_llm")
    @patch("src.agents.langchain_scraper.create_scraper_tools")
    @patch("src.agents.langchain_scraper.create_react_agent")
    @patch("src.agents.langchain_scraper.AgentExecutor")
    def test_scrape_error_handling(
        self,
        mock_executor_class,
        mock_create_react_agent,
        mock_create_tools,
        mock_create_llm,
        mem_fake,
        tmp_path,
    ):
        """Test error handling when scraping fails."""
        # Mock setup
        mock_llm_instance = Mock()
        mock_llm_instance.llm = Mock()
        mock_create_llm.return_value = mock_llm_instance
        mock_create_tools.return_value = []
        mock_create_react_agent.return_value = Mock()

        # Simulate executor error
        mock_executor = Mock()
        mock_executor.invoke.side_effect = Exception("Network timeout")
        mock_executor_class.return_value = mock_executor

        # Create agent and scrape
        agent = SurveillanceScraperAgent("TestAgent", mem_fake)
        result = agent.scrape({"city": "ErrorCity", "overpass_dir": str(tmp_path)})

        # Verify error was caught and returned
        assert result["success"] is False
        assert "error" in result
        assert "Network timeout" in result["error"]
        assert "Network timeout" in result["agent_output"]

    @patch("src.agents.langchain_scraper.create_surveillance_llm")
    @patch("src.agents.langchain_scraper.create_scraper_tools")
    def test_achieve_goal_backward_compatibility(
        self,
        mock_create_tools,
        mock_create_llm,
        mem_fake,
        tmp_path,
    ):
        """Test that achieve_goal() method works for backward compatibility."""
        # Mock LLM
        mock_llm_instance = Mock()
        mock_llm_instance.llm = Mock()
        mock_create_llm.return_value = mock_llm_instance
        mock_create_tools.return_value = []

        with patch.object(SurveillanceScraperAgent, "scrape") as mock_scrape:
            mock_scrape.return_value = {"success": True, "city": "TestCity"}

            agent = SurveillanceScraperAgent("TestAgent", mem_fake)
            input_data = {"city": "TestCity", "overpass_dir": str(tmp_path)}
            result = agent.achieve_goal(input_data)

            # Verify achieve_goal calls scrape
            mock_scrape.assert_called_once_with(input_data)
            assert result["success"] is True
            assert result["city"] == "TestCity"

    @patch("src.agents.langchain_scraper.create_surveillance_llm")
    @patch("src.agents.langchain_scraper.create_scraper_tools")
    def test_prompt_template_loading(
        self,
        mock_create_tools,
        mock_create_llm,
        mem_fake,
    ):
        """Test that prompt template is loaded correctly."""
        # Mock LLM
        mock_llm_instance = Mock()
        mock_llm_instance.llm = Mock()
        mock_create_llm.return_value = mock_llm_instance
        mock_create_tools.return_value = []

        # Create agent
        agent = SurveillanceScraperAgent("TestAgent", mem_fake)

        # Verify prompt template exists and has correct variables
        assert agent.prompt is not None
        assert "input" in agent.prompt.input_variables
        assert "tools" in agent.prompt.input_variables
        assert "tool_names" in agent.prompt.input_variables
        assert "agent_scratchpad" in agent.prompt.input_variables

    @patch("src.agents.langchain_scraper.create_surveillance_llm")
    @patch("src.agents.langchain_scraper.create_scraper_tools")
    @patch("src.agents.langchain_scraper.create_react_agent")
    @patch("src.agents.langchain_scraper.AgentExecutor")
    def test_directory_creation(
        self,
        mock_executor_class,
        mock_create_react_agent,
        mock_create_tools,
        mock_create_llm,
        mem_fake,
        tmp_path,
    ):
        """Test that output directory is created automatically."""
        # Mock setup
        mock_llm_instance = Mock()
        mock_llm_instance.llm = Mock()
        mock_create_llm.return_value = mock_llm_instance
        mock_create_tools.return_value = []
        mock_create_react_agent.return_value = Mock()

        mock_executor = Mock()
        mock_executor.invoke.return_value = {
            "output": "Success",
            "intermediate_steps": [],
        }
        mock_executor_class.return_value = mock_executor

        agent = SurveillanceScraperAgent("TestAgent", mem_fake)

        # Scrape with non-existent directory
        output_dir = tmp_path / "new_dir"
        assert not output_dir.exists()

        result = agent.scrape({"city": "Stockholm", "overpass_dir": str(output_dir)})
        print(result)
        # Verify directory was created
        expected_dir = output_dir / "stockholm"
        assert expected_dir.exists()
        assert expected_dir.is_dir()
