from typing import Dict, List, Any
from src.config.logger import logger

from src.agents.base_agent import Agent


class DummyAgent(Agent):
    """
    A simple agent for testing purposes.
    It returns a fixed plan and logs each action it "executes."
    """

    def __init__(
        self, name: str = "DummyAgent", tools: Dict[str, Any] = None, memory: Any = None
    ) -> None:
        """
        Initialize the DummyAgent.

        :param name: Unique identifier for the agent (defaults to "DummyAgent").
        :param tools: Optional tools mapping (ignored here).
        :param memory: Optional memory store (ignored here).
        """
        super().__init__(name=name, tools=tools or {}, memory=memory)

    def perceive(self, input_data: Any) -> Any:
        """
        Simply returns the raw input data.
        """
        logger.info(f"{self.name} perceived input: {input_data}")
        return input_data

    def plan(self, observation: Any) -> List[str]:
        """
        Returns a dummy list of steps regardless of observation.
        """
        logger.info(f"{self.name} planning based on observation: {observation}")
        return ["dummy_step1", "dummy_step2", "dummy_step3"]

    def act(self, action: str, context: Dict[str, Any]) -> Any:
        """
        Logs and prints the action name, returns a dummy result string.

        :param action: The name of the action to "execute".
        :param context: The current context dict (observation + past results).
        :return: A string indicating a dummy result.
        """
        logger.info(f"{self.name} executing action: {action} with context: {context}")
        result = f"result_of_{action}"
        logger.info(f"{self.name} executed action: {action} -> {result}")
        print(f"{self.name} executed action: {action} -> {result}")
        return result
