from abc import ABC, abstractmethod
from typing import Any, Dict, List


class Agent(ABC):
    """
    Abstract agent class for all agents in the system.
    """

    def __init__(self, name: str, tools: Dict[str, Any], memory: Any = None) -> None:
        """
        Initialize the Agent.

        :param name: A unique identifier for the agent.
        :param tools: A mapping of tool names to callable tool instances.
        :param memory: An optional memory store for persisting state.
        """
        self.name = name
        self.tools: Dict[str, Any] = tools
        self.memory = memory

    @abstractmethod
    def perceive(self, input_data: Any) -> Any:
        """
        Observe or parse incoming data.

        :param input_data: Raw input (e.g., document, URL).
        :return: Processed observation.
        """
        pass

    @abstractmethod
    def plan(self, observation: Any) -> List[str]:
        """
        Decide on a list of actions to achieve the agent's goal.

        :param observation: Output from perceive().
        :return: A sequence of action names.
        """
        pass

    @abstractmethod
    def act(self, action: str, context: Dict[str, Any]) -> Any:
        """
        Execute a single action using the specified tool.

        :param action: The name of the action/tool to invoke.
        :param context: A dict carrying necessary data from previous steps.
        :return: Result of the action.
        """
        pass

    def think(self, intermediate: Any) -> Any:
        """
        Optionally reason or update internal state between actions.

        :param intermediate: The output of the last action.
        :return: Potentially modified data for the next step.
        """
        return intermediate

    def remember(self, step: str, result: Any) -> Any:
        """
        Persist the outcome of an action to memory.

        :param step: The action that was executed.
        :param result: The result of the action.
        """
        if self.memory:
            self.memory.store(self.name, step, result)

    def achieve_goal(self, input_data: Any) -> None:
        """
        Orchestrate the sense-plan-act-remember cycle.

        :param input_data: Initial data for perception.
        """
        observation = self.perceive(input_data)
        plan_steps = self.plan(observation)
        context: Dict[str, Any] = {"observation": observation}

        for step in plan_steps:
            result = self.act(step, context)
            self.remember(step, result)
            # context = {**context, step: result} TODO: think(self, context: Dict[…]) should return an updated context
            context = self.think(result)
