import pytest
from src.agents.base_agent import Agent


def test_base_agent_abstract():
    with pytest.raises(TypeError):
        Agent(name="X", tools={}, memory=None)


class SimpleAgent(Agent):
    def perceive(self, input_data):
        return {"in": input_data}

    def plan(self, observation):
        return ["step"]

    def act(self, action, context):
        return "ok"


def test_simple_agent_runs():
    agent = SimpleAgent(name="Simple", tools={}, memory=None)
    # Should run without errors
    agent.achieve_goal("DATA")
