from src.agents.dummy_agent import DummyAgent


def test_dummy_agent_cycle(capsys):
    agent = DummyAgent(name="TestDummy")
    agent.achieve_goal("INPUT")
    captured = capsys.readouterr().out.strip().splitlines()
    # Expect three lines for dummy_step1..3
    assert len(captured) == 3
    for i, line in enumerate(captured, start=1):
        assert f"executed action: dummy_step{i}" in line
