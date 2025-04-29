from src.config.settings import OllamaSettings
from src.tools.llm_wrapper import LocalLLM
from tests.conftest import StubClient


def make_wrapper(resp):
    settings = OllamaSettings(base_url="http://x", timeout_seconds=1, model="m")
    stub = StubClient(settings)
    stub._response = resp
    # ensure LocalLLM uses our stub instance
    wrapper = LocalLLM(settings)
    wrapper.client = stub
    return wrapper


def test_generate_from_response_field():
    wrapper = make_wrapper({"response": "hi there"})
    assert wrapper.generate_response("x") == "hi there"


def test_generate_from_choices_field():
    wrapper = make_wrapper({"choices": [{"text": "foo "}]})
    assert wrapper.generate_response("x") == "foo"


def test_generate_empty_and_warn(caplog):
    wrapper = make_wrapper({})
    caplog.set_level("WARNING")
    out = wrapper.generate_response("x")
    assert out == ""
    assert "" in caplog.text.lower()
