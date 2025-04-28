from src.config.settings import OllamaSettings
from src.tools.ollama_client import OllamaClient


def main():
    settings = OllamaSettings()
    client = OllamaClient(settings)
    client("to what question is the answer 42?")


if __name__ == "__main__":
    main()
