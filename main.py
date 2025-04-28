from src.tools.llm_wrapper import LocalLLM


def main():
    # settings = OllamaSettings()
    # client = OllamaClient(settings)
    # client("to what question is the answer 42?")
    local_llm = LocalLLM()
    response_text = local_llm.generate_response(
        "who is roberto bolano. briefly please", max_tokens=256
    )
    return response_text


if __name__ == "__main__":
    main()
