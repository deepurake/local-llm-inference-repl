def truncate_context(tokens: list[int], max_tokens: int) -> list[int]:
    if len(tokens) <= max_tokens:
        return tokens
    return tokens[-max_tokens:]
