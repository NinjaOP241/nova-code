import tiktoken


def get_tokenizer(model: str):
    try:
        # Get the tokenizer encoding associated with the specified model.
        encoding = tiktoken.encoding_for_model(model)
        return encoding.encode
    except Exception:
        # If the model is not recognized, fall back to cl100k_base encoding.
        encoding = tiktoken.get_encoding("cl100k_base")
        return encoding.encode


def count_tokens(text: str, model: str = "gpt-4") -> int:
    tokenizer = get_tokenizer(model)
    return len(tokenizer(text))


def truncate_text(
    text: str,
    max_tokens: int,
    suffix: str = "\n... [truncated]",
    model: str = "gpt-4",
    preserve_lines: bool = True,
):
    # If the text already fits within the token limit, no truncation is needed.
    current_tokens = count_tokens(text, model)
    if current_tokens <= max_tokens:
        return text

    # Reserve some tokens for the truncation message.
    suffix_tokens = count_tokens(suffix, model)
    target_tokens = max_tokens - suffix_tokens

    # If there is no room for original content, return only the suffix.
    if target_tokens <= 0:
        return suffix.strip()

    if preserve_lines:
        return _truncate_by_lines(text, target_tokens, suffix, model)

    return _truncate_by_chars(text, target_tokens, suffix, model)


def _truncate_by_lines(text: str, target_tokens: int, suffix: str, model: str) -> str:
    lines = text.split("\n")
    result_lines: list[str] = []
    current_tokens = 0

    for line in lines:
        line_tokens = count_tokens(line + "\n", model)
        if current_tokens + line_tokens > target_tokens:
            break
        result_lines.append(line)
        current_tokens += line_tokens

    if not result_lines:
        # Fall back to character-based truncation if no lines fit within the token limit.
        return _truncate_by_chars(text, target_tokens, suffix, model)

    return "\n".join(result_lines) + suffix


def _truncate_by_chars(text: str, target_tokens: int, suffix: str, model: str) -> str:
    # Binary search for the last character index that fits within the token limit.
    end_index = -1
    low, high = 0, len(text) - 1

    while low <= high:
        mid = (low + high) // 2

        if count_tokens(text[: mid + 1], model) <= target_tokens:
            end_index = mid
            low = mid + 1
        else:
            high = mid - 1

    return text[: end_index + 1] + suffix
