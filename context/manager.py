from dataclasses import dataclass, field
from typing import Any

from config.config import Config
from prompts.system import get_system_prompt
from utils.text import count_tokens


@dataclass
class MessageItem:
    """
    Represents one message in our conversation history.

    The to_dict() method converts this object into the dictionary format
    expected by the LLM API.
    """

    # Who sent this message: "user", "assistant", or "tool".
    role: str

    # The actual text content of the message.
    # For tool messages, this contains the tool's output.
    content: str

    # If this message is a tool call, this field contains the unique tool call ID.
    tool_call_id: str | None = None

    # Stores the tool calls requested by the assistant.
    #
    # For example, the assistant might ask to execute:
    # {"name": "read_file", "arguments": {"path": "main.py"}}
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    # Locally estimated token count for THIS message, calculated
    # using tiktoken.
    #
    # This is NOT the actual token usage reported by the LLM.
    # We maintain it so the ContextManager can estimate the size
    # of individual messages and selectively prune old tool outputs
    # without having to recalculate their token counts every time.
    token_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert MessageItem into the dictionary format expected by the LLM API.

        The LLM API doesn't know about our MessageItem class.
        It expects ordinary dictionaries containing fields such as
        role, content, tool_calls, and tool_call_id.

        We only include optional fields when they have values.
        """

        # Every message must have a role.
        result: dict[str, Any] = {"role": self.role}

        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id

        if self.tool_calls:
            result["tool_calls"] = self.tool_calls

        if self.content:
            result["content"] = self.content

        return result


class ContextManager:
    def __init__(self, config: Config) -> None:
        self._system_prompt = get_system_prompt()
        self.config = config
        self._model_name = config.model_name

        # Actual conversation history.
        #
        # This contains user messages, assistant messages, and tool results.
        # The system prompt is stored separately in _system_prompt.
        self._messages: list[MessageItem] = []

    def add_user_message(self, content: str) -> None:
        """
        Add a new message sent by the user to conversation history.
        """

        item = MessageItem(
            role="user",
            content=content,
            # Estimate the token count now so that we don't have to
            # repeatedly calculate it later.
            token_count=count_tokens(
                content,
                self._model_name,
            ),
        )

        # Append the message to the conversation history.
        self._messages.append(item)

    def add_assistant_message(
        self,
        content: str,
        tool_calls: dict[str, Any] | None = None,
    ) -> None:
        """
        Add the assistant's response to conversation history.

        An assistant message can contain:
        1. Normal text written by the LLM.
        2. Tool calls requested by the LLM.
        3. Both text and tool calls.
        """

        item = MessageItem(
            role="assistant",
            # The LLM may return None/empty content when it only wants
            # to call a tool. We normalize that to an empty string.
            content=content or "",
            # Estimate tokens for the assistant's text content.
            token_count=count_tokens(
                content or "",
                self._model_name,
            ),
            # Store the tool calls requested by the assistant, if any.
            tool_calls=tool_calls or [],
        )

        self._messages.append(item)

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        """
        Add the output returned by a tool to conversation history.

        Example:
        The assistant requests read_file("main.py").
        Our application executes the tool.
        The tool returns the file contents.
        We add that output here so the LLM can see it in its next request.
        """

        item = MessageItem(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
            token_count=count_tokens(
                content,
                self._model_name,
            ),
        )

        self._messages.append(item)

    def get_messages(self) -> list[dict[str, Any]]:
        """
        Return the complete conversation in the format expected by the LLM.

        The system prompt must come first, followed by the conversation history.

        This is what we pass to the LLM whenever we make a new request.
        """

        messages = []

        # The system prompt contains the agent's core instructions.
        # Add it at the beginning of every LLM request.
        if self._system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": self._system_prompt,
                }
            )

        # Convert each MessageItem into a regular dictionary.
        #
        # This also ensures that our internal fields such as token_count
        # and pruned_at are NOT sent to the LLM.
        for item in self._messages:
            messages.append(item.to_dict())

        return messages
