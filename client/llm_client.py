import asyncio
from typing import Any, AsyncGenerator
from openai import APIConnectionError, APIError, AsyncOpenAI, RateLimitError

from client.response import (
    StreamEventType,
    StreamEvent,
    TextDelta,
    TokenUsage,
    ToolCall,
    ToolCallDelta,
    parse_tool_call_arguments,
)
from config.config import Config


class LLMClient:
    """Handles communication with the configured LLM provider."""

    def __init__(self, config: Config) -> None:
        # We don't create the actual API client immediately.
        # It will be created the first time get_client() is called.
        self._client: AsyncOpenAI | None = None

        # Maximum number of retries after the initial request fails.
        self._max_retries: int = 3

        # Stores the application's configuration, including
        # API key, base URL, and model name.
        self.config = config

    def get_client(self) -> AsyncOpenAI:
        """Create the AsyncOpenAI client on first use and reuse it."""

        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )
        return self._client

    async def close(self) -> None:
        """Close the AsyncOpenAI client and release its HTTP resources."""

        if self._client is not None:
            await self._client.close()
            self._client = None

    def _build_tools(self, tools: list[dict[str, Any]]):
        """
        Converts our application's tool schemas into the format
        expected by the Chat Completions API.

        WHY DO WE NEED THIS?

        Tool.to_openai_schema() returns the function definition:

            {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {...}
            }

        But Chat Completions expects that definition to be wrapped:

            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {...}
                }
            }

        This method performs that conversion for every tool.
        """

        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get(
                        "parameters",
                        {
                            "type": "object",
                            "properties": {},
                        },
                    ),
                },
            }
            for tool in tools
        ]

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = True,
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Sends a conversation to the LLM and yields the response
        in our application's common StreamEvent format.

        messages:
            The conversation history sent to the LLM.

        tools:
            Tool definitions that the LLM is allowed to use.

        stream:
            True  -> Receive the response incrementally.
            False -> Receive the complete response at once.

        WHY IS THIS AN ASYNC GENERATOR?

        This method uses yield, so callers consume its output using:

            async for event in llm_client.chat_completion(...):
                ...

        This allows the Agent to handle events as they arrive.
        """

        # Get the shared API client, creating it if necessary.
        client = self.get_client()

        # These are the arguments that will be passed to:
        #
        # client.chat.completions.create(**kwargs)
        kwargs = {
            "model": self.config.model_name,
            "messages": messages,
            "stream": stream,
        }

        if tools is not None:
            # Convert our internal tool schemas into the structure
            # expected by the Chat Completions API.
            kwargs["tools"] = self._build_tools(tools)
            kwargs["tool_choice"] = "auto"

        for attempt in range(self._max_retries + 1):
            try:
                if stream:
                    # Receive and forward events as the response arrives.
                    #
                    # We use async for because _stream_response()
                    # is itself an async generator.
                    async for event in self._stream_response(client, kwargs):
                        yield event

                else:
                    # Wait for the complete response.
                    event = await self._non_stream_response(client, kwargs)
                    yield event

                return

            except RateLimitError as e:
                # The provider is temporarily refusing requests because
                # its rate limit has been reached.
                #
                # Instead of failing immediately, retry with exponential backoff.
                if attempt < self._max_retries:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                else:

                    # All attempts have failed. Report the error
                    # through the same StreamEvent interface.
                    yield StreamEvent(
                        type=StreamEventType.ERROR, error=f"Rate limit exceeded {e}"
                    )

                    return

            except APIConnectionError as e:
                if attempt < self._max_retries:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                else:
                    yield StreamEvent(
                        type=StreamEventType.ERROR, error=f"Connection error {e}"
                    )

                    return

            except APIError as e:
                yield StreamEvent(type=StreamEventType.ERROR, error=f"API error {e}")

                return

            except Exception as e:
                yield StreamEvent(
                    type=StreamEventType.ERROR, error=f"Unexpected error {e}"
                )

                return

    async def _stream_response(
        self, client: AsyncOpenAI, kwargs: dict[str, Any]
    ) -> AsyncGenerator[StreamEvent, None]:
        """Convert streaming API chunks into our application's StreamEvents."""

        # stream=True was already included in kwargs.
        #
        # The API returns a stream that we can consume using async for.
        response = await client.chat.completions.create(**kwargs)

        finish_reason: str | None = None
        usage: TokenUsage | None = None
        tool_calls: dict[int, dict[str, Any]] = {}

        # Read the response one chunk at a time.
        async for chunk in response:
            if hasattr(chunk, "usage") and chunk.usage:
                usage = TokenUsage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                    cached_tokens=chunk.usage.prompt_tokens_details.cached_tokens,
                )

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            if choice.finish_reason:
                finish_reason = choice.finish_reason

            # Forward each piece of normal assistant text immediately.
            if delta.content:
                yield StreamEvent(
                    type=StreamEventType.TEXT_DELTA,
                    text_delta=TextDelta(delta.content),
                )

            if delta.tool_calls:
                for tool_call_delta in delta.tool_calls:
                    # Each tool call has an index.
                    # We use it to collect the correct pieces together.
                    idx = tool_call_delta.index

                    # The first time we see this index, create an empty
                    # entry where we can accumulate its information.
                    if idx not in tool_calls:
                        tool_calls[idx] = {
                            "tool_call_id": "",
                            "tool_name": "",
                            "arguments": "",
                        }

                    # Save the tool call's ID
                    if tool_call_delta.id:
                        tool_calls[idx]["tool_call_id"] = tool_call_delta.id

                    if tool_call_delta.function:
                        if tool_call_delta.function.name:
                            tool_calls[idx]["tool_name"] = tool_call_delta.function.name

                            yield StreamEvent(
                                type=StreamEventType.TOOL_CALL_START,
                                tool_call_delta=ToolCallDelta(
                                    tool_call_id=tool_calls[idx]["tool_call_id"],
                                    tool_name=tool_calls[idx]["tool_name"],
                                ),
                            )

                        # Accumulate every incoming argument fragment.
                        if tool_call_delta.function.arguments:
                            tool_calls[idx][
                                "arguments"
                            ] += tool_call_delta.function.arguments

                            yield StreamEvent(
                                type=StreamEventType.TOOL_CALL_DELTA,
                                tool_call_delta=ToolCallDelta(
                                    tool_call_id=tool_calls[idx]["tool_call_id"],
                                    tool_name=tool_calls[idx]["tool_name"],
                                    arguments_delta=tool_call_delta.function.arguments,
                                ),
                            )

        # All chunks have arrived. Emit each fully reconstructed tool call.
        for tc in tool_calls.values():
            yield StreamEvent(
                type=StreamEventType.TOOL_CALL_COMPLETE,
                tool_call=ToolCall(
                    tool_call_id=tc["tool_call_id"],
                    tool_name=tc["tool_name"],
                    arguments=parse_tool_call_arguments(tc["arguments"]),
                ),
            )

        # Notify the Agent that the entire LLM response has finished.
        yield StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            finish_reason=finish_reason,
            usage=usage,
        )

    async def _non_stream_response(
        self, client: AsyncOpenAI, kwargs: dict[str, Any]
    ) -> StreamEvent:
        """Convert a complete, non-streaming response into a StreamEvent."""

        response = await client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        message = choice.message

        text_delta = None
        # Extract the text content from the message if it exists
        if message.content:
            text_delta = TextDelta(content=message.content)

        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        tool_call_id=tc.id,
                        tool_name=tc.function.name,
                        arguments=parse_tool_call_arguments(tc.function.arguments),
                    )
                )

        # Extract the token usage from the response if it exists
        usage = None
        if response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                cached_tokens=response.usage.prompt_tokens_details.cached_tokens,
            )

        return StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            text_delta=text_delta,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            usage=usage,
        )
