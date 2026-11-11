from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field, StringConstraints

from tools.base import Tool, ToolInvocation, ToolKind, ToolResult
from utils.text import truncate_text


class WebFetchParams(BaseModel):
    """
    A pydantic model that defines the arguments accepted by WebFetchTool.
    """

    url: str = Field(
        description="The URL to fetch content from. (must be http:// or https://)",
    )

    timeout: int = Field(
        30,
        ge=5,
        le=120,
        description="Request timeout in seconds. (default: 120)",
    )


class WebFetchTool(Tool):
    """
    A tool that fetches the content of a specific webpage using its URL.
    """

    name = "web_fetch"

    description = (
        "Fetch the content of a specific webpage using its URL. "
        "Use this when you already have a URL and want to retrieve its "
        "response body as text. Unlike web_search, this tool does not "
        "discover webpages."
    )

    kind = ToolKind.NETWORK

    schema = WebFetchParams

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        # Validate the params
        params = WebFetchParams(**invocation.params)

        # Validate the structure of the URL
        parsed_url = urlparse(params.url)
        if parsed_url.scheme not in ("http", "https"):
            return ToolResult.error_result(f"Url must be http:// or https://")

        try:
            async with httpx.AsyncClient(
                timeout=params.timeout,
                follow_redirects=True,
            ) as client:
                response = await client.get(params.url)
                response.raise_for_status()
                text = truncate_text(
                    response.text.strip(),
                    self.MAX_OUTPUT_TOKENS,
                    model=self.config.model_name,
                )
        except httpx.HTTPStatusError as e:
            return ToolResult.error_result(
                f"HTTP {e.response.status_code}: {e.response.reason_phrase}",
            )
        except Exception as e:
            return ToolResult.error_result(f"Request failed: {e}")

        return ToolResult.success_result(
            text,
            metadata={
                "status_code": response.status_code,
                "content_length": len(response.content),
            },
        )
