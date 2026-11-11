from typing import Annotated, Literal

from ddgs import DDGS
from pydantic import BaseModel, Field, StringConstraints

from tools.base import Tool, ToolInvocation, ToolKind, ToolResult


class WebSearchParams(BaseModel):
    """
    A pydantic model that defines the arguments accepted by WebSearchTool.
    """

    query: Annotated[
        str,
        StringConstraints(
            min_length=1,
            strip_whitespace=True,
        ),
    ] = Field(
        description="Search query",
    )

    max_results: int = Field(
        10, ge=1, le=20, description="Maximum results to return (default: 10)"
    )

    timelimit: Literal["d", "w", "m", "y"] | None = Field(
        default=None,
        description=(
            "Optionally restrict results by recency: "
            "'d' for past day, 'w' for past week, "
            "'m' for past month, and 'y' for past year. "
            "Use only when the query requires recent information. "
            "Otherwise, leave this as None."
        ),
    )


class WebSearchTool(Tool):
    """
    Searches the web for information based on a search query.

    This tool uses the DDGS library to perform web searches.

    It returns search results containing:
    - Page titles.
    - Page URLs.
    - Short descriptions/snippets.

    The agent can use these results to identify relevant web pages.
    If it needs to understand the actual contents of a particular
    page, it can subsequently call the web_fetch tool.
    """

    name = "web_search"

    description = (
        "Search the web for information. "
        "Returns search results with titles, URLs and snippets"
    )

    kind = ToolKind.NETWORK

    schema = WebSearchParams

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        params = WebSearchParams(**invocation.params)

        try:
            results = DDGS().text(
                params.query,
                region="us-en",
                safesearch="off",
                timelimit=params.timelimit,
                max_results=params.max_results,
                page=1,
                backend="auto",
            )

        except Exception as e:
            return ToolResult.error_result(f"Search failed: {e}")

        if not results:
            return ToolResult.success_result(
                f"No search results found for: {params.query}",
                metadata={
                    "results": 0,
                },
            )

        output_lines = [f"Search results for: {params.query}"]

        for i, result in enumerate(results, start=1):
            title = result.get("title") or "Untitled"
            url = result.get("href") or "URL unavailable"
            snippet = (result.get("body") or "").strip()

            output_lines.append(f"{i}. Title: {title}")
            output_lines.append(f"   URL: {url}")
            if result.get("body"):
                output_lines.append(f"   Snippet: {snippet}")

            output_lines.append("")

        return ToolResult.success_result(
            "\n\n".join(output_lines),
            metadata={
                "results": len(results),
            },
        )
