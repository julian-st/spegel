from __future__ import annotations

import os
import logging
from typing import AsyncIterator, Dict, Any

"""Light abstraction layer over one or more LLM back-ends.

Right now we only implement Google Gemini via `google-genai`, but the
interface allows us to add more providers later without touching UI code.
"""

# Configure logger for LLM interactions (disabled by default)
logger = logging.getLogger("spegel.llm")
logger.setLevel(logging.CRITICAL + 1)  # Effectively disabled by default

def enable_llm_logging(level: int = logging.INFO) -> None:
    """Enable LLM interaction logging at the specified level."""
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover – dependency is optional until used
    genai = None  # type: ignore

from mistralai import Mistral  # Import Mistral client library
from mistralai.extra.run.context import RunContext
from mcp import StdioServerParameters
from mistralai.extra.mcp.stdio import MCPClientSTDIO
from mistralai.types import BaseModel
import os

cwd = os.getcwd()  # Get the current working directory

__all__ = [
    "LLMClient",
    "GeminiClient",
    "MistralClient",
    "get_default_client",
]


class LLMClient:
    """Abstract asynchronous client interface."""

    async def stream(self, prompt: str, content: str, **kwargs) -> AsyncIterator[str]:
        """Yield chunks of markdown text."""
        raise NotImplementedError
        yield # This is unreachable, but makes this an async generator


class GeminiClient(LLMClient):
    """Wrapper around google-genai async streaming API."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-lite-preview-06-17"):
        if genai is None:
            raise RuntimeError("google-genai not installed but GeminiClient requested")
        self._client = genai.Client(api_key=api_key)
        self.model_name = model_name

    async def stream(
        self,
        prompt: str,
        content: str,
        generation_config: Dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        if generation_config is None:
            generation_config = types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens =8192,
                response_mime_type="text/plain",
                thinking_config = types.ThinkingConfig(
            thinking_budget=0,
        )
            )
        user_content = f"{prompt}\n\n{content}" if content else prompt
        stream = self._client.aio.models.generate_content_stream(
            model=self.model_name,
            contents=user_content,
            config=generation_config,
        )

        # Log the prompt if logging is enabled
        logger.info("LLM Prompt: %s", user_content)

        collected: list[str] = []

        async for chunk in await stream:
            try:
                text = chunk.candidates[0].content.parts[0].text  # type: ignore[attr-defined]
                if text:
                    collected.append(text)
                    yield text
            except Exception:
                continue

        # Log the complete response if logging is enabled
        if collected:
            logger.info("LLM Response: %s", "".join(collected))


class MistralClient(LLMClient):
    """Wrapper around Mistral AI async streaming API."""
    

    def __init__(self, api_key: str):
        from mistralai import Mistral  # Ensure correct import
        self._client = Mistral(api_key=api_key)
        self._mistral_id = 0

    async def stream(
        self,
        prompt: str,
        content: str,
        generation_config: Dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        if generation_config is None:
            generation_config = {
                "temperature": 0.2,
                "max_tokens": 8192,
            }
        websearch_agent = self._client.beta.agents.create(
            model="mistral-medium-2505",
            description="Agent able to search information over the web, such as news, weather, sport results...",
            name="Websearch Agent",
            instructions="You have the ability to perform web searches with `web_search` to find up-to-date information.",
            tools=[{"type": "web_search"}],
            completion_args={
                "temperature": 0.3,
                "top_p": 0.95,
            }
        )
        model = "mistral-large-latest"
        user_content = f"{prompt}\n\n{content}" if content else prompt

        # Create a run context for the agent inside the async function
        async with RunContext(
            agent_id=websearch_agent.id,  # Replace with actual agent ID
            output_format=SpegelResult,
            continue_on_fn_error=True,
        ) as run_ctx:

            #mcp_client = MCPClientSTDIO(stdio_params=server_params)
            #await run_ctx.register_mcp_client(mcp_client=mcp_client)
            if self._mistral_id == 0:
                # Use the correct method for generating text
                response = await self._client.beta.conversations.run_stream_async(
                    run_ctx=run_ctx,
                    inputs=user_content
                )

            #self._mistral_id = response.conversation_id  # Store the conversation ID for future reference
            # Process the streamed events
            run_result = None
            collected: list[str] = []
            print("All run entries as dict:")
            async for event in response:
                try:
                    print("Event received:", event)
                    if isinstance(event, dict):
                        run_result = event
                    else:
                        # Ensure `event.data` is converted to a string or extract the correct field
                        if hasattr(event, "data"):
                            # todo print("Event data:", event.data)
                            if hasattr(event.data, "content"):
                                # If event.data has a content attribute, use it
                                print("Event data content:", event.data.content)
                                collected.append(str(event.data.content))
                                yield str(event.data.content)
                            else:
                                collected.append(str(event.data))  # Convert to string explicitly
                                yield str(event.data)  # Yield the data directly
                            # Ensure the yielded value is a string
                        else:
                            print("Event does not have a 'data' attribute:", event)
                except Exception:
                    continue

            if not run_result:
                print("No run result found in the response.")
                raise RuntimeError("No run result found")

            # Print the results
           
            print("All run entries:")
            #for entry in run_result.output_entries:
            #    print(f"{entry}")
            #    if entry:
            #        collected.append(entry.data.content)
            #        yield entry.data.content
            print(f"Final model: {run_result.output_as_model}")

            # Iterate over the response chunks
            #for entry in response.output_entries:
            #    text = entry.get("text", "")
            #    if text:
            #        collected.append(text)
            #        yield text

            # Log the complete response if logging is enabled
            if collected:
                logger.info("LLM Response: %s", "".join(collected))

class SpegelResult(BaseModel):
        content: str

# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def get_default_client() -> tuple[LLMClient | None, bool]:
    """Return an LLMClient instance if credentials exist, else (None, False)."""
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    mistral_api_key = os.getenv("MISTRAL_API_KEY")

    if gemini_api_key and genai is not None:
        return GeminiClient(gemini_api_key), True
    elif mistral_api_key:
        return MistralClient(mistral_api_key), True
    return None, False


if __name__ == "__main__":
    import argparse
    import asyncio
    import sys

    parser = argparse.ArgumentParser(
        description="Quick CLI wrapper around the configured LLM to answer a prompt."
    )
    parser.add_argument("prompt", help="User prompt/question to send to the model")
    args = parser.parse_args()

    client, ok = get_default_client()
    if not ok or client is None:
        print("Error: GEMINI_API_KEY not set or google-genai unavailable", file=sys.stderr)
        sys.exit(1)

    async def _main() -> None:
        async for chunk in client.stream(args.prompt, ""):
            print(chunk, end="", flush=True)
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass