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

#enable_llm_logging(logging.INFO)

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

    async def stream(
        self,
        prompt: str,
        content: str,
        generation_config: Dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        
        model = "mistral-medium-latest"
        user_content = f"{prompt}\n\n{content}" if content else prompt

        response = await self._client.chat.stream_async(
            model=model,
            messages=[{"role": "user", "content": user_content}],
            stream=True)
        collected: list[str] = []
        logger.info("LLM data has run")
        async for chunk in response:
            try:
                print("Event received:", chunk)
                if chunk.data.choices[0].delta.content is not None:
                    print(chunk.data.choices[0].delta.content, end="")
                    text = chunk.data.choices[0].delta.content
                    collected.append(text)
                    yield text
                else:
                    print("Event does not have a 'data' attribute:", event)
            except Exception as e:
                print(f"Error processing event: {e}")
                continue

        if collected:
            logger.info("LLM Response: %s", "".join(collected))
        yield "".join(collected)  # Yield the complete response at the end

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