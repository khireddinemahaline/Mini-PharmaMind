import os
import asyncio
import logging
import inspect
from pathlib import Path

from dotenv import load_dotenv
from autogen_ext.models.anthropic import AnthropicChatCompletionClient

env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)

logger = logging.getLogger(__name__)
if not logger.handlers:
    # Basic config for standalone runs; the application may override this.
    logging.basicConfig(level=logging.INFO)

api_key = os.getenv("DEEPSEEK_API")
if not api_key:
    logger.error("DEEPSEEK_API environment variable is not set.")
    raise ValueError("DEEPSEEK_API environment variable is not set.")


def _build_raw_client() -> AnthropicChatCompletionClient:
    return AnthropicChatCompletionClient(
        model="deepseek-v4-flash",
        api_key=api_key,
        base_url="https://api.deepseek.com/anthropic",

        model_info={
            "family": "deepseek",
            "function_calling": True,
            "vision": False,
            "json_output": False,
            "structured_output": False,
        },

        reasoning_effort="max",

        extra_body={
            "thinking": {
                "type": "enabled"
            }
        },
        parallel_tool_calls=False,
        max_tokens=300000,
        timeout=200,
        max_retries=10,
    )


class ProxyModelClient:
    """A minimal proxy around the real client that adds retries and
    centralized logging while preserving attribute access.

    It wraps async callables to retry transient failures and logs
    `asyncio.CancelledError` clearly before re-raising.
    """

    def __init__(self, client, retries: int = 3):
        self._client = client
        self._retries = retries

    def __getattr__(self, name):
        attr = getattr(self._client, name)

        # Async generator functions need to be wrapped with an async
        # generator wrapper so that calling them returns an async
        # iterator (object with __aiter__), not a coroutine.
        if inspect.isasyncgenfunction(attr):
            async def _asyncgen_wrapped(*args, **kwargs):
                attempt = 0
                while True:
                    try:
                        async for item in attr(*args, **kwargs):
                            yield item
                        return
                    except asyncio.CancelledError:
                        logger.warning(
                            "Model client async generator cancelled",
                            exc_info=True,
                        )
                        raise
                    except Exception as exc:
                        attempt += 1
                        logger.warning(
                            "Model client async generator failed (attempt %s/%s): %s",
                            attempt,
                            self._retries,
                            exc,
                            exc_info=True,
                        )
                        if attempt > self._retries:
                            logger.exception(
                                "Model client async generator failed after retries"
                            )
                            raise
                        await asyncio.sleep(min(1 * attempt, 5))

            return _asyncgen_wrapped

        # Coroutine functions should remain async callables that return
        # coroutines when invoked (caller awaits them).
        if inspect.iscoroutinefunction(attr):
            class _AwaitableAsyncIterable:
                def __init__(self, call_func, args, kwargs, retries):
                    self._call_func = call_func
                    self._args = args
                    self._kwargs = kwargs
                    self._retries = retries

                async def _invoke(self):
                    attempt = 0
                    while True:
                        try:
                            return await self._call_func(*self._args, **self._kwargs)
                        except asyncio.CancelledError:
                            logger.warning(
                                "Model client call cancelled (CancelledError)",
                                exc_info=True,
                            )
                            raise
                        except Exception as exc:
                            attempt += 1
                            logger.warning(
                                "Model client coroutine failed (attempt %s/%s): %s",
                                attempt,
                                self._retries,
                                exc,
                                exc_info=True,
                            )
                            if attempt > self._retries:
                                logger.exception(
                                    "Model client coroutine failed after retries"
                                )
                                raise
                            await asyncio.sleep(min(1 * attempt, 5))

                def __await__(self):
                    return self._invoke().__await__()

                async def __aiter__(self):
                    attempt = 0
                    while True:
                        try:
                            result = await self._call_func(*self._args, **self._kwargs)

                            # If the result is an async iterable, forward it.
                            if hasattr(result, "__aiter__"):
                                async for item in result:
                                    yield item
                                return

                            # If the result is a regular iterable, iterate synchronously.
                            if hasattr(result, "__iter__"):
                                for item in result:
                                    yield item
                                return

                            # Not iterable: nothing to yield.
                            return

                        except asyncio.CancelledError:
                            logger.warning(
                                "Model client async iteration cancelled",
                                exc_info=True,
                            )
                            raise
                        except Exception as exc:
                            attempt += 1
                            logger.warning(
                                "Model client async iteration failed (attempt %s/%s): %s",
                                attempt,
                                self._retries,
                                exc,
                                exc_info=True,
                            )
                            if attempt > self._retries:
                                logger.exception(
                                    "Model client async iteration failed after retries"
                                )
                                raise
                            await asyncio.sleep(min(1 * attempt, 5))

            def _wrapped(*args, **kwargs):
                return _AwaitableAsyncIterable(attr, args, kwargs, self._retries)

            return _wrapped

        # Synchronous callables: wrap in a sync function that may return an
        # async iterator or regular result. We don't make the wrapper async
        # to preserve the original call semantics.
        if callable(attr):
            def _sync_wrapped(*args, **kwargs):
                attempt = 0
                while True:
                    try:
                        result = attr(*args, **kwargs)
                        return result
                    except Exception as exc:
                        attempt += 1
                        logger.warning(
                            "Model client call failed (attempt %s/%s): %s",
                            attempt,
                            self._retries,
                            exc,
                            exc_info=True,
                        )
                        if attempt > self._retries:
                            logger.exception(
                                "Model client call failed after retries"
                            )
                            raise
                        # sleep synchronously is not ideal; break to avoid blocking
                        # and escalate to caller for retry logic if needed.
                        raise

            return _sync_wrapped

        return attr


try:
    _raw = _build_raw_client()
    model_client = ProxyModelClient(_raw, retries=3)
    logger.info("Model client initialized successfully.")
except Exception:
    logger.exception("Failed to initialize model client.")
    # Provide a sentinel that will raise informative errors on use.
    class _MissingClient:
        def __getattr__(self, name):
            def _fail(*args, **kwargs):
                raise RuntimeError(
                    "Model client is not available due to initialization failure."
                )

            return _fail

    model_client = _MissingClient()