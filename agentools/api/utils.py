import time
import asyncio
from uuid import uuid4

from openai.types.chat.chat_completion import (
    ChatCompletion,  # Overall Completion, has id, stats, choices
    Choice,  # completion.choice[0], has finish_reason, index, message
    ChatCompletionMessage,  # completion.choice[0].message, has role, content, tool_calls
)
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice as ChoiceChunk,
    ChoiceDelta,
)

MOCK_INITIAL_DELAY = 0.5
MOCK_STREAMING_DELAY = 0.05


def set_mock_initial_delay(delay):
    global MOCK_INITIAL_DELAY
    MOCK_INITIAL_DELAY = delay


def set_mock_streaming_delay(delay):
    global MOCK_STREAMING_DELAY
    MOCK_STREAMING_DELAY = delay


async def mock_response(message: str, initial_delay=None):
    await asyncio.sleep(MOCK_INITIAL_DELAY if initial_delay is None else initial_delay)
    return ChatCompletion(
        id="mock",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content=message),
            )
        ],
        created=int(time.time()),
        model="mock",
        object="chat.completion",
    )


async def mock_streaming_response(message: str, initial_delay=None, delay=None):
    async def gen():
        id = str(uuid4())[:8]
        t = int(time.time())

        await asyncio.sleep(
            MOCK_INITIAL_DELAY if initial_delay is None else initial_delay
        )
        for i, c in enumerate(message):
            last = i == len(message) - 1
            await asyncio.sleep(MOCK_STREAMING_DELAY if delay is None else delay)
            yield ChatCompletionChunk(
                id=f"chatcmpl-mock-{id}",
                choices=[
                    ChoiceChunk(
                        delta=ChoiceDelta(role="assistant", content=c),
                        finish_reason="stop" if last else None,
                        index=0,
                    )
                ],
                created=t,
                model="mock",
                object="chat.completion.chunk",
            )

    return MockAsyncGeneratorWrapper(gen())


class MockAsyncGeneratorWrapper:
    def __init__(self, async_gen):
        self._async_gen = async_gen
        self.response = MockResponse()

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self._async_gen.__anext__()


class MockResponse:
    async def aclose(self) -> None:
        pass
