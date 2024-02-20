import time
import asyncio
from uuid import uuid4
from copy import deepcopy

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


# ========== Recorder ========== #
class AsyncGeneratorRecorder:
    """
    An adapter class that wraps an async generator and records the output, including the time between each output.
    Each call is recorded in their order, and the replayer will replay the responses in the same order.
    """

    def __init__(self):
        self.recordings = []
        self.replay_index = 0

    async def _record(self, async_generator):
        recording = []
        t = time.time()
        async for c in async_generator:
            dt = time.time() - t
            t = time.time()
            recording.append((dt, deepcopy(c)))
            yield c
        self.recordings.append(recording)

    async def record(self, async_generator):
        return MockAsyncGeneratorWrapper(self._record(async_generator))

    async def _replay(self, index):
        recording = self.recordings[index]
        for dt, c in recording:
            await asyncio.sleep(dt)
            yield c

    async def replay(self):
        r = self._replay(self.replay_index)
        self.replay_index = (self.replay_index + 1) % len(self.recordings)
        return MockAsyncGeneratorWrapper(r)


class Recordings:
    def __init__(self):
        self.recordings = []
        self.current_recorder = None

    def start(self):
        self.current_recorder = AsyncGeneratorRecorder()

    def stop(self):
        if self.current_recorder and self.current_recorder.recordings:
            self.recordings.append(self.current_recorder)
        self.current_recorder = None

    @property
    def replay_models(self):
        return [f"replay:{i}" for i in range(len(self.recordings))]


GLOBAL_RECORDINGS = Recordings()
