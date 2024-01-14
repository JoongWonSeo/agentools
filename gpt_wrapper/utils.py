import time
import asyncio
from uuid import uuid4

import tiktoken

# API return values
from openai.types.chat.chat_completion import (
    ChatCompletion, # Overall Completion, has id, stats, choices
    Choice, # completion.choice[0], has finish_reason, index, message
    ChatCompletionMessage, # completion.choice[0].message, has role, content, tool_calls
)
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice as ChoiceChunk,
    ChoiceDelta,
)


tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


def mock_response(message: str):
    return ChatCompletion(
            id='mock',
            choices=[Choice(
                finish_reason='stop',
                index=0,
                message=ChatCompletionMessage(
                    role='assistant',
                    content=message
                )
            )],
            created=int(time.time()),
            model='mock',
            object='chat.completion',
        )

async def mock_streaming_response(message: str):
    async def gen():
        id = str(uuid4())[:8]
        t = int(time.time())

        for i, c in enumerate(message):
            last = i == len(message) - 1
            await asyncio.sleep(0.1)
            yield ChatCompletionChunk(
                    id=f'chatcmpl-mock-{id}',
                    choices=[ChoiceChunk(
                        delta=ChoiceDelta(
                            role='assistant',
                            content=c
                        ),
                        finish_reason='stop' if last else None,
                        index=0,
                    )],
                    created=t,
                    model='mock',
                    object='chat.completion.chunk',
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


async def atuple(*vals):
    '''A generalized "tuple" wrapper which will await any coroutine values and simply pass normal values through'''
    return tuple([await v if asyncio.iscoroutine(v) else v for v in vals])
