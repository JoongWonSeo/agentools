import pytest

from agentools.api.openai import openai_chat, accumulate_partial
from agentools.messages import msg


@pytest.mark.asyncio
async def test_openai_chat_mock():
    completion = await openai_chat(model="mock:hi", messages=[msg(user="hi")])
    assert completion.choices[0].message.content == "hi"


@pytest.mark.asyncio
async def test_openai_chat_stream_accumulate():
    stream = await openai_chat(model="mock:abc", messages=[msg(user="go")], stream=True)
    parts = []
    async for chunk, partial in accumulate_partial(stream):
        parts.append(chunk.choices[0].delta.content or "")
    assert "".join(parts) == "abc"
    assert partial.choices[0].message.content == "abc"
