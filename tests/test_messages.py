import pytest

from agentools.messages import msg, SimpleHistory


@pytest.mark.asyncio
async def test_simple_history_append_reset():
    hist = SimpleHistory()
    await hist.append(msg(user="hello"))
    assert hist.history[-1]["content"] == "hello"
    await hist.reset()
    assert hist.history == hist.initial


def test_msg_helper():
    m = msg(system="sys")
    assert m["role"] == "system"
    assert m["content"] == "sys"
