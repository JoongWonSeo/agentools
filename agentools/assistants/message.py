from typing import overload
from copy import deepcopy

from pydantic import BaseModel
from openai.types.chat import ChatCompletionMessage

# TODO: allow for more complex messages, like image and audio


@overload
def msg(*, system: str) -> dict[str, str]: ...


@overload
def msg(*, user: str) -> dict[str, str]: ...


@overload
def msg(*, assistant: str) -> dict[str, str]: ...


@overload
def msg(*, tool: str, tool_call_id: str) -> dict[str, str]: ...


def msg(*, system=None, user=None, assistant=None, tool=None, tool_call_id=None):
    """
    Define a system/user/assistant/tool message by keyword arguments.
    """
    assert (
        sum(arg is not None for arg in [system, user, assistant, tool]) == 1
    ), "Only one of system/user/assistant/tool should be specified"
    assert (
        tool is None or tool_call_id is not None
    ), "tool_call_id must be specified if tool is specified"

    if system is not None:
        return {"role": "system", "content": system}
    elif user is not None:
        return {"role": "user", "content": user}
    elif assistant is not None:  # TODO: assistant tool calls
        return {"role": "assistant", "content": assistant}
    elif tool is not None:
        return {"role": "tool", "content": tool, "tool_call_id": tool_call_id}


class MessageHistory:
    """
    Simple message history that stores messages as a list of dicts.
    Kind of acts like a list.

    Extend this class to customize.
    """

    def __init__(self, initial: list[dict] = []):
        self.initial = initial
        self.history = deepcopy(initial)

    async def append(self, message: dict | ChatCompletionMessage):
        """Add a message to the history"""
        message = self.ensure_dict(message)
        self.history.append(message)

    async def reset(self):
        """Reset the history"""
        self.history = deepcopy(self.initial)

    @classmethod
    def system(cls, system: str):
        """Convenience method for creating a history with a single system message"""
        return cls([msg(system=system)])

    @staticmethod
    def ensure_dict(message: dict | ChatCompletionMessage):
        """Convert assistant pydantic message to dict"""
        if isinstance(message, BaseModel):
            message = message.model_dump()
            # remove any None values (tool calls)
            message = {k: v for k, v in message.items() if v is not None}
        return message
