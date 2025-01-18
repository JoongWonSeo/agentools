from typing import Iterable, Literal, Unpack, overload
from abc import ABC, abstractmethod
from copy import deepcopy

from pydantic import BaseModel
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam as Message,
)
from openai.types.chat.chat_completion_developer_message_param import (
    ChatCompletionDeveloperMessageParam as DeveloperMessage,
)
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam as SystemMessage,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam as UserMessage,
)
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam as AssistantMessage,
)
from openai.types.chat.chat_completion_tool_message_param import (
    ChatCompletionToolMessageParam as ToolMessage,
)

from .content import Content, TextContent


@overload
def msg(*, developer: str | Iterable[TextContent]) -> DeveloperMessage:
    """
    Define a developer message by keyword argument.

    Note that the developer message replaces the system message in o1 and newer.
    """
    ...


@overload
def msg(*, system: str | Iterable[TextContent]) -> SystemMessage:
    """Define a system message by keyword argument"""
    ...


@overload
def msg(*, user: str | Iterable[Content]) -> UserMessage:
    """Define a user message by keyword argument"""
    ...


@overload
def msg(*, assistant: str) -> AssistantMessage:
    """Define an assistant message by keyword argument"""
    ...


@overload
def msg(*, tool: str, tool_call_id: str) -> ToolMessage:
    """Define a tool message (tool result) by keyword argument"""
    ...


@overload
def msg(**kwargs: Unpack[Message]) -> Message:
    """Define a message by keyword arguments"""


def msg(
    *,
    developer=None,
    system=None,
    user=None,
    assistant=None,
    tool=None,
    tool_call_id=None,
    role: Literal["developer", "system", "user", "assistant", "tool"] | None = None,
    **kwargs,
) -> Message:
    """
    Define a developer/system/user/assistant/tool message by keyword arguments.
    """
    assert (
        sum(arg is not None for arg in [developer, system, user, assistant, tool, role])
        == 1
    ), "Only one of developer/system/user/assistant/tool/role should be specified"
    assert (
        tool is None or tool_call_id is not None
    ), "tool_call_id must be specified if tool is specified"

    if developer is not None:
        return {"role": "developer", "content": developer}
    elif system is not None:
        return {"role": "system", "content": system}
    elif user is not None:
        return {"role": "user", "content": user}
    elif assistant is not None:  # TODO: assistant tool calls
        return {"role": "assistant", "content": assistant}
    elif tool is not None:
        return {"role": "tool", "content": tool, "tool_call_id": tool_call_id}
    elif role is not None:
        # TODO: runtime validation would be great, but TypeAdapter currently doesn't support nested TypedDict validation
        # m = {"role": role, **kwargs}
        # TypeAdapter(Message).validate_python(m)
        # return m

        return {"role": role, **kwargs}


class MessageHistory(ABC):
    # history of already completed messages
    history: list[dict]

    @abstractmethod
    async def append(self, message: dict | ChatCompletionMessage):
        """Add a message to the history"""
        ...

    @abstractmethod
    async def reset(self):
        """Reset the history"""
        ...

    @staticmethod
    def ensure_dict(message: dict | ChatCompletionMessage):
        """Convert assistant pydantic message to dict"""
        if isinstance(message, BaseModel):
            message = message.model_dump()
            # remove any None values (tool calls)
            message = {k: v for k, v in message.items() if v is not None}
        return message


class SimpleHistory(MessageHistory):
    """
    Simple message history that stores messages as a list of dicts.
    Kind of acts like a list
    """

    def __init__(self, initial: list[dict] = []):
        self.initial = initial
        self.history = deepcopy(initial)

    async def append(self, message: dict | ChatCompletionMessage):
        message = self.ensure_dict(message)
        self.history.append(message)

    async def reset(self):
        self.history = deepcopy(self.initial)

    @classmethod
    def system(cls, system: str):
        """Convenience method for creating a history with a single system message"""
        return cls([msg(system=system)])


# # example of extended features, simply override
# class StatefulSystemMessage(MessageHistory):
#     def __init__(self, system_template: str, messages: list[dict] = []):
#         assert all(message['role'] != 'system' for message in messages), "Do not include system messages in messages list, use system_template instead"
#         super().__init__(messages)
#         self.system_template = system_template
#         self.state = {'x': 1}

#     @property
#     def history(self):
#         system_message = self.system_template.format(**self.state)
#         return [system_message] + super().history

# # another common example
# class RAGMessageHistory(MessageHistory):
#     def __init__(self, db, messages: list[dict] = []):
#         super().__init__(messages)
#         self.db = db

#     # just use add and check 'role' field...
#     def add_user(self, message: dict | ChatCompletionUserMessageParam):
#         relevant_context = self.db.retrieve(message)
#         super().add_user(message)
