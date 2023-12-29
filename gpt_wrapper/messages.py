from typing import overload
from abc import ABC, abstractmethod

# API return values
from openai.types.chat import (
    ChatCompletion, # Overall Completion, has id, stats, choices
    ChatCompletionMessage, # completion.choice[0], has role, content, tool_calls
)

@overload
def msg(*, system: str) -> dict[str, str]:
    ...
@overload
def msg(*, user: str) -> dict[str, str]:
    ...
@overload
def msg(*, assistant: str) -> dict[str, str]:
    ...
@overload
def msg(*, tool: str, tool_call_id: str) -> dict[str, str]:
    ...
def msg(*, system=None, user=None, assistant=None, tool=None, tool_call_id=None):
    '''
    Define a system/user/assistant/tool message by keyword arguments.
    '''
    assert sum(arg is not None for arg in [system, user, assistant, tool]) == 1, "Only one of system/user/assistant/tool should be specified"
    assert tool is None or tool_call_id is not None, "tool_call_id must be specified if tool is specified"

    if system is not None:
        return {'role': 'system', 'content': system}
    elif user is not None:
        return {'role': 'user', 'content': user}
    elif assistant is not None: # TODO: assistant tool calls
        return {'role': 'assistant', 'content': assistant}
    elif tool is not None:
        return {'role': 'tool', 'content': tool, 'tool_call_id': tool_call_id}

class MessageHistory(ABC):
    @property
    @abstractmethod
    def history(self) -> list[dict]:
        '''The message history'''
        ...
    
    @abstractmethod
    def append(self, message: dict | ChatCompletionMessage):
        '''Add a message to the history'''
        ...
    
    @staticmethod
    def ensure_dict(message: dict | ChatCompletionMessage):
        '''Convert assistant pydantic message to dict'''
        if isinstance(message, ChatCompletionMessage):
            message = message.model_dump()
            # remove any None values (tool calls)
            message = {k: v for k, v in message.items() if v is not None}
        return message


class SimpleHistory(MessageHistory):
    '''
    Simple message history that stores messages as a list of dicts.
    Kind of acts like a list
    '''
    def __init__(self, messages: list[dict] = [msg(system='You are a helpful assistant')]):
        self._messages = messages
    
    @property
    def history(self):
        return self._messages
    
    def append(self, message: dict | ChatCompletionMessage):
        message = self.ensure_dict(message)
        self._messages.append(message)
    

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