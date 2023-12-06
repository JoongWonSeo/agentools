from typing import Optional, Callable
from collections.abc import Iterator
from abc import ABC, abstractmethod
import asyncio

from .api import openai_chat, NOT_GIVEN, ChatCompletion, ChatCompletionMessage, ToolCall
from .messages import MessageHistory, msg
from .tools import Tools, call_requested_function


#================ Assistant =================#

class Assistant(ABC):
    '''Generic AI Assistant with memory and function calling high-level overridable methods and hooks for flexible customization'''




class ChatGPT(Assistant):
    '''ChatGPT with default model and toolkit'''
    def __init__(self, messages: MessageHistory, tools: Optional[Tools] = None, model: str = 'gpt-3.5-turbo'):
        self.default_model = model
        self.default_tools = tools

        self.messages = messages

    #================ High-Level Functions =================#
    async def request_chat(self, messages: list[dict], tools: Tools, model: str, **openai_kwargs) -> ChatCompletion:
        '''Request a chat from OpenAI API'''
        return await openai_chat(messages=messages, tools=tools.schema, model=model, **openai_kwargs)        
    
    async def get_message(self, completion: ChatCompletion) -> ChatCompletionMessage:
        '''Get the message from a chat completion. You could implement the streaming preview here.'''
        return completion.choices[0].message
    
    async def get_tool_result(self, call: ToolCall, tools: Tools) -> str:
        '''Execute the requested tool call and return its result.'''
        print(f"[Tool Call]: {call.function.name} {call.function.arguments}")
        result = await call_requested_function(call.function, tools.lookup)
        result = str(result)
        print(f"[Tool Result]: {result}")
        return result
    
    #================ Hooks =================#
    async def on_call(self, user: str, tools: Optional[Tools], model: str, max_function_calls: int, **openai_kwargs):
        pass

    async def on_message(self, message: str):
        print(f"[ChatGPT]: {message}")

    async def on_finish(self, completion: ChatCompletion):
        pass

    async def on_max_function_calls(self, completion: ChatCompletion):
        pass


    async def __call__(self, user: str, tools: Optional[Tools] = None, model: Optional[str] = None, max_function_calls: int = 100, **openai_kwargs) -> str:

        model = model or self.default_model
        tools = tools or self.default_tools

        await self.on_call(user=user, tools=tools, model=model, max_function_calls=max_function_calls, **openai_kwargs)

        self.messages.append(msg(user=user))

        for i in range(max_function_calls):
            completion = await self.request_chat(
                messages=self.messages.history,
                tools=tools if tools else NOT_GIVEN,
                model=model,
                **openai_kwargs
            )
            response = await self.get_message(completion)
            self.messages.append(response)

            if response.content:
                # handle message
                await self.on_message(response.content)

            if tools and response.tool_calls:
                # handle tool calls
                for call in response.tool_calls:
                    result = await self.get_tool_result(call, tools)
                    self.messages.append(msg(tool=result, tool_call_id=call.id))
            else:
                # IMPORTANT no more tool calls, we're done, return the final response
                await self.on_finish(completion)
                return response.content
            
        await self.on_max_function_calls(response)
        raise Exception(f"Exceeded max function calls ({max_function_calls})")