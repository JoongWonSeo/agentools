from typing import Optional
from collections.abc import Iterator

from .api import openai_chat, NOT_GIVEN, ChatCompletion, ChatCompletionMessage
from .messages import MessageHistory, msg
from .tools import Toolkit, call_requested_function

#================ Callback-based GPTs =================#
async def chat(prompt, event_hander) -> str:
    '''
    Prompt a model and returns the response.
    Offers a callbacks/hooks for handling events, such as:
    - on_function_call (tool)
    - on_function_return (tool)
    - on_partial (streaming)
    - on_completion (final)
    '''
    pass

#================ Generator-based GPTs =================#
async def chat_event_stream(prompt) -> Iterator[dict]:
    '''
    Yield events from a chat completion stream.
    '''
    pass


#================ Class-based GPTs =================#

class ChatGPT:
    '''Supports tools, and conversation histories'''
    def __init__(self, messages: MessageHistory, toolkit: Optional[Toolkit]=None, model='gpt-3.5-turbo'):
        self.default_model = model
        self.default_toolkit = toolkit

        self.messages = messages

    async def __call__(self, user: str, toolkit=None, model=None, max_function_calls=100, **openai_kwargs) -> str:

        model = model or self.default_model

        toolkit = toolkit or self.default_toolkit
        if toolkit:
            tools = toolkit.schema
            lookup = toolkit.lookup

        self.messages.append(msg(user=user))

        for i in range(max_function_calls):
            completion: ChatCompletion = await openai_chat(
                messages=self.messages.history,
                tools=tools if toolkit else NOT_GIVEN,
                model=model,
                **openai_kwargs
            )
            
            response: ChatCompletionMessage = completion.choices[0].message
            self.messages.append(response)

            if response.content:
                # handle message
                print(f"[ChatGPT]: {response.content}")
            if toolkit and response.tool_calls:
                for call in response.tool_calls:
                    # handle tool calls (usually nothing more than logging/streaming)
                    print(f"[Tool Call]: {call.function.name} {call.function.arguments}")
                    result = await call_requested_function(call.function, lookup)
                    # stringify result
                    result = str(result)
                    print(f"[Tool Result]: {result}")

                    self.messages.append(msg(tool=result, tool_call_id=call.id))
            else:
                # IMPORTANT no more tool calls, we're done, return the final response
                return response.content
        raise Exception(f"Exceeded max function calls ({max_function_calls})")