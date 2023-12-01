from collections.abc import Iterator
import json
import inspect

from openai import OpenAI, AsyncOpenAI, AsyncStream
# from openai.types.chat.chat_completion import ChatCompletion, Choice, ChatCompletionMessage
# from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice as ChunkChoice
import openai.types.chat.chat_completion as Normal
import openai.types.chat.chat_completion_chunk as Stream
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall as NormalToolCall,
    Function as NormalFunction
)


from .utils import mock_response


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

async def accumulate_chunks(stream: AsyncStream[Stream.ChatCompletionChunk], track_usage=False) -> AsyncStream[Normal.ChatCompletion]:
    '''
    Adapter that accumulates a stream of deltas into a stream of partial messages,
    e.g. "I", "love", "sushi" -> "I", "I love", "I love sushi"
    Almost everything will be indistinguishable from a normal completion, except:
    - `completion.usage` will be None, unless `track_usage` is set and prompt is provided.
    - `completion.choices[].finish_reason` will be 'length' during the partials.
    - `completion.choices[].message.tool_calls[].id` will be an empty string during the partials.
        -> HOWEVER, experiments show that tool ids and names are always included in the initial delta, so this should never be actually visible.
    '''
    completion = None

    async for chunk in stream:
        if completion is None:
            completion = Normal.ChatCompletion(
                id=chunk.id,
                choices=[], # populated later
                created=chunk.created,
                model=chunk.model,
                object="chat.completion",
                system_fingerprint=chunk.system_fingerprint,
                usage=None # TODO: populate this by counting tokens
            )

        for delta_choice in chunk.choices:
            # ensure this choice exists in the completion
            if len(completion.choices) <= delta_choice.index:
                completion.choices.extend([
                    Normal.Choice(
                        finish_reason='length', # NOTE: this is a fallback
                        index=i,
                        message=Normal.ChatCompletionMessage(role='assistant', content=None)
                    )
                    for i in range(len(completion.choices), delta_choice.index + 1)
                ])
            
            choice = completion.choices[delta_choice.index]
            message = choice.message
            delta_message = delta_choice.delta

            # update the choice
            choice.finish_reason = delta_choice.finish_reason or choice.finish_reason

            # update the message
            message.role = delta_message.role or message.role
            
            if delta_message.content:
                if not message.content:
                    message.content = delta_message.content
                else:
                    message.content += delta_message.content
                # TODO: update usage stats

            if delta_message.tool_calls:
                # ensure tool_calls list exists
                if message.tool_calls is None:
                    message.tool_calls = [] # populated later

                for delta_tool_call in delta_message.tool_calls:
                    # ensure this tool_call exists in the message
                    if len(message.tool_calls) <= delta_tool_call.index:
                        message.tool_calls.extend([
                            NormalToolCall(
                                id='', # NOTE: no initial id
                                type='function',
                                function=NormalFunction(name='', arguments='')
                            )
                            for i in range(len(message.tool_calls), delta_tool_call.index + 1)
                        ])
                    
                    tool_call = message.tool_calls[delta_tool_call.index]

                    # update the tool_call
                    tool_call.id = delta_tool_call.id or tool_call.id
                    tool_call.type = delta_tool_call.type or tool_call.type

                    # update the function
                    delta_function = delta_tool_call.function
                    if delta_function:
                        # experimental testing shows that the function name is always fully returned, no matter how long
                        tool_call.function.name = delta_function.name or tool_call.function.name

                        if delta_function.arguments:
                            tool_call.function.arguments += delta_function.arguments

            yield completion


#================ Class-based GPTs =================#



async def openai_chat(**openai_kwargs):
    '''
    Thin wrapper around openai with mocking and potential for other features/backend
    '''
    # TODO: support mocked streaming
    if openai_kwargs['model'] == 'mock':
        return mock_response("Hello, world!")
    elif openai_kwargs['model'] == 'echo':
        return mock_response(openai_kwargs['messages'][-1]['content'])
    client = AsyncOpenAI()
    return await client.chat.completions.create(**openai_kwargs)


async def call_requested_function(call_request, func_lookup):
    '''
    Call the requested function generated by the model.
    '''
    # parse function call
    func_name = call_request.name
    arguments = call_request.arguments

    if func_name not in func_lookup:
        return f"Error: Function {func_name} does not exist."
    try:
        params = json.loads(arguments)
    except Exception as e:    
        return f"Error: Failed to parse arguments, make sure your arguments is a valid JSON object: {e}"

    # call function
    try:
        return_value = func_lookup[func_name](**params)
        # if it's a coroutine, await it
        if inspect.iscoroutine(return_value):
            print("awaiting coroutine")
            return await return_value
        else:
            return return_value
    except Exception as e:
        return f"Error: {e}"