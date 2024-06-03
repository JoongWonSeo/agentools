"""
Interactions with the OpenAI API and wrappers around the API.
"""

from typing import AsyncIterator

from openai import AsyncStream
import openai.types.chat.chat_completion as Normal
from openai.types.chat.chat_completion import (
    ChatCompletion as ChatCompletion,
    ChatCompletionMessage as ChatCompletionMessage,
)  # re-export to __init__.py
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall as ToolCall,
    Function,
)


from .model import Model
from .mocking import mock, GLOBAL_RECORDINGS


async def execute_api(model: Model, **openai_kwargs):
    """
    Args:
        client: the client to use, or None to use the default AsyncOpenAI client
    """

    # The generator to return
    gen = None

    stream = openai_kwargs.get("stream")
    match model.client:
        case Model.Mock():
            # model_name being the word to be mocked
            gen = await mock(stream, model.model_name)
        case Model.Echo():
            last_msg = openai_kwargs["messages"][-1]["content"]
            gen = await mock(stream, last_msg)
        case Model.Replay():
            r = model.model_name
            replay_model = GLOBAL_RECORDINGS.recordings[int(r)]
            gen = await replay_model.replay()

        case _:
            # AsyncGroq or
            # AsyncOpenAI or
            # AsyncOpenAI(base_url=url)
            client = model.client
            # Pass on the model name
            gen = await client.chat.completions.create(
                model=model.model_name, **openai_kwargs
            )

    # Record the response if recording
    if GLOBAL_RECORDINGS.current_recorder:
        if stream:
            return await GLOBAL_RECORDINGS.current_recorder.record(gen)
        else:
            raise ValueError("Recording non-streaming responses is not supported yet.")
    else:
        return gen


async def accumulate_partial(
    stream: AsyncStream[ChatCompletionChunk],
) -> AsyncIterator[tuple[ChatCompletionChunk, Normal.ChatCompletion]]:
    """
    Adapter that accumulates a stream of deltas into a stream of partial messages,
    e.g. "I", "love", "sushi" -> "I", "I love", "I love sushi"
    Almost everything will be indistinguishable from a normal completion, except:
    - `completion.usage` will be None, unless `track_usage` is set and prompt is provided.
    - `completion.choices[].finish_reason` will be 'length' during the partials.
    - `completion.choices[].message.tool_calls[].id` *could* be empty for a few chunks, HOWEVER, experiments show that tool ids and names are always included in the initial delta, so this should never be actually visible.

    Args:
        stream: the streaming response from OpenAI API

    """
    completion = None

    try:
        async for chunk in stream:
            if completion is None:
                completion = Normal.ChatCompletion(
                    id=chunk.id,
                    choices=[],  # populated later
                    created=chunk.created,
                    model=chunk.model,
                    object="chat.completion",
                    system_fingerprint=chunk.system_fingerprint,
                    usage=None,
                )

            for delta_choice in chunk.choices:
                # ensure this choice exists in the completion
                if len(completion.choices) <= delta_choice.index:
                    completion.choices.extend(
                        [
                            Normal.Choice(
                                finish_reason="length",  # NOTE: this is a fallback
                                index=i,
                                logprobs=None,
                                message=Normal.ChatCompletionMessage(
                                    role="assistant", content=None
                                ),
                            )
                            for i in range(
                                len(completion.choices), delta_choice.index + 1
                            )
                        ]
                    )

                choice = completion.choices[delta_choice.index]
                message = choice.message
                delta_message = delta_choice.delta

                # update the choice
                choice.finish_reason = (
                    delta_choice.finish_reason or choice.finish_reason
                )

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
                        message.tool_calls = []  # populated later

                    for delta_tool_call in delta_message.tool_calls:
                        # ensure this tool_call exists in the message
                        if len(message.tool_calls) <= delta_tool_call.index:
                            message.tool_calls.extend(
                                [
                                    ToolCall(
                                        id="",  # NOTE: no initial id
                                        type="function",
                                        function=Function(name="", arguments=""),
                                    )
                                    for i in range(
                                        len(message.tool_calls),
                                        delta_tool_call.index + 1,
                                    )
                                ]
                            )

                        tool_call = message.tool_calls[delta_tool_call.index]

                        # update the tool_call
                        tool_call.id = delta_tool_call.id or tool_call.id
                        tool_call.type = delta_tool_call.type or tool_call.type

                        # update the function
                        delta_function = delta_tool_call.function
                        if delta_function:
                            # experimental testing shows that the function name is always fully returned, no matter how long
                            tool_call.function.name = (
                                delta_function.name or tool_call.function.name
                            )

                            if delta_function.arguments:
                                tool_call.function.arguments += delta_function.arguments

            yield chunk, completion
    except Exception as e:
        raise e
    finally:
        await stream.response.aclose()
