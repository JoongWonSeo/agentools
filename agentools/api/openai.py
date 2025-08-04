"""
Interactions with the OpenAI API and wrappers around the API.
"""

from typing import AsyncIterator, Callable, ParamSpec, TypeVar

import litellm
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat.chat_completion import ChatCompletion as ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_message import (
    ChatCompletionMessage as ChatCompletionMessage,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall as ToolCall,
)
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.types.chat.completion_create_params import ResponseFormat
from pydantic import BaseModel

P = ParamSpec("P")
R = TypeVar("R")


def create_wrapped_openai(
    openai_func_for_typing: Callable[P, R],
):
    """
    Due to python's typing limitations, to receive static type checking without re-declaring the entire OpenAI API,
    we need to create a wrapper around the OpenAI API.
    So the given parameter `openai_func_for_typing` is only used for type checking,
    it won't actually be called.
    """

    def wrapped_openai_chat_completion_create(
        client: AsyncOpenAI | Callable | None = None,
        # Litellm params
        mock_response: str | None = None,
        custom_llm_provider: str | None = None,
        # Modified OpenAI params
        response_format: type[BaseModel] | ResponseFormat | None = None,
        # Remaining OpenAI params
        *openai_args: P.args,
        **openai_kwargs: P.kwargs,
    ) -> R:
        """
        Thin wrapper around openai with mocking and potential for other features/backend
        - Use model='mock' to simply return "Hello, world"
        - Use model="mock:<message>" to return the message
        - Use model='echo' to echo the last message

        Args:
            client: the client or async completion function to use, or None to use the default litellm client
            mock_response: Litellm param, response to return even if the model is not 'mock'
            custom_llm_provider: Litellm param, the provider to use for the model
            *openai_args: OpenAI params
            **openai_kwargs: OpenAI params
        """

        model = openai_kwargs.get("model")
        assert isinstance(model, str), "model must be a string"
        if model.startswith("mock"):
            mocked = model.split(":", 1)[1] if ":" in model else "Hello, world!"
            mock_response = mock_response or mocked  # prioritize the explicit param
            custom_llm_provider = custom_llm_provider or "openai"
        elif model == "echo":
            messages = openai_kwargs.get("messages", [])
            assert isinstance(messages, list), "messages must be a list"
            last_msg = messages[-1]["content"] if messages else "Hello, world!"
            mock_response = mock_response or last_msg
            custom_llm_provider = custom_llm_provider or "openai"

        if client is None:
            acompletion = litellm.acompletion
            # FIXME: since the return value is slightly different, we need to convert it back (lossy)
            # wrap with ChatCompletion.model_validate(r.model_dump())
        elif isinstance(client, Callable):
            acompletion = client
        else:
            acompletion = client.chat.completions.create
        return acompletion(
            *openai_args,  # type: ignore
            response_format=response_format,
            **openai_kwargs,
            mock_response=mock_response,
            custom_llm_provider=custom_llm_provider,
        )

    return wrapped_openai_chat_completion_create


openai_chat = create_wrapped_openai(
    openai_func_for_typing=AsyncOpenAI(api_key="").chat.completions.create
)

litellm_chat = create_wrapped_openai(openai_func_for_typing=litellm.acompletion)


async def accumulate_partial(
    stream: AsyncStream[ChatCompletionChunk],
) -> AsyncIterator[tuple[ChatCompletionChunk, ChatCompletion]]:
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
                completion = ChatCompletion(
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
                            Choice(
                                finish_reason="length",  # NOTE: this is a fallback
                                index=i,
                                logprobs=None,
                                message=ChatCompletionMessage(
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

                if delta_message is None:
                    # message itself was not updated, only misc. update like content_filter_*
                    continue

                # update the message
                message.role = delta_message.role or message.role  # type: ignore

                if delta_message.content:
                    if not message.content:
                        message.content = delta_message.content
                    else:
                        message.content += delta_message.content

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

            if hasattr(chunk, "usage") and chunk.usage:
                completion.usage = chunk.usage

            yield chunk, completion
    except Exception as e:
        raise e
    finally:
        try:
            await stream.response.aclose()
        except Exception:
            pass
