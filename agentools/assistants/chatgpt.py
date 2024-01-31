import asyncio
from typing import Callable, Optional, AsyncIterator
from collections.abc import Iterator

from ..api.openai import (
    openai_chat,
    accumulate_partial,
    NOT_GIVEN,
    ChatCompletion,
    ToolCall,
)
from ..tools import Tools, call_requested_function
from ..messages import MessageHistory, SimpleHistory, msg
from .utils import atuple, format_event
from .core import Assistant


class ChatGPT(Assistant):
    """ChatGPT with default model and toolkit"""

    DEFAULT_MODEL = "gpt-3.5-turbo"
    MAX_FUNCTION_CALLS = 100

    def __init__(
        self,
        messages: Optional[MessageHistory] = None,
        tools: Optional[Tools] = None,
        model: str = DEFAULT_MODEL,
    ):
        self.default_model = model
        self.default_tools = tools

        self.messages = messages or SimpleHistory()

    # ========== Event Handlers ========== #
    async def __call__(
        self,
        prompt: str,
        tools: Optional[Tools] = None,
        model: Optional[str] = None,
        max_function_calls: int = MAX_FUNCTION_CALLS,
        event_logger: Optional[Callable] = None,
        **openai_kwargs,
    ) -> str:
        """
        Prompt the assistant, returning its final response.

        Args:
            prompt: the user's prompt
            tools: override the default tools
            model: override the default model
            max_function_calls: maximum number of function calls
            **openai_kwargs: additional arguments to pass to the OpenAI API

        Returns:
            the assistant's final response
        """
        async for event in self.response_events(
            prompt, tools, model, max_function_calls, **openai_kwargs
        ):
            if event_logger:
                event_logger(format_event(event))

            match event:
                case self.ResponseEndEvent():
                    return event.content

    # ========== Event Generators ========== #
    async def response_events(
        self,
        prompt: str,
        tools: Optional[Tools] = None,
        model: Optional[str] = None,
        max_function_calls: int = MAX_FUNCTION_CALLS,
        parallel_calls=True,
        **openai_kwargs,
    ) -> Iterator[Assistant.Event]:
        """
        Generate events from a single user prompt, yielding each message and tool call and everything else as it is received.
        This is the overall generalized form, and you would likely only override the smaller *_events() generators to customize each aspect instead.
        """
        model = model or self.default_model
        tools = tools or self.default_tools

        yield self.ResponseStartEvent(
            prompt, tools, model, max_function_calls, openai_kwargs
        )
        await self.messages.append(msg(user=prompt))

        # prompt the assistant and let it use the tools
        for call_index in range(max_function_calls):
            yield self.CompletionStartEvent(call_index)

            # call the underlying API and handle the events
            completion_events = self.completion_events(
                call_index,
                tools,
                parallel_calls,
                openai_args=dict(
                    messages=self.messages.history,
                    tools=tools.schema if tools else NOT_GIVEN,
                    model=model,
                    **openai_kwargs,
                ),
            )
            completion: ChatCompletion = None
            async for partial_event in completion_events:
                yield partial_event
                match partial_event:
                    case self.CompletionEvent():
                        completion = partial_event.completion

            assert (
                completion is not None
            ), "Full Model Completion was not yielded by the completion_events() generator"

            # select the message if there are multiple choices
            choice_index = 0
            message = completion.choices[choice_index].message
            await self.messages.append(message)
            yield self.FullMessageEvent(message, choice_index=choice_index)

            if message.content is not None:
                # handle text message content
                yield self.TextMessageEvent(message.content)

            if tools and message.tool_calls:
                # handle tool calls
                async for tool_event in self.tool_events(
                    message.tool_calls, tools, parallel_calls
                ):
                    yield tool_event
            else:
                # IMPORTANT no more tool calls, we're done, return the final response
                yield self.ResponseEndEvent(message.content)
                return

        yield self.MaxCallsExceededEvent(max_function_calls)

    async def completion_events(
        self,
        call_index: int,
        tools: Tools | None,
        parallel_calls: bool,
        openai_args: dict,
    ) -> AsyncIterator[Assistant.Event]:
        """
        Model API call to generate the response from the prompt, always yielding the full completion object at the end.
        This implementation does optional streaming generation, yielding partial completions as they come in.
        You should override this to customize the stream handling, also defining your own events to yield.
        """

        # ========== simple non-streaming case ========== #
        if not openai_args.get("stream", False):
            # call the underlying API
            completion = await openai_chat(**openai_args)
            yield self.CompletionEvent(completion, call_index)
            return

        # ========== streaming case ========== #
        # call the underlying API
        completion_stream = await openai_chat(**openai_args)

        # yield events during streaming
        async for chunk, partial in accumulate_partial(completion_stream):
            yield self.PartialCompletionEvent(chunk, partial, call_index)

            # select the message if there are multiple choices
            choice_index = 0
            delta = chunk.choices[choice_index].delta
            message = partial.choices[choice_index].message
            yield self.PartialMessageEvent(message, choice_index=choice_index)

            # when text content was updated
            if delta.content:
                # handle text message content
                yield self.PartialTextMessageEvent(message.content)

            # when tool calls was updated
            if tools and delta.tool_calls:
                # handle tool calls
                async for tool_event in self.partial_tool_events(
                    message.tool_calls, tools, parallel_calls
                ):
                    yield tool_event

        # The "partial" completion should be fully complete by now
        yield self.CompletionEvent(completion=partial, call_index=call_index)

    async def tool_events(
        self, tool_calls: list[ToolCall], tools: Tools, parallel_calls: bool
    ) -> AsyncIterator[Assistant.Event]:
        """
        Generate events from a list of tool calls, yielding each tool call, executing them, and yielding the result. You should override this to customize the tool call handling, also defining your own events to yield.
        """
        yield self.ToolCallsEvent(tool_calls)

        # awaitables for each tool call
        calls = [
            atuple(i, call, call_requested_function(call.function, tools.lookup))
            for i, call in enumerate(tool_calls)
        ]
        # handle each call results as they come in (or in order)
        calls = asyncio.as_completed(calls) if parallel_calls else calls
        for completed in calls:
            i, call, result = await completed
            # stringify the result
            result = str(result)
            await self.messages.append(msg(tool=result, tool_call_id=call.id))
            yield self.ToolResultEvent(result, call, i)

    async def partial_tool_events(
        self, tool_calls: list[ToolCall], tools: Tools, parallel_calls: bool
    ) -> AsyncIterator[Assistant.Event]:
        """
        Generate events from a list of tool calls, yielding each tool call, executing them, and yielding the result. You should override this to customize the tool call handling, also defining your own events to yield.
        """
        yield self.PartialToolCallsEvent(tool_calls)

        # TODO: call function preview functions


class GPT(ChatGPT):
    """
    GPT with no memory, i.e. the chat history resets after each prompt
    """

    async def __call__(
        self,
        prompt: str,
        tools: Optional[Tools] = None,
        model: Optional[str] = None,
        max_function_calls: int = ChatGPT.MAX_FUNCTION_CALLS,
        event_logger: Optional[Callable] = None,
        **openai_kwargs,
    ) -> str:
        """
        Prompt the assistant, returning its final response.

        Args:
            prompt: the user's prompt
            tools: override the default tools
            model: override the default model
            max_function_calls: maximum number of function calls
            **openai_kwargs: additional arguments to pass to the OpenAI API

        Returns:
            the assistant's final response
        """
        async for event in self.response_events(
            prompt, tools, model, max_function_calls, **openai_kwargs
        ):
            if event_logger:
                event_logger(format_event(event))

            match event:
                case self.ResponseEndEvent():
                    await self.messages.reset()
                    return event.content
