import asyncio
import json
from typing import Callable, AsyncIterator, Iterable

from json_autocomplete import json_autocomplete

from ..api.openai import (
    openai_chat,
    accumulate_partial,
    ChatCompletion,
    ToolCall,
)
from ..tools import Tools, ToolList, call_requested_function, call_function_preview
from ..messages import MessageHistory, SimpleHistory, msg, Message, Content
from .utils import atuple, format_event
from .core import Assistant


class ChatGPT(Assistant):
    """ChatGPT with default model and toolkit"""

    DEFAULT_MODEL = "gpt-4o-mini"
    MAX_FUNCTION_CALLS = 10

    def __init__(
        self,
        messages: MessageHistory | None = None,
        tools: Tools | list[Tools] | None = None,
        model: str = DEFAULT_MODEL,
    ):
        self.default_model = model
        self.default_tools = tools

        self.messages = messages or SimpleHistory()

    # ========== Event Handlers ========== #
    async def __call__(
        self,
        prompt: str | Iterable[Content] | Message | None,
        messages: MessageHistory | None = None,
        tools: Tools | list[Tools] | None = None,
        model: str | None = None,
        max_function_calls: int = MAX_FUNCTION_CALLS,
        event_logger: Callable | None = None,
        **openai_kwargs,
    ) -> str:
        """
        Prompt the assistant, returning its final response.

        Args:
            prompt: The user's prompt
            messages: The message history to use, defaults to self.messages
            tools: Override the default tools
            model: Override the default model
            max_function_calls: Maximum number of function calls
            **openai_kwargs: Additional arguments to pass to the OpenAI API

        Returns:
            The assistant's final response
        """
        async for event in self.new_message_handler(
            self.response_events(
                prompt,
                messages,
                tools,
                model,
                max_function_calls,
                **openai_kwargs,
            )
        ):
            if event_logger:
                event_logger(format_event(event))

            match event:
                case self.ResponseEndEvent():
                    return event.content

    async def new_message_handler(
        self, response_events: AsyncIterator[Assistant.Event]
    ):
        """
        Takes care of automatically appending new messages to the message history
        """
        async for e in response_events:
            # handle events that require appending to the message history
            match e:
                # full assistant response message arrived
                case self.FullMessageEvent():
                    await self.messages.append(e.message)
                # full tool result message arrived
                case self.ToolResultEvent():
                    await self.messages.append(
                        msg(tool=e.result, tool_call_id=e.tool_call.id)
                    )
            # transparently yield the event
            yield e

    # ========== Event Generators ========== #
    async def response_events(
        self,
        prompt: str | Iterable[Content] | Message | None,
        messages: MessageHistory | None = None,
        tools: Tools | list[Tools] | None = None,
        model: str | None = None,
        max_function_calls: int = MAX_FUNCTION_CALLS,
        parallel_calls=True,
        **openai_kwargs,
    ) -> AsyncIterator[Assistant.Event]:
        """
        Generate events from a single user prompt, yielding each message and tool call and everything else as it is received.
        This is the overall generalized form, and you would likely only override the smaller *_events() generators to customize each aspect instead.

        Args:
            prompt: Either the user prompt as string/list of content, or a dict Message
            messages: The message history to use, defaults to self.messages
            tools: Override the default tools
            model: Override the default model
            max_function_calls: Maximum number of function calls
            parallel_calls: Whether to run tool calls in parallel
            **openai_kwargs: Additional arguments to pass to the OpenAI API
        """
        messages: MessageHistory = messages or self.messages
        model = model or self.default_model
        tools = tools or self.default_tools
        tools = ToolList(*tools) if isinstance(tools, list) else tools

        yield self.ResponseStartEvent(
            prompt, tools, model, max_function_calls, openai_kwargs
        )

        # add the prompt to the message history
        if prompt is not None:
            # non-dict prompt is assumed to be a user message
            m = prompt if isinstance(prompt, dict) else msg(user=prompt)
            await messages.append(m)

        # prompt the assistant and let it use the tools
        for call_index in range(max_function_calls):
            yield self.CompletionStartEvent(call_index)

            # call the underlying API and handle the events
            completion_events = self.completion_events(
                call_index,
                tools,
                parallel_calls,
                openai_args=dict(
                    messages=messages.history,
                    tools=tools.schema if tools else None,
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
        tools: Tools | list[Tools] | None,
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
        self,
        tool_calls: list[ToolCall],
        tools: Tools | list[Tools],
        parallel_calls: bool,
    ) -> AsyncIterator[Assistant.Event]:
        """
        Generate events from a list of tool calls, yielding each tool call, executing them, and yielding the result. You should override this to customize the tool call handling, also defining your own events to yield.
        """
        yield self.ToolCallsEvent(tool_calls)

        tools = ToolList(*tools) if isinstance(tools, list) else tools
        lookup = tools.lookup

        # awaitables for each tool call
        calls = []
        for i, call in enumerate(tool_calls):
            call_id = call.id
            name = call.function.name
            args = call.function.arguments
            task = atuple(i, call, call_requested_function(name, args, lookup, call_id))
            calls.append(task)

        # handle each call results as they come in (or in order)
        calls = asyncio.as_completed(calls) if parallel_calls else calls
        for completed in calls:
            i, call, result = await completed
            result = str(result)  # stringify the result
            yield self.ToolResultEvent(result, call, i)

    async def partial_tool_events(
        self, tool_calls: list[ToolCall], tools: Tools, parallel_calls: bool
    ) -> AsyncIterator[Assistant.Event]:
        """
        Generate events from a list of tool preview calls, yielding each partial tool call, executing them, and yielding the result of the preview function. You should override this to customize the tool preview call handling, also defining your own events to yield.
        """
        yield self.PartialToolCallsEvent(tool_calls)

        tools = ToolList(*tools) if isinstance(tools, list) else tools
        lookup_preview = tools.lookup_preview

        if not lookup_preview:
            return

        # create preview function tasks
        calls = []
        for i, call in enumerate(tool_calls):
            # TODO: in theory, since this is streaming, we can do the following optimizations for all i < last_i, since they are already completed:
            # - we can skip the preview call
            # - we could in theory already start the final call task, and gather in the final call later
            # easier if chunk/delta is also passed, since ca__id is always included in each chunk, i.e. use that to detect which call was changed.
            # to help coordinate between partial and final, we can have a dict of call_id to task, so that partial may already kickstart the final task
            call_id = call.id
            func = call.function.name
            partial = call.function.arguments

            if func not in lookup_preview or not partial.strip():
                continue
            try:
                autocompleted = json_autocomplete(call.function.arguments)
                args = json.loads(autocompleted)
            except Exception:
                continue

            yield self.PartialFunctionToolCallEvent(
                function_name=func,
                partial=partial,
                autocompleted=autocompleted,
                arguments=args,
                index=i,
            )

            task = atuple(
                i, call, call_function_preview(func, args, lookup_preview, call_id)
            )
            calls.append(task)

        # handle each call results as they come in (or in order)
        calls = asyncio.as_completed(calls) if parallel_calls else calls
        for completed in calls:
            i, call, result = await completed
            yield self.PartialToolResultEvent(result, call, i)


class GPT(ChatGPT):
    """
    GPT with no memory, i.e. the chat history resets after each prompt
    """

    async def __call__(
        self,
        prompt: str | Iterable[Content] | Message | None,
        messages: MessageHistory | None = None,
        tools: Tools | list[Tools] | None = None,
        model: str | None = None,
        max_function_calls: int = ChatGPT.MAX_FUNCTION_CALLS,
        event_logger: Callable | None = None,
        **openai_kwargs,
    ) -> str:
        """
        Prompt the assistant, returning its final response.

        Args:
            prompt: The user's prompt
            messages: The message history to use, defaults to self.messages
            tools: Override the default tools
            model: Override the default model
            max_function_calls: Maximum number of function calls
            **openai_kwargs: Additional arguments to pass to the OpenAI API

        Returns:
            The assistant's final response
        """
        messages: MessageHistory = messages or self.messages
        async for event in self.response_events(
            prompt, messages, tools, model, max_function_calls, **openai_kwargs
        ):
            if event_logger:
                event_logger(format_event(event))

            match event:
                case self.ResponseEndEvent():
                    await messages.reset()
                    return event.content
