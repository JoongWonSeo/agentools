import asyncio
import json
from typing import Callable, AsyncIterator

from json_autocomplete import json_autocomplete

from ..api.execute import (
    execute_api,
    accumulate_partial,
    ChatCompletion,
    ToolCall,
    Model,
)
from ..tools import Tools, call_requested_function, call_function_preview
from .message import MessageHistory, msg
from .utils import atuple, format_event
from .event import (
    Event,
    ToolResultEvent,
    PartialToolResultEvent,
    PartialFunctionToolCallEvent,
    PartialToolCallsEvent,
    ToolCallsEvent,
    CompletionEvent,
    PartialCompletionEvent,
    PartialMessageEvent,
    PartialTextMessageEvent,
    ResponseEndEvent,
    ResponseStartEvent,
    CompletionStartEvent,
    MaxCallsExceededEvent,
    FullMessageEvent,
    TextMessageEvent,
)


class Assistant:
    """ChatGPT with default model and toolkit"""

    MAX_FUNCTION_CALLS = 100

    def __init__(
        self,
        messages: MessageHistory | None = None,
        tools: Tools | None = None,
        model: Model | None = None,
        forget: bool = False,
    ):
        self.default_model = model or Model.default()
        self.default_tools = tools

        # Replacement for GPT (the class without memory)
        self.forget = forget

        self.messages = messages or MessageHistory()

    # ========== Event Handlers ========== #
    async def __call__(
        self,
        prompt: str,
        tools: Tools | None = None,
        model: Model | None = None,
        max_function_calls: int = MAX_FUNCTION_CALLS,
        event_logger: Callable | None = None,
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
                case ResponseEndEvent():
                    if self.forget:
                        await self.messages.reset()
                    return event.content

        async for event in self.response_events(
            prompt, tools, model, max_function_calls, **openai_kwargs
        ):
            if event_logger:
                event_logger(format_event(event))

            match event:
                case ResponseEndEvent():
                    return event.content

    # ========== Event Generators ========== #
    async def response_events(
        self,
        prompt: str,
        tools: Tools | None = None,
        model: Model | None = None,
        max_function_calls: int = MAX_FUNCTION_CALLS,
        parallel_calls=True,
        **openai_kwargs,
    ) -> AsyncIterator[Event]:
        """
        Generate events from a single user prompt, yielding each message and tool call and everything else as it is received.
        This is the overall generalized form, and you would likely only override the smaller *_events() generators to customize each aspect instead.
        """
        model = model or self.default_model
        tools = tools or self.default_tools

        yield ResponseStartEvent(
            prompt, tools, model.model_name, max_function_calls, openai_kwargs
        )
        await self.messages.append(msg(user=prompt))

        # prompt the assistant and let it use the tools
        for call_index in range(max_function_calls):
            yield CompletionStartEvent(call_index)

            # call the underlying API and handle the events
            completion_events = self.completion_events(
                call_index,
                tools,
                model,
                parallel_calls,
                openai_args=dict(
                    messages=self.messages.history,
                    tools=tools.schema if tools else None,
                    **openai_kwargs,
                ),
            )
            completion: ChatCompletion = None
            async for partial_event in completion_events:
                yield partial_event
                match partial_event:
                    case CompletionEvent():
                        completion = partial_event.completion

            assert (
                completion is not None
            ), "Full Model Completion was not yielded by the completion_events() generator"

            # select the message if there are multiple choices
            choice_index = 0
            message = completion.choices[choice_index].message
            await self.messages.append(message)
            yield FullMessageEvent(message, choice_index=choice_index)

            if message.content is not None:
                # handle text message content
                yield TextMessageEvent(message.content)

            if tools and message.tool_calls:
                # handle tool calls
                async for tool_event in self.tool_events(
                    message.tool_calls, tools, parallel_calls
                ):
                    yield tool_event
            else:
                # IMPORTANT no more tool calls, we're done, return the final response
                yield ResponseEndEvent(message.content)
                return

        yield MaxCallsExceededEvent(max_function_calls)

    async def completion_events(
        self,
        call_index: int,
        tools: Tools | None,
        model: Model,
        parallel_calls: bool,
        openai_args: dict,
    ) -> AsyncIterator[Event]:
        """
        Model API call to generate the response from the prompt, always yielding the full completion object at the end.
        This implementation does optional streaming generation, yielding partial completions as they come in.
        You should override this to customize the stream handling, also defining your own events to yield.
        """

        # ========== simple non-streaming case ========== #
        if not openai_args.get("stream", False):
            # call the underlying API
            completion = await execute_api(model, **openai_args)
            yield CompletionEvent(completion, call_index)
            return

        # ========== streaming case ========== #
        # call the underlying API
        completion_stream = await execute_api(model, **openai_args)

        # yield events during streaming
        async for chunk, partial in accumulate_partial(completion_stream):
            yield PartialCompletionEvent(chunk, partial, call_index)

            # select the message if there are multiple choices
            choice_index = 0
            delta = chunk.choices[choice_index].delta
            message = partial.choices[choice_index].message
            yield PartialMessageEvent(message, choice_index=choice_index)

            # when text content was updated
            if delta.content:
                # handle text message content
                yield PartialTextMessageEvent(message.content)

            # when tool calls was updated
            if tools and delta.tool_calls:
                # handle tool calls
                async for tool_event in self.partial_tool_events(
                    message.tool_calls, tools, parallel_calls
                ):
                    yield tool_event

        # The "partial" completion should be fully complete by now
        yield CompletionEvent(completion=partial, call_index=call_index)

    async def tool_events(
        self, tool_calls: list[ToolCall], tools: Tools, parallel_calls: bool
    ) -> AsyncIterator[Event]:
        """
        Generate events from a list of tool calls, yielding each tool call, executing them, and yielding the result. You should override this to customize the tool call handling, also defining your own events to yield.
        """
        yield ToolCallsEvent(tool_calls)

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
            await self.messages.append(msg(tool=result, tool_call_id=call.id))
            yield ToolResultEvent(result, call, i)

    async def partial_tool_events(
        self, tool_calls: list[ToolCall], tools: Tools, parallel_calls: bool
    ) -> AsyncIterator[Event]:
        """
        Generate events from a list of tool preview calls, yielding each partial tool call, executing them, and yielding the result of the preview function. You should override this to customize the tool preview call handling, also defining your own events to yield.
        """
        yield PartialToolCallsEvent(tool_calls)

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

            yield PartialFunctionToolCallEvent(
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
            yield PartialToolResultEvent(result, call, i)
