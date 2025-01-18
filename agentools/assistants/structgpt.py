import asyncio
from dataclasses import dataclass
import json
from typing import Callable, AsyncIterator, Generic, Iterable, TypeVar

from pydantic import BaseModel
from json_autocomplete import json_autocomplete

from ..api import ToolCall
from ..tools import Tools, function_tool, call_requested_function
from ..messages import MessageHistory, msg, Message, Content
from .core import Assistant
from .chatgpt import ChatGPT
from .utils import atuple, format_event


S = TypeVar("S", bound=BaseModel)


class StructGPT(ChatGPT, Generic[S]):
    """Use GPT to create a structure S from a prompt"""

    DEFAULT_MODEL = "gpt-4o-mini"
    DEFAULT_ATTEMPTS = 5

    @dataclass
    class StructCreatedEvent(Assistant.Event, Generic[S]):
        result: S

    @dataclass
    class StructFailedEvent(Assistant.Event):
        result: str

    def __init__(
        self,
        struct: type[S],
        model: str = DEFAULT_MODEL,
        tool_name: str = None,
        on_preview: Callable | None = None,
        messages: MessageHistory | None = None,
    ):
        @function_tool(
            name=tool_name,
            require_doc=False,
            schema=struct,
        )
        async def create(**kwargs) -> S:
            return struct(**kwargs)

        # wrap the preview function to register it to the tool
        self.on_preview = create.preview(on_preview) if on_preview else None

        super().__init__(messages, tools=create, model=model)

        self.struct = struct

    async def __call__(
        self,
        prompt: str | Iterable[Content] | Message | None,
        max_attempts: int = DEFAULT_ATTEMPTS,
        model: str | None = None,
        event_logger: Callable | None = None,
        **openai_kwargs,
    ) -> S:
        await self.messages.reset()

        async for event in self.response_events(
            prompt,
            model=model,
            max_function_calls=max_attempts,
            tool_choice={
                "type": "function",
                "function": {"name": self.default_tools.name},
            },
            stream=True if self.on_preview else False,
            **openai_kwargs,
        ):
            if event_logger:
                event_logger(format_event(event))

            match event:
                case self.StructCreatedEvent():
                    # print(f"[Struct]: {event.result}", flush=True)
                    return event.result

                case self.StructFailedEvent():
                    # print(f"[Struct Failed]: {event.result}", flush=True)
                    pass

                case self.MaxCallsExceededEvent():
                    raise Exception("Max attempts exceeded")

    async def stream_json(
        self,
        prompt: str | Iterable[Content] | Message | None,
        max_attempts: int = DEFAULT_ATTEMPTS,
        model: str | None = None,
        event_logger: Callable | None = None,
        **openai_kwargs,
    ) -> AsyncIterator[dict]:
        """
        Stream the struct creation process, yielding each partial json result.
        """
        await self.messages.reset()

        async for event in self.response_events(
            prompt,
            model=model,
            max_function_calls=max_attempts,
            tool_choice={
                "type": "function",
                "function": {"name": self.default_tools.name},
            },
            stream=True,
            **openai_kwargs,
        ):
            if event_logger:
                event_logger(format_event(event))

            match event:
                case self.PartialToolCallsEvent():
                    try:
                        autocompleted = json_autocomplete(
                            event.tool_calls[0].function.arguments
                        )
                        args = json.loads(autocompleted)
                        if args:  # don't yield null/empty dict/empty list yet
                            yield args
                    except Exception:
                        continue

                # This handler is not necessary, because the last event will always be the full json
                case self.StructCreatedEvent():
                    return

                case self.StructFailedEvent():
                    raise Exception("Struct creation failed")

                case self.MaxCallsExceededEvent():
                    raise Exception("Max attempts exceeded")

    async def tool_events(
        self, tool_calls: list[ToolCall], tools: Tools, parallel_calls: bool
    ) -> AsyncIterator[Assistant.Event]:
        """
        Generate events from a list of tool calls, yielding each tool call, executing them, and yielding the result. You should override this to customize the tool call handling, also defining your own events to yield.
        """
        yield self.ToolCallsEvent(tool_calls)

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

            # if the result successfully executed, yield it
            if type(result) is self.struct:
                yield self.StructCreatedEvent(result)
                result = "Success"
            else:
                yield self.StructFailedEvent(result)
                result = str(result)

            await self.messages.append(msg(tool=result, tool_call_id=call.id))
            yield self.ToolResultEvent(result, call, i)
