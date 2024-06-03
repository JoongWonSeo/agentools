import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from pydantic import BaseModel

from ..api import ToolCall, Model
from ..tools import Tools, function_tool, call_requested_function
from .message import MessageHistory, msg
from .event import Event, ToolCallsEvent, ToolResultEvent, MaxCallsExceededEvent
from .core import Assistant
from .utils import atuple, format_event


@dataclass
class StructCreatedEvent(Event):
    result: BaseModel


@dataclass
class StructFailedEvent(Event):
    result: str


class StructAssistant(Assistant):
    def __init__(
        self, struct: BaseModel, model: Model = Model.default(), tool_name: str = None
    ):
        @function_tool(
            name=tool_name,
            require_doc=False,
            json_schema=struct.model_json_schema(),
        )
        async def create(**kwargs):
            return struct(**kwargs)

        # TODO: define preview function for struct

        super().__init__(MessageHistory(), tools=create, model=model)

        self.struct = struct

    async def __call__(
        self,
        prompt: str,
        max_attempts: int = 5,
        model: Model | None = None,
        event_logger: Optional[callable] = None,
        **openai_kwargs,
    ) -> BaseModel:
        await self.messages.reset()

        async for event in self.response_events(
            prompt,
            model=model,
            max_function_calls=max_attempts,
            tool_choice={
                "type": "function",
                "function": {"name": self.default_tools.name},
            },
            **openai_kwargs,
        ):
            if event_logger:
                event_logger(format_event(event))

            match event:
                case StructCreatedEvent():
                    # print(f"[Struct]: {event.result}", flush=True)
                    return event.result

                case StructFailedEvent():
                    # print(f"[Struct Failed]: {event.result}", flush=True)
                    pass

                case MaxCallsExceededEvent():
                    raise Exception("Max attempts exceeded")

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

            # if the result successfully executed, yield it
            if type(result) is self.struct:
                yield StructCreatedEvent(result)
                result = "Success"
            else:
                yield StructFailedEvent(result)
                result = str(result)

            await self.messages.append(msg(tool=result, tool_call_id=call.id))
            yield ToolResultEvent(result, call, i)
