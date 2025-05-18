from dataclasses import dataclass
import json
from typing import Callable, AsyncIterator, Generic, Iterable, TypeVar, override, Type

from pydantic import BaseModel
from json_autocomplete import json_autocomplete

from ..tools import function_tool, FunctionTool
from ..messages import MessageHistory, Message, Content
from .core import Assistant
from .chatgpt import ChatGPT
from .utils import format_event


S = TypeVar("S", bound=BaseModel)
_S_EVENT = TypeVar("_S_EVENT", bound=BaseModel)


@dataclass
class StructCreatedEvent(Assistant.Event, Generic[_S_EVENT]):
    result: _S_EVENT


@dataclass
class StructFailedEvent(Assistant.Event):
    result: str


class StructGPT(ChatGPT, Generic[S]):
    """Use GPT to create a structure S from a prompt"""

    DEFAULT_MODEL = "gpt-4o-mini"
    DEFAULT_ATTEMPTS = 5

    def __init__(
        self,
        struct: Type[S],
        model: str = DEFAULT_MODEL,
        tool_name: str | None = None,
        on_preview: Callable | None = None,
        messages: MessageHistory | None = None,
    ):
        @function_tool(
            name=tool_name or struct.__name__,
            require_doc=False,
            schema=struct,
        )
        async def create(**kwargs) -> S:
            return struct(**kwargs)

        # wrap the preview function to register it to the tool
        self.on_preview = create.preview(on_preview) if on_preview else None

        super().__init__(messages, tools=create, model=model)  # type: ignore

        self.default_tools: FunctionTool
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

        async for event in self.event_adapter(
            self.response_events(
                prompt,
                model=model,
                max_function_calls=max_attempts,
                tool_choice={
                    "type": "function",
                    "function": {"name": self.default_tools.name},
                },
                stream=True if self.on_preview else False,
                **openai_kwargs,
            )
        ):
            if event_logger:
                event_logger(format_event(event))

            match event:
                case StructCreatedEvent():
                    if isinstance(event.result, self.struct):
                        return event.result  # type: ignore # result is S
                    else:
                        # This case should ideally not be reached if event_adapter is correct
                        raise TypeError(
                            f"StructCreatedEvent payload has unexpected type: {type(event.result)}"
                        )
                case StructFailedEvent():
                    # Continue loop, eventually will hit MaxCallsExceededEvent or end of attempts
                    pass
                case self.MaxCallsExceededEvent():
                    raise Exception("Max attempts exceeded")

        # If loop finishes without returning S or raising for MaxCallsExceededEvent
        raise Exception(
            "Struct creation failed to produce a valid structure after all attempts."
        )

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

        async for event in self.event_adapter(
            self.response_events(
                prompt,
                model=model,
                max_function_calls=max_attempts,
                tool_choice={
                    "type": "function",
                    "function": {"name": self.default_tools.name},
                },
                stream=True,
                **openai_kwargs,
            )
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
                case StructCreatedEvent():
                    return

                case StructFailedEvent():
                    raise Exception("Struct creation failed")

                case self.MaxCallsExceededEvent():
                    raise Exception("Max attempts exceeded")

    # ========== Event Adapters ========== #
    @override
    async def event_adapter(
        self,
        response_events: AsyncIterator[Assistant.Event],
        messages: MessageHistory | None = None,
    ):
        """
        Converts the tool result event to a StructCreatedEvent or StructFailedEvent.
        """

        async for e in super().event_adapter(response_events, messages):
            match e:
                # full tool result message arrived
                case self.ToolResultEvent():
                    # Ensure e.result is not None before checking its type with isinstance
                    if e.result is not None and isinstance(e.result, self.struct):
                        yield StructCreatedEvent[S](result=e.result)
                        # override the result, although it wouldn't matter
                        e.result = "Success"
                    else:
                        yield StructFailedEvent(e.result)

            # transparently yield the event
            yield e
