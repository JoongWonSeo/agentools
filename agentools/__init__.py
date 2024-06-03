"""
.. include:: ../README.md
"""

# __all__ = [
#     "api",
#     "assistants",
#     "messages",
#     "retrieval",
#     "tools",
# ]

# from .api import *  # noqa: F403
# from .assistants import *  # noqa: F403
# from .messages import *  # noqa: F403
# from .retrieval import *  # noqa: F403
# from .tools import *  # noqa: F403

__all__ = [
    # API
    "ChatCompletion",
    "ChatCompletionMessage",
    "ToolCall",
    "ChatCompletionChunk",
    "execute_api",
    "accumulate_partial",
    "set_mock_initial_delay",
    "set_mock_streaming_delay",
    "AsyncGeneratorRecorder",
    "Recordings",
    "GLOBAL_RECORDINGS",
    "Model",
    # assistants
    "Assistant",
    "StructAssistant",
    # messages
    "msg",
    "MessageHistory",
    # retrieval
    "EmbeddableDataCollection",
    "EmbeddableData",
    "EmbeddableField",
    "EmbeddingModel",
    "MockEmbedding",
    "OpenAIEmbedding",
    # tools
    "Tools",
    "Toolkit",
    "ToolList",
    "call_requested_function",
    "call_function_preview",
    "function_tool",
    "streaming_function_tool",
    "fail_with_message",
    "Event",
    "ResponseStartEvent",
    "CompletionStartEvent",
    "CompletionEvent",
    "FullMessageEvent",
    "TextMessageEvent",
    "ToolCallsEvent",
    "ToolResultEvent",
    "ResponseEndEvent",
    "PartialCompletionEvent",
    "PartialMessageEvent",
    "MaxCallsExceededEvent",
    "PartialTextMessageEvent",
    "PartialToolCallsEvent",
    "PartialFunctionToolCallEvent",
    "PartialToolResultEvent",
    "MaxTokensExceededEvent",
    "ModelTimeoutEvent",
    "ToolTimeoutEvent",
]

from .api import (
    ChatCompletion,
    ChatCompletionMessage,
    ToolCall,
    ChatCompletionChunk,
    execute_api,
    accumulate_partial,
    set_mock_initial_delay,
    set_mock_streaming_delay,
    AsyncGeneratorRecorder,
    Recordings,
    GLOBAL_RECORDINGS,
    Model,
)
from .assistant import (
    Assistant,
    StructAssistant,
    msg,
    MessageHistory,
    Event,
    ResponseStartEvent,
    CompletionStartEvent,
    CompletionEvent,
    FullMessageEvent,
    TextMessageEvent,
    ToolCallsEvent,
    ToolResultEvent,
    ResponseEndEvent,
    PartialCompletionEvent,
    PartialMessageEvent,
    MaxCallsExceededEvent,
    PartialTextMessageEvent,
    PartialToolCallsEvent,
    PartialFunctionToolCallEvent,
    PartialToolResultEvent,
    MaxTokensExceededEvent,
    ModelTimeoutEvent,
    ToolTimeoutEvent,
)
from .retrieval import (
    EmbeddableDataCollection,
    EmbeddableData,
    EmbeddableField,
    EmbeddingModel,
    MockEmbedding,
    OpenAIEmbedding,
)
from .tools import (
    Tools,
    Toolkit,
    ToolList,
    call_requested_function,
    call_function_preview,
    function_tool,
    streaming_function_tool,
    fail_with_message,
)
