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
    "openai_chat",
    "accumulate_partial",
    "set_mock_initial_delay",
    "set_mock_streaming_delay",
    "AsyncGeneratorRecorder",
    "Recordings",
    "GLOBAL_RECORDINGS",
    # assistants
    "Assistant",
    "ChatGPT",
    "GPT",
    "StructGPT",
    # messages
    "msg",
    "MessageHistory",
    "SimpleHistory",
    "Message",
    "DeveloperMessage",
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "ToolMessage",
    # contents,
    "content",
    "Content",
    "TextContent",
    "ImageContent",
    "ImageURL",
    "InputAudioContent",
    "InputAudio",
    "format_contents",
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
]

from .api import (
    ChatCompletion,
    ChatCompletionMessage,
    ToolCall,
    ChatCompletionChunk,
    openai_chat,
    accumulate_partial,
    set_mock_initial_delay,
    set_mock_streaming_delay,
    AsyncGeneratorRecorder,
    Recordings,
    GLOBAL_RECORDINGS,
)
from .assistants import Assistant, ChatGPT, GPT, StructGPT
from .messages import (
    msg,
    MessageHistory,
    SimpleHistory,
    Message,
    DeveloperMessage,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    content,
    Content,
    TextContent,
    ImageContent,
    ImageURL,
    InputAudioContent,
    InputAudio,
    format_contents,
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
