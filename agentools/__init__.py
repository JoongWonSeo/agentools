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
    # contents
    "content",
    "Content",
    "RefusalContent",
    "TextContent",
    "ImageContent",
    "ImageURL",
    "InputAudioContent",
    "InputAudio",
    "ensure_content",
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
    "FunctionTool",
    "Toolkit",
    "ToolList",
    "call_requested_function",
    "call_function_preview",
    "function_tool",
    "streaming_function_tool",
    "fail_with_message",
]

from .api import (
    GLOBAL_RECORDINGS,
    AsyncGeneratorRecorder,
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Recordings,
    ToolCall,
    accumulate_partial,
    openai_chat,
    set_mock_initial_delay,
    set_mock_streaming_delay,
)
from .assistants import GPT, Assistant, ChatGPT, StructGPT
from .messages import (
    AssistantMessage,
    Content,
    DeveloperMessage,
    ImageContent,
    ImageURL,
    InputAudio,
    InputAudioContent,
    Message,
    MessageHistory,
    RefusalContent,
    SimpleHistory,
    SystemMessage,
    TextContent,
    ToolMessage,
    UserMessage,
    content,
    ensure_content,
    format_contents,
    msg,
)
from .retrieval import (
    EmbeddableData,
    EmbeddableDataCollection,
    EmbeddableField,
    EmbeddingModel,
    MockEmbedding,
    OpenAIEmbedding,
)
from .tools import (
    FunctionTool,
    Toolkit,
    ToolList,
    Tools,
    call_function_preview,
    call_requested_function,
    fail_with_message,
    function_tool,
    streaming_function_tool,
)
