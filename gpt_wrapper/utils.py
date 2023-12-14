import time
import asyncio

import tiktoken

# API return values
from openai.types.chat.chat_completion import (
    ChatCompletion, # Overall Completion, has id, stats, choices
    Choice, # completion.choice[0], has finish_reason, index, message
    ChatCompletionMessage, # completion.choice[0].message, has role, content, tool_calls
)

tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


def mock_response(message: str):
    # TODO: support streaming
    return ChatCompletion(
            id='mock',
            choices=[Choice(
                finish_reason='stop',
                index=0,
                message=ChatCompletionMessage(
                    role='assistant',
                    content=message
                )
            )],
            created=int(time.time()),
            model='mock',
            object='chat.completion',
        )

async def atuple(*vals):
    '''A generalized "tuple" wrapper which will await any coroutine values and simply pass normal values through'''
    return tuple([await v if asyncio.iscoroutine(v) else v for v in vals])
