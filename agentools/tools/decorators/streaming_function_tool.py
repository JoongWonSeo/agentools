import asyncio
from typing import Callable

from .function_tool import function_tool


class AsyncStream:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.closed = False

    async def send(self, data, is_final=False):
        """
        Send data through the tunnel, which the receiver can read using `async for`.
        """
        await self.queue.put(data)
        if is_final:
            self.closed = True

    def __aiter__(self):
        """
        Returns the iterator object itself.
        """
        return self

    async def __anext__(self):
        """
        Returns the next item from the queue.
        """
        if self.closed and self.queue.empty():
            raise StopAsyncIteration
        data = await self.queue.get()
        return data


def streaming_function_tool(
    function: Callable = None,
    *,
    name: str | None = None,
    require_doc: bool = True,
    json_schema: dict,
    in_thread: bool | None = None,
    include_call_id: bool = False,
):
    """


    Args:
        name: The name that the model will see. If not provided, the function name will be used.
        require_doc: Whether to require a docstring to be present for the function.
        json_schema: A JSON schema given to model and for validation. If not provided, the docstring will be used.
        in_thread: Whether to run the function in a separate thread, regardless of whether it is async or not.
        include_call_id: Whether to include the call_id in the function arguments.
    """

    def decorator(func):
        func_name = name or func.__name__

        @function_tool(
            name=func_name,
            require_doc=require_doc,
            json_schema=json_schema,
            in_thread=in_thread,
            include_call_id=True,
        )
        async def function_final(call_id, **args):
            task, stream = func.tasks[call_id]

            if include_call_id:
                args = args | {"call_id": call_id}

            await stream.send(args, is_final=True)

            await task  # wait for the task to finish
            del func.tasks[call_id]
            return task.result()

        @function_final.preview
        async def function_preview(call_id, **args):
            if call_id not in func.tasks:
                # function is now starting
                stream = AsyncStream()
                task = asyncio.create_task(func(stream))
                func.tasks[call_id] = (task, stream)
            else:
                task, stream = func.tasks[call_id]

            if include_call_id:
                args = args | {"call_id": call_id}

            await stream.send(args)

        # ========== Attach the tool attributes ========== #
        func.name = function_final.name
        func.tool_enabled = function_final.tool_enabled
        func.schema = function_final.schema
        func.validator = function_final.validator
        func.validate_and_call = function_final.validate_and_call
        func.lookup = {func.name: func.validate_and_call}
        func.lookup_preview = {func.name: function_preview}

        func.tasks = {}  # {call_id: (task, stream)}

        return func

    if function:  # user did `@streaming_function_tool`
        raise ValueError(
            "Please provide the JSON schema for the streaming function tool."
        )
    else:  # user did `@streaming_function_tool()`
        return decorator
