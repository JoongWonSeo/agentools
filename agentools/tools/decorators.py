import asyncio
from typing import Optional, Callable
from functools import wraps
from copy import deepcopy

from .utils import (
    ValidationError,
    validator_from_doc,
    validator_from_schema,
    schema_to_openai_func,
)


def function_tool(
    function=None,
    *,
    name: Optional[str] = None,
    require_doc: bool = True,
    json_schema: Optional[dict] = None,
    in_thread: Optional[bool] = None,
):
    """
    Simple decorator that:
    - Marks a function as a tool and enables it: `func.tool_enabled = True`
    - Attaches a lookup dict for OpenAI: `func.lookup = {func.name: func}`
    - Attaches a list of OpenAI tool schema: `func.schema = [{...}]`
    - Attaches a pydantic argument validator: `func.validator`
    - Attaches a validate and call function: `func.validate_and_call(args)`
    - If a name is not specified, use the function name as the tool name
    """

    def decorator(func):
        def validate_and_call(args: dict) -> str:
            try:
                args_without_self = {k: v for k, v in args.items() if k != "self"}
                func.validator(**args_without_self)
            except ValidationError as e:
                # if the underlying function is async make sure to return a coroutine
                if asyncio.iscoroutinefunction(func):

                    async def async_error(e):
                        return f"Invalid Argument: {e}"

                    return async_error(e)
                else:
                    return f"Invalid Argument: {e}"
            return func(**args)

        func.name = name or func.__name__
        func.tool_enabled = True

        if json_schema:
            # take the given json schema
            schema_copy = deepcopy(json_schema)
            func.validator = validator_from_schema(
                schema_copy,
                name=func.name,
                override_with_doc_from=func if require_doc else None,
            )
            func.schema = [schema_to_openai_func(schema_copy)]
        else:
            # parse the docstring and create a pydantic model as validator
            model, func.validator = validator_from_doc(
                func, name=func.name, require_doc=require_doc
            )
            func.schema = [schema_to_openai_func(model)]

        func.validate_and_call = validate_and_call
        func.lookup = {func.name: func.validate_and_call}
        func.validate_and_call.in_thread = (
            not asyncio.iscoroutinefunction(func) if in_thread is None else in_thread
        )
        return func

    if function:  # user did `@function_tool`, i.e. we were used directly as a decorator
        return decorator(function)
    else:  # user did `@function_tool()` or `@function_tool(name='foo')`
        return decorator


def fail_with_message(
    message="[Internal Error] ", include_exception=True, logger: Callable = print
):
    """A decorator that catches exceptions from synchronous and asynchronous functions and returns the given message instead. Useful for agent tools."""

    def log_exception(func, args, kwargs, e):
        if logger:
            logger(
                f"""
Tool call {func.__name__}({
    ', '.join(list(map(repr, args)) + [f'{k}={repr(v)}' for k,v in kwargs.items()])
}) failed: {e}
""".strip()
            )
        return (message + (f": {str(e)}" if include_exception else "")).strip()

    def decorator(func):
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    return log_exception(func, args, kwargs, e)

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    return log_exception(func, args, kwargs, e)

            return sync_wrapper

    return decorator
