import asyncio
from functools import wraps
from copy import deepcopy
from typing import Callable

from pydantic import BaseModel

from ..types import JSONSchema
from .utils import (
    ValidationError,
    awaitable,
    pydantic_from_doc,
    validator_from_schema,
    validator_from_pydantic,
    schema_to_openai_func,
)


def function_tool(
    function: Callable = None,
    *,
    name: str | None = None,
    require_doc: bool = True,
    schema: JSONSchema | type[BaseModel] | None = None,
    in_thread: bool | None = None,
    include_call_id: bool = False,
):
    """
    Simple decorator that:
    - Marks a function as a tool and enables it: `func.tool_enabled = True`
    - Attaches a lookup dict for OpenAI: `func.lookup = {func.name: func}`
    - Attaches a list of OpenAI tool schema: `func.schema = [{...}]`
    - Attaches a pydantic argument validator: `func.validator`
    - Attaches a validate and call function: `func.validate_and_call(args)`
    - If a name is not specified, use the function name as the tool name

    Args:
        name: The name that the model will see. If not provided, the function name will be used.
        require_doc: Whether to require a docstring to be present for the function.
        schema: A JSON schema/Pydantic model given to model and for validation. If not provided, the docstring will be used.
        in_thread: Whether to run the function in a separate thread, regardless of whether it is async or not.
        include_call_id: Whether to include the call_id in the function arguments.
    """

    def decorator(func):
        func_name = name or func.__name__

        # ========== Create a validator and schema ========== #
        if isinstance(schema, dict):
            # take the given json schema
            schema_copy = deepcopy(schema)
            arg_validator = validator_from_schema(
                schema_copy,
                name=func_name,
                override_with_doc_from=func if require_doc else None,
            )
            arg_schema = [schema_to_openai_func(schema_copy)]
        elif schema is not None and issubclass(schema, BaseModel):
            # take the given pydantic model
            arg_validator = validator_from_pydantic(schema)
            arg_schema = [schema_to_openai_func(schema)]
            # override the pydantic model title with the function name
            arg_schema[0]["function"]["name"] = func_name  # fixme
        else:
            # parse the docstring and create a pydantic model as validator
            model = pydantic_from_doc(
                func,
                name=func_name,
                require_doc=require_doc,
            )
            arg_validator = validator_from_pydantic(model)
            arg_schema = [schema_to_openai_func(model)]

        # ========== Attach the tool attributes ========== #
        func.name = func_name
        func.tool_enabled = True
        func.schema = arg_schema
        func.validator = arg_validator
        func.validate_and_call = _create_validate_and_call(func)
        func.lookup = {func.name: func.validate_and_call}
        func.validate_and_call.in_thread = (
            not asyncio.iscoroutinefunction(func) if in_thread is None else in_thread
        )
        func.validate_and_call.include_call_id = include_call_id

        # ========= Attach subdecorators ========= #
        # slot for potential preview functions, registered by subdecorators
        func.lookup_preview = {}  # dict of tool name to preview function
        func.preview = _create_preview_decorator(func)

        return func

    if function:  # user did `@function_tool`, i.e. we were used directly as a decorator
        return decorator(function)
    else:  # user did `@function_tool()` or `@function_tool(name='foo')`
        return decorator


def _create_validate_and_call(func):
    """
    A function factory that creates a validate_and_call function for the given function."""

    def validate_and_call(args: dict) -> str:
        """
        Given a dictionary of arguments, validate them and call the underlying function, which we are currently decorating.
        Also dynamically become async if the underlying function is async.
        """
        try:
            args_without_self = {k: v for k, v in args.items() if k != "self"}
            func.validator(**args_without_self)
        except ValidationError as e:
            # if the underlying function is async make sure to return a coroutine
            if asyncio.iscoroutinefunction(func):
                return awaitable(f"Invalid Argument: {e}")
            else:
                return f"Invalid Argument: {e}"
        return func(**args)

    return validate_and_call


def _create_preview_decorator(orig_functool):
    """
    A decorator factory that creates a preview decorator for the given function.
    This enables the following syntax:
    ```
    @function_tool
    def my_function(...):
        ...

    @my_function.preview
    def my_function_preview(...):
        ...
    ```
    """

    def decorator(preview_func):
        """
        A decorator that registers a preview function for the given function.
        This function will be available in the `lookup_preview` dict.
        """

        @wraps(preview_func)
        def wrapper(kwargs: dict[str, any]):
            return preview_func(**kwargs)

        orig_functool.lookup_preview[orig_functool.name] = wrapper

        # inherit the in_thread setting from the original function
        # TODO: make this controllable by the user
        wrapper.in_thread = orig_functool.validate_and_call.in_thread
        wrapper.include_call_id = orig_functool.validate_and_call.include_call_id
        return wrapper

    return decorator
