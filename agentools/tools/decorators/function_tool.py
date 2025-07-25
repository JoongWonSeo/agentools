import asyncio
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Coroutine

from pydantic import BaseModel

from ..types import JSONSchema
from .utils import (
    ValidationError,
    awaitable,
    docstring_description,
    pydantic_from_doc,
    schema_to_openai_func,
    set_description,
    validator_from_schema,
)


def function_tool(
    function: Callable | None = None,
    *,
    name: str | None = None,
    require_doc: bool = True,
    schema: type[BaseModel] | JSONSchema | None = None,
    in_thread: bool | None = None,
    include_call_id: bool = False,
    strict: bool | None = None,
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
        require_doc: Whether to use the docstring as schema for the function.
        schema: A Pydantic model / JSON schema given to model and for validation. If not provided, the docstring will be used.
        in_thread: Whether to run the function in a separate thread, regardless of whether it is async or not.
        include_call_id: Whether to include the call_id in the function arguments.
        strict: Whether the final schema should have strict mode enabled. None means use the default value of the schema.
    """

    def decorator(func):
        nonlocal schema

        func_name = name or func.__name__

        # ========== Create a validator and schema ========== #
        if schema is None:
            # docstring -> Pydantic model
            model = pydantic_from_doc(
                func,
                name=func_name,
                require_doc=require_doc,
            )
        elif isinstance(schema, type) and issubclass(schema, BaseModel):
            # Pydantic model is provided
            model = schema
        else:
            # JSON schema is provided
            model = None

        description = d if require_doc and (d := docstring_description(func)) else None

        if model:
            # Pydantic model -> OpenAI schema, Validator
            arg_validator = validator_from_schema(model.model_json_schema())

            arg_schema = [schema_to_openai_func(model, description=description)]
        elif isinstance(schema, dict):
            # JSON schema -> OpenAI schema, Validator
            schema_copy = deepcopy(schema)
            if require_doc:
                set_description(schema_copy, func, override=False)
            arg_validator = validator_from_schema(
                schema_copy,
                name=func_name,
            )
            arg_schema = [
                schema_to_openai_func(
                    schema_copy, nested=False, description=description
                )
            ]
        else:
            raise ValueError(
                "No schema could be created, please provide either a Pydantic model, a JSON schema, or a docstring with type hints."
            )

        # override the pydantic model / schema title with the function name
        arg_schema[0]["function"]["name"] = func_name

        # set strict mode
        if strict is not None:
            arg_schema[0]["function"]["strict"] = strict

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

    def validate_and_call(args: dict) -> str | Coroutine[Any, Any, str]:
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
        def wrapper(kwargs: dict[str, Any]):
            return preview_func(**kwargs)

        orig_functool.lookup_preview[orig_functool.name] = wrapper

        # inherit the in_thread setting from the original function
        # TODO: make this controllable by the user
        wrapper.in_thread = orig_functool.validate_and_call.in_thread  # type: ignore
        wrapper.include_call_id = orig_functool.validate_and_call.include_call_id  # type: ignore
        return wrapper

    return decorator
