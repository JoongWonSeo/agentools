from abc import ABC
import asyncio
from typing import Callable
from functools import wraps
from itertools import chain
import json

from ..api.openai import Function


class Tools(ABC):
    """A (fake) interface for any tool (function/ToolList/Toolkit) that can be used by the assistant."""

    tool_enabled: bool  # whether this tool is enabled
    schema: list[dict]  # list of OpenAI function schemas
    lookup: dict[str, Callable]  # dict of tool name to function implementation


class ToolList(Tools):
    """A simple collection of tools/toolkits"""

    def __init__(self, *tools: Tools, tool_enabled=True):
        self.tools = list(tools)
        self.tool_enabled = tool_enabled

    @property
    def schema(self) -> list[dict]:
        """list of OpenAI function schemas"""
        if not self.tool_enabled:
            return []

        return list(chain(*[t.schema for t in self.tools if t.tool_enabled]))

    @property
    def lookup(self) -> dict[str, Callable]:
        """dict of TOOL NAME to argument-validated function"""
        if not self.tool_enabled:
            return {}

        lookups = [t.lookup for t in self.tools if t.tool_enabled]
        assert len(set(chain(*[lookup.keys() for lookup in lookups]))) == sum(
            [len(lookup) for lookup in lookups]
        ), "Duplicate tool names detected!"
        return {k: v for lookup in lookups for k, v in lookup.items()}


class Toolkit(Tools):
    """
    A base class for a collection of tools and their shared states.
    Simply inherit this class and mark your methods as tools with the `@function_tool` decorator.
    After instantiating your toolkit, you can either:
    - [Code]: Simply use the functions as normal, e.g. `toolkit.my_tool(**args)`
    - [Model]: Use the `toolkit.lookup` dict to call the function by name, e.g. `toolkit.lookup['my_tool'](args)`
    """

    def __init__(self):
        self.tool_enabled = True
        self.registered_tools = (
            {}
        )  # explicitly registered, i.e. dynamically defined tools

    def register_tool(self, tool):
        """Explicitly register a tool if it's not in the class definition"""
        self.registered_tools[tool.name] = tool

    @property
    def schema(self) -> list[dict]:
        """list of OpenAI function schemas"""
        return list(chain(*[tool.schema for tool in self._function_tools.values()]))

    @property
    def lookup(self) -> dict[str, Callable]:
        """dict of TOOL NAME to argument-validated function"""
        return {
            tool.name: self._with_self(tool.validate_and_call)
            for tool in self._function_tools.values()
        }

    @property
    def _function_tools(self) -> dict[str, Callable]:
        """dict of RAW FUNCTION NAME to function"""
        return (
            self.registered_tools
            | {
                attr: getattr(self, attr)
                for attr in dir(type(self))
                if not isinstance(
                    getattr(type(self), attr), property
                )  # ignore properties to prevent infinite recursion
                and getattr(getattr(self, attr), "tool_enabled", False)
            }
            if self.tool_enabled
            else {}
        )

    # util to prevent late-binding of func in a dict comprehension
    def _with_self(self, func: Callable):
        """Make a function which automatically receives self as the first argument"""

        @wraps(func)
        def wrapper(kwargs: dict[str, any]):
            return func({"self": self, **kwargs})

        return wrapper


# ========== Model ========== #
async def call_requested_function(
    call_request: Function, func_lookup: dict[str, Callable]
):
    """
    Call the requested function generated by the model.
    """
    # parse function call
    func_name = call_request.name
    arguments = call_request.arguments

    if func_name not in func_lookup:
        return f"Error: Function {func_name} does not exist."
    try:
        args = json.loads(arguments)
    except Exception as e:
        return f"Error: Failed to parse arguments, make sure your arguments is a valid JSON object: {e}"

    # call function
    try:
        f = func_lookup[func_name]
        if not getattr(f, "in_thread", True):
            return await f(args)
        else:
            return await asyncio.to_thread(f, args)
    except Exception as e:
        return f"Error: {e}"
