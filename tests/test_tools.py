import pytest

from agentools.tools.decorators.function_tool import function_tool
from agentools.tools.core import call_requested_function


@function_tool(require_doc=False)
def add(a: int, b: int) -> str:
    """Add two numbers"""
    return str(a + b)


@pytest.mark.asyncio
async def test_call_requested_function_success():
    result = await call_requested_function(
        "add", '{"a":1,"b":2}', {"add": add.validate_and_call}
    )
    assert result == "3"


@pytest.mark.asyncio
async def test_call_requested_function_invalid():
    result = await call_requested_function(
        "add", '{"a":1}', {"add": add.validate_and_call}
    )
    assert "Invalid Argument" in result
