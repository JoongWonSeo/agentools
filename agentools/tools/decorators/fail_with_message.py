import asyncio
from typing import Callable
from functools import wraps


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
