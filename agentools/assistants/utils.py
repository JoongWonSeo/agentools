import asyncio
from dataclasses import dataclass


def format_event(event: dataclass) -> str:
    """Format an event as a string"""
    return f"[{event.__class__.__name__}]: {', '.join([f'{k}={v}' for k, v in event.__dict__.items()])}"
    # return str(event)


async def atuple(*vals):
    """A generalized "tuple" wrapper which will await any coroutine values and simply pass normal values through"""
    return tuple([await v if asyncio.iscoroutine(v) else v for v in vals])
