"""Small version-tolerant helpers for instrumenting native adapter objects."""

from __future__ import annotations

import functools
import inspect
from typing import Any, Mapping

from .tracing import stage_span


def instrument_method(
    target: Any,
    name: str,
    category: str,
    operation: str,
    *,
    backend: str | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> bool:
    """Wrap an instance method if the installed native version exposes it.

    A missing or immutable implementation returns False and is recorded by the
    adapter's tracing manifest instead of making the benchmark fail.
    """
    method = getattr(target, name, None)
    if method is None or not callable(method):
        return False
    if getattr(method, "__track_c_instrumented__", False):
        return True

    if inspect.iscoroutinefunction(method):

        @functools.wraps(method)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            with stage_span(
                category, operation, backend=backend, attributes=attributes
            ):
                return await method(*args, **kwargs)

        wrapper: Any = async_wrapper
    else:

        @functools.wraps(method)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with stage_span(
                category, operation, backend=backend, attributes=attributes
            ):
                return method(*args, **kwargs)

        wrapper = sync_wrapper
    wrapper.__track_c_instrumented__ = True
    try:
        setattr(target, name, wrapper)
    except (AttributeError, TypeError, ValueError):
        return False
    return True
