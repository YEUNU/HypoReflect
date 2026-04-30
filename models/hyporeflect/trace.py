from typing import Any, Optional


def make_trace_event(
    *,
    step: str,
    input: Any = None,
    output: Any = None,
    error: Optional[str] = None,
    duration_ms: Optional[float] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {"step": step}
    if input is not None:
        event["input"] = input
    if output is not None:
        event["output"] = output
    if error is not None:
        event["error"] = error
    if duration_ms is not None:
        event["duration_ms"] = round(float(duration_ms), 3)
    if extra:
        event.update(extra)
    return event


def append_trace(
    trace: list[dict[str, Any]],
    *,
    step: str,
    input: Any = None,
    output: Any = None,
    error: Optional[str] = None,
    duration_ms: Optional[float] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    trace.append(
        make_trace_event(
            step=step,
            input=input,
            output=output,
            error=error,
            duration_ms=duration_ms,
            extra=extra,
        )
    )
