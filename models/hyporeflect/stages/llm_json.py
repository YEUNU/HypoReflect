import logging
import json
from typing import Any, Callable, Optional


ValidationFn = Callable[[dict[str, Any]], tuple[bool, str]]
RetryMessageFn = Callable[[dict[str, Any], str], str]


def ensure_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def compact_json(data: Any, max_chars: int = 1800) -> str:
    try:
        text = json.dumps(data, ensure_ascii=True)
    except TypeError:
        text = str(data)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


async def generate_json_with_retries(
    llm: Any,
    base_messages: list[dict[str, str]],
    validate: ValidationFn,
    build_retry_message: Optional[RetryMessageFn],
    *,
    max_attempts: int = 3,
    temperature: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
    warning_prefix: str = "generate_json failed",
    model: str = "",
) -> tuple[dict[str, Any], bool, list[dict[str, Any]]]:
    messages = list(base_messages)
    attempts: list[dict[str, Any]] = []
    last_data: dict[str, Any] = {}

    for attempt in range(max_attempts):
        try:
            _kwargs: dict = dict(
                temperature=temperature,
                json_debug_label=warning_prefix,
                apply_default_sampling=False,
            )
            if model:
                _kwargs["model"] = model
            raw = await llm.generate_json(messages, **_kwargs)
        except Exception as e:
            if logger is not None:
                logger.warning("%s (attempt %d/%d): %s", warning_prefix, attempt + 1, max_attempts, e)
            raw = {}

        data = ensure_json_object(raw)
        ok, reason = validate(data)
        attempts.append({
            "attempt": attempt + 1,
            "raw": data,
            "accepted": bool(ok),
            "reason": reason,
        })
        last_data = data
        if ok:
            return data, True, attempts

        if attempt < max_attempts - 1 and build_retry_message is not None:
            messages = list(base_messages)
            messages.append({
                "role": "user",
                "content": build_retry_message(data, reason or "schema violation"),
            })

    return last_data, False, attempts
