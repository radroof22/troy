"""Load and validate agent execution traces from JSON files."""

from pathlib import Path

from pydantic import ValidationError

from agent_audit.models import Trace


def load_trace(path: Path) -> Trace:
    """Load a trace from a JSON file, validating with Pydantic.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        pydantic.ValidationError: If the JSON doesn't match the Trace schema.
        json.JSONDecodeError: If the file isn't valid JSON.
    """
    raw = path.read_text()
    try:
        return Trace.model_validate_json(raw)
    except ValidationError:
        raise
