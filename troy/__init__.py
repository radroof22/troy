"""troy — audit agent execution traces against configurable policies."""

__version__ = "0.1.0"

from troy.guard.core import TroyGuard, guarded
from troy.guard.decision import Decision
from troy.models import Step, StepType, Trace, Violation

__all__ = [
    "TroyGuard",
    "Decision",
    "Step",
    "StepType",
    "Trace",
    "Violation",
    "guarded",
]
