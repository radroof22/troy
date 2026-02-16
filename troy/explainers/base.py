"""Abstract base class for step/trace explainers."""

from abc import ABC, abstractmethod

from troy.models import Step, StepExplanation, Trace, TraceSummary


class BaseExplainer(ABC):
    """Interface for generating explanations of agent behavior."""

    @abstractmethod
    def explain_step(self, step: Step, preceding_steps: list[Step], following_steps: list[Step]) -> StepExplanation:
        """Generate a semantic explanation for a single step, given surrounding context."""

    @abstractmethod
    def summarize_trace(self, trace: Trace) -> TraceSummary:
        """Generate a high-level summary of the entire trace."""

    @abstractmethod
    async def aexplain_step(self, step: Step, preceding_steps: list[Step], following_steps: list[Step]) -> StepExplanation:
        """Async variant of explain_step."""

    @abstractmethod
    async def asummarize_trace(self, trace: Trace) -> TraceSummary:
        """Async variant of summarize_trace."""
