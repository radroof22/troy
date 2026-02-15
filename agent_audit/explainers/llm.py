"""LLM-based explainer using the OpenAI SDK."""

import json
import logging

from openai import AsyncOpenAI, OpenAI

logger = logging.getLogger(__name__)

from agent_audit.explainers.base import BaseExplainer
from agent_audit.models import Step, StepExplanation, Trace, TraceSummary

STEP_PROMPT = """\
You are an AI auditor. Your job is to explain WHY an agent made a decision, not just what it did. The agent's internal reasoning is not available to you — you must infer intent from the inputs, outputs, and surrounding execution context.

## Current step
Step ID: {step_id}
Type: {type}
Description: {description}
Input: {input}
Output: {output}
Metadata: {metadata}

## What happened before this step
{preceding_context}

## What happened after this step
{following_context}

Using the execution chain above, infer WHY the agent chose this action. Look at what information it had from prior steps, what it produced, and how later steps used the output.

Respond with JSON only (no markdown fences):
{{
  "step_id": "{step_id}",
  "summary": "<what this step did>",
  "rationale": "<your best inference of WHY the agent chose this action based on the execution context. What information from prior steps motivated it? What was it trying to achieve for subsequent steps?>",
  "alternatives": ["<what else could the agent have done instead? list safer or more appropriate alternatives if any, empty list if the choice was reasonable>"],
  "risk_factors": ["<security/privacy/safety concerns, empty list if none>"],
  "data_accessed": ["<data sources, APIs, or sensitive info accessed, empty list if none>"]
}}
"""

TRACE_PROMPT = """\
You are an AI auditor. Analyze this complete agent execution trace. The agent's internal reasoning is not available — you must reconstruct the decision-making chain from the sequence of actions, their inputs, and their outputs.

For each key decision, explain: what information did the agent have, what did it choose to do, and why was that the likely choice given the context?

Agent: {agent_name}
Steps:
{steps_summary}

Respond with JSON only (no markdown fences):
{{
  "overview": "<2-3 sentence summary reconstructing the agent's decision-making chain — not just what it did, but the inferred reasoning that connects each step to the next>",
  "key_actions": ["<the most important actions and your inferred reasoning for why the agent chose them>"],
  "concerns": ["<concerns about the agent's decision-making: unjustified leaps, skipped safer options, actions without sufficient basis, unnecessary data access, etc. Empty list if none>"]
}}
"""


import re


def _extract_json(raw: str) -> dict:
    """Parse JSON from LLM output, handling fences, preamble text, etc."""
    text = raw.strip()
    if not text:
        return {}
    # Try direct parse first (fast path)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip ```json ... ``` or ``` ... ``` fences
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1).strip())
    # Find the first { ... last } (handles preamble like "Here is the response:")
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])
    return {}


def _coerce_to_strings(items: list) -> list[str]:
    """Coerce a list of mixed items (strings, dicts, etc.) to flat strings.

    Smaller models sometimes return [{"action": "..."}, ...] instead of ["..."].
    """
    result = []
    for item in items:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict):
            result.append(" — ".join(str(v) for v in item.values()))
        else:
            result.append(str(item))
    return result


def _format_step_brief(step: Step) -> str:
    """Format a step as a brief one-liner for context."""
    return f"[{step.step_id}] ({step.type.value}) {step.description} | input: {json.dumps(step.input)} | output: {json.dumps(step.output)}"


class LLMExplainer(BaseExplainer):
    """Explainer that uses OpenAI chat completions for semantic analysis."""

    def __init__(self, model: str = "gpt-4o-mini", base_url: str | None = None, api_key: str | None = None):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.aclient = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def explain_step(self, step: Step, preceding_steps: list[Step], following_steps: list[Step]) -> StepExplanation:
        preceding_context = (
            "\n".join(_format_step_brief(s) for s in preceding_steps)
            if preceding_steps
            else "This is the first step."
        )
        following_context = (
            "\n".join(_format_step_brief(s) for s in following_steps)
            if following_steps
            else "This is the last step."
        )

        prompt = STEP_PROMPT.format(
            step_id=step.step_id,
            type=step.type.value,
            description=step.description,
            input=json.dumps(step.input),
            output=json.dumps(step.output),
            metadata=json.dumps(step.metadata),
            preceding_context=preceding_context,
            following_context=following_context,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        content = response.choices[0].message.content or ""
        try:
            data = _extract_json(content)
        except json.JSONDecodeError:
            logger.error("Failed to parse explain_step response for step %s. Raw content: %s", step.step_id, content[:500])
            raise
        return StepExplanation(
            step_id=step.step_id,
            summary=data.get("summary", ""),
            rationale=data.get("rationale", ""),
            alternatives=_coerce_to_strings(data.get("alternatives", [])),
            risk_factors=_coerce_to_strings(data.get("risk_factors", [])),
            data_accessed=_coerce_to_strings(data.get("data_accessed", [])),
            raw_response=content,
        )

    def summarize_trace(self, trace: Trace) -> TraceSummary:
        steps_summary = "\n".join(
            f"  - {_format_step_brief(s)}" for s in trace.steps
        )
        prompt = TRACE_PROMPT.format(
            agent_name=trace.agent_name,
            steps_summary=steps_summary,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        content = response.choices[0].message.content or ""
        try:
            data = _extract_json(content)
        except json.JSONDecodeError:
            logger.error("Failed to parse summarize_trace response for trace %s. Raw content: %s", trace.trace_id, content[:500])
            raise
        return TraceSummary(
            overview=data.get("overview", ""),
            key_actions=_coerce_to_strings(data.get("key_actions", [])),
            concerns=_coerce_to_strings(data.get("concerns", [])),
            raw_response=content,
        )

    async def aexplain_step(self, step: Step, preceding_steps: list[Step], following_steps: list[Step]) -> StepExplanation:
        preceding_context = (
            "\n".join(_format_step_brief(s) for s in preceding_steps)
            if preceding_steps
            else "This is the first step."
        )
        following_context = (
            "\n".join(_format_step_brief(s) for s in following_steps)
            if following_steps
            else "This is the last step."
        )

        prompt = STEP_PROMPT.format(
            step_id=step.step_id,
            type=step.type.value,
            description=step.description,
            input=json.dumps(step.input),
            output=json.dumps(step.output),
            metadata=json.dumps(step.metadata),
            preceding_context=preceding_context,
            following_context=following_context,
        )

        response = await self.aclient.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        content = response.choices[0].message.content or ""
        try:
            data = _extract_json(content)
        except json.JSONDecodeError:
            logger.error("Failed to parse aexplain_step response for step %s. Raw content: %s", step.step_id, content[:500])
            raise
        return StepExplanation(
            step_id=step.step_id,
            summary=data.get("summary", ""),
            rationale=data.get("rationale", ""),
            alternatives=_coerce_to_strings(data.get("alternatives", [])),
            risk_factors=_coerce_to_strings(data.get("risk_factors", [])),
            data_accessed=_coerce_to_strings(data.get("data_accessed", [])),
            raw_response=content,
        )

    async def asummarize_trace(self, trace: Trace) -> TraceSummary:
        steps_summary = "\n".join(
            f"  - {_format_step_brief(s)}" for s in trace.steps
        )
        prompt = TRACE_PROMPT.format(
            agent_name=trace.agent_name,
            steps_summary=steps_summary,
        )

        response = await self.aclient.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        content = response.choices[0].message.content or ""
        try:
            data = _extract_json(content)
        except json.JSONDecodeError:
            logger.error("Failed to parse asummarize_trace response for trace %s. Raw content: %s", trace.trace_id, content[:500])
            raise
        return TraceSummary(
            overview=data.get("overview", ""),
            key_actions=_coerce_to_strings(data.get("key_actions", [])),
            concerns=_coerce_to_strings(data.get("concerns", [])),
            raw_response=content,
        )
