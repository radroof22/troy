"""Policy evaluation engine for agent execution traces."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agent_audit.models import Step, Trace, Violation


class PolicyRule:
    """A single policy rule loaded from JSON."""

    def __init__(self, rule_id: str, description: str, condition: str, severity: str = "medium", weight: int = 10):
        self.rule_id = rule_id
        self.description = description
        self.condition = condition
        self.severity = severity
        self.weight = weight


def load_policy(path: Path) -> list[PolicyRule]:
    """Load policy rules from a JSON file."""
    data = json.loads(path.read_text())
    rules = []
    for r in data.get("rules", []):
        rules.append(PolicyRule(
            rule_id=r["rule_id"],
            description=r["description"],
            condition=r["condition"],
            severity=r.get("severity", "medium"),
            weight=r.get("weight", 10),
        ))
    return rules


def _make_step_dict(step: Step) -> dict[str, Any]:
    """Convert a Step to a dict for use in condition evaluation."""
    return {
        "step_id": step.step_id,
        "type": step.type.value,
        "description": step.description,
        "input": step.input,
        "output": step.output,
        "metadata": step.metadata,
    }


def _make_trace_dict(trace: Trace) -> dict[str, Any]:
    """Convert trace-level info to a dict for condition evaluation."""
    return {
        "trace_id": trace.trace_id,
        "agent_name": trace.agent_name,
        "metadata": trace.metadata,
    }


def _safe_get(d: Any, path: str, default: Any = None) -> Any:
    """Safely traverse nested dicts using dot-separated path.

    Example: _safe_get(step, 'metadata.data_classification') navigates
    step['metadata']['data_classification'], returning default if any key is missing.
    """
    keys = path.split(".")
    current = d
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return default
        if current is None:
            return default
    return current


def _matches(text: Any, pattern: str) -> bool:
    """Case-insensitive regex search on text."""
    return bool(re.search(pattern, str(text), re.IGNORECASE))


def evaluate_policy(trace: Trace, rules: list[PolicyRule]) -> list[Violation]:
    """Evaluate all policy rules against all steps in a trace.

    Conditions are evaluated with cross-step variables and helper functions
    exposed. A condition returning True means the rule is violated.
    """
    violations = []
    all_steps = [_make_step_dict(s) for s in trace.steps]
    trace_dict = _make_trace_dict(trace)
    agent_dict = {"name": trace.agent_name, "metadata": trace.metadata}

    for i, step in enumerate(trace.steps):
        step_dict = all_steps[i]
        prev_steps = all_steps[:i]
        next_steps = all_steps[i + 1:]

        eval_context = {
            # Current step
            "step": step_dict,
            # Cross-step access
            "steps": all_steps,
            "step_index": i,
            "prev_steps": prev_steps,
            "next_steps": next_steps,
            # Trace / agent metadata
            "trace": trace_dict,
            "agent": agent_dict,
            # Helper functions
            "any_step": lambda fn, _s=all_steps: any(fn(s) for s in _s),
            "any_next": lambda fn, _s=next_steps: any(fn(s) for s in _s),
            "any_prev": lambda fn, _s=prev_steps: any(fn(s) for s in _s),
            "matches": _matches,
            "get": _safe_get,
            # Builtins
            "True": True,
            "False": False,
            "None": None,
            "str": str,
        }

        eval_globals = {"__builtins__": {}, **eval_context}

        for rule in rules:
            try:
                violated = eval(rule.condition, eval_globals)
            except Exception:
                # If condition can't be evaluated, skip
                continue
            if violated:
                violations.append(Violation(
                    rule_id=rule.rule_id,
                    rule_description=rule.description,
                    step_id=step.step_id,
                    severity=rule.severity,
                    details=f"Rule '{rule.description}' violated at step {step.step_id}",
                ))
    return violations


def evaluate_step(
    step_dict: dict[str, Any],
    rules: list[PolicyRule],
    prev_steps: list[dict[str, Any]] | None = None,
    agent: dict[str, Any] | None = None,
    trace: dict[str, Any] | None = None,
) -> list[Violation]:
    """Evaluate all rules against a single step in real-time.

    Unlike evaluate_policy(), this has no knowledge of future steps:
    next_steps is always empty and any_next() always returns False.
    Cross-step rules (e.g. PII exfiltration) should use any_prev() instead.
    """
    prev_steps = prev_steps or []
    agent = agent or {}
    trace = trace or {}
    all_steps = prev_steps + [step_dict]

    eval_context = {
        "step": step_dict,
        "steps": all_steps,
        "step_index": len(prev_steps),
        "prev_steps": prev_steps,
        "next_steps": [],
        "trace": trace,
        "agent": agent,
        "any_step": lambda fn, _s=all_steps: any(fn(s) for s in _s),
        "any_next": lambda fn: False,
        "any_prev": lambda fn, _s=prev_steps: any(fn(s) for s in _s),
        "matches": _matches,
        "get": _safe_get,
        "True": True,
        "False": False,
        "None": None,
        "str": str,
    }
    eval_globals = {"__builtins__": {}, **eval_context}

    violations = []
    for rule in rules:
        try:
            violated = eval(rule.condition, eval_globals)
        except Exception:
            continue
        if violated:
            violations.append(Violation(
                rule_id=rule.rule_id,
                rule_description=rule.description,
                step_id=step_dict.get("step_id", "unknown"),
                severity=rule.severity,
                details=f"Rule '{rule.description}' violated at step {step_dict.get('step_id', 'unknown')}",
            ))
    return violations


def compute_risk_score(violations: list[Violation], rules: list[PolicyRule]) -> int:
    """Compute overall risk score: min(100, sum of weights for violated rules)."""
    rule_weights = {r.rule_id: r.weight for r in rules}
    violated_rule_ids = {v.rule_id for v in violations}
    total = sum(rule_weights.get(rid, 10) for rid in violated_rule_ids)
    return min(100, total)
