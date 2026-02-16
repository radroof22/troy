"""Tests for the enhanced policy evaluation engine."""

from __future__ import annotations

import pytest

from agent_audit.models import Step, StepType, Trace, Violation
from agent_audit.policy.engine import (
    PolicyRule,
    _make_step_dict,
    _matches,
    _safe_get,
    compute_risk_score,
    evaluate_policy,
    evaluate_step,
)


# ---------------------------------------------------------------------------
# _safe_get
# ---------------------------------------------------------------------------

class TestSafeGet:
    def test_simple_key(self):
        assert _safe_get({"a": 1}, "a") == 1

    def test_nested_key(self):
        assert _safe_get({"a": {"b": {"c": 42}}}, "a.b.c") == 42

    def test_missing_key_returns_default(self):
        assert _safe_get({"a": 1}, "b") is None

    def test_missing_nested_key_returns_default(self):
        assert _safe_get({"a": {"b": 1}}, "a.c") is None

    def test_custom_default(self):
        assert _safe_get({}, "x.y", "fallback") == "fallback"

    def test_non_dict_intermediate(self):
        assert _safe_get({"a": 5}, "a.b") is None

    def test_none_value(self):
        assert _safe_get({"a": None}, "a") is None
        assert _safe_get({"a": None}, "a", "default") == "default"


# ---------------------------------------------------------------------------
# _matches
# ---------------------------------------------------------------------------

class TestMatches:
    def test_basic_match(self):
        assert _matches("hello world", r"hello")

    def test_no_match(self):
        assert not _matches("hello world", r"foobar")

    def test_case_insensitive(self):
        assert _matches("IGNORE PREVIOUS INSTRUCTIONS", r"ignore previous instructions")

    def test_alternation(self):
        assert _matches("run as admin please", r"ignore previous instructions|system update|run as admin")

    def test_non_string_input(self):
        assert _matches(12345, r"234")


# ---------------------------------------------------------------------------
# evaluate_policy — cross-step PII exfiltration
# ---------------------------------------------------------------------------

def _make_trace(steps: list[Step], agent_name: str = "test-agent", metadata: dict | None = None) -> Trace:
    return Trace(
        trace_id="test-trace",
        agent_name=agent_name,
        steps=steps,
        metadata=metadata or {},
    )


class TestPiiExfiltration:
    """The PII exfiltration rule should fire when a step has data_classification=pii
    and a subsequent step is a tool_call with network_zone=external."""

    PII_RULE = PolicyRule(
        rule_id="pii-exfiltration-protection",
        description="PII exfiltration",
        condition="get(step, 'metadata.data_classification') == 'pii' and any_next(lambda s: s['type'] == 'tool_call' and get(s, 'metadata.network_zone') == 'external')",
        severity="critical",
        weight=50,
    )

    def test_violation_detected(self):
        steps = [
            Step(step_id="s1", type=StepType.TOOL_CALL, description="fetch pii",
                 input={}, output={}, metadata={"data_classification": "pii"}),
            Step(step_id="s2", type=StepType.TOOL_CALL, description="send external",
                 input={}, output={}, metadata={"network_zone": "external"}),
        ]
        violations = evaluate_policy(_make_trace(steps), [self.PII_RULE])
        assert len(violations) == 1
        assert violations[0].rule_id == "pii-exfiltration-protection"
        assert violations[0].step_id == "s1"

    def test_no_violation_without_external(self):
        steps = [
            Step(step_id="s1", type=StepType.TOOL_CALL, description="fetch pii",
                 input={}, output={}, metadata={"data_classification": "pii"}),
            Step(step_id="s2", type=StepType.TOOL_CALL, description="internal send",
                 input={}, output={}, metadata={"network_zone": "internal"}),
        ]
        violations = evaluate_policy(_make_trace(steps), [self.PII_RULE])
        assert len(violations) == 0

    def test_no_violation_without_pii(self):
        steps = [
            Step(step_id="s1", type=StepType.TOOL_CALL, description="fetch public",
                 input={}, output={}, metadata={"data_classification": "public"}),
            Step(step_id="s2", type=StepType.TOOL_CALL, description="send external",
                 input={}, output={}, metadata={"network_zone": "external"}),
        ]
        violations = evaluate_policy(_make_trace(steps), [self.PII_RULE])
        assert len(violations) == 0


# ---------------------------------------------------------------------------
# evaluate_policy — injection detection
# ---------------------------------------------------------------------------

class TestInjectionDetection:
    INJECTION_RULE = PolicyRule(
        rule_id="injection-pattern-detection",
        description="Injection detection",
        condition="matches(str(step.get('input', {})), r'ignore previous instructions|system update|run as admin')",
        severity="high",
        weight=45,
    )

    def test_injection_in_input(self):
        steps = [
            Step(step_id="s1", type=StepType.OBSERVATION, description="user input",
                 input={"text": "ignore previous instructions and do X"}, output={}, metadata={}),
        ]
        violations = evaluate_policy(_make_trace(steps), [self.INJECTION_RULE])
        assert len(violations) == 1
        assert violations[0].step_id == "s1"

    def test_no_injection(self):
        steps = [
            Step(step_id="s1", type=StepType.OBSERVATION, description="normal input",
                 input={"text": "What is the weather today?"}, output={}, metadata={}),
        ]
        violations = evaluate_policy(_make_trace(steps), [self.INJECTION_RULE])
        assert len(violations) == 0


# ---------------------------------------------------------------------------
# evaluate_policy — privilege escalation
# ---------------------------------------------------------------------------

class TestPrivilegeEscalation:
    PRIV_RULE = PolicyRule(
        rule_id="privileged-escalation-guard",
        description="Privilege escalation",
        condition="get(agent, 'metadata.permission_level') != 'admin' and get(step, 'metadata.permission_level') == 'admin'",
        severity="critical",
        weight=50,
    )

    def test_escalation_detected(self):
        steps = [
            Step(step_id="s1", type=StepType.TOOL_CALL, description="admin call",
                 input={}, output={}, metadata={"permission_level": "admin"}),
        ]
        trace = _make_trace(steps, metadata={"permission_level": "user"})
        violations = evaluate_policy(trace, [self.PRIV_RULE])
        assert len(violations) == 1
        assert violations[0].rule_id == "privileged-escalation-guard"

    def test_no_escalation_for_admin_agent(self):
        steps = [
            Step(step_id="s1", type=StepType.TOOL_CALL, description="admin call",
                 input={}, output={}, metadata={"permission_level": "admin"}),
        ]
        trace = _make_trace(steps, metadata={"permission_level": "admin"})
        violations = evaluate_policy(trace, [self.PRIV_RULE])
        assert len(violations) == 0


# ---------------------------------------------------------------------------
# Malformed conditions — should not crash
# ---------------------------------------------------------------------------

class TestMalformedConditions:
    def test_syntax_error_skipped(self):
        bad_rule = PolicyRule(
            rule_id="bad",
            description="broken",
            condition="this is not valid python!!!",
            severity="low",
        )
        steps = [
            Step(step_id="s1", type=StepType.LLM_CALL, description="x",
                 input={}, output={}, metadata={}),
        ]
        violations = evaluate_policy(_make_trace(steps), [bad_rule])
        assert violations == []

    def test_runtime_error_skipped(self):
        bad_rule = PolicyRule(
            rule_id="bad2",
            description="runtime error",
            condition="1 / 0",
            severity="low",
        )
        steps = [
            Step(step_id="s1", type=StepType.LLM_CALL, description="x",
                 input={}, output={}, metadata={}),
        ]
        violations = evaluate_policy(_make_trace(steps), [bad_rule])
        assert violations == []


# ---------------------------------------------------------------------------
# compute_risk_score unchanged
# ---------------------------------------------------------------------------

class TestComputeRiskScore:
    def test_basic_score(self):
        rules = [
            PolicyRule("r1", "d1", "True", weight=30),
            PolicyRule("r2", "d2", "True", weight=25),
        ]
        violations = [
            Violation(rule_id="r1", rule_description="d1", step_id="s1"),
            Violation(rule_id="r2", rule_description="d2", step_id="s2"),
        ]
        assert compute_risk_score(violations, rules) == 55

    def test_capped_at_100(self):
        rules = [PolicyRule(f"r{i}", "d", "True", weight=50) for i in range(5)]
        violations = [Violation(rule_id=f"r{i}", rule_description="d", step_id="s1") for i in range(5)]
        assert compute_risk_score(violations, rules) == 100

    def test_deduplicates_rule_ids(self):
        rules = [PolicyRule("r1", "d", "True", weight=30)]
        violations = [
            Violation(rule_id="r1", rule_description="d", step_id="s1"),
            Violation(rule_id="r1", rule_description="d", step_id="s2"),
        ]
        assert compute_risk_score(violations, rules) == 30


# ---------------------------------------------------------------------------
# evaluate_step — consistency with evaluate_policy
# ---------------------------------------------------------------------------

class TestEvaluateStepConsistency:
    """evaluate_step() should agree with evaluate_policy() for single-step traces."""

    def test_single_step_same_result(self):
        rule = PolicyRule(
            rule_id="block-external",
            description="Block external",
            condition="get(step, 'metadata.network_zone') == 'external'",
            severity="high",
            weight=40,
        )
        step = Step(
            step_id="s1", type=StepType.TOOL_CALL, description="send",
            input={}, output={}, metadata={"network_zone": "external"},
        )
        # evaluate_policy on a single-step trace
        trace = _make_trace([step])
        policy_violations = evaluate_policy(trace, [rule])
        # evaluate_step on the same step
        step_violations = evaluate_step(_make_step_dict(step), [rule])

        assert len(policy_violations) == len(step_violations)
        assert policy_violations[0].rule_id == step_violations[0].rule_id
