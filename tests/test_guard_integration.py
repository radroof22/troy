"""Integration tests: replay example traces through both engines and compare.

These tests validate that the real-time guard is a sound replacement for
post-hoc auditing by replaying the same traces through both engines and
checking that:
  1. Non-temporal rules (single-step) produce identical violations.
  2. Temporal rules rewritten from any_next → any_prev catch the same
     threats (at the later step instead of the earlier one).
  3. Adversarial patterns (interleaved safe steps, reset-retry) are handled.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from troy.guard import TroyGuard
from troy.models import Step, StepType, Trace
from troy.policy.engine import (
    PolicyRule,
    _make_step_dict,
    compute_risk_score,
    evaluate_policy,
    evaluate_step,
    load_policy,
)


EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
POLICY_PATH = EXAMPLES / "policy.json"

TRACE_FILES = sorted(EXAMPLES.glob("trace*.json"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_trace(path: Path) -> Trace:
    return Trace(**json.loads(path.read_text()))


def replay_through_guard(
    trace: Trace,
    rules: list[PolicyRule],
    mode: str = "enforce",
) -> list[dict]:
    """Replay a trace step-by-step through the guard, returning per-step results."""
    guard = TroyGuard(
        policy=rules,
        agent_name=trace.agent_name,
        mode=mode,
        agent_metadata=trace.metadata,
    )
    results = []
    for step in trace.steps:
        decision = guard.check(
            action=step.description,
            input=step.input,
            metadata=step.metadata,
            step_type=step.type,
        )
        # Record output so subsequent steps can reference it
        guard.record_output(decision.step_id, step.output)
        results.append({
            "original_step_id": step.step_id,
            "guard_step_id": decision.step_id,
            "allowed": decision.allowed,
            "violation_rule_ids": sorted(v.rule_id for v in decision.violations),
        })
    return results


# ---------------------------------------------------------------------------
# The real-time equivalents of the post-hoc rules
# ---------------------------------------------------------------------------

# Original post-hoc PII rule uses any_next (fires on the PII step):
#   get(step, 'metadata.data_classification') == 'pii'
#     and any_next(lambda s: s['type'] == 'tool_call'
#       and get(s, 'metadata.network_zone') == 'external')
#
# Real-time equivalent uses any_prev (fires on the external-send step):
PII_RULE_POSTHOC = PolicyRule(
    rule_id="pii-exfiltration-protection",
    description="PII exfiltration (post-hoc)",
    condition=(
        "get(step, 'metadata.data_classification') == 'pii' "
        "and any_next(lambda s: s['type'] == 'tool_call' "
        "and get(s, 'metadata.network_zone') == 'external')"
    ),
    severity="critical",
    weight=50,
)

PII_RULE_REALTIME = PolicyRule(
    rule_id="pii-exfiltration-protection",
    description="PII exfiltration (real-time)",
    condition=(
        "step['type'] == 'tool_call' "
        "and get(step, 'metadata.network_zone') == 'external' "
        "and any_prev(lambda s: get(s, 'metadata.data_classification') == 'pii')"
    ),
    severity="critical",
    weight=50,
)

# All other rules in the example policy are single-step (no any_next/any_prev),
# so they work identically in both engines.


# ---------------------------------------------------------------------------
# 1. Single-step rules produce identical results in both engines
# ---------------------------------------------------------------------------

class TestSingleStepRulesIdentical:
    """Rules that don't use any_next/any_prev should fire on exactly the
    same steps in both engines."""

    SINGLE_STEP_RULES = [
        PolicyRule(
            rule_id="injection-pattern-detection",
            description="Injection detection",
            condition="matches(str(step.get('input', {})), r'ignore previous instructions|system update|run as admin')",
            severity="high",
            weight=45,
        ),
        PolicyRule(
            rule_id="secure-db-access",
            description="Raw SQL detection",
            condition="step['type'] == 'tool_call' and matches(get(step, 'input.query', ''), r'SELECT|INSERT|UPDATE|DELETE|DROP|UNION')",
            severity="critical",
            weight=40,
        ),
        PolicyRule(
            rule_id="mandatory-human-approval",
            description="Missing approval token",
            condition="get(step, 'metadata.requires_approval') is True and get(step, 'metadata.approval_token') is None",
            severity="high",
            weight=35,
        ),
        PolicyRule(
            rule_id="privileged-escalation-guard",
            description="Privilege escalation",
            condition="get(agent, 'metadata.permission_level') != 'admin' and get(step, 'metadata.permission_level') == 'admin'",
            severity="critical",
            weight=50,
        ),
        PolicyRule(
            rule_id="unauthorized-communication",
            description="Unauthorized messaging",
            condition="step['type'] == 'tool_call' and get(step, 'metadata.category') == 'communication'",
            severity="medium",
            weight=20,
        ),
    ]

    @pytest.mark.parametrize("trace_file", TRACE_FILES, ids=lambda p: p.stem)
    def test_same_violations_per_step(self, trace_file: Path):
        trace = load_trace(trace_file)
        # Post-hoc: collect violations grouped by step
        posthoc_violations = evaluate_policy(trace, self.SINGLE_STEP_RULES)
        posthoc_by_step = {}
        for v in posthoc_violations:
            posthoc_by_step.setdefault(v.step_id, set()).add(v.rule_id)

        # Real-time: replay step by step
        guard_results = replay_through_guard(trace, self.SINGLE_STEP_RULES)
        guard_by_step = {}
        for r in guard_results:
            if r["violation_rule_ids"]:
                guard_by_step[r["original_step_id"]] = set(r["violation_rule_ids"])

        assert posthoc_by_step == guard_by_step, (
            f"Mismatch for {trace_file.name}:\n"
            f"  post-hoc: {posthoc_by_step}\n"
            f"  guard:    {guard_by_step}"
        )


# ---------------------------------------------------------------------------
# 2. PII exfiltration: any_next rule misses in real-time, any_prev catches it
# ---------------------------------------------------------------------------

class TestPiiExfiltrationEquivalence:
    """The original any_next rule should fire in post-hoc but NOT in real-time.
    The rewritten any_prev rule should fire in real-time (at a different step)."""

    def _make_pii_exfil_trace(self):
        return Trace(
            trace_id="pii-exfil-test",
            agent_name="test-agent",
            metadata={},
            steps=[
                Step(step_id="s1", type=StepType.TOOL_CALL, description="read PII",
                     metadata={"data_classification": "pii"}),
                Step(step_id="s2", type=StepType.LLM_CALL, description="format data"),
                Step(step_id="s3", type=StepType.TOOL_CALL, description="send externally",
                     metadata={"network_zone": "external"}),
            ],
        )

    def test_posthoc_catches_with_any_next(self):
        trace = self._make_pii_exfil_trace()
        violations = evaluate_policy(trace, [PII_RULE_POSTHOC])
        assert len(violations) == 1
        assert violations[0].step_id == "s1"  # fires on PII step

    def test_realtime_misses_with_any_next(self):
        """The any_next rule should NOT fire in real-time (any_next → False)."""
        trace = self._make_pii_exfil_trace()
        results = replay_through_guard(trace, [PII_RULE_POSTHOC])
        violations = [r for r in results if r["violation_rule_ids"]]
        assert len(violations) == 0  # correctly misses — can't see future

    def test_realtime_catches_with_any_prev(self):
        """The rewritten any_prev rule should fire on the send step."""
        trace = self._make_pii_exfil_trace()
        results = replay_through_guard(trace, [PII_RULE_REALTIME])
        violations = [r for r in results if r["violation_rule_ids"]]
        assert len(violations) == 1
        assert violations[0]["original_step_id"] == "s3"  # fires on send step

    def test_posthoc_also_catches_with_any_prev(self):
        """The any_prev rule works in post-hoc too (just fires at s3 not s1)."""
        trace = self._make_pii_exfil_trace()
        violations = evaluate_policy(trace, [PII_RULE_REALTIME])
        assert len(violations) == 1
        assert violations[0].step_id == "s3"

    def test_both_engines_same_risk_score(self):
        """Both rules have the same weight, so risk score should match."""
        trace = self._make_pii_exfil_trace()
        posthoc_v = evaluate_policy(trace, [PII_RULE_POSTHOC])
        realtime_v = evaluate_policy(trace, [PII_RULE_REALTIME])
        assert compute_risk_score(posthoc_v, [PII_RULE_POSTHOC]) == \
               compute_risk_score(realtime_v, [PII_RULE_REALTIME])


# ---------------------------------------------------------------------------
# 3. Full policy replay: real-time with rewritten rules catches everything
# ---------------------------------------------------------------------------

class TestFullPolicyReplay:
    """Load the example policy, rewrite any_next rules to any_prev,
    and verify the guard catches all threats the post-hoc engine catches."""

    @staticmethod
    def _rewrite_rules_for_realtime(rules: list[PolicyRule]) -> list[PolicyRule]:
        """Replace any_next-based rules with any_prev equivalents."""
        rewritten = []
        for rule in rules:
            if "any_next" in rule.condition and rule.rule_id == "pii-exfiltration-protection":
                rewritten.append(PII_RULE_REALTIME)
            else:
                rewritten.append(rule)
        return rewritten

    @pytest.mark.parametrize("trace_file", TRACE_FILES, ids=lambda p: p.stem)
    def test_same_rule_ids_violated(self, trace_file: Path):
        """The set of violated rule IDs should be the same (step may differ)."""
        trace = load_trace(trace_file)
        posthoc_rules = load_policy(POLICY_PATH)
        realtime_rules = self._rewrite_rules_for_realtime(posthoc_rules)

        posthoc_v = evaluate_policy(trace, posthoc_rules)
        posthoc_rule_ids = {v.rule_id for v in posthoc_v}

        guard_results = replay_through_guard(trace, realtime_rules)
        guard_rule_ids = set()
        for r in guard_results:
            guard_rule_ids.update(r["violation_rule_ids"])

        assert posthoc_rule_ids == guard_rule_ids, (
            f"Rule ID mismatch for {trace_file.name}:\n"
            f"  post-hoc: {posthoc_rule_ids}\n"
            f"  guard:    {guard_rule_ids}"
        )

    @pytest.mark.parametrize("trace_file", TRACE_FILES, ids=lambda p: p.stem)
    def test_risk_scores_match(self, trace_file: Path):
        """Risk scores should be identical since same rules fire."""
        trace = load_trace(trace_file)
        posthoc_rules = load_policy(POLICY_PATH)
        realtime_rules = self._rewrite_rules_for_realtime(posthoc_rules)

        posthoc_v = evaluate_policy(trace, posthoc_rules)
        posthoc_score = compute_risk_score(posthoc_v, posthoc_rules)

        guard_results = replay_through_guard(trace, realtime_rules)
        # Collect all violations from replay
        guard_violations = []
        for r in guard_results:
            for rid in r["violation_rule_ids"]:
                from troy.models import Violation
                guard_violations.append(Violation(
                    rule_id=rid, rule_description="", step_id=r["original_step_id"],
                ))
        guard_score = compute_risk_score(guard_violations, realtime_rules)

        assert posthoc_score == guard_score, (
            f"Risk score mismatch for {trace_file.name}: "
            f"post-hoc={posthoc_score}, guard={guard_score}"
        )


# ---------------------------------------------------------------------------
# 4. Adversarial scenarios
# ---------------------------------------------------------------------------

class TestAdversarialScenarios:
    """Patterns an attacker might use to evade detection."""

    def test_interleaved_safe_steps(self):
        """PII → safe → safe → safe → external send: guard still catches it."""
        guard = TroyGuard(policy=[PII_RULE_REALTIME], mode="enforce")

        d1 = guard.check("read_pii", metadata={"data_classification": "pii"})
        assert d1.allowed is True

        # Interleave benign steps
        for i in range(5):
            d = guard.check(f"safe_step_{i}", metadata={"data_classification": "public"})
            assert d.allowed is True

        # The external send should still be caught
        d_send = guard.check("send_external", metadata={"network_zone": "external"})
        assert d_send.allowed is False
        assert d_send.violation_rule_ids[0] if hasattr(d_send, 'violation_rule_ids') else d_send.violations[0].rule_id == "pii-exfiltration-protection"

    def test_reset_clears_history(self):
        """After reset(), prior PII access is forgotten — send should be allowed."""
        guard = TroyGuard(policy=[PII_RULE_REALTIME], mode="enforce")

        guard.check("read_pii", metadata={"data_classification": "pii"})
        guard.reset()

        d = guard.check("send_external", metadata={"network_zone": "external"})
        assert d.allowed is True  # no history of PII access

    def test_reset_then_replay_catches_again(self):
        """After reset, a new PII→send sequence is still caught."""
        guard = TroyGuard(policy=[PII_RULE_REALTIME], mode="enforce")

        guard.check("read_pii", metadata={"data_classification": "pii"})
        guard.reset()

        # New session: PII again
        guard.check("read_pii_again", metadata={"data_classification": "pii"})
        d = guard.check("send_external", metadata={"network_zone": "external"})
        assert d.allowed is False

    def test_multiple_pii_sources(self):
        """Multiple PII steps before send — still caught, only one violation."""
        guard = TroyGuard(policy=[PII_RULE_REALTIME], mode="enforce")

        guard.check("read_pii_1", metadata={"data_classification": "pii"})
        guard.check("read_pii_2", metadata={"data_classification": "pii"})
        d = guard.check("send_external", metadata={"network_zone": "external"})
        assert d.allowed is False
        assert len(d.violations) == 1  # rule fires once per step

    def test_send_before_pii_is_safe(self):
        """External send BEFORE PII access — no violation."""
        guard = TroyGuard(policy=[PII_RULE_REALTIME], mode="enforce")

        d1 = guard.check("send_external", metadata={"network_zone": "external"})
        assert d1.allowed is True  # no prior PII

        d2 = guard.check("read_pii", metadata={"data_classification": "pii"})
        assert d2.allowed is True  # PII alone is fine

    def test_monitor_mode_sees_all_violations(self):
        """Monitor mode allows everything but collects all violations."""
        alerts = []
        guard = TroyGuard(
            policy=[PII_RULE_REALTIME],
            mode="monitor",
            on_violation=lambda d: alerts.append(d),
        )

        guard.check("read_pii", metadata={"data_classification": "pii"})
        d = guard.check("send_external", metadata={"network_zone": "external"})
        assert d.allowed is True  # monitor allows
        assert len(alerts) == 1
        assert alerts[0].violations[0].rule_id == "pii-exfiltration-protection"

    def test_blocked_step_still_in_history(self):
        """A blocked step is still recorded, so rules referencing it work."""
        # Rule: block if any previous step was blocked (meta.blocked=True)
        # We simulate this by checking that the step dict is in history
        guard = TroyGuard(policy=[PII_RULE_REALTIME], mode="enforce")

        guard.check("read_pii", metadata={"data_classification": "pii"})
        d_blocked = guard.check("send_external", metadata={"network_zone": "external"})
        assert d_blocked.allowed is False

        # The blocked step should still be in the trace
        trace = guard.get_trace()
        assert len(trace.steps) == 2  # both steps recorded
        assert trace.steps[1].description == "send_external"


# ---------------------------------------------------------------------------
# 5. Trace accumulation fidelity
# ---------------------------------------------------------------------------

class TestTraceAccumulationFidelity:
    """The trace built by the guard should be valid for post-hoc re-audit."""

    @pytest.mark.parametrize("trace_file", TRACE_FILES, ids=lambda p: p.stem)
    def test_guard_trace_is_reauditable(self, trace_file: Path):
        """Replay a trace through the guard, then run the resulting trace
        through evaluate_policy. The post-hoc result should find the same
        violations as running evaluate_policy on the original trace
        (using rules that work in both directions)."""
        original = load_trace(trace_file)

        # Use only single-step rules (no any_next) for this comparison
        single_step_rules = [
            r for r in load_policy(POLICY_PATH)
            if "any_next" not in r.condition
        ]

        guard = TroyGuard(
            policy=single_step_rules,
            agent_name=original.agent_name,
            mode="monitor",  # allow all so we get the full trace
            agent_metadata=original.metadata,
        )

        for step in original.steps:
            decision = guard.check(
                action=step.description,
                input=step.input,
                metadata=step.metadata,
                step_type=step.type,
            )
            guard.record_output(decision.step_id, step.output)

        guard_trace = guard.get_trace()

        # Re-audit the guard's trace
        reaudit_v = evaluate_policy(guard_trace, single_step_rules)
        original_v = evaluate_policy(original, single_step_rules)

        assert {v.rule_id for v in reaudit_v} == {v.rule_id for v in original_v}
        assert len(reaudit_v) == len(original_v)
