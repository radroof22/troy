"""Demo: Real-time interception with TroyGuard.

Run with:  uv run python examples/demo_guard.py
"""

from troy.guard import TroyGuard, Decision
from troy.policy.engine import PolicyRule


# -- Policy rules using real-time any_prev() pattern --------------------------

RULES = [
    PolicyRule(
        rule_id="pii-exfiltration-rt",
        description="PII exfiltration (real-time)",
        condition=(
            "get(step, 'metadata.network_zone') == 'external' "
            "and step['type'] == 'tool_call' "
            "and any_prev(lambda s: get(s, 'metadata.data_classification') == 'pii')"
        ),
        severity="critical",
        weight=50,
    ),
    PolicyRule(
        rule_id="block-external-unauth",
        description="Block unauthorized external calls",
        condition="get(step, 'metadata.network_zone') == 'external' and get(step, 'metadata.authorized') is not True",
        severity="high",
        weight=30,
    ),
]


def print_decision(label: str, decision: Decision) -> None:
    status = "ALLOWED" if decision.allowed else "BLOCKED"
    print(f"  [{status}] {label} (risk={decision.risk_score})")
    for v in decision.violations:
        print(f"    ! {v.rule_description} [{v.severity}]")


# -- Scenario 1: Safe workflow (enforce mode) ---------------------------------

def demo_safe_workflow():
    print("\n=== Scenario 1: Safe workflow (enforce mode) ===")
    guard = TroyGuard(policy=RULES, agent_name="safe-agent", mode="enforce")

    d1 = guard.check("read_public_data", metadata={"data_classification": "public"})
    print_decision("read_public_data", d1)

    d2 = guard.check("send_internal_email", metadata={"network_zone": "internal"})
    print_decision("send_internal_email", d2)

    print(f"  Trace has {len(guard.get_trace().steps)} steps, all allowed.\n")


# -- Scenario 2: PII exfiltration blocked (enforce mode) ----------------------

def demo_pii_exfiltration():
    print("=== Scenario 2: PII exfiltration blocked (enforce mode) ===")
    guard = TroyGuard(policy=RULES, agent_name="risky-agent", mode="enforce")

    d1 = guard.check("read_customer_pii", metadata={"data_classification": "pii"})
    print_decision("read_customer_pii", d1)

    d2 = guard.check("send_to_external_api", metadata={"network_zone": "external"})
    print_decision("send_to_external_api", d2)

    if not d2.allowed:
        print("  >> Agent prevented from exfiltrating PII!\n")


# -- Scenario 3: Monitor mode (alert but allow) ------------------------------

def demo_monitor_mode():
    print("=== Scenario 3: Monitor mode (alert but allow) ===")
    alerts: list[Decision] = []

    guard = TroyGuard(
        policy=RULES,
        agent_name="monitored-agent",
        mode="monitor",
        on_violation=lambda d: alerts.append(d),
    )

    d1 = guard.check("read_customer_pii", metadata={"data_classification": "pii"})
    print_decision("read_customer_pii", d1)

    d2 = guard.check("send_to_external_api", metadata={"network_zone": "external"})
    print_decision("send_to_external_api", d2)

    print(f"  >> {len(alerts)} alert(s) fired, but execution was allowed.")
    for alert in alerts:
        for v in alert.violations:
            print(f"     Alert: {v.rule_description}")
    print()


if __name__ == "__main__":
    print("TroyGuard — Real-Time Interception Demo")
    print("=" * 50)
    demo_safe_workflow()
    demo_pii_exfiltration()
    demo_monitor_mode()
    print("Done.")
