"""Demo: Framework adapter integration patterns for troy.

These examples show the integration API for each framework adapter.
They won't run without the frameworks installed — see tests/test_adapters.py
for runnable tests that mock the framework imports.
"""

from troy.models import StepType
from troy.policy.engine import PolicyRule

# Shared policy for all examples
POLICY = [
    PolicyRule(
        rule_id="no-delete",
        description="Block destructive file operations",
        condition="'delete' in step.description or 'rm' in step.description",
        severity="critical",
    ),
]


# ---------------------------------------------------------------------------
# LangChain
# ---------------------------------------------------------------------------

def langchain_example():
    """LangChain — callback handler."""
    print("=== LangChain Integration ===")

    from troy.adapters.langchain import TroyHandler

    # Basic usage
    handler = TroyHandler(policy=POLICY)
    # agent.invoke(input, config={"callbacks": [handler]})

    # With metadata_fn — map tool calls to security metadata
    def my_metadata(action, input_data, step_type):
        meta = {"network_zone": "internal", "data_classification": "public"}
        if action == "send_email":
            meta["network_zone"] = "external"
            meta["category"] = "communication"
        return meta

    handler = TroyHandler(policy=POLICY, metadata_fn=my_metadata)
    # agent.invoke(input, config={"callbacks": [handler]})

    # Access the guard for trace/violations:
    trace = handler.guard.get_trace()
    print(f"  Trace has {len(trace.steps)} steps")


# ---------------------------------------------------------------------------
# OpenAI Agents SDK
# ---------------------------------------------------------------------------

def openai_agents_example():
    """OpenAI Agents SDK — agent hooks."""
    print("=== OpenAI Agents SDK Integration ===")

    from troy.adapters.openai_agents import TroyHooks

    # Basic usage
    hooks = TroyHooks(policy=POLICY)
    # agent = Agent(name="bot", hooks=hooks)

    # With metadata_fn
    hooks = TroyHooks(
        policy=POLICY,
        metadata_fn=lambda action, inp, st: {"network_zone": "external"},
    )
    # agent = Agent(name="bot", hooks=hooks)

    # Access the guard for trace/violations:
    trace = hooks.guard.get_trace()
    print(f"  Trace has {len(trace.steps)} steps")


# ---------------------------------------------------------------------------
# CrewAI
# ---------------------------------------------------------------------------

def crewai_example():
    """CrewAI — global hooks."""
    print("=== CrewAI Integration ===")

    from troy.adapters.crewai import enable_troy, disable_troy

    # Basic usage
    guard = enable_troy(policy=POLICY)
    # ... run your crew ...
    trace = guard.get_trace()
    print(f"  Trace has {len(trace.steps)} steps")
    disable_troy()

    # With metadata_fn
    guard = enable_troy(
        policy=POLICY,
        metadata_fn=lambda action, inp, st: {"network_zone": "external"},
    )
    disable_troy()


if __name__ == "__main__":
    # These will fail without the frameworks installed.
    # Install with: pip install troy[langchain] troy[openai-agents] troy[crewai]
    try:
        langchain_example()
    except ImportError as e:
        print(f"  Skipped: {e}")

    try:
        openai_agents_example()
    except ImportError as e:
        print(f"  Skipped: {e}")

    try:
        crewai_example()
    except ImportError as e:
        print(f"  Skipped: {e}")
