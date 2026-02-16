"""Demo: Framework adapter integration patterns for agent-audit.

These examples show the 2-3 line integration API. They won't run
without the frameworks installed — see tests/test_adapters.py for
runnable tests that mock the framework imports.
"""

from agent_audit.policy.engine import PolicyRule

# Shared policy for all examples
POLICY = [
    PolicyRule(
        rule_id="no-delete",
        description="Block destructive file operations",
        condition="'delete' in step.description or 'rm' in step.description",
        severity="critical",
    ),
]


def langchain_example():
    """LangChain — callback handler (2 lines to integrate)."""
    print("=== LangChain Integration ===")
    print(
        """
    from agent_audit.adapters.langchain import AuditHandler

    handler = AuditHandler(policy="policy.json")
    agent.invoke(input, config={"callbacks": [handler]})

    # Access the guard for trace/violations:
    trace = handler.guard.get_trace()
    """
    )


def openai_agents_example():
    """OpenAI Agents SDK — agent hooks (2 lines to integrate)."""
    print("=== OpenAI Agents SDK Integration ===")
    print(
        """
    from agent_audit.adapters.openai_agents import AuditHooks

    hooks = AuditHooks(policy="policy.json")
    agent = Agent(name="bot", hooks=hooks)

    # Access the guard:
    trace = hooks.guard.get_trace()
    """
    )


def crewai_example():
    """CrewAI — global one-liner."""
    print("=== CrewAI Integration ===")
    print(
        """
    from agent_audit.adapters.crewai import enable_audit, disable_audit

    guard = enable_audit(policy="policy.json")
    # ... run your crew ...
    trace = guard.get_trace()
    disable_audit()  # cleanup
    """
    )


if __name__ == "__main__":
    langchain_example()
    openai_agents_example()
    crewai_example()
    print("\nAll adapter patterns shown. Install frameworks to use them for real.")
