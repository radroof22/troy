"""Generate audit reports in markdown and JSON formats."""

from __future__ import annotations

import json
from pathlib import Path

from agent_audit.models import AuditResult, BatchResult


def generate_markdown_report(result: AuditResult) -> str:
    """Generate a markdown audit report."""
    lines: list[str] = []

    lines.append(f"# Audit Report: {result.trace.agent_name}")
    lines.append(f"\n**Trace ID:** {result.trace.trace_id}")
    lines.append(f"**Risk Score:** {result.risk_score}/100")
    lines.append(f"**Steps Analyzed:** {len(result.trace.steps)}")
    lines.append(f"**Violations Found:** {len(result.violations)}")

    # Summary
    if result.summary:
        lines.append("\n## Summary")
        lines.append(f"\n{result.summary.overview}")
        if result.summary.key_actions:
            lines.append("\n### Key Actions")
            for action in result.summary.key_actions:
                lines.append(f"- {action}")
        if result.summary.concerns:
            lines.append("\n### Concerns")
            for concern in result.summary.concerns:
                lines.append(f"- {concern}")

    # Step explanations
    if result.explanations:
        lines.append("\n## Step Analysis")
        for explanation in result.explanations:
            lines.append(f"\n### Step: {explanation.step_id}")
            lines.append(f"\n{explanation.summary}")
            if explanation.rationale:
                lines.append(f"\n**Why:** {explanation.rationale}")
            if explanation.alternatives:
                lines.append("\n**Alternatives:**")
                for alt in explanation.alternatives:
                    lines.append(f"- {alt}")
            if explanation.risk_factors:
                lines.append("\n**Risk Factors:**")
                for rf in explanation.risk_factors:
                    lines.append(f"- {rf}")
            if explanation.data_accessed:
                lines.append("\n**Data Accessed:**")
                for da in explanation.data_accessed:
                    lines.append(f"- {da}")

    # Violations
    if result.violations:
        lines.append("\n## Policy Violations")
        for v in result.violations:
            lines.append(f"\n### [{v.severity.upper()}] {v.rule_description}")
            lines.append(f"\n- **Rule ID:** {v.rule_id}")
            lines.append(f"- **Step:** {v.step_id}")
            lines.append(f"- **Details:** {v.details}")
    else:
        lines.append("\n## Policy Violations")
        lines.append("\nNo policy violations detected.")

    lines.append("")
    return "\n".join(lines)


def generate_json_report(result: AuditResult) -> str:
    """Generate a JSON audit report."""
    return result.model_dump_json(indent=2)


def generate_batch_summary(batch: BatchResult) -> str:
    """Generate a markdown summary table across all batch traces."""
    lines: list[str] = []

    lines.append("# Batch Audit Summary")
    lines.append(f"\n**Traces Processed:** {len(batch.results)}")
    lines.append(f"**Traces Skipped:** {len(batch.skipped)}")

    lines.append("\n## Results")
    lines.append("")
    lines.append("| Trace | Agent | Steps | Violations | Risk Score |")
    lines.append("|-------|-------|-------|------------|------------|")
    for result in batch.results:
        lines.append(
            f"| {result.trace.trace_id} | {result.trace.agent_name} "
            f"| {len(result.trace.steps)} | {len(result.violations)} "
            f"| {result.risk_score} |"
        )

    if batch.skipped:
        lines.append("\n## Skipped Files")
        for entry in batch.skipped:
            lines.append(f"- **{entry['file']}**: {entry['error']}")

    lines.append("")
    return "\n".join(lines)


def generate_batch_json(batch: BatchResult) -> str:
    """Generate a JSON report for a batch audit."""
    return batch.model_dump_json(indent=2)


def write_report(content: str, path: Path) -> None:
    """Write report content to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
