"""Interactive replay viewer for agent audit results."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import click

from troy.models import AuditResult, StepExplanation, Violation


class ReplaySession:
    """State management for navigating an audit replay."""

    def __init__(
        self,
        result: AuditResult,
        reeval_violations: list[Violation] | None = None,
        reeval_risk_score: int | None = None,
    ):
        self.result = result
        self.current_step_index: int = 0
        self.view_mode: str = "step"  # "step" | "summary" | "violations"
        self.reeval_violations = reeval_violations
        self.reeval_risk_score = reeval_risk_score

    @property
    def total_steps(self) -> int:
        return len(self.result.trace.steps)

    @property
    def current_step(self):
        return self.result.trace.steps[self.current_step_index]

    @property
    def current_explanation(self) -> StepExplanation | None:
        step_id = self.current_step.step_id
        for exp in self.result.explanations:
            if exp.step_id == step_id:
                return exp
        return None

    @property
    def current_violations(self) -> list[Violation]:
        source = self.reeval_violations if self.reeval_violations is not None else self.result.violations
        return [v for v in source if v.step_id == self.current_step.step_id]

    @property
    def all_violations(self) -> list[Violation]:
        return self.reeval_violations if self.reeval_violations is not None else self.result.violations

    @property
    def active_risk_score(self) -> int:
        return self.reeval_risk_score if self.reeval_risk_score is not None else self.result.risk_score

    def step_has_violation(self, index: int) -> bool:
        step_id = self.result.trace.steps[index].step_id
        source = self.reeval_violations if self.reeval_violations is not None else self.result.violations
        return any(v.step_id == step_id for v in source)

    def next_step(self) -> bool:
        if self.current_step_index < self.total_steps - 1:
            self.current_step_index += 1
            return True
        return False

    def prev_step(self) -> bool:
        if self.current_step_index > 0:
            self.current_step_index -= 1
            return True
        return False

    def jump_to_violation(self, direction: int = 1) -> bool:
        """Jump to next (+1) or previous (-1) step that has a violation."""
        start = self.current_step_index + direction
        indices = range(start, self.total_steps) if direction == 1 else range(start, -1, -1)
        for i in indices:
            if self.step_has_violation(i):
                self.current_step_index = i
                return True
        return False


class ReplayRenderer:
    """Terminal output renderer using click.style()."""

    def __init__(self, session: ReplaySession):
        self.session = session

    def _risk_color(self, score: int) -> str:
        if score < 40:
            return "green"
        elif score < 70:
            return "yellow"
        return "red"

    def _severity_color(self, severity: str) -> str:
        if severity in ("critical", "high"):
            return "red"
        elif severity == "medium":
            return "yellow"
        return "cyan"

    def _progress_bar(self, width: int = 40) -> str:
        total = self.session.total_steps
        if total == 0:
            return ""
        parts = []
        for i in range(total):
            if i == self.session.current_step_index:
                char = click.style("\u25cf", fg="white", bold=True)
            elif self.session.step_has_violation(i):
                char = click.style("\u25cf", fg="red")
            else:
                char = click.style("\u2500", fg="bright_black")
            parts.append(char)
        return "[" + "".join(parts) + "]"

    def _format_json(self, data: dict, max_lines: int = 12) -> str:
        text = json.dumps(data, indent=2, default=str)
        lines = text.splitlines()
        if len(lines) > max_lines:
            lines = lines[:max_lines] + [click.style(f"  ... ({len(lines) - max_lines} more lines)", fg="bright_black")]
        return "\n".join(lines)

    def render_header(self) -> str:
        r = self.session.result
        s = self.session
        risk = s.active_risk_score
        risk_str = click.style(f"{risk}/100", fg=self._risk_color(risk), bold=True)
        violations = s.all_violations
        viol_str = click.style(str(len(violations)), fg="red" if violations else "green", bold=True)

        lines = [
            click.style("=" * 60, fg="bright_black"),
            click.style("  TROY REPLAY", fg="cyan", bold=True),
            click.style("=" * 60, fg="bright_black"),
            f"  Agent: {click.style(r.trace.agent_name, bold=True)}    Trace: {r.trace.trace_id}",
            f"  Risk Score: {risk_str}    Steps: {s.total_steps}    Violations: {viol_str}",
        ]

        if s.reeval_violations is not None:
            orig_risk = click.style(str(r.risk_score), fg=self._risk_color(r.risk_score))
            lines.append(f"  {click.style('Re-evaluated', fg='yellow')} (original risk: {orig_risk}, original violations: {len(r.violations)})")

        lines.append(f"  {self._progress_bar()}")
        lines.append(click.style("-" * 60, fg="bright_black"))
        return "\n".join(lines)

    def render_step_view(self) -> str:
        step = self.session.current_step
        idx = self.session.current_step_index
        total = self.session.total_steps

        lines = [
            click.style(f"  Step {idx + 1}/{total}", fg="cyan", bold=True)
            + "  "
            + click.style(f"[{step.type.value}]", fg="yellow")
            + f"  {step.step_id}",
            "",
            f"  {step.description}",
            "",
        ]

        # Input
        if step.input:
            lines.append(click.style("  Input:", fg="white", bold=True))
            lines.append(textwrap.indent(self._format_json(step.input), "    "))
            lines.append("")

        # Output
        if step.output:
            lines.append(click.style("  Output:", fg="white", bold=True))
            lines.append(textwrap.indent(self._format_json(step.output), "    "))
            lines.append("")

        # Violations on this step
        violations = self.session.current_violations
        if violations:
            lines.append(click.style("  \u26a0 VIOLATIONS", fg="red", bold=True))
            for v in violations:
                sev_color = self._severity_color(v.severity)
                lines.append(
                    f"    {click.style('\u2022', fg='red')} "
                    f"{click.style(f'[{v.severity.upper()}]', fg=sev_color)} "
                    f"{v.rule_description}"
                )
                if v.details:
                    lines.append(f"      {click.style(v.details, fg='bright_black')}")
            lines.append("")

        # LLM explanation
        exp = self.session.current_explanation
        if exp:
            lines.append(click.style("  Analysis", fg="green", bold=True))
            lines.append(f"    {exp.summary}")
            if exp.rationale:
                lines.append("")
                lines.append(click.style("  Rationale:", fg="green", bold=True))
                lines.append(textwrap.indent(textwrap.fill(exp.rationale, 72), "    "))
            if exp.risk_factors:
                lines.append("")
                lines.append(click.style("  Risk Factors:", fg="green", bold=True))
                for rf in exp.risk_factors:
                    lines.append(f"    \u2022 {rf}")
            if exp.data_accessed:
                lines.append("")
                lines.append(click.style("  Data Accessed:", fg="green", bold=True))
                for da in exp.data_accessed:
                    lines.append(f"    \u2022 {da}")
            if exp.alternatives:
                lines.append("")
                lines.append(click.style("  Alternatives:", fg="green", bold=True))
                for alt in exp.alternatives:
                    lines.append(f"    \u2022 {alt}")

        return "\n".join(lines)

    def render_summary_view(self) -> str:
        summary = self.session.result.summary
        if not summary:
            return click.style("  No trace summary available.", fg="bright_black")

        lines = [
            click.style("  Trace Summary", fg="cyan", bold=True),
            click.style("  " + "-" * 40, fg="bright_black"),
            "",
            click.style("  Overview:", fg="green", bold=True),
            textwrap.indent(textwrap.fill(summary.overview, 72), "    "),
        ]

        if summary.key_actions:
            lines.append("")
            lines.append(click.style("  Key Actions:", fg="green", bold=True))
            for action in summary.key_actions:
                lines.append(f"    \u2022 {action}")

        if summary.concerns:
            lines.append("")
            lines.append(click.style("  Concerns:", fg="yellow", bold=True))
            for concern in summary.concerns:
                lines.append(f"    \u2022 {click.style(concern, fg='yellow')}")

        return "\n".join(lines)

    def render_violations_view(self) -> str:
        violations = self.session.all_violations
        if not violations:
            return click.style("  No violations found.", fg="green", bold=True)

        # Group by step
        by_step: dict[str, list[Violation]] = {}
        for v in violations:
            by_step.setdefault(v.step_id, []).append(v)

        lines = [
            click.style(f"  All Violations ({len(violations)})", fg="red", bold=True),
            click.style("  " + "-" * 40, fg="bright_black"),
        ]

        for step_id, step_violations in by_step.items():
            lines.append("")
            lines.append(click.style(f"  Step {step_id}:", fg="cyan", bold=True))
            for v in step_violations:
                sev_color = self._severity_color(v.severity)
                lines.append(
                    f"    {click.style('\u2022', fg='red')} "
                    f"{click.style(f'[{v.severity.upper()}]', fg=sev_color)} "
                    f"{v.rule_description}"
                )
                if v.details:
                    lines.append(f"      {click.style(v.details, fg='bright_black')}")

        return "\n".join(lines)

    def render_footer(self) -> str:
        keys = [
            (click.style("\u2192/n", fg="cyan", bold=True), "next"),
            (click.style("\u2190/p", fg="cyan", bold=True), "prev"),
            (click.style("d", fg="cyan", bold=True), "step detail"),
            (click.style("s", fg="cyan", bold=True), "summary"),
            (click.style("v", fg="cyan", bold=True), "violations"),
            (click.style("j/k", fg="cyan", bold=True), "next/prev violation"),
            (click.style("q", fg="cyan", bold=True), "quit"),
        ]
        parts = "  ".join(f"{k} {desc}" for k, desc in keys)
        return "\n" + click.style("-" * 60, fg="bright_black") + "\n  " + parts

    def render_non_interactive(self) -> str:
        """Render the full replay to a string for non-interactive output."""
        sections = [self.render_header(), ""]

        # Summary
        sections.append(self.render_summary_view())
        sections.append("")

        # All steps
        for i in range(self.session.total_steps):
            self.session.current_step_index = i
            sections.append(click.style("=" * 60, fg="bright_black"))
            sections.append(self.render_step_view())
            sections.append("")

        # All violations
        sections.append(click.style("=" * 60, fg="bright_black"))
        sections.append(self.render_violations_view())

        return "\n".join(sections)


def run_interactive(session: ReplaySession, renderer: ReplayRenderer) -> None:
    """Main interactive loop: clear screen, render, read key, repeat."""
    while True:
        click.clear()

        output = renderer.render_header() + "\n\n"
        if session.view_mode == "step":
            output += renderer.render_step_view()
        elif session.view_mode == "summary":
            output += renderer.render_summary_view()
        elif session.view_mode == "violations":
            output += renderer.render_violations_view()
        output += renderer.render_footer()

        click.echo(output)

        try:
            ch = click.getchar()
        except (KeyboardInterrupt, EOFError):
            break

        if ch in ("q", "Q"):
            break
        elif ch in ("n", "\x1b[C"):  # n or right arrow
            session.view_mode = "step"
            session.next_step()
        elif ch in ("p", "\x1b[D"):  # p or left arrow
            session.view_mode = "step"
            session.prev_step()
        elif ch == "d":
            session.view_mode = "step"
        elif ch == "s":
            session.view_mode = "summary"
        elif ch == "v":
            session.view_mode = "violations"
        elif ch == "j":
            session.view_mode = "step"
            session.jump_to_violation(direction=1)
        elif ch == "k":
            session.view_mode = "step"
            session.jump_to_violation(direction=-1)
