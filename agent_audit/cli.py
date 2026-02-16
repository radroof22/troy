"""CLI entry point for agent-audit."""

import asyncio
import logging
from datetime import date
from pathlib import Path

import click
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()

from agent_audit.explainers.llm import LLMExplainer
from agent_audit.graph.builder import build_execution_graph
from agent_audit.ingestion.loader import load_trace
from agent_audit.models import AuditResult, BatchResult, Trace
from agent_audit.policy.engine import PolicyRule, compute_risk_score, evaluate_policy, load_policy
from agent_audit.reporting.reporter import (
    generate_batch_json,
    generate_batch_summary,
    generate_json_report,
    generate_markdown_report,
    write_report,
)


def _write_raw_responses(result: AuditResult, trace_log_dir: Path) -> None:
    """Write raw LLM responses to disk for audit trail."""
    resp_dir = trace_log_dir / "llm_responses"
    resp_dir.mkdir(parents=True, exist_ok=True)

    for explanation in result.explanations:
        (resp_dir / f"step_{explanation.step_id}.txt").write_text(explanation.raw_response)

    if result.summary:
        (resp_dir / "trace_summary.txt").write_text(result.summary.raw_response)


def _run_audit(trace: Trace, rules: list[PolicyRule], explainer: LLMExplainer | None) -> AuditResult:
    """Run the core audit pipeline on a single trace."""
    # Build execution graph
    click.echo("Building execution graph...")
    graph = build_execution_graph(trace)
    click.echo(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")

    # Generate explanations (skip if no explainer)
    explanations = []
    summary = None
    if explainer:
        click.echo("Generating explanations...")
        for i, step in enumerate(trace.steps):
            click.echo(f"  Explaining step {step.step_id}...")
            preceding = trace.steps[:i]
            following = trace.steps[i + 1 :]
            explanation = explainer.explain_step(step, preceding, following)
            explanations.append(explanation)

        click.echo("Generating trace summary...")
        summary = explainer.summarize_trace(trace)
    else:
        click.echo("Skipping LLM explanations (--no-explain).")

    # Evaluate policy
    violations = evaluate_policy(trace, rules)
    risk_score = compute_risk_score(violations, rules)
    click.echo(f"Found {len(violations)} violation(s). Risk score: {risk_score}/100.")

    return AuditResult(
        trace=trace,
        explanations=explanations,
        summary=summary,
        violations=violations,
        risk_score=risk_score,
    )


@click.group()
def main():
    """Agent Audit — analyze agent execution traces against policies."""


@main.command()
@click.argument("trace_file", type=click.Path(exists=True, path_type=Path))
@click.argument("policy_file", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Output markdown report path.")
@click.option("--json-output", "-j", type=click.Path(path_type=Path), default=None, help="Output JSON report path.")
@click.option("--model", "-m", envvar="AGENT_AUDIT_MODEL", default="gpt-4o-mini", help="Model name (or set AGENT_AUDIT_MODEL).")
@click.option("--base-url", envvar="OPENAI_BASE_URL", default=None, help="API base URL (or set OPENAI_BASE_URL).")
@click.option("--api-key", envvar="OPENAI_API_KEY", default=None, help="API key (or set OPENAI_API_KEY).")
@click.option("--no-explain", is_flag=True, default=False, help="Skip LLM explanations (policy evaluation and scoring only, no API key needed).")
def audit(trace_file: Path, policy_file: Path, output: Path | None, json_output: Path | None, model: str, base_url: str | None, api_key: str | None, no_explain: bool):
    """Audit an agent execution trace against a policy file."""
    # 1. Ingest trace
    click.echo(f"Loading trace from {trace_file}...")
    trace = load_trace(trace_file)
    click.echo(f"Loaded trace '{trace.trace_id}' with {len(trace.steps)} steps.")

    # 2. Load policy
    click.echo(f"Evaluating policy from {policy_file}...")
    rules = load_policy(policy_file)

    # 3. Run audit pipeline
    explainer = None
    if not no_explain:
        explainer = LLMExplainer(model=model, base_url=base_url, api_key=api_key)
    result = _run_audit(trace, rules, explainer)

    # 4. Generate reports — default to ./logs/{date}/
    log_dir = Path("logs") / date.today().isoformat()
    if not output:
        output = log_dir / "report.md"
    if not json_output:
        json_output = log_dir / "audit.json"

    md_report = generate_markdown_report(result)
    write_report(md_report, output)
    click.echo(f"Markdown report written to {output}")

    json_report = generate_json_report(result)
    write_report(json_report, json_output)
    click.echo(f"JSON report written to {json_output}")

    if not no_explain:
        _write_raw_responses(result, log_dir)
        click.echo(f"Raw LLM responses written to {log_dir / 'llm_responses'}")


async def _run_audit_async(trace: Trace, rules: list[PolicyRule], explainer: LLMExplainer | None) -> AuditResult:
    """Run the core audit pipeline on a single trace, using async API calls."""
    graph = build_execution_graph(trace)

    explanations = []
    summary = None
    if explainer:
        tasks = []
        for i, step in enumerate(trace.steps):
            preceding = trace.steps[:i]
            following = trace.steps[i + 1 :]
            tasks.append(explainer.aexplain_step(step, preceding, following))
        explanations = list(await asyncio.gather(*tasks))
        summary = await explainer.asummarize_trace(trace)

    violations = evaluate_policy(trace, rules)
    risk_score = compute_risk_score(violations, rules)

    return AuditResult(
        trace=trace,
        explanations=explanations,
        summary=summary,
        violations=violations,
        risk_score=risk_score,
    )


async def _run_batch(
    trace_dir: Path,
    policy_file: Path,
    model: str,
    base_url: str | None,
    api_key: str | None,
    no_explain: bool = False,
):
    """Async batch audit: run all traces concurrently with a semaphore."""
    trace_files = sorted(trace_dir.glob("*.json"))
    if not trace_files:
        click.echo(f"No .json files found in {trace_dir}")
        return

    # Filter out the policy file itself if it's in the same directory
    policy_resolved = policy_file.resolve()
    trace_files = [f for f in trace_files if f.resolve() != policy_resolved]

    click.echo(f"Found {len(trace_files)} JSON file(s) in {trace_dir}")

    # Load policy once
    click.echo(f"Loading policy from {policy_file}...")
    rules = load_policy(policy_file)

    # Create explainer once (shared async client) — or None if skipping
    explainer = None
    if not no_explain:
        explainer = LLMExplainer(model=model, base_url=base_url, api_key=api_key)
    else:
        click.echo("Skipping LLM explanations (--no-explain).")

    log_dir = Path("logs") / date.today().isoformat()
    sem = asyncio.Semaphore(5)

    # Load all traces up front, separating successes from failures
    loaded: list[tuple[Path, Trace]] = []
    skipped: list[dict] = []
    for trace_file in trace_files:
        try:
            trace = load_trace(trace_file)
            loaded.append((trace_file, trace))
        except Exception as e:
            click.echo(f"  WARNING: Skipping {trace_file.name} — {e}")
            skipped.append({"file": trace_file.name, "error": str(e)})

    async def _process_one(trace_file: Path, trace: Trace) -> AuditResult | None:
        async with sem:
            click.echo(f"Processing {trace_file.name} ({len(trace.steps)} steps)...")
            try:
                result = await _run_audit_async(trace, rules, explainer)
            except Exception as e:
                logger.error("Failed to audit %s (trace %s): %s", trace_file.name, trace.trace_id, e)
                skipped.append({"file": trace_file.name, "error": str(e)})
                return None
            click.echo(f"  {trace_file.name}: {len(result.violations)} violation(s), risk {result.risk_score}/100")

            # Write per-trace reports
            trace_log_dir = log_dir / trace_file.stem
            md_report = generate_markdown_report(result)
            write_report(md_report, trace_log_dir / "report.md")

            json_report = generate_json_report(result)
            write_report(json_report, trace_log_dir / "audit.json")

            if not no_explain:
                _write_raw_responses(result, trace_log_dir)
            return result

    raw_results = await asyncio.gather(*[_process_one(tf, t) for tf, t in loaded])
    results = [r for r in raw_results if r is not None]

    # Write batch summary
    batch = BatchResult(results=results, skipped=skipped)

    summary_md = generate_batch_summary(batch)
    write_report(summary_md, log_dir / "summary.md")

    batch_json = generate_batch_json(batch)
    write_report(batch_json, log_dir / "batch.json")

    click.echo(f"\n{'='*60}")
    click.echo(f"Batch complete: {len(results)} processed, {len(skipped)} skipped.")
    click.echo(f"Summary written to {log_dir / 'summary.md'}")
    click.echo(f"Batch JSON written to {log_dir / 'batch.json'}")


@main.command("audit-batch")
@click.argument("trace_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("policy_file", type=click.Path(exists=True, path_type=Path))
@click.option("--model", "-m", envvar="AGENT_AUDIT_MODEL", default="gpt-4o-mini", help="Model name (or set AGENT_AUDIT_MODEL).")
@click.option("--base-url", envvar="OPENAI_BASE_URL", default=None, help="API base URL (or set OPENAI_BASE_URL).")
@click.option("--api-key", envvar="OPENAI_API_KEY", default=None, help="API key (or set OPENAI_API_KEY).")
@click.option("--no-explain", is_flag=True, default=False, help="Skip LLM explanations (policy evaluation and scoring only, no API key needed).")
def audit_batch(trace_dir: Path, policy_file: Path, model: str, base_url: str | None, api_key: str | None, no_explain: bool):
    """Audit all traces in a directory against a policy file."""
    asyncio.run(_run_batch(trace_dir, policy_file, model, base_url, api_key, no_explain=no_explain))


@main.command()
@click.argument("audit_file", type=click.Path(exists=True, path_type=Path))
@click.option("--policy", type=click.Path(exists=True, path_type=Path), default=None, help="Re-evaluate with a different policy (no LLM calls).")
@click.option("--no-interactive", is_flag=True, default=False, help="Dump full replay to stdout (pipeable).")
def replay(audit_file: Path, policy: Path | None, no_interactive: bool):
    """Interactively replay a previously-generated audit."""
    from agent_audit.replay.viewer import ReplayRenderer, ReplaySession, run_interactive

    click.echo(f"Loading audit from {audit_file}...")
    result = AuditResult.model_validate_json(audit_file.read_text())
    click.echo(f"Loaded trace '{result.trace.trace_id}' — {len(result.trace.steps)} steps, {len(result.violations)} violation(s).")

    reeval_violations = None
    reeval_risk_score = None
    if policy:
        click.echo(f"Re-evaluating with policy {policy}...")
        rules = load_policy(policy)
        reeval_violations = evaluate_policy(result.trace, rules)
        reeval_risk_score = compute_risk_score(reeval_violations, rules)
        click.echo(f"Re-evaluation: {len(reeval_violations)} violation(s), risk {reeval_risk_score}/100.")

    session = ReplaySession(result, reeval_violations=reeval_violations, reeval_risk_score=reeval_risk_score)
    renderer = ReplayRenderer(session)

    if no_interactive:
        click.echo(renderer.render_non_interactive())
    else:
        run_interactive(session, renderer)


@main.command()
@click.argument("policy_file", type=click.Path(exists=True, path_type=Path))
@click.option("--action", "-a", required=True, help="Tool or action name to evaluate.")
@click.option("--input", "-i", "input_json", default=None, help="JSON string of input data.")
@click.option("--step-type", "-t", type=click.Choice(["tool_call", "llm_call", "decision", "observation"]), default="tool_call", help="Step type.")
@click.option("--mode", type=click.Choice(["enforce", "monitor", "dry-run"]), default="enforce", help="Guard mode.")
@click.option("--agent-name", default="agent", help="Agent name for policy context.")
@click.option("--metadata", "metadata_json", default=None, help="JSON string of step metadata.")
@click.option("--agent-metadata", "agent_metadata_json", default=None, help="JSON string of agent metadata.")
def check(
    policy_file: Path,
    action: str,
    input_json: str | None,
    step_type: str,
    mode: str,
    agent_name: str,
    metadata_json: str | None,
    agent_metadata_json: str | None,
):
    """Evaluate a single action against a policy (no LLM needed).

    Returns JSON with {allowed, violations, risk_score, step_id, mode}.
    Exit code 0 if allowed, 2 if blocked.

    \b
    Examples:
        agent-audit check policy.json -a search -i '{"query": "SELECT * FROM users"}'
        agent-audit check policy.json -a send_email --mode monitor
        agent-audit check policy.json -a bash --metadata '{"permission_level": "admin"}'
    """
    import json as json_mod

    from agent_audit.guard.core import AgentAuditGuard
    from agent_audit.models import StepType

    input_data = None
    if input_json:
        try:
            input_data = json_mod.loads(input_json)
        except json_mod.JSONDecodeError as e:
            raise click.BadParameter(f"Invalid JSON for --input: {e}") from e

    metadata = None
    if metadata_json:
        try:
            metadata = json_mod.loads(metadata_json)
        except json_mod.JSONDecodeError as e:
            raise click.BadParameter(f"Invalid JSON for --metadata: {e}") from e

    agent_metadata = None
    if agent_metadata_json:
        try:
            agent_metadata = json_mod.loads(agent_metadata_json)
        except json_mod.JSONDecodeError as e:
            raise click.BadParameter(f"Invalid JSON for --agent-metadata: {e}") from e

    guard = AgentAuditGuard(
        policy=policy_file,
        agent_name=agent_name,
        mode=mode,
        agent_metadata=agent_metadata,
    )

    stype = StepType(step_type)
    decision = guard.check(
        action=action,
        input=input_data,
        metadata=metadata,
        step_type=stype,
    )

    output = {
        "allowed": decision.allowed,
        "step_id": decision.step_id,
        "risk_score": decision.risk_score,
        "mode": decision.mode,
        "violations": [
            {
                "rule_id": v.rule_id,
                "rule_description": v.rule_description,
                "severity": v.severity,
                "details": v.details,
            }
            for v in decision.violations
        ],
    }

    click.echo(json_mod.dumps(output, indent=2))
    raise SystemExit(0 if decision.allowed else 2)


def _policies_dir() -> Path:
    """Return the path to the bundled policy templates."""
    return Path(__file__).parent / "policies"


@main.group("policies")
def policies_group():
    """Browse and use bundled policy templates."""


@policies_group.command("list")
def policies_list():
    """List all available policy templates.

    \b
    Example:
        agent-audit policies list
    """
    import json as json_mod

    policies_dir = _policies_dir()
    if not policies_dir.exists():
        click.echo("No bundled policies found.")
        return

    files = sorted(policies_dir.glob("*.json"))
    if not files:
        click.echo("No bundled policies found.")
        return

    for f in files:
        data = json_mod.loads(f.read_text())
        pid = data.get("policy_id", f.stem)
        desc = data.get("description", "")
        n_rules = len(data.get("rules", []))
        click.echo(f"  {pid:<25} {n_rules:>2} rules  — {desc}")


@policies_group.command("show")
@click.argument("name")
def policies_show(name: str):
    """Show details of a policy template: description, rules, severities.

    NAME is the policy id (e.g. 'soc2', 'hipaa', 'minimal').

    \b
    Example:
        agent-audit policies show soc2
    """
    import json as json_mod

    policies_dir = _policies_dir()
    policy_file = policies_dir / f"{name}.json"
    if not policy_file.exists():
        # Try matching by policy_id
        for f in policies_dir.glob("*.json"):
            data = json_mod.loads(f.read_text())
            if data.get("policy_id") == name:
                policy_file = f
                break
        else:
            click.echo(f"Policy '{name}' not found. Run 'agent-audit policies list' to see available policies.")
            raise SystemExit(1)

    data = json_mod.loads(policy_file.read_text())
    click.echo(f"\n  {data.get('policy_id', policy_file.stem)}")
    click.echo(f"  {data.get('description', '')}\n")

    rules = data.get("rules", [])
    for r in rules:
        sev = r.get("severity", "medium")
        sev_color = {"critical": "red", "high": "yellow", "medium": "cyan", "low": "green"}.get(sev, "white")
        sev_label = click.style(f"[{sev}]", fg=sev_color)
        click.echo(f"  {sev_label:>20}  {r['rule_id']}")
        click.echo(f"              {r['description']}")

    click.echo(f"\n  {len(rules)} rules total")
    click.echo(f"  File: {policy_file}\n")


@policies_group.command("copy")
@click.argument("name")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Destination path (default: ./<name>.json).")
def policies_copy(name: str, output: Path | None):
    """Copy a policy template to your project for customization.

    NAME is the policy id (e.g. 'soc2', 'hipaa', 'minimal').

    \b
    Examples:
        agent-audit policies copy soc2
        agent-audit policies copy hipaa -o my_policy.json
    """
    import shutil

    policies_dir = _policies_dir()
    policy_file = policies_dir / f"{name}.json"
    if not policy_file.exists():
        import json as json_mod
        for f in policies_dir.glob("*.json"):
            data = json_mod.loads(f.read_text())
            if data.get("policy_id") == name:
                policy_file = f
                break
        else:
            click.echo(f"Policy '{name}' not found. Run 'agent-audit policies list' to see available policies.")
            raise SystemExit(1)

    dest = output or Path(f"{name}.json")
    if dest.exists():
        click.confirm(f"{dest} already exists. Overwrite?", abort=True)

    shutil.copy2(policy_file, dest)
    click.echo(f"Copied {policy_file.name} → {dest}")
    click.echo(f"Edit {dest} to customize rules for your project.")


@policies_group.command("init")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=Path("policy.json"), help="Destination path (default: ./policy.json).")
@click.option("--template", "-t", multiple=True, help="Policy template(s) to include (e.g. --template soc2 --template owasp_llm_top10). Default: minimal.")
def policies_init(output: Path, template: tuple[str, ...]):
    """Create a new policy file by combining one or more templates.

    \b
    Examples:
        agent-audit policies init
        agent-audit policies init -t soc2 -t hipaa -o enterprise_policy.json
        agent-audit policies init -t owasp_llm_top10 -t safe_browsing
    """
    import json as json_mod

    policies_dir = _policies_dir()
    templates = template or ("minimal",)

    combined_rules = []
    descriptions = []
    seen_rule_ids: set[str] = set()

    for name in templates:
        policy_file = policies_dir / f"{name}.json"
        if not policy_file.exists():
            click.echo(f"Policy '{name}' not found. Run 'agent-audit policies list' to see available policies.")
            raise SystemExit(1)
        data = json_mod.loads(policy_file.read_text())
        descriptions.append(data.get("description", name))
        for rule in data.get("rules", []):
            if rule["rule_id"] not in seen_rule_ids:
                combined_rules.append(rule)
                seen_rule_ids.add(rule["rule_id"])

    if output.exists():
        click.confirm(f"{output} already exists. Overwrite?", abort=True)

    result = {
        "policy_id": "custom-policy",
        "description": "Combined policy from: " + ", ".join(templates),
        "rules": combined_rules,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json_mod.dumps(result, indent=2) + "\n")
    click.echo(f"Created {output} with {len(combined_rules)} rules from {len(templates)} template(s):")
    for t in templates:
        click.echo(f"  - {t}")
    click.echo(f"\nEdit {output} to customize rules for your project.")


if __name__ == "__main__":
    main()
