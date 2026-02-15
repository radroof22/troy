# agent-audit

CLI tool that audits agent execution traces against configurable policies. It ingests trace JSON, builds an execution graph, generates semantic explanations via an LLM, evaluates policy compliance, and outputs structured audit reports.

## Installation

Requires Python 3.11+.

```bash
uv sync
```

Set up your LLM provider credentials:

```bash
cp .env.example .env
# Edit .env with your API key, base URL, and model
```

Or pass them as flags / environment variables (see [Configuration](#configuration)).

## Quick Start

```bash
# Audit a single trace
uv run agent-audit audit traces/agent_run.json examples/policy.json

# Batch audit every trace in a directory
uv run agent-audit audit-batch traces/ examples/policy.json

# Replay a previous audit interactively
uv run agent-audit replay logs/2026-02-15/trace3/audit.json

# Replay with a different policy (no LLM calls, instant)
uv run agent-audit replay logs/2026-02-15/trace3/audit.json --policy examples/policy.json

# Dump replay to stdout for piping / CI
uv run agent-audit replay logs/2026-02-15/trace3/audit.json --policy examples/policy.json --no-interactive
```

## Commands

### `audit` — Single trace audit

Runs the full pipeline: graph building, LLM explanation, policy evaluation, scoring, and reporting.

```bash
uv run agent-audit audit <trace_file> <policy_file> [OPTIONS]
```

| Option | Env var | Default | Description |
|---|---|---|---|
| `--output`, `-o` | — | `logs/{date}/report.md` | Markdown report output path |
| `--json-output`, `-j` | — | `logs/{date}/audit.json` | JSON report output path |
| `--model`, `-m` | `AGENT_AUDIT_MODEL` | `gpt-4o-mini` | LLM model name |
| `--base-url` | `OPENAI_BASE_URL` | — | API base URL |
| `--api-key` | `OPENAI_API_KEY` | — | API key |

**Output files:**

```
logs/{date}/
├── report.md              # Markdown audit report
├── audit.json             # Full audit result (replayable)
└── llm_responses/         # Raw LLM responses for audit trail
    ├── step_s1.txt
    ├── step_s2.txt
    └── trace_summary.txt
```

### `audit-batch` — Batch audit

Audits all `.json` trace files in a directory concurrently (up to 5 in parallel).

```bash
uv run agent-audit audit-batch <trace_dir> <policy_file> [OPTIONS]
```

Same options as `audit` (model, base-url, api-key). Generates per-trace reports plus a batch summary:

```
logs/{date}/
├── summary.md             # Table of all traces with violation counts
├── batch.json             # Full batch result
├── trace1/
│   ├── report.md
│   └── audit.json
└── trace2/
    ├── report.md
    └── audit.json
```

### `replay` — Interactive audit replay

Replays a previously-generated `audit.json` in the terminal. No LLM calls needed.

```bash
uv run agent-audit replay <audit_file> [OPTIONS]
```

| Option | Description |
|---|---|
| `--policy <file>` | Re-evaluate with a different policy file (pure computation, instant) |
| `--no-interactive` | Dump full replay to stdout instead of interactive mode |

**Interactive controls:**

| Key | Action |
|---|---|
| `→` / `n` | Next step |
| `←` / `p` | Previous step |
| `d` | Step detail view |
| `s` | Trace summary view |
| `v` | Violations view |
| `j` / `k` | Jump to next / previous violation |
| `q` | Quit |

## How It Works

1. **Ingestion** — Loads and validates trace JSON using Pydantic models
2. **Graph Building** — Constructs a directed execution graph (NetworkX) representing step dependencies
3. **Explanation** — Sends each step + surrounding context to an LLM to infer *why* the agent made each decision, what data it accessed, what alternatives existed, and what risks are present
4. **Policy Evaluation** — Evaluates Python expression conditions against each step with full cross-step context
5. **Scoring** — Computes a risk score: `min(100, sum(weights of violated rules))`
6. **Reporting** — Generates markdown and JSON audit reports

## Trace Format

agent-audit consumes traces — it doesn't generate them. Your agent logging system needs to produce JSON in this format:

```json
{
  "trace_id": "trace-001",
  "agent_name": "my-agent",
  "steps": [
    {
      "step_id": "step-1",
      "type": "tool_call",
      "description": "Fetch user profile from database",
      "input": { "user_id": "usr_882" },
      "output": { "name": "Jane Doe", "email": "jane@example.com" },
      "metadata": { "data_classification": "pii" },
      "timestamp": "2026-02-15T11:15:00Z",
      "parent_step_id": null
    }
  ],
  "metadata": {
    "environment": "production",
    "permission_level": "user"
  }
}
```

### Step fields

| Field | Required | Description |
|---|---|---|
| `step_id` | Yes | Unique identifier referenced in violations and reports |
| `type` | Yes | One of `llm_call`, `tool_call`, `decision`, `observation` |
| `description` | Yes | Human-readable description of what the step does |
| `input` | Yes | Full inputs — prompts, tool args, queries. Without this, auditing is blind |
| `output` | Yes | Full outputs — responses, return values. Needed to verify what actually happened |
| `metadata` | No | Labels like `data_classification`, `network_zone`, `permission_level`, `requires_approval`. Used by policy rules |
| `timestamp` | No | ISO 8601 timestamp for ordering and timeline analysis |
| `parent_step_id` | No | For nested/branching execution (e.g. sub-agent calls) |

### Metadata conventions

Policy rules reference these metadata keys. Annotate your steps with them to enable detection:

| Key | Values | Used by |
|---|---|---|
| `data_classification` | `pii`, `internal`, `public` | PII exfiltration detection |
| `network_zone` | `external`, `internal` | External data transmission detection |
| `permission_level` | `user`, `admin` | Privilege escalation detection |
| `requires_approval` | `true` / `false` | Mandatory approval checks |
| `approval_token` | token string or `null` | Approval verification |
| `category` | `communication`, etc. | Communication channel controls |

The more context you log per step, the better the audit. At minimum: capture full inputs and outputs. The auditor infers *why* the agent made each decision by analyzing the execution chain — what came before, what came after, and how data flowed between steps.

## Policy Format

Policies are JSON files containing a list of rules. Each rule has a `condition` — a Python expression that returns `True` when the rule is **violated**.

```json
{
  "policy_id": "my-policy",
  "description": "Safety policy for production agents",
  "rules": [
    {
      "rule_id": "pii-exfiltration-protection",
      "description": "Detects PII handling followed by transmission to external endpoints",
      "condition": "get(step, 'metadata.data_classification') == 'pii' and any_next(lambda s: s['type'] == 'tool_call' and get(s, 'metadata.network_zone') == 'external')",
      "severity": "critical",
      "weight": 50
    }
  ]
}
```

### Rule fields

| Field | Required | Default | Description |
|---|---|---|---|
| `rule_id` | Yes | — | Unique identifier for the rule |
| `description` | Yes | — | Human-readable description shown in reports |
| `condition` | Yes | — | Python expression (see below). `True` = violated |
| `severity` | No | `medium` | `critical`, `high`, `medium`, `low` |
| `weight` | No | `10` | Points added to risk score when violated |

### Writing conditions

Conditions are Python expressions evaluated per-step with these variables and helpers in scope:

**Variables:**

| Variable | Type | Description |
|---|---|---|
| `step` | `dict` | Current step being evaluated |
| `steps` | `list[dict]` | All steps in the trace |
| `step_index` | `int` | Current step's index |
| `prev_steps` | `list[dict]` | Steps before the current one |
| `next_steps` | `list[dict]` | Steps after the current one |
| `trace` | `dict` | Trace-level info: `trace_id`, `agent_name`, `metadata` |
| `agent` | `dict` | Agent info: `name`, `metadata` (from trace) |

**Helper functions:**

| Function | Description |
|---|---|
| `get(d, 'a.b.c', default)` | Safe nested dict access via dot-separated path. Returns `default` (or `None`) if any key is missing |
| `matches(text, pattern)` | Case-insensitive regex search. Returns truthy if pattern is found |
| `any_step(fn)` | `True` if `fn(step_dict)` is true for any step in the trace |
| `any_next(fn)` | `True` if `fn(step_dict)` is true for any step after the current one |
| `any_prev(fn)` | `True` if `fn(step_dict)` is true for any step before the current one |

**Example conditions:**

```python
# PII data followed by an external tool call
"get(step, 'metadata.data_classification') == 'pii' and any_next(lambda s: s['type'] == 'tool_call' and get(s, 'metadata.network_zone') == 'external')"

# Prompt injection patterns in step input
"matches(str(step.get('input', {})), r'ignore previous instructions|system update|run as admin')"

# Raw SQL in tool call inputs
"step['type'] == 'tool_call' and matches(get(step, 'input.query', ''), r'SELECT|INSERT|UPDATE|DELETE|DROP|UNION')"

# Missing approval token on steps that require approval
"get(step, 'metadata.requires_approval') is True and get(step, 'metadata.approval_token') is None"

# Non-admin agent accessing admin-level step
"get(agent, 'metadata.permission_level') != 'admin' and get(step, 'metadata.permission_level') == 'admin'"

# Any tool call categorized as communication
"step['type'] == 'tool_call' and get(step, 'metadata.category') == 'communication'"
```

Malformed or erroring conditions are silently skipped — they won't crash the engine.

## Configuration

LLM settings can be configured three ways (in order of precedence):

1. **CLI flags:** `--model`, `--base-url`, `--api-key`
2. **Environment variables:** `AGENT_AUDIT_MODEL`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`
3. **`.env` file** (loaded automatically via python-dotenv)

The tool uses the OpenAI client library, so it works with any OpenAI-compatible API (OpenAI, Azure, local models via LiteLLM/Ollama, etc).

## Testing

```bash
uv run pytest tests/ -v
```

## Roadmap

- **Drift detection** — Detect when agent behavior drifts from established baselines
- **Regression comparison** — Compare audit results across trace versions to catch regressions
- **Structured semantic diffing** — Diff two traces at the semantic level, not just textual
- **Risk dashboards** — Visual dashboard for risk scores and violation trends over time
- **RBAC** — Role-based access control for multi-user audit workflows
- **SOC 2 compliance** — Built-in policy templates and reporting aligned with SOC 2 requirements
