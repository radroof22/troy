"""Tests for the `troy policies` CLI subcommands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from troy.cli import main


@pytest.fixture
def runner():
    return CliRunner()


class TestPoliciesList:
    def test_lists_all_policies(self, runner):
        result = runner.invoke(main, ["policies", "list"])
        assert result.exit_code == 0
        assert "minimal" in result.output
        assert "soc2" in result.output
        assert "hipaa" in result.output
        assert "owasp" in result.output
        assert "agent-safety" in result.output
        assert "data-protection" in result.output
        assert "safe-browsing" in result.output

    def test_shows_rule_counts(self, runner):
        result = runner.invoke(main, ["policies", "list"])
        assert "rules" in result.output

    def test_shows_descriptions(self, runner):
        result = runner.invoke(main, ["policies", "list"])
        # SOC 2 description mentions Trust Services
        assert "Trust Services" in result.output


class TestPoliciesShow:
    def test_show_soc2(self, runner):
        result = runner.invoke(main, ["policies", "show", "soc2"])
        assert result.exit_code == 0
        assert "soc2-access-control" in result.output
        assert "soc2-change-approval" in result.output
        assert "10 rules total" in result.output

    def test_show_hipaa(self, runner):
        result = runner.invoke(main, ["policies", "show", "hipaa"])
        assert result.exit_code == 0
        assert "hipaa-phi-access-control" in result.output
        assert "8 rules total" in result.output

    def test_show_minimal(self, runner):
        result = runner.invoke(main, ["policies", "show", "minimal"])
        assert result.exit_code == 0
        assert "3 rules total" in result.output

    def test_show_not_found(self, runner):
        result = runner.invoke(main, ["policies", "show", "nonexistent"])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_show_includes_severities(self, runner):
        result = runner.invoke(main, ["policies", "show", "soc2"])
        assert "critical" in result.output
        assert "high" in result.output

    def test_show_includes_file_path(self, runner):
        result = runner.invoke(main, ["policies", "show", "soc2"])
        assert "File:" in result.output
        assert "soc2.json" in result.output


class TestPoliciesCopy:
    def test_copy_creates_file(self, runner, tmp_path):
        dest = tmp_path / "my_policy.json"
        result = runner.invoke(main, ["policies", "copy", "minimal", "-o", str(dest)])
        assert result.exit_code == 0
        assert dest.exists()
        data = json.loads(dest.read_text())
        assert data["policy_id"] == "minimal"
        assert len(data["rules"]) == 3

    def test_copy_default_name(self, runner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(main, ["policies", "copy", "soc2"])
        assert result.exit_code == 0
        assert (tmp_path / "soc2.json").exists()

    def test_copy_not_found(self, runner, tmp_path):
        result = runner.invoke(main, ["policies", "copy", "nonexistent", "-o", str(tmp_path / "out.json")])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_copy_overwrite_prompts(self, runner, tmp_path):
        dest = tmp_path / "policy.json"
        dest.write_text("{}")
        # Decline overwrite
        result = runner.invoke(main, ["policies", "copy", "minimal", "-o", str(dest)], input="n\n")
        assert result.exit_code != 0  # Aborted

    def test_copy_produces_valid_policy(self, runner, tmp_path):
        dest = tmp_path / "policy.json"
        runner.invoke(main, ["policies", "copy", "hipaa", "-o", str(dest)])
        # Should work with `check`
        result = runner.invoke(main, ["check", str(dest), "-a", "safe_tool"])
        assert result.exit_code in (0, 2)
        json.loads(result.output)  # Valid JSON


class TestPoliciesInit:
    def test_init_default_minimal(self, runner, tmp_path):
        dest = tmp_path / "policy.json"
        result = runner.invoke(main, ["policies", "init", "-o", str(dest)])
        assert result.exit_code == 0
        data = json.loads(dest.read_text())
        assert data["policy_id"] == "custom-policy"
        assert len(data["rules"]) == 3  # minimal has 3 rules
        assert "minimal" in data["description"]

    def test_init_single_template(self, runner, tmp_path):
        dest = tmp_path / "policy.json"
        result = runner.invoke(main, ["policies", "init", "-t", "soc2", "-o", str(dest)])
        assert result.exit_code == 0
        data = json.loads(dest.read_text())
        assert len(data["rules"]) == 10

    def test_init_multiple_templates(self, runner, tmp_path):
        dest = tmp_path / "policy.json"
        result = runner.invoke(main, ["policies", "init", "-t", "soc2", "-t", "hipaa", "-o", str(dest)])
        assert result.exit_code == 0
        data = json.loads(dest.read_text())
        # 10 soc2 + 8 hipaa = 18 (no overlapping rule_ids)
        assert len(data["rules"]) == 18
        assert "soc2" in data["description"]
        assert "hipaa" in data["description"]

    def test_init_deduplicates_rules(self, runner, tmp_path):
        """Combining a policy with itself should not duplicate rules."""
        dest = tmp_path / "policy.json"
        result = runner.invoke(main, ["policies", "init", "-t", "minimal", "-t", "minimal", "-o", str(dest)])
        assert result.exit_code == 0
        data = json.loads(dest.read_text())
        assert len(data["rules"]) == 3

    def test_init_bad_template(self, runner, tmp_path):
        dest = tmp_path / "policy.json"
        result = runner.invoke(main, ["policies", "init", "-t", "nonexistent", "-o", str(dest)])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_init_output_works_with_check(self, runner, tmp_path):
        dest = tmp_path / "policy.json"
        runner.invoke(main, ["policies", "init", "-t", "agent_safety", "-o", str(dest)])
        result = runner.invoke(main, ["check", str(dest), "-a", "delete_records"])
        assert result.exit_code == 2
        data = json.loads(result.output)
        assert data["allowed"] is False

    def test_init_overwrite_prompts(self, runner, tmp_path):
        dest = tmp_path / "policy.json"
        dest.write_text("{}")
        result = runner.invoke(main, ["policies", "init", "-o", str(dest)], input="n\n")
        assert result.exit_code != 0
