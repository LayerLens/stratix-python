from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from layerlens.cli._app import cli

from .conftest import _make_runner


class TestTraceCommands:
    """Test trace CLI commands."""

    @pytest.fixture
    def runner(self):
        return _make_runner()

    @pytest.fixture
    def mock_traces(self):
        trace = Mock()
        trace.id = "trace-123"
        trace.created_at = "2026-01-01T00:00:00Z"
        trace.filename = "test.jsonl"
        trace.evaluations_count = 2
        # Make to_dict work
        trace.model_dump.return_value = {
            "id": "trace-123",
            "created_at": "2026-01-01T00:00:00Z",
            "filename": "test.jsonl",
            "evaluations_count": 2,
        }
        return trace

    @patch("layerlens.cli.commands.trace.get_client")
    def test_trace_list(self, mock_get_client, runner, mock_traces):
        """trace list displays traces in table format."""
        client = Mock()
        resp = Mock()
        resp.traces = [mock_traces]
        resp.count = 1
        resp.total_count = 1
        client.traces.get_many.return_value = resp
        mock_get_client.return_value = client

        result = runner.invoke(cli, ["trace", "list"], env={"LAYERLENS_STRATIX_API_KEY": "test"})

        assert result.exit_code == 0
        assert "trace-123" in result.output

    @patch("layerlens.cli.commands.trace.get_client")
    def test_trace_list_empty(self, mock_get_client, runner):
        """trace list shows message when no traces found."""
        client = Mock()
        client.traces.get_many.return_value = Mock(traces=[])
        mock_get_client.return_value = client

        result = runner.invoke(cli, ["trace", "list"], env={"LAYERLENS_STRATIX_API_KEY": "test"})

        assert result.exit_code == 0
        assert "No traces found" in result.output

    @patch("layerlens.cli.commands.trace.get_client")
    def test_trace_get(self, mock_get_client, runner, mock_traces):
        """trace get displays a single trace."""
        client = Mock()
        client.traces.get.return_value = mock_traces
        mock_get_client.return_value = client

        result = runner.invoke(cli, ["trace", "get", "trace-123"], env={"LAYERLENS_STRATIX_API_KEY": "test"})

        assert result.exit_code == 0
        assert "trace-123" in result.output

    @patch("layerlens.cli.commands.trace.get_client")
    def test_trace_get_not_found(self, mock_get_client, runner):
        """trace get exits with error when trace not found."""
        client = Mock()
        client.traces.get.return_value = None
        mock_get_client.return_value = client

        result = runner.invoke(cli, ["trace", "get", "nonexistent"], env={"LAYERLENS_STRATIX_API_KEY": "test"})

        assert result.exit_code != 0

    @patch("layerlens.cli.commands.trace.get_client")
    def test_trace_delete_confirms(self, mock_get_client, runner):
        """trace delete prompts for confirmation."""
        client = Mock()
        mock_get_client.return_value = client

        result = runner.invoke(
            cli, ["trace", "delete", "trace-123"], input="y\n", env={"LAYERLENS_STRATIX_API_KEY": "test"}
        )

        client.traces.delete.assert_called_once()

    @patch("layerlens.cli.commands.trace.get_client")
    def test_trace_delete_skip_confirm(self, mock_get_client, runner):
        """trace delete --yes skips confirmation."""
        client = Mock()
        client.traces.delete.return_value = True
        mock_get_client.return_value = client

        result = runner.invoke(
            cli, ["trace", "delete", "trace-123", "--yes"], env={"LAYERLENS_STRATIX_API_KEY": "test"}
        )

        assert result.exit_code == 0
        client.traces.delete.assert_called_once()


class TestJudgeCommands:
    """Test judge CLI commands."""

    @pytest.fixture
    def runner(self):
        return _make_runner()

    @patch("layerlens.cli.commands.judge.get_client")
    def test_judge_list(self, mock_get_client, runner):
        """judge list displays judges."""
        judge = Mock()
        judge.model_dump.return_value = {
            "id": "j-1",
            "name": "Quality",
            "version": 1,
            "run_count": 5,
            "created_at": "2026-01-01T00:00:00Z",
        }
        client = Mock()
        resp = Mock()
        resp.judges = [judge]
        resp.count = 1
        resp.total_count = 1
        client.judges.get_many.return_value = resp
        mock_get_client.return_value = client

        result = runner.invoke(cli, ["judge", "list"], env={"LAYERLENS_STRATIX_API_KEY": "test"})

        assert result.exit_code == 0
        assert "Quality" in result.output

    @patch("layerlens.cli.commands.judge.get_client")
    def test_judge_create(self, mock_get_client, runner):
        """judge create creates and displays a judge."""
        judge = Mock()
        judge.id = "j-new"
        judge.model_dump.return_value = {"id": "j-new", "name": "Test"}
        client = Mock()
        client.judges.create.return_value = judge
        mock_get_client.return_value = client

        result = runner.invoke(
            cli,
            ["judge", "create", "--name", "Test", "--goal", "Evaluate accuracy and completeness"],
            env={"LAYERLENS_STRATIX_API_KEY": "test"},
        )

        assert result.exit_code == 0
        assert "j-new" in result.output

    @patch("layerlens.cli.commands.judge.get_client")
    def test_judge_test(self, mock_get_client, runner):
        """judge test creates a trace evaluation."""
        te = Mock()
        te.id = "te-1"
        te.model_dump.return_value = {"id": "te-1", "trace_id": "t-1", "judge_id": "j-1", "status": "pending"}
        client = Mock()
        client.trace_evaluations.create.return_value = te
        mock_get_client.return_value = client

        result = runner.invoke(
            cli,
            ["judge", "test", "--judge-id", "j-1", "--trace-id", "t-1"],
            env={"LAYERLENS_STRATIX_API_KEY": "test"},
        )

        assert result.exit_code == 0
        assert "te-1" in result.output


class TestEvaluateCommands:
    """Test evaluate CLI commands."""

    @pytest.fixture
    def runner(self):
        return _make_runner()

    @patch("layerlens.cli.commands.evaluate.get_client")
    def test_evaluate_list(self, mock_get_client, runner):
        """evaluate list displays evaluations."""
        ev = Mock()
        ev.model_dump.return_value = {
            "id": "ev-1",
            "status": "success",
            "model_name": "GPT-4",
            "benchmark_name": "MATH",
            "accuracy": 0.95,
            "submitted_at": 1700000000,
        }
        client = Mock()
        resp = Mock()
        resp.evaluations = [ev]
        resp.pagination = Mock(page=1, total_pages=1, total_count=1)
        client.evaluations.get_many.return_value = resp
        mock_get_client.return_value = client

        result = runner.invoke(cli, ["evaluate", "list"], env={"LAYERLENS_STRATIX_API_KEY": "test"})

        assert result.exit_code == 0
        assert "GPT-4" in result.output


class TestScorerCommands:
    """Test scorer CLI commands."""

    @pytest.fixture
    def runner(self):
        return _make_runner()

    @patch("layerlens.cli.commands.scorer.get_client")
    def test_scorer_list(self, mock_get_client, runner):
        """scorer list displays scorers."""
        scorer = Mock()
        scorer.model_dump.return_value = {
            "id": "s-1",
            "name": "Quality",
            "model_name": "GPT-4",
            "model_company": "OpenAI",
            "created_at": "2026-01-01",
        }
        client = Mock()
        resp = Mock()
        resp.scorers = [scorer]
        resp.count = 1
        resp.total_count = 1
        client.scorers.get_many.return_value = resp
        mock_get_client.return_value = client

        result = runner.invoke(cli, ["scorer", "list"], env={"LAYERLENS_STRATIX_API_KEY": "test"})

        assert result.exit_code == 0
        assert "Quality" in result.output

    @patch("layerlens.cli.commands.scorer.get_client")
    def test_scorer_create_dry_run(self, mock_get_client, runner):
        """scorer create --dry-run previews without executing."""
        result = runner.invoke(
            cli,
            [
                "scorer",
                "create",
                "--name",
                "Test",
                "--description",
                "A test scorer for quality",
                "--model-id",
                "m-1",
                "--prompt",
                "Rate quality",
                "--dry-run",
            ],
            env={"LAYERLENS_STRATIX_API_KEY": "test"},
        )

        assert result.exit_code == 0
        assert "[dry-run]" in result.output
        mock_get_client.assert_not_called()

    @patch("layerlens.cli.commands.scorer.get_client")
    def test_scorer_delete_yes(self, mock_get_client, runner):
        """scorer delete --yes skips confirmation."""
        client = Mock()
        client.scorers.delete.return_value = True
        mock_get_client.return_value = client

        result = runner.invoke(cli, ["scorer", "delete", "s-1", "--yes"], env={"LAYERLENS_STRATIX_API_KEY": "test"})

        assert result.exit_code == 0
        client.scorers.delete.assert_called_once_with("s-1")


class TestSpaceCommands:
    """Test space CLI commands."""

    @pytest.fixture
    def runner(self):
        return _make_runner()

    @patch("layerlens.cli.commands.space.get_client")
    def test_space_create_dry_run(self, mock_get_client, runner):
        """space create --dry-run previews without executing."""
        result = runner.invoke(
            cli,
            ["space", "create", "--name", "Test Space", "--dry-run"],
            env={"LAYERLENS_STRATIX_API_KEY": "test"},
        )

        assert result.exit_code == 0
        assert "[dry-run]" in result.output
        mock_get_client.assert_not_called()


class TestBulkCommands:
    """Test bulk CLI commands."""

    @pytest.fixture
    def runner(self):
        return _make_runner()

    @patch("layerlens.cli.commands.bulk.get_client")
    def test_bulk_eval_file_dry_run(self, _mock_get_client, runner):
        """bulk eval --file --dry-run previews jobs."""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"model": "gpt-4", "benchmark": "mmlu"}\n')
            f.write('{"model": "claude", "benchmark": "mmlu"}\n')
            jobs_path = f.name

        try:
            result = runner.invoke(
                cli,
                ["bulk", "eval", "--file", jobs_path, "--dry-run"],
                env={"LAYERLENS_STRATIX_API_KEY": "test"},
            )

            assert result.exit_code == 0, f"stdout={result.output!r} stderr={result.stderr!r}"
            assert "[dry-run]" in result.output
            assert "2 evaluation(s)" in result.output
        finally:
            os.unlink(jobs_path)

    def test_bulk_eval_no_args(self, runner):
        """bulk eval with no arguments shows error."""
        result = runner.invoke(cli, ["bulk", "eval"], env={"LAYERLENS_STRATIX_API_KEY": "test"})

        assert result.exit_code != 0

    @patch("layerlens.cli.commands.bulk.get_client")
    def test_bulk_eval_judge_traces_dry_run(self, _mock_get_client, runner, tmp_path):
        """bulk eval --judge-id --traces --dry-run previews trace evaluations."""
        traces_file = tmp_path / "traces.txt"
        traces_file.write_text("t-1\nt-2\nt-3\n")

        result = runner.invoke(
            cli,
            ["bulk", "eval", "--judge-id", "j-1", "--traces", str(traces_file), "--dry-run"],
            env={"LAYERLENS_STRATIX_API_KEY": "test"},
        )

        assert result.exit_code == 0
        assert "3 trace evaluation(s)" in result.output


class TestCiCommands:
    """Test ci CLI commands."""

    @pytest.fixture
    def runner(self):
        return _make_runner()

    def test_ci_report_dry_run(self, runner):
        """ci report --dry-run previews."""
        result = runner.invoke(cli, ["ci", "report", "--dry-run"], env={"LAYERLENS_STRATIX_API_KEY": "test"})

        assert result.exit_code == 0
        assert "[dry-run]" in result.output

    @patch("layerlens.cli.commands.ci.get_client")
    def test_ci_report_markdown(self, mock_get_client, runner):
        """ci report generates markdown."""
        ev = Mock()
        ev.id = "ev-1"
        ev.status = "success"
        ev.model_name = "GPT-4"
        ev.benchmark_name = "MATH"
        ev.accuracy = 0.95
        ev.model_dump.return_value = {"id": "ev-1", "status": "success"}

        client = Mock()
        resp = Mock()
        resp.evaluations = [ev]
        client.evaluations.get_many.return_value = resp
        mock_get_client.return_value = client

        result = runner.invoke(cli, ["ci", "report"], env={"LAYERLENS_STRATIX_API_KEY": "test"})

        assert result.exit_code == 0
        assert "# Stratix Evaluation Report" in result.output
        assert "GPT-4" in result.output

    @patch("layerlens.cli.commands.ci.get_client")
    def test_ci_report_to_file(self, mock_get_client, runner, tmp_path):
        """ci report --output writes to file."""
        ev = Mock()
        ev.id = "ev-1"
        ev.status = "success"
        ev.model_name = "GPT-4"
        ev.benchmark_name = "MATH"
        ev.accuracy = 0.95
        ev.model_dump.return_value = {"id": "ev-1"}

        client = Mock()
        resp = Mock()
        resp.evaluations = [ev]
        client.evaluations.get_many.return_value = resp
        mock_get_client.return_value = client

        out_file = tmp_path / "report.md"
        result = runner.invoke(cli, ["ci", "report", "-o", str(out_file)], env={"LAYERLENS_STRATIX_API_KEY": "test"})

        assert result.exit_code == 0
        assert out_file.exists()
        content = out_file.read_text()
        assert "Stratix Evaluation Report" in content


class TestGlobalOptions:
    """Test global CLI options."""

    @pytest.fixture
    def runner(self):
        return _make_runner()

    def test_version(self, runner):
        """--version prints version."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "layerlens" in result.output

    def test_help(self, runner):
        """--help shows all command groups."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        for cmd in ["trace", "judge", "evaluate", "integration", "scorer", "space", "bulk", "ci"]:
            assert cmd in result.output

    @patch("layerlens.cli.commands.trace.get_client")
    def test_json_format(self, mock_get_client, runner):
        """--format json outputs JSON."""
        trace = Mock()
        trace.model_dump.return_value = {"id": "t-1", "filename": "test.json"}
        client = Mock()
        client.traces.get.return_value = trace
        mock_get_client.return_value = client

        result = runner.invoke(
            cli, ["--format", "json", "trace", "get", "t-1"], env={"LAYERLENS_STRATIX_API_KEY": "test"}
        )

        assert result.exit_code == 0
        import json

        parsed = json.loads(result.output)
        assert parsed["id"] == "t-1"

    def test_quiet_flag(self, runner):
        """--quiet suppresses banner."""
        result = runner.invoke(cli, ["-q", "--help"])
        assert result.exit_code == 0
        # Banner goes to stderr; with -q it should be empty
        assert "STRATIX" not in result.stderr
