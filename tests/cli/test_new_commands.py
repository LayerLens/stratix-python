"""CLI tests for replay / synthetic / evaluations subcommands."""

from __future__ import annotations

import sys
import json

import pytest
from click.testing import CliRunner

from layerlens.cli._app import cli
from layerlens.evaluation_runs.models import (
    RunAggregate,
    EvaluationRun,
    EvaluationRunStatus,
)


@pytest.fixture
def runner():
    # `mix_stderr` is incompatible across click versions in this repo's baseline;
    # use the default runner which still separates streams via --catch.
    return CliRunner()


# ---------------------------------------------------------------------------
# synthetic
# ---------------------------------------------------------------------------


class TestSyntheticCommands:
    def test_templates_lists_known_ids(self, runner):
        result = runner.invoke(cli, ["--quiet", "synthetic", "templates"])
        assert result.exit_code == 0
        assert "llm.chat.basic" in result.output
        assert "rag.retrieval" in result.output

    def test_generate_to_stdout(self, runner):
        result = runner.invoke(
            cli,
            ["--quiet", "synthetic", "generate", "--template", "llm.chat.basic", "--count", "2"],
        )
        assert result.exit_code == 0
        lines = [line for line in result.output.splitlines() if line.startswith("{")]
        assert len(lines) == 2
        parsed = json.loads(lines[0])
        assert parsed["data"]["synthetic"] is True

    def test_generate_to_file(self, runner, tmp_path):
        out = tmp_path / "traces.jsonl"
        result = runner.invoke(
            cli,
            [
                "--quiet",
                "synthetic",
                "generate",
                "--template",
                "rag.retrieval",
                "--count",
                "3",
                "--out",
                str(out),
            ],
        )
        assert result.exit_code == 0
        lines = out.read_text().strip().splitlines()
        assert len(lines) == 3

    def test_generate_unknown_template_exits_nonzero(self, runner):
        result = runner.invoke(
            cli,
            [
                "--quiet",
                "synthetic",
                "generate",
                "--template",
                "does.not.exist",
                "--count",
                "1",
            ],
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------


class TestReplayCommands:
    def test_run_fallback_prints_json(self, runner):
        result = runner.invoke(cli, ["--quiet", "replay", "run", "--trace-id", "t1"])
        assert result.exit_code == 0
        payload = json.loads(
            result.output.split("\n{", 1)[-1] if not result.output.lstrip().startswith("{") else result.output
        )
        assert payload["original_trace_id"] == "t1"
        assert payload["status"] == "completed"

    def test_run_propagates_model_override_into_metadata(self, runner):
        result = runner.invoke(
            cli,
            [
                "--quiet",
                "replay",
                "run",
                "--trace-id",
                "t1",
                "--model-override",
                "gpt-4o-mini",
            ],
        )
        assert result.exit_code == 0
        payload = json.loads(_last_json_blob(result.output))
        assert payload["metadata"]["replay_type"] == "model_swap"
        assert payload["metadata"]["overrides"]["model"] == "gpt-4o-mini"

    def test_bad_replay_fn_spec_errors(self, runner):
        result = runner.invoke(
            cli,
            [
                "--quiet",
                "replay",
                "run",
                "--trace-id",
                "t1",
                "--replay-fn",
                "no_colon",
            ],
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# evaluations
# ---------------------------------------------------------------------------


_TARGET_MODULE = "layerlens_test_target_module"


def _register_test_target():
    """Register an in-memory module so --target can resolve to a real callable."""
    import types

    module = types.ModuleType(_TARGET_MODULE)

    def identity(x):
        return x

    def scorer(actual, expected, _meta):
        return 1.0 if actual == expected else 0.0

    module.identity = identity
    module.scorer = scorer
    sys.modules[_TARGET_MODULE] = module


class TestEvaluationsCommands:
    def setup_method(self):
        _register_test_target()

    def test_run_requires_dataset_file(self, runner):
        result = runner.invoke(
            cli,
            [
                "--quiet",
                "evaluations",
                "run",
                "--dataset-id",
                "d1",
                "--target",
                f"{_TARGET_MODULE}:identity",
            ],
        )
        assert result.exit_code != 0
        assert "dataset-file" in result.output

    def test_run_reads_dataset_file_and_emits_run(self, runner, tmp_path):
        ds_path = tmp_path / "ds.json"
        ds_path.write_text(
            json.dumps(
                [
                    {"id": "a", "input": 1, "expected_output": 1},
                    {"id": "b", "input": 2, "expected_output": 3},  # will fail
                ]
            )
        )
        result = runner.invoke(
            cli,
            [
                "--quiet",
                "evaluations",
                "run",
                "--dataset-id",
                "local",
                "--dataset-file",
                str(ds_path),
                "--target",
                f"{_TARGET_MODULE}:identity",
                "--scorer",
                f"exact={_TARGET_MODULE}:scorer",
            ],
        )
        assert result.exit_code == 0
        payload = json.loads(_last_json_blob(result.output))
        assert payload["status"] == "completed"
        assert 0.4 < payload["aggregate"]["pass_rate"] < 0.6  # 1 of 2 items pass

    def test_compare_exits_nonzero_on_regression(self, runner, tmp_path):
        base = _run_with(pass_rate=1.0, mean=1.0, items=[("a", True)])
        cand = _run_with(pass_rate=0.0, mean=0.0, items=[("a", False)])
        base_path = tmp_path / "base.json"
        cand_path = tmp_path / "cand.json"
        base_path.write_text(base.model_dump_json())
        cand_path.write_text(cand.model_dump_json())
        result = runner.invoke(
            cli,
            ["--quiet", "evaluations", "compare", str(base_path), str(cand_path)],
        )
        assert result.exit_code == 1
        payload = json.loads(_last_json_blob(result.output))
        assert payload["is_regression"] is True

    def test_compare_exits_zero_when_stable(self, runner, tmp_path):
        base = _run_with(pass_rate=1.0, mean=1.0, items=[("a", True)])
        cand = _run_with(pass_rate=1.0, mean=1.0, items=[("a", True)])
        base_path = tmp_path / "base.json"
        cand_path = tmp_path / "cand.json"
        base_path.write_text(base.model_dump_json())
        cand_path.write_text(cand.model_dump_json())
        result = runner.invoke(
            cli,
            ["--quiet", "evaluations", "compare", str(base_path), str(cand_path)],
        )
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _last_json_blob(output: str) -> str:
    """Return the last top-level JSON object in the CLI output."""
    stripped = output.strip()
    # Output may include extraneous lines (banner disabled via --quiet, but stderr may
    # still emit messages). Find the outermost JSON object.
    for idx, ch in enumerate(stripped):
        if ch == "{":
            return stripped[idx:]
    raise AssertionError(f"no JSON object found in output: {output!r}")


def _run_with(*, pass_rate: float, mean: float, items):
    from layerlens.evaluation_runs.models import EvaluationRunItem

    return EvaluationRun(
        id="run-" + str(int(pass_rate * 100)),
        dataset_id="d",
        dataset_version=1,
        status=EvaluationRunStatus.COMPLETED,
        items=[EvaluationRunItem(item_id=i, passed=p) for i, p in items],
        aggregate=RunAggregate(mean_scores={"exact": mean}, pass_rate=pass_rate, item_count=len(items)),
    )
