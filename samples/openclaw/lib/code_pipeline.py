"""
Code Pipeline -- Multi-Stage Code Generation with Quality Gate
================================================================

Orchestrates a four-stage pipeline for AI-assisted code generation:
Coder -> Reviewer -> Tester -> Judge.  If the Judge returns FAIL, the
pipeline loops back to the Coder with feedback, up to max retries.
"""

from __future__ import annotations

import time
import uuid
import hashlib
import logging
from typing import Any

from ..judges.code_quality import CodeQualityJudge

logger = logging.getLogger(__name__)


class StageResult:
    """Captures the output of a single pipeline stage."""

    def __init__(
        self,
        stage: str,
        output: str,
        metadata: dict[str, Any] | None = None,
        duration_ms: int = 0,
    ) -> None:
        self.stage = stage
        self.output = output
        self.metadata = metadata or {}
        self.duration_ms = duration_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "output": self.output[:500],
            "metadata": self.metadata,
            "duration_ms": self.duration_ms,
        }


class PipelineIteration:
    """Records all stage results for one iteration of the pipeline."""

    def __init__(self, iteration: int) -> None:
        self.iteration = iteration
        self.stages: list[StageResult] = []
        self.verdict: str = ""
        self.aggregate_score: float = 0.0

    def add_stage(self, result: StageResult) -> None:
        self.stages.append(result)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "verdict": self.verdict,
            "aggregate_score": self.aggregate_score,
            "stages": [s.to_dict() for s in self.stages],
        }


def _simulate_coder(task: str, iteration: int, feedback: str = "") -> StageResult:
    start = time.monotonic()
    digest = hashlib.sha256(f"{task}:{iteration}".encode()).hexdigest()[:12]
    func_name = "process_" + digest[:8]
    lines = [f"def {func_name}(data):", f'    """Generated for: {task[:60]}"""']
    if iteration >= 2:
        lines.extend(
            [
                "    # Input validation added based on reviewer feedback",
                "    if not data:",
                '        raise ValueError("Input data cannot be empty")',
            ]
        )
    if iteration >= 3:
        lines.extend(["    # Security check added based on judge feedback", "    data = _sanitize(data)"])
    lines.extend(["    result = _transform(data)", "    return result", ""])
    if iteration >= 2:
        lines.extend(
            [
                f"def test_{func_name}():",
                f'    """Tests for {func_name}."""',
                f'    assert {func_name}({{"key": "value"}}) is not None',
                f'    assert {func_name}({{"items": [1,2,3]}}) is not None',
            ]
        )
    if iteration >= 3:
        lines.extend(
            [
                f"    # Edge case tests",
                f"    import pytest",
                f"    with pytest.raises(ValueError):",
                f"        {func_name}(None)",
                f"    with pytest.raises(ValueError):",
                f"        {func_name}({{}})",
            ]
        )
    code = "\n".join(lines)
    duration = int((time.monotonic() - start) * 1000) + 150 * iteration
    return StageResult(
        stage="coder",
        output=code,
        metadata={
            "function_name": func_name,
            "line_count": len(lines),
            "iteration": iteration,
            "incorporated_feedback": bool(feedback),
        },
        duration_ms=duration,
    )


def _simulate_reviewer(code: str, task: str, iteration: int) -> StageResult:
    start = time.monotonic()
    if iteration == 1:
        comments = [
            "Missing input validation.",
            "No error handling for edge cases.",
            "Consider adding type hints.",
            "Security: sanitize external inputs.",
        ]
    elif iteration == 2:
        comments = [
            "Input validation looks good.",
            "Consider more comprehensive test coverage.",
            "Minor: could extract helper functions.",
        ]
    else:
        comments = ["Code looks solid.", "Minor: consider adding a docstring to the test function."]
    review_text = "REVIEW COMMENTS:\n" + "\n".join(f"  - {c}" for c in comments)
    duration = int((time.monotonic() - start) * 1000) + 80
    return StageResult(
        stage="reviewer",
        output=review_text,
        metadata={"comment_count": len(comments), "severity": "major" if iteration == 1 else "minor"},
        duration_ms=duration,
    )


def _simulate_tester(code: str, task: str, iteration: int) -> StageResult:
    start = time.monotonic()
    total_tests = 5 + iteration * 2
    if iteration == 1:
        passed = int(total_tests * 0.5)
    elif iteration == 2:
        passed = int(total_tests * 0.75)
    else:
        passed = int(total_tests * 0.95)
    failed = total_tests - passed
    test_output = (
        f"Test Results (iteration {iteration}):\n  Total:  {total_tests}\n"
        f"  Passed: {passed}\n  Failed: {failed}\n"
        f"  Pass Rate: {passed / total_tests * 100:.0f}%"
    )
    if failed > 0:
        test_output += "\n\n  Failed tests:"
        for i in range(min(failed, 3)):
            test_output += f"\n    - test_edge_case_{i + 1}: AssertionError"
    duration = int((time.monotonic() - start) * 1000) + 200
    return StageResult(
        stage="tester",
        output=test_output,
        metadata={
            "total_tests": total_tests,
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / total_tests, 2),
        },
        duration_ms=duration,
    )


class CodePipeline:
    """Multi-stage code generation pipeline with quality gate and retry loop."""

    def __init__(self, judge: CodeQualityJudge | None = None, max_iterations: int = 3) -> None:
        self.judge = judge or CodeQualityJudge()
        self.max_iterations = max_iterations
        self._iterations: list[PipelineIteration] = []

    def execute(self, task: str) -> dict[str, Any]:
        self._iterations = []
        feedback = ""
        final_verdict = "FAIL"
        final_score = 0.0
        final_result: dict[str, Any] = {}

        for iteration in range(1, self.max_iterations + 1):
            logger.info("Pipeline iteration %d/%d", iteration, self.max_iterations)
            iter_record = PipelineIteration(iteration)

            coder_result = _simulate_coder(task, iteration, feedback)
            iter_record.add_stage(coder_result)

            reviewer_result = _simulate_reviewer(coder_result.output, task, iteration)
            iter_record.add_stage(reviewer_result)

            tester_result = _simulate_tester(coder_result.output, task, iteration)
            iter_record.add_stage(tester_result)

            trace_id = str(uuid.uuid4())
            judge_result = self.judge.evaluate(
                trace_id=trace_id,
                output=coder_result.output,
                context={
                    "task": task,
                    "iteration": iteration,
                    "review_comments": reviewer_result.output,
                    "test_results": tester_result.output,
                },
            )

            judge_stage = StageResult(
                stage="judge",
                output=judge_result["rationale"],
                metadata={
                    "scores": judge_result["scores"],
                    "aggregate_score": judge_result["aggregate_score"],
                    "verdict": judge_result["verdict"],
                    "suggestions": judge_result["suggestions"],
                    "trace_id": trace_id,
                },
            )
            iter_record.add_stage(judge_stage)
            iter_record.verdict = judge_result["verdict"]
            iter_record.aggregate_score = judge_result["aggregate_score"]
            self._iterations.append(iter_record)
            final_verdict = judge_result["verdict"]
            final_score = judge_result["aggregate_score"]
            final_result = judge_result

            if judge_result["verdict"] == "PASS":
                logger.info("Pipeline PASSED on iteration %d", iteration)
                break

            suggestions = judge_result.get("suggestions", [])
            feedback = (
                f"Previous iteration scored {judge_result['aggregate_score']:.1f}. "
                f"Suggestions: {'; '.join(suggestions)}"
            )

        return {
            "task": task,
            "iterations": [it.to_dict() for it in self._iterations],
            "final_verdict": final_verdict,
            "final_score": final_score,
            "total_iterations": len(self._iterations),
            "gate_threshold": self.judge.gate_threshold,
            "passed": final_verdict == "PASS",
            "final_evaluation": final_result,
        }

    @property
    def iterations(self) -> list[PipelineIteration]:
        return list(self._iterations)
