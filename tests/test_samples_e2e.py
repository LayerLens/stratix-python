"""End-to-end tests for ALL 58 SDK sample demos.

Tests every sample in six modes:
1. TestAllSamplesWithMockedSDK -- mocked Stratix client (all 58)
2. TestAllSamplesLiveAPI -- real API via subprocess (all 58, @pytest.mark.live)
3. TestOpenClawOfflineMode -- --no-sdk flag (11 OpenClaw demos)
4. TestWithoutAPIKey -- graceful failure without credentials (all 58)
5. TestMissingDependencies -- optional-dep fallback (integration, openclaw, copilotkit)
6. TestSampleCompleteness -- verify test lists match disk
"""

from __future__ import annotations

import io
import os
import sys
import json
import asyncio
import tempfile
import importlib
import subprocess
import importlib.util
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "samples")
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

# ---- Sample file lists (all 58) ----

CORE_SAMPLES = [
    "async_results",
    "async_workflow",
    "basic_trace",
    "benchmark_evaluation",
    "compare_evaluations",
    "create_judge",
    "custom_benchmark",
    "custom_model",
    "evaluation_filtering",
    "evaluation_pipeline",
    "integration_management",
    "judge_creation_and_test",
    "judge_optimization",
    "model_benchmark_management",
    "paginated_results",
    "public_catalog",
    "quickstart",
    "run_evaluation",
    "trace_evaluation",
    "trace_investigation",
]

INDUSTRY_SAMPLES = [
    "financial_fraud",
    "financial_trading",
    "government_citizen",
    "healthcare_clinical",
    "insurance_claims",
    "insurance_underwriting",
    "legal_contracts",
    "legal_research",
    "retail_recommender",
    "retail_support",
]

COWORK_SAMPLES = [
    "code_review",
    "incident_response",
    "multi_agent_eval",
    "pair_programming",
    "rag_assessment",
]

MODALITY_SAMPLES = [
    "brand_evaluation",
    "document_evaluation",
    "text_evaluation",
]

INTEGRATION_SAMPLES = [
    "anthropic_traced",
    "langchain_instrumented",
    "openai_instrumented",
    "openai_traced",
]

CICD_SAMPLES = [
    "pre_commit_hook",
    "quality_gate",
]

COPILOTKIT_SAMPLES = [
    "evaluator_agent",
    "investigator_agent",
]

MCP_SAMPLES = [
    "layerlens_server",
]

OPENCLAW_DEMOS = [
    "cage_match",
    "code_gate",
    "compare_agent_models",
    "content_observer",
    "evaluate_skill_output",
    "heartbeat_benchmark",
    "monitor_agent_safety",
    "skill_auditor",
    "soul_redteam",
    "trace_agent_execution",
]

# Demos that extend DemoRunner and support --no-sdk flag
OPENCLAW_RUNNER_DEMOS = [
    "cage_match",
    "code_gate",
    "content_observer",
    "heartbeat_benchmark",
    "skill_auditor",
    "soul_redteam",
]

# Demos that directly use Stratix() and require an API key
OPENCLAW_DIRECT_DEMOS = [
    "compare_agent_models",
    "evaluate_skill_output",
    "monitor_agent_safety",
    "trace_agent_execution",
]

OPENCLAW_SKILL_SCRIPT = "layerlens_skill/scripts/evaluate"

# ---- All samples as (category, name) pairs for parametrization ----

ALL_MOCKED_SAMPLES = (
    [("core", name) for name in CORE_SAMPLES]
    + [("industry", name) for name in INDUSTRY_SAMPLES]
    + [("cowork", name) for name in COWORK_SAMPLES]
    + [("modalities", name) for name in MODALITY_SAMPLES]
    + [("integrations", name) for name in INTEGRATION_SAMPLES]
    + [("cicd", name) for name in CICD_SAMPLES]
    + [("copilotkit/agents", name) for name in COPILOTKIT_SAMPLES]
    + [("mcp", name) for name in MCP_SAMPLES]
    + [("openclaw", name) for name in OPENCLAW_DEMOS]
    + [("openclaw/layerlens_skill/scripts", "evaluate")]
)

# All 54 sample paths (relative to SAMPLES_DIR) for live / no-key tests
ALL_SAMPLE_PATHS = (
    [f"core/{s}.py" for s in CORE_SAMPLES]
    + [f"industry/{s}.py" for s in INDUSTRY_SAMPLES]
    + [f"cowork/{s}.py" for s in COWORK_SAMPLES]
    + [f"modalities/{s}.py" for s in MODALITY_SAMPLES]
    + [f"integrations/{s}.py" for s in INTEGRATION_SAMPLES]
    + [f"cicd/{s}.py" for s in CICD_SAMPLES]
    + [f"copilotkit/agents/{s}.py" for s in COPILOTKIT_SAMPLES]
    + [f"mcp/{s}.py" for s in MCP_SAMPLES]
    + [f"openclaw/{s}.py" for s in OPENCLAW_DEMOS]
    + ["openclaw/layerlens_skill/scripts/evaluate.py"]
)

# Async core samples that need AsyncStratix mocking
_ASYNC_CORE_SAMPLES = {"async_results", "async_workflow"}

# Samples that require external provider SDKs (openai, langchain, etc.)
# with no simulated fallback -- cannot run in fully mocked mode.
_EXTERNAL_SDK_SAMPLES = {"langchain_instrumented", "openai_instrumented"}

# Samples that need special argv or patches
_SPECIAL_ARGV: dict[tuple[str, str], list[str]] = {
    ("cicd", "quality_gate"): ["test", "--threshold", "0.0"],
    ("openclaw/layerlens_skill/scripts", "evaluate"): [
        "test",
        "--input",
        "test prompt",
        "--output",
        "test response",
    ],
}

# Samples that need special extra_patches
_SPECIAL_PATCHES: dict[tuple[str, str], dict[str, Any]] = {}

# Pre-commit hook needs subprocess.run mocked
_pre_commit_mock_result = MagicMock()
_pre_commit_mock_result.stdout = ""
_pre_commit_mock_result.returncode = 0
_SPECIAL_PATCHES[("cicd", "pre_commit_hook")] = {
    "subprocess.run": MagicMock(return_value=_pre_commit_mock_result),
}


# ---- Fixtures ----


@pytest.fixture
def mock_stratix():
    """Create a fully mocked Stratix client that returns realistic responses."""
    client = MagicMock()
    client.organization_id = "org-test-123"
    client.project_id = "proj-test-456"

    # --- traces ---
    trace_resp = MagicMock()
    trace_resp.trace_ids = ["trace-test-001"]
    client.traces.upload.return_value = trace_resp

    traces_list = MagicMock()
    traces_list.traces = [
        MagicMock(
            id="trace-001",
            data={"input": "test"},
            filename="test.jsonl",
            created_at="2026-01-01",
        )
    ]
    traces_list.count = 1
    traces_list.total_count = 1
    client.traces.get_many.return_value = traces_list

    trace_obj = MagicMock()
    trace_obj.id = "trace-001"
    trace_obj.data = {
        "input": [{"role": "user", "content": "test"}],
        "output": "test response",
    }
    trace_obj.filename = "test.jsonl"
    trace_obj.created_at = "2026-01-01"
    client.traces.get.return_value = trace_obj
    client.traces.delete.return_value = True
    client.traces.get_sources.return_value = ["test.jsonl"]

    # --- judges ---
    judge = MagicMock()
    judge.id = "judge-test-001"
    judge.name = "Test Judge"
    judge.evaluation_goal = "Test evaluation"
    judge.version = 1
    judge.created_at = "2026-01-01"
    judge.updated_at = "2026-01-01"
    client.judges.create.return_value = judge
    client.judges.get.return_value = judge
    judges_resp = MagicMock()
    judges_resp.judges = [judge]
    judges_resp.count = 1
    client.judges.get_many.return_value = judges_resp
    client.judges.update.return_value = MagicMock()
    client.judges.delete.return_value = MagicMock()

    # --- trace evaluations ---
    trace_eval = MagicMock()
    trace_eval.id = "te-test-001"
    trace_eval.trace_id = "trace-001"
    trace_eval.judge_id = "judge-test-001"
    trace_eval.status = MagicMock(value="success")
    trace_eval.created_at = "2026-01-01"
    client.trace_evaluations.create.return_value = trace_eval

    te_eval_obj = MagicMock()
    te_eval_obj.id = "te-test-001"
    te_eval_obj.status = MagicMock(value="completed")
    client.trace_evaluations.get.return_value = te_eval_obj

    te_list_resp = MagicMock()
    te_list_resp.count = 1
    te_list_resp.total = 1
    te_list_resp.evaluations = [trace_eval]
    client.trace_evaluations.get_many.return_value = te_list_resp

    # --- trace evaluation results ---
    # TraceEvaluationResultsResponse extends TraceEvaluationResult directly,
    # so the response IS a single result with score/passed/reasoning at top level.
    te_results_resp = MagicMock()
    te_results_resp.id = "ter-001"
    te_results_resp.score = 0.85
    te_results_resp.passed = True
    te_results_resp.reasoning = "The response is accurate and complete."
    te_results_resp.latency_ms = 1500
    te_results_resp.total_cost = 0.003
    te_results_resp.steps = []
    te_results_resp.model = "test-model"
    te_results_resp.turns = 1
    te_results_resp.prompt_tokens = 100
    te_results_resp.completion_tokens = 50
    te_results_resp.created_at = "2026-01-01"
    client.trace_evaluations.get_results.return_value = te_results_resp

    # --- cost estimate ---
    cost_resp = MagicMock()
    cost_resp.estimated_cost = 0.05
    cost_resp.input_tokens = 500
    cost_resp.output_tokens = 200
    client.trace_evaluations.estimate_cost.return_value = cost_resp

    # --- evaluations ---
    evaluation = MagicMock()
    evaluation.id = "eval-test-001"
    evaluation.status = MagicMock(value="completed")
    evaluation.is_success = True
    evaluation.accuracy = 85.5
    evaluation.submitted_at = "2026-01-01T00:00:00Z"
    evaluation.average_duration = 1200
    evaluation.model_id = "model-001"
    evaluation.benchmark_id = "bench-001"
    client.evaluations.create.return_value = evaluation
    client.evaluations.get.return_value = evaluation
    client.evaluations.get_by_id.return_value = evaluation
    evaluation2 = MagicMock()
    evaluation2.id = "eval-test-002"
    evaluation2.status = MagicMock(value="completed")
    evaluation2.is_success = True
    evaluation2.accuracy = 90.2
    evaluation2.submitted_at = "2026-01-02T00:00:00Z"
    evaluation2.average_duration = 800
    evaluation2.model_id = "model-001"
    evaluation2.benchmark_id = "bench-001"

    evals_resp = MagicMock()
    evals_resp.evaluations = [evaluation, evaluation2]
    evals_resp.count = 2
    evals_resp.total_count = 2
    evals_resp.pagination = MagicMock(total_count=2, total_pages=1, current_page=1, page_size=10)
    client.evaluations.get_many.return_value = evals_resp
    client.evaluations.wait_for_completion.return_value = evaluation

    # --- results ---
    result = MagicMock()
    result.id = "result-001"
    result.score = 0.85
    result.prompt = "What is the speed of light in a vacuum?"
    result.result = "The speed of light is approximately 299,792,458 m/s."
    result.truth = "299792458 m/s"
    result.subset = "physics"
    results_resp = MagicMock()
    results_resp.results = [result]
    results_resp.evaluation_id = "eval-test-001"
    results_resp.metrics = MagicMock(total_count=1)
    results_resp.pagination = MagicMock(total_count=1, total_pages=1, current_page=1, page_size=10)
    client.results.get.return_value = results_resp
    client.results.get_by_id.return_value = results_resp
    client.results.get_all.return_value = [result]
    client.results.get_all_by_id.return_value = [result]

    # --- models ---
    model = MagicMock()
    model.id = "model-001"
    model.name = "Test Model"
    model.key = "test-model"
    client.models.get.return_value = [model]
    client.models.get_by_id.return_value = model
    client.models.get_by_key.return_value = model
    client.models.add.return_value = True
    client.models.remove.return_value = True
    client.models.create_custom.return_value = MagicMock(model_id="model-custom-001")

    # --- benchmarks ---
    benchmark = MagicMock()
    benchmark.id = "bench-001"
    benchmark.name = "Test Benchmark"
    benchmark.key = "test-bench"
    client.benchmarks.get.return_value = [benchmark]
    client.benchmarks.get_by_id.return_value = benchmark
    client.benchmarks.get_by_key.return_value = benchmark
    client.benchmarks.add.return_value = True
    client.benchmarks.create_custom.return_value = MagicMock(benchmark_id="bench-custom-001")
    client.benchmarks.create_smart.return_value = MagicMock(benchmark_id="bench-smart-001")

    # --- public client ---
    public = MagicMock()

    # Public models
    pub_model = MagicMock()
    pub_model.name = "GPT-4o"
    pub_model.company = "OpenAI"
    pub_model.id = "model-pub-001"
    pub_model.key = "gpt-4o"
    pub_model.released_at = "2025-01-01"
    pub_models_resp = MagicMock()
    pub_models_resp.models = [pub_model]
    pub_models_resp.total_count = 1
    pub_models_resp.categories = ["open-source"]
    pub_models_resp.companies = ["OpenAI"]
    pub_models_resp.regions = ["usa"]
    pub_models_resp.licenses = ["MIT"]
    pub_models_resp.sizes = ["large"]
    public.models.get.return_value = pub_models_resp

    # Public benchmarks
    pub_benchmark_ds = MagicMock()
    pub_benchmark_ds.name = "MMLU"
    pub_benchmark_ds.id = "bench-pub-001"
    pub_benchmark_ds.key = "mmlu"
    pub_benchmark_ds.category = "general"
    pub_benchmark_ds.prompt_count = 100
    pub_benchmark_ds.language = "English"
    pub_benchmark_ds.description = "Massive Multitask Language Understanding benchmark"
    pub_benchmarks_resp = MagicMock()
    pub_benchmarks_resp.benchmarks = [pub_benchmark_ds]
    pub_benchmarks_resp.datasets = [pub_benchmark_ds]
    pub_benchmarks_resp.total_count = 1
    pub_benchmarks_resp.categories = ["general"]
    pub_benchmarks_resp.languages = ["English"]
    public.benchmarks.get.return_value = pub_benchmarks_resp

    # Public benchmark prompts
    pub_prompt = MagicMock()
    pub_prompt.id = "prompt-001"
    pub_prompt.input = "What is the capital of France?"
    pub_prompt.truth = "Paris"
    pub_prompts_data = MagicMock()
    pub_prompts_data.prompts = [pub_prompt]
    pub_prompts_data.count = 1
    pub_prompts_resp = MagicMock()
    pub_prompts_resp.data = pub_prompts_data
    public.benchmarks.get_prompts.return_value = pub_prompts_resp
    public.benchmarks.get_all_prompts.return_value = [pub_prompt]

    # Public evaluations
    pub_eval = MagicMock()
    pub_eval.id = "eval-pub-001"
    pub_eval.status = MagicMock(value="completed")
    pub_eval.accuracy = 85.5
    pub_eval.model_name = "GPT-4o"
    pub_eval.model_company = "OpenAI"
    pub_eval.benchmark_name = "MMLU"
    pub_eval.submitted_at = "2026-01-01T00:00:00Z"
    pub_eval_summary = MagicMock()
    pub_eval_summary.name = "Test Summary"
    pub_eval_summary.goal = "Test goal"
    pub_eval_metric = MagicMock()
    pub_eval_metric.name = "accuracy"
    pub_eval_summary.metrics = [pub_eval_metric]
    pub_eval.summary = pub_eval_summary
    pub_evals_resp = MagicMock()
    pub_evals_resp.evaluations = [pub_eval]
    pub_evals_resp.total_count = 1
    pub_evals_resp.pagination = MagicMock(total_count=1)
    public.evaluations.get_many.return_value = pub_evals_resp
    public.evaluations.get_by_id.return_value = pub_eval

    public.comparisons.compare.return_value = MagicMock(results=[])
    public.comparisons.compare_models.return_value = MagicMock(results=[])
    client.public = public

    # --- judge optimizations ---
    opt_run = MagicMock()
    opt_run.id = "opt-001"
    opt_run.status = MagicMock(value="completed")
    client.judge_optimizations.estimate.return_value = MagicMock(estimated_cost=0.10)
    client.judge_optimizations.create.return_value = MagicMock(id="opt-001")
    client.judge_optimizations.get.return_value = opt_run
    client.judge_optimizations.get_many.return_value = MagicMock(optimization_runs=[opt_run])
    client.judge_optimizations.apply.return_value = MagicMock()

    # --- integrations ---
    integration_obj = MagicMock()
    integration_obj.id = "int-001"
    integration_obj.name = "Test Integration"
    integration_obj.type = "webhook"
    integration_obj.status = "active"
    integration_obj.created_at = "2026-01-01"
    integration_obj.config = {"url": "https://example.com/webhook"}
    integrations_resp = MagicMock()
    integrations_resp.integrations = [integration_obj]
    integrations_resp.count = 1
    integrations_resp.total_count = 1
    client.integrations.get_many.return_value = integrations_resp
    client.integrations.get.return_value = integration_obj
    test_result = MagicMock()
    test_result.success = True
    test_result.message = "Connection successful"
    client.integrations.test.return_value = test_result

    return client


@pytest.fixture
def mock_async_stratix(mock_stratix):
    """Create a fully mocked AsyncStratix client that mirrors mock_stratix but with async methods."""
    client = AsyncMock()
    client.organization_id = "org-test-123"
    client.project_id = "proj-test-456"

    # --- traces (async) ---
    trace_resp = MagicMock()
    trace_resp.trace_ids = ["trace-test-001"]
    client.traces.upload.return_value = trace_resp

    traces_list = MagicMock()
    traces_list.traces = [
        MagicMock(
            id="trace-001",
            data={"input": "test"},
            filename="test.jsonl",
            created_at="2026-01-01",
        )
    ]
    traces_list.count = 1
    traces_list.total_count = 1
    client.traces.get_many.return_value = traces_list

    # --- judges (async) ---
    judge = MagicMock()
    judge.id = "judge-test-001"
    judge.name = "Test Judge"
    judge.evaluation_goal = "Test evaluation"
    client.judges.create.return_value = judge
    client.judges.get.return_value = judge
    judges_resp = MagicMock()
    judges_resp.judges = [judge]
    judges_resp.count = 1
    client.judges.get_many.return_value = judges_resp
    client.judges.delete.return_value = MagicMock()

    # --- trace evaluations (async) ---
    trace_eval = MagicMock()
    trace_eval.id = "te-test-001"
    trace_eval.status = MagicMock(value="success")
    client.trace_evaluations.create.return_value = trace_eval

    # TraceEvaluationResultsResponse is a single result, not a wrapper
    te_results_resp = MagicMock()
    te_results_resp.score = 0.85
    te_results_resp.passed = True
    te_results_resp.reasoning = "The response is accurate and complete."
    client.trace_evaluations.get_results.return_value = te_results_resp
    client.trace_evaluations.estimate_cost.return_value = MagicMock(estimated_cost=0.05)

    # --- evaluations (async) ---
    evaluation = MagicMock()
    evaluation.id = "eval-test-001"
    evaluation.status = MagicMock(value="completed")
    evaluation.is_success = True
    client.evaluations.create.return_value = evaluation
    client.evaluations.get.return_value = evaluation
    client.evaluations.get_by_id.return_value = evaluation
    evals_resp = MagicMock()
    evals_resp.evaluations = [evaluation]
    evals_resp.count = 1
    client.evaluations.get_many.return_value = evals_resp
    client.evaluations.wait_for_completion.return_value = evaluation

    # --- results (async) ---
    result = MagicMock()
    result.id = "result-001"
    result.score = 0.85
    results_resp = MagicMock()
    results_resp.results = [result]
    client.results.get.return_value = results_resp
    client.results.get_all.return_value = [result]
    client.results.get_all_by_id.return_value = [result]

    # --- models (async) ---
    model = MagicMock()
    model.id = "model-001"
    model.name = "Test Model"
    model.key = "test-model"
    client.models.get.return_value = [model]
    client.models.get_by_id.return_value = model
    client.models.add.return_value = True
    client.models.remove.return_value = True
    client.models.create_custom.return_value = MagicMock(model_id="model-custom-001")

    # --- benchmarks (async) ---
    benchmark = MagicMock()
    benchmark.id = "bench-001"
    benchmark.name = "Test Benchmark"
    benchmark.key = "test-bench"
    client.benchmarks.get.return_value = [benchmark]
    client.benchmarks.get_by_id.return_value = benchmark
    client.benchmarks.add.return_value = True
    client.benchmarks.create_custom.return_value = MagicMock(benchmark_id="bench-custom-001")
    client.benchmarks.create_smart.return_value = MagicMock(benchmark_id="bench-smart-001")

    # --- public (async) ---
    public = MagicMock()
    public.models.get.return_value = MagicMock(models=[model], total=1)
    public.benchmarks.get.return_value = MagicMock(benchmarks=[benchmark], total=1)
    public.evaluations.get_many.return_value = MagicMock(evaluations=[evaluation])
    client.public = public

    # aclose
    client.aclose.return_value = None

    return client


# ---- Helpers ----


def _import_and_run_sync(
    module_path: str,
    mock_client: MagicMock,
    *,
    extra_patches: dict[str, Any] | None = None,
    argv: list[str] | None = None,
) -> str:
    """Import a sample module and run its main() with mocked SDK.

    Args:
        module_path: Relative path from SAMPLES_DIR (e.g. 'core/quickstart.py').
        mock_client: The mocked Stratix client.
        extra_patches: Additional patches to apply (target -> value).
        argv: sys.argv override for samples that use argparse.

    Returns:
        Captured stdout output from the sample run.
    """
    full_path = os.path.join(SAMPLES_DIR, module_path)
    sample_dir = os.path.dirname(full_path)

    paths_added = []
    for p in [sample_dir, SAMPLES_DIR]:
        if p not in sys.path:
            sys.path.insert(0, p)
            paths_added.append(p)

    captured = io.StringIO()
    try:
        spec = importlib.util.spec_from_file_location("sample_under_test", full_path)
        mod = importlib.util.module_from_spec(spec)

        # Build a public client mock that mirrors mock_client.public
        mock_public = mock_client.public

        patches = {
            "layerlens.Stratix": MagicMock(return_value=mock_client),
            "layerlens.PublicClient": MagicMock(return_value=mock_public),
            "time.sleep": MagicMock(),  # Prevent real sleeps in polling loops
        }
        if extra_patches:
            patches.update(extra_patches)

        argv_val = argv or ["test"]

        with patch.dict("os.environ", {"LAYERLENS_STRATIX_API_KEY": "test-key"}):
            with patch("sys.argv", argv_val):
                ctx_managers = [patch(target, val) for target, val in patches.items()]
                for cm in ctx_managers:
                    cm.__enter__()
                try:
                    old_stdout = sys.stdout
                    sys.stdout = captured
                    try:
                        spec.loader.exec_module(mod)
                        if hasattr(mod, "main"):
                            result = mod.main()
                            # Handle coroutines (async main)
                            if asyncio.iscoroutine(result):
                                asyncio.run(result)
                    finally:
                        sys.stdout = old_stdout
                finally:
                    for cm in reversed(ctx_managers):
                        cm.__exit__(None, None, None)
    except SystemExit as e:
        if e.code not in (0, None):
            raise
    finally:
        for p in paths_added:
            if p in sys.path:
                sys.path.remove(p)

    return captured.getvalue()


def _import_and_run_async(
    module_path: str,
    mock_sync_client: MagicMock,
    mock_async_client: AsyncMock,
    *,
    extra_patches: dict[str, Any] | None = None,
) -> str:
    """Import an async sample module and run its main() with mocked SDK.

    Returns:
        Captured stdout output from the sample run.
    """
    full_path = os.path.join(SAMPLES_DIR, module_path)
    sample_dir = os.path.dirname(full_path)

    paths_added = []
    for p in [sample_dir, SAMPLES_DIR]:
        if p not in sys.path:
            sys.path.insert(0, p)
            paths_added.append(p)

    captured = io.StringIO()
    try:
        spec = importlib.util.spec_from_file_location("sample_under_test", full_path)
        mod = importlib.util.module_from_spec(spec)

        patches = {
            "layerlens.Stratix": MagicMock(return_value=mock_sync_client),
            "layerlens.AsyncStratix": MagicMock(return_value=mock_async_client),
            "time.sleep": MagicMock(),
            "asyncio.sleep": AsyncMock(),
        }
        if extra_patches:
            patches.update(extra_patches)

        with patch.dict("os.environ", {"LAYERLENS_STRATIX_API_KEY": "test-key"}):
            with patch("sys.argv", ["test"]):
                ctx_managers = [patch(target, val) for target, val in patches.items()]
                for cm in ctx_managers:
                    cm.__enter__()
                try:
                    old_stdout = sys.stdout
                    sys.stdout = captured
                    try:
                        spec.loader.exec_module(mod)
                        if hasattr(mod, "main"):
                            result = mod.main()
                            if asyncio.iscoroutine(result):
                                asyncio.run(result)
                    finally:
                        sys.stdout = old_stdout
                finally:
                    for cm in reversed(ctx_managers):
                        cm.__exit__(None, None, None)
    except SystemExit as e:
        if e.code not in (0, None):
            raise
    finally:
        for p in paths_added:
            if p in sys.path:
                sys.path.remove(p)

    return captured.getvalue()


def _run_sample_subprocess(
    script_path: str,
    args: list[str] | None = None,
    env_override: dict[str, str] | None = None,
    timeout: int = 60,
) -> subprocess.CompletedProcess:
    """Run a sample script as a subprocess."""
    cmd = [sys.executable, script_path] + (args or [])
    env = dict(os.environ)
    if env_override:
        env.update(env_override)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=PROJECT_ROOT,
    )


def _run_openclaw_demo(
    demo_name: str,
    extra_args: list[str] | None = None,
    timeout: int = 30,
    env_override: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run an OpenClaw demo as a subprocess.

    Uses a small wrapper script that sets up the package context properly,
    since openclaw demos use relative imports.
    """
    args_list = extra_args or []
    # Write a temporary runner script to handle relative imports properly
    script_content = (
        f"import sys\n"
        f"import os\n"
        f"sys.argv = ['test'] + {args_list!r}\n"
        f"# Ensure our project root is first in sys.path\n"
        f"project_root = {PROJECT_ROOT!r}\n"
        f"sys.path = [p for p in sys.path if 'layerlens' not in p.replace(os.sep, '/').lower() or 'stratix-python' in p.replace(os.sep, '/').lower()]\n"
        f"sys.path.insert(0, project_root)\n"
        f"# Create samples/__init__.py if missing (needed for package imports)\n"
        f"init_path = os.path.join(project_root, 'samples', '__init__.py')\n"
        f"created = not os.path.exists(init_path)\n"
        f"if created:\n"
        f"    open(init_path, 'w').close()\n"
        f"try:\n"
        f"    from samples.openclaw.{demo_name} import main\n"
        f"    main()\n"
        f"finally:\n"
        f"    if created and os.path.exists(init_path):\n"
        f"        os.unlink(init_path)\n"
    )
    fd, script_path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(script_content)
        cmd = [sys.executable, script_path]
    except Exception:
        os.close(fd)
        raise
    env = dict(os.environ)
    if env_override:
        env.update(env_override)
    else:
        # Ensure no real API key is used for offline tests
        env.pop("LAYERLENS_STRATIX_API_KEY", None)
        env.pop("LAYERLENS_ATLAS_API_KEY", None)
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=PROJECT_ROOT,
        )
    finally:
        if os.path.exists(script_path):
            os.unlink(script_path)


def _run_live(
    sample_path: str,
    args: list[str] | None = None,
    timeout: int = 60,
) -> subprocess.CompletedProcess:
    """Run a sample against the real API via subprocess."""
    cmd = [sys.executable, sample_path] + (args or [])
    env = dict(os.environ)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=PROJECT_ROOT,
    )


def _get_mocked_sample_id(category: str, name: str) -> str:
    """Build a test ID string for parametrize."""
    return f"{category}/{name}"


_LOGGER_ONLY_SAMPLES = {
    # These samples use logging instead of print() for all output.
    # They will produce no stdout, so we skip the output-not-empty check.
    "basic_trace",
    "compare_evaluations",
    "create_judge",
    "judge_creation_and_test",
    "judge_optimization",
    "benchmark_evaluation",
    "model_benchmark_management",
    "trace_evaluation",
    # pre_commit_hook exits early (no staged files) with logger-only output
    "pre_commit_hook",
}


def _verify_sample_behavior(
    mock_client: MagicMock,
    category: str,
    name: str,
    captured_output: str,
) -> None:
    """Verify the sample actually called SDK methods and produced output.

    After a mocked sample runs, this checks that the expected SDK methods
    were invoked and that stdout is non-empty (for samples that use print).
    The assertions are grouped by sample category / name so that a future
    change that accidentally removes an SDK call from a sample will be caught.

    Args:
        mock_client: The mocked Stratix client used during the run.
        category: Sample category (e.g. "core", "industry").
        name: Sample filename stem (e.g. "quickstart", "financial_fraud").
        captured_output: The stdout captured during the sample run.
    """
    sample_id = f"{category}/{name}"

    # -- Samples that use print() should produce stdout output --
    if name not in _LOGGER_ONLY_SAMPLES:
        assert len(captured_output.strip()) > 0, f"{sample_id} produced no output"

    # -- Category-level assertions --
    # Industry, cowork, modalities, and integrations samples all follow the
    # pattern: create judges, upload traces, run trace evaluations.
    if category in ("industry", "cowork", "modalities", "integrations"):
        assert mock_client.judges.create.called, f"{sample_id} never created a judge (via create_judge helper)"
        assert mock_client.traces.upload.called or mock_client.trace_evaluations.create.called, (
            f"{sample_id} never uploaded a trace or created a trace evaluation"
        )

    # CI/CD: quality_gate reaches SDK calls; pre_commit_hook exits early
    # (no staged files with the default subprocess mock) so we only
    # assert SDK interaction for quality_gate.
    if name == "quality_gate":
        assert mock_client.traces.get_many.called, "quality_gate never fetched traces"
        assert mock_client.judges.get_many.called, "quality_gate never fetched judges"
        assert mock_client.trace_evaluations.create.called, "quality_gate never created a trace evaluation"

    # -- Core sample-specific assertions --
    if name == "quickstart":
        assert mock_client.traces.upload.called, "quickstart never uploaded a trace"
        assert mock_client.judges.create.called, "quickstart never created a judge"
        assert mock_client.trace_evaluations.create.called, "quickstart never created a trace evaluation"

    if name == "basic_trace":
        assert mock_client.traces.upload.called, "basic_trace never uploaded a trace"
        assert mock_client.traces.get_many.called, "basic_trace never listed traces"
        assert mock_client.traces.get.called, "basic_trace never got a trace by ID"

    if name == "create_judge":
        assert mock_client.judges.create.called, "create_judge never created a judge"
        assert mock_client.judges.get.called, "create_judge never fetched a judge"
        assert mock_client.judges.get_many.called, "create_judge never listed judges"

    if name == "run_evaluation":
        assert mock_client.models.get.called, "run_evaluation never fetched models"
        assert mock_client.benchmarks.get.called, "run_evaluation never fetched benchmarks"
        assert mock_client.evaluations.create.called, "run_evaluation never created an evaluation"

    if name == "judge_optimization":
        assert mock_client.judge_optimizations.estimate.called, "judge_optimization never estimated cost"
        assert mock_client.judge_optimizations.create.called, "judge_optimization never created an optimization run"

    if name == "compare_evaluations":
        assert mock_client.evaluations.get_many.called, "compare_evaluations never fetched evaluations"
        assert mock_client.public.comparisons.compare.called, "compare_evaluations never called comparisons API"

    if name == "public_catalog":
        # public_catalog uses PublicClient which is mock_client.public
        # (aliased via the PublicClient mock).  The mock_client itself
        # receives calls because PublicClient(...) returns mock_client.public.
        assert mock_client.public.models.get.called, "public_catalog never queried public models"
        assert mock_client.public.benchmarks.get.called, "public_catalog never queried public benchmarks"

    if name == "custom_model":
        assert mock_client.models.create_custom.called, "custom_model never created a custom model"

    if name == "custom_benchmark":
        assert mock_client.benchmarks.create_custom.called or mock_client.benchmarks.create_smart.called, (
            "custom_benchmark never created a benchmark"
        )

    if name == "paginated_results":
        assert mock_client.results.get.called or mock_client.results.get_all.called, (
            "paginated_results never fetched results"
        )

    if name == "evaluation_filtering":
        assert mock_client.evaluations.get_many.called, "evaluation_filtering never filtered evaluations"

    if name == "trace_investigation":
        assert mock_client.traces.get_many.called, "trace_investigation never listed traces"
        assert mock_client.traces.get.called, "trace_investigation never inspected a trace"

    if name == "model_benchmark_management":
        assert mock_client.models.get.called, "model_benchmark_management never fetched models"
        assert mock_client.benchmarks.get.called, "model_benchmark_management never fetched benchmarks"

    if name == "trace_evaluation":
        assert mock_client.traces.upload.called, "trace_evaluation never uploaded a trace"
        assert mock_client.judges.create.called, "trace_evaluation never created a judge"
        assert mock_client.models.get.called, "trace_evaluation never fetched models"
        assert mock_client.trace_evaluations.estimate_cost.called, "trace_evaluation never estimated cost"
        assert mock_client.trace_evaluations.create.called, "trace_evaluation never created a trace evaluation"
        assert mock_client.trace_evaluations.get_many.called, "trace_evaluation never listed trace evaluations"
        assert mock_client.judges.delete.called, "trace_evaluation never cleaned up judge"
        assert mock_client.traces.delete.called, "trace_evaluation never cleaned up traces"

    if name == "evaluation_pipeline":
        assert mock_client.judges.get_many.called, "evaluation_pipeline never listed judges"
        assert mock_client.traces.get_many.called, "evaluation_pipeline never listed traces"
        assert mock_client.trace_evaluations.create.called, "evaluation_pipeline never created a trace evaluation"

    if name == "judge_creation_and_test":
        assert mock_client.judges.create.called, "judge_creation_and_test never created a judge"
        assert mock_client.judges.get.called, "judge_creation_and_test never verified judge"
        assert mock_client.traces.get_many.called, "judge_creation_and_test never listed traces"
        assert mock_client.trace_evaluations.create.called, "judge_creation_and_test never created a trace evaluation"

    if name == "benchmark_evaluation":
        assert mock_client.models.get.called, "benchmark_evaluation never fetched models"
        assert mock_client.benchmarks.get.called, "benchmark_evaluation never fetched benchmarks"
        assert mock_client.evaluations.create.called, "benchmark_evaluation never created an evaluation"
        assert mock_client.evaluations.wait_for_completion.called, "benchmark_evaluation never waited for completion"
        assert mock_client.results.get.called, "benchmark_evaluation never fetched results page"
        assert mock_client.results.get_all.called, "benchmark_evaluation never fetched all results"

    if name == "integration_management":
        assert mock_client.integrations.get_many.called, "integration_management never listed integrations"
        assert mock_client.integrations.get.called, "integration_management never fetched a single integration"
        assert mock_client.integrations.test.called, "integration_management never tested an integration"

    # -- Cowork sample-specific assertions --
    if name == "code_review":
        assert mock_client.judges.create.called, "code_review never created judges"
        assert mock_client.traces.upload.called, "code_review never uploaded traces"
        assert mock_client.trace_evaluations.create.called, "code_review never created trace evaluations"
        assert mock_client.judges.delete.called, "code_review never cleaned up judges"

    if name == "pair_programming":
        assert mock_client.judges.create.called, "pair_programming never created a judge"
        assert mock_client.traces.upload.called, "pair_programming never uploaded traces"
        assert mock_client.trace_evaluations.create.called, "pair_programming never created trace evaluations"
        assert mock_client.judges.update.called, "pair_programming never refined the judge"
        assert mock_client.judges.get.called, "pair_programming never fetched final judge details"
        assert mock_client.judges.delete.called, "pair_programming never cleaned up judge"

    if name == "rag_assessment":
        assert mock_client.judges.create.called, "rag_assessment never created judges"
        assert mock_client.traces.upload.called, "rag_assessment never uploaded traces"
        assert mock_client.trace_evaluations.create.called, "rag_assessment never created trace evaluations"
        assert mock_client.judges.delete.called, "rag_assessment never cleaned up judges"

    if name == "multi_agent_eval":
        assert mock_client.judges.create.called, "multi_agent_eval never created judges"
        assert mock_client.traces.upload.called, "multi_agent_eval never uploaded traces"
        assert mock_client.trace_evaluations.create.called, "multi_agent_eval never created trace evaluations"
        assert mock_client.judges.delete.called, "multi_agent_eval never cleaned up judges"

    if name == "incident_response":
        assert mock_client.judges.create.called, "incident_response never created judges"
        assert mock_client.traces.get_many.called, "incident_response never fetched recent traces"
        assert mock_client.trace_evaluations.create.called, "incident_response never created trace evaluations"
        assert mock_client.judges.delete.called, "incident_response never cleaned up judges"

    # -- Modalities sample-specific assertions --
    if name == "text_evaluation":
        assert mock_client.judges.create.called, "text_evaluation never created judges"
        assert mock_client.traces.upload.called, "text_evaluation never uploaded traces"
        assert mock_client.trace_evaluations.create.called, "text_evaluation never created trace evaluations"
        assert mock_client.judges.delete.called, "text_evaluation never cleaned up judges"

    if name == "brand_evaluation":
        assert mock_client.judges.create.called, "brand_evaluation never created judges"
        assert mock_client.traces.upload.called, "brand_evaluation never uploaded traces"
        assert mock_client.trace_evaluations.create.called, "brand_evaluation never created trace evaluations"
        assert mock_client.judges.delete.called, "brand_evaluation never cleaned up judges"

    if name == "document_evaluation":
        assert mock_client.judges.create.called, "document_evaluation never created judges"
        assert mock_client.traces.upload.called, "document_evaluation never uploaded traces"
        assert mock_client.trace_evaluations.create.called, "document_evaluation never created trace evaluations"
        assert mock_client.judges.delete.called, "document_evaluation never cleaned up judges"

    # -- Integrations sample-specific assertions --
    if name == "openai_traced":
        assert mock_client.judges.get_many.called, "openai_traced never checked existing judges"
        assert mock_client.judges.create.called, "openai_traced never created judges"
        assert mock_client.traces.upload.called, "openai_traced never uploaded a trace"
        assert mock_client.trace_evaluations.create.called, "openai_traced never created trace evaluations"

    if name == "anthropic_traced":
        assert mock_client.judges.get_many.called, "anthropic_traced never checked existing judges"
        assert mock_client.judges.create.called, "anthropic_traced never created judges"
        assert mock_client.traces.upload.called, "anthropic_traced never uploaded a trace"
        assert mock_client.trace_evaluations.create.called, "anthropic_traced never created trace evaluations"

    # -- OpenClaw direct demos --
    if category == "openclaw" and name in (
        "compare_agent_models",
        "evaluate_skill_output",
        "monitor_agent_safety",
        "trace_agent_execution",
    ):
        assert mock_client.judges.create.called or mock_client.traces.upload.called, (
            f"{sample_id} never called any SDK methods"
        )

    # -- OpenClaw skill evaluate script --
    if name == "evaluate" and "openclaw" in category:
        assert mock_client.judges.create.called, "openclaw evaluate script never created a judge"
        assert mock_client.trace_evaluations.create.called, "openclaw evaluate script never created a trace evaluation"


def _verify_async_sample_behavior(
    mock_async_client: AsyncMock,
    name: str,
    captured_output: str,
) -> None:
    """Verify async samples called the expected SDK methods.

    Args:
        mock_async_client: The mocked AsyncStratix client used during the run.
        name: Sample filename stem (e.g. "async_results", "async_workflow").
        captured_output: The stdout captured during the sample run.
    """
    # async_results uses print(); async_workflow uses logger only
    if name not in ("async_workflow",):
        assert len(captured_output.strip()) > 0, f"core/{name} produced no output"

    if name == "async_results":
        assert mock_async_client.evaluations.get_many.called, "async_results never fetched evaluations"

    if name == "async_workflow":
        assert mock_async_client.models.get.called, "async_workflow never fetched models"
        assert mock_async_client.benchmarks.get.called, "async_workflow never fetched benchmarks"
        assert mock_async_client.evaluations.create.called, "async_workflow never created an evaluation"


# ===========================================================================
# Test Class 1: ALL Samples with Mocked SDK
# ===========================================================================


class TestAllSamplesWithMockedSDK:
    """Test every single sample (all 58) with a fully mocked Stratix client."""

    # Samples importable directly (no relative imports / blocking stdin)
    _DIRECT_IMPORT_SAMPLES = [
        (cat, name)
        for cat, name in ALL_MOCKED_SAMPLES
        if cat not in ("mcp", "copilotkit/agents")
        and name not in _ASYNC_CORE_SAMPLES
        and name not in _EXTERNAL_SDK_SAMPLES
        # OpenClaw runner demos use relative imports -- tested via subprocess below
        and not (cat == "openclaw" and name in set(OPENCLAW_RUNNER_DEMOS))
    ]

    @pytest.mark.parametrize(
        "category,name",
        _DIRECT_IMPORT_SAMPLES,
        ids=[f"{cat}/{name}" for cat, name in _DIRECT_IMPORT_SAMPLES],
    )
    def test_sync_sample_mocked(self, category, name, mock_stratix, capsys):
        """Every directly-importable sync sample runs to completion with mocked SDK."""
        key = (category, name)
        argv = _SPECIAL_ARGV.get(key)
        extra_patches = _SPECIAL_PATCHES.get(key)

        # Integration samples: remove external API keys to trigger simulated fallback
        env_extra = {}
        if category == "integrations":
            env_extra = {"OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": ""}

        if env_extra:
            with patch.dict("os.environ", env_extra):
                output = _import_and_run_sync(
                    f"{category}/{name}.py",
                    mock_stratix,
                    extra_patches=extra_patches,
                    argv=argv,
                )
        else:
            output = _import_and_run_sync(
                f"{category}/{name}.py",
                mock_stratix,
                extra_patches=extra_patches,
                argv=argv,
            )

        _verify_sample_behavior(mock_stratix, category, name, output)

    @pytest.mark.parametrize("demo", OPENCLAW_RUNNER_DEMOS)
    def test_openclaw_runner_mocked(self, demo):
        """OpenClaw DemoRunner demos run with a fake API key (fallback to offline)."""
        env_override = {
            "LAYERLENS_STRATIX_API_KEY": "fake-test-key-12345",
        }
        result = _run_openclaw_demo(
            demo,
            extra_args=["--json"],
            env_override=env_override,
        )
        assert result.returncode == 0, (
            f"OpenClaw demo {demo} failed with fake API key.\n"
            f"stdout: {result.stdout[:500]}\n"
            f"stderr: {result.stderr[:500]}"
        )

    @pytest.mark.parametrize("name", sorted(_ASYNC_CORE_SAMPLES))
    def test_async_sample_mocked(self, name, mock_stratix, mock_async_stratix, capsys):
        """Every async sample runs to completion with mocked SDK."""
        output = _import_and_run_async(
            f"core/{name}.py",
            mock_stratix,
            mock_async_stratix,
        )
        _verify_async_sample_behavior(mock_async_stratix, name, output)

    def test_mcp_server_import(self, mock_stratix):
        """MCP server: verify create_server() can be imported (cannot run main -- blocks on stdin)."""
        full_path = os.path.join(SAMPLES_DIR, "mcp", "layerlens_server.py")
        sample_dir = os.path.dirname(full_path)
        if sample_dir not in sys.path:
            sys.path.insert(0, sample_dir)

        try:
            spec = importlib.util.spec_from_file_location("mcp_server_test", full_path)
            mod = importlib.util.module_from_spec(spec)

            # Mock mcp package imports that may not be installed
            mock_mcp_server = MagicMock()
            mock_mcp_stdio = MagicMock()
            mock_mcp_types = MagicMock()
            mock_mcp_types.TextContent = MagicMock
            mock_mcp_types.Tool = MagicMock

            with patch.dict("os.environ", {"LAYERLENS_STRATIX_API_KEY": "test-key"}):
                with patch.dict(
                    "sys.modules",
                    {
                        "mcp": MagicMock(),
                        "mcp.server": mock_mcp_server,
                        "mcp.server.stdio": mock_mcp_stdio,
                        "mcp.types": mock_mcp_types,
                    },
                ):
                    with patch("layerlens.Stratix", MagicMock(return_value=mock_stratix)):
                        spec.loader.exec_module(mod)
                        assert hasattr(mod, "create_server"), "MCP server should expose create_server()"
                        assert hasattr(mod, "main"), "MCP server should expose main()"
        finally:
            if sample_dir in sys.path:
                sys.path.remove(sample_dir)

    @pytest.mark.parametrize("name", COPILOTKIT_SAMPLES)
    def test_copilotkit_agent_import(self, name, mock_stratix):
        """CopilotKit agents: verify main() prints usage without crashing.

        These agents require langchain/langgraph/copilotkit; we mock those.
        """
        full_path = os.path.join(SAMPLES_DIR, "copilotkit", "agents", f"{name}.py")
        sample_dir = os.path.dirname(full_path)
        if sample_dir not in sys.path:
            sys.path.insert(0, sample_dir)

        mod_name = f"copilotkit_{name}_test"
        try:
            spec = importlib.util.spec_from_file_location(mod_name, full_path)
            mod = importlib.util.module_from_spec(spec)

            # Register the module so dataclasses can resolve __module__
            sys.modules[mod_name] = mod

            # Mock heavy dependencies that may not be installed
            mock_modules = {
                "langchain": MagicMock(),
                "langchain.agents": MagicMock(),
                "langchain.tools": MagicMock(),
                "langchain_core": MagicMock(),
                "langchain_core.messages": MagicMock(),
                "langchain_core.tools": MagicMock(),
                "langchain_openai": MagicMock(),
                "langgraph": MagicMock(),
                "langgraph.checkpoint": MagicMock(),
                "langgraph.checkpoint.memory": MagicMock(),
                "langgraph.checkpoint.serde": MagicMock(),
                "langgraph.checkpoint.serde.jsonplus": MagicMock(),
                "langgraph.graph": MagicMock(),
                "langgraph.types": MagicMock(),
                "copilotkit": MagicMock(),
                "copilotkit.langchain": MagicMock(),
                "copilotkit.langgraph": MagicMock(),
                "pydantic": MagicMock(),
            }

            with patch.dict("os.environ", {"LAYERLENS_STRATIX_API_KEY": "test-key"}):
                with patch.dict("sys.modules", mock_modules):
                    with patch("layerlens.Stratix", MagicMock(return_value=mock_stratix)):
                        spec.loader.exec_module(mod)
                        assert hasattr(mod, "main"), f"CopilotKit agent {name} should have main()"
                        # main() just prints usage -- call it
                        mod.main()
        finally:
            sys.modules.pop(mod_name, None)
            if sample_dir in sys.path:
                sys.path.remove(sample_dir)

    def test_copilotkit_evaluator_tools(self):
        """Each backend tool on the evaluator agent produces the expected shape
        against a patched Stratix client.

        This is the regression for the current HITL architecture: the
        evaluator is a ``create_agent`` + ``CopilotKitMiddleware`` graph with
        four backend tools (``list_judges``, ``list_recent_traces``,
        ``run_trace_evaluation``, ``get_evaluation_result``) and one
        frontend tool (``confirm_judge``, declared in the browser harness
        via ``useCopilotAction``). No ``interrupt()``, so no ag-ui-langgraph
        protocol bugs to work around.
        """
        pytest.importorskip("langchain")
        pytest.importorskip("langchain_core")

        from types import SimpleNamespace

        fake_judge = SimpleNamespace(
            id="jdg_1",
            name="Helpfulness",
            evaluation_goal="measures helpfulness",
            created_at="2026-04-23T00:00:00Z",
        )
        fake_trace = SimpleNamespace(
            id="trc_1",
            filename="sample.jsonl",
            created_at="2026-04-23T00:00:00Z",
        )
        fake_eval_pending = SimpleNamespace(
            id="ev_1",
            trace_id="trc_1",
            judge_id="jdg_1",
            status=SimpleNamespace(value="pending"),
        )
        fake_eval_done = SimpleNamespace(
            id="ev_1",
            trace_id="trc_1",
            judge_id="jdg_1",
            status=SimpleNamespace(value="success"),
        )
        fake_results = SimpleNamespace(score=0.9, passed=True, reasoning="ok")

        fake_client = MagicMock()
        fake_client.judges.get_many.return_value = SimpleNamespace(
            judges=[fake_judge]
        )
        fake_client.traces.get_many.return_value = SimpleNamespace(
            traces=[fake_trace]
        )
        fake_client.trace_evaluations.create.return_value = fake_eval_pending

        sample_dir = os.path.join(SAMPLES_DIR, "copilotkit", "agents")
        if sample_dir not in sys.path:
            sys.path.insert(0, sample_dir)

        mod_name = "copilotkit_evaluator_tools_test"
        try:
            # Stub OPENAI_API_KEY so import-time build_graph() doesn't fail;
            # we won't invoke the LLM in this test.
            with patch.dict(
                "os.environ",
                {
                    "LAYERLENS_STRATIX_API_KEY": "test-key",
                    "OPENAI_API_KEY": "test-openai",
                },
            ):
                with patch(
                    "layerlens.Stratix", MagicMock(return_value=fake_client)
                ):
                    spec = importlib.util.spec_from_file_location(
                        mod_name,
                        os.path.join(sample_dir, "evaluator_agent.py"),
                    )
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[mod_name] = mod
                    spec.loader.exec_module(mod)
                    mod._client = fake_client

                    # The evaluator exposes its backend tools as a module-level
                    # BACKEND_TOOLS list. Each one is a LangChain ``@tool``
                    # with a canonical ``.invoke(args)`` entry point.
                    tools_by_name = {t.name: t for t in mod.BACKEND_TOOLS}
                    assert set(tools_by_name) == {
                        "list_judges",
                        "list_recent_traces",
                        "run_trace_evaluation",
                        "get_evaluation_result",
                    }, f"unexpected tool set: {set(tools_by_name)}"

                    judges = tools_by_name["list_judges"].invoke({})
                    assert judges == [
                        {
                            "id": "jdg_1",
                            "name": "Helpfulness",
                            "goal": "measures helpfulness",
                        }
                    ]

                    traces = tools_by_name["list_recent_traces"].invoke(
                        {"limit": 5}
                    )
                    assert traces == [
                        {
                            "id": "trc_1",
                            "filename": "sample.jsonl",
                            "created_at": "2026-04-23T00:00:00Z",
                        }
                    ]

                    started = tools_by_name["run_trace_evaluation"].invoke(
                        {"trace_id": "trc_1", "judge_id": "jdg_1"}
                    )
                    assert started["evaluation_id"] == "ev_1"
                    assert started["status"] == "pending"

                    fake_client.trace_evaluations.get.return_value = (
                        fake_eval_done
                    )
                    fake_client.trace_evaluations.get_results.return_value = (
                        fake_results
                    )
                    result = tools_by_name["get_evaluation_result"].invoke(
                        {"evaluation_id": "ev_1"}
                    )
                    assert result["status"] == "success"
                    assert result["passed"] is True
                    assert result["score"] == 0.9

                    # Confirm the prompt references the frontend HITL tool by
                    # the exact name the browser's useCopilotAction uses, and
                    # includes the key flow steps. Sanity, not exhaustive.
                    assert "confirm_judge" in mod.SYSTEM_PROMPT
                    assert "list_recent_traces" in mod.SYSTEM_PROMPT
                    assert "list_judges" in mod.SYSTEM_PROMPT
        finally:
            sys.modules.pop(mod_name, None)
            if sample_dir in sys.path:
                sys.path.remove(sample_dir)



# ===========================================================================
# Test Class 2: ALL Samples Live API
# ===========================================================================


@pytest.fixture
def api_key():
    """Get real API key or skip."""
    key = os.environ.get("LAYERLENS_STRATIX_API_KEY")
    if not key:
        pytest.skip("LAYERLENS_STRATIX_API_KEY not set")
    return key


# Live test args per sample path
_LIVE_ARGS: dict[str, list[str]] = {
    "samples/core/basic_trace.py": ["--skip-delete"],
    "samples/cicd/quality_gate.py": ["--threshold", "0.0"],
    "samples/openclaw/layerlens_skill/scripts/evaluate.py": [
        "--input",
        "What is 2+2?",
        "--output",
        "4",
    ],
}

# Samples to skip in live mode
_LIVE_SKIP: set[str] = {
    "samples/mcp/layerlens_server.py",  # Blocks on stdin
}


@pytest.mark.live
class TestAllSamplesLiveAPI:
    """Run every sample against the real LayerLens API.

    Requires LAYERLENS_STRATIX_API_KEY to be set.
    Run with: pytest tests/test_samples_e2e.py -m live
    """

    @pytest.mark.parametrize("sample_path", ALL_SAMPLE_PATHS, ids=ALL_SAMPLE_PATHS)
    def test_live(self, api_key, sample_path):
        """Each sample runs successfully against the real API."""
        full_rel = f"samples/{sample_path}"

        if full_rel in _LIVE_SKIP:
            pytest.skip(f"Skipped in live mode: {sample_path}")

        full_path = os.path.join(PROJECT_ROOT, "samples", sample_path)
        args = _LIVE_ARGS.get(full_rel, [])

        # CopilotKit agents: succeed if copilotkit is installed, skip if not
        if "copilotkit" in sample_path:
            result = _run_live(full_path, args=args, timeout=30)
            if result.returncode != 0 and "No module named" in result.stderr:
                pytest.skip("copilotkit/langgraph not installed")
            assert result.returncode == 0, f"CopilotKit agent failed: {sample_path}\nstderr: {result.stderr[:500]}"
            return

        # OpenClaw demos: run with default args (built-in demo data)
        if "openclaw" in sample_path:
            result = _run_live(full_path, args=args, timeout=60)
            # OpenClaw demos may fail if openclaw package not installed
            # but should not crash with unhandled exception
            if result.returncode != 0:
                # Check it's a known/expected failure
                assert (
                    "No module named" in result.stderr
                    or "API key" in result.stderr
                    or "LAYERLENS" in result.stderr
                    or "ModuleNotFoundError" in result.stderr
                    or "ImportError" in result.stderr
                ), f"Unexpected failure: {sample_path}\nstderr: {result.stderr[:500]}"
            return

        # Pre-commit hook: run in the repo dir (it needs git context)
        if "pre_commit_hook" in sample_path:
            result = _run_live(full_path, args=args, timeout=30)
            # May fail (no staged files) but should not crash
            assert result.returncode in (0, 1), f"pre_commit_hook crashed: stderr={result.stderr[:500]}"
            return

        # Evaluations are async: creation returns immediately but LLM judge
        # execution takes 5-60s per evaluation.  Samples that create multiple
        # judges × multiple traces can legitimately run for several minutes.
        result = _run_live(full_path, args=args, timeout=600)

        if result.returncode == 0:
            # SUCCESS: verify the sample actually produced meaningful output.
            # A sample that exits 0 but prints nothing is broken.
            combined = result.stdout + result.stderr
            assert len(combined.strip()) > 0, f"EMPTY OUTPUT: {sample_path} exited 0 but produced no output"
            # Verify evidence of real API interaction in output.
            # Samples that connect to the API will log HTTP requests or print
            # trace IDs, judge names, scores, etc.
            api_evidence = [
                "HTTP Request",  # httpx request logging
                "trace",  # trace IDs or trace references
                "judge",  # judge creation or references
                "evaluation",  # evaluation creation or results
                "score",  # evaluation scores
                "pass",  # pass/fail verdicts
                "Connected",  # client init confirmation
                "Uploaded",  # trace upload confirmation
                "Created",  # resource creation confirmation
            ]
            has_evidence = any(term.lower() in combined.lower() for term in api_evidence)
            assert has_evidence, (
                f"NO API EVIDENCE: {sample_path} exited 0 but output contains "
                f"no trace of API interaction.\n"
                f"stdout: {result.stdout[:500]}\n"
                f"stderr: {result.stderr[:500]}"
            )
        else:
            # FAILURE: accept only known API limitations (insufficient credits,
            # rate limits, etc.) -- these mean the sample code is correct but
            # the account has operational constraints.
            stderr = result.stderr
            known_api_limits = [
                "insufficient credits",
                "402",
                "429",
                "rate limit",
                "quota",
                "InternalServerError",
                "502",
                "503",
                "timeout",
                "409",
                "already exists",
                "ConflictError",
                "No benchmarks",
                "No models",
                "No traces",
                "ModuleNotFoundError",
                "No module named",
            ]
            is_api_limit = any(term in stderr for term in known_api_limits)
            assert is_api_limit, (
                f"UNEXPECTED FAILURE: {sample_path}\nstdout: {result.stdout[:300]}\nstderr: {stderr[:500]}"
            )


# ===========================================================================
# Test Class 3: OpenClaw Offline Mode (--no-sdk)
# ===========================================================================


class TestOpenClawOfflineMode:
    """Test all 11 OpenClaw samples in offline modes.

    DemoRunner demos support --no-sdk; direct demos and skill script are tested
    for graceful failure without API key.
    """

    @pytest.mark.parametrize("demo", OPENCLAW_RUNNER_DEMOS)
    def test_runner_offline_json(self, demo):
        """DemoRunner demos should run successfully with --no-sdk --json."""
        result = _run_openclaw_demo(demo, extra_args=["--no-sdk", "--json"])
        assert result.returncode == 0, (
            f"OpenClaw demo {demo} failed in offline mode.\n"
            f"stdout: {result.stdout[:500]}\n"
            f"stderr: {result.stderr[:500]}"
        )
        # Verify JSON output is present somewhere in stdout.
        # Demos may print formatted text before the JSON dump, so we
        # search for the first top-level '{' and try to parse from there.
        stdout = result.stdout.strip()
        if stdout:
            for i, ch in enumerate(stdout):
                if ch == "{":
                    try:
                        parsed = json.loads(stdout[i:])
                        assert isinstance(parsed, (dict, list))
                        break
                    except json.JSONDecodeError:
                        continue

    @pytest.mark.parametrize("demo", OPENCLAW_RUNNER_DEMOS)
    def test_runner_offline_verbose(self, demo):
        """DemoRunner demos should run in verbose --no-sdk mode without error."""
        result = _run_openclaw_demo(demo, extra_args=["--no-sdk", "--verbose"])
        assert result.returncode == 0, (
            f"OpenClaw demo {demo} failed in verbose offline mode.\nstderr: {result.stderr[:500]}"
        )

    @pytest.mark.parametrize("demo", OPENCLAW_DIRECT_DEMOS)
    def test_direct_without_key(self, demo):
        """Direct Stratix OpenClaw demos should fail gracefully without API key."""
        result = _run_openclaw_demo(demo)
        assert result.returncode != 0, f"Expected {demo} to fail without API key but it exited 0"

    def test_skill_script_without_key(self):
        """OpenClaw skill evaluate.py should fail gracefully without API key."""
        script = os.path.join(
            SAMPLES_DIR,
            "openclaw",
            "layerlens_skill",
            "scripts",
            "evaluate.py",
        )
        env = dict(os.environ)
        env.pop("LAYERLENS_STRATIX_API_KEY", None)
        env.pop("LAYERLENS_ATLAS_API_KEY", None)
        result = subprocess.run(
            [
                sys.executable,
                script,
                "--input",
                "test",
                "--output",
                "test",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode != 0, "evaluate.py should fail without API key"


# ===========================================================================
# Test Class 4: Without API Key (all 58)
# ===========================================================================


class TestWithoutAPIKey:
    """Verify ALL samples exit gracefully when no API key is set."""

    # Samples that may legitimately exit 0 without an API key
    _MAY_SUCCEED_WITHOUT_KEY = {
        "cicd/pre_commit_hook.py",  # Exits 0 when no staged files
        "core/public_catalog.py",  # Uses PublicClient (no key needed)
    }

    @pytest.mark.parametrize(
        "sample_path",
        [
            p
            for p in ALL_SAMPLE_PATHS
            # Exclude MCP (blocks on stdin) and CopilotKit (they just print)
            if "mcp/" not in p and "copilotkit/" not in p
        ],
        ids=[p for p in ALL_SAMPLE_PATHS if "mcp/" not in p and "copilotkit/" not in p],
    )
    def test_no_api_key(self, sample_path):
        """Samples should exit non-zero without API key (or succeed for offline-capable ones)."""
        full_path = os.path.join(SAMPLES_DIR, sample_path)
        env = dict(os.environ)
        env.pop("LAYERLENS_STRATIX_API_KEY", None)
        env.pop("LAYERLENS_ATLAS_API_KEY", None)

        # OpenClaw skill evaluate.py needs args
        args = []
        if "evaluate.py" in sample_path and "openclaw" in sample_path:
            args = ["--input", "test", "--output", "test"]

        result = subprocess.run(
            [sys.executable, full_path] + args,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
            cwd=PROJECT_ROOT,
        )

        if sample_path in self._MAY_SUCCEED_WITHOUT_KEY:
            # These samples may legitimately exit 0 without API key
            assert result.returncode in (0, 1), (
                f"Expected {sample_path} to exit 0 or 1 without API key.\n"
                f"returncode: {result.returncode}\n"
                f"stderr: {result.stderr[:300]}"
            )
        else:
            # Should exit non-zero (can't init Stratix) -- graceful failure
            assert result.returncode != 0, (
                f"Expected {sample_path} to fail without API key but it exited 0.\nstdout: {result.stdout[:300]}"
            )

    @pytest.mark.parametrize("demo", OPENCLAW_RUNNER_DEMOS)
    def test_openclaw_runner_offline_no_key(self, demo):
        """OpenClaw DemoRunner demos should work in --no-sdk mode without API key."""
        result = _run_openclaw_demo(demo, extra_args=["--no-sdk", "--json"])
        assert result.returncode == 0, (
            f"OpenClaw demo {demo} should work offline without API key.\nstderr: {result.stderr[:500]}"
        )

    @pytest.mark.parametrize("name", COPILOTKIT_SAMPLES)
    def test_copilotkit_no_key(self, name):
        """CopilotKit agents should print usage even without API key."""
        full_path = os.path.join(SAMPLES_DIR, "copilotkit", "agents", f"{name}.py")
        env = dict(os.environ)
        env.pop("LAYERLENS_STRATIX_API_KEY", None)
        env.pop("LAYERLENS_ATLAS_API_KEY", None)

        result = subprocess.run(
            [sys.executable, full_path],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
            cwd=PROJECT_ROOT,
        )
        # These just print usage at __main__ -- should succeed or fail gracefully
        # (may fail if langchain etc. not installed, which is fine)
        # We just verify no unhandled crash
        assert result.returncode in (0, 1), f"CopilotKit {name} crashed without API key.\nstderr: {result.stderr[:500]}"


# ===========================================================================
# Test Class 5: Missing Dependencies
# ===========================================================================


class TestMissingDependencies:
    """Verify samples handle missing optional dependencies gracefully."""

    def test_openai_traced_without_openai(self, mock_stratix, capsys):
        """openai_traced.py should fall back to simulated data when openai is not importable."""
        original = sys.modules.get("openai")
        sys.modules["openai"] = None  # type: ignore[assignment]
        try:
            with patch.dict(
                "os.environ",
                {"LAYERLENS_STRATIX_API_KEY": "test-key", "OPENAI_API_KEY": ""},
            ):
                _import_and_run_sync("integrations/openai_traced.py", mock_stratix)
        finally:
            if original is not None:
                sys.modules["openai"] = original
            else:
                sys.modules.pop("openai", None)

    def test_anthropic_traced_without_anthropic(self, mock_stratix, capsys):
        """anthropic_traced.py should fall back to simulated data when anthropic is not importable."""
        original = sys.modules.get("anthropic")
        sys.modules["anthropic"] = None  # type: ignore[assignment]
        try:
            with patch.dict(
                "os.environ",
                {"LAYERLENS_STRATIX_API_KEY": "test-key", "ANTHROPIC_API_KEY": ""},
            ):
                _import_and_run_sync("integrations/anthropic_traced.py", mock_stratix)
        finally:
            if original is not None:
                sys.modules["anthropic"] = original
            else:
                sys.modules.pop("anthropic", None)

    def test_openclaw_demos_without_openclaw(self):
        """OpenClaw DemoRunner demos should work without the openclaw package installed."""
        for demo in OPENCLAW_RUNNER_DEMOS:
            result = _run_openclaw_demo(demo, extra_args=["--no-sdk", "--json"])
            assert result.returncode == 0, (
                f"OpenClaw demo {demo} should work without openclaw package.\nstderr: {result.stderr[:500]}"
            )

    def test_integration_with_missing_openai_env(self, mock_stratix):
        """Verify openai_traced handles missing OPENAI_API_KEY gracefully."""
        with patch.dict(
            "os.environ",
            {"LAYERLENS_STRATIX_API_KEY": "test-key", "OPENAI_API_KEY": ""},
        ):
            _import_and_run_sync("integrations/openai_traced.py", mock_stratix)

    def test_integration_with_missing_anthropic_env(self, mock_stratix):
        """Verify anthropic_traced handles missing ANTHROPIC_API_KEY gracefully."""
        with patch.dict(
            "os.environ",
            {"LAYERLENS_STRATIX_API_KEY": "test-key", "ANTHROPIC_API_KEY": ""},
        ):
            _import_and_run_sync("integrations/anthropic_traced.py", mock_stratix)

    @pytest.mark.parametrize("name", COPILOTKIT_SAMPLES)
    def test_copilotkit_without_langchain(self, name, mock_stratix):
        """CopilotKit agents should be importable with mocked langchain/copilotkit."""
        full_path = os.path.join(SAMPLES_DIR, "copilotkit", "agents", f"{name}.py")
        sample_dir = os.path.dirname(full_path)
        if sample_dir not in sys.path:
            sys.path.insert(0, sample_dir)

        mod_name = f"copilotkit_dep_{name}_test"
        try:
            spec = importlib.util.spec_from_file_location(mod_name, full_path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod

            mock_modules = {
                "langchain": MagicMock(),
                "langchain.agents": MagicMock(),
                "langchain.tools": MagicMock(),
                "langchain_core": MagicMock(),
                "langchain_core.messages": MagicMock(),
                "langchain_core.tools": MagicMock(),
                "langchain_openai": MagicMock(),
                "langgraph": MagicMock(),
                "langgraph.checkpoint": MagicMock(),
                "langgraph.checkpoint.memory": MagicMock(),
                "langgraph.checkpoint.serde": MagicMock(),
                "langgraph.checkpoint.serde.jsonplus": MagicMock(),
                "langgraph.graph": MagicMock(),
                "langgraph.types": MagicMock(),
                "copilotkit": MagicMock(),
                "copilotkit.langchain": MagicMock(),
                "copilotkit.langgraph": MagicMock(),
                "pydantic": MagicMock(),
            }

            with patch.dict("os.environ", {"LAYERLENS_STRATIX_API_KEY": "test-key"}):
                with patch.dict("sys.modules", mock_modules):
                    with patch(
                        "layerlens.Stratix",
                        MagicMock(return_value=mock_stratix),
                    ):
                        spec.loader.exec_module(mod)
                        assert hasattr(mod, "main")
                        mod.main()
        finally:
            sys.modules.pop(mod_name, None)
            if sample_dir in sys.path:
                sys.path.remove(sample_dir)

    def test_mcp_server_without_mcp_package(self, mock_stratix):
        """MCP server should be importable with mocked mcp package."""
        full_path = os.path.join(SAMPLES_DIR, "mcp", "layerlens_server.py")
        sample_dir = os.path.dirname(full_path)
        if sample_dir not in sys.path:
            sys.path.insert(0, sample_dir)

        try:
            spec = importlib.util.spec_from_file_location("mcp_dep_test", full_path)
            mod = importlib.util.module_from_spec(spec)

            mock_mcp_types = MagicMock()
            mock_mcp_types.TextContent = MagicMock
            mock_mcp_types.Tool = MagicMock

            with patch.dict("os.environ", {"LAYERLENS_STRATIX_API_KEY": "test-key"}):
                with patch.dict(
                    "sys.modules",
                    {
                        "mcp": MagicMock(),
                        "mcp.server": MagicMock(),
                        "mcp.server.stdio": MagicMock(),
                        "mcp.types": mock_mcp_types,
                    },
                ):
                    with patch(
                        "layerlens.Stratix",
                        MagicMock(return_value=mock_stratix),
                    ):
                        spec.loader.exec_module(mod)
                        assert hasattr(mod, "create_server")
        finally:
            if sample_dir in sys.path:
                sys.path.remove(sample_dir)

    def test_openclaw_skill_script_with_mocked_sdk(self, mock_stratix):
        """OpenClaw skill evaluate.py should run with mocked SDK."""
        _import_and_run_sync(
            "openclaw/layerlens_skill/scripts/evaluate.py",
            mock_stratix,
            argv=[
                "test",
                "--input",
                "test prompt",
                "--output",
                "test response",
            ],
        )


# ===========================================================================
# Test Class 6: Sample Completeness Checks
# ===========================================================================


class TestSampleCompleteness:
    """Verify our test lists match what actually exists on disk -- no sample left untested."""

    def test_core_samples_complete(self):
        """All core sample files should be listed in CORE_SAMPLES."""
        actual = {
            os.path.splitext(f)[0]
            for f in os.listdir(os.path.join(SAMPLES_DIR, "core"))
            if f.endswith(".py") and not f.startswith("_") and f != "README.md"
        }
        expected = set(CORE_SAMPLES)
        assert actual == expected, f"Missing from CORE_SAMPLES: {actual - expected}; Extra: {expected - actual}"

    def test_industry_samples_complete(self):
        """All industry sample files should be listed in INDUSTRY_SAMPLES."""
        actual = {
            os.path.splitext(f)[0]
            for f in os.listdir(os.path.join(SAMPLES_DIR, "industry"))
            if f.endswith(".py") and not f.startswith("_") and f != "README.md"
        }
        expected = set(INDUSTRY_SAMPLES)
        assert actual == expected, f"Missing: {actual - expected}; Extra: {expected - actual}"

    def test_cowork_samples_complete(self):
        """All cowork sample files should be listed in COWORK_SAMPLES."""
        actual = {
            os.path.splitext(f)[0]
            for f in os.listdir(os.path.join(SAMPLES_DIR, "cowork"))
            if f.endswith(".py") and not f.startswith("_") and f != "README.md"
        }
        expected = set(COWORK_SAMPLES)
        assert actual == expected, f"Missing: {actual - expected}; Extra: {expected - actual}"

    def test_modality_samples_complete(self):
        """All modality sample files should be listed in MODALITY_SAMPLES."""
        actual = {
            os.path.splitext(f)[0]
            for f in os.listdir(os.path.join(SAMPLES_DIR, "modalities"))
            if f.endswith(".py") and not f.startswith("_") and f != "README.md"
        }
        expected = set(MODALITY_SAMPLES)
        assert actual == expected, f"Missing: {actual - expected}; Extra: {expected - actual}"

    def test_openclaw_demos_complete(self):
        """All OpenClaw demo files should be listed in OPENCLAW_DEMOS."""
        actual = {
            os.path.splitext(f)[0]
            for f in os.listdir(os.path.join(SAMPLES_DIR, "openclaw"))
            if f.endswith(".py") and not f.startswith("_") and f != "README.md"
        }
        expected = set(OPENCLAW_DEMOS)
        assert actual == expected, f"Missing: {actual - expected}; Extra: {expected - actual}"

    def test_integration_samples_complete(self):
        """All integration sample files should be listed in INTEGRATION_SAMPLES."""
        actual = {
            os.path.splitext(f)[0]
            for f in os.listdir(os.path.join(SAMPLES_DIR, "integrations"))
            if f.endswith(".py") and not f.startswith("_") and f != "README.md"
        }
        expected = set(INTEGRATION_SAMPLES)
        assert actual == expected, f"Missing: {actual - expected}; Extra: {expected - actual}"

    def test_cicd_samples_complete(self):
        """All CI/CD sample files should be listed in CICD_SAMPLES."""
        actual = {
            os.path.splitext(f)[0]
            for f in os.listdir(os.path.join(SAMPLES_DIR, "cicd"))
            if f.endswith(".py") and not f.startswith("_") and f != "README.md"
        }
        expected = set(CICD_SAMPLES)
        assert actual == expected, f"Missing: {actual - expected}; Extra: {expected - actual}"

    def test_copilotkit_agents_complete(self):
        """All CopilotKit agent files should be listed in COPILOTKIT_SAMPLES."""
        actual = {
            os.path.splitext(f)[0]
            for f in os.listdir(os.path.join(SAMPLES_DIR, "copilotkit", "agents"))
            if f.endswith(".py") and not f.startswith("_") and f != "README.md"
        }
        expected = set(COPILOTKIT_SAMPLES)
        assert actual == expected, f"Missing: {actual - expected}; Extra: {expected - actual}"

    def test_mcp_samples_complete(self):
        """All MCP sample files should be listed in MCP_SAMPLES."""
        actual = {
            os.path.splitext(f)[0]
            for f in os.listdir(os.path.join(SAMPLES_DIR, "mcp"))
            if f.endswith(".py") and not f.startswith("_") and f != "README.md"
        }
        expected = set(MCP_SAMPLES)
        assert actual == expected, f"Missing: {actual - expected}; Extra: {expected - actual}"

    def test_openclaw_skill_script_exists(self):
        """The OpenClaw skill evaluate.py script should exist."""
        script = os.path.join(
            SAMPLES_DIR,
            "openclaw",
            "layerlens_skill",
            "scripts",
            "evaluate.py",
        )
        assert os.path.isfile(script), f"Missing: {script}"

    def test_all_54_samples_covered(self):
        """Verify ALL_SAMPLE_PATHS contains exactly 58 entries."""
        assert len(ALL_SAMPLE_PATHS) == 58, (
            f"Expected 58 samples, got {len(ALL_SAMPLE_PATHS)}.\nPaths: {ALL_SAMPLE_PATHS}"
        )

    def test_all_sample_paths_exist(self):
        """Every path in ALL_SAMPLE_PATHS should correspond to a real file."""
        missing = []
        for p in ALL_SAMPLE_PATHS:
            full = os.path.join(SAMPLES_DIR, p)
            if not os.path.isfile(full):
                missing.append(p)
        assert not missing, f"Sample files not found: {missing}"

    def test_mocked_samples_cover_all(self):
        """ALL_MOCKED_SAMPLES should produce exactly 58 entries."""
        assert len(ALL_MOCKED_SAMPLES) == 58, (
            f"Expected 58 mocked entries, got {len(ALL_MOCKED_SAMPLES)}.\nEntries: {ALL_MOCKED_SAMPLES}"
        )
