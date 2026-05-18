# Core Samples

These samples cover the foundational operations of the LayerLens Python SDK. They address the
core problem every AI team faces: how to systematically trace, evaluate, and improve LLM
interactions across your application. Start here to build familiarity with the SDK before
exploring domain-specific or advanced patterns.

## Prerequisites

```bash
pip install layerlens --index-url https://sdk.layerlens.ai/package
export LAYERLENS_STRATIX_API_KEY=your-api-key
```

## Quick Start

Run `quickstart.py` for a minimal end-to-end walkthrough that creates a trace, defines a judge,
and runs an evaluation in under 30 lines of code:

```bash
python quickstart.py
```

Expected output: a trace ID, judge ID, and evaluation summary printed to the console.

## Samples

| File | Scenario | Description |
|------|----------|-------------|
| `quickstart.py` | First-time setup | Minimal end-to-end example covering trace creation, judge setup, and evaluation in a single script. |
| `basic_trace.py` | Observability engineers instrumenting LLM calls | Create, retrieve, and manage trace records for individual LLM interactions. |
| `run_evaluation.py` | QA leads running scheduled quality checks | Execute a full evaluation lifecycle: configure judges, submit traces, and collect scored results. |
| `create_judge.py` | Platform teams defining quality standards | CRUD operations for judge resources -- create, read, update, and delete evaluation judges. |
| `trace_evaluation.py` | Developers validating prompt changes | Evaluate traced LLM interactions against one or more judges to detect regressions. |
| `benchmark_evaluation.py` | ML teams comparing model performance | Run a model against a benchmark, wait for completion, retrieve and analyze scored results. |
| `judge_optimization.py` | ML engineers tuning evaluation criteria | Programmatically refine judge configurations to improve scoring precision and recall. |
| `compare_evaluations.py` | Teams comparing prompt or model variants | Compare scored results across multiple evaluation runs to identify the best-performing configuration. |
| `async_workflow.py` | High-throughput pipelines | Asynchronous SDK usage for non-blocking trace uploads and evaluation submissions. |
| `model_benchmark_management.py` | Platform teams cataloging models | Register models and benchmarks, then query benchmark results for reporting. |
| `integration_management.py` | Teams managing LayerLens integrations | List, inspect, and test configured integrations using the `client.integrations` API. |
| `custom_model.py` | Teams using private or fine-tuned models | Register and manage custom model definitions within the LayerLens model catalog. |
| `custom_benchmark.py` | Teams defining proprietary test suites | Create and run custom benchmarks tailored to your application's specific quality criteria. |
| `trace_investigation.py` | On-call engineers debugging production issues | Investigate traces for errors, high latency, and anomalous token usage patterns. |
| `evaluation_pipeline.py` | CI/CD and batch workflows | Orchestrate a multi-step evaluation pipeline combining judges, traces, and result aggregation. |
| `evaluation_filtering.py` | Analysts querying evaluation history | Filter and paginate evaluation results by status, date range, judge, or score threshold. |
| `paginated_results.py` | Large-scale data retrieval | Demonstrate cursor-based pagination for traces, evaluations, and judge listings. |
| `public_catalog.py` | Teams exploring built-in judges | Browse and query the public judge and model catalog provided by LayerLens. |
| `judge_creation_and_test.py` | Compliance teams building PII detectors | Create a custom PII judge, run it against sample traces, and verify detection accuracy. |
| `async_results.py` | Polling for long-running evaluations | Submit evaluations asynchronously and poll for completion with timeout handling. |

## Expected Behavior

Each sample prints its results to stdout. Trace and evaluation IDs are generated server-side
and will differ between runs. Samples that create resources (judges, evaluations) will persist
those resources in your LayerLens workspace unless explicitly deleted.
