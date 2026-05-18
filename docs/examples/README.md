# Code Examples

This section provides practical code examples for common SDK use cases. All examples are available as runnable scripts in the [`samples/`](../../samples/) directory.

## Quick Reference

| Sample | Description |
|--------|-------------|
| [`benchmark_evaluation.py`](../../samples/core/benchmark_evaluation.py) | Run a model against a benchmark, wait for completion, retrieve results |
| [`quickstart.py`](../../samples/core/quickstart.py) | Minimal end-to-end trace evaluation |
| [`async_workflow.py`](../../samples/core/async_workflow.py) | Full async evaluation workflow with concurrent operations |
| [`async_results.py`](../../samples/core/async_results.py) | Fetch results for multiple evaluations concurrently |
| [`model_benchmark_management.py`](../../samples/core/model_benchmark_management.py) | Filter models by name/company/region, add/remove from project |
| [`evaluation_filtering.py`](../../samples/core/evaluation_filtering.py) | Sort and filter evaluations by status, accuracy, date |
| [`compare_evaluations.py`](../../samples/core/compare_evaluations.py) | Compare two models on a benchmark with outcome filtering |
| [`paginated_results.py`](../../samples/core/paginated_results.py) | Paginate through results or fetch all at once |
| [`custom_model.py`](../../samples/core/custom_model.py) | Register a custom model with an OpenAI-compatible API |
| [`custom_benchmark.py`](../../samples/core/custom_benchmark.py) | Create custom and smart benchmarks from data files |
| [`create_judge.py`](../../samples/core/create_judge.py) | Create, list, update, and delete judges |
| [`basic_trace.py`](../../samples/core/basic_trace.py) | Upload, list, get, and delete traces |
| [`trace_evaluation.py`](../../samples/core/trace_evaluation.py) | Run judges on traces, estimate cost, get results with steps |
| [`judge_optimization.py`](../../samples/core/judge_optimization.py) | Estimate, run, and apply judge optimizations |
| [`public_catalog.py`](../../samples/core/public_catalog.py) | Browse public models, benchmarks, evaluations, and prompts |
| [`integration_management.py`](../../samples/core/integration_management.py) | List, inspect, and test configured integrations |

## Guides

- [Creating Evaluations](creating-evaluations.md) -- Sync, async, and parallel evaluations
- [Retrieving Results](retrieving-results.md) -- Paginated, bulk, and concurrent result fetching
- [Models and Benchmarks](models-and-benchmarks.md) -- Filtering, custom models, custom/smart benchmarks, project management
- [Judges and Traces](judges-and-traces.md) -- Judge CRUD, trace uploads, trace evaluations, and optimizations
- [Public API](public-api.md) -- Public models, benchmarks, evaluations, and comparisons

For the complete samples catalog including industry solutions, OpenClaw agent evaluation, CI/CD integration, and more, see the [Samples Guide](../samples-guide.md).
