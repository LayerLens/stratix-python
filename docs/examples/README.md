# Examples

This section provides practical code examples for common SDK use cases. All examples are available as runnable scripts in the [`examples/`](../../examples/) directory.

## Quick Reference

| Example | Description |
| ------- | ----------- |
| [`client_simple.py`](../../examples/client_simple.py) | Minimal sync client usage |
| [`client.py`](../../examples/client.py) | Full sync evaluation workflow |
| [`async_client_simple.py`](../../examples/async_client_simple.py) | Minimal async client usage |
| [`async_client.py`](../../examples/async_client.py) | Full async evaluation workflow |
| [`async_run_evaluations.py`](../../examples/async_run_evaluations.py) | Run multiple evaluations in parallel |
| [`get_models.py`](../../examples/get_models.py) | Filter models by name, company, region, type |
| [`get_benchmarks.py`](../../examples/get_benchmarks.py) | Filter benchmarks by name and type |
| [`get_evaluation.py`](../../examples/get_evaluation.py) | Fetch an evaluation by ID |
| [`evaluation_sorting.py`](../../examples/evaluation_sorting.py) | Sort and filter evaluations |
| [`compare_evaluations.py`](../../examples/compare_evaluations.py) | Compare two models on a benchmark |
| [`paginated_results.py`](../../examples/paginated_results.py) | Paginate through evaluation results |
| [`all_results_no_pagination.py`](../../examples/all_results_no_pagination.py) | Fetch all results at once |
| [`fetch_results_async.py`](../../examples/fetch_results_async.py) | Fetch results for multiple evaluations concurrently |
| [`create_custom_model.py`](../../examples/create_custom_model.py) | Create a custom model with an OpenAI-compatible API |
| [`create_custom_benchmark.py`](../../examples/create_custom_benchmark.py) | Create a custom benchmark from a JSONL file |
| [`create_smart_benchmark.py`](../../examples/create_smart_benchmark.py) | Create an AI-generated benchmark from documents |
| [`manage_project_models_benchmarks.py`](../../examples/manage_project_models_benchmarks.py) | Add/remove models and benchmarks from a project |
| [`judges.py`](../../examples/judges.py) | Create, list, update, and delete judges |
| [`traces.py`](../../examples/traces.py) | Upload, list, get, and delete traces |
| [`trace_evaluations.py`](../../examples/trace_evaluations.py) | Run judges on traces, estimate cost, get results |
| [`async_judges_and_traces.py`](../../examples/async_judges_and_traces.py) | Async judge and trace evaluation workflow |
| [`judge_optimizations.py`](../../examples/judge_optimizations.py) | Estimate, run, and apply judge optimizations |
| [`public_models.py`](../../examples/public_models.py) | Browse, search, and filter public models |
| [`public_benchmarks.py`](../../examples/public_benchmarks.py) | Browse public benchmarks and download prompts |
| [`public_evaluations.py`](../../examples/public_evaluations.py) | Get public evaluation details and results |

## Guides

- [Creating Evaluations](creating-evaluations.md) - Sync, async, and parallel evaluations
- [Retrieving Results](retrieving-results.md) - Paginated, bulk, and concurrent result fetching
- [Models and Benchmarks](models-and-benchmarks.md) - Filtering, custom models, custom/smart benchmarks, project management
- [Judges and Traces](judges-and-traces.md) - Judge CRUD, trace uploads, trace evaluations, and optimizations
- [Public API](public-api.md) - Public models, benchmarks, evaluations, and comparisons
