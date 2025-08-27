# Models & Benchmarks

This page provides reference information about accessing the available models and benchmarks on your organization using the layerlens python sdk.

## Overview

Running evaluations on the atlas platform require a model and benchmark to be selected. The availability of models and benchmarks depends on your organizations configuration.

> Before running the below examples ensure the model and benchmark being run are present on your organiztion.

### Finding Available Models and Benchmarks

#### 1. Using the Atlas Dashboard

The most reliable way to find available models and benchmarks:

1. Log into your Atlas dashboard.
2. Navigate to the evaluation creation page.
3. View dropdown lists of available models and benchmarks.

#### 2. Using the python sdk

```python
from layerlens import Atlas

# Construct sync client (API key from env or inline)
client = Atlas()

# --- Models
models = client.models.get()

# --- Benchmarks
benchmarks = client.benchmarks.get()
```

## Models

### `get(type=None, name=None, companies=None, regions=None, licenses=None, timeout=None)`

Retrieves a list of available models with optional filtering parameters. Both the `Atlas` and `AsyncAtlas` clients have this method.

#### Parameters

| Parameter   | Type                                  | Required | Description                                                            |
| ----------- | ------------------------------------- | -------- | ---------------------------------------------------------------------- |
| `type`      | `Literal["custom", "public"] \| None` | No       | Filter by model type. If `None`, returns both custom and public models |
| `name`      | `str \| None`                         | No       | Filter models by name (partial match search)                           |
| `companies` | `List[str] \| None`                   | No       | Filter by model companies/providers                                    |
| `regions`   | `List[str] \| None`                   | No       | Filter by supported regions                                            |
| `licenses`  | `List[str] \| None`                   | No       | Filter by license types                                                |
| `timeout`   | `float \| httpx.Timeout \| None`      | No       | Override request timeout                                               |

#### Returns

Returns an `Optional[List[Model]]` - a list of `Model` objects that match the filter criteria. Returns an empty list `[]` if no models match the criteria, or `None` if there's an error.

#### Model Object Properties

Each `Model` object in the returned list contains:

| Property      | Type  | Description                                             |
| ------------- | ----- | ------------------------------------------------------- |
| `id`          | `str` | Unique model identifier for use in evaluations          |
| `name`        | `str` | Human-readable model name                               |
| `key`         | `str` | Unique model key/identifier that is similar to the name |
| `description` | `str` | Text description of the model                           |

## Benchmarks

### `get(type=None, name=None, timeout=None)`

Retrieves a list of available benchmarks with optional filtering parameters. Both the `Atlas` and `AsyncAtlas` clients have this method.

#### Parameters

| Parameter | Type                                  | Required | Description                                                                    |
| --------- | ------------------------------------- | -------- | ------------------------------------------------------------------------------ |
| `type`    | `Literal["custom", "public"] \| None` | No       | Filter by benchmark type. If `None`, returns both custom and public benchmarks |
| `name`    | `str \| None`                         | No       | Filter benchmarks by name (partial match search)                               |
| `timeout` | `float \| httpx.Timeout \| None`      | No       | Override request timeout                                                       |

#### Returns

Returns a `List[Benchmark]` containing available benchmarks that match the filter criteria. Returns `None` if no benchmarks are found or if there's an error.

Returns `Optional[List[Benchmark]]` - a list of `Benchmark` objects that match the filter criteria. Returns an empty list `[]` if no benchmarks match the criteria, or `None` if there's an error.

#### Benchmark Object Properties

Each `Benchmark` object in the returned list contains:

| Property | Type  | Description                                                 |
| -------- | ----- | ----------------------------------------------------------- |
| `id`     | `str` | Unique benchmark identifier for use in evaluations          |
| `key`    | `str` | Unique benchmark key/identifier that is similar to the name |
| `name`   | `str` | Human-readable benchmark name                               |
