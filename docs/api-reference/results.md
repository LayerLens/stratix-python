# Results

The `results` resource allows you to retrieve detailed results from completed or partially completed evaluations. This provides granular insight into how your model performed on individual test cases.

## Overview

Results contain detailed information about each test case in an evaluation.

> Both the `Stratix` (synchronous) and `AsyncStratix` (asynchronous) clients support the following methods.

## Methods

### `get_all(evaluation, timeout=None)`

Retrieves all results for a specific evaluation by automatically iterating through all pages. This is a convenience method that handles pagination through all results internally.

#### Parameters

| Parameter    | Type                             | Required | Description                              |
| ------------ | -------------------------------- | -------- | ---------------------------------------- |
| `evaluation` | `Evaluation`                     | Yes      | The evaluation object to get results for |
| `timeout`    | `float \| httpx.Timeout \| None` | No       | Override request timeout                 |

#### Returns

Returns a `List[Result]` containing all result objects across all pages. Returns an empty list if no results are found.

#### Example

```python
from layerlens import Stratix

client = Stratix()

# Get evaluation first
evaluation = client.evaluations.get_by_id("eval_12345")
if not evaluation:
    print("Evaluation not found")
else:
    # Get all results at once
    all_results = client.results.get_all(evaluation=evaluation)

    print(f"Retrieved {len(all_results)} total results")
```

#### Async Usage

```python
from layerlens import AsyncStratix
import asyncio

async def get_all_results():
    client = AsyncStratix()

    # Get evaluation first
    evaluation = await client.evaluations.get_by_id("eval_12345")
    if not evaluation:
        print("Evaluation not found")
        return

    # Get all results asynchronously
    all_results = await client.results.get_all(evaluation=evaluation)
    print(f"Retrieved {len(all_results)} total results asynchronously")

    return all_results

# Run the async fetching of results
asyncio.run(get_all_results())
```

### `get_all_by_id(evaluation_id, timeout=None)`

Retrieves all results for a specific evaluation by evaluation ID, automatically iterating through all pages. This is a convenience method that handles pagination internally.

#### Parameters

| Parameter       | Type                             | Required | Description                                  |
| --------------- | -------------------------------- | -------- | -------------------------------------------- |
| `evaluation_id` | `str`                            | Yes      | The evaluation identifier to get results for |
| `timeout`       | `float \| httpx.Timeout \| None` | No       | Override request timeout                     |

#### Returns

Returns a `List[Result]` containing all result objects across all pages. Returns an empty list if no results are found or the evaluation doesn't exist.

#### Example

```python
from layerlens import Stratix

client = Stratix()

# Get all results directly by evaluation ID
all_results = client.results.get_all_by_id(evaluation_id="eval_12345")
```

#### Async Usage

```python
from layerlens import AsyncStratix
import asyncio

async def get_all_results():
    client = AsyncStratix()

    # Get all results asynchronously
    all_results = await client.results.get_all_by_id(evaluation_id="eval_12345")

    if all_results:
        print(f"Retrieved {len(all_results)} total results")

    else:
        print("No results found")

# Run the async function
asyncio.run(get_all_results())
```

### `get(evaluation_id, page=None, page_size=None, timeout=None)`

Retrieves detailed results for a specific evaluation with optional pagination support.

#### Parameters

| Parameter       | Type                             | Required | Description                                                                            |
| --------------- | -------------------------------- | -------- | -------------------------------------------------------------------------------------- |
| `evaluation_id` | `str`                            | Yes      | The evaluation identifier to get results for                                           |
| `page`          | `int \| None`                    | No       | Page number for pagination. If not provided, returns first page is returned by default |
| `page_size`     | `int \| None`                    | No       | Number of results per page (default: 100). Maximum allowed page_size is 500            |
| `timeout`       | `float \| httpx.Timeout \| None` | No       | Override request timeout                                                               |

#### Returns

Returns a `ResultsResponse` object containing results, evaluation metadata, and pagination information if successful, `None` if no results are found or the evaluation doesn't exist.

The `ResultsResponse` object includes:

- `results`: List of `Result` objects for the current page
- `evaluation_id`: The evaluation ID
- `pagination`: Pagination metadata (total_count, page_size, total_pages)

#### Examples

##### Basic Usage (All Results)

```python
from layerlens import Stratix

client = Stratix()

# Get all results for a specific evaluation
results_data = client.results.get(evaluation_id="eval_12345")

if results_data:
    print(f"Evaluation ID: {results_data.evaluation_id}")
    print(f"Retrieved {len(results_data.results)} results")
    print(f"Total available: {results_data.pagination.total_count}")
    print(f"Page size: {results_data.pagination.page_size}")
    print(f"Total pages: {results_data.pagination.total_pages}")

    # Access individual results
    for i, result in enumerate(results_data.results[:3]):  # Show first 3
        print(f"\nResult {i+1}:")
        print(f"  Subset: {result.subset}")
        print(f"  Score: {result.score}")
        print(f"  Duration: {result.duration}")
else:
    print("No results found or evaluation doesn't exist")
```

##### Paginated Access

```python
# Get specific page with custom page size
results_data = client.results.get(
    evaluation_id="eval_12345",
    page=2,
    page_size=50
)

if results_data:
    print(f"Page 2 of {results_data.pagination.total_pages}")
    print(f"Showing {len(results_data.results)} of {results_data.pagination.total_count} total results")

    # Process current page
    for result in results_data.results:
        # Process each result
        pass
```

##### Iterating Through All Pages

```python
# Process all results by iterating through pages
evaluation_id = "eval_12345"
page = 1
page_size = 100

while True:
    results_data = client.results.get(
        evaluation_id=evaluation_id,
        page=page,
        page_size=page_size
    )

    if not results_data or not results_data.results:
        break

    print(f"Processing page {page}/{results_data.pagination.total_pages}")

    # Process current page results
    for result in results_data.results:
        # Your processing logic here
        pass

    # Move to next page
    if page >= results_data.pagination.total_pages:
        break
    page += 1

print("Finished processing all results")
```

## Pagination Information

The `pagination` object in the response provides detailed pagination metadata:

```python
results_data = client.results.get(evaluation_id="eval_12345", page=1, page_size=50)

if results_data:
    pagination = results_data.pagination

    print(f"Current page info:")
    print(f"  Total results available: {pagination.total_count}")
    print(f"  Results per page: {pagination.page_size}")
    print(f"  Total pages: {pagination.total_pages}")
    print(f"  Results on current page: {len(results_data.results)}")

    # Calculate current page number (if needed)
    # Page number isn't stored in pagination object, so track it yourself
    current_page = 1  # You would track this in your code
    print(f"  Current page: {current_page}")

    # Check if there are more pages
    has_more_pages = current_page < pagination.total_pages
    print(f"  Has more pages: {has_more_pages}")
```

### Pagination Properties

| Property      | Type  | Description                                          |
| ------------- | ----- | ---------------------------------------------------- |
| `total_count` | `int` | Total number of results available across all pages   |
| `page_size`   | `int` | Number of results per page (as requested or default) |
| `total_pages` | `int` | Total number of pages available                      |

## Result Object

Each `Result` object contains the following properties:

### Core Properties

| Property   | Type               | Description                                                |
| ---------- | ------------------ | ---------------------------------------------------------- |
| `subset`   | `str`              | The benchmark subset or category this test case belongs to |
| `prompt`   | `str`              | The input prompt given to the model                        |
| `result`   | `str`              | The model's response/output                                |
| `truth`    | `str`              | The expected or correct answer                             |
| `score`    | `float`            | Individual score for this test case (typically 0.0 to 1.0) |
| `duration` | `timedelta`        | Time taken for the model to respond                        |
| `metrics`  | `Dict[str, float]` | Additional metrics specific to this test case              |

### Understanding Properties

- **`subset`**: Groups related test cases (e.g., "elementary_mathematics", "world_history")
- **`prompt`**: The exact input sent to the model
- **`result`**: The model's actual response
- **`truth`**: The ground truth or expected answer for comparison
- **`score`**: Individual test case score, usually binary (0.0 or 1.0) for correctness
- **`duration`**: Response latency as a Python `timedelta` object
- **`metrics`**: Additional scoring metrics that may be benchmark-specific

## Working with Large Result Sets

### Fetching results async

Results can contain thousands of individual test cases. Consider using the async client to load results asynchronously:

```python
import asyncio
from layerlens import AsyncStratix

async def fetch_results_async():
    async_client = AsyncStratix()

    # Get evaluation first
    evaluation = await async_client.evaluations.get_by_id("eval_12345")
    if not evaluation:
        print("Evaluation not found")
        return None

    # async results fetching all pages of results
    results = await async_client.results.get_all(evaluation=evaluation)
    if results:
        return results
    else:
        return None

# Run the async function
asyncio.run(fetch_results_async())
```

## Next Steps

- Explore [code examples](../examples/retrieving-results.md) for common analysis patterns
