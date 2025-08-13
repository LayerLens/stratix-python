# Advanced Usage Patterns

This guide covers practical advanced techniques for using the Atlas Python SDK in production environments.

## Environment Variables Setup

The Atlas SDK reads your credentials from environment variables. You can set them up however you prefer:

```python
import os
from atlas import Atlas

# Option 1: Load from system environment variables
client = Atlas()  # Automatically uses LAYERLENS_ATLAS_API_KEY, etc.

# Option 2: Using python-dotenv (if you prefer .env files)
from dotenv import load_dotenv
load_dotenv()  # Loads from .env file
client = Atlas()
```

Required environment variables:
- `LAYERLENS_ATLAS_API_KEY` - Your Atlas API key
- `LAYERLENS_ATLAS_ORG_ID` - Your organization ID  
- `LAYERLENS_ATLAS_PROJECT_ID` - Your project ID

## Pagination Best Practices

### Understanding Pagination

The Atlas SDK automatically handles pagination for large result sets. When evaluation results exceed the default page size (100), you'll need to iterate through pages to access all data.

```python
from atlas import Atlas

def understand_pagination(evaluation_id: str):
    """Understand pagination metadata"""
    client = Atlas()
    
    # Get first page
    results_data = client.results.get(evaluation_id=evaluation_id)
    
    if results_data:
        pagination = results_data.pagination
        
        print(f" Pagination Overview:")
        print(f"   Total results: {pagination.total_count:,}")
        print(f"   Page size: {pagination.page_size}")
        print(f"   Total pages: {pagination.total_pages}")
        print(f"   Current page has: {len(results_data.results)} results")
        
        # Calculate some useful info
        is_paginated = pagination.total_pages > 1
        results_per_page = pagination.page_size
        last_page_size = pagination.total_count % pagination.page_size or pagination.page_size
        
        print(f"\n Analysis:")
        print(f"   Is paginated: {is_paginated}")
        print(f"   Results per page: {results_per_page}")
        print(f"   Last page size: {last_page_size}")
        
        if is_paginated:
            print(f"\n To access all {pagination.total_count:,} results:")
            print(f"   - Iterate through {pagination.total_pages} pages")
            print(f"   - Or use batch processing patterns")
        
        return pagination
    
    return None

# Usage
pagination_info = understand_pagination("eval_12345")
```

### Efficient Pagination Strategies

```python
def efficient_pagination_strategies():
    """Demonstrate different pagination approaches"""
    client = Atlas()
    evaluation_id = "eval_12345"
    
    # Strategy 1: Small pages for real-time processing
    print(" Strategy 1: Small pages for real-time feedback")
    page_size = 25
    page = 1
    
    while True:
        results_data = client.results.get(
            evaluation_id=evaluation_id,
            page=page,
            page_size=page_size
        )
        
        if not results_data or not results_data.results:
            break
            
        print(f"   Processing page {page}: {len(results_data.results)} results")
        
        # Process immediately
        for result in results_data.results:
            # Real-time processing logic
            pass
        
        if page >= results_data.pagination.total_pages:
            break
        page += 1
    
    print("\n Strategy 2: Large pages for batch processing")
    page_size = 200  # Larger pages
    page = 1
    
    while True:
        results_data = client.results.get(
            evaluation_id=evaluation_id,
            page=page,
            page_size=page_size
        )
        
        if not results_data or not results_data.results:
            break
            
        print(f"   Batch processing page {page}: {len(results_data.results)} results")
        
        # Batch process entire page
        process_batch(results_data.results)
        
        if page >= results_data.pagination.total_pages:
            break
        page += 1

def process_batch(results):
    """Process a batch of results efficiently"""
    # Batch processing logic here
    pass

# Usage
efficient_pagination_strategies()
```

### Memory Management with Pagination

```python
import gc
from atlas import Atlas

def memory_efficient_processing(evaluation_id: str):
    """Process large evaluations without memory issues"""
    client = Atlas()
    
    # Track memory usage
    processed_count = 0
    page = 1
    page_size = 100
    
    print(" Memory-efficient processing with pagination")
    
    while True:
        print(f" Processing page {page}...")
        
        # Get current page
        results_data = client.results.get(
            evaluation_id=evaluation_id,
            page=page,
            page_size=page_size
        )
        
        if not results_data or not results_data.results:
            break
        
        # Process current page results
        page_stats = process_page_results(results_data.results)
        processed_count += len(results_data.results)
        
        print(f"    Processed {len(results_data.results)} results")
        print(f"    Page accuracy: {page_stats['accuracy']:.1%}")
        
        # Clear references to help garbage collection
        del results_data
        del page_stats
        
        # Force garbage collection periodically
        if page % 10 == 0:
            gc.collect()
            print(f"   🧹 Garbage collection at page {page}")
        
        # Progress update
        print(f"    Total processed: {processed_count:,}")
        
        page += 1
    
    print(f" Completed! Processed {processed_count:,} results total")

def process_page_results(results):
    """Process a single page of results and return summary stats"""
    if not results:
        return {"accuracy": 0, "avg_score": 0}
    
    correct = sum(1 for r in results if r.score > 0.5)
    total_score = sum(r.score for r in results)
    
    return {
        "accuracy": correct / len(results),
        "avg_score": total_score / len(results),
        "count": len(results)
    }

# Usage
memory_efficient_processing("eval_12345")
```

## Batch Processing

### Running Multiple Evaluations

```python
import time
from atlas import Atlas
import atlas

def run_evaluation_batch(models, benchmarks):
    """Run evaluations for multiple model-benchmark combinations"""
    client = Atlas()
    
    results = {'successful': [], 'failed': []}
    
    for model in models:
        for benchmark in benchmarks:
            print(f"Creating evaluation: {model} on {benchmark}")
            
            try:
                evaluation = client.evaluations.create(
                    model=model,
                    benchmark=benchmark
                )
                
                if evaluation:
                    results['successful'].append({
                        'model': model,
                        'benchmark': benchmark, 
                        'evaluation_id': evaluation.id
                    })
                    print(f" Created: {evaluation.id}")
                else:
                    results['failed'].append({
                        'model': model,
                        'benchmark': benchmark,
                        'error': 'No evaluation returned'
                    })
                    
            except atlas.RateLimitError:
                print("Rate limited, waiting 60 seconds...")
                time.sleep(60)
                
            except atlas.APIError as e:
                print(f" Failed: {e}")
                results['failed'].append({
                    'model': model,
                    'benchmark': benchmark, 
                    'error': str(e)
                })
            
            time.sleep(2)
    
    return results

# Usage
models = ["gpt-4", "claude-3-opus"]
benchmarks = ["mmlu", "hellaswag"]

batch_results = run_evaluation_batch(models, benchmarks)
print(f" Successful: {len(batch_results['successful'])}")
print(f" Failed: {len(batch_results['failed'])}")
```

## Error Handling Patterns

### Robust Error Handling

```python
import time
from atlas import Atlas
import atlas

def create_evaluation_with_retries(model, benchmark, max_retries=3):
    """Create evaluation with automatic retries"""
    client = Atlas()
    
    for attempt in range(max_retries):
        try:
            evaluation = client.evaluations.create(
                model=model,
                benchmark=benchmark
            )
            
            if evaluation:
                print(f" Success on attempt {attempt + 1}")
                return evaluation
            
        except atlas.RateLimitError as e:
            print(f"Rate limited on attempt {attempt + 1}")
            if attempt < max_retries - 1:
                # Check if server provided retry-after header
                retry_after = getattr(e.response, 'headers', {}).get('retry-after', 60)
                wait_time = int(retry_after)
                print(f"Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise
                
        except atlas.NotFoundError:
            print(f" Model '{model}' or benchmark '{benchmark}' not found")
            return None
            
        except atlas.AuthenticationError:
            print(" Authentication failed - check your API key")
            raise
            
        except atlas.APIError as e:
            print(f" API error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
    
    return None

# Usage
evaluation = create_evaluation_with_retries("gpt-4", "mmlu")
```

## Result Processing

### Processing Large Result Sets

```python
from atlas import Atlas
import json
from typing import Dict, List

def analyze_evaluation_results(evaluation_id: str) -> Dict:
    """Analyze results from an evaluation"""
    client = Atlas()
    
    try:
        results = client.results.get(evaluation_id=evaluation_id)
        
        if not results:
            return {"error": "No results found"}
        
        # Basic analysis
        analysis = {
            "total_results": len(results),
            "subsets": {},
            "overall_accuracy": 0,
            "avg_duration": 0
        }
        
        total_score = 0
        total_duration = 0
        
        for result in results:
            # Track by subset
            if result.subset not in analysis["subsets"]:
                analysis["subsets"][result.subset] = {
                    "count": 0,
                    "total_score": 0,
                    "accuracy": 0
                }
            
            analysis["subsets"][result.subset]["count"] += 1
            analysis["subsets"][result.subset]["total_score"] += result.score
            
            total_score += result.score
            total_duration += result.duration.total_seconds()
        
        # Calculate averages
        analysis["overall_accuracy"] = total_score / len(results)
        analysis["avg_duration"] = total_duration / len(results)
        
        # Calculate subset accuracies
        for subset_data in analysis["subsets"].values():
            subset_data["accuracy"] = subset_data["total_score"] / subset_data["count"]
        
        return analysis
        
    except atlas.APIError as e:
        return {"error": str(e)}

# Usage
analysis = analyze_evaluation_results("eval_123")
if "error" not in analysis:
    print(f" Analysis Results:")
    print(f"   Total results: {analysis['total_results']}")
    print(f"   Overall accuracy: {analysis['overall_accuracy']:.2%}")
    print(f"   Average duration: {analysis['avg_duration']:.2f}s")
    
    print(f"   By subset:")
    for subset, data in analysis['subsets'].items():
        print(f"     {subset}: {data['accuracy']:.2%} ({data['count']} results)")
```

## Production Timeouts

### Different Timeout Strategies

```python
from atlas import Atlas

# Different timeout configurations for different use cases

# Development: Fail fast
dev_client = Atlas(timeout=30.0)  # 30 seconds

# Production: More patient
prod_client = Atlas(timeout=600.0)  # 10 minutes

# Long-running batch jobs: Very patient  
batch_client = Atlas(timeout=1800.0)  # 30 minutes

def adaptive_timeout_client(operation_type="default"):
    """Get client with timeout appropriate for operation"""
    timeouts = {
        "quick": 30.0,      # For testing connectivity
        "default": 300.0,   # For normal operations
        "batch": 1800.0,    # For batch processing
        "patient": 3600.0   # For very long evaluations
    }
    
    timeout = timeouts.get(operation_type, timeouts["default"])
    return Atlas(timeout=timeout)

# Usage
quick_client = adaptive_timeout_client("quick")
batch_client = adaptive_timeout_client("batch")
```

## Logging and Monitoring

### Simple Logging Setup

```python
import logging
import time
from atlas import Atlas
import atlas

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('atlas-client')

def create_evaluation_with_logging(model, benchmark):
    """Create evaluation with comprehensive logging"""
    client = Atlas()
    
    logger.info(f"Creating evaluation: {model} on {benchmark}")
    start_time = time.time()
    
    try:
        evaluation = client.evaluations.create(
            model=model,
            benchmark=benchmark
        )
        
        duration = time.time() - start_time
        
        if evaluation:
            logger.info(
                f"Evaluation created successfully: {evaluation.id} "
                f"(duration: {duration:.2f}s)"
            )
            return evaluation
        else:
            logger.warning(
                f"No evaluation returned for {model}+{benchmark} "
                f"(duration: {duration:.2f}s)"
            )
            return None
            
    except atlas.APIError as e:
        duration = time.time() - start_time
        logger.error(
            f"Failed to create evaluation {model}+{benchmark}: {e} "
            f"(duration: {duration:.2f}s)"
        )
        raise

# Usage
evaluation = create_evaluation_with_logging("gpt-4", "mmlu")
```

## Health Checks

### Simple Health Check

```python
from atlas import Atlas
import atlas

def check_atlas_health():
    """Simple health check for Atlas service"""
    try:
        client = Atlas(timeout=10.0)  # Short timeout for health check
        
        # Try to create a test evaluation (will fail but tests connectivity)
        try:
            client.evaluations.create(
                model="__health_check__", 
                benchmark="__health_check__"
            )
        except atlas.NotFoundError:
            # Expected - health check resources don't exist
            return {"status": "healthy", "message": "API is reachable"}
        except atlas.BadRequestError:
            # Also expected - invalid parameters
            return {"status": "healthy", "message": "API is reachable"}
            
    except atlas.AuthenticationError:
        return {
            "status": "unhealthy", 
            "error": "Authentication failed - check API key"
        }
    except atlas.APIConnectionError:
        return {
            "status": "unhealthy",
            "error": "Cannot connect to Atlas API"
        }
    except atlas.APITimeoutError:
        return {
            "status": "unhealthy", 
            "error": "Health check timed out"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": f"Unexpected error: {e}"
        }

# Usage
health = check_atlas_health()
if health["status"] == "healthy":
    print(" Atlas service is healthy")
else:
    print(f" Atlas service is unhealthy: {health['error']}")
```

## Integration Patterns

### Using with Flask/FastAPI

```python
from flask import Flask, jsonify, request
from atlas import Atlas
import atlas

app = Flask(__name__)

# Initialize Atlas client once
atlas_client = Atlas()

@app.route('/health')
def health_check():
    """Health check endpoint"""
    health = check_atlas_health()  # From example above
    status_code = 200 if health["status"] == "healthy" else 503
    return jsonify(health), status_code

@app.route('/evaluations', methods=['POST'])
def create_evaluation():
    """Create evaluation endpoint"""
    try:
        data = request.get_json()
        model = data.get('model')
        benchmark = data.get('benchmark')
        
        if not model or not benchmark:
            return jsonify({
                "error": "Missing required fields: model, benchmark"
            }), 400
        
        evaluation = atlas_client.evaluations.create(
            model=model,
            benchmark=benchmark
        )
        
        if evaluation:
            return jsonify({
                "success": True,
                "evaluation_id": evaluation.id,
                "status": evaluation.status
            })
        else:
            return jsonify({
                "success": False,
                "error": "Failed to create evaluation"
            }), 500
            
    except atlas.NotFoundError:
        return jsonify({
            "success": False,
            "error": "Model or benchmark not found"
        }), 404
        
    except atlas.APIError as e:
        return jsonify({
            "success": False, 
            "error": str(e)
        }), 500

@app.route('/evaluations/<evaluation_id>/results')
def get_results(evaluation_id):
    """Get evaluation results endpoint"""
    try:
        results = atlas_client.results.get(evaluation_id=evaluation_id)
        
        if results:
            return jsonify({
                "success": True,
                "result_count": len(results),
                "results": [
                    {
                        "subset": r.subset,
                        "score": r.score,
                        "duration_seconds": r.duration.total_seconds()
                    }
                    for r in results
                ]
            })
        else:
            return jsonify({
                "success": False,
                "error": "No results found"
            }), 404
            
    except atlas.APIError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
```
