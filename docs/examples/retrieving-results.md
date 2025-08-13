# Retrieving Results

This guide provides practical examples for retrieving and analyzing evaluation results with the Atlas Python SDK.

## Basic Result Retrieval

### Simple Result Fetching

```python
from atlas import Atlas

# Initialize client
client = Atlas()

# Get results for a specific evaluation
evaluation_id = "eval_12345"  # Replace with your evaluation ID
results_data = client.results.get(evaluation_id=evaluation_id)

if results_data:
    print(f"Evaluation: {results_data.evaluation_id}")
    print(f"Retrieved {len(results_data.results)} results (page 1)")
    print(f"Total available: {results_data.pagination.total_count}")
    print(f"Total pages: {results_data.pagination.total_pages}")
    
    # Show first few results
    for i, result in enumerate(results_data.results[:3]):
        print(f"\nResult {i+1}:")
        print(f"  Subset: {result.subset}")
        print(f"  Prompt: {result.prompt[:100]}...")
        print(f"  Model Response: {result.result[:100]}...")
        print(f"  Expected: {result.truth}")
        print(f"  Score: {result.score}")
        print(f"  Duration: {result.duration}")
else:
    print("No results found")
```

### Paginated Result Retrieval

```python
from atlas import Atlas

# Initialize client
client = Atlas()

def get_paginated_results(evaluation_id: str, page_size: int = 50):
    """Get results with pagination control"""
    
    # Get specific page
    results_data = client.results.get(
        evaluation_id=evaluation_id,
        page=2,  # Get second page
        page_size=page_size
    )
    
    if results_data:
        pagination = results_data.pagination
        print(f"Pagination Info:")
        print(f"   Total results: {pagination.total_count}")
        print(f"   Page size: {pagination.page_size}")
        print(f"   Total pages: {pagination.total_pages}")
        print(f"   Current page results: {len(results_data.results)}")
        
        return results_data
    else:
        print("No results found")
        return None

# Usage
paginated_results = get_paginated_results("eval_12345", page_size=25)
```

### Complete Evaluation Workflow

```python
from atlas import Atlas
import time

def complete_evaluation_workflow(model: str, benchmark: str):
    """Complete workflow: create evaluation and retrieve results"""
    client = Atlas()
    
    # Step 1: Create evaluation
    print(f"Creating evaluation: {model} + {benchmark}")
    evaluation = client.evaluations.create(model=model, benchmark=benchmark)
    
    if not evaluation:
        print("Failed to create evaluation")
        return None
    
    print(f"Evaluation created: {evaluation.id}")
    print(f"   Status: {evaluation.status}")
    
    # Step 2: Wait for completion (simplified polling)
    # In production, use webhooks instead of polling
    print("Waiting for evaluation to complete...")
    
    # Note: This is a simplified example. In practice, you'd:
    # 1. Use webhooks for real-time updates
    # 2. Store evaluation ID and check periodically
    # 3. Handle various status states properly
    
    if evaluation.status == "completed":
        print("Evaluation completed!")
        
        # Step 3: Retrieve results
        results_data = client.results.get(evaluation_id=evaluation.id)
        
        if results_data:
            results = results_data.results
            print(f"Retrieved {len(results)} results from page 1")
            print(f"Total results available: {results_data.pagination.total_count}")
            
            # Basic analysis for current page
            correct_answers = sum(1 for r in results if r.score > 0.5)
            accuracy = correct_answers / len(results)
            avg_duration = sum(r.duration for r in results) / len(results)
            
            print(f"Quick Analysis (Page 1):")
            print(f"   Accuracy: {accuracy:.1%} ({correct_answers}/{len(results)})")
            print(f"   Average Duration: {avg_duration}")
            
            # Note about pagination
            if results_data.pagination.total_pages > 1:
                print(f"Note: This evaluation has {results_data.pagination.total_pages} pages total")
                print(f"    Use pagination to process all {results_data.pagination.total_count} results")
            
            return results_data
        else:
            print("No results available")
    else:
        print(f"Evaluation status: {evaluation.status}")
        print("   Check back later for results")
    
    return None

# Usage
results = complete_evaluation_workflow("gpt-4", "mmlu")
```

## Result Analysis Patterns

### Performance Analysis

```python
from atlas import Atlas
from collections import defaultdict, Counter
import statistics
from datetime import timedelta

def analyze_evaluation_performance(evaluation_id: str, use_all_pages: bool = True):
    """Comprehensive performance analysis of evaluation results"""
    client = Atlas()
    
    if use_all_pages:
        # Get all results across all pages for complete analysis
        all_results = []
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
                
            all_results.extend(results_data.results)
            
            # Use pagination info from the first page
            if page == 1:
                total_count = results_data.pagination.total_count
                total_pages = results_data.pagination.total_pages
                print(f"Loading {total_count} results from {total_pages} pages...")
            
            print(f"   Loaded page {page}/{total_pages}")
            
            if page >= results_data.pagination.total_pages:
                break
                
            page += 1
        
        results = all_results
        
        if not results:
            print(f"No results found for evaluation {evaluation_id}")
            return None
            
    else:
        # Analyze just the first page
        results_data = client.results.get(evaluation_id=evaluation_id, page=1, page_size=100)
        if not results_data:
            print(f"No results found for evaluation {evaluation_id}")
            return None
            
        results = results_data.results
        print(f"Analyzing first page only ({len(results)} of {results_data.pagination.total_count} total results)")
    
    print(f"Performance Analysis for {evaluation_id}")
    print(f"{'='*60}")
    
    # Overall statistics
    total_cases = len(results)
    correct_answers = sum(1 for r in results if r.score > 0.5)
    total_score = sum(r.score for r in results)
    
    accuracy = correct_answers / total_cases
    avg_score = total_score / total_cases
    
    print(f"\n🎯 Overall Performance:")
    print(f"   Total test cases: {total_cases:,}")
    print(f"   Correct answers: {correct_answers:,}")
    print(f"   Accuracy: {accuracy:.1%}")
    print(f"   Average score: {avg_score:.3f}")
    
    # Timing analysis
    durations = [r.duration for r in results]
    avg_duration = sum(durations, timedelta()) / len(durations)
    min_duration = min(durations)
    max_duration = max(durations)
    median_duration = statistics.median(durations)
    
    print(f"\n⏱️  Timing Analysis:")
    print(f"   Average duration: {avg_duration}")
    print(f"   Median duration: {median_duration}")
    print(f"   Min duration: {min_duration}")
    print(f"   Max duration: {max_duration}")
    
    # Score distribution
    score_ranges = {
        "Perfect (1.0)": 0,
        "High (0.8-0.99)": 0,
        "Medium (0.5-0.79)": 0,
        "Low (0.1-0.49)": 0,
        "Zero (0.0)": 0
    }
    
    for result in results:
        score = result.score
        if score == 1.0:
            score_ranges["Perfect (1.0)"] += 1
        elif 0.8 <= score < 1.0:
            score_ranges["High (0.8-0.99)"] += 1
        elif 0.5 <= score < 0.8:
            score_ranges["Medium (0.5-0.79)"] += 1
        elif 0.1 <= score < 0.5:
            score_ranges["Low (0.1-0.49)"] += 1
        else:
            score_ranges["Zero (0.0)"] += 1
    
    print(f"\nScore Distribution:")
    for range_name, count in score_ranges.items():
        percentage = count / total_cases * 100
        print(f"   {range_name}: {count:,} ({percentage:.1f}%)")
    
    # Subset analysis
    subset_stats = defaultdict(lambda: {"scores": [], "durations": []})
    
    for result in results:
        subset_stats[result.subset]["scores"].append(result.score)
        subset_stats[result.subset]["durations"].append(result.duration)
    
    print(f"\nPerformance by Subset:")
    print(f"{'Subset':<25} {'Cases':<8} {'Accuracy':<10} {'Avg Score':<10} {'Avg Duration':<12}")
    print("-" * 75)
    
    for subset, data in sorted(subset_stats.items()):
        case_count = len(data["scores"])
        subset_accuracy = sum(1 for s in data["scores"] if s > 0.5) / case_count
        subset_avg_score = sum(data["scores"]) / case_count
        subset_avg_duration = sum(data["durations"], timedelta()) / case_count
        
        print(f"{subset:<25} {case_count:<8} {subset_accuracy:<10.1%} {subset_avg_score:<10.3f} {str(subset_avg_duration):<12}")
    
    return {
        "total_cases": total_cases,
        "accuracy": accuracy,
        "avg_score": avg_score,
        "avg_duration": avg_duration,
        "score_distribution": score_ranges,
        "subset_stats": dict(subset_stats)
    }

# Usage - analyze all results across all pages
analysis = analyze_evaluation_performance("eval_12345", use_all_pages=True)

# Usage - analyze only first page (faster for quick checks)
quick_analysis = analyze_evaluation_performance("eval_12345", use_all_pages=False)
```

## Pagination Patterns

### Pattern 1: Processing All Results Across Pages

```python
from atlas import Atlas

def process_all_results(evaluation_id: str):
    """Process all results by iterating through all pages"""
    client = Atlas()
    
    # Aggregate statistics across all pages
    total_results = 0
    total_score = 0
    total_correct = 0
    all_subsets = set()
    
    page = 1
    page_size = 100
    
    print("Processing all pages...")
    
    while True:
        print(f"Fetching page {page}...")
        
        results_data = client.results.get(
            evaluation_id=evaluation_id,
            page=page,
            page_size=page_size
        )
        
        if not results_data or not results_data.results:
            break
        
        # Show progress on first page
        if page == 1:
            print(f"Total: {results_data.pagination.total_count} results across {results_data.pagination.total_pages} pages")
        
        # Process current page
        current_results = results_data.results
        page_score = sum(r.score for r in current_results)
        page_correct = sum(1 for r in current_results if r.score > 0.5)
        page_subsets = set(r.subset for r in current_results)
        
        # Aggregate
        total_results += len(current_results)
        total_score += page_score
        total_correct += page_correct
        all_subsets.update(page_subsets)
        
        print(f"   Page {page}: {len(current_results)} results, {page_correct} correct, {len(page_subsets)} subsets")
        
        # Check if we're done
        if page >= results_data.pagination.total_pages:
            break
            
        page += 1
    
    # Final summary
    if total_results > 0:
        overall_accuracy = total_correct / total_results
        overall_avg_score = total_score / total_results
        
        print(f"\n Final Statistics:")
        print(f"   Total results processed: {total_results:,}")
        print(f"   Overall accuracy: {overall_accuracy:.1%}")
        print(f"   Overall average score: {overall_avg_score:.3f}")
        print(f"   Unique subsets: {len(all_subsets)}")
        print(f"   Subsets: {', '.join(sorted(all_subsets))}")
    
    return {
        "total_results": total_results,
        "accuracy": overall_accuracy if total_results > 0 else 0,
        "avg_score": overall_avg_score if total_results > 0 else 0,
        "subsets": list(all_subsets)
    }

# Usage
stats = process_all_results("eval_12345")
```

### Pattern 2: Selective Page Processing

```python
def process_specific_pages(evaluation_id: str, start_page: int = 1, end_page: int = None):
    """Process only specific pages of results"""
    client = Atlas()
    
    # Get first page to understand scope
    first_page = client.results.get(evaluation_id=evaluation_id, page=1, page_size=100)
    if not first_page:
        print(" No results found")
        return None
    
    total_pages = first_page.pagination.total_pages
    total_count = first_page.pagination.total_count
    
    # Set end page if not specified
    if end_page is None:
        end_page = total_pages
    
    # Validate range
    end_page = min(end_page, total_pages)
    start_page = max(start_page, 1)
    
    print(f" Processing pages {start_page}-{end_page} of {total_pages} (total: {total_count} results)")
    
    processed_results = []
    
    for page_num in range(start_page, end_page + 1):
        # Reuse first page if processing from page 1
        if page_num == 1 and start_page == 1:
            results_data = first_page
        else:
            results_data = client.results.get(
                evaluation_id=evaluation_id,
                page=page_num,
                page_size=100
            )
        
        if not results_data:
            print(f" Failed to get page {page_num}")
            continue
        
        processed_results.extend(results_data.results)
        print(f" Processed page {page_num}: {len(results_data.results)} results")
    
    print(f" Processed {len(processed_results)} results from pages {start_page}-{end_page}")
    return processed_results

# Usage examples
first_100_results = process_specific_pages("eval_12345", start_page=1, end_page=1)
middle_pages = process_specific_pages("eval_12345", start_page=5, end_page=10)
last_few_pages = process_specific_pages("eval_12345", start_page=18, end_page=20)
```

### Pattern 3: Smart Pagination with Early Stopping

```python
def analyze_with_early_stopping(evaluation_id: str, min_accuracy_threshold: float = 0.7):
    """Stop processing if accuracy drops below threshold"""
    client = Atlas()
    
    page = 1
    page_size = 100
    total_processed = 0
    total_correct = 0
    
    print(f"🎯 Processing until accuracy drops below {min_accuracy_threshold:.1%}")
    
    while True:
        results_data = client.results.get(
            evaluation_id=evaluation_id,
            page=page,
            page_size=page_size
        )
        
        if not results_data or not results_data.results:
            break
        
        # Process current page
        current_results = results_data.results
        page_correct = sum(1 for r in current_results if r.score > 0.5)
        
        total_processed += len(current_results)
        total_correct += page_correct
        
        current_accuracy = total_correct / total_processed
        page_accuracy = page_correct / len(current_results)
        
        print(f" Page {page}: {page_accuracy:.1%} accuracy ({page_correct}/{len(current_results)})")
        print(f" Running total: {current_accuracy:.1%} accuracy ({total_correct}/{total_processed})")
        
        # Check early stopping condition
        if current_accuracy < min_accuracy_threshold and page > 1:
            print(f" Stopping early: accuracy ({current_accuracy:.1%}) below threshold ({min_accuracy_threshold:.1%})")
            break
        
        # Check if we've processed all pages
        if page >= results_data.pagination.total_pages:
            print(f" Processed all {results_data.pagination.total_pages} pages")
            break
        
        page += 1
    
    final_accuracy = total_correct / total_processed if total_processed > 0 else 0
    print(f"\n Final Results:")
    print(f"   Pages processed: {page}/{results_data.pagination.total_pages if 'results_data' in locals() else '?'}")
    print(f"   Results processed: {total_processed}")
    print(f"   Final accuracy: {final_accuracy:.1%}")
    
    return {
        "pages_processed": page,
        "results_processed": total_processed,
        "accuracy": final_accuracy,
        "stopped_early": page < (results_data.pagination.total_pages if 'results_data' in locals() else 1)
    }

# Usage
early_stop_results = analyze_with_early_stopping("eval_12345", min_accuracy_threshold=0.8)
```

### Comparative Analysis

```python
from atlas import Atlas
from typing import List, Dict

def compare_evaluation_results(evaluation_ids: List[str], labels: List[str] = None):
    """Compare results across multiple evaluations"""
    client = Atlas()
    
    if labels and len(labels) != len(evaluation_ids):
        labels = [f"Eval {i+1}" for i in range(len(evaluation_ids))]
    elif not labels:
        labels = [f"Eval {i+1}" for i in range(len(evaluation_ids))]
    
    print(f" Comparing {len(evaluation_ids)} evaluations")
    print(f"{'='*80}")
    
    # Collect results for all evaluations
    all_results = {}
    for eval_id, label in zip(evaluation_ids, labels):
        results = client.results.get(evaluation_id=eval_id)
        if results:
            all_results[label] = results
            print(f" Loaded {len(results)} results for {label}")
        else:
            print(f" No results found for {label} ({eval_id})")
    
    if not all_results:
        print(" No results to compare")
        return
    
    print(f"\n Comparative Analysis:")
    print(f"{'Metric':<20} " + " ".join(f"{label:<15}" for label in labels))
    print("-" * (20 + 15 * len(labels)))
    
    # Compare key metrics
    metrics = {}
    for label, results in all_results.items():
        total_cases = len(results)
        correct_answers = sum(1 for r in results if r.score > 0.5)
        accuracy = correct_answers / total_cases
        avg_score = sum(r.score for r in results) / total_cases
        avg_duration = sum(r.duration for r in results) / len(results)
        
        metrics[label] = {
            "total_cases": total_cases,
            "accuracy": accuracy,
            "avg_score": avg_score,
            "avg_duration": avg_duration
        }
    
    # Print comparison table
    print(f"{'Total Cases':<20} " + " ".join(f"{metrics[label]['total_cases']:<15,}" for label in labels))
    print(f"{'Accuracy':<20} " + " ".join(f"{metrics[label]['accuracy']:<15.1%}" for label in labels))
    print(f"{'Average Score':<20} " + " ".join(f"{metrics[label]['avg_score']:<15.3f}" for label in labels))
    print(f"{'Average Duration':<20} " + " ".join(f"{str(metrics[label]['avg_duration']):<15}" for label in labels))
    
    # Find best performing evaluation
    best_accuracy = max(metrics.values(), key=lambda x: x["accuracy"])
    best_speed = min(metrics.values(), key=lambda x: x["avg_duration"])
    
    best_accuracy_label = next(label for label, data in metrics.items() if data == best_accuracy)
    best_speed_label = next(label for label, data in metrics.items() if data == best_speed)
    
    print(f"\n Winners:")
    print(f"   Best Accuracy: {best_accuracy_label} ({best_accuracy['accuracy']:.1%})")
    print(f"   Fastest: {best_speed_label} ({best_speed['avg_duration']})")
    
    # Subset-level comparison (if results have same subsets)
    if len(all_results) >= 2:
        first_subsets = set(r.subset for r in list(all_results.values())[0])
        common_subsets = first_subsets
        
        for results in list(all_results.values())[1:]:
            result_subsets = set(r.subset for r in results)
            common_subsets = common_subsets.intersection(result_subsets)
        
        if common_subsets:
            print(f"\n Subset Comparison ({len(common_subsets)} common subsets):")
            print(f"{'Subset':<25} " + " ".join(f"{label} Acc":<12 for label in labels))
            print("-" * (25 + 12 * len(labels)))
            
            for subset in sorted(common_subsets):
                subset_accuracies = []
                for label, results in all_results.items():
                    subset_results = [r for r in results if r.subset == subset]
                    if subset_results:
                        subset_accuracy = sum(1 for r in subset_results if r.score > 0.5) / len(subset_results)
                        subset_accuracies.append(f"{subset_accuracy:.1%}")
                    else:
                        subset_accuracies.append("N/A")
                
                print(f"{subset:<25} " + " ".join(f"{acc:<12}" for acc in subset_accuracies))
    
    return metrics

# Usage - compare GPT-4 vs Claude-3 on MMLU
evaluation_ids = ["eval_gpt4_mmlu", "eval_claude3_mmlu", "eval_llama2_mmlu"]
labels = ["GPT-4", "Claude-3", "Llama-2"]

comparison = compare_evaluation_results(evaluation_ids, labels)
```

### Error Analysis

```python
from atlas import Atlas

def analyze_failures(evaluation_id: str, error_threshold: float = 0.3):
    """Analyze cases where the model performed poorly"""
    client = Atlas()
    
    results = client.results.get(evaluation_id=evaluation_id)
    if not results:
        print(f" No results found for evaluation {evaluation_id}")
        return None
    
    # Find poor-performing cases
    poor_results = [r for r in results if r.score < error_threshold]
    good_results = [r for r in results if r.score >= error_threshold]
    
    print(f" Error Analysis for {evaluation_id}")
    print(f"{'='*60}")
    print(f"Total cases: {len(results)}")
    print(f"Poor performance (< {error_threshold}): {len(poor_results)} ({len(poor_results)/len(results):.1%})")
    print(f"Good performance (>= {error_threshold}): {len(good_results)} ({len(good_results)/len(results):.1%})")
    
    if not poor_results:
        print(" No poor-performing cases found!")
        return {"poor_results": [], "analysis": "No errors to analyze"}
    
    # Analyze failure patterns by subset
    failure_by_subset = {}
    for result in poor_results:
        if result.subset not in failure_by_subset:
            failure_by_subset[result.subset] = []
        failure_by_subset[result.subset].append(result)
    
    print(f"\n Failure Distribution by Subset:")
    for subset, failures in sorted(failure_by_subset.items(), key=lambda x: len(x[1]), reverse=True):
        total_in_subset = len([r for r in results if r.subset == subset])
        failure_rate = len(failures) / total_in_subset
        print(f"   {subset}: {len(failures)}/{total_in_subset} failures ({failure_rate:.1%})")
    
    # Show worst-performing examples
    worst_results = sorted(poor_results, key=lambda x: x.score)[:5]
    
    print(f"\n Worst Performing Examples:")
    for i, result in enumerate(worst_results, 1):
        print(f"\n   Example {i} [Score: {result.score:.3f}]")
        print(f"   Subset: {result.subset}")
        print(f"   Prompt: {result.prompt[:200]}...")
        print(f"   Model Answer: {result.result[:100]}...")
        print(f"   Expected: {result.truth[:100]}...")
        print(f"   Duration: {result.duration}")
        
        if result.metrics:
            print(f"   Additional Metrics: {result.metrics}")
    
    # Common failure patterns
    print(f"\n Common Patterns in Failures:")
    
    # Analyze prompt lengths
    poor_prompt_lengths = [len(r.prompt) for r in poor_results]
    good_prompt_lengths = [len(r.prompt) for r in good_results]
    
    avg_poor_prompt_len = sum(poor_prompt_lengths) / len(poor_prompt_lengths)
    avg_good_prompt_len = sum(good_prompt_lengths) / len(good_prompt_lengths)
    
    print(f"   Average prompt length in failures: {avg_poor_prompt_len:.0f} chars")
    print(f"   Average prompt length in successes: {avg_good_prompt_len:.0f} chars")
    
    # Analyze response lengths
    poor_response_lengths = [len(r.result) for r in poor_results]
    good_response_lengths = [len(r.result) for r in good_results]
    
    avg_poor_response_len = sum(poor_response_lengths) / len(poor_response_lengths)
    avg_good_response_len = sum(good_response_lengths) / len(good_response_lengths)
    
    print(f"   Average response length in failures: {avg_poor_response_len:.0f} chars")
    print(f"   Average response length in successes: {avg_good_response_len:.0f} chars")
    
    # Analyze durations
    avg_poor_duration = sum(r.duration for r in poor_results) / len(poor_results)
    avg_good_duration = sum(r.duration for r in good_results) / len(good_results)
    
    print(f"   Average duration for failures: {avg_poor_duration}")
    print(f"   Average duration for successes: {avg_good_duration}")
    
    return {
        "poor_results": poor_results,
        "failure_by_subset": failure_by_subset,
        "worst_examples": worst_results,
        "patterns": {
            "avg_poor_prompt_len": avg_poor_prompt_len,
            "avg_good_prompt_len": avg_good_prompt_len,
            "avg_poor_response_len": avg_poor_response_len,
            "avg_good_response_len": avg_good_response_len,
            "avg_poor_duration": avg_poor_duration,
            "avg_good_duration": avg_good_duration
        }
    }

# Usage
error_analysis = analyze_failures("eval_12345", error_threshold=0.5)
```

## Advanced Result Processing

### Batch Processing Large Result Sets

```python
from atlas import Atlas
from typing import Iterator, List
import time

def process_results_in_batches(evaluation_id: str, batch_size: int = 100, processor_func=None):
    """Process large result sets in manageable batches"""
    client = Atlas()
    
    results = client.results.get(evaluation_id=evaluation_id)
    if not results:
        print(f" No results found for evaluation {evaluation_id}")
        return None
    
    total_results = len(results)
    print(f" Processing {total_results:,} results in batches of {batch_size}")
    
    if not processor_func:
        # Default processor: just count scores
        def processor_func(batch):
            return {
                "count": len(batch),
                "avg_score": sum(r.score for r in batch) / len(batch),
                "correct": sum(1 for r in batch if r.score > 0.5)
            }
    
    batch_results = []
    
    for i in range(0, total_results, batch_size):
        batch = results[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_results + batch_size - 1) // batch_size
        
        print(f" Processing batch {batch_num}/{total_batches} ({len(batch)} items)")
        
        start_time = time.time()
        batch_result = processor_func(batch)
        end_time = time.time()
        
        batch_result.update({
            "batch_num": batch_num,
            "processing_time": end_time - start_time,
            "items_processed": len(batch)
        })
        
        batch_results.append(batch_result)
        
        print(f"    Completed in {batch_result['processing_time']:.2f}s")
        
        # Small delay to prevent overwhelming the system
        if batch_num < total_batches:
            time.sleep(0.1)
    
    # Aggregate results
    total_processing_time = sum(br["processing_time"] for br in batch_results)
    total_correct = sum(br.get("correct", 0) for br in batch_results)
    overall_accuracy = total_correct / total_results
    
    print(f"\n Batch Processing Summary:")
    print(f"   Total batches: {len(batch_results)}")
    print(f"   Total processing time: {total_processing_time:.2f}s")
    print(f"   Average time per batch: {total_processing_time/len(batch_results):.2f}s")
    print(f"   Overall accuracy: {overall_accuracy:.1%}")
    
    return {
        "batch_results": batch_results,
        "summary": {
            "total_items": total_results,
            "total_batches": len(batch_results),
            "total_processing_time": total_processing_time,
            "overall_accuracy": overall_accuracy
        }
    }

# Custom processor for subset analysis
def subset_analyzer(batch):
    """Custom processor that analyzes subsets in a batch"""
    subset_stats = {}
    
    for result in batch:
        if result.subset not in subset_stats:
            subset_stats[result.subset] = {"count": 0, "total_score": 0, "correct": 0}
        
        subset_stats[result.subset]["count"] += 1
        subset_stats[result.subset]["total_score"] += result.score
        if result.score > 0.5:
            subset_stats[result.subset]["correct"] += 1
    
    return {
        "subset_stats": subset_stats,
        "unique_subsets": len(subset_stats)
    }

# Usage
batch_results = process_results_in_batches(
    evaluation_id="eval_12345",
    batch_size=50,
    processor_func=subset_analyzer
)
```

### Result Caching and Persistence

```python
import json
import pickle
from pathlib import Path
from datetime import datetime
from atlas import Atlas
import atlas

class ResultsCache:
    """Cache evaluation results to avoid repeated API calls"""
    
    def __init__(self, cache_dir: str = "results_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_path(self, evaluation_id: str, format: str = "json") -> Path:
        """Get cache file path for an evaluation"""
        return self.cache_dir / f"{evaluation_id}_results.{format}"
    
    def _get_metadata_path(self, evaluation_id: str) -> Path:
        """Get metadata file path for an evaluation"""
        return self.cache_dir / f"{evaluation_id}_metadata.json"
    
    def is_cached(self, evaluation_id: str) -> bool:
        """Check if results are already cached"""
        return self._get_cache_path(evaluation_id).exists()
    
    def save_results(self, evaluation_id: str, results: list, metadata: dict = None):
        """Save results to cache"""
        try:
            # Save as JSON (human-readable)
            json_path = self._get_cache_path(evaluation_id, "json")
            with open(json_path, 'w') as f:
                # Convert results to serializable format
                serializable_results = []
                for result in results:
                    result_dict = {
                        "subset": result.subset,
                        "prompt": result.prompt,
                        "result": result.result,
                        "truth": result.truth,
                        "score": result.score,
                        "duration": str(result.duration),  # Convert timedelta to string
                        "metrics": result.metrics
                    }
                    serializable_results.append(result_dict)
                
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            # Save as pickle (preserves exact object types)
            pickle_path = self._get_cache_path(evaluation_id, "pkl")
            with open(pickle_path, 'wb') as f:
                pickle.dump(results, f)
            
            # Save metadata
            if not metadata:
                metadata = {}
            
            metadata.update({
                "evaluation_id": evaluation_id,
                "cached_at": datetime.now().isoformat(),
                "result_count": len(results),
                "cache_format": "both"
            })
            
            metadata_path = self._get_metadata_path(evaluation_id)
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"💾 Cached {len(results)} results for {evaluation_id}")
            
        except Exception as e:
            print(f" Error caching results: {e}")
    
    def load_results(self, evaluation_id: str, format: str = "pickle"):
        """Load results from cache"""
        try:
            if format == "pickle":
                cache_path = self._get_cache_path(evaluation_id, "pkl")
                with open(cache_path, 'rb') as f:
                    results = pickle.load(f)
            else:
                cache_path = self._get_cache_path(evaluation_id, "json")
                with open(cache_path, 'r') as f:
                    results = json.load(f)
            
            print(f"💾 Loaded {len(results)} results from cache for {evaluation_id}")
            return results
            
        except Exception as e:
            print(f" Error loading cached results: {e}")
            return None
    
    def get_metadata(self, evaluation_id: str):
        """Get cached metadata"""
        try:
            metadata_path = self._get_metadata_path(evaluation_id)
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f" Error loading metadata: {e}")
            return None

def get_results_with_cache(evaluation_id: str, cache: ResultsCache = None, force_refresh: bool = False):
    """Get results with automatic caching"""
    if not cache:
        cache = ResultsCache()
    
    # Check cache first (unless force refresh)
    if not force_refresh and cache.is_cached(evaluation_id):
        print(f"📂 Loading results from cache...")
        cached_results = cache.load_results(evaluation_id)
        
        if cached_results:
            metadata = cache.get_metadata(evaluation_id)
            if metadata:
                cached_at = metadata.get("cached_at", "unknown")
                print(f"📅 Cached at: {cached_at}")
            return cached_results
    
    # Fetch from API
    print(f"🌐 Fetching fresh results from API...")
    client = Atlas()
    
    try:
        results = client.results.get(evaluation_id=evaluation_id)
        
        if results:
            # Cache the results
            cache.save_results(evaluation_id, results)
            return results
        else:
            print(f" No results found for evaluation {evaluation_id}")
            return None
            
    except atlas.APIError as e:
        print(f" Error fetching results: {e}")
        
        # Try to return cached results as fallback
        if cache.is_cached(evaluation_id):
            print(f" Falling back to cached results...")
            return cache.load_results(evaluation_id)
        
        return None

# Usage examples
cache = ResultsCache("./my_results_cache")

# First call - fetches from API and caches
results1 = get_results_with_cache("eval_12345", cache)

# Second call - loads from cache
results2 = get_results_with_cache("eval_12345", cache)

# Force refresh from API
results3 = get_results_with_cache("eval_12345", cache, force_refresh=True)

# Batch cache multiple evaluations
evaluation_ids = ["eval_001", "eval_002", "eval_003"]

for eval_id in evaluation_ids:
    results = get_results_with_cache(eval_id, cache)
    if results:
        print(f" {eval_id}: {len(results)} results cached")

print(f"\n📁 Cache contents:")
for cache_file in cache.cache_dir.glob("*.json"):
    if cache_file.name.endswith("_metadata.json"):
        continue
    evaluation_id = cache_file.stem.replace("_results", "")
    metadata = cache.get_metadata(evaluation_id)
    if metadata:
        count = metadata.get("result_count", "unknown")
        cached_at = metadata.get("cached_at", "unknown")
        print(f"   {evaluation_id}: {count} results (cached: {cached_at})")
```

### Export and Reporting

```python
import csv
from pathlib import Path
from datetime import datetime
from atlas import Atlas

def export_results_to_csv(evaluation_id: str, output_path: str = None):
    """Export evaluation results to CSV format"""
    client = Atlas()
    
    results = client.results.get(evaluation_id=evaluation_id)
    if not results:
        print(f" No results found for evaluation {evaluation_id}")
        return None
    
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results_{evaluation_id}_{timestamp}.csv"
    
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'subset', 'prompt', 'model_response', 'expected_answer', 
                'score', 'duration_ms', 'prompt_length', 'response_length'
            ]
            
            # Add metric columns if they exist
            if results and results[0].metrics:
                metric_keys = list(results[0].metrics.keys())
                fieldnames.extend([f"metric_{key}" for key in metric_keys])
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                row = {
                    'subset': result.subset,
                    'prompt': result.prompt,
                    'model_response': result.result,
                    'expected_answer': result.truth,
                    'score': result.score,
                    'duration_ms': int(result.duration.total_seconds() * 1000),
                    'prompt_length': len(result.prompt),
                    'response_length': len(result.result)
                }
                
                # Add metrics if present
                if result.metrics:
                    for key, value in result.metrics.items():
                        row[f"metric_{key}"] = value
                
                writer.writerow(row)
        
        print(f" Exported {len(results)} results to {output_path}")
        return output_path
        
    except Exception as e:
        print(f" Error exporting to CSV: {e}")
        return None

def generate_summary_report(evaluation_ids: list, output_path: str = None):
    """Generate a summary report comparing multiple evaluations"""
    client = Atlas()
    
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"evaluation_summary_{timestamp}.txt"
    
    with open(output_path, 'w') as f:
        f.write("ATLAS EVALUATION SUMMARY REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Evaluations analyzed: {len(evaluation_ids)}\n\n")
        
        for i, eval_id in enumerate(evaluation_ids, 1):
            f.write(f"EVALUATION {i}: {eval_id}\n")
            f.write("-" * 30 + "\n")
            
            results = client.results.get(evaluation_id=eval_id)
            
            if not results:
                f.write(" No results found\n\n")
                continue
            
            # Calculate statistics
            total_cases = len(results)
            correct_answers = sum(1 for r in results if r.score > 0.5)
            accuracy = correct_answers / total_cases
            avg_score = sum(r.score for r in results) / total_cases
            avg_duration = sum(r.duration for r in results) / len(results)
            
            # Write statistics
            f.write(f"Total test cases: {total_cases:,}\n")
            f.write(f"Correct answers: {correct_answers:,}\n")
            f.write(f"Accuracy: {accuracy:.1%}\n")
            f.write(f"Average score: {avg_score:.3f}\n")
            f.write(f"Average duration: {avg_duration}\n")
            
            # Subset breakdown
            subset_stats = {}
            for result in results:
                if result.subset not in subset_stats:
                    subset_stats[result.subset] = []
                subset_stats[result.subset].append(result.score)
            
            f.write(f"\nSubset Performance:\n")
            for subset, scores in sorted(subset_stats.items()):
                subset_accuracy = sum(1 for s in scores if s > 0.5) / len(scores)
                subset_avg = sum(scores) / len(scores)
                f.write(f"  {subset}: {subset_accuracy:.1%} accuracy, {subset_avg:.3f} avg score ({len(scores)} cases)\n")
            
            f.write("\n")
        
        f.write("END OF REPORT\n")
    
    print(f" Summary report generated: {output_path}")
    return output_path

# Usage examples

# Export single evaluation to CSV
csv_path = export_results_to_csv("eval_12345")

# Generate summary report for multiple evaluations
evaluation_list = ["eval_gpt4_mmlu", "eval_claude3_mmlu", "eval_llama2_mmlu"]
report_path = generate_summary_report(evaluation_list)

print(f"Files generated:")
print(f"  CSV Export: {csv_path}")
print(f"  Summary Report: {report_path}")
```
