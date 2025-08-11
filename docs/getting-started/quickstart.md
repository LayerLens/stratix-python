# Quick Start Guide

This guide will help you make your first API call with the Atlas Python SDK. We'll walk through creating an evaluation and retrieving results.

## Prerequisites

Before you begin, ensure you have:

1. ✅ [Installed the Atlas SDK](installation.md)
2. ✅ [Configured authentication](authentication.md) with your API key, organization ID, and project ID
3. ✅ Access to LayerLens Atlas platform

## Your First Evaluation

Let's create a simple evaluation to test your setup:

```python
import os
from atlas import Atlas

# Initialize the client (uses environment variables)
client = Atlas(
    api_key=os.environ.get("LAYERLENS_ATLAS_API_KEY"),
    organization_id=os.environ.get("LAYERLENS_ATLAS_ORG_ID"),
    project_id=os.environ.get("LAYERLENS_ATLAS_PROJECT_ID")
)

# Create an evaluation
evaluation = client.evaluations.create(
    model="gpt-3.5-turbo",  # Replace with your model ID
    benchmark="mmlu"        # Replace with your benchmark ID
)

if evaluation:
    print(f"✅ Evaluation created successfully!")
    print(f"   ID: {evaluation.id}")
    print(f"   Status: {evaluation.status}")
    print(f"   Model: {evaluation.model_name}")
    print(f"   Benchmark: {evaluation.dataset_name}")
else:
    print("❌ Failed to create evaluation")
```

## Understanding the Response

A successful evaluation creation returns an `Evaluation` object with the following key properties:

```python
evaluation = client.evaluations.create(model="gpt-4", benchmark="mmlu")

print(f"Evaluation ID: {evaluation.id}")
print(f"Status: {evaluation.status}")
print(f"Status Description: {evaluation.status_description}")
print(f"Submitted At: {evaluation.submitted_at}")
print(f"Model: {evaluation.model_name} ({evaluation.model_company})")
print(f"Dataset: {evaluation.dataset_name}")

# Available when evaluation is completed
if evaluation.status == "completed":
    print(f"Accuracy: {evaluation.accuracy}")
    print(f"Readability Score: {evaluation.readability_score}")
    print(f"Toxicity Score: {evaluation.toxicity_score}")
    print(f"Ethics Score: {evaluation.ethics_score}")
```

## Retrieving Results

Once your evaluation is complete, you can retrieve detailed results:

```python
# Wait for evaluation to complete, then get results
if evaluation and evaluation.status == "completed":
    results = client.results.get(evaluation_id=evaluation.id)
    
    if results:
        print(f"📊 Retrieved {len(results)} results")
        
        # Examine the first result
        first_result = results[0]
        print(f"\nFirst Result:")
        print(f"  Subset: {first_result.subset}")
        print(f"  Prompt: {first_result.prompt[:100]}...")  # First 100 chars
        print(f"  Model Output: {first_result.result[:100]}...")
        print(f"  Expected Answer: {first_result.truth}")
        print(f"  Score: {first_result.score}")
        print(f"  Duration: {first_result.duration}")
        print(f"  Metrics: {first_result.metrics}")
```

## Complete Example

Here's a complete example that creates an evaluation and waits for results:

```python
import os
import time
from atlas import Atlas

def main():
    # Initialize client
    client = Atlas()
    
    print("🚀 Creating evaluation...")
    
    try:
        # Create evaluation
        evaluation = client.evaluations.create(
            model="gpt-3.5-turbo",
            benchmark="mmlu"
        )
        
        if not evaluation:
            print("❌ Failed to create evaluation")
            return
            
        print(f"✅ Evaluation created: {evaluation.id}")
        print(f"   Status: {evaluation.status}")
        
        # Poll for completion (in a real app, use webhooks instead)
        print("\n⏳ Waiting for evaluation to complete...")
        
        while evaluation.status not in ["completed", "failed", "cancelled"]:
            time.sleep(30)  # Wait 30 seconds
            
            # In practice, you'd re-fetch the evaluation status
            # This is a simplified example
            print(f"   Status: {evaluation.status}")
        
        if evaluation.status == "completed":
            print(f"🎉 Evaluation completed!")
            print(f"   Accuracy: {evaluation.accuracy:.2%}")
            
            # Get detailed results
            results = client.results.get(evaluation_id=evaluation.id)
            print(f"📊 Retrieved {len(results) if results else 0} detailed results")
            
        else:
            print(f"❌ Evaluation failed with status: {evaluation.status}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
```

## Error Handling

Always wrap your API calls in try-catch blocks:

```python
import atlas
from atlas import Atlas

client = Atlas()

try:
    evaluation = client.evaluations.create(
        model="gpt-4",
        benchmark="mmlu"
    )
except atlas.AuthenticationError:
    print("❌ Authentication failed. Check your API key.")
except atlas.PermissionDeniedError:
    print("❌ Permission denied. Check your organization/project access.")
except atlas.RateLimitError:
    print("❌ Rate limit exceeded. Please wait and try again.")
except atlas.APIConnectionError as e:
    print(f"❌ Connection error: {e}")
except atlas.APIStatusError as e:
    print(f"❌ API error: {e.status_code} - {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
```

## Available Models and Benchmarks

To see available models and benchmarks, you can:

1. **Check the LayerLens Atlas dashboard** for the most up-to-date list
2. **Contact support** for specific model or benchmark IDs

## What's Next?

Now that you've successfully made your first API call:

1. **[Explore the API Reference](../api-reference/)** - Learn about all available methods
2. **[Check out Code Examples](../examples/)** - See practical usage patterns
3. **[Review Error Handling](../api-reference/errors.md)** - Handle edge cases gracefully
4. **[Security Best Practices](../security/)** - Secure your API usage

## Need Help?

- **Documentation**: Browse the complete [API Reference](../api-reference/)
- **Examples**: Check out more [Code Examples](../examples/)
- **Support**: Contact LayerLens support through your dashboard for technical assistance
- **Status**: Check [status.layerlens.com](https://status.layerlens.com) for service updates