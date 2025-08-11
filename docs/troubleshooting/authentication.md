# Authentication Problems

This guide covers authentication-related issues and their solutions when using the Atlas Python SDK.

## Understanding Atlas Authentication

The Atlas SDK uses API key-based authentication with three required components:

1. **API Key**: Your secret authentication token
2. **Organization ID**: Your organization identifier
3. **Project ID**: The specific project you're working with

## Common Authentication Errors

### Invalid or Missing API Key

**Error**: `AuthenticationError: Invalid API key`

**Symptoms**:

- 401 Unauthorized responses
- "Invalid API key" error messages
- Authentication fails immediately

### Missing Required Configuration

**Error**: `AtlasError: The api_key client option must be set either by passing api_key to the client or by setting the LAYERLENS_ATLAS_API_KEY environment variable`

**Solutions**:

1. **Check all required environment variables**:

   ```bash
   # Linux/macOS
   echo $LAYERLENS_ATLAS_API_KEY

   # Windows
   echo %LAYERLENS_ATLAS_API_KEY%
   ```

2. **Set environment variables properly**:

   ```bash
   # Linux/macOS - in your shell profile (.bashrc, .zshrc, etc.)
   export LAYERLENS_ATLAS_API_KEY="sk-..."

   # Windows - persistently
   setx LAYERLENS_ATLAS_API_KEY "sk-..."
   ```

3. **Use .env file**:

   ```bash
   # Create .env file in your project root
   LAYERLENS_ATLAS_API_KEY=sk-your-key-here
   ```

   ```python
   # Load .env file in your Python code
   from dotenv import load_dotenv
   import os

   load_dotenv()

   from atlas import Atlas
   client = Atlas()
   ```

### Permission Denied Errors

**Error**: `PermissionDeniedError: 403 Forbidden`

**Symptoms**:

- Valid API key but still get 403 errors
- Can authenticate but cannot create evaluations
- Access denied to specific models or benchmarks

**Diagnosis**:

```python
import atlas
from atlas import Atlas

def diagnose_permissions():
    client = Atlas()

    print("🔍 Permission Diagnosis:")

    # Test basic access
    try:
        # This should fail with specific error types
        evaluation = client.evaluations.create(
            model="test-model",
            benchmark="test-benchmark"
        )
    except atlas.AuthenticationError:
        print("   ❌ Authentication failed - invalid API key")
        return
    except atlas.PermissionDeniedError:
        print("   ❌ Permission denied - valid key, insufficient permissions")
    except atlas.NotFoundError:
        print("   ✅ Authentication works (model/benchmark not found is normal)")
    except Exception as e:
        print(f"   ❓ Unexpected error: {e}")

    # Test with common models/benchmarks
    test_combinations = [
        ("gpt-3.5-turbo", "mmlu"),
        ("gpt-4", "hellaswag"),
        ("claude-3-sonnet", "arc-challenge")
    ]

    print("\n   Testing access to specific resources:")

    for model, benchmark in test_combinations:
        try:
            evaluation = client.evaluations.create(model=model, benchmark=benchmark)
            if evaluation:
                print(f"   ✅ {model} + {benchmark}: Access granted")
        except atlas.PermissionDeniedError:
            print(f"   ❌ {model} + {benchmark}: Permission denied")
        except atlas.NotFoundError:
            print(f"   ⚠️ {model} + {benchmark}: Resource not found")
        except Exception as e:
            print(f"   ❓ {model} + {benchmark}: {e}")

diagnose_permissions()
```

### Organization/Project Access Issues

**Problem**: Valid API key but wrong organization or project

**Symptoms**:

- Authentication succeeds
- Cannot access expected models or benchmarks
- Permission errors for resources you should have access to

**Diagnosis**:

```python
import os
from atlas import Atlas
import atlas

def verify_org_project_access():
    # Test with different org/project combinations
    api_key = os.getenv('LAYERLENS_ATLAS_API_KEY')

    if not api_key:
        print("❌ No API key found")
        return

    try:
        client = Atlas(api_key=api_key)
        evaluation = client.evaluations.create(model="test", benchmark="test")

    except atlas.AuthenticationError:
        print("  ❌ Authentication failed")
    except atlas.PermissionDeniedError:
        print("  ❌ Permission denied - check org/project IDs")
    except atlas.NotFoundError:
        print("  ✅ Access granted (test model not found is expected)")
    except Exception as e:
        print(f"  ❓ Error: {e}")

verify_org_project_access()
```
