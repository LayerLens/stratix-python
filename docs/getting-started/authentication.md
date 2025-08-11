# Authentication & Configuration

The Atlas Python SDK uses API key authentication to securely access the LayerLens Atlas API. This guide covers how to set up authentication and configure your client.

## Required Credentials

You need three pieces of information to use the Atlas SDK:

1. **API Key** - Your secret API key for authentication
2. **Organization ID** - Your organization identifier 
3. **Project ID** - The project you want to work with

## Getting Your Credentials

1. **Log in to LayerLens Atlas**: Visit the LayerLens Atlas dashboard
2. **Navigate to Settings**: Go to your account or organization settings
3. **Generate API Key**: Create a new API key if you don't have one
4. **Copy IDs**: Note your Organization ID and Project ID from the dashboard

## Environment Variables (Recommended)

The most secure way to configure authentication is using environment variables:

### Setting Environment Variables

**Linux/macOS:**
```bash
export LAYERLENS_ATLAS_API_KEY="your_api_key_here"
export LAYERLENS_ATLAS_ORG_ID="your_org_id_here"
export LAYERLENS_ATLAS_PROJECT_ID="your_project_id_here"
```

**Windows (Command Prompt):**
```cmd
set LAYERLENS_ATLAS_API_KEY=your_api_key_here
set LAYERLENS_ATLAS_ORG_ID=your_org_id_here
set LAYERLENS_ATLAS_PROJECT_ID=your_project_id_here
```

**Windows (PowerShell):**
```powershell
$env:LAYERLENS_ATLAS_API_KEY="your_api_key_here"
$env:LAYERLENS_ATLAS_ORG_ID="your_org_id_here" 
$env:LAYERLENS_ATLAS_PROJECT_ID="your_project_id_here"
```

### Using a `.env` File

Create a `.env` file in your project root:

```bash
LAYERLENS_ATLAS_API_KEY=your_api_key_here
LAYERLENS_ATLAS_ORG_ID=your_org_id_here
LAYERLENS_ATLAS_PROJECT_ID=your_project_id_here
```

Then load it in your Python code:

```python
from dotenv import load_dotenv
import os
from atlas import Atlas

# Load environment variables from .env file
load_dotenv()

# Client will automatically use environment variables
client = Atlas()
```

> **⚠️ Security Note**: Never commit `.env` files to version control. Add `.env` to your `.gitignore` file.

## Client Configuration

### Automatic Configuration

When environment variables are set, the client configures itself automatically:

```python
from atlas import Atlas

# Uses environment variables automatically
client = Atlas()
```

### Explicit Configuration

You can also pass credentials directly to the client:

```python
from atlas import Atlas

client = Atlas(
    api_key="your_api_key_here",
    organization_id="your_org_id_here",
    project_id="your_project_id_here"
)
```

### Mixed Configuration

You can mix environment variables with explicit parameters:

```python
import os
from atlas import Atlas

client = Atlas(
    api_key=os.environ.get("LAYERLENS_ATLAS_API_KEY"),
    organization_id="override_org_id",  # Override from environment
    project_id=os.environ.get("LAYERLENS_ATLAS_PROJECT_ID")
)
```

## Advanced Configuration

### Timeout Configuration

Configure request timeouts:

```python
from atlas import Atlas
import httpx

# Simple timeout (10 seconds)
client = Atlas(timeout=10.0)

# Advanced timeout configuration
client = Atlas(
    timeout=httpx.Timeout(
        connect=5.0,    # Connection timeout
        read=30.0,      # Read timeout
        write=10.0,     # Write timeout
        pool=2.0        # Pool timeout
    )
)
```

## Validation

The SDK will validate your configuration on first use:

```python
from atlas import Atlas

try:
    client = Atlas()
    # Test the connection
    evaluation = client.evaluations.create(model="test", benchmark="test")
except atlas.AuthenticationError:
    print("Invalid API key or authentication failed")
except atlas.PermissionDeniedError:
    print("Valid API key but insufficient permissions")
except atlas.AtlasError as e:
    print(f"Configuration error: {e}")
```

## Security Best Practices

1. **Never hardcode credentials** in your source code
2. **Use environment variables** or secure credential management systems
3. **Rotate API keys regularly** for enhanced security
4. **Use different API keys** for different environments (dev, staging, prod)
5. **Monitor API key usage** in the LayerLens dashboard
6. **Revoke unused keys** immediately

## Next Steps

Once authentication is configured, proceed to the [Quick Start Guide](quickstart.md) to make your first API call.