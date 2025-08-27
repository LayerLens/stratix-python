# Environment Variables

This guide covers secure practices for managing environment variables when using the Atlas Python SDK.

## Overview

Environment variables provide a secure way to configure your Atlas SDK without hardcoding sensitive credentials in your source code. This approach separates configuration from code and enables different configurations for different environments.

## Required Environment Variables

The Atlas SDK uses these primary environment variables:

| Variable                  | Description        | Required | Example        |
| ------------------------- | ------------------ | -------- | -------------- |
| `LAYERLENS_ATLAS_API_KEY` | Your Atlas API key | Yes      | `sk-abc123...` |

## Setting Environment Variables

### Development Environment

**Linux/macOS (Bash/Zsh)**:

```bash
# Set for current session
export LAYERLENS_ATLAS_API_KEY="sk-your-key-here"

# Add to shell profile for persistence (.bashrc, .zshrc, etc.)
echo 'export LAYERLENS_ATLAS_API_KEY="sk-your-key-here"' >> ~/.bashrc

# Reload shell configuration
source ~/.bashrc
```

**Windows Command Prompt**:

```cmd
# Set for current session
set LAYERLENS_ATLAS_API_KEY=sk-your-key-here

# Set permanently (requires admin rights)
setx LAYERLENS_ATLAS_API_KEY "sk-your-key-here"
```

**Windows PowerShell**:

```powershell
# Set for current session
$env:LAYERLENS_ATLAS_API_KEY="sk-your-key-here"

# Set permanently for current user
[Environment]::SetEnvironmentVariable("LAYERLENS_ATLAS_API_KEY", "sk-your-key-here", "User")
```

### Verification

**Check if variables are set correctly**:

```python
import os

def verify_atlas_environment():
    """Verify Atlas environment variables are configured"""
    required_vars = {
        'LAYERLENS_ATLAS_API_KEY': 'API Key',
    }

    print("🔍 Atlas Environment Variable Check")
    print("=" * 40)

    all_set = True
    for var_name, description in required_vars.items():
        value = os.getenv(var_name)

        if value:
            # Don't print the full value for security
            masked_value = f"{value[:8]}..." if len(value) > 8 else "***"
            print(f"✅ {description}: {masked_value}")
        else:
            print(f"❌ {description}: Not set")
            all_set = False


    if all_set:
        print(f"\n🎉 All required variables are set!")
    else:
        print(f"\n⚠️ Some required variables are missing")

    return all_set

# Run verification
verify_atlas_environment()
```

## Using .env Files

### Creating .env Files

**.env file for development**:

```bash
# .env
LAYERLENS_ATLAS_API_KEY=sk-development-key-here

# Optional: Set environment name
ATLAS_ENV=development
```

**Loading .env files in Python**:

```python
from dotenv import load_dotenv
import os

# Load .env file from current directory
load_dotenv()

# Or load specific .env file
load_dotenv('.env.development')

# Or load from specific path
load_dotenv('/path/to/your/.env')

# Verify variables are loaded
from layerlens import Atlas

try:
    client = Atlas()  # Will use environment variables
    print("✅ Atlas client initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize client: {e}")
```
