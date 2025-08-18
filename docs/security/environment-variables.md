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
from atlas import Atlas

try:
    client = Atlas()  # Will use environment variables
    print("✅ Atlas client initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize client: {e}")
```

### Environment-Specific .env Files

**Create separate files for each environment**:

**.env.development**:

```bash
LAYERLENS_ATLAS_API_KEY=sk-dev-key-here
```

**.env.staging**:

```bash
LAYERLENS_ATLAS_API_KEY=sk-staging-key-here
```

**.env.production**:

```bash
LAYERLENS_ATLAS_API_KEY=sk-prod-key-here
```

**Load environment-specific configuration**:

```python
import os
from dotenv import load_dotenv
from atlas import Atlas

def load_environment_config():
    """Load environment-specific configuration"""
    # Determine environment
    env = os.getenv('ATLAS_ENV', 'development')

    # Load base .env file first
    load_dotenv('.env')

    # Override with environment-specific file
    env_file = f'.env.{env}'
    if os.path.exists(env_file):
        load_dotenv(env_file, override=True)
        print(f"📄 Loaded configuration from {env_file}")
    else:
        print(f"⚠️ Environment file {env_file} not found, using base configuration")

    return env

def get_atlas_client():
    """Get Atlas client with environment-specific configuration"""
    env = load_environment_config()

    # Create client with loaded environment variables
    client = Atlas()

    # Log configuration (without sensitive data)
    print(f"🌍 Environment: {env}")
    print(f"🔗 Base URL: {client.base_url}")
    print(f"⏱️ Timeout: {client.timeout}s")

    return client

# Usage
client = get_atlas_client()
```

## Security Best Practices

### Environment Variable Security Checklist

- [ ] ✅ No sensitive values hardcoded in source code
- [ ] ✅ .env files added to .gitignore
- [ ] ✅ Different credentials for each environment (dev/staging/prod)
- [ ] ✅ Environment variables validated before use
- [ ] ✅ Production secrets managed through secure systems (not .env files)
- [ ] ✅ Regular rotation of API keys
- [ ] ✅ Monitoring for credential exposure in logs
- [ ] ✅ Team members trained on secure credential handling
