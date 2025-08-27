# API Key Management

This guide covers best practices for securely managing your Atlas API keys throughout the development lifecycle.

## API Key Security Fundamentals

### What Makes API Keys Sensitive

API keys are sensitive credentials that provide access to your Atlas organization and projects. They should be treated with the same level of security as passwords or other authentication tokens.

**Risks of compromised API keys**:

- Unauthorized access to your evaluations and data
- Unintended usage charges on your account
- Potential data breaches or intellectual property theft
- Abuse of your API quotas and rate limits

### API Key Best Practices

1. **Never hardcode API keys in source code**
2. **Use environment variables or secure credential stores**
3. **Rotate keys regularly**
4. **Use different keys for different environments**
5. **Monitor key usage and access patterns**
6. **Revoke unused or compromised keys immediately**

## Secure API Key Storage

### Environment Variables (Recommended)

**✅ Good - Using environment variables**:

```python
import os
from layerlens import Atlas

# Secure: Load from environment variables
client = Atlas(
    api_key=os.getenv('LAYERLENS_ATLAS_API_KEY'),
)
```

### Setting Environment Variables Securely

**Linux/macOS**:

```bash
# Add to your shell profile (.bashrc, .zshrc, etc.)
export LAYERLENS_ATLAS_API_KEY="sk-your-key-here"

# Reload your shell configuration
source ~/.bashrc  # or ~/.zshrc
```

**Windows**:

```cmd
# Command Prompt (persistent)
setx LAYERLENS_ATLAS_API_KEY "sk-your-key-here"

# PowerShell (session-only)
$env:LAYERLENS_ATLAS_API_KEY="sk-your-key-here"
```

### Using .env Files

**Create a .env file** (never commit this to version control):

```bash
# .env
LAYERLENS_ATLAS_API_KEY=sk-your-key-here
```

**Load .env file in Python**:

```python
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

from layerlens import Atlas

# Now environment variables are available
client = Atlas()
```

**Important**: Add `.env` to your `.gitignore` file:

```bash
# .gitignore
.env
.env.local
.env.*.local
*.env
```
