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
from atlas import Atlas

# Secure: Load from environment variables
client = Atlas(
    api_key=os.getenv('LAYERLENS_ATLAS_API_KEY'),
    organization_id=os.getenv('LAYERLENS_ATLAS_ORG_ID'),
    project_id=os.getenv('LAYERLENS_ATLAS_PROJECT_ID')
)
```
### Setting Environment Variables Securely

**Linux/macOS**:
```bash
# Add to your shell profile (.bashrc, .zshrc, etc.)
export LAYERLENS_ATLAS_API_KEY="sk-your-key-here"
export LAYERLENS_ATLAS_ORG_ID="org-your-org-here" 
export LAYERLENS_ATLAS_PROJECT_ID="proj-your-project-here"

# Reload your shell configuration
source ~/.bashrc  # or ~/.zshrc
```

**Windows**:
```cmd
# Command Prompt (persistent)
setx LAYERLENS_ATLAS_API_KEY "sk-your-key-here"
setx LAYERLENS_ATLAS_ORG_ID "org-your-org-here"
setx LAYERLENS_ATLAS_PROJECT_ID "proj-your-project-here"

# PowerShell (session-only)
$env:LAYERLENS_ATLAS_API_KEY="sk-your-key-here"
$env:LAYERLENS_ATLAS_ORG_ID="org-your-org-here"
$env:LAYERLENS_ATLAS_PROJECT_ID="proj-your-project-here"
```

### Using .env Files

**Create a .env file** (never commit this to version control):
```bash
# .env
LAYERLENS_ATLAS_API_KEY=sk-your-key-here
LAYERLENS_ATLAS_ORG_ID=org-your-org-here
LAYERLENS_ATLAS_PROJECT_ID=proj-your-project-here
```

**Load .env file in Python**:
```python
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

from atlas import Atlas

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

### Advanced Credential Management

#### Using External Secret Managers

**AWS Secrets Manager**:
```python
import boto3
import json
from atlas import Atlas

def get_atlas_credentials_from_aws():
    """Retrieve Atlas credentials from AWS Secrets Manager"""
    session = boto3.session.Session()
    client = session.client('secretsmanager', region_name='us-east-1')
    
    try:
        response = client.get_secret_value(SecretId='layerlens/atlas/credentials')
        secrets = json.loads(response['SecretString'])
        
        return {
            'api_key': secrets['api_key'],
            'organization_id': secrets['organization_id'],
            'project_id': secrets['project_id']
        }
    except Exception as e:
        print(f"Error retrieving secrets: {e}")
        return None

# Usage
credentials = get_atlas_credentials_from_aws()
if credentials:
    client = Atlas(**credentials)
```
## Environment-Specific Key Management

### Separating Development and Production Keys

**Use different API keys for different environments**:

```python
import os
from atlas import Atlas

def get_atlas_client():
    """Get Atlas client based on environment"""
    environment = os.getenv('ATLAS_ENV', 'development')
    
    if environment == 'development':
        return Atlas(
            api_key=os.getenv('DEV_ATLAS_API_KEY'),
            organization_id=os.getenv('DEV_ATLAS_ORG_ID'),
            project_id=os.getenv('DEV_ATLAS_PROJECT_ID'),
            base_url=os.getenv('DEV_ATLAS_BASE_URL')  # Dev server if applicable
        )
    elif environment == 'staging':
        return Atlas(
            api_key=os.getenv('STAGING_ATLAS_API_KEY'),
            organization_id=os.getenv('STAGING_ATLAS_ORG_ID'),
            project_id=os.getenv('STAGING_ATLAS_PROJECT_ID')
        )
    elif environment == 'production':
        return Atlas(
            api_key=os.getenv('PROD_ATLAS_API_KEY'),
            organization_id=os.getenv('PROD_ATLAS_ORG_ID'),
            project_id=os.getenv('PROD_ATLAS_PROJECT_ID')
        )
    else:
        raise ValueError(f"Unknown environment: {environment}")

# Usage
client = get_atlas_client()
```

**Environment-specific .env files**:
```bash
# .env.development
DEV_ATLAS_API_KEY=sk-dev-key-here
DEV_ATLAS_ORG_ID=dev-org-id
DEV_ATLAS_PROJECT_ID=dev-project-id
DEV_ATLAS_BASE_URL=https://dev-api.layerlens.com

# .env.production
PROD_ATLAS_API_KEY=sk-prod-key-here
PROD_ATLAS_ORG_ID=prod-org-id
PROD_ATLAS_PROJECT_ID=prod-project-id
```

### Container and Deployment Security

**Docker Secrets**:
```yaml
# docker-compose.yml
version: '3.8'

services:
  atlas-app:
    image: your-app:latest
    secrets:
      - atlas_api_key
      - atlas_org_id
      - atlas_project_id
    environment:
      - LAYERLENS_ATLAS_API_KEY_FILE=/run/secrets/atlas_api_key
      - LAYERLENS_ATLAS_ORG_ID_FILE=/run/secrets/atlas_org_id
      - LAYERLENS_ATLAS_PROJECT_ID_FILE=/run/secrets/atlas_project_id

secrets:
  atlas_api_key:
    file: ./secrets/atlas_api_key.txt
  atlas_org_id:
    file: ./secrets/atlas_org_id.txt
  atlas_project_id:
    file: ./secrets/atlas_project_id.txt
```

**Reading Docker secrets in Python**:
```python
import os
from atlas import Atlas

def read_docker_secret(secret_name):
    """Read secret from Docker secrets file"""
    secret_file = f"/run/secrets/{secret_name}"
    try:
        with open(secret_file, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return None

def get_atlas_client_from_docker_secrets():
    """Initialize Atlas client using Docker secrets"""
    # Try Docker secrets first, fall back to environment variables
    api_key = (read_docker_secret('atlas_api_key') or 
               os.getenv('LAYERLENS_ATLAS_API_KEY'))
    
    org_id = (read_docker_secret('atlas_org_id') or 
              os.getenv('LAYERLENS_ATLAS_ORG_ID'))
    
    project_id = (read_docker_secret('atlas_project_id') or 
                  os.getenv('LAYERLENS_ATLAS_PROJECT_ID'))
    
    if not all([api_key, org_id, project_id]):
        raise ValueError("Missing required Atlas credentials")
    
    return Atlas(
        api_key=api_key,
        organization_id=org_id,
        project_id=project_id
    )

# Usage
client = get_atlas_client_from_docker_secrets()
```


## Security Checklist

### Development Security Checklist

- [ ] ✅ API keys stored in environment variables, not hardcoded
- [ ] ✅ `.env` files added to `.gitignore`
- [ ] ✅ Different API keys for development, staging, and production
- [ ] ✅ API key validation implemented before deployment
- [ ] ✅ Error handling doesn't expose API keys in logs
- [ ] ✅ Code review process includes credential security checks

### Production Security Checklist

- [ ] ✅ API keys stored in secure credential management system
- [ ] ✅ Key rotation schedule established and automated
- [ ] ✅ API usage monitoring and alerting configured
- [ ] ✅ Audit logging enabled for all API operations
- [ ] ✅ Network security controls (firewalls, VPNs) in place
- [ ] ✅ Least privilege access principles applied
- [ ] ✅ Incident response plan includes credential compromise scenarios
