# Authentication Problems

This guide covers common authentication-related issues when using the Layerlens python sdk.

## Common Authentication Errors

### Invalid or Missing API Key

**Error**: `AuthenticationError: Invalid API key`

**Symptoms**:

- 401 Unauthorized responses
- "Invalid API key" error messages
- Authentication fails immediately

### Missing Required Configuration

**Error**: `AtlasError: The api_key client option must be set either by passing api_key to the client or by setting the LAYERLENS_ATLAS_API_KEY environment variable`