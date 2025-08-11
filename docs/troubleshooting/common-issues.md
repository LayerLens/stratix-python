# Common Issues

This guide covers the most frequently encountered issues when using the Atlas Python SDK and provides step-by-step solutions.

## Installation Issues

### Package Not Found

**Problem**: `pip install atlas` fails with "No matching distribution found"

**Solutions**:

1. **Check Python version compatibility**:

   ```bash
   python --version
   # Atlas requires Python 3.8+
   ```

2. **Update pip and try again**:

   ```bash
   python -m pip install --upgrade pip
   pip install atlas
   ```

3. **Use Python 3 explicitly**:
   ```bash
   python3 -m pip install atlas
   ```

## Configuration Issues

### Missing Environment Variables

**Problem**: `AtlasError: The api_key client option must be set`

**Diagnosis**:

```python
import os
print(f"API Key: {os.getenv('LAYERLENS_ATLAS_API_KEY', 'NOT SET')}")
```

**Solutions**:

1. **Set environment variables**:

   ```bash
   # Linux/macOS
   export LAYERLENS_ATLAS_API_KEY="your_api_key_here"

   # Windows
   set LAYERLENS_ATLAS_API_KEY=your_api_key_here
   ```

2. **Use .env file**:

   ```bash
   # Create .env file
   LAYERLENS_ATLAS_API_KEY=your_api_key_here
   ```

   ```python
   from dotenv import load_dotenv
   load_dotenv()

   from atlas import Atlas
   client = Atlas()
   ```

3. **Pass explicitly to client**:

   ```python
   from atlas import Atlas

   client = Atlas(api_key="your_api_key_here")
   ```

### Where to Get Help

1. **LayerLens Support**: Contact support through your LayerLens dashboard for technical issues
2. **Documentation**: Check the [complete documentation](../README.md)
3. **Community**: Join LayerLens community channels for discussions

### Creating a Good Bug Report

Include this information when reporting issues:

1. **Environment details** (from debug info above)
2. **Complete error message** with stack trace
3. **Minimal reproducible example**:

   ```python
   from atlas import Atlas

   client = Atlas()

   # Minimal code that demonstrates the problem
   evaluation = client.evaluations.create(
       model="gpt-4",
       benchmark="mmlu"
   )
   ```

4. **Expected vs actual behavior**
5. **Steps to reproduce**
6. **Workarounds attempted**
