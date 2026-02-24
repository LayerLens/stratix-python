# Stratix Python API library

The Stratix Python library provides convenient access to the Stratix REST API from any Python 3.8+ application. The library offers synchronous clients powered by [httpx](https://github.com/encode/httpx).

## Installation

```sh
# Install from PyPI
pip install stratix --index-url https://sdk.layerlens.ai
```

## Usage

```python
import os
from layerlens import Stratix

client = Stratix(
    # This is the default and can be omitted
    api_key=os.environ.get("LAYERLENS_STRATIX_API_KEY"),
)

evaluation = client.evaluations.create(
    model="random-model",
    benchmark="random-benchmark",
)

print(response.output_text)
```

## Handling errors

When the library is unable to connect to the API (for example, due to network connection problems or a timeout), a subclass of `layerlens.APIConnectionError` is raised.

When the API returns a non-success status code (that is, 4xx or 5xx response), a subclass of `layerlens.APIStatusError` is raised, containing `status_code` and `response` properties.

All errors inherit from `layerlens.APIError`.

```python
import layerlens
from layerlens import Stratix

client = Stratix()

try:
    client.evaluations.create(
        model="random-model",
        benchmark="random-benchmark",
    )
except layerlens.APIConnectionError as e:
    print("The server could not be reached")
    print(e.__cause__)  # an underlying Exception, likely raised within httpx.
except layerlens.RateLimitError as e:
    print("A 429 status code was received; we should back off a bit.")
except layerlens.APIStatusError as e:
    print("Another non-200-range status code was received")
    print(e.status_code)
    print(e.response)
```

Error codes are as follows:

| Status Code | Error Type                 |
| ----------- | -------------------------- |
| 400         | `BadRequestError`          |
| 401         | `AuthenticationError`      |
| 403         | `PermissionDeniedError`    |
| 404         | `NotFoundError`            |
| 422         | `UnprocessableEntityError` |
| 429         | `RateLimitError`           |
| >=500       | `InternalServerError`      |
| N/A         | `APIConnectionError`       |

## Timeouts

By default requests time out after 10 minutes. You can configure this with a `timeout` option, which accepts a float or an [`httpx.Timeout`](https://www.python-httpx.org/advanced/timeouts/#fine-tuning-the-configuration) object:

```python
from layerlens import Stratix

# Configure the default for all requests:
client = Stratix(
    # 20 seconds (default is 10 minutes)
    timeout=20.0,
)

# More granular control:
client = Stratix(
    timeout=httpx.Timeout(60.0, read=5.0, write=10.0, connect=2.0),
)

# Override per-request:
client.with_options(timeout=5.0).evaluations.create(
    model="random-model",
    benchmark="random-benchmark",
)
```

On timeout, an `APITimeoutError` is thrown.

## Versioning

This package follows [SemVer](https://semver.org/spec/v2.0.0.html) conventions.

### Determining the installed version

If you've upgraded to the latest version but aren't seeing any new features you were expecting then your python environment is likely still using an older version.

You can determine the version that is being used at runtime with:

```py
import layerlens
print(layerlens.__version__)
```

## Requirements

Python 3.8 or higher.

## Contributing

See [the contributing documentation](./CONTRIBUTING.md).
