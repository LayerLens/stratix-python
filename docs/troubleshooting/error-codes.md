# Error Codes Reference

This reference guide provides detailed information about all error codes and exceptions in the Layerlens Python SDK.

## Exception Hierarchy

```
AtlasError (Base exception)
├── APIError (Base for API-related errors)
│   ├── APIConnectionError (Network/connection issues)
│   │   └── APITimeoutError (Request timeouts)
│   ├── APIResponseValidationError (Invalid response format)
│   └── APIStatusError (HTTP status errors)
│       ├── BadRequestError (400)
│       ├── AuthenticationError (401)
│       ├── PermissionDeniedError (403)
│       ├── NotFoundError (404)
│       ├── ConflictError (409)
│       ├── UnprocessableEntityError (422)
│       ├── RateLimitError (429)
│       └── InternalServerError (500+)
```