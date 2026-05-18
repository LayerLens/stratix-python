# Security Policy

We take the security of the Stratix Python SDK seriously. Thanks for helping us keep it safe.

## Reporting a vulnerability

**Do not file a public GitHub issue for security vulnerabilities.**

Email **support@layerlens.ai** with the subject line "Security report: stratix-python" and include:

- A description of the vulnerability and where it lives in the codebase.
- Steps to reproduce, including any proof-of-concept code if you have it.
- The version of `layerlens` you tested against (`pip show layerlens`).
- Your assessment of the impact (data exposure, RCE, auth bypass, denial of service, etc.).
- Whether you would like credit in the disclosure, and if so, how you would like to be credited.

We will acknowledge receipt within 3 business days, give you an initial assessment within 7 business days, and keep you updated as we work on a fix.

## Scope

In scope:

- The `layerlens` Python package published to PyPI.
- Source code in this repository (`src/`, `tests/`, `samples/`, `scripts/`).
- The `stratix` CLI binary distributed with the SDK.

Out of scope (please report to the relevant team instead):

- Vulnerabilities in the hosted Stratix platform itself ([stratix.layerlens.ai](https://stratix.layerlens.ai)). Email **support@layerlens.ai** with subject "Security report: Stratix platform."
- Third-party dependencies. Please file with the upstream project.
- Issues that require physical access to a user's machine.

## Supported versions

We provide security fixes for the latest minor release of `layerlens`. Older versions may receive fixes at our discretion.

| Version | Supported          |
| ------- | ------------------ |
| 1.6.x   | Yes                |
| < 1.6   | No, please upgrade |

## Disclosure

We follow coordinated disclosure. Once a fix is released, we will publish an advisory on the [GitHub Security Advisories](https://github.com/LayerLens/stratix-python/security/advisories) page and credit the reporter unless they prefer to remain anonymous.

Thanks for keeping the community safe.
