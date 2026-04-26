"""Vertex AI authentication helpers.

The LayerLens Vertex adapter does **not** mint Google credentials itself.
It defers to Google's standard authentication chain so the adapter
behaves identically to a non-instrumented Vertex client. This module
documents the supported credential sources and provides small helper
functions used by the sample / docs.

Two credential sources are supported (in priority order):

1. **Service Account JSON** — set ``GOOGLE_APPLICATION_CREDENTIALS`` to
   the absolute path of an SA-JSON key file. The
   ``google-cloud-aiplatform`` SDK picks this up automatically.
2. **Application Default Credentials (ADC)** — gcloud user credentials
   from ``gcloud auth application-default login`` or the metadata-server
   identity attached to a GCE / GKE / Cloud Run workload.

Project / region selection follows the standard env vars used by the
SDK: ``GOOGLE_CLOUD_PROJECT`` and ``GOOGLE_CLOUD_REGION`` (or
``GOOGLE_CLOUD_LOCATION``). The adapter records both as event metadata
when present so traces are correlatable with billing dashboards.
"""

from __future__ import annotations

import os
from typing import Optional

_PROJECT_ENV_VARS = ("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT", "GCP_PROJECT")
_LOCATION_ENV_VARS = ("GOOGLE_CLOUD_REGION", "GOOGLE_CLOUD_LOCATION", "VERTEX_LOCATION")
_SA_JSON_ENV_VAR = "GOOGLE_APPLICATION_CREDENTIALS"


def detect_project_id() -> Optional[str]:
    """Return the configured Google Cloud project id, if any.

    Reads (in order) the ``GOOGLE_CLOUD_PROJECT``, ``GCLOUD_PROJECT``,
    and ``GCP_PROJECT`` environment variables. Returns ``None`` if
    none is set — the SDK will then fall back to the project embedded
    in the active credentials.
    """
    for name in _PROJECT_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return value
    return None


def detect_location() -> Optional[str]:
    """Return the configured Vertex location, if any.

    Reads ``GOOGLE_CLOUD_REGION``, ``GOOGLE_CLOUD_LOCATION``, and
    ``VERTEX_LOCATION`` (in that order). The Vertex SDK defaults to
    ``us-central1`` when no value is provided.
    """
    for name in _LOCATION_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return value
    return None


def detect_credential_source() -> str:
    """Describe which credential source the SDK will use.

    Returns:
        One of ``"service_account_json"`` (when
        ``GOOGLE_APPLICATION_CREDENTIALS`` is set and points at an
        existing file), ``"application_default"`` (otherwise), or
        ``"unknown"`` if the path is set but the file is missing.
    """
    sa_path = os.environ.get(_SA_JSON_ENV_VAR)
    if sa_path:
        if os.path.isfile(sa_path):
            return "service_account_json"
        return "unknown"
    return "application_default"
