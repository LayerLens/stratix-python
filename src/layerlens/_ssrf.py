"""SSRF guard for presigned-upload URLs.

The SDK PUTs trace/benchmark bytes to whatever host the presigned-URL response
names. A compromised or malicious server (or a MITM on the presign request)
could redirect that PUT to an attacker-controlled host (data exfil) or an
internal SSRF target (cloud metadata, internal services). ``ensure_safe_upload_url``
is called before every presign PUT.

Default policy (no allowlist): require ``https`` and reject loopback / private /
link-local addresses and ``localhost``. Cloud-metadata endpoints are *always*
rejected. Self-hosted / MinIO deployments (http + a private host) opt back in by
listing their host in the client option ``trusted_upload_hosts`` or the
``LAYERLENS_TRUSTED_UPLOAD_HOSTS`` environment variable (comma-separated).
"""

from __future__ import annotations

import os
import ipaddress
from typing import List, Union, Iterable, Optional
from urllib.parse import urlparse

from ._exceptions import StratixError

# Endpoints that are never a legitimate upload target — blocked even if a caller
# (mis)configures them as trusted.
_ALWAYS_BLOCKED_HOSTS = frozenset(
    {
        "169.254.169.254",  # AWS/GCP/Azure IMDS (IPv4)
        "fd00:ec2::254",  # AWS IMDS (IPv6)
        "metadata.google.internal",  # GCP metadata hostname
    }
)

_LOCAL_HOSTNAMES = frozenset({"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"})

ENV_TRUSTED_UPLOAD_HOSTS = "LAYERLENS_TRUSTED_UPLOAD_HOSTS"


def resolve_trusted_upload_hosts(option: Optional[Iterable[str]] = None) -> List[str]:
    """Merge an explicit ``trusted_upload_hosts`` option with the env var.

    Returns a de-duplicated list of lower-cased hostnames. Order: option first,
    then env entries, preserving first occurrence.
    """
    hosts: List[str] = []
    if isinstance(option, (list, tuple, set)):
        hosts.extend(str(h).strip().lower() for h in option if str(h).strip())
    env = os.environ.get(ENV_TRUSTED_UPLOAD_HOSTS, "") or ""
    hosts.extend(h.strip().lower() for h in env.split(",") if h.strip())

    seen = set()
    out: List[str] = []
    for h in hosts:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


def _parse_ip(host: str) -> Optional[Union[ipaddress.IPv4Address, ipaddress.IPv6Address]]:
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def ensure_safe_upload_url(url: str, trusted_hosts: Optional[Iterable[str]] = None) -> None:
    """Raise ``StratixError`` if ``url`` is an unsafe presigned-upload target.

    ``trusted_hosts`` is the resolved allowlist; anything that is not a
    list/tuple/set is treated as empty (defensive — never crash the upload path).
    """
    if not isinstance(trusted_hosts, (list, tuple, set)):
        trusted_hosts = []
    trusted = {str(h).strip().lower() for h in trusted_hosts if str(h).strip()}

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    scheme = (parsed.scheme or "").lower()

    if not host:
        raise StratixError(f"refusing upload: presigned URL has no host ({url!r})")

    # 1. Cloud-metadata endpoints are never a legitimate upload target.
    if host in _ALWAYS_BLOCKED_HOSTS:
        raise StratixError(f"refusing upload to cloud-metadata endpoint {host!r}")

    # 2. Explicitly trusted hosts bypass the default scheme/IP policy
    #    (this is how self-hosted / MinIO http+private endpoints opt back in).
    if host in trusted:
        return

    # 3. Default policy for untrusted hosts.
    if scheme != "https":
        raise StratixError(
            f"refusing non-https upload to untrusted host {host!r} (scheme {scheme!r}); "
            f"add it to trusted_upload_hosts / {ENV_TRUSTED_UPLOAD_HOSTS} to allow self-hosted/MinIO"
        )
    if host in _LOCAL_HOSTNAMES:
        raise StratixError(f"refusing upload to local hostname {host!r}; add it to trusted_upload_hosts to allow")
    ip = _parse_ip(host)
    if ip is not None and (
        ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_unspecified or ip.is_multicast
    ):
        raise StratixError(f"refusing upload to non-public address {host!r}; add it to trusted_upload_hosts to allow")
