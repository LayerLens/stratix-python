"""Unit tests for the presigned-upload SSRF guard (LAY-3635, F-L12-001)."""

import pytest

from layerlens._ssrf import ensure_safe_upload_url, resolve_trusted_upload_hosts
from layerlens._exceptions import StratixError


@pytest.mark.invariant
class TestEnsureSafeUploadURL:
    @pytest.mark.parametrize(
        "url",
        [
            "https://bucket.s3.amazonaws.com/key?sig=1",
            "https://my-bucket.s3.us-east-1.amazonaws.com/x",
            "https://layerlens-private-abc.s3.amazonaws.com/traces/x.json?X-Amz-Signature=z",
        ],
    )
    def test_allows_public_https(self, url):
        ensure_safe_upload_url(url, None)  # must not raise

    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost/steal",
            "https://localhost/steal",
            "http://127.0.0.1/x",
            "https://127.0.0.1/x",
            "http://10.0.0.1/x",
            "https://10.0.0.1/x",
            "https://192.168.1.5/x",
            "https://169.254.169.254/x",
            "http://169.254.169.254/latest/meta-data/iam/",
            "http://s3.example.com/x",  # public but non-https → rejected by default
        ],
    )
    def test_rejects_unsafe(self, url):
        with pytest.raises(StratixError):
            ensure_safe_upload_url(url, None)

    def test_allowlist_reenables_self_host(self):
        # MinIO (http + private host) opts back in by listing its host.
        ensure_safe_upload_url("http://minio:9000/bucket/key", ["minio"])
        ensure_safe_upload_url("https://10.0.0.1/x", ["10.0.0.1"])
        ensure_safe_upload_url("http://localhost:9000/x", ["localhost"])

    def test_metadata_blocked_even_if_trusted(self):
        # Cloud metadata is never a legitimate target, allowlist or not.
        with pytest.raises(StratixError):
            ensure_safe_upload_url("http://169.254.169.254/x", ["169.254.169.254"])

    def test_non_list_trusted_is_coerced_not_crashed(self):
        # A Mock-like / garbage trusted arg must be treated as empty, not crash.
        ensure_safe_upload_url("https://s3.amazonaws.com/x", object())
        with pytest.raises(StratixError):
            ensure_safe_upload_url("http://10.0.0.1/x", object())

    def test_no_host_rejected(self):
        with pytest.raises(StratixError):
            ensure_safe_upload_url("not-a-url", None)


class TestResolveTrustedUploadHosts:
    def test_empty_default(self, monkeypatch):
        monkeypatch.delenv("LAYERLENS_TRUSTED_UPLOAD_HOSTS", raising=False)
        assert resolve_trusted_upload_hosts(None) == []

    def test_merges_option_and_env(self, monkeypatch):
        monkeypatch.setenv("LAYERLENS_TRUSTED_UPLOAD_HOSTS", "Minio, host2 ,")
        got = resolve_trusted_upload_hosts(["Host3", "minio"])
        assert got == ["host3", "minio", "host2"]  # lower-cased, de-duped, order-preserving
