"""Tests for AgentForce Salesforce authentication, adapter, and error types.

Ported from ``ateam/tests/adapters/agentforce/test_auth.py``.

Source-of-truth API parity: the agentforce auth module exports the same
public surface in ``layerlens.instrument.adapters.frameworks.agentforce``
as it did under ``stratix.sdk.python.adapters.agentforce``. The only
behavioural delta is the new BaseAdapter multi-tenancy contract — every
``AgentForceAdapter`` constructor here passes ``org_id="test-org"``.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from layerlens.instrument.adapters._base.adapter import AdapterStatus
from layerlens.instrument.adapters.frameworks.agentforce.auth import (
    NormalizationError,
    SalesforceAuthError,
    SalesforceConnection,
    SalesforceQueryError,
    SalesforceCredentials,
)
from layerlens.instrument.adapters.frameworks.agentforce.adapter import (
    AgentForceAdapter,
)

_TEST_ORG_ID = "test-org"


# ---------------------------------------------------------------------------
# SalesforceCredentials
# ---------------------------------------------------------------------------


class TestSalesforceCredentialsMasking:
    """Tests for credential masking and key resolution."""

    def test_repr_masks_private_key(self) -> None:
        """Sensitive fields are masked in repr."""
        creds = SalesforceCredentials(
            client_id="my_client_id_12345",
            username="admin@example.com",
            private_key="-----BEGIN RSA PRIVATE KEY-----\nSECRET...",
            access_token="bearer_token_xyz",
        )
        r = repr(creds)
        assert "***REDACTED***" in r
        assert "PRIVATE KEY" not in r
        assert "bearer_token_xyz" not in r
        assert "admin@example.com" in r
        assert "my_clien..." in r  # client_id truncated

    def test_repr_shows_none_when_no_token(self) -> None:
        creds = SalesforceCredentials(
            client_id="test",
            username="u",
            private_key="k",
        )
        r = repr(creds)
        assert "access_token=None" in r

    def test_resolve_private_key_raw_pem(self) -> None:
        """Raw PEM string is returned as-is."""
        creds = SalesforceCredentials(
            client_id="c",
            username="u",
            private_key="-----BEGIN RSA PRIVATE KEY-----\ndata\n-----END RSA PRIVATE KEY-----",
        )
        assert creds.resolve_private_key().startswith("-----BEGIN RSA PRIVATE KEY-----")

    def test_resolve_private_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Private key resolved from environment variable."""
        monkeypatch.setenv("SF_PRIVATE_KEY", "resolved-key-data")
        creds = SalesforceCredentials(
            client_id="c",
            username="u",
            private_key="$SF_PRIVATE_KEY",
        )
        assert creds.resolve_private_key() == "resolved-key-data"

    def test_resolve_private_key_env_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Private key resolved from env: prefix."""
        monkeypatch.setenv("MY_KEY", "env-key")
        creds = SalesforceCredentials(
            client_id="c",
            username="u",
            private_key="env:MY_KEY",
        )
        assert creds.resolve_private_key() == "env-key"

    def test_resolve_private_key_missing_env_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing env var raises SalesforceAuthError."""
        monkeypatch.delenv("NONEXISTENT_KEY", raising=False)
        creds = SalesforceCredentials(
            client_id="c",
            username="u",
            private_key="$NONEXISTENT_KEY",
        )
        with pytest.raises(SalesforceAuthError, match="not set"):
            creds.resolve_private_key()

    def test_resolve_private_key_from_file(self, tmp_path: Any) -> None:
        """Private key resolved from file path."""
        key_file = tmp_path / "key.pem"
        key_file.write_text("file-key-data")
        creds = SalesforceCredentials(
            client_id="c",
            username="u",
            private_key=str(key_file),
        )
        assert creds.resolve_private_key() == "file-key-data"


# ---------------------------------------------------------------------------
# Structured Error Types
# ---------------------------------------------------------------------------


class TestStructuredErrors:
    """Tests for SalesforceAuthError, SalesforceQueryError, NormalizationError."""

    def test_auth_error_has_status_code(self) -> None:
        err = SalesforceAuthError("failed", status_code=401, endpoint="/token")
        assert err.status_code == 401
        assert err.endpoint == "/token"
        assert "failed" in str(err)

    def test_query_error_has_soql(self) -> None:
        err = SalesforceQueryError("timeout", status_code=500, soql="SELECT Id FROM X")
        assert err.status_code == 500
        assert err.soql == "SELECT Id FROM X"

    def test_normalization_error(self) -> None:
        err = NormalizationError("bad record")
        assert "bad record" in str(err)


# ---------------------------------------------------------------------------
# SalesforceConnection.authenticate
# ---------------------------------------------------------------------------


class TestSalesforceConnectionAuth:
    """Tests for JWT Bearer authentication flow."""

    def _make_connection(self, **overrides: Any) -> SalesforceConnection:
        creds = SalesforceCredentials(
            client_id="test_client",
            username="admin@test.com",
            private_key="fake-pem-key",
        )
        defaults: dict[str, Any] = {"credentials": creds, "max_retries": 1}
        defaults.update(overrides)
        return SalesforceConnection(**defaults)

    def test_authenticate_success(self) -> None:
        """Successful authentication stores access token."""
        conn = self._make_connection()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "access-xyz",
            "instance_url": "https://myorg.my.salesforce.com",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("jwt.encode", return_value="jwt-token-123"):  # noqa: SIM117
            with patch("requests.post", return_value=mock_response):
                conn.authenticate()

        assert conn.credentials.access_token == "access-xyz"
        assert conn.instance_url == "https://myorg.my.salesforce.com"
        assert not conn.credentials.is_expired

    def test_authenticate_http_401_raises_auth_error(self) -> None:
        """401 response raises SalesforceAuthError without retry."""
        import requests

        conn = self._make_connection(max_retries=3)

        mock_response = MagicMock()
        mock_response.status_code = 401
        http_error = requests.exceptions.HTTPError(response=mock_response)
        mock_response.raise_for_status.side_effect = http_error

        with patch("jwt.encode", return_value="token"):  # noqa: SIM117
            with patch("requests.post", return_value=mock_response):
                with pytest.raises(SalesforceAuthError, match="HTTP 401"):
                    conn.authenticate()

    def test_authenticate_timeout_retries(self) -> None:
        """Timeout triggers retry up to max_retries."""
        import requests

        conn = self._make_connection(max_retries=2)

        with patch("jwt.encode", return_value="token"):  # noqa: SIM117
            with patch("requests.post", side_effect=requests.exceptions.Timeout("timeout")):
                with patch("time.sleep"):
                    with pytest.raises(SalesforceAuthError, match="2 attempts"):
                        conn.authenticate()

    def test_authenticate_retry_succeeds_second_attempt(self) -> None:
        """Successful on second attempt after transient failure."""
        import requests

        conn = self._make_connection(max_retries=3)

        success_response = MagicMock()
        success_response.json.return_value = {
            "access_token": "success-token",
            "instance_url": "https://myorg.salesforce.com",
        }
        success_response.raise_for_status = MagicMock()

        call_count = 0

        def side_effect(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise requests.exceptions.Timeout("first try timeout")
            return success_response

        with patch("jwt.encode", return_value="token"):  # noqa: SIM117
            with patch("requests.post", side_effect=side_effect):
                with patch("time.sleep"):
                    conn.authenticate()

        assert conn.credentials.access_token == "success-token"
        assert call_count == 2


# ---------------------------------------------------------------------------
# SalesforceConnection.query
# ---------------------------------------------------------------------------


class TestSalesforceConnectionQuery:
    """Tests for SOQL query execution."""

    def _make_authenticated_connection(self, **overrides: Any) -> SalesforceConnection:
        creds = SalesforceCredentials(
            client_id="c",
            username="u",
            private_key="k",
            access_token="valid-token",
            token_expiry=time.time() + 3600,
        )
        defaults: dict[str, Any] = {
            "credentials": creds,
            "instance_url": "https://myorg.salesforce.com",
            "max_retries": 1,
        }
        defaults.update(overrides)
        return SalesforceConnection(**defaults)

    def test_query_returns_records(self) -> None:
        """Basic query returns records from response."""
        conn = self._make_authenticated_connection()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "records": [{"Id": "001"}, {"Id": "002"}],
            "done": True,
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            records = conn.query("SELECT Id FROM Account")

        assert len(records) == 2
        assert records[0]["Id"] == "001"

    def test_query_paginates(self) -> None:
        """Query follows nextRecordsUrl for pagination."""
        conn = self._make_authenticated_connection()

        page1 = MagicMock()
        page1.json.return_value = {
            "records": [{"Id": "001"}],
            "nextRecordsUrl": "/services/data/v60.0/query/next-123",
        }
        page1.raise_for_status = MagicMock()

        page2 = MagicMock()
        page2.json.return_value = {
            "records": [{"Id": "002"}],
        }
        page2.raise_for_status = MagicMock()

        call_count = 0

        def mock_get(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            return page1 if call_count == 1 else page2

        with patch("requests.get", side_effect=mock_get):
            records = conn.query("SELECT Id FROM Account")

        assert len(records) == 2
        assert call_count == 2

    def test_query_http_400_raises_query_error(self) -> None:
        """4xx client error raises SalesforceQueryError without retry."""
        import requests

        conn = self._make_authenticated_connection(max_retries=3)

        mock_response = MagicMock()
        mock_response.status_code = 400
        http_error = requests.exceptions.HTTPError(response=mock_response)
        mock_response.raise_for_status.side_effect = http_error

        with patch("requests.get", return_value=mock_response):  # noqa: SIM117
            with pytest.raises(SalesforceQueryError, match="HTTP 400"):
                conn.query("SELECT Bad FROM X")

    def test_query_timeout_retries(self) -> None:
        """Query retries on timeout."""
        import requests

        conn = self._make_authenticated_connection(max_retries=2)

        with patch("requests.get", side_effect=requests.exceptions.Timeout):  # noqa: SIM117
            with patch("time.sleep"):
                with pytest.raises(SalesforceQueryError, match="2 attempts"):
                    conn.query("SELECT Id FROM Account")

    def test_query_re_authenticates_on_expired_token(self) -> None:
        """Expired token triggers re-authentication before query."""
        creds = SalesforceCredentials(
            client_id="c",
            username="u",
            private_key="k",
            access_token="old-token",
            token_expiry=0.0,  # expired
        )
        conn = SalesforceConnection(
            credentials=creds,
            instance_url="https://myorg.salesforce.com",
            max_retries=1,
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {"records": [{"Id": "001"}]}
        mock_response.raise_for_status = MagicMock()

        with patch.object(conn, "authenticate") as mock_auth:
            # After authenticate, mark token as valid
            def fix_token() -> None:
                creds.access_token = "new-token"
                creds.token_expiry = time.time() + 3600

            mock_auth.side_effect = fix_token

            with patch("requests.get", return_value=mock_response):
                records = conn.query("SELECT Id FROM Account")

        mock_auth.assert_called_once()
        assert len(records) == 1


# ---------------------------------------------------------------------------
# Rate Limit Header Parsing
# ---------------------------------------------------------------------------


class TestRateLimitParsing:
    """Tests for Sforce-Limit-Info header parsing."""

    def test_parses_normal_header(self) -> None:
        """Normal api-usage header is parsed without error."""
        SalesforceConnection._check_rate_limit({"Sforce-Limit-Info": "api-usage=25/15000"})

    def test_warns_when_threshold_exceeded(self, caplog: pytest.LogCaptureFixture) -> None:
        """Warning logged when usage exceeds 80% threshold."""
        import logging

        with caplog.at_level(logging.WARNING):
            SalesforceConnection._check_rate_limit({"Sforce-Limit-Info": "api-usage=13000/15000"})
        assert "rate limit warning" in caplog.text.lower()
        assert "13000/15000" in caplog.text

    def test_no_warning_below_threshold(self, caplog: pytest.LogCaptureFixture) -> None:
        """No warning when usage is below threshold."""
        import logging

        with caplog.at_level(logging.WARNING):
            SalesforceConnection._check_rate_limit({"Sforce-Limit-Info": "api-usage=100/15000"})
        assert "rate limit" not in caplog.text.lower()

    def test_handles_missing_header(self) -> None:
        """Missing header is silently ignored."""
        SalesforceConnection._check_rate_limit({})

    def test_handles_malformed_header(self) -> None:
        """Malformed header is silently ignored."""
        SalesforceConnection._check_rate_limit({"Sforce-Limit-Info": "garbage"})
        SalesforceConnection._check_rate_limit({"Sforce-Limit-Info": "api-usage=bad/data"})

    def test_query_response_checks_rate_limit(self) -> None:
        """Query method checks rate limit header on successful response."""
        creds = SalesforceCredentials(
            client_id="c",
            username="u",
            private_key="k",
            access_token="token",
            token_expiry=time.time() + 3600,
        )
        conn = SalesforceConnection(
            credentials=creds,
            instance_url="https://myorg.salesforce.com",
            max_retries=1,
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {"records": [], "done": True}
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {"Sforce-Limit-Info": "api-usage=50/15000"}

        with patch("requests.get", return_value=mock_response):  # noqa: SIM117
            with patch.object(SalesforceConnection, "_check_rate_limit") as mock_check:
                conn.query("SELECT Id FROM Account")
                mock_check.assert_called_once()


# ---------------------------------------------------------------------------
# AgentForceAdapter (BaseAdapter wrapper)
# ---------------------------------------------------------------------------


class _MockLayerLens:
    """Minimal fake LayerLens client used to satisfy ``BaseAdapter``.

    Provides ``org_id`` (via constructor arg, propagated through
    BaseAdapter's tenancy guard) and a no-op event recorder.
    """

    def __init__(self, org_id: str = _TEST_ORG_ID) -> None:
        self.org_id = org_id
        self.events: list[dict[str, Any]] = []

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append({"type": event_type, "payload": payload})

    def get_events(self, event_type: str | None = None) -> list[dict[str, Any]]:
        if event_type:
            return [e for e in self.events if e["type"] == event_type]
        return self.events


class TestAgentForceAdapter:
    """Tests for the BaseAdapter-compliant AgentForceAdapter."""

    def _mock_connection(self, query_results: dict[str, list[dict[str, Any]]] | None = None) -> MagicMock:
        conn = MagicMock(spec=SalesforceConnection)
        conn.query.return_value = []
        if query_results:

            def query_side_effect(soql: str) -> list[dict[str, Any]]:
                sorted_keys = sorted(query_results.keys(), key=len, reverse=True)
                for key in sorted_keys:
                    if f"FROM {key}" in soql:
                        return query_results[key]
                return []

            conn.query.side_effect = query_side_effect
        return conn

    def test_adapter_framework_info(self) -> None:
        adapter = AgentForceAdapter(org_id=_TEST_ORG_ID)
        info = adapter.get_adapter_info()
        assert info.framework == "salesforce_agentforce"
        assert info.name == "AgentForceAdapter"

    def test_connect_and_disconnect(self) -> None:
        conn = self._mock_connection()
        adapter = AgentForceAdapter(connection=conn, org_id=_TEST_ORG_ID)

        adapter.connect()
        assert adapter.is_connected
        assert adapter.status == AdapterStatus.HEALTHY

        adapter.disconnect()
        assert not adapter.is_connected
        assert adapter.status == AdapterStatus.DISCONNECTED

    def test_connect_requires_credentials_or_connection(self) -> None:
        adapter = AgentForceAdapter(org_id=_TEST_ORG_ID)
        with pytest.raises(SalesforceAuthError, match="credentials.*connection"):
            adapter.connect()

    def test_health_check(self) -> None:
        conn = self._mock_connection()
        adapter = AgentForceAdapter(connection=conn, org_id=_TEST_ORG_ID)
        adapter.connect()

        health = adapter.health_check()
        assert health.status == AdapterStatus.HEALTHY
        assert health.framework_name == "salesforce_agentforce"

    def test_import_routes_through_emit(self) -> None:
        """Import events go through BaseAdapter emit_dict_event for known event types."""
        stratix = _MockLayerLens()
        conn = self._mock_connection(
            {
                "AIAgentSession": [
                    {
                        "Id": "0Xx5f00000SeSS01Ab",
                        "StartTimestamp": "2026-02-21T10:00:00Z",
                        "EndTimestamp": "2026-02-21T10:15:00Z",
                        "AiAgentChannelTypeId": "Chat",
                        "AiAgentSessionEndType": "Resolved",
                        "VoiceCallId": None,
                        "MessagingSessionId": None,
                        "PreviousSessionId": None,
                    }
                ],
                "AIAgentSessionParticipant": [],
                "AIAgentInteraction": [
                    {
                        "Id": "0Xx5f00000InT001Ab",
                        "AiAgentSessionId": "0Xx5f00000SeSS01Ab",
                        "AiAgentInteractionTypeId": "Turn",
                        "TelemetryTraceId": "trace-1",
                        "TelemetryTraceSpanId": "span-1",
                        "TopicApiName": "Test",
                        "AttributeText": None,
                        "PrevInteractionId": None,
                    }
                ],
                "AIAgentInteractionStep": [
                    {
                        "Id": "0Xx5f00000StEP01Ab",
                        "AiAgentInteractionId": "0Xx5f00000InT001Ab",
                        "AiAgentInteractionStepTypeId": "LLMExecutionStep",
                        "InputValueText": "hello",
                        "OutputValueText": "world",
                        "ErrorMessageText": None,
                        "GenerationId": None,
                        "GenAiGatewayRequestId": None,
                        "GenAiGatewayResponseId": None,
                        "Name": "test_model",
                        "TelemetryTraceSpanId": "span-s1",
                    }
                ],
                "AIAgentInteractionMessage": [],
            }
        )
        adapter = AgentForceAdapter(stratix=stratix, connection=conn, org_id=_TEST_ORG_ID)
        adapter.connect()

        result = adapter.import_sessions()

        assert result.sessions_imported == 1
        # model.invoke is L3 (enabled by default) — should be emitted through STRATIX
        model_events = stratix.get_events("model.invoke")
        assert len(model_events) >= 1

    def test_import_not_connected_raises(self) -> None:
        adapter = AgentForceAdapter(org_id=_TEST_ORG_ID)
        with pytest.raises(RuntimeError, match="not connected"):
            adapter.import_sessions()

    def test_serialize_for_replay(self) -> None:
        conn = self._mock_connection()
        adapter = AgentForceAdapter(connection=conn, org_id=_TEST_ORG_ID)
        adapter.connect()

        trace = adapter.serialize_for_replay()
        assert trace.framework == "salesforce_agentforce"
        assert trace.trace_id
