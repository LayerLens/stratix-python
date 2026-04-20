from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass

from ._utils import truncate
from ._base_framework import FrameworkAdapter
from ..._capture_config import CaptureConfig

log = logging.getLogger(__name__)

try:
    import httpx  # pyright: ignore[reportMissingImports]

    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

_SF_API_VERSION = "v62.0"

_SOQL_SESSIONS = (
    "SELECT Id, Name, StartTime, EndTime, Status, AgentId, AgentName, "
    "ParticipantId, ParticipantName, Channel, Outcome "
    "FROM AIAgentSession__dlm "
    "{where_clause} "
    "ORDER BY StartTime DESC "
    "{limit_clause}"
)

_SOQL_INTERACTIONS = (
    "SELECT Id, SessionId, StepType, StepName, Sequence, StartTime, EndTime, "
    "Input, Output, ModelName, PromptTokens, CompletionTokens, "
    "ToolName, ToolInput, ToolOutput, EscalationTarget, ErrorMessage "
    "FROM AIAgentInteraction__dlm "
    "WHERE SessionId = '{session_id}' "
    "ORDER BY Sequence ASC"
)

_SOQL_AGENT_CONFIG = (
    "SELECT Id, AgentId, AgentName, Description, ModelName, "
    "Instructions, TopicCount, ActionCount "
    "FROM AIAgentConfiguration__dlm "
    "WHERE AgentId = '{agent_id}' "
    "LIMIT 1"
)

_STEP_DISPATCH = {
    "llm": "_on_llm_step",
    "model": "_on_llm_step",
    "generative": "_on_llm_step",
    "action": "_on_tool_step",
    "function": "_on_tool_step",
    "tool": "_on_tool_step",
    "flow": "_on_tool_step",
    "escalation": "_on_handoff_step",
    "handoff": "_on_handoff_step",
    "transfer": "_on_handoff_step",
}


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _int_or_zero(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _sf_datetime(date_str: str) -> str:
    try:
        dt = datetime.fromisoformat(date_str)
    except ValueError:
        return date_str
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------
# Salesforce connection helpers
# ------------------------------------------------------------------


@dataclass
class _SalesforceCredentials:
    client_id: str
    client_secret: str
    instance_url: str
    access_token: Optional[str] = None
    token_url: Optional[str] = None

    def __post_init__(self) -> None:
        self.instance_url = self.instance_url.rstrip("/")
        if not self.token_url:
            self.token_url = f"{self.instance_url}/services/oauth2/token"


class _SalesforceConnection:
    """Thin HTTP wrapper around the Salesforce REST API."""

    def __init__(self, credentials: _SalesforceCredentials) -> None:
        self._creds = credentials
        self._http: Any = None

    def authenticate(self) -> None:
        self._http = httpx.Client(timeout=30.0)
        resp = self._http.post(
            self._creds.token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": self._creds.client_id,
                "client_secret": self._creds.client_secret,
            },
        )
        resp.raise_for_status()
        body = resp.json()
        self._creds.access_token = body["access_token"]
        if "instance_url" in body:
            self._creds.instance_url = body["instance_url"].rstrip("/")

    def query(self, soql: str) -> List[Dict[str, Any]]:
        if self._http is None or self._creds.access_token is None:
            raise RuntimeError("Not authenticated — call authenticate() first")
        url = f"{self._creds.instance_url}/services/data/{_SF_API_VERSION}/query/"
        headers = {"Authorization": f"Bearer {self._creds.access_token}"}
        records: List[Dict[str, Any]] = []
        resp = self._http.get(url, params={"q": soql}, headers=headers)
        resp.raise_for_status()
        body = resp.json()
        records.extend(body.get("records", []))
        while body.get("nextRecordsUrl"):
            next_url = f"{self._creds.instance_url}{body['nextRecordsUrl']}"
            resp = self._http.get(next_url, headers=headers)
            resp.raise_for_status()
            body = resp.json()
            records.extend(body.get("records", []))
        return records

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
            self._http = None


class AgentforceAdapter(FrameworkAdapter):
    """Salesforce Agentforce adapter — batch import from Data Cloud DMOs.

    Connects to Salesforce via OAuth, queries ``AIAgentSession`` and
    ``AIAgentInteraction`` objects, and emits normalised events.
    Each session is a separate trace via ``_begin_run`` / ``_end_run``.

    Usage::

        adapter = AgentforceAdapter(client)
        adapter.connect(
            credentials={
                "client_id": "...",
                "client_secret": "...",
                "instance_url": "https://myorg.my.salesforce.com",
            },
        )
        summary = adapter.import_sessions(limit=50)
        adapter.disconnect()
    """

    name = "agentforce"
    package = "httpx"

    def __init__(self, client: Any, capture_config: Optional[CaptureConfig] = None) -> None:
        super().__init__(client, capture_config)
        self._connection: Optional[_SalesforceConnection] = None
        self._credentials: Optional[_SalesforceCredentials] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _on_connect(self, target: Any = None, **kwargs: Any) -> None:
        self._check_dependency(_HAS_HTTPX)
        credentials = kwargs.get("credentials")
        instance_url = kwargs.get("instance_url")

        if credentials is None:
            raise ValueError(
                "Salesforce credentials are required. Pass a dict with "
                "'client_id', 'client_secret', and 'instance_url'."
            )

        creds = _SalesforceCredentials(
            client_id=credentials["client_id"],
            client_secret=credentials["client_secret"],
            instance_url=instance_url or credentials.get("instance_url", ""),
        )
        if not creds.instance_url:
            raise ValueError("instance_url is required in credentials or as a keyword argument")

        conn = _SalesforceConnection(creds)
        try:
            conn.authenticate()
        except Exception:
            conn.close()
            raise

        self._credentials = creds
        self._connection = conn
        if creds.instance_url:
            self._metadata["instance_url"] = creds.instance_url

    def _on_disconnect(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        self._credentials = None

    # ------------------------------------------------------------------
    # Batch import
    # ------------------------------------------------------------------

    def import_sessions(
        self,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
        since_cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Incrementally import Agentforce sessions.

        ``since_cursor`` — when provided, only sessions whose ``StartTime``
        strictly exceeds the cursor are imported. On return, the summary
        includes a ``next_cursor`` set to the max ``StartTime`` seen so the
        caller can persist it and pass it into the next run for exactly-once
        incremental sync.
        """
        conn = self._connection
        if conn is None or not self._connected:
            raise RuntimeError("Adapter is not connected — call connect() first")

        where_parts: List[str] = []
        if since_cursor:
            where_parts.append(f"StartTime > {_sf_datetime(since_cursor)}")
        if start_date:
            where_parts.append(f"StartTime >= {_sf_datetime(start_date)}")
        if end_date:
            where_parts.append(f"StartTime < {_sf_datetime(end_date)}")
        where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
        limit_clause = f"LIMIT {limit}" if limit else ""

        soql = _SOQL_SESSIONS.format(where_clause=where_clause, limit_clause=limit_clause)
        summary: Dict[str, Any] = {
            "sessions_imported": 0,
            "events_emitted": 0,
            "errors": 0,
            "next_cursor": since_cursor,
        }

        try:
            sessions = conn.query(soql)
        except Exception:
            log.error("layerlens: failed to query Agentforce sessions", exc_info=True)
            summary["errors"] += 1
            return summary

        max_cursor = since_cursor
        for session in sessions:
            try:
                emitted = self._import_session(conn, session)
                summary["sessions_imported"] += 1
                summary["events_emitted"] += emitted
                # Advance the cursor to the latest StartTime seen. StartTime
                # values are ISO-8601 so lexicographic comparison is correct.
                start_time = session.get("StartTime")
                if start_time and (max_cursor is None or str(start_time) > str(max_cursor)):
                    max_cursor = str(start_time)
            except Exception:
                log.warning("layerlens: error importing session %s", session.get("Id"), exc_info=True)
                summary["errors"] += 1

        if max_cursor is not None:
            summary["next_cursor"] = max_cursor
        return summary

    # ------------------------------------------------------------------
    # Per-session import
    # ------------------------------------------------------------------

    def _import_session(self, conn: _SalesforceConnection, session: Dict[str, Any]) -> int:
        session_id = session.get("Id", "")
        agent_id = session.get("AgentId", "")
        emitted = 0

        self._begin_run()
        try:
            root = self._get_root_span()

            # -- environment.config --
            emitted += self._emit_agent_config(conn, agent_id)

            # -- agent.input --
            payload = self._payload(
                session_id=session_id,
                agent_id=agent_id,
                agent_name=session.get("AgentName", ""),
                participant_id=session.get("ParticipantId", ""),
                participant_name=session.get("ParticipantName", ""),
                channel=session.get("Channel", ""),
                start_time=session.get("StartTime", ""),
            )
            self._emit("agent.input", payload, span_id=root, parent_span_id=None, span_name="session")
            emitted += 1

            # -- interaction steps --
            try:
                interactions = conn.query(_SOQL_INTERACTIONS.format(session_id=session_id))
            except Exception:
                log.warning("layerlens: failed to query interactions for %s", session_id, exc_info=True)
                interactions = []

            for step in interactions:
                emitted += self._process_step(step)

            # -- agent.output --
            out_payload = self._payload(
                session_id=session_id,
                status=session.get("Status", ""),
                outcome=session.get("Outcome", ""),
                end_time=session.get("EndTime", ""),
            )
            self._emit("agent.output", out_payload, span_name="session_end")
            emitted += 1

        finally:
            self._end_run()

        return emitted

    # ------------------------------------------------------------------
    # Step dispatch
    # ------------------------------------------------------------------

    def _process_step(self, step: Dict[str, Any]) -> int:
        step_type = (step.get("StepType") or "").lower()
        handler_name = _STEP_DISPATCH.get(step_type)
        if handler_name is not None:
            return getattr(self, handler_name)(step)

        # Unknown step type
        payload = self._payload(
            step_type=step.get("StepType", "unknown"),
            step_name=step.get("StepName", ""),
        )
        self._set_if_capturing(payload, "input", truncate(step.get("Input"), 4000))
        self._set_if_capturing(payload, "output", truncate(step.get("Output"), 4000))
        self._emit("agent.interaction", payload, span_name=step.get("StepName", "interaction"))
        return 1

    # ------------------------------------------------------------------
    # Step handlers
    # ------------------------------------------------------------------

    def _on_llm_step(self, step: Dict[str, Any]) -> int:
        prompt_tokens = _int_or_zero(step.get("PromptTokens"))
        completion_tokens = _int_or_zero(step.get("CompletionTokens"))
        model = step.get("ModelName", "")
        emitted = 0

        span_id = self._new_span_id()
        payload = self._payload(step_name=step.get("StepName", ""))
        if model:
            payload["model"] = model
        if prompt_tokens:
            payload["tokens_prompt"] = prompt_tokens
        if completion_tokens:
            payload["tokens_completion"] = completion_tokens
        if prompt_tokens or completion_tokens:
            payload["tokens_total"] = prompt_tokens + completion_tokens
        self._set_if_capturing(payload, "messages", truncate(step.get("Input"), 4000))
        self._set_if_capturing(payload, "output_message", truncate(step.get("Output"), 4000))
        self._emit("model.invoke", payload, span_id=span_id, span_name=step.get("StepName", "llm_call"))
        emitted += 1

        if prompt_tokens or completion_tokens:
            cost_payload = self._payload(
                tokens_prompt=prompt_tokens,
                tokens_completion=completion_tokens,
                tokens_total=prompt_tokens + completion_tokens,
            )
            if model:
                cost_payload["model"] = model
            self._emit("cost.record", cost_payload, span_id=span_id)
            emitted += 1

        return emitted

    def _on_tool_step(self, step: Dict[str, Any]) -> int:
        payload = self._payload(
            tool_name=step.get("ToolName") or step.get("StepName", "unknown"),
            step_type=step.get("StepType", ""),
        )
        self._set_if_capturing(payload, "input", truncate(step.get("ToolInput") or step.get("Input"), 4000))
        self._set_if_capturing(payload, "output", truncate(step.get("ToolOutput") or step.get("Output"), 4000))
        self._emit("tool.call", payload, span_name=step.get("ToolName") or step.get("StepName", "tool_call"))
        return 1

    def _on_handoff_step(self, step: Dict[str, Any]) -> int:
        payload = self._payload(
            step_name=step.get("StepName", ""),
            escalation_target=step.get("EscalationTarget", ""),
            error_message=step.get("ErrorMessage", ""),
        )
        self._set_if_capturing(payload, "reason", truncate(step.get("Input"), 4000))
        self._emit("agent.handoff", payload, span_name="handoff")
        return 1

    # ------------------------------------------------------------------
    # Agent config
    # ------------------------------------------------------------------

    def _emit_agent_config(self, conn: _SalesforceConnection, agent_id: str) -> int:
        if not agent_id:
            return 0
        try:
            records = conn.query(_SOQL_AGENT_CONFIG.format(agent_id=agent_id))
        except Exception:
            log.debug("layerlens: could not fetch agent config for %s", agent_id, exc_info=True)
            return 0
        if not records:
            return 0
        cfg = records[0]
        self._emit(
            "environment.config",
            self._payload(
                agent_id=agent_id,
                agent_name=cfg.get("AgentName", ""),
                description=cfg.get("Description", ""),
                model=cfg.get("ModelName", ""),
                instructions=cfg.get("Instructions", ""),
                topic_count=cfg.get("TopicCount"),
                action_count=cfg.get("ActionCount"),
            ),
            span_name="agent_config",
        )
        return 1
