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

# Salesforce stores this literal string in place of null in the Session
# Tracing DMOs; treat it as empty everywhere.
_NOT_SET = "NOT_SET"


# ------------------------------------------------------------------
# Session Tracing Data Model (STDM) — object + field names
# ------------------------------------------------------------------
#
# Verified against a live ``describe`` of a provisioned Agentforce + Data
# Cloud org (LAY-3599). Centralised here so a future migration to the ateam
# Agentforce subsystem is a move, not a rewrite, and so a schema change is a
# one-place edit. The DMOs link on the **business UUID** carried in
# ``ssot__Id__c`` / ``ssot__<parent>Id__c`` — NOT the Salesforce surrogate
# ``Id`` (a message references its interaction by the interaction's
# ``ssot__Id__c``).


class _STDM:
    # Objects
    SESSION = "ssot__AiAgentSession__dlm"
    PARTICIPANT = "ssot__AiAgentSessionParticipant__dlm"
    INTERACTION = "ssot__AiAgentInteraction__dlm"
    STEP = "ssot__AiAgentInteractionStep__dlm"
    MESSAGE = "ssot__AiAgentInteractionMessage__dlm"
    MOMENT = "ssot__AiAgentMoment__dlm"

    # Shared — every DMO carries its own business UUID here.
    BIZ_ID = "ssot__Id__c"

    # Session
    S_CHANNEL = "ssot__AiAgentChannelType__c"
    S_END_TYPE = "ssot__AiAgentSessionEndType__c"
    S_START = "ssot__StartTimestamp__c"
    S_END = "ssot__EndTimestamp__c"

    # Participant — carries agent identity (live-populated; Moment is empty).
    P_SESSION_ID = "ssot__AiAgentSessionId__c"
    P_AGENT_API = "ssot__AiAgentApiName__c"
    P_AGENT_VERSION = "ssot__AiAgentVersionApiName__c"
    P_AGENT_TYPE = "ssot__AiAgentType__c"
    P_ROLE = "ssot__AiAgentSessionParticipantRole__c"
    P_PARTICIPANT_ID = "ssot__ParticipantId__c"

    # Interaction (a conversational TURN)
    I_SESSION_ID = "ssot__AiAgentSessionId__c"
    I_TYPE = "ssot__AiAgentInteractionType__c"
    I_TOPIC = "ssot__TopicApiName__c"
    I_TRACE_ID = "ssot__TelemetryTraceId__c"
    I_PREV = "ssot__PrevInteractionId__c"
    I_START = "ssot__StartTimestamp__c"
    I_END = "ssot__EndTimestamp__c"

    # Interaction step (the internal model / tool execution)
    ST_INTERACTION_ID = "ssot__AiAgentInteractionId__c"
    ST_TYPE = "ssot__AiAgentInteractionStepType__c"
    ST_SUBTYPE = "SubType__c"
    ST_NAME = "ssot__Name__c"
    ST_INPUT = "ssot__InputValueText__c"
    ST_OUTPUT = "ssot__OutputValueText__c"
    ST_GENERATION_ID = "ssot__GenerationId__c"
    ST_GW_REQUEST_ID = "ssot__GenAiGatewayRequestId__c"
    ST_GW_RESPONSE_ID = "ssot__GenAiGatewayResponseId__c"
    ST_ERROR = "ssot__ErrorMessageText__c"
    ST_START = "ssot__StartTimestamp__c"
    ST_END = "ssot__EndTimestamp__c"

    # Interaction message (the human-readable conversation turn)
    M_INTERACTION_ID = "ssot__AiAgentInteractionId__c"
    M_TYPE = "ssot__AiAgentInteractionMessageType__c"  # "Input" | "Output"
    M_CONTENT = "ssot__ContentText__c"
    M_SENT = "ssot__MessageSentTimestamp__c"


_SOQL_SESSIONS = (
    "SELECT ssot__Id__c, ssot__AiAgentChannelType__c, ssot__AiAgentSessionEndType__c, "
    "ssot__StartTimestamp__c, ssot__EndTimestamp__c "
    "FROM ssot__AiAgentSession__dlm "
    "{where_clause} "
    "ORDER BY ssot__StartTimestamp__c DESC "
    "{limit_clause}"
)

_SOQL_PARTICIPANTS = (
    "SELECT ssot__AiAgentApiName__c, ssot__AiAgentVersionApiName__c, ssot__AiAgentType__c, "
    "ssot__AiAgentSessionParticipantRole__c, ssot__ParticipantId__c "
    "FROM ssot__AiAgentSessionParticipant__dlm "
    "WHERE ssot__AiAgentSessionId__c = '{session_id}'"
)

_SOQL_INTERACTIONS = (
    "SELECT ssot__Id__c, ssot__AiAgentInteractionType__c, ssot__TopicApiName__c, "
    "ssot__TelemetryTraceId__c, ssot__PrevInteractionId__c, "
    "ssot__StartTimestamp__c, ssot__EndTimestamp__c "
    "FROM ssot__AiAgentInteraction__dlm "
    "WHERE ssot__AiAgentSessionId__c = '{session_id}' "
    "ORDER BY ssot__StartTimestamp__c ASC"
)

_SOQL_STEPS = (
    "SELECT ssot__Id__c, ssot__AiAgentInteractionStepType__c, SubType__c, ssot__Name__c, "
    "ssot__InputValueText__c, ssot__OutputValueText__c, ssot__GenerationId__c, "
    "ssot__GenAiGatewayRequestId__c, ssot__GenAiGatewayResponseId__c, ssot__ErrorMessageText__c, "
    "ssot__StartTimestamp__c, ssot__EndTimestamp__c "
    "FROM ssot__AiAgentInteractionStep__dlm "
    "WHERE ssot__AiAgentInteractionId__c = '{interaction_id}' "
    "ORDER BY ssot__StartTimestamp__c ASC"
)

_SOQL_MESSAGES = (
    "SELECT ssot__Id__c, ssot__AiAgentInteractionMessageType__c, ssot__ContentText__c, "
    "ssot__MessageSentTimestamp__c "
    "FROM ssot__AiAgentInteractionMessage__dlm "
    "WHERE ssot__AiAgentInteractionId__c = '{interaction_id}' "
    "ORDER BY ssot__MessageSentTimestamp__c ASC"
)

# Step-type classifier. ``ssot__AiAgentInteractionStepType__c`` is a free-text
# string (not a picklist). Values observed live: ``LLM_STEP`` (carries the
# generation/gateway ids → model.invoke), ``ACTION_STEP`` (a tool/Apex/flow
# invocation → tool.call), and ``TOPIC_STEP`` (topic routing → the generic
# agent.interaction). Matching is on substrings of type/subtype plus a
# GenerationId fallback, so unseen variants in other orgs (e.g. escalation /
# handoff steps) still classify sensibly.
_MODEL_KEYWORDS = ("llm", "generat", "reason", "plan", "respond", "completion", "think", "model")
_TOOL_KEYWORDS = ("action", "tool", "function", "apex", "flow", "retriev", "search")
_HANDOFF_KEYWORDS = ("escalat", "handoff", "transfer")


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _clean(value: Any) -> Optional[str]:
    """Return a stripped string, or None for null / empty / the ``NOT_SET``
    sentinel Salesforce uses in these DMOs."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == _NOT_SET:
        return None
    return text


def _sf_datetime(date_str: str) -> str:
    try:
        dt = datetime.fromisoformat(date_str)
    except ValueError:
        return date_str
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _classify_step(step: Dict[str, Any]) -> str:
    """Map a step to one of ``model`` / ``tool`` / ``handoff`` / ``unknown``."""
    blob = " ".join(
        filter(None, [(_clean(step.get(_STDM.ST_TYPE)) or ""), (_clean(step.get(_STDM.ST_SUBTYPE)) or "")])
    ).lower()
    if any(k in blob for k in _HANDOFF_KEYWORDS):
        return "handoff"
    if any(k in blob for k in _MODEL_KEYWORDS):
        return "model"
    if any(k in blob for k in _TOOL_KEYWORDS):
        return "tool"
    # A generation / gateway id means the gen-AI gateway was called → LLM step.
    if (
        _clean(step.get(_STDM.ST_GENERATION_ID))
        or _clean(step.get(_STDM.ST_GW_REQUEST_ID))
        or _clean(step.get(_STDM.ST_GW_RESPONSE_ID))
    ):
        return "model"
    return "unknown"


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
    """Thin HTTP wrapper around the Salesforce REST API.

    Mirrors ateam's ``SalesforceConnection`` (the one piece truly shared with
    the MVP's richer Agentforce methods) so a later migration is a move.
    """

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
    """Salesforce Agentforce adapter — batch import from the Session Tracing
    Data Model (STDM) in Data Cloud.

    Connects via OAuth client-credentials, then for each
    ``ssot__AiAgentSession__dlm`` walks its interactions
    (``ssot__AiAgentInteraction__dlm``, type ``TURN``) and, per interaction,
    its steps (``ssot__AiAgentInteractionStep__dlm``) and messages
    (``ssot__AiAgentInteractionMessage__dlm``). Each session is one trace.

    Event mapping
    -------------
    * session → an ``agent.lifecycle`` start event + ``agent.input`` (opening
      user message + agent identity) + ``agent.output`` (end type + final
      agent message).
    * each step → ``model.invoke`` (carries the input/output text and the
      ``generation_id`` / ``gateway_request_id`` / ``gateway_response_id``
      metadata) or ``tool.call`` / ``agent.handoff`` / ``agent.error``.
    * each message → an ``agent.interaction`` carrying the readable turn.

    Tokens / cost
    -------------
    The STDM has **no token-count fields**, so this adapter emits **no
    ``cost.record``** (matching ateam, which does not fabricate cost). It
    instead carries the generation / gateway ids on ``model.invoke`` as the
    hook for a future ``GenAIGeneration`` token join.

    Migration safety (toward the ateam Agentforce subsystem)
    --------------------------------------------------------
    1. One trace per session and ``session_id`` carried on every payload, so
       the session id is the trace's identity (ateam keys its richer methods
       off the session id) — realised the same way the langfuse importer
       carries its external trace id, rather than overriding the core
       collector's trace id.
    2. The ``agent.lifecycle`` start event (``lifecycle_action="start"`` +
       ``session_id``) that ateam's ``import_service`` keys on.
    3. STDM names centralised in :class:`_STDM`.

    Usage::

        adapter = AgentforceAdapter(client)
        adapter.connect(
            credentials={"client_id": "...", "client_secret": "...", "instance_url": "https://myorg.my.salesforce.com"}
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

        ``since_cursor`` — when provided, only sessions whose start timestamp
        strictly exceeds the cursor are imported. On return the summary's
        ``next_cursor`` is the max start timestamp seen, so the caller can
        persist it and pass it into the next run for exactly-once sync.
        """
        conn = self._connection
        if conn is None or not self._connected:
            raise RuntimeError("Adapter is not connected — call connect() first")

        where_parts: List[str] = []
        if since_cursor:
            where_parts.append(f"{_STDM.S_START} > {_sf_datetime(since_cursor)}")
        if start_date:
            where_parts.append(f"{_STDM.S_START} >= {_sf_datetime(start_date)}")
        if end_date:
            where_parts.append(f"{_STDM.S_START} < {_sf_datetime(end_date)}")
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
                # Advance the cursor to the latest start timestamp seen. Values
                # are ISO-8601 so lexicographic comparison is correct.
                start_time = session.get(_STDM.S_START)
                if start_time and (max_cursor is None or str(start_time) > str(max_cursor)):
                    max_cursor = str(start_time)
            except Exception:
                log.warning(
                    "layerlens: error importing session %s",
                    session.get(_STDM.BIZ_ID),
                    exc_info=True,
                )
                summary["errors"] += 1

        if max_cursor is not None:
            summary["next_cursor"] = max_cursor
        return summary

    # ------------------------------------------------------------------
    # Per-session import
    # ------------------------------------------------------------------

    def _safe_query(self, conn: _SalesforceConnection, soql: str, what: str, ref: str) -> List[Dict[str, Any]]:
        try:
            return conn.query(soql)
        except Exception:
            log.warning("layerlens: failed to query Agentforce %s for %s", what, ref, exc_info=True)
            return []

    def _import_session(self, conn: _SalesforceConnection, session: Dict[str, Any]) -> int:
        session_id = _clean(session.get(_STDM.BIZ_ID)) or ""
        emitted = 0

        self._begin_run()
        try:
            root = self._get_root_span()

            participants = self._safe_query(
                conn, _SOQL_PARTICIPANTS.format(session_id=session_id), "participants", session_id
            )
            identity = self._agent_identity(participants)

            interactions = self._safe_query(
                conn, _SOQL_INTERACTIONS.format(session_id=session_id), "interactions", session_id
            )

            # Gather each interaction's children up front so the session-level
            # input/output content can reference the first/last message.
            turns: List[tuple] = []
            for inter in interactions:
                interaction_id = _clean(inter.get(_STDM.BIZ_ID)) or ""
                steps = self._safe_query(
                    conn, _SOQL_STEPS.format(interaction_id=interaction_id), "steps", interaction_id
                )
                messages = self._safe_query(
                    conn, _SOQL_MESSAGES.format(interaction_id=interaction_id), "messages", interaction_id
                )
                turns.append((inter, steps, messages))

            first_input = self._edge_message(turns, "Input", last=False)
            last_output = self._edge_message(turns, "Output", last=True)

            emitted += self._emit_lifecycle_start(session_id, identity, root)
            emitted += self._emit_agent_input(session, session_id, identity, first_input, root)
            for inter, steps, messages in turns:
                emitted += self._process_interaction(session_id, inter, steps, messages)
            emitted += self._emit_agent_output(session, session_id, last_output)
        finally:
            self._end_run()

        return emitted

    # ------------------------------------------------------------------
    # Payload helper — session_id on every event (migration contract)
    # ------------------------------------------------------------------

    def _session_payload(self, session_id: str, **extra: Any) -> Dict[str, Any]:
        payload = self._payload(session_id=session_id)
        for key, value in extra.items():
            if value is not None:
                payload[key] = value
        return payload

    # ------------------------------------------------------------------
    # Agent identity (from participants — live-populated; Moment is empty)
    # ------------------------------------------------------------------

    def _agent_identity(self, participants: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
        identity: Dict[str, Optional[str]] = {}
        for p in participants:
            name = _clean(p.get(_STDM.P_AGENT_API))
            if name and "agent_name" not in identity:
                identity["agent_name"] = name
                identity["agent_version"] = _clean(p.get(_STDM.P_AGENT_VERSION))
                identity["agent_type"] = _clean(p.get(_STDM.P_AGENT_TYPE))
            role = (_clean(p.get(_STDM.P_ROLE)) or "").upper()
            if role == "USER" and "participant_id" not in identity:
                identity["participant_id"] = _clean(p.get(_STDM.P_PARTICIPANT_ID))
        return identity

    @staticmethod
    def _edge_message(turns: List[tuple], message_type: str, *, last: bool) -> Optional[str]:
        """First (or last) message content of ``message_type`` across turns."""
        found: Optional[str] = None
        wanted = message_type.lower()
        for _inter, _steps, messages in turns:
            for msg in messages:
                if (_clean(msg.get(_STDM.M_TYPE)) or "").lower() != wanted:
                    continue
                content = _clean(msg.get(_STDM.M_CONTENT))
                if content is None:
                    continue
                if not last:
                    return content
                found = content
        return found

    # ------------------------------------------------------------------
    # Session-level events
    # ------------------------------------------------------------------

    def _emit_lifecycle_start(self, session_id: str, identity: Dict[str, Optional[str]], root: str) -> int:
        payload = self._session_payload(
            session_id,
            lifecycle_action="start",
            agent_name=identity.get("agent_name"),
        )
        self._emit(
            "agent.lifecycle",
            payload,
            span_id=self._new_span_id(),
            parent_span_id=root,
            span_name="session_start",
        )
        return 1

    def _emit_agent_input(
        self,
        session: Dict[str, Any],
        session_id: str,
        identity: Dict[str, Optional[str]],
        first_input: Optional[str],
        root: str,
    ) -> int:
        payload = self._session_payload(
            session_id,
            agent_name=identity.get("agent_name"),
            agent_version=identity.get("agent_version"),
            agent_type=identity.get("agent_type"),
            participant_id=identity.get("participant_id"),
            channel=_clean(session.get(_STDM.S_CHANNEL)),
            start_time=_clean(session.get(_STDM.S_START)),
        )
        self._set_if_capturing(payload, "content", truncate(first_input, 4000))
        self._emit("agent.input", payload, span_id=root, parent_span_id=None, span_name="session")
        return 1

    def _emit_agent_output(self, session: Dict[str, Any], session_id: str, last_output: Optional[str]) -> int:
        payload = self._session_payload(
            session_id,
            outcome=_clean(session.get(_STDM.S_END_TYPE)),
            end_time=_clean(session.get(_STDM.S_END)),
        )
        self._set_if_capturing(payload, "content", truncate(last_output, 4000))
        self._emit("agent.output", payload, span_name="session_end")
        return 1

    # ------------------------------------------------------------------
    # Per-interaction processing
    # ------------------------------------------------------------------

    def _process_interaction(
        self,
        session_id: str,
        inter: Dict[str, Any],
        steps: List[Dict[str, Any]],
        messages: List[Dict[str, Any]],
    ) -> int:
        emitted = 0
        for step in steps:
            emitted += self._process_step(session_id, inter, step)
        for msg in messages:
            emitted += self._emit_message(session_id, inter, msg)
        return emitted

    def _interaction_meta(self, session_id: str, inter: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
        return self._session_payload(
            session_id,
            interaction_id=_clean(inter.get(_STDM.BIZ_ID)),
            topic=_clean(inter.get(_STDM.I_TOPIC)),
            **extra,
        )

    def _process_step(self, session_id: str, inter: Dict[str, Any], step: Dict[str, Any]) -> int:
        # An errored step is reported as agent.error regardless of its kind.
        if _clean(step.get(_STDM.ST_ERROR)):
            return self._emit_step_error(session_id, inter, step)

        kind = _classify_step(step)
        if kind == "tool":
            return self._emit_tool(session_id, inter, step)
        if kind == "handoff":
            return self._emit_handoff(session_id, inter, step)
        if kind == "model":
            return self._emit_model_invoke(session_id, inter, step)
        return self._emit_unknown_step(session_id, inter, step)

    def _emit_model_invoke(self, session_id: str, inter: Dict[str, Any], step: Dict[str, Any]) -> int:
        payload = self._interaction_meta(session_id, inter, step_name=_clean(step.get(_STDM.ST_NAME)))
        # Generation / gateway ids — metadata, never content-gated (the hook
        # for a future token/cost join; the STDM itself has no token counts).
        for key, field in (
            ("generation_id", _STDM.ST_GENERATION_ID),
            ("gateway_request_id", _STDM.ST_GW_REQUEST_ID),
            ("gateway_response_id", _STDM.ST_GW_RESPONSE_ID),
        ):
            value = _clean(step.get(field))
            if value:
                payload[key] = value
        self._set_if_capturing(payload, "messages", truncate(_clean(step.get(_STDM.ST_INPUT)), 4000))
        self._set_if_capturing(payload, "output_message", truncate(_clean(step.get(_STDM.ST_OUTPUT)), 4000))
        self._emit(
            "model.invoke",
            payload,
            span_id=self._new_span_id(),
            span_name=_clean(step.get(_STDM.ST_NAME)) or "llm_call",
        )
        return 1

    def _emit_tool(self, session_id: str, inter: Dict[str, Any], step: Dict[str, Any]) -> int:
        tool_name = _clean(step.get(_STDM.ST_NAME)) or "tool"
        payload = self._interaction_meta(
            session_id,
            inter,
            tool_name=tool_name,
            step_type=_clean(step.get(_STDM.ST_TYPE)),
        )
        self._set_if_capturing(payload, "input", truncate(_clean(step.get(_STDM.ST_INPUT)), 4000))
        self._set_if_capturing(payload, "output", truncate(_clean(step.get(_STDM.ST_OUTPUT)), 4000))
        self._emit("tool.call", payload, span_id=self._new_span_id(), span_name=tool_name)
        return 1

    def _emit_handoff(self, session_id: str, inter: Dict[str, Any], step: Dict[str, Any]) -> int:
        payload = self._interaction_meta(
            session_id,
            inter,
            step_name=_clean(step.get(_STDM.ST_NAME)),
            step_type=_clean(step.get(_STDM.ST_TYPE)),
        )
        self._set_if_capturing(payload, "reason", truncate(_clean(step.get(_STDM.ST_INPUT)), 4000))
        self._emit("agent.handoff", payload, span_id=self._new_span_id(), span_name="handoff")
        return 1

    def _emit_step_error(self, session_id: str, inter: Dict[str, Any], step: Dict[str, Any]) -> int:
        payload = self._interaction_meta(
            session_id,
            inter,
            step_name=_clean(step.get(_STDM.ST_NAME)),
            step_type=_clean(step.get(_STDM.ST_TYPE)),
            error_message=_clean(step.get(_STDM.ST_ERROR)),
            error_type="step_error",
            status="error",
        )
        self._emit("agent.error", payload, span_id=self._new_span_id(), span_name="error")
        return 1

    def _emit_unknown_step(self, session_id: str, inter: Dict[str, Any], step: Dict[str, Any]) -> int:
        step_name = _clean(step.get(_STDM.ST_NAME))
        payload = self._interaction_meta(
            session_id,
            inter,
            step_type=_clean(step.get(_STDM.ST_TYPE)) or "unknown",
            step_name=step_name,
        )
        self._set_if_capturing(payload, "input", truncate(_clean(step.get(_STDM.ST_INPUT)), 4000))
        self._set_if_capturing(payload, "output", truncate(_clean(step.get(_STDM.ST_OUTPUT)), 4000))
        self._emit("agent.interaction", payload, span_id=self._new_span_id(), span_name=step_name or "interaction")
        return 1

    def _emit_message(self, session_id: str, inter: Dict[str, Any], msg: Dict[str, Any]) -> int:
        message_type = _clean(msg.get(_STDM.M_TYPE))
        role = {"input": "user", "output": "agent"}.get((message_type or "").lower(), "unknown")
        payload = self._session_payload(
            session_id,
            interaction_id=_clean(inter.get(_STDM.BIZ_ID)),
            role=role,
            message_type=message_type,
        )
        self._set_if_capturing(payload, "content", truncate(_clean(msg.get(_STDM.M_CONTENT)), 4000))
        self._emit("agent.interaction", payload, span_id=self._new_span_id(), span_name="message")
        return 1
