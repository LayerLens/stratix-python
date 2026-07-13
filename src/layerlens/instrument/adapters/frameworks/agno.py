from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from ._utils import safe_serialize
from ..._identity import _s, _is_generic
from ._base_framework import FrameworkAdapter
from ..._capture_config import CaptureConfig

log = logging.getLogger(__name__)

try:
    import agno  # pyright: ignore[reportMissingImports]  # noqa: F401

    _HAS_AGNO = True
except ImportError:
    _HAS_AGNO = False


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _model_id(agent: Any) -> Optional[str]:
    model = getattr(agent, "model", None)
    if model is None:
        return None
    return getattr(model, "id", None) or str(model)


def _extract_tokens(result: Any) -> Dict[str, int]:
    metrics = getattr(result, "metrics", None)
    if metrics is None:
        return {}

    inp = getattr(metrics, "input_tokens", None)
    out = getattr(metrics, "output_tokens", None)
    reasoning = getattr(metrics, "reasoning_tokens", None) or getattr(metrics, "thinking_tokens", None)
    cached = getattr(metrics, "cached_tokens", None) or getattr(metrics, "cache_read_tokens", None)
    audio = getattr(metrics, "audio_tokens", None)
    time_ms = getattr(metrics, "duration_ms", None) or getattr(metrics, "time", None)

    if inp is not None or out is not None:
        tokens: Dict[str, int] = {}
        if inp:
            tokens["tokens_prompt"] = int(inp)
        if out:
            tokens["tokens_completion"] = int(out)
        if inp or out:
            tokens["tokens_total"] = (int(inp) if inp else 0) + (int(out) if out else 0)
        if reasoning:
            tokens["reasoning_tokens"] = int(reasoning)
        if cached:
            tokens["cached_tokens"] = int(cached)
        if audio:
            tokens["audio_tokens"] = int(audio)
        if time_ms:
            try:
                tokens["duration_ms"] = int(float(time_ms))
            except (TypeError, ValueError):
                pass
        return tokens

    details = getattr(metrics, "details", None)
    if not isinstance(details, dict):
        return {}
    total_in = total_out = total_reason = total_cached = 0
    per_model: Dict[str, Dict[str, int]] = {}
    for model_name, model_metrics_list in details.items():
        if not isinstance(model_metrics_list, list):
            continue
        model_in = model_out = 0
        for mm in model_metrics_list:
            model_in += getattr(mm, "input_tokens", 0) or 0
            model_out += getattr(mm, "output_tokens", 0) or 0
            total_reason += getattr(mm, "reasoning_tokens", 0) or 0
            total_cached += getattr(mm, "cached_tokens", 0) or 0
        total_in += model_in
        total_out += model_out
        if model_in or model_out:
            per_model[str(model_name)] = {
                "tokens_prompt": model_in,
                "tokens_completion": model_out,
                "tokens_total": model_in + model_out,
            }
    if not total_in and not total_out:
        return {}
    tokens = {}
    if total_in:
        tokens["tokens_prompt"] = total_in
    if total_out:
        tokens["tokens_completion"] = total_out
    tokens["tokens_total"] = total_in + total_out
    if total_reason:
        tokens["reasoning_tokens"] = total_reason
    if total_cached:
        tokens["cached_tokens"] = total_cached
    # Multi-model aggregation: surface per-model breakdown so we can see which
    # model contributed how many tokens in a hybrid run.
    if len(per_model) > 1:
        tokens["per_model"] = per_model  # type: ignore[assignment]
    return tokens


def _extract_tools(result: Any) -> List[Dict[str, Any]]:
    tools = getattr(result, "tools", None)
    if not tools:
        return []
    out = []
    for te in tools:
        entry: Dict[str, Any] = {
            "tool_name": getattr(te, "tool_name", None) or getattr(te, "name", "unknown"),
            "tool_args": getattr(te, "tool_args", None) or getattr(te, "arguments", None),
            "result": getattr(te, "result", None),
        }
        te_metrics = getattr(te, "metrics", None)
        if te_metrics is not None:
            duration = getattr(te_metrics, "execution_time", None) or getattr(te_metrics, "duration", None)
            if duration is not None:
                entry["latency_ms"] = float(duration) * 1000
        out.append(entry)
    return out


class AgnoAdapter(FrameworkAdapter):
    """Agno adapter wrapping ``Agent.run()`` / ``Agent.arun()``.

    Uses ``_begin_run`` / ``_end_run`` for ContextVar-based collector
    lifecycle. All telemetry is extracted post-hoc from ``RunOutput``.

    Usage::

        adapter = AgnoAdapter(client)
        agent = adapter.connect(target=agent)
        result = agent.run("hello")
        adapter.disconnect()
    """

    name = "agno"

    def __init__(self, client: Any, capture_config: Optional[CaptureConfig] = None) -> None:
        super().__init__(client, capture_config)
        self._originals: Dict[int, Dict[str, Any]] = {}
        self._wrapped_agents: List[Any] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _on_connect(self, target: Any = None, **kwargs: Any) -> None:
        self._check_dependency(_HAS_AGNO)
        if target is not None:
            self._instrument_agent(target)

    def _on_disconnect(self) -> None:
        for agent in self._wrapped_agents:
            self._unwrap_agent(agent)
        self._wrapped_agents.clear()
        self._originals.clear()

    # ------------------------------------------------------------------
    # Instrumentation
    # ------------------------------------------------------------------

    def _instrument_agent(self, agent: Any) -> None:
        agent_id = id(agent)
        if agent_id in self._originals:
            return
        originals: Dict[str, Any] = {}
        if hasattr(agent, "run"):
            originals["run"] = agent.run
            agent.run = self._wrap_sync(agent, agent.run)
        if hasattr(agent, "arun"):
            originals["arun"] = agent.arun
            agent.arun = self._wrap_async(agent, agent.arun)
        self._originals[agent_id] = originals
        self._wrapped_agents.append(agent)

    def _unwrap_agent(self, agent: Any) -> None:
        originals = self._originals.get(id(agent))
        if not originals:
            return
        for method_name, original in originals.items():
            try:
                setattr(agent, method_name, original)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Wrappers
    # ------------------------------------------------------------------

    def _wrap_sync(self, agent: Any, original: Any) -> Any:
        adapter = self

        def _traced_run(*args: Any, **kwargs: Any) -> Any:
            if not adapter._connected:
                return original(*args, **kwargs)
            input_data = kwargs.get("message") or (args[0] if args else None)
            adapter._begin_run()
            adapter._start_timer("run")
            adapter._on_run_start(agent, input_data)
            error: Optional[Exception] = None
            result = None
            try:
                result = original(*args, **kwargs)
            except Exception as exc:
                error = exc
                raise
            finally:
                adapter._on_run_end(agent, result, error)
                adapter._end_run()
            return result

        _traced_run._layerlens_original = original  # type: ignore[attr-defined]
        return _traced_run

    def _wrap_async(self, agent: Any, original: Any) -> Any:
        adapter = self

        async def _traced_arun(*args: Any, **kwargs: Any) -> Any:
            if not adapter._connected:
                return await original(*args, **kwargs)
            input_data = kwargs.get("message") or (args[0] if args else None)
            adapter._begin_run()
            adapter._start_timer("run")
            adapter._on_run_start(agent, input_data)
            error: Optional[Exception] = None
            result = None
            try:
                result = await original(*args, **kwargs)
            except Exception as exc:
                error = exc
                raise
            finally:
                adapter._on_run_end(agent, result, error)
                adapter._end_run()
            return result

        _traced_arun._layerlens_original = original  # type: ignore[attr-defined]
        return _traced_arun

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    @staticmethod
    def _span_name(name: Optional[str], kind: str = "agent") -> str:
        """A clean span label — never embeds a fabricated ``None`` identity.

        (span_name is never an Agent-column source; this keeps the label tidy.)"""
        return f"agno:{name}" if name else f"agno:{kind}"

    def _on_run_start(self, agent: Any, input_data: Any) -> None:
        root = self._get_root_span()
        name = _agent_name(agent) or _raw_agent_name(agent)
        model = _model_id(agent)
        self._emit_config(agent, name, model)
        payload = self._payload()
        if name:
            payload["agent_name"] = name
        if model:
            payload["model"] = model
        self._set_if_capturing(payload, "input", safe_serialize(input_data))
        self._emit(
            "agent.input",
            payload,
            span_id=root,
            parent_span_id=None,
            span_name=self._span_name(name),
        )

    def _emit_config(self, agent: Any, name: Optional[str], model: Optional[str]) -> None:
        """Emit environment.config carrying the declared team roster.

        The graph engine reads a run's agent roster from ``config.team_members``;
        re-emitting it (the honest original did — this adapter had dropped it)
        lets a multi-agent Agno Team render its members. Content-free config
        (model + roster) — no run I/O — and only member names the developer
        declared. Emitted per run so every trace carries its own config node."""
        config: Dict[str, Any] = {}
        if model:
            config["model"] = model
        members = _team_members(agent)
        if members:
            config["team_members"] = members
        payload = self._payload(config=config)
        if name:
            payload["agent_name"] = name
        self._emit(
            "environment.config",
            payload,
            span_id=self._new_span_id(),
            parent_span_id=self._get_root_span(),
            span_name=self._span_name(name, kind="config"),
        )

    def _on_run_end(self, agent: Any, result: Any, error: Optional[Exception]) -> None:
        self._emit_output(agent, result, error)
        if result is not None:
            self._emit_model(agent, result)
            self._emit_tools(agent, result)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _emit_output(self, agent: Any, result: Any, error: Optional[Exception]) -> None:
        root = self._get_root_span()
        name = _agent_name(agent) or _raw_agent_name(agent)
        model = _model_id(agent)
        latency_ms = self._stop_timer("run")

        output = getattr(result, "content", None) if result is not None else None
        payload = self._payload()
        if name:
            payload["agent_name"] = name
        if model:
            payload["model"] = model
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms
        if error:
            payload["error"] = str(error)
            payload["error_type"] = type(error).__name__
        self._set_if_capturing(payload, "output", safe_serialize(output))
        self._emit(
            "agent.output",
            payload,
            span_id=root,
            parent_span_id=None,
            span_name=self._span_name(name),
        )

    def _emit_model(self, agent: Any, result: Any) -> None:
        model = _model_id(agent)
        if not model:
            return
        root = self._get_root_span()
        tokens = _extract_tokens(result)

        name = _agent_name(agent) or _raw_agent_name(agent)
        span_id = self._new_span_id()
        payload = self._payload(model=model)
        if name:
            payload["agent_name"] = name
        payload.update(tokens)
        self._emit(
            "model.invoke",
            payload,
            span_id=span_id,
            parent_span_id=root,
            span_name="model.invoke",
        )

        if tokens:
            cost_payload = self._payload(model=model)
            cost_payload.update(tokens)
            self._emit("cost.record", cost_payload, span_id=span_id, parent_span_id=root)

    def _emit_tools(self, agent: Any, result: Any) -> None:
        root = self._get_root_span()
        from_name = _agent_name(agent)
        for tool in _extract_tools(result):
            # A team delegation surfaces as a transfer/forward tool call; classify
            # it into an agent.handoff edge rather than burying it in a tool.call.
            if _is_transfer_tool(tool["tool_name"]):
                self._emit_handoff(from_name, tool)
                continue
            span_id = self._new_span_id()

            call_payload = self._payload(tool_name=tool["tool_name"])
            self._set_if_capturing(call_payload, "input", safe_serialize(tool.get("tool_args")))
            self._emit("tool.call", call_payload, span_id=span_id, parent_span_id=root)

            result_payload = self._payload(tool_name=tool["tool_name"])
            self._set_if_capturing(result_payload, "output", safe_serialize(tool.get("result")))
            if tool.get("latency_ms") is not None:
                result_payload["latency_ms"] = tool["latency_ms"]
            self._emit("tool.result", result_payload, span_id=span_id, parent_span_id=root)

    def _emit_handoff(self, from_name: Optional[str], tool: Dict[str, Any]) -> None:
        """Emit an agent.handoff edge for a team-delegation transfer tool call.

        ``from_agent`` is the delegating agent/team; ``to_agent`` is parsed from
        the transfer tool's real arguments. Either endpoint is omitted when it is
        not producer-declared (an honest blank beats a fabricated node)."""
        root = self._get_root_span()
        to_agent = _parse_handoff_target(tool.get("tool_args"))
        payload = self._payload(reason="team_delegation")
        if from_name:
            payload["from_agent"] = from_name
        if to_agent:
            payload["to_agent"] = to_agent
        self._emit(
            "agent.handoff",
            payload,
            span_id=self._new_span_id(),
            parent_span_id=root,
            span_name="agno:handoff",
        )


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _agent_name(agent: Any) -> Optional[str]:
    """The producer-DECLARED agent/team name, honest-guarded — or ``None``.

    Honesty is the whole point of the graph contract: an unnamed Agno agent
    surfaces the ``agno_agent`` placeholder, which is on the generic denylist and
    must NEVER become an Agent-column identity. When the developer declared no
    name we emit no ``agent_name`` at all (an honest blank) rather than inventing
    one. A genuine developer name (``research_team``, ``researcher``) survives.
    """
    raw = getattr(agent, "name", None)
    name = raw.strip() if isinstance(raw, str) else None
    if not name or _is_generic(name):
        return None
    return name


def _raw_agent_name(agent: Any) -> Optional[str]:
    """ateam-parity verbatim fallback (#3): the raw declared/framework agent name,
    sanitized (control/bidi stripped via :func:`_s`) but NOT honesty-guarded — so an
    unnamed/generic Agno agent still surfaces its (generic) identity like ateam,
    which reads ``agent_id`` verbatim. Used ONLY for the per-event ``agent_name``
    stamp; ``agent.handoff`` endpoints stay on the honest :func:`_agent_name` so no
    fabricated edges between class-default agents are drawn."""
    return _s(getattr(agent, "name", None))


def _team_members(agent: Any) -> Optional[List[str]]:
    """Declared member names for a Team (or a team-bearing agent), else None.

    Reads the roster from either ``agent.team.members`` (an agent that belongs to
    a team) or ``agent.members`` (the instrumented object IS an Agno ``Team``).
    Only producer-declared member ``name`` values are surfaced — an unnamed
    member contributes nothing rather than a fabricated label."""
    team = getattr(agent, "team", None)
    container = team if team is not None else agent
    members = getattr(container, "members", None) or getattr(container, "agents", None)
    if not members:
        return None
    names: List[str] = []
    for m in members:
        raw = getattr(m, "name", None)
        name = raw.strip() if isinstance(raw, str) else None
        if name and not _is_generic(name):
            names.append(name)
    return names or None


def _is_transfer_tool(name: Optional[str]) -> bool:
    """True if a tool-call name is Agno's team-delegation transfer/forward call.

    Agno teams delegate via tools named ``transfer_task_to_member`` /
    ``forward_task_to_member`` (and close variants). Matching the real tool name
    is how we tell a genuine handoff from an ordinary tool call so the delegation
    surfaces as an ``agent.handoff`` edge instead of being buried in a tool.call.
    Mirrors the ateam original (``adapters/agno/lifecycle.py``)."""
    if not name:
        return False
    low = name.lower()
    return ("transfer" in low or "forward" in low) and ("member" in low or "agent" in low or "task" in low)


def _parse_handoff_target(args: Any) -> Optional[str]:
    """Best-effort target-agent name from a transfer tool's real arguments.

    Returns ``None`` (not a fabricated ``"unknown"``) when no declared target is
    present — an honest blank endpoint beats a placeholder node."""
    data: Any = args
    if isinstance(args, str):
        try:
            data = json.loads(args)
        except (ValueError, TypeError):
            return None
    if isinstance(data, dict):
        for key in ("member_id", "agent_name", "agent_id", "to_agent", "member", "to"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return None
