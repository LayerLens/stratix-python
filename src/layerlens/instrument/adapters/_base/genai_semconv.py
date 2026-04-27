"""OpenTelemetry GenAI semantic-convention helper.

Implements spec ``07-otel-genai-semantic-conventions.md`` for the
LayerLens instrument adapters: every LLM-call event payload is
additively stamped with the standard ``gen_ai.*`` attribute set so OTel
exporters and downstream backends (Datadog, Grafana, Jaeger, Honeycomb)
can read traces using the upstream-defined namespace.

The helper is **additive** (CLAUDE.md "complete means complete"):

* It never removes existing custom keys (``provider``, ``model``,
  ``parameters``, ``prompt_tokens`` etc.) — those continue to flow
  alongside, so dashboards and queries built on the legacy attribute
  set keep working during the migration.
* It writes ``gen_ai.*`` keys directly into the dict payload (in place)
  so the same attribute set survives the BaseAdapter circuit-breaker
  path, the sink dispatch loop, and every downstream sink/exporter.
* It tolerates partial / missing fields silently — adapters often emit
  on the error path with neither usage nor response object available.

Spec reference:
    https://opentelemetry.io/docs/specs/semconv/gen-ai/
    docs/incubation-docs/adapter-framework/07-otel-genai-semantic-conventions.md

The set of attributes emitted here matches the stratix-side exporter so
that round-tripping through export → import preserves the GenAI shape
(see ``stratix/observability/otel.py`` and
``tests/sdk/python/test_otel_exporter_genai.py``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

# ---------------------------------------------------------------------------
# OTel GenAI attribute keys (verbatim from upstream semantic conventions).
# These constants are intentionally module-level strings so adapter call
# sites read like the spec and so the test suite can pin every key the
# helper writes.
# ---------------------------------------------------------------------------

# Core / system attributes.
GEN_AI_SYSTEM: str = "gen_ai.system"
GEN_AI_PROVIDER_NAME: str = "gen_ai.provider.name"
GEN_AI_OPERATION_NAME: str = "gen_ai.operation.name"

# Request attributes.
GEN_AI_REQUEST_MODEL: str = "gen_ai.request.model"
GEN_AI_REQUEST_MAX_TOKENS: str = "gen_ai.request.max_tokens"
GEN_AI_REQUEST_TEMPERATURE: str = "gen_ai.request.temperature"
GEN_AI_REQUEST_TOP_P: str = "gen_ai.request.top_p"
GEN_AI_REQUEST_TOP_K: str = "gen_ai.request.top_k"
GEN_AI_REQUEST_FREQUENCY_PENALTY: str = "gen_ai.request.frequency_penalty"
GEN_AI_REQUEST_PRESENCE_PENALTY: str = "gen_ai.request.presence_penalty"
GEN_AI_REQUEST_STOP_SEQUENCES: str = "gen_ai.request.stop_sequences"
GEN_AI_REQUEST_SEED: str = "gen_ai.request.seed"
GEN_AI_REQUEST_CHOICE_COUNT: str = "gen_ai.request.choice.count"
GEN_AI_REQUEST_ENCODING_FORMATS: str = "gen_ai.request.encoding_formats"

# Response attributes.
GEN_AI_RESPONSE_ID: str = "gen_ai.response.id"
GEN_AI_RESPONSE_MODEL: str = "gen_ai.response.model"
GEN_AI_RESPONSE_FINISH_REASONS: str = "gen_ai.response.finish_reasons"

# Usage attributes.
GEN_AI_USAGE_INPUT_TOKENS: str = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS: str = "gen_ai.usage.output_tokens"

# Tool attributes.
GEN_AI_TOOL_NAME: str = "gen_ai.tool.name"
GEN_AI_TOOL_CALL_ID: str = "gen_ai.tool.call.id"
GEN_AI_TOOL_DESCRIPTION: str = "gen_ai.tool.description"
GEN_AI_TOOL_TYPE: str = "gen_ai.tool.type"

# Agent attributes.
GEN_AI_AGENT_ID: str = "gen_ai.agent.id"
GEN_AI_AGENT_NAME: str = "gen_ai.agent.name"
GEN_AI_AGENT_DESCRIPTION: str = "gen_ai.agent.description"

# Provider-specific (OpenAI).
GEN_AI_OPENAI_REQUEST_SERVICE_TIER: str = "gen_ai.openai.request.service_tier"
GEN_AI_OPENAI_REQUEST_RESPONSE_FORMAT: str = "gen_ai.openai.request.response_format"
GEN_AI_OPENAI_RESPONSE_SERVICE_TIER: str = "gen_ai.openai.response.service_tier"
GEN_AI_OPENAI_RESPONSE_SYSTEM_FINGERPRINT: str = "gen_ai.openai.response.system_fingerprint"

# Provider-specific (Anthropic).
GEN_AI_ANTHROPIC_CACHE_CREATION_INPUT_TOKENS: str = (
    "gen_ai.anthropic.cache_creation_input_tokens"
)
GEN_AI_ANTHROPIC_CACHE_READ_INPUT_TOKENS: str = (
    "gen_ai.anthropic.cache_read_input_tokens"
)

# Provider-specific (AWS Bedrock — namespaced under aws.bedrock per spec §4.3).
AWS_BEDROCK_GUARDRAIL_ID: str = "aws.bedrock.guardrail.id"
AWS_BEDROCK_KNOWLEDGE_BASE_ID: str = "aws.bedrock.knowledge_base.id"
AWS_BEDROCK_AGENT_ID: str = "aws.bedrock.agent.id"

# Provider-specific (Google Vertex).
GEN_AI_GOOGLE_SAFETY_RATINGS: str = "gen_ai.google.safety_ratings"


# ---------------------------------------------------------------------------
# Operation enum values (per spec §3.1 span naming).
# ---------------------------------------------------------------------------

OPERATION_CHAT: str = "chat"
OPERATION_TEXT_COMPLETION: str = "text_completion"
OPERATION_EMBED: str = "embeddings"
OPERATION_GENERATE_CONTENT: str = "generate_content"


# ---------------------------------------------------------------------------
# Canonical system / provider names (per spec §4 and the upstream
# ``gen_ai.system`` / ``gen_ai.provider.name`` enumeration).
# ---------------------------------------------------------------------------

SYSTEM_OPENAI: str = "openai"
SYSTEM_AZURE_OPENAI: str = "azure.openai"
SYSTEM_ANTHROPIC: str = "anthropic"
SYSTEM_AWS_BEDROCK: str = "aws.bedrock"
SYSTEM_GCP_VERTEX: str = "gcp.vertex_ai"
SYSTEM_GCP_GEMINI: str = "gcp.gemini"
SYSTEM_COHERE: str = "cohere"
SYSTEM_MISTRAL: str = "mistral_ai"
SYSTEM_OLLAMA: str = "ollama"
SYSTEM_LITELLM: str = "litellm"

# Framework systems — used when the framework adapter cannot identify
# the underlying LLM provider (e.g. langchain wrapping a custom client
# whose class name does not map to a known provider). The framework
# name is reported as a fallback so dashboards still see SOMETHING in
# ``gen_ai.system``. Per spec, ``_OTHER`` is the documented fallback
# value when the system is not enumerated.
SYSTEM_FALLBACK_OTHER: str = "_OTHER"


# Mapping from common substrings (lowercased) found in adapter class /
# framework / provider strings to the canonical ``gen_ai.system`` value.
# Order matters — the first substring match wins, so longer / more
# specific keys must precede their generic suffix.
_SYSTEM_DETECTION_TABLE: List[tuple[str, str]] = [
    ("azure_openai", SYSTEM_AZURE_OPENAI),
    ("azure-openai", SYSTEM_AZURE_OPENAI),
    ("azureopenai", SYSTEM_AZURE_OPENAI),
    ("aws_bedrock", SYSTEM_AWS_BEDROCK),
    ("bedrock", SYSTEM_AWS_BEDROCK),
    ("vertex", SYSTEM_GCP_VERTEX),
    ("google_vertex", SYSTEM_GCP_VERTEX),
    ("gemini", SYSTEM_GCP_GEMINI),
    ("google_adk", SYSTEM_GCP_GEMINI),
    ("anthropic", SYSTEM_ANTHROPIC),
    ("openai", SYSTEM_OPENAI),
    ("cohere", SYSTEM_COHERE),
    ("mistral", SYSTEM_MISTRAL),
    ("ollama", SYSTEM_OLLAMA),
    ("litellm", SYSTEM_LITELLM),
]


def detect_gen_ai_system(name: Optional[str]) -> str:
    """Map an adapter / framework / provider name to a canonical OTel system.

    Args:
        name: Adapter class name (``"AnthropicAdapter"``), framework
            string (``"openai"``), or provider field from a payload
            (``"aws_bedrock"``). May be ``None`` or empty — the helper
            returns the documented spec fallback ``_OTHER`` rather than
            crashing.

    Returns:
        One of the ``SYSTEM_*`` constants. Returns
        :data:`SYSTEM_FALLBACK_OTHER` (``"_OTHER"``) when no match is
        possible. Never returns ``None`` — every emission must carry a
        ``gen_ai.system`` value.
    """
    if not name:
        return SYSTEM_FALLBACK_OTHER
    needle = name.lower()
    for pattern, system in _SYSTEM_DETECTION_TABLE:
        if pattern in needle:
            return system
    return SYSTEM_FALLBACK_OTHER


# ---------------------------------------------------------------------------
# Operation detection.
# ---------------------------------------------------------------------------


def detect_operation(
    payload: Mapping[str, Any],
    request_kwargs: Optional[Mapping[str, Any]] = None,
) -> str:
    """Infer ``gen_ai.operation.name`` from a payload + request kwargs.

    Defaults to :data:`OPERATION_CHAT` because every adapter currently
    in the SDK targets chat-completion / message APIs. Embeddings and
    text-completion adapters override this when they call
    :func:`stamp_genai_attributes` with an explicit ``operation``.

    Args:
        payload: The dict payload being emitted. May include an
            ``operation`` field that overrides detection.
        request_kwargs: Optional original SDK kwargs (e.g. the dict
            passed to ``client.messages.create``). Used to look at
            ``encoding_format`` / ``input`` shape for embedding APIs.

    Returns:
        One of the ``OPERATION_*`` string constants.
    """
    explicit = payload.get("operation") or payload.get("gen_ai_operation")
    if isinstance(explicit, str) and explicit.strip():
        return explicit
    if request_kwargs is not None:
        # Embeddings APIs typically take an ``input`` (string / list)
        # and ``encoding_format`` rather than ``messages``.
        if "encoding_format" in request_kwargs and "messages" not in request_kwargs:
            return OPERATION_EMBED
        if "input" in request_kwargs and "messages" not in request_kwargs:
            # Could be embedding OR completion — embeddings is the
            # safer default for the OpenAI-style endpoint family.
            return OPERATION_EMBED
    return OPERATION_CHAT


# ---------------------------------------------------------------------------
# Field extraction helpers.
# ---------------------------------------------------------------------------


def _coerce_finish_reasons(value: Any) -> Optional[List[str]]:
    """Coerce a finish-reason value into a list of strings.

    OTel spec mandates ``gen_ai.response.finish_reasons`` is an array
    even when only one reason is present.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for item in value:
            if item is None:
                continue
            out.append(str(item))
        return out if out else None
    return [str(value)]


def _read_model_field(payload: Mapping[str, Any]) -> Optional[str]:
    """Read the request-model field, supporting both flat and nested layouts.

    Most adapters write ``payload["model"]`` as a string. The langchain
    adapter writes it as a nested ``{"name": ..., "provider": ...}``
    dict. Both layouts are handled here so the gen_ai stamper does not
    need adapter-specific awareness.
    """
    model = payload.get("model")
    if isinstance(model, str):
        return model
    if isinstance(model, Mapping):
        name = model.get("name")
        if isinstance(name, str):
            return name
    return None


def _read_provider_field(payload: Mapping[str, Any]) -> Optional[str]:
    """Read the provider field, supporting both flat and nested layouts."""
    provider = payload.get("provider")
    if isinstance(provider, str) and provider:
        return provider
    model = payload.get("model")
    if isinstance(model, Mapping):
        nested = model.get("provider")
        if isinstance(nested, str) and nested:
            return nested
    return None


# ---------------------------------------------------------------------------
# Public stamping API.
# ---------------------------------------------------------------------------


def stamp_genai_attributes(
    payload: Dict[str, Any],
    request_kwargs: Optional[Mapping[str, Any]] = None,
    response_obj: Any = None,
    *,
    system: Optional[str] = None,
    operation: Optional[str] = None,
) -> Dict[str, Any]:
    """Add OTel ``gen_ai.*`` attributes to a ``model.invoke`` payload.

    The payload is mutated **in place** and also returned so call sites
    can chain. All standard fields defined in the OTel GenAI semantic
    conventions are populated when their source data is available;
    missing fields are silently skipped (no defaults are invented).

    The helper is purely additive — existing custom keys
    (``provider``, ``model``, ``parameters``, ``prompt_tokens``, etc.)
    are never removed or rewritten. This guarantees gradual migration:
    legacy dashboards / sinks keep reading their custom keys while
    OTel-aware tooling reads the new ``gen_ai.*`` namespace.

    Args:
        payload: The ``model.invoke`` event payload dict (mutated).
        request_kwargs: Optional original SDK kwargs (e.g. the dict
            passed to ``client.messages.create``). Used to extract
            request-side attributes (``temperature``, ``max_tokens``,
            ``top_p`` etc.) when they are not already in ``payload``.
        response_obj: Optional provider response object. Inspected for
            response-side attributes (``response.id``, system
            fingerprint, finish reasons) via duck-typed ``getattr`` /
            mapping access — never crashes on missing fields.
        system: Optional explicit ``gen_ai.system`` value. When ``None``
            the helper detects it from the payload's ``provider`` /
            ``framework`` fields via :func:`detect_gen_ai_system`.
        operation: Optional explicit ``gen_ai.operation.name``. When
            ``None`` the helper infers it via :func:`detect_operation`.

    Returns:
        The same ``payload`` reference, mutated with ``gen_ai.*`` keys.
    """
    # --- gen_ai.system / gen_ai.provider.name (always required) ---
    detected_system: str
    if system is not None and system.strip():
        detected_system = system
    else:
        provider = _read_provider_field(payload) or payload.get("framework")
        detected_system = detect_gen_ai_system(provider if isinstance(provider, str) else None)
    payload[GEN_AI_SYSTEM] = detected_system
    payload[GEN_AI_PROVIDER_NAME] = detected_system

    # --- gen_ai.operation.name ---
    payload[GEN_AI_OPERATION_NAME] = operation if operation is not None else detect_operation(
        payload, request_kwargs
    )

    # --- gen_ai.request.model ---
    request_model = _read_model_field(payload)
    if request_kwargs is not None:
        kw_model = request_kwargs.get("model")
        if isinstance(kw_model, str) and kw_model:
            request_model = kw_model
    if request_model:
        payload[GEN_AI_REQUEST_MODEL] = request_model

    # --- gen_ai.request.* (parameters) ---
    params: Optional[Mapping[str, Any]] = None
    raw_params = payload.get("parameters")
    if isinstance(raw_params, Mapping):
        params = raw_params

    def _request_param(key: str) -> Any:
        """Look up a parameter from request_kwargs, then from the
        payload's ``parameters`` dict, then from the payload itself.
        Mirrors how every adapter populates ``parameters``.
        """
        if request_kwargs is not None and key in request_kwargs:
            return request_kwargs[key]
        if params is not None and key in params:
            return params[key]
        return payload.get(key)

    max_tokens = _request_param("max_tokens")
    if max_tokens is not None:
        payload[GEN_AI_REQUEST_MAX_TOKENS] = max_tokens

    temperature = _request_param("temperature")
    if temperature is not None:
        payload[GEN_AI_REQUEST_TEMPERATURE] = temperature

    top_p = _request_param("top_p")
    if top_p is not None:
        payload[GEN_AI_REQUEST_TOP_P] = top_p

    top_k = _request_param("top_k")
    if top_k is not None:
        payload[GEN_AI_REQUEST_TOP_K] = top_k

    freq_penalty = _request_param("frequency_penalty")
    if freq_penalty is not None:
        payload[GEN_AI_REQUEST_FREQUENCY_PENALTY] = freq_penalty

    pres_penalty = _request_param("presence_penalty")
    if pres_penalty is not None:
        payload[GEN_AI_REQUEST_PRESENCE_PENALTY] = pres_penalty

    stop = _request_param("stop") or _request_param("stop_sequences")
    if stop is not None:
        if isinstance(stop, str):
            payload[GEN_AI_REQUEST_STOP_SEQUENCES] = [stop]
        elif isinstance(stop, (list, tuple)):
            payload[GEN_AI_REQUEST_STOP_SEQUENCES] = [str(s) for s in stop if s is not None]

    seed = _request_param("seed")
    if seed is not None:
        payload[GEN_AI_REQUEST_SEED] = seed

    n_choices = _request_param("n")
    if n_choices is not None:
        payload[GEN_AI_REQUEST_CHOICE_COUNT] = n_choices

    encoding_formats = _request_param("encoding_format")
    if encoding_formats is not None:
        if isinstance(encoding_formats, str):
            payload[GEN_AI_REQUEST_ENCODING_FORMATS] = [encoding_formats]
        elif isinstance(encoding_formats, (list, tuple)):
            payload[GEN_AI_REQUEST_ENCODING_FORMATS] = [str(e) for e in encoding_formats]

    # --- OpenAI-specific request attributes ---
    if detected_system in (SYSTEM_OPENAI, SYSTEM_AZURE_OPENAI):
        service_tier = _request_param("service_tier")
        if service_tier is not None:
            payload[GEN_AI_OPENAI_REQUEST_SERVICE_TIER] = service_tier
        response_format = _request_param("response_format")
        if response_format is not None:
            # The OTel spec records response_format as a string. If a
            # dict is provided (the common ``{"type": "json_object"}``
            # shape), surface only the ``type`` so cardinality stays
            # bounded and the value is human-readable.
            if isinstance(response_format, Mapping):
                rf_type = response_format.get("type")
                if isinstance(rf_type, str):
                    payload[GEN_AI_OPENAI_REQUEST_RESPONSE_FORMAT] = rf_type
            elif isinstance(response_format, str):
                payload[GEN_AI_OPENAI_REQUEST_RESPONSE_FORMAT] = response_format

    # --- AWS Bedrock-specific request attributes ---
    if detected_system == SYSTEM_AWS_BEDROCK and request_kwargs is not None:
        guardrail_cfg = request_kwargs.get("guardrailConfig") or request_kwargs.get(
            "guardrail_config"
        )
        if isinstance(guardrail_cfg, Mapping):
            guardrail_id = guardrail_cfg.get("guardrailIdentifier") or guardrail_cfg.get(
                "guardrail_id"
            )
            if guardrail_id:
                payload[AWS_BEDROCK_GUARDRAIL_ID] = guardrail_id
        kb_id = request_kwargs.get("knowledgeBaseId") or request_kwargs.get("knowledge_base_id")
        if kb_id:
            payload[AWS_BEDROCK_KNOWLEDGE_BASE_ID] = kb_id
        agent_id = request_kwargs.get("agentId") or request_kwargs.get("agent_id")
        if agent_id:
            payload[AWS_BEDROCK_AGENT_ID] = agent_id

    # --- gen_ai.usage.* (input / output tokens) ---
    prompt_tokens = (
        payload.get("prompt_tokens")
        if payload.get("prompt_tokens") is not None
        else payload.get("tokens_prompt")
    )
    if prompt_tokens is None:
        # langchain nests usage in token_usage dict
        token_usage = payload.get("token_usage")
        if isinstance(token_usage, Mapping):
            prompt_tokens = token_usage.get("prompt_tokens") or token_usage.get("input_tokens")
    if prompt_tokens is not None:
        payload[GEN_AI_USAGE_INPUT_TOKENS] = prompt_tokens

    completion_tokens = (
        payload.get("completion_tokens")
        if payload.get("completion_tokens") is not None
        else payload.get("tokens_completion")
    )
    if completion_tokens is None:
        token_usage = payload.get("token_usage")
        if isinstance(token_usage, Mapping):
            completion_tokens = token_usage.get("completion_tokens") or token_usage.get(
                "output_tokens"
            )
    if completion_tokens is not None:
        payload[GEN_AI_USAGE_OUTPUT_TOKENS] = completion_tokens

    # --- gen_ai.response.* ---
    response_id = payload.get("response_id")
    if response_id is None and response_obj is not None:
        response_id = _safe_get(response_obj, "id")
    if response_id is not None:
        payload[GEN_AI_RESPONSE_ID] = response_id

    response_model = payload.get("response_model")
    if response_model is None and response_obj is not None:
        response_model = _safe_get(response_obj, "model")
    if response_model is not None:
        payload[GEN_AI_RESPONSE_MODEL] = response_model

    finish_raw = payload.get("finish_reason") or payload.get("finish_reasons")
    if finish_raw is None and response_obj is not None:
        finish_raw = _safe_get(response_obj, "finish_reason") or _safe_get(
            response_obj, "stop_reason"
        )
    finish = _coerce_finish_reasons(finish_raw)
    if finish:
        payload[GEN_AI_RESPONSE_FINISH_REASONS] = finish

    # --- Anthropic-specific response attributes ---
    if detected_system == SYSTEM_ANTHROPIC:
        cache_create = payload.get("cache_creation_input_tokens")
        if cache_create is None and response_obj is not None:
            usage = _safe_get(response_obj, "usage")
            if usage is not None:
                cache_create = _safe_get(usage, "cache_creation_input_tokens")
        if cache_create is not None:
            payload[GEN_AI_ANTHROPIC_CACHE_CREATION_INPUT_TOKENS] = cache_create

        cache_read = payload.get("cache_read_input_tokens")
        if cache_read is None and response_obj is not None:
            usage = _safe_get(response_obj, "usage")
            if usage is not None:
                cache_read = _safe_get(usage, "cache_read_input_tokens")
        if cache_read is not None:
            payload[GEN_AI_ANTHROPIC_CACHE_READ_INPUT_TOKENS] = cache_read

    # --- OpenAI-specific response attributes ---
    if detected_system in (SYSTEM_OPENAI, SYSTEM_AZURE_OPENAI):
        sys_fp = payload.get("system_fingerprint")
        if sys_fp is None and response_obj is not None:
            sys_fp = _safe_get(response_obj, "system_fingerprint")
        if sys_fp is not None:
            payload[GEN_AI_OPENAI_RESPONSE_SYSTEM_FINGERPRINT] = sys_fp

        resp_tier = payload.get("service_tier")
        if resp_tier is None and response_obj is not None:
            resp_tier = _safe_get(response_obj, "service_tier")
        if resp_tier is not None:
            payload[GEN_AI_OPENAI_RESPONSE_SERVICE_TIER] = resp_tier

    return payload


def _safe_get(obj: Any, attr: str) -> Any:
    """Read ``attr`` from a Pydantic-like object OR a Mapping.

    Returns ``None`` on any failure — the helper must never raise on
    partial / unexpected response shapes.
    """
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(attr)
    try:
        return getattr(obj, attr, None)
    except Exception:
        return None


__all__ = [
    # System / operation enums
    "OPERATION_CHAT",
    "OPERATION_EMBED",
    "OPERATION_GENERATE_CONTENT",
    "OPERATION_TEXT_COMPLETION",
    "SYSTEM_ANTHROPIC",
    "SYSTEM_AWS_BEDROCK",
    "SYSTEM_AZURE_OPENAI",
    "SYSTEM_COHERE",
    "SYSTEM_FALLBACK_OTHER",
    "SYSTEM_GCP_GEMINI",
    "SYSTEM_GCP_VERTEX",
    "SYSTEM_LITELLM",
    "SYSTEM_MISTRAL",
    "SYSTEM_OLLAMA",
    "SYSTEM_OPENAI",
    # Attribute keys
    "AWS_BEDROCK_AGENT_ID",
    "AWS_BEDROCK_GUARDRAIL_ID",
    "AWS_BEDROCK_KNOWLEDGE_BASE_ID",
    "GEN_AI_AGENT_DESCRIPTION",
    "GEN_AI_AGENT_ID",
    "GEN_AI_AGENT_NAME",
    "GEN_AI_ANTHROPIC_CACHE_CREATION_INPUT_TOKENS",
    "GEN_AI_ANTHROPIC_CACHE_READ_INPUT_TOKENS",
    "GEN_AI_GOOGLE_SAFETY_RATINGS",
    "GEN_AI_OPENAI_REQUEST_RESPONSE_FORMAT",
    "GEN_AI_OPENAI_REQUEST_SERVICE_TIER",
    "GEN_AI_OPENAI_RESPONSE_SERVICE_TIER",
    "GEN_AI_OPENAI_RESPONSE_SYSTEM_FINGERPRINT",
    "GEN_AI_OPERATION_NAME",
    "GEN_AI_PROVIDER_NAME",
    "GEN_AI_REQUEST_CHOICE_COUNT",
    "GEN_AI_REQUEST_ENCODING_FORMATS",
    "GEN_AI_REQUEST_FREQUENCY_PENALTY",
    "GEN_AI_REQUEST_MAX_TOKENS",
    "GEN_AI_REQUEST_MODEL",
    "GEN_AI_REQUEST_PRESENCE_PENALTY",
    "GEN_AI_REQUEST_SEED",
    "GEN_AI_REQUEST_STOP_SEQUENCES",
    "GEN_AI_REQUEST_TEMPERATURE",
    "GEN_AI_REQUEST_TOP_K",
    "GEN_AI_REQUEST_TOP_P",
    "GEN_AI_RESPONSE_FINISH_REASONS",
    "GEN_AI_RESPONSE_ID",
    "GEN_AI_RESPONSE_MODEL",
    "GEN_AI_SYSTEM",
    "GEN_AI_TOOL_CALL_ID",
    "GEN_AI_TOOL_DESCRIPTION",
    "GEN_AI_TOOL_NAME",
    "GEN_AI_TOOL_TYPE",
    "GEN_AI_USAGE_INPUT_TOKENS",
    "GEN_AI_USAGE_OUTPUT_TOKENS",
    # Functions
    "detect_gen_ai_system",
    "detect_operation",
    "stamp_genai_attributes",
]
