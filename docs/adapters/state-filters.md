# Adapter State Filters

Framework adapters emit dict-shaped state into trace events
(`agent.input`, `agent.output`, `tool.call`, `tool.result`,
`model.invoke`, etc.). Without filtering, that state can carry
credentials, PII, or unbounded cardinality straight into telemetry
sinks. The state-filter subsystem is the last line of defence between
user state and the wire.

## Default behaviour

Every multi-agent framework adapter ships with a conservative default
filter — `default_state_filter()` — that excludes a denylist of common
PII and credential field names. Customers who do nothing still get
baseline protection out of the box.

The default denylist (case-insensitive substring match) covers:

- **Credentials**: `password`, `passwd`, `pwd`, `api_key`, `apikey`,
  `api_secret`, `secret`, `secret_key`, `access_token`,
  `refresh_token`, `auth_token`, `bearer_token`, `token`,
  `session_token`, `cookie`, `cookies`, `private_key`,
  `client_secret`, `service_account`
- **Personal identifiers**: `ssn`, `social_security`,
  `social_security_number`, `tax_id`, `national_id`, `passport`,
  `passport_number`, `drivers_license`
- **Financial**: `credit_card`, `credit_card_number`, `card_number`,
  `cvv`, `cvc`, `iban`, `account_number`, `routing_number`
- **Contact / location**: `email`, `email_address`, `phone`,
  `phone_number`, `address`, `street_address`, `home_address`,
  `billing_address`, `shipping_address`
- **Authn material**: `authorization`, `x-api-key`, `set-cookie`

Substring matching after non-alphanumeric normalisation means
`X-Api-Key`, `stripe_customer_email`, `USER_API_KEY`, and
`customer.email_address` all match without the caller having to
enumerate every variant.

## Configuration

Pass a `StateFilter` instance to the adapter constructor:

```python
from layerlens.instrument.adapters._base import StateFilter
from layerlens.instrument.adapters.frameworks.agno import AgnoAdapter

# Add custom keys to the default denylist
filter = StateFilter.with_extra_excludes(["internal_user_id", "session_attributes"])
adapter = AgnoAdapter(client, state_filter=filter)
```

### Three filter operations

A `StateFilter` declares three operations, applied in this order:

1. **`exclude_keys`** — keys (case-insensitive substring match)
   removed from the output entirely. Defaults to the PII denylist.
2. **`mask_keys`** — keys (case-insensitive substring match) whose
   values are replaced with `[REDACTED]`. The key remains visible (so
   dashboards see the field exists), but the value is hidden. Default
   empty — opt-in.
3. **`include_keys`** — if non-empty, restricts the output to ONLY
   these keys (case-insensitive equality). Acts as a strict
   allowlist after exclude/mask have run.

```python
filter = StateFilter(
    exclude_keys=frozenset({"password", "ssn"}),       # drop entirely
    mask_keys=frozenset({"phone", "address"}),         # show key, hide value
    include_keys=frozenset({"name", "phone", "model"}),  # allowlist
)
```

### Disabling the filter

For tests or explicit opt-out, use `StateFilter.permissive()`:

```python
adapter = AgnoAdapter(client, state_filter=StateFilter.permissive())
```

This is **strongly discouraged** in production. The active filter is
surfaced in `adapter.adapter_info().metadata['state_filter']` so
operators can detect accidental disablement.

## Recursion

By default, the filter walks nested dicts (and dicts inside lists)
recursively, so a structure like:

```python
{
    "messages": [
        {"role": "user", "content": "hi", "api_key": "sk-..."},
    ],
}
```

emits as:

```python
{
    "messages": [
        {"role": "user", "content": "hi"},
    ],
    "_filtered_keys": ["api_key"],
}
```

Set `recursive=False` to filter only the top-level dict.

## Auditability

Every event payload that has been touched by the filter carries a
`_filtered_keys` field listing the (lowercased) names of every key
that was excluded or masked anywhere in the payload. Operators can
correlate this with the active filter config (in
`adapter_info().metadata['state_filter']`) to verify exactly what was
clipped without exposing the values themselves.

## Replay reproducibility

The active filter snapshot is included under
`serialize_state_filter_for_replay()` so the replay engine can
reconstruct an equivalent `StateFilter` on the other side. Replays
must apply the SAME filter as the original run so the captured payload
shapes match.

## Multi-agent adapters with state-filter wiring

| Adapter        | Status   | Notes                                                  |
|----------------|----------|--------------------------------------------------------|
| `agno`         | Wired    | Filter applied on `agent.input`/`output`/`tool.*`      |
| `openai_agents`| Wired    | Filter on `agent.input`, `tool.call/result`, generation messages |
| `llamaindex`   | Wired    | Filter on LLM messages, retrieval, query, agent_step  |
| `google_adk`   | Wired    | Filter on user_content (agent + tool input/output)    |
| `strands`      | Wired    | Filter on invocation messages + tool input/output     |
| `pydantic_ai`  | Wired    | Filter on agent input/output, deps_summary, tool args |

For mature adapters (LangChain, LangGraph, CrewAI, AutoGen,
Agentforce, Semantic Kernel) state filtering is performed at the
framework-native layer (e.g. LangGraph's `LangGraphStateAdapter`
include/exclude keys) — the cross-pollination here brings the same
contract to the lighter multi-agent adapters.

## Reference

- Implementation:
  `src/layerlens/instrument/adapters/_base/state_filters.py`
- Adapter-base wiring:
  `src/layerlens/instrument/adapters/frameworks/_base_framework.py`
- Tests: `tests/instrument/adapters/_base/test_state_filters.py`
- See also: [Data Privacy](../security/data-privacy.md)
