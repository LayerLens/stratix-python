# OpenInference Python↔Go conformance corpus

The OpenInference span→event mapping exists in TWO languages, and an OpenInference
trace must render identically whichever way it arrived:

| | |
|---|---|
| Python (this repo) | `src/layerlens/instrument/adapters/frameworks/openinference.py` — the SDK arrival path |
| Go (atlas) | `atlas-app: apps/otlp-ingest/ingest/openinference.go` — the OTLP arrival path |

`../test_openinference_conformance.py` pins them against each other over the files
here. It follows the proven dual-oracle pattern from `tests/e2e/live/graph_contract/`
(fixtures + an oracle generated from the REAL Go engine, committed here for a
deterministic, stack-free lane).

## Files

- **`spans.otlp.json`** — the shared corpus: a proto3-JSON `ExportTraceServiceRequest`
  with 24 spans, one per mapping lane (all 9 span kinds, the never-drop UNKNOWN
  default, every fallback, the content caps, the honest-omission branches). Fed
  VERBATIM to both languages.
- **`oracle.json`** — the Go bridge's REAL output over that corpus (26 events; AGENT
  and CHAIN each emit an `agent.input`/`agent.output` pair). **Generated from the Go
  code, never hand-written.**
- **`_generate_corpus.py`** — the corpus generator (deterministic; run it twice, get
  byte-identical output). This is the single authoring source for the corpus.

## Honest note on what the corpus is

The spans are **SYNTHETIC, deliberately-constructed contract inputs — not a
recording of a production run**, and must never be presented as one. Each lane
exercises one branch of the mapping, including branches a real trace shows only
rarely (GUARDRAIL, EVALUATOR, RERANKER) or only under failure.

The lane is still honest, because the **oracle is generated from the real Go
bridge** and the assertion is agreement between two REAL implementations over a
stated input. Nothing here is presented as measured production data. The attribute
keys — and the LLM lane's values — come from the real OpenInference semantic
conventions as exercised by the real wire fixture at
`atlas-app: apps/otlp-ingest/ingest/testdata/otlp-fixtures/03-openinference-arize.otlp.json`.

## Why drift on EITHER side is caught

- **Python drift** → `test_openinference_conformance.py` fails here (normal pytest,
  no live stack, runs in SDK CI).
- **Go drift** → `TestOpenInferenceConformanceOracleIsCurrent` fails in atlas CI. It
  asserts the committed oracle still equals what the Go bridge emits today. Without
  it, a Go change would leave a stale oracle that this side still happily matched.
- **Corpus edited without regenerating** → `corpus_sha256` in `oracle.json` fails
  both sides, instead of silently comparing against a transcript of different spans.

## Changing the mapping

The two languages move together. After an intended change:

1. change both implementations,
2. regenerate the oracle:
   `cd atlas-app/apps && OI_CONFORMANCE_REGEN=1 go test ./otlp-ingest/ingest/ -run TestDumpOpenInferenceConformance -count=1 -v`
3. copy `spans.otlp.json` + `oracle.json` back here,
4. run both suites.

If the corpus itself changes, regenerate it with `_generate_corpus.py`, copy it to
`atlas-app/apps/otlp-ingest/ingest/testdata/oi-conformance/`, then redo steps 2–4.

## Known, deliberate divergence (D1) — the duration field name

Go emits `duration_ms`; Python emits `latency_ms`. `duration_ms` is the OTLP path's
platform-wide convention (`convert.go`, `merge.go`, `writer.go`, `openinference.go`;
every OTLP golden uses it, none uses `latency_ms`), while `latency_ms` is the SDK
canon (`tests/instrument/_event_schema.py`). Renaming either side alone would break
it against its OWN siblings, so this is **documented and deliberately left**. The
lane exempts the NAME and still pins the VALUE, so the exemption cannot hide a real
timing divergence. See `_KNOWN_EXCEPTIONS` in the test for the full, justified list.
