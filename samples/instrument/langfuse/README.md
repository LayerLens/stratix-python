# Langfuse sample

Runnable end-to-end sample for the
`layerlens.instrument.adapters.frameworks.langfuse` adapter.

The sample is **fully mocked** — every call to the Langfuse HTTP API is
intercepted in-process. It makes no network calls and requires no
Langfuse credentials. It exists to demonstrate the adapter's API
surface and act as a smoke test that the `[langfuse-importer]` extra
installs cleanly.

## Install

```bash
pip install 'layerlens[langfuse-importer]'
```

The `[langfuse-importer]` extra is intentionally empty — the Langfuse
adapter talks to a remote REST surface and uses only `urllib` from the
Python stdlib. No additional dependencies are pulled in.

## Run

```bash
python -m samples.instrument.langfuse.main
```

You should see three labeled flows print to stdout:

* `[import]` — Backfill two synthetic Langfuse traces (one with a
  `GENERATION` observation, one with a `TOOL` `SPAN`) into LayerLens
  canonical events. Prints the per-event-type histogram.
* `[export]` — Push one synthetic LayerLens trace
  (`agent.input` + `model.invoke` + `cost.record` + `agent.output`)
  back into Langfuse via the batch ingestion endpoint and confirms
  the loop-prevention tags (`layerlens-exported`, `stratix-exported`).
* `[bidirectional]` — Run a dry-run `sync()` in `BIDIRECTIONAL` mode.

The sample exits 0 on success.

## Live Langfuse smoke (optional)

If you have a Langfuse instance, set these environment variables before
running and the sample will additionally exercise a single live
`connect()` against the API after the mocked flows complete:

```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_HOST="https://cloud.langfuse.com"   # default
```

The live smoke only runs the health check that `connect()` performs;
no traces are imported or exported against the live instance.

## See also

`docs/adapters/frameworks-langfuse.md` — install + usage guide,
import/export pipeline architecture, capability matrix, version
compatibility, and the `STRATIX → LayerLens` deprecation aliases.
