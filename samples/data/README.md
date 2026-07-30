# Sample Data

The SDK samples reference pre-built trace files, evaluation datasets, and industry-specific
test data. This directory provides all the data needed to run the samples without connecting
to a live AI provider. Use these files to test SDK operations locally, seed your LayerLens
workspace with representative data, or build automated test suites for your evaluation pipelines.

## Prerequisites

No additional dependencies are required. The data files are consumed by samples in other
directories via relative path references.

## Quick Start

Upload a trace file to your LayerLens workspace:

```bash
export LAYERLENS_STRATIX_API_KEY=your-api-key
python -c "from layerlens import Stratix; Stratix().traces.upload('samples/data/traces/simple_llm_trace.json')"
```

Expected output: the trace ID of the newly uploaded trace record.

## Traces

| File | Description |
|------|-------------|
| `traces/simple_llm_trace.json` | A single-agent OpenAI support-assistant trace with one LLM call and a cost record. The simplest structured trace for getting started. |
| `traces/rag_pipeline_trace.json` | A LangChain retrieval-augmented generation trace with retriever, reranker, and answer-synthesis agents handed off across multiple spans. |
| `traces/multi_agent_trace.json` | A CrewAI multi-agent trace where a researcher, fact-verifier, and analyst collaborate on an executive brief with peer review and one corrected error. |
| `traces/error_trace.json` | A failed LangChain trace covering context-length errors, rate-limit retries with exponential backoff, fallback agent handoff, and final failure with diagnostic guidance. |
| `traces/example_traces.jsonl` | A collection of example traces in JSONL format for batch processing samples. |
| `traces/batch_traces.jsonl` | Fifty structured traces across multiple frameworks, models, and statuses. Designed for batch ingestion testing. |

## Datasets

| File | Description |
|------|-------------|
| `datasets/golden_test_set.jsonl` | Ten curated question-answer pairs for evaluation and regression testing. Each entry includes an expected answer for judge validation. |
| `datasets/generic_qa.jsonl` | A larger QA dataset spanning factual, reasoning, analytical, and creative categories. Suitable for benchmark runs and model comparison. |

## Industry Data

Domain-specific evaluation datasets with expected outcomes for judge testing. Each file is
referenced by the corresponding sample in `samples/industry/`.

| File | Domain |
|------|--------|
| `industry/education_essays.jsonl` | Education -- student essays with grading rubrics |
| `industry/healthcare_patient_cases.jsonl` | Healthcare -- patient cases with expected diagnoses |
| `industry/healthcare_triage.jsonl` | Healthcare -- emergency triage with acuity levels |
| `industry/financial_loans.jsonl` | Finance -- loan applications with risk ratings |
| `industry/financial_transactions.jsonl` | Finance -- transactions with fraud indicators |
| `industry/legal_contracts.jsonl` | Legal -- contract clauses with risk assessments |
| `industry/legal_research.jsonl` | Legal -- research documents with analysis |
| `industry/insurance_claims.jsonl` | Insurance -- claims processing data |
| `industry/government_eligibility.jsonl` | Government -- eligibility determination cases |
| `industry/retail_products.jsonl` | Retail -- product recommendations with user profiles |
| `industry/energy_grid.jsonl` | Energy -- grid performance and diagnostics |
| `industry/manufacturing_equipment.jsonl` | Manufacturing -- predictive maintenance data |
| `industry/media_moderation.jsonl` | Media -- content moderation decisions |
| `industry/real_estate_listings.jsonl` | Real estate -- property listings with valuations |
| `industry/telecom_interactions.jsonl` | Telecom -- customer service interactions |
| `industry/travel_bookings.jsonl` | Travel -- booking transactions with preferences |

## Recorded real traces (`traces/industry/`, `traces/cowork/`)

The industry and co-work samples upload **recorded real traces** rather than
hand-authored stubs. Each fixture is a JSONL file (one trace per line) captured
from a genuine, instrumented agent run over that sample's scenarios: real
`model.invoke`/`cost.record` events, a real `agent.identity`, and an intact
attestation chain. Because the data is real, the LayerLens UI renders the Agent,
Framework, and Status columns from actual producer values -- the Framework
column shows the provider that really ran (`openai`, `anthropic`, or `ollama`),
not a fabricated label.

Each `traces/industry/<sample>.jsonl` / `traces/cowork/<sample>.jsonl` file is
consumed by the matching sample in `samples/industry/` / `samples/cowork/` via
`_helpers.upload_recorded_trace`. The one exception is
`traces/cowork/incident_response.jsonl`, which mixes real recorded traces with a
small set of clearly-labeled synthetic adversarial entries (`metadata.synthetic`
= true) -- unsafe outputs a real aligned model refuses to produce, kept so the
Safety judge has known-bad inputs to flag.

To regenerate the fixtures from the domain scenarios, run:

```bash
export LAYERLENS_STRATIX_API_KEY=your-api-key
export OPENAI_API_KEY=...  ANTHROPIC_API_KEY=...   # + a local Ollama
python samples/data/_generate_fixtures.py
```

`_generate_fixtures.py` runs each scenario through a real instrumented model
call and captures the resulting trace **without uploading** it, so regenerating
never pollutes your workspace.
