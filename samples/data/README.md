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
| `traces/simple_llm_trace.json` | A single LLM call trace (OpenAI support assistant) with prompt, completion, and token usage. The simplest trace format for getting started. |
| `traces/rag_pipeline_trace.json` | A RAG pipeline trace (LangChain) with retrieval and LLM spans, including document references. Useful for testing multi-span evaluation. |
| `traces/multi_agent_trace.json` | A multi-agent trace (CrewAI) with sequential researcher, analyst, and writer collaboration spans. |
| `traces/error_trace.json` | A failed trace containing a TimeoutError after retry attempts. Useful for testing error handling and investigation workflows. |
| `traces/example_traces.jsonl` | A collection of example traces in JSONL format for batch processing samples. |
| `traces/batch_traces.jsonl` | Ten traces across multiple frameworks, models, and statuses. Designed for batch ingestion testing. |

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
