# Modality Samples

AI applications produce content in many forms -- text, structured documents, brand-sensitive
marketing copy -- and each form demands specialized evaluation criteria. These samples
demonstrate how to apply modality-specific judges that go beyond generic quality scoring to
assess the unique attributes of each content type: factual accuracy for text, brand alignment
for marketing content, and structural integrity for extracted documents.

## Prerequisites

```bash
pip install layerlens --index-url https://sdk.layerlens.ai/package
export LAYERLENS_STRATIX_API_KEY=your-api-key
```

## Quick Start

Start with `text_evaluation.py` for the most common modality. It applies safety, relevance,
and factual accuracy judges to a text completion:

```bash
python text_evaluation.py
```

Expected output: per-judge scores and a pass/fail verdict for each evaluation dimension.

## Samples

| File | Scenario | Description |
|------|----------|-------------|
| `text_evaluation.py` | Content teams validating chatbot or assistant responses | Evaluates text completions against safety, relevance, and factual accuracy judges. Suitable as a baseline for any text-generating application. |
| `brand_evaluation.py` | Marketing teams enforcing brand guidelines at scale | Evaluates content against brand voice, tone, and visual identity criteria. Useful for organizations that require consistent messaging across AI-generated outputs. |
| `document_evaluation.py` | Data engineering teams validating document pipelines | Evaluates document extraction accuracy, field completeness, and structural integrity. Applies to OCR, PDF parsing, and other document-processing workflows. |

## Expected Behavior

Each sample creates a trace representing the modality-specific content, applies the relevant
judges, and prints a scored summary. Brand and document evaluations will produce dimension-level
breakdowns in addition to the aggregate score.
