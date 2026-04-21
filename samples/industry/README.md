# Industry Samples

Regulated and domain-critical AI applications require evaluation criteria that go far beyond
generic quality checks. A healthcare chatbot must be medically accurate. A trading assistant
must comply with fiduciary obligations. A government service must be accessible and equitable.
These samples demonstrate how to build industry-specific evaluation pipelines using judges
tailored to the compliance, safety, and accuracy requirements of each vertical.

## Prerequisites

```bash
pip install layerlens --index-url https://sdk.layerlens.ai/package
export LAYERLENS_STRATIX_API_KEY=your-api-key
```

Industry samples reference domain-specific test data located in `samples/data/industry/`.

## Quick Start

Start with `financial_fraud.py` for a representative example of domain-specific evaluation:

```bash
python financial_fraud.py
```

Expected output: risk scores, AML pattern detection results, and compliance verdicts for
each evaluated transaction trace.

## Samples

### Financial Services

| File | Scenario | Description |
|------|----------|-------------|
| `financial_fraud.py` | Fraud analysts validating detection models | Risk scoring accuracy and anti-money-laundering pattern detection against labeled transaction data. |
| `financial_trading.py` | Compliance officers auditing trading assistants | SOX suitability checks, fiduciary duty evaluation, and regulatory compliance for AI-assisted trading recommendations. |

### Healthcare

| File | Scenario | Description |
|------|----------|-------------|
| `healthcare_clinical.py` | Clinical informatics teams deploying decision support | Medical accuracy, drug interaction detection, and guideline adherence for clinical AI outputs. |

### Insurance

| File | Scenario | Description |
|------|----------|-------------|
| `insurance_claims.py` | Claims adjusters validating AI-assisted processing | Coverage determination accuracy and settlement fairness evaluation for automated claims workflows. |
| `insurance_underwriting.py` | Underwriting teams auditing risk models | Risk assessment accuracy and fair lending compliance for AI-driven underwriting decisions. |

### Legal

| File | Scenario | Description |
|------|----------|-------------|
| `legal_contracts.py` | Legal teams reviewing AI-assisted contract analysis | Clause detection accuracy, risk flag identification, and obligation extraction for contract review tools. |
| `legal_research.py` | Attorneys validating research assistants | Citation accuracy, jurisdictional correctness, and precedent relevance for legal research AI. |

### Government

| File | Scenario | Description |
|------|----------|-------------|
| `government_citizen.py` | Public sector teams deploying citizen-facing AI | Regulatory accuracy, accessibility compliance, equity assessment, and plain-language evaluation for government services. |

### Retail

| File | Scenario | Description |
|------|----------|-------------|
| `retail_recommender.py` | Product teams auditing recommendation engines | Recommendation relevance, safety filtering, and bias detection for AI-powered product suggestions. |
| `retail_support.py` | Customer experience teams evaluating support bots | Response accuracy, tone appropriateness, and resolution quality for AI customer service agents. |

## Expected Behavior

Each sample loads domain-specific test data, creates traces representing AI interactions in
that vertical, and evaluates them with industry-appropriate judges. Results include per-criterion
scores and compliance verdicts relevant to the regulatory framework of each domain.
