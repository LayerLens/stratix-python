# LayerLens SDK Samples

Production-ready code samples that show you how to evaluate, score, and govern AI outputs at every stage of your development lifecycle -- from a single trace to enterprise-wide compliance pipelines.

Whether you are shipping your first LLM feature or operating hundreds of models in regulated industries, these samples give you a working starting point you can run today and adapt for production tomorrow.

---

## Getting Started

Three steps to your first evaluation:

**1. Install the SDK**

```bash
pip install layerlens --index-url https://sdk.layerlens.ai/package
```

**2. Set your API key**

```bash
export LAYERLENS_STRATIX_API_KEY=your-api-key
```

**3. Run the quickstart**

```bash
python samples/core/quickstart.py
```

`quickstart.py` walks through the complete workflow end-to-end: upload a trace, create a judge, run an evaluation, and retrieve results. Everything else in this repository builds on that foundation.

---

## Choose Your Path

Different roles need different entry points. Start with the path that matches your goal.

### New to LayerLens

Run `quickstart.py`, then explore the `core/` samples in order. You will learn how traces, judges, evaluations, and results fit together, and how to work with them using both synchronous and asynchronous calls.

### Platform Engineer

Start with `cicd/` to integrate evaluation gates into your deployment pipeline. Then review `cowork/` for multi-agent evaluation patterns (usable with [Claude Cowork](https://claude.com/product/cowork), [Agent Teams](https://code.claude.com/docs/en/agent-teams), or any framework) and `mcp/` to expose LayerLens as a tool server for AI assistants.

### Enterprise Evaluator

Go directly to `industry/` for domain-specific evaluation scenarios in healthcare, financial services, legal, government, insurance, and retail. Each sample includes the judges and scoring criteria that matter in regulated environments.

### Framework Integrator

See `integrations/` for provider-level tracing with OpenAI and Anthropic. Then explore `copilotkit/` for CopilotKit CoAgent patterns and `openclaw/` for OpenClaw agent integration with advanced evaluation patterns including model tournaments, safety audits, and red-teaming.

---

## Samples by Use Case

### Core SDK Operations -- `core/` (18 samples)

Master the building blocks of trace management, judge creation, evaluation execution, and result analysis.

**What you will learn:** How every LayerLens resource (traces, judges, evaluations, results, models, benchmarks) works individually and together, including async patterns and pagination.

| Sample | Scenario |
|--------|----------|
| `quickstart.py` | Run your first end-to-end evaluation in under 30 lines of code |
| `basic_trace.py` | Upload, list, retrieve, and delete traces to manage your evaluation corpus |
| `run_evaluation.py` | Create a model evaluation and poll for completion to automate scoring |
| `create_judge.py` | Define, read, update, and delete judges to codify your quality criteria |
| `judge_creation_and_test.py` | Build a custom PII detection judge and validate it against known inputs |
| `trace_evaluation.py` | Connect a trace to a judge and run a complete evaluation workflow |
| `benchmark_evaluation.py` | Run a model against a benchmark, wait for completion, retrieve scored results |
| `trace_investigation.py` | Surface errors, latency outliers, and anomalies in trace data |
| `evaluation_pipeline.py` | Chain judges, traces, and result retrieval into a single automated pipeline |
| `judge_optimization.py` | Estimate, execute, and apply judge optimizations to improve scoring accuracy |
| `compare_evaluations.py` | Compare evaluation runs side-by-side to measure improvement over time |
| `async_workflow.py` | Use AsyncStratix for concurrent operations when throughput matters |
| `model_benchmark_management.py` | Register models and benchmarks to organize large-scale evaluations |
| `public_catalog.py` | Browse the public catalog of models, benchmarks, and community evaluations |
| `custom_model.py` | Register your own models via OpenAI-compatible endpoints for evaluation |
| `custom_benchmark.py` | Create custom and smart benchmarks tailored to your domain |
| `evaluation_filtering.py` | Filter and sort evaluations by status, accuracy, and date to find what matters |
| `paginated_results.py` | Handle large result sets with manual and automatic pagination |
| `async_results.py` | Fetch results from multiple concurrent evaluations without blocking |

---

### Content-Type Evaluations -- `modalities/` (3 samples)

Apply specialized judges to different content types -- text responses, brand assets, and structured documents.

**What you will learn:** How to configure judges for content-specific quality dimensions such as safety, brand voice consistency, and document extraction accuracy.

| Sample | Scenario |
|--------|----------|
| `text_evaluation.py` | Score text outputs across five dimensions: safety, relevance, helpfulness, factual accuracy, and compliance |
| `brand_evaluation.py` | Enforce brand voice consistency and visual identity compliance across generated content |
| `document_evaluation.py` | Validate extraction accuracy, cross-field consistency, and structural integrity in document processing |

---

### CI/CD Integration -- `cicd/` (2 samples + workflow)

Embed evaluation quality gates into your build and deployment pipelines so regressions never reach production.

**What you will learn:** How to set pass-rate thresholds that block deployments, run smoke-test evaluations on every commit, and integrate with GitHub Actions.

| Sample | Scenario |
|--------|----------|
| `quality_gate.py` | Gate deployments on evaluation pass rates so only quality-approved models ship |
| `pre_commit_hook.py` | Catch evaluation regressions at commit time before they enter the review cycle |
| `github_actions_gate.yml` | Drop-in GitHub Actions workflow that runs evaluation gates on every pull request |

---

### LLM Provider Integrations -- `integrations/` (4 samples)

Trace and evaluate outputs from major LLM providers -- both manual trace upload and auto-instrumentation.

**What you will learn:** How to capture provider API calls with LayerLens tracing using two approaches: manual trace upload for full control, and auto-instrumentation via `layerlens.instrument` for zero-code observability.

| Sample | Scenario |
|--------|----------|
| `openai_traced.py` | Trace an OpenAI completion and evaluate it with a judge (manual trace upload) |
| `anthropic_traced.py` | Capture multi-turn Claude conversations with tracing and evaluation (manual trace upload) |
| `openai_instrumented.py` | Auto-instrument OpenAI with `instrument_openai()`, `@trace`, and `span()` for zero-code span capture |
| `langchain_instrumented.py` | Auto-capture LangChain LCEL chain execution with `LangChainCallbackHandler` |

---

### Industry Solutions -- `industry/` (10 samples)

Domain-specific evaluation scenarios with judges tuned for regulated and high-stakes verticals.

**What you will learn:** How to apply evaluation criteria that reflect real regulatory and operational requirements -- from HIPAA-adjacent clinical checks to SOX trading compliance and fair lending assessments.

| Sample | Scenario |
|--------|----------|
| `healthcare_clinical.py` | Evaluate clinical decision support for diagnostic accuracy, triage correctness, and drug interaction safety |
| `financial_fraud.py` | Score fraud detection and anti-money-laundering pattern analysis for accuracy and coverage |
| `financial_trading.py` | Enforce SOX-aligned trading compliance including suitability, disclosure, and audit readiness |
| `legal_contracts.py` | Assess contract review quality for clause detection, risk identification, and obligation extraction |
| `legal_research.py` | Validate legal research outputs for citation accuracy, jurisdictional correctness, and reasoning quality |
| `government_citizen.py` | Evaluate citizen-facing services for regulatory accuracy, plain language, and accessibility compliance |
| `retail_recommender.py` | Score product recommendations for relevance, safety, and demographic bias |
| `retail_support.py` | Measure customer service interactions for accuracy, empathy, resolution quality, and escalation handling |
| `insurance_claims.py` | Evaluate claims processing for coverage determination accuracy, compliance, and fairness |
| `insurance_underwriting.py` | Score underwriting decisions for risk accuracy, fair lending adherence, and pricing consistency |

---

### Multi-Agent Evaluation (Cowork Patterns) -- `cowork/` (5 samples)

Multi-agent evaluation patterns for use with [Claude Cowork](https://claude.com/product/cowork) sub-agent orchestration, [Claude Code Agent Teams](https://code.claude.com/docs/en/agent-teams), or any multi-agent framework. When multiple agents collaborate, each agent's output needs independent quality assessment -- these samples show how LayerLens judges serve as the shared quality signal between agents.

**What you will learn:** How to structure multi-agent workflows where generator, reviewer, and responder agents use LayerLens judges as evaluation feedback -- enabling automated quality loops, iterative judge refinement, and real-time incident triage.

| Sample | Scenario |
|--------|----------|
| `multi_agent_eval.py` | **Generator-Evaluator**: one agent produces responses while a second scores them with safety and factual accuracy judges |
| `code_review.py` | **Instrumentor-Reviewer**: one agent uploads code traces while a reviewer evaluates them with code execution, security, and metrics judges |
| `pair_programming.py` | **Rubric Writer-Tester**: one agent creates a judge, the other validates it against traces, and they refine iteratively via `judges.update()` |
| `rag_assessment.py` | **RAG Runner-Quality Judge**: one agent executes retrieval queries, the other evaluates groundedness and retrieval quality |
| `incident_response.py` | **Detector-Responder**: a detector evaluates recent traces for safety violations, a responder triages flagged traces with additional judges |

---

### MCP Server -- `mcp/` (1 sample)

Expose LayerLens capabilities as tools for Claude, Cursor, and any MCP-compatible AI assistant.

**What you will learn:** How to stand up a Model Context Protocol server that lets AI assistants list traces, create judges, and run evaluations through natural language.

| Sample | Scenario |
|--------|----------|
| `layerlens_server.py` | Run an MCP server that exposes trace management, judge creation, and evaluation execution as callable tools |

---

### CopilotKit CoAgents -- `copilotkit/` (2 agents + UI components)

Full-stack integration with CopilotKit using LangGraph CoAgents and generative UI card components.

**What you will learn:** How to build interactive evaluation and investigation experiences with LangGraph agents on the backend and rich React card components on the frontend.

| Sample | Scenario |
|--------|----------|
| `agents/evaluator_agent.py` | LangGraph CoAgent that drives evaluation workflows from a CopilotKit interface |
| `agents/investigator_agent.py` | LangGraph CoAgent that investigates trace data interactively |
| `components/*.tsx` | React card components for rendering evaluations, traces, judge verdicts, metrics, and compliance status |
| `hooks/*.ts` | CopilotKit hooks for wiring LayerLens actions and context into your React application |

---

### OpenClaw Agent Evaluation -- `openclaw/` (10 demos + skill)

Trace, evaluate, and monitor [OpenClaw](https://openclaw.ai/) autonomous AI agents using LayerLens. OpenClaw is an open-source autonomous AI agent that runs locally and uses messaging platforms as its UI. Each agent is governed by a `soul.md` file defining personality, constraints, and tool boundaries.

**What you will learn:** How to integrate OpenClaw agents with LayerLens for tracing, multi-judge evaluation, model comparison, and advanced patterns including comparative model selection, code generation gating, continuous drift detection, population-level content auditing, behavioral safety testing with honeypot tools, and adversarial alignment probing.

**Integration Samples:**

| Sample | Scenario |
|--------|----------|
| `trace_agent_execution.py` | Trace a single OpenClaw execution and evaluate with a quality judge |
| `evaluate_skill_output.py` | Run test prompts against a skill, evaluate with safety/accuracy/helpfulness judges, print quality report |
| `monitor_agent_safety.py` | Execute a mix of safe and adversarial prompts, flag safety failures, print incident report |
| `compare_agent_models.py` | Run the same tasks on multiple LLM backends, evaluate all, print a comparison table |

**Advanced Evaluation Patterns:**

| Sample | Scenario |
|--------|----------|
| `cage_match.py` | **Cage Match**: dispatch a task to N OpenClaw agents with different model backends, score side-by-side, produce a ranked leaderboard |
| `code_gate.py` | **Code Gate**: OpenClaw Coder-Reviewer-Tester-Judge pipeline with a PASS/FAIL gate before code executes |
| `heartbeat_benchmark.py` | **Heartbeat**: versioned task batteries with drift detection to catch regressions after model updates |
| `content_observer.py` | **Content Observer**: stratified content sampling for population-level quality monitoring (descended from the Moltbook/Moltbot content quality system) |
| `skill_auditor.py` | **Skill Auditor**: sandbox execution with honeypot decoys to detect unauthorized OpenClaw skill actions |
| `soul_redteam.py` | **Soul Red-Team**: adversarial probes targeting soul.md constraints with ALIGNED/DRIFT/VIOLATION verdicts |

**LayerLens Skill for OpenClaw:**

| File | Purpose |
|------|---------|
| `layerlens_skill/SKILL.md` | OpenClaw skill definition that lets agents interact with LayerLens directly |
| `layerlens_skill/scripts/evaluate.py` | Evaluation script for trace upload, judge creation, and structured results |

---

### Claude Code Skills -- `claude-code/` (6 skills)

Slash commands that bring LayerLens workflows directly into the Claude Code CLI.

**What you will learn:** How to manage traces, judges, evaluations, optimizations, benchmarks, and investigations without leaving your terminal.

| Skill | Scenario |
|-------|----------|
| `skills/trace.md` | `/trace` -- Upload, list, inspect, and delete traces from the command line |
| `skills/judge.md` | `/judge` -- Create, read, update, and delete judges interactively |
| `skills/evaluate.md` | `/evaluate` -- Run trace and model evaluations with guided prompts |
| `skills/optimize.md` | `/optimize` -- Estimate costs, run optimizations, and apply results to judges |
| `skills/benchmark.md` | `/benchmark` -- Manage models, benchmarks, and run comparisons |
| `skills/investigate.md` | `/investigate` -- Investigate trace data for errors, latency, and anomalies |

---

### Sample Data -- `data/`

Pre-built trace files, test datasets, and industry evaluation data so you can run every sample without generating your own data first.

| Directory | Contents |
|-----------|----------|
| `traces/` | 6 trace files covering simple LLM calls, RAG pipelines, multi-agent flows, error cases, and batch operations |
| `datasets/` | 2 general-purpose datasets: a golden test set for regression testing and a generic QA corpus |
| `industry/` | 16 domain-specific evaluation datasets spanning healthcare, finance, legal, government, insurance, retail, education, energy, manufacturing, media, real estate, telecom, and travel |

---

## SDK API Surface

These samples exercise the following SDK resources:

- **Traces** -- `client.traces.upload()`, `.get()`, `.get_many()`, `.delete()`, `.get_sources()`
- **Judges** -- `client.judges.create(name=, evaluation_goal=)`, `.get()`, `.get_many()`, `.update()`, `.delete()`
- **Evaluations** -- `client.evaluations.create(model=, benchmark=)`, `.get_many()`, `.wait_for_completion()`
- **Trace Evaluations** -- `client.trace_evaluations.create(trace_id=, judge_id=)`, `.get()`, `.get_results()`, `.estimate_cost()`
- **Judge Optimizations** -- `client.judge_optimizations.estimate()`, `.create()`, `.get()`, `.apply()`
- **Results** -- `client.results.get()`, `.get_all()`
- **Models** -- `client.models.get()`, `.get_by_key()`, `.add()`, `.create_custom()`
- **Benchmarks** -- `client.benchmarks.get()`, `.create_custom()`, `.create_smart()`
- **Comparisons** -- `client.public.comparisons.compare()`, `.compare_models()`

A shared helper (`_helpers.py`) provides `upload_trace_dict()` for creating traces from in-memory data.
