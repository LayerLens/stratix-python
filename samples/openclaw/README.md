# OpenClaw Agent Evaluation -- LayerLens Python SDK

Trace, evaluate, and monitor [OpenClaw](https://openclaw.ai/) autonomous AI agents using the LayerLens evaluation platform.

[OpenClaw](https://openclaw.ai/) is an open-source autonomous AI agent (60,000+ GitHub stars) that runs locally on your machine and uses messaging platforms (Telegram, Discord, WhatsApp, Slack) as its UI. It executes real tasks: shell commands, browser automation, email, calendar, and file operations -- all driven by a skill system with YAML-configured capabilities.

Each OpenClaw agent is governed by a **soul.md** file -- a markdown spec that defines the agent's personality, ethical constraints, and tool boundaries. Think of it as a constitution for the agent's behavior.

LayerLens integrates with OpenClaw at two levels:

- **Tracing** -- capture every agent execution (input task, output result, metadata) as a LayerLens trace for auditability and analysis.
- **Evaluation** -- score agent outputs with AI judges for safety, accuracy, helpfulness, and any custom quality dimension.

---

## Prerequisites

```bash
pip install layerlens --index-url https://sdk.layerlens.ai/package openclaw
export LAYERLENS_STRATIX_API_KEY=your-api-key
```

All samples gracefully fall back to simulated data when OpenClaw is not installed or not running, so you can explore the LayerLens evaluation workflow without a live agent.

---

## Quick Start

```bash
# Run a single traced execution with evaluation
python samples/openclaw/trace_agent_execution.py

# Compare LLM backends for agent quality
python samples/openclaw/compare_agent_models.py

# Run a cage match between models
python -m samples.openclaw.cage_match --models claude-sonnet-4-20250514,gpt-4o,deepseek-v3

# Red-team test an agent against its soul.md
python -m samples.openclaw.soul_redteam --models claude-sonnet-4-20250514,gpt-4o
```

---

## Integration Samples

End-to-end examples showing how to connect OpenClaw agents with LayerLens tracing and evaluation.

| Sample | Scenario |
|--------|----------|
| `trace_agent_execution.py` | Trace a single OpenClaw execution and evaluate with a quality judge |
| `evaluate_skill_output.py` | Run test prompts against a skill, evaluate with safety/accuracy/helpfulness judges, print quality report |
| `monitor_agent_safety.py` | Execute a mix of safe and adversarial prompts, flag safety failures, print incident report |
| `compare_agent_models.py` | Run the same tasks on multiple LLM backends, evaluate all, print a comparison table |

---

## Advanced Evaluation Patterns

Deeper evaluation patterns for assessing OpenClaw agents across quality, safety, and alignment dimensions. Each demo uses the `_runner.py` base class which provides both OpenClaw execution (via `execute_with_openclaw()`) and LayerLens tracing/evaluation. All demos support `--no-sdk` for offline mode and `--json` for structured output.

| Sample | Question It Answers | Scenario |
|--------|---------------------|----------|
| `cage_match.py` | Which LLM backend should my OpenClaw agent use for this skill? | Dispatch a task to N OpenClaw agents with different model backends, judge outputs side-by-side, publish a ranked leaderboard |
| `code_gate.py` | Is the code my OpenClaw agent produces safe to execute? | Coder-Reviewer-Tester-Judge pipeline with a PASS/FAIL gate before code runs on your machine |
| `heartbeat_benchmark.py` | Has my OpenClaw agent's performance degraded after a model update? | Versioned task batteries with drift detection to catch regressions before they affect agent behavior |
| `content_observer.py` | What is the aggregate quality of content my OpenClaw agents produce? | Stratified content sampling for population-level quality monitoring across communities (descended from the Moltbook/Moltbot content quality system) |
| `skill_auditor.py` | Does this OpenClaw skill attempt unauthorized actions? | Sandbox execution with honeypot decoys to detect data exfiltration, privilege escalation, and unauthorized outbound requests |
| `soul_redteam.py` | Does my OpenClaw agent stay aligned with its soul.md constraints? | Adversarial probes targeting soul spec constraints with ALIGNED/DRIFT/VIOLATION verdicts |

### What is a Soul Spec?

OpenClaw agents are configured with a `soul.md` file that acts as the agent's constitution. It defines:

- **Purpose** -- what the agent is for
- **Persona** -- how the agent communicates
- **Ethical Constraints** -- what the agent must never do
- **Tool Boundaries** -- which tools the agent can access

The `soul_redteam.py` demo probes whether an agent faithfully follows its soul spec under adversarial pressure, while `skill_auditor.py` tests whether individual skills respect the boundaries defined in the soul spec.

### Content Observer Heritage

The `content_observer.py` demo descends from the "Moltbook Observer" -- a population-level content quality monitoring system originally built for Moltbook (later rebranded Moltbot), an AI-powered social platform. The sampling strategies, karma-tier weighting, and community-level breakdowns reflect real patterns from monitoring AI-generated content at scale.

---

## LayerLens Skill for OpenClaw

The `layerlens_skill/` directory contains an OpenClaw skill that lets agents interact with LayerLens directly. Install it by copying to your OpenClaw skills directory:

```bash
cp -r samples/openclaw/layerlens_skill ~/.openclaw/skills/layerlens
```

Then ask your agent:

```
Evaluate the last response for safety using LayerLens.
```

The skill calls `scripts/evaluate.py` which accepts input via arguments or JSON on stdin and returns structured results:

```bash
# Direct usage
python layerlens_skill/scripts/evaluate.py \
  --input "What is 2+2?" \
  --output "2+2 is 4." \
  --goal "factual accuracy"

# Via stdin
echo '{"input": "What is 2+2?", "output": "4", "goal": "accuracy"}' \
  | python layerlens_skill/scripts/evaluate.py
```

### Skill Files

| File | Purpose |
|------|---------|
| `layerlens_skill/SKILL.md` | Skill definition with YAML frontmatter, description, and usage instructions |
| `layerlens_skill/scripts/evaluate.py` | Evaluation script that uploads traces, creates judges, and returns JSON results |

---

## Supporting Modules

The advanced evaluation demos share infrastructure in two sub-packages:

### `judges/` -- Local Evaluation Judges

| Module | Purpose |
|--------|---------|
| `comparative.py` | Side-by-side multi-model evaluator across 4 quality dimensions |
| `code_quality.py` | Code quality evaluator with binary gate enforcement |
| `benchmark.py` | Multi-method scoring against golden answers |
| `population_quality.py` | Batch content quality evaluator for feed monitoring |
| `behavioral_safety.py` | Multi-category threat assessment for skill auditing |
| `alignment_fidelity.py` | Soul spec alignment evaluator with 3-tier verdicts |

### `lib/` -- Shared Utilities

| Module | Purpose |
|--------|---------|
| `code_pipeline.py` | Multi-stage code generation pipeline (Coder-Reviewer-Tester-Judge) |
| `drift_detector.py` | Rolling-baseline performance drift detection engine |
| `honeypot.py` | Decoy tools that log violation attempts |
| `notifier.py` | Multi-channel alert and leaderboard publisher |
| `probe_generator.py` | Adversarial probe factory for red-team testing |
| `sampler.py` | Stratified post sampler for population monitoring |
| `schemas.py` | Shared Pydantic schemas for request/response envelopes |
| `soul_parser.py` | Soul.md markdown parser |
| `task_battery.py` | Versioned benchmark task battery loader |

---

## How It Works

```
OpenClaw Agent                    LayerLens Platform
+-----------------+               +-------------------+
| Execute task    |               |                   |
| (shell, browse, |  upload trace | Upload trace      |
|  email, etc.)   | ------------> | (input + output   |
|                 |               |  + metadata)       |
+-----------------+               +-------------------+
                                          |
                                          v
                                  +-------------------+
                                  | Create judge      |
                                  | (safety, accuracy,|
                                  |  helpfulness)     |
                                  +-------------------+
                                          |
                                          v
                                  +-------------------+
                                  | Run evaluation    |
                                  | score + verdict   |
                                  | + reasoning       |
                                  +-------------------+
```

Each sample follows this pattern:

1. **Execute** -- run a task via the OpenClaw agent (or use simulated data).
2. **Trace** -- upload the execution as a LayerLens trace.
3. **Judge** -- create one or more judges with `client.judges.create(name=, evaluation_goal=)`.
4. **Evaluate** -- run `client.trace_evaluations.create(trace_id=, judge_id=)`.
5. **Results** -- poll with `poll_evaluation_results()` and display.

---

## SDK Methods Used

| Method | Purpose |
|--------|---------|
| `Stratix()` | Initialize the LayerLens client |
| `client.traces.upload(path)` | Upload a JSONL trace file |
| `client.judges.create(name=, evaluation_goal=)` | Create an evaluation judge |
| `client.judges.get_many()` | List existing judges |
| `client.trace_evaluations.create(trace_id=, judge_id=)` | Start an evaluation |
| `client.trace_evaluations.get_results(id)` | Retrieve evaluation results |
