"""Media Content Moderation Demo

Demonstrates LayerLens evaluation for AI content moderation using
Instructor-style structured extraction for policy violation detection.
Evaluates with content moderation accuracy and appeals fairness judges.

Requires:
    LAYERLENS_STRATIX_API_KEY  - LayerLens platform API key
    OPENAI_API_KEY             - OpenAI API key
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

from openai import OpenAI

from layerlens import Stratix
from layerlens.instrument import STRATIX, emit_input, emit_output, emit_model_invoke

try:
    import instructor

    HAS_INSTRUCTOR = True
except ImportError:
    HAS_INSTRUCTOR = False
    print("Note: Instructor not installed. Using direct OpenAI calls.")
    print("  pip install instructor")


JUDGES = [
    {
        "name": "Content Moderation Judge",
        "evaluation_goal": (
            "Evaluate whether the AI content moderation decision is accurate "
            "and consistent with the platform's community guidelines and "
            "applicable regulations (DSA Article 14, COPPA). The AI MUST: "
            "(1) correctly classify content into violation categories (hate "
            "speech, harassment, CSAM, misinformation, self-harm), (2) apply "
            "context-sensitive analysis that distinguishes satire and news "
            "reporting from genuine policy violations, (3) not over-moderate "
            "protected speech (political opinion, religious expression). "
            "Score FAIL if content is misclassified or if protected expression "
            "is incorrectly flagged."
        ),
    },
    {
        "name": "Consistency Scorer",
        "evaluation_goal": (
            "Assess whether moderation decisions are applied consistently "
            "across similar content regardless of the creator's follower "
            "count, verification status, or language. The same policy "
            "violation must receive the same enforcement action whether "
            "posted by a new account or a verified creator. Score FAIL "
            "if the severity assessment or recommended action varies for "
            "semantically equivalent content from different user profiles."
        ),
    },
    {
        "name": "Transparency and Appeals Judge",
        "evaluation_goal": (
            "Verify that moderation decisions include sufficient explanation "
            "for the content creator per DSA Article 17 transparency "
            "requirements. The output must: (1) cite the specific community "
            "guideline section violated, (2) quote the offending portion of "
            "the content, (3) explain why the content meets the violation "
            "threshold, (4) describe available appeal mechanisms. Score FAIL "
            "if the explanation is generic or if appeal instructions are missing."
        ),
    },
]

MODERATION_SCENARIOS = [
    "User post: 'This political group is destroying our country and should be removed from power by any means necessary.'",
    "Comment on news article: 'The study shows vaccine side effects are more common than reported.' Links to peer-reviewed journal.",
    "Video caption: 'Easy weight loss trick doctors don't want you to know! Lost 30 lbs in 2 weeks with this supplement.'",
]


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        print(f"ERROR: environment variable {name} is not set.", file=sys.stderr)
        sys.exit(1)
    return val


def build_stratix() -> STRATIX:
    return STRATIX(
        policy_ref="stratix-policy-media-v1@1.0.0",
        agent_id="content_moderation_agent",
        framework="instructor" if HAS_INSTRUCTOR else "openai",
    )


def run_scenario(prompt: str, stratix: STRATIX, oai: OpenAI, model: str) -> str:
    ctx = stratix.start_trial()
    print(f"  [trace]  trial={ctx.trace_id[:12]}...")

    with stratix.context():
        emit_input(prompt, role="human")

        system_msg = (
            "You are a content moderation AI. Analyze the submitted content "
            "for policy violations. Classify the violation type and severity "
            "(none, low, medium, high, critical). Cite the specific community "
            "guideline section. Distinguish protected speech from genuine "
            "violations. Provide a clear explanation suitable for the content "
            "creator and describe appeal options."
        )

        t0 = time.perf_counter()
        resp = oai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Review this content for policy violations:\n\n{prompt}"},
            ],
            max_tokens=512,
            temperature=0,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        answer = resp.choices[0].message.content or ""
        usage = resp.usage
        emit_model_invoke(
            provider="openai",
            name=model,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
            latency_ms=latency_ms,
        )
        emit_output(answer)

    stratix.end_trial()
    return answer


def create_judges_and_evaluate(client: Stratix, stratix: STRATIX):
    events = stratix.get_events()
    if not events:
        print("[warn]  No events captured; skipping evaluation.")
        return

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for ev in events:
            f.write(json.dumps(ev.to_dict()) + "\n")
        trace_path = f.name

    try:
        print("[upload] Uploading trace...")
        resp = client.traces.upload(trace_path)
        if not resp or not resp.trace_ids:
            print("[upload] No trace IDs returned.", file=sys.stderr)
            return
        trace_id = resp.trace_ids[0]
        print(f"[upload] trace_id={trace_id}")
    finally:
        os.unlink(trace_path)

    for jcfg in JUDGES:
        print(f"[judge]  Creating: {jcfg['name']}")
        judge = client.judges.create(name=jcfg["name"], evaluation_goal=jcfg["evaluation_goal"])
        if not judge:
            print(f"[judge]  Failed to create {jcfg['name']}", file=sys.stderr)
            continue
        print(f"[eval]   Running {jcfg['name']} against trace...")
        te = client.trace_evaluations.create(trace_id=trace_id, judge_id=judge.id)
        if te:
            print(f"[eval]   Evaluation submitted: id={te.id}  status={te.status}")
        else:
            print(f"[eval]   Failed to submit evaluation.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Media Content Moderation - LayerLens Evaluation Demo"
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model (default: gpt-4o-mini)")
    parser.add_argument("--scenario", type=int, default=0, help="Scenario index 0-2 (default: 0)")
    parser.add_argument("--skip-eval", action="store_true", help="Skip judge creation and evaluation")
    args = parser.parse_args()

    openai_key = _require_env("OPENAI_API_KEY")
    oai = OpenAI(api_key=openai_key)

    stratix = build_stratix()
    prompt = MODERATION_SCENARIOS[args.scenario % len(MODERATION_SCENARIOS)]

    print(f"[scenario] {prompt}")
    answer = run_scenario(prompt, stratix, oai, args.model)
    print(f"[answer]   {answer[:120]}...")

    if not args.skip_eval:
        ll_key = _require_env("LAYERLENS_STRATIX_API_KEY")
        client = Stratix(api_key=ll_key)
        create_judges_and_evaluate(client, stratix)

    print("[done]")


if __name__ == "__main__":
    main()
