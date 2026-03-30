"""
Quickstart -- LayerLens Python SDK
==================================
Shows the core trace-evaluation workflow in under 50 lines:
init client, upload a trace, create a judge, evaluate, get results.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY="your-api-key"
"""

import json, os, sys, tempfile
from layerlens import Stratix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _helpers import poll_evaluation_results, create_judge


def main() -> None:
    # --- 1. Initialize the client (reads LAYERLENS_STRATIX_API_KEY from env)
    client = Stratix()

    # --- 2. Create a temporary JSONL trace file and upload it
    trace_data = {
        "input": [{"role": "user", "content": "What is the speed of light?"}],
        "output": "The speed of light in a vacuum is approximately 299,792,458 m/s.",
    }
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    tmp.write(json.dumps(trace_data) + "\n")
    tmp.close()
    try:
        upload = client.traces.upload(tmp.name)
    finally:
        os.unlink(tmp.name)
    if not upload or not upload.trace_ids:
        print("ERROR: Trace upload returned no IDs")
        return
    trace_id = upload.trace_ids[0]
    print(f"Uploaded trace: {trace_id}")

    # --- 3. Create a judge
    judge = create_judge(
        client,
        name="Quickstart Judge",
        evaluation_goal="Evaluate whether the response is factually accurate and complete.",
    )
    print(f"Created judge: {judge.name} (ID: {judge.id})")

    try:
        # --- 4. Run a trace evaluation
        evaluation = client.trace_evaluations.create(trace_id=trace_id, judge_id=judge.id)
        print(f"Evaluation started: {evaluation.id}")

        # --- 5. Poll for results
        results = poll_evaluation_results(client, evaluation.id)
        if results:
            r = results[0]
            print(f"Score:     {r.score}")
            print(f"Passed:    {r.passed}")
            print(f"Reasoning: {r.reasoning}")
        else:
            print("No results yet (evaluation may still be processing)")
    finally:
        client.judges.delete(judge.id)


if __name__ == "__main__":
    main()
