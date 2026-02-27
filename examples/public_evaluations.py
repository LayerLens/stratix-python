#!/usr/bin/env -S poetry run python

from layerlens import PublicClient
from layerlens.models import EvaluationStatus


def main():
    # Construct public client (API key from LAYERLENS_STRATIX_API_KEY env var or inline)
    client = PublicClient()

    # --- Get a specific evaluation by ID
    evaluation_id = "699f1426c1212b2d9c78e947"
    evaluation = client.evaluations.get_by_id(evaluation_id)
    if evaluation:
        print(f"Evaluation: {evaluation.id}")
        print(f"  Model: {evaluation.model_name} ({evaluation.model_company})")
        print(f"  Benchmark: {evaluation.benchmark_name}")
        print(f"  Status: {evaluation.status.value}")
        print(f"  Accuracy: {evaluation.accuracy:.2f}%")

        if evaluation.summary:
            print(f"  Summary: {evaluation.summary.name}")
            print(f"  Goal: {evaluation.summary.goal}")
            if evaluation.summary.metrics:
                print(f"  Metrics: {', '.join(m.name for m in evaluation.summary.metrics)}")
            if evaluation.summary.performance_details:
                print(f"  Strengths: {evaluation.summary.performance_details.strengths}")
            if evaluation.summary.analysis_summary:
                print(f"  Key takeaways: {evaluation.summary.analysis_summary.key_takeaways}")
    else:
        print(f"Evaluation {evaluation_id} not found")

    # --- List evaluations for a specific organization/project
    organization_id = "683e63925ef7e1c53c1f4b28"
    project_id = "683e63925ef7e1c53c1f4b29"

    response = client.evaluations.get_many(
        organization_id=organization_id,
        project_id=project_id,
        page=1,
        page_size=5,
        sort_by="submittedAt",
        order="desc",
    )
    if response:
        print(f"\nLatest evaluations ({response.pagination.total_count} total):")
        for e in response.evaluations:
            print(f"  - {e.id}: {e.model_name} on {e.benchmark_name} -> {e.accuracy:.2f}% ({e.status.value})")

    # --- Filter by status (only successful)
    response = client.evaluations.get_many(
        organization_id=organization_id,
        project_id=project_id,
        status=EvaluationStatus.SUCCESS,
        sort_by="accuracy",
        order="desc",
        page_size=5,
    )
    if response:
        print(f"\nTop successful evaluations ({response.pagination.total_count} total):")
        for e in response.evaluations:
            print(f"  - {e.model_name}: {e.accuracy:.2f}%")


if __name__ == "__main__":
    main()
