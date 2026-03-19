"""Create and configure AI judges on the LayerLens Stratix platform.

Demonstrates:
- Creating a judge with a name and evaluation goal
- Listing existing judges with pagination
- Fetching and displaying a single judge's configuration
- Updating a judge's evaluation goal
- Deleting a judge

Requires:
    LAYERLENS_STRATIX_API_KEY  - LayerLens platform API key
"""

from __future__ import annotations

import argparse
import os
import sys


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        print(f"ERROR: environment variable {name} is not set.", file=sys.stderr)
        sys.exit(1)
    return val


def cmd_create(client, args) -> None:
    """Create a new judge."""
    print(f"[create]  Creating judge '{args.name}'...")
    judge = client.judges.create(
        name=args.name,
        evaluation_goal=args.goal,
        model_id=args.model_id,
    )
    if judge is None:
        print("ERROR: Failed to create judge. Check API logs.", file=sys.stderr)
        sys.exit(1)
    print(f"[create]  Judge created successfully.")
    _print_judge(judge)


def cmd_list(client, args) -> None:
    """List existing judges."""
    resp = client.judges.get_many(page=args.page, page_size=args.page_size)
    if resp is None or not resp.judges:
        print("[list]    No judges found.")
        return

    sep = "-" * 72
    print(sep)
    print(f"  {'Name':<30} {'ID':<38}")
    print(sep)
    for j in resp.judges:
        print(f"  {j.name:<30} {j.id:<38}")
    print(sep)
    print(f"  Showing {resp.count} of {resp.total_count} judges (page {args.page})")


def cmd_get(client, args) -> None:
    """Fetch and display a single judge."""
    judge = client.judges.get(args.judge_id)
    if judge is None:
        print(f"ERROR: Judge {args.judge_id} not found.", file=sys.stderr)
        sys.exit(1)
    _print_judge(judge)


def cmd_update(client, args) -> None:
    """Update a judge's configuration."""
    print(f"[update]  Updating judge {args.judge_id}...")
    resp = client.judges.update(
        args.judge_id,
        name=args.name,
        evaluation_goal=args.goal,
        model_id=args.model_id,
    )
    if resp is None:
        print("ERROR: Update failed. Check API logs.", file=sys.stderr)
        sys.exit(1)
    print("[update]  Judge updated successfully.")
    # Re-fetch to show updated state
    judge = client.judges.get(args.judge_id)
    if judge:
        _print_judge(judge)


def cmd_delete(client, args) -> None:
    """Delete a judge."""
    print(f"[delete]  Deleting judge {args.judge_id}...")
    resp = client.judges.delete(args.judge_id)
    if resp is None:
        print("ERROR: Delete failed. Check API logs.", file=sys.stderr)
        sys.exit(1)
    print("[delete]  Judge deleted successfully.")


def _print_judge(judge) -> None:
    """Pretty-print a judge object."""
    sep = "-" * 72
    print(f"\n{sep}")
    print(f"  Judge Configuration")
    print(sep)
    print(f"  ID              : {judge.id}")
    print(f"  Name            : {judge.name}")
    if hasattr(judge, "evaluation_goal") and judge.evaluation_goal:
        print(f"  Evaluation Goal : {judge.evaluation_goal}")
    if hasattr(judge, "model_id") and judge.model_id:
        print(f"  Model ID        : {judge.model_id}")
    if hasattr(judge, "created_at") and judge.created_at:
        print(f"  Created At      : {judge.created_at}")
    if hasattr(judge, "updated_at") and judge.updated_at:
        print(f"  Updated At      : {judge.updated_at}")
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create and manage AI judges on LayerLens Stratix."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # create
    p_create = sub.add_parser("create", help="Create a new judge.")
    p_create.add_argument("--name", required=True, help="Judge name.")
    p_create.add_argument(
        "--goal",
        required=True,
        help="Evaluation goal describing what the judge assesses.",
    )
    p_create.add_argument("--model-id", default=None, help="Model ID to power the judge.")

    # list
    p_list = sub.add_parser("list", help="List existing judges.")
    p_list.add_argument("--page", type=int, default=1, help="Page number (default: 1).")
    p_list.add_argument("--page-size", type=int, default=20, help="Page size (default: 20).")

    # get
    p_get = sub.add_parser("get", help="Get a single judge by ID.")
    p_get.add_argument("judge_id", help="Judge ID to fetch.")

    # update
    p_update = sub.add_parser("update", help="Update a judge.")
    p_update.add_argument("judge_id", help="Judge ID to update.")
    p_update.add_argument("--name", default=None, help="New judge name.")
    p_update.add_argument("--goal", default=None, help="New evaluation goal.")
    p_update.add_argument("--model-id", default=None, help="New model ID.")

    # delete
    p_delete = sub.add_parser("delete", help="Delete a judge.")
    p_delete.add_argument("judge_id", help="Judge ID to delete.")

    args = parser.parse_args()

    from layerlens import Stratix

    api_key = _require_env("LAYERLENS_STRATIX_API_KEY")
    client = Stratix(api_key=api_key)
    print(f"[init]    Connected to LayerLens (org={client.organization_id})")

    dispatch = {
        "create": cmd_create,
        "list": cmd_list,
        "get": cmd_get,
        "update": cmd_update,
        "delete": cmd_delete,
    }
    dispatch[args.command](client, args)


if __name__ == "__main__":
    main()
