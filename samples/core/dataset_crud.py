"""Dataset (benchmark) lifecycle management on LayerLens Stratix.

Demonstrates:
- Uploading a JSONL dataset as a custom benchmark
- Listing datasets with pagination
- Previewing a dataset by fetching its details
- Deleting a dataset

The SDK calls datasets "benchmarks" -- this sample uses both terms
interchangeably to match the platform UI language.

Requires:
    LAYERLENS_STRATIX_API_KEY  - LayerLens platform API key
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        print(f"ERROR: environment variable {name} is not set.", file=sys.stderr)
        sys.exit(1)
    return val


def _generate_sample_jsonl(path: str, num_rows: int = 5) -> None:
    """Create a minimal JSONL file for demonstration."""
    prompts = [
        {"messages": [{"role": "user", "content": "What is the capital of France?"}]},
        {"messages": [{"role": "user", "content": "Explain photosynthesis in one sentence."}]},
        {"messages": [{"role": "user", "content": "Translate 'hello' to Japanese."}]},
        {"messages": [{"role": "user", "content": "What is 42 * 37?"}]},
        {"messages": [{"role": "user", "content": "Name three programming paradigms."}]},
    ]
    with open(path, "w") as f:
        for p in prompts[:num_rows]:
            f.write(json.dumps(p) + "\n")
    print(f"[gen]     Generated sample JSONL with {min(num_rows, len(prompts))} rows at {path}")


def cmd_upload(client, args) -> None:
    """Upload a JSONL file as a custom benchmark."""
    file_path = args.file

    # If no file provided, generate a sample
    if file_path is None:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        tmp.close()
        _generate_sample_jsonl(tmp.name)
        file_path = tmp.name

    if not os.path.isfile(file_path):
        print(f"ERROR: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    size_kb = os.path.getsize(file_path) / 1024
    print(f"[upload]  File: {file_path} ({size_kb:.1f} KB)")
    print(f"[upload]  Creating custom benchmark '{args.name}'...")

    try:
        resp = client.benchmarks.create_custom(
            name=args.name,
            description=args.description,
            file_path=file_path,
        )
        if resp and resp.benchmark_id:
            print(f"[upload]  Success! benchmark_id={resp.benchmark_id}")
        else:
            print("[upload]  Upload completed but no benchmark_id returned.", file=sys.stderr)
    except Exception as exc:
        print(f"[upload]  Failed: {exc}", file=sys.stderr)
        sys.exit(1)


def cmd_list(client, args) -> None:
    """List datasets with pagination."""
    benchmarks = client.benchmarks.get(type=args.type, name=args.name)
    if not benchmarks:
        print("[list]    No datasets found.")
        return

    sep = "-" * 80
    print(sep)
    print(f"  {'Name':<35} {'Key':<25} {'ID':<20}")
    print(sep)
    for b in benchmarks:
        name_short = (b.name[:32] + "...") if len(b.name) > 35 else b.name
        key_short = (b.key[:22] + "...") if len(b.key) > 25 else b.key
        print(f"  {name_short:<35} {key_short:<25} {b.id:<20}")
    print(sep)
    print(f"  Total: {len(benchmarks)} datasets")


def cmd_preview(client, args) -> None:
    """Preview a dataset by fetching its details."""
    benchmark = client.benchmarks.get_by_id(args.dataset_id)
    if benchmark is None:
        print(f"ERROR: Dataset {args.dataset_id} not found.", file=sys.stderr)
        sys.exit(1)

    sep = "-" * 72
    print(f"\n{sep}")
    print(f"  Dataset Preview")
    print(sep)
    print(f"  ID          : {benchmark.id}")
    print(f"  Name        : {benchmark.name}")
    print(f"  Key         : {benchmark.key}")
    if hasattr(benchmark, "description") and benchmark.description:
        print(f"  Description : {benchmark.description}")
    if hasattr(benchmark, "prompt_count") and benchmark.prompt_count:
        print(f"  Prompts     : {benchmark.prompt_count}")
    if hasattr(benchmark, "categories") and benchmark.categories:
        print(f"  Categories  : {', '.join(benchmark.categories)}")
    if hasattr(benchmark, "created_at") and benchmark.created_at:
        print(f"  Created     : {benchmark.created_at}")
    print(sep)


def cmd_delete(client, args) -> None:
    """Delete a dataset."""
    print(f"[delete]  Looking up dataset {args.dataset_id}...")
    benchmark = client.benchmarks.get_by_id(args.dataset_id)
    if benchmark is None:
        print(f"ERROR: Dataset {args.dataset_id} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"[delete]  Removing dataset '{benchmark.name}' ...")
    try:
        success = client.benchmarks.remove(benchmark.id)
        if success:
            print("[delete]  Dataset removed from project successfully.")
        else:
            print("[delete]  Remove returned False. The dataset may already be removed.", file=sys.stderr)
    except Exception as exc:
        print(f"[delete]  Failed: {exc}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dataset (benchmark) CRUD operations on LayerLens Stratix."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # upload
    p_upload = sub.add_parser("upload", help="Upload a JSONL dataset.")
    p_upload.add_argument("--name", required=True, help="Dataset name (max 64 chars).")
    p_upload.add_argument(
        "--description", default="Sample dataset created by SDK sample.", help="Dataset description."
    )
    p_upload.add_argument("--file", default=None, help="Path to JSONL file. Omit to use a generated sample.")

    # list
    p_list = sub.add_parser("list", help="List datasets.")
    p_list.add_argument("--type", choices=["custom", "public"], default=None, help="Filter by type.")
    p_list.add_argument("--name", default=None, help="Filter by name substring.")

    # preview
    p_preview = sub.add_parser("preview", help="Preview a dataset.")
    p_preview.add_argument("dataset_id", help="Dataset ID to preview.")

    # delete
    p_delete = sub.add_parser("delete", help="Remove a dataset from the project.")
    p_delete.add_argument("dataset_id", help="Dataset ID to remove.")

    args = parser.parse_args()

    from layerlens import Stratix

    api_key = _require_env("LAYERLENS_STRATIX_API_KEY")
    client = Stratix(api_key=api_key)
    print(f"[init]    Connected to LayerLens (org={client.organization_id})")

    dispatch = {
        "upload": cmd_upload,
        "list": cmd_list,
        "preview": cmd_preview,
        "delete": cmd_delete,
    }
    dispatch[args.command](client, args)


if __name__ == "__main__":
    main()
