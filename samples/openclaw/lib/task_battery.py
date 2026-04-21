"""
BenchmarkTaskBattery -- Versioned Task Battery Loader & Validator
==================================================================

Loads a benchmark task battery from a JSON file (or built-in defaults),
validates the schema version, ensures all tasks have golden answers,
and provides iteration/filtering utilities.
"""

from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict, List

from pydantic import Field, BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class BenchmarkTask(BaseModel):
    """A single benchmark task with golden answer."""

    task_id: str
    prompt: str
    golden_answer: str
    scoring_method: str = "semantic_similarity"
    weight: float = 1.0
    category: str = "general"
    difficulty: str = "medium"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatteryManifest(BaseModel):
    """Validated task battery manifest."""

    version: str
    battery_id: str
    description: str = ""
    tasks: List[BenchmarkTask]
    task_count: int = 0
    total_weight: float = 0.0
    categories: List[str] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        self.task_count = len(self.tasks)
        self.total_weight = sum(t.weight for t in self.tasks)
        self.categories = sorted(set(t.category for t in self.tasks))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_VERSIONS: set[str] = {"1.0", "1.1"}
REQUIRED_TASK_FIELDS: set[str] = {"task_id", "prompt", "golden_answer"}
VALID_SCORING_METHODS: set[str] = {"semantic_similarity", "rubric", "exact_match"}


# ---------------------------------------------------------------------------
# Default built-in battery
# ---------------------------------------------------------------------------

DEFAULT_BATTERY: dict[str, Any] = {
    "version": "1.0",
    "battery_id": "demo-benchmark-v1",
    "description": "Built-in demo benchmark for Heartbeat demo",
    "tasks": [
        {
            "task_id": "factual-001",
            "prompt": "What is the capital of France?",
            "golden_answer": "The capital of France is Paris.",
            "scoring_method": "semantic_similarity",
            "weight": 1.0,
            "category": "factual",
            "difficulty": "easy",
        },
        {
            "task_id": "factual-002",
            "prompt": "What is the speed of light in a vacuum?",
            "golden_answer": "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
            "scoring_method": "semantic_similarity",
            "weight": 1.0,
            "category": "factual",
            "difficulty": "easy",
        },
        {
            "task_id": "reasoning-001",
            "prompt": "A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left?",
            "golden_answer": "The farmer has 9 sheep left.",
            "scoring_method": "semantic_similarity",
            "weight": 1.5,
            "category": "reasoning",
            "difficulty": "medium",
        },
        {
            "task_id": "reasoning-002",
            "prompt": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
            "golden_answer": "It would take 5 minutes. Each machine makes one widget in 5 minutes, so 100 machines can make 100 widgets in 5 minutes.",
            "scoring_method": "rubric",
            "weight": 2.0,
            "category": "reasoning",
            "difficulty": "medium",
        },
        {
            "task_id": "coding-001",
            "prompt": "Write a Python function that reverses a string without using slicing.",
            "golden_answer": "def reverse_string(s):\n    result = ''\n    for char in s:\n        result = char + result\n    return result",
            "scoring_method": "rubric",
            "weight": 1.5,
            "category": "coding",
            "difficulty": "easy",
        },
        {
            "task_id": "coding-002",
            "prompt": "Implement a function to check if a binary tree is balanced.",
            "golden_answer": "def is_balanced(root):\n    def check(node):\n        if not node:\n            return 0\n        left = check(node.left)\n        right = check(node.right)\n        if left == -1 or right == -1 or abs(left - right) > 1:\n            return -1\n        return max(left, right) + 1\n    return check(root) != -1",
            "scoring_method": "rubric",
            "weight": 2.5,
            "category": "coding",
            "difficulty": "hard",
        },
        {
            "task_id": "math-001",
            "prompt": "What is the integral of x^2 dx?",
            "golden_answer": "x^3/3 + C",
            "scoring_method": "semantic_similarity",
            "weight": 1.0,
            "category": "math",
            "difficulty": "easy",
        },
        {
            "task_id": "math-002",
            "prompt": "Find the derivative of f(x) = ln(x^2 + 1).",
            "golden_answer": "f'(x) = 2x / (x^2 + 1)",
            "scoring_method": "semantic_similarity",
            "weight": 1.5,
            "category": "math",
            "difficulty": "medium",
        },
        {
            "task_id": "exact-001",
            "prompt": "What is 7 * 8?",
            "golden_answer": "56",
            "scoring_method": "exact_match",
            "weight": 0.5,
            "category": "math",
            "difficulty": "easy",
        },
        {
            "task_id": "exact-002",
            "prompt": "What HTTP status code means 'Not Found'?",
            "golden_answer": "404",
            "scoring_method": "exact_match",
            "weight": 0.5,
            "category": "factual",
            "difficulty": "easy",
        },
    ],
}


# ---------------------------------------------------------------------------
# Battery loader
# ---------------------------------------------------------------------------


class BenchmarkTaskBattery:
    """Versioned benchmark task battery with validation."""

    def __init__(self, manifest: BatteryManifest) -> None:
        self._manifest = manifest

    @property
    def version(self) -> str:
        return self._manifest.version

    @property
    def battery_id(self) -> str:
        return self._manifest.battery_id

    @property
    def description(self) -> str:
        return self._manifest.description

    @property
    def tasks(self) -> list[BenchmarkTask]:
        return self._manifest.tasks

    @property
    def task_count(self) -> int:
        return self._manifest.task_count

    @property
    def total_weight(self) -> float:
        return self._manifest.total_weight

    @property
    def categories(self) -> list[str]:
        return self._manifest.categories

    @classmethod
    def load_file(cls, path: str) -> BenchmarkTaskBattery:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Task battery file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        battery = cls._validate_and_build(data)
        logger.info("Loaded task battery '%s' v%s from %s", battery.battery_id, battery.version, path)
        return battery

    @classmethod
    def load_default(cls) -> BenchmarkTaskBattery:
        battery = cls._validate_and_build(DEFAULT_BATTERY)
        logger.info("Loaded default battery '%s': %d tasks", battery.battery_id, battery.task_count)
        return battery

    def filter_by_category(self, category: str) -> list[BenchmarkTask]:
        return [t for t in self.tasks if t.category == category]

    def filter_by_difficulty(self, difficulty: str) -> list[BenchmarkTask]:
        return [t for t in self.tasks if t.difficulty == difficulty]

    def filter_by_method(self, method: str) -> list[BenchmarkTask]:
        return [t for t in self.tasks if t.scoring_method == method]

    def get_task(self, task_id: str) -> BenchmarkTask | None:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "battery_id": self.battery_id,
            "version": self.version,
            "task_count": self.task_count,
            "total_weight": self.total_weight,
            "categories": self.categories,
            "difficulty_distribution": {
                diff: sum(1 for t in self.tasks if t.difficulty == diff) for diff in ("easy", "medium", "hard")
            },
            "method_distribution": {
                m: sum(1 for t in self.tasks if t.scoring_method == m) for m in VALID_SCORING_METHODS
            },
        }

    @classmethod
    def _validate_and_build(cls, data: dict[str, Any]) -> BenchmarkTaskBattery:
        version = data.get("version", "")
        if version not in SUPPORTED_VERSIONS:
            raise ValueError(f"Unsupported battery version '{version}'. Supported: {SUPPORTED_VERSIONS}")
        if "tasks" not in data or not isinstance(data["tasks"], list):
            raise ValueError("Battery must contain a 'tasks' array")
        if not data["tasks"]:
            raise ValueError("Battery must contain at least one task")

        battery_id = data.get("battery_id", "unknown")
        seen_ids: set[str] = set()
        validated_tasks: list[BenchmarkTask] = []
        errors: list[str] = []

        for i, raw_task in enumerate(data["tasks"]):
            missing = REQUIRED_TASK_FIELDS - set(raw_task.keys())
            if missing:
                errors.append(f"Task {i}: missing fields {missing}")
                continue
            task_id = raw_task["task_id"]
            if task_id in seen_ids:
                errors.append(f"Task {i}: duplicate task_id '{task_id}'")
                continue
            seen_ids.add(task_id)
            if not raw_task.get("golden_answer", "").strip():
                errors.append(f"Task '{task_id}': empty golden_answer")
                continue
            method = raw_task.get("scoring_method", "semantic_similarity")
            if method not in VALID_SCORING_METHODS:
                errors.append(f"Task '{task_id}': invalid scoring_method '{method}'")
                continue
            weight = raw_task.get("weight", 1.0)
            if weight <= 0:
                errors.append(f"Task '{task_id}': weight must be positive, got {weight}")
                continue
            validated_tasks.append(
                BenchmarkTask(
                    task_id=task_id,
                    prompt=raw_task["prompt"],
                    golden_answer=raw_task["golden_answer"],
                    scoring_method=method,
                    weight=weight,
                    category=raw_task.get("category", "general"),
                    difficulty=raw_task.get("difficulty", "medium"),
                    metadata=raw_task.get("metadata", {}),
                )
            )

        if errors:
            for err in errors:
                logger.error("Validation error: %s", err)
            raise ValueError(f"Battery validation failed with {len(errors)} error(s): " + "; ".join(errors[:5]))

        manifest = BatteryManifest(
            version=version,
            battery_id=battery_id,
            description=data.get("description", ""),
            tasks=validated_tasks,
        )
        return cls(manifest)
