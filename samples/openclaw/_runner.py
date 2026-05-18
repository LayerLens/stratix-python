"""Shared base for OpenClaw + LayerLens agent evaluation demos.

Provides a DemoRunner base class with OpenClaw SDK support, allowing demos
to execute tasks via real OpenClaw agents and evaluate results through
LayerLens judges.
"""

from __future__ import annotations

import os
import sys
import json
import time
import asyncio
import logging
import argparse
import tempfile
from abc import ABC, abstractmethod
from typing import Any

from layerlens import Stratix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _helpers import get_default_model_id, poll_evaluation_results

# Optional OpenClaw SDK import
try:
    from openclaw import OpenClawClient  # type: ignore[import-untyped]

    _OPENCLAW_AVAILABLE = True
except ImportError:
    _OPENCLAW_AVAILABLE = False

logger = logging.getLogger("layerlens.samples.openclaw")


def _print_scores(
    scores: dict[str, float],
    aggregate: float,
    verdict: str | None = None,
) -> None:
    """Pretty-print evaluation scores to stdout."""
    print("\n" + "=" * 60)
    if verdict:
        print(f"  Verdict: {verdict}")
    print(f"  Aggregate Score: {aggregate:.2f} / 10.0")
    print("-" * 60)
    for dim, score in sorted(scores.items()):
        bar = "#" * int(score) + "." * (10 - int(score))
        print(f"  {dim:<30} {score:>5.1f}  [{bar}]")
    print("=" * 60 + "\n")


def _print_json(data: Any) -> None:
    """Pretty-print a JSON-serializable object."""
    print(json.dumps(data, indent=2, default=str))


class DemoRunner(ABC):
    """
    Abstract base for all OpenClaw + LayerLens evaluation demo scripts.

    Subclasses implement ``run()`` with demo-specific logic.  The base class
    provides both a ``Stratix`` SDK client (LayerLens) and an optional
    ``openclaw_client`` (OpenClaw), along with CLI argument parsing and
    shared helpers.
    """

    # Override in subclass
    demo_id: str = ""
    demo_name: str = ""
    description: str = ""

    def __init__(self) -> None:
        self.args: argparse.Namespace | None = None
        self.client: Stratix | None = None
        self.openclaw_client: Any = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def build_parser(self) -> argparse.ArgumentParser:
        """Build CLI parser with shared flags.  Subclasses call ``super()`` then add more."""
        parser = argparse.ArgumentParser(
            description=self.description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="Enable debug logging.",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Output results as JSON instead of tables.",
        )
        parser.add_argument(
            "--no-sdk",
            action="store_true",
            help="Skip SDK initialization (run in offline/demo mode).",
        )
        return parser

    @abstractmethod
    async def run(self) -> dict[str, Any]:
        """Execute the demo.  Returns the result dict."""
        ...

    def _init_sdk(self) -> None:
        """Initialize the Stratix SDK client and OpenClaw client."""
        if self.args and self.args.no_sdk:
            self.logger.info("SDK disabled (--no-sdk). Running in offline mode.")
            self.client = None
            self.openclaw_client = None
            return
        try:
            self.client = Stratix()
            self.logger.info("Stratix SDK client initialized.")
        except Exception as exc:
            self.logger.warning(
                "Could not initialize Stratix SDK (%s). Running in offline mode.",
                exc,
            )
            self.client = None

        # Initialize OpenClaw client
        if _OPENCLAW_AVAILABLE:
            try:
                self.openclaw_client = OpenClawClient()
                self.logger.info("OpenClaw SDK client initialized.")
            except Exception as exc:
                self.logger.warning(
                    "Could not initialize OpenClaw SDK (%s). Using simulated data.",
                    exc,
                )
                self.openclaw_client = None
        else:
            self.logger.info("OpenClaw SDK not installed. Using simulated data.")
            self.openclaw_client = None

    def execute_with_openclaw(
        self,
        task: str,
        model: str | None = None,
        agent_name: str = "openclaw-agent",
    ) -> dict[str, Any]:
        """Execute a task via OpenClaw and return structured results.

        Falls back to simulated data if OpenClaw is not installed or not
        available.

        Returns:
            dict with keys: input, output, model, duration_ms
        """
        model = model or "claude-sonnet-4-20250514"
        if self.openclaw_client is not None:
            try:
                agent = self.openclaw_client.agents.create(
                    name=agent_name,
                    model=model,
                    description=f"OpenClaw agent ({model}) for {self.demo_id}",
                )
                start = time.monotonic()
                result = agent.execute(task)
                duration_ms = round((time.monotonic() - start) * 1000)
                return {
                    "input": task,
                    "output": str(result),
                    "model": model,
                    "duration_ms": duration_ms,
                }
            except Exception as exc:
                self.logger.warning("OpenClaw execution failed (%s). Using simulated data.", exc)

        # Simulated fallback
        import random
        import hashlib

        seed = int(hashlib.sha256(f"{model}:{task}".encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        return {
            "input": task,
            "output": f"[Simulated {model} response to: {task[:80]}...]",
            "model": model,
            "duration_ms": rng.randint(200, 3000),
        }

    def upload_trace(self, input_text: str, output_text: str, metadata: dict) -> str:
        """Upload a trace via the SDK and return the trace ID."""
        if self.client is None:
            self.logger.debug("SDK not available; skipping trace upload.")
            return ""
        trace_data = {
            "input": [{"role": "user", "content": input_text}],
            "output": output_text,
            "metadata": metadata,
        }
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps(trace_data) + "\n")
            result = self.client.traces.upload(path)
            return result.trace_ids[0] if result and result.trace_ids else ""
        except Exception as exc:
            self.logger.debug("Trace upload failed: %s", exc)
            return ""
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def evaluate_trace(self, trace_id: str, judge_id: str) -> dict[str, Any] | None:
        """Run a real SDK trace evaluation with polling.

        Creates a trace evaluation via the SDK and polls for results using
        the shared ``poll_evaluation_results`` helper.  Returns a dict with
        ``score``, ``passed``, and ``reasoning`` keys when successful, or
        ``None`` in offline mode or on failure.
        """
        if not self.client or not trace_id or not judge_id:
            self.logger.debug("SDK not available or missing IDs; skipping trace evaluation.")
            return None
        try:
            evaluation = self.client.trace_evaluations.create(
                trace_id=trace_id,
                judge_id=judge_id,
            )
            if not evaluation:
                return None
            # Use shared polling helper
            results = poll_evaluation_results(self.client, evaluation.id, max_attempts=15)
            if results:
                r = results[0]
                return {"score": r.score, "passed": r.passed, "reasoning": r.reasoning}
        except Exception as exc:
            self.logger.debug("SDK evaluation failed: %s", exc)
        return None

    def create_judge(self, name: str, evaluation_goal: str) -> str:
        """Create a LayerLens judge via the SDK and return the judge ID.

        Returns an empty string in offline mode or on failure.
        """
        if not self.client:
            return ""
        try:
            model_id = get_default_model_id(self.client)
            try:
                judge = self.client.judges.create(name=name, evaluation_goal=evaluation_goal, model_id=model_id)
                return judge.id if judge else ""
            except Exception as create_exc:
                # Handle 409 Conflict by reusing existing judge
                if "already exists" in str(create_exc) or "409" in str(create_exc):
                    self.logger.info("Judge '%s' already exists, reusing.", name)
                    resp = self.client.judges.get_many()
                    if resp and resp.judges:
                        for j in resp.judges:
                            if j.name == name:
                                return j.id
                raise
        except Exception as exc:
            self.logger.debug("Judge creation failed: %s", exc)
            return ""

    def execute(self, argv: list[str] | None = None) -> None:
        """Parse CLI args and run the demo."""
        parser = self.build_parser()
        self.args = parser.parse_args(argv)

        # Configure logging
        level = logging.DEBUG if self.args.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            force=True,
        )

        logger.info("Starting %s demo (demo_id=%s)", self.demo_name, self.demo_id)
        self._init_sdk()

        try:
            result = asyncio.run(self.run())
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            sys.exit(130)
        except Exception:
            logger.exception("Demo failed")
            sys.exit(1)

        if self.args.json:
            _print_json(result)
        else:
            logger.info("Demo completed successfully")
