"use client";

import React, { useState } from "react";
import { CopilotChat } from "@copilotkit/react-ui";
import {
  useCoAgent,
  useCopilotAction,
  useCopilotChat,
} from "@copilotkit/react-core";
import { TextMessage, Role } from "@copilotkit/runtime-client-gql";

import {
  EvaluationCard,
  JudgeVerdictCard,
  MetricCard,
  TraceCard,
  type Severity,
  type Verdict,
} from "@layerlens/copilotkit-cards";
import { ThemeToggle } from "./theme-toggle";

/**
 * Sample harness page.
 *
 * Architecture mirrors CopilotKit's ``coagents-research-canvas`` reference:
 * the LangGraph agent owns the LLM and all backend tool implementations
 * (``list_recent_traces``, ``list_judges``, ``run_trace_evaluation``,
 * ``get_evaluation_result``); the only frontend-defined action is the
 * HITL ``confirm_judge`` picker, which uses the documented
 * ``useCopilotAction({ available: "remote", renderAndWaitForResponse })``
 * pattern.
 *
 * Progressive cards (recent traces, available judges, running
 * evaluations, completed EvaluationCards) render from agent state via
 * ``useCoAgentStateRender`` -- the canonical CopilotKit hook for
 * state-driven UI. The agent's tools return ``Command(update={...})``
 * so each tool call appends/replaces the corresponding state field,
 * which streams to the browser as ``STATE_SNAPSHOT`` events.
 */

type JudgeCandidate = {
  id: string;
  name: string;
  goal: string;
};

type TraceRecord = {
  id: string;
  filename: string;
  created_at: string;
  model?: string;
  duration_ms?: number;
  tokens?: number;
  evaluations_count?: number;
};
type JudgeRecord = { id: string; name: string; goal: string };
type EvaluationRecord = {
  evaluation_id: string;
  trace_id: string;
  judge_id: string;
  status: string;
};
type ResultRecord = {
  evaluation_id: string;
  status: string;
  trace_id?: string;
  judge_id?: string;
  passed?: boolean;
  score?: number;
  reasoning?: string;
};

type EvaluatorState = {
  traces?: TraceRecord[];
  judges?: JudgeRecord[];
  evaluations?: EvaluationRecord[];
  results?: ResultRecord[];
};

/** Section heading for canvas rows. */
function CanvasSection({
  title,
  count,
  children,
}: {
  title: string;
  count?: number;
  children: React.ReactNode;
}) {
  return (
    <section className="lg:col-span-2">
      <h2 className="mb-3 flex items-baseline gap-2 text-sm font-medium uppercase tracking-wide text-muted-foreground">
        <span>{title}</span>
        {count !== undefined ? (
          <span className="rounded-full bg-muted px-2 py-0.5 text-[10px] font-medium text-muted-foreground tabular-nums">
            {count}
          </span>
        ) : null}
      </h2>
      {children}
    </section>
  );
}

/** LayerLens dashboard base URL for "Trace Explorer" / "Agent Graph"
 *  links inside the SDK ``TraceCard``. Set via
 *  ``NEXT_PUBLIC_LAYERLENS_DASHBOARD_URL``. When unset (e.g., the
 *  Trace Explorer / Agent Graph features aren't deployed yet), the
 *  links are simply not rendered — better than 404ing. */
const DASHBOARD_BASE_URL =
  process.env.NEXT_PUBLIC_LAYERLENS_DASHBOARD_URL || undefined;

/** Base URL of the FastAPI backend ("/evaluations/{id}" etc.).
 *  Defaults to the dev port the sample boots on; override via
 *  ``NEXT_PUBLIC_EVALUATOR_BACKEND_URL`` when running the backend
 *  elsewhere. */
const BACKEND_BASE_URL =
  process.env.NEXT_PUBLIC_EVALUATOR_BACKEND_URL || "http://127.0.0.1:8123";

function TracesCard({
  traces,
  activeIds,
}: {
  traces: TraceRecord[];
  activeIds: Set<string>;
}) {
  // Only render full ``TraceCard``s for the traces currently being
  // evaluated (cap at 4 so the canvas stays scannable). The remaining
  // traces are surfaced as a compact summary row — same data, no
  // whitespace cost.
  const featured = activeIds.size
    ? traces.filter((t) => activeIds.has(t.id)).slice(0, 4)
    : traces.slice(0, 4);
  const featuredIds = new Set(featured.map((t) => t.id));
  const others = traces.filter((t) => !featuredIds.has(t.id));
  return (
    <CanvasSection
      title={activeIds.size ? "Traces under evaluation" : "Recent traces"}
      count={traces.length}
    >
      {featured.length > 0 ? (
        <div
          className="grid gap-3 lg:grid-cols-2"
          data-testid="state-traces-card"
        >
          {featured.map((t) => (
            <TraceCard
              key={t.id}
              traceId={t.id}
              framework={t.model ? "openai" : "stratix"}
              agentName={t.filename || "trace"}
              // ``status`` intentionally omitted — the
              // ``traces.get_many`` LayerLens API response doesn't
              // expose a per-trace lifecycle status today, so we don't
              // fabricate one here. When the API exposes it, pass it
              // through and the card will render the pill.
              duration_ms={t.duration_ms ?? 0}
              tokenCount={t.tokens ?? 0}
              costUsd={0}
              eventCount={t.evaluations_count ?? 0}
              agentCount={1}
              timestamp={t.created_at}
              tags={t.model ? [t.model] : undefined}
              dashboardBaseUrl={DASHBOARD_BASE_URL}
            />
          ))}
        </div>
      ) : null}
      {others.length > 0 ? (
        <details className="mt-3 rounded-md border bg-muted/30 p-2 text-xs text-muted-foreground">
          <summary className="cursor-pointer select-none">
            + {others.length} more in this project
          </summary>
          <ul className="mt-2 grid gap-1">
            {others.slice(0, 30).map((t) => (
              <li
                key={t.id}
                className="flex items-baseline justify-between gap-3"
              >
                <code className="truncate font-mono">{t.id}</code>
                <span className="truncate">{t.filename}</span>
              </li>
            ))}
          </ul>
        </details>
      ) : null}
    </CanvasSection>
  );
}

function JudgesCard({ judges }: { judges: JudgeRecord[] }) {
  return (
    <CanvasSection title="Available judges" count={judges.length}>
      <ul className="grid gap-2" data-testid="state-judges-card">
        {judges.slice(0, 8).map((j) => (
          <li
            key={j.id}
            className="rounded-lg border bg-card p-3 transition-shadow duration-200 hover:shadow-md"
          >
            <div className="flex items-baseline justify-between gap-3">
              <span className="font-medium">{j.name}</span>
              <code className="font-mono text-xs text-muted-foreground">
                {j.id}
              </code>
            </div>
            {j.goal ? (
              <p className="mt-1 line-clamp-2 text-xs text-muted-foreground">
                {j.goal}
              </p>
            ) : null}
          </li>
        ))}
        {judges.length > 8 ? (
          <li className="text-xs text-muted-foreground">
            …and {judges.length - 8} more.
          </li>
        ) : null}
      </ul>
    </CanvasSection>
  );
}

function MetricStrip({
  traces,
  judges,
  evaluations,
  results,
}: {
  traces: TraceRecord[];
  judges: JudgeRecord[];
  evaluations: EvaluationRecord[];
  results: ResultRecord[];
}) {
  const passed = results.filter((r) => r.passed === true).length;
  const failed = results.filter((r) => r.passed === false).length;
  const completed = results.length;
  const pending = Math.max(0, evaluations.length - completed);
  const passRate =
    completed > 0 ? Math.round((passed / completed) * 100) : 0;
  const avgScore =
    completed > 0
      ? results.reduce((acc, r) => acc + (r.score ?? 0), 0) / completed
      : 0;
  return (
    <div
      className="lg:col-span-2 sticky top-0 z-10 -mx-6 -mt-6 grid gap-3 border-b bg-background/90 px-6 py-4 backdrop-blur sm:grid-cols-2 md:grid-cols-4"
      data-testid="state-metric-strip"
    >
      <MetricCard
        label="Recent traces"
        value={traces.length}
        description="In project"
      />
      <MetricCard
        label="Available judges"
        value={judges.length}
        description="Evaluation criteria"
      />
      <MetricCard
        label="Pass rate"
        value={completed > 0 ? passRate : "—"}
        unit={completed > 0 ? "%" : undefined}
        trend={
          completed === 0
            ? "flat"
            : passRate >= 80
            ? "up"
            : passRate >= 50
            ? "flat"
            : "down"
        }
        description={`${passed} pass · ${failed} fail · ${pending} pending`}
      />
      <MetricCard
        label="Avg score"
        value={completed > 0 ? Math.round(avgScore * 100) : "—"}
        unit={completed > 0 ? "%" : undefined}
        description="Across completed evaluations"
      />
    </div>
  );
}

/**
 * HITL judge picker. Lives as its own component so it can subscribe to
 * ``useCoAgent`` directly — when ``state.judges`` updates while the
 * picker is mounted (waiting for HITL response), this component
 * re-renders with the fresh list. The previous implementation read
 * ``judges`` from a closure in ``renderAndWaitForResponse``, which
 * didn't re-run when state updated mid-pause.
 */
function JudgePicker({
  respond,
  status,
}: {
  respond: ((value: { id: string; name: string }) => void) | undefined;
  status: ToolCallStatus;
}) {
  // Subscribe to live agent state inside the picker so it re-renders
  // when ``state.judges`` populates after the LLM calls ``list_judges``.
  // The previous prop-based approach only saw the closure value at the
  // moment ``useCopilotAction`` invoked its render function; state
  // updates after that didn't reach the mounted picker.
  const { state } = useCoAgent<EvaluatorState>({ name: "evaluator" });
  const candidates = (state?.judges ?? []) as JudgeCandidate[];

  if (status === "complete") {
    return (
      <div
        className="my-2 inline-flex items-center gap-2 rounded-full border border-green-200 bg-green-50 px-3 py-1 text-sm font-medium text-green-700 dark:border-green-900/60 dark:bg-green-900/20 dark:text-green-300"
        data-testid="judge-picker-complete"
      >
        <span aria-hidden>✓</span>
        Judge selected
      </div>
    );
  }

  if (!candidates.length) {
    return (
      <div className="my-2 rounded-lg border bg-muted/40 p-3 text-sm text-muted-foreground">
        Loading judges…
      </div>
    );
  }

  return (
    <div
      className="my-2 rounded-lg border bg-card p-4 text-sm"
      data-testid="judge-picker"
    >
      <p className="mb-3 flex items-baseline justify-between font-medium">
        <span>Pick a judge for this evaluation</span>
        <span className="text-xs font-normal text-muted-foreground">
          {candidates.length} available — scroll to see all
        </span>
      </p>
      <ul className="grid max-h-[55vh] gap-2 overflow-y-auto pr-1">
        {candidates.map((judge) => {
          const ready = Boolean(judge?.id && judge?.name && respond);
          return (
            <li
              key={judge.id}
              className="rounded-md border bg-background p-3 transition-shadow duration-200 hover:shadow-md"
            >
              <div className="flex items-baseline justify-between gap-3">
                <span className="font-semibold">{judge.name}</span>
                <code className="font-mono text-xs text-muted-foreground">
                  {judge.id}
                </code>
              </div>
              {judge.goal ? (
                <p className="mt-1 text-xs text-muted-foreground">{judge.goal}</p>
              ) : null}
              <button
                type="button"
                className="mt-2 inline-flex h-7 items-center rounded-md bg-[#6766FC] px-3 text-xs font-medium text-white shadow-sm transition hover:bg-[#5755e0] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-[#6766FC] disabled:cursor-default disabled:opacity-50"
                data-testid={`judge-card-select-${judge.id}`}
                onClick={() =>
                  ready && respond?.({ id: judge.id, name: judge.name })
                }
                disabled={!ready}
              >
                Select {judge.name}
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

type ToolCallStatus = "inProgress" | "executing" | "complete";

function severityFromScore(score: number): Severity {
  if (score >= 0.8) return "low";
  if (score >= 0.5) return "medium";
  if (score >= 0.25) return "high";
  return "critical";
}

function ResultsSection({
  results,
  evaluations,
  judges,
}: {
  results: ResultRecord[];
  evaluations: EvaluationRecord[];
  judges: JudgeRecord[];
}) {
  const completed = results.length;
  if (completed === 0) {
    if (evaluations.length === 0) return null;
    // We have evaluations in flight but no completed results yet.
    return (
      <CanvasSection title="Evaluation results" count={0}>
        <p className="rounded-md border border-dashed bg-muted/30 p-4 text-sm text-muted-foreground">
          {evaluations.length} evaluation{evaluations.length === 1 ? "" : "s"}{" "}
          running — verdicts will appear here as they complete.
        </p>
      </CanvasSection>
    );
  }

  const judgeName = (judgeId: string | undefined) =>
    judges.find((j) => j.id === judgeId)?.name ?? judgeId ?? "Judge";
  const passed = results.filter((r) => r.passed === true).length;
  const failed = results.filter((r) => r.passed === false).length;
  const totalScore = results.reduce((acc, r) => acc + (r.score ?? 0), 0);
  const avgScore = completed > 0 ? totalScore / completed : 0;
  const passRate =
    completed > 0 ? Math.round((passed / completed) * 100) : 0;
  const summaryJudgeName =
    new Set(results.map((r) => r.judge_id)).size === 1
      ? judgeName(results[0]?.judge_id)
      : "Mixed judges";

  return (
    <>
      <CanvasSection title="Run summary">
        <EvaluationCard
          evaluationId={`run-${results[0]?.evaluation_id ?? "summary"}`}
          name={summaryJudgeName}
          passRate={passRate}
          totalCases={completed}
          passedCases={passed}
          failedCases={failed}
          errorCases={0}
          scores={[{ label: "Average score", value: avgScore }]}
          status={evaluations.length > completed ? "running" : "completed"}
        />
      </CanvasSection>
      <CanvasSection title="Verdicts" count={completed}>
        <div className="grid gap-3 lg:grid-cols-2" data-testid="state-results-cards">
          {results.map((r) => {
            const score = typeof r.score === "number" ? r.score : 0;
            const verdict: Verdict =
              r.passed === true ? "pass" : r.passed === false ? "fail" : "error";
            return (
              <JudgeVerdictCard
                key={r.evaluation_id}
                judgeName={judgeName(r.judge_id)}
                verdict={verdict}
                score={Math.max(0, Math.min(1, score))}
                reasoning={r.reasoning ?? ""}
                evidence={[]}
                severity={severityFromScore(score)}
              />
            );
          })}
        </div>
      </CanvasSection>
    </>
  );
}

export default function Page() {
  const { appendMessage, isLoading } = useCopilotChat();

  // ---- Live agent state (research-canvas pattern) -------------------------
  // ``useCoAgent`` exposes the agent's state as a live React value. It
  // updates every time the backend emits a STATE_SNAPSHOT (after each
  // ``Command(update={...})`` from a tool). We render the four progress
  // cards as regular JSX in a side panel — this matches CopilotKit's
  // canonical ``coagents-research-canvas`` pattern, where the canvas
  // (state-driven UI) lives next to the chat panel rather than inline.
  const { state } = useCoAgent<EvaluatorState>({
    name: "evaluator",
    initialState: {
      traces: [],
      judges: [],
      evaluations: [],
      results: [],
    },
  });
  const traces = state?.traces ?? [];
  const judges = state?.judges ?? [];
  const evaluations = state?.evaluations ?? [];
  const agentResults = state?.results ?? [];

  // ---- Out-of-band polling for pending evaluations -------------------------
  // The agent kicks off evaluations and returns immediately — real
  // LayerLens evaluations take 30+ seconds to complete, far longer
  // than a chat turn should block. The frontend polls the backend's
  // ``/evaluations/{id}`` endpoint every few seconds for any
  // evaluation that hasn't yet shown up in ``state.results``. Verdicts
  // land in ``polledResults`` and are merged with ``agentResults`` for
  // display; the user sees pending evaluations flip to passed/failed
  // as each one finishes.
  const [polledResults, setPolledResults] = useState<ResultRecord[]>([]);

  React.useEffect(() => {
    const knownIds = new Set([
      ...agentResults.map((r) => r.evaluation_id),
      ...polledResults.map((r) => r.evaluation_id),
    ]);
    const pendingIds = evaluations
      .map((e) => e.evaluation_id)
      .filter((id) => id && !knownIds.has(id));
    if (pendingIds.length === 0) return;

    let cancelled = false;
    const poll = async () => {
      const updates = await Promise.all(
        pendingIds.map(async (id) => {
          try {
            const resp = await fetch(
              `${BACKEND_BASE_URL}/evaluations/${id}`,
              { cache: "no-store" },
            );
            if (!resp.ok) return null;
            return (await resp.json()) as ResultRecord;
          } catch {
            return null;
          }
        }),
      );
      if (cancelled) return;
      const completed = updates.filter(
        (u): u is ResultRecord =>
          !!u && u.status === "success" && typeof u.score === "number",
      );
      if (completed.length > 0) {
        setPolledResults((prev) => {
          const seen = new Set(prev.map((r) => r.evaluation_id));
          return [...prev, ...completed.filter((c) => !seen.has(c.evaluation_id))];
        });
      }
    };
    poll(); // immediate kick once on change
    const interval = setInterval(poll, 5000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [evaluations, agentResults, polledResults]);

  // Merge: agent's view first, then anything the frontend polled in.
  const results: ResultRecord[] = React.useMemo(() => {
    const seen = new Set(agentResults.map((r) => r.evaluation_id));
    return [
      ...agentResults,
      ...polledResults.filter((r) => !seen.has(r.evaluation_id)),
    ];
  }, [agentResults, polledResults]);

  const hasAny =
    traces.length > 0 ||
    judges.length > 0 ||
    evaluations.length > 0 ||
    results.length > 0;

  // ---- Frontend HITL tool: confirm_judge -----------------------------------
  // The user-facing widget. The LLM emits this tool call once judges have
  // been fetched into ``state.judges``; the picker reads candidates
  // straight from agent state (via ``useCoAgent`` above) so they don't
  // have to be replayed through tool arguments. Streaming a 38-judge
  // array through ``parameters`` was producing JSON.parse failures
  // ("Unterminated string in JSON at position N") when the LLM truncated
  // mid-stream. State-driven candidates avoid that pathology entirely.
  useCopilotAction({
    name: "confirm_judge",
    description:
      "Ask the user to choose which judge to apply. Take no arguments — the picker reads available judges from agent state. Returns the chosen judge's id and name.",
    parameters: [],
    renderAndWaitForResponse: ({ respond, status }) => (
      <JudgePicker
        respond={respond as ((value: { id: string; name: string }) => void) | undefined}
        status={status as ToolCallStatus}
      />
    ),
  });

  const sendEvaluate = () => {
    void appendMessage(
      new TextMessage({
        role: Role.User,
        content: "Please evaluate my recent traces.",
      }),
    );
  };

  return (
    <main
      className="flex h-screen flex-col bg-background text-foreground"
      data-testid="harness-root"
    >
      <header className="flex items-center justify-between border-b bg-background/80 px-6 py-3 text-sm font-medium backdrop-blur">
        <span>
          LayerLens Stratix Evaluator{" "}
          <span className="text-xs font-normal text-muted-foreground">
            (agent: <code className="font-mono">evaluator</code>)
          </span>
        </span>
        <span className="flex items-center gap-3">
          <ThemeToggle />
          <button
            type="button"
            data-testid="evaluate-start"
            className="inline-flex h-8 items-center rounded-md bg-[#6766FC] px-3 text-xs font-medium text-white shadow-sm transition hover:bg-[#5755e0] disabled:opacity-50"
            onClick={sendEvaluate}
            disabled={isLoading}
            title="Send the initial evaluation request"
          >
            {isLoading ? "Running…" : "Evaluate my traces"}
          </button>
        </span>
      </header>
      <div
        className="flex flex-1 min-h-0 overflow-hidden"
        data-testid="harness-body"
      >
        {/* Canvas — wide main panel, state-driven cards. Mirrors the
            ``ResearchCanvas`` layout in coagents-research-canvas. */}
        <section
          className="flex-1 min-w-0 overflow-y-auto p-6"
          data-testid="harness-canvas"
        >
          {hasAny ? (
            <div className="grid gap-6 lg:grid-cols-2">
              <MetricStrip
                traces={traces}
                judges={judges}
                evaluations={evaluations}
                results={results}
              />
              {traces.length > 0 ? (
                <TracesCard
                  traces={traces}
                  activeIds={
                    new Set([
                      ...evaluations.map((e) => e.trace_id),
                      ...results
                        .map((r) => r.trace_id)
                        .filter((id): id is string => Boolean(id)),
                    ])
                  }
                />
              ) : null}
              {judges.length > 0 ? <JudgesCard judges={judges} /> : null}
              <ResultsSection
                results={results}
                evaluations={evaluations}
                judges={judges}
              />
            </div>
          ) : (
            <div
              className="rounded-lg border border-dashed bg-muted/40 p-12 text-center text-sm text-muted-foreground"
              data-testid="canvas-empty"
            >
              <p className="text-base text-foreground">
                Welcome to the LayerLens Stratix evaluator.
              </p>
              <p className="mt-2">
                Click <span className="font-medium text-foreground">Evaluate
                my traces</span> in the header (or ask in chat) to populate
                this canvas with your real LayerLens Stratix traces, judges,
                and evaluation results.
              </p>
            </div>
          )}
        </section>
        {/* Chat — narrow sidebar on the right, fixed width. The
            ``confirm_judge`` HITL widget renders inline here.
            ``min-h-0 + overflow-hidden`` are required so CopilotChat's
            internal message scroller activates instead of growing past
            the viewport. */}
        <aside
          className="flex h-full w-[420px] min-h-0 shrink-0 flex-col overflow-hidden border-l bg-background"
          data-testid="harness-chat"
        >
          <CopilotChat
            labels={{
              title: "Evaluator",
              initial:
                "Hi — I can evaluate your LayerLens Stratix traces. Click Evaluate my traces, or ask in your own words.",
            }}
            instructions=""
          />
        </aside>
      </div>
    </main>
  );
}
