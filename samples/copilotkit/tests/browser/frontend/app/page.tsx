"use client";

import React, { useState } from "react";
import { CopilotChat } from "@copilotkit/react-ui";
import { useCopilotAction, useCopilotChat } from "@copilotkit/react-core";
import { TextMessage, Role } from "@copilotkit/runtime-client-gql";

// Import a polished, production-grade card from the SDK reference set
// (samples/copilotkit/components/). Our backend-tool render callbacks
// below show how the same useCopilotAction pattern can either compose
// these existing cards or render lightweight inline cards using the
// matching Tailwind tokens.
import { EvaluationCard } from "@layerlens/copilotkit-cards/EvaluationCard";

/**
 * Sample harness page.
 *
 * HITL is wired as a **frontend tool** -- the LLM calls ``confirm_judge``
 * and the user picks an option from a card rendered inline in the chat.
 * Backend tool calls (``list_judges``, ``list_recent_traces``,
 * ``run_trace_evaluation``, ``get_evaluation_result``) are also surfaced
 * with their own per-tool cards via ``useCopilotAction({ available:
 * "remote", render: ... })``. The render callback receives the streaming
 * args and the final result; CopilotKit calls it on every stream tick.
 *
 * The visual style here matches CopilotKit's own showcase samples:
 * Tailwind 4, dark mode, rounded cards, status badges.
 */

type JudgeCandidate = {
  id: string;
  name: string;
  goal: string;
};

type ToolCallStatus = "inProgress" | "executing" | "complete";

/** Normalise the ``result`` arg from ``useCopilotAction``. It can arrive
 *  as a parsed object, a JSON string, or undefined depending on which
 *  point in the stream the renderer fires. */
function parseToolResult<T = unknown>(result: unknown): T | null {
  if (result == null) return null;
  if (typeof result === "string") {
    try {
      return JSON.parse(result) as T;
    } catch {
      return null;
    }
  }
  return result as T;
}

/** Status pill: pulsing dot while running, check on done. */
function ToolStatus({ status }: { status: ToolCallStatus }) {
  if (status === "complete") {
    return (
      <span
        className="inline-flex items-center rounded-full bg-emerald-900/40 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-emerald-300"
        aria-label="Done"
      >
        ✓ Done
      </span>
    );
  }
  return (
    <span
      className="inline-flex items-center gap-1 rounded-full bg-blue-900/40 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-blue-300"
      aria-label="Running"
    >
      <span className="size-1.5 animate-pulse rounded-full bg-blue-400" />
      Running
    </span>
  );
}

/** Generic per-tool card chrome -- header with icon + title + status,
 *  followed by tool-specific body content. */
function ToolCard({
  icon,
  title,
  status,
  testid,
  children,
}: {
  icon: string;
  title: string;
  status: ToolCallStatus;
  testid: string;
  children: React.ReactNode;
}) {
  return (
    <div
      className="my-2 rounded-lg border border-slate-700 bg-slate-900/50 p-3 text-sm"
      data-testid={testid}
    >
      <div className="mb-2 flex items-center gap-2">
        <span className="text-base" aria-hidden>
          {icon}
        </span>
        <span className="flex-1 font-medium text-slate-200">{title}</span>
        <ToolStatus status={status} />
      </div>
      <div className="text-slate-300">{children}</div>
    </div>
  );
}

export default function Page() {
  const { appendMessage, isLoading } = useCopilotChat();

  // ---- Backend-tool render: list_recent_traces -----------------------------
  useCopilotAction({
    name: "list_recent_traces",
    available: "remote",
    description: "List the most recent LayerLens traces.",
    parameters: [{ name: "limit", type: "number", required: false }],
    render: ({ status, args, result }) => {
      const traces =
        parseToolResult<Array<{ id: string; filename: string; created_at: string }>>(
          result,
        ) ?? [];
      const limit = (args as { limit?: number } | undefined)?.limit;
      return (
        <ToolCard
          icon="📂"
          title="Recent traces"
          status={status as ToolCallStatus}
          testid="tool-list-recent-traces"
        >
          {status === "complete" ? (
            traces.length > 0 ? (
              <ul className="grid gap-1.5">
                {traces.map((t, i) => (
                  <li
                    key={t.id ?? `t-${i}`}
                    className="flex items-baseline justify-between gap-3 rounded border border-slate-800 bg-slate-950/40 px-2.5 py-1.5"
                  >
                    <code className="text-xs text-slate-400">{t.id}</code>
                    <span className="truncate text-slate-300">{t.filename}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-slate-500">No traces found.</p>
            )
          ) : (
            <p className="text-slate-400">
              Fetching {limit ?? "recent"} traces…
            </p>
          )}
        </ToolCard>
      );
    },
  });

  // ---- Backend-tool render: list_judges ------------------------------------
  useCopilotAction({
    name: "list_judges",
    available: "remote",
    description: "List available LayerLens judges.",
    parameters: [],
    render: ({ status, result }) => {
      const judges =
        parseToolResult<Array<{ id: string; name: string; goal: string }>>(
          result,
        ) ?? [];
      return (
        <ToolCard
          icon="⚖️"
          title="Available judges"
          status={status as ToolCallStatus}
          testid="tool-list-judges"
        >
          {status === "complete" ? (
            judges.length > 0 ? (
              <ul className="grid gap-1.5">
                {judges.map((j, i) => (
                  <li
                    key={j.id ?? `j-${i}`}
                    className="rounded border border-slate-800 bg-slate-950/40 px-2.5 py-1.5"
                  >
                    <div className="flex items-baseline justify-between gap-3">
                      <span className="font-medium text-slate-200">{j.name}</span>
                      <code className="text-xs text-slate-500">{j.id}</code>
                    </div>
                    {j.goal ? (
                      <p className="mt-0.5 text-xs text-slate-400">{j.goal}</p>
                    ) : null}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-slate-500">No judges found.</p>
            )
          ) : (
            <p className="text-slate-400">Loading judges…</p>
          )}
        </ToolCard>
      );
    },
  });

  // ---- Backend-tool render: run_trace_evaluation ---------------------------
  useCopilotAction({
    name: "run_trace_evaluation",
    available: "remote",
    description: "Start a LayerLens evaluation.",
    parameters: [
      { name: "trace_id", type: "string", required: true },
      { name: "judge_id", type: "string", required: true },
    ],
    render: ({ status, args, result }) => {
      const a = (args as { trace_id?: string; judge_id?: string } | undefined) ?? {};
      const r = parseToolResult<{ evaluation_id?: string; status?: string }>(result);
      return (
        <ToolCard
          icon="🧪"
          title="Running evaluation"
          status={status as ToolCallStatus}
          testid="tool-run-trace-evaluation"
        >
          <p>
            trace <code className="text-slate-400">{a.trace_id ?? "…"}</code>{" "}
            against judge{" "}
            <code className="text-slate-400">{a.judge_id ?? "…"}</code>
            {r?.evaluation_id ? (
              <>
                {" "}
                — id{" "}
                <code className="text-slate-400">{r.evaluation_id}</code> (
                {r.status})
              </>
            ) : null}
          </p>
        </ToolCard>
      );
    },
  });

  // ---- Backend-tool render: get_evaluation_result --------------------------
  // Uses the polished ``EvaluationCard`` from
  // ``samples/copilotkit/components/`` so the final result is rendered
  // with the production-grade SDK card. The streaming/pending state
  // falls back to a lightweight inline card to match the others.
  useCopilotAction({
    name: "get_evaluation_result",
    available: "remote",
    description: "Get the result of a previously-started LayerLens evaluation.",
    parameters: [{ name: "evaluation_id", type: "string", required: true }],
    render: ({ status, args, result }) => {
      const a = (args as { evaluation_id?: string } | undefined) ?? {};
      const r = parseToolResult<{
        status?: string;
        passed?: boolean;
        score?: number;
        reasoning?: string;
      }>(result);

      if (status === "complete" && r?.status === "success" && typeof r.score === "number") {
        const passed = r.passed === true;
        const passRate = passed ? 100 : 0;
        return (
          <div data-testid="tool-get-evaluation-result">
            <EvaluationCard
              evaluationId={a.evaluation_id ?? "—"}
              name="Evaluation"
              passRate={passRate}
              totalCases={1}
              passedCases={passed ? 1 : 0}
              failedCases={passed ? 0 : 1}
              errorCases={0}
              scores={[
                {
                  label: "score",
                  value: Math.max(0, Math.min(1, r.score)),
                },
              ]}
              status="completed"
            />
            {r.reasoning ? (
              <p className="mt-1 text-xs text-slate-400">
                <span className="font-medium text-slate-300">Reasoning:</span>{" "}
                {r.reasoning}
              </p>
            ) : null}
          </div>
        );
      }

      return (
        <ToolCard
          icon="📊"
          title="Evaluation result"
          status={status as ToolCallStatus}
          testid="tool-get-evaluation-result"
        >
          <p className="text-slate-400">
            Polling{" "}
            <code className="text-slate-300">{a.evaluation_id ?? "…"}</code> (
            {r?.status ?? "pending"})
          </p>
        </ToolCard>
      );
    },
  });

  // ---- Frontend HITL tool: confirm_judge -----------------------------------
  // The user-facing widget. The LLM emits this tool call once it has the
  // candidate list; the render runs continuously while args stream in,
  // then the user clicks Select and ``respond({id, name})`` resolves the
  // tool call and feeds the choice back into the LLM's context.
  useCopilotAction({
    name: "confirm_judge",
    description:
      "Ask the user to choose which judge (evaluation criteria) to apply to their traces. Pass the full list of candidate judges as `candidates`. The user will pick one via an inline widget in the chat. Returns the chosen judge's id and name.",
    parameters: [
      {
        name: "candidates",
        type: "object[]",
        required: true,
        description: "The full list of judges to choose from.",
        attributes: [
          { name: "id", type: "string", required: true },
          { name: "name", type: "string", required: true },
          { name: "goal", type: "string", required: true },
        ],
      },
    ],
    renderAndWaitForResponse: ({ args, respond, status }) => {
      const candidates: JudgeCandidate[] =
        (args?.candidates as JudgeCandidate[] | undefined) ?? [];

      if (status === "complete") {
        return (
          <div
            className="my-2 inline-flex items-center gap-2 rounded-lg border border-emerald-700/60 bg-emerald-900/30 px-3 py-1.5 text-sm text-emerald-200"
            data-testid="judge-picker-complete"
          >
            <span aria-hidden>✓</span>
            Judge selected.
          </div>
        );
      }

      if (!candidates.length) {
        return (
          <div className="my-2 rounded-lg border border-rose-800/60 bg-rose-950/30 p-3 text-sm text-rose-300">
            No judges were provided. Ask LayerLens to list your judges first.
          </div>
        );
      }

      return (
        <div
          className="my-2 rounded-lg border border-slate-700 bg-slate-900/60 p-4 text-sm"
          data-testid="judge-picker"
        >
          <p className="mb-3 font-medium text-slate-200">
            Pick a judge for this evaluation:
          </p>
          <ul className="grid gap-2">
            {candidates.map((judge, index) => {
              const id = judge?.id ?? `pending-${index}`;
              const name = judge?.name ?? "Loading…";
              const goal = judge?.goal;
              const ready = Boolean(judge?.id && judge?.name && respond);
              return (
                <li
                  key={id}
                  className="rounded-md border border-slate-700 bg-slate-950/60 p-3"
                >
                  <div className="flex items-baseline justify-between gap-3">
                    <span className="font-semibold text-slate-100">{name}</span>
                    {judge?.id ? (
                      <code className="text-xs text-slate-500">{judge.id}</code>
                    ) : null}
                  </div>
                  {goal ? (
                    <p className="mt-1 text-xs text-slate-400">{goal}</p>
                  ) : null}
                  <button
                    type="button"
                    className="mt-2 rounded-md bg-blue-700 px-3 py-1 text-xs font-medium text-white hover:bg-blue-600 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-blue-400 disabled:cursor-default disabled:opacity-50"
                    data-testid={
                      judge?.id ? `judge-card-select-${judge.id}` : undefined
                    }
                    onClick={() =>
                      ready &&
                      respond?.({ id: judge!.id, name: judge!.name })
                    }
                    disabled={!ready}
                  >
                    {ready ? `Select ${name}` : "Loading…"}
                  </button>
                </li>
              );
            })}
          </ul>
        </div>
      );
    },
  });

  // ---- Diagnostic panel (kept from earlier session for support/debug) -----
  const [diag, setDiag] = useState<string>("");
  const runDiagnostic = async () => {
    setDiag("running...");
    try {
      const info = await fetch("/api/copilotkit", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ method: "info", params: {}, id: 1 }),
      })
        .then((r) => r.json())
        .catch((e) => ({ error: String(e) }));
      setDiag(JSON.stringify({ runtime_info: info }, null, 2));
    } catch (err) {
      setDiag("ERROR: " + String(err));
    }
  };

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
      className="flex h-screen flex-col bg-slate-950 text-slate-100"
      data-testid="harness-root"
    >
      <header className="flex items-center justify-between border-b border-slate-800 px-5 py-3 text-sm font-semibold">
        <span>
          LayerLens Evaluator{" "}
          <span className="text-xs font-normal text-slate-400">
            (agent: <code>evaluator</code>)
          </span>
        </span>
        <span className="flex gap-2">
          <button
            type="button"
            data-testid="harness-diag"
            onClick={runDiagnostic}
            className="rounded-md border border-slate-700 bg-slate-800/60 px-3 py-1 text-xs hover:bg-slate-800"
            title="Run a diagnostic and show results below"
          >
            Run diagnostic
          </button>
          <button
            type="button"
            data-testid="harness-start"
            className="rounded-md bg-blue-700 px-3 py-1 text-xs font-medium hover:bg-blue-600 disabled:opacity-50"
            onClick={sendEvaluate}
            disabled={isLoading}
            title="Send the initial evaluation request"
          >
            {isLoading ? "Running…" : "Evaluate my traces"}
          </button>
        </span>
      </header>
      {diag ? (
        <pre
          data-testid="harness-diag-output"
          className="max-h-[40vh] overflow-auto whitespace-pre-wrap break-words border-b border-slate-800 bg-slate-900/40 px-4 py-3 font-mono text-xs leading-snug text-slate-300"
        >
          {diag}
        </pre>
      ) : null}
      <div className="flex-1 min-h-0" data-testid="harness-chat">
        <CopilotChat
          labels={{
            title: "Evaluator",
            initial:
              "Hi — I can evaluate your LayerLens traces. Click Evaluate my traces, or ask in your own words.",
          }}
          instructions=""
        />
      </div>
    </main>
  );
}
