"use client";

import { useState } from "react";
import { CopilotChat } from "@copilotkit/react-ui";
import { useCopilotAction, useCopilotChat } from "@copilotkit/react-core";
import { TextMessage, Role } from "@copilotkit/runtime-client-gql";

/**
 * Sample harness page.
 *
 * HITL is wired as a **frontend tool**: the backend's system prompt tells
 * the LLM to call ``confirm_judge`` when it has the judge candidates, and
 * this page defines that tool via ``useCopilotAction`` with
 * ``renderAndWaitForResponse``. When the LLM emits the tool call, the
 * widget below appears inline in the chat. The user picks a judge; the
 * component calls ``respond({id, name})`` which completes the tool; the
 * LLM receives the selection and continues the evaluation flow.
 *
 * This pattern is CopilotKit's current idiom for HITL (matches the
 * ``hitl_in_chat_agent.py`` showcase). It sidesteps the
 * ``ag-ui-langgraph`` + ``interrupt()`` pipeline entirely, which had two
 * protocol-level bugs (``ag-ui-protocol/ag-ui#1582`` and ``#1584``) and
 * required a private-API workaround subclass in an earlier revision of
 * this sample.
 */

type JudgeCandidate = {
  id: string;
  name: string;
  goal: string;
};

export default function Page() {
  const { appendMessage, isLoading } = useCopilotChat();

  // Frontend tool: ``confirm_judge``. The backend agent's system prompt
  // instructs the LLM to call this once it has the list of judges.
  // ``renderAndWaitForResponse`` pauses the tool call, renders the
  // widget in the chat, and waits for ``respond(value)`` to continue.
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
        // Post-selection, compact state. We don't have direct access to
        // the resolved value here, so show a generic confirmation.
        return (
          <div className="judge-picker judge-picker-complete" data-testid="judge-picker-complete">
            <span className="judge-picker-check" aria-hidden>
              ✓
            </span>
            Judge selected.
          </div>
        );
      }

      if (!candidates.length) {
        return (
          <div className="judge-picker judge-picker-empty">
            No judges were provided. Ask LayerLens to list your judges first.
          </div>
        );
      }

      return (
        <div className="judge-picker" data-testid="judge-picker">
          <p className="judge-picker-prompt">Pick a judge for this evaluation:</p>
          <ul className="judge-picker-list">
            {candidates.map((judge, index) => {
              // The LLM occasionally streams partial tool-call args before
              // the full payload lands (CopilotKit's renderAndWaitForResponse
              // re-renders progressively as JSON streams in). During that
              // window ``judge.id`` may be undefined for one tick, which
              // tripped React's "unique key" warning. Fall back to index
              // so the warning stays quiet and the row stays stable enough
              // for the user to click once the args finalise.
              const id = judge?.id ?? `pending-${index}`;
              const name = judge?.name ?? "Loading...";
              const goal = judge?.goal;
              const ready = Boolean(judge?.id && judge?.name && respond);
              return (
                <li key={id} className="judge-card">
                  <div className="judge-card-head">
                    <span className="judge-card-name">{name}</span>
                    {judge?.id ? (
                      <span className="judge-card-id" title="Judge id">
                        {judge.id}
                      </span>
                    ) : null}
                  </div>
                  {goal ? <p className="judge-card-goal">{goal}</p> : null}
                  <button
                    type="button"
                    className="judge-card-select"
                    data-testid={
                      judge?.id ? `judge-card-select-${judge.id}` : undefined
                    }
                    onClick={() =>
                      ready &&
                      respond?.({ id: judge!.id, name: judge!.name })
                    }
                    disabled={!ready}
                  >
                    {ready ? `Select ${name}` : "Loading..."}
                  </button>
                </li>
              );
            })}
          </ul>
        </div>
      );
    },
  });

  const sendEvaluate = () => {
    void appendMessage(
      new TextMessage({
        role: Role.User,
        content: "Please evaluate my recent traces.",
      })
    );
  };

  // Diagnostic panel: probes the runtime, the textarea state, and any
  // error banners, then renders a JSON dump on the page so it can be
  // read without copy/paste from DevTools. Click "Run diagnostic" to
  // populate it.
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

      const ta = document.querySelector(
        'textarea[placeholder*="message" i]'
      ) as HTMLTextAreaElement | null;
      const send = (
        Array.from(document.querySelectorAll("button")) as HTMLButtonElement[]
      ).find((b) => b.getAttribute("aria-label") === "Send");

      const banners = (
        Array.from(
          document.querySelectorAll(
            '[role=alert],[class*="banner" i],[class*="error" i]'
          )
        ) as HTMLElement[]
      )
        .map((el) => ({
          cls: el.className,
          txt: (el.textContent || "").trim().slice(0, 200),
        }))
        .filter((b) => b.txt);

      const taBox = ta ? ta.getBoundingClientRect() : null;
      const sendBox = send ? send.getBoundingClientRect() : null;

      // Try sending a real message via the hook to see what the runtime
      // returns. We intercept fetch to capture the body and response.
      const captured: Array<{ url: string; status: number; body: string }> = [];
      const origFetch = window.fetch;
      window.fetch = async (...args) => {
        const resp = await origFetch.apply(window, args as Parameters<typeof fetch>);
        const url =
          typeof args[0] === "string"
            ? args[0]
            : (args[0] as Request | URL).toString();
        if (url.includes("/api/copilotkit")) {
          const clone = resp.clone();
          const body = await clone
            .text()
            .catch(() => "<unreadable>");
          captured.push({
            url,
            status: clone.status,
            body: body.slice(0, 600),
          });
        }
        return resp;
      };
      try {
        await appendMessage(
          new TextMessage({
            role: Role.User,
            content: "Diagnostic: hello.",
          })
        );
      } catch (err) {
        captured.push({
          url: "<exception>",
          status: 0,
          body: String(err).slice(0, 400),
        });
      }
      // Give it a moment to fire
      await new Promise((r) => setTimeout(r, 1500));
      window.fetch = origFetch;

      const report = {
        runtime_info: info,
        textarea: ta
          ? {
              disabled: ta.disabled,
              readonly: ta.readOnly,
              pointer_events: getComputedStyle(ta).pointerEvents,
              display: getComputedStyle(ta).display,
              visibility: getComputedStyle(ta).visibility,
              size: taBox && {
                w: Math.round(taBox.width),
                h: Math.round(taBox.height),
              },
            }
          : "NO TEXTAREA",
        send_button: send
          ? {
              disabled: send.disabled,
              testid: send.getAttribute("data-test-id"),
              size: sendBox && {
                w: Math.round(sendBox.width),
                h: Math.round(sendBox.height),
              },
            }
          : "NO SEND",
        banners,
        appendMessage_state: {
          isLoading,
        },
        post_calls_during_appendMessage: captured,
      };
      setDiag(JSON.stringify(report, null, 2));
    } catch (err) {
      setDiag("ERROR: " + String(err));
    }
  };

  return (
    <main className="harness-shell" data-testid="harness-root">
      <header className="harness-header">
        <span>
          LayerLens Evaluator (agent: <code>evaluator</code>)
        </span>
        <span style={{ display: "flex", gap: 8 }}>
          <button
            type="button"
            data-testid="harness-diag"
            className="harness-start-button"
            onClick={runDiagnostic}
            title="Run a diagnostic and show results below"
          >
            Run diagnostic
          </button>
          <button
            type="button"
            data-testid="harness-start"
            className="harness-start-button"
            onClick={sendEvaluate}
            disabled={isLoading}
            title="Send the initial evaluation request"
          >
            {isLoading ? "Running..." : "Evaluate my traces"}
          </button>
        </span>
      </header>
      {diag ? (
        <pre
          data-testid="harness-diag-output"
          style={{
            background: "#0b0d12",
            color: "#b3b9c5",
            padding: "12px 16px",
            margin: 0,
            fontSize: 12,
            lineHeight: 1.4,
            maxHeight: "40vh",
            overflow: "auto",
            borderBottom: "1px solid #1d222d",
            whiteSpace: "pre-wrap",
            wordBreak: "break-word",
          }}
        >
          {diag}
        </pre>
      ) : null}
      <div className="harness-chat" data-testid="harness-chat">
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
