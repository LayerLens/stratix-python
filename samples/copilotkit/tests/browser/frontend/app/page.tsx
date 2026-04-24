"use client";

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
            {candidates.map((judge) => (
              <li key={judge.id} className="judge-card">
                <div className="judge-card-head">
                  <span className="judge-card-name">{judge.name}</span>
                  <span className="judge-card-id" title="Judge id">
                    {judge.id}
                  </span>
                </div>
                {judge.goal ? (
                  <p className="judge-card-goal">{judge.goal}</p>
                ) : null}
                <button
                  type="button"
                  className="judge-card-select"
                  data-testid={`judge-card-select-${judge.id}`}
                  onClick={() => respond?.({ id: judge.id, name: judge.name })}
                  disabled={!respond}
                >
                  Select {judge.name}
                </button>
              </li>
            ))}
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

  return (
    <main className="harness-shell" data-testid="harness-root">
      <header className="harness-header">
        <span>
          LayerLens Evaluator (agent: <code>evaluator</code>)
        </span>
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
      </header>
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
