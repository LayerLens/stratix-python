"use client";

import { useState } from "react";
import { CopilotChat } from "@copilotkit/react-ui";
import {
  useCopilotChat,
  useLangGraphInterrupt,
} from "@copilotkit/react-core";
import { TextMessage, Role } from "@copilotkit/runtime-client-gql";

/**
 * Sample harness page.
 *
 * Wires ``useLangGraphInterrupt`` so that when the backend evaluator
 * hits its ``copilotkit_interrupt(...)`` HITL pause, the user can reply
 * through a dedicated prompt widget and the frontend sends the answer
 * back as ``forwardedProps.command.resume``. Without this hook, a plain
 * ``<CopilotChat>`` sends the user's reply as an ordinary new chat
 * message, the backend sees no resume command, and the graph stays
 * paused at the ``interrupt()`` boundary. That's the correct AG-UI
 * protocol behavior -- the frontend must explicitly signal a resume.
 *
 * ``copilotkit_interrupt`` on the server extracts the answer from
 * ``response[-1].content``, so ``resolve`` must be called with a list of
 * message-shaped objects whose last element has ``.content``.
 *
 * The "Start evaluation" button exists so automated tests (and humans
 * without an LLM-backed chat flow) can kick off the graph without
 * having to type into the CopilotChat textarea. It uses
 * ``useCopilotChat().appendMessage`` -- the same primitive the chat
 * widget uses internally -- so the runtime path is identical.
 */
export default function Page() {
  const { appendMessage, isLoading } = useCopilotChat();
  const [draft, setDraft] = useState("");

  useLangGraphInterrupt({
    render: ({ event, resolve }) => {
      // ``event.value`` is whatever was passed to the server-side
      // ``copilotkit_interrupt(message=...)``. That helper wraps the
      // payload as ``{__copilotkit_interrupt_value__, __copilotkit_messages__}``.
      const raw = (event as { value?: unknown }).value;
      let prompt: string;
      if (typeof raw === "string") {
        prompt = raw;
      } else if (raw && typeof raw === "object") {
        const obj = raw as { __copilotkit_interrupt_value__?: string };
        prompt = obj.__copilotkit_interrupt_value__ ?? JSON.stringify(raw);
      } else {
        prompt = "The agent is waiting for a reply.";
      }

      const submit = () => {
        const answer = draft.trim() || "ok";
        setDraft("");
        // ``copilotkit_interrupt`` does ``answer = response[-1].content``,
        // so ``resolve`` must receive a list whose last element has
        // ``.content``.
        resolve([{ role: "user", content: answer }]);
      };

      return (
        <div className="harness-interrupt" data-testid="harness-interrupt">
          <p
            className="harness-interrupt-prompt"
            data-testid="harness-interrupt-prompt"
          >
            {prompt}
          </p>
          <div className="harness-interrupt-row">
            <input
              data-testid="harness-interrupt-input"
              className="harness-interrupt-input"
              value={draft}
              placeholder="Type your answer (default: ok)"
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  submit();
                }
              }}
            />
            <button
              type="button"
              data-testid="harness-interrupt-submit"
              className="harness-interrupt-submit"
              onClick={submit}
            >
              Submit
            </button>
          </div>
        </div>
      );
    },
  });

  const sendEvaluate = () => {
    // ``appendMessage`` is the non-premium chat-send primitive in
    // @copilotkit/react-core 1.56.3. It accepts a TextMessage instance;
    // we use that here rather than ``useCopilotChatHeadless_c`` +
    // ``sendMessage`` because the headless hook is premium-gated
    // (requires ``publicApiKey``) and silently no-ops without one.
    void appendMessage(
      new TextMessage({
        role: Role.User,
        content: "evaluate my traces",
      })
    );
  };

  return (
    <main className="harness-shell" data-testid="harness-root">
      <header className="harness-header">
        CopilotKit Evaluator Harness (agent: <code>evaluator</code>)
        <button
          type="button"
          data-testid="harness-start"
          className="harness-start-button"
          onClick={sendEvaluate}
          disabled={isLoading}
        >
          {isLoading ? "Running..." : "Start evaluation"}
        </button>
      </header>
      <div className="harness-chat" data-testid="harness-chat">
        <CopilotChat
          labels={{
            title: "Evaluator",
            initial: "Ask me to evaluate your traces.",
          }}
          instructions=""
        />
      </div>
    </main>
  );
}
