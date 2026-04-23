"use client";

import { CopilotChat } from "@copilotkit/react-ui";

/**
 * Single-page harness used by the Playwright test.
 *
 * We render the full ``CopilotChat`` component (not the floating popup),
 * so the chat surface is always visible and always has a stable input
 * locator for Playwright to drive.
 */
export default function Page() {
  return (
    <main className="harness-shell" data-testid="harness-root">
      <header className="harness-header">
        CopilotKit Evaluator Harness (agent: <code>evaluator</code>)
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
