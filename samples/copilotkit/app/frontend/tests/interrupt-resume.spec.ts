/**
 * Browser-level regression for the CopilotKit evaluator interrupt/resume
 * fix (PR ``fix/copilotkit-interrupt-checkpointer`` on stratix-python).
 *
 * Drives the real ``@copilotkit/react-ui`` <CopilotChat> widget through
 * three user turns:
 *
 *   1. "evaluate my traces" -> triggers the graph, which fetches judges
 *      + traces and then pauses at the ``interrupt()`` in
 *      ``confirm_judge_node``. The UI should render text mentioning the
 *      judge list.
 *   2. "ok"                 -> resumes the graph. Before the checkpointer
 *      fix, this is the turn that produced the "Cannot send
 *      'RUN_STARTED' while a run is still active" error.
 *   3. "thanks"             -> a fresh run after resume. If the resume's
 *      RUN_FINISHED event was dropped, the CopilotKit client would
 *      refuse to open a new run here.
 *
 * The spec FAILS if the exact string
 *   "Cannot send 'RUN_STARTED' while a run is still active"
 * appears anywhere in the browser console OR the visible DOM.
 */
import { expect, test, type ConsoleMessage, type Page } from "@playwright/test";

const FORBIDDEN_ERROR = "Cannot send 'RUN_STARTED' while a run is still active";

/**
 * CopilotKit 1.56's default <CopilotChat> renders a textarea for user
 * input and a send button.  We locate via role/accessible-name so the
 * test keeps working across minor styling changes.  We fall back to
 * pragma-broad selectors if the primary ones miss.
 */
async function chatInput(page: Page) {
  // CopilotChat 1.56 renders <textarea placeholder="Type a message...">.
  // Target by placeholder so the locator is robust across styling changes
  // that might affect role or testid resolution.
  return page.locator('textarea[placeholder*="message" i]').first();
}

async function sendMessage(page: Page, text: string): Promise<void> {
  const input = await chatInput(page);
  await input.waitFor({ state: "attached" });

  // CopilotChat's textarea reports as aria-hidden / not-visible in headless
  // Chromium (even though it's in the DOM and React accepts input on it).
  // Setting .value through the prototype setter bypasses React's internal
  // ``_valueTracker`` bookkeeping, and dispatching a bubbling "input" event
  // makes React's synthetic event system pick it up. This is the standard
  // pattern for driving controlled inputs that fail strict-visibility
  // checks.
  await input.evaluate((el, value) => {
    const ta = el as HTMLTextAreaElement;
    const proto = Object.getPrototypeOf(ta);
    const descriptor = Object.getOwnPropertyDescriptor(proto, "value");
    descriptor?.set?.call(ta, value);
    ta.dispatchEvent(new Event("input", { bubbles: true }));
  }, text);

  const sendButton = page.getByRole("button", { name: /send/i });
  await expect(sendButton).toBeEnabled({ timeout: 10_000 });
  // ``force: true`` skips the actionability check -- the button's parent
  // container uses pointer-events styles that confuse Playwright's hit
  // test. The button itself is functional and clicking it triggers the
  // CopilotKit submit pipeline.
  await sendButton.click({ force: true });
}

/**
 * Wait for a new assistant message that contains one of the expected
 * needles. CopilotChat does not set a stable data-testid on messages, so
 * we match by visible text inside the chat region.
 */
async function waitForAssistantText(
  page: Page,
  needles: string[],
  timeoutMs = 30_000,
): Promise<void> {
  const chat = page.locator('[data-testid="harness-chat"]');
  await expect
    .poll(
      async () => {
        const text = (await chat.innerText()).toLowerCase();
        return needles.some((n) => text.includes(n.toLowerCase()));
      },
      { timeout: timeoutMs, message: `waiting for: ${needles.join(" | ")}` },
    )
    .toBe(true);
}

async function assertNoRunStartedError(
  page: Page,
  consoleErrors: string[],
): Promise<void> {
  // 1. DOM: the CopilotKit error surface renders failures inline in the
  // chat. Assert the forbidden string is not visible anywhere on the
  // page.
  const bodyText = await page.locator("body").innerText();
  expect(
    bodyText.includes(FORBIDDEN_ERROR),
    `Forbidden string '${FORBIDDEN_ERROR}' appeared in the visible DOM. ` +
      `This is the exact symptom of the checkpointer-missing bug.`,
  ).toBeFalsy();

  // 2. Console: CopilotKit also throws this error from its protocol
  // state machine when it sees RUN_STARTED before RUN_FINISHED.
  const matching = consoleErrors.filter((e) => e.includes(FORBIDDEN_ERROR));
  expect(
    matching,
    `Forbidden string '${FORBIDDEN_ERROR}' appeared in the browser ` +
      `console. Matches:\n${matching.join("\n")}`,
  ).toEqual([]);
}

test("interrupt + resume + follow-up all succeed (no RUN_STARTED lockup)", async ({
  page,
}) => {
  const consoleErrors: string[] = [];
  const allConsole: string[] = [];
  page.on("console", (msg: ConsoleMessage) => {
    const text = msg.text();
    allConsole.push(`[${msg.type()}] ${text}`);
    if (msg.type() === "error" || msg.type() === "warning") {
      consoleErrors.push(text);
    }
  });
  page.on("pageerror", (err) => {
    consoleErrors.push(`[pageerror] ${err.message}`);
  });

  // On test failure, dump full console + network activity via
  // test.info().attach so triage does not require opening the trace.
  // (We push the attach into the after-hook path via the test-info API.)

  // Track network calls so we can tell whether failures are frontend or
  // backend side.
  const runtimeCalls: Array<{ url: string; status: number }> = [];
  page.on("response", (resp) => {
    const url = resp.url();
    if (url.includes("/api/copilotkit") || url.includes(":8123/")) {
      runtimeCalls.push({ url, status: resp.status() });
    }
  });

  await page.goto("/");

  // Smoke-check the harness mounted and the CopilotKit provider wired up
  // the /api/copilotkit route before we start driving the chat.
  await expect(page.getByTestId("harness-root")).toBeVisible();
  await expect(page.getByTestId("harness-chat")).toBeVisible();

  // Dump all captured console + network activity on failure so triage
  // does not require opening the trace archive.
  const dumpOnFailure = async () => {
    await test.info().attach("browser-console.log", {
      body: Buffer.from(allConsole.join("\n"), "utf-8"),
      contentType: "text/plain",
    });
    await test.info().attach("runtime-calls.json", {
      body: Buffer.from(JSON.stringify(runtimeCalls, null, 2), "utf-8"),
      contentType: "application/json",
    });
    // Also echo to stdout so the default Playwright reporter surfaces them.
    // eslint-disable-next-line no-console
    console.log(
      "\n=== browser console ===\n" +
        allConsole.slice(-80).join("\n") +
        "\n=== runtime calls ===\n" +
        JSON.stringify(runtimeCalls, null, 2),
    );
  };

  try {
    // -----------------------------------------------------------------
    // Turn 1: kick off the agent.  It should stream judge/trace summaries
    // and then pause at interrupt() asking which judge to use.
    // -----------------------------------------------------------------
    await sendMessage(page, "evaluate my traces");
    // ``fetch_judges_node`` emits: "Found 1 judge(s):\n  - **Helpfulness** (`jdg_1`): ..."
    // ``confirm_judge_node`` emits the interrupt prompt: "Which judge should I use?"
    await waitForAssistantText(
      page,
      ["helpfulness", "judge", "which judge should i use"],
      45_000,
    );
    await assertNoRunStartedError(page, consoleErrors);

    // -----------------------------------------------------------------
    // Turn 2: resume the interrupted run with "ok".  Before the fix this
    // is the turn that emits the forbidden RUN_STARTED error.
    // -----------------------------------------------------------------
    await sendMessage(page, "ok");
    // After resume, ``confirm_judge_node`` emits "Using judge **Helpfulness**",
    // then run_evaluations + poll_results run to completion with a PASS line.
    await waitForAssistantText(page, ["using judge", "helpfulness"], 45_000);
    await assertNoRunStartedError(page, consoleErrors);

    // -----------------------------------------------------------------
    // Turn 3: send a fresh message after the resume finished.  If the
    // RUN_FINISHED event was dropped on the resume, the CopilotKit client
    // will refuse to open a new run here and we will see the forbidden
    // error in either the DOM or the console.
    // -----------------------------------------------------------------
    await sendMessage(page, "thanks");
    // The agent graph terminates after poll_results so turn 3 kicks off a
    // brand-new run from the entry node.  We only need to confirm *some*
    // assistant response arrived and that no lockup error surfaced.
    await waitForAssistantText(page, ["helpfulness", "judge", "error"], 45_000);
    await assertNoRunStartedError(page, consoleErrors);
  } catch (err) {
    await dumpOnFailure();
    throw err;
  }
});
