import { defineConfig, devices } from "@playwright/test";
import * as path from "path";

/**
 * Playwright config for the CopilotKit evaluator browser harness.
 *
 * Process topology:
 *   - ``globalSetup.ts`` spawns ``uvicorn`` against ``backend/server.py``
 *     on port 8123 and waits for ``/healthz`` to return 200.
 *   - ``webServer`` below boots ``next dev`` on port 3000 and waits for
 *     the root route to return 200.
 *
 * ``globalTeardown.ts`` kills the FastAPI process when the suite ends.
 */
export default defineConfig({
  testDir: path.join(__dirname, "tests"),
  timeout: 60_000,
  expect: { timeout: 20_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],

  globalSetup: path.join(__dirname, "tests", "globalSetup.ts"),
  globalTeardown: path.join(__dirname, "tests", "globalTeardown.ts"),

  use: {
    baseURL: "http://127.0.0.1:3000",
    trace: "retain-on-failure",
    video: "retain-on-failure",
    screenshot: "only-on-failure",
  },

  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],

  webServer: {
    command: "npm run dev",
    cwd: __dirname,
    url: "http://127.0.0.1:3000",
    reuseExistingServer: false,
    timeout: 120_000,
    stdout: "pipe",
    stderr: "pipe",
  },
});
