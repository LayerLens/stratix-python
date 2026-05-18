/**
 * Starts the FastAPI evaluator backend (``backend/server.py``) for the
 * duration of the Playwright suite.
 *
 * We do NOT run this via Playwright's ``webServer`` option because that
 * slot is already used for ``next dev``. Instead we spawn uvicorn
 * manually, poll ``/healthz``, and stash the PID in a global var that
 * ``globalTeardown.ts`` reads to kill the process cleanly.
 */
import { spawn, type ChildProcess } from "child_process";
import * as fs from "fs";
import * as path from "path";

// Tests live at app/frontend/tests/; backend/ is a sibling of frontend/.
const BACKEND_DIR = path.resolve(__dirname, "..", "..", "backend");
const PID_FILE = path.resolve(__dirname, ".backend.pid");

const HOST = "127.0.0.1";
const PORT = 8123;
const HEALTH_URL = `http://${HOST}:${PORT}/healthz`;
const READY_TIMEOUT_MS = 30_000;

function pickPythonCmd(): string {
  // Honor explicit override first.
  if (process.env.HARNESS_PYTHON) return process.env.HARNESS_PYTHON;
  // On Windows the launcher ``py`` is usually present; elsewhere
  // ``python3`` is the safer default.
  return process.platform === "win32" ? "python" : "python3";
}

async function waitForHealth(url: string, timeoutMs: number): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  let lastErr: unknown = null;
  while (Date.now() < deadline) {
    try {
      const resp = await fetch(url);
      if (resp.ok) return;
      lastErr = new Error(`status=${resp.status}`);
    } catch (err) {
      lastErr = err;
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error(
    `Backend did not become healthy at ${url} within ${timeoutMs}ms: ${String(
      lastErr,
    )}`,
  );
}

export default async function globalSetup(): Promise<void> {
  // Reuse an already-running backend if the test author started one
  // manually (useful when iterating on the backend in a separate shell).
  try {
    const resp = await fetch(HEALTH_URL);
    if (resp.ok) {
      // Do NOT record a PID; teardown should not kill a backend we did
      // not start.
      return;
    }
  } catch {
    // fall through -- start our own
  }

  const python = pickPythonCmd();
  const args = [
    "-m",
    "uvicorn",
    "server:app",
    "--host",
    HOST,
    "--port",
    String(PORT),
    "--log-level",
    "warning",
  ];

  const child: ChildProcess = spawn(python, args, {
    cwd: BACKEND_DIR,
    stdio: ["ignore", "inherit", "inherit"],
    env: {
      ...process.env,
      PYTHONUNBUFFERED: "1",
      LAYERLENS_STRATIX_API_KEY:
        process.env.LAYERLENS_STRATIX_API_KEY || "test-key",
    },
    // ``detached: false`` keeps the backend as a child of the Playwright
    // runner so it dies with the runner if teardown is skipped.
    detached: false,
    windowsHide: true,
  });

  if (child.pid === undefined) {
    throw new Error("Failed to spawn uvicorn process (no pid)");
  }

  fs.writeFileSync(PID_FILE, String(child.pid), "utf-8");

  child.on("exit", (code, signal) => {
    // Swallow; teardown handles intentional termination. Surface
    // unexpected early exits to stderr so the test logs are useful.
    if (code !== 0 && code !== null && signal === null) {
      // eslint-disable-next-line no-console
      console.error(
        `[globalSetup] FastAPI backend exited unexpectedly: code=${code}`,
      );
    }
  });

  await waitForHealth(HEALTH_URL, READY_TIMEOUT_MS);
}
