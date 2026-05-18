/**
 * Tears down the FastAPI backend started by ``globalSetup.ts``.
 *
 * Reads the PID out of ``.backend.pid``; if the file does not exist we
 * assume the backend was already running before the suite started and
 * leave it alone.
 */
import * as fs from "fs";
import * as path from "path";

const PID_FILE = path.resolve(__dirname, ".backend.pid");

export default async function globalTeardown(): Promise<void> {
  if (!fs.existsSync(PID_FILE)) return;
  const raw = fs.readFileSync(PID_FILE, "utf-8").trim();
  const pid = Number.parseInt(raw, 10);
  fs.unlinkSync(PID_FILE);

  if (!Number.isFinite(pid) || pid <= 0) return;

  try {
    if (process.platform === "win32") {
      // ``taskkill`` is the only reliable way to murder a detached
      // uvicorn process tree on Windows.
      const { spawnSync } = await import("child_process");
      spawnSync("taskkill", ["/pid", String(pid), "/T", "/F"], {
        stdio: "ignore",
        windowsHide: true,
      });
    } else {
      process.kill(pid, "SIGTERM");
      // Give uvicorn a beat to shut down; then hard-kill if still alive.
      await new Promise((r) => setTimeout(r, 1500));
      try {
        process.kill(pid, 0); // probe
        process.kill(pid, "SIGKILL");
      } catch {
        // already exited -- great
      }
    }
  } catch {
    // Best effort; no point failing the whole suite if the process is
    // already gone.
  }
}
