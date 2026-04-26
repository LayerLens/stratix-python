"use client";

import { useEffect, useState } from "react";

type Theme = "light" | "dark" | "system";

const STORAGE_KEY = "layerlens-evaluator:theme";

function applyTheme(theme: Theme): void {
  const root = document.documentElement;
  const prefersDark =
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-color-scheme: dark)").matches;
  const effective: "light" | "dark" =
    theme === "system" ? (prefersDark ? "dark" : "light") : theme;
  root.classList.toggle("dark", effective === "dark");
  root.style.colorScheme = effective;
}

/**
 * Three-state theme toggle (light / dark / system) persisted to
 * localStorage. Exposed as a client component so the rest of the page
 * can stay a server component.
 */
export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>("light");
  const [mounted, setMounted] = useState(false);

  // Hydrate from localStorage on mount.
  useEffect(() => {
    const stored =
      (typeof window !== "undefined" &&
        (window.localStorage.getItem(STORAGE_KEY) as Theme | null)) ||
      "light";
    setTheme(stored);
    applyTheme(stored);
    setMounted(true);
  }, []);

  // React to OS theme changes when in "system" mode.
  useEffect(() => {
    if (theme !== "system") return;
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => applyTheme("system");
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, [theme]);

  const select = (next: Theme) => {
    setTheme(next);
    if (typeof window !== "undefined") {
      window.localStorage.setItem(STORAGE_KEY, next);
    }
    applyTheme(next);
  };

  // Avoid hydration mismatch — render an inert placeholder until we
  // know the persisted preference.
  if (!mounted) {
    return (
      <span className="inline-flex h-7 w-[114px] rounded-md border border-slate-300 bg-slate-100 dark:border-slate-700 dark:bg-slate-800" />
    );
  }

  const button = (label: string, value: Theme, icon: string) => {
    const active = theme === value;
    return (
      <button
        key={value}
        type="button"
        onClick={() => select(value)}
        aria-pressed={active}
        title={`Theme: ${label}`}
        className={[
          "flex h-7 w-9 items-center justify-center text-xs transition",
          active
            ? "bg-white text-slate-900 shadow-sm dark:bg-slate-700 dark:text-slate-50"
            : "text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200",
        ].join(" ")}
      >
        <span aria-hidden>{icon}</span>
        <span className="sr-only">{label}</span>
      </button>
    );
  };

  return (
    <div
      role="group"
      aria-label="Theme"
      className="inline-flex items-center rounded-md border border-slate-300 bg-slate-100 p-0.5 dark:border-slate-700 dark:bg-slate-800"
    >
      {button("Light", "light", "☀")}
      {button("System", "system", "◑")}
      {button("Dark", "dark", "☾")}
    </div>
  );
}

export default ThemeToggle;
