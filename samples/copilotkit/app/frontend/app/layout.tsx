import "@copilotkit/react-ui/styles.css";
import "./globals.css";

import type { Metadata } from "next";
import type { ReactNode } from "react";

import { CopilotKit } from "@copilotkit/react-core";

export const metadata: Metadata = {
  title: "LayerLens Evaluator — CopilotKit Sample",
  description:
    "CopilotKit + LangGraph + LayerLens sample: evaluate agent traces against judges.",
};

export default function RootLayout({
  children,
}: {
  children: ReactNode;
}) {
  // Default to light to match CopilotKit's official samples
  // (``coagents-research-canvas``, ``travel``, ``with-shadcn-ui``).
  // The ``ThemeToggle`` client component hydrates the user's
  // persisted preference from localStorage on mount.
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        {/*
          ``agent="evaluator"`` matches the key used when wiring the
          LangGraphHttpAgent in ``app/api/copilotkit/route.ts``:
            new CopilotRuntime({ agents: { evaluator: ... } })
        */}
        <CopilotKit
          runtimeUrl="/api/copilotkit"
          agent="evaluator"
          showDevConsole={false}
          enableInspector={false}
        >
          {children}
        </CopilotKit>
      </body>
    </html>
  );
}
