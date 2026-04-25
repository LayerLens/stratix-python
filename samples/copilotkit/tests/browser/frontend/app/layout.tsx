import "@copilotkit/react-ui/styles.css";
import "./globals.css";

import type { Metadata } from "next";
import type { ReactNode } from "react";

import { CopilotKit } from "@copilotkit/react-core";

export const metadata: Metadata = {
  title: "CopilotKit Evaluator Browser Harness",
  description:
    "End-to-end browser test harness for the LayerLens CopilotKit evaluator agent.",
};

export default function RootLayout({
  children,
}: {
  children: ReactNode;
}) {
  return (
    <html lang="en" className="dark">
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
        >
          {children}
        </CopilotKit>
      </body>
    </html>
  );
}
