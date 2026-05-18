/**
 * CopilotKit runtime route handler.
 *
 * Bridges the browser-side CopilotKit client (``<CopilotKit runtimeUrl="...">``)
 * to the FastAPI + ag_ui_langgraph backend served by ``backend/server.py``.
 *
 * The backend exposes the evaluator graph at ``POST /evaluator`` (see
 * ``add_langgraph_fastapi_endpoint(app, agent=..., path="/evaluator")``),
 * so ``LangGraphHttpAgent.url`` must point at that full path -- NOT the
 * server root.
 *
 * ``ExperimentalEmptyAdapter`` is correct here because the LangGraph
 * backend owns all LLM calls; the CopilotKit runtime does not need its
 * own model adapter.
 */
import {
  CopilotRuntime,
  ExperimentalEmptyAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
import { LangGraphHttpAgent } from "@copilotkit/runtime/langgraph";
import { NextRequest } from "next/server";

// Runtime config -- run on Node, never cache.
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const BACKEND_URL =
  process.env.EVALUATOR_BACKEND_URL || "http://127.0.0.1:8123/evaluator";

const serviceAdapter = new ExperimentalEmptyAdapter();

const runtimeInstance = new CopilotRuntime({
  agents: {
    evaluator: new LangGraphHttpAgent({
      url: BACKEND_URL,
    }),
  },
});

export const POST = async (req: NextRequest) => {
  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime: runtimeInstance,
    serviceAdapter,
    endpoint: "/api/copilotkit",
  });
  return handleRequest(req);
};
