"""Customer-run render sweep (ADP W1/W2 samples-audit, Objective 2).

Uploads every Family-B industry fixture the way a paying customer does
(``upload_recorded_trace``, LayerLens-key-ONLY) to a running atlas and asserts
each one renders correctly end-to-end: the server-computed agent graph (a
multi-agent DAG where the trace is genuinely multi-agent; an honest empty-state
for providers/ingestion/non-agentic), the count-aware Agent column, the
Framework and Status columns, and a populated waterfall.

The expectation is derived from each fixture's OWN events by a faithful Python
port of the server's ``services.InferAgentGraph`` node-identity extraction
(``_render_oracle``), validated against the live-proven graph-contract oracle —
so the sweep asserts what the SERVER returns, never the SDK ``_identity`` view.
"""
