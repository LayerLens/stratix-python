"""ADP-W2 Family-B record-real-once generators (one module per adapter).

Each module exposes ``generate_<adapter>_single(client)`` and
``generate_<adapter>_multi(client)`` which run a REAL instrumented framework/
provider under the capture seam of ``samples/data/_generate_fixtures.py`` and
write a sealed real-trace fixture to ``samples/data/traces/industry/<stem>.jsonl``.
Framework dependencies are imported function-locally so this package imports in
any venv.
"""
