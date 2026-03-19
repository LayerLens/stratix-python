"""Allow running the simulator as a module.

Usage:
    python -m layerlens.instrument.simulators generate --source openai --count 10
    python -m layerlens.instrument.simulators list-sources
    python -m layerlens.instrument.simulators list-scenarios
    python -m layerlens.instrument.simulators validate --source generic_otel
"""

from .cli import main

main()
