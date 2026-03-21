from __future__ import annotations

# ANSI color codes
_CYAN = "\033[38;2;54;191;250m"  # #36BFFA
_GRAY = "\033[90m"
_RESET = "\033[0m"

_ART = r"""
  ____ _____ ____      _  _____ _____  __
 / ___|_   _|  _ \    / \|_   _|_ _\ \/ /
 \___ \ | | | |_) |  / _ \ | |  | | \  /
  ___) || | |  _ <  / ___ \| |  | | /  \
 |____/ |_| |_| \_\/_/   \_\_| |___/_/\_\
"""


def banner(version: str) -> str:
    """Return the colored CLI banner with version line."""
    lines = _ART.rstrip("\n")
    colored_art = f"{_CYAN}{lines}{_RESET}"
    version_line = f"{_GRAY}  v{version} — layerlens.ai{_RESET}"
    return f"{colored_art}\n{version_line}\n"
