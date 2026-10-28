"""Compliance check script verifying public API exports in `zero_zoo`."""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from pathlib import Path


def check_compliance(
    plan_path: Path = Path("SEMANTIC_PLAN.md"),
    todo_path: Path = Path("TODO.md"),
) -> int:
    """Verify public `zero_zoo` exports against a semantic plan if present.

    Args:
        plan_path: Path to `SEMANTIC_PLAN.md`.
        todo_path: Path to `TODO.md`.

    Returns:
        `0` if compliant or if `plan_path` does not exist; `1` if missing exports exist.
    """
    if not plan_path.exists():
        print("SEMANTIC_PLAN.md not found, skipping compliance check.")
        return 0

    content = plan_path.read_text(encoding="utf-8")
    init_text = Path("src/zero_zoo/__init__.py").read_text(encoding="utf-8")
    exported = set(re.findall(r'"([A-Za-z0-9_]+)"', init_text))
    required_symbols = re.findall(r"`([a-zA-Z_][a-zA-Z0-9_]*)`", content)

    missing = [sym for sym in required_symbols if sym not in exported]
    if todo_path.exists():
        _ = todo_path.read_text(encoding="utf-8")

    if missing:
        print(f"Missing exports: {missing}")
        return 1

    print("Compliance Check Passed!")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the compliance check script.

    Args:
        argv: Optional CLI arguments (`[plan_path, todo_path]`).

    Returns:
        Exit status code (`0` on success, `1` on failure).
    """
    args = list(argv) if argv is not None else []
    plan_path = Path(args[0]) if len(args) >= 1 else Path("SEMANTIC_PLAN.md")
    todo_path = Path(args[1]) if len(args) >= 2 else Path("TODO.md")
    return check_compliance(plan_path=plan_path, todo_path=todo_path)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or None))
